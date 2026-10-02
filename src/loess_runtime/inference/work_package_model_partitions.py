"""Commit model Partition rasters and own package-local Fusion coverage."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import rasterio  # type: ignore[import-untyped]
from affine import Affine  # type: ignore[import-untyped]
from numpy.typing import NDArray

from labeling_tool.shared.contracts.run_spec import sha256_file
from labeling_tool.shared.state.artifact_repository import ArtifactRepository
from loess_runtime.geometry.authoritative_raster import (
    apply_range_mask_to_core,
    core_mask_tags,
)
from loess_runtime.inference.incremental_fusion import FusionAccumulator
from loess_runtime.inference.partition_mosaic import write_partition_rasters
from loess_runtime.inference.work_package_storage import WorkPackageStorageBudget
from loess_runtime.system.artifact_publication import publish_artifact
from loess_runtime.system.runtime_errors import WorkPackageRuntimeError
from loess_runtime.system.runtime_metrics import directory_size


def _artifact_record_is_valid(
    artifact: Mapping[str, Any] | None,
    *,
    expected_count: int,
    expected_dtype: str,
    expected_width: int,
    expected_height: int,
    expected_crs: str,
) -> bool:
    if artifact is None or str(artifact.get("status")) != "ready":
        return False
    path = Path(str(artifact.get("path") or ""))
    try:
        if (
            not path.is_file()
            or path.stat().st_size != int(artifact["byte_count"])
            or sha256_file(path) != str(artifact["sha256"])
        ):
            return False
        with rasterio.open(path) as source:
            return (
                source.count == int(expected_count)
                and source.width == int(expected_width)
                and source.height == int(expected_height)
                and all(dtype == str(expected_dtype) for dtype in source.dtypes)
                and str(source.crs or "") == str(expected_crs)
            )
    except (OSError, ValueError, KeyError, rasterio.errors.RasterioError):
        return False


class WorkPackageModelPartitions:
    """Own model output commits and matched Fusion accumulator/coverage state."""

    def __init__(
        self,
        *,
        run_id: str,
        run_dir: Path,
        package_root: Path,
        fusion_id: str,
        partitions: Sequence[Mapping[str, Any]],
        transform: Affine,
        crs: str,
        range_geometry: Any,
        profile: Mapping[str, Any] | None,
        artifacts: ArtifactRepository,
        lease_guard: Callable[[], None] | None,
    ) -> None:
        self._run_id = run_id
        self._run_dir = run_dir
        self._partitions = tuple(partitions)
        self._transform = transform
        self._crs = crs
        self._range_geometry = range_geometry
        self._profile = profile
        self._artifacts = artifacts
        self._lease_guard = lease_guard
        self._coverage_masks: dict[str, NDArray[Any]] = {}
        self._accumulators: dict[str, FusionAccumulator] = {}
        if profile:
            for partition in self._partitions:
                partition_id = str(partition["partition_id"])
                halo = partition["halo_window"]
                shape = (14, halo["y1"] - halo["y0"], halo["x1"] - halo["x0"])
                self._accumulators[partition_id] = FusionAccumulator(
                    package_root / "fusion" / fusion_id / partition_id,
                    profile,
                    shape,
                )

    def model_outputs_reusable(self, stream_id: str, model_id: str) -> bool:
        """Validate committed outputs; restore all Fusion coverage or none."""

        artifacts = self._artifacts.artifacts_for_stream(
            self._run_id, stream_id, status=None
        )
        by_key = {
            (str(item.get("unit_id") or ""), str(item.get("kind") or "")): item
            for item in artifacts
        }
        for partition in self._partitions:
            partition_id = str(partition["partition_id"])
            core = partition["core_window"]
            width = int(core["x1"]) - int(core["x0"])
            height = int(core["y1"]) - int(core["y0"])
            if not _artifact_record_is_valid(
                by_key.get((partition_id, "core_mask")),
                expected_count=1,
                expected_dtype="int16",
                expected_width=width,
                expected_height=height,
                expected_crs=self._crs,
            ):
                return False
            if not _artifact_record_is_valid(
                by_key.get((partition_id, "core_confidence")),
                expected_count=1,
                expected_dtype="float32",
                expected_width=width,
                expected_height=height,
                expected_crs=self._crs,
            ):
                return False
            probability = by_key.get((partition_id, "partition_probability"))
            if probability is not None and str(probability.get("status")) == "ready":
                halo = partition["halo_window"]
                if not _artifact_record_is_valid(
                    probability,
                    expected_count=14,
                    expected_dtype="uint16",
                    expected_width=int(halo["x1"]) - int(halo["x0"]),
                    expected_height=int(halo["y1"]) - int(halo["y0"]),
                    expected_crs=self._crs,
                ):
                    return False
            elif probability is None or str(probability.get("status")) != "cleaned":
                return False
            accumulator = self._accumulators.get(partition_id)
            if (
                accumulator is not None
                and model_id not in accumulator.completed_model_ids()
            ):
                return False
        if not self._profile:
            return True

        # An accumulator need not encode coverage, so recover it only from a
        # verified committed probability raster. Do not publish partial state.
        ready_probabilities = self._artifacts.artifacts_for_stream(
            self._run_id,
            stream_id,
            kind="partition_probability",
            status="ready",
        )
        by_partition = {
            str(item.get("unit_id") or ""): item for item in ready_probabilities
        }
        restored: dict[str, NDArray[Any]] = {}
        try:
            for partition in self._partitions:
                partition_id = str(partition["partition_id"])
                artifact = by_partition.get(partition_id)
                if artifact is None:
                    return False
                with rasterio.open(Path(str(artifact["path"]))) as source:
                    coverage = np.any(source.read() > 0, axis=0)
                previous = self._coverage_masks.get(partition_id)
                if previous is not None and not np.array_equal(previous, coverage):
                    raise WorkPackageRuntimeError(
                        f"model coverage differs inside Partition: {partition_id}"
                    )
                restored[partition_id] = coverage
        except (OSError, ValueError, KeyError, rasterio.errors.RasterioError):
            return False
        self._coverage_masks.update(restored)
        return True

    def commit_model_partition(
        self,
        model_id: str,
        stream_id: str,
        partition: Mapping[str, Any],
        arrays: Mapping[str, NDArray[Any]],
        storage_budget: WorkPackageStorageBudget,
    ) -> None:
        """Synchronously commit one model Partition on the caller thread."""

        partition_id = str(partition["partition_id"])
        arrays, range_report = apply_range_mask_to_core(
            arrays,
            partition,
            global_transform=self._transform,
            range_geometry=self._range_geometry,
        )
        probability_path = (
            self._run_dir
            / "tmp"
            / "probability_parts"
            / model_id
            / f"{partition_id}.tif"
        )
        raster_root = self._run_dir / "models" / model_id / "raster_parts"
        previous_probability_bytes = (
            probability_path.stat().st_size if probability_path.is_file() else 0
        )
        probability_write_bytes = int(np.asarray(arrays["halo_probabilities"]).size * 2)
        permanent_write_bytes = int(
            np.asarray(arrays["core_mask"]).nbytes
            + np.asarray(arrays["core_confidence"]).nbytes
        )
        raster_write_overhead_bytes = 3 * 64 * 1024
        reservation = storage_budget.reserve_write(
            f"partition_rasters:{stream_id}:{partition_id}",
            probability_write_bytes
            + permanent_write_bytes
            + raster_write_overhead_bytes,
            managed_growth_bytes=max(
                0,
                probability_write_bytes + 64 * 1024 - previous_probability_bytes,
            ),
        )
        try:
            paths = write_partition_rasters(
                arrays,
                partition,
                global_transform=self._transform,
                crs=self._crs,
                output_probability=probability_path,
                output_mask=raster_root / f"{partition_id}_mask.tif",
                output_confidence=raster_root / f"{partition_id}_confidence.tif",
                core_mask_tags=core_mask_tags(
                    {"authority": "partition_core_argmax_v1", **range_report}
                ),
            )
        finally:
            current_probability_bytes = (
                probability_path.stat().st_size if probability_path.is_file() else 0
            )
            reservation.settle(current_probability_bytes - previous_probability_bytes)
        for kind, key in (
            ("core_mask", "mask"),
            ("core_confidence", "confidence"),
            ("partition_probability", "probability"),
        ):
            if self._lease_guard is not None:
                self._lease_guard()
            publish_artifact(
                self._artifacts,
                self._run_id,
                path=Path(paths[key]),
                kind=kind,
                stream_id=stream_id,
                unit_id=partition_id,
            )
            if kind in {"core_mask", "core_confidence"}:
                storage_budget.mark_permanent_ready(stream_id, partition_id, kind)
        if self._profile:
            coverage = arrays["halo_weights"] > 0
            previous_coverage = self._coverage_masks.get(partition_id)
            if previous_coverage is None:
                self._coverage_masks[partition_id] = coverage
            elif not np.array_equal(previous_coverage, coverage):
                raise WorkPackageRuntimeError(
                    f"model coverage differs inside Partition: {partition_id}"
                )
            accumulator = self._accumulators[partition_id]
            accumulator_before = directory_size(accumulator.root)
            halo_probabilities = np.asarray(arrays["halo_probabilities"])
            accumulator_channels = (
                len(list(self._profile.get("models") or [])) * 14
                if str(self._profile.get("strategy") or "") == "linear_1x1"
                else 14
            )
            accumulator_write_bytes = int(
                accumulator_channels
                * halo_probabilities.shape[1]
                * halo_probabilities.shape[2]
                * np.dtype(np.float32).itemsize
            )
            accumulator_estimated_write_bytes = accumulator_write_bytes + 64 * 1024
            completed_accumulator_models = accumulator.completed_model_ids()
            generation = len(completed_accumulator_models) + 1
            next_accumulator_path = (
                accumulator.root / f"accumulator_{generation:03d}.npy"
            )
            previous_accumulator_path = (
                accumulator.root / f"accumulator_{generation - 1:03d}.npy"
                if generation > 1
                else None
            )
            replaced_accumulator_bytes = (
                next_accumulator_path.stat().st_size
                if next_accumulator_path.is_file()
                else 0
            ) + (
                previous_accumulator_path.stat().st_size
                if previous_accumulator_path is not None
                and previous_accumulator_path.is_file()
                else 0
            )
            reservation = storage_budget.working_cache.reserve(
                f"fusion_accumulator:{partition_id}:{model_id}",
                write_bytes=accumulator_estimated_write_bytes,
                # Reserve the whole next generation while it coexists with the
                # active one; only those two files affect managed growth.
                managed_growth_bytes=max(
                    0, accumulator_estimated_write_bytes - replaced_accumulator_bytes
                ),
            )
            try:
                accumulator.add_model(model_id, arrays["halo_probabilities"])
            finally:
                reservation.settle(
                    directory_size(accumulator.root) - accumulator_before
                )

    def finalize_partition(
        self,
        partition_id: str,
        fusion_head: Any = None,
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Finalize the existing accumulator and return its matched coverage."""

        probabilities = self._accumulators[partition_id].finalize(
            fusion_head=fusion_head
        )
        coverage = self._coverage_masks[partition_id]
        return probabilities, coverage
