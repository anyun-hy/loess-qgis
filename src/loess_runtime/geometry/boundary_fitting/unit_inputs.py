"""Read and validate the raster inputs for one geometry Unit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, TypeAlias, cast

import numpy as np
import rasterio  # type: ignore[import-untyped]
from numpy.typing import NDArray
from rasterio.windows import Window  # type: ignore[import-untyped]

from labeling_tool.shared.state.artifact_repository import ArtifactRepository
from loess_runtime.geometry.boundary_fitting.unit_errors import UnitRuntimeError

__all__ = [
    "PartitionLookup",
    "UnitFitInputs",
    "load_unit_fit_inputs",
    "read_unit_probabilities",
]

Float32Array: TypeAlias = NDArray[np.float32]
Int16Array: TypeAlias = NDArray[np.int16]
BoolArray: TypeAlias = NDArray[np.bool_]


class PartitionLookup(Protocol):
    """Read one Partition without exposing the rest of RunStateDB."""

    def __call__(
        self,
        run_id: str,
        partition_id: str,
        /,
    ) -> Mapping[str, Any] | None: ...


@dataclass(frozen=True)
class UnitFitInputs:
    """Validated arrays consumed by Unit polygonization and divider fitting."""

    labels: Int16Array
    confidence: Float32Array
    valid_mask: BoolArray


def _decode_partition_window(
    artifacts: ArtifactRepository,
    read_partition: PartitionLookup,
    run_id: str,
    stream_id: str,
    partition_id: str,
    unit_window: Mapping[str, int],
) -> Float32Array:
    partition = read_partition(run_id, partition_id)
    artifact = artifacts.artifact_for_stream_unit(
        run_id,
        stream_id,
        partition_id,
        "partition_probability",
    )
    if partition is None or artifact is None:
        raise UnitRuntimeError(
            f"Partition probability dependency is missing: {stream_id}/{partition_id}"
        )
    halo = partition["halo_window"]
    x0 = int(unit_window["x0"])
    y0 = int(unit_window["y0"])
    x1 = int(unit_window["x1"])
    y1 = int(unit_window["y1"])
    if not (
        int(halo["x0"]) <= x0 < x1 <= int(halo["x1"])
        and int(halo["y0"]) <= y0 < y1 <= int(halo["y1"])
    ):
        raise UnitRuntimeError(f"unit window is outside Partition Halo: {partition_id}")
    window = Window(
        x0 - int(halo["x0"]),
        y0 - int(halo["y0"]),
        x1 - x0,
        y1 - y0,
    )
    with rasterio.open(artifact["path"]) as source:
        # Decode directly into float32 and scale in place. Keeping this order
        # avoids retaining the quantized raster plus two conversion buffers.
        raw = cast(
            Float32Array,
            source.read(window=window, out_dtype="float32"),
        )
        scales = np.asarray(source.scales, dtype=np.float32)
    if raw.shape != (14, y1 - y0, x1 - x0):
        raise UnitRuntimeError(
            f"Partition probability crop has unexpected shape: {raw.shape}"
        )
    if scales.shape != (14,) or np.any(scales <= 0):
        raise UnitRuntimeError("Partition probability scale metadata is invalid")
    raw *= scales[:, None, None]
    return raw


def read_unit_probabilities(
    artifacts: ArtifactRepository,
    read_partition: PartitionLookup,
    run_id: str,
    stream_id: str,
    unit: Mapping[str, Any],
) -> tuple[Float32Array, BoolArray]:
    """Read and normalize probability Halos in dependency order.

    The first decoded array remains the accumulator and is returned to avoid a
    second fourteen-band allocation.
    """

    probabilities: Float32Array | None = None
    dependency_count = 0
    for partition_id in unit["dependency_ids"]:
        decoded = _decode_partition_window(
            artifacts,
            read_partition,
            run_id,
            stream_id,
            str(partition_id),
            unit["pixel_window"],
        )
        dependency_count += 1
        if probabilities is None:
            probabilities = decoded
        else:
            if decoded.shape != probabilities.shape:
                raise UnitRuntimeError("Partition Halo crops disagree on unit shape")
            np.add(probabilities, decoded, out=probabilities)
            del decoded
    if probabilities is None:
        raise UnitRuntimeError("spatial unit has no Partition dependencies")
    if dependency_count > 1:
        probabilities /= np.float32(dependency_count)
    denominator = probabilities.sum(axis=0, dtype=np.float32)
    valid = cast(BoolArray, denominator > 0)
    if np.any(valid):
        np.divide(
            probabilities,
            denominator[None, :, :],
            out=probabilities,
            where=valid[None, :, :],
        )
    if np.any(~valid):
        probabilities[:, ~valid] = 0.0
    return probabilities, valid


def _read_unit_confidence_surface(
    artifacts: ArtifactRepository,
    run_id: str,
    stream_id: str,
    unit: Mapping[str, Any],
) -> tuple[Float32Array, BoolArray]:
    """Read the lossless confidence surface produced before V3.3 finalize."""

    artifact = artifacts.artifact_for_stream_unit(
        run_id,
        stream_id,
        str(unit["unit_id"]),
        "unit_confidence",
    )
    if artifact is None:
        raise UnitRuntimeError(
            f"Unit confidence dependency is missing: {stream_id}/{unit['unit_id']}"
        )
    window = unit["pixel_window"]
    expected_shape = (
        int(window["y1"]) - int(window["y0"]),
        int(window["x1"]) - int(window["x0"]),
    )
    with rasterio.open(artifact["path"]) as source:
        if source.count != 1 or source.dtypes != ("float32",):
            raise UnitRuntimeError("Unit confidence raster contract is invalid")
        confidence = cast(
            Float32Array,
            source.read(1).astype(np.float32, copy=False),
        )
    if confidence.shape != expected_shape:
        raise UnitRuntimeError(
            f"Unit confidence raster has unexpected shape: {confidence.shape}"
        )
    valid = cast(BoolArray, confidence >= 0.0)
    if np.any(~np.isfinite(confidence[valid])) or np.any(confidence[valid] > 1.0):
        raise UnitRuntimeError("Unit confidence values are outside [0, 1]")
    return confidence, valid


def _read_unit_authoritative_labels(
    artifacts: ArtifactRepository,
    read_partition: PartitionLookup,
    run_id: str,
    stream_id: str,
    unit: Mapping[str, Any],
) -> tuple[Int16Array, BoolArray]:
    """Read the non-overlapping authoritative Core masks for one Unit."""

    window = unit["pixel_window"]
    x0, y0, x1, y1 = (int(window[key]) for key in ("x0", "y0", "x1", "y1"))
    labels = cast(
        Int16Array,
        np.full((y1 - y0, x1 - x0), -1, dtype=np.int16),
    )
    written = cast(BoolArray, np.zeros(labels.shape, dtype=bool))
    for partition_id in unit["dependency_ids"]:
        partition_key = str(partition_id)
        partition = read_partition(run_id, partition_key)
        artifact = artifacts.artifact_for_stream_unit(
            run_id,
            stream_id,
            partition_key,
            "core_mask",
        )
        if partition is None or artifact is None:
            raise UnitRuntimeError(
                "authoritative Core mask dependency is missing: "
                f"{stream_id}/{partition_key}"
            )
        core = partition["core_window"]
        cx0, cy0, cx1, cy1 = (int(core[key]) for key in ("x0", "y0", "x1", "y1"))
        ix0, iy0 = max(x0, cx0), max(y0, cy0)
        ix1, iy1 = min(x1, cx1), min(y1, cy1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        source_window = Window(ix0 - cx0, iy0 - cy0, ix1 - ix0, iy1 - iy0)
        destination = np.s_[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0]
        with rasterio.open(artifact["path"]) as source:
            values = cast(
                Int16Array,
                source.read(1, window=source_window).astype(np.int16),
            )
        if values.shape != labels[destination].shape:
            raise UnitRuntimeError(
                f"authoritative Core mask crop has unexpected shape: {partition_key}"
            )
        overlap = written[destination]
        if np.any(overlap & (labels[destination] != values)):
            raise UnitRuntimeError(
                f"overlapping authoritative Core masks disagree: {partition_key}"
            )
        labels[destination] = values
        written[destination] = True
    valid = cast(BoolArray, labels >= 0)
    if not np.all(written):
        raise UnitRuntimeError("authoritative Core masks leave a Unit coverage gap")
    return labels, valid


def load_unit_fit_inputs(
    artifacts: ArtifactRepository,
    read_partition: PartitionLookup,
    run_id: str,
    stream_id: str,
    unit: Mapping[str, Any],
    *,
    compact_confidence: bool,
) -> UnitFitInputs:
    """Load labels, confidence, and validity for geometry computation.

    Labels are int16 class indexes with negative values excluded by the
    validity mask. Confidence is float32 and valid values are within [0, 1].
    """

    if compact_confidence:
        confidence, probability_valid = _read_unit_confidence_surface(
            artifacts,
            run_id,
            stream_id,
            unit,
        )
    else:
        probabilities, probability_valid = read_unit_probabilities(
            artifacts,
            read_partition,
            run_id,
            stream_id,
            unit,
        )
        confidence = cast(Float32Array, probabilities.max(axis=0))
        # Unit fitting retains only the confidence surface. Release fourteen
        # probability bands before Rasterio/Shapely build geometry structures.
        del probabilities
    labels, valid_mask = _read_unit_authoritative_labels(
        artifacts,
        read_partition,
        run_id,
        stream_id,
        unit,
    )
    if np.any(valid_mask & ~probability_valid):
        raise UnitRuntimeError(
            "authoritative Core mask is valid where probability coverage is absent"
        )
    return UnitFitInputs(
        labels=labels,
        confidence=confidence,
        valid_mask=valid_mask,
    )
