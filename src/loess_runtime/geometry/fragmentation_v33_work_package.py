"""Run V3.3 as a resumable second-stage Work Package.

The production mode waits for every V3 baseline Core, neighbouring V3 context,
and matching Fusion probability Artifact. It stages owner outputs, validates
the stitched global domain, then atomically publishes authoritative Fusion
``core_mask`` artifacts before geometry jobs may run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import rasterio
from affine import Affine

from labeling_tool.shared.contracts.run_spec import CLASS_ORDER, load_json, sha256_file
from labeling_tool.shared.state.run_state_db import RunStateDB, run_state_from_spec
from loess_runtime.geometry.fragmentation_v33_artifact_io import (
    ready_artifact_index,
    verified_artifact_path,
    write_atomic_json,
)
from loess_runtime.geometry.fragmentation_v33_candidate import (
    V33_POLICY_ID,
    apply_v33_candidate,
    executor_snapshot_sha256,
    policy_snapshot_sha256,
    runtime_policy,
)
from loess_runtime.geometry.fragmentation_v33_contract import (
    FragmentationV33WorkPackageError,
    expand_core_window,
    intersect_windows,
    normalize_window,
    physical_metrics,
    raster_window,
    window_shape,
    window_slices,
)
from loess_runtime.geometry.fragmentation_v33_finalization import (
    finalize_authoritative_v33,
    prepare_v33_finalization,
)
from loess_runtime.inference.partition_mosaic import write_atomic_partition_raster

CANDIDATE_JOB_TYPE = "fragmentation_v33"


def _execution_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    fragmentation = dict(spec.get("fragmentation_regularization") or {})
    if (
        fragmentation.get("enabled") is True
        and fragmentation.get("policy_id") == V33_POLICY_ID
        and fragmentation.get("publication") == "authoritative_fusion_core"
    ):
        return {
            "production": True,
            "buffer_pixels": int(fragmentation.get("buffer_pixels", 256)),
            "policy_sha256": str(fragmentation.get("policy_sha256") or ""),
            "executor_sha256": str(fragmentation.get("executor_sha256") or ""),
        }
    raise FragmentationV33WorkPackageError(
        "run spec does not select V3.3 authoritative production"
    )


def _empty_budget_audit() -> dict[str, Any]:
    """Return the exact no-op result for a Core with no strict-valid pixels."""

    empty_metrics = {
        "dynamic_fragments_4_connected": 0,
        "components_4_connected": 0,
    }
    return {
        "candidate_label": "V3.3",
        "full_audit": False,
        "audit_truncated": False,
        "empty_class_budget": True,
        "changed_pixel_count": 0,
        "protected_source_loss_pixel_count": 0,
        "transport_source_loss_pixel_count": 0,
        "gap_pixels": 0,
        "overlap_pixels": 0,
        "outside_pixels": 0,
        "raw_generated": 0,
        "proposals_canonical": 0,
        "duplicate_proposal_count": 0,
        "proposals_accepted": 0,
        "baseline": dict(empty_metrics),
        "result": dict(empty_metrics),
    }


def _read_probability(
    path: Path,
    *,
    owner_halo: Mapping[str, int],
    selected: Mapping[str, int],
    expected_crs: str,
    global_transform: Affine,
) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.dtype != np.float32 or values.shape != (
            len(CLASS_ORDER),
            *window_shape(owner_halo),
        ):
            raise FragmentationV33WorkPackageError(
                f"probability NPY contract differs: {path}"
            )
        rows, columns = window_slices(owner_halo, selected)
        return np.asarray(values[(slice(None), rows, columns)], dtype=np.float32)
    with rasterio.open(path) as source:
        expected_transform = global_transform * Affine.translation(
            int(owner_halo["x0"]), int(owner_halo["y0"])
        )
        if (
            source.count != len(CLASS_ORDER)
            or str(source.crs or "") != expected_crs
            or source.dtypes != ("uint16",) * len(CLASS_ORDER)
            or not source.transform.almost_equals(expected_transform)
        ):
            raise FragmentationV33WorkPackageError(
                f"probability raster contract differs: {path}"
            )
        expected_shape = window_shape(owner_halo)
        if (source.height, source.width) != expected_shape:
            raise FragmentationV33WorkPackageError(
                f"probability Halo shape differs: {path}"
            )
        raw = source.read(
            window=raster_window(owner_halo, selected),
            out_dtype="float32",
        )
        scales = np.asarray(source.scales, dtype=np.float32)
        if scales.shape != (len(CLASS_ORDER),) or not np.allclose(
            scales, np.float32(1.0 / 65535.0), rtol=0.0, atol=1e-12
        ):
            raise FragmentationV33WorkPackageError(f"probability scales differ: {path}")
        return raw * scales[:, None, None]


def _read_core(
    path: Path,
    *,
    owner_core: Mapping[str, int],
    selected: Mapping[str, int],
    expected_crs: str,
    global_transform: Affine,
    encoding: str = "indices",
) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.dtype != np.int16 or values.shape != window_shape(owner_core):
            raise FragmentationV33WorkPackageError(f"Core NPY contract differs: {path}")
        rows, columns = window_slices(owner_core, selected)
        result = np.asarray(values[rows, columns], dtype=np.int16)
        if encoding == "class_codes":
            mapped = np.full(result.shape, -1, dtype=np.int16)
            for index, code in enumerate(CLASS_ORDER):
                mapped[result == int(code)] = index
            if np.any((result != -1) & (mapped < 0)):
                raise FragmentationV33WorkPackageError(
                    f"Core NPY contains unknown class codes: {path}"
                )
            return mapped
        if (
            encoding != "indices"
            or np.any(result < -1)
            or np.any(result >= len(CLASS_ORDER))
        ):
            raise FragmentationV33WorkPackageError(
                f"Core NPY contains invalid class indices: {path}"
            )
        return result
    with rasterio.open(path) as source:
        expected_transform = global_transform * Affine.translation(
            int(owner_core["x0"]), int(owner_core["y0"])
        )
        if (
            source.count != 1
            or str(source.crs or "") != expected_crs
            or source.dtypes != ("int16",)
            or source.nodata != -1
            or not source.transform.almost_equals(expected_transform)
        ):
            raise FragmentationV33WorkPackageError(
                f"Core raster contract differs: {path}"
            )
        if (source.height, source.width) != window_shape(owner_core):
            raise FragmentationV33WorkPackageError(f"Core raster shape differs: {path}")
        return source.read(1, window=raster_window(owner_core, selected))


def _run_partition(
    spec: Mapping[str, Any],
    target: Mapping[str, Any],
    partitions: Sequence[Mapping[str, Any]],
    artifacts: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    stream_id: str,
    buffer_pixels: int,
    verified: set[tuple[str, int, str]],
    lease_guard: Callable[[], None],
    policy_sha256: str,
    executor_sha256: str,
    staging_key: str,
    production: bool,
    stage_only: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_id = str(spec["run_id"])
    run_dir = Path(str(spec["run_dir"]))
    transform = Affine(*[float(value) for value in spec["raster"]["transform"]])
    crs = str(spec["raster"]["crs"])
    global_window = {
        "x0": min(int(item["core_window"]["x0"]) for item in partitions),
        "y0": min(int(item["core_window"]["y0"]) for item in partitions),
        "x1": max(int(item["core_window"]["x1"]) for item in partitions),
        "y1": max(int(item["core_window"]["y1"]) for item in partitions),
    }
    target_id = str(target["partition_id"])
    target_core = normalize_window(target["core_window"])
    expanded = expand_core_window(target_core, global_window, buffer_pixels)
    height, width = window_shape(expanded)
    baseline = np.full((height, width), -1, dtype=np.int16)
    probabilities = np.zeros((len(CLASS_ORDER), height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.uint8)
    source_records: list[dict[str, Any]] = []

    for owner in partitions:
        owner_core = normalize_window(owner["core_window"])
        selected = intersect_windows(expanded, owner_core)
        if selected is None:
            continue
        owner_id = str(owner["partition_id"])
        context_artifact = artifacts.get((owner_id, "v3_context_core"))
        probability_artifact = artifacts.get((owner_id, "partition_probability"))
        context_path = verified_artifact_path(
            context_artifact,
            kind="v3_context_core",
            partition_id=owner_id,
            verified=verified,
        )
        probability_path = verified_artifact_path(
            probability_artifact,
            kind="partition_probability",
            partition_id=owner_id,
            verified=verified,
        )
        destination = window_slices(expanded, selected)
        baseline[destination] = _read_core(
            context_path,
            owner_core=owner_core,
            selected=selected,
            expected_crs=crs,
            global_transform=transform,
        )
        probabilities[(slice(None), *destination)] = _read_probability(
            probability_path,
            owner_halo=normalize_window(owner["halo_window"]),
            selected=selected,
            expected_crs=crs,
            global_transform=transform,
        )
        coverage[destination] += 1
        source_records.append(
            {
                "partition_id": owner_id,
                "v3_context_sha256": str(context_artifact["sha256"]),
                "probability_sha256": str(probability_artifact["sha256"]),
                "selected_window": selected,
            }
        )

    if not np.all(coverage == 1):
        raise FragmentationV33WorkPackageError(
            f"{target_id}: owner Core coverage differs; "
            f"missing={int(np.count_nonzero(coverage == 0))}, "
            f"overlap={int(np.count_nonzero(coverage > 1))}"
        )
    context_valid = baseline >= 0
    core_slice = window_slices(expanded, target_core)
    baseline_kind = "v3_baseline_core"
    target_mask_path = verified_artifact_path(
        artifacts.get((target_id, baseline_kind)),
        kind=baseline_kind,
        partition_id=target_id,
        verified=verified,
    )
    authoritative_v3 = _read_core(
        target_mask_path,
        owner_core=target_core,
        selected=target_core,
        expected_crs=crs,
        global_transform=transform,
        encoding=(
            "class_codes" if target_mask_path.suffix.lower() == ".npy" else "indices"
        ),
    ).astype(np.int16, copy=False)
    strict_valid = authoritative_v3 >= 0
    if not np.array_equal(
        baseline[core_slice][strict_valid], authoritative_v3[strict_valid]
    ):
        raise FragmentationV33WorkPackageError(
            f"{target_id}: V3 context does not match authoritative V3 Core"
        )
    budget = np.zeros((height, width), dtype=bool)
    budget[core_slice] = strict_valid
    probability_sums = probabilities.sum(axis=0, dtype=np.float64)
    probability_valid = context_valid
    probability_nonfinite_pixels = int(
        np.count_nonzero(
            probability_valid & ~np.all(np.isfinite(probabilities), axis=0)
        )
    )
    probability_negative_pixels = int(
        np.count_nonzero(probability_valid & np.any(probabilities < 0.0, axis=0))
    )
    probability_zero_sum_pixels = int(
        np.count_nonzero(probability_valid & (probability_sums <= 0.0))
    )
    probability_sum_tolerance = len(CLASS_ORDER) / 65535.0 + 1e-6
    probability_bad_sum_pixels = int(
        np.count_nonzero(
            probability_valid
            & (np.abs(probability_sums - 1.0) > probability_sum_tolerance)
        )
    )
    sorted_probabilities = np.partition(probabilities, -2, axis=0)
    top = sorted_probabilities[-1]
    second = sorted_probabilities[-2]
    argmax_tie_pixels = int(
        np.count_nonzero(strict_valid & (top[core_slice] == second[core_slice]))
    )
    near_tie_pixels = int(
        np.count_nonzero(
            strict_valid
            & ((top[core_slice] - second[core_slice]) <= (1.0 / 65535.0 + 1e-12))
        )
    )
    if any(
        (
            probability_nonfinite_pixels,
            probability_negative_pixels,
            probability_zero_sum_pixels,
            probability_bad_sum_pixels,
        )
    ):
        raise FragmentationV33WorkPackageError(
            f"{target_id}: Fusion probability contract failed"
        )
    metrics = physical_metrics(transform, crs, expanded)
    if np.any(budget):
        result, audit = apply_v33_candidate(
            baseline,
            class_codes=CLASS_ORDER,
            pixel_area_m2=metrics["pixel_area_m2"],
            pixel_size_m=(metrics["row_step_m"], metrics["column_step_m"]),
            valid_mask=context_valid,
            class_budget_mask=budget,
            probabilities=probabilities,
            baseline_kind="v3_cleaned",
            full_audit=False,
        )
    else:
        result = baseline.copy()
        audit = _empty_budget_audit()
    candidate_core = np.asarray(result[core_slice], dtype=np.int16).copy()
    candidate_core[~strict_valid] = -1
    if production and not stage_only:
        raise FragmentationV33WorkPackageError(
            "V3.3 authoritative output must use the durable finalize publication"
        )
    if production and stage_only:
        if not stream_id.startswith("fusion:"):
            raise FragmentationV33WorkPackageError(
                "V3.3 production requires a Fusion stream"
            )
        output_root = (
            run_dir
            / "fusion"
            / stream_id.split(":", 1)[1]
            / "fragmentation_v33_staging"
            / "attempts"
            / staging_key
        )
        mask_path = output_root / "raster_parts" / f"{target_id}_mask.tif"
        audit_path = output_root / "audits" / f"{target_id}.json"
    else:
        output_root = run_dir / "candidates" / "fragmentation_v33"
        mask_path = output_root / "raster_parts" / f"{target_id}_mask.tif"
        audit_path = output_root / "audits" / f"{target_id}.json"
    if target_mask_path.suffix.lower() == ".npy":
        profile = {
            "driver": "GTiff",
            "count": 1,
            "width": int(target_core["x1"]) - int(target_core["x0"]),
            "height": int(target_core["y1"]) - int(target_core["y0"]),
            "dtype": "int16",
            "nodata": -1,
            "crs": crs,
            "transform": transform
            * Affine.translation(int(target_core["x0"]), int(target_core["y0"])),
            "compress": "deflate",
            "BIGTIFF": "IF_SAFER",
        }
    else:
        with rasterio.open(target_mask_path) as source:
            profile = dict(source.profile)
    staged_mask = mask_path.with_name(f".{mask_path.name}.{staging_key}.staged")
    staged_audit = audit_path.with_name(f".{audit_path.name}.{staging_key}.staged")
    write_atomic_partition_raster(
        staged_mask,
        candidate_core,
        profile,
        tags={
            "classification_authority": (
                (
                    "fragmentation_v33_staged_fusion_core_v1"
                    if stage_only
                    else "fragmentation_v33_authoritative_fusion_core_v1"
                )
                if production
                else "isolated_fragmentation_v33_replay_v1"
            ),
            "fragmentation_policy_id": V33_POLICY_ID,
            "fragmentation_policy_sha256": policy_snapshot_sha256(),
            "fragmentation_executor_sha256": executor_sha256,
            "baseline": "authoritative_v3_owner_core",
            "production_replacement": str(bool(production and not stage_only)).lower(),
        },
    )
    lease_guard()
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged_mask, mask_path)
    valid_class = (candidate_core >= 0) & (candidate_core < len(CLASS_ORDER))
    gap_pixels = int(np.count_nonzero(strict_valid & ~valid_class))
    outside_pixels = int(np.count_nonzero(~strict_valid & (candidate_core >= 0)))
    invalid_pixels = int(
        np.count_nonzero((candidate_core < -1) | (candidate_core >= len(CLASS_ORDER)))
    )
    protected_codes = runtime_policy().protected_source_codes
    protected_indices = np.asarray(
        [CLASS_ORDER.index(int(code)) for code in protected_codes], dtype=np.int16
    )
    protected_source_loss = int(
        np.count_nonzero(
            strict_valid
            & np.isin(authoritative_v3, protected_indices)
            & (candidate_core != authoritative_v3)
        )
    )
    overlap_pixels = 0
    if any(
        (
            gap_pixels,
            overlap_pixels,
            outside_pixels,
            invalid_pixels,
            protected_source_loss,
        )
    ):
        raise FragmentationV33WorkPackageError(
            f"{target_id}: authoritative Core acceptance failed"
        )
    if protected_source_loss != int(audit["protected_source_loss_pixel_count"]):
        raise FragmentationV33WorkPackageError(
            f"{target_id}: protected-source audit disagrees with raster"
        )
    if int(audit["result"]["dynamic_fragments_4_connected"]) > int(
        audit["baseline"]["dynamic_fragments_4_connected"]
    ):
        raise FragmentationV33WorkPackageError(
            f"{target_id}: dynamic fragmentation increased"
        )
    acceptance = {
        "single_label": True,
        "gap_pixels": gap_pixels,
        "overlap_pixels": overlap_pixels,
        "outside_pixels": outside_pixels,
        "invalid_pixels": invalid_pixels,
        "protected_source_loss_pixel_count": protected_source_loss,
        "owner_core_coverage_min": int(coverage.min()),
        "owner_core_coverage_max": int(coverage.max()),
        "probability_nonfinite_pixels": probability_nonfinite_pixels,
        "probability_negative_pixels": probability_negative_pixels,
        "probability_zero_sum_pixels": probability_zero_sum_pixels,
        "probability_bad_sum_pixels": probability_bad_sum_pixels,
        "probability_sum_tolerance": probability_sum_tolerance,
        "argmax_tie_pixels": argmax_tie_pixels,
        "near_tie_pixels": near_tie_pixels,
    }
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "stream_id": stream_id,
        "partition_id": target_id,
        "global_core_window": target_core,
        "global_expanded_window": expanded,
        "buffer_pixels": int(buffer_pixels),
        "physical_metrics": metrics,
        "source_inputs": source_records,
        "baseline_core_matches_authoritative_v3": True,
        "candidate": audit,
        "acceptance": acceptance,
        "publication": (
            "staged_fusion_core"
            if stage_only
            else ("authoritative_fusion_core" if production else "isolated_replay")
        ),
        "production_replacement": bool(production and not stage_only),
        "candidate_policy_sha256": policy_sha256,
        "candidate_executor_sha256": executor_sha256,
        "output_mask_sha256": sha256_file(mask_path),
    }
    report["audit_sha256"] = hashlib.sha256(
        json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    write_atomic_json(staged_audit, report)
    lease_guard()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged_audit, audit_path)
    lease_guard()
    output_pair = {
        "run_id": run_id,
        "stream_id": stream_id,
        "partition_id": target_id,
        "mask_path": mask_path,
        "mask_byte_count": mask_path.stat().st_size,
        "mask_sha256": sha256_file(mask_path),
        "audit_path": audit_path,
        "audit_byte_count": audit_path.stat().st_size,
        "audit_sha256": sha256_file(audit_path),
        "production": None if stage_only else production,
    }
    summary = {
        "partition_id": target_id,
        "changed_pixel_count": int(audit["changed_pixel_count"]),
        "baseline_dynamic_fragments": int(
            audit["baseline"]["dynamic_fragments_4_connected"]
        ),
        "candidate_dynamic_fragments": int(
            audit["result"]["dynamic_fragments_4_connected"]
        ),
        "acceptance": acceptance,
        "output_mask_sha256": report["output_mask_sha256"],
    }
    return summary, output_pair


class _Heartbeat:
    def __init__(
        self,
        database: RunStateDB,
        job: Mapping[str, Any],
        *,
        lease_seconds: int,
    ) -> None:
        self.database = database
        self.job_id = int(job["job_id"])
        self.token = str(job["lease_token"])
        self.lease_seconds = max(30, int(lease_seconds))
        self.current = 0
        self.total = 0
        self.failed = threading.Event()
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stopped.wait(min(15.0, self.lease_seconds / 3)):
            if not self.database.jobs.heartbeat(
                self.job_id,
                self.token,
                current=self.current,
                total=self.total,
                lease_seconds=self.lease_seconds,
            ):
                self.failed.set()
                return

    def start(self, total: int) -> None:
        self.total = int(total)
        self.thread.start()

    def progress(self, current: int) -> None:
        if self.failed.is_set():
            raise FragmentationV33WorkPackageError("V3.3 job lease was lost")
        self.current = int(current)

    def fence(self) -> None:
        if self.failed.is_set() or not self.database.jobs.heartbeat(
            self.job_id,
            self.token,
            current=self.current,
            total=self.total,
            lease_seconds=self.lease_seconds,
        ):
            self.failed.set()
            raise FragmentationV33WorkPackageError("V3.3 job lease was lost")

    def stop_for_publication(self) -> None:
        """Fence the lease, then stop external DB traffic before publication."""

        self.fence()
        self.close()

    def close(self) -> None:
        self.stopped.set()
        self.thread.join(timeout=20)


def _finish_or_requeue_v33(
    database: RunStateDB, job: Mapping[str, Any], error: Exception
) -> None:
    if database.jobs.finish_job(
        int(job["job_id"]), str(job["lease_token"]), status="failed", error=str(error)
    ):
        database.jobs.requeue_failed_job(int(job["job_id"]))


def _fragmentation_attempt_key(job_id: int, lease_token: str) -> str:
    token_digest = hashlib.sha256(str(lease_token).encode()).hexdigest()
    return f"job{int(job_id)}.{token_digest}"


def _run_durable_partition_job(
    spec: Mapping[str, Any],
    database: RunStateDB,
    job: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    lease_seconds: int,
) -> dict[str, Any]:
    """Execute exactly one owner Core; publication remains staged."""

    run_id = str(spec["run_id"])
    unit = database.control_graph.get_spatial_unit(run_id, str(job["unit_id"]))
    if unit is None or str(unit.get("unit_type")) != "FragmentationV33Partition":
        raise FragmentationV33WorkPackageError("unexpected durable V3.3 partition unit")
    target = database.control_graph.get_partition(run_id, str(unit["owner_key"]))
    if target is None:
        raise FragmentationV33WorkPackageError("V3.3 owner Partition is unavailable")
    partitions = database.control_graph.partitions_for_run(run_id)
    artifacts = ready_artifact_index(database.artifacts, run_id, str(job["stream_id"]))
    heartbeat = _Heartbeat(database, job, lease_seconds=lease_seconds)
    heartbeat.start(1)
    try:
        attempt_key = _fragmentation_attempt_key(
            int(job["job_id"]), str(job["lease_token"])
        )
        summary, output_pair = _run_partition(
            spec,
            target,
            partitions,
            artifacts,
            stream_id=str(job["stream_id"]),
            buffer_pixels=int(contract["buffer_pixels"]),
            verified=set(),
            lease_guard=heartbeat.fence,
            policy_sha256=str(contract["policy_sha256"]),
            executor_sha256=str(contract["executor_sha256"]),
            staging_key=attempt_key,
            production=True,
            stage_only=True,
        )
        heartbeat.progress(1)
        heartbeat.stop_for_publication()
        with database.fragmentation_v33_attempt_commit(
            int(job["job_id"]), str(job["lease_token"])
        ) as publication:
            publication.artifacts.publish_fragmentation_v33_output_pair(
                output_pair["run_id"],
                output_pair["stream_id"],
                output_pair["partition_id"],
                mask_path=output_pair["mask_path"],
                mask_byte_count=output_pair["mask_byte_count"],
                mask_sha256=output_pair["mask_sha256"],
                audit_path=output_pair["audit_path"],
                audit_byte_count=output_pair["audit_byte_count"],
                audit_sha256=output_pair["audit_sha256"],
                production=output_pair["production"],
            )
            if not publication.jobs.complete_fragmentation_v33_job(
                int(job["job_id"]), str(job["lease_token"])
            ):
                raise FragmentationV33WorkPackageError(
                    "V3.3 partition lease expired before commit"
                )
        return {
            "status": "ready",
            "stage": "partition",
            "partition_id": str(target["partition_id"]),
            "summary": summary,
        }
    except Exception as error:
        _finish_or_requeue_v33(database, job, error)
        raise
    finally:
        heartbeat.close()


def _run_durable_finalize_job(
    spec: Mapping[str, Any],
    database: RunStateDB,
    job: Mapping[str, Any],
    *,
    lease_seconds: int,
) -> dict[str, Any]:
    """Audit the stitched domain, then cross the single publication barrier."""

    run_id = str(spec["run_id"])
    unit = database.control_graph.get_spatial_unit(run_id, str(job["unit_id"]))
    if unit is None or str(unit.get("unit_type")) != "FragmentationV33Finalize":
        raise FragmentationV33WorkPackageError("unexpected V3.3 finalize unit")
    partitions = database.control_graph.partitions_for_run(run_id)
    if not partitions:
        raise FragmentationV33WorkPackageError("V3.3 finalize has no Partition owners")
    prepared = prepare_v33_finalization(
        database.artifacts,
        run_id,
        str(job["stream_id"]),
        partitions,
    )
    heartbeat = _Heartbeat(database, job, lease_seconds=lease_seconds)
    heartbeat.start(len(prepared.partitions) * 3)
    run_dir = Path(str(spec["run_dir"]))
    stream_name = str(job["stream_id"]).split(":", 1)[1]
    canonical_root = run_dir / "fusion" / stream_name
    try:
        return finalize_authoritative_v33(
            spec,
            database,
            job,
            prepared,
            heartbeat,
            canonical_root,
        )
    except Exception as error:
        _finish_or_requeue_v33(database, job, error)
        raise
    finally:
        heartbeat.close()


def run_worker(
    run_spec_path: str | Path,
    *,
    worker_id: str,
    lease_seconds: int = 120,
    job_id: int | None = None,
    lease_token: str = "",
) -> dict[str, Any]:
    spec = load_json(Path(run_spec_path).resolve())
    contract = _execution_contract(spec)
    production = bool(contract["production"])
    current_policy_sha256 = policy_snapshot_sha256()
    current_executor_sha256 = executor_snapshot_sha256()
    if contract["policy_sha256"] != current_policy_sha256:
        raise FragmentationV33WorkPackageError(
            "V3.3 policy differs from the frozen Run contract"
        )
    if contract["executor_sha256"] != current_executor_sha256:
        raise FragmentationV33WorkPackageError(
            "V3.3 executor differs from the frozen Run contract"
        )
    database = run_state_from_spec(spec)
    run_id = str(spec["run_id"])
    if job_id is not None or lease_token:
        if job_id is None or not lease_token:
            raise FragmentationV33WorkPackageError(
                "external lease requires job_id and lease_token"
            )
        job = database.jobs.get_job(int(job_id))
        if (
            job is None
            or str(job.get("run_id")) != run_id
            or str(job.get("job_type")) != CANDIDATE_JOB_TYPE
            or str(job.get("status")) != "running"
            or str(job.get("lease_token")) != str(lease_token)
        ):
            raise FragmentationV33WorkPackageError(
                "external V3.3 lease is not owned by this worker"
            )
    else:
        job = database.jobs.lease_next_fragmentation_v33(
            run_id,
            str(worker_id),
            lease_seconds=max(30, int(lease_seconds)),
            max_running=max(
                1,
                min(
                    4,
                    int(
                        (spec.get("fragmentation_regularization") or {}).get(
                            "max_workers", 4
                        )
                    ),
                ),
            ),
        )
    if job is None:
        counts = database.jobs.job_counts(run_id, job_type=CANDIDATE_JOB_TYPE)
        return {
            "status": "ready" if counts.get("ready") else "not_ready",
            "job_counts": counts,
        }
    durable_unit = database.control_graph.get_spatial_unit(run_id, str(job["unit_id"]))
    if durable_unit is not None:
        durable_type = str(durable_unit.get("unit_type") or "")
        if durable_type == "FragmentationV33Partition":
            if not production:
                raise FragmentationV33WorkPackageError(
                    "durable V3.3 partition jobs require production policy"
                )
            return _run_durable_partition_job(
                spec, database, job, contract, lease_seconds=lease_seconds
            )
        if durable_type == "FragmentationV33Finalize":
            if not production:
                raise FragmentationV33WorkPackageError(
                    "durable V3.3 finalize requires production policy"
                )
            return _run_durable_finalize_job(
                spec, database, job, lease_seconds=lease_seconds
            )
    raise FragmentationV33WorkPackageError(
        "V3.3 production requires a durable partition or finalize unit"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the V3.3 partitioned production Work Package"
    )
    parser.add_argument("--run-spec", required=True)
    parser.add_argument("--worker-id", default=f"v33-{os.getpid()}")
    parser.add_argument("--lease-seconds", type=int, default=120)
    parser.add_argument("--job-id", type=int)
    parser.add_argument("--lease-token", default="")
    args = parser.parse_args()
    try:
        print(
            json.dumps(
                run_worker(
                    args.run_spec,
                    worker_id=args.worker_id,
                    lease_seconds=args.lease_seconds,
                    job_id=args.job_id,
                    lease_token=args.lease_token,
                ),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        print(
            json.dumps(
                {"status": "failed", "error": str(error)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
