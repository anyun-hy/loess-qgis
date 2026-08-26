"""Run V3.3 as a resumable second-stage Work Package.

The production mode waits for every V3 baseline Core, neighbouring V3 context,
and matching Fusion probability Artifact. It validates and publishes V3.3 as
the authoritative Fusion ``core_mask`` before geometry jobs may run. The same
worker retains an explicit isolated replay mode for historical equivalence
tests; replay outputs never become production Artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
for plugin_root in (ROOT / "qgis_plugins", ROOT / "runtime"):
    if plugin_root.is_dir() and str(plugin_root) not in sys.path:
        sys.path.insert(0, str(plugin_root))

from labeling_tool.core.run_spec import sha256_file
from labeling_tool.core.run_state_db import RunStateDB

from deployment_config import CLASS_ORDER, load_json
from fragmentation_v33_candidate import (
    V33_POLICY_ID,
    apply_v33_candidate,
    executor_snapshot_sha256,
    policy_snapshot_sha256,
    runtime_policy,
)
from partition_mosaic import _atomic_raster
from small_component_regularizer import physical_pixel_area_m2


CANDIDATE_JOB_TYPE = "fragmentation_v33"
PRODUCTION_UNIT_ID = "fragmentation_v33"
REPLAY_UNIT_ID = "fragmentation_v33_candidate"


class FragmentationV33WorkPackageError(RuntimeError):
    pass


def _execution_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    fragmentation = dict(spec.get("fragmentation_regularization") or {})
    if (
        fragmentation.get("enabled") is True
        and fragmentation.get("policy_id") == V33_POLICY_ID
        and fragmentation.get("publication") == "authoritative_fusion_core"
    ):
        return {
            "production": True,
            "unit_id": PRODUCTION_UNIT_ID,
            "buffer_pixels": int(fragmentation.get("buffer_pixels", 256)),
            "policy_sha256": str(fragmentation.get("policy_sha256") or ""),
            "executor_sha256": str(fragmentation.get("executor_sha256") or ""),
        }
    comparison = dict(fragmentation.get("comparison") or {})
    if comparison.get("enabled") is True:
        return {
            "production": False,
            "unit_id": REPLAY_UNIT_ID,
            "buffer_pixels": int(comparison.get("buffer_pixels", 256)),
            "policy_sha256": str(
                comparison.get("candidate_policy_sha256") or ""
            ),
            "executor_sha256": str(
                comparison.get("candidate_executor_sha256") or ""
            ),
        }
    raise FragmentationV33WorkPackageError(
        "run spec does not select V3.3 production or isolated replay"
    )


def _window(value: Mapping[str, Any]) -> dict[str, int]:
    return {key: int(value[key]) for key in ("x0", "y0", "x1", "y1")}


def _intersection(
    first: Mapping[str, int], second: Mapping[str, int]
) -> dict[str, int] | None:
    result = {
        "x0": max(int(first["x0"]), int(second["x0"])),
        "y0": max(int(first["y0"]), int(second["y0"])),
        "x1": min(int(first["x1"]), int(second["x1"])),
        "y1": min(int(first["y1"]), int(second["y1"])),
    }
    if result["x0"] >= result["x1"] or result["y0"] >= result["y1"]:
        return None
    return result


def _shape(value: Mapping[str, int]) -> tuple[int, int]:
    return int(value["y1"]) - int(value["y0"]), int(value["x1"]) - int(
        value["x0"]
    )


def _slices(
    parent: Mapping[str, int], child: Mapping[str, int]
) -> tuple[slice, slice]:
    return (
        slice(int(child["y0"]) - int(parent["y0"]), int(child["y1"]) - int(parent["y0"])),
        slice(int(child["x0"]) - int(parent["x0"]), int(child["x1"]) - int(parent["x0"])),
    )


def _expanded_core(
    core: Mapping[str, int], global_window: Mapping[str, int], margin: int
) -> dict[str, int]:
    return {
        "x0": max(int(global_window["x0"]), int(core["x0"]) - int(margin)),
        "y0": max(int(global_window["y0"]), int(core["y0"]) - int(margin)),
        "x1": min(int(global_window["x1"]), int(core["x1"]) + int(margin)),
        "y1": min(int(global_window["y1"]), int(core["y1"]) + int(margin)),
    }


def _raster_window(parent: Mapping[str, int], child: Mapping[str, int]) -> Window:
    return Window(
        col_off=int(child["x0"]) - int(parent["x0"]),
        row_off=int(child["y0"]) - int(parent["y0"]),
        width=int(child["x1"]) - int(child["x0"]),
        height=int(child["y1"]) - int(child["y0"]),
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact_map(
    database: RunStateDB, run_id: str, stream_id: str
) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(item["unit_id"]), str(item["kind"])): dict(item)
        for item in database.artifacts_for_stream(
            run_id, stream_id, status="ready"
        )
    }


def _verified_path(
    artifact: Mapping[str, Any] | None,
    *,
    kind: str,
    partition_id: str,
    verified: set[tuple[str, int, str]] | None = None,
) -> Path:
    if artifact is None:
        raise FragmentationV33WorkPackageError(
            f"missing {kind} Artifact for {partition_id}"
        )
    path = Path(str(artifact.get("path") or ""))
    expected_size = int(artifact.get("byte_count") or -1)
    expected_sha = str(artifact.get("sha256") or "")
    key = (str(path), expected_size, expected_sha)
    if key in (verified or set()):
        return path
    if not path.is_file() or path.stat().st_size != expected_size or sha256_file(path) != expected_sha:
        raise FragmentationV33WorkPackageError(
            f"changed {kind} Artifact for {partition_id}: {path}"
        )
    if verified is not None:
        verified.add(key)
    return path


def _physical_metrics(
    transform: Affine, crs: str, window: Mapping[str, int]
) -> dict[str, float]:
    local = transform * Affine.translation(int(window["x0"]), int(window["y0"]))
    height, width = _shape(window)
    area = float(physical_pixel_area_m2(local, crs, height=height, width=width))
    determinant = abs(local.a * local.e - local.b * local.d)
    if determinant <= 0:
        raise FragmentationV33WorkPackageError("processing transform has zero area")
    scale = math.sqrt(area / determinant)
    row_m = math.hypot(local.b, local.e) * scale
    return {
        "pixel_area_m2": area,
        "row_step_m": row_m,
        "column_step_m": area / row_m,
    }


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
            *_shape(owner_halo),
        ):
            raise FragmentationV33WorkPackageError(
                f"probability NPY contract differs: {path}"
            )
        rows, columns = _slices(owner_halo, selected)
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
        expected_shape = _shape(owner_halo)
        if (source.height, source.width) != expected_shape:
            raise FragmentationV33WorkPackageError(
                f"probability Halo shape differs: {path}"
            )
        raw = source.read(
            window=_raster_window(owner_halo, selected),
            out_dtype="float32",
        )
        scales = np.asarray(source.scales, dtype=np.float32)
        if scales.shape != (len(CLASS_ORDER),) or not np.allclose(
            scales, np.float32(1.0 / 65535.0), rtol=0.0, atol=1e-12
        ):
            raise FragmentationV33WorkPackageError(
                f"probability scales differ: {path}"
            )
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
        if values.dtype != np.int16 or values.shape != _shape(owner_core):
            raise FragmentationV33WorkPackageError(
                f"Core NPY contract differs: {path}"
            )
        rows, columns = _slices(owner_core, selected)
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
        if encoding != "indices" or np.any(result < -1) or np.any(
            result >= len(CLASS_ORDER)
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
        if (source.height, source.width) != _shape(owner_core):
            raise FragmentationV33WorkPackageError(
                f"Core raster shape differs: {path}"
            )
        return source.read(1, window=_raster_window(owner_core, selected))


def _commit_output_artifact(
    database: RunStateDB,
    run_id: str,
    stream_id: str,
    partition_id: str,
    *,
    kind: str,
    path: Path,
) -> int:
    artifact_id = database.register_artifact(
        run_id,
        kind,
        path,
        stream_id=stream_id,
        unit_id=partition_id,
    )
    existing = database.get_artifact(artifact_id)
    digest = sha256_file(path)
    if existing and str(existing.get("status")) == "ready":
        if (
            int(existing.get("byte_count") or -1) == path.stat().st_size
            and str(existing.get("sha256")) == digest
        ):
            return artifact_id
        raise FragmentationV33WorkPackageError(
            f"ready V3.3 Artifact changed: {path}"
        )
    if not database.mark_artifact_ready(
        artifact_id,
        byte_count=path.stat().st_size,
        sha256=digest,
    ):
        raise FragmentationV33WorkPackageError(
            f"cannot commit V3.3 Artifact: {path}"
        )
    return artifact_id


def _run_partition(
    spec: Mapping[str, Any],
    database: RunStateDB,
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
) -> dict[str, Any]:
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
    target_core = _window(target["core_window"])
    expanded = _expanded_core(target_core, global_window, buffer_pixels)
    height, width = _shape(expanded)
    baseline = np.full((height, width), -1, dtype=np.int16)
    probabilities = np.zeros((len(CLASS_ORDER), height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.uint8)
    source_records: list[dict[str, Any]] = []

    for owner in partitions:
        owner_core = _window(owner["core_window"])
        selected = _intersection(expanded, owner_core)
        if selected is None:
            continue
        owner_id = str(owner["partition_id"])
        context_artifact = artifacts.get((owner_id, "v3_context_core"))
        probability_artifact = artifacts.get((owner_id, "partition_probability"))
        context_path = _verified_path(
            context_artifact,
            kind="v3_context_core",
            partition_id=owner_id,
            verified=verified,
        )
        probability_path = _verified_path(
            probability_artifact,
            kind="partition_probability",
            partition_id=owner_id,
            verified=verified,
        )
        destination = _slices(expanded, selected)
        baseline[destination] = _read_core(
            context_path,
            owner_core=owner_core,
            selected=selected,
            expected_crs=crs,
            global_transform=transform,
        )
        probabilities[(slice(None), *destination)] = _read_probability(
            probability_path,
            owner_halo=_window(owner["halo_window"]),
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
    core_slice = _slices(expanded, target_core)
    baseline_kind = "v3_baseline_core"
    target_mask_path = _verified_path(
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
    if not np.array_equal(baseline[core_slice][strict_valid], authoritative_v3[strict_valid]):
        raise FragmentationV33WorkPackageError(
            f"{target_id}: V3 context does not match authoritative V3 Core"
        )
    budget = np.zeros((height, width), dtype=bool)
    budget[core_slice] = strict_valid
    probability_sums = probabilities.sum(axis=0, dtype=np.float64)
    probability_valid = context_valid
    probability_nonfinite_pixels = int(
        np.count_nonzero(
            probability_valid
            & ~np.all(np.isfinite(probabilities), axis=0)
        )
    )
    probability_negative_pixels = int(
        np.count_nonzero(
            probability_valid & np.any(probabilities < 0.0, axis=0)
        )
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
    metrics = _physical_metrics(transform, crs, expanded)
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
    if production:
        if not stream_id.startswith("fusion:"):
            raise FragmentationV33WorkPackageError(
                "V3.3 production requires a Fusion stream"
            )
        output_root = run_dir / "fusion" / stream_id.split(":", 1)[1]
        mask_path = output_root / "raster_parts" / f"{target_id}_mask.tif"
        audit_path = output_root / "fragmentation_v33_audits" / f"{target_id}.json"
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
    _atomic_raster(
        staged_mask,
        candidate_core,
        profile,
        tags={
            "classification_authority": (
                "fragmentation_v33_authoritative_fusion_core_v1"
                if production
                else "isolated_fragmentation_v33_replay_v1"
            ),
            "fragmentation_policy_id": V33_POLICY_ID,
            "fragmentation_policy_sha256": policy_snapshot_sha256(),
            "fragmentation_executor_sha256": executor_sha256,
            "baseline": "authoritative_v3_owner_core",
            "production_replacement": str(bool(production)).lower(),
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
    if any((gap_pixels, overlap_pixels, outside_pixels, invalid_pixels, protected_source_loss)):
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
            "authoritative_fusion_core" if production else "isolated_replay"
        ),
        "production_replacement": bool(production),
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
    _atomic_json(staged_audit, report)
    lease_guard()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged_audit, audit_path)
    lease_guard()
    database.publish_fragmentation_v33_output_pair(
        run_id,
        stream_id,
        target_id,
        mask_path=mask_path,
        mask_byte_count=mask_path.stat().st_size,
        mask_sha256=sha256_file(mask_path),
        audit_path=audit_path,
        audit_byte_count=audit_path.stat().st_size,
        audit_sha256=sha256_file(audit_path),
        production=production,
    )
    return {
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
            if not self.database.heartbeat(
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
        if self.failed.is_set() or not self.database.heartbeat(
            self.job_id,
            self.token,
            current=self.current,
            total=self.total,
            lease_seconds=self.lease_seconds,
        ):
            self.failed.set()
            raise FragmentationV33WorkPackageError("V3.3 job lease was lost")

    def close(self) -> None:
        self.stopped.set()
        self.thread.join(timeout=20)


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
    database = RunStateDB(spec["state_db"])
    run_id = str(spec["run_id"])
    if job_id is not None or lease_token:
        if job_id is None or not lease_token:
            raise FragmentationV33WorkPackageError(
                "external lease requires job_id and lease_token"
            )
        job = database.get_job(int(job_id))
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
        job = database.lease_next_fragmentation_v33(
            run_id,
            str(worker_id),
            lease_seconds=max(30, int(lease_seconds)),
        )
    if job is None:
        counts = database.job_counts(run_id, job_type=CANDIDATE_JOB_TYPE)
        return {"status": "ready" if counts.get("ready") else "not_ready", "job_counts": counts}
    if str(job["unit_id"]) != str(contract["unit_id"]):
        raise FragmentationV33WorkPackageError("unexpected V3.3 unit")
    partitions = database.partitions_for_run(run_id)
    if not partitions:
        raise FragmentationV33WorkPackageError("V3.3 has no Partition owners")
    global_window = {
        "x0": min(int(item["core_window"]["x0"]) for item in partitions),
        "y0": min(int(item["core_window"]["y0"]) for item in partitions),
        "x1": max(int(item["core_window"]["x1"]) for item in partitions),
        "y1": max(int(item["core_window"]["y1"]) for item in partitions),
    }
    core_area = sum(
        (int(item["core_window"]["x1"]) - int(item["core_window"]["x0"]))
        * (int(item["core_window"]["y1"]) - int(item["core_window"]["y0"]))
        for item in partitions
    )
    global_area = (
        (global_window["x1"] - global_window["x0"])
        * (global_window["y1"] - global_window["y0"])
    )
    overlap_pairs = 0
    for index, first in enumerate(partitions):
        for second in partitions[index + 1 :]:
            if _intersection(
                _window(first["core_window"]), _window(second["core_window"])
            ) is not None:
                overlap_pairs += 1
    if core_area != global_area or overlap_pairs:
        raise FragmentationV33WorkPackageError(
            "Partition Core ownership has a geometric gap or overlap"
        )
    artifacts = _artifact_map(database, run_id, str(job["stream_id"]))
    heartbeat = _Heartbeat(database, job, lease_seconds=lease_seconds)
    heartbeat.start(len(partitions))
    summaries: list[dict[str, Any]] = []
    verified: set[tuple[str, int, str]] = set()
    mask_kind = "core_mask" if production else "v33_candidate_mask"
    audit_kind = (
        "fragmentation_v33_audit" if production else "v33_candidate_audit"
    )
    try:
        for index, partition in enumerate(partitions, start=1):
            heartbeat.progress(index - 1)
            existing = artifacts.get(
                (str(partition["partition_id"]), mask_kind)
            )
            existing_audit = artifacts.get(
                (str(partition["partition_id"]), audit_kind)
            )
            if (existing is None) != (existing_audit is None):
                raise FragmentationV33WorkPackageError(
                    f"{partition['partition_id']}: partial V3.3 publication requires reset"
                )
            if existing is not None and existing_audit is not None:
                _verified_path(
                    existing,
                    kind=mask_kind,
                    partition_id=str(partition["partition_id"]),
                    verified=verified,
                )
                _verified_path(
                    existing_audit,
                    kind=audit_kind,
                    partition_id=str(partition["partition_id"]),
                    verified=verified,
                )
                existing_report = load_json(Path(str(existing_audit["path"])))
                if (
                    existing_report.get("candidate_policy_sha256")
                    != current_policy_sha256
                    or existing_report.get("candidate_executor_sha256")
                    != current_executor_sha256
                ):
                    raise FragmentationV33WorkPackageError(
                        "resumed V3.3 candidate uses a different frozen contract"
                    )
                acceptance = dict(existing_report.get("acceptance") or {})
                if any(
                    int(acceptance.get(key, -1)) != 0
                    for key in (
                        "gap_pixels",
                        "overlap_pixels",
                        "outside_pixels",
                        "invalid_pixels",
                        "protected_source_loss_pixel_count",
                        "probability_nonfinite_pixels",
                        "probability_negative_pixels",
                        "probability_zero_sum_pixels",
                        "probability_bad_sum_pixels",
                    )
                ):
                    raise FragmentationV33WorkPackageError(
                        "resumed V3.3 output did not pass acceptance"
                    )
                candidate = dict(existing_report.get("candidate") or {})
                summaries.append(
                    {
                        "partition_id": str(partition["partition_id"]),
                        "output_mask_sha256": str(existing["sha256"]),
                        "changed_pixel_count": int(
                            candidate.get("changed_pixel_count", 0)
                        ),
                        "baseline_dynamic_fragments": int(
                            (candidate.get("baseline") or {}).get(
                                "dynamic_fragments_4_connected", 0
                            )
                        ),
                        "candidate_dynamic_fragments": int(
                            (candidate.get("result") or {}).get(
                                "dynamic_fragments_4_connected", 0
                            )
                        ),
                        "acceptance": acceptance,
                    }
                )
                continue
            summaries.append(
                _run_partition(
                    spec,
                    database,
                    partition,
                    partitions,
                    artifacts,
                    stream_id=str(job["stream_id"]),
                    buffer_pixels=int(contract["buffer_pixels"]),
                    verified=verified,
                    lease_guard=heartbeat.fence,
                    policy_sha256=current_policy_sha256,
                    executor_sha256=current_executor_sha256,
                    staging_key=(
                        f"job{int(job['job_id'])}.{str(job['lease_token'])[:16]}"
                    ),
                    production=production,
                )
            )
            heartbeat.progress(index)
        heartbeat.close()
        acceptance_keys = (
            "gap_pixels",
            "overlap_pixels",
            "outside_pixels",
            "invalid_pixels",
            "protected_source_loss_pixel_count",
            "probability_nonfinite_pixels",
            "probability_negative_pixels",
            "probability_zero_sum_pixels",
            "probability_bad_sum_pixels",
        )
        acceptance_totals = {
            key: sum(int(item["acceptance"][key]) for item in summaries)
            for key in acceptance_keys
        }
        acceptance_totals.update(
            {
                "partition_core_area": int(core_area),
                "global_core_area": int(global_area),
                "core_overlap_pair_count": int(overlap_pairs),
                "argmax_tie_pixels": sum(
                    int(item["acceptance"]["argmax_tie_pixels"])
                    for item in summaries
                ),
                "near_tie_pixels": sum(
                    int(item["acceptance"]["near_tie_pixels"])
                    for item in summaries
                ),
                "changed_pixel_count": sum(
                    int(item["changed_pixel_count"]) for item in summaries
                ),
                "baseline_dynamic_fragments": sum(
                    int(item["baseline_dynamic_fragments"]) for item in summaries
                ),
                "result_dynamic_fragments": sum(
                    int(item["candidate_dynamic_fragments"]) for item in summaries
                ),
            }
        )
        if any(acceptance_totals[key] for key in acceptance_keys):
            raise FragmentationV33WorkPackageError(
                "V3.3 global raster acceptance failed"
            )
        if acceptance_totals["result_dynamic_fragments"] > acceptance_totals[
            "baseline_dynamic_fragments"
        ]:
            raise FragmentationV33WorkPackageError(
                "V3.3 global dynamic fragmentation increased"
            )
        report = {
            "schema_version": 1,
            "status": "ready",
            "validation_status": "passed",
            "publication": (
                "authoritative_fusion_core" if production else "isolated_replay"
            ),
            "production_replacement": production,
            "run_id": run_id,
            "stream_id": str(job["stream_id"]),
            "partition_count": len(partitions),
            "acceptance": acceptance_totals,
            "partitions": summaries,
        }
        if production:
            report_path = (
                Path(str(spec["run_dir"]))
                / "fusion"
                / str(job["stream_id"]).split(":", 1)[1]
                / "fragmentation_v33_report.json"
            )
        else:
            report_path = (
                Path(str(spec["run_dir"]))
                / "candidates"
                / "fragmentation_v33"
                / "work_package_report.json"
            )
        report_kind = (
            "fragmentation_v33_report" if production else "v33_candidate_report"
        )
        existing_report_artifact = artifacts.get(
            (str(contract["unit_id"]), report_kind)
        )
        if existing_report_artifact is not None:
            existing_report_path = _verified_path(
                existing_report_artifact,
                kind=report_kind,
                partition_id=str(contract["unit_id"]),
                verified=verified,
            )
            if load_json(existing_report_path) != report:
                raise FragmentationV33WorkPackageError(
                    "ready V3.3 acceptance report differs on resume"
                )
        else:
            _atomic_json(report_path, report)
            _commit_output_artifact(
                database,
                run_id,
                str(job["stream_id"]),
                str(contract["unit_id"]),
                kind=report_kind,
                path=report_path,
            )
        if not database.complete_fragmentation_v33_job(
            int(job["job_id"]),
            str(job["lease_token"]),
        ):
            raise FragmentationV33WorkPackageError(
                "V3.3 lease expired before final commit"
            )
        return report
    except Exception as error:
        heartbeat.close()
        if database.finish_job(
            int(job["job_id"]),
            str(job["lease_token"]),
            status="failed",
            error=str(error),
        ):
            database.requeue_failed_job(int(job["job_id"]))
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the V3.3 production or isolated replay Work Package"
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
