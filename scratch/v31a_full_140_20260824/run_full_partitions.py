#!/usr/bin/env python3
"""Run the complete 140-Core raw -> production V3 -> V3.1-A comparison.

The source snapshot stores only 128 pixels of halo per Partition.  This runner
does not treat those halos as independent test panels.  For every output Core
it first assembles an exact 256-pixel global context from the non-overlapping
owner Cores.  Stage V3 budgets the decoder-valid owner Core and publishes only
that owner's V3 Core.  Only after every owner V3 output has passed its SHA
barrier may Stage V3.1 assemble a shared baseline from those publications;
V3.1 budgets and publishes the strict owner Core.  Archived halo overlaps are
compared bit-for-bit once during global preflight.

Real runs are deliberately all-or-nothing: exactly 140 snapshot Partitions are
required and ``--partitions`` is forbidden.  Partial execution is available
only for the generated multi-Partition ``--self-test`` fixture.  Output arrays
contain class codes (not class indices); invalid pixels are -1.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping
import uuid

import numpy as np
from rasterio.transform import Affine


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "inference_scripts"))

from deployment_config import CLASS_ORDER  # noqa: E402
from fragmentation_v3 import policy_snapshot as v3_policy_snapshot  # noqa: E402
from fragmentation_v3 import production_policy  # noqa: E402
from fragmentation_v31_candidate import (  # noqa: E402
    apply_v31a_candidate,
    policy_snapshot as v31a_policy_snapshot,
)
from small_component_regularizer import (  # noqa: E402
    physical_pixel_area_m2,
    regularize_small_components,
)


SCHEMA_VERSION = 1
REAL_PARTITION_COUNT = 140
REAL_CONTEXT_PIXELS = 256
PROBABILITY_ARTIFACT = "input_blended_probabilities_f32_npy"
DECODER_VALID_ARTIFACT = "input_decoder_valid_npy"
STRICT_VALID_ARTIFACT = "final_core_strict_range_valid_npy"


class FullRunError(RuntimeError):
    """Raised when an input or all-Core execution contract is violated."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _save_npy(path: Path, values: np.ndarray) -> None:
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".npy",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.save(handle, np.ascontiguousarray(values), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _window(value: Mapping[str, Any], label: str) -> dict[str, int]:
    try:
        result = {key: int(value[key]) for key in ("x0", "y0", "x1", "y1")}
    except (KeyError, TypeError, ValueError) as exc:
        raise FullRunError(f"{label} must contain integer x0,y0,x1,y1") from exc
    if result["x0"] >= result["x1"] or result["y0"] >= result["y1"]:
        raise FullRunError(f"{label} is empty or reversed: {result}")
    return result


def _shape(window: Mapping[str, int]) -> tuple[int, int]:
    return window["y1"] - window["y0"], window["x1"] - window["x0"]


def _intersection(*windows: Mapping[str, int]) -> dict[str, int] | None:
    result = {
        "x0": max(item["x0"] for item in windows),
        "y0": max(item["y0"] for item in windows),
        "x1": min(item["x1"] for item in windows),
        "y1": min(item["y1"] for item in windows),
    }
    return result if result["x0"] < result["x1"] and result["y0"] < result["y1"] else None


def _local_slices(container: Mapping[str, int], selected: Mapping[str, int]) -> tuple[slice, slice]:
    return (
        slice(selected["y0"] - container["y0"], selected["y1"] - container["y0"]),
        slice(selected["x0"] - container["x0"], selected["x1"] - container["x0"]),
    )


def _expand(window: Mapping[str, int], global_window: Mapping[str, int], pixels: int) -> dict[str, int]:
    return {
        "x0": max(global_window["x0"], window["x0"] - pixels),
        "y0": max(global_window["y0"], window["y0"] - pixels),
        "x1": min(global_window["x1"], window["x1"] + pixels),
        "y1": min(global_window["y1"], window["y1"] + pixels),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FullRunError(f"cannot read JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FullRunError(f"JSON root must be an object: {path}")
    return value


def _artifact(
    partition_root: Path, stage: str, artifact_name: str,
) -> tuple[Path, dict[str, Any], str]:
    partition_manifest_path = partition_root / "manifest.json"
    partition_manifest = _read_json(partition_manifest_path)
    stage_ref = (partition_manifest.get("stages") or {}).get(stage)
    if not isinstance(stage_ref, dict):
        raise FullRunError(f"{partition_root.name}: missing archived {stage} stage")
    stage_path = partition_root / str(stage_ref.get("path", ""))
    if not stage_path.is_file():
        raise FullRunError(f"{partition_root.name}: missing stage manifest {stage_path}")
    actual_stage_sha = _sha256_file(stage_path)
    if actual_stage_sha != stage_ref.get("sha256"):
        raise FullRunError(f"{partition_root.name}: {stage} stage SHA-256 mismatch")
    stage_manifest = _read_json(stage_path)
    item = (stage_manifest.get("artifacts") or {}).get(artifact_name)
    if not isinstance(item, dict):
        raise FullRunError(f"{partition_root.name}: missing artifact {artifact_name}")
    path = partition_root / str(item.get("path", ""))
    if not path.is_file():
        raise FullRunError(f"{partition_root.name}: missing artifact file {path}")
    expected = str(item.get("sha256", ""))
    if len(expected) != 64:
        raise FullRunError(f"{partition_root.name}: {artifact_name} lacks SHA-256")
    return path, item, expected


def _validate_grid(partitions: list[dict[str, Any]]) -> dict[str, int]:
    by_y: dict[tuple[int, int], list[dict[str, int]]] = {}
    for entry in partitions:
        core = entry["core_window"]
        by_y.setdefault((core["y0"], core["y1"]), []).append(core)
    rows = sorted(by_y)
    if not rows:
        raise FullRunError("snapshot has no Partitions")
    for previous, current in zip(rows, rows[1:]):
        if previous[1] != current[0]:
            raise FullRunError(f"Core row coverage has a gap or overlap: {previous}, {current}")
    reference_x: list[tuple[int, int]] | None = None
    for row in rows:
        spans = sorted((item["x0"], item["x1"]) for item in by_y[row])
        for previous, current in zip(spans, spans[1:]):
            if previous[1] != current[0]:
                raise FullRunError(f"Core column coverage has a gap or overlap in row {row}")
        if reference_x is None:
            reference_x = spans
        elif spans != reference_x:
            raise FullRunError("Core owner columns differ between rows")
    assert reference_x is not None
    return {"x0": reference_x[0][0], "y0": rows[0][0], "x1": reference_x[-1][1], "y1": rows[-1][1]}


def _code_sha256() -> dict[str, str]:
    paths = [
        REPO_ROOT / "inference_scripts" / "deployment_config.py",
        REPO_ROOT / "inference_scripts" / "fragmentation_v3.py",
        REPO_ROOT / "inference_scripts" / "small_component_regularizer.py",
        REPO_ROOT / "inference_scripts" / "fragmentation_v31_candidate" / "__init__.py",
        REPO_ROOT / "inference_scripts" / "fragmentation_v31_candidate" / "candidate.py",
        Path(__file__).resolve(),
    ]
    return {str(path.relative_to(REPO_ROOT)): _sha256_file(path) for path in paths}


def _load_snapshot(path: Path, *, self_test: bool) -> dict[str, Any]:
    manifest = _read_json(path)
    raw_partitions = manifest.get("partitions")
    if not isinstance(raw_partitions, list):
        raise FullRunError("snapshot_manifest.partitions must be a list")
    if not self_test and len(raw_partitions) != REAL_PARTITION_COUNT:
        raise FullRunError(
            f"real execution requires exactly {REAL_PARTITION_COUNT} Partitions; found {len(raw_partitions)}"
        )
    root = path.parent
    seen: set[str] = set()
    partitions: list[dict[str, Any]] = []
    for raw in raw_partitions:
        if not isinstance(raw, dict):
            raise FullRunError("each snapshot Partition must be an object")
        partition_id = str(raw.get("partition_id", ""))
        if not partition_id or partition_id in seen:
            raise FullRunError(f"empty or duplicate partition_id: {partition_id!r}")
        seen.add(partition_id)
        core = _window(raw.get("core_window") or {}, f"{partition_id}.core_window")
        halo = _window(raw.get("halo_window") or {}, f"{partition_id}.halo_window")
        if _intersection(core, halo) != core:
            raise FullRunError(f"{partition_id}: halo_window does not contain Core")
        partition_root = root / "partitions" / partition_id
        probability_path, probability_meta, probability_sha = _artifact(
            partition_root, "input", PROBABILITY_ARTIFACT,
        )
        decoder_valid_path, decoder_valid_meta, decoder_valid_sha = _artifact(
            partition_root, "input", DECODER_VALID_ARTIFACT,
        )
        strict_valid_path, strict_valid_meta, strict_valid_sha = _artifact(
            partition_root, "final", STRICT_VALID_ARTIFACT,
        )
        probability_shape = tuple(int(v) for v in probability_meta.get("shape", ()))
        decoder_valid_shape = tuple(int(v) for v in decoder_valid_meta.get("shape", ()))
        strict_valid_shape = tuple(int(v) for v in strict_valid_meta.get("shape", ()))
        if probability_shape != (len(CLASS_ORDER), *_shape(halo)):
            raise FullRunError(f"{partition_id}: archived probability shape/window mismatch")
        if decoder_valid_shape != _shape(halo):
            raise FullRunError(f"{partition_id}: decoder-valid shape/halo mismatch")
        if strict_valid_shape != _shape(core):
            raise FullRunError(f"{partition_id}: strict-valid shape/Core mismatch")
        if (
            probability_meta.get("dtype") != "float32"
            or decoder_valid_meta.get("dtype") != "bool"
            or strict_valid_meta.get("dtype") != "bool"
        ):
            raise FullRunError(f"{partition_id}: source dtype contract mismatch")
        partitions.append({
            "partition_id": partition_id,
            "row": int(raw.get("row", -1)), "col": int(raw.get("col", -1)),
            "core_window": core, "halo_window": halo,
            "probability_path": str(probability_path),
            "probability_expected_sha256": probability_sha,
            "probability_shape": list(probability_shape),
            "decoder_valid_path": str(decoder_valid_path),
            "decoder_valid_expected_sha256": decoder_valid_sha,
            "decoder_valid_shape": list(decoder_valid_shape),
            "strict_valid_path": str(strict_valid_path),
            "strict_valid_expected_sha256": strict_valid_sha,
            "strict_valid_shape": list(strict_valid_shape),
        })
    global_window = _validate_grid(partitions)
    transform_values = manifest.get("processing_transform")
    if not isinstance(transform_values, list) or len(transform_values) != 6:
        raise FullRunError("snapshot processing_transform must contain six values")
    transform = [float(value) for value in transform_values]
    if not all(math.isfinite(value) for value in transform):
        raise FullRunError("snapshot processing_transform is not finite")
    crs = str(manifest.get("source_raster_crs") or "")
    if not crs:
        raise FullRunError("snapshot source_raster_crs is missing")
    return {
        "path": str(path.resolve()), "sha256": _sha256_file(path),
        "root": str(root.resolve()), "manifest": manifest,
        "partitions": partitions, "global_window": global_window,
        "processing_transform": transform, "crs": crs,
    }


def _verify_source(entry: dict[str, Any]) -> dict[str, Any]:
    probability_path = Path(entry["probability_path"])
    decoder_valid_path = Path(entry["decoder_valid_path"])
    strict_valid_path = Path(entry["strict_valid_path"])
    probability_sha = _sha256_file(probability_path)
    decoder_valid_sha = _sha256_file(decoder_valid_path)
    strict_valid_sha = _sha256_file(strict_valid_path)
    if probability_sha != entry["probability_expected_sha256"]:
        raise FullRunError(f"{entry['partition_id']}: probability SHA-256 mismatch")
    if decoder_valid_sha != entry["decoder_valid_expected_sha256"]:
        raise FullRunError(f"{entry['partition_id']}: decoder-valid SHA-256 mismatch")
    if strict_valid_sha != entry["strict_valid_expected_sha256"]:
        raise FullRunError(f"{entry['partition_id']}: strict-valid SHA-256 mismatch")
    probability = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    decoder_valid = np.load(decoder_valid_path, mmap_mode="r", allow_pickle=False)
    strict_valid = np.load(strict_valid_path, mmap_mode="r", allow_pickle=False)
    if probability.shape != tuple(entry["probability_shape"]) or probability.dtype != np.float32:
        raise FullRunError(f"{entry['partition_id']}: probability NPY metadata changed")
    if decoder_valid.shape != tuple(entry["decoder_valid_shape"]) or decoder_valid.dtype != np.bool_:
        raise FullRunError(f"{entry['partition_id']}: decoder-valid NPY metadata changed")
    if strict_valid.shape != tuple(entry["strict_valid_shape"]) or strict_valid.dtype != np.bool_:
        raise FullRunError(f"{entry['partition_id']}: strict-valid NPY metadata changed")
    return {
        "partition_id": entry["partition_id"],
        "probability_sha256": probability_sha,
        "decoder_valid_sha256": decoder_valid_sha,
        "strict_valid_sha256": strict_valid_sha,
    }


def _verify_all_sources(entries: list[dict[str, Any]], workers: int) -> dict[str, dict[str, str]]:
    results: dict[str, dict[str, str]] = {}
    if workers == 1:
        values = map(_verify_source, entries)
        for value in values:
            results[value["partition_id"]] = value
        return results
    # macOS Codex/QGIS sandboxes can deny SC_SEM_NSEMS_MAX even though threads
    # are available.  Tencent/Linux uses isolated worker processes; the local
    # fallback keeps --workers covered by the synthetic contract test.
    executor_class = ThreadPoolExecutor if sys.platform == "darwin" else ProcessPoolExecutor
    with executor_class(max_workers=workers) as executor:
        futures = {executor.submit(_verify_source, entry): entry["partition_id"] for entry in entries}
        for future in as_completed(futures):
            value = future.result()
            results[value["partition_id"]] = value
    return results


def _physical_metrics(transform_values: list[float], crs: str, global_window: Mapping[str, int]) -> dict[str, float]:
    transform = Affine(*transform_values) * Affine.translation(
        global_window["x0"], global_window["y0"],
    )
    height, width = _shape(global_window)
    area = float(physical_pixel_area_m2(transform, crs, height=height, width=width))
    determinant = abs(transform.a * transform.e - transform.b * transform.d)
    if determinant <= 0:
        raise FullRunError("processing_transform has zero area")
    scale = math.sqrt(area / determinant)
    row_m = math.hypot(transform.b, transform.e) * scale
    col_m = area / row_m
    return {"pixel_area_m2": area, "row_step_m": row_m, "column_step_m": col_m}


def _compare_duplicate_probabilities(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate every archived overlapping halo pair exactly once per run."""

    pairs: list[dict[str, Any]] = []
    pixel_count = 0
    ordered = sorted(entries, key=lambda item: item["partition_id"])
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1:]:
            overlap = _intersection(left["halo_window"], right["halo_window"])
            if overlap is None:
                continue
            left_array = np.load(left["probability_path"], mmap_mode="r", allow_pickle=False)
            right_array = np.load(right["probability_path"], mmap_mode="r", allow_pickle=False)
            left_rows, left_cols = _local_slices(left["halo_window"], overlap)
            right_rows, right_cols = _local_slices(right["halo_window"], overlap)
            overlap_height = overlap["y1"] - overlap["y0"]
            for offset in range(0, overlap_height, 64):
                count = min(64, overlap_height - offset)
                left_slice = left_array[:, slice(left_rows.start + offset, left_rows.start + offset + count), left_cols]
                right_slice = right_array[:, slice(right_rows.start + offset, right_rows.start + offset + count), right_cols]
                if not np.array_equal(left_slice, right_slice):
                    difference = float(np.max(np.abs(left_slice - right_slice)))
                    raise FullRunError(
                        f"duplicate source probabilities disagree: {left['partition_id']} vs "
                        f"{right['partition_id']} at {overlap}; max_abs={difference}"
                    )
            overlap_pixels = (overlap["x1"] - overlap["x0"]) * overlap_height
            pixel_count += overlap_pixels
            pairs.append({
                "left_partition_id": left["partition_id"],
                "right_partition_id": right["partition_id"],
                "global_overlap_window": overlap,
                "overlap_pixel_count": overlap_pixels,
            })
    result = {
        "method": "bitwise_array_equal_float32", "consistent": True,
        "scope": "global_each_overlapping_halo_pair_exactly_once",
        "overlap_pair_count": len(pairs), "overlap_pixel_count": pixel_count,
        "pairs": pairs,
    }
    result["audit_sha256"] = _sha256_json(result)
    return result


def _stitch_raw_context(
    target: dict[str, Any], entries: list[dict[str, Any]], global_window: dict[str, int],
    context_pixels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int], list[dict[str, Any]]]:
    expanded = _expand(target["core_window"], global_window, context_pixels)
    height, width = _shape(expanded)
    probabilities = np.empty((len(CLASS_ORDER), height, width), dtype=np.float32)
    context_valid = np.zeros((height, width), dtype=bool)
    strict_valid = np.zeros((height, width), dtype=bool)
    owner_coverage = np.zeros((height, width), dtype=np.uint8)
    owners = [entry for entry in entries if _intersection(expanded, entry["core_window"]) is not None]
    for owner in owners:
        selected = _intersection(expanded, owner["core_window"])
        assert selected is not None
        destination = _local_slices(expanded, selected)
        source_probability = np.load(owner["probability_path"], mmap_mode="r", allow_pickle=False)
        source_rows, source_cols = _local_slices(owner["halo_window"], selected)
        probabilities[(slice(None), *destination)] = source_probability[(slice(None), source_rows, source_cols)]
        decoder_valid = np.load(owner["decoder_valid_path"], mmap_mode="r", allow_pickle=False)
        context_valid[destination] = decoder_valid[source_rows, source_cols]
        owner_strict_valid = np.load(owner["strict_valid_path"], mmap_mode="r", allow_pickle=False)
        strict_valid[destination] = owner_strict_valid[_local_slices(owner["core_window"], selected)]
        owner_coverage[destination] += 1
    if not np.all(owner_coverage == 1):
        missing = int(np.count_nonzero(owner_coverage == 0))
        overlap = int(np.count_nonzero(owner_coverage > 1))
        raise FullRunError(
            f"{target['partition_id']}: owner Core coverage is not exact; missing={missing}, overlap={overlap}"
        )
    core_mask = np.zeros((height, width), dtype=bool)
    core_mask[_local_slices(expanded, target["core_window"])] = True
    if np.any(strict_valid & ~context_valid):
        raise FullRunError(
            f"{target['partition_id']}: strict-valid pixels are absent from decoder-valid context"
        )
    core_budget = core_mask & context_valid
    return probabilities, context_valid, strict_valid, core_budget, expanded, owners


def _core_transform(transform_values: list[float], core: Mapping[str, int]) -> list[float]:
    value = Affine(*transform_values) * Affine.translation(core["x0"], core["y0"])
    return [float(item) for item in tuple(value)[:6]]


def _source_inputs(owners: list[dict[str, Any]], job: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "partition_id": owner["partition_id"],
            "probability_path": owner["probability_path"],
            "probability_sha256": job["verified_sources"][owner["partition_id"]]["probability_sha256"],
            "decoder_valid_path": owner["decoder_valid_path"],
            "decoder_valid_sha256": job["verified_sources"][owner["partition_id"]]["decoder_valid_sha256"],
            "strict_valid_path": owner["strict_valid_path"],
            "strict_valid_sha256": job["verified_sources"][owner["partition_id"]]["strict_valid_sha256"],
            "global_core_window": owner["core_window"],
            "global_halo_window": owner["halo_window"],
        }
        for owner in owners
    ]


def _run_v3_partition(job: dict[str, Any]) -> dict[str, Any]:
    target = next(item for item in job["entries"] if item["partition_id"] == job["partition_id"])
    probabilities, context_valid, strict_valid, core_budget, expanded, owners = _stitch_raw_context(
        target, job["entries"], job["global_window"], job["context_pixels"],
    )
    raw = np.argmax(probabilities, axis=0).astype(np.int16)
    raw[~context_valid] = -1
    confidence = probabilities.max(axis=0).astype(np.float32)
    metrics = _physical_metrics(job["processing_transform"], job["crs"], expanded)
    if np.any(context_valid):
        selected = probabilities[:, context_valid]
        if not np.all(np.isfinite(selected)) or np.any(selected < 0) or np.any(selected > 1):
            raise FullRunError(f"{target['partition_id']}: invalid probability value over decoder-valid pixels")
        if not np.allclose(selected.sum(axis=0, dtype=np.float64), 1.0, rtol=0, atol=1e-3):
            raise FullRunError(f"{target['partition_id']}: probabilities do not sum to one")
    if np.any(core_budget):
        v3, v3_audit = regularize_small_components(
            raw, class_codes=CLASS_ORDER, pixel_area_m2=metrics["pixel_area_m2"],
            policy=production_policy(), valid_mask=context_valid, confidence=confidence,
            class_budget_mask=core_budget,
        )
    else:
        v3 = raw.copy()
        v3_audit = {
            "skipped": True, "reason": "empty_owner_core_decoder_valid",
            "class_budget_pixel_count": 0, "changed_pixel_count": 0,
        }
    core_slice = _local_slices(expanded, target["core_window"])
    core_context_valid = np.asarray(context_valid[core_slice], dtype=bool)
    core_valid = np.asarray(strict_valid[core_slice], dtype=bool)
    v3_context_core = np.asarray(v3[core_slice], dtype=np.int16).copy()
    v3_context_core[~core_context_valid] = -1
    code_values = np.asarray(CLASS_ORDER, dtype=np.int16)
    raw_core = np.full(core_valid.shape, -1, dtype=np.int16)
    v3_core = np.full(core_valid.shape, -1, dtype=np.int16)
    raw_indices = raw[core_slice]
    v3_indices = v3[core_slice]
    raw_core[core_valid] = code_values[raw_indices[core_valid]]
    v3_core[core_valid] = code_values[v3_indices[core_valid]]
    core_geometry = np.zeros(context_valid.shape, dtype=bool)
    core_geometry[core_slice] = True
    discarded_nonowner_changes = int(np.count_nonzero(context_valid & ~core_geometry & (v3 != raw)))
    staging = Path(job["staging_root"]) / f"v3-{target['partition_id']}"
    final_dir = Path(job["output_root"]) / "partitions" / target["partition_id"] / "stage_v3"
    if final_dir.exists():
        raise FullRunError(f"refusing to overwrite existing V3 stage: {final_dir}")
    staging.mkdir(parents=True, exist_ok=False)
    _save_npy(staging / "raw_core.npy", raw_core)
    _save_npy(staging / "v3_context_core.npy", v3_context_core)
    _save_npy(staging / "v3_core.npy", v3_core)
    _save_npy(staging / "valid_core.npy", core_valid)
    output_sha = {
        name: _sha256_file(staging / name)
        for name in ("raw_core.npy", "v3_context_core.npy", "v3_core.npy", "valid_core.npy")
    }
    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "stage": "v3", "partition_id": target["partition_id"],
        "execution_fingerprint_sha256": job["execution_fingerprint_sha256"],
        "snapshot_manifest": job["snapshot_manifest"],
        "snapshot_manifest_sha256": job["snapshot_manifest_sha256"],
        "code_sha256": job["code_sha256"],
        "class_codes": list(CLASS_ORDER), "label_encoding": "class_codes_int16_invalid_minus_one",
        "global_window": job["global_window"], "global_core_window": target["core_window"],
        "global_expanded_window": expanded, "context_pixels": job["context_pixels"],
        "processing_transform": job["processing_transform"],
        "core_transform": _core_transform(job["processing_transform"], target["core_window"]),
        "crs": job["crs"], "physical_metrics": metrics,
        "source_inputs": _source_inputs(owners, job),
        "v3_policy_snapshot": job["v3_policy_snapshot"],
        "v3_policy_snapshot_sha256": job["v3_policy_snapshot_sha256"],
        "v3_audit": v3_audit,
        "discarded_nonowner_v3_change_pixel_count": discarded_nonowner_changes,
        "coverage": {
            "published_owner_core_only": True, "owner_core_pixel_count": int(core_valid.size),
            "context_valid_pixel_count": int(context_valid.sum()),
            "owner_core_decoder_valid_pixel_count": int(core_context_valid.sum()),
            "core_strict_valid_pixel_count": int(core_valid.sum()),
            "core_strict_invalid_pixel_count": int(core_valid.size - core_valid.sum()),
            "expanded_owner_coverage_exact_once": True,
        },
        "outputs": {
            "raw": {"path": "raw_core.npy", "sha256": output_sha["raw_core.npy"], "shape": list(raw_core.shape), "dtype": "int16"},
            "v3_context": {"path": "v3_context_core.npy", "sha256": output_sha["v3_context_core.npy"], "shape": list(v3_context_core.shape), "dtype": "int16", "encoding": "class_indices_invalid_minus_one"},
            "v3": {"path": "v3_core.npy", "sha256": output_sha["v3_core.npy"], "shape": list(v3_core.shape), "dtype": "int16"},
            "valid": {"path": "valid_core.npy", "sha256": output_sha["valid_core.npy"], "shape": list(core_valid.shape), "dtype": "bool"},
        },
    }
    audit["audit_sha256"] = _sha256_json(audit)
    _atomic_json(staging / "audit.json", audit)
    output_sha["audit.json"] = _sha256_file(staging / "audit.json")
    _atomic_json(staging / "outputs_sha256.json", {"files": output_sha})
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.rename(staging, final_dir)
    return {
        "partition_id": target["partition_id"],
        "global_core_window": target["core_window"], "global_expanded_window": expanded,
        "core_transform": audit["core_transform"], "crs": job["crs"],
        "physical_metrics": metrics,
        "expanded_context_valid_pixel_count": int(context_valid.sum()),
        "owner_core_decoder_valid_pixel_count": int(core_context_valid.sum()),
        "core_strict_valid_pixel_count": int(core_valid.sum()),
        "owner_core_pixel_count": int(core_valid.size),
        "discarded_nonowner_v3_change_pixel_count": discarded_nonowner_changes,
        "outputs": {
            key: {**value, "path": f"partitions/{target['partition_id']}/stage_v3/{value['path']}"}
            for key, value in audit["outputs"].items()
        },
        "audit": {
            "path": f"partitions/{target['partition_id']}/stage_v3/audit.json",
            "sha256": output_sha["audit.json"],
        },
    }


def _validate_stage_dir(
    stage_dir: Path, fingerprint: str, expected_files: set[str],
) -> tuple[dict[str, Any], dict[str, str]]:
    audit_path = stage_dir / "audit.json"
    hashes_path = stage_dir / "outputs_sha256.json"
    if not audit_path.is_file() or not hashes_path.is_file():
        raise FullRunError(f"resume stage is incomplete: {stage_dir}")
    audit = _read_json(audit_path)
    if audit.get("execution_fingerprint_sha256") != fingerprint:
        raise FullRunError(f"resume stage fingerprint differs: {stage_dir}")
    hashes = _read_json(hashes_path).get("files")
    if not isinstance(hashes, dict):
        raise FullRunError(f"invalid outputs_sha256.json: {stage_dir}")
    if set(hashes) != expected_files:
        raise FullRunError(
            f"resume stage fixed output set differs: {stage_dir}; "
            f"expected={sorted(expected_files)}, actual={sorted(hashes)}"
        )
    actual_files = {path.name for path in stage_dir.iterdir() if path.is_file()}
    if actual_files != expected_files | {"outputs_sha256.json"}:
        raise FullRunError(f"resume stage contains missing or extra files: {stage_dir}")
    for name, expected in hashes.items():
        path = stage_dir / name
        if not path.is_file() or _sha256_file(path) != expected:
            raise FullRunError(f"resume stage output missing or changed: {path}")
    return audit, hashes


def _validate_v3_stage(
    output_root: Path, partition_id: str, fingerprint: str,
) -> dict[str, Any]:
    root = output_root / "partitions" / partition_id / "stage_v3"
    audit, hashes = _validate_stage_dir(
        root, fingerprint,
        {"raw_core.npy", "v3_context_core.npy", "v3_core.npy", "valid_core.npy", "audit.json"},
    )
    outputs = audit.get("outputs") or {}
    return {
        "partition_id": partition_id,
        "global_core_window": audit["global_core_window"],
        "global_expanded_window": audit["global_expanded_window"],
        "core_transform": audit["core_transform"], "crs": audit["crs"],
        "physical_metrics": audit["physical_metrics"],
        "expanded_context_valid_pixel_count": audit["coverage"]["context_valid_pixel_count"],
        "owner_core_decoder_valid_pixel_count": audit["coverage"]["owner_core_decoder_valid_pixel_count"],
        "core_strict_valid_pixel_count": audit["coverage"]["core_strict_valid_pixel_count"],
        "owner_core_pixel_count": audit["coverage"]["owner_core_pixel_count"],
        "discarded_nonowner_v3_change_pixel_count": audit.get("discarded_nonowner_v3_change_pixel_count", 0),
        "outputs": {
            key: {**value, "path": f"partitions/{partition_id}/stage_v3/{value['path']}"}
            for key, value in outputs.items()
        },
        "audit": {"path": f"partitions/{partition_id}/stage_v3/audit.json", "sha256": hashes["audit.json"]},
    }


def _stitch_v31_context(
    target: dict[str, Any], entries: list[dict[str, Any]], global_window: dict[str, int],
    context_pixels: int, output_root: Path, v3_stages: Mapping[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int], list[dict[str, Any]]]:
    expanded = _expand(target["core_window"], global_window, context_pixels)
    height, width = _shape(expanded)
    baseline = np.full((height, width), -1, dtype=np.int16)
    probabilities = np.empty((len(CLASS_ORDER), height, width), dtype=np.float32)
    context_valid = np.zeros((height, width), dtype=bool)
    strict_valid = np.zeros((height, width), dtype=bool)
    owner_coverage = np.zeros((height, width), dtype=np.uint8)
    owners = [entry for entry in entries if _intersection(expanded, entry["core_window"]) is not None]
    for owner in owners:
        selected = _intersection(expanded, owner["core_window"])
        assert selected is not None
        destination = _local_slices(expanded, selected)
        owner_core_slice = _local_slices(owner["core_window"], selected)
        stage_entry = v3_stages[owner["partition_id"]]
        baseline_path = output_root / stage_entry["outputs"]["v3_context"]["path"]
        owner_baseline = np.load(baseline_path, mmap_mode="r", allow_pickle=False)
        baseline[destination] = owner_baseline[owner_core_slice]
        probability = np.load(owner["probability_path"], mmap_mode="r", allow_pickle=False)
        halo_slice = _local_slices(owner["halo_window"], selected)
        probabilities[(slice(None), *destination)] = probability[(slice(None), *halo_slice)]
        decoder_valid = np.load(owner["decoder_valid_path"], mmap_mode="r", allow_pickle=False)
        context_valid[destination] = decoder_valid[halo_slice]
        owner_strict = np.load(owner["strict_valid_path"], mmap_mode="r", allow_pickle=False)
        strict_valid[destination] = owner_strict[owner_core_slice]
        owner_coverage[destination] += 1
    if not np.all(owner_coverage == 1):
        raise FullRunError(f"{target['partition_id']}: V3 owner baseline coverage is not exact")
    if np.any(strict_valid & ~context_valid):
        raise FullRunError(f"{target['partition_id']}: strict-valid is outside decoder-valid")
    if np.any(context_valid & ((baseline < 0) | (baseline >= len(CLASS_ORDER)))):
        raise FullRunError(f"{target['partition_id']}: owner V3 baseline has invalid class indices")
    if np.any(~context_valid & (baseline != -1)):
        raise FullRunError(f"{target['partition_id']}: owner V3 baseline did not preserve decoder-invalid pixels")
    core_mask = np.zeros((height, width), dtype=bool)
    core_mask[_local_slices(expanded, target["core_window"])] = True
    return baseline, probabilities, context_valid, strict_valid, core_mask & strict_valid, expanded, owners


def _run_v31_partition(job: dict[str, Any]) -> dict[str, Any]:
    target = next(item for item in job["entries"] if item["partition_id"] == job["partition_id"])
    output_root = Path(job["output_root"])
    baseline, probabilities, context_valid, strict_valid, core_budget, expanded, owners = _stitch_v31_context(
        target, job["entries"], job["global_window"], job["context_pixels"],
        output_root, job["v3_stages"],
    )
    metrics = _physical_metrics(job["processing_transform"], job["crs"], expanded)
    confidence = probabilities.max(axis=0).astype(np.float32)
    if np.any(core_budget):
        v31a, v31a_audit = apply_v31a_candidate(
            baseline, class_codes=CLASS_ORDER, pixel_area_m2=metrics["pixel_area_m2"],
            pixel_size_m=(metrics["row_step_m"], metrics["column_step_m"]),
            valid_mask=context_valid, class_budget_mask=core_budget,
            probabilities=probabilities, confidence=confidence,
            baseline_kind="v3_cleaned", full_audit=True,
        )
    else:
        v31a = baseline.copy()
        v31a_audit = {
            "skipped": True, "reason": "empty_owner_core_strict_valid",
            "class_budget_mask_pixel_count": 0, "changed_pixel_count": 0,
            "proposal_reject_reason_counts": {}, "full_audit": True,
            "audit_truncated": False,
        }
    core_slice = _local_slices(expanded, target["core_window"])
    core_valid = np.asarray(strict_valid[core_slice], dtype=bool)
    result_indices = v31a[core_slice]
    code_values = np.asarray(CLASS_ORDER, dtype=np.int16)
    v31a_core = np.full(core_valid.shape, -1, dtype=np.int16)
    v31a_core[core_valid] = code_values[result_indices[core_valid]]
    reject_counts = v31a_audit.get("proposal_reject_reason_counts") or {}
    generation_rejects = v31a_audit.get("proposal_generation_reject_reason_counts") or {}
    outside_core = int(reject_counts.get("outside_core_owner", 0)) + int(generation_rejects.get("outside_core_owner", 0))
    staging = Path(job["staging_root"]) / f"v31-{target['partition_id']}"
    final_dir = output_root / "partitions" / target["partition_id"] / "stage_v31"
    if final_dir.exists():
        raise FullRunError(f"refusing to overwrite existing V3.1 stage: {final_dir}")
    staging.mkdir(parents=True, exist_ok=False)
    _save_npy(staging / "v31a_core.npy", v31a_core)
    output_sha = {"v31a_core.npy": _sha256_file(staging / "v31a_core.npy")}
    v3_entry = job["v3_stages"][target["partition_id"]]
    baseline_sources = [
        {
            "partition_id": owner["partition_id"],
            "v3_context_path": job["v3_stages"][owner["partition_id"]]["outputs"]["v3_context"]["path"],
            "v3_context_sha256": job["v3_stages"][owner["partition_id"]]["outputs"]["v3_context"]["sha256"],
            "global_core_window": owner["core_window"],
        }
        for owner in owners
    ]
    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "stage": "v31", "partition_id": target["partition_id"],
        "execution_fingerprint_sha256": job["execution_fingerprint_sha256"],
        "snapshot_manifest": job["snapshot_manifest"],
        "snapshot_manifest_sha256": job["snapshot_manifest_sha256"],
        "code_sha256": job["code_sha256"],
        "class_codes": list(CLASS_ORDER), "label_encoding": "class_codes_int16_invalid_minus_one",
        "global_window": job["global_window"], "global_core_window": target["core_window"],
        "global_expanded_window": expanded, "context_pixels": job["context_pixels"],
        "processing_transform": job["processing_transform"],
        "core_transform": _core_transform(job["processing_transform"], target["core_window"]),
        "crs": job["crs"], "physical_metrics": metrics,
        "stage_v3_complete_required": True,
        "owner_v3_baseline_sources": baseline_sources,
        "target_v3_stage_audit": v3_entry["audit"],
        "v31a_policy_snapshot": job["v31a_policy_snapshot"],
        "v31a_policy_snapshot_sha256": job["v31a_policy_snapshot_sha256"],
        "v31a_audit": v31a_audit,
        "v31a_outside_core_owner_rejection_count": outside_core,
        "coverage": {
            "published_strict_core_only": True,
            "expanded_context_valid_pixel_count": int(context_valid.sum()),
            "core_strict_valid_pixel_count": int(core_valid.sum()),
            "owner_v3_baseline_coverage_exact_once": True,
        },
        "outputs": {
            "v31a": {"path": "v31a_core.npy", "sha256": output_sha["v31a_core.npy"], "shape": list(v31a_core.shape), "dtype": "int16"},
        },
    }
    audit["audit_sha256"] = _sha256_json(audit)
    _atomic_json(staging / "audit.json", audit)
    output_sha["audit.json"] = _sha256_file(staging / "audit.json")
    _atomic_json(staging / "outputs_sha256.json", {"files": output_sha})
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.rename(staging, final_dir)
    return {
        "partition_id": target["partition_id"],
        "global_core_window": target["core_window"], "global_expanded_window": expanded,
        "core_transform": audit["core_transform"], "crs": job["crs"],
        "physical_metrics": metrics,
        "expanded_context_valid_pixel_count": int(context_valid.sum()),
        "owner_core_decoder_valid_pixel_count": v3_entry["owner_core_decoder_valid_pixel_count"],
        "core_strict_valid_pixel_count": int(core_valid.sum()),
        "valid_pixel_count": int(core_valid.sum()),
        "owner_core_pixel_count": v3_entry["owner_core_pixel_count"],
        "discarded_nonowner_v3_change_pixel_count": v3_entry["discarded_nonowner_v3_change_pixel_count"],
        "v31a_outside_core_owner_rejection_count": outside_core,
        "outputs": {
            "raw": v3_entry["outputs"]["raw"],
            "v3": v3_entry["outputs"]["v3"],
            "valid": v3_entry["outputs"]["valid"],
            "v31a": {**audit["outputs"]["v31a"], "path": f"partitions/{target['partition_id']}/stage_v31/v31a_core.npy"},
        },
        "stage_v3_audit": v3_entry["audit"],
        "audit": {"path": f"partitions/{target['partition_id']}/stage_v31/audit.json", "sha256": output_sha["audit.json"]},
    }


def _validate_v31_stage(
    output_root: Path, partition_id: str, fingerprint: str,
    v3_entry: dict[str, Any],
) -> dict[str, Any]:
    root = output_root / "partitions" / partition_id / "stage_v31"
    audit, hashes = _validate_stage_dir(root, fingerprint, {"v31a_core.npy", "audit.json"})
    return {
        "partition_id": partition_id,
        "global_core_window": audit["global_core_window"],
        "global_expanded_window": audit["global_expanded_window"],
        "core_transform": audit["core_transform"], "crs": audit["crs"],
        "physical_metrics": audit["physical_metrics"],
        "expanded_context_valid_pixel_count": audit["coverage"]["expanded_context_valid_pixel_count"],
        "owner_core_decoder_valid_pixel_count": v3_entry["owner_core_decoder_valid_pixel_count"],
        "core_strict_valid_pixel_count": audit["coverage"]["core_strict_valid_pixel_count"],
        "valid_pixel_count": audit["coverage"]["core_strict_valid_pixel_count"],
        "owner_core_pixel_count": v3_entry["owner_core_pixel_count"],
        "discarded_nonowner_v3_change_pixel_count": v3_entry["discarded_nonowner_v3_change_pixel_count"],
        "v31a_outside_core_owner_rejection_count": audit.get("v31a_outside_core_owner_rejection_count", 0),
        "outputs": {
            "raw": v3_entry["outputs"]["raw"], "v3": v3_entry["outputs"]["v3"],
            "valid": v3_entry["outputs"]["valid"],
            "v31a": {**audit["outputs"]["v31a"], "path": f"partitions/{partition_id}/stage_v31/v31a_core.npy"},
        },
        "stage_v3_audit": v3_entry["audit"],
        "audit": {"path": f"partitions/{partition_id}/stage_v31/audit.json", "sha256": hashes["audit.json"]},
    }


def _base_manifest(
    snapshot: dict[str, Any], selected: list[dict[str, Any]], context_pixels: int,
    code_sha: dict[str, str], fingerprint_payload: dict[str, Any],
    physical_metrics: dict[str, float], self_test: bool,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "v31a_full_partition_core_comparison",
        "status": "preflight_complete", "self_test": self_test,
        "snapshot_manifest": snapshot["path"],
        "snapshot_manifest_sha256": snapshot["sha256"],
        "snapshot_schema_version": snapshot["manifest"].get("schema_version"),
        "execution_fingerprint": fingerprint_payload,
        "execution_fingerprint_sha256": _sha256_json(fingerprint_payload),
        "class_codes": list(CLASS_ORDER), "label_encoding": "class_codes_int16_invalid_minus_one",
        "processing_transform": snapshot["processing_transform"], "crs": snapshot["crs"],
        "physical_metrics": physical_metrics, "global_window": snapshot["global_window"],
        "physical_metrics_scope": "per_partition_expanded_window_center",
        "context_pixels": context_pixels, "code_sha256": code_sha,
        "v3_policy_snapshot": v3_policy_snapshot(),
        "v3_policy_snapshot_sha256": _sha256_json(v3_policy_snapshot()),
        "v31a_policy_snapshot": v31a_policy_snapshot(),
        "v31a_policy_snapshot_sha256": _sha256_json(v31a_policy_snapshot()),
        "requested_partition_count": len(selected), "completed_partition_count": 0,
        "stage_v3_complete": False,
        "stage_v3": {"required_partition_count": len(snapshot["partitions"]), "completed_partition_count": 0, "complete": False},
        "stage_v31": {"required_partition_count": len(selected), "completed_partition_count": 0, "complete": False},
        "stage_v3_partitions": [],
        "coverage": {
            "all_snapshot_partitions_requested": len(selected) == len(snapshot["partitions"]),
            "core_windows_nonoverlapping": True, "global_core_grid_exact": True,
            "complete": False,
        },
        "partitions": [],
    }


def run(
    snapshot_manifest: Path, output_root: Path, *, workers: int, resume: bool,
    self_test: bool, partition_ids: list[str] | None = None,
    stop_after_stage_v3: bool = False,
) -> dict[str, Any]:
    if workers < 1:
        raise FullRunError("--workers must be at least one")
    if not self_test and workers > 2:
        raise FullRunError("real execution hard-limits --workers to 2 for bounded memory")
    if not self_test and partition_ids:
        raise FullRunError("--partitions is forbidden for real runs; real execution is exactly all 140")
    snapshot = _load_snapshot(snapshot_manifest.resolve(), self_test=self_test)
    context_pixels = int(snapshot["manifest"].get("self_test_context_pixels", REAL_CONTEXT_PIXELS)) if self_test else REAL_CONTEXT_PIXELS
    if context_pixels <= 0 or (not self_test and context_pixels != REAL_CONTEXT_PIXELS):
        raise FullRunError("context contract is invalid")
    by_id = {entry["partition_id"]: entry for entry in snapshot["partitions"]}
    if partition_ids:
        unknown = sorted(set(partition_ids) - set(by_id))
        if unknown:
            raise FullRunError(f"unknown --partitions: {unknown}")
        selected = [by_id[partition_id] for partition_id in partition_ids]
    else:
        selected = list(snapshot["partitions"])
    if not self_test and len(selected) != REAL_PARTITION_COUNT:
        raise FullRunError("real execution did not select all 140 Partitions")
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not resume:
        raise FullRunError(f"refusing non-empty output root without --resume: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    unexpected = []
    if resume:
        unexpected = [item.name for item in output_root.iterdir() if item.name not in {"run_manifest.json", "partitions"}]
    if unexpected:
        raise FullRunError(f"resume output root contains unmanaged entries: {sorted(unexpected)}")
    print(
        f"preflight: verifying {len(snapshot['partitions'])} archived "
        "probability/decoder-valid/strict-valid SHA triples",
        flush=True,
    )
    verified_sources = _verify_all_sources(snapshot["partitions"], workers)
    print("preflight: validating every overlapping probability halo pair once", flush=True)
    duplicate_validation = _compare_duplicate_probabilities(snapshot["partitions"])
    code_sha = _code_sha256()
    metrics = _physical_metrics(snapshot["processing_transform"], snapshot["crs"], snapshot["global_window"])
    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION, "snapshot_manifest_sha256": snapshot["sha256"],
        "selected_partition_ids": [entry["partition_id"] for entry in selected],
        "context_pixels": context_pixels, "class_codes": list(CLASS_ORDER),
        "code_sha256": code_sha,
        "v3_policy_snapshot_sha256": _sha256_json(v3_policy_snapshot()),
        "v31a_policy_snapshot_sha256": _sha256_json(v31a_policy_snapshot()),
        "source_sha256": verified_sources,
        "duplicate_probability_validation_sha256": duplicate_validation["audit_sha256"],
    }
    run_manifest = _base_manifest(
        snapshot, selected, context_pixels, code_sha, fingerprint_payload, metrics, self_test,
    )
    run_manifest["duplicate_probability_validation"] = duplicate_validation
    expanded_shapes = [
        _shape(_expand(entry["core_window"], snapshot["global_window"], context_pixels))
        for entry in snapshot["partitions"]
    ]
    max_height, max_width = max(expanded_shapes, key=lambda value: value[0] * value[1])
    conservative_bytes_per_pixel = len(CLASS_ORDER) * 4 + 256
    run_manifest["resource_plan"] = {
        "workers": workers, "real_workers_hard_limit": 2,
        "maximum_expanded_probability_shape": [len(CLASS_ORDER), max_height, max_width],
        "maximum_expanded_pixel_count": max_height * max_width,
        "conservative_bytes_per_pixel_per_worker": conservative_bytes_per_pixel,
        "conservative_peak_bytes_per_worker": max_height * max_width * conservative_bytes_per_pixel,
        "estimate_includes": "float32 probability cube plus conservative scipy/topology working arrays",
    }
    fingerprint = run_manifest["execution_fingerprint_sha256"]
    manifest_path = output_root / "run_manifest.json"
    prior_manifest: dict[str, Any] | None = None
    if resume and manifest_path.is_file():
        prior_manifest = _read_json(manifest_path)
        if prior_manifest.get("execution_fingerprint_sha256") != fingerprint:
            raise FullRunError("resume run_manifest execution fingerprint differs")
    elif resume and any(output_root.iterdir()):
        raise FullRunError("non-empty resume output lacks run_manifest.json")
    elif not resume:
        _atomic_json(manifest_path, run_manifest)
    v3_completed: dict[str, dict[str, Any]] = {}
    v31_completed: dict[str, dict[str, Any]] = {}
    partition_root = output_root / "partitions"
    if partition_root.exists():
        expected_ids = {entry["partition_id"] for entry in snapshot["partitions"]}
        unexpected_partitions = [item.name for item in partition_root.iterdir() if item.name not in expected_ids]
        if unexpected_partitions:
            raise FullRunError(f"output contains unexpected Partitions: {sorted(unexpected_partitions)}")
        for item in partition_root.iterdir():
            children = {child.name for child in item.iterdir()}
            if not children <= {"stage_v3", "stage_v31"}:
                raise FullRunError(f"Partition output contains unmanaged stages: {item}")
            if (item / "stage_v3").exists():
                v3_completed[item.name] = _validate_v3_stage(output_root, item.name, fingerprint)
    if any((partition_root / entry["partition_id"] / "stage_v31").exists() for entry in snapshot["partitions"]) and len(v3_completed) != len(snapshot["partitions"]):
        raise FullRunError("V3.1 stage exists before the complete V3 owner barrier")
    if len(v3_completed) == len(snapshot["partitions"]):
        for entry in selected:
            stage_dir = partition_root / entry["partition_id"] / "stage_v31"
            if stage_dir.exists():
                v31_completed[entry["partition_id"]] = _validate_v31_stage(
                    output_root, entry["partition_id"], fingerprint, v3_completed[entry["partition_id"]],
                )
    run_manifest["stage_v3_partitions"] = [v3_completed[key] for key in sorted(v3_completed)]
    run_manifest["stage_v3"] = {
        "required_partition_count": len(snapshot["partitions"]),
        "completed_partition_count": len(v3_completed),
        "complete": len(v3_completed) == len(snapshot["partitions"]),
    }
    run_manifest["stage_v3_complete"] = run_manifest["stage_v3"]["complete"]
    run_manifest["partitions"] = [v31_completed[key] for key in sorted(v31_completed)]
    run_manifest["completed_partition_count"] = len(v31_completed)
    if resume:
        _atomic_json(manifest_path, run_manifest)
    staging_root = output_root.parent / f".{output_root.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    staging_root.mkdir(parents=False, exist_ok=False)
    common = {
        "entries": snapshot["partitions"], "global_window": snapshot["global_window"],
        "context_pixels": context_pixels,
        "processing_transform": snapshot["processing_transform"], "crs": snapshot["crs"],
        "snapshot_manifest": snapshot["path"], "snapshot_manifest_sha256": snapshot["sha256"],
        "code_sha256": code_sha,
        "verified_sources": verified_sources, "output_root": str(output_root),
        "duplicate_probability_validation_sha256": duplicate_validation["audit_sha256"],
        "staging_root": str(staging_root), "execution_fingerprint_sha256": fingerprint,
        "v3_policy_snapshot": run_manifest["v3_policy_snapshot"],
        "v3_policy_snapshot_sha256": run_manifest["v3_policy_snapshot_sha256"],
        "v31a_policy_snapshot": run_manifest["v31a_policy_snapshot"],
        "v31a_policy_snapshot_sha256": run_manifest["v31a_policy_snapshot_sha256"],
    }

    def checkpoint() -> None:
        run_manifest["stage_v3_partitions"] = [v3_completed[key] for key in sorted(v3_completed)]
        run_manifest["stage_v3"]["completed_partition_count"] = len(v3_completed)
        run_manifest["stage_v3"]["complete"] = len(v3_completed) == len(snapshot["partitions"])
        run_manifest["stage_v3_complete"] = run_manifest["stage_v3"]["complete"]
        run_manifest["partitions"] = [v31_completed[key] for key in sorted(v31_completed)]
        run_manifest["stage_v31"]["completed_partition_count"] = len(v31_completed)
        run_manifest["completed_partition_count"] = len(v31_completed)
        _atomic_json(manifest_path, run_manifest)

    def execute(function: Any, pending_entries: list[dict[str, Any]], completed: dict[str, dict[str, Any]], label: str, extra: dict[str, Any] | None = None) -> None:
        payload = {**common, **(extra or {})}
        if workers == 1:
            for entry in pending_entries:
                result = function({**payload, "partition_id": entry["partition_id"]})
                completed[result["partition_id"]] = result
                print(f"{label} {len(completed)}/{len(pending_entries) + len(completed) - 1} {result['partition_id']}", flush=True)
                checkpoint()
            return
        executor_class = ThreadPoolExecutor if sys.platform == "darwin" else ProcessPoolExecutor
        with executor_class(max_workers=workers) as executor:
            futures = {
                executor.submit(function, {**payload, "partition_id": entry["partition_id"]}): entry["partition_id"]
                for entry in pending_entries
            }
            for future in as_completed(futures):
                result = future.result()
                completed[result["partition_id"]] = result
                print(f"{label} {len(completed)} {result['partition_id']}", flush=True)
                checkpoint()

    try:
        run_manifest["status"] = "stage_v3_running"
        checkpoint()
        pending_v3 = [entry for entry in snapshot["partitions"] if entry["partition_id"] not in v3_completed]
        execute(_run_v3_partition, pending_v3, v3_completed, "stage_v3")
        if len(v3_completed) != len(snapshot["partitions"]):
            raise FullRunError("V3 owner stage did not reach its complete barrier")
        run_manifest["stage_v3_complete"] = True
        run_manifest["stage_v3"]["complete"] = True
        run_manifest["status"] = "stage_v3_complete"
        checkpoint()
        if stop_after_stage_v3:
            raise FullRunError("self-test injected interruption after stage_v3_complete")
        run_manifest["status"] = "stage_v31_running"
        checkpoint()
        pending_v31 = [entry for entry in selected if entry["partition_id"] not in v31_completed]
        execute(
            _run_v31_partition, pending_v31, v31_completed, "stage_v31",
            {"v3_stages": v3_completed},
        )
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    if len(v31_completed) != len(selected):
        raise FullRunError(f"V3.1 run incomplete: {len(v31_completed)}/{len(selected)}")
    run_manifest["partitions"] = [v31_completed[key] for key in sorted(v31_completed)]
    run_manifest["completed_partition_count"] = len(v31_completed)
    run_manifest["stage_v31"] = {
        "required_partition_count": len(selected), "completed_partition_count": len(v31_completed), "complete": True,
    }
    run_manifest["status"] = "complete"
    all_requested = len(selected) == len(snapshot["partitions"])
    run_manifest["coverage"] = {
        "all_snapshot_partitions_requested": all_requested,
        "core_windows_nonoverlapping": True, "global_core_grid_exact": True,
        "complete": True, "partial_self_test": bool(self_test and not all_requested),
        "published_core_pixel_count": int(sum(item["owner_core_pixel_count"] for item in v31_completed.values())),
        "published_valid_pixel_count": int(sum(item["valid_pixel_count"] for item in v31_completed.values())),
        "expanded_context_valid_pixel_count_sum_by_partition": int(sum(item["expanded_context_valid_pixel_count"] for item in v31_completed.values())),
        "v31a_outside_core_owner_rejection_count": int(sum(item["v31a_outside_core_owner_rejection_count"] for item in v31_completed.values())),
        "duplicate_probability_overlap_pair_count": duplicate_validation["overlap_pair_count"],
        "duplicate_probability_overlap_pixel_count": duplicate_validation["overlap_pixel_count"],
    }
    run_manifest["manifest_sha256"] = _sha256_json(run_manifest)
    _atomic_json(manifest_path, run_manifest)
    return run_manifest


def _write_stage_manifest(
    partition_root: Path, stage: str,
    artifacts: Mapping[str, tuple[str, np.ndarray]],
) -> dict[str, Any]:
    items: dict[str, Any] = {}
    for name, (relative, array) in artifacts.items():
        path = partition_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array, allow_pickle=False)
        items[name] = {
            "path": relative, "kind": "npy", "dtype": str(array.dtype),
            "shape": list(array.shape), "byte_count": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    body = {"schema_version": 1, "kind": "spatial_joint_stage", "stage": stage, "artifacts": items, "aliases": {}}
    body["manifest_sha256"] = _sha256_json(body)
    stage_path = partition_root / "manifests" / f"{stage}.json"
    _atomic_json(stage_path, body)
    return {
        "path": str(stage_path.relative_to(partition_root)), "sha256": _sha256_file(stage_path),
        "byte_count": stage_path.stat().st_size,
    }


def _write_self_test_fixture(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    height, width = 44, 48
    probabilities = np.full((len(CLASS_ORDER), height, width), 0.0, dtype=np.float32)
    probabilities[1] = 0.91
    probabilities[0] = 0.09
    # Decoder-valid but strict-excluded source-class support.  Together with
    # the one-pixel island at (5,5), this makes V3's 8% source budget pass only
    # when its denominator is the owner decoder-valid Core, never strict Core.
    probabilities[:, 0, 0:16] = 0.0
    probabilities[2, 0, 0:16] = 0.90
    probabilities[1, 0, 0:16] = 0.10
    probabilities[:, 5, 5] = 0.0
    probabilities[2, 5, 5] = 0.55
    probabilities[1, 5, 5] = 0.45
    # A water bridge footprint crosses the x=24 owner boundary.  V3 protects
    # water sources/targets; V3.1 must generate then reject the cross-Core
    # proposal as outside_core_owner rather than partially publishing it.
    probabilities[:, 8:16, 21:27] = 0.0
    probabilities[1, 8:16, 21:27] = 0.41
    probabilities[0, 8:16, 21:27] = 0.40
    probabilities[2, 8:16, 21:27] = 0.19
    probabilities[:, 8:16, 21:23] = 0.0
    probabilities[0, 8:16, 21:23] = 0.90
    probabilities[1, 8:16, 21:23] = 0.10
    probabilities[:, 8:16, 25:27] = 0.0
    probabilities[0, 8:16, 25:27] = 0.90
    probabilities[1, 8:16, 25:27] = 0.10
    # This one-pixel class-21 island belongs to the right owner but lies in the
    # left target's V3 halo.  A target-local V3 changes it with zero budget
    # charge; the right owner correctly retains it because its own denominator
    # is one pixel.  Stage V3 must discard the local halo rewrite and Stage V31
    # must read the right owner's retained publication.
    probabilities[:, 17, 26] = 0.0
    probabilities[2, 17, 26] = 0.55
    probabilities[1, 17, 26] = 0.45
    global_decoder_valid = np.ones((height, width), dtype=bool)
    global_decoder_valid[-1, -1] = False
    global_strict_valid = global_decoder_valid.copy()
    # These six pixels are outside the publish range but remain valid decoder
    # context.  The self-test requires algorithms to see them and outputs to
    # mask them as invalid.
    global_strict_valid[0, 0:16] = False
    x_spans, y_spans = ((0, 24), (24, 48)), ((0, 22), (22, 44))
    partitions = []
    for row, (y0, y1) in enumerate(y_spans):
        for col, (x0, x1) in enumerate(x_spans):
            partition_id = f"partition_{row:05d}_{col:05d}"
            core = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
            halo = {"x0": max(0, x0 - 4), "y0": max(0, y0 - 4), "x1": min(width, x1 + 4), "y1": min(height, y1 + 4)}
            partition_root = root / "partitions" / partition_id
            input_ref = _write_stage_manifest(
                partition_root, "input", {
                    PROBABILITY_ARTIFACT: (
                        "input/blended_probabilities_f32.npy",
                        probabilities[:, halo["y0"]:halo["y1"], halo["x0"]:halo["x1"]],
                    ),
                    DECODER_VALID_ARTIFACT: (
                        "input/decoder_valid.npy",
                        global_decoder_valid[halo["y0"]:halo["y1"], halo["x0"]:halo["x1"]],
                    ),
                },
            )
            final_ref = _write_stage_manifest(
                partition_root, "final", {
                    STRICT_VALID_ARTIFACT: (
                        "final/core_strict_range_valid.npy",
                        global_strict_valid[y0:y1, x0:x1],
                    ),
                },
            )
            partition_manifest = {
                "schema_version": 1, "kind": "spatial_joint_archive",
                "stages": {"input": input_ref, "final": final_ref},
            }
            partition_manifest["manifest_sha256"] = _sha256_json(partition_manifest)
            _atomic_json(partition_root / "manifest.json", partition_manifest)
            partitions.append({
                "partition_id": partition_id, "row": row, "col": col,
                "core_window": core, "halo_window": halo,
            })
    manifest = {
        "schema_version": 2, "experiment_id": "v31a_full_runner_self_test",
        "source_raster_crs": "EPSG:3857",
        "processing_transform": [1.0, 0.0, 12300000.0, 0.0, -1.0, 4540000.0],
        "self_test_context_pixels": 8, "partitions": partitions,
    }
    path = root / "snapshot_manifest.json"
    _atomic_json(path, manifest)
    return path


def _refresh_self_test_input_archive(partition_root: Path) -> None:
    """Refresh declared hashes after an intentional duplicate-content mutation."""

    stage_path = partition_root / "manifests" / "input.json"
    stage = _read_json(stage_path)
    for item in (stage.get("artifacts") or {}).values():
        path = partition_root / item["path"]
        item["sha256"] = _sha256_file(path)
        item["byte_count"] = path.stat().st_size
    stage.pop("manifest_sha256", None)
    stage["manifest_sha256"] = _sha256_json(stage)
    _atomic_json(stage_path, stage)
    archive_path = partition_root / "manifest.json"
    archive = _read_json(archive_path)
    archive["stages"]["input"]["sha256"] = _sha256_file(stage_path)
    archive["stages"]["input"]["byte_count"] = stage_path.stat().st_size
    archive.pop("manifest_sha256", None)
    archive["manifest_sha256"] = _sha256_json(archive)
    _atomic_json(archive_path, archive)


def _assert_preflight_rejections() -> None:
    with tempfile.TemporaryDirectory(prefix="v31a-full-selftest-bad-sha-") as temporary:
        root = Path(temporary)
        snapshot = _write_self_test_fixture(root)
        path = root / "partitions" / "partition_00000_00000" / "input" / "blended_probabilities_f32.npy"
        values = np.load(path, allow_pickle=False)
        values[0, 1, 1] += np.float32(0.001)
        np.save(path, values, allow_pickle=False)
        try:
            run(snapshot, root / "out", workers=1, resume=False, self_test=True)
        except FullRunError as exc:
            if "probability SHA-256 mismatch" not in str(exc):
                raise
        else:
            raise FullRunError("self-test accepted a source probability SHA mismatch")
    with tempfile.TemporaryDirectory(prefix="v31a-full-selftest-bad-overlap-") as temporary:
        root = Path(temporary)
        snapshot = _write_self_test_fixture(root)
        partition_root = root / "partitions" / "partition_00000_00000"
        path = partition_root / "input" / "blended_probabilities_f32.npy"
        values = np.load(path, allow_pickle=False)
        values[0, 10, 20] += np.float32(0.01)
        values[1, 10, 20] -= np.float32(0.01)
        np.save(path, values, allow_pickle=False)
        _refresh_self_test_input_archive(partition_root)
        try:
            run(snapshot, root / "out", workers=1, resume=False, self_test=True)
        except FullRunError as exc:
            if "duplicate source probabilities disagree" not in str(exc):
                raise
        else:
            raise FullRunError("self-test accepted inconsistent duplicate halo probabilities")


def _self_test(output_root: Path | None, workers: int, resume: bool, partitions: list[str] | None) -> dict[str, Any]:
    if "class_budget_mask" not in inspect.signature(apply_v31a_candidate).parameters:
        raise FullRunError("V3.1 candidate does not expose required class_budget_mask")
    with tempfile.TemporaryDirectory(prefix="v31a-full-selftest-input-") as temporary:
        snapshot = _write_self_test_fixture(Path(temporary))
        _assert_preflight_rejections()
        with tempfile.TemporaryDirectory(prefix="v31a-full-selftest-interrupted-") as interrupted:
            interrupted_root = Path(interrupted)
            try:
                run(
                    snapshot, interrupted_root, workers=workers, resume=False,
                    self_test=True, partition_ids=partitions, stop_after_stage_v3=True,
                )
            except FullRunError as exc:
                if "injected interruption after stage_v3_complete" not in str(exc):
                    raise
            else:
                raise FullRunError("self-test Stage V3 interruption was not injected")
            interrupted_manifest = _read_json(interrupted_root / "run_manifest.json")
            if not interrupted_manifest.get("stage_v3_complete") or any(interrupted_root.glob("partitions/*/stage_v31")):
                raise FullRunError("self-test violated the V3 completion barrier during interruption")
            extra = interrupted_root / "partitions" / "partition_00000_00000" / "stage_v3" / "unexpected.bin"
            extra.write_bytes(b"must be rejected")
            try:
                run(
                    snapshot, interrupted_root, workers=workers, resume=True,
                    self_test=True, partition_ids=partitions,
                )
            except FullRunError as exc:
                if "missing or extra files" not in str(exc):
                    raise
            else:
                raise FullRunError("self-test resume accepted an extra fixed-stage output")
            extra.unlink()
            resumed = run(
                snapshot, interrupted_root, workers=workers, resume=True,
                self_test=True, partition_ids=partitions,
            )
            _assert_self_test(interrupted_root, resumed, snapshot)
        if output_root is None:
            with tempfile.TemporaryDirectory(prefix="v31a-full-selftest-output-") as output:
                result = run(snapshot, Path(output), workers=workers, resume=False, self_test=True, partition_ids=partitions)
                _assert_self_test(Path(output), result, snapshot)
                return result
        result = run(snapshot, output_root, workers=workers, resume=resume, self_test=True, partition_ids=partitions)
        _assert_self_test(output_root, result, snapshot)
        return result


def _assert_self_test(output_root: Path, manifest: dict[str, Any], snapshot_path: Path) -> None:
    if manifest.get("status") != "complete" or manifest["completed_partition_count"] != manifest["requested_partition_count"]:
        raise FullRunError("self-test did not complete every requested synthetic Core")
    if not manifest.get("stage_v3_complete") or not manifest["stage_v3"].get("complete"):
        raise FullRunError("self-test crossed into V3.1 before the V3 owner barrier")
    duplicate = manifest.get("duplicate_probability_validation") or {}
    pairs = duplicate.get("pairs") or []
    unique_pairs = {
        (item["left_partition_id"], item["right_partition_id"])
        for item in pairs
    }
    if (
        not duplicate.get("consistent")
        or duplicate.get("scope") != "global_each_overlapping_halo_pair_exactly_once"
        or int(duplicate.get("overlap_pair_count", -1)) != len(pairs)
        or len(unique_pairs) != len(pairs)
        or int(duplicate.get("overlap_pixel_count", 0)) <= 0
    ):
        raise FullRunError("self-test global duplicate-pair preflight is incomplete or repeated")
    boundary_rejected = 0
    saw_decoder_context_outside_strict = False
    for entry in manifest["partitions"]:
        raw = np.load(output_root / entry["outputs"]["raw"]["path"], allow_pickle=False)
        v3 = np.load(output_root / entry["outputs"]["v3"]["path"], allow_pickle=False)
        v31a = np.load(output_root / entry["outputs"]["v31a"]["path"], allow_pickle=False)
        valid = np.load(output_root / entry["outputs"]["valid"]["path"], allow_pickle=False)
        expected_shape = _shape(entry["global_core_window"])
        if raw.shape != expected_shape or v3.shape != expected_shape or v31a.shape != expected_shape or valid.shape != expected_shape:
            raise FullRunError(f"self-test output shape mismatch: {entry['partition_id']}")
        if np.any(raw[~valid] != -1) or np.any(v3[~valid] != -1) or np.any(v31a[~valid] != -1):
            raise FullRunError(f"self-test invalid pixels not preserved: {entry['partition_id']}")
        if np.any(~np.isin(raw[valid], CLASS_ORDER)) or np.any(~np.isin(v3[valid], CLASS_ORDER)) or np.any(~np.isin(v31a[valid], CLASS_ORDER)):
            raise FullRunError(f"self-test output is not single-class coded: {entry['partition_id']}")
        audit = _read_json(output_root / entry["audit"]["path"])
        if not audit["v31a_audit"].get("full_audit") or audit["v31a_audit"].get("audit_truncated"):
            raise FullRunError("self-test V3.1 audit is not complete")
        coverage = audit["coverage"]
        if coverage["expanded_context_valid_pixel_count"] > coverage["core_strict_valid_pixel_count"]:
            saw_decoder_context_outside_strict = True
        boundary_rejected += int(audit.get("v31a_outside_core_owner_rejection_count", 0))
    if not saw_decoder_context_outside_strict:
        raise FullRunError("self-test did not exercise decoder-valid context outside strict publish range")
    selected_ids = {entry["partition_id"] for entry in manifest["partitions"]}
    if {"partition_00000_00000", "partition_00000_00001"}.issubset(selected_ids) and boundary_rejected <= 0:
        raise FullRunError("self-test did not exercise outside_core_owner proposal rejection")
    v3_stages = {entry["partition_id"]: entry for entry in manifest["stage_v3_partitions"]}
    left = v3_stages["partition_00000_00000"]
    left_raw = np.load(output_root / left["outputs"]["raw"]["path"], allow_pickle=False)
    left_v3 = np.load(output_root / left["outputs"]["v3"]["path"], allow_pickle=False)
    left_valid = np.load(output_root / left["outputs"]["valid"]["path"], allow_pickle=False)
    left_audit = _read_json(output_root / left["audit"]["path"])
    if left_raw[5, 5] != 21 or left_v3[5, 5] != 13:
        raise FullRunError("self-test did not prove decoder-Core V3 budget changes the strict island")
    if np.any(left_valid[0, 0:16]):
        raise FullRunError("self-test strict-excluded decoder-valid support was published")
    if (
        left_audit["v3_audit"].get("class_budget_pixel_count")
        != left_audit["coverage"]["owner_core_decoder_valid_pixel_count"]
        or left_audit["coverage"]["owner_core_decoder_valid_pixel_count"]
        <= left_audit["coverage"]["core_strict_valid_pixel_count"]
    ):
        raise FullRunError("self-test V3 budget denominator is not owner decoder-valid Core")
    if left_audit.get("discarded_nonowner_v3_change_pixel_count", 0) <= 0:
        raise FullRunError("self-test did not create a discarded target-local halo V3 rewrite")
    right_context = np.load(
        output_root / v3_stages["partition_00000_00001"]["outputs"]["v3_context"]["path"],
        allow_pickle=False,
    )
    if right_context[17, 2] != 2:
        raise FullRunError("right owner did not retain the counterexample class-21 pixel")
    snapshot = _load_snapshot(snapshot_path, self_test=True)
    entries = snapshot["partitions"]
    targets = {entry["partition_id"]: entry for entry in entries}
    left_stitched = _stitch_v31_context(
        targets["partition_00000_00000"], entries, snapshot["global_window"], 8,
        output_root, v3_stages,
    )
    right_stitched = _stitch_v31_context(
        targets["partition_00000_00001"], entries, snapshot["global_window"], 8,
        output_root, v3_stages,
    )
    shared = _intersection(left_stitched[5], right_stitched[5])
    assert shared is not None
    if not np.array_equal(
        left_stitched[0][_local_slices(left_stitched[5], shared)],
        right_stitched[0][_local_slices(right_stitched[5], shared)],
    ):
        raise FullRunError("neighboring V3.1 contexts do not share the same owner V3 baseline")


def _validate_only(snapshot_manifest: Path, workers: int) -> dict[str, Any]:
    if workers < 1 or workers > 2:
        raise FullRunError("real --validate-only requires 1 or 2 workers")
    snapshot = _load_snapshot(snapshot_manifest.resolve(), self_test=False)
    verified = _verify_all_sources(snapshot["partitions"], workers)
    duplicate = _compare_duplicate_probabilities(snapshot["partitions"])
    shapes = [
        _shape(_expand(entry["core_window"], snapshot["global_window"], REAL_CONTEXT_PIXELS))
        for entry in snapshot["partitions"]
    ]
    height, width = max(shapes, key=lambda value: value[0] * value[1])
    return {
        "schema_valid": True, "effect_evaluation_performed": False,
        "partition_count": len(snapshot["partitions"]),
        "snapshot_manifest_sha256": snapshot["sha256"],
        "verified_source_partition_count": len(verified),
        "duplicate_probability_validation": duplicate,
        "global_window": snapshot["global_window"],
        "maximum_expanded_probability_shape": [len(CLASS_ORDER), height, width],
        "conservative_peak_bytes_per_worker": height * width * (len(CLASS_ORDER) * 4 + 256),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--partitions", nargs="+")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        if args.snapshot_manifest is not None or args.validate_only:
            parser.error("--self-test generates its own snapshot; do not pass --snapshot-manifest")
    elif args.snapshot_manifest is None or (not args.validate_only and args.output_root is None):
        parser.error("real execution requires --snapshot-manifest and --output-root")
    if args.validate_only and args.partitions:
        parser.error("--validate-only validates the complete real 140-Partition snapshot")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.validate_only:
            result = _validate_only(args.snapshot_manifest, args.workers)
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
        elif args.self_test:
            result = _self_test(args.output_root, args.workers, args.resume, args.partitions)
            print(
                f"SELF-TEST PASS: {result['completed_partition_count']}/"
                f"{result['requested_partition_count']} synthetic Partitions; "
                f"outside_core_owner={result['coverage']['v31a_outside_core_owner_rejection_count']}",
                flush=True,
            )
        else:
            result = run(
                args.snapshot_manifest, args.output_root, workers=args.workers,
                resume=args.resume, self_test=False, partition_ids=args.partitions,
            )
            print(
                f"FULL RUN COMPLETE: {result['completed_partition_count']}/"
                f"{REAL_PARTITION_COUNT}; manifest={args.output_root / 'run_manifest.json'}",
                flush=True,
            )
        return 0
    except FullRunError as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
