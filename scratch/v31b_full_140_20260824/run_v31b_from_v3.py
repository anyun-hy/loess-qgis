#!/usr/bin/env python3
"""Run V3.1-B from a complete, immutable V3.1-A Stage-V3 publication.

This intentionally does *not* execute V3.  The input is the completed 140-Core
V3.1-A ``run_manifest.json`` plus its archived probability snapshot.  For each
strict owner Core, the runner builds a 256-pixel context from the already
published owner ``v3_context`` arrays, calls the separately implemented B API,
and publishes only that owner's strict Core.

The output keeps the V3.1-A evaluator schema (``raw``, ``v3``, ``v31a``, and
``valid``) because its ``v31a`` slot means "candidate under evaluation".  The
manifest and every audit explicitly label this result as candidate ``B``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping
import uuid

import numpy as np
from rasterio.transform import Affine


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "inference_scripts"))

from deployment_config import CLASS_ORDER  # noqa: E402
from fragmentation_v3 import policy_snapshot as v3_policy_snapshot  # noqa: E402
from small_component_regularizer import physical_pixel_area_m2  # noqa: E402


SCHEMA_VERSION = 1
REAL_PARTITION_COUNT = 140
CONTEXT_PIXELS = 256
PROBABILITY_ARTIFACT = "input_blended_probabilities_f32_npy"
DECODER_VALID_ARTIFACT = "input_decoder_valid_npy"


class V31BRunError(RuntimeError):
    """An integrity or execution contract failure that must stop the run."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".npy", delete=False) as handle:
        temporary = Path(handle.name)
        np.save(handle, np.ascontiguousarray(values), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V31BRunError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V31BRunError(f"JSON root must be an object: {path}")
    return value


def _resolve(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _window(value: Any, name: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise V31BRunError(f"{name} must be an x0/y0/x1/y1 object")
    try:
        result = {key: int(value[key]) for key in ("x0", "y0", "x1", "y1")}
    except (KeyError, TypeError, ValueError) as exc:
        raise V31BRunError(f"{name} must contain integer x0/y0/x1/y1") from exc
    if result["x0"] >= result["x1"] or result["y0"] >= result["y1"]:
        raise V31BRunError(f"{name} is empty or reversed")
    return result


def _shape(window: Mapping[str, int]) -> tuple[int, int]:
    return window["y1"] - window["y0"], window["x1"] - window["x0"]


def _intersect(left: Mapping[str, int], right: Mapping[str, int]) -> dict[str, int] | None:
    result = {"x0": max(left["x0"], right["x0"]), "y0": max(left["y0"], right["y0"]), "x1": min(left["x1"], right["x1"]), "y1": min(left["y1"], right["y1"])}
    return result if result["x0"] < result["x1"] and result["y0"] < result["y1"] else None


def _slices(container: Mapping[str, int], selected: Mapping[str, int]) -> tuple[slice, slice]:
    return slice(selected["y0"] - container["y0"], selected["y1"] - container["y0"]), slice(selected["x0"] - container["x0"], selected["x1"] - container["x0"])


def _expand(core: Mapping[str, int], global_window: Mapping[str, int]) -> dict[str, int]:
    return {"x0": max(global_window["x0"], core["x0"] - CONTEXT_PIXELS), "y0": max(global_window["y0"], core["y0"] - CONTEXT_PIXELS), "x1": min(global_window["x1"], core["x1"] + CONTEXT_PIXELS), "y1": min(global_window["y1"], core["y1"] + CONTEXT_PIXELS)}


def _code_sha256(api_module: Any | None = None, apply: Any | None = None) -> dict[str, str]:
    paths = [
        REPO_ROOT / "inference_scripts" / "deployment_config.py",
        REPO_ROOT / "inference_scripts" / "fragmentation_v3.py",
        REPO_ROOT / "inference_scripts" / "small_component_regularizer.py",
        Path(__file__).resolve(),
    ]
    if api_module is not None and getattr(api_module, "__file__", None):
        paths.append(Path(api_module.__file__).resolve())
    if apply is not None:
        paths.append(Path(inspect.getsourcefile(apply) or "").resolve())
    paths = list(dict.fromkeys(paths))
    return {str(path.relative_to(REPO_ROOT)): _sha256_file(path) for path in paths}


def _b_api() -> tuple[Any, Any, Any, Any]:
    """Load B by its explicit contract; never silently fall back to V3.1-A."""
    try:
        module = importlib.import_module("fragmentation_v31_candidate")
    except ImportError as exc:
        raise V31BRunError("V3.1-B API unavailable: cannot import fragmentation_v31_candidate") from exc
    missing = [name for name in ("v31b_policy", "apply_v31b_candidate", "policy_snapshot") if not callable(getattr(module, name, None))]
    if missing:
        raise V31BRunError(
            "V3.1-B API unavailable: expected fragmentation_v31_candidate."
            + ", ".join(missing)
            + "; refusing to substitute V3.1-A"
        )
    apply = module.apply_v31b_candidate
    required = {"class_budget_mask", "valid_mask", "probabilities", "baseline_kind", "full_audit"}
    absent = required - set(inspect.signature(apply).parameters)
    if absent:
        raise V31BRunError(f"V3.1-B API contract is incomplete; apply_v31b_candidate lacks {sorted(absent)}")
    policy = module.v31b_policy()
    snapshot = module.policy_snapshot(policy)
    if not isinstance(snapshot, dict):
        raise V31BRunError("V3.1-B API policy_snapshot must return an object")
    return module, apply, policy, snapshot


def _archive_artifact(snapshot_root: Path, partition_id: str, stage: str, artifact: str) -> tuple[Path, dict[str, Any]]:
    root = snapshot_root / "partitions" / partition_id
    partition = _read_json(root / "manifest.json")
    stage_ref = (partition.get("stages") or {}).get(stage)
    if not isinstance(stage_ref, dict):
        raise V31BRunError(f"{partition_id}: archived {stage} manifest is missing")
    stage_path = _resolve(str(stage_ref.get("path", "")), root)
    if not stage_path.is_file() or _sha256_file(stage_path) != stage_ref.get("sha256"):
        raise V31BRunError(f"{partition_id}: archived {stage} manifest SHA-256 mismatch")
    item = (_read_json(stage_path).get("artifacts") or {}).get(artifact)
    if not isinstance(item, dict):
        raise V31BRunError(f"{partition_id}: missing archived artifact {artifact}")
    path = _resolve(str(item.get("path", "")), root)
    if not path.is_file() or not isinstance(item.get("sha256"), str) or _sha256_file(path) != item["sha256"]:
        raise V31BRunError(f"{partition_id}: archived {artifact} SHA-256 mismatch")
    return path, item


def _validate_grid(entries: list[dict[str, Any]]) -> dict[str, int]:
    rows: dict[tuple[int, int], list[dict[str, int]]] = {}
    for entry in entries:
        rows.setdefault((entry["core_window"]["y0"], entry["core_window"]["y1"]), []).append(entry["core_window"])
    ordered = sorted(rows)
    for previous, current in zip(ordered, ordered[1:]):
        if previous[1] != current[0]:
            raise V31BRunError("V3 owner Core rows have a gap or overlap")
    expected_columns: list[tuple[int, int]] | None = None
    for row in ordered:
        spans = sorted((item["x0"], item["x1"]) for item in rows[row])
        if any(left[1] != right[0] for left, right in zip(spans, spans[1:])):
            raise V31BRunError("V3 owner Core columns have a gap or overlap")
        if expected_columns is None:
            expected_columns = spans
        elif expected_columns != spans:
            raise V31BRunError("V3 owner Core columns differ between rows")
    if not ordered or not expected_columns:
        raise V31BRunError("V3 owner Core grid is empty")
    return {"x0": expected_columns[0][0], "y0": ordered[0][0], "x1": expected_columns[-1][1], "y1": ordered[-1][1]}


def _verify_npy(path: Path, expected_sha: str, shape: tuple[int, ...], dtype: np.dtype, label: str) -> None:
    if not path.is_file() or _sha256_file(path) != expected_sha:
        raise V31BRunError(f"{label}: fixed output SHA-256 mismatch")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != shape or array.dtype != dtype:
        raise V31BRunError(f"{label}: fixed output array metadata changed")


def _load_parent(parent_path: Path, *, self_test: bool) -> dict[str, Any]:
    parent = _read_json(parent_path)
    if parent.get("kind") != "v31a_full_partition_core_comparison":
        raise V31BRunError("input must be a V3.1-A full-partition run_manifest")
    expected_count = 4 if self_test else REAL_PARTITION_COUNT
    if parent.get("status") != "complete" or parent.get("completed_partition_count") != expected_count:
        raise V31BRunError(f"V3.1-A manifest must be complete with {expected_count} Cores")
    stage = parent.get("stage_v3")
    stages = parent.get("stage_v3_partitions")
    if not isinstance(stage, dict) or not stage.get("complete") or stage.get("completed_partition_count") != expected_count or not isinstance(stages, list) or len(stages) != expected_count:
        raise V31BRunError("V3.1-A Stage-V3 owner publication is not complete")
    if parent.get("manifest_sha256") and parent["manifest_sha256"] != _sha256_json({key: value for key, value in parent.items() if key != "manifest_sha256"}):
        raise V31BRunError("V3.1-A run_manifest self SHA-256 mismatch")
    if not isinstance(parent.get("execution_fingerprint"), dict) or parent.get("execution_fingerprint_sha256") != _sha256_json(parent["execution_fingerprint"]):
        raise V31BRunError("V3.1-A execution fingerprint SHA-256 mismatch")
    root = parent_path.parent.resolve()
    source_manifest = _resolve(str(parent.get("snapshot_manifest", "")), root)
    if not source_manifest.is_file() or _sha256_file(source_manifest) != parent.get("snapshot_manifest_sha256"):
        raise V31BRunError("V3.1-A snapshot manifest SHA-256 mismatch")
    source = _read_json(source_manifest)
    raw_parts = source.get("partitions")
    if not isinstance(raw_parts, list) or len(raw_parts) != expected_count:
        raise V31BRunError("source snapshot partition count disagrees with V3.1-A manifest")
    current_v3_snapshot = v3_policy_snapshot()
    if parent.get("v3_policy_snapshot_sha256") != _sha256_json(parent.get("v3_policy_snapshot")) or parent.get("v3_policy_snapshot") != current_v3_snapshot:
        raise V31BRunError("V3.1-A V3 policy fingerprint differs from the frozen/current policy")
    code = parent.get("code_sha256")
    fingerprint = parent.get("execution_fingerprint_sha256")
    if not isinstance(code, dict) or not code or not isinstance(fingerprint, str):
        raise V31BRunError("V3.1-A code or execution fingerprint is missing")
    stage_by_id = {str(entry.get("partition_id", "")): entry for entry in stages if isinstance(entry, dict)}
    source_by_id = {str(entry.get("partition_id", "")): entry for entry in raw_parts if isinstance(entry, dict)}
    if len(stage_by_id) != expected_count or set(stage_by_id) != set(source_by_id):
        raise V31BRunError("V3.1-A Stage-V3 and source partition identities disagree")
    entries: list[dict[str, Any]] = []
    for partition_id in sorted(stage_by_id):
        inherited = stage_by_id[partition_id]
        source_part = source_by_id[partition_id]
        core = _window(inherited.get("global_core_window"), f"{partition_id}.global_core_window")
        if core != _window(source_part.get("core_window"), f"{partition_id}.source_core_window"):
            raise V31BRunError(f"{partition_id}: V3 Core window differs from source snapshot")
        audit_ref = inherited.get("audit")
        if not isinstance(audit_ref, dict):
            raise V31BRunError(f"{partition_id}: V3 audit reference missing")
        audit_path = _resolve(str(audit_ref.get("path", "")), root)
        if not audit_path.is_file() or _sha256_file(audit_path) != audit_ref.get("sha256"):
            raise V31BRunError(f"{partition_id}: V3 audit SHA-256 mismatch")
        audit = _read_json(audit_path)
        if audit.get("execution_fingerprint_sha256") != fingerprint or audit.get("code_sha256") != code:
            raise V31BRunError(f"{partition_id}: V3 code/execution fingerprint differs from parent")
        if audit.get("v3_policy_snapshot_sha256") != parent["v3_policy_snapshot_sha256"] or audit.get("v3_policy_snapshot") != parent["v3_policy_snapshot"]:
            raise V31BRunError(f"{partition_id}: V3 policy fingerprint differs from parent")
        outputs = inherited.get("outputs")
        if not isinstance(outputs, dict) or set(("raw", "v3_context", "v3", "valid")) - set(outputs):
            raise V31BRunError(f"{partition_id}: V3 fixed raw/v3_context/v3/valid outputs are missing")
        expected_shape = _shape(core)
        fixed: dict[str, dict[str, Any]] = {}
        for key, dtype in (("raw", np.dtype("int16")), ("v3_context", np.dtype("int16")), ("v3", np.dtype("int16")), ("valid", np.dtype("bool"))):
            item = outputs[key]
            if not isinstance(item, dict) or not isinstance(item.get("sha256"), str):
                raise V31BRunError(f"{partition_id}: V3 {key} fixed SHA declaration missing")
            path = _resolve(str(item.get("path", "")), root)
            _verify_npy(path, item["sha256"], expected_shape, dtype, f"{partition_id}: V3 {key}")
            audit_item = (audit.get("outputs") or {}).get(key)
            if not isinstance(audit_item, dict) or audit_item.get("sha256") != item["sha256"]:
                raise V31BRunError(f"{partition_id}: V3 {key} audit/output SHA declarations differ")
            fixed[key] = {**item, "path": str(path)}
        prob_path, prob_meta = _archive_artifact(source_manifest.parent, partition_id, "input", PROBABILITY_ARTIFACT)
        valid_path, valid_meta = _archive_artifact(source_manifest.parent, partition_id, "input", DECODER_VALID_ARTIFACT)
        halo = _window(source_part.get("halo_window"), f"{partition_id}.halo_window")
        if tuple(prob_meta.get("shape", ())) != (len(CLASS_ORDER), *_shape(halo)) or prob_meta.get("dtype") != "float32":
            raise V31BRunError(f"{partition_id}: probability metadata/window mismatch")
        if tuple(valid_meta.get("shape", ())) != _shape(halo) or valid_meta.get("dtype") != "bool":
            raise V31BRunError(f"{partition_id}: decoder-valid metadata/window mismatch")
        entries.append({"partition_id": partition_id, "core_window": core, "halo_window": halo, "outputs": fixed, "stage_v3_audit_path": str(audit_path), "stage_v3_audit_sha256": str(audit_ref["sha256"]), "probability_path": str(prob_path), "probability_sha256": prob_meta["sha256"], "decoder_valid_path": str(valid_path), "decoder_valid_sha256": valid_meta["sha256"]})
    global_window = _validate_grid(entries)
    transform = parent.get("processing_transform")
    if not isinstance(transform, list) or len(transform) != 6 or not all(math.isfinite(float(value)) for value in transform):
        raise V31BRunError("V3.1-A processing transform is invalid")
    crs = str(parent.get("crs") or "")
    if not crs:
        raise V31BRunError("V3.1-A CRS is missing")
    return {"parent": parent, "parent_path": str(parent_path.resolve()), "parent_sha256": _sha256_file(parent_path), "entries": entries, "global_window": global_window, "transform": [float(value) for value in transform], "crs": crs}


def _verify_source(entry: Mapping[str, Any]) -> dict[str, str]:
    for key in ("probability", "decoder_valid"):
        path = Path(str(entry[f"{key}_path"]))
        expected = str(entry[f"{key}_sha256"])
        if _sha256_file(path) != expected:
            raise V31BRunError(f"{entry['partition_id']}: source {key} SHA-256 mismatch")
    probabilities = np.load(entry["probability_path"], mmap_mode="r", allow_pickle=False)
    decoder_valid = np.load(entry["decoder_valid_path"], mmap_mode="r", allow_pickle=False)
    if probabilities.shape != (len(CLASS_ORDER), *_shape(entry["halo_window"])) or probabilities.dtype != np.float32 or decoder_valid.shape != _shape(entry["halo_window"]) or decoder_valid.dtype != np.bool_:
        raise V31BRunError(f"{entry['partition_id']}: source probability/decoder-valid metadata changed")
    return {"probability_sha256": str(entry["probability_sha256"]), "decoder_valid_sha256": str(entry["decoder_valid_sha256"])}


def _physical(transform: list[float], crs: str, window: Mapping[str, int]) -> dict[str, float]:
    affine = Affine(*transform) * Affine.translation(window["x0"], window["y0"])
    area = float(physical_pixel_area_m2(affine, crs, height=_shape(window)[0], width=_shape(window)[1]))
    determinant = abs(affine.a * affine.e - affine.b * affine.d)
    if determinant <= 0 or not math.isfinite(area) or area <= 0:
        raise V31BRunError("invalid physical metrics")
    scale = math.sqrt(area / determinant)
    row = math.hypot(affine.b, affine.e) * scale
    return {"pixel_area_m2": area, "row_step_m": row, "column_step_m": area / row}


def _stitch(target: Mapping[str, Any], entries: list[dict[str, Any]], global_window: Mapping[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int], list[dict[str, Any]]]:
    expanded = _expand(target["core_window"], global_window)
    height, width = _shape(expanded)
    baseline = np.full((height, width), -1, dtype=np.int16)
    probabilities = np.empty((len(CLASS_ORDER), height, width), dtype=np.float32)
    valid = np.zeros((height, width), dtype=bool)
    strict = np.zeros((height, width), dtype=bool)
    coverage = np.zeros((height, width), dtype=np.uint8)
    owners = [entry for entry in entries if _intersect(expanded, entry["core_window"]) is not None]
    for owner in owners:
        selected = _intersect(expanded, owner["core_window"])
        assert selected is not None
        destination = _slices(expanded, selected)
        owner_slice = _slices(owner["core_window"], selected)
        baseline[destination] = np.load(owner["outputs"]["v3_context"]["path"], mmap_mode="r", allow_pickle=False)[owner_slice]
        halo_slice = _slices(owner["halo_window"], selected)
        probabilities[(slice(None), *destination)] = np.load(owner["probability_path"], mmap_mode="r", allow_pickle=False)[(slice(None), *halo_slice)]
        valid[destination] = np.load(owner["decoder_valid_path"], mmap_mode="r", allow_pickle=False)[halo_slice]
        strict[destination] = np.load(owner["outputs"]["valid"]["path"], mmap_mode="r", allow_pickle=False)[owner_slice]
        coverage[destination] += 1
    if not np.all(coverage == 1):
        raise V31BRunError(f"{target['partition_id']}: owner V3 context coverage is not exact")
    if np.any(strict & ~valid) or np.any(valid & ((baseline < 0) | (baseline >= len(CLASS_ORDER)))) or np.any(~valid & (baseline != -1)):
        raise V31BRunError(f"{target['partition_id']}: V3 context/decoder-valid contract violated")
    budget = np.zeros((height, width), dtype=bool)
    budget[_slices(expanded, target["core_window"])] = True
    return baseline, probabilities, valid, strict, expanded, owners


def _validate_b_stage(root: Path, partition_id: str, fingerprint: str, shape: tuple[int, int]) -> dict[str, Any]:
    directory = root / "partitions" / partition_id / "stage_v31b"
    audit_path, hashes_path = directory / "audit.json", directory / "outputs_sha256.json"
    if not audit_path.is_file() or not hashes_path.is_file():
        raise V31BRunError(f"resume B stage incomplete: {directory}")
    audit, hashes = _read_json(audit_path), _read_json(hashes_path).get("files")
    if audit.get("execution_fingerprint_sha256") != fingerprint or not isinstance(hashes, dict) or set(hashes) != {"v31a_core.npy", "audit.json"}:
        raise V31BRunError(f"resume B stage fingerprint/output set differs: {directory}")
    if any(_sha256_file(directory / name) != value for name, value in hashes.items()):
        raise V31BRunError(f"resume B stage SHA-256 mismatch: {directory}")
    output = audit.get("outputs", {}).get("v31a")
    if not isinstance(output, dict) or output.get("sha256") != hashes["v31a_core.npy"]:
        raise V31BRunError(f"resume B stage audit/output SHA declarations differ: {directory}")
    _verify_npy(directory / "v31a_core.npy", hashes["v31a_core.npy"], shape, np.dtype("int16"), f"resume B {partition_id}")
    return audit


def _run_partition(job: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    entry = job["entry"]
    baseline, probabilities, valid, strict, expanded, owners = _stitch(entry, job["entries"], job["global_window"])
    core_slice = _slices(expanded, entry["core_window"])
    budget = np.zeros(valid.shape, dtype=bool)
    budget[core_slice] = strict[core_slice]
    metrics = _physical(job["transform"], job["crs"], expanded)
    if np.any(budget):
        try:
            result, candidate_audit = job["apply"](baseline, class_codes=CLASS_ORDER, pixel_area_m2=metrics["pixel_area_m2"], pixel_size_m=(metrics["row_step_m"], metrics["column_step_m"]), valid_mask=valid, class_budget_mask=budget, probabilities=probabilities, confidence=probabilities.max(axis=0).astype(np.float32), policy=job["policy"], baseline_kind="v3_cleaned", full_audit=True)
        except Exception as exc:
            raise V31BRunError(
                f"{entry['partition_id']}: V3.1-B API execution failed "
                f"({type(exc).__name__}: {exc})"
            ) from exc
    else:
        result, candidate_audit = baseline.copy(), {"skipped": True, "reason": "empty_owner_core_strict_valid", "raw_generated": 0, "proposals_generated": 0, "proposals_canonical": 0, "duplicate_proposal_count": 0, "proposals_accepted": 0, "proposal_reject_reason_counts": {}, "proposal_generation_reject_reason_counts": {}, "final_topology_rollback": 0, "full_audit": True, "audit_truncated": False}
    if not candidate_audit.get("full_audit") or candidate_audit.get("audit_truncated"):
        raise V31BRunError(f"{entry['partition_id']}: B candidate did not return a complete audit")
    core_valid = strict[core_slice]
    output = np.full(core_valid.shape, -1, dtype=np.int16)
    codes = np.asarray(CLASS_ORDER, dtype=np.int16)
    indices = result[core_slice]
    if np.any(indices[core_valid] < 0) or np.any(indices[core_valid] >= len(CLASS_ORDER)):
        raise V31BRunError(f"{entry['partition_id']}: B candidate produced invalid label indices")
    output[core_valid] = codes[indices[core_valid]]
    directory = Path(job["output_root"]) / "partitions" / entry["partition_id"] / "stage_v31b"
    staging = Path(job["staging_root"]) / entry["partition_id"]
    if directory.exists():
        raise V31BRunError(f"refusing to overwrite existing B stage: {directory}")
    staging.mkdir(parents=True, exist_ok=False)
    _save_npy(staging / "v31a_core.npy", output)
    hashes = {"v31a_core.npy": _sha256_file(staging / "v31a_core.npy")}
    raw_generated = int(candidate_audit.get("raw_generated", candidate_audit.get("proposals_generated", 0)))
    canonical_generated = int(candidate_audit.get("proposals_canonical", raw_generated))
    duplicate_count = int(candidate_audit.get("duplicate_proposal_count", raw_generated - canonical_generated))
    accepted_count = int(candidate_audit.get("proposals_accepted", 0))
    if min(raw_generated, canonical_generated, duplicate_count, accepted_count) < 0 or raw_generated != canonical_generated + duplicate_count or accepted_count > canonical_generated:
        raise V31BRunError(f"{entry['partition_id']}: B proposal-count closure is invalid")
    generation_events = {str(key): int(value) for key, value in (candidate_audit.get("proposal_generation_reject_reason_counts") or {}).items()}
    if any(value < 0 for value in generation_events.values()):
        raise V31BRunError(f"{entry['partition_id']}: B generation-rejection events are invalid")
    proposal_counts = {
        "raw_generated": raw_generated,
        "canonical_generated": canonical_generated,
        "duplicate_proposal_count": duplicate_count,
        "canonical_accepted": accepted_count,
        "canonical_rejected": canonical_generated - accepted_count,
        "rollback": int(candidate_audit.get("final_topology_rollback", 0)),
        "closure": "raw_generated=canonical_generated+duplicate_proposal_count;canonical_generated=canonical_accepted+canonical_rejected",
    }
    generation_rejection_events = {"count": int(sum(generation_events.values())), "by_reason": dict(sorted(generation_events.items()))}
    audit = {"schema_version": SCHEMA_VERSION, "stage": "v31b", "candidate_label": "B", "partition_id": entry["partition_id"], "execution_fingerprint_sha256": job["fingerprint"], "parent_v31a_manifest": job["parent_path"], "parent_v31a_manifest_sha256": job["parent_sha256"], "class_codes": list(CLASS_ORDER), "label_encoding": "class_codes_int16_invalid_minus_one", "global_core_window": entry["core_window"], "global_expanded_window": expanded, "context_pixels": CONTEXT_PIXELS, "processing_transform": job["transform"], "crs": job["crs"], "physical_metrics": metrics, "owner_v3_context_sources": [{"partition_id": owner["partition_id"], "v3_context_path": owner["outputs"]["v3_context"]["path"], "v3_context_sha256": owner["outputs"]["v3_context"]["sha256"], "global_core_window": owner["core_window"]} for owner in owners], "source_inputs": [{"partition_id": owner["partition_id"], "probability_path": owner["probability_path"], "probability_sha256": owner["probability_sha256"], "decoder_valid_path": owner["decoder_valid_path"], "decoder_valid_sha256": owner["decoder_valid_sha256"]} for owner in owners], "v31b_policy_snapshot": job["policy_snapshot"], "v31b_policy_snapshot_sha256": job["policy_sha"], "v31b_audit": candidate_audit, "runtime_seconds": time.monotonic() - started, "proposal_counts": proposal_counts, "generation_rejection_events": generation_rejection_events, "coverage": {"published_strict_core_only": True, "expanded_owner_v3_context_coverage_exact_once": True, "expanded_decoder_valid_pixel_count": int(valid.sum()), "core_strict_valid_pixel_count": int(core_valid.sum())}, "outputs": {"v31a": {"path": "v31a_core.npy", "sha256": hashes["v31a_core.npy"], "shape": list(output.shape), "dtype": "int16", "candidate_label": "B"}}}
    audit["audit_sha256"] = _sha256_json(audit)
    _atomic_json(staging / "audit.json", audit)
    hashes["audit.json"] = _sha256_file(staging / "audit.json")
    _atomic_json(staging / "outputs_sha256.json", {"files": hashes})
    directory.parent.mkdir(parents=True, exist_ok=True)
    os.rename(staging, directory)
    return {"partition_id": entry["partition_id"], "global_core_window": entry["core_window"], "global_expanded_window": expanded, "core_transform": list(tuple(Affine(*job["transform"]) * Affine.translation(entry["core_window"]["x0"], entry["core_window"]["y0"]))[:6]), "crs": job["crs"], "physical_metrics": metrics, "owner_core_pixel_count": int(core_valid.size), "valid_pixel_count": int(core_valid.sum()), "runtime_seconds": audit["runtime_seconds"], "proposal_counts": audit["proposal_counts"], "generation_rejection_events": audit["generation_rejection_events"], "outputs": {"raw": entry["outputs"]["raw"], "v3": entry["outputs"]["v3"], "valid": entry["outputs"]["valid"], "v31a": {**audit["outputs"]["v31a"], "path": str((directory / "v31a_core.npy").resolve())}}, "stage_v3_audit": {"path": entry["stage_v3_audit_path"], "sha256": entry["stage_v3_audit_sha256"]}, "audit": {"path": str((directory / "audit.json").resolve()), "sha256": hashes["audit.json"]}}


def run(parent_manifest: Path, output_root: Path, *, workers: int, resume: bool, self_test: bool = False) -> dict[str, Any]:
    if workers < 1 or workers > 2:
        raise V31BRunError("--workers must be 1 or 2")
    module, apply, policy, policy_snapshot = _b_api()
    source = _load_parent(parent_manifest.resolve(), self_test=self_test)
    if output_root.exists() and any(output_root.iterdir()) and not resume:
        raise V31BRunError(f"refusing non-empty output root without --resume: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    verified: dict[str, dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_verify_source, entry): entry["partition_id"] for entry in source["entries"]}
        for future in as_completed(futures):
            verified[futures[future]] = future.result()
    code = _code_sha256(module, apply)
    policy_sha = _sha256_json(policy_snapshot)
    fingerprint_payload = {"schema_version": SCHEMA_VERSION, "candidate_label": "B", "parent_v31a_manifest_sha256": source["parent_sha256"], "parent_execution_fingerprint_sha256": source["parent"]["execution_fingerprint_sha256"], "class_codes": list(CLASS_ORDER), "context_pixels": CONTEXT_PIXELS, "v31b_policy_snapshot_sha256": policy_sha, "code_sha256": code, "source_probability_decoder_valid_sha256": verified}
    fingerprint = _sha256_json(fingerprint_payload)
    manifest_path = output_root / "run_manifest.json"
    completed: dict[str, dict[str, Any]] = {}
    if resume and manifest_path.is_file():
        prior = _read_json(manifest_path)
        if prior.get("execution_fingerprint_sha256") != fingerprint:
            raise V31BRunError("resume execution fingerprint differs")
        unexpected = [item.name for item in output_root.iterdir() if item.name not in {"run_manifest.json", "partitions"}]
        if unexpected:
            raise V31BRunError(f"resume output root contains unmanaged entries: {sorted(unexpected)}")
        partition_root = output_root / "partitions"
        if partition_root.exists():
            expected_ids = {entry["partition_id"] for entry in source["entries"]}
            unknown = [item.name for item in partition_root.iterdir() if item.name not in expected_ids]
            if unknown:
                raise V31BRunError(f"resume contains unknown Partition directories: {sorted(unknown)}")
        for entry in source["entries"]:
            directory = output_root / "partitions" / entry["partition_id"] / "stage_v31b"
            if directory.exists():
                children = {item.name for item in directory.parent.iterdir()}
                if children != {"stage_v31b"}:
                    raise V31BRunError(f"resume Partition has unmanaged stage entries: {directory.parent}")
                audit = _validate_b_stage(output_root, entry["partition_id"], fingerprint, _shape(entry["core_window"]))
                completed[entry["partition_id"]] = {"partition_id": entry["partition_id"], "global_core_window": entry["core_window"], "global_expanded_window": audit["global_expanded_window"], "core_transform": audit.get("core_transform", list(tuple(Affine(*source["transform"]) * Affine.translation(entry["core_window"]["x0"], entry["core_window"]["y0"]))[:6])), "crs": source["crs"], "physical_metrics": audit["physical_metrics"], "owner_core_pixel_count": int(np.prod(_shape(entry["core_window"]))), "valid_pixel_count": audit["coverage"]["core_strict_valid_pixel_count"], "runtime_seconds": audit["runtime_seconds"], "proposal_counts": audit["proposal_counts"], "generation_rejection_events": audit["generation_rejection_events"], "outputs": {"raw": entry["outputs"]["raw"], "v3": entry["outputs"]["v3"], "valid": entry["outputs"]["valid"], "v31a": {**audit["outputs"]["v31a"], "path": str((directory / "v31a_core.npy").resolve())}}, "stage_v3_audit": {"path": entry["stage_v3_audit_path"], "sha256": entry["stage_v3_audit_sha256"]}, "audit": {"path": str((directory / "audit.json").resolve()), "sha256": _sha256_file(directory / "audit.json")}}
    elif resume and any(output_root.iterdir()):
        raise V31BRunError("non-empty resume output lacks run_manifest.json")
    manifest: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "kind": "v31a_full_partition_core_comparison", "candidate_label": "B", "candidate_algorithm": "v31b_policy/apply_v31b_candidate", "status": "running", "self_test": self_test, "parent_v31a_manifest": source["parent_path"], "parent_v31a_manifest_sha256": source["parent_sha256"], "snapshot_manifest": source["parent"]["snapshot_manifest"], "snapshot_manifest_sha256": source["parent"]["snapshot_manifest_sha256"], "execution_fingerprint": fingerprint_payload, "execution_fingerprint_sha256": fingerprint, "class_codes": list(CLASS_ORDER), "label_encoding": "class_codes_int16_invalid_minus_one", "processing_transform": source["transform"], "crs": source["crs"], "global_window": source["global_window"], "context_pixels": CONTEXT_PIXELS, "code_sha256": code, "v3_policy_snapshot": source["parent"]["v3_policy_snapshot"], "v3_policy_snapshot_sha256": source["parent"]["v3_policy_snapshot_sha256"], "v31a_policy_snapshot": policy_snapshot, "v31a_policy_snapshot_sha256": policy_sha, "v31b_policy_snapshot": policy_snapshot, "v31b_policy_snapshot_sha256": policy_sha, "requested_partition_count": len(source["entries"]), "completed_partition_count": len(completed), "stage_v3_complete": True, "stage_v3": source["parent"]["stage_v3"], "stage_v3_partitions": source["parent"]["stage_v3_partitions"], "resource_plan": {"workers": workers, "real_workers_hard_limit": 2, "V3_execution": "forbidden_reused_parent_stage_v3_only"}, "partitions": [completed[key] for key in sorted(completed)]}
    _atomic_json(manifest_path, manifest)
    staging_root = output_root.parent / f".{output_root.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    staging_root.mkdir(exist_ok=False)
    try:
        pending = [entry for entry in source["entries"] if entry["partition_id"] not in completed]
        payload = {"entries": source["entries"], "global_window": source["global_window"], "transform": source["transform"], "crs": source["crs"], "apply": apply, "policy": policy, "policy_snapshot": policy_snapshot, "policy_sha": policy_sha, "fingerprint": fingerprint, "parent_path": source["parent_path"], "parent_sha256": source["parent_sha256"], "output_root": str(output_root.resolve()), "staging_root": str(staging_root)}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_partition, {**payload, "entry": entry}): entry["partition_id"] for entry in pending}
            for future in as_completed(futures):
                result = future.result()
                completed[result["partition_id"]] = result
                manifest["completed_partition_count"] = len(completed)
                manifest["partitions"] = [completed[key] for key in sorted(completed)]
                _atomic_json(manifest_path, manifest)
                print(f"stage_v31b {len(completed)}/{len(source['entries'])} {result['partition_id']}", flush=True)
    finally:
        for path in staging_root.glob("*"):
            if path.is_dir():
                import shutil
                shutil.rmtree(path, ignore_errors=True)
        staging_root.rmdir()
    if len(completed) != len(source["entries"]):
        raise V31BRunError("V3.1-B run incomplete")
    manifest.update({"status": "complete", "completed_partition_count": len(completed), "partitions": [completed[key] for key in sorted(completed)], "coverage": {"all_snapshot_partitions_requested": len(source["entries"]) == REAL_PARTITION_COUNT if not self_test else True, "core_windows_nonoverlapping": True, "global_core_grid_exact": True, "complete": True, "published_core_pixel_count": int(sum(item["owner_core_pixel_count"] for item in completed.values())), "published_valid_pixel_count": int(sum(item["valid_pixel_count"] for item in completed.values()))}, "proposal_counts": {key: int(sum(item["proposal_counts"][key] for item in completed.values())) for key in ("raw_generated", "canonical_generated", "duplicate_proposal_count", "canonical_accepted", "canonical_rejected", "rollback")}, "generation_rejection_events": {"count": int(sum(item["generation_rejection_events"]["count"] for item in completed.values())), "by_reason": {key: int(sum(item["generation_rejection_events"]["by_reason"].get(key, 0) for item in completed.values())) for key in sorted({reason for item in completed.values() for reason in item["generation_rejection_events"]["by_reason"]})}}, "runtime_seconds": float(sum(item["runtime_seconds"] for item in completed.values()))})
    manifest["manifest_sha256"] = _sha256_json(manifest)
    _atomic_json(manifest_path, manifest)
    return manifest


def _fixture_stage(partition_root: Path, probabilities: np.ndarray, valid: np.ndarray) -> None:
    """Write the minimal archived input contract used by the four-Core test."""
    input_dir = partition_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    probability_path, valid_path = input_dir / "probabilities.npy", input_dir / "decoder_valid.npy"
    np.save(probability_path, probabilities, allow_pickle=False)
    np.save(valid_path, valid, allow_pickle=False)
    artifacts = {
        PROBABILITY_ARTIFACT: {"path": "input/probabilities.npy", "sha256": _sha256_file(probability_path), "shape": list(probabilities.shape), "dtype": "float32"},
        DECODER_VALID_ARTIFACT: {"path": "input/decoder_valid.npy", "sha256": _sha256_file(valid_path), "shape": list(valid.shape), "dtype": "bool"},
    }
    stage = {"schema_version": 1, "stage": "input", "artifacts": artifacts}
    stage_path = partition_root / "manifests" / "input.json"
    _atomic_json(stage_path, stage)
    partition = {"schema_version": 1, "stages": {"input": {"path": "manifests/input.json", "sha256": _sha256_file(stage_path)}}}
    _atomic_json(partition_root / "manifest.json", partition)


def _write_self_test_parent(root: Path) -> Path:
    """Create a complete synthetic A manifest whose V3 outputs are immutable."""
    source_root, parent_root = root / "source", root / "parent"
    source_root.mkdir(parents=True)
    parent_root.mkdir(parents=True)
    parts, stages = [], []
    execution = {"self_test": True, "v3_stage": "immutable"}
    fingerprint = _sha256_json(execution)
    code = {"synthetic_v3": "0" * 64}
    policy = v3_policy_snapshot()
    for row in range(2):
        for col in range(2):
            partition_id = f"partition_{row:05d}_{col:05d}"
            core = {"x0": col * 12, "y0": row * 12, "x1": (col + 1) * 12, "y1": (row + 1) * 12}
            probabilities = np.zeros((len(CLASS_ORDER), 12, 12), dtype=np.float32)
            probabilities[0] = 0.9  # Raw differs from the inherited V3 baseline.
            probabilities[1] = 0.1
            valid = np.ones((12, 12), dtype=bool)
            _fixture_stage(source_root / "partitions" / partition_id, probabilities, valid)
            parts.append({"partition_id": partition_id, "row": row, "col": col, "core_window": core, "halo_window": core})
            v3_dir = parent_root / "partitions" / partition_id / "stage_v3"
            v3_dir.mkdir(parents=True)
            raw = np.full((12, 12), CLASS_ORDER[0], dtype=np.int16)
            v3_context = np.full((12, 12), 1, dtype=np.int16)
            v3 = np.full((12, 12), CLASS_ORDER[1], dtype=np.int16)
            for name, array in (("raw_core.npy", raw), ("v3_context_core.npy", v3_context), ("v3_core.npy", v3), ("valid_core.npy", valid)):
                np.save(v3_dir / name, array, allow_pickle=False)
            output_map = {
                "raw": {"path": str((v3_dir / "raw_core.npy").resolve()), "sha256": _sha256_file(v3_dir / "raw_core.npy")},
                "v3_context": {"path": str((v3_dir / "v3_context_core.npy").resolve()), "sha256": _sha256_file(v3_dir / "v3_context_core.npy")},
                "v3": {"path": str((v3_dir / "v3_core.npy").resolve()), "sha256": _sha256_file(v3_dir / "v3_core.npy")},
                "valid": {"path": str((v3_dir / "valid_core.npy").resolve()), "sha256": _sha256_file(v3_dir / "valid_core.npy")},
            }
            audit = {"execution_fingerprint_sha256": fingerprint, "code_sha256": code, "v3_policy_snapshot": policy, "v3_policy_snapshot_sha256": _sha256_json(policy), "global_core_window": core, "outputs": output_map}
            _atomic_json(v3_dir / "audit.json", audit)
            hashes = {name: _sha256_file(v3_dir / name) for name in ("raw_core.npy", "v3_context_core.npy", "v3_core.npy", "valid_core.npy", "audit.json")}
            _atomic_json(v3_dir / "outputs_sha256.json", {"files": hashes})
            stages.append({"partition_id": partition_id, "global_core_window": core, "outputs": output_map, "audit": {"path": str((v3_dir / "audit.json").resolve()), "sha256": hashes["audit.json"]}})
    snapshot = {"schema_version": 1, "source_raster_crs": "EPSG:3857", "processing_transform": [1.0, 0.0, 12300000.0, 0.0, -1.0, 4540000.0], "partitions": parts}
    snapshot_path = source_root / "snapshot_manifest.json"
    _atomic_json(snapshot_path, snapshot)
    parent = {"schema_version": 1, "kind": "v31a_full_partition_core_comparison", "status": "complete", "self_test": True, "snapshot_manifest": str(snapshot_path.resolve()), "snapshot_manifest_sha256": _sha256_file(snapshot_path), "execution_fingerprint": execution, "execution_fingerprint_sha256": _sha256_json(execution), "code_sha256": code, "v3_policy_snapshot": policy, "v3_policy_snapshot_sha256": _sha256_json(policy), "processing_transform": snapshot["processing_transform"], "crs": "EPSG:3857", "completed_partition_count": 4, "stage_v3": {"complete": True, "completed_partition_count": 4}, "stage_v3_partitions": stages, "partitions": []}
    parent["manifest_sha256"] = _sha256_json(parent)
    parent_path = parent_root / "run_manifest.json"
    _atomic_json(parent_path, parent)
    return parent_path


def _assert_self_test(manifest: Mapping[str, Any], parent_path: Path) -> None:
    if manifest.get("status") != "complete" or manifest.get("candidate_label") != "B" or manifest.get("completed_partition_count") != 4:
        raise V31BRunError("self-test did not finish four candidate-B Cores")
    parent = _read_json(parent_path)
    inherited = {entry["partition_id"]: entry for entry in parent["stage_v3_partitions"]}
    for entry in manifest["partitions"]:
        parent_entry = inherited[entry["partition_id"]]
        for key in ("raw", "v3", "valid"):
            if entry["outputs"][key]["sha256"] != parent_entry["outputs"][key]["sha256"] or entry["outputs"][key]["path"] != parent_entry["outputs"][key]["path"]:
                raise V31BRunError("self-test did not reuse fixed V3 output pointers")
        audit = _read_json(Path(entry["audit"]["path"]))
        owners = audit.get("owner_v3_context_sources") or []
        if not owners or any(not item.get("v3_context_sha256") for item in owners):
            raise V31BRunError("self-test did not record reused V3 context sources")
        output = np.load(entry["outputs"]["v31a"]["path"], allow_pickle=False)
        valid = np.load(entry["outputs"]["valid"]["path"], allow_pickle=False)
        if output.shape != valid.shape or np.any(output[~valid] != -1) or np.any(~np.isin(output[valid], CLASS_ORDER)):
            raise V31BRunError("self-test B publication violates strict single-label output")
        counts = entry["proposal_counts"]
        if counts["raw_generated"] != counts["canonical_generated"] + counts["duplicate_proposal_count"] or counts["canonical_generated"] != counts["canonical_accepted"] + counts["canonical_rejected"]:
            raise V31BRunError("self-test B proposal-count closure is invalid")


def _self_test(output_root: Path | None, workers: int) -> dict[str, Any]:
    # A B API is required even for the synthetic check: otherwise a passing test
    # would be evidence only for the runner shell, not the candidate experiment.
    _b_api()
    with tempfile.TemporaryDirectory(prefix="v31b-selftest-") as temporary:
        root = Path(temporary)
        parent = _write_self_test_parent(root)
        if output_root is None:
            result_root = root / "out"
            result = run(parent, result_root, workers=workers, resume=False, self_test=True)
            _assert_self_test(result, parent)
            first_bytes = (result_root / "run_manifest.json").read_bytes()
            first_sha = result["manifest_sha256"]
            resumed = run(parent, result_root, workers=workers, resume=True, self_test=True)
            _assert_self_test(resumed, parent)
            if resumed["manifest_sha256"] != first_sha or (result_root / "run_manifest.json").read_bytes() != first_bytes:
                raise V31BRunError("self-test resume changed the completed run_manifest")
            return resumed
        result = run(parent, output_root, workers=workers, resume=False, self_test=True)
        _assert_self_test(result, parent)
        first_bytes = (output_root / "run_manifest.json").read_bytes()
        first_sha = result["manifest_sha256"]
        resumed = run(parent, output_root, workers=workers, resume=True, self_test=True)
        _assert_self_test(resumed, parent)
        if resumed["manifest_sha256"] != first_sha or (output_root / "run_manifest.json").read_bytes() != first_bytes:
            raise V31BRunError("self-test resume changed the completed run_manifest")
        return resumed


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v31a-manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        if args.v31a_manifest is not None:
            parser.error("--self-test generates its own fixture after the B API lands")
    elif args.v31a_manifest is None or args.output_root is None:
        parser.error("real execution requires --v31a-manifest and --output-root")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    try:
        if args.self_test:
            result = _self_test(args.output_root, args.workers)
        else:
            result = run(args.v31a_manifest, args.output_root, workers=args.workers, resume=args.resume)
        message: dict[str, Any] = {"status": result["status"], "candidate_label": "B"}
        if args.output_root is not None:
            message["manifest"] = str((args.output_root / "run_manifest.json").resolve())
        else:
            message["completed_partition_count"] = result["completed_partition_count"]
        print(json.dumps(message, ensure_ascii=False), flush=True)
        return 0
    except V31BRunError as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
