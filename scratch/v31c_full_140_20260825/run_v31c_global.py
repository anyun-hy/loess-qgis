#!/usr/bin/env python3
"""Run isolated V3.1-C as collect -> global plan -> per-Core publish.

The input is an immutable complete V3.1-B manifest.  C reuses the same frozen
V3/probability snapshot to regenerate only cross-Core proposal discoveries,
deduplicates them in global pixel coordinates, builds a sparse global topology
index, solves one lexicographic constrained plan, and overlays selected actions
on B.  Production V3 and the B publication are read-only inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np
from rasterio.transform import Affine


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "inference_scripts"))

from deployment_config import CLASS_ORDER  # noqa: E402
from fragmentation_v31_candidate import (  # noqa: E402
    CrossCoreDiscovery,
    PlannedAction,
    V31C_COORDINATION_MODE,
    V31C_POLICY_ID,
    V31C_POLICY_VERSION,
    canonicalize_global_discoveries,
    collect_cross_core_discoveries,
    global_action_to_dict,
    select_global_actions,
    v31b_policy,
    policy_snapshot,
)
from fragmentation_v31_candidate import v31c_components as component_api  # noqa: E402


SCHEMA_VERSION = 1
REAL_PARTITION_COUNT = 140
B_RUNNER_PATH = REPO_ROOT / "scratch" / "v31b_full_140_20260824" / "run_v31b_from_v3.py"
EVALUATOR_PATH = REPO_ROOT / "scratch" / "v31a_full_140_20260824" / "evaluate_global_fragmentation.py"
COLLECT_COMPATIBLE_PREDECESSOR_RUNNER_SHA256 = frozenset(
    {
        # v1 collection code is identical; the successor only teaches the
        # post-collection budget reader about B's audited empty-Core shortcut.
        "51f94ac66020b5ce909822e0986b944fe66fe842948a4a0b0cacea7cda8ad231",
    }
)


class V31CRunError(RuntimeError):
    """An input, coordination, publication, or validation contract failed."""


def _load_python_module(name: str, path: Path) -> Any:
    if not path.is_file():
        raise V31CRunError(f"required isolated helper is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise V31CRunError(f"cannot load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


B_RUNNER = _load_python_module("_v31c_frozen_b_runner", B_RUNNER_PATH)
EVALUATOR = _load_python_module("_v31c_global_evaluator", EVALUATOR_PATH)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V31CRunError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V31CRunError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".npy", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.save(handle, np.ascontiguousarray(values), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _resolve(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _code_sha256() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        B_RUNNER_PATH.resolve(),
        EVALUATOR_PATH.resolve(),
        REPO_ROOT / "inference_scripts" / "fragmentation_v31_candidate" / "candidate.py",
        REPO_ROOT / "inference_scripts" / "fragmentation_v31_candidate" / "__init__.py",
        REPO_ROOT / "inference_scripts" / "fragmentation_v31_candidate" / "v31c.py",
        REPO_ROOT / "inference_scripts" / "fragmentation_v31_candidate" / "v31c_components.py",
        REPO_ROOT / "inference_scripts" / "deployment_config.py",
    ]
    return {str(path.relative_to(REPO_ROOT)): _sha256_file(path) for path in paths}


def _window_contains(window: Mapping[str, int], row: int, col: int) -> bool:
    return (
        int(window["y0"]) <= int(row) < int(window["y1"])
        and int(window["x0"]) <= int(col) < int(window["x1"])
    )


def _owner_for(entries: Sequence[Mapping[str, Any]], row: int, col: int) -> str | None:
    owners = [
        str(entry["partition_id"])
        for entry in entries
        if _window_contains(entry["core_window"], row, col)
    ]
    if len(owners) > 1:
        raise V31CRunError(f"global pixel ({row},{col}) has multiple Core owners")
    return owners[0] if owners else None


def _load_b_manifest(path: Path, *, self_test: bool) -> dict[str, Any]:
    manifest = _read_json(path)
    expected = 4 if self_test else REAL_PARTITION_COUNT
    if (
        manifest.get("status") != "complete"
        or manifest.get("candidate_label") != "B"
        or int(manifest.get("completed_partition_count", -1)) != expected
        or len(manifest.get("partitions") or []) != expected
    ):
        raise V31CRunError(f"B manifest must be complete with {expected} Cores")
    if manifest.get("class_codes") != list(CLASS_ORDER):
        raise V31CRunError("B manifest class_codes differ from the current frozen CLASS_ORDER")
    declared = manifest.get("manifest_sha256")
    if declared and declared != _sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    ):
        raise V31CRunError("B manifest self SHA-256 mismatch")
    parent_path = _resolve(str(manifest.get("parent_v31a_manifest", "")), path.parent)
    if not parent_path.is_file() or _sha256_file(parent_path) != manifest.get(
        "parent_v31a_manifest_sha256"
    ):
        raise V31CRunError("B parent V3.1-A manifest SHA-256 mismatch")
    source = B_RUNNER._load_parent(parent_path, self_test=self_test)
    source_by_id = {entry["partition_id"]: entry for entry in source["entries"]}
    parts: dict[str, dict[str, Any]] = {}
    for raw in manifest["partitions"]:
        partition_id = str(raw.get("partition_id", ""))
        if partition_id not in source_by_id or partition_id in parts:
            raise V31CRunError(f"B partition identity mismatch: {partition_id}")
        if raw.get("global_core_window") != source_by_id[partition_id]["core_window"]:
            raise V31CRunError(f"{partition_id}: B and V3 Core windows differ")
        outputs = raw.get("outputs") or {}
        if set(("raw", "v3", "valid", "v31a")) - set(outputs):
            raise V31CRunError(f"{partition_id}: B fixed outputs are incomplete")
        resolved_outputs: dict[str, dict[str, Any]] = {}
        for key, dtype in (
            ("raw", np.dtype("int16")),
            ("v3", np.dtype("int16")),
            ("valid", np.dtype("bool")),
            ("v31a", np.dtype("int16")),
        ):
            item = outputs[key]
            output_path = _resolve(str(item.get("path", "")), path.parent)
            expected_shape = B_RUNNER._shape(source_by_id[partition_id]["core_window"])
            B_RUNNER._verify_npy(
                output_path,
                str(item.get("sha256", "")),
                expected_shape,
                dtype,
                f"{partition_id}: B {key}",
            )
            resolved_outputs[key] = {**item, "path": str(output_path)}
        audit_ref = raw.get("audit") or {}
        audit_path = _resolve(str(audit_ref.get("path", "")), path.parent)
        if not audit_path.is_file() or _sha256_file(audit_path) != audit_ref.get("sha256"):
            raise V31CRunError(f"{partition_id}: B audit SHA-256 mismatch")
        audit = _read_json(audit_path)
        if audit.get("candidate_label") != "B" or audit.get("partition_id") != partition_id:
            raise V31CRunError(f"{partition_id}: B audit identity mismatch")
        parts[partition_id] = {
            **raw,
            "outputs": resolved_outputs,
            "audit_path": str(audit_path),
            "audit_data": audit,
        }
    if set(parts) != set(source_by_id):
        raise V31CRunError("B and frozen source partition sets differ")
    return {
        "manifest": manifest,
        "manifest_path": str(path.resolve()),
        "manifest_file_sha256": _sha256_file(path),
        "source": source,
        "parts": parts,
    }


def _discovery_from_dict(value: Mapping[str, Any]) -> CrossCoreDiscovery:
    return CrossCoreDiscovery(
        discovery_id=str(value["discovery_id"]),
        discovery_partition_id=str(value["discovery_partition_id"]),
        kind=str(value["kind"]),
        target_index=int(value["target_index"]),
        target_code=int(value["target_code"]),
        footprint=tuple((int(row), int(col)) for row, col in value["footprint"]),
        involved_core_ids=tuple(str(item) for item in value["involved_core_ids"]),
        source_codes=tuple(int(item) for item in value["source_codes"]),
        source_anchors=tuple((int(row), int(col)) for row, col in value["source_anchors"]),
        target_anchors=tuple((int(row), int(col)) for row, col in value["target_anchors"]),
        dynamic_reduction=int(value["dynamic_reduction"]),
        component_reduction=int(value["component_reduction"]),
        probability_support=float(value["probability_support"]),
        area_m2=float(value["area_m2"]),
        edge_distance_m=(None if value.get("edge_distance_m") is None else float(value["edge_distance_m"])),
        path_length_m=(None if value.get("path_length_m") is None else float(value["path_length_m"])),
        local_proposal_id=str(value["local_proposal_id"]),
        footprint_sha256=str(value["footprint_sha256"]),
    )


def _collect_partition(job: Mapping[str, Any]) -> dict[str, Any]:
    entry = job["entry"]
    shard_path = Path(job["collect_root"]) / f"{entry['partition_id']}.json"
    if shard_path.is_file():
        prior = _read_json(shard_path)
        body = {key: value for key, value in prior.items() if key != "shard_sha256"}
        compatible_fingerprints = {
            str(job["fingerprint"]),
            *(str(value) for value in job.get("compatible_collect_fingerprints", ())),
        }
        if (
            prior.get("execution_fingerprint_sha256") in compatible_fingerprints
            and prior.get("shard_sha256") == _sha256_json(body)
        ):
            return prior
        raise V31CRunError(f"collect resume shard differs: {shard_path}")
    baseline, probabilities, _decoder_valid, strict, expanded, _owners = B_RUNNER._stitch(
        entry, job["entries"], job["global_window"]
    )
    metrics = B_RUNNER._physical(job["transform"], job["crs"], expanded)
    started = time.monotonic()
    discoveries, audit = collect_cross_core_discoveries(
        baseline,
        class_codes=CLASS_ORDER,
        pixel_area_m2=metrics["pixel_area_m2"],
        pixel_size_m=(metrics["row_step_m"], metrics["column_step_m"]),
        # C's global topology/evaluator domain is the union of strict owner
        # validity, not decoder-valid pixels outside the study range.  Using
        # decoder validity here could create target anchors or connectivity
        # paths that the global component index correctly excludes.
        valid_mask=strict,
        probabilities=probabilities,
        confidence=probabilities.max(axis=0).astype(np.float32),
        global_origin=(expanded["y0"], expanded["x0"]),
        discovery_partition_id=entry["partition_id"],
        owner_for_global_pixel=lambda row, col: (
            _owner_for(job["entries"], row, col)
            if (
                int(expanded["y0"]) <= int(row) < int(expanded["y1"])
                and int(expanded["x0"]) <= int(col) < int(expanded["x1"])
                and bool(strict[int(row) - int(expanded["y0"]), int(col) - int(expanded["x0"])])
            )
            else None
        ),
        policy=job["policy"],
    )
    shard: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "v31c_collect",
        "partition_id": entry["partition_id"],
        "execution_fingerprint_sha256": job["fingerprint"],
        "global_core_window": entry["core_window"],
        "global_expanded_window": expanded,
        "physical_metrics": metrics,
        "runtime_seconds": time.monotonic() - started,
        "audit": audit,
        "discoveries": [asdict(item) for item in discoveries],
    }
    shard["shard_sha256"] = _sha256_json(shard)
    _atomic_json(shard_path, shard)
    return shard


def _collect_all(
    source: Mapping[str, Any],
    *,
    output_root: Path,
    workers: int,
    fingerprint: str,
    policy: Any,
    compatible_collect_fingerprints: Sequence[str] = (),
) -> tuple[list[CrossCoreDiscovery], list[dict[str, Any]]]:
    collect_root = output_root / "collect"
    collect_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "entries": source["entries"],
        "global_window": source["global_window"],
        "transform": source["transform"],
        "crs": source["crs"],
        "collect_root": str(collect_root),
        "fingerprint": fingerprint,
        "policy": policy,
        "compatible_collect_fingerprints": tuple(compatible_collect_fingerprints),
    }
    shards: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_collect_partition, {**payload, "entry": entry}): entry["partition_id"]
            for entry in source["entries"]
        }
        for future in as_completed(futures):
            shard = future.result()
            shards[str(shard["partition_id"])] = shard
            print(
                f"v31c_collect {len(shards)}/{len(source['entries'])} {shard['partition_id']}",
                flush=True,
            )
    discoveries = [
        _discovery_from_dict(value)
        for partition_id in sorted(shards)
        for value in shards[partition_id]["discoveries"]
    ]
    return discoveries, [shards[key] for key in sorted(shards)]


def _local_point(
    window: Mapping[str, int], point: tuple[int, int]
) -> tuple[int, int]:
    row, col = point
    return row - int(window["y0"]), col - int(window["x0"])


def _valid_global_point(
    point: tuple[int, int],
    *,
    source_entries: Sequence[Mapping[str, Any]],
    b_parts: Mapping[str, Mapping[str, Any]],
    valid_cache: dict[str, np.ndarray],
) -> bool:
    owner = _owner_for(source_entries, *point)
    if owner is None:
        return False
    values = valid_cache.setdefault(
        owner,
        np.load(b_parts[owner]["outputs"]["valid"]["path"], mmap_mode="r", allow_pickle=False),
    )
    row, col = _local_point(b_parts[owner]["global_core_window"], point)
    return bool(values[row, col])


def _b_changes_and_target_anchors(
    *,
    source_entries: Sequence[Mapping[str, Any]],
    b_parts: Mapping[str, Mapping[str, Any]],
) -> tuple[
    set[tuple[int, int]],
    list[tuple[tuple[int, int], int, int]],
    set[tuple[int, int]],
]:
    """Return B changed cells and V3 target anchors adjacent to its overlay."""

    changed: set[tuple[int, int]] = set()
    records: list[tuple[tuple[int, int], int, int]] = []
    target_anchors: set[tuple[int, int]] = set()
    label_cache: dict[str, np.ndarray] = {}
    valid_cache: dict[str, np.ndarray] = {}
    for partition_id in sorted(b_parts):
        part = b_parts[partition_id]
        v3 = np.load(part["outputs"]["v3"]["path"], mmap_mode="r", allow_pickle=False)
        result = np.load(part["outputs"]["v31a"]["path"], mmap_mode="r", allow_pickle=False)
        valid = np.load(part["outputs"]["valid"]["path"], mmap_mode="r", allow_pickle=False)
        rows, cols = np.nonzero(valid & (v3 != result))
        window = part["global_core_window"]
        for row, col in zip(rows.tolist(), cols.tolist()):
            point = (int(window["y0"]) + row, int(window["x0"]) + col)
            source_code, target_code = int(v3[row, col]), int(result[row, col])
            changed.add(point)
            records.append((point, source_code, target_code))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = (point[0] + dr, point[1] + dc)
                owner = _owner_for(source_entries, *neighbor)
                if owner is None:
                    continue
                owner_part = b_parts[owner]
                owner_valid = valid_cache.setdefault(
                    owner,
                    np.load(owner_part["outputs"]["valid"]["path"], mmap_mode="r", allow_pickle=False),
                )
                rr, cc = _local_point(owner_part["global_core_window"], neighbor)
                if not owner_valid[rr, cc]:
                    continue
                owner_v3 = label_cache.setdefault(
                    owner,
                    np.load(owner_part["outputs"]["v3"]["path"], mmap_mode="r", allow_pickle=False),
                )
                if int(owner_v3[rr, cc]) == target_code:
                    target_anchors.add(neighbor)
    return changed, records, target_anchors


def _component_key_text(value: Any) -> str:
    key = value.key
    return f"{int(key.class_code)}:{int(key.min_row)}:{int(key.min_col)}"


def _global_component_lookup(
    *,
    query_points: Sequence[tuple[int, int]],
    b_parts: Mapping[str, Mapping[str, Any]],
    global_window: Mapping[str, int],
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, dict[str, Any]], int]:
    if not query_points:
        return {}, {}, 0
    tiles = []
    for partition_id in sorted(b_parts):
        part = b_parts[partition_id]
        window = part["global_core_window"]
        metrics = part.get("physical_metrics") or {}
        tiles.append(
            component_api.CoreTile(
                core_id=partition_id,
                window=(
                    int(window["y0"]),
                    int(window["x0"]),
                    int(window["y1"]) - int(window["y0"]),
                    int(window["x1"]) - int(window["x0"]),
                ),
                labels=np.load(part["outputs"]["v3"]["path"], mmap_mode="r", allow_pickle=False),
                valid=np.load(part["outputs"]["valid"]["path"], mmap_mode="r", allow_pickle=False),
                pixel_area_m2=float(metrics["pixel_area_m2"]),
            )
        )
    ordered_queries = tuple(sorted(set(query_points)))
    result = component_api.build_global_component_index(
        tiles,
        ordered_queries,
    )
    by_point: dict[tuple[int, int], dict[str, Any]] = {}
    by_key: dict[str, dict[str, Any]] = {}
    for item in result.query_components:
        key = _component_key_text(item)
        info = {
            "key": key,
            "class_code": int(item.key.class_code),
            "pixel_count": int(item.pixel_count),
            "area_m2": float(item.area_m2),
        }
        point = (int(item.point[0]), int(item.point[1]))
        by_point[point] = info
        previous = by_key.get(key)
        if previous is not None and previous != info:
            raise V31CRunError(f"global component metadata changed for {key}")
        by_key[key] = info
    if set(by_point) != set(ordered_queries):
        raise V31CRunError("global component index omitted query points")
    return by_point, by_key, int(result.global_component_count)


def _budget_remaining(
    b_parts: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[tuple[str, int], int], dict[tuple[str, int], int]]:
    source_remaining: dict[tuple[str, int], int] = {}
    target_remaining: dict[tuple[str, int], int] = {}
    for partition_id in sorted(b_parts):
        part = b_parts[partition_id]
        candidate_audit = part["audit_data"].get("v31b_audit") or {}
        budgets = candidate_audit.get("class_budget_pixels") or {}
        if candidate_audit.get("skipped") and candidate_audit.get("reason") == "empty_owner_core_strict_valid":
            audit_valid = int((part["audit_data"].get("coverage") or {}).get("core_strict_valid_pixel_count", -1))
            manifest_valid = int(part.get("valid_pixel_count", -1))
            if audit_valid != 0 or manifest_valid != 0 or budgets:
                raise V31CRunError(
                    f"{partition_id}: empty-Core B audit contradicts its validity or budgets"
                )
            for code in CLASS_ORDER:
                source_remaining[(partition_id, int(code))] = 0
                target_remaining[(partition_id, int(code))] = 0
            continue
        if set(int(code) for code in budgets) != set(int(code) for code in CLASS_ORDER):
            raise V31CRunError(f"{partition_id}: B class-budget audit is incomplete")
        for code_text, row in budgets.items():
            code = int(code_text)
            source_limit = math.floor(float(row["source_loss_limit"]) + 1e-12)
            target_limit = math.floor(float(row["target_gain_limit"]) + 1e-12)
            source_remaining[(partition_id, code)] = max(
                0, source_limit - int(row["source_loss"])
            )
            target_remaining[(partition_id, code)] = max(
                0, target_limit - int(row["target_gain"])
            )
    return source_remaining, target_remaining


def _prepare_global_plan(
    *,
    discoveries: Sequence[CrossCoreDiscovery],
    source_entries: Sequence[Mapping[str, Any]],
    b_parts: Mapping[str, Mapping[str, Any]],
    global_window: Mapping[str, int],
    policy: Any,
) -> tuple[list[PlannedAction], dict[str, Any], list[Any]]:
    actions, duplicate_audit = canonicalize_global_discoveries(discoveries)
    b_changed, b_change_records, b_target_anchors = _b_changes_and_target_anchors(
        source_entries=source_entries, b_parts=b_parts
    )
    query_points = {
        point
        for action in actions
        for point in (*action.footprint, *action.source_anchors, *action.target_anchors)
    }
    query_points.update(point for point, _source, _target in b_change_records)
    query_points.update(b_target_anchors)
    point_info, component_info, global_component_count = _global_component_lookup(
        query_points=sorted(query_points),
        b_parts=b_parts,
        global_window=global_window,
    )
    b_affected_components = {
        point_info[point]["key"] for point, _source, _target in b_change_records
    } | {point_info[point]["key"] for point in b_target_anchors}
    pixel_area_by_core = {
        partition_id: float(part["physical_metrics"]["pixel_area_m2"])
        for partition_id, part in b_parts.items()
    }
    planned: list[PlannedAction] = []
    rejected: dict[str, str] = {}
    global_scores: dict[str, dict[str, Any]] = {}
    for action in actions:
        if any(point in b_changed for point in action.footprint):
            rejected[action.action_id] = "footprint_conflict_with_b"
            continue
        footprint_infos = [point_info[point] for point in action.footprint]
        if any(info["class_code"] == action.target_code for info in footprint_infos):
            rejected[action.action_id] = "frozen_action_no_longer_changes_target"
            continue
        if any(info["class_code"] not in action.source_codes for info in footprint_infos):
            rejected[action.action_id] = "global_source_code_mismatch"
            continue
        source_keys = tuple(sorted({info["key"] for info in footprint_infos}))
        target_infos = [point_info[point] for point in action.target_anchors]
        if not target_infos or any(info["class_code"] != action.target_code for info in target_infos):
            rejected[action.action_id] = "global_target_anchor_mismatch"
            continue
        target_keys = tuple(sorted({info["key"] for info in target_infos}))
        if (set(source_keys) | set(target_keys)) & b_affected_components:
            rejected[action.action_id] = "topology_component_touched_by_b"
            continue
        removed_pixels: Counter[str] = Counter(info["key"] for info in footprint_infos)
        removed_area: Counter[str] = Counter()
        source_charge_counter: Counter[tuple[str, int]] = Counter()
        target_charge_counter: Counter[tuple[str, int]] = Counter()
        added_area = 0.0
        for point, info in zip(action.footprint, footprint_infos):
            owner = _owner_for(source_entries, *point)
            if owner is None:
                raise V31CRunError(f"{action.action_id}: footprint pixel has no Core owner")
            pixel_area = pixel_area_by_core[owner]
            removed_area[info["key"]] += pixel_area
            source_charge_counter[(owner, info["class_code"])] += 1
            target_charge_counter[(owner, action.target_code)] += 1
            added_area += pixel_area
        component_reduction = 0
        dynamic_reduction = 0
        for key in source_keys:
            info = component_info[key]
            threshold = float(policy.class_policies[int(info["class_code"])].dynamic_fragmentation_m2)
            before_dynamic = 0.0 < info["area_m2"] < threshold
            remaining_pixels = info["pixel_count"] - removed_pixels[key]
            remaining_area = info["area_m2"] - removed_area[key]
            if remaining_pixels < 0 or remaining_area < -1e-7:
                raise V31CRunError(f"{action.action_id}: source removal exceeds global component")
            after_dynamic = remaining_pixels > 0 and 0.0 < remaining_area < threshold
            dynamic_reduction += int(before_dynamic) - int(after_dynamic)
            component_reduction += int(remaining_pixels == 0)
        target_area = sum(component_info[key]["area_m2"] for key in target_keys)
        target_threshold = float(policy.class_policies[action.target_code].dynamic_fragmentation_m2)
        dynamic_reduction += sum(
            int(0.0 < component_info[key]["area_m2"] < target_threshold)
            for key in target_keys
        ) - int(0.0 < target_area + added_area < target_threshold)
        component_reduction += len(target_keys) - 1
        if component_reduction <= 0 or dynamic_reduction < 0:
            rejected[action.action_id] = "global_topology_score_no_gain"
            continue
        globally_scored = replace(
            action,
            dynamic_reduction=int(dynamic_reduction),
            component_reduction=int(component_reduction),
            area_m2=float(added_area),
        )
        global_scores[action.action_id] = {
            "local_conservative_dynamic_reduction": action.dynamic_reduction,
            "local_conservative_component_reduction": action.component_reduction,
            "global_dynamic_reduction": dynamic_reduction,
            "global_component_reduction": component_reduction,
            "global_changed_area_m2": added_area,
        }
        planned.append(
            PlannedAction(
                action=globally_scored,
                source_component_keys=source_keys,
                target_component_keys=target_keys,
                source_charges=tuple(
                    (core, code, count)
                    for (core, code), count in sorted(source_charge_counter.items())
                ),
                target_charges=tuple(
                    (core, code, count)
                    for (core, code), count in sorted(target_charge_counter.items())
                ),
            )
        )
    source_remaining, target_remaining = _budget_remaining(b_parts)
    selected, solver_audit = select_global_actions(
        planned,
        source_remaining=source_remaining,
        target_remaining=target_remaining,
    )
    plan_audit = {
        "coordination_mode": V31C_COORDINATION_MODE,
        "discovery_occurrence_count": len(discoveries),
        "canonical_action_count": len(actions),
        "duplicate_global_discovery_count": len(duplicate_audit),
        "preselection_eligible_count": len(planned),
        "preselection_rejections": dict(sorted(rejected.items())),
        "selected_action_count": len(selected),
        "b_changed_pixel_count": len(b_changed),
        "b_affected_global_component_count": len(b_affected_components),
        "global_v3_component_count": global_component_count,
        "global_scores": global_scores,
        "duplicate_global_discovery_audit": duplicate_audit,
        "source_budget_remaining_before_c": {
            f"{core}:{code}": value for (core, code), value in sorted(source_remaining.items())
        },
        "target_budget_remaining_before_c": {
            f"{core}:{code}": value for (core, code), value in sorted(target_remaining.items())
        },
        "solver": solver_audit,
    }
    return selected, plan_audit, actions


def _selected_overlay_by_core(
    selected: Sequence[PlannedAction],
    source_entries: Sequence[Mapping[str, Any]],
) -> dict[str, dict[tuple[int, int], tuple[int, str]]]:
    overlays: dict[str, dict[tuple[int, int], tuple[int, str]]] = {
        str(entry["partition_id"]): {} for entry in source_entries
    }
    for planned in selected:
        action = planned.action
        for point in action.footprint:
            owner = _owner_for(source_entries, *point)
            if owner is None:
                raise V31CRunError(f"{action.action_id}: selected pixel has no owner")
            prior = overlays[owner].get(point)
            assignment = (int(action.target_code), action.action_id)
            if prior is not None and prior != assignment:
                raise V31CRunError(f"selected actions overlap at {point}")
            overlays[owner][point] = assignment
    return overlays


def _validate_published_stage(
    directory: Path,
    *,
    partition_id: str,
    fingerprint: str,
    shape: tuple[int, int],
) -> dict[str, Any]:
    audit_path = directory / "audit.json"
    hashes_path = directory / "outputs_sha256.json"
    if not audit_path.is_file() or not hashes_path.is_file():
        raise V31CRunError(f"resume C stage incomplete: {directory}")
    audit = _read_json(audit_path)
    hashes = _read_json(hashes_path).get("files")
    if (
        audit.get("partition_id") != partition_id
        or audit.get("execution_fingerprint_sha256") != fingerprint
        or not isinstance(hashes, dict)
        or set(hashes) != {"v31c_core.npy", "audit.json"}
    ):
        raise V31CRunError(f"resume C stage identity differs: {directory}")
    if any(_sha256_file(directory / name) != value for name, value in hashes.items()):
        raise V31CRunError(f"resume C stage SHA-256 mismatch: {directory}")
    B_RUNNER._verify_npy(
        directory / "v31c_core.npy",
        hashes["v31c_core.npy"],
        shape,
        np.dtype("int16"),
        f"resume C {partition_id}",
    )
    return audit


def _publish_partition(job: Mapping[str, Any]) -> dict[str, Any]:
    partition_id = str(job["partition_id"])
    part = job["b_part"]
    directory = Path(job["output_root"]) / "partitions" / partition_id / "stage_v31c"
    shape = B_RUNNER._shape(part["global_core_window"])
    if directory.exists():
        audit = _validate_published_stage(
            directory,
            partition_id=partition_id,
            fingerprint=job["fingerprint"],
            shape=shape,
        )
        return {
            "partition_id": partition_id,
            "audit": audit,
            "audit_path": str((directory / "audit.json").resolve()),
            "audit_sha256": _sha256_file(directory / "audit.json"),
            "output_path": str((directory / "v31c_core.npy").resolve()),
            "output_sha256": _sha256_file(directory / "v31c_core.npy"),
        }
    output = np.load(part["outputs"]["v31a"]["path"], allow_pickle=False).copy()
    valid = np.load(part["outputs"]["valid"]["path"], mmap_mode="r", allow_pickle=False)
    window = part["global_core_window"]
    action_ids: set[str] = set()
    for point, (target_code, action_id) in sorted(job["overlay"].items()):
        row, col = _local_point(window, point)
        if not valid[row, col] or output[row, col] < 0:
            raise V31CRunError(f"{partition_id}: selected overlay targets an invalid pixel {point}")
        output[row, col] = int(target_code)
        action_ids.add(action_id)
    if np.any(output[~valid] != -1) or np.any(~np.isin(output[valid], CLASS_ORDER)):
        raise V31CRunError(f"{partition_id}: C publication violates single-label coverage")
    staging = Path(job["staging_root"]) / partition_id
    staging.mkdir(parents=True, exist_ok=False)
    _save_npy(staging / "v31c_core.npy", output)
    output_sha = _sha256_file(staging / "v31c_core.npy")
    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "v31c_publish",
        "candidate_label": "C",
        "partition_id": partition_id,
        "execution_fingerprint_sha256": job["fingerprint"],
        "v31b_manifest": job["b_manifest_path"],
        "v31b_manifest_sha256": job["b_manifest_sha256"],
        "selection_plan_sha256": job["selection_plan_sha256"],
        "global_core_window": window,
        "core_transform": part["core_transform"],
        "crs": part["crs"],
        "physical_metrics": part["physical_metrics"],
        "selected_action_ids": sorted(action_ids),
        "selected_action_pixel_count": len(job["overlay"]),
        "coverage": {
            "published_strict_core_only": True,
            "valid_pixel_count": int(valid.sum()),
            "single_label": True,
            "gap_pixels": 0,
            "overlap_pixels": 0,
            "outside_pixels": 0,
        },
        "outputs": {
            "v31a": {
                "candidate_label": "C",
                "path": "v31c_core.npy",
                "sha256": output_sha,
                "shape": list(output.shape),
                "dtype": "int16",
            }
        },
    }
    audit["audit_sha256"] = _sha256_json(audit)
    _atomic_json(staging / "audit.json", audit)
    hashes = {
        "v31c_core.npy": output_sha,
        "audit.json": _sha256_file(staging / "audit.json"),
    }
    _atomic_json(staging / "outputs_sha256.json", {"files": hashes})
    directory.parent.mkdir(parents=True, exist_ok=True)
    os.rename(staging, directory)
    return {
        "partition_id": partition_id,
        "audit": audit,
        "audit_path": str((directory / "audit.json").resolve()),
        "audit_sha256": hashes["audit.json"],
        "output_path": str((directory / "v31c_core.npy").resolve()),
        "output_sha256": output_sha,
    }


def _publish_all(
    *,
    selected: Sequence[PlannedAction],
    source_entries: Sequence[Mapping[str, Any]],
    b_parts: Mapping[str, Mapping[str, Any]],
    output_root: Path,
    workers: int,
    fingerprint: str,
    selection_plan_sha256: str,
    b_manifest_path: str,
    b_manifest_sha256: str,
) -> list[dict[str, Any]]:
    overlays = _selected_overlay_by_core(selected, source_entries)
    staging_root = output_root / f".publish-staging-{os.getpid()}-{uuid.uuid4().hex}"
    staging_root.mkdir(exist_ok=False)
    completed: dict[str, dict[str, Any]] = {}
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _publish_partition,
                    {
                        "partition_id": partition_id,
                        "b_part": b_parts[partition_id],
                        "overlay": overlays[partition_id],
                        "output_root": str(output_root),
                        "staging_root": str(staging_root),
                        "fingerprint": fingerprint,
                        "selection_plan_sha256": selection_plan_sha256,
                        "b_manifest_path": b_manifest_path,
                        "b_manifest_sha256": b_manifest_sha256,
                    },
                ): partition_id
                for partition_id in sorted(b_parts)
            }
            for future in as_completed(futures):
                result = future.result()
                completed[result["partition_id"]] = result
                print(
                    f"v31c_publish {len(completed)}/{len(b_parts)} {result['partition_id']}",
                    flush=True,
                )
    finally:
        for child in staging_root.iterdir():
            if child.is_dir():
                import shutil

                shutil.rmtree(child, ignore_errors=True)
        staging_root.rmdir()
    return [completed[key] for key in sorted(completed)]


def _evaluation_manifest(
    *,
    output_root: Path,
    published: Sequence[Mapping[str, Any]],
    b_parts: Mapping[str, Mapping[str, Any]],
    b_data: Mapping[str, Any],
    policy: Any,
    selection_plan_sha256: str,
) -> Path:
    publication_by_id = {str(item["partition_id"]): item for item in published}
    parts = []
    for partition_id in sorted(b_parts):
        b_part = b_parts[partition_id]
        publication = publication_by_id[partition_id]
        parts.append(
            {
                "partition_id": partition_id,
                "candidate_label": "C",
                "global_core_window": b_part["global_core_window"],
                "core_transform": b_part["core_transform"],
                "crs": b_part["crs"],
                "physical_metrics": b_part["physical_metrics"],
                "outputs": {
                    "raw": b_part["outputs"]["raw"],
                    "v3": b_part["outputs"]["v3"],
                    "valid": b_part["outputs"]["valid"],
                    "v31a": {
                        "candidate_label": "C",
                        "path": publication["output_path"],
                        "sha256": publication["output_sha256"],
                        "shape": list(B_RUNNER._shape(b_part["global_core_window"])),
                        "dtype": "int16",
                    },
                },
                "audit": {
                    "path": publication["audit_path"],
                    "sha256": publication["audit_sha256"],
                },
            }
        )
    mmu = {
        str(code): float(row.dynamic_fragmentation_m2)
        for code, row in sorted(policy.class_policies.items())
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "v31c_global_evaluation_input",
        "candidate_label": "C",
        "v31b_manifest": b_data["manifest_path"],
        "v31b_manifest_sha256": b_data["manifest_file_sha256"],
        "selection_plan_sha256": selection_plan_sha256,
        "approved_dynamic_mmu_m2": mmu,
        "processing_transform": b_data["manifest"]["processing_transform"],
        "crs": b_data["manifest"]["crs"],
        "global_window": b_data["manifest"]["global_window"],
        "partitions": parts,
    }
    manifest["manifest_sha256"] = _sha256_json(manifest)
    path = output_root / "evaluation_manifest.json"
    _atomic_json(path, manifest)
    return path


def _hard_coverage_gate(result: Mapping[str, Any]) -> dict[str, bool]:
    coverage = result["coverage"]
    return {
        "core_overlap_zero": int(coverage["core_overlap_pixels"]) == 0,
        "geometric_gap_zero": int(coverage["geometric_coverage_gap_pixels"]) == 0,
        "outside_zero": all(int(value) == 0 for value in coverage["outside_valid_label_pixels"].values()),
        "invalid_inside_zero": all(int(value) == 0 for value in coverage["invalid_label_inside_valid_pixels"].values()),
    }


def _method_summary(method: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "components_total": int(method["components_total"]),
        "dynamic_fragment_count": int(method["dynamic_fragments"]["count"]),
        "dynamic_fragment_area_m2": float(method["dynamic_fragments"]["area_m2"]),
        "cross_class_boundary_m": float(
            method["boundary"]["total_cross_class_boundary"]["metres"]
        ),
    }


def _validate_global_results(
    *,
    b_result: Mapping[str, Any],
    c_result: Mapping[str, Any],
) -> dict[str, Any]:
    b = _method_summary(b_result["methods"]["v31"])
    c = _method_summary(c_result["methods"]["v31"])
    coverage = _hard_coverage_gate(c_result)
    b_per_class = {int(item["class_code"]): item for item in b_result["methods"]["v31"]["per_class"]}
    c_per_class = {int(item["class_code"]): item for item in c_result["methods"]["v31"]["per_class"]}
    per_class_components_nonincreasing = all(
        int(c_per_class[code]["components"]) <= int(b_per_class[code]["components"])
        for code in b_per_class
    )
    return {
        "coverage": coverage,
        "hard_coverage_pass": all(coverage.values()),
        "b": b,
        "c": c,
        "c_minus_b": {key: c[key] - b[key] for key in b},
        "fragmentation_dominance": {
            "components_nonincreasing": c["components_total"] <= b["components_total"],
            "dynamic_count_nonincreasing": c["dynamic_fragment_count"] <= b["dynamic_fragment_count"],
            "dynamic_area_nonincreasing": c["dynamic_fragment_area_m2"] <= b["dynamic_fragment_area_m2"] + 1e-7,
            "strict_component_improvement": c["components_total"] < b["components_total"],
            "per_class_components_nonincreasing": per_class_components_nonincreasing,
        },
        "boundary_delta_is_report_only": True,
    }


def _verify_completed_artifacts(manifest: Mapping[str, Any]) -> None:
    if not manifest.get("validation_pass"):
        raise V31CRunError("completed C manifest lacks a passing global validation")
    references = [
        (
            (manifest.get("selection_plan") or {}).get("path"),
            (manifest.get("selection_plan") or {}).get("sha256"),
            "selection plan",
        ),
        (
            (manifest.get("evaluation_manifest") or {}).get("path"),
            (manifest.get("evaluation_manifest") or {}).get("sha256"),
            "evaluation manifest",
        ),
        (
            (manifest.get("global_evaluation") or {}).get("result"),
            (manifest.get("global_evaluation") or {}).get("result_sha256"),
            "global evaluation result",
        ),
        (
            (manifest.get("global_evaluation") or {}).get("audit"),
            (manifest.get("global_evaluation") or {}).get("audit_sha256"),
            "global evaluation audit",
        ),
        (
            (manifest.get("baseline_b_global_evaluation") or {}).get("result"),
            (manifest.get("baseline_b_global_evaluation") or {}).get("result_sha256"),
            "isolated B baseline evaluation result",
        ),
        (
            (manifest.get("baseline_b_global_evaluation") or {}).get("audit"),
            (manifest.get("baseline_b_global_evaluation") or {}).get("audit_sha256"),
            "isolated B baseline evaluation audit",
        ),
    ]
    partitions = manifest.get("partitions") or []
    if len(partitions) != int(manifest.get("completed_partition_count", -1)):
        raise V31CRunError("completed C partition count does not close")
    for part in partitions:
        partition_id = str(part.get("partition_id", ""))
        output = (part.get("outputs") or {}).get("v31a") or {}
        audit = part.get("audit") or {}
        references.extend(
            (
                (output.get("path"), output.get("sha256"), f"{partition_id} C output"),
                (audit.get("path"), audit.get("sha256"), f"{partition_id} C audit"),
            )
        )
    for raw_path, expected_sha, label in references:
        if not isinstance(raw_path, str) or not isinstance(expected_sha, str):
            raise V31CRunError(f"completed C {label} reference is incomplete")
        path = Path(raw_path)
        if not path.is_file() or _sha256_file(path) != expected_sha:
            raise V31CRunError(f"completed C {label} SHA-256 mismatch")


def _compatible_collect_fingerprint(
    prior_manifest: Mapping[str, Any], current_payload: Mapping[str, Any]
) -> str:
    """Allow one audited runner-only fix to reuse immutable collect shards."""

    prior_payload = prior_manifest.get("execution_fingerprint")
    prior_fingerprint = prior_manifest.get("execution_fingerprint_sha256")
    if (
        prior_manifest.get("status") != "running"
        or not isinstance(prior_payload, dict)
        or not isinstance(prior_fingerprint, str)
        or _sha256_json(prior_payload) != prior_fingerprint
    ):
        raise V31CRunError("resume predecessor is not a valid running C fingerprint")
    runner_key = "scratch/v31c_full_140_20260825/run_v31c_global.py"
    prior_code = prior_payload.get("code_sha256")
    current_code = current_payload.get("code_sha256")
    if not isinstance(prior_code, dict) or not isinstance(current_code, dict):
        raise V31CRunError("resume predecessor code fingerprint is incomplete")
    prior_runner_sha = prior_code.get(runner_key)
    if prior_runner_sha not in COLLECT_COMPATIBLE_PREDECESSOR_RUNNER_SHA256:
        raise V31CRunError("resume predecessor runner is not collect-compatible")
    if set(prior_code) != set(current_code) or any(
        prior_code[key] != current_code[key] for key in prior_code if key != runner_key
    ):
        raise V31CRunError("resume predecessor changed collection dependencies")
    prior_without_code = {key: value for key, value in prior_payload.items() if key != "code_sha256"}
    current_without_code = {key: value for key, value in current_payload.items() if key != "code_sha256"}
    if prior_without_code != current_without_code:
        raise V31CRunError("resume predecessor changed non-code collection inputs")
    return prior_fingerprint


def run(
    b_manifest_path: Path,
    output_root: Path,
    *,
    workers: int,
    resume: bool,
    self_test: bool = False,
) -> dict[str, Any]:
    if workers < 1 or workers > 2:
        raise V31CRunError("--workers must be 1 or 2")
    b_data = _load_b_manifest(b_manifest_path.resolve(), self_test=self_test)
    if output_root.exists() and any(output_root.iterdir()) and not resume:
        raise V31CRunError(f"refusing non-empty output root without --resume: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    policy = replace(
        v31b_policy(),
        policy_id=V31C_POLICY_ID,
        policy_version=V31C_POLICY_VERSION,
    )
    c_policy_snapshot = policy_snapshot(policy)
    c_policy_snapshot["coordination_mode"] = V31C_COORDINATION_MODE
    c_policy_snapshot["algorithm_contract"]["global_coordination"] = (
        "collect_global_component_recompute_conservative_conflict_graph_lexicographic_milp_publish"
    )
    c_policy_sha = _sha256_json(c_policy_snapshot)
    code_sha = _code_sha256()
    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION,
        "candidate_label": "C",
        "v31b_manifest_sha256": b_data["manifest_file_sha256"],
        "v31c_policy_snapshot_sha256": c_policy_sha,
        "coordination_mode": V31C_COORDINATION_MODE,
        "code_sha256": code_sha,
    }
    fingerprint = _sha256_json(fingerprint_payload)
    manifest_path = output_root / "run_manifest.json"
    compatible_collect_fingerprints: list[str] = []
    if resume and manifest_path.is_file():
        prior = _read_json(manifest_path)
        if prior.get("status") == "complete":
            if prior.get("execution_fingerprint_sha256") != fingerprint:
                raise V31CRunError("resume execution fingerprint differs")
            declared = prior.get("manifest_sha256")
            if declared != _sha256_json(
                {key: value for key, value in prior.items() if key != "manifest_sha256"}
            ):
                raise V31CRunError("completed C manifest self SHA-256 mismatch")
            if _sha256_file(b_manifest_path) != b_data["manifest_file_sha256"]:
                raise V31CRunError("B manifest changed after completed C run")
            _verify_completed_artifacts(prior)
            return prior
        if prior.get("execution_fingerprint_sha256") != fingerprint:
            compatible_collect_fingerprints.append(
                _compatible_collect_fingerprint(prior, fingerprint_payload)
            )
    running: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "v31c_cross_core_global_candidate",
        "candidate_label": "C",
        "status": "running",
        "self_test": bool(self_test),
        "v31b_manifest": b_data["manifest_path"],
        "v31b_manifest_sha256": b_data["manifest_file_sha256"],
        "execution_fingerprint": fingerprint_payload,
        "execution_fingerprint_sha256": fingerprint,
        "v31c_policy_snapshot": c_policy_snapshot,
        "v31c_policy_snapshot_sha256": c_policy_sha,
        "coordination_mode": V31C_COORDINATION_MODE,
        "compatible_collect_predecessor_fingerprints": compatible_collect_fingerprints,
        "requested_partition_count": len(b_data["source"]["entries"]),
        "completed_partition_count": 0,
    }
    _atomic_json(manifest_path, running)
    started = time.monotonic()
    discoveries, collect_shards = _collect_all(
        b_data["source"],
        output_root=output_root,
        workers=workers,
        fingerprint=fingerprint,
        policy=policy,
        compatible_collect_fingerprints=compatible_collect_fingerprints,
    )
    selected, plan_audit, canonical_actions = _prepare_global_plan(
        discoveries=discoveries,
        source_entries=b_data["source"]["entries"],
        b_parts=b_data["parts"],
        global_window=b_data["source"]["global_window"],
        policy=policy,
    )
    selected_ledgers = [
        {
            **global_action_to_dict(item.action),
            "source_component_keys": list(item.source_component_keys),
            "target_component_keys": list(item.target_component_keys),
            "source_charges": [list(value) for value in item.source_charges],
            "target_charges": [list(value) for value in item.target_charges],
        }
        for item in selected
    ]
    selection_plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "v31c_global_plan",
        "execution_fingerprint_sha256": fingerprint,
        "v31b_manifest_sha256": b_data["manifest_file_sha256"],
        "v31c_policy_snapshot_sha256": c_policy_sha,
        "coordination_mode": V31C_COORDINATION_MODE,
        "canonical_actions": [global_action_to_dict(action) for action in canonical_actions],
        "selected_actions": selected_ledgers,
        "audit": plan_audit,
    }
    selection_plan["selection_plan_sha256"] = _sha256_json(selection_plan)
    selection_path = output_root / "selection_plan.json"
    if selection_path.is_file():
        prior_plan = _read_json(selection_path)
        if prior_plan != selection_plan:
            raise V31CRunError("resume global selection plan differs")
    else:
        _atomic_json(selection_path, selection_plan)
    selection_file_sha = _sha256_file(selection_path)
    published = _publish_all(
        selected=selected,
        source_entries=b_data["source"]["entries"],
        b_parts=b_data["parts"],
        output_root=output_root,
        workers=workers,
        fingerprint=fingerprint,
        selection_plan_sha256=selection_plan["selection_plan_sha256"],
        b_manifest_path=b_data["manifest_path"],
        b_manifest_sha256=b_data["manifest_file_sha256"],
    )
    evaluation_manifest = _evaluation_manifest(
        output_root=output_root,
        published=published,
        b_parts=b_data["parts"],
        b_data=b_data,
        policy=policy,
        selection_plan_sha256=selection_plan["selection_plan_sha256"],
    )
    b_evaluation = EVALUATOR.evaluate(
        Path(b_data["manifest_path"]),
        output_root / "baseline_b_global_evaluation",
        resume=True,
    )
    c_evaluation = EVALUATOR.evaluate(
        evaluation_manifest,
        output_root / "global_evaluation",
        resume=True,
    )
    validation = _validate_global_results(
        b_result=b_evaluation["result"], c_result=c_evaluation["result"]
    )
    hard_fragmentation = validation["fragmentation_dominance"]
    validation_pass = (
        validation["hard_coverage_pass"]
        and hard_fragmentation["components_nonincreasing"]
        and hard_fragmentation["dynamic_count_nonincreasing"]
        and hard_fragmentation["dynamic_area_nonincreasing"]
        and hard_fragmentation["per_class_components_nonincreasing"]
    )
    if _sha256_file(b_manifest_path) != b_data["manifest_file_sha256"]:
        raise V31CRunError("B manifest changed during C run")
    manifest: dict[str, Any] = {
        **running,
        "status": "complete" if validation_pass else "rejected_validation",
        "completed_partition_count": len(published),
        "code_sha256": code_sha,
        "class_codes": list(CLASS_ORDER),
        "label_encoding": "class_codes_int16_invalid_minus_one",
        "processing_transform": b_data["manifest"]["processing_transform"],
        "crs": b_data["manifest"]["crs"],
        "global_window": b_data["manifest"]["global_window"],
        "collect": {
            "completed_partition_count": len(collect_shards),
            "discovery_occurrence_count": len(discoveries),
            "runtime_seconds": float(
                sum(float(item.get("runtime_seconds", 0.0)) for item in collect_shards)
            ),
            "compatible_predecessor_fingerprints": compatible_collect_fingerprints,
        },
        "selection_plan": {
            "path": str(selection_path.resolve()),
            "sha256": selection_file_sha,
            "semantic_sha256": selection_plan["selection_plan_sha256"],
            "canonical_action_count": len(canonical_actions),
            "selected_action_count": len(selected),
        },
        "evaluation_manifest": {
            "path": str(evaluation_manifest.resolve()),
            "sha256": _sha256_file(evaluation_manifest),
        },
        "global_evaluation": {
            "result": str((output_root / "global_evaluation" / "global_fragmentation.json").resolve()),
            "result_sha256": _sha256_file(output_root / "global_evaluation" / "global_fragmentation.json"),
            "audit": str((output_root / "global_evaluation" / "audit.json").resolve()),
            "audit_sha256": _sha256_file(output_root / "global_evaluation" / "audit.json"),
        },
        "baseline_b_global_evaluation": {
            "result": str((output_root / "baseline_b_global_evaluation" / "global_fragmentation.json").resolve()),
            "result_sha256": _sha256_file(output_root / "baseline_b_global_evaluation" / "global_fragmentation.json"),
            "audit": str((output_root / "baseline_b_global_evaluation" / "audit.json").resolve()),
            "audit_sha256": _sha256_file(output_root / "baseline_b_global_evaluation" / "audit.json"),
        },
        "validation": validation,
        "validation_pass": bool(validation_pass),
        "runtime_seconds": time.monotonic() - started,
        "partitions": [
            {
                "partition_id": item["partition_id"],
                "global_core_window": b_data["parts"][item["partition_id"]]["global_core_window"],
                "core_transform": b_data["parts"][item["partition_id"]]["core_transform"],
                "crs": b_data["parts"][item["partition_id"]]["crs"],
                "physical_metrics": b_data["parts"][item["partition_id"]]["physical_metrics"],
                "outputs": {
                    "raw": b_data["parts"][item["partition_id"]]["outputs"]["raw"],
                    "v3": b_data["parts"][item["partition_id"]]["outputs"]["v3"],
                    "valid": b_data["parts"][item["partition_id"]]["outputs"]["valid"],
                    "v31a": {
                        "candidate_label": "C",
                        "path": item["output_path"],
                        "sha256": item["output_sha256"],
                        "shape": list(B_RUNNER._shape(b_data["parts"][item["partition_id"]]["global_core_window"])),
                        "dtype": "int16",
                    },
                },
                "audit": {"path": item["audit_path"], "sha256": item["audit_sha256"]},
            }
            for item in published
        ],
    }
    manifest["manifest_sha256"] = _sha256_json(manifest)
    _atomic_json(manifest_path, manifest)
    if not validation_pass:
        raise V31CRunError(
            "C global validation rejected the publication; see run_manifest.json validation"
        )
    return manifest


def _self_test(output_root: Path | None, workers: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v31c-selftest-") as temporary:
        root = Path(temporary)
        parent = B_RUNNER._write_self_test_parent(root)
        b_root = root / "b"
        B_RUNNER.run(parent, b_root, workers=workers, resume=False, self_test=True)
        target = root / "c" if output_root is None else output_root
        result = run(
            b_root / "run_manifest.json",
            target,
            workers=workers,
            resume=False,
            self_test=True,
        )
        if (
            result.get("status") != "complete"
            or result.get("candidate_label") != "C"
            or result.get("completed_partition_count") != 4
            or not result.get("validation_pass")
        ):
            raise V31CRunError("self-test C did not complete four valid Core publications")
        first_sha = result["manifest_sha256"]
        resumed = run(
            b_root / "run_manifest.json",
            target,
            workers=workers,
            resume=True,
            self_test=True,
        )
        if resumed["manifest_sha256"] != first_sha:
            raise V31CRunError("self-test completed resume changed C manifest")
        return resumed


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v31b-manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        if args.v31b_manifest is not None:
            parser.error("--self-test creates an immutable four-Core B fixture")
    elif args.v31b_manifest is None or args.output_root is None:
        parser.error("real execution requires --v31b-manifest and --output-root")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    try:
        result = (
            _self_test(args.output_root, args.workers)
            if args.self_test
            else run(
                args.v31b_manifest,
                args.output_root,
                workers=args.workers,
                resume=args.resume,
                self_test=False,
            )
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "candidate_label": "C",
                    "completed_partition_count": result["completed_partition_count"],
                    "selected_action_count": result["selection_plan"]["selected_action_count"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0
    except (V31CRunError, component_api.ComponentIndexError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
