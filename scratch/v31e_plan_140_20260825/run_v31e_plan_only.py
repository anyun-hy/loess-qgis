#!/usr/bin/env python3
"""Build an isolated V3.1-E global plan without publishing a raster.

The runner regenerates exact B-relative proposals on all frozen probability
parts, rebuilds global B component identities, computes exact cross-class
boundary deltas and solves three separate plans:

1. a dependency-relaxed independent-action optimum without a boundary constraint;
2. a dependency-relaxed boundary-nonincreasing independent-action optimum;
3. a strict plan that retains D's complete B-affected-component lock.

The relaxed plans are not global-method upper bounds.  They cannot be published
until the accepted B action ledger has been regenerated and replayed exactly.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "inference_scripts"))

from deployment_config import CLASS_ORDER  # noqa: E402
from fragmentation_v31_candidate.candidate import (  # noqa: E402
    policy_snapshot,
    v31b_policy,
)
from fragmentation_v31_candidate.v31c import (  # noqa: E402
    CrossCoreDiscovery,
    GlobalAction,
    global_action_to_dict,
)
from fragmentation_v31_candidate import v31c_components as component_api  # noqa: E402
from fragmentation_v31_candidate.v31e import (  # noqa: E402
    BoundaryPlannedAction,
    V31E_COORDINATION_MODE,
    V31E_POLICY_ID,
    V31E_POLICY_VERSION,
    canonicalise_discoveries,
    collect_global_b_discoveries,
    exact_boundary_delta,
    select_boundary_aware_actions,
)


C_RUNNER_PATH = (
    REPO_ROOT / "scratch" / "v31c_full_140_20260825" / "run_v31c_global.py"
)
D_RUNNER_PATH = (
    REPO_ROOT
    / "scratch"
    / "v31d_full_140_20260825"
    / "run_v31d_second_round.py"
)
SCHEMA_VERSION = 1
REAL_PARTITION_COUNT = 140
REQUIRED_DYNAMIC_REDUCTION = 130
ENGINEERING_HEADROOM_REDUCTION = 150


class V31EPlanRunError(RuntimeError):
    """The plan-only run failed an integrity or proof contract."""


def _load_module(name: str, path: Path) -> Any:
    if not path.is_file():
        raise V31EPlanRunError(f"required helper is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise V31EPlanRunError(f"cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C_RUNNER = _load_module("_v31e_frozen_c_runner", C_RUNNER_PATH)
D_RUNNER = _load_module("_v31e_frozen_d_runner", D_RUNNER_PATH)
B_RUNNER = C_RUNNER.B_RUNNER


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
        raise V31EPlanRunError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V31EPlanRunError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _code_sha256() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        C_RUNNER_PATH.resolve(),
        D_RUNNER_PATH.resolve(),
        C_RUNNER.B_RUNNER_PATH.resolve(),
        C_RUNNER.EVALUATOR_PATH.resolve(),
        REPO_ROOT / "inference_scripts" / "deployment_config.py",
        REPO_ROOT / "inference_scripts" / "small_component_regularizer.py",
        REPO_ROOT
        / "inference_scripts"
        / "fragmentation_v31_candidate"
        / "__init__.py",
        REPO_ROOT
        / "inference_scripts"
        / "fragmentation_v31_candidate"
        / "candidate.py",
        REPO_ROOT
        / "inference_scripts"
        / "fragmentation_v31_candidate"
        / "v31c.py",
        REPO_ROOT
        / "inference_scripts"
        / "fragmentation_v31_candidate"
        / "v31c_components.py",
        REPO_ROOT
        / "inference_scripts"
        / "fragmentation_v31_candidate"
        / "v31d.py",
        REPO_ROOT
        / "inference_scripts"
        / "fragmentation_v31_candidate"
        / "v31e.py",
    )
    return {
        str(path.relative_to(REPO_ROOT)): _sha256_file(path) for path in paths
    }


def _discovery_from_dict(value: Mapping[str, Any]) -> CrossCoreDiscovery:
    return C_RUNNER._discovery_from_dict(value)


def _collect_partition(job: Mapping[str, Any]) -> dict[str, Any]:
    entry = job["entry"]
    partition_id = str(entry["partition_id"])
    shard_path = Path(job["collect_root"]) / f"{partition_id}.json"
    if shard_path.is_file():
        prior = _read_json(shard_path)
        body = {key: value for key, value in prior.items() if key != "shard_sha256"}
        if (
            prior.get("execution_fingerprint_sha256") == job["fingerprint"]
            and prior.get("shard_sha256") == _sha256_json(body)
        ):
            return prior
        raise V31EPlanRunError(f"collect resume shard differs: {shard_path}")
    _v3, probabilities, _decoder_valid, strict, expanded, _owners = B_RUNNER._stitch(
        entry, job["entries"], job["global_window"]
    )
    b_context, b_expanded = D_RUNNER._stitch_b(
        entry, job["entries"], job["b_parts"], job["global_window"]
    )
    if expanded != b_expanded:
        raise V31EPlanRunError(f"{partition_id}: B/probability windows differ")
    metrics = B_RUNNER._physical(job["transform"], job["crs"], expanded)
    started = time.monotonic()

    def owner_for(row: int, col: int) -> str | None:
        if not (
            int(expanded["y0"]) <= int(row) < int(expanded["y1"])
            and int(expanded["x0"]) <= int(col) < int(expanded["x1"])
        ):
            return None
        local_row = int(row) - int(expanded["y0"])
        local_col = int(col) - int(expanded["x0"])
        if not bool(strict[local_row, local_col]):
            return None
        return C_RUNNER._owner_for(job["entries"], int(row), int(col))

    discoveries, audit = collect_global_b_discoveries(
        b_context,
        class_codes=CLASS_ORDER,
        pixel_area_m2=metrics["pixel_area_m2"],
        pixel_size_m=(metrics["row_step_m"], metrics["column_step_m"]),
        valid_mask=strict,
        probabilities=probabilities,
        confidence=probabilities.max(axis=0).astype(np.float32),
        global_origin=(int(expanded["y0"]), int(expanded["x0"])),
        discovery_partition_id=partition_id,
        owner_for_global_pixel=owner_for,
        policy=job["policy"],
    )
    shard: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "v31e_plan_collect",
        "partition_id": partition_id,
        "execution_fingerprint_sha256": job["fingerprint"],
        "global_core_window": entry["core_window"],
        "global_expanded_window": expanded,
        "runtime_seconds": time.monotonic() - started,
        "audit": audit,
        "discoveries": [asdict(item) for item in discoveries],
    }
    shard["shard_sha256"] = _sha256_json(shard)
    _atomic_json(shard_path, shard)
    return shard


def _collect_all(
    b_data: Mapping[str, Any],
    *,
    output_root: Path,
    workers: int,
    fingerprint: str,
    policy: Any,
) -> tuple[list[CrossCoreDiscovery], list[dict[str, Any]]]:
    collect_root = output_root / "collect"
    collect_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "entries": b_data["source"]["entries"],
        "global_window": b_data["source"]["global_window"],
        "transform": b_data["source"]["transform"],
        "crs": b_data["source"]["crs"],
        "b_parts": b_data["parts"],
        "collect_root": str(collect_root),
        "fingerprint": fingerprint,
        "policy": policy,
    }
    shards: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_collect_partition, {**payload, "entry": entry}): str(
                entry["partition_id"]
            )
            for entry in b_data["source"]["entries"]
        }
        for future in as_completed(futures):
            shard = future.result()
            shards[str(shard["partition_id"])] = shard
            print(
                f"v31e_collect {len(shards)}/{len(futures)} {shard['partition_id']}",
                flush=True,
            )
    discoveries = [
        _discovery_from_dict(value)
        for partition_id in sorted(shards)
        for value in shards[partition_id]["discoveries"]
    ]
    return discoveries, [shards[key] for key in sorted(shards)]


class _GlobalBLookup:
    def __init__(
        self,
        entries: Sequence[Mapping[str, Any]],
        parts: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.entries = entries
        self.parts = parts
        self.labels: dict[str, np.ndarray] = {}
        self.valid_values: dict[str, np.ndarray] = {}

    def owner(self, point: tuple[int, int]) -> str | None:
        return C_RUNNER._owner_for(self.entries, int(point[0]), int(point[1]))

    def _local(self, owner: str, point: tuple[int, int]) -> tuple[int, int]:
        window = self.parts[owner]["global_core_window"]
        return (
            int(point[0]) - int(window["y0"]),
            int(point[1]) - int(window["x0"]),
        )

    def valid(self, point: tuple[int, int]) -> bool:
        owner = self.owner(point)
        if owner is None:
            return False
        values = self.valid_values.setdefault(
            owner,
            np.load(
                self.parts[owner]["outputs"]["valid"]["path"],
                mmap_mode="r",
                allow_pickle=False,
            ),
        )
        row, col = self._local(owner, point)
        return bool(values[row, col])

    def label(self, point: tuple[int, int]) -> int:
        owner = self.owner(point)
        if owner is None:
            raise V31EPlanRunError(f"global pixel {point} has no owner")
        if not self.valid(point):
            raise V31EPlanRunError(f"global pixel {point} is not strict-valid")
        values = self.labels.setdefault(
            owner,
            np.load(
                self.parts[owner]["outputs"]["v31a"]["path"],
                mmap_mode="r",
                allow_pickle=False,
            ),
        )
        row, col = self._local(owner, point)
        return int(values[row, col])


def _b_changed_points(b_parts: Mapping[str, Mapping[str, Any]]) -> set[tuple[int, int]]:
    changed: set[tuple[int, int]] = set()
    for partition_id in sorted(b_parts):
        part = b_parts[partition_id]
        v3 = np.load(part["outputs"]["v3"]["path"], mmap_mode="r", allow_pickle=False)
        b = np.load(part["outputs"]["v31a"]["path"], mmap_mode="r", allow_pickle=False)
        valid = np.load(part["outputs"]["valid"]["path"], mmap_mode="r", allow_pickle=False)
        rows, cols = np.nonzero(valid & (v3 != b))
        window = part["global_core_window"]
        changed.update(
            (
                int(window["y0"]) + int(row),
                int(window["x0"]) + int(col),
            )
            for row, col in zip(rows.tolist(), cols.tolist())
        )
    return changed


def _component_key_text(value: Any) -> str:
    key = value.key
    return f"{int(key.class_code)}:{int(key.min_row)}:{int(key.min_col)}"


def _global_b_component_lookup(
    *,
    query_points: Sequence[tuple[int, int]],
    b_parts: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, dict[str, Any]], int]:
    if not query_points:
        return {}, {}, 0
    tiles = []
    for partition_id in sorted(b_parts):
        part = b_parts[partition_id]
        window = part["global_core_window"]
        tiles.append(
            component_api.CoreTile(
                core_id=partition_id,
                window=(
                    int(window["y0"]),
                    int(window["x0"]),
                    int(window["y1"]) - int(window["y0"]),
                    int(window["x1"]) - int(window["x0"]),
                ),
                labels=np.load(
                    part["outputs"]["v31a"]["path"],
                    mmap_mode="r",
                    allow_pickle=False,
                ),
                valid=np.load(
                    part["outputs"]["valid"]["path"],
                    mmap_mode="r",
                    allow_pickle=False,
                ),
                pixel_area_m2=float(part["physical_metrics"]["pixel_area_m2"]),
            )
        )
    ordered = tuple(sorted(set(query_points)))
    result = component_api.build_global_component_index(tiles, ordered)
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
            raise V31EPlanRunError(f"component metadata changed for {key}")
        by_key[key] = info
    if set(by_point) != set(ordered):
        raise V31EPlanRunError("global B component index omitted query points")
    return by_point, by_key, int(result.global_component_count)


def _planned_action_dict(value: BoundaryPlannedAction) -> dict[str, Any]:
    return {
        **global_action_to_dict(value.action),
        "source_component_keys": list(value.source_component_keys),
        "target_component_keys": list(value.target_component_keys),
        "source_charges": [list(item) for item in value.source_charges],
        "target_charges": [list(item) for item in value.target_charges],
        "component_delta_by_class": {
            str(code): delta for code, delta in value.component_delta_by_class
        },
        "dynamic_delta_by_class": {
            str(code): delta for code, delta in value.dynamic_delta_by_class
        },
        "boundary_delta_edges": value.boundary_delta_edges,
        "boundary_delta_metres": value.boundary_delta_metres,
        "dependency_proof": value.dependency_proof,
    }


def _prepare_actions(
    *,
    discoveries: Sequence[CrossCoreDiscovery],
    b_data: Mapping[str, Any],
    policy: Any,
) -> tuple[list[BoundaryPlannedAction], dict[str, Any], list[GlobalAction]]:
    actions, duplicate_audit = canonicalise_discoveries(discoveries)
    lookup = _GlobalBLookup(b_data["source"]["entries"], b_data["parts"])
    b_changed = _b_changed_points(b_data["parts"])
    target_neighbours: dict[str, tuple[tuple[int, int], ...]] = {}
    query_points = set(b_changed)
    for action in actions:
        query_points.update(action.footprint)
        target_points: set[tuple[int, int]] = set()
        footprint = set(action.footprint)
        for row, col in action.footprint:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                point = (int(row) + dr, int(col) + dc)
                if point in footprint or not lookup.valid(point):
                    continue
                if lookup.label(point) == int(action.target_code):
                    target_points.add(point)
        target_neighbours[action.action_id] = tuple(sorted(target_points))
        query_points.update(target_points)
    point_info, component_info, global_component_count = _global_b_component_lookup(
        query_points=sorted(query_points), b_parts=b_data["parts"]
    )
    b_affected_components = {point_info[point]["key"] for point in b_changed}
    pixel_area_by_core = {
        partition_id: float(part["physical_metrics"]["pixel_area_m2"])
        for partition_id, part in b_data["parts"].items()
    }
    physical_by_core = {
        partition_id: part["physical_metrics"]
        for partition_id, part in b_data["parts"].items()
    }
    planned: list[BoundaryPlannedAction] = []
    rejected: dict[str, str] = {}
    for action in actions:
        footprint = tuple(action.footprint)
        if any(point in b_changed for point in footprint):
            rejected[action.action_id] = "b_changed_pixel_immutable"
            continue
        footprint_infos = [point_info[point] for point in footprint]
        if any(info["class_code"] == action.target_code for info in footprint_infos):
            rejected[action.action_id] = "frozen_action_no_longer_changes_target"
            continue
        actual_source_codes = tuple(
            sorted({int(info["class_code"]) for info in footprint_infos})
        )
        if actual_source_codes != tuple(sorted(action.source_codes)):
            rejected[action.action_id] = "global_source_code_mismatch"
            continue
        source_keys = tuple(sorted({str(info["key"]) for info in footprint_infos}))
        target_points = target_neighbours[action.action_id]
        if not target_points:
            rejected[action.action_id] = "global_target_attachment_missing"
            continue
        target_infos = [point_info[point] for point in target_points]
        target_keys = tuple(sorted({str(info["key"]) for info in target_infos}))
        if any(int(info["class_code"]) != int(action.target_code) for info in target_infos):
            raise V31EPlanRunError("target-neighbour class lookup changed")
        removed_pixels: Counter[str] = Counter(str(info["key"]) for info in footprint_infos)
        removed_area: Counter[str] = Counter()
        source_charges: Counter[tuple[str, int]] = Counter()
        target_charges: Counter[tuple[str, int]] = Counter()
        added_area = 0.0
        for point, info in zip(footprint, footprint_infos):
            owner = lookup.owner(point)
            if owner is None:
                raise V31EPlanRunError(f"{action.action_id}: footprint has no owner")
            area = pixel_area_by_core[owner]
            removed_area[str(info["key"])] += area
            source_charges[(owner, int(info["class_code"]))] += 1
            target_charges[(owner, int(action.target_code))] += 1
            added_area += area
        component_delta: Counter[int] = Counter()
        dynamic_delta: Counter[int] = Counter()
        for key in source_keys:
            info = component_info[key]
            code = int(info["class_code"])
            threshold = float(policy.class_policies[code].dynamic_fragmentation_m2)
            remaining_pixels = int(info["pixel_count"]) - int(removed_pixels[key])
            remaining_area = float(info["area_m2"]) - float(removed_area[key])
            if remaining_pixels < 0 or remaining_area < -1e-7:
                raise V31EPlanRunError(f"{action.action_id}: source removal underflow")
            before_dynamic = 0.0 < float(info["area_m2"]) < threshold
            after_dynamic = remaining_pixels > 0 and 0.0 < remaining_area < threshold
            dynamic_delta[code] += int(after_dynamic) - int(before_dynamic)
            if remaining_pixels == 0:
                component_delta[code] -= 1
        target_code = int(action.target_code)
        target_threshold = float(
            policy.class_policies[target_code].dynamic_fragmentation_m2
        )
        target_old_dynamic = sum(
            int(0.0 < float(component_info[key]["area_m2"]) < target_threshold)
            for key in target_keys
        )
        target_area = sum(float(component_info[key]["area_m2"]) for key in target_keys)
        target_new_dynamic = int(0.0 < target_area + added_area < target_threshold)
        dynamic_delta[target_code] += target_new_dynamic - target_old_dynamic
        component_delta[target_code] -= len(target_keys) - 1
        component_reduction = -int(sum(component_delta.values()))
        dynamic_reduction = -int(sum(dynamic_delta.values()))
        if component_reduction <= 0 or dynamic_reduction < 0:
            rejected[action.action_id] = "global_b_topology_score_no_gain"
            continue
        rescored = replace(
            action,
            component_reduction=component_reduction,
            dynamic_reduction=dynamic_reduction,
            area_m2=float(added_area),
        )
        boundary_edges, boundary_metres = exact_boundary_delta(
            rescored,
            label_for_global_pixel=lookup.label,
            valid_for_global_pixel=lookup.valid,
            owner_for_global_pixel=lookup.owner,
            physical_metrics_by_owner=physical_by_core,
        )
        touched = set(source_keys) | set(target_keys)
        dependency_proof = (
            "strict_b_component_lock_clear"
            if not touched & b_affected_components
            else "relaxed_requires_b_dependency_replay"
        )
        planned.append(
            BoundaryPlannedAction(
                action=rescored,
                source_component_keys=source_keys,
                target_component_keys=target_keys,
                source_charges=tuple(
                    (core, code, count)
                    for (core, code), count in sorted(source_charges.items())
                ),
                target_charges=tuple(
                    (core, code, count)
                    for (core, code), count in sorted(target_charges.items())
                ),
                component_delta_by_class=tuple(sorted(component_delta.items())),
                dynamic_delta_by_class=tuple(sorted(dynamic_delta.items())),
                boundary_delta_edges=boundary_edges,
                boundary_delta_metres=boundary_metres,
                dependency_proof=dependency_proof,
            )
        )
    return planned, {
        "discovery_occurrence_count": len(discoveries),
        "canonical_action_count": len(actions),
        "duplicate_global_discovery_count": len(duplicate_audit),
        "globally_scored_action_count": len(planned),
        "preselection_rejections": dict(sorted(rejected.items())),
        "b_changed_pixel_count": len(b_changed),
        "b_affected_global_component_count": len(b_affected_components),
        "global_b_component_count": global_component_count,
        "duplicate_global_discovery_audit": duplicate_audit,
    }, actions


def _selection_ids(selected: Sequence[BoundaryPlannedAction]) -> list[str]:
    return [item.action.action_id for item in selected]


def _status(
    independent_without_boundary: Mapping[str, Any],
    relaxed_boundary: Mapping[str, Any],
    strict_boundary: Mapping[str, Any],
) -> str:
    if not independent_without_boundary["effect_gate"]["pass"]:
        return "fixed_independent_action_model_below_130"
    if not relaxed_boundary["effect_gate"]["pass"]:
        return "fixed_boundary_independent_action_model_below_130"
    if strict_boundary["effect_gate"]["pass"]:
        return (
            "strict_safe_plan_has_headroom"
            if strict_boundary["engineering_headroom_gate"]["pass"]
            else "strict_safe_plan_reaches_130_without_headroom"
        )
    return "dependency_replay_required_before_publication"


def run(
    b_manifest_path: Path,
    output_root: Path,
    *,
    workers: int,
    resume: bool,
) -> dict[str, Any]:
    if workers < 1 or workers > 2:
        raise V31EPlanRunError("--workers must be 1 or 2")
    b_data = C_RUNNER._load_b_manifest(b_manifest_path.resolve(), self_test=False)
    if len(b_data["parts"]) != REAL_PARTITION_COUNT:
        raise V31EPlanRunError("E plan requires the complete 140-Core B baseline")
    if output_root.exists() and any(output_root.iterdir()) and not resume:
        raise V31EPlanRunError(
            f"refusing non-empty output root without --resume: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    policy = replace(
        v31b_policy(),
        policy_id=V31E_POLICY_ID,
        policy_version=V31E_POLICY_VERSION,
    )
    snapshot = policy_snapshot(policy)
    snapshot["coordination_mode"] = V31E_COORDINATION_MODE
    snapshot["algorithm_contract"]["publication"] = "plan_only_no_raster_output"
    snapshot["algorithm_contract"]["dependency_tiers"] = (
        "strict_complete_b_affected_component_lock_and_relaxed_dependency_replay_required"
    )
    snapshot_sha = _sha256_json(snapshot)
    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION,
        "candidate_label": "E-plan",
        "v31b_manifest_sha256": b_data["manifest_file_sha256"],
        "policy_snapshot_sha256": snapshot_sha,
        "coordination_mode": V31E_COORDINATION_MODE,
        "required_dynamic_reduction": REQUIRED_DYNAMIC_REDUCTION,
        "engineering_headroom_reduction": ENGINEERING_HEADROOM_REDUCTION,
        "code_sha256": _code_sha256(),
    }
    fingerprint = _sha256_json(fingerprint_payload)
    manifest_path = output_root / "plan_manifest.json"
    if resume and manifest_path.is_file():
        prior = _read_json(manifest_path)
        if prior.get("execution_fingerprint_sha256") != fingerprint:
            raise V31EPlanRunError("resume E plan fingerprint differs")
        if prior.get("status") != "running":
            declared = prior.get("manifest_sha256")
            if declared != _sha256_json(
                {key: value for key, value in prior.items() if key != "manifest_sha256"}
            ):
                raise V31EPlanRunError("resume E plan manifest SHA mismatch")
            return prior
    running = {
        "schema_version": SCHEMA_VERSION,
        "kind": "v31e_global_boundary_plan_only",
        "candidate_label": "E-plan",
        "status": "running",
        "v31b_manifest": b_data["manifest_path"],
        "v31b_manifest_sha256": b_data["manifest_file_sha256"],
        "execution_fingerprint": fingerprint_payload,
        "execution_fingerprint_sha256": fingerprint,
        "policy_snapshot": snapshot,
        "policy_snapshot_sha256": snapshot_sha,
        "requested_partition_count": REAL_PARTITION_COUNT,
        "completed_partition_count": 0,
    }
    _atomic_json(manifest_path, running)
    started = time.monotonic()
    discoveries, shards = _collect_all(
        b_data,
        output_root=output_root,
        workers=workers,
        fingerprint=fingerprint,
        policy=policy,
    )
    planned, preparation_audit, canonical_actions = _prepare_actions(
        discoveries=discoveries, b_data=b_data, policy=policy
    )
    source_remaining, target_remaining = C_RUNNER._budget_remaining(b_data["parts"])
    independent_selected, independent_audit = select_boundary_aware_actions(
        planned,
        source_remaining=source_remaining,
        target_remaining=target_remaining,
        enforce_boundary=False,
        required_dynamic_reduction=REQUIRED_DYNAMIC_REDUCTION,
        engineering_headroom_reduction=ENGINEERING_HEADROOM_REDUCTION,
    )
    relaxed_selected, relaxed_audit = select_boundary_aware_actions(
        planned,
        source_remaining=source_remaining,
        target_remaining=target_remaining,
        enforce_boundary=True,
        required_dynamic_reduction=REQUIRED_DYNAMIC_REDUCTION,
        engineering_headroom_reduction=ENGINEERING_HEADROOM_REDUCTION,
    )
    strict = [
        item
        for item in planned
        if item.dependency_proof == "strict_b_component_lock_clear"
    ]
    strict_selected, strict_audit = select_boundary_aware_actions(
        strict,
        source_remaining=source_remaining,
        target_remaining=target_remaining,
        enforce_boundary=True,
        required_dynamic_reduction=REQUIRED_DYNAMIC_REDUCTION,
        engineering_headroom_reduction=ENGINEERING_HEADROOM_REDUCTION,
    )
    status = _status(independent_audit, relaxed_audit, strict_audit)
    manifest: dict[str, Any] = {
        **running,
        "status": status,
        "completed_partition_count": len(shards),
        "runtime_seconds": time.monotonic() - started,
        "collect": {
            "shard_count": len(shards),
            "raw_generated": int(sum(item["audit"]["raw_generated"] for item in shards)),
            "canonical_generated": int(
                sum(item["audit"]["canonical_generated"] for item in shards)
            ),
            "global_discovery_count": len(discoveries),
        },
        "canonical_actions": [global_action_to_dict(item) for item in canonical_actions],
        "planned_actions": [_planned_action_dict(item) for item in planned],
        "preparation_audit": preparation_audit,
        "plans": {
            "dependency_relaxed_independent_plan_without_boundary": {
                "proof_status": (
                    "fixed_conservative_independent_action_model_optimum_without_"
                    "boundary_or_b_dependency_replay_not_a_global_method_upper_bound"
                ),
                "selected_action_ids": _selection_ids(independent_selected),
                "audit": independent_audit,
            },
            "dependency_relaxed_boundary_independent_plan": {
                "proof_status": (
                    "fixed_conservative_independent_action_model_optimum_requires_"
                    "exact_b_dependency_replay_not_a_global_method_upper_bound"
                ),
                "selected_action_ids": _selection_ids(relaxed_selected),
                "audit": relaxed_audit,
            },
            "strict_b_component_lock_boundary_plan": {
                "proof_status": "plan_safe_under_complete_b_affected_component_lock",
                "selected_action_ids": _selection_ids(strict_selected),
                "audit": strict_audit,
            },
        },
        "decision": {
            "required_dynamic_reduction": REQUIRED_DYNAMIC_REDUCTION,
            "engineering_headroom_reduction": ENGINEERING_HEADROOM_REDUCTION,
            "formal_effect_gate_uses_130_not_headroom": True,
            "publication_performed": False,
            "next_step": (
                "stop_this_fixed_independent_action_model_only"
                if status.startswith("fixed_")
                else (
                    "implement_and_publish_strict_plan_for_full_evaluation"
                    if status.startswith("strict_safe_plan")
                    else "regenerate_and_replay_exact_b_accepted_action_ledger"
                )
            ),
        },
    }
    manifest["manifest_sha256"] = _sha256_json(manifest)
    _atomic_json(manifest_path, manifest)
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v31b-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = run(
            args.v31b_manifest,
            args.output_root,
            workers=args.workers,
            resume=args.resume,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "completed_partition_count": result["completed_partition_count"],
                "plans": {
                    key: value["audit"]["selected_dynamic_reduction"]
                    for key, value in result["plans"].items()
                },
                "next_step": result["decision"]["next_step"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
