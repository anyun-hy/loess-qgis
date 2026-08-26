#!/usr/bin/env python3
"""Checkpointed, read-only census of all remaining V3.1-B fragments.

The runner reads a completed frozen B manifest, creates compact per-Core
component shards, rebuilds exact global four-connected identities and then
streams the frozen 14-class probabilities only for closed single-neighbour
dynamic fragments. It writes JSON/JSONL analysis only: no label raster,
proposal or production entry point is created or modified.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
from fragmentation_v31_candidate.v31b_census import (  # noqa: E402
    CensusError,
    CoreInput,
    add_probability_evidence,
    collect_core_shard,
    coordinate_shards,
    label_core_component_map,
    probability_selection_by_local_component,
)


C_RUNNER_PATH = (
    REPO_ROOT / "scratch" / "v31c_full_140_20260825" / "run_v31c_global.py"
)
B_RUNNER_PATH = (
    REPO_ROOT / "scratch" / "v31b_full_140_20260824" / "run_v31b_from_v3.py"
)
MODULE_PATH = (
    REPO_ROOT
    / "inference_scripts"
    / "fragmentation_v31_candidate"
    / "v31b_census.py"
)
SCHEMA_VERSION = 1
REAL_CORE_COUNT = 140
EXPECTED_DYNAMIC_COUNT = 25_983
EXPECTED_DYNAMIC_AREA_M2 = 423_392.75095022254
EXPECTED_GLOBAL_COMPONENT_COUNT = 105_236
EXPECTED_CORE_PIXEL_COUNT = 1_318_813_696
EXPECTED_VALID_PIXEL_COUNT = 831_531_565
EXPECTED_PER_CLASS = {
    12: (257, 3_762.7642225007744),
    13: (3_707, 79_026.8470128412),
    21: (1_727, 47_685.94109947764),
    31: (2_565, 23_245.867382888526),
    32: (2_326, 62_433.95274380232),
    33: (1_429, 25_514.540024463895),
    43: (4_231, 41_792.43247443872),
    51: (303, 5_601.299001708527),
    52: (1_854, 35_289.55158494536),
    53: (383, 5_311.381528347651),
    54: (154, 2_281.123361410452),
    61: (247, 2_193.3659272569444),
    62: (4_665, 62_559.93389268045),
    71: (2_135, 26_693.75069346003),
}
REQUIRED_DYNAMIC_REDUCTION = 130
ENGINEERING_HEADROOM_REDUCTION = 150


class CensusRunError(RuntimeError):
    """A frozen-input, checkpoint or output closure contract failed."""


def _load_module(name: str, path: Path) -> Any:
    if not path.is_file():
        raise CensusRunError(f"required helper is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise CensusRunError(f"cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C_RUNNER = _load_module("_v31b_census_c_runner", C_RUNNER_PATH)


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
        raise CensusRunError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CensusRunError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
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


def _write_run_manifest(path: Path, value: Mapping[str, Any]) -> None:
    payload = dict(value)
    payload.pop("run_manifest_sha256", None)
    payload["run_manifest_sha256"] = _sha256_json(payload)
    _atomic_json(path, payload)


def _validate_run_manifest(value: Mapping[str, Any]) -> None:
    body = {key: item for key, item in value.items() if key != "run_manifest_sha256"}
    if value.get("run_manifest_sha256") != _sha256_json(body):
        raise CensusRunError("run manifest self SHA-256 mismatch")


def _validate_b_policy(
    manifest: Mapping[str, Any], snapshot: Mapping[str, Any], snapshot_sha256: str
) -> None:
    if (
        manifest.get("v31b_policy_snapshot") != dict(snapshot)
        or manifest.get("v31b_policy_snapshot_sha256") != snapshot_sha256
    ):
        raise CensusRunError("B manifest policy differs from the census policy")


def _validate_summary(value: Mapping[str, Any]) -> None:
    body = {key: item for key, item in value.items() if key != "summary_sha256"}
    if value.get("summary_sha256") != _sha256_json(body):
        raise CensusRunError("summary self SHA-256 mismatch")


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _code_sha256() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        MODULE_PATH.resolve(),
        C_RUNNER_PATH.resolve(),
        B_RUNNER_PATH.resolve(),
        REPO_ROOT / "inference_scripts" / "deployment_config.py",
        REPO_ROOT / "inference_scripts" / "fragmentation_v3.py",
        REPO_ROOT / "inference_scripts" / "small_component_regularizer.py",
        REPO_ROOT
        / "inference_scripts"
        / "fragmentation_v31_candidate"
        / "candidate.py",
        REPO_ROOT
        / "inference_scripts"
        / "fragmentation_v31_candidate"
        / "__init__.py",
    )
    return {
        str(path.relative_to(REPO_ROOT)): _sha256_file(path)
        for path in paths
    }


def _shape(window: Mapping[str, int]) -> tuple[int, int]:
    return int(window["y1"]) - int(window["y0"]), int(window["x1"]) - int(window["x0"])


def _core_input(
    entry: Mapping[str, Any], part: Mapping[str, Any], *, include_v3: bool
) -> CoreInput:
    window = entry["core_window"]
    labels = np.load(
        part["outputs"]["v31a"]["path"], mmap_mode="r", allow_pickle=False
    )
    valid = np.load(
        part["outputs"]["valid"]["path"], mmap_mode="r", allow_pickle=False
    )
    v3 = (
        np.load(part["outputs"]["v3"]["path"], mmap_mode="r", allow_pickle=False)
        if include_v3
        else None
    )
    return CoreInput(
        str(entry["partition_id"]),
        (
            int(window["y0"]),
            int(window["x0"]),
            int(window["y1"]) - int(window["y0"]),
            int(window["x1"]) - int(window["x0"]),
        ),
        labels,
        valid,
        float(part["physical_metrics"]["pixel_area_m2"]),
        v3,
    )


def _save_collect_shard(
    root: Path, core_id: str, fingerprint: str, shard: Mapping[str, Any]
) -> dict[str, Any]:
    npz_path = root / f"{core_id}.npz"
    json_path = root / f"{core_id}.json"
    arrays = {
        key: np.asarray(value)
        for key, value in shard.items()
        if key not in {"core_id", "component_count"}
    }
    _atomic_npz(npz_path, arrays)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "stage": "v31b_fragment_census_collect",
        "core_id": core_id,
        "execution_fingerprint_sha256": fingerprint,
        "component_count": int(shard["component_count"]),
        "npz_path": str(npz_path.resolve()),
        "npz_sha256": _sha256_file(npz_path),
    }
    metadata["metadata_sha256"] = _sha256_json(metadata)
    _atomic_json(json_path, metadata)
    return metadata


def _load_collect_shard(
    root: Path, core_id: str, fingerprint: str
) -> dict[str, Any] | None:
    json_path, npz_path = root / f"{core_id}.json", root / f"{core_id}.npz"
    if not json_path.exists() and not npz_path.exists():
        return None
    if not json_path.is_file() or not npz_path.is_file():
        raise CensusRunError(f"{core_id}: incomplete collect checkpoint")
    metadata = _read_json(json_path)
    body = {key: value for key, value in metadata.items() if key != "metadata_sha256"}
    if (
        metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("stage") != "v31b_fragment_census_collect"
        or metadata.get("core_id") != core_id
        or metadata.get("npz_path") != str(npz_path.resolve())
        or
        metadata.get("execution_fingerprint_sha256") != fingerprint
        or metadata.get("metadata_sha256") != _sha256_json(body)
        or metadata.get("npz_sha256") != _sha256_file(npz_path)
    ):
        raise CensusRunError(f"{core_id}: collect checkpoint fingerprint/hash mismatch")
    with np.load(npz_path, allow_pickle=False) as archive:
        shard = {name: archive[name].copy() for name in archive.files}
    shard["core_id"] = core_id
    shard["component_count"] = int(metadata["component_count"])
    required = {
        "window",
        "class_code",
        "pixel_count",
        "area_m2",
        "min_row",
        "min_col",
        "boundary_internal",
        "b_changed_pixels",
        "different_class_pairs",
        "edge_top",
        "edge_bottom",
        "edge_left",
        "edge_right",
    }
    if required - set(shard):
        raise CensusRunError(f"{core_id}: collect checkpoint fields are incomplete")
    return shard


def _save_probability_shard(
    root: Path,
    core_id: str,
    fingerprint: str,
    arrays: Mapping[str, np.ndarray],
) -> None:
    npz_path = root / f"{core_id}.npz"
    json_path = root / f"{core_id}.json"
    _atomic_npz(npz_path, arrays)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "stage": "v31b_fragment_census_probability",
        "core_id": core_id,
        "coordination_fingerprint_sha256": fingerprint,
        "pixel_count": int(len(arrays["group_id"])),
        "npz_path": str(npz_path.resolve()),
        "npz_sha256": _sha256_file(npz_path),
    }
    metadata["metadata_sha256"] = _sha256_json(metadata)
    _atomic_json(json_path, metadata)


def _load_probability_shard(
    root: Path, core_id: str, fingerprint: str
) -> dict[str, np.ndarray] | None:
    json_path, npz_path = root / f"{core_id}.json", root / f"{core_id}.npz"
    if not json_path.exists() and not npz_path.exists():
        return None
    if not json_path.is_file() or not npz_path.is_file():
        raise CensusRunError(f"{core_id}: incomplete probability checkpoint")
    metadata = _read_json(json_path)
    body = {key: value for key, value in metadata.items() if key != "metadata_sha256"}
    if (
        metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("stage") != "v31b_fragment_census_probability"
        or metadata.get("core_id") != core_id
        or metadata.get("npz_path") != str(npz_path.resolve())
        or
        metadata.get("coordination_fingerprint_sha256") != fingerprint
        or metadata.get("metadata_sha256") != _sha256_json(body)
        or metadata.get("npz_sha256") != _sha256_file(npz_path)
    ):
        raise CensusRunError(f"{core_id}: probability checkpoint fingerprint/hash mismatch")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    required = {"group_id", "current", "target", "confidence", "class_values"}
    if required != set(arrays):
        raise CensusRunError(f"{core_id}: probability checkpoint fields are invalid")
    pixel_count = int(metadata["pixel_count"])
    if (
        any(arrays[key].ndim != 1 or len(arrays[key]) != pixel_count for key in ("group_id", "current", "target", "confidence"))
        or arrays["class_values"].shape != (len(CLASS_ORDER), pixel_count)
    ):
        raise CensusRunError(f"{core_id}: probability checkpoint length mismatch")
    return arrays


def _probability_core_slice(
    entry: Mapping[str, Any], cube: np.ndarray
) -> tuple[slice, slice]:
    halo, core = entry["halo_window"], entry["core_window"]
    rows = slice(int(core["y0"]) - int(halo["y0"]), int(core["y1"]) - int(halo["y0"]))
    cols = slice(int(core["x0"]) - int(halo["x0"]), int(core["x1"]) - int(halo["x0"]))
    if (
        rows.start < 0
        or cols.start < 0
        or rows.stop > cube.shape[1]
        or cols.stop > cube.shape[2]
        or (rows.stop - rows.start, cols.stop - cols.start) != _shape(core)
    ):
        raise CensusRunError(f"{entry['partition_id']}: probability halo does not contain Core")
    return rows, cols


def _collect_probability_core(
    entry: Mapping[str, Any],
    part: Mapping[str, Any],
    shard: Mapping[str, Any],
    local_group: np.ndarray,
    ledger_by_group: Mapping[int, Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    core = _core_input(entry, part, include_v3=False)
    component, count = label_core_component_map(core.labels, core.valid, CLASS_ORDER)
    if count != int(shard["component_count"]):
        raise CensusRunError(f"{core.core_id}: deterministic component rebuild count differs")
    group_for_local, current_index, target_index = probability_selection_by_local_component(
        shard, local_group, ledger_by_group, CLASS_ORDER
    )
    selected_local = group_for_local >= 0
    selected = selected_local[component]
    rows, cols = np.nonzero(selected)
    if not len(rows):
        return {
            "group_id": np.empty(0, dtype=np.int64),
            "current": np.empty(0, dtype=np.float32),
            "target": np.empty(0, dtype=np.float32),
            "confidence": np.empty(0, dtype=np.float32),
            "class_values": np.empty((len(CLASS_ORDER), 0), dtype=np.float32),
        }
    local_ids = component[rows, cols]
    groups = group_for_local[local_ids]
    source_indices = current_index[local_ids].astype(np.int64)
    target_indices = target_index[local_ids].astype(np.int64)
    if np.any(source_indices < 0) or np.any(target_indices < 0):
        raise CensusRunError(f"{core.core_id}: selected probability pixel lacks class mapping")
    cube = np.load(entry["probability_path"], mmap_mode="r", allow_pickle=False)
    if cube.dtype != np.dtype("float32") or cube.shape[0] != len(CLASS_ORDER):
        raise CensusRunError(f"{core.core_id}: frozen probability cube metadata changed")
    core_rows, core_cols = _probability_core_slice(entry, cube)
    probability_rows = rows + int(core_rows.start)
    probability_cols = cols + int(core_cols.start)
    class_values = np.asarray(
        cube[:, probability_rows, probability_cols], dtype=np.float32
    )
    pixel_index = np.arange(len(rows), dtype=np.int64)
    current = class_values[source_indices, pixel_index]
    target = class_values[target_indices, pixel_index]
    confidence = np.max(class_values, axis=0).astype(np.float32, copy=False)
    return {
        "group_id": groups.astype(np.int64, copy=False),
        "current": current,
        "target": target,
        "confidence": confidence,
        "class_values": class_values,
    }


def _counter_summary(
    ledger: Sequence[Mapping[str, Any]], key: str
) -> dict[str, dict[str, float | int]]:
    counts: Counter[str] = Counter()
    areas: defaultdict[str, float] = defaultdict(float)
    for row in ledger:
        value = str(row[key])
        counts[value] += 1
        areas[value] += float(row["area_m2"])
    return {
        value: {"count": int(counts[value]), "area_m2": float(areas[value])}
        for value in sorted(counts)
    }


def _build_summary(
    ledger: Sequence[Mapping[str, Any]],
    *,
    b_manifest_path: Path,
    b_manifest_sha256: str,
    execution_fingerprint: str,
    policy_sha256: str,
    runtime_seconds: float,
    global_component_count: int,
) -> dict[str, Any]:
    topology = _counter_summary(ledger, "topology_class")
    per_class = _counter_summary(ledger, "class_code")
    ratio_bins = _counter_summary(ledger, "area_ratio_bin")
    total_count = len(ledger)
    total_area = float(sum(float(row["area_m2"]) for row in ledger))
    fragment_ids = [str(row["fragment_id"]) for row in ledger]
    component_groups = [int(row["global_component_group"]) for row in ledger]
    if len(set(fragment_ids)) != total_count or len(set(component_groups)) != total_count:
        raise CensusRunError("dynamic fragment stable IDs/global groups are not unique")
    if sum(int(value["count"]) for value in topology.values()) != total_count:
        raise CensusRunError("T0-T4 topology axis does not close")
    if sum(int(value["count"]) for value in per_class.values()) != total_count:
        raise CensusRunError("per-class axis does not close")
    if sum(int(value["count"]) for value in ratio_bins.values()) != total_count:
        raise CensusRunError("area/MMU axis does not close")
    if total_count != EXPECTED_DYNAMIC_COUNT:
        raise CensusRunError(
            f"dynamic fragment count {total_count} differs from frozen reference {EXPECTED_DYNAMIC_COUNT}"
        )
    if not math.isclose(
        total_area, EXPECTED_DYNAMIC_AREA_M2, rel_tol=0.0, abs_tol=1e-6
    ):
        raise CensusRunError(
            f"dynamic fragment area {total_area} differs from frozen reference {EXPECTED_DYNAMIC_AREA_M2}"
        )
    if global_component_count != EXPECTED_GLOBAL_COMPONENT_COUNT:
        raise CensusRunError("global component count differs from frozen reference")
    for code, (reference_count, reference_area) in EXPECTED_PER_CLASS.items():
        actual = per_class.get(str(code))
        if (
            actual is None
            or int(actual["count"]) != reference_count
            or not math.isclose(
                float(actual["area_m2"]), reference_area, rel_tol=0.0, abs_tol=1e-6
            )
        ):
            raise CensusRunError(f"class {code} does not reproduce the frozen B reference")

    def selected(predicate: Any) -> int:
        return sum(bool(predicate(row)) for row in ledger)

    t2 = [row for row in ledger if row["topology_class"] == "T2_closed_single_neighbor"]
    waterfall: list[dict[str, Any]] = []
    current = list(t2)
    waterfall.append({"stage": "closed_single_neighbor_class", "count": len(current)})
    gates = (
        ("exact_single_adjacent_component", "exact_single_adjacent_component"),
        ("source_unprotected", "source_unprotected"),
        ("within_source_area_cap", "within_source_area_cap"),
        ("semantic_compatible_target", "semantic_compatible_target"),
        ("target_not_ordinary_protected", "target_not_ordinary_protected"),
        ("mean_confidence_pass", "mean_confidence_pass"),
        ("probability_gate_pass", "probability_gate_pass"),
    )
    for stage, field in gates:
        current = [
            row
            for row in current
            if row["policy_island_evidence"].get(field) is True
        ]
        waterfall.append({"stage": stage, "count": len(current)})
    full_gate_count = selected(
        lambda row: row["policy_island_evidence"].get("full_policy_gate_pass") is True
    )
    if full_gate_count != len(current):
        raise CensusRunError("policy-island gate waterfall does not close")
    transition_counts: Counter[str] = Counter(
        f"{row['class_code']}->{row['unique_neighbor_code']}" for row in t2
    )
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "v31b_remaining_fragment_structure_census",
        "status": "complete",
        "scope": "single_swin_b_same_domain_isolated_v31b_candidate",
        "not_approved_fusion": True,
        "not_production_v3": True,
        "not_rag": True,
        "publication_performed": False,
        "raster_written": False,
        "b_manifest": str(b_manifest_path.resolve()),
        "b_manifest_sha256": b_manifest_sha256,
        "execution_fingerprint_sha256": execution_fingerprint,
        "v31b_policy_snapshot_sha256": policy_sha256,
        "runtime_seconds": float(runtime_seconds),
        "connectivity": 4,
        "class_order": list(CLASS_ORDER),
        "dynamic_definition": "0 < global_component_area_m2 < class_dynamic_mmu_m2",
        "dynamic_fragment_count": total_count,
        "dynamic_fragment_area_m2": total_area,
        "reference_closure": {
            "expected_count": EXPECTED_DYNAMIC_COUNT,
            "expected_area_m2": EXPECTED_DYNAMIC_AREA_M2,
            "count_matches": True,
            "area_matches_abs_1e_6_m2": True,
            "global_component_count": global_component_count,
            "expected_global_component_count": EXPECTED_GLOBAL_COMPONENT_COUNT,
            "global_component_count_matches": True,
            "core_count": REAL_CORE_COUNT,
            "core_pixel_count": EXPECTED_CORE_PIXEL_COUNT,
            "valid_pixel_count": EXPECTED_VALID_PIXEL_COUNT,
        },
        "topology_axis": topology,
        "per_class": per_class,
        "area_to_mmu_axis": ratio_bins,
        "adjacent_global_component_count_axis": _counter_summary(
            ledger, "adjacent_global_component_count"
        ),
        "cross_core": {
            "count": selected(lambda row: row["cross_core"]),
            "not_cross_core_count": selected(lambda row: not row["cross_core"]),
            "area_m2": float(sum(float(row["area_m2"]) for row in ledger if row["cross_core"])),
        },
        "protected_source": {
            "count": selected(lambda row: row["protected_source"]),
            "unprotected_source_count": selected(
                lambda row: not row["protected_source"]
            ),
            "area_m2": float(sum(float(row["area_m2"]) for row in ledger if row["protected_source"])),
        },
        "b_affected_component": {
            "count": selected(lambda row: row["b_affected_component"]),
            "unchanged_component_count": selected(
                lambda row: not row["b_affected_component"]
            ),
            "area_m2": float(sum(float(row["area_m2"]) for row in ledger if row["b_affected_component"])),
        },
        "closed_single_neighbor_contact_matrix": dict(sorted(transition_counts.items())),
        "policy_island_descriptive_waterfall": waterfall,
        "full_existing_policy_island_gate_count": full_gate_count,
        "effect_gate": {
            "required_actual_dynamic_reduction": REQUIRED_DYNAMIC_REDUCTION,
            "engineering_headroom_reduction": ENGINEERING_HEADROOM_REDUCTION,
            "descriptive_pool_at_least_130": bool(len(t2) >= REQUIRED_DYNAMIC_REDUCTION),
            "full_existing_policy_gate_pool_at_least_130": bool(
                full_gate_count >= REQUIRED_DYNAMIC_REDUCTION
            ),
            "interpretation": (
                "A descriptive pool count is not a predicted or safe net reduction. "
                "Only a conflict/budget/dependency/boundary constrained plan and a full "
                "published-raster recomputation can pass the effect gate."
            ),
        },
        "limitations": [
            "Neighbour classes and probability gates are structural/evidence observations, not ground truth.",
            "No fragment is labelled correct, wrong, safe to replace or repairable.",
            "This census does not propose footprints or rerun the failed RAG method.",
        ],
    }
    return summary


def run(
    b_manifest_path: Path,
    output_root: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    b_manifest_path = b_manifest_path.resolve()
    if not b_manifest_path.is_file():
        raise CensusRunError(f"B manifest is missing: {b_manifest_path}")
    b_manifest_file_sha = _sha256_file(b_manifest_path)
    b_data = C_RUNNER._load_b_manifest(b_manifest_path, self_test=False)
    if len(b_data["parts"]) != REAL_CORE_COUNT:
        raise CensusRunError("real census requires exactly 140 completed B Cores")
    coverage = b_data["manifest"].get("coverage") or {}
    if (
        coverage.get("complete") is not True
        or coverage.get("core_windows_nonoverlapping") is not True
        or coverage.get("global_core_grid_exact") is not True
        or int(coverage.get("published_core_pixel_count", -1)) != EXPECTED_CORE_PIXEL_COUNT
        or int(coverage.get("published_valid_pixel_count", -1)) != EXPECTED_VALID_PIXEL_COUNT
    ):
        raise CensusRunError("B coverage does not reproduce the frozen 140-Core domain")
    policy = v31b_policy()
    policy_data = policy_snapshot(policy)
    policy_sha = _sha256_json(policy_data)
    _validate_b_policy(b_data["manifest"], policy_data, policy_sha)
    code = _code_sha256()
    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "v31b_remaining_fragment_structure_census",
        "b_manifest_sha256": b_manifest_file_sha,
        "b_manifest_internal_sha256": b_data["manifest"].get("manifest_sha256"),
        "class_codes": list(CLASS_ORDER),
        "v31b_policy_snapshot_sha256": policy_sha,
        "expected_dynamic_count": EXPECTED_DYNAMIC_COUNT,
        "expected_dynamic_area_m2": EXPECTED_DYNAMIC_AREA_M2,
        "expected_global_component_count": EXPECTED_GLOBAL_COMPONENT_COUNT,
        "code_sha256": code,
    }
    fingerprint = _sha256_json(fingerprint_payload)
    run_manifest_path = output_root / "run_manifest.json"
    if output_root.exists() and any(output_root.iterdir()) and not resume:
        raise CensusRunError("refusing non-empty output root without --resume")
    output_root.mkdir(parents=True, exist_ok=True)
    if resume and run_manifest_path.is_file():
        prior = _read_json(run_manifest_path)
        _validate_run_manifest(prior)
        if prior.get("execution_fingerprint_sha256") != fingerprint:
            raise CensusRunError("resume execution fingerprint differs")
        if prior.get("status") == "complete":
            summary_path = output_root / "summary.json"
            ledger_path = output_root / "fragment_ledger.jsonl"
            if (
                summary_path.is_file()
                and ledger_path.is_file()
                and prior.get("summary_sha256") == _sha256_file(summary_path)
                and prior.get("ledger_sha256") == _sha256_file(ledger_path)
            ):
                summary = _read_json(summary_path)
                _validate_summary(summary)
                return summary
            raise CensusRunError("complete resume outputs changed")
    elif resume and any(output_root.iterdir()):
        raise CensusRunError("non-empty resume root lacks run_manifest.json")
    run_manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "v31b_remaining_fragment_structure_census_run",
        "status": "running",
        "publication_performed": False,
        "b_manifest": str(b_manifest_path),
        "b_manifest_sha256": b_manifest_file_sha,
        "execution_fingerprint": fingerprint_payload,
        "execution_fingerprint_sha256": fingerprint,
        "completed_collect_core_count": 0,
        "completed_probability_core_count": 0,
        "requested_core_count": REAL_CORE_COUNT,
    }
    _write_run_manifest(run_manifest_path, run_manifest)

    entries = sorted(b_data["source"]["entries"], key=lambda item: item["partition_id"])
    parts = b_data["parts"]
    collect_root = output_root / "collect"
    collect_root.mkdir(exist_ok=True)
    shards: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, 1):
        core_id = str(entry["partition_id"])
        shard = _load_collect_shard(collect_root, core_id, fingerprint) if resume else None
        if shard is None:
            core = _core_input(entry, parts[core_id], include_v3=True)
            shard = collect_core_shard(core, CLASS_ORDER)
            _save_collect_shard(collect_root, core_id, fingerprint, shard)
        expected_window = np.asarray(
            (
                int(entry["core_window"]["y0"]),
                int(entry["core_window"]["x0"]),
                int(entry["core_window"]["y1"]) - int(entry["core_window"]["y0"]),
                int(entry["core_window"]["x1"]) - int(entry["core_window"]["x0"]),
            ),
            dtype=np.int64,
        )
        if not np.array_equal(np.asarray(shard["window"], dtype=np.int64), expected_window):
            raise CensusRunError(f"{core_id}: collect checkpoint window differs from source")
        shards.append(shard)
        run_manifest["completed_collect_core_count"] = index
        _write_run_manifest(run_manifest_path, run_manifest)
        print(f"v31b_census_collect {index}/{len(entries)} {core_id}", flush=True)

    global_raw = b_data["source"]["global_window"]
    global_window = (
        int(global_raw["y0"]),
        int(global_raw["x0"]),
        int(global_raw["y1"]) - int(global_raw["y0"]),
        int(global_raw["x1"]) - int(global_raw["x0"]),
    )
    ledger, coordination = coordinate_shards(
        shards,
        class_codes=CLASS_ORDER,
        policy=policy,
        global_window=global_window,
    )
    if coordination["global_component_count"] != EXPECTED_GLOBAL_COMPONENT_COUNT:
        raise CensusRunError(
            f"global component count {coordination['global_component_count']} differs from frozen reference {EXPECTED_GLOBAL_COMPONENT_COUNT}"
        )
    if coordination["dynamic_fragment_count"] != EXPECTED_DYNAMIC_COUNT or not math.isclose(
        coordination["dynamic_fragment_area_m2"], EXPECTED_DYNAMIC_AREA_M2, rel_tol=0.0, abs_tol=1e-6
    ):
        raise CensusRunError(
            "structural coordination does not reproduce the frozen B dynamic reference"
        )
    coordination_fingerprint = _sha256_json(
        {
            "execution_fingerprint_sha256": fingerprint,
            "fragment_id_sha256": _sha256_json([row["fragment_id"] for row in ledger]),
            "dynamic_fragment_count": coordination["dynamic_fragment_count"],
            "dynamic_fragment_area_m2": coordination["dynamic_fragment_area_m2"],
        }
    )
    probability_root = output_root / "probability"
    probability_root.mkdir(exist_ok=True)
    probability_arrays: list[dict[str, np.ndarray]] = []
    for index, (entry, shard) in enumerate(zip(entries, shards), 1):
        core_id = str(entry["partition_id"])
        arrays = (
            _load_probability_shard(probability_root, core_id, coordination_fingerprint)
            if resume
            else None
        )
        if arrays is None:
            arrays = _collect_probability_core(
                entry,
                parts[core_id],
                shard,
                coordination["local_group_by_core"][core_id],
                coordination["ledger_by_group"],
            )
            _save_probability_shard(
                probability_root, core_id, coordination_fingerprint, arrays
            )
        probability_arrays.append(arrays)
        run_manifest["completed_probability_core_count"] = index
        _write_run_manifest(run_manifest_path, run_manifest)
        print(f"v31b_census_probability {index}/{len(entries)} {core_id}", flush=True)
    combined = {
        key: (
            np.concatenate([item[key] for item in probability_arrays], axis=1)
            if key == "class_values"
            else np.concatenate([item[key] for item in probability_arrays])
        )
        for key in ("group_id", "current", "target", "confidence", "class_values")
    }
    add_probability_evidence(
        coordination["ledger_by_group"],
        combined["group_id"],
        combined["current"],
        combined["target"],
        combined["confidence"],
        combined["class_values"],
        class_codes=CLASS_ORDER,
        policy=policy,
    )
    missing_probability = [
        row["fragment_id"]
        for row in ledger
        if row["topology_class"] == "T2_closed_single_neighbor"
        and not row["policy_island_evidence"]["probability_available"]
    ]
    if missing_probability:
        raise CensusRunError(
            f"T2 probability evidence is incomplete for {len(missing_probability)} fragments"
        )

    summary = _build_summary(
        ledger,
        b_manifest_path=b_manifest_path,
        b_manifest_sha256=b_manifest_file_sha,
        execution_fingerprint=fingerprint,
        policy_sha256=policy_sha,
        runtime_seconds=time.monotonic() - started,
        global_component_count=int(coordination["global_component_count"]),
    )
    ledger_path = output_root / "fragment_ledger.jsonl"
    temporary_ledger = ledger_path.with_name(f".{ledger_path.name}.{os.getpid()}.tmp")
    with temporary_ledger.open("w", encoding="utf-8") as handle:
        for row in ledger:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_ledger, ledger_path)
    summary["ledger_sha256"] = _sha256_file(ledger_path)
    summary_body = dict(summary)
    summary["summary_sha256"] = _sha256_json(summary_body)
    summary_path = output_root / "summary.json"
    _atomic_json(summary_path, summary)
    run_manifest.update(
        {
            "status": "complete",
            "completed_collect_core_count": REAL_CORE_COUNT,
            "completed_probability_core_count": REAL_CORE_COUNT,
            "coordination_fingerprint_sha256": coordination_fingerprint,
            "dynamic_fragment_count": len(ledger),
            "summary": str(summary_path.resolve()),
            "summary_sha256": _sha256_file(summary_path),
            "ledger": str(ledger_path.resolve()),
            "ledger_sha256": _sha256_file(ledger_path),
            "runtime_seconds": float(time.monotonic() - started),
        }
    )
    run_manifest["global_component_count"] = int(coordination["global_component_count"])
    _write_run_manifest(run_manifest_path, run_manifest)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run(
            args.b_manifest,
            args.output_root,
            resume=args.resume,
        )
    except (CensusRunError, CensusError, C_RUNNER.V31CRunError) as exc:
        print(f"v31b fragment census failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
