from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
from scipy import ndimage

from inference_scripts.deployment_config import CLASS_ORDER
from inference_scripts.fragmentation_v31_candidate.candidate import v31b_policy
from inference_scripts.fragmentation_v31_candidate.v31b_census import (
    FOUR_CONNECTED,
    CensusError,
    CoreInput,
    add_probability_evidence,
    collect_core_shard,
    coordinate_shards,
    empirical_p10,
    label_core_component_map,
    topology_class,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "scratch"
    / "v31b_fragment_census_20260825"
    / "run_v31b_fragment_census.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("_v31b_census_test_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _four_core_shards(grid: np.ndarray, v3: np.ndarray | None = None):
    shards = []
    for row0 in (0, 4):
        for col0 in (0, 4):
            labels = grid[row0 : row0 + 4, col0 : col0 + 4].copy()
            before = None if v3 is None else v3[row0 : row0 + 4, col0 : col0 + 4].copy()
            shards.append(
                collect_core_shard(
                    CoreInput(
                        f"core-{row0}-{col0}",
                        (row0, col0, 4, 4),
                        labels,
                        np.ones((4, 4), dtype=bool),
                        1.0,
                        before,
                    ),
                    CLASS_ORDER,
                )
            )
    return shards


def test_global_census_matches_dense_components_and_is_order_independent():
    grid = np.full((8, 8), 13, dtype=np.int16)
    grid[2, 2] = 21  # closed, one neighbour class and one neighbour component
    grid[5, 3:5] = 52  # one component crosses the vertical Core seam
    grid[4, 6] = 31
    grid[3, 6] = 13
    grid[5, 6] = 32
    grid[4, 5] = 43
    grid[4, 7] = 51  # four neighbour classes around class 31
    v3 = grid.copy()
    v3[2, 2] = 13
    shards = _four_core_shards(grid, v3)
    ledger, audit = coordinate_shards(
        shards,
        class_codes=CLASS_ORDER,
        policy=v31b_policy(),
        global_window=(0, 0, 8, 8),
    )
    dense_count = 0
    for code in CLASS_ORDER:
        _labels, count = ndimage.label(grid == code, structure=FOUR_CONNECTED)
        dense_count += int(count)
    assert audit["global_component_count"] == dense_count
    unique = next(row for row in ledger if row["class_code"] == 21)
    assert unique["fragment_id"] == "21:2:2"
    assert unique["topology_class"] == "T2_closed_single_neighbor"
    assert unique["neighbor_class_set"] == [13]
    assert unique["adjacent_global_component_count"] == 1
    assert unique["b_changed_pixel_count"] == 1
    crossing = next(row for row in ledger if row["class_code"] == 52)
    assert crossing["cross_core"] is True
    assert crossing["owner_core_count"] == 2
    multi = next(row for row in ledger if row["class_code"] == 31)
    assert multi["topology_class"] == "T4_closed_multi_neighbor"
    reversed_ledger, reversed_audit = coordinate_shards(
        list(reversed(shards)),
        class_codes=CLASS_ORDER,
        policy=v31b_policy(),
        global_window=(0, 0, 8, 8),
    )
    assert reversed_audit["global_component_count"] == dense_count
    assert reversed_ledger == ledger


def test_diagonal_cells_do_not_merge_and_internal_invalid_is_t0():
    labels = np.array(
        [[21, 13, 13], [13, 21, 13], [13, 13, 13]], dtype=np.int16
    )
    valid = np.ones((3, 3), dtype=bool)
    valid[0, 1] = False
    shard = collect_core_shard(
        CoreInput("one", (0, 0, 3, 3), labels, valid, 1.0), CLASS_ORDER
    )
    ledger, audit = coordinate_shards(
        [shard],
        class_codes=CLASS_ORDER,
        policy=v31b_policy(),
        global_window=(0, 0, 3, 3),
    )
    class21 = [row for row in ledger if row["class_code"] == 21]
    assert len(class21) == 2
    assert audit["global_component_count"] >= 3
    assert next(row for row in class21 if row["fragment_id"] == "21:0:0")[
        "topology_class"
    ] == "T0_range_exposed"


def test_topology_axis_is_mutually_exclusive_and_p10_is_frozen():
    assert topology_class(True, 1) == "T0_range_exposed"
    assert topology_class(False, 0) == "T1_closed_zero_neighbor"
    assert topology_class(False, 1) == "T2_closed_single_neighbor"
    assert topology_class(False, 2) == "T3_closed_two_neighbors"
    assert topology_class(False, 3) == "T4_closed_multi_neighbor"
    assert empirical_p10([0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.0]) == 0.0
    assert empirical_p10([]) is None


def test_probability_gate_uses_target_policy_and_records_all_14_means():
    record = {
        "fragment_id": "13:1:1",
        "pixel_count": 2,
        "unique_neighbor_code": 61,
        "policy_island_evidence": {
            "exact_single_adjacent_component": True,
            "source_unprotected": True,
            "within_source_area_cap": True,
            "semantic_compatible_target": True,
            "target_not_ordinary_protected": True,
        },
    }
    all_values = np.full((14, 2), 0.01, dtype=np.float32)
    all_values[1] = 0.30  # source class 13
    all_values[11] = 0.35  # target 61: below target's 0.40 mean gate
    all_values[3] = 0.80  # max-confidence gate also fails
    add_probability_evidence(
        {0: record},
        np.array([0, 0], dtype=np.int64),
        all_values[1],
        all_values[11],
        np.max(all_values, axis=0),
        all_values,
        class_codes=CLASS_ORDER,
        policy=v31b_policy(),
    )
    evidence = record["policy_island_evidence"]
    assert evidence["target_minimum_probability_mean"] == 0.40
    assert evidence["probability_gate_pass"] is False
    assert evidence["mean_confidence_pass"] is False
    assert list(evidence["mean_probability_by_class_code"]) == [
        str(code) for code in CLASS_ORDER
    ]
    assert evidence["full_policy_gate_pass"] is False


def test_probability_halo_slice_selects_owner_core_only():
    runner = _load_runner()
    entry = {
        "partition_id": "p",
        "halo_window": {"x0": 8, "x1": 14, "y0": 18, "y1": 24},
        "core_window": {"x0": 10, "x1": 14, "y0": 20, "y1": 24},
    }
    rows, cols = runner._probability_core_slice(
        entry, np.zeros((14, 6, 6), dtype=np.float32)
    )
    assert rows == slice(2, 6)
    assert cols == slice(2, 6)


def test_component_rebuild_is_deterministic_and_unknown_classes_fail():
    labels = np.array([[13, 21], [13, 13]], dtype=np.int16)
    valid = np.ones((2, 2), dtype=bool)
    first, count = label_core_component_map(labels, valid, CLASS_ORDER)
    second, second_count = label_core_component_map(labels, valid, CLASS_ORDER)
    assert count == second_count == 2
    assert np.array_equal(first, second)
    invalid = labels.copy()
    invalid[0, 0] = 99
    try:
        collect_core_shard(
            CoreInput("bad", (0, 0, 2, 2), invalid, valid, 1.0), CLASS_ORDER
        )
    except CensusError as exc:
        assert "unknown valid class codes" in str(exc)
    else:
        raise AssertionError("unknown class code was not rejected")


def test_checkpoint_identity_and_run_manifest_hash_fail_closed(tmp_path):
    runner = _load_runner()
    labels = np.array([[13, 21], [13, 13]], dtype=np.int16)
    shard = collect_core_shard(
        CoreInput("A", (0, 0, 2, 2), labels, np.ones((2, 2), bool), 1.0),
        CLASS_ORDER,
    )
    root = tmp_path / "collect"
    runner._save_collect_shard(root, "A", "fingerprint", shard)
    shutil.copy2(root / "A.json", root / "B.json")
    shutil.copy2(root / "A.npz", root / "B.npz")
    with pytest.raises(runner.CensusRunError, match="fingerprint/hash mismatch"):
        runner._load_collect_shard(root, "B", "fingerprint")

    probability = {
        "group_id": np.array([1], dtype=np.int64),
        "current": np.array([0.4], dtype=np.float32),
        "target": np.array([0.5], dtype=np.float32),
        "confidence": np.array([0.5], dtype=np.float32),
        "class_values": np.full((14, 1), 1 / 14, dtype=np.float32),
    }
    probability_root = tmp_path / "probability"
    runner._save_probability_shard(
        probability_root, "A", "coordination", probability
    )
    shutil.copy2(probability_root / "A.json", probability_root / "B.json")
    shutil.copy2(probability_root / "A.npz", probability_root / "B.npz")
    with pytest.raises(runner.CensusRunError, match="fingerprint/hash mismatch"):
        runner._load_probability_shard(probability_root, "B", "coordination")

    manifest_path = tmp_path / "run_manifest.json"
    runner._write_run_manifest(manifest_path, {"status": "running"})
    manifest = runner._read_json(manifest_path)
    runner._validate_run_manifest(manifest)
    manifest["status"] = "complete"
    with pytest.raises(runner.CensusRunError, match="self SHA-256 mismatch"):
        runner._validate_run_manifest(manifest)


def test_b_policy_binding_rejects_current_snapshot_mismatch():
    runner = _load_runner()
    snapshot = runner.policy_snapshot(runner.v31b_policy())
    snapshot_sha = runner._sha256_json(snapshot)
    runner._validate_b_policy(
        {
            "v31b_policy_snapshot": snapshot,
            "v31b_policy_snapshot_sha256": snapshot_sha,
        },
        snapshot,
        snapshot_sha,
    )
    changed = dict(snapshot)
    changed["policy_version"] = "different"
    with pytest.raises(runner.CensusRunError, match="policy differs"):
        runner._validate_b_policy(
            {
                "v31b_policy_snapshot": changed,
                "v31b_policy_snapshot_sha256": runner._sha256_json(changed),
            },
            snapshot,
            snapshot_sha,
        )
