import importlib.util
import json
from pathlib import Path
import sys

import pytest
import numpy as np

from inference_scripts.fragmentation_v31_candidate import GlobalAction, PlannedAction


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scratch" / "v31c_full_140_20260825" / "run_v31c_global.py"
SPEC = importlib.util.spec_from_file_location("_test_v31c_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def _planned_cross_core_action():
    action = GlobalAction(
        action_id="cross",
        kind="same_class_bridge",
        target_index=1,
        target_code=21,
        footprint=((0, 1), (0, 2)),
        involved_core_ids=("left", "right"),
        source_codes=(13,),
        source_anchors=((0, 1),),
        target_anchors=((0, 0), (0, 3)),
        dynamic_reduction=1,
        component_reduction=1,
        probability_support=.2,
        area_m2=2.0,
        discovery_partition_ids=("left", "right"),
        discovery_count=2,
        footprint_sha256="f",
        score_disagreement=False,
    )
    return PlannedAction(
        action=action,
        source_component_keys=("13:0:1",),
        target_component_keys=("21:0:0", "21:0:3"),
        source_charges=(("left", 13, 1), ("right", 13, 1)),
        target_charges=(("left", 21, 1), ("right", 21, 1)),
    )


def test_selected_cross_core_action_has_one_decision_and_owner_split():
    entries = [
        {"partition_id": "left", "core_window": {"x0": 0, "x1": 2, "y0": 0, "y1": 1}},
        {"partition_id": "right", "core_window": {"x0": 2, "x1": 4, "y0": 0, "y1": 1}},
    ]

    overlays = RUNNER._selected_overlay_by_core([_planned_cross_core_action()], entries)

    assert overlays == {
        "left": {(0, 1): (21, "cross")},
        "right": {(0, 2): (21, "cross")},
    }


def test_runner_four_core_self_test_completes_and_resume_is_stable():
    result = RUNNER._self_test(None, workers=2)

    assert result["status"] == "complete"
    assert result["candidate_label"] == "C"
    assert result["completed_partition_count"] == 4
    assert result["validation_pass"] is True
    assert result["validation"]["hard_coverage_pass"] is True


def test_global_validation_requires_each_class_component_count_nonincrease():
    def method(total, dynamic, area, boundary, per_class):
        return {
            "components_total": total,
            "dynamic_fragments": {"count": dynamic, "area_m2": area},
            "boundary": {"total_cross_class_boundary": {"metres": boundary}},
            "per_class": [
                {"class_code": code, "components": count} for code, count in per_class.items()
            ],
        }

    coverage = {
        "core_overlap_pixels": 0,
        "geometric_coverage_gap_pixels": 0,
        "outside_valid_label_pixels": {"raw": 0, "v3": 0, "v31": 0},
        "invalid_label_inside_valid_pixels": {"raw": 0, "v3": 0, "v31": 0},
    }
    b = {"coverage": coverage, "methods": {"v31": method(10, 4, 3.0, 8.0, {1: 5, 2: 5})}}
    c = {"coverage": coverage, "methods": {"v31": method(9, 4, 3.0, 8.5, {1: 6, 2: 3})}}

    validation = RUNNER._validate_global_results(b_result=b, c_result=c)

    assert validation["hard_coverage_pass"] is True
    assert validation["fragmentation_dominance"]["components_nonincreasing"] is True
    assert validation["fragmentation_dominance"]["per_class_components_nonincreasing"] is False
    assert validation["c_minus_b"]["cross_class_boundary_m"] == .5


def _write_four_core_b(root: Path) -> Path:
    parent = RUNNER.B_RUNNER._write_self_test_parent(root)
    b_root = root / "b"
    RUNNER.B_RUNNER.run(parent, b_root, workers=2, resume=False, self_test=True)
    return b_root / "run_manifest.json"


def _tree_hashes(root: Path):
    return {
        str(path.relative_to(root)): RUNNER._sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_b_class_order_mismatch_is_rejected_before_collect(tmp_path):
    b_manifest = _write_four_core_b(tmp_path)
    payload = json.loads(b_manifest.read_text(encoding="utf-8"))
    payload["class_codes"] = list(reversed(payload["class_codes"]))
    payload["manifest_sha256"] = RUNNER._sha256_json(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    RUNNER._atomic_json(b_manifest, payload)

    with pytest.raises(RUNNER.V31CRunError, match="class_codes"):
        RUNNER._load_b_manifest(b_manifest, self_test=True)


def test_completed_resume_revalidates_selection_plan_sha(tmp_path):
    b_manifest = _write_four_core_b(tmp_path)
    b_before = _tree_hashes(b_manifest.parent)
    c_root = tmp_path / "c"
    result = RUNNER.run(
        b_manifest, c_root, workers=2, resume=False, self_test=True
    )
    assert result["status"] == "complete"
    assert _tree_hashes(b_manifest.parent) == b_before
    assert Path(result["baseline_b_global_evaluation"]["result"]).is_relative_to(c_root)
    selection = Path(result["selection_plan"]["path"])
    selection.write_bytes(selection.read_bytes() + b"\n")

    with pytest.raises(RUNNER.V31CRunError, match="selection plan SHA-256"):
        RUNNER.run(b_manifest, c_root, workers=2, resume=True, self_test=True)


def test_collect_uses_strict_owner_validity_for_topology_not_decoder_validity(
    tmp_path, monkeypatch
):
    baseline = np.zeros((2, 2), dtype=np.int16)
    probabilities = np.zeros((len(RUNNER.CLASS_ORDER), 2, 2), dtype=np.float32)
    probabilities[0] = 1.0
    decoder_valid = np.ones((2, 2), dtype=bool)
    strict = np.array([[True, True], [True, False]], dtype=bool)
    captured = {}

    monkeypatch.setattr(
        RUNNER.B_RUNNER,
        "_stitch",
        lambda *_args: (
            baseline,
            probabilities,
            decoder_valid,
            strict,
            {"x0": 0, "y0": 0, "x1": 2, "y1": 2},
            [],
        ),
    )
    monkeypatch.setattr(
        RUNNER.B_RUNNER,
        "_physical",
        lambda *_args: {"pixel_area_m2": 1.0, "row_step_m": 1.0, "column_step_m": 1.0},
    )

    def fake_collect(_baseline, **kwargs):
        captured["valid_mask"] = kwargs["valid_mask"].copy()
        return [], {
            "canonical_generated": 0,
            "raw_generated": 0,
            "duplicate_proposal_count": 0,
            "cross_core_discovery_count": 0,
            "collection_rejection_events": {},
            "generation_rejection_events": {},
        }

    monkeypatch.setattr(RUNNER, "collect_cross_core_discoveries", fake_collect)
    RUNNER._collect_partition(
        {
            "entry": {
                "partition_id": "only",
                "core_window": {"x0": 0, "y0": 0, "x1": 2, "y1": 2},
            },
            "entries": [
                {
                    "partition_id": "only",
                    "core_window": {"x0": 0, "y0": 0, "x1": 2, "y1": 2},
                }
            ],
            "global_window": {"x0": 0, "y0": 0, "x1": 2, "y1": 2},
            "transform": [1, 0, 0, 0, -1, 2],
            "crs": "EPSG:3857",
            "collect_root": str(tmp_path),
            "fingerprint": "f",
            "policy": RUNNER.v31b_policy(),
        }
    )

    assert np.array_equal(captured["valid_mask"], strict)


def test_empty_b_core_has_explicit_zero_remaining_budgets():
    b_parts = {
        "empty": {
            "valid_pixel_count": 0,
            "audit_data": {
                "coverage": {"core_strict_valid_pixel_count": 0},
                "v31b_audit": {
                    "skipped": True,
                    "reason": "empty_owner_core_strict_valid",
                },
            },
        }
    }

    source, target = RUNNER._budget_remaining(b_parts)

    assert len(source) == len(target) == len(RUNNER.CLASS_ORDER)
    assert set(source.values()) == set(target.values()) == {0}


def test_running_predecessor_can_reuse_collect_only_for_the_audited_runner_fix():
    runner_key = "scratch/v31c_full_140_20260825/run_v31c_global.py"
    current = {
        "candidate_label": "C",
        "input": "same",
        "code_sha256": {runner_key: "new", "candidate.py": "same"},
    }
    prior_payload = {
        **current,
        "code_sha256": {
            runner_key: next(iter(RUNNER.COLLECT_COMPATIBLE_PREDECESSOR_RUNNER_SHA256)),
            "candidate.py": "same",
        },
    }
    prior = {
        "status": "running",
        "execution_fingerprint": prior_payload,
        "execution_fingerprint_sha256": RUNNER._sha256_json(prior_payload),
    }

    assert RUNNER._compatible_collect_fingerprint(prior, current) == prior[
        "execution_fingerprint_sha256"
    ]
    changed = {**current, "code_sha256": {runner_key: "new", "candidate.py": "changed"}}
    with pytest.raises(RUNNER.V31CRunError, match="collection dependencies"):
        RUNNER._compatible_collect_fingerprint(prior, changed)
