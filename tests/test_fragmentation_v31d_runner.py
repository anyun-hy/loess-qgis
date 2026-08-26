import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scratch" / "v31d_full_140_20260825" / "run_v31d_second_round.py"
SPEC = importlib.util.spec_from_file_location("_test_v31d_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def _method(dynamic_count):
    return {
        "components_total": 1000,
        "dynamic_fragments": {"count": dynamic_count, "area_m2": 100.0},
        "boundary": {"total_cross_class_boundary": {"edges": 50, "metres": 40.0}},
        "per_class": [
            {"class_code": code, "components": 1} for code in RUNNER.CLASS_ORDER
        ],
    }


def _result(dynamic_count):
    return {
        "coverage": {
            "core_overlap_pixels": 0,
            "geometric_coverage_gap_pixels": 0,
            "outside_valid_label_pixels": {"raw": 0, "v3": 0, "v31": 0},
            "invalid_label_inside_valid_pixels": {"raw": 0, "v3": 0, "v31": 0},
        },
        "methods": {"v31": _method(dynamic_count)},
    }


def test_real_b_effect_gate_rejects_129_and_accepts_130_dynamic_fragments():
    b = _result(25983)
    below = RUNNER._validate_results(b, _result(25854))
    at_gate = RUNNER._validate_results(b, _result(25853))

    assert below["effect_gate"] == {
        "fraction": .005,
        "required_dynamic_fragment_reduction": 130,
        "actual_dynamic_fragment_reduction": 129,
        "pass": False,
    }
    assert below["acceptance_pass"] is False
    assert at_gate["effect_gate"]["actual_dynamic_fragment_reduction"] == 130
    assert at_gate["acceptance_pass"] is True


def test_four_core_runner_self_test_completes_and_resume_is_stable():
    result = RUNNER._self_test(None, workers=2)

    assert result["status"] == "complete"
    assert result["candidate_label"] == "D"
    assert result["completed_partition_count"] == 4
    assert result["validation_pass"] is True


def test_safety_gate_rejects_boundary_growth_even_with_fragment_gain():
    b = _result(25983)
    d = _result(25800)
    d["methods"]["v31"]["boundary"]["total_cross_class_boundary"] = {
        "edges": 51,
        "metres": 40.1,
    }

    validation = RUNNER._validate_results(b, d)

    assert validation["effect_gate"]["pass"] is True
    assert validation["safety_gates"]["boundary_edges_nonincreasing"] is False
    assert validation["safety_gates"]["boundary_metres_nonincreasing"] is False
    assert validation["acceptance_pass"] is False
