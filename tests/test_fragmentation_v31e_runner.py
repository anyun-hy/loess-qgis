from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT / "scratch" / "v31e_plan_140_20260825" / "run_v31e_plan_only.py"
)


def _runner():
    name = "_test_v31e_plan_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _audit(effect: bool, headroom: bool = False):
    return {
        "effect_gate": {"pass": effect},
        "engineering_headroom_gate": {"pass": headroom},
    }


def test_plan_status_stops_this_fixed_model_when_its_optimum_is_too_low():
    runner = _runner()
    assert (
        runner._status(_audit(False), _audit(False), _audit(False))
        == "fixed_independent_action_model_below_130"
    )
    assert (
        runner._status(_audit(True), _audit(False), _audit(False))
        == "fixed_boundary_independent_action_model_below_130"
    )


def test_plan_status_never_calls_relaxed_dependency_bound_safe():
    runner = _runner()
    assert (
        runner._status(_audit(True), _audit(True), _audit(False))
        == "dependency_replay_required_before_publication"
    )
    assert (
        runner._status(_audit(True), _audit(True), _audit(True, False))
        == "strict_safe_plan_reaches_130_without_headroom"
    )
    assert (
        runner._status(_audit(True), _audit(True), _audit(True, True))
        == "strict_safe_plan_has_headroom"
    )


def test_runner_is_plan_only_and_has_no_publish_stage():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert '"publication_performed": False' in source
    assert "_publish_all(" not in source
    assert "_save_npy(" not in source
    assert "theoretical_ceiling" not in source
    assert "boundary_upper_bound" not in source
