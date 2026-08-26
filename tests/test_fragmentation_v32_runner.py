from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scratch" / "v32_full_140_20260825" / "run_v32_from_v3.py"
SPEC = importlib.util.spec_from_file_location("_test_v32_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def test_four_core_runner_self_test_publishes_only_v32_and_resume_is_stable():
    result = RUNNER._self_test(None, workers=2)

    assert result["status"] == "complete"
    assert result["candidate_label"] == "V3.2"
    assert result["completed_partition_count"] == 4
    assert result["frozen_global_class_pixel_totals_sum"] == 4 * 12 * 12
    assert all(item["outputs"]["v31a"]["stage"] == "v32" for item in result["partitions"])
    assert "inference_scripts/fragmentation_v31_candidate/v32.py" in result["code_sha256"]
    assert "inference_scripts/fragmentation_v31_candidate/candidate.py" in result["code_sha256"]
    assert result["resource_plan"]["workers"] == 2
    assert result["proposal_counts"]["raw_generated"] >= result["proposal_counts"]["canonical_accepted"]


def test_full_totals_counts_each_strict_owner_core_once_and_real_guard_is_enforced():
    with tempfile.TemporaryDirectory(prefix="v32-runner-test-") as temporary:
        parent = RUNNER._B._write_self_test_parent(Path(temporary))
        source = RUNNER._B._load_parent(parent, self_test=True)
        totals = RUNNER._full_totals(source["entries"], self_test=True)

        assert sum(totals.values()) == 576
        assert totals[13] == 576
        with pytest.raises(RUNNER.V32RunError, match="831531565"):
            RUNNER._full_totals(source["entries"], self_test=False)
