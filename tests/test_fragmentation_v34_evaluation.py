from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load(
    "_test_v34_eval_runner",
    "scratch/v34_full_140_20260826/run_v34_from_v33.py",
)
EVALUATOR = _load(
    "_test_v34_evaluator",
    "scratch/v34_full_140_20260826/evaluate_v34.py",
)


def test_v34_four_core_global_evaluation_is_safe_no_effect(monkeypatch, tmp_path):
    parent = RUNNER._B._write_self_test_parent(tmp_path)
    v33_root = tmp_path / "v33"
    v34_root = tmp_path / "v34"
    RUNNER.V33_RUNNER.run(parent, v33_root, workers=2, resume=False, self_test=True)
    RUNNER.run(
        v33_root / "run_manifest.json", v34_root,
        workers=2, resume=False, self_test=True,
    )
    monkeypatch.setattr(EVALUATOR, "REAL_PARTITION_COUNT", 4)
    monkeypatch.setattr(EVALUATOR, "REAL_STRICT_VALID_TOTAL", 4 * 12 * 12)
    result = EVALUATOR.evaluate(
        v33_root / "run_manifest.json",
        v34_root / "run_manifest.json",
        tmp_path / "comparison",
    )
    assert result["validation_pass"] is True
    assert result["effect_pass"] is False
    assert result["status"] == "safe_no_effect"
    assert result["direct_transition"]["changed_pixels"] == 0
    assert result["candidate_audit_summary"]["cumulative_budgets_pass"] is True
    assert (tmp_path / "comparison" / "V34_V33_COMPARISON.json").is_file()


def test_empty_owner_core_without_budget_rows_is_not_a_budget_failure(tmp_path):
    parent_audit = {"path": "/frozen/v33/audit.json", "sha256": "parent-sha"}
    stage = {
        "parent_v33_partition_audit": parent_audit,
        "coverage": {"core_strict_valid_pixel_count": 0},
        "v34_audit": {
            "full_audit": True,
            "audit_truncated": False,
            "changed_pixel_count": 0,
            "protected_source_loss_pixel_count": 0,
            "gap_pixels": 0,
            "overlap_pixels": 0,
            "outside_pixels": 0,
            "raw_generated": 0,
            "proposals_canonical": 0,
            "duplicate_proposal_count": 0,
            "proposals_accepted": 0,
        },
    }
    audit_path = tmp_path / "audit.json"
    EVALUATOR._atomic_json(audit_path, stage)
    summary = EVALUATOR._audit(
        {"empty": {"audit": parent_audit}},
        {"empty": {"audit": {"path": str(audit_path), "sha256": EVALUATOR._sha_file(audit_path)}}},
    )
    assert summary["cumulative_budgets_pass"] is True
    assert summary["parent_partition_audit_match"] is True
