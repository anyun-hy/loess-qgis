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


B_RUNNER = _load("_test_v32_eval_b", "scratch/v31b_full_140_20260824/run_v31b_from_v3.py")
V32_RUNNER = _load("_test_v32_eval_runner", "scratch/v32_full_140_20260825/run_v32_from_v3.py")
COMPARATOR = _load("_test_v32_eval_comparator", "scratch/v32_full_140_20260825/evaluate_v32_against_b.py")


def test_complete_four_core_b_v32_global_comparison(monkeypatch, tmp_path):
    parent = B_RUNNER._write_self_test_parent(tmp_path)
    b_root = tmp_path / "b"
    v32_root = tmp_path / "v32"
    B_RUNNER.run(parent, b_root, workers=2, resume=False, self_test=True)
    V32_RUNNER.run(parent, v32_root, workers=2, resume=False, self_test=True)
    monkeypatch.setattr(COMPARATOR, "REAL_PARTITION_COUNT", 4)
    monkeypatch.setattr(COMPARATOR, "REAL_STRICT_VALID_TOTAL", 4 * 12 * 12)

    result = COMPARATOR.evaluate(
        b_root / "run_manifest.json",
        v32_root / "run_manifest.json",
        tmp_path / "evaluation",
    )

    assert result["validation_pass"] is True
    assert result["acceptance_pass"] is True
    assert result["direct_b_to_v32"]["changed_pixels"] == 0
    assert result["candidate_audit_summary"]["protected_source_loss_pixel_count"] == 0
    assert (tmp_path / "evaluation" / "V32_V31B_COMPARISON.json").is_file()
