"""Native regression for the refinement dialog admission-summary wiring."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_native_refinement_admission_summary():
    environment = os.environ.copy()
    if environment.get("LOESS_TEST_QGIS_PYTHONPATH"):
        environment["PYTHONPATH"] = environment["LOESS_TEST_QGIS_PYTHONPATH"]
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [
            environment.get("LOESS_TEST_QGIS_PYTHON", sys.executable),
            "-B",
            str(ROOT / "tests/support/refinement_admission_summary_probe.py"),
            str(ROOT),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode == 77 and "LOESS_TEST_QGIS_PYTHON" not in environment:
        return
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith("refinement admission summary: passed"), result.stdout
