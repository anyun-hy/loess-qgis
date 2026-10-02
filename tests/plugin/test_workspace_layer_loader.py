"""Native Qt queue scheduling and QGIS dialog loading regressions."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "scenario",
    [
        "sequential",
        "priority_pause",
        "failure_retry",
        "reset_during_load",
        "loaded_reset",
        "completed_restart",
        "pause_during_load",
        "dialog_lifecycle",
    ],
)
def test_native_workspace_layer_loader(scenario):
    environment = os.environ.copy()
    if environment.get("LOESS_TEST_QGIS_PYTHONPATH"):
        environment["PYTHONPATH"] = environment["LOESS_TEST_QGIS_PYTHONPATH"]
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [
            environment.get("LOESS_TEST_QGIS_PYTHON", sys.executable),
            "-B",
            str(ROOT / "tests/support/workspace_layer_loader_probe.py"),
            str(ROOT),
            scenario,
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=40,
    )
    if result.returncode == 77 and "LOESS_TEST_QGIS_PYTHON" not in environment:
        pytest.skip("Native QGIS unavailable; set LOESS_TEST_QGIS_PYTHON")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith(scenario + ": passed"), result.stdout
