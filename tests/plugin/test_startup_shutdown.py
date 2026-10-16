"""Startup callbacks must respect the real dock's shutdown boundary."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "scenario",
    [
        "early_unload",
        "early_quit",
        "late_environment_report",
        "cleanup_started",
        "normal_startup",
        "active_workflow",
    ],
)
def test_startup_callbacks_during_shutdown(scenario):
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    if env.get("LOESS_TEST_QGIS_PYTHONPATH"):
        env["PYTHONPATH"] = env["LOESS_TEST_QGIS_PYTHONPATH"]
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [
            env.get("LOESS_TEST_QGIS_PYTHON", sys.executable),
            "-B",
            str(root / "tests/support/startup_shutdown_probe.py"),
            str(root),
            scenario,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode == 77 and "LOESS_TEST_QGIS_PYTHON" not in env:
        pytest.skip("Native QGIS unavailable; set LOESS_TEST_QGIS_PYTHON")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith(scenario + ": passed"), result.stdout
