"""Native QgsTask ownership, cancellation and workspace dialog regressions."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "scenario",
    [
        "replacement",
        "cancel_lifetime",
        "terminal_reentry",
        "submission_failure",
        "late_cancelled_success",
        "thread_affinity",
        "dialog_lifecycle",
        "dialog_confirmation",
    ],
)
def test_native_workspace_tasks(scenario):
    environment = os.environ.copy()
    if environment.get("LOESS_TEST_QGIS_PYTHONPATH"):
        environment["PYTHONPATH"] = environment["LOESS_TEST_QGIS_PYTHONPATH"]
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [
            environment.get("LOESS_TEST_QGIS_PYTHON", sys.executable),
            "-B",
            str(ROOT / "tests/support/workspace_tasks_probe.py"),
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
