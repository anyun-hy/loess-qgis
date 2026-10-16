"""Native QGIS coverage for unified Run loading with synthetic data only."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "scenario",
    [
        "initialize",
        "copied_initialize",
        "copied_rebuild",
        "classes",
        "all",
        "layers",
        "sam_failure",
        "switch_guard",
    ],
)
def test_unified_run_loading(scenario):
    _run_probe("run_loading_probe.py", scenario)


@pytest.mark.parametrize(
    "scenario",
    [
        "entry_classes",
        "entry_all",
        "entry_cancel",
        "entry_class_failure",
        "entry_class_pause",
        "image_binding",
    ],
)
def test_unified_run_entry(scenario):
    _run_probe("run_entry_probe.py", scenario)


def _run_probe(script, scenario):
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    if env.get("LOESS_TEST_QGIS_PYTHONPATH"):
        env["PYTHONPATH"] = env["LOESS_TEST_QGIS_PYTHONPATH"]
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [
            env.get("LOESS_TEST_QGIS_PYTHON", sys.executable),
            "-B",
            str(root / "tests/support" / script),
            str(root),
            scenario,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    if result.returncode == 77 and "LOESS_TEST_QGIS_PYTHON" not in env:
        pytest.skip("Native QGIS unavailable; set LOESS_TEST_QGIS_PYTHON")
    assert result.returncode == 0, result.stdout + result.stderr
    if script == "run_entry_probe.py":
        assert scenario + ": qgis_shutdown_complete" in result.stdout, result.stdout
    assert result.stdout.strip().endswith(scenario + ": passed")
