"""Native regressions for temporary class-layer construction."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("scenario", ["invalid_provider", "background_batches"])
def test_memory_class_layer(scenario):
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    if env.get("LOESS_TEST_QGIS_PYTHONPATH"):
        env["PYTHONPATH"] = env["LOESS_TEST_QGIS_PYTHONPATH"]
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [
            env.get("LOESS_TEST_QGIS_PYTHON", sys.executable),
            "-B",
            str(root / "tests/support/memory_class_layer_probe.py"),
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
    assert result.stdout.strip().endswith(scenario + ": passed"), result.stdout
