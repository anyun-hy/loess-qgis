"""Native screenshots and layout checks for the focused model-selection dialog."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_native_model_config_layout_and_details_scroll():
    environment = os.environ.copy()
    if environment.get("LOESS_TEST_QGIS_PYTHONPATH"):
        environment["PYTHONPATH"] = environment["LOESS_TEST_QGIS_PYTHONPATH"]
    environment["QT_QPA_PLATFORM"] = "offscreen"
    with tempfile.TemporaryDirectory(prefix="loess-model-config-ui-") as directory:
        result = subprocess.run(
            [
                environment.get("LOESS_TEST_QGIS_PYTHON", sys.executable),
                "-B",
                str(ROOT / "tests/support/model_config_ui_probe.py"),
                str(ROOT),
                directory,
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=40,
        )
        if result.returncode == 77 and "LOESS_TEST_QGIS_PYTHON" not in environment:
            pytest.skip("Native QGIS unavailable; set LOESS_TEST_QGIS_PYTHON")
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.strip().endswith("model_config_ui: passed"), result.stdout
        for name in (
            "model-config-900x560.png",
            "model-config-1120x680.png",
            "model-config-900x560-details.png",
        ):
            screenshot = Path(directory) / name
            assert screenshot.is_file() and screenshot.stat().st_size > 0
