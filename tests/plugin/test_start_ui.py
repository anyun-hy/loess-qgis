"""Focused native regression checks for fixed narrow-dock actions and readiness."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_native_start_ui_readiness_and_narrow_layout():
    environment = os.environ.copy()
    if environment.get("LOESS_TEST_QGIS_PYTHONPATH"):
        environment["PYTHONPATH"] = environment["LOESS_TEST_QGIS_PYTHONPATH"]
    environment["QT_QPA_PLATFORM"] = "offscreen"
    with tempfile.TemporaryDirectory(prefix="loess-start-ui-") as directory:
        result = subprocess.run(
            [
                environment.get("LOESS_TEST_QGIS_PYTHON", sys.executable),
                "-B",
                str(ROOT / "tests/support/start_ui_probe.py"),
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
        assert result.stdout.strip().endswith("start_ui: passed"), result.stdout
        for name in ("start-ui-450x800.png", "start-ui-400x800.png"):
            screenshot = Path(directory) / name
            assert screenshot.is_file() and screenshot.stat().st_size > 0
