"""Final plugin unload must finish native workers before QGIS tears down Qt."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "scenario",
    [
        "completed_run",
        "busy_control",
        "cleanup_error",
        "monitor_cleanup_error",
        "registry_skip",
    ],
)
def test_real_qgis_desktop_exit(tmp_path, scenario):
    executable = os.environ.get("LOESS_TEST_QGIS_DESKTOP")
    if not executable:
        pytest.skip(
            "Set LOESS_TEST_QGIS_DESKTOP for native Cocoa/Wayland desktop tests"
        )
    environment = os.environ.copy()
    for key in (
        "PYTHONHOME",
        "PYTHONPATH",
        "QT_PLUGIN_PATH",
        "QT_QPA_PLATFORM_PLUGIN_PATH",
    ):
        environment.pop(key, None)
    environment.update(
        LOESS_DESKTOP_TEST_ROOT=str(ROOT),
        LOESS_DESKTOP_TEST_RESULT=str(tmp_path / "result.json"),
        LOESS_DESKTOP_TEST_SCENARIO=scenario,
        LOESS_DESKTOP_TEST_PLUGIN_ROOT=environment.get(
            "LOESS_TEST_QGIS_PLUGIN_ROOT", ""
        ),
        LOESS_DESKTOP_TEST_SCRIPTS_DIR=environment.get(
            "LOESS_TEST_QGIS_SCRIPTS_DIR", ""
        ),
        QT_QPA_PLATFORM="cocoa" if sys.platform == "darwin" else "wayland",
        PYTHONDONTWRITEBYTECODE="1",
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_CACHE_HOME=str(tmp_path / "cache"),
        XDG_DATA_HOME=str(tmp_path / "data"),
    )
    # Seed only the disposable profile, before QGIS starts its font task.
    organization = "qgis.org" if sys.platform == "darwin" else "QGIS"
    ini = tmp_path / "profiles/profiles/shutdown" / organization / "QGIS4.ini"
    ini.parent.mkdir(parents=True)
    ini.write_text("[fonts]\ndownloadMissingFonts=false\n")
    result = subprocess.run(
        [
            executable,
            "--nologo",
            "--noversioncheck",
            "--noplugins",
            "--profiles-path",
            str(tmp_path / "profiles"),
            "--profile",
            "shutdown",
            "--code",
            str(ROOT / "tests/support/qgis_desktop_shutdown_probe.py"),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=40,
    )
    report_file = tmp_path / "result.json"
    report = json.loads(report_file.read_text()) if report_file.exists() else {}
    assert result.returncode == 0, (report, result.stdout, result.stderr)
    assert report.get("status") == "passed", (report, result.stdout, result.stderr)
    assert report["live_threads_after_quit"] == []
    assert report["awaiting_shutdown"] == 0
    if scenario in {"cleanup_error", "monitor_cleanup_error"}:
        assert report["injected_cleanup_error"]


@pytest.mark.parametrize("scenario", ["quit_idle", "quit_busy", "unload_busy"])
def test_native_plugin_shutdown(scenario):
    environment = os.environ.copy()
    if environment.get("LOESS_TEST_QGIS_PYTHONPATH"):
        environment["PYTHONPATH"] = environment["LOESS_TEST_QGIS_PYTHONPATH"]
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [
            environment.get("LOESS_TEST_QGIS_PYTHON", sys.executable),
            "-B",
            str(ROOT / "tests/support/plugin_shutdown_probe.py"),
            str(ROOT),
            scenario,
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode == 77 and "LOESS_TEST_QGIS_PYTHON" not in environment:
        pytest.skip("Native QGIS unavailable; set LOESS_TEST_QGIS_PYTHON")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith(scenario + ": passed"), result.stdout
