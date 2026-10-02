"""Native background I/O acceptance with synthetic files only."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "scenario",
    [
        "loader_responsive",
        "loader_cancel",
        "commit_cancel_boundary",
        "accepted_success",
        "accepted_cancel_wait",
        "accepted_failure_rollback",
        "accepted_commit_failure_rollback",
        "accepted_external_writer_blocked",
        "accepted_new_target_no_overwrite",
        "accepted_spec_changed_during_verification",
        "accepted_missing_ogr_binding",
        "accepted_internal_wal_identity",
        "accepted_post_publish_warning",
        "dialog_lifecycle",
        "monitor_navigation",
    ],
)
def test_native_background_io(scenario):
    environment = os.environ.copy()
    if environment.get("LOESS_TEST_QGIS_PYTHONPATH"):
        environment["PYTHONPATH"] = environment["LOESS_TEST_QGIS_PYTHONPATH"]
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [
            environment.get("LOESS_TEST_QGIS_PYTHON", sys.executable),
            "-B",
            str(ROOT / "tests/support/background_io_tasks_probe.py"),
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
