"""Native Qt wiring checks for runner-owned monitor history recording."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "scenario",
    [
        "structured_history_wiring",
        "process_lifecycle",
    ],
)
def test_native_run_monitor_history_wiring(scenario: str, request) -> None:
    database_arguments = []
    if scenario == "process_lifecycle":
        postgres_database = request.getfixturevalue("postgres_database")
        postgres_database.run_streams.create_run(
            "native-run",
            "a" * 64,
            status="planned",
        )
        database_arguments = [
            postgres_database.session.location,
            postgres_database.session.schema,
        ]
    environment = os.environ.copy()
    if environment.get("LOESS_TEST_QGIS_PYTHONPATH"):
        environment["PYTHONPATH"] = environment["LOESS_TEST_QGIS_PYTHONPATH"]
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [
            environment.get("LOESS_TEST_QGIS_PYTHON", sys.executable),
            "-B",
            str(ROOT / "tests/support/run_monitor_history_probe.py"),
            str(ROOT / "src"),
            str(ROOT / "scripts/runtime"),
            scenario,
            *database_arguments,
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode == 77 and "LOESS_TEST_QGIS_PYTHON" not in environment:
        pytest.skip("Native QGIS unavailable; set LOESS_TEST_QGIS_PYTHON")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith(f"{scenario}: passed"), result.stdout
