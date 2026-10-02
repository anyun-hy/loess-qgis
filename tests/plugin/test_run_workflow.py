"""Run orchestration is exercised with native Qt signals and synthetic I/O."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "scenario",
    [
        "start_and_finish",
        "stop_before_probe",
        "stop_preparation",
        "stop_builder",
        "stop_inference",
        "shutdown_preparation",
        "shutdown_builder",
        "stale_callbacks",
        "failures",
        "recovery",
        "reservation_guards",
        "dock_integration",
        "native_task_shutdown",
    ],
)
def test_native_run_workflow(scenario):
    environment = os.environ.copy()
    if environment.get("LOESS_TEST_QGIS_PYTHONPATH"):
        environment["PYTHONPATH"] = environment["LOESS_TEST_QGIS_PYTHONPATH"]
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [
            environment.get("LOESS_TEST_QGIS_PYTHON", sys.executable),
            "-B",
            str(ROOT / "tests/support/run_workflow_probe.py"),
            str(ROOT),
            scenario,
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=35,
    )
    if result.returncode == 77 and "LOESS_TEST_QGIS_PYTHON" not in environment:
        pytest.skip("Native QGIS unavailable; set LOESS_TEST_QGIS_PYTHON")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith(scenario + ": passed"), result.stdout


def test_workflow_owns_runtime_without_importing_presenters():
    source = (ROOT / "src/labeling_tool/runs/run_workflow.py").read_text()
    tree = ast.parse(source)
    imports = [
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    forbidden = (
        "labeling_tool.main.main_dock",
        "labeling_tool.monitor",
        "labeling_tool.qgis_support.layer_manager",
        "qgis.PyQt.QtWidgets",
    )
    assert not [name for name in imports if name.startswith(forbidden)]
    dock = ast.parse((ROOT / "src/labeling_tool/main/main_dock.py").read_text())
    old_owners = {
        "runner",
        "_pending_run",
        "_tile_cache_probe",
        "_run_preparation_task",
        "_run_builder_task",
        "_pipeline_state",
        "_pipeline_running",
    }
    dock_members = {
        node.attr
        for node in ast.walk(dock)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }
    assert not old_owners & dock_members
