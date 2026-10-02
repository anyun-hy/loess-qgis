"""Native Qt/QGIS regressions, launched from the Conda test environment."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "scenario",
    [
        "shutdown",
        "monitor_query_shutdown",
        "monitor_dialog_shutdown",
        "sam_shutdown",
        "preparation",
        "workspace",
        "result_layers",
        "edits",
        "manual_candidates",
        "refinement",
        "refinement_freshness",
        "monitor",
        "monitor_snapshot_contracts",
        "monitor_page_queries",
        "monitor_page_selection",
        "monitor_tables",
        "monitor_typography",
        "monitor_combos",
    ],
)
def test_native_ui_design(scenario):
    environment = os.environ.copy()
    if environment.get("LOESS_TEST_QGIS_PYTHONPATH"):
        environment["PYTHONPATH"] = environment["LOESS_TEST_QGIS_PYTHONPATH"]
    # Headless test process only; never changes the user's QGIS/Wayland session.
    environment["QT_QPA_PLATFORM"] = "offscreen"
    probe = (
        "monitor_pages_probe.py"
        if scenario in {
            "monitor_snapshot_contracts",
            "monitor_page_queries",
            "monitor_page_selection",
        }
        else "qt_ui_design_probe.py"
    )
    if scenario == "edits":
        probe = "edit_tracking_probe.py"
    result = subprocess.run(
        [
            environment.get("LOESS_TEST_QGIS_PYTHON", sys.executable),
            "-B",
            str(ROOT / "tests/support" / probe),
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
    print(result.stdout.strip())


def test_gui_shutdown_paths_do_not_wait():
    runner = (ROOT / "src/labeling_tool/runs/v5_async_runner.py").read_text()
    facade = runner.split("class ThreadedV5AsyncInferenceRunner", 1)[1]
    assert ".wait(" not in facade.replace(".wait(0)", "")
    monitor = (ROOT / "src/labeling_tool/monitor/inference_monitor.py").read_text()
    assert ".wait(" not in monitor.split("def shutdown", 1)[1].replace(".wait(0)", "")
    sam = (ROOT / "src/labeling_tool/refinement/sam3_worker_runner.py").read_text()
    assert "waitFor" not in sam


def test_monitor_theme_is_scoped_to_the_dialog():
    """A monitor theme must never restyle QGIS's application-wide widgets."""

    monitor = (ROOT / "src/labeling_tool/monitor/inference_monitor.py").read_text()
    assert "QApplication.setStyleSheet" not in monitor


def test_start_freeze_uses_a_detached_source_task():
    source = (ROOT / "src/labeling_tool/runs/run_workflow.py").read_text()
    preparation = source.split("def _start_preparation", 1)[1].split(
        "def _matches_preparation", 1
    )[0]
    assert "RunPreparationTask(" in preparation
    assert "QgsVectorLayerFeatureSource(" in preparation
    assert "self._freeze_pending_range_snapshot(" not in preparation
    assert "self._freeze_pending_accepted_snapshot(" not in preparation
