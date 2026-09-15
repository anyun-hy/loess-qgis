"""Native Qt/QGIS regressions, launched from the Conda test environment."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "scenario",
    ["shutdown", "sam_shutdown", "preparation", "workspace", "result_layers", "edits", "manual_candidates", "refinement", "monitor", "monitor_tables", "monitor_typography", "monitor_combos"],
)
def test_native_ui_design(scenario):
    environment = os.environ.copy()
    if environment.get("LOESS_TEST_QGIS_PYTHONPATH"):
        environment["PYTHONPATH"] = environment["LOESS_TEST_QGIS_PYTHONPATH"]
    # Headless test process only; never changes the user's QGIS/Wayland session.
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [environment.get("LOESS_TEST_QGIS_PYTHON", sys.executable), "-B",
         str(ROOT / "tests/qt_ui_design_probe.py"), str(ROOT), scenario],
        env=environment, capture_output=True, text=True, timeout=40,
    )
    if result.returncode == 77 and "LOESS_TEST_QGIS_PYTHON" not in environment:
        pytest.skip("Native QGIS unavailable; set LOESS_TEST_QGIS_PYTHON")
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout.strip())


def test_gui_shutdown_paths_do_not_wait():
    runner = (ROOT / "qgis_plugins/labeling_tool/core/v5_async_runner.py").read_text()
    facade = runner.split("class ThreadedV5AsyncInferenceRunner", 1)[1]
    assert ".wait(" not in facade.replace(".wait(0)", "")
    monitor = (ROOT / "qgis_plugins/labeling_tool/gui/inference_monitor.py").read_text()
    assert ".wait(" not in monitor.split("def shutdown", 1)[1].replace(".wait(0)", "")
    sam = (ROOT / "qgis_plugins/labeling_tool/core/sam3_worker_runner.py").read_text()
    assert "waitFor" not in sam


def test_monitor_theme_is_scoped_to_the_dialog():
    """A monitor theme must never restyle QGIS's application-wide widgets."""

    monitor = (ROOT / "qgis_plugins/labeling_tool/gui/inference_monitor.py").read_text()
    assert "QApplication.setStyleSheet" not in monitor


def test_start_freeze_uses_a_detached_source_task():
    source = (ROOT / "qgis_plugins/labeling_tool/gui/main_dock.py").read_text()
    preparation = source.split("def _start_inference_after_tile_cache_probe", 1)[1].split("def _on_run_builder_progress", 1)[0]
    assert "RunPreparationTask(" in preparation
    assert "QgsVectorLayerFeatureSource(" in preparation
    assert "self._freeze_pending_range_snapshot(" not in preparation
    assert "self._freeze_pending_accepted_snapshot(" not in preparation
