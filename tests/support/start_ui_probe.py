# ruff: noqa: E402
"""Native narrow-dock acceptance using report and input doubles only."""

from __future__ import annotations

import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import PropertyMock, patch

ROOT = Path(sys.argv[1])
OUTPUT = Path(sys.argv[2])
sys.path.insert(0, str(ROOT / "src"))
try:
    from qgis.core import QgsApplication
    from qgis.PyQt.QtCore import QCoreApplication, QEvent, pyqtSignal
    from qgis.PyQt.QtWidgets import QDialog, QScrollArea
except ModuleNotFoundError:
    raise SystemExit(77)


def _drain(app) -> None:
    app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _report(model_status: str = "ready") -> dict:
    return {
        "status": "ready",
        "checks": [{"id": "semantic_model_fixture", "status": model_status}],
        "effective": {
            "schema_version": 2,
            "semantic_models": [
                {"model_id": "fixture", "display_name": "布局检查模型"}
            ],
            "runtime": {"effective_device": "cpu"},
        },
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    app = QgsApplication([], False)
    app.initQgis()
    from labeling_tool.main import main_dock

    class Monitor(QDialog):
        stop_requested = pyqtSignal()
        request_main_run_handling = pyqtSignal(object)

        def detach(self):
            pass

        def shutdown(self):
            self.close()

    with ExitStack() as patches:
        for name in (
            "_save_settings",
            "_load_settings_and_defaults",
            "_restore_latest_ready_run",
        ):
            patches.enter_context(
                patch.object(main_dock.LabelingDockWidget, name, lambda *_: None)
            )
        patches.enter_context(patch.object(main_dock, "InferenceMonitorDialog", Monitor))
        patches.enter_context(
            patch.object(main_dock, "validate_raster_layer", return_value=object())
        )
        dock = main_dock.LabelingDockWidget()
        try:
            dock.resize(400, 800)
            dock.show()
            _drain(app)
            assert dock.start_btn.isVisible() and dock.stop_btn.isVisible()
            assert dock.load_manual_run_btn.isVisible()
            initial = OUTPUT / "start-ui-400x800.png"
            assert dock.grab().save(str(initial))
            assert initial.is_file() and initial.stat().st_size > 0

            dock.workspace_edit.setText("/tmp/loess-ui-workspace")
            dock.output_path_edit.setText("/tmp/loess-ui-accepted.gpkg")
            data = _report()
            dock.config_manager.last_report = data
            dock._environment_report_current = True
            dock.plan_panel.set_environment(data)
            dock.plan_panel._apply_configuration(["fixture"], None, True)

            with patch.object(
                dock,
                "_range_readiness",
                return_value=("", ("开始时将自动读取当前视图范围",)),
            ):
                dock._update_start_enabled()
                assert dock.start_btn.isEnabled()
                assert "自动读取当前视图范围" in dock.start_readiness_label.text()

                dock._environment_report_current = False
                dock._update_start_enabled()
                assert not dock.start_btn.isEnabled()
                assert "报告已过期" in dock.start_readiness_label.text()

                dock._environment_report_current = True
                with patch.object(
                    dock,
                    "_range_readiness",
                    return_value=("当前范围与影像层没有重叠", ()),
                ):
                    dock._update_start_enabled()
                    assert not dock.start_btn.isEnabled()
                    assert "没有重叠" in dock.start_readiness_label.text()

                with patch.object(
                    dock,
                    "_range_readiness",
                    return_value=("", ("开始时将自动读取当前视图范围",)),
                ), patch.object(
                    type(dock.workflow),
                    "is_active",
                    new_callable=PropertyMock,
                    return_value=True,
                ):
                    dock._update_start_enabled()
                    assert not dock.start_btn.isEnabled()
                    assert "当前任务正在运行或停止中" in dock.start_readiness_label.text()

                dock._environment_report_current = True
                dock._update_start_enabled()
                assert dock.start_btn.isEnabled()
                assert "自动读取当前视图范围" in dock.start_readiness_label.text()
                dock.resize(450, 800)
                dock.show()
                _drain(app)
                assert dock.start_btn.isVisible() and dock.stop_btn.isVisible()
                assert dock.load_manual_run_btn.isVisible()
                assert dock.findChild(QScrollArea) is not None
                screenshot = OUTPUT / "start-ui-450x800.png"
                assert dock.grab().save(str(screenshot))
                assert screenshot.is_file() and screenshot.stat().st_size > 0
        finally:
            dock.cleanup()
            dock.deleteLater()
            _drain(app)
    app.exitQgis()


if __name__ == "__main__":
    main()
    print("start_ui: passed")
