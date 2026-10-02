# ruff: noqa: E402
"""Native widget behavior with synthetic reports; no inference, DB or user settings."""

from __future__ import annotations

import sys
import tempfile
import traceback
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
try:
    from qgis.core import QgsApplication
    from qgis.PyQt import sip
    from qgis.PyQt.QtCore import (
        QCoreApplication,
        QEvent,
        QItemSelectionModel,
        pyqtSignal,
    )
    from qgis.PyQt.QtGui import QDesktopServices
    from qgis.PyQt.QtTest import QTest
    from qgis.PyQt.QtWidgets import (
        QApplication,
        QDialog,
        QFileDialog,
        QGroupBox,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QTableWidget,
        QTabWidget,
    )
except ModuleNotFoundError:
    raise SystemExit(77)

from labeling_tool.main.environment_panel import EnvironmentPanel


def drain(app):
    app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def report():
    return {
        "status": "error",
        "check_id": "native-environment-check",
        "stderr": "native stderr\nline 2",
        "process": {"exit_code": 1, "report_source": "report_file"},
        "effective": {"runtime": {"effective_device": "cpu"}},
        "checks": [
            {"id": "dependency_torch", "status": "ready", "value": "fixture"},
            {"id": "warning_first", "status": "warning", "message": "warning one"},
            {
                "id": "semantic_model_fixture",
                "status": "error",
                "message": "model failed\nTraceback: full diagnostic\n" + "x" * 400,
                "fix": "config.yaml",
            },
            {"id": "error_second", "status": "error", "message": "second error"},
            {"id": "unknown_status", "status": "future", "message": "preserve me"},
        ],
    }


def dialogs(panel):
    return [dialog for dialog in panel.findChildren(QDialog) if dialog.isVisible()]


def button(parent, text):
    return next(
        item for item in parent.findChildren(QPushButton) if item.text() == text
    )


def panel_states(app, root):
    panel = EnvironmentPanel()
    requests, changed = [], []
    panel.check_requested.connect(lambda: requests.append("check"))
    panel.details_requested.connect(lambda: requests.append("details"))
    panel.script_path_changed.connect(changed.append)
    try:
        assert not panel.details_button.isEnabled()
        (root / "config.yaml").write_text("schema_version: 2\n")
        panel.scripts_directory = f"  {root}  "
        assert panel.scripts_directory == str(root)
        assert changed and not requests
        panel.mark_check_required()
        assert panel.config_path_label.text() == str(root / "config.yaml")
        assert panel.open_config_button.isEnabled()
        with patch.object(QDesktopServices, "openUrl", return_value=True) as opened:
            panel.open_config_button.click()
            assert opened.call_args.args[0].toLocalFile() == str(root / "config.yaml")
        (root / "config.yaml").unlink()
        with patch.object(QMessageBox, "warning") as warning:
            panel.open_config_button.click()
            assert warning.call_count == 1
            assert "没有 config.yaml" in warning.call_args.args[-1]
        assert "请检查推理环境" in panel.status_label.text()
        panel.check_button.click()
        assert requests == ["check"]
        panel.show_checking()
        assert not panel.check_button.isEnabled()
        assert not panel.details_button.isEnabled()
        assert "正在检查" in panel.status_label.text()
        data = report()
        panel.mark_check_required()
        assert not panel.check_button.isEnabled()
        panel.show_report(data, [], check_finished=False)
        assert not panel.check_button.isEnabled()
        panel.show_report(data, [])
        assert panel.check_button.isEnabled() and panel.details_button.isEnabled()
        assert panel.status_label.text() == "检查未通过：model failed"
        assert "Traceback" not in panel.status_label.text()
        assert not dialogs(panel)
        panel.details_button.click()
        assert requests == ["check", "details"] and not dialogs(panel)
        data["status"] = "warning"
        data["checks"] = [data["checks"][1]]
        panel.show_report(data, [])
        assert panel.status_label.text() == "检查通过但有警告：warning one"
        data["status"] = "ready"
        data["checks"] = []
        panel.show_report(data, [{"status": "error", "message": "workspace missing"}])
        assert panel.status_label.text() == "环境检查通过：配置已加载"
        panel.show_report(data, [])
        assert panel.status_label.text() == "环境检查通过：配置已加载"
        assert panel.details_button.isEnabled()  # stderr alone remains inspectable.
        panel.show_report({"status": "ready"}, [])
        assert not panel.details_button.isEnabled()
        panel.scripts_directory = ""
        panel.mark_check_required()
        assert panel.config_path_label.text() == "未选择"
        assert not panel.open_config_button.isEnabled()
        with patch.object(QFileDialog, "getExistingDirectory", return_value=str(root)):
            panel.browse_button.click()
            assert panel.scripts_directory == str(root)
    finally:
        panel.cleanup()
        panel.deleteLater()
        drain(app)


def details_copy(app, root):
    panel = EnvironmentPanel()
    copied = []
    data = report()
    try:
        with patch.object(
            QApplication,
            "clipboard",
            return_value=SimpleNamespace(setText=copied.append),
        ):
            panel.show_details(data, [])
            dialog = dialogs(panel)[0]
            assert not dialog.isModal()
            table = dialog.findChild(QTableWidget)
            text = dialog.findChild(QPlainTextEdit)
            tabs = dialog.findChild(QTabWidget)
            assert table.rowCount() == 5
            assert [table.item(row, 0).text() for row in range(5)] == [
                "错误",
                "错误",
                "警告",
                "正常",
                "future",
            ]
            assert table.item(1, 1).text() == "error_second"
            full_text = text.toPlainText()
            assert data["checks"][2]["message"] in full_text
            assert "native stderr\nline 2" in full_text
            assert "检查编号: native-environment-check" in full_text
            assert text.isReadOnly()
            # Selection follows the sorted visible rows, not the original report order.
            selection = table.selectionModel()
            for row in (0, 3):
                selection.select(
                    table.model().index(row, 0),
                    QItemSelectionModel.SelectionFlag.Select
                    | QItemSelectionModel.SelectionFlag.Rows,
                )
            data["checks"][2]["message"] = "changed after opening"
            button(dialog, "复制选中项").click()
            assert "[ERROR] semantic_model_fixture" in copied[-1]
            assert "[READY] dependency_torch" in copied[-1]
            assert "warning_first" not in copied[-1]
            assert "Traceback: full diagnostic" in copied[-1]
            button(dialog, "复制全部结果").click()
            assert copied[-1] == full_text
            tabs.setCurrentWidget(text)
            button(dialog, "全选").click()
            assert text.textCursor().hasSelection()
            panel.show_details(report(), [])
            current = dialogs(panel)[0]
            button(current, "复制选中项").click()
            assert copied[-1] == current.findChild(QPlainTextEdit).toPlainText()
            button(current, "全选").click()
            assert (
                len(current.findChild(QTableWidget).selectionModel().selectedRows())
                == 5
            )
    finally:
        panel.cleanup()
        panel.deleteLater()
        drain(app)


def details_lifecycle(app, root):
    panel = EnvironmentPanel()
    with patch.object(
        QApplication, "clipboard", return_value=SimpleNamespace(setText=lambda _: None)
    ):
        panel.show_details(report(), [])
        old = dialogs(panel)[0]
        button(old, "复制全部结果").click()
        panel.show_details(report(), [])
        current = dialogs(panel)[0]
        assert current is not old
        drain(app)
        assert sip.isdeleted(old) and not sip.isdeleted(current)
        panel.cleanup()
        panel.cleanup()
        drain(app)
        assert sip.isdeleted(current)
        panel.show_details(report(), [])
        reopened = dialogs(panel)[0]
        button(reopened, "关闭").click()
        drain(app)
        assert sip.isdeleted(reopened)
        panel.show_details(report(), [])
        panel.cleanup()
        panel.deleteLater()
        drain(app)
        # Expired copy feedback must not call a deleted button or dialog.
        QTest.qWait(2100)


def dock_gate(app, root):
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
        patches.enter_context(
            patch.object(main_dock, "InferenceMonitorDialog", Monitor)
        )
        dock = main_dock.LabelingDockWidget()
        panel = dock.environment_panel
        calls = []

        def start_check(scripts, workspace):
            calls.append((scripts, workspace))
            dock.config_manager.check_started.emit()

        patches.enter_context(
            patch.object(dock.config_manager, "start_check", start_check)
        )
        stale = patches.enter_context(
            patch.object(dock.config_manager, "is_stale", return_value=False)
        )
        try:
            layout = dock.widget().layout()
            assert [
                layout.itemAt(index).widget().title()
                for index in range(layout.count())
                if isinstance(layout.itemAt(index).widget(), QGroupBox)
            ] == ["开始与当前状态"]
            panel.scripts_directory = str(root)
            dock.workspace_edit.setText(str(root / "workspace"))
            dock.output_path_edit.setText(str(root / "accepted.gpkg"))
            drain(app)
            assert not calls and not dock.start_btn.isEnabled()
            panel.check_button.click()
            assert calls == [(str(root), str(root / "workspace"))]
            data = {
                "status": "ready",
                "checks": [{"id": "semantic_model_fixture", "status": "ready"}],
                "effective": {
                    "schema_version": 2,
                    "semantic_models": [
                        {"model_id": "fixture", "display_name": "Fixture"}
                    ],
                },
            }
            dock.config_manager.last_report = data
            dock.config_manager.report_ready.emit(data)
            assert not dialogs(panel) and not dock.start_btn.isEnabled()
            assert dock.plan_panel.configure_button.isEnabled()
            dock.plan_panel.configuration_dialog.configuration_applied.emit(
                ["fixture"], None, True
            )
            assert not dock.start_btn.isEnabled()
            assert "影像层" in dock.start_readiness_label.text()
            panel.details_button.click()
            details = dialogs(panel)[0]
            assert (
                str(root / "accepted.gpkg")
                in details.findChild(QPlainTextEdit).toPlainText()
            )
            dock.output_path_edit.setText(str(root / "other.gpkg"))
            assert (
                not dock.start_btn.isEnabled() and not panel.details_button.isEnabled()
            )
            assert len(calls) == 1
            panel.check_button.click()
            assert not panel.check_button.isEnabled()
            dock.config_manager.report_ready.emit(data)
            assert not dock.start_btn.isEnabled()
            dock.plan_panel.configuration_dialog.configuration_applied.emit(
                ["fixture"], None, True
            )
            assert not dock.start_btn.isEnabled()
            stale.return_value = True
            dock.plan_panel.configuration_dialog.configuration_applied.emit(
                ["fixture"], None, True
            )
            assert not dock.start_btn.isEnabled()
            stale.return_value = False
            dock.config_manager.last_report = report()
            dock.config_manager.report_ready.emit(dock.config_manager.last_report)
            assert not dock.start_btn.isEnabled()
            assert not dock.plan_panel.configure_button.isEnabled()
            dock.config_manager.last_report = data
            dock.config_manager.report_ready.emit(data)
            assert dock.plan_panel.configure_button.isEnabled()
            assert not dock.start_btn.isEnabled()
            dock.plan_panel.configuration_dialog.configuration_applied.emit(
                ["fixture"], None, True
            )
            assert not dock.start_btn.isEnabled()
            dock.workspace_edit.setText("")
            assert not dock.start_btn.isEnabled() and len(calls) == 2
        finally:
            dock.cleanup()
            dock.deleteLater()
            drain(app)


if __name__ == "__main__":
    app = QgsApplication([], False)
    app.initQgis()
    exceptions = []

    def capture_exception(*error):
        exceptions.append(error)
        traceback.print_exception(*error)

    sys.excepthook = capture_exception
    scenario = sys.argv[2]
    try:
        with tempfile.TemporaryDirectory(prefix="loess-environment-") as temporary:
            globals()[scenario](app, Path(temporary))
        assert not exceptions, exceptions
        print(scenario + ": passed")
    finally:
        drain(app)
        app.exitQgis()
