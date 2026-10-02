# ruff: noqa: E402
"""Exercise the refinement dialog's display-only admission summary wiring."""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))

try:
    from qgis.core import QgsApplication
    from qgis.gui import QgsMapCanvas
    from qgis.PyQt.QtCore import QCoreApplication, QEvent, Qt
    from qgis.PyQt.QtTest import QTest
    from qgis.PyQt.QtWidgets import QApplication, QLabel, QScrollArea
except ModuleNotFoundError:
    raise SystemExit(77)

from labeling_tool.refinement import class_refinement_dialog as ui
from labeling_tool.refinement.admission_summary_panel import AdmissionSummaryPanel
from labeling_tool.shared.contracts.run_spec import CLASS_ORDER


def _label(panel, name: str) -> QLabel:
    value = panel.findChild(QLabel, name)
    assert value is not None, name
    return value


def _workspace(root: Path) -> dict:
    return {
        "classes": {
            str(code): {"confirmed": True, "path": str(root / f"class-{code}.gpkg")}
            for code in CLASS_ORDER
        }
    }


def run(app: QgsApplication, screenshot_directory: Path | None = None) -> None:
    canvas = QgsMapCanvas()
    unloaded = ui.ClassRefinementDialog(
        SimpleNamespace(
            mapCanvas=lambda: canvas,
            activeLayer=lambda: None,
            cadDockWidget=lambda: None,
        ),
        SimpleNamespace(),
    )
    unloaded._run_spec = None
    unloaded._refresh_table()
    unloaded.cleanup()
    unloaded.close()
    unloaded.deleteLater()
    dialog = ui.ClassRefinementDialog(
        SimpleNamespace(
            mapCanvas=lambda: canvas,
            activeLayer=lambda: None,
            cadDockWidget=lambda: None,
        ),
        SimpleNamespace(),
    )
    root = Path(tempfile.mkdtemp(prefix="admission-summary-"))
    long_target_path = str(
        root
        / "黄土高原长期标签库"
        / "人工审核完成后的超长中文目录用于验证完整路径复制"
        / "accepted_labels.gpkg"
    )
    panel = dialog.findChild(AdmissionSummaryPanel, "AdmissionSummaryPanel")
    assert panel is not None
    try:
        dialog._run_spec = {
            "run_id": "admission-probe",
            "run_dir": str(root),
            "accepted_target_gpkg": long_target_path,
            "accepted_write_manifest": str(root / "run_manifest.json"),
        }
        dialog._workspace = _workspace(root)
        dialog._final_path = str(root / "final.gpkg")
        dialog._final_feature_count = 12
        dialog._issue_count = 0
        dialog._final_matches_workspace = lambda: True
        dialog._update_actions()
        assert panel._target_path.text() == long_target_path
        panel._target_path.setFocus()
        QTest.keyClick(
            panel._target_path,
            Qt.Key.Key_A,
            Qt.KeyboardModifier.ControlModifier,
        )
        QTest.keyClick(
            panel._target_path,
            Qt.Key.Key_C,
            Qt.KeyboardModifier.ControlModifier,
        )
        assert QApplication.clipboard().text() == long_target_path
        assert _label(panel, "AdmissionFinalFeatures").text().startswith("12 个面")
        assert _label(panel, "AdmissionClassConfirmation").text() == "14/14"
        assert panel._write_button.isEnabled()

        with patch.object(ui.QgsApplication, "taskManager") as manager:
            panel._write_button.click()
            task = manager.return_value.addTask.call_args.args[0]
        assert dialog._accepted_task is task
        assert dialog._accepted_feature_count is None
        assert not panel._write_button.isEnabled()
        assert "等待后台写入" in _label(panel, "AdmissionBackgroundStage").text()

        task.commit_started = True
        task.progress_message = "正在提交，不能取消"
        task.progressChanged.emit(82)
        assert _label(panel, "AdmissionBackgroundStage").text() == "正在提交，不能取消"
        task.published = True
        task.result_data = {
            "run_id": "admission-probe",
            "feature_count": 3,
            "warnings": ["accepted_labels 已写入，但目录同步未确认"],
        }
        task.taskCompleted.emit()
        assert _label(panel, "AdmissionAcceptedResult").text() == "3 个面。"
        assert "已写入" in _label(panel, "AdmissionAcceptedWarnings").text()
        assert not panel._write_button.isEnabled()

        dialog._invalidate_final()
        assert _label(panel, "AdmissionFinalFeatures").text() == "未提供。"
        assert "当前窗口未记录" in _label(panel, "AdmissionAcceptedResult").text()

        dialog._final_path = str(root / "refreshed-final.gpkg")
        dialog._final_input_identities = {"synthetic": None}
        dialog._final_feature_count = 12
        dialog._issue_count = 2
        dialog._update_actions()
        assert not panel._write_button.isEnabled()
        assert "明确勾选带问题入库" in _label(panel, "AdmissionBlocker").text()
        assert "完整性检查仍会执行" in _label(panel, "AdmissionIntegrityNote").text()
        dialog.allow_issues_check.setChecked(True)
        assert panel._write_button.isEnabled()

        with (
            patch.object(ui.QgsApplication, "taskManager") as manager,
            patch.object(ui.QMessageBox, "warning"),
        ):
            panel._write_button.click()
            failed_task = manager.return_value.addTask.call_args.args[0]
            failed_task.error_message = "injected failure"
            failed_task.taskTerminated.emit()
        assert "当前窗口未记录" in _label(panel, "AdmissionAcceptedResult").text()

        dialog._final_feature_count = 8
        dialog._accepted_feature_count = 4
        with (
            patch.object(dialog._workspace_tasks, "probe"),
            patch.object(
                ui.class_workspace,
                "save_workspace",
                side_effect=lambda _spec, value, **_kwargs: value,
            ),
        ):
            dialog.set_run(
                {},
                {
                    "run_id": "replacement",
                    "run_dir": str(root),
                    "accepted_target_gpkg": str(root / "replacement.gpkg"),
                },
                {},
                "",
            )
        assert dialog._final_feature_count is None
        assert dialog._accepted_feature_count is None
        assert "当前窗口未记录" in _label(panel, "AdmissionAcceptedResult").text()

        scroll = dialog.findChild(QScrollArea, "classRefinementScrollArea")
        assert scroll is not None
        for width, height, filename in (
            (960, 540, "admission-dialog-960x540.png"),
            (640, 360, "admission-dialog-640x360.png"),
        ):
            dialog.resize(width, height)
            dialog.show()
            app.processEvents()
            scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
            app.processEvents()
            blocker = _label(panel, "AdmissionBlocker")
            required_height = blocker.heightForWidth(blocker.width())
            assert required_height <= blocker.height()
            center = panel._write_button.mapTo(
                scroll.viewport(), panel._write_button.rect().center()
            )
            assert scroll.viewport().rect().contains(center)
            if screenshot_directory is not None:
                screenshot_directory.mkdir(parents=True, exist_ok=True)
                image = screenshot_directory / filename
                assert dialog.grab().save(str(image)) and image.stat().st_size > 1000
    finally:
        dialog._workspace = None
        dialog.cleanup()
        dialog.close()
        dialog.deleteLater()
        canvas.close()


app = QgsApplication([], False)
app.initQgis()
try:
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    run(app, output)
    print("refinement admission summary: passed", flush=True)
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
finally:
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.exitQgis()
