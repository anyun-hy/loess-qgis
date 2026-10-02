# ruff: noqa: E402
"""Native controls and task integration; only temporary synthetic layer data."""

from __future__ import annotations

import sys
import tempfile
import traceback
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
try:
    from qgis.core import QgsApplication, QgsFeature, QgsProject
    from qgis.gui import QgsAdvancedDigitizingDockWidget, QgsMapCanvas, QgsMapToolPan
    from qgis.PyQt.QtCore import QCoreApplication, QEvent
    from qgis.PyQt.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QLabel,
        QPushButton,
        QSpinBox,
    )
except ModuleNotFoundError:
    raise SystemExit(77)

from refinement_fixtures import SPEC, layer, rectangle

from labeling_tool.refinement.manual_edit_panel import (
    ManualEditPanel,
    ManualPanelSnapshot,
)


def child(panel, kind, name):
    widget = panel.findChild(kind, "manual" + name)
    assert widget is not None, name
    return widget


def button(panel, name):
    return child(panel, QPushButton, name)


def drain(app):
    for _ in range(3):
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def snapshot(**values):
    return replace(
        ManualPanelSnapshot(
            current_class_text="12 Test",
            active_layer_name="test layer",
            selected_count=0,
            edit_text="已保存",
            idle_enabled=False,
            has_features=True,
            has_workspace=True,
        ),
        **values,
    )


def controls(app, root):
    panel = ManualEditPanel((1, 0.25, 180.0))
    primary, retry = button(panel, "Primary"), button(panel, "Retry")
    target = child(panel, QComboBox, "TargetClass")
    smooth = child(panel, QCheckBox, "SmoothEnabled")
    try:
        panel.render(snapshot(has_workspace=False, has_features=False))
        assert all(
            not button(panel, name).isEnabled()
            for name in ("ModifyTask", "DeleteTask", "AddTask")
        )
        panel.render(snapshot(idle_enabled=True, has_features=False))
        assert button(panel, "AddTask").isEnabled()
        assert not button(panel, "ModifyTask").isEnabled()
        assert primary.isHidden() and target.isHidden()
        modify = snapshot(kind="modify", state="selecting", modify_selected_count=1)
        panel.render(modify)
        assert not primary.isHidden() and not primary.isEnabled()
        assert retry.text() == "绘制新边界" and retry.isEnabled()
        assert not target.isHidden() and smooth.isHidden()
        panel.render(replace(modify, target_changed=True))
        assert primary.isEnabled(), "Changing class must not require new geometry"
        pending = replace(modify, pending_count=1)
        for values, expected in (
            ({}, True),
            ({"pending_has_error": True}, False),
            ({"smoothing_enabled": True, "smoothing_ready": False}, False),
            ({"smoothing_enabled": True, "smoothing_ready": True}, True),
            ({"state": "committing"}, False),
        ):
            panel.render(replace(pending, **values))
            assert primary.isEnabled() is expected, values
        for kind in ("modify", "add", "delete"):
            panel.render(replace(pending, kind=kind, state="paused"))
            assert primary.isHidden() and not button(panel, "Continue").isHidden()
            assert not target.isEnabled() and not smooth.isEnabled()
            if kind == "delete":
                assert not button(panel, "Cancel").isHidden()
            else:
                assert not button(panel, "Finish").isHidden()
        for count in (0, 2):
            panel.render(
                snapshot(kind="delete", state="selecting", delete_selected_count=count)
            )
            assert primary.text() == f"删除选中的 {count} 个面"
            assert primary.isEnabled() is bool(count)
            assert button(panel, "Clear").isEnabled() is bool(count)
            assert target.isHidden() and smooth.isHidden()
        # Rendering a different task must not inherit a prior task's buttons.
        panel.render(modify)
        panel.render(snapshot(kind="add", state="capture_cancelled"))
        assert retry.isEnabled() and retry.text() == "重新绘制当前面"
        assert primary.isHidden() and button(panel, "Cancel").isHidden()
        assert button(panel, "Finish").text() == "结束新增"
        panel.render(snapshot(kind="add", state="committing", pending_count=1))
        assert not primary.isEnabled() and not retry.isEnabled()
        panel.render(snapshot(kind="add", state="failed"))
        assert retry.isEnabled(), "A previous committing state must not disable retry"
        panel.render(snapshot(idle_enabled=True))
        assert all(
            button(panel, name).isHidden()
            for name in ("Primary", "Retry", "Clear", "Continue", "Cancel", "Finish")
        )
    finally:
        panel.close()


def signals(app, root):
    panel = ManualEditPanel((1, 0.25, 180.0))
    counts = []
    try:
        panel.target_changed.connect(lambda code: counts.append(("target", code)))
        panel.smoothing_enabled_changed.connect(
            lambda value: counts.append(("smooth", value))
        )
        panel.smoothing_parameters_changed.connect(
            lambda: counts.append(("parameters",))
        )
        panel.set_target_code(21)
        panel.set_smoothing_parameters((2, 0.3, 150.0))
        panel.set_smoothing_enabled(True)
        assert not counts, "Programmatic synchronization emitted user requests"
        assert panel.target_code() == 21
        assert panel.smoothing_parameters() == (2, 0.3, 150.0)
        combo = child(panel, QComboBox, "TargetClass")
        combo.setCurrentIndex(combo.findData(12))
        child(panel, QCheckBox, "SmoothEnabled").click()
        child(panel, QSpinBox, "SmoothIterations").setValue(3)
        child(panel, QDoubleSpinBox, "SmoothOffset").setValue(0.35)
        child(panel, QDoubleSpinBox, "SmoothAngle").setValue(160.0)
        assert counts == [("target", 12), ("smooth", False)] + [("parameters",)] * 3
        counts.clear()
        for name, signal in (
            ("ModifyTask", panel.modify_requested),
            ("DeleteTask", panel.delete_requested),
            ("AddTask", panel.add_requested),
            ("Primary", panel.primary_requested),
            ("Retry", panel.retry_requested),
            ("Clear", panel.clear_requested),
            ("Continue", panel.continue_requested),
            ("Cancel", panel.cancel_requested),
            ("Finish", panel.finish_requested),
        ):
            signal.connect(lambda n=name: counts.append(n))
            button(panel, name).click()
        assert counts == [
            "ModifyTask",
            "DeleteTask",
            "AddTask",
            "Primary",
            "Retry",
            "Clear",
            "Continue",
            "Cancel",
            "Finish",
        ]
        panel.set_instruction("paused")
        panel.set_smoothing_status("preview ready")
        assert child(panel, QLabel, "Instruction").text() == "paused"
        assert child(panel, QLabel, "SmoothStatus").text() == "preview ready"
    finally:
        panel.close()


def dialog_smoothing(app, root):
    from labeling_tool.refinement import class_refinement_dialog as ui

    source = layer(root, "panel", positions=(0, 10))
    QgsProject.instance().addMapLayer(source)
    canvas = QgsMapCanvas()
    canvas.setLayers([source])
    cad = QgsAdvancedDigitizingDockWidget(canvas)
    pan = QgsMapToolPan(canvas)
    canvas.setMapTool(pan)
    dialog = ui.ClassRefinementDialog(
        SimpleNamespace(
            mapCanvas=lambda: canvas,
            activeLayer=lambda: source,
            setActiveLayer=lambda _: None,
            cadDockWidget=lambda: cad,
        ),
        SimpleNamespace(),
    )
    panel = dialog.findChild(ManualEditPanel, "manualEditPanel")
    assert panel is not None
    dialog._run_spec = dict(SPEC, run_dir=str(root))
    dialog._workspace = {"baseline_stream_id": "fusion:probe", "classes": {}}
    dialog._class_layers = {12: source.id()}
    primary = button(panel, "Primary")
    try:
        with ExitStack() as stack:
            for name in (
                "_update_actions",
                "_refresh_table",
                "_select_class_context",
                "_set_visible",
            ):
                stack.enter_context(patch.object(dialog, name))
            store = stack.enter_context(
                patch.object(dialog, "_store_smoothing_parameters")
            )
            panel.set_smoothing_parameters((1, 0.25, 180.0))
            dialog._sync_smoothing_parameter_widgets((1, 0.25, 180.0), "manual")
            dialog._update_manual_panel()
            button(panel, "AddTask").click()
            task = dialog._manual_task
            assert task is not None and task.kind == "add" and task.state == "capturing"
            captured = QgsFeature(source.fields())
            captured.setGeometry(rectangle(20))
            canvas.mapTool().digitizingCompleted.emit(captured)
            drain(app)
            assert primary.isEnabled() and source.featureCount() == 2
            toggle = child(panel, QCheckBox, "SmoothEnabled")
            toggle.click()
            assert task.smoothing_enabled and not primary.isEnabled()
            assert dialog._manual_smoothing_timer.isActive()
            dialog._manual_smoothing_timer.stop()
            dialog._refresh_manual_smoothing_preview()
            assert dialog._manual_smoothing_preview_is_current()
            assert primary.isEnabled()
            assert source.featureCount() == 2 and not source.isModified()
            child(panel, QSpinBox, "SmoothIterations").setValue(2)
            assert dialog.qgis_smooth_iterations_spin.value() == 2
            assert store.call_count == 1 and not primary.isEnabled()
            dialog._manual_smoothing_timer.stop()
            dialog._refresh_manual_smoothing_preview()
            assert primary.isEnabled()
            # Advanced control updates synchronize without recursive manual writes.
            dialog.qgis_smooth_offset_spin.setValue(0.3)
            assert panel.smoothing_parameters() == (2, 0.3, 180.0)
            assert store.call_count == 2
            assert not dialog._manual_smoothing_preview_is_current()
            dialog._schedule_manual_smoothing_preview()
            dialog._manual_smoothing_timer.stop()
            dialog._refresh_manual_smoothing_preview()
            assert primary.isEnabled()
            button(panel, "Retry").click()
            assert task.pending_geometries == [] and task.state == "capturing"
            assert task.smoothing_preview is None
            button(panel, "Finish").click()
            drain(app)
            assert dialog._manual_task is None and canvas.mapTool() is pan
            assert source.featureCount() == 2 and not source.isEditable()
            button(panel, "AddTask").click()
            assert not dialog._manual_task.smoothing_enabled and not toggle.isChecked()
            dialog._cancel_manual_task(silent=True)
    finally:
        dialog._manual_smoothing_timer.stop()
        dialog._workspace = None
        dialog._cancel_manual_task(silent=True)
        source.blockSignals(True)
        if source.isEditable():
            source.rollBack()
        dialog.cleanup()
        dialog.close()
        dialog.deleteLater()
        drain(app)
        QgsProject.instance().removeAllMapLayers()
        canvas.close()
        cad.close()


app = QgsApplication([], False)
app.initQgis()
try:
    with tempfile.TemporaryDirectory(prefix="loess-manual-panel-") as directory:
        globals()[sys.argv[2]](app, Path(directory))
        print(sys.argv[2] + ": passed", flush=True)
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
finally:
    app.exitQgis()
