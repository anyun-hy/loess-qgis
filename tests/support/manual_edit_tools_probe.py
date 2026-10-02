# ruff: noqa: E402
"""Native map-tool transitions on temporary synthetic layers, without a Run."""

from __future__ import annotations

import sys
import tempfile
import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
try:
    from qgis.core import Qgis, QgsApplication, QgsFeature, QgsPointXY, QgsProject
    from qgis.gui import QgsAdvancedDigitizingDockWidget, QgsMapCanvas, QgsMapToolPan
    from qgis.PyQt.QtCore import QCoreApplication, QEvent, Qt
    from qgis.PyQt.QtWidgets import QPushButton
except ModuleNotFoundError:
    raise SystemExit(77)

from refinement_fixtures import SPEC, layer, rectangle

from labeling_tool.refinement import class_workspace
from labeling_tool.refinement.manual_edit_tools import ManualEditTools


def drain(app):
    for _ in range(3):
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def candidate(source, x=20):
    result = QgsFeature(source.fields())
    result.setGeometry(rectangle(x))
    return result


@contextmanager
def native_tools(app, root, *, previous=True):
    source = layer(root, "tools", positions=(0, 10))
    canvas = QgsMapCanvas()
    canvas.setDestinationCrs(source.crs())
    canvas.setLayers([source])
    cad = QgsAdvancedDigitizingDockWidget(canvas)
    pan = QgsMapToolPan(canvas) if previous else None
    if pan is not None:
        canvas.setMapTool(pan)
    owner = ManualEditTools(canvas, lambda: cad)
    assert source.startEditing()
    try:
        yield source, canvas, cad, pan, owner
    finally:
        owner.cleanup()
        if source.isEditable():
            source.rollBack()
        drain(app)
        canvas.close()
        cad.close()


def tools_lifecycle(app, root):
    with native_tools(app, root) as (source, canvas, cad, pan, owner):
        clicked, captured, cancelled, interrupted = [], [], [], []
        owner.map_clicked.connect(lambda *args: clicked.append(args))
        owner.feature_captured.connect(captured.append)
        owner.capture_cancelled.connect(lambda: cancelled.append(True))
        owner.interrupted.connect(lambda: interrupted.append(True))
        owner.begin_session()
        owner.start_picker()
        picker = canvas.mapTool()
        picker.canvasClicked.emit(QgsPointXY(1, 1), Qt.MouseButton.LeftButton)
        assert len(clicked) == 1 and not interrupted
        owner.stop_picker()
        owner.start_capture(source)
        capture = canvas.mapTool()
        assert capture.currentCaptureTechnique() == Qgis.CaptureTechnique.PolyBezier
        capture.digitizingCompleted.emit(candidate(source))
        assert len(captured) == 1 and source.featureCount() == 2
        restarted = []

        def restart():
            restarted.append(True)
            owner.start_capture(source)

        owner.restart_requested.connect(restart)
        owner.schedule_transition("restart")
        assert canvas.mapTool() is capture and not restarted
        drain(app)
        assert restarted == [True] and canvas.mapTool() is not capture
        # Disconnected tools cannot send new events into the current task.
        capture.digitizingCompleted.emit(candidate(source, 30))
        picker.canvasClicked.emit(QgsPointXY(2, 2), Qt.MouseButton.LeftButton)
        assert len(captured) == len(clicked) == 1
        current = canvas.mapTool()
        current.digitizingCanceled.emit()
        assert cancelled == [True]
        owner.schedule_transition("restore")
        assert canvas.mapTool() is current
        drain(app)
        assert canvas.mapTool() is pan and not interrupted
        owner.start_capture(source)
        owner.schedule_transition("restart")
        outside = QgsMapToolPan(canvas)
        canvas.setMapTool(outside)
        assert interrupted == [True]
        drain(app)
        assert canvas.mapTool() is outside and restarted == [True]
        owner.start_picker()
        assert interrupted == [True]
        owner.end_session()
        assert canvas.mapTool() is pan
        owner.cleanup()
        owner.cleanup()
        owner.begin_session()
        owner.start_picker()
        owner.end_session()
        assert canvas.mapTool() is pan


def stale_restart(app, root):
    with native_tools(app, root, previous=False) as (source, canvas, cad, pan, owner):
        restarted = []
        owner.restart_requested.connect(lambda: restarted.append(True))
        owner.begin_session()
        owner.start_capture(source)
        old = canvas.mapTool()
        owner.schedule_transition("restart")
        owner.end_session()
        assert canvas.mapTool() is None
        owner.begin_session()
        owner.start_picker()
        current = canvas.mapTool()
        drain(app)
        assert not restarted and canvas.mapTool() is current
        old.digitizingCanceled.emit()
        assert canvas.mapTool() is current
        owner.stop_picker()
        owner.start_capture(source)
        owner.schedule_transition("restart")
        owner.cleanup()
        drain(app)
        assert not restarted and canvas.mapTool() is None
        # A capability failure must leave no active capture tool or pending
        # restart and must still permit ending/restarting the task.
        from labeling_tool.refinement import manual_edit_tools as module

        make_tool = module.QgsMapToolDigitizeFeature

        def unsupported(*args):
            tool = make_tool(*args)
            tool.supportsTechnique = lambda _: False
            return tool

        owner.begin_session()
        with patch.object(module, "QgsMapToolDigitizeFeature", side_effect=unsupported):
            try:
                owner.start_capture(source)
            except RuntimeError as exc:
                assert "PolyBezier" in str(exc)
            else:
                raise AssertionError("unsupported capture accepted")
        owner.end_session()
        drain(app)
        assert canvas.mapTool() is None


def dialog_tasks(app, root):
    from labeling_tool.qgis_support.qt6_api import YES
    from labeling_tool.refinement import class_refinement_dialog as ui

    source = layer(root, "dialog", positions=(0, 10))
    QgsProject.instance().addMapLayer(source)
    canvas = QgsMapCanvas()
    canvas.setDestinationCrs(source.crs())
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
    dialog._run_spec = dict(SPEC, run_dir=str(root))
    dialog._workspace = {"baseline_stream_id": "fusion:probe", "classes": {}}
    dialog._class_layers = {12: source.id()}
    history = []

    def request(name):
        button = dialog.findChild(QPushButton, "manual" + name)
        assert button is not None and button.isEnabled(), name
        button.click()

    try:
        with ExitStack() as stack:
            for name in (
                "_update_actions",
                "_refresh_table",
                "_refresh_class_display",
                "_set_class_modified",
                "_select_class_context",
                "_set_visible",
            ):
                stack.enter_context(patch.object(dialog, name))
            stack.enter_context(
                patch.object(
                    dialog,
                    "_optional_confidence_statistics",
                    return_value=(0.6, 0.1, ""),
                )
            )
            stack.enter_context(
                patch.object(dialog, "_local_topology_hint", return_value="ok")
            )
            stack.enter_context(
                patch.object(
                    class_workspace,
                    "save_workspace",
                    side_effect=lambda _, workspace, **kw: workspace,
                )
            )
            stack.enter_context(
                patch.object(
                    class_workspace,
                    "append_history",
                    side_effect=lambda _, event, **kw: history.append(event),
                )
            )
            stack.enter_context(
                patch.object(ui.QMessageBox, "question", return_value=YES)
            )
            warning = stack.enter_context(patch.object(ui.QMessageBox, "warning"))
            dialog._update_manual_panel()
            request("AddTask")
            assert dialog._manual_task.state == "capturing" and source.isEditable()
            tool = canvas.mapTool()
            tool.digitizingCompleted.emit(candidate(source))
            assert len(dialog._manual_task.pending_geometries) == 1
            assert canvas.mapTool() is tool
            outside = QgsMapToolPan(canvas)
            canvas.setMapTool(outside)
            assert dialog._manual_task.state == "paused"
            drain(app)
            assert canvas.mapTool() is outside
            request("Continue")
            assert dialog._manual_task.state == "capturing"
            canvas.mapTool().digitizingCanceled.emit()
            assert len(dialog._manual_task.pending_geometries) == 1
            drain(app)
            assert canvas.mapTool() is pan
            request("Primary")
            assert not warning.called, warning.call_args_list
            assert source.featureCount() == 3 and history == ["feature_added"]
            canvas.mapTool().digitizingCompleted.emit(candidate(source, 30))
            dialog._cancel_manual_task(silent=True)
            drain(app)
            assert dialog._manual_task is None and canvas.mapTool() is pan
            assert source.featureCount() == 3 and not source.isEditable()

            original = next(source.getFeatures())
            source.selectByIds([original.id()])
            request("ModifyTask")
            assert dialog._manual_task.selected_feature_ids == [original.id()]
            with patch.object(
                dialog, "_features_at_map_point", return_value=[original]
            ):
                canvas.mapTool().canvasClicked.emit(
                    QgsPointXY(1, 1), Qt.MouseButton.LeftButton
                )
                assert dialog._manual_task.selected_feature_ids == []
                canvas.mapTool().canvasClicked.emit(
                    QgsPointXY(1, 1), Qt.MouseButton.LeftButton
                )
            request("Retry")
            canvas.mapTool().digitizingCompleted.emit(candidate(source, 40))
            request("Finish")
            drain(app)
            assert dialog._manual_task is None and canvas.mapTool() is pan
            assert source.featureCount() == 3 and not source.isEditable()
            source.selectByIds([original.id()])
            request("DeleteTask")
            source.removeSelection()
            with patch.object(
                dialog, "_features_at_map_point", return_value=[original]
            ):
                canvas.mapTool().canvasClicked.emit(
                    QgsPointXY(1, 1), Qt.MouseButton.LeftButton
                )
            assert dialog._manual_task.selected_count == 1
            request("Cancel")
            assert source.featureCount() == 3 and dialog._manual_task is None
            assert source.selectedFeatureIds() == [original.id()]
            dialog._workspace = None
            dialog.cleanup()
            dialog.cleanup()
            drain(app)
            assert canvas.mapTool() is pan
    finally:
        source.blockSignals(True)
        if source.isEditable():
            source.rollBack()
        dialog._workspace = None
        dialog._run_spec = None
        dialog._manual_task = None
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
    with tempfile.TemporaryDirectory(prefix="loess-manual-tools-") as directory:
        scenario = sys.argv[2]
        globals()[scenario](app, Path(directory))
        print(scenario + ": passed", flush=True)
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
finally:
    app.exitQgis()
