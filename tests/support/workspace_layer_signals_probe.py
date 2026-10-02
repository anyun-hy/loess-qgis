# ruff: noqa: E402
"""Signal and lifetime contracts on disposable native QGIS layers."""

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
    from qgis.core import QgsApplication, QgsFeature, QgsLayerTreeLayer, QgsProject
    from qgis.gui import QgsMapCanvas
    from qgis.PyQt import sip
    from qgis.PyQt.QtCore import QCoreApplication, QEvent, QObject, pyqtSignal
except ModuleNotFoundError:
    raise SystemExit(77)

from refinement_fixtures import SPEC, layer, rectangle

from labeling_tool.refinement import class_refinement_dialog as ui
from labeling_tool.refinement.workspace_layer_signals import WorkspaceLayerSignals


class Iface(QObject):
    currentLayerChanged = pyqtSignal(object)

    def __init__(self, canvas=None, active=None):
        super().__init__()
        self.canvas, self.active = canvas, active

    def mapCanvas(self):
        return self.canvas

    def activeLayer(self):
        return self.active

    def setActiveLayer(self, value):
        self.active = value
        self.currentLayerChanged.emit(value)


def drain(app):
    for _ in range(3):
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def layer_events(app, root):
    source = layer(root, "events", positions=(0, 10))
    node = QgsLayerTreeLayer(source)
    owner = WorkspaceLayerSignals()
    started, finished, selected, changed, visible, events = [], [], [], [], [], []
    owner.editing_started.connect(started.append)
    owner.editing_stopped.connect(finished.append)
    owner.selection_changed.connect(selected.append)
    owner.edit_changed.connect(changed.append)
    owner.visibility_changed.connect(visible.append)
    owner.before_commit.connect(
        lambda code, value: events.append(("before", code, value))
    )
    owner.features_committed.connect(
        lambda code, values: events.append(("added", code, values))
    )
    owner.editing_stopped.connect(lambda code: events.append(("finished", code)))
    try:
        owner.bind_layer(12, source, node)
        owner.bind_layer(12, source, node)
        assert source.startEditing() and started == [12]
        fid = next(source.getFeatures()).id()
        source.selectByIds([fid])
        assert selected == [12]
        node.setItemVisibilityChecked(False)
        assert visible == [12]
        before = source.getFeature(fid).geometry()
        source.beginEditCommand("change shape")
        assert source.changeGeometry(fid, rectangle(0, 2))
        source.endEditCommand()
        assert changed and set(changed) == {12}
        count = len(changed)
        source.undoStack().undo()
        assert len(changed) > count and source.getFeature(fid).geometry().equals(before)
        count = len(changed)
        assert source.changeAttributeValue(
            fid, source.fields().indexFromName("reviewed"), 0
        )
        assert len(changed) > count
        added = QgsFeature(source.fields())
        added.setGeometry(rectangle(20))
        added["object_id"] = "new-signal-feature"
        count = len(changed)
        assert source.addFeature(added) and len(changed) > count
        count = len(changed)
        assert source.deleteFeature(fid) and len(changed) > count
        assert source.commitChanges(), source.commitErrors()
        assert finished == [12, 12], (
            "afterCommitChanges and editingStopped remain separate"
        )
        assert [event[0] for event in events] == [
            "before",
            "added",
            "finished",
            "finished",
        ]
        assert events[0][1:] == (12, source)
        assert events[1][1] == 12
        assert [f["object_id"] for f in events[1][2]] == ["new-signal-feature"]
        assert events[1][2][0].id() >= 0
        owner.clear_layers()
        sizes = [
            len(v) for v in (started, finished, selected, changed, visible, events)
        ]
        assert source.startEditing()
        source.removeSelection()
        source.selectAll()
        node.setItemVisibilityChecked(True)
        assert source.rollBack()
        assert sizes == [
            len(v) for v in (started, finished, selected, changed, visible, events)
        ]
    finally:
        owner.cleanup()
        if source.isEditable():
            source.rollBack()


def tree_rebinding(app, root):
    source = layer(root, "tree", positions=(0,))
    owner = WorkspaceLayerSignals()
    visible, started = [], []
    owner.visibility_changed.connect(visible.append)
    owner.editing_started.connect(started.append)
    old, current = QgsLayerTreeLayer(source), QgsLayerTreeLayer(source)
    try:
        owner.bind_layer(12, source)
        owner.bind_layer(12, source, old)
        owner.bind_layer(12, source, old)
        old.setItemVisibilityChecked(False)
        assert visible == [12]
        owner.bind_layer(12, source, current)
        old.setItemVisibilityChecked(True)
        assert visible == [12], "Replaced tree nodes must no longer notify the dialog"
        current.setItemVisibilityChecked(False)
        assert visible == [12, 12]
        assert source.startEditing() and started == [12]
        owner.clear_layers()
        owner.clear_layers()
        current.setItemVisibilityChecked(True)
        assert visible == [12, 12]
        assert source.rollBack()
        owner.bind_layer(12, source, current)
        assert source.startEditing() and started == [12, 12]
    finally:
        owner.cleanup()
        if source.isEditable():
            source.rollBack()


def detached_layer(app, root):
    project = QgsProject.instance()
    source = layer(root, "detached", positions=(0,))
    project.addMapLayer(source)
    owner = WorkspaceLayerSignals()
    received = []
    owner.editing_started.connect(received.append)
    owner.selection_changed.connect(received.append)
    owner.edit_changed.connect(received.append)
    try:
        owner.bind_layer(12, source, project.layerTreeRoot().findLayer(source.id()))
        retained = project.takeMapLayer(source)
        assert retained is source and project.mapLayer(source.id()) is None
        owner.cleanup()
        owner.cleanup()
        assert source.startEditing()
        source.selectAll()
        assert source.changeGeometry(next(source.getFeatures()).id(), rectangle(30))
        assert received == [], "Detached but live layers must be disconnected too"
        assert source.rollBack()
        owner.bind_layer(12, source)
        assert source.startEditing() and received == [12]
    finally:
        owner.cleanup()
        if source.isEditable():
            source.rollBack()


def destroyed_sources(app, root):
    project = QgsProject.instance()
    source = layer(root, "destroyed", positions=(0,))
    project.addMapLayer(source)
    owner = WorkspaceLayerSignals()
    node = project.layerTreeRoot().findLayer(source.id())
    owner.bind_layer(12, source, node)
    project.removeMapLayer(source.id())
    drain(app)
    assert sip.isdeleted(source) and sip.isdeleted(node)
    owner.clear_layers()
    owner.cleanup()
    live = layer(root, "owner-deleted", positions=(0,))
    received = []
    owner.editing_started.connect(received.append)
    owner.bind_layer(12, live)
    owner.cleanup()
    owner.deleteLater()
    drain(app)
    assert live.startEditing() and received == []
    assert live.rollBack()


def iface_lifecycle(app, root):
    first, second = Iface(), Iface()
    owner = WorkspaceLayerSignals()
    received = []
    order = []
    owner.current_layer_changed.connect(received.append)
    owner.current_layer_changed.connect(lambda _: order.append("owner"))
    source = layer(root, "active", positions=(0,))
    try:
        owner.connect_current_layer(None)
        owner.connect_current_layer(first.currentLayerChanged)
        first.currentLayerChanged.connect(lambda _: order.append("outside"))
        owner.connect_current_layer(first.currentLayerChanged)
        first.setActiveLayer(source)
        assert received == [source]
        assert order == ["owner", "outside"], (
            "Repeated setup must retain callback order"
        )
        owner.clear_layers()
        first.setActiveLayer(None)
        assert received == [source, None], (
            "Switching Run retains active-layer observation"
        )
        owner.connect_current_layer(second.currentLayerChanged)
        first.setActiveLayer(source)
        second.setActiveLayer(source)
        assert received == [source, None, source]
        owner.cleanup()
        owner.cleanup()
        second.setActiveLayer(None)
        assert received == [source, None, source]
        owner.connect_current_layer(first.currentLayerChanged)
        first.setActiveLayer(None)
        assert received == [source, None, source, None]
        first.deleteLater()
        drain(app)
        owner.cleanup()
    finally:
        owner.cleanup()


def failed_registration(app, root):
    source = layer(root, "register-retry", positions=(0,))
    owner = WorkspaceLayerSignals()
    received = []
    owner.editing_started.connect(received.append)
    try:
        # Failure late in registration must undo the connections already made.
        with patch.object(source, "undoStack", side_effect=RuntimeError("unavailable")):
            try:
                owner.bind_layer(12, source)
            except RuntimeError as exc:
                assert str(exc) == "unavailable"
            else:
                raise AssertionError("Registration failure must reach the caller")
        assert source.startEditing() and received == []
        assert source.rollBack()
        owner.bind_layer(12, source)
        assert source.startEditing() and received == [12]
    finally:
        owner.cleanup()
        if source.isEditable():
            source.rollBack()


def dialog_lifecycle(app, root):
    project = QgsProject.instance()
    source = layer(root, "dialog", positions=(0,))
    project.addMapLayer(source)
    canvas = QgsMapCanvas()
    iface = Iface(canvas, source)
    with ExitStack() as stack:
        # Patch before construction so real connections retain these recipients.
        callbacks = {
            name: stack.enter_context(patch.object(ui.ClassRefinementDialog, name))
            for name in (
                "_editing_started",
                "_editing_stopped",
                "_selection_changed",
                "_layer_edit_changed",
                "_sync_visibility_from_layer_tree",
                "_active_layer_changed",
                "_update_manual_panel",
                "_update_actions",
                "_refresh_table",
            )
        }
        stack.enter_context(
            patch.object(
                ui.class_workspace,
                "save_workspace",
                side_effect=lambda _, value, **kw: value,
            )
        )
        dialog = ui.ClassRefinementDialog(iface, SimpleNamespace())
        dialog._run_spec = dict(SPEC, run_dir=str(root))
        dialog._workspace = {"baseline_stream_id": "fusion:probe", "classes": {}}
        dialog._class_layers = {12: source.id()}
        try:
            dialog._register_workspace_layer(12, source.id())
            dialog._register_workspace_layer(12, source.id())
            for callback in callbacks.values():
                callback.reset_mock()
            assert source.startEditing()
            source.selectAll()
            callbacks["_editing_started"].assert_called_once_with(12)
            callbacks["_selection_changed"].assert_called_once_with(12)
            node = project.layerTreeRoot().findLayer(source.id())
            node.setItemVisibilityChecked(False)
            callbacks["_sync_visibility_from_layer_tree"].assert_called_once_with(12)
            iface.setActiveLayer(source)
            callbacks["_active_layer_changed"].assert_called_once_with(source)
            assert source.rollBack()
            with patch.object(ui.QgsApplication, "taskManager"):
                dialog.set_run({}, dict(dialog._run_spec), {}, "")
            for callback in callbacks.values():
                callback.reset_mock()
            assert source.startEditing()
            source.removeSelection()
            node.setItemVisibilityChecked(True)
            callbacks["_editing_started"].assert_not_called()
            callbacks["_selection_changed"].assert_not_called()
            callbacks["_sync_visibility_from_layer_tree"].assert_not_called()
            iface.setActiveLayer(None)
            callbacks["_active_layer_changed"].assert_called_once_with(None)
            assert source.rollBack()
            dialog.cleanup()
            callbacks["_active_layer_changed"].reset_mock()
            iface.setActiveLayer(source)
            callbacks["_active_layer_changed"].assert_not_called()
        finally:
            dialog._workspace = None
            dialog.cleanup()
            dialog.close()
            dialog.deleteLater()
            drain(app)
            project.removeAllMapLayers()
            canvas.close()


app = QgsApplication([], False)
app.initQgis()
try:
    with tempfile.TemporaryDirectory(prefix="loess-layer-signals-") as directory:
        globals()[sys.argv[2]](app, Path(directory))
        print(sys.argv[2] + ": passed", flush=True)
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
finally:
    app.exitQgis()
