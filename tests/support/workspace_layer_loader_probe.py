# ruff: noqa: E402
"""Exercise real Qt timers, nested event loops, and disposable QGIS layers."""

from __future__ import annotations

import sys
import tempfile
import time
import traceback
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
try:
    from qgis.core import QgsApplication, QgsProject
    from qgis.gui import QgsMapCanvas
    from qgis.PyQt.QtCore import QCoreApplication, QEvent, QEventLoop, QTimer
    from qgis.PyQt.QtTest import QTest
except ModuleNotFoundError:
    raise SystemExit(77)

from refinement_fixtures import SPEC, layer

from labeling_tool.refinement import class_refinement_dialog as ui
from labeling_tool.refinement.workspace_layer_loader import WorkspaceLayerLoader
from labeling_tool.shared.contracts.run_spec import CLASS_ORDER


def wait_for(predicate):
    deadline = time.monotonic() + 4
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(10)
    assert predicate(), "Expected Qt event was not delivered"


def observe(owner):
    events = SimpleNamespace(loaded=[], failed=[], completed=[])
    owner.loaded.connect(lambda code, active: events.loaded.append((code, active)))
    owner.failed.connect(lambda code, message: events.failed.append((code, message)))
    owner.completed.connect(lambda: events.completed.append(True))
    return events


def sequential(app, root):
    calls, times = [], []

    def load(code):
        calls.append(code)
        times.append(time.monotonic())

    owner = WorkspaceLayerLoader(load)
    events = observe(owner)
    try:
        owner.start([12, 13, 12, 31])
        assert calls == [] and owner.pending_codes == (12, 13, 31)
        wait_for(lambda: events.completed)
        assert calls == [12, 13, 31]
        assert events.loaded == [(12, False), (13, False), (31, False)]
        assert all(b - a >= 0.15 for a, b in zip(times, times[1:]))
        QTest.qWait(350)
        assert events.completed == [True] and events.failed == []
    finally:
        owner.reset()


def priority_pause(app, root):
    calls, times = [], []

    def load(code):
        calls.append(code)
        times.append(time.monotonic())

    owner = WorkspaceLayerLoader(load)
    events = observe(owner)
    owner.loaded.connect(lambda code, _: owner.pause() if code == 12 else None)
    try:
        owner.start([12, 13, 31])
        owner.prioritize(31, activate=True)
        owner.prioritize(31, activate=True)
        assert owner.pending_codes == (31, 12, 13)
        owner.pause()
        QTest.qWait(350)
        assert calls == []
        owner.prioritize(12, activate=True)
        wait_for(lambda: calls)
        assert calls == [12] and events.loaded == [(12, True)]
        assert owner.pending_codes == (31, 13)
        QTest.qWait(350)
        assert calls == [12] and events.completed == []
        owner.prioritize(13)
        wait_for(lambda: events.completed)
        assert calls == [12, 13, 31]
        assert events.loaded == [(12, True), (13, False), (31, False)]
        assert times[2] - times[1] >= 0.15, "Priority must not make later ticks spin"
        assert events.completed == [True]
    finally:
        owner.reset()


def failure_retry(app, root):
    calls = []

    def load(code):
        calls.append(code)
        if len(calls) == 1:
            raise RuntimeError("provider unavailable")

    owner = WorkspaceLayerLoader(load)
    events = observe(owner)
    try:
        owner.start([12, 13])
        wait_for(lambda: events.failed)
        assert events.failed == [(12, "provider unavailable")]
        assert owner.pending_codes == (13,)
        QTest.qWait(350)
        assert calls == [12] and events.loaded == events.completed == []
        owner.prioritize(12, activate=True)
        wait_for(lambda: events.completed)
        assert calls == [12, 12, 13]
        assert events.loaded == [(12, True), (13, False)]
        assert events.completed == [True] and len(events.failed) == 1
    finally:
        owner.reset()


def reset_during_load(app, root):
    calls, depths = [], []
    depth = 0

    def load(code):
        nonlocal depth
        depth += 1
        calls.append(code)
        depths.append(depth)
        if code == 12:
            owner.reset()
            owner.start([21])
            # A modal Qt loop can deliver the new timer before this call returns.
            nested = QEventLoop()
            QTimer.singleShot(350, nested.quit)
            nested.exec()
        depth -= 1

    owner = WorkspaceLayerLoader(load)
    events = observe(owner)
    try:
        owner.start([13])
        owner.reset()
        QTest.qWait(350)
        assert calls == [] and owner.pending_codes == ()
        owner.start([12, 13])
        wait_for(lambda: events.completed)
        assert calls == [12, 21] and depths == [1, 1]
        assert events.loaded == [(21, False)], "Old Run must not publish its result"
        assert events.completed == [True] and events.failed == []
    finally:
        owner.reset()


def loaded_reset(app, root):
    calls = []
    owner = WorkspaceLayerLoader(calls.append)
    events = observe(owner)

    def replace(code, active):
        if code == 12:
            owner.reset()
            owner.start([21])

    owner.loaded.connect(replace)
    try:
        owner.start([12, 13])
        wait_for(lambda: events.completed)
        assert calls == [12, 21]
        assert events.loaded == [(12, False), (21, False)]
        QTest.qWait(350)
        assert events.completed == [True]
    finally:
        owner.reset()


def completed_restart(app, root):
    calls, during_completion = [], []
    completing = False

    def load(code):
        calls.append(code)
        during_completion.append(completing)

    owner = WorkspaceLayerLoader(load)
    events = observe(owner)

    def restart():
        nonlocal completing
        if len(events.completed) == 1:
            completing = True
            owner.prioritize(21, activate=True)
            nested = QEventLoop()
            QTimer.singleShot(350, nested.quit)
            nested.exec()
            completing = False

    owner.completed.connect(restart)
    try:
        owner.start([])
        wait_for(lambda: len(events.completed) == 2)
        assert calls == [21] and during_completion == [False]
        assert events.loaded == [(21, True)] and events.failed == []
    finally:
        owner.reset()


def pause_during_load(app, root):
    calls = []

    def load(code):
        calls.append(code)
        if code == 12:
            owner.pause()

    owner = WorkspaceLayerLoader(load)
    events = observe(owner)
    try:
        owner.start([12, 13])
        wait_for(lambda: calls)
        QTest.qWait(350)
        assert calls == [12] and owner.pending_codes == (13,)
        assert events.loaded == events.completed == []
        owner.prioritize(13)
        wait_for(lambda: events.completed)
        assert calls == [12, 13] and events.loaded == [(13, False)]
    finally:
        owner.reset()


def dialog_lifecycle(app, root):
    project = QgsProject.instance()
    sources = {code: layer(root, f"class-{code}", code, (0,)) for code in CLASS_ORDER}
    for code, source in sources.items():
        if code not in (12, 13):
            project.addMapLayer(source)
    canvas = QgsMapCanvas()
    iface = SimpleNamespace(mapCanvas=lambda: canvas, activeLayer=lambda: None)
    calls = []

    def load(run_id, record, *, visible):
        code = record["class_code"]
        calls.append((run_id, code, visible))
        project.addMapLayer(sources[code])
        return sources[code].id()

    def visibility(layer_id, visible):
        project.layerTreeRoot().findLayer(layer_id).setItemVisibilityChecked(visible)

    manager = SimpleNamespace(
        load_workspace_class=load, set_layer_visibility=visibility
    )
    workspace = {
        "baseline_stream_id": "fusion:probe",
        "formal_sha256": "a" * 64,
        "feature_count": 2,
        "classes": {str(code): {"class_code": code} for code in CLASS_ORDER},
    }
    with ExitStack() as stack:
        callbacks = {
            name: stack.enter_context(patch.object(ui.ClassRefinementDialog, name))
            for name in (
                "_refresh_table",
                "_update_actions",
                "_update_manual_panel",
                "_active_layer_changed",
                "_select_class_context",
            )
        }
        task_manager = stack.enter_context(
            patch.object(ui.QgsApplication, "taskManager")
        )
        stack.enter_context(
            patch.object(
                ui.class_workspace,
                "save_workspace",
                side_effect=lambda _, value, **kwargs: value,
            )
        )
        dialog = ui.ClassRefinementDialog(iface, manager)
        dialog._run_spec = dict(SPEC, run_dir=str(root))
        dialog._workspace = workspace
        dialog._class_layers = {
            code: source.id()
            for code, source in sources.items()
            if code not in (12, 13)
        }
        refresh_queues = []
        callbacks["_refresh_table"].side_effect = lambda: refresh_queues.append(
            dialog._layer_loader.pending_codes
        )
        try:
            dialog._load_workspace_layers()
            assert refresh_queues[0] == (12, 13)
            dialog._prioritize_class_load(13, activate=True)
            dialog._layer_loader.loaded.connect(
                lambda code, _: dialog._cancel_background_load() if code == 13 else None
            )
            wait_for(lambda: calls)
            assert calls == [(SPEC["run_id"], 13, False)]
            callbacks["_select_class_context"].assert_called_once_with(
                13, activate_layer=True
            )
            assert dialog._class_layers[13] == sources[13].id()
            assert dialog._layer_loader.pending_codes == (12,)
            assert "已暂停" in dialog.baseline_label.text()
            QTest.qWait(350)
            assert len(calls) == 1
            dialog._prioritize_class_load(12)
            wait_for(
                lambda: dialog._layer_loader.pending_codes == () and len(calls) == 2
            )
            wait_for(lambda: "工作层已就绪" in dialog.baseline_label.text())
            assert calls[-1] == (SPEC["run_id"], 12, False)
            assert dialog.cancel_load_btn.isHidden()

            # A newly selected Run invalidates both queued layers and old QgsTasks.
            dialog._layer_loader.start([31])
            dialog.close()
            QTest.qWait(350)
            assert len(calls) == 2 and dialog._layer_loader.pending_codes == (31,)
            dialog.set_run({}, dict(SPEC, run_id="first", run_dir=str(root)), {}, "")
            old_task = task_manager.return_value.addTask.call_args.args[0]
            dialog.set_run({}, dict(SPEC, run_id="second", run_dir=str(root)), {}, "")
            new_task = task_manager.return_value.addTask.call_args.args[0]
            assert task_manager.return_value.addTask.call_count == 2
            before = dialog.baseline_label.text()
            old_task.result_data = {"workspace": workspace}
            old_task.progressChanged.emit(99)
            old_task.taskCompleted.emit()
            assert not new_task.isCanceled() and dialog._workspace is None
            assert dialog._layer_loader.pending_codes == ()
            assert dialog.baseline_label.text() == before
            new_task.result_data = {"workspace": workspace}
            new_task.taskCompleted.emit()
            assert dialog._workspace is workspace
            assert dialog._layer_loader.pending_codes == tuple(CLASS_ORDER)
            dialog.cleanup()
            new_task.taskCompleted.emit()
            QTest.qWait(350)
            assert len(calls) == 2 and dialog._layer_loader.pending_codes == ()
        finally:
            dialog._workspace = None
            dialog.cleanup()
            dialog.close()
            dialog.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            project.removeAllMapLayers()
            canvas.close()


app = QgsApplication([], False)
app.initQgis()
try:
    with tempfile.TemporaryDirectory(prefix="loess-layer-loader-") as directory:
        globals()[sys.argv[2]](app, Path(directory))
        print(sys.argv[2] + ": passed", flush=True)
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
finally:
    app.exitQgis()
