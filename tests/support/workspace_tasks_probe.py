# ruff: noqa: E402
"""Use native QgsTasks with controlled results and isolated workspace inputs."""

from __future__ import annotations

import sys
import tempfile
import threading
import time
import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
try:
    from qgis.core import QgsApplication
    from qgis.gui import QgsMapCanvas
    from qgis.PyQt.QtCore import QCoreApplication, QEvent, QThread
    from qgis.PyQt.QtTest import QTest
except ModuleNotFoundError:
    raise SystemExit(77)

from labeling_tool.refinement import class_refinement_dialog as ui
from labeling_tool.refinement import class_workspace as workspace
from labeling_tool.refinement.workspace_tasks import WorkspaceTasks

STREAM = {"stream_id": "fusion:probe", "kind": "fusion", "status": "ready"}


def spec(root, run_id="probe"):
    return {"run_id": run_id, "run_dir": str(root / run_id)}


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(10)
    assert predicate(), "Expected Qt event was not delivered"


def observe(owner):
    events = SimpleNamespace(progress=[], probed=[], initialized=[], terminated=[])
    owner.progress.connect(lambda op, value: events.progress.append((op, value)))
    owner.probed.connect(events.probed.append)
    owner.initialized.connect(events.initialized.append)
    owner.terminated.connect(lambda op, error: events.terminated.append((op, error)))
    return events


def replacement(app, root):
    submitted = []
    owner = WorkspaceTasks(submit_task=submitted.append)
    events = observe(owner)
    try:
        owner.probe(spec(root, "old"), [STREAM])
        old = submitted[-1]
        old.progressChanged.emit(10)
        assert owner.busy and events.progress == [("probe", 10.0)]
        owner.probe(spec(root, "new"), [STREAM])
        current = submitted[-1]
        assert old.isCanceled() and current.run_spec["run_id"] == "new"
        old.result_data = {"workspace": "obsolete"}
        old.progressChanged.emit(99)
        old.taskCompleted.emit()
        old.taskTerminated.emit()
        assert owner.busy and not events.probed and not events.terminated
        assert events.progress == [("probe", 10.0)]
        payload = {"workspace": {"baseline_stream_id": STREAM["stream_id"]}}
        current.result_data = payload
        current.progressChanged.emit(75)
        current.taskCompleted.emit()
        assert not owner.busy and events.probed == [payload]
        current.taskCompleted.emit()
        current.taskTerminated.emit()
        current.progressChanged.emit(100)
        assert events.probed == [payload] and events.terminated == []
        assert events.progress[-1] == ("probe", 75.0)
    finally:
        owner.cancel()


def cancel_lifetime(app, root):
    submitted = []
    owner = WorkspaceTasks(submit_task=submitted.append)
    events = observe(owner)
    owner.cancel()
    owner.probe(spec(root), [STREAM])
    old = submitted[-1]
    owner.cancel()
    owner.cancel()
    assert old.isCanceled() and not owner.busy
    old.progressChanged.emit(100)
    old.taskCompleted.emit()
    old.taskTerminated.emit()
    assert events.probed == events.terminated == events.progress == []
    owner.initialize(spec(root), STREAM)
    failed = submitted[-1]
    failed.error_message = "invalid class geometry"
    failed.taskTerminated.emit()
    assert events.terminated == [("initialize", "invalid class geometry")]
    assert not owner.busy
    owner.probe(spec(root), [STREAM])
    submitted[-1].taskTerminated.emit()
    assert events.terminated[-1] == ("probe", "") and not owner.busy
    owner.probe(spec(root), [STREAM])
    retired = submitted[-1]
    owner.cancel()
    owner.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    retired.progressChanged.emit(100)
    retired.taskCompleted.emit()
    retired.taskTerminated.emit()
    assert events.probed == [] and len(events.terminated) == 2


def terminal_reentry(app, root):
    submitted = []
    owner = WorkspaceTasks(submit_task=submitted.append)
    events = observe(owner)
    owner.probed.connect(lambda _: owner.initialize(spec(root), STREAM, replace=True))
    try:
        owner.probe(spec(root), [STREAM])
        first = submitted[-1]
        try:
            owner.initialize(spec(root), STREAM)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Initialization must not replace an active check")
        assert len(submitted) == 1 and not first.isCanceled()
        first.result_data = {"eligible_fusions": [STREAM]}
        first.taskCompleted.emit()
        second = submitted[-1]
        assert owner.busy and len(submitted) == 2 and second.replace
        first.taskTerminated.emit()
        first.progressChanged.emit(99)
        assert owner.busy and events.terminated == []
        second.result_data = {"workspace": {"classes": {}}}
        second.taskCompleted.emit()
        assert not owner.busy and events.initialized == [second.result_data]
    finally:
        owner.cancel()


def submission_failure(app, root):
    submitted = []

    def submit(task):
        submitted.append(task)
        if len(submitted) == 1:
            raise RuntimeError("task manager unavailable")

    owner = WorkspaceTasks(submit_task=submit)
    events = observe(owner)
    try:
        owner.probe(spec(root), [STREAM])
        assert not owner.busy
        assert events.terminated == [("probe", "task manager unavailable")]
        submitted[0].taskCompleted.emit()
        assert events.probed == []
        owner.probe(spec(root, "retry"), [STREAM])
        assert owner.busy
        submitted[-1].result_data = {"workspace": None}
        submitted[-1].taskCompleted.emit()
        assert events.probed == [{"workspace": None}] and not owner.busy
        with patch.object(
            workspace,
            "ClassWorkspaceProbeTask",
            side_effect=RuntimeError("cannot create task"),
        ):
            owner.probe(spec(root), [STREAM])
        assert not owner.busy and events.terminated[-1] == (
            "probe",
            "cannot create task",
        )
        owner.initialize(spec(root), STREAM)
        assert owner.busy
    finally:
        owner.cancel()


def late_cancelled_success(app, root):
    submitted = []
    cancelled = []

    def submit(task):
        submitted.append(task)
        return QgsApplication.taskManager().addTask(task)

    owner = WorkspaceTasks(submit_task=submit)
    events = observe(owner)
    value = {"baseline_stream_id": STREAM["stream_id"], "classes": {}}
    try:

        def cancel_statistics(*args, **kwargs):
            task = submitted[-1]
            task.cancel()
            cancelled.append(task.isCanceled())
            raise workspace.ClassWorkspaceCancelled("statistics cancelled")

        with (
            patch.object(
                workspace, "initialize_workspace", return_value=value
            ) as create,
            patch.object(
                workspace, "workspace_source_statistics", side_effect=cancel_statistics
            ),
        ):
            owner.initialize(spec(root), STREAM, replace=True)
            wait_for(lambda: events.initialized or events.terminated)
            assert create.call_args.kwargs["replace"] is True
        assert cancelled == [True]
        assert events.initialized == [{"workspace": value, "statistics": {}}], vars(
            events
        )
        assert not owner.busy and events.terminated == []
    finally:
        owner.cancel()


def thread_affinity(app, root):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    worker_threads, receiver_threads = [], []
    owner = WorkspaceTasks()
    events = observe(owner)
    original = workspace.ClassWorkspaceProbeTask.run

    def run(task):
        worker_threads.append(QThread.currentThread() == app.thread())
        task.setProgress(25)
        entered.set()
        try:
            if not release.wait(5):
                task.error_message = "test gate expired"
                return False
            return original(task)
        finally:
            finished.set()

    owner.progress.connect(
        lambda *_: receiver_threads.append(QThread.currentThread() == app.thread())
    )
    owner.probed.connect(
        lambda *_: receiver_threads.append(QThread.currentThread() == app.thread())
    )
    try:
        with (
            patch.object(workspace.ClassWorkspaceProbeTask, "run", new=run),
            patch.object(workspace, "approved_fusion_streams", return_value=[STREAM]),
        ):
            owner.probe(spec(root), [STREAM])
            wait_for(entered.is_set)
            wait_for(lambda: events.progress)
            assert owner.busy and events.probed == []
            release.set()
            wait_for(lambda: events.probed or events.terminated)
            assert events.terminated == []
            assert events.probed[0]["eligible_fusions"] == [STREAM]
            assert worker_threads == [False]
            assert receiver_threads and all(receiver_threads)
    finally:
        release.set()
        owner.cancel()
        wait_for(finished.is_set)


@contextmanager
def dialog_fixture(root):
    canvas = QgsMapCanvas()
    iface = SimpleNamespace(mapCanvas=lambda: canvas, activeLayer=lambda: None)
    with ExitStack() as stack:
        callbacks = {
            name: stack.enter_context(patch.object(ui.ClassRefinementDialog, name))
            for name in (
                "_refresh_table",
                "_update_actions",
                "_update_manual_panel",
                "_load_workspace_layers",
            )
        }
        manager = stack.enter_context(patch.object(ui.QgsApplication, "taskManager"))
        warning = stack.enter_context(patch.object(ui.QMessageBox, "warning"))
        stack.enter_context(
            patch.object(
                workspace,
                "save_workspace",
                side_effect=lambda _, value, **kwargs: value,
            )
        )
        dialog = ui.ClassRefinementDialog(iface, SimpleNamespace())
        try:
            yield dialog, manager.return_value.addTask, warning, callbacks
        finally:
            dialog._workspace = None
            dialog.cleanup()
            dialog.close()
            dialog.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            canvas.close()


def dialog_lifecycle(app, root):
    with dialog_fixture(root) as (dialog, submit, warning, callbacks):
        dialog.set_run(
            {"ready_streams": [STREAM, dict(STREAM, status="pending")]},
            spec(root),
            {},
            "",
        )
        task = submit.call_args.args[0]
        assert task.streams == [STREAM]
        task.progressChanged.emit(35)
        assert "35%" in dialog.baseline_label.text()
        task.result_data = {"eligible_fusions": [STREAM], "workspace": None}
        task.taskCompleted.emit()
        assert dialog.initialize_btn.isEnabled()
        dialog._initialize_workspace()
        initialized = submit.call_args.args[0]
        initialized.error_message = "bad input identity"
        initialized.taskTerminated.emit()
        assert "bad input identity" in dialog.baseline_label.text()
        assert dialog.initialize_btn.isEnabled() and warning.call_count == 1
        dialog._initialize_workspace()
        pending = submit.call_args.args[0]
        dialog._cancel_background_load()
        assert pending.isCanceled() and "已暂停" in dialog.baseline_label.text()
        before = dialog.baseline_label.text()
        pending.result_data = {"workspace": {"baseline_stream_id": STREAM["stream_id"]}}
        pending.taskCompleted.emit()
        assert dialog._workspace is None and dialog.baseline_label.text() == before
        callbacks["_load_workspace_layers"].assert_not_called()
        # A pure manual Run with one valid Fusion still initializes automatically.
        dialog.set_run(
            {"ready_streams": [STREAM]},
            dict(spec(root, "manual"), manual_only=True),
            {},
            "",
        )
        manual_probe = submit.call_args.args[0]
        manual_probe.result_data = {"eligible_fusions": [STREAM], "workspace": None}
        count = submit.call_count
        manual_probe.taskCompleted.emit()
        assert submit.call_count == count + 1
        manual_init = submit.call_args.args[0]
        value = {"baseline_stream_id": STREAM["stream_id"], "classes": {}}
        manual_init.result_data = {"workspace": value, "statistics": {12: 3}}
        manual_init.taskCompleted.emit()
        assert dialog._workspace is value and dialog._workspace_statistics == {12: 3}
        callbacks["_load_workspace_layers"].assert_called_once_with()
        assert not dialog.initialize_btn.isEnabled()


def dialog_confirmation(app, root):
    with dialog_fixture(root) as (dialog, submit, warning, callbacks):
        dialog.set_run({"ready_streams": [STREAM]}, spec(root, "old"), {}, "")
        task = submit.call_args.args[0]
        value = {"baseline_stream_id": STREAM["stream_id"], "classes": {}}
        task.result_data = {
            "eligible_fusions": [STREAM],
            "workspace": value,
            "review_refresh": {"required": True, "safe_to_replace": True},
        }
        task.taskCompleted.emit()
        assert dialog.initialize_btn.isEnabled()
        callbacks["_load_workspace_layers"].assert_not_called()
        with patch.object(ui.QMessageBox, "question", return_value=ui.NO):
            dialog._initialize_workspace()
        assert submit.call_count == 1
        with patch.object(ui.QMessageBox, "question", return_value=ui.YES):
            dialog._initialize_workspace()
        assert submit.call_args.args[0].replace
        dialog._cancel_background_load(silent=True)

        def change_run(*args, **kwargs):
            dialog.set_run({}, spec(root, "new"), {}, "")
            return ui.YES

        count = submit.call_count
        with patch.object(ui.QMessageBox, "question", side_effect=change_run):
            dialog._initialize_workspace()
        assert submit.call_count == count + 1, (
            "An old confirmation cannot initialize the new Run"
        )
        new = submit.call_args.args[0]
        assert new.run_spec["run_id"] == "new"
        dialog.cleanup()
        assert new.isCanceled()
        before = dialog.baseline_label.text()
        new.progressChanged.emit(99)
        new.result_data = {"workspace": value}
        new.taskCompleted.emit()
        assert dialog._workspace is None and dialog.baseline_label.text() == before
        assert warning.call_count == 0


app = QgsApplication([], False)
app.initQgis()
try:
    with tempfile.TemporaryDirectory(prefix="loess-workspace-tasks-") as directory:
        globals()[sys.argv[2]](app, Path(directory))
        print(sys.argv[2] + ": passed", flush=True)
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
finally:
    app.exitQgis()
