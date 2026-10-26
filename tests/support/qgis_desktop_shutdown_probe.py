"""Run inside a disposable QGIS desktop via --code; close its real main window."""

import json
import os
import sys
import threading
import time
import traceback
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import qgis.utils
from qgis.core import Qgis, QgsApplication, QgsProject
from qgis.PyQt import sip
from qgis.PyQt.QtCore import (
    QObject,
    QTimer,
    QtMsgType,
    pyqtSignal,
    qInstallMessageHandler,
)
from qgis.PyQt.QtGui import QGuiApplication
from qgis.utils import iface

ROOT = Path(os.environ["LOESS_DESKTOP_TEST_ROOT"]).resolve()
PLUGIN_ROOT = Path(
    os.environ.get("LOESS_DESKTOP_TEST_PLUGIN_ROOT") or ROOT / "src/labeling_tool"
).resolve()
SCRIPTS_DIR = Path(
    os.environ.get("LOESS_DESKTOP_TEST_SCRIPTS_DIR") or ROOT / "scripts/runtime"
).resolve()
RESULT = Path(os.environ["LOESS_DESKTOP_TEST_RESULT"])
SCENARIO = os.environ["LOESS_DESKTOP_TEST_SCENARIO"]
patches = ExitStack()
threads = []
errors = []
report = {"scenario": SCENARIO, "status": "starting", "pid": os.getpid()}


def save():
    RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def fail(message, code=1):
    report.update(status="failed", error=str(message))
    save()
    # A failed synthetic process must not create/replace an OS crash report.
    os._exit(code)


def message_handler(kind, _context, message):
    if kind == QtMsgType.QtFatalMsg:
        fail(message, 86)


qInstallMessageHandler(message_handler)


def guarded(callback):
    def call():
        try:
            callback()
        except BaseException:
            fail(traceback.format_exc())

    return call


class Request(QObject):
    work = pyqtSignal()


request = Request()
entered = threading.Event()
startup_deadline = 0


@guarded
def check_exit():
    from labeling_tool.qgis_support import qt_lifecycle

    live = [t.objectName() for t in threads if not sip.isdeleted(t) and not t.wait(0)]
    report.update(
        live_threads_after_quit=live,
        awaiting_shutdown=len(qt_lifecycle._awaiting_shutdown),
        python_errors=errors,
    )
    assert report.get("cleanup_called"), report
    assert not live and not qt_lifecycle._awaiting_shutdown and not errors, report
    assert bool(report.get("registry_unload_called")) == (
        SCENARIO != "registry_skip"
    ), report
    report["status"] = "passed"
    save()


@guarded
def close_window():
    tasks = QgsApplication.taskManager().activeTasks()
    if tasks:
        report["startup_tasks"] = [task.description() for task in tasks]
        assert time.monotonic() < startup_deadline, report["startup_tasks"]
        QTimer.singleShot(100, close_window)
        return
    if SCENARIO == "busy_control":
        request.work.emit()
        assert entered.wait(2), "control worker never started"
    report["live_threads_before_close"] = [
        t.objectName() for t in threads if t.isRunning()
    ]
    assert "loess-v5-runtime-control" in report["live_threads_before_close"]
    assert (
        not QgsProject.instance().mapLayers() and not QgsProject.instance().fileName()
    )
    QgsProject.instance().setDirty(False)
    report["status"] = "closing_real_main_window"
    save()
    # QGIS routes closeEvent through fileExit, its registry, and aboutToQuit.
    iface.mainWindow().close()


@guarded
def start():
    global startup_deadline
    report.update(
        qgis=Qgis.QGIS_VERSION,
        qpa=QGuiApplication.platformName(),
        profile=QgsApplication.qgisSettingsDirPath(),
    )
    expected_qpa = "cocoa" if sys.platform == "darwin" else "wayland"
    assert report["qpa"] == expected_qpa, report
    assert Path(report["profile"]).resolve().is_relative_to(RESULT.parent.resolve())
    sys.path.insert(0, str(PLUGIN_ROOT.parent))
    from labeling_tool.main.main_dock import LabelingDockWidget
    from labeling_tool.runs.run_workflow import RunFlowState

    for name in (
        "_load_settings_and_defaults",
        "_save_settings",
        "_restore_latest_ready_run",
    ):
        patches.enter_context(patch.object(LabelingDockWidget, name, lambda *_: None))
    # Keep QGIS's error dialog from blocking only the injected failure case.
    patches.enter_context(
        patch.object(
            qgis.utils,
            "showException",
            lambda *a, **k: errors.append(traceback.format_exc()),
        )
    )
    qgis.utils.plugin_paths.insert(0, str(PLUGIN_ROOT.parent))
    qgis.utils.updateAvailablePlugins()
    assert qgis.utils.loadPlugin("labeling_tool")
    assert qgis.utils.startPlugin("labeling_tool")
    plugin = qgis.utils.plugins["labeling_tool"]
    dock = plugin.dock_widget
    report["source"] = str(
        Path(sys.modules[type(plugin).__module__].__file__).resolve()
    )
    assert Path(report["source"]) == PLUGIN_ROOT / "plugin.py"
    if os.environ.get("LOESS_DESKTOP_TEST_PLUGIN_ROOT"):
        from labeling_tool.runs.deployment_contract import verify_project_runtime

        report["deployment_contract"] = verify_project_runtime(
            SCRIPTS_DIR, plugin_root=PLUGIN_ROOT
        )
        assert report["deployment_contract"]["status"] == "ready", report
        manifest = json.loads((PLUGIN_ROOT / "deployment_manifest.json").read_text())
        report["installed_git"] = manifest["git_sha"]
        report["installed_bundle"] = manifest["source"]["source_bundle_sha256"]
    original_cleanup, original_unload = dock.cleanup, plugin.unload

    def observed_cleanup():
        report["cleanup_called"] = True
        return original_cleanup()

    def observed_unload():
        report["registry_unload_called"] = True
        return original_unload()

    dock.cleanup = observed_cleanup
    plugin.unload = observed_unload
    workflow = dock.workflow
    runner = workflow._create_runner(str(SCRIPTS_DIR))
    workflow._runner = runner
    workflow._state = RunFlowState.FINISHED
    threads.extend(
        (
            runner._runtime_thread,
            runner._log_thread,
            dock.monitor_dialog._query_client._thread,
        )
    )

    def slow_control():
        entered.set()
        time.sleep(0.25)

    runner._worker._schedule = slow_control
    request.work.connect(runner._worker._schedule_safely)
    if SCENARIO == "cleanup_error":

        def broken_cleanup():
            report["injected_cleanup_error"] = True
            raise RuntimeError("synthetic result-panel cleanup failure")

        dock.results_workspace_panel.cleanup = broken_cleanup
    elif SCENARIO == "monitor_cleanup_error":

        def broken_monitor_cleanup(_active):
            report["injected_cleanup_error"] = True
            raise RuntimeError("synthetic monitor-page cleanup failure")

        dock.monitor_dialog.detail_page.set_active = broken_monitor_cleanup
    elif SCENARIO == "registry_skip":
        qgis.utils.active_plugins.remove("labeling_tool")
    QgsApplication.instance().aboutToQuit.connect(check_exit)
    # Only this disposable process exists here; its news feed is not part of
    # thread shutdown coverage and must not make the test depend on a network.
    tasks = QgsApplication.taskManager().activeTasks()
    report["startup_tasks_before_cancel"] = [task.description() for task in tasks]
    for task in tasks:
        task.cancel()
    report["status"] = "ready"
    save()
    startup_deadline = time.monotonic() + 8
    QTimer.singleShot(300, close_window)


def arm():
    iface.initializationCompleted.disconnect(arm)
    QTimer.singleShot(0, start)


iface.initializationCompleted.connect(arm)
QTimer.singleShot(25000, lambda: fail("real QGIS desktop exit timed out"))
