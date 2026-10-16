# ruff: noqa: E402
"""Exercise QGIS's aboutToQuit -> plugin.unload boundary in one event loop.

Real dock, workflow, runtime/log and monitor threads; synthetic slow work only.
No model, database, user settings or deployed plugin is touched.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(sys.argv[1])
SCENARIO = sys.argv[2]
sys.path.insert(0, str(ROOT / "src"))
try:
    from qgis.core import QgsApplication
    from qgis.PyQt import sip
    from qgis.PyQt.QtCore import QObject, QTimer, pyqtSignal
    from qgis.PyQt.QtWidgets import QMainWindow
except ModuleNotFoundError:
    raise SystemExit(77)

from labeling_tool.main.main_dock import LabelingDockWidget
from labeling_tool.plugin import LabelingTool
from labeling_tool.qgis_support import qt_lifecycle


def main():
    app = QgsApplication([], False)
    app.initQgis()
    app.setQuitOnLastWindowClosed(False)
    window = QMainWindow()
    iface = SimpleNamespace(
        mapCanvas=lambda: None,
        mainWindow=lambda: window,
        removeDockWidget=window.removeDockWidget,
    )
    errors = []
    sys.excepthook = lambda kind, value, tb: errors.append(str(value))
    with ExitStack() as patches:
        for name in (
            "_load_settings_and_defaults",
            "_save_settings",
            "_restore_latest_ready_run",
        ):
            patches.enter_context(
                patch.object(LabelingDockWidget, name, lambda *_: None)
            )
        plugin = LabelingTool(iface)
        dock = LabelingDockWidget(window)
        plugin.dock_widget = dock
        workflow = dock.workflow
        # Also cover a previous runner already retiring when QGIS exits.
        previous = workflow._create_runner(str(ROOT / "scripts/runtime"))
        workflow._runner = previous
        previous_threads = (previous._runtime_thread, previous._log_thread)
        workflow._retire_current_runner()
        runner = workflow._create_runner(str(ROOT / "scripts/runtime"))
        workflow._runner = runner
        completed = []
        previous.shutdown_finished.connect(lambda: completed.append("previous"))
        runner.shutdown_finished.connect(lambda: completed.append("current"))
        threads = previous_threads + (
            runner._runtime_thread,
            runner._log_thread,
            dock.monitor_dialog._query_client._thread,
        )
        entered = threading.Event()

        def slow_read():
            entered.set()
            time.sleep(0.15)

        class Request(QObject):
            work = pyqtSignal()

        request = Request()
        runner._worker._schedule = slow_read
        request.work.connect(runner._worker._schedule_safely)
        observations = []

        def unload():
            plugin.unload()
            observations.append(
                {
                    "live_threads": [
                        t.objectName()
                        for t in threads
                        if not sip.isdeleted(t) and t.isRunning()
                    ],
                    "awaiting_shutdown": len(qt_lifecycle._awaiting_shutdown),
                    "errors": list(errors),
                }
            )
            # Failure exits only this synthetic subprocess without producing a
            # native crash report. No second event loop may make the test pass.
            if any(observations[-1].values()):
                print(observations[-1], flush=True)
                os._exit(1)
            plugin.unload()  # Unload remains idempotent.
            if SCENARIO == "unload_busy":
                QTimer.singleShot(0, app.quit)

        if SCENARIO != "unload_busy":
            app.aboutToQuit.connect(unload)

        def begin():
            if SCENARIO.endswith("busy"):
                request.work.emit()
                assert entered.wait(2), "native worker did not start"
            if SCENARIO == "unload_busy":
                unload()
            else:
                app.quit()

        QTimer.singleShot(0, begin)

        def timed_out():
            print(
                {
                    "timeout_live_threads": [
                        t.objectName()
                        for t in threads
                        if not sip.isdeleted(t) and t.isRunning()
                    ],
                    "retiring_types": [
                        (
                            type(owner).__name__,
                            sip.isdeleted(owner),
                            "previous"
                            if owner is previous
                            else "current"
                            if owner is runner
                            else "other",
                        )
                        for owner in qt_lifecycle._retiring
                    ],
                    "completed": completed,
                    "errors": errors,
                },
                flush=True,
            )
            os._exit(2)

        QTimer.singleShot(5000, timed_out)
        app.exec()
        assert len(observations) == 1 and not errors, observations
        assert not qt_lifecycle._retiring, "completed owners were not deleted at exit"
        app.exitQgis()
    print(SCENARIO + ": passed", flush=True)


if __name__ == "__main__":
    main()
