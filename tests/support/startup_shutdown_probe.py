# ruff: noqa: E402
"""Real queued startup callbacks and native shutdown; no production I/O."""

import os
import sys
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

ROOT = Path(sys.argv[1]).resolve()
SCENARIO = sys.argv[2]
sys.path.insert(0, str(ROOT / "src"))
try:
    from qgis.core import QgsApplication
    from qgis.PyQt import sip
    from qgis.PyQt.QtCore import QCoreApplication, QEvent, QTimer
    from qgis.PyQt.QtWidgets import QMainWindow
except ModuleNotFoundError:
    raise SystemExit(77)

from labeling_tool.main import main_dock
from labeling_tool.plugin import LabelingTool
from labeling_tool.qgis_support import qt_lifecycle
from labeling_tool.qgis_support.qt6_api import RIGHT_DOCK_WIDGET_AREA


def main():
    assert Path(main_dock.__file__).resolve().is_relative_to(ROOT / "src")
    profile = TemporaryDirectory(prefix="loess-startup-shutdown-")
    app = QgsApplication([], False, profile.name)
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
    index_calls = []
    observations = []
    plugin = LabelingTool(iface)

    with ExitStack() as patches:
        for name in ("_load_settings_and_defaults", "_save_settings"):
            patches.enter_context(
                patch.object(main_dock.LabelingDockWidget, name, lambda *_: None)
            )

        def index(output):
            index_calls.append(output)
            return {"latest": None, "latest_ready": None}

        patches.enter_context(
            patch.object(main_dock.run_index, "load_startup_candidates", index)
        )

        def unload():
            dock = plugin.dock_widget
            threads = (dock.monitor_dialog._query_client._thread,)
            plugin.unload()
            assert dock.workflow is None
            assert not qt_lifecycle._awaiting_shutdown
            assert not any(
                not sip.isdeleted(thread) and thread.isRunning() for thread in threads
            )
            assert not errors, errors
            assert len(index_calls) == (1 if SCENARIO == "normal_startup" else 0)
            if SCENARIO == "late_environment_report":
                assert not dock._environment_report_current
                assert not environment_updates
            observations.append("unloaded")
            plugin.unload()

        def create():
            dock = main_dock.LabelingDockWidget(window)
            plugin.dock_widget = dock
            window.addDockWidget(RIGHT_DOCK_WIDGET_AREA, dock)
            return dock

        environment_updates = []

        def start():
            dock = create()
            if SCENARIO == "active_workflow":
                patches.enter_context(
                    patch.object(
                        type(dock.workflow),
                        "is_active",
                        PropertyMock(return_value=True),
                    )
                )
            if SCENARIO == "late_environment_report":
                patches.enter_context(
                    patch.object(
                        dock.plan_panel, "set_environment", environment_updates.append
                    )
                )
                QTimer.singleShot(
                    0,
                    lambda: dock.config_manager.report_ready.emit(
                        {"status": "ready", "checks": [], "effective": {}}
                    ),
                )
            if SCENARIO == "cleanup_started":
                real_retire = dock._retire_manual_load_task

                def restore_during_cleanup():
                    assert dock.workflow is not None
                    dock._restore_latest_ready_run()
                    real_retire()

                patches.enter_context(
                    patch.object(
                        dock, "_retire_manual_load_task", restore_during_cleanup
                    )
                )
            if SCENARIO == "early_quit":
                app.aboutToQuit.connect(unload)
                app.quit()
            elif SCENARIO in {"normal_startup", "active_workflow"}:
                QTimer.singleShot(0, finish)
            else:
                # The constructor's real singleShot callback is still pending.
                unload()
                QTimer.singleShot(0, app.quit)

        def finish():
            unload()
            app.quit()

        def timeout():
            print({"timeout": SCENARIO, "errors": errors}, flush=True)
            os._exit(2)

        QTimer.singleShot(5000, timeout)
        QTimer.singleShot(0, start)
        app.exec()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert observations == ["unloaded"], (observations, errors)
        assert not errors and not qt_lifecycle._retiring, errors
        app.exitQgis()
    profile.cleanup()
    print(SCENARIO + ": passed", flush=True)


if __name__ == "__main__":
    main()
