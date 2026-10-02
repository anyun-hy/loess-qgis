"""Isolated native-QGIS probe; no inference, database, map, or output writes.

Run with QGIS's Python and pass the plugin parent and inference script directory.
The pytest launcher stays in Conda and starts this probe in a separate process
so native QGIS libraries never enter the Conda test runner.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

CALLBACKS = (
    "_schedule_safely",
    "_heartbeat_and_watchdog",
    "_flush_job_heartbeats",
    "_flush_ui_events",
    "_persist_log",
)


def probe(plugin_parent: str, scripts_dir: str) -> dict:
    sys.path.insert(0, plugin_parent)
    from qgis.PyQt.QtCore import (
        PYQT_VERSION_STR,
        QT_VERSION_STR,
        QCoreApplication,
        QObject,
        QTimer,
        pyqtSignal,
        pyqtSlot,
    )

    from labeling_tool.runs.v5_async_runner import (
        ThreadedV5AsyncInferenceRunner,
        V5AsyncInferenceRunner,
    )

    app = QCoreApplication([])
    main_tid = threading.get_ident()
    calls = {name: [] for name in CALLBACKS}
    codes = {getattr(V5AsyncInferenceRunner, name).__code__: name for name in CALLBACKS}
    ticks = []
    waits = []
    logs = []
    progress = []
    gui_deliveries = []
    errors = []

    # Python 3.12+ monitoring observes the original methods across native Qt
    # threads without wrapping slots or accidentally supplying missing decorators.
    monitor = sys.monitoring
    tool = monitor.PROFILER_ID
    monitor.use_tool_id(tool, "loess-qt-thread-probe")

    def entered(code, _offset):
        calls[codes[code]].append(threading.get_ident())

    monitor.register_callback(tool, monitor.events.PY_START, entered)
    for code in codes:
        monitor.set_local_events(tool, code, monitor.events.PY_START)

    runner = ThreadedV5AsyncInferenceRunner(scripts_dir)
    worker = runner._worker

    def slow_read():
        start = time.monotonic()
        time.sleep(0.4)  # Controlled I/O delay, not real database access.
        waits.append((start, time.monotonic()))

    # Keep every real connection and slot intact; replace only the operation
    # called by the scheduler so this probe cannot launch or mutate a Run.
    worker._schedule = slow_read

    class Driver(QObject):
        @pyqtSlot()
        def arm(self):
            worker.log_line.emit("system", "thread-affinity-probe")
            for current in (1, 2):
                worker._ui_event_buffer.enqueue_stream_progress(
                    {
                        "event": "fit_progress",
                        "stream_id": "model:probe",
                        "current": current,
                        "total": 2,
                    }
                )
            for index, timer in enumerate(
                (
                    worker._scheduler,
                    worker._watchdog,
                    worker._heartbeat_timer,
                    worker._ui_flush_timer,
                )
            ):
                timer.setSingleShot(True)
                timer.start(30 + index * 20)

    class Receiver(QObject):
        @pyqtSlot(object)
        def log_batch(self, records):
            gui_deliveries.append(threading.get_ident())
            logs.extend(records)

        @pyqtSlot(object)
        def progress_batch(self, records):
            gui_deliveries.append(threading.get_ident())
            progress.extend(records)

    class Request(QObject):
        start = pyqtSignal()

    driver = Driver()
    driver.moveToThread(runner._runtime_thread)
    runner._runtime_thread.finished.connect(driver.deleteLater)
    request = Request()
    request.start.connect(driver.arm)
    receiver = Receiver()
    runner.log_batch.connect(receiver.log_batch)
    runner.stream_progress_batch.connect(receiver.progress_batch)
    heartbeat = QTimer()
    heartbeat.setInterval(20)
    heartbeat.timeout.connect(lambda: ticks.append(time.monotonic()))
    deadline = QTimer()
    deadline.setSingleShot(True)
    deadline.timeout.connect(app.quit)
    original_excepthook = sys.excepthook
    sys.excepthook = lambda kind, value, tb: (errors.append(str(value)), app.quit())
    try:
        ticks.append(time.monotonic())
        heartbeat.start()
        deadline.start(1200)
        request.start.emit()
        app.exec()
    finally:
        heartbeat.stop()
        deadline.stop()
        # Shutdown is asynchronous; keep the native loop alive while workers
        # release their resources. This wait belongs to the test, not the UI.
        runner.shutdown_finished.connect(app.quit)
        runner.shutdown()
        QTimer.singleShot(2000, app.quit)
        app.exec()
        for code in codes:
            monitor.set_local_events(tool, code, 0)
        monitor.register_callback(tool, monitor.events.PY_START, None)
        monitor.free_tool_id(tool)
        sys.excepthook = original_excepthook

    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    responsive_ticks = sum(start < tick < end for start, end in waits for tick in ticks)
    result = {
        "qt": QT_VERSION_STR,
        "pyqt": PYQT_VERSION_STR,
        "callbacks_on_gui": {name: main_tid in tids for name, tids in calls.items()},
        "callback_counts": {name: len(tids) for name, tids in calls.items()},
        "max_main_loop_gap_ms": round(max(gaps, default=0) * 1000, 1),
        "heartbeats_during_slow_read": responsive_ticks,
        "gui_delivery_count": len(gui_deliveries),
    }
    print(json.dumps(result), flush=True)
    assert not errors, errors
    assert all(calls.values()), result
    assert not any(result["callbacks_on_gui"].values()), result
    assert len({tid for tids in calls.values() for tid in tids}) == 1, result
    assert len(waits) == 1 and responsive_ticks >= 5, result
    assert max(gaps) < 0.2, result
    assert gui_deliveries and all(tid == main_tid for tid in gui_deliveries), result
    assert [record["message"] for record in logs] == ["thread-affinity-probe"]
    assert len(progress) == 1 and progress[0]["current"] == 2, progress
    assert not runner._runtime_thread.isRunning()
    assert not runner._log_thread.isRunning()
    return result


if __name__ == "__main__":
    try:
        probe(str(Path(sys.argv[1]).resolve()), str(Path(sys.argv[2]).resolve()))
    except ModuleNotFoundError as error:
        if error.name in {"qgis", "PyQt6"}:
            print("Native QGIS/PyQt6 runtime is unavailable", file=sys.stderr)
            raise SystemExit(77)
        raise
