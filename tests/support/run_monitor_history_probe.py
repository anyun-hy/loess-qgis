# ruff: noqa: E402
"""Exercise Qt history wiring; process launch uses an isolated database owner."""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

PLUGIN_ROOT = Path(sys.argv[1])
SCRIPTS_DIR = Path(sys.argv[2])
SCENARIO = sys.argv[3]
POSTGRES_DSN, POSTGRES_SCHEMA = (
    sys.argv[4:6] if SCENARIO == "process_lifecycle" else ("", "")
)
sys.path.insert(0, str(PLUGIN_ROOT))

try:
    from qgis.PyQt.QtCore import QCoreApplication, QProcess, QTimer
except ModuleNotFoundError:
    raise SystemExit(77)

from labeling_tool.runs.v5_async_runner import V5AsyncInferenceRunner
from labeling_tool.shared.state.run_execution_ownership import (
    RunExecutionOwnership,
    RunOwnershipLostError,
)
from labeling_tool.shared.state.run_state_db import RunStateDB

APP = QCoreApplication.instance() or QCoreApplication([])


class Recorder:
    """Record runner calls while leaving persistence rules to the owner tests."""

    def __init__(self, execution_id: str, *, fail_record: bool = False) -> None:
        self.execution_id = execution_id
        self.fail_record = fail_record
        self.events: list[dict] = []
        self.process_starts: list[tuple[str, str, dict]] = []
        self.process_finishes: list[tuple[str, str, bool, str, int]] = []

    def record(self, event: dict) -> None:
        if self.fail_record:
            raise RuntimeError("recorder unavailable")
        self.events.append(dict(event))

    def start_process(self, token: str, label: str, context: dict) -> str:
        self.process_starts.append((token, label, dict(context)))
        return f"span:{token}"

    def finish_process(
        self,
        span_id: str,
        stream_id: str,
        *,
        success: bool,
        error: str,
        exit_code: int,
    ) -> None:
        self.process_finishes.append((span_id, stream_id, success, error, exit_code))


def runner(run_dir: str) -> V5AsyncInferenceRunner:
    value = V5AsyncInferenceRunner(str(SCRIPTS_DIR))
    value._running = True
    value._stopped = False
    value._spec = {"run_id": "native-run", "run_dir": run_dir}
    value._execution_id = "execution-one"
    value._monitor_history_incomplete = False
    value._phase_timing = None
    value._persist_phase_timing = lambda: None
    value._timing_stage = lambda _context: ""
    value._start_assembly = lambda: None
    return value


def close_runner(value: V5AsyncInferenceRunner) -> None:
    for entry in tuple(value._processes.values()):
        process = entry["process"]
        if process.state() != QProcess.ProcessState.NotRunning:
            process.kill()
            process.waitForFinished(1000)
    value._cleanup_executor.shutdown(wait=True)
    value.deleteLater()
    APP.processEvents()


def structured_history_wiring() -> None:
    with tempfile.TemporaryDirectory(prefix="loess-run-history-") as directory:
        value = runner(directory)
        try:
            first = Recorder("execution-one")
            second = Recorder("execution-two")
            value._monitor_history = first
            entry = {
                "monitor_span_id": "parent-one",
                "forced_error": "",
                "context": {"kind": "assemble", "stream_id": "model:a"},
            }
            value._structured(
                entry,
                json.dumps(
                    {
                        "event": "assembly_progress",
                        "stream_id": "model:a",
                        "phase": "assembly",
                        "status": "running",
                    }
                ),
            )
            value._monitor_history = second
            value._execution_id = "execution-two"
            entry["monitor_span_id"] = "parent-two"
            value._structured(
                entry,
                json.dumps(
                    {
                        "event": "assembly_progress",
                        "stream_id": "model:b",
                        "phase": "assembly",
                        "status": "running",
                    }
                ),
            )
            assert first.events[0]["parent_span_id"] == "parent-one"
            assert second.events[0]["parent_span_id"] == "parent-two"
            assert second.events[0]["stream_id"] == "model:b"
            value._monitor_history = Recorder("execution-two", fail_record=True)
            entry = {
                "monitor_span_id": "parent-one",
                "forced_error": "",
                "context": {"kind": "assemble", "stream_id": "model:a"},
            }
            progress: list[dict] = []
            value.stream_progress.connect(progress.append)
            value._structured(
                entry,
                json.dumps(
                    {
                        "event": "assembly_progress",
                        "stream_id": "model:a",
                        "phase": "assembly",
                        "status": "running",
                        "current": 1,
                        "total": 2,
                    }
                ),
            )
            assert value._monitor_history_incomplete is True
            assert entry["forced_error"] == (
                "monitor history persistence failed: recorder unavailable"
            )
            assert progress[0]["parent_span_id"] == "parent-one"
        finally:
            close_runner(value)


def process_lifecycle() -> None:
    with tempfile.TemporaryDirectory(prefix="loess-run-history-") as run_dir:
        value = runner(run_dir)
        database = None
        ownership = None
        recorder = Recorder("execution-one")
        value._monitor_history = recorder
        finished: list[tuple[bool, str]] = []
        value._finish = lambda success, error: finished.append((success, error))
        try:
            with tempfile.TemporaryDirectory(
                prefix="loess-run-history-script-"
            ) as directory:
                script = Path(directory) / "emit-history.sh"
                script.write_text(
                    "#!/bin/sh\n"
                    'echo \'{"event":"assembly_progress",'
                    '"stream_id":"model:a","phase":"assembly",'
                    '"status":"running"}\'\n',
                    encoding="utf-8",
                )
                script.chmod(0o755)
                value.scripts_dir = directory
                deadline = QTimer()
                deadline.setSingleShot(True)
                deadline.timeout.connect(APP.quit)
                value.step_finished.connect(lambda *_args: APP.quit())
                deadline.start(3000)
                try:
                    value._start_process(
                        "assemble_stream:model:a",
                        script.name,
                        [],
                        {"kind": "assemble", "stream_id": "model:a"},
                    )
                except RunOwnershipLostError as error:
                    assert str(error) == "Run execution has no active owner"
                else:
                    raise AssertionError("process started without an active Run owner")
                assert recorder.process_starts == []

                database = RunStateDB(
                    POSTGRES_DSN,
                    postgres_schema=POSTGRES_SCHEMA,
                )
                ownership = RunExecutionOwnership.acquire(
                    database,
                    "native-run",
                    run_dir=run_dir,
                    worker_id="monitor-history-probe",
                    trigger_type="start",
                )
                value._database = database
                value._run_ownership = ownership
                value._execution_id = ownership.identity.execution_id
                value._start_process(
                    "assemble_stream:model:a",
                    script.name,
                    [],
                    {"kind": "assemble", "stream_id": "model:a"},
                )
                APP.exec()
                deadline.stop()
                assert len(recorder.process_starts) == 1
                token, label, context = recorder.process_starts[0]
                assert label == "assemble_stream:model:a"
                assert context["stream_id"] == "model:a"
                assert recorder.events[0]["parent_span_id"] == f"span:{token}"
                assert recorder.process_finishes == [
                    (f"span:{token}", "model:a", True, "", 0)
                ]
                assert finished == []
        finally:
            try:
                close_runner(value)
            finally:
                if ownership is not None:
                    try:
                        assert database is not None
                        assert database.monitor_history.finish_execution(
                            "native-run",
                            ownership.identity.execution_id,
                            status="stopped",
                            message="native process lifecycle probe complete",
                        )
                    finally:
                        ownership.close()


SCENARIOS = {
    "structured_history_wiring": structured_history_wiring,
    "process_lifecycle": process_lifecycle,
}


try:
    SCENARIOS[SCENARIO]()
    print(f"{SCENARIO}: passed", flush=True)
except Exception:
    traceback.print_exc()
    raise SystemExit(1) from None
