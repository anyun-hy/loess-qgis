import importlib
import sys
import types
from pathlib import Path

import pytest


class _Signal:
    def connect(self, _callback):
        return None

    def emit(self, *_args):
        return None


class _QObject:
    def __init__(self, *_args, **_kwargs):
        pass


class _QProcess:
    NotRunning = 0


class _FinishedProcess:
    def deleteLater(self):
        return None


class _StopTimer:
    def __init__(self):
        self.stop_count = 0

    def stop(self):
        self.stop_count += 1


class _QProcessEnvironment:
    @staticmethod
    def systemEnvironment():
        return _QProcessEnvironment()

    def insert(self, *_args):
        return None


class _QTimer:
    def __init__(self, *_args):
        self.timeout = _Signal()

    def setInterval(self, _value):
        return None

    @staticmethod
    def singleShot(_delay, callback):
        callback()


def _load_runner_module(monkeypatch):
    qgis_module = types.ModuleType("qgis")
    pyqt_module = types.ModuleType("qgis.PyQt")
    qtcore_module = types.ModuleType("qgis.PyQt.QtCore")
    qtcore_module.QObject = _QObject
    qtcore_module.QProcess = _QProcess
    qtcore_module.QProcessEnvironment = _QProcessEnvironment
    qtcore_module.QTimer = _QTimer
    qtcore_module.pyqtSignal = lambda *_args: _Signal()
    monkeypatch.setitem(sys.modules, "qgis", qgis_module)
    monkeypatch.setitem(sys.modules, "qgis.PyQt", pyqt_module)
    monkeypatch.setitem(sys.modules, "qgis.PyQt.QtCore", qtcore_module)
    sys.modules.pop("labeling_tool.core.v5_async_runner", None)
    return importlib.import_module("labeling_tool.core.v5_async_runner")


def _runner(module):
    runner = module.V5AsyncInferenceRunner.__new__(
        module.V5AsyncInferenceRunner
    )
    runner._assembly_queue = [
        {"stream_id": "model:a"},
        {"stream_id": "model:b"},
        {"stream_id": "fusion:approved"},
    ]
    runner._processes = {}
    runner._phase = "assembly"
    runner._spec_path = "/tmp/run_spec.json"
    runner._worker_id = "qgis-test"
    runner._accelerator_worker_id = "qgis-test-accelerator"
    runner._accelerator_done = False
    runner._accelerator_crash_count = 0
    runner.log_line = _Signal()
    return runner


def test_assembly_queue_starts_streams_in_run_spec_order(monkeypatch):
    module = _load_runner_module(monkeypatch)
    runner = _runner(module)
    starts = []
    runner._start_process = lambda *args: starts.append(args)

    runner._start_assembly()
    runner._start_assembly()
    runner._start_assembly()
    runner._start_assembly()

    assert [item[0] for item in starts] == [
        "assemble_stream:model:a",
        "assemble_stream:model:b",
        "assemble_stream:fusion:approved",
        "scale_acceptance",
    ]
    assert runner._phase == "acceptance"


def test_active_assembly_process_blocks_second_stream(monkeypatch):
    module = _load_runner_module(monkeypatch)
    runner = _runner(module)
    runner._processes = {
        "active": {
            "context": {
                "kind": "assemble",
                "stream_id": "model:a",
            }
        }
    }
    starts = []
    messages = []
    runner._start_process = lambda *args: starts.append(args)
    runner.log_line = types.SimpleNamespace(
        emit=lambda level, message: messages.append((level, message))
    )

    runner._start_assembly()

    assert starts == []
    assert messages == [
        (
            "system",
            "[assembly-queue] waiting for active stream: model:a",
        )
    ]
    assert [item["stream_id"] for item in runner._assembly_queue] == [
        "model:a",
        "model:b",
        "fusion:approved",
    ]


def test_resource_budget_reduces_geometry_pool_only_while_package_is_active(monkeypatch):
    module = _load_runner_module(monkeypatch)
    spec = {
        "scaling": {
            "max_cpu_partition_workers": 20,
            "max_cpu_partition_workers_with_package": 16,
        }
    }

    assert module.cpu_worker_limit(spec, package_active=False) == 20
    assert module.cpu_worker_limit(spec, package_active=True) == 16


def test_child_process_thread_limits_prevent_nested_cpu_oversubscription(monkeypatch):
    module = _load_runner_module(monkeypatch)
    spec = {
        "resource_tuning": {
            "resolved": {
                "package_process_threads": 4,
                "unit_process_threads": 1,
                "assembly_process_threads": 1,
            }
        }
    }
    package_values = module.process_thread_environment_values(
        spec,
        {"job": {"job_type": "work_package"}},
    )
    unit_values = module.process_thread_environment_values(
        spec,
        {"job": {"job_type": "unit_fit"}},
    )
    worker_values = module.process_thread_environment_values(
        spec,
        {"kind": "accelerator_worker"},
    )

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert package_values[name] == "4"
        assert worker_values[name] == "4"
        assert unit_values[name] == "1"
    assert package_values["OMP_DYNAMIC"] == "FALSE"
    assert package_values["MKL_DYNAMIC"] == "FALSE"


def test_scheduler_starts_one_persistent_accelerator_and_unit_jobs(monkeypatch):
    module = _load_runner_module(monkeypatch)
    runner = _runner(module)
    runner._running = True
    runner._stopped = False
    runner._phase = "jobs"
    runner._accelerator_done = False
    runner._accelerator_worker_id = "qgis-test-accelerator"
    runner._spec = {
        "run_id": "run-1",
        "scaling": {
            "max_cpu_partition_workers": 2,
            "max_cpu_partition_workers_with_package": 1,
        },
        "boundary_fitting": {"enabled": True},
    }
    runner._cleanup_released_artifacts = lambda: None
    runner._disk_below_reserve = lambda: False
    runner._emit_progress = lambda *_args: None
    calls = []
    unit_jobs = [
        {
            "job_id": 10,
            "job_type": "unit_fit",
            "stream_id": "model:a",
            "unit_id": "core_00001",
            "lease_token": "unit-token",
        }
    ]

    class Database:
        def job_counts(self, _run_id, *, job_type=""):
            assert job_type == "work_package"
            return {"queued": 2}

        def lease_next_job(self, *_args, **_kwargs):
            return unit_jobs.pop(0) if unit_jobs else None

        def lease_next_work_package(self, *_args, **_kwargs):
            raise AssertionError("QGIS must not lease Work Packages")

    runner._database = Database()
    runner._start_accelerator_worker = lambda: calls.append("accelerator")
    runner._start_job = lambda job: calls.append(("unit", job["job_id"]))

    runner._schedule()

    assert calls == ["accelerator", ("unit", 10)]


def test_active_accelerator_process_prevents_a_second_worker(monkeypatch):
    module = _load_runner_module(monkeypatch)
    runner = _runner(module)
    runner._running = True
    runner._stopped = False
    runner._phase = "jobs"
    runner._accelerator_done = False
    runner._spec = {
        "run_id": "run-1",
        "scaling": {
            "max_cpu_partition_workers": 1,
            "max_cpu_partition_workers_with_package": 1,
        },
        "boundary_fitting": {"enabled": True},
    }
    runner._processes = {
        "accelerator": {
            "context": {
                "kind": "accelerator_worker",
                "worker_id": "qgis-test-accelerator",
            }
        }
    }
    runner._cleanup_released_artifacts = lambda: None
    runner._disk_below_reserve = lambda: False
    runner._emit_progress = lambda *_args: None
    starts = []
    runner._start_accelerator_worker = lambda: starts.append("duplicate")
    runner._start_job = lambda job: starts.append(job)
    runner._database = types.SimpleNamespace(
        job_counts=lambda _run_id, job_type="": {"queued": 2},
        lease_next_job=lambda *_args, **_kwargs: None,
    )

    runner._schedule()

    assert starts == []


def test_accelerator_launch_uses_worker_mode_without_package_lease(monkeypatch):
    module = _load_runner_module(monkeypatch)
    runner = _runner(module)
    runner._spec = {
        "runtime": {"effective_device": "cuda"},
        "scaling": {"max_open_frontier_units": 72},
    }
    runner._accelerator_worker_id = "qgis-test-accelerator"
    starts = []
    runner._start_process = lambda *args: starts.append(args)

    runner._start_accelerator_worker()

    assert len(starts) == 1
    label, script, arguments, context = starts[0]
    assert label == "accelerator_worker"
    assert script == "run_work_package.sh"
    assert arguments == [
        "--run-spec",
        "/tmp/run_spec.json",
        "--worker-id",
        "qgis-test-accelerator",
        "--device",
        "cuda",
        "--max-open-frontier-units",
        "72",
        "--resume",
    ]
    assert "--package-id" not in arguments
    assert "--job-id" not in arguments
    assert "--lease-token" not in arguments
    assert context == {
        "kind": "accelerator_worker",
        "worker_id": "qgis-test-accelerator",
    }


def test_qprocess_start_error_is_forced_through_terminal_handler(monkeypatch):
    module = _load_runner_module(monkeypatch)
    runner = _runner(module)
    runner._running = True
    finished = []

    class FailedProcess:
        def errorString(self):
            return "executable not found"

    runner._processes = {
        "failed-start": {
            "process": FailedProcess(),
            "context": {"kind": "assemble", "label": "assemble:model:a"},
            "forced_error": "",
        }
    }
    monkeypatch.setattr(module, "process_is_running", lambda _process: False)
    runner._process_finished = lambda token, code, status: finished.append(
        (token, code, status)
    )

    runner._process_error("failed-start", None)

    assert runner._processes["failed-start"]["forced_error"] == (
        "assemble:model:a process error: executable not found"
    )
    assert finished == [("failed-start", -1, None)]


def test_failed_finish_kills_and_interrupts_other_active_children(
    tmp_path, monkeypatch
):
    module = _load_runner_module(monkeypatch)
    runner = _runner(module)
    runner._running = True
    runner._stopped = False
    runner._scheduler = _StopTimer()
    runner._watchdog = _StopTimer()
    runner._started_at = 0.0
    runner._manual_package_reset = {}
    spec_path = tmp_path / "run_spec.json"
    spec_path.write_text("{}", encoding="utf-8")
    runner._spec_path = str(spec_path)
    runner._spec = {
        "run_id": "run-1",
        "run_dir": str(tmp_path / "run"),
        "output_root": str(tmp_path),
        "streams": [],
    }
    job = {"job_id": 7, "lease_token": "unit-token"}
    runner._processes = {
        "unit": {
            "process": object(),
            "context": {"kind": "job", "job": job},
        }
    }
    terminated = []
    interrupted = []
    runner._terminate_entry = lambda entry, graceful: terminated.append(
        (entry, graceful)
    )
    runner._database = types.SimpleNamespace(
        interrupt_job=lambda job_id, token: interrupted.append((job_id, token)),
        interrupt_work_package_worker=lambda *_args: None,
        job_counts=lambda *_args, **_kwargs: {},
        artifact_cleanup_summary=lambda *_args: {},
        set_run_status=lambda *_args, **_kwargs: True,
    )
    runner._record_startup_index = lambda *_args: None
    runner.pipeline_finished = _Signal()
    monkeypatch.setattr(module, "atomic_write_json", lambda *_args, **_kwargs: None)

    runner._finish(False, "injected crash guard")

    assert len(terminated) == 1
    assert terminated[0][1] is False
    assert interrupted == [(7, "unit-token")]
    assert runner._processes == {}
    assert runner._running is False


def test_failed_stream_stops_queue_before_fusion_and_acceptance(monkeypatch):
    module = _load_runner_module(monkeypatch)
    runner = _runner(module)
    runner._running = True
    runner._processes = {
        "failed": {
            "process": _FinishedProcess(),
            "context": {
                "kind": "assemble",
                "label": "assemble_stream:model:a",
                "stream_id": "model:a",
            },
            "stdout": bytearray(),
            "stderr": bytearray(),
            "forced_error": "",
        }
    }
    finishes = []
    starts = []
    runner._read = lambda *_args: None
    runner._flush = lambda *_args, **_kwargs: None
    runner._finish = lambda success, error: finishes.append((success, error))
    runner._start_assembly = lambda: starts.append("continued")
    runner.step_finished = _Signal()

    runner._process_finished("failed", 2, None)

    assert starts == []
    assert finishes == [
        (False, "assemble_stream:model:a failed (rc=2)"),
    ]
    assert [item["stream_id"] for item in runner._assembly_queue] == [
        "model:a",
        "model:b",
        "fusion:approved",
    ]


def test_user_stop_terminates_active_assembly_and_marks_run_stopped(monkeypatch):
    module = _load_runner_module(monkeypatch)
    runner = _runner(module)
    runner._running = True
    runner._stopped = False
    runner._spec = {"run_id": "run-1"}
    runner._scheduler = _StopTimer()
    runner._watchdog = _StopTimer()
    runner._processes = {
        "active": {
            "context": {
                "kind": "assemble",
                "label": "assemble_stream:model:a",
                "stream_id": "model:a",
            }
        }
    }
    terminated = []
    status_updates = []
    finishes = []
    runner._terminate_entry = lambda entry, graceful: terminated.append(
        (entry["context"]["stream_id"], graceful)
    )
    runner._database = types.SimpleNamespace(
        set_run_status=lambda run_id, status, expected: status_updates.append(
            (run_id, status, expected)
        )
    )
    runner._finish = lambda success, error: finishes.append((success, error))

    runner.stop()

    assert runner._stopped is True
    assert runner._processes == {}
    assert runner._scheduler.stop_count == 1
    assert runner._watchdog.stop_count == 1
    assert terminated == [("model:a", True)]
    assert status_updates == [
        ("run-1", "stopped", ("running", "raster_ready")),
    ]
    assert finishes == [(False, "Pipeline stopped by user")]


def test_user_stop_interrupts_the_worker_owned_package(monkeypatch):
    module = _load_runner_module(monkeypatch)
    runner = _runner(module)
    runner._running = True
    runner._stopped = False
    runner._spec = {"run_id": "run-1"}
    runner._scheduler = _StopTimer()
    runner._watchdog = _StopTimer()
    runner._processes = {
        "accelerator": {
            "context": {
                "kind": "accelerator_worker",
                "label": "accelerator_worker",
                "worker_id": "qgis-test-accelerator",
            }
        }
    }
    terminated = []
    interrupted = []
    runner._terminate_entry = lambda entry, graceful: terminated.append(
        (entry["context"]["worker_id"], graceful)
    )
    runner._database = types.SimpleNamespace(
        interrupt_work_package_worker=lambda run_id, worker_id: interrupted.append(
            (run_id, worker_id)
        ),
        set_run_status=lambda *_args, **_kwargs: None,
    )
    runner._finish = lambda *_args: None

    runner.stop()

    assert terminated == [("qgis-test-accelerator", True)]
    assert interrupted == [("run-1", "qgis-test-accelerator")]
    assert runner._processes == {}


@pytest.mark.parametrize("exit_code", [0, 2])
def test_accelerator_exit_with_pending_package_interrupts_and_restarts(
    monkeypatch,
    exit_code,
):
    module = _load_runner_module(monkeypatch)
    runner = _runner(module)
    runner._running = True
    runner._stopped = False
    runner._accelerator_done = False
    runner._accelerator_crash_count = 0
    runner._spec = {"run_id": "run-1"}
    runner._processes = {
        "accelerator": {
            "process": _FinishedProcess(),
            "context": {
                "kind": "accelerator_worker",
                "label": "accelerator_worker",
                "worker_id": "qgis-test-accelerator",
            },
            "stdout": bytearray(),
            "stderr": bytearray(),
            "forced_error": "",
        }
    }
    interrupted = []
    scheduled = []
    finishes = []
    runner._read = lambda *_args: None
    runner._flush = lambda *_args, **_kwargs: None
    runner._emit_progress = lambda *_args: None
    runner._schedule = lambda: scheduled.append("restart")
    runner._finish = lambda success, error: finishes.append((success, error))
    runner._database = types.SimpleNamespace(
        interrupt_work_package_worker=lambda run_id, worker_id: interrupted.append(
            (run_id, worker_id)
        ),
        job_counts=lambda _run_id, job_type="": {"interrupted": 1},
    )
    runner.step_finished = _Signal()

    runner._process_finished("accelerator", exit_code, None)

    assert interrupted == [("run-1", "qgis-test-accelerator")]
    assert scheduled == ["restart"]
    assert finishes == []
    assert runner._accelerator_done is False
    assert runner._accelerator_crash_count == 1


def test_graceful_worker_stop_escalates_when_sigterm_does_not_exit(monkeypatch):
    module = _load_runner_module(monkeypatch)
    runner = _runner(module)
    signals = []

    class Process:
        def __init__(self):
            self.running = True
            self.waits = []
            self.kill_count = 0

        def processId(self):
            return 4321

        def waitForFinished(self, timeout):
            self.waits.append(timeout)
            return not self.running

        def kill(self):
            self.kill_count += 1
            self.running = False

    process = Process()
    monkeypatch.setattr(
        module,
        "process_is_running",
        lambda candidate: candidate.running,
    )

    def kill_group(pid, signum):
        signals.append((pid, signum))
        if signum == module.signal.SIGKILL:
            process.running = False

    monkeypatch.setattr(module.os, "killpg", kill_group)

    runner._terminate_entry(
        {"process": process, "owns_process_group": True},
        graceful=True,
    )

    assert signals == [
        (4321, module.signal.SIGTERM),
        (4321, module.signal.SIGKILL),
    ]
    assert process.waits == [2500, 1000]
    assert process.running is False


def test_manual_retry_selects_package_reset_instead_of_job_requeue(monkeypatch):
    module = _load_runner_module(monkeypatch)
    runner = module.V5AsyncInferenceRunner.__new__(
        module.V5AsyncInferenceRunner
    )
    calls = []
    runner.run_from_spec = lambda path, **kwargs: calls.append((path, kwargs))

    runner.retry_failed("/tmp/run_spec.json")

    assert calls == [
        (
            "/tmp/run_spec.json",
            {
                "accepted_layer": None,
                "resume": True,
                "reset_failed_packages": True,
            },
        )
    ]


def test_resume_contract_failure_precedes_database_mutation(monkeypatch):
    module = _load_runner_module(monkeypatch)
    runner = module.V5AsyncInferenceRunner.__new__(
        module.V5AsyncInferenceRunner
    )
    runner._running = False
    runner.scripts_dir = "/project/inference_scripts"
    events = []

    def reject_recovery(*_args):
        events.append("validate")
        raise RuntimeError("deployment changed")

    monkeypatch.setattr(module, "validate_recovery_run", reject_recovery)
    monkeypatch.setattr(
        module,
        "RunStateDB",
        lambda *_args: events.append("database-open"),
    )

    with pytest.raises(RuntimeError, match="deployment changed"):
        runner.resume("/run/run_spec.json")

    assert events == ["validate"]


def test_unified_plugin_version_includes_startup_hardening():
    metadata = (
        Path(__file__).resolve().parents[1]
        / "qgis_plugins"
        / "labeling_tool"
        / "metadata.txt"
    ).read_text(encoding="utf-8")

    assert "version=0.4.0" in metadata
    assert "-linux" not in metadata
