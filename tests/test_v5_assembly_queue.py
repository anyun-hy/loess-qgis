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

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert package_values[name] == "4"
        assert unit_values[name] == "1"
    assert package_values["OMP_DYNAMIC"] == "FALSE"
    assert package_values["MKL_DYNAMIC"] == "FALSE"


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
