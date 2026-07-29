import importlib
import sys
import types
from pathlib import Path


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
    failures = []
    runner._start_process = lambda *args: starts.append(args)
    runner._finish = lambda success, error: failures.append((success, error))

    runner._start_assembly()

    assert starts == []
    assert failures == [
        (
            False,
            "assembly queue invariant violated: another stream assembly is active",
        )
    ]
    assert [item["stream_id"] for item in runner._assembly_queue] == [
        "model:a",
        "model:b",
        "fusion:approved",
    ]


def test_ubuntu_plugin_version_marks_queue_hardening():
    metadata = (
        Path(__file__).resolve().parents[1]
        / "qgis_plugins"
        / "labeling_tool"
        / "metadata.txt"
    ).read_text(encoding="utf-8")

    assert "version=0.3.0-linux.2" in metadata

