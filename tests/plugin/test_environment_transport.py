import importlib
import json
import sys
import types
from enum import IntEnum
from pathlib import Path

import pytest

from labeling_tool.main.environment_transport import (
    environment_report_path,
    load_environment_report,
    persist_environment_report,
)


class _Signal:
    def __init__(self):
        self.values = []

    def connect(self, _callback):
        return None

    def emit(self, *args):
        self.values.append(args)


class _QObject:
    def __init__(self, *_args, **_kwargs):
        pass


class _ProcessError(IntEnum):
    FailedToStart = 0
    Crashed = 1
    ReadError = 3


class _ProcessState:
    NotRunning = 0


class _UnixProcessFlag:
    CreateNewSession = 1


class _UnixProcessParameters:
    def __init__(self):
        self.flags = 0


class _QProcess:
    ProcessError = _ProcessError
    ProcessState = _ProcessState
    UnixProcessFlag = _UnixProcessFlag
    UnixProcessParameters = _UnixProcessParameters


class _QProcessEnvironment:
    @staticmethod
    def systemEnvironment():
        return _QProcessEnvironment()


class _FinishedProcess:
    def __init__(self, stdout=b"", stderr=b"", error="process error"):
        self.stdout = bytearray(stdout)
        self.stderr = bytearray(stderr)
        self.error = error
        self.deleted = 0

    def readAllStandardOutput(self):
        value = bytes(self.stdout)
        self.stdout.clear()
        return value

    def readAllStandardError(self):
        value = bytes(self.stderr)
        self.stderr.clear()
        return value

    def errorString(self):
        return self.error

    def deleteLater(self):
        self.deleted += 1


def _load_inference_config(monkeypatch):
    qgis_module = types.ModuleType("qgis")
    pyqt_module = types.ModuleType("qgis.PyQt")
    qtcore_module = types.ModuleType("qgis.PyQt.QtCore")
    core_module = types.ModuleType("qgis.core")
    qtcore_module.PYQT_VERSION_STR = "6.10.2"
    qtcore_module.QT_VERSION_STR = "6.10.2"
    qtcore_module.QObject = _QObject
    qtcore_module.QProcess = _QProcess
    qtcore_module.QProcessEnvironment = _QProcessEnvironment
    qtcore_module.pyqtSignal = lambda *_args: _Signal()
    core_module.Qgis = types.SimpleNamespace(QGIS_VERSION="4.2.2")
    monkeypatch.setitem(sys.modules, "qgis", qgis_module)
    monkeypatch.setitem(sys.modules, "qgis.PyQt", pyqt_module)
    monkeypatch.setitem(sys.modules, "qgis.PyQt.QtCore", qtcore_module)
    monkeypatch.setitem(sys.modules, "qgis.core", core_module)
    inference_name = 'labeling_tool.main.inference_config'
    runtime_name = 'labeling_tool.qgis_support.process_runtime'
    sys.modules.pop(inference_name, None)
    sys.modules.pop(runtime_name, None)
    module = importlib.import_module(inference_name)
    sys.modules.pop(inference_name, None)
    sys.modules.pop(runtime_name, None)
    return module


def _manager(module, process, report_path, check_id="current", generation=4):
    manager = module.InferenceConfigManager.__new__(
        module.InferenceConfigManager
    )
    manager._process = process
    manager._stdout = bytearray()
    manager._stderr = bytearray()
    manager._scripts_dir = str(report_path.parent)
    manager._owns_process_group = True
    manager._generation = generation
    manager._check_id = check_id
    manager._report_path = str(report_path)
    manager._started_at = "2026-09-04T01:00:00+00:00"
    manager._process_error = ""
    manager.last_report = None
    manager.report_ready = _Signal()
    return manager


def _report(check_id, status="ready"):
    return {
        "schema_version": 1,
        "status": status,
        "check_id": check_id,
        "config_fingerprint": "abc",
        "effective": {},
        "checks": [],
    }


def test_report_path_is_scoped_to_the_output_workspace(tmp_path):
    assert environment_report_path(str(tmp_path)) == str(
        tmp_path / "cache" / "environment_check" / "latest.json"
    )
    assert environment_report_path("") == ""


def test_stdout_report_wins_over_stale_disk_report(tmp_path):
    path = tmp_path / "latest.json"
    persist_environment_report(str(path), _report("old", "error"))

    result, source = load_environment_report(
        "Conda notice\n" + json.dumps(_report("new")),
        str(path),
        "new",
    )

    assert result["status"] == "ready"
    assert result["check_id"] == "new"
    assert source == "stdout"


def test_current_disk_report_recovers_truncated_qprocess_stdout(tmp_path):
    path = tmp_path / "latest.json"
    persist_environment_report(str(path), _report("new"))

    result, source = load_environment_report(
        '{"schema_version":1',
        str(path),
        "new",
    )

    assert result == _report("new")
    assert source == "report_file"


def test_stale_disk_report_is_not_reused_for_a_new_check(tmp_path):
    path = tmp_path / "latest.json"
    persist_environment_report(str(path), _report("old"))

    result, source = load_environment_report("", str(path), "new")

    assert result is None
    assert source == ""


def test_report_writer_refuses_a_symlink_target(tmp_path):
    actual = tmp_path / "actual.json"
    actual.write_text("unchanged", encoding="utf-8")
    link = tmp_path / "latest.json"
    link.symlink_to(actual)

    with pytest.raises(OSError, match="symlink"):
        persist_environment_report(str(link), _report("new"))

    assert actual.read_text(encoding="utf-8") == "unchanged"


def test_delayed_old_process_signal_cannot_replace_active_check(
    monkeypatch,
    tmp_path,
):
    module = _load_inference_config(monkeypatch)
    current = _FinishedProcess()
    old = _FinishedProcess(
        stdout=(json.dumps(_report("old", "error")) + "\n").encode()
    )
    manager = _manager(module, current, tmp_path / "latest.json")

    manager._on_finished(old, manager._generation - 1, 2, "CrashExit")

    assert manager._process is current
    assert manager.last_report is None
    assert manager.report_ready.values == []
    assert old.deleted == 0


def test_qprocess_read_error_waits_for_final_json_and_keeps_valid_result(
    monkeypatch,
    tmp_path,
):
    module = _load_inference_config(monkeypatch)
    process = _FinishedProcess(
        stdout=(json.dumps(_report("current")) + "\n").encode(),
        error="temporary channel error",
    )
    path = tmp_path / "latest.json"
    manager = _manager(module, process, path)

    manager._on_process_error(
        process,
        manager._generation,
        module.QProcess.ProcessError.ReadError,
    )

    assert manager._process is process
    assert manager.report_ready.values == []

    manager._on_finished(process, manager._generation, 0, "NormalExit")

    assert manager._process is None
    assert manager.last_report["status"] == "ready"
    assert manager.last_report["process"]["report_source"] == "stdout"
    assert "ReadError" in manager.last_report["process"]["error"]
    assert "temporary channel error" in manager.last_report["process"]["error"]
    assert json.loads(path.read_text(encoding="utf-8"))["check_id"] == "current"
    assert len(manager.report_ready.values) == 1
