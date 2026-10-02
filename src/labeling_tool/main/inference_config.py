import os
import signal
import sys
import uuid
from datetime import datetime, timezone

from qgis.PyQt.QtCore import (
    PYQT_VERSION_STR,
    QT_VERSION_STR,
    QObject,
    QProcess,
    QProcessEnvironment,
    pyqtSignal,
)
from qgis.core import Qgis

from labeling_tool.runs.deployment_contract import deployment_fingerprint, verify_project_runtime
from labeling_tool.main.environment_transport import (
    environment_report_path,
    load_environment_report,
    persist_environment_report,
)
from labeling_tool.qgis_support.process_runtime import configure_process, process_is_running


def config_fingerprint(scripts_dir):
    return deployment_fingerprint(scripts_dir)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _enum_name(value):
    name = getattr(value, "name", None)
    return str(name if name is not None else value)


def _report(status, checks, fingerprint="", effective=None, stderr=""):
    return {
        "schema_version": 1,
        "status": status,
        "config_fingerprint": fingerprint,
        "effective": effective or {},
        "checks": checks,
        "stderr": stderr,
    }


def static_check(scripts_dir):
    path = os.path.abspath(os.path.expanduser(str(scripts_dir or "").strip()))
    if not str(scripts_dir or "").strip():
        return _report("error", [{
            "id": "scripts_dir",
            "status": "error",
            "value": "未选择",
            "source": "QGIS 面板:脚本目录",
            "message": "请选择 inference_scripts 目录",
            "fix": "点击选择按钮指定脚本目录",
        }])
    if not os.path.isdir(path):
        return _report("error", [{
            "id": "scripts_dir",
            "status": "error",
            "value": path,
            "source": "QGIS 面板:脚本目录",
            "message": "目录不存在",
            "fix": "重新选择 inference_scripts 目录",
        }])

    checks = [verify_project_runtime(path)]
    status = "error" if any(item["status"] == "error" for item in checks) else "ready"
    return _report(status, checks, config_fingerprint(path))


class InferenceConfigManager(QObject):
    check_started = pyqtSignal()
    report_ready = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process = None
        self._stdout = bytearray()
        self._stderr = bytearray()
        self._scripts_dir = ""
        self._owns_process_group = False
        self._generation = 0
        self._check_id = ""
        self._report_path = ""
        self._started_at = ""
        self._process_error = ""
        self.last_report = None

    def start_check(self, scripts_dir, output_dir=""):
        self.cancel()
        self._generation += 1
        generation = self._generation
        self._check_id = uuid.uuid4().hex
        self._started_at = _utc_now()
        self._report_path = environment_report_path(output_dir)
        self._process_error = ""
        self._scripts_dir = os.path.abspath(os.path.expanduser(str(scripts_dir or "").strip()))
        static_report = static_check(self._scripts_dir)
        if static_report["status"] == "error":
            static_report["scripts_dir"] = self._scripts_dir
            static_report["check_id"] = self._check_id
            static_report["started_at"] = self._started_at
            static_report["finished_at"] = _utc_now()
            static_report["diagnostics_path"] = self._report_path
            static_report["process"] = {
                "state": "not_started",
                "report_source": "static_check",
            }
            self._persist_report(static_report)
            self.last_report = static_report
            self.report_ready.emit(static_report)
            return

        self._stdout = bytearray()
        self._stderr = bytearray()
        process = QProcess(self)
        self._process = process
        arguments = [os.path.join(self._scripts_dir, "run_env_check.sh")]
        if output_dir:
            arguments.append(output_dir)
            arguments.append(self._report_path)
        self._owns_process_group = configure_process(
            process, "/bin/bash", arguments
        )
        process.setWorkingDirectory(self._scripts_dir)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("LOESS_ENV_CHECK_ID", self._check_id)
        environment.insert("LOESS_ENV_CHECK_STARTED_AT", self._started_at)
        environment.insert("LOESS_QGIS_VERSION", Qgis.QGIS_VERSION)
        environment.insert("LOESS_QGIS_PYTHON_VERSION", sys.version.split()[0])
        environment.insert("LOESS_QGIS_PYTHON_EXECUTABLE", sys.executable)
        environment.insert("LOESS_PYQT_VERSION", PYQT_VERSION_STR)
        environment.insert("LOESS_QT_VERSION", QT_VERSION_STR)
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(
            lambda p=process, g=generation: self._read_stdout(p, g)
        )
        process.readyReadStandardError.connect(
            lambda p=process, g=generation: self._read_stderr(p, g)
        )
        process.finished.connect(
            lambda code, status, p=process, g=generation: self._on_finished(
                p, g, code, status
            )
        )
        process.errorOccurred.connect(
            lambda error, p=process, g=generation: self._on_process_error(
                p, g, error
            )
        )
        self.check_started.emit()
        process.start()

    def is_stale(self, scripts_dir=None):
        report = self.last_report or {}
        expected = report.get("config_fingerprint", "")
        path = os.path.abspath(os.path.expanduser(scripts_dir or self._scripts_dir))
        if report.get("scripts_dir") != path:
            return True
        if not expected or not path or not os.path.isdir(path):
            return True
        return expected != config_fingerprint(path)

    def cancel(self):
        process = self._process
        self._process = None
        self._generation += 1
        if process is not None:
            process.blockSignals(True)
            if process_is_running(process):
                pid = int(process.processId())
                if pid > 0 and self._owns_process_group:
                    try:
                        os.killpg(pid, signal.SIGTERM)
                    except (ProcessLookupError, OSError):
                        process.terminate()
                else:
                    process.terminate()
                if not process.waitForFinished(3000):
                    if pid > 0 and self._owns_process_group:
                        try:
                            os.killpg(pid, signal.SIGKILL)
                        except (ProcessLookupError, OSError):
                            process.kill()
                    else:
                        process.kill()
                    process.waitForFinished(2000)
            process.deleteLater()
        self._owns_process_group = False

    def _is_current(self, process, generation):
        return process is self._process and generation == self._generation

    def _read_stdout(self, process, generation):
        if self._is_current(process, generation):
            self._stdout.extend(bytes(process.readAllStandardOutput()))

    def _read_stderr(self, process, generation):
        if self._is_current(process, generation):
            self._stderr.extend(bytes(process.readAllStandardError()))

    def _persist_report(self, report):
        try:
            persist_environment_report(self._report_path, report)
        except OSError as error:
            report["diagnostics_persist_error"] = str(error)

    def _on_finished(self, process, generation, exit_code, exit_status):
        if not self._is_current(process, generation):
            return
        self._read_stdout(process, generation)
        self._read_stderr(process, generation)
        stdout = self._stdout.decode("utf-8", errors="replace").strip()
        stderr = self._stderr.decode("utf-8", errors="replace").strip()
        report, report_source = load_environment_report(
            stdout,
            self._report_path,
            self._check_id,
        )

        if report is None:
            message = (
                stderr
                or self._process_error
                or stdout
                or f"环境检查进程退出，返回码 {exit_code}"
            )
            report = _report("error", [{
                "id": "environment_process",
                "status": "error",
                "value": f"退出码 {exit_code}",
                "source": "run_env_check.sh",
                "message": message,
                "fix": "检查 config.sh 的 CONDA_EXE、CONDA_ENV 和 Conda 环境",
            }], config_fingerprint(self._scripts_dir), stderr=message)
        else:
            report["stderr"] = stderr
        report["scripts_dir"] = self._scripts_dir
        report["check_id"] = self._check_id
        report["started_at"] = self._started_at
        report["finished_at"] = _utc_now()
        report["diagnostics_path"] = self._report_path
        report["process"] = {
            "state": "finished",
            "exit_code": int(exit_code),
            "exit_status": _enum_name(exit_status),
            "error": self._process_error,
            "report_source": report_source or "none",
        }
        self._persist_report(report)

        self.last_report = report
        self._process = None
        process.deleteLater()
        self._owns_process_group = False
        self.report_ready.emit(report)

    def _on_process_error(self, process, generation, error):
        if not self._is_current(process, generation):
            return
        message = process.errorString() or "环境检查进程发生未知错误"
        self._process_error = f"{_enum_name(error)}: {message}"
        if error == QProcess.ProcessError.FailedToStart:
            self._on_finished(process, generation, -1, "FailedToStart")

    def cleanup(self):
        self.cancel()
