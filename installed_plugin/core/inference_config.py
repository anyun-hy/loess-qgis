import hashlib
import json
import os
import signal
import sys

from qgis.PyQt.QtCore import (
    PYQT_VERSION_STR,
    QT_VERSION_STR,
    QObject,
    QProcess,
    QProcessEnvironment,
    pyqtSignal,
)
from qgis.core import Qgis

from .process_compat import configure_linux_process


PIPELINE_FILES = (
    "run_work_package.sh",
    "run_finalize_partition_rasters.sh",
    "run_unit_fit.sh",
    "run_assemble_stream.sh",
    "run_scale_acceptance.sh",
    "run_sam3_interactive_worker.sh",
)
REQUIRED_FILES = PIPELINE_FILES + (
    "config.sh",
    "config.yaml",
    "deployment_config.py",
    "semantic_batch.py",
    "torchscript_runtime.py",
    "work_package_runtime.py",
    "partition_mosaic.py",
    "incremental_fusion.py",
    "finalize_partition_rasters.py",
    "assemble_stream.py",
    "scale_acceptance.py",
    "runtime_metrics.py",
    "rasterio_compat.py",
    "difference_runtime.py",
    "accepted_score.py",
    "boundary_fitting/__init__.py",
    "boundary_fitting/unit_runtime.py",
    "polyline_smoother.py",
    "common_boundary_smoother.py",
    "run_polyline_smooth.sh",
    "sam3_interactive_worker.py",
    "check_environment.py",
    "environment-linux-cu124.yml",
    "run_env_check.sh",
)
FINGERPRINT_FILES = (
    "config.sh",
    "config.yaml",
    "_device.py",
    "deployment_config.py",
    "check_environment.py",
    "environment-linux-cu124.yml",
    "semantic_batch.py",
    "torchscript_runtime.py",
    "work_package_runtime.py",
    "partition_mosaic.py",
    "incremental_fusion.py",
    "finalize_partition_rasters.py",
    "assemble_stream.py",
    "scale_acceptance.py",
    "runtime_metrics.py",
    "rasterio_compat.py",
    "difference_runtime.py",
    "accepted_score.py",
    "boundary_fitting/__init__.py",
    "boundary_fitting/unit_runtime.py",
    "polyline_smoother.py",
    "common_boundary_smoother.py",
    "sam3_interactive_worker.py",
    "sam3_refine.py",
    "run_polyline_smooth.sh",
) + PIPELINE_FILES


def config_fingerprint(scripts_dir):
    digest = hashlib.sha256()
    for name in FINGERPRINT_FILES:
        path = os.path.join(scripts_dir, name)
        digest.update(name.encode("utf-8"))
        if os.path.isfile(path):
            with open(path, "rb") as handle:
                digest.update(handle.read())
        else:
            digest.update(b"<missing>")
    return "sha256:" + digest.hexdigest()


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

    checks = []
    for name in REQUIRED_FILES:
        file_path = os.path.join(path, name)
        exists = os.path.isfile(file_path)
        executable = not name.endswith(".sh") or os.access(file_path, os.X_OK)
        ok = exists and executable
        message = "文件存在"
        if not exists:
            message = "文件缺失"
        elif not executable:
            message = "脚本没有执行权限"
        checks.append({
            "id": "file_" + name.replace(".", "_"),
            "status": "ready" if ok else "error",
            "value": name,
            "source": path,
            "message": message,
            "fix": f"修复 {file_path}",
        })

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
        self.last_report = None

    def start_check(self, scripts_dir, output_dir=""):
        self.cancel()
        self._scripts_dir = os.path.abspath(os.path.expanduser(str(scripts_dir or "").strip()))
        static_report = static_check(self._scripts_dir)
        if static_report["status"] == "error":
            static_report["scripts_dir"] = self._scripts_dir
            self.last_report = static_report
            self.report_ready.emit(static_report)
            return

        self._stdout = bytearray()
        self._stderr = bytearray()
        self._process = QProcess(self)
        arguments = [os.path.join(self._scripts_dir, "run_env_check.sh")]
        if output_dir:
            arguments.append(output_dir)
        self._owns_process_group = configure_linux_process(
            self._process, "/bin/bash", arguments
        )
        self._process.setWorkingDirectory(self._scripts_dir)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("LOESS_QGIS_VERSION", Qgis.QGIS_VERSION)
        environment.insert("LOESS_QGIS_PYTHON_VERSION", sys.version.split()[0])
        environment.insert("LOESS_QGIS_PYTHON_EXECUTABLE", sys.executable)
        environment.insert("LOESS_PYQT_VERSION", PYQT_VERSION_STR)
        environment.insert("LOESS_QT_VERSION", QT_VERSION_STR)
        self._process.setProcessEnvironment(environment)
        self._process.readyReadStandardOutput.connect(self._read_stdout)
        self._process.readyReadStandardError.connect(self._read_stderr)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_process_error)
        self.check_started.emit()
        self._process.start()

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
        if process is not None:
            process.blockSignals(True)
            if process.state() != QProcess.NotRunning:
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

    def _read_stdout(self):
        if self._process is not None:
            self._stdout.extend(bytes(self._process.readAllStandardOutput()))

    def _read_stderr(self):
        if self._process is not None:
            self._stderr.extend(bytes(self._process.readAllStandardError()))

    def _on_finished(self, exit_code, _exit_status):
        if self._process is None:
            return
        self._read_stdout()
        self._read_stderr()
        stdout = self._stdout.decode("utf-8", errors="replace").strip()
        stderr = self._stderr.decode("utf-8", errors="replace").strip()
        report = None
        if stdout:
            lines = [line.strip() for line in stdout.splitlines() if line.strip()]
            for line in reversed(lines):
                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict) and candidate.get("schema_version") == 1:
                    report = candidate
                    break

        if report is None:
            message = stderr or stdout or f"环境检查进程退出，返回码 {exit_code}"
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

        self.last_report = report
        process = self._process
        self._process = None
        if process is not None:
            process.deleteLater()
        self._owns_process_group = False
        self.report_ready.emit(report)

    def _on_process_error(self, _error):
        process = self._process
        if (
            process is None
            or process.state() != QProcess.NotRunning
        ):
            return
        message = process.errorString() or "无法启动环境检查进程"
        report = _report("error", [{
            "id": "environment_process",
            "status": "error",
            "value": "启动失败",
            "source": "run_env_check.sh",
            "message": message,
            "fix": "检查脚本执行权限、config.sh 的 CONDA_EXE 和 Conda 安装",
        }], config_fingerprint(self._scripts_dir), stderr=message)
        report["scripts_dir"] = self._scripts_dir
        self.last_report = report
        self._process = None
        self._owns_process_group = False
        process.deleteLater()
        self.report_ready.emit(report)

    def cleanup(self):
        self.cancel()
