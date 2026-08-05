"""Asynchronous QGIS-side controller for the Conda Tile cache probe."""

from __future__ import annotations

import json
import os
import shlex
import signal

from qgis.PyQt.QtCore import QObject, QProcess, QProcessEnvironment, pyqtSignal

from .process_compat import configure_process, process_is_running


class TileCacheProbeRunner(QObject):
    succeeded = pyqtSignal(dict)
    failed = pyqtSignal(str)
    log_line = pyqtSignal(str, str)

    def __init__(self, scripts_dir, parent=None):
        super().__init__(parent)
        self.scripts_dir = os.path.abspath(str(scripts_dir))
        self._process = None
        self._owns_process_group = False
        self._stdout = bytearray()
        self._stderr = bytearray()

    @property
    def is_running(self):
        return bool(self._process is not None and process_is_running(self._process))

    def start(self, *, raster_path, output_root, tile):
        if self._process is not None:
            raise RuntimeError("Tile cache probe is already running")
        script = os.path.join(self.scripts_dir, "run_tile_cache_probe.sh")
        if not os.path.isfile(script):
            raise FileNotFoundError(script)
        arguments = [
            script,
            "--raster",
            os.path.abspath(os.path.expanduser(str(raster_path))),
            "--output-root",
            os.path.abspath(os.path.expanduser(str(output_root))),
            "--tile-json",
            json.dumps(dict(tile), ensure_ascii=False, separators=(",", ":")),
        ]
        process = QProcess(self)
        self._owns_process_group = configure_process(
            process, "/bin/bash", arguments
        )
        process.setWorkingDirectory(self.scripts_dir)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_process_error)
        self._stdout = bytearray()
        self._stderr = bytearray()
        self._process = process
        self.log_line.emit(
            "system", "[cmd] " + shlex.join(["/bin/bash", *arguments])
        )
        process.start()

    def _read_stdout(self):
        if self._process is not None:
            self._stdout.extend(bytes(self._process.readAllStandardOutput()))

    def _read_stderr(self):
        if self._process is not None:
            self._stderr.extend(bytes(self._process.readAllStandardError()))

    @staticmethod
    def _parse_report(stdout):
        for line in reversed([item.strip() for item in stdout.splitlines()]):
            if not line:
                continue
            try:
                report = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(report, dict)
                and report.get("schema_version") == 1
                and report.get("kind") == "tile_cache_probe"
            ):
                return report
        return None

    @staticmethod
    def _validate_report(report):
        if not isinstance(report, dict) or report.get("status") != "passed":
            raise ValueError(str((report or {}).get("message") or "探针未返回成功报告"))
        if (
            int(report.get("width") or 0) != 512
            or int(report.get("height") or 0) != 512
            or int(report.get("band_count") or 0) != 3
        ):
            raise ValueError("探针结果不是 512x512 三波段 Tile")
        if int(report.get("uncompressed_bytes") or 0) <= 0:
            raise ValueError("探针未返回压缩前字节数")
        if int(report.get("materialized_cache_bytes") or 0) <= 0:
            raise ValueError("探针未返回实际缓存字节数")
        if report.get("measurement_method") != "tile_materializer._materialize_one":
            raise ValueError("探针没有使用正式 Tile 物化路径")
        if not isinstance(report.get("sample_source_window"), dict):
            raise ValueError("探针未返回真实源像素窗口")
        return dict(report)

    def _on_finished(self, exit_code, _exit_status):
        process = self._process
        if process is None:
            return
        self._read_stdout()
        self._read_stderr()
        stdout = self._stdout.decode("utf-8", errors="replace").strip()
        stderr = self._stderr.decode("utf-8", errors="replace").strip()
        report = self._parse_report(stdout)
        self._process = None
        self._owns_process_group = False
        process.deleteLater()
        try:
            validated = self._validate_report(report)
            if int(exit_code) != 0:
                raise ValueError(
                    str(validated.get("message") or f"探针退出码 {exit_code}")
                )
        except (TypeError, ValueError) as error:
            message = str(error) or stderr or stdout or f"探针退出码 {exit_code}"
            if stderr and stderr not in message:
                message = f"{message}\n{stderr}"
            self.failed.emit(message)
            return
        self.succeeded.emit(validated)

    def _on_process_error(self, _error):
        process = self._process
        if process is None or process_is_running(process):
            return
        message = process.errorString() or "无法启动 Tile 缓存探针"
        self._process = None
        self._owns_process_group = False
        process.deleteLater()
        self.failed.emit(message)

    def cancel(self):
        process = self._process
        self._process = None
        if process is None:
            return
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
        self._owns_process_group = False
        process.deleteLater()

    def cleanup(self):
        self.cancel()
