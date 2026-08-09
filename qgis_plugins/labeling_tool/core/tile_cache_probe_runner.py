"""Asynchronous QGIS-side controller for the Conda Tile cache probe."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import secrets
import shlex
import shutil
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
        self._generation = 0
        self._expected = {}
        self._probe_dir = None

    @property
    def is_running(self):
        return bool(self._process is not None and process_is_running(self._process))

    def start(self, *, raster_path, output_root, tile):
        if self._process is not None:
            raise RuntimeError("Tile cache probe is already running")
        script = os.path.join(self.scripts_dir, "run_tile_cache_probe.sh")
        if not os.path.isfile(script):
            raise FileNotFoundError(script)
        source_path = os.path.realpath(
            os.path.abspath(os.path.expanduser(str(raster_path)))
        )
        configured_workspace = os.path.abspath(
            os.path.expanduser(str(output_root))
        )
        workspace = os.path.realpath(configured_workspace)
        tile_value = dict(tile)
        token = secrets.token_hex(16)
        probe_dir = os.path.join(workspace, f".loess-tile-cache-probe-{token}")
        arguments = [
            script,
            "--raster",
            source_path,
            "--output-root",
            configured_workspace,
            "--tile-json",
            json.dumps(tile_value, ensure_ascii=False, separators=(",", ":")),
            "--probe-token",
            token,
        ]
        process = QProcess(self)
        self._owns_process_group = configure_process(
            process, "/bin/bash", arguments
        )
        process.setWorkingDirectory(self.scripts_dir)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(environment)
        self._generation += 1
        generation = self._generation
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
        self._stdout = bytearray()
        self._stderr = bytearray()
        self._expected = {
            "probe_token": token,
            "measurement_workspace": workspace,
            "sample_artifact_directory": probe_dir,
            "sample_source_path": source_path,
            "sample_tile_id": str(tile_value.get("tile_id") or ""),
            "sample_row": int(tile_value.get("row_no", tile_value.get("row", 0))),
            "sample_col": int(tile_value.get("col_no", tile_value.get("col", 0))),
            "sample_bounds": {
                key: float((tile_value.get("bounds") or {})[key])
                for key in ("xmin", "ymin", "xmax", "ymax")
            },
        }
        self._probe_dir = probe_dir
        self._process = process
        self.log_line.emit(
            "system", "[cmd] " + shlex.join(["/bin/bash", *arguments])
        )
        process.start()

    def _is_active(self, process, generation):
        return process is self._process and generation == self._generation

    def _read_stdout(self, process, generation):
        if self._is_active(process, generation):
            self._stdout.extend(bytes(process.readAllStandardOutput()))

    def _read_stderr(self, process, generation):
        if self._is_active(process, generation):
            self._stderr.extend(bytes(process.readAllStandardError()))

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
    def _validate_report(report, expected=None):
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
        source_window = report["sample_source_window"]
        try:
            x0 = int(source_window["x0"])
            y0 = int(source_window["y0"])
            x1 = int(source_window["x1"])
            y1 = int(source_window["y1"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("探针返回的源像素窗口无效") from error
        if x1 - x0 != 512 or y1 - y0 != 512:
            raise ValueError("探针返回的源像素窗口不是精确 512x512")
        materialized_tile_bytes = int(report.get("materialized_tile_bytes") or 0)
        metadata_bytes = int(report.get("metadata_bytes") or 0)
        cache_bytes = int(report.get("materialized_cache_bytes") or 0)
        if materialized_tile_bytes <= 0 or metadata_bytes <= 0:
            raise ValueError("探针未返回 Tile 与元数据的实际字节数")
        if materialized_tile_bytes + metadata_bytes != cache_bytes:
            raise ValueError("探针返回的缓存字节数不一致")
        if not str(report.get("measurement_method_version") or ""):
            raise ValueError("探针未返回正式 Tile 物化版本")
        expected = dict(expected or {})
        for key in (
            "probe_token",
            "measurement_workspace",
            "sample_artifact_directory",
            "sample_source_path",
            "sample_tile_id",
        ):
            if key in expected and str(report.get(key) or "") != str(expected[key]):
                raise ValueError(f"探针结果与请求的 {key} 不一致")
        for key in ("sample_row", "sample_col"):
            if key in expected and int(report.get(key, -1)) != int(expected[key]):
                raise ValueError(f"探针结果与请求的 {key} 不一致")
        expected_bounds = expected.get("sample_bounds")
        actual_bounds = report.get("sample_bounds")
        if expected_bounds is not None:
            if not isinstance(actual_bounds, dict):
                raise ValueError("探针未返回真实 Tile 范围")
            for key in ("xmin", "ymin", "xmax", "ymax"):
                try:
                    matches = math.isclose(
                        float(actual_bounds[key]),
                        float(expected_bounds[key]),
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError("探针返回的真实 Tile 范围无效") from error
                if not matches:
                    raise ValueError(f"探针结果与请求的 {key} 不一致")
        return dict(report)

    def _detach(self, process, generation):
        if not self._is_active(process, generation):
            return False
        self._process = None
        self._owns_process_group = False
        return True

    def _cleanup_probe_directory(self):
        value = self._probe_dir
        self._probe_dir = None
        if not value:
            return
        candidate = Path(value)
        expected_workspace = Path(str(self._expected.get("measurement_workspace") or ""))
        token = str(self._expected.get("probe_token") or "")
        expected_name = f".loess-tile-cache-probe-{token}"
        if candidate.parent != expected_workspace or candidate.name != expected_name:
            return
        try:
            if candidate.is_symlink():
                candidate.unlink()
            elif candidate.exists():
                shutil.rmtree(candidate)
        except OSError:
            pass

    def _clear_request(self):
        self._cleanup_probe_directory()
        self._expected = {}

    def _on_finished(self, process, generation, exit_code, _exit_status):
        if not self._is_active(process, generation):
            return
        self._read_stdout(process, generation)
        self._read_stderr(process, generation)
        stdout = self._stdout.decode("utf-8", errors="replace").strip()
        stderr = self._stderr.decode("utf-8", errors="replace").strip()
        report = self._parse_report(stdout)
        expected = dict(self._expected)
        self._detach(process, generation)
        process.deleteLater()
        self._clear_request()
        try:
            if int(exit_code) != 0:
                raise ValueError(
                    str((report or {}).get("message") or f"探针退出码 {exit_code}")
                )
            validated = self._validate_report(report, expected)
        except (TypeError, ValueError) as error:
            message = str(error) or stderr or stdout or f"探针退出码 {exit_code}"
            if stderr and stderr not in message:
                message = f"{message}\n{stderr}"
            self.failed.emit(message)
            return
        self.succeeded.emit(validated)

    def _on_process_error(self, process, generation, _error):
        if not self._is_active(process, generation) or process_is_running(process):
            return
        message = process.errorString() or "无法启动 Tile 缓存探针"
        self._detach(process, generation)
        process.deleteLater()
        self._clear_request()
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
        self._clear_request()

    def cleanup(self):
        self.cancel()
