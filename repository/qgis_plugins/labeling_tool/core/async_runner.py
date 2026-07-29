"""Non-blocking multi-stream semantic inference orchestration."""

from __future__ import annotations

import datetime
import json
import logging
import os
import shlex
import shutil
import signal
import time
from pathlib import Path

from qgis.PyQt.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal
from qgis.core import QgsVectorFileWriter, QgsVectorLayer, QgsProject

from . import difference_filter
from .process_compat import configure_process, process_is_running
from .layer_names import LAYER_NAMES
from .pipeline_plan import build_pipeline_steps
from .result_catalog import (
    create_result_catalog,
    load_result_catalog,
    record_stream_outputs,
    stream_by_id,
    update_stream,
    valid_ready_stream_ids,
)
from .run_spec import atomic_write_json, load_run_spec
from .qgis_writer import write_vector_layer


logger = logging.getLogger("labeling_tool.async_runner")

PIPELINE_SCRIPTS = (
    "run_semantic_batch.sh",
    "run_fusion.sh",
    "run_mosaic.sh",
    "run_polygonize.sh",
    "run_subpixel_vectorize.sh",
)

STAGE_NAMES = {
    "running_models": "多模型推理",
    "building_model_streams": "模型结果构建",
    "running_fusion": "模型融合",
    "building_fusion_stream": "融合结果构建",
    "regularizing_model_streams": "模型亚像元共享边",
    "regularizing_fusion_stream": "Fusion 亚像元共享边",
    "applying_difference": "已确认区域差值",
}


def _tail(text: str, lines: int = 30, max_chars: int = 10000) -> str:
    value = "\n".join((text or "").splitlines()[-lines:])
    return value[-max_chars:]


def _failure_reason(stdout: str, stderr: str) -> str:
    lines = [line.strip() for line in f"{stdout}\n{stderr}".splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            event = None
        if isinstance(event, dict) and event.get("error"):
            return str(event["error"])[:1600]
        if "ERROR:" in line or "Traceback" in line or "failed" in line.lower():
            return line[:1600]
    return lines[-1][:1600] if lines else "process exited without diagnostic output"


def _cuda_visible_device(device: str) -> str | None:
    value = str(device or "").strip().lower()
    if value == "cuda":
        return "0"
    if value.startswith("cuda:"):
        return value.split(":", 1)[1]
    return None


class AsyncInferenceRunner(QObject):
    log_line = pyqtSignal(str, str)
    step_started = pyqtSignal(str)
    step_finished = pyqtSignal(str, int, dict)
    pipeline_progress = pyqtSignal(int, int, str)
    stage_progress = pyqtSignal(object)
    stream_progress = pyqtSignal(object)
    pipeline_finished = pyqtSignal(dict)

    def __init__(self, scripts_dir: str, parent=None):
        super().__init__(parent)
        self.scripts_dir = os.path.abspath(os.path.expanduser(scripts_dir))
        self._validate_scripts()
        self._proc: QProcess | None = None
        self._owns_process_group = False
        self._step_queue: list[dict] = []
        self._current_step: dict | None = None
        self._stdout = bytearray()
        self._stderr = bytearray()
        self._stdout_pending = bytearray()
        self._stderr_pending = bytearray()
        self._step_started_at = 0.0
        self._run_started_at = 0.0
        self._stopped = False
        self._finished = True
        self._spec: dict = {}
        self._catalog: dict = {}
        self._accepted_layer = None
        self._resume = False
        self._failed_streams: set[str] = set()
        self._step_records: list[dict] = []
        self._pipeline_log_path: Path | None = None

    @property
    def is_running(self) -> bool:
        return not self._finished

    def run_from_spec(self, run_spec_path: str, *, accepted_layer=None, resume=False):
        """Start a new or resumed semantic job and return immediately."""
        if self.is_running:
            raise RuntimeError("an inference pipeline is already running")
        self._spec = load_run_spec(run_spec_path)
        run_dir = Path(self._spec["run_dir"])
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.exists():
            if not resume:
                raise RuntimeError("run_manifest already exists; use resume=True")
            self._catalog = load_result_catalog(manifest_path)
            if Path(self._catalog.get("run_spec") or "").resolve() != Path(run_spec_path).resolve():
                raise RuntimeError("run manifest belongs to a different run_spec")
        else:
            self._catalog = create_result_catalog(self._spec)

        ready = set(valid_ready_stream_ids(self._catalog)) if resume else set()
        for stream in self._catalog.get("streams") or []:
            stream_id = str(stream["stream_id"])
            if stream.get("status") == "ready" and stream_id not in ready:
                stream["status"] = "pending"
                stream["error"] = "stored outputs failed checksum validation; rebuilding"
        self._step_queue = build_pipeline_steps(
            self._spec,
            self._catalog,
            resume=bool(resume),
            ready_stream_ids=sorted(ready),
        )
        self._accepted_layer = accepted_layer
        self._resume = bool(resume)
        self._failed_streams = set()
        self._step_records = []
        self._stopped = False
        self._finished = False
        self._run_started_at = time.time()
        self._pipeline_log_path = run_dir / "logs" / "pipeline.jsonl"
        self._append_event(
            "pipeline_started",
            run_id=self._spec["run_id"],
            resume=self._resume,
            ready_stream_ids=sorted(ready),
            step_count=len(self._step_queue),
        )
        self._emit_log("system", f"[run] {self._spec['run_id']}，结果目录: {run_dir}")
        QTimer.singleShot(0, self._start_next_step)

    def resume(self, run_spec_path: str, *, accepted_layer=None):
        self.run_from_spec(run_spec_path, accepted_layer=accepted_layer, resume=True)

    def retry_failed(self, run_spec_path: str, *, accepted_layer=None):
        # Batch runtimes validate and reuse every complete tile, so resume only
        # executes missing or failed work while retaining verified outputs.
        self.run_from_spec(run_spec_path, accepted_layer=accepted_layer, resume=True)

    def stop(self):
        if self._finished:
            return
        self._stopped = True
        self._emit_log("system", "[stop] 正在停止当前推理子进程组")
        proc = self._proc
        self._proc = None
        if proc is not None:
            proc.blockSignals(True)
            if process_is_running(proc):
                pid = int(proc.processId())
                if pid > 0 and self._owns_process_group:
                    try:
                        os.killpg(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    except OSError as exc:
                        logger.warning("cannot terminate process group %s: %s", pid, exc)
                        proc.terminate()
                else:
                    proc.terminate()
                if not proc.waitForFinished(4000):
                    if pid > 0 and self._owns_process_group:
                        try:
                            os.killpg(pid, signal.SIGKILL)
                        except (ProcessLookupError, OSError):
                            proc.kill()
                    else:
                        proc.kill()
                    proc.waitForFinished(4000)
            proc.deleteLater()
        self._owns_process_group = False
        if self._current_step:
            stream_id = self._current_step.get("stream_id")
            if stream_id:
                update_stream(self._catalog, stream_id, status="stopped", error="stopped by user")
        self._finalize()

    def cleanup(self):
        self.stop()

    def _validate_scripts(self):
        missing = [name for name in PIPELINE_SCRIPTS if not os.path.isfile(os.path.join(self.scripts_dir, name))]
        if missing:
            raise FileNotFoundError("缺少推理脚本: " + ", ".join(missing))

    def _dependencies_failed(self, step: dict) -> list[str]:
        required = []
        if step.get("requires_stream"):
            required.append(step["requires_stream"])
        required.extend(step.get("requires_streams") or [])
        return [stream_id for stream_id in required if stream_id in self._failed_streams]

    def _start_next_step(self):
        if self._stopped:
            self._finalize()
            return
        while self._step_queue:
            step = self._step_queue.pop(0)
            blocked = self._dependencies_failed(step)
            if blocked:
                stream_id = step.get("stream_id")
                reason = "dependency failed: " + ", ".join(blocked)
                if stream_id and stream_id not in self._failed_streams:
                    self._failed_streams.add(stream_id)
                    update_stream(self._catalog, stream_id, status="failed", error=reason)
                self._append_event("step_skipped", label=step["label"], stream_id=stream_id, reason=reason)
                self.step_finished.emit(step["label"], 4, {"success": False, "skipped": True, "error": reason})
                continue
            self._begin_step(step)
            return
        self._finalize()

    def _begin_step(self, step: dict):
        self._current_step = step
        self._step_started_at = time.time()
        self.step_started.emit(step["label"])
        stream_id = step.get("stream_id")
        stage = step.get("stage") or ""
        if stream_id:
            status = {
                "running_models": "running",
                "running_fusion": "running",
                "building_model_streams": "mosaicking" if step["label"].startswith("mosaic:") else "polygonizing",
                "building_fusion_stream": "mosaicking" if step["label"].startswith("mosaic:") else "polygonizing",
                "regularizing_model_streams": "regularizing",
                "regularizing_fusion_stream": "regularizing",
                "applying_difference": "difference",
            }.get(stage, "running")
            update_stream(self._catalog, stream_id, status=status)
        self._emit_stage(stage, stream_id, 0, 0, step["label"])
        if step.get("python_action") == "difference":
            QTimer.singleShot(0, self._run_difference_step)
            return
        self._start_process(step)

    def _start_process(self, step: dict):
        script_path = os.path.join(self.scripts_dir, step["script"])
        args = [str(item) for item in step.get("args") or []]
        command = ["/bin/bash", script_path, *args]
        self._emit_log("system", "[cmd] " + shlex.join(command))
        self._append_event("step_started", label=step["label"], stream_id=step.get("stream_id"), command=command)

        self._stdout = bytearray()
        self._stderr = bytearray()
        self._stdout_pending = bytearray()
        self._stderr_pending = bytearray()
        proc = QProcess(self)
        self._owns_process_group = configure_process(
            proc, "/bin/bash", [script_path, *args]
        )
        proc.setWorkingDirectory(self.scripts_dir)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        cuda_device = _cuda_visible_device((self._spec.get("runtime") or {}).get("effective_device"))
        if cuda_device is not None:
            env.insert("CUDA_VISIBLE_DEVICES", cuda_device)
        proc.setProcessEnvironment(env)
        proc.readyReadStandardOutput.connect(self._read_stdout)
        proc.readyReadStandardError.connect(self._read_stderr)
        proc.errorOccurred.connect(self._process_error)
        proc.finished.connect(self._process_finished)
        self._proc = proc
        proc.start()

    def _read_stdout(self):
        if self._proc is None:
            return
        chunk = bytes(self._proc.readAllStandardOutput())
        self._stdout.extend(chunk)
        self._stdout_pending.extend(chunk)
        self._flush_lines(self._stdout_pending, "stdout")

    def _read_stderr(self):
        if self._proc is None:
            return
        chunk = bytes(self._proc.readAllStandardError())
        self._stderr.extend(chunk)
        self._stderr_pending.extend(chunk)
        self._flush_lines(self._stderr_pending, "stderr")

    def _flush_lines(self, buffer: bytearray, level: str, final=False):
        text = buffer.decode("utf-8", errors="replace")
        lines = text.split("\n")
        remainder = "" if final else lines.pop()
        buffer[:] = remainder.encode("utf-8")
        for line in lines:
            line = line.rstrip("\r")
            if line:
                self._emit_log(level, line)
                if level == "stdout":
                    self._handle_structured_line(line)

    def _handle_structured_line(self, line: str):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return
        if not isinstance(event, dict) or not event.get("event"):
            return
        self._append_event("runtime_event", payload=event)
        stream_id = str(event.get("stream_id") or (self._current_step or {}).get("stream_id") or "")
        current = int(event.get("current") or event.get("completed") or 0)
        total = int(event.get("total") or len(self._spec.get("tiles") or []))
        message = str(event.get("tile_id") or event.get("event"))
        self.pipeline_progress.emit(current, total, message)
        self.stream_progress.emit({
            "stream_id": stream_id,
            "event": event.get("event"),
            "current": current,
            "total": total,
            "tile_id": event.get("tile_id"),
            "error": event.get("error", ""),
        })
        stage = (self._current_step or {}).get("stage") or ""
        self._emit_stage(stage, stream_id, current, total, message)

    def _process_error(self, _error):
        if self._proc is not None:
            self._emit_log("stderr", "[QProcess] " + self._proc.errorString())

    def _process_finished(self, exit_code, _exit_status):
        if self._proc is None or self._finished:
            return
        self._read_stdout()
        self._read_stderr()
        self._flush_lines(self._stdout_pending, "stdout", final=True)
        self._flush_lines(self._stderr_pending, "stderr", final=True)
        stdout = self._stdout.decode("utf-8", errors="replace")
        stderr = self._stderr.decode("utf-8", errors="replace")
        proc = self._proc
        self._proc = None
        self._owns_process_group = False
        proc.deleteLater()
        self._complete_process_step(int(exit_code), stdout, stderr)

    def _complete_process_step(self, exit_code: int, stdout: str, stderr: str):
        step = self._current_step or {}
        label = step.get("label", "unknown")
        stream_id = step.get("stream_id")
        elapsed = round(time.time() - self._step_started_at, 3)
        error = "" if exit_code == 0 else _failure_reason(stdout, stderr)
        record = {
            "label": label,
            "stream_id": stream_id,
            "returncode": exit_code,
            "elapsed_sec": elapsed,
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
            "error": error,
        }
        self._step_records.append(record)
        self._append_event("step_finished", **record)
        ok = exit_code == 0
        if ok and label.startswith(("model_batch:", "fusion_batch:")):
            manifest = self._stream_manifest(stream_id)
            failures = int(manifest.get("failed_count", 0)) if manifest else 1
            ok = failures == 0
            if ok:
                update_stream(self._catalog, stream_id, status="tiles_ready", failure_count=0, error="")
            else:
                error = error or f"stream contains {failures} failed tile(s)"
        if not ok:
            self._failed_streams.add(stream_id)
            failures = self._stream_failure_count(stream_id)
            update_stream(self._catalog, stream_id, status="failed", failure_count=failures, error=error)
            self._emit_log("system", f"[failure] {label}: {error}")
        self.step_finished.emit(label, exit_code if ok else (exit_code or 3), {
            "success": ok,
            "error": error,
            "stream_id": stream_id,
            "elapsed_sec": elapsed,
        })
        self._current_step = None
        QTimer.singleShot(0, self._start_next_step)

    def _stream_manifest(self, stream_id: str) -> dict:
        path = Path(stream_by_id(self._catalog, stream_id)["paths"]["stream_manifest"])
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return {}

    def _stream_failure_count(self, stream_id: str) -> int:
        return int(self._stream_manifest(stream_id).get("failed_count", 1))

    def _run_difference_step(self):
        step = self._current_step or {}
        stream_id = step.get("stream_id")
        started = time.time()
        try:
            stream = stream_by_id(self._catalog, stream_id)
            source_path = stream["paths"]["semantic_polygons"]
            source = QgsVectorLayer(
                f"{source_path}|layername={LAYER_NAMES.SEMANTIC}",
                LAYER_NAMES.SEMANTIC,
                "ogr",
            )
            if not source.isValid():
                raise RuntimeError(f"cannot open semantic polygons: {source_path}")
            use_difference = (
                bool(self._spec.get("skip_accepted"))
                and self._accepted_layer is not None
                and self._accepted_layer.isValid()
                and self._accepted_layer.featureCount() > 0
            )
            if use_difference:
                result = difference_filter.filter_difference(
                    source,
                    self._accepted_layer,
                    target_crs=source.crs().authid(),
                )
                output = stream["paths"]["difference_polygons"]
                options = QgsVectorFileWriter.SaveVectorOptions()
                options.driverName = "GPKG"
                options.layerName = LAYER_NAMES.CANDIDATES
                options.actionOnExistingFile = (
                    QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteFile
                )
                error, message = write_vector_layer(result, output, options)
                if error != QgsVectorFileWriter.WriterError.NoError:
                    raise RuntimeError(f"difference output failed: {message}")
                stream["review_polygons"] = output
                stream["review_layer_name"] = LAYER_NAMES.CANDIDATES
            else:
                stream["review_polygons"] = source_path
                stream["review_layer_name"] = LAYER_NAMES.SEMANTIC
            record_stream_outputs(self._catalog, stream_id)
            elapsed = round(time.time() - started, 3)
            self._append_event("difference_finished", stream_id=stream_id, elapsed_sec=elapsed)
            self.step_finished.emit(step["label"], 0, {
                "success": True, "stream_id": stream_id, "elapsed_sec": elapsed,
            })
        except Exception as exc:
            elapsed = round(time.time() - started, 3)
            error = str(exc)
            self._failed_streams.add(stream_id)
            update_stream(self._catalog, stream_id, status="failed", error=error)
            self._append_event("difference_failed", stream_id=stream_id, error=error)
            self._emit_log("system", f"[failure] {step.get('label')}: {error}")
            self.step_finished.emit(step.get("label", "difference"), 3, {
                "success": False, "stream_id": stream_id, "error": error,
                "elapsed_sec": elapsed,
            })
        self._current_step = None
        QTimer.singleShot(0, self._start_next_step)

    def _emit_stage(self, stage: str, stream_id: str, current: int, total: int, message: str):
        stages = list(STAGE_NAMES)
        index = stages.index(stage) + 1 if stage in stages else 0
        self.stage_progress.emit({
            "key": stage,
            "name": STAGE_NAMES.get(stage, stage or "处理中"),
            "index": index,
            "stage_total": len(stages),
            "stream_id": stream_id or "",
            "current": int(current),
            "total": int(total),
            "message": message,
        })

    def _emit_log(self, level: str, message: str):
        self.log_line.emit(level, message)
        self._append_event("log", level=level, message=message)

    def _append_event(self, event: str, **payload):
        if self._pipeline_log_path is None:
            return
        record = {
            "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "event": event,
            **payload,
        }
        try:
            self._pipeline_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._pipeline_log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        except OSError as exc:
            logger.error("cannot append pipeline log: %s", exc)

    def _finalize(self):
        if self._finished:
            return
        self._finished = True
        ready_streams = [item for item in self._catalog.get("streams") or [] if item.get("status") == "ready"]
        failed_streams = [item for item in self._catalog.get("streams") or [] if item.get("status") == "failed"]
        if self._stopped:
            status = "stopped"
        elif failed_streams:
            status = "failed"
        elif ready_streams and len(ready_streams) == len(self._catalog.get("streams") or []):
            status = "ready"
        else:
            status = "failed"
        self._catalog["status"] = status
        if status == "ready" and not bool((self._spec.get("runtime") or {}).get("keep_score_cache", False)):
            removed = []
            for stream in self._catalog.get("streams") or []:
                score_value = str((stream.get("paths") or {}).get("tile_score_dir") or "").strip()
                if score_value:
                    score_dir = Path(score_value)
                    if score_dir.is_dir():
                        shutil.rmtree(score_dir)
                        removed.append(str(score_dir))
                probability_value = str(
                    (stream.get("paths") or {}).get("probability_mosaic") or ""
                ).strip()
                if probability_value:
                    probability_path = Path(probability_value)
                    if probability_path.is_file():
                        probability_path.unlink()
                        removed.append(str(probability_path))
            self._catalog["score_cache_removed"] = removed
        self._catalog["updated_at"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        run_dir = Path(self._spec["run_dir"])
        atomic_write_json(run_dir / "run_manifest.json", self._catalog)
        failures = [{
            "stream_id": item["stream_id"],
            "failure_count": item.get("failure_count", 0),
            "error": item.get("error", ""),
        } for item in failed_streams]
        atomic_write_json(run_dir / "logs" / "failures.json", {"failures": failures})
        result = {
            "run_id": self._spec.get("run_id"),
            "run_dir": str(run_dir),
            "run_spec": str(run_dir / "run_spec.json"),
            "run_manifest": str(run_dir / "run_manifest.json"),
            "run_report": str(run_dir / "logs" / "run_report.json"),
            "success": status == "ready",
            "stopped": self._stopped,
            "status": status,
            "streams": self._catalog.get("streams") or [],
            "ready_streams": ready_streams,
            "failed_streams": failed_streams,
            "error": (
                "Pipeline stopped by user" if self._stopped else
                "; ".join(f"{item['stream_id']}: {item.get('error', '')}" for item in failed_streams)
            ),
        }
        report = {
            "schema_version": 1,
            "run_id": self._spec.get("run_id"),
            "status": status,
            "elapsed_sec": round(time.time() - self._run_started_at, 3),
            "resume": self._resume,
            "steps": self._step_records,
            "streams": self._catalog.get("streams") or [],
            "failures": failures,
        }
        atomic_write_json(result["run_report"], report)
        self._append_event("pipeline_finished", status=status, failures=failures)
        self._current_step = None
        self._step_queue = []
        self.pipeline_finished.emit(result)
