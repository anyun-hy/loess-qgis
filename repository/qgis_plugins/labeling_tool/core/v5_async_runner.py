"""Qt orchestration for the SQLite-backed bounded v5 pipeline."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import time
import uuid
from pathlib import Path

from qgis.PyQt.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal

from .process_compat import configure_process, process_is_running
from .result_catalog import artifact_sha256
from .run_index import record_run_state
from .run_spec import atomic_write_json, sha256_file
from .run_state_db import RunStateDB


class V5AsyncInferenceRunner(QObject):
    """Run one accelerator package and a bounded CPU geometry pool."""

    log_line = pyqtSignal(str, str)
    step_started = pyqtSignal(str)
    step_finished = pyqtSignal(str, int, dict)
    pipeline_progress = pyqtSignal(int, int, str)
    stage_progress = pyqtSignal(object)
    stream_progress = pyqtSignal(object)
    pipeline_finished = pyqtSignal(dict)

    REQUIRED_SCRIPTS = (
        "run_work_package.sh",
        "run_unit_fit.sh",
        "run_finalize_partition_rasters.sh",
        "run_assemble_stream.sh",
        "run_scale_acceptance.sh",
    )

    def __init__(self, scripts_dir: str, parent=None):
        super().__init__(parent)
        self.scripts_dir = str(Path(scripts_dir).expanduser().resolve())
        missing = [
            name for name in self.REQUIRED_SCRIPTS
            if not (Path(self.scripts_dir) / name).is_file()
        ]
        if missing:
            raise FileNotFoundError("缺少 v5 推理脚本: " + ", ".join(missing))
        self._spec = {}
        self._spec_path = ""
        self._database = None
        self._running = False
        self._stopped = False
        self._phase = "idle"
        self._worker_id = f"qgis-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._processes = {}
        self._assembly_queue = []
        self._started_at = 0.0
        self._retry_failed_on_start = False
        self._scheduler = QTimer(self)
        self._scheduler.setInterval(500)
        self._scheduler.timeout.connect(self._schedule)
        self._watchdog = QTimer(self)
        self._watchdog.setInterval(15000)
        self._watchdog.timeout.connect(self._heartbeat_and_watchdog)
        self.log_line.connect(self._persist_log)

    @property
    def is_running(self):
        return self._running

    def run_from_spec(
        self,
        run_spec_path: str,
        *,
        accepted_layer=None,
        resume=False,
        retry_failed=False,
    ):
        del accepted_layer
        if self._running:
            raise RuntimeError("an inference pipeline is already running")
        self._spec_path = str(Path(run_spec_path).resolve())
        with open(self._spec_path, "r", encoding="utf-8") as handle:
            self._spec = json.load(handle)
        if self._spec.get("schema_version") != 2:
            raise RuntimeError("V5 runner requires run_spec schema 2")
        self._database = RunStateDB(self._spec["state_db"])
        if resume:
            self._database.interrupt_run_jobs(self._spec["run_id"])
        if retry_failed:
            self._database.requeue_failed_jobs(self._spec["run_id"])
        self._database.set_run_status(
            self._spec["run_id"],
            "running",
            expected=("planned", "running", "stopped", "failed", "raster_ready"),
        )
        self._record_startup_index("running")
        self._running = True
        self._stopped = False
        self._phase = "jobs"
        self._processes.clear()
        self._assembly_queue = []
        self._started_at = time.time()
        self.log_line.emit("system", f"[run-v5] {self._spec['run_id']}")
        self._scheduler.start()
        self._watchdog.start()
        QTimer.singleShot(0, self._schedule)

    def resume(self, run_spec_path: str, *, accepted_layer=None):
        self.run_from_spec(run_spec_path, accepted_layer=accepted_layer, resume=True)

    def retry_failed(self, run_spec_path: str, *, accepted_layer=None):
        self.run_from_spec(
            run_spec_path,
            accepted_layer=accepted_layer,
            resume=True,
            retry_failed=True,
        )

    def stop(self):
        if not self._running:
            return
        self._stopped = True
        self._scheduler.stop()
        self._watchdog.stop()
        entries = list(self._processes.values())
        for entry in entries:
            self._terminate_entry(entry, graceful=True)
        for entry in entries:
            job = entry["context"].get("job")
            if job:
                self._database.interrupt_job(job["job_id"], job["lease_token"])
        self._processes.clear()
        self._database.set_run_status(
            self._spec["run_id"], "stopped", expected=("running", "raster_ready")
        )
        self._finish(False, "Pipeline stopped by user")

    def cleanup(self):
        self.stop()

    def _schedule(self):
        if not self._running or self._stopped:
            return
        if self._phase != "jobs":
            return
        try:
            self._cleanup_released_artifacts()
        except RuntimeError as error:
            self._finish(False, str(error))
            return
        if self._disk_below_reserve():
            self._emit_progress("磁盘空间低于保留阈值，已暂停派发新任务")
            return

        active_jobs = [
            entry["context"].get("job") for entry in self._processes.values()
            if entry["context"].get("kind") == "job"
        ]
        package_active = any(job and job["job_type"] == "work_package" for job in active_jobs)
        unit_active = sum(1 for job in active_jobs if job and job["job_type"] == "unit_fit")
        started = False

        if not package_active:
            job = self._database.lease_next_work_package(
                self._spec["run_id"],
                self._worker_id + "-accelerator",
                max_open_frontier_units=int(
                    (self._spec.get("scaling") or {}).get(
                        "max_open_frontier_units", 64
                    )
                ),
                lease_seconds=1800,
            )
            if job:
                self._start_job(job)
                started = True

        cpu_limit = max(
            1, int((self._spec.get("scaling") or {}).get("max_cpu_partition_workers", 2))
        )
        while unit_active < cpu_limit:
            job = self._database.lease_next_job(
                self._spec["run_id"],
                self._worker_id + f"-geometry-{unit_active}",
                job_types=("unit_fit",),
                lease_seconds=300,
            )
            if not job:
                break
            self._start_job(job)
            unit_active += 1
            started = True

        if started or self._processes:
            boundary_enabled = bool(
                (self._spec.get("boundary_fitting") or {}).get("enabled", True)
            )
            geometry_stage = (
                "公共分界线拟合中" if boundary_enabled else "原始类别边界组装中"
            )
            self._emit_progress(f"有界 Work Package / {geometry_stage}")
            return

        counts = self._database.job_counts(self._spec["run_id"])
        if counts.get("failed"):
            self._finish(False, f"v5 jobs exhausted retries: {counts}")
        elif counts.get("queued") or counts.get("interrupted") or counts.get("running"):
            self._finish(False, f"v5 job graph has blocked dependencies: {counts}")
        else:
            self._phase = "finalize"
            self._start_process(
                "finalize_partition_rasters",
                "run_finalize_partition_rasters.sh",
                ["--run-spec", self._spec_path],
                {"kind": "finalize_rasters"},
            )

    def _start_job(self, job):
        if job["job_type"] == "work_package":
            self._start_process(
                f"work_package:{job['package_id']}",
                "run_work_package.sh",
                [
                    "--run-spec", self._spec_path,
                    "--package-id", job["package_id"],
                    "--device", self._spec["runtime"]["effective_device"],
                    "--resume",
                ],
                {"kind": "job", "job": job},
            )
        else:
            self._start_process(
                f"unit_fit:{job['stream_id']}:{job['unit_id']}",
                "run_unit_fit.sh",
                [
                    "--run-spec", self._spec_path,
                    "--stream-id", job["stream_id"],
                    "--unit-id", job["unit_id"],
                    "--job-id", str(job["job_id"]),
                    "--lease-token", job["lease_token"],
                ],
                {"kind": "job", "job": job},
            )

    def _start_assembly(self):
        active_assemblies = [
            entry
            for entry in self._processes.values()
            if (entry.get("context") or {}).get("kind") == "assemble"
        ]
        if active_assemblies:
            active_stream = str(
                (active_assemblies[0].get("context") or {}).get("stream_id")
                or "unknown"
            )
            self.log_line.emit(
                "system",
                "[assembly-queue] waiting for active stream: "
                + active_stream,
            )
            return
        if not self._assembly_queue:
            self._phase = "acceptance"
            self._start_process(
                "scale_acceptance",
                "run_scale_acceptance.sh",
                ["--run-spec", self._spec_path],
                {"kind": "scale_acceptance"},
            )
            return
        stream = self._assembly_queue.pop(0)
        self._start_process(
            f"assemble_stream:{stream['stream_id']}",
            "run_assemble_stream.sh",
            ["--run-spec", self._spec_path, "--stream-id", stream["stream_id"]],
            {"kind": "assemble", "stream_id": stream["stream_id"]},
        )

    def _start_process(self, label, script, arguments, context):
        token = uuid.uuid4().hex
        path = str(Path(self.scripts_dir) / script)
        process = QProcess(self)
        owns_process_group = configure_process(
            process, "/bin/bash", [path, *arguments]
        )
        process.setWorkingDirectory(self.scripts_dir)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(environment)
        entry = {
            "token": token,
            "process": process,
            "context": {**context, "label": label, "started_at": time.time()},
            "stdout": bytearray(),
            "stderr": bytearray(),
            "forced_error": "",
            "owns_process_group": owns_process_group,
        }
        self._processes[token] = entry
        process.readyReadStandardOutput.connect(lambda t=token: self._read(t, "stdout"))
        process.readyReadStandardError.connect(lambda t=token: self._read(t, "stderr"))
        process.finished.connect(
            lambda code, status, t=token: self._process_finished(t, code, status)
        )
        self.step_started.emit(label)
        self.log_line.emit("system", "[cmd] " + shlex.join(["/bin/bash", path, *arguments]))
        process.start()

    def _read(self, token, level):
        entry = self._processes.get(token)
        if not entry:
            return
        process = entry["process"]
        chunk = (
            process.readAllStandardOutput()
            if level == "stdout" else process.readAllStandardError()
        )
        entry[level].extend(bytes(chunk))
        self._flush(entry, level)

    def _flush(self, entry, level, final=False):
        buffer = entry[level]
        decoded = buffer.decode("utf-8", errors="replace")
        lines = decoded.split("\n")
        remainder = "" if final else lines.pop()
        buffer[:] = remainder.encode("utf-8")
        for line in lines:
            line = line.rstrip("\r")
            if not line:
                continue
            self.log_line.emit(level, line)
            if level == "stdout":
                self._structured(entry, line)

    def _structured(self, entry, line):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return
        if not isinstance(event, dict) or not event.get("event"):
            return
        self.stream_progress.emit(event)
        current = int(event.get("current") or 0)
        total = int(event.get("total") or 0)
        job = entry["context"].get("job")
        if job:
            self._database.heartbeat(
                job["job_id"], job["lease_token"],
                current=current, total=total, lease_seconds=300,
            )
        self.pipeline_progress.emit(
            current,
            total,
            str(event.get("unit_id") or event.get("tile_id") or event["event"]),
        )

    def _process_finished(self, token, exit_code, _exit_status):
        entry = self._processes.get(token)
        if not entry or not self._running:
            return
        self._read(token, "stdout")
        self._read(token, "stderr")
        self._flush(entry, "stdout", final=True)
        self._flush(entry, "stderr", final=True)
        self._processes.pop(token, None)
        entry["process"].deleteLater()
        context = entry["context"]
        label = context["label"]
        success = int(exit_code) == 0 and not entry["forced_error"]
        error = entry["forced_error"] or ("" if success else f"{label} failed (rc={int(exit_code)})")

        if context.get("kind") == "job":
            job = context["job"]
            current = self._database.get_job(job["job_id"])
            if current and current["status"] == "running":
                self._database.finish_job(
                    job["job_id"],
                    job["lease_token"],
                    status="ready" if success else "failed",
                    error=error,
                )
            retried = False
            if not success and int(entry.get("timeout_count", 0)) < 2:
                retried = self._database.requeue_failed_job(job["job_id"])
                if retried:
                    self.log_line.emit("system", f"[retry] {label}")
            self.step_finished.emit(
                label,
                int(exit_code),
                {
                    "success": success or retried,
                    "error": error,
                    "stream_id": job.get("stream_id") or "",
                },
            )
            self._emit_progress(label)
            QTimer.singleShot(0, self._schedule)
            return

        self.step_finished.emit(label, int(exit_code), {"success": success, "error": error})
        if not success:
            self._finish(False, error)
        elif context.get("kind") == "finalize_rasters":
            self._phase = "assembly"
            self._assembly_queue = list(self._spec["streams"])
            QTimer.singleShot(0, self._start_assembly)
        elif context.get("kind") == "assemble":
            QTimer.singleShot(0, self._start_assembly)
        elif context.get("kind") == "scale_acceptance":
            self._finish(True, "")

    def _heartbeat_and_watchdog(self):
        if not self._running:
            return
        timeout = float(
            (self._spec.get("scaling") or {}).get("max_partition_runtime_sec", 900)
        )
        now = time.time()
        for entry in list(self._processes.values()):
            context = entry["context"]
            job = context.get("job")
            if job:
                current = self._database.get_job(job["job_id"])
                if current and current["status"] == "running":
                    self._database.heartbeat(
                        job["job_id"], job["lease_token"],
                        current=current["progress_current"],
                        total=current["progress_total"],
                        lease_seconds=300,
                    )
            if (
                job and job["job_type"] == "unit_fit"
                and now - context["started_at"] > timeout
                and not entry["forced_error"]
            ):
                entry["forced_error"] = f"{context['label']} timed out after {timeout:.0f}s"
                marker = (
                    Path(self._spec["run_dir"])
                    / "tmp"
                    / "failed_jobs"
                    / (
                        f"{job['stream_id'].replace(':', '_')}__"
                        f"{job['unit_id']}_force_split.json"
                    )
                )
                timeout_count = 1
                if marker.is_file():
                    try:
                        with open(marker, "r", encoding="utf-8") as handle:
                            timeout_count = int(json.load(handle).get("timeout_count", 1)) + 1
                    except (OSError, ValueError, json.JSONDecodeError):
                        timeout_count = 2
                atomic_write_json(
                    marker,
                    {
                        "run_id": self._spec["run_id"],
                        "stream_id": job["stream_id"],
                        "unit_id": job["unit_id"],
                        "timeout_count": timeout_count,
                        "next_attempt": "force_one_recursive_split" if timeout_count == 1 else "fail",
                    },
                )
                entry["timeout_count"] = timeout_count
                self.log_line.emit("stderr", entry["forced_error"])
                self._terminate_entry(entry, graceful=False)

    def _terminate_entry(self, entry, *, graceful):
        process = entry["process"]
        if not process_is_running(process):
            return
        pid = int(process.processId())
        try:
            if pid > 0 and entry.get("owns_process_group"):
                os.killpg(pid, signal.SIGTERM if graceful else signal.SIGKILL)
            elif graceful:
                process.terminate()
            else:
                process.kill()
        except (ProcessLookupError, OSError):
            process.kill()
        process.waitForFinished(2500 if graceful else 1000)

    def _disk_below_reserve(self):
        reserve = int(
            float((self._spec.get("scaling") or {}).get("min_free_disk_gb", 0))
            * 1024**3
        )
        return shutil.disk_usage(self._spec["output_root"]).free <= reserve

    def _cleanup_released_artifacts(self):
        if bool((self._spec.get("runtime") or {}).get("keep_score_cache", False)):
            return
        candidates = self._database.cleanup_candidates(
            self._spec["run_id"],
            limit=1000,
            kinds=("partition_probability",),
        )
        for candidate in candidates:
            claimed = self._database.claim_artifact_cleanup(candidate["artifact_id"])
            if claimed is None:
                continue
            path = Path(claimed["path"])
            try:
                if path.exists():
                    actual_size = path.stat().st_size
                    actual_sha = sha256_file(path)
                    if (
                        actual_size != int(claimed["byte_count"])
                        or actual_sha != str(claimed["sha256"])
                    ):
                        raise RuntimeError(
                            "temporary Artifact changed before cleanup: " + str(path)
                        )
                    path.unlink()
                else:
                    self.log_line.emit(
                        "system",
                        "[cleanup-missing] unreferenced temporary Artifact: " + str(path),
                    )
                if not self._database.finish_artifact_cleanup(
                    claimed["artifact_id"], success=True
                ):
                    raise RuntimeError(
                        "temporary Artifact cleanup state changed: " + str(path)
                    )
            except Exception:
                self._database.finish_artifact_cleanup(
                    claimed["artifact_id"], success=False
                )
                raise

    def _emit_progress(self, message):
        counts = self._database.job_counts(self._spec["run_id"])
        total = sum(counts.values())
        current = counts.get("ready", 0)
        boundary_enabled = bool(
            (self._spec.get("boundary_fitting") or {}).get("enabled", True)
        )
        stage_name = (
            "分区推理与公共分界线拟合"
            if boundary_enabled else "分区推理与原始边界组装"
        )
        self.pipeline_progress.emit(current, total, message)
        self.stage_progress.emit(
            {
                "key": "v5_jobs",
                "name": stage_name,
                "index": 1,
                "stage_total": 3,
                "current": current,
                "total": total,
                "message": message,
            }
        )

    def _persist_log(self, level, message):
        if not self._spec.get("run_dir"):
            return
        path = Path(self._spec["run_dir"]) / "logs" / "pipeline.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": time.time(),
            "level": str(level),
            "message": str(message),
        }
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
        except OSError:
            pass

    def _result_stream(self, stream):
        run_dir = Path(self._spec["run_dir"])
        root = (
            run_dir / "models" / stream["model_id"]
            if stream["kind"] == "model"
            else run_dir / "fusion" / stream["profile_id"]
        )
        paths = {
            "mask_mosaic": str(root / "mask_mosaic.vrt"),
            "confidence_mosaic": str(root / "confidence_mosaic.vrt"),
            "semantic_polygons_raw": str(root / "semantic_polygons_raw.gpkg"),
            "semantic_polygons": str(root / "semantic_polygons.gpkg"),
            "boundary_fitting_report": str(root / "boundary_fitting_report.json"),
            "fitted_edges": str(root / "fitted_edges.gpkg"),
        }
        boundary_status = "failed"
        try:
            with open(paths["boundary_fitting_report"], "r", encoding="utf-8") as handle:
                boundary_report = json.load(handle)
            if (
                boundary_report.get("status") == "passed"
                and (boundary_report.get("validation") or {}).get("passed") is True
            ):
                boundary_status = "passed"
        except (OSError, ValueError, TypeError):
            pass
        result = {
            "stream_id": stream["stream_id"],
            "kind": stream["kind"],
            "model_id": stream.get("model_id", ""),
            "fusion_profile_id": stream.get("profile_id", ""),
            "version": stream.get("version", ""),
            "status": "ready",
            "boundary_smoothing_enabled": bool(
                (self._spec.get("boundary_fitting") or {}).get("enabled", True)
            ),
            "boundary_fitting_status": boundary_status,
            "paths": paths,
            "output_sha256": {
                key: artifact_sha256(path) for key, path in paths.items()
            },
        }
        candidate_path = root / "semantic_candidates.gpkg"
        if candidate_path.is_file():
            result["review_polygons"] = str(candidate_path)
            result["review_layer_name"] = "semantic_candidates"
            result["output_sha256"]["review_polygons"] = sha256_file(candidate_path)
        return result

    def _finish(self, success, error):
        if not self._running:
            return
        self._scheduler.stop()
        self._watchdog.stop()
        self._running = False
        ready_streams = (
            [self._result_stream(stream) for stream in self._spec.get("streams", [])]
            if success else []
        )
        failed_streams = [] if success else list(self._spec.get("streams", []))
        result = {
            "schema_version": 2,
            "run_id": self._spec["run_id"],
            "run_spec": self._spec_path,
            "run_spec_sha256": sha256_file(self._spec_path),
            "run_dir": self._spec["run_dir"],
            "success": bool(success),
            "status": "ready" if success else "stopped" if self._stopped else "failed",
            "error": str(error),
            "ready_streams": ready_streams,
            "failed_streams": failed_streams,
            "streams": ready_streams if success else failed_streams,
            "elapsed_sec": round(time.time() - self._started_at, 3),
        }
        run_dir = Path(self._spec["run_dir"])
        scale_report = run_dir / "logs" / "scale_acceptance_report.json"
        if scale_report.is_file():
            result["scale_acceptance_report"] = str(scale_report)
            result["scale_acceptance_report_sha256"] = sha256_file(scale_report)
        atomic_write_json(run_dir / "run_manifest.json", result)
        counts = self._database.job_counts(self._spec["run_id"])
        run_report = {
            "schema_version": 2,
            "run_id": self._spec["run_id"],
            "status": result["status"],
            "success": bool(success),
            "error": str(error),
            "elapsed_sec": result["elapsed_sec"],
            "tile_grid": self._spec.get("tile_grid") or {},
            "spatial_plan_summary": self._spec.get("spatial_plan_summary") or {},
            "storage_preflight": self._spec.get("storage_preflight") or {},
            "job_counts": counts,
            "artifact_cleanup": self._database.artifact_cleanup_summary(
                self._spec["run_id"]
            ),
            "ready_stream_ids": [item["stream_id"] for item in ready_streams],
        }
        atomic_write_json(run_dir / "logs" / "run_report.json", run_report)
        atomic_write_json(
            run_dir / "logs" / "failures.json",
            {
                "run_id": self._spec["run_id"],
                "failed_job_count": int(counts.get("failed", 0)),
                "error": str(error),
            },
        )
        if success:
            self._database.set_run_status(
                self._spec["run_id"], "ready", expected=("running", "raster_ready")
            )
        elif not self._stopped:
            self._database.set_run_status(
                self._spec["run_id"], "failed", expected=("running", "raster_ready")
            )
        self._record_startup_index(result["status"])
        self.pipeline_finished.emit(result)

    def _record_startup_index(self, status):
        try:
            record_run_state(
                self._spec["output_root"],
                self._spec["run_id"],
                status=str(status),
            )
        except (KeyError, OSError, ValueError) as exc:
            self.log_line.emit(
                "system", f"[run-index-warning] 无法更新轻量 Run 启动索引: {exc}"
            )
