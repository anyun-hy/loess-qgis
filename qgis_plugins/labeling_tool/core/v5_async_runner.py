"""Qt orchestration for the PostgreSQL-backed bounded v5 pipeline."""

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

from .manual_package_reset import reset_failed_work_packages
from .process_compat import configure_process, process_is_running
from .recovery_contract import validate_recovery_run
from .result_catalog import artifact_sha256
from .run_index import record_run_state
from .run_spec import atomic_write_json, sha256_file
from .run_state_db import RunStateDB


def _resource_value(spec, key, default):
    tuning = ((spec.get("resource_tuning") or {}).get("resolved") or {})
    return max(1, int(tuning.get(key, default)))


def cpu_worker_limit(spec, *, package_active):
    scaling = spec.get("scaling") or {}
    full = max(1, int(scaling.get("max_cpu_partition_workers", 2)))
    if not package_active:
        return full
    return max(
        1,
        int(scaling.get("max_cpu_partition_workers_with_package", full)),
    )


def process_thread_environment_values(spec, context):
    job = context.get("job") or {}
    if (
        context.get("kind") == "accelerator_worker"
        or job.get("job_type") == "work_package"
        or job.get("job_type") == "fragmentation_v33"
    ):
        threads = _resource_value(spec, "package_process_threads", 2)
    elif job.get("job_type") == "unit_fit":
        threads = _resource_value(spec, "unit_process_threads", 1)
    else:
        threads = _resource_value(spec, "assembly_process_threads", 1)
    value = str(threads)
    return {
        "OMP_NUM_THREADS": value,
        "OMP_DYNAMIC": "FALSE",
        "MKL_NUM_THREADS": value,
        "MKL_DYNAMIC": "FALSE",
        "OPENBLAS_NUM_THREADS": value,
        "BLIS_NUM_THREADS": value,
        "VECLIB_MAXIMUM_THREADS": value,
        "NUMEXPR_NUM_THREADS": value,
        "NUMEXPR_MAX_THREADS": value,
    }


class V5AsyncInferenceRunner(QObject):
    """Run one persistent accelerator worker and a bounded CPU geometry pool."""

    log_line = pyqtSignal(str, str)
    step_started = pyqtSignal(str)
    step_finished = pyqtSignal(str, int, dict)
    pipeline_progress = pyqtSignal(int, int, str)
    stage_progress = pyqtSignal(object)
    stream_progress = pyqtSignal(object)
    pipeline_finished = pyqtSignal(dict)

    REQUIRED_SCRIPTS = (
        "run_work_package.sh",
        "run_fragmentation_v33_work_package.sh",
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
        self._accelerator_worker_id = self._worker_id + "-accelerator"
        self._accelerator_done = False
        self._accelerator_crash_count = 0
        self._processes = {}
        self._assembly_queue = []
        self._started_at = 0.0
        self._manual_package_reset = {}
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
        reset_failed_packages=False,
    ):
        del accepted_layer
        if self._running:
            raise RuntimeError("an inference pipeline is already running")
        if reset_failed_packages and not resume:
            raise RuntimeError("failed Package reset requires resume validation")
        if resume:
            spec, database, spec_path = validate_recovery_run(
                run_spec_path,
                self.scripts_dir,
            )
            self._spec = spec
            self._database = database
            self._spec_path = str(spec_path)
        else:
            self._spec_path = str(Path(run_spec_path).resolve())
            with open(self._spec_path, "r", encoding="utf-8") as handle:
                self._spec = json.load(handle)
            if self._spec.get("schema_version") != 2:
                raise RuntimeError("V5 runner requires run_spec schema 2")
            self._database = RunStateDB(self._spec["state_db"])
        if resume:
            recovered_package_jobs = self._database.recover_ready_work_package_jobs(
                self._spec["run_id"]
            )
            self._database.interrupt_run_jobs(self._spec["run_id"])
        self._manual_package_reset = {}
        if reset_failed_packages:
            self._manual_package_reset = reset_failed_work_packages(
                self._spec,
                database=self._database,
            )
            self.log_line.emit(
                "system",
                "[manual-package-reset] "
                + json.dumps(
                    self._manual_package_reset,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
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
        self._accelerator_worker_id = self._worker_id + "-accelerator"
        self._accelerator_done = False
        self._accelerator_crash_count = 0
        self._assembly_queue = []
        self._started_at = time.time()
        self.log_line.emit("system", f"[run-v5] {self._spec['run_id']}")
        if resume and recovered_package_jobs:
            self.log_line.emit(
                "system",
                "[recovery] finalized ready Work Package jobs: "
                + str(recovered_package_jobs),
            )
        tuning = self._spec.get("resource_tuning") or {}
        if tuning:
            self.log_line.emit(
                "system",
                "[resource-tuning] "
                + json.dumps(tuning, ensure_ascii=False, separators=(",", ":")),
            )
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
            reset_failed_packages=True,
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
            context = entry["context"]
            if context.get("kind") == "accelerator_worker":
                self._database.interrupt_work_package_worker(
                    self._spec["run_id"],
                    context["worker_id"],
                )
                continue
            job = context.get("job")
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
        package_counts = self._database.job_counts(
            self._spec["run_id"],
            job_type="work_package",
        )
        if int(package_counts.get("failed", 0)):
            self._finish(
                False,
                "Work Package exhausted retries; remaining work was stopped: "
                + str(package_counts),
            )
            return
        try:
            self._cleanup_released_artifacts()
        except RuntimeError as error:
            self._finish(False, str(error))
            return
        if self._disk_below_reserve():
            self._emit_progress("磁盘空间低于保留阈值，已暂停派发新任务")
            return

        accelerator_active = any(
            entry["context"].get("kind") == "accelerator_worker"
            for entry in self._processes.values()
        )
        active_jobs = [
            entry["context"].get("job") for entry in self._processes.values()
            if entry["context"].get("kind") == "job"
        ]
        unit_active = sum(1 for job in active_jobs if job and job["job_type"] == "unit_fit")
        candidate_active = any(
            job and job["job_type"] == "fragmentation_v33"
            for job in active_jobs
        )
        started = False

        package_pending = any(
            package_counts.get(status, 0)
            for status in ("queued", "interrupted", "running")
        )
        if not self._accelerator_done and not accelerator_active:
            if package_pending:
                self._start_accelerator_worker()
                started = True
                accelerator_active = True
            else:
                self._accelerator_done = True

        fragmentation = dict(self._spec.get("fragmentation_regularization") or {})
        production_v33 = bool(
            fragmentation.get("enabled") is True
            and fragmentation.get("policy_id")
            == "fragmentation_v33_configurable_absorption_v1"
            and fragmentation.get("publication") == "authoritative_fusion_core"
        )
        replay_v33 = bool(
            (fragmentation.get("comparison") or {}).get("enabled", False)
        )
        v33_enabled = production_v33 or replay_v33
        if v33_enabled and not package_pending:
            candidate_counts = self._database.job_counts(
                self._spec["run_id"], job_type="fragmentation_v33"
            )
            if int(candidate_counts.get("failed", 0)):
                self._finish(
                    False,
                    "V3.3 exhausted retries: "
                    + str(candidate_counts),
                )
                return
            if not candidate_active and not candidate_counts.get("ready", 0):
                job = self._database.lease_next_fragmentation_v33(
                    self._spec["run_id"],
                    self._worker_id + "-fragmentation-v33",
                    lease_seconds=300,
                )
                if job:
                    self._start_job(job)
                    started = True

        # The state DB refuses same-Fusion-stream unit jobs until its V3.3 job
        # is ready, while model streams remain leasable.  Keep dispatch here
        # after the V3.3 attempt so an all-Package-ready Run starts the gate
        # without waiting for unrelated model geometry work.
        cpu_limit = cpu_worker_limit(
            self._spec,
            package_active=accelerator_active,
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
        if job["job_type"] == "fragmentation_v33":
            self._start_process(
                "fragmentation_v33_candidate",
                "run_fragmentation_v33_work_package.sh",
                [
                    "--run-spec", self._spec_path,
                    "--worker-id", self._worker_id + "-fragmentation-v33",
                    "--job-id", str(job["job_id"]),
                    "--lease-token", job["lease_token"],
                    "--lease-seconds", "300",
                ],
                {"kind": "job", "job": job},
            )
            return
        if job["job_type"] != "unit_fit":
            raise RuntimeError(
                "QGIS may only launch unit_fit jobs directly; "
                "Work Packages belong to the persistent accelerator worker"
            )
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

    def _start_accelerator_worker(self):
        self._start_process(
            "accelerator_worker",
            "run_work_package.sh",
            [
                "--run-spec", self._spec_path,
                "--worker-id", self._accelerator_worker_id,
                "--device", self._spec["runtime"]["effective_device"],
                "--max-open-frontier-units",
                str(
                    int(
                        (self._spec.get("scaling") or {}).get(
                            "max_open_frontier_units", 64
                        )
                    )
                ),
                "--resume",
            ],
            {
                "kind": "accelerator_worker",
                "worker_id": self._accelerator_worker_id,
            },
        )

    def _start_assembly(self):
        scaling = self._spec.get("scaling") or {}
        max_concurrent = max(
            1,
            min(
                int(scaling.get("max_concurrent_assembly", 2)),
                max(1, len(self._spec.get("streams") or [])),
            ),
        )
        while self._assembly_queue:
            active_assemblies = [
                entry
                for entry in self._processes.values()
                if (entry.get("context") or {}).get("kind") == "assemble"
            ]
            if len(active_assemblies) >= max_concurrent:
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
            stream = self._assembly_queue.pop(0)
            self._start_process(
                f"assemble_stream:{stream['stream_id']}",
                "run_assemble_stream.sh",
                ["--run-spec", self._spec_path, "--stream-id", stream["stream_id"]],
                {"kind": "assemble", "stream_id": stream["stream_id"]},
            )
        active_assemblies = [
            entry
            for entry in self._processes.values()
            if (entry.get("context") or {}).get("kind") == "assemble"
        ]
        if not active_assemblies and not self._assembly_queue:
            QTimer.singleShot(0, self._start_acceptance)

    def _start_acceptance(self):
        """Continue from the one assembly pass to acceptance.

        Historical completed Runs can still be repaired explicitly with the
        standalone fragmentation script.  New v5 Runs never launch it or let
        it replace the formal assembled GPKG.
        """
        self._phase = "acceptance"
        self._start_process(
            "scale_acceptance",
            "run_scale_acceptance.sh",
            ["--run-spec", self._spec_path],
            {"kind": "scale_acceptance"},
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
        for name, value in process_thread_environment_values(
            self._spec, context
        ).items():
            environment.insert(name, value)
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
        process.errorOccurred.connect(
            lambda error, t=token: self._process_error(t, error)
        )
        self.step_started.emit(label)
        self.log_line.emit("system", "[cmd] " + shlex.join(["/bin/bash", path, *arguments]))
        process.start()

    def _process_error(self, token, _process_error):
        entry = self._processes.get(token)
        if not entry or not self._running:
            return
        process = entry["process"]
        entry["forced_error"] = (
            entry["forced_error"]
            or f"{entry['context']['label']} process error: {process.errorString()}"
        )
        if not process_is_running(process):
            QTimer.singleShot(
                0,
                lambda t=token: self._process_finished(t, -1, None),
            )

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
        if event["event"] == "work_package_finished":
            # A completed Package proves the persistent worker reached useful
            # work; only consecutive process crashes count toward the guard.
            self._accelerator_crash_count = 0
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

        if context.get("kind") == "accelerator_worker":
            worker_id = context["worker_id"]
            if not success:
                self._database.interrupt_work_package_worker(
                    self._spec["run_id"],
                    worker_id,
                )
            package_counts = self._database.job_counts(
                self._spec["run_id"],
                job_type="work_package",
            )
            if int(package_counts.get("failed", 0)):
                error = (
                    "Work Package exhausted retries; remaining work was stopped: "
                    + str(package_counts)
                )
                self.step_finished.emit(
                    label,
                    int(exit_code),
                    {"success": False, "error": error, "stream_id": ""},
                )
                self._accelerator_done = True
                self._finish(False, error)
                return
            package_pending = any(
                package_counts.get(status, 0)
                for status in ("queued", "interrupted", "running")
            )
            if success and package_pending:
                success = False
                error = (
                    "accelerator_worker exited while Work Packages remain: "
                    + str(package_counts)
                )
                self._database.interrupt_work_package_worker(
                    self._spec["run_id"],
                    worker_id,
                )
                package_counts = self._database.job_counts(
                    self._spec["run_id"],
                    job_type="work_package",
                )
                package_pending = any(
                    package_counts.get(status, 0)
                    for status in ("queued", "interrupted", "running")
                )
            self.step_finished.emit(
                label,
                int(exit_code),
                {
                    "success": success,
                    "error": error,
                    "stream_id": "",
                },
            )
            if success:
                self._accelerator_done = True
                self._accelerator_crash_count = 0
            elif package_pending:
                self._accelerator_crash_count += 1
                if self._accelerator_crash_count >= 3:
                    self._finish(
                        False,
                        "persistent accelerator worker crashed repeatedly: "
                        + error,
                    )
                    return
                self.log_line.emit(
                    "system",
                    "[accelerator-restart] "
                    f"attempt={self._accelerator_crash_count} error={error}",
                )
            else:
                # No Package can be retried.  Let the normal job graph check
                # report exhausted failures instead of respawning the worker.
                self._accelerator_done = True
            self._emit_progress(label)
            QTimer.singleShot(0, self._schedule)
            return

        if context.get("kind") == "job":
            job = context["job"]
            current = self._database.get_job(job["job_id"])
            if current and current["status"] == "running":
                if job.get("job_type") == "fragmentation_v33" and success:
                    success = False
                    error = (
                        "V3.3 worker exited without its atomic output commit"
                    )
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
        if graceful and process_is_running(process):
            try:
                if pid > 0 and entry.get("owns_process_group"):
                    os.killpg(pid, signal.SIGKILL)
                else:
                    process.kill()
            except (ProcessLookupError, OSError):
                process.kill()
            process.waitForFinished(1000)

    def _disk_below_reserve(self):
        storage = self._spec.get("storage_preflight") or {}
        reserve = int(
            storage.get("effective_min_free_disk_bytes")
            or float((self._spec.get("scaling") or {}).get("min_free_disk_gb", 0))
            * 1024**3
        )
        return shutil.disk_usage(self._spec["output_root"]).free <= reserve

    def _cleanup_released_artifacts(self):
        candidates = self._database.cleanup_candidates(
            self._spec["run_id"],
            limit=1000,
            kinds=("partition_probability", "v3_context_core", "v3_baseline_core"),
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
        # The formal assembled geometry is the review source for every new
        # stream.  Candidate and historical postprocess layers are auxiliary
        # diagnostics only and may not silently replace production geometry.
        result["review_polygons"] = paths["semantic_polygons"]
        result["review_layer_name"] = "semantic_polygons"
        result["output_sha256"]["review_polygons"] = result["output_sha256"][
            "semantic_polygons"
        ]
        return result

    def _finish(self, success, error):
        if not self._running:
            return
        self._scheduler.stop()
        self._watchdog.stop()
        self._running = False
        if not success and self._processes:
            entries = list(self._processes.values())
            for entry in entries:
                self._terminate_entry(entry, graceful=False)
            for entry in entries:
                context = entry["context"]
                if context.get("kind") == "accelerator_worker":
                    self._database.interrupt_work_package_worker(
                        self._spec["run_id"],
                        context["worker_id"],
                    )
                    continue
                job = context.get("job")
                if job:
                    self._database.interrupt_job(
                        job["job_id"], job["lease_token"]
                    )
            self._processes.clear()
        if not success and not self._stopped:
            self._database.set_run_status(
                self._spec["run_id"],
                "failed",
                expected=("running", "raster_ready"),
            )
            self._database.fail_open_streams(
                self._spec["run_id"],
                str(error),
            )
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
        if self._manual_package_reset:
            result["manual_package_reset"] = dict(self._manual_package_reset)
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
