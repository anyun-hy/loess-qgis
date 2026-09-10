"""Qt orchestration for the PostgreSQL-backed bounded v5 pipeline."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from qgis.PyQt.QtCore import (
    QObject,
    QProcess,
    QProcessEnvironment,
    QThread,
    QTimer,
    pyqtSignal,
    pyqtSlot,
)

from .manual_package_reset import reset_failed_work_packages
from .memory_admission import (
    AdaptiveMemoryAdmissionController,
    MemoryAdmissionDecision,
    MemoryPressureSample,
)
from .process_runtime import configure_process, process_is_running
from .recovery_contract import validate_recovery_run
from .result_catalog import artifact_sha256
from .run_index import record_run_state
from .run_spec import atomic_write_json, sha256_file
from .run_state_db import run_state_from_spec
from .monitor_contract import MONITOR_EXECUTION_ENV


def _resource_value(spec, key, default):
    tuning = ((spec.get("resource_tuning") or {}).get("resolved") or {})
    return max(1, int(tuning.get(key, default)))


def _fragmentation_v33_process_threads(spec):
    """Return the frozen CPU-thread reservation of one V3.3 worker."""

    return _resource_value(
        spec,
        "fragmentation_v33_process_threads",
        _resource_value(spec, "package_process_threads", 2),
    )


def _unit_fit_process_threads(spec):
    """Return the frozen CPU-thread reservation of one unit-fit worker."""

    return _resource_value(spec, "unit_process_threads", 1)


def geometry_thread_budget(spec, *, package_active):
    """Return the frozen CPU-thread ceiling available to geometry work."""

    scaling = spec.get("scaling") or {}
    full = max(1, int(scaling.get("max_cpu_partition_workers", 2)))
    if not package_active:
        return full
    return max(
        1,
        int(scaling.get("max_cpu_partition_workers_with_package", full)),
    )


def cpu_worker_limit(
    spec,
    *,
    package_active,
    fragmentation_v33_active=0,
    geometry_slot_limit=None,
):
    """Return the remaining unit-fit process capacity in the shared CPU pool.

    ``max_cpu_partition_workers[_with_package]`` is the frozen geometry budget.
    A V3.3 process uses its own bounded native-thread pool, so every active
    V3.3 process consumes that many unit-fit slots.  Returning zero is valid:
    it prevents oversubscription until a V3.3 worker completes.
    """

    geometry_budget = geometry_thread_budget(spec, package_active=package_active)
    if geometry_slot_limit is not None:
        geometry_budget = min(
            geometry_budget,
            max(0, int(geometry_slot_limit)),
        )
    v33_active = max(0, int(fragmentation_v33_active))
    remaining_threads = max(
        0,
        geometry_budget - v33_active * _fragmentation_v33_process_threads(spec),
    )
    return remaining_threads // _unit_fit_process_threads(spec)


def fragmentation_v33_worker_limit(
    spec,
    *,
    package_active,
    unit_fit_active=0,
    geometry_slot_limit=None,
):
    """Bound V3.3 workers so all active CPU processes fit the frozen budget."""

    scaling = spec.get("scaling") or {}
    full = max(1, int(scaling.get("max_cpu_partition_workers", 2)))
    if geometry_slot_limit is None:
        package_reservation = (
            _resource_value(spec, "package_process_threads", 2)
            if package_active
            else 0
        )
        geometry_budget = max(0, full - package_reservation)
    else:
        geometry_budget = max(0, min(full, int(geometry_slot_limit)))
    unit_reservation = max(0, int(unit_fit_active)) * _unit_fit_process_threads(spec)
    available = max(0, geometry_budget - unit_reservation)
    return available // _fragmentation_v33_process_threads(spec)


def process_thread_environment_values(spec, context):
    job = context.get("job") or {}
    if job.get("job_type") == "fragmentation_v33":
        threads = _fragmentation_v33_process_threads(spec)
    elif (
        context.get("kind") == "accelerator_worker"
        or job.get("job_type") == "work_package"
    ):
        threads = _resource_value(spec, "package_process_threads", 2)
    elif job.get("job_type") in {"unit_fit", "unit_confidence"}:
        threads = _unit_fit_process_threads(spec)
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


def _final_artifact_size_prediction_log_message(prediction):
    """Return the per-Run forecast line written to the monitor log."""

    value = dict(prediction or {})
    try:
        predicted = int(value["predicted_final_artifact_bytes"])
    except (KeyError, TypeError, ValueError):
        return ""
    return f"本次 Run 预计最终保存 {predicted / 1024**3:.2f} GiB；完成后回报实际值和差额"


def _final_artifact_size_observation_log_message(observation):
    """Return the per-Run final-size result written to the monitor log."""

    value = dict(observation or {})
    try:
        actual = int(value["actual_final_artifact_bytes"])
    except (KeyError, TypeError, ValueError):
        return ""
    gib = 1024**3
    try:
        predicted = int(value["predicted_final_artifact_bytes"])
        difference = int(value["signed_difference_bytes"])
        ratio = float(value["signed_difference_ratio"])
    except (KeyError, TypeError, ValueError):
        return f"本次 Run 最终保存 {actual / gib:.2f} GiB"
    sign = "+" if difference >= 0 else "−"
    return (
        f"本次 Run：预计最终保存 {predicted / gib:.2f} GiB；"
        f"实际最终保存 {actual / gib:.2f} GiB；"
        f"差额 {sign}{abs(difference) / gib:.2f} GiB（{ratio:+.2%}）"
    )


class PipelinePhaseTiming:
    """Record high-level wall-clock stage timing without summing parallel work."""

    STAGES = (
        "work_package",
        "fragmentation_v33",
        "unit_confidence",
        "unit_fit",
        "finalize",
        "assembly",
        "acceptance",
    )

    def __init__(self, spans=None, *, recovered_incomplete_span_count=0):
        self._spans = list(spans or [])
        self._recovered_incomplete_span_count = max(
            0, int(recovered_incomplete_span_count)
        )

    @classmethod
    def from_state(cls, value):
        """Recover durable completed observations, never counting downtime."""

        if not isinstance(value, dict) or value.get("schema_version") != 1:
            return cls()
        try:
            recorded_at = float(value.get("recorded_at"))
        except (TypeError, ValueError):
            recorded_at = 0.0
        # A recovered state serializes its prior recovery count in ``summary``.
        # Retain that history across later restarts, but only accept the exact
        # non-negative integer produced by this schema.  In particular, avoid
        # coercing malformed values (including booleans) into a false count.
        previous_recovered_incomplete = 0
        summary = value.get("summary")
        if isinstance(summary, dict):
            recorded_count = summary.get("recovered_incomplete_span_count")
            if (
                isinstance(recorded_count, int)
                and not isinstance(recorded_count, bool)
                and recorded_count >= 0
            ):
                previous_recovered_incomplete = recorded_count
        spans = []
        recovered_incomplete = previous_recovered_incomplete
        for item in value.get("spans") or []:
            if not isinstance(item, dict) or item.get("stage") not in cls.STAGES:
                continue
            try:
                started_at = float(item["started_at"])
                finished_at = item.get("finished_at")
                if finished_at is None:
                    finished_at = float(item.get("last_observed_at") or recorded_at)
                    recovered_incomplete += 1
                else:
                    finished_at = float(finished_at)
            except (KeyError, TypeError, ValueError):
                continue
            if started_at <= 0 or finished_at < started_at:
                continue
            spans.append(
                {
                    "stage": item["stage"],
                    "started_at": started_at,
                    "finished_at": finished_at,
                }
            )
        return cls(
            spans,
            recovered_incomplete_span_count=recovered_incomplete,
        )

    def start(self, stage, token, started_at):
        if stage not in self.STAGES:
            return
        self._spans.append(
            {
                "stage": stage,
                "token": str(token),
                "started_at": float(started_at),
                "finished_at": None,
                "last_observed_at": float(started_at),
            }
        )

    def observe_active(self, observed_at):
        for item in self._spans:
            if item.get("finished_at") is None:
                item["last_observed_at"] = max(
                    float(observed_at), item["started_at"]
                )

    def finish(self, token, finished_at):
        for item in reversed(self._spans):
            if item.get("token") == str(token) and item.get("finished_at") is None:
                item["finished_at"] = max(float(finished_at), item["started_at"])
                item["last_observed_at"] = item["finished_at"]
                return

    def finish_active(self, finished_at):
        for item in self._spans:
            if item.get("finished_at") is None:
                item["finished_at"] = max(float(finished_at), item["started_at"])
                item["last_observed_at"] = item["finished_at"]

    @staticmethod
    def _union_elapsed(spans, now):
        intervals = sorted(
            (
                float(item["started_at"]),
                max(float(item["started_at"]), float(item.get("finished_at") or now)),
            )
            for item in spans
        )
        elapsed = 0.0
        left = right = None
        for start, end in intervals:
            if left is None:
                left, right = start, end
            elif start > right:
                elapsed += right - left
                left, right = start, end
            else:
                right = max(right, end)
        return elapsed + (right - left if left is not None else 0.0)

    def summary(self, now):
        now = float(now)
        result = {
            "schema_version": 1,
            "measurement": "wall_clock_union_sec",
            "status": (
                "partial_recovered"
                if self._recovered_incomplete_span_count
                else "in_progress"
                if any(item.get("finished_at") is None for item in self._spans)
                else "complete"
            ),
            "recovered_incomplete_span_count": self._recovered_incomplete_span_count,
            "stages": {},
        }
        by_stage = {stage: [] for stage in self.STAGES}
        for item in self._spans:
            if item.get("stage") in by_stage:
                by_stage[item["stage"]].append(item)
        for stage, spans in by_stage.items():
            result["stages"][stage] = {
                "wall_clock_sec": round(self._union_elapsed(spans, now), 3),
                "invocation_count": len(spans),
            }
        unit_spans = by_stage["unit_fit"]
        package_spans = by_stage["work_package"]
        last_unit_end = max(
            (float(item.get("finished_at") or now) for item in unit_spans), default=0.0
        )
        last_package_end = max(
            (float(item.get("finished_at") or now) for item in package_spans), default=0.0
        )
        result["unit_fit_total_wall_clock_sec"] = result["stages"]["unit_fit"][
            "wall_clock_sec"
        ]
        result["unit_fit_tail_after_work_package_sec"] = round(
            (
                max(0.0, last_unit_end - last_package_end)
                if last_unit_end and last_package_end
                else 0.0
            ),
            3,
        )
        return result

    def state(self, now):
        return {
            "schema_version": 1,
            "recorded_at": float(now),
            "spans": [
                {
                    "stage": item["stage"],
                    "started_at": item["started_at"],
                    "finished_at": item.get("finished_at"),
                    "last_observed_at": item.get("last_observed_at"),
                }
                for item in self._spans
            ],
            "summary": self.summary(now),
        }


class _BatchedPipelineLogWriter(QObject):
    """Write complete pipeline logs away from both Qt control-plane threads."""

    def __init__(self):
        super().__init__(None)
        self._pending = []
        self._handles = {}
        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(1000)
        self._flush_timer.timeout.connect(self.flush)

    @pyqtSlot()
    def start(self):
        self._flush_timer.start()

    @pyqtSlot(object)
    def append_batch(self, records):
        self._pending.extend(dict(record) for record in records or ())

    @pyqtSlot()
    def flush(self):
        if not self._pending:
            return
        records = self._pending
        self._pending = []
        grouped = {}
        for record in records:
            run_dir = str(record.get("run_dir") or "")
            if not run_dir:
                continue
            grouped.setdefault(run_dir, []).append(record)
        for run_dir, values in grouped.items():
            try:
                handle = self._handles.get(run_dir)
                if handle is None or handle.closed:
                    path = Path(run_dir) / "logs" / "pipeline.jsonl"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    handle = open(path, "a", encoding="utf-8")
                    self._handles[run_dir] = handle
                handle.write(
                    "".join(
                        json.dumps(
                            {
                                "timestamp": float(record.get("timestamp") or time.time()),
                                "level": str(record.get("source") or "system"),
                                "message": str(record.get("message") or ""),
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                        for record in values
                    )
                )
                handle.flush()
            except OSError:
                # Logging must never terminate a scientifically valid Run.
                continue

    @pyqtSlot(object)
    def finalize_run(self, run_dir):
        self.flush()
        key = str(run_dir or "")
        handle = self._handles.pop(key, None)
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass

    @pyqtSlot()
    def shutdown(self):
        self._flush_timer.stop()
        self.flush()
        for handle in tuple(self._handles.values()):
            try:
                handle.close()
            except OSError:
                pass
        self._handles.clear()
        QThread.currentThread().quit()


class V5AsyncInferenceRunner(QObject):
    """Run one persistent accelerator worker and a bounded CPU geometry pool."""

    log_line = pyqtSignal(str, str)
    process_log = pyqtSignal(object)
    step_started = pyqtSignal(str)
    step_finished = pyqtSignal(str, int, dict)
    pipeline_progress = pyqtSignal(int, int, str)
    stage_progress = pyqtSignal(object)
    stream_progress = pyqtSignal(object)
    pipeline_finished = pyqtSignal(dict)
    ui_log_batch = pyqtSignal(object)
    ui_progress_batch = pyqtSignal(object)
    log_run_finalized = pyqtSignal(object)
    runtime_start_failed = pyqtSignal(object)

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
        self._execution_id = ""
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
        self._phase_timing = PipelinePhaseTiming()
        self._manual_package_reset = {}
        self._model_monitor_spans = {}
        self._assembly_monitor_spans = {}
        self._monitor_history_incomplete = False
        self._memory_admission = AdaptiveMemoryAdmissionController()
        self._last_memory_admission_log_at = 0.0
        self._pending_ui_logs = []
        self._pending_stream_progress = {}
        self._priority_stream_progress = []
        self._pending_pipeline_progress = None
        self._pending_job_progress = {}
        self._cleanup_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="loess-artifact-cleanup",
        )
        self._cleanup_future = None
        self._last_cleanup_started_at = 0.0
        # These connections are made before moveToThread(). Keep their handlers
        # as native @pyqtSlot methods, otherwise PyQt's Python-slot proxies can
        # stay on the GUI thread after the worker and its timers move away.
        self._scheduler = QTimer(self)
        self._scheduler.setInterval(500)
        self._scheduler.timeout.connect(self._schedule_safely)
        self._watchdog = QTimer(self)
        self._watchdog.setInterval(15000)
        self._watchdog.timeout.connect(self._heartbeat_and_watchdog)
        self._heartbeat_timer = QTimer(self)
        self._heartbeat_timer.setInterval(5000)
        self._heartbeat_timer.timeout.connect(self._flush_job_heartbeats)
        self._ui_flush_timer = QTimer(self)
        self._ui_flush_timer.setInterval(100)
        self._ui_flush_timer.timeout.connect(self._flush_ui_events)
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
            self._database = run_state_from_spec(self._spec)
        if resume:
            recovered_package_jobs = self._database.recover_ready_work_package_jobs(
                self._spec["run_id"]
            )
            self._database.interrupt_run_jobs(self._spec["run_id"])
        trigger_type = (
            "redo_failed_packages"
            if reset_failed_packages
            else "resume" if resume else "start"
        )
        self._execution_id = self._database.begin_monitor_execution(
            self._spec["run_id"],
            trigger_type,
            metadata={"worker_id": self._worker_id},
        )
        self._database.monitor_execution_id = self._execution_id
        self._manual_package_reset = {}
        self._phase_timing = self._load_phase_timing()
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
        entered_running = self._database.set_run_status(
            self._spec["run_id"],
            "running",
            expected=("planned", "running", "stopped", "failed", "raster_ready"),
        )
        if not entered_running:
            self._database.finish_monitor_execution(
                self._spec["run_id"],
                self._execution_id,
                status="failed",
                message="Run state changed before execution",
            )
            raise RuntimeError(
                "Run state changed before execution; it may have been archived "
                "or claimed by another runner"
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
        self._model_monitor_spans.clear()
        self._assembly_monitor_spans.clear()
        self._monitor_history_incomplete = False
        self._started_at = time.time()
        memory_policy = (
            ((self._spec.get("resource_tuning") or {}).get("resolved") or {}).get(
                "memory_admission"
            )
            or {}
        )
        self._memory_admission = AdaptiveMemoryAdmissionController(memory_policy)
        self._last_memory_admission_log_at = 0.0
        self._persist_phase_timing()
        self.log_line.emit("system", f"[run-v5] {self._spec['run_id']}")
        if not resume:
            try:
                run_row = self._database.get_run(self._spec["run_id"]) or {}
                run_metadata = json.loads(
                    str(run_row.get("metadata_json") or "{}")
                )
                cleanup = dict(
                    run_metadata.get("incomplete_run_cleanup") or {}
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                cleanup = {}
            if cleanup.get("status") == "warning":
                self.log_line.emit(
                    "stderr",
                    "[incomplete-run-cleanup-warning] "
                    + str(cleanup.get("error") or "unknown cleanup error"),
                )
            elif int(cleanup.get("skipped_active_run_count") or 0) > 0:
                self.log_line.emit(
                    "stderr",
                    "[incomplete-run-cleanup-warning] 旧未完成 Run 仍有活动 Job，"
                    "已保留: "
                    + ", ".join(
                        cleanup.get("skipped_active_run_ids") or []
                    ),
                )
            elif int(cleanup.get("archived_run_count") or 0) > 0:
                self.log_line.emit(
                    "system",
                    "[incomplete-run-cleanup] 已归档旧未完成 Run: "
                    + ", ".join(cleanup.get("archived_run_ids") or []),
                )
        size_prediction = (
            (self._spec.get("storage_preflight") or {}).get(
                "final_artifact_size_prediction"
            )
            or {}
        )
        size_prediction_message = _final_artifact_size_prediction_log_message(
            size_prediction
        )
        if size_prediction_message:
            self.log_line.emit("system", f"[final-artifact-size] {size_prediction_message}")
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
        self._heartbeat_timer.start()
        self._ui_flush_timer.start()
        QTimer.singleShot(0, self._schedule_safely)

    @pyqtSlot(object)
    def execute_request(self, request):
        """Execute one queued facade request inside the runtime thread."""

        value = dict(request or {})
        try:
            self.run_from_spec(
                str(value.get("run_spec_path") or ""),
                resume=bool(value.get("resume")),
                reset_failed_packages=bool(value.get("reset_failed_packages")),
            )
        except Exception as error:
            if self._running:
                self._finish(False, str(error))
                return
            spec = dict(getattr(self, "_spec", {}) or {})
            execution_id = str(getattr(self, "_execution_id", "") or "")
            database = getattr(self, "_database", None)
            if database is not None and execution_id and spec.get("run_id"):
                try:
                    database.finish_monitor_execution(
                        spec["run_id"],
                        execution_id,
                        status="failed",
                        message=f"{type(error).__name__}: {error}",
                    )
                except Exception as history_error:
                    self.log_line.emit(
                        "stderr",
                        "[monitor-history] cannot seal failed start: "
                        + str(history_error),
                    )
            result = {
                "schema_version": 2,
                "run_id": str(spec.get("run_id") or ""),
                "run_spec": str(value.get("run_spec_path") or ""),
                "run_dir": str(spec.get("run_dir") or ""),
                "success": False,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "ready_streams": [],
                "failed_streams": list(spec.get("streams") or []),
                "streams": list(spec.get("streams") or []),
            }
            self.log_line.emit("stderr", "[runtime-start-error] " + result["error"])
            self._flush_ui_events()
            self.runtime_start_failed.emit(result)

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
        self._heartbeat_timer.stop()
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

    @pyqtSlot()
    def shutdown_runtime(self):
        """Stop all owned work before terminating the dedicated Qt thread."""

        self.stop()
        self._flush_ui_events()
        executor = getattr(self, "_cleanup_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
            self._cleanup_executor = None
        QThread.currentThread().quit()

    @staticmethod
    def _geometry_job_threads(spec, job):
        if (job or {}).get("job_type") == "fragmentation_v33":
            return _fragmentation_v33_process_threads(spec)
        if (job or {}).get("job_type") in {"unit_fit", "unit_confidence"}:
            return _unit_fit_process_threads(spec)
        return 0

    def _memory_admission_decision(
        self,
        *,
        static_limit,
        active_jobs,
        package_active,
    ):
        active_slots = sum(
            self._geometry_job_threads(self._spec, job) for job in active_jobs
        )
        controller = getattr(self, "_memory_admission", None)
        if controller is None:
            return MemoryAdmissionDecision(
                geometry_slot_limit=max(1, int(static_limit)),
                pause_new_work=False,
                shed_active_work=False,
                reason="controller_unavailable_static_fallback",
                worker_peak_estimate_bytes=0,
                changed=False,
                sample=MemoryPressureSample(),
            )
        decision = controller.decide(
            static_limit=static_limit,
            active_slots=active_slots,
            package_active=package_active,
        )
        now = time.monotonic()
        last_log = float(getattr(self, "_last_memory_admission_log_at", 0.0))
        if decision.changed or now - last_log >= 60.0:
            payload = decision.payload()
            payload.update(
                {
                    "static_geometry_slot_limit": int(static_limit),
                    "active_geometry_slots": int(active_slots),
                    "package_active": bool(package_active),
                }
            )
            self.log_line.emit(
                "system",
                "[memory-admission] "
                + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            )
            self._last_memory_admission_log_at = now
        return decision

    def _force_memory_shed(self, token):
        entry = self._processes.get(token)
        if not entry or not entry.get("memory_shed"):
            return
        process = entry["process"]
        if not process_is_running(process):
            return
        pid = int(process.processId())
        try:
            if pid > 0 and entry.get("owns_process_group"):
                os.killpg(pid, signal.SIGKILL)
            else:
                process.kill()
        except (ProcessLookupError, OSError):
            process.kill()

    def _request_memory_shed(self, entry, reason):
        """Interrupt one atomic geometry job without blocking the Qt thread."""

        if entry.get("memory_shed"):
            return False
        job = (entry.get("context") or {}).get("job") or {}
        if job.get("job_type") not in {
            "unit_fit", "unit_confidence", "fragmentation_v33"
        }:
            return False
        if not self._database.interrupt_job(job["job_id"], job["lease_token"]):
            return False
        entry["memory_shed"] = True
        entry["forced_error"] = "memory-pressure load shedding: " + str(reason)
        process = entry["process"]
        if process_is_running(process):
            pid = int(process.processId())
            try:
                if pid > 0 and entry.get("owns_process_group"):
                    os.killpg(pid, signal.SIGTERM)
                else:
                    process.terminate()
            except (ProcessLookupError, OSError):
                process.terminate()
            token = entry["token"]
            QTimer.singleShot(3000, lambda t=token: self._force_memory_shed(t))
        return True

    def _shed_geometry_to_limit(self, geometry_slot_limit, reason):
        target = max(1, int(geometry_slot_limit))
        active = []
        active_slots = 0
        for entry in self._processes.values():
            if entry.get("memory_shed"):
                continue
            job = (entry.get("context") or {}).get("job") or {}
            threads = self._geometry_job_threads(self._spec, job)
            if threads <= 0:
                continue
            active.append((entry, job, threads))
            active_slots += threads
        if active_slots <= target:
            return 0

        def shed_order(item):
            entry, job, _threads = item
            unit_id = str(job.get("unit_id") or "")
            job_type = str(job.get("job_type") or "")
            if job_type == "unit_fit" and unit_id.startswith("core_"):
                priority = 0
            elif job_type == "unit_fit":
                priority = 1
            elif job_type == "unit_confidence":
                priority = 2
            else:
                priority = 3
            started_at = float((entry.get("context") or {}).get("started_at") or 0.0)
            return priority, -started_at

        interrupted = 0
        for entry, _job, threads in sorted(active, key=shed_order):
            if active_slots <= target:
                break
            if self._request_memory_shed(entry, reason):
                interrupted += 1
                active_slots -= threads
        if interrupted:
            self.log_line.emit(
                "system",
                "[memory-shed] "
                f"interrupted_jobs={interrupted} target_geometry_slots={target} "
                f"reason={reason}",
            )
        return interrupted

    @pyqtSlot()
    def _schedule_safely(self):
        """Make scheduler failures visible and terminate the Run coherently."""

        try:
            self._schedule()
        except Exception as error:
            message = f"[scheduler-error] {type(error).__name__}: {error}"
            self.log_line.emit("stderr", message)
            self._finish(False, message)

    def _schedule(self):
        if not self._running or self._stopped:
            return
        if self._phase != "jobs":
            return
        recover_expired = getattr(self._database, "interrupt_expired_jobs", None)
        if callable(recover_expired):
            recover_expired(run_id=self._spec["run_id"])
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
        unit_active = sum(
            1 for job in active_jobs
            if job and job["job_type"] in {"unit_fit", "unit_confidence"}
        )
        candidate_active = sum(
            1 for job in active_jobs
            if job and job["job_type"] == "fragmentation_v33"
        )
        started = False

        package_pending = any(
            package_counts.get(status, 0)
            for status in ("queued", "interrupted", "running")
        )
        package_expected = bool(
            accelerator_active
            or (not self._accelerator_done and package_pending)
        )
        static_geometry_limit = geometry_thread_budget(
            self._spec,
            package_active=package_expected,
        )
        memory_decision = self._memory_admission_decision(
            static_limit=static_geometry_limit,
            active_jobs=active_jobs,
            package_active=package_expected,
        )
        if memory_decision.shed_active_work:
            self._shed_geometry_to_limit(
                memory_decision.geometry_slot_limit,
                memory_decision.reason,
            )
        if memory_decision.pause_new_work:
            self._emit_progress(
                "检测到内存压力，已暂停派发并动态降低并发；现有安全任务完成后自动恢复"
            )
            return

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
        v33_enabled = production_v33
        if v33_enabled:
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
            configured_candidate_limit = min(
                2 if accelerator_active else 4,
                max(1, int(fragmentation.get("max_workers", 4))),
            )
            candidate_limit = min(
                configured_candidate_limit,
                fragmentation_v33_worker_limit(
                    self._spec,
                    package_active=accelerator_active,
                    unit_fit_active=unit_active,
                    geometry_slot_limit=memory_decision.geometry_slot_limit,
                ),
            )
            leased_candidate_ids = set()
            while candidate_active < candidate_limit:
                job = self._database.lease_next_fragmentation_v33(
                    self._spec["run_id"],
                    self._worker_id + f"-fragmentation-v33-{candidate_active}",
                    lease_seconds=300,
                    max_running=candidate_limit,
                )
                if not job:
                    break
                if int(job["job_id"]) in leased_candidate_ids:
                    raise RuntimeError("state backend leased one V3.3 job twice")
                leased_candidate_ids.add(int(job["job_id"]))
                self._start_job(job)
                candidate_active += 1
                started = True

        # The state DB refuses same-Fusion-stream unit jobs until its V3.3 job
        # is ready, while model streams remain leasable.  Keep dispatch here
        # after the V3.3 attempt so an all-Package-ready Run starts the gate
        # without waiting for unrelated model geometry work.
        cpu_limit = cpu_worker_limit(
            self._spec,
            package_active=accelerator_active,
            fragmentation_v33_active=candidate_active,
            geometry_slot_limit=memory_decision.geometry_slot_limit,
        )
        while unit_active < cpu_limit:
            job = self._database.lease_next_job(
                self._spec["run_id"],
                self._worker_id + f"-geometry-{unit_active}",
                job_types=("unit_confidence", "unit_fit"),
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
                f"fragmentation_v33:{job['unit_id']}",
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
        if job["job_type"] == "unit_confidence":
            self._start_process(
                f"unit_confidence:{job['stream_id']}:{job['unit_id']}",
                "run_unit_confidence.sh",
                [
                    "--run-spec", self._spec_path,
                    "--stream-id", job["stream_id"],
                    "--unit-id", job["unit_id"],
                    "--job-id", str(job["job_id"]),
                    "--lease-token", job["lease_token"],
                ],
                {"kind": "job", "job": job},
            )
            return
        if job["job_type"] != "unit_fit":
            raise RuntimeError(
                "QGIS may only launch unit confidence or unit_fit jobs directly; "
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
                int(scaling.get("max_concurrent_assembly", 4)),
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

    @staticmethod
    def _timing_stage(context):
        kind = str((context or {}).get("kind") or "")
        job = (context or {}).get("job") or {}
        if kind == "accelerator_worker":
            return "work_package"
        if job.get("job_type") == "fragmentation_v33":
            return "fragmentation_v33"
        if job.get("job_type") == "unit_confidence":
            return "unit_confidence"
        if job.get("job_type") == "unit_fit":
            return "unit_fit"
        return {
            "finalize_rasters": "finalize",
            "assemble": "assembly",
            "scale_acceptance": "acceptance",
        }.get(kind, "")

    def _phase_timing_path(self):
        return Path(self._spec["run_dir"]) / "logs" / "phase_timing.json"

    def _load_phase_timing(self):
        try:
            value = json.loads(self._phase_timing_path().read_text(encoding="utf-8"))
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            return PipelinePhaseTiming()
        return PipelinePhaseTiming.from_state(value)

    def _persist_phase_timing(self):
        if not self._spec.get("run_dir"):
            return
        atomic_write_json(
            self._phase_timing_path(),
            self._phase_timing.state(time.time()),
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
        if self._execution_id:
            environment.insert(MONITOR_EXECUTION_ENV, self._execution_id)
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
        if context.get("kind") in {"finalize_rasters", "assemble", "scale_acceptance"}:
            object_type = "stream" if context.get("stream_id") else "run"
            object_id = str(context.get("stream_id") or self._spec.get("run_id") or "")
            entry["monitor_span_id"] = self._database.start_monitor_span(
                self._spec["run_id"],
                execution_id=self._execution_id,
                span_kind="runtime_phase",
                object_type=object_type,
                object_id=object_id,
                stream_id=str(context.get("stream_id") or ""),
                phase=str(context.get("kind") or ""),
                idempotency_key=f"process:{self._execution_id}:{token}",
                metadata={"label": str(label)},
            )
        self._processes[token] = entry
        stage = self._timing_stage(context)
        if stage:
            self._phase_timing.start(stage, token, entry["context"]["started_at"])
            self._persist_phase_timing()
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
            context = entry.get("context") or {}
            job = context.get("job") or {}
            event = {
                "source": level,
                "message": line,
                "step": str(context.get("label") or ""),
                "stream_id": str(
                    context.get("stream_id") or job.get("stream_id") or ""
                ),
                "unit_id": str(
                    context.get("unit_id") or job.get("unit_id") or ""
                ),
                "attempt": int(job.get("attempt") or 0),
            }
            # Keep the compatibility signal for direct runner consumers, but
            # the production facade forwards only the bounded batch signal.
            self.process_log.emit(event)
            self._record_log(level, line, context=event)
            if level == "stdout":
                self._structured(entry, line)

    def _structured(self, entry, line):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return
        if not isinstance(event, dict) or not event.get("event"):
            return
        if event["event"] == "assembly_progress":
            event["parent_span_id"] = str(entry.get("monitor_span_id") or "")
        try:
            self._record_structured_history(event)
        except Exception as history_error:
            self._monitor_history_incomplete = True
            entry["forced_error"] = entry.get("forced_error", "") or (
                "monitor history persistence failed: " + str(history_error)
            )
            self.log_line.emit(
                "stderr",
                "[monitor-history] structured event was not recorded: "
                + str(history_error),
            )
        if event["event"] == "work_package_finished":
            # A completed Package proves the persistent worker reached useful
            # work; only consecutive process crashes count toward the guard.
            self._accelerator_crash_count = 0
        if event["event"] == "validation_finished":
            controller = getattr(self, "_memory_admission", None)
            if controller is not None:
                controller.observe_worker_peak(event.get("peak_rss_bytes", 0))
        self.stream_progress.emit(event)
        self._queue_stream_progress(event)
        current = int(event.get("current") or 0)
        total = int(event.get("total") or 0)
        job = entry["context"].get("job")
        if job:
            self._pending_job_progress[str(job["job_id"])] = (
                str(job["lease_token"]),
                current,
                total,
            )
        self._pending_pipeline_progress = (
            current,
            total,
            str(event.get("unit_id") or event.get("tile_id") or event["event"]),
        )

    def _record_structured_history(self, event):
        """Persist only attempt and phase transitions, never per-item progress."""

        execution_id = str(getattr(self, "_execution_id", "") or "")
        if not execution_id or self._database is None:
            return
        name = str(event.get("event") or "")
        run_id = str(self._spec.get("run_id") or event.get("run_id") or "")
        package_id = str(event.get("package_id") or "")
        stream_id = str(event.get("stream_id") or "")
        parent_span_id = str(event.get("parent_span_id") or "")
        if event.get("execution_id") and str(event["execution_id"]) != execution_id:
            return
        if name == "package_model_loading" and package_id and stream_id:
            model_spans = getattr(self, "_model_monitor_spans", {})
            self._model_monitor_spans = model_spans
            key = (package_id, stream_id, parent_span_id)
            if key not in self._model_monitor_spans:
                self._model_monitor_spans[key] = self._database.start_monitor_span(
                    run_id,
                    execution_id=execution_id,
                    parent_span_id=parent_span_id,
                    job_id=event.get("job_id"),
                    span_kind="package_model",
                    object_type="model_in_package",
                    object_id=f"{package_id}:{stream_id}",
                    stream_id=stream_id,
                    package_id=package_id,
                    model_id=stream_id.split(":", 1)[-1],
                    idempotency_key=(
                        f"model:{execution_id}:{parent_span_id}:{package_id}:{stream_id}"
                    ),
                    metadata={"configured_batch_size": event.get("configured_batch_size")},
                )
        elif name in {"package_model_outputs_reused", "package_model_completed"} and package_id and stream_id:
            span_id = self._model_monitor_spans.pop((package_id, stream_id, parent_span_id), "")
            if span_id:
                self._database.finish_monitor_span(
                    span_id,
                    status="reused" if name.endswith("reused") else "completed",
                    message="复用产物并校验通过" if name.endswith("reused") else "模型计算与产物写入完成",
                    metadata={key: event[key] for key in ("configured_tile_batch_size", "effective_tile_batch_size", "tile_count", "inference_sec") if key in event},
                )
        elif name == "work_package_finished" and package_id:
            for key, span_id in tuple(self._model_monitor_spans.items()):
                if key[0] == package_id and key[2] == parent_span_id:
                    self._database.finish_monitor_span(span_id, status="interrupted", message="模型缺少独立完成记录；不根据包完成推断")
                    self._model_monitor_spans.pop(key, None)
        elif name in {"work_package_failed", "work_package_interrupted"} and package_id:
            status = "failed" if name.endswith("failed") else "interrupted"
            for key, span_id in tuple(self._model_monitor_spans.items()):
                if key[0] == package_id and key[2] == parent_span_id:
                    self._database.finish_monitor_span(
                        span_id, status=status, message=str(event.get("error") or "")
                    )
                    self._model_monitor_spans.pop(key, None)
        elif name == "assembly_progress" and stream_id:
            phase = str(event.get("phase") or "")
            assembly_spans = getattr(self, "_assembly_monitor_spans", {})
            self._assembly_monitor_spans = assembly_spans
            previous = assembly_spans.get(stream_id)
            if previous and previous[0] != phase:
                self._database.finish_monitor_span(previous[1], status="interrupted", message="缺少独立完成记录；不根据后续阶段推断")
                self._assembly_monitor_spans.pop(stream_id, None)
            if phase and stream_id not in self._assembly_monitor_spans:
                span_id = self._database.start_monitor_span(
                    run_id,
                    execution_id=execution_id,
                    span_kind="assembly_phase",
                    parent_span_id=parent_span_id,
                    object_type="stream_phase",
                    object_id=f"{stream_id}:{phase}",
                    stream_id=stream_id,
                    phase=phase,
                    idempotency_key=(
                        f"assembly:{execution_id}:{parent_span_id}:{stream_id}:{phase}"
                    ),
                    metadata={"phase_name": str(event.get("phase_name") or "")},
                )
                self._assembly_monitor_spans[stream_id] = (phase, span_id)
            event_status = str(event.get("status") or "running")
            if event_status in {"completed", "reused", "skipped", "failed"}:
                current = self._assembly_monitor_spans.pop(stream_id, None)
                if current:
                    self._database.finish_monitor_span(
                        current[1],
                        status=event_status,
                        message=str(event.get("message") or ""),
                        metadata={
                            "current": int(event.get("current") or 0),
                            "total": int(event.get("total") or 0),
                            "feature_count": int(event.get("feature_count") or 0),
                        },
                    )

        important = {
            "package_model_outputs_reused",
            "package_model_completed",
            "package_tile_batch_reduced",
            "work_package_finished",
            "work_package_failed",
            "work_package_interrupted",
            "assembly_progress",
            "stream_coverage_validation",
            "work_package_paused_low_disk",
            "accelerator_worker_paused_low_disk",
        }
        status = str(event.get("status") or "")
        if name not in important or (
            name == "assembly_progress" and status == "running"
        ):
            return
        severity = "error" if status == "failed" or name.endswith("failed") else (
            "warning" if "reduced" in name or "paused" in name else "info"
        )
        discriminator = ":".join(
            str(event.get(key) or "")
            for key in ("parent_span_id", "phase", "current", "total", "attempt", "effective_batch_size", "pause_count")
        )
        self._database.append_monitor_event(
            run_id,
            name,
            execution_id=execution_id,
            span_id=parent_span_id,
            job_id=event.get("job_id"),
            level=severity,
            object_type=(
                "package" if package_id else "stream" if stream_id else "run"
            ),
            object_id=package_id or stream_id or run_id,
            stream_id=stream_id,
            package_id=package_id,
            unit_id=str(event.get("unit_id") or ""),
            message=str(event.get("message") or event.get("error") or name),
            payload={
                key: value
                for key, value in event.items()
                if key not in {"lease_token", "state_db", "dsn", "environment"}
            },
            idempotency_key=(
                f"event:{execution_id}:{name}:{package_id}:{stream_id}:{discriminator}"
            ),
        )

    @staticmethod
    def _stream_progress_key(event):
        name = str(event.get("event") or "")
        stream_id = str(event.get("stream_id") or "")
        if name.startswith(("package_", "work_package_", "accelerator_worker_")):
            return "package", str(event.get("package_id") or "active")
        if stream_id:
            return "stream", stream_id
        return "global", name

    def _queue_stream_progress(self, event):
        name = str(event.get("event") or "").lower()
        status = str(event.get("status") or "").lower()
        priority = (
            name.endswith(("_failed", "_warning", "_paused_low_disk"))
            or status in {"failed", "error", "warning"}
            or event.get("success") is False
        )
        value = dict(event)
        if priority:
            self._priority_stream_progress.append(value)
            return
        self._pending_stream_progress[self._stream_progress_key(value)] = value

    @pyqtSlot()
    def _flush_ui_events(self):
        logs = self._pending_ui_logs
        self._pending_ui_logs = []
        if logs:
            self.ui_log_batch.emit(logs)
        stream_events = [
            *self._priority_stream_progress,
            *self._pending_stream_progress.values(),
        ]
        self._priority_stream_progress = []
        self._pending_stream_progress = {}
        progress = self._pending_pipeline_progress
        self._pending_pipeline_progress = None
        if stream_events or progress is not None:
            self.ui_progress_batch.emit(
                {"stream_events": stream_events, "pipeline_progress": progress}
            )

    def _flush_one_job_heartbeat(self, job, *, allow_database_fallback):
        job_id = str(job["job_id"])
        pending = self._pending_job_progress.pop(job_id, None)
        if pending is None:
            if not allow_database_fallback:
                return
            current = self._database.get_job(job_id)
            if not current or current["status"] != "running":
                return
            lease_token = str(job["lease_token"])
            progress_current = int(current["progress_current"] or 0)
            progress_total = int(current["progress_total"] or 0)
        else:
            lease_token, progress_current, progress_total = pending
        self._database.heartbeat(
            job_id,
            lease_token,
            current=progress_current,
            total=progress_total,
            lease_seconds=300,
        )

    @pyqtSlot()
    def _flush_job_heartbeats(self):
        if not self._running:
            return
        for entry in tuple(self._processes.values()):
            job = (entry.get("context") or {}).get("job")
            if job:
                self._flush_one_job_heartbeat(
                    job,
                    allow_database_fallback=True,
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
        phase_timing = getattr(self, "_phase_timing", None)
        if phase_timing is not None:
            phase_timing.finish(token, time.time())
            self._persist_phase_timing()
        label = context["label"]
        success = int(exit_code) == 0 and not entry["forced_error"]
        error = entry["forced_error"] or ("" if success else f"{label} failed (rc={int(exit_code)})")
        monitor_span_id = str(entry.get("monitor_span_id") or "")
        if monitor_span_id:
            assembly = getattr(self, "_assembly_monitor_spans", {}).pop(
                str(context.get("stream_id") or ""), None
            )
            if assembly:
                self._database.finish_monitor_span(
                    assembly[1], status="interrupted" if success else "failed",
                    message=error or "进程结束但缺少阶段完成记录",
                )
            self._database.finish_monitor_span(
                monitor_span_id,
                status="completed" if success else "failed",
                message=error,
                metadata={"exit_code": int(exit_code)},
            )

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
            QTimer.singleShot(0, self._schedule_safely)
            return

        if context.get("kind") == "job":
            job = context["job"]
            self._flush_one_job_heartbeat(
                job,
                allow_database_fallback=False,
            )
            memory_shed = bool(entry.get("memory_shed"))
            current = self._database.get_job(job["job_id"])
            if current and current["status"] == "running":
                if job.get("job_type") in {
                    "fragmentation_v33", "unit_confidence"
                } and success:
                    success = False
                    if job.get("job_type") == "fragmentation_v33":
                        error = (
                            "V3.3 worker exited without its atomic output commit"
                        )
                    else:
                        error = (
                            "unit confidence worker exited without its atomic "
                            "output commit"
                        )
                self._database.finish_job(
                    job["job_id"],
                    job["lease_token"],
                    status="ready" if success else "failed",
                    error=error,
                )
            # A pressure-shed job was atomically moved to ``interrupted``
            # before SIGTERM.  It is already resumable and must not consume a
            # retry attempt or surface as a terminal worker failure.
            retried = memory_shed
            if (
                not success
                and not memory_shed
                and int(entry.get("timeout_count", 0)) < 2
            ):
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
            QTimer.singleShot(0, self._schedule_safely)
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

    @pyqtSlot()
    def _heartbeat_and_watchdog(self):
        if not self._running:
            return
        timeout = float(
            (self._spec.get("scaling") or {}).get("max_partition_runtime_sec", 900)
        )
        now = time.time()
        self._phase_timing.observe_active(now)
        self._persist_phase_timing()
        for entry in list(self._processes.values()):
            context = entry["context"]
            job = context.get("job")
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

    @staticmethod
    def _perform_released_artifact_cleanup(spec):
        database = run_state_from_spec(spec)
        candidates = database.cleanup_candidates(
            spec["run_id"],
            limit=1000,
            kinds=(
                "partition_probability", "v3_context_core", "v3_baseline_core",
                "v33_staged_mask", "v33_staged_audit", "unit_confidence",
            ),
        )
        missing = []
        for candidate in candidates:
            claimed = database.claim_artifact_cleanup(candidate["artifact_id"])
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
                    missing.append(str(path))
                if not database.finish_artifact_cleanup(
                    claimed["artifact_id"], success=True
                ):
                    raise RuntimeError(
                        "temporary Artifact cleanup state changed: " + str(path)
                    )
            except Exception:
                database.finish_artifact_cleanup(
                    claimed["artifact_id"], success=False
                )
                raise
        return missing

    def _cleanup_released_artifacts(self):
        """Poll one independent cleanup worker without blocking Qt or QProcess."""

        future = getattr(self, "_cleanup_future", None)
        if future is not None:
            if not future.done():
                return
            self._cleanup_future = None
            for path in future.result():
                self.log_line.emit(
                    "system",
                    "[cleanup-missing] unreferenced temporary Artifact: " + path,
                )
        executor = getattr(self, "_cleanup_executor", None)
        if executor is None:
            return
        now = time.monotonic()
        if now - float(getattr(self, "_last_cleanup_started_at", 0.0)) < 2.0:
            return
        self._last_cleanup_started_at = now
        self._cleanup_future = executor.submit(
            self._perform_released_artifact_cleanup,
            dict(self._spec),
        )

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

    @pyqtSlot(str, str)
    def _persist_log(self, level, message):
        self._record_log(level, message)

    def _record_log(self, level, message, *, context=None):
        record = {
            "timestamp": time.time(),
            "run_dir": str(self._spec.get("run_dir") or ""),
            "source": str(level),
            "message": str(message),
        }
        record.update(dict(context or {}))
        self._pending_ui_logs.append(record)

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
        self._heartbeat_timer.stop()
        self._ui_flush_timer.stop()
        phase_timing = getattr(self, "_phase_timing", None)
        if phase_timing is not None:
            phase_timing.finish_active(time.time())
            self._persist_phase_timing()
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
            "phase_timing": (
                phase_timing.summary(time.time())
                if phase_timing is not None
                else PipelinePhaseTiming().summary(time.time())
            ),
            "deployment_identity": self._spec.get("deployment_identity") or {},
        }
        if self._manual_package_reset:
            result["manual_package_reset"] = dict(self._manual_package_reset)
        run_dir = Path(self._spec["run_dir"])
        scale_report = run_dir / "logs" / "scale_acceptance_report.json"
        if scale_report.is_file():
            result["scale_acceptance_report"] = str(scale_report)
            result["scale_acceptance_report_sha256"] = sha256_file(scale_report)
            try:
                scale_value = json.loads(scale_report.read_text(encoding="utf-8"))
                observation = (
                    (scale_value.get("storage") or {}).get(
                        "final_artifact_size_observation"
                    )
                    or {}
                )
                if isinstance(observation, dict):
                    result["final_artifact_size_observation"] = observation
            except (OSError, ValueError):
                pass
        size_observation_message = _final_artifact_size_observation_log_message(
            result.get("final_artifact_size_observation")
        )
        if size_observation_message:
            self.log_line.emit("system", f"[final-artifact-size] {size_observation_message}")
        counts = self._database.job_counts(self._spec["run_id"])
        if success:
            self._database.set_run_status(
                self._spec["run_id"], "ready", expected=("running", "raster_ready")
            )
        history_complete = True
        execution_id = str(getattr(self, "_execution_id", "") or "")
        if execution_id and hasattr(self._database, "finish_monitor_execution"):
            try:
                self._database.finish_monitor_execution(
                    self._spec["run_id"],
                    execution_id,
                    status="completed" if success else "stopped" if self._stopped else "failed",
                    message=str(error),
                    recording_complete=not bool(
                        getattr(self, "_monitor_history_incomplete", False)
                    ),
                )
            except Exception as history_error:
                history_complete = False
                self.log_line.emit(
                    "stderr",
                    "[monitor-history] execution history is incomplete: "
                    + str(history_error),
                )
        result["monitor_execution_id"] = execution_id
        history_complete = history_complete and not bool(
            getattr(self, "_monitor_history_incomplete", False)
        )
        result["monitor_history_complete"] = history_complete
        atomic_write_json(run_dir / "run_manifest.json", result)
        run_report = {
            "schema_version": 2,
            "run_id": self._spec["run_id"],
            "status": result["status"],
            "success": bool(success),
            "error": str(error),
            "elapsed_sec": result["elapsed_sec"],
            "phase_timing": result["phase_timing"],
            "deployment_identity": result["deployment_identity"],
            "tile_grid": self._spec.get("tile_grid") or {},
            "spatial_plan_summary": self._spec.get("spatial_plan_summary") or {},
            "storage_preflight": self._spec.get("storage_preflight") or {},
            "final_artifact_size_observation": result.get(
                "final_artifact_size_observation"
            ) or {},
            "job_counts": counts,
            "artifact_cleanup": self._database.artifact_cleanup_summary(
                self._spec["run_id"]
            ),
            "ready_stream_ids": [item["stream_id"] for item in ready_streams],
            "monitor_execution_id": execution_id,
            "monitor_history_complete": history_complete,
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
        self._record_startup_index(result["status"])
        self._pending_job_progress.clear()
        self._flush_ui_events()
        self.log_run_finalized.emit(str(self._spec.get("run_dir") or ""))
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


class ThreadedV5AsyncInferenceRunner(QObject):
    """GUI-safe facade for the v5 runtime and its process-control event loop."""

    log_line = pyqtSignal(str, str)
    process_log = pyqtSignal(object)
    log_batch = pyqtSignal(object)
    step_started = pyqtSignal(str)
    step_finished = pyqtSignal(str, int, dict)
    pipeline_progress = pyqtSignal(int, int, str)
    stage_progress = pyqtSignal(object)
    stream_progress = pyqtSignal(object)
    stream_progress_batch = pyqtSignal(object)
    pipeline_finished = pyqtSignal(dict)
    shutdown_finished = pyqtSignal()

    _start_requested = pyqtSignal(object)
    _stop_requested = pyqtSignal()
    _runtime_shutdown_requested = pyqtSignal()
    _log_shutdown_requested = pyqtSignal()

    REQUIRED_SCRIPTS = V5AsyncInferenceRunner.REQUIRED_SCRIPTS

    def __init__(self, scripts_dir: str, parent=None):
        super().__init__(parent)
        self.scripts_dir = str(Path(scripts_dir).expanduser().resolve())
        self._running = False
        self._shutdown = False

        self._runtime_thread = QThread(self)
        self._runtime_thread.setObjectName("loess-v5-runtime-control")
        self._worker = V5AsyncInferenceRunner(self.scripts_dir, parent=None)
        self._worker.moveToThread(self._runtime_thread)
        self._start_requested.connect(self._worker.execute_request)
        self._stop_requested.connect(self._worker.stop)
        self._runtime_shutdown_requested.connect(self._worker.shutdown_runtime)
        self._worker.ui_log_batch.connect(self._relay_log_batch)
        self._worker.ui_progress_batch.connect(self._relay_progress_batch)
        self._worker.step_started.connect(self._relay_step_started)
        self._worker.step_finished.connect(self._relay_step_finished)
        self._worker.stage_progress.connect(self._relay_stage_progress)
        self._worker.pipeline_finished.connect(self._relay_finished)
        self._worker.runtime_start_failed.connect(self._relay_finished)

        self._log_thread = QThread(self)
        self._log_thread.setObjectName("loess-v5-pipeline-log")
        self._log_writer = _BatchedPipelineLogWriter()
        self._log_writer.moveToThread(self._log_thread)
        self._log_thread.started.connect(self._log_writer.start)
        self._worker.ui_log_batch.connect(self._log_writer.append_batch)
        self._worker.log_run_finalized.connect(self._log_writer.finalize_run)
        self._log_shutdown_requested.connect(self._log_writer.shutdown)

        self._runtime_thread.finished.connect(self._worker.deleteLater)
        # Chain shutdown between the owning threads; never wait in a GUI slot.
        self._runtime_thread.finished.connect(self._log_writer.shutdown)
        self._log_thread.finished.connect(self._log_writer.deleteLater)
        self._log_thread.finished.connect(self._on_shutdown_finished)
        self._runtime_thread.start()
        self._log_thread.start()

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
        if self._shutdown:
            raise RuntimeError("inference runtime has already shut down")
        if self._running:
            raise RuntimeError("an inference pipeline is already running")
        if reset_failed_packages and not resume:
            raise RuntimeError("failed Package reset requires resume validation")
        self._running = True
        self._start_requested.emit(
            {
                "run_spec_path": str(Path(run_spec_path).expanduser().resolve()),
                "resume": bool(resume),
                "reset_failed_packages": bool(reset_failed_packages),
            }
        )

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
        if self._shutdown or not self._running:
            return
        self._stop_requested.emit()

    def cleanup(self):
        self.shutdown()

    def shutdown(self, timeout_ms=None):
        """Request shutdown and return immediately; observe shutdown_finished."""
        del timeout_ms  # Retained for callers of the former blocking API.
        if self._shutdown:
            return
        self._shutdown = True
        if self._runtime_thread.isRunning():
            self._runtime_shutdown_requested.emit()
        elif self._log_thread.isRunning():
            self._log_shutdown_requested.emit()
        else:
            self._on_shutdown_finished()

    @pyqtSlot()
    def _on_shutdown_finished(self):
        # QThread.finished precedes thread-local teardown. A zero-time join
        # never blocks the GUI and prevents deleting an owner whose native
        # thread is still unwinding Python/Qt resources.
        if not self._runtime_thread.wait(0) or not self._log_thread.wait(0):
            QTimer.singleShot(10, self._on_shutdown_finished)
            return
        self._running = False
        self.shutdown_finished.emit()

    @pyqtSlot(object)
    def _relay_log_batch(self, records):
        self.log_batch.emit(list(records or ()))

    @pyqtSlot(object)
    def _relay_progress_batch(self, payload):
        value = dict(payload or {})
        events = list(value.get("stream_events") or ())
        if events:
            self.stream_progress_batch.emit(events)
        progress = value.get("pipeline_progress")
        if progress is not None:
            current, total, message = progress
            self.pipeline_progress.emit(int(current), int(total), str(message))

    @pyqtSlot(str)
    def _relay_step_started(self, name):
        self.step_started.emit(str(name))

    @pyqtSlot(str, int, dict)
    def _relay_step_finished(self, name, return_code, result):
        self.step_finished.emit(str(name), int(return_code), dict(result or {}))

    @pyqtSlot(object)
    def _relay_stage_progress(self, info):
        self.stage_progress.emit(dict(info or {}))

    @pyqtSlot(object)
    def _relay_finished(self, result):
        self._running = False
        self.pipeline_finished.emit(dict(result or {}))
