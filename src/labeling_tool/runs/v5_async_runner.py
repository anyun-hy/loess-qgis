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

from labeling_tool.qgis_support.process_runtime import (
    configure_process,
    process_is_running,
)
from labeling_tool.runs.manual_package_reset import reset_failed_work_packages
from labeling_tool.runs.memory_admission import (
    AdaptiveMemoryAdmissionController,
    MemoryAdmissionDecision,
    MemoryPressureSample,
)
from labeling_tool.runs.monitor_history import RunHistoryRecorder
from labeling_tool.runs.recovery_contract import validate_recovery_run
from labeling_tool.runs.run_index import record_run_state
from labeling_tool.runs.run_job_scheduler import (
    RunJobScheduler,
    geometry_thread_budget,
    minimum_geometry_slots,
    resource_value,
)
from labeling_tool.runs.run_job_scheduler import (
    fragmentation_v33_process_threads as _fragmentation_v33_process_threads,
)
from labeling_tool.runs.run_job_scheduler import (
    unit_fit_process_threads as _unit_fit_process_threads,
)
from labeling_tool.runs.run_terminal_finalizer import (
    RunTerminalContext,
    RunTerminalStateConflictError,
    build_run_result,
    finalize_run_outputs,
)
from labeling_tool.runs.ui_event_buffer import UIEventBuffer
from labeling_tool.shared.contracts.monitor_contract import MONITOR_EXECUTION_ENV
from labeling_tool.shared.contracts.run_spec import atomic_write_json, sha256_file
from labeling_tool.shared.state.run_execution_ownership import (
    RUN_OWNER_ENV,
    RunExecutionIdentity,
    RunExecutionOwnership,
    RunOwnershipConflictError,
    RunOwnershipLostError,
)
from labeling_tool.shared.state.run_state_db import run_state_from_spec


def process_thread_environment_values(spec, context):
    job = context.get("job") or {}
    if job.get("job_type") == "fragmentation_v33":
        threads = _fragmentation_v33_process_threads(spec)
    elif (
        context.get("kind") == "accelerator_worker"
        or job.get("job_type") == "work_package"
    ):
        threads = resource_value(spec, "package_process_threads", 2)
    elif job.get("job_type") in {"unit_fit", "unit_confidence"}:
        threads = _unit_fit_process_threads(spec)
    else:
        threads = resource_value(spec, "assembly_process_threads", 1)
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
        self._run_ownership = None
        self._monitor_history = None
        self._running = False
        self._stopped = False
        self._phase = "idle"
        self._worker_id = f"qgis-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._accelerator_worker_id = self._worker_id + "-accelerator"
        self._job_scheduler: RunJobScheduler | None = None
        self._processes = {}
        self._assembly_queue = []
        self._started_at = 0.0
        self._phase_timing = PipelinePhaseTiming()
        self._manual_package_reset = {}
        self._monitor_history = None
        self._monitor_history_incomplete = False
        self._memory_admission = AdaptiveMemoryAdmissionController()
        self._last_memory_admission_log_at = 0.0
        self._ui_event_buffer = UIEventBuffer()
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
        trigger_type = (
            "redo_failed_packages"
            if reset_failed_packages
            else "resume" if resume else "start"
        )
        recovered_package_jobs = self._claim_and_prepare_execution(
            trigger_type=trigger_type,
            resume=bool(resume),
            reset_failed_packages=bool(reset_failed_packages),
        )
        self._record_startup_index("running")
        self._running = True
        self._stopped = False
        self._phase = "jobs"
        self._processes.clear()
        self._assembly_queue = []
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
                run_row = self._database.run_streams.get_run(self._spec["run_id"]) or {}
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

    def _claim_and_prepare_execution(
        self,
        *,
        trigger_type: str,
        resume: bool,
        reset_failed_packages: bool,
    ) -> int:
        """Acquire the Run before recovery, reset, or any Job mutation."""

        self._execution_id = ""
        self._run_ownership = None
        ownership = RunExecutionOwnership.acquire(
            self._database,
            self._spec["run_id"],
            run_dir=self._spec["run_dir"],
            worker_id=self._worker_id,
            trigger_type=trigger_type,
        )
        self._run_ownership = ownership
        self._execution_id = ownership.identity.execution_id
        self._accelerator_worker_id = self._worker_id + "-accelerator"
        self._job_scheduler = RunJobScheduler(
            self._spec,
            self._database.jobs,
            worker_id=self._worker_id,
            accelerator_worker_id=self._accelerator_worker_id,
            ownership_guard=ownership.assert_current,
        )
        try:
            recovered_package_jobs = (
                self._job_scheduler.recover_for_resume() if resume else 0
            )
            self._monitor_history = RunHistoryRecorder(
                self._database.monitor_history,
                self._spec["run_id"],
                self._execution_id,
            )
            self._manual_package_reset = {}
            self._phase_timing = self._load_phase_timing()
            if reset_failed_packages:
                with ownership.publication_barrier():
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
            entered_running = self._database.run_streams.set_run_status(
                self._spec["run_id"],
                "running",
                expected=(
                    "planned",
                    "running",
                    "stopped",
                    "failed",
                    "raster_ready",
                ),
            )
            if not entered_running:
                raise RuntimeError(
                    "Run state changed before execution; it may have been archived"
                )
            return int(recovered_package_jobs)
        except Exception as error:
            self._seal_execution_attempt_failure(str(error))
            self._release_run_ownership()
            raise

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
            message = f"{type(error).__name__}: {error}"
            self._seal_execution_attempt_failure(message)
            self._release_run_ownership()
            result = self._attempt_failure_result(
                message,
                status=(
                    "ownership_conflict"
                    if isinstance(error, RunOwnershipConflictError)
                    else "attempt_failed"
                ),
            )
            result["run_id"] = str(spec.get("run_id") or "")
            result["run_spec"] = str(value.get("run_spec_path") or "")
            result["run_dir"] = str(spec.get("run_dir") or "")
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
        self._assembly_queue.clear()
        self._scheduler.stop()
        self._watchdog.stop()
        self._heartbeat_timer.stop()
        entries = list(self._processes.values())
        for entry in entries:
            self._terminate_entry(entry, graceful=True)
        for entry in entries:
            context = entry["context"]
            try:
                if context.get("kind") == "accelerator_worker":
                    self._job_scheduler.interrupt_accelerator(context["worker_id"])
                    continue
                job = context.get("job")
                if job:
                    self._job_scheduler.interrupt_job(job)
            except RunOwnershipLostError as error:
                self._processes.clear()
                self._abort_lost_ownership(str(error))
                return
        self._processes.clear()
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

    def _seal_execution_attempt_failure(self, message: str) -> None:
        ownership = getattr(self, "_run_ownership", None)
        if ownership is None or not self._execution_id:
            return
        try:
            ownership.assert_current()
            self._database.monitor_history.finish_execution(
                self._spec["run_id"],
                self._execution_id,
                status="failed",
                message=str(message),
                recording_complete=False,
            )
        except Exception:
            pass

    def _release_run_ownership(self) -> None:
        ownership = getattr(self, "_run_ownership", None)
        self._run_ownership = None
        if ownership is not None:
            try:
                ownership.close()
            except Exception:
                # The local identity is retired even if the broken database
                # socket reports another error while closing.
                pass

    def _attempt_failure_result(self, error: str, *, status: str) -> dict:
        return {
            "schema_version": 2,
            "run_id": str(self._spec.get("run_id") or ""),
            "run_spec": str(self._spec_path or ""),
            "run_dir": str(self._spec.get("run_dir") or ""),
            "success": False,
            "status": str(status),
            "error": str(error),
            "ready_streams": [],
            "failed_streams": [],
            "streams": [],
            "terminal_published": False,
        }

    def _abort_lost_ownership(self, error: str) -> None:
        """Stop local work without claiming a terminal state for the Run."""

        if not self._running and self._run_ownership is None:
            return
        self._scheduler.stop()
        self._watchdog.stop()
        self._heartbeat_timer.stop()
        self._ui_flush_timer.stop()
        self._assembly_queue.clear()
        entries = list(self._processes.values())
        for entry in entries:
            self._terminate_entry(entry, graceful=False)
        self._processes.clear()
        self._pending_job_progress.clear()
        self._running = False
        self._release_run_ownership()
        message = "Run execution ownership was lost; this attempt stopped: " + str(error)
        self.log_line.emit("stderr", "[run-ownership-lost] " + str(error))
        self._flush_ui_events()
        self.pipeline_finished.emit(
            self._attempt_failure_result(message, status="ownership_lost")
        )

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
        minimum_geometry_slots=1,
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
            minimum_geometry_slots=minimum_geometry_slots,
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
        if not self._job_scheduler.interrupt_job(job):
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
        except RunOwnershipLostError as error:
            self._abort_lost_ownership(str(error))
        except Exception as error:
            message = f"[scheduler-error] {type(error).__name__}: {error}"
            self.log_line.emit("stderr", message)
            self._finish(False, message)

    def _schedule(self):
        if not self._running or self._stopped:
            return
        if self._phase != "jobs":
            return
        accelerator_active = any(
            entry["context"].get("kind") == "accelerator_worker"
            for entry in self._processes.values()
        )
        cycle = self._job_scheduler.begin_cycle(
            accelerator_active=accelerator_active
        )
        if cycle.terminal_error:
            self._finish(False, cycle.terminal_error)
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
        static_geometry_limit = geometry_thread_budget(
            self._spec,
            package_active=cycle.package_expected,
        )
        minimum_slots = minimum_geometry_slots(self._spec)
        memory_decision = self._memory_admission_decision(
            static_limit=static_geometry_limit,
            active_jobs=active_jobs,
            package_active=cycle.package_expected,
            minimum_geometry_slots=minimum_slots,
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

        dispatch = self._job_scheduler.dispatch(
            cycle,
            active_jobs=[job for job in active_jobs if job],
            accelerator_active=accelerator_active,
            geometry_slot_limit=memory_decision.geometry_slot_limit,
            start_accelerator=self._start_accelerator_worker,
            start_job=self._start_job,
        )
        if dispatch.terminal_error:
            self._finish(False, dispatch.terminal_error)
            return
        if dispatch.active:
            boundary_enabled = bool(
                (self._spec.get("boundary_fitting") or {}).get("enabled", True)
            )
            geometry_stage = (
                "公共分界线拟合中" if boundary_enabled else "原始类别边界组装中"
            )
            self._emit_progress(f"有界 Work Package / {geometry_stage}")
            return
        if dispatch.blocked_counts is not None:
            if memory_decision.geometry_slot_limit < min(
                static_geometry_limit, minimum_slots
            ):
                self._emit_progress("等待内存预算恢复，暂时无法容纳一个几何任务")
                return
            self._finish(
                False,
                "v5 job graph has blocked dependencies: "
                + str(dispatch.blocked_counts),
            )
            return
        if dispatch.complete:
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

    def _start_assembly_safely(self):
        try:
            self._start_assembly()
        except RunOwnershipLostError as error:
            self._abort_lost_ownership(str(error))

    def _start_assembly(self):
        if not self._running or self._stopped or self._phase != "assembly":
            return
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
            QTimer.singleShot(0, self._start_acceptance_safely)

    def _start_acceptance_safely(self):
        try:
            self._start_acceptance()
        except RunOwnershipLostError as error:
            self._abort_lost_ownership(str(error))

    def _start_acceptance(self):
        """Continue from the one assembly pass to acceptance.

        Historical completed Runs can still be repaired explicitly with the
        standalone fragmentation script.  New v5 Runs never launch it or let
        it replace the formal assembled GPKG.
        """
        if not self._running or self._stopped or self._phase != "assembly":
            return
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
        with self._database.owner_publication(
            self._spec["run_id"], self._spec["run_dir"]
        ):
            atomic_write_json(
                self._phase_timing_path(),
                self._phase_timing.state(time.time()),
            )

    def _start_process(self, label, script, arguments, context):
        ownership = getattr(self, "_run_ownership", None)
        if ownership is None:
            raise RunOwnershipLostError("Run execution has no active owner")
        ownership.assert_current()
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
            environment.insert(RUN_OWNER_ENV, ownership.identity.environment_value())
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
        if self._monitor_history is not None:
            monitor_span_id = self._monitor_history.start_process(
                token, label, context
            )
            if monitor_span_id:
                entry["monitor_span_id"] = monitor_span_id
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
            if self._monitor_history is not None:
                self._monitor_history.record(event)
        except RunOwnershipLostError as ownership_error:
            self._abort_lost_ownership(str(ownership_error))
            return
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
            self._job_scheduler.record_work_package_finished()
        if event["event"] == "validation_finished":
            controller = getattr(self, "_memory_admission", None)
            if controller is not None:
                controller.observe_worker_peak(event.get("peak_rss_bytes", 0))
        self.stream_progress.emit(event)
        self._ui_event_buffer.enqueue_stream_progress(event)
        current = int(event.get("current") or 0)
        total = int(event.get("total") or 0)
        job = entry["context"].get("job")
        if job:
            self._pending_job_progress[str(job["job_id"])] = (
                str(job["lease_token"]),
                current,
                total,
            )
        self._ui_event_buffer.set_pipeline_progress(
            current,
            total,
            str(event.get("unit_id") or event.get("tile_id") or event["event"]),
        )

    @pyqtSlot()
    def _flush_ui_events(self):
        logs = self._ui_event_buffer.take_logs()
        if logs:
            self.ui_log_batch.emit(logs)
        stream_events, progress = self._ui_event_buffer.take_progress()
        if stream_events or progress is not None:
            self.ui_progress_batch.emit(
                {"stream_events": stream_events, "pipeline_progress": progress}
            )

    def _flush_one_job_heartbeat(self, job, *, allow_database_fallback):
        job_id = str(job["job_id"])
        pending = self._pending_job_progress.pop(job_id, None)
        self._job_scheduler.heartbeat(
            job,
            pending,
            allow_database_fallback=allow_database_fallback,
        )

    @pyqtSlot()
    def _flush_job_heartbeats(self):
        if not self._running:
            return
        try:
            for entry in tuple(self._processes.values()):
                job = (entry.get("context") or {}).get("job")
                if job:
                    self._flush_one_job_heartbeat(
                        job,
                        allow_database_fallback=True,
                    )
        except RunOwnershipLostError as error:
            self._abort_lost_ownership(str(error))

    def _process_finished(self, token, exit_code, _exit_status):
        try:
            self._process_finished_owned(token, exit_code, _exit_status)
        except RunOwnershipLostError as error:
            self._abort_lost_ownership(str(error))

    def _process_finished_owned(self, token, exit_code, _exit_status):
        entry = self._processes.get(token)
        if not entry or not self._running:
            return
        try:
            self._run_ownership.assert_current()
        except RunOwnershipLostError as error:
            self._abort_lost_ownership(str(error))
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
        if self._monitor_history is not None:
            self._monitor_history.finish_process(
                monitor_span_id,
                str(context.get("stream_id") or ""),
                success=success,
                error=error,
                exit_code=int(exit_code),
            )

        if context.get("kind") == "accelerator_worker":
            worker_id = context["worker_id"]
            completion = self._job_scheduler.complete_accelerator(
                worker_id,
                success=success,
                error=error,
            )
            self.step_finished.emit(
                label,
                int(exit_code),
                {
                    "success": completion.success,
                    "error": completion.error,
                    "stream_id": "",
                },
            )
            if completion.terminal_error:
                self._finish(False, completion.terminal_error)
                return
            if completion.restart_attempt:
                self.log_line.emit(
                    "system",
                    "[accelerator-restart] "
                    f"attempt={completion.restart_attempt} "
                    f"error={completion.error}",
                )
            self._emit_progress(label)
            if completion.should_schedule:
                QTimer.singleShot(0, self._schedule_safely)
            return

        if context.get("kind") == "job":
            job = context["job"]
            self._flush_one_job_heartbeat(
                job,
                allow_database_fallback=False,
            )
            completion = self._job_scheduler.complete_job(
                job,
                success=success,
                error=error,
                memory_shed=bool(entry.get("memory_shed")),
                timeout_count=int(entry.get("timeout_count", 0)),
            )
            if completion.retried and not entry.get("memory_shed"):
                self.log_line.emit("system", f"[retry] {label}")
            self.step_finished.emit(
                label,
                int(exit_code),
                {
                    "success": completion.signal_success,
                    "error": completion.error,
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
            QTimer.singleShot(0, self._start_assembly_safely)
        elif context.get("kind") == "assemble":
            QTimer.singleShot(0, self._start_assembly_safely)
        elif context.get("kind") == "scale_acceptance":
            self._finish(True, "")

    @pyqtSlot()
    def _heartbeat_and_watchdog(self):
        try:
            self._heartbeat_and_watchdog_owned()
        except RunOwnershipLostError as error:
            self._abort_lost_ownership(str(error))

    def _heartbeat_and_watchdog_owned(self):
        if not self._running:
            return
        try:
            self._run_ownership.assert_current()
        except RunOwnershipLostError as error:
            self._abort_lost_ownership(str(error))
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
                self._run_ownership.assert_current()
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
    def _perform_released_artifact_cleanup(spec, owner_identity):
        database = run_state_from_spec(spec)
        database.session.bind_execution_owner(
            RunExecutionIdentity.from_mapping(owner_identity)
        )
        candidates = database.artifacts.cleanup_candidates(
            spec["run_id"],
            limit=1000,
            kinds=(
                "partition_probability", "v3_context_core", "v3_baseline_core",
                "v33_staged_mask", "v33_staged_audit", "unit_confidence",
            ),
        )
        missing = []
        for candidate in candidates:
            with database.owner_publication(
                spec["run_id"], spec["run_dir"]
            ) as publication:
                claimed = publication.artifacts.claim_artifact_cleanup(
                    candidate["artifact_id"]
                )
                if claimed is None:
                    continue
                path = Path(claimed["path"])
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
                if not publication.artifacts.finish_artifact_cleanup(
                    claimed["artifact_id"], success=True
                ):
                    raise RuntimeError(
                        "temporary Artifact cleanup state changed: " + str(path)
                    )
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
        ownership = getattr(self, "_run_ownership", None)
        if ownership is None:
            raise RunOwnershipLostError("artifact cleanup requires the Run owner")
        self._cleanup_future = executor.submit(
            self._perform_released_artifact_cleanup,
            dict(self._spec),
            ownership.identity.as_mapping(),
        )

    def _emit_progress(self, message):
        counts = self._job_scheduler.job_counts()
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
        self._ui_event_buffer.enqueue_log(record)

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
        self._running = False
        if not success and self._processes:
            entries = list(self._processes.values())
            for entry in entries:
                self._terminate_entry(entry, graceful=False)
            for entry in entries:
                context = entry["context"]
                try:
                    if context.get("kind") == "accelerator_worker":
                        self._job_scheduler.interrupt_accelerator(
                            context["worker_id"]
                        )
                        continue
                    job = context.get("job")
                    if job:
                        self._job_scheduler.interrupt_job(job)
                except RunOwnershipLostError as ownership_error:
                    self._processes.clear()
                    self._abort_lost_ownership(str(ownership_error))
                    return
            self._processes.clear()
        try:
            if phase_timing is not None:
                self._persist_phase_timing()
            context = RunTerminalContext(
                spec=self._spec,
                spec_path=self._spec_path,
                success=bool(success),
                stopped=bool(self._stopped),
                error=str(error),
                started_at=self._started_at,
                manual_package_reset=self._manual_package_reset,
                execution_id=str(getattr(self, "_execution_id", "") or ""),
                monitor_history_incomplete=bool(
                    getattr(self, "_monitor_history_incomplete", False)
                ),
            )
            phase_summary = (
                phase_timing.summary
                if phase_timing is not None
                else PipelinePhaseTiming().summary
            )
            result = build_run_result(
                context,
                clock=time.time,
                phase_summary=phase_summary,
            )
            size_observation_message = _final_artifact_size_observation_log_message(
                result.get("final_artifact_size_observation")
            )
            if size_observation_message:
                self.log_line.emit(
                    "system", f"[final-artifact-size] {size_observation_message}"
                )
            with self._database.owner_publication(
                self._spec["run_id"], self._spec["run_dir"]
            ) as publication:
                result = finalize_run_outputs(
                    context,
                    result,
                    jobs=publication.jobs,
                    artifacts=publication.artifacts,
                    history=publication.monitor_history,
                    set_run_status=publication.run_streams.set_run_status,
                    fail_open_streams=publication.run_streams.fail_open_streams,
                    emit_history_error=lambda message: self.log_line.emit(
                        "stderr",
                        "[monitor-history] execution history is incomplete: "
                        + message,
                    ),
                )
        except RunOwnershipLostError as ownership_error:
            self._abort_lost_ownership(str(ownership_error))
            return
        except RunTerminalStateConflictError as publication_error:
            self._seal_execution_attempt_failure(str(publication_error))
            self._release_run_ownership()
            self._pending_job_progress.clear()
            self._flush_ui_events()
            self.pipeline_finished.emit(
                self._attempt_failure_result(
                    str(publication_error), status="terminal_state_conflict"
                )
            )
            return
        except Exception as publication_error:
            globally_failed = False
            failure_message = "terminal publication failed: " + str(
                publication_error
            )
            try:
                self._run_ownership.assert_current()
                globally_failed = self._database.run_streams.fail_terminal_publication(
                    self._spec["run_id"],
                    failure_message,
                )
                self._seal_execution_attempt_failure(str(publication_error))
            except RunOwnershipLostError as ownership_error:
                self._abort_lost_ownership(str(ownership_error))
                return
            except Exception as fallback_error:
                failure_message += (
                    "; fallback state update failed: " + str(fallback_error)
                )
            finally:
                self._release_run_ownership()
            self._pending_job_progress.clear()
            self._flush_ui_events()
            self.pipeline_finished.emit(
                self._attempt_failure_result(
                    failure_message,
                    status="failed" if globally_failed else "attempt_failed",
                )
            )
            return
        self._record_startup_index(result["status"])
        self._release_run_ownership()
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
