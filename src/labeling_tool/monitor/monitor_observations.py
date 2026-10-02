"""Pure owners for inference-monitor runner observations and log deduplication."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, TypedDict

from labeling_tool.monitor.monitor_logs import log_fingerprint, log_presentation
from labeling_tool.monitor.monitor_progress import (
    stage_from_step,
    stream_from_step,
    stream_progress_view,
)
from labeling_tool.monitor.monitor_time import elapsed_text, timestamp_epoch


class StreamObservationView(TypedDict):
    state: dict[str, object]
    runtime: dict[str, object]
    coverage: dict[str, object]
    phase_statuses: dict[str, object]


class PresentedLog(TypedDict):
    text: str
    source: str
    severity: str
    title: str
    affected: str
    system_action: str
    user_action: str
    fingerprint: str
    context_key: str
    event_timestamp: object


@dataclass(frozen=True)
class PendingLog:
    level: str
    message: str
    context: Mapping[str, object] | None = None


@dataclass(frozen=True)
class TileUpdate:
    stream_id: str
    tile_id: str
    state: Mapping[str, object]


@dataclass(frozen=True)
class RunnerChange:
    refresh_stream_ids: tuple[str, ...] = ()
    tile_updates: tuple[TileUpdate, ...] = ()
    pending_logs: tuple[PendingLog, ...] = ()
    coverage_changed: bool = False


@dataclass(frozen=True)
class SnapshotChange:
    refresh_stream_ids: tuple[str, ...]


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


class MonitorObservations:
    """Own and interpret transient runner and persisted snapshot observations."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._stream_state: dict[str, dict[str, Any]] = {}
        self._tile_state: dict[str, dict[str, dict[str, Any]]] = {}
        self._step_started_at: dict[str, float] = {}
        self._step_attempts: dict[str, int] = {}
        self._active_stream_stages: dict[str, dict[str, int]] = {}
        self._active_global_stage = ""
        self._active_inference_stream = ""
        self._package_activity: dict[str, Any] = {}
        self._runtime_progress: dict[str, dict[str, Any]] = {}
        self._coverage_state: dict[str, dict[str, Any]] = {}
        self._assembly_phase_statuses: dict[str, dict[str, Any]] = {}
        self._run_created_epoch: float | None = None
        self._terminal_run_status = ""

    def begin_binding(self) -> None:
        self._terminal_run_status = ""
        self._run_created_epoch = None

    def end_binding(self) -> None:
        self._run_created_epoch = None

    def mark_terminal(self, status: str) -> None:
        self._terminal_run_status = str(status)

    @property
    def terminal_status(self) -> str:
        return self._terminal_run_status

    @property
    def run_created_epoch(self) -> float | None:
        return self._run_created_epoch

    @property
    def active_global_stage(self) -> str:
        return self._active_global_stage

    def stream_ids(self) -> tuple[str, ...]:
        return tuple(self._stream_state)

    def has_stream(self, stream_id: str) -> bool:
        return str(stream_id) in self._stream_state

    def stream_view(self, stream_id: str) -> StreamObservationView:
        key = str(stream_id)
        return StreamObservationView(
            state=deepcopy(self._stream_state.get(key) or {}),
            runtime=deepcopy(self._runtime_progress.get(key) or {}),
            coverage=deepcopy(self._coverage_state.get(key) or {}),
            phase_statuses=deepcopy(self._assembly_phase_statuses.get(key) or {}),
        )

    def package_view(self) -> dict[str, object]:
        return deepcopy(self._package_activity)

    def coverage_view(self) -> dict[str, dict[str, object]]:
        return deepcopy(self._coverage_state)

    def tiles_snapshot(self, stream_id: str) -> dict[str, dict[str, object]]:
        return deepcopy(self._tile_state.get(str(stream_id)) or {})

    def attempt_for(self, affected: str) -> int:
        return int(self._step_attempts.get(str(affected)) or 0)

    def observe_preparation_progress(
        self, *, stream_id: str, name: str, current: int, total: int
    ) -> RunnerChange:
        refresh: list[str] = []
        self._set_stream(
            str(stream_id),
            refresh,
            stage=str(name),
            progress=f"{int(current)}/{int(total)}" if int(total) else "-",
        )
        return RunnerChange(refresh_stream_ids=tuple(refresh))

    def observe_step_started(self, name: str, *, epoch_now: float) -> RunnerChange:
        step = str(name)
        stream_id = stream_from_step(step)
        stage = stage_from_step(step)
        self._step_started_at[step] = float(epoch_now)
        self._step_attempts[step] = int(self._step_attempts.get(step) or 0) + 1
        self._active_global_stage = stage
        refresh: list[str] = []
        if stream_id:
            stage_counts = self._active_stream_stages.setdefault(stream_id, {})
            stage_counts[stage] = int(stage_counts.get(stage, 0)) + 1
            self._set_stream(stream_id, refresh, stage=stage, status="运行中")
        return RunnerChange(refresh_stream_ids=tuple(refresh))

    def observe_step_finished(
        self,
        name: str,
        return_code: int,
        result: Mapping[str, object],
        *,
        epoch_now: float,
        database_bound: bool,
    ) -> RunnerChange:
        step = str(name)
        value: dict[str, Any] = dict(result)
        pending_logs: list[PendingLog] = []
        failed = not value.get("success") and not value.get("skipped")
        if failed:
            pending_logs.append(
                PendingLog(
                    level="system",
                    message=json.dumps(
                        {
                            "event": "monitor_step_failed",
                            "step": step,
                            "stream_id": str(
                                value.get("stream_id") or stream_from_step(step)
                            ),
                            "attempt": int(self._step_attempts.get(step) or 1),
                            "return_code": int(return_code),
                            "error": str(value.get("error") or "任务执行失败"),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                )
            )

        stream_id = str(value.get("stream_id") or stream_from_step(step))
        stage = stage_from_step(step)
        started = self._step_started_at.pop(step, None)
        elapsed = (
            float(epoch_now) - started
            if started
            else float(value.get("elapsed_sec") or 0)
        )
        if self._active_global_stage == stage:
            self._active_global_stage = ""
        if not stream_id:
            return RunnerChange(pending_logs=tuple(pending_logs))

        stage_counts = self._active_stream_stages.get(stream_id) or {}
        remaining = max(0, int(stage_counts.get(stage, 0)) - 1)
        if remaining:
            stage_counts[stage] = remaining
        else:
            stage_counts.pop(stage, None)
        if not stage_counts:
            self._active_stream_stages.pop(stream_id, None)

        status = (
            "成功"
            if value.get("success")
            else "跳过"
            if value.get("skipped")
            else "失败"
        )
        changes: dict[str, Any] = {"elapsed": f"{elapsed:.1f}s"}
        if not database_bound:
            changes["status"] = status
        elif status == "失败":
            changes.update({"status": "失败", "stage": "任务失败"})
        if status == "失败":
            changes["failures"] = (
                int(self._stream_state.get(stream_id, {}).get("failures", 0)) + 1
            )
        refresh: list[str] = []
        self._set_stream(stream_id, refresh, **changes)
        return RunnerChange(
            refresh_stream_ids=tuple(refresh), pending_logs=tuple(pending_logs)
        )

    def observe_stream_progress(
        self,
        info: Mapping[str, object],
        *,
        database_bound: bool,
        configured_batch_size: int,
        fusion_profile_id: str,
        epoch_now: float,
        monotonic_now: float,
    ) -> RunnerChange:
        value: dict[str, Any] = dict(info)
        event = str(value.get("event") or "")
        refresh: list[str] = []
        if event.startswith(("package_", "work_package_", "accelerator_worker_")):
            self._update_package_activity(
                value,
                refresh,
                configured_batch_size=int(configured_batch_size),
                fusion_profile_id=str(fusion_profile_id),
                epoch_now=float(epoch_now),
                monotonic_now=float(monotonic_now),
            )
            if database_bound:
                return RunnerChange(refresh_stream_ids=tuple(refresh))

        stream_id = str(value.get("stream_id") or "")
        if not stream_id:
            return RunnerChange(refresh_stream_ids=tuple(refresh))
        current = int(value.get("current") or 0)
        total = int(value.get("total") or 0)
        failure = str(value.get("error") or "")
        coverage_changed = False
        tile_updates: list[TileUpdate] = []

        if event == "assembly_progress":
            progress_status = str(value.get("status") or "running")
            status = {"completed": "成功", "failed": "失败"}.get(
                progress_status, "运行中"
            )
            self._runtime_progress[stream_id] = dict(value)
            self._set_stream(
                stream_id,
                refresh,
                stage=str(value.get("phase_name") or "并行组装"),
                stage_progress=(
                    f"{current}/{total}"
                    if total
                    else f"步骤 {int(value.get('phase_index') or 0)}/"
                    f"{int(value.get('phase_total') or 0)}"
                ),
                activity="—",
                feature_count=int(value.get("feature_count") or 0),
                elapsed=elapsed_text(float(value.get("elapsed_sec") or 0)),
                status=status,
                failures=int(self._stream_state.get(stream_id, {}).get("failures", 0))
                + (1 if progress_status == "failed" else 0),
            )
            return RunnerChange(refresh_stream_ids=tuple(refresh))

        if event == "stream_coverage_validation":
            self._coverage_state[stream_id] = dict(value)
            coverage_status = str(value.get("status") or "")
            passed = coverage_status == "passed"
            failed = coverage_status == "failed"
            self._set_stream(
                stream_id,
                refresh,
                stage="空白/重叠验收",
                stage_progress="1/1",
                status="成功" if passed else "失败" if failed else "未验证",
                failures=int(self._stream_state.get(stream_id, {}).get("failures", 0))
                + (1 if failed else 0),
            )
            coverage_changed = True
            return RunnerChange(
                refresh_stream_ids=tuple(refresh), coverage_changed=coverage_changed
            )

        status = "失败" if event.endswith("failed") else "运行中"
        self._set_stream(
            stream_id,
            refresh,
            progress=f"{current}/{total}" if total else "-",
            status=status,
            failures=int(self._stream_state.get(stream_id, {}).get("failures", 0))
            + (1 if failure else 0),
        )
        tile_id_value = value.get("tile_id")
        if tile_id_value:
            tile_id = str(tile_id_value)
            state: dict[str, Any] = {
                "status": (
                    "失败"
                    if failure
                    else "完成"
                    if event.endswith(("completed", "reused"))
                    else "运行中"
                ),
                "progress": f"{current}/{total}" if total else "-",
                "error": failure,
            }
            self._tile_state.setdefault(stream_id, {})[tile_id] = state
            tile_updates.append(
                TileUpdate(stream_id=stream_id, tile_id=tile_id, state=deepcopy(state))
            )
        return RunnerChange(
            refresh_stream_ids=tuple(refresh), tile_updates=tuple(tile_updates)
        )

    def observe_pipeline_finished(self, result: Mapping[str, object]) -> RunnerChange:
        if result.get("terminal_published") is False:
            return RunnerChange()
        refresh: list[str] = []
        result_value: dict[str, Any] = dict(result)
        for stream in result_value.get("streams") or []:
            value: dict[str, Any] = dict(stream)
            raw_status = value.get("status")
            status_map: dict[object, object] = {
                "ready": "成功",
                "failed": "失败",
                "stopped": "已停止",
            }
            status = status_map.get(raw_status, raw_status)
            self._set_stream(
                str(value["stream_id"]),
                refresh,
                status=status,
                failures=int(value.get("failure_count") or 0),
            )
        return RunnerChange(refresh_stream_ids=tuple(refresh))

    def apply_snapshot(
        self, snapshot: Mapping[str, object], *, epoch_now: float
    ) -> SnapshotChange | None:
        value: Any = snapshot
        run_row = value.get("run") or {}
        run_status = str(run_row.get("status") or "planned")
        if self._terminal_run_status and run_status != self._terminal_run_status:
            return None
        if self._run_created_epoch is None:
            self._run_created_epoch = timestamp_epoch(run_row.get("created_at") or "")

        job_counts = value.get("job_counts") or {}
        package_counts = job_counts.get("work_package") or {}
        active_package = value.get("active_work_package")
        if active_package is not None:
            package_id = str(active_package.get("package_id") or "")
            attempt = int(active_package.get("attempt") or 0)
            previous_package = str(self._package_activity.get("package_id") or "")
            previous_attempt = self._package_activity.get("attempt")
            attempt_changed = (
                previous_attempt is not None and int(previous_attempt) != attempt
            )
            if package_id != previous_package or attempt_changed:
                self._active_inference_stream = ""
                self._package_activity = {
                    "package_id": package_id,
                    "attempt": attempt,
                    "status": "运行中",
                }
            self._package_activity.update(
                {
                    "package_id": package_id,
                    "attempt": attempt,
                    "sequence_no": int(active_package.get("sequence_no") or 0),
                    "db_current": int(active_package.get("progress_current") or 0),
                    "db_total": int(active_package.get("progress_total") or 0),
                    "started_epoch": timestamp_epoch(
                        active_package.get("package_started_at") or ""
                    ),
                }
            )
            observed = json.loads(
                str(active_package.get("monitor_runtime_json") or "{}")
            )
            observed_at = timestamp_epoch(observed.get("observed_at") or "") or 0
            if observed_at > float(
                self._package_activity.get("_event_observed_at") or 0
            ):
                self._package_activity.update(observed)
                self._active_inference_stream = str(observed.get("stream_id") or "")
        elif int(package_counts.get("running", 0)) == 0:
            self._active_inference_stream = ""

        old_runtime_ids = tuple(self._runtime_progress)
        old_phase_ids = tuple(self._assembly_phase_statuses)
        old_coverage_ids = tuple(self._coverage_state)
        all_runtime_progress = value.get("stream_runtime_progress") or {}
        self._runtime_progress = {
            str(key): dict(item) for key, item in all_runtime_progress.items()
        }
        all_phase_statuses = value.get("assembly_phase_statuses") or {}
        self._assembly_phase_statuses = {
            str(key): dict(item) for key, item in all_phase_statuses.items()
        }
        persisted_coverage = value.get("stream_coverage_validation") or {}
        coverage_replaced = bool(persisted_coverage)
        if coverage_replaced:
            self._coverage_state = {
                str(key): dict(item) for key, item in persisted_coverage.items()
            }

        refresh: list[str] = []
        streams = value.get("streams") or []
        all_type_counts = value.get("stream_unit_type_counts") or {}
        all_job_type_counts = value.get("stream_unit_job_type_counts") or {}
        for stream in streams:
            stream_value = dict(stream)
            stream_id = str(stream_value["stream_id"])
            type_counts = all_type_counts.get(stream_id) or {}
            durable_counts: dict[str, int] = {}
            for counts in type_counts.values():
                for state, count in counts.items():
                    durable_counts[state] = int(durable_counts.get(state, 0)) + int(
                        count
                    )
            progress = stream_progress_view(
                run_status=run_status,
                stream=stream_value,
                durable_counts=durable_counts,
                job_type_counts=all_job_type_counts.get(stream_id) or {},
                package_counts=package_counts,
                inference_active=(
                    int(package_counts.get("running", 0)) > 0
                    and stream_id == self._active_inference_stream
                ),
                active_stage=self._active_stage_for_stream(stream_id),
                assembly_info=all_runtime_progress.get(stream_id) or {},
                assembly_phase_statuses=self._assembly_phase_statuses.get(stream_id)
                or {},
                previous_elapsed=str(
                    self._stream_state.get(stream_id, {}).get("elapsed", "-")
                ),
                now=float(epoch_now),
            )
            self._set_stream(stream_id, refresh, **asdict(progress))

        supplemental_ids = (
            old_runtime_ids
            + tuple(self._runtime_progress)
            + old_phase_ids
            + tuple(self._assembly_phase_statuses)
        )
        if coverage_replaced:
            supplemental_ids += old_coverage_ids + tuple(self._coverage_state)
        for stream_id in supplemental_ids:
            if self.has_stream(stream_id):
                _append_unique(refresh, stream_id)
        return SnapshotChange(refresh_stream_ids=tuple(refresh))

    def _ensure_stream(self, stream_id: str, refresh: list[str]) -> dict[str, Any]:
        key = str(stream_id)
        if key not in self._stream_state:
            self._stream_state[key] = {
                "stage": "等待计划",
                "progress": "-",
                "unit_progress": "-",
                "stage_progress": "-",
                "activity": "0/0",
                "feature_count": None,
                "status": "等待",
                "elapsed": "-",
                "failures": 0,
            }
            self._tile_state.setdefault(key, {})
        _append_unique(refresh, key)
        return self._stream_state[key]

    def _set_stream(
        self, stream_id: str, refresh: list[str], **changes: object
    ) -> None:
        state = self._ensure_stream(str(stream_id), refresh)
        state.update(
            {key: item for key, item in changes.items() if state.get(key) != item}
        )

    def _active_stage_for_stream(self, stream_id: str) -> str:
        stage_counts = self._active_stream_stages.get(str(stream_id)) or {}
        for stage in ("并行组装", "Accepted 差分", "边界矢量化", "空间单元拟合"):
            if int(stage_counts.get(stage, 0)) > 0:
                return stage
        return next(iter(stage_counts), "")

    def _update_package_activity(
        self,
        info: Mapping[str, object],
        refresh: list[str],
        *,
        configured_batch_size: int,
        fusion_profile_id: str,
        epoch_now: float,
        monotonic_now: float,
    ) -> None:
        value: dict[str, Any] = dict(info)
        event = str(value.get("event") or "")
        package_id = str(value.get("package_id") or "")
        stream_id = str(value.get("stream_id") or "")
        previous_package = str(self._package_activity.get("package_id") or "")
        if package_id and package_id != previous_package:
            self._active_inference_stream = ""
            self._package_activity = {
                "package_id": package_id,
                "started_at": float(monotonic_now),
                "status": "运行中",
            }
        if package_id:
            self._package_activity["package_id"] = package_id
        self._package_activity["_event_observed_at"] = float(epoch_now)
        if stream_id:
            self._package_activity["stream_id"] = stream_id
            self._active_inference_stream = stream_id
            if configured_batch_size:
                self._package_activity["configured_batch_size"] = int(
                    configured_batch_size
                )
                self._package_activity.setdefault(
                    "effective_batch_size", int(configured_batch_size)
                )
            self._set_stream(
                stream_id, refresh, stage="Work Package 推理", status="运行中"
            )
        if event == "package_model_loading":
            self._package_activity.update(
                {
                    "model_current": int(value.get("current") or 0),
                    "model_total": int(value.get("total") or 0),
                    "tile_current": 0,
                    "tile_total": 0,
                    "status": "模型加载/推理",
                }
            )
        elif event in ("package_tile_materialized", "package_tile_completed"):
            tile_current = int(value.get("current") or 0)
            tile_total = int(value.get("total") or 0)
            status = "Tile 物化" if event.endswith("materialized") else "模型推理"
            if event == "package_tile_materialized" and tile_current <= 1:
                self._active_inference_stream = ""
                for key in (
                    "stream_id",
                    "model_current",
                    "model_total",
                    "configured_batch_size",
                    "effective_batch_size",
                    "notice",
                    "elapsed_sec",
                ):
                    self._package_activity.pop(key, None)
            if (
                event == "package_tile_completed"
                and tile_total > 0
                and tile_current >= tile_total
                and int(self._package_activity.get("model_current") or 0)
                >= int(self._package_activity.get("model_total") or 0)
            ):
                status = "Fusion / Package 收口"
            self._package_activity.update(
                {
                    "tile_current": tile_current,
                    "tile_total": tile_total,
                    "status": status,
                }
            )
            if status == "Fusion / Package 收口" and fusion_profile_id:
                fusion_stream = f"fusion:{fusion_profile_id}"
                self._package_activity["stream_id"] = fusion_stream
                self._active_inference_stream = fusion_stream
                self._set_stream(
                    fusion_stream,
                    refresh,
                    stage="Work Package Fusion / 收口",
                    status="运行中",
                )
        elif event == "package_tile_batch_reduced":
            self._package_activity.update(
                {
                    "effective_batch_size": int(value.get("effective_batch_size") or 0),
                    "status": "Batch 降档后重试",
                    "notice": "OOM 降档",
                }
            )
        elif event == "package_model_outputs_reused":
            self._package_activity.update({"status": "复用已有模型结果"})
        elif event == "package_tiles_cleaned":
            self._package_activity.update({"status": "缓存清理/提交"})
        elif event == "work_package_finished":
            self._package_activity.update(
                {
                    "status": "已完成",
                    "elapsed_sec": float(value.get("elapsed_sec") or 0),
                    "notice": "",
                }
            )
            self._active_inference_stream = ""
        elif event == "accelerator_worker_finished":
            self._active_inference_stream = ""
        elif event == "accelerator_worker_paused_low_disk":
            self._package_activity.update(
                {"status": "低磁盘暂停", "notice": "等待磁盘空间"}
            )
        elif event.endswith("failed"):
            self._package_activity.update(
                {"status": "失败", "notice": str(value.get("error") or "")}
            )
            self._active_inference_stream = ""


class MonitorLogObservations:
    """Own rich/legacy pairing and displayed-error deduplication."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._process_log_suppressions: dict[tuple[str, str], int] = {}
        self._logged_error_texts: set[str] = set()

    def clear_logged_errors(self) -> None:
        self._logged_error_texts.clear()

    def observe_process_log(
        self,
        event: Mapping[str, object],
        *,
        attempt_for: Callable[[str], int],
    ) -> PresentedLog:
        info: dict[str, Any] = dict(event)
        level = str(info.get("source") or "system")
        message = str(info.get("message") or "")
        key = (level, message)
        self._process_log_suppressions[key] = (
            int(self._process_log_suppressions.get(key) or 0) + 1
        )
        presented = self.observe_log(
            level, message, context=info, attempt_for=attempt_for
        )
        assert presented is not None
        return presented

    def observe_log(
        self,
        level: object,
        message: object,
        *,
        context: Mapping[str, object] | None,
        attempt_for: Callable[[str], int],
    ) -> PresentedLog | None:
        if context is None:
            key = (str(level), str(message))
            count = int(self._process_log_suppressions.get(key) or 0)
            if count:
                if count == 1:
                    self._process_log_suppressions.pop(key, None)
                else:
                    self._process_log_suppressions[key] = count - 1
                return None
        presentation = log_presentation(level, message)
        log_context: dict[str, Any] = dict(context or {})
        affected = str(
            log_context.get("step")
            or log_context.get("unit_id")
            or log_context.get("stream_id")
            or presentation.get("affected")
            or ""
        )
        attempt = int(log_context.get("attempt") or presentation.get("attempt") or 0)
        if not attempt and affected:
            attempt = int(attempt_for(affected) or 0)
        context_key = f"{affected}:attempt={attempt}" if affected else "unscoped"
        fingerprint = log_fingerprint(
            presentation["severity"], presentation["error"], affected, attempt
        )
        if presentation["severity"] == "error":
            self._logged_error_texts.add(self._normalized_error(presentation["error"]))
        return PresentedLog(
            text=str(message),
            source=str(presentation["source"]),
            severity=str(presentation["severity"]),
            title=str(presentation["title"]),
            affected=affected,
            system_action=str(presentation["system_action"]),
            user_action=str(presentation["user_action"]),
            fingerprint=fingerprint,
            context_key=context_key,
            event_timestamp=log_context.get("timestamp"),
        )

    def observe_pipeline_failure(
        self,
        result: Mapping[str, object],
        *,
        attempt_for: Callable[[str], int],
    ) -> PresentedLog | None:
        value = dict(result)
        error = value.get("error")
        if (
            value.get("success")
            or value.get("status") == "stopped"
            or not error
            or self._normalized_error(error) in self._logged_error_texts
        ):
            return None
        message = json.dumps(
            {
                "event": (
                    "monitor_attempt_failed"
                    if value.get("terminal_published") is False
                    else "monitor_pipeline_failed"
                ),
                "error": str(error),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return self.observe_log(
            "system", message, context=None, attempt_for=attempt_for
        )

    @staticmethod
    def _normalized_error(error: object) -> str:
        return re.sub(r"\s+", " ", str(error)).strip().lower()
