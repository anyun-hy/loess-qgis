"""Persist one pipeline execution's low-frequency monitor history."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from labeling_tool.shared.state.monitor_history_repository import (
        MonitorHistoryRepository,
    )


_IMPORTANT_EVENTS = {
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
_PRIVATE_PAYLOAD_FIELDS = {"lease_token", "state_db", "dsn", "environment"}
_RUNTIME_PHASE_KINDS = {"finalize_rasters", "assemble", "scale_acceptance"}


class RunHistoryRecorder:
    """Own monitor spans and event persistence for exactly one Execution."""

    def __init__(
        self,
        history_repository: MonitorHistoryRepository,
        run_id: str,
        execution_id: str,
    ) -> None:
        self._history = history_repository
        self._run_id = str(run_id)
        self._execution_id = str(execution_id)
        if not self._run_id or not self._execution_id:
            raise ValueError("run and execution IDs are required for monitor history")
        self._model_spans: dict[tuple[str, str, str], str] = {}
        self._assembly_spans: dict[str, tuple[str, str]] = {}

    def record(self, event: Mapping[str, Any]) -> None:
        """Record one structured transition from the bound Execution."""
        value = dict(event)
        if (
            value.get("execution_id")
            and str(value["execution_id"]) != self._execution_id
        ):
            return
        name = str(value.get("event") or "")
        package_id = str(value.get("package_id") or "")
        stream_id = str(value.get("stream_id") or "")
        parent_span_id = str(value.get("parent_span_id") or "")
        self._record_model_span(
            value,
            name=name,
            package_id=package_id,
            stream_id=stream_id,
            parent_span_id=parent_span_id,
        )
        self._record_assembly_span(
            value,
            name=name,
            stream_id=stream_id,
            parent_span_id=parent_span_id,
        )
        self._record_important_event(
            value,
            name=name,
            package_id=package_id,
            stream_id=stream_id,
            parent_span_id=parent_span_id,
        )

    def start_process(
        self,
        token: str,
        label: str,
        context: Mapping[str, Any],
    ) -> str:
        """Start a runtime-phase span for a process that has such a phase."""
        kind = str(context.get("kind") or "")
        if kind not in _RUNTIME_PHASE_KINDS:
            return ""
        stream_id = str(context.get("stream_id") or "")
        return str(
            self._history.start_span(
                self._run_id,
                execution_id=self._execution_id,
                span_kind="runtime_phase",
                object_type="stream" if stream_id else "run",
                object_id=stream_id or self._run_id,
                stream_id=stream_id,
                phase=kind,
                idempotency_key=f"process:{self._execution_id}:{token}",
                metadata={"label": str(label)},
            )
        )

    def finish_process(
        self,
        span_id: str,
        stream_id: str,
        *,
        success: bool,
        error: str,
        exit_code: int,
    ) -> None:
        """Finish the process span without inventing an assembly-phase success."""
        if not span_id:
            return
        assembly = self._assembly_spans.pop(str(stream_id), None)
        if assembly:
            self._history.finish_span(
                assembly[1],
                status="interrupted" if success else "failed",
                message=str(error) or "进程结束但缺少阶段完成记录",
            )
        self._history.finish_span(
            str(span_id),
            status="completed" if success else "failed",
            message=str(error),
            metadata={"exit_code": int(exit_code)},
        )

    def _record_model_span(
        self,
        event: Mapping[str, Any],
        *,
        name: str,
        package_id: str,
        stream_id: str,
        parent_span_id: str,
    ) -> None:
        key = (package_id, stream_id, parent_span_id)
        if name == "package_model_loading" and package_id and stream_id:
            if key not in self._model_spans:
                self._model_spans[key] = self._history.start_span(
                    self._run_id,
                    execution_id=self._execution_id,
                    parent_span_id=parent_span_id,
                    job_id=event.get("job_id"),
                    span_kind="package_model",
                    object_type="model_in_package",
                    object_id=f"{package_id}:{stream_id}",
                    stream_id=stream_id,
                    package_id=package_id,
                    model_id=stream_id.split(":", 1)[-1],
                    idempotency_key=(
                        f"model:{self._execution_id}:{parent_span_id}:"
                        f"{package_id}:{stream_id}"
                    ),
                    metadata={
                        "configured_batch_size": event.get("configured_batch_size")
                    },
                )
            return
        if (
            name in {"package_model_outputs_reused", "package_model_completed"}
            and package_id
            and stream_id
        ):
            span_id = self._model_spans.pop(key, "")
            if span_id:
                self._history.finish_span(
                    span_id,
                    status="reused" if name.endswith("reused") else "completed",
                    message=(
                        "复用产物并校验通过"
                        if name.endswith("reused")
                        else "模型计算与产物写入完成"
                    ),
                    metadata={
                        field: event[field]
                        for field in (
                            "configured_tile_batch_size",
                            "effective_tile_batch_size",
                            "tile_count",
                            "inference_sec",
                        )
                        if field in event
                    },
                )
            return
        if name == "work_package_finished" and package_id:
            self._finish_package_model_spans(
                package_id,
                parent_span_id,
                status="interrupted",
                message="模型缺少独立完成记录；不根据包完成推断",
            )
            return
        if name in {"work_package_failed", "work_package_interrupted"} and package_id:
            self._finish_package_model_spans(
                package_id,
                parent_span_id,
                status="failed" if name.endswith("failed") else "interrupted",
                message=str(event.get("error") or ""),
            )

    def _finish_package_model_spans(
        self,
        package_id: str,
        parent_span_id: str,
        *,
        status: str,
        message: str,
    ) -> None:
        for key, span_id in tuple(self._model_spans.items()):
            if key[0] == package_id and key[2] == parent_span_id:
                self._history.finish_span(
                    span_id,
                    status=status,
                    message=message,
                )
                self._model_spans.pop(key, None)

    def _record_assembly_span(
        self,
        event: Mapping[str, Any],
        *,
        name: str,
        stream_id: str,
        parent_span_id: str,
    ) -> None:
        if name != "assembly_progress" or not stream_id:
            return
        phase = str(event.get("phase") or "")
        previous = self._assembly_spans.get(stream_id)
        if previous and previous[0] != phase:
            self._history.finish_span(
                previous[1],
                status="interrupted",
                message="缺少独立完成记录；不根据后续阶段推断",
            )
            self._assembly_spans.pop(stream_id, None)
        if phase and stream_id not in self._assembly_spans:
            span_id = self._history.start_span(
                self._run_id,
                execution_id=self._execution_id,
                span_kind="assembly_phase",
                parent_span_id=parent_span_id,
                object_type="stream_phase",
                object_id=f"{stream_id}:{phase}",
                stream_id=stream_id,
                phase=phase,
                idempotency_key=(
                    f"assembly:{self._execution_id}:{parent_span_id}:"
                    f"{stream_id}:{phase}"
                ),
                metadata={"phase_name": str(event.get("phase_name") or "")},
            )
            self._assembly_spans[stream_id] = (phase, span_id)
        event_status = str(event.get("status") or "running")
        if event_status in {"completed", "reused", "skipped", "failed"}:
            current = self._assembly_spans.pop(stream_id, None)
            if current:
                self._history.finish_span(
                    current[1],
                    status=event_status,
                    message=str(event.get("message") or ""),
                    metadata={
                        "current": int(event.get("current") or 0),
                        "total": int(event.get("total") or 0),
                        "feature_count": int(event.get("feature_count") or 0),
                    },
                )

    def _record_important_event(
        self,
        event: Mapping[str, Any],
        *,
        name: str,
        package_id: str,
        stream_id: str,
        parent_span_id: str,
    ) -> None:
        status = str(event.get("status") or "")
        if name not in _IMPORTANT_EVENTS or (
            name == "assembly_progress" and status == "running"
        ):
            return
        severity = (
            "error"
            if status == "failed" or name.endswith("failed")
            else "warning"
            if "reduced" in name or "paused" in name
            else "info"
        )
        discriminator = ":".join(
            str(event.get(key) or "")
            for key in (
                "parent_span_id",
                "phase",
                "current",
                "total",
                "attempt",
                "effective_batch_size",
                "pause_count",
            )
        )
        self._history.append_event(
            self._run_id,
            name,
            execution_id=self._execution_id,
            span_id=parent_span_id,
            job_id=event.get("job_id"),
            level=severity,
            object_type=("package" if package_id else "stream" if stream_id else "run"),
            object_id=package_id or stream_id or self._run_id,
            stream_id=stream_id,
            package_id=package_id,
            unit_id=str(event.get("unit_id") or ""),
            message=str(event.get("message") or event.get("error") or name),
            payload={
                key: value
                for key, value in event.items()
                if key not in _PRIVATE_PAYLOAD_FIELDS
            },
            idempotency_key=(
                f"event:{self._execution_id}:{name}:{package_id}:"
                f"{stream_id}:{discriminator}"
            ),
        )
