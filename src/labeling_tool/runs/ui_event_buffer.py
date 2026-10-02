"""UI-facing event buffering without Qt ownership or side effects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

UIEvent: TypeAlias = dict[str, Any]
PipelineProgress: TypeAlias = tuple[int, int, str]
ProgressDrain: TypeAlias = tuple[list[UIEvent], PipelineProgress | None]


class UIEventBuffer:
    """Own pending UI logs and coalesced progress until the runner drains them."""

    def __init__(self) -> None:
        self._pending_logs: list[UIEvent] = []
        self._pending_stream_progress: dict[tuple[str, str], UIEvent] = {}
        self._priority_stream_progress: list[UIEvent] = []
        self._pending_pipeline_progress: PipelineProgress | None = None

    def enqueue_log(self, record: Mapping[str, Any]) -> None:
        """Keep every log in arrival order until a UI flush consumes it."""
        self._pending_logs.append(dict(record))

    def enqueue_stream_progress(self, event: Mapping[str, Any]) -> None:
        """Keep priority events and coalesce ordinary updates by their UI key."""
        value = dict(event)
        if self._is_priority(value):
            self._priority_stream_progress.append(value)
            return
        self._pending_stream_progress[self._stream_progress_key(value)] = value

    def set_pipeline_progress(self, current: int, total: int, message: str) -> None:
        """Retain only the latest aggregate pipeline observation."""
        self._pending_pipeline_progress = (int(current), int(total), str(message))

    def take_logs(self) -> list[UIEvent]:
        """Drain logs before signals can synchronously enqueue progress."""
        logs = self._pending_logs
        self._pending_logs = []
        return logs

    def take_progress(self) -> ProgressDrain:
        """Drain priority then ordinary stream events with the latest pipeline value."""
        stream_events = [
            *self._priority_stream_progress,
            *self._pending_stream_progress.values(),
        ]
        self._priority_stream_progress = []
        self._pending_stream_progress = {}
        progress = self._pending_pipeline_progress
        self._pending_pipeline_progress = None
        return stream_events, progress

    @staticmethod
    def _stream_progress_key(event: Mapping[str, Any]) -> tuple[str, str]:
        name = str(event.get("event") or "")
        stream_id = str(event.get("stream_id") or "")
        if name.startswith(("package_", "work_package_", "accelerator_worker_")):
            return "package", str(event.get("package_id") or "active")
        if stream_id:
            return "stream", stream_id
        return "global", name

    @staticmethod
    def _is_priority(event: Mapping[str, Any]) -> bool:
        name = str(event.get("event") or "").lower()
        status = str(event.get("status") or "").lower()
        return (
            name.endswith(("_failed", "_warning", "_paused_low_disk"))
            or status in {"failed", "error", "warning"}
            or event.get("success") is False
        )
