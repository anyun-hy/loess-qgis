"""Qt ownership boundary for monitor-query worker execution."""

from __future__ import annotations

from typing import Mapping, cast

from qgis.PyQt.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot

from labeling_tool.monitor.monitor_query import (
    MonitorQueryCoordinator,
    QueryFailure,
    QueryKind,
    QueryMessage,
)
from labeling_tool.monitor.monitor_query_io import MonitorQueryExecutor


class MonitorQueryWorker(QObject):
    """Invoke executor I/O only from the dedicated query thread."""

    result_ready = pyqtSignal(object)
    query_failed = pyqtSignal(object)

    def __init__(self, executor: MonitorQueryExecutor | None = None) -> None:
        super().__init__(None)
        self._executor = executor or MonitorQueryExecutor()

    @pyqtSlot(object)
    def execute(self, request: object) -> None:
        value = dict(request) if isinstance(request, Mapping) else {}
        try:
            self.result_ready.emit(self._executor.execute(cast(QueryMessage, value)))
        except Exception as error:
            failure: QueryFailure = {
                "kind": cast(QueryKind, str(value.get("kind") or "snapshot")),
                "generation": int(value.get("generation") or 0),
                "request_id": int(value.get("request_id") or 0),
                "run_id": str(value.get("run_id") or ""),
                "error": f"{type(error).__name__}: {error}",
            }
            self.query_failed.emit(failure)


class MonitorQueryClient(QObject):
    """GUI-thread facade for serialized monitor reads and lifecycle ownership."""

    result_ready = pyqtSignal(object, object)
    query_failed = pyqtSignal(object, object)
    shutdown_finished = pyqtSignal()
    _query_requested = pyqtSignal(object)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        executor: MonitorQueryExecutor | None = None,
    ) -> None:
        super().__init__(parent)
        self._coordinator = MonitorQueryCoordinator()
        self._thread = QThread(self)
        self._thread.setObjectName("loess-monitor-postgresql")
        self._worker = MonitorQueryWorker(executor)
        self._worker.moveToThread(self._thread)
        self._query_requested.connect(self._worker.execute)
        self._worker.result_ready.connect(self._on_result)
        self._worker.query_failed.connect(self._on_failure)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._confirm_shutdown)
        self._shutdown_started = False
        self._shutdown_emitted = False
        self._thread.start()

    @property
    def is_bound(self) -> bool:
        return self._coordinator.is_bound

    @property
    def run_id(self) -> str:
        return self._coordinator.run_id

    def run_spec(self) -> dict[str, object]:
        return self._coordinator.run_spec()

    def is_running(self) -> bool:
        return self._thread.isRunning()

    def bind(self, run_id: str, run_spec: Mapping[str, object]) -> None:
        if self._shutdown_started:
            return
        self._coordinator.bind(run_id, run_spec)
        self._dispatch_next()

    def unbind(self) -> None:
        self._coordinator.unbind()

    def fence(self) -> None:
        self._coordinator.fence()

    def queue_snapshot(self) -> int | None:
        request_id = self._coordinator.enqueue_snapshot()
        self._dispatch_next()
        return request_id

    def queue_detail(
        self,
        *,
        stream_id: str,
        detail_kind: str,
        status: str,
        search: str,
        page: int,
        page_size: int,
    ) -> int | None:
        request_id = self._coordinator.enqueue_detail(
            stream_id=stream_id,
            detail_kind=detail_kind,
            status=status,
            search=search,
            page=page,
            page_size=page_size,
        )
        self._dispatch_next()
        return request_id

    def queue_history(
        self,
        *,
        scope: str,
        execution_id: str,
        search: str,
        before_event_id: int | None,
        append: bool,
        page_size: int,
        context: Mapping[str, object],
    ) -> int | None:
        request_id = self._coordinator.enqueue_history(
            scope=scope,
            execution_id=execution_id,
            search=search,
            before_event_id=before_event_id,
            append=append,
            page_size=page_size,
            context=context,
        )
        self._dispatch_next()
        return request_id

    def queue_object_history(
        self,
        *,
        object_id: str,
        detail_kind: str,
        stream_id: str,
        job_id: object,
        span_id: str,
        append: bool,
        before_started_at: str,
        before_span_id: str,
    ) -> int | None:
        request_id = self._coordinator.enqueue_object_history(
            object_id=object_id,
            detail_kind=detail_kind,
            stream_id=stream_id,
            job_id=job_id,
            span_id=span_id,
            append=append,
            before_started_at=before_started_at,
            before_span_id=before_span_id,
        )
        self._dispatch_next()
        return request_id

    def continuation_is_current(
        self,
        kind: QueryKind,
        *,
        generation: int,
        request_id: int,
    ) -> bool:
        return self._coordinator.continuation_is_current(
            kind,
            generation=generation,
            request_id=request_id,
        )

    def _dispatch_next(self) -> None:
        if self._shutdown_started:
            return
        request = self._coordinator.take_next()
        if request is not None:
            self._query_requested.emit(request)

    @pyqtSlot(object)
    def _on_result(self, payload: object) -> None:
        value = dict(payload) if isinstance(payload, Mapping) else {}
        completion = self._coordinator.complete(value)
        if completion is not None and completion.applicable:
            self.result_ready.emit(completion.request, value)
        self._dispatch_next()

    @pyqtSlot(object)
    def _on_failure(self, failure: object) -> None:
        value = dict(failure) if isinstance(failure, Mapping) else {}
        completion = self._coordinator.complete(value)
        if completion is not None and completion.applicable:
            self.query_failed.emit(completion.request, value)
        self._dispatch_next()

    def shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        self._coordinator.unbind()
        if self._thread.isRunning():
            self._thread.quit()
        self._confirm_shutdown()

    @pyqtSlot()
    def _confirm_shutdown(self) -> None:
        if not self._shutdown_started or self._shutdown_emitted:
            return
        if not self._thread.wait(0):
            QTimer.singleShot(10, self._confirm_shutdown)
            return
        self._shutdown_emitted = True
        self.shutdown_finished.emit()
