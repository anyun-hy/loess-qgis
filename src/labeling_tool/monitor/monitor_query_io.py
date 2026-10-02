"""Worker-thread I/O for inference-monitor query messages."""

from __future__ import annotations

from typing import Callable, Mapping, Protocol, Sequence, cast

from labeling_tool.monitor.monitor_logs import read_persisted_log_page
from labeling_tool.monitor.monitor_query import (
    DetailResult,
    HistoryResult,
    ObjectHistoryResult,
    QueryIdentityFields,
    QueryMessage,
    QueryResult,
    RawHistoryResult,
    RawLogPage,
    SnapshotResult,
)
from labeling_tool.shared.contracts.monitor_contract import MONITOR_EVENT_PAGE_SIZE
from labeling_tool.shared.state.run_state_db import run_state_from_spec


class MonitorHistoryReader(Protocol):
    """Read the bounded monitor-history projections for one state database."""

    def page_events(
        self,
        run_id: str,
        *,
        before_event_id: int | None,
        execution_id: str,
        stream_id: str,
        object_id: str,
        job_id: int | None,
        span_id: str,
        scope: str,
        levels: tuple[str, ...],
        search: str,
        limit: int,
    ) -> Sequence[Mapping[str, object]]: ...

    def page_spans(
        self,
        run_id: str,
        *,
        object_id: str = "",
        package_id: str = "",
        stream_id: str = "",
        span_kind: str = "",
        job_id: int | None = None,
        before_started_at: str = "",
        before_span_id: str = "",
        parent_span_id: str = "",
        limit: int = 200,
    ) -> Sequence[Mapping[str, object]]: ...


class MonitorControlGraphReader(Protocol):
    """Read Tile and Stream Unit detail pages from the spatial control graph."""

    def count_tiles(self, run_id: str, *, status: str | None, search: str) -> int: ...

    def page_tiles(
        self,
        run_id: str,
        *,
        limit: int,
        offset: int,
        status: str | None,
        search: str,
    ) -> Sequence[Mapping[str, object]]: ...

    def count_stream_units(
        self,
        run_id: str,
        stream_id: str,
        *,
        status: str,
        search: str,
    ) -> int: ...

    def page_stream_units(
        self,
        run_id: str,
        stream_id: str,
        *,
        limit: int,
        offset: int,
        status: str,
        search: str,
    ) -> Sequence[Mapping[str, object]]: ...


class MonitorReadReader(Protocol):
    def snapshot(self, run_id: str) -> Mapping[str, object]: ...

    def count_objects(
        self,
        run_id: str,
        *,
        kind: str,
        stream_id: str,
        status: str,
        search: str,
    ) -> int: ...

    def page_objects(
        self,
        run_id: str,
        *,
        kind: str,
        stream_id: str,
        limit: int,
        offset: int,
        status: str,
        search: str,
    ) -> Sequence[Mapping[str, object]]: ...


class MonitorDatabase(Protocol):
    @property
    def monitor_history(self) -> MonitorHistoryReader: ...

    @property
    def control_graph(self) -> MonitorControlGraphReader: ...

    @property
    def monitor_read(self) -> MonitorReadReader: ...


DatabaseFactory = Callable[[Mapping[str, object]], MonitorDatabase]


class PersistedLogReader(Protocol):
    def __call__(
        self,
        run_spec: Mapping[str, object],
        severity: str,
        before_event_id: int | None,
        *,
        search: str,
    ) -> RawLogPage: ...


def _integer(value: object, default: int = 0) -> int:
    return (
        int(value)
        if isinstance(value, (int, float, str, bytes, bytearray))
        else default
    )


class MonitorQueryExecutor:
    """Execute bounded monitor reads and cache one database per Run identity."""

    def __init__(
        self,
        database_factory: DatabaseFactory | None = None,
        log_reader: PersistedLogReader | None = None,
    ) -> None:
        self._database_factory = (
            database_factory
            if database_factory is not None
            else cast(DatabaseFactory, run_state_from_spec)
        )
        self._log_reader = (
            log_reader
            if log_reader is not None
            else cast(PersistedLogReader, read_persisted_log_page)
        )
        self._database_key: tuple[str, str] | None = None
        self._database: MonitorDatabase | None = None

    def _database_for(self, request: Mapping[str, object]) -> MonitorDatabase:
        spec = dict(cast(Mapping[str, object], request.get("run_spec") or {}))
        key = (str(request.get("run_id") or ""), str(spec.get("state_db") or ""))
        if key != self._database_key or self._database is None:
            self._database = self._database_factory(spec)
            self._database_key = key
        return self._database

    @staticmethod
    def _identity(request: QueryMessage) -> QueryIdentityFields:
        return {
            "generation": request["generation"],
            "request_id": request["request_id"],
            "run_id": request["run_id"],
        }

    def execute(self, request: QueryMessage) -> QueryResult:
        """Return a plain-value result. Exceptions are handled by the Qt worker."""

        value = dict(request)
        kind = str(value.get("kind") or "")
        identity = self._identity(request)
        run_id = str(identity["run_id"])
        scope = str(value.get("scope") or "all")
        if request["kind"] == "history" and scope in {"raw_warning", "raw_error"}:
            page = self._log_reader(
                request["run_spec"],
                scope.removeprefix("raw_"),
                request["before_event_id"],
                search=str(value.get("search") or ""),
            )
            raw_result: RawHistoryResult = {
                **identity,
                "kind": "history",
                "append": bool(value.get("append")),
                **page,
            }
            return raw_result

        database = self._database_for(value)
        if kind == "snapshot":
            snapshot: SnapshotResult = {
                **identity,
                "kind": "snapshot",
                "snapshot": dict(database.monitor_read.snapshot(run_id)),
            }
            return snapshot
        if kind == "detail":
            return self._detail(database, value, identity)
        if kind == "history":
            return self._history(database, value, identity)
        if kind == "object_history":
            return self._object_history(database, value, identity)
        raise ValueError("unknown monitor query kind: " + kind)

    def _detail(
        self,
        database: MonitorDatabase,
        value: Mapping[str, object],
        identity: QueryIdentityFields,
    ) -> DetailResult:
        run_id = str(identity["run_id"])
        stream_id = str(value.get("stream_id") or "")
        detail_kind = str(value.get("detail_kind") or "unit")
        status = str(value.get("status") or "")
        search = str(value.get("search") or "")
        page_size = max(1, min(_integer(value.get("page_size"), 500), 500))
        page = max(0, _integer(value.get("page")))
        if detail_kind == "tile":
            total = database.control_graph.count_tiles(
                run_id, status=status or None, search=search
            )
        elif detail_kind == "unit":
            total = database.control_graph.count_stream_units(
                run_id, stream_id, status=status, search=search
            )
        else:
            total = database.monitor_read.count_objects(
                run_id,
                kind=detail_kind,
                stream_id=stream_id,
                status=status,
                search=search,
            )
        page_total = max(1, (total + page_size - 1) // page_size)
        page = min(page, page_total - 1)
        offset = page * page_size
        if detail_kind == "tile":
            rows = database.control_graph.page_tiles(
                run_id,
                limit=page_size,
                offset=offset,
                status=status or None,
                search=search,
            )
        elif detail_kind == "unit":
            rows = database.control_graph.page_stream_units(
                run_id,
                stream_id,
                limit=page_size,
                offset=offset,
                status=status,
                search=search,
            )
        else:
            rows = database.monitor_read.page_objects(
                run_id,
                kind=detail_kind,
                stream_id=stream_id,
                limit=page_size,
                offset=offset,
                status=status,
                search=search,
            )
        return {
            **identity,
            "kind": "detail",
            "stream_id": stream_id,
            "detail_kind": detail_kind,
            "status": status,
            "search": search,
            "page": page,
            "page_size": page_size,
            "page_total": page_total,
            "total": int(total),
            "rows": [dict(row) for row in rows],
        }

    def _history(
        self,
        database: MonitorDatabase,
        value: Mapping[str, object],
        identity: QueryIdentityFields,
    ) -> HistoryResult:
        scope = str(value.get("scope") or "all")
        levels = ("warning", "error") if scope in {"issues", "warnings"} else ()
        rows = database.monitor_history.page_events(
            str(identity["run_id"]),
            before_event_id=cast(int | None, value.get("before_event_id")),
            execution_id=str(value.get("execution_id") or ""),
            stream_id=str(value.get("stream_id") or ""),
            object_id=str(value.get("object_id") or ""),
            job_id=cast(int | None, value.get("job_id")),
            span_id=str(value.get("span_id") or ""),
            scope=scope,
            levels=levels,
            search=str(value.get("search") or ""),
            limit=_integer(value.get("page_size"), MONITOR_EVENT_PAGE_SIZE),
        )
        return {
            **identity,
            "kind": "history",
            "append": bool(value.get("append")),
            "rows": [dict(row) for row in rows],
            "page_size": _integer(value.get("page_size"), MONITOR_EVENT_PAGE_SIZE),
        }

    def _object_history(
        self,
        database: MonitorDatabase,
        value: Mapping[str, object],
        identity: QueryIdentityFields,
    ) -> ObjectHistoryResult:
        run_id = str(identity["run_id"])
        object_id = str(value.get("object_id") or "")
        detail_kind = str(value.get("detail_kind") or "")
        spans = database.monitor_history.page_spans(
            run_id,
            object_id="" if detail_kind == "package" else object_id,
            package_id=object_id if detail_kind == "package" else "",
            stream_id=str(value.get("stream_id") or "")
            if detail_kind != "package"
            else "",
            span_kind="job_attempt",
            job_id=cast(int | None, value.get("job_id")),
            before_started_at=str(value.get("before_started_at") or ""),
            before_span_id=str(value.get("before_span_id") or ""),
            limit=200,
        )
        attempt_id = str(
            value.get("span_id") or (spans[0].get("span_id") if spans else "")
        )
        models = (
            database.monitor_history.page_spans(
                run_id,
                package_id=object_id,
                span_kind="package_model",
                parent_span_id=attempt_id,
                limit=500,
            )
            if detail_kind == "package" and attempt_id
            else []
        )
        events = database.monitor_history.page_events(
            run_id,
            before_event_id=None,
            execution_id="",
            object_id=object_id,
            stream_id=str(value.get("stream_id") or "")
            if detail_kind != "package"
            else "",
            job_id=cast(int | None, value.get("job_id")),
            span_id="",
            scope="all",
            levels=(),
            search="",
            limit=200,
        )
        return {
            **identity,
            "kind": "object_history",
            "object_id": object_id,
            "detail_kind": detail_kind,
            "spans": [dict(row) for row in spans],
            "events": [dict(row) for row in events],
            "models": [dict(row) for row in models],
            "attempt_id": attempt_id,
            "append": bool(value.get("append")),
            "has_more": len(spans) == 200,
        }
