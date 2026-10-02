"""Pure I/O contracts for monitor query execution."""

from __future__ import annotations

from typing import Mapping

import pytest

from labeling_tool.monitor.monitor_query_io import MonitorQueryExecutor


class _History:
    def page_events(self, *_args, **_kwargs):
        return [{"monitor_event_id": 1}]

    def page_spans(self, *_args, **_kwargs):
        return [{"span_id": "span-1"}]


class _RecordingHistory(_History):
    def __init__(self, calls) -> None:
        self.calls = calls

    def page_events(
        self,
        run_id,
        *,
        before_event_id,
        execution_id,
        stream_id,
        object_id,
        job_id,
        span_id,
        scope,
        levels,
        search,
        limit,
    ):
        self.calls.append(
            (
                "page_events",
                (run_id,),
                {
                    "before_event_id": before_event_id,
                    "execution_id": execution_id,
                    "stream_id": stream_id,
                    "object_id": object_id,
                    "job_id": job_id,
                    "span_id": span_id,
                    "scope": scope,
                    "levels": levels,
                    "search": search,
                    "limit": limit,
                },
            )
        )
        return [{"monitor_event_id": 1}]

    def page_spans(
        self,
        run_id,
        *,
        object_id="",
        package_id="",
        stream_id="",
        span_kind="",
        job_id=None,
        before_started_at="",
        before_span_id="",
        parent_span_id="",
        limit=200,
    ):
        self.calls.append(
            (
                "page_spans",
                (run_id,),
                {
                    "object_id": object_id,
                    "package_id": package_id,
                    "stream_id": stream_id,
                    "span_kind": span_kind,
                    "job_id": job_id,
                    "before_started_at": before_started_at,
                    "before_span_id": before_span_id,
                    "parent_span_id": parent_span_id,
                    "limit": limit,
                },
            )
        )
        return [{"span_id": "attempt-7"}]


class _Database:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.monitor_history = _History()

    @property
    def control_graph(self):
        return self

    @property
    def monitor_read(self):
        return self

    def snapshot(self, run_id: str) -> Mapping[str, object]:
        self.calls.append(("snapshot", run_id))
        return {"run": {"run_id": run_id}}

    def count_tiles(self, *_args, **_kwargs):
        return 2

    def page_tiles(self, *_args, **_kwargs):
        return [{"tile_id": "tile-1"}]

    def count_stream_units(self, *_args, **_kwargs):
        return 2

    def page_stream_units(self, *_args, **_kwargs):
        return [{"unit_id": "unit-1"}]

    def count_objects(self, *_args, **_kwargs):
        return 2

    def page_objects(self, *_args, **_kwargs):
        return [{"object_id": "object-1"}]


class _RecordingDatabase(_Database):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.monitor_history = _RecordingHistory(self.calls)

    def count_tiles(self, run_id, *, status, search):
        self.calls.append(
            ("count_tiles", (run_id,), {"status": status, "search": search})
        )
        return 501

    def page_tiles(self, run_id, *, limit, offset, status, search):
        self.calls.append(
            (
                "page_tiles",
                (run_id,),
                {"limit": limit, "offset": offset, "status": status, "search": search},
            )
        )
        return [{"tile_id": "tile-1"}]

    def count_stream_units(self, run_id, stream_id, *, status, search):
        self.calls.append(
            (
                "count_stream_units",
                (run_id, stream_id),
                {"status": status, "search": search},
            )
        )
        return 1

    def page_stream_units(self, run_id, stream_id, *, limit, offset, status, search):
        self.calls.append(
            (
                "page_stream_units",
                (run_id, stream_id),
                {"limit": limit, "offset": offset, "status": status, "search": search},
            )
        )
        return [{"unit_id": "unit-1"}]

    def count_objects(self, run_id, *, kind, stream_id, status, search):
        self.calls.append(
            (
                "count_objects",
                (run_id,),
                {
                    "kind": kind,
                    "stream_id": stream_id,
                    "status": status,
                    "search": search,
                },
            )
        )
        return 1

    def page_objects(self, run_id, *, kind, stream_id, limit, offset, status, search):
        self.calls.append(
            (
                "page_objects",
                (run_id,),
                {
                    "kind": kind,
                    "stream_id": stream_id,
                    "limit": limit,
                    "offset": offset,
                    "status": status,
                    "search": search,
                },
            )
        )
        return [{"object_id": "object-1"}]


def _request(kind: str, **values: object) -> dict[str, object]:
    return {
        "kind": kind,
        "generation": 3,
        "request_id": 4,
        "run_id": "run-a",
        "run_spec": {"state_db": "postgres://a"},
        **values,
    }


def test_executor_returns_identity_and_reuses_database_by_run_and_state_db():
    databases: list[_Database] = []

    def factory(_spec: Mapping[str, object]) -> _Database:
        database = _Database()
        databases.append(database)
        return database

    executor = MonitorQueryExecutor(database_factory=factory)
    first = executor.execute(_request("snapshot"))
    second = executor.execute(_request("snapshot"))
    executor.execute(_request("snapshot", run_id="run-b"))
    assert first["snapshot"] == second["snapshot"]
    assert {
        name: first[name] for name in ("kind", "generation", "request_id", "run_id")
    } == {
        "kind": "snapshot",
        "generation": 3,
        "request_id": 4,
        "run_id": "run-a",
    }
    assert len(databases) == 2


def test_executor_detail_bounds_history_and_object_queries():
    executor = MonitorQueryExecutor(database_factory=lambda _spec: _Database())
    detail = executor.execute(
        _request(
            "detail",
            stream_id="s",
            detail_kind="tile",
            status="",
            search="",
            page=99,
            page_size=900,
        )
    )
    assert (detail["page_size"], detail["page"], detail["page_total"]) == (500, 0, 1)
    history = executor.execute(
        _request(
            "history",
            scope="issues",
            execution_id="",
            search="",
            before_event_id=None,
            append=False,
            page_size=200,
        )
    )
    assert history["rows"] == [{"monitor_event_id": 1}]
    object_history = executor.execute(
        _request(
            "object_history",
            object_id="package-1",
            detail_kind="package",
            stream_id="",
            job_id=1,
            span_id="",
            append=False,
            before_started_at="",
            before_span_id="",
        )
    )
    assert object_history["attempt_id"] == "span-1"
    assert object_history["models"] == [{"span_id": "span-1"}]


def test_executor_uses_raw_log_reader_without_creating_database():
    calls: list[object] = []

    def reader(*args: object, **kwargs: object) -> Mapping[str, object]:
        calls.append((args, kwargs))
        return {"rows": [], "next_cursor": 0, "has_more": False, "skipped_records": 0}

    executor = MonitorQueryExecutor(
        database_factory=lambda _spec: (_ for _ in ()).throw(AssertionError("no db")),
        log_reader=reader,
    )
    result = executor.execute(
        _request(
            "history",
            scope="raw_error",
            search="boom",
            before_event_id=None,
            append=True,
        )
    )
    assert result["append"] is True
    assert len(calls) == 1


def test_executor_rejects_unknown_kinds():
    executor = MonitorQueryExecutor(database_factory=lambda _spec: _Database())
    with pytest.raises(ValueError, match="unknown monitor query kind"):
        executor.execute(_request("unexpected"))


def test_executor_forwards_detail_filters_to_tile_unit_and_object_branches():
    database = _RecordingDatabase()
    executor = MonitorQueryExecutor(database_factory=lambda _spec: database)
    for detail_kind, stream_id in (
        ("tile", ""),
        ("unit", "model:a"),
        ("package", "model:a"),
    ):
        executor.execute(
            _request(
                "detail",
                stream_id=stream_id,
                detail_kind=detail_kind,
                status="ready",
                search="needle",
                page=1,
                page_size=500,
            )
        )
    assert database.calls == [
        ("count_tiles", ("run-a",), {"status": "ready", "search": "needle"}),
        (
            "page_tiles",
            ("run-a",),
            {"limit": 500, "offset": 500, "status": "ready", "search": "needle"},
        ),
        (
            "count_stream_units",
            ("run-a", "model:a"),
            {"status": "ready", "search": "needle"},
        ),
        (
            "page_stream_units",
            ("run-a", "model:a"),
            {"limit": 500, "offset": 0, "status": "ready", "search": "needle"},
        ),
        (
            "count_objects",
            ("run-a",),
            {
                "kind": "package",
                "stream_id": "model:a",
                "status": "ready",
                "search": "needle",
            },
        ),
        (
            "page_objects",
            ("run-a",),
            {
                "kind": "package",
                "stream_id": "model:a",
                "limit": 500,
                "offset": 0,
                "status": "ready",
                "search": "needle",
            },
        ),
    ]


def test_executor_forwards_history_context_and_object_attempt_cursor():
    database = _RecordingDatabase()
    executor = MonitorQueryExecutor(database_factory=lambda _spec: database)
    executor.execute(
        _request(
            "history",
            scope="all",
            execution_id="exec-1",
            search="needle",
            before_event_id=42,
            append=True,
            page_size=123,
            stream_id="model:a",
            object_id="object-1",
            job_id=9,
            span_id="span-9",
        )
    )
    executor.execute(
        _request(
            "object_history",
            object_id="package-1",
            detail_kind="package",
            stream_id="ignored",
            job_id=10,
            span_id="attempt-selected",
            append=True,
            before_started_at="2026-01-02T03:04:05+00:00",
            before_span_id="span-cursor",
        )
    )
    assert database.calls[0] == (
        "page_events",
        ("run-a",),
        {
            "before_event_id": 42,
            "execution_id": "exec-1",
            "stream_id": "model:a",
            "object_id": "object-1",
            "job_id": 9,
            "span_id": "span-9",
            "scope": "all",
            "levels": (),
            "search": "needle",
            "limit": 123,
        },
    )
    assert database.calls[1][0:2] == ("page_spans", ("run-a",))
    assert database.calls[1][2] == {
        "object_id": "",
        "package_id": "package-1",
        "stream_id": "",
        "span_kind": "job_attempt",
        "job_id": 10,
        "before_started_at": "2026-01-02T03:04:05+00:00",
        "before_span_id": "span-cursor",
        "parent_span_id": "",
        "limit": 200,
    }
    assert database.calls[2][2] == {
        "object_id": "",
        "package_id": "package-1",
        "stream_id": "",
        "span_kind": "package_model",
        "job_id": None,
        "before_started_at": "",
        "before_span_id": "",
        "parent_span_id": "attempt-selected",
        "limit": 500,
    }
