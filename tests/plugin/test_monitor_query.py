"""Direct behavior tests for pure monitor query coordination."""

from __future__ import annotations

from labeling_tool.monitor.monitor_query import (
    MonitorQueryCoordinator,
    query_filter,
    query_matches_filter,
)


def _complete(request: dict[str, object]) -> dict[str, object]:
    return {
        name: request[name] for name in ("kind", "generation", "request_id", "run_id")
    }


def test_requests_are_serialized_prioritized_and_latest_wins_per_kind():
    coordinator = MonitorQueryCoordinator()
    coordinator.bind("run-a", {"state_db": "postgres://a"})
    first_snapshot = coordinator.enqueue_snapshot()
    assert first_snapshot == 1
    active = coordinator.take_next()
    assert active is not None and active["kind"] == "snapshot"
    coordinator.enqueue_snapshot()
    coordinator.enqueue_snapshot()
    coordinator.enqueue_history(
        scope="all",
        execution_id="",
        search="",
        before_event_id=None,
        append=False,
        page_size=200,
        context={},
    )
    coordinator.enqueue_detail(
        stream_id="model:a",
        detail_kind="unit",
        status="",
        search="",
        page=0,
        page_size=500,
    )
    assert coordinator.take_next() is None
    assert coordinator.complete(_complete(active)).applicable
    detail = coordinator.take_next()
    assert detail is not None and detail["kind"] == "detail"
    assert coordinator.complete(_complete(detail)).applicable
    history = coordinator.take_next()
    assert history is not None and history["kind"] == "history"
    assert coordinator.complete(_complete(history)).applicable
    newest_snapshot = coordinator.take_next()
    assert newest_snapshot is not None
    assert newest_snapshot["kind"] == "snapshot"
    assert newest_snapshot["request_id"] == 3


def test_old_active_releases_after_bind_without_starting_two_calls():
    coordinator = MonitorQueryCoordinator()
    coordinator.bind("run-a", {"state_db": "a"})
    coordinator.enqueue_snapshot()
    old_active = coordinator.take_next()
    assert old_active is not None
    coordinator.bind("run-b", {"state_db": "b"})
    coordinator.enqueue_snapshot()
    assert coordinator.take_next() is None
    completion = coordinator.complete(_complete(old_active))
    assert completion is not None and not completion.applicable
    new_active = coordinator.take_next()
    assert new_active is not None and new_active["run_id"] == "run-b"


def test_unrelated_or_duplicate_callbacks_never_release_current_active():
    coordinator = MonitorQueryCoordinator()
    coordinator.bind("run", {})
    coordinator.enqueue_snapshot()
    active = coordinator.take_next()
    assert active is not None
    unrelated = {**_complete(active), "request_id": 999}
    assert coordinator.complete(unrelated) is None
    assert coordinator.take_next() is None
    assert coordinator.complete(_complete(active)).applicable
    assert coordinator.complete(_complete(active)) is None


def test_active_completion_remains_applicable_when_later_poll_coalesces():
    coordinator = MonitorQueryCoordinator()
    coordinator.bind("run", {})
    coordinator.enqueue_snapshot()
    active = coordinator.take_next()
    assert active is not None
    coordinator.enqueue_snapshot()
    coordinator.enqueue_snapshot()
    completion = coordinator.complete(_complete(active))
    assert completion is not None and completion.applicable
    pending = coordinator.take_next()
    assert pending is not None and pending["request_id"] != active["request_id"]


def test_fence_and_unbind_preserve_active_but_cancel_pending():
    coordinator = MonitorQueryCoordinator()
    coordinator.bind("run", {})
    coordinator.enqueue_snapshot()
    active = coordinator.take_next()
    assert active is not None
    coordinator.enqueue_snapshot()
    coordinator.fence()
    assert coordinator.take_next() is None
    completion = coordinator.complete(_complete(active))
    assert completion is not None and not completion.applicable
    coordinator.enqueue_snapshot()
    coordinator.unbind()
    assert coordinator.take_next() is None


def test_filter_contract_checks_every_ui_field_and_raw_continuations():
    detail = {
        "kind": "detail",
        "stream_id": "s",
        "detail_kind": "unit",
        "status": "ready",
        "search": "needle",
        "page": 2,
        "page_size": 100,
    }
    assert query_matches_filter(detail, dict(detail))
    for name in ("stream_id", "detail_kind", "status", "search", "page", "page_size"):
        changed = dict(detail)
        changed[name] = "other" if isinstance(changed[name], str) else 3
        assert not query_matches_filter(detail, changed)
    raw = {"kind": "history", "scope": "raw_error", "search": "boom"}
    assert query_filter(raw) == ("raw_error", "boom")
    coordinator = MonitorQueryCoordinator()
    coordinator.bind("run", {})
    request_id = coordinator.enqueue_history(
        scope="raw_error",
        execution_id="",
        search="boom",
        before_event_id=None,
        append=False,
        page_size=200,
        context={},
    )
    assert request_id is not None
    assert coordinator.continuation_is_current(
        "history",
        generation=coordinator.generation,
        request_id=request_id,
    )
    coordinator.fence()
    assert not coordinator.continuation_is_current(
        "history",
        generation=coordinator.generation - 1,
        request_id=request_id,
    )


def test_binding_and_returned_specs_are_independent_plain_snapshots():
    coordinator = MonitorQueryCoordinator()
    source = {"models": [{"model_id": "first"}]}
    coordinator.bind("run", source)
    source["models"][0]["model_id"] = "changed"
    returned = coordinator.run_spec()
    returned["models"][0]["model_id"] = "returned-change"
    assert coordinator.run_spec()["models"] == [{"model_id": "first"}]


def test_all_query_filters_reject_one_changed_current_control():
    cases = (
        (
            {
                "kind": "history",
                "scope": "all",
                "execution_id": "exec",
                "search": "word",
                "context": {"stream_id": "s"},
            },
            "execution_id",
            "other",
        ),
        (
            {"kind": "history", "scope": "raw_warning", "search": "word"},
            "search",
            "other",
        ),
        (
            {
                "kind": "object_history",
                "object_id": "object",
                "detail_kind": "package",
                "stream_id": "s",
                "job_id": 7,
                "span_id": "span",
            },
            "span_id",
            "other",
        ),
    )
    for request, field, changed in cases:
        current = dict(request)
        current[field] = changed
        assert not query_matches_filter(request, current)


def test_object_history_beats_history_and_snapshot_after_active_completion():
    coordinator = MonitorQueryCoordinator()
    coordinator.bind("run", {})
    coordinator.enqueue_snapshot()
    active = coordinator.take_next()
    assert active is not None
    coordinator.enqueue_snapshot()
    coordinator.enqueue_history(
        scope="all",
        execution_id="",
        search="",
        before_event_id=None,
        append=False,
        page_size=200,
        context={},
    )
    coordinator.enqueue_object_history(
        object_id="object",
        detail_kind="package",
        stream_id="",
        job_id=None,
        span_id="",
        append=False,
        before_started_at="",
        before_span_id="",
    )
    assert coordinator.complete(_complete(active)).applicable
    next_request = coordinator.take_next()
    assert next_request is not None and next_request["kind"] == "object_history"
