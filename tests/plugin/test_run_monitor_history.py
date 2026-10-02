"""Persisted behavior of runner monitor-history recording."""

RUN = "run-monitor-history"


def recorder(database, *, trigger="start"):
    from labeling_tool.runs.monitor_history import RunHistoryRecorder

    database.run_streams.create_run(RUN, "a" * 64)
    execution_id = database.monitor_history.begin_execution(RUN, trigger)
    return RunHistoryRecorder(database.monitor_history, RUN, execution_id), execution_id


def model_event(event, parent, *, package="package-a", stream="model:a"):
    return {
        "event": event,
        "package_id": package,
        "stream_id": stream,
        "parent_span_id": parent,
        "configured_batch_size": 8,
    }


def parent_span(database, execution_id, key):
    return database.monitor_history.start_span(
        RUN,
        execution_id=execution_id,
        span_kind="job_attempt",
        object_type="package",
        object_id="package-a",
        package_id="package-a",
        idempotency_key=f"parent:{execution_id}:{key}",
    )


def test_model_spans_are_idempotent_per_parent_and_finish_independently(
    postgres_database,
):
    history, execution_id = recorder(postgres_database)
    first_parent = parent_span(postgres_database, execution_id, "first")
    second_parent = parent_span(postgres_database, execution_id, "second")
    history.record(model_event("package_model_loading", first_parent))
    history.record(model_event("package_model_loading", first_parent))
    history.record(model_event("package_model_loading", second_parent))
    history.record(model_event("package_model_completed", first_parent))
    history.record(model_event("package_model_outputs_reused", second_parent))

    spans = postgres_database.monitor_history.page_spans(
        RUN, execution_id=execution_id, span_kind="package_model"
    )
    assert len(spans) == 2
    by_parent = {span["parent_span_id"]: span for span in spans}
    assert by_parent[first_parent]["status"] == "completed"
    assert by_parent[second_parent]["status"] == "reused"
    assert {span["object_id"] for span in spans} == {"package-a:model:a"}


def test_unfinished_models_and_assembly_phases_are_not_inferred_success(
    postgres_database,
):
    history, execution_id = recorder(postgres_database)
    finished_parent = parent_span(postgres_database, execution_id, "finished")
    failed_parent = parent_span(postgres_database, execution_id, "failed")
    history.record(model_event("package_model_loading", finished_parent))
    history.record(
        {
            "event": "work_package_finished",
            "package_id": "package-a",
            "parent_span_id": finished_parent,
        }
    )
    history.record(model_event("package_model_loading", failed_parent))
    history.record(
        {
            "event": "work_package_failed",
            "package_id": "package-a",
            "parent_span_id": failed_parent,
            "error": "worker failed",
        }
    )
    history.record(
        {
            "event": "assembly_progress",
            "stream_id": "model:a",
            "phase": "validate_inputs",
            "status": "running",
        }
    )
    history.record(
        {
            "event": "assembly_progress",
            "stream_id": "model:a",
            "phase": "write_raw",
            "status": "running",
        }
    )
    process = history.start_process(
        "process-token",
        "assemble model:a",
        {"kind": "assemble", "stream_id": "model:a"},
    )
    history.finish_process(process, "model:a", success=True, error="", exit_code=0)

    model_spans = postgres_database.monitor_history.page_spans(
        RUN, execution_id=execution_id, span_kind="package_model"
    )
    assert {span["parent_span_id"]: span["status"] for span in model_spans} == {
        finished_parent: "interrupted",
        failed_parent: "failed",
    }
    assembly = postgres_database.monitor_history.page_spans(
        RUN, execution_id=execution_id, span_kind="assembly_phase"
    )
    assert {span["phase"]: span["status"] for span in assembly} == {
        "validate_inputs": "interrupted",
        "write_raw": "interrupted",
    }
    runtime = postgres_database.monitor_history.page_spans(
        RUN, execution_id=execution_id, span_kind="runtime_phase"
    )
    assert len(runtime) == 1 and runtime[0]["status"] == "completed"


def test_old_execution_events_are_rejected_and_new_recorder_starts_clean(
    postgres_database,
):
    from labeling_tool.runs.monitor_history import RunHistoryRecorder

    history, old_execution = recorder(postgres_database)
    old_parent = parent_span(postgres_database, old_execution, "old")
    history.record(model_event("package_model_loading", old_parent))
    new_execution = postgres_database.monitor_history.begin_execution(RUN, "resume")
    current = RunHistoryRecorder(
        postgres_database.monitor_history,
        RUN,
        new_execution,
    )
    current.record(
        model_event("package_model_loading", old_parent)
        | {"execution_id": old_execution}
    )
    current.record(
        {
            "event": "package_tile_batch_reduced",
            "package_id": "package-a",
            "stream_id": "model:a",
            "execution_id": old_execution,
            "effective_batch_size": 2,
        }
    )
    assert not [
        event
        for event in postgres_database.monitor_history.page_events(RUN, limit=20)
        if event["event_type"] == "package_tile_batch_reduced"
    ]
    new_parent = parent_span(postgres_database, new_execution, "new")
    current.record(model_event("package_model_loading", new_parent))

    old = postgres_database.monitor_history.page_spans(
        RUN, execution_id=old_execution, span_kind="package_model"
    )
    assert len(old) == 1 and old[0]["status"] == "interrupted"
    new = postgres_database.monitor_history.page_spans(
        RUN, execution_id=new_execution, span_kind="package_model"
    )
    assert len(new) == 1
    assert new[0]["parent_span_id"] == new_parent and new[0]["status"] == "running"


def test_important_events_are_filtered_idempotent_and_redacted(postgres_database):
    history, execution_id = recorder(postgres_database)
    parent = parent_span(postgres_database, execution_id, "filter")
    history.record(model_event("package_model_loading", parent))
    history.record(
        {
            "event": "assembly_progress",
            "stream_id": "model:a",
            "parent_span_id": parent,
            "phase": "validate_inputs",
            "status": "running",
            "current": 1,
            "total": 2,
        }
    )
    reduced = {
        "event": "package_tile_batch_reduced",
        "package_id": "package-a",
        "stream_id": "model:a",
        "effective_batch_size": 2,
        "lease_token": "secret-lease",
        "state_db": "postgresql://secret",
        "dsn": "postgresql://secret",
        "environment": {"TOKEN": "secret"},
    }
    history.record(reduced)
    history.record(reduced)

    events = postgres_database.monitor_history.page_events(
        RUN, execution_id=execution_id, limit=20
    )
    reduced_events = [
        event for event in events if event["event_type"] == "package_tile_batch_reduced"
    ]
    assert len(reduced_events) == 1
    assert not {
        "lease_token",
        "state_db",
        "dsn",
        "environment",
    } & set(reduced_events[0]["payload"])
    assert {event["event_type"] for event in events} == {
        "execution_started",
        "package_tile_batch_reduced",
    }
    assert reduced_events[0]["level"] == "warning"
    assert reduced_events[0]["payload"]["effective_batch_size"] == 2


def test_database_write_failure_propagates_from_important_event():
    import pytest

    from labeling_tool.runs.monitor_history import RunHistoryRecorder

    class FailingRepository:
        def append_event(self, *_args, **_kwargs):
            raise RuntimeError("monitor database unavailable")

    history = RunHistoryRecorder(FailingRepository(), RUN, "execution-a")
    with pytest.raises(RuntimeError, match="monitor database unavailable"):
        history.record(
            {
                "event": "package_tile_batch_reduced",
                "package_id": "package-a",
                "stream_id": "model:a",
                "effective_batch_size": 2,
            }
        )
