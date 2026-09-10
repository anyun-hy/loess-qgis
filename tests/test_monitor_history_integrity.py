"""Attempt history is an audit trail, not a copy of mutable scheduling state."""

RUN = "monitor-integrity"


def start(database):
    database.create_run(RUN, "a" * 64)
    return database.begin_monitor_execution(RUN, "start")


def test_retry_preserves_failed_attempt_and_exact_recovery(postgres_database):
    db = postgres_database
    start(db)
    db.insert_work_packages(RUN, [{"package_id": "p1", "sequence_no": 0}])
    db.insert_jobs(RUN, [{"job_type": "work_package", "package_id": "p1", "max_attempts": 2}])
    first = db.lease_next_work_package(RUN, "a", max_open_frontier_units=64)
    assert db.fail_or_requeue_work_package_job(
        RUN, "p1", first["job_id"], first["lease_token"], "test failure"
    ) == "queued"
    span = db.page_monitor_spans(RUN)[0]
    assert (span["status"], span["message"]) == ("failed", "test failure")
    assert db.get_job(first["job_id"])["error"] == ""
    event = db.page_monitor_events(RUN, scope="issues")[0]
    assert event["span_id"] == first["monitor_span_id"]
    assert event["job_id"] == first["job_id"]
    second = db.lease_next_work_package(RUN, "b", max_open_frontier_units=64)
    assert db.complete_work_package_job(RUN, "p1", second["job_id"], second["lease_token"])
    failure = next(e for e in db.page_monitor_events(RUN) if e["event_type"] == "job_attempt_failed")
    assert failure["recovered_by_span_id"] == second["monitor_span_id"]
    assert [s["status"] for s in db.page_monitor_spans(RUN)] == ["completed", "failed"]


def test_success_cannot_recover_another_stream_or_job_type(postgres_database):
    db = postgres_database
    start(db)
    db.insert_jobs(RUN, [
        {"job_type": "unit_fit", "stream_id": "model:b", "unit_id": "core:1"},
    ])
    failed = db.lease_next_job(RUN, "failure")
    db.insert_jobs(RUN, [
        {"job_type": "unit_confidence", "stream_id": "model:b", "unit_id": "core:1"},
        {"job_type": "unit_fit", "stream_id": "model:a", "unit_id": "core:1"},
    ])
    successes = [db.lease_next_job(RUN, "success") for _ in range(2)]
    assert db.finish_job(failed["job_id"], failed["lease_token"], status="failed", error="bad fit")
    for success in successes:
        assert db.finish_job(success["job_id"], success["lease_token"], status="ready")
    events = db.page_monitor_events(RUN, scope="issues")
    assert len(events) == 1
    assert events[0]["job_id"] == failed["job_id"]
    assert events[0]["recovered_by_span_id"] is None


def test_assembly_snapshot_is_scoped_to_current_execution(postgres_database):
    db = postgres_database
    old = start(db)
    phase = db.start_monitor_span(RUN, execution_id=old, span_kind="assembly_phase",
                                 stream_id="model:a", phase="write_formal")
    db.finish_monitor_span(phase, status="completed")
    db.finish_monitor_execution(RUN, old, status="stopped")
    new = db.begin_monitor_execution(RUN, "resume")
    db.start_monitor_span(RUN, execution_id=new, span_kind="assembly_phase",
                          stream_id="model:a", phase="validate_inputs")
    phases = db.monitor_snapshot(RUN)["assembly_phase_statuses"]["model:a"]
    assert set(phases) == {"validate_inputs"}
    assert phases["validate_inputs"]["execution_id"] == new
    assert db.page_monitor_spans(RUN, execution_id=old)[0]["status"] == "completed"


def test_crash_recovery_does_not_invent_end_time(postgres_database):
    db = postgres_database
    old = start(db)
    db.start_monitor_span(RUN, execution_id=old, span_kind="job_attempt")
    db.begin_monitor_execution(RUN, "resume")
    span = db.page_monitor_spans(RUN, execution_id=old)[0]
    assert span["status"] == "interrupted"
    assert span["ended_at"] is None


def test_scope_filter_happens_before_cursor_limit(postgres_database):
    db = postgres_database
    start(db)
    old = db.append_monitor_event(RUN, "job_retry_started", message="older recovery", level="warning")
    for index in range(205):
        db.append_monitor_event(RUN, "ordinary", message=str(index))
    assert [e["monitor_event_id"] for e in db.page_monitor_events(RUN, scope="recovery", limit=200)] == [old]
    assert [e["monitor_event_id"] for e in db.page_monitor_events(RUN, scope="issues", limit=200)] == [old]
