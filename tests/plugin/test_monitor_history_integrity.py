"""Attempt history is an audit trail, not a copy of mutable scheduling state."""

RUN = "monitor-integrity"


def test_unit_commit_rolls_back_partial_artifacts_and_fences_old_owner(
    postgres_database, tmp_path
):
    import pytest

    from labeling_tool.shared.state.run_state_db import RunStateError

    db = postgres_database
    start(db)
    db.jobs.insert_jobs(
        RUN, [{"job_type": "unit_fit", "stream_id": "model:a", "unit_id": "core:1"}]
    )
    first = db.jobs.lease_next_job(RUN, "first")
    path = tmp_path / "attempt_report.json"
    path.write_text("{}")
    with pytest.raises(RuntimeError, match="injected"):
        with db.unit_attempt_commit(first["job_id"], first["lease_token"]) as tx:
            aid = tx.artifacts.register_artifact(
                RUN, "unit_boundary_report", path, stream_id="model:a", unit_id="core:1"
            )
            assert tx.artifacts.mark_artifact_ready(aid, byte_count=2, sha256="a" * 64)
            # Another connection sees no partial ready publication.
            assert (
                db.artifacts.artifact_for_stream_unit(
                    RUN, "model:a", "core:1", "unit_boundary_report"
                )
                is None
            )
            # Memory shedding must skip a writer holding the commit lock.
            assert not db.jobs.interrupt_job(first["job_id"], first["lease_token"])
            raise RuntimeError("injected publication failure")
    assert (
        db.artifacts.artifact_for_stream_unit(
            RUN, "model:a", "core:1", "unit_boundary_report"
        )
        is None
    )
    assert db.jobs.interrupt_job(first["job_id"], first["lease_token"])
    second = db.jobs.lease_next_job(RUN, "second")
    with pytest.raises(RunStateError, match="lease"):
        with db.unit_attempt_commit(first["job_id"], first["lease_token"]):
            raise AssertionError("old writer acquired publication rights")
    with db.unit_attempt_commit(second["job_id"], second["lease_token"]) as tx:
        aid = tx.artifacts.register_artifact(
            RUN, "unit_boundary_report", path, stream_id="model:a", unit_id="core:1"
        )
        assert tx.artifacts.mark_artifact_ready(aid, byte_count=2, sha256="b" * 64)
        assert tx.jobs.finish_job(second["job_id"], second["lease_token"])
    assert db.jobs.get_job(second["job_id"])["status"] == "ready"
    assert (
        db.artifacts.artifact_for_stream_unit(
            RUN, "model:a", "core:1", "unit_boundary_report"
        )["sha256"]
        == "b" * 64
    )


def test_unit_superseding_is_atomic_and_does_not_overwrite_old_file(
    postgres_database, tmp_path
):
    import pytest

    db = postgres_database
    start(db)
    old = tmp_path / "old.json"
    old.write_text("old evidence")
    aid = db.artifacts.register_artifact(
        RUN, "unit_boundary_report", old, stream_id="model:a", unit_id="core:1"
    )
    db.artifacts.mark_artifact_ready(aid, byte_count=12, sha256="a" * 64)
    db.jobs.insert_jobs(
        RUN, [{"job_type": "unit_fit", "stream_id": "model:a", "unit_id": "core:1"}]
    )
    job = db.jobs.lease_next_job(RUN, "worker")
    with pytest.raises(RuntimeError):
        with db.unit_attempt_commit(job["job_id"], job["lease_token"]) as tx:
            tx.artifacts.supersede_unit_attempt_artifacts(RUN, "model:a", "core:1")
            raise RuntimeError("publication failed")
    assert db.artifacts.get_artifact(aid)["status"] == "ready"
    with db.unit_attempt_commit(job["job_id"], job["lease_token"]) as tx:
        tx.artifacts.supersede_unit_attempt_artifacts(RUN, "model:a", "core:1")
        tx.jobs.finish_job(job["job_id"], job["lease_token"])
    assert db.artifacts.get_artifact(aid)["status"] == "superseded"
    assert old.read_text() == "old evidence"


def test_unit_lease_expiry_at_final_commit_rolls_back_ready_artifact(
    postgres_database, tmp_path
):
    import pytest

    db = postgres_database
    start(db)
    db.jobs.insert_jobs(
        RUN, [{"job_type": "unit_fit", "stream_id": "model:a", "unit_id": "core:1"}]
    )
    job = db.jobs.lease_next_job(RUN, "worker")
    with pytest.raises(RuntimeError, match="expired"):
        with db.unit_attempt_commit(job["job_id"], job["lease_token"]) as tx:
            aid = tx.artifacts.register_artifact(
                RUN,
                "unit_boundary_report",
                tmp_path / "report.json",
                stream_id="model:a",
                unit_id="core:1",
            )
            tx.artifacts.mark_artifact_ready(aid, byte_count=2, sha256="a" * 64)
            with tx.session.transaction() as connection:
                connection.execute(
                    "UPDATE jobs SET lease_expires=0 WHERE job_id=%s", (job["job_id"],)
                )
            if not tx.jobs.finish_job(job["job_id"], job["lease_token"]):
                raise RuntimeError("expired at final fence")
    assert (
        db.artifacts.artifact_for_stream_unit(
            RUN, "model:a", "core:1", "unit_boundary_report"
        )
        is None
    )
    assert db.jobs.get_job(job["job_id"])["status"] == "running"


def start(database):
    database.run_streams.create_run(RUN, "a" * 64)
    return database.monitor_history.begin_execution(RUN, "start")


def test_retry_preserves_failed_attempt_and_exact_recovery(postgres_database):
    db = postgres_database
    start(db)
    db.control_graph.insert_work_packages(RUN, [{"package_id": "p1", "sequence_no": 0}])
    db.jobs.insert_jobs(
        RUN, [{"job_type": "work_package", "package_id": "p1", "max_attempts": 2}]
    )
    first = db.jobs.lease_next_work_package(RUN, "a", max_open_frontier_units=64)
    assert (
        db.jobs.fail_or_requeue_work_package_job(
            RUN, "p1", first["job_id"], first["lease_token"], "test failure"
        )
        == "queued"
    )
    span = db.monitor_history.page_spans(RUN)[0]
    assert (span["status"], span["message"]) == ("failed", "test failure")
    assert db.jobs.get_job(first["job_id"])["error"] == ""
    event = db.monitor_history.page_events(RUN, scope="issues")[0]
    assert event["span_id"] == first["monitor_span_id"]
    assert event["job_id"] == first["job_id"]
    second = db.jobs.lease_next_work_package(RUN, "b", max_open_frontier_units=64)
    assert db.jobs.complete_work_package_job(
        RUN, "p1", second["job_id"], second["lease_token"]
    )
    failure = next(
        e
        for e in db.monitor_history.page_events(RUN)
        if e["event_type"] == "job_attempt_failed"
    )
    assert failure["recovered_by_span_id"] == second["monitor_span_id"]
    assert [s["status"] for s in db.monitor_history.page_spans(RUN)] == [
        "completed",
        "failed",
    ]


def test_success_cannot_recover_another_stream_or_job_type(postgres_database):
    db = postgres_database
    start(db)
    db.jobs.insert_jobs(
        RUN,
        [
            {"job_type": "unit_fit", "stream_id": "model:b", "unit_id": "core:1"},
        ],
    )
    failed = db.jobs.lease_next_job(RUN, "failure")
    db.jobs.insert_jobs(
        RUN,
        [
            {
                "job_type": "unit_confidence",
                "stream_id": "model:b",
                "unit_id": "core:1",
            },
            {"job_type": "unit_fit", "stream_id": "model:a", "unit_id": "core:1"},
        ],
    )
    successes = [db.jobs.lease_next_job(RUN, "success") for _ in range(2)]
    assert db.jobs.finish_job(
        failed["job_id"], failed["lease_token"], status="failed", error="bad fit"
    )
    for success in successes:
        assert db.jobs.finish_job(
            success["job_id"], success["lease_token"], status="ready"
        )
    events = db.monitor_history.page_events(RUN, scope="issues")
    assert len(events) == 1
    assert events[0]["job_id"] == failed["job_id"]
    assert events[0]["recovered_by_span_id"] is None


def test_assembly_snapshot_is_scoped_to_current_execution(postgres_database):
    db = postgres_database
    old = start(db)
    phase = db.monitor_history.start_span(
        RUN,
        execution_id=old,
        span_kind="assembly_phase",
        stream_id="model:a",
        phase="write_formal",
    )
    db.monitor_history.finish_span(phase, status="completed")
    db.monitor_history.finish_execution(RUN, old, status="stopped")
    new = db.monitor_history.begin_execution(RUN, "resume")
    db.monitor_history.start_span(
        RUN,
        execution_id=new,
        span_kind="assembly_phase",
        stream_id="model:a",
        phase="validate_inputs",
    )
    phases = db.monitor_read.snapshot(RUN)["assembly_phase_statuses"]["model:a"]
    assert set(phases) == {"validate_inputs"}
    assert phases["validate_inputs"]["execution_id"] == new
    assert (
        db.monitor_history.page_spans(RUN, execution_id=old)[0]["status"] == "completed"
    )


def test_crash_recovery_does_not_invent_end_time(postgres_database):
    db = postgres_database
    old = start(db)
    db.monitor_history.start_span(RUN, execution_id=old, span_kind="job_attempt")
    db.monitor_history.begin_execution(RUN, "resume")
    span = db.monitor_history.page_spans(RUN, execution_id=old)[0]
    assert span["status"] == "interrupted"
    assert span["ended_at"] is None


def test_scope_filter_happens_before_cursor_limit(postgres_database):
    db = postgres_database
    start(db)
    old = db.monitor_history.append_event(
        RUN, "job_retry_started", message="older recovery", level="warning"
    )
    for index in range(205):
        db.monitor_history.append_event(RUN, "ordinary", message=str(index))
    assert [
        e["monitor_event_id"]
        for e in db.monitor_history.page_events(RUN, scope="recovery", limit=200)
    ] == [old]
    assert [
        e["monitor_event_id"]
        for e in db.monitor_history.page_events(RUN, scope="issues", limit=200)
    ] == [old]
