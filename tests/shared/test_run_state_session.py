"""Run-state transaction ownership and scoped monitor-history contracts."""

from __future__ import annotations

import hashlib
import json

import pytest

from labeling_tool.shared.state.run_state_session import RunStateError

RUN = "run-state-session"
UNIT = {
    "unit_id": "core:1",
    "unit_type": "Core",
    "owner_key": "partition:1",
    "pixel_window": {"x0": 0, "y0": 0, "x1": 8, "y1": 8},
}


def _leased_unit(database):
    database.run_streams.create_run(RUN, "a" * 64)
    database.jobs.insert_jobs(
        RUN,
        [{"job_type": "unit_fit", "stream_id": "model:a", "unit_id": "core:1"}],
    )
    job = database.jobs.lease_next_job(RUN, "session-test", lease_seconds=120)
    assert job is not None
    return job


def _leased_fragmentation_v33(database):
    database.run_streams.create_run(RUN, "b" * 64)
    database.jobs.insert_jobs(
        RUN,
        [
            {
                "job_type": "fragmentation_v33",
                "stream_id": "fusion:a",
                "unit_id": "fragmentation_v33_partition:partition:1",
            }
        ],
    )
    with database.session.connection() as connection:
        job_id = int(
            connection.execute(
                "SELECT job_id FROM jobs WHERE run_id=%s", (RUN,)
            ).fetchone()[0]
        )
    job = database.jobs.lease_job(job_id, "v33-session-test", lease_seconds=120)
    assert job is not None
    return job


def _record_scoped_report(scoped, path):
    report = {"status": "passed", "chain_count": 2, "diagnostics": []}
    path.write_text(json.dumps(report), encoding="utf-8")
    scoped.run_streams.register_streams(
        RUN, [{"stream_id": "model:a", "kind": "model", "model_id": "a"}]
    )
    scoped.control_graph.insert_stream_units(RUN, ["model:a"], ["core:1"])
    artifact_id = scoped.artifacts.register_artifact(
        RUN, "unit_boundary_report", path, stream_id="model:a", unit_id="core:1"
    )
    assert scoped.artifacts.mark_artifact_ready(
        artifact_id,
        byte_count=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    # The report repository must see the Artifact that is still uncommitted.
    scoped.unit_reports.upsert_unit_report_summary(RUN, "model:a", "core:1", report)
    return artifact_id


def test_scoped_commit_keeps_related_repositories_on_one_connection(
    postgres_database, monkeypatch, tmp_path
):
    job = _leased_unit(postgres_database)
    postgres_database.monitor_history.begin_execution(RUN, "start")
    original_connect = postgres_database.session.connect
    connections = []

    class TrackedConnection:
        def __init__(self, connection):
            self._connection = connection
            self.commits = 0
            self.rollbacks = 0

        def __getattr__(self, name):
            return getattr(self._connection, name)

        def commit(self):
            self.commits += 1
            return self._connection.commit()

        def rollback(self):
            self.rollbacks += 1
            return self._connection.rollback()

    def tracked_connect(*, autocommit=True):
        connection = TrackedConnection(original_connect(autocommit=autocommit))
        connections.append(connection)
        return connection

    monkeypatch.setattr(postgres_database.session, "connect", tracked_connect)
    with postgres_database.unit_attempt_commit(
        job["job_id"], job["lease_token"]
    ) as scoped:
        assert scoped.control_graph.insert_spatial_units(RUN, [UNIT]) == 1
        artifact_id = _record_scoped_report(scoped, tmp_path / "report.json")
        event_id = scoped.monitor_history.append_event(
            RUN,
            "session_commit",
            object_type="unit",
            object_id="core:1",
        )
        assert scoped.monitor_read.count_objects(RUN, kind="unit_fit") == 1
        assert scoped.jobs.finish_job(job["job_id"], job["lease_token"])

    assert len(connections) == 1
    assert connections[0].commits == 1
    assert connections[0].rollbacks == 0
    assert (
        postgres_database.control_graph.get_spatial_unit(RUN, "core:1")["pixel_window"]
        == UNIT["pixel_window"]
    )
    assert postgres_database.jobs.get_job(job["job_id"])["status"] == "ready"
    assert postgres_database.artifacts.get_artifact(artifact_id)["status"] == "ready"
    reports = postgres_database.unit_reports.unit_report_summaries(RUN, "model:a")
    assert len(reports) == 1
    assert reports[0]["chain_count"] == 2
    assert (
        postgres_database.monitor_history.page_events(RUN, limit=10)[0][
            "monitor_event_id"
        ]
        == event_id
    )


def test_scoped_failure_rolls_back_related_repositories_without_changing_root_execution(
    postgres_database, tmp_path
):
    job = _leased_unit(postgres_database)
    root_execution = postgres_database.monitor_history.begin_execution(RUN, "start")
    with pytest.raises(RuntimeError, match="rollback"):
        with postgres_database.unit_attempt_commit(
            job["job_id"], job["lease_token"]
        ) as scoped:
            assert scoped.control_graph.insert_spatial_units(RUN, [UNIT]) == 1
            artifact_id = _record_scoped_report(scoped, tmp_path / "report.json")
            scoped.monitor_history.append_event(RUN, "rollback_event")
            assert scoped.jobs.finish_job(job["job_id"], job["lease_token"])
            raise RuntimeError("rollback")

    assert postgres_database.session.execution_id == root_execution
    assert postgres_database.control_graph.get_spatial_unit(RUN, "core:1") is None
    assert postgres_database.run_streams.stream_rows(RUN) == []
    assert postgres_database.unit_reports.unit_report_summaries(RUN, "model:a") == []
    assert postgres_database.artifacts.get_artifact(artifact_id) is None
    assert postgres_database.jobs.get_job(job["job_id"])["status"] == "running"
    assert not [
        event
        for event in postgres_database.monitor_history.page_events(RUN, limit=20)
        if event["event_type"] == "rollback_event"
    ]
    with postgres_database.unit_attempt_commit(
        job["job_id"], job["lease_token"]
    ) as scoped:
        with pytest.raises(RunStateError, match="scoped"):
            scoped.monitor_history.begin_execution(RUN, "resume")
    with postgres_database.session.connection() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM monitor_executions WHERE run_id=%s", (RUN,)
            ).fetchone()[0]
            == 1
        )


def test_scoped_session_and_repositories_reject_use_after_commit_scope(
    postgres_database,
):
    job = _leased_unit(postgres_database)
    with postgres_database.unit_attempt_commit(
        job["job_id"], job["lease_token"]
    ) as scoped:
        saved = scoped
        assert scoped.session.unit_identity == (RUN, "model:a", "core:1")
        with pytest.raises(RunStateError, match="cannot initialize schemas"):
            scoped.initialize()
        with pytest.raises(RunStateError, match="cannot run health checks"):
            scoped.pragmas()

    with pytest.raises(RunStateError, match="no longer active"):
        with saved.session.connection():
            raise AssertionError("inactive scoped session yielded a connection")
    with pytest.raises(RunStateError, match="no longer active"):
        saved.monitor_history.append_event(RUN, "late_event")
    with pytest.raises(RunStateError, match="no longer active"):
        saved.monitor_read.count_objects(RUN, kind="unit_fit")
    with pytest.raises(RunStateError, match="no longer active"):
        saved.jobs.get_job(job["job_id"])
    with pytest.raises(RunStateError, match="no longer active"):
        saved.package_resets.begin_failed_package_reset(RUN)
    with pytest.raises(RunStateError, match="no longer active"):
        saved.run_archive.archive_incomplete_run_details(protected_run_id=RUN)
    with pytest.raises(RunStateError, match="no longer active"):
        saved.control_graph.get_spatial_unit(RUN, "core:1")
    with pytest.raises(RunStateError, match="no longer active"):
        saved.run_streams.get_run(RUN)
    with pytest.raises(RunStateError, match="no longer active"):
        saved.unit_reports.unit_report_summaries(RUN, "model:a")


def test_fragmentation_v33_attempt_commit_uses_one_lease_fenced_transaction(
    postgres_database,
):
    job = _leased_fragmentation_v33(postgres_database)

    with postgres_database.fragmentation_v33_attempt_commit(
        job["job_id"], job["lease_token"]
    ) as scoped:
        saved = scoped
        assert scoped.session.unit_identity == (
            RUN,
            "fusion:a",
            "fragmentation_v33_partition:partition:1",
        )
        scoped.monitor_history.append_event(
            RUN,
            "fragmentation_v33_attempt_committed",
            object_type="job",
            object_id=str(job["job_id"]),
        )

    assert any(
        event["event_type"] == "fragmentation_v33_attempt_committed"
        for event in postgres_database.monitor_history.page_events(RUN, limit=20)
    )
    with pytest.raises(RunStateError, match="no longer active"):
        saved.jobs.get_job(job["job_id"])
    with pytest.raises(RunStateError, match="no longer owns its lease"):
        with postgres_database.fragmentation_v33_attempt_commit(
            job["job_id"], "wrong-token"
        ):
            raise AssertionError("invalid lease entered the publication scope")


def test_fragmentation_v33_attempt_commit_rolls_back_all_scoped_writes(
    postgres_database,
):
    job = _leased_fragmentation_v33(postgres_database)

    with pytest.raises(RuntimeError, match="rollback"):
        with postgres_database.fragmentation_v33_attempt_commit(
            job["job_id"], job["lease_token"]
        ) as scoped:
            scoped.monitor_history.append_event(
                RUN,
                "fragmentation_v33_attempt_rolled_back",
                object_type="job",
                object_id=str(job["job_id"]),
            )
            raise RuntimeError("rollback")

    assert not any(
        event["event_type"] == "fragmentation_v33_attempt_rolled_back"
        for event in postgres_database.monitor_history.page_events(RUN, limit=20)
    )
    assert postgres_database.jobs.get_job(job["job_id"])["status"] == "running"
