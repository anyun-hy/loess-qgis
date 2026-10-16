"""Deterministic PostgreSQL controls for the pre-fix and fixed publication paths."""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from psycopg2.errors import DeadlockDetected
from tests.shared.test_fragmentation_publication_locking import (
    RUN_ID,
    STREAM_ID,
    _assert_rows_lockable,
    _assert_rows_locked,
    _BoundedRunStateSession,
    _build_adjacent_publication_state,
    _dependency_rows,
    _output_arguments,
)
from tests.support import v33_prelock_baseline

from labeling_tool.shared.state.artifact_repository import ArtifactRepository
from labeling_tool.shared.state.job_repository import JobRepository
from labeling_tool.shared.state.run_state_db import RunStateDB
from loess_runtime.geometry.fragmentation_v33_work_package import (
    _finish_or_requeue_v33,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("LOESS_TEST_POSTGRES_DSN"),
    reason="deadlock controls require an explicit isolated PostgreSQL DSN",
)


class _PreFixJobs(JobRepository):
    complete_fragmentation_v33_job = v33_prelock_baseline.complete_fragmentation_v33_job


class _PreFixDatabase(RunStateDB):
    fragmentation_v33_attempt_commit = (
        v33_prelock_baseline.fragmentation_v33_attempt_commit
    )

    @classmethod
    def _from_session(cls, session):
        database = super()._from_session(session)
        database.jobs = _PreFixJobs(session)
        return database


class _ObservedSession(_BoundedRunStateSession):
    def __init__(self, *args, transaction_started, **kwargs):
        super().__init__(*args, **kwargs)
        self.transaction_started = transaction_started

    def connect(self, *, autocommit=True):
        connection = super().connect(autocommit=autocommit)
        if not autocommit:
            connection.execute("SET LOCAL lock_timeout = '10s'")
            connection.execute("SET LOCAL statement_timeout = '15s'")
            pid = int(connection.execute("SELECT pg_backend_pid()").fetchone()[0])
            self.transaction_started(pid)
        return connection


def _database_for_worker(database, database_type, transaction_started=lambda pid: None):
    return database_type._from_session(
        _ObservedSession(
            database.session.location,
            schema=database.session.schema,
            transaction_started=transaction_started,
        )
    )


def _publish_and_complete(database, partition_id, job, output):
    with database.fragmentation_v33_attempt_commit(
        int(job["job_id"]), str(job["lease_token"])
    ) as publication:
        publication.artifacts.publish_fragmentation_v33_output_pair(
            RUN_ID, STREAM_ID, partition_id, **output
        )
        assert publication.jobs.complete_fragmentation_v33_job(
            int(job["job_id"]), str(job["lease_token"])
        )


def _assert_reference_counts(database):
    with database.session.connection() as connection:
        rows = connection.execute(
            """SELECT a.artifact_id, a.ref_count, COUNT(d.artifact_id)
               FROM artifacts a LEFT JOIN artifact_dependencies d
                 ON d.artifact_id=a.artifact_id
               WHERE a.run_id=%s
               GROUP BY a.artifact_id, a.ref_count ORDER BY a.artifact_id""",
            (RUN_ID,),
        ).fetchall()
    assert rows
    assert all(int(row[1]) == int(row[2]) for row in rows)


def _assert_both_completed(database, state):
    for job in state.jobs_by_partition.values():
        assert database.jobs.get_job(int(job["job_id"]))["status"] == "ready"
    with database.session.connection() as connection:
        rows = connection.execute(
            """SELECT a.kind, a.unit_id, a.ref_count FROM artifact_dependencies d
               JOIN artifacts a ON a.artifact_id=d.artifact_id
               WHERE d.job_id=%s ORDER BY a.artifact_id""",
            (state.finalize_job_id,),
        ).fetchall()
        staged = connection.execute(
            """SELECT kind, status, ref_count FROM artifacts WHERE run_id=%s
               AND kind IN ('v33_staged_mask','v33_staged_audit')""",
            (RUN_ID,),
        ).fetchall()
    assert len(rows) == 6
    assert sorted(row[0] for row in rows) == sorted(
        ["v3_baseline_core", "v33_staged_mask", "v33_staged_audit"] * 2
    )
    assert all(int(row[2]) == 1 for row in rows)
    assert len(staged) == 4
    assert all(row[1] == "ready" and int(row[2]) == 1 for row in staged)
    assert not [
        row for row in _dependency_rows(database) if row[0] != state.finalize_job_id
    ]
    for artifact_id in state.input_ids:
        artifact = database.artifacts.get_artifact(artifact_id)
        assert int(artifact["ref_count"]) == (
            1 if artifact["kind"] == "v3_baseline_core" else 0
        )
    _assert_reference_counts(database)
    _assert_rows_lockable(database, state.input_ids)


def _report(mode, trial, database, **details):
    print(
        "V33_CONTROL_RESULT "
        + json.dumps(
            {
                "mode": mode,
                "trial": trial,
                "schema": database.session.schema,
                "baseline_revision": v33_prelock_baseline.PRE_FIX_REVISION,
                **details,
            },
            sort_keys=True,
        ),
        flush=True,
    )


@pytest.mark.parametrize("trial", range(5))
def test_pre_fix_real_publication_deadlocks_then_rolls_back_and_retries(
    tmp_path, postgres_database, monkeypatch, trial
):
    state = _build_adjacent_publication_state(postgres_database, tmp_path)
    outputs = {
        partition: _output_arguments(tmp_path, partition)
        for partition in state.jobs_by_partition
    }
    before = {
        partition: postgres_database.jobs.get_job(int(job["job_id"]))
        for partition, job in state.jobs_by_partition.items()
    }
    baseline_barrier = threading.Barrier(2, timeout=5)
    publish = ArtifactRepository.publish_fragmentation_v33_output_pair

    def publish_then_synchronize(self, *args, **kwargs):
        result = publish(self, *args, **kwargs)
        # Both real publish calls have returned while their outer transactions
        # still hold the distinct owner baselines. Do not fabricate a DB error.
        baseline_barrier.wait()
        return result

    monkeypatch.setattr(
        ArtifactRepository,
        "publish_fragmentation_v33_output_pair",
        publish_then_synchronize,
    )

    def worker(partition):
        database = _database_for_worker(postgres_database, _PreFixDatabase)
        job = state.jobs_by_partition[partition]
        try:
            _publish_and_complete(database, partition, job, outputs[partition])
        except DeadlockDetected as error:
            return partition, error
        return partition, None

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker, p) for p in state.jobs_by_partition]
        results = [future.result(timeout=15) for future in futures]

    failures = [(partition, error) for partition, error in results if error]
    assert len(failures) == 1
    victim, error = failures[0]
    survivor = next(partition for partition, error in results if error is None)
    assert error.pgcode == "40P01"
    assert "artifact_dependency_refcount_delete" in error.diag.context
    victim_job = state.jobs_by_partition[victim]
    victim_id = int(victim_job["job_id"])
    failed_statement = error.cursor.query.decode("utf-8")
    assert "DELETE FROM artifact_dependencies" in failed_statement
    assert f"job_id={victim_id}" in failed_statement
    after = postgres_database.jobs.get_job(victim_id)
    for field in (
        "status",
        "attempt",
        "lease_token",
        "lease_expires",
        "progress_current",
        "progress_total",
    ):
        assert after[field] == before[victim][field]
    assert after["status"] == "running"
    assert (
        postgres_database.jobs.get_job(
            int(state.jobs_by_partition[survivor]["job_id"])
        )["status"]
        == "ready"
    )
    dependencies = _dependency_rows(postgres_database)
    assert {artifact for job, artifact in dependencies if job == victim_id} == set(
        state.input_ids
    )
    with postgres_database.session.connection() as connection:
        assert (
            connection.execute(
                """SELECT COUNT(*) FROM artifacts WHERE run_id=%s AND unit_id=%s
               AND kind IN ('v33_staged_mask','v33_staged_audit')""",
                (RUN_ID, victim),
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM artifact_dependencies WHERE job_id=%s",
                (state.finalize_job_id,),
            ).fetchone()[0]
            == 3
        )
    _assert_reference_counts(postgres_database)
    _assert_rows_lockable(postgres_database, state.input_ids)

    # Exercise the production failure/requeue helper and the *same old code*
    # again after the competing transaction has finished.
    monkeypatch.setattr(
        ArtifactRepository, "publish_fragmentation_v33_output_pair", publish
    )
    database = _database_for_worker(postgres_database, _PreFixDatabase)
    _finish_or_requeue_v33(database, victim_job, error)
    retry = database.jobs.lease_next_fragmentation_v33(
        RUN_ID, "control-retry", lease_seconds=120, max_running=4
    )
    assert retry is not None and int(retry["job_id"]) == victim_id
    assert retry["attempt"] == 2
    _publish_and_complete(database, victim, retry, outputs[victim])
    _assert_both_completed(postgres_database, state)
    _report(
        "pre_fix",
        trial,
        postgres_database,
        sqlstate=error.pgcode,
        victim_job=victim_id,
        survivor_job=int(state.jobs_by_partition[survivor]["job_id"]),
        context=error.diag.context,
        detail=error.diag.message_detail,
        failed_statement=failed_statement,
        victim_rollback_verified=True,
        same_code_retry_attempt=2,
        references_verified=True,
    )


@pytest.mark.parametrize("trial", range(5))
def test_fixed_publication_waits_before_owner_lock_then_both_complete(
    tmp_path, postgres_database, monkeypatch, trial
):
    state = _build_adjacent_publication_state(postgres_database, tmp_path)
    outputs = {
        partition: _output_arguments(tmp_path, partition)
        for partition in state.jobs_by_partition
    }
    start = threading.Event()
    first_published = threading.Event()
    release_first = threading.Event()
    guard = threading.Lock()
    pids = {}
    first_partition = []
    publish = ArtifactRepository.publish_fragmentation_v33_output_pair

    def publish_then_hold_first(self, run_id, stream_id, partition, **kwargs):
        result = publish(self, run_id, stream_id, partition, **kwargs)
        with guard:
            first = not first_partition
            if first:
                first_partition.append(partition)
        if first:
            first_published.set()
            assert release_first.wait(timeout=10), "test did not release publisher"
        return result

    monkeypatch.setattr(
        ArtifactRepository,
        "publish_fragmentation_v33_output_pair",
        publish_then_hold_first,
    )

    def worker(partition):
        def observe(pid):
            with guard:
                pids[partition] = pid

        database = _database_for_worker(postgres_database, RunStateDB, observe)
        assert start.wait(timeout=5)
        _publish_and_complete(
            database, partition, state.jobs_by_partition[partition], outputs[partition]
        )

    executor = ThreadPoolExecutor(max_workers=2)
    futures = [executor.submit(worker, p) for p in state.jobs_by_partition]
    observation = None
    try:
        start.set()
        assert first_published.wait(timeout=5)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            with guard:
                first = first_partition[0]
                second = next(p for p in state.jobs_by_partition if p != first)
                first_pid = pids.get(first)
                second_pid = pids.get(second)
            if second_pid is not None and first_pid is not None:
                with postgres_database.session.connection() as connection:
                    row = connection.execute(
                        """SELECT wait_event_type, query, pg_blocking_pids(pid)
                           FROM pg_stat_activity WHERE pid=%s""",
                        (second_pid,),
                    ).fetchone()
                if row and row[0] == "Lock" and first_pid in row[2]:
                    observation = {
                        "wait_event_type": row[0],
                        "waiting_query": row[1],
                        "blocking_pids": list(row[2]),
                        "waiting_pid": second_pid,
                        "publisher_pid": first_pid,
                    }
                    break
            if any(future.done() for future in futures):
                for future in futures:
                    if future.done():
                        future.result()
                pytest.fail("publisher completed before the controlled release")
            threading.Event().wait(0.01)
        assert observation is not None, "second publisher never waited on input locks"
        assert "ORDER BY artifact_id FOR UPDATE" in observation["waiting_query"]
        assert "SELECT artifact_id FROM artifacts" in observation["waiting_query"]
        _assert_rows_locked(postgres_database, state.input_ids)
        assert len(first_partition) == 1
        release_first.set()
        for future in futures:
            future.result(timeout=10)
    finally:
        release_first.set()
        executor.shutdown(wait=True, cancel_futures=True)

    _assert_both_completed(postgres_database, state)
    _report(
        "fixed",
        trial,
        postgres_database,
        sqlstate=None,
        blocked_before_publication_verified=True,
        observation=observation,
        references_verified=True,
    )
