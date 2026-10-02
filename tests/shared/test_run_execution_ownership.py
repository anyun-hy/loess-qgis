"""Cross-process Run ownership, fencing, and filesystem handover tests."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from labeling_tool.shared.state.run_execution_ownership import (
    RUN_OWNER_ENV,
    RunExecutionOwnership,
    RunOwnershipConflictError,
    RunOwnershipLostError,
    execution_identity_from_environment,
    execution_lock_keys,
)
from labeling_tool.shared.state.run_state_db import RunStateDB
from labeling_tool.shared.state.run_state_session import RunStateError

RUN_ID = "run-owner-test"


def _peer(database) -> RunStateDB:
    return RunStateDB(
        database.session.location,
        postgres_schema=database.session.schema,
    )


def _create_run(database, tmp_path: Path) -> Path:
    run_dir = tmp_path / RUN_ID
    (run_dir / "tmp").mkdir(parents=True)
    database.run_streams.create_run(RUN_ID, "a" * 64, status="planned")
    return run_dir


def _wait_for(path: Path, process: subprocess.Popen[str], timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(f"probe exited early: {stdout}\n{stderr}")
        time.sleep(0.02)
    process.kill()
    stdout, stderr = process.communicate()
    raise AssertionError(f"probe did not become ready: {stdout}\n{stderr}")


def test_lock_key_is_stable_and_schema_scoped():
    assert execution_lock_keys("schema-a", RUN_ID) == execution_lock_keys(
        "schema-a", RUN_ID
    )
    assert execution_lock_keys("schema-a", RUN_ID) != execution_lock_keys(
        "schema-b", RUN_ID
    )


@pytest.mark.parametrize("raw", ["{}", "[]", '{"run_id":"run-owner-test"}'])
def test_incomplete_worker_owner_binding_is_rejected(monkeypatch, raw):
    monkeypatch.setenv(RUN_OWNER_ENV, raw)
    with pytest.raises(RunOwnershipLostError, match="environment is invalid"):
        execution_identity_from_environment(schema="test", execution_id="execution")


def test_owner_publication_cannot_open_outside_schema_or_health_connections(
    postgres_database, tmp_path
):
    run_dir = _create_run(postgres_database, tmp_path)
    owner = RunExecutionOwnership.acquire(
        postgres_database,
        RUN_ID,
        run_dir=run_dir,
        worker_id="owner-one",
        trigger_type="start",
    )
    try:
        with postgres_database.owner_publication(RUN_ID, run_dir) as publication:
            assert publication.session.unit_identity is None
            with pytest.raises(RunStateError, match="cannot initialize schemas"):
                publication.initialize()
            with pytest.raises(RunStateError, match="cannot run health checks"):
                publication.pragmas()
    finally:
        owner.close()


def test_second_connection_is_rejected_before_any_recovery_write(
    postgres_database, tmp_path
):
    run_dir = _create_run(postgres_database, tmp_path)
    owner = RunExecutionOwnership.acquire(
        postgres_database,
        RUN_ID,
        run_dir=run_dir,
        worker_id="owner-one",
        trigger_type="start",
    )
    peer = _peer(postgres_database)
    try:
        with pytest.raises(RunOwnershipConflictError, match="already owned"):
            RunExecutionOwnership.acquire(
                peer,
                RUN_ID,
                run_dir=run_dir,
                worker_id="owner-two",
                trigger_type="resume",
            )
        assert peer.run_streams.get_run(RUN_ID)["status"] == "planned"
        assert peer.jobs.job_counts(RUN_ID) == {}
        with peer.session.connection() as connection:
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM monitor_executions WHERE run_id=%s",
                    (RUN_ID,),
                ).fetchone()[0]
                == 1
            )
    finally:
        owner.close()
    assert (run_dir / "tmp" / "run_publication.lock").is_file()
    with pytest.raises(RunOwnershipLostError):
        postgres_database.monitor_history.append_event(
            RUN_ID,
            "late_callback_after_owner_close",
        )


def test_independent_process_cannot_acquire_the_same_run_owner(
    postgres_database, tmp_path
):
    run_dir = _create_run(postgres_database, tmp_path)
    owner = RunExecutionOwnership.acquire(
        postgres_database,
        RUN_ID,
        run_dir=run_dir,
        worker_id="owner-one",
        trigger_type="start",
    )
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.pop("LOESS_RUN_EXECUTION_OWNER", None)
    environment.pop("LOESS_MONITOR_EXECUTION_ID", None)
    environment["PYTHONPATH"] = str(root / "src")
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "tests" / "support" / "run_owner_acquire_probe.py"),
                postgres_database.session.location,
                postgres_database.session.schema,
                RUN_ID,
                str(run_dir),
            ],
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
        assert completed.returncode == 23, completed.stdout + completed.stderr
        assert "already owned" in completed.stdout
        with postgres_database.session.connection() as connection:
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM monitor_executions WHERE run_id=%s",
                    (RUN_ID,),
                ).fetchone()[0]
                == 1
            )
    finally:
        owner.close()


def test_lost_lock_backend_fences_old_transactions_before_new_owner_begins(
    postgres_database, tmp_path
):
    run_dir = _create_run(postgres_database, tmp_path)
    owner = RunExecutionOwnership.acquire(
        postgres_database,
        RUN_ID,
        run_dir=run_dir,
        worker_id="owner-one",
        trigger_type="start",
    )
    peer = _peer(postgres_database)
    with peer.session.connection() as connection:
        assert connection.execute(
            "SELECT pg_terminate_backend(%s)",
            (owner.identity.lock_backend_pid,),
        ).fetchone()[0]

    with pytest.raises(RunOwnershipLostError, match="connection was lost"):
        postgres_database.run_streams.set_run_status(
            RUN_ID,
            "running",
            expected=("planned",),
        )
    assert peer.run_streams.get_run(RUN_ID)["status"] == "planned"
    with peer.session.connection() as connection:
        row = connection.execute(
            """SELECT status FROM monitor_executions
               WHERE run_id=%s AND execution_id=%s""",
            (RUN_ID, owner.identity.execution_id),
        ).fetchone()
    assert row["status"] == "running"
    owner.close()


def test_file_lock_handover_survives_owner_backend_loss_with_old_worker_alive(
    postgres_database, tmp_path
):
    run_dir = _create_run(postgres_database, tmp_path)
    owner = RunExecutionOwnership.acquire(
        postgres_database,
        RUN_ID,
        run_dir=run_dir,
        worker_id="owner-one",
        trigger_type="start",
    )
    ready = tmp_path / "probe.ready"
    release = tmp_path / "probe.release"
    output = run_dir / "canonical.txt"
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    probe = subprocess.Popen(
        [
            sys.executable,
            str(root / "tests" / "support" / "run_owner_publication_probe.py"),
            postgres_database.session.location,
            postgres_database.session.schema,
            RUN_ID,
            str(run_dir),
            owner.identity.environment_value(),
            str(ready),
            str(release),
            str(output),
        ],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_for(ready, probe)
        peer = _peer(postgres_database)
        with peer.session.connection() as connection:
            assert connection.execute(
                "SELECT pg_terminate_backend(%s)",
                (owner.identity.lock_backend_pid,),
            ).fetchone()[0]

        with pytest.raises(RunOwnershipConflictError, match="file publication"):
            RunExecutionOwnership.acquire(
                peer,
                RUN_ID,
                run_dir=run_dir,
                worker_id="owner-two",
                trigger_type="resume",
            )
        assert output.exists() is False

        release.write_text("release", encoding="utf-8")
        stdout, stderr = probe.communicate(timeout=10)
        assert probe.returncode == 0, stdout + stderr
        assert output.read_text(encoding="utf-8") == "old-owner-publication"

        replacement = RunExecutionOwnership.acquire(
            peer,
            RUN_ID,
            run_dir=run_dir,
            worker_id="owner-two",
            trigger_type="resume",
        )
        try:
            late = run_dir / "late-old-owner.txt"
            with pytest.raises(RunOwnershipLostError):
                with postgres_database.owner_publication(RUN_ID, run_dir):
                    late.write_text("must-not-write", encoding="utf-8")
            assert late.exists() is False
        finally:
            replacement.close()
    finally:
        owner.close()
        if probe.poll() is None:
            probe.kill()
            probe.communicate()


def test_owner_publication_nests_v33_lease_fence_before_file_write(
    postgres_database, tmp_path
):
    run_dir = _create_run(postgres_database, tmp_path)
    postgres_database.jobs.insert_jobs(
        RUN_ID,
        [
            {
                "job_type": "fragmentation_v33",
                "stream_id": "fusion:a",
                "unit_id": "fragmentation_v33_finalize",
            }
        ],
    )
    with postgres_database.session.connection() as connection:
        job_id = int(
            connection.execute(
                "SELECT job_id FROM jobs WHERE run_id=%s", (RUN_ID,)
            ).fetchone()[0]
        )
    job = postgres_database.jobs.lease_job(job_id, "v33-finalize", lease_seconds=120)
    assert job is not None
    owner = RunExecutionOwnership.acquire(
        postgres_database,
        RUN_ID,
        run_dir=run_dir,
        worker_id="owner-one",
        trigger_type="start",
    )
    writes = []
    try:
        with pytest.raises(RunStateError, match="no longer owns its lease"):
            with postgres_database.owner_publication(RUN_ID, run_dir) as publication:
                with publication.fragmentation_v33_attempt_commit(
                    job_id, "stale-token"
                ):
                    writes.append("canonical-file")
        assert writes == []

        with postgres_database.owner_publication(RUN_ID, run_dir) as publication:
            with publication.fragmentation_v33_attempt_commit(
                job_id, job["lease_token"]
            ) as attempt:
                assert attempt is publication
                attempt.monitor_history.append_event(
                    RUN_ID,
                    "v33_finalize_owner_and_lease_fenced",
                    object_type="job",
                    object_id=str(job_id),
                )
    finally:
        owner.close()

    assert any(
        event["event_type"] == "v33_finalize_owner_and_lease_fenced"
        for event in _peer(postgres_database).monitor_history.page_events(
            RUN_ID, limit=20
        )
    )
