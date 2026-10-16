"""End-to-end recovery of current V3.3 publication lock failures."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from psycopg2.errors import LockNotAvailable
from tests.support.fragmentation_v33_recovery_fixture import (
    RUN_ID,
    STREAM_ID,
    artifact_reference_rows,
    build_recovery_run,
    worker_database,
)

from labeling_tool.runs.run_job_scheduler import RunJobScheduler
from labeling_tool.shared.contracts.monitor_contract import MONITOR_EXECUTION_ENV
from labeling_tool.shared.state import job_repository as job_repository_module
from labeling_tool.shared.state import run_state_db as run_state_db_module
from labeling_tool.shared.state.publication_diagnostics import (
    publication_failure_payload,
)
from labeling_tool.shared.state.run_execution_ownership import (
    RUN_OWNER_ENV,
    RunExecutionOwnership,
)
from labeling_tool.shared.state.run_state_db import RunStateError
from loess_runtime.geometry import fragmentation_v33_work_package as work_package


def _row_locked_nowait(database, table: str, row_id: int) -> bool:
    assert table in {"jobs", "artifacts"}
    column = "job_id" if table == "jobs" else "artifact_id"
    connection = database.session.connect(autocommit=False)
    try:
        connection.execute(
            f"SELECT {column} FROM {table} WHERE {column}=%s FOR UPDATE NOWAIT",
            (int(row_id),),
        ).fetchone()
        return False
    except LockNotAvailable:
        return True
    finally:
        connection.rollback()
        connection.close()


def _job_input_ids(database, job_id: int) -> tuple[int, ...]:
    with database.session.connection() as connection:
        return tuple(
            int(row["artifact_id"])
            for row in connection.execute(
                """SELECT artifact_id FROM artifact_dependencies
                   WHERE job_id=%s ORDER BY artifact_id""",
                (int(job_id),),
            ).fetchall()
        )


def _staged_count(database, partition_id: str) -> int:
    with database.session.connection() as connection:
        return int(
            connection.execute(
                """SELECT COUNT(*) FROM artifacts WHERE run_id=%s
                   AND stream_id=%s AND unit_id=%s
                   AND kind IN ('v33_staged_mask','v33_staged_audit')""",
                (RUN_ID, STREAM_ID, str(partition_id)),
            ).fetchone()[0]
        )


def _dispatch_one(scheduler: RunJobScheduler) -> dict:
    started: list[dict] = []

    def unexpected_accelerator() -> None:
        raise AssertionError("recovery fixture has no Work Package Jobs")

    cycle = scheduler.begin_cycle(accelerator_active=False)
    result = scheduler.dispatch(
        cycle,
        active_jobs=[],
        accelerator_active=False,
        geometry_slot_limit=2,
        start_accelerator=unexpected_accelerator,
        start_job=lambda job: started.append(dict(job)),
    )
    assert result.started
    assert result.active
    assert len(started) == 1
    return started[0]


def _wait_for_job_lock(database, job_id: int, future) -> None:
    deadline = time.monotonic() + 10
    while not _row_locked_nowait(database, "jobs", job_id):
        if future.done():
            future.result()
            raise AssertionError("worker returned before publication lock wait")
        assert time.monotonic() < deadline, "worker did not enter publication scope"
        threading.Event().wait(0.01)


def _assert_final_authority(recovery, reports: list[dict]) -> None:
    database = recovery.database
    assert database.jobs.job_counts(RUN_ID, job_type="fragmentation_v33") == {
        "ready": 3
    }
    finalize_reports = [
        report for report in reports if report.get("partition_count") == 2
    ]
    assert len(finalize_reports) == 1
    report = finalize_reports[0]
    assert report["status"] == "ready"
    assert report["validation_status"] == "passed"
    assert report["production_replacement"] is True
    assert report["acceptance"]["gap_pixels"] == 0
    assert report["acceptance"]["overlap_pixels"] == 0

    canonical_root = recovery.run_dir / "fusion" / "recovery"
    assert (canonical_root / "fragmentation_v33_report.json").is_file()
    for partition in recovery.partitions:
        partition_id = str(partition["partition_id"])
        mask = canonical_root / "raster_parts" / f"{partition_id}_mask.tif"
        audit = canonical_root / "fragmentation_v33_audits" / f"{partition_id}.json"
        assert mask.is_file()
        assert audit.is_file()
        artifact = database.artifacts.artifact_for_stream_unit(
            RUN_ID,
            STREAM_ID,
            partition_id,
            "core_mask",
        )
        assert artifact is not None
        assert artifact["path"] == str(mask.resolve())
        assert artifact["sha256"] == hashlib.sha256(mask.read_bytes()).hexdigest()

    with database.session.connection() as connection:
        dependencies = int(
            connection.execute(
                """SELECT COUNT(*) FROM artifact_dependencies d
                   JOIN jobs j ON j.job_id=d.job_id
                   WHERE j.run_id=%s AND j.job_type='fragmentation_v33'""",
                (RUN_ID,),
            ).fetchone()[0]
        )
    assert dependencies == 0
    reference_rows = artifact_reference_rows(database)
    assert reference_rows
    assert all(ref_count == count for _, ref_count, count in reference_rows)


def _assert_failure_diagnostics(
    recovery,
    job: dict,
    partition_id: str,
    error: BaseException,
    *,
    phase: str,
    sqlstate: str | None,
    input_lock_completed: bool,
) -> None:
    payload = publication_failure_payload(error)
    diagnostics = payload["diagnostics"]
    assert diagnostics["event"] == "v33_job_failure"
    assert diagnostics["run_id"] == RUN_ID
    assert diagnostics["stream_id"] == STREAM_ID
    assert diagnostics["job_id"] == int(job["job_id"])
    assert diagnostics["unit_id"] == str(job["unit_id"])
    assert diagnostics["partition_id"] == partition_id
    assert diagnostics["attempt"] == int(job["attempt"])
    assert diagnostics["phase"] == phase
    assert diagnostics["sqlstate"] == sqlstate
    assert diagnostics["input_lock_completed"] is input_lock_completed
    if input_lock_completed:
        assert diagnostics["input_lock_elapsed_ms"] >= 0
    else:
        assert diagnostics["input_lock_elapsed_ms"] is None
    serialized = json.dumps(diagnostics, sort_keys=True)
    assert str(job["lease_token"]) not in serialized
    assert str(recovery.database.session.location) not in serialized


def _run_scheduler_graph(recovery, scheduler, retry, label: str) -> None:
    reports = [
        work_package.run_worker(
            recovery.spec_path,
            worker_id=f"retry-{label}",
            lease_seconds=30,
            job_id=int(retry["job_id"]),
            lease_token=str(retry["lease_token"]),
        )
    ]
    while recovery.database.jobs.job_counts(
        RUN_ID,
        job_type="fragmentation_v33",
    ) != {"ready": 3}:
        job = _dispatch_one(scheduler)
        reports.append(
            work_package.run_worker(
                recovery.spec_path,
                worker_id=f"finish-{label}",
                lease_seconds=30,
                job_id=int(job["job_id"]),
                lease_token=str(job["lease_token"]),
            )
        )
    _assert_final_authority(recovery, reports)


@pytest.mark.parametrize("failure_mode", ["lock_timeout", "lease_expiry"])
def test_publication_failure_recovers_through_scheduler_to_final_authority(
    tmp_path,
    postgres_database,
    monkeypatch,
    failure_mode,
):
    recovery = build_recovery_run(postgres_database, tmp_path)
    ownership = RunExecutionOwnership.acquire(
        recovery.database,
        RUN_ID,
        run_dir=recovery.run_dir,
        worker_id=f"recovery-{failure_mode}",
        trigger_type="start",
    )
    monkeypatch.setenv(RUN_OWNER_ENV, ownership.identity.environment_value())
    monkeypatch.setenv(MONITOR_EXECUTION_ENV, ownership.identity.execution_id)
    lock_timeout = "1s" if failure_mode == "lock_timeout" else "10s"
    worker_store = worker_database(
        recovery.database,
        lock_timeout=lock_timeout,
    )
    monkeypatch.setattr(
        work_package,
        "run_state_from_spec",
        lambda _spec: worker_store,
    )
    scheduler = RunJobScheduler(
        recovery.spec,
        recovery.database.jobs,
        worker_id=f"scheduler-{failure_mode}",
        accelerator_worker_id=f"scheduler-{failure_mode}-accelerator",
        ownership_guard=ownership.assert_current,
    )
    before_references = artifact_reference_rows(recovery.database)
    blocker = recovery.database.session.connect(autocommit=False)
    executor = ThreadPoolExecutor(max_workers=1)
    blocker_released = False
    try:
        first = _dispatch_one(scheduler)
        partition_id = str(
            recovery.database.control_graph.get_spatial_unit(
                RUN_ID,
                str(first["unit_id"]),
            )["owner_key"]
        )
        input_ids = _job_input_ids(recovery.database, int(first["job_id"]))
        assert len(input_ids) == 6
        blocker.execute(
            "SELECT artifact_id FROM artifacts WHERE artifact_id=%s FOR UPDATE",
            (input_ids[0],),
        ).fetchone()
        future = executor.submit(
            work_package.run_worker,
            recovery.spec_path,
            worker_id=f"worker-{failure_mode}",
            lease_seconds=30,
            job_id=int(first["job_id"]),
            lease_token=str(first["lease_token"]),
        )
        _wait_for_job_lock(recovery.database, int(first["job_id"]), future)

        if failure_mode == "lock_timeout":
            with pytest.raises(LockNotAvailable) as failure:
                future.result(timeout=10)
            _assert_failure_diagnostics(
                recovery,
                first,
                partition_id,
                failure.value,
                phase="input_lock",
                sqlstate="55P03",
                input_lock_completed=False,
            )
            assert recovery.database.jobs.get_job(int(first["job_id"]))["status"] == (
                "queued"
            )
            retry = _dispatch_one(scheduler)
        else:
            current = recovery.database.jobs.get_job(int(first["job_id"]))
            expired_time = float(current["lease_expires"]) + 1.0
            future_clock = SimpleNamespace(time=lambda: expired_time)
            with monkeypatch.context() as clock_patch:
                clock_patch.setattr(run_state_db_module, "time", future_clock)
                clock_patch.setattr(job_repository_module, "time", future_clock)
                blocker.rollback()
                blocker_released = True
                with pytest.raises(
                    RunStateError,
                    match="lease expired while waiting for artifact inputs",
                ) as failure:
                    future.result(timeout=10)
                _assert_failure_diagnostics(
                    recovery,
                    first,
                    partition_id,
                    failure.value,
                    phase="lease_recheck",
                    sqlstate=None,
                    input_lock_completed=True,
                )
                assert (
                    recovery.database.jobs.get_job(int(first["job_id"]))["status"]
                    == "running"
                )
                retry = _dispatch_one(scheduler)

        if not blocker_released:
            blocker.rollback()
            blocker_released = True
        assert int(retry["job_id"]) == int(first["job_id"])
        assert _staged_count(recovery.database, partition_id) == 0
        assert artifact_reference_rows(recovery.database) == before_references

        _run_scheduler_graph(recovery, scheduler, retry, failure_mode)
    finally:
        if not blocker_released:
            blocker.rollback()
        blocker.close()
        executor.shutdown(wait=True, cancel_futures=True)
        ownership.close()


def test_killed_private_publication_rolls_back_and_scheduler_retries(
    tmp_path,
    postgres_database,
    monkeypatch,
    request,
):
    recovery = build_recovery_run(postgres_database, tmp_path)
    ownership = RunExecutionOwnership.acquire(
        recovery.database,
        RUN_ID,
        run_dir=recovery.run_dir,
        worker_id="recovery-kill",
        trigger_type="start",
    )
    monkeypatch.setenv(RUN_OWNER_ENV, ownership.identity.environment_value())
    monkeypatch.setenv(MONITOR_EXECUTION_ENV, ownership.identity.execution_id)
    request.addfinalizer(ownership.close)
    scheduler = RunJobScheduler(
        recovery.spec,
        recovery.database.jobs,
        worker_id="scheduler-kill",
        accelerator_worker_id="scheduler-kill-accelerator",
        ownership_guard=ownership.assert_current,
    )
    first = _dispatch_one(scheduler)
    partition_id = str(
        recovery.database.control_graph.get_spatial_unit(
            RUN_ID,
            str(first["unit_id"]),
        )["owner_key"]
    )
    before_references = artifact_reference_rows(recovery.database)
    marker = tmp_path / "private-publication-open"
    release = tmp_path / "private-publication-release"
    payload = {
        "spec_path": str(recovery.spec_path),
        "job_id": int(first["job_id"]),
        "lease_token": str(first["lease_token"]),
        "marker": str(marker),
        "release": str(release),
    }
    environment = os.environ.copy()
    environment["LOESS_RECOVERY_WORKER"] = json.dumps(payload)
    repository_root = Path(__file__).resolve().parents[2]
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repository_root / "src"), environment.get("PYTHONPATH", "")]
    )
    script = """
import json, os, time
from pathlib import Path
from labeling_tool.shared.state.artifact_repository import ArtifactRepository
from loess_runtime.geometry.fragmentation_v33_work_package import run_worker

value = json.loads(os.environ['LOESS_RECOVERY_WORKER'])
original = ArtifactRepository.publish_fragmentation_v33_output_pair

def publish_then_hold(self, *args, **kwargs):
    result = original(self, *args, **kwargs)
    Path(value['marker']).write_text('open', encoding='utf-8')
    deadline = time.monotonic() + 20
    while not Path(value['release']).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('parent did not release private publication')
        time.sleep(0.02)
    return result

ArtifactRepository.publish_fragmentation_v33_output_pair = publish_then_hold
run_worker(
    value['spec_path'],
    worker_id='killed-private-worker',
    lease_seconds=30,
    job_id=value['job_id'],
    lease_token=value['lease_token'],
)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=str(repository_root),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 15
        while not marker.exists() and process.poll() is None:
            assert time.monotonic() < deadline, (
                "child did not reach private publication"
            )
            threading.Event().wait(0.02)
        if process.poll() is not None:
            _, stderr = process.communicate(timeout=2)
            pytest.fail(f"private publication worker exited before marker: {stderr}")

        # The real publisher has returned inside the child, but its outer
        # attempt transaction is still open, so no other connection can see it.
        assert _staged_count(recovery.database, partition_id) == 0
        current = recovery.database.jobs.get_job(int(first["job_id"]))
        assert current["status"] == "running"
        assert current["lease_token"] == first["lease_token"]
        process.terminate()
        process.wait(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)

    deadline = time.monotonic() + 5
    while _row_locked_nowait(recovery.database, "jobs", int(first["job_id"])):
        assert time.monotonic() < deadline, "child connection did not release Job lock"
        threading.Event().wait(0.02)
    assert _staged_count(recovery.database, partition_id) == 0
    assert artifact_reference_rows(recovery.database) == before_references
    current = recovery.database.jobs.get_job(int(first["job_id"]))
    assert current["status"] == "running"
    assert current["lease_token"] == first["lease_token"]

    completion = scheduler.complete_job(
        first,
        success=False,
        error=f"worker terminated with exit code {process.returncode}",
        memory_shed=False,
        timeout_count=0,
    )
    assert completion.retried
    assert recovery.database.jobs.get_job(int(first["job_id"]))["status"] == "queued"
    retry = _dispatch_one(scheduler)
    assert int(retry["job_id"]) == int(first["job_id"])
    _run_scheduler_graph(recovery, scheduler, retry, "kill")
    ownership.close()
