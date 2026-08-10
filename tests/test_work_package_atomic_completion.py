import sqlite3
import time
from pathlib import Path

import pytest

from labeling_tool.core.run_state_db import RunStateDB


RUN_ID = "20260805_180000_atomic"


def _leased_package(tmp_path, *, package_id="package_00000"):
    database = RunStateDB(tmp_path / "run_state.sqlite")
    database.initialize()
    database.create_run(RUN_ID, "a" * 64)
    database.insert_work_packages(
        RUN_ID,
        [
            {
                "package_id": package_id,
                "sequence_no": 0,
                "partition_ids": [],
                "tile_windows": [],
            }
        ],
    )
    database.insert_jobs(
        RUN_ID,
        [{"job_type": "work_package", "package_id": package_id}],
    )
    job = database.lease_next_work_package(
        RUN_ID,
        "atomic-test",
        max_open_frontier_units=64,
        lease_seconds=120,
    )
    assert job is not None
    assert database.get_work_package(RUN_ID, package_id)["status"] == "running"
    return database, job


def test_package_and_exact_leased_job_commit_ready_together(tmp_path):
    database, job = _leased_package(tmp_path)

    assert not database.complete_work_package_job(
        RUN_ID,
        "package_00000",
        job["job_id"],
        "wrong-token",
    )
    assert database.get_work_package(RUN_ID, "package_00000")["status"] == "running"
    assert database.get_job(job["job_id"])["status"] == "running"
    assert not database.complete_work_package_job(
        RUN_ID,
        "package_00000",
        job["job_id"] + 1000,
        job["lease_token"],
    )
    assert database.get_work_package(RUN_ID, "package_00000")["status"] == "running"

    assert database.complete_work_package_job(
        RUN_ID,
        "package_00000",
        job["job_id"],
        job["lease_token"],
    )
    assert database.get_work_package(RUN_ID, "package_00000")["status"] == "ready"
    completed = database.get_job(job["job_id"])
    assert completed["status"] == "ready"
    assert completed["lease_token"] == ""
    assert completed["lease_expires"] is None


@pytest.mark.parametrize("target", ["ready", "failed", "interrupted"])
def test_exact_lease_transitions_package_and_job_together(tmp_path, target):
    database, job = _leased_package(tmp_path)

    assert database.transition_work_package_job(
        RUN_ID,
        "package_00000",
        job["job_id"],
        job["lease_token"],
        target,
        "injected failure",
    )
    assert database.get_work_package(RUN_ID, "package_00000")["status"] == target
    completed = database.get_job(job["job_id"])
    assert completed["status"] == target
    assert completed["error"] == ("" if target == "ready" else "injected failure")
    assert completed["worker_id"] == ""
    assert completed["lease_token"] == ""
    assert completed["lease_expires"] is None


@pytest.mark.parametrize("target", ["ready", "failed", "interrupted"])
def test_job_update_failure_rolls_back_package_transition(tmp_path, target):
    database, job = _leased_package(tmp_path)
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            f"""CREATE TRIGGER reject_package_job_transition
               BEFORE UPDATE OF status ON jobs
               WHEN NEW.status='{target}' AND OLD.job_type='work_package'
               BEGIN SELECT RAISE(ABORT, 'injected job commit failure'); END"""
        )

    with pytest.raises(sqlite3.IntegrityError, match="injected job commit failure"):
        database.transition_work_package_job(
            RUN_ID,
            "package_00000",
            job["job_id"],
            job["lease_token"],
            target,
        )

    assert database.get_work_package(RUN_ID, "package_00000")["status"] == "running"
    assert database.get_job(job["job_id"])["status"] == "running"


def test_expired_lease_cannot_transition_or_retry_package(tmp_path):
    database, job = _leased_package(tmp_path)
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            "UPDATE jobs SET lease_expires=? WHERE job_id=?",
            (time.time() - 1, job["job_id"]),
        )

    assert not database.work_package_job_holds_lease(
        RUN_ID,
        "package_00000",
        job["job_id"],
        job["lease_token"],
    )
    assert not database.transition_work_package_job(
        RUN_ID,
        "package_00000",
        job["job_id"],
        job["lease_token"],
        "failed",
        "too late",
    )
    assert database.fail_or_requeue_work_package_job(
        RUN_ID,
        "package_00000",
        job["job_id"],
        job["lease_token"],
        "too late",
    ) is None
    assert database.get_work_package(RUN_ID, "package_00000")["status"] == "running"
    assert database.get_job(job["job_id"])["status"] == "running"


def test_expired_lease_cannot_be_revived_by_late_heartbeat(tmp_path):
    database, job = _leased_package(tmp_path)
    expired_at = time.time() - 1
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            "UPDATE jobs SET lease_expires=? WHERE job_id=?",
            (expired_at, job["job_id"]),
        )

    assert not database.heartbeat(
        job["job_id"],
        job["lease_token"],
        current=1,
        total=2,
        lease_seconds=120,
    )
    unchanged = database.get_job(job["job_id"])
    assert unchanged["lease_expires"] == pytest.approx(expired_at)
    assert not database.work_package_job_holds_lease(
        RUN_ID,
        "package_00000",
        job["job_id"],
        job["lease_token"],
    )


def test_stale_worker_cannot_change_package_after_new_lease(tmp_path):
    database, old_job = _leased_package(tmp_path)
    assert database.interrupt_work_package_job(
        RUN_ID,
        "package_00000",
        old_job["job_id"],
        old_job["lease_token"],
        error="replacement requested",
    )
    new_job = database.lease_next_work_package(
        RUN_ID,
        "replacement-worker",
        max_open_frontier_units=64,
        lease_seconds=120,
    )
    assert new_job is not None
    assert new_job["lease_token"] != old_job["lease_token"]

    for target in ("ready", "failed", "interrupted"):
        assert not database.transition_work_package_job(
            RUN_ID,
            "package_00000",
            old_job["job_id"],
            old_job["lease_token"],
            target,
            "stale worker",
        )
    assert database.fail_or_requeue_work_package_job(
        RUN_ID,
        "package_00000",
        old_job["job_id"],
        old_job["lease_token"],
        "stale worker",
    ) is None
    assert database.interrupt_work_package_worker(RUN_ID, "atomic-test") == 0

    package = database.get_work_package(RUN_ID, "package_00000")
    current = database.get_job(new_job["job_id"])
    assert package["status"] == "running"
    assert current["status"] == "running"
    assert current["worker_id"] == "replacement-worker"
    assert current["lease_token"] == new_job["lease_token"]


def test_fail_or_requeue_uses_attempt_limit_atomically(tmp_path):
    database, job = _leased_package(tmp_path)

    for expected_attempt in (1, 2):
        assert job["attempt"] == expected_attempt
        assert database.heartbeat(
            job["job_id"],
            job["lease_token"],
            current=300,
            total=382,
        )
        assert database.fail_or_requeue_work_package_job(
            RUN_ID,
            "package_00000",
            job["job_id"],
            job["lease_token"],
            f"attempt {expected_attempt}",
        ) == "queued"
        assert database.get_work_package(RUN_ID, "package_00000")["status"] == "queued"
        assert database.get_job(job["job_id"])["status"] == "queued"
        assert database.get_job(job["job_id"])["error"] == ""
        job = database.lease_next_work_package(
            RUN_ID,
            f"retry-worker-{expected_attempt}",
            max_open_frontier_units=64,
            lease_seconds=120,
        )
        assert job is not None
        assert job["progress_current"] == 0
        assert job["progress_total"] == 0
        assert database.get_work_package(
            RUN_ID, "package_00000"
        )["status"] == "running"

    assert job["attempt"] == 3
    assert database.fail_or_requeue_work_package_job(
        RUN_ID,
        "package_00000",
        job["job_id"],
        job["lease_token"],
        "attempt 3",
    ) == "failed"
    assert database.get_work_package(RUN_ID, "package_00000")["status"] == "failed"
    exhausted = database.get_job(job["job_id"])
    assert exhausted["status"] == "failed"
    assert exhausted["error"] == "attempt 3"


def test_retry_job_update_failure_rolls_back_package_requeue(tmp_path):
    database, job = _leased_package(tmp_path)
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            """CREATE TRIGGER reject_package_job_requeue
               BEFORE UPDATE OF status ON jobs
               WHEN NEW.status='queued' AND OLD.job_type='work_package'
               BEGIN SELECT RAISE(ABORT, 'injected retry commit failure'); END"""
        )

    with pytest.raises(sqlite3.IntegrityError, match="injected retry commit failure"):
        database.fail_or_requeue_work_package_job(
            RUN_ID,
            "package_00000",
            job["job_id"],
            job["lease_token"],
            "retry",
        )
    assert database.get_work_package(RUN_ID, "package_00000")["status"] == "running"
    assert database.get_job(job["job_id"])["status"] == "running"


def test_interrupt_work_package_worker_updates_both_rows(tmp_path):
    database, job = _leased_package(tmp_path)

    assert database.interrupt_work_package_worker(RUN_ID, "wrong-worker") == 0
    assert database.get_work_package(RUN_ID, "package_00000")["status"] == "running"
    assert database.interrupt_work_package_worker(RUN_ID, "atomic-test") == 1
    assert database.get_work_package(RUN_ID, "package_00000")["status"] == "interrupted"
    interrupted = database.get_job(job["job_id"])
    assert interrupted["status"] == "interrupted"
    assert interrupted["worker_id"] == ""
    assert interrupted["lease_token"] == ""
    assert interrupted["lease_expires"] is None


def test_expired_package_lease_interrupts_package_and_job_together(tmp_path):
    database, job = _leased_package(tmp_path)
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            "UPDATE jobs SET lease_expires=? WHERE job_id=?",
            (time.time() - 1, job["job_id"]),
        )

    assert database.interrupt_expired_jobs(now_epoch=time.time()) == 1
    assert database.get_work_package(RUN_ID, "package_00000")["status"] == "interrupted"
    interrupted = database.get_job(job["job_id"])
    assert interrupted["status"] == "interrupted"
    assert interrupted["worker_id"] == ""
    assert interrupted["lease_token"] == ""
    assert interrupted["lease_expires"] is None


def test_interrupt_run_updates_running_package_and_job_together(tmp_path):
    database, job = _leased_package(tmp_path)

    assert database.interrupt_run_jobs(RUN_ID) == 1
    assert database.get_work_package(RUN_ID, "package_00000")["status"] == "interrupted"
    interrupted = database.get_job(job["job_id"])
    assert interrupted["status"] == "interrupted"
    assert interrupted["worker_id"] == ""
    assert interrupted["lease_token"] == ""
    assert interrupted["lease_expires"] is None


@pytest.mark.parametrize("interruption", ("worker", "expired", "run"))
def test_package_interruption_does_not_consume_last_failure_attempt(
    tmp_path, interruption
):
    database, job = _leased_package(tmp_path)
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            "UPDATE jobs SET attempt=max_attempts WHERE job_id=?",
            (job["job_id"],),
        )
        if interruption == "expired":
            connection.execute(
                "UPDATE jobs SET lease_expires=? WHERE job_id=?",
                (time.time() - 1, job["job_id"]),
            )

    if interruption == "worker":
        assert database.interrupt_work_package_worker(RUN_ID, "atomic-test") == 1
    elif interruption == "expired":
        assert database.interrupt_expired_jobs(now_epoch=time.time()) == 1
    else:
        assert database.interrupt_run_jobs(RUN_ID) == 1

    interrupted = database.get_job(job["job_id"])
    assert interrupted["status"] == "interrupted"
    assert interrupted["attempt"] == interrupted["max_attempts"] - 1
    resumed = database.lease_next_work_package(
        RUN_ID,
        "resume-worker",
        max_open_frontier_units=64,
        lease_seconds=120,
    )
    assert resumed is not None
    assert resumed["attempt"] == resumed["max_attempts"]


@pytest.mark.parametrize("job_status", ["running", "interrupted"])
def test_recovery_finalizes_legacy_ready_package_job_window(tmp_path, job_status):
    database, job = _leased_package(tmp_path)
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            "UPDATE work_packages SET status='ready' WHERE run_id=? AND package_id=?",
            (RUN_ID, "package_00000"),
        )
        if job_status == "interrupted":
            connection.execute(
                """UPDATE jobs SET status='interrupted', worker_id='', lease_token='',
                   lease_expires=NULL WHERE job_id=?""",
                (job["job_id"],),
            )

    assert database.recover_ready_work_package_jobs(RUN_ID) == 1
    recovered = database.get_job(job["job_id"])
    assert recovered["status"] == "ready"
    assert recovered["lease_token"] == ""
    assert recovered["lease_expires"] is None
    assert database.recover_ready_work_package_jobs(RUN_ID) == 0


def test_production_runner_delegates_package_leases_to_persistent_worker():
    repository = Path(__file__).resolve().parents[1]
    runner_source = (
        repository / "qgis_plugins/labeling_tool/core/v5_async_runner.py"
    ).read_text(encoding="utf-8")
    runtime_source = (
        repository / "inference_scripts/work_package_runtime.py"
    ).read_text(encoding="utf-8")

    accelerator_start = runner_source.split(
        "def _start_accelerator_worker", 1
    )[1].split(
        "def _start_assembly", 1
    )[0]
    assert '"--worker-id", self._accelerator_worker_id' in accelerator_start
    assert '"--package-id"' not in accelerator_start
    assert '"--job-id"' not in accelerator_start
    assert '"--lease-token"' not in accelerator_start
    assert "lease_next_work_package(" not in runner_source
    assert 'add_argument("--worker-id"' in runtime_source
    assert "run_persistent_worker(" in runtime_source
    assert "complete_work_package_job(" in runtime_source
