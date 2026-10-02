from __future__ import annotations

import pytest

from labeling_tool.runs.run_job_scheduler import RunJobScheduler


def _spec():
    return {
        "run_id": "run-1",
        "scaling": {
            "max_cpu_partition_workers": 4,
            "max_cpu_partition_workers_with_package": 4,
        },
        "resource_tuning": {
            "resolved": {
                "package_process_threads": 1,
                "fragmentation_v33_process_threads": 1,
                "unit_process_threads": 1,
            }
        },
        "fragmentation_regularization": {
            "enabled": True,
            "policy_id": "fragmentation_v33_configurable_absorption_v1",
            "publication": "authoritative_fusion_core",
            "max_workers": 2,
        },
    }


def test_dispatch_starts_every_lease_before_requesting_the_next_one():
    events = []
    candidates = [
        {"job_id": 20, "job_type": "fragmentation_v33"},
        {"job_id": 21, "job_type": "fragmentation_v33"},
    ]
    units = [{"job_id": 30, "job_type": "unit_fit"}]

    class Jobs:
        def interrupt_expired_jobs(self, *, run_id):
            events.append(("recover", run_id))

        def job_counts(self, _run_id, *, job_type=""):
            events.append(("counts", job_type or "all"))
            if job_type == "work_package":
                return {"queued": 1}
            if job_type == "fragmentation_v33":
                return {"queued": 2}
            return {"ready": 4}

        def lease_next_fragmentation_v33(self, *_args, **_kwargs):
            job = candidates.pop(0) if candidates else None
            events.append(("lease-v33", None if job is None else job["job_id"]))
            return job

        def lease_next_job(self, *_args, **_kwargs):
            job = units.pop(0) if units else None
            events.append(("lease-unit", None if job is None else job["job_id"]))
            return job

    scheduler = RunJobScheduler(
        _spec(),
        Jobs(),
        worker_id="qgis-test",
        accelerator_worker_id="qgis-test-accelerator",
    )
    cycle = scheduler.begin_cycle(accelerator_active=False)
    events.clear()

    outcome = scheduler.dispatch(
        cycle,
        active_jobs=[],
        accelerator_active=False,
        geometry_slot_limit=4,
        start_accelerator=lambda: events.append(("start-accelerator", None)),
        start_job=lambda job: events.append(("start-job", job["job_id"])),
    )

    assert outcome.active is True
    assert events == [
        ("start-accelerator", None),
        ("counts", "fragmentation_v33"),
        ("lease-v33", 20),
        ("start-job", 20),
        ("lease-v33", 21),
        ("start-job", 21),
        ("lease-unit", 30),
        ("start-job", 30),
        ("lease-unit", None),
    ]




def test_accelerator_crash_state_resets_only_after_completed_package():
    class Jobs:
        def __init__(self):
            self.interruptions = []

        def interrupt_work_package_worker(self, run_id, worker_id):
            self.interruptions.append((run_id, worker_id))

        def job_counts(self, _run_id, *, job_type=""):
            assert job_type == "work_package"
            return {"interrupted": 1}

    jobs = Jobs()
    scheduler = RunJobScheduler(
        _spec(),
        jobs,
        worker_id="qgis-test",
        accelerator_worker_id="qgis-test-accelerator",
    )

    first = scheduler.complete_accelerator(
        "qgis-test-accelerator", success=False, error="crash-one"
    )
    assert first.restart_attempt == 1
    scheduler.record_work_package_finished()
    assert scheduler.accelerator_crash_count == 0

    scheduler.complete_accelerator(
        "qgis-test-accelerator", success=False, error="crash-two"
    )
    scheduler.complete_accelerator(
        "qgis-test-accelerator", success=False, error="crash-three"
    )
    terminal = scheduler.complete_accelerator(
        "qgis-test-accelerator", success=False, error="crash-four"
    )

    assert terminal.terminal_error == (
        "persistent accelerator worker crashed repeatedly: crash-four"
    )
    assert scheduler.accelerator_crash_count == 3
    assert len(jobs.interruptions) == 4


@pytest.mark.parametrize(
    ("job_type", "expected_error"),
    [
        (
            "fragmentation_v33",
            "V3.3 worker exited without its atomic output commit",
        ),
        (
            "unit_confidence",
            "unit confidence worker exited without its atomic output commit",
        ),
    ],
)
@pytest.mark.parametrize(
    ("timeout_count", "expected_retry"),
    [(0, True), (2, False)],
)
def test_job_completion_keeps_atomic_gate_before_retry(
    job_type,
    expected_error,
    timeout_count,
    expected_retry,
):
    events = []
    job = {
        "job_id": 33,
        "job_type": job_type,
        "lease_token": "lease-33",
    }

    class Jobs:
        def get_job(self, job_id):
            events.append(("get", job_id))
            return {**job, "status": "running"}

        def finish_job(self, job_id, token, *, status, error):
            events.append(("finish", job_id, token, status, error))
            return True

        def requeue_failed_job(self, job_id):
            events.append(("requeue", job_id))
            return True

    scheduler = RunJobScheduler(
        _spec(),
        Jobs(),
        worker_id="qgis-test",
        accelerator_worker_id="qgis-test-accelerator",
    )

    result = scheduler.complete_job(
        job,
        success=True,
        error="",
        memory_shed=False,
        timeout_count=timeout_count,
    )

    assert result.success is False
    assert result.retried is expected_retry
    assert result.error == expected_error
    expected_events = [
        ("get", 33),
        (
            "finish",
            33,
            "lease-33",
            "failed",
            expected_error,
        ),
    ]
    if expected_retry:
        expected_events.append(("requeue", 33))
    assert events == expected_events


def test_memory_shed_is_already_resumable_and_does_not_retry():
    class Jobs:
        def get_job(self, _job_id):
            return {"status": "interrupted"}

        def requeue_failed_job(self, _job_id):
            raise AssertionError("memory shedding must not consume a retry")

    scheduler = RunJobScheduler(
        _spec(),
        Jobs(),
        worker_id="qgis-test",
        accelerator_worker_id="qgis-test-accelerator",
    )
    result = scheduler.complete_job(
        {"job_id": 9, "job_type": "unit_fit", "lease_token": "lease-9"},
        success=False,
        error="memory pressure",
        memory_shed=True,
        timeout_count=0,
    )

    assert result.retried is True
    assert result.signal_success is True
