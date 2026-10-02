"""Runner/scheduler behavior after cross-process Run ownership is lost."""

from __future__ import annotations

import pytest

from labeling_tool.runs.run_job_scheduler import RunJobScheduler
from labeling_tool.shared.state.run_execution_ownership import RunOwnershipLostError


def _spec():
    return {
        "run_id": "run-owner",
        "scaling": {"max_cpu_partition_workers": 1},
        "resource_tuning": {"resolved": {"unit_process_threads": 1}},
    }


def test_scheduler_rejects_recovery_before_touching_jobs_when_owner_is_lost():
    touched = []

    class Jobs:
        def recover_ready_work_package_jobs(self, _run_id):
            touched.append("recover")
            return 0

        def interrupt_run_jobs(self, _run_id):
            touched.append("interrupt")
            return 0

    scheduler = RunJobScheduler(
        _spec(),
        Jobs(),
        worker_id="worker",
        accelerator_worker_id="accelerator",
        ownership_guard=lambda: (_ for _ in ()).throw(
            RunOwnershipLostError("lost")
        ),
    )

    with pytest.raises(RunOwnershipLostError, match="lost"):
        scheduler.recover_for_resume()
    assert touched == []


def test_scheduler_rejects_dispatch_before_starting_or_leasing_work():
    touched = []

    class Jobs:
        def lease_next_job(self, *_args, **_kwargs):
            touched.append("lease")

    scheduler = RunJobScheduler(
        _spec(),
        Jobs(),
        worker_id="worker",
        accelerator_worker_id="accelerator",
        ownership_guard=lambda: (_ for _ in ()).throw(
            RunOwnershipLostError("lost")
        ),
    )

    with pytest.raises(RunOwnershipLostError, match="lost"):
        scheduler.dispatch(
            type(
                "Cycle",
                (),
                {"package_pending": True, "package_expected": True},
            )(),
            active_jobs=[],
            accelerator_active=False,
            geometry_slot_limit=1,
            start_accelerator=lambda: touched.append("accelerator"),
            start_job=lambda _job: touched.append("job"),
        )
    assert touched == []
