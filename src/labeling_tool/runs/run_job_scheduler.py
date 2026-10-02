"""Job leasing and completion state for the bounded v5 runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol


class JobStore(Protocol):
    """Database operations used by the runner's Job lifecycle owner."""

    def interrupt_expired_jobs(self, *, run_id: str) -> int: ...

    def job_counts(self, run_id: str, *, job_type: str = "") -> dict[str, int]: ...

    def lease_next_fragmentation_v33(
        self,
        run_id: str,
        worker_id: str,
        *,
        lease_seconds: int,
        max_running: int,
        exclude_job_ids: tuple[int, ...] = (),
    ) -> dict[str, Any] | None: ...

    def lease_next_job(
        self,
        run_id: str,
        worker_id: str,
        *,
        job_types: tuple[str, ...],
        lease_seconds: int,
        exclude_job_ids: tuple[int, ...] = (),
    ) -> dict[str, Any] | None: ...

    def interrupt_work_package_worker(self, run_id: str, worker_id: str) -> int: ...

    def interrupt_job(self, job_id: int, lease_token: str) -> bool: ...

    def get_job(self, job_id: int) -> dict[str, Any] | None: ...

    def heartbeat(
        self,
        job_id: int,
        lease_token: str,
        *,
        current: int,
        total: int,
        lease_seconds: int,
    ) -> bool: ...

    def finish_job(
        self,
        job_id: int,
        lease_token: str,
        *,
        status: str,
        error: str,
    ) -> bool: ...

    def requeue_failed_job(self, job_id: int) -> bool: ...

    def recover_ready_work_package_jobs(self, run_id: str) -> int: ...

    def interrupt_run_jobs(self, run_id: str) -> int: ...


StartAccelerator = Callable[[], None]
StartJob = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class ScheduleCycle:
    """Package state captured before cleanup, disk, and memory gates."""

    package_counts: dict[str, int]
    package_pending: bool
    package_expected: bool
    terminal_error: str = ""


@dataclass(frozen=True)
class DispatchResult:
    """Outcome of one post-admission dispatch pass."""

    started: bool = False
    active: bool = False
    complete: bool = False
    terminal_error: str = ""
    blocked_counts: dict[str, int] | None = None


@dataclass(frozen=True)
class AcceleratorCompletion:
    """State transition after the persistent accelerator process exits."""

    success: bool
    error: str
    terminal_error: str = ""
    restart_attempt: int = 0
    should_schedule: bool = False


@dataclass(frozen=True)
class JobCompletion:
    """Database result after one geometry Job process exits."""

    success: bool
    error: str
    retried: bool

    @property
    def signal_success(self) -> bool:
        return self.success or self.retried


@dataclass(frozen=True)
class JobHeartbeat:
    """Observed lease state after one heartbeat attempt."""

    state: str

    @property
    def lease_lost(self) -> bool:
        return self.state == "lease_lost"


def resource_value(spec: Mapping[str, Any], key: str, default: int) -> int:
    tuning = (spec.get("resource_tuning") or {}).get("resolved") or {}
    return max(1, int(tuning.get(key, default)))


def fragmentation_v33_process_threads(spec: Mapping[str, Any]) -> int:
    """Return the frozen CPU-thread reservation of one V3.3 worker."""

    return resource_value(
        spec,
        "fragmentation_v33_process_threads",
        resource_value(spec, "package_process_threads", 2),
    )


def unit_fit_process_threads(spec: Mapping[str, Any]) -> int:
    """Return the frozen CPU-thread reservation of one unit-fit worker."""

    return resource_value(spec, "unit_process_threads", 1)


def geometry_thread_budget(spec: Mapping[str, Any], *, package_active: bool) -> int:
    """Return the frozen CPU-thread ceiling available to geometry work."""

    scaling = spec.get("scaling") or {}
    full = max(1, int(scaling.get("max_cpu_partition_workers", 2)))
    if not package_active:
        return full
    return max(
        1,
        int(scaling.get("max_cpu_partition_workers_with_package", full)),
    )


def cpu_worker_limit(
    spec: Mapping[str, Any],
    *,
    package_active: bool,
    fragmentation_v33_active: int = 0,
    geometry_slot_limit: int | None = None,
) -> int:
    """Return the unit-fit capacity left in the shared CPU budget."""

    geometry_budget = geometry_thread_budget(spec, package_active=package_active)
    if geometry_slot_limit is not None:
        geometry_budget = min(geometry_budget, max(0, int(geometry_slot_limit)))
    remaining_threads = max(
        0,
        geometry_budget
        - max(0, int(fragmentation_v33_active))
        * fragmentation_v33_process_threads(spec),
    )
    return remaining_threads // unit_fit_process_threads(spec)


def fragmentation_v33_worker_limit(
    spec: Mapping[str, Any],
    *,
    package_active: bool,
    unit_fit_active: int = 0,
    geometry_slot_limit: int | None = None,
) -> int:
    """Bound V3.3 workers so all active CPU processes fit the budget."""

    scaling = spec.get("scaling") or {}
    full = max(1, int(scaling.get("max_cpu_partition_workers", 2)))
    if geometry_slot_limit is None:
        package_reservation = (
            resource_value(spec, "package_process_threads", 2) if package_active else 0
        )
        geometry_budget = max(0, full - package_reservation)
    else:
        geometry_budget = max(0, min(full, int(geometry_slot_limit)))
    unit_reservation = max(0, int(unit_fit_active)) * unit_fit_process_threads(spec)
    available = max(0, geometry_budget - unit_reservation)
    return available // fragmentation_v33_process_threads(spec)


def minimum_geometry_slots(spec: Mapping[str, Any]) -> int:
    """Return the smallest slot budget that can admit any configured Job."""

    minimum = unit_fit_process_threads(spec)
    fragmentation = dict(spec.get("fragmentation_regularization") or {})
    if _v33_enabled(fragmentation):
        minimum = max(minimum, fragmentation_v33_process_threads(spec))
    return minimum


def _v33_enabled(fragmentation: Mapping[str, Any]) -> bool:
    return bool(
        fragmentation.get("enabled") is True
        and fragmentation.get("policy_id")
        == "fragmentation_v33_configurable_absorption_v1"
        and fragmentation.get("publication") == "authoritative_fusion_core"
    )


class RunJobScheduler:
    """Own Job leases, completion state, and accelerator crash state."""

    def __init__(
        self,
        spec: Mapping[str, Any],
        jobs: JobStore,
        *,
        worker_id: str,
        accelerator_worker_id: str,
        ownership_guard: Callable[[], None] | None = None,
    ) -> None:
        self._spec = spec
        self._jobs = jobs
        self._run_id = str(spec["run_id"])
        self._worker_id = str(worker_id)
        self._accelerator_worker_id = str(accelerator_worker_id)
        self._ownership_guard = ownership_guard
        self.reset()

    @property
    def accelerator_done(self) -> bool:
        return self._accelerator_done

    @property
    def accelerator_crash_count(self) -> int:
        return self._accelerator_crash_count

    def reset(self) -> None:
        self._accelerator_done = False
        self._accelerator_crash_count = 0

    def _guard_ownership(self) -> None:
        if self._ownership_guard is not None:
            self._ownership_guard()

    def recover_for_resume(self) -> int:
        self._guard_ownership()
        recovered = self._jobs.recover_ready_work_package_jobs(self._run_id)
        self._jobs.interrupt_run_jobs(self._run_id)
        return recovered

    def begin_cycle(self, *, accelerator_active: bool) -> ScheduleCycle:
        self._guard_ownership()
        recover_expired = getattr(self._jobs, "interrupt_expired_jobs", None)
        if callable(recover_expired):
            recover_expired(run_id=self._run_id)
        package_counts = dict(
            self._jobs.job_counts(self._run_id, job_type="work_package")
        )
        if int(package_counts.get("failed", 0)):
            return ScheduleCycle(
                package_counts,
                False,
                bool(accelerator_active),
                "Work Package exhausted retries; remaining work was stopped: "
                + str(package_counts),
            )
        package_pending = any(
            package_counts.get(status, 0)
            for status in ("queued", "interrupted", "running")
        )
        return ScheduleCycle(
            package_counts,
            bool(package_pending),
            bool(
                accelerator_active or (not self._accelerator_done and package_pending)
            ),
        )

    def dispatch(
        self,
        cycle: ScheduleCycle,
        *,
        active_jobs: list[dict[str, Any]],
        accelerator_active: bool,
        geometry_slot_limit: int,
        start_accelerator: StartAccelerator,
        start_job: StartJob,
    ) -> DispatchResult:
        """Lease and immediately start each admitted process in contract order."""

        self._guard_ownership()

        unit_active = sum(
            1
            for job in active_jobs
            if job.get("job_type") in {"unit_fit", "unit_confidence"}
        )
        candidate_active = sum(
            1 for job in active_jobs if job.get("job_type") == "fragmentation_v33"
        )
        active_job_ids = tuple(
            sorted(
                {
                    int(job["job_id"])
                    for job in active_jobs
                    if job.get("job_id") is not None
                }
            )
        )
        started = False
        if not self._accelerator_done and not accelerator_active:
            if cycle.package_pending:
                start_accelerator()
                started = True
                accelerator_active = True
            else:
                self._accelerator_done = True

        fragmentation = dict(self._spec.get("fragmentation_regularization") or {})
        if _v33_enabled(fragmentation):
            candidate_counts = dict(
                self._jobs.job_counts(
                    self._run_id,
                    job_type="fragmentation_v33",
                )
            )
            if int(candidate_counts.get("failed", 0)):
                return DispatchResult(
                    started=started,
                    active=bool(started or accelerator_active or active_jobs),
                    terminal_error="V3.3 exhausted retries: " + str(candidate_counts),
                )
            configured_limit = min(
                2 if accelerator_active else 4,
                max(1, int(fragmentation.get("max_workers", 4))),
            )
            candidate_limit = min(
                configured_limit,
                fragmentation_v33_worker_limit(
                    self._spec,
                    package_active=accelerator_active,
                    unit_fit_active=unit_active,
                    geometry_slot_limit=geometry_slot_limit,
                ),
            )
            leased_candidate_ids: set[int] = set()
            while candidate_active < candidate_limit:
                job = self._jobs.lease_next_fragmentation_v33(
                    self._run_id,
                    self._worker_id + f"-fragmentation-v33-{candidate_active}",
                    lease_seconds=300,
                    max_running=candidate_limit,
                    exclude_job_ids=active_job_ids,
                )
                if not job:
                    break
                job_id = int(job["job_id"])
                if job_id in leased_candidate_ids:
                    raise RuntimeError("state backend leased one V3.3 job twice")
                leased_candidate_ids.add(job_id)
                start_job(job)
                candidate_active += 1
                started = True

        cpu_limit = cpu_worker_limit(
            self._spec,
            package_active=accelerator_active,
            fragmentation_v33_active=candidate_active,
            geometry_slot_limit=geometry_slot_limit,
        )
        while unit_active < cpu_limit:
            job = self._jobs.lease_next_job(
                self._run_id,
                self._worker_id + f"-geometry-{unit_active}",
                job_types=("unit_confidence", "unit_fit"),
                lease_seconds=300,
                exclude_job_ids=active_job_ids,
            )
            if not job:
                break
            start_job(job)
            unit_active += 1
            started = True

        if started or accelerator_active or active_jobs:
            return DispatchResult(started=started, active=True)

        counts = dict(self._jobs.job_counts(self._run_id))
        if counts.get("failed"):
            return DispatchResult(terminal_error=f"v5 jobs exhausted retries: {counts}")
        if counts.get("queued") or counts.get("interrupted") or counts.get("running"):
            return DispatchResult(blocked_counts=counts)
        return DispatchResult(complete=True)

    def record_work_package_finished(self) -> None:
        self._accelerator_crash_count = 0

    def complete_accelerator(
        self,
        worker_id: str,
        *,
        success: bool,
        error: str,
    ) -> AcceleratorCompletion:
        self._guard_ownership()
        if not success:
            self.interrupt_accelerator(worker_id)
        package_counts = dict(
            self._jobs.job_counts(self._run_id, job_type="work_package")
        )
        if int(package_counts.get("failed", 0)):
            self._accelerator_done = True
            terminal_error = (
                "Work Package exhausted retries; remaining work was stopped: "
                + str(package_counts)
            )
            return AcceleratorCompletion(False, terminal_error, terminal_error)

        package_pending = any(
            package_counts.get(status, 0)
            for status in ("queued", "interrupted", "running")
        )
        if success and package_pending:
            success = False
            error = "accelerator_worker exited while Work Packages remain: " + str(
                package_counts
            )
            self.interrupt_accelerator(worker_id)
            package_counts = dict(
                self._jobs.job_counts(self._run_id, job_type="work_package")
            )
            package_pending = any(
                package_counts.get(status, 0)
                for status in ("queued", "interrupted", "running")
            )

        if success:
            self._accelerator_done = True
            self._accelerator_crash_count = 0
            return AcceleratorCompletion(True, error, should_schedule=True)
        if package_pending:
            self._accelerator_crash_count += 1
            if self._accelerator_crash_count >= 3:
                terminal_error = (
                    "persistent accelerator worker crashed repeatedly: " + error
                )
                return AcceleratorCompletion(False, error, terminal_error)
            return AcceleratorCompletion(
                False,
                error,
                restart_attempt=self._accelerator_crash_count,
                should_schedule=True,
            )

        self._accelerator_done = True
        return AcceleratorCompletion(False, error, should_schedule=True)

    def complete_job(
        self,
        job: Mapping[str, Any],
        *,
        success: bool,
        error: str,
        memory_shed: bool,
        timeout_count: int,
        lease_lost: bool = False,
    ) -> JobCompletion:
        self._guard_ownership()
        current = self._jobs.get_job(job["job_id"])
        if current and current["status"] == "running":
            if (
                job.get("job_type")
                in {
                    "fragmentation_v33",
                    "unit_confidence",
                }
                and success
            ):
                success = False
                if job.get("job_type") == "fragmentation_v33":
                    error = "V3.3 worker exited without its atomic output commit"
                else:
                    error = (
                        "unit confidence worker exited without its atomic output commit"
                    )
            self._jobs.finish_job(
                int(job["job_id"]),
                str(job["lease_token"]),
                status="ready" if success else "failed",
                error=error,
            )

        resumable_interrupt = bool(memory_shed or lease_lost)
        retried = resumable_interrupt
        if not success and not resumable_interrupt and int(timeout_count) < 2:
            retried = self._jobs.requeue_failed_job(int(job["job_id"]))
        return JobCompletion(success, error, retried)

    def heartbeat(
        self,
        job: Mapping[str, Any],
        progress: tuple[str, int, int] | None,
        *,
        allow_database_fallback: bool,
    ) -> JobHeartbeat:
        self._guard_ownership()
        if progress is None:
            if not allow_database_fallback:
                return JobHeartbeat("skipped")
            current = self._jobs.get_job(int(job["job_id"]))
            if current and current["status"] in {"ready", "failed", "stopped"}:
                return JobHeartbeat("terminal")
            if (
                not current
                or current["status"] != "running"
                or str(current.get("lease_token") or "") != str(job["lease_token"])
            ):
                return JobHeartbeat("lease_lost")
            lease_token = str(job["lease_token"])
            progress_current = int(current["progress_current"] or 0)
            progress_total = int(current["progress_total"] or 0)
        else:
            lease_token, progress_current, progress_total = progress
        renewed = self._jobs.heartbeat(
            int(job["job_id"]),
            lease_token,
            current=progress_current,
            total=progress_total,
            lease_seconds=300,
        )
        if renewed:
            return JobHeartbeat("renewed")
        current = self._jobs.get_job(int(job["job_id"]))
        if current and current["status"] in {"ready", "failed", "stopped"}:
            return JobHeartbeat("terminal")
        return JobHeartbeat("lease_lost")

    def interrupt_job(self, job: Mapping[str, Any]) -> bool:
        self._guard_ownership()
        return bool(
            self._jobs.interrupt_job(
                int(job["job_id"]),
                str(job["lease_token"]),
            )
        )

    def interrupt_accelerator(self, worker_id: str | None = None) -> None:
        self._guard_ownership()
        self._jobs.interrupt_work_package_worker(
            self._run_id,
            str(worker_id or self._accelerator_worker_id),
        )

    def job_counts(self) -> dict[str, int]:
        self._guard_ownership()
        return dict(self._jobs.job_counts(self._run_id))
