"""Lifecycle for one already-leased persistent-worker Work Package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Protocol

from loess_runtime.inference.score_batch_cache import ScoreBatchDiskReserveError
from loess_runtime.system.runtime_errors import (
    LeaseLostError,
    WorkerStopRequested,
    storage_error_is_transient,
)
from loess_runtime.system.storage_guard import StorageReserveError


class LeaseHeartbeat(Protocol):
    """Heartbeat operations owned by one exact Package lease."""

    @property
    def heartbeat_count(self) -> int: ...

    def start(self) -> None: ...

    def check(self) -> None: ...

    def update_progress(self, current: int, total: int) -> None: ...

    def close(self) -> None: ...


class StopSignal(Protocol):
    """Minimum stop-event behavior needed during a low-disk pause."""

    def is_set(self) -> bool: ...

    def wait(self, timeout: float | None = None) -> bool: ...


SessionStatus = Literal["ready", "failed", "stopped", "fenced"]
ExecutePackage = Callable[[Callable[[], None], Callable[[int, int], None]], object]
PauseObserver = Callable[[int, BaseException], None]
LeaseRepair = Callable[[BaseException], None]


@dataclass(frozen=True, slots=True)
class LeasedPackageSessionResult:
    """Outcome and absolute counters for one already-leased Package."""

    status: SessionStatus
    low_disk_pause_count: int
    heartbeat_count: int


def _best_effort_repair(repair: LeaseRepair, error: BaseException) -> None:
    try:
        repair(error)
    except Exception:
        # The original session failure determines the result. Recovery can
        # later settle a lease when this exact repair attempt also fails.
        pass


def execute_leased_package_session(
    *,
    heartbeat: LeaseHeartbeat,
    execute_package: ExecutePackage,
    stop_event: StopSignal,
    low_disk_poll_sec: float,
    low_disk_pause_count: int,
    pause_observer: PauseObserver,
    repair_leased_state: LeaseRepair,
) -> LeasedPackageSessionResult:
    """Execute one exact lease, retrying transient storage failures in place.

    ``low_disk_pause_count`` is the absolute worker-session count on entry and
    return. Package execution owns its own ordinary failure transition; repair
    is reserved for failures in heartbeat startup and pause coordination.
    """

    pause_count = low_disk_pause_count
    status: SessionStatus = "failed"
    heartbeat_count = 0
    try:
        heartbeat.start()
        while not stop_event.is_set():
            try:
                execute_package(heartbeat.check, heartbeat.update_progress)
            except (ScoreBatchDiskReserveError, StorageReserveError) as error:
                if not storage_error_is_transient(error):
                    status = "failed"
                    break
                pause_count += 1
                pause_observer(pause_count, error)
                if stop_event.wait(max(0.05, float(low_disk_poll_sec))):
                    status = "stopped"
                    break
                # Keep the exact lease and reuse committed checkpoints.
                heartbeat.check()
                continue
            except WorkerStopRequested:
                status = "stopped"
                break
            except LeaseLostError:
                status = "fenced"
                break
            except Exception:
                # Package execution already owns its ordinary failure/retry
                # transition and must not be repaired a second time here.
                status = "failed"
                break
            status = "ready"
            break
        else:
            status = "stopped"
    except WorkerStopRequested:
        status = "stopped"
    except LeaseLostError:
        status = "fenced"
    except Exception as error:
        _best_effort_repair(repair_leased_state, error)
        status = "failed"
    finally:
        # Sampling precedes close because close may reset or invalidate the
        # heartbeat object. A close failure remains visible to the caller.
        heartbeat_count = heartbeat.heartbeat_count
        heartbeat.close()

    return LeasedPackageSessionResult(
        status=status,
        low_disk_pause_count=pause_count,
        heartbeat_count=heartbeat_count,
    )
