from __future__ import annotations

from collections.abc import Callable

import pytest

from loess_runtime.inference.persistent_package_session import (
    execute_leased_package_session,
)
from loess_runtime.inference.score_batch_cache import ScoreBatchDiskReserveError
from loess_runtime.system.runtime_errors import (
    LeaseLostError,
    WorkerStopRequested,
)
from loess_runtime.system.storage_guard import StorageReserveError


def _transient_score_error() -> ScoreBatchDiskReserveError:
    return ScoreBatchDiskReserveError("disk reserve", transient=True)


class _Heartbeat:
    def __init__(
        self,
        events: list[str],
        *,
        start_error: Exception | None = None,
        check_error: Exception | None = None,
        close_error: Exception | None = None,
        heartbeat_count: int = 3,
    ) -> None:
        self.events = events
        self.start_error = start_error
        self.check_error = check_error
        self.close_error = close_error
        self.heartbeat_count = heartbeat_count

    def start(self) -> None:
        self.events.append("start")
        if self.start_error is not None:
            raise self.start_error

    def check(self) -> None:
        self.events.append("check")
        if self.check_error is not None:
            raise self.check_error

    def update_progress(self, current: int, total: int) -> None:
        self.events.append(f"progress:{current}/{total}")

    def close(self) -> None:
        self.events.append("close")
        if self.close_error is not None:
            raise self.close_error


class _StopEvent:
    def __init__(
        self,
        events: list[str],
        *,
        wait_result: bool = False,
        wait_error: Exception | None = None,
    ) -> None:
        self.events = events
        self.wait_result = wait_result
        self.wait_error = wait_error
        self.set_state = False

    def is_set(self) -> bool:
        self.events.append("is_set")
        return self.set_state

    def wait(self, timeout: float | None = None) -> bool:
        self.events.append(f"wait:{timeout}")
        if self.wait_error is not None:
            raise self.wait_error
        if self.wait_result:
            self.set_state = True
        return self.wait_result


def _run_session(
    *,
    heartbeat: _Heartbeat,
    stop_event: _StopEvent,
    execute_package: Callable[[Callable[[], None], Callable[[int, int], None]], object],
    pause_count: int = 0,
    pause_observer: Callable[[int, BaseException], None] | None = None,
    repair: Callable[[BaseException], None] | None = None,
):
    return execute_leased_package_session(
        heartbeat=heartbeat,
        execute_package=execute_package,
        stop_event=stop_event,
        low_disk_poll_sec=0.01,
        low_disk_pause_count=pause_count,
        pause_observer=pause_observer or (lambda _count, _error: None),
        repair_leased_state=repair or (lambda _error: None),
    )


def test_ready_session_preserves_heartbeat_order_and_count():
    events: list[str] = []
    heartbeat = _Heartbeat(events, heartbeat_count=4)

    def execute(lease_check, progress_update):
        events.append("execute")
        lease_check()
        progress_update(2, 5)

    result = _run_session(
        heartbeat=heartbeat,
        stop_event=_StopEvent(events),
        execute_package=execute,
    )

    assert result.status == "ready"
    assert result.low_disk_pause_count == 0
    assert result.heartbeat_count == 4
    assert events == [
        "start",
        "is_set",
        "execute",
        "check",
        "progress:2/5",
        "close",
    ]


def test_transient_storage_waits_checks_and_retries_with_absolute_pause_count():
    events: list[str] = []
    attempts = 0

    def execute(_lease_check, _progress_update):
        nonlocal attempts
        attempts += 1
        events.append(f"execute:{attempts}")
        if attempts == 1:
            raise _transient_score_error()

    def observe(count: int, error: BaseException) -> None:
        events.append(f"pause:{count}:{error}")

    result = _run_session(
        heartbeat=_Heartbeat(events),
        stop_event=_StopEvent(events),
        execute_package=execute,
        pause_count=6,
        pause_observer=observe,
    )

    assert result.status == "ready"
    assert result.low_disk_pause_count == 7
    assert events == [
        "start",
        "is_set",
        "execute:1",
        "pause:7:disk reserve",
        "wait:0.05",
        "check",
        "is_set",
        "execute:2",
        "close",
    ]


def test_stop_during_low_disk_wait_does_not_check_or_retry():
    events: list[str] = []

    def execute(_lease_check, _progress_update):
        events.append("execute")
        raise _transient_score_error()

    result = _run_session(
        heartbeat=_Heartbeat(events),
        stop_event=_StopEvent(events, wait_result=True),
        execute_package=execute,
        pause_observer=lambda count, _error: events.append(f"pause:{count}"),
    )

    assert result.status == "stopped"
    assert events == [
        "start",
        "is_set",
        "execute",
        "pause:1",
        "wait:0.05",
        "close",
    ]


def test_package_ordinary_failure_is_not_repaired_again():
    events: list[str] = []
    repairs: list[BaseException] = []

    def execute(_lease_check, _progress_update):
        raise RuntimeError("package failed")

    result = _run_session(
        heartbeat=_Heartbeat(events),
        stop_event=_StopEvent(events),
        execute_package=execute,
        repair=repairs.append,
    )

    assert result.status == "failed"
    assert repairs == []
    assert events == ["start", "is_set", "close"]


def test_ordinary_error_with_transient_attribute_is_not_retried_or_repaired():
    events: list[str] = []
    repairs: list[BaseException] = []

    class OrdinaryError(RuntimeError):
        transient = True

    def execute(_lease_check, _progress_update):
        events.append("execute")
        raise OrdinaryError("ordinary package failure")

    result = _run_session(
        heartbeat=_Heartbeat(events),
        stop_event=_StopEvent(events),
        execute_package=execute,
        repair=repairs.append,
    )

    assert result.status == "failed"
    assert repairs == []
    assert events == ["start", "is_set", "execute", "close"]


@pytest.mark.parametrize(
    "storage_error",
    (
        ScoreBatchDiskReserveError("score reserve", transient=True),
        StorageReserveError(
            "tile cache",
            free_bytes=1,
            required_free_bytes=2,
            write_bytes=1,
            managed_bytes=0,
            managed_budget_bytes=0,
        ),
    ),
)
def test_both_transient_storage_error_types_retry(storage_error):
    events: list[str] = []
    attempts = 0

    def execute(_lease_check, _progress_update):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise storage_error

    result = _run_session(
        heartbeat=_Heartbeat(events),
        stop_event=_StopEvent(events),
        execute_package=execute,
    )

    assert result.status == "ready"
    assert attempts == 2


@pytest.mark.parametrize("stage", ("start", "observer", "wait", "check"))
def test_session_owned_ordinary_failure_repairs_exact_lease(stage):
    events: list[str] = []
    failure = RuntimeError(f"{stage} failed")
    repairs: list[BaseException] = []
    heartbeat = _Heartbeat(
        events,
        start_error=failure if stage == "start" else None,
        check_error=failure if stage == "check" else None,
    )
    stop_event = _StopEvent(
        events,
        wait_error=failure if stage == "wait" else None,
    )

    def execute(_lease_check, _progress_update):
        raise _transient_score_error()

    def observe(_count: int, _error: BaseException) -> None:
        if stage == "observer":
            raise failure

    result = _run_session(
        heartbeat=heartbeat,
        stop_event=stop_event,
        execute_package=execute,
        pause_observer=observe,
        repair=repairs.append,
    )

    assert result.status == "failed"
    assert repairs == [failure]
    assert events[-1] == "close"


@pytest.mark.parametrize("error_type", (WorkerStopRequested, LeaseLostError))
@pytest.mark.parametrize("stage", ("start", "package", "observer", "wait", "check"))
def test_stop_and_lease_loss_are_classified_from_every_session_stage(stage, error_type):
    events: list[str] = []
    failure = error_type(f"{stage} signal")
    repairs: list[BaseException] = []
    heartbeat = _Heartbeat(
        events,
        start_error=failure if stage == "start" else None,
        check_error=failure if stage == "check" else None,
    )
    stop_event = _StopEvent(
        events,
        wait_error=failure if stage == "wait" else None,
    )

    def execute(_lease_check, _progress_update):
        if stage == "package":
            raise failure
        raise _transient_score_error()

    def observe(_count: int, _error: BaseException) -> None:
        if stage == "observer":
            raise failure

    result = _run_session(
        heartbeat=heartbeat,
        stop_event=stop_event,
        execute_package=execute,
        pause_observer=observe,
        repair=repairs.append,
    )

    assert result.status == (
        "stopped" if error_type is WorkerStopRequested else "fenced"
    )
    assert repairs == []
    assert events[-1] == "close"


def test_repair_failure_does_not_replace_original_failed_result():
    events: list[str] = []

    def repair(_error: BaseException) -> None:
        raise RuntimeError("repair failed")

    result = _run_session(
        heartbeat=_Heartbeat(events, start_error=RuntimeError("start failed")),
        stop_event=_StopEvent(events),
        execute_package=lambda _check, _update: None,
        repair=repair,
    )

    assert result.status == "failed"
    assert events == ["start", "close"]


def test_stop_set_by_heartbeat_start_prevents_package_execution():
    events: list[str] = []
    stop_event = _StopEvent(events)

    class StartStopsHeartbeat(_Heartbeat):
        def start(self) -> None:
            super().start()
            stop_event.set_state = True

    result = _run_session(
        heartbeat=StartStopsHeartbeat(events),
        stop_event=stop_event,
        execute_package=lambda _check, _update: events.append("execute"),
    )

    assert result.status == "stopped"
    assert "execute" not in events
    assert events == ["start", "is_set", "close"]


def test_stop_set_by_retry_check_prevents_another_package_attempt():
    events: list[str] = []
    stop_event = _StopEvent(events)
    attempts = 0

    class CheckStopsHeartbeat(_Heartbeat):
        def check(self) -> None:
            super().check()
            stop_event.set_state = True

    def execute(_lease_check, _progress_update):
        nonlocal attempts
        attempts += 1
        raise _transient_score_error()

    result = _run_session(
        heartbeat=CheckStopsHeartbeat(events),
        stop_event=stop_event,
        execute_package=execute,
    )

    assert result.status == "stopped"
    assert attempts == 1
    assert events[-3:] == ["check", "is_set", "close"]


def test_heartbeat_count_is_sampled_before_close_and_close_failure_propagates():
    events: list[str] = []

    class SampledHeartbeat(_Heartbeat):
        def __init__(self) -> None:
            self._heartbeat_count = 9
            super().__init__(events, close_error=RuntimeError("close failed"))

        @property
        def heartbeat_count(self) -> int:
            events.append("sample")
            return self._heartbeat_count

        @heartbeat_count.setter
        def heartbeat_count(self, value: int) -> None:
            self._heartbeat_count = value

    with pytest.raises(RuntimeError, match="close failed"):
        _run_session(
            heartbeat=SampledHeartbeat(),
            stop_event=_StopEvent(events),
            execute_package=lambda _check, _update: events.append("execute"),
        )

    assert events == ["start", "is_set", "execute", "sample", "close"]
