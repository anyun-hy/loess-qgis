"""Failure-only V3.3 diagnostics, independent of transaction ownership.

Phase clocks measure elapsed application time, including SQL execution, lock
waits and any rollback before the worker catches an exception. They do not
measure pure PostgreSQL wait time. No SQL, credentials or lease tokens are kept.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any

_ACTIVE: ContextVar[PublicationDiagnostics | None] = ContextVar(
    "v33_publication_diagnostics", default=None
)


def _sqlstate(error: BaseException) -> str | None:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen and len(seen) < 8:
        seen.add(id(current))
        try:
            code = getattr(current, "pgcode", None) or getattr(
                current, "sqlstate", None
            )
        except Exception:
            return None
        if (
            isinstance(code, str)
            and len(code) == 5
            and code.isascii()
            and code.isalnum()
        ):
            return code.upper()
        current = current.__cause__ or current.__context__
    return None


@dataclass(frozen=True)
class PublicationDiagnostics:
    identity: dict[str, Any]
    started: float
    phase_started: float
    phase_ms: dict[str, float]
    group: object
    phase: str = "prepare"

    def advance(self, phase: str) -> PublicationDiagnostics:
        now = time.perf_counter()
        phases = dict(self.phase_ms)
        phases[self.phase] = (
            phases.get(self.phase, 0.0) + (now - self.phase_started) * 1000
        )
        return replace(self, phase=phase, phase_started=now, phase_ms=phases)

    def capture(self, error: BaseException) -> None:
        # Capture before failure bookkeeping, and preserve that snapshot while
        # the same exception unwinds through the outer diagnostic context.
        if getattr(error, "_v33_diagnostic_trace", None) is self.group:
            return
        now = time.perf_counter()
        phases = dict(self.phase_ms)
        phases[self.phase] = (
            phases.get(self.phase, 0.0) + (now - self.phase_started) * 1000
        )
        error._v33_diagnostic_trace = self.group
        lock_completed = "input_lock" in self.phase_ms
        error._v33_diagnostics = {
            "event": "v33_job_failure",
            **self.identity,
            "phase": self.phase,
            "exception_type": type(error).__name__,
            "sqlstate": _sqlstate(error),
            "elapsed_ms": round((now - self.started) * 1000, 3),
            "phase_to_catch_ms": {
                key: round(value, 3) for key, value in phases.items()
            },
            "input_lock_completed": lock_completed,
            "input_lock_elapsed_ms": (
                round(self.phase_ms["input_lock"], 3) if lock_completed else None
            ),
            "release_input_lock_elapsed_ms": (
                round(self.phase_ms["release_input_lock"], 3)
                if "release_input_lock" in self.phase_ms
                else None
            ),
        }


@contextlib.contextmanager
def v33_job_diagnostics(job: Mapping[str, Any]) -> Iterator[None]:
    try:
        unit = str(job["unit_id"])[:256]
        prefix = "fragmentation_v33_partition:"
        began = time.perf_counter()
        trace = PublicationDiagnostics(
            {
                "run_id": str(job["run_id"])[:256],
                "stream_id": str(job["stream_id"])[:256],
                "job_id": int(job["job_id"]),
                "unit_id": unit,
                "partition_id": unit[len(prefix) :]
                if unit.startswith(prefix)
                else None,
                "attempt": int(job["attempt"]),
            },
            started=began,
            phase_started=began,
            phase_ms={},
            group=object(),
        )
    except Exception:
        trace = None
    token = _ACTIVE.set(trace)
    try:
        yield
    except Exception as error:
        capture_publication_failure(error)
        raise
    finally:
        _ACTIVE.reset(token)


def publication_phase(phase: str, *, job_id: int | None = None) -> None:
    try:
        trace = _ACTIVE.get()
        if trace is not None and (
            job_id is None or int(job_id) == trace.identity["job_id"]
        ):
            _ACTIVE.set(trace.advance(phase))
    except Exception:
        pass  # Diagnostics must never change publication or exception behavior.


def capture_publication_failure(error: BaseException) -> None:
    try:
        trace = _ACTIVE.get()
        if trace is not None:
            trace.capture(error)
    except Exception:
        pass


def publication_failure_payload(error: BaseException) -> dict[str, Any]:
    try:
        diagnostics = getattr(error, "_v33_diagnostics", None)
        if not isinstance(diagnostics, dict):
            raise TypeError("no diagnostic snapshot")
        json.dumps(diagnostics, allow_nan=False)
    except Exception:
        diagnostics = {
            "event": "v33_job_failure",
            "phase": "preflight",
            "exception_type": type(error).__name__,
            "sqlstate": _sqlstate(error),
        }
    return {"status": "failed", "error": str(error), "diagnostics": diagnostics}
