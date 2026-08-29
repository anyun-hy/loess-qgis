"""Short cross-process disk reservations for concurrent vector writers."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
from typing import Iterator
import uuid

from storage_guard import StorageGuard


def _pid_is_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_state(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": 1, "reservations": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not isinstance(
        value.get("reservations"), dict
    ):
        raise RuntimeError(f"invalid vector storage reservation state: {path}")
    return value


def _write_state(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as destination:
            json.dump(value, destination, sort_keys=True, separators=(",", ":"))
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def concurrent_storage_reservation(
    storage_guard: StorageGuard | None,
    lock_path: Path | None,
    operation: str,
    write_bytes: int,
) -> Iterator[None]:
    """Reserve under a short lock, then let independent files write together."""

    if storage_guard is None:
        yield
        return
    shared_lock = Path(lock_path) if lock_path is not None else (
        storage_guard.root / "tmp" / ".vector-storage-reserve.lock"
    )
    shared_lock.parent.mkdir(parents=True, exist_ok=True)
    state_path = shared_lock.with_name(f"{shared_lock.name}.json")
    token = uuid.uuid4().hex
    reservation = None

    try:
        with shared_lock.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                state = _read_state(state_path)
                active = {
                    key: item
                    for key, item in state["reservations"].items()
                    if _pid_is_alive(int(item.get("pid", -1)))
                }
                other_pending = sum(
                    max(0, int(item.get("write_bytes", 0)))
                    for item in active.values()
                )
                reservation = storage_guard.reserve(
                    str(operation),
                    write_bytes=max(0, int(write_bytes)),
                    additional_reserve_bytes=other_pending,
                    managed_growth_bytes=0,
                )
                active[token] = {
                    "pid": os.getpid(),
                    "operation": str(operation),
                    "write_bytes": max(0, int(write_bytes)),
                }
                state["reservations"] = active
                _write_state(state_path, state)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except Exception:
        if reservation is not None:
            reservation.release()
        raise

    try:
        yield
    finally:
        try:
            with shared_lock.open("a+b") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    state = _read_state(state_path)
                    state["reservations"].pop(token, None)
                    _write_state(state_path, state)
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            if reservation is not None:
                reservation.release()
