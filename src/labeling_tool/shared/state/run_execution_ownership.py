"""Cross-process ownership and fencing for one executing Run."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

RUN_OWNER_ENV = "LOESS_RUN_EXECUTION_OWNER"


class RunOwnershipError(RuntimeError):
    """Base error for an execution that cannot safely act for a Run."""


class RunOwnershipConflictError(RunOwnershipError):
    """Another process owns the selected Run or is finishing publication."""


class RunOwnershipLostError(RunOwnershipError):
    """The execution identity no longer owns the Run's PostgreSQL lock."""


def _positive_lock_key(value: str) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def execution_lock_keys(schema: str, run_id: str) -> tuple[int, int]:
    """Return the schema-scoped two-int PostgreSQL advisory-lock key."""

    namespace = _positive_lock_key(f"loess-qgis:run-owner:{str(schema)}")
    identifier = _positive_lock_key(f"run:{str(run_id)}")
    return namespace, identifier


@dataclass(frozen=True)
class RunExecutionIdentity:
    """Database-verifiable identity of one advisory-lock holder."""

    run_id: str
    execution_id: str
    schema: str
    worker_id: str
    lock_backend_pid: int
    lock_key_1: int
    lock_key_2: int
    publication_lock_path: str

    def as_mapping(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "execution_id": self.execution_id,
            "schema": self.schema,
            "worker_id": self.worker_id,
            "lock_backend_pid": int(self.lock_backend_pid),
            "lock_key_1": int(self.lock_key_1),
            "lock_key_2": int(self.lock_key_2),
            "publication_lock_path": self.publication_lock_path,
        }

    def environment_value(self) -> str:
        return json.dumps(self.as_mapping(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RunExecutionIdentity":
        identity = cls(
            run_id=str(value["run_id"]),
            execution_id=str(value["execution_id"]),
            schema=str(value["schema"]),
            worker_id=str(value.get("worker_id") or ""),
            lock_backend_pid=int(value["lock_backend_pid"]),
            lock_key_1=int(value["lock_key_1"]),
            lock_key_2=int(value["lock_key_2"]),
            publication_lock_path=str(value["publication_lock_path"]),
        )
        if not identity.run_id or not identity.execution_id or not identity.schema:
            raise ValueError("Run execution owner identity is incomplete")
        if identity.lock_backend_pid <= 0:
            raise ValueError("Run execution owner backend PID is invalid")
        expected = execution_lock_keys(identity.schema, identity.run_id)
        if (identity.lock_key_1, identity.lock_key_2) != expected:
            raise ValueError("Run execution owner lock key does not match Run identity")
        lock_path = Path(identity.publication_lock_path)
        if not lock_path.is_absolute() or lock_path.name != "run_publication.lock":
            raise ValueError("Run publication lock path is invalid")
        return identity

    def assert_current(self, connection: Any) -> None:
        """Fence a transaction against both execution state and the live lock."""

        row = connection.execute(
            """SELECT status, metadata_json FROM monitor_executions
               WHERE run_id=%s AND execution_id=%s FOR SHARE""",
            (self.run_id, self.execution_id),
        ).fetchone()
        if row is None or str(row["status"]) != "running":
            raise RunOwnershipLostError(
                f"Run {self.run_id} execution {self.execution_id} is no longer current"
            )
        try:
            metadata = json.loads(str(row["metadata_json"] or "{}"))
            recorded = RunExecutionIdentity.from_mapping(metadata["run_owner"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RunOwnershipLostError(
                f"Run {self.run_id} execution ownership metadata is invalid"
            ) from error
        if recorded != self:
            raise RunOwnershipLostError(
                f"Run {self.run_id} execution ownership identity changed"
            )
        held = connection.execute(
            """SELECT 1 FROM pg_locks
               WHERE locktype='advisory' AND granted
                 AND database=(SELECT oid FROM pg_database
                               WHERE datname=current_database())
                 AND pid=%s AND classid::bigint=%s AND objid::bigint=%s
                 AND objsubid=2 AND mode='ExclusiveLock'""",
            (
                int(self.lock_backend_pid),
                int(self.lock_key_1),
                int(self.lock_key_2),
            ),
        ).fetchone()
        if held is None:
            raise RunOwnershipLostError(
                f"Run {self.run_id} execution ownership connection was lost"
            )


def execution_identity_from_environment(
    *, schema: str, execution_id: str
) -> RunExecutionIdentity | None:
    """Read the explicit owner binding inherited by a worker subprocess."""

    raw = str(os.environ.get(RUN_OWNER_ENV) or "").strip()
    if not raw:
        return None
    try:
        value = json.loads(raw)
        if not isinstance(value, Mapping):
            raise ValueError("owner identity must be an object")
        identity = RunExecutionIdentity.from_mapping(value)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RunOwnershipLostError(
            "Run execution owner environment is invalid"
        ) from error
    if identity.schema != str(schema) or identity.execution_id != str(execution_id):
        raise RunOwnershipLostError(
            "Run execution owner environment does not match this state session"
        )
    return identity


class RunExecutionOwnership:
    """Hold one Run's dedicated advisory lock until execution publication ends."""

    def __init__(self, database: Any, connection: Any, identity: RunExecutionIdentity):
        self._database = database
        self._connection = connection
        self.identity = identity
        self._closed = False

    @classmethod
    def acquire(
        cls,
        database: Any,
        run_id: str,
        *,
        run_dir: str | Path,
        worker_id: str,
        trigger_type: str,
    ) -> "RunExecutionOwnership":
        schema = str(database.session.schema)
        owner_keys = execution_lock_keys(schema, run_id)
        publication_path = publication_lock_path(run_dir)
        connection = database.session.connect(autocommit=True)
        execution_id = ""
        try:
            acquired = bool(
                connection.execute(
                    "SELECT pg_try_advisory_lock(%s,%s)", owner_keys
                ).fetchone()[0]
            )
            if not acquired:
                raise RunOwnershipConflictError(
                    f"Run {run_id} is already owned by another execution"
                )
            backend_pid = int(
                connection.execute("SELECT pg_backend_pid()").fetchone()[0]
            )
            with RunPublicationLock(publication_path, blocking=False):
                execution_id = database.monitor_history.begin_execution(
                    str(run_id),
                    str(trigger_type),
                    metadata={
                        "worker_id": str(worker_id),
                        "run_owner": {
                            "run_id": str(run_id),
                            "execution_id": "pending",
                            "schema": schema,
                            "worker_id": str(worker_id),
                            "lock_backend_pid": backend_pid,
                            "lock_key_1": owner_keys[0],
                            "lock_key_2": owner_keys[1],
                            "publication_lock_path": str(publication_path),
                        },
                    },
                )
                identity = RunExecutionIdentity(
                    run_id=str(run_id),
                    execution_id=str(execution_id),
                    schema=schema,
                    worker_id=str(worker_id),
                    lock_backend_pid=backend_pid,
                    lock_key_1=owner_keys[0],
                    lock_key_2=owner_keys[1],
                    publication_lock_path=str(publication_path),
                )
                # begin_execution creates the ID, so replace the temporary
                # metadata before releasing the cross-filesystem handover lock.
                updated = connection.execute(
                    """UPDATE monitor_executions
                       SET metadata_json=%s
                       WHERE run_id=%s AND execution_id=%s AND status='running'""",
                    (
                        json.dumps(
                            {
                                "worker_id": str(worker_id),
                                "run_owner": identity.as_mapping(),
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        str(run_id),
                        str(execution_id),
                    ),
                ).rowcount
                if updated != 1:
                    raise RunOwnershipLostError(
                        f"Run {run_id} execution changed while ownership was starting"
                    )
                database.session.bind_execution_owner(identity)
            return cls(database, connection, identity)
        except BlockingIOError as error:
            try:
                connection.execute("SELECT pg_advisory_unlock(%s,%s)", owner_keys)
            except Exception:
                pass
            connection.close()
            raise RunOwnershipConflictError(
                f"Run {run_id} is still completing a prior file publication"
            ) from error
        except Exception as error:
            if execution_id:
                try:
                    database.monitor_history.finish_execution(
                        str(run_id),
                        str(execution_id),
                        status="failed",
                        message=f"Run ownership acquisition failed: {error}",
                        recording_complete=False,
                    )
                except Exception:
                    pass
            try:
                connection.execute("SELECT pg_advisory_unlock(%s,%s)", owner_keys)
            except Exception:
                pass
            connection.close()
            raise

    def assert_current(self) -> None:
        if self._closed:
            raise RunOwnershipLostError(
                f"Run {self.identity.run_id} execution ownership is closed"
            )
        try:
            backend_pid = int(
                self._connection.execute("SELECT pg_backend_pid()").fetchone()[0]
            )
            if backend_pid != self.identity.lock_backend_pid:
                raise RunOwnershipLostError(
                    f"Run {self.identity.run_id} lock backend identity changed"
                )
            with self._database.session.connection() as connection:
                self.identity.assert_current(connection)
        except RunOwnershipLostError:
            raise
        except Exception as error:
            raise RunOwnershipLostError(
                f"Run {self.identity.run_id} ownership connection is unavailable: {error}"
            ) from error

    @contextlib.contextmanager
    def publication_barrier(self):
        """Hold the Run file barrier across multi-transaction maintenance."""

        if self._closed:
            raise RunOwnershipLostError(
                f"Run {self.identity.run_id} execution ownership is closed"
            )
        with RunPublicationLock(self.identity.publication_lock_path):
            # File lock first, then database owner verification.  Every write
            # transaction inside the scope applies the same fence again.
            self.assert_current()
            yield

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Keep the database session bound to this retired identity.  Late
        # callbacks must fail the ownership fence instead of silently falling
        # back to an unowned write-capable session.  A later execution opens a
        # fresh RunStateDB/session before acquiring its own owner.
        try:
            self._connection.execute(
                "SELECT pg_advisory_unlock(%s,%s)",
                (self.identity.lock_key_1, self.identity.lock_key_2),
            )
        except Exception:
            pass
        finally:
            self._connection.close()


def publication_lock_path(run_dir: str | Path) -> Path:
    root = Path(run_dir).expanduser()
    if root.is_symlink():
        raise RunOwnershipError(f"Run directory cannot be a symlink: {root}")
    root = root.resolve()
    if not root.is_dir():
        raise RunOwnershipError(f"Run directory is missing: {root}")
    lock_parent = root / "tmp"
    if lock_parent.is_symlink():
        raise RunOwnershipError(
            f"Run publication lock directory cannot be a symlink: {lock_parent}"
        )
    return lock_parent / "run_publication.lock"


class RunPublicationLock:
    """Filesystem handover barrier for canonical Run output publication.

    The stable lock inode and its ``tmp`` parent are durable Run coordination
    state.  Releasing this lock closes the handle but never unlinks either path.
    """

    def __init__(self, path: str | Path, *, blocking: bool = True) -> None:
        self.path = Path(path)
        self.blocking = bool(blocking)
        self._handle = None

    def __enter__(self) -> "RunPublicationLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        self._handle = os.fdopen(descriptor, "a+b")
        operation = fcntl.LOCK_EX
        if not self.blocking:
            operation |= fcntl.LOCK_NB
        try:
            fcntl.flock(self._handle.fileno(), operation)
        except Exception:
            self._handle.close()
            self._handle = None
            raise
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None
