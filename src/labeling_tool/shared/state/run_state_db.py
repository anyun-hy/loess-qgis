"""PostgreSQL production state store for large, resumable inference runs."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Mapping, Sequence

from labeling_tool.shared.state.acceptance_read_repository import (
    AcceptanceReadRepository,
)
from labeling_tool.shared.state.artifact_repository import ArtifactRepository
from labeling_tool.shared.state.job_repository import JobRepository
from labeling_tool.shared.state.monitor_history_repository import (
    MonitorHistoryRepository,
)
from labeling_tool.shared.state.monitor_read_repository import MonitorReadRepository
from labeling_tool.shared.state.package_reset_repository import (
    PackageResetRepository,
)
from labeling_tool.shared.state.postgres_state import (
    initialize_postgres,
    is_postgres_location,
    postgres_health,
)
from labeling_tool.shared.state.run_archive_repository import RunArchiveRepository
from labeling_tool.shared.state.run_control_graph_repository import (
    RunControlGraphRepository,
)
from labeling_tool.shared.state.run_execution_ownership import (
    RunPublicationLock,
    publication_lock_path,
)
from labeling_tool.shared.state.run_state_session import (
    RunStateError,
    RunStateSession,
    production_state_database,
    production_state_schema,
)
from labeling_tool.shared.state.run_stream_repository import (
    SCHEMA_VERSION,
    RunStreamRepository,
)
from labeling_tool.shared.state.state_values import (
    utc_now as _now,
)
from labeling_tool.shared.state.unit_report_repository import UnitReportRepository

__all__ = [
    "RunStateDB",
    "RunStateError",
    "production_state_database",
    "production_state_schema",
    "run_state_from_spec",
]


def run_state_from_spec(spec: Mapping[str, Any]) -> "RunStateDB":
    """Open the PostgreSQL state store frozen into a Run Spec."""

    backend = str(spec.get("state_backend") or "").strip().lower()
    location = str(spec.get("state_db") or "").strip()
    if backend != "postgresql" or not is_postgres_location(location):
        raise RunStateError(
            "Run Spec must declare state_backend=postgresql and a PostgreSQL DSN; "
            "legacy SQLite Run state is no longer supported"
        )
    schema = str(spec.get("state_schema") or "").strip() or None
    return RunStateDB(location, postgres_schema=schema)


class RunStateDB:
    """PostgreSQL-backed Run-state API."""

    def __init__(
        self,
        dsn: str,
        *,
        postgres_schema: str | None = None,
    ) -> None:
        self.session = RunStateSession(dsn, schema=postgres_schema)
        self.acceptance_read = AcceptanceReadRepository(self.session)
        self.artifacts = ArtifactRepository(self.session)
        self.jobs = JobRepository(self.session)
        self.monitor_history = MonitorHistoryRepository(self.session)
        self.monitor_read = MonitorReadRepository(self.session, self.monitor_history)
        self.package_resets = PackageResetRepository(self.session)
        self.run_archive = RunArchiveRepository(self.session)
        self.control_graph = RunControlGraphRepository(self.session)
        self.run_streams = RunStreamRepository(self.session)
        self.unit_reports = UnitReportRepository(self.session, self.artifacts)

    @classmethod
    def _from_session(cls, session: RunStateSession) -> "RunStateDB":
        scoped = cls.__new__(cls)
        scoped.session = session
        scoped.acceptance_read = AcceptanceReadRepository(session)
        scoped.artifacts = ArtifactRepository(session)
        scoped.jobs = JobRepository(session)
        scoped.monitor_history = MonitorHistoryRepository(session)
        scoped.monitor_read = MonitorReadRepository(session, scoped.monitor_history)
        scoped.package_resets = PackageResetRepository(session)
        scoped.run_archive = RunArchiveRepository(session)
        scoped.control_graph = RunControlGraphRepository(session)
        scoped.run_streams = RunStreamRepository(session)
        scoped.unit_reports = UnitReportRepository(session, scoped.artifacts)
        return scoped

    @contextlib.contextmanager
    def unit_attempt_commit(
        self, job_id: int, lease_token: str
    ) -> Iterator["RunStateDB"]:
        """Fence one unit publication and reuse its transaction on a private facade.

        Files must already live in an immutable attempt directory. An exception
        rolls back every metadata write; no other thread shares this facade.
        """
        with self.session.transaction() as connection:
            job = connection.execute(
                """SELECT * FROM jobs WHERE job_id=%s AND job_type='unit_fit'
                   AND status='running' AND lease_token=%s
                   AND lease_expires>=%s FOR UPDATE""",
                (int(job_id), str(lease_token), time.time()),
            ).fetchone()
            if job is None:
                raise RunStateError("unit attempt no longer owns its lease")
            scoped_session = self.session.scoped(
                connection,
                (
                    str(job["run_id"]),
                    str(job["stream_id"]),
                    str(job["unit_id"]),
                ),
            )
            try:
                yield self._from_session(scoped_session)
            finally:
                scoped_session.invalidate()

    @contextlib.contextmanager
    def fragmentation_v33_attempt_commit(
        self, job_id: int, lease_token: str
    ) -> Iterator["RunStateDB"]:
        """Fence one V3.3 attempt and atomically publish its staged result.

        The caller must write files beneath its immutable attempt directory
        before entering this scope.  Artifact publication and Job completion
        then use only the yielded facade, so an expired or superseded lease
        cannot expose a mixed pair of staged Artifacts.
        """

        with self.session.transaction() as connection:
            job = connection.execute(
                """SELECT * FROM jobs
                   WHERE job_id=%s AND job_type='fragmentation_v33'
                     AND status='running' AND lease_token=%s
                     AND lease_expires IS NOT NULL AND lease_expires>=%s
                   FOR UPDATE""",
                (int(job_id), str(lease_token), time.time()),
            ).fetchone()
            if job is None:
                raise RunStateError(
                    "fragmentation V3.3 attempt no longer owns its lease"
                )
            if self.session.is_scoped_transaction:
                if self.session.unit_identity not in {
                    None,
                    (
                        str(job["run_id"]),
                        str(job["stream_id"]),
                        str(job["unit_id"]),
                    ),
                }:
                    raise RunStateError(
                        "fragmentation V3.3 attempt does not match scoped identity"
                    )
                yield self
                return
            scoped_session = self.session.scoped(
                connection,
                (
                    str(job["run_id"]),
                    str(job["stream_id"]),
                    str(job["unit_id"]),
                ),
            )
            try:
                yield self._from_session(scoped_session)
            finally:
                scoped_session.invalidate()

    @contextlib.contextmanager
    def owner_publication(
        self,
        run_id: str,
        run_dir: str | Path,
    ) -> Iterator["RunStateDB"]:
        """Fence canonical files and their control-plane publication together.

        Callers must use only the yielded facade for database operations inside
        this context.  The filesystem lock is always acquired before the
        owner-fenced transaction, matching the takeover lock order.
        """

        identity = self.session.execution_owner
        if identity is None:
            raise RunStateError("canonical Run publication requires an execution owner")
        if str(run_id) != identity.run_id:
            raise RunStateError("canonical publication Run does not match its owner")
        lock_path = publication_lock_path(run_dir)
        if str(lock_path) != identity.publication_lock_path:
            raise RunStateError(
                "canonical publication directory does not match its frozen owner"
            )
        with RunPublicationLock(lock_path):
            with self.session.transaction() as connection:
                scoped_session = self.session.publication_scoped(connection)
                try:
                    yield self._from_session(scoped_session)
                finally:
                    scoped_session.invalidate()

    def initialize(self) -> None:
        if self.session.is_scoped_transaction:
            raise RunStateError("scoped Run-state facades cannot initialize schemas")
        initialize_postgres(
            self.session.location,
            schema=self.session.schema,
            schema_version=SCHEMA_VERSION,
            now=_now(),
        )

    def pragmas(self) -> dict[str, Any]:
        if self.session.is_scoped_transaction:
            raise RunStateError("scoped Run-state facades cannot run health checks")
        report = postgres_health(
            self.session.location,
            schema=self.session.schema,
            schema_version=SCHEMA_VERSION,
        )
        return {
            **report,
            "journal_mode": "postgresql-mvcc",
            "foreign_keys": 1,
            "user_version": SCHEMA_VERSION,
        }

    def complete_fragmentation_v33_finalize(
        self,
        job_id: int,
        lease_token: str,
        outputs: Sequence[Mapping[str, Any]],
        *,
        report_path: str | Path,
        report_byte_count: int,
        report_sha256: str,
    ) -> bool:
        """Atomically publish every authoritative Core and cross the barrier.

        The caller writes and verifies files before entering this transaction.
        Until this transaction commits, no ``core_mask`` or authoritative
        audit row is visible and the finalize job remains running.  A crash can
        therefore leave reusable files on disk, but never a partly published
        authority set in the control plane.
        """

        token = str(lease_token)
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in outputs:
            partition_id = str(raw.get("partition_id") or "")
            if not partition_id or partition_id in seen:
                raise ValueError(
                    "V3.3 finalize outputs require unique partition_id values"
                )
            seen.add(partition_id)
            item = {
                "partition_id": partition_id,
                "mask_path": str(Path(raw["mask_path"]).expanduser().resolve()),
                "mask_byte_count": int(raw["mask_byte_count"]),
                "mask_sha256": str(raw["mask_sha256"]).lower(),
                "audit_path": str(Path(raw["audit_path"]).expanduser().resolve()),
                "audit_byte_count": int(raw["audit_byte_count"]),
                "audit_sha256": str(raw["audit_sha256"]).lower(),
            }
            for size_key, sha_key in (
                ("mask_byte_count", "mask_sha256"),
                ("audit_byte_count", "audit_sha256"),
            ):
                if item[size_key] < 0:
                    raise ValueError("artifact byte_count cannot be negative")
                digest = item[sha_key]
                if len(digest) != 64:
                    raise ValueError(
                        "artifact sha256 must contain 64 hexadecimal characters"
                    )
                try:
                    int(digest, 16)
                except ValueError as error:
                    raise ValueError(
                        "artifact sha256 must contain 64 hexadecimal characters"
                    ) from error
            normalized.append(item)
        report_digest = str(report_sha256).lower()
        if int(report_byte_count) < 0:
            raise ValueError("artifact byte_count cannot be negative")
        if len(report_digest) != 64:
            raise ValueError("artifact sha256 must contain 64 hexadecimal characters")
        try:
            int(report_digest, 16)
        except ValueError as error:
            raise ValueError(
                "artifact sha256 must contain 64 hexadecimal characters"
            ) from error

        now = _now()
        fence_time = time.time()
        with self.session.transaction() as connection:
            job = connection.execute(
                """SELECT * FROM jobs WHERE job_id=%s
                   AND job_type='fragmentation_v33' AND status='running'
                   AND lease_token=%s AND lease_expires IS NOT NULL
                   AND lease_expires>=%s""",
                (int(job_id), token, fence_time),
            ).fetchone()
            if job is None:
                return False
            unit = connection.execute(
                """SELECT unit_type FROM spatial_units
                   WHERE run_id=%s AND unit_id=%s""",
                (str(job["run_id"]), str(job["unit_id"])),
            ).fetchone()
            if unit is None or str(unit["unit_type"]) != "FragmentationV33Finalize":
                raise RunStateError("atomic V3.3 finalize requires the finalize unit")
            expected_rows = connection.execute(
                """SELECT partition_id FROM unit_dependencies
                   WHERE run_id=%s AND unit_id=%s ORDER BY partition_id""",
                (str(job["run_id"]), str(job["unit_id"])),
            ).fetchall()
            expected = {str(row["partition_id"]) for row in expected_rows}
            if seen != expected:
                missing = sorted(expected - seen)
                extra = sorted(seen - expected)
                raise RunStateError(
                    f"V3.3 finalize output set mismatch: missing={missing}, extra={extra}"
                )
            unfinished = int(
                connection.execute(
                    """SELECT COUNT(*) FROM jobs owner_job
                       JOIN spatial_units owner_unit
                         ON owner_unit.run_id=owner_job.run_id
                        AND owner_unit.unit_id=owner_job.unit_id
                       WHERE owner_job.run_id=%s AND owner_job.stream_id=%s
                         AND owner_job.job_type='fragmentation_v33'
                         AND owner_unit.unit_type='FragmentationV33Partition'
                         AND owner_job.status!='ready'""",
                    (str(job["run_id"]), str(job["stream_id"])),
                ).fetchone()[0]
            )
            if unfinished:
                raise RunStateError(
                    "V3.3 finalize cannot publish before all owner jobs are ready"
                )

            def publish(
                unit_id: str,
                kind: str,
                path: str,
                byte_count: int,
                digest: str,
            ) -> None:
                other = connection.execute(
                    """SELECT path FROM artifacts
                       WHERE run_id=%s AND stream_id=%s AND unit_id=%s AND kind=%s
                         AND path!=%s AND status='ready'""",
                    (
                        str(job["run_id"]),
                        str(job["stream_id"]),
                        str(unit_id),
                        str(kind),
                        str(path),
                    ),
                ).fetchone()
                if other is not None:
                    raise RunStateError(
                        f"V3.3 {kind} already has another ready path: {other['path']}"
                    )
                connection.execute(
                    """INSERT INTO artifacts
                       (run_id, stream_id, unit_id, kind, path, byte_count,
                        sha256, status, created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, 'ready', %s, %s)
                       ON CONFLICT(run_id, stream_id, unit_id, kind, path) DO NOTHING""",
                    (
                        str(job["run_id"]),
                        str(job["stream_id"]),
                        str(unit_id),
                        str(kind),
                        str(path),
                        int(byte_count),
                        str(digest),
                        now,
                        now,
                    ),
                )
                artifact = connection.execute(
                    """SELECT status, byte_count, sha256 FROM artifacts
                       WHERE run_id=%s AND stream_id=%s AND unit_id=%s
                         AND kind=%s AND path=%s""",
                    (
                        str(job["run_id"]),
                        str(job["stream_id"]),
                        str(unit_id),
                        str(kind),
                        str(path),
                    ),
                ).fetchone()
                if artifact is None:
                    raise RunStateError(
                        "V3.3 atomic finalize did not create an artifact"
                    )
                if str(artifact["status"]) == "ready":
                    if int(artifact["byte_count"]) != int(byte_count) or str(
                        artifact["sha256"]
                    ) != str(digest):
                        raise RunStateError(
                            f"ready V3.3 {kind} changed on disk: {path}"
                        )
                    return
                if str(artifact["status"]) not in {"writing", "failed"}:
                    raise RunStateError(f"V3.3 {kind} cannot be published")
                changed = connection.execute(
                    """UPDATE artifacts SET status='ready', byte_count=%s,
                       sha256=%s, updated_at=%s WHERE run_id=%s AND stream_id=%s
                       AND unit_id=%s AND kind=%s AND path=%s
                       AND status IN ('writing','failed')""",
                    (
                        int(byte_count),
                        str(digest),
                        now,
                        str(job["run_id"]),
                        str(job["stream_id"]),
                        str(unit_id),
                        str(kind),
                        str(path),
                    ),
                ).rowcount
                if changed != 1:
                    raise RunStateError(f"cannot publish V3.3 {kind}")

            for item in normalized:
                publish(
                    item["partition_id"],
                    "core_mask",
                    item["mask_path"],
                    item["mask_byte_count"],
                    item["mask_sha256"],
                )
                publish(
                    item["partition_id"],
                    "fragmentation_v33_audit",
                    item["audit_path"],
                    item["audit_byte_count"],
                    item["audit_sha256"],
                )
            publish(
                str(job["unit_id"]),
                "fragmentation_v33_report",
                str(Path(report_path).expanduser().resolve()),
                int(report_byte_count),
                report_digest,
            )
            changed = connection.execute(
                """UPDATE jobs SET status='ready', error='', worker_id='',
                   progress_current=%s, progress_total=%s, lease_token='',
                   lease_expires=NULL, heartbeat_at=%s, updated_at=%s
                   WHERE job_id=%s AND job_type='fragmentation_v33'
                     AND status='running' AND lease_token=%s
                     AND lease_expires IS NOT NULL AND lease_expires>=%s""",
                (
                    len(expected),
                    len(expected),
                    now,
                    now,
                    int(job_id),
                    token,
                    fence_time,
                ),
            ).rowcount
            if changed != 1:
                return False
            connection.execute(
                "DELETE FROM artifact_dependencies WHERE job_id=%s",
                (int(job_id),),
            )
            return True
