"""Artifact publication, reference, cleanup, and query persistence."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

from labeling_tool.shared.state.run_state_session import RunStateError, RunStateSession
from labeling_tool.shared.state.state_values import row_dict as _row_dict
from labeling_tool.shared.state.state_values import utc_now as _now

__all__ = ["ArtifactRepository"]


class ArtifactRepository:
    """Own the ordinary Artifact lifecycle on one Run-state session."""

    def __init__(self, session: RunStateSession) -> None:
        self._session = session

    def supersede_unit_attempt_artifacts(
        self,
        run_id: str,
        stream_id: str,
        unit_id: str,
    ) -> None:
        """Hide partial older publications, without deleting files or audit rows."""
        identity = self._session.unit_identity
        if identity is None:
            raise RunStateError(
                "unit artifacts can only be superseded inside a fenced commit"
            )
        if identity != (str(run_id), str(stream_id), str(unit_id)):
            raise RunStateError("unit publication identity differs from leased job")
        with self._session.transaction() as connection:
            connection.execute(
                """UPDATE artifacts SET status='superseded', updated_at=%s
                   WHERE run_id=%s AND stream_id=%s AND unit_id=%s
                     AND kind IN ('unit_raw_geoparquet','unit_formal_geoparquet',
                                  'unit_boundary_report','unit_boundary_signatures',
                                  'unit_fitted_edges_geoparquet') AND status='ready'""",
                (_now(), str(run_id), str(stream_id), str(unit_id)),
            )

    def register_artifact(
        self,
        run_id: str,
        kind: str,
        path: str | Path,
        *,
        stream_id: str = "",
        unit_id: str = "",
    ) -> int:
        """Register an artifact before its writer starts the atomic file write."""
        now = _now()
        resolved_path = str(Path(path).expanduser().resolve())
        with self._session.transaction() as connection:
            connection.execute(
                """INSERT INTO artifacts
                   (run_id, stream_id, unit_id, kind, path, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT(run_id, stream_id, unit_id, kind, path) DO NOTHING""",
                (
                    str(run_id),
                    str(stream_id),
                    str(unit_id),
                    str(kind),
                    resolved_path,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                """SELECT artifact_id FROM artifacts
                   WHERE run_id=%s AND stream_id=%s AND unit_id=%s AND kind=%s AND path=%s""",
                (
                    str(run_id),
                    str(stream_id),
                    str(unit_id),
                    str(kind),
                    resolved_path,
                ),
            ).fetchone()
            if row is None:
                raise RunStateError(
                    "artifact registration did not create or find a row"
                )
            return int(row["artifact_id"])

    def get_artifact(self, artifact_id: int) -> dict[str, Any] | None:
        with self._session.connection() as connection:
            return _row_dict(
                connection.execute(
                    "SELECT * FROM artifacts WHERE artifact_id=%s",
                    (int(artifact_id),),
                ).fetchone()
            )

    def mark_artifact_ready(
        self,
        artifact_id: int,
        *,
        byte_count: int,
        sha256: str,
    ) -> bool:
        if int(byte_count) < 0:
            raise ValueError("artifact byte_count cannot be negative")
        if len(str(sha256)) != 64:
            raise ValueError("artifact sha256 must contain 64 hexadecimal characters")
        try:
            int(str(sha256), 16)
        except ValueError as error:
            raise ValueError(
                "artifact sha256 must contain 64 hexadecimal characters"
            ) from error
        with self._session.transaction() as connection:
            return bool(
                connection.execute(
                    """UPDATE artifacts SET status='ready', byte_count=%s, sha256=%s, updated_at=%s
                   WHERE artifact_id=%s AND status IN ('writing','failed')""",
                    (int(byte_count), str(sha256).lower(), _now(), int(artifact_id)),
                ).rowcount
                == 1
            )

    def publish_partition_artifact(
        self,
        run_id: str,
        stream_id: str,
        partition_id: str,
        path: str | Path,
        *,
        byte_count: int,
        sha256: str,
    ) -> int:
        """Publish one Partition probability and link live consumers atomically.

        The scheduler is allowed to delete a ready probability Artifact with a
        zero reference count.  Therefore the ready transition and dependency
        insertion must be committed by the same transaction; exposing ready in
        an earlier transaction creates a cleanup race.

        A cleaned row may be republished only when none of its dependent jobs
        has started.  This supports an immediate Work Package retry after the
        old race without resurrecting inputs already consumed by completed
        geometry jobs.
        """
        size = int(byte_count)
        if size < 0:
            raise ValueError("artifact byte_count cannot be negative")
        digest = str(sha256).lower()
        if len(digest) != 64:
            raise ValueError("artifact sha256 must contain 64 hexadecimal characters")
        try:
            int(digest, 16)
        except ValueError as error:
            raise ValueError(
                "artifact sha256 must contain 64 hexadecimal characters"
            ) from error

        identifier = str(run_id)
        stream = str(stream_id)
        partition = str(partition_id)
        resolved_path = str(Path(path).expanduser().resolve())
        now = _now()
        with self._session.transaction() as connection:
            connection.execute(
                """INSERT INTO artifacts
                   (run_id, stream_id, unit_id, kind, path, created_at, updated_at)
                   VALUES (%s, %s, %s, 'partition_probability', %s, %s, %s)
                   ON CONFLICT(run_id, stream_id, unit_id, kind, path) DO NOTHING""",
                (identifier, stream, partition, resolved_path, now, now),
            )
            artifact = connection.execute(
                """SELECT * FROM artifacts
                   WHERE run_id=%s AND stream_id=%s AND unit_id=%s
                     AND kind='partition_probability' AND path=%s"""
                + " FOR UPDATE",
                (identifier, stream, partition, resolved_path),
            ).fetchone()
            if artifact is None:
                raise RunStateError(
                    "partition Artifact registration did not create or find a row"
                )

            status = str(artifact["status"])
            artifact_id = int(artifact["artifact_id"])
            if status == "ready":
                if (
                    int(artifact["byte_count"]) != size
                    or str(artifact["sha256"]) != digest
                ):
                    raise RunStateError(
                        f"ready Partition Artifact changed on disk: {resolved_path}"
                    )
            elif status == "cleaned":
                started = connection.execute(
                    """SELECT COUNT(*) FROM jobs j
                       JOIN unit_dependencies d
                         ON d.run_id=j.run_id AND d.unit_id=j.unit_id
                       WHERE j.run_id=%s AND j.stream_id=%s
                         AND j.job_type IN (
                           'unit_fit','unit_confidence','fragmentation_v33'
                         ) AND d.partition_id=%s
                         AND j.status NOT IN ('queued','interrupted')""",
                    (identifier, stream, partition),
                ).fetchone()[0]
                if int(artifact["ref_count"]) != 0 or int(started) != 0:
                    raise RunStateError(
                        "cleaned Partition Artifact requires a full Package reset"
                    )
                connection.execute(
                    """UPDATE artifacts SET status='ready', byte_count=%s, sha256=%s,
                       updated_at=%s WHERE artifact_id=%s AND status='cleaned'
                       AND ref_count=0""",
                    (size, digest, now, artifact_id),
                )
            elif status in {"writing", "failed"}:
                changed = connection.execute(
                    """UPDATE artifacts SET status='ready', byte_count=%s, sha256=%s,
                       updated_at=%s WHERE artifact_id=%s
                       AND status IN ('writing','failed')""",
                    (size, digest, now, artifact_id),
                ).rowcount
                if changed != 1:
                    raise RunStateError(
                        f"cannot publish Partition Artifact: {resolved_path}"
                    )
            else:
                raise RunStateError(
                    f"Partition Artifact is unavailable for publish ({status}): "
                    + resolved_path
                )

            # Link only jobs that can still consume this input. Completed or
            # exhausted jobs have already released their dependencies and must
            # not gain a reference that can never be released.
            connection.execute(
                """INSERT INTO artifact_dependencies
                   (job_id, artifact_id, created_at)
                   SELECT j.job_id, %s, %s FROM jobs j
                   JOIN unit_dependencies d
                     ON d.run_id=j.run_id AND d.unit_id=j.unit_id
                   WHERE j.run_id=%s AND j.stream_id=%s
                     AND j.job_type='unit_fit' AND d.partition_id=%s
                     AND NOT EXISTS (
                       SELECT 1 FROM jobs compact
                       WHERE compact.run_id=j.run_id
                         AND compact.stream_id=j.stream_id
                         AND compact.unit_id=j.unit_id
                         AND compact.job_type='unit_confidence'
                     )
                     AND j.status IN ('queued','interrupted','running') ON CONFLICT DO NOTHING""",
                (artifact_id, now, identifier, stream, partition),
            )
            # V3.3 Fusion geometry consumes a compact, lossless confidence
            # surface.  Link the 14-band probability only to the compactor so
            # it can be released before the global publication barrier.
            connection.execute(
                """INSERT INTO artifact_dependencies
                   (job_id, artifact_id, created_at)
                   SELECT j.job_id, %s, %s FROM jobs j
                   JOIN unit_dependencies d
                     ON d.run_id=j.run_id AND d.unit_id=j.unit_id
                   WHERE j.run_id=%s AND j.stream_id=%s
                     AND j.job_type='unit_confidence' AND d.partition_id=%s
                     AND j.status IN ('queued','interrupted','running') ON CONFLICT DO NOTHING""",
                (artifact_id, now, identifier, stream, partition),
            )
            # V3.3 candidate jobs are planned before the first Work Package
            # starts.  Link them in this same ready-publication transaction so
            # the zero-ref cleanup worker can never observe an unreferenced
            # probability between publication and candidate registration.
            connection.execute(
                """INSERT INTO artifact_dependencies
                   (job_id, artifact_id, created_at)
                   SELECT j.job_id, %s, %s FROM jobs j
                   JOIN spatial_units u
                     ON u.run_id=j.run_id AND u.unit_id=j.unit_id
                   JOIN unit_dependencies d
                     ON d.run_id=j.run_id AND d.unit_id=j.unit_id
                   WHERE j.run_id=%s AND j.stream_id=%s
                     AND j.job_type='fragmentation_v33'
                     AND u.unit_type='FragmentationV33Partition'
                     AND d.partition_id=%s
                     AND j.status IN ('queued','interrupted','running') ON CONFLICT DO NOTHING""",
                (artifact_id, now, identifier, stream, partition),
            )
            return artifact_id

    def complete_unit_confidence_job(
        self,
        job_id: int,
        lease_token: str,
        *,
        path: str | Path,
        byte_count: int,
        sha256: str,
    ) -> bool:
        """Publish one compact confidence surface and release probabilities.

        The ready Artifact, its geometry-job dependency, the compaction Job
        completion, and release of every 14-band probability dependency share
        one transaction.  A cleanup worker can therefore observe neither an
        unreferenced ready confidence surface nor an early probability release.
        """

        size = int(byte_count)
        digest = str(sha256).lower()
        if size < 0:
            raise ValueError("unit confidence byte_count cannot be negative")
        if len(digest) != 64:
            raise ValueError(
                "unit confidence sha256 must contain 64 hexadecimal characters"
            )
        try:
            int(digest, 16)
        except ValueError as error:
            raise ValueError(
                "unit confidence sha256 must contain 64 hexadecimal characters"
            ) from error
        resolved_path = str(Path(path).expanduser().resolve())
        now = _now()
        fence_time = time.time()
        with self._session.transaction() as connection:
            job = connection.execute(
                """SELECT * FROM jobs WHERE job_id=%s
                   AND job_type='unit_confidence' AND status='running'
                   AND lease_token=%s AND lease_expires IS NOT NULL
                   AND lease_expires>=%s""",
                (int(job_id), str(lease_token), fence_time),
            ).fetchone()
            if job is None:
                return False
            run_id = str(job["run_id"])
            stream_id = str(job["stream_id"])
            unit_id = str(job["unit_id"])
            connection.execute(
                """INSERT INTO artifacts
                   (run_id, stream_id, unit_id, kind, path, created_at, updated_at)
                   VALUES (%s, %s, %s, 'unit_confidence', %s, %s, %s)
                   ON CONFLICT(run_id, stream_id, unit_id, kind, path) DO NOTHING""",
                (run_id, stream_id, unit_id, resolved_path, now, now),
            )
            artifact = connection.execute(
                """SELECT * FROM artifacts
                   WHERE run_id=%s AND stream_id=%s AND unit_id=%s
                     AND kind='unit_confidence' AND path=%s FOR UPDATE""",
                (run_id, stream_id, unit_id, resolved_path),
            ).fetchone()
            if artifact is None:
                raise RunStateError("unit confidence registration did not create a row")
            artifact_id = int(artifact["artifact_id"])
            status = str(artifact["status"])
            if status == "ready":
                if (
                    int(artifact["byte_count"]) != size
                    or str(artifact["sha256"]) != digest
                ):
                    raise RunStateError(
                        "ready unit confidence changed: " + resolved_path
                    )
            elif status in {"writing", "failed"}:
                changed = connection.execute(
                    """UPDATE artifacts SET status='ready', byte_count=%s,
                       sha256=%s, updated_at=%s WHERE artifact_id=%s
                       AND status IN ('writing','failed')""",
                    (size, digest, now, artifact_id),
                ).rowcount
                if changed != 1:
                    raise RunStateError("cannot publish unit confidence")
            else:
                raise RunStateError(
                    f"unit confidence is unavailable for publish: {status}"
                )
            fit_jobs = connection.execute(
                """SELECT job_id FROM jobs WHERE run_id=%s AND stream_id=%s
                   AND unit_id=%s AND job_type='unit_fit'
                   AND status IN ('queued','interrupted','running')""",
                (run_id, stream_id, unit_id),
            ).fetchall()
            if len(fit_jobs) != 1:
                raise RunStateError(
                    "unit confidence requires exactly one live geometry consumer"
                )
            connection.execute(
                """INSERT INTO artifact_dependencies
                   (job_id, artifact_id, created_at) VALUES (%s, %s, %s)
                   ON CONFLICT DO NOTHING""",
                (int(fit_jobs[0]["job_id"]), artifact_id, now),
            )
            changed = connection.execute(
                """UPDATE jobs SET status='ready', error='', worker_id='',
                   progress_current=1, progress_total=1, lease_token='',
                   lease_expires=NULL, heartbeat_at=%s, updated_at=%s
                   WHERE job_id=%s AND job_type='unit_confidence'
                     AND status='running' AND lease_token=%s
                     AND lease_expires IS NOT NULL AND lease_expires>=%s""",
                (
                    now,
                    now,
                    int(job_id),
                    str(lease_token),
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

    def _publish_fragmentation_v33_input(
        self,
        run_id: str,
        stream_id: str,
        partition_id: str,
        path: str | Path,
        *,
        byte_count: int,
        sha256: str,
        kind: str,
        label: str,
    ) -> int:
        """Publish one frozen V3.3 input and link its candidate atomically."""

        if kind not in {"v3_context_core", "v3_baseline_core"}:
            raise ValueError(f"unsupported V3.3 input kind: {kind}")

        size = int(byte_count)
        if size < 0:
            raise ValueError("artifact byte_count cannot be negative")
        digest = str(sha256).lower()
        if len(digest) != 64:
            raise ValueError("artifact sha256 must contain 64 hexadecimal characters")
        try:
            int(digest, 16)
        except ValueError as error:
            raise ValueError(
                "artifact sha256 must contain 64 hexadecimal characters"
            ) from error

        identifier = str(run_id)
        stream = str(stream_id)
        partition = str(partition_id)
        resolved_path = str(Path(path).expanduser().resolve())
        now = _now()
        with self._session.transaction() as connection:
            connection.execute(
                """INSERT INTO artifacts
                   (run_id, stream_id, unit_id, kind, path, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT(run_id, stream_id, unit_id, kind, path) DO NOTHING""",
                (identifier, stream, partition, kind, resolved_path, now, now),
            )
            artifact = connection.execute(
                """SELECT * FROM artifacts
                   WHERE run_id=%s AND stream_id=%s AND unit_id=%s
                     AND kind=%s AND path=%s"""
                + " FOR UPDATE",
                (identifier, stream, partition, kind, resolved_path),
            ).fetchone()
            if artifact is None:
                raise RunStateError(f"{label} registration did not create a row")
            artifact_id = int(artifact["artifact_id"])
            status = str(artifact["status"])
            if status == "ready":
                if (
                    int(artifact["byte_count"]) != size
                    or str(artifact["sha256"]) != digest
                ):
                    raise RunStateError(
                        f"ready {label} changed on disk: {resolved_path}"
                    )
            elif status == "cleaned":
                started = connection.execute(
                    """SELECT COUNT(*) FROM jobs j
                       JOIN spatial_units u
                         ON u.run_id=j.run_id AND u.unit_id=j.unit_id
                       JOIN unit_dependencies d
                         ON d.run_id=j.run_id AND d.unit_id=j.unit_id
                       WHERE j.run_id=%s AND j.stream_id=%s
                         AND j.job_type='fragmentation_v33'
                         AND u.unit_type='FragmentationV33Partition'
                         AND d.partition_id=%s
                         AND j.status NOT IN ('queued','interrupted')""",
                    (identifier, stream, partition),
                ).fetchone()[0]
                if int(artifact["ref_count"]) != 0 or int(started) != 0:
                    raise RunStateError(
                        f"cleaned {label} requires a full candidate reset"
                    )
                changed = connection.execute(
                    """UPDATE artifacts SET status='ready', byte_count=%s, sha256=%s,
                       updated_at=%s WHERE artifact_id=%s AND status='cleaned'
                       AND ref_count=0""",
                    (size, digest, now, artifact_id),
                ).rowcount
                if changed != 1:
                    raise RunStateError(f"cannot republish {label}: {resolved_path}")
            elif status in {"writing", "failed"}:
                changed = connection.execute(
                    """UPDATE artifacts SET status='ready', byte_count=%s, sha256=%s,
                       updated_at=%s WHERE artifact_id=%s
                       AND status IN ('writing','failed')""",
                    (size, digest, now, artifact_id),
                ).rowcount
                if changed != 1:
                    raise RunStateError(f"cannot publish {label}: {resolved_path}")
            else:
                raise RunStateError(
                    f"{label} is unavailable for publish ({status}): " + resolved_path
                )
            connection.execute(
                """INSERT INTO artifact_dependencies
                   (job_id, artifact_id, created_at)
                   SELECT j.job_id, %s, %s FROM jobs j
                   JOIN spatial_units u
                     ON u.run_id=j.run_id AND u.unit_id=j.unit_id
                   JOIN unit_dependencies d
                     ON d.run_id=j.run_id AND d.unit_id=j.unit_id
                   WHERE j.run_id=%s AND j.stream_id=%s
                     AND j.job_type='fragmentation_v33'
                     AND u.unit_type='FragmentationV33Partition'
                     AND d.partition_id=%s
                     AND j.status IN ('queued','interrupted','running') ON CONFLICT DO NOTHING""",
                (artifact_id, now, identifier, stream, partition),
            )
            return artifact_id

    def publish_fragmentation_v33_context(
        self,
        run_id: str,
        stream_id: str,
        partition_id: str,
        path: str | Path,
        *,
        byte_count: int,
        sha256: str,
    ) -> int:
        """Publish one V3 owner-Core context and link V3.3 atomically."""

        return self._publish_fragmentation_v33_input(
            run_id,
            stream_id,
            partition_id,
            path,
            byte_count=byte_count,
            sha256=sha256,
            kind="v3_context_core",
            label="V3 context",
        )

    def publish_fragmentation_v33_baseline_core(
        self,
        run_id: str,
        stream_id: str,
        partition_id: str,
        path: str | Path,
        *,
        byte_count: int,
        sha256: str,
    ) -> int:
        """Publish one immutable V3 baseline Core and link V3.3 atomically."""

        return self._publish_fragmentation_v33_input(
            run_id,
            stream_id,
            partition_id,
            path,
            byte_count=byte_count,
            sha256=sha256,
            kind="v3_baseline_core",
            label="V3 baseline Core",
        )

    def publish_fragmentation_v33_output_pair(
        self,
        run_id: str,
        stream_id: str,
        partition_id: str,
        *,
        mask_path: str | Path,
        mask_byte_count: int,
        mask_sha256: str,
        audit_path: str | Path,
        audit_byte_count: int,
        audit_sha256: str,
        production: bool | None,
    ) -> tuple[int, int]:
        """Publish one V3.3 mask/audit pair atomically.

        Files are written before this transaction. A crash before commit leaves
        no ready Artifact; a crash after commit leaves both, so resume never
        observes a half-published authority pair.
        """

        identifier = str(run_id)
        stream = str(stream_id)
        partition = str(partition_id)
        kinds = (
            ("v33_staged_mask", "v33_staged_audit")
            if production is None
            else (
                ("core_mask", "fragmentation_v33_audit")
                if production
                else ("v33_candidate_mask", "v33_candidate_audit")
            )
        )
        records = (
            (
                kinds[0],
                str(Path(mask_path).expanduser().resolve()),
                int(mask_byte_count),
                str(mask_sha256).lower(),
            ),
            (
                kinds[1],
                str(Path(audit_path).expanduser().resolve()),
                int(audit_byte_count),
                str(audit_sha256).lower(),
            ),
        )
        for _kind, _path, size, digest in records:
            if size < 0:
                raise ValueError("artifact byte_count cannot be negative")
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
        now = _now()
        artifact_ids: list[int] = []
        with self._session.transaction() as connection:
            for kind, resolved_path, size, digest in records:
                other = connection.execute(
                    """SELECT path FROM artifacts
                       WHERE run_id=%s AND stream_id=%s AND unit_id=%s AND kind=%s
                         AND path!=%s AND status='ready'""",
                    (identifier, stream, partition, kind, resolved_path),
                ).fetchone()
                if other is not None:
                    raise RunStateError(
                        f"V3.3 {kind} already has another ready path: {other['path']}"
                    )
                connection.execute(
                    """INSERT INTO artifacts
                       (run_id, stream_id, unit_id, kind, path, created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT(run_id, stream_id, unit_id, kind, path) DO NOTHING""",
                    (identifier, stream, partition, kind, resolved_path, now, now),
                )
                artifact = connection.execute(
                    """SELECT * FROM artifacts
                       WHERE run_id=%s AND stream_id=%s AND unit_id=%s
                         AND kind=%s AND path=%s""",
                    (identifier, stream, partition, kind, resolved_path),
                ).fetchone()
                if artifact is None:
                    raise RunStateError("V3.3 output registration did not create a row")
                artifact_id = int(artifact["artifact_id"])
                if str(artifact["status"]) == "ready":
                    if (
                        int(artifact["byte_count"]) != size
                        or str(artifact["sha256"]) != digest
                    ):
                        raise RunStateError(
                            f"ready V3.3 {kind} changed on disk: {resolved_path}"
                        )
                elif str(artifact["status"]) in {"writing", "failed"}:
                    changed = connection.execute(
                        """UPDATE artifacts SET status='ready', byte_count=%s,
                           sha256=%s, updated_at=%s WHERE artifact_id=%s
                           AND status IN ('writing','failed')""",
                        (size, digest, now, artifact_id),
                    ).rowcount
                    if changed != 1:
                        raise RunStateError(f"cannot publish V3.3 {kind}")
                else:
                    raise RunStateError(
                        f"V3.3 {kind} is unavailable for publish: {artifact['status']}"
                    )
                artifact_ids.append(artifact_id)
            if production is None:
                finalize_rows = connection.execute(
                    """SELECT j.job_id FROM jobs j
                       JOIN spatial_units u
                         ON u.run_id=j.run_id AND u.unit_id=j.unit_id
                       WHERE j.run_id=%s AND j.stream_id=%s
                         AND j.job_type='fragmentation_v33'
                         AND u.unit_type='FragmentationV33Finalize'
                         AND j.status IN ('queued','interrupted','running')""",
                    (identifier, stream),
                ).fetchall()
                if len(finalize_rows) != 1:
                    raise RunStateError(
                        "staged V3.3 output requires exactly one active finalize job"
                    )
                finalize_job_id = int(finalize_rows[0]["job_id"])
                # The global audit still reads the frozen baseline after the
                # candidate releases its inputs. Pin it in this publication
                # transaction so cleanup never observes an unreferenced gap.
                baseline_rows = connection.execute(
                    """SELECT artifact_id FROM artifacts
                       WHERE run_id=%s AND stream_id=%s AND unit_id=%s
                         AND kind='v3_baseline_core' AND status='ready'
                       ORDER BY artifact_id FOR UPDATE""",
                    (identifier, stream, partition),
                ).fetchall()
                if len(baseline_rows) != 1:
                    raise RunStateError(
                        "staged V3.3 output requires exactly one ready "
                        f"v3_baseline_core Artifact for {partition}"
                    )
                baseline_id = int(baseline_rows[0]["artifact_id"])
                for artifact_id in [baseline_id, *artifact_ids]:
                    connection.execute(
                        """INSERT INTO artifact_dependencies
                           (job_id, artifact_id, created_at) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING""",
                        (finalize_job_id, artifact_id, now),
                    )
        return artifact_ids[0], artifact_ids[1]

    def add_artifact_dependency(self, job_id: int, artifact_id: int) -> bool:
        """Attach a ready input to a job; the trigger updates ref_count atomically."""
        with self._session.transaction() as connection:
            relation = connection.execute(
                """SELECT 1 FROM jobs j JOIN artifacts a ON a.run_id=j.run_id
                   WHERE j.job_id=%s AND a.artifact_id=%s AND a.status='ready'"""
                + " FOR UPDATE OF a",
                (int(job_id), int(artifact_id)),
            ).fetchone()
            if relation is None:
                raise RunStateError(
                    "artifact dependency requires a ready artifact from the same run"
                )
            cursor = connection.execute(
                """INSERT INTO artifact_dependencies
                   (job_id, artifact_id, created_at) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING""",
                (int(job_id), int(artifact_id), _now()),
            )
            return bool(cursor.rowcount == 1)

    def release_artifact_dependency(self, job_id: int, artifact_id: int) -> bool:
        """Release one job input; the trigger prevents a negative ref_count."""
        with self._session.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM artifact_dependencies WHERE job_id=%s AND artifact_id=%s",
                (int(job_id), int(artifact_id)),
            )
            return bool(cursor.rowcount == 1)

    def release_job_artifacts(self, job_id: int) -> int:
        with self._session.transaction() as connection:
            artifact_ids = [
                int(row["artifact_id"])
                for row in connection.execute(
                    """SELECT artifact_id FROM artifact_dependencies
                       WHERE job_id=%s ORDER BY artifact_id""",
                    (int(job_id),),
                ).fetchall()
            ]
            if artifact_ids:
                placeholders = ",".join("%s" for _ in artifact_ids)
                # Every releaser locks shared Artifact rows in the same order
                # before DELETE triggers decrement ref_count.  This prevents
                # cross-unit deadlocks without reducing worker concurrency.
                connection.execute(
                    f"""SELECT artifact_id FROM artifacts
                        WHERE artifact_id IN ({placeholders})
                        ORDER BY artifact_id FOR UPDATE""",
                    artifact_ids,
                ).fetchall()
            return int(
                connection.execute(
                    "DELETE FROM artifact_dependencies WHERE job_id=%s",
                    (int(job_id),),
                ).rowcount
            )

    def cleanup_candidates(
        self,
        run_id: str,
        *,
        limit: int = 100,
        kinds: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        """Return ready, unreferenced artifacts; deletion remains an explicit caller action."""
        sql = (
            "SELECT * FROM artifacts WHERE run_id=%s AND status='ready' AND ref_count=0"
        )
        values: list[Any] = [str(run_id)]
        if kinds:
            sql += " AND kind IN (" + ",".join("%s" for _ in kinds) + ")"
            values.extend(str(item) for item in kinds)
        sql += " ORDER BY artifact_id LIMIT %s"
        values.append(max(1, min(int(limit), 1000)))
        with self._session.connection() as connection:
            return [dict(row) for row in connection.execute(sql, values).fetchall()]

    def claim_artifact_cleanup(self, artifact_id: int) -> dict[str, Any] | None:
        """Atomically reserve one unreferenced ready Artifact for deletion."""
        with self._session.transaction() as connection:
            changed = connection.execute(
                """UPDATE artifacts SET status='cleaning', updated_at=%s
                   WHERE artifact_id=%s AND status='ready' AND ref_count=0""",
                (_now(), int(artifact_id)),
            ).rowcount
            if changed != 1:
                return None
            return _row_dict(
                connection.execute(
                    "SELECT * FROM artifacts WHERE artifact_id=%s",
                    (int(artifact_id),),
                ).fetchone()
            )

    def finish_artifact_cleanup(
        self,
        artifact_id: int,
        *,
        success: bool,
    ) -> bool:
        """Commit cleanup or return the claimed Artifact to ready state."""
        next_status = "cleaned" if success else "ready"
        with self._session.transaction() as connection:
            return bool(
                connection.execute(
                    """UPDATE artifacts SET status=%s, updated_at=%s
                   WHERE artifact_id=%s AND status='cleaning' AND ref_count=0""",
                    (next_status, _now(), int(artifact_id)),
                ).rowcount
                == 1
            )

    def artifact_cleanup_summary(self, run_id: str) -> dict[str, int]:
        with self._session.connection() as connection:
            row = connection.execute(
                """SELECT COUNT(*) AS artifact_count,
                          COALESCE(SUM(byte_count), 0) AS cleaned_bytes
                   FROM artifacts WHERE run_id=%s AND status='cleaned'""",
                (str(run_id),),
            ).fetchone()
        return {
            "artifact_count": int(row["artifact_count"]),
            "cleaned_bytes": int(row["cleaned_bytes"]),
        }

    def artifact_byte_count(
        self,
        run_id: str,
        *,
        kind: str,
        statuses: Sequence[str],
    ) -> int:
        """Return persisted bytes for one artifact lifecycle class."""

        values = tuple(str(item) for item in statuses)
        if not values:
            return 0
        placeholders = ",".join("%s" for _ in values)
        with self._session.connection() as connection:
            return int(
                connection.execute(
                    f"""SELECT COALESCE(SUM(byte_count), 0) FROM artifacts
                        WHERE run_id=%s AND kind=%s
                          AND status IN ({placeholders})""",
                    (str(run_id), str(kind), *values),
                ).fetchone()[0]
            )

    def artifacts_for_stream(
        self,
        run_id: str,
        stream_id: str,
        *,
        kind: str | None = None,
        status: str | None = "ready",
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM artifacts WHERE run_id=%s AND stream_id=%s"
        values: list[Any] = [str(run_id), str(stream_id)]
        if kind is not None:
            sql += " AND kind=%s"
            values.append(str(kind))
        if status is not None:
            sql += " AND status=%s"
            values.append(str(status))
        sql += " ORDER BY unit_id, artifact_id"
        with self._session.connection() as connection:
            return [dict(row) for row in connection.execute(sql, values).fetchall()]

    def artifact_for_stream_unit(
        self,
        run_id: str,
        stream_id: str,
        unit_id: str,
        kind: str,
        *,
        status: str = "ready",
    ) -> dict[str, Any] | None:
        with self._session.connection() as connection:
            return _row_dict(
                connection.execute(
                    """SELECT * FROM artifacts WHERE run_id=%s AND stream_id=%s
                       AND unit_id=%s AND kind=%s AND status=%s""",
                    (
                        str(run_id),
                        str(stream_id),
                        str(unit_id),
                        str(kind),
                        str(status),
                    ),
                ).fetchone()
            )
