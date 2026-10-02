"""Job scheduling, lease fencing, transitions, and recovery persistence."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from typing import Any, Iterable, Mapping, Sequence

from labeling_tool.shared.state.run_state_session import RunStateError, RunStateSession
from labeling_tool.shared.state.state_values import bounded_utf8 as _bounded_utf8
from labeling_tool.shared.state.state_values import json_value as _json
from labeling_tool.shared.state.state_values import row_dict as _row_dict
from labeling_tool.shared.state.state_values import utc_now as _now

__all__ = ["JobRepository"]


class JobRepository:
    """Own Job scheduling, attempt fencing, transitions, and recovery."""

    def __init__(self, session: RunStateSession) -> None:
        self._session = session

    def open_frontier_summary(self, run_id: str) -> dict[str, Any]:
        """Return cross-package spatial units with only some dependencies ready."""
        unit_state = """
            SELECT ud.unit_id,
                   SUM(CASE WHEN wp.status='ready' THEN 1 ELSE 0 END) AS ready_count,
                   COUNT(*) AS dependency_count
            FROM unit_dependencies ud
            JOIN partitions p
              ON p.run_id=ud.run_id AND p.partition_id=ud.partition_id
            JOIN work_packages wp
              ON wp.run_id=p.run_id AND wp.package_id=p.package_id
            WHERE ud.run_id=%s
            GROUP BY ud.unit_id
        """
        with self._session.connection() as connection:
            rows = connection.execute(unit_state, (str(run_id),)).fetchall()
            open_unit_ids = [
                str(row["unit_id"])
                for row in rows
                if 0 < int(row["ready_count"]) < int(row["dependency_count"])
            ]
            if not open_unit_ids:
                return {"unit_count": 0, "unit_ids": [], "neighbor_package_ids": []}
            placeholders = ",".join("%s" for _ in open_unit_ids)
            package_rows = connection.execute(
                f"""SELECT DISTINCT wp.package_id, wp.sequence_no
                    FROM unit_dependencies ud
                    JOIN partitions p
                      ON p.run_id=ud.run_id AND p.partition_id=ud.partition_id
                    JOIN work_packages wp
                      ON wp.run_id=p.run_id AND wp.package_id=p.package_id
                    WHERE ud.run_id=%s AND ud.unit_id IN ({placeholders})
                      AND wp.status IN ('queued','interrupted')
                    ORDER BY wp.sequence_no""",
                [str(run_id), *open_unit_ids],
            ).fetchall()
        return {
            "unit_count": len(open_unit_ids),
            "unit_ids": open_unit_ids,
            "neighbor_package_ids": [str(row["package_id"]) for row in package_rows],
        }

    def active_work_package_job(self, run_id: str) -> dict[str, Any] | None:
        """Return the single active accelerator Package, if one is leased."""
        with self._session.connection() as connection:
            return _row_dict(
                connection.execute(
                    """SELECT j.*, wp.sequence_no,
                              wp.updated_at AS package_started_at
                       FROM jobs j
                       JOIN work_packages wp
                         ON wp.run_id=j.run_id AND wp.package_id=j.package_id
                       WHERE j.run_id=%s AND j.job_type='work_package'
                         AND j.status='running' AND wp.status='running'
                       ORDER BY wp.sequence_no, j.job_id LIMIT 1""",
                    (str(run_id),),
                ).fetchone()
            )

    def insert_jobs(self, run_id: str, jobs: Iterable[Mapping[str, Any]]) -> int:
        now = _now()
        count = 0

        def rows() -> Iterator[tuple[Any, ...]]:
            nonlocal count
            for item in jobs:
                count += 1
                yield (
                    str(run_id),
                    str(item["job_type"]),
                    str(item.get("stream_id") or ""),
                    str(item.get("tile_id") or ""),
                    str(item.get("unit_id") or ""),
                    str(item.get("package_id") or ""),
                    str(item.get("status") or "queued"),
                    int(item.get("priority", 0)),
                    int(item.get("max_attempts", 3)),
                    now,
                    now,
                )

        with self._session.transaction() as connection:
            connection.executemany(
                """INSERT INTO jobs
                   (run_id, job_type, stream_id, tile_id, unit_id, package_id,
                    status, priority, max_attempts, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                rows(),
            )
        return count

    def get_job(self, job_id: int) -> dict[str, Any] | None:
        with self._session.connection() as connection:
            return _row_dict(
                connection.execute(
                    "SELECT * FROM jobs WHERE job_id=%s", (int(job_id),)
                ).fetchone()
            )

    def update_job_monitor_runtime(
        self, job_id: int, span_id: str, payload: Mapping[str, Any]
    ) -> bool:
        """Replace bounded observation data only for the exact active attempt.

        This is not a lease renewal or a scheduling transition. Old callbacks
        cannot update a replacement attempt, even when its job ID is reused.
        """
        if not span_id:
            return False
        allowed = {
            "event",
            "stream_id",
            "effective_device",
            "tile_current",
            "tile_total",
            "configured_batch_size",
            "effective_batch_size",
            "status",
            "notice",
        }
        data = {
            key: value[:2048] if isinstance(value, str) else value
            for key, value in payload.items()
            if key in allowed
            and (value is None or isinstance(value, (str, int, float, bool)))
        }
        data["observed_at"] = _now()
        with self._session.transaction() as connection:
            return bool(
                connection.execute(
                    """UPDATE jobs SET monitor_runtime_json=%s
                   WHERE job_id=%s AND monitor_span_id=%s AND status='running'""",
                    (_json(data), int(job_id), str(span_id)),
                ).rowcount
                == 1
            )

    def _lease_selected_job(
        self,
        connection: Any,
        row: Any,
        worker_id: str,
        token: str,
        expires: float,
        now: str,
    ) -> dict[str, Any]:
        """Lease one selected Job and its Work Package in one transaction."""
        job_id = int(row["job_id"])
        execution_id = str(self._session.execution_id or "").strip()
        span_id = ""
        updated = connection.execute(
            """UPDATE jobs SET status='running', attempt=attempt+1,
               progress_current=0, progress_total=0,
               worker_id=%s, lease_token=%s, lease_expires=%s, heartbeat_at=%s,
               monitor_execution_id=%s, monitor_span_id='', monitor_runtime_json='{}',
               updated_at=%s WHERE job_id=%s
               AND status IN ('queued','interrupted') AND attempt < max_attempts""",
            (
                str(worker_id),
                token,
                expires,
                now,
                execution_id,
                now,
                job_id,
            ),
        )
        if updated.rowcount != 1:
            raise RunStateError("Job state changed during lease acquisition")
        if str(row["job_type"]) == "work_package":
            package_updated = connection.execute(
                """UPDATE work_packages SET status='running', updated_at=%s
                   WHERE run_id=%s AND package_id=%s
                     AND status IN ('queued','interrupted')""",
                (now, str(row["run_id"]), str(row["package_id"])),
            )
            if package_updated.rowcount != 1:
                raise RunStateError(
                    "Work Package state changed during lease acquisition"
                )
        if execution_id:
            budget_attempt = int(row["attempt"] or 0) + 1
            history_attempt = 1 + int(
                connection.execute(
                    "SELECT COUNT(*) FROM monitor_spans WHERE run_id=%s AND job_id=%s AND span_kind='job_attempt'",
                    (str(row["run_id"]), job_id),
                ).fetchone()[0]
            )
            idempotency_key = (
                f"job:{job_id}:execution:{execution_id}:"
                f"history-attempt:{history_attempt}"
            )
            inserted = connection.execute(
                """INSERT INTO monitor_spans
                   (span_id,run_id,execution_id,span_kind,object_type,object_id,
                    stream_id,package_id,unit_id,job_id,attempt_no,budget_attempt,
                    status,started_at,last_observed_at,metadata_json,idempotency_key)
                   SELECT %s,%s,%s,'job_attempt',%s,%s,%s,%s,%s,%s,%s,%s,
                          'running',%s,%s,%s,%s
                   WHERE EXISTS (
                     SELECT 1 FROM monitor_executions
                     WHERE execution_id=%s AND run_id=%s AND status='running'
                   )
                   ON CONFLICT(run_id,idempotency_key) DO UPDATE SET
                     last_observed_at=EXCLUDED.last_observed_at
                   RETURNING span_id""",
                (
                    str(uuid.uuid4()),
                    str(row["run_id"]),
                    execution_id,
                    "package" if str(row["job_type"]) == "work_package" else "job",
                    str(row["package_id"] or row["unit_id"] or job_id),
                    str(row["stream_id"] or ""),
                    str(row["package_id"] or ""),
                    str(row["unit_id"] or ""),
                    job_id,
                    history_attempt,
                    budget_attempt,
                    now,
                    now,
                    _json({"job_type": str(row["job_type"])}),
                    idempotency_key,
                    execution_id,
                    str(row["run_id"]),
                ),
            ).fetchone()
            if inserted is not None:
                span_id = str(inserted[0])
                connection.execute(
                    """UPDATE jobs SET monitor_span_id=%s
                       WHERE job_id=%s AND status='running'
                         AND lease_token=%s""",
                    (span_id, job_id, token),
                )
                if history_attempt > 1:
                    object_type = (
                        "package" if str(row["job_type"]) == "work_package" else "job"
                    )
                    object_id = str(row["package_id"] or row["unit_id"] or job_id)
                    connection.execute(
                        """INSERT INTO monitor_events
                           (run_id,execution_id,span_id,timestamp,level,event_type,
                            object_type,object_id,stream_id,package_id,unit_id,job_id,
                            message,payload_json,idempotency_key)
                           VALUES (%s,%s,%s,%s,'warning','job_retry_started',%s,%s,
                                   %s,%s,%s,%s,%s,%s,%s)
                           ON CONFLICT(run_id,idempotency_key) DO NOTHING""",
                        (
                            str(row["run_id"]),
                            execution_id,
                            span_id,
                            now,
                            object_type,
                            object_id,
                            str(row["stream_id"] or ""),
                            str(row["package_id"] or ""),
                            str(row["unit_id"] or ""),
                            job_id,
                            f"Job history attempt {history_attempt} started",
                            _json(
                                {
                                    "job_type": str(row["job_type"]),
                                    "attempt_no": history_attempt,
                                    "budget_attempt": budget_attempt,
                                }
                            ),
                            f"{idempotency_key}:started",
                        ),
                    )
        leased = connection.execute(
            "SELECT * FROM jobs WHERE job_id=%s", (job_id,)
        ).fetchone()
        if leased is None:
            raise RunStateError("leased Job disappeared during lease acquisition")
        return dict(leased)

    def lease_next_job(
        self,
        run_id: str,
        worker_id: str,
        *,
        job_types: Sequence[str] = (),
        lease_seconds: float = 60.0,
        exclude_job_ids: Sequence[int] = (),
    ) -> dict[str, Any] | None:
        token = uuid.uuid4().hex
        expires = time.time() + max(1.0, float(lease_seconds))
        now = _now()
        sql = (
            "SELECT * FROM jobs WHERE run_id=%s "
            "AND status IN ('queued','interrupted') AND attempt < max_attempts "
            "AND EXISTS (SELECT 1 FROM runs r WHERE r.run_id=jobs.run_id "
            "AND r.status IN ('preflight','planned','running','raster_ready')) "
            "AND (job_type!='work_package' OR (EXISTS ("
            " SELECT 1 FROM work_packages wp WHERE wp.run_id=jobs.run_id"
            " AND wp.package_id=jobs.package_id"
            " AND wp.status IN ('queued','interrupted'))"
            " AND NOT EXISTS (SELECT 1 FROM jobs failed_package"
            " WHERE failed_package.run_id=jobs.run_id"
            " AND failed_package.job_type='work_package'"
            " AND failed_package.status='failed'))) "
            "AND (job_type NOT IN ('unit_fit','unit_confidence') OR NOT EXISTS ("
            "  SELECT 1 FROM unit_dependencies ud"
            "  LEFT JOIN partitions p ON p.run_id=ud.run_id"
            "   AND p.partition_id=ud.partition_id"
            "  LEFT JOIN work_packages wp ON wp.run_id=p.run_id"
            "   AND wp.package_id=p.package_id"
            "  WHERE ud.run_id=jobs.run_id AND ud.unit_id=jobs.unit_id"
            "   AND COALESCE(wp.status,'')!='ready')) "
            "AND (job_type!='unit_confidence' OR "
            " (SELECT COUNT(*) FROM artifact_dependencies ad"
            "  JOIN artifacts a ON a.artifact_id=ad.artifact_id"
            "  WHERE ad.job_id=jobs.job_id"
            "    AND a.kind='partition_probability' AND a.status='ready')="
            " (SELECT COUNT(*) FROM unit_dependencies ud"
            "  WHERE ud.run_id=jobs.run_id AND ud.unit_id=jobs.unit_id)) "
            "AND (job_type!='unit_fit' OR ("
            " (NOT EXISTS (SELECT 1 FROM jobs v33"
            "  WHERE v33.run_id=jobs.run_id AND v33.stream_id=jobs.stream_id"
            "    AND v33.job_type='fragmentation_v33')"
            " OR NOT EXISTS (SELECT 1 FROM jobs v33"
            "  WHERE v33.run_id=jobs.run_id AND v33.stream_id=jobs.stream_id"
            "    AND v33.job_type='fragmentation_v33' AND v33.status!='ready'))"
            " AND ((EXISTS (SELECT 1 FROM jobs compact"
            "  WHERE compact.run_id=jobs.run_id"
            "    AND compact.stream_id=jobs.stream_id"
            "    AND compact.unit_id=jobs.unit_id"
            "    AND compact.job_type='unit_confidence')"
            " AND 1=(SELECT COUNT(*) FROM artifact_dependencies ad"
            "  JOIN artifacts a ON a.artifact_id=ad.artifact_id"
            "  WHERE ad.job_id=jobs.job_id"
            "    AND a.kind='unit_confidence' AND a.status='ready'))"
            " OR (NOT EXISTS (SELECT 1 FROM jobs compact"
            "  WHERE compact.run_id=jobs.run_id"
            "    AND compact.stream_id=jobs.stream_id"
            "    AND compact.unit_id=jobs.unit_id"
            "    AND compact.job_type='unit_confidence')"
            " AND (SELECT COUNT(*) FROM artifact_dependencies ad"
            "  WHERE ad.job_id=jobs.job_id)="
            " (SELECT COUNT(*) FROM unit_dependencies ud"
            "  WHERE ud.run_id=jobs.run_id AND ud.unit_id=jobs.unit_id)))))"
        )
        values: list[Any] = [str(run_id)]
        if job_types:
            sql += " AND job_type IN (" + ",".join("%s" for _ in job_types) + ")"
            values.extend(str(item) for item in job_types)
        excluded = tuple(sorted({int(item) for item in exclude_job_ids}))
        if excluded:
            sql += " AND job_id NOT IN (" + ",".join("%s" for _ in excluded) + ")"
            values.extend(excluded)
        sql += " ORDER BY priority DESC, job_id LIMIT 1"
        sql += " FOR UPDATE SKIP LOCKED"
        with self._session.transaction() as connection:
            row = connection.execute(sql, values).fetchone()
            if row is None:
                return None
            return self._lease_selected_job(
                connection, row, worker_id, token, expires, now
            )

    def lease_next_work_package(
        self,
        run_id: str,
        worker_id: str,
        *,
        max_open_frontier_units: int,
        lease_seconds: float = 60.0,
    ) -> dict[str, Any] | None:
        """Prefer a package that closes open Seam/Junction dependencies."""
        frontier = self.open_frontier_summary(run_id)
        preferred = list(frontier["neighbor_package_ids"])
        if (
            int(frontier["unit_count"]) >= max(1, int(max_open_frontier_units))
            and preferred
        ):
            placeholders = ",".join("%s" for _ in preferred)
            with self._session.connection() as connection:
                row = connection.execute(
                    f"""SELECT j.job_id
                        FROM jobs j
                        JOIN work_packages wp
                          ON wp.run_id=j.run_id AND wp.package_id=j.package_id
                        WHERE j.run_id=%s AND j.job_type='work_package'
                          AND j.status IN ('queued','interrupted')
                          AND j.attempt < j.max_attempts
                          AND NOT EXISTS (
                            SELECT 1 FROM jobs failed_package
                            WHERE failed_package.run_id=j.run_id
                              AND failed_package.job_type='work_package'
                              AND failed_package.status='failed'
                          )
                          AND EXISTS (
                            SELECT 1 FROM runs r WHERE r.run_id=j.run_id
                              AND r.status IN (
                                'preflight','planned','running','raster_ready'
                              )
                          )
                          AND wp.status IN ('queued','interrupted')
                          AND j.package_id IN ({placeholders})
                        ORDER BY j.priority DESC, wp.sequence_no, j.job_id LIMIT 1""",
                    [str(run_id), *preferred],
                ).fetchone()
            if row is not None:
                leased = self.lease_job(
                    int(row["job_id"]), worker_id, lease_seconds=lease_seconds
                )
                if leased is not None:
                    return leased
        return self.lease_next_job(
            run_id,
            worker_id,
            job_types=("work_package",),
            lease_seconds=lease_seconds,
        )

    def lease_next_fragmentation_v33(
        self,
        run_id: str,
        worker_id: str,
        *,
        lease_seconds: float = 120.0,
        max_running: int = 4,
        exclude_job_ids: Sequence[int] = (),
    ) -> dict[str, Any] | None:
        """Lease one V3.3 candidate only after every owner input is ready.

        A candidate dependency is complete only when it owns the frozen V3
        owner-Core context, preserved V3 baseline Core, and matching probability
        Halo for every Partition listed in ``unit_dependencies``.  The owner
        Work Packages must also be atomically ready, so a candidate never
        observes a half-committed first stage.
        """

        token = uuid.uuid4().hex
        expires = time.time() + max(1.0, float(lease_seconds))
        now = _now()
        with self._session.transaction() as connection:
            locked_run = connection.execute(
                "SELECT run_id FROM runs WHERE run_id=%s FOR UPDATE",
                (str(run_id),),
            ).fetchone()
            if locked_run is None:
                return None
            running = int(
                connection.execute(
                    """SELECT COUNT(*) FROM jobs WHERE run_id=%s
                       AND job_type='fragmentation_v33' AND status='running'""",
                    (str(run_id),),
                ).fetchone()[0]
            )
            if running >= min(4, max(1, int(max_running))):
                return None
            lock_clause = " FOR UPDATE SKIP LOCKED"
            excluded = tuple(sorted({int(item) for item in exclude_job_ids}))
            exclusion_sql = ""
            values: list[Any] = [str(run_id)]
            if excluded:
                exclusion_sql = (
                    " AND j.job_id NOT IN (" + ",".join("%s" for _ in excluded) + ")"
                )
                values.extend(excluded)
            row = connection.execute(
                """SELECT j.* FROM jobs j
                   JOIN spatial_units u
                     ON u.run_id=j.run_id AND u.unit_id=j.unit_id
                   WHERE j.run_id=%s AND j.job_type='fragmentation_v33'
                     AND j.status IN ('queued','interrupted')
                     AND j.attempt < j.max_attempts
                     AND EXISTS (
                       SELECT 1 FROM runs r WHERE r.run_id=j.run_id
                         AND r.status IN (
                           'preflight','planned','running','raster_ready'
                         )
                     )
                     AND (u.unit_type='FragmentationV33Finalize' OR NOT EXISTS (
                       SELECT 1 FROM unit_dependencies d
                       LEFT JOIN partitions p
                         ON p.run_id=d.run_id
                        AND p.partition_id=d.partition_id
                       LEFT JOIN work_packages wp
                         ON wp.run_id=p.run_id
                        AND wp.package_id=p.package_id
                       WHERE d.run_id=j.run_id AND d.unit_id=j.unit_id
                         AND COALESCE(wp.status,'')!='ready'
                     ))
                     AND (u.unit_type='FragmentationV33Finalize' OR NOT EXISTS (
                       SELECT 1 FROM unit_dependencies d
                       WHERE d.run_id=j.run_id AND d.unit_id=j.unit_id
                         AND (
                           NOT EXISTS (
                             SELECT 1 FROM artifact_dependencies ad
                             JOIN artifacts a ON a.artifact_id=ad.artifact_id
                             WHERE ad.job_id=j.job_id
                               AND a.run_id=j.run_id
                               AND a.stream_id=j.stream_id
                               AND a.unit_id=d.partition_id
                               AND a.kind='partition_probability'
                               AND a.status='ready'
                           )
                           OR NOT EXISTS (
                             SELECT 1 FROM artifact_dependencies ad
                             JOIN artifacts a ON a.artifact_id=ad.artifact_id
                             WHERE ad.job_id=j.job_id
                               AND a.run_id=j.run_id
                               AND a.stream_id=j.stream_id
                               AND a.unit_id=d.partition_id
                               AND a.kind='v3_context_core'
                               AND a.status='ready'
                           )
                           OR NOT EXISTS (
                             SELECT 1 FROM artifact_dependencies ad
                             JOIN artifacts a ON a.artifact_id=ad.artifact_id
                             WHERE ad.job_id=j.job_id
                               AND a.run_id=j.run_id
                               AND a.stream_id=j.stream_id
                               AND a.unit_id=d.partition_id
                               AND a.kind='v3_baseline_core'
                               AND a.status='ready'
                           )
                         )
                     ))
                     AND (u.unit_type!='FragmentationV33Finalize' OR (
                       NOT EXISTS (
                         SELECT 1 FROM jobs owner_job
                         JOIN spatial_units owner_unit
                           ON owner_unit.run_id=owner_job.run_id
                          AND owner_unit.unit_id=owner_job.unit_id
                         WHERE owner_job.run_id=j.run_id
                           AND owner_job.stream_id=j.stream_id
                           AND owner_job.job_type='fragmentation_v33'
                           AND owner_unit.unit_type='FragmentationV33Partition'
                           AND owner_job.status!='ready'
                       )
                       AND NOT EXISTS (
                         SELECT 1 FROM unit_dependencies d
                         WHERE d.run_id=j.run_id AND d.unit_id=j.unit_id
                           AND (NOT EXISTS (
                             SELECT 1 FROM artifacts a
                             WHERE a.run_id=j.run_id AND a.stream_id=j.stream_id
                               AND a.unit_id=d.partition_id
                               AND a.kind='v33_staged_mask' AND a.status='ready'
                           ) OR NOT EXISTS (
                             SELECT 1 FROM artifacts a
                             WHERE a.run_id=j.run_id AND a.stream_id=j.stream_id
                               AND a.unit_id=d.partition_id
                               AND a.kind='v33_staged_audit' AND a.status='ready'
                           ))
                       )
                     ))"""
                + exclusion_sql
                + " ORDER BY j.priority DESC, j.job_id LIMIT 1"
                + lock_clause,
                values,
            ).fetchone()
            if row is None:
                return None
            return self._lease_selected_job(
                connection, row, worker_id, token, expires, now
            )

    def lease_job(
        self,
        job_id: int,
        worker_id: str,
        *,
        lease_seconds: float = 60.0,
    ) -> dict[str, Any] | None:
        token = uuid.uuid4().hex
        expires = time.time() + max(1.0, float(lease_seconds))
        now = _now()
        with self._session.transaction() as connection:
            lock_clause = " FOR UPDATE SKIP LOCKED"
            row = connection.execute(
                """SELECT * FROM jobs WHERE job_id=%s
                   AND status IN ('queued','interrupted') AND attempt < max_attempts
                   AND EXISTS (
                     SELECT 1 FROM runs r WHERE r.run_id=jobs.run_id
                       AND r.status IN (
                         'preflight','planned','running','raster_ready'
                       )
                   )
                   AND (job_type!='work_package' OR EXISTS (
                     SELECT 1 FROM work_packages wp
                     WHERE wp.run_id=jobs.run_id
                       AND wp.package_id=jobs.package_id
                       AND wp.status IN ('queued','interrupted')
                   ))
                   AND (job_type!='work_package' OR NOT EXISTS (
                     SELECT 1 FROM jobs failed_package
                     WHERE failed_package.run_id=jobs.run_id
                       AND failed_package.job_type='work_package'
                       AND failed_package.status='failed'
                   ))
                   AND (job_type NOT IN ('unit_fit','unit_confidence') OR
                      NOT EXISTS (
                        SELECT 1 FROM unit_dependencies ud
                        LEFT JOIN partitions p
                          ON p.run_id=ud.run_id
                         AND p.partition_id=ud.partition_id
                        LEFT JOIN work_packages wp
                          ON wp.run_id=p.run_id AND wp.package_id=p.package_id
                        WHERE ud.run_id=jobs.run_id
                          AND ud.unit_id=jobs.unit_id
                          AND COALESCE(wp.status,'')!='ready'
                      ))"""
                + lock_clause,
                (int(job_id),),
            ).fetchone()
            if row is None:
                return None
            job_type = str(row["job_type"])
            if job_type == "unit_confidence":
                ready_probabilities = int(
                    connection.execute(
                        """SELECT COUNT(*) FROM artifact_dependencies ad
                           JOIN artifacts a ON a.artifact_id=ad.artifact_id
                           WHERE ad.job_id=%s
                             AND a.kind='partition_probability'
                             AND a.status='ready'""",
                        (int(job_id),),
                    ).fetchone()[0]
                )
                unit_dependencies = int(
                    connection.execute(
                        """SELECT COUNT(*) FROM unit_dependencies ud
                           WHERE ud.run_id=%s AND ud.unit_id=%s""",
                        (str(row["run_id"]), str(row["unit_id"])),
                    ).fetchone()[0]
                )
                if ready_probabilities != unit_dependencies:
                    return None
            elif job_type == "unit_fit":
                v33_incomplete = int(
                    connection.execute(
                        """SELECT COUNT(*) FROM jobs v33
                           WHERE v33.run_id=%s AND v33.stream_id=%s
                             AND v33.job_type='fragmentation_v33'
                             AND v33.status!='ready'""",
                        (str(row["run_id"]), str(row["stream_id"])),
                    ).fetchone()[0]
                )
                if v33_incomplete:
                    return None
                compact_jobs = int(
                    connection.execute(
                        """SELECT COUNT(*) FROM jobs compact
                           WHERE compact.run_id=%s AND compact.stream_id=%s
                             AND compact.unit_id=%s
                             AND compact.job_type='unit_confidence'""",
                        (
                            str(row["run_id"]),
                            str(row["stream_id"]),
                            str(row["unit_id"]),
                        ),
                    ).fetchone()[0]
                )
                dependency_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM artifact_dependencies WHERE job_id=%s",
                        (int(job_id),),
                    ).fetchone()[0]
                )
                if compact_jobs:
                    ready_confidence = int(
                        connection.execute(
                            """SELECT COUNT(*) FROM artifact_dependencies ad
                               JOIN artifacts a
                                 ON a.artifact_id=ad.artifact_id
                               WHERE ad.job_id=%s
                                 AND a.kind='unit_confidence'
                                 AND a.status='ready'""",
                            (int(job_id),),
                        ).fetchone()[0]
                    )
                    if dependency_count != 1 or ready_confidence != 1:
                        return None
                else:
                    unit_dependencies = int(
                        connection.execute(
                            """SELECT COUNT(*) FROM unit_dependencies ud
                               WHERE ud.run_id=%s AND ud.unit_id=%s""",
                            (str(row["run_id"]), str(row["unit_id"])),
                        ).fetchone()[0]
                    )
                    if dependency_count != unit_dependencies:
                        return None
            return self._lease_selected_job(
                connection, row, worker_id, token, expires, now
            )

    def heartbeat(
        self,
        job_id: int,
        lease_token: str,
        *,
        current: int,
        total: int,
        lease_seconds: float = 60.0,
    ) -> bool:
        now = _now()
        fence_time = time.time()
        expires = fence_time + max(1.0, float(lease_seconds))
        with self._session.transaction() as connection:
            return bool(
                connection.execute(
                    """UPDATE jobs SET progress_current=%s, progress_total=%s,
                   heartbeat_at=%s, lease_expires=%s, updated_at=%s
                   WHERE job_id=%s AND status='running' AND lease_token=%s
                     AND lease_expires IS NOT NULL AND lease_expires>=%s""",
                    (
                        max(0, int(current)),
                        max(0, int(total)),
                        now,
                        expires,
                        now,
                        int(job_id),
                        str(lease_token),
                        fence_time,
                    ),
                ).rowcount
                == 1
            )

    def finish_job(
        self,
        job_id: int,
        lease_token: str,
        *,
        status: str = "ready",
        error: str = "",
    ) -> bool:
        if status not in {"ready", "failed", "stopped"}:
            raise ValueError(f"invalid terminal job status: {status}")
        now = _now()
        fence_time = time.time()
        with self._session.transaction() as connection:
            return bool(
                connection.execute(
                    """UPDATE jobs SET status=%s, error=%s, worker_id='',
                   lease_token='', lease_expires=NULL, heartbeat_at=%s, updated_at=%s
                   WHERE job_id=%s AND status='running' AND lease_token=%s
                     AND lease_expires IS NOT NULL AND lease_expires>=%s""",
                    (
                        str(status),
                        str(error),
                        now,
                        now,
                        int(job_id),
                        str(lease_token),
                        fence_time,
                    ),
                ).rowcount
                == 1
            )

    def complete_fragmentation_v33_job(
        self,
        job_id: int,
        lease_token: str,
    ) -> bool:
        """Atomically complete V3.3 and release all retained owner inputs."""

        now = _now()
        fence_time = time.time()
        with self._session.transaction() as connection:
            job = connection.execute(
                """SELECT * FROM jobs WHERE job_id=%s
                   AND job_type='fragmentation_v33' AND status='running'
                   AND lease_token=%s AND lease_expires IS NOT NULL
                   AND lease_expires>=%s""",
                (int(job_id), str(lease_token), fence_time),
            ).fetchone()
            if job is None:
                return False
            unit = connection.execute(
                "SELECT unit_type, owner_key FROM spatial_units "
                "WHERE run_id=%s AND unit_id=%s",
                (str(job["run_id"]), str(job["unit_id"])),
            ).fetchone()
            if unit is None:
                raise RunStateError("V3.3 spatial unit disappeared")
            unit_type = str(unit["unit_type"])
            production = unit_type == "FragmentationV33Finalize"
            staged = unit_type == "FragmentationV33Partition"
            mask_kind, audit_kind, report_kind = (
                (
                    "core_mask",
                    "fragmentation_v33_audit",
                    "fragmentation_v33_report",
                )
                if production
                else (
                    "v33_candidate_mask",
                    "v33_candidate_audit",
                    "v33_candidate_report",
                )
            )
            if staged:
                mask_kind, audit_kind, report_kind = (
                    "v33_staged_mask",
                    "v33_staged_audit",
                    "",
                )
            expected = int(
                connection.execute(
                    """SELECT COUNT(*) FROM unit_dependencies
                       WHERE run_id=%s AND unit_id=%s""",
                    (str(job["run_id"]), str(job["unit_id"])),
                ).fetchone()[0]
            )
            if expected < 1:
                raise RunStateError("V3.3 has no owner dependencies")
            expected_owner = str(unit["owner_key"]) if staged else None
            for kind in (mask_kind, audit_kind):
                ready_row = connection.execute(
                    """SELECT COUNT(*) AS artifact_count,
                                  COUNT(DISTINCT a.unit_id) AS owner_count
                           FROM artifacts a
                           JOIN unit_dependencies d
                             ON d.run_id=a.run_id AND d.partition_id=a.unit_id
                           WHERE a.run_id=%s AND a.stream_id=%s
                             AND d.unit_id=%s AND a.kind=%s AND a.status='ready'""",
                    (
                        str(job["run_id"]),
                        str(job["stream_id"]),
                        str(job["unit_id"]),
                        kind,
                    ),
                ).fetchone()
                if expected_owner is not None:
                    ready_row = connection.execute(
                        """SELECT COUNT(*) AS artifact_count,
                                  COUNT(DISTINCT unit_id) AS owner_count
                           FROM artifacts WHERE run_id=%s AND stream_id=%s
                             AND unit_id=%s AND kind=%s AND status='ready'""",
                        (
                            str(job["run_id"]),
                            str(job["stream_id"]),
                            expected_owner,
                            kind,
                        ),
                    ).fetchone()
                ready = int(ready_row["artifact_count"])
                owners = int(ready_row["owner_count"])
                needed = 1 if expected_owner is not None else expected
                if ready != needed or owners != needed:
                    raise RunStateError(
                        f"V3.3 {kind} incomplete or duplicated: "
                        f"artifacts={ready}, owners={owners}, expected={needed}"
                    )
            report_ready = int(
                connection.execute(
                    """SELECT COUNT(*) FROM artifacts WHERE run_id=%s
                       AND stream_id=%s AND unit_id=%s AND kind=%s AND status='ready'""",
                    (
                        str(job["run_id"]),
                        str(job["stream_id"]),
                        str(job["unit_id"]),
                        report_kind,
                    ),
                ).fetchone()[0]
            )
            if not staged and report_ready != 1:
                raise RunStateError("V3.3 acceptance report is not ready")
            changed = connection.execute(
                """UPDATE jobs SET status='ready', error='', worker_id='',
                   progress_current=%s, progress_total=%s, lease_token='',
                   lease_expires=NULL, heartbeat_at=%s, updated_at=%s
                   WHERE job_id=%s AND job_type='fragmentation_v33'
                     AND status='running' AND lease_token=%s
                     AND lease_expires IS NOT NULL AND lease_expires>=%s""",
                (
                    1 if staged else expected,
                    1 if staged else expected,
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

    def work_package_job_holds_lease(
        self,
        run_id: str,
        package_id: str,
        job_id: int,
        lease_token: str,
    ) -> bool:
        """Return whether the exact Package job currently owns this lease."""
        token = str(lease_token)
        if not token:
            return False
        with self._session.connection() as connection:
            row = connection.execute(
                """SELECT 1 FROM jobs
                   WHERE job_id=%s AND run_id=%s AND job_type='work_package'
                     AND package_id=%s AND status='running' AND lease_token=%s
                     AND lease_expires IS NOT NULL AND lease_expires>=%s""",
                (
                    int(job_id),
                    str(run_id),
                    str(package_id),
                    token,
                    time.time(),
                ),
            ).fetchone()
        return row is not None

    def transition_work_package_job(
        self,
        run_id: str,
        package_id: str,
        job_id: int,
        lease_token: str,
        status: str,
        error: str = "",
    ) -> bool:
        """Atomically transition a Package and its exact leased job.

        Only the current, unexpired ``job_id``/``lease_token`` pair may change
        either row. A stale worker, a job for another Package, or a lost lease
        changes neither row. Both updates share one PostgreSQL transaction so
        the Package and control-plane Job cannot diverge.
        """
        target = str(status)
        if target not in {"ready", "failed", "interrupted"}:
            raise ValueError(f"invalid Work Package/job transition status: {target}")
        identifier = str(run_id)
        package = str(package_id)
        token = str(lease_token)
        if not token:
            return False
        now = _now()
        fence_time = time.time()
        job_error = "" if target == "ready" else str(error)
        with self._session.transaction() as connection:
            matching = connection.execute(
                """SELECT 1 FROM jobs
                   WHERE job_id=%s AND run_id=%s AND job_type='work_package'
                     AND package_id=%s AND status='running' AND lease_token=%s
                     AND lease_expires IS NOT NULL AND lease_expires>=%s""",
                (int(job_id), identifier, package, token, fence_time),
            ).fetchone()
            package_running = connection.execute(
                """SELECT 1 FROM work_packages
                   WHERE run_id=%s AND package_id=%s AND status='running'""",
                (identifier, package),
            ).fetchone()
            if matching is None or package_running is None:
                return False
            package_update = connection.execute(
                """UPDATE work_packages SET status=%s, updated_at=%s
                   WHERE run_id=%s AND package_id=%s AND status='running'""",
                (target, now, identifier, package),
            )
            job_update = connection.execute(
                """UPDATE jobs SET status=%s, error=%s, worker_id='',
                   lease_token='', lease_expires=NULL, heartbeat_at=%s, updated_at=%s
                   , attempt=CASE WHEN %s='interrupted'
                                  THEN GREATEST(0, attempt-1) ELSE attempt END
                   WHERE job_id=%s AND run_id=%s AND job_type='work_package'
                     AND package_id=%s AND status='running' AND lease_token=%s
                     AND lease_expires IS NOT NULL AND lease_expires>=%s""",
                (
                    target,
                    job_error,
                    now,
                    now,
                    target,
                    int(job_id),
                    identifier,
                    package,
                    token,
                    fence_time,
                ),
            )
            if package_update.rowcount != 1 or job_update.rowcount != 1:
                raise RunStateError(
                    "Work Package/job state changed during atomic transition"
                )
            return True

    def complete_work_package_job(
        self,
        run_id: str,
        package_id: str,
        job_id: int,
        lease_token: str,
    ) -> bool:
        """Atomically mark a Package and its exact leased job ready."""
        return self.transition_work_package_job(
            run_id, package_id, job_id, lease_token, status="ready"
        )

    def fail_work_package_job(
        self,
        run_id: str,
        package_id: str,
        job_id: int,
        lease_token: str,
        *,
        error: str = "",
    ) -> bool:
        """Atomically fail a Package and its exact leased job."""
        return self.transition_work_package_job(
            run_id,
            package_id,
            job_id,
            lease_token,
            status="failed",
            error=error,
        )

    def interrupt_work_package_job(
        self,
        run_id: str,
        package_id: str,
        job_id: int,
        lease_token: str,
        *,
        error: str = "",
    ) -> bool:
        """Atomically interrupt a Package and its exact leased job."""
        return self.transition_work_package_job(
            run_id,
            package_id,
            job_id,
            lease_token,
            status="interrupted",
            error=error,
        )

    def fail_or_requeue_work_package_job(
        self,
        run_id: str,
        package_id: str,
        job_id: int,
        lease_token: str,
        error: str = "",
    ) -> str | None:
        """Atomically fail or requeue the exact leased Package attempt.

        Returns ``queued`` while another attempt remains, ``failed`` when the
        attempt limit is exhausted, and ``None`` when the lease fence no longer
        belongs to the caller. A stale worker therefore cannot alter the
        Package currently owned by a newer lease.
        """
        identifier = str(run_id)
        package = str(package_id)
        token = str(lease_token)
        if not token:
            return None
        now = _now()
        fence_time = time.time()
        with self._session.transaction() as connection:
            leased = connection.execute(
                """SELECT attempt, max_attempts FROM jobs
                   WHERE job_id=%s AND run_id=%s AND job_type='work_package'
                     AND package_id=%s AND status='running' AND lease_token=%s
                     AND lease_expires IS NOT NULL AND lease_expires>=%s""",
                (int(job_id), identifier, package, token, fence_time),
            ).fetchone()
            package_running = connection.execute(
                """SELECT 1 FROM work_packages
                   WHERE run_id=%s AND package_id=%s AND status='running'""",
                (identifier, package),
            ).fetchone()
            if leased is None or package_running is None:
                return None
            target = (
                "queued"
                if int(leased["attempt"]) < int(leased["max_attempts"])
                else "failed"
            )
            package_update = connection.execute(
                """UPDATE work_packages SET status=%s, updated_at=%s
                   WHERE run_id=%s AND package_id=%s AND status='running'""",
                (target, now, identifier, package),
            )
            job_update = connection.execute(
                """UPDATE jobs SET status=%s, error=%s, worker_id='',
                   lease_token='', lease_expires=NULL, heartbeat_at=%s, updated_at=%s
                   WHERE job_id=%s AND run_id=%s AND job_type='work_package'
                     AND package_id=%s AND status='running' AND lease_token=%s
                     AND lease_expires IS NOT NULL AND lease_expires>=%s""",
                (
                    target,
                    "" if target == "queued" else str(error),
                    now,
                    now,
                    int(job_id),
                    identifier,
                    package,
                    token,
                    fence_time,
                ),
            )
            if package_update.rowcount != 1 or job_update.rowcount != 1:
                raise RunStateError(
                    "Work Package/job state changed during atomic retry decision"
                )
            if target == "queued":
                # The job may be queued again; the attempt which just failed may not.
                # Preserve that distinction inside the SAME fenced business transaction.
                connection.execute(
                    """UPDATE monitor_spans SET status='failed',message=%s
                       WHERE span_id=(SELECT monitor_span_id FROM jobs WHERE job_id=%s)
                         AND run_id=%s AND status='queued'""",
                    (_bounded_utf8(error, 8000), int(job_id), identifier),
                )
                connection.execute(
                    """INSERT INTO monitor_events
                       (run_id,execution_id,span_id,timestamp,level,event_type,
                        object_type,object_id,stream_id,package_id,unit_id,job_id,
                        message,payload_json,idempotency_key)
                       SELECT s.run_id,s.execution_id,s.span_id,%s,'error','job_attempt_failed',
                              s.object_type,s.object_id,s.stream_id,s.package_id,s.unit_id,s.job_id,
                              %s,%s,'job:' || s.job_id::text || ':span:' || s.span_id || ':terminal:failed'
                       FROM monitor_spans s JOIN jobs j ON j.monitor_span_id=s.span_id
                       WHERE j.job_id=%s AND s.run_id=%s AND s.status='failed'
                       ON CONFLICT(run_id,idempotency_key) DO NOTHING""",
                    (
                        now,
                        _bounded_utf8(error, 8000),
                        _json({"retry_scheduled": True}),
                        int(job_id),
                        identifier,
                    ),
                )
            return target

    def interrupt_work_package_worker(
        self,
        run_id: str,
        worker_id: str,
    ) -> int:
        """Atomically interrupt every running Package owned by one worker.

        Worker IDs are unique to a runner instance. Jobs already re-leased to
        another worker do not match and their Package rows remain untouched.
        """
        identifier = str(run_id)
        worker = str(worker_id)
        if not worker:
            return 0
        now = _now()
        with self._session.transaction() as connection:
            leased = connection.execute(
                """SELECT job_id, package_id, lease_token FROM jobs
                   WHERE run_id=%s AND job_type='work_package'
                     AND status='running' AND worker_id=%s
                   ORDER BY job_id""",
                (identifier, worker),
            ).fetchall()
            for row in leased:
                package_update = connection.execute(
                    """UPDATE work_packages SET status='interrupted', updated_at=%s
                       WHERE run_id=%s AND package_id=%s AND status='running'
                         AND EXISTS (
                           SELECT 1 FROM jobs
                           WHERE job_id=%s AND run_id=%s
                             AND job_type='work_package' AND package_id=%s
                             AND status='running' AND worker_id=%s
                             AND lease_token=%s
                         )""",
                    (
                        now,
                        identifier,
                        str(row["package_id"]),
                        int(row["job_id"]),
                        identifier,
                        str(row["package_id"]),
                        worker,
                        str(row["lease_token"]),
                    ),
                )
                job_update = connection.execute(
                    """UPDATE jobs SET status='interrupted', worker_id='',
                       lease_token='', lease_expires=NULL, heartbeat_at=%s, updated_at=%s
                       , attempt=GREATEST(0, attempt-1)
                       WHERE job_id=%s AND run_id=%s AND job_type='work_package'
                         AND package_id=%s AND status='running' AND worker_id=%s
                         AND lease_token=%s""",
                    (
                        now,
                        now,
                        int(row["job_id"]),
                        identifier,
                        str(row["package_id"]),
                        worker,
                        str(row["lease_token"]),
                    ),
                )
                if package_update.rowcount != 1 or job_update.rowcount != 1:
                    raise RunStateError(
                        "Work Package/job state changed during worker interruption"
                    )
            return len(leased)

    def recover_ready_work_package_jobs(self, run_id: str) -> int:
        """Heal the legacy crash window where Package was ready before its job.

        Older workers committed the two rows separately.  A Package marked
        ready was written only after all formal Package outputs had committed,
        so its corresponding control-plane job can be finalized without
        rerunning models.  New workers do not create this state.
        """
        now = _now()
        with self._session.transaction() as connection:
            return int(
                connection.execute(
                    """UPDATE jobs SET status='ready', error='', worker_id='',
                   lease_token='', lease_expires=NULL, heartbeat_at=%s, updated_at=%s
                   WHERE run_id=%s AND job_type='work_package' AND status!='ready'
                     AND EXISTS (
                       SELECT 1 FROM work_packages wp
                       WHERE wp.run_id=jobs.run_id
                         AND wp.package_id=jobs.package_id
                         AND wp.status='ready'
                     )""",
                    (now, now, str(run_id)),
                ).rowcount
            )

    def interrupt_job(self, job_id: int, lease_token: str) -> bool:
        with self._session.transaction() as connection:
            return bool(
                connection.execute(
                    """UPDATE jobs SET status='interrupted', worker_id='', lease_token='',
                   lease_expires=NULL, updated_at=%s, attempt=GREATEST(0, attempt-1)
                   WHERE job_id IN (SELECT job_id FROM jobs WHERE job_id=%s
                     AND status='running' AND lease_token=%s
                     FOR UPDATE SKIP LOCKED)""",
                    (_now(), int(job_id), str(lease_token)),
                ).rowcount
                == 1
            )

    def requeue_failed_job(self, job_id: int) -> bool:
        with self._session.transaction() as connection:
            return bool(
                connection.execute(
                    """UPDATE jobs SET status='queued', error='', worker_id='',
                   lease_token='', lease_expires=NULL, updated_at=%s
                   WHERE job_id=%s AND status='failed' AND attempt < max_attempts""",
                    (_now(), int(job_id)),
                ).rowcount
                == 1
            )

    def interrupt_expired_jobs(
        self,
        *,
        run_id: str | None = None,
        now_epoch: float | None = None,
    ) -> int:
        """Recover expired leases, optionally limited to one Run."""

        now_value = time.time() if now_epoch is None else float(now_epoch)
        now = _now()
        identifier = str(run_id) if run_id is not None else None
        with self._session.transaction() as connection:
            if identifier is None:
                connection.execute(
                    """UPDATE work_packages SET status='interrupted', updated_at=%s
                       WHERE status='running' AND EXISTS (
                         SELECT 1 FROM jobs
                         WHERE jobs.run_id=work_packages.run_id
                           AND jobs.package_id=work_packages.package_id
                           AND jobs.job_type='work_package'
                           AND jobs.status='running'
                           AND jobs.lease_expires IS NOT NULL
                           AND jobs.lease_expires < %s
                       )""",
                    (now, now_value),
                )
                return int(
                    connection.execute(
                        """UPDATE jobs SET status='interrupted', worker_id='',
                       lease_token='', lease_expires=NULL, updated_at=%s,
                       attempt=GREATEST(0, attempt-1)
                       WHERE status='running' AND lease_expires IS NOT NULL
                       AND lease_expires < %s""",
                        (now, now_value),
                    ).rowcount
                )
            connection.execute(
                """UPDATE work_packages SET status='interrupted', updated_at=%s
                   WHERE run_id=%s AND status='running' AND EXISTS (
                     SELECT 1 FROM jobs
                     WHERE jobs.run_id=work_packages.run_id
                       AND jobs.package_id=work_packages.package_id
                       AND jobs.job_type='work_package'
                       AND jobs.status='running'
                       AND jobs.lease_expires IS NOT NULL
                       AND jobs.lease_expires < %s
                   )""",
                (now, identifier, now_value),
            )
            return int(
                connection.execute(
                    """UPDATE jobs SET status='interrupted', worker_id='',
                   lease_token='', lease_expires=NULL, updated_at=%s,
                   attempt=GREATEST(0, attempt-1)
                   WHERE run_id=%s AND status='running'
                   AND lease_expires IS NOT NULL AND lease_expires < %s""",
                    (now, identifier, now_value),
                ).rowcount
            )

    def interrupt_run_jobs(self, run_id: str) -> int:
        """Recover only the selected run after a QGIS/process interruption."""
        identifier = str(run_id)
        now = _now()
        with self._session.transaction() as connection:
            connection.execute(
                """UPDATE work_packages SET status='interrupted', updated_at=%s
                   WHERE run_id=%s AND status='running' AND EXISTS (
                     SELECT 1 FROM jobs
                     WHERE jobs.run_id=work_packages.run_id
                       AND jobs.package_id=work_packages.package_id
                       AND jobs.job_type='work_package'
                       AND jobs.status='running'
                   )""",
                (now, identifier),
            )
            return int(
                connection.execute(
                    """UPDATE jobs SET status='interrupted', worker_id='',
                   lease_token='', lease_expires=NULL, updated_at=%s,
                   attempt=GREATEST(0, attempt-1)
                   WHERE run_id=%s AND status='running'""",
                    (now, identifier),
                ).rowcount
            )

    def job_for_unit(
        self, run_id: str, stream_id: str, unit_id: str
    ) -> dict[str, Any] | None:
        with self._session.connection() as connection:
            return _row_dict(
                connection.execute(
                    """SELECT * FROM jobs WHERE run_id=%s AND stream_id=%s
                       AND unit_id=%s AND job_type='unit_fit'""",
                    (str(run_id), str(stream_id), str(unit_id)),
                ).fetchone()
            )

    def job_counts(
        self,
        run_id: str,
        *,
        stream_id: str = "",
        job_type: str = "",
    ) -> dict[str, int]:
        sql = "SELECT status, COUNT(*) AS n FROM jobs WHERE run_id=%s"
        values: list[Any] = [str(run_id)]
        if stream_id:
            sql += " AND stream_id=%s"
            values.append(str(stream_id))
        if job_type:
            sql += " AND job_type=%s"
            values.append(str(job_type))
        sql += " GROUP BY status"
        with self._session.connection() as connection:
            return {
                str(row["status"]): int(row["n"])
                for row in connection.execute(sql, values).fetchall()
            }
