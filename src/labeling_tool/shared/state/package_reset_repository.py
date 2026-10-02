"""Transactional reset of failed Work Packages in PostgreSQL Run state."""

from __future__ import annotations

from typing import Any, Sequence, TypedDict

from labeling_tool.shared.state.run_state_session import RunStateError, RunStateSession
from labeling_tool.shared.state.state_values import json_value as _json
from labeling_tool.shared.state.state_values import utc_now as _now

__all__ = [
    "PackageResetPlan",
    "PackageResetRepository",
    "PackageResetResult",
]


class PackageResetPlan(TypedDict):
    """Frozen database identities whose run-owned files may be removed."""

    run_id: str
    package_ids: list[str]
    partition_ids: list[str]
    affected_unit_ids: list[str]
    job_ids: list[int]
    artifacts: list[dict[str, Any]]


class PackageResetResult(TypedDict):
    """Counts committed after filesystem cleanup completes."""

    package_count: int
    affected_unit_count: int
    reset_job_count: int
    deleted_artifact_count: int
    relinked_dependency_count: int


class PackageResetRepository:
    """Own the two-phase Package reset transaction on one state session."""

    def __init__(self, session: RunStateSession) -> None:
        self._session = session

    def begin_failed_package_reset(self, run_id: str) -> PackageResetPlan:
        """Freeze and inventory every Package implicated by failed work.

        The filesystem is intentionally not touched inside the database
        transaction.  Callers delete only the returned, run-owned paths and
        then call :meth:`complete_failed_package_reset`.  If deletion is
        interrupted, calling this method again resumes the same ``resetting``
        Package set.
        """
        identifier = str(run_id)
        now = _now()
        with self._session.transaction() as connection:
            run = connection.execute(
                "SELECT status FROM runs WHERE run_id=%s", (identifier,)
            ).fetchone()
            if run is None:
                raise RunStateError(f"unknown Run: {identifier}")
            run_status = str(run["status"])
            if run_status not in {"failed", "stopped", "resetting"}:
                raise RunStateError(
                    "manual Package reset requires a failed, stopped, or "
                    f"resetting Run; got {run_status}"
                )
            running_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM jobs WHERE run_id=%s AND status='running'",
                    (identifier,),
                ).fetchone()[0]
            )
            if running_count:
                raise RunStateError(
                    "manual Package reset is blocked while Jobs are running"
                )

            connection.execute(
                "CREATE TEMP TABLE reset_packages(package_id TEXT PRIMARY KEY)"
            )
            if run_status == "resetting":
                # A resumed filesystem cleanup must retain the exact Package
                # set frozen by the first call.  Its downstream unit_fit Jobs
                # are also marked resetting, but using those units to discover
                # Packages again would incorrectly absorb ready neighbours of
                # a cross-Package Seam/Junction.
                connection.execute(
                    """INSERT INTO reset_packages(package_id)
                           SELECT package_id FROM work_packages
                           WHERE run_id=%s AND status='resetting' ON CONFLICT DO NOTHING""",
                    (identifier,),
                )
                connection.execute(
                    """INSERT INTO reset_packages(package_id)
                           SELECT package_id FROM jobs
                           WHERE run_id=%s AND job_type='work_package'
                             AND status='resetting' AND package_id!='' ON CONFLICT DO NOTHING""",
                    (identifier,),
                )
            else:
                connection.execute(
                    """INSERT INTO reset_packages(package_id)
                           SELECT package_id FROM work_packages
                           WHERE run_id=%s AND status='failed' ON CONFLICT DO NOTHING""",
                    (identifier,),
                )
                connection.execute(
                    """INSERT INTO reset_packages(package_id)
                           SELECT package_id FROM jobs
                           WHERE run_id=%s AND job_type='work_package'
                             AND status='failed' AND package_id!='' ON CONFLICT DO NOTHING""",
                    (identifier,),
                )
                connection.execute(
                    """INSERT INTO reset_packages(package_id)
                           SELECT DISTINCT p.package_id
                           FROM jobs j
                           JOIN unit_dependencies d
                             ON d.run_id=j.run_id AND d.unit_id=j.unit_id
                           JOIN partitions p
                             ON p.run_id=d.run_id AND p.partition_id=d.partition_id
                           WHERE j.run_id=%s AND j.job_type IN (
                             'unit_fit','unit_confidence','fragmentation_v33'
                           )
                             AND j.status='failed'
                             AND p.package_id IS NOT NULL ON CONFLICT DO NOTHING""",
                    (identifier,),
                )
            package_ids = [
                str(row["package_id"])
                for row in connection.execute(
                    """SELECT rp.package_id FROM reset_packages rp
                           JOIN work_packages wp
                             ON wp.run_id=%s AND wp.package_id=rp.package_id
                           ORDER BY wp.sequence_no""",
                    (identifier,),
                ).fetchall()
            ]
            if not package_ids:
                raise RunStateError(
                    "manual Package reset could not locate a failed Work Package"
                )

            connection.execute(
                """CREATE TEMP TABLE reset_units(unit_id TEXT PRIMARY KEY)"""
            )
            connection.execute(
                """INSERT INTO reset_units(unit_id)
                       SELECT DISTINCT d.unit_id
                       FROM unit_dependencies d
                       JOIN partitions p
                         ON p.run_id=d.run_id AND p.partition_id=d.partition_id
                       JOIN reset_packages rp ON rp.package_id=p.package_id
                       WHERE d.run_id=%s ON CONFLICT DO NOTHING""",
                (identifier,),
            )
            connection.execute(
                """CREATE TEMP TABLE reset_jobs(job_id INTEGER PRIMARY KEY)"""
            )
            connection.execute(
                """INSERT INTO reset_jobs(job_id)
                       SELECT j.job_id FROM jobs j
                       JOIN reset_packages rp ON rp.package_id=j.package_id
                       WHERE j.run_id=%s AND j.job_type='work_package' ON CONFLICT DO NOTHING""",
                (identifier,),
            )
            connection.execute(
                """INSERT INTO reset_jobs(job_id)
                       SELECT j.job_id FROM jobs j
                       JOIN reset_units ru ON ru.unit_id=j.unit_id
                       WHERE j.run_id=%s AND j.job_type IN (
                         'unit_fit','unit_confidence','fragmentation_v33'
                       ) ON CONFLICT DO NOTHING""",
                (identifier,),
            )

            connection.execute(
                """CREATE TEMP TABLE reset_v33_owners(
                         stream_id TEXT NOT NULL,
                         partition_id TEXT NOT NULL,
                         PRIMARY KEY(stream_id, partition_id)
                       )"""
            )
            connection.execute(
                """INSERT INTO reset_v33_owners(stream_id, partition_id)
                       SELECT DISTINCT j.stream_id, u.owner_key
                       FROM jobs j
                       JOIN reset_jobs rj ON rj.job_id=j.job_id
                       JOIN spatial_units u
                         ON u.run_id=j.run_id AND u.unit_id=j.unit_id
                       WHERE j.run_id=%s AND j.job_type='fragmentation_v33'
                         AND u.unit_type='FragmentationV33Partition'
                       ON CONFLICT DO NOTHING""",
                (identifier,),
            )

            connection.execute(
                """UPDATE work_packages SET status='resetting', updated_at=%s
                       WHERE run_id=%s AND package_id IN (
                         SELECT package_id FROM reset_packages
                       )""",
                (now, identifier),
            )
            connection.execute(
                """UPDATE jobs SET status='resetting', worker_id='',
                       lease_token='', lease_expires=NULL, updated_at=%s
                       WHERE run_id=%s AND job_id IN (SELECT job_id FROM reset_jobs)""",
                (now, identifier),
            )
            connection.execute(
                """DELETE FROM artifact_dependencies
                       WHERE job_id IN (SELECT job_id FROM reset_jobs)"""
            )
            connection.execute(
                """UPDATE artifacts SET status='resetting', updated_at=%s
                       WHERE run_id=%s AND (
                         unit_id IN ('assembled','mosaic')
                         OR EXISTS (
                           SELECT 1 FROM partitions p
                           JOIN reset_packages rp ON rp.package_id=p.package_id
                           WHERE p.run_id=artifacts.run_id
                             AND p.partition_id=artifacts.unit_id
                         )
                         OR EXISTS (
                           SELECT 1 FROM reset_units ru
                           WHERE ru.unit_id=artifacts.unit_id
                         )
                         OR EXISTS (
                           SELECT 1 FROM reset_v33_owners rv
                           WHERE rv.stream_id=artifacts.stream_id
                             AND rv.partition_id=artifacts.unit_id
                             AND artifacts.kind IN (
                               'v33_staged_mask','v33_staged_audit',
                               'core_mask','fragmentation_v33_audit'
                             )
                         )
                       )""",
                (now, identifier),
            )
            connection.execute(
                """UPDATE runs SET status='resetting', updated_at=%s
                       WHERE run_id=%s""",
                (now, identifier),
            )

            partition_ids = [
                str(row["partition_id"])
                for row in connection.execute(
                    """SELECT p.partition_id FROM partitions p
                           JOIN reset_packages rp ON rp.package_id=p.package_id
                           WHERE p.run_id=%s ORDER BY p.row_no, p.col_no""",
                    (identifier,),
                ).fetchall()
            ]
            unit_ids = [
                str(row["unit_id"])
                for row in connection.execute(
                    "SELECT unit_id FROM reset_units ORDER BY unit_id"
                ).fetchall()
            ]
            job_ids = [
                int(row["job_id"])
                for row in connection.execute(
                    "SELECT job_id FROM reset_jobs ORDER BY job_id"
                ).fetchall()
            ]
            artifacts = [
                dict(row)
                for row in connection.execute(
                    """SELECT * FROM artifacts
                           WHERE run_id=%s AND status='resetting'
                           ORDER BY artifact_id""",
                    (identifier,),
                ).fetchall()
            ]
            connection.execute(
                """INSERT INTO events
                       (run_id, timestamp, level, event_type, message, payload_json)
                       VALUES (%s, %s, 'warning', 'manual_package_reset_started', %s, %s)""",
                (
                    identifier,
                    now,
                    ",".join(package_ids),
                    _json(
                        {
                            "package_ids": package_ids,
                            "partition_count": len(partition_ids),
                            "affected_unit_count": len(unit_ids),
                            "reset_job_count": len(job_ids),
                            "artifact_count": len(artifacts),
                            "tile_cache_action": "preserved",
                        }
                    ),
                ),
            )
        return {
            "run_id": identifier,
            "package_ids": package_ids,
            "partition_ids": partition_ids,
            "affected_unit_ids": unit_ids,
            "job_ids": job_ids,
            "artifacts": artifacts,
        }

    def complete_failed_package_reset(
        self,
        run_id: str,
        package_ids: Sequence[str],
    ) -> PackageResetResult:
        """Commit a completed filesystem cleanup and rebuild ready inputs."""
        identifier = str(run_id)
        expected_packages = tuple(str(item) for item in package_ids)
        if not expected_packages:
            raise RunStateError("manual Package reset requires at least one Package")
        now = _now()
        with self._session.transaction() as connection:
            run = connection.execute(
                "SELECT status FROM runs WHERE run_id=%s", (identifier,)
            ).fetchone()
            if run is None or str(run["status"]) != "resetting":
                raise RunStateError(
                    "manual Package reset completion requires Run status resetting"
                )
            connection.execute(
                "CREATE TEMP TABLE reset_packages(package_id TEXT PRIMARY KEY)"
            )
            connection.executemany(
                "INSERT INTO reset_packages(package_id) VALUES (%s)",
                ((item,) for item in expected_packages),
            )
            actual_packages = {
                str(row["package_id"])
                for row in connection.execute(
                    """SELECT wp.package_id FROM work_packages wp
                           JOIN reset_packages rp ON rp.package_id=wp.package_id
                           WHERE wp.run_id=%s AND wp.status='resetting'""",
                    (identifier,),
                ).fetchall()
            }
            if actual_packages != set(expected_packages):
                raise RunStateError(
                    "manual Package reset Package set changed before completion"
                )
            running_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM jobs WHERE run_id=%s AND status='running'",
                    (identifier,),
                ).fetchone()[0]
            )
            if running_count:
                raise RunStateError(
                    "manual Package reset completion is blocked by running Jobs"
                )

            connection.execute(
                """CREATE TEMP TABLE reset_units(unit_id TEXT PRIMARY KEY)"""
            )
            connection.execute(
                """INSERT INTO reset_units(unit_id)
                       SELECT DISTINCT d.unit_id
                       FROM unit_dependencies d
                       JOIN partitions p
                         ON p.run_id=d.run_id AND p.partition_id=d.partition_id
                       JOIN reset_packages rp ON rp.package_id=p.package_id
                       WHERE d.run_id=%s ON CONFLICT DO NOTHING""",
                (identifier,),
            )
            connection.execute(
                """CREATE TEMP TABLE reset_jobs(job_id INTEGER PRIMARY KEY)"""
            )
            connection.execute(
                """INSERT INTO reset_jobs(job_id)
                       SELECT j.job_id FROM jobs j
                       JOIN reset_packages rp ON rp.package_id=j.package_id
                       WHERE j.run_id=%s AND j.job_type='work_package' ON CONFLICT DO NOTHING""",
                (identifier,),
            )
            connection.execute(
                """INSERT INTO reset_jobs(job_id)
                       SELECT j.job_id FROM jobs j
                       JOIN reset_units ru ON ru.unit_id=j.unit_id
                       WHERE j.run_id=%s AND j.job_type IN (
                         'unit_fit','unit_confidence','fragmentation_v33'
                       ) ON CONFLICT DO NOTHING""",
                (identifier,),
            )
            unit_count = int(
                connection.execute("SELECT COUNT(*) FROM reset_units").fetchone()[0]
            )
            job_count = int(
                connection.execute("SELECT COUNT(*) FROM reset_jobs").fetchone()[0]
            )
            artifact_count = int(
                connection.execute(
                    """SELECT COUNT(*) FROM artifacts
                           WHERE run_id=%s AND status='resetting'""",
                    (identifier,),
                ).fetchone()[0]
            )

            connection.execute(
                """DELETE FROM artifact_dependencies
                       WHERE job_id IN (SELECT job_id FROM reset_jobs)"""
            )
            connection.execute(
                """DELETE FROM unit_report_summaries
                       WHERE run_id=%s AND unit_id IN (
                         SELECT unit_id FROM reset_units
                       )""",
                (identifier,),
            )
            connection.execute(
                "DELETE FROM object_links WHERE run_id=%s", (identifier,)
            )
            connection.execute(
                "DELETE FROM object_nodes WHERE run_id=%s", (identifier,)
            )
            connection.execute(
                "DELETE FROM artifacts WHERE run_id=%s AND status='resetting'",
                (identifier,),
            )
            connection.execute(
                """UPDATE partitions SET status='queued', updated_at=%s
                       WHERE run_id=%s AND package_id IN (
                         SELECT package_id FROM reset_packages
                       )""",
                (now, identifier),
            )
            connection.execute(
                """UPDATE spatial_units SET status='queued', updated_at=%s
                       WHERE run_id=%s AND unit_id IN (
                         SELECT unit_id FROM reset_units
                       )""",
                (now, identifier),
            )
            connection.execute(
                """UPDATE stream_units SET status='queued', error='', updated_at=%s
                       WHERE run_id=%s AND unit_id IN (
                         SELECT unit_id FROM reset_units
                       )""",
                (now, identifier),
            )
            connection.execute(
                """UPDATE work_packages SET status='queued', attempt=0,
                       updated_at=%s WHERE run_id=%s AND package_id IN (
                         SELECT package_id FROM reset_packages
                       )""",
                (now, identifier),
            )
            connection.execute(
                """UPDATE jobs SET status='queued', attempt=0,
                       progress_current=0, progress_total=0, worker_id='',
                       lease_token='', lease_expires=NULL, heartbeat_at=NULL,
                       pid=NULL, error='', updated_at=%s
                       WHERE run_id=%s AND job_id IN (SELECT job_id FROM reset_jobs)""",
                (now, identifier),
            )
            connection.execute(
                """UPDATE streams SET status='pending', error='', updated_at=%s
                       WHERE run_id=%s""",
                (now, identifier),
            )
            relinked = connection.execute(
                """INSERT INTO artifact_dependencies
                       (job_id, artifact_id, created_at)
                       SELECT j.job_id, a.artifact_id, %s
                       FROM jobs j
                       JOIN reset_jobs rj ON rj.job_id=j.job_id
                       JOIN unit_dependencies d
                         ON d.run_id=j.run_id AND d.unit_id=j.unit_id
                       JOIN artifacts a
                         ON a.run_id=j.run_id
                        AND a.stream_id=j.stream_id
                        AND a.unit_id=d.partition_id
                        AND a.kind='partition_probability'
                        AND a.status='ready'
                       WHERE j.run_id=%s AND (
                         j.job_type='unit_confidence'
                         OR (
                           j.job_type='unit_fit'
                           AND NOT EXISTS (
                             SELECT 1 FROM jobs compact
                             WHERE compact.run_id=j.run_id
                               AND compact.stream_id=j.stream_id
                               AND compact.unit_id=j.unit_id
                               AND compact.job_type='unit_confidence'
                           )
                         )
                       ) ON CONFLICT DO NOTHING""",
                (now, identifier),
            ).rowcount
            relinked += connection.execute(
                """INSERT INTO artifact_dependencies
                       (job_id, artifact_id, created_at)
                       SELECT j.job_id, a.artifact_id, %s
                       FROM jobs j
                       JOIN reset_jobs rj ON rj.job_id=j.job_id
                       JOIN spatial_units u
                         ON u.run_id=j.run_id AND u.unit_id=j.unit_id
                       JOIN unit_dependencies d
                         ON d.run_id=j.run_id AND d.unit_id=j.unit_id
                       JOIN artifacts a
                         ON a.run_id=j.run_id
                        AND a.stream_id=j.stream_id
                        AND a.unit_id=d.partition_id
                        AND a.kind IN (
                          'partition_probability','v3_context_core','v3_baseline_core'
                        )
                        AND a.status='ready'
                       WHERE j.run_id=%s AND j.job_type='fragmentation_v33'
                         AND u.unit_type='FragmentationV33Partition'
                       ON CONFLICT DO NOTHING""",
                (now, identifier),
            ).rowcount
            relinked += connection.execute(
                """INSERT INTO artifact_dependencies
                       (job_id, artifact_id, created_at)
                       SELECT j.job_id, a.artifact_id, %s
                       FROM jobs j
                       JOIN reset_jobs rj ON rj.job_id=j.job_id
                       JOIN spatial_units u
                         ON u.run_id=j.run_id AND u.unit_id=j.unit_id
                       JOIN artifacts a
                         ON a.run_id=j.run_id
                        AND a.stream_id=j.stream_id
                        AND a.kind IN ('v33_staged_mask','v33_staged_audit')
                        AND a.status='ready'
                       WHERE j.run_id=%s AND j.job_type='fragmentation_v33'
                         AND u.unit_type='FragmentationV33Finalize'
                       ON CONFLICT DO NOTHING""",
                (now, identifier),
            ).rowcount
            connection.execute(
                """UPDATE runs SET status='planned', updated_at=%s
                       WHERE run_id=%s AND status='resetting'""",
                (now, identifier),
            )
            connection.execute(
                """INSERT INTO events
                       (run_id, timestamp, level, event_type, message, payload_json)
                       VALUES (%s, %s, 'info', 'manual_package_reset_completed', %s, %s)""",
                (
                    identifier,
                    now,
                    ",".join(expected_packages),
                    _json(
                        {
                            "package_ids": list(expected_packages),
                            "affected_unit_count": unit_count,
                            "reset_job_count": job_count,
                            "deleted_artifact_count": artifact_count,
                            "relinked_dependency_count": int(relinked),
                            "tile_cache_action": "preserved",
                        }
                    ),
                ),
            )
        return {
            "package_count": len(expected_packages),
            "affected_unit_count": unit_count,
            "reset_job_count": job_count,
            "deleted_artifact_count": artifact_count,
            "relinked_dependency_count": int(relinked),
        }
