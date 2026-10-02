"""Cross-domain read models for the persisted Run monitor."""

from __future__ import annotations

import json
from typing import Any

from labeling_tool.shared.contracts.monitor_contract import (
    MONITOR_DETAIL_PAGE_LIMIT,
    MONITOR_HISTORY_VERSION,
)
from labeling_tool.shared.state.monitor_history_repository import (
    MonitorHistoryRepository,
)
from labeling_tool.shared.state.run_state_session import RunStateSession
from labeling_tool.shared.state.state_values import row_dict, utc_now

__all__ = ["MonitorReadRepository"]


class MonitorReadRepository:
    """Read bounded monitor projections without owning mutable Run state."""

    def __init__(
        self,
        session: RunStateSession,
        monitor_history: MonitorHistoryRepository,
    ) -> None:
        self._session = session
        self._monitor_history = monitor_history

    def count_objects(
        self,
        run_id: str,
        *,
        kind: str,
        stream_id: str = "",
        status: str = "",
        search: str = "",
    ) -> int:
        """Count one monitor detail category without loading its rows."""

        category = str(kind)
        if category == "package":
            sql = "SELECT COUNT(*) FROM work_packages WHERE run_id=%s"
            values: list[Any] = [str(run_id)]
            status_column = "status"
            search_column = "package_id"
        elif category in {"fragmentation_v33", "unit_confidence", "unit_fit"}:
            sql = "SELECT COUNT(*) FROM jobs WHERE run_id=%s AND job_type=%s"
            values = [str(run_id), category]
            status_column = "status"
            search_column = "unit_id"
            if stream_id:
                sql += " AND stream_id=%s"
                values.append(str(stream_id))
        else:
            raise ValueError(f"unknown monitor detail category: {kind}")
        if status:
            sql += f" AND {status_column}=%s"
            values.append(str(status))
        if search:
            escaped = (
                str(search)
                .replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            sql += f" AND {search_column} LIKE %s ESCAPE '\\'"
            values.append(f"%{escaped}%")
        with self._session.connection() as connection:
            return int(connection.execute(sql, values).fetchone()[0])

    def page_objects(
        self,
        run_id: str,
        *,
        kind: str,
        stream_id: str = "",
        status: str = "",
        search: str = "",
        limit: int = 500,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Page packages or spatial jobs with execution and artifact states."""

        category = str(kind)
        values: list[Any]
        if category == "package":
            sql = """SELECT wp.package_id AS object_id,
                            'Work Package' AS object_label,
                            wp.sequence_no,wp.status AS execution_status,
                            wp.status AS artifact_status,
                            COALESCE(j.error,'') AS reason,
                            COALESCE(j.progress_current,0) AS progress_current,
                            COALESCE(j.progress_total,0) AS progress_total,
                            COALESCE(j.attempt,wp.attempt,0) AS budget_attempt,
                            COALESCE(j.monitor_execution_id,'') AS execution_id,
                            COALESCE(j.monitor_span_id,'') AS span_id,j.job_id,
                            COALESCE(j.stream_id,'') AS stream_id,
                            wp.updated_at
                       FROM work_packages wp
                       LEFT JOIN jobs j ON j.run_id=wp.run_id
                         AND j.package_id=wp.package_id
                         AND j.job_type='work_package'
                       WHERE wp.run_id=%s"""
            values = [str(run_id)]
            if status:
                sql += " AND wp.status=%s"
                values.append(str(status))
            if search:
                escaped = (
                    str(search)
                    .replace("\\", "\\\\")
                    .replace("%", "\\%")
                    .replace("_", "\\_")
                )
                sql += " AND wp.package_id LIKE %s ESCAPE '\\'"
                values.append(f"%{escaped}%")
            sql += " ORDER BY wp.sequence_no"
        elif category in {"fragmentation_v33", "unit_confidence", "unit_fit"}:
            sql = """SELECT j.unit_id AS object_id,
                            COALESCE(u.unit_type,j.job_type) AS object_label,
                            0 AS sequence_no,j.status AS execution_status,
                            COALESCE(su.status,'') AS artifact_status,
                            CASE WHEN j.error<>'' THEN j.error
                                 WHEN COALESCE(su.error,'')<>'' THEN su.error
                                 WHEN j.status IN ('queued','interrupted','resetting')
                                   THEN '原因尚未确认'
                                 ELSE '' END AS reason,
                            j.progress_current,j.progress_total,
                            j.attempt AS budget_attempt,
                            j.monitor_execution_id AS execution_id,
                            j.monitor_span_id AS span_id,j.job_id,j.stream_id,j.updated_at
                       FROM jobs j
                       LEFT JOIN spatial_units u ON u.run_id=j.run_id
                         AND u.unit_id=j.unit_id
                       LEFT JOIN stream_units su ON su.run_id=j.run_id
                         AND su.stream_id=j.stream_id AND su.unit_id=j.unit_id
                       WHERE j.run_id=%s AND j.job_type=%s"""
            values = [str(run_id), category]
            if stream_id:
                sql += " AND j.stream_id=%s"
                values.append(str(stream_id))
            if status:
                sql += " AND j.status=%s"
                values.append(str(status))
            if search:
                escaped = (
                    str(search)
                    .replace("\\", "\\\\")
                    .replace("%", "\\%")
                    .replace("_", "\\_")
                )
                sql += " AND j.unit_id LIKE %s ESCAPE '\\'"
                values.append(f"%{escaped}%")
            sql += " ORDER BY u.unit_type,j.unit_id,j.job_id"
        else:
            raise ValueError(f"unknown monitor detail category: {kind}")
        sql += " LIMIT %s OFFSET %s"
        values.extend(
            (
                max(1, min(int(limit), MONITOR_DETAIL_PAGE_LIMIT)),
                max(0, int(offset)),
            )
        )
        with self._session.connection() as connection:
            return [dict(row) for row in connection.execute(sql, values).fetchall()]

    def snapshot(self, run_id: str) -> dict[str, Any]:
        """Read the complete bounded monitor summary through one connection."""

        identifier = str(run_id)
        with self._session.connection() as connection:
            # Keep every aggregate in one read transaction. Without an
            # explicit repeatable-read snapshot, different commits may be
            # exposed to individual SELECT statements in one polling cycle.
            connection.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
            run = row_dict(
                connection.execute(
                    "SELECT * FROM runs WHERE run_id=%s", (identifier,)
                ).fetchone()
            )
            monitored_job_types = (
                "work_package",
                "fragmentation_v33",
                "unit_confidence",
                "unit_fit",
            )
            job_counts: dict[str, dict[str, int]] = {
                job_type: {} for job_type in monitored_job_types
            }
            for row in connection.execute(
                """SELECT job_type, status, COUNT(*) AS n FROM jobs
                   WHERE run_id=%s AND job_type IN (
                     'work_package','fragmentation_v33','unit_confidence','unit_fit'
                   )
                   GROUP BY job_type, status""",
                (identifier,),
            ).fetchall():
                job_counts[str(row["job_type"])][str(row["status"])] = int(row["n"])
            job_progress = {
                job_type: {"completed": 0.0, "total": 0}
                for job_type in monitored_job_types
            }
            for row in connection.execute(
                """SELECT job_type, COUNT(*) AS total,
                          SUM(CASE
                            WHEN status='ready' THEN 1.0
                            WHEN status='running' AND progress_total>0 THEN
                              CASE WHEN progress_current>=progress_total THEN 1.0
                                   ELSE CAST(progress_current AS REAL)
                                        / CAST(progress_total AS REAL) END
                            ELSE 0.0 END) AS completed
                   FROM jobs WHERE run_id=%s AND job_type IN (
                     'work_package','fragmentation_v33','unit_confidence','unit_fit'
                   ) GROUP BY job_type""",
                (identifier,),
            ).fetchall():
                job_progress[str(row["job_type"])] = {
                    "completed": float(row["completed"] or 0.0),
                    "total": int(row["total"] or 0),
                }
            active_package = row_dict(
                connection.execute(
                    """SELECT j.*, wp.sequence_no,
                              wp.updated_at AS package_started_at
                       FROM jobs j
                       JOIN work_packages wp
                         ON wp.run_id=j.run_id AND wp.package_id=j.package_id
                       WHERE j.run_id=%s AND j.job_type='work_package'
                         AND j.status='running' AND wp.status='running'
                       ORDER BY wp.sequence_no, j.job_id LIMIT 1""",
                    (identifier,),
                ).fetchone()
            )
            streams = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM streams WHERE run_id=%s ORDER BY stream_id",
                    (identifier,),
                ).fetchall()
            ]
            stream_unit_type_counts: dict[str, dict[str, dict[str, int]]] = {}
            for row in connection.execute(
                """SELECT su.stream_id, u.unit_type, su.status, COUNT(*) AS n
                   FROM stream_units su
                   JOIN spatial_units u
                     ON u.run_id=su.run_id AND u.unit_id=su.unit_id
                   WHERE su.run_id=%s
                   GROUP BY su.stream_id, u.unit_type, su.status""",
                (identifier,),
            ).fetchall():
                stream_unit_type_counts.setdefault(
                    str(row["stream_id"]), {}
                ).setdefault(str(row["unit_type"]), {})[str(row["status"])] = int(
                    row["n"]
                )
            stream_unit_job_type_counts: dict[str, dict[str, dict[str, int]]] = {}
            for row in connection.execute(
                """SELECT j.stream_id, u.unit_type, j.status, COUNT(*) AS n
                   FROM jobs j
                   JOIN spatial_units u
                     ON u.run_id=j.run_id AND u.unit_id=j.unit_id
                   WHERE j.run_id=%s AND j.job_type='unit_fit'
                   GROUP BY j.stream_id, u.unit_type, j.status""",
                (identifier,),
            ).fetchall():
                stream_unit_job_type_counts.setdefault(
                    str(row["stream_id"]), {}
                ).setdefault(str(row["unit_type"]), {})[str(row["status"])] = int(
                    row["n"]
                )
            # Historical schema-v2 Runs created before structured assembly
            # monitoring do not have this additive table. Keep them readable
            # with coarse Stream status instead of breaking the entire panel.
            try:
                stream_runtime_progress = {
                    str(row["stream_id"]): dict(row)
                    for row in connection.execute(
                        """SELECT * FROM stream_runtime_progress
                           WHERE run_id=%s ORDER BY stream_id""",
                        (identifier,),
                    ).fetchall()
                }
            except Exception:
                stream_runtime_progress = {}
            try:
                stream_coverage_validation = {}
                coverage_rows = connection.execute(
                    """SELECT e.stream_id, e.payload_json
                       FROM events e
                       WHERE e.run_id=%s
                         AND e.event_type='stream_coverage_validation'
                         AND e.event_id=(
                           SELECT MAX(latest.event_id) FROM events latest
                           WHERE latest.run_id=e.run_id
                             AND latest.stream_id=e.stream_id
                             AND latest.event_type=e.event_type
                         )
                       ORDER BY e.stream_id""",
                    (identifier,),
                ).fetchall()
                for row in coverage_rows:
                    try:
                        payload = json.loads(str(row["payload_json"] or "{}"))
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                    if isinstance(payload, dict):
                        stream_coverage_validation[str(row["stream_id"])] = payload
            except Exception:
                stream_coverage_validation = {}
            assembly_phase_statuses, monitor_history = (
                self._monitor_history.read_snapshot_sections(
                    connection,
                    identifier,
                    run,
                )
            )
        return {
            "run": run,
            "job_counts": job_counts,
            "job_progress": job_progress,
            "active_work_package": active_package,
            "streams": streams,
            "stream_runtime_progress": stream_runtime_progress,
            "stream_coverage_validation": stream_coverage_validation,
            "stream_unit_type_counts": stream_unit_type_counts,
            "stream_unit_job_type_counts": stream_unit_job_type_counts,
            "monitor_history": monitor_history,
            "monitor_data_version": MONITOR_HISTORY_VERSION,
            "snapshot_at": utc_now(),
            "assembly_phase_statuses": assembly_phase_statuses,
        }
