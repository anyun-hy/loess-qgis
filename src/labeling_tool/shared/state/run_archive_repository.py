"""Archival of incomplete Run database details without filesystem deletion."""

from __future__ import annotations

import json
import time
from typing import Any, TypedDict

from labeling_tool.shared.contracts.monitor_contract import MONITOR_HISTORY_VERSION
from labeling_tool.shared.state.run_state_session import RunStateError, RunStateSession
from labeling_tool.shared.state.state_values import bounded_utf8 as _bounded_utf8
from labeling_tool.shared.state.state_values import json_value as _json
from labeling_tool.shared.state.state_values import utc_now as _now

__all__ = [
    "ARCHIVABLE_INCOMPLETE_RUN_STATES",
    "RUN_ARCHIVE_REPORT_ID_LIMIT",
    "RUN_DETAIL_ARCHIVE_KEY",
    "RUN_DETAIL_TABLES",
    "RunArchiveReport",
    "RunArchiveRepository",
]

ARCHIVABLE_INCOMPLETE_RUN_STATES = ("failed", "stopped")
RUN_DETAIL_ARCHIVE_KEY = "database_detail_archive"
RUN_ARCHIVE_REPORT_ID_LIMIT = 50
RUN_DETAIL_TABLES = (
    "monitor_events",
    "monitor_spans",
    "monitor_executions",
    "streams",
    "stream_runtime_progress",
    "work_packages",
    "partitions",
    "tiles",
    "spatial_units",
    "unit_dependencies",
    "stream_units",
    "jobs",
    "artifacts",
    "artifact_dependencies",
    "unit_report_summaries",
    "object_links",
    "object_nodes",
    "events",
)


class RunArchiveReport(TypedDict):
    """Bounded summary of incomplete Run detail archival."""

    schema_version: int
    status: str
    protected_run_id: str
    archived_run_ids: list[str]
    archived_run_count: int
    archived_run_ids_truncated: bool
    skipped_active_run_ids: list[str]
    skipped_active_run_count: int
    skipped_active_run_ids_truncated: bool
    deleted_detail_counts: dict[str, int]


class RunArchiveRepository:
    """Own incomplete Run archival on one state session and transaction."""

    def __init__(self, session: RunStateSession) -> None:
        self._session = session

    def archive_incomplete_run_details(
        self,
        *,
        protected_run_id: str,
        reason: str = "new_run_housekeeping",
    ) -> RunArchiveReport:
        """Archive terminal incomplete Runs and remove their process details.

        ``failed`` and ``stopped`` are normally recoverable.  Explicitly
        starting a new Run makes older terminal attempts non-resumable, but it
        must not erase the small amount of evidence needed to explain them.
        Keep one immutable Run tombstone with bounded diagnostics while
        deleting only database control-plane details.  Filesystem Run outputs
        are deliberately outside this transaction and are never removed here.
        """

        protected = str(protected_run_id).strip()
        if not protected:
            raise RunStateError("protected Run ID is required for housekeeping")
        archived_at = _now()
        lease_now = time.time()
        archived_run_ids: list[str] = []
        skipped_active_run_ids: list[str] = []
        deleted_totals = {table: 0 for table in RUN_DETAIL_TABLES}

        with self._session.transaction() as connection:
            placeholders = ",".join("%s" for _state in ARCHIVABLE_INCOMPLETE_RUN_STATES)
            candidate_sql = (
                "SELECT run_id, status, run_spec_sha256, metadata_json, "
                "created_at, updated_at FROM runs "
                f"WHERE status IN ({placeholders}) AND run_id<>%s "
                "ORDER BY created_at, run_id"
            )
            candidate_sql += " FOR UPDATE SKIP LOCKED"
            candidates = connection.execute(
                candidate_sql,
                (*ARCHIVABLE_INCOMPLETE_RUN_STATES, protected),
            ).fetchall()

            for candidate in candidates:
                run_id = str(candidate["run_id"])
                active_count = int(
                    connection.execute(
                        """SELECT COUNT(*) FROM jobs
                               WHERE run_id=%s AND (
                                 status=%s OR (
                                   lease_token<>%s AND lease_expires IS NOT NULL
                                   AND lease_expires>%s
                                 )
                               )""",
                        (run_id, "running", "", lease_now),
                    ).fetchone()[0]
                )
                if active_count:
                    skipped_active_run_ids.append(run_id)
                    continue

                original_metadata_invalid = False
                try:
                    original_metadata = json.loads(
                        str(candidate["metadata_json"] or "{}")
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    original_metadata = {}
                    original_metadata_invalid = True
                if not isinstance(original_metadata, dict):
                    original_metadata = {}
                    original_metadata_invalid = True
                preserved_metadata: dict[str, Any] = {}
                run_spec_path = original_metadata.get("run_spec")
                if run_spec_path:
                    preserved_metadata["run_spec"] = _bounded_utf8(run_spec_path, 4096)
                for key in ("tile_count", "partition_count", "package_count"):
                    value = original_metadata.get(key)
                    if isinstance(value, int) and not isinstance(value, bool):
                        preserved_metadata[key] = value

                detail_counts: dict[str, int] = {}
                for table in RUN_DETAIL_TABLES:
                    if table == "artifact_dependencies":
                        row = connection.execute(
                            """SELECT COUNT(*) FROM artifact_dependencies
                                   WHERE job_id IN (
                                     SELECT job_id FROM jobs WHERE run_id=%s
                                   ) OR artifact_id IN (
                                     SELECT artifact_id FROM artifacts WHERE run_id=%s
                                   )""",
                            (run_id, run_id),
                        ).fetchone()
                    else:
                        row = connection.execute(
                            f"SELECT COUNT(*) FROM {table} WHERE run_id=%s",
                            (run_id,),
                        ).fetchone()
                    detail_counts[table] = int(row[0])

                job_status_counts = {
                    str(row["status"]): int(row["n"])
                    for row in connection.execute(
                        """SELECT status, COUNT(*) AS n FROM jobs
                               WHERE run_id=%s GROUP BY status ORDER BY status""",
                        (run_id,),
                    ).fetchall()
                }
                artifact_status_counts = {
                    str(row["status"]): int(row["n"])
                    for row in connection.execute(
                        """SELECT status, COUNT(*) AS n FROM artifacts
                               WHERE run_id=%s GROUP BY status ORDER BY status""",
                        (run_id,),
                    ).fetchall()
                }
                monitor_execution_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM monitor_executions WHERE run_id=%s",
                        (run_id,),
                    ).fetchone()[0]
                )
                monitor_span_status_counts = {
                    str(row["status"]): int(row["n"])
                    for row in connection.execute(
                        """SELECT status,COUNT(*) AS n FROM monitor_spans
                               WHERE run_id=%s GROUP BY status ORDER BY status""",
                        (run_id,),
                    ).fetchall()
                }
                monitor_attempt_count = sum(monitor_span_status_counts.values())
                monitor_failed_count = int(monitor_span_status_counts.get("failed", 0))
                monitor_recovered_count = int(
                    connection.execute(
                        """SELECT COUNT(*) FROM monitor_events
                               WHERE run_id=%s AND recovered_by_span_id IS NOT NULL""",
                        (run_id,),
                    ).fetchone()[0]
                )
                monitor_history_summary = {
                    "history_version": MONITOR_HISTORY_VERSION,
                    "details_archived": True,
                    "execution_count": monitor_execution_count,
                    "attempt_count": monitor_attempt_count,
                    "failed_count": monitor_failed_count,
                    "recovered_count": monitor_recovered_count,
                    "span_status_counts": monitor_span_status_counts,
                    "final_status": str(candidate["status"]),
                }
                error_record_count = sum(
                    int(connection.execute(sql, (run_id, "")).fetchone()[0])
                    for sql in (
                        "SELECT COUNT(*) FROM jobs WHERE run_id=%s AND error<>%s",
                        "SELECT COUNT(*) FROM streams WHERE run_id=%s AND error<>%s",
                        """SELECT COUNT(*) FROM events WHERE run_id=%s
                               AND level IN ('warning','error') AND message<>%s""",
                    )
                )
                errors: list[dict[str, Any]] = []
                for row in connection.execute(
                    """SELECT job_type, stream_id, unit_id, package_id,
                                  status, error, updated_at
                           FROM jobs WHERE run_id=%s AND error<>%s
                           ORDER BY updated_at DESC, job_id DESC LIMIT 8""",
                    (run_id, ""),
                ).fetchall():
                    errors.append(
                        {
                            "source": "job",
                            "job_type": str(row["job_type"]),
                            "stream_id": str(row["stream_id"] or ""),
                            "unit_id": str(row["unit_id"] or ""),
                            "package_id": str(row["package_id"] or ""),
                            "status": str(row["status"]),
                            "message": _bounded_utf8(row["error"], 2000),
                            "timestamp": str(row["updated_at"] or ""),
                        }
                    )
                for row in connection.execute(
                    """SELECT stream_id, status, error, updated_at
                           FROM streams WHERE run_id=%s AND error<>%s
                           ORDER BY updated_at DESC, stream_id LIMIT 8""",
                    (run_id, ""),
                ).fetchall():
                    errors.append(
                        {
                            "source": "stream",
                            "stream_id": str(row["stream_id"]),
                            "status": str(row["status"]),
                            "message": _bounded_utf8(row["error"], 2000),
                            "timestamp": str(row["updated_at"] or ""),
                        }
                    )
                for row in connection.execute(
                    """SELECT level, event_type, stream_id, message, timestamp
                           FROM events WHERE run_id=%s AND level IN (%s,%s)
                             AND message<>%s
                           ORDER BY event_id DESC LIMIT 8""",
                    (run_id, "warning", "error", ""),
                ).fetchall():
                    errors.append(
                        {
                            "source": "event",
                            "level": str(row["level"]),
                            "event_type": str(row["event_type"]),
                            "stream_id": str(row["stream_id"] or ""),
                            "message": _bounded_utf8(row["message"], 2000),
                            "timestamp": str(row["timestamp"] or ""),
                        }
                    )
                errors.sort(
                    key=lambda item: str(item.get("timestamp") or ""),
                    reverse=True,
                )
                errors = errors[:8]

                archive = {
                    "schema_version": 1,
                    "status": "archived",
                    "non_resumable": True,
                    "reason": _bounded_utf8(reason, 256),
                    "original_status": str(candidate["status"]),
                    "run_spec_sha256": str(candidate["run_spec_sha256"]),
                    "created_at": str(candidate["created_at"]),
                    "last_updated_at": str(candidate["updated_at"]),
                    "archived_at": archived_at,
                    "detail_counts": detail_counts,
                    "job_status_counts": job_status_counts,
                    "artifact_status_counts": artifact_status_counts,
                    "errors": errors,
                    "error_record_count": error_record_count,
                    "errors_truncated": error_record_count > len(errors),
                    "original_metadata_invalid": original_metadata_invalid,
                    "monitor_history_summary": monitor_history_summary,
                }
                metadata = {
                    **preserved_metadata,
                    RUN_DETAIL_ARCHIVE_KEY: archive,
                }

                deleted_totals["artifact_dependencies"] += connection.execute(
                    """DELETE FROM artifact_dependencies
                           WHERE job_id IN (
                             SELECT job_id FROM jobs WHERE run_id=%s
                           ) OR artifact_id IN (
                             SELECT artifact_id FROM artifacts WHERE run_id=%s
                           )""",
                    (run_id, run_id),
                ).rowcount
                for table in (
                    "monitor_events",
                    "monitor_spans",
                    "monitor_executions",
                    "unit_report_summaries",
                    "stream_runtime_progress",
                    "stream_units",
                    "unit_dependencies",
                    "tiles",
                    "partitions",
                    "events",
                    "jobs",
                    "artifacts",
                    "object_links",
                    "object_nodes",
                    "work_packages",
                    "spatial_units",
                    "streams",
                ):
                    deleted_totals[table] += connection.execute(
                        f"DELETE FROM {table} WHERE run_id=%s", (run_id,)
                    ).rowcount

                archived_status = "archived_" + str(candidate["status"])
                updated = connection.execute(
                    """UPDATE runs SET status=%s, metadata_json=%s, updated_at=%s
                           WHERE run_id=%s AND status=%s""",
                    (
                        archived_status,
                        _json(metadata),
                        archived_at,
                        run_id,
                        str(candidate["status"]),
                    ),
                ).rowcount
                if updated != 1:
                    raise RunStateError(f"Run changed during detail archive: {run_id}")
                connection.execute(
                    """INSERT INTO events
                           (run_id, timestamp, level, event_type, stream_id,
                            job_id, message, payload_json)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        run_id,
                        archived_at,
                        "info",
                        "run_details_archived",
                        "",
                        None,
                        "Incomplete Run details archived before a new Run",
                        _json(archive),
                    ),
                )
                archived_run_ids.append(run_id)

        archived_sample = archived_run_ids[:RUN_ARCHIVE_REPORT_ID_LIMIT]
        skipped_sample = skipped_active_run_ids[:RUN_ARCHIVE_REPORT_ID_LIMIT]
        return {
            "schema_version": 1,
            "status": "completed",
            "protected_run_id": protected,
            "archived_run_ids": archived_sample,
            "archived_run_count": len(archived_run_ids),
            "archived_run_ids_truncated": (
                len(archived_run_ids) > len(archived_sample)
            ),
            "skipped_active_run_ids": skipped_sample,
            "skipped_active_run_count": len(skipped_active_run_ids),
            "skipped_active_run_ids_truncated": (
                len(skipped_active_run_ids) > len(skipped_sample)
            ),
            "deleted_detail_counts": deleted_totals,
        }
