"""Persistence and bounded queries for structured monitor history."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from labeling_tool.shared.contracts.monitor_contract import (
    MONITOR_DETAIL_PAGE_LIMIT,
    MONITOR_EVENT_PAGE_LIMIT,
    MONITOR_EVENT_PAGE_SIZE,
    MONITOR_HISTORY_VERSION,
    TERMINAL_SPAN_STATUSES,
    archived_history_summary,
)
from labeling_tool.shared.state.postgres_state import PostgresConnection
from labeling_tool.shared.state.run_state_session import RunStateError, RunStateSession
from labeling_tool.shared.state.state_values import (
    bounded_utf8,
    json_value,
    row_dict,
    utc_now,
)


class MonitorHistoryRepository:
    """Own monitor Execution, span, event, and history-query persistence."""

    def __init__(self, session: RunStateSession) -> None:
        self._session = session

    def begin_execution(
        self,
        run_id: str,
        trigger_type: str,
        *,
        execution_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Begin one start/resume/redo execution and seal older live attempts."""
        if self._session.unit_identity is not None:
            raise RunStateError(
                "monitor executions cannot begin inside a scoped unit transaction"
            )
        identifier = str(execution_id or uuid.uuid4())
        now = utc_now()
        with self._session.transaction() as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM runs WHERE run_id=%s", (str(run_id),)
                ).fetchone()
                is None
            ):
                raise RunStateError(f"Run does not exist: {run_id}")
            old_executions = [
                str(row["execution_id"])
                for row in connection.execute(
                    """SELECT execution_id FROM monitor_executions
                       WHERE run_id=%s AND status='running' FOR UPDATE""",
                    (str(run_id),),
                ).fetchall()
            ]
            if old_executions:
                connection.execute(
                    """UPDATE monitor_spans SET status='interrupted', ended_at=NULL,
                              message=CASE WHEN message='' THEN %s ELSE message END
                       WHERE run_id=%s AND status='running'""",
                    (
                        "A later execution confirmed this attempt was no longer active",
                        str(run_id),
                    ),
                )
                connection.execute(
                    """UPDATE monitor_executions SET status='interrupted', ended_at=NULL,
                              recording_complete=FALSE,
                              message=CASE WHEN message='' THEN %s ELSE message END
                       WHERE run_id=%s AND status='running'""",
                    (
                        "Superseded by a later start, resume, or redo",
                        str(run_id),
                    ),
                )
            connection.execute(
                """INSERT INTO monitor_executions
                   (execution_id, run_id, trigger_type, status, started_at,
                    ended_at, last_observed_at, recording_complete, message,
                    metadata_json)
                   VALUES (%s,%s,%s,'running',%s,NULL,%s,TRUE,'',%s)""",
                (
                    identifier,
                    str(run_id),
                    str(trigger_type),
                    now,
                    now,
                    json_value(dict(metadata or {})),
                ),
            )
            connection.execute(
                """INSERT INTO monitor_events
                   (run_id, execution_id, timestamp, level, event_type,
                    object_type, object_id, message, payload_json,
                    idempotency_key)
                   VALUES (%s,%s,%s,'info','execution_started','run',%s,%s,%s,%s)
                   ON CONFLICT(run_id,idempotency_key) DO NOTHING""",
                (
                    str(run_id),
                    identifier,
                    now,
                    str(run_id),
                    f"Execution started: {trigger_type}",
                    json_value({"trigger_type": str(trigger_type)}),
                    f"execution:{identifier}:started",
                ),
            )
        self._session.execution_id = identifier
        return identifier

    def finish_execution(
        self,
        run_id: str,
        execution_id: str,
        *,
        status: str,
        message: str = "",
        recording_complete: bool = True,
    ) -> bool:
        """Seal the current execution without fabricating span success."""
        final_status = str(status)
        if final_status not in {"completed", "failed", "stopped", "interrupted"}:
            raise ValueError(f"invalid monitor execution status: {status}")
        now = utc_now()
        with self._session.transaction() as connection:
            changed = connection.execute(
                """UPDATE monitor_executions
                   SET status=%s, ended_at=%s, last_observed_at=%s,
                       recording_complete=%s, message=%s
                   WHERE run_id=%s AND execution_id=%s AND status='running'""",
                (
                    final_status,
                    now,
                    now,
                    bool(recording_complete),
                    bounded_utf8(message, 8000),
                    str(run_id),
                    str(execution_id),
                ),
            ).rowcount
            if changed != 1:
                return False
            connection.execute(
                """UPDATE monitor_spans SET status='interrupted', ended_at=%s,
                          last_observed_at=%s,
                          message=CASE WHEN message='' THEN %s ELSE message END
                   WHERE run_id=%s AND execution_id=%s AND status='running'""",
                (
                    now,
                    now,
                    "Execution ended without a recorded terminal span outcome",
                    str(run_id),
                    str(execution_id),
                ),
            )
            connection.execute(
                """INSERT INTO monitor_events
                   (run_id, execution_id, timestamp, level, event_type,
                    object_type, object_id, message, payload_json,
                    idempotency_key)
                   VALUES (%s,%s,%s,%s,'execution_finished','run',%s,%s,%s,%s)
                   ON CONFLICT(run_id,idempotency_key) DO NOTHING""",
                (
                    str(run_id),
                    str(execution_id),
                    now,
                    "error" if final_status == "failed" else "info",
                    str(run_id),
                    bounded_utf8(message, 8000),
                    json_value({"status": final_status}),
                    f"execution:{execution_id}:finished:{final_status}",
                ),
            )
        return True

    def start_span(
        self,
        run_id: str,
        *,
        span_kind: str,
        execution_id: str = "",
        parent_span_id: str = "",
        object_type: str = "",
        object_id: str = "",
        stream_id: str = "",
        package_id: str = "",
        unit_id: str = "",
        model_id: str = "",
        phase: str = "",
        job_id: int | None = None,
        attempt_no: int = 0,
        budget_attempt: int = 0,
        idempotency_key: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Start an idempotent low-frequency attempt or phase span."""
        execution = str(execution_id or self._session.execution_id).strip()
        if not execution:
            raise RunStateError("monitor execution ID is required for a span")
        key = str(idempotency_key).strip() or uuid.uuid4().hex
        span_id = str(uuid.uuid4())
        now = utc_now()
        with self._session.transaction() as connection:
            if int(attempt_no) < 1:
                attempt_no = 1 + int(
                    connection.execute(
                        """SELECT COUNT(*) FROM monitor_spans
                           WHERE run_id=%s AND span_kind=%s AND object_type=%s
                             AND object_id=%s""",
                        (
                            str(run_id),
                            str(span_kind),
                            str(object_type),
                            str(object_id),
                        ),
                    ).fetchone()[0]
                )
            row = connection.execute(
                """INSERT INTO monitor_spans
                   (span_id,run_id,execution_id,parent_span_id,span_kind,
                    object_type,object_id,stream_id,package_id,unit_id,model_id,
                    phase,job_id,attempt_no,budget_attempt,status,started_at,
                    ended_at,last_observed_at,message,metadata_json,idempotency_key)
                   VALUES (%s,%s,%s,NULLIF(%s,''),%s,%s,%s,%s,%s,%s,%s,%s,%s,
                           %s,%s,'running',%s,NULL,%s,'',%s,%s)
                   ON CONFLICT(run_id,idempotency_key) DO UPDATE SET
                     last_observed_at=EXCLUDED.last_observed_at
                   RETURNING span_id""",
                (
                    span_id,
                    str(run_id),
                    execution,
                    str(parent_span_id),
                    str(span_kind),
                    str(object_type),
                    str(object_id),
                    str(stream_id),
                    str(package_id),
                    str(unit_id),
                    str(model_id),
                    str(phase),
                    job_id,
                    max(1, int(attempt_no)),
                    max(0, int(budget_attempt)),
                    now,
                    now,
                    json_value(dict(metadata or {})),
                    key,
                ),
            ).fetchone()
        return str(row[0])

    def finish_span(
        self,
        span_id: str,
        *,
        status: str,
        message: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        final_status = str(status)
        if final_status not in TERMINAL_SPAN_STATUSES:
            raise ValueError(f"invalid terminal monitor span status: {status}")
        now = utc_now()
        with self._session.transaction() as connection:
            row = connection.execute(
                """UPDATE monitor_spans SET status=%s, ended_at=%s,
                          last_observed_at=%s, message=%s,
                          metadata_json=CASE WHEN %s='{}' THEN metadata_json ELSE %s END
                   WHERE span_id=%s AND status='running'""",
                (
                    final_status,
                    now,
                    now,
                    bounded_utf8(message, 8000),
                    json_value(dict(metadata or {})),
                    json_value(dict(metadata or {})),
                    str(span_id),
                ),
            )
            if row.rowcount != 1:
                return False
            if final_status in {"completed", "reused"}:
                connection.execute(
                    """UPDATE monitor_events e SET recovered_by_span_id=%s
                       FROM monitor_spans s
                       WHERE s.span_id=%s AND e.run_id=s.run_id
                         AND e.recovered_by_span_id IS NULL
                         AND e.level IN ('warning','error')
                         AND e.timestamp<=%s
                         AND e.stream_id=s.stream_id
                         AND (
                           (s.job_id IS NOT NULL AND e.job_id=s.job_id) OR
                           (e.job_id IS NULL AND s.object_id<>''
                            AND e.object_type=s.object_type AND e.object_id=s.object_id)
                         )""",
                    (str(span_id), str(span_id), now),
                )
            return True

    def append_event(
        self,
        run_id: str,
        event_type: str,
        *,
        execution_id: str = "",
        span_id: str = "",
        level: str = "info",
        object_type: str = "",
        object_id: str = "",
        stream_id: str = "",
        package_id: str = "",
        unit_id: str = "",
        job_id: int | None = None,
        message: str = "",
        payload: Mapping[str, Any] | None = None,
        idempotency_key: str = "",
    ) -> int:
        key = str(idempotency_key).strip() or uuid.uuid4().hex
        now = utc_now()
        execution = str(execution_id or self._session.execution_id).strip()
        with self._session.transaction() as connection:
            row = connection.execute(
                """INSERT INTO monitor_events
                   (run_id,execution_id,span_id,timestamp,level,event_type,
                    object_type,object_id,stream_id,package_id,unit_id,job_id,
                    message,payload_json,idempotency_key)
                   VALUES (%s,NULLIF(%s,''),NULLIF(%s,''),%s,%s,%s,%s,%s,%s,%s,
                           %s,%s,%s,%s,%s)
                   ON CONFLICT(run_id,idempotency_key) DO UPDATE SET
                     idempotency_key=EXCLUDED.idempotency_key
                   RETURNING monitor_event_id""",
                (
                    str(run_id),
                    execution,
                    str(span_id),
                    now,
                    str(level),
                    str(event_type),
                    str(object_type),
                    str(object_id),
                    str(stream_id),
                    str(package_id),
                    str(unit_id),
                    job_id,
                    bounded_utf8(message, 8000),
                    json_value(dict(payload or {})),
                    key,
                ),
            ).fetchone()
        return int(row[0])

    def snapshot(self, run_id: str) -> dict[str, Any]:
        """Return the bounded standalone history summary contract."""
        with self._session.connection() as connection:
            run = row_dict(
                connection.execute(
                    "SELECT * FROM runs WHERE run_id=%s", (str(run_id),)
                ).fetchone()
            )
            if run is None:
                return {"available": False, "reason": "run_missing"}
            if str(run.get("status") or "").startswith("archived_"):
                return {
                    "available": False,
                    "archived": True,
                    "summary": archived_history_summary(run),
                }
            latest = row_dict(
                connection.execute(
                    """SELECT * FROM monitor_executions WHERE run_id=%s
                       ORDER BY started_at DESC, execution_id DESC LIMIT 1""",
                    (str(run_id),),
                ).fetchone()
            )
            counts = {
                str(row["status"]): int(row["n"])
                for row in connection.execute(
                    """SELECT status,COUNT(*) AS n FROM monitor_spans
                       WHERE run_id=%s GROUP BY status""",
                    (str(run_id),),
                ).fetchall()
            }
            recent = [
                dict(row)
                for row in connection.execute(
                    """SELECT * FROM monitor_events WHERE run_id=%s
                       ORDER BY monitor_event_id DESC LIMIT 5""",
                    (str(run_id),),
                ).fetchall()
            ]
        for row in recent:
            row["payload"] = json.loads(str(row.pop("payload_json") or "{}"))
        return {
            "available": latest is not None,
            "history_version": MONITOR_HISTORY_VERSION,
            "latest_execution": latest or {},
            "span_status_counts": counts,
            "recent_events": recent,
            "reason": "upgrade_precedes_history" if latest is None else "",
        }

    def page_events(
        self,
        run_id: str,
        *,
        before_event_id: int | None = None,
        execution_id: str = "",
        object_type: str = "",
        object_id: str = "",
        stream_id: str = "",
        package_id: str = "",
        job_id: int | None = None,
        span_id: str = "",
        scope: str = "all",
        levels: Sequence[str] = (),
        search: str = "",
        limit: int = MONITOR_EVENT_PAGE_SIZE,
    ) -> list[dict[str, Any]]:
        size = max(1, min(int(limit), MONITOR_EVENT_PAGE_LIMIT))
        sql = "SELECT * FROM monitor_events WHERE run_id=%s"
        values: list[Any] = [str(run_id)]
        if before_event_id is not None:
            sql += " AND monitor_event_id<%s"
            values.append(int(before_event_id))
        if execution_id:
            sql += " AND execution_id=%s"
            values.append(str(execution_id))
        if object_type:
            sql += " AND object_type=%s"
            values.append(str(object_type))
        if object_id:
            sql += " AND object_id=%s"
            values.append(str(object_id))
        for column, value in (
            ("stream_id", stream_id),
            ("package_id", package_id),
            ("job_id", job_id),
            ("span_id", span_id),
        ):
            if value not in (None, ""):
                sql += f" AND {column}=%s"
                values.append(value)
        if scope == "issues":
            sql += " AND level IN ('warning','error') AND recovered_by_span_id IS NULL"
        elif scope == "warnings":
            sql += " AND level IN ('warning','error')"
        elif scope == "recovery":
            sql += """ AND (recovered_by_span_id IS NOT NULL
                         OR event_type ~ '(retry|reused|reduced|resume)')"""
        if levels:
            placeholders = ",".join("%s" for _level in levels)
            sql += f" AND level IN ({placeholders})"
            values.extend(str(level) for level in levels)
        if search:
            pattern = (
                "%"
                + str(search)
                .replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
                + "%"
            )
            sql += " AND (message ILIKE %s ESCAPE '\\' OR event_type ILIKE %s ESCAPE '\\' OR object_id ILIKE %s ESCAPE '\\')"
            values.extend((pattern, pattern, pattern))
        sql += " ORDER BY monitor_event_id DESC LIMIT %s"
        values.append(size)
        with self._session.connection() as connection:
            rows = [dict(row) for row in connection.execute(sql, values).fetchall()]
        for row in rows:
            row["payload"] = json.loads(str(row.pop("payload_json") or "{}"))
        return rows

    def page_spans(
        self,
        run_id: str,
        *,
        execution_id: str = "",
        object_type: str = "",
        object_id: str = "",
        stream_id: str = "",
        package_id: str = "",
        span_kind: str = "",
        parent_span_id: str = "",
        job_id: int | None = None,
        before_started_at: str = "",
        before_span_id: str = "",
        status: str = "",
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM monitor_spans WHERE run_id=%s"
        values: list[Any] = [str(run_id)]
        for column, value in (
            ("execution_id", execution_id),
            ("object_type", object_type),
            ("object_id", object_id),
            ("stream_id", stream_id),
            ("package_id", package_id),
            ("span_kind", span_kind),
            ("parent_span_id", parent_span_id),
            ("job_id", job_id),
            ("status", status),
        ):
            if value:
                sql += f" AND {column}=%s"
                values.append(str(value))
        if before_started_at and before_span_id:
            sql += " AND (started_at,span_id)<(%s,%s)"
            values.extend((str(before_started_at), str(before_span_id)))
        sql += " ORDER BY started_at DESC,span_id DESC LIMIT %s OFFSET %s"
        values.extend(
            (
                max(1, min(int(limit), MONITOR_DETAIL_PAGE_LIMIT)),
                max(0, int(offset)),
            )
        )
        with self._session.connection() as connection:
            rows = [dict(row) for row in connection.execute(sql, values).fetchall()]
        for row in rows:
            row["metadata"] = json.loads(str(row.pop("metadata_json") or "{}"))
        return rows

    def read_snapshot_sections(
        self,
        connection: PostgresConnection,
        run_id: str,
        run: Mapping[str, Any] | None,
    ) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
        """Read aggregate sections without owning, closing, or committing connection."""
        identifier = str(run_id)
        assembly_phase_statuses: dict[str, dict[str, dict[str, Any]]] = {}
        try:
            for row in connection.execute(
                """SELECT DISTINCT ON (stream_id,phase)
                          stream_id,phase,status,started_at,ended_at,execution_id,span_id,parent_span_id,
                          message,metadata_json
                   FROM monitor_spans
                   WHERE run_id=%s AND span_kind='assembly_phase'
                     AND execution_id=(
                       SELECT execution_id FROM monitor_executions WHERE run_id=%s
                       ORDER BY started_at DESC,execution_id DESC LIMIT 1)
                   ORDER BY stream_id,phase,started_at DESC,span_id DESC""",
                (identifier, identifier),
            ).fetchall():
                item = dict(row)
                try:
                    metadata = json.loads(str(item.pop("metadata_json") or "{}"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    metadata = {}
                item.update(metadata if isinstance(metadata, dict) else {})
                assembly_phase_statuses.setdefault(str(item["stream_id"]), {})[
                    str(item["phase"])
                ] = item
        except Exception:
            assembly_phase_statuses = {}
        try:
            latest_execution = row_dict(
                connection.execute(
                    """SELECT * FROM monitor_executions WHERE run_id=%s
                       ORDER BY started_at DESC,execution_id DESC LIMIT 1""",
                    (identifier,),
                ).fetchone()
            )
            monitor_executions = [
                dict(row)
                for row in connection.execute(
                    """SELECT * FROM monitor_executions WHERE run_id=%s
                       ORDER BY started_at DESC,execution_id DESC LIMIT 50""",
                    (identifier,),
                ).fetchall()
            ]
            monitor_span_status_counts = {
                str(row["status"]): int(row["n"])
                for row in connection.execute(
                    """SELECT status,COUNT(*) AS n FROM monitor_spans
                       WHERE run_id=%s GROUP BY status""",
                    (identifier,),
                ).fetchall()
            }
            recent_monitor_events = [
                dict(row)
                for row in connection.execute(
                    """SELECT * FROM monitor_events WHERE run_id=%s
                       ORDER BY monitor_event_id DESC LIMIT 5""",
                    (identifier,),
                ).fetchall()
            ]
            for event in recent_monitor_events:
                try:
                    event["payload"] = json.loads(
                        str(event.pop("payload_json") or "{}")
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    event["payload"] = {}
            monitor_history = {
                "available": latest_execution is not None,
                "history_version": MONITOR_HISTORY_VERSION,
                "latest_execution": latest_execution or {},
                "executions": monitor_executions,
                "span_status_counts": monitor_span_status_counts,
                "recent_events": recent_monitor_events,
                "reason": (
                    "" if latest_execution is not None else "upgrade_precedes_history"
                ),
            }
            if str((run or {}).get("status") or "").startswith("archived_"):
                monitor_history = {
                    "available": False,
                    "archived": True,
                    "history_version": MONITOR_HISTORY_VERSION,
                    "summary": archived_history_summary(run),
                    "recent_events": [],
                }
        except Exception as error:
            assembly_phase_statuses = {}
            monitor_history = {
                "available": False,
                "reason": "history_query_failed",
                "error": f"{type(error).__name__}: {error}",
            }
        return assembly_phase_statuses, monitor_history
