"""Run and result-Stream lifecycle persistence for PostgreSQL Run state."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from labeling_tool.shared.state.run_state_session import RunStateError, RunStateSession
from labeling_tool.shared.state.state_values import json_value as _json
from labeling_tool.shared.state.state_values import row_dict as _row_dict
from labeling_tool.shared.state.state_values import utc_now as _now

__all__ = ["SCHEMA_VERSION", "RunStreamRepository"]

SCHEMA_VERSION = 2


class RunStreamRepository:
    """Own Run and result-Stream identity, status, and runtime progress."""

    def __init__(self, session: RunStateSession) -> None:
        self._session = session

    def create_run(
        self,
        run_id: str,
        run_spec_sha256: str,
        *,
        status: str = "preflight",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        now = _now()
        with self._session.transaction() as connection:
            connection.execute(
                """INSERT INTO runs
                   (run_id, schema_version, status, run_spec_sha256,
                    metadata_json, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (
                    str(run_id),
                    SCHEMA_VERSION,
                    str(status),
                    str(run_spec_sha256),
                    _json(dict(metadata or {})),
                    now,
                    now,
                ),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._session.connection() as connection:
            result: dict[str, Any] | None = _row_dict(
                connection.execute(
                    "SELECT * FROM runs WHERE run_id=%s", (str(run_id),)
                ).fetchone()
            )
        return result

    def set_run_status(
        self,
        run_id: str,
        status: str,
        *,
        expected: str | Sequence[str] | None = None,
    ) -> bool:
        values: list[Any] = [str(status), _now(), str(run_id)]
        sql = (
            "UPDATE runs SET status=%s, updated_at=%s WHERE run_id=%s "
            "AND status NOT LIKE 'archived_%%'"
        )
        if expected is not None:
            states = [expected] if isinstance(expected, str) else list(expected)
            if not states:
                return False
            sql += " AND status IN (" + ",".join("%s" for _ in states) + ")"
            values.extend(str(item) for item in states)
        with self._session.transaction() as connection:
            return bool(connection.execute(sql, values).rowcount == 1)

    def update_run_metadata(
        self,
        run_id: str,
        values: Mapping[str, Any],
    ) -> None:
        """Merge bounded control-plane metadata without changing Run identity."""

        identifier = str(run_id)
        with self._session.transaction() as connection:
            row = connection.execute(
                "SELECT metadata_json FROM runs WHERE run_id=%s FOR UPDATE",
                (identifier,),
            ).fetchone()
            if row is None:
                raise RunStateError(f"unknown Run metadata target: {identifier}")
            try:
                metadata = json.loads(str(row["metadata_json"] or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise RunStateError(
                    f"Run metadata is not valid JSON: {identifier}"
                ) from error
            if not isinstance(metadata, dict):
                raise RunStateError(f"Run metadata must be an object: {identifier}")
            metadata.update(dict(values))
            updated = connection.execute(
                "UPDATE runs SET metadata_json=%s, updated_at=%s WHERE run_id=%s",
                (_json(metadata), _now(), identifier),
            ).rowcount
            if updated != 1:
                raise RunStateError(f"Run metadata changed during update: {identifier}")

    def register_streams(
        self, run_id: str, streams: Iterable[Mapping[str, Any]]
    ) -> None:
        now = _now()
        rows = (
            (
                str(run_id),
                str(item["stream_id"]),
                str(item["kind"]),
                str(item.get("model_id") or ""),
                str(item.get("profile_id") or ""),
                str(item.get("version") or ""),
                str(item.get("status") or "pending"),
                now,
                now,
            )
            for item in streams
        )
        with self._session.transaction() as connection:
            connection.executemany(
                """INSERT INTO streams
                   (run_id, stream_id, kind, model_id, profile_id, version,
                    status, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                rows,
            )

    def set_stream_status(
        self,
        run_id: str,
        stream_id: str,
        status: str,
        *,
        error: str = "",
    ) -> bool:
        with self._session.transaction() as connection:
            return bool(
                connection.execute(
                    """UPDATE streams SET status=%s, error=%s, updated_at=%s
                   WHERE run_id=%s AND stream_id=%s""",
                    (str(status), str(error), _now(), str(run_id), str(stream_id)),
                ).rowcount
                == 1
            )

    def upsert_stream_runtime_progress(
        self,
        run_id: str,
        stream_id: str,
        *,
        stage: str,
        phase: str,
        phase_name: str,
        phase_index: int,
        phase_total: int,
        current: int = 0,
        total: int = 0,
        feature_count: int = 0,
        status: str = "running",
        message: str = "",
    ) -> None:
        """Persist the latest bounded progress row for one result Stream.

        Progress is an overwriteable control-plane snapshot, not an event log.
        Keeping one row per Stream lets a reopened QGIS monitor recover the
        current phase without accumulating one database row per feature.
        """

        now = _now()
        with self._session.transaction() as connection:
            connection.execute(
                """INSERT INTO stream_runtime_progress
                   (run_id, stream_id, stage, phase, phase_name,
                    phase_index, phase_total, progress_current,
                    progress_total, feature_count, status, message,
                    phase_started_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT(run_id, stream_id) DO UPDATE SET
                     stage=excluded.stage,
                     phase=excluded.phase,
                     phase_name=excluded.phase_name,
                     phase_index=excluded.phase_index,
                     phase_total=excluded.phase_total,
                     progress_current=excluded.progress_current,
                     progress_total=excluded.progress_total,
                     feature_count=excluded.feature_count,
                     status=excluded.status,
                     message=excluded.message,
                     phase_started_at=CASE
                       WHEN stream_runtime_progress.phase!=excluded.phase
                         OR stream_runtime_progress.stage!=excluded.stage
                       THEN excluded.phase_started_at
                       ELSE stream_runtime_progress.phase_started_at
                     END,
                     updated_at=excluded.updated_at""",
                (
                    str(run_id),
                    str(stream_id),
                    str(stage),
                    str(phase),
                    str(phase_name),
                    max(0, int(phase_index)),
                    max(0, int(phase_total)),
                    max(0, int(current)),
                    max(0, int(total)),
                    max(0, int(feature_count)),
                    str(status),
                    str(message),
                    now,
                    now,
                ),
            )

    def stream_runtime_progress(
        self, run_id: str, stream_id: str = ""
    ) -> dict[str, dict[str, Any]]:
        sql = "SELECT * FROM stream_runtime_progress WHERE run_id=%s"
        values: list[Any] = [str(run_id)]
        if stream_id:
            sql += " AND stream_id=%s"
            values.append(str(stream_id))
        sql += " ORDER BY stream_id"
        with self._session.connection() as connection:
            rows = connection.execute(sql, values).fetchall()
        return {str(row["stream_id"]): dict(row) for row in rows}

    def fail_stream_runtime_progress(
        self, run_id: str, stream_id: str, error: str
    ) -> None:
        progress = (
            self.stream_runtime_progress(run_id, stream_id).get(str(stream_id)) or {}
        )
        self.upsert_stream_runtime_progress(
            run_id,
            stream_id,
            stage=str(progress.get("stage") or "assembly"),
            phase=str(progress.get("phase") or "failed"),
            phase_name=str(progress.get("phase_name") or "组装失败"),
            phase_index=int(progress.get("phase_index") or 0),
            phase_total=int(progress.get("phase_total") or 0),
            current=int(progress.get("progress_current") or 0),
            total=int(progress.get("progress_total") or 0),
            feature_count=int(progress.get("feature_count") or 0),
            status="failed",
            message=str(error),
        )

    def fail_open_streams(self, run_id: str, error: str) -> int:
        """Fail every non-ready stream after a terminal Run failure."""
        with self._session.transaction() as connection:
            return int(
                connection.execute(
                    """UPDATE streams SET status='failed', error=%s, updated_at=%s
                   WHERE run_id=%s AND status!='ready'""",
                    (str(error), _now(), str(run_id)),
                ).rowcount
            )

    def fail_terminal_publication(
        self,
        run_id: str,
        error: str,
        *,
        expected: str | Sequence[str] = ("running", "raster_ready", "ready"),
    ) -> bool:
        """Record one owned terminal-file failure without a false completion."""

        states = [expected] if isinstance(expected, str) else list(expected)
        if not states:
            return False
        identifier = str(run_id)
        message = str(error)
        now = _now()
        with self._session.transaction() as connection:
            changed = connection.execute(
                """UPDATE runs SET status='failed', updated_at=%s
                   WHERE run_id=%s AND status NOT LIKE 'archived_%%'
                     AND status IN ("""
                + ",".join("%s" for _ in states)
                + ")",
                [now, identifier, *(str(item) for item in states)],
            ).rowcount
            if changed != 1:
                return False
            connection.execute(
                """UPDATE streams SET status='failed', error=%s, updated_at=%s
                   WHERE run_id=%s AND status!='ready'""",
                (message, now, identifier),
            )
            connection.execute(
                """INSERT INTO events
                   (run_id,timestamp,level,event_type,message,payload_json)
                   VALUES (%s,%s,'error','terminal_publication_failed',%s,%s)""",
                (
                    identifier,
                    now,
                    message,
                    _json({"error": message}),
                ),
            )
        return True

    def stream_rows(self, run_id: str) -> list[dict[str, Any]]:
        with self._session.connection() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM streams WHERE run_id=%s ORDER BY stream_id",
                    (str(run_id),),
                ).fetchall()
            ]

    def append_event(
        self,
        run_id: str,
        event_type: str,
        *,
        level: str = "info",
        stream_id: str = "",
        job_id: int | None = None,
        message: str = "",
        payload: Mapping[str, Any] | None = None,
    ) -> int:
        with self._session.transaction() as connection:
            cursor = connection.execute(
                """INSERT INTO events
                   (run_id, timestamp, level, event_type, stream_id, job_id,
                    message, payload_json) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING event_id""",
                (
                    str(run_id),
                    _now(),
                    str(level),
                    str(event_type),
                    str(stream_id),
                    job_id,
                    str(message),
                    _json(dict(payload or {})),
                ),
            )
            row = cursor.fetchone()
            if row is None:
                raise RunStateError("event insert did not return an event_id")
            return int(row[0])
