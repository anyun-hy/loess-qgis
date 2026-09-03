"""PostgreSQL production state store for large, resumable inference runs."""

from __future__ import annotations

import contextlib
import datetime as _datetime
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .postgres_state import (
    DEFAULT_POSTGRES_DSN,
    DEFAULT_POSTGRES_SCHEMA,
    connect_postgres,
    initialize_postgres,
    is_postgres_location,
    postgres_health,
    validate_schema,
)


SCHEMA_VERSION = 2
MAX_TILE_PAGE_SIZE = 500
STATE_DB_DSN_ENV = "LOESS_STATE_DB_DSN"
STATE_DB_SCHEMA_ENV = "LOESS_STATE_DB_SCHEMA"
ARCHIVABLE_INCOMPLETE_RUN_STATES = ("failed", "stopped")
RUN_DETAIL_ARCHIVE_KEY = "database_detail_archive"
RUN_ARCHIVE_REPORT_ID_LIMIT = 50
RUN_DETAIL_TABLES = (
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


class RunStateError(RuntimeError):
    pass


def _now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat(timespec="milliseconds")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _bounded_utf8(value: Any, maximum_bytes: int) -> str:
    """Bound diagnostic text by encoded bytes without returning invalid UTF-8."""

    encoded = str(value or "").encode("utf-8")
    if len(encoded) <= int(maximum_bytes):
        return encoded.decode("utf-8")
    return encoded[: int(maximum_bytes)].decode("utf-8", errors="ignore")


def _row_dict(row: Any | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def production_state_database() -> str:
    """Return the password-free PostgreSQL DSN frozen into new Run Specs."""

    return str(os.environ.get(STATE_DB_DSN_ENV) or DEFAULT_POSTGRES_DSN).strip()


def production_state_schema() -> str:
    return validate_schema(
        os.environ.get(STATE_DB_SCHEMA_ENV) or DEFAULT_POSTGRES_SCHEMA
    )


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
    ):
        location = str(dsn).strip()
        if not is_postgres_location(location):
            raise RunStateError(
                "Run state requires a PostgreSQL DSN; filesystem databases are "
                "no longer supported"
            )
        self.location = location
        self.postgres_schema = validate_schema(
            postgres_schema or production_state_schema()
        )

    def _connect(self, *, autocommit: bool = True):
        return connect_postgres(
            self.location,
            schema=self.postgres_schema,
            autocommit=autocommit,
        )

    @contextlib.contextmanager
    def _connection(self) -> Iterator[Any]:
        """Yield one short-lived connection and always close its file handles."""
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextlib.contextmanager
    def transaction(self) -> Iterator[Any]:
        connection = self._connect(autocommit=False)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        initialize_postgres(
            self.location,
            schema=self.postgres_schema,
            schema_version=SCHEMA_VERSION,
            now=_now(),
        )

    def pragmas(self) -> dict[str, Any]:
        report = postgres_health(
            self.location,
            schema=self.postgres_schema,
            schema_version=SCHEMA_VERSION,
        )
        return {
            **report,
            "journal_mode": "postgresql-mvcc",
            "foreign_keys": 1,
            "user_version": SCHEMA_VERSION,
        }

    def create_run(
        self,
        run_id: str,
        run_spec_sha256: str,
        *,
        status: str = "preflight",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        now = _now()
        with self.transaction() as connection:
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
        with self._connection() as connection:
            return _row_dict(
                connection.execute(
                    "SELECT * FROM runs WHERE run_id=%s", (str(run_id),)
                ).fetchone()
            )

    def archive_incomplete_run_details(
        self,
        *,
        protected_run_id: str,
        reason: str = "new_run_housekeeping",
    ) -> dict[str, Any]:
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

        with self.transaction() as connection:
            placeholders = ",".join(
                "%s" for _state in ARCHIVABLE_INCOMPLETE_RUN_STATES
            )
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
                    preserved_metadata["run_spec"] = _bounded_utf8(
                        run_spec_path, 4096
                    )
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
                error_record_count = sum(
                    int(
                        connection.execute(sql, (run_id, "")).fetchone()[0]
                    )
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
                    raise RunStateError(
                        f"Run changed during detail archive: {run_id}"
                    )
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
        with self.transaction() as connection:
            return connection.execute(sql, values).rowcount == 1

    def update_run_metadata(
        self,
        run_id: str,
        values: Mapping[str, Any],
    ) -> None:
        """Merge bounded control-plane metadata without changing Run identity."""

        identifier = str(run_id)
        with self.transaction() as connection:
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
                raise RunStateError(
                    f"Run metadata must be an object: {identifier}"
                )
            metadata.update(dict(values))
            updated = connection.execute(
                "UPDATE runs SET metadata_json=%s, updated_at=%s WHERE run_id=%s",
                (_json(metadata), _now(), identifier),
            ).rowcount
            if updated != 1:
                raise RunStateError(
                    f"Run metadata changed during update: {identifier}"
                )

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
        with self.transaction() as connection:
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
        with self.transaction() as connection:
            return connection.execute(
                """UPDATE streams SET status=%s, error=%s, updated_at=%s
                   WHERE run_id=%s AND stream_id=%s""",
                (str(status), str(error), _now(), str(run_id), str(stream_id)),
            ).rowcount == 1

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
        with self.transaction() as connection:
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
        with self._connection() as connection:
            rows = connection.execute(sql, values).fetchall()
        return {str(row["stream_id"]): dict(row) for row in rows}

    def fail_stream_runtime_progress(
        self, run_id: str, stream_id: str, error: str
    ) -> None:
        progress = self.stream_runtime_progress(run_id, stream_id).get(
            str(stream_id)
        ) or {}
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
        with self.transaction() as connection:
            return connection.execute(
                """UPDATE streams SET status='failed', error=%s, updated_at=%s
                   WHERE run_id=%s AND status!='ready'""",
                (str(error), _now(), str(run_id)),
            ).rowcount

    def insert_work_packages(
        self, run_id: str, packages: Iterable[Mapping[str, Any]]
    ) -> int:
        now = _now()
        count = 0

        def rows():
            nonlocal count
            for item in packages:
                count += 1
                yield (
                    str(run_id),
                    str(item["package_id"]),
                    int(item["sequence_no"]),
                    int(item.get("estimated_bytes", 0)),
                    _json({
                        "partition_ids": item.get("partition_ids") or [],
                        "tile_count": int(item.get("tile_count", 0)),
                        "tile_windows": item.get("tile_windows") or [],
                        "neighbor_package_ids": item.get("neighbor_package_ids") or [],
                    }),
                    str(item.get("status") or "queued"),
                    now,
                    now,
                )

        with self.transaction() as connection:
            connection.executemany(
                """INSERT INTO work_packages
                   (run_id, package_id, sequence_no, estimated_bytes, metadata_json,
                    status, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                rows(),
            )
        return count

    def get_work_package(self, run_id: str, package_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM work_packages WHERE run_id=%s AND package_id=%s",
                (str(run_id), str(package_id)),
            ).fetchone()
        result = _row_dict(row)
        if result is not None:
            result["metadata"] = json.loads(result.pop("metadata_json"))
        return result

    def set_work_package_status(
        self,
        run_id: str,
        package_id: str,
        status: str,
        *,
        expected: str | Sequence[str] | None = None,
    ) -> bool:
        values: list[Any] = [str(status), _now(), str(run_id), str(package_id)]
        sql = (
            "UPDATE work_packages SET status=%s, updated_at=%s "
            "WHERE run_id=%s AND package_id=%s"
        )
        if expected is not None:
            states = [expected] if isinstance(expected, str) else list(expected)
            if not states:
                return False
            sql += " AND status IN (" + ",".join("%s" for _ in states) + ")"
            values.extend(str(item) for item in states)
        with self.transaction() as connection:
            return connection.execute(sql, values).rowcount == 1

    def package_partitions(self, run_id: str, package_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT * FROM partitions WHERE run_id=%s AND package_id=%s
                   ORDER BY row_no, col_no""",
                (str(run_id), str(package_id)),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["row"] = item.pop("row_no")
            item["col"] = item.pop("col_no")
            item["core_window"] = json.loads(item.pop("core_window_json"))
            item["halo_window"] = json.loads(item.pop("halo_window_json"))
            result.append(item)
        return result

    def partitions_for_run(self, run_id: str) -> list[dict[str, Any]]:
        """Return every Run Partition in deterministic row/column order."""
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT * FROM partitions WHERE run_id=%s
                   ORDER BY row_no, col_no""",
                (str(run_id),),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["row"] = item.pop("row_no")
            item["col"] = item.pop("col_no")
            item["core_window"] = json.loads(item.pop("core_window_json"))
            item["halo_window"] = json.loads(item.pop("halo_window_json"))
            result.append(item)
        return result

    def get_partition(self, run_id: str, partition_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM partitions WHERE run_id=%s AND partition_id=%s",
                (str(run_id), str(partition_id)),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["row"] = item.pop("row_no")
        item["col"] = item.pop("col_no")
        item["core_window"] = json.loads(item.pop("core_window_json"))
        item["halo_window"] = json.loads(item.pop("halo_window_json"))
        return item

    def package_tiles(self, run_id: str, package_id: str) -> list[dict[str, Any]]:
        package = self.get_work_package(run_id, package_id)
        if package is None:
            raise RunStateError(f"unknown Work Package: {package_id}")
        windows = list(package["metadata"].get("tile_windows") or [])
        selected: dict[str, dict[str, Any]] = {}
        with self._connection() as connection:
            for raw_window in windows:
                if not isinstance(raw_window, list) or len(raw_window) != 4:
                    raise RunStateError(f"invalid Tile window in Work Package: {raw_window}")
                row_start, row_stop, col_start, col_stop = map(int, raw_window)
                for row in connection.execute(
                    """SELECT * FROM tiles WHERE run_id=%s
                       AND row_no>=%s AND row_no<%s AND col_no>=%s AND col_no<%s
                       ORDER BY row_no, col_no""",
                    (str(run_id), row_start, row_stop, col_start, col_stop),
                ):
                    selected[str(row["tile_id"])] = dict(row)
        result = sorted(selected.values(), key=lambda item: (item["row_no"], item["col_no"]))
        expected = int(package["metadata"].get("tile_count", len(result)))
        if len(result) != expected:
            raise RunStateError(
                f"Work Package Tile count mismatch: expected {expected}, found {len(result)}"
            )
        return result

    def releasable_package_tile_ids(
        self,
        run_id: str,
        completing_package_id: str,
    ) -> list[str]:
        """Return current Package Tiles with no unfinished Package consumers."""
        current_tiles = self.package_tiles(run_id, completing_package_id)
        candidates = {
            (int(tile["row_no"]), int(tile["col_no"])): str(tile["tile_id"])
            for tile in current_tiles
            if str(tile.get("status")) != "excluded"
        }
        candidate_ids = set(candidates.values())
        blocked: set[str] = set()
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT package_id, status, metadata_json
                   FROM work_packages
                   WHERE run_id=%s AND package_id!=%s AND status!='ready'
                   ORDER BY sequence_no""",
                (str(run_id), str(completing_package_id)),
            ).fetchall()
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            windows = list(metadata.get("tile_windows") or [])
            for raw_window in windows:
                if not isinstance(raw_window, list) or len(raw_window) != 4:
                    raise RunStateError(
                        "invalid Tile window in Work Package: "
                        f"{row['package_id']}={raw_window}"
                    )
                row_start, row_stop, col_start, col_stop = map(int, raw_window)
                for (tile_row, tile_col), tile_id in candidates.items():
                    if (
                        row_start <= tile_row < row_stop
                        and col_start <= tile_col < col_stop
                    ):
                        blocked.add(tile_id)
        return [
            str(tile["tile_id"])
            for tile in current_tiles
            if str(tile["tile_id"]) in candidate_ids
            and str(tile["tile_id"]) not in blocked
        ]

    def work_package_counts(self, run_id: str) -> dict[str, int]:
        with self._connection() as connection:
            return {
                str(row["status"]): int(row["n"])
                for row in connection.execute(
                    """SELECT status, COUNT(*) AS n FROM work_packages
                       WHERE run_id=%s GROUP BY status""",
                    (str(run_id),),
                ).fetchall()
            }

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
        with self._connection() as connection:
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

    def page_work_packages(
        self,
        run_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM work_packages WHERE run_id=%s"
        values: list[Any] = [str(run_id)]
        if status is not None:
            sql += " AND status=%s"
            values.append(str(status))
        sql += " ORDER BY sequence_no LIMIT %s OFFSET %s"
        values.extend((max(1, min(int(limit), 500)), max(0, int(offset))))
        with self._connection() as connection:
            rows = connection.execute(sql, values).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            result.append(item)
        return result

    def insert_partitions(
        self, run_id: str, partitions: Iterable[Mapping[str, Any]]
    ) -> int:
        now = _now()
        count = 0

        def rows():
            nonlocal count
            for item in partitions:
                count += 1
                yield (
                    str(run_id),
                    str(item["partition_id"]),
                    int(item["row"]),
                    int(item["col"]),
                    _json(item["core_window"]),
                    _json(item["halo_window"]),
                    str(item.get("package_id") or "") or None,
                    str(item.get("status") or "queued"),
                    now,
                    now,
                )

        with self.transaction() as connection:
            connection.executemany(
                """INSERT INTO partitions
                   (run_id, partition_id, row_no, col_no, core_window_json,
                    halo_window_json, package_id, status, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                rows(),
            )
        return count

    def insert_spatial_units(
        self, run_id: str, units: Iterable[Mapping[str, Any]]
    ) -> int:
        now = _now()
        unit_values = list(units)
        rows = [
                (
                    str(run_id),
                    str(item["unit_id"]),
                    str(item["unit_type"]),
                    str(item["owner_key"]),
                    _json(item["pixel_window"]),
                    _json(item.get("dependency_ids") or []),
                    str(item.get("status") or "queued"),
                    now,
                    now,
                )
                for item in unit_values
        ]
        dependency_rows = [
            (str(run_id), str(item["unit_id"]), str(partition_id))
            for item in unit_values
            for partition_id in item.get("dependency_ids") or []
        ]

        with self.transaction() as connection:
            connection.executemany(
                """INSERT INTO spatial_units
                   (run_id, unit_id, unit_type, owner_key, pixel_window_json,
                    dependency_ids_json, status, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                rows,
            )
            connection.executemany(
                """INSERT INTO unit_dependencies
                   (run_id, unit_id, partition_id) VALUES (%s, %s, %s)""",
                dependency_rows,
            )
        return len(unit_values)

    def get_spatial_unit(self, run_id: str, unit_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM spatial_units WHERE run_id=%s AND unit_id=%s",
                (str(run_id), str(unit_id)),
            ).fetchone()
        result = _row_dict(row)
        if result is not None:
            result["pixel_window"] = json.loads(result.pop("pixel_window_json"))
            result["dependency_ids"] = json.loads(result.pop("dependency_ids_json"))
        return result

    def spatial_units(self, run_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT * FROM spatial_units WHERE run_id=%s
                   ORDER BY unit_type, unit_id""",
                (str(run_id),),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["pixel_window"] = json.loads(item.pop("pixel_window_json"))
            item["dependency_ids"] = json.loads(item.pop("dependency_ids_json"))
            result.append(item)
        return result

    def spatial_units_for_stream(
        self, run_id: str, stream_id: str
    ) -> list[dict[str, Any]]:
        """Return only geometry units registered for one result stream."""

        with self._connection() as connection:
            rows = connection.execute(
                """SELECT u.* FROM spatial_units u
                   JOIN stream_units su
                     ON su.run_id=u.run_id AND su.unit_id=u.unit_id
                   WHERE u.run_id=%s AND su.stream_id=%s
                   ORDER BY u.unit_type, u.unit_id""",
                (str(run_id), str(stream_id)),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["pixel_window"] = json.loads(item.pop("pixel_window_json"))
            item["dependency_ids"] = json.loads(item.pop("dependency_ids_json"))
            result.append(item)
        return result

    def set_spatial_unit_status(
        self, run_id: str, unit_id: str, status: str
    ) -> bool:
        with self.transaction() as connection:
            return connection.execute(
                """UPDATE spatial_units SET status=%s, updated_at=%s
                   WHERE run_id=%s AND unit_id=%s""",
                (str(status), _now(), str(run_id), str(unit_id)),
            ).rowcount == 1

    def insert_stream_units(
        self,
        run_id: str,
        stream_ids: Iterable[str],
        unit_ids: Iterable[str],
    ) -> int:
        now = _now()
        streams = [str(value) for value in stream_ids]
        units = [str(value) for value in unit_ids]
        rows = [
            (str(run_id), stream_id, unit_id, now, now)
            for stream_id in streams
            for unit_id in units
        ]
        with self.transaction() as connection:
            connection.executemany(
                """INSERT INTO stream_units
                   (run_id, stream_id, unit_id, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s)""",
                rows,
            )
        return len(rows)

    def set_stream_unit_status(
        self,
        run_id: str,
        stream_id: str,
        unit_id: str,
        status: str,
        *,
        error: str = "",
    ) -> bool:
        with self.transaction() as connection:
            return connection.execute(
                """UPDATE stream_units SET status=%s, error=%s, updated_at=%s
                   WHERE run_id=%s AND stream_id=%s AND unit_id=%s""",
                (
                    str(status),
                    str(error),
                    _now(),
                    str(run_id),
                    str(stream_id),
                    str(unit_id),
                ),
            ).rowcount == 1

    def stream_unit_counts(self, run_id: str, stream_id: str) -> dict[str, int]:
        with self._connection() as connection:
            return {
                str(row["status"]): int(row["n"])
                for row in connection.execute(
                    """SELECT status, COUNT(*) AS n FROM stream_units
                       WHERE run_id=%s AND stream_id=%s GROUP BY status""",
                    (str(run_id), str(stream_id)),
                ).fetchall()
            }

    def stream_unit_type_counts(
        self, run_id: str, stream_id: str
    ) -> dict[str, dict[str, int]]:
        """Return bounded status counts grouped by spatial-unit type."""
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT u.unit_type, su.status, COUNT(*) AS n
                   FROM stream_units su
                   JOIN spatial_units u
                     ON u.run_id=su.run_id AND u.unit_id=su.unit_id
                   WHERE su.run_id=%s AND su.stream_id=%s
                   GROUP BY u.unit_type, su.status""",
                (str(run_id), str(stream_id)),
            ).fetchall()
        result: dict[str, dict[str, int]] = {}
        for row in rows:
            result.setdefault(str(row["unit_type"]), {})[
                str(row["status"])
            ] = int(row["n"])
        return result

    def insert_tiles(
        self, run_id: str, tiles: Iterable[Mapping[str, Any]]
    ) -> int:
        now = _now()
        count = 0

        def rows():
            nonlocal count
            for item in tiles:
                count += 1
                yield (
                    str(run_id),
                    str(item["tile_id"]),
                    int(item["row"]),
                    int(item["col"]),
                    int(item.get("width", 512)),
                    int(item.get("height", 512)),
                    _json(item.get("pixel_window") or {}),
                    _json(item.get("bounds") or {}),
                    str(item.get("raster_path") or item.get("path") or ""),
                    str(item.get("sha256") or ""),
                    str(item.get("partition_id") or "") or None,
                    str(item.get("status") or "queued"),
                    now,
                    now,
                )

        with self.transaction() as connection:
            connection.executemany(
                """INSERT INTO tiles
                   (run_id, tile_id, row_no, col_no, width, height,
                    pixel_window_json, bounds_json, raster_path, sha256,
                    partition_id, status, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                rows(),
            )
        return count

    def count_tiles(
        self,
        run_id: str,
        *,
        status: str | None = None,
        search: str = "",
    ) -> int:
        sql = "SELECT COUNT(*) FROM tiles WHERE run_id=%s"
        values: list[Any] = [str(run_id)]
        if status is not None:
            sql += " AND status=%s"
            values.append(str(status))
        if search:
            sql += " AND tile_id LIKE %s ESCAPE '\\'"
            escaped = (
                str(search)
                .replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            values.append(f"%{escaped}%")
        with self._connection() as connection:
            return int(connection.execute(sql, values).fetchone()[0])

    def active_work_package_job(self, run_id: str) -> dict[str, Any] | None:
        """Return the single active accelerator Package, if one is leased."""
        with self._connection() as connection:
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

    def monitor_snapshot(self, run_id: str) -> dict[str, Any]:
        """Read the complete bounded monitor summary through one connection."""
        identifier = str(run_id)
        with self._connection() as connection:
            # Keep every aggregate in one read transaction. Without an
            # explicit repeatable-read snapshot, different commits may be
            # exposed to individual SELECT statements in one polling cycle.
            connection.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
            run = _row_dict(
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
            job_counts = {job_type: {} for job_type in monitored_job_types}
            for row in connection.execute(
                """SELECT job_type, status, COUNT(*) AS n FROM jobs
                   WHERE run_id=%s AND job_type IN (
                     'work_package','fragmentation_v33','unit_confidence','unit_fit'
                   )
                   GROUP BY job_type, status""",
                (identifier,),
            ).fetchall():
                job_counts[str(row["job_type"])][str(row["status"])] = int(
                    row["n"]
                )
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
            active_package = _row_dict(
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
            stream_unit_type_counts: dict[
                str, dict[str, dict[str, int]]
            ] = {}
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
                ).setdefault(str(row["unit_type"]), {})[
                    str(row["status"])
                ] = int(row["n"])
            stream_unit_job_type_counts: dict[
                str, dict[str, dict[str, int]]
            ] = {}
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
                ).setdefault(str(row["unit_type"]), {})[
                    str(row["status"])
                ] = int(row["n"])
            # Historical schema-v2 Runs created before structured assembly
            # monitoring do not have this additive table.  Keep them readable
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
        }

    def update_tile_raster(
        self,
        run_id: str,
        tile_id: str,
        *,
        raster_path: str,
        sha256: str,
    ) -> bool:
        """Record a Tile materialized lazily by a Work Package."""
        digest = str(sha256).lower()
        if len(digest) != 64:
            raise ValueError("Tile sha256 must contain 64 hexadecimal characters")
        try:
            int(digest, 16)
        except ValueError as error:
            raise ValueError(
                "Tile sha256 must contain 64 hexadecimal characters"
            ) from error
        with self.transaction() as connection:
            return connection.execute(
                """UPDATE tiles SET raster_path=%s, sha256=%s, updated_at=%s
                   WHERE run_id=%s AND tile_id=%s AND status!='excluded'""",
                (
                    str(raster_path),
                    digest,
                    _now(),
                    str(run_id),
                    str(tile_id),
                ),
            ).rowcount == 1

    def page_tiles(
        self,
        run_id: str,
        *,
        limit: int = MAX_TILE_PAGE_SIZE,
        offset: int = 0,
        status: str | None = None,
        partition_id: str | None = None,
        search: str = "",
    ) -> list[dict[str, Any]]:
        page_size = max(1, min(int(limit), MAX_TILE_PAGE_SIZE))
        sql = "SELECT * FROM tiles WHERE run_id=%s"
        values: list[Any] = [str(run_id)]
        if status is not None:
            sql += " AND status=%s"
            values.append(str(status))
        if partition_id is not None:
            sql += " AND partition_id=%s"
            values.append(str(partition_id))
        if search:
            sql += " AND tile_id LIKE %s ESCAPE '\\'"
            escaped = str(search).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            values.append(f"%{escaped}%")
        sql += " ORDER BY row_no, col_no LIMIT %s OFFSET %s"
        values.extend((page_size, max(0, int(offset))))
        with self._connection() as connection:
            return [dict(row) for row in connection.execute(sql, values).fetchall()]

    def count_stream_units(
        self,
        run_id: str,
        stream_id: str,
        *,
        unit_type: str = "",
        status: str = "",
        search: str = "",
    ) -> int:
        sql = (
            "SELECT COUNT(*) FROM stream_units su "
            "JOIN spatial_units u ON u.run_id=su.run_id AND u.unit_id=su.unit_id "
            "WHERE su.run_id=%s AND su.stream_id=%s"
        )
        values: list[Any] = [str(run_id), str(stream_id)]
        if unit_type:
            sql += " AND u.unit_type=%s"
            values.append(str(unit_type))
        if status:
            sql += " AND su.status=%s"
            values.append(str(status))
        if search:
            sql += " AND su.unit_id LIKE %s ESCAPE '\\'"
            escaped = str(search).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            values.append(f"%{escaped}%")
        with self._connection() as connection:
            return int(connection.execute(sql, values).fetchone()[0])

    def page_stream_units(
        self,
        run_id: str,
        stream_id: str,
        *,
        limit: int = MAX_TILE_PAGE_SIZE,
        offset: int = 0,
        unit_type: str = "",
        status: str = "",
        search: str = "",
    ) -> list[dict[str, Any]]:
        page_size = max(1, min(int(limit), MAX_TILE_PAGE_SIZE))
        sql = (
            "SELECT su.stream_id, su.unit_id, u.unit_type, su.status, su.error, "
            "u.owner_key, u.pixel_window_json FROM stream_units su "
            "JOIN spatial_units u ON u.run_id=su.run_id AND u.unit_id=su.unit_id "
            "WHERE su.run_id=%s AND su.stream_id=%s"
        )
        values: list[Any] = [str(run_id), str(stream_id)]
        if unit_type:
            sql += " AND u.unit_type=%s"
            values.append(str(unit_type))
        if status:
            sql += " AND su.status=%s"
            values.append(str(status))
        if search:
            sql += " AND su.unit_id LIKE %s ESCAPE '\\'"
            escaped = str(search).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            values.append(f"%{escaped}%")
        sql += " ORDER BY u.unit_type, su.unit_id LIMIT %s OFFSET %s"
        values.extend((page_size, max(0, int(offset))))
        with self._connection() as connection:
            rows = [dict(row) for row in connection.execute(sql, values).fetchall()]
        for row in rows:
            row["pixel_window"] = json.loads(row.pop("pixel_window_json"))
        return rows

    def stream_rows(self, run_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM streams WHERE run_id=%s ORDER BY stream_id",
                    (str(run_id),),
                ).fetchall()
            ]

    def insert_jobs(
        self, run_id: str, jobs: Iterable[Mapping[str, Any]]
    ) -> int:
        now = _now()
        count = 0

        def rows():
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

        with self.transaction() as connection:
            connection.executemany(
                """INSERT INTO jobs
                   (run_id, job_type, stream_id, tile_id, unit_id, package_id,
                    status, priority, max_attempts, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                rows(),
            )
        return count

    def get_job(self, job_id: int) -> dict[str, Any] | None:
        with self._connection() as connection:
            return _row_dict(
                connection.execute(
                    "SELECT * FROM jobs WHERE job_id=%s", (int(job_id),)
                ).fetchone()
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
        updated = connection.execute(
            """UPDATE jobs SET status='running', attempt=attempt+1,
               progress_current=0, progress_total=0,
               worker_id=%s, lease_token=%s, lease_expires=%s, heartbeat_at=%s,
               updated_at=%s WHERE job_id=%s
               AND status IN ('queued','interrupted') AND attempt < max_attempts""",
            (str(worker_id), token, expires, now, now, job_id),
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
        sql += " ORDER BY priority DESC, job_id LIMIT 1"
        sql += " FOR UPDATE SKIP LOCKED"
        with self.transaction() as connection:
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
        if int(frontier["unit_count"]) >= max(1, int(max_open_frontier_units)) and preferred:
            placeholders = ",".join("%s" for _ in preferred)
            with self._connection() as connection:
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
        with self.transaction() as connection:
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
                     ))
                   ORDER BY j.priority DESC, j.job_id LIMIT 1"""
                + lock_clause,
                (str(run_id),),
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
        with self.transaction() as connection:
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
        with self.transaction() as connection:
            return connection.execute(
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
            ).rowcount == 1

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
        with self.transaction() as connection:
            return connection.execute(
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
            ).rowcount == 1

    def complete_fragmentation_v33_job(
        self,
        job_id: int,
        lease_token: str,
    ) -> bool:
        """Atomically complete V3.3 and release all retained owner inputs."""

        now = _now()
        fence_time = time.time()
        with self.transaction() as connection:
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
                    "v33_staged_mask", "v33_staged_audit", ""
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
                            str(job["run_id"]), str(job["stream_id"]),
                            expected_owner, kind,
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
                raise ValueError("V3.3 finalize outputs require unique partition_id values")
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
                    raise ValueError("artifact sha256 must contain 64 hexadecimal characters")
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
        with self.transaction() as connection:
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
                raise RunStateError("V3.3 finalize cannot publish before all owner jobs are ready")

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
                    raise RunStateError("V3.3 atomic finalize did not create an artifact")
                if str(artifact["status"]) == "ready":
                    if (
                        int(artifact["byte_count"]) != int(byte_count)
                        or str(artifact["sha256"]) != str(digest)
                    ):
                        raise RunStateError(f"ready V3.3 {kind} changed on disk: {path}")
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
        with self._connection() as connection:
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
            raise ValueError(
                f"invalid Work Package/job transition status: {target}"
            )
        identifier = str(run_id)
        package = str(package_id)
        token = str(lease_token)
        if not token:
            return False
        now = _now()
        fence_time = time.time()
        job_error = "" if target == "ready" else str(error)
        with self.transaction() as connection:
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
        with self.transaction() as connection:
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
        with self.transaction() as connection:
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
        with self.transaction() as connection:
            return connection.execute(
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

    def interrupt_job(self, job_id: int, lease_token: str) -> bool:
        with self.transaction() as connection:
            return connection.execute(
                """UPDATE jobs SET status='interrupted', worker_id='', lease_token='',
                   lease_expires=NULL, updated_at=%s, attempt=GREATEST(0, attempt-1)
                   WHERE job_id=%s AND status='running' AND lease_token=%s""",
                (_now(), int(job_id), str(lease_token)),
            ).rowcount == 1

    def requeue_failed_job(self, job_id: int) -> bool:
        with self.transaction() as connection:
            return connection.execute(
                """UPDATE jobs SET status='queued', error='', worker_id='',
                   lease_token='', lease_expires=NULL, updated_at=%s
                   WHERE job_id=%s AND status='failed' AND attempt < max_attempts""",
                (_now(), int(job_id)),
            ).rowcount == 1

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
        with self.transaction() as connection:
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
                return connection.execute(
                    """UPDATE jobs SET status='interrupted', worker_id='',
                       lease_token='', lease_expires=NULL, updated_at=%s,
                       attempt=GREATEST(0, attempt-1)
                       WHERE status='running' AND lease_expires IS NOT NULL
                       AND lease_expires < %s""",
                    (now, now_value),
                ).rowcount
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
            return connection.execute(
                """UPDATE jobs SET status='interrupted', worker_id='',
                   lease_token='', lease_expires=NULL, updated_at=%s,
                   attempt=GREATEST(0, attempt-1)
                   WHERE run_id=%s AND status='running'
                   AND lease_expires IS NOT NULL AND lease_expires < %s""",
                (now, identifier, now_value),
            ).rowcount

    def interrupt_run_jobs(self, run_id: str) -> int:
        """Recover only the selected run after a QGIS/process interruption."""
        identifier = str(run_id)
        now = _now()
        with self.transaction() as connection:
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
            return connection.execute(
                """UPDATE jobs SET status='interrupted', worker_id='',
                   lease_token='', lease_expires=NULL, updated_at=%s,
                   attempt=GREATEST(0, attempt-1)
                   WHERE run_id=%s AND status='running'""",
                (now, identifier),
            ).rowcount

    def begin_failed_package_reset(self, run_id: str) -> dict[str, Any]:
        """Freeze and inventory every Package implicated by failed work.

        The filesystem is intentionally not touched inside the database
        transaction.  Callers delete only the returned, run-owned paths and
        then call :meth:`complete_failed_package_reset`.  If deletion is
        interrupted, calling this method again resumes the same ``resetting``
        Package set.
        """
        identifier = str(run_id)
        now = _now()
        with self.transaction() as connection:
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
    ) -> dict[str, int]:
        """Commit a completed filesystem cleanup and rebuild ready inputs."""
        identifier = str(run_id)
        expected_packages = tuple(str(item) for item in package_ids)
        if not expected_packages:
            raise RunStateError("manual Package reset requires at least one Package")
        now = _now()
        with self.transaction() as connection:
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
        with self.transaction() as connection:
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
                raise RunStateError("artifact registration did not create or find a row")
            return int(row["artifact_id"])

    def get_artifact(self, artifact_id: int) -> dict[str, Any] | None:
        with self._connection() as connection:
            return _row_dict(
                connection.execute(
                    "SELECT * FROM artifacts WHERE artifact_id=%s", (int(artifact_id),)
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
            raise ValueError("artifact sha256 must contain 64 hexadecimal characters") from error
        with self.transaction() as connection:
            return connection.execute(
                """UPDATE artifacts SET status='ready', byte_count=%s, sha256=%s, updated_at=%s
                   WHERE artifact_id=%s AND status IN ('writing','failed')""",
                (int(byte_count), str(sha256).lower(), _now(), int(artifact_id)),
            ).rowcount == 1

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
        with self.transaction() as connection:
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
            raise ValueError("unit confidence sha256 must contain 64 hexadecimal characters")
        try:
            int(digest, 16)
        except ValueError as error:
            raise ValueError(
                "unit confidence sha256 must contain 64 hexadecimal characters"
            ) from error
        resolved_path = str(Path(path).expanduser().resolve())
        now = _now()
        fence_time = time.time()
        with self.transaction() as connection:
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
                raise RunStateError(
                    "unit confidence registration did not create a row"
                )
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
        with self.transaction() as connection:
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
                    raise RunStateError(
                        f"cannot republish {label}: {resolved_path}"
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
                        f"cannot publish {label}: {resolved_path}"
                    )
            else:
                raise RunStateError(
                    f"{label} is unavailable for publish ({status}): "
                    + resolved_path
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
                raise ValueError("artifact sha256 must contain 64 hexadecimal characters")
            try:
                int(digest, 16)
            except ValueError as error:
                raise ValueError(
                    "artifact sha256 must contain 64 hexadecimal characters"
                ) from error
        now = _now()
        artifact_ids: list[int] = []
        with self.transaction() as connection:
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
                for artifact_id in artifact_ids:
                    connection.execute(
                        """INSERT INTO artifact_dependencies
                           (job_id, artifact_id, created_at) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING""",
                        (finalize_job_id, artifact_id, now),
                    )
        return artifact_ids[0], artifact_ids[1]

    def mark_artifact_failed(self, artifact_id: int) -> bool:
        with self.transaction() as connection:
            return connection.execute(
                """UPDATE artifacts SET status='failed', updated_at=%s
                   WHERE artifact_id=%s AND status='writing'""",
                (_now(), int(artifact_id)),
            ).rowcount == 1

    def add_artifact_dependency(self, job_id: int, artifact_id: int) -> bool:
        """Attach a ready input to a job; the trigger updates ref_count atomically."""
        with self.transaction() as connection:
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
            return cursor.rowcount == 1

    def link_fragmentation_v33_input(
        self,
        run_id: str,
        stream_id: str,
        partition_id: str,
        artifact_id: int,
    ) -> int:
        """Attach one ready owner input to every waiting V3.3 candidate."""

        identifier = str(run_id)
        stream = str(stream_id)
        partition = str(partition_id)
        now = _now()
        with self.transaction() as connection:
            artifact = connection.execute(
                """SELECT kind FROM artifacts
                   WHERE artifact_id=%s AND run_id=%s AND stream_id=%s
                     AND unit_id=%s AND status='ready'
                     AND kind IN ('partition_probability','v3_context_core',
                                  'v3_baseline_core')""",
                (int(artifact_id), identifier, stream, partition),
            ).fetchone()
            if artifact is None:
                raise RunStateError(
                    "V3.3 dependency Artifact is not ready or mismatched"
                )
            cursor = connection.execute(
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
                (int(artifact_id), now, identifier, stream, partition),
            )
            return cursor.rowcount

    def link_partition_artifact(
        self,
        run_id: str,
        stream_id: str,
        partition_id: str,
        artifact_id: int,
    ) -> int:
        """Link one ready Partition probability to every dependent unit job."""
        with self.transaction() as connection:
            artifact = connection.execute(
                """SELECT 1 FROM artifacts WHERE artifact_id=%s AND run_id=%s
                   AND stream_id=%s AND unit_id=%s AND kind='partition_probability'
                   AND status='ready'"""
                + " FOR UPDATE",
                (int(artifact_id), str(run_id), str(stream_id), str(partition_id)),
            ).fetchone()
            if artifact is None:
                raise RunStateError("Partition dependency Artifact is not ready or mismatched")
            job_ids = [
                int(row["job_id"])
                for row in connection.execute(
                    """SELECT j.job_id FROM jobs j
                       JOIN unit_dependencies d
                         ON d.run_id=j.run_id AND d.unit_id=j.unit_id
                       WHERE j.run_id=%s AND j.stream_id=%s AND j.job_type='unit_fit'
                         AND d.partition_id=%s""",
                    (str(run_id), str(stream_id), str(partition_id)),
                ).fetchall()
            ]
            now = _now()
            inserted = 0
            for job_id in job_ids:
                cursor = connection.execute(
                    """INSERT INTO artifact_dependencies
                       (job_id, artifact_id, created_at) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING""",
                    (job_id, int(artifact_id), now),
                )
                inserted += cursor.rowcount
            return inserted

    def release_artifact_dependency(self, job_id: int, artifact_id: int) -> bool:
        """Release one job input; the trigger prevents a negative ref_count."""
        with self.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM artifact_dependencies WHERE job_id=%s AND artifact_id=%s",
                (int(job_id), int(artifact_id)),
            )
            return cursor.rowcount == 1

    def release_job_artifacts(self, job_id: int) -> int:
        with self.transaction() as connection:
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
            return connection.execute(
                "DELETE FROM artifact_dependencies WHERE job_id=%s", (int(job_id),)
            ).rowcount

    def job_for_unit(
        self, run_id: str, stream_id: str, unit_id: str
    ) -> dict[str, Any] | None:
        with self._connection() as connection:
            return _row_dict(
                connection.execute(
                    """SELECT * FROM jobs WHERE run_id=%s AND stream_id=%s
                       AND unit_id=%s AND job_type='unit_fit'""",
                    (str(run_id), str(stream_id), str(unit_id)),
                ).fetchone()
            )

    def cleanup_candidates(
        self,
        run_id: str,
        *,
        limit: int = 100,
        kinds: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        """Return ready, unreferenced artifacts; deletion remains an explicit caller action."""
        sql = "SELECT * FROM artifacts WHERE run_id=%s AND status='ready' AND ref_count=0"
        values: list[Any] = [str(run_id)]
        if kinds:
            sql += " AND kind IN (" + ",".join("%s" for _ in kinds) + ")"
            values.extend(str(item) for item in kinds)
        sql += " ORDER BY artifact_id LIMIT %s"
        values.append(max(1, min(int(limit), 1000)))
        with self._connection() as connection:
            return [dict(row) for row in connection.execute(sql, values).fetchall()]

    def claim_artifact_cleanup(self, artifact_id: int) -> dict[str, Any] | None:
        """Atomically reserve one unreferenced ready Artifact for deletion."""
        with self.transaction() as connection:
            changed = connection.execute(
                """UPDATE artifacts SET status='cleaning', updated_at=%s
                   WHERE artifact_id=%s AND status='ready' AND ref_count=0""",
                (_now(), int(artifact_id)),
            ).rowcount
            if changed != 1:
                return None
            return _row_dict(
                connection.execute(
                    "SELECT * FROM artifacts WHERE artifact_id=%s", (int(artifact_id),)
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
        with self.transaction() as connection:
            return connection.execute(
                """UPDATE artifacts SET status=%s, updated_at=%s
                   WHERE artifact_id=%s AND status='cleaning' AND ref_count=0""",
                (next_status, _now(), int(artifact_id)),
            ).rowcount == 1

    def artifact_cleanup_summary(self, run_id: str) -> dict[str, int]:
        with self._connection() as connection:
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
        with self._connection() as connection:
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
        with self._connection() as connection:
            return [dict(row) for row in connection.execute(sql, values).fetchall()]

    def upsert_unit_report_summary(
        self,
        run_id: str,
        stream_id: str,
        unit_id: str,
        report: Mapping[str, Any],
        *,
        fitted_edge_count: int = 0,
    ) -> None:
        """Persist scalar report evidence after its JSON Artifact is ready."""
        artifact = self.artifact_for_stream_unit(
            run_id,
            stream_id,
            unit_id,
            "unit_boundary_report",
        )
        if artifact is None:
            raise RunStateError(
                "unit report summary requires a ready unit_boundary_report Artifact"
            )
        report_path = Path(str(artifact["path"]))
        if not report_path.is_file():
            raise RunStateError(f"unit report Artifact is missing: {report_path}")
        stat = report_path.stat()
        if int(artifact["byte_count"]) != int(stat.st_size):
            raise RunStateError(f"unit report Artifact size changed: {report_path}")
        diagnostics = report.get("diagnostics") or []
        if not isinstance(diagnostics, list):
            raise RunStateError("unit boundary report diagnostics must be a list")
        edge_count = int(fitted_edge_count)
        if edge_count < 0 or edge_count > len(diagnostics):
            raise RunStateError(
                "unit fitted edge count is outside the diagnostic report range"
            )
        now = _now()
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO unit_report_summaries
                   (run_id, stream_id, unit_id, status, fit_version,
                    chain_count, shared_chain_count, spline_count,
                    unchanged_count, skipped_invalid_count,
                    max_displacement_px, diagnostic_count, fitted_edge_count,
                    report_path, report_byte_count, report_sha256,
                    report_mtime_ns, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT(run_id, stream_id, unit_id) DO UPDATE SET
                     status=excluded.status,
                     fit_version=excluded.fit_version,
                     chain_count=excluded.chain_count,
                     shared_chain_count=excluded.shared_chain_count,
                     spline_count=excluded.spline_count,
                     unchanged_count=excluded.unchanged_count,
                     skipped_invalid_count=excluded.skipped_invalid_count,
                     max_displacement_px=excluded.max_displacement_px,
                     diagnostic_count=excluded.diagnostic_count,
                     fitted_edge_count=excluded.fitted_edge_count,
                     report_path=excluded.report_path,
                     report_byte_count=excluded.report_byte_count,
                     report_sha256=excluded.report_sha256,
                     report_mtime_ns=excluded.report_mtime_ns,
                     updated_at=excluded.updated_at""",
                (
                    str(run_id),
                    str(stream_id),
                    str(unit_id),
                    str(report.get("status") or ""),
                    str(report.get("fit_version") or ""),
                    int(report.get("chain_count", 0)),
                    int(report.get("shared_chain_count", 0)),
                    int(report.get("spline_count", 0)),
                    int(report.get("unchanged_count", 0)),
                    int(report.get("skipped_invalid_count", 0)),
                    float(report.get("max_displacement_px", 0.0)),
                    len(diagnostics),
                    edge_count,
                    str(report_path.resolve()),
                    int(stat.st_size),
                    str(artifact["sha256"]),
                    int(stat.st_mtime_ns),
                    now,
                    now,
                ),
            )

    def unit_report_summaries(
        self,
        run_id: str,
        stream_id: str,
    ) -> list[dict[str, Any]]:
        """Return summary rows, or no rows when the current contract is incomplete."""
        with self._connection() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    """SELECT * FROM unit_report_summaries
                       WHERE run_id=%s AND stream_id=%s ORDER BY unit_id""",
                    (str(run_id), str(stream_id)),
                ).fetchall()
            ]

    def unit_report_summary_aggregate(
        self,
        run_id: str,
        stream_id: str,
    ) -> dict[str, Any]:
        """Aggregate report scalars in the state database without loading JSON."""
        with self._connection() as connection:
            row = connection.execute(
                """SELECT COUNT(*) AS unit_count,
                          COALESCE(SUM(chain_count), 0) AS chain_count,
                          COALESCE(SUM(shared_chain_count), 0)
                            AS shared_chain_count,
                          COALESCE(SUM(spline_count), 0) AS spline_count,
                          COALESCE(SUM(unchanged_count), 0) AS unchanged_count,
                          COALESCE(SUM(skipped_invalid_count), 0)
                            AS skipped_invalid_count,
                          COALESCE(SUM(CASE WHEN status='passed' THEN 0 ELSE 1 END), 0)
                            AS failed_unit_count,
                          COALESCE(MAX(max_displacement_px), 0)
                            AS max_displacement_px,
                          COALESCE(SUM(diagnostic_count), 0)
                            AS diagnostic_count,
                          COALESCE(SUM(fitted_edge_count), 0)
                            AS fitted_edge_count
                   FROM unit_report_summaries
                   WHERE run_id=%s AND stream_id=%s""",
                (str(run_id), str(stream_id)),
            ).fetchone()
        return dict(row)

    def object_ids_for_parts(
        self,
        run_id: str,
        stream_id: str,
        part_ids: Sequence[str],
    ) -> dict[str, str]:
        """Resolve a bounded part batch using one connection, not one per feature."""
        values = [str(part_id) for part_id in part_ids]
        if not values:
            return {}
        result: dict[str, str] = {}
        with self._connection() as connection:
            for offset in range(0, len(values), 400):
                batch = values[offset : offset + 400]
                placeholders = ",".join("%s" for _ in batch)
                rows = connection.execute(
                    f"""SELECT part_id, object_id FROM object_nodes
                        WHERE run_id=%s AND stream_id=%s
                          AND part_id IN ({placeholders})""",
                    (str(run_id), str(stream_id), *batch),
                ).fetchall()
                result.update(
                    {
                        str(row["part_id"]): str(row["object_id"])
                        for row in rows
                        if row["object_id"]
                    }
                )
        missing = [part_id for part_id in values if part_id not in result]
        if missing:
            raise RunStateError(
                f"object components are unresolved: {missing[:3]}"
            )
        return result

    def register_object_parts(
        self,
        run_id: str,
        stream_id: str,
        parts: Iterable[Mapping[str, Any]],
    ) -> int:
        now = _now()
        rows = [
            (
                str(run_id),
                str(stream_id),
                str(item["part_id"]),
                int(item["class_code"]),
                str(item["unit_id"]),
                str(item["part_id"]),
                now,
                now,
            )
            for item in parts
        ]
        with self.transaction() as connection:
            connection.executemany(
                """INSERT INTO object_nodes
                   (run_id, stream_id, part_id, class_code, unit_id,
                    parent_id, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""",
                rows,
            )
            return len(rows)

    def add_object_link(
        self,
        run_id: str,
        stream_id: str,
        left_part_id: str,
        right_part_id: str,
        class_code: int,
    ) -> bool:
        left, right = sorted((str(left_part_id), str(right_part_id)))
        if left == right:
            return False
        with self.transaction() as connection:
            rows = connection.execute(
                """SELECT part_id, class_code FROM object_nodes
                   WHERE run_id=%s AND stream_id=%s AND part_id IN (%s, %s)""",
                (str(run_id), str(stream_id), left, right),
            ).fetchall()
            if len(rows) != 2 or any(int(row["class_code"]) != int(class_code) for row in rows):
                raise RunStateError("object link parts are missing or have different classes")
            cursor = connection.execute(
                """INSERT INTO object_links
                   (run_id, stream_id, left_part_id, right_part_id, class_code, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""",
                (str(run_id), str(stream_id), left, right, int(class_code), _now()),
            )
            return cursor.rowcount == 1

    def object_link_count(self, run_id: str, stream_id: str) -> int:
        with self._connection() as connection:
            return int(
                connection.execute(
                    """SELECT COUNT(*) FROM object_links
                       WHERE run_id=%s AND stream_id=%s""",
                    (str(run_id), str(stream_id)),
                ).fetchone()[0]
            )

    def resolve_object_components(self, run_id: str, stream_id: str) -> int:
        """Resolve object links with an in-memory union-find and deterministic IDs."""
        import hashlib

        with self.transaction() as connection:
            rows = connection.execute(
                """SELECT part_id, parent_id, rank_value FROM object_nodes
                   WHERE run_id=%s AND stream_id=%s ORDER BY part_id""",
                (str(run_id), str(stream_id)),
            ).fetchall()
            if not rows:
                return 0

            parent: dict[str, str] = {
                str(row["part_id"]): str(row["parent_id"] or row["part_id"])
                for row in rows
            }
            rank: dict[str, int] = {
                str(row["part_id"]): int(row["rank_value"] or 0)
                for row in rows
            }

            def find(part_id: str) -> str:
                path = []
                curr = part_id
                while parent.get(curr, curr) != curr:
                    path.append(curr)
                    curr = parent[curr]
                for child in path:
                    parent[child] = curr
                return curr

            links = connection.execute(
                """SELECT left_part_id, right_part_id FROM object_links
                   WHERE run_id=%s AND stream_id=%s ORDER BY left_part_id, right_part_id""",
                (str(run_id), str(stream_id)),
            ).fetchall()

            for link in links:
                left = str(link["left_part_id"])
                right = str(link["right_part_id"])
                if left not in parent or right not in parent:
                    continue
                root_left = find(left)
                root_right = find(right)
                if root_left == root_right:
                    continue
                rank_left = rank[root_left]
                rank_right = rank[root_right]
                if rank_left < rank_right or (
                    rank_left == rank_right and root_left > root_right
                ):
                    root_left, root_right = root_right, root_left
                    rank_left, rank_right = rank_right, rank_left
                parent[root_right] = root_left
                if rank_left == rank_right:
                    rank[root_left] = rank_left + 1

            now = _now()
            update_rows = []
            distinct_roots = set()
            for part_id in parent:
                root = find(part_id)
                distinct_roots.add(root)
                digest = hashlib.sha1(
                    f"{run_id}|{stream_id}|{root}".encode("utf-8")
                ).hexdigest()[:24]
                obj_id = f"obj_{digest}"
                update_rows.append(
                    (
                        root,
                        rank.get(part_id, 0),
                        obj_id,
                        now,
                        str(run_id),
                        str(stream_id),
                        part_id,
                    )
                )

            for offset in range(0, len(update_rows), 2000):
                chunk = update_rows[offset : offset + 2000]
                connection.executemany(
                    """UPDATE object_nodes
                       SET parent_id=%s, rank_value=%s, object_id=%s, updated_at=%s
                       WHERE run_id=%s AND stream_id=%s AND part_id=%s""",
                    chunk,
                )
            return len(distinct_roots)

    def object_id_for_part(
        self, run_id: str, stream_id: str, part_id: str
    ) -> str:
        with self._connection() as connection:
            row = connection.execute(
                """SELECT object_id FROM object_nodes
                   WHERE run_id=%s AND stream_id=%s AND part_id=%s""",
                (str(run_id), str(stream_id), str(part_id)),
            ).fetchone()
        if row is None or not row["object_id"]:
            raise RunStateError(f"object component is unresolved: {part_id}")
        return str(row["object_id"])

    def artifact_for_stream_unit(
        self,
        run_id: str,
        stream_id: str,
        unit_id: str,
        kind: str,
        *,
        status: str = "ready",
    ) -> dict[str, Any] | None:
        with self._connection() as connection:
            return _row_dict(
                connection.execute(
                    """SELECT * FROM artifacts WHERE run_id=%s AND stream_id=%s
                       AND unit_id=%s AND kind=%s AND status=%s""",
                    (str(run_id), str(stream_id), str(unit_id), str(kind), str(status)),
                ).fetchone()
            )

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
        with self.transaction() as connection:
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
        with self._connection() as connection:
            return {
                str(row["status"]): int(row["n"])
                for row in connection.execute(sql, values).fetchall()
            }
