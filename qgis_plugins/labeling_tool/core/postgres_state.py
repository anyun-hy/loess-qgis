"""PostgreSQL connection and schema support for the v5 run-state store.

The production control plane uses PostgreSQL so independent QGIS, accelerator,
and geometry processes can write different rows concurrently.  The adapter
keeps the small DB-API surface used by :mod:`run_state_db` while translating
the legacy qmark SQL parameters during the backend migration.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence


DEFAULT_POSTGRES_DSN = (
    "dbname=anyun user=anyun host=/var/run/postgresql port=5432"
)
DEFAULT_POSTGRES_SCHEMA = "loess_qgis"
POSTGRES_SCHEME_PREFIXES = ("postgresql://", "postgres://")
_SAFE_SCHEMA = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


class PostgresDependencyError(RuntimeError):
    pass


def is_postgres_location(value: str | Any) -> bool:
    text = str(value or "").strip().lower()
    return text.startswith(POSTGRES_SCHEME_PREFIXES) or any(
        token in text for token in ("dbname=", "host=", "user=", "port=")
    )


def validate_schema(schema: str) -> str:
    value = str(schema or DEFAULT_POSTGRES_SCHEMA).strip().lower()
    if not _SAFE_SCHEMA.fullmatch(value):
        raise ValueError(f"invalid PostgreSQL schema name: {schema!r}")
    return value


def _driver():
    try:
        import psycopg2
        from psycopg2.extras import DictCursor, execute_batch
    except ImportError as error:
        raise PostgresDependencyError(
            "PostgreSQL run state requires psycopg2 in the active Python "
            "environment"
        ) from error
    return psycopg2, DictCursor, execute_batch


def _postgres_sql(statement: str) -> str:
    """Translate the SQLite-compatible routine SQL used by RunStateDB."""

    value = str(statement)
    ignored = bool(re.search(r"\bINSERT\s+OR\s+IGNORE\b", value, re.I))
    if ignored:
        value = re.sub(r"\bINSERT\s+OR\s+IGNORE\b", "INSERT", value, flags=re.I)
    value = re.sub(r"\bMAX\s*\(\s*0\s*,", "GREATEST(0,", value, flags=re.I)
    value = value.replace("?", "%s")
    if ignored:
        stripped = value.rstrip().rstrip(";")
        if " ON CONFLICT " not in stripped.upper():
            value = stripped + " ON CONFLICT DO NOTHING"
    return value


class PostgresConnection:
    """Small connection facade exposing SQLite-style ``execute`` helpers."""

    def __init__(self, raw: Any, cursor_factory: Any, execute_batch: Any):
        self.raw = raw
        self._cursor_factory = cursor_factory
        self._execute_batch = execute_batch

    def execute(
        self,
        statement: str,
        parameters: Sequence[Any] | None = None,
    ):
        cursor = self.raw.cursor(cursor_factory=self._cursor_factory)
        cursor.execute(_postgres_sql(statement), tuple(parameters or ()))
        return cursor

    def executemany(
        self,
        statement: str,
        parameters: Iterable[Sequence[Any]],
    ):
        cursor = self.raw.cursor(cursor_factory=self._cursor_factory)
        self._execute_batch(
            cursor,
            _postgres_sql(statement),
            parameters,
            page_size=1000,
        )
        return cursor

    def commit(self) -> None:
        self.raw.commit()

    def rollback(self) -> None:
        self.raw.rollback()

    def close(self) -> None:
        self.raw.close()


def connect_postgres(
    dsn: str,
    *,
    schema: str = DEFAULT_POSTGRES_SCHEMA,
    autocommit: bool,
) -> PostgresConnection:
    psycopg2, dict_cursor, execute_batch = _driver()
    raw = psycopg2.connect(
        str(dsn),
        connect_timeout=10,
        application_name="loess-qgis",
    )
    raw.autocommit = bool(autocommit)
    safe_schema = validate_schema(schema)
    with raw.cursor() as cursor:
        cursor.execute(
            f'SET search_path TO "{safe_schema}", public'
        )
        cursor.execute("SET lock_timeout TO '60s'")
    return PostgresConnection(raw, dict_cursor, execute_batch)


POSTGRES_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_metadata (
    name TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    status TEXT NOT NULL,
    run_spec_sha256 TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS streams (
    run_id TEXT NOT NULL,
    stream_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    model_id TEXT NOT NULL DEFAULT '',
    profile_id TEXT NOT NULL DEFAULT '',
    version TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, stream_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS work_packages (
    run_id TEXT NOT NULL,
    package_id TEXT NOT NULL,
    sequence_no INTEGER NOT NULL,
    estimated_bytes BIGINT NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'queued',
    attempt INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, package_id),
    UNIQUE (run_id, sequence_no),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS partitions (
    run_id TEXT NOT NULL,
    partition_id TEXT NOT NULL,
    row_no INTEGER NOT NULL,
    col_no INTEGER NOT NULL,
    core_window_json TEXT NOT NULL,
    halo_window_json TEXT NOT NULL,
    package_id TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, partition_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (run_id, package_id)
        REFERENCES work_packages(run_id, package_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS tiles (
    run_id TEXT NOT NULL,
    tile_id TEXT NOT NULL,
    row_no INTEGER NOT NULL,
    col_no INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    pixel_window_json TEXT NOT NULL DEFAULT '{}',
    bounds_json TEXT NOT NULL DEFAULT '{}',
    raster_path TEXT NOT NULL DEFAULT '',
    sha256 TEXT NOT NULL DEFAULT '',
    partition_id TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, tile_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (run_id, partition_id)
        REFERENCES partitions(run_id, partition_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS spatial_units (
    run_id TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    unit_type TEXT NOT NULL,
    owner_key TEXT NOT NULL,
    pixel_window_json TEXT NOT NULL,
    dependency_ids_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'queued',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, unit_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS unit_dependencies (
    run_id TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    partition_id TEXT NOT NULL,
    PRIMARY KEY (run_id, unit_id, partition_id),
    FOREIGN KEY (run_id, unit_id)
        REFERENCES spatial_units(run_id, unit_id) ON DELETE CASCADE,
    FOREIGN KEY (run_id, partition_id)
        REFERENCES partitions(run_id, partition_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS stream_units (
    run_id TEXT NOT NULL,
    stream_id TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, stream_id, unit_id),
    FOREIGN KEY (run_id, stream_id)
        REFERENCES streams(run_id, stream_id) ON DELETE CASCADE,
    FOREIGN KEY (run_id, unit_id)
        REFERENCES spatial_units(run_id, unit_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    job_type TEXT NOT NULL,
    stream_id TEXT NOT NULL DEFAULT '',
    tile_id TEXT NOT NULL DEFAULT '',
    unit_id TEXT NOT NULL DEFAULT '',
    package_id TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'queued',
    priority INTEGER NOT NULL DEFAULT 0,
    attempt INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    progress_current BIGINT NOT NULL DEFAULT 0,
    progress_total BIGINT NOT NULL DEFAULT 0,
    worker_id TEXT NOT NULL DEFAULT '',
    lease_token TEXT NOT NULL DEFAULT '',
    lease_expires DOUBLE PRECISION,
    heartbeat_at TEXT,
    pid BIGINT,
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    stream_id TEXT NOT NULL DEFAULT '',
    unit_id TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    byte_count BIGINT NOT NULL DEFAULT 0,
    sha256 TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'writing',
    ref_count BIGINT NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (run_id, stream_id, unit_id, kind, path),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS artifact_dependencies (
    job_id BIGINT NOT NULL,
    artifact_id BIGINT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (job_id, artifact_id),
    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE,
    FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS unit_report_summaries (
    run_id TEXT NOT NULL,
    stream_id TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    status TEXT NOT NULL,
    fit_version TEXT NOT NULL DEFAULT '',
    chain_count BIGINT NOT NULL DEFAULT 0,
    shared_chain_count BIGINT NOT NULL DEFAULT 0,
    spline_count BIGINT NOT NULL DEFAULT 0,
    unchanged_count BIGINT NOT NULL DEFAULT 0,
    skipped_invalid_count BIGINT NOT NULL DEFAULT 0,
    max_displacement_px DOUBLE PRECISION NOT NULL DEFAULT 0,
    diagnostic_count BIGINT NOT NULL DEFAULT 0,
    fitted_edge_count BIGINT NOT NULL DEFAULT 0,
    report_path TEXT NOT NULL,
    report_byte_count BIGINT NOT NULL,
    report_sha256 TEXT NOT NULL,
    report_mtime_ns BIGINT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, stream_id, unit_id),
    FOREIGN KEY (run_id, stream_id, unit_id)
        REFERENCES stream_units(run_id, stream_id, unit_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS object_links (
    run_id TEXT NOT NULL,
    stream_id TEXT NOT NULL,
    left_part_id TEXT NOT NULL,
    right_part_id TEXT NOT NULL,
    class_code INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, stream_id, left_part_id, right_part_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS object_nodes (
    run_id TEXT NOT NULL,
    stream_id TEXT NOT NULL,
    part_id TEXT NOT NULL,
    class_code INTEGER NOT NULL,
    unit_id TEXT NOT NULL,
    parent_id TEXT NOT NULL,
    rank_value INTEGER NOT NULL DEFAULT 0,
    object_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, stream_id, part_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS events (
    event_id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    level TEXT NOT NULL,
    event_type TEXT NOT NULL,
    stream_id TEXT NOT NULL DEFAULT '',
    job_id BIGINT,
    message TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_streams_status
    ON streams(run_id, status, stream_id);
CREATE INDEX IF NOT EXISTS idx_packages_status
    ON work_packages(run_id, status, sequence_no);
CREATE INDEX IF NOT EXISTS idx_partitions_status
    ON partitions(run_id, status, package_id);
CREATE INDEX IF NOT EXISTS idx_tiles_grid_status
    ON tiles(run_id, row_no, col_no, status);
CREATE INDEX IF NOT EXISTS idx_tiles_partition
    ON tiles(run_id, partition_id, status);
CREATE INDEX IF NOT EXISTS idx_units_status
    ON spatial_units(run_id, unit_type, status, unit_id);
CREATE INDEX IF NOT EXISTS idx_unit_dependencies_partition
    ON unit_dependencies(run_id, partition_id, unit_id);
CREATE INDEX IF NOT EXISTS idx_stream_units_status
    ON stream_units(run_id, stream_id, status, unit_id);
CREATE INDEX IF NOT EXISTS idx_jobs_claim
    ON jobs(run_id, status, priority DESC, job_id);
CREATE INDEX IF NOT EXISTS idx_jobs_claim_type
    ON jobs(run_id, job_type, status, priority DESC, job_id);
CREATE INDEX IF NOT EXISTS idx_jobs_stream_unit
    ON jobs(run_id, stream_id, unit_id, status);
CREATE INDEX IF NOT EXISTS idx_jobs_tile
    ON jobs(run_id, stream_id, tile_id, status);
CREATE INDEX IF NOT EXISTS idx_jobs_monitor
    ON jobs(run_id, job_type, status);
CREATE INDEX IF NOT EXISTS idx_artifacts_state
    ON artifacts(run_id, status, ref_count, artifact_id);
CREATE INDEX IF NOT EXISTS idx_unit_report_summaries_stream
    ON unit_report_summaries(run_id, stream_id, status, unit_id);
CREATE INDEX IF NOT EXISTS idx_object_nodes_root
    ON object_nodes(run_id, stream_id, parent_id, part_id);
CREATE INDEX IF NOT EXISTS idx_events_time
    ON events(run_id, event_id, timestamp);

CREATE OR REPLACE FUNCTION artifact_dependency_refcount_insert()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    UPDATE artifacts
       SET ref_count=ref_count + 1, updated_at=NEW.created_at
     WHERE artifact_id=NEW.artifact_id;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION artifact_dependency_refcount_delete()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    UPDATE artifacts
       SET ref_count=GREATEST(0, ref_count - 1),
           updated_at=clock_timestamp()::text
     WHERE artifact_id=OLD.artifact_id;
    RETURN OLD;
END;
$$;

DO $trigger$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
         WHERE tgname='artifact_dependency_after_insert'
           AND tgrelid='artifact_dependencies'::regclass
    ) THEN
        CREATE TRIGGER artifact_dependency_after_insert
        AFTER INSERT ON artifact_dependencies
        FOR EACH ROW EXECUTE FUNCTION artifact_dependency_refcount_insert();
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
         WHERE tgname='artifact_dependency_after_delete'
           AND tgrelid='artifact_dependencies'::regclass
    ) THEN
        CREATE TRIGGER artifact_dependency_after_delete
        AFTER DELETE ON artifact_dependencies
        FOR EACH ROW EXECUTE FUNCTION artifact_dependency_refcount_delete();
    END IF;
END;
$trigger$;
"""


def initialize_postgres(
    dsn: str,
    *,
    schema: str,
    schema_version: int,
    now: str,
) -> None:
    psycopg2, _dict_cursor, _execute_batch = _driver()
    safe_schema = validate_schema(schema)
    raw = psycopg2.connect(
        str(dsn),
        connect_timeout=10,
        application_name="loess-qgis-schema",
    )
    raw.autocommit = True
    try:
        with raw.cursor() as cursor:
            cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{safe_schema}"')
            cursor.execute(f'SET search_path TO "{safe_schema}", public')
            cursor.execute(
                "SELECT pg_advisory_lock(hashtext(%s))",
                (f"loess-qgis:{safe_schema}:schema",),
            )
            try:
                cursor.execute(POSTGRES_SCHEMA_SQL)
                cursor.execute(
                    "SELECT value FROM schema_metadata WHERE name='schema_version'"
                )
                row = cursor.fetchone()
                if row is not None and int(row[0]) != int(schema_version):
                    raise RuntimeError(
                        "unsupported PostgreSQL run-state schema "
                        f"{row[0]}; expected {schema_version}"
                    )
                cursor.execute(
                    """INSERT INTO schema_metadata(name,value,updated_at)
                       VALUES ('schema_version',%s,%s)
                       ON CONFLICT(name) DO UPDATE SET
                         value=EXCLUDED.value,updated_at=EXCLUDED.updated_at""",
                    (str(int(schema_version)), str(now)),
                )
            finally:
                cursor.execute(
                    "SELECT pg_advisory_unlock(hashtext(%s))",
                    (f"loess-qgis:{safe_schema}:schema",),
                )
    finally:
        raw.close()


def postgres_health(
    dsn: str,
    *,
    schema: str,
    schema_version: int,
) -> dict[str, Any]:
    connection = connect_postgres(dsn, schema=schema, autocommit=True)
    try:
        row = connection.execute(
            """SELECT current_database() AS database_name,
                      current_user AS user_name,
                      current_setting('server_version') AS server_version"""
        ).fetchone()
        version = connection.execute(
            "SELECT value FROM schema_metadata WHERE name='schema_version'"
        ).fetchone()
        if version is None or int(version[0]) != int(schema_version):
            raise RuntimeError(
                "PostgreSQL run-state schema version is missing or incompatible"
            )
        return {
            "backend": "postgresql",
            "database": str(row["database_name"]),
            "user": str(row["user_name"]),
            "server_version": str(row["server_version"]),
            "schema": validate_schema(schema),
            "schema_version": int(version[0]),
            "integrity_check": "ok",
        }
    finally:
        connection.close()
