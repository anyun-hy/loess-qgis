"""Fast source-level checks for the production PostgreSQL state contract."""

from __future__ import annotations

import pytest

from labeling_tool.core.postgres_state import (
    DEFAULT_POSTGRES_DSN,
    DEFAULT_POSTGRES_SCHEMA,
    POSTGRES_SCHEMA_SQL,
    _postgres_sql,
    is_postgres_location,
    validate_schema,
)
from labeling_tool.core.run_state_db import (
    RunStateDB,
    production_state_database,
    production_state_schema,
)


def test_production_postgres_defaults_use_peer_socket_without_password(monkeypatch):
    monkeypatch.delenv("LOESS_STATE_DB_DSN", raising=False)
    monkeypatch.delenv("LOESS_STATE_DB_SCHEMA", raising=False)

    assert production_state_database() == DEFAULT_POSTGRES_DSN
    assert production_state_schema() == DEFAULT_POSTGRES_SCHEMA
    assert "host=/var/run/postgresql" in DEFAULT_POSTGRES_DSN
    assert "password=" not in DEFAULT_POSTGRES_DSN.lower()
    assert is_postgres_location(DEFAULT_POSTGRES_DSN)

    database = RunStateDB(DEFAULT_POSTGRES_DSN)
    assert database.backend == "postgresql"
    assert database.path is None
    assert database.postgres_schema == DEFAULT_POSTGRES_SCHEMA


def test_postgres_environment_override_and_schema_validation(monkeypatch):
    dsn = "postgresql://anyun@localhost/anyun"
    monkeypatch.setenv("LOESS_STATE_DB_DSN", dsn)
    monkeypatch.setenv("LOESS_STATE_DB_SCHEMA", "loess_isolated")

    assert production_state_database() == dsn
    assert production_state_schema() == "loess_isolated"
    assert RunStateDB(dsn).postgres_schema == "loess_isolated"
    with pytest.raises(ValueError, match="invalid PostgreSQL schema"):
        validate_schema('loess_qgis; DROP SCHEMA public')


def test_postgres_sql_adapter_preserves_compare_and_set_semantics():
    assert _postgres_sql("SELECT * FROM jobs WHERE job_id=?") == (
        "SELECT * FROM jobs WHERE job_id=%s"
    )
    assert _postgres_sql(
        "INSERT OR IGNORE INTO artifact_dependencies VALUES (?, ?, ?)"
    ) == (
        "INSERT INTO artifact_dependencies VALUES (%s, %s, %s) "
        "ON CONFLICT DO NOTHING"
    )
    assert _postgres_sql("UPDATE jobs SET attempt=MAX(0, attempt-1)") == (
        "UPDATE jobs SET attempt=GREATEST(0, attempt-1)"
    )


def test_postgres_schema_has_concurrent_control_plane_primitives():
    required_tables = (
        "runs",
        "streams",
        "work_packages",
        "partitions",
        "tiles",
        "spatial_units",
        "stream_units",
        "jobs",
        "artifacts",
        "artifact_dependencies",
        "events",
    )
    for table in required_tables:
        assert f"CREATE TABLE IF NOT EXISTS {table}" in POSTGRES_SCHEMA_SQL
    assert "BIGSERIAL PRIMARY KEY" in POSTGRES_SCHEMA_SQL
    assert "idx_jobs_claim_type" in POSTGRES_SCHEMA_SQL
    assert "artifact_dependency_after_insert" in POSTGRES_SCHEMA_SQL
    assert "artifact_dependency_after_delete" in POSTGRES_SCHEMA_SQL
