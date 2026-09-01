"""Fast source-level checks for the production PostgreSQL state contract."""

from __future__ import annotations

import getpass
import pytest

from labeling_tool.core.postgres_state import (
    DEFAULT_POSTGRES_DSN,
    DEFAULT_POSTGRES_SCHEMA,
    POSTGRES_SCHEMA_SQL,
    is_postgres_location,
    validate_schema,
)
from labeling_tool.core.run_state_db import (
    RunStateDB,
    RunStateError,
    production_state_database,
    production_state_schema,
    run_state_from_spec,
)


def test_production_postgres_defaults_use_peer_socket_without_password(monkeypatch):
    monkeypatch.delenv("LOESS_STATE_DB_DSN", raising=False)
    monkeypatch.delenv("LOESS_STATE_DB_SCHEMA", raising=False)

    assert production_state_database() == DEFAULT_POSTGRES_DSN
    assert production_state_schema() == DEFAULT_POSTGRES_SCHEMA
    assert "host=/var/run/postgresql" in DEFAULT_POSTGRES_DSN
    assert f"dbname={getpass.getuser()}" in DEFAULT_POSTGRES_DSN
    assert f"user={getpass.getuser()}" in DEFAULT_POSTGRES_DSN
    assert "password=" not in DEFAULT_POSTGRES_DSN.lower()
    assert is_postgres_location(DEFAULT_POSTGRES_DSN)

    database = RunStateDB(DEFAULT_POSTGRES_DSN)
    assert database.location == DEFAULT_POSTGRES_DSN
    assert database.postgres_schema == DEFAULT_POSTGRES_SCHEMA


def test_postgres_environment_override_and_schema_validation(monkeypatch):
    dsn = "postgresql://tester@localhost/tester"
    monkeypatch.setenv("LOESS_STATE_DB_DSN", dsn)
    monkeypatch.setenv("LOESS_STATE_DB_SCHEMA", "loess_isolated")

    assert production_state_database() == dsn
    assert production_state_schema() == "loess_isolated"
    assert RunStateDB(dsn).postgres_schema == "loess_isolated"
    with pytest.raises(ValueError, match="invalid PostgreSQL schema"):
        validate_schema('loess_qgis; DROP SCHEMA public')


def test_run_state_rejects_filesystem_backends_and_requires_explicit_spec_backend(
    tmp_path,
):
    legacy_path = tmp_path / "legacy-state.db"
    legacy_path.write_bytes(b"do-not-modify")
    before = legacy_path.read_bytes()

    with pytest.raises(RunStateError, match="requires a PostgreSQL DSN"):
        RunStateDB(legacy_path)
    with pytest.raises(RunStateError, match="state_backend=postgresql"):
        run_state_from_spec({"state_db": str(legacy_path)})
    with pytest.raises(RunStateError, match="state_backend=postgresql"):
        run_state_from_spec(
            {"state_backend": "filesystem", "state_db": str(legacy_path)}
        )

    assert legacy_path.read_bytes() == before


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
