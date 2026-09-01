import os
import secrets
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "inference_scripts", ROOT / "qgis_plugins"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


@pytest.fixture
def postgres_database_factory(monkeypatch):
    """Create isolated PostgreSQL schemas and remove them after each test.

    Control-plane tests must never use the production ``loess_qgis`` schema.
    The default DSN works through the platform socket on Tencent and is
    resolved to the Homebrew socket by the production connector on macOS.
    """

    from labeling_tool.core.postgres_state import (
        DEFAULT_POSTGRES_DSN,
        DEFAULT_POSTGRES_SCHEMA,
        _resolve_postgres_dsn,
    )
    from labeling_tool.core.run_state_db import RunStateDB

    dsn = str(
        os.environ.get("LOESS_TEST_POSTGRES_DSN") or DEFAULT_POSTGRES_DSN
    ).strip()
    schemas: list[str] = []

    def create_database() -> RunStateDB:
        schema = f"loess_test_{secrets.token_hex(8)}"
        if schema == DEFAULT_POSTGRES_SCHEMA:
            raise RuntimeError("tests must not use the production PostgreSQL schema")
        monkeypatch.setenv("LOESS_STATE_DB_DSN", dsn)
        monkeypatch.setenv("LOESS_STATE_DB_SCHEMA", schema)
        database = RunStateDB(dsn, postgres_schema=schema)
        schemas.append(schema)
        database.initialize()
        return database

    yield create_database

    if not schemas:
        return
    import psycopg2

    connection = psycopg2.connect(_resolve_postgres_dsn(dsn))
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            for schema in reversed(schemas):
                cursor.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    finally:
        connection.close()


@pytest.fixture
def postgres_database(postgres_database_factory):
    """Return one initialized, per-test PostgreSQL Run-state database."""

    return postgres_database_factory()
