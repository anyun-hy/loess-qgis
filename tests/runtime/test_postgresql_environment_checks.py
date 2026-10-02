"""First-use database failures remain actionable in the complete check record."""

from types import SimpleNamespace

import pytest

from labeling_tool.main.environment_report import (
    check_label,
    first_problem,
    format_check_details,
)
from labeling_tool.shared.state.postgres_state import PostgresDependencyError
from loess_runtime.system import environment_checks


class DatabaseFailure(Exception):
    def __init__(self, message, sqlstate=""):
        super().__init__(message)
        self.pgcode = sqlstate


@pytest.mark.parametrize(
    ("error", "summary", "remedy"),
    [
        (
            DatabaseFailure('FATAL: role "new_user" does not exist'),
            "角色不存在",
            "已有角色",
        ),
        (DatabaseFailure("本地化的数据库错误", "3D000"), "数据库不存在", "已有库"),
        (
            DatabaseFailure('FATAL: database "new_db" does not exist'),
            "数据库不存在",
            "已有库",
        ),
        (DatabaseFailure("本地化的认证错误", "28P01"), "认证未通过", "凭据"),
        (DatabaseFailure("no pg_hba.conf entry for host"), "认证未通过", "认证规则"),
        (
            DatabaseFailure("permission denied for database fixture", "42501"),
            "权限不足",
            "表读写",
        ),
        (
            DatabaseFailure(
                'connection to server on socket "/tmp/probe" failed: No such file or directory'
            ),
            "连接不可用",
            "socket/host",
        ),
        (
            PostgresDependencyError("psycopg2 is unavailable"),
            "缺少 PostgreSQL 驱动",
            "psycopg2",
        ),
        (
            DatabaseFailure("unexpected schema validation detail"),
            "检查失败",
            "完整错误信息",
        ),
        (DatabaseFailure(""), "检查失败", "完整错误信息"),
    ],
)
def test_failed_database_check_explains_blocker_and_preserves_diagnostic(
    monkeypatch, error, summary, remedy
):
    calls = []

    def initialize():
        calls.append("initialize")
        raise error

    def health():
        pytest.fail("A failed initialization must not be reported as healthy")

    monkeypatch.setattr(
        environment_checks, "production_state_database", lambda: "fixture"
    )
    monkeypatch.setattr(
        environment_checks,
        "RunStateDB",
        lambda dsn: SimpleNamespace(initialize=initialize, pragmas=health),
    )
    checks = []
    environment_checks.append_postgresql_check(checks, "qgis")

    assert calls == ["initialize"]
    assert len(checks) == 1
    check = checks[0]
    assert check["status"] == "error"
    assert check_label(check) == "PostgreSQL 任务数据库"
    assert summary in first_problem(checks)
    assert remedy in check["fix"]
    assert (str(error) or "DatabaseFailure") in format_check_details(checks)
    assert environment_checks.overall_status(checks) == "error"


def test_ready_database_report_uses_actual_health_and_has_no_repair_prompt(monkeypatch):
    calls = []

    def database(dsn):
        assert dsn == "fixture"
        return SimpleNamespace(
            initialize=lambda: calls.append("initialize"),
            pragmas=lambda: {
                "server_version": "18.6",
                "database": "example",
                "schema": "example_schema",
            },
        )

    monkeypatch.setattr(
        environment_checks, "production_state_database", lambda: "fixture"
    )
    monkeypatch.setattr(environment_checks, "RunStateDB", database)
    checks = []
    environment_checks.append_postgresql_check(checks, "qgis")

    assert calls == ["initialize"]
    assert checks[0]["status"] == "ready"
    assert checks[0]["value"] == "PostgreSQL 18.6 / example / example_schema"
    assert checks[0]["fix"] == ""
    assert first_problem(checks) == ""
