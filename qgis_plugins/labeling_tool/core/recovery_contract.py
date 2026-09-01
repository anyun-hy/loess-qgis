"""Fail-closed validation for resuming or resetting an existing v5 Run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .deployment_contract import deployment_fingerprint, verify_project_runtime
from .postgres_state import is_postgres_location
from .run_spec import RUN_ID_PATTERN, sha256_file
from .run_state_db import RunStateDB, SCHEMA_VERSION


class RecoveryContractError(RuntimeError):
    pass


def _content_sha256(spec: Mapping[str, Any]) -> str:
    payload = dict(spec)
    payload.pop("run_spec_content_sha256", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_recovery_run(
    run_spec_path: str | Path,
    scripts_dir: str | Path,
) -> tuple[dict[str, Any], RunStateDB, Path]:
    """Validate deployment, immutable spec and database identity without writes."""

    deployment = verify_project_runtime(scripts_dir)
    if deployment.get("status") != "ready":
        raise RecoveryContractError(
            "恢复前部署一致性检查失败: "
            + str(deployment.get("message") or "unknown deployment error")
        )

    declared_spec_path = Path(run_spec_path).expanduser()
    if declared_spec_path.is_symlink():
        raise RecoveryContractError(
            f"Run Spec 不能是符号链接: {declared_spec_path}"
        )
    spec_path = declared_spec_path.resolve()
    if not spec_path.is_file() or spec_path.stat().st_size > 2 * 1024 * 1024:
        raise RecoveryContractError(f"Run Spec 缺失或过大: {spec_path}")
    try:
        value = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise RecoveryContractError(f"Run Spec 无法读取: {error}") from error
    if not isinstance(value, Mapping):
        raise RecoveryContractError("Run Spec 顶层必须是对象")
    spec = dict(value)
    if int(spec.get("schema_version") or 0) != SCHEMA_VERSION:
        raise RecoveryContractError(
            f"Run Spec schema 必须为 {SCHEMA_VERSION}"
        )

    run_id = str(spec.get("run_id") or "")
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise RecoveryContractError(f"Run ID 无效: {run_id!r}")
    declared_run_dir = Path(str(spec.get("run_dir") or "")).expanduser()
    if declared_run_dir.is_symlink():
        raise RecoveryContractError(
            f"Run 目录不能是符号链接: {declared_run_dir}"
        )
    run_dir = declared_run_dir.resolve()
    output_root = Path(str(spec.get("output_root") or "")).expanduser().resolve()
    expected_run_dir = output_root / "runs" / run_id
    if (
        run_dir != spec_path.parent
        or run_dir != expected_run_dir
        or run_dir.name != run_id
    ):
        raise RecoveryContractError(
            "Run Spec 路径、output_root、run_dir 与 run_id 身份不一致"
        )

    claimed_content_sha = str(spec.get("run_spec_content_sha256") or "")
    if (
        len(claimed_content_sha) != 64
        or _content_sha256(spec) != claimed_content_sha
    ):
        raise RecoveryContractError("Run Spec 内容身份 SHA256 不一致")

    expected_fingerprint = str(spec.get("config_fingerprint") or "")
    current_fingerprint = deployment_fingerprint(scripts_dir)
    if not expected_fingerprint or expected_fingerprint != current_fingerprint:
        raise RecoveryContractError(
            "Run 创建时的部署指纹与当前代码/项目配置不一致；"
            "旧 Run 不能继续，请使用当前部署创建新 Run"
        )

    state_backend = str(spec.get("state_backend") or "").strip().lower()
    state_location = str(spec.get("state_db") or "").strip()
    if state_backend != "postgresql" or not is_postgres_location(state_location):
        raise RecoveryContractError(
            "当前版本仅支持 PostgreSQL Run 状态库；旧文件状态库不能恢复，"
            "请使用当前部署创建新 Run"
        )
    state_schema = str(spec.get("state_schema") or "").strip() or None
    try:
        database = RunStateDB(
            state_location,
            postgres_schema=state_schema,
        )
    except Exception as error:
        raise RecoveryContractError(
            f"PostgreSQL Run 状态库连接标识无效: {error}"
        ) from error
    try:
        run = database.get_run(run_id)
    except Exception as error:
        raise RecoveryContractError(f"Run 状态库无法读取: {error}") from error
    if run is None:
        raise RecoveryContractError(f"Run 状态库中不存在 {run_id}")
    if str(run.get("status") or "").startswith("archived_"):
        raise RecoveryContractError(
            "Run 已在创建新 Run 时归档为不可恢复状态"
        )
    if int(run.get("schema_version") or 0) != SCHEMA_VERSION:
        raise RecoveryContractError("Run 状态库 schema 与 Run Spec 不一致")
    if str(run.get("run_spec_sha256") or "") != sha256_file(spec_path):
        raise RecoveryContractError("Run 状态库绑定的 Run Spec SHA256 不一致")
    try:
        metadata = json.loads(str(run.get("metadata_json") or "{}"))
    except json.JSONDecodeError as error:
        raise RecoveryContractError("Run 状态库 metadata_json 无效") from error
    metadata_spec = Path(str(metadata.get("run_spec") or "")).expanduser().resolve()
    if metadata_spec != spec_path:
        raise RecoveryContractError("Run 状态库绑定的 Run Spec 路径不一致")

    return spec, database, spec_path
