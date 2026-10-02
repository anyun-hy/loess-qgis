"""Formatting helpers for environment-check summaries and full diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

PROBLEM_STATUSES = ("error", "warning")

CHECK_LABELS = {
    "conda_env": "Conda 环境",
    "config_yaml": "配置文件",
    "semantic_version": "语义版本",
    "device": "计算设备",
    "sam3_enabled": "SAM3",
    "sam3_backend": "SAM3 实现",
    "sam3_checkpoint": "SAM3 权重",
    "sam3_version": "SAM3 版本",
    "sam_buffer_px": "SAM3 缓冲",
    "class_mapping": "类别映射",
    "index_to_code": "输出通道映射",
    "output_dir": "输出目录",
    "semantic_model_load": "语义模型加载",
    "sam3_model_load": "SAM3 模型加载",
    "model_load_check": "模型加载检查",
    "environment_process": "环境检查进程",
    "scripts_dir": "脚本目录",
    "tile_parameters": "Tile 参数",
    "output_path": "输出 GPKG",
    "dependency_postgresql_state": "PostgreSQL 任务数据库",
}


def check_label(check: Mapping[str, Any]) -> str:
    """Return the stable Chinese label for one environment check mapping."""

    check_id = str(check.get("id", ""))
    if check_id in CHECK_LABELS:
        return CHECK_LABELS[check_id]
    if check_id.startswith("dependency_"):
        return "依赖: " + check_id.removeprefix("dependency_")
    if check_id.startswith("file_"):
        return "文件: " + str(check.get("value", ""))
    if check_id.startswith("semantic_model_"):
        return "语义模型: " + check_id.removeprefix("semantic_model_")
    if check_id.startswith("fusion_profile_"):
        return "融合配置: " + check_id.removeprefix("fusion_profile_")
    if check_id.startswith("deprecated_"):
        return "废弃配置"
    return CHECK_LABELS.get(check_id, check_id or "检查项")


def compact_problem(check: Mapping[str, Any], max_chars: int = 180) -> str:
    """Return one bounded line for the dock status area."""
    value = check.get("message") or check.get("value") or check.get("id") or "未知问题"
    lines = [line.strip() for line in str(value).splitlines() if line.strip()]
    summary = lines[0] if lines else "未知问题"
    summary = " ".join(summary.split())
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3].rstrip() + "..."
    return summary


def first_problem(checks: Sequence[Mapping[str, Any]]) -> str:
    """Return the first error, or first warning when no error is present."""

    for wanted in PROBLEM_STATUSES:
        for check in checks:
            if check.get("status") == wanted:
                return compact_problem(check)
    return ""


def _format_check_details(
    checks: Sequence[Mapping[str, Any]],
    stderr: str = "",
    statuses: tuple[str, ...] | None = None,
) -> str:
    blocks = []
    for check in checks:
        status = str(check.get("status") or "")
        if statuses is not None and status not in statuses:
            continue
        lines = [f"[{status.upper()}] {check.get('id') or 'unknown'}"]
        for label, key in (
            ("当前值", "value"),
            ("来源", "source"),
            ("修改位置", "fix"),
            ("完整信息", "message"),
        ):
            value = check.get(key)
            if value not in (None, ""):
                lines.append(f"{label}: {value}")
        blocks.append("\n".join(lines))

    joined = "\n\n".join(blocks)
    process_stderr = str(stderr or "").strip()
    if process_stderr and process_stderr not in joined:
        blocks.append("[PROCESS STDERR]\n" + process_stderr)

    return "\n\n".join(blocks) or "所有检查项均正常。"


def format_problem_details(
    checks: Sequence[Mapping[str, Any]], stderr: str = ""
) -> str:
    """Build selectable text containing only warnings and errors."""
    return _format_check_details(checks, stderr, PROBLEM_STATUSES)


def format_check_details(checks: Sequence[Mapping[str, Any]], stderr: str = "") -> str:
    """Build selectable text containing every environment check."""
    return _format_check_details(checks, stderr)


def format_execution_details(report: Mapping[str, Any]) -> str:
    """Build the stable identity and QProcess result for one check attempt."""

    process = report.get("process") or {}
    values = (
        ("检查编号", report.get("check_id")),
        ("开始时间", report.get("started_at")),
        ("完成时间", report.get("finished_at")),
        ("退出码", process.get("exit_code")),
        ("退出状态", process.get("exit_status")),
        ("结果来源", process.get("report_source")),
        ("进程错误", process.get("error")),
        ("诊断文件", report.get("diagnostics_path")),
    )
    lines = [f"{label}: {value}" for label, value in values if value not in (None, "")]
    return "[本次检查]\n" + ("\n".join(lines) if lines else "无进程元数据")
