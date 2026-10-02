"""Qt-independent contracts for inference monitoring and execution history."""

from __future__ import annotations

from typing import Any, Mapping


MONITOR_HISTORY_VERSION = 1
MONITOR_EVENT_PAGE_SIZE = 200
MONITOR_EVENT_PAGE_LIMIT = 500
MONITOR_DETAIL_PAGE_LIMIT = 500
MONITOR_EXECUTION_ENV = "LOESS_MONITOR_EXECUTION_ID"

ASSEMBLY_PHASES = (
    ("validate_inputs", "校验单元产物", "项"),
    ("register_objects", "登记对象部件", "部件"),
    ("link_objects", "连接跨单元对象", "连接"),
    ("write_raw", "写入 Raw GPKG", "要素"),
    ("write_formal", "写入正式 GPKG", "单元"),
    ("aggregate_reports", "汇总拟合边界", "报告"),
    ("range_clip", "精确范围裁剪", "项"),
    ("coverage_validation", "空白/重叠验收", "项"),
    ("accepted_difference", "Accepted 差分", "项"),
    ("publish_cleanup", "提交产物并清理中间文件", "产物"),
)
ASSEMBLY_PHASE_NAMES = {key: name for key, name, _unit in ASSEMBLY_PHASES}
ASSEMBLY_PHASE_UNITS = {key: unit for key, _name, unit in ASSEMBLY_PHASES}

TERMINAL_SPAN_STATUSES = {
    "completed",
    "reused",
    "skipped",
    "failed",
    "interrupted",
    "stopped",
}

SPAN_STATUS_LABELS = {
    "pending": "未开始",
    "running": "运行中",
    "completed": "完成",
    "reused": "复用并校验通过",
    "skipped": "跳过",
    "failed": "失败",
    "interrupted": "中断",
    "stopped": "已停止",
    "missing": "记录缺失",
}


def progress_text(current: Any, total: Any, *, unit: str = "项") -> str:
    """Render a truthful progress value without conflating zero and unknown."""

    if total is None:
        return "—"
    try:
        total_value = int(total)
        current_value = int(current or 0)
    except (TypeError, ValueError):
        return "—"
    if total_value <= 0:
        return "0" if total_value == 0 and current is not None else "—"
    return f"{max(0, current_value):,}/{total_value:,}{unit}"


def effective_device_text(run_spec: Mapping[str, Any] | None) -> tuple[str, str]:
    """Return backend and verified/frozen device label for one Run."""

    spec = dict(run_spec or {})
    runtime = dict(spec.get("runtime") or {})
    tuning = dict(spec.get("resource_tuning") or {})
    hardware = dict(tuning.get("hardware") or tuning.get("snapshot") or {})
    device = str(runtime.get("effective_device") or "").strip().lower()
    accelerator = dict(hardware.get("accelerator") or {})
    name = str(
        accelerator.get("name")
        or hardware.get("accelerator_name")
        or hardware.get("device_name")
        or ""
    ).strip()
    if device.startswith("cuda"):
        return "CUDA", name or "CUDA 设备"
    if device == "mps":
        return "MPS", name or "Apple MPS"
    if device == "cpu":
        return "CPU", name if name and "cpu" in name.lower() else "CPU"
    return (device.upper() or "未知后端", name or "设备信息缺失")


def execution_trigger_label(trigger: str) -> str:
    return {
        "start": "开始",
        "resume": "恢复",
        "redo_failed_packages": "重做失败包",
    }.get(str(trigger), str(trigger) or "未知")


def archived_history_summary(run_row: Mapping[str, Any] | None) -> dict[str, Any]:
    """Read the bounded archived monitor summary without inventing detail."""

    import json

    row = dict(run_row or {})
    try:
        metadata = json.loads(str(row.get("metadata_json") or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        metadata = {}
    archive = dict((metadata or {}).get("database_detail_archive") or {})
    return dict(archive.get("monitor_history_summary") or {})
