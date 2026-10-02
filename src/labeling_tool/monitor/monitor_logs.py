"""Qt- and QGIS-free helpers for inference-monitor log presentation.

The persisted-log reader is intentionally bounded and is invoked only by the
inference monitor query worker.  It does not retain state or poll for changes.
"""

from __future__ import annotations

import json
from pathlib import Path
import re


def log_payload(message):
    text = str(message).strip()
    if not (text.startswith("{") and text.endswith("}")):
        return {}
    try:
        value = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def log_severity(level, message):
    """Separate semantic severity from the stdout/stderr/system source."""

    lowered = str(message).lower()
    if lowered.startswith("[resource-tuning] "):
        return "info"
    payload = log_payload(message)
    event = str(payload.get("event") or "").lower()
    status = str(payload.get("status") or "").lower()
    if (
        event.endswith("failed")
        or status in {"failed", "error"}
        or payload.get("success") is False
    ):
        return "error"
    if any(
        token in event
        for token in ("warning", "retry", "reduced", "paused_low_disk")
    ) or status == "warning":
        return "warning"
    explicit_failure = any(
        marker in lowered
        for marker in (
            " failed (rc=",
            "[scheduler-error]",
            "[accelerator-restart]",
            " timed out after ",
            " process error:",
            "exhausted retries",
            "crashed repeatedly",
            "[monitor-db]",
            "fatal error",
        )
    )
    named_exception = (
        lowered.startswith(("error:", "[error]", "fatal:"))
        or re.search(r"\b[a-z_][\w.]*?(?:error|exception):", lowered)
    )
    if explicit_failure or (str(level) != "stderr" and named_exception):
        return "error"
    if any(
        token in lowered for token in ("warning", "warn", "警告")
    ) or any(
        marker in lowered
        for marker in ("[retry]", "fallback", "降档", "自动重试")
    ):
        return "warning"
    return "info"


def read_persisted_log_page(
    run_spec,
    severity,
    before=None,
    *,
    search="",
    limit=200,
    byte_budget=2 * 1024 * 1024,
):
    """Read one bounded reverse page on the query worker, not the GUI thread.

    The byte cursor advances even through pages with no severity matches.
    A raw log is evidence of an occurrence, not evidence of current recovery.
    """
    run_dir = str(run_spec.get("run_dir") or "")
    if not run_dir:
        raise ValueError("当前 Run 未提供日志目录")
    path = Path(run_dir) / "logs" / "pipeline.jsonl"
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        end = size if before is None else int(before)
        if end > size or end < 0:
            raise ValueError("日志文件已截断，请重新查询")
        start = max(0, end - byte_budget)
        handle.seek(start)
        data = handle.read(end - start)
    if start:
        boundary = data.find(b"\n")
        if boundary < 0:
            raise ValueError("单条日志超过读取上限，无法安全分页")
        start += boundary + 1
        data = data[boundary + 1:]
    lines = data.splitlines(keepends=True)
    cursor = start + len(data)
    rows = []
    skipped = 0
    for line in reversed(lines):
        cursor -= len(line)
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("not a log record")
        except (ValueError, UnicodeError):
            skipped += 1
            continue
        message = str(record.get("message") or "")
        source = str(record.get("level") or "system")
        level = log_severity(source, message)
        if level != severity or (search and search.casefold() not in message.casefold()):
            continue
        rows.append({
            "monitor_event_id": cursor, "timestamp": record.get("timestamp"),
            "object_type": "原始日志", "object_id": "pipeline.jsonl",
            "event_type": "原始日志 · " + level, "level": level,
            "message": message[:240], "raw_log": True,
            "payload": {"source": source, "message": message[:16000],
                        "truncated": len(message) > 16000, "byte_offset": cursor},
        })
        if len(rows) >= limit:
            break
    return {"rows": rows, "next_cursor": cursor, "has_more": cursor > 0,
            "raw_log": True, "skipped_records": skipped}


def log_fingerprint(severity, error, affected, attempt=0):
    if severity not in {"warning", "error"} or not str(affected).strip():
        return ""
    normalized_error = re.sub(r"\s+", " ", str(error)).strip().lower()
    normalized_target = re.sub(r"\s+", " ", str(affected)).strip().lower()
    return (
        f"{severity}:{normalized_target}:attempt={int(attempt or 0)}:"
        f"{normalized_error}"
    )


def log_presentation(level, message):
    """Build a stable, readable summary while retaining the raw message."""

    source = str(level) if str(level) in {"stdout", "stderr", "system"} else "system"
    raw = str(message)
    lowered = raw.lower()
    payload = log_payload(raw)
    severity = log_severity(source, raw)
    event = str(payload.get("event") or "")
    error = str(payload.get("error") or raw)
    affected = str(
        payload.get("step")
        or payload.get("label")
        or payload.get("unit_id")
        or payload.get("stream_id")
        or event
        or ""
    )
    if not affected and " timed out after " in lowered:
        affected = raw[: lowered.index(" timed out after ")].strip()
    if not affected and " failed (rc=" in lowered:
        affected = raw[: lowered.index(" failed (rc=")].strip()
    if not affected and "scheduler-error" in lowered:
        affected = "调度器"
    if not affected and "monitor-db" in lowered:
        affected = "监控数据库"
    if not affected and "accelerator-restart" in lowered:
        affected = "加速器进程"
    attempt = int(payload.get("attempt") or 0)

    if severity == "warning":
        if "retry" in lowered or "重试" in raw or "reduced" in lowered or "降档" in raw:
            title = "任务正在自动重试"
            system_action = "系统已调整本次执行并继续运行"
            user_action = "通常不需要处理；重复出现时再查看技术详情"
        elif "paused_low_disk" in lowered or "低磁盘" in raw:
            title = "磁盘空间不足，任务已暂停"
            system_action = "系统保留当前进度，等待空间恢复"
            user_action = "释放磁盘空间后恢复任务"
        else:
            title = "运行警告"
            system_action = "系统继续运行并保留该警告"
            user_action = "通常不需要处理；重复出现时再检查"
    elif severity == "error":
        if "timed out after" in lowered or "超时" in raw:
            title = "任务处理超时"
            system_action = "进程已终止，系统将按恢复规则处理"
            user_action = "等待自动重试；若再次失败，再查看技术详情"
        elif "scheduler-error" in lowered:
            title = "任务调度异常"
            system_action = "系统已停止本次调度操作"
            user_action = "查看技术详情，修复后恢复任务"
        elif "monitor-db" in lowered:
            title = "监控状态读取失败"
            system_action = "推理任务不受影响，监控稍后会再次读取"
            user_action = "若持续出现，再检查 PostgreSQL 连接"
        elif "process error" in lowered:
            title = "进程启动失败"
            system_action = "本次进程没有继续执行"
            user_action = "查看技术详情并检查运行环境"
        elif "assembly" in event.lower() or "assemble" in lowered:
            title = "结果流组装失败"
            system_action = "本次结果流已标记为失败"
            user_action = "查看技术详情，修复后恢复该结果流"
        elif "coverage" in event.lower():
            title = "结果完整性验收失败"
            system_action = "结果未被发布为权威版本"
            user_action = "检查空白、重叠和范围外统计"
        elif "failed (rc=" in lowered:
            title = "进程异常退出"
            system_action = "本次任务已标记为失败"
            user_action = "查看技术详情中的返回码和原始输出"
        else:
            title = "任务执行失败"
            system_action = "本次任务已记录为失败"
            user_action = "查看技术详情，修复后再恢复任务"
    else:
        title = ""
        system_action = ""
        user_action = ""

    fingerprint = log_fingerprint(severity, error, affected, attempt)
    return {
        "source": source,
        "severity": severity,
        "title": title,
        "affected": affected,
        "system_action": system_action,
        "user_action": user_action,
        "error": error,
        "attempt": attempt,
        "fingerprint": fingerprint,
    }
