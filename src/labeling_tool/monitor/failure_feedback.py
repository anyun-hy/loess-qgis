"""Pure, evidence-bounded failure summaries for the monitor overview."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

_PENDING_STATUSES = ("queued", "running", "interrupted", "resetting")


def failure_feedback(
    *,
    run_status: str,
    run: Mapping[str, object],
    package_counts: Mapping[str, object],
    unit_job_counts: Mapping[str, object],
    streams: Sequence[Mapping[str, object]],
    latest_execution: Mapping[str, object],
) -> dict[str, str] | None:
    """Present only failure facts provided by the current monitor snapshot."""

    status = str(run_status)
    package_failed = _count(package_counts, "failed")
    unit_failed = _count(unit_job_counts, "failed")
    failed_streams = [
        str(stream.get("stream_id") or "")
        for stream in streams
        if str(stream.get("status") or "") == "failed"
    ]
    attention = status == "failed" or any(
        value is not None and value > 0 for value in (package_failed, unit_failed)
    ) or bool(failed_streams)
    if status == "stopped":
        attention = True
    if not attention:
        return None

    terminal = status in {"failed", "stopped"}
    return {
        "title": _title(status, terminal),
        "reason": _reason(status, run, streams, latest_execution),
        "impact": _impact(package_failed, unit_failed, failed_streams),
        "completed": _completed(package_counts, unit_job_counts, streams),
        "pending": _pending(package_counts, unit_job_counts, streams),
        "next_step": _next_step(status, terminal),
    }


def _count(counts: Mapping[str, object], status: str) -> int | None:
    value = counts.get(status)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _count_text(value: int | None) -> str:
    return str(value) if value is not None else "未提供"


def _first_error(
    run: Mapping[str, object],
    streams: Sequence[Mapping[str, object]],
    latest_execution: Mapping[str, object],
) -> str:
    for source in (run, latest_execution):
        value = str(source.get("error") or "").strip()
        if value:
            return value
    for stream in streams:
        value = str(stream.get("error") or "").strip()
        if value:
            stream_id = str(stream.get("stream_id") or "未命名结果流")
            return f"{stream_id}：{value}"
    return ""


def _title(status: str, terminal: bool) -> str:
    if status == "stopped":
        return "任务已停止"
    if status == "failed":
        return "本次任务失败"
    if terminal:
        return "任务需要处理"
    return "发现已登记的失败任务"


def _reason(
    status: str,
    run: Mapping[str, object],
    streams: Sequence[Mapping[str, object]],
    latest_execution: Mapping[str, object],
) -> str:
    error = _first_error(run, streams, latest_execution)
    if not error and status in {"failed", "stopped"}:
        if str(latest_execution.get("status") or "") == status:
            error = str(latest_execution.get("message") or "").strip()
    if error:
        prefix = "停止原因" if status == "stopped" else "已记录原因"
        return f"{prefix}：{error}"
    if status == "stopped":
        return "停止原因：当前快照未提供。"
    return "失败原因：当前快照未提供。"


def _impact(
    package_failed: int | None,
    unit_failed: int | None,
    failed_streams: Sequence[str],
) -> str:
    stream_text = ", ".join(value for value in failed_streams if value) or "未提供"
    return (
        "受影响范围：失败推理包 "
        f"{_count_text(package_failed)}；失败空间任务 "
        f"{_count_text(unit_failed)}；失败结果流 {stream_text}。"
    )


def _completed(
    package_counts: Mapping[str, object],
    unit_job_counts: Mapping[str, object],
    streams: Sequence[Mapping[str, object]],
) -> str:
    ready_streams = sum(
        1 for stream in streams if str(stream.get("status") or "") == "ready"
    )
    stream_text = str(ready_streams) if streams else "未提供"
    return (
        "已登记完成：推理包 "
        f"{_count_text(_count(package_counts, 'ready'))}；空间任务 "
        f"{_count_text(_count(unit_job_counts, 'ready'))}；完成结果流 {stream_text}。"
    )


def _pending(
    package_counts: Mapping[str, object],
    unit_job_counts: Mapping[str, object],
    streams: Sequence[Mapping[str, object]],
) -> str:
    package_pending = _sum_counts(package_counts)
    unit_pending = _sum_counts(unit_job_counts)
    pending_streams = sum(
        1
        for stream in streams
        if str(stream.get("status") or "") not in {"ready", "failed"}
    )
    stream_text = str(pending_streams) if streams else "未提供"
    return (
        "仍待核验：未完成推理包 "
        f"{_count_text(package_pending)}；未完成空间任务 "
        f"{_count_text(unit_pending)}；未完成结果流 {stream_text}。"
    )


def _sum_counts(counts: Mapping[str, object]) -> int | None:
    values = [_count(counts, status) for status in _PENDING_STATUSES]
    known = [value for value in values if value is not None]
    return sum(known) if known else None


def _next_step(status: str, terminal: bool) -> str:
    if terminal:
        return "下一步：可打开主界面定位此任务，再按既有流程核对恢复或重做；不会自动执行。"
    if status == "running":
        return "下一步：任务仍在运行，请查看事件与日志；当前不提供恢复建议。"
    return "下一步：请查看事件与日志，等待当前状态同步后再决定处理。"
