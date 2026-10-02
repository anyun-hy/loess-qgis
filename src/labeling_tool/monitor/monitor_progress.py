"""Pure presentation helpers for inference-monitor progress.

Overall completion is the proportion of planned task groups.  It is an
auxiliary progress indicator, not an ETA or evidence that a stage completed.
This module deliberately has no Qt, QGIS, database, I/O, or thread dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from labeling_tool.monitor.monitor_time import elapsed_text, timestamp_epoch


@dataclass(frozen=True)
class StreamProgressView:
    """Display values for one result stream, independent of widgets."""

    stage: str
    status: str
    unit_progress: str
    stage_progress: str
    activity: str
    feature_count: object
    failures: int
    elapsed: str


def stream_progress_view(
    *,
    run_status: str,
    stream: Mapping[str, Any],
    durable_counts: Mapping[str, int],
    job_type_counts: Mapping[str, Mapping[str, int]],
    package_counts: Mapping[str, int],
    inference_active: bool,
    active_stage: str,
    assembly_info: Mapping[str, Any],
    assembly_phase_statuses: Mapping[str, Mapping[str, Any]],
    previous_elapsed: str,
    now: float,
) -> StreamProgressView:
    """Interpret a snapshot without changing it or reading the clock.

    Durable units supply completion counts; current jobs supply activity and
    failures. ``now`` is epoch seconds, captured by the GUI caller. Persisted
    assembly phases replace unit activity once assembly has begun.
    """

    job_counts: dict[str, int] = {}
    for counts in job_type_counts.values():
        for state, count in counts.items():
            job_counts[state] = job_counts.get(state, 0) + int(count)
    total = sum(int(value) for value in durable_counts.values())
    ready = int(durable_counts.get("ready", 0))
    running = int(job_counts.get("running", 0))
    waiting = waiting_count(job_counts)
    failed = int(job_counts.get("failed", 0))
    package_failed = int(package_counts.get("failed", 0))
    stream_status = str(stream.get("status") or "pending")
    assembly_status = str(assembly_info.get("status") or "")
    assembly_phase = str(assembly_info.get("phase_name") or "并行组装")

    if run_status == "stopped":
        if stream_status == "ready":
            stage, status = "组装完成 / Run 已停止", "成功"
        else:
            stage, status = "Run 已停止；恢复入口位于主界面", "已停止"
    elif run_status == "failed":
        if stream_status == "ready":
            stage, status = "组装完成 / Run 未通过", "成功"
        elif package_failed:
            stage, status = "上游 Work Package 失败", "失败"
        else:
            stage, status = "Run 失败", "失败"
    elif package_failed:
        if stream_status == "ready":
            stage, status = "组装完成 / 上游 Package 失败", "成功"
        else:
            stage, status = "上游 Work Package 失败", "失败"
    elif assembly_status == "failed":
        stage, status = f"组装失败：{assembly_phase}", "失败"
    elif failed or stream_status == "failed":
        stage, status = "空间单元任务失败", "失败"
    elif active_stage == "并行组装" or stream_status == "assembling":
        stage, status = assembly_phase, "运行中"
    elif stream_status == "ready" and run_status == "ready":
        stage, status = "完成", "成功"
    elif stream_status == "ready":
        stage, status = "已组装 / 等待整体验收", "成功"
    elif stream_status == "raster_ready":
        stage, status = "等待并行组装", "等待"
    elif inference_active and running:
        stage, status = f"推理 + {unit_stage_label(job_type_counts)}", "运行中"
    elif inference_active:
        stage, status = "Work Package 推理", "运行中"
    elif running:
        stage, status = unit_stage_label(job_type_counts), "运行中"
    elif active_stage:
        stage, status = active_stage, "运行中"
    elif waiting:
        stage, status = "空间单元拟合 / 等待依赖", "等待"
    elif waiting_count(package_counts):
        stage, status = "等待上游 Work Package", "等待"
    elif total and ready == total:
        stage, status = "等待分区栅格收口", "等待"
    else:
        stage, status = "等待计划", "等待"

    unit_progress = f"{ready}/{total}" if total else "-"
    stage_progress = unit_progress
    activity = f"{running}/{waiting}"
    feature_count = None
    elapsed = previous_elapsed
    if assembly_info:
        current = int(assembly_info.get("progress_current") or 0)
        assembly_total = int(assembly_info.get("progress_total") or 0)
        phase_index = int(assembly_info.get("phase_index") or 0)
        phase_total = int(assembly_info.get("phase_total") or 0)
        stage_progress = (
            f"{current}/{assembly_total}"
            if assembly_total
            else f"步骤 {phase_index}/{phase_total}"
        )
        activity = "—"
        feature_count = assembly_info.get("feature_count")
        phase_started = timestamp_epoch(assembly_info.get("phase_started_at") or "")
        if phase_started is not None:
            phase_history = (
                assembly_phase_statuses.get(str(assembly_info.get("phase") or "")) or {}
            )
            phase_end = timestamp_epoch(phase_history.get("ended_at") or "")
            if phase_end is not None:
                elapsed = elapsed_text(phase_end - phase_started)
            elif assembly_status == "running" and run_status not in {
                "ready",
                "failed",
                "stopped",
            }:
                elapsed = elapsed_text(now - phase_started)
            else:
                elapsed = "—"

    return StreamProgressView(
        stage=stage,
        status=status,
        unit_progress=unit_progress,
        stage_progress=stage_progress,
        activity=activity,
        feature_count=feature_count,
        failures=failed + (1 if assembly_status == "failed" else 0),
        elapsed=elapsed,
    )


@dataclass(frozen=True)
class DatabasePhase:
    key: str
    title: str
    current: int
    total: int


def database_phase(
    run_status: str,
    package_counts: Mapping[str, int],
    unit_job_counts: Mapping[str, int],
    streams: Sequence[Mapping[str, object]],
    active_global_stage: str,
) -> DatabasePhase:
    """Derive the durable global stage without reading window or clock state."""

    package_total = sum(int(value) for value in package_counts.values())
    package_ready = int(package_counts.get("ready", 0))
    package_active = int(package_counts.get("running", 0))
    package_waiting = waiting_count(package_counts)
    unit_total = sum(int(value) for value in unit_job_counts.values())
    unit_ready = int(unit_job_counts.get("ready", 0))
    unit_active = int(unit_job_counts.get("running", 0))
    unit_waiting = waiting_count(unit_job_counts)
    stream_total = len(streams)
    stream_ready = sum(str(stream.get("status")) == "ready" for stream in streams)
    raster_ready = sum(
        str(stream.get("status")) in {"raster_ready", "ready"} for stream in streams
    )
    if run_status == "ready":
        return DatabasePhase("ready", "已完成", 1, 1)
    if run_status == "stopped":
        return DatabasePhase("stopped", "已停止；恢复操作位于主界面", 0, 1)
    if run_status == "resetting":
        return DatabasePhase("resetting", "正在重置失败 Work Package", 0, 0)
    package_failed = int(package_counts.get("failed", 0))
    if run_status == "failed" and not package_failed:
        return DatabasePhase("failed", "运行失败", 0, 1)
    if package_failed:
        return DatabasePhase(
            "package_failed",
            "Work Package 失败，后续计算已停止",
            package_ready + package_failed,
            package_total,
        )
    if active_global_stage == "分区概率栅格收口":
        return DatabasePhase("finalize", "分区概率栅格收口", raster_ready, stream_total)
    if active_global_stage == "并行组装":
        return DatabasePhase("assembly", "结果流并行组装", stream_ready, stream_total)
    if active_global_stage == "整体验收":
        return DatabasePhase("acceptance", "整体验收", 0, 0)
    if package_active or package_waiting:
        return DatabasePhase(
            "packages",
            "Work Package 推理 + 空间单元拟合" if unit_active else "Work Package 推理",
            package_ready,
            package_total,
        )
    if unit_active or unit_waiting:
        return DatabasePhase("units", "空间单元拟合", unit_ready, unit_total)
    if int(unit_job_counts.get("failed", 0)):
        return DatabasePhase("unit_failed", "空间单元失败处理", unit_ready, unit_total)
    if stream_total and stream_ready == stream_total:
        return DatabasePhase("acceptance", "整体验收", 0, 0)
    if raster_ready:
        return DatabasePhase("assembly", "结果流并行组装", stream_ready, stream_total)
    return DatabasePhase("finalize", "分区概率栅格收口", raster_ready, stream_total)


def stream_from_step(name: str) -> str:
    if name.startswith("model_batch:"):
        return "model:" + name.split(":", 1)[1]
    if name.startswith("fusion_batch:"):
        return "fusion:" + name.split(":", 1)[1]
    for prefix in ("mosaic:", "polygonize:", "subpixel_vectorize:", "difference:"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    if name.startswith("unit_fit:"):
        return name[len("unit_fit:") :].rsplit(":", 1)[0]
    if name.startswith("assemble_stream:"):
        return name[len("assemble_stream:") :]
    return ""


def stage_from_step(name: str) -> str:
    if name.startswith("unit_fit:"):
        return "空间单元拟合"
    if name.startswith("assemble_stream:"):
        return "并行组装"
    if name.startswith("model_batch:") or name.startswith("fusion_batch:"):
        return "Work Package 推理"
    if name.startswith("mosaic:"):
        return "概率拼接"
    if name.startswith(("polygonize:", "subpixel_vectorize:")):
        return "边界矢量化"
    if name.startswith("difference:"):
        return "Accepted 差分"
    if name == "finalize_partition_rasters":
        return "分区概率栅格收口"
    if name == "scale_acceptance":
        return "整体验收"
    if name == "accelerator_worker":
        return "Work Package 推理"
    return name.split(":", 1)[0]


def waiting_count(counts: Mapping[str, Any]) -> int:
    return sum(
        int(counts.get(key, 0)) for key in ("queued", "interrupted", "resetting")
    )


def assembly_fraction(stream_status: object, progress: Mapping[str, Any]) -> float:
    if (
        str(stream_status) == "ready"
        or str(progress.get("status") or "") == "completed"
    ):
        return 1.0
    phase_total = int(progress.get("phase_total") or 0)
    phase_index = int(progress.get("phase_index") or 0)
    if phase_total < 1 or phase_index < 1:
        return 0.0
    current = int(progress.get("progress_current") or 0)
    total = int(progress.get("progress_total") or 0)
    within_phase = min(1.0, max(0.0, current / total)) if total else 0.0
    return min(1.0, max(0.0, (phase_index - 1 + within_phase) / phase_total))


def overall_completion_fraction(
    run_status: object,
    job_counts: Mapping[str, Mapping[str, Any]],
    job_progress: Mapping[str, Mapping[str, Any]],
    streams: Sequence[Mapping[str, Any]],
    stream_runtime_progress: Mapping[str, Mapping[str, Any]],
) -> tuple[float, int]:
    """Return completion across planned task groups, not estimated time."""

    if str(run_status) == "ready":
        return 1.0, 1
    groups = []
    for job_type in (
        "work_package",
        "fragmentation_v33",
        "unit_confidence",
        "unit_fit",
    ):
        counts = job_counts.get(job_type) or {}
        progress = job_progress.get(job_type) or {}
        total = int(progress.get("total") or sum(int(v) for v in counts.values()))
        if total < 1:
            continue
        completed_value = (
            progress.get("completed")
            if progress.get("completed") is not None
            else counts.get("ready", 0)
        )
        completed = (
            float(completed_value)
            if isinstance(completed_value, (int, float, str, bytes, bytearray))
            else 0.0
        )
        groups.append(min(1.0, max(0.0, completed / total)))
    if streams:
        raster_finalized = all(
            str(stream.get("status") or "") in {"raster_ready", "assembling", "ready"}
            for stream in streams
        )
        groups.append(1.0 if raster_finalized else 0.0)
        groups.append(
            sum(
                assembly_fraction(
                    stream.get("status"),
                    stream_runtime_progress.get(str(stream["stream_id"])) or {},
                )
                for stream in streams
            )
            / len(streams)
        )
        groups.append(0.0)
    if not groups:
        return 0.0, 0
    return sum(groups) / len(groups), len(groups)


def overview_work_label(stage: object) -> str:
    """Plain-language overview; retain technical stages in the detail table."""
    value = str(stage or "等待计划")
    labels = {
        "等待计划": "等待开始",
        "Work Package 推理": "正在识别地物",
        "等待上游 Work Package": "等待识别结果",
        "上游 Work Package 失败": "地物识别失败",
        "Run 失败": "任务失败",
        "组装完成 / Run 已停止": "结果已合并，任务已停止",
        "Run 已停止；恢复入口位于主界面": "已停止，可在主界面恢复",
        "组装完成 / Run 未通过": "结果已合并，任务未通过检查",
        "组装完成 / 上游 Package 失败": "结果已合并，部分识别失败",
        "空间单元任务失败": "边界处理失败",
        "空间单元拟合 / 等待依赖": "等待前一步结果",
        "等待分区栅格收口": "等待整理识别结果",
        "分区概率栅格收口": "正在整理识别结果",
        "等待并行组装": "等待合并结果",
        "并行组装": "正在合并结果",
        "已组装 / 等待整体验收": "已合并，等待检查",
        "整体验收": "正在检查结果",
        "完成": "已完成",
        "概率拼接": "正在拼接识别结果",
        "边界矢量化": "正在生成地物边界",
        "校验单元产物": "正在检查各部分结果",
        "登记对象部件": "正在整理地物",
        "连接跨单元对象": "正在连接相邻地物",
        "写入 Raw GPKG": "正在保存初步结果",
        "写入正式 GPKG": "正在保存正式结果",
        "汇总拟合边界": "正在汇总边界结果",
        "精确范围裁剪": "正在保留选定范围内的结果",
        "空白/重叠验收": "正在检查遗漏和重叠",
        "Accepted 差分": "正在排除已确认的区域",
        "提交产物并清理中间文件": "正在保存结果并清理临时文件",
    }
    if value.startswith("组装失败："):
        return "结果合并失败，请查看详情"
    if value in labels:
        return labels[value]
    if value.endswith("拟合"):
        return (
            "识别地物，同时处理边界" if value.startswith("推理 + ") else "正在处理边界"
        )
    return "当前步骤待确认，请查看详情"


def unit_stage_label(type_counts: Mapping[str, Mapping[str, Any]]) -> str:
    running_types = {
        unit_type
        for unit_type, counts in type_counts.items()
        if int(counts.get("running", 0)) > 0
    }
    labels = []
    if "core" in running_types:
        labels.append("Core")
    if running_types.intersection({"seam_horizontal", "seam_vertical"}):
        labels.append("Seam")
    if "junction" in running_types:
        labels.append("Junction")
    return "/".join(labels) + " 拟合" if labels else "空间单元拟合"
