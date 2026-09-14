"""Pure presentation helpers for inference-monitor progress.

Overall completion is the proportion of planned task groups.  It is an
auxiliary progress indicator, not an ETA or evidence that a stage completed.
This module deliberately has no Qt, QGIS, database, I/O, or thread dependency.
"""

from __future__ import annotations


def stream_from_step(name: str) -> str:
    if name.startswith("model_batch:"):
        return "model:" + name.split(":", 1)[1]
    if name.startswith("fusion_batch:"):
        return "fusion:" + name.split(":", 1)[1]
    for prefix in ("mosaic:", "polygonize:", "subpixel_vectorize:", "difference:"):
        if name.startswith(prefix):
            return name[len(prefix):]
    if name.startswith("unit_fit:"):
        return name[len("unit_fit:"):].rsplit(":", 1)[0]
    if name.startswith("assemble_stream:"):
        return name[len("assemble_stream:"):]
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


def waiting_count(counts) -> int:
    return sum(
        int(counts.get(key, 0))
        for key in ("queued", "interrupted", "resetting")
    )


def assembly_fraction(stream_status, progress) -> float:
    if str(stream_status) == "ready" or str(progress.get("status") or "") == "completed":
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
    run_status,
    job_counts,
    job_progress,
    streams,
    stream_runtime_progress,
):
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
        completed = float(
            progress.get("completed")
            if progress.get("completed") is not None
            else counts.get("ready", 0)
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


def overview_work_label(stage) -> str:
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
        return "识别地物，同时处理边界" if value.startswith("推理 + ") else "正在处理边界"
    return "当前步骤待确认，请查看详情"


def unit_stage_label(type_counts) -> str:
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
