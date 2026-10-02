"""Read-only admission summary for the refinement window."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdmissionSummarySnapshot:
    """Existing refinement facts rendered by :class:`AdmissionSummaryPanel`."""

    target_path: str = ""
    final_feature_count: int | None = None
    confirmed_class_count: int | None = None
    class_total: int = 14
    unsaved_edit_count: int | None = None
    topology_executed: bool = False
    topology_issue_count: int | None = None
    allow_issues: bool = False
    background_stage: str = ""
    blocker_reason: str = ""
    write_enabled: bool = False
    write_reason: str = ""
    accepted_feature_count: int | None = None
    accepted_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdmissionSummaryPresentation:
    """Text-only rendering of one dialog-provided admission snapshot."""

    target_path: str
    final_features: str
    class_confirmation: str
    unsaved_edits: str
    topology: str
    background_stage: str
    blocker: str
    accepted_result: str
    accepted_warnings: tuple[str, ...]
    integrity_note: str
    next_action: str


def _count(value: int | None) -> str:
    return "未提供" if value is None else str(value)


def admission_summary_presentation(
    snapshot: AdmissionSummarySnapshot,
) -> AdmissionSummaryPresentation:
    """Format supplied facts without reading storage or inferring acceptance."""

    target_path = str(snapshot.target_path or "")
    final_features = (
        f"{snapshot.final_feature_count} 个面；"
        "实际新增数量将在后台核对。"
        if snapshot.final_feature_count is not None
        else "未提供。"
    )
    class_total = max(int(snapshot.class_total), 0)
    class_confirmation = (
        f"{_count(snapshot.confirmed_class_count)}/{class_total}"
        if snapshot.confirmed_class_count is not None
        else f"未提供/{class_total}"
    )
    unsaved_edits = _count(snapshot.unsaved_edit_count)
    if not snapshot.topology_executed:
        topology = "尚未执行。"
    elif snapshot.topology_issue_count is None:
        topology = "已执行；问题数未提供。"
    else:
        topology = f"已执行；问题数 {snapshot.topology_issue_count}。"

    stage = str(snapshot.background_stage or "").strip()
    background_stage = stage if stage else "当前没有进行中的校验或提交。"
    reason = str(snapshot.write_reason or snapshot.blocker_reason or "").strip()
    blocker = (
        "无（写入前仍会执行完整性检查）。"
        if snapshot.write_enabled
        else reason if reason else "未提供。"
    )
    accepted_result = (
        f"{snapshot.accepted_feature_count} 个面。"
        if snapshot.accepted_feature_count is not None
        else "当前窗口未记录本次实际新增数量。"
    )
    if snapshot.topology_executed and (snapshot.topology_issue_count or 0) > 0:
        integrity_note = (
            "已明确允许带问题入库；写入前完整性检查仍会执行。"
            if snapshot.allow_issues
            else "存在拓扑问题；解决问题或明确允许带问题入库后再继续，"
            "写入前完整性检查仍会执行。"
        )
    else:
        integrity_note = "写入前完整性检查仍会执行；本摘要不替代该检查。"
    if stage:
        next_action = "下一步：等待当前后台阶段完成。"
    elif snapshot.accepted_feature_count is not None:
        next_action = "下一步：本次写入已完成；继续编辑后需重新组装和检查。"
    elif snapshot.write_enabled:
        next_action = "下一步：检查并写入标签库。"
    elif reason:
        next_action = f"下一步：{reason}"
    else:
        next_action = "下一步：完成入库前检查后再尝试写入。"
    return AdmissionSummaryPresentation(
        target_path=target_path,
        final_features=final_features,
        class_confirmation=class_confirmation,
        unsaved_edits=unsaved_edits,
        topology=topology,
        background_stage=background_stage,
        blocker=blocker,
        accepted_result=accepted_result,
        accepted_warnings=tuple(
            str(warning) for warning in snapshot.accepted_warnings if str(warning)
        ),
        integrity_note=integrity_note,
        next_action=next_action,
    )
