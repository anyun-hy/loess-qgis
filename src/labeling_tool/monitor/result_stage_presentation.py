"""Evidence-bounded stage text for a monitored result stream."""

from __future__ import annotations

from collections.abc import Mapping


def result_stage_presentation(
    state: Mapping[str, object], coverage: Mapping[str, object]
) -> dict[str, str]:
    """Describe only the result-stream and coverage projections supplied."""

    stream_status = str(state.get("status") or "")
    output = {
        "成功": "结果流已登记为成功；不代表几何验收、人工确认或入库。",
        "运行中": "结果流仍在处理；当前产出尚不能作为完成结论。",
        "失败": "结果流已登记失败；请在事件与日志查看已记录原因。",
        "等待": "结果流等待处理；尚无推理产出结论。",
        "已停止": "结果流已停止；当前产出不代表几何验收、人工确认或入库。",
        "跳过": "结果流已跳过；当前没有可作为完成结论的推理产出。",
        "未验证": "结果流尚未完成验证；当前产出不代表人工确认或入库。",
    }.get(
        stream_status,
        f"结果流当前状态：{stream_status}；本页未提供对应的完成结论。"
        if stream_status
        else "结果流状态未提供。",
    )
    coverage_status = str(coverage.get("status") or "")
    geometry = {
        "passed": "覆盖验收通过（仅该结果流的几何覆盖检查）。",
        "failed": "覆盖验收失败；该结果流未通过几何覆盖检查。",
        "skipped": "覆盖验收未执行。",
    }.get(coverage_status, "覆盖验收尚未执行或未在此同步。")
    unavailable = "未在此同步；请在分类修整窗口查看。"
    return {
        "output": output,
        "geometry": geometry,
        "review": unavailable,
        "accepted": unavailable,
    }
