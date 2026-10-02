"""Pure, current-known launch readiness derived by the main dock."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StartReadiness:
    """A displayable launch gate; final input checks remain at start time."""

    blockers: tuple[str, ...] = ()
    notices: tuple[str, ...] = ()

    @property
    def can_start(self) -> bool:
        return not self.blockers

    @property
    def summary(self) -> str:
        if self.blockers:
            return "当前不能开始：" + "；".join(self.blockers)
        if self.notices:
            return "可以开始：" + "；".join(self.notices)
        return "可以开始本次标注"


def derive_start_readiness(
    *,
    environment_problem: str = "",
    raster_problem: str = "",
    range_problem: str = "",
    output_path: str = "",
    workspace_path: str = "",
    plan_problem: str = "",
    workflow_active: bool = False,
    notices: tuple[str, ...] = (),
) -> StartReadiness:
    """Combine already-known readiness facts without touching filesystem or QGIS."""
    blockers = [
        problem
        for problem in (
            environment_problem,
            raster_problem,
            range_problem,
            "请设置 Accepted GPKG 输出位置" if not output_path else "",
            "请设置运行工作区" if not workspace_path else "",
            plan_problem,
            "当前任务正在运行或停止中" if workflow_active else "",
        )
        if problem
    ]
    return StartReadiness(tuple(blockers), tuple(notices))
