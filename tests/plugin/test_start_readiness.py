from labeling_tool.main.start_readiness import derive_start_readiness


def test_readiness_collects_known_blockers_without_performing_checks():
    readiness = derive_start_readiness(
        environment_problem="推理环境报告已过期，请重新检查",
        raster_problem="请选择有效的本地影像层",
        range_problem="手绘矩形范围与影像没有重叠",
        output_path="",
        workspace_path="",
        plan_problem="请确认本次推理方案",
        workflow_active=True,
    )

    assert readiness.blockers == (
        "推理环境报告已过期，请重新检查",
        "请选择有效的本地影像层",
        "手绘矩形范围与影像没有重叠",
        "请设置 Accepted GPKG 输出位置",
        "请设置运行工作区",
        "请确认本次推理方案",
        "当前任务正在运行或停止中",
    )
    assert not readiness.can_start
    assert "范围与影像没有重叠" in readiness.summary


def test_readiness_keeps_current_view_notice_without_blocking_start():
    readiness = derive_start_readiness(
        output_path="/tmp/accepted.gpkg",
        workspace_path="/tmp/workspace",
        notices=("开始时将自动读取当前视图范围",),
    )

    assert readiness.can_start
    assert readiness.summary == "可以开始：开始时将自动读取当前视图范围"

def test_model_unavailable_and_required_paths_are_blockers():
    readiness = derive_start_readiness(
        output_path="",
        workspace_path="",
        plan_problem="模型方案不可用：所选模型未通过当前设备检查",
    )

    assert readiness.can_start is False
    assert readiness.blockers == (
        "请设置 Accepted GPKG 输出位置",
        "请设置运行工作区",
        "模型方案不可用：所选模型未通过当前设备检查",
    )
