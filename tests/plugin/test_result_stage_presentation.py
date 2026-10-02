from labeling_tool.monitor.result_stage_presentation import result_stage_presentation


def test_ready_stream_without_coverage_never_claims_geometry_passed():
    stages = result_stage_presentation({"status": "成功"}, {})

    assert "登记为成功" in stages["output"]
    assert "覆盖验收通过" not in stages["geometry"]
    assert "尚未执行" in stages["geometry"]
    assert "未在此同步" in stages["review"]
    assert "未在此同步" in stages["accepted"]


def test_coverage_pass_is_limited_to_the_stream_geometry_check():
    stages = result_stage_presentation(
        {"status": "成功"}, {"status": "passed", "gap_area_m2": 0}
    )

    assert stages["geometry"] == "覆盖验收通过（仅该结果流的几何覆盖检查）。"
    assert "人工确认" not in stages["geometry"]


def test_stopped_skipped_and_unknown_stream_statuses_remain_explicit():
    stopped = result_stage_presentation({"status": "已停止"}, {})
    skipped = result_stage_presentation({"status": "跳过"}, {})
    unknown = result_stage_presentation({"status": "等待人工复核"}, {})

    assert "已停止" in stopped["output"]
    assert "跳过" in skipped["output"]
    assert "等待人工复核" in unknown["output"]
    assert "成功" not in unknown["output"]
