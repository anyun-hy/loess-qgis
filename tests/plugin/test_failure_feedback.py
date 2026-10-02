from labeling_tool.monitor.failure_feedback import failure_feedback


def test_failed_feedback_uses_snapshot_errors_and_counts_without_guessing():
    feedback = failure_feedback(
        run_status="failed",
        run={},
        package_counts={"ready": 2, "failed": 1, "queued": 3},
        unit_job_counts={"ready": 8, "failed": 4, "interrupted": 2},
        streams=[
            {"stream_id": "model:a", "status": "ready", "error": ""},
            {"stream_id": "fusion:a", "status": "failed", "error": "write denied"},
        ],
        latest_execution={},
    )

    assert feedback == {
        "title": "本次任务失败",
        "reason": "已记录原因：fusion:a：write denied",
        "impact": "受影响范围：失败推理包 1；失败空间任务 4；失败结果流 fusion:a。",
        "completed": "已登记完成：推理包 2；空间任务 8；完成结果流 1。",
        "pending": "仍待核验：未完成推理包 3；未完成空间任务 2；未完成结果流 0。",
        "next_step": "下一步：可打开主界面定位此任务，再按既有流程核对恢复或重做；不会自动执行。",
    }


def test_stopped_feedback_marks_missing_reason_as_not_provided():
    feedback = failure_feedback(
        run_status="stopped",
        run={},
        package_counts={},
        unit_job_counts={},
        streams=[],
        latest_execution={},
    )

    assert feedback is not None
    assert feedback["title"] == "任务已停止"
    assert feedback["reason"] == "停止原因：当前快照未提供。"
    assert "未提供" in feedback["impact"]
    assert "不会自动执行" in feedback["next_step"]


def test_running_failure_does_not_recommend_recovery():
    feedback = failure_feedback(
        run_status="running",
        run={},
        package_counts={"failed": 1},
        unit_job_counts={},
        streams=[],
        latest_execution={},
    )

    assert feedback is not None
    assert feedback["title"] == "发现已登记的失败任务"
    assert feedback["next_step"] == "下一步：任务仍在运行，请查看事件与日志；当前不提供恢复建议。"


def test_terminal_reason_uses_matching_execution_message():
    feedback = failure_feedback(
        run_status="failed",
        run={},
        package_counts={},
        unit_job_counts={},
        streams=[],
        latest_execution={"status": "failed", "message": "输出磁盘已满"},
    )
    assert feedback["reason"] == "已记录原因：输出磁盘已满"


def test_old_execution_message_does_not_explain_current_running_failures():
    feedback = failure_feedback(
        run_status="running",
        run={},
        package_counts={"failed": 1},
        unit_job_counts={},
        streams=[],
        latest_execution={"status": "failed", "message": "上一次执行的错误"},
    )
    assert feedback["reason"] == "失败原因：当前快照未提供。"
