from qgis_plugins.labeling_tool.core.environment_report import (
    compact_problem,
    format_check_details,
    format_execution_details,
    format_problem_details,
)


def test_compact_problem_keeps_traceback_out_of_dock_status():
    check = {
        "id": "semantic_model_upernet_mambaout_b",
        "status": "error",
        "message": "TorchScript contract failed on mps.\nTraceback (most recent call last):\n"
        + "x" * 400,
    }

    summary = compact_problem(check, max_chars=80)

    assert summary == "TorchScript contract failed on mps."
    assert "Traceback" not in summary


def test_format_problem_details_preserves_copyable_full_error():
    traceback = "Traceback (most recent call last):\n  File model.py, line 1\nRuntimeError: bad op"
    text = format_problem_details(
        [
            {
                "id": "semantic_model_upernet_mambaout_b",
                "status": "error",
                "value": "mps",
                "source": "check_environment.py",
                "fix": "修改 inference_scripts/config.yaml",
                "message": traceback,
            },
            {"id": "ready_item", "status": "ready", "message": "正常"},
        ],
        "native stderr",
    )

    assert "[ERROR] semantic_model_upernet_mambaout_b" in text
    assert traceback in text
    assert "修改 inference_scripts/config.yaml" in text
    assert "native stderr" in text
    assert "ready_item" not in text


def test_format_check_details_includes_ready_and_problem_items():
    text = format_check_details(
        [
            {"id": "dependency_torch", "status": "ready", "value": "2.7.0"},
            {"id": "dependency_fiona", "status": "error", "message": "未安装"},
        ]
    )

    assert "[READY] dependency_torch" in text
    assert "当前值: 2.7.0" in text
    assert "[ERROR] dependency_fiona" in text
    assert "完整信息: 未安装" in text


def test_format_execution_details_exposes_attempt_identity_and_result_source():
    text = format_execution_details({
        "check_id": "check-123",
        "started_at": "2026-09-04T01:00:00+00:00",
        "finished_at": "2026-09-04T01:02:00+00:00",
        "diagnostics_path": "/workspace/cache/environment_check/latest.json",
        "process": {
            "exit_code": 0,
            "exit_status": "NormalExit",
            "report_source": "report_file",
            "error": "",
        },
    })

    assert "检查编号: check-123" in text
    assert "退出码: 0" in text
    assert "结果来源: report_file" in text
    assert "/workspace/cache/environment_check/latest.json" in text
