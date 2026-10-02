"""Direct contracts for the pure inference-monitor progress presentation."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest

from labeling_tool.monitor.monitor_progress import (
    DatabasePhase,
    assembly_fraction,
    database_phase,
    overall_completion_fraction,
    overview_work_label,
    stage_from_step,
    stream_from_step,
    stream_progress_view,
    unit_stage_label,
    waiting_count,
)

ROOT = Path(__file__).resolve().parents[2]


def _stream_view(**changes):
    values = {
        "run_status": "running",
        "stream": {"status": "pending"},
        "durable_counts": {},
        "job_type_counts": {},
        "package_counts": {},
        "inference_active": False,
        "active_stage": "",
        "assembly_info": {},
        "assembly_phase_statuses": {},
        "previous_elapsed": "previous",
        "now": 1_704_067_260.0,
        **changes,
    }
    before = copy.deepcopy(values)
    result = stream_progress_view(**values)
    assert values == before
    return result


@pytest.mark.parametrize(
    ("changes", "stage", "status"),
    [
        ({}, "等待计划", "等待"),
        ({"run_status": "stopped"}, "Run 已停止；恢复入口位于主界面", "已停止"),
        (
            {"run_status": "stopped", "stream": {"status": "ready"}},
            "组装完成 / Run 已停止",
            "成功",
        ),
        ({"run_status": "failed"}, "Run 失败", "失败"),
        (
            {"run_status": "failed", "stream": {"status": "ready"}},
            "组装完成 / Run 未通过",
            "成功",
        ),
        (
            {"run_status": "failed", "package_counts": {"failed": 1}},
            "上游 Work Package 失败",
            "失败",
        ),
        (
            {"package_counts": {"failed": 1}, "stream": {"status": "ready"}},
            "组装完成 / 上游 Package 失败",
            "成功",
        ),
        (
            {"assembly_info": {"status": "failed", "phase_name": "写入"}},
            "组装失败：写入",
            "失败",
        ),
        ({"job_type_counts": {"core": {"failed": 1}}}, "空间单元任务失败", "失败"),
        ({"stream": {"status": "assembling"}}, "并行组装", "运行中"),
        ({"stream": {"status": "ready"}, "run_status": "ready"}, "完成", "成功"),
        ({"stream": {"status": "ready"}}, "已组装 / 等待整体验收", "成功"),
        ({"stream": {"status": "raster_ready"}}, "等待并行组装", "等待"),
        (
            {"inference_active": True, "job_type_counts": {"core": {"running": 1}}},
            "推理 + Core 拟合",
            "运行中",
        ),
        ({"inference_active": True}, "Work Package 推理", "运行中"),
        (
            {"job_type_counts": {"seam_horizontal": {"running": 1}}},
            "Seam 拟合",
            "运行中",
        ),
        ({"active_stage": "Accepted 差分"}, "Accepted 差分", "运行中"),
        ({"package_counts": {"queued": 2}}, "等待上游 Work Package", "等待"),
        ({"durable_counts": {"ready": 2}}, "等待分区栅格收口", "等待"),
    ],
)
def test_stream_stage_precedence(changes, stage, status):
    view = _stream_view(**changes)
    assert (view.stage, view.status) == (stage, status)


def test_package_failure_marks_open_streams_failed_before_run_poll_catches_up():
    for state in ("running", "pending"):
        view = _stream_view(
            stream={"status": state},
            package_counts={"failed": 1, "running": 1, "queued": 45},
            durable_counts={"queued": 6},
            job_type_counts={"core": {"queued": 6}},
        )
        assert (view.stage, view.status) == ("上游 Work Package 失败", "失败")
        assert view.failures == 0


def test_poll_uses_unit_fit_jobs_instead_of_stale_stream_unit_activity():
    view = _stream_view(
        durable_counts={"running": 1},
        job_type_counts={"core": {"interrupted": 1}},
        package_counts={"ready": 1},
    )
    assert (view.unit_progress, view.activity, view.failures) == ("0/1", "0/1", 0)
    assert (view.stage, view.status) == ("空间单元拟合 / 等待依赖", "等待")


def test_persisted_assembly_progress_replaces_completed_unit_counts():
    view = _stream_view(
        stream={"status": "assembling"},
        durable_counts={"ready": 12},
        job_type_counts={"core": {"ready": 12}},
        assembly_info={
            "status": "running",
            "phase_name": "写入正式 GPKG",
            "phase_index": 5,
            "phase_total": 9,
            "progress_current": 4,
            "progress_total": 12,
            "feature_count": 999,
            "phase_started_at": "2024-01-01T00:00:00+00:00",
        },
    )
    assert view.stage == "写入正式 GPKG"
    assert (view.unit_progress, view.stage_progress) == ("12/12", "4/12")
    assert (view.feature_count, view.activity, view.elapsed) == (999, "—", "00:01:00")


@pytest.mark.parametrize("run_status", ["running", "ready", "failed", "stopped"])
def test_assembly_elapsed_uses_persisted_end_and_never_advances_terminal_runs(
    run_status,
):
    assembly = {
        "status": "running",
        "phase": "write",
        "phase_index": 2,
        "phase_total": 4,
        "phase_started_at": "2024-01-01T00:00:00+00:00",
    }
    view = _stream_view(run_status=run_status, assembly_info=assembly)
    assert view.stage_progress == "步骤 2/4"
    assert view.elapsed == ("00:01:00" if run_status == "running" else "—")
    view = _stream_view(
        run_status=run_status,
        assembly_info=assembly,
        assembly_phase_statuses={"write": {"ended_at": "2024-01-01T00:00:09+00:00"}},
    )
    assert view.elapsed == "00:00:09"


def test_missing_phase_start_retains_observed_elapsed_and_counts_assembly_failures():
    view = _stream_view(
        assembly_info={"status": "failed", "phase_started_at": "invalid"},
        job_type_counts={"core": {"failed": 2}},
    )
    assert (view.elapsed, view.failures) == ("previous", 3)


def test_progress_module_import_is_qgis_free_in_a_clean_subprocess():
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "from labeling_tool.monitor.monitor_progress import overall_completion_fraction; "
            "import sys; assert 'qgis' not in sys.modules; "
            "assert overall_completion_fraction('ready', {}, {}, [], {}) == (1.0, 1)",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_step_stream_and_stage_mappings_include_unknown_fallbacks():
    stream_cases = {
        "model_batch:alpha": "model:alpha",
        "fusion_batch:beta": "fusion:beta",
        "mosaic:model:alpha": "model:alpha",
        "polygonize:model:alpha": "model:alpha",
        "subpixel_vectorize:model:alpha": "model:alpha",
        "difference:model:alpha": "model:alpha",
        "unit_fit:model:alpha:core": "model:alpha",
        "assemble_stream:model:alpha": "model:alpha",
        "future_step:model:alpha": "",
    }
    assert {name: stream_from_step(name) for name in stream_cases} == stream_cases

    stage_cases = {
        "unit_fit:model:alpha:core": "空间单元拟合",
        "assemble_stream:model:alpha": "并行组装",
        "model_batch:alpha": "Work Package 推理",
        "fusion_batch:alpha": "Work Package 推理",
        "mosaic:model:alpha": "概率拼接",
        "polygonize:model:alpha": "边界矢量化",
        "subpixel_vectorize:model:alpha": "边界矢量化",
        "difference:model:alpha": "Accepted 差分",
        "finalize_partition_rasters": "分区概率栅格收口",
        "scale_acceptance": "整体验收",
        "accelerator_worker": "Work Package 推理",
        "future_step:model:alpha": "future_step",
    }
    assert {name: stage_from_step(name) for name in stage_cases} == stage_cases


def test_counts_fractions_and_labels_cover_zero_ready_partial_and_unknown_states():
    assert waiting_count({}) == 0
    assert waiting_count({"queued": 1, "interrupted": 2, "resetting": 3}) == 6
    assert assembly_fraction("pending", {}) == 0.0
    assert assembly_fraction("ready", {}) == 1.0
    assert (
        assembly_fraction(
            "assembling",
            {
                "phase_index": 2,
                "phase_total": 4,
                "progress_current": 5,
                "progress_total": 10,
            },
        )
        == 0.375
    )
    assert overall_completion_fraction("running", {}, {}, [], {}) == (0.0, 0)
    assert overall_completion_fraction("ready", {}, {}, [], {}) == (1.0, 1)

    fraction, group_count = overall_completion_fraction(
        "running",
        {"work_package": {"ready": 1, "queued": 1}},
        {"work_package": {"completed": 1, "total": 2}},
        [{"stream_id": "model:alpha", "status": "raster_ready"}],
        {
            "model:alpha": {
                "phase_index": 1,
                "phase_total": 1,
                "progress_current": 1,
                "progress_total": 2,
            }
        },
    )
    assert (fraction, group_count) == (0.5, 4)
    assert (
        unit_stage_label({"core": {"running": 1}, "junction": {"running": 1}})
        == "Core/Junction 拟合"
    )
    assert unit_stage_label({}) == "空间单元拟合"
    assert overview_work_label("Work Package 推理") == "正在识别地物"
    assert overview_work_label("组装失败：写入正式 GPKG") == "结果合并失败，请查看详情"
    assert overview_work_label("future_unknown_stage") == "当前步骤待确认，请查看详情"


def test_progress_helpers_do_not_mutate_their_input_data():
    counts = {"queued": 1, "interrupted": 2, "resetting": 3}
    progress = {
        "phase_index": 1,
        "phase_total": 2,
        "progress_current": 1,
        "progress_total": 2,
    }
    job_counts = {"work_package": {"ready": 1, "queued": 1}}
    job_progress = {"work_package": {"completed": 1, "total": 2}}
    streams = [{"stream_id": "model:alpha", "status": "raster_ready"}]
    runtime = {"model:alpha": progress}
    type_counts = {"core": {"running": 1}}
    values = (counts, progress, job_counts, job_progress, streams, runtime, type_counts)
    before = copy.deepcopy(values)

    waiting_count(counts)
    assembly_fraction("assembling", progress)
    overall_completion_fraction("running", job_counts, job_progress, streams, runtime)
    unit_stage_label(type_counts)

    assert values == before


@pytest.mark.parametrize(
    ("packages", "units", "streams", "active_stage", "expected"),
    [
        (
            {"ready": 3, "running": 1, "queued": 2},
            {"ready": 5, "running": 2, "queued": 3},
            [{"status": "pending"}, {"status": "pending"}],
            "",
            ("packages", "Work Package 推理 + 空间单元拟合", 3, 6),
        ),
        (
            {"ready": 6},
            {"ready": 5, "running": 2, "queued": 3},
            [{"status": "pending"}, {"status": "pending"}],
            "",
            ("units", "空间单元拟合", 5, 10),
        ),
        (
            {"ready": 3, "running": 1, "queued": 2},
            {"ready": 5, "running": 2, "queued": 3},
            [{"status": "pending"}, {"status": "pending"}],
            "并行组装",
            ("assembly", "结果流并行组装", 0, 2),
        ),
        (
            {"ready": 6},
            {"ready": 10},
            [{"status": "raster_ready"}, {"status": "pending"}],
            "",
            ("assembly", "结果流并行组装", 0, 2),
        ),
    ],
)
def test_database_phase_uses_only_the_current_lane_denominator(
    packages, units, streams, active_stage, expected
):
    values = (packages, units, streams)
    before = copy.deepcopy(values)
    assert database_phase(
        "running", packages, units, streams, active_stage
    ) == DatabasePhase(*expected)
    assert values == before


@pytest.mark.parametrize(
    ("status", "failed_packages", "active_stage", "expected"),
    [
        ("stopped", 0, "并行组装", ("stopped", "已停止；恢复操作位于主界面", 0, 1)),
        ("failed", 0, "并行组装", ("failed", "运行失败", 0, 1)),
        (
            "running",
            1,
            "",
            ("package_failed", "Work Package 失败，后续计算已停止", 4, 7),
        ),
        ("ready", 0, "并行组装", ("ready", "已完成", 1, 1)),
        ("resetting", 1, "", ("resetting", "正在重置失败 Work Package", 0, 0)),
    ],
)
def test_database_phase_terminal_states_are_unambiguous(
    status, failed_packages, active_stage, expected
):
    counts = {"ready": 3, "running": 1, "queued": 2}
    packages = {**counts, "failed": failed_packages}
    assert database_phase(
        status, packages, counts, [{"status": "pending"}], active_stage
    ) == DatabasePhase(*expected)
