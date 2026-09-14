"""Direct contracts for the pure inference-monitor progress presentation."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess
import sys

from labeling_tool.core.monitor_progress import (
    assembly_fraction,
    overall_completion_fraction,
    overview_work_label,
    stage_from_step,
    stream_from_step,
    unit_stage_label,
    waiting_count,
)


ROOT = Path(__file__).resolve().parents[1]


def test_progress_module_import_is_qgis_free_in_a_clean_subprocess():
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "qgis_plugins")
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "from labeling_tool.core.monitor_progress import overall_completion_fraction; "
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
    assert assembly_fraction(
        "assembling",
        {"phase_index": 2, "phase_total": 4, "progress_current": 5, "progress_total": 10},
    ) == 0.375
    assert overall_completion_fraction("running", {}, {}, [], {}) == (0.0, 0)
    assert overall_completion_fraction("ready", {}, {}, [], {}) == (1.0, 1)

    fraction, group_count = overall_completion_fraction(
        "running",
        {"work_package": {"ready": 1, "queued": 1}},
        {"work_package": {"completed": 1, "total": 2}},
        [{"stream_id": "model:alpha", "status": "raster_ready"}],
        {"model:alpha": {"phase_index": 1, "phase_total": 1, "progress_current": 1, "progress_total": 2}},
    )
    assert (fraction, group_count) == (0.5, 4)
    assert unit_stage_label({"core": {"running": 1}, "junction": {"running": 1}}) == "Core/Junction 拟合"
    assert unit_stage_label({}) == "空间单元拟合"
    assert overview_work_label("Work Package 推理") == "正在识别地物"
    assert overview_work_label("组装失败：写入正式 GPKG") == "结果合并失败，请查看详情"
    assert overview_work_label("future_unknown_stage") == "当前步骤待确认，请查看详情"


def test_progress_helpers_do_not_mutate_their_input_data():
    counts = {"queued": 1, "interrupted": 2, "resetting": 3}
    progress = {"phase_index": 1, "phase_total": 2, "progress_current": 1, "progress_total": 2}
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
