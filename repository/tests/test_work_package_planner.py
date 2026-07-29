import pytest

from labeling_tool.core.spatial_planner import plan_spatial_units
from labeling_tool.core.work_package_planner import (
    GIB,
    WorkPackagePlanError,
    calculate_package_tile_limit,
    plan_work_packages,
    storage_preflight,
)


def _budget(tile_limit=80):
    per_tile = 1024
    return calculate_package_tile_limit(
        score_cache_budget_gb=(tile_limit * per_tile) / GIB,
        current_model_probability_bytes=512,
        fusion_accumulator_bytes=256,
        mask_confidence_workspace_bytes=128,
        safety_margin_bytes=128,
        available_disk_bytes=10 * GIB,
        min_free_disk_gb=1,
        permanent_estimated_bytes=0,
    )


def test_package_limit_uses_cache_and_disk_minimum():
    budget = _budget(80)
    assert budget["working_bytes_per_tile"] == 1024
    assert budget["cache_tile_limit"] == 80
    assert budget["package_tile_limit"] == 80


def test_work_packages_are_partition_aligned_bounded_and_connected():
    spatial = plan_spatial_units(tile_rows=24, tile_cols=24)
    packages = plan_work_packages(
        spatial,
        package_tile_limit=100,
        estimated_bytes_per_tile=1024,
    )
    assert packages["package_count"] > 1
    assert packages["peak_package_tiles"] <= 100
    assigned = [
        partition_id
        for package in packages["packages"]
        for partition_id in package["partition_ids"]
    ]
    assert sorted(assigned) == sorted(
        partition["partition_id"] for partition in spatial["partitions"]
    )
    assert all(
        package["estimated_bytes"] == package["tile_count"] * 1024
        for package in packages["packages"]
    )


def test_work_package_plan_scales_to_100k_without_tile_id_output():
    spatial = plan_spatial_units(tile_rows=250, tile_cols=400)
    packages = plan_work_packages(
        spatial,
        package_tile_limit=600,
        estimated_bytes_per_tile=4096,
    )
    assert packages["peak_package_tiles"] <= 600
    assert all("tile_ids" not in package for package in packages["packages"])
    assert len(packages["package_by_partition"]) == spatial["partition_count"]


def test_partition_that_exceeds_budget_blocks_start():
    spatial = plan_spatial_units(tile_rows=16, tile_cols=16)
    with pytest.raises(WorkPackagePlanError, match="exceeding package limit"):
        plan_work_packages(
            spatial,
            package_tile_limit=10,
            estimated_bytes_per_tile=1024,
        )


def test_storage_preflight_uses_measured_values_and_preserves_reserve(tmp_path):
    report = storage_preflight(
        tmp_path,
        tile_count=1000,
        stream_count=4,
        permanent_bytes_per_tile_per_stream=2048,
        input_tile_bytes_per_tile=512,
        score_cache_budget_gb=1,
        min_free_disk_gb=1,
        current_model_probability_bytes=4096,
        fusion_accumulator_bytes=2048,
        mask_confidence_workspace_bytes=1024,
        safety_margin_bytes=1024,
        available_disk_bytes=10 * GIB,
    )
    assert report["status"] == "passed"
    assert report["estimated_input_tile_bytes"] == 1000 * 512
    assert report["estimated_permanent_output_bytes"] == 1000 * 4 * 2048
    assert report["estimated_permanent_bytes"] == 1000 * (4 * 2048 + 512)
    assert report["estimated_required_bytes"] <= report["available_disk_bytes"]


def test_storage_preflight_blocks_when_not_even_one_tile_fits(tmp_path):
    with pytest.raises(WorkPackagePlanError, match="cannot hold one"):
        storage_preflight(
            tmp_path,
            tile_count=1000,
            stream_count=4,
            permanent_bytes_per_tile_per_stream=1024 * 1024,
            score_cache_budget_gb=1,
            min_free_disk_gb=2,
            current_model_probability_bytes=4096,
            fusion_accumulator_bytes=2048,
            mask_confidence_workspace_bytes=1024,
            safety_margin_bytes=1024,
            available_disk_bytes=2 * GIB,
        )
