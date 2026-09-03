import pytest

from labeling_tool.core.spatial_planner import plan_spatial_units
from labeling_tool.core.work_package_planner import (
    GIB,
    MIN_VECTOR_OUTPUT_RESERVE_BYTES,
    PERMANENT_RASTER_BYTES_PER_PIXEL_PER_STREAM,
    WorkPackagePlanError,
    calculate_package_tile_limit,
    final_artifact_size_prediction,
    fusion_accumulator_atomic_overhead,
    fusion_accumulator_bytes_per_tile,
    permanent_output_reserve,
    plan_work_packages,
    resolve_frozen_tile_batch_size,
    storage_preflight,
    unit_confidence_reserve,
    unit_confidence_write_reserve,
)


def test_final_artifact_prediction_is_a_single_observation_only_value():
    prediction = final_artifact_size_prediction(
        core_pixel_count=330_319_374,
        stream_count=4,
    )

    assert prediction == {
        "schema_version": 1,
        "observation_only": True,
        "core_pixel_count": 330_319_374,
        "stream_count": 4,
        "status": "predicted",
        "predicted_final_artifact_bytes": 5_232_930_535,
    }
    assert final_artifact_size_prediction(
        core_pixel_count=330_319_374,
        stream_count=3,
    )["status"] == "not_applicable"


def test_fusion_accumulator_budget_matches_strategy_channels():
    assert fusion_accumulator_bytes_per_tile(None, pixel_count=10) == 0
    assert fusion_accumulator_bytes_per_tile(
        {"strategy": "equal_probability_average", "models": [{"model_id": "a"}]},
        pixel_count=10,
    ) == 10 * 15 * 4
    assert fusion_accumulator_bytes_per_tile(
        {
            "strategy": "linear_1x1",
            "models": [{"model_id": str(index)} for index in range(5)],
        },
        pixel_count=10,
    ) == 10 * 71 * 4


def test_fusion_atomic_overhead_uses_one_largest_partition_halo_generation():
    spatial = plan_spatial_units(
        tile_rows=3,
        tile_cols=5,
        tile_size=512,
        overlap=128,
        partition_tile_rows=2,
        partition_tile_cols=3,
        seam_band_px=64,
        halo_px=128,
    )
    largest_halo_pixels = max(
        (item["halo_window"]["x1"] - item["halo_window"]["x0"])
        * (item["halo_window"]["y1"] - item["halo_window"]["y0"])
        for item in spatial["partitions"]
    )
    average = {
        "strategy": "equal_probability_average",
        "models": [{"model_id": "a"}, {"model_id": "b"}],
    }
    linear = {
        "strategy": "linear_1x1",
        "models": [{"model_id": str(index)} for index in range(5)],
    }

    assert fusion_accumulator_atomic_overhead(None, spatial) == 0
    assert fusion_accumulator_atomic_overhead(average, spatial) == (
        largest_halo_pixels * 14 * 4 + 64 * 1024
    )
    assert fusion_accumulator_atomic_overhead(linear, spatial) == (
        largest_halo_pixels * 70 * 4 + 64 * 1024
    )


def test_unit_confidence_reserve_is_exact_core_domain_plus_file_overhead():
    spatial = plan_spatial_units(
        tile_rows=3,
        tile_cols=5,
        tile_size=512,
        overlap=128,
        partition_tile_rows=2,
        partition_tile_cols=3,
        seam_band_px=64,
        halo_px=128,
    )

    reserve = unit_confidence_reserve(spatial)

    processing = spatial["processing_window"]
    expected_pixels = (
        processing["x1"] - processing["x0"]
    ) * (processing["y1"] - processing["y0"])
    assert reserve["pixel_count"] == expected_pixels
    assert reserve["payload_bytes"] == expected_pixels * 4
    assert reserve["reserve_bytes"] == sum(
        unit_confidence_write_reserve(
            (unit["pixel_window"]["x1"] - unit["pixel_window"]["x0"])
            * (unit["pixel_window"]["y1"] - unit["pixel_window"]["y0"])
        )
        for unit in spatial["spatial_units"]
    )
    assert reserve["file_overhead_bytes"] >= (
        len(spatial["spatial_units"]) * 64 * 1024
    )
    assert reserve["reserve_bytes"] == (
        reserve["payload_bytes"] + reserve["file_overhead_bytes"]
    )
    assert unit_confidence_write_reserve(10_000_000) == 40_400_000


def test_storage_preflight_reserves_deferred_confidence_without_double_guessing(
    tmp_path,
):
    raster_bytes, vector_reserve = _permanent_bytes(1_000, 4)
    deferred = 3 * GIB
    report = storage_preflight(
        tmp_path,
        tile_count=1_000,
        stream_count=4,
        permanent_raster_bytes=raster_bytes,
        vector_output_reserve_bytes=vector_reserve,
        permanent_core_pixel_count=1_000,
        score_cache_budget_gb="auto",
        min_free_disk_gb=10,
        current_model_probability_bytes=4096,
        fusion_accumulator_bytes=2048,
        mask_confidence_workspace_bytes=1024,
        safety_margin_bytes=1024,
        deferred_temporary_reserve_bytes=deferred,
        available_disk_bytes=100 * GIB,
        total_disk_bytes=200 * GIB,
    )

    assert report["formula_version"] == (
        "disk-aware-cache-v4-streamed-v33-confidence"
    )
    assert report["deferred_temporary_reserve_bytes"] == deferred
    assert report["safe_headroom_bytes"] == (
        report["available_disk_bytes"]
        - report["protected_permanent_estimated_bytes"]
        - report["effective_min_free_disk_bytes"]
        - report["atomic_write_overhead_bytes"]
        - deferred
    )
    assert report["estimated_required_bytes"] <= report["available_disk_bytes"]


def test_storage_preflight_freezes_fusion_atomic_peak_separately(tmp_path):
    raster_bytes, vector_reserve = _permanent_bytes(1, 1)
    fusion_atomic = 128 * 1024 * 1024
    report = storage_preflight(
        tmp_path,
        tile_count=64,
        stream_count=1,
        permanent_raster_bytes=raster_bytes,
        vector_output_reserve_bytes=vector_reserve,
        permanent_core_pixel_count=1,
        input_tile_bytes_per_tile=1024,
        score_cache_budget_gb="auto",
        min_free_disk_gb=1,
        current_model_probability_bytes=2048,
        fusion_accumulator_bytes=4096,
        mask_confidence_workspace_bytes=1024,
        safety_margin_bytes=1024,
        fixed_temporary_overhead_bytes=64 * 1024 * 1024,
        fusion_atomic_write_overhead_bytes=fusion_atomic,
        available_disk_bytes=900 * GIB,
        total_disk_bytes=1024 * GIB,
        tile_batch_size=16,
    )

    assert report["fusion_accumulator_atomic_overhead_bytes"] == fusion_atomic
    assert report["atomic_write_overhead_bytes"] == (
        report["atomic_checkpoint_overhead_bytes"] + fusion_atomic
    )
    assert report["fixed_temporary_overhead_bytes"] == report[
        "atomic_write_overhead_bytes"
    ]


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


def _permanent_bytes(core_pixels, streams):
    return (
        core_pixels * streams * PERMANENT_RASTER_BYTES_PER_PIXEL_PER_STREAM,
        max(MIN_VECTOR_OUTPUT_RESERVE_BYTES, core_pixels * streams),
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


def test_permanent_reserve_counts_every_exact_core_including_edge_partitions():
    spatial = plan_spatial_units(
        tile_rows=3,
        tile_cols=5,
        tile_size=512,
        overlap=128,
        partition_tile_rows=2,
        partition_tile_cols=3,
        seam_band_px=64,
        halo_px=128,
    )
    estimate = permanent_output_reserve(spatial, stream_count=4)
    full_width = (5 - 1) * (512 - 128) + 512
    full_height = (3 - 1) * (512 - 128) + 512
    expected_core_pixels = full_width * full_height

    assert spatial["partition_count"] == 4
    assert estimate["core_pixel_count"] == expected_core_pixels
    assert estimate["permanent_raster_bytes"] == (
        expected_core_pixels
        * 4
        * PERMANENT_RASTER_BYTES_PER_PIXEL_PER_STREAM
    )
    assert estimate["vector_output_reserve_bytes"] == (
        MIN_VECTOR_OUTPUT_RESERVE_BYTES
    )
    assert any(
        (partition["core_window"]["x1"] - partition["core_window"]["x0"])
        < 3 * (512 - 128)
        for partition in spatial["partitions"]
    )


def test_auto_batch_requires_a_frozen_integer_candidate():
    assert resolve_frozen_tile_batch_size("auto", None, 16) == 16
    assert resolve_frozen_tile_batch_size(8, 16) == 8
    for invalid in (None, "auto", 16.5, True):
        with pytest.raises(WorkPackagePlanError, match="must be resolved"):
            resolve_frozen_tile_batch_size("auto", invalid)


@pytest.mark.parametrize("field", ["raster", "vector"])
def test_storage_preflight_rejects_non_exact_permanent_formula(tmp_path, field):
    raster_bytes, vector_reserve = _permanent_bytes(1000, 4)
    if field == "raster":
        raster_bytes -= 1
        message = "raster bytes"
    else:
        vector_reserve -= 1
        message = "vector output reserve"
    with pytest.raises(WorkPackagePlanError, match=message):
        storage_preflight(
            tmp_path,
            tile_count=1000,
            stream_count=4,
            permanent_raster_bytes=raster_bytes,
            vector_output_reserve_bytes=vector_reserve,
            permanent_core_pixel_count=1000,
            score_cache_budget_gb=1,
            min_free_disk_gb=1,
            current_model_probability_bytes=4096,
            fusion_accumulator_bytes=2048,
            mask_confidence_workspace_bytes=1024,
            safety_margin_bytes=1024,
            available_disk_bytes=200 * GIB,
            total_disk_bytes=200 * GIB,
        )


def test_storage_preflight_uses_measured_values_and_preserves_reserve(tmp_path):
    core_pixels = 1000
    raster_bytes, vector_reserve = _permanent_bytes(core_pixels, 4)
    report = storage_preflight(
        tmp_path,
        tile_count=1000,
        stream_count=4,
        permanent_raster_bytes=raster_bytes,
        vector_output_reserve_bytes=vector_reserve,
        permanent_core_pixel_count=core_pixels,
        input_tile_bytes_per_tile=512,
        score_cache_budget_gb=1,
        min_free_disk_gb=1,
        current_model_probability_bytes=4096,
        fusion_accumulator_bytes=2048,
        mask_confidence_workspace_bytes=1024,
        safety_margin_bytes=1024,
        available_disk_bytes=200 * GIB,
        total_disk_bytes=200 * GIB,
    )
    assert report["status"] == "passed"
    assert 0 < report["estimated_input_tile_bytes"] <= 1000 * 512
    assert report["input_tile_storage_mode"] == "work_package_temporary"
    assert report["estimated_permanent_output_bytes"] == (
        raster_bytes + vector_reserve
    )
    assert report["final_artifact_size_prediction"]["status"] == "predicted"
    assert "lower_bound_bytes" not in report["final_artifact_size_prediction"]
    assert "upper_bound_bytes" not in report["final_artifact_size_prediction"]
    assert report["estimated_permanent_bytes"] == raster_bytes
    assert report["vector_output_reserve_bytes"] == vector_reserve
    assert report["nondecaying_permanent_reserve_bytes"] == (
        vector_reserve
        + int((raster_bytes + vector_reserve) * 0.25)
    )
    assert report["score_cache_budget_mode"] == "explicit"
    assert report["permanent_uncertainty_ratio"] == 0.25
    assert report["input_tile_bytes"] == 512
    assert report["estimated_required_bytes"] <= report["available_disk_bytes"]


def test_1_42tb_auto_budget_keeps_full_permanent_reserve_with_excluded_tiles(
    tmp_path,
):
    pixel_count = 512 * 512
    available = 1_418_682_363_904
    filesystem_total = 1_967_317_549_056
    spatial = plan_spatial_units(
        tile_rows=267,
        tile_cols=766,
        tile_size=512,
        overlap=128,
        partition_tile_rows=8,
        partition_tile_cols=8,
        seam_band_px=64,
        halo_px=128,
    )
    permanent = permanent_output_reserve(spatial, stream_count=4)
    report = storage_preflight(
        tmp_path,
        tile_count=131_678,
        stream_count=4,
        permanent_raster_bytes=permanent["permanent_raster_bytes"],
        vector_output_reserve_bytes=permanent["vector_output_reserve_bytes"],
        permanent_core_pixel_count=permanent["core_pixel_count"],
        input_tile_bytes_per_tile=pixel_count * 3,
        score_cache_budget_gb="auto",
        min_free_disk_gb=50,
        current_model_probability_bytes=pixel_count * 14 * 2,
        fusion_accumulator_bytes=pixel_count * 15 * 4,
        mask_confidence_workspace_bytes=pixel_count * (14 * 4 + 5),
        safety_margin_bytes=pixel_count * 3,
        fixed_temporary_overhead_bytes=pixel_count * 14 * 2 * 16,
        available_disk_bytes=available,
        total_disk_bytes=filesystem_total,
        tile_batch_size=16,
    )

    expected_headroom = (
        available
        - report["estimated_permanent_bytes"]
        - report["permanent_uncertainty_bytes"]
        - report["effective_min_free_disk_bytes"]
        - report["atomic_checkpoint_overhead_bytes"]
    )
    assert report["score_cache_budget_mode"] == "auto"
    assert report["configured_score_cache_budget_gb"] == "auto"
    assert report["tile_count"] == 131_678 < 267 * 766
    assert report["safe_headroom_bytes"] == expected_headroom
    expected_core_pixels = (
        ((766 - 1) * (512 - 128) + 512)
        * ((267 - 1) * (512 - 128) + 512)
    )
    assert report["permanent_core_pixel_count"] == expected_core_pixels
    assert report["estimated_permanent_raster_bytes"] == (
        expected_core_pixels * 4 * 6
    )
    assert report["vector_output_reserve_bytes"] == expected_core_pixels * 4
    assert report["package_tile_limit"] > 208
    assert report["package_tile_limit"] % 16 == 0
    assert report["effective_min_free_disk_bytes"] == pytest.approx(
        filesystem_total * 0.05, abs=1
    )
    assert report["estimated_required_bytes"] <= available


def test_explicit_8_gib_retains_the_small_package_boundary(tmp_path):
    pixel_count = 512 * 512
    raster_bytes, vector_reserve = _permanent_bytes(1, 4)
    report = storage_preflight(
        tmp_path,
        tile_count=131_678,
        stream_count=4,
        permanent_raster_bytes=raster_bytes,
        vector_output_reserve_bytes=vector_reserve,
        permanent_core_pixel_count=1,
        input_tile_bytes_per_tile=pixel_count * 3,
        score_cache_budget_gb=8,
        min_free_disk_gb=50,
        current_model_probability_bytes=pixel_count * 14 * 2,
        fusion_accumulator_bytes=pixel_count * 15 * 4,
        mask_confidence_workspace_bytes=pixel_count * (14 * 4 + 5),
        safety_margin_bytes=pixel_count * 3,
        fixed_temporary_overhead_bytes=pixel_count * 14 * 2 * 16,
        available_disk_bytes=1_418_682_363_904,
        total_disk_bytes=1_967_317_549_056,
        tile_batch_size=16,
    )

    assert report["score_cache_budget_mode"] == "explicit"
    assert report["resolved_score_cache_budget_gb"] == 8
    assert report["package_tile_limit"] == 208


def test_auto_cache_caps_a_small_run_to_its_full_working_set(tmp_path):
    raster_bytes, vector_reserve = _permanent_bytes(100, 1)
    report = storage_preflight(
        tmp_path,
        tile_count=100,
        stream_count=1,
        permanent_raster_bytes=raster_bytes,
        vector_output_reserve_bytes=vector_reserve,
        permanent_core_pixel_count=100,
        input_tile_bytes_per_tile=2048,
        score_cache_budget_gb="auto",
        min_free_disk_gb=10,
        current_model_probability_bytes=4096,
        fusion_accumulator_bytes=0,
        mask_confidence_workspace_bytes=2048,
        safety_margin_bytes=1024,
        available_disk_bytes=900 * GIB,
        total_disk_bytes=1024 * GIB,
        tile_batch_size=16,
    )

    assert report["package_tile_limit"] == 100
    assert report["working_cache_budget_bytes"] == (
        100 * report["working_bytes_per_tile"]
    )
    assert report["resolved_score_cache_budget_gb"] < 0.01


def test_storage_preflight_auto_blocks_before_consuming_disk_reserve(tmp_path):
    raster_bytes, vector_reserve = _permanent_bytes(1000, 4)
    with pytest.raises(WorkPackagePlanError, match="自动缓存预算"):
        storage_preflight(
            tmp_path,
            tile_count=1000,
            stream_count=4,
            permanent_raster_bytes=raster_bytes,
            vector_output_reserve_bytes=vector_reserve,
            permanent_core_pixel_count=1000,
            score_cache_budget_gb="auto",
            min_free_disk_gb=2,
            current_model_probability_bytes=4096,
            fusion_accumulator_bytes=2048,
            mask_confidence_workspace_bytes=1024,
            safety_margin_bytes=1024,
            fixed_temporary_overhead_bytes=128 * 1024 * 1024,
            available_disk_bytes=2 * GIB,
            total_disk_bytes=2 * GIB,
        )


def test_storage_preflight_blocks_when_not_even_one_tile_fits(tmp_path):
    raster_bytes, vector_reserve = _permanent_bytes(1000, 4)
    with pytest.raises(WorkPackagePlanError, match="磁盘空间预检失败"):
        storage_preflight(
            tmp_path,
            tile_count=1000,
            stream_count=4,
            permanent_raster_bytes=raster_bytes,
            vector_output_reserve_bytes=vector_reserve,
            permanent_core_pixel_count=1000,
            score_cache_budget_gb=1,
            min_free_disk_gb=2,
            current_model_probability_bytes=4096,
            fusion_accumulator_bytes=2048,
            mask_confidence_workspace_bytes=1024,
            safety_margin_bytes=1024,
            available_disk_bytes=2 * GIB,
            total_disk_bytes=2 * GIB,
        )
