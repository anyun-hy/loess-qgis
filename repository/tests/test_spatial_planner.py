import json

import pytest

from labeling_tool.core.spatial_planner import (
    MAX_LOCAL_TILES,
    SpatialPlanError,
    plan_spatial_units,
)


@pytest.mark.parametrize(
    ("rows", "cols"),
    [(1, 1), (1, 17), (17, 1), (2, 2), (17, 17), (31, 23)],
)
def test_spatial_plan_has_exact_mutually_exclusive_ownership(rows, cols):
    plan = plan_spatial_units(tile_rows=rows, tile_cols=cols)
    partition_rows = plan["partition_rows"]
    partition_cols = plan["partition_cols"]
    assert plan["partition_count"] == partition_rows * partition_cols
    assert plan["unit_counts"]["core"] == partition_rows * partition_cols
    assert plan["unit_counts"].get("seam_vertical", 0) == (
        partition_rows * max(0, partition_cols - 1)
    )
    assert plan["unit_counts"].get("seam_horizontal", 0) == (
        max(0, partition_rows - 1) * partition_cols
    )
    assert plan["unit_counts"].get("junction", 0) == (
        max(0, partition_rows - 1) * max(0, partition_cols - 1)
    )
    processing = plan["processing_window"]
    expected_width = 512 + (cols - 1) * (512 - 192)
    expected_height = 512 + (rows - 1) * (512 - 192)
    assert processing == {"x0": 0, "y0": 0, "x1": expected_width, "y1": expected_height}


def test_partition_halo_is_clamped_and_meets_context_contract():
    plan = plan_spatial_units(tile_rows=17, tile_cols=17, halo_px=256)
    processing = plan["processing_window"]
    assert plan["halo_px"] == 256
    for partition in plan["partitions"]:
        core = partition["core_window"]
        halo = partition["halo_window"]
        assert 0 <= halo["x0"] <= core["x0"]
        assert 0 <= halo["y0"] <= core["y0"]
        assert core["x1"] <= halo["x1"] <= processing["x1"]
        assert core["y1"] <= halo["y1"] <= processing["y1"]


def test_500k_tile_plan_is_bounded_deterministic_and_json_serializable():
    first = plan_spatial_units(tile_rows=1000, tile_cols=500)
    second = plan_spatial_units(tile_rows=1000, tile_cols=500)
    assert first["tile_count"] == MAX_LOCAL_TILES
    assert first == second
    assert first["partition_count"] == 125 * 63
    assert len(first["spatial_units"]) == 125 * 63 + 125 * 62 + 124 * 63 + 124 * 62
    assert len(json.dumps(first, separators=(",", ":"))) < 25_000_000


def test_spatial_plan_rejects_unsupported_or_unsafe_contracts():
    with pytest.raises(SpatialPlanError, match="cannot exceed"):
        plan_spatial_units(tile_rows=1000, tile_cols=501)
    with pytest.raises(SpatialPlanError, match="at least 2 x 2"):
        plan_spatial_units(tile_rows=10, tile_cols=10, partition_tile_rows=1)
    with pytest.raises(SpatialPlanError, match="halo_px"):
        plan_spatial_units(tile_rows=10, tile_cols=10, halo_px=64)
    with pytest.raises(SpatialPlanError, match="too narrow"):
        plan_spatial_units(
            tile_rows=10,
            tile_cols=10,
            overlap=500,
            partition_tile_rows=2,
            partition_tile_cols=2,
            seam_band_px=64,
        )
