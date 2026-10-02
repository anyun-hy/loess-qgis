from __future__ import annotations

from pathlib import Path

import pytest
import rasterio
from rasterio.transform import from_origin

from labeling_tool.runs.run_planning import build_run_builder_kwargs
from tools.validation.prepare_v5_real_run import (
    OVERLAP,
    STRIDE,
    TILE_SIZE,
    Extent,
    PreparationError,
    RasterLayer,
    _effective_for_run,
    _grid,
    _parser,
)


@pytest.fixture
def source_raster(tmp_path: Path) -> Path:
    path = tmp_path / "source.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2000,
        height=1800,
        count=3,
        dtype="uint8",
        crs="EPSG:4490",
        transform=from_origin(100.0, 200.0, 0.25, 0.25),
        compress="deflate",
    ):
        pass
    return path


def _required_cli() -> list[str]:
    return [
        "--plugin-parent",
        "/plugin-parent",
        "--scripts-dir",
        "/scripts",
        "--environment-report",
        "/environment.json",
        "--source-raster",
        "/source.tif",
        "--output-root",
        "/output",
        "--state-dsn",
        "dbname=qa",
        "--state-schema",
        "qa_schema",
        "--expected-source-bundle-sha256",
        "0" * 64,
    ]


def test_parser_preserves_centered_2x2_defaults_and_accepts_explicit_controls():
    default = _parser().parse_args(_required_cli())
    assert (default.grid_rows, default.grid_cols) == (2, 2)
    assert default.row_offset is None
    assert default.col_offset is None
    assert default.score_cache_budget_gb is None

    explicit = _parser().parse_args(
        _required_cli()
        + [
            "--grid-rows",
            "7",
            "--grid-cols",
            "9",
            "--row-offset",
            "11",
            "--col-offset",
            "13",
            "--score-cache-budget-gb",
            "8.5",
        ]
    )
    assert (explicit.grid_rows, explicit.grid_cols) == (7, 9)
    assert (explicit.row_offset, explicit.col_offset) == (11, 13)
    assert explicit.score_cache_budget_gb == 8.5


@pytest.mark.parametrize(
    "option,value",
    [
        ("--grid-rows", "0"),
        ("--grid-cols", "-1"),
        ("--row-offset", "-1"),
        ("--col-offset", "-1"),
        ("--score-cache-budget-gb", "0"),
        ("--score-cache-budget-gb", "nan"),
        ("--score-cache-budget-gb", "inf"),
    ],
)
def test_parser_rejects_invalid_grid_and_budget_values(option: str, value: str):
    with pytest.raises(SystemExit):
        _parser().parse_args(_required_cli() + [option, value])


def test_grid_preserves_centered_default_source_pixel_window(source_raster: Path):
    _layer, _extent, tiles, window = _grid(source_raster)

    assert window == {
        "row_off": 484,
        "col_off": 584,
        "row_end": 1316,
        "col_end": 1416,
        "width": 832,
        "height": 832,
        "source_width": 2000,
        "source_height": 1800,
    }
    assert len(tiles) == 4
    assert [(tile["row"], tile["col"]) for tile in tiles] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]


def test_grid_uses_explicit_offsets_and_reports_actual_large_window(
    source_raster: Path,
):
    _layer, extent, tiles, window = _grid(
        source_raster,
        grid_rows=3,
        grid_cols=4,
        row_offset=17,
        col_offset=23,
    )

    assert window == {
        "row_off": 17,
        "col_off": 23,
        "row_end": 1169,
        "col_end": 1495,
        "width": TILE_SIZE + 3 * STRIDE,
        "height": TILE_SIZE + 2 * STRIDE,
        "source_width": 2000,
        "source_height": 1800,
    }
    assert len(tiles) == 12
    assert (tiles[-1]["row"], tiles[-1]["col"]) == (2, 3)
    assert extent.as_dict() == pytest.approx(
        {
            "xmin": 105.75,
            "ymin": -92.25,
            "xmax": 473.75,
            "ymax": 195.75,
        }
    )


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"grid_rows": 6}, "smaller than the requested grid window"),
        (
            {"grid_rows": 2, "row_offset": 1000},
            "row window exceeds raster height",
        ),
        (
            {"grid_cols": 2, "col_offset": 1200},
            "column window exceeds raster width",
        ),
        ({"grid_rows": 0}, "rows and columns must be at least 1"),
        ({"row_offset": -1}, "row offset must be non-negative"),
    ],
)
def test_grid_rejects_out_of_bounds_or_invalid_windows(
    source_raster: Path,
    kwargs: dict[str, int],
    message: str,
):
    with pytest.raises(PreparationError, match=message):
        _grid(source_raster, **kwargs)


def test_explicit_score_cache_budget_flows_through_production_planner(tmp_path: Path):
    effective = {
        "schema_version": 2,
        "runtime": {
            "tile_batch_size": 1,
            "effective_device": "cuda:0",
            "keep_score_cache": False,
        },
        "scaling": {
            "partition_tile_rows": 2,
            "partition_tile_cols": 2,
            "partition_halo_px": 192,
            "seam_band_px": 64,
            "score_cache_budget_gb": "auto",
            "min_free_disk_gb": 0.001,
        },
        "semantic_models": [
            {
                "model_id": "model_a",
                "display_name": "Model A",
                "version": "test",
                "artifact": "model_a.pt",
                "artifact_path": "/models/model_a.pt",
                "sha256": "1" * 64,
                "enabled": True,
            }
        ],
        "fusion_profiles": [],
        "boundary_fitting": {"enabled": False},
        "fragmentation_regularization": {"enabled": False},
        "resource_tuning": {
            "resolved": {
                "tile_batch_size": 1,
                "tile_batch_size_by_model": {"model_a": 1},
            }
        },
    }
    run_effective = _effective_for_run(effective, 0.5)

    bounds = Extent(0.0, 0.0, 128.0, 128.0)
    kwargs = build_run_builder_kwargs(
        scripts_dir=str(tmp_path / "scripts"),
        output_root=str(tmp_path),
        accepted_target_gpkg=str(tmp_path / "accepted_labels.gpkg"),
        raster_layer=RasterLayer(
            tmp_path / "source.tif",
            "EPSG:4490",
            0.25,
            0.25,
        ),
        requested_extent=bounds,
        processing_extent=bounds,
        grid_tiles=({"row": 0, "col": 0, "bounds": bounds},),
        active_tiles=({"row": 0, "col": 0, "bounds": bounds},),
        range_selection={
            "mode": "extent",
            "selected_tile_count": 1,
            "excluded_tile_count": 0,
            "clip_outputs": True,
        },
        effective_config=run_effective,
        environment_report={"config_fingerprint": "sha256:test"},
        accepted_validation={"status": "passed"},
        skip_accepted=False,
        selected_model_ids=("model_a",),
        fusion_profile_id=None,
        boundary_smoothing_enabled=False,
        overlap=OVERLAP,
        run_id="20260930_120000_test",
        run_dir=str(tmp_path / "runs" / "20260930_120000_test"),
        accepted_snapshot="",
        skipped_tiles=(),
        tile_cache_sample={
            "status": "passed",
            "materialized_cache_bytes": 1024,
        },
    )

    assert effective["scaling"]["score_cache_budget_gb"] == "auto"
    assert kwargs["scaling"]["score_cache_budget_mode"] == "explicit"
    assert kwargs["scaling"]["score_cache_budget_gb"] == 0.5
    assert kwargs["storage_report"]["configured_score_cache_budget_gb"] == 0.5
    assert kwargs["storage_report"]["package_tile_limit"] == 1
