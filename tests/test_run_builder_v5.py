import json
import sqlite3
from pathlib import Path

from labeling_tool.core.run_builder_v5 import create_v5_run
from labeling_tool.core.run_state_db import RunStateDB


def _scaling():
    return {
        "partition_tile_rows": 8,
        "partition_tile_cols": 8,
        "partition_halo_px": 256,
        "seam_band_px": 64,
        "max_job_retries": 2,
    }


def _boundary():
    return {
        "enabled": True,
        "mode": "divider_cubic_bspline_adaptive_v2",
        "smoothing_factor": 1.0,
        "curve_sampling_spacing_px": 0.5,
        "max_chord_error_px": 0.25,
        "max_segment_arc_length_px": 8.0,
        "diagnostic_level": "changed_and_failed",
    }


def _tiles(rows, cols, path):
    for row in range(rows):
        for col in range(cols):
            yield {
                "row": row,
                "col": col,
                "path": str(path),
                "sha256": "f" * 64,
                "pixel_window": {
                    "x0": col * 320,
                    "y0": row * 320,
                    "x1": col * 320 + 512,
                    "y1": row * 320 + 512,
                },
            }


def test_v5_run_keeps_100k_tile_details_out_of_json(tmp_path):
    raster = tmp_path / "source.tif"
    raster.write_bytes(b"fixture")
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    storage = {
        "package_tile_limit": 600,
        "working_bytes_per_tile": 4096,
        "status": "passed",
    }
    spec, spec_path, database_path = create_v5_run(
        state_database=tmp_path / "state.sqlite",
        output_root=tmp_path / "output",
        raster={
            "path": raster,
            "crs": "EPSG:4490",
            "transform": [1, 0, 0, 0, -1, 0],
            "nodata": None,
        },
        requested_extent={"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
        processing_extent={"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
        tile_rows=250,
        tile_cols=400,
        tiles=_tiles(250, 400, raster),
        models=[
            {
                "model_id": "fixture_model",
                "artifact_path": str(model),
                "sha256": "e" * 64,
                "version": "fixture-v1",
            }
        ],
        effective_device="cpu",
        overlap=192,
        scaling=_scaling(),
        boundary_fitting=_boundary(),
        storage_report=storage,
        accepted_target_gpkg=tmp_path / "output" / "accepted.gpkg",
        accepted_validation={
            "status": "passed",
            "feature_count": 7,
            "overlap_pair_count": 0,
            "overlap_tolerance": 0.000001,
        },
        run_id="20260717_210000_fixture",
    )
    assert spec["schema_version"] == 2
    assert spec["fragmentation_regularization"] == {
        "enabled": True,
        "policy_id": "semantic_optimized_200_v3",
        "policy_version": "semantic_optimized_200_v3_core_bounded_v1",
        "baseline_policy_id": "semantic_optimized_200_v3",
        "baseline_policy_version": "semantic_optimized_200_v3_core_bounded_v1",
        "buffer_pixels": 256,
        "max_workers": 4,
    }
    assert spec["range_selection"]["mode"] == "extent"
    assert spec["range_selection"]["clip_outputs"] is True
    assert spec["coverage_validation"] == {
        "policy_id": "exact_range_zero_gap_v1",
        "area_tolerance_pixels": 0.01,
    }
    assert "tiles" not in spec
    assert spec_path.stat().st_size < 50_000
    expected_tile_cache = (
        tmp_path
        / "output"
        / "cache"
        / "20260717_210000_fixture"
        / "tile_cache"
    ).resolve()
    assert Path(spec["cache_root"]) == expected_tile_cache.parent
    assert Path(spec["tile_cache_dir"]) == expected_tile_cache
    assert expected_tile_cache.is_dir()
    assert Path(spec["accepted_target_gpkg"]) == (
        tmp_path / "output" / "accepted.gpkg"
    ).resolve()
    assert spec["accepted_validation"] == {
        "status": "passed",
        "feature_count": 7,
        "overlap_pair_count": 0,
        "overlap_tolerance": 0.000001,
    }
    startup_index = json.loads(
        (tmp_path / "output" / "run_index.json").read_text(encoding="utf-8")
    )
    assert startup_index["latest_run_id"] == spec["run_id"]
    assert startup_index["latest_run_status"] == "planned"
    assert startup_index["latest_ready_run_id"] == ""

    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM tiles").fetchone()[0] == 100_000
        assert connection.execute("SELECT COUNT(*) FROM partitions").fetchone()[0] == 32 * 50
        package_count = connection.execute("SELECT COUNT(*) FROM work_packages").fetchone()[0]
        first_package = connection.execute(
            "SELECT package_id FROM work_packages ORDER BY sequence_no LIMIT 1"
        ).fetchone()[0]
        unit_count = connection.execute("SELECT COUNT(*) FROM spatial_units").fetchone()[0]
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == (
            package_count + unit_count
        )
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        first_tile = connection.execute(
            "SELECT raster_path, sha256 FROM tiles ORDER BY row_no, col_no LIMIT 1"
        ).fetchone()
        assert Path(first_tile[0]) == expected_tile_cache / "tile_0_0.tif"
        assert first_tile[1] == ""
    assert len(RunStateDB(database_path).package_tiles(spec["run_id"], first_package)) <= 600
    assert json.loads(spec_path.read_text())["state_db"] == str(database_path)


def test_v33_plans_one_authoritative_fusion_job(tmp_path):
    raster = tmp_path / "source.tif"
    raster.write_bytes(b"fixture")
    models = []
    for model_id in ("a", "b"):
        artifact = tmp_path / f"{model_id}.pt"
        artifact.write_bytes(model_id.encode())
        models.append(
            {
                "model_id": model_id,
                "artifact_path": str(artifact),
                "sha256": "e" * 64,
                "version": "fixture",
            }
        )
    profile = {
        "profile_id": "approved",
        "status": "approved",
        "approval": {"passed": True},
        "strategy": "equal_probability_average",
        "models": [{"model_id": "a"}, {"model_id": "b"}],
        "weights": [[0.5, 0.5] for _ in range(14)],
    }
    spec, _spec_path, database_path = create_v5_run(
        state_database=tmp_path / "state.sqlite",
        output_root=tmp_path / "output",
        raster={
            "path": raster,
            "crs": "EPSG:3857",
            "transform": [1, 0, 0, 0, -1, 512],
            "nodata": None,
        },
        requested_extent={"xmin": 0, "ymin": 0, "xmax": 512, "ymax": 512},
        processing_extent={"xmin": 0, "ymin": 0, "xmax": 512, "ymax": 512},
        tile_rows=1,
        tile_cols=1,
        tiles=_tiles(1, 1, raster),
        models=models,
        effective_device="cpu",
        overlap=192,
        scaling=_scaling(),
        boundary_fitting=_boundary(),
        storage_report={
            "package_tile_limit": 4,
            "working_bytes_per_tile": 4096,
            "status": "passed",
        },
        fragmentation_regularization={
            "enabled": True,
            "policy_id": "fragmentation_v33_configurable_absorption_v1",
            "policy_version": "v33_production_20260826",
            "baseline_policy_id": "semantic_optimized_200_v3",
            "baseline_policy_version": "semantic_optimized_200_v3_core_bounded_v1",
            "buffer_pixels": 256,
            "max_workers": 4,
            "publication": "authoritative_fusion_core",
            "policy_sha256": "a" * 64,
            "executor_sha256": "b" * 64,
        },
        fusion={"profile_id": "approved", "profile": profile},
        run_id="20260826_120000_v33builder",
    )

    assert spec["fragmentation_regularization"]["publication"] == (
        "authoritative_fusion_core"
    )
    with sqlite3.connect(database_path) as connection:
        job = connection.execute(
            "SELECT job_type, stream_id, unit_id, status FROM jobs "
            "WHERE job_type='fragmentation_v33'"
        ).fetchall()
        dependencies = connection.execute(
            "SELECT partition_id FROM unit_dependencies "
            "WHERE unit_id='fragmentation_v33'"
        ).fetchall()
        partition_count = connection.execute(
            "SELECT COUNT(*) FROM partitions"
        ).fetchone()[0]
    assert job == [
        (
            "fragmentation_v33",
            "fusion:approved",
            "fragmentation_v33",
            "queued",
        )
    ]
    assert len(dependencies) == partition_count
