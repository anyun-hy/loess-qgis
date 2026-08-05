import hashlib
import json
import sqlite3
from pathlib import Path

import fiona
import numpy as np
import pytest
import rasterio
import work_package_runtime
from affine import Affine
from rasterio.transform import from_origin
from shapely.geometry import box, shape
from shapely.ops import unary_union

from labeling_tool.core.run_builder_v5 import create_v5_run
from labeling_tool.core.run_state_db import RunStateDB
from labeling_tool.core.run_spec import atomic_write_json
from labeling_tool.core.spatial_planner import plan_spatial_units
from boundary_fitting.unit_runtime import (
    _polygonize,
    _write_diagnostic_gpkg,
    _write_gpkg,
    run_unit_fit,
)
from assemble_stream import StreamAssemblyError, assemble_stream
from finalize_partition_rasters import finalize_partition_rasters
from scale_acceptance import build_scale_acceptance_report
from work_package_runtime import (
    WorkPackageRuntimeError,
    _commit_artifact,
    run_work_package,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scaling():
    return {
        "partition_tile_rows": 8,
        "partition_tile_cols": 8,
        "partition_halo_px": 192,
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


def test_unit_polygonize_does_not_emit_deprecated_memory_driver_warning(capfd):
    probabilities = np.zeros((14, 2, 2), dtype=np.float32)
    probabilities[0, 0, 0] = 1.0
    probabilities[1, 0, 1] = 1.0
    probabilities[1, 1, :] = 1.0
    records = _polygonize(
        probabilities,
        {
            "unit_id": "core_00000_00000",
            "pixel_window": {"x0": 0, "y0": 0, "x1": 2, "y1": 2},
        },
        list(range(14)),
    )
    captured = capfd.readouterr()
    assert len(records) == 2
    assert "'Memory' driver is deprecated" not in captured.err


def test_work_package_loads_each_model_once_and_writes_model_and_fusion_parts(
    tmp_path,
    capsys,
    monkeypatch,
):
    tile = tmp_path / "tile_0_0.tif"
    with rasterio.open(
        tile,
        "w",
        driver="GTiff",
        width=512,
        height=512,
        count=3,
        dtype="uint8",
        crs="EPSG:4490",
        transform=from_origin(0, 512, 1, 1),
        compress="deflate",
    ) as destination:
        destination.write(np.zeros((3, 512, 512), dtype=np.uint8))
    source_sha256 = _sha(tile)
    model_a = tmp_path / "a.pt"
    model_b = tmp_path / "b.pt"
    model_a.write_bytes(b"model-a")
    model_b.write_bytes(b"model-b")
    profile = {
        "profile_id": "fixture_fusion",
        "status": "approved",
        "approval": {"passed": True},
        "strategy": "equal_probability_average",
        "models": [{"model_id": "a"}, {"model_id": "b"}],
        "weights": [[0.5, 0.5] for _ in range(14)],
    }
    spec, spec_path, database_path = create_v5_run(
        output_root=tmp_path / "output",
        raster={
            "path": tile,
            "crs": "EPSG:4490",
            "transform": [1, 0, 0, 0, -1, 512],
            "nodata": None,
        },
        requested_extent={"xmin": 0, "ymin": 0, "xmax": 512, "ymax": 512},
        processing_extent={"xmin": 0, "ymin": 0, "xmax": 512, "ymax": 512},
        tile_rows=1,
        tile_cols=1,
        tiles=[
            {
                "row": 0,
                "col": 0,
                "path": str(tile),
                "sha256": _sha(tile),
                "pixel_window": {"x0": 0, "y0": 0, "x1": 512, "y1": 512},
            }
        ],
        models=[
            {
                "model_id": "a",
                "artifact_path": str(model_a),
                "sha256": _sha(model_a),
                "version": "fixture",
            },
            {
                "model_id": "b",
                "artifact_path": str(model_b),
                "sha256": _sha(model_b),
                "version": "fixture",
            },
        ],
        effective_device="cpu",
        overlap=192,
        scaling=_scaling(),
        boundary_fitting=_boundary(),
        storage_report={
            "package_tile_limit": 4,
            "working_bytes_per_tile": 4096,
            "status": "passed",
        },
        fusion={"profile_id": "fixture_fusion", "profile": profile},
        run_id="20260717_220000_fixture",
    )
    loaded = []

    def loader(model_entry, _device):
        loaded.append(model_entry["model_id"])
        return model_entry["model_id"]

    def infer(model_id, _tile_path, _device):
        probabilities = np.zeros((14, 512, 512), dtype=np.float32)
        probabilities[0 if model_id == "a" else 1] = 1.0
        return probabilities

    with sqlite3.connect(database_path) as connection:
        package_id = connection.execute(
            "SELECT package_id FROM work_packages"
        ).fetchone()[0]
    model_a.write_bytes(b"corrupt-model-a")
    with pytest.raises(WorkPackageRuntimeError, match="model SHA256 mismatch"):
        run_work_package(
            spec_path,
            package_id,
            device="cpu",
            model_loader=loader,
            infer_tile=infer,
        )
    assert tile.is_file()
    assert _sha(tile) == source_sha256
    assert (
        Path(spec["tile_cache_dir"]) / "tile_0_0.tif"
    ).is_file()
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT status FROM work_packages WHERE package_id=?", (package_id,)
        ).fetchone()[0] == "failed"

    model_a.write_bytes(b"model-a")
    original_build_partition_arrays = work_package_runtime.build_partition_arrays
    injected_partition_failures = 0

    def fail_first_partition(*args, **kwargs):
        nonlocal injected_partition_failures
        if injected_partition_failures == 0:
            injected_partition_failures += 1
            raise RuntimeError("partition fault injection")
        return original_build_partition_arrays(*args, **kwargs)

    monkeypatch.setattr(
        work_package_runtime, "build_partition_arrays", fail_first_partition
    )
    with pytest.raises(RuntimeError, match="partition fault injection"):
        run_work_package(
            spec_path,
            package_id,
            device="cpu",
            model_loader=loader,
            infer_tile=infer,
        )
    checkpoint_root = (
        spec_path.parent
        / "tmp"
        / "work_packages"
        / package_id
        / "score_batches"
        / "a"
    )
    assert list(checkpoint_root.glob("batch_*.npy"))
    assert list(checkpoint_root.glob("batch_*.json"))

    result = run_work_package(
        spec_path,
        package_id,
        device="cpu",
        resume=True,
        model_loader=loader,
        infer_tile=infer,
    )
    assert result["status"] == "ready"
    assert tile.is_file()
    assert _sha(tile) == source_sha256
    assert not Path(spec["cache_root"]).exists()
    assert loaded == ["a", "b"]
    run_dir = spec_path.parent
    for relative in (
        "models/a/raster_parts/partition_00000_00000_mask.tif",
        "models/b/raster_parts/partition_00000_00000_mask.tif",
        "fusion/fixture_fusion/raster_parts/partition_00000_00000_mask.tif",
    ):
        assert (run_dir / relative).is_file()
    fusion_mask = run_dir / "fusion/fixture_fusion/raster_parts/partition_00000_00000_mask.tif"
    with rasterio.open(fusion_mask) as source:
        assert np.all(source.read(1) == 0)
    assert not list((run_dir / "tmp/work_packages" / package_id / "scores").rglob("*.npz"))
    assert not list((run_dir / "tmp/work_packages" / package_id / "scores").rglob("*.json"))
    assert not (run_dir / "tmp/work_packages" / package_id / "score_batches").exists()
    assert not (run_dir / "tmp/work_packages" / package_id / "fusion").exists()
    assert result["cleaned_bytes"] > 0
    assert result["model_load_count"] == 1
    assert result["requested_device"] == "cpu"
    assert result["effective_device"] == "cpu"
    assert result["peak_cache_bytes"] > 0
    assert result["tile_cache_released_count"] == 1
    assert result["tile_cache_retained_count"] == 0
    assert result["peak_rss_bytes"] > 0
    assert result["elapsed_sec"] > 0
    assert result["models"][0]["checkpoint_reused_count"] == 1
    assert result["models"][0]["checkpoint_written_count"] == 0
    assert result["models"][1]["checkpoint_written_count"] == 1
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT status FROM work_packages WHERE package_id=?", (package_id,)
        ).fetchone()[0] == "ready"
        assert connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 9
        unit_job_id = connection.execute(
            """SELECT job_id FROM jobs WHERE stream_id='fusion:fixture_fusion'
               AND job_type='unit_fit'"""
        ).fetchone()[0]
    database = RunStateDB(database_path)
    leased = database.lease_job(unit_job_id, "unit-test", lease_seconds=60)
    assert leased is not None
    unit_report = run_unit_fit(
        spec_path,
        "fusion:fixture_fusion",
        leased["unit_id"],
        job_id=leased["job_id"],
        lease_token=leased["lease_token"],
    )
    assert unit_report["status"] == "passed"
    persisted_unit_report = database.artifact_for_stream_unit(
        spec["run_id"],
        "fusion:fixture_fusion",
        leased["unit_id"],
        "unit_boundary_report",
    )
    with open(persisted_unit_report["path"], encoding="utf-8") as handle:
        persisted_payload = json.load(handle)
    assert "diagnostics" not in persisted_payload
    assert persisted_payload["diagnostic_storage"] == {
        "mode": "none",
        "fitted_edge_count": 0,
        "raw_points_persisted": False,
        "fitted_points_in_json": False,
    }
    assert database.stream_unit_counts(spec["run_id"], "fusion:fixture_fusion") == {
        "ready": 1
    }
    probability = database.artifact_for_stream_unit(
        spec["run_id"],
        "fusion:fixture_fusion",
        "partition_00000_00000",
        "partition_probability",
    )
    assert probability["ref_count"] == 0
    assembled = assemble_stream(spec_path, "fusion:fixture_fusion")
    assert assembled["status"] == "passed"
    assert assembled["assembly_mode"] == "full"
    assert assembled["report_queue_capacity"] == 32
    assert assembled["report_summary_source"] == "sqlite"
    assert assembled["report_processed_count"] == 1
    assert assembled["report_peak_loaded_count"] == 0
    assert assembled["report_json_parse_count"] == 0
    assert assembled["gpkg_write_mode"] == "gdal_batch_writerecords"
    assert assembled["object_id_lookup_batch_size"] == 512
    events = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("{")
    ]
    completed = [
        event
        for event in events
        if event.get("event") == "report_assembly_completed"
    ]
    assert len(completed) == 1
    assert completed[0]["run_id"] == spec["run_id"]
    assert completed[0]["stream_id"] == "fusion:fixture_fusion"
    assert completed[0]["assembly_mode"] == "full"
    assert completed[0]["current"] == completed[0]["total"] == 1
    assert completed[0]["report_queue_capacity"] == 32
    assert completed[0]["report_peak_loaded_count"] == 0
    assert completed[0]["report_summary_source"] == "sqlite"
    assert completed[0]["report_json_parse_count"] == 0
    assert completed[0]["summary_validation_peak_in_flight"] >= 1
    assert completed[0]["failed_unit_count"] == 0
    assert assembled["unit_count"] == 1
    assert assembled["object_count"] == 1
    assert assembled["fit_version"] == "divider_cubic_bspline_adaptive_v2"
    assert assembled["chain_count"] >= 0
    assert assembled["shared_chain_count"] >= 0
    assert assembled["spline_count"] >= 0
    assert assembled["topology_checks_performed"] is False
    assert assembled["validation"]["scope"] == "all_output_polygons"
    assert assembled["validation"]["invalid_count"] == 0
    assert (
        spec_path.parent / "fusion/fixture_fusion/semantic_polygons.gpkg"
    ).is_file()
    assert (spec_path.parent / "fusion/fixture_fusion/fitted_edges.gpkg").is_file()
    finalized = finalize_partition_rasters(spec_path)
    assert finalized["status"] == "raster_ready"
    assert len(finalized["streams"]) == 3
    assert all(Path(item["mask_vrt"]).is_file() for item in finalized["streams"])


def test_two_models_multiple_work_packages_complete_fusion_seam_and_assembly(tmp_path):
    tile = tmp_path / "shared_tile.tif"
    with rasterio.open(
        tile,
        "w",
        driver="GTiff",
        width=1472,
        height=832,
        count=3,
        dtype="uint8",
        crs="EPSG:4490",
        transform=from_origin(0, 832, 1, 1),
        compress="deflate",
    ) as destination:
        destination.write(np.zeros((3, 832, 1472), dtype=np.uint8))
    model_a = tmp_path / "a.pt"
    model_b = tmp_path / "b.pt"
    model_a.write_bytes(b"model-a")
    model_b.write_bytes(b"model-b")
    profile = {
        "profile_id": "fixture_fusion",
        "status": "approved",
        "approval": {"passed": True},
        "strategy": "equal_probability_average",
        "models": [{"model_id": "a"}, {"model_id": "b"}],
        "weights": [[0.5, 0.5] for _ in range(14)],
    }
    stride = 320
    tiles = [
        {
            "row": row,
            "col": col,
            "path": str(tile),
            "sha256": _sha(tile),
            "bounds": {
                "xmin": col * stride,
                "ymin": 832 - row * stride - 512,
                "xmax": col * stride + 512,
                "ymax": 832 - row * stride,
            },
            "pixel_window": {
                "x0": col * stride,
                "y0": row * stride,
                "x1": col * stride + 512,
                "y1": row * stride + 512,
            },
        }
        for row in range(2)
        for col in range(4)
    ]
    scaling = {
        **_scaling(),
        "partition_tile_rows": 2,
        "partition_tile_cols": 2,
    }
    spec, spec_path, database_path = create_v5_run(
        output_root=tmp_path / "output",
        raster={
            "path": tile,
            "crs": "EPSG:4490",
            "transform": [1, 0, 0, 0, -1, 832],
            "nodata": None,
        },
        requested_extent={"xmin": 0, "ymin": 0, "xmax": 1472, "ymax": 832},
        processing_extent={"xmin": 0, "ymin": 0, "xmax": 1472, "ymax": 832},
        tile_rows=2,
        tile_cols=4,
        tiles=tiles,
        models=[
            {
                "model_id": "a",
                "artifact_path": str(model_a),
                "sha256": _sha(model_a),
                "version": "fixture",
            },
            {
                "model_id": "b",
                "artifact_path": str(model_b),
                "sha256": _sha(model_b),
                "version": "fixture",
            },
        ],
        effective_device="cpu",
        tile_batch_size=4,
        overlap=192,
        scaling=scaling,
        boundary_fitting=_boundary(),
        storage_report={
            "package_tile_limit": 6,
            "working_bytes_per_tile": 4096,
            "status": "passed",
        },
        fusion={"profile_id": "fixture_fusion", "profile": profile},
        run_id="20260717_221000_multipkg",
    )
    loaded = []

    def loader(model_entry, _device):
        loaded.append(model_entry["model_id"])
        return model_entry["model_id"]

    batch_attempts = []

    def infer_images(model_id, images, _device):
        assert images.shape[1:] == (3, 512, 512)
        batch_attempts.append(len(images))
        if len(images) > 2:
            raise RuntimeError("fixture batch capacity is two")
        probabilities = np.zeros(
            (len(images), 14, 512, 512), dtype=np.float32
        )
        probabilities[:, 0 if model_id == "a" else 1] = 1.0
        return probabilities

    database = RunStateDB(database_path)
    with sqlite3.connect(database_path) as connection:
        package_ids = [
            row[0]
            for row in connection.execute(
                "SELECT package_id FROM work_packages ORDER BY sequence_no"
            )
        ]
    assert package_ids == ["package_00000", "package_00001"]
    source_sha256 = _sha(tile)
    first_tile_ids = {
        str(item["tile_id"])
        for item in database.package_tiles(spec["run_id"], package_ids[0])
    }
    second_tile_ids = {
        str(item["tile_id"])
        for item in database.package_tiles(spec["run_id"], package_ids[1])
    }
    shared_tile_ids = first_tile_ids & second_tile_ids
    first_only_tile_ids = first_tile_ids - second_tile_ids
    assert shared_tile_ids
    assert first_only_tile_ids

    first_report = run_work_package(
        spec_path,
        package_ids[0],
        device="cpu",
        model_loader=loader,
        infer_images=infer_images,
    )
    tile_cache_dir = Path(spec["tile_cache_dir"])
    assert tile.is_file()
    assert _sha(tile) == source_sha256
    for tile_id in shared_tile_ids:
        row, col = tile_id.split("_")
        assert (tile_cache_dir / f"tile_{row}_{col}.tif").is_file()
    for tile_id in first_only_tile_ids:
        row, col = tile_id.split("_")
        assert not (tile_cache_dir / f"tile_{row}_{col}.tif").exists()

    second_report = run_work_package(
        spec_path,
        package_ids[1],
        device="cpu",
        model_loader=loader,
        infer_images=infer_images,
    )
    package_reports = [first_report, second_report]
    assert all(report["status"] == "ready" for report in package_reports)
    assert first_report["tile_cache_retained_count"] == len(shared_tile_ids)
    assert second_report["tile_cache_retained_count"] == 0
    assert tile.is_file()
    assert _sha(tile) == source_sha256
    assert not Path(spec["cache_root"]).exists()
    assert loaded == ["a", "b", "a", "b"]
    assert 4 in batch_attempts
    assert 2 in batch_attempts
    assert all(
        model["configured_tile_batch_size"] == 4
        and model["effective_tile_batch_size"] == 2
        and model["peak_tile_batch_size"] <= 2
        and model["batch_reduction_count"] >= 1
        and model["input_queue_capacity"] == 2
        and model["input_queue_peak_batches"] <= 2
        and model["result_queue_capacity"] == 1
        and model["result_queue_peak_batches"] <= 1
        and model["checkpoint_written_count"] >= 1
        for report in package_reports
        for model in report["models"]
    )
    assert database.work_package_counts(spec["run_id"]) == {"ready": 2}
    for package_id in package_ids:
        with sqlite3.connect(database_path) as connection:
            package_job_id = connection.execute(
                "SELECT job_id FROM jobs WHERE job_type='work_package' AND package_id=?",
                (package_id,),
            ).fetchone()[0]
        leased = database.lease_job(
            package_job_id, "multi-package-test", lease_seconds=120
        )
        assert leased is not None
        assert database.finish_job(
            leased["job_id"], leased["lease_token"], status="ready"
        )

    finalized = finalize_partition_rasters(spec_path)
    assert finalized["status"] == "raster_ready"
    assert len(finalized["streams"]) == 3

    with sqlite3.connect(database_path) as connection:
        unit_job_ids = [
            row[0]
            for row in connection.execute(
                "SELECT job_id FROM jobs WHERE job_type='unit_fit' ORDER BY job_id"
            )
        ]
    assert len(unit_job_ids) == 9
    for job_id in unit_job_ids:
        leased = database.lease_job(job_id, "multi-package-test", lease_seconds=120)
        assert leased is not None
        report = run_unit_fit(
            spec_path,
            leased["stream_id"],
            leased["unit_id"],
            job_id=leased["job_id"],
            lease_token=leased["lease_token"],
        )
        assert report["status"] == "passed"

    assembly_reports = [
        assemble_stream(spec_path, stream["stream_id"])
        for stream in spec["streams"]
    ]
    assert all(report["status"] == "passed" for report in assembly_reports)
    assert all(report["unit_count"] == 3 for report in assembly_reports)
    assert all(report["validation"]["passed"] for report in assembly_reports)
    assert all(
        report["validation"]["scope"] == "all_output_polygons"
        for report in assembly_reports
    )
    assert {row["status"] for row in database.stream_rows(spec["run_id"])} == {"ready"}

    scale_report = build_scale_acceptance_report(spec_path)
    assert scale_report["hard_gate_passed"] is True
    assert scale_report["status"] == "passed"
    assert scale_report["package_count"] == 2
    assert scale_report["spatial_unit_count"] == 3
    assert scale_report["model_load_counts"] == {"a": 2, "b": 2}
    assert scale_report["failed_count"] == 0
    assert scale_report["retry_count"] == 0
    assert scale_report["peak_cache_bytes"] > 0
    assert scale_report["peak_rss_bytes"] > 0
    assert scale_report["cleaned_bytes"] > 0
    assert not scale_report["artifact_integrity_errors"]

    for relative in (
        "models/a/semantic_polygons.gpkg",
        "models/b/semantic_polygons.gpkg",
        "fusion/fixture_fusion/semantic_polygons.gpkg",
    ):
        assert (spec_path.parent / relative).is_file()

    corrupted = spec_path.parent / "fusion/fixture_fusion/semantic_polygons.gpkg"
    with corrupted.open("ab") as handle:
        handle.write(b"corrupt-after-commit")
    rejected = build_scale_acceptance_report(spec_path)
    assert rejected["hard_gate_passed"] is False
    assert rejected["status"] == "failed"
    assert rejected["hard_gates"]["no_artifact_integrity_errors"] is False
    assert any(
        str(corrupted) in item for item in rejected["artifact_integrity_errors"]
    )


def test_multi_partition_seam_junction_assembly_is_gap_free(tmp_path):
    run_id = "20260717_230000_multi_unit"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state_path = run_dir / "run_state.sqlite"
    class_path = run_dir / "class_mapping_snapshot.json"
    atomic_write_json(
        class_path,
        {
            "class_mapping": {"12": "水浇地"},
            "index_to_code": {"0": 12},
            "background_index": -1,
        },
    )
    plan = plan_spatial_units(
        tile_rows=9,
        tile_cols=9,
        overlap=192,
        partition_tile_rows=8,
        partition_tile_cols=8,
        seam_band_px=64,
        halo_px=192,
    )
    spec = {
        "schema_version": 2,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "state_db": str(state_path),
        "raster": {
            "crs": "EPSG:4490",
            "transform": [1, 0, 0, 0, -1, plan["processing_window"]["y1"]],
        },
        "streams": [
            {
                "stream_id": "model:a",
                "kind": "model",
                "model_id": "a",
                "version": "fixture",
            }
        ],
        "class_mapping_snapshot": str(class_path),
        "accepted_gpkg": "",
        "boundary_fitting": _boundary(),
    }
    spec_path = run_dir / "run_spec.json"
    atomic_write_json(spec_path, spec)
    database = RunStateDB(state_path)
    database.initialize()
    database.create_run(run_id, _sha(spec_path))
    database.register_streams(run_id, spec["streams"])
    database.insert_work_packages(
        run_id, [{"package_id": "package_00000", "sequence_no": 0}]
    )
    database.insert_partitions(
        run_id,
        [
            {**partition, "package_id": "package_00000"}
            for partition in plan["partitions"]
        ],
    )
    database.insert_spatial_units(run_id, plan["spatial_units"])
    database.insert_stream_units(
        run_id,
        ["model:a"],
        [unit["unit_id"] for unit in plan["spatial_units"]],
    )
    affine = Affine(*spec["raster"]["transform"])
    output_root = run_dir / "tmp" / "unit_outputs" / "model_a"
    for unit in plan["spatial_units"]:
        window = unit["pixel_window"]
        geometry = box(window["x0"], window["y0"], window["x1"], window["y1"])
        base = {
            "polygon_id": f"{unit['unit_id']}_0000000",
            "class_code": 12,
            "geometry": geometry,
            "confidence_mean": 1.0,
            "confidence_std": 0.0,
        }
        raw_path = output_root / f"{unit['unit_id']}_raw.gpkg"
        formal_path = output_root / f"{unit['unit_id']}_formal.gpkg"
        report_path = output_root / f"{unit['unit_id']}_report.json"
        _write_gpkg(
            raw_path,
            [base],
            transform=affine,
            crs="EPSG:4490",
            include_fit=False,
        )
        _write_gpkg(
            formal_path,
            [
                {
                    **base,
                    "fit_method": "unchanged",
                    "fit_status": "unchanged",
                    "fit_version": "divider_cubic_bspline_adaptive_v2",
                    "vertex_count_before": 5,
                    "vertex_count_after": 5,
                    "max_shift_px": 0.0,
                    "mean_shift_px": 0.0,
                    "area_change_ratio": 0.0,
                }
            ],
            transform=affine,
            crs="EPSG:4490",
            include_fit=True,
        )
        diagnostics = (
            [
                {
                    "chain_id": "fixture_edge",
                    "method": "cubic_bspline_adaptive",
                    "status": "changed",
                    "fitted_points": [[0.0, 0.0], [1.0, 1.0]],
                    "max_displacement_px": 0.25,
                    "point_count_dense": 17,
                    "point_count_after": 2,
                    "max_chord_error_px": 0.125,
                    "max_segment_arc_length_px": 1.5,
                }
            ]
            if unit["unit_id"] == plan["spatial_units"][0]["unit_id"]
            else []
        )
        unit_report = {
            "status": "passed",
            "fit_version": "divider_cubic_bspline_adaptive_v2",
            "chain_count": 0,
            "shared_chain_count": 0,
            "spline_count": 0,
            "unchanged_count": 1,
            "max_displacement_px": 0.0,
            "diagnostics": diagnostics,
            "validation": {
                "passed": True,
                "scope": "all_output_polygons",
                "invalid_count": 0,
            },
        }
        report_path.write_text(json.dumps(unit_report), encoding="utf-8")
        fitted_edges_path = output_root / f"{unit['unit_id']}_fitted_edges.gpkg"
        fitted_edge_count = _write_diagnostic_gpkg(
            fitted_edges_path,
            unit_report,
            run_id=run_id,
            stream_id="model:a",
            unit_id=unit["unit_id"],
            transform=affine,
            crs="EPSG:4490",
        )
        artifacts = [
            ("unit_raw", raw_path),
            ("unit_formal", formal_path),
            ("unit_boundary_report", report_path),
        ]
        if fitted_edge_count:
            artifacts.append(("unit_fitted_edges", fitted_edges_path))
        for kind, path in artifacts:
            _commit_artifact(
                database,
                run_id,
                path=path,
                kind=kind,
                stream_id="model:a",
                unit_id=unit["unit_id"],
            )
        database.upsert_unit_report_summary(
            run_id,
            "model:a",
            unit["unit_id"],
            unit_report,
            fitted_edge_count=fitted_edge_count,
        )
        database.set_stream_unit_status(run_id, "model:a", unit["unit_id"], "ready")

    report = assemble_stream(spec_path, "model:a")
    assert report["unit_count"] == 9
    assert report["object_count"] == 1
    assert report["object_link_count"] > 0
    assert report["validation"]["passed"] is True
    assert report["validation"]["scope"] == "all_output_polygons"
    assert report["validation"]["invalid_count"] == 0
    assert report["curve_sampling_spacing_px"] == 0.5
    assert report["max_chord_error_limit_px"] == 0.25
    assert report["max_segment_arc_length_limit_px"] == 8.0
    assert report["dense_curve_point_count"] == 17
    assert report["sparse_curve_point_count"] == 2
    assert report["max_chord_error_px"] == 0.125
    assert report["max_segment_arc_length_px"] == 1.5
    assert report["adaptive_point_reduction"] == pytest.approx(15 / 17)
    with fiona.open(run_dir / "models/a/semantic_polygons.gpkg") as source:
        features = list(source)
    with fiona.open(run_dir / "models/a/fitted_edges.gpkg") as source:
        assert len(source) == 1
    assert len({feature["properties"]["object_id"] for feature in features}) == 1
    merged = unary_union([shape(feature["geometry"]) for feature in features])
    processing = plan["processing_window"]
    expected = box(
        processing["x0"],
        0,
        processing["x1"],
        processing["y1"],
    )
    assert merged.symmetric_difference(expected).area == 0
    reassembled = assemble_stream(spec_path, "model:a")
    assert reassembled["assembly_mode"] == "reused"
    assert reassembled["object_link_count"] == report["object_link_count"]


def test_full_assembly_streams_64_spatial_unit_reports(tmp_path):
    run_id = "20260729_120000_queue64"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state_path = run_dir / "run_state.sqlite"
    class_path = run_dir / "class_mapping_snapshot.json"
    atomic_write_json(
        class_path,
        {
            "class_mapping": {"12": "水浇地"},
            "index_to_code": {"0": 12},
            "background_index": -1,
        },
    )
    units = [
        {
            "unit_id": f"core_{index:05d}",
            "unit_type": "core",
            "owner_key": f"owner_{index:05d}",
            "pixel_window": {
                "x0": index * 2,
                "y0": 0,
                "x1": index * 2 + 1,
                "y1": 1,
            },
            "dependency_ids": [],
        }
        for index in range(64)
    ]
    spec = {
        "schema_version": 2,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "state_db": str(state_path),
        "raster": {
            "crs": "EPSG:4490",
            "transform": [1, 0, 0, 0, -1, 1],
        },
        "streams": [
            {
                "stream_id": "model:a",
                "kind": "model",
                "model_id": "a",
                "version": "fixture",
            }
        ],
        "class_mapping_snapshot": str(class_path),
        "accepted_gpkg": "",
        "boundary_fitting": _boundary(),
    }
    spec_path = run_dir / "run_spec.json"
    atomic_write_json(spec_path, spec)
    database = RunStateDB(state_path)
    database.initialize()
    database.create_run(run_id, _sha(spec_path))
    database.register_streams(run_id, spec["streams"])
    database.insert_spatial_units(run_id, units)
    database.insert_stream_units(
        run_id,
        ["model:a"],
        [unit["unit_id"] for unit in units],
    )
    affine = Affine(*spec["raster"]["transform"])
    output_root = run_dir / "tmp" / "unit_outputs" / "model_a"

    for unit in units:
        window = unit["pixel_window"]
        geometry = box(
            window["x0"],
            window["y0"],
            window["x1"],
            window["y1"],
        )
        base = {
            "polygon_id": f"{unit['unit_id']}_0000000",
            "class_code": 12,
            "geometry": geometry,
            "confidence_mean": 1.0,
            "confidence_std": 0.0,
        }
        raw_path = output_root / f"{unit['unit_id']}_raw.gpkg"
        formal_path = output_root / f"{unit['unit_id']}_formal.gpkg"
        report_path = output_root / f"{unit['unit_id']}_report.json"
        _write_gpkg(
            raw_path,
            [base],
            transform=affine,
            crs="EPSG:4490",
            include_fit=False,
        )
        _write_gpkg(
            formal_path,
            [
                {
                    **base,
                    "fit_method": "unchanged",
                    "fit_status": "unchanged",
                    "fit_version": "divider_cubic_bspline_adaptive_v2",
                    "vertex_count_before": 5,
                    "vertex_count_after": 5,
                    "max_shift_px": 0.0,
                    "mean_shift_px": 0.0,
                    "area_change_ratio": 0.0,
                }
            ],
            transform=affine,
            crs="EPSG:4490",
            include_fit=True,
        )
        unit_report = {
            "status": "passed",
            "fit_version": "divider_cubic_bspline_adaptive_v2",
            "chain_count": 0,
            "shared_chain_count": 0,
            "spline_count": 0,
            "unchanged_count": 1,
            "max_displacement_px": 0.0,
            "diagnostics": [],
            "validation": {
                "passed": True,
                "scope": "all_output_polygons",
                "invalid_count": 0,
            },
        }
        report_path.write_text(json.dumps(unit_report), encoding="utf-8")
        for kind, path in (
            ("unit_raw", raw_path),
            ("unit_formal", formal_path),
            ("unit_boundary_report", report_path),
        ):
            _commit_artifact(
                database,
                run_id,
                path=path,
                kind=kind,
                stream_id="model:a",
                unit_id=unit["unit_id"],
            )
        database.upsert_unit_report_summary(
            run_id,
            "model:a",
            unit["unit_id"],
            unit_report,
        )
        database.set_stream_unit_status(
            run_id,
            "model:a",
            unit["unit_id"],
            "ready",
        )

    report = assemble_stream(spec_path, "model:a")

    assert report["status"] == "passed"
    assert report["assembly_mode"] == "full"
    assert report["unit_count"] == 64
    assert report["object_count"] == 64
    assert report["report_processed_count"] == 64
    assert report["report_queue_capacity"] == 32
    assert report["report_peak_loaded_count"] == 0
    assert report["report_summary_source"] == "sqlite"
    assert report["report_json_parse_count"] == 0
    assert report["summary_validation_peak_in_flight"] <= 32
    stream_root = run_dir / "models" / "a"
    raw_path = stream_root / "semantic_polygons_raw.gpkg"
    formal_path = stream_root / "semantic_polygons.gpkg"
    report_path = stream_root / "boundary_fitting_report.json"
    fitted_edges_path = stream_root / "fitted_edges.gpkg"
    with fiona.open(formal_path) as source:
        assert len(source) == 64
    with fiona.open(fitted_edges_path) as source:
        assert len(source) == 0
    assert not list(stream_root.glob(".*.stage.*"))
    assert not list(stream_root.glob(".*.tmp.*"))

    output_hashes = {
        "raw": _sha(raw_path),
        "formal": _sha(formal_path),
        "report": _sha(report_path),
        "fitted_edges": _sha(fitted_edges_path),
    }
    broken_unit_report = output_root / "core_00032_report.json"
    valid_unit_report = broken_unit_report.read_text(encoding="utf-8")
    broken_unit_report.write_text("{broken-json", encoding="utf-8")
    database.set_stream_status(
        run_id,
        "model:a",
        "failed",
        error="injected report recovery failure",
    )

    with pytest.raises(
        StreamAssemblyError,
        match="unit report Artifact changed",
    ):
        assemble_stream(spec_path, "model:a", resume_from_reports=True)

    stream = {
        row["stream_id"]: row for row in database.stream_rows(run_id)
    }["model:a"]
    assert stream["status"] == "failed"
    assert _sha(raw_path) == output_hashes["raw"]
    assert _sha(formal_path) == output_hashes["formal"]
    assert _sha(report_path) == output_hashes["report"]
    assert _sha(fitted_edges_path) == output_hashes["fitted_edges"]
    assert not list(stream_root.glob(".*.stage.*"))
    assert not list(stream_root.glob(".*.tmp.*"))

    broken_unit_report.write_text(valid_unit_report, encoding="utf-8")
    with sqlite3.connect(state_path) as connection:
        connection.execute(
            """UPDATE artifacts SET status='failed'
               WHERE run_id=? AND stream_id=? AND unit_id='assembled'
               AND kind IN ('boundary_fitting_report', 'fitted_edges')""",
            (run_id, "model:a"),
        )
    resumed = assemble_stream(spec_path, "model:a", resume_from_reports=True)

    assert resumed["status"] == "passed"
    assert resumed["assembly_mode"] == "report_resume"
    assert resumed["report_processed_count"] == 64
    assert resumed["report_summary_source"] == "sqlite"
    assert resumed["report_json_parse_count"] == 0
    assert _sha(raw_path) == output_hashes["raw"]
    assert _sha(formal_path) == output_hashes["formal"]
    assert not list(stream_root.glob(".*.stage.*"))
    assert not list(stream_root.glob(".*.tmp.*"))
    reused = assemble_stream(spec_path, "model:a")
    assert reused["assembly_mode"] == "reused"
