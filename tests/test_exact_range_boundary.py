"""Regression behaviour for exact, immutable range boundaries."""

from __future__ import annotations

from pathlib import Path

import fiona
import numpy as np
import pytest
from affine import Affine
from fiona.crs import CRS
from shapely.geometry import box, mapping

from authoritative_raster import regularize_partition_core
from inference_scripts.range_clip_runtime import extract_range_mask_geometry
from labeling_tool.core.run_builder_v5 import create_v5_run
from labeling_tool.core.run_spec import reserve_run_directory, sha256_file
import work_package_runtime


def _write_range_snapshot(path: Path) -> Path:
    with fiona.open(
        path,
        "w",
        driver="GPKG",
        layer="range_mask",
        schema={"geometry": "Polygon", "properties": {}},
        crs=CRS.from_epsg(3857),
    ) as destination:
        destination.write({"geometry": mapping(box(0, 0, 20, 40)), "properties": {}})
    return path


def test_vector_range_mask_invalidates_pixels_outside_the_published_core(
    tmp_path: Path,
):
    """Halo inference may extend past the polygon, but the Core never may."""
    snapshot = _write_range_snapshot(tmp_path / "range_snapshot.gpkg")
    probabilities = np.zeros((14, 4, 4), dtype=np.float32)
    probabilities[2, :, :] = 1.0
    partition = {
        "partition_id": "partition_00000_00000",
        "halo_window": {"x0": 0, "y0": 0, "x1": 4, "y1": 4},
        "core_window": {"x0": 0, "y0": 0, "x1": 4, "y1": 4},
    }
    range_spec = {
        "range_selection": {
            "mode": "vector_tile_intersection",
            "vector_source": str(snapshot),
            "vector_sha256": sha256_file(snapshot),
            "clip_outputs": True,
        },
    }

    arrays, _report = regularize_partition_core(
        {
            "halo_probabilities": probabilities,
            "halo_weights": np.ones((4, 4), dtype=np.float32),
            "core_mask": np.full((4, 4), 2, dtype=np.int16),
            "core_confidence": np.ones((4, 4), dtype=np.float32),
        },
        partition,
        global_transform=Affine(10, 0, 0, 0, -10, 40),
        crs="EPSG:3857",
        range_geometry=extract_range_mask_geometry(range_spec, "EPSG:3857"),
    )

    assert np.all(arrays["core_mask"][:, :2] == 2)
    assert np.all(arrays["core_mask"][:, 2:] == -1)
    assert np.all(arrays["core_confidence"][:, 2:] == -1.0)


def test_vector_run_spec_records_the_run_local_range_snapshot_hash(tmp_path: Path):
    """The frozen range snapshot, not a mutable external layer, is the run input."""
    raster = tmp_path / "source.tif"
    model = tmp_path / "model.pt"
    raster.write_bytes(b"raster")
    model.write_bytes(b"model")
    output_root = tmp_path / "output"
    run_id, run_dir = reserve_run_directory(output_root, "20260820_010000_range")
    snapshot = _write_range_snapshot(run_dir / "range_snapshot.gpkg")
    selection = {
        "mode": "vector_tile_intersection",
        "vector_source": str(snapshot),
        "clip_outputs": True,
        "selected_tile_count": 4,
        "excluded_tile_count": 0,
    }

    spec, _path, _database = create_v5_run(
        output_root=output_root,
        reserved_run_dir=run_dir,
        run_id=run_id,
        state_database=tmp_path / "state.sqlite",
        raster={
            "path": raster,
            "crs": "EPSG:3857",
            "transform": [10, 0, 0, 0, -10, 40],
            "nodata": None,
        },
        requested_extent={"xmin": 0, "ymin": 0, "xmax": 40, "ymax": 40},
        processing_extent={"xmin": 0, "ymin": 0, "xmax": 40, "ymax": 40},
        tile_rows=2,
        tile_cols=2,
        tiles=[
            {
                "row": row,
                "col": col,
                "path": str(raster),
                "sha256": "a" * 64,
                "pixel_window": {
                    "x0": col * 320,
                    "y0": row * 320,
                    "x1": col * 320 + 512,
                    "y1": row * 320 + 512,
                },
            }
            for row in range(2)
            for col in range(2)
        ],
        models=[
            {
                "model_id": "fixture",
                "artifact_path": str(model),
                "sha256": "b" * 64,
                "version": "fixture",
            }
        ],
        effective_device="cpu",
        overlap=192,
        scaling={
            "partition_tile_rows": 2,
            "partition_tile_cols": 2,
            "partition_halo_px": 256,
            "seam_band_px": 64,
            "max_job_retries": 2,
        },
        boundary_fitting={"enabled": True},
        storage_report={
            "package_tile_limit": 4,
            "working_bytes_per_tile": 1024,
            "status": "passed",
        },
        range_selection=selection,
    )

    assert spec["range_selection"]["clip_outputs"] is True
    assert Path(spec["range_vector_path"]) == snapshot.resolve()
    assert spec["range_selection"]["vector_sha256"] == sha256_file(snapshot)


def test_worker_rejects_a_range_snapshot_that_changed_after_run_creation(
    tmp_path: Path,
):
    """A retry must not use a range boundary different from the frozen RunSpec."""
    snapshot = _write_range_snapshot(tmp_path / "range_snapshot.gpkg")
    spec = {
        "range_selection": {
            "mode": "vector_tile_intersection",
            "vector_source": str(snapshot),
            "vector_sha256": sha256_file(snapshot),
            "clip_outputs": True,
        }
    }
    with snapshot.open("ab") as destination:
        destination.write(b"changed")

    with pytest.raises(work_package_runtime.WorkPackageRuntimeError, match="SHA256 changed"):
        work_package_runtime._range_geometry_for_run(spec, "EPSG:3857")
