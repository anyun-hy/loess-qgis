import hashlib

import fiona
import numpy as np
import rasterio
from fiona.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from labeling_tool.core.run_builder_v5 import create_v5_run
from labeling_tool.core.run_state_db import RunStateDB
from accepted_score import AcceptedScoreError, accepted_probabilities
from work_package_runtime import run_work_package


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tile(path):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=512,
        height=512,
        count=3,
        dtype="uint8",
        crs="EPSG:4490",
        transform=from_origin(0, 512, 1, 1),
    ) as destination:
        destination.write(np.zeros((3, 512, 512), dtype=np.uint8))


def _accepted(path, geometry):
    with fiona.open(
        path,
        "w",
        driver="GPKG",
        layer="accepted_labels",
        schema={"geometry": "Polygon", "properties": {"class_code": "int"}},
        crs=CRS.from_epsg(4490),
    ) as destination:
        destination.write(
            {"geometry": mapping(geometry), "properties": {"class_code": 13}}
        )


def test_fully_accepted_tile_becomes_one_hot_probability(tmp_path):
    tile = tmp_path / "tile.tif"
    accepted = tmp_path / "accepted.gpkg"
    _tile(tile)
    _accepted(accepted, box(0, 0, 512, 512))
    probabilities = accepted_probabilities(accepted, tile)
    assert probabilities.shape == (14, 512, 512)
    assert np.all(probabilities[1] == 1)
    assert np.all(probabilities.sum(axis=0) == 1)


def test_incompletely_accepted_tile_is_rejected(tmp_path):
    tile = tmp_path / "tile.tif"
    accepted = tmp_path / "accepted.gpkg"
    _tile(tile)
    _accepted(accepted, box(0, 0, 256, 512))
    try:
        accepted_probabilities(accepted, tile)
    except AcceptedScoreError as error:
        assert "uncovered pixels" in str(error)
    else:
        raise AssertionError("partially accepted Tile was treated as fully accepted")


def test_work_package_skips_model_for_fully_accepted_tile(tmp_path):
    tile = tmp_path / "tile.tif"
    accepted = tmp_path / "accepted.gpkg"
    model = tmp_path / "model.pt"
    _tile(tile)
    _accepted(accepted, box(0, 0, 512, 512))
    model.write_bytes(b"fixture")
    spec, spec_path, database_path = create_v5_run(
        state_database=tmp_path / "state.sqlite",
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
                "status": "accepted",
                "pixel_window": {"x0": 0, "y0": 0, "x1": 512, "y1": 512},
            }
        ],
        models=[
            {
                "model_id": "a",
                "artifact_path": str(model),
                "sha256": _sha(model),
                "version": "fixture",
            }
        ],
        effective_device="cpu",
        overlap=192,
        scaling={
            "partition_tile_rows": 8,
            "partition_tile_cols": 8,
            "partition_halo_px": 192,
            "seam_band_px": 64,
            "max_job_retries": 2,
        },
        boundary_fitting={"enabled": True},
        storage_report={
            "package_tile_limit": 4,
            "working_bytes_per_tile": 4096,
            "status": "passed",
        },
        accepted_gpkg=accepted,
        skip_accepted=True,
        run_id="20260717_230000_acce55",
    )
    database = RunStateDB(database_path)
    package_id = database.page_work_packages(spec["run_id"], limit=1)[0]["package_id"]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("semantic model was invoked for a fully accepted Tile")

    result = run_work_package(
        spec_path,
        package_id,
        device="cpu",
        model_loader=forbidden,
        infer_tile=forbidden,
    )
    assert result["models"][0]["inferred_count"] == 0
    assert result["models"][0]["accepted_count"] == 1
    mask = (
        spec_path.parent
        / "models/a/raster_parts/partition_00000_00000_mask.tif"
    )
    with rasterio.open(mask) as source:
        assert np.all(source.read(1) == 1)
