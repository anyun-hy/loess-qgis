"""Automated unit tests for adaptive range clipping runtime."""

from __future__ import annotations

import os
import hashlib
from pathlib import Path
import tempfile
import pytest
import fiona
from fiona.crs import CRS
from shapely.geometry import Polygon, box, mapping

from inference_scripts.range_clip_runtime import (
    apply_adaptive_range_clip,
    extract_range_mask_geometry,
    RangeClipRuntimeError,
)


@pytest.fixture
def sample_data(tmp_path: Path):
    """Create test source polygons and mask vector layers."""
    crs = CRS.from_epsg(3857)
    schema = {
        "geometry": "Polygon",
        "properties": {
            "object_id": "int",
            "class_code": "int",
            "part_id": "str",
        },
    }

    # 1. Create source polygons: 4 squares covering (0, 0) to (200, 200)
    source_path = tmp_path / "semantic_polygons.gpkg"
    with fiona.open(source_path, "w", driver="GPKG", layer="semantic_polygons", schema=schema, crs=crs) as dst:
        polygons = [
            (Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]), 1, 13, "part_01"),
            (Polygon([(100, 0), (200, 0), (200, 100), (100, 100)]), 2, 31, "part_02"),
            (Polygon([(0, 100), (100, 100), (100, 200), (0, 200)]), 3, 52, "part_03"),
            (Polygon([(100, 100), (200, 100), (200, 200), (100, 200)]), 4, 71, "part_04"),
        ]
        for poly, obj_id, code, part_id in polygons:
            dst.write({
                "geometry": mapping(poly),
                "properties": {"object_id": obj_id, "class_code": code, "part_id": part_id},
            })

    # 2. Create vector range mask: Circle/Polygon centered at (100, 100) with radius 60
    mask_path = tmp_path / "custom_range_mask.gpkg"
    mask_poly = Polygon([(50, 50), (150, 50), (150, 150), (50, 150)])
    mask_schema = {"geometry": "Polygon", "properties": {"name": "str"}}
    with fiona.open(mask_path, "w", driver="GPKG", layer="mask", schema=mask_schema, crs=crs) as dst:
        dst.write({
            "geometry": mapping(mask_poly),
            "properties": {"name": "test_mask"},
        })

    return {
        "source_path": source_path,
        "mask_path": mask_path,
        "tmp_path": tmp_path,
        "crs": crs,
    }


def test_clip_with_vector_mask(sample_data):
    """Test clipping with an explicit vector range file."""
    source_path = sample_data["source_path"]
    mask_path = sample_data["mask_path"]

    spec = {
        "range_vector_path": str(mask_path),
        "requested_extent": {"xmin": 0, "ymin": 0, "xmax": 200, "ymax": 200},
    }

    result = apply_adaptive_range_clip(source_path, spec)
    assert result["status"] == "passed"
    assert result["source_feature_count"] == 4
    assert result["output_feature_count"] == 4
    assert result["trimmed_feature_count"] == 4

    # Verify geometries are inside (50, 50, 150, 150)
    with fiona.open(source_path, layer="semantic_polygons") as src:
        assert len(src) == 4
        for feature in src:
            bounds = fiona.bounds(feature["geometry"])
            assert bounds[0] >= 50 - 1e-5
            assert bounds[1] >= 50 - 1e-5
            assert bounds[2] <= 150 + 1e-5
            assert bounds[3] <= 150 + 1e-5


def test_clip_with_requested_extent_bbox(sample_data):
    """Test clipping with requested_extent when no vector file is specified."""
    source_path = sample_data["source_path"]

    spec = {
        "range_vector_path": "",
        "requested_extent": {
            "xmin": 20,
            "ymin": 20,
            "xmax": 80,
            "ymax": 80,
        },
    }

    result = apply_adaptive_range_clip(source_path, spec)
    assert result["status"] == "passed"
    assert result["source_feature_count"] == 4
    # Only the first polygon (0..100, 0..100) overlaps (20..80, 20..80)
    assert result["output_feature_count"] == 1
    assert result["discarded_feature_count"] == 3

    with fiona.open(source_path, layer="semantic_polygons") as src:
        features = list(src)
        assert len(features) == 1
        assert features[0]["properties"]["class_code"] == 13
        bounds = fiona.bounds(features[0]["geometry"])
        assert bounds[0] == pytest.approx(20)
        assert bounds[1] == pytest.approx(20)
        assert bounds[2] == pytest.approx(80)
        assert bounds[3] == pytest.approx(80)


def test_vector_mode_rejects_missing_vector_instead_of_using_requested_extent(
    sample_data,
):
    """A vector-range run must never silently publish its bounding rectangle."""
    spec = {
        "range_selection": {
            "mode": "vector_tile_intersection",
            "vector_source": str(sample_data["tmp_path"] / "missing.gpkg"),
            "clip_outputs": True,
        },
        "requested_extent": {"xmin": 0, "ymin": 0, "xmax": 200, "ymax": 200},
    }

    with pytest.raises(RangeClipRuntimeError, match="cannot read required vector"):
        extract_range_mask_geometry(spec, "EPSG:3857")


def test_extent_mode_uses_requested_extent_even_if_legacy_vector_path_is_present(
    sample_data,
):
    """View and hand-drawn extent runs retain the requested rectangle contract."""
    spec = {
        "range_selection": {"mode": "extent", "clip_outputs": True},
        "range_vector_path": str(sample_data["mask_path"]),
        "requested_extent": {"xmin": 20, "ymin": 20, "xmax": 80, "ymax": 80},
    }

    geometry = extract_range_mask_geometry(spec, "EPSG:3857")

    assert geometry.bounds == pytest.approx((20, 20, 80, 80))


def test_vector_mode_intersects_exact_boundary_with_available_raster_extent(
    sample_data,
):
    """Topology and publication must not demand coverage outside the raster."""
    mask_path = sample_data["mask_path"]
    spec = {
        "range_selection": {
            "mode": "vector_tile_intersection",
            "vector_source": str(mask_path),
            "vector_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
            "clip_outputs": True,
        },
        "raster": {"crs": "EPSG:3857"},
        "requested_extent": {
            "xmin": 0,
            "ymin": 0,
            "xmax": 100,
            "ymax": 100,
        },
    }

    geometry = extract_range_mask_geometry(spec, "EPSG:3857")

    assert geometry.bounds == pytest.approx((50, 50, 100, 100))


def test_vector_range_reprojects_wgs84_to_web_mercator_on_current_gdal(tmp_path):
    mask_path = tmp_path / "wgs84_range.gpkg"
    schema = {"geometry": "Polygon", "properties": {"name": "str"}}
    with fiona.open(
        mask_path,
        "w",
        driver="GPKG",
        layer="range",
        schema=schema,
        crs=CRS.from_epsg(4326),
    ) as destination:
        destination.write(
            {
                "geometry": mapping(box(0.0, 0.0, 0.001, 0.001)),
                "properties": {"name": "cross-crs"},
            }
        )
    spec = {
        "run_dir": str(tmp_path),
        "raster": {"crs": "EPSG:3857"},
        "range_selection": {
            "mode": "vector_tile_intersection",
            "vector_source": str(mask_path),
            "vector_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
            "clip_outputs": True,
        },
        "requested_extent": {
            "xmin": -1.0,
            "ymin": -1.0,
            "xmax": 200.0,
            "ymax": 200.0,
            "crs": "EPSG:3857",
        },
    }

    geometry = extract_range_mask_geometry(spec, "EPSG:3857")

    assert geometry.bounds == pytest.approx(
        (0.0, 0.0, 111.319490793, 111.319490799), rel=1e-8
    )


def test_reapplying_the_same_range_clip_keeps_the_formal_gpkg_fingerprint(
    sample_data,
):
    """A report-resume safety gate must not rewrite an already exact result."""
    source_path = sample_data["source_path"]
    spec = {
        "range_selection": {"mode": "extent", "clip_outputs": True},
        "requested_extent": {"xmin": 20, "ymin": 20, "xmax": 80, "ymax": 80},
    }
    apply_adaptive_range_clip(source_path, spec)
    before = hashlib.sha256(source_path.read_bytes()).hexdigest()

    result = apply_adaptive_range_clip(source_path, spec)

    assert result["status"] == "already_clipped"
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == before


def test_already_clipped_source_is_copied_when_a_distinct_output_is_requested(
    sample_data,
):
    source_path = sample_data["source_path"]
    output_path = sample_data["tmp_path"] / "copied_formal.gpkg"
    spec = {
        "range_selection": {"mode": "extent", "clip_outputs": True},
        "requested_extent": {"xmin": 20, "ymin": 20, "xmax": 80, "ymax": 80},
    }
    apply_adaptive_range_clip(source_path, spec)

    result = apply_adaptive_range_clip(source_path, spec, output_path=output_path)

    assert result["status"] == "already_clipped"
    assert output_path.is_file()
