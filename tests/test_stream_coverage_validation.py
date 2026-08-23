from __future__ import annotations

from pathlib import Path

import fiona
import pytest
from affine import Affine
from fiona.crs import CRS
from shapely.geometry import box, mapping

from assemble_stream import _validate_exact_range_coverage


def _write_polygons(path: Path, polygons) -> Path:
    with fiona.open(
        path,
        "w",
        driver="GPKG",
        layer="semantic_polygons",
        schema={"geometry": "Polygon", "properties": {"class_code": "int"}},
        crs=CRS.from_epsg(3857),
    ) as destination:
        for index, geometry in enumerate(polygons):
            destination.write(
                {
                    "geometry": mapping(geometry),
                    "properties": {"class_code": 12 + index},
                }
            )
    return path


def _spec():
    return {
        "raster": {
            "crs": "EPSG:3857",
            "transform": list(Affine(1, 0, 0, 0, -1, 10))[:6],
        },
        "requested_extent": {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10},
        "coverage_validation": {"area_tolerance_pixels": 0.01},
    }


def test_exact_range_coverage_accepts_gap_free_non_overlapping_polygons(tmp_path):
    path = _write_polygons(
        tmp_path / "passed.gpkg",
        [box(0, 0, 5, 10), box(5, 0, 10, 10)],
    )

    report = _validate_exact_range_coverage(
        path,
        layer="semantic_polygons",
        spec=_spec(),
    )

    assert report["status"] == "passed"
    assert report["gap_area_m2"] == pytest.approx(0.0)
    assert report["overlap_area_m2"] == pytest.approx(0.0)
    assert report["outside_area_m2"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("name", "polygons", "failed_metric"),
    [
        ("gap", [box(0, 0, 4, 10), box(5, 0, 10, 10)], "gap_area_m2"),
        ("overlap", [box(0, 0, 6, 10), box(5, 0, 10, 10)], "overlap_area_m2"),
        ("outside", [box(0, 0, 11, 10)], "outside_area_m2"),
    ],
)
def test_exact_range_coverage_reports_each_hard_failure(
    tmp_path,
    name,
    polygons,
    failed_metric,
):
    path = _write_polygons(tmp_path / f"{name}.gpkg", polygons)

    report = _validate_exact_range_coverage(
        path,
        layer="semantic_polygons",
        spec=_spec(),
    )

    assert report["status"] == "failed"
    assert report[failed_metric] > report["area_tolerance_m2"]


def test_legacy_run_without_frozen_range_is_explicitly_unverified(tmp_path):
    path = _write_polygons(tmp_path / "legacy.gpkg", [box(0, 0, 10, 10)])
    spec = _spec()
    spec.pop("requested_extent")

    report = _validate_exact_range_coverage(
        path,
        layer="semantic_polygons",
        spec=spec,
    )

    assert report["status"] == "skipped_legacy_no_range"
    assert report["hard_gate_applied"] is False
    assert report["reason"] == "frozen_exact_range_missing"
