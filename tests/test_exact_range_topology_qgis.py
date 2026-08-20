"""QGIS topology targets must use the frozen exact vector boundary."""

from __future__ import annotations

from pathlib import Path

import fiona
import pytest
from fiona.crs import CRS
from shapely.geometry import box, mapping

pytest.importorskip("qgis.core", exc_type=ImportError)

from qgis.core import QgsApplication, QgsCoordinateReferenceSystem  # noqa: E402

from labeling_tool.core.run_spec import sha256_file  # noqa: E402
from labeling_tool.core.topology_validator import _selected_tile_target  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qgis_application():
    application = QgsApplication.instance() or QgsApplication([], False)
    if QgsApplication.instance() is application:
        application.initQgis()
    yield application


def test_vector_topology_target_is_the_exact_snapshot_not_selected_tile_union(tmp_path: Path):
    snapshot = tmp_path / "range_snapshot.gpkg"
    with fiona.open(
        snapshot,
        "w",
        driver="GPKG",
        layer="range_mask",
        schema={"geometry": "Polygon", "properties": {}},
        crs=CRS.from_epsg(3857),
    ) as destination:
        destination.write({"geometry": mapping(box(10, 10, 30, 30)), "properties": {}})
    spec = {
        "range_selection": {
            "mode": "vector_tile_intersection",
            "vector_source": str(snapshot),
            "vector_sha256": sha256_file(snapshot),
        },
        "raster": {"crs": "EPSG:3857"},
        "requested_extent": {"xmin": 0, "ymin": 0, "xmax": 20, "ymax": 20},
    }

    target = _selected_tile_target(spec, QgsCoordinateReferenceSystem("EPSG:3857"))

    bounds = target.boundingBox()
    assert (
        bounds.xMinimum(),
        bounds.yMinimum(),
        bounds.xMaximum(),
        bounds.yMaximum(),
    ) == pytest.approx((10, 10, 20, 20))
