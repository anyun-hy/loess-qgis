from pathlib import Path

import fiona
from fiona.crs import CRS
from shapely.geometry import MultiPolygon, Polygon, mapping, shape

from difference_runtime import apply_accepted_difference


def _write(path: Path, layer: str, geometries, *, accepted=False):
    properties = {"class_code": "int"} if accepted else {
        "object_id": "str:64", "part_id": "str:96", "class_code": "int"
    }
    with fiona.open(
        path,
        "w",
        driver="GPKG",
        layer=layer,
        schema={"geometry": "MultiPolygon", "properties": properties},
        crs=CRS.from_epsg(4490),
    ) as destination:
        for index, geometry in enumerate(geometries):
            values = {"class_code": 13}
            if not accepted:
                values.update({"object_id": "object-a", "part_id": f"part-{index}"})
            destination.write(
                {
                    "geometry": mapping(MultiPolygon([geometry])),
                    "properties": values,
                }
            )


def test_difference_preserves_object_id_and_reassigns_split_part_ids(tmp_path):
    source = tmp_path / "source.gpkg"
    accepted = tmp_path / "accepted.gpkg"
    output = tmp_path / "candidates.gpkg"
    _write(source, "semantic_polygons", [Polygon([(0, 0), (4, 0), (4, 2), (0, 2)])])
    _write(
        accepted,
        "accepted_labels",
        [Polygon([(1.5, -1), (2.5, -1), (2.5, 3), (1.5, 3)])],
        accepted=True,
    )
    report = apply_accepted_difference(source, accepted, output)
    assert report["status"] == "passed"
    assert report["source_feature_count"] == 1
    assert report["output_feature_count"] == 2
    with fiona.open(output, layer="semantic_candidates") as layer:
        features = list(layer)
    assert {feature["properties"]["object_id"] for feature in features} == {"object-a"}
    assert {feature["properties"]["part_id"] for feature in features} == {
        "part-0:d000", "part-0:d001"
    }
    assert sum(shape(feature["geometry"]).area for feature in features) == 6.0


def test_difference_skips_cleanly_without_accepted_layer(tmp_path):
    source = tmp_path / "source.gpkg"
    _write(source, "semantic_polygons", [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])])
    report = apply_accepted_difference(source, tmp_path / "missing.gpkg", tmp_path / "out.gpkg")
    assert report["status"] == "skipped"
