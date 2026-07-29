"""Standalone boundary-regularization smoke test for the production environment."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import fiona
import numpy as np
import rasterio
import shapely
from shapely.geometry import shape


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference_scripts"))

from boundary_regularizer import regularize_coverage
from polygonize_mosaic import polygonize_mask


CLASS_ORDER = [12, 13, 21, 31, 32, 33, 43, 51, 52, 53, 54, 61, 62, 71]
CLASS_NAMES = {
    12: "水浇地", 13: "旱地", 21: "果园", 31: "有林地", 32: "灌木林地",
    33: "其他林地", 43: "其他草地", 51: "城镇建设用地", 52: "农村建设用地",
    53: "人为扰动用地", 54: "其他建设用地", 61: "农村道路", 62: "其他交通用地",
    71: "河湖库塘",
}


def main():
    root = Path(tempfile.mkdtemp(prefix="loess_boundary_regularizer_"))
    mask = np.zeros((64, 64), dtype=np.int16)
    for row in range(mask.shape[0]):
        mask[row, 16 + row // 2:] = 1
    confidence = np.full(mask.shape, 0.9, dtype=np.float32)
    transform = rasterio.transform.from_origin(100, 40, 0.001, 0.001)
    mask_path = root / "mask.tif"
    confidence_path = root / "confidence.tif"
    for path, values, dtype in (
        (mask_path, mask, "int16"),
        (confidence_path, confidence, "float32"),
    ):
        with rasterio.open(
            path, "w", driver="GTiff", width=64, height=64, count=1,
            dtype=dtype, crs="EPSG:4490", transform=transform,
        ) as destination:
            destination.write(values.astype(dtype), 1)
    class_map = root / "classes.json"
    class_map.write_text(json.dumps({
        "background_index": -1,
        "index_to_code": {str(index): code for index, code in enumerate(CLASS_ORDER)},
        "class_mapping": {str(code): CLASS_NAMES[code] for code in CLASS_ORDER},
    }, ensure_ascii=False), encoding="utf-8")
    raw = root / "semantic_polygons_raw.gpkg"
    formal = root / "semantic_polygons.gpkg"
    report_path = root / "boundary_regularization_report.json"
    polygonize_mask(
        mask_path, confidence_path, raw, "run_smoke", "fusion:smoke", "fusion",
        "smoke-v1", class_map, fusion_profile_id="smoke",
    )
    report = regularize_coverage(raw, mask_path, formal, report_path, {
        "enabled": True,
        "mode": "standard",
        "coverage_tolerance_px": 1.5,
        "angle_threshold_deg": 12.0,
        "max_deviation_px": 1.5,
        "minimum_chain_vertices": 4,
        "preserve_outer_boundary": True,
        "natural_smoothing": False,
    })
    with fiona.open(raw, layer="semantic_polygons") as source:
        raw_features = list(source)
    with fiona.open(formal, layer="semantic_polygons") as source:
        formal_features = list(source)
    raw_geometries = np.asarray([shape(item["geometry"]) for item in raw_features], dtype=object)
    formal_geometries = np.asarray([shape(item["geometry"]) for item in formal_features], dtype=object)
    assert report["status"] == "passed"
    assert len(raw_features) == len(formal_features) == 2
    assert [item["properties"]["object_id"] for item in raw_features] == [
        item["properties"]["object_id"] for item in formal_features
    ]
    assert shapely.coverage_is_valid(formal_geometries)
    assert shapely.area(
        shapely.symmetric_difference(
            shapely.union_all(raw_geometries), shapely.union_all(formal_geometries)
        )
    ) < 1e-12
    assert np.sum(shapely.get_num_coordinates(formal_geometries)) < np.sum(
        shapely.get_num_coordinates(raw_geometries)
    )
    print(f"boundary regularizer smoke: OK ({root})")


if __name__ == "__main__":
    main()
