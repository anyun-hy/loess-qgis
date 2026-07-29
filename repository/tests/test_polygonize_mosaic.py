import json

import numpy as np
import pytest
import rasterio
import shapely


fiona = pytest.importorskip("fiona")

from polygonize_mosaic import polygonize_mask
from shapely.geometry import shape


def _write_raster(path, array, dtype):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=array.shape[1],
        height=array.shape[0],
        count=1,
        dtype=dtype,
        crs="EPSG:4490",
        transform=rasterio.transform.from_origin(100, 40, 0.001, 0.001),
    ) as dst:
        dst.write(array.astype(dtype), 1)


def test_polygonize_writes_model_stream_identity_and_confidence_stats(tmp_path, capfd):
    mask = np.zeros((4, 4), dtype=np.int16)
    mask[:, 2:] = 1
    confidence = np.full((4, 4), 0.8, dtype=np.float32)
    mask_path = tmp_path / "mask.tif"
    confidence_path = tmp_path / "confidence.tif"
    output = tmp_path / "semantic.gpkg"
    class_map = tmp_path / "classes.json"
    _write_raster(mask_path, mask, "int16")
    _write_raster(confidence_path, confidence, "float32")
    class_map.write_text(json.dumps({
        "background_index": -1,
        "index_to_code": {str(index): code for index, code in enumerate([12, 13, 21, 31, 32, 33, 43, 51, 52, 53, 54, 61, 62, 71])},
        "class_mapping": {
            "12": "水浇地", "13": "旱地", "21": "果园", "31": "有林地",
            "32": "灌木林地", "33": "其他林地", "43": "其他草地",
            "51": "城镇建设用地", "52": "农村建设用地", "53": "人为扰动用地",
            "54": "其他建设用地", "61": "农村道路", "62": "其他交通用地", "71": "河湖库塘",
        },
    }, ensure_ascii=False), encoding="utf-8")

    count = polygonize_mask(
        mask_path,
        confidence_path,
        output,
        "run_a",
        "model:model_a",
        "model",
        "fixture-v1",
        class_map,
        model_id="model_a",
    )
    captured = capfd.readouterr()

    assert count == 2
    assert "'Memory' driver is deprecated" not in captured.err
    with fiona.open(output, layer="semantic_polygons") as src:
        records = list(src)
    assert {record["properties"]["class_code"] for record in records} == {12, 13}
    assert all(record["properties"]["result_stream_id"] == "model:model_a" for record in records)
    assert all(record["properties"]["source"] == "semantic_model" for record in records)
    assert all(record["properties"]["confidence_mean"] == pytest.approx(0.8) for record in records)
    assert len({record["properties"]["object_id"] for record in records}) == 2


def test_polygonize_object_ids_are_unique_across_runs(tmp_path):
    mask = np.zeros((3, 3), dtype=np.int16)
    mask[:, 1:] = 1
    confidence = np.full((3, 3), 0.75, dtype=np.float32)
    mask_path = tmp_path / "mask.tif"
    confidence_path = tmp_path / "confidence.tif"
    class_map = tmp_path / "classes.json"
    _write_raster(mask_path, mask, "int16")
    _write_raster(confidence_path, confidence, "float32")
    class_map.write_text(json.dumps({
        "background_index": -1,
        "index_to_code": {
            str(index): code
            for index, code in enumerate([12, 13, 21, 31, 32, 33, 43, 51, 52, 53, 54, 61, 62, 71])
        },
        "class_mapping": {
            "12": "水浇地", "13": "旱地", "21": "果园", "31": "有林地",
            "32": "灌木林地", "33": "其他林地", "43": "其他草地",
            "51": "城镇建设用地", "52": "农村建设用地", "53": "人为扰动用地",
            "54": "其他建设用地", "61": "农村道路", "62": "其他交通用地", "71": "河湖库塘",
        },
    }, ensure_ascii=False), encoding="utf-8")

    outputs = []
    for run_id in ("run_a", "run_b"):
        output = tmp_path / f"{run_id}.gpkg"
        polygonize_mask(
            mask_path,
            confidence_path,
            output,
            run_id,
            "model:model_a",
            "model",
            "fixture-v1",
            class_map,
            model_id="model_a",
        )
        with fiona.open(output, layer="semantic_polygons") as src:
            outputs.append({record["properties"]["object_id"] for record in src})

    assert outputs[0]
    assert outputs[1]
    assert outputs[0].isdisjoint(outputs[1])


def test_polygonize_preserves_raw_shared_staircase_for_separate_regularizer(tmp_path):
    mask = np.zeros((32, 32), dtype=np.int16)
    for row in range(mask.shape[0]):
        mask[row, 8 + row // 2:] = 1
    confidence = np.full(mask.shape, 0.9, dtype=np.float32)
    mask_path = tmp_path / "mask.tif"
    confidence_path = tmp_path / "confidence.tif"
    class_map = tmp_path / "classes.json"
    _write_raster(mask_path, mask, "int16")
    _write_raster(confidence_path, confidence, "float32")
    class_map.write_text(json.dumps({
        "background_index": -1,
        "index_to_code": {
            str(index): code
            for index, code in enumerate([12, 13, 21, 31, 32, 33, 43, 51, 52, 53, 54, 61, 62, 71])
        },
        "class_mapping": {
            "12": "水浇地", "13": "旱地", "21": "果园", "31": "有林地",
            "32": "灌木林地", "33": "其他林地", "43": "其他草地",
            "51": "城镇建设用地", "52": "农村建设用地", "53": "人为扰动用地",
            "54": "其他建设用地", "61": "农村道路", "62": "其他交通用地", "71": "河湖库塘",
        },
    }, ensure_ascii=False), encoding="utf-8")

    output = tmp_path / "semantic_polygons_raw.gpkg"
    polygonize_mask(
        mask_path,
        confidence_path,
        output,
        "run_a",
        "model:model_a",
        "model",
        "fixture-v1",
        class_map,
        model_id="model_a",
    )
    with fiona.open(output, layer="semantic_polygons") as source:
        raw = np.asarray([shape(feature["geometry"]) for feature in source], dtype=object)

    assert shapely.coverage_is_valid(raw)
    assert np.all(shapely.is_valid(raw))
    assert np.sum(shapely.get_num_coordinates(raw)) > 50
