import json

import fiona
import numpy as np
import rasterio
import shapely

import subpixel_vectorizer
from polygonize_mosaic import polygonize_mask
from subpixel_vectorizer import METHOD, vectorize_probability_mosaic


CLASS_CODES = [12, 13, 21, 31, 32, 33, 43, 51, 52, 53, 54, 61, 62, 71]


def _probabilities(size=64):
    rows, columns = np.mgrid[:size, :size]
    margin = (columns - (18.0 + 0.5 * rows + 1.5 * np.sin(rows / 5.0))) / 1.4
    first = 1.0 / (1.0 + np.exp(margin))
    values = np.full((14, size, size), 1e-5, dtype=np.float32)
    values[0] = first
    values[1] = 1.0 - first
    values /= values.sum(axis=0, keepdims=True)
    return values


def _write_inputs(tmp_path):
    probabilities = _probabilities()
    labels = probabilities.argmax(axis=0).astype(np.int16)
    confidence = probabilities.max(axis=0).astype(np.float32)
    transform = rasterio.transform.from_origin(110.0, 38.0, 0.0001, 0.0001)
    common = {
        "driver": "GTiff",
        "width": labels.shape[1],
        "height": labels.shape[0],
        "crs": "EPSG:4490",
        "transform": transform,
    }
    mask_path = tmp_path / "mask.tif"
    confidence_path = tmp_path / "confidence.tif"
    probability_path = tmp_path / "probability.tif"
    with rasterio.open(mask_path, "w", count=1, dtype="int16", nodata=-1, **common) as dst:
        dst.write(labels, 1)
    with rasterio.open(confidence_path, "w", count=1, dtype="float32", **common) as dst:
        dst.write(confidence, 1)
    quantized = np.rint(probabilities * 65535.0).astype(np.uint16)
    with rasterio.open(probability_path, "w", count=14, dtype="uint16", **common) as dst:
        dst.scales = tuple(1.0 / 65535.0 for _ in range(14))
        dst.update_tags(probability_encoding="uint16_scale_1_over_65535")
        dst.write(quantized)
    class_map_path = tmp_path / "classes.json"
    class_map_path.write_text(
        json.dumps({
            "class_mapping": {str(code): f"class-{code}" for code in CLASS_CODES},
            "index_to_code": {str(index): code for index, code in enumerate(CLASS_CODES)},
            "background_index": -1,
        }),
        encoding="utf-8",
    )
    return mask_path, confidence_path, probability_path, class_map_path


def test_formal_subpixel_vectorization_writes_valid_shared_coverage(tmp_path, capsys):
    mask, confidence, probability, class_map = _write_inputs(tmp_path)
    raw = tmp_path / "raw.gpkg"
    formal = tmp_path / "formal.gpkg"
    report_path = tmp_path / "report.json"
    polygonize_mask(
        mask,
        confidence,
        raw,
        "run-test",
        "fusion:test",
        "fusion",
        "test-v1",
        class_map,
        fusion_profile_id="test",
    )

    report = vectorize_probability_mosaic(
        probability,
        mask,
        confidence,
        raw,
        formal,
        report_path,
        class_map,
        run_id="run-test",
        stream_id="fusion:test",
        result_kind="fusion",
        fusion_profile_id="test",
        model_version="test-v1",
        stripe_rows=9,
    )

    assert report["status"] == "passed"
    assert report["method"] == METHOD
    assert report["validation"]["passed"]
    assert report["validation"]["checks"]["interpolation_movement_within_tolerance"]
    assert report["validation"]["checks"]["simplification_movement_within_tolerance"]
    assert report["validation"]["effective_coverage_tolerance_px"] <= 1.0
    assert report["validation"]["formal"]["overlap_area_px2"] <= 1e-6
    assert report["validation"]["union_symmetric_difference_px2"] <= 1e-6
    assert (
        report["metrics"]["formal"]["staircase_turn_density_per_100px"]
        < report["metrics"]["raw"]["staircase_turn_density_per_100px"]
    )
    with fiona.open(formal, layer="semantic_polygons") as source:
        features = list(source)
    geometries = np.asarray(
        [shapely.geometry.shape(feature["geometry"]) for feature in features],
        dtype=object,
    )
    assert features
    assert np.all(shapely.is_valid(geometries))
    assert len({feature["properties"]["object_id"] for feature in features}) == len(features)
    assert {feature["properties"]["class_code"] for feature in features} == {12, 13}
    assert all(feature["properties"]["regularization_method"] == METHOD for feature in features)
    events = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("{")
    ]
    progress = [event for event in events if event["event"] == "subpixel_linework_progress"]
    assert progress
    assert progress[-1]["stream_id"] == "fusion:test"
    assert progress[-1]["current"] == progress[-1]["total"] == 63
    finished = [
        event for event in events if event["event"] == "subpixel_vectorization_finished"
    ]
    assert finished[-1]["current"] == finished[-1]["total"] == 63


def test_coverage_simplify_lowers_tolerance_instead_of_relaxing_gate(monkeypatch):
    marker = shapely.geometry.box(0, 0, 1, 1)

    def fake_simplify(_records, tolerance):
        return [(float(tolerance), marker, marker)]

    def fake_displacement(simplified):
        value = float(simplified[0][0]) * 2.0
        return value, {"0": value}

    monkeypatch.setattr(subpixel_vectorizer, "_simplify_coverage", fake_simplify)
    monkeypatch.setattr(
        subpixel_vectorizer,
        "_simplification_displacement",
        fake_displacement,
    )
    _simplified, effective, maximum, _by_class, attempts = (
        subpixel_vectorizer._simplify_coverage_bounded(
            [(0, marker)],
            target_tolerance=1.0,
            max_deviation=1.5,
        )
    )

    assert effective == 0.75
    assert maximum == 1.5
    assert [item["coverage_tolerance_px"] for item in attempts] == [1.0, 0.75]


def test_formal_contract_rejects_nonstandard_tolerance(tmp_path):
    mask, confidence, probability, class_map = _write_inputs(tmp_path)
    raw = tmp_path / "raw.gpkg"
    polygonize_mask(
        mask,
        confidence,
        raw,
        "run-test",
        "model:test",
        "model",
        "test-v1",
        class_map,
        model_id="test",
    )
    try:
        vectorize_probability_mosaic(
            probability,
            mask,
            confidence,
            raw,
            tmp_path / "formal.gpkg",
            tmp_path / "report.json",
            class_map,
            run_id="run-test",
            stream_id="model:test",
            result_kind="model",
            model_id="test",
            coverage_tolerance_px=1.5,
        )
    except Exception as exc:
        assert "coverage_tolerance_px must equal 1.0" in str(exc)
    else:
        raise AssertionError("nonstandard formal tolerance was accepted")
