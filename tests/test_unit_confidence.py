from copy import deepcopy
import time

import numpy as np
import pytest
from affine import Affine
from rasterio.features import geometry_mask
from shapely.geometry import MultiPolygon, Polygon, box, mapping

import boundary_fitting.unit_runtime as unit_runtime
from boundary_fitting.unit_runtime import (
    BATCHED_CONFIDENCE_WORK_THRESHOLD,
    _attach_confidence,
    _fit_or_subdivide,
)
from polyline_smoother import SmoothingConfig
from vector_data_plane import unit_boundary_signatures


# Stable row-major grouping retains the former NumPy reduction order, so the
# compatibility gate is exact rather than merely numerically close.
CONFIDENCE_ATOL = 0.0
CLASS_CODES = [12, 13, 21, 31, 32, 33, 43, 51, 52, 53, 54, 61, 62, 71]


def _reference_attach_confidence(records, probabilities, valid_mask, unit):
    """The pre-optimization implementation, retained as an independent oracle."""

    confidence = probabilities.max(axis=0)
    window = unit["pixel_window"]
    transform = Affine.translation(int(window["x0"]), int(window["y0"]))
    for record in records:
        selected = geometry_mask(
            [mapping(record["geometry"])],
            out_shape=confidence.shape,
            transform=transform,
            invert=True,
        )
        values = confidence[selected & valid_mask]
        record["confidence_mean"] = float(values.mean()) if values.size else 0.0
        record["confidence_std"] = float(values.std()) if values.size else 0.0


def _probabilities(height, width):
    rows, columns = np.indices((height, width), dtype=np.float32)
    values = np.empty((14, height, width), dtype=np.float32)
    for class_index in range(14):
        values[class_index] = (
            0.01 * (class_index + 1)
            + 0.0017 * rows
            + 0.0009 * columns
        )
    values /= values.sum(axis=0, keepdims=True)
    return values


def _assert_confidence_matches_reference(records, probabilities, valid_mask, unit):
    expected = deepcopy(records)
    actual = deepcopy(records)
    _reference_attach_confidence(expected, probabilities, valid_mask, unit)
    _attach_confidence(actual, probabilities, valid_mask, unit)

    assert [record["polygon_id"] for record in actual] == [
        record["polygon_id"] for record in expected
    ]
    assert [record["class_code"] for record in actual] == [
        record["class_code"] for record in expected
    ]
    assert [record["geometry"].wkb for record in actual] == [
        record["geometry"].wkb for record in expected
    ]
    for expected_record, actual_record in zip(expected, actual):
        assert actual_record["confidence_mean"] == pytest.approx(
            expected_record["confidence_mean"],
            rel=0.0,
            abs=CONFIDENCE_ATOL,
        )
        assert actual_record["confidence_std"] == pytest.approx(
            expected_record["confidence_std"],
            rel=0.0,
            abs=CONFIDENCE_ATOL,
        )
    return expected, actual


def test_attach_confidence_preserves_pixel_center_semantics_on_shared_boundary():
    unit = {
        "unit_id": "translated",
        "pixel_window": {"x0": 100, "y0": 200, "x1": 108, "y1": 206},
    }
    records = [
        {"polygon_id": "left", "class_code": 12, "geometry": box(100, 200, 104, 206)},
        {"polygon_id": "right", "class_code": 13, "geometry": box(104, 200, 108, 206)},
    ]
    valid_mask = np.ones((6, 8), dtype=bool)
    valid_mask[1, 1] = False
    valid_mask[4, 6] = False

    _assert_confidence_matches_reference(
        records,
        _probabilities(6, 8),
        valid_mask,
        unit,
    )


def test_attach_confidence_preserves_holes_and_multipolygons():
    unit = {
        "unit_id": "complex",
        "pixel_window": {"x0": 20, "y0": 40, "x1": 30, "y1": 50},
    }
    outer = Polygon(
        [(20, 40), (30, 40), (30, 50), (20, 50), (20, 40)],
        [[(23, 43), (27, 43), (27, 47), (23, 47), (23, 43)]],
    )
    island = box(23, 43, 27, 47)
    multipart = MultiPolygon(
        [
            box(20, 40, 22, 42),
            box(28, 48, 30, 50),
        ]
    )
    # The multipart record deliberately overlaps the outer polygon. The
    # optimized implementation must detect this and retain independent masks.
    records = [
        {"polygon_id": "outer", "class_code": 12, "geometry": outer},
        {"polygon_id": "island", "class_code": 13, "geometry": island},
        {"polygon_id": "multipart", "class_code": 21, "geometry": multipart},
    ]
    valid_mask = np.ones((10, 10), dtype=bool)
    valid_mask[0, 0] = False
    valid_mask[9, 9] = False

    _assert_confidence_matches_reference(
        records,
        _probabilities(10, 10),
        valid_mask,
        unit,
    )


def test_attach_confidence_empty_selection_is_exact_zero():
    unit = {
        "unit_id": "empty",
        "pixel_window": {"x0": 0, "y0": 0, "x1": 4, "y1": 4},
    }
    records = [
        {
            "polygon_id": "subpixel",
            "class_code": 12,
            "geometry": box(0.05, 0.05, 0.45, 0.45),
        },
        {
            "polygon_id": "masked",
            "class_code": 13,
            "geometry": box(1, 1, 3, 3),
        },
    ]
    valid_mask = np.ones((4, 4), dtype=bool)
    valid_mask[1:3, 1:3] = False

    expected, actual = _assert_confidence_matches_reference(
        records,
        _probabilities(4, 4),
        valid_mask,
        unit,
    )
    assert [(item["confidence_mean"], item["confidence_std"]) for item in expected] == [
        (0.0, 0.0),
        (0.0, 0.0),
    ]
    assert [(item["confidence_mean"], item["confidence_std"]) for item in actual] == [
        (0.0, 0.0),
        (0.0, 0.0),
    ]


def test_raw_and_formal_confidence_do_not_change_geometry_identity_or_coverage():
    probabilities = np.zeros((14, 24, 24), dtype=np.float32)
    for row in range(24):
        threshold = 9 + row // 3
        probabilities[0, row, :threshold] = 0.82
        probabilities[1, row, :threshold] = 0.18
        probabilities[0, row, threshold:] = 0.23
        probabilities[1, row, threshold:] = 0.77
    labels = probabilities.argmax(axis=0).astype(np.int16)
    valid_mask = np.ones((24, 24), dtype=bool)
    valid_mask[0, 0] = False
    unit = {
        "unit_id": "core_00000_00000",
        "pixel_window": {"x0": 0, "y0": 0, "x1": 24, "y1": 24},
    }
    raw, formal, _report = _fit_or_subdivide(
        labels,
        unit,
        CLASS_CODES,
        valid_mask=valid_mask,
        smoothing_config=SmoothingConfig(
            smoothing_factor=1.0,
            curve_sampling_spacing=0.5,
            max_deviation=None,
            min_point_count=4,
        ),
        max_features=1000,
        max_segments=10000,
        min_core_px=4,
    )

    for records in (raw, formal):
        identity_before = [
            (item["polygon_id"], item["class_code"], item["geometry"].wkb)
            for item in records
        ]
        area_before = sum(item["geometry"].area for item in records)
        signatures_before = unit_boundary_signatures(
            records,
            stream_id="raw" if records is raw else "formal",
            unit_id=unit["unit_id"],
            pixel_window=unit["pixel_window"],
        )
        expected = deepcopy(records)
        _reference_attach_confidence(expected, probabilities, valid_mask, unit)

        _attach_confidence(records, probabilities, valid_mask, unit)

        assert [
            (item["polygon_id"], item["class_code"], item["geometry"].wkb)
            for item in records
        ] == identity_before
        assert sum(item["geometry"].area for item in records) == area_before
        assert unit_boundary_signatures(
            records,
            stream_id="raw" if records is raw else "formal",
            unit_id=unit["unit_id"],
            pixel_window=unit["pixel_window"],
        ) == signatures_before
        for expected_record, actual_record in zip(expected, records):
            assert actual_record["confidence_mean"] == pytest.approx(
                expected_record["confidence_mean"], rel=0.0, abs=CONFIDENCE_ATOL
            )
            assert actual_record["confidence_std"] == pytest.approx(
                expected_record["confidence_std"], rel=0.0, abs=CONFIDENCE_ATOL
            )


def test_attach_confidence_small_unit_uses_individual_path(monkeypatch):
    size = 512
    records = [
        {
            "polygon_id": f"strip_{index}",
            "class_code": 12,
            "geometry": box(index * 4, 0, index * 4 + 4, size),
        }
        for index in range(124)
    ]
    assert len(records) * size * size < BATCHED_CONFIDENCE_WORK_THRESHOLD
    probabilities = np.zeros((14, size, size), dtype=np.float32)
    probabilities[0] = 1.0

    def unexpected_rasterize(*_args, **_kwargs):
        raise AssertionError("small unit unexpectedly selected the batched path")

    monkeypatch.setattr(unit_runtime, "rasterize", unexpected_rasterize)
    _attach_confidence(
        records,
        probabilities,
        np.ones((size, size), dtype=bool),
        {
            "unit_id": "small",
            "pixel_window": {"x0": 0, "y0": 0, "x1": size, "y1": size},
        },
    )

    assert all(record["confidence_mean"] == 1.0 for record in records)
    assert all(record["confidence_std"] == 0.0 for record in records)


def test_attach_confidence_large_unit_uses_batched_path(monkeypatch):
    size = 501
    cell = 25
    records = [
        {
            "polygon_id": f"cell_{row}_{column}",
            "class_code": 12,
            "geometry": box(column, row, column + cell, row + cell),
        }
        for row in range(0, 500, cell)
        for column in range(0, 500, cell)
    ]
    assert len(records) * size * size >= BATCHED_CONFIDENCE_WORK_THRESHOLD
    probabilities = np.zeros((14, size, size), dtype=np.float32)
    probabilities[0] = 1.0

    def unexpected_individual(*_args, **_kwargs):
        raise AssertionError("large unit unexpectedly selected the individual path")

    monkeypatch.setattr(
        unit_runtime,
        "_attach_confidence_individually",
        unexpected_individual,
    )
    _attach_confidence(
        records,
        probabilities,
        np.ones((size, size), dtype=bool),
        {
            "unit_id": "large",
            "pixel_window": {"x0": 0, "y0": 0, "x1": size, "y1": size},
        },
    )

    assert all(record["confidence_mean"] == 1.0 for record in records)
    assert all(record["confidence_std"] == 0.0 for record in records)


def test_forced_batched_path_preserves_holes_multipolygons_and_valid_edges(
    monkeypatch,
):
    monkeypatch.setattr(unit_runtime, "BATCHED_CONFIDENCE_WORK_THRESHOLD", 0)
    unit = {
        "unit_id": "forced_complex",
        "pixel_window": {"x0": 0, "y0": 0, "x1": 16, "y1": 12},
    }
    outer = Polygon(
        [(0, 0), (10, 0), (10, 12), (0, 12), (0, 0)],
        [[(2, 2), (5, 2), (5, 6), (2, 6), (2, 2)]],
    )
    records = [
        {"polygon_id": "outer", "class_code": 12, "geometry": outer},
        {"polygon_id": "island", "class_code": 13, "geometry": box(2, 2, 5, 6)},
        {
            "polygon_id": "multipart",
            "class_code": 21,
            "geometry": MultiPolygon([box(10, 0, 13, 4), box(13, 8, 16, 12)]),
        },
    ]
    valid_mask = np.ones((12, 16), dtype=bool)
    valid_mask[0, :] = False
    valid_mask[:, -1] = False

    _assert_confidence_matches_reference(
        records,
        _probabilities(12, 16),
        valid_mask,
        unit,
    )


def test_forced_batched_path_detects_pixel_center_overlap_and_falls_back(
    monkeypatch,
):
    monkeypatch.setattr(unit_runtime, "BATCHED_CONFIDENCE_WORK_THRESHOLD", 0)
    original = unit_runtime._attach_confidence_individually
    calls = []

    def tracked(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(unit_runtime, "_attach_confidence_individually", tracked)
    unit = {
        "unit_id": "forced_overlap",
        "pixel_window": {"x0": 0, "y0": 0, "x1": 8, "y1": 8},
    }
    records = [
        {"polygon_id": "base", "class_code": 12, "geometry": box(0, 0, 8, 8)},
        {"polygon_id": "overlap", "class_code": 13, "geometry": box(2, 2, 6, 6)},
    ]

    _assert_confidence_matches_reference(
        records,
        _probabilities(8, 8),
        np.ones((8, 8), dtype=bool),
        unit,
    )
    assert calls == [True]


def test_attach_confidence_performance_microbenchmark():
    """Report old/new timing without turning machine speed into a CI gate."""

    size = 512
    cell = 16
    unit = {
        "unit_id": "benchmark",
        "pixel_window": {"x0": 0, "y0": 0, "x1": size, "y1": size},
    }
    records = [
        {
            "polygon_id": f"cell_{row}_{column}",
            "class_code": 12 + (row + column) % 2,
            "geometry": box(column, row, column + cell, row + cell),
        }
        for row in range(0, size, cell)
        for column in range(0, size, cell)
    ]
    probabilities = _probabilities(size, size)
    valid_mask = np.ones((size, size), dtype=bool)
    expected = deepcopy(records)
    actual = deepcopy(records)

    reference_started = time.perf_counter()
    _reference_attach_confidence(expected, probabilities, valid_mask, unit)
    reference_sec = time.perf_counter() - reference_started
    optimized_started = time.perf_counter()
    _attach_confidence(actual, probabilities, valid_mask, unit)
    optimized_sec = time.perf_counter() - optimized_started

    for expected_record, actual_record in zip(expected, actual):
        assert actual_record["confidence_mean"] == pytest.approx(
            expected_record["confidence_mean"], rel=0.0, abs=CONFIDENCE_ATOL
        )
        assert actual_record["confidence_std"] == pytest.approx(
            expected_record["confidence_std"], rel=0.0, abs=CONFIDENCE_ATOL
        )
    speedup = reference_sec / optimized_sec if optimized_sec else float("inf")
    print(
        "confidence microbenchmark: "
        f"reference={reference_sec:.6f}s optimized={optimized_sec:.6f}s "
        f"speedup={speedup:.2f}x"
    )
