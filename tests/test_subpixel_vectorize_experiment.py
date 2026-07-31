import numpy as np

import subpixel_vectorize_experiment as experiment


def _diagonal_probabilities(size=64):
    rows, columns = np.mgrid[:size, :size]
    margin = (
        columns - (20.0 + 0.45 * rows + 2.0 * np.sin(rows / 6.0))
    ) / 1.3
    first = 1.0 / (1.0 + np.exp(margin))
    probabilities = np.full((14, size, size), 1e-5, dtype=np.float32)
    probabilities[0] = first
    probabilities[1] = 1.0 - first
    probabilities /= probabilities.sum(axis=0, keepdims=True)
    return probabilities


def test_subpixel_coverage_closes_shared_edges_and_reduces_staircase():
    probabilities = _diagonal_probabilities()
    labels = probabilities.argmax(axis=0).astype(np.int16)
    raw = experiment._raw_coverage(labels)
    formal = experiment._simplify_records(
        experiment.subpixel_coverage(probabilities), 1.0
    )

    checks = experiment._coverage_checks(formal, 64, 64)
    assert checks["geometry_valid"]
    assert checks["coverage_valid"]
    assert checks["gap_area_px2"] == 0.0
    assert checks["overlap_area_px2"] <= 1e-8
    assert checks["covered_area_px2"] == 4096.0

    raw_metrics = experiment._staircase_metrics(raw)
    formal_metrics = experiment._staircase_metrics(formal)
    assert formal_metrics["staircase_turn_count"] < raw_metrics["staircase_turn_count"]
    assert formal_metrics["short_segment_ratio"] < raw_metrics["short_segment_ratio"]

    raw_area = {
        index: float(np.count_nonzero(labels == index))
        for index in (0, 1)
    }
    formal_area = {0: 0.0, 1: 0.0}
    for class_index, geometry in formal:
        formal_area[class_index] += geometry.area
    for index in (0, 1):
        assert abs(formal_area[index] - raw_area[index]) <= 64.0
