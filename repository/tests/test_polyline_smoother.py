import math
from pathlib import Path
import time

import fiona
import numpy as np
from shapely.geometry import LineString, MultiLineString, mapping, shape

from polyline_smoother import (
    SmoothingConfig,
    adaptive_sample_curve,
    smooth_polyline,
    smooth_vector_file,
)


def _staircase(length=80):
    points = [(0.0, 0.0)]
    for index in range(length):
        if index % 2 == 0:
            points.append((points[-1][0] + 1.0, points[-1][1]))
        else:
            points.append((points[-1][0], points[-1][1] + 1.0))
    return np.asarray(points)


def _direction_change_energy(points):
    vectors = np.diff(points, axis=0)
    angles = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
    return float(np.mean(np.abs(np.diff(angles))))


def test_cubic_bspline_visibly_reduces_staircase_direction_changes():
    points = _staircase()
    result = smooth_polyline(points)
    assert result.status == "smoothed"
    assert result.strength == 1.0
    assert np.array_equal(result.points[0], points[0])
    assert np.array_equal(result.points[-1], points[-1])
    assert result.dense_point_count > result.output_point_count
    assert result.max_chord_error <= 0.25 + 1e-12
    assert result.max_segment_arc_length <= 8.0 + 1e-12
    assert _direction_change_energy(result.points) < _direction_change_energy(points) * 0.15


def test_adaptive_curve_sampling_enforces_error_and_arc_bounds():
    x = np.linspace(0.0, 1000.0, 20_001)
    dense = np.column_stack((x, 4.0 * np.sin(x / 35.0)))
    sparse, chord_error, arc_length = adaptive_sample_curve(
        dense,
        max_chord_error=0.25,
        max_segment_arc_length=8.0,
    )

    assert np.array_equal(sparse[0], dense[0])
    assert np.array_equal(sparse[-1], dense[-1])
    assert chord_error <= 0.25 + 1e-12
    assert arc_length <= 8.0 + 1e-12
    assert len(sparse) < len(dense) * 0.02


def test_adaptive_closed_curve_remains_closed_with_structural_points():
    angles = np.linspace(0.0, 2.0 * math.pi, 4001)
    dense = np.column_stack((20.0 * np.cos(angles), 20.0 * np.sin(angles)))
    dense[-1] = dense[0]
    sparse, chord_error, arc_length = adaptive_sample_curve(
        dense,
        max_chord_error=0.25,
        max_segment_arc_length=8.0,
    )

    assert len(sparse) >= 4
    assert np.array_equal(sparse[0], sparse[-1])
    assert chord_error <= 0.25 + 1e-12
    assert arc_length <= 8.0 + 1e-12


def test_sparse_four_point_staircase_does_not_overshoot():
    points = np.asarray(
        [(1132.0, 2079.0), (1132.0, 2056.0), (1131.0, 2056.0), (1131.0, 2055.0)]
    )
    result = smooth_polyline(
        points,
        SmoothingConfig(min_point_count=4, curve_sampling_spacing=0.5),
    )
    assert result.status == "smoothed"
    assert result.max_deviation < 1.0
    assert np.array_equal(result.points[0], points[0])
    assert np.array_equal(result.points[-1], points[-1])


def test_closed_polyline_remains_closed_and_is_resampled():
    angles = np.linspace(0, 2 * math.pi, 65)
    radii = 20 + np.where(np.arange(len(angles)) % 2 == 0, 0.8, -0.8)
    points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    points[-1] = points[0]
    result = smooth_polyline(points)
    assert result.status == "smoothed"
    assert result.closed is True
    assert np.allclose(result.points[0], result.points[-1])


def test_deviation_limit_reduces_strength_instead_of_moving_outline_too_far():
    x = np.linspace(0, 100, 201)
    points = np.column_stack((x, 5 * np.sin(x / 8)))
    result = smooth_polyline(
        points,
        SmoothingConfig(smoothing_factor=5.0, max_deviation=0.4),
    )
    assert result.status == "smoothed"
    assert 0.05 <= result.strength < 1.0
    assert result.max_deviation <= 0.4


def test_large_polyline_runs_in_linear_practical_time():
    x = np.linspace(0, 20_000, 20_001)
    points = np.column_stack((x, np.sin(x / 20) + (np.arange(len(x)) % 2) * 0.3))
    started = time.monotonic()
    result = smooth_polyline(
        points,
        SmoothingConfig(curve_sampling_spacing=1.0),
    )
    assert result.status == "smoothed"
    assert time.monotonic() - started < 5.0


def test_vector_file_cli_core_preserves_attributes_and_crs(tmp_path: Path):
    source = tmp_path / "input.geojson"
    output = tmp_path / "output.gpkg"
    schema = {"geometry": "LineString", "properties": {"name": "str"}}
    with fiona.open(source, "w", driver="GeoJSON", schema=schema, crs="EPSG:4490") as sink:
        sink.write(
            {
                "geometry": mapping(LineString(_staircase(20))),
                "properties": {"name": "test"},
            }
        )
    report = smooth_vector_file(source, output)
    assert report["algorithm"] == "divider_cubic_bspline_adaptive_v2"
    assert report["smoothed_count"] == 1
    with fiona.open(output, layer="smoothed_lines") as result:
        feature = next(iter(result))
        assert feature["properties"]["name"] == "test"
        assert shape(feature["geometry"]).geom_type == "LineString"
        assert result.crs.to_epsg() == 4490


def test_vector_file_supports_multiline_and_geojson_output(tmp_path: Path):
    source = tmp_path / "multi.gpkg"
    output = tmp_path / "multi_smoothed.geojson"
    schema = {"geometry": "MultiLineString", "properties": {"value": "int"}}
    geometry = MultiLineString([LineString(_staircase(20)), LineString(_staircase(24) + 40)])
    with fiona.open(source, "w", driver="GPKG", layer="lines", schema=schema, crs="EPSG:4490") as sink:
        sink.write({"geometry": mapping(geometry), "properties": {"value": 7}})
    report = smooth_vector_file(source, output, input_layer="lines")
    assert report["line_count"] == 2
    assert report["smoothed_count"] == 2
    with fiona.open(output) as result:
        feature = next(iter(result))
        assert feature["properties"]["value"] == 7
        assert shape(feature["geometry"]).geom_type == "MultiLineString"
