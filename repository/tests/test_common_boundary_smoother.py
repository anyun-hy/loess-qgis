import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box
from shapely.ops import linemerge, split

from boundary_fitting.unit_runtime import _fit_or_subdivide
from common_boundary_smoother import _pair_validation_failure, smooth_common_boundaries
from polyline_smoother import SmoothingConfig


def _adjacent_staircase_polygons():
    shared = [(10.0, 0.0)]
    x = 10.0
    y = 0.0
    for index in range(10):
        y += 1.0
        shared.append((x, y))
        if index < 9:
            x += 1.0
            shared.append((x, y))
    left = Polygon([(0, 0), *shared, (0, 10), (0, 0)])
    right = Polygon(
        [(10, 0), (30, 0), (30, 10), shared[-1], *reversed(shared[:-1]), (10, 0)]
    )
    return shared, [
        {"polygon_id": "left", "class_code": 12, "geometry": left},
        {"polygon_id": "right", "class_code": 31, "geometry": right},
    ]


def _one_line(geometry):
    if isinstance(geometry, LineString):
        return geometry
    if isinstance(geometry, MultiLineString):
        return linemerge(geometry)
    raise AssertionError(geometry.geom_type)


def test_common_divider_is_fitted_once_and_reused_by_both_polygons():
    raw_shared, records = _adjacent_staircase_polygons()
    formal, report = smooth_common_boundaries(
        records,
        SmoothingConfig(smoothing_factor=1.0, output_spacing=0.5, max_deviation=None),
    )
    assert report["fit_version"] == "divider_cubic_bspline_v1"
    assert report["spline_count"] == 1
    assert report["candidate_validation"]["scope"] == "per_common_divider"
    assert report["validation"] == {
        "passed": True,
        "scope": "all_output_polygons",
        "invalid_count": 0,
        "coordinate_spaces": ["input"],
    }
    assert all(item["fit_status"] == "changed" for item in formal)
    assert all(item["geometry"].is_valid for item in formal)
    assert all(item["geometry"].area > 0 for item in formal)
    assert abs(
        sum(item["geometry"].area for item in formal)
        - sum(item["geometry"].area for item in records)
    ) <= 1e-6

    divider = _one_line(
        formal[0]["geometry"].boundary.intersection(formal[1]["geometry"].boundary)
    )
    coordinates = np.asarray(divider.coords)
    deltas = np.diff(coordinates, axis=0)
    axis_aligned = (np.abs(deltas[:, 0]) < 1e-8) | (np.abs(deltas[:, 1]) < 1e-8)
    assert float(axis_aligned.mean()) < 0.1
    assert divider.distance(LineString(raw_shared).boundary) == 0.0
    assert np.allclose(
        np.asarray(report["diagnostics"][0]["fitted_points"]),
        coordinates,
    ) or np.allclose(
        np.asarray(report["diagnostics"][0]["fitted_points"]),
        coordinates[::-1],
    )


def test_two_dividers_can_replace_different_sections_of_one_ring():
    raw_shared, pair = _adjacent_staircase_polygons()
    right_parts = list(
        split(
            pair[1]["geometry"],
            LineString([(0.0, 5.5), (40.0, 5.5)]),
        ).geoms
    )
    records = [pair[0]] + [
        {
            "polygon_id": f"right_{index}",
            "class_code": 31 + index,
            "geometry": geometry,
        }
        for index, geometry in enumerate(right_parts)
    ]
    formal, report = smooth_common_boundaries(
        records,
        SmoothingConfig(
            smoothing_factor=1.0,
            output_spacing=0.5,
            max_deviation=None,
            min_point_count=4,
        ),
    )

    assert report["spline_count"] == 2
    assert formal[0]["fit_status"] == "changed"
    assert all(item["geometry"].is_valid for item in formal)
    assert all(item["geometry"].area > 0 for item in formal)
    assert abs(
        sum(item["geometry"].area for item in formal)
        - sum(item["geometry"].area for item in records)
    ) <= 1e-6
    for right in formal[1:]:
        divider = _one_line(
            formal[0]["geometry"].boundary.intersection(right["geometry"].boundary)
        )
        assert divider.length > 0
    assert LineString(raw_shared).length > 0


def test_closed_island_divider_is_fitted_once_for_outer_hole_and_inner_polygon():
    shared = [
        (5, 5),
        (7, 5),
        (7, 6),
        (9, 6),
        (9, 8),
        (8, 8),
        (8, 10),
        (6, 10),
        (6, 9),
        (4, 9),
        (4, 7),
        (5, 7),
        (5, 5),
    ]
    records = [
        {
            "polygon_id": "outer",
            "class_code": 12,
            "geometry": Polygon([(0, 0), (15, 0), (15, 15), (0, 15), (0, 0)], [shared]),
        },
        {
            "polygon_id": "island",
            "class_code": 31,
            "geometry": Polygon(shared),
        },
    ]
    formal, report = smooth_common_boundaries(
        records,
        SmoothingConfig(smoothing_factor=1.0, output_spacing=0.5, max_deviation=None),
    )

    assert report["spline_count"] == 1
    shared_after = _one_line(
        formal[0]["geometry"].boundary.intersection(formal[1]["geometry"].boundary)
    )
    expected = np.asarray(report["diagnostics"][0]["fitted_points"])
    actual = np.asarray(shared_after.coords)
    assert np.allclose(expected, actual) or np.allclose(expected, actual[::-1])


def test_tiny_closed_divider_keeps_original_geometry_when_spline_collapses_it():
    shared = [(5, 5), (7, 5), (7, 6), (5, 6), (5, 5)]
    records = [
        {
            "polygon_id": "outer",
            "class_code": 12,
            "geometry": Polygon(
                [(0, 0), (15, 0), (15, 15), (0, 15), (0, 0)],
                [shared],
            ),
        },
        {
            "polygon_id": "tiny_island",
            "class_code": 31,
            "geometry": Polygon(shared),
        },
    ]

    formal, report = smooth_common_boundaries(
        records,
        SmoothingConfig(
            smoothing_factor=1.0,
            output_spacing=0.5,
            max_deviation=None,
            min_point_count=4,
        ),
    )

    assert report["spline_count"] == 0
    assert report["skipped_invalid_count"] == 1
    assert report["unchanged_count"] == 1
    assert report["diagnostics"][0]["status"] == "skipped_validation_failed"
    assert all(item["geometry"].is_valid for item in formal)
    assert all(item["geometry"].area > 0 for item in formal)
    assert formal[0]["geometry"].equals(records[0]["geometry"])
    assert formal[1]["geometry"].equals(records[1]["geometry"])


def test_pair_validation_rejects_geometry_that_collapses_in_output_crs():
    left_before = MultiPolygon([box(0, 0, 10, 10)])
    right_before = MultiPolygon([box(10, 0, 20, 10)])
    left_after = MultiPolygon(
        [box(0, 0, 10, 10), box(30, 30, 30 + 1e-12, 30 + 1e-12)]
    )
    right_after = right_before

    assert (
        _pair_validation_failure(
            left_before,
            right_before,
            left_after,
            right_after,
            1e-6,
        )
        == ""
    )
    assert (
        _pair_validation_failure(
            left_before,
            right_before,
            left_after,
            right_after,
            1e-6,
            (1.63206857239e-5, 0.0, 0.0, -1.63206857239e-5, 110.268, 37.653),
        )
        == "output_crs_invalid_polygon"
    )


def test_partition_unit_runtime_uses_common_divider_mode():
    probabilities = np.zeros((14, 24, 24), dtype=np.float32)
    for row in range(24):
        threshold = 9 + row // 3
        probabilities[0, row, :threshold] = 1.0
        probabilities[1, row, threshold:] = 1.0
    raw, formal, report = _fit_or_subdivide(
        probabilities,
        {
            "unit_id": "core_00000_00000",
            "pixel_window": {"x0": 0, "y0": 0, "x1": 24, "y1": 24},
        },
        [12, 13, 21, 31, 32, 33, 43, 51, 52, 53, 54, 61, 62, 71],
        smoothing_config=SmoothingConfig(
            smoothing_factor=1.0,
            output_spacing=0.5,
            max_deviation=None,
            min_point_count=4,
        ),
        max_features=1000,
        max_segments=10000,
        min_core_px=4,
    )

    assert len(raw) == len(formal) == 2
    assert report["fit_version"] == "divider_cubic_bspline_v1"
    assert report["spline_count"] == 1
    assert all(record["fit_status"] == "changed" for record in formal)
