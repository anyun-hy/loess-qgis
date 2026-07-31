import numpy as np
import rasterio
import shapely
from rasterio import features
from shapely.geometry import shape

from boundary_regularizer import _trend_break_indices, regularize_pixel_coverage


def test_trend_breaks_ignore_pixel_staircase_but_preserve_large_corner():
    staircase = np.asarray(
        [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3), (4, 4)],
        dtype=float,
    )
    right_angle = np.asarray(
        [(0, 0), (5, 0), (10, 0), (10, 5), (10, 10)],
        dtype=float,
    )

    assert _trend_break_indices(staircase, 12.0) == [0, len(staircase) - 1]
    assert 2 in _trend_break_indices(right_angle, 12.0)


def test_coverage_simplify_removes_stair_steps_without_overlap_or_coverage_change():
    mask = np.zeros((64, 64), dtype=np.int16)
    for row in range(mask.shape[0]):
        mask[row, 16 + row // 2:] = 1
    transform = rasterio.transform.from_origin(0, 64, 1, 1)
    before = np.asarray([
        shape(geometry) for geometry, _value in features.shapes(mask, transform=transform)
    ], dtype=object)
    before_union = shapely.union_all(before)

    after, network = regularize_pixel_coverage(before, {
        "coverage_tolerance_px": 1.5,
        "angle_threshold_deg": 12.0,
        "max_deviation_px": 1.5,
        "minimum_chain_vertices": 4,
        "preserve_outer_boundary": True,
    })

    after = np.asarray(after, dtype=object)
    after_union = shapely.union_all(after)
    assert network["shared_edge_count"] > 0
    assert np.sum(shapely.get_num_coordinates(after)) < 0.5 * np.sum(
        shapely.get_num_coordinates(before)
    )
    assert np.all(shapely.is_valid(after))
    assert np.sum(shapely.area(after)) - shapely.area(after_union) < 1e-12
    assert shapely.area(shapely.symmetric_difference(before_union, after_union)) < 1e-12


def test_closed_shared_ring_is_regularized_without_removing_the_inner_face():
    mask = np.zeros((64, 64), dtype=np.int16)
    for row in range(16, 48):
        left = 16 + (row - 16) // 2
        mask[row, left:48] = 1
    before = np.asarray(
        [shape(geometry) for geometry, _value in features.shapes(mask)],
        dtype=object,
    )

    after, network = regularize_pixel_coverage(before, {
        "coverage_tolerance_px": 1.5,
        "angle_threshold_deg": 12.0,
        "max_deviation_px": 1.5,
        "minimum_chain_vertices": 4,
        "preserve_outer_boundary": True,
    })

    after = np.asarray(after, dtype=object)
    assert network["closed_shared_edge_count"] > 0
    assert np.sum(shapely.get_num_coordinates(after)) < np.sum(
        shapely.get_num_coordinates(before)
    )
    assert len(after) == len(before)
    assert shapely.coverage_is_valid(after)
    assert shapely.equals(shapely.union_all(before), shapely.union_all(after))
