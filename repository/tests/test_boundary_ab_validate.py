import numpy as np
import shapely
from shapely.geometry import Polygon

from boundary_ab_validate import (
    _boundary_distances,
    _coverage_simplify_baseline,
)


def test_qsdk_distance_contract_uses_paired_directions():
    predicted = [Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])]
    reference = [Polygon([(1, 0), (5, 0), (5, 4), (1, 4)])]

    distances = _boundary_distances(predicted, reference)

    assert set(distances) == {
        "reference_to_prediction",
        "prediction_to_reference",
        "sampled_hausdorff",
    }
    assert distances["reference_to_prediction"].size > 0
    assert distances["prediction_to_reference"].size > 0
    assert distances["sampled_hausdorff"] > 0


def test_fixed_coverage_baseline_does_not_use_regularizer_displacement_budget():
    left = Polygon([(0, 0), (2, 0), (2, 1), (3, 1), (3, 2), (4, 2), (4, 4), (0, 4)])
    right = Polygon([(2, 0), (6, 0), (6, 4), (4, 4), (4, 2), (3, 2), (3, 1), (2, 1)])
    before = np.asarray([left, right], dtype=object)

    after, displacement = _coverage_simplify_baseline(before, 1.5)

    assert displacement > 0
    assert np.sum(shapely.get_num_coordinates(after)) < np.sum(
        shapely.get_num_coordinates(before)
    )
    assert shapely.coverage_is_valid(after)
    assert shapely.equals(shapely.union_all(before), shapely.union_all(after))
