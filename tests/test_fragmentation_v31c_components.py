import numpy as np
import pytest

from inference_scripts.fragmentation_v31_candidate.v31c_components import (
    ComponentIndexError,
    CoreTile,
    GlobalComponentKey,
    build_global_component_index,
)


def _tile(core_id, window, labels, valid=None, pixel_area_m2=1.0):
    labels = np.asarray(labels, dtype=np.int16)
    return CoreTile(core_id, window, labels, np.ones(labels.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool), pixel_area_m2)


def test_cross_core_components_have_global_counts_and_stable_minimum_key():
    left = _tile("left", (0, 0, 2, 2), [[7, 7], [2, 7]])
    right = _tile("right", (0, 2, 2, 2), [[7, 3], [7, 7]])
    result = build_global_component_index([right, left], [(1, 3), (1, 0)], global_shape=(2, 4))

    assert result.query_components[0].key == GlobalComponentKey(7, 0, 0)
    assert result.query_components[0].pixel_count == 6
    assert result.query_components[1] == result.query_components[1].__class__(
        (1, 0), GlobalComponentKey(2, 1, 0), 1, 1.0
    )
    assert result.global_component_count == 3


def test_component_answers_are_independent_of_tile_order_and_join_vertical_seam():
    top = _tile("top", (0, 0, 1, 2), [[4, 4]])
    bottom = _tile("bottom", (1, 0, 2, 2), [[4, 9], [4, 4]])
    points = [(2, 1), (0, 1)]
    forward = build_global_component_index([top, bottom], points, global_shape=(3, 2))
    reverse = build_global_component_index([bottom, top], points, global_shape=(3, 2))

    assert forward == reverse
    assert forward.query_components[0].key == GlobalComponentKey(4, 0, 0)
    assert forward.query_components[0].pixel_count == 5


def test_area_is_summed_across_cores_with_the_component_pixel_count():
    left = _tile("left", (0, 0, 1, 1), [[5]], pixel_area_m2=2.5)
    right = _tile("right", (0, 1, 1, 1), [[5]], pixel_area_m2=3.0)
    answer = build_global_component_index([left, right], [(0, 1)]).query_components[0]

    assert (answer.pixel_count, answer.area_m2) == (2, 5.5)


def test_different_classes_touching_at_a_seam_are_not_merged():
    left = _tile("left", (0, 0, 1, 1), [[1]])
    right = _tile("right", (0, 1, 1, 1), [[2]])
    result = build_global_component_index([left, right], [(0, 0), (0, 1)])

    assert [answer.key for answer in result.query_components] == [GlobalComponentKey(1, 0, 0), GlobalComponentKey(2, 0, 1)]
    assert [answer.pixel_count for answer in result.query_components] == [1, 1]


def test_query_expected_class_mismatch_is_rejected():
    tile = _tile("only", (0, 0, 1, 1), [[7]])
    with pytest.raises(ComponentIndexError, match="expected class 2, found 7"):
        build_global_component_index([tile], [(0, 0, 2)])


@pytest.mark.parametrize(
    ("tiles", "message"),
    [
        ([_tile("a", (0, 0, 1, 2), [[1, 1]]), _tile("b", (0, 1, 1, 2), [[1, 1]])], "overlap"),
        ([_tile("a", (0, 0, 1, 1), [[1]]), _tile("b", (0, 2, 1, 1), [[1]])], "gap"),
    ],
)
def test_partition_overlap_and_gap_are_rejected_without_dense_global_array(tiles, message):
    with pytest.raises(ComponentIndexError, match=message):
        build_global_component_index(tiles, [])


def test_invalid_and_outside_queries_are_rejected():
    tile = _tile("only", (0, 0, 2, 2), [[1, 1], [1, 1]], [[True, False], [True, True]])
    with pytest.raises(ComponentIndexError, match="not valid"):
        build_global_component_index([tile], [(0, 1)])
    with pytest.raises(ComponentIndexError, match="outside"):
        build_global_component_index([tile], [(2, 0)])
