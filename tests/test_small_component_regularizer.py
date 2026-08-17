import numpy as np
import pytest
from affine import Affine

from small_component_regularizer import (
    SmallComponentPolicy,
    physical_pixel_area_m2,
    regularize_small_components,
)


CLASS_CODES = [13, 31, 61, 71]


def _policy(**overrides):
    values = {
        "thresholds_m2": {13: 20.0, 31: 20.0, 61: 20.0, 71: 20.0},
        "hard_absorb_below_m2": 4.0,
        "maximum_mean_confidence": 0.7,
        "maximum_probability_drop": 0.15,
        "preserve_border_components": True,
    }
    values.update(overrides)
    return SmallComponentPolicy(**values)


def test_low_confidence_island_is_absorbed_into_larger_neighbor():
    labels = np.zeros((9, 9), dtype=np.int16)
    labels[4, 4] = 1
    confidence = np.full(labels.shape, 0.9, dtype=np.float32)
    confidence[4, 4] = 0.4

    result, report = regularize_small_components(
        labels,
        class_codes=CLASS_CODES,
        pixel_area_m2=1.0,
        policy=_policy(),
        confidence=confidence,
    )

    assert result[4, 4] == 0
    assert report["changed_component_count"] == 1
    assert report["changed_pixel_count"] == 1
    assert report["changed_pair_counts"] == {"31->13": 1}


def test_high_confidence_component_above_hard_absorb_limit_is_preserved():
    labels = np.zeros((12, 12), dtype=np.int16)
    labels[4:6, 4:7] = 1
    confidence = np.full(labels.shape, 0.95, dtype=np.float32)

    result, report = regularize_small_components(
        labels,
        class_codes=CLASS_CODES,
        pixel_area_m2=1.0,
        policy=_policy(),
        confidence=confidence,
    )

    assert np.array_equal(result, labels)
    assert report["changed_component_count"] == 0
    assert report["kept_reason_counts"]["high_confidence"] == 1


def test_component_below_hard_absorb_limit_ignores_confidence_guard():
    labels = np.zeros((9, 9), dtype=np.int16)
    labels[4, 4] = 1
    confidence = np.full(labels.shape, 0.99, dtype=np.float32)

    result, report = regularize_small_components(
        labels,
        class_codes=CLASS_CODES,
        pixel_area_m2=1.0,
        policy=_policy(),
        confidence=confidence,
    )

    assert result[4, 4] == 0
    assert report["changed_component_count"] == 1


@pytest.mark.parametrize("class_index", [2, 3])
def test_protected_linear_classes_are_never_absorbed_by_area(class_index):
    labels = np.zeros((9, 9), dtype=np.int16)
    labels[4, 4] = class_index

    result, report = regularize_small_components(
        labels,
        class_codes=CLASS_CODES,
        pixel_area_m2=1.0,
        policy=_policy(),
        confidence=np.full(labels.shape, 0.1, dtype=np.float32),
    )

    assert np.array_equal(result, labels)
    assert report["kept_reason_counts"]["protected_class"] == 1


def test_processing_border_component_is_preserved_for_seam_safety():
    labels = np.zeros((9, 9), dtype=np.int16)
    labels[0, 4] = 1

    result, report = regularize_small_components(
        labels,
        class_codes=CLASS_CODES,
        pixel_area_m2=1.0,
        policy=_policy(),
    )

    assert np.array_equal(result, labels)
    assert report["kept_reason_counts"]["touches_processing_border"] == 1


def test_disallowed_target_class_does_not_receive_absorbed_component():
    labels = np.zeros((9, 9), dtype=np.int16)
    labels[:, 6:] = 2
    labels[4, 5] = 1

    result, report = regularize_small_components(
        labels,
        class_codes=CLASS_CODES,
        pixel_area_m2=1.0,
        policy=_policy(
            protected_class_codes=frozenset(),
            disallowed_target_class_codes=frozenset({61}),
        ),
    )

    assert result[4, 5] == 0
    assert report["changed_pair_counts"] == {"31->13": 1}


def test_semantic_compatibility_selects_allowed_neighbor():
    labels = np.zeros((11, 11), dtype=np.int16)
    labels[:, 8:] = 2
    labels[4:7, 6:8] = 1

    result, report = regularize_small_components(
        labels,
        class_codes=CLASS_CODES,
        pixel_area_m2=1.0,
        policy=_policy(
            protected_class_codes=frozenset(),
            allow_protected_targets=True,
            compatible_target_class_codes={31: frozenset({61})},
        ),
    )

    assert np.all(result[4:7, 6:8] == 2)
    assert report["changed_pair_counts"] == {"31->61": 1}


def test_tiny_component_can_bypass_semantic_compatibility():
    labels = np.zeros((9, 9), dtype=np.int16)
    labels[4, 4] = 1

    result, report = regularize_small_components(
        labels,
        class_codes=CLASS_CODES,
        pixel_area_m2=1.0,
        policy=_policy(
            compatible_target_class_codes={31: frozenset({61})},
            compatibility_bypass_below_m2=2.0,
        ),
    )

    assert result[4, 4] == 0
    assert report["changed_pair_counts"] == {"31->13": 1}


def test_elongated_component_is_preserved_by_shape_guard():
    labels = np.zeros((15, 20), dtype=np.int16)
    labels[7, 5:15] = 1

    result, report = regularize_small_components(
        labels,
        class_codes=CLASS_CODES,
        pixel_area_m2=1.0,
        policy=_policy(
            preserve_elongated_components=True,
            elongated_minimum_area_m2=5.0,
            elongated_minimum_aspect_ratio=6.0,
            elongated_maximum_mean_width_m=2.0,
        ),
    )

    assert np.array_equal(result, labels)
    assert report["kept_reason_counts"]["elongated_component"] == 1


def test_compact_component_is_not_preserved_by_shape_guard():
    labels = np.zeros((15, 20), dtype=np.int16)
    labels[6:9, 8:11] = 1

    result, report = regularize_small_components(
        labels,
        class_codes=CLASS_CODES,
        pixel_area_m2=1.0,
        policy=_policy(
            preserve_elongated_components=True,
            elongated_minimum_area_m2=5.0,
            elongated_minimum_aspect_ratio=6.0,
            elongated_maximum_mean_width_m=2.0,
        ),
    )

    assert np.all(result[6:9, 8:11] == 0)
    assert report["changed_component_count"] == 1


def test_source_class_loss_budget_preserves_excess_components():
    labels = np.zeros((15, 20), dtype=np.int16)
    labels[4:6, 4:6] = 1
    labels[9:11, 14:16] = 1

    result, report = regularize_small_components(
        labels,
        class_codes=CLASS_CODES,
        pixel_area_m2=1.0,
        policy=_policy(maximum_source_class_loss_fraction=0.5),
    )

    assert np.count_nonzero(result == 1) == 4
    assert report["changed_component_count"] == 1
    assert report["kept_reason_counts"]["source_class_loss_budget"] == 1


def test_minimum_remaining_class_area_prevents_class_disappearance():
    labels = np.zeros((15, 20), dtype=np.int16)
    labels[7, 8:11] = 1

    result, report = regularize_small_components(
        labels,
        class_codes=CLASS_CODES,
        pixel_area_m2=1.0,
        policy=_policy(
            maximum_source_class_loss_fraction=None,
            minimum_remaining_class_area_m2=1.0,
        ),
    )

    assert np.array_equal(result, labels)
    assert report["kept_reason_counts"]["source_class_minimum_area"] == 1


def test_target_class_gain_budget_limits_absorbed_area():
    labels = np.zeros((15, 20), dtype=np.int16)
    labels[4:6, 4:6] = 1
    labels[9:11, 14:16] = 1

    result, report = regularize_small_components(
        labels,
        class_codes=CLASS_CODES,
        pixel_area_m2=1.0,
        policy=_policy(maximum_target_class_gain_fraction=0.02),
    )

    assert np.count_nonzero(result == 1) == 4
    assert report["changed_component_count"] == 1
    assert report["kept_reason_counts"]["target_class_gain_budget"] == 1


def test_class_budget_mask_scopes_source_loss_to_output_core():
    labels = np.zeros((15, 20), dtype=np.int16)
    labels[4:6, 4:6] = 1
    labels[9:11, 14:16] = 1
    budget_mask = np.zeros(labels.shape, dtype=bool)
    budget_mask[8:, 10:] = True

    result, report = regularize_small_components(
        labels,
        class_codes=CLASS_CODES,
        pixel_area_m2=1.0,
        policy=_policy(maximum_source_class_loss_fraction=0.5),
        class_budget_mask=budget_mask,
    )

    assert np.count_nonzero(result[4:6, 4:6] == 0) == 4
    assert np.count_nonzero(result[9:11, 14:16] == 1) == 4
    assert report["class_budget_pixel_count"] == int(np.count_nonzero(budget_mask))


def test_probability_selects_plausible_neighbor_over_longer_contact():
    labels = np.zeros((11, 11), dtype=np.int16)
    labels[:, 7:] = 2
    labels[4:7, 4:7] = 1
    probabilities = np.zeros((len(CLASS_CODES), *labels.shape), dtype=np.float32)
    probabilities[labels, *np.indices(labels.shape)] = 0.9
    probabilities[:, 4:7, 4:7] = 0.0
    probabilities[1, 4:7, 4:7] = 0.45
    probabilities[0, 4:7, 4:7] = 0.10
    probabilities[2, 4:7, 4:7] = 0.44

    result, report = regularize_small_components(
        labels,
        class_codes=CLASS_CODES,
        pixel_area_m2=1.0,
        policy=_policy(
            protected_class_codes=frozenset(),
            allow_protected_targets=True,
            probability_weight=4.0,
            adjacency_weight=1.0,
            maximum_probability_drop=0.05,
        ),
        probabilities=probabilities,
    )

    assert np.all(result[4:7, 4:7] == 2)
    assert report["changed_pair_counts"] == {"31->61": 1}


def test_probability_drop_guard_retains_semantically_strong_component():
    labels = np.zeros((12, 12), dtype=np.int16)
    labels[4:7, 4:7] = 1
    probabilities = np.full((len(CLASS_CODES), *labels.shape), 0.01, dtype=np.float32)
    probabilities[0, labels == 0] = 0.9
    probabilities[1, 4:7, 4:7] = 0.9
    probabilities[0, 4:7, 4:7] = 0.05

    result, report = regularize_small_components(
        labels,
        class_codes=CLASS_CODES,
        pixel_area_m2=1.0,
        policy=_policy(maximum_probability_drop=0.15),
        probabilities=probabilities,
    )

    assert np.array_equal(result, labels)
    assert report["kept_reason_counts"]["probability_drop"] == 1


def test_web_mercator_pixel_area_is_corrected_at_local_latitude():
    latitude_degrees = 37.5
    radius = 6378137.0
    mercator_y = radius * np.log(np.tan(np.pi / 4.0 + np.deg2rad(latitude_degrees) / 2.0))
    transform = Affine(0.25, 0.0, 0.0, 0.0, -0.25, mercator_y)

    area = physical_pixel_area_m2(
        transform,
        "EPSG:3857",
        height=1,
        width=1,
    )

    assert area == pytest.approx(0.25 * 0.25 * np.cos(np.deg2rad(latitude_degrees)) ** 2)
