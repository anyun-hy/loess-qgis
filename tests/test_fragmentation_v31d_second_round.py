import numpy as np
import pytest

from inference_scripts.fragmentation_v31_candidate import (
    CandidateError,
    apply_v31d_candidate,
    v31d_policy,
)


def _fixture(round1_target_pixels=0):
    class_codes = (13, 21)
    original = np.zeros((50, 50), dtype=np.int16)
    original[20:30, 5:15] = 1
    original[20:30, 17:27] = 1
    baseline = original.copy()
    if round1_target_pixels:
        baseline[2, 2 : 2 + round1_target_pixels] = 1
    immutable = baseline != original
    probabilities = np.zeros((2, 50, 50), dtype=np.float32)
    probabilities[0] = 1.0
    probabilities[1, baseline == 1] = 1.0
    probabilities[0, baseline == 1] = 0.0
    probabilities[0, 20:30, 15:17] = .35
    probabilities[1, 20:30, 15:17] = .65
    return class_codes, original, baseline, immutable, probabilities


def _apply(round1_target_pixels=0):
    class_codes, original, baseline, immutable, probabilities = _fixture(
        round1_target_pixels
    )
    result, audit = apply_v31d_candidate(
        baseline,
        original_v3_labels=original,
        round1_immutable_mask=immutable,
        round1_source_loss_pixels={13: round1_target_pixels, 21: 0},
        round1_target_gain_pixels={13: 0, 21: round1_target_pixels},
        round1_protected_bridge_gain_pixels={13: 0, 21: 0},
        class_codes=class_codes,
        pixel_area_m2=1.0,
        pixel_size_m=(1.0, 1.0),
        valid_mask=np.ones(baseline.shape, dtype=bool),
        class_budget_mask=np.ones(baseline.shape, dtype=bool),
        probabilities=probabilities,
        confidence=probabilities.max(axis=0),
        policy=v31d_policy(),
        full_audit=True,
    )
    return original, baseline, immutable, result, audit


def test_bounded_second_round_accepts_new_bridge_and_preserves_round1_pixels():
    _original, baseline, immutable, result, audit = _apply(round1_target_pixels=1)

    assert audit["additional_generation_rounds"] == 1
    assert audit["proposals_accepted"] >= 1
    assert audit["changed_pixel_count"] > 0
    assert np.array_equal(result[immutable], baseline[immutable])
    assert audit["round1_immutable_preserved"] is True
    assert audit["result"]["components_4_connected"] < audit["baseline"][
        "components_4_connected"
    ]


def test_cumulative_target_budget_blocks_second_round_when_b_used_the_limit():
    original, baseline, immutable, result, audit = _apply(round1_target_pixels=4)

    assert np.array_equal(result, baseline)
    assert audit["proposals_accepted"] == 0
    assert audit["proposal_reject_reason_counts"]["cumulative_target_budget"] >= 1
    budget = audit["cumulative_class_budget_pixels"]["21"]
    assert budget["round1_target_gain"] == 4
    assert budget["round2_target_gain"] == 0
    assert budget["cumulative_target_gain"] == 4
    assert budget["target_gain_limit"] == 4.0
    assert np.array_equal(immutable, baseline != original)


def test_second_round_requires_exact_b_to_v3_immutable_mask():
    class_codes, original, baseline, immutable, probabilities = _fixture(1)
    immutable[:] = False

    with pytest.raises(CandidateError, match="exactly equal"):
        apply_v31d_candidate(
            baseline,
            original_v3_labels=original,
            round1_immutable_mask=immutable,
            round1_source_loss_pixels={13: 1, 21: 0},
            round1_target_gain_pixels={13: 0, 21: 1},
            round1_protected_bridge_gain_pixels={13: 0, 21: 0},
            class_codes=class_codes,
            pixel_area_m2=1.0,
            valid_mask=np.ones(baseline.shape, dtype=bool),
            class_budget_mask=np.ones(baseline.shape, dtype=bool),
            probabilities=probabilities,
            policy=v31d_policy(),
        )


def test_second_round_api_recomputes_round1_budget_ledger_from_pixels():
    class_codes, original, baseline, immutable, probabilities = _fixture(1)

    with pytest.raises(CandidateError, match="source_loss_pixels disagree"):
        apply_v31d_candidate(
            baseline,
            original_v3_labels=original,
            round1_immutable_mask=immutable,
            round1_source_loss_pixels={13: 0, 21: 0},
            round1_target_gain_pixels={13: 0, 21: 1},
            round1_protected_bridge_gain_pixels={13: 0, 21: 0},
            class_codes=class_codes,
            pixel_area_m2=1.0,
            valid_mask=np.ones(baseline.shape, dtype=bool),
            class_budget_mask=np.ones(baseline.shape, dtype=bool),
            probabilities=probabilities,
            policy=v31d_policy(),
        )


def test_second_round_policy_is_distinct_but_class_rules_are_unchanged():
    policy = v31d_policy()

    assert policy.policy_id == "fragmentation_v31d_bounded_second_round_candidate_v1"
    assert policy.policy_version == "v31d_bounded_second_round_20260825"
    assert policy.maximum_source_loss_fraction == .02
    assert policy.maximum_target_gain_fraction == .02
    assert policy.protected_bridge_gain_fraction == .01


def test_b_created_target_component_is_locked_but_can_anchor_round2():
    class_codes = (13, 21)
    original = np.zeros((50, 50), dtype=np.int16)
    original[20:30, 5:15] = 1
    original[20:30, 17:27] = 1
    baseline = original.copy()
    baseline[20, 15] = 1  # B-created target pixel attached to the left component.
    immutable = baseline != original
    probabilities = np.zeros((2, 50, 50), dtype=np.float32)
    probabilities[0] = 1.0
    probabilities[1, baseline == 1] = 1.0
    probabilities[0, baseline == 1] = 0.0
    probabilities[0, 20, 16] = .35
    probabilities[1, 20, 16] = .65

    result, audit = apply_v31d_candidate(
        baseline,
        original_v3_labels=original,
        round1_immutable_mask=immutable,
        round1_source_loss_pixels={13: 1, 21: 0},
        round1_target_gain_pixels={13: 0, 21: 1},
        round1_protected_bridge_gain_pixels={13: 0, 21: 0},
        class_codes=class_codes,
        pixel_area_m2=1.0,
        valid_mask=np.ones(baseline.shape, dtype=bool),
        class_budget_mask=np.ones(baseline.shape, dtype=bool),
        probabilities=probabilities,
        confidence=probabilities.max(axis=0),
        policy=v31d_policy(),
        full_audit=True,
    )

    assert audit["round1_dependency_lock_pixel_count"] > int(immutable.sum())
    assert np.array_equal(result[immutable], baseline[immutable])
    assert audit["round1_dependency_lock_preserved"] is True
    assert audit["proposals_accepted"] >= 1
