from __future__ import annotations

import numpy as np
import pytest

from fragmentation_v34_candidate import V34CandidateError, apply_v34_candidate


CLASS_CODES = (12, 13, 21, 31, 32, 33, 43, 51, 52, 53, 54, 61, 62, 71)
INDEX = {code: index for index, code in enumerate(CLASS_CODES)}


def _probabilities(labels: np.ndarray) -> np.ndarray:
    result = np.full(
        (len(CLASS_CODES), *labels.shape),
        0.01 / (len(CLASS_CODES) - 1),
        dtype=np.float32,
    )
    for index in range(len(CLASS_CODES)):
        result[index, labels == index] = 0.99
    return result


def _zeros() -> dict[int, int]:
    return {code: 0 for code in CLASS_CODES}


def _fixture(*, source_budget_full: bool = False):
    original = np.full((60, 60), INDEX[52], dtype=np.int16)
    original[2:9, 2:9] = INDEX[13]
    original[30, 30] = INDEX[13]
    original[20:31, 40:50] = INDEX[43]
    baseline = original.copy()
    if source_budget_full:
        baseline[2, 2] = INDEX[52]
        unlock = (2, 2)
        source = {**_zeros(), 13: 1}
    else:
        baseline[30, 29] = INDEX[52]
        original[30, 29] = INDEX[43]
        unlock = (30, 29)
        source = {**_zeros(), 43: 1}
    target = {**_zeros(), 52: 1}
    immutable = baseline != original
    assert immutable[unlock]
    return original, baseline, immutable, source, target


def _apply(original, baseline, immutable, source, target):
    return apply_v34_candidate(
        baseline,
        original_v3_labels=original,
        round1_immutable_mask=immutable,
        round1_source_loss_pixels=source,
        round1_target_gain_pixels=target,
        round1_protected_bridge_gain_pixels=_zeros(),
        class_codes=CLASS_CODES,
        pixel_area_m2=1.0,
        pixel_size_m=(1.0, 1.0),
        valid_mask=np.ones(baseline.shape, dtype=bool),
        class_budget_mask=np.ones(baseline.shape, dtype=bool),
        probabilities=_probabilities(baseline),
        baseline_kind="v33_cleaned",
        full_audit=True,
    )


def test_second_pass_uses_v33_created_target_as_anchor_and_preserves_round1():
    original, baseline, immutable, source, target = _fixture()
    result, audit = _apply(original, baseline, immutable, source, target)
    assert result[30, 30] == INDEX[52]
    assert np.array_equal(result[immutable], baseline[immutable])
    assert audit["round1_immutable_preserved"] is True
    assert audit["proposals_accepted"] >= 1
    assert audit["protected_source_loss_pixel_count"] == 0
    assert audit["gap_pixels"] == audit["overlap_pixels"] == audit["outside_pixels"] == 0


def test_cumulative_source_budget_blocks_another_round():
    original, baseline, immutable, source, target = _fixture(source_budget_full=True)
    result, audit = _apply(original, baseline, immutable, source, target)
    assert result[30, 30] == baseline[30, 30]
    assert audit["proposal_reject_reason_counts"]["cumulative_source_budget"] >= 1
    budget = audit["cumulative_class_budget_pixels"]["13"]
    assert budget["round1_source_loss"] == 1
    assert budget["round2_source_loss"] == 0
    assert budget["cumulative_source_loss"] == 1
    assert budget["source_loss_limit"] == pytest.approx(1.0)


def test_round1_immutable_and_transition_ledgers_are_recomputed():
    original, baseline, immutable, source, target = _fixture()
    with pytest.raises(V34CandidateError, match="immutable_mask"):
        _apply(original, baseline, np.zeros_like(immutable), source, target)
    bad_source = dict(source)
    bad_source[43] = 0
    with pytest.raises(V34CandidateError, match="source ledger"):
        _apply(original, baseline, immutable, bad_source, target)


def test_protected_direct_target_gain_does_not_consume_bridge_ledger():
    original = np.full((20, 20), INDEX[13], dtype=np.int16)
    original[0:10, 0:10] = INDEX[52]
    baseline = original.copy()
    baseline[10, 4:6] = INDEX[52]
    immutable = baseline != original
    source = {**_zeros(), 13: 2}
    target = {**_zeros(), 52: 2}
    result, audit = _apply(original, baseline, immutable, source, target)
    assert np.array_equal(result[immutable], baseline[immutable])
    row = audit["cumulative_class_budget_pixels"]["52"]
    assert row["round1_target_gain"] == 2
    assert row["round1_protected_bridge_gain"] == 0
    assert row["target_gain_limit"] == pytest.approx(2.0)
    assert row["protected_bridge_gain_limit"] == pytest.approx(1.0)
