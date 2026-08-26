from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from fragmentation_v31_candidate import CandidateError, apply_v31b, apply_v32, v32_policy, v32_policy_snapshot


CLASS_CODES = [12, 13, 21, 31, 32, 33, 43, 51, 52, 53, 54, 61, 62, 71]


def _index(code: int) -> int:
    return CLASS_CODES.index(code)


def _probabilities(labels: np.ndarray) -> np.ndarray:
    values = np.full((len(CLASS_CODES), *labels.shape), 0.1 / (len(CLASS_CODES) - 1), dtype=np.float32)
    for index in range(len(CLASS_CODES)):
        values[index][labels == index] = 0.9
    return values


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One closed 13 source: common 21 has three contacts, rare 31 one."""

    labels = np.full((30, 30), _index(21), dtype=np.int16)
    labels[1:11, 1:11] = _index(13)  # Budget denominator only; external.
    labels[10:20, 4:15] = _index(31)  # 110 target pixels, touches source once.
    source = np.array([[15, 15]], dtype=np.int32)
    labels[15, 15] = _index(13)
    probabilities = _probabilities(labels)
    row, col = source[0]
    probabilities[:, row, col] = (1.0 - .34 - .31 - .31) / (len(CLASS_CODES) - 3)
    probabilities[_index(13), row, col] = .34
    probabilities[_index(21), row, col] = .31
    probabilities[_index(31), row, col] = .31
    confidence = np.full(labels.shape, .5, dtype=np.float32)
    return labels, probabilities, confidence


def _apply(totals: dict[int, int]):
    labels, probabilities, confidence = _fixture()
    return apply_v32(
        labels, class_codes=CLASS_CODES, pixel_area_m2=1.0,
        pixel_size_m=(1.0, 1.0), probabilities=probabilities,
        confidence=confidence, baseline_kind="v3_cleaned", full_audit=True,
        frozen_global_class_pixel_totals=totals,
    )


def _totals(overrides: dict[int, int] | None = None) -> dict[int, int]:
    result = {code: 1000 for code in CLASS_CODES}
    result.update(overrides or {})
    return result


def test_short_contact_rare_class_beats_long_contact_common_class():
    result, audit = _apply(_totals({21: 1_000_000, 31: 1}))

    assert result[15, 15] == _index(31)
    item = next(row for row in audit["closed_multi_neighbor_rarity_generation_audit"] if row["reason"] == "generated")
    assert item["contact_edges_by_class"] == {"21": 3, "31": 1}
    assert item["contact_length_m_by_class"] == {"21": 3.0, "31": 1.0}
    assert item["chosen_target_class_code"] == 31
    assert item["contact_edges_used_for_filtering"] is False
    assert item["contact_edges_used_for_selection"] is False
    assert audit["target_selection_contract"].endswith("contact_edges_and_lengths_audit_only")


def test_contact_length_audit_uses_anisotropic_physical_edge_lengths_only():
    labels, probabilities, confidence = _fixture()
    result, audit = apply_v32(
        labels, class_codes=CLASS_CODES, pixel_area_m2=6.0,
        pixel_size_m=(2.0, 3.0), probabilities=probabilities,
        confidence=confidence, baseline_kind="v3_cleaned", full_audit=True,
        frozen_global_class_pixel_totals=_totals({21: 1_000_000, 31: 1}),
    )
    assert result[15, 15] == _index(31)
    item = next(row for row in audit["closed_multi_neighbor_rarity_generation_audit"] if row["reason"] == "generated")
    assert item["contact_edges_by_class"] == {"21": 3, "31": 1}
    assert item["contact_length_m_by_class"] == {"21": 8.0, "31": 2.0}
    assert item["contact_length_used_for_filtering"] is False
    assert item["contact_length_used_for_selection"] is False


def test_external_frozen_totals_change_only_target_choice_and_inputs_are_deterministic():
    rare_31, audit_31 = _apply(_totals({21: 1_000_000, 31: 1}))
    rare_21, audit_21 = _apply(_totals({21: 1, 31: 1_000_000}))
    repeat, repeat_audit = _apply(_totals({21: 1, 31: 1_000_000}))

    assert rare_31[15, 15] == _index(31)
    assert rare_21[15, 15] == _index(21)
    assert np.array_equal(rare_21, repeat)
    assert audit_21 == repeat_audit
    assert audit_31["frozen_global_class_pixel_totals_sha256"] != audit_21["frozen_global_class_pixel_totals_sha256"]


def test_equal_rarity_uses_probability_then_smaller_class_code_without_contact_length():
    labels, probabilities, confidence = _fixture()
    probabilities[_index(21), 15, 15] = .30
    probabilities[_index(31), 15, 15] = .31
    probabilities[_index(12), 15, 15] += .01
    probability_winner, _audit = apply_v32(
        labels, class_codes=CLASS_CODES, pixel_area_m2=1.0,
        probabilities=probabilities, confidence=confidence,
        baseline_kind="v3_cleaned", full_audit=True,
        frozen_global_class_pixel_totals=_totals({21: 1, 31: 1}),
    )
    assert probability_winner[15, 15] == _index(31)

    labels, probabilities, confidence = _fixture()
    code_winner, _audit = apply_v32(
        labels, class_codes=CLASS_CODES, pixel_area_m2=1.0,
        probabilities=probabilities, confidence=confidence,
        baseline_kind="v3_cleaned", full_audit=True,
        frozen_global_class_pixel_totals=_totals({21: 1, 31: 1}),
    )
    assert code_winner[15, 15] == _index(21)


def test_rarest_probability_failure_falls_through_to_next_legal_target():
    labels, probabilities, confidence = _fixture()
    probabilities[_index(31), 15, 15] = .19
    probabilities[_index(12), 15, 15] += .12
    result, audit = apply_v32(
        labels, class_codes=CLASS_CODES, pixel_area_m2=1.0,
        probabilities=probabilities, confidence=confidence,
        baseline_kind="v3_cleaned", full_audit=True,
        frozen_global_class_pixel_totals=_totals({21: 1000, 31: 1}),
    )
    assert result[15, 15] == _index(21)
    item = next(row for row in audit["closed_multi_neighbor_rarity_generation_audit"] if row["reason"] == "generated")
    failed = next(row for row in item["target_gate_audit"] if row["target_class_code"] == 31)
    assert failed["reason"] == "probability_gate"


def test_frozen_totals_fail_closed_and_inputs_are_not_mutated():
    labels, probabilities, confidence = _fixture()
    labels_before = labels.copy()
    probabilities_before = probabilities.copy()
    confidence_before = confidence.copy()
    invalid = _totals(); invalid.pop(31)
    with pytest.raises(CandidateError, match="every class code exactly once"):
        apply_v32(labels, class_codes=CLASS_CODES, pixel_area_m2=1.0,
                  probabilities=probabilities, confidence=confidence,
                  baseline_kind="v3_cleaned", frozen_global_class_pixel_totals=invalid)
    invalid = _totals(); invalid[31] = -1
    with pytest.raises(CandidateError, match="non-negative integers"):
        apply_v32(labels, class_codes=CLASS_CODES, pixel_area_m2=1.0,
                  probabilities=probabilities, confidence=confidence,
                  baseline_kind="v3_cleaned", frozen_global_class_pixel_totals=invalid)
    apply_v32(labels, class_codes=CLASS_CODES, pixel_area_m2=1.0,
              probabilities=probabilities, confidence=confidence,
              baseline_kind="v3_cleaned",
              frozen_global_class_pixel_totals=_totals({21: 1, 31: 2}))
    assert np.array_equal(labels, labels_before)
    assert np.array_equal(probabilities, probabilities_before)
    assert np.array_equal(confidence, confidence_before)


def test_t3_source_with_one_target_failing_a_b_gate_uses_next_legal_target():
    labels, probabilities, confidence = _fixture()
    policy = v32_policy()
    rows = dict(policy.class_policies)
    rows[31] = replace(rows[31], ordinary_protected=True)
    restricted = replace(policy, class_policies=MappingProxyType(rows))
    result, audit = apply_v32(
        labels, class_codes=CLASS_CODES, pixel_area_m2=1.0,
        pixel_size_m=(1.0, 1.0), probabilities=probabilities,
        confidence=confidence, baseline_kind="v3_cleaned", full_audit=True,
        frozen_global_class_pixel_totals=_totals({21: 1000, 31: 1}), policy=restricted,
    )

    assert result[15, 15] == _index(21)
    item = next(row for row in audit["closed_multi_neighbor_rarity_generation_audit"] if row["reason"] == "generated")
    assert item["chosen_target_class_code"] == 21
    denied = next(row for row in item["target_gate_audit"] if row["target_class_code"] == 31)
    assert denied == {"target_class_code": 31, "contact_edge_count_audit_only": 1, "contact_length_m_audit_only": 1.0, "eligible": False, "reason": "protected_ordinary_target"}


def test_v32_full_audit_records_totals_and_all_multi_neighbor_target_gates():
    _result, audit = _apply(_totals({21: 1, 31: 2}))

    assert audit["candidate_label"] == "V3.2"
    assert audit["frozen_global_class_pixel_totals_sum"] == sum(audit["frozen_global_class_pixel_totals"].values())
    assert audit["frozen_totals_contract"].startswith("full_owner_core_strict_valid_v3_baseline_once")
    item = next(row for row in audit["closed_multi_neighbor_rarity_generation_audit"] if row["reason"] == "generated")
    assert {row["target_class_code"] for row in item["target_gate_audit"] if row["eligible"]} == {21, 31}
    assert audit["audit_truncated"] is False
    assert v32_policy_snapshot()["v32_algorithm_contract"]["contact_measurement"] == "audit_only_not_filter_or_sort"


def test_v31b_public_api_remains_available_and_unchanged_for_same_input():
    labels, probabilities, confidence = _fixture()
    first, first_audit = apply_v31b(labels, class_codes=CLASS_CODES, pixel_area_m2=1.0, pixel_size_m=(1.0, 1.0), probabilities=probabilities, confidence=confidence, baseline_kind="v3_cleaned", full_audit=True)
    second, second_audit = apply_v31b(labels, class_codes=CLASS_CODES, pixel_area_m2=1.0, pixel_size_m=(1.0, 1.0), probabilities=probabilities, confidence=confidence, baseline_kind="v3_cleaned", full_audit=True)

    assert np.array_equal(first, second)
    assert first_audit == second_audit
    assert v32_policy().policy_id != first_audit["policy_id"]
