from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from fragmentation_policy import load_policy
from fragmentation_v33_candidate import (
    V33CandidateError,
    apply_v33_candidate,
    policy_snapshot_sha256,
    runtime_policy,
)


CLASS_CODES = (12, 13, 21, 31, 32, 33, 43, 51, 52, 53, 54, 61, 62, 71)
INDEX = {code: index for index, code in enumerate(CLASS_CODES)}


def _probabilities(labels: np.ndarray) -> np.ndarray:
    values = np.full((len(CLASS_CODES), *labels.shape), 0.01 / (len(CLASS_CODES) - 1), dtype=np.float32)
    for index in range(len(CLASS_CODES)):
        values[index, labels == index] = 0.99
    return values


def _run(labels: np.ndarray) -> tuple[np.ndarray, dict]:
    return apply_v33_candidate(
        labels,
        class_codes=CLASS_CODES,
        pixel_area_m2=1.0,
        pixel_size_m=(1.0, 1.0),
        valid_mask=np.ones(labels.shape, dtype=bool),
        class_budget_mask=np.ones(labels.shape, dtype=bool),
        probabilities=_probabilities(labels),
        confidence=None,
        baseline_kind="v3_cleaned",
        full_audit=True,
    )


def _budgeted_singleton(background: int, source: int) -> np.ndarray:
    labels = np.full((30, 30), INDEX[background], dtype=np.int16)
    labels[15, 15] = INDEX[source]
    labels[2:9, 2:9] = INDEX[source]
    return labels


def test_runtime_policy_matches_approved_source_target_split():
    policy = runtime_policy()
    assert policy.protected_source_codes == frozenset({12, 33, 52, 71})
    assert policy.class_policies[52].ordinary_protected is True
    assert policy.class_policies[52].allow_same_class_bridge is True
    assert policy.class_policies[61].ordinary_protected is False
    assert policy.class_policies[61].allow_same_class_bridge is False
    assert policy_snapshot_sha256() == policy_snapshot_sha256(load_policy())


def test_unique_enclosure_absorbs_complete_13_into_protected_52_target():
    labels = _budgeted_singleton(52, 13)
    result, audit = _run(labels)
    assert result[15, 15] == INDEX[52]
    assert np.all(result[2:9, 2:9] == INDEX[13])
    assert audit["proposals_accepted"] == 1
    assert audit["accepted"][0]["kind"] == "unique_enclosure"
    assert audit["accepted"][0]["target_class_code"] == 52
    assert audit["protected_source_loss_pixel_count"] == 0


def test_executor_honours_queryable_relation_deny_rule():
    labels = _budgeted_singleton(52, 13)
    policy = deepcopy(load_policy())
    policy["decision_engine"]["relation_rules"].append({
        "id": "deny_13_to_52_runtime_test",
        "effect": "deny",
        "source": "13",
        "target": "52",
        "scenario": "*",
        "priority": 1,
        "specificity": 2,
    })
    result, audit = apply_v33_candidate(
        labels, class_codes=CLASS_CODES, pixel_area_m2=1.0,
        pixel_size_m=(1.0, 1.0), valid_mask=np.ones(labels.shape, dtype=bool),
        class_budget_mask=np.ones(labels.shape, dtype=bool),
        probabilities=_probabilities(labels), policy_document=policy,
        baseline_kind="v3_cleaned", full_audit=True,
    )
    assert np.array_equal(result, labels)
    assert audit["proposal_generation_reject_reason_counts"]["target_growth_denied"] >= 1


def test_52_source_is_immutable_but_transport_singleton_may_disappear():
    protected = _budgeted_singleton(43, 52)
    protected_result, protected_audit = _run(protected)
    assert np.array_equal(protected_result, protected)
    assert protected_audit["protected_source_loss_pixel_count"] == 0

    transport = _budgeted_singleton(52, 61)
    transport_result, transport_audit = _run(transport)
    assert transport_result[15, 15] == INDEX[52]
    assert transport_audit["transport_source_loss_pixel_count"] == 1
    assert transport_audit["transport_overlay_obligation"]["required"] is True


def test_multi_neighbour_uses_frozen_raw_v31_rarity_not_contact_length():
    labels = _budgeted_singleton(52, 13)
    labels[:15, :] = INDEX[33]
    labels[21:28, 2:9] = INDEX[13]
    labels[15, 15] = INDEX[13]
    # One 33 edge and three 52 edges around the source; rarity must still pick 33.
    labels[14, 15] = INDEX[33]
    labels[15, 14] = INDEX[52]
    labels[15, 16] = INDEX[52]
    labels[16, 15] = INDEX[52]
    result, audit = _run(labels)
    assert result[15, 15] == INDEX[33]
    accepted = next(item for item in audit["accepted"] if item["baseline_source_component_ids"])
    assert accepted["kind"] == "multi_neighbour"
    assert accepted["target_class_code"] == 33


def test_multi_class_source_bridges_two_52_components_before_rarity_fallback():
    labels = np.full((32, 32), INDEX[43], dtype=np.int16)
    labels[10:22, 10:15] = INDEX[52]
    labels[10:22, 16:21] = INDEX[52]
    labels[15, 15] = INDEX[13]
    labels[2:9, 2:9] = INDEX[13]
    result, audit = _run(labels)
    assert result[15, 15] == INDEX[52]
    accepted = audit["accepted"][0]
    assert accepted["kind"] == "same_class_bridge"
    assert accepted["target_class_code"] == 52
    assert len(accepted["baseline_target_component_ids"]) == 2
    assert accepted["component_reduction"] >= 2


def test_two_legal_overlapping_bridges_use_approved_rarity_tie_break():
    labels = np.full((42, 42), INDEX[43], dtype=np.int16)
    labels[2:12, 16:26] = INDEX[33]
    labels[12:20, 20] = INDEX[33]
    labels[30:40, 16:26] = INDEX[33]
    labels[21:30, 20] = INDEX[33]
    labels[16:26, 2:12] = INDEX[52]
    labels[20, 12:20] = INDEX[52]
    labels[16:26, 30:40] = INDEX[52]
    labels[20, 21:30] = INDEX[52]
    labels[20, 20] = INDEX[13]
    labels[32:39, 2:9] = INDEX[13]
    result, audit = _run(labels)
    assert result[20, 20] == INDEX[33]
    accepted = audit["accepted"][0]
    assert accepted["target_class_code"] == 33
    assert accepted["evidence"]["target_rarity_share"] == pytest.approx(0.0017)
    assert audit["proposal_reject_reason_counts"]["footprint_conflict"] == 1


def test_executor_uses_live_configured_conflict_priority():
    labels = np.full((42, 42), INDEX[43], dtype=np.int16)
    labels[2:12, 16:26] = INDEX[33]
    labels[12:20, 20] = INDEX[33]
    labels[30:40, 16:26] = INDEX[33]
    labels[21:30, 20] = INDEX[33]
    labels[16:26, 2:12] = INDEX[52]
    labels[20, 12:20] = INDEX[52]
    labels[16:26, 30:40] = INDEX[52]
    labels[20, 21:30] = INDEX[52]
    labels[20, 20] = INDEX[13]
    labels[32:39, 2:9] = INDEX[13]
    policy = deepcopy(load_policy())
    rank = policy["decision_engine"]["proposal_adjudication"]["proposal_rank"]
    class_code = next(item for item in rank if item["field"] == "target_class_code")
    class_code["order"] = "descending"
    policy["decision_engine"]["proposal_adjudication"]["proposal_rank"] = [
        class_code,
        *[item for item in rank if item is not class_code],
    ]
    result, audit = apply_v33_candidate(
        labels, class_codes=CLASS_CODES, pixel_area_m2=1.0,
        pixel_size_m=(1.0, 1.0), valid_mask=np.ones(labels.shape, dtype=bool),
        class_budget_mask=np.ones(labels.shape, dtype=bool),
        probabilities=_probabilities(labels), policy_document=policy,
        baseline_kind="v3_cleaned", full_audit=True,
    )
    assert result[20, 20] == INDEX[52]
    assert audit["accepted"][0]["target_class_code"] == 52


def test_external_component_and_exact_area_threshold_are_not_absorbed():
    labels = np.full((30, 30), INDEX[52], dtype=np.int16)
    labels[0, 10] = INDEX[13]
    labels[5:15, 5:20] = INDEX[13]  # exactly 150 m2, strict threshold rejects
    result, audit = _run(labels)
    assert np.array_equal(result, labels)
    assert audit["proposal_generation_reject_reason_counts"]["external_or_invalid_boundary"] >= 1
    assert audit["proposal_generation_reject_reason_counts"]["not_small_fragment"] >= 1


def test_candidate_closes_single_label_topology_boundary_and_full_audit_contracts():
    labels = _budgeted_singleton(52, 13)
    result, audit = _run(labels)
    assert result.shape == labels.shape
    assert audit["single_label"] is True
    assert audit["gap_pixels"] == audit["overlap_pixels"] == audit["outside_pixels"] == 0
    assert audit["result"]["components_4_connected"] <= audit["baseline"]["components_4_connected"]
    assert audit["result"]["dynamic_fragments_4_connected"] <= audit["baseline"]["dynamic_fragments_4_connected"]
    for key, value in audit["result"]["boundary"].items():
        assert value <= audit["baseline"]["boundary"][key]
    assert audit["full_audit"] is True
    assert audit["audit_truncated"] is False
    assert audit["raw_generated"] == audit["proposals_canonical"] + audit["duplicate_proposal_count"]


def test_candidate_rejects_noncanonical_class_order_and_missing_probabilities():
    labels = np.full((5, 5), INDEX[52], dtype=np.int16)
    with pytest.raises(V33CandidateError, match="class_codes"):
        apply_v33_candidate(
            labels, class_codes=tuple(reversed(CLASS_CODES)), pixel_area_m2=1,
            valid_mask=np.ones_like(labels, dtype=bool), probabilities=_probabilities(labels),
            baseline_kind="v3_cleaned",
        )
    with pytest.raises(V33CandidateError, match="probability"):
        apply_v33_candidate(
            labels, class_codes=CLASS_CODES, pixel_area_m2=1,
            valid_mask=np.ones_like(labels, dtype=bool), probabilities=None,
            baseline_kind="v3_cleaned",
        )
