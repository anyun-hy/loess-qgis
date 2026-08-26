from __future__ import annotations

from dataclasses import replace
import json
from types import MappingProxyType

import numpy as np
import pytest

import fragmentation_v31_candidate.candidate as candidate_module
from fragmentation_v31_candidate import (
    CandidateError,
    apply_v31a,
    apply_v31b,
    policy_snapshot,
    policy_snapshot_sha256,
    policy_v31a,
    policy_v31b,
)


CLASS_CODES = [12, 13, 21, 31, 32, 33, 43, 51, 52, 53, 54, 61, 62, 71]


def _index(code: int) -> int:
    return CLASS_CODES.index(code)


def _confident_probabilities(labels: np.ndarray) -> np.ndarray:
    probabilities = np.full(
        (len(CLASS_CODES), *labels.shape),
        0.1 / (len(CLASS_CODES) - 1),
        dtype=np.float32,
    )
    for index in range(len(CLASS_CODES)):
        probabilities[index][labels == index] = 0.9
    return probabilities


def _set_target_evidence(
    probabilities: np.ndarray,
    labels: np.ndarray,
    pixels: np.ndarray,
    *,
    target_code: int,
    current_probability: float,
    target_probability: float,
) -> None:
    rows, cols = pixels[:, 0], pixels[:, 1]
    target_index = _index(target_code)
    current_indices = labels[rows, cols]
    remainder = 1.0 - current_probability - target_probability
    other_count = len(CLASS_CODES) - 2
    probabilities[:, rows, cols] = remainder / other_count
    probabilities[current_indices, rows, cols] = current_probability
    probabilities[target_index, rows, cols] = target_probability


def _apply(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    confidence: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    class_budget_mask: np.ndarray | None = None,
    policy=None,
    pixel_area_m2: float = 1.0,
    pixel_size_m: tuple[float, float] = (1.0, 1.0),
    baseline_kind: str = "v3_cleaned",
    full_audit: bool = False,
):
    return apply_v31a(
        labels,
        class_codes=CLASS_CODES,
        pixel_area_m2=pixel_area_m2,
        pixel_size_m=pixel_size_m,
        valid_mask=valid_mask,
        class_budget_mask=class_budget_mask,
        probabilities=probabilities,
        confidence=confidence,
        policy=policy,
        baseline_kind=baseline_kind,
        full_audit=full_audit,
    )


def _apply_b(
    labels: np.ndarray,
    probabilities: np.ndarray,
    **kwargs,
):
    return apply_v31b(
        labels,
        class_codes=CLASS_CODES,
        pixel_area_m2=kwargs.pop("pixel_area_m2", 1.0),
        pixel_size_m=kwargs.pop("pixel_size_m", (1.0, 1.0)),
        probabilities=probabilities,
        baseline_kind=kwargs.pop("baseline_kind", "v3_cleaned"),
        **kwargs,
    )


EXPECTED_ROWS = {
    12: (50, True, 0, True, 6, 50, .35, .05, .25),
    13: (150, False, 150, False, 0, 0, .30, .10, .20),
    21: (80, False, 80, True, 12, 160, .30, .10, .20),
    31: (100, False, 100, False, 0, 0, .30, .10, .20),
    32: (100, False, 100, True, 12, 200, .30, .10, .20),
    33: (60, True, 0, True, 6, 60, .35, .05, .25),
    43: (100, False, 100, False, 0, 0, .30, .10, .20),
    51: (80, False, 80, True, 8, 160, .35, .08, .20),
    52: (60, False, 60, True, 10, 120, .35, .08, .20),
    53: (50, False, 50, False, 0, 0, .35, .08, .20),
    54: (50, False, 50, False, 0, 0, .35, .08, .20),
    61: (30, True, 0, True, 4, 30, .40, .05, .25),
    62: (50, True, 0, True, 6, 50, .40, .05, .25),
    71: (50, True, 0, True, 6, 50, .40, .05, .25),
}


def test_approved_policy_snapshot_is_exact_stable_and_detached():
    snapshot = policy_snapshot()

    assert snapshot["policy_id"] == "fragmentation_v31a_class_topology_candidate_v1"
    assert snapshot["maximum_source_loss_fraction"] == 0.02
    assert snapshot["maximum_target_gain_fraction"] == 0.02
    assert snapshot["protected_bridge_gain_fraction"] == 0.01
    assert snapshot["island_maximum_mean_confidence"] == 0.65
    assert snapshot["protected_source_codes"] == [12, 33, 61, 62, 71]
    assert snapshot["algorithm_contract"]["bridge_edge_distance"] == (
        "euclidean_cell_polygon_edge_distance_m"
    )
    assert snapshot["algorithm_contract"]["bridge_path_length"] == (
        "four_neighbour_path_length_m"
    )
    assert set(map(int, snapshot["class_policies"])) == set(CLASS_CODES)
    for code, expected in EXPECTED_ROWS.items():
        row = snapshot["class_policies"][str(code)]
        assert tuple(row.values()) == expected

    original_sha = policy_snapshot_sha256()
    snapshot["class_policies"]["12"]["dynamic_fragmentation_m2"] = -1
    assert policy_snapshot()["class_policies"]["12"]["dynamic_fragmentation_m2"] == 50
    assert policy_snapshot_sha256() == original_sha
    assert json.dumps(policy_snapshot(), sort_keys=True) == json.dumps(
        policy_snapshot(), sort_keys=True
    )


def test_v31b_policy_is_independent_and_records_incremental_mode():
    b_policy = policy_v31b()
    assert b_policy.policy_id == "fragmentation_v31b_dependency_incremental_candidate_v1"
    assert b_policy.policy_version == "v31b_dependency_incremental_20260824"
    assert policy_snapshot(b_policy)["adjudication_mode"] == "dependency_incremental_v1"
    # V3.1-A's approved snapshot and default alias must remain untouched.
    assert policy_v31a().policy_id == "fragmentation_v31a_class_topology_candidate_v1"
    assert policy_snapshot()["policy_id"] == "fragmentation_v31a_class_topology_candidate_v1"


@pytest.mark.parametrize(
    ("bad_probability", "pixel_area", "pixel_size", "baseline_kind", "message"),
    [
        (1.01, 1.0, (1.0, 1.0), "v3_cleaned", "probabilities must lie"),
        (0.8, 1.0, (1.0, 1.0), "v3_cleaned", "sum to one"),
        (0.9, 2.0, (1.0, 1.0), "v3_cleaned", "product must equal"),
        (0.9, 1.0, (1.0, 1.0), "", "baseline_kind"),
    ],
)
def test_invalid_probability_geometry_and_baseline_metadata_are_rejected(
    bad_probability, pixel_area, pixel_size, baseline_kind, message
):
    labels = np.full((3, 3), _index(13), dtype=np.int16)
    probabilities = _confident_probabilities(labels)
    probabilities[_index(13), 1, 1] = bad_probability

    with pytest.raises(CandidateError, match=message):
        _apply(
            labels,
            probabilities,
            pixel_area_m2=pixel_area,
            pixel_size_m=pixel_size,
            baseline_kind=baseline_kind,
        )


def test_components_and_dynamic_fragments_use_four_connectivity():
    labels = np.array(
        [[_index(13), _index(21)], [_index(21), _index(13)]], dtype=np.int16
    )
    probabilities = _confident_probabilities(labels)

    result, report = _apply(labels, probabilities)

    assert np.array_equal(result, labels)
    assert report["topology_connectivity"] == 4
    assert report["baseline"]["components_4_connected"] == 4
    assert report["single_pass_from_frozen_baseline"] is True
    assert report["cascade_generation"] is False


@pytest.mark.parametrize(("component_area", "expected_dynamic"), [(149, 1), (150, 0)])
def test_dynamic_fragment_threshold_is_strict(component_area, expected_dynamic):
    labels = np.full((20, 20), _index(21), dtype=np.int16)
    labels[:10, :15] = _index(13)
    if component_area == 149:
        labels[9, 14] = _index(21)
    probabilities = _confident_probabilities(labels)

    _result, report = _apply(labels, probabilities)

    assert report["baseline"]["dynamic_fragments_4_connected"] == expected_dynamic


def _island_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.full((50, 50), _index(21), dtype=np.int16)
    labels[:20, :30] = _index(13)
    labels[30:33, 30:33] = _index(13)
    island = np.argwhere(np.zeros(labels.shape, dtype=bool))
    island_mask = np.zeros(labels.shape, dtype=bool)
    island_mask[30:33, 30:33] = True
    island = np.argwhere(island_mask)
    probabilities = _confident_probabilities(labels)
    _set_target_evidence(
        probabilities,
        labels,
        island,
        target_code=21,
        current_probability=0.34,
        target_probability=0.31,
    )
    confidence = np.full(labels.shape, 0.5, dtype=np.float32)
    confidence[island_mask] = 0.65
    return labels, probabilities, confidence, island_mask


def test_enclosed_island_is_reassigned_as_one_complete_component():
    labels, probabilities, confidence, island = _island_fixture()

    result, report = _apply(labels, probabilities, confidence=confidence)

    assert np.all(result[island] == _index(21))
    assert report["proposals_accepted"] == 1
    accepted = report["accepted"][0]
    assert accepted["kind"] == "enclosed_island"
    assert accepted["changed_pixels"] == 9
    assert accepted["footprint_bbox"] == [30, 30, 32, 32]
    assert accepted["decision"] == "accepted"
    assert accepted["baseline_source_component_ids"]


def test_island_confidence_guard_is_strictly_greater_than_point_65():
    labels, probabilities, confidence, island = _island_fixture()
    confidence[island] = 0.650001

    result, report = _apply(labels, probabilities, confidence=confidence)

    assert np.array_equal(result, labels)
    assert report["proposals_accepted"] == 0


def test_island_requires_one_surrounding_class_and_component():
    labels, probabilities, confidence, island = _island_fixture()
    labels[29, 31] = _index(31)
    probabilities = _confident_probabilities(labels)
    _set_target_evidence(
        probabilities,
        labels,
        np.argwhere(island),
        target_code=21,
        current_probability=0.34,
        target_probability=0.31,
    )

    result, report = _apply(labels, probabilities, confidence=confidence)

    assert np.array_equal(result, labels)
    assert report["proposals_accepted"] == 0


def test_protected_source_cannot_be_an_island_but_keeps_bridge_switch():
    labels = np.full((50, 50), _index(21), dtype=np.int16)
    labels[:20, :30] = _index(12)
    labels[30:32, 30:32] = _index(12)
    probabilities = _confident_probabilities(labels)

    result, report = _apply(labels, probabilities)

    assert np.array_equal(result, labels)
    assert report["proposals_accepted"] == 0
    assert policy_v31a().class_policies[12].allow_same_class_bridge is True


def test_protected_ordinary_target_is_rejected_even_if_semantically_enabled():
    labels, probabilities, confidence, island = _island_fixture()
    labels[:] = _index(71)
    labels[:20, :30] = _index(13)
    labels[30:33, 30:33] = _index(13)
    probabilities = _confident_probabilities(labels)
    _set_target_evidence(
        probabilities,
        labels,
        np.argwhere(island),
        target_code=71,
        current_probability=0.42,
        target_probability=0.40,
    )
    base = policy_v31a()
    compatible = dict(base.semantic_compatible_targets)
    compatible[13] = frozenset({71})
    custom = replace(base, semantic_compatible_targets=MappingProxyType(compatible))

    result, report = _apply(
        labels, probabilities, confidence=confidence, policy=custom
    )

    assert np.array_equal(result, labels)
    assert report["proposals_accepted"] == 0


def _bridge_fixture(
    *, target_code: int = 12, gap: int = 4, sever_source: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    height = 60
    width = 55 + gap
    background = 21 if target_code == 13 else 13
    labels = np.full((height, width), _index(background), dtype=np.int16)
    if sever_source:
        labels[:, :20] = _index(target_code)
        labels[:, 20 + gap :] = _index(target_code)
    else:
        labels[10:50, 5:25] = _index(target_code)
        labels[10:50, 25 + gap : 45 + gap] = _index(target_code)
    probabilities = _confident_probabilities(labels)
    gap_pixels = np.argwhere(labels == _index(background))
    target_policy = policy_v31a().class_policies[target_code]
    target_probability = max(
        target_policy.minimum_target_probability_mean,
        target_policy.minimum_target_probability_p10,
    ) + 0.01
    current_probability = target_probability + min(
        0.03,
        target_policy.maximum_current_minus_target_probability_mean / 2.0,
    )
    _set_target_evidence(
        probabilities,
        labels,
        gap_pixels,
        target_code=target_code,
        current_probability=current_probability,
        target_probability=target_probability,
    )
    return labels, probabilities


def test_bridge_uses_edge_distance_and_connects_two_frozen_components():
    labels, probabilities = _bridge_fixture(gap=6)

    result, report = _apply(labels, probabilities)

    assert report["proposals_accepted"] == 1
    accepted = report["accepted"][0]
    assert accepted["kind"] == "same_class_bridge"
    assert accepted["edge_distance_m"] == 6.0
    assert accepted["changed_pixels"] == 6
    assert report["result"]["components_4_connected"] < report["baseline"][
        "components_4_connected"
    ]
    assert np.count_nonzero(result != labels) == 6


@pytest.mark.parametrize(
    ("target_code", "gap"),
    [(12, 6), (21, 12), (32, 12), (33, 6), (51, 8), (52, 10), (61, 4), (62, 6), (71, 6)],
)
def test_every_enabled_bridge_class_accepts_its_distance_boundary(target_code, gap):
    labels, probabilities = _bridge_fixture(target_code=target_code, gap=gap)

    result, report = _apply(labels, probabilities)

    assert report["proposals_accepted"] == 1
    assert report["accepted"][0]["target_class_code"] == target_code
    assert report["accepted"][0]["edge_distance_m"] == float(gap)
    assert np.count_nonzero(result != labels) == gap


def test_bridge_distance_above_class_limit_is_rejected():
    labels, probabilities = _bridge_fixture(gap=7)

    result, report = _apply(labels, probabilities)

    assert np.array_equal(result, labels)
    assert report["proposals_accepted"] == 0
    assert report["proposal_generation_reject_reason_counts"]["bridge_distance"] >= 1


def test_bridge_footprint_cap_is_a_separate_hard_gate():
    labels, probabilities = _bridge_fixture(gap=4)
    base = policy_v31a()
    rows = dict(base.class_policies)
    rows[12] = replace(rows[12], bridge_max_new_footprint_m2=3.0)
    custom = replace(base, class_policies=MappingProxyType(rows))

    result, report = _apply(labels, probabilities, policy=custom)

    assert np.array_equal(result, labels)
    assert report["proposals_accepted"] == 0
    assert report["proposal_generation_reject_reason_counts"]["bridge_footprint"] >= 1


def test_bridge_cannot_cross_a_protected_source_obstacle():
    labels, probabilities = _bridge_fixture(gap=4)
    labels[10:50, 25:29] = _index(61)
    probabilities = _confident_probabilities(labels)

    result, report = _apply(labels, probabilities)

    assert np.array_equal(result, labels)
    assert report["proposals_accepted"] == 0
    assert report["proposal_generation_reject_reason_counts"]["protected_source"] >= 1


def test_diagonal_bridge_reports_geometric_edge_distance_and_path_separately():
    labels = np.full((20, 20), _index(13), dtype=np.int16)
    labels[5, 5] = _index(12)
    labels[7, 7] = _index(12)
    probabilities = _confident_probabilities(labels)
    background = np.argwhere(labels == _index(13))
    _set_target_evidence(
        probabilities,
        labels,
        background,
        target_code=12,
        current_probability=0.38,
        target_probability=0.36,
    )
    policy = replace(
        policy_v31a(),
        maximum_source_loss_fraction=1.0,
        maximum_target_gain_fraction=1.0,
        protected_bridge_gain_fraction=2.0,
    )

    result, report = _apply(labels, probabilities, policy=policy)

    assert report["proposals_accepted"] == 1
    accepted = report["accepted"][0]
    assert accepted["edge_distance_m"] == pytest.approx(np.sqrt(2.0))
    assert accepted["path_length_m"] == 3.0
    assert np.count_nonzero(result != labels) == 3


@pytest.mark.parametrize("target_code", [13, 31, 43, 53, 54])
def test_disabled_bridge_classes_remain_unchanged(target_code):
    labels, probabilities = _bridge_fixture(target_code=target_code, gap=2)

    result, report = _apply(labels, probabilities)

    assert np.array_equal(result, labels)
    assert report["proposals_accepted"] == 0
    assert report["proposal_generation_reject_reason_counts"][
        "bridge_disabled"
    ] >= 1


def test_bridge_cannot_cut_a_four_connected_source_neck():
    labels = np.full((7, 7), _index(21), dtype=np.int16)
    labels[1:6, 1:3] = _index(13)
    labels[3, 3] = _index(13)
    labels[1:6, 4:6] = _index(13)
    valid = np.ones(labels.shape, dtype=bool)
    component_map, components = candidate_module._component_index(
        labels, valid, CLASS_CODES
    )

    assert candidate_module._source_connectivity_safe(
        np.array([[3, 3]], dtype=np.int32),
        labels,
        component_map,
        components,
        valid,
    ) is False


def test_bridge_probability_mean_and_drop_gates_reject_contradiction():
    labels, probabilities = _bridge_fixture(gap=4)
    changed = np.argwhere(labels == _index(13))
    _set_target_evidence(
        probabilities,
        labels,
        changed,
        target_code=12,
        current_probability=0.46,
        target_probability=0.35,
    )

    result, report = _apply(labels, probabilities)

    assert np.array_equal(result, labels)
    assert report["proposals_accepted"] == 0
    assert report["proposal_generation_reject_reason_counts"][
        "probability_current_minus_target"
    ] >= 1


def test_probability_p10_gate_is_applied_to_the_changed_footprint():
    labels = np.full((10, 10), _index(13), dtype=np.int16)
    footprint = np.argwhere(np.ones(labels.shape, dtype=bool))
    probabilities = _confident_probabilities(labels)
    _set_target_evidence(
        probabilities,
        labels,
        footprint,
        target_code=12,
        current_probability=0.42,
        target_probability=0.40,
    )
    low_pixels = np.argwhere(
        np.indices(labels.shape)[0] < 2
    )
    _set_target_evidence(
        probabilities,
        labels,
        low_pixels,
        target_code=12,
        current_probability=0.23,
        target_probability=0.20,
    )
    row = policy_v31a().class_policies[12]

    allowed, evidence = candidate_module._probability_evidence(
        footprint, labels, _index(12), probabilities, row
    )

    assert allowed is False
    assert evidence["mean_target_probability"] >= row.minimum_target_probability_mean
    assert evidence["p10_target_probability"] < row.minimum_target_probability_p10


def test_adjudication_rejects_exact_cross_target_tie_without_class_priority():
    labels = np.full((5, 5), _index(13), dtype=np.int16)
    valid = np.ones(labels.shape, dtype=bool)
    component_map, components = candidate_module._component_index(
        labels, valid, CLASS_CODES
    )
    footprint = np.array([[2, 2]], dtype=np.int32)
    digest = candidate_module._footprint_digest(footprint)

    def proposal(target_code):
        return candidate_module._Proposal(
            kind="same_class_bridge",
            target_index=_index(target_code),
            target_code=target_code,
            footprint=footprint,
            source_indices=(_index(13),),
            source_codes=(13,),
            source_component_ids=(1,),
            baseline_target_component_ids=(),
            dynamic_reduction=1,
            component_reduction=1,
            probability_support=0.1,
            area_m2=1.0,
            digest=digest,
            proposal_id=f"proposal-{target_code}",
            edge_distance_m=1.0,
            path_length_m=1.0,
            evidence={},
        )

    result, accepted, skipped, decisions, rollback_count = candidate_module._adjudicate(
        [proposal(21), proposal(31)],
        labels,
        CLASS_CODES,
        valid,
        policy_v31a(),
        1.0,
        component_map,
        components,
    )

    assert np.array_equal(result, labels)
    assert accepted == []
    assert skipped == {"ambiguous_target_tie": 2}
    assert rollback_count == 0
    assert {value for value in decisions.values()} == {
        ("rejected", "ambiguous_target_tie")
    }


def test_stable_rank_key_uses_the_approved_complete_hierarchy():
    footprint = np.array([[1, 1]], dtype=np.int32)

    def proposal(
        proposal_id,
        *,
        dynamic=1,
        components=1,
        probability=0.1,
        area=1.0,
        digest=None,
    ):
        return candidate_module._Proposal(
            kind="enclosed_island",
            target_index=_index(21),
            target_code=21,
            footprint=footprint,
            source_indices=(_index(13),),
            source_codes=(13,),
            source_component_ids=(1,),
            baseline_target_component_ids=(2,),
            dynamic_reduction=dynamic,
            component_reduction=components,
            probability_support=probability,
            area_m2=area,
            digest=digest or proposal_id,
            proposal_id=proposal_id,
            edge_distance_m=None,
            path_length_m=None,
            evidence={},
        )

    ordered = sorted(
        [
            proposal("e", digest="f"),
            proposal("d", area=2.0),
            proposal("c", probability=0.2),
            proposal("b", components=2),
            proposal("a", dynamic=2),
        ],
        key=candidate_module._rank_key,
    )

    assert [item.proposal_id for item in ordered] == ["a", "b", "c", "e", "d"]


def test_budget_rejection_is_reported_and_uses_frozen_baseline_denominator():
    labels, probabilities, confidence, _island = _island_fixture()
    base = policy_v31a()
    zero_gain = replace(base, maximum_target_gain_fraction=0.0)

    result, report = _apply(
        labels, probabilities, confidence=confidence, policy=zero_gain
    )

    assert np.array_equal(result, labels)
    assert report["proposal_reject_reason_counts"] == {"target_budget": 1}
    assert report["class_budget_pixels"]["21"]["target_gain_limit"] == 0.0


@pytest.mark.parametrize(
    "class_budget_mask, message",
    [
        (np.ones((2, 2), dtype=bool), "class_budget_mask shape"),
        (np.zeros((50, 50), dtype=bool), "class_budget_mask must contain"),
    ],
)
def test_class_budget_mask_requires_matching_nonempty_valid_core(
    class_budget_mask, message
):
    labels, probabilities, confidence, _island = _island_fixture()

    with pytest.raises(CandidateError, match=message):
        _apply(
            labels,
            probabilities,
            confidence=confidence,
            class_budget_mask=class_budget_mask,
        )


def _mask_class_pixels(
    labels: np.ndarray,
    mask: np.ndarray,
    *,
    code: int,
    count: int,
    required: np.ndarray | None = None,
) -> None:
    required_cells = set() if required is None else {
        (int(row), int(col)) for row, col in required
    }
    assert len(required_cells) <= count
    selected = set(required_cells)
    for row, col in np.argwhere(labels == _index(code)):
        selected.add((int(row), int(col)))
        if len(selected) == count:
            break
    assert len(selected) == count
    rows, cols = np.asarray(sorted(selected), dtype=np.int32).T
    mask[rows, cols] = True


def test_cross_core_proposal_is_rejected_instead_of_being_partially_published():
    labels, probabilities, confidence, island = _island_fixture()
    core = np.zeros(labels.shape, dtype=bool)
    core_island = np.argwhere(island)[:4]
    _mask_class_pixels(labels, core, code=21, count=200)
    _mask_class_pixels(
        labels, core, code=13, count=200, required=core_island
    )

    result, report = _apply(
        labels,
        probabilities,
        confidence=confidence,
        class_budget_mask=core,
    )

    assert np.array_equal(result, labels)
    assert report["proposals_accepted"] == 0
    assert report["proposal_reject_reason_counts"] == {
        "outside_core_owner": 1
    }
    assert report["proposal_audit"][0]["reason"] == "outside_core_owner"
    assert report["changed_pixel_count"] == 0
    assert report["class_budget_mask_pixel_count"] == 400
    assert report["class_budget_pixels"]["13"]["denominator"] == 200
    assert report["class_budget_pixels"]["21"]["denominator"] == 200
    assert report["class_budget_pixels"]["13"]["source_loss"] == 0
    assert report["class_budget_pixels"]["21"]["target_gain"] == 0
    assert report["class_budget_pixels"]["13"]["source_loss_limit"] == 4.0
    assert report["class_budget_pixels"]["21"]["target_gain_limit"] == 4.0


def test_halo_only_proposal_is_rejected_without_reserving_core_budget():
    labels, probabilities, confidence, _island = _island_fixture()
    core = np.zeros(labels.shape, dtype=bool)
    _mask_class_pixels(labels, core, code=21, count=200)
    _mask_class_pixels(labels, core, code=13, count=200)

    result, report = _apply(
        labels,
        probabilities,
        confidence=confidence,
        class_budget_mask=core,
    )

    assert np.array_equal(result, labels)
    assert report["proposals_accepted"] == 0
    assert report["proposal_reject_reason_counts"] == {
        "outside_core_owner": 1
    }
    assert all(
        values["source_loss"] == 0 and values["target_gain"] == 0
        for values in report["class_budget_pixels"].values()
    )


def test_core_budget_exact_protected_bridge_one_percent_is_allowed():
    labels = np.full((50, 40), _index(13), dtype=np.int16)
    labels[10:40, 3:13] = _index(12)
    labels[10:40, 19:29] = _index(12)
    probabilities = _confident_probabilities(labels)
    background = np.argwhere(labels == _index(13))
    _set_target_evidence(
        probabilities,
        labels,
        background,
        target_code=12,
        current_probability=0.38,
        target_probability=0.36,
    )
    full_result, _full_report = _apply(labels, probabilities)
    bridge = np.argwhere(full_result != labels)
    assert len(bridge) == 6
    core = np.zeros(labels.shape, dtype=bool)
    _mask_class_pixels(labels, core, code=12, count=600)
    _mask_class_pixels(labels, core, code=13, count=600, required=bridge)

    result, report = _apply(
        labels, probabilities, class_budget_mask=core
    )

    assert np.array_equal(result, full_result)
    budget = report["class_budget_pixels"]["12"]
    assert budget["denominator"] == 600
    assert budget["target_gain"] == 6
    assert budget["protected_bridge_gain"] == 6
    assert budget["target_gain_limit"] == 6.0
    assert budget["protected_bridge_gain_limit"] == 6.0


def test_core_budget_mask_default_is_valid_and_its_audit_is_deterministic():
    labels, probabilities, confidence, _island = _island_fixture()
    valid = np.ones(labels.shape, dtype=bool)
    valid[0, 0] = False
    labels[0, 0] = -1

    implicit_result, implicit_report = _apply(
        labels, probabilities, confidence=confidence, valid_mask=valid
    )
    explicit_result, explicit_report = _apply(
        labels,
        probabilities,
        confidence=confidence,
        valid_mask=valid,
        class_budget_mask=valid,
    )
    repeated_result, repeated_report = _apply(
        labels,
        probabilities,
        confidence=confidence,
        valid_mask=valid,
        class_budget_mask=valid,
    )

    assert np.array_equal(implicit_result, explicit_result)
    assert implicit_report == explicit_report
    assert np.array_equal(explicit_result, repeated_result)
    assert explicit_report == repeated_report
    assert implicit_report["class_budget_mask_sha256"] == implicit_report[
        "valid_mask_sha256"
    ]
    assert implicit_report["class_budget_mask_pixel_count"] == int(valid.sum())


def test_whole_proposal_on_core_boundary_is_published_at_exact_two_percent():
    labels = np.full((40, 40), _index(13), dtype=np.int16)
    labels[5:22, 5:32] = _index(21)
    labels[10:13, 10:13] = _index(13)
    island = np.zeros(labels.shape, dtype=bool)
    island[10:13, 10:13] = True
    probabilities = _confident_probabilities(labels)
    _set_target_evidence(
        probabilities,
        labels,
        np.argwhere(island),
        target_code=21,
        current_probability=0.34,
        target_probability=0.31,
    )
    confidence = np.full(labels.shape, 0.5, dtype=np.float32)
    core = np.zeros(labels.shape, dtype=bool)
    _mask_class_pixels(labels, core, code=21, count=450)
    _mask_class_pixels(
        labels, core, code=13, count=450, required=np.argwhere(island)
    )

    result, report = _apply(
        labels,
        probabilities,
        confidence=confidence,
        class_budget_mask=core,
    )

    assert np.all(result[island] == _index(21))
    assert report["proposals_accepted"] == 1
    assert report["proposal_reject_reason_counts"] == {}
    assert report["changed_pixel_count"] == 9
    assert np.array_equal(result[~core], labels[~core])
    assert report["class_budget_pixels"]["21"]["baseline"] == 450
    assert report["class_budget_pixels"]["21"]["target_gain"] == 9
    assert report["class_budget_pixels"]["21"]["target_gain_limit"] == 9.0
    assert report["class_budget_pixels"]["13"]["denominator"] == 450
    assert report["class_budget_pixels"]["13"]["source_loss"] == 9
    assert report["class_budget_pixels"]["13"]["source_loss_limit"] == 9.0


def test_protected_bridge_gain_equal_to_one_percent_is_allowed():
    labels = np.full((50, 40), _index(13), dtype=np.int16)
    labels[10:40, 3:13] = _index(12)
    labels[10:40, 19:29] = _index(12)
    probabilities = _confident_probabilities(labels)
    background = np.argwhere(labels == _index(13))
    _set_target_evidence(
        probabilities,
        labels,
        background,
        target_code=12,
        current_probability=0.38,
        target_probability=0.36,
    )

    result, report = _apply(labels, probabilities)

    assert np.count_nonzero(result != labels) == 6
    assert report["class_budget_pixels"]["12"]["baseline"] == 600
    assert report["class_budget_pixels"]["12"]["target_gain"] == 6
    assert report["class_budget_pixels"]["12"]["target_gain_limit"] == 6.0


def test_cumulative_adjudication_cannot_undo_prior_target_topology_gain():
    class_codes = [13, 21, 32]
    labels = np.zeros((10, 10), dtype=np.int16)
    labels[5, 1:3] = 1
    labels[5, 6:8] = 1
    labels[4, 2] = 2
    labels[6, 2] = 2
    labels[4, 6] = 2
    labels[6, 6] = 2
    valid = np.ones(labels.shape, dtype=bool)
    component_map, components = candidate_module._component_index(
        labels, valid, class_codes
    )
    policy = replace(
        candidate_module.v31a_policy(),
        maximum_source_loss_fraction=1.0,
        maximum_target_gain_fraction=1.0,
    )

    def component_ids(cells):
        return tuple(sorted({int(component_map[row, col]) for row, col in cells}))

    def proposal(proposal_id, target_index, footprint_values, target_cells):
        footprint = np.asarray(footprint_values, dtype=np.int32)
        rows, cols = footprint[:, 0], footprint[:, 1]
        source_indices = tuple(sorted(set(int(v) for v in labels[rows, cols])))
        return candidate_module._Proposal(
            kind="same_class_bridge",
            target_index=target_index,
            target_code=class_codes[target_index],
            footprint=footprint,
            source_indices=source_indices,
            source_codes=tuple(class_codes[index] for index in source_indices),
            source_component_ids=component_ids(footprint_values),
            baseline_target_component_ids=component_ids(target_cells),
            dynamic_reduction=0,
            component_reduction=1,
            probability_support=0.1,
            area_m2=float(len(footprint)),
            digest=candidate_module._footprint_digest(footprint),
            proposal_id=proposal_id,
            edge_distance_m=1.0,
            path_length_m=float(len(footprint)),
            evidence={},
        )

    proposals = [
        proposal("p1", 1, [[5, 3], [5, 4], [5, 5]], [[5, 1], [5, 6]]),
        proposal("p2", 2, [[5, 2]], [[4, 2], [6, 2]]),
        proposal("p3", 2, [[5, 6]], [[4, 6], [6, 6]]),
    ]

    result, accepted, skipped, decisions, rollback_count = (
        candidate_module._adjudicate(
            proposals,
            labels,
            class_codes,
            valid,
            policy,
            1.0,
            component_map,
            components,
        )
    )
    after = candidate_module._per_class_metrics(
        result, valid, class_codes, policy, 1.0
    )

    assert after[21]["component_count_4_connected"] <= 2
    assert len(accepted) == 2
    assert rollback_count == 1
    assert skipped["final_topology_rollback"] == 1
    assert ("rejected", "final_topology_rollback") in decisions.values()


def test_v31b_rejects_target_attachment_cause_and_preserves_later_unrelated_proposal():
    """B fixes the A LIFO failure mode without changing A's implementation."""

    class_codes = [13, 21, 32]
    labels = np.zeros((10, 10), dtype=np.int16)
    # Two 21 anchors joined by p1.  p2 later removes p1's left anchor; p3 is
    # unrelated and deliberately ranked last, so A rolls p3 back before p2.
    labels[5, 1] = 1
    labels[5, 5] = 1
    labels[8, 8] = 1
    labels[4, 1] = 2
    labels[6, 1] = 2
    labels[2, 1] = 2
    valid = np.ones(labels.shape, dtype=bool)
    component_map, components = candidate_module._component_index(labels, valid, class_codes)
    policy = replace(
        candidate_module.v31b_policy(),
        maximum_source_loss_fraction=1.0,
        maximum_target_gain_fraction=1.0,
    )

    def ids(cells):
        return tuple(sorted({int(component_map[row, col]) for row, col in cells}))

    def proposal(name, target_index, footprint_values, target_cells, reduction, support):
        footprint = np.asarray(footprint_values, dtype=np.int32)
        rows, cols = footprint[:, 0], footprint[:, 1]
        source_indices = tuple(sorted(set(int(value) for value in labels[rows, cols])))
        return candidate_module._Proposal(
            kind="same_class_bridge" if name == "p1" else "enclosed_island",
            target_index=target_index,
            target_code=class_codes[target_index],
            footprint=footprint,
            source_indices=source_indices,
            source_codes=tuple(class_codes[index] for index in source_indices),
            source_component_ids=ids(footprint_values),
            baseline_target_component_ids=ids(target_cells),
            dynamic_reduction=reduction,
            component_reduction=reduction,
            probability_support=support,
            area_m2=float(len(footprint)),
            digest=candidate_module._footprint_digest(footprint),
            proposal_id=name,
            edge_distance_m=1.0,
            path_length_m=float(len(footprint)),
            evidence={},
        )

    p1 = proposal("p1", 1, [[5, 2], [5, 3], [5, 4]], [[5, 1], [5, 5]], 2, .3)
    p2 = proposal("p2", 2, [[5, 1]], [[4, 1], [6, 1]], 1, .2)
    p3 = proposal("p3", 2, [[2, 2]], [[2, 1]], 0, .1)
    proposals = [p1, p2, p3]

    _a_result, a_accepted, a_skipped, _a_decisions, a_rollbacks = candidate_module._adjudicate(
        proposals, labels, class_codes, valid, policy, 1.0, component_map, components,
    )
    b_result, b_accepted, b_skipped, b_decisions, b_rollbacks, interactions, duplicates = candidate_module._adjudicate_v31b(
        proposals, labels, class_codes, valid, policy, 1.0, component_map, components,
    )

    assert a_rollbacks == 2
    assert a_skipped["final_topology_rollback"] == 2
    assert [item.proposal_id for item in a_accepted] == ["p1"]
    assert b_rollbacks == 0
    assert duplicates == 0
    assert [item.proposal_id for item in b_accepted] == ["p1", "p3"]
    assert b_skipped["target_attachment"] == 1
    assert b_decisions["p2"] == ("rejected", "target_attachment")
    assert b_result[2, 2] == 2
    p2_interaction = next(item for item in interactions if item["proposal_id"] == "p2")
    assert p2_interaction["affected_accepted_proposal_ids"] == ["p1"]
    assert p2_interaction["target_attachment_checks"] == 2


def test_v31b_canonicalizes_duplicate_candidates_with_complete_audit():
    class_codes = [13, 21, 32]
    labels = np.zeros((6, 6), dtype=np.int16)
    labels[2, 1] = 2
    valid = np.ones(labels.shape, dtype=bool)
    component_map, components = candidate_module._component_index(labels, valid, class_codes)
    footprint = np.array([[2, 2]], dtype=np.int32)
    duplicate = candidate_module._Proposal(
        kind="enclosed_island",
        target_index=2,
        target_code=32,
        footprint=footprint,
        source_indices=(0,), source_codes=(13,),
        source_component_ids=(int(component_map[2, 2]),),
        baseline_target_component_ids=(int(component_map[2, 1]),),
        dynamic_reduction=1, component_reduction=0,
        probability_support=.2, area_m2=1.0,
        digest=candidate_module._footprint_digest(footprint), proposal_id="same",
        edge_distance_m=None, path_length_m=None, evidence={},
    )
    policy = replace(
        candidate_module.v31b_policy(),
        maximum_source_loss_fraction=1.0,
        maximum_target_gain_fraction=1.0,
    )
    _result, accepted, skipped, decisions, rollbacks, interactions, duplicates = candidate_module._adjudicate_v31b(
        [duplicate, duplicate], labels, class_codes, valid, policy, 1.0, component_map, components,
    )

    assert [item.proposal_id for item in accepted] == ["same"]
    assert skipped == {"duplicate_proposal": 1}
    assert decisions["same"] == ("accepted", "selected")
    assert rollbacks == 0
    assert duplicates == 1
    duplicate_rows = [item for item in interactions if item["reason"] == "duplicate_proposal"]
    assert duplicate_rows == [{
        "occurrence_id": "same:duplicate:1", "proposal_id": "same", "canonical_proposal_id": "same",
        "decision": "rejected", "reason": "duplicate_proposal",
        "stable_rank_key": list(candidate_module._rank_key(duplicate)),
        "footprint_sha256": duplicate.digest,
        "discovery_count": 1, "edge_distance_m": None, "path_length_m": None,
        "occurrence_edge_distance_m": None, "occurrence_path_length_m": None,
        "canonical_edge_distance_m": None, "canonical_path_length_m": None,
    }]

    with pytest.raises(CandidateError, match="inconsistent rank or evidence"):
        candidate_module._canonicalize_v31b_proposals([
            duplicate, replace(duplicate, probability_support=.3),
        ])

    alternate_id = replace(duplicate, proposal_id="same-alternate")
    canonical, duplicate_counts, alternate_audit = candidate_module._canonicalize_v31b_proposals(
        [alternate_id, duplicate]
    )
    reversed_canonical, reversed_counts, reversed_audit = candidate_module._canonicalize_v31b_proposals(
        [duplicate, alternate_id]
    )
    assert [item.proposal_id for item in canonical] == ["same"]
    assert [item.proposal_id for item in reversed_canonical] == ["same"]
    assert duplicate_counts == reversed_counts == {"duplicate_proposal": 1}
    assert alternate_audit == reversed_audit == [{
        "occurrence_id": "same-alternate:duplicate:1",
        "proposal_id": "same-alternate", "canonical_proposal_id": "same",
        "decision": "rejected", "reason": "duplicate_proposal",
        "stable_rank_key": list(candidate_module._rank_key(duplicate)),
        "footprint_sha256": duplicate.digest,
        "discovery_count": 1, "edge_distance_m": None, "path_length_m": None,
        "occurrence_edge_distance_m": None, "occurrence_path_length_m": None,
        "canonical_edge_distance_m": None, "canonical_path_length_m": None,
    }]


def test_v31b_metrics_union_same_class_proposal_contacts_without_false_rejection():
    class_codes = [13, 21]
    labels = np.zeros((6, 7), dtype=np.int16)
    labels[2, 1] = 1
    labels[2, 4] = 1
    valid = np.ones(labels.shape, dtype=bool)
    component_map, components = candidate_module._component_index(labels, valid, class_codes)
    policy = replace(
        candidate_module.v31b_policy(),
        maximum_source_loss_fraction=1.0,
        maximum_target_gain_fraction=1.0,
    )

    def proposal(name, footprint_value, target_cell, support):
        footprint = np.asarray([footprint_value], dtype=np.int32)
        row, col = footprint_value
        return candidate_module._Proposal(
            kind="enclosed_island", target_index=1, target_code=21,
            footprint=footprint, source_indices=(0,), source_codes=(13,),
            source_component_ids=(int(component_map[row, col]),),
            baseline_target_component_ids=(int(component_map[target_cell]),),
            dynamic_reduction=0, component_reduction=0,
            probability_support=support, area_m2=1.0,
            digest=candidate_module._footprint_digest(footprint), proposal_id=name,
            edge_distance_m=None, path_length_m=None, evidence={},
        )

    first = proposal("first", (2, 2), (2, 1), .2)
    second = proposal("second", (2, 3), (2, 4), .1)
    result, accepted, skipped, decisions, rollback, interactions, duplicates = candidate_module._adjudicate_v31b(
        [first, second], labels, class_codes, valid, policy, 1.0, component_map, components,
    )
    reversed_result, reversed_accepted, reversed_skipped, reversed_decisions, reversed_rollback, reversed_interactions, reversed_duplicates = candidate_module._adjudicate_v31b(
        [second, first], labels, class_codes, valid, policy, 1.0, component_map, components,
    )

    assert [item.proposal_id for item in accepted] == ["first", "second"]
    assert skipped == {}
    assert decisions == {"first": ("accepted", "selected"), "second": ("accepted", "selected")}
    assert rollback == 0
    assert duplicates == 0
    assert result[2, 1:5].tolist() == [1, 1, 1, 1]
    assert np.array_equal(result, reversed_result)
    assert [item.proposal_id for item in reversed_accepted] == ["first", "second"]
    assert reversed_skipped == skipped
    assert reversed_decisions == decisions
    assert reversed_rollback == rollback
    assert reversed_interactions == interactions
    assert reversed_duplicates == duplicates


def test_v31b_same_action_multi_target_discovery_merges_distance_metadata_deterministically(monkeypatch):
    labels = np.full((9, 9), _index(13), dtype=np.int16)
    labels[4, 4] = _index(21)
    probabilities = _confident_probabilities(labels)
    valid = np.ones(labels.shape, dtype=bool)
    component_map, _components = candidate_module._component_index(labels, valid, CLASS_CODES)
    footprint = np.asarray([[4, 5]], dtype=np.int32)

    def discovery(edge_distance):
        return candidate_module._Proposal(
            kind="same_class_bridge", target_index=_index(21), target_code=21,
            footprint=footprint, source_indices=(_index(13),), source_codes=(13,),
            source_component_ids=(int(component_map[4, 5]),),
            baseline_target_component_ids=(int(component_map[4, 4]),),
            dynamic_reduction=0, component_reduction=0, probability_support=.2,
            area_m2=1.0, digest=candidate_module._footprint_digest(footprint),
            proposal_id="multi-target-discovery", edge_distance_m=edge_distance,
            path_length_m=1.0, evidence={},
        )

    near = discovery(0.0)
    far = discovery(0.792356989)
    candidates = [far, near]
    monkeypatch.setattr(candidate_module, "_island_proposals", lambda *_args: list(candidates))
    monkeypatch.setattr(candidate_module, "_bridge_proposals_for_code", lambda *_args: [])
    policy = replace(candidate_module.v31b_policy(), maximum_target_gain_fraction=1.0)
    first_result, first_report = _apply_b(labels, probabilities, policy=policy, full_audit=True)
    candidates[:] = [near, far]
    second_result, second_report = _apply_b(labels, probabilities, policy=policy, full_audit=True)

    assert np.array_equal(first_result, second_result)
    assert first_report["audit_sha256"] == second_report["audit_sha256"]
    accepted = first_report["accepted"][0]
    assert accepted["edge_distance_m"] == 0.0
    assert accepted["path_length_m"] == 1.0
    assert accepted["discovery_count"] == 2
    assert accepted["discovery_edge_distances_m"] == [0.0, 0.792356989]
    raw = first_report["raw_proposal_audit"]
    assert [row["occurrence_edge_distance_m"] for row in raw] == [0.0, 0.792356989]
    assert raw[1]["canonical_edge_distance_m"] == 0.0


def test_v31b_island_attachment_allows_indirect_accepted_proposal_route_after_anchor_edge_loss():
    class_codes = [13, 21, 32]
    labels = np.zeros((8, 8), dtype=np.int16)
    # A is a two-cell 21 residual. q initially touches A at (3, 1); r touches
    # both q and A at (2, 1). p changes only q's old A contact to 32, leaving
    # q -- r -- A as the exact residual/proposal graph route.
    labels[2, 1] = 1
    labels[3, 1] = 1
    labels[3, 0] = 2
    valid = np.ones(labels.shape, dtype=bool)
    component_map, components = candidate_module._component_index(labels, valid, class_codes)
    policy = replace(
        candidate_module.v31b_policy(),
        maximum_source_loss_fraction=1.0,
        maximum_target_gain_fraction=1.0,
    )

    def proposal(name, target_index, footprint_value, target_cell, reduction):
        footprint = np.asarray([footprint_value], dtype=np.int32)
        row, col = footprint_value
        source_index = int(labels[row, col])
        return candidate_module._Proposal(
            kind="enclosed_island", target_index=target_index,
            target_code=class_codes[target_index], footprint=footprint,
            source_indices=(source_index,), source_codes=(class_codes[source_index],),
            source_component_ids=(int(component_map[row, col]),),
            baseline_target_component_ids=(int(component_map[target_cell]),),
            dynamic_reduction=reduction, component_reduction=reduction,
            probability_support=float(reduction) / 10.0, area_m2=1.0,
            digest=candidate_module._footprint_digest(footprint), proposal_id=name,
            edge_distance_m=None, path_length_m=None, evidence={},
        )

    q = proposal("q", 1, (3, 2), (3, 1), 3)
    r = proposal("r", 1, (2, 2), (2, 1), 2)
    p = proposal("p", 2, (3, 1), (3, 0), 1)
    proposals = [q, r, p]
    result, accepted, skipped, decisions, rollbacks, interactions, duplicates = candidate_module._adjudicate_v31b(
        proposals, labels, class_codes, valid, policy, 1.0, component_map, components,
    )
    reversed_result, reversed_accepted, reversed_skipped, reversed_decisions, reversed_rollbacks, reversed_interactions, reversed_duplicates = candidate_module._adjudicate_v31b(
        list(reversed(proposals)), labels, class_codes, valid, policy, 1.0, component_map, components,
    )

    assert [item.proposal_id for item in accepted] == ["q", "r", "p"]
    assert skipped == {}
    assert decisions == {
        "q": ("accepted", "selected"), "r": ("accepted", "selected"), "p": ("accepted", "selected"),
    }
    assert rollbacks == duplicates == 0
    assert candidate_module._final_topology_holds(
        labels, result, valid, class_codes, policy, 1.0, accepted, components,
    ) is True
    p_interaction = next(item for item in interactions if item["proposal_id"] == "p")
    assert p_interaction["affected_accepted_proposal_ids"] == ["q", "r"]
    assert np.array_equal(result, reversed_result)
    assert [item.proposal_id for item in reversed_accepted] == ["q", "r", "p"]
    assert reversed_skipped == skipped
    assert reversed_decisions == decisions
    assert reversed_rollbacks == rollbacks
    assert reversed_interactions == interactions
    assert reversed_duplicates == duplicates


def test_v31b_full_audit_keeps_all_raw_duplicate_occurrences(monkeypatch):
    labels = np.full((9, 9), _index(13), dtype=np.int16)
    labels[4, 4] = _index(21)
    probabilities = _confident_probabilities(labels)
    valid = np.ones(labels.shape, dtype=bool)
    component_map, _components = candidate_module._component_index(labels, valid, CLASS_CODES)
    footprint = np.asarray([[4, 5]], dtype=np.int32)
    duplicate = candidate_module._Proposal(
        kind="enclosed_island", target_index=_index(21), target_code=21,
        footprint=footprint, source_indices=(_index(13),), source_codes=(13,),
        source_component_ids=(int(component_map[4, 5]),),
        baseline_target_component_ids=(int(component_map[4, 4]),),
        dynamic_reduction=0, component_reduction=0, probability_support=.2,
        area_m2=1.0, digest=candidate_module._footprint_digest(footprint),
        proposal_id="duplicate-audit", edge_distance_m=None, path_length_m=None, evidence={},
    )
    monkeypatch.setattr(candidate_module, "_island_proposals", lambda *_args: [duplicate, duplicate])
    monkeypatch.setattr(candidate_module, "_bridge_proposals_for_code", lambda *_args: [])
    policy = replace(candidate_module.v31b_policy(), maximum_target_gain_fraction=1.0)
    _result, report = _apply_b(labels, probabilities, policy=policy, full_audit=True)

    assert report["raw_generated"] == report["proposals_canonical"] + report["duplicate_proposal_count"] == 2
    assert len(report["raw_proposal_audit"]) == 2
    assert len(report["duplicate_proposal_audit"]) == 1
    assert {item["occurrence_id"] for item in report["raw_proposal_audit"]} == {
        "duplicate-audit:canonical", "duplicate-audit:duplicate:1",
    }
    nonfull_policy = replace(policy, audit_proposal_limit=1)
    _nonfull_result, nonfull_report = _apply_b(
        labels, probabilities, policy=nonfull_policy, full_audit=False,
    )
    assert nonfull_report["audit_truncated"] is True
    assert nonfull_report["raw_generated"] == 2
    assert len(nonfull_report["raw_proposal_audit"]) == 1


def test_v31b_output_is_deterministic_has_no_rollback_and_preserves_contracts():
    labels, probabilities, confidence, island = _island_fixture()
    first_result, first_report = _apply_b(
        labels, probabilities, confidence=confidence, full_audit=True,
    )
    second_result, second_report = _apply_b(
        labels, probabilities, confidence=confidence, full_audit=True,
    )

    assert np.array_equal(first_result, second_result)
    assert first_report == second_report
    assert first_report["adjudication_mode"] == "dependency_incremental_v1"
    assert first_report["final_topology_rollback"] == 0
    assert first_report["raw_generated"] == (
        first_report["proposals_canonical"] + first_report["duplicate_proposal_count"]
    )
    assert first_report["single_label"] is True
    assert first_report["gap_pixels"] == 0
    assert first_report["overlap_pixels"] == 0
    assert first_report["outside_pixels"] == 0
    assert np.all(first_result[island] == _index(21))


def test_final_topology_rejects_an_island_detached_from_its_frozen_target():
    class_codes = [13, 21, 32]
    labels = np.zeros((10, 10), dtype=np.int16)
    labels[2, 1] = 2
    labels[2, 2] = 1
    labels[7, 2] = 1
    labels[7, 7] = 1
    valid = np.ones(labels.shape, dtype=bool)
    component_map, components = candidate_module._component_index(
        labels, valid, class_codes
    )
    target_component_id = int(component_map[2, 2])
    footprint = np.array([[2, 3]], dtype=np.int32)
    proposal = candidate_module._Proposal(
        kind="enclosed_island",
        target_index=1,
        target_code=21,
        footprint=footprint,
        source_indices=(0,),
        source_codes=(13,),
        source_component_ids=(int(component_map[2, 3]),),
        baseline_target_component_ids=(target_component_id,),
        dynamic_reduction=1,
        component_reduction=1,
        probability_support=0.1,
        area_m2=1.0,
        digest=candidate_module._footprint_digest(footprint),
        proposal_id="detached-island",
        edge_distance_m=None,
        path_length_m=None,
        evidence={},
    )
    result = labels.copy()
    result[2, 3] = 1
    result[2, 2] = 2
    result[7, 2:8] = 1

    assert candidate_module._final_topology_holds(
        labels,
        result,
        valid,
        class_codes,
        candidate_module.v31a_policy(),
        1.0,
        [proposal],
        components,
    ) is False


def test_full_audit_returns_every_eligible_proposal_without_truncation():
    labels, probabilities, confidence, _island = _island_fixture()

    _result, report = _apply(
        labels,
        probabilities,
        confidence=confidence,
        full_audit=True,
    )

    assert report["full_audit"] is True
    assert report["audit_truncated"] is False
    assert len(report["proposal_audit"]) == report["proposals_generated"]
    assert report["proposal_generation_reject_reason_counts"]


def test_invalid_pixels_are_unchanged_and_candidate_report_is_deterministic():
    labels, probabilities, confidence, island = _island_fixture()
    valid = np.ones(labels.shape, dtype=bool)
    valid[0, 0] = False
    labels[0, 0] = -1

    first_result, first_report = _apply(
        labels, probabilities, confidence=confidence, valid_mask=valid
    )
    second_result, second_report = _apply(
        labels, probabilities, confidence=confidence, valid_mask=valid
    )

    assert first_result[0, 0] == -1
    assert np.array_equal(first_result, second_result)
    assert first_report == second_report
    assert first_report["baseline_kind"] == "v3_cleaned"
    assert first_report["single_label"] is True
    assert first_report["gap_pixels"] == 0
    assert first_report["overlap_pixels"] == 0
    assert first_report["outside_pixels"] == 0
    assert first_report["policy_snapshot_sha256"] == policy_snapshot_sha256()
    assert np.all(first_result[island] == _index(21))
