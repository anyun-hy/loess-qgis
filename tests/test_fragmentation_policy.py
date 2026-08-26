from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import pytest

from fragmentation_policy import (
    PolicyError,
    audit_legacy_migration,
    explain_fragment_decision,
    load_policy,
    policy_sha256,
    rank_conflicting_proposals,
)
from fragmentation_policy.loader import diff_policies


POLICY_PATH = Path(__file__).parents[1] / "inference_scripts" / "fragmentation_policy" / "policies" / "v33_draft.yaml"


def test_v33_draft_loads_all_classes_and_has_stable_sha():
    policy = load_policy(POLICY_PATH)
    assert sorted(map(int, policy["classes"])) == [12, 13, 21, 31, 32, 33, 43, 51, 52, 53, 54, 61, 62, 71]
    assert policy_sha256(policy) == policy_sha256(load_policy(POLICY_PATH))


def test_protection_is_split_from_growth_and_transport_has_overlay_obligation():
    policy = load_policy(POLICY_PATH)
    assert policy["classes"]["52"] == {
        "code": 52, "name": "农村建设用地", "source_absorption": "deny", "target_growth": "allow",
        "same_class_bridge": "allow", "fragment_max_m2": 60, "relation_profile": "policy_rules",
        "post_cleanup_obligation": "none",
    }
    for code in ("61", "62"):
        assert policy["classes"][code]["source_absorption"] == "allow"
        assert policy["classes"][code]["target_growth"] == "deny"
        assert policy["classes"][code]["post_cleanup_obligation"] == "manual_transport_vector_overlay"


def test_unique_and_multi_rules_follow_confirmed_13_and_rarity_behavior():
    policy = load_policy(POLICY_PATH)
    unique = explain_fragment_decision(policy, source_code=13, surrounding_codes=[71], area_m2=149.9, scenario="unique_enclosure", source_component_complete=True, boundary_evidence_complete=True, boundary_status="closed_valid")
    assert unique["decision"] == "PERMIT"
    assert unique["selected_target_class_code"] == 71
    multi = explain_fragment_decision(policy, source_code=13, surrounding_codes=[31, 43], area_m2=40, scenario="multi_neighbour", source_component_complete=True, boundary_evidence_complete=True, boundary_status="closed_valid", bridge_candidates_checked=True)
    assert multi["decision"] == "PERMIT"
    assert multi["selected_target_class_code"] == 43  # 20.02% is rarer than 21.96%
    assert [row["class_code"] for row in multi["target_ranking"]] == [43, 31]


def test_protected_source_and_transport_obligation_are_explained():
    policy = load_policy(POLICY_PATH)
    protected = explain_fragment_decision(policy, source_code=52, surrounding_codes=[43], area_m2=1, scenario="unique_enclosure", boundary_status="closed_valid")
    assert protected["decision"] == "DENY"
    assert protected["trace"] == ["source_permission: source_absorption=deny"]
    transport = explain_fragment_decision(policy, source_code=61, surrounding_codes=[52], area_m2=29, scenario="unique_enclosure", source_component_complete=True, boundary_evidence_complete=True, boundary_status="closed_valid")
    assert transport["decision"] == "PERMIT_WITH_OBLIGATIONS"
    assert transport["selected_target_class_code"] == 52
    assert transport["obligations"][0]["kind"] == "manual_transport_vector_overlay"
    assert transport["obligations"][0]["artifact_sha256"] == "required_at_runtime"
    bridge = explain_fragment_decision(policy, source_code=13, surrounding_codes=[43, 52], area_m2=10, scenario="same_class_bridge", source_component_complete=True, boundary_evidence_complete=True, boundary_status="closed_valid", bridge_target_code=52, source_component_id=7, target_component_ids=[101, 102], edge_distance_m=10, bridge_footprint_m2=10)
    assert bridge["decision"] == "PERMIT"
    assert bridge["selected_target_class_code"] == 52


def test_relation_deny_overrides_a_broader_allow_even_at_lower_priority():
    policy = deepcopy(load_policy(POLICY_PATH))
    policy["decision_engine"]["relation_rules"].append(
        {"id": "deny_13_to_43_for_review", "effect": "deny", "source": "13", "target": "43", "scenario": "*", "priority": 1, "specificity": 2}
    )
    denied = explain_fragment_decision(policy, source_code=13, surrounding_codes=[43], area_m2=10, scenario="unique_enclosure", source_component_complete=True, boundary_evidence_complete=True, boundary_status="closed_valid")
    assert denied["decision"] == "DENY"
    assert "relation_permission: 13->43=deny" in denied["trace"]
    audit = denied["target_rule_audit"]["43"]
    assert audit["selected_specificity"] == 2
    assert audit["effective_rules"][0]["id"] == "deny_13_to_43_for_review"


def test_relation_specificity_then_priority_and_equal_rank_deny_are_real():
    policy = deepcopy(load_policy(POLICY_PATH))
    policy["decision_engine"]["relation_rules"].extend([
        {"id": "broad_high_priority_deny", "effect": "deny", "source": "*", "target": "*", "scenario": "*", "priority": 9999, "specificity": 0},
        {"id": "target_43_equal_rank_deny", "effect": "deny", "source": "*", "target": "43", "scenario": "*", "priority": 300, "specificity": 1},
    ])
    denied = explain_fragment_decision(policy, source_code=13, surrounding_codes=[43], area_m2=10, scenario="unique_enclosure", source_component_complete=True, boundary_evidence_complete=True, boundary_status="closed_valid")
    assert denied["decision"] == "DENY"
    audit = denied["target_rule_audit"]["43"]
    assert {rule["id"] for rule in audit["effective_rules"]} == {
        "allow_dryland_to_any_surrounding", "target_43_equal_rank_deny",
    }
    assert audit["reason"] == "equal_rank_deny_overrides"


@pytest.mark.parametrize("area", [0, -1, math.nan, math.inf])
def test_decision_rejects_nonpositive_or_nonfinite_area(area: float):
    with pytest.raises(PolicyError, match="finite and greater than zero"):
        explain_fragment_decision(load_policy(POLICY_PATH), source_code=13, surrounding_codes=[43], area_m2=area, scenario="unique_enclosure", source_component_complete=True, boundary_evidence_complete=True, boundary_status="closed_valid")


def test_complete_component_and_strict_threshold_are_required():
    policy = load_policy(POLICY_PATH)
    with pytest.raises(PolicyError, match="complete"):
        explain_fragment_decision(policy, source_code=13, surrounding_codes=[43], area_m2=10, scenario="unique_enclosure", boundary_status="closed_valid")
    at_limit = explain_fragment_decision(policy, source_code=13, surrounding_codes=[43], area_m2=150, scenario="unique_enclosure", source_component_complete=True, boundary_evidence_complete=True, boundary_status="closed_valid")
    assert at_limit["decision"] == "DENY"
    assert at_limit["trace"] == ["area_gate: source_fragment_must_be_strictly_below_class_limit"]


def test_bridge_requires_two_targets_complete_source_and_active_caps():
    policy = load_policy(POLICY_PATH)
    common = dict(source_code=13, surrounding_codes=[43, 52], area_m2=10, scenario="same_class_bridge", source_component_complete=True, boundary_evidence_complete=True, boundary_status="closed_valid", bridge_target_code=52, source_component_id=7, edge_distance_m=10, bridge_footprint_m2=10)
    with pytest.raises(PolicyError, match="at least two"):
        explain_fragment_decision(policy, target_component_ids=[101], **common)
    too_far = explain_fragment_decision(policy, target_component_ids=[101, 102], **{**common, "edge_distance_m": 10.1})
    assert too_far["decision"] == "DENY"
    partial = explain_fragment_decision(policy, target_component_ids=[101, 102], **{**common, "bridge_footprint_m2": 9})
    assert partial["decision"] == "DENY"
    assert partial["trace"][-1] == "bridge_gate: footprint_must_equal_complete_source_component_area"


def test_scenario_router_preserves_one_class_direct_absorption_and_prevents_manual_bypass():
    policy = load_policy(POLICY_PATH)
    # The approved rule is based on one distinct surrounding class, not the
    # number of components belonging to that class.  Therefore this remains a
    # direct enclosure decision even though it would connect two 52 pieces.
    unique = explain_fragment_decision(
        policy, source_code=13, surrounding_codes=[52], area_m2=130,
        scenario="unique_enclosure", source_component_complete=True,
        boundary_evidence_complete=True, boundary_status="closed_valid",
        target_component_ids=[101, 102],
    )
    assert unique["decision"] == "PERMIT"
    with pytest.raises(PolicyError, match="scenario must be unique_enclosure"):
        explain_fragment_decision(
            policy, source_code=13, surrounding_codes=[52], area_m2=130,
            scenario="same_class_bridge", source_component_complete=True,
            boundary_evidence_complete=True, boundary_status="closed_valid",
            bridge_target_code=52,
            source_component_id=7, target_component_ids=[101, 102],
            edge_distance_m=10, bridge_footprint_m2=130,
        )
    with pytest.raises(PolicyError, match="scenario must be same_class_bridge"):
        explain_fragment_decision(
            policy, source_code=13, surrounding_codes=[43, 52], area_m2=10,
            scenario="multi_neighbour", source_component_complete=True,
            boundary_evidence_complete=True, boundary_status="closed_valid",
            bridge_candidates_checked=True,
            bridge_target_code=52, target_component_ids=[101, 102],
        )


def test_multi_neighbour_requires_complete_boundary_and_bridge_check_evidence():
    policy = load_policy(POLICY_PATH)
    with pytest.raises(PolicyError, match="boundary_evidence_complete"):
        explain_fragment_decision(
            policy, source_code=13, surrounding_codes=[31, 43], area_m2=10,
            scenario="multi_neighbour", source_component_complete=True,
            boundary_status="closed_valid",
        )


def test_external_or_invalid_boundary_cannot_be_reported_as_enclosed():
    policy = load_policy(POLICY_PATH)
    with pytest.raises(PolicyError, match="must route to invalid_or_ambiguous"):
        explain_fragment_decision(
            policy, source_code=13, surrounding_codes=[43], area_m2=10,
            scenario="unique_enclosure", boundary_status="external_contact",
        )
    denied = explain_fragment_decision(
        policy, source_code=13, surrounding_codes=[43], area_m2=10,
        scenario="invalid_or_ambiguous", boundary_status="invalid_pixel_contact",
    )
    assert denied["decision"] == "DENY"
    assert denied["trace"] == ["hard_deny: boundary_status=invalid_pixel_contact"]
    with pytest.raises(PolicyError, match="boundary_status"):
        explain_fragment_decision(
            policy, source_code=13, surrounding_codes=[43], area_m2=10,
            scenario="unique_enclosure",
        )
    with pytest.raises(PolicyError, match="bridge_candidates_checked"):
        explain_fragment_decision(
            policy, source_code=13, surrounding_codes=[31, 43], area_m2=10,
            scenario="multi_neighbour", source_component_complete=True,
            boundary_evidence_complete=True, boundary_status="closed_valid",
        )


def test_schema_rejects_cross_field_permission_conflicts():
    policy = deepcopy(load_policy(POLICY_PATH))
    policy["classes"]["52"]["source_absorption"] = "allow"
    with pytest.raises(PolicyError, match="protected-source"):
        policy_sha256(policy)
    policy = deepcopy(load_policy(POLICY_PATH))
    policy["classes"]["61"]["target_growth"] = "allow"
    with pytest.raises(PolicyError, match="transport target"):
        policy_sha256(policy)


def test_strict_schema_rejects_unknown_field(tmp_path: Path):
    policy_text = POLICY_PATH.read_text(encoding="utf-8").replace(
        "  cascade_generation: false",
        "  cascade_generation: false\n  accidental_field: true",
        1,
    )
    bad = tmp_path / "bad.yaml"
    bad.write_text(policy_text, encoding="utf-8")
    with pytest.raises(PolicyError, match="unknown keys"):
        load_policy(bad)


def test_strict_loader_rejects_shadowed_yaml_keys(tmp_path: Path):
    bad = tmp_path / "duplicate.yaml"
    bad.write_text(POLICY_PATH.read_text(encoding="utf-8") + "\npolicy: {id: shadowed}\n", encoding="utf-8")
    with pytest.raises(PolicyError, match="duplicate YAML key"):
        load_policy(bad)


def test_migration_audit_reports_old_rules_not_moved_and_no_silent_diff():
    policy = load_policy(POLICY_PATH)
    audit = audit_legacy_migration(policy)
    assert audit["counts"] == {"moved": 13, "superseded": 11, "not_moved": 0}
    assert audit["not_moved"] == []
    assert audit["live_legacy_validation"]["all_verified"] is True
    assert audit["live_legacy_validation"]["field_checks"]["active_execution_contract"] is True
    assert audit["live_legacy_validation"]["field_checks"]["active_proposal_rank"] is True
    assert all(row["sha256_matches"] for row in audit["live_legacy_validation"]["implementation_file_sha256"].values())
    assert diff_policies(policy, load_policy(POLICY_PATH)) == []


def test_migration_audit_checks_active_moved_values_not_only_legacy_reference():
    policy = deepcopy(load_policy(POLICY_PATH))
    policy["classes"]["13"]["fragment_max_m2"] = 149
    policy["constraints"]["budgets"]["source_loss_fraction"] = 0.03
    policy["constraints"]["budgets"]["bridge_limits"]["21"]["max_edge_distance_m"] = 11
    checks = audit_legacy_migration(policy)["live_legacy_validation"]["field_checks"]
    assert checks["active_dynamic_fragmentation_thresholds"] is False
    assert checks["active_source_and_target_budgets"] is False
    assert checks["active_bridge_limits"] is False


def _proposal(
    proposal_id: str, *, dynamic: int = 1, components: int = 1,
    target: int = 52, probability: float = 0.5, area: float = 50,
    digest: str | None = None,
) -> dict[str, object]:
    return {
        "proposal_id": proposal_id,
        "dynamic_fragment_reduction": dynamic,
        "component_reduction": components,
        "target_class_code": target,
        "target_probability": probability,
        "changed_area_m2": area,
        "proposal_digest": digest or f"digest-{proposal_id}",
    }


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (_proposal("more-fragments", dynamic=2, target=52), _proposal("rarer-but-less", dynamic=1, target=33)),
        (_proposal("more-components", components=2, target=52), _proposal("rarer-but-one", components=1, target=33)),
        (_proposal("rarer", target=33), _proposal("common", target=52)),
        (_proposal("higher-probability", probability=0.9), _proposal("lower-probability", probability=0.2)),
        (_proposal("smaller-area", area=20), _proposal("larger-area", area=60)),
    ],
)
def test_approved_conflict_rank_applies_each_business_priority(first, second):
    ranked = rank_conflicting_proposals(load_policy(POLICY_PATH), [second, first])
    assert [row["proposal_id"] for row in ranked["ranked_proposals"]] == [first["proposal_id"], second["proposal_id"]]
    assert ranked["decision_scope"] == "conflict_order_only_for_fully_eligible_canonical_proposals"


def test_conflict_rank_uses_class_code_then_stable_technical_tie_break():
    policy = deepcopy(load_policy(POLICY_PATH))
    policy["decision_engine"]["statistics"]["raw_v31_optimization_before_share"]["33"] = 0.0199
    class_tie = rank_conflicting_proposals(
        policy, [_proposal("class-52", target=52), _proposal("class-33", target=33)],
    )
    assert [row["proposal_id"] for row in class_tie["ranked_proposals"]] == ["class-33", "class-52"]
    exact = rank_conflicting_proposals(
        load_policy(POLICY_PATH),
        [_proposal("z-id", digest="b"), _proposal("a-id", digest="a")],
    )
    assert [row["proposal_id"] for row in exact["ranked_proposals"]] == ["a-id", "z-id"]


def test_conflict_rank_priority_order_is_live_configuration():
    policy = deepcopy(load_policy(POLICY_PATH))
    rank = policy["decision_engine"]["proposal_adjudication"]["proposal_rank"]
    rarity = next(item for item in rank if item["field"] == "target_rarity_share")
    dynamic = next(item for item in rank if item["field"] == "dynamic_fragment_reduction")
    policy["decision_engine"]["proposal_adjudication"]["proposal_rank"] = [
        rarity,
        dynamic,
        *[item for item in rank if item not in (rarity, dynamic)],
    ]
    ranked = rank_conflicting_proposals(
        policy,
        [_proposal("more-fragments", dynamic=2, target=52), _proposal("rarer", dynamic=1, target=33)],
    )
    assert [row["proposal_id"] for row in ranked["ranked_proposals"]] == ["rarer", "more-fragments"]


def test_conflict_rank_rejects_incomplete_or_invalid_evidence():
    with pytest.raises(PolicyError, match="exactly"):
        rank_conflicting_proposals(load_policy(POLICY_PATH), [{"proposal_id": "incomplete"}])
    with pytest.raises(PolicyError, match="non-negative"):
        rank_conflicting_proposals(load_policy(POLICY_PATH), [_proposal("bad", dynamic=-1)])
