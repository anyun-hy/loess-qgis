"""Strict structural validation for the V3.3 fragmentation policy document."""

from __future__ import annotations

import math
from typing import Any, Mapping


CLASS_CODES = (12, 13, 21, 31, 32, 33, 43, 51, 52, 53, 54, 61, 62, 71)
CLASS_KEYS = frozenset(str(code) for code in CLASS_CODES)


class PolicyError(ValueError):
    """Raised when a policy document is incomplete, ambiguous, or unsafe."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyError(f"{path} must be a mapping")
    return value


def _keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    unknown = actual - expected
    missing = expected - actual
    if unknown:
        raise PolicyError(f"{path} has unknown keys: {sorted(unknown)}")
    if missing:
        raise PolicyError(f"{path} is missing keys: {sorted(missing)}")


def _enum(value: Any, choices: set[str], path: str) -> None:
    if value not in choices:
        raise PolicyError(f"{path} must be one of {sorted(choices)}")


def _number(value: Any, path: str, *, minimum: float = 0.0, maximum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PolicyError(f"{path} must be a number")
    if float(value) < minimum or (maximum is not None and float(value) > maximum):
        raise PolicyError(f"{path} is outside the allowed range")


def validate_policy(document: Any) -> None:
    """Validate the complete schema; unknown fields fail closed by design."""

    root = _mapping(document, "policy")
    _keys(root, {"schema_version", "policy", "classes", "scenarios", "decision_engine", "constraints", "migration"}, "policy")
    if root["schema_version"] != "fragmentation-policy/v1":
        raise PolicyError("policy.schema_version must be fragmentation-policy/v1")

    meta = _mapping(root["policy"], "policy.policy")
    _keys(meta, {"id", "version", "status", "baseline", "connectivity", "single_pass", "cascade_generation"}, "policy.policy")
    if not isinstance(meta["id"], str) or not meta["id"]:
        raise PolicyError("policy.policy.id must be a non-empty string")
    if not isinstance(meta["version"], str) or not meta["version"]:
        raise PolicyError("policy.policy.version must be a non-empty string")
    _enum(meta["status"], {"draft", "candidate", "approved", "frozen"}, "policy.policy.status")
    _enum(meta["baseline"], {"frozen_v3", "frozen_v31", "frozen_v32"}, "policy.policy.baseline")
    if meta["connectivity"] != 4 or meta["single_pass"] is not True or meta["cascade_generation"] is not False:
        raise PolicyError("policy must use one frozen 4-connected, non-cascading pass")

    classes = _mapping(root["classes"], "policy.classes")
    if set(classes) != CLASS_KEYS:
        raise PolicyError("policy.classes must contain the 14 approved class codes exactly once")
    class_fields = {"code", "name", "source_absorption", "target_growth", "same_class_bridge", "fragment_max_m2", "relation_profile", "post_cleanup_obligation"}
    for key in sorted(classes):
        row = _mapping(classes[key], f"policy.classes.{key}")
        _keys(row, class_fields, f"policy.classes.{key}")
        if row["code"] != int(key):
            raise PolicyError(f"policy.classes.{key}.code must equal {key}")
        if not isinstance(row["name"], str) or not row["name"]:
            raise PolicyError(f"policy.classes.{key}.name must be non-empty")
        for field in ("source_absorption", "target_growth", "same_class_bridge"):
            _enum(row[field], {"allow", "deny"}, f"policy.classes.{key}.{field}")
        _number(row["fragment_max_m2"], f"policy.classes.{key}.fragment_max_m2", minimum=0.000001)
        _enum(row["relation_profile"], {"policy_rules"}, f"policy.classes.{key}.relation_profile")
        _enum(row["post_cleanup_obligation"], {"none", "manual_transport_vector_overlay"}, f"policy.classes.{key}.post_cleanup_obligation")
        if row["post_cleanup_obligation"] != "none" and row["source_absorption"] != "allow":
            raise PolicyError(f"policy.classes.{key} overlay obligation requires source absorption")
        if row["same_class_bridge"] == "allow" and row["target_growth"] != "allow":
            raise PolicyError(f"policy.classes.{key} bridge permission requires target growth")

    protected_keys = {"12", "33", "52", "71"}
    transport_keys = {"61", "62"}
    for key, row in classes.items():
        expected_source = "deny" if key in protected_keys else "allow"
        if row["source_absorption"] != expected_source:
            raise PolicyError(f"policy.classes.{key}.source_absorption conflicts with the approved protected-source set")
        expected_target = "deny" if key in transport_keys else "allow"
        if row["target_growth"] != expected_target:
            raise PolicyError(f"policy.classes.{key}.target_growth conflicts with the transport target contract")
        expected_obligation = "manual_transport_vector_overlay" if key in transport_keys else "none"
        if row["post_cleanup_obligation"] != expected_obligation:
            raise PolicyError(f"policy.classes.{key}.post_cleanup_obligation conflicts with the transport contract")

    scenarios = _mapping(root["scenarios"], "policy.scenarios")
    expected_scenarios = {"unique_enclosure", "multi_neighbour", "same_class_bridge", "invalid_or_ambiguous"}
    if set(scenarios) != expected_scenarios:
        raise PolicyError("policy.scenarios must define the four mutually exclusive scenario routes")
    unique = _mapping(scenarios["unique_enclosure"], "policy.scenarios.unique_enclosure")
    _keys(unique, {"match", "source_scope", "target_selection", "evidence_gates"}, "policy.scenarios.unique_enclosure")
    if unique["match"] != "one_surrounding_class" or unique["target_selection"] != "unique_surrounding_class":
        raise PolicyError("unique enclosure must select its sole surrounding class")
    multi = _mapping(scenarios["multi_neighbour"], "policy.scenarios.multi_neighbour")
    _keys(multi, {"match", "source_scope", "target_selection", "rarity_source", "contact_measurement", "evidence_gates"}, "policy.scenarios.multi_neighbour")
    if multi["match"] != "two_or_more_surrounding_classes_without_eligible_bridge":
        raise PolicyError("multi-neighbour match is invalid")
    if multi["target_selection"] != ["raw_v31_share_ascending", "target_probability_descending", "class_code_ascending"]:
        raise PolicyError("multi-neighbour target order must use raw V3.1 share, target probability, then class code")
    if multi["rarity_source"] != "statistics.raw_v31_optimization_before_share" or multi["contact_measurement"] != "audit_only":
        raise PolicyError("multi-neighbour rarity/contact contract is invalid")
    bridge = _mapping(scenarios["same_class_bridge"], "policy.scenarios.same_class_bridge")
    _keys(bridge, {"match", "source_scope", "target_selection", "evidence_gates"}, "policy.scenarios.same_class_bridge")
    if bridge["match"] != "two_or_more_surrounding_classes_and_two_or_more_components_of_selected_target" or bridge["target_selection"] != "same_class_only":
        raise PolicyError("same-class bridge route is invalid")
    invalid = _mapping(scenarios["invalid_or_ambiguous"], "policy.scenarios.invalid_or_ambiguous")
    _keys(invalid, {"match", "decision"}, "policy.scenarios.invalid_or_ambiguous")
    if invalid["match"] != "external_contact_or_invalid_pixel_or_incomplete_routing_evidence" or invalid["decision"] != "deny":
        raise PolicyError("invalid or ambiguous structure must deny")
    for name in ("unique_enclosure", "multi_neighbour", "same_class_bridge"):
        if scenarios[name]["source_scope"] != "complete_source_component_only":
            raise PolicyError(f"policy.scenarios.{name}.source_scope must reject partial components")
        gates = scenarios[name]["evidence_gates"]
        if gates != {"semantic": "policy_relation_only", "probability": "audit_only", "confidence": "audit_only"}:
            raise PolicyError(f"policy.scenarios.{name}.evidence_gates is invalid")

    engine = _mapping(root["decision_engine"], "policy.decision_engine")
    _keys(engine, {"default_decision", "precedence", "conflict_resolution", "scenario_routing", "relation_rules", "proposal_adjudication", "statistics"}, "policy.decision_engine")
    if engine["default_decision"] != "deny":
        raise PolicyError("policy.decision_engine.default_decision must deny")
    if engine["precedence"] != ["hard_deny", "scenario_routing", "source_permission", "target_permission", "relation_permission", "area_budget_topology", "target_selection"]:
        raise PolicyError("policy.decision_engine.precedence is invalid")
    if engine["conflict_resolution"] != ["class_permission_hard_deny", "higher_specificity", "higher_priority", "equal_rank_deny_overrides", "default_deny"]:
        raise PolicyError("policy.decision_engine.conflict_resolution is invalid")
    routing = _mapping(engine["scenario_routing"], "policy.decision_engine.scenario_routing")
    _keys(routing, {"order", "unique_enclosure_rule", "bridge_rule", "multi_neighbour_rule"}, "policy.decision_engine.scenario_routing")
    if routing != {
        "order": ["invalid_or_ambiguous", "unique_enclosure", "same_class_bridge", "multi_neighbour"],
        "unique_enclosure_rule": "one_distinct_surrounding_class_regardless_of_target_component_count",
        "bridge_rule": "multi_class_boundary_with_at_least_two_components_of_one_selected_target",
        "multi_neighbour_rule": "complete_bridge_check_found_no_eligible_bridge",
    }:
        raise PolicyError("policy.decision_engine.scenario_routing is invalid")
    rules = engine["relation_rules"]
    if not isinstance(rules, list) or not rules:
        raise PolicyError("policy.decision_engine.relation_rules are required")
    ids: set[str] = set()
    seen: dict[tuple[str, str, str, int, int], str] = {}
    rule_fields = {"id", "effect", "source", "target", "scenario", "priority", "specificity"}
    scenarios_allowed = {"unique_enclosure", "multi_neighbour", "same_class_bridge", "*"}
    for index, raw_rule in enumerate(rules):
        rule = _mapping(raw_rule, f"policy.decision_engine.relation_rules[{index}]")
        _keys(rule, rule_fields, f"policy.decision_engine.relation_rules[{index}]")
        if not isinstance(rule["id"], str) or not rule["id"] or rule["id"] in ids:
            raise PolicyError("relation rule ids must be non-empty and unique")
        ids.add(rule["id"])
        _enum(rule["effect"], {"allow", "deny"}, f"relation rule {rule['id']} effect")
        for field in ("source", "target"):
            if rule[field] != "*" and rule[field] not in CLASS_KEYS:
                raise PolicyError(f"relation rule {rule['id']} {field} is invalid")
        if rule["scenario"] not in scenarios_allowed:
            raise PolicyError(f"relation rule {rule['id']} scenario is invalid")
        if isinstance(rule["priority"], bool) or not isinstance(rule["priority"], int):
            raise PolicyError(f"relation rule {rule['id']} priority must be an integer")
        expected_specificity = sum(rule[field] != "*" for field in ("source", "target", "scenario"))
        if rule["specificity"] != expected_specificity:
            raise PolicyError(f"relation rule {rule['id']} specificity must equal its exact selectors")
        key = (str(rule["source"]), str(rule["target"]), str(rule["scenario"]), rule["priority"], rule["specificity"])
        other = seen.get(key)
        if other:
            raise PolicyError(f"relation rules {other} and {rule['id']} are an ambiguous same-rank conflict")
        seen[key] = rule["id"]
    adjudication = _mapping(engine["proposal_adjudication"], "policy.decision_engine.proposal_adjudication")
    _keys(adjudication, {"conflict_set", "canonical_deduplication", "proposal_id_collision", "footprint_conflict", "core_owner", "dependency_mode", "proposal_rank", "exact_tie_break"}, "policy.decision_engine.proposal_adjudication")
    fixed_adjudication = {key: value for key, value in adjudication.items() if key != "proposal_rank"}
    if fixed_adjudication != {
        "conflict_set": "all_scenarios_single_incremental_pass",
        "canonical_deduplication": "required",
        "proposal_id_collision": "reject",
        "footprint_conflict": "reject",
        "core_owner": "required",
        "dependency_mode": "v31b_incremental",
        "exact_tie_break": ["proposal_digest_ascending", "proposal_id_ascending"],
    }:
        raise PolicyError("policy.decision_engine.proposal_adjudication is invalid")
    rank_sources = {
        "dynamic_fragment_reduction": "proposal",
        "component_reduction": "proposal",
        "target_rarity_share": "decision_engine.statistics.raw_v31_optimization_before_share",
        "target_probability": "proposal",
        "changed_area_m2": "proposal",
        "target_class_code": "proposal",
    }
    proposal_rank = adjudication["proposal_rank"]
    if not isinstance(proposal_rank, list) or len(proposal_rank) != len(rank_sources):
        raise PolicyError("proposal_rank must configure every supported field exactly once")
    seen_rank_fields: set[str] = set()
    for index, raw_rank in enumerate(proposal_rank):
        rank = _mapping(raw_rank, f"proposal_rank[{index}]")
        _keys(rank, {"field", "order", "value_source"}, f"proposal_rank[{index}]")
        field = rank["field"]
        if field not in rank_sources or field in seen_rank_fields:
            raise PolicyError("proposal_rank fields must be supported and unique")
        seen_rank_fields.add(field)
        _enum(rank["order"], {"ascending", "descending"}, f"proposal_rank[{index}] order")
        if rank["value_source"] != rank_sources[field]:
            raise PolicyError(f"proposal_rank[{index}] value_source is invalid")
    statistics = _mapping(engine["statistics"], "policy.decision_engine.statistics")
    _keys(statistics, {"source", "raw_v31_optimization_before_share", "raw_v31_optimization_before_area_km2"}, "policy.decision_engine.statistics")
    if not isinstance(statistics["source"], str) or not statistics["source"]:
        raise PolicyError("raw V3.1 statistics source is required")
    shares = _mapping(statistics["raw_v31_optimization_before_share"], "policy.decision_engine.statistics.raw_v31_optimization_before_share")
    if set(shares) != CLASS_KEYS:
        raise PolicyError("raw V3.1 statistics must contain every class")
    for code, share in shares.items():
        _number(share, f"policy raw V3.1 share {code}", minimum=0.0, maximum=1.0)
    areas = _mapping(statistics["raw_v31_optimization_before_area_km2"], "policy.decision_engine.statistics.raw_v31_optimization_before_area_km2")
    if set(areas) != CLASS_KEYS:
        raise PolicyError("raw V3.1 area statistics must contain every class")
    for code, area in areas.items():
        _number(area, f"policy raw V3.1 area {code}", minimum=0.0)

    constraints = _mapping(root["constraints"], "policy.constraints")
    _keys(constraints, {"hard_gates", "budgets", "transport_overlay", "execution_contract", "evaluation", "post_cleanup_acceptance"}, "policy.constraints")
    if constraints["hard_gates"] != ["valid_single_label", "no_gap_overlap_outside", "core_owner_only", "proposal_footprint_conflict_reject", "canonical_proposal_dedup_required", "single_incremental_adjudication", "source_connectivity_preserved", "target_attachment_preserved", "per_class_components_nonincreasing", "global_components_nonincreasing", "dynamic_fragments_nonincreasing", "dynamic_fragment_area_nonincreasing", "total_boundary_edges_nonincreasing", "internal_boundary_edges_nonincreasing", "total_boundary_meters_nonincreasing", "internal_boundary_meters_nonincreasing", "full_audit_required", "lineage_required"]:
        raise PolicyError("policy.constraints.hard_gates is invalid")
    budgets = _mapping(constraints["budgets"], "policy.constraints.budgets")
    _keys(budgets, {"source_loss_fraction", "target_gain_fraction", "protected_source_loss_fraction", "protected_bridge_gain_fraction", "protected_source_codes", "bridge_limits"}, "policy.constraints.budgets")
    _number(budgets["source_loss_fraction"], "source loss budget", maximum=1.0)
    _number(budgets["target_gain_fraction"], "target gain budget", maximum=1.0)
    if budgets["protected_bridge_gain_fraction"] != 0.01:
        raise PolicyError("active protected bridge budget contract is invalid")
    if budgets["protected_source_loss_fraction"] != 0.0 or budgets["protected_source_codes"] != [12, 33, 52, 71]:
        raise PolicyError("protected source contract is invalid")
    bridge_limits = _mapping(budgets["bridge_limits"], "policy.constraints.budgets.bridge_limits")
    if set(bridge_limits) != CLASS_KEYS:
        raise PolicyError("active bridge limits must contain every class")
    for code, row in bridge_limits.items():
        item = _mapping(row, f"active bridge limit {code}")
        _keys(item, {"allow", "max_edge_distance_m", "max_new_footprint_m2"}, f"active bridge limit {code}")
        _enum(item["allow"], {"allow", "deny"}, f"active bridge limit {code} allow")
        _number(item["max_edge_distance_m"], f"active bridge limit {code} distance", minimum=0.0)
        _number(item["max_new_footprint_m2"], f"active bridge limit {code} footprint", minimum=0.0)
        if (item["allow"] == "allow") != (classes[code]["same_class_bridge"] == "allow"):
            raise PolicyError(f"active bridge limit {code} disagrees with class bridge permission")
        if item["allow"] == "deny" and (item["max_edge_distance_m"] != 0 or item["max_new_footprint_m2"] != 0):
            raise PolicyError(f"denied bridge {code} must have zero caps")
        if item["allow"] == "allow" and (item["max_edge_distance_m"] <= 0 or item["max_new_footprint_m2"] <= 0):
            raise PolicyError(f"allowed bridge {code} must have positive caps")
    if bridge_limits["52"] != {"allow": "allow", "max_edge_distance_m": 10, "max_new_footprint_m2": 120}:
        raise PolicyError("52 active bridge contract is invalid")
    for code in ("61", "62"):
        if bridge_limits[code] != {"allow": "deny", "max_edge_distance_m": 0, "max_new_footprint_m2": 0}:
            raise PolicyError(f"{code} transport bridge contract is invalid")
    overlay = _mapping(constraints["transport_overlay"], "policy.constraints.transport_overlay")
    _keys(overlay, {"required", "artifact_sha256", "artifact_version", "precedence", "post_overlay_budget_audit", "budget_treatment"}, "policy.constraints.transport_overlay")
    if overlay != {"required": True, "artifact_sha256": "required_at_runtime", "artifact_version": "required_at_runtime", "precedence": "manual_transport_vector_over_fragmentation", "post_overlay_budget_audit": True, "budget_treatment": "authoritative_overlay_exempt_from_fragmentation_target_gain_report_separately"}:
        raise PolicyError("transport overlay contract is invalid")
    execution = _mapping(constraints["execution_contract"], "policy.constraints.execution_contract")
    _keys(execution, {"bridge_edge_distance", "bridge_path_length", "budget_denominator", "canonical_proposal_identity", "duplicate_representative", "canonical_distance_evidence", "inconsistent_duplicate_rank_or_evidence"}, "policy.constraints.execution_contract")
    if execution != {
        "bridge_edge_distance": "euclidean_cell_polygon_edge_distance_m",
        "bridge_path_length": "four_neighbour_path_length_m",
        "budget_denominator": "class_budget_mask_and_valid_frozen_baseline",
        "canonical_proposal_identity": "kind_target_index_target_code_sorted_footprint_source_indices_source_codes_source_component_ids_baseline_target_component_ids",
        "duplicate_representative": "proposal_id_then_occurrence_edge_distance_then_occurrence_path_length_none_last",
        "canonical_distance_evidence": "minimum_present_edge_distance_and_path_length",
        "inconsistent_duplicate_rank_or_evidence": "reject",
    }:
        raise PolicyError("active V3.3 execution contract is invalid")
    evaluation = _mapping(constraints["evaluation"], "policy.constraints.evaluation")
    _keys(evaluation, {"comparison_baseline", "required_partition_count", "effect_metric", "effect_fraction", "lineage_gates", "coverage_gates"}, "policy.constraints.evaluation")
    if evaluation != {
        "comparison_baseline": "frozen_v32",
        "required_partition_count": 140,
        "effect_metric": "dynamic_fragment_count_reduction_fraction",
        "effect_fraction": 0.005,
        "lineage_gates": ["same_v31a_parent", "same_snapshot", "same_v3_policy", "same_partition_ids_and_raw_v3_valid_sha256"],
        "coverage_gates": ["part_count_140", "core_overlap_zero", "geometric_gap_zero", "candidate_invalid_inside_zero", "candidate_outside_valid_zero"],
    }:
        raise PolicyError("V3.3 evaluation contract is invalid")
    if constraints["post_cleanup_acceptance"] != ["manual_transport_vector_overlay_if_obligated", "rerun_all_hard_gates", "rerun_fragmentation_budget_audit", "audit_authoritative_overlay_separately"]:
        raise PolicyError("post-cleanup acceptance contract is invalid")

    migration = _mapping(root["migration"], "policy.migration")
    _keys(migration, {"legacy_policy", "evidence_files", "legacy_reference", "rules"}, "policy.migration")
    if migration["legacy_policy"] != "V3.1-B/V3.2 implementation evidence":
        raise PolicyError("migration legacy policy label is invalid")
    if not isinstance(migration["evidence_files"], list) or not migration["evidence_files"]:
        raise PolicyError("migration evidence files are required")
    legacy_reference = _mapping(migration["legacy_reference"], "policy.migration.legacy_reference")
    _keys(legacy_reference, {"v31b_dynamic_fragmentation_m2", "v31b_enclosed_island_max_m2", "v31b_ordinary_protected", "v31b_protected_source_codes", "v31b_probability_thresholds", "v31b_source_confidence_max", "v31b_bridge_limits", "v31b_semantic_compatible_targets", "v31b_budgets", "v31b_audit_proposal_limit", "v31b_adjudication_mode", "v31b_algorithm_contract", "v31b_global_acceptance", "v32_selector_and_contact"}, "policy.migration.legacy_reference")
    if legacy_reference["v31b_source_confidence_max"] != 0.65:
        raise PolicyError("legacy V3.1-B confidence cap must be recorded exactly")
    if legacy_reference["v31b_budgets"] != {"source_loss_fraction": 0.02, "target_gain_fraction": 0.02, "protected_bridge_gain_fraction": 0.01}:
        raise PolicyError("legacy V3.1-B budgets must be recorded exactly")
    for field in ("v31b_dynamic_fragmentation_m2", "v31b_enclosed_island_max_m2", "v31b_ordinary_protected", "v31b_probability_thresholds", "v31b_bridge_limits"):
        table = _mapping(legacy_reference[field], f"policy.migration.legacy_reference.{field}")
        if set(table) != CLASS_KEYS:
            raise PolicyError(f"legacy {field} must contain every approved class")
    semantic = _mapping(legacy_reference["v31b_semantic_compatible_targets"], "policy.migration.legacy_reference.v31b_semantic_compatible_targets")
    if set(semantic) != CLASS_KEYS:
        raise PolicyError("legacy V3.1-B semantic targets must contain every class")
    rules = _mapping(migration["rules"], "policy.migration.rules")
    if not rules:
        raise PolicyError("migration rules are required")
    for key, row_raw in rules.items():
        row = _mapping(row_raw, f"policy.migration.rules.{key}")
        _keys(row, {"legacy_behavior", "status", "v33_location", "reason"}, f"policy.migration.rules.{key}")
        _enum(row["status"], {"moved", "superseded", "not_moved"}, f"policy.migration.rules.{key}.status")
