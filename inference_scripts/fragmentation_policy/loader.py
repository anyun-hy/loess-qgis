"""Load, query, hash, compare, and audit fragmentation policy documents."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import math
from typing import Any, Iterable, Mapping

import yaml

from .schema import CLASS_CODES, PolicyError, validate_policy


DEFAULT_POLICY_PATH = Path(__file__).with_name("policies") / "v33.yaml"


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader which refuses a configuration with shadowed keys."""


def _construct_unique_mapping(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, deep: bool = False,
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise PolicyError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping,
)


def _normalise(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalise(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_normalise(item) for item in value]
    return value


def policy_snapshot(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached, canonical JSON-safe policy snapshot."""

    validate_policy(policy)
    return _normalise(deepcopy(dict(policy)))


def policy_sha256(policy: Mapping[str, Any]) -> str:
    payload = json.dumps(policy_snapshot(policy), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def load_policy(path: str | Path | None = None) -> dict[str, Any]:
    """Load a policy file strictly; duplicate YAML keys and invalid schema fail."""

    chosen = Path(path) if path else DEFAULT_POLICY_PATH
    try:
        raw = yaml.load(chosen.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader)
    except OSError as exc:
        raise PolicyError(f"cannot read policy {chosen}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise PolicyError(f"cannot parse policy {chosen}: {exc}") from exc
    validate_policy(raw)
    return policy_snapshot(raw)


def _as_code(value: int | str, field: str) -> str:
    try:
        code = int(value)
    except (TypeError, ValueError) as exc:
        raise PolicyError(f"{field} must be an approved integer class code") from exc
    if code not in CLASS_CODES:
        raise PolicyError(f"{field} must be an approved integer class code")
    return str(code)


def _finite_positive(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
        raise PolicyError(f"{field} must be finite and greater than zero")
    return float(value)


def _component_id(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PolicyError(f"{field} must be a positive component ID")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PolicyError(f"{field} must be a non-negative integer")
    return value


def rank_conflicting_proposals(
    policy: Mapping[str, Any], proposals: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Order fully eligible canonical proposals without applying any raster."""

    snapshot = policy_snapshot(policy)
    required = {
        "proposal_id", "dynamic_fragment_reduction", "component_reduction",
        "target_class_code", "target_probability", "changed_area_m2",
        "proposal_digest",
    }
    shares = snapshot["decision_engine"]["statistics"]["raw_v31_optimization_before_share"]
    normalized: list[dict[str, Any]] = []
    proposal_ids: set[str] = set()
    for index, raw in enumerate(proposals):
        if not isinstance(raw, Mapping):
            raise PolicyError(f"proposals[{index}] must be a mapping")
        if set(raw) != required:
            raise PolicyError(f"proposals[{index}] must contain exactly {sorted(required)}")
        proposal_id = raw["proposal_id"]
        if not isinstance(proposal_id, str) or not proposal_id or proposal_id in proposal_ids:
            raise PolicyError("proposal_id values must be non-empty and unique")
        proposal_ids.add(proposal_id)
        target = _as_code(raw["target_class_code"], f"proposals[{index}].target_class_code")
        probability = raw["target_probability"]
        if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not math.isfinite(float(probability)) or not 0.0 <= float(probability) <= 1.0:
            raise PolicyError(f"proposals[{index}].target_probability must be in [0, 1]")
        digest = raw["proposal_digest"]
        if not isinstance(digest, str) or not digest:
            raise PolicyError(f"proposals[{index}].proposal_digest must be non-empty")
        normalized.append({
            "proposal_id": proposal_id,
            "dynamic_fragment_reduction": _nonnegative_int(raw["dynamic_fragment_reduction"], f"proposals[{index}].dynamic_fragment_reduction"),
            "component_reduction": _nonnegative_int(raw["component_reduction"], f"proposals[{index}].component_reduction"),
            "target_class_code": int(target),
            "target_probability": float(probability),
            "changed_area_m2": _finite_positive(raw["changed_area_m2"], f"proposals[{index}].changed_area_m2"),
            "proposal_digest": digest,
            "target_rarity_share": float(shares[target]),
        })

    rank_rules = snapshot["decision_engine"]["proposal_adjudication"]["proposal_rank"]

    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        values: list[Any] = []
        for rule in rank_rules:
            value = row[rule["field"]]
            values.append(-value if rule["order"] == "descending" else value)
        values.extend((str(row["proposal_digest"]), str(row["proposal_id"])))
        return tuple(values)

    ranked = []
    for position, row in enumerate(sorted(normalized, key=key), start=1):
        ranked.append({
            **row,
            "rank_position": position,
            "rank_evidence": {
                "dynamic_fragment_reduction": row["dynamic_fragment_reduction"],
                "component_reduction": row["component_reduction"],
                "target_rarity_share": row["target_rarity_share"],
                "target_probability": row["target_probability"],
                "changed_area_m2": row["changed_area_m2"],
                "target_class_code": row["target_class_code"],
            },
        })
    return {
        "policy_id": snapshot["policy"]["id"],
        "policy_sha256": policy_sha256(snapshot),
        "decision_scope": "conflict_order_only_for_fully_eligible_canonical_proposals",
        "proposal_rank": snapshot["decision_engine"]["proposal_adjudication"]["proposal_rank"],
        "exact_tie_break": snapshot["decision_engine"]["proposal_adjudication"]["exact_tie_break"],
        "ranked_proposals": ranked,
    }


def _resolve_relation(
    policy: Mapping[str, Any], source: str, target: str, scenario: str,
) -> dict[str, Any]:
    """Resolve one ABAC-style relation and return its complete explanation."""

    matches = [
        rule for rule in policy["decision_engine"]["relation_rules"]
        if rule["source"] in ("*", source)
        and rule["target"] in ("*", target)
        and rule["scenario"] in ("*", scenario)
    ]
    ordered = sorted(
        matches,
        key=lambda rule: (-rule["specificity"], -rule["priority"], rule["id"]),
    )
    if not ordered:
        return {
            "allowed": False,
            "reason": "default_deny_no_matching_rule",
            "matched_rules": [],
            "effective_rules": [],
        }

    # Class permissions are hard denies handled before this resolver.  Normal
    # relation rules first select the most specific rank, then the highest
    # priority within that rank.  A deny wins only among equal-rank rules.
    specificity = int(ordered[0]["specificity"])
    specific = [rule for rule in ordered if int(rule["specificity"]) == specificity]
    priority = max(int(rule["priority"]) for rule in specific)
    effective = [rule for rule in specific if int(rule["priority"]) == priority]
    denied = any(rule["effect"] == "deny" for rule in effective)
    return {
        "allowed": not denied and any(rule["effect"] == "allow" for rule in effective),
        "reason": "equal_rank_deny_overrides" if denied else "highest_rank_allow",
        "selected_specificity": specificity,
        "selected_priority": priority,
        "matched_rules": ordered,
        "effective_rules": sorted(effective, key=lambda rule: rule["id"]),
    }


def explain_fragment_decision(
    policy: Mapping[str, Any], *, source_code: int | str,
    surrounding_codes: Iterable[int | str], area_m2: float, scenario: str,
    target_probabilities: Mapping[int | str, float] | None = None,
    source_component_complete: bool | None = None,
    boundary_evidence_complete: bool | None = None,
    boundary_status: str | None = None,
    bridge_candidates_checked: bool | None = None,
    bridge_target_code: int | str | None = None,
    source_component_id: int | None = None,
    target_component_ids: Iterable[int] | None = None,
    edge_distance_m: float | None = None,
    bridge_footprint_m2: float | None = None,
) -> dict[str, Any]:
    """Explain one prospective V3.3 decision without modifying any raster."""

    snapshot = policy_snapshot(policy)
    source = _as_code(source_code, "source_code")
    surrounding = sorted({_as_code(code, "surrounding_codes") for code in surrounding_codes}, key=int)
    supplied_target_component_ids = tuple(target_component_ids) if target_component_ids is not None else None
    if not surrounding:
        raise PolicyError("surrounding_codes cannot be empty")
    if scenario not in snapshot["scenarios"]:
        raise PolicyError("unknown scenario")
    result: dict[str, Any] = {
        "policy_id": snapshot["policy"]["id"],
        "policy_sha256": policy_sha256(snapshot),
        "source_class_code": int(source),
        "surrounding_class_codes": [int(code) for code in surrounding],
        "scenario": scenario,
        "decision": "DENY",
        "decision_scope": "policy_precheck_only_not_raster_acceptance",
        "trace": [],
        "obligations": [],
    }
    source_row = snapshot["classes"][source]
    if boundary_status not in {"closed_valid", "external_contact", "invalid_pixel_contact", "incomplete_or_ambiguous"}:
        raise PolicyError("boundary_status must explicitly describe the complete source boundary")
    if boundary_status != "closed_valid":
        if scenario != "invalid_or_ambiguous":
            raise PolicyError("external, invalid-pixel, or incomplete boundary evidence must route to invalid_or_ambiguous")
        result["trace"].append(f"hard_deny: boundary_status={boundary_status}")
        return result
    if scenario == "invalid_or_ambiguous":
        result["trace"].append("hard_deny: invalid_or_ambiguous_structure")
        return result
    if source_row["source_absorption"] != "allow":
        result["trace"].append("source_permission: source_absorption=deny")
        return result
    if source_component_complete is not True:
        raise PolicyError("source_component_complete=true is required; partial components cannot be absorbed")
    if boundary_evidence_complete is not True:
        raise PolicyError("boundary_evidence_complete=true is required; partial boundary evidence cannot select a scenario")
    source_area = _finite_positive(area_m2, "area_m2")
    if source_area >= float(source_row["fragment_max_m2"]):
        result["trace"].append("area_gate: source_fragment_must_be_strictly_below_class_limit")
        return result
    bridge_target = _as_code(bridge_target_code, "bridge_target_code") if bridge_target_code is not None else None
    if len(surrounding) == 1:
        expected_scenario = "unique_enclosure"
    elif bridge_target is not None:
        if bridge_target not in surrounding:
            raise PolicyError("bridge_target_code must be one of the surrounding classes")
        if supplied_target_component_ids is None or len(set(supplied_target_component_ids)) < 2:
            raise PolicyError("bridge routing requires at least two target component IDs")
        expected_scenario = "same_class_bridge"
    else:
        if supplied_target_component_ids is not None:
            raise PolicyError("target_component_ids require an explicit bridge_target_code")
        if bridge_candidates_checked is not True:
            raise PolicyError("bridge_candidates_checked=true is required before multi_neighbour routing")
        expected_scenario = "multi_neighbour"
    if scenario != expected_scenario:
        raise PolicyError(f"scenario must be {expected_scenario} for the supplied complete boundary evidence")
    if scenario == "same_class_bridge":
        # The source is the *small intervening footprint*.  Bridge permission
        # belongs to the target class which will grow across that footprint;
        # it must not be inferred from whether the target may itself disappear.
        _component_id(source_component_id, "source_component_id")
        if supplied_target_component_ids is None:
            raise PolicyError("target_component_ids are required for same_class_bridge")
        component_ids = sorted({_component_id(value, "target_component_ids") for value in supplied_target_component_ids})
        if len(component_ids) < 2:
            raise PolicyError("same_class_bridge requires at least two target component IDs")
        distance = _finite_positive(edge_distance_m, "edge_distance_m")
        footprint = _finite_positive(bridge_footprint_m2, "bridge_footprint_m2")
        if bridge_target is None:  # guarded by scenario routing; defensive only
            raise PolicyError("bridge_target_code is required for same_class_bridge")
        target = bridge_target
        target_row = snapshot["classes"][target]
        if target_row["target_growth"] != "allow" or target_row["same_class_bridge"] != "allow":
            result["trace"].append("bridge_permission: target_growth_or_same_class_bridge=deny")
            return result
        relation = _resolve_relation(snapshot, source, target, scenario)
        result["target_rule_audit"] = {target: relation}
        if not relation["allowed"]:
            result["trace"].append(f"relation_permission: {source}->{target}=deny")
            return result
        cap = snapshot["constraints"]["budgets"]["bridge_limits"][target]
        if distance > float(cap["max_edge_distance_m"]):
            result["trace"].append("bridge_gate: edge_distance_exceeds_target_cap")
            return result
        if footprint > float(cap["max_new_footprint_m2"]):
            result["trace"].append("bridge_gate: footprint_exceeds_target_cap")
            return result
        if not math.isclose(footprint, source_area, rel_tol=0.0, abs_tol=1e-9):
            result["trace"].append("bridge_gate: footprint_must_equal_complete_source_component_area")
            return result
        result["decision"] = "PERMIT"
        result["selected_target_class_code"] = int(target)
        result["source_component_id"] = source_component_id
        result["target_component_ids"] = component_ids
        result["trace"].append("same_class_bridge: target class may grow across the small footprint; topology and budget checks remain mandatory")
        if source_row["post_cleanup_obligation"] != "none":
            result["decision"] = "PERMIT_WITH_OBLIGATIONS"
            result["obligations"].append({
                "kind": source_row["post_cleanup_obligation"],
                **snapshot["constraints"]["transport_overlay"],
            })
        return result
    legal = []
    target_rule_audit: dict[str, Any] = {}
    for target in surrounding:
        target_row = snapshot["classes"][target]
        if target_row["target_growth"] != "allow":
            result["trace"].append(f"target_permission: {target}=deny")
        else:
            relation = _resolve_relation(snapshot, source, target, scenario)
            target_rule_audit[target] = relation
            if not relation["allowed"]:
                result["trace"].append(f"relation_permission: {source}->{target}=deny")
            else:
                legal.append(target)
    result["target_rule_audit"] = target_rule_audit
    if not legal:
        result["trace"].append("default_deny: no legal surrounding target")
        return result
    if scenario == "unique_enclosure":
        selected = legal[0]
        result["trace"].append("unique_enclosure: sole legal surrounding class selected")
    else:
        supplied = target_probabilities or {}
        probabilities = {str(key): float(value) for key, value in supplied.items()}
        for code, probability in probabilities.items():
            _as_code(code, "target_probabilities")
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise PolicyError("target probabilities must be in [0, 1]")
            if code not in surrounding:
                raise PolicyError("target probabilities may only describe surrounding classes")
        shares = snapshot["decision_engine"]["statistics"]["raw_v31_optimization_before_share"]
        selected = min(legal, key=lambda code: (float(shares[code]), -probabilities.get(code, 0.0), int(code)))
        result["target_ranking"] = [
            {"class_code": int(code), "raw_v31_share": float(shares[code]), "target_probability": probabilities.get(code, 0.0)}
            for code in sorted(legal, key=lambda code: (float(shares[code]), -probabilities.get(code, 0.0), int(code)))
        ]
        result["trace"].append("multi_neighbour: raw V3.1 share ascending, target probability descending, then class code")
    result["decision"] = "PERMIT_WITH_OBLIGATIONS" if source_row["post_cleanup_obligation"] != "none" else "PERMIT"
    result["selected_target_class_code"] = int(selected)
    if source_row["post_cleanup_obligation"] != "none":
        result["obligations"].append({
            "kind": source_row["post_cleanup_obligation"],
            **snapshot["constraints"]["transport_overlay"],
        })
    result["trace"].append("pending_hard_gates: topology, budgets, final acceptance")
    return result


def _diff(before: Any, after: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        rows: list[dict[str, Any]] = []
        for key in sorted(set(before) | set(after), key=str):
            child = f"{path}.{key}" if path else str(key)
            if key not in before:
                rows.append({"path": child, "before": None, "after": after[key]})
            elif key not in after:
                rows.append({"path": child, "before": before[key], "after": None})
            else:
                rows.extend(_diff(before[key], after[key], child))
        return rows
    if before != after:
        return [{"path": path, "before": before, "after": after}]
    return []


def diff_policies(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _diff(policy_snapshot(before), policy_snapshot(after))


def audit_legacy_migration(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Report the frozen V3.1-B/V3.2 migration ledger stored in the policy."""

    snapshot = policy_snapshot(policy)
    rules = snapshot["migration"]["rules"]
    rows = [{"rule": name, **detail} for name, detail in sorted(rules.items())]
    return {
        "policy_id": snapshot["policy"]["id"],
        "policy_sha256": policy_sha256(snapshot),
        "legacy_policy": snapshot["migration"]["legacy_policy"],
        "evidence_files": list(snapshot["migration"]["evidence_files"]),
        "legacy_reference": snapshot["migration"]["legacy_reference"],
        "counts": {status: sum(row["status"] == status for row in rows) for status in ("moved", "superseded", "not_moved")},
        "rules": rows,
        "not_moved": [row for row in rows if row["status"] == "not_moved"],
    }
