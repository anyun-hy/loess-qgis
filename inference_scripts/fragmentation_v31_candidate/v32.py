"""Isolated V3.2 candidate: B proposals plus closed multi-neighbour rarity.

V3.2 never mutates V3 or V3.1-B.  It generates B's frozen-baseline proposals
and one additional proposal for a dynamic, non-external source component that
touches two or more legal *classes*.  The legal destination is selected once,
before adjudication, by the immutable full-owner-Core V3 class census.  Contact
length is retained as audit evidence only; it is deliberately absent from the
selection key and class totals never change while actions are accepted.  A
target is ranked by ``(global_count ASC, mean_target_probability DESC,
class_code ASC)`` only.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from . import candidate as _b


V32_POLICY_ID = "fragmentation_v32_frozen_total_multineighbour_candidate_v1"
V32_POLICY_VERSION = "v32_20260825"
V32_ADJUDICATION_MODE = "b_single_incremental_adjudication_with_frozen_total_target_choice_v1"


def v32_policy() -> _b.CandidatePolicy:
    """Use the B gates unchanged under a distinct, isolated identity."""

    return replace(
        _b.v31b_policy(), policy_id=V32_POLICY_ID, policy_version=V32_POLICY_VERSION
    )


def policy_snapshot(policy: _b.CandidatePolicy | None = None) -> dict[str, Any]:
    """V3.2's policy snapshot, including the frozen-total algorithm contract."""

    snapshot = _b.policy_snapshot(policy or v32_policy())
    snapshot["v32_algorithm_contract"] = {
        "additional_proposal": "closed_multi_neighbor_rarity",
        "source_eligibility": "dynamic_nonprotected_nonexternal_closed_source_touching_at_least_two_classes",
        "target_gates": "existing_B_semantic_protected_target_probability_and_budget_adjudication_gates",
        "target_selection": "full_owner_core_strict_valid_v3_frozen_total_ascending_then_mean_target_probability_descending_then_class_code_ascending",
        "contact_measurement": "audit_only_not_filter_or_sort",
        "adjudication": "one_B_incremental_adjudication_over_B_original_and_V32_new_proposals",
    }
    return snapshot


def policy_snapshot_sha256(policy: _b.CandidatePolicy | None = None) -> str:
    return _sha256_json(policy_snapshot(policy))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _frozen_totals(
    totals: Mapping[int, int], class_codes: Sequence[int]
) -> dict[int, int]:
    expected = {int(code) for code in class_codes}
    received = {int(code) for code in totals}
    if received != expected or len(totals) != len(expected):
        raise _b.CandidateError("frozen_global_class_pixel_totals must contain every class code exactly once")
    result: dict[int, int] = {}
    for raw_code, raw_count in totals.items():
        if isinstance(raw_count, bool) or int(raw_count) != raw_count or int(raw_count) < 0:
            raise _b.CandidateError("frozen_global_class_pixel_totals values must be non-negative integers")
        result[int(raw_code)] = int(raw_count)
    return result


def _contact_edges(
    source: _b._Component,
    component_map: np.ndarray,
    components_by_id: Mapping[int, _b._Component],
    valid: np.ndarray,
    pixel_size_m: tuple[float, float],
) -> tuple[dict[int, tuple[int, ...]], dict[int, int], dict[int, float], bool]:
    """Return target IDs plus count/physical-length audit; invalid contact rejects."""

    height, width = valid.shape
    by_code: dict[int, set[int]] = defaultdict(set)
    contacts: Counter[int] = Counter()
    lengths: Counter[int] = Counter()
    row_step_m, column_step_m = pixel_size_m
    for row, col in source.pixels:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = int(row + dr), int(col + dc)
            if rr < 0 or rr >= height or cc < 0 or cc >= width or not valid[rr, cc]:
                return {}, {}, {}, False
            component_id = int(component_map[rr, cc])
            if component_id == source.component_id:
                continue
            target = components_by_id[component_id]
            by_code[target.class_code].add(component_id)
            contacts[target.class_code] += 1
            # A row-to-row neighbour shares the column-oriented cell edge;
            # a column-to-column neighbour shares the row-oriented cell edge.
            lengths[target.class_code] += column_step_m if dr else row_step_m
    return (
        {code: tuple(sorted(ids)) for code, ids in by_code.items()},
        dict(contacts),
        {code: float(value) for code, value in lengths.items()},
        True,
    )


def _multi_neighbour_proposals(
    labels: np.ndarray,
    valid: np.ndarray,
    probabilities: np.ndarray,
    confidence: np.ndarray | None,
    class_codes: Sequence[int],
    component_map: np.ndarray,
    components: Sequence[_b._Component],
    policy: _b.CandidatePolicy,
    pixel_area_m2: float,
    pixel_size_m: tuple[float, float],
    frozen_totals: Mapping[int, int],
    rejections: Counter[str],
) -> tuple[list[_b._Proposal], list[dict[str, Any]]]:
    """Generate exactly one census-selected proposal per eligible source."""

    components_by_id = {item.component_id: item for item in components}
    proposals: list[_b._Proposal] = []
    audit: list[dict[str, Any]] = []
    for source in components:
        source_policy = policy.class_policies[source.class_code]
        record: dict[str, Any] = {
            "source_component_id": int(source.component_id),
            "source_class_code": int(source.class_code),
            "source_pixel_count": int(len(source.pixels)),
        }
        if source.class_code in policy.protected_source_codes or source_policy.ordinary_protected:
            rejections["multi_neighbour_protected_source"] += 1; record["reason"] = "protected_source"
        elif source.touches_external:
            rejections["multi_neighbour_external_source"] += 1; record["reason"] = "external_source"
        elif source_policy.enclosed_island_max_m2 <= 0 or len(source.pixels) * pixel_area_m2 > source_policy.enclosed_island_max_m2:
            rejections["multi_neighbour_area_cap"] += 1; record["reason"] = "area_cap"
        elif len(source.pixels) * pixel_area_m2 >= source_policy.dynamic_fragmentation_m2:
            rejections["multi_neighbour_not_dynamic_fragment"] += 1; record["reason"] = "not_dynamic_fragment"
        else:
            rows, cols = source.pixels[:, 0], source.pixels[:, 1]
            mean_confidence = float(np.mean(confidence[rows, cols] if confidence is not None else probabilities[source.class_index, rows, cols]))
            record["mean_source_confidence"] = mean_confidence
            if mean_confidence > policy.island_maximum_mean_confidence:
                rejections["multi_neighbour_confidence"] += 1; record["reason"] = "confidence"
            else:
                neighbours, contacts, contact_lengths, safe = _contact_edges(
                    source, component_map, components_by_id, valid, pixel_size_m,
                )
                record["contact_edges_by_class"] = {str(key): value for key, value in sorted(contacts.items())}
                record["contact_length_m_by_class"] = {str(key): value for key, value in sorted(contact_lengths.items())}
                if not safe:
                    rejections["multi_neighbour_external_contact"] += 1; record["reason"] = "external_contact"
                else:
                    if len(neighbours) < 2:
                        rejections["multi_neighbour_fewer_than_two_touching_classes"] += 1
                        record["reason"] = "fewer_than_two_touching_classes"
                        continue
                    # A detailed record is reserved for true T3/T4 sources.
                    # All other components remain aggregate rejection counts.
                    record["structural_T3_T4"] = True
                    record["_detail"] = True
                    legal: list[tuple[int, _b._Proposal]] = []
                    target_checks: list[dict[str, Any]] = []
                    for code in sorted(neighbours):
                        check: dict[str, Any] = {
                            "target_class_code": int(code),
                            "contact_edge_count_audit_only": int(contacts.get(code, 0)),
                            "contact_length_m_audit_only": float(contact_lengths.get(code, 0.0)),
                        }
                        if code not in policy.semantic_compatible_targets.get(source.class_code, frozenset()):
                            check["eligible"] = False; check["reason"] = "semantic_incompatible_target"; target_checks.append(check)
                            continue
                        target_policy = policy.class_policies[code]
                        if target_policy.ordinary_protected:
                            check["eligible"] = False; check["reason"] = "protected_ordinary_target"; target_checks.append(check)
                            continue
                        target_index = list(class_codes).index(code)
                        proposal = _b._proposal_with_scores(
                            "closed_multi_neighbor_rarity", target_index, source.pixels,
                            labels, valid, class_codes, policy, pixel_area_m2, probabilities,
                            target_policy, component_map,
                            target_component_seed_ids=neighbours[code],
                            components_by_id=components_by_id,
                            extra_evidence={"mean_source_confidence": mean_confidence, "contact_edges_audit_only": float(contacts.get(code, 0)), "contact_length_m_audit_only": float(contact_lengths.get(code, 0.0))},
                            generation_rejections=rejections,
                        )
                        if proposal is not None:
                            check.update({"eligible": True, "reason": "passes_existing_b_gates", "mean_target_probability": float(proposal.evidence["mean_target_probability"])})
                            legal.append((code, proposal))
                        else:
                            check.update({"eligible": False, "reason": "probability_gate"})
                        target_checks.append(check)
                    record["target_gate_audit"] = target_checks
                    record["legal_target_class_codes"] = [code for code, _ in legal]
                    record["frozen_total_by_legal_target"] = {str(code): int(frozen_totals[code]) for code, _ in legal}
                    if not legal:
                        rejections["multi_neighbour_no_legal_target"] += 1; record["reason"] = "no_legal_target"
                    else:
                        chosen_code, chosen = min(
                            legal,
                            key=lambda item: (
                                frozen_totals[item[0]],
                                -float(item[1].evidence["mean_target_probability"]),
                                item[0],
                            ),
                        )
                        record.update({"reason": "generated", "chosen_target_class_code": chosen_code, "chosen_frozen_total": int(frozen_totals[chosen_code]), "target_selection_key": [int(frozen_totals[chosen_code]), -float(chosen.evidence["mean_target_probability"]), int(chosen_code)], "contact_edges_used_for_filtering": False, "contact_edges_used_for_selection": False, "contact_length_used_for_filtering": False, "contact_length_used_for_selection": False})
                        proposals.append(chosen)
        if record.pop("_detail", False):
            audit.append(record)
    return proposals, audit


def apply_v32_candidate(
    labels: np.ndarray, *, class_codes: Sequence[int], pixel_area_m2: float,
    frozen_global_class_pixel_totals: Mapping[int, int],
    pixel_size_m: tuple[float, float] | None = None, valid_mask: np.ndarray | None = None,
    class_budget_mask: np.ndarray | None = None, probabilities: np.ndarray | None = None,
    confidence: np.ndarray | None = None, policy: _b.CandidatePolicy | None = None,
    baseline_kind: str, full_audit: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply V3.2 from one frozen V3-cleaned baseline and immutable totals."""

    if baseline_kind not in _b.ALLOWED_BASELINE_KINDS:
        raise _b.CandidateError(f"baseline_kind must be one of {sorted(_b.ALLOWED_BASELINE_KINDS)}")
    selected_policy = policy or v32_policy()
    totals = _frozen_totals(frozen_global_class_pixel_totals, class_codes)
    baseline, valid, probs, conf, sizes = _b._validate(labels, class_codes, valid_mask, probabilities, confidence, pixel_area_m2, pixel_size_m, selected_policy)
    budget_mask = _b._class_budget_mask(class_budget_mask, valid)
    component_map, components = _b._component_index(baseline, valid, class_codes)
    rejections: Counter[str] = Counter()
    proposals = _b._island_proposals(baseline, valid, probs, conf, class_codes, component_map, components, selected_policy, pixel_area_m2, rejections)
    for target_index in range(len(class_codes)):
        proposals.extend(_b._bridge_proposals_for_code(target_index, baseline, valid, probs, class_codes, component_map, components, selected_policy, pixel_area_m2, sizes, rejections))
    multi, multi_audit = _multi_neighbour_proposals(baseline, valid, probs, conf, class_codes, component_map, components, selected_policy, pixel_area_m2, sizes, totals, rejections)
    b_count = len(proposals)
    proposals.extend(multi)
    working, accepted, skipped, decisions, rollback, interactions, duplicate_count = _b._adjudicate_v31b(proposals, baseline, class_codes, valid, selected_policy, pixel_area_m2, component_map, components, budget_mask)
    result = baseline.copy(); result[budget_mask] = working[budget_mask]
    if not np.array_equal(result[~valid], baseline[~valid]) or np.any(result[valid] < 0) or np.any(result[valid] >= len(class_codes)):
        raise _b.CandidateError("candidate violated single-label or invalid-pixel preservation")
    if not _b._final_topology_holds(baseline, result, valid, class_codes, selected_policy, pixel_area_m2, accepted, components):
        raise _b.CandidateError("candidate violated final published topology")
    canonical, _duplicates, duplicate_audit = _b._canonicalize_v31b_proposals(proposals)
    before_dynamic, before_components = _b._dynamic_count(baseline, valid, class_codes, selected_policy, pixel_area_m2)
    after_dynamic, after_components = _b._dynamic_count(result, valid, class_codes, selected_policy, pixel_area_m2)
    before_by_class = _b._per_class_metrics(baseline, valid, class_codes, selected_policy, pixel_area_m2)
    after_by_class = _b._per_class_metrics(result, valid, class_codes, selected_policy, pixel_area_m2)
    source_loss: Counter[int] = Counter(); target_gain: Counter[int] = Counter(); bridge_gain: Counter[int] = Counter()
    for proposal in accepted:
        rows, cols = proposal.footprint[:, 0], proposal.footprint[:, 1]; core = budget_mask[rows, cols]
        source_loss.update(int(class_codes[int(value)]) for value in baseline[rows[core], cols[core]])
        target_gain[proposal.target_code] += int(np.count_nonzero(core))
        if proposal.kind == "same_class_bridge": bridge_gain[proposal.target_code] += int(np.count_nonzero(core))
    summaries = []
    for proposal in canonical:
        summary = _b._summary(proposal, decision=decisions.get(proposal.proposal_id, ("generated", "unresolved"))[0], reason=decisions.get(proposal.proposal_id, ("generated", "unresolved"))[1]); summary["occurrence_id"] = f"{proposal.proposal_id}:canonical"; summaries.append(summary)
    limit = len(summaries) if full_audit else max(0, int(selected_policy.audit_proposal_limit))
    changed = valid & (result != baseline)
    report: dict[str, Any] = {
        "policy_snapshot": policy_snapshot(selected_policy), "policy_snapshot_sha256": policy_snapshot_sha256(selected_policy), "policy_id": selected_policy.policy_id, "policy_version": selected_policy.policy_version,
        "candidate_label": "V3.2", "adjudication_mode": V32_ADJUDICATION_MODE, "baseline_kind": baseline_kind, "confidence_semantics": "explicit confidence if supplied else P[baseline_label]", "confidence_source": "explicit" if conf is not None else "baseline_label_probability", "target_attachment_contract": "direct_residual_target_contact_for_each_baseline_target_component; same_class_bridge_retains_all_baseline_target_pixels", "incremental_metric_contract": "per_class_components_nonincreasing; global_components_and_dynamic_nonincreasing", "single_pass_from_frozen_baseline": True, "cascade_generation": False, "topology_connectivity": 4, "single_label": True, "gap_pixels": 0, "overlap_pixels": 0, "outside_pixels": 0,
        "frozen_global_class_pixel_totals": {str(code): totals[code] for code in sorted(totals)}, "frozen_global_class_pixel_totals_sum": int(sum(totals.values())), "frozen_global_class_pixel_totals_sha256": _sha256_json({str(code): totals[code] for code in sorted(totals)}), "frozen_totals_contract": "full_owner_core_strict_valid_v3_baseline_once; immutable_during_adjudication",
        "baseline_mask_sha256": hashlib.sha256(np.ascontiguousarray(baseline).tobytes()).hexdigest(), "output_mask_sha256": hashlib.sha256(np.ascontiguousarray(result).tobytes()).hexdigest(), "valid_mask_sha256": hashlib.sha256(np.ascontiguousarray(valid).tobytes()).hexdigest(), "class_budget_mask_sha256": hashlib.sha256(np.ascontiguousarray(budget_mask).tobytes()).hexdigest(), "class_budget_mask_pixel_count": int(budget_mask.sum()), "class_budget_mask_semantics": "complete_proposal_core_owner_and_valid_mask; none_means_valid_mask",
        "class_codes": [int(code) for code in class_codes], "physical_metrics": {"pixel_area_m2": float(pixel_area_m2), "row_step_m": float(sizes[0]), "column_step_m": float(sizes[1]), "source": "explicit_caller_supplied_physical_metrics"}, "baseline": {"components_4_connected": before_components, "dynamic_fragments_4_connected": before_dynamic}, "result": {"components_4_connected": after_components, "dynamic_fragments_4_connected": after_dynamic}, "changed_pixel_count": int(np.count_nonzero(changed)), "changed_area_m2": float(np.count_nonzero(changed) * pixel_area_m2), "protected_source_loss_pixel_count": int(sum(source_loss[code] for code in selected_policy.protected_source_codes)), "protected_source_retention": 1.0 if sum(source_loss[code] for code in selected_policy.protected_source_codes) == 0 else 0.0, "source_split_violation_count": 0,
        "proposals_generated": len(proposals), "raw_generated": len(proposals), "b_original_proposals_generated": b_count, "closed_multi_neighbor_rarity_proposals_generated": len(multi), "proposals_canonical": len(canonical), "duplicate_proposal_count": duplicate_count, "proposals_accepted": len(accepted), "proposal_generation_reject_reason_counts": dict(sorted(rejections.items())), "proposal_reject_reason_counts": dict(sorted(skipped.items())), "proposals_rejected_by_adjudication": dict(sorted(skipped.items())), "closed_multi_neighbor_rarity_generation_audit": multi_audit if full_audit else multi_audit[:limit], "target_selection_contract": "global_class_pixel_total_ascending_then_mean_target_probability_descending_then_class_code_ascending; contact_edges_and_lengths_audit_only", "proposal_audit": summaries[:limit], "accepted": [_b._summary(item, decision="accepted", reason="selected") for item in accepted][:limit], "interaction_audit": interactions if full_audit else interactions[:limit], "duplicate_proposal_audit": duplicate_audit if full_audit else duplicate_audit[:limit], "raw_proposal_audit": (sorted([*summaries, *duplicate_audit], key=lambda item: str(item["occurrence_id"])) if full_audit else sorted([*summaries, *duplicate_audit], key=lambda item: str(item["occurrence_id"]))[:limit]), "full_audit": bool(full_audit), "audit_truncated": not full_audit and (len(summaries) > limit or len(multi_audit) > limit or len(interactions) > limit or len(summaries) + len(duplicate_audit) > limit), "final_topology_rollback": int(rollback), "valid_pixel_count": int(valid.sum()),
        "per_class": {str(code): {"baseline": before_by_class[int(code)], "result": after_by_class[int(code)], "source_loss": int(source_loss[int(code)]), "target_gain": int(target_gain[int(code)]), "bridge_gain": int(bridge_gain[int(code)]), "net_pixel_drift": int(target_gain[int(code)] - source_loss[int(code)])} for code in class_codes},
        "class_budget_pixels": {str(code): {"baseline": int(np.sum(baseline[budget_mask] == index)), "denominator": int(np.sum(baseline[budget_mask] == index)), "source_loss": int(source_loss[int(code)]), "target_gain": int(target_gain[int(code)]), "protected_bridge_gain": int(bridge_gain[int(code)] if int(code) in selected_policy.protected_source_codes else 0), "source_loss_limit": float(0.0 if int(code) in selected_policy.protected_source_codes else np.sum(baseline[budget_mask] == index) * selected_policy.maximum_source_loss_fraction), "target_gain_limit": float(np.sum(baseline[budget_mask] == index) * (selected_policy.protected_bridge_gain_fraction if int(code) in selected_policy.protected_source_codes else selected_policy.maximum_target_gain_fraction)), "protected_bridge_gain_limit": float(np.sum(baseline[budget_mask] == index) * selected_policy.protected_bridge_gain_fraction if int(code) in selected_policy.protected_source_codes else 0.0)} for index, code in enumerate(class_codes)},
    }
    report["audit_sha256"] = _sha256_json(report)
    return result, report


apply_v32 = apply_v32_candidate
