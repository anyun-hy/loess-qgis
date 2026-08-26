"""Isolated, bounded second-round candidate built on a frozen V3.1-B result.

The second round may use B-created target pixels as topology anchors, but it
may never change a pixel already changed by B.  Class budgets remain cumulative
against the original V3 owner-Core denominator.  Exactly one additional
proposal-generation/adjudication pass is allowed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from . import candidate as _candidate


V31D_POLICY_ID = "fragmentation_v31d_bounded_second_round_candidate_v1"
V31D_POLICY_VERSION = "v31d_bounded_second_round_20260825"
V31D_ADJUDICATION_MODE = "bounded_second_round_dependency_incremental_v1"


def v31d_policy() -> _candidate.CandidatePolicy:
    """Reuse B's class/evidence policy under a distinct experiment identity."""

    return replace(
        _candidate.v31b_policy(),
        policy_id=V31D_POLICY_ID,
        policy_version=V31D_POLICY_VERSION,
    )


def _integer_count_map(
    values: Mapping[int, int], class_codes: Sequence[int], name: str
) -> Counter[int]:
    expected = {int(code) for code in class_codes}
    if {int(code) for code in values} != expected:
        raise _candidate.CandidateError(f"{name} must contain every class code exactly once")
    result: Counter[int] = Counter()
    for raw_code, raw_value in values.items():
        if isinstance(raw_value, bool) or int(raw_value) != raw_value or int(raw_value) < 0:
            raise _candidate.CandidateError(f"{name} values must be non-negative integers")
        result[int(raw_code)] = int(raw_value)
    return result


def _adjudicate_second_round(
    proposals: Sequence[_candidate._Proposal],
    labels: np.ndarray,
    class_codes: Sequence[int],
    valid: np.ndarray,
    policy: _candidate.CandidatePolicy,
    pixel_area_m2: float,
    component_map: np.ndarray,
    components: Sequence[_candidate._Component],
    budget_mask: np.ndarray,
    immutable: np.ndarray,
    dependency_lock: np.ndarray,
    original_totals: Counter[int],
    round1_source_loss: Counter[int],
    round1_target_gain: Counter[int],
) -> tuple[
    np.ndarray,
    list[_candidate._Proposal],
    Counter[str],
    dict[str, tuple[str, str]],
    list[dict[str, Any]],
    int,
    Counter[int],
    Counter[int],
]:
    """B's incremental adjudication with immutable round 1 and cumulative budgets."""

    source_loss = round1_source_loss.copy()
    target_gain = round1_target_gain.copy()
    round2_source_loss: Counter[int] = Counter()
    round2_target_gain: Counter[int] = Counter()
    accepted: list[_candidate._Proposal] = []
    occupied = np.zeros(labels.shape, dtype=bool)
    result = labels.copy()
    skipped: Counter[str] = Counter()
    decisions: dict[str, tuple[str, str]] = {}
    components_by_id = {item.component_id: item for item in components}
    metric_state = _candidate._incremental_metric_state(components, policy, pixel_area_m2)
    proposal_pixel_roots: dict[tuple[int, int], int] = {}
    ordered, duplicates, duplicate_audit = _candidate._canonicalize_v31b_proposals(proposals)
    skipped.update(duplicates)
    removed_by_component: dict[int, set[tuple[int, int]]] = {}
    target_dependents: dict[int, list[_candidate._Proposal]] = {}
    accepted_by_id: dict[str, _candidate._Proposal] = {}
    proposal_pixel_owner: dict[tuple[int, int], str] = {}
    interaction_audit: list[dict[str, Any]] = []
    rank_groups: dict[tuple[float, float, float, float, str], list[_candidate._Proposal]] = {}
    for proposal in ordered:
        rank_groups.setdefault(_candidate._rank_key(proposal), []).append(proposal)
    ambiguous = {
        proposal.proposal_id
        for group in rank_groups.values()
        if len({item.target_index for item in group}) > 1
        for proposal in group
    }
    for proposal in ordered:
        rows, cols = proposal.footprint[:, 0], proposal.footprint[:, 1]
        interaction: dict[str, Any] = {
            "proposal_id": proposal.proposal_id,
            "source_component_ids": list(proposal.source_component_ids),
            "affected_accepted_proposal_ids": [],
            "target_attachment_checks": 0,
        }
        if np.any(immutable[rows, cols]):
            reason = "round1_immutable"
        elif np.any(dependency_lock[rows, cols]):
            reason = "round1_dependency_lock"
        elif not np.all(budget_mask[rows, cols]):
            reason = "outside_core_owner"
        elif proposal.proposal_id in ambiguous:
            reason = "ambiguous_target_tie"
        elif np.any(occupied[rows, cols]):
            reason = "footprint_conflict"
        else:
            changed_codes = Counter(
                int(class_codes[int(value)]) for value in labels[rows, cols]
            )
            target_budget_fraction = (
                policy.protected_bridge_gain_fraction
                if proposal.kind == "same_class_bridge"
                and proposal.target_code in policy.protected_source_codes
                else policy.maximum_target_gain_fraction
            )
            if any(
                source_loss[code] + count
                > original_totals[code] * policy.maximum_source_loss_fraction + 1e-12
                for code, count in changed_codes.items()
            ):
                reason = "cumulative_source_budget"
            elif (
                target_gain[proposal.target_code] + len(proposal.footprint)
                > original_totals[proposal.target_code] * target_budget_fraction + 1e-12
            ):
                reason = "cumulative_target_budget"
            else:
                affected_components = tuple(
                    sorted(set(int(value) for value in component_map[rows, cols] if value))
                )
                tentative_removed = dict(removed_by_component)
                for row, col, component_id in zip(rows, cols, component_map[rows, cols]):
                    if component_id:
                        tentative_removed[int(component_id)] = set(
                            tentative_removed.get(int(component_id), set())
                        )
                        tentative_removed[int(component_id)].add((int(row), int(col)))
                if not _candidate._source_connectivity_safe_incremental(
                    tentative_removed,
                    affected_components,
                    component_map,
                    components_by_id,
                ):
                    reason = "source_connectivity"
                else:
                    metric_plan, metric_reason = _candidate._prospective_metric_plan(
                        metric_state,
                        proposal,
                        component_map,
                        labels,
                        result,
                        proposal_pixel_roots,
                        policy,
                        pixel_area_m2,
                    )
                    if metric_reason is not None:
                        reason = metric_reason
                    else:
                        affected = {
                            item.proposal_id: item
                            for component_id in affected_components
                            for item in target_dependents.get(component_id, [])
                        }
                        checked = [
                            *sorted(affected.values(), key=lambda item: item.proposal_id),
                            proposal,
                        ]
                        interaction["affected_accepted_proposal_ids"] = [
                            item.proposal_id for item in checked if item is not proposal
                        ]
                        interaction["target_attachment_checks"] = len(checked)
                        result[rows, cols] = proposal.target_index
                        transient_by_id = dict(accepted_by_id)
                        transient_by_id[proposal.proposal_id] = proposal
                        for row, col in proposal.footprint:
                            proposal_pixel_owner[(int(row), int(col))] = proposal.proposal_id
                        target_safe = all(
                            _candidate._target_attachment_safe_incremental(
                                item,
                                result,
                                labels,
                                component_map,
                                components_by_id,
                                transient_by_id,
                                proposal_pixel_owner,
                                target_dependents,
                            )
                            for item in checked
                        )
                        if not target_safe:
                            result[rows, cols] = labels[rows, cols]
                            for row, col in proposal.footprint:
                                proposal_pixel_owner.pop((int(row), int(col)), None)
                            reason = "target_attachment"
                        else:
                            reason = "selected"
                            removed_by_component = tentative_removed
        if reason != "selected":
            skipped[reason] += 1
            decisions[proposal.proposal_id] = ("rejected", reason)
            interaction["decision"] = "rejected"
            interaction["reason"] = reason
            interaction_audit.append(interaction)
            continue
        occupied[rows, cols] = True
        accepted.append(proposal)
        decisions[proposal.proposal_id] = ("accepted", "selected")
        source_loss.update(changed_codes)
        round2_source_loss.update(changed_codes)
        target_gain[proposal.target_code] += len(proposal.footprint)
        round2_target_gain[proposal.target_code] += len(proposal.footprint)
        representative = _candidate._commit_metric_plan(metric_state, metric_plan)
        for row, col in proposal.footprint:
            proposal_pixel_roots[(int(row), int(col))] = representative
        for component_id in proposal.baseline_target_component_ids:
            target_dependents.setdefault(int(component_id), []).append(proposal)
        accepted_by_id[proposal.proposal_id] = proposal
        interaction["decision"] = "accepted"
        interaction["reason"] = "selected"
        interaction_audit.append(interaction)
    interaction_audit.extend(duplicate_audit)
    return (
        result,
        accepted,
        skipped,
        decisions,
        interaction_audit,
        int(duplicates["duplicate_proposal"]),
        round2_source_loss,
        round2_target_gain,
    )


def apply_v31d_candidate(
    labels: np.ndarray,
    *,
    original_v3_labels: np.ndarray,
    round1_immutable_mask: np.ndarray,
    round1_source_loss_pixels: Mapping[int, int],
    round1_target_gain_pixels: Mapping[int, int],
    round1_protected_bridge_gain_pixels: Mapping[int, int],
    class_codes: Sequence[int],
    pixel_area_m2: float,
    pixel_size_m: tuple[float, float] | None = None,
    valid_mask: np.ndarray | None = None,
    class_budget_mask: np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    policy: _candidate.CandidatePolicy | None = None,
    baseline_kind: str = "v31b_cleaned",
    full_audit: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply exactly one immutable, cumulative-budget second round to B."""

    if baseline_kind != "v31b_cleaned":
        raise _candidate.CandidateError("V3.1-D baseline_kind must be v31b_cleaned")
    selected_policy = policy or v31d_policy()
    baseline, valid, probs, conf, sizes = _candidate._validate(
        labels,
        class_codes,
        valid_mask,
        probabilities,
        confidence,
        pixel_area_m2,
        pixel_size_m,
        selected_policy,
    )
    original = np.asarray(original_v3_labels)
    if original.shape != baseline.shape or not np.issubdtype(original.dtype, np.integer):
        raise _candidate.CandidateError("original_v3_labels must be an integer array matching labels")
    original = original.astype(np.int16, copy=False)
    if np.any(valid & ((original < 0) | (original >= len(class_codes)))):
        raise _candidate.CandidateError("original_v3_labels contain invalid class indices")
    immutable = np.asarray(round1_immutable_mask, dtype=bool)
    if immutable.shape != baseline.shape:
        raise _candidate.CandidateError("round1_immutable_mask shape does not match labels")
    expected_immutable = valid & (baseline != original)
    if not np.array_equal(immutable & valid, expected_immutable):
        raise _candidate.CandidateError(
            "round1_immutable_mask must exactly equal valid B-to-V3 changed pixels"
        )
    budget_mask = _candidate._class_budget_mask(class_budget_mask, valid)
    original_totals = Counter(
        int(class_codes[int(value)]) for value in original[budget_mask]
    )
    round1_source = _integer_count_map(
        round1_source_loss_pixels, class_codes, "round1_source_loss_pixels"
    )
    round1_target = _integer_count_map(
        round1_target_gain_pixels, class_codes, "round1_target_gain_pixels"
    )
    round1_protected_bridge = _integer_count_map(
        round1_protected_bridge_gain_pixels,
        class_codes,
        "round1_protected_bridge_gain_pixels",
    )
    owner_changed = budget_mask & (baseline != original)
    computed_round1_source = Counter(
        int(class_codes[int(value)]) for value in original[owner_changed]
    )
    computed_round1_target = Counter(
        int(class_codes[int(value)]) for value in baseline[owner_changed]
    )
    if any(
        round1_source[int(code)] != computed_round1_source[int(code)]
        for code in class_codes
    ):
        raise _candidate.CandidateError(
            "round1_source_loss_pixels disagree with V3-to-B owner transitions"
        )
    if any(
        round1_target[int(code)] != computed_round1_target[int(code)]
        for code in class_codes
    ):
        raise _candidate.CandidateError(
            "round1_target_gain_pixels disagree with V3-to-B owner transitions"
        )
    for code in class_codes:
        code = int(code)
        source_limit = (
            0.0
            if code in selected_policy.protected_source_codes
            else original_totals[code] * selected_policy.maximum_source_loss_fraction
        )
        target_fraction = (
            selected_policy.protected_bridge_gain_fraction
            if code in selected_policy.protected_source_codes
            else selected_policy.maximum_target_gain_fraction
        )
        if round1_source[code] > source_limit + 1e-12:
            raise _candidate.CandidateError(f"round1 source loss already exceeds class {code} budget")
        if round1_target[code] > original_totals[code] * target_fraction + 1e-12:
            raise _candidate.CandidateError(f"round1 target gain already exceeds class {code} budget")
        expected_protected_bridge = (
            round1_target[code] if code in selected_policy.protected_source_codes else 0
        )
        if round1_protected_bridge[code] != expected_protected_bridge:
            raise _candidate.CandidateError(
                f"round1 protected bridge ledger disagrees for class {code}"
            )
    component_map, components = _candidate._component_index(baseline, valid, class_codes)
    changed_target_component_ids = {
        int(value) for value in component_map[immutable & valid] if int(value) > 0
    }
    dependency_lock = np.isin(
        component_map,
        np.fromiter(sorted(changed_target_component_ids), dtype=np.int32),
    )
    generation_rejections: Counter[str] = Counter()
    proposals = _candidate._island_proposals(
        baseline,
        valid,
        probs,
        conf,
        class_codes,
        component_map,
        components,
        selected_policy,
        pixel_area_m2,
        generation_rejections,
    )
    for target_index in range(len(class_codes)):
        proposals.extend(
            _candidate._bridge_proposals_for_code(
                target_index,
                baseline,
                valid,
                probs,
                class_codes,
                component_map,
                components,
                selected_policy,
                pixel_area_m2,
                sizes,
                generation_rejections,
            )
        )
    (
        working,
        accepted,
        skipped,
        decisions,
        interactions,
        duplicate_count,
        round2_source,
        round2_target,
    ) = _adjudicate_second_round(
        proposals,
        baseline,
        class_codes,
        valid,
        selected_policy,
        pixel_area_m2,
        component_map,
        components,
        budget_mask,
        immutable,
        dependency_lock,
        original_totals,
        round1_source,
        round1_target,
    )
    result = baseline.copy()
    result[budget_mask] = working[budget_mask]
    eligibility = budget_mask & ~immutable & ~dependency_lock
    if np.any(result[valid & ~eligibility] != baseline[valid & ~eligibility]):
        raise _candidate.CandidateError("second round changed a non-eligible pixel")
    if not _candidate._final_topology_holds(
        baseline,
        result,
        valid,
        class_codes,
        selected_policy,
        pixel_area_m2,
        accepted,
        components,
    ):
        raise _candidate.CandidateError("second round violated final B-relative topology")
    before_dynamic, before_components = _candidate._dynamic_count(
        baseline, valid, class_codes, selected_policy, pixel_area_m2
    )
    after_dynamic, after_components = _candidate._dynamic_count(
        result, valid, class_codes, selected_policy, pixel_area_m2
    )
    canonical, _duplicates, duplicate_audit = _candidate._canonicalize_v31b_proposals(
        proposals
    )
    all_summaries = []
    for proposal in canonical:
        summary = _candidate._summary(
            proposal,
            decision=decisions.get(proposal.proposal_id, ("generated", "unresolved"))[0],
            reason=decisions.get(proposal.proposal_id, ("generated", "unresolved"))[1],
        )
        summary["occurrence_id"] = f"{proposal.proposal_id}:canonical"
        all_summaries.append(summary)
    limit = len(all_summaries) if full_audit else max(0, int(selected_policy.audit_proposal_limit))
    changed = valid & (result != baseline)
    report: dict[str, Any] = {
        "policy_id": selected_policy.policy_id,
        "policy_version": selected_policy.policy_version,
        "adjudication_mode": V31D_ADJUDICATION_MODE,
        "baseline_kind": baseline_kind,
        "round_count": 2,
        "additional_generation_rounds": 1,
        "cascade_generation": "one_bounded_additional_round",
        "round1_immutable_pixel_count": int(np.count_nonzero(immutable & valid)),
        "round1_immutable_preserved": True,
        "round1_dependency_component_count": len(changed_target_component_ids),
        "round1_dependency_lock_pixel_count": int(np.count_nonzero(dependency_lock & valid)),
        "round1_dependency_lock_preserved": True,
        "single_label": True,
        "gap_pixels": 0,
        "overlap_pixels": 0,
        "outside_pixels": 0,
        "baseline_mask_sha256": hashlib.sha256(np.ascontiguousarray(baseline).tobytes()).hexdigest(),
        "original_v3_mask_sha256": hashlib.sha256(np.ascontiguousarray(original).tobytes()).hexdigest(),
        "output_mask_sha256": hashlib.sha256(np.ascontiguousarray(result).tobytes()).hexdigest(),
        "valid_mask_sha256": hashlib.sha256(np.ascontiguousarray(valid).tobytes()).hexdigest(),
        "class_budget_mask_sha256": hashlib.sha256(np.ascontiguousarray(budget_mask).tobytes()).hexdigest(),
        "physical_metrics": {
            "pixel_area_m2": float(pixel_area_m2),
            "row_step_m": float(sizes[0]),
            "column_step_m": float(sizes[1]),
        },
        "baseline": {
            "components_4_connected": before_components,
            "dynamic_fragments_4_connected": before_dynamic,
        },
        "result": {
            "components_4_connected": after_components,
            "dynamic_fragments_4_connected": after_dynamic,
        },
        "changed_pixel_count": int(np.count_nonzero(changed)),
        "changed_area_m2": float(np.count_nonzero(changed) * pixel_area_m2),
        "raw_generated": len(proposals),
        "proposals_canonical": len(canonical),
        "duplicate_proposal_count": duplicate_count,
        "proposals_accepted": len(accepted),
        "proposal_generation_reject_reason_counts": dict(sorted(generation_rejections.items())),
        "proposal_reject_reason_counts": dict(sorted(skipped.items())),
        "proposal_audit": all_summaries[:limit],
        "interaction_audit": interactions[:limit],
        "duplicate_proposal_audit": duplicate_audit if full_audit else duplicate_audit[:limit],
        "full_audit": bool(full_audit),
        "audit_truncated": not full_audit
        and (len(all_summaries) > limit or len(interactions) > limit),
        "cumulative_class_budget_pixels": {
            str(code): {
                "original_v3_denominator": int(original_totals[int(code)]),
                "round1_source_loss": int(round1_source[int(code)]),
                "round2_source_loss": int(round2_source[int(code)]),
                "cumulative_source_loss": int(
                    round1_source[int(code)] + round2_source[int(code)]
                ),
                "round1_target_gain": int(round1_target[int(code)]),
                "round2_target_gain": int(round2_target[int(code)]),
                "cumulative_target_gain": int(
                    round1_target[int(code)] + round2_target[int(code)]
                ),
                "source_loss_limit": float(
                    0.0
                    if int(code) in selected_policy.protected_source_codes
                    else original_totals[int(code)]
                    * selected_policy.maximum_source_loss_fraction
                ),
                "target_gain_limit": float(
                    original_totals[int(code)]
                    * (
                        selected_policy.protected_bridge_gain_fraction
                        if int(code) in selected_policy.protected_source_codes
                        else selected_policy.maximum_target_gain_fraction
                    )
                ),
                "round1_protected_bridge_gain": int(
                    round1_protected_bridge[int(code)]
                ),
                "round2_protected_bridge_gain": int(
                    round2_target[int(code)]
                    if int(code) in selected_policy.protected_source_codes
                    else 0
                ),
                "cumulative_protected_bridge_gain": int(
                    round1_protected_bridge[int(code)]
                    + (
                        round2_target[int(code)]
                        if int(code) in selected_policy.protected_source_codes
                        else 0
                    )
                ),
            }
            for code in class_codes
        },
    }
    report["audit_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return result, report


apply_v31d = apply_v31d_candidate
