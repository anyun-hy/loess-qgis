"""Isolated V3.4: one frozen second pass over an immutable V3.3 result.

V3.3-changed pixels are immutable.  Source loss, target gain, and protected
same-class bridge gain remain cumulative against the original V3 owner-Core
denominator.  The live V3.3 YAML still controls permissions, thresholds,
scenario routing, rarity, and proposal priority.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from fragmentation_policy import load_policy, policy_sha256 as config_sha256
from fragmentation_policy.loader import policy_snapshot as config_snapshot
from fragmentation_v33_candidate import candidate as _v33


V34_POLICY_ID = "fragmentation_v34_bounded_second_pass_candidate_v1"
V34_POLICY_VERSION = "v34_20260826"
V34_ADJUDICATION_MODE = "v33_config_cumulative_bounded_second_pass_v1"


class V34CandidateError(RuntimeError):
    """The V3.4 input or a cumulative hard contract is invalid."""


def _sha_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def policy_snapshot(document: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = config_snapshot(document or load_policy())
    return {
        "policy_id": V34_POLICY_ID,
        "policy_version": V34_POLICY_VERSION,
        "config_policy": config,
        "config_policy_sha256": config_sha256(config),
        "executor_contract": {
            "baseline": "one_frozen_complete_v33_publication",
            "additional_generation_rounds": 1,
            "round1_changed_pixels": "immutable",
            "round1_target_components": "dependency_locked_as_sources_but_available_as_targets",
            "budget_denominator": "original_v3_owner_core",
            "source_and_target_budget": "cumulative_two_percent",
            "protected_same_class_bridge_budget": "cumulative_one_percent",
            "scenario_and_rank": "live_v33_configuration",
        },
    }


def policy_snapshot_sha256(document: Mapping[str, Any] | None = None) -> str:
    return _sha_json(policy_snapshot(document))


def _count_map(
    values: Mapping[int, int], class_codes: Sequence[int], name: str,
) -> Counter[int]:
    expected = {int(code) for code in class_codes}
    if {int(code) for code in values} != expected:
        raise V34CandidateError(f"{name} must contain every class code exactly once")
    result: Counter[int] = Counter()
    for raw_code, raw_value in values.items():
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, np.integer)) or int(raw_value) < 0:
            raise V34CandidateError(f"{name} values must be non-negative integers")
        result[int(raw_code)] = int(raw_value)
    return result


def apply_v34_candidate(
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
    policy_document: Mapping[str, Any] | None = None,
    baseline_kind: str = "v33_cleaned",
    full_audit: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply exactly one cumulative, immutable second pass to V3.3."""

    if baseline_kind != "v33_cleaned":
        raise V34CandidateError("baseline_kind must be v33_cleaned")
    config = config_snapshot(policy_document or load_policy())
    policy = _v33.runtime_policy(config)
    if tuple(int(code) for code in class_codes) != tuple(sorted(policy.class_policies)):
        raise V34CandidateError("class_codes must exactly match the configured classes")
    try:
        baseline, valid, probs, _conf, sizes = _v33._b._validate(
            labels, class_codes, valid_mask, probabilities, confidence,
            pixel_area_m2, pixel_size_m, policy,
        )
        budget = _v33._b._class_budget_mask(class_budget_mask, valid)
    except Exception as exc:
        raise V34CandidateError(str(exc)) from exc
    original = np.asarray(original_v3_labels)
    if original.shape != baseline.shape or not np.issubdtype(original.dtype, np.integer):
        raise V34CandidateError("original_v3_labels must be an integer array matching labels")
    original = original.astype(np.int16, copy=False)
    if np.any(valid & ((original < 0) | (original >= len(class_codes)))):
        raise V34CandidateError("original_v3_labels contain invalid class indices")
    immutable = np.asarray(round1_immutable_mask, dtype=bool)
    if immutable.shape != baseline.shape or not np.array_equal(
        immutable & valid, valid & (baseline != original),
    ):
        raise V34CandidateError(
            "round1_immutable_mask must exactly equal valid V3-to-V3.3 changed pixels"
        )
    original_totals = Counter(
        int(class_codes[int(value)]) for value in original[budget]
    )
    round1_source = _count_map(
        round1_source_loss_pixels, class_codes, "round1_source_loss_pixels",
    )
    round1_target = _count_map(
        round1_target_gain_pixels, class_codes, "round1_target_gain_pixels",
    )
    round1_bridge = _count_map(
        round1_protected_bridge_gain_pixels,
        class_codes,
        "round1_protected_bridge_gain_pixels",
    )
    owner_changed = budget & (baseline != original)
    computed_source = Counter(
        int(class_codes[int(value)]) for value in original[owner_changed]
    )
    computed_target = Counter(
        int(class_codes[int(value)]) for value in baseline[owner_changed]
    )
    for code in class_codes:
        code = int(code)
        if round1_source[code] != computed_source[code]:
            raise V34CandidateError(f"round1 source ledger disagrees for class {code}")
        if round1_target[code] != computed_target[code]:
            raise V34CandidateError(f"round1 target ledger disagrees for class {code}")
        source_limit = (
            0.0 if code in policy.protected_source_codes
            else original_totals[code] * policy.maximum_source_loss_fraction
        )
        if round1_source[code] > source_limit + 1e-12:
            raise V34CandidateError(f"round1 source loss exceeds class {code} budget")
        if round1_target[code] > original_totals[code] * policy.maximum_target_gain_fraction + 1e-12:
            raise V34CandidateError(f"round1 target gain exceeds class {code} budget")
        if round1_bridge[code] > round1_target[code]:
            raise V34CandidateError(f"round1 bridge gain exceeds target gain for class {code}")
        if code in policy.protected_source_codes:
            if round1_bridge[code] > original_totals[code] * policy.protected_bridge_gain_fraction + 1e-12:
                raise V34CandidateError(f"round1 protected bridge gain exceeds class {code} budget")
        elif round1_bridge[code] != 0:
            raise V34CandidateError(f"round1 protected bridge ledger must be zero for class {code}")

    component_map, components = _v33._b._component_index(baseline, valid, class_codes)
    changed_target_ids = {
        int(value) for value in component_map[immutable & valid] if int(value) > 0
    }
    dependency_lock = np.isin(
        component_map,
        np.fromiter(sorted(changed_target_ids), dtype=np.int32),
    )
    generation_rejections: Counter[str] = Counter()
    proposals, generation_audit = _v33._generate_proposals(
        baseline, valid, probs, class_codes, component_map, components, policy,
        config, pixel_area_m2, sizes, generation_rejections,
    )
    working, accepted, skipped, decisions, interactions, duplicate_count = _v33._adjudicate(
        proposals, baseline, class_codes, valid, policy, pixel_area_m2, sizes,
        component_map, components, budget, config,
        original_budget_totals=original_totals,
        prior_source_loss=round1_source,
        prior_target_gain=round1_target,
        prior_protected_bridge_gain=round1_bridge,
        immutable_mask=immutable,
        dependency_lock_mask=dependency_lock,
        cumulative_budget_mode=True,
    )
    published = baseline.copy()
    published[budget] = working[budget]
    if np.any(published[valid & (~budget | immutable | dependency_lock)] != baseline[valid & (~budget | immutable | dependency_lock)]):
        raise V34CandidateError("second pass changed a non-eligible pixel")
    if not _v33._b._final_topology_holds(
        baseline, published, valid, class_codes, policy, pixel_area_m2,
        accepted, components,
    ):
        raise V34CandidateError("second pass violated final topology")
    before_boundary = _v33._boundary_metrics(baseline, valid, sizes)
    after_boundary = _v33._boundary_metrics(published, valid, sizes)
    if any(float(after_boundary[key]) > float(before_boundary[key]) + 1e-9 for key in before_boundary):
        raise V34CandidateError("second pass increased boundary")
    before_dynamic, before_components = _v33._b._dynamic_count(
        baseline, valid, class_codes, policy, pixel_area_m2,
    )
    after_dynamic, after_components = _v33._b._dynamic_count(
        published, valid, class_codes, policy, pixel_area_m2,
    )
    before_by_class = _v33._b._per_class_metrics(
        baseline, valid, class_codes, policy, pixel_area_m2,
    )
    after_by_class = _v33._b._per_class_metrics(
        published, valid, class_codes, policy, pixel_area_m2,
    )
    if any(
        int(after_by_class[int(code)]["component_count_4_connected"])
        > int(before_by_class[int(code)]["component_count_4_connected"])
        for code in class_codes
    ):
        raise V34CandidateError("second pass increased per-class components")
    if sum(float(row["dynamic_fragment_area_m2"]) for row in after_by_class.values()) > sum(float(row["dynamic_fragment_area_m2"]) for row in before_by_class.values()) + 1e-9:
        raise V34CandidateError("second pass increased dynamic-fragment area")

    round2_source: Counter[int] = Counter()
    round2_target: Counter[int] = Counter()
    round2_bridge: Counter[int] = Counter()
    for proposal in accepted:
        rows, cols = proposal.footprint[:, 0], proposal.footprint[:, 1]
        round2_source.update(int(class_codes[int(value)]) for value in baseline[rows, cols])
        round2_target[int(proposal.target_code)] += len(rows)
        if proposal.kind == "same_class_bridge" and proposal.target_code in policy.protected_source_codes:
            round2_bridge[int(proposal.target_code)] += len(rows)
    cumulative_changed = budget & (published != original)
    actual_source = Counter(
        int(class_codes[int(value)]) for value in original[cumulative_changed]
    )
    actual_target = Counter(
        int(class_codes[int(value)]) for value in published[cumulative_changed]
    )
    for code in class_codes:
        code = int(code)
        if actual_source[code] != round1_source[code] + round2_source[code]:
            raise V34CandidateError(f"cumulative source ledger does not close for class {code}")
        if actual_target[code] != round1_target[code] + round2_target[code]:
            raise V34CandidateError(f"cumulative target ledger does not close for class {code}")
    canonical, _duplicates, duplicate_audit = _v33._canonicalize(proposals, config)
    summaries = [
        _v33._summary(
            item,
            decisions.get(item.proposal_id, ("generated", "unresolved")),
            config,
        )
        for item in canonical
    ]
    changed = valid & (published != baseline)
    report: dict[str, Any] = {
        "candidate_label": "V3.4",
        "policy_id": V34_POLICY_ID,
        "policy_version": V34_POLICY_VERSION,
        "policy_snapshot": policy_snapshot(config),
        "policy_snapshot_sha256": policy_snapshot_sha256(config),
        "config_policy_sha256": config_sha256(config),
        "adjudication_mode": V34_ADJUDICATION_MODE,
        "baseline_kind": baseline_kind,
        "round_count": 2,
        "additional_generation_rounds": 1,
        "round1_immutable_pixel_count": int(np.count_nonzero(immutable & valid)),
        "round1_immutable_preserved": True,
        "round1_dependency_component_count": len(changed_target_ids),
        "round1_dependency_lock_pixel_count": int(np.count_nonzero(dependency_lock & valid)),
        "round1_dependency_lock_preserved": True,
        "single_label": True,
        "gap_pixels": 0,
        "overlap_pixels": 0,
        "outside_pixels": 0,
        "baseline": {
            "components_4_connected": before_components,
            "dynamic_fragments_4_connected": before_dynamic,
            "boundary": before_boundary,
        },
        "result": {
            "components_4_connected": after_components,
            "dynamic_fragments_4_connected": after_dynamic,
            "boundary": after_boundary,
        },
        "changed_pixel_count": int(np.count_nonzero(changed)),
        "changed_area_m2": float(np.count_nonzero(changed) * pixel_area_m2),
        "protected_source_loss_pixel_count": int(sum(round2_source[code] for code in policy.protected_source_codes)),
        "raw_generated": len(proposals),
        "proposals_canonical": len(canonical),
        "duplicate_proposal_count": duplicate_count,
        "proposals_accepted": len(accepted),
        "proposal_generation_reject_reason_counts": dict(sorted(generation_rejections.items())),
        "proposal_reject_reason_counts": dict(sorted(skipped.items())),
        "generation_audit": generation_audit if full_audit else [],
        "proposal_audit": summaries if full_audit else [],
        "accepted": [
            _v33._summary(item, ("accepted", "selected"), config)
            for item in accepted
        ] if full_audit else [],
        "interaction_audit": interactions if full_audit else [],
        "duplicate_proposal_audit": duplicate_audit if full_audit else [],
        "full_audit": bool(full_audit),
        "audit_truncated": not full_audit,
        "cumulative_class_budget_pixels": {
            str(code): {
                "original_v3_denominator": int(original_totals[int(code)]),
                "round1_source_loss": int(round1_source[int(code)]),
                "round2_source_loss": int(round2_source[int(code)]),
                "cumulative_source_loss": int(round1_source[int(code)] + round2_source[int(code)]),
                "source_loss_limit": float(
                    0.0 if int(code) in policy.protected_source_codes
                    else original_totals[int(code)] * policy.maximum_source_loss_fraction
                ),
                "round1_target_gain": int(round1_target[int(code)]),
                "round2_target_gain": int(round2_target[int(code)]),
                "cumulative_target_gain": int(round1_target[int(code)] + round2_target[int(code)]),
                "target_gain_limit": float(original_totals[int(code)] * policy.maximum_target_gain_fraction),
                "round1_protected_bridge_gain": int(round1_bridge[int(code)]),
                "round2_protected_bridge_gain": int(round2_bridge[int(code)]),
                "cumulative_protected_bridge_gain": int(round1_bridge[int(code)] + round2_bridge[int(code)]),
                "protected_bridge_gain_limit": float(
                    original_totals[int(code)] * policy.protected_bridge_gain_fraction
                    if int(code) in policy.protected_source_codes else 0.0
                ),
            }
            for code in class_codes
        },
        "per_class": {
            str(code): {
                "baseline": before_by_class[int(code)],
                "result": after_by_class[int(code)],
                "round2_source_loss": int(round2_source[int(code)]),
                "round2_target_gain": int(round2_target[int(code)]),
                "round2_bridge_gain": int(round2_bridge[int(code)]),
            }
            for code in class_codes
        },
    }
    report["audit_sha256"] = _sha_json(report)
    return published, report


apply_v34 = apply_v34_candidate
