"""Isolated V3.1-C cross-Core collection and global coordination.

V3.1-C deliberately leaves the V3.1-A/B candidate functions unchanged.  A
runner first calls :func:`collect_cross_core_discoveries` on the same frozen
V3 label/probability windows used by B, converts the returned local witnesses
to global component identities, and then calls :func:`select_global_actions`.

The selector is exact only for the supplied, conservatively constructed
conflict graph and remaining per-Core budgets.  It is not a claim of a global
optimum over every possible raster edit.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy import optimize, sparse

from . import candidate as _candidate


V31C_COORDINATION_MODE = "cross_core_global_milp_v1"
V31C_POLICY_ID = "fragmentation_v31c_cross_core_global_candidate_v1"
V31C_POLICY_VERSION = "v31c_cross_core_global_20260825"
_PROBABILITY_SCALE = 1_000_000_000
_AREA_SCALE = 1_000_000


class V31CCoordinationError(RuntimeError):
    """Raised when a C collection or global-plan contract is invalid."""


@dataclass(frozen=True)
class CrossCoreDiscovery:
    """One partition's witness of a proposal spanning multiple owner Cores."""

    discovery_id: str
    discovery_partition_id: str
    kind: str
    target_index: int
    target_code: int
    footprint: tuple[tuple[int, int], ...]
    involved_core_ids: tuple[str, ...]
    source_codes: tuple[int, ...]
    source_anchors: tuple[tuple[int, int], ...]
    target_anchors: tuple[tuple[int, int], ...]
    dynamic_reduction: int
    component_reduction: int
    probability_support: float
    area_m2: float
    edge_distance_m: float | None
    path_length_m: float | None
    local_proposal_id: str
    footprint_sha256: str


@dataclass(frozen=True)
class GlobalAction:
    """A globally canonical cross-Core raster action before component mapping."""

    action_id: str
    kind: str
    target_index: int
    target_code: int
    footprint: tuple[tuple[int, int], ...]
    involved_core_ids: tuple[str, ...]
    source_codes: tuple[int, ...]
    source_anchors: tuple[tuple[int, int], ...]
    target_anchors: tuple[tuple[int, int], ...]
    dynamic_reduction: int
    component_reduction: int
    probability_support: float
    area_m2: float
    discovery_partition_ids: tuple[str, ...]
    discovery_count: int
    footprint_sha256: str
    score_disagreement: bool


@dataclass(frozen=True)
class PlannedAction:
    """A canonical action with global topology identities and budget charges."""

    action: GlobalAction
    source_component_keys: tuple[str, ...]
    target_component_keys: tuple[str, ...]
    source_charges: tuple[tuple[str, int, int], ...]
    target_charges: tuple[tuple[str, int, int], ...]


def _sha256_json(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _global_points(points: np.ndarray, origin: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    row0, col0 = int(origin[0]), int(origin[1])
    return tuple(sorted((row0 + int(row), col0 + int(col)) for row, col in points))


def collect_cross_core_discoveries(
    labels: np.ndarray,
    *,
    class_codes: Sequence[int],
    pixel_area_m2: float,
    pixel_size_m: tuple[float, float],
    valid_mask: np.ndarray,
    probabilities: np.ndarray,
    confidence: np.ndarray | None,
    global_origin: tuple[int, int],
    discovery_partition_id: str,
    owner_for_global_pixel: Callable[[int, int], str | None],
    policy: _candidate.CandidatePolicy | None = None,
) -> tuple[list[CrossCoreDiscovery], dict[str, Any]]:
    """Regenerate and collect only proposals whose footprint spans Cores.

    Proposal generation is the frozen B generation path.  No proposal is
    applied here, so discovery order and worker completion cannot select an
    action implicitly.
    """

    selected_policy = policy or _candidate.v31b_policy()
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
    component_map, components = _candidate._component_index(baseline, valid, class_codes)
    components_by_id = {item.component_id: item for item in components}
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
    canonical, duplicates, _duplicate_audit = _candidate._canonicalize_v31b_proposals(proposals)
    discoveries: list[CrossCoreDiscovery] = []
    collection_rejections: Counter[str] = Counter()
    for proposal in canonical:
        footprint = _global_points(proposal.footprint, global_origin)
        pixel_owners = tuple(owner_for_global_pixel(row, col) for row, col in footprint)
        if any(owner is None for owner in pixel_owners):
            collection_rejections["unowned_or_strict_invalid_footprint"] += 1
            continue
        owners = tuple(sorted(set(str(owner) for owner in pixel_owners)))
        if len(owners) < 2:
            continue
        source_anchors: list[tuple[int, int]] = []
        footprint_array = np.asarray(proposal.footprint, dtype=np.int32)
        footprint_component_ids = component_map[
            footprint_array[:, 0], footprint_array[:, 1]
        ]
        # If a bridge consumes every visible pixel of a source component that
        # reaches the expanded-window edge, the local witness cannot prove it
        # consumed the complete global component.  B never publishes this
        # cross-Core case; C keeps that conservative safety boundary.
        unsafe_external_source = any(
            components_by_id[int(component_id)].touches_external
            and int(np.count_nonzero(footprint_component_ids == int(component_id)))
            >= len(components_by_id[int(component_id)].pixels)
            for component_id in proposal.source_component_ids
        )
        if unsafe_external_source:
            collection_rejections["unproven_external_source_connectivity"] += 1
            continue
        for component_id in proposal.source_component_ids:
            matches = footprint_array[footprint_component_ids == int(component_id)]
            chosen = matches[0] if len(matches) else components_by_id[int(component_id)].pixels[0]
            source_anchors.append(
                (int(global_origin[0] + chosen[0]), int(global_origin[1] + chosen[1]))
            )
        target_anchors = [
            (
                int(global_origin[0] + components_by_id[int(component_id)].pixels[0, 0]),
                int(global_origin[1] + components_by_id[int(component_id)].pixels[0, 1]),
            )
            for component_id in proposal.baseline_target_component_ids
        ]
        identity = {
            "kind": proposal.kind,
            "target_code": int(proposal.target_code),
            "footprint": footprint,
            "source_codes": sorted(int(value) for value in proposal.source_codes),
        }
        discovery_id = _sha256_json(
            {"identity": identity, "partition_id": discovery_partition_id}
        )
        discoveries.append(
            CrossCoreDiscovery(
                discovery_id=discovery_id,
                discovery_partition_id=str(discovery_partition_id),
                kind=str(proposal.kind),
                target_index=int(proposal.target_index),
                target_code=int(proposal.target_code),
                footprint=footprint,
                involved_core_ids=owners,
                source_codes=tuple(sorted(int(value) for value in proposal.source_codes)),
                source_anchors=tuple(sorted(set(source_anchors))),
                target_anchors=tuple(sorted(set(target_anchors))),
                dynamic_reduction=int(proposal.dynamic_reduction),
                component_reduction=int(proposal.component_reduction),
                probability_support=float(proposal.probability_support),
                area_m2=float(proposal.area_m2),
                edge_distance_m=(
                    None if proposal.edge_distance_m is None else float(proposal.edge_distance_m)
                ),
                path_length_m=(
                    None if proposal.path_length_m is None else float(proposal.path_length_m)
                ),
                local_proposal_id=str(proposal.proposal_id),
                footprint_sha256=_sha256_json(footprint),
            )
        )
    discoveries.sort(key=lambda item: item.discovery_id)
    return discoveries, {
        "canonical_generated": len(canonical),
        "raw_generated": len(proposals),
        "duplicate_proposal_count": int(duplicates["duplicate_proposal"]),
        "cross_core_discovery_count": len(discoveries),
        "collection_rejection_events": dict(sorted(collection_rejections.items())),
        "generation_rejection_events": dict(sorted(generation_rejections.items())),
    }


def canonicalize_global_discoveries(
    discoveries: Iterable[CrossCoreDiscovery],
) -> tuple[list[GlobalAction], list[dict[str, Any]]]:
    """Deduplicate window-local discoveries by their global raster action."""

    grouped: dict[tuple[Any, ...], list[CrossCoreDiscovery]] = defaultdict(list)
    for item in discoveries:
        key = (
            item.kind,
            item.target_index,
            item.target_code,
            item.footprint,
            item.source_codes,
        )
        grouped[key].append(item)
    actions: list[GlobalAction] = []
    duplicate_audit: list[dict[str, Any]] = []
    for key, group in grouped.items():
        ordered = sorted(group, key=lambda item: item.discovery_id)
        first = ordered[0]
        score_rows = {
            (
                item.dynamic_reduction,
                item.component_reduction,
                round(item.probability_support, 12),
                round(item.area_m2, 9),
            )
            for item in ordered
        }
        # A window-edge witness can see a truncated baseline component.  Use
        # conservative scores so duplicate discovery cannot inflate the MILP.
        dynamic = min(item.dynamic_reduction for item in ordered)
        component = min(item.component_reduction for item in ordered)
        support = min(item.probability_support for item in ordered)
        area = max(item.area_m2 for item in ordered)
        identity = {
            "kind": first.kind,
            "target_code": first.target_code,
            "footprint": first.footprint,
            "source_codes": first.source_codes,
        }
        action_id = _sha256_json(identity)
        actions.append(
            GlobalAction(
                action_id=action_id,
                kind=first.kind,
                target_index=first.target_index,
                target_code=first.target_code,
                footprint=first.footprint,
                involved_core_ids=first.involved_core_ids,
                source_codes=first.source_codes,
                source_anchors=tuple(
                    sorted({point for item in ordered for point in item.source_anchors})
                ),
                target_anchors=tuple(
                    sorted({point for item in ordered for point in item.target_anchors})
                ),
                dynamic_reduction=int(dynamic),
                component_reduction=int(component),
                probability_support=float(support),
                area_m2=float(area),
                discovery_partition_ids=tuple(
                    sorted({item.discovery_partition_id for item in ordered})
                ),
                discovery_count=len(ordered),
                footprint_sha256=first.footprint_sha256,
                score_disagreement=len(score_rows) > 1,
            )
        )
        duplicate_audit.extend(
            {
                "discovery_id": item.discovery_id,
                "canonical_action_id": action_id,
                "discovery_partition_id": item.discovery_partition_id,
                "reason": "duplicate_global_action",
            }
            for item in ordered[1:]
        )
    actions.sort(key=lambda item: item.action_id)
    duplicate_audit.sort(key=lambda item: item["discovery_id"])
    return actions, duplicate_audit


def action_conflicts(actions: Sequence[PlannedAction]) -> set[tuple[int, int]]:
    """Return the conservative topology/footprint conflict graph.

    Sharing any baseline source or target component is deliberately a conflict.
    That makes each selected action's frozen connectivity proof independent of
    every other selected C action.
    """

    conflicts: set[tuple[int, int]] = set()
    inverted_pixels: dict[tuple[int, int], list[int]] = defaultdict(list)
    inverted_components: dict[str, list[int]] = defaultdict(list)
    for index, planned in enumerate(actions):
        for pixel in planned.action.footprint:
            inverted_pixels[pixel].append(index)
        for key in set(planned.source_component_keys) | set(planned.target_component_keys):
            inverted_components[key].append(index)
    for members in [*inverted_pixels.values(), *inverted_components.values()]:
        unique = sorted(set(members))
        for position, left in enumerate(unique):
            conflicts.update((left, right) for right in unique[position + 1 :])
    return conflicts


def _constraint_matrix(
    actions: Sequence[PlannedAction],
    conflicts: set[tuple[int, int]],
    source_remaining: Mapping[tuple[str, int], int],
    target_remaining: Mapping[tuple[str, int], int],
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row_index = 0
    for left, right in sorted(conflicts):
        rows.extend((row_index, row_index))
        cols.extend((left, right))
        values.extend((1.0, 1.0))
        lower.append(-np.inf)
        upper.append(1.0)
        row_index += 1
    for charge_kind, remaining in (
        ("source", source_remaining),
        ("target", target_remaining),
    ):
        for budget_key in sorted(remaining):
            found = False
            for index, planned in enumerate(actions):
                charges = planned.source_charges if charge_kind == "source" else planned.target_charges
                charge = sum(
                    count for core_id, code, count in charges if (core_id, code) == budget_key
                )
                if charge:
                    rows.append(row_index)
                    cols.append(index)
                    values.append(float(charge))
                    found = True
            if found:
                lower.append(-np.inf)
                upper.append(float(max(0, int(remaining[budget_key]))))
                row_index += 1
    matrix = sparse.coo_matrix(
        (values, (rows, cols)), shape=(row_index, len(actions)), dtype=np.float64
    ).tocsr()
    return matrix, np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def select_global_actions(
    actions: Sequence[PlannedAction],
    *,
    source_remaining: Mapping[tuple[str, int], int],
    target_remaining: Mapping[tuple[str, int], int],
    time_limit_seconds: float | None = None,
) -> tuple[list[PlannedAction], dict[str, Any]]:
    """Solve the lexicographic global selection over C's frozen action set."""

    ordered = sorted(actions, key=lambda item: item.action.action_id)
    rejected_individual: dict[str, str] = {}
    eligible: list[PlannedAction] = []
    rank_groups: dict[tuple[Any, ...], list[PlannedAction]] = defaultdict(list)
    for planned in ordered:
        action = planned.action
        rank_groups[
            (
                int(action.dynamic_reduction),
                int(action.component_reduction),
                round(float(action.probability_support), 12),
                round(float(action.area_m2), 9),
                action.footprint_sha256,
            )
        ].append(planned)
    ambiguous_ids = {
        item.action.action_id
        for group in rank_groups.values()
        if len({item.action.target_code for item in group}) > 1
        for item in group
    }
    for planned in ordered:
        source = Counter((core, code) for core, code, _count in planned.source_charges)
        target = Counter((core, code) for core, code, _count in planned.target_charges)
        source_amount = Counter({key: 0 for key in source})
        target_amount = Counter({key: 0 for key in target})
        for core, code, count in planned.source_charges:
            source_amount[(core, code)] += int(count)
        for core, code, count in planned.target_charges:
            target_amount[(core, code)] += int(count)
        if planned.action.action_id in ambiguous_ids:
            rejected_individual[planned.action.action_id] = "ambiguous_target_tie"
        elif any(value > max(0, int(source_remaining.get(key, 0))) for key, value in source_amount.items()):
            rejected_individual[planned.action.action_id] = "source_budget"
        elif any(value > max(0, int(target_remaining.get(key, 0))) for key, value in target_amount.items()):
            rejected_individual[planned.action.action_id] = "target_budget"
        else:
            eligible.append(planned)
    if not eligible:
        return [], {
            "solver": "scipy.optimize.milp/HiGHS",
            "coordination_mode": V31C_COORDINATION_MODE,
            "eligible_action_count": 0,
            "selected_action_count": 0,
            "individual_rejections": rejected_individual,
            "conflict_edge_count": 0,
            "optimal": True,
        }
    conflicts = action_conflicts(eligible)
    matrix, lower, upper = _constraint_matrix(
        eligible, conflicts, source_remaining, target_remaining
    )
    base_rows = matrix
    base_lower = lower
    base_upper = upper
    metrics = [
        np.asarray([item.action.dynamic_reduction for item in eligible], dtype=np.int64),
        np.asarray([item.action.component_reduction for item in eligible], dtype=np.int64),
        np.asarray(
            [round(item.action.probability_support * _PROBABILITY_SCALE) for item in eligible],
            dtype=np.int64,
        ),
        -np.asarray(
            [round(item.action.area_m2 * _AREA_SCALE) for item in eligible],
            dtype=np.int64,
        ),
        -np.arange(1, len(eligible) + 1, dtype=np.int64),
    ]
    objective_names = (
        "dynamic_reduction",
        "component_reduction",
        "probability_support",
        "negative_area",
        "negative_stable_action_ordinal",
    )
    fixed_rows: list[np.ndarray] = []
    fixed_values: list[int] = []
    last_result: optimize.OptimizeResult | None = None
    options: dict[str, Any] = {"mip_rel_gap": 0.0, "presolve": True}
    if time_limit_seconds is not None:
        options["time_limit"] = float(time_limit_seconds)
    for metric in metrics:
        if fixed_rows:
            fixed_matrix = sparse.csr_matrix(np.vstack(fixed_rows).astype(np.float64))
            constraint_matrix = sparse.vstack((base_rows, fixed_matrix), format="csr")
            constraint_lower = np.concatenate((base_lower, np.asarray(fixed_values, dtype=np.float64)))
            constraint_upper = np.concatenate((base_upper, np.asarray(fixed_values, dtype=np.float64)))
        else:
            constraint_matrix, constraint_lower, constraint_upper = base_rows, base_lower, base_upper
        last_result = optimize.milp(
            c=-metric.astype(np.float64),
            integrality=np.ones(len(eligible), dtype=np.uint8),
            bounds=optimize.Bounds(np.zeros(len(eligible)), np.ones(len(eligible))),
            constraints=optimize.LinearConstraint(
                constraint_matrix, constraint_lower, constraint_upper
            ),
            options=options,
        )
        if not last_result.success or last_result.x is None:
            raise V31CCoordinationError(
                f"global MILP did not prove an optimum: status={last_result.status} message={last_result.message}"
            )
        selected_mask = last_result.x > 0.5
        optimum = int(metric @ selected_mask.astype(np.int64))
        fixed_rows.append(metric)
        fixed_values.append(optimum)
    assert last_result is not None and last_result.x is not None
    selected_mask = last_result.x > 0.5
    selected = [item for item, chosen in zip(eligible, selected_mask) if chosen]
    selected.sort(key=lambda item: item.action.action_id)
    selected_ids = {item.action.action_id for item in selected}
    global_rejections = {
        item.action.action_id: "global_conflict_or_budget_selection"
        for item in eligible
        if item.action.action_id not in selected_ids
    }
    return selected, {
        "solver": "scipy.optimize.milp/HiGHS",
        "coordination_mode": V31C_COORDINATION_MODE,
        "eligible_action_count": len(eligible),
        "selected_action_count": len(selected),
        "individual_rejections": dict(sorted(rejected_individual.items())),
        "global_rejections": dict(sorted(global_rejections.items())),
        "conflict_edge_count": len(conflicts),
        "lexicographic_objectives": {
            name: value for name, value in zip(objective_names, fixed_values)
        },
        "optimal": True,
        "solver_status": int(last_result.status),
        "solver_message": str(last_result.message),
    }


def global_action_to_dict(action: GlobalAction) -> dict[str, Any]:
    """Return the stable JSON ledger representation used by the runner."""

    return {
        "action_id": action.action_id,
        "kind": action.kind,
        "target_index": action.target_index,
        "target_code": action.target_code,
        "footprint": [list(point) for point in action.footprint],
        "involved_core_ids": list(action.involved_core_ids),
        "source_codes": list(action.source_codes),
        "source_anchors": [list(point) for point in action.source_anchors],
        "target_anchors": [list(point) for point in action.target_anchors],
        "dynamic_reduction": action.dynamic_reduction,
        "component_reduction": action.component_reduction,
        "probability_support": action.probability_support,
        "area_m2": action.area_m2,
        "discovery_partition_ids": list(action.discovery_partition_ids),
        "discovery_count": action.discovery_count,
        "footprint_sha256": action.footprint_sha256,
        "score_disagreement": action.score_disagreement,
    }
