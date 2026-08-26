"""Plan-only global coordination primitives for the isolated V3.1-E study.

E does not publish a raster.  It regenerates proposals from a frozen B mask,
canonicalises their exact global footprints, scores exact boundary changes and
computes two auditable optima inside one conservative independent-action model:
an unconstrained plan and a boundary-nonincreasing plan.  A caller must still
distinguish proposals whose B first-round dependency obligations have been
proved from proposals that require an exact B dependency replay.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy import optimize, sparse

from . import candidate as _candidate
from .v31c import CrossCoreDiscovery, GlobalAction, canonicalize_global_discoveries


V31E_POLICY_ID = "fragmentation_v31e_global_boundary_plan_candidate_v1"
V31E_POLICY_VERSION = "v31e_plan_only_20260825"
V31E_COORDINATION_MODE = "global_b_boundary_lexicographic_milp_plan_only_v1"

_PROBABILITY_SCALE = 1_000_000_000
_AREA_SCALE = 1_000_000
_BOUNDARY_METRES_SCALE = 1_000_000


class V31EPlanningError(RuntimeError):
    """Raised when a plan-only proof input or optimisation is invalid."""


@dataclass(frozen=True)
class BoundaryPlannedAction:
    """A globally scored B-relative action before plan-only selection."""

    action: GlobalAction
    source_component_keys: tuple[str, ...]
    target_component_keys: tuple[str, ...]
    source_charges: tuple[tuple[str, int, int], ...]
    target_charges: tuple[tuple[str, int, int], ...]
    component_delta_by_class: tuple[tuple[int, int], ...]
    dynamic_delta_by_class: tuple[tuple[int, int], ...]
    boundary_delta_edges: int
    boundary_delta_metres: float
    dependency_proof: str = "unclassified"


def _sha256_json(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _window_source_connectivity_is_provable(
    footprint: np.ndarray,
    component_map: np.ndarray,
    components: Sequence[Any],
) -> bool:
    """Return whether a window contains every residual port of the removal.

    A footprint on the expanded-window edge can have an unseen same-class
    neighbour immediately outside the window.  Local connectivity then is not
    a proof of global connectivity.  A fully consumed visible source component
    that continues outside the window is equally unproved.
    """

    values = np.asarray(footprint, dtype=np.int32)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
        return False
    height, width = component_map.shape
    if np.any(
        (values[:, 0] <= 0)
        | (values[:, 0] >= height - 1)
        | (values[:, 1] <= 0)
        | (values[:, 1] >= width - 1)
    ):
        return False
    by_id = {int(item.component_id): item for item in components}
    ids, counts = np.unique(
        component_map[values[:, 0], values[:, 1]], return_counts=True
    )
    for component_id, removed_count in zip(ids, counts):
        if not int(component_id):
            continue
        component = by_id[int(component_id)]
        if component.touches_external and int(removed_count) >= len(component.pixels):
            return False
    return True


def collect_global_b_discoveries(
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
    """Regenerate every locally proved B proposal with an exact global footprint.

    Unlike C's collector this includes both one-Core and cross-Core footprints.
    A local source component that reaches the context edge is accepted only if
    the footprint leaves a locally connected residual port set; consuming the
    complete visible component is conservatively rejected.
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
    component_map, components = _candidate._component_index(
        baseline, valid, class_codes
    )
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
    canonical, duplicates, _duplicate_audit = (
        _candidate._canonicalize_v31b_proposals(proposals)
    )
    discoveries: list[CrossCoreDiscovery] = []
    collection_rejections: Counter[str] = Counter()
    row0, col0 = int(global_origin[0]), int(global_origin[1])
    for proposal in canonical:
        global_footprint = tuple(
            sorted(
                (row0 + int(row), col0 + int(col))
                for row, col in proposal.footprint
            )
        )
        owners_raw = tuple(
            owner_for_global_pixel(row, col) for row, col in global_footprint
        )
        if any(owner is None for owner in owners_raw):
            collection_rejections["unowned_or_strict_invalid_footprint"] += 1
            continue
        owners = tuple(sorted({str(owner) for owner in owners_raw}))
        local_footprint = np.asarray(proposal.footprint, dtype=np.int32)
        footprint_component_ids = component_map[
            local_footprint[:, 0], local_footprint[:, 1]
        ]
        if not _window_source_connectivity_is_provable(
            local_footprint, component_map, components
        ):
            collection_rejections[
                "unproven_window_edge_or_external_source_connectivity"
            ] += 1
            continue
        # Generation already proves the residual ports of every partially
        # consumed local source component are connected after removal.
        if not _candidate._source_connectivity_safe(
            local_footprint, baseline, component_map, components, valid
        ):
            collection_rejections["source_residual_ports_disconnected"] += 1
            continue
        source_anchors: list[tuple[int, int]] = []
        for component_id in proposal.source_component_ids:
            matches = local_footprint[
                footprint_component_ids == int(component_id)
            ]
            chosen = (
                matches[0]
                if len(matches)
                else components_by_id[int(component_id)].pixels[0]
            )
            source_anchors.append(
                (row0 + int(chosen[0]), col0 + int(chosen[1]))
            )
        target_anchors = tuple(
            sorted(
                {
                    (
                        row0
                        + int(components_by_id[int(component_id)].pixels[0, 0]),
                        col0
                        + int(components_by_id[int(component_id)].pixels[0, 1]),
                    )
                    for component_id in proposal.baseline_target_component_ids
                }
            )
        )
        if not target_anchors or any(
            owner_for_global_pixel(row, col) is None for row, col in target_anchors
        ):
            collection_rejections["unowned_or_strict_invalid_target_anchor"] += 1
            continue
        identity = {
            "kind": proposal.kind,
            "target_code": int(proposal.target_code),
            "footprint": global_footprint,
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
                footprint=global_footprint,
                involved_core_ids=owners,
                source_codes=tuple(
                    sorted(int(value) for value in proposal.source_codes)
                ),
                source_anchors=tuple(sorted(set(source_anchors))),
                target_anchors=target_anchors,
                dynamic_reduction=int(proposal.dynamic_reduction),
                component_reduction=int(proposal.component_reduction),
                probability_support=float(proposal.probability_support),
                area_m2=float(proposal.area_m2),
                edge_distance_m=(
                    None
                    if proposal.edge_distance_m is None
                    else float(proposal.edge_distance_m)
                ),
                path_length_m=(
                    None
                    if proposal.path_length_m is None
                    else float(proposal.path_length_m)
                ),
                local_proposal_id=str(proposal.proposal_id),
                footprint_sha256=_sha256_json(global_footprint),
            )
        )
    discoveries.sort(key=lambda item: item.discovery_id)
    return discoveries, {
        "raw_generated": len(proposals),
        "canonical_generated": len(canonical),
        "duplicate_proposal_count": int(duplicates["duplicate_proposal"]),
        "global_discovery_count": len(discoveries),
        "collection_rejection_events": dict(sorted(collection_rejections.items())),
        "generation_rejection_events": dict(sorted(generation_rejections.items())),
        "source_connectivity_proof": (
            "all_removal_ports_inside_window_local_residual_ports_connected_and_"
            "external_complete_consumption_rejected"
        ),
    }


def exact_boundary_delta(
    action: GlobalAction,
    *,
    label_for_global_pixel: Callable[[tuple[int, int]], int],
    valid_for_global_pixel: Callable[[tuple[int, int]], bool],
    owner_for_global_pixel: Callable[[tuple[int, int]], str | None],
    physical_metrics_by_owner: Mapping[str, Mapping[str, float]],
) -> tuple[int, float]:
    """Return exact B-relative cross-class boundary edge/metre deltas.

    The edge lengths intentionally match the full evaluator: a horizontal
    pixel pair uses ``row_step_m``, a vertical pair uses ``column_step_m`` and
    a cross-Core edge uses the mean of both owner steps.
    """

    footprint = frozenset((int(row), int(col)) for row, col in action.footprint)
    if not footprint:
        raise V31EPlanningError("boundary action footprint cannot be empty")
    if any(not valid_for_global_pixel(point) for point in footprint):
        raise V31EPlanningError("boundary action footprint includes an invalid pixel")
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for row, col in footprint:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbour = (row + dr, col + dc)
            if valid_for_global_pixel(neighbour):
                edges.add(tuple(sorted(((row, col), neighbour))))
    delta_edges = 0
    delta_metres = 0.0
    for first, second in sorted(edges):
        before_first = int(label_for_global_pixel(first))
        before_second = int(label_for_global_pixel(second))
        after_first = int(action.target_code) if first in footprint else before_first
        after_second = int(action.target_code) if second in footprint else before_second
        before = int(before_first != before_second)
        after = int(after_first != after_second)
        delta = after - before
        if not delta:
            continue
        first_owner = owner_for_global_pixel(first)
        second_owner = owner_for_global_pixel(second)
        if first_owner is None or second_owner is None:
            raise V31EPlanningError("valid boundary edge lacks a Core owner")
        dr = abs(first[0] - second[0])
        dc = abs(first[1] - second[1])
        if dr + dc != 1:
            raise V31EPlanningError("boundary edge is not four-neighbour adjacent")
        metric_name = "column_step_m" if dr else "row_step_m"
        first_step = float(physical_metrics_by_owner[first_owner][metric_name])
        second_step = float(physical_metrics_by_owner[second_owner][metric_name])
        if not all(
            math.isfinite(value) and value > 0
            for value in (first_step, second_step)
        ):
            raise V31EPlanningError("boundary physical step must be finite and positive")
        if first_owner == second_owner:
            step = first_step
        else:
            relative_difference = abs(first_step - second_step) / max(
                first_step, second_step
            )
            if relative_difference > 0.02:
                raise V31EPlanningError(
                    "abnormal shared-Core physical step: "
                    f"{first_step} vs {second_step} m"
                )
            step = (first_step + second_step) / 2.0
        delta_edges += delta
        delta_metres += delta * step
    return int(delta_edges), float(delta_metres)


def action_conflicts(
    actions: Sequence[BoundaryPlannedAction],
) -> set[tuple[int, int]]:
    """Return conflicts that preserve topology and boundary-delta additivity."""

    conflicts: set[tuple[int, int]] = set()
    pixels: dict[tuple[int, int], list[int]] = defaultdict(list)
    components: dict[str, list[int]] = defaultdict(list)
    for index, planned in enumerate(actions):
        for point in planned.action.footprint:
            pixels[(int(point[0]), int(point[1]))].append(index)
        for key in set(planned.source_component_keys) | set(
            planned.target_component_keys
        ):
            components[str(key)].append(index)
    for members in components.values():
        ordered = sorted(set(members))
        for position, left in enumerate(ordered):
            conflicts.update((left, right) for right in ordered[position + 1 :])
    for point, members in pixels.items():
        nearby = set(members)
        row, col = point
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nearby.update(pixels.get((row + dr, col + dc), ()))
        ordered = sorted(nearby)
        for position, left in enumerate(ordered):
            conflicts.update((left, right) for right in ordered[position + 1 :])
    return conflicts


def _constraint_matrix(
    actions: Sequence[BoundaryPlannedAction],
    *,
    conflicts: set[tuple[int, int]],
    source_remaining: Mapping[tuple[str, int], int],
    target_remaining: Mapping[tuple[str, int], int],
    enforce_boundary: bool,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row_index = 0

    def add_row(entries: Iterable[tuple[int, float]], ceiling: float) -> None:
        nonlocal row_index
        found = False
        for column, value in entries:
            if value:
                rows.append(row_index)
                cols.append(int(column))
                values.append(float(value))
                found = True
        if found:
            lower.append(-np.inf)
            upper.append(float(ceiling))
            row_index += 1

    for left, right in sorted(conflicts):
        add_row(((left, 1.0), (right, 1.0)), 1.0)
    for charge_name, remaining in (
        ("source_charges", source_remaining),
        ("target_charges", target_remaining),
    ):
        for budget_key in sorted(remaining):
            add_row(
                (
                    (
                        index,
                        sum(
                            count
                            for core, code, count in getattr(planned, charge_name)
                            if (core, code) == budget_key
                        ),
                    )
                    for index, planned in enumerate(actions)
                ),
                max(0, int(remaining[budget_key])),
            )
    class_codes = sorted(
        {
            int(code)
            for planned in actions
            for code, _delta in planned.component_delta_by_class
        }
    )
    for code in class_codes:
        add_row(
            (
                (
                    index,
                    sum(
                        delta
                        for item_code, delta in planned.component_delta_by_class
                        if int(item_code) == code
                    ),
                )
                for index, planned in enumerate(actions)
            ),
            0.0,
        )
    if enforce_boundary:
        add_row(
            (
                (index, planned.boundary_delta_edges)
                for index, planned in enumerate(actions)
            ),
            0.0,
        )
        add_row(
            (
                (index, planned.boundary_delta_metres)
                for index, planned in enumerate(actions)
            ),
            1e-7,
        )
    matrix = sparse.coo_matrix(
        (values, (rows, cols)),
        shape=(row_index, len(actions)),
        dtype=np.float64,
    ).tocsr()
    return matrix, np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def select_boundary_aware_actions(
    actions: Sequence[BoundaryPlannedAction],
    *,
    source_remaining: Mapping[tuple[str, int], int],
    target_remaining: Mapping[tuple[str, int], int],
    enforce_boundary: bool,
    required_dynamic_reduction: int = 130,
    engineering_headroom_reduction: int = 150,
    time_limit_seconds: float | None = None,
) -> tuple[list[BoundaryPlannedAction], dict[str, Any]]:
    """Solve a deterministic plan-only lexicographic action selection."""

    if required_dynamic_reduction < 0 or engineering_headroom_reduction < required_dynamic_reduction:
        raise V31EPlanningError("invalid effect/headroom thresholds")
    ordered = sorted(actions, key=lambda item: item.action.action_id)
    eligible: list[BoundaryPlannedAction] = []
    rejected: dict[str, str] = {}
    rank_groups: dict[tuple[Any, ...], list[BoundaryPlannedAction]] = defaultdict(list)
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
        action = planned.action
        if action.action_id in ambiguous_ids:
            rejected[action.action_id] = "ambiguous_target_tie"
            continue
        if action.dynamic_reduction < 0 or action.component_reduction <= 0:
            rejected[action.action_id] = "nonpositive_global_topology_gain"
            continue
        if not math.isfinite(planned.boundary_delta_metres):
            rejected[action.action_id] = "invalid_boundary_delta"
            continue
        component_rows = dict(planned.component_delta_by_class)
        if any(int(delta) > 0 for delta in component_rows.values()):
            rejected[action.action_id] = "individual_per_class_component_increase"
            continue
        source_amount: Counter[tuple[str, int]] = Counter()
        target_amount: Counter[tuple[str, int]] = Counter()
        for core, code, count in planned.source_charges:
            source_amount[(str(core), int(code))] += int(count)
        for core, code, count in planned.target_charges:
            target_amount[(str(core), int(code))] += int(count)
        if any(
            amount > max(0, int(source_remaining.get(key, 0)))
            for key, amount in source_amount.items()
        ):
            rejected[action.action_id] = "source_budget"
        elif any(
            amount > max(0, int(target_remaining.get(key, 0)))
            for key, amount in target_amount.items()
        ):
            rejected[action.action_id] = "target_budget"
        else:
            eligible.append(planned)
    if not eligible:
        audit = {
            "solver": "scipy.optimize.milp/HiGHS",
            "coordination_mode": V31E_COORDINATION_MODE,
            "enforce_boundary": bool(enforce_boundary),
            "eligible_action_count": 0,
            "selected_action_count": 0,
            "individual_rejections": dict(sorted(rejected.items())),
            "optimal": True,
            "selected_dynamic_reduction": 0,
            "selected_component_reduction": 0,
            "selected_boundary_delta_edges": 0,
            "selected_boundary_delta_metres": 0.0,
            "effect_gate": {
                "required": int(required_dynamic_reduction),
                "actual": 0,
                "pass": required_dynamic_reduction == 0,
            },
            "engineering_headroom_gate": {
                "required": int(engineering_headroom_reduction),
                "actual": 0,
                "pass": engineering_headroom_reduction == 0,
            },
        }
        return [], audit
    conflicts = action_conflicts(eligible)
    matrix, lower, upper = _constraint_matrix(
        eligible,
        conflicts=conflicts,
        source_remaining=source_remaining,
        target_remaining=target_remaining,
        enforce_boundary=enforce_boundary,
    )
    base_rows, base_lower, base_upper = matrix, lower, upper
    metrics = [
        np.asarray(
            [item.action.dynamic_reduction for item in eligible], dtype=np.int64
        ),
        np.asarray(
            [item.action.component_reduction for item in eligible], dtype=np.int64
        ),
        -np.asarray(
            [item.boundary_delta_edges for item in eligible], dtype=np.int64
        ),
        -np.asarray(
            [
                round(item.boundary_delta_metres * _BOUNDARY_METRES_SCALE)
                for item in eligible
            ],
            dtype=np.int64,
        ),
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
        "negative_boundary_edges",
        "negative_boundary_metres",
        "probability_support",
        "negative_area",
        "negative_stable_action_ordinal",
    )
    fixed_rows: list[np.ndarray] = []
    fixed_values: list[int] = []
    result: optimize.OptimizeResult | None = None
    options: dict[str, Any] = {"mip_rel_gap": 0.0, "presolve": True}
    if time_limit_seconds is not None:
        options["time_limit"] = float(time_limit_seconds)
    for metric in metrics:
        if fixed_rows:
            fixed_matrix = sparse.csr_matrix(np.vstack(fixed_rows).astype(np.float64))
            constraints = sparse.vstack((base_rows, fixed_matrix), format="csr")
            constraint_lower = np.concatenate(
                (base_lower, np.asarray(fixed_values, dtype=np.float64))
            )
            constraint_upper = np.concatenate(
                (base_upper, np.asarray(fixed_values, dtype=np.float64))
            )
        else:
            constraints, constraint_lower, constraint_upper = (
                base_rows,
                base_lower,
                base_upper,
            )
        result = optimize.milp(
            c=-metric.astype(np.float64),
            integrality=np.ones(len(eligible), dtype=np.uint8),
            bounds=optimize.Bounds(
                np.zeros(len(eligible)), np.ones(len(eligible))
            ),
            constraints=optimize.LinearConstraint(
                constraints, constraint_lower, constraint_upper
            ),
            options=options,
        )
        if not result.success or result.x is None:
            raise V31EPlanningError(
                "global E MILP did not prove an optimum: "
                f"status={result.status} message={result.message}"
            )
        selected_mask = result.x > 0.5
        optimum = int(metric @ selected_mask.astype(np.int64))
        fixed_rows.append(metric)
        fixed_values.append(optimum)
    assert result is not None and result.x is not None
    selected = [
        item for item, chosen in zip(eligible, result.x > 0.5) if bool(chosen)
    ]
    selected.sort(key=lambda item: item.action.action_id)
    dynamic = int(sum(item.action.dynamic_reduction for item in selected))
    components = int(sum(item.action.component_reduction for item in selected))
    boundary_edges = int(sum(item.boundary_delta_edges for item in selected))
    boundary_metres = float(sum(item.boundary_delta_metres for item in selected))
    selected_ids = {item.action.action_id for item in selected}
    return selected, {
        "solver": "scipy.optimize.milp/HiGHS",
        "coordination_mode": V31E_COORDINATION_MODE,
        "enforce_boundary": bool(enforce_boundary),
        "eligible_action_count": len(eligible),
        "selected_action_count": len(selected),
        "individual_rejections": dict(sorted(rejected.items())),
        "global_rejections": {
            item.action.action_id: "global_conflict_budget_or_boundary_selection"
            for item in eligible
            if item.action.action_id not in selected_ids
        },
        "conflict_edge_count": len(conflicts),
        "optimal": True,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "lexicographic_objectives": {
            name: value for name, value in zip(objective_names, fixed_values)
        },
        "selected_dynamic_reduction": dynamic,
        "selected_component_reduction": components,
        "selected_boundary_delta_edges": boundary_edges,
        "selected_boundary_delta_metres": boundary_metres,
        "effect_gate": {
            "required": int(required_dynamic_reduction),
            "actual": dynamic,
            "pass": dynamic >= required_dynamic_reduction,
        },
        "engineering_headroom_gate": {
            "required": int(engineering_headroom_reduction),
            "actual": dynamic,
            "pass": dynamic >= engineering_headroom_reduction,
        },
    }


def canonicalise_discoveries(
    discoveries: Iterable[CrossCoreDiscovery],
) -> tuple[list[GlobalAction], list[dict[str, Any]]]:
    """Public E spelling for the proven C global-footprint canonicaliser."""

    return canonicalize_global_discoveries(discoveries)
