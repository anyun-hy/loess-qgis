"""Single-round, probability-gated V3.1-A topology proposals.

The module is intentionally independent of V3.  It never writes files,
changes a production default, or relies on vector buffering.  All proposals
are computed from one frozen hard-label baseline, then adjudicated once.  This
is an in-memory reference implementation for tests and isolated panels, not a
production-scale Partition/Core runner.

Labels are class *indices* into ``class_codes`` and probabilities have shape
``[len(class_codes), height, width]``.  Distances and areas are supplied in
physical units by the authoritative-raster caller; no fixed CRS or pixel size
is embedded here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import heapq
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


FOUR_CONNECTED = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
POLICY_ID = "fragmentation_v31a_class_topology_candidate_v1"
POLICY_VERSION = "v31a_approved_20260824"
V31B_POLICY_ID = "fragmentation_v31b_dependency_incremental_candidate_v1"
V31B_POLICY_VERSION = "v31b_dependency_incremental_20260824"
V31B_ADJUDICATION_MODE = "dependency_incremental_v1"
ALLOWED_BASELINE_KINDS = frozenset({"raw_argmax", "v3_cleaned"})
P10_METHOD = "empirical_nearest_rank_ceil"


class CandidateError(RuntimeError):
    """Raised for an invalid V3.1 candidate invocation or policy."""


@dataclass(frozen=True)
class ClassPolicy:
    """Frozen physical and evidence limits for one land-cover class."""

    dynamic_fragmentation_m2: float
    ordinary_protected: bool
    enclosed_island_max_m2: float
    allow_same_class_bridge: bool
    bridge_max_edge_distance_m: float
    bridge_max_new_footprint_m2: float
    minimum_target_probability_mean: float
    maximum_current_minus_target_probability_mean: float
    minimum_target_probability_p10: float


@dataclass(frozen=True)
class CandidatePolicy:
    """Policy is data, never an implicit fallback to production V3."""

    class_policies: Mapping[int, ClassPolicy]
    semantic_compatible_targets: Mapping[int, frozenset[int]]
    protected_source_codes: frozenset[int]
    maximum_source_loss_fraction: float = 0.02
    maximum_target_gain_fraction: float = 0.02
    protected_bridge_gain_fraction: float = 0.01
    island_maximum_mean_confidence: float = 0.65
    audit_proposal_limit: int = 256
    policy_id: str = POLICY_ID
    policy_version: str = POLICY_VERSION


def _class_policy(
    dynamic: float, ordinary_protected: bool, island: float, bridge: bool,
    distance: float, footprint: float, prob_mean: float, prob_drop: float,
    prob_p10: float,
) -> ClassPolicy:
    return ClassPolicy(dynamic, ordinary_protected, island, bridge, distance,
                       footprint, prob_mean, prob_drop, prob_p10)


def v31a_policy() -> CandidatePolicy:
    """Return the user-approved V3.1-A table as an immutable policy object."""

    rows = {
        12: _class_policy(50, True, 0, True, 6, 50, .35, .05, .25),
        13: _class_policy(150, False, 150, False, 0, 0, .30, .10, .20),
        21: _class_policy(80, False, 80, True, 12, 160, .30, .10, .20),
        31: _class_policy(100, False, 100, False, 0, 0, .30, .10, .20),
        32: _class_policy(100, False, 100, True, 12, 200, .30, .10, .20),
        33: _class_policy(60, True, 0, True, 6, 60, .35, .05, .25),
        43: _class_policy(100, False, 100, False, 0, 0, .30, .10, .20),
        51: _class_policy(80, False, 80, True, 8, 160, .35, .08, .20),
        52: _class_policy(60, False, 60, True, 10, 120, .35, .08, .20),
        53: _class_policy(50, False, 50, False, 0, 0, .35, .08, .20),
        54: _class_policy(50, False, 50, False, 0, 0, .35, .08, .20),
        61: _class_policy(30, True, 0, True, 4, 30, .40, .05, .25),
        62: _class_policy(50, True, 0, True, 6, 50, .40, .05, .25),
        71: _class_policy(50, True, 0, True, 6, 50, .40, .05, .25),
    }
    compatible = {
        13: frozenset({21, 31, 32, 43}), 21: frozenset({13}),
        31: frozenset({13, 32, 43}), 32: frozenset({31, 43}),
        43: frozenset({13, 31, 32}), 51: frozenset({52, 53, 54}),
        52: frozenset({51, 53, 54}), 53: frozenset({51, 52, 54}),
        54: frozenset({51, 52, 53}),
    }
    protected = frozenset(code for code, row in rows.items() if row.ordinary_protected)
    return CandidatePolicy(
        class_policies=MappingProxyType(rows),
        semantic_compatible_targets=MappingProxyType(compatible),
        protected_source_codes=protected,
    )


# Concise names retained as the intended public candidate API.
policy_v31a = v31a_policy


def v31b_policy() -> CandidatePolicy:
    """Return the isolated V3.1-B policy.

    B intentionally reuses the approved class/evidence/budget policy from A.
    Its only policy change is the conflict-adjudication implementation, which
    is recorded by a distinct immutable ID and version.
    """

    approved = v31a_policy()
    return CandidatePolicy(
        class_policies=approved.class_policies,
        semantic_compatible_targets=approved.semantic_compatible_targets,
        protected_source_codes=approved.protected_source_codes,
        maximum_source_loss_fraction=approved.maximum_source_loss_fraction,
        maximum_target_gain_fraction=approved.maximum_target_gain_fraction,
        protected_bridge_gain_fraction=approved.protected_bridge_gain_fraction,
        island_maximum_mean_confidence=approved.island_maximum_mean_confidence,
        audit_proposal_limit=approved.audit_proposal_limit,
        policy_id=V31B_POLICY_ID,
        policy_version=V31B_POLICY_VERSION,
    )


policy_v31b = v31b_policy


def policy_snapshot(policy: CandidatePolicy | None = None) -> dict[str, Any]:
    """Return a deterministic JSON-safe representation of the exact policy."""

    chosen = policy or v31a_policy()
    snapshot = {
        "policy_id": chosen.policy_id,
        "policy_version": chosen.policy_version,
        "class_policies": {
            str(code): {
                "dynamic_fragmentation_m2": float(row.dynamic_fragmentation_m2),
                "ordinary_protected": bool(row.ordinary_protected),
                "enclosed_island_max_m2": float(row.enclosed_island_max_m2),
                "allow_same_class_bridge": bool(row.allow_same_class_bridge),
                "bridge_max_edge_distance_m": float(row.bridge_max_edge_distance_m),
                "bridge_max_new_footprint_m2": float(row.bridge_max_new_footprint_m2),
                "minimum_target_probability_mean": float(row.minimum_target_probability_mean),
                "maximum_current_minus_target_probability_mean": float(row.maximum_current_minus_target_probability_mean),
                "minimum_target_probability_p10": float(row.minimum_target_probability_p10),
            }
            for code, row in sorted(chosen.class_policies.items())
        },
        "semantic_compatible_targets": {
            str(code): sorted(int(v) for v in targets)
            for code, targets in sorted(chosen.semantic_compatible_targets.items())
        },
        "protected_source_codes": sorted(int(v) for v in chosen.protected_source_codes),
        "maximum_source_loss_fraction": float(chosen.maximum_source_loss_fraction),
        "maximum_target_gain_fraction": float(chosen.maximum_target_gain_fraction),
        "protected_bridge_gain_fraction": float(chosen.protected_bridge_gain_fraction),
        "island_maximum_mean_confidence": float(chosen.island_maximum_mean_confidence),
        "island_confidence_semantics": "explicit confidence if supplied else P[baseline_label]",
        "audit_proposal_limit": int(chosen.audit_proposal_limit),
        "algorithm_contract": {
            "topology_connectivity": 4,
            "dynamic_fragment_area_test": "0 < area_m2 < class_mmu_m2",
            "enclosed_island_area_test": (
                "0 < area_m2 <= island_cap_m2 and dynamic_fragment"
            ),
            "probability_p10_method": P10_METHOD,
            "probability_support": "minimum_gate_margin",
            "single_pass_from_frozen_baseline": True,
            "cascade_generation": False,
            "exact_cross_target_tie": "reject_ambiguous_group",
            "bridge_edge_distance": "euclidean_cell_polygon_edge_distance_m",
            "bridge_path_length": "four_neighbour_path_length_m",
            "budget_denominator": "class_budget_mask_and_valid_frozen_baseline",
        },
    }
    # Do not add this field to the V3.1-A snapshot: its approved SHA is a
    # frozen contract.  B has a distinct policy ID, so its mode can be safely
    # part of its independently recorded policy identity.
    if chosen.policy_id == V31B_POLICY_ID:
        snapshot["adjudication_mode"] = V31B_ADJUDICATION_MODE
        snapshot["algorithm_contract"]["adjudication"] = (
            "incremental_source_connectivity_target_attachment_and_component_gate"
        )
        snapshot["algorithm_contract"]["incremental_metric_gate"] = (
            "per_class_components_nonincreasing_global_components_and_dynamic_nonincreasing"
        )
    return snapshot


def policy_snapshot_sha256(policy: CandidatePolicy | None = None) -> str:
    body = json.dumps(policy_snapshot(policy), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _Component:
    component_id: int
    class_index: int
    class_code: int
    pixels: np.ndarray  # N, 2 row/column pairs
    touches_external: bool
    slices: tuple[slice, slice]


@dataclass(frozen=True)
class _Proposal:
    kind: str
    target_index: int
    target_code: int
    footprint: np.ndarray
    source_indices: tuple[int, ...]
    source_codes: tuple[int, ...]
    source_component_ids: tuple[int, ...]
    baseline_target_component_ids: tuple[int, ...]
    dynamic_reduction: int
    component_reduction: int
    probability_support: float
    area_m2: float
    digest: str
    proposal_id: str
    edge_distance_m: float | None
    path_length_m: float | None
    evidence: Mapping[str, float]
    discovery_count: int = 1
    discovery_edge_distances_m: tuple[float | None, ...] = ()
    discovery_path_lengths_m: tuple[float | None, ...] = ()
    occurrence_edge_distance_m: float | None = None
    occurrence_path_length_m: float | None = None


def _validate(
    labels: np.ndarray, class_codes: Sequence[int], valid_mask: np.ndarray | None,
    probabilities: np.ndarray | None, confidence: np.ndarray | None,
    pixel_area_m2: float, pixel_size_m: tuple[float, float] | None,
    policy: CandidatePolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, tuple[float, float]]:
    values = np.asarray(labels)
    if values.ndim != 2 or not class_codes:
        raise CandidateError("labels must be two-dimensional and class_codes cannot be empty")
    if len(set(int(v) for v in class_codes)) != len(class_codes):
        raise CandidateError("class_codes must be unique")
    valid = np.ones(values.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != values.shape:
        raise CandidateError("valid_mask shape does not match labels")
    if np.any(valid & ((values < 0) | (values >= len(class_codes)))):
        raise CandidateError("valid labels contain a class index outside class_codes")
    if not math.isfinite(pixel_area_m2) or pixel_area_m2 <= 0:
        raise CandidateError("pixel_area_m2 must be positive")
    if probabilities is None:
        raise CandidateError("V3.1-A requires the full probability cube for every proposal")
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.shape != (len(class_codes), *values.shape) or not np.all(np.isfinite(probs[:, valid])):
        raise CandidateError("probabilities must be finite with shape [len(class_codes), H, W]")
    if np.any(probs[:, valid] < 0) or np.any(probs[:, valid] > 1):
        raise CandidateError("probabilities must lie in [0, 1] over valid pixels")
    probability_sums = np.sum(probs[:, valid], axis=0, dtype=np.float64)
    if not np.allclose(probability_sums, 1.0, rtol=0.0, atol=1e-3):
        raise CandidateError(
            "probabilities must sum to one over every valid pixel"
        )
    conf = None if confidence is None else np.asarray(confidence, dtype=np.float32)
    if conf is not None and (conf.shape != values.shape or not np.all(np.isfinite(conf[valid]))):
        raise CandidateError("confidence must be finite and match labels")
    if conf is not None and (np.any(conf[valid] < 0) or np.any(conf[valid] > 1)):
        raise CandidateError("confidence must lie in [0, 1] over valid pixels")
    sizes = pixel_size_m or (math.sqrt(pixel_area_m2), math.sqrt(pixel_area_m2))
    if len(sizes) != 2 or any(not math.isfinite(float(v)) or float(v) <= 0 for v in sizes):
        raise CandidateError("pixel_size_m must contain positive row and column metre sizes")
    if not math.isclose(float(sizes[0]) * float(sizes[1]), float(pixel_area_m2), rel_tol=1e-9, abs_tol=1e-9):
        raise CandidateError("pixel_size_m product must equal pixel_area_m2")
    unknown = set(int(v) for v in class_codes) - set(policy.class_policies)
    if unknown:
        raise CandidateError(f"policy lacks class codes: {sorted(unknown)}")
    return values.astype(np.int16, copy=True), valid, probs, conf, (float(sizes[0]), float(sizes[1]))


def _class_budget_mask(
    class_budget_mask: np.ndarray | None,
    valid: np.ndarray,
) -> np.ndarray:
    """Return the Core-owner pixels eligible for frozen-class budgets.

    V3.1 proposals may inspect and temporarily modify halo pixels so that
    topology is evaluated with context.  A caller-supplied owner/Core mask,
    however, is the only region whose class-change budgets are charged and
    whose labels are released in the returned raster.
    """

    if class_budget_mask is None:
        return valid.copy()
    mask = np.asarray(class_budget_mask, dtype=bool)
    if mask.shape != valid.shape:
        raise CandidateError("class_budget_mask shape does not match labels")
    mask = mask & valid
    if not np.any(mask):
        raise CandidateError("class_budget_mask must contain at least one valid pixel")
    return mask


def _component_index(labels: np.ndarray, valid: np.ndarray, class_codes: Sequence[int]) -> tuple[np.ndarray, list[_Component]]:
    component_map = np.zeros(labels.shape, dtype=np.int32)
    components: list[_Component] = []
    component_id = 1
    height, width = labels.shape
    for index, code in enumerate(class_codes):
        local, count = ndimage.label(valid & (labels == index), structure=FOUR_CONNECTED)
        selected = local > 0
        component_map[selected] = local[selected].astype(np.int32) + component_id - 1
        objects = ndimage.find_objects(local, max_label=count)
        for local_id, slices in enumerate(objects, start=1):
            if slices is None:
                continue
            row_slice, col_slice = slices
            local_pixels = np.argwhere(local[row_slice, col_slice] == local_id)
            local_pixels[:, 0] += int(row_slice.start)
            local_pixels[:, 1] += int(col_slice.start)
            pixels = local_pixels.astype(np.int32, copy=False)
            rows, cols = pixels[:, 0], pixels[:, 1]
            external = bool(np.any(rows == 0) or np.any(cols == 0) or np.any(rows == height - 1) or np.any(cols == width - 1))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = rows + dr, cols + dc
                inside = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
                if np.any(~inside) or np.any(~valid[rr[inside], cc[inside]]):
                    external = True
                    break
            components.append(
                _Component(
                    component_id + local_id - 1,
                    index,
                    int(code),
                    pixels,
                    external,
                    (row_slice, col_slice),
                )
            )
        component_id += count
    if np.any(valid & (component_map == 0)):
        raise CandidateError("could not index every valid label")
    return component_map, components


def _footprint_digest(pixels: np.ndarray) -> str:
    ordered = np.asarray(sorted((int(r), int(c)) for r, c in pixels), dtype="<i4")
    return hashlib.sha256(ordered.tobytes()).hexdigest()


def _probability_evidence(
    footprint: np.ndarray, labels: np.ndarray, target_index: int, probs: np.ndarray,
    target_policy: ClassPolicy,
) -> tuple[bool, dict[str, float]]:
    rows, cols = footprint[:, 0], footprint[:, 1]
    target = probs[target_index, rows, cols].astype(np.float64, copy=False)
    # ``current`` is the probability for each pixel's actual frozen-baseline
    # label, not the most favourable alternative source class.
    current = probs[labels[rows, cols], rows, cols].astype(np.float64, copy=False)
    mean_target = float(np.mean(target))
    mean_drop = float(np.mean(current - target))
    rank = max(0, int(math.ceil(0.10 * len(target))) - 1)
    p10 = float(np.partition(target, rank)[rank])
    gate_margin = min(
        mean_target - target_policy.minimum_target_probability_mean,
        target_policy.maximum_current_minus_target_probability_mean - mean_drop,
        p10 - target_policy.minimum_target_probability_p10,
    )
    values = {
        "mean_target_probability": mean_target,
        "mean_current_minus_target": mean_drop,
        "p10_target_probability": p10,
        "minimum_probability_gate_margin": float(gate_margin),
    }
    return (
        mean_target >= target_policy.minimum_target_probability_mean
        and mean_drop <= target_policy.maximum_current_minus_target_probability_mean
        and p10 >= target_policy.minimum_target_probability_p10,
        values,
    )


def _dynamic_count(labels: np.ndarray, valid: np.ndarray, class_codes: Sequence[int], policy: CandidatePolicy, pixel_area_m2: float) -> tuple[int, int]:
    _map, components = _component_index(labels, valid, class_codes)
    dynamic = sum(
        len(item.pixels) * pixel_area_m2 < policy.class_policies[item.class_code].dynamic_fragmentation_m2
        for item in components
    )
    return int(dynamic), len(components)


def _per_class_metrics(
    labels: np.ndarray,
    valid: np.ndarray,
    class_codes: Sequence[int],
    policy: CandidatePolicy,
    pixel_area_m2: float,
) -> dict[int, dict[str, float | int]]:
    _component_map, components = _component_index(labels, valid, class_codes)
    pixel_counts = np.bincount(labels[valid], minlength=len(class_codes))
    result: dict[int, dict[str, float | int]] = {
        int(code): {
            "pixel_count": int(pixel_counts[index]),
            "area_m2": float(pixel_counts[index] * pixel_area_m2),
            "component_count_4_connected": 0,
            "dynamic_fragment_count_4_connected": 0,
            "dynamic_fragment_area_m2": 0.0,
        }
        for index, code in enumerate(class_codes)
    }
    for component in components:
        area_m2 = float(len(component.pixels) * pixel_area_m2)
        metrics = result[component.class_code]
        metrics["component_count_4_connected"] = int(
            metrics["component_count_4_connected"]
        ) + 1
        if area_m2 < policy.class_policies[component.class_code].dynamic_fragmentation_m2:
            metrics["dynamic_fragment_count_4_connected"] = int(
                metrics["dynamic_fragment_count_4_connected"]
            ) + 1
            metrics["dynamic_fragment_area_m2"] = float(
                metrics["dynamic_fragment_area_m2"]
            ) + area_m2
    return result


def _source_connectivity_safe(
    footprint: np.ndarray, labels: np.ndarray, component_map: np.ndarray,
    components: Sequence[_Component], valid: np.ndarray,
) -> bool:
    removed = {(int(r), int(c)) for r, c in footprint}
    removed_by_component: dict[int, set[tuple[int, int]]] = {}
    for row, col in removed:
        component_id = int(component_map[row, col])
        if component_id:
            removed_by_component.setdefault(component_id, set()).add((row, col))
    by_id = {component.component_id: component for component in components}
    height, width = component_map.shape
    for component_id, local_removed in removed_by_component.items():
        component = by_id[component_id]
        if len(local_removed) >= len(component.pixels):
            continue
        boundary: set[tuple[int, int]] = set()
        for row, col in local_removed:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = row + dr, col + dc
                if (
                    0 <= rr < height
                    and 0 <= cc < width
                    and int(component_map[rr, cc]) == component_id
                    and (rr, cc) not in local_removed
                ):
                    boundary.add((rr, cc))
        if not boundary:
            return False

        component_rows, component_cols = component.slices
        component_row0, component_row1 = int(component_rows.start), int(component_rows.stop)
        component_col0, component_col1 = int(component_cols.start), int(component_cols.stop)
        relevant = local_removed | boundary
        base_row0 = min(row for row, _col in relevant)
        base_row1 = max(row for row, _col in relevant) + 1
        base_col0 = min(col for _row, col in relevant)
        base_col1 = max(col for _row, col in relevant) + 1
        padding = 1
        while True:
            row0 = max(component_row0, base_row0 - padding)
            row1 = min(component_row1, base_row1 + padding)
            col0 = max(component_col0, base_col0 - padding)
            col1 = min(component_col1, base_col1 + padding)
            local = component_map[row0:row1, col0:col1] == component_id
            for row, col in local_removed:
                if row0 <= row < row1 and col0 <= col < col1:
                    local[row - row0, col - col0] = False
            labeled, _count = ndimage.label(local, structure=FOUR_CONNECTED)
            boundary_ids = {
                int(labeled[row - row0, col - col0]) for row, col in boundary
            }
            boundary_ids.discard(0)
            if len(boundary_ids) == 1:
                break
            full_component_window = (
                row0 == component_row0
                and row1 == component_row1
                and col0 == component_col0
                and col1 == component_col1
            )
            if full_component_window:
                return False
            padding *= 2
    return True


def _target_component_ids_for_footprint(
    footprint: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    component_map: np.ndarray,
    target_index: int,
    seed_ids: Sequence[int] = (),
) -> tuple[int, ...]:
    """Return baseline target components touched by the connected footprint."""

    height, width = labels.shape
    ids = {int(value) for value in seed_ids}
    for r, c in footprint:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < height and 0 <= cc < width and valid[rr, cc] and labels[rr, cc] == target_index:
                ids.add(int(component_map[rr, cc]))
    return tuple(sorted(value for value in ids if value))


def _local_topology_delta(
    footprint: np.ndarray,
    component_map: np.ndarray,
    components_by_id: Mapping[int, _Component],
    target_component_ids: Sequence[int],
    policy: CandidatePolicy,
    pixel_area_m2: float,
) -> tuple[int, int]:
    """Exact proposal delta without copying/re-labelling the full raster.

    Proposal generation already guarantees a connected footprint, a connected
    remaining bridge source, and target contact.  Therefore only directly
    touched baseline components can change count or dynamic-fragment status.
    """

    ids, removed_counts = np.unique(
        component_map[footprint[:, 0], footprint[:, 1]], return_counts=True
    )
    component_reduction = 0
    dynamic_reduction = 0
    for component_id, removed_count in zip(ids, removed_counts):
        if not component_id:
            continue
        source = components_by_id[int(component_id)]
        before_dynamic = len(source.pixels) * pixel_area_m2 < policy.class_policies[source.class_code].dynamic_fragmentation_m2
        remaining = len(source.pixels) - int(removed_count)
        after_dynamic = remaining > 0 and remaining * pixel_area_m2 < policy.class_policies[source.class_code].dynamic_fragmentation_m2
        dynamic_reduction += int(before_dynamic) - int(after_dynamic)
        if remaining == 0:
            component_reduction += 1
    targets = [components_by_id[int(value)] for value in target_component_ids]
    if targets:
        target_policy = policy.class_policies[targets[0].class_code]
        before_dynamic = sum(
            len(component.pixels) * pixel_area_m2 < target_policy.dynamic_fragmentation_m2
            for component in targets
        )
        after_dynamic = (
            (sum(len(component.pixels) for component in targets) + len(footprint))
            * pixel_area_m2 < target_policy.dynamic_fragmentation_m2
        )
        dynamic_reduction += int(before_dynamic) - int(after_dynamic)
        component_reduction += len(targets) - 1
    return int(dynamic_reduction), int(component_reduction)


def _proposal_with_scores(
    kind: str, target_index: int, footprint: np.ndarray, labels: np.ndarray,
    valid: np.ndarray, class_codes: Sequence[int], policy: CandidatePolicy,
    pixel_area_m2: float, probs: np.ndarray, target_policy: ClassPolicy,
    component_map: np.ndarray, edge_distance_m: float | None = None,
    path_length_m: float | None = None,
    target_component_seed_ids: Sequence[int] = (),
    components_by_id: Mapping[int, _Component] | None = None,
    extra_evidence: Mapping[str, float] | None = None,
    generation_rejections: Counter[str] | None = None,
) -> _Proposal | None:
    # A footprint must change baseline labels and consists only of valid pixels.
    footprint = np.unique(np.asarray(footprint, dtype=np.int32), axis=0)
    if len(footprint) == 0:
        return None
    rows, cols = footprint[:, 0], footprint[:, 1]
    if not np.all(valid[rows, cols]) or np.any(labels[rows, cols] == target_index):
        return None
    allowed, evidence = _probability_evidence(footprint, labels, target_index, probs, target_policy)
    if not allowed:
        if generation_rejections is not None:
            if evidence["mean_target_probability"] < target_policy.minimum_target_probability_mean:
                generation_rejections["probability_mean_target"] += 1
            if evidence["mean_current_minus_target"] > target_policy.maximum_current_minus_target_probability_mean:
                generation_rejections["probability_current_minus_target"] += 1
            if evidence["p10_target_probability"] < target_policy.minimum_target_probability_p10:
                generation_rejections["probability_p10_target"] += 1
        return None
    if extra_evidence:
        evidence.update({str(key): float(value) for key, value in extra_evidence.items()})
    sources = tuple(sorted(set(int(v) for v in labels[rows, cols])))
    source_component_ids = tuple(sorted(set(int(v) for v in component_map[rows, cols] if v)))
    target_component_ids = _target_component_ids_for_footprint(
        footprint, labels, valid, component_map, target_index, target_component_seed_ids,
    )
    if not target_component_ids or components_by_id is None:
        return None
    dynamic_reduction, component_reduction = _local_topology_delta(
        footprint, component_map, components_by_id, target_component_ids, policy,
        pixel_area_m2,
    )
    digest = _footprint_digest(footprint)
    proposal_id_body = json.dumps(
        {
            "kind": kind,
            "target_class_code": int(class_codes[target_index]),
            "source_component_ids": source_component_ids,
            "footprint_sha256": digest,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return _Proposal(
        kind=kind, target_index=int(target_index), target_code=int(class_codes[target_index]),
        footprint=footprint, source_indices=sources,
        source_codes=tuple(int(class_codes[v]) for v in sources),
        source_component_ids=source_component_ids,
        baseline_target_component_ids=target_component_ids,
        dynamic_reduction=dynamic_reduction,
        component_reduction=component_reduction,
        probability_support=evidence["minimum_probability_gate_margin"],
        area_m2=float(len(footprint) * pixel_area_m2), digest=digest,
        proposal_id=hashlib.sha256(proposal_id_body.encode("utf-8")).hexdigest(),
        edge_distance_m=edge_distance_m, path_length_m=path_length_m,
        evidence=evidence,
    )


def _island_proposals(
    labels: np.ndarray, valid: np.ndarray, probs: np.ndarray, confidence: np.ndarray | None,
    class_codes: Sequence[int], component_map: np.ndarray, components: Sequence[_Component],
    policy: CandidatePolicy, pixel_area_m2: float, generation_rejections: Counter[str],
) -> list[_Proposal]:
    by_id = {item.component_id: item for item in components}
    proposals: list[_Proposal] = []
    height, width = labels.shape
    for source in components:
        source_policy = policy.class_policies[source.class_code]
        area = len(source.pixels) * pixel_area_m2
        if source.class_code in policy.protected_source_codes or source_policy.ordinary_protected:
            generation_rejections["protected_source"] += 1
            continue
        if source.touches_external:
            generation_rejections["island_external_boundary"] += 1
            continue
        if source_policy.enclosed_island_max_m2 <= 0:
            generation_rejections["island_disabled"] += 1
            continue
        if area > source_policy.enclosed_island_max_m2:
            generation_rejections["island_area_cap"] += 1
            continue
        if area >= source_policy.dynamic_fragmentation_m2:
            generation_rejections["island_not_dynamic_fragment"] += 1
            continue
        source_rows, source_cols = source.pixels[:, 0], source.pixels[:, 1]
        source_confidence = float(
            np.mean(
                confidence[source_rows, source_cols]
                if confidence is not None
                else probs[source.class_index, source_rows, source_cols]
            )
        )
        if source_confidence > policy.island_maximum_mean_confidence:
            generation_rejections["island_confidence"] += 1
            continue
        neighbours: set[int] = set()
        rejected = False
        for r, c in source.pixels:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = int(r + dr), int(c + dc)
                if rr < 0 or rr >= height or cc < 0 or cc >= width or not valid[rr, cc]:
                    rejected = True
                    break
                neighbour_id = int(component_map[rr, cc])
                if neighbour_id != source.component_id:
                    neighbours.add(neighbour_id)
            if rejected:
                break
        if rejected or len(neighbours) != 1:
            generation_rejections["island_not_uniquely_enclosed"] += 1
            continue
        target = by_id[next(iter(neighbours))]
        allowed_codes = policy.semantic_compatible_targets.get(source.class_code, frozenset())
        if target.class_code not in allowed_codes:
            generation_rejections["semantic_incompatible_target"] += 1
            continue
        if policy.class_policies[target.class_code].ordinary_protected:
            generation_rejections["protected_ordinary_target"] += 1
            continue
        proposal = _proposal_with_scores(
            "enclosed_island", target.class_index, source.pixels, labels, valid,
            class_codes, policy, pixel_area_m2, probs,
            policy.class_policies[target.class_code], component_map,
            target_component_seed_ids=(target.component_id,), components_by_id=by_id,
            extra_evidence={"mean_source_confidence": source_confidence},
            generation_rejections=generation_rejections,
        )
        if proposal is not None:
            proposals.append(proposal)
    return proposals


def _trace(predecessor: np.ndarray, cell: tuple[int, int]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    r, c = cell
    while r >= 0:
        result.append((r, c))
        r, c = (int(predecessor[r, c, 0]), int(predecessor[r, c, 1]))
    return result


def _cell_polygon_edge_distance_m(
    first: _Component,
    second: _Component,
    pixel_size_m: tuple[float, float],
    maximum_distance_m: float,
) -> float | None:
    """Exact axis-aligned pixel-polygon edge distance within a bounded radius."""

    row_m, col_m = pixel_size_m
    first_xy = first.pixels.astype(np.float64) * np.array((row_m, col_m))
    second_xy = second.pixels.astype(np.float64) * np.array((row_m, col_m))
    radius = maximum_distance_m + math.hypot(row_m, col_m)
    tree = cKDTree(second_xy)
    nearest = tree.query_ball_point(first_xy, r=radius)
    best = math.inf
    for point, candidates in zip(first.pixels, nearest):
        if not candidates:
            continue
        other = second.pixels[np.asarray(candidates, dtype=np.int32)]
        row_gap = np.maximum(0.0, np.abs(other[:, 0] - point[0]) - 1.0) * row_m
        col_gap = np.maximum(0.0, np.abs(other[:, 1] - point[1]) - 1.0) * col_m
        best = min(best, float(np.min(np.hypot(row_gap, col_gap))))
    return None if not math.isfinite(best) else best


def _bridge_proposals_for_code(
    target_index: int, labels: np.ndarray, valid: np.ndarray, probs: np.ndarray,
    class_codes: Sequence[int], component_map: np.ndarray, components: Sequence[_Component],
    policy: CandidatePolicy, pixel_area_m2: float, pixel_size_m: tuple[float, float],
    generation_rejections: Counter[str],
) -> list[_Proposal]:
    code = int(class_codes[target_index])
    row = policy.class_policies[code]
    if not row.allow_same_class_bridge or row.bridge_max_edge_distance_m <= 0 or row.bridge_max_new_footprint_m2 <= 0:
        generation_rejections["bridge_disabled"] += 1
        return []
    target_components = [item for item in components if item.class_index == target_index]
    if len(target_components) < 2:
        generation_rejections["bridge_insufficient_components"] += 1
        return []
    h, w = labels.shape
    by_id = {item.component_id: item for item in components}
    owner = np.zeros((h, w), dtype=np.int32)
    distance = np.full((h, w), np.inf, dtype=np.float64)
    predecessor = np.full((h, w, 2), -1, dtype=np.int32)
    queue: list[tuple[float, int, int, int]] = []
    for component in target_components:
        for r, c in component.pixels:
            owner[r, c] = component.component_id
            distance[r, c] = 0.0
            heapq.heappush(queue, (0.0, component.component_id, int(r), int(c)))
    step_lengths = ((-1, 0, pixel_size_m[0]), (1, 0, pixel_size_m[0]), (0, -1, pixel_size_m[1]), (0, 1, pixel_size_m[1]))
    examined_pairs: set[tuple[int, int]] = set()
    proposals: list[_Proposal] = []
    max_distance = float(row.bridge_max_edge_distance_m)
    # Manhattan path growth must cover diagonally separated cell polygons.
    search_limit = max_distance * math.sqrt(2.0) + 2.0 * max(pixel_size_m)
    while queue:
        current_distance, component_id, r, c = heapq.heappop(queue)
        if current_distance != distance[r, c] or component_id != owner[r, c] or current_distance > search_limit:
            continue
        for dr, dc, step in step_lengths:
            rr, cc = r + dr, c + dc
            if rr < 0 or rr >= h or cc < 0 or cc >= w or not valid[rr, cc]:
                continue
            # A protected baseline class is an immutable bridge obstacle.
            neighbour_code = int(class_codes[int(labels[rr, cc])])
            if neighbour_code in policy.protected_source_codes and int(labels[rr, cc]) != target_index:
                generation_rejections["protected_source"] += 1
                continue
            candidate_distance = current_distance + step
            other_owner = int(owner[rr, cc])
            if other_owner and other_owner != component_id:
                pair = tuple(sorted((component_id, other_owner)))
                if pair in examined_pairs:
                    continue
                examined_pairs.add(pair)
                path_length = candidate_distance + float(distance[rr, cc]) - step
                edge_distance = _cell_polygon_edge_distance_m(
                    by_id[pair[0]], by_id[pair[1]], pixel_size_m, max_distance,
                )
                if edge_distance is None or edge_distance > max_distance:
                    generation_rejections["bridge_distance"] += 1
                    continue
                cells = _trace(predecessor, (r, c)) + _trace(predecessor, (rr, cc))
                footprint = np.array([cell for cell in cells if labels[cell] != target_index], dtype=np.int32)
                if len(footprint) == 0 or len(footprint) * pixel_area_m2 > row.bridge_max_new_footprint_m2:
                    generation_rejections["bridge_footprint"] += 1
                    continue
                if not _source_connectivity_safe(footprint, labels, component_map, components, valid):
                    generation_rejections["source_connectivity"] += 1
                    continue
                proposal = _proposal_with_scores(
                    "same_class_bridge", target_index, footprint, labels,
                    valid, class_codes, policy, pixel_area_m2, probs, row,
                    component_map, edge_distance_m=float(edge_distance),
                    path_length_m=float(path_length),
                    target_component_seed_ids=pair, components_by_id=by_id,
                    generation_rejections=generation_rejections,
                )
                if proposal is not None and proposal.dynamic_reduction >= 0 and proposal.component_reduction > 0:
                    proposals.append(proposal)
                continue
            if candidate_distance < distance[rr, cc] and candidate_distance <= search_limit:
                owner[rr, cc] = component_id
                distance[rr, cc] = candidate_distance
                predecessor[rr, cc] = (r, c)
                heapq.heappush(queue, (candidate_distance, component_id, rr, cc))
    return proposals


def _rank_key(proposal: _Proposal) -> tuple[float, float, float, float, str]:
    """The frozen policy ranking, expressed in ascending sort order."""

    return (
        -float(proposal.dynamic_reduction),
        -float(proposal.component_reduction),
        -float(proposal.probability_support),
        float(proposal.area_m2),
        proposal.digest,
    )


def _summary(
    proposal: _Proposal, *, decision: str = "generated", reason: str = "pending_adjudication",
) -> dict[str, Any]:
    rows, cols = proposal.footprint[:, 0], proposal.footprint[:, 1]
    return {
        "proposal_id": proposal.proposal_id,
        "kind": proposal.kind, "target_class_code": proposal.target_code,
        "source_class_codes": list(proposal.source_codes),
        "baseline_source_component_ids": list(proposal.source_component_ids),
        "baseline_target_component_ids": list(proposal.baseline_target_component_ids),
        "changed_pixels": int(len(proposal.footprint)), "area_m2": proposal.area_m2,
        "footprint_bbox": [int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())],
        "edge_distance_m": proposal.edge_distance_m,
        "path_length_m": proposal.path_length_m,
        "discovery_count": int(proposal.discovery_count),
        "discovery_edge_distances_m": list(
            proposal.discovery_edge_distances_m or (proposal.edge_distance_m,)
        ),
        "discovery_path_lengths_m": list(
            proposal.discovery_path_lengths_m or (proposal.path_length_m,)
        ),
        "occurrence_edge_distance_m": (
            proposal.edge_distance_m
            if proposal.occurrence_edge_distance_m is None
            else proposal.occurrence_edge_distance_m
        ),
        "occurrence_path_length_m": (
            proposal.path_length_m
            if proposal.occurrence_path_length_m is None
            else proposal.occurrence_path_length_m
        ),
        "dynamic_fragment_reduction": proposal.dynamic_reduction,
        "component_reduction": proposal.component_reduction,
        "probability_support": proposal.probability_support,
        "stable_rank_key": list(_rank_key(proposal)),
        "footprint_sha256": proposal.digest, "evidence": dict(proposal.evidence),
        "decision": decision, "reason": reason,
    }


def _final_topology_holds(
    baseline: np.ndarray,
    result: np.ndarray,
    valid: np.ndarray,
    class_codes: Sequence[int],
    policy: CandidatePolicy,
    pixel_area_m2: float,
    accepted: Sequence[_Proposal],
    components: Sequence[_Component],
) -> bool:
    before = _per_class_metrics(baseline, valid, class_codes, policy, pixel_area_m2)
    after = _per_class_metrics(result, valid, class_codes, policy, pixel_area_m2)
    if any(
        after[int(code)]["component_count_4_connected"] > before[int(code)]["component_count_4_connected"]
        for code in class_codes
    ):
        return False
    before_dynamic, before_components = _dynamic_count(baseline, valid, class_codes, policy, pixel_area_m2)
    after_dynamic, after_components = _dynamic_count(result, valid, class_codes, policy, pixel_area_m2)
    if after_components > before_components or after_dynamic > before_dynamic:
        return False
    output_components, _items = _component_index(result, valid, class_codes)
    by_id = {item.component_id: item for item in components}
    for proposal in accepted:
        footprint_rows = proposal.footprint[:, 0]
        footprint_cols = proposal.footprint[:, 1]
        if np.any(result[footprint_rows, footprint_cols] != proposal.target_index):
            return False
        result_ids = {
            int(value)
            for value in output_components[footprint_rows, footprint_cols]
            if value
        }
        for component_id in proposal.baseline_target_component_ids:
            pixels = by_id[component_id].pixels
            rows, cols = pixels[:, 0], pixels[:, 1]
            retained = result[rows, cols] == proposal.target_index
            if proposal.kind == "same_class_bridge" and not np.all(retained):
                return False
            if not np.any(retained):
                return False
            result_ids.update(
                int(value)
                for value in output_components[rows[retained], cols[retained]]
                if value
            )
        if len(result_ids) != 1:
            return False
    return True


def _adjudicate(
    proposals: Sequence[_Proposal], labels: np.ndarray, class_codes: Sequence[int], valid: np.ndarray,
    policy: CandidatePolicy, pixel_area_m2: float, component_map: np.ndarray,
    components: Sequence[_Component], class_budget_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[_Proposal], Counter[str], dict[str, tuple[str, str]], int]:
    budget_mask = valid if class_budget_mask is None else class_budget_mask & valid
    baseline_totals = Counter(int(class_codes[index]) for index in labels[budget_mask])
    source_loss: Counter[int] = Counter()
    target_gain: Counter[int] = Counter()
    accepted: list[_Proposal] = []
    occupied = np.zeros(labels.shape, dtype=bool)
    result = labels.copy()
    skipped: Counter[str] = Counter()
    decisions: dict[str, tuple[str, str]] = {}
    rank_groups: dict[tuple[float, float, float, float, str], list[_Proposal]] = {}
    for proposal in proposals:
        rank_groups.setdefault(_rank_key(proposal), []).append(proposal)
    ambiguous = {
        proposal.proposal_id
        for group in rank_groups.values()
        if len({item.target_index for item in group}) > 1
        for proposal in group
    }
    ordered = sorted(proposals, key=lambda item: (*_rank_key(item), item.proposal_id))
    for proposal in ordered:
        rows, cols = proposal.footprint[:, 0], proposal.footprint[:, 1]
        # A Core owner publishes whole topology proposals only.  Retaining a
        # clipped island or bridge would make its recorded topology evidence
        # false and could let a halo-only proposal reserve a Core budget.
        if not np.all(budget_mask[rows, cols]):
            skipped["outside_core_owner"] += 1
            decisions[proposal.proposal_id] = ("rejected", "outside_core_owner")
            continue
        if proposal.proposal_id in ambiguous:
            skipped["ambiguous_target_tie"] += 1
            decisions[proposal.proposal_id] = ("rejected", "ambiguous_target_tie")
            continue
        if np.any(occupied[rows, cols]):
            skipped["footprint_conflict"] += 1
            decisions[proposal.proposal_id] = ("rejected", "footprint_conflict")
            continue
        core = budget_mask[rows, cols]
        changed_codes = [
            int(class_codes[int(v)]) for v in labels[rows[core], cols[core]]
        ]
        additions = Counter(changed_codes)
        target_budget = policy.protected_bridge_gain_fraction if proposal.kind == "same_class_bridge" and proposal.target_code in policy.protected_source_codes else policy.maximum_target_gain_fraction
        if any(source_loss[code] + count > baseline_totals[code] * policy.maximum_source_loss_fraction + 1e-12 for code, count in additions.items()):
            skipped["source_budget"] += 1
            decisions[proposal.proposal_id] = ("rejected", "source_budget")
            continue
        core_gain = int(np.count_nonzero(core))
        if target_gain[proposal.target_code] + core_gain > baseline_totals[proposal.target_code] * target_budget + 1e-12:
            skipped["target_budget"] += 1
            decisions[proposal.proposal_id] = ("rejected", "target_budget")
            continue
        prospective = occupied.copy()
        prospective[rows, cols] = True
        # Independent proposals may remove different pixels from the same
        # baseline component.  Recheck their union only after the cheap frozen
        # budget gates, so known-over-budget proposals do not trigger topology
        # work on large candidate sets.
        if not _source_connectivity_safe(np.argwhere(prospective), labels, component_map, components, valid):
            skipped["source_connectivity"] += 1
            decisions[proposal.proposal_id] = ("rejected", "source_connectivity")
            continue
        occupied[rows, cols] = True
        result[rows, cols] = proposal.target_index
        accepted.append(proposal)
        decisions[proposal.proposal_id] = ("accepted", "selected")
        source_loss.update(additions)
        target_gain[proposal.target_code] += core_gain
    rollback_count = 0
    while accepted and not _final_topology_holds(
        labels, result, valid, class_codes, policy, pixel_area_m2, accepted, components,
    ):
        proposal = accepted.pop()
        rows, cols = proposal.footprint[:, 0], proposal.footprint[:, 1]
        result[rows, cols] = labels[rows, cols]
        decisions[proposal.proposal_id] = ("rejected", "final_topology_rollback")
        skipped["final_topology_rollback"] += 1
        rollback_count += 1
    return result, accepted, skipped, decisions, rollback_count


def _canonical_proposal_key(proposal: _Proposal) -> tuple[Any, ...]:
    """Identity of a proposal before it enters the B dependency graph."""

    return (
        proposal.kind,
        int(proposal.target_index),
        int(proposal.target_code),
        tuple(sorted((int(row), int(col)) for row, col in proposal.footprint)),
        tuple(int(value) for value in proposal.source_indices),
        tuple(int(value) for value in proposal.source_codes),
        tuple(int(value) for value in proposal.source_component_ids),
        tuple(int(value) for value in proposal.baseline_target_component_ids),
    )


def _proposal_score_signature(proposal: _Proposal) -> tuple[Any, ...]:
    """Every ranked/audited field that must agree for one topology identity."""

    return (
        _rank_key(proposal),
        int(proposal.dynamic_reduction),
        int(proposal.component_reduction),
        float(proposal.probability_support),
        float(proposal.area_m2),
        proposal.digest,
        json.dumps(dict(proposal.evidence), sort_keys=True, separators=(",", ":")),
    )


def _discovery_order_key(proposal: _Proposal) -> tuple[str, float, float]:
    """Deterministically choose an original discovery occurrence."""

    return (
        proposal.proposal_id,
        float("inf") if proposal.edge_distance_m is None else float(proposal.edge_distance_m),
        float("inf") if proposal.path_length_m is None else float(proposal.path_length_m),
    )


def _minimum_discovery_distance(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return min(present) if present else None


def _canonicalize_v31b_proposals(
    proposals: Sequence[_Proposal],
) -> tuple[list[_Proposal], Counter[str], list[dict[str, Any]]]:
    """Deduplicate exact candidates and reject accidental ID collisions.

    The proposal action, rather than a generator-specific proposal ID, is the
    semantic identity.  Keeping one stable representative makes decisions and
    the interaction audit one-to-one even if two generators discover the same
    path under different IDs.
    """

    grouped: dict[tuple[Any, ...], list[_Proposal]] = {}
    duplicates: Counter[str] = Counter()
    duplicate_audit: list[dict[str, Any]] = []
    ids: dict[str, tuple[Any, ...]] = {}
    for proposal in proposals:
        key = _canonical_proposal_key(proposal)
        previous_key = ids.get(proposal.proposal_id)
        if previous_key is not None and previous_key != key:
            raise CandidateError(
                "V3.1-B proposal_id collision between non-identical proposals"
            )
        ids[proposal.proposal_id] = key
        grouped.setdefault(key, []).append(proposal)
    unique: list[_Proposal] = []
    for key, group in grouped.items():
        # The proposal ID is not semantic.  Pick a stable representative so
        # reversing generator order cannot alter a canonical action or audit.
        ordered_group = sorted(group, key=_discovery_order_key)
        original_representative = ordered_group[0]
        edge_distances = tuple(sorted(
            (item.edge_distance_m for item in ordered_group),
            key=lambda value: (value is None, float("inf") if value is None else float(value)),
        ))
        path_lengths = tuple(sorted(
            (item.path_length_m for item in ordered_group),
            key=lambda value: (value is None, float("inf") if value is None else float(value)),
        ))
        representative = replace(
            original_representative,
            edge_distance_m=_minimum_discovery_distance(edge_distances),
            path_length_m=_minimum_discovery_distance(path_lengths),
            discovery_count=len(ordered_group),
            discovery_edge_distances_m=edge_distances,
            discovery_path_lengths_m=path_lengths,
            occurrence_edge_distance_m=original_representative.edge_distance_m,
            occurrence_path_length_m=original_representative.path_length_m,
        )
        for ordinal, proposal in enumerate(ordered_group[1:], start=1):
            if _proposal_score_signature(proposal) != _proposal_score_signature(original_representative):
                raise CandidateError(
                    "V3.1-B duplicate proposal topology has inconsistent rank or evidence"
                )
            duplicates["duplicate_proposal"] += 1
            duplicate_audit.append({
                "occurrence_id": f"{proposal.proposal_id}:duplicate:{ordinal}",
                "proposal_id": proposal.proposal_id,
                "canonical_proposal_id": representative.proposal_id,
                "decision": "rejected",
                "reason": "duplicate_proposal",
                "stable_rank_key": list(_rank_key(proposal)),
                "footprint_sha256": proposal.digest,
                "discovery_count": 1,
                "edge_distance_m": proposal.edge_distance_m,
                "path_length_m": proposal.path_length_m,
                "occurrence_edge_distance_m": proposal.edge_distance_m,
                "occurrence_path_length_m": proposal.path_length_m,
                "canonical_edge_distance_m": representative.edge_distance_m,
                "canonical_path_length_m": representative.path_length_m,
            })
        unique.append(representative)
    duplicate_audit.sort(key=lambda item: str(item["occurrence_id"]))
    return (
        sorted(unique, key=lambda item: (*_rank_key(item), item.proposal_id)),
        duplicates,
        duplicate_audit,
    )


def _source_connectivity_safe_incremental(
    removed_by_component: Mapping[int, set[tuple[int, int]]],
    changed_component_ids: Sequence[int],
    component_map: np.ndarray,
    components_by_id: Mapping[int, _Component],
) -> bool:
    """Check only baseline source components changed by the new proposal."""

    height, width = component_map.shape
    for component_id in sorted(set(int(value) for value in changed_component_ids)):
        local_removed = removed_by_component.get(component_id, set())
        if not local_removed:
            continue
        component = components_by_id[component_id]
        if len(local_removed) >= len(component.pixels):
            continue
        boundary: set[tuple[int, int]] = set()
        for row, col in local_removed:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = row + dr, col + dc
                if (
                    0 <= rr < height
                    and 0 <= cc < width
                    and int(component_map[rr, cc]) == component_id
                    and (rr, cc) not in local_removed
                ):
                    boundary.add((rr, cc))
        if not boundary:
            return False
        row_slice, col_slice = component.slices
        component_row0, component_row1 = int(row_slice.start), int(row_slice.stop)
        component_col0, component_col1 = int(col_slice.start), int(col_slice.stop)
        relevant = local_removed | boundary
        base_row0 = min(row for row, _ in relevant)
        base_row1 = max(row for row, _ in relevant) + 1
        base_col0 = min(col for _, col in relevant)
        base_col1 = max(col for _, col in relevant) + 1
        padding = 1
        while True:
            row0 = max(component_row0, base_row0 - padding)
            row1 = min(component_row1, base_row1 + padding)
            col0 = max(component_col0, base_col0 - padding)
            col1 = min(component_col1, base_col1 + padding)
            local = component_map[row0:row1, col0:col1] == component_id
            for row, col in local_removed:
                if row0 <= row < row1 and col0 <= col < col1:
                    local[row - row0, col - col0] = False
            labeled, _count = ndimage.label(local, structure=FOUR_CONNECTED)
            boundary_ids = {
                int(labeled[row - row0, col - col0]) for row, col in boundary
            }
            boundary_ids.discard(0)
            if len(boundary_ids) == 1:
                break
            if (
                row0 == component_row0
                and row1 == component_row1
                and col0 == component_col0
                and col1 == component_col1
            ):
                return False
            padding *= 2
    return True


def _target_attachment_safe_incremental(
    proposal: _Proposal,
    result: np.ndarray,
    labels: np.ndarray,
    component_map: np.ndarray,
    components_by_id: Mapping[int, _Component],
    proposals_by_id: Mapping[str, _Proposal],
    proposal_pixel_owner: Mapping[tuple[int, int], str],
    target_dependents: Mapping[int, Sequence[_Proposal]],
) -> bool:
    """Prove one proposal is connected to every residual target anchor.

    Source connectivity proves each residual baseline component remains one
    component.  The local graph has proposal footprints and residual baseline
    target components as nodes.  Its edges are current 4-neighbour contacts,
    so an island may retain an indirect proposal-to-anchor route after one old
    direct contact disappears.  This avoids both a full output component index
    and the false rejection caused by requiring an island's original edge.
    Same-class bridges still retain *all* baseline target pixels.
    """

    rows, cols = proposal.footprint[:, 0], proposal.footprint[:, 1]
    if np.any(result[rows, cols] != proposal.target_index):
        return False
    height, width = result.shape
    direct_cache: dict[tuple[str, int], bool] = {}

    def directly_attached(item: _Proposal, component_id: int) -> bool:
        key = (item.proposal_id, int(component_id))
        if key in direct_cache:
            return direct_cache[key]
        for row, col in item.footprint:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = int(row + dr), int(col + dc)
                if (
                    0 <= rr < height and 0 <= cc < width
                    and result[rr, cc] == item.target_index
                    and int(component_map[rr, cc]) == int(component_id)
                ):
                    direct_cache[key] = True
                    return True
        direct_cache[key] = False
        return False

    def connected_to_anchor(component_id: int) -> bool:
        queue: list[tuple[str, str | int]] = [("proposal", proposal.proposal_id)]
        visited: set[tuple[str, str | int]] = set()
        while queue:
            node_kind, node_id = queue.pop()
            node = (node_kind, node_id)
            if node in visited:
                continue
            visited.add(node)
            if node_kind == "component":
                if int(node_id) == int(component_id):
                    return True
                for dependent in target_dependents.get(int(node_id), []):
                    if (
                        dependent.target_index == proposal.target_index
                        and directly_attached(dependent, int(node_id))
                    ):
                        queue.append(("proposal", dependent.proposal_id))
                continue
            item = proposals_by_id[str(node_id)]
            for target_component_id in item.baseline_target_component_ids:
                if directly_attached(item, int(target_component_id)):
                    queue.append(("component", int(target_component_id)))
            for row, col in item.footprint:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = int(row + dr), int(col + dc)
                    if not (0 <= rr < height and 0 <= cc < width):
                        continue
                    if result[rr, cc] != proposal.target_index:
                        continue
                    owner = proposal_pixel_owner.get((rr, cc))
                    if owner is not None and owner != item.proposal_id:
                        queue.append(("proposal", owner))
        return False

    for component_id in proposal.baseline_target_component_ids:
        component = components_by_id[int(component_id)]
        component_rows, component_cols = component.pixels[:, 0], component.pixels[:, 1]
        retained = result[component_rows, component_cols] == proposal.target_index
        if proposal.kind == "same_class_bridge" and not np.all(retained):
            return False
        if not np.any(retained):
            return False
        if not connected_to_anchor(int(component_id)):
            return False
    return True


@dataclass
class _IncrementalMetricState:
    """Exact component-size bookkeeping for accepted baseline/proposal groups."""

    parent: dict[int, int]
    residual: dict[int, int]
    group_size: dict[int, int]
    active: dict[int, bool]
    group_code: dict[int, int]
    components: Counter[int]
    dynamic: Counter[int]
    baseline_components: Counter[int]
    baseline_dynamic: Counter[int]

    def find(self, component_id: int) -> int:
        parent = self.parent[component_id]
        if parent != component_id:
            self.parent[component_id] = self.find(parent)
        return self.parent[component_id]


def _incremental_metric_state(
    components: Sequence[_Component], policy: CandidatePolicy, pixel_area_m2: float,
) -> _IncrementalMetricState:
    parent = {item.component_id: item.component_id for item in components}
    residual = {item.component_id: len(item.pixels) for item in components}
    group_size = dict(residual)
    active = {item.component_id: True for item in components}
    group_code = {item.component_id: item.class_code for item in components}
    counts = Counter(item.class_code for item in components)
    dynamic = Counter(
        item.class_code
        for item in components
        if len(item.pixels) * pixel_area_m2 < policy.class_policies[item.class_code].dynamic_fragmentation_m2
    )
    return _IncrementalMetricState(
        parent, residual, group_size, active, group_code,
        counts.copy(), dynamic.copy(), counts.copy(), dynamic.copy(),
    )


def _is_dynamic_size(size: int, code: int, policy: CandidatePolicy, pixel_area_m2: float) -> bool:
    return size > 0 and size * pixel_area_m2 < policy.class_policies[code].dynamic_fragmentation_m2


def _prospective_metric_plan(
    state: _IncrementalMetricState,
    proposal: _Proposal,
    component_map: np.ndarray,
    labels: np.ndarray,
    result: np.ndarray,
    proposal_pixel_roots: Mapping[tuple[int, int], int],
    policy: CandidatePolicy,
    pixel_area_m2: float,
) -> tuple[dict[str, Any] | None, str | None]:
    """Evaluate only component groups touched by the current proposal."""

    rows, cols = proposal.footprint[:, 0], proposal.footprint[:, 1]
    source_counts = Counter(int(value) for value in component_map[rows, cols] if value)
    source_losses: Counter[int] = Counter()
    for component_id, count in source_counts.items():
        source_losses[state.find(component_id)] += count
    target_root_set = {state.find(int(value)) for value in proposal.baseline_target_component_ids}
    height, width = result.shape
    # A same-class footprint can join an earlier proposal even where no frozen
    # baseline target component is shared.  Resolve the neighbour to either a
    # residual baseline component or the accepted proposal's DSU node.
    for row, col in proposal.footprint:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = int(row + dr), int(col + dc)
            if not (0 <= rr < height and 0 <= cc < width):
                continue
            if result[rr, cc] != proposal.target_index:
                continue
            if labels[rr, cc] == proposal.target_index:
                target_root_set.add(state.find(int(component_map[rr, cc])))
            else:
                node = proposal_pixel_roots.get((rr, cc))
                if node is None:
                    raise CandidateError("missing accepted proposal node for target contact")
                target_root_set.add(state.find(node))
    target_roots = tuple(sorted(target_root_set))
    if not target_roots or any(not state.active[root] for root in target_roots):
        return None, "target_attachment"
    if any(state.group_code[root] != proposal.target_code for root in target_roots):
        raise CandidateError("proposal target component class does not match target code")
    # A proposal never relabels its own target class in the frozen baseline;
    # consequently source and target groups cannot overlap in a valid proposal.
    if set(source_losses) & set(target_roots):
        raise CandidateError("proposal source and target component groups overlap")
    predicted_components = state.components.copy()
    predicted_dynamic = state.dynamic.copy()
    source_sizes: dict[int, int] = {}
    for root, removed in source_losses.items():
        old_size = state.group_size[root]
        new_size = old_size - removed
        if new_size < 0:
            raise CandidateError("incremental source accounting underflow")
        code = state.group_code[root]
        source_sizes[root] = new_size
        predicted_dynamic[code] += int(_is_dynamic_size(new_size, code, policy, pixel_area_m2)) - int(_is_dynamic_size(old_size, code, policy, pixel_area_m2))
        if new_size == 0:
            predicted_components[code] -= 1
    target_code = proposal.target_code
    target_old_dynamic = sum(
        int(_is_dynamic_size(state.group_size[root], target_code, policy, pixel_area_m2))
        for root in target_roots
    )
    target_size = sum(state.group_size[root] for root in target_roots) + len(proposal.footprint)
    target_new_dynamic = int(_is_dynamic_size(target_size, target_code, policy, pixel_area_m2))
    predicted_dynamic[target_code] += target_new_dynamic - target_old_dynamic
    predicted_components[target_code] -= len(target_roots) - 1
    if any(predicted_components[code] > state.baseline_components[code] for code in predicted_components):
        return None, "component_count_increase"
    if sum(predicted_components.values()) > sum(state.baseline_components.values()):
        return None, "component_count_increase"
    if sum(predicted_dynamic.values()) > sum(state.baseline_dynamic.values()):
        return None, "dynamic_fragment_increase"
    return {
        "source_counts": source_counts,
        "source_losses": source_losses,
        "source_sizes": source_sizes,
        "target_roots": target_roots,
        "target_size": target_size,
        "predicted_components": predicted_components,
        "predicted_dynamic": predicted_dynamic,
    }, None


def _commit_metric_plan(state: _IncrementalMetricState, plan: Mapping[str, Any]) -> int:
    for component_id, removed in plan["source_counts"].items():
        state.residual[int(component_id)] -= int(removed)
    for root, size in plan["source_sizes"].items():
        state.group_size[int(root)] = int(size)
        if size == 0:
            state.active[int(root)] = False
    roots = tuple(int(value) for value in plan["target_roots"])
    representative = roots[0]
    for root in roots[1:]:
        state.parent[root] = representative
        state.group_size.pop(root, None)
        state.active.pop(root, None)
        state.group_code.pop(root, None)
    state.group_size[representative] = int(plan["target_size"])
    state.active[representative] = True
    state.components = Counter(plan["predicted_components"])
    state.dynamic = Counter(plan["predicted_dynamic"])
    return representative


def _adjudicate_v31b(
    proposals: Sequence[_Proposal], labels: np.ndarray, class_codes: Sequence[int], valid: np.ndarray,
    policy: CandidatePolicy, pixel_area_m2: float, component_map: np.ndarray,
    components: Sequence[_Component], class_budget_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[_Proposal], Counter[str], dict[str, tuple[str, str]], int, list[dict[str, Any]], int]:
    """Incremental B adjudication: reject the causing proposal, never roll back."""

    budget_mask = valid if class_budget_mask is None else class_budget_mask & valid
    baseline_totals = Counter(int(class_codes[index]) for index in labels[budget_mask])
    source_loss: Counter[int] = Counter()
    target_gain: Counter[int] = Counter()
    accepted: list[_Proposal] = []
    occupied = np.zeros(labels.shape, dtype=bool)
    result = labels.copy()
    skipped: Counter[str] = Counter()
    decisions: dict[str, tuple[str, str]] = {}
    components_by_id = {item.component_id: item for item in components}
    metric_state = _incremental_metric_state(components, policy, pixel_area_m2)
    proposal_pixel_roots: dict[tuple[int, int], int] = {}
    ordered, duplicates, duplicate_audit = _canonicalize_v31b_proposals(proposals)
    skipped.update(duplicates)
    removed_by_component: dict[int, set[tuple[int, int]]] = {}
    target_dependents: dict[int, list[_Proposal]] = {}
    accepted_by_id: dict[str, _Proposal] = {}
    proposal_pixel_owner: dict[tuple[int, int], str] = {}
    interaction_audit: list[dict[str, Any]] = []
    rank_groups: dict[tuple[float, float, float, float, str], list[_Proposal]] = {}
    for proposal in ordered:
        rank_groups.setdefault(_rank_key(proposal), []).append(proposal)
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
        if not np.all(budget_mask[rows, cols]):
            reason = "outside_core_owner"
        elif proposal.proposal_id in ambiguous:
            reason = "ambiguous_target_tie"
        elif np.any(occupied[rows, cols]):
            reason = "footprint_conflict"
        else:
            core = budget_mask[rows, cols]
            additions = Counter(int(class_codes[int(value)]) for value in labels[rows[core], cols[core]])
            target_budget = (
                policy.protected_bridge_gain_fraction
                if proposal.kind == "same_class_bridge" and proposal.target_code in policy.protected_source_codes
                else policy.maximum_target_gain_fraction
            )
            if any(
                source_loss[code] + count > baseline_totals[code] * policy.maximum_source_loss_fraction + 1e-12
                for code, count in additions.items()
            ):
                reason = "source_budget"
            elif target_gain[proposal.target_code] + int(np.count_nonzero(core)) > baseline_totals[proposal.target_code] * target_budget + 1e-12:
                reason = "target_budget"
            else:
                affected_components = tuple(sorted(set(int(value) for value in component_map[rows, cols] if value)))
                tentative_removed = dict(removed_by_component)
                for row, col, component_id in zip(rows, cols, component_map[rows, cols]):
                    if component_id:
                        tentative_removed[int(component_id)] = set(tentative_removed.get(int(component_id), set()))
                        tentative_removed[int(component_id)].add((int(row), int(col)))
                if not _source_connectivity_safe_incremental(
                    tentative_removed, affected_components, component_map, components_by_id,
                ):
                    reason = "source_connectivity"
                else:
                    metric_plan, metric_reason = _prospective_metric_plan(
                        metric_state, proposal, component_map, labels, result,
                        proposal_pixel_roots, policy, pixel_area_m2,
                    )
                    if metric_reason is not None:
                        reason = metric_reason
                    else:
                        affected = {
                            item.proposal_id: item
                            for component_id in affected_components
                            for item in target_dependents.get(component_id, [])
                        }
                        checked = [*sorted(affected.values(), key=lambda item: item.proposal_id), proposal]
                        interaction["affected_accepted_proposal_ids"] = [item.proposal_id for item in checked if item is not proposal]
                        interaction["target_attachment_checks"] = len(checked)
                        result[rows, cols] = proposal.target_index
                        transient_by_id = dict(accepted_by_id)
                        transient_by_id[proposal.proposal_id] = proposal
                        for row, col in proposal.footprint:
                            proposal_pixel_owner[(int(row), int(col))] = proposal.proposal_id
                        target_safe = all(
                            _target_attachment_safe_incremental(
                                item, result, labels, component_map, components_by_id,
                                transient_by_id, proposal_pixel_owner, target_dependents,
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
        source_loss.update(additions)
        target_gain[proposal.target_code] += int(np.count_nonzero(core))
        representative = _commit_metric_plan(metric_state, metric_plan)
        for row, col in proposal.footprint:
            proposal_pixel_roots[(int(row), int(col))] = representative
        for component_id in proposal.baseline_target_component_ids:
            target_dependents.setdefault(int(component_id), []).append(proposal)
        accepted_by_id[proposal.proposal_id] = proposal
        interaction["decision"] = "accepted"
        interaction["reason"] = "selected"
        interaction_audit.append(interaction)
    interaction_audit.extend(duplicate_audit)
    return result, accepted, skipped, decisions, 0, interaction_audit, int(duplicates["duplicate_proposal"])


def apply_v31a_candidate(
    labels: np.ndarray, *, class_codes: Sequence[int], pixel_area_m2: float,
    pixel_size_m: tuple[float, float] | None = None, valid_mask: np.ndarray | None = None,
    class_budget_mask: np.ndarray | None = None,
    probabilities: np.ndarray | None = None, confidence: np.ndarray | None = None,
    policy: CandidatePolicy | None = None, baseline_kind: str, full_audit: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply approved V3.1-A candidate mechanisms without production mutation.

    ``pixel_size_m`` is ``(row_step_m, column_step_m)``.  Supplying it from the
    current affine is required for non-square pixels; omission is only a safe
    square-pixel shorthand.  Invalid pixels are copied unchanged and excluded
    from every component, bridge, budget, and topology computation.  When
    ``class_budget_mask`` is supplied, it is the owned Core: budget denominators
    and charges use ``class_budget_mask & valid_mask``.  Proposals may inspect
    halo context, but an island or bridge is accepted only when its *entire*
    footprint is owned by that Core; every out-of-Core proposal is audited as
    ``outside_core_owner``.  ``None`` retains the historic all-valid behavior.
    """

    if baseline_kind not in ALLOWED_BASELINE_KINDS:
        raise CandidateError(
            "baseline_kind must be one of "
            f"{sorted(ALLOWED_BASELINE_KINDS)}"
        )
    selected_policy = policy or v31a_policy()
    baseline, valid, probs, conf, sizes = _validate(labels, class_codes, valid_mask, probabilities, confidence, pixel_area_m2, pixel_size_m, selected_policy)
    budget_mask = _class_budget_mask(class_budget_mask, valid)
    component_map, components = _component_index(baseline, valid, class_codes)
    generation_rejections: Counter[str] = Counter()
    proposals = _island_proposals(
        baseline, valid, probs, conf, class_codes, component_map, components,
        selected_policy, pixel_area_m2, generation_rejections,
    )
    for target_index in range(len(class_codes)):
        proposals.extend(_bridge_proposals_for_code(
            target_index, baseline, valid, probs, class_codes, component_map,
            components, selected_policy, pixel_area_m2, sizes,
            generation_rejections,
        ))
    working_result, accepted, skipped, decisions, rollback_count = _adjudicate(
        proposals, baseline, class_codes, valid, selected_policy, pixel_area_m2,
        component_map, components, budget_mask,
    )
    # Every accepted footprint is Core-owned, so this publication copy cannot
    # clip a topology proposal.  Keep the explicit Core assignment as a guard
    # against future adjudication changes.
    result = baseline.copy()
    result[budget_mask] = working_result[budget_mask]
    if not np.array_equal(result[~valid], baseline[~valid]) or np.any(result[valid] < 0) or np.any(result[valid] >= len(class_codes)):
        raise CandidateError("candidate violated single-label or invalid-pixel preservation")
    if not _final_topology_holds(
        baseline, result, valid, class_codes, selected_policy, pixel_area_m2,
        accepted, components,
    ):
        raise CandidateError("candidate violated final published topology")
    before_dynamic, before_components = _dynamic_count(baseline, valid, class_codes, selected_policy, pixel_area_m2)
    after_dynamic, after_components = _dynamic_count(result, valid, class_codes, selected_policy, pixel_area_m2)
    before_by_class = _per_class_metrics(
        baseline, valid, class_codes, selected_policy, pixel_area_m2
    )
    after_by_class = _per_class_metrics(
        result, valid, class_codes, selected_policy, pixel_area_m2
    )
    actual_source_loss: Counter[int] = Counter()
    actual_target_gain: Counter[int] = Counter()
    actual_bridge_gain: Counter[int] = Counter()
    for proposal in accepted:
        rows, cols = proposal.footprint[:, 0], proposal.footprint[:, 1]
        core = budget_mask[rows, cols]
        actual_source_loss.update(
            int(class_codes[int(value)])
            for value in baseline[rows[core], cols[core]]
        )
        actual_target_gain[proposal.target_code] += int(np.count_nonzero(core))
        if proposal.kind == "same_class_bridge":
            actual_bridge_gain[proposal.target_code] += int(np.count_nonzero(core))
    all_summaries = [
        _summary(item, decision=decisions.get(item.proposal_id, ("generated", "unresolved"))[0], reason=decisions.get(item.proposal_id, ("generated", "unresolved"))[1])
        for item in sorted(proposals, key=lambda item: (*_rank_key(item), item.proposal_id))
    ]
    accepted_summaries = [
        _summary(item, decision="accepted", reason="selected") for item in accepted
    ]
    limit = len(all_summaries) if full_audit else max(0, int(selected_policy.audit_proposal_limit))
    changed = valid & (result != baseline)
    protected_loss = sum(
        actual_source_loss[code] for code in selected_policy.protected_source_codes
    )
    report: dict[str, Any] = {
        "policy_snapshot": policy_snapshot(selected_policy),
        "policy_snapshot_sha256": policy_snapshot_sha256(selected_policy),
        "policy_id": selected_policy.policy_id,
        "policy_version": selected_policy.policy_version,
        "baseline_kind": baseline_kind,
        "confidence_semantics": "explicit confidence if supplied else P[baseline_label]",
        "confidence_source": "explicit" if conf is not None else "baseline_label_probability",
        "topology_connectivity": 4,
        "single_pass_from_frozen_baseline": True,
        "cascade_generation": False,
        "single_label": True,
        "gap_pixels": 0,
        "overlap_pixels": 0,
        "outside_pixels": 0,
        "baseline_mask_sha256": hashlib.sha256(np.ascontiguousarray(baseline).tobytes()).hexdigest(),
        "output_mask_sha256": hashlib.sha256(np.ascontiguousarray(result).tobytes()).hexdigest(),
        "valid_mask_sha256": hashlib.sha256(
            np.ascontiguousarray(valid).tobytes()
        ).hexdigest(),
        "class_budget_mask_sha256": hashlib.sha256(
            np.ascontiguousarray(budget_mask).tobytes()
        ).hexdigest(),
        "class_budget_mask_pixel_count": int(budget_mask.sum()),
        "class_budget_mask_semantics": (
            "complete_proposal_core_owner_and_valid_mask; none_means_valid_mask"
        ),
        "class_codes": [int(code) for code in class_codes],
        "physical_metrics": {
            "pixel_area_m2": float(pixel_area_m2),
            "row_step_m": float(sizes[0]),
            "column_step_m": float(sizes[1]),
            "source": "explicit_caller_supplied_physical_metrics",
        },
        "baseline": {"components_4_connected": before_components, "dynamic_fragments_4_connected": before_dynamic},
        "result": {"components_4_connected": after_components, "dynamic_fragments_4_connected": after_dynamic},
        "changed_pixel_count": int(np.count_nonzero(changed)),
        "changed_area_m2": float(np.count_nonzero(changed) * pixel_area_m2),
        "protected_source_loss_pixel_count": int(protected_loss),
        "protected_source_retention": 1.0 if protected_loss == 0 else 0.0,
        "source_split_violation_count": 0,
        "proposals_generated": len(proposals), "proposals_accepted": len(accepted),
        "proposal_generation_reject_reason_counts": dict(sorted(generation_rejections.items())),
        "proposal_reject_reason_counts": dict(sorted(skipped.items())),
        "proposals_rejected_by_adjudication": dict(sorted(skipped.items())),
        "accepted": accepted_summaries[:limit],
        "proposal_audit": all_summaries[:limit],
        "full_audit": bool(full_audit),
        "audit_truncated": not full_audit and (len(accepted_summaries) > limit or len(all_summaries) > limit),
        "final_topology_rollback": int(rollback_count),
        "valid_pixel_count": int(valid.sum()),
        "per_class": {
            str(code): {
                "baseline": before_by_class[int(code)],
                "result": after_by_class[int(code)],
                "source_loss": int(actual_source_loss[int(code)]),
                "target_gain": int(actual_target_gain[int(code)]),
                "bridge_gain": int(actual_bridge_gain[int(code)]),
                "net_pixel_drift": int(
                    actual_target_gain[int(code)] - actual_source_loss[int(code)]
                ),
            }
            for code in class_codes
        },
        "class_budget_pixels": {
            str(code): {
                "baseline": int(np.sum(baseline[budget_mask] == index)),
                "denominator": int(np.sum(baseline[budget_mask] == index)),
                "source_loss": int(actual_source_loss[code]),
                "target_gain": int(actual_target_gain[code]),
                "protected_bridge_gain": int(
                    actual_bridge_gain[code]
                    if code in selected_policy.protected_source_codes
                    else 0
                ),
                "source_loss_limit": float(
                    0.0
                    if code in selected_policy.protected_source_codes
                    else np.sum(baseline[budget_mask] == index)
                    * selected_policy.maximum_source_loss_fraction
                ),
                "target_gain_limit": float(np.sum(baseline[budget_mask] == index) * (selected_policy.protected_bridge_gain_fraction if code in selected_policy.protected_source_codes else selected_policy.maximum_target_gain_fraction)),
                "protected_bridge_gain_limit": float(
                    np.sum(baseline[budget_mask] == index)
                    * selected_policy.protected_bridge_gain_fraction
                    if code in selected_policy.protected_source_codes
                    else 0.0
                ),
            }
            for index, code in enumerate(class_codes)
        },
    }
    digest_body = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["audit_sha256"] = hashlib.sha256(digest_body.encode("utf-8")).hexdigest()
    return result, report


apply_v31a = apply_v31a_candidate


def apply_v31b_candidate(
    labels: np.ndarray, *, class_codes: Sequence[int], pixel_area_m2: float,
    pixel_size_m: tuple[float, float] | None = None, valid_mask: np.ndarray | None = None,
    class_budget_mask: np.ndarray | None = None,
    probabilities: np.ndarray | None = None, confidence: np.ndarray | None = None,
    policy: CandidatePolicy | None = None, baseline_kind: str, full_audit: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply V3.1-B with dependency-incremental topology adjudication.

    Proposal generation, ranking, probabilities, budgets, and Core ownership
    are byte-for-byte the A candidate's policy path.  B differs only in that a
    proposal which severs an earlier target attachment is rejected in place;
    accepted proposals are never popped in a final LIFO rollback.
    """

    if baseline_kind not in ALLOWED_BASELINE_KINDS:
        raise CandidateError(
            "baseline_kind must be one of "
            f"{sorted(ALLOWED_BASELINE_KINDS)}"
        )
    selected_policy = policy or v31b_policy()
    baseline, valid, probs, conf, sizes = _validate(
        labels, class_codes, valid_mask, probabilities, confidence,
        pixel_area_m2, pixel_size_m, selected_policy,
    )
    budget_mask = _class_budget_mask(class_budget_mask, valid)
    component_map, components = _component_index(baseline, valid, class_codes)
    generation_rejections: Counter[str] = Counter()
    proposals = _island_proposals(
        baseline, valid, probs, conf, class_codes, component_map, components,
        selected_policy, pixel_area_m2, generation_rejections,
    )
    for target_index in range(len(class_codes)):
        proposals.extend(_bridge_proposals_for_code(
            target_index, baseline, valid, probs, class_codes, component_map,
            components, selected_policy, pixel_area_m2, sizes,
            generation_rejections,
        ))
    working_result, accepted, skipped, decisions, rollback_count, interaction_audit, duplicate_count = _adjudicate_v31b(
        proposals, baseline, class_codes, valid, selected_policy, pixel_area_m2,
        component_map, components, budget_mask,
    )
    result = baseline.copy()
    result[budget_mask] = working_result[budget_mask]
    if (
        not np.array_equal(result[~valid], baseline[~valid])
        or np.any(result[valid] < 0)
        or np.any(result[valid] >= len(class_codes))
    ):
        raise CandidateError("candidate violated single-label or invalid-pixel preservation")
    # This is deliberately the only global output topology pass in B.  The
    # incremental checks above decide the current proposal and never mutate a
    # prior decision as a recovery mechanism.
    if not _final_topology_holds(
        baseline, result, valid, class_codes, selected_policy, pixel_area_m2,
        accepted, components,
    ):
        raise CandidateError("candidate violated final published topology")
    before_dynamic, before_components = _dynamic_count(
        baseline, valid, class_codes, selected_policy, pixel_area_m2
    )
    after_dynamic, after_components = _dynamic_count(
        result, valid, class_codes, selected_policy, pixel_area_m2
    )
    before_by_class = _per_class_metrics(
        baseline, valid, class_codes, selected_policy, pixel_area_m2
    )
    after_by_class = _per_class_metrics(
        result, valid, class_codes, selected_policy, pixel_area_m2
    )
    actual_source_loss: Counter[int] = Counter()
    actual_target_gain: Counter[int] = Counter()
    actual_bridge_gain: Counter[int] = Counter()
    for proposal in accepted:
        rows, cols = proposal.footprint[:, 0], proposal.footprint[:, 1]
        core = budget_mask[rows, cols]
        actual_source_loss.update(
            int(class_codes[int(value)]) for value in baseline[rows[core], cols[core]]
        )
        actual_target_gain[proposal.target_code] += int(np.count_nonzero(core))
        if proposal.kind == "same_class_bridge":
            actual_bridge_gain[proposal.target_code] += int(np.count_nonzero(core))
    # B canonicalizes before decisions, so a duplicate is represented in the
    # reason counts and metadata rather than as an ambiguous second audit row.
    canonical, _duplicates, duplicate_audit = _canonicalize_v31b_proposals(proposals)
    all_summaries = []
    for item in canonical:
        summary = _summary(
            item,
            decision=decisions.get(item.proposal_id, ("generated", "unresolved"))[0],
            reason=decisions.get(item.proposal_id, ("generated", "unresolved"))[1],
        )
        summary["occurrence_id"] = f"{item.proposal_id}:canonical"
        all_summaries.append(summary)
    accepted_summaries = [
        _summary(item, decision="accepted", reason="selected") for item in accepted
    ]
    limit = len(all_summaries) if full_audit else max(0, int(selected_policy.audit_proposal_limit))
    changed = valid & (result != baseline)
    protected_loss = sum(actual_source_loss[code] for code in selected_policy.protected_source_codes)
    report: dict[str, Any] = {
        "policy_snapshot": policy_snapshot(selected_policy),
        "policy_snapshot_sha256": policy_snapshot_sha256(selected_policy),
        "policy_id": selected_policy.policy_id,
        "policy_version": selected_policy.policy_version,
        "adjudication_mode": V31B_ADJUDICATION_MODE,
        "target_attachment_contract": (
            "direct_residual_target_contact_for_each_baseline_target_component; "
            "same_class_bridge_retains_all_baseline_target_pixels"
        ),
        "incremental_metric_contract": (
            "per_class_components_nonincreasing; global_components_and_dynamic_nonincreasing"
        ),
        "baseline_kind": baseline_kind,
        "confidence_semantics": "explicit confidence if supplied else P[baseline_label]",
        "confidence_source": "explicit" if conf is not None else "baseline_label_probability",
        "topology_connectivity": 4,
        "single_pass_from_frozen_baseline": True,
        "cascade_generation": False,
        "single_label": True,
        "gap_pixels": 0,
        "overlap_pixels": 0,
        "outside_pixels": 0,
        "baseline_mask_sha256": hashlib.sha256(np.ascontiguousarray(baseline).tobytes()).hexdigest(),
        "output_mask_sha256": hashlib.sha256(np.ascontiguousarray(result).tobytes()).hexdigest(),
        "valid_mask_sha256": hashlib.sha256(np.ascontiguousarray(valid).tobytes()).hexdigest(),
        "class_budget_mask_sha256": hashlib.sha256(np.ascontiguousarray(budget_mask).tobytes()).hexdigest(),
        "class_budget_mask_pixel_count": int(budget_mask.sum()),
        "class_budget_mask_semantics": "complete_proposal_core_owner_and_valid_mask; none_means_valid_mask",
        "class_codes": [int(code) for code in class_codes],
        "physical_metrics": {
            "pixel_area_m2": float(pixel_area_m2),
            "row_step_m": float(sizes[0]),
            "column_step_m": float(sizes[1]),
            "source": "explicit_caller_supplied_physical_metrics",
        },
        "baseline": {"components_4_connected": before_components, "dynamic_fragments_4_connected": before_dynamic},
        "result": {"components_4_connected": after_components, "dynamic_fragments_4_connected": after_dynamic},
        "changed_pixel_count": int(np.count_nonzero(changed)),
        "changed_area_m2": float(np.count_nonzero(changed) * pixel_area_m2),
        "protected_source_loss_pixel_count": int(protected_loss),
        "protected_source_retention": 1.0 if protected_loss == 0 else 0.0,
        "source_split_violation_count": 0,
        "proposals_generated": len(proposals),
        "raw_generated": len(proposals),
        "proposals_canonical": len(canonical),
        "duplicate_proposal_count": duplicate_count,
        "proposals_accepted": len(accepted),
        "proposal_generation_reject_reason_counts": dict(sorted(generation_rejections.items())),
        "proposal_reject_reason_counts": dict(sorted(skipped.items())),
        "proposals_rejected_by_adjudication": dict(sorted(skipped.items())),
        "accepted": accepted_summaries[:limit],
        "proposal_audit": all_summaries[:limit],
        "interaction_audit": interaction_audit[:limit],
        # Duplicate occurrence rows are an accounting ledger, not a sampled
        # proposal list.  In a full audit they must never disappear behind the
        # canonical-proposal limit.
        "duplicate_proposal_audit": duplicate_audit if full_audit else duplicate_audit[:limit],
        "raw_proposal_audit": (
            sorted([*all_summaries, *duplicate_audit], key=lambda item: str(item["occurrence_id"]))
            if full_audit
            else sorted([*all_summaries, *duplicate_audit], key=lambda item: str(item["occurrence_id"]))[:limit]
        ),
        "full_audit": bool(full_audit),
        "audit_truncated": not full_audit and (
            len(accepted_summaries) > limit
            or len(all_summaries) > limit
            or len(interaction_audit) > limit
            or len(all_summaries) + len(duplicate_audit) > limit
        ),
        "final_topology_rollback": int(rollback_count),
        "valid_pixel_count": int(valid.sum()),
        "per_class": {
            str(code): {
                "baseline": before_by_class[int(code)],
                "result": after_by_class[int(code)],
                "source_loss": int(actual_source_loss[int(code)]),
                "target_gain": int(actual_target_gain[int(code)]),
                "bridge_gain": int(actual_bridge_gain[int(code)]),
                "net_pixel_drift": int(actual_target_gain[int(code)] - actual_source_loss[int(code)]),
            }
            for code in class_codes
        },
        "class_budget_pixels": {
            str(code): {
                "baseline": int(np.sum(baseline[budget_mask] == index)),
                "denominator": int(np.sum(baseline[budget_mask] == index)),
                "source_loss": int(actual_source_loss[code]),
                "target_gain": int(actual_target_gain[code]),
                "protected_bridge_gain": int(actual_bridge_gain[code] if code in selected_policy.protected_source_codes else 0),
                "source_loss_limit": float(0.0 if code in selected_policy.protected_source_codes else np.sum(baseline[budget_mask] == index) * selected_policy.maximum_source_loss_fraction),
                "target_gain_limit": float(np.sum(baseline[budget_mask] == index) * (selected_policy.protected_bridge_gain_fraction if code in selected_policy.protected_source_codes else selected_policy.maximum_target_gain_fraction)),
                "protected_bridge_gain_limit": float(np.sum(baseline[budget_mask] == index) * selected_policy.protected_bridge_gain_fraction if code in selected_policy.protected_source_codes else 0.0),
            }
            for index, code in enumerate(class_codes)
        },
    }
    digest_body = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["audit_sha256"] = hashlib.sha256(digest_body.encode("utf-8")).hexdigest()
    return result, report


apply_v31b = apply_v31b_candidate
