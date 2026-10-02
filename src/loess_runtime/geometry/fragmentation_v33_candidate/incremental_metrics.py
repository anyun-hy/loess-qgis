"""Incremental component metrics for fragmentation adjudication."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray

from loess_runtime.geometry.fragmentation_v33_candidate.contracts import (
    CandidateError,
    CandidatePolicy,
    Component,
    Proposal,
)

__all__ = [
    "IncrementalMetricState",
    "MetricPlan",
    "commit_metric_plan",
    "initialize_metric_state",
    "is_dynamic_size",
    "prospective_metric_plan",
]

Int16Array: TypeAlias = NDArray[np.int16]
Int32Array: TypeAlias = NDArray[np.int32]


class MetricPlan(TypedDict):
    """Existing prospective metric values committed after proposal acceptance."""

    source_counts: Counter[int]
    source_losses: Counter[int]
    source_sizes: dict[int, int]
    target_roots: tuple[int, ...]
    target_size: int
    predicted_components: Counter[int]
    predicted_dynamic: Counter[int]


@dataclass
class IncrementalMetricState:
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


def initialize_metric_state(
    components: Sequence[Component],
    policy: CandidatePolicy,
    pixel_area_m2: float,
) -> IncrementalMetricState:
    parent = {item.component_id: item.component_id for item in components}
    residual = {item.component_id: len(item.pixels) for item in components}
    group_size = dict(residual)
    active = {item.component_id: True for item in components}
    group_code = {item.component_id: item.class_code for item in components}
    counts = Counter(item.class_code for item in components)
    dynamic = Counter(
        item.class_code
        for item in components
        if len(item.pixels) * pixel_area_m2
        < policy.class_policies[item.class_code].dynamic_fragmentation_m2
    )
    return IncrementalMetricState(
        parent,
        residual,
        group_size,
        active,
        group_code,
        counts.copy(),
        dynamic.copy(),
        counts.copy(),
        dynamic.copy(),
    )


def is_dynamic_size(
    size: int, code: int, policy: CandidatePolicy, pixel_area_m2: float
) -> bool:
    return (
        size > 0
        and size * pixel_area_m2 < policy.class_policies[code].dynamic_fragmentation_m2
    )


def prospective_metric_plan(
    state: IncrementalMetricState,
    proposal: Proposal,
    component_map: Int32Array,
    labels: Int16Array,
    result: Int16Array,
    proposal_pixel_roots: Mapping[tuple[int, int], int],
    policy: CandidatePolicy,
    pixel_area_m2: float,
) -> tuple[MetricPlan | None, str | None]:
    """Evaluate only component groups touched by the current proposal."""

    rows, cols = proposal.footprint[:, 0], proposal.footprint[:, 1]
    source_counts = Counter(int(value) for value in component_map[rows, cols] if value)
    source_losses: Counter[int] = Counter()
    for component_id, count in source_counts.items():
        source_losses[state.find(component_id)] += count
    target_root_set = {
        state.find(int(value)) for value in proposal.baseline_target_component_ids
    }
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
                    raise CandidateError(
                        "missing accepted proposal node for target contact"
                    )
                target_root_set.add(state.find(node))
    target_roots = tuple(sorted(target_root_set))
    if not target_roots or any(not state.active[root] for root in target_roots):
        return None, "target_attachment"
    if any(state.group_code[root] != proposal.target_code for root in target_roots):
        raise CandidateError(
            "proposal target component class does not match target code"
        )
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
        predicted_dynamic[code] += int(
            is_dynamic_size(new_size, code, policy, pixel_area_m2)
        ) - int(is_dynamic_size(old_size, code, policy, pixel_area_m2))
        if new_size == 0:
            predicted_components[code] -= 1
    target_code = proposal.target_code
    target_old_dynamic = sum(
        int(is_dynamic_size(state.group_size[root], target_code, policy, pixel_area_m2))
        for root in target_roots
    )
    target_size = sum(state.group_size[root] for root in target_roots) + len(
        proposal.footprint
    )
    target_new_dynamic = int(
        is_dynamic_size(target_size, target_code, policy, pixel_area_m2)
    )
    predicted_dynamic[target_code] += target_new_dynamic - target_old_dynamic
    predicted_components[target_code] -= len(target_roots) - 1
    if any(
        predicted_components[code] > state.baseline_components[code]
        for code in predicted_components
    ):
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


def commit_metric_plan(state: IncrementalMetricState, plan: MetricPlan) -> int:
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
