"""Shared four-connected topology primitives for fragmentation candidates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage  # type: ignore[import-untyped]
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from loess_runtime.geometry.fragmentation_v33_candidate.contracts import (
    CandidateError,
    CandidatePolicy,
    Component,
    Proposal,
)

__all__ = [
    "FOUR_CONNECTED",
    "cell_polygon_edge_distance_m",
    "component_index",
    "dynamic_count",
    "final_topology_holds",
    "local_topology_delta",
    "per_class_metrics",
    "source_connectivity_safe_incremental",
    "target_attachment_safe_incremental",
]

BoolArray: TypeAlias = NDArray[np.bool_]
Int16Array: TypeAlias = NDArray[np.int16]
Int32Array: TypeAlias = NDArray[np.int32]

FOUR_CONNECTED: BoolArray = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def component_index(
    labels: Int16Array,
    valid: BoolArray,
    class_codes: Sequence[int],
) -> tuple[Int32Array, list[Component]]:
    component_map = np.zeros(labels.shape, dtype=np.int32)
    components: list[Component] = []
    component_id = 1
    height, width = labels.shape
    for index, code in enumerate(class_codes):
        local, count = ndimage.label(
            valid & (labels == index), structure=FOUR_CONNECTED
        )
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
            external = bool(
                np.any(rows == 0)
                or np.any(cols == 0)
                or np.any(rows == height - 1)
                or np.any(cols == width - 1)
            )
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = rows + dr, cols + dc
                inside = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
                if np.any(~inside) or np.any(~valid[rr[inside], cc[inside]]):
                    external = True
                    break
            components.append(
                Component(
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


def dynamic_count(
    labels: Int16Array,
    valid: BoolArray,
    class_codes: Sequence[int],
    policy: CandidatePolicy,
    pixel_area_m2: float,
) -> tuple[int, int]:
    _map, components = component_index(labels, valid, class_codes)
    dynamic = sum(
        len(item.pixels) * pixel_area_m2
        < policy.class_policies[item.class_code].dynamic_fragmentation_m2
        for item in components
    )
    return int(dynamic), len(components)


def per_class_metrics(
    labels: Int16Array,
    valid: BoolArray,
    class_codes: Sequence[int],
    policy: CandidatePolicy,
    pixel_area_m2: float,
) -> dict[int, dict[str, float | int]]:
    _component_map, components = component_index(labels, valid, class_codes)
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
        metrics["component_count_4_connected"] = (
            int(metrics["component_count_4_connected"]) + 1
        )
        if (
            area_m2
            < policy.class_policies[component.class_code].dynamic_fragmentation_m2
        ):
            metrics["dynamic_fragment_count_4_connected"] = (
                int(metrics["dynamic_fragment_count_4_connected"]) + 1
            )
            metrics["dynamic_fragment_area_m2"] = (
                float(metrics["dynamic_fragment_area_m2"]) + area_m2
            )
    return result


def local_topology_delta(
    footprint: Int32Array,
    component_map: Int32Array,
    components_by_id: Mapping[int, Component],
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
        before_dynamic = (
            len(source.pixels) * pixel_area_m2
            < policy.class_policies[source.class_code].dynamic_fragmentation_m2
        )
        remaining = len(source.pixels) - int(removed_count)
        after_dynamic = (
            remaining > 0
            and remaining * pixel_area_m2
            < policy.class_policies[source.class_code].dynamic_fragmentation_m2
        )
        dynamic_reduction += int(before_dynamic) - int(after_dynamic)
        if remaining == 0:
            component_reduction += 1
    targets = [components_by_id[int(value)] for value in target_component_ids]
    if targets:
        target_policy = policy.class_policies[targets[0].class_code]
        target_before_dynamic = sum(
            len(component.pixels) * pixel_area_m2
            < target_policy.dynamic_fragmentation_m2
            for component in targets
        )
        target_after_dynamic = (
            sum(len(component.pixels) for component in targets) + len(footprint)
        ) * pixel_area_m2 < target_policy.dynamic_fragmentation_m2
        dynamic_reduction += int(target_before_dynamic) - int(target_after_dynamic)
        component_reduction += len(targets) - 1
    return int(dynamic_reduction), int(component_reduction)


def cell_polygon_edge_distance_m(
    first: Component,
    second: Component,
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


def final_topology_holds(
    baseline: Int16Array,
    result: Int16Array,
    valid: BoolArray,
    class_codes: Sequence[int],
    policy: CandidatePolicy,
    pixel_area_m2: float,
    accepted: Sequence[Proposal],
    components: Sequence[Component],
) -> bool:
    before = per_class_metrics(baseline, valid, class_codes, policy, pixel_area_m2)
    after = per_class_metrics(result, valid, class_codes, policy, pixel_area_m2)
    if any(
        after[int(code)]["component_count_4_connected"]
        > before[int(code)]["component_count_4_connected"]
        for code in class_codes
    ):
        return False
    before_dynamic, before_components = dynamic_count(
        baseline, valid, class_codes, policy, pixel_area_m2
    )
    after_dynamic, after_components = dynamic_count(
        result, valid, class_codes, policy, pixel_area_m2
    )
    if after_components > before_components or after_dynamic > before_dynamic:
        return False
    output_components, _items = component_index(result, valid, class_codes)
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


def source_connectivity_safe_incremental(
    removed_by_component: Mapping[int, set[tuple[int, int]]],
    changed_component_ids: Sequence[int],
    component_map: Int32Array,
    components_by_id: Mapping[int, Component],
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


def target_attachment_safe_incremental(
    proposal: Proposal,
    result: Int16Array,
    labels: Int16Array,
    component_map: Int32Array,
    components_by_id: Mapping[int, Component],
    proposals_by_id: Mapping[str, Proposal],
    proposal_pixel_owner: Mapping[tuple[int, int], str],
    target_dependents: Mapping[int, Sequence[Proposal]],
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

    def directly_attached(item: Proposal, component_id: int) -> bool:
        key = (item.proposal_id, int(component_id))
        if key in direct_cache:
            return direct_cache[key]
        for row, col in item.footprint:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = int(row + dr), int(col + dc)
                if (
                    0 <= rr < height
                    and 0 <= cc < width
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
