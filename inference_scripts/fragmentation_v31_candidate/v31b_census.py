"""Read-only structural census primitives for the frozen V3.1-B mask.

The helpers in this module never mutate labels. A dense component map exists
for only one owner Core at a time. The retained shard contains component
metadata, different-class adjacency pairs and one-pixel Core borders, which is
enough to rebuild exact global four-connected identities.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage

from .candidate import CandidatePolicy, P10_METHOD


FOUR_CONNECTED = np.array(
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
)
AREA_RATIO_BINS = (
    ("ratio_0_0p1", 0.0, 0.1),
    ("ratio_0p1_0p25", 0.1, 0.25),
    ("ratio_0p25_0p5", 0.25, 0.5),
    ("ratio_0p5_0p75", 0.5, 0.75),
    ("ratio_0p75_1", 0.75, 1.0),
)


class CensusError(RuntimeError):
    """The census input or a closure invariant is invalid."""


@dataclass(frozen=True)
class CoreInput:
    """One non-overlapping owner Core in global row/column coordinates."""

    core_id: str
    window: tuple[int, int, int, int]  # row0, col0, height, width
    labels: np.ndarray
    valid: np.ndarray
    pixel_area_m2: float
    v3_labels: np.ndarray | None = None


class UnionFind:
    """Deterministic integer union-find for local-component nodes."""

    def __init__(self, size: int) -> None:
        if size < 0:
            raise CensusError("union-find size cannot be negative")
        self.parent = np.arange(size, dtype=np.int64)

    def find(self, value: int) -> int:
        value = int(value)
        root = value
        while int(self.parent[root]) != root:
            root = int(self.parent[root])
        while value != root:
            parent = int(self.parent[value])
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> int:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return left_root
        root, child = sorted((left_root, right_root))
        self.parent[child] = root
        return root


def empirical_p10(values: Sequence[float] | np.ndarray) -> float | None:
    """Return the frozen nearest-rank ceil P10 used by V3.1."""

    selected = np.asarray(values, dtype=np.float32)
    if selected.size == 0:
        return None
    rank = max(0, int(math.ceil(0.10 * selected.size)) - 1)
    return float(np.partition(selected, rank)[rank])


def area_ratio_bin(value: float) -> str:
    """Return one mutually-exclusive bin for a dynamic area/MMU ratio."""

    for name, low, high in AREA_RATIO_BINS:
        if low <= value < high:
            return name
    raise CensusError(f"dynamic area ratio lies outside [0,1): {value}")


def topology_class(boundary_exposed: bool, neighbor_class_count: int) -> str:
    """Assign the mutually-exclusive T0--T4 structural axis."""

    if boundary_exposed:
        return "T0_range_exposed"
    if neighbor_class_count == 0:
        return "T1_closed_zero_neighbor"
    if neighbor_class_count == 1:
        return "T2_closed_single_neighbor"
    if neighbor_class_count == 2:
        return "T3_closed_two_neighbors"
    return "T4_closed_multi_neighbor"


def _require_core(core: CoreInput, class_codes: Sequence[int]) -> None:
    row0, col0, height, width = core.window
    if not core.core_id or min(row0, col0) < 0 or min(height, width) <= 0:
        raise CensusError(f"{core.core_id!r}: invalid Core identity/window")
    if (
        core.labels.dtype != np.dtype("int16")
        or core.valid.dtype != np.dtype(bool)
        or core.labels.shape != (height, width)
        or core.valid.shape != core.labels.shape
    ):
        raise CensusError(f"{core.core_id}: labels/valid metadata mismatch")
    if core.v3_labels is not None and (
        core.v3_labels.dtype != np.dtype("int16")
        or core.v3_labels.shape != core.labels.shape
    ):
        raise CensusError(f"{core.core_id}: V3 labels metadata mismatch")
    if not math.isfinite(core.pixel_area_m2) or core.pixel_area_m2 <= 0:
        raise CensusError(f"{core.core_id}: invalid pixel area")
    present = set(int(value) for value in np.unique(core.labels[core.valid]))
    unknown = present - set(int(value) for value in class_codes)
    if unknown:
        raise CensusError(f"{core.core_id}: unknown valid class codes {sorted(unknown)}")


def label_core_components(
    core: CoreInput,
    class_codes: Sequence[int],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Label one Core and return its dense map plus compact component arrays.

    The implementation performs one compiled SciPy label pass per present
    class. It never scans the full Core once per component.
    """

    _require_core(core, class_codes)
    height, width = core.labels.shape
    component = np.zeros((height, width), dtype=np.int32)
    codes: list[np.ndarray] = []
    pixels: list[np.ndarray] = []
    min_linear: list[np.ndarray] = []
    linear = np.arange(height * width, dtype=np.int64).reshape(height, width)
    offset = 0
    for raw_code in class_codes:
        code = int(raw_code)
        mask = core.valid & (core.labels == code)
        numbered, count = ndimage.label(mask, structure=FOUR_CONNECTED)
        if not count:
            del mask, numbered
            continue
        local_ids = np.arange(1, int(count) + 1, dtype=np.int32)
        counts = np.bincount(numbered.ravel(), minlength=int(count) + 1)[1:]
        minimum = np.asarray(
            ndimage.minimum(linear, labels=numbered, index=local_ids),
            dtype=np.int64,
        )
        selected = numbered > 0
        component[selected] = numbered[selected].astype(np.int32) + offset
        codes.append(np.full(int(count), code, dtype=np.int16))
        pixels.append(counts.astype(np.int64, copy=False))
        min_linear.append(minimum)
        offset += int(count)
        del mask, numbered, selected
    del linear
    if offset == 0:
        empty_i64 = np.empty(0, dtype=np.int64)
        return component, {
            "class_code": np.empty(0, dtype=np.int16),
            "pixel_count": empty_i64,
            "area_m2": np.empty(0, dtype=np.float64),
            "min_row": empty_i64,
            "min_col": empty_i64,
            "boundary_internal": np.empty(0, dtype=bool),
            "b_changed_pixels": empty_i64,
        }
    class_code = np.concatenate(codes)
    pixel_count = np.concatenate(pixels)
    minimum = np.concatenate(min_linear)
    row0, col0, _height, _width = core.window
    changed = np.zeros(offset + 1, dtype=np.int64)
    if core.v3_labels is not None:
        changed = np.bincount(
            component[core.valid & (core.labels != core.v3_labels)],
            minlength=offset + 1,
        ).astype(np.int64, copy=False)
    arrays = {
        "class_code": class_code,
        "pixel_count": pixel_count,
        "area_m2": pixel_count.astype(np.float64) * float(core.pixel_area_m2),
        "min_row": (minimum // width + row0).astype(np.int64, copy=False),
        "min_col": (minimum % width + col0).astype(np.int64, copy=False),
        "boundary_internal": np.zeros(offset, dtype=bool),
        "b_changed_pixels": changed[1:],
    }
    return component, arrays


def label_core_component_map(
    labels: np.ndarray,
    valid: np.ndarray,
    class_codes: Sequence[int],
) -> tuple[np.ndarray, int]:
    """Rebuild only the deterministic local component map for probability pass."""

    if (
        labels.dtype != np.dtype("int16")
        or valid.dtype != np.dtype(bool)
        or labels.ndim != 2
        or valid.shape != labels.shape
    ):
        raise CensusError("labels/valid metadata mismatch for component-map rebuild")
    component = np.zeros(labels.shape, dtype=np.int32)
    offset = 0
    for raw_code in class_codes:
        mask = valid & (labels == int(raw_code))
        numbered, count = ndimage.label(mask, structure=FOUR_CONNECTED)
        if not count:
            del mask, numbered
            continue
        selected = numbered > 0
        component[selected] = numbered[selected].astype(np.int32) + offset
        offset += int(count)
        del mask, numbered, selected
    return component, offset


def _unique_component_pairs(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    pairs = np.column_stack((left.astype(np.int64), right.astype(np.int64)))
    pairs.sort(axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if not len(pairs):
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(pairs, axis=0)


def collect_core_shard(core: CoreInput, class_codes: Sequence[int]) -> dict[str, Any]:
    """Build the compact, serialization-ready shard for one owner Core."""

    component, arrays = label_core_components(core, class_codes)
    count = len(arrays["class_code"])
    different_pairs: list[np.ndarray] = []
    boundary_ids: list[np.ndarray] = []
    for axis in (0, 1):
        if axis == 0:
            first_id, second_id = component[:-1, :], component[1:, :]
            first_valid, second_valid = core.valid[:-1, :], core.valid[1:, :]
            first_code, second_code = core.labels[:-1, :], core.labels[1:, :]
        else:
            first_id, second_id = component[:, :-1], component[:, 1:]
            first_valid, second_valid = core.valid[:, :-1], core.valid[:, 1:]
            first_code, second_code = core.labels[:, :-1], core.labels[:, 1:]
        different = first_valid & second_valid & (first_code != second_code)
        if np.any(different):
            different_pairs.append(
                _unique_component_pairs(first_id[different], second_id[different])
            )
        first_exposed = first_valid & ~second_valid
        second_exposed = second_valid & ~first_valid
        if np.any(first_exposed):
            boundary_ids.append(np.unique(first_id[first_exposed]))
        if np.any(second_exposed):
            boundary_ids.append(np.unique(second_id[second_exposed]))
    if boundary_ids:
        selected = np.unique(np.concatenate(boundary_ids))
        selected = selected[selected > 0]
        arrays["boundary_internal"][selected - 1] = True
    pairs = (
        np.unique(np.concatenate(different_pairs), axis=0)
        if different_pairs
        else np.empty((0, 2), dtype=np.int64)
    )
    result: dict[str, Any] = {
        "core_id": core.core_id,
        "window": np.asarray(core.window, dtype=np.int64),
        "component_count": count,
        "different_class_pairs": pairs,
        **arrays,
        "edge_top": component[0, :].copy(),
        "edge_bottom": component[-1, :].copy(),
        "edge_left": component[:, 0].copy(),
        "edge_right": component[:, -1].copy(),
    }
    return result


def _window(shard: Mapping[str, Any]) -> tuple[int, int, int, int]:
    values = tuple(int(value) for value in np.asarray(shard["window"]).tolist())
    if len(values) != 4:
        raise CensusError("shard window must contain four integers")
    return values  # type: ignore[return-value]


def _edge_slice(
    shard: Mapping[str, Any], side: str, start: int, stop: int
) -> np.ndarray:
    row0, col0, _height, _width = _window(shard)
    values = np.asarray(shard[f"edge_{side}"], dtype=np.int32)
    if side in {"top", "bottom"}:
        return values[start - col0 : stop - col0]
    return values[start - row0 : stop - row0]


def coordinate_shards(
    shards: Sequence[Mapping[str, Any]],
    *,
    class_codes: Sequence[int],
    policy: CandidatePolicy,
    global_window: tuple[int, int, int, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge compact Core shards and return the exact dynamic ledger."""

    if not shards:
        raise CensusError("at least one Core shard is required")
    ordered = sorted(shards, key=lambda item: str(item["core_id"]))
    offsets: dict[str, int] = {}
    total = 0
    for shard in ordered:
        core_id = str(shard["core_id"])
        if core_id in offsets:
            raise CensusError(f"duplicate Core shard: {core_id}")
        offsets[core_id] = total
        total += int(shard["component_count"])
    uf = UnionFind(total)
    code_by_node = np.concatenate(
        [np.asarray(item["class_code"], dtype=np.int16) for item in ordered]
    )
    boundary_node = np.concatenate(
        [np.asarray(item["boundary_internal"], dtype=bool) for item in ordered]
    )
    cross_node = np.zeros(total, dtype=bool)
    adjacency_nodes: list[np.ndarray] = []
    for shard in ordered:
        pairs = np.asarray(shard["different_class_pairs"], dtype=np.int64)
        if pairs.size:
            adjacency_nodes.append(pairs + offsets[str(shard["core_id"])] - 1)

    for first_index, first in enumerate(ordered):
        fr0, fc0, fh, fw = _window(first)
        fr1, fc1 = fr0 + fh, fc0 + fw
        for second in ordered[first_index + 1 :]:
            sr0, sc0, sh, sw = _window(second)
            sr1, sc1 = sr0 + sh, sc0 + sw
            first_side: str | None = None
            second_side: str | None = None
            if fc1 == sc0 and max(fr0, sr0) < min(fr1, sr1):
                first_side, second_side = "right", "left"
                start, stop = max(fr0, sr0), min(fr1, sr1)
            elif sc1 == fc0 and max(fr0, sr0) < min(fr1, sr1):
                first_side, second_side = "left", "right"
                start, stop = max(fr0, sr0), min(fr1, sr1)
            elif fr1 == sr0 and max(fc0, sc0) < min(fc1, sc1):
                first_side, second_side = "bottom", "top"
                start, stop = max(fc0, sc0), min(fc1, sc1)
            elif sr1 == fr0 and max(fc0, sc0) < min(fc1, sc1):
                first_side, second_side = "top", "bottom"
                start, stop = max(fc0, sc0), min(fc1, sc1)
            else:
                continue
            left_local = _edge_slice(first, first_side, start, stop)
            right_local = _edge_slice(second, second_side, start, stop)
            if left_local.shape != right_local.shape:
                raise CensusError("adjacent Core seam lengths differ")
            left_valid, right_valid = left_local > 0, right_local > 0
            left_node = left_local.astype(np.int64) + offsets[str(first["core_id"])] - 1
            right_node = right_local.astype(np.int64) + offsets[str(second["core_id"])] - 1
            both = left_valid & right_valid
            if np.any(both):
                same = both & (code_by_node[left_node] == code_by_node[right_node])
                same_pairs = (
                    _unique_component_pairs(left_node[same] + 1, right_node[same] + 1)
                    - 1
                )
                for left, right in same_pairs:
                    uf.union(int(left), int(right))
                    cross_node[int(left)] = True
                    cross_node[int(right)] = True
                different = both & ~same
                if np.any(different):
                    adjacency_nodes.append(
                        _unique_component_pairs(
                            left_node[different] + 1, right_node[different] + 1
                        )
                        - 1
                    )
            only_left = left_valid & ~right_valid
            only_right = right_valid & ~left_valid
            boundary_node[left_node[only_left]] = True
            boundary_node[right_node[only_right]] = True

    global_row0, global_col0, global_height, global_width = global_window
    global_row1, global_col1 = global_row0 + global_height, global_col0 + global_width
    for shard in ordered:
        row0, col0, height, width = _window(shard)
        offset = offsets[str(shard["core_id"])]
        for exposed, side in (
            (row0 == global_row0, "top"),
            (row0 + height == global_row1, "bottom"),
            (col0 == global_col0, "left"),
            (col0 + width == global_col1, "right"),
        ):
            if exposed:
                values = np.asarray(shard[f"edge_{side}"], dtype=np.int64)
                values = values[values > 0] + offset - 1
                boundary_node[values] = True

    roots = np.fromiter((uf.find(index) for index in range(total)), dtype=np.int64, count=total)
    unique_roots, node_group = np.unique(roots, return_inverse=True)
    group_count = len(unique_roots)
    pixels_node = np.concatenate(
        [np.asarray(item["pixel_count"], dtype=np.int64) for item in ordered]
    )
    area_node = np.concatenate(
        [np.asarray(item["area_m2"], dtype=np.float64) for item in ordered]
    )
    min_row_node = np.concatenate(
        [np.asarray(item["min_row"], dtype=np.int64) for item in ordered]
    )
    min_col_node = np.concatenate(
        [np.asarray(item["min_col"], dtype=np.int64) for item in ordered]
    )
    changed_node = np.concatenate(
        [np.asarray(item["b_changed_pixels"], dtype=np.int64) for item in ordered]
    )
    group_pixels = np.bincount(node_group, weights=pixels_node).astype(np.int64)
    group_area = np.bincount(node_group, weights=area_node).astype(np.float64)
    group_changed = np.bincount(node_group, weights=changed_node).astype(np.int64)
    node_core_index = np.concatenate(
        [
            np.full(int(item["component_count"]), index, dtype=np.int64)
            for index, item in enumerate(ordered)
        ]
    )
    group_core_pairs = np.unique(
        node_group.astype(np.int64) * len(ordered) + node_core_index
    )
    group_core_count = np.bincount(
        group_core_pairs // len(ordered), minlength=group_count
    ).astype(np.int64)
    group_code = code_by_node[unique_roots]
    if np.any(code_by_node != group_code[node_group]):
        raise CensusError("union-find merged unlike classes")
    encoded_min = (min_row_node - global_row0) * global_width + (min_col_node - global_col0)
    group_min = np.full(group_count, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(group_min, node_group, encoded_min)
    group_boundary = np.zeros(group_count, dtype=bool)
    np.logical_or.at(group_boundary, node_group, boundary_node)
    group_cross = np.zeros(group_count, dtype=bool)
    np.logical_or.at(group_cross, node_group, cross_node)
    if np.any(group_cross != (group_core_count > 1)):
        raise CensusError("cross-Core component closure failed")

    if adjacency_nodes:
        node_pairs = np.concatenate(adjacency_nodes)
        group_pairs = np.column_stack(
            (node_group[node_pairs[:, 0]], node_group[node_pairs[:, 1]])
        )
        group_pairs.sort(axis=1)
        group_pairs = group_pairs[group_pairs[:, 0] != group_pairs[:, 1]]
        group_pairs = np.unique(group_pairs, axis=0)
    else:
        group_pairs = np.empty((0, 2), dtype=np.int64)
    neighbor_count = np.zeros(group_count, dtype=np.int64)
    neighbor_mask = np.zeros(group_count, dtype=np.uint16)
    if len(group_pairs):
        np.add.at(neighbor_count, group_pairs[:, 0], 1)
        np.add.at(neighbor_count, group_pairs[:, 1], 1)
        code_index = {int(code): index for index, code in enumerate(class_codes)}
        left_bits = np.left_shift(
            np.uint16(1),
            np.asarray(
                [code_index[int(group_code[value])] for value in group_pairs[:, 1]],
                dtype=np.uint16,
            ),
        )
        right_bits = np.left_shift(
            np.uint16(1),
            np.asarray(
                [code_index[int(group_code[value])] for value in group_pairs[:, 0]],
                dtype=np.uint16,
            ),
        )
        np.bitwise_or.at(neighbor_mask, group_pairs[:, 0], left_bits)
        np.bitwise_or.at(neighbor_mask, group_pairs[:, 1], right_bits)

    mmu_by_code = {
        int(code): float(policy.class_policies[int(code)].dynamic_fragmentation_m2)
        for code in class_codes
    }
    dynamic = np.asarray(
        [0.0 < area < mmu_by_code[int(code)] for area, code in zip(group_area, group_code)],
        dtype=bool,
    )
    local_group_by_core: dict[str, np.ndarray] = {}
    for shard in ordered:
        core_id = str(shard["core_id"])
        start = offsets[core_id]
        stop = start + int(shard["component_count"])
        local_group_by_core[core_id] = node_group[start:stop].copy()

    ledger: list[dict[str, Any]] = []
    ledger_by_group: dict[int, dict[str, Any]] = {}
    for group in np.flatnonzero(dynamic).tolist():
        code = int(group_code[group])
        mmu = mmu_by_code[code]
        row = int(group_min[group] // global_width + global_row0)
        col = int(group_min[group] % global_width + global_col0)
        neighbor_codes = [
            int(class_codes[index])
            for index in range(len(class_codes))
            if int(neighbor_mask[group]) & (1 << index)
        ]
        area = float(group_area[group])
        topology = topology_class(bool(group_boundary[group]), len(neighbor_codes))
        unique_neighbor = neighbor_codes[0] if len(neighbor_codes) == 1 else None
        source_policy = policy.class_policies[code]
        exact_enclosure = bool(
            topology == "T2_closed_single_neighbor" and neighbor_count[group] == 1
        )
        compatible = bool(
            unique_neighbor is not None
            and unique_neighbor in policy.semantic_compatible_targets.get(code, frozenset())
        )
        target_protected = bool(
            unique_neighbor is not None
            and policy.class_policies[unique_neighbor].ordinary_protected
        )
        record: dict[str, Any] = {
            "fragment_id": f"{code}:{row}:{col}",
            "global_component_group": int(group),
            "class_code": code,
            "pixel_count": int(group_pixels[group]),
            "area_m2": area,
            "dynamic_mmu_m2": mmu,
            "area_to_mmu_ratio": area / mmu,
            "area_ratio_bin": area_ratio_bin(area / mmu),
            "topology_class": topology,
            "boundary_exposed": bool(group_boundary[group]),
            "neighbor_class_set": neighbor_codes,
            "adjacent_global_component_count": int(neighbor_count[group]),
            "unique_neighbor_code": unique_neighbor,
            "cross_core": bool(group_cross[group]),
            "owner_core_count": int(group_core_count[group]),
            "protected_source": code in policy.protected_source_codes,
            "b_changed_pixel_count": int(group_changed[group]),
            "b_affected_component": bool(group_changed[group] > 0),
            "policy_island_evidence": {
                "exact_single_adjacent_component": exact_enclosure,
                "source_unprotected": code not in policy.protected_source_codes,
                "within_source_area_cap": bool(
                    source_policy.enclosed_island_max_m2 > 0
                    and area <= source_policy.enclosed_island_max_m2
                ),
                "semantic_compatible_target": compatible,
                "target_not_ordinary_protected": bool(
                    unique_neighbor is not None and not target_protected
                ),
                "probability_available": False,
                "mean_confidence_pass": None,
                "probability_gate_pass": None,
                "full_policy_gate_pass": None,
                "p10_method": P10_METHOD,
            },
        }
        ledger.append(record)
        ledger_by_group[int(group)] = record
    ledger.sort(key=lambda item: item["fragment_id"])
    coordination = {
        "global_component_count": int(group_count),
        "dynamic_fragment_count": int(dynamic.sum()),
        "dynamic_fragment_area_m2": float(group_area[dynamic].sum()),
        "local_group_by_core": local_group_by_core,
        "ledger_by_group": ledger_by_group,
    }
    return ledger, coordination


def add_probability_evidence(
    ledger_by_group: Mapping[int, dict[str, Any]],
    group_ids: np.ndarray,
    current_values: np.ndarray,
    target_values: np.ndarray,
    confidence_values: np.ndarray,
    class_values: np.ndarray,
    *,
    class_codes: Sequence[int],
    policy: CandidatePolicy,
) -> None:
    """Attach exact frozen probability evidence to eligible T2 fragments."""

    if not (
        group_ids.ndim
        == current_values.ndim
        == target_values.ndim
        == confidence_values.ndim
        == 1
        and len(group_ids)
        == len(current_values)
        == len(target_values)
        == len(confidence_values)
        and class_values.ndim == 2
        and class_values.shape[0] == len(class_codes)
        and class_values.shape[1] == len(group_ids)
    ):
        raise CensusError("probability evidence arrays must be aligned one-dimensional arrays")
    if len(group_ids) == 0:
        return
    order = np.argsort(group_ids, kind="stable")
    groups = group_ids[order]
    current = current_values[order]
    target = target_values[order]
    confidence = confidence_values[order]
    all_classes = class_values[:, order]
    unique, starts, counts = np.unique(groups, return_index=True, return_counts=True)
    for raw_group, start, count in zip(unique.tolist(), starts.tolist(), counts.tolist()):
        group = int(raw_group)
        record = ledger_by_group.get(group)
        if record is None:
            raise CensusError(f"probability evidence references unknown dynamic group {group}")
        chosen_current = current[start : start + count].astype(np.float64, copy=False)
        chosen_target = target[start : start + count].astype(np.float64, copy=False)
        chosen_confidence = confidence[start : start + count].astype(np.float64, copy=False)
        chosen_classes = all_classes[:, start : start + count].astype(
            np.float64, copy=False
        )
        if count != int(record["pixel_count"]):
            raise CensusError(
                f"{record['fragment_id']}: probability pixel count {count} differs from component {record['pixel_count']}"
            )
        target_code = record.get("unique_neighbor_code")
        if target_code is None:
            raise CensusError("probability evidence was collected for a non-unique-neighbor fragment")
        target_policy = policy.class_policies[int(target_code)]
        target_mean = float(np.mean(chosen_target))
        current_minus_target = float(np.mean(chosen_current - chosen_target))
        target_p10 = empirical_p10(chosen_target)
        mean_confidence = float(np.mean(chosen_confidence))
        probability_pass = bool(
            target_mean >= target_policy.minimum_target_probability_mean
            and current_minus_target
            <= target_policy.maximum_current_minus_target_probability_mean
            and target_p10 is not None
            and target_p10 >= target_policy.minimum_target_probability_p10
        )
        confidence_pass = bool(mean_confidence <= policy.island_maximum_mean_confidence)
        evidence = record["policy_island_evidence"]
        evidence.update(
            {
                "probability_available": True,
                "mean_current_probability": float(np.mean(chosen_current)),
                "mean_target_probability": target_mean,
                "mean_current_minus_target": current_minus_target,
                "p10_target_probability": target_p10,
                "mean_max_probability_confidence": mean_confidence,
                "mean_probability_by_class_code": {
                    str(code): float(value)
                    for code, value in zip(
                        class_codes,
                        np.mean(chosen_classes, axis=1, dtype=np.float64).tolist(),
                    )
                },
                "target_minimum_probability_mean": float(
                    target_policy.minimum_target_probability_mean
                ),
                "target_maximum_current_minus_target_probability_mean": float(
                    target_policy.maximum_current_minus_target_probability_mean
                ),
                "target_minimum_probability_p10": float(
                    target_policy.minimum_target_probability_p10
                ),
                "mean_confidence_pass": confidence_pass,
                "probability_gate_pass": probability_pass,
            }
        )
        evidence["full_policy_gate_pass"] = bool(
            evidence["exact_single_adjacent_component"]
            and evidence["source_unprotected"]
            and evidence["within_source_area_cap"]
            and evidence["semantic_compatible_target"]
            and evidence["target_not_ordinary_protected"]
            and confidence_pass
            and probability_pass
        )


def probability_selection_by_local_component(
    shard: Mapping[str, Any],
    local_group: np.ndarray,
    ledger_by_group: Mapping[int, Mapping[str, Any]],
    class_codes: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local-component group/current/target indices for T2 evidence."""

    count = int(shard["component_count"])
    if local_group.shape != (count,):
        raise CensusError("local-to-global group map shape mismatch")
    group_for_local = np.full(count + 1, -1, dtype=np.int64)
    current_index = np.full(count + 1, -1, dtype=np.int16)
    target_index = np.full(count + 1, -1, dtype=np.int16)
    code_index = {int(code): index for index, code in enumerate(class_codes)}
    class_code = np.asarray(shard["class_code"], dtype=np.int16)
    for local_zero, group_raw in enumerate(local_group.tolist()):
        group = int(group_raw)
        record = ledger_by_group.get(group)
        if record is None or record["topology_class"] != "T2_closed_single_neighbor":
            continue
        target_code = int(record["unique_neighbor_code"])
        group_for_local[local_zero + 1] = group
        current_index[local_zero + 1] = code_index[int(class_code[local_zero])]
        target_index[local_zero + 1] = code_index[target_code]
    return group_for_local, current_index, target_index


def compact_tile_shard(core: CoreInput, class_codes: Sequence[int]) -> dict[str, Any]:
    """Concise compatibility name used by census tests."""

    return collect_core_shard(core, class_codes)
