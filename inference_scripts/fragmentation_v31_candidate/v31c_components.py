"""Streaming 4-connected component identities for isolated V3.1-C work.

This module deliberately indexes each Core separately.  It retains only Core
border component ids, the requested query-component ids, and a union-find
table; it never creates a label or component array for the whole raster.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from numbers import Real
from typing import Iterable, Sequence

import numpy as np
from scipy import ndimage


FOUR_CONNECTED = np.array(
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
)


class ComponentIndexError(ValueError):
    """Raised when the Core partition or a component query is invalid."""


@dataclass(frozen=True)
class CoreTile:
    """One non-overlapping owner Core in global pixel coordinates.

    ``window`` is ``(row_offset, column_offset, height, width)``.  ``labels``
    must be an ``int16`` 2-D array and ``valid`` a boolean array of the same
    shape.  Invalid cells do not belong to any component.
    """

    core_id: str
    window: tuple[int, int, int, int]
    labels: np.ndarray
    valid: np.ndarray
    pixel_area_m2: float = 1.0


@dataclass(frozen=True, order=True)
class GlobalComponentKey:
    """Stable identity: class code plus the component's least global pixel."""

    class_code: int
    min_row: int
    min_col: int


@dataclass(frozen=True)
class QueryComponent:
    """The global component containing one requested global pixel."""

    point: tuple[int, int]
    key: GlobalComponentKey
    pixel_count: int
    area_m2: float


@dataclass(frozen=True)
class ComponentIndexResult:
    """Query answers in the supplied order and the total global component count."""

    query_components: tuple[QueryComponent, ...]
    global_component_count: int


@dataclass(frozen=True)
class _CoreBorders:
    core_id: str
    row0: int
    col0: int
    row1: int
    col1: int
    top: np.ndarray
    bottom: np.ndarray
    left: np.ndarray
    right: np.ndarray


@dataclass(frozen=True)
class _NodeStats:
    class_code: int
    pixel_count: int
    area_m2: float
    min_point: tuple[int, int]


_Node = tuple[str, int]


class _ComponentUnionFind:
    """Union-find whose deterministic root is independent of Core input order."""

    def __init__(self) -> None:
        self.parent: dict[_Node, _Node] = {}
        self.stats: dict[_Node, _NodeStats] = {}

    def add(self, node: _Node, stats: _NodeStats) -> None:
        self.parent[node] = node
        self.stats[node] = stats

    def find(self, node: _Node) -> _Node:
        parent = self.parent[node]
        if parent != node:
            parent = self.find(parent)
            self.parent[node] = parent
        return parent

    def union(self, first: _Node, second: _Node) -> _Node:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return first_root
        first_stats = self.stats[first_root]
        second_stats = self.stats[second_root]
        if first_stats.class_code != second_stats.class_code:
            raise ComponentIndexError(
                "attempted to merge components with different class codes: "
                f"{first_stats.class_code} and {second_stats.class_code}"
            )
        root, child = sorted((first_root, second_root))
        root_stats = self.stats[root]
        child_stats = self.stats[child]
        self.parent[child] = root
        self.stats[root] = _NodeStats(
            class_code=root_stats.class_code,
            pixel_count=root_stats.pixel_count + child_stats.pixel_count,
            area_m2=root_stats.area_m2 + child_stats.area_m2,
            min_point=min(root_stats.min_point, child_stats.min_point),
        )
        return root


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ComponentIndexError(f"{name} must be an integer")
    return int(value)


def _normalise_window(tile: CoreTile) -> tuple[int, int, int, int]:
    if len(tile.window) != 4:
        raise ComponentIndexError(f"Core {tile.core_id!r} window must have four values")
    row0, col0, height, width = (
        _require_int(value, f"Core {tile.core_id!r} window value") for value in tile.window
    )
    if row0 < 0 or col0 < 0 or height <= 0 or width <= 0:
        raise ComponentIndexError(f"Core {tile.core_id!r} has an invalid global window")
    if not isinstance(tile.labels, np.ndarray) or tile.labels.dtype != np.dtype(np.int16):
        raise ComponentIndexError(f"Core {tile.core_id!r} labels must be an int16 ndarray")
    if not isinstance(tile.valid, np.ndarray) or tile.valid.dtype != np.dtype(bool):
        raise ComponentIndexError(f"Core {tile.core_id!r} valid must be a bool ndarray")
    if tile.labels.ndim != 2 or tile.valid.ndim != 2 or tile.labels.shape != tile.valid.shape:
        raise ComponentIndexError(f"Core {tile.core_id!r} labels and valid must be same-shaped 2-D arrays")
    if tile.labels.shape != (height, width):
        raise ComponentIndexError(f"Core {tile.core_id!r} window size does not match its arrays")
    if (
        isinstance(tile.pixel_area_m2, bool)
        or not isinstance(tile.pixel_area_m2, Real)
        or not math.isfinite(float(tile.pixel_area_m2))
        or float(tile.pixel_area_m2) <= 0.0
    ):
        raise ComponentIndexError(f"Core {tile.core_id!r} pixel_area_m2 must be a positive finite number")
    return row0, col0, height, width


def _validate_partition(
    tiles: Sequence[CoreTile], global_shape: tuple[int, int] | None,
) -> tuple[list[tuple[CoreTile, tuple[int, int, int, int]]], tuple[int, int, int, int]]:
    if not tiles:
        raise ComponentIndexError("at least one Core tile is required")
    seen: set[str] = set()
    checked: list[tuple[CoreTile, tuple[int, int, int, int]]] = []
    for tile in tiles:
        if not isinstance(tile.core_id, str) or not tile.core_id:
            raise ComponentIndexError("every Core tile needs a non-empty string core_id")
        if tile.core_id in seen:
            raise ComponentIndexError(f"duplicate core_id: {tile.core_id!r}")
        seen.add(tile.core_id)
        checked.append((tile, _normalise_window(tile)))

    if global_shape is None:
        row_low = min(window[0] for _, window in checked)
        col_low = min(window[1] for _, window in checked)
        row_high = max(window[0] + window[2] for _, window in checked)
        col_high = max(window[1] + window[3] for _, window in checked)
    else:
        if len(global_shape) != 2:
            raise ComponentIndexError("global_shape must be (height, width)")
        height, width = (_require_int(value, "global_shape value") for value in global_shape)
        if height <= 0 or width <= 0:
            raise ComponentIndexError("global_shape dimensions must be positive")
        row_low, col_low, row_high, col_high = 0, 0, height, width
        for tile, (row0, col0, tile_height, tile_width) in checked:
            if row0 + tile_height > height or col0 + tile_width > width:
                raise ComponentIndexError(f"Core {tile.core_id!r} lies outside global_shape")

    events: dict[int, list[tuple[int, str, int, int]]] = defaultdict(list)
    windows = {tile.core_id: window for tile, window in checked}
    for tile, (row0, col0, height, width) in checked:
        events[row0].append((1, tile.core_id, col0, col0 + width))
        events[row0 + height].append((-1, tile.core_id, col0, col0 + width))
    active: dict[str, tuple[int, int]] = {}
    event_rows = sorted(events)
    for index, row in enumerate(event_rows):
        for operation, core_id, col0, col1 in sorted(events[row]):
            if operation < 0:
                active.pop(core_id, None)
            else:
                active[core_id] = (col0, col1)
        next_row = event_rows[index + 1] if index + 1 < len(event_rows) else row
        if next_row <= row:
            continue
        if row < row_low or next_row > row_high:
            continue
        cursor = col_low
        for col0, col1 in sorted(active.values()):
            if col0 < cursor:
                raise ComponentIndexError("Core global windows overlap")
            if col0 > cursor:
                raise ComponentIndexError("Core global windows contain a gap")
            cursor = col1
        if cursor != col_high:
            raise ComponentIndexError("Core global windows contain a gap")
    if event_rows[0] != row_low or event_rows[-1] != row_high:
        raise ComponentIndexError("Core global windows contain a gap")

    # Canonical iteration makes component roots and query results independent of
    # the input tile order.  ``windows`` is retained only to make this explicit.
    del windows
    return sorted(checked, key=lambda item: item[0].core_id), (row_low, col_low, row_high, col_high)


def _label_core(
    tile: CoreTile,
    window: tuple[int, int, int, int],
    union_find: _ComponentUnionFind,
    query_points: Sequence[tuple[int, int]],
) -> tuple[_CoreBorders, dict[tuple[int, int], _Node]]:
    row0, col0, height, width = window
    labels, valid = tile.labels, tile.valid
    component_ids = np.zeros((height, width), dtype=np.int32)
    component_count = 0
    linear_indices = np.arange(height * width, dtype=np.int64).reshape(height, width)
    # The C implementation must handle the 140 real Cores.  SciPy performs the
    # raster scan in compiled code; a Python flood fill over every valid pixel
    # would be functionally correct but unusable at the full 831M-pixel scope.
    for class_code_raw in np.unique(labels[valid]):
        class_code = int(class_code_raw)
        local, count = ndimage.label(
            valid & (labels == class_code_raw), structure=FOUR_CONNECTED
        )
        if not count:
            continue
        selected = local > 0
        component_ids[selected] = local[selected].astype(np.int32) + component_count
        local_ids = np.arange(1, count + 1, dtype=np.int32)
        pixel_counts = np.bincount(local.ravel(), minlength=count + 1)[1:]
        minimum_linear = np.asarray(
            ndimage.minimum(linear_indices, labels=local, index=local_ids),
            dtype=np.int64,
        )
        for local_id, pixels, minimum in zip(
            local_ids.tolist(), pixel_counts.tolist(), minimum_linear.tolist()
        ):
            minimum_row, minimum_col = divmod(int(minimum), width)
            global_id = component_count + int(local_id)
            union_find.add(
                (tile.core_id, global_id),
                _NodeStats(
                    class_code,
                    int(pixels),
                    float(pixels) * float(tile.pixel_area_m2),
                    (row0 + minimum_row, col0 + minimum_col),
                ),
            )
        component_count += int(count)
    query_nodes: dict[tuple[int, int], _Node] = {}
    for row, col in query_points:
        local_row, local_col = row - row0, col - col0
        if not valid[local_row, local_col]:
            raise ComponentIndexError(f"query point {(row, col)} is not valid")
        query_nodes[(row, col)] = (tile.core_id, int(component_ids[local_row, local_col]))
    return _CoreBorders(
        core_id=tile.core_id,
        row0=row0,
        col0=col0,
        row1=row0 + height,
        col1=col0 + width,
        top=component_ids[0].copy(),
        bottom=component_ids[-1].copy(),
        left=component_ids[:, 0].copy(),
        right=component_ids[:, -1].copy(),
    ), query_nodes


def _union_matching_edges(
    first_core: str,
    first_ids: np.ndarray,
    second_core: str,
    second_ids: np.ndarray,
    union_find: _ComponentUnionFind,
) -> None:
    for first_id, second_id in zip(first_ids, second_ids):
        if not first_id or not second_id:
            continue
        first_node = (first_core, int(first_id))
        second_node = (second_core, int(second_id))
        if union_find.stats[union_find.find(first_node)].class_code == union_find.stats[
            union_find.find(second_node)
        ].class_code:
            union_find.union(first_node, second_node)


def _merge_seams(borders: Sequence[_CoreBorders], union_find: _ComponentUnionFind) -> None:
    tops: dict[int, list[_CoreBorders]] = defaultdict(list)
    bottoms: dict[int, list[_CoreBorders]] = defaultdict(list)
    lefts: dict[int, list[_CoreBorders]] = defaultdict(list)
    rights: dict[int, list[_CoreBorders]] = defaultdict(list)
    for border in borders:
        tops[border.row0].append(border)
        bottoms[border.row1].append(border)
        lefts[border.col0].append(border)
        rights[border.col1].append(border)
    for seam in sorted(set(tops).intersection(bottoms)):
        for lower in sorted(tops[seam], key=lambda item: item.core_id):
            for upper in sorted(bottoms[seam], key=lambda item: item.core_id):
                start, stop = max(lower.col0, upper.col0), min(lower.col1, upper.col1)
                if start < stop:
                    _union_matching_edges(
                        upper.core_id,
                        upper.bottom[start - upper.col0 : stop - upper.col0],
                        lower.core_id,
                        lower.top[start - lower.col0 : stop - lower.col0],
                        union_find,
                    )
    for seam in sorted(set(lefts).intersection(rights)):
        for right in sorted(lefts[seam], key=lambda item: item.core_id):
            for left in sorted(rights[seam], key=lambda item: item.core_id):
                start, stop = max(right.row0, left.row0), min(right.row1, left.row1)
                if start < stop:
                    _union_matching_edges(
                        left.core_id,
                        left.right[start - left.row0 : stop - left.row0],
                        right.core_id,
                        right.left[start - right.row0 : stop - right.row0],
                        union_find,
                    )


def build_global_component_index(
    tiles: Iterable[CoreTile],
    query_points: Iterable[tuple[int, int] | tuple[int, int, int]],
    *,
    global_shape: tuple[int, int] | None = None,
) -> ComponentIndexResult:
    """Return stable global components for a small set of global query pixels.

    Core rectangles must exactly partition ``global_shape`` when it is supplied.
    Without it, they must exactly partition their own bounding rectangle.  The
    implementation holds dense data only for one Core while it is labelled,
    then retains its four component-id borders; this is safe for domains too
    large to materialise as one dense global array.  A query may optionally be
    ``(row, col, expected_class_code)``; a mismatched class is rejected.
    """

    source_tiles = list(tiles)
    raw_points = list(query_points)
    points: list[tuple[int, int]] = []
    expected_codes: list[int | None] = []
    for point in raw_points:
        if not isinstance(point, tuple) or len(point) not in (2, 3):
            raise ComponentIndexError("each query point must be (row, col) or (row, col, expected_class_code)")
        points.append((_require_int(point[0], "query row"), _require_int(point[1], "query col")))
        expected_codes.append(None if len(point) == 2 else _require_int(point[2], "query expected_class_code"))
    checked, (row_low, col_low, row_high, col_high) = _validate_partition(source_tiles, global_shape)
    for row, col in points:
        if not (row_low <= row < row_high and col_low <= col < col_high):
            raise ComponentIndexError(f"query point {(row, col)} is outside the Core partition")

    queries_by_core: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for point in sorted(set(points)):
        row, col = point
        for tile, (tile_row0, tile_col0, height, width) in checked:
            if tile_row0 <= row < tile_row0 + height and tile_col0 <= col < tile_col0 + width:
                queries_by_core[tile.core_id].append(point)
                break
        else:  # Partition validation makes this unreachable, but keep it explicit.
            raise ComponentIndexError(f"query point {point} is not owned by a Core")

    union_find = _ComponentUnionFind()
    borders: list[_CoreBorders] = []
    answer_nodes: dict[tuple[int, int], _Node] = {}
    for tile, window in checked:
        border, nodes = _label_core(tile, window, union_find, queries_by_core[tile.core_id])
        borders.append(border)
        answer_nodes.update(nodes)
    _merge_seams(borders, union_find)

    roots = {union_find.find(node) for node in union_find.parent}
    answers = []
    for point, expected_code in zip(points, expected_codes):
        stats = union_find.stats[union_find.find(answer_nodes[point])]
        if expected_code is not None and stats.class_code != expected_code:
            raise ComponentIndexError(
                f"query point {point} expected class {expected_code}, found {stats.class_code}"
            )
        answers.append(
            QueryComponent(
                point,
                GlobalComponentKey(stats.class_code, *stats.min_point),
                stats.pixel_count,
                stats.area_m2,
            )
        )
    return ComponentIndexResult(tuple(answers), len(roots))
