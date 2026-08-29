"""Bounded-memory global connectivity metrics for partitioned categorical masks.

The implementation labels horizontal runs inside one Core at a time, collapses
them to Core-local components, and then unions only component identifiers that
touch across Core seams.  It never builds the full-domain label mosaic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window


class GlobalConnectivityError(RuntimeError):
    pass


@dataclass(frozen=True)
class _Edges:
    window: dict[str, int]
    top: np.ndarray
    bottom: np.ndarray
    left: np.ndarray
    right: np.ndarray


class _UnionFind:
    def __init__(self) -> None:
        self.parent: list[int] = []
        self.rank: list[int] = []
        self.class_index: list[int] = []
        self.area_m2: list[float] = []
        self.pixel_count: list[int] = []

    def add(self, class_index: int, *, pixel_count: int, area_m2: float) -> int:
        node = len(self.parent)
        self.parent.append(node)
        self.rank.append(0)
        self.class_index.append(int(class_index))
        self.area_m2.append(float(area_m2))
        self.pixel_count.append(int(pixel_count))
        return node

    def find(self, node: int) -> int:
        root = int(node)
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != node:
            parent = self.parent[node]
            self.parent[node] = root
            node = parent
        return root

    def union(self, first: int, second: int) -> int:
        first_root = self.find(int(first))
        second_root = self.find(int(second))
        if first_root == second_root:
            return first_root
        if self.class_index[first_root] != self.class_index[second_root]:
            raise GlobalConnectivityError("attempted to union different classes")
        if self.rank[first_root] < self.rank[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        self.area_m2[first_root] += self.area_m2[second_root]
        self.pixel_count[first_root] += self.pixel_count[second_root]
        if self.rank[first_root] == self.rank[second_root]:
            self.rank[first_root] += 1
        return first_root


def _window(value: Mapping[str, Any]) -> dict[str, int]:
    result = {key: int(value[key]) for key in ("x0", "y0", "x1", "y1")}
    if result["x0"] >= result["x1"] or result["y0"] >= result["y1"]:
        raise GlobalConnectivityError("Core window has no positive area")
    return result


def _class_indices(values: np.ndarray, class_codes: Sequence[int], encoding: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.int16)
    if encoding == "indices":
        if np.any(result < -1) or np.any(result >= len(class_codes)):
            raise GlobalConnectivityError("mask contains invalid class indices")
        return result
    if encoding != "class_codes":
        raise GlobalConnectivityError(f"unknown mask encoding: {encoding}")
    mapped = np.full(result.shape, -1, dtype=np.int16)
    for index, code in enumerate(class_codes):
        mapped[result == int(code)] = index
    if np.any((result != -1) & (mapped < 0)):
        raise GlobalConnectivityError("mask contains unknown class codes")
    return mapped


def _rows(
    record: Mapping[str, Any],
    class_codes: Sequence[int],
    *,
    expected_crs: str,
    global_transform: Affine,
    block_rows: int,
):
    path = Path(str(record["path"]))
    core = _window(record["core_window"])
    height = core["y1"] - core["y0"]
    width = core["x1"] - core["x0"]
    encoding = str(record.get("encoding") or "indices")
    if path.suffix.lower() == ".npy":
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.dtype != np.int16 or values.shape != (height, width):
            raise GlobalConnectivityError(f"Core NPY contract differs: {path}")
        for row0 in range(0, height, block_rows):
            block = _class_indices(
                np.asarray(values[row0 : row0 + block_rows]), class_codes, encoding
            )
            for row in block:
                yield row
        return
    with rasterio.open(path) as source:
        expected_transform = global_transform * Affine.translation(core["x0"], core["y0"])
        if (
            source.count != 1
            or str(source.crs or "") != expected_crs
            or source.dtypes != ("int16",)
            or source.nodata != -1
            or (source.height, source.width) != (height, width)
            or not source.transform.almost_equals(expected_transform)
        ):
            raise GlobalConnectivityError(f"Core raster contract differs: {path}")
        if encoding != "indices":
            raise GlobalConnectivityError("GeoTIFF Core masks must use class indices")
        for row0 in range(0, height, block_rows):
            count = min(block_rows, height - row0)
            block = _class_indices(
                source.read(1, window=Window(0, row0, width, count)),
                class_codes,
                encoding,
            )
            for row in block:
                yield row


def _runs(row: np.ndarray, local: _UnionFind, pixel_area_m2: float):
    width = int(row.shape[0])
    cuts = np.flatnonzero(row[1:] != row[:-1]) + 1
    bounds = np.concatenate((np.asarray([0]), cuts, np.asarray([width])))
    result: list[tuple[int, int, int, int]] = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        class_index = int(row[int(start)])
        if class_index < 0:
            continue
        count = int(end - start)
        node = local.add(
            class_index,
            pixel_count=count,
            area_m2=float(count) * pixel_area_m2,
        )
        result.append((int(start), int(end), class_index, node))
    return result


def _label_core(
    record: Mapping[str, Any],
    class_codes: Sequence[int],
    global_components: _UnionFind,
    *,
    expected_crs: str,
    global_transform: Affine,
    block_rows: int,
) -> tuple[_Edges, int]:
    core = _window(record["core_window"])
    height = core["y1"] - core["y0"]
    width = core["x1"] - core["x0"]
    pixel_area_m2 = float(record["pixel_area_m2"])
    if not np.isfinite(pixel_area_m2) or pixel_area_m2 <= 0:
        raise GlobalConnectivityError("pixel area must be finite and positive")
    local = _UnionFind()
    top = np.full(width, -1, dtype=np.int64)
    bottom = np.full(width, -1, dtype=np.int64)
    left = np.full(height, -1, dtype=np.int64)
    right = np.full(height, -1, dtype=np.int64)
    previous: list[tuple[int, int, int, int]] = []
    row_count = 0
    for row_index, row in enumerate(
        _rows(
            record,
            class_codes,
            expected_crs=expected_crs,
            global_transform=global_transform,
            block_rows=block_rows,
        )
    ):
        current = _runs(row, local, pixel_area_m2)
        first = second = 0
        while first < len(previous) and second < len(current):
            p_start, p_end, p_class, p_node = previous[first]
            c_start, c_end, c_class, c_node = current[second]
            if max(p_start, c_start) < min(p_end, c_end) and p_class == c_class:
                local.union(p_node, c_node)
            if p_end <= c_end:
                first += 1
            else:
                second += 1
        row_edges = np.full(width, -1, dtype=np.int64)
        for start, end, _class_index, node in current:
            row_edges[start:end] = node
        if row_index == 0:
            top[:] = row_edges
        bottom[:] = row_edges
        left[row_index] = row_edges[0]
        right[row_index] = row_edges[-1]
        previous = current
        row_count += 1
    if row_count != height:
        raise GlobalConnectivityError("Core row reader returned the wrong height")

    local_root_to_global: dict[int, int] = {}
    for node in range(len(local.parent)):
        root = local.find(node)
        if root != node:
            continue
        local_root_to_global[root] = global_components.add(
            local.class_index[root],
            pixel_count=local.pixel_count[root],
            area_m2=local.area_m2[root],
        )

    def translate(edge: np.ndarray) -> np.ndarray:
        translated = np.full(edge.shape, -1, dtype=np.int64)
        valid = edge >= 0
        if np.any(valid):
            translated[valid] = np.fromiter(
                (
                    local_root_to_global[local.find(int(node))]
                    for node in edge[valid]
                ),
                dtype=np.int64,
                count=int(np.count_nonzero(valid)),
            )
        return translated

    return (
        _Edges(
            window=core,
            top=translate(top),
            bottom=translate(bottom),
            left=translate(left),
            right=translate(right),
        ),
        len(local.parent),
    )


def _union_edge_pairs(
    components: _UnionFind, first: np.ndarray, second: np.ndarray
) -> None:
    valid = (first >= 0) & (second >= 0)
    if not np.any(valid):
        return
    pairs = np.unique(np.column_stack((first[valid], second[valid])), axis=0)
    for first_node, second_node in pairs:
        if components.class_index[components.find(int(first_node))] == components.class_index[
            components.find(int(second_node))
        ]:
            components.union(int(first_node), int(second_node))


def _union_seams(components: _UnionFind, edges: Sequence[_Edges]) -> int:
    seam_pair_count = 0
    for index, first in enumerate(edges):
        a = first.window
        for second in edges[index + 1 :]:
            b = second.window
            if a["x1"] == b["x0"] or b["x1"] == a["x0"]:
                left, right = (first, second) if a["x1"] == b["x0"] else (second, first)
                y0 = max(left.window["y0"], right.window["y0"])
                y1 = min(left.window["y1"], right.window["y1"])
                if y0 < y1:
                    _union_edge_pairs(
                        components,
                        left.right[y0 - left.window["y0"] : y1 - left.window["y0"]],
                        right.left[y0 - right.window["y0"] : y1 - right.window["y0"]],
                    )
                    seam_pair_count += 1
            if a["y1"] == b["y0"] or b["y1"] == a["y0"]:
                top, bottom = (first, second) if a["y1"] == b["y0"] else (second, first)
                x0 = max(top.window["x0"], bottom.window["x0"])
                x1 = min(top.window["x1"], bottom.window["x1"])
                if x0 < x1:
                    _union_edge_pairs(
                        components,
                        top.bottom[x0 - top.window["x0"] : x1 - top.window["x0"]],
                        bottom.top[x0 - bottom.window["x0"] : x1 - bottom.window["x0"]],
                    )
                    seam_pair_count += 1
    return seam_pair_count


def audit_partitioned_connectivity(
    records: Sequence[Mapping[str, Any]],
    *,
    class_codes: Sequence[int],
    dynamic_thresholds_m2: Mapping[int, float],
    expected_crs: str,
    global_transform: Affine,
    block_rows: int = 256,
    progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Return exact global 4-connected metrics without a full-domain mosaic."""

    if not records:
        raise GlobalConnectivityError("global connectivity audit has no Cores")
    if block_rows <= 0:
        raise GlobalConnectivityError("block_rows must be positive")
    thresholds = {
        int(code): float(dynamic_thresholds_m2[int(code)]) for code in class_codes
    }
    components = _UnionFind()
    edges: list[_Edges] = []
    local_run_count = 0
    peak_partition_pixels = 0
    for index, record in enumerate(records, start=1):
        core = _window(record["core_window"])
        peak_partition_pixels = max(
            peak_partition_pixels,
            (core["x1"] - core["x0"]) * (core["y1"] - core["y0"]),
        )
        item_edges, item_runs = _label_core(
            record,
            class_codes,
            components,
            expected_crs=expected_crs,
            global_transform=global_transform,
            block_rows=block_rows,
        )
        edges.append(item_edges)
        local_run_count += item_runs
        if progress is not None:
            progress(index)
    seam_pair_count = _union_seams(components, edges)

    by_class = {
        str(int(code)): {
            "components_4_connected": 0,
            "dynamic_fragments_4_connected": 0,
            "pixel_count": 0,
            "area_m2": 0.0,
            "dynamic_fragment_area_m2": 0.0,
        }
        for code in class_codes
    }
    for node in range(len(components.parent)):
        if components.find(node) != node:
            continue
        code = int(class_codes[components.class_index[node]])
        metrics = by_class[str(code)]
        metrics["components_4_connected"] += 1
        metrics["pixel_count"] += int(components.pixel_count[node])
        metrics["area_m2"] += float(components.area_m2[node])
        if components.area_m2[node] < thresholds[code]:
            metrics["dynamic_fragments_4_connected"] += 1
            metrics["dynamic_fragment_area_m2"] += float(components.area_m2[node])
    return {
        "components_4_connected": sum(
            int(item["components_4_connected"]) for item in by_class.values()
        ),
        "dynamic_fragments_4_connected": sum(
            int(item["dynamic_fragments_4_connected"]) for item in by_class.values()
        ),
        "by_class": by_class,
        "algorithm": "partition_row_run_union_4_connected_v1",
        "partition_count": len(records),
        "seam_pair_count": seam_pair_count,
        "core_local_component_node_count": len(components.parent),
        "core_local_run_count": local_run_count,
        "block_rows": int(block_rows),
        "peak_partition_pixels": int(peak_partition_pixels),
        "memory_model": "one raster row block plus local run union and Core seam vectors",
    }

def connectivity_hard_gate(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare global total and per-class component/fragment counts."""

    by_class: dict[str, dict[str, Any]] = {}
    for code, baseline_metrics in dict(baseline["by_class"]).items():
        candidate_metrics = dict(candidate["by_class"])[code]
        component_delta = int(candidate_metrics["components_4_connected"]) - int(
            baseline_metrics["components_4_connected"]
        )
        fragment_delta = int(candidate_metrics["dynamic_fragments_4_connected"]) - int(
            baseline_metrics["dynamic_fragments_4_connected"]
        )
        by_class[str(code)] = {
            "component_delta": component_delta,
            "dynamic_fragment_delta": fragment_delta,
            "components_nonincrease": component_delta <= 0,
            "dynamic_fragments_nonincrease": fragment_delta <= 0,
        }
    total_component_delta = int(candidate["components_4_connected"]) - int(
        baseline["components_4_connected"]
    )
    total_fragment_delta = int(candidate["dynamic_fragments_4_connected"]) - int(
        baseline["dynamic_fragments_4_connected"]
    )
    passed = (
        total_component_delta <= 0
        and total_fragment_delta <= 0
        and all(
            row["components_nonincrease"] and row["dynamic_fragments_nonincrease"]
            for row in by_class.values()
        )
    )
    return {
        "passed": bool(passed),
        "total_component_delta": total_component_delta,
        "total_dynamic_fragment_delta": total_fragment_delta,
        "by_class": by_class,
    }
