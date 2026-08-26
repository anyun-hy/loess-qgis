#!/usr/bin/env python3
"""Exact global 4-neighbour fragmentation evaluator for the 140-Core V3/V3.1 run.

This is deliberately a *read-only* evaluator.  It does not execute V3, V3.1,
or any RAG code.  Its input is the final Core manifest produced by the full
runner.  A Core may be an NPZ containing ``v3``, ``v31``/``v31a`` and ``valid``
arrays, or the three arrays may be named separately in the manifest.

Required manifest contract (paths may be absolute or relative to the manifest):

  {"parts": [{"part_id": "P001", "core_npz": "P001.npz",
              "transform": [xres, 0, x0, 0, -yres, y0],
              "core_window": [row0, row1, col0, col1]}],
   "approved_dynamic_mmu_m2": {"12": 100.0, "33": 25.0}}

``core_window`` is optional if each array is already a Core.  Separate files
are also accepted as ``raw``, ``v3``, ``v31`` (or ``v31a``), and ``valid``.  Every Core
must be north-up, aligned to one common integer pixel grid, provide its own
ground ``physical_metrics``, and be non-overlapping.  Ground pixel area may
legitimately vary between EPSG:3857 latitude bands.  These restrictions make cross-Core component
merging exact rather than an approximation at partition boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import ndimage


SCHEMA_VERSION = 1
METHODS = ("raw", "v3", "v31")
PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
CURVE_M2 = (25.0, 50.0, 100.0, 200.0)
TOL = 1e-7


class EvaluationError(RuntimeError):
    """A manifest/data contract error that must stop a full evaluation."""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
        handle.write("\n")
    os.replace(temporary, path)


def _as_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def _window(value: Any, shape: tuple[int, int], identity: str) -> tuple[int, int, int, int]:
    if value is None:
        return (0, shape[0], 0, shape[1])
    if isinstance(value, dict):
        keys = ("row_start", "row_stop", "col_start", "col_stop")
        if not all(key in value for key in keys):
            raise EvaluationError(f"{identity}: core_window dict needs {keys}")
        raw = [value[key] for key in keys]
    else:
        raw = list(value)
    if len(raw) != 4 or any(isinstance(item, bool) or int(item) != item for item in raw):
        raise EvaluationError(f"{identity}: core_window must be [row0,row1,col0,col1]")
    r0, r1, c0, c1 = (int(item) for item in raw)
    if not (0 <= r0 < r1 <= shape[0] and 0 <= c0 < c1 <= shape[1]):
        raise EvaluationError(f"{identity}: core_window {raw} is outside array shape {shape}")
    return r0, r1, c0, c1


def _global_window(value: Any, identity: str) -> tuple[int, int, int, int]:
    """Full runner names global grid axes x/y; internal arrays use row/column."""
    if isinstance(value, dict) and all(key in value for key in ("x0", "x1", "y0", "y1")):
        raw = [value["y0"], value["y1"], value["x0"], value["x1"]]
        return _window(raw, (2**31 - 1, 2**31 - 1), identity)
    return _window(value, (2**31 - 1, 2**31 - 1), identity)


def _transform(value: Any, identity: str) -> tuple[float, float, float, float, float, float]:
    if value is None or len(value) != 6:
        raise EvaluationError(f"{identity}: a six-value affine transform is required")
    try:
        affine = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"{identity}: transform contains a non-numeric value") from exc
    if not all(math.isfinite(item) for item in affine):
        raise EvaluationError(f"{identity}: transform must be finite")
    a, b, _c, d, e, _f = affine
    if abs(b) > TOL or abs(d) > TOL or a <= 0 or e >= 0:
        raise EvaluationError(f"{identity}: only north-up transforms [xres,0,x0,0,-yres,y0] are supported")
    return affine


def _get(part: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in part:
            return part[name]
    arrays = part.get("arrays")
    if isinstance(arrays, dict):
        for name in names:
            if name in arrays:
                return arrays[name]
    return default


def _load_array(value: Any, key: str, manifest_path: Path, identity: str, opened: dict[Path, Any]) -> tuple[np.ndarray, list[Path]]:
    """Load ``path`` or {path,key}; NPZ fields are selected by ``key`` by default."""
    source_key = key
    if isinstance(value, dict):
        filename = value.get("path", value.get("file"))
        source_key = str(value.get("key", source_key))
        declared_sha = value.get("sha256")
    else:
        filename = value
        declared_sha = None
    if not isinstance(filename, str):
        raise EvaluationError(f"{identity}: {key} must name an array file")
    path = _as_path(filename, manifest_path)
    if not path.is_file():
        raise EvaluationError(f"{identity}: missing {key} input: {path}")
    if declared_sha is not None:
        if not isinstance(declared_sha, str) or _sha256_path(path) != declared_sha:
            raise EvaluationError(f"{identity}: declared SHA-256 mismatch for {key}: {path}")
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False), [path]
    if path not in opened:
        opened[path] = np.load(path, allow_pickle=False)
    archive = opened[path]
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise EvaluationError(f"{identity}: {path} is not an .npy/.npz array input")
    candidates = (source_key, "v31a" if source_key == "v31" else source_key)
    for candidate in candidates:
        if candidate in archive.files:
            return archive[candidate], [path]
    raise EvaluationError(f"{identity}: {path} has no '{source_key}' field")


def _part_arrays(part: dict[str, Any], manifest_path: Path, identity: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Path]]:
    opened: dict[Path, Any] = {}
    files: list[Path] = []
    container = _get(part, "core_npz", "core_output", "output_npz", "npz", "path")
    outputs = _get(part, "outputs", "output_files", "core_outputs", default={})
    if not isinstance(outputs, dict):
        outputs = {}
    values = {
        "raw": _get(part, "raw", "raw_path", default=outputs.get("raw_core", outputs.get("raw"))),
        "v3": _get(part, "v3", "v3_path", default=outputs.get("v3_core", outputs.get("v3"))),
        "v31": _get(part, "v31", "v31a", "v31_path", "v31a_path", default=outputs.get("v31a_core", outputs.get("v31_core", outputs.get("v31a", outputs.get("v31"))))),
        "valid": _get(part, "valid", "valid_path", "valid_mask", default=outputs.get("valid_core", outputs.get("valid"))),
    }
    arrays: dict[str, np.ndarray] = {}
    for key, value in values.items():
        selected = value if value is not None else container
        if selected is None:
            raise EvaluationError(f"{identity}: no {key} input or shared core_npz")
        array, used = _load_array(selected, key, manifest_path, identity, opened)
        arrays[key] = np.asarray(array)
        files.extend(used)
    for archive in opened.values():
        if isinstance(archive, np.lib.npyio.NpzFile):
            archive.close()
    return arrays["raw"], arrays["v3"], arrays["v31"], arrays["valid"], sorted(set(files))


def _part_with_audit(raw: dict[str, Any], manifest_path: Path, identity: str) -> tuple[dict[str, Any], Path | None]:
    """Runner manifests may keep geospatial/physical metadata in each audit.json."""
    pointer = _get(raw, "audit_path", "audit")
    if isinstance(pointer, dict):
        declared_sha = pointer.get("sha256")
        pointer = pointer.get("path", pointer.get("file"))
    else:
        declared_sha = None
    if pointer is None:
        return raw, None
    if not isinstance(pointer, str):
        raise EvaluationError(f"{identity}: audit path must be a string")
    audit_path = _as_path(pointer, manifest_path)
    if not audit_path.is_file():
        raise EvaluationError(f"{identity}: missing audit input: {audit_path}")
    if declared_sha is not None:
        if not isinstance(declared_sha, str) or _sha256_path(audit_path) != declared_sha:
            raise EvaluationError(f"{identity}: declared SHA-256 mismatch for audit: {audit_path}")
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"{identity}: invalid audit JSON {audit_path}: {exc}") from exc
    if not isinstance(audit, dict):
        raise EvaluationError(f"{identity}: audit must contain an object")
    merged = dict(audit)
    merged.update(raw)  # the run manifest is authoritative for paths/windows
    for name in ("outputs", "output_files", "physical_metrics"):
        if isinstance(audit.get(name), dict) and isinstance(raw.get(name), dict):
            value = dict(audit[name]); value.update(raw[name]); merged[name] = value
    return merged, audit_path


def _extract_mmu(manifest: dict[str, Any]) -> dict[int, float]:
    direct = manifest.get("approved_dynamic_mmu_m2", manifest.get("dynamic_fragmentation_m2_by_class"))
    if direct is None:
        snapshots = manifest.get("policy_snapshot", manifest.get("v31a_policy_snapshot"))
        if isinstance(snapshots, dict) and "v31a" in snapshots:
            snapshots = snapshots["v31a"]
        if isinstance(snapshots, dict):
            direct = snapshots.get("dynamic_fragmentation_m2_by_class", snapshots.get("class_policies"))
    if not isinstance(direct, dict) or not direct:
        raise EvaluationError("manifest needs approved_dynamic_mmu_m2 (class code -> m²); no policy is inferred")
    result: dict[int, float] = {}
    for raw_code, raw_value in direct.items():
        try:
            code = int(raw_code)
            value = raw_value.get("dynamic_fragmentation_m2") if isinstance(raw_value, dict) else raw_value
            mmu = float(value)
        except (TypeError, ValueError) as exc:
            raise EvaluationError(f"invalid approved MMU entry {raw_code!r}: {raw_value!r}") from exc
        if code < 0 or not math.isfinite(mmu) or mmu < 0:
            raise EvaluationError(f"invalid approved MMU for class {code}: {mmu}")
        result[code] = mmu
    return result


@dataclass
class Part:
    part_id: str
    raw: np.ndarray
    v3: np.ndarray
    v31: np.ndarray
    valid: np.ndarray
    row0: int
    col0: int
    grid_xres: float
    grid_yres: float
    pixel_area_m2: float
    row_step_m: float
    column_step_m: float
    files: list[Path]
    transform: tuple[float, float, float, float, float, float]

    @property
    def shape(self) -> tuple[int, int]:
        return self.valid.shape


def _physical_metrics(raw: dict[str, Any], manifest: dict[str, Any], identity: str) -> tuple[float, float, float]:
    metrics = _get(raw, "physical_metrics", default=manifest.get("physical_metrics"))
    if not isinstance(metrics, dict):
        raise EvaluationError(f"{identity}: physical_metrics with pixel_area_m2/row_step_m/column_step_m is required")
    try:
        area, row, column = (float(metrics[name]) for name in ("pixel_area_m2", "row_step_m", "column_step_m"))
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationError(f"{identity}: malformed physical_metrics") from exc
    if not all(math.isfinite(value) and value > 0 for value in (area, row, column)):
        raise EvaluationError(f"{identity}: physical_metrics values must be finite positive numbers")
    return area, row, column


def _prepare(manifest_path: Path) -> tuple[dict[str, Any], list[Part], dict[int, float], list[dict[str, str]]]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read manifest {manifest_path}: {exc}") from exc
    raw_parts = manifest.get("partitions", manifest.get("parts", manifest.get("core_parts")))
    if not isinstance(raw_parts, list) or not raw_parts:
        raise EvaluationError("manifest needs a non-empty parts/core_parts list")
    if manifest.get("kind") == "v31a_full_partition_core_comparison" and not manifest.get("self_test", False):
        if manifest.get("status") != "complete":
            raise EvaluationError("full runner manifest is not complete")
        if len(raw_parts) != 140 or manifest.get("completed_partition_count") != 140:
            raise EvaluationError("full runner evaluation requires all 140 completed Core partitions")
    mmu = _extract_mmu(manifest)
    root_transform = manifest.get("processing_transform", manifest.get("transform", manifest.get("affine")))
    root_affine = _transform(root_transform, "manifest processing_transform") if root_transform is not None else None
    root_crs = manifest.get("processing_crs", manifest.get("crs"))
    first_crs: str | None = None
    staged: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float, float, float, float], tuple[float, float, float], list[Path], tuple[int, int] | None]] = []
    file_hashes: dict[Path, str] = {manifest_path.resolve(): _sha256_path(manifest_path)}
    first_affine: tuple[float, float, float, float, float, float] | None = None
    max_y: float | None = None
    for index, raw in enumerate(raw_parts):
        if not isinstance(raw, dict):
            raise EvaluationError(f"part {index}: expected an object")
        identity = str(_get(raw, "part_id", "partition_id", "id", "name", default=f"part_{index:03d}"))
        if manifest.get("kind") == "v31a_full_partition_core_comparison" and not manifest.get("self_test", False):
            required = raw.get("outputs")
            if not isinstance(required, dict) or set(("raw", "v3", "v31a", "valid")) - set(required):
                raise EvaluationError(f"{identity}: full runner requires fixed raw/v3/v31a/valid outputs")
            if any(not isinstance(required[name], dict) or not isinstance(required[name].get("sha256"), str) for name in ("raw", "v3", "v31a", "valid")):
                raise EvaluationError(f"{identity}: full runner output SHA-256 declarations are required")
        raw, audit_path = _part_with_audit(raw, manifest_path, identity)
        raw_labels, v3, v31, valid, paths = _part_arrays(raw, manifest_path, identity)
        if audit_path is not None:
            paths.append(audit_path)
        if raw_labels.ndim != 2 or v3.ndim != 2 or v31.ndim != 2 or valid.ndim != 2 or raw_labels.shape != v3.shape or v3.shape != v31.shape or v3.shape != valid.shape:
            raise EvaluationError(f"{identity}: raw/v3/v31/valid must be same-shape 2-D arrays")
        if not all(np.issubdtype(value.dtype, np.integer) for value in (raw_labels, v3, v31)):
            raise EvaluationError(f"{identity}: raw, v3 and v31 labels must be integer arrays")
        valid = valid.astype(bool, copy=False)
        window = _window(_get(raw, "core_window", default=manifest.get("core_window")), v3.shape, identity)
        r0, r1, c0, c1 = window
        raw_labels, v3, v31, valid = raw_labels[r0:r1, c0:c1], v3[r0:r1, c0:c1], v31[r0:r1, c0:c1], valid[r0:r1, c0:c1]
        affine = _transform(_get(raw, "core_transform", "transform", "affine", default=root_transform), identity)
        crs = _get(raw, "crs", "core_crs", default=root_crs)
        if not isinstance(crs, str) or not crs.strip():
            raise EvaluationError(f"{identity}: CRS is required")
        if first_crs is None:
            first_crs = crs
        elif crs != first_crs:
            raise EvaluationError(f"{identity}: CRS differs from first Core ({crs!r} != {first_crs!r})")
        if isinstance(root_crs, str) and crs != root_crs:
            raise EvaluationError(f"{identity}: CRS disagrees with manifest processing_crs")
        a, _b, c, _d, e, f = affine
        # Window offsets must affect the Core's geospatial origin.
        affine = (a, 0.0, c + c0 * a, 0.0, e, f + r0 * e)
        physical = _physical_metrics(raw, manifest, identity)
        global_core = _get(raw, "global_core_window")
        global_origin: tuple[int, int] | None = None
        if global_core is not None:
            gr0, gr1, gc0, gc1 = _global_window(global_core, f"{identity}: global_core_window")
            if (gr1 - gr0, gc1 - gc0) != v3.shape:
                raise EvaluationError(f"{identity}: global_core_window size does not equal Core array shape {v3.shape}")
            global_origin = (gr0, gc0)
        if first_affine is None:
            first_affine = affine
            max_y = affine[5]
        else:
            if abs(a - first_affine[0]) > TOL or abs(e - first_affine[4]) > TOL:
                raise EvaluationError(f"{identity}: pixel size differs from first Core")
            max_y = max(float(max_y), affine[5])
        for path in paths:
            file_hashes.setdefault(path.resolve(), _sha256_path(path))
        staged.append((identity, raw_labels, v3, v31, valid, affine, physical, paths, global_origin))
    assert first_affine is not None and max_y is not None
    parts: list[Part] = []
    for identity, raw_labels, v3, v31, valid, affine, physical, paths, global_origin in staged:
        a, _b, c, _d, e, f = affine
        if root_affine is None:
            col_float = (c - first_affine[2]) / a
            row_float = (max_y - f) / abs(e)
        else:
            if abs(a - root_affine[0]) > TOL or abs(e - root_affine[4]) > TOL:
                raise EvaluationError(f"{identity}: core_transform grid step differs from processing_transform")
            col_float = (c - root_affine[2]) / a
            row_float = (root_affine[5] - f) / abs(e)
        calculated_col0, calculated_row0 = round(col_float), round(row_float)
        if abs(col_float - calculated_col0) > TOL or abs(row_float - calculated_row0) > TOL:
            raise EvaluationError(f"{identity}: Core origin is not aligned to the common integer pixel grid")
        if global_origin is None:
            row0, col0 = calculated_row0, calculated_col0
        else:
            row0, col0 = global_origin
            # A global window and its affine must describe the same Core placement.
            if (row0, col0) != (calculated_row0, calculated_col0):
                raise EvaluationError(f"{identity}: global_core_window disagrees with core_transform/common affine")
        pixel_area, row_step, column_step = physical
        parts.append(Part(identity, raw_labels, v3, v31, valid, row0, col0, a, abs(e), pixel_area, row_step, column_step, paths, affine))
    ids = [part.part_id for part in parts]
    if len(set(ids)) != len(ids):
        raise EvaluationError("Core part_id values must be unique")
    # Rectangle-overlap checking is exact in the shared pixel grid and costs little for 140 Cores.
    overlap_pixels = 0
    for left in range(len(parts)):
        a = parts[left]
        for right in range(left + 1, len(parts)):
            b = parts[right]
            high_r, low_r = min(a.row0 + a.shape[0], b.row0 + b.shape[0]), max(a.row0, b.row0)
            high_c, low_c = min(a.col0 + a.shape[1], b.col0 + b.shape[1]), max(a.col0, b.col0)
            if high_r > low_r and high_c > low_c:
                overlap_pixels += (high_r - low_r) * (high_c - low_c)
    if overlap_pixels:
        raise EvaluationError(f"Core windows overlap by {overlap_pixels} pixels; refusing ambiguous global topology")
    declared_global = manifest.get("global_window")
    if declared_global is not None:
        gr0, gr1, gc0, gc1 = _global_window(declared_global, "manifest global_window")
        for part in parts:
            if not (gr0 <= part.row0 and part.row0 + part.shape[0] <= gr1 and gc0 <= part.col0 and part.col0 + part.shape[1] <= gc1):
                raise EvaluationError(f"{part.part_id}: Core lies outside manifest global_window")
    hashes = [{"path": str(path), "sha256": value} for path, value in sorted(file_hashes.items(), key=lambda item: str(item[0]))]
    return manifest, parts, mmu, hashes


class UnionFind:
    def __init__(self) -> None:
        self.parent: list[int] = []

    def add(self, count: int) -> int:
        first = len(self.parent)
        self.parent.extend(range(first, first + count))
        return first

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra > rb:
                ra, rb = rb, ra
            self.parent[rb] = ra


def _perimeter(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    cols = np.arange(width, dtype=np.int64)
    rows = np.arange(height, dtype=np.int64)
    if height == 1:
        return np.zeros(width, dtype=np.int64), cols
    if width == 1:
        return rows, np.zeros(height, dtype=np.int64)
    return (np.concatenate((np.zeros(width, dtype=np.int64), np.full(width, height - 1, dtype=np.int64), rows[1:-1], rows[1:-1])),
            np.concatenate((cols, cols, np.zeros(height - 2, dtype=np.int64), np.full(height - 2, width - 1, dtype=np.int64))))


def _internal_edges(labels: np.ndarray, valid: np.ndarray, row_step_m: float, column_step_m: float) -> dict[str, float | int]:
    horizontal = valid[:, :-1] & valid[:, 1:] & (labels[:, :-1] != labels[:, 1:])
    vertical = valid[:-1, :] & valid[1:, :] & (labels[:-1, :] != labels[1:, :])
    # left/right neighbours share a vertical edge (row step); top/bottom a horizontal edge (column step).
    return {"edges": int(horizontal.sum() + vertical.sum()), "metres": float(horizontal.sum() * row_step_m + vertical.sum() * column_step_m)}


def _component_labels(labels: np.ndarray, valid: np.ndarray, code: int) -> tuple[np.ndarray, int]:
    # SciPy's labeller executes C-level scanning.  Calling it per present class is
    # intentional: it is the only way to preserve label identity in 4-connectivity.
    return ndimage.label(valid & (labels == code), structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))


def _evaluate_method(parts: list[Part], method: str, mmu: dict[int, float]) -> dict[str, Any]:
    uf = UnionFind()
    component_codes: list[int] = []
    component_pixels: list[int] = []
    component_area_m2: list[float] = []
    boundary_keys: list[np.ndarray] = []
    boundary_codes: list[np.ndarray] = []
    boundary_components: list[np.ndarray] = []
    boundary_parts: list[np.ndarray] = []
    outer_target_keys: list[np.ndarray] = []
    outer_target_metres: list[np.ndarray] = []
    internal_edges = 0
    internal_metres = 0.0
    strict_outer_edges = 0
    strict_outer_metres = 0.0
    for part_index, part in enumerate(parts):
        labels = part.raw if method == "raw" else (part.v3 if method == "v3" else part.v31)
        edge = _internal_edges(labels, part.valid, part.row_step_m, part.column_step_m)
        internal_edges += int(edge["edges"])
        internal_metres += float(edge["metres"])
        invalid_horizontal = part.valid[:, :-1] != part.valid[:, 1:]
        invalid_vertical = part.valid[:-1, :] != part.valid[1:, :]
        strict_outer_edges += int(invalid_horizontal.sum() + invalid_vertical.sum())
        strict_outer_metres += float(invalid_horizontal.sum() * part.row_step_m + invalid_vertical.sum() * part.column_step_m)
        rows, cols = _perimeter(part.shape)
        local_component = np.zeros(part.shape, dtype=np.int64)
        # Labels outside valid are excluded, while labels inside valid must have an approved MMU.
        codes = np.unique(labels[part.valid])
        for raw_code in codes:
            code = int(raw_code)
            if code not in mmu:
                raise EvaluationError(f"{part.part_id}: class {code} lacks an approved dynamic MMU")
            tagged, count = _component_labels(labels, part.valid, code)
            if not count:
                continue
            counts = np.bincount(tagged.ravel(), minlength=count + 1)[1:]
            start = uf.add(count)
            component_codes.extend([code] * count)
            component_pixels.extend(int(value) for value in counts)
            component_area_m2.extend(float(value * part.pixel_area_m2) for value in counts)
            target = tagged > 0
            local_component[target] = tagged[target] + start - 1
        perimeter_valid = part.valid[rows, cols]
        if perimeter_valid.any():
            gy = rows[perimeter_valid] + part.row0
            gx = cols[perimeter_valid] + part.col0
            # r/gx are non-negative owing to common-grid normalization.  Packing
            # is collision-free for imagery dimensions below 2^32 pixels per axis.
            if int(gy.max(initial=0)) >= 2**32 or int(gx.max(initial=0)) >= 2**32:
                raise EvaluationError("Core grid exceeds 32-bit coordinate packing limit")
            boundary_keys.append((gy.astype(np.uint64) << np.uint64(32)) | gx.astype(np.uint64))
            boundary_codes.append(labels[rows[perimeter_valid], cols[perimeter_valid]].astype(np.int64, copy=False))
            boundary_components.append(local_component[rows[perimeter_valid], cols[perimeter_valid]])
            boundary_parts.append(np.full(int(perimeter_valid.sum()), part_index, dtype=np.int32))
        # Directed external sides only: a valid cell contributes when its
        # neighbour is outside this Core. Later lookup decides whether that
        # neighbour is valid in another Core, strict-invalid, or globally absent.
        height, width = part.shape
        sides = ((np.zeros(width, dtype=np.int64), np.arange(width, dtype=np.int64), -1, 0, part.column_step_m),
                 (np.full(width, height - 1, dtype=np.int64), np.arange(width, dtype=np.int64), 1, 0, part.column_step_m),
                 (np.arange(height, dtype=np.int64), np.zeros(height, dtype=np.int64), 0, -1, part.row_step_m),
                 (np.arange(height, dtype=np.int64), np.full(height, width - 1, dtype=np.int64), 0, 1, part.row_step_m))
        for side_rows, side_cols, dr, dc, step in sides:
            keep = part.valid[side_rows, side_cols]
            if keep.any():
                gy = side_rows[keep] + part.row0 + dr
                gx = side_cols[keep] + part.col0 + dc
                outer_target_keys.append((gy.astype(np.uint64) << np.uint64(32)) | gx.astype(np.uint64))
                outer_target_metres.append(np.full(int(keep.sum()), step, dtype=np.float64))
    if not component_codes:
        return {"components_total": 0, "per_class": [], "dynamic_fragments": {"count": 0, "area_m2": 0.0}, "curve": [], "component_area_bins": [], "area_percentiles_m2": {}, "boundary_definition": "cross-class boundaries exclude strict-valid study-range outer boundaries", "boundary": {"internal_cross_class_boundary": {"edges": internal_edges, "metres": internal_metres}, "cross_core_cross_class_boundary": {"edges": 0, "metres": 0.0}, "total_cross_class_boundary": {"edges": internal_edges, "metres": internal_metres}, "strict_valid_study_range_outer_boundary": {"edges": strict_outer_edges, "metres": strict_outer_metres}}}
    keys = np.concatenate(boundary_keys)
    codes = np.concatenate(boundary_codes)
    comps = np.concatenate(boundary_components)
    owner = np.concatenate(boundary_parts)
    order = np.argsort(keys, kind="stable")
    keys, codes, comps, owner = keys[order], codes[order], comps[order], owner[order]
    if len(keys) > 1 and np.any(keys[1:] == keys[:-1]):
        raise EvaluationError("duplicate Core boundary pixel detected after overlap validation")
    external_keys = np.concatenate(outer_target_keys)
    external_metres = np.concatenate(outer_target_metres)
    external_positions = np.searchsorted(keys, external_keys)
    external_found = external_positions < len(keys)
    matched = np.zeros(len(external_keys), dtype=bool)
    matches = np.flatnonzero(external_found)
    matched[matches] = keys[external_positions[matches]] == external_keys[matches]
    strict_outer_edges += int((~matched).sum())
    strict_outer_metres += float(external_metres[~matched].sum())
    cross_edges = 0
    cross_metres = 0.0
    # All adjacency candidates lie on a Core perimeter; sorted lookup only visits
    # existing valid pixels and never treats diagonal contact as connected.
    for delta, axis in ((np.uint64(1), "row_step_m"), (np.uint64(1) << np.uint64(32), "column_step_m")):
        targets = keys + delta
        positions = np.searchsorted(keys, targets)
        found = positions < len(keys)
        source_indices = np.flatnonzero(found)
        source_indices = source_indices[keys[positions[source_indices]] == targets[source_indices]]
        for source in source_indices:
            target = int(positions[source])
            if owner[source] == owner[target]:
                continue
            if codes[source] == codes[target]:
                uf.union(int(comps[source]), int(comps[target]))
                continue
            # A class boundary exists only where the two valid labels differ.
            # Same-class neighbours are topology unions, never boundary edges.
            source_step = float(getattr(parts[int(owner[source])], axis))
            target_step = float(getattr(parts[int(owner[target])], axis))
            relative_difference = abs(source_step - target_step) / max(source_step, target_step)
            if relative_difference > 0.02:
                raise EvaluationError(f"abnormal shared-Core {axis}: {source_step} vs {target_step} m")
            cross_edges += 1
            cross_metres += (source_step + target_step) / 2.0
    local_codes = np.asarray(component_codes, dtype=np.int64)
    local_pixels = np.asarray(component_pixels, dtype=np.int64)
    local_area_m2 = np.asarray(component_area_m2, dtype=np.float64)
    roots = np.fromiter((uf.find(index) for index in range(len(component_codes))), dtype=np.int64, count=len(component_codes))
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    global_pixels = np.bincount(inverse, weights=local_pixels).astype(np.int64)
    global_area_m2 = np.bincount(inverse, weights=local_area_m2)
    global_codes = local_codes[np.searchsorted(np.arange(len(component_codes)), unique_roots)]
    # A union only occurs across equal labels, but preserve an explicit guard for audit integrity.
    for group, root in enumerate(unique_roots):
        if np.any(local_codes[roots == root] != global_codes[group]):
            raise EvaluationError("union-find merged unlike classes")
    areas = global_area_m2.astype(np.float64)
    per_class: list[dict[str, Any]] = []
    dynamic_count = 0
    dynamic_area = 0.0
    curve: list[dict[str, Any]] = []
    for code in sorted(int(value) for value in np.unique(global_codes)):
        class_areas = areas[global_codes == code]
        mmu_value = mmu[code]
        fragment = class_areas < mmu_value
        count = int(class_areas.size)
        frag_count = int(fragment.sum())
        frag_area = float(class_areas[fragment].sum())
        dynamic_count += frag_count
        dynamic_area += frag_area
        per_class.append({"class_code": code, "components": count, "component_area_m2": float(class_areas.sum()), "dynamic_mmu_m2": mmu_value, "dynamic_fragments": frag_count, "dynamic_fragment_area_m2": frag_area,
                          "fixed_area_curve": [{"lt_m2": threshold, "components": int((class_areas < threshold).sum()), "component_area_m2": float(class_areas[class_areas < threshold].sum())} for threshold in CURVE_M2],
                          "component_area_bins": [{"bin_m2": name, "components": int(((class_areas >= low) & (class_areas < high)).sum()), "component_area_m2": float(class_areas[(class_areas >= low) & (class_areas < high)].sum())} for name, low, high in (("[0,25)", 0.0, 25.0), ("[25,50)", 25.0, 50.0), ("[50,100)", 50.0, 100.0), ("[100,200)", 100.0, 200.0), ("[200,+inf)", 200.0, math.inf))],
                          "component_area_percentiles_m2": {str(percentile): float(np.percentile(class_areas, percentile)) for percentile in PERCENTILES}})
    for threshold in CURVE_M2:
        chosen = areas < threshold
        curve.append({"lt_m2": threshold, "components": int(chosen.sum()), "component_area_m2": float(areas[chosen].sum())})
    bins = (("[0,25)", 0.0, 25.0), ("[25,50)", 25.0, 50.0), ("[50,100)", 50.0, 100.0), ("[100,200)", 100.0, 200.0), ("[200,+inf)", 200.0, math.inf))
    component_area_bins = [{"bin_m2": name, "components": int(((areas >= low) & (areas < high)).sum()), "component_area_m2": float(areas[(areas >= low) & (areas < high)].sum())} for name, low, high in bins]
    percentiles = {str(percentile): float(np.percentile(areas, percentile)) for percentile in PERCENTILES}
    boundary = {"internal_cross_class_boundary": {"edges": internal_edges, "metres": internal_metres}, "cross_core_cross_class_boundary": {"edges": cross_edges, "metres": cross_metres}, "strict_valid_study_range_outer_boundary": {"edges": strict_outer_edges, "metres": strict_outer_metres}}
    boundary["total_cross_class_boundary"] = {"edges": internal_edges + cross_edges, "metres": internal_metres + cross_metres}
    return {"components_total": int(areas.size), "component_area_total_m2": float(areas.sum()), "per_class": per_class, "dynamic_fragments": {"count": dynamic_count, "area_m2": dynamic_area}, "curve": curve, "component_area_bins": component_area_bins, "area_percentiles_m2": percentiles,
            "boundary_definition": "cross-class boundaries compare two strict-valid labels; strict_valid_study_range_outer_boundary separately counts strict-valid to invalid/global-outside edges and is not included in total_cross_class_boundary", "boundary": boundary}


def _coverage(parts: list[Part], manifest: dict[str, Any]) -> dict[str, Any]:
    min_r = min(part.row0 for part in parts); max_r = max(part.row0 + part.shape[0] for part in parts)
    min_c = min(part.col0 for part in parts); max_c = max(part.col0 + part.shape[1] for part in parts)
    declared = manifest.get("global_window")
    if declared is not None:
        min_r, max_r, min_c, max_c = _global_window(declared, "manifest global_window")
    core_pixels = sum(part.shape[0] * part.shape[1] for part in parts)
    bbox_pixels = (max_r - min_r) * (max_c - min_c)
    valid_pixels = sum(int(part.valid.sum()) for part in parts)
    invalid_pixels = core_pixels - valid_pixels
    label_arrays = {"raw": "raw", "v3": "v3", "v31": "v31"}
    outside = {name: sum(int(np.count_nonzero(~part.valid & (getattr(part, field) >= 0))) for part in parts) for name, field in label_arrays.items()}
    invalid = {name: sum(int(np.count_nonzero(part.valid & (getattr(part, field) < 0))) for part in parts) for name, field in label_arrays.items()}
    # Core overlaps were rejected in preflight, so this zero is an explicit invariant.
    return {"core_parts": len(parts), "core_bbox_pixels": bbox_pixels, "core_footprint_pixels": core_pixels, "geometric_coverage_gap_pixels": bbox_pixels - core_pixels, "core_overlap_pixels": 0,
            "valid_pixels": valid_pixels,
            "strict_invalid_outside_range_pixel_count": invalid_pixels,
            "outside_valid_label_pixels": outside,
            "true_coverage_gap_invalid_label_inside_valid_pixels": invalid,
            "invalid_label_inside_valid_pixels": invalid,
            "part_physical_metrics": [{"part_id": part.part_id, "pixel_area_m2": part.pixel_area_m2, "row_step_m": part.row_step_m, "column_step_m": part.column_step_m} for part in parts]}


def _transitions(parts: list[Part], source_method: str, target_method: str) -> dict[str, Any]:
    counts: dict[tuple[int, int], int] = {}
    source_area: dict[int, int] = {}
    target_area: dict[int, int] = {}
    changed = 0
    for part in parts:
        left, right = getattr(part, source_method)[part.valid].astype(np.int64, copy=False), getattr(part, target_method)[part.valid].astype(np.int64, copy=False)
        if left.size == 0:
            continue
        encoded = (left.astype(np.uint64) << np.uint64(32)) | right.astype(np.uint64)
        values, number = np.unique(encoded, return_counts=True)
        for value, amount in zip(values, number):
            source, target = int(value >> np.uint64(32)), int(value & np.uint64(0xffffffff))
            counts[(source, target)] = counts.get((source, target), 0) + int(amount)
        for code, amount in zip(*np.unique(left, return_counts=True)):
            source_area[int(code)] = source_area.get(int(code), 0) + int(amount)
        for code, amount in zip(*np.unique(right, return_counts=True)):
            target_area[int(code)] = target_area.get(int(code), 0) + int(amount)
        changed += int(np.count_nonzero(left != right))
    # Physical pixel areas may legitimately differ between EPSG:3857 partitions.
    transition_area: dict[tuple[int, int], float] = {}
    source_m2: dict[int, float] = {}; target_m2: dict[int, float] = {}; changed_m2 = 0.0
    for part in parts:
        left, right = getattr(part, source_method)[part.valid].astype(np.int64, copy=False), getattr(part, target_method)[part.valid].astype(np.int64, copy=False)
        encoded = (left.astype(np.uint64) << np.uint64(32)) | right.astype(np.uint64)
        values, number = np.unique(encoded, return_counts=True)
        for value, amount in zip(values, number):
            pair = (int(value >> np.uint64(32)), int(value & np.uint64(0xffffffff)))
            transition_area[pair] = transition_area.get(pair, 0.0) + float(amount * part.pixel_area_m2)
        for code, amount in zip(*np.unique(left, return_counts=True)):
            source_m2[int(code)] = source_m2.get(int(code), 0.0) + float(amount * part.pixel_area_m2)
        for code, amount in zip(*np.unique(right, return_counts=True)):
            target_m2[int(code)] = target_m2.get(int(code), 0.0) + float(amount * part.pixel_area_m2)
        changed_m2 += float(np.count_nonzero(left != right) * part.pixel_area_m2)
    transitions = [{"from_class": source, "to_class": target, "pixels": amount, "area_m2": transition_area[(source, target)]} for (source, target), amount in sorted(counts.items())]
    drift = [{"class_code": code, f"{source_method}_area_m2": source_m2.get(code, 0.0), f"{target_method}_area_m2": target_m2.get(code, 0.0), "delta_m2": target_m2.get(code, 0.0) - source_m2.get(code, 0.0)} for code in sorted(set(source_m2) | set(target_m2))]
    return {"from_method": source_method, "to_method": target_method, "changed_pixels": changed, "changed_area_m2": changed_m2, "transitions": transitions, "per_class_area_drift": drift}


def _comparison(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    return {"components_delta": target["components_total"] - source["components_total"], "components_reduction": source["components_total"] - target["components_total"], "dynamic_fragments_delta": target["dynamic_fragments"]["count"] - source["dynamic_fragments"]["count"], "dynamic_fragment_area_delta_m2": target["dynamic_fragments"]["area_m2"] - source["dynamic_fragments"]["area_m2"], "cross_class_boundary_delta_m": target["boundary"]["total_cross_class_boundary"]["metres"] - source["boundary"]["total_cross_class_boundary"]["metres"]}


def _bin_comparison(raw: dict[str, Any], v3: dict[str, Any], v31: dict[str, Any]) -> list[dict[str, Any]]:
    by_method = [{item["bin_m2"]: item for item in method["component_area_bins"]} for method in (raw, v3, v31)]
    return [{"bin_m2": name, "raw_components": by_method[0][name]["components"], "v3_components": by_method[1][name]["components"], "v31_components": by_method[2][name]["components"],
             "raw_area_m2": by_method[0][name]["component_area_m2"], "v3_area_m2": by_method[1][name]["component_area_m2"], "v31_area_m2": by_method[2][name]["component_area_m2"],
             "raw_to_v3_component_reduction": by_method[0][name]["components"] - by_method[1][name]["components"], "v3_to_v31_component_reduction": by_method[1][name]["components"] - by_method[2][name]["components"], "raw_to_v31_component_reduction": by_method[0][name]["components"] - by_method[2][name]["components"]}
            for name in ("[0,25)", "[25,50)", "[50,100)", "[100,200)", "[200,+inf)")]


def evaluate(manifest_path: Path, output_dir: Path, resume: bool = False) -> dict[str, Any]:
    manifest_path = manifest_path.resolve(); output_dir = output_dir.resolve()
    manifest, parts, mmu, input_hashes = _prepare(manifest_path)
    code_hash = _sha256_path(Path(__file__).resolve())
    fingerprint = hashlib.sha256(_canonical({"schema": SCHEMA_VERSION, "code_sha256": code_hash, "inputs": input_hashes, "mmu": mmu}).encode()).hexdigest()
    result_path, audit_path = output_dir / "global_fragmentation.json", output_dir / "audit.json"
    if resume and result_path.is_file() and audit_path.is_file():
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if audit.get("input_fingerprint_sha256") == fingerprint and audit.get("output_sha256") == _sha256_path(result_path):
                return {"resumed": True, "result": json.loads(result_path.read_text(encoding="utf-8")), "audit": audit}
        except (OSError, json.JSONDecodeError):
            pass
    coverage = _coverage(parts, manifest)
    if any(coverage["invalid_label_inside_valid_pixels"][method] for method in METHODS):
        raise EvaluationError("valid pixels contain negative labels")
    raw, v3, v31 = (_evaluate_method(parts, method, mmu) for method in METHODS)
    result = {"schema_version": SCHEMA_VERSION, "manifest": str(manifest_path), "part_count": len(parts), "approved_dynamic_mmu_m2": {str(code): mmu[code] for code in sorted(mmu)}, "coverage": coverage,
              "methods": {"raw": raw, "v3": v3, "v31": v31},
              "transitions": {"raw_to_v3": _transitions(parts, "raw", "v3"), "v3_to_v31": _transitions(parts, "v3", "v31"), "raw_to_v31": _transitions(parts, "raw", "v31")},
              "comparisons": {"raw_to_v3": _comparison(raw, v3), "v3_to_v31": _comparison(v3, v31), "raw_to_v31": _comparison(raw, v31)},
              "component_area_bin_comparison": _bin_comparison(raw, v3, v31)}
    _atomic_json(result_path, result)
    audit = {"schema_version": SCHEMA_VERSION, "evaluator": str(Path(__file__).resolve()), "code_sha256": code_hash, "manifest_sha256": _sha256_path(manifest_path), "input_sha256": input_hashes, "input_fingerprint_sha256": fingerprint, "output": str(result_path), "output_sha256": _sha256_path(result_path), "deterministic_json": True, "resume_supported": True}
    _atomic_json(audit_path, audit)
    return {"resumed": False, "result": result, "audit": audit}


def _write_npz(path: Path, v3: np.ndarray, v31: np.ndarray, valid: np.ndarray, raw: np.ndarray | None = None) -> None:
    raw = v3 if raw is None else raw
    np.savez_compressed(path, raw=raw.astype(np.int16), v3=v3.astype(np.int16), v31=v31.astype(np.int16), valid=valid.astype(np.uint8))


def self_test(workdir: Path | None = None) -> None:
    root = Path(tempfile.mkdtemp(prefix="global_fragmentation_selftest_", dir=str(workdir) if workdir else None))
    try:
        # A/B share one same-class edge and must merge without a boundary.  D
        # shares one unlike-class edge with A, so it contributes exactly one
        # cross-Core boundary. C only touches A diagonally: class 7 remains two
        # components, proving strict 4-neighbour logic.
        a = root / "a.npz"; b = root / "b.npz"; c = root / "c.npz"; d = root / "d.npz"
        _write_npz(a, np.array([[1, 1], [2, 7]]), np.array([[1, 1], [2, 7]]), np.array([[0, 1], [1, 1]], bool), raw=np.array([[1, 1], [3, 7]]))
        _write_npz(b, np.array([[1]]), np.array([[1]]), np.ones((1, 1), bool))
        _write_npz(c, np.array([[7]]), np.array([[7]]), np.ones((1, 1), bool))
        _write_npz(d, np.array([[3]]), np.array([[3]]), np.ones((1, 1), bool))
        physical = {"pixel_area_m2": 25, "row_step_m": 5, "column_step_m": 5}
        base = {"crs": "EPSG:3857", "approved_dynamic_mmu_m2": {"1": 25, "2": 25, "3": 25, "4": 25, "7": 25}, "parts": [
            {"part_id": "A", "core_npz": "a.npz", "transform": [5, 0, 0, 0, -5, 10], "physical_metrics": physical},
            {"part_id": "B", "core_npz": "b.npz", "transform": [5, 0, 10, 0, -5, 10], "physical_metrics": physical},
            {"part_id": "C", "core_npz": "c.npz", "transform": [5, 0, 10, 0, -5, 0], "physical_metrics": physical},
            {"part_id": "D", "core_npz": "d.npz", "transform": [5, 0, 0, 0, -5, 0], "physical_metrics": physical},
        ]}
        manifest = root / "manifest.json"; _atomic_json(manifest, base)
        one, two = root / "one", root / "two"
        first = evaluate(manifest, one); second = evaluate(manifest, two)
        if _sha256_path(one / "global_fragmentation.json") != _sha256_path(two / "global_fragmentation.json"):
            raise AssertionError("deterministic JSON failed")
        rows = {item["class_code"]: item for item in first["result"]["methods"]["v3"]["per_class"]}
        if rows[1]["components"] != 1 or rows[7]["components"] != 2:
            raise AssertionError("cross-Core merge or diagonal rejection failed")
        if first["result"]["coverage"]["strict_invalid_outside_range_pixel_count"] != 1:
            raise AssertionError("strict-invalid outside-range pixels were not counted")
        if first["result"]["coverage"]["true_coverage_gap_invalid_label_inside_valid_pixels"] != {"raw": 0, "v3": 0, "v31": 0}:
            raise AssertionError("invalid labels inside strict-valid coverage were not counted")
        if first["result"]["methods"]["v3"]["boundary"]["cross_core_cross_class_boundary"]["edges"] != 1:
            raise AssertionError("same-class cross-Core union was incorrectly counted as a boundary")
        if first["result"]["methods"]["v3"]["boundary"]["strict_valid_study_range_outer_boundary"]["edges"] <= 0:
            raise AssertionError("strict-valid study-range outer boundary was not counted")
        if first["result"]["transitions"]["raw_to_v3"]["changed_pixels"] != 1:
            raise AssertionError("raw-to-v3 transition was not counted")
        if sum(item["components"] for item in first["result"]["methods"]["v3"]["component_area_bins"]) != first["result"]["methods"]["v3"]["components_total"]:
            raise AssertionError("mutually-exclusive area bins do not partition components")
        resumed = evaluate(manifest, one, resume=True)
        if not resumed["resumed"]:
            raise AssertionError("resume did not validate its fingerprint")
        variable_ground = dict(base); variable_ground["parts"] = [dict(item) for item in base["parts"]]
        variable_ground["parts"][2]["physical_metrics"] = {"pixel_area_m2": 24.5, "row_step_m": 4.95, "column_step_m": 4.95}
        _atomic_json(root / "variable_ground.json", variable_ground)
        if evaluate(root / "variable_ground.json", root / "variable_ground")["result"]["methods"]["v3"]["components_total"] != first["result"]["methods"]["v3"]["components_total"]:
            raise AssertionError("legitimate variable EPSG:3857 ground metrics changed topology")
        # Exercise the actual full-runner layout: run_manifest -> per-part
        # audit.json + separate Core NPY files, rather than the convenience NPZ.
        runner_parts = []
        arrays = [("A", np.array([[1, 1], [3, 7]]), np.array([[1, 1], [2, 7]]), np.array([[0, 1], [1, 1]], bool), [0, 2, 0, 2], [5, 0, 0, 0, -5, 10]),
                  ("B", np.array([[1]]), np.array([[1]]), np.ones((1, 1), bool), [0, 1, 2, 3], [5, 0, 10, 0, -5, 10]),
                  ("C", np.array([[7]]), np.array([[7]]), np.ones((1, 1), bool), [2, 3, 2, 3], [5, 0, 10, 0, -5, 0]),
                  ("D", np.array([[3]]), np.array([[3]]), np.ones((1, 1), bool), [2, 3, 0, 1], [5, 0, 0, 0, -5, 0])]
        for name, raw_label, label, mask, global_window, affine in arrays:
            directory = root / "partitions" / name; directory.mkdir(parents=True)
            for filename, array in (("raw_core.npy", raw_label.astype(np.int16)), ("v3_core.npy", label.astype(np.int16)), ("v31a_core.npy", label.astype(np.int16)), ("valid_core.npy", mask.astype(bool))):
                np.save(directory / filename, array, allow_pickle=False)
            global_window = {"x0": global_window[2], "x1": global_window[3], "y0": global_window[0], "y1": global_window[1]}
            output_map = {key: {"path": str((directory / filename).relative_to(root)), "sha256": _sha256_path(directory / filename)} for key, filename in (("raw", "raw_core.npy"), ("v3", "v3_core.npy"), ("v31a", "v31a_core.npy"), ("valid", "valid_core.npy"))}
            audit_data = {"partition_id": name, "global_core_window": global_window, "core_transform": affine, "crs": "EPSG:3857", "physical_metrics": physical, "outputs": output_map}
            _atomic_json(directory / "audit.json", audit_data)
            runner_parts.append({"partition_id": name, "global_core_window": global_window, "audit": {"path": str((directory / "audit.json").relative_to(root)), "sha256": _sha256_path(directory / "audit.json")}, "outputs": output_map,
                                 "core_transform": affine, "crs": "EPSG:3857", "physical_metrics": physical})
        runner_manifest = root / "run_manifest.json"
        _atomic_json(runner_manifest, {"partitions": runner_parts, "approved_dynamic_mmu_m2": base["approved_dynamic_mmu_m2"], "processing_transform": [5, 0, 0, 0, -5, 10], "processing_crs": "EPSG:3857", "global_window": {"x0": 0, "x1": 3, "y0": 0, "y1": 3}})
        runner_result = evaluate(runner_manifest, root / "runner")
        if runner_result["result"]["methods"]["v3"]["per_class"] != first["result"]["methods"]["v3"]["per_class"]:
            raise AssertionError("full-runner manifest layout differs from direct layout")
        bad_area = dict(base); bad_area["parts"] = [dict(item) for item in base["parts"]]; bad_area["parts"][1]["transform"] = [10, 0, 10, 0, -5, 10]
        _atomic_json(root / "bad_area.json", bad_area)
        try:
            evaluate(root / "bad_area.json", root / "bad_area")
        except EvaluationError as exc:
            if "pixel size" not in str(exc): raise
        else:
            raise AssertionError("different pixel area was accepted")
        bad_window = dict(base); bad_window["parts"] = [dict(item) for item in base["parts"]]; bad_window["parts"][0]["core_window"] = [0, 3, 0, 2]
        _atomic_json(root / "bad_window.json", bad_window)
        try:
            evaluate(root / "bad_window.json", root / "bad_window")
        except EvaluationError as exc:
            if "core_window" not in str(exc): raise
        else:
            raise AssertionError("invalid Core window was accepted")
    finally:
        shutil.rmtree(root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, help="full runner 140-Core manifest")
    parser.add_argument("--output-dir", type=Path, help="directory for global_fragmentation.json and audit.json")
    parser.add_argument("--resume", action="store_true", help="reuse only a SHA-matching complete output")
    parser.add_argument("--self-test", action="store_true", help="run synthetic cross-Core, gap, window, area, resume tests")
    args = parser.parse_args(argv)
    if args.self_test:
        if args.manifest or args.output_dir:
            parser.error("--self-test is standalone")
        self_test()
        print("self-test: PASS")
        return 0
    if not args.manifest or not args.output_dir:
        parser.error("--manifest and --output-dir are required unless --self-test")
    try:
        answer = evaluate(args.manifest, args.output_dir, resume=args.resume)
    except EvaluationError as exc:
        print(f"evaluation rejected: {exc}", file=sys.stderr)
        return 2
    print(f"{'resumed' if answer['resumed'] else 'completed'}: {args.output_dir / 'global_fragmentation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
