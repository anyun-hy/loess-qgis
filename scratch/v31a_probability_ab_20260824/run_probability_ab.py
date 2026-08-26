#!/usr/bin/env python3
"""Isolated probability-backed V3 vs V3.1-A panel comparison.

This file is deliberately kept under ``scratch/``.  It accepts immutable panel
NPZs plus a manifest, writes only below ``--output-root``, and has no deployment
or remote side effects.  The primary comparison is raw probability argmax ->
V3 (A) -> V3.1-A (B) on the identical 2,200-pixel halo.

Input manifest schema (all paths relative to the manifest):
{
  "schema_version": 1, "class_codes": [12, ...],
  "panels": [{"panel_id": "p00012_00003", "npz": "...npz",
    "sha256": "<content sha256>", "transform": [a,b,c,d,e,f],
    "crs": "EPSG:3857", "core_window": [256,1944]}]
}
Each NPZ requires ``probabilities`` [14,H,W] and ``valid``/``valid_mask`` [H,W],
and may contain ``manual``/``manual_labels`` [H,W] (class codes or indices).
Spatial metadata can live in the manifest or NPZ. ``core_window`` defaults to
[256,1944], the stable scoring core requested for the six panels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import platform
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "inference_scripts"))
from deployment_config import CLASS_ORDER  # noqa: E402
from fragmentation_v3 import policy_snapshot as v3_policy_snapshot  # noqa: E402
from fragmentation_v3 import production_policy  # noqa: E402
from fragmentation_v31_candidate import apply_v31a_candidate, policy_snapshot as v31_policy_snapshot  # noqa: E402
from small_component_regularizer import physical_pixel_area_m2, regularize_small_components  # noqa: E402

SCHEMA_VERSION = 1
FOUR_CONNECTED = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
PROTECTED = frozenset({12, 33, 61, 62, 71})
DEFAULT_CORE = (256, 1944)


class ABError(RuntimeError):
    pass


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_bytes(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def atomic_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_tif(path: Path, values: np.ndarray, valid: np.ndarray, transform: Affine, crs: str, nodata: int = -1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values)
    output = array.astype(np.int16 if array.min(initial=0) >= -32768 and array.max(initial=0) <= 32767 else np.int32, copy=True)
    output[~valid] = nodata
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tif", prefix=f".{path.stem}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with rasterio.open(temporary, "w", driver="GTiff", width=output.shape[1], height=output.shape[0], count=1,
                           dtype=output.dtype, crs=crs, transform=transform, nodata=nodata,
                           compress="deflate") as dataset:
            dataset.write(output, 1)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _pixel_size(transform: Affine, crs: str, height: int, width: int) -> tuple[float, float, float]:
    area = physical_pixel_area_m2(transform, crs, height=height, width=width)
    affine_area = abs(transform.a * transform.e - transform.b * transform.d)
    if affine_area <= 0:
        raise ABError("affine has no positive determinant")
    scale = math.sqrt(area / affine_area)
    row_m = math.hypot(transform.b, transform.e) * scale
    col_m = math.hypot(transform.a, transform.d) * scale
    if not math.isclose(row_m * col_m, area, rel_tol=1e-7):
        # Candidate requires an exact product.  Preserve measured area and a
        # non-square row step, then derive the corresponding column step.
        col_m = area / row_m
    return float(area), float(row_m), float(col_m)


def _decode_labels(manual: np.ndarray, codes: list[int], encoding: str = "auto") -> np.ndarray:
    values = np.asarray(manual)
    result = np.full(values.shape, -1, dtype=np.int16)
    code_to_index = {code: index for index, code in enumerate(codes)}
    if encoding not in {"auto", "codes", "indices"}:
        raise ABError("manual_labels_encoding must be auto, codes, or indices")
    raw_known = (values >= 0) & (values < len(codes))
    # Auto is unambiguous for the real 14-class code rasters once a code such
    # as 21 is present.  Synthetic/index rasters must declare ``indices``.
    inferred = "codes" if np.any(np.isin(values, [code for code in codes if code >= len(codes)])) else "indices"
    if encoding == "codes" or (encoding == "auto" and inferred == "codes"):
        for code, index in code_to_index.items():
            result[values == code] = index
    else:
        result[raw_known] = values[raw_known].astype(np.int16)
    return result


def _load_panel(entry: dict[str, Any], manifest_dir: Path, class_codes: list[int], *, is_self_test: bool = False) -> dict[str, Any]:
    relative = entry.get("npz") or entry.get("path")
    if not isinstance(relative, str):
        raise ABError(f"{entry.get('panel_id', '<unknown>')}: missing npz path")
    path = (manifest_dir / relative).resolve()
    if not path.is_file():
        raise ABError(f"panel NPZ is missing: {path}")
    actual_sha = sha256_path(path)
    expected_sha = entry.get("sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ABError(f"{path}: manifest requires expected SHA-256")
    if expected_sha != actual_sha:
        raise ABError(f"{path}: SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as source:
        valid_key = "valid_mask" if "valid_mask" in source else "valid"
        manual_key = "manual_labels" if "manual_labels" in source else "manual"
        if "probabilities" not in source or valid_key not in source:
            raise ABError(f"{path}: NPZ requires probabilities and valid/valid_mask")
        probabilities = np.asarray(source["probabilities"], dtype=np.float32)
        valid = np.asarray(source[valid_key], dtype=bool)
        raw_manual = np.asarray(source[manual_key]) if manual_key in source else None
        edge = np.asarray(source["edge"], dtype=np.float32) if "edge" in source else np.zeros(valid.shape, dtype=np.float32)
        manual = (
            None
            if raw_manual is None or not np.any(raw_manual)
            else _decode_labels(raw_manual, class_codes, str(entry.get("manual_labels_encoding", "auto")))
        )
        npz_transform = np.asarray(source["transform"], dtype=np.float64).tolist() if "transform" in source else None
        npz_crs = str(np.asarray(source["crs"]).item()) if "crs" in source else None
        stored_pixel_area = float(np.asarray(source["pixel_area_m2"]).item()) if "pixel_area_m2" in source else None
    if probabilities.ndim != 3 or probabilities.shape[0] != len(class_codes) or probabilities.shape[1:] != valid.shape:
        raise ABError(f"{path}: probabilities must be [14,H,W] matching valid_mask")
    if probabilities.shape[1] < 3 or probabilities.shape[2] < 3 or not np.all(np.isfinite(probabilities[:, valid])):
        raise ABError(f"{path}: non-finite or too-small probability cube")
    if np.any(probabilities[:, valid] < 0) or np.any(probabilities[:, valid] > 1) or not np.allclose(probabilities[:, valid].sum(axis=0), 1.0, atol=1e-3, rtol=0):
        raise ABError(f"{path}: valid probabilities must lie in [0,1] and sum to one")
    transform_values = entry.get("transform") or npz_transform
    if not isinstance(transform_values, list) or len(transform_values) not in {6, 9}:
        raise ABError(f"{path}: manifest requires transform [a,b,c,d,e,f]")
    transform = Affine(*[float(value) for value in transform_values[:6]])
    crs = str(entry.get("crs") or npz_crs or "")
    if not crs:
        raise ABError(f"{path}: manifest requires crs")
    core = tuple(int(value) for value in entry.get("core_window", DEFAULT_CORE))
    if len(core) != 2 or not (0 <= core[0] < core[1] <= valid.shape[0] and core[1] <= valid.shape[1]):
        raise ABError(f"{path}: invalid core_window {core} for {valid.shape}")
    if not is_self_test and "self_test_fixture" in entry:
        raise ABError(f"{path}: self_test_fixture is forbidden in real manifests")
    required_context = 0 if is_self_test else 256
    if core[0] < required_context or valid.shape[0] - core[1] < required_context or valid.shape[1] - core[1] < required_context:
        raise ABError(f"{path}: stable core must retain at least 256 pixels of context on every side")
    core_mask = np.zeros(valid.shape, dtype=bool)
    core_mask[core[0]:core[1], core[0]:core[1]] = True
    core_mask &= valid
    if not np.any(core_mask):
        raise ABError(f"{path}: stable score core has no valid pixels")
    if edge.shape != valid.shape or not np.all(np.isfinite(edge)):
        raise ABError(f"{path}: edge raster must be finite and match valid mask")
    return {"panel_id": str(entry["panel_id"]), "path": path, "sha256": actual_sha, "probabilities": probabilities,
            "valid": valid, "manual": manual, "edge": edge, "transform": transform, "crs": crs, "core": core, "core_mask": core_mask,
            "entry_sha256": sha256_json(entry), "manual_labels_encoding": str(entry.get("manual_labels_encoding", "auto")),
            "stored_pixel_area_m2": stored_pixel_area}


def _components(labels: np.ndarray, valid: np.ndarray, codes: list[int], pixel_area: float, dynamic: dict[int, float] | None = None) -> dict[str, Any]:
    total = dynamic_total = 0
    per_class: dict[str, dict[str, Any]] = {}
    for index, code in enumerate(codes):
        _labeled, count = ndimage.label(valid & (labels == index), structure=FOUR_CONNECTED)
        sizes = np.bincount(_labeled.ravel())[1:] if count else np.empty(0, dtype=int)
        total += int(count)
        threshold = float((dynamic or {}).get(code, 200.0))
        dynamic_count = int(np.count_nonzero(sizes * pixel_area < threshold)) if threshold > 0 else 0
        dynamic_total += dynamic_count
        per_class[str(code)] = {"components_4_connected": int(count), "dynamic_fragments_4_connected": dynamic_count,
                                "pixel_count": int(np.count_nonzero(valid & (labels == index))), "area_m2": float(np.count_nonzero(valid & (labels == index)) * pixel_area)}
    return {"components_4_connected": total, "dynamic_fragments_4_connected": dynamic_total, "per_class": per_class}


def _boundaries(labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.zeros(labels.shape, dtype=bool)
    for dr, dc in ((0, 1), (1, 0)):
        left = (slice(None, -dr or None), slice(None, -dc or None))
        right = (slice(dr, None), slice(dc, None))
        different = valid[left] & valid[right] & (labels[left] != labels[right])
        result[left] |= different
        result[right] |= different
    return result & valid


def _boundary_f1(predicted: np.ndarray, expected: np.ndarray, valid: np.ndarray, row_m: float, col_m: float, tolerance_m: float) -> dict[str, Any]:
    pred, truth = _boundaries(predicted, valid), _boundaries(expected, valid)
    if not np.any(pred) and not np.any(truth):
        return {"f1": 1.0, "predicted_total": 0, "predicted_matched": 0, "truth_total": 0, "truth_matched": 0}
    if not np.any(pred) or not np.any(truth):
        return {"f1": 0.0, "predicted_total": int(pred.sum()), "predicted_matched": 0, "truth_total": int(truth.sum()), "truth_matched": 0}
    pred_match = np.count_nonzero(pred & (ndimage.distance_transform_edt(~truth, sampling=(row_m, col_m)) <= tolerance_m))
    truth_match = np.count_nonzero(truth & (ndimage.distance_transform_edt(~pred, sampling=(row_m, col_m)) <= tolerance_m))
    precision, recall = pred_match / np.count_nonzero(pred), truth_match / np.count_nonzero(truth)
    return {"f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
            "predicted_total": int(pred.sum()), "predicted_matched": int(pred_match),
            "truth_total": int(truth.sum()), "truth_matched": int(truth_match)}


def _semantic_metrics(labels: np.ndarray, manual: np.ndarray | None, score: np.ndarray, codes: list[int], row_m: float, col_m: float) -> dict[str, Any]:
    if manual is None:
        return {"available": False}
    selected = score & (manual >= 0)
    if not np.any(selected):
        return {"available": False, "reason": "no manual labels in stable core"}
    expected, predicted = manual[selected], labels[selected]
    confusion = np.bincount(expected * len(codes) + predicted, minlength=len(codes) ** 2).reshape(len(codes), len(codes))
    oa = float(np.mean(expected == predicted))
    per_class: dict[str, Any] = {}
    ious: list[float] = []
    for index, code in enumerate(codes):
        inter = int(np.count_nonzero((expected == index) & (predicted == index)))
        union = int(np.count_nonzero((expected == index) | (predicted == index)))
        value = None if union == 0 else inter / union
        per_class[str(code)] = {"iou": value, "manual_pixels": int(np.count_nonzero(expected == index)), "prediction_pixels": int(np.count_nonzero(predicted == index))}
        if value is not None:
            ious.append(float(value))
    boundary_2m = _boundary_f1(labels, manual, selected, row_m, col_m, 2.0)
    boundary_5m = _boundary_f1(labels, manual, selected, row_m, col_m, 5.0)
    return {"available": True, "valid_pixel_count": int(selected.sum()), "oa": oa, "macro_iou": float(np.mean(ious)) if ious else None,
            "per_class": per_class, "confusion_matrix": confusion.tolist(),
            "boundary_f1_2m": boundary_2m["f1"], "boundary_counts_2m": boundary_2m,
            "boundary_f1_5m": boundary_5m["f1"], "boundary_counts_5m": boundary_5m}


def _changed_semantic(source: np.ndarray, target: np.ndarray, manual: np.ndarray | None, score: np.ndarray) -> dict[str, Any]:
    if manual is None:
        return {"available": False}
    selected = score & (manual >= 0) & (source != target)
    if not np.any(selected):
        return {"available": True, "changed_manual_pixels": 0, "source_wrong_target_right": 0, "source_right_target_wrong": 0, "both_wrong_changed": 0}
    source_correct = source[selected] == manual[selected]
    target_correct = target[selected] == manual[selected]
    return {"available": True, "changed_manual_pixels": int(selected.sum()),
            "source_wrong_target_right": int(np.count_nonzero(~source_correct & target_correct)),
            "source_right_target_wrong": int(np.count_nonzero(source_correct & ~target_correct)),
            "both_wrong_changed": int(np.count_nonzero(~source_correct & ~target_correct))}


def _transitions(source: np.ndarray, target: np.ndarray, score: np.ndarray, codes: list[int]) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    changed = score & (source != target)
    counter = Counter(zip(source[changed].tolist(), target[changed].tolist()))
    rows = [{"from_code": codes[int(a)], "to_code": codes[int(b)], "pixel_count": int(n)} for (a, b), n in sorted(counter.items())]
    per_class: dict[int, dict[str, Any]] = {}
    for index, code in enumerate(codes):
        baseline = int(np.count_nonzero(score & (source == index)))
        result = int(np.count_nonzero(score & (target == index)))
        lost = int(np.count_nonzero(score & (source == index) & (target != index)))
        gained = int(np.count_nonzero(score & (source != index) & (target == index)))
        per_class[code] = {"class_code": code, "baseline_pixels": baseline, "result_pixels": result, "source_loss_pixels": lost, "target_gain_pixels": gained,
                           "net_pixel_drift": result - baseline, "drift_fraction": None if baseline == 0 else (result - baseline) / baseline}
    return rows, per_class


def _change_cases(source: np.ndarray, target: np.ndarray, probabilities: np.ndarray, edge: np.ndarray, manual: np.ndarray | None, valid: np.ndarray, codes: list[int], limit: int = 50) -> list[dict[str, Any]]:
    changed = valid & (source != target)
    labeled, count = ndimage.label(changed, structure=FOUR_CONNECTED)
    cases = []
    for component in range(1, count + 1):
        rows, cols = np.where(labeled == component)
        if not len(rows):
            continue
        transitions = Counter(zip(source[rows, cols].tolist(), target[rows, cols].tolist()))
        item = {"pixel_count": int(len(rows)), "row_min": int(rows.min()), "row_max": int(rows.max()), "col_min": int(cols.min()), "col_max": int(cols.max()),
                "mean_max_probability": float(probabilities[:, rows, cols].max(axis=0).mean()), "mean_edge": float(edge[rows, cols].mean()),
                "transitions": [{"from_code": codes[int(a)], "to_code": codes[int(b)], "pixels": int(n)} for (a, b), n in sorted(transitions.items())]}
        if manual is not None:
            selected = manual[rows, cols] >= 0
            item["manual_pixels"] = int(np.count_nonzero(selected))
            item["source_correct_pixels"] = int(np.count_nonzero(selected & (source[rows, cols] == manual[rows, cols])))
            item["target_correct_pixels"] = int(np.count_nonzero(selected & (target[rows, cols] == manual[rows, cols])))
        cases.append(item)
    return sorted(cases, key=lambda item: (-item["pixel_count"], item["row_min"], item["col_min"]))[:limit]


def _hard_gates(raw: np.ndarray, result: np.ndarray, valid: np.ndarray, score: np.ndarray, codes: list[int], method: str,
                topology_before: dict[str, Any], topology_after: dict[str, Any], core_before: dict[str, Any], core_after: dict[str, Any],
                per_class: dict[int, dict[str, Any]], policy_limit: float) -> dict[str, Any]:
    protected_bad = {str(code): int(np.count_nonzero(score & (raw == i) & (result != i))) for i, code in enumerate(codes) if code in PROTECTED}
    class_budget = {}
    for code, row in per_class.items():
        protected = code in PROTECTED
        source_fraction = 0.0 if protected else policy_limit
        target_fraction = 0.01 if protected and method == "v31a" else policy_limit
        source_limit = source_fraction * row["baseline_pixels"]
        target_limit = target_fraction * row["baseline_pixels"]
        class_budget[str(code)] = {
            "source_loss": row["source_loss_pixels"],
            "target_gain": row["target_gain_pixels"],
            "source_limit": source_limit,
            "target_limit": target_limit,
            "passed": row["source_loss_pixels"] <= source_limit + 1e-9 and row["target_gain_pixels"] <= target_limit + 1e-9,
        }
    component_increase = {
        str(code): topology_after["per_class"][str(code)]["components_4_connected"]
        - topology_before["per_class"][str(code)]["components_4_connected"]
        for code in codes
    }
    core_component_increase = {
        str(code): core_after["per_class"][str(code)]["components_4_connected"]
        - core_before["per_class"][str(code)]["components_4_connected"]
        for code in codes
    }
    return {"method": method, "single_label": bool(np.all((result[valid] >= 0) & (result[valid] < len(codes)))),
            "invalid_preserved": bool(np.array_equal(raw[~valid], result[~valid])), "protected_source_retention": all(value == 0 for value in protected_bad.values()),
            "protected_source_loss_pixels": protected_bad,
            "no_4_connected_component_increase_halo": topology_after["components_4_connected"] <= topology_before["components_4_connected"],
            "no_per_class_component_increase_halo": all(value <= 0 for value in component_increase.values()),
            "per_class_component_delta_halo": component_increase,
            "no_per_class_component_increase_core": all(value <= 0 for value in core_component_increase.values()),
            "per_class_component_delta_core": core_component_increase,
            "core_class_budget": class_budget,
            "passed": False}


def _quicklook(path: Path, edge: np.ndarray, v3: np.ndarray, v31: np.ndarray, manual: np.ndarray | None, score: np.ndarray) -> None:
    # Pillow is available in the pinned QGIS environment whereas an accidental
    # user-site matplotlib can be incomplete; keep this diagnostic dependency
    # minimal and deterministic.
    from PIL import Image, ImageDraw
    palette = np.asarray([(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120), (152, 223, 138), (255, 152, 150)], dtype=np.uint8)
    def render(image: np.ndarray) -> Image.Image:
        result = np.full((*image.shape, 3), 245, dtype=np.uint8)
        known = score & (image >= 0) & (image < len(palette))
        result[known] = palette[image[known]]
        item = Image.fromarray(result, mode="RGB")
        maximum = 800
        if max(item.size) > maximum:
            scale = maximum / max(item.size)
            item = item.resize((round(item.width * scale), round(item.height * scale)), Image.Resampling.NEAREST)
        return item
    edge_rgb = np.repeat(np.clip(edge * 255.0, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
    edge_rgb[~score] = 245
    edge_image = Image.fromarray(edge_rgb, mode="RGB")
    change_rgb = np.full((*v31.shape, 3), 245, dtype=np.uint8)
    change_rgb[score & (v3 == v31)] = (215, 215, 215)
    change_rgb[score & (v3 != v31)] = (255, 0, 255)
    change_image = Image.fromarray(change_rgb, mode="RGB")
    label_images = [("A V3", render(v3)), ("B V3.1-A", render(v31))]
    source_images = [("probability edge", edge_image), *label_images, ("A/B change", change_image)]
    if manual is not None:
        source_images.append(("manual", render(manual)))
    images = []
    maximum = 800
    for title, item in source_images:
        if max(item.size) > maximum:
            scale = maximum / max(item.size)
            item = item.resize((round(item.width * scale), round(item.height * scale)), Image.Resampling.NEAREST)
        framed = Image.new("RGB", (item.width, item.height + 24), (245, 245, 245))
        framed.paste(item, (0, 24)); ImageDraw.Draw(framed).text((6, 5), title, fill=(20, 20, 20))
        images.append(framed)
    canvas = Image.new("RGB", (sum(item.width for item in images), max(item.height for item in images)), (245, 245, 245))
    offset = 0
    for item in images:
        canvas.paste(item, (offset, 0)); offset += item.width
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".png", prefix=f".{path.stem}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        canvas.save(temporary, format="PNG", optimize=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _case_images(panel_dir: Path, cases: list[dict[str, Any]], edge: np.ndarray, v3: np.ndarray, v31: np.ndarray, manual: np.ndarray | None, limit: int = 20) -> None:
    from PIL import Image, ImageDraw
    palette = np.asarray([(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120), (152, 223, 138), (255, 152, 150)], dtype=np.uint8)
    output = panel_dir / "cases"; output.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(cases[:limit], start=1):
        pad = 24
        row0, row1 = max(0, int(case["row_min"]) - pad), min(v3.shape[0], int(case["row_max"]) + pad + 1)
        col0, col1 = max(0, int(case["col_min"]) - pad), min(v3.shape[1], int(case["col_max"]) + pad + 1)
        crop_v3, crop_v31 = v3[row0:row1, col0:col1], v31[row0:row1, col0:col1]
        edge_crop = np.repeat(np.clip(edge[row0:row1, col0:col1] * 255, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
        change = np.full((*crop_v3.shape, 3), 220, dtype=np.uint8); change[crop_v3 != crop_v31] = (255, 0, 255)
        def labels_image(values: np.ndarray) -> np.ndarray:
            image = np.full((*values.shape, 3), 245, dtype=np.uint8)
            known = (values >= 0) & (values < len(palette)); image[known] = palette[values[known]]
            return image
        panels = [("edge", edge_crop), ("A V3", labels_image(crop_v3)), ("B V3.1-A", labels_image(crop_v31)), ("change", change)]
        if manual is not None:
            panels.append(("manual", labels_image(manual[row0:row1, col0:col1])))
        rendered = []
        for title, values in panels:
            item = Image.fromarray(values, mode="RGB").resize((256, 256), Image.Resampling.NEAREST)
            framed = Image.new("RGB", (256, 280), (245, 245, 245)); framed.paste(item, (0, 24)); ImageDraw.Draw(framed).text((5, 5), title, fill=(20, 20, 20)); rendered.append(framed)
        canvas = Image.new("RGB", (sum(item.width for item in rendered), 280), (245, 245, 245))
        offset = 0
        for item in rendered:
            canvas.paste(item, (offset, 0)); offset += item.width
        canvas.save(output / f"case_{index:03d}_r{case['row_min']}_c{case['col_min']}.png", format="PNG", optimize=True)


def _code_sha() -> dict[str, str]:
    files = [REPO_ROOT / "inference_scripts" / "deployment_config.py", REPO_ROOT / "inference_scripts" / "fragmentation_v3.py",
             REPO_ROOT / "inference_scripts" / "small_component_regularizer.py", REPO_ROOT / "inference_scripts" / "fragmentation_v31_candidate" / "__init__.py",
             REPO_ROOT / "inference_scripts" / "fragmentation_v31_candidate" / "candidate.py", Path(__file__).resolve()]
    return {str(path.relative_to(REPO_ROOT)): sha256_path(path) for path in files}


def _execution_fingerprint(panel: dict[str, Any], full_audit: bool, diagnostic_none: bool) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "npz_sha256": panel["sha256"],
        "entry_sha256": panel["entry_sha256"],
        "core": list(panel["core"]),
        "transform": list(panel["transform"])[:6],
        "crs": panel["crs"],
        "manual_labels_encoding": panel["manual_labels_encoding"],
        "full_audit": bool(full_audit),
        "diagnostic_confidence_none": bool(diagnostic_none),
        "primary_confidence_semantics": "explicit_max_probability",
        "code_sha256": _code_sha(),
        "v3_policy_snapshot_sha256": sha256_json(v3_policy_snapshot()),
        "v31a_policy_snapshot_sha256": sha256_json(v31_policy_snapshot()),
    }
    return {"payload": payload, "sha256": sha256_json(payload)}


def _validate_resume_outputs(panel_dir: Path) -> None:
    manifest_path = panel_dir / "outputs_sha256.json"
    if not manifest_path.is_file():
        raise ABError(f"resume output manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest.get("files", {}).items():
        path = panel_dir / relative
        if not path.is_file() or sha256_path(path) != expected:
            raise ABError(f"resume output is missing or changed: {path}")


def _run_panel(panel: dict[str, Any], output_root: Path, full_audit: bool, diagnostic_none: bool, resume: bool) -> dict[str, Any]:
    panel_dir = output_root / "panels" / panel["panel_id"]
    final_audit = panel_dir / "audit.json"
    execution_fingerprint = _execution_fingerprint(panel, full_audit, diagnostic_none)
    if resume and final_audit.is_file():
        prior = json.loads(final_audit.read_text())
        saved_audit_sha = prior.pop("audit_sha256", None)
        if (
            saved_audit_sha == sha256_json(prior)
            and prior.get("execution_fingerprint_sha256") == execution_fingerprint["sha256"]
        ):
            _validate_resume_outputs(panel_dir)
            return prior["summary"]
        raise ABError(f"{panel['panel_id']}: resume fingerprint mismatch")
    probabilities, valid, transform, crs, core_mask = panel["probabilities"], panel["valid"], panel["transform"], panel["crs"], panel["core_mask"]
    raw = np.argmax(probabilities, axis=0).astype(np.int16)
    raw[~valid] = -1
    confidence = probabilities.max(axis=0).astype(np.float32)
    pixel_area, row_m, col_m = _pixel_size(transform, crs, *raw.shape)
    if panel["stored_pixel_area_m2"] is not None and not math.isclose(pixel_area, panel["stored_pixel_area_m2"], rel_tol=1e-9, abs_tol=1e-9):
        raise ABError(f"{panel['panel_id']}: stored and recomputed physical pixel area differ")
    v3, v3_audit = regularize_small_components(raw, class_codes=CLASS_ORDER, pixel_area_m2=pixel_area, policy=production_policy(), valid_mask=valid,
                                                confidence=confidence, class_budget_mask=core_mask)
    v31, v31_audit = apply_v31a_candidate(v3, class_codes=CLASS_ORDER, pixel_area_m2=pixel_area, pixel_size_m=(row_m, col_m), valid_mask=valid,
                                           probabilities=probabilities, confidence=confidence, baseline_kind="v3_cleaned", full_audit=full_audit)
    diagnostic = None
    if diagnostic_none:
        diagnostic, diagnostic_audit = apply_v31a_candidate(v3, class_codes=CLASS_ORDER, pixel_area_m2=pixel_area, pixel_size_m=(row_m, col_m), valid_mask=valid,
                                                             probabilities=probabilities, confidence=None, baseline_kind="v3_cleaned", full_audit=full_audit)
    else:
        diagnostic_audit = None
    # Reconstruct candidate thresholds for measurement without altering its
    # proposal semantics or policy object.
    from fragmentation_v31_candidate import v31a_policy
    v31_dynamic = {int(code): float(item.dynamic_fragmentation_m2) for code, item in v31a_policy().class_policies.items()}
    topology = {"raw_halo": _components(raw, valid, CLASS_ORDER, pixel_area, v31_dynamic), "raw_core": _components(raw, core_mask, CLASS_ORDER, pixel_area, v31_dynamic),
                "v3_halo": _components(v3, valid, CLASS_ORDER, pixel_area, v31_dynamic), "v3_core": _components(v3, core_mask, CLASS_ORDER, pixel_area, v31_dynamic),
                "v31_halo": _components(v31, valid, CLASS_ORDER, pixel_area, v31_dynamic), "v31_core": _components(v31, core_mask, CLASS_ORDER, pixel_area, v31_dynamic)}
    semantic = {"raw": _semantic_metrics(raw, panel["manual"], core_mask, CLASS_ORDER, row_m, col_m), "v3": _semantic_metrics(v3, panel["manual"], core_mask, CLASS_ORDER, row_m, col_m),
                "v31": _semantic_metrics(v31, panel["manual"], core_mask, CLASS_ORDER, row_m, col_m),
                "v31_changed": _changed_semantic(v3, v31, panel["manual"], core_mask)}
    transitions, per_v3 = _transitions(raw, v3, core_mask, CLASS_ORDER)
    transitions31, per_v31 = _transitions(v3, v31, core_mask, CLASS_ORDER)
    gates_v3 = _hard_gates(raw, v3, valid, core_mask, CLASS_ORDER, "v3", topology["raw_halo"], topology["v3_halo"],
                           topology["raw_core"], topology["v3_core"], per_v3, 0.08)
    gates_v31 = _hard_gates(v3, v31, valid, core_mask, CLASS_ORDER, "v31a", topology["v3_halo"], topology["v31_halo"],
                            topology["v3_core"], topology["v31_core"], per_v31, 0.02)
    for gates in (gates_v3, gates_v31):
        gates["passed"] = all((gates["single_label"], gates["invalid_preserved"], gates["protected_source_retention"], gates["no_4_connected_component_increase_halo"],
                               gates["no_per_class_component_increase_halo"], gates["no_per_class_component_increase_core"],
                               all(item["passed"] for item in gates["core_class_budget"].values())))
    gates_v31["full_audit"] = bool(v31_audit.get("full_audit"))
    gates_v31["audit_not_truncated"] = gates_v31["full_audit"] and not bool(v31_audit.get("audit_truncated"))
    gates_v31["final_topology_rollback_zero"] = int(v31_audit.get("final_topology_rollback", 0)) == 0
    gates_v31["passed"] = gates_v31["passed"] and gates_v31["audit_not_truncated"] and gates_v31["final_topology_rollback_zero"]
    raw_codes = np.asarray(CLASS_ORDER, dtype=np.int16)[np.maximum(raw, 0)]
    v3_codes = np.asarray(CLASS_ORDER, dtype=np.int16)[np.maximum(v3, 0)]
    v31_codes = np.asarray(CLASS_ORDER, dtype=np.int16)[np.maximum(v31, 0)]
    atomic_tif(panel_dir / "raw_argmax.tif", raw_codes, valid, transform, crs)
    atomic_tif(panel_dir / "v3.tif", v3_codes, valid, transform, crs)
    atomic_tif(panel_dir / "v31a.tif", v31_codes, valid, transform, crs)
    atomic_tif(panel_dir / "change_raw_to_v3.tif", (raw * 100 + v3).astype(np.int32), valid, transform, crs, nodata=-1)
    atomic_tif(panel_dir / "change_v3_to_v31a.tif", (v3 * 100 + v31).astype(np.int32), valid, transform, crs, nodata=-1)
    if diagnostic is not None:
        atomic_tif(panel_dir / "v31a_confidence_none.tif", np.asarray(CLASS_ORDER, dtype=np.int16)[np.maximum(diagnostic, 0)], valid, transform, crs)
    rows = []
    for method, per_class in (("v3", per_v3), ("v31a", per_v31)):
        rows.extend({"method": method, **value} for value in per_class.values())
    atomic_csv(panel_dir / "per_class.csv", rows, ["method", "class_code", "baseline_pixels", "result_pixels", "source_loss_pixels", "target_gain_pixels", "net_pixel_drift", "drift_fraction"])
    atomic_csv(panel_dir / "transitions_v3.csv", transitions, ["from_code", "to_code", "pixel_count"])
    atomic_csv(panel_dir / "transitions_v31a.csv", transitions31, ["from_code", "to_code", "pixel_count"])
    proposal_rows = []
    for item in v31_audit.get("proposal_audit", []):
        proposal_rows.append({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in item.items()})
    proposal_fields = ["proposal_id", "kind", "target_class_code", "source_class_codes", "baseline_source_component_ids", "baseline_target_component_ids", "footprint_bbox",
                       "changed_pixels", "area_m2", "edge_distance_m", "path_length_m", "dynamic_fragment_reduction", "component_reduction",
                       "probability_support", "footprint_sha256", "decision", "reason", "evidence", "stable_rank_key"]
    atomic_csv(panel_dir / "proposals.csv", proposal_rows, proposal_fields)
    cases_v3 = _change_cases(raw, v3, probabilities, panel["edge"], panel["manual"], core_mask, CLASS_ORDER)
    cases_v31 = _change_cases(v3, v31, probabilities, panel["edge"], panel["manual"], core_mask, CLASS_ORDER)
    atomic_json(panel_dir / "top_change_cases_v3.json", cases_v3)
    atomic_json(panel_dir / "top_change_cases_v31a.json", cases_v31)
    _case_images(panel_dir, cases_v31, panel["edge"], v3, v31, panel["manual"])
    _quicklook(panel_dir / "quicklook.png", panel["edge"], v3, v31, panel["manual"], core_mask)
    summary = {"panel_id": panel["panel_id"], "raw_components_core": topology["raw_core"]["components_4_connected"], "v3_components_core": topology["v3_core"]["components_4_connected"],
               "v31a_components_core": topology["v31_core"]["components_4_connected"], "raw_dynamic_halo": topology["raw_halo"]["dynamic_fragments_4_connected"],
               "v3_dynamic_halo": topology["v3_halo"]["dynamic_fragments_4_connected"], "v31a_dynamic_halo": topology["v31_halo"]["dynamic_fragments_4_connected"],
               "v3_dynamic_core": topology["v3_core"]["dynamic_fragments_4_connected"], "v31a_dynamic_core": topology["v31_core"]["dynamic_fragments_4_connected"],
               "v3_changed_core_pixels": int(np.count_nonzero(core_mask & (raw != v3))), "v31a_changed_core_pixels": int(np.count_nonzero(core_mask & (v3 != v31))),
               "v31a_proposals_generated": int(v31_audit.get("proposals_generated", 0)), "v31a_proposals_accepted": int(v31_audit.get("proposals_accepted", 0)),
               "v31a_final_topology_rollback": int(v31_audit.get("final_topology_rollback", 0)),
               "v3_oa": semantic["v3"].get("oa"), "v31a_oa": semantic["v31"].get("oa"), "v3_macro_iou": semantic["v3"].get("macro_iou"), "v31a_macro_iou": semantic["v31"].get("macro_iou"),
               "v3_boundary_f1_2m": semantic["v3"].get("boundary_f1_2m"), "v31a_boundary_f1_2m": semantic["v31"].get("boundary_f1_2m"),
               "v3_boundary_f1_5m": semantic["v3"].get("boundary_f1_5m"), "v31a_boundary_f1_5m": semantic["v31"].get("boundary_f1_5m"),
               "changed_source_wrong_target_right": semantic["v31_changed"].get("source_wrong_target_right"),
               "changed_source_right_target_wrong": semantic["v31_changed"].get("source_right_target_wrong"),
               "v3_gates_passed": gates_v3["passed"], "v31a_gates_passed": gates_v31["passed"]}
    audit = {"schema_version": SCHEMA_VERSION, "execution_fingerprint_sha256": execution_fingerprint["sha256"], "execution_fingerprint": execution_fingerprint["payload"],
             "input": {"npz": str(panel["path"]), "npz_sha256": panel["sha256"], "class_codes": CLASS_ORDER, "probability_shape": list(probabilities.shape), "probability_dtype": str(probabilities.dtype), "valid_pixels": int(valid.sum()), "stable_score_core": list(panel["core"]), "stable_score_core_valid_pixels": int(core_mask.sum()), "transform": list(transform)[:6], "crs": crs, "stored_pixel_area_m2": panel["stored_pixel_area_m2"]},
             "code_sha256": _code_sha(), "v3_policy_snapshot": v3_policy_snapshot(), "v3_policy_snapshot_sha256": sha256_json(v3_policy_snapshot()), "v31a_policy_snapshot": v31_policy_snapshot(), "v31a_policy_snapshot_sha256": sha256_json(v31_policy_snapshot()),
             "physical_metrics": {"pixel_area_m2": pixel_area, "row_step_m": row_m, "column_step_m": col_m}, "v3_audit": v3_audit, "v31a_audit": v31_audit, "v31a_confidence_none_audit": diagnostic_audit,
             "topology": topology, "semantic": semantic, "hard_gates": {"v3": gates_v3, "v31a": gates_v31}, "summary": summary}
    audit["audit_sha256"] = sha256_json(audit)
    atomic_json(final_audit, audit)
    output_files = [path for path in panel_dir.rglob("*") if path.is_file() and path.name != "outputs_sha256.json"]
    atomic_json(panel_dir / "outputs_sha256.json", {"files": {str(path.relative_to(panel_dir)): sha256_path(path) for path in sorted(output_files)}})
    return summary


def _write_selftest_dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260824)
    probabilities = rng.random((14, 96, 96), dtype=np.float32); probabilities /= probabilities.sum(axis=0, keepdims=True)
    # Give one class a small isolated island that is safe for both algorithms.
    probabilities[:, 40:43, 40:43] = 0.0; probabilities[1, 40:43, 40:43] = 0.52; probabilities[2, 40:43, 40:43] = 0.48
    valid = np.ones((96, 96), dtype=bool); manual = np.argmax(probabilities, axis=0).astype(np.int16)
    npz = root / "synthetic_panel.npz"; np.savez_compressed(npz, probabilities=probabilities.astype(np.float16), valid_mask=valid, manual_labels=manual)
    manifest = {"schema_version": 1, "class_codes": CLASS_ORDER, "panels": [{"panel_id": "selftest", "npz": npz.name, "sha256": sha256_path(npz), "transform": [1, 0, 12310000, 0, -1, 4540000], "crs": "EPSG:3857", "core_window": [16, 80], "manual_labels_encoding": "indices"}]}
    path = root / "manifest.json"; atomic_json(path, manifest); return path


def _pooled_semantic(audits: list[dict[str, Any]], method: str) -> dict[str, Any]:
    confusion = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=np.int64)
    boundary = {"2m": Counter(), "5m": Counter()}
    available = 0
    for audit in audits:
        item = audit["semantic"][method]
        if not item.get("available"):
            continue
        available += 1
        confusion += np.asarray(item["confusion_matrix"], dtype=np.int64)
        for label in ("2m", "5m"):
            boundary[label].update(item[f"boundary_counts_{label}"])
    if not available:
        return {"available": False}
    total = int(confusion.sum())
    true_positive = np.diag(confusion)
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - true_positive
    ious = np.divide(true_positive, union, out=np.zeros_like(true_positive, dtype=np.float64), where=union > 0)
    result = {"available": True, "panel_count": available, "valid_pixel_count": total,
              "oa": float(true_positive.sum() / total) if total else None,
              "macro_iou": float(ious[union > 0].mean()) if np.any(union > 0) else None,
              "confusion_matrix": confusion.tolist(),
              "per_class_iou": {str(code): (float(ious[index]) if union[index] > 0 else None) for index, code in enumerate(CLASS_ORDER)}}
    for label, counts in boundary.items():
        pred_total, truth_total = counts["predicted_total"], counts["truth_total"]
        precision = counts["predicted_matched"] / pred_total if pred_total else 1.0
        recall = counts["truth_matched"] / truth_total if truth_total else 1.0
        result[f"boundary_f1_{label}"] = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        result[f"boundary_counts_{label}"] = dict(counts)
    return result


def _acceptance(audits: list[dict[str, Any]], panel_ids: set[str]) -> dict[str, Any]:
    required_panels = {f"P{value:02d}" for value in range(1, 7)}
    complete_panel_set = panel_ids == required_panels
    v3_components = sum(item["topology"]["v3_core"]["components_4_connected"] for item in audits)
    v31_components = sum(item["topology"]["v31_core"]["components_4_connected"] for item in audits)
    v3_dynamic = sum(item["topology"]["v3_core"]["dynamic_fragments_4_connected"] for item in audits)
    v31_dynamic = sum(item["topology"]["v31_core"]["dynamic_fragments_4_connected"] for item in audits)
    component_reduction = 0.0 if v3_components == 0 else (v3_components - v31_components) / v3_components
    dynamic_reduction = 0.0 if v3_dynamic == 0 else (v3_dynamic - v31_dynamic) / v3_dynamic
    v3_semantic, v31_semantic = _pooled_semantic(audits, "v3"), _pooled_semantic(audits, "v31")
    semantic_gates = {"available": v3_semantic.get("available") and v31_semantic.get("available")}
    if semantic_gates["available"]:
        semantic_gates.update({
            "oa_delta": v31_semantic["oa"] - v3_semantic["oa"],
            "oa_passed": v31_semantic["oa"] - v3_semantic["oa"] >= -0.001,
            "macro_iou_delta": v31_semantic["macro_iou"] - v3_semantic["macro_iou"],
            "macro_iou_passed": v31_semantic["macro_iou"] - v3_semantic["macro_iou"] >= -0.001,
            "boundary_f1_2m_delta": v31_semantic["boundary_f1_2m"] - v3_semantic["boundary_f1_2m"],
            "boundary_f1_2m_passed": v31_semantic["boundary_f1_2m"] - v3_semantic["boundary_f1_2m"] >= -0.002,
            "boundary_f1_5m_delta": v31_semantic["boundary_f1_5m"] - v3_semantic["boundary_f1_5m"],
            "boundary_f1_5m_passed": v31_semantic["boundary_f1_5m"] - v3_semantic["boundary_f1_5m"] >= -0.002,
        })
    integrity_passed = all(item["hard_gates"]["v31a"]["passed"] for item in audits)
    benefit_gates = {"component_reduction_fraction": component_reduction, "component_reduction_passed": component_reduction >= 0.02,
                     "dynamic_fragment_reduction_fraction": dynamic_reduction, "dynamic_fragment_reduction_passed": dynamic_reduction >= 0.05}
    semantic_passed = bool(semantic_gates.get("available")) and all(
        semantic_gates.get(key, False) for key in ("oa_passed", "macro_iou_passed", "boundary_f1_2m_passed", "boundary_f1_5m_passed")
    )
    passed = complete_panel_set and integrity_passed and benefit_gates["component_reduction_passed"] and benefit_gates["dynamic_fragment_reduction_passed"] and semantic_passed
    status = "passed" if passed else ("failed" if complete_panel_set else "incomplete")
    return {"status": status, "complete_six_panel_set": complete_panel_set, "required_panels": sorted(required_panels), "executed_panels": sorted(panel_ids), "integrity_passed": integrity_passed,
            "benefit_gates": benefit_gates, "semantic_gates": semantic_gates,
            "pooled_semantic": {"v3": v3_semantic, "v31a": v31_semantic},
            "determinism": {"status": "not_run", "required_before_production_discussion": True},
            "scope": "complete six-panel acceptance" if complete_panel_set else "partial exploratory smoke; not eligible for method acceptance"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="input manifest JSON")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "out")
    parser.add_argument("--panels", nargs="*", help="optional panel_id subset")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--full-audit", action="store_true", help="retain every V3.1-A proposal audit record")
    parser.add_argument("--diagnostic-confidence-none", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="create an isolated synthetic NPZ and verify runner contracts")
    args = parser.parse_args(argv)
    if not args.self_test and not args.full_audit:
        raise ABError("real panel runs require --full-audit")
    if args.self_test:
        test_root = args.output_root / "_selftest_input"
        shutil.rmtree(test_root, ignore_errors=True)
        manifest_path = _write_selftest_dataset(test_root)
    elif args.manifest:
        manifest_path = args.manifest.resolve()
    else:
        parser.error("--manifest is required unless --self-test is selected")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or list(manifest.get("class_codes", [])) != CLASS_ORDER:
        raise ABError("manifest must be schema_version=1 with canonical 14-class class_codes")
    wanted = None if not args.panels else set(args.panels)
    entries = [entry for entry in manifest.get("panels", []) if wanted is None or entry.get("panel_id") in wanted]
    if not entries or (wanted is not None and {entry.get("panel_id") for entry in entries} != wanted):
        raise ABError("requested panels are absent from manifest")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.resume and not args.self_test:
        raise ABError(f"output root is non-empty; use a new directory or --resume: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = [_run_panel(_load_panel(entry, manifest_path.parent, CLASS_ORDER, is_self_test=args.self_test), output_root, args.full_audit, args.diagnostic_confidence_none, args.resume) for entry in entries]
    fields = list(summaries[0])
    atomic_csv(output_root / "summary.csv", summaries, fields)
    audits = [json.loads((output_root / "panels" / summary["panel_id"] / "audit.json").read_text()) for summary in summaries]
    acceptance = _acceptance(audits, {summary["panel_id"] for summary in summaries})
    atomic_json(output_root / "acceptance.json", acceptance)
    atomic_json(output_root / "policy_snapshot.json", {"v3": v3_policy_snapshot(), "v31a": v31_policy_snapshot()})
    atomic_json(output_root / "source_code_sha256.json", _code_sha())
    atomic_json(output_root / "environment.json", {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                                                    "scipy": __import__("scipy").__version__, "rasterio": rasterio.__version__})
    atomic_csv(output_root / "inputs_sha256.csv", [{"panel_id": entry["panel_id"], "sha256": entry.get("sha256"), "path": entry.get("npz") or entry.get("path")} for entry in entries], ["panel_id", "sha256", "path"])
    report = {"schema_version": SCHEMA_VERSION, "manifest": str(manifest_path), "manifest_sha256": sha256_path(manifest_path), "panel_count": len(summaries), "full_audit": bool(args.full_audit), "diagnostic_confidence_none": bool(args.diagnostic_confidence_none), "summaries": summaries, "acceptance": acceptance}
    report["report_sha256"] = sha256_json(report); atomic_json(output_root / "summary.json", report)
    print(json.dumps({"execution_status": "completed", "acceptance_status": acceptance["status"], "output_root": str(output_root), "panel_count": len(summaries), "report_sha256": report["report_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ABError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
