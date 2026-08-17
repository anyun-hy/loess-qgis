"""Run a bounded, read-only A/B experiment for semantic-mask fragmentation.

The script consumes already committed partition mask/confidence rasters.  It
selects representative cells inside a manual-label/common-raster intersection,
loads each cell with a surrounding pixel buffer, evaluates multiple physical
area thresholds, and writes only new experiment artifacts below ``output-dir``.
It never modifies the source Run.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from affine import Affine
import fiona
import numpy as np
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.merge import merge
from rasterio.warp import transform_bounds, transform_geom
from scipy import ndimage
import shapely
from shapely.geometry import box, mapping, shape

from deployment_config import CLASS_NAMES, CLASS_ORDER
from small_component_regularizer import (
    EIGHT_CONNECTED,
    SmallComponentPolicy,
    physical_pixel_area_m2,
    regularize_small_components,
)


PARTITION_PATTERN = re.compile(r"partition_(\d+)_(\d+)_mask\.tif$")
PROTECTED_CODES = frozenset({61, 62, 71})
CLASS_AWARE_PROTECTED_SOURCE_CODES = frozenset({12, 33, 61, 62, 71})
CLASS_AWARE_DISALLOWED_TARGET_CODES = frozenset(
    {12, 21, 32, 33, 51, 52, 53, 54, 61, 62, 71}
)
SEMANTIC_COMPATIBLE_TARGET_CODES = {
    13: frozenset({21, 31, 32, 43}),
    21: frozenset({13}),
    31: frozenset({13, 32, 43}),
    32: frozenset({31, 43}),
    43: frozenset({13, 31, 32}),
    51: frozenset({52, 53, 54}),
    52: frozenset({51, 53, 54}),
    53: frozenset({51, 52, 54}),
    54: frozenset({51, 52, 53}),
}
SEMANTIC_OPTIMIZED_MAXIMUM_CONFIDENCE = {
    "semantic_optimized_200_balanced": 0.60,
    "semantic_optimized_200_strong": 0.70,
    "semantic_optimized_200_bounded": 0.65,
}
DEFAULT_THRESHOLDS_M2 = (25.0, 50.0, 100.0, 200.0)
FRAGMENT_BINS_M2 = (1.0, 10.0, 25.0, 50.0, 100.0, 200.0, 1000.0)
CLASS_AWARE_POLICIES_M2 = {
    "class_aware_verified_candidate": {
        12: 0.0,
        13: 25.0,
        21: 1.0,
        31: 25.0,
        32: 2.0,
        33: 0.0,
        43: 25.0,
        51: 1.0,
        52: 1.0,
        53: 1.0,
        54: 1.0,
        61: 0.0,
        62: 0.0,
        71: 0.0,
    },
    "class_aware_strict": {
        12: 0.0,
        13: 40.0,
        21: 2.0,
        31: 40.0,
        32: 4.0,
        33: 0.0,
        43: 40.0,
        51: 1.0,
        52: 4.0,
        53: 2.0,
        54: 4.0,
        61: 0.0,
        62: 0.0,
        71: 0.0,
    },
    "class_aware_safe": {
        12: 0.0,
        13: 100.0,
        21: 3.0,
        31: 100.0,
        32: 8.0,
        33: 0.0,
        43: 100.0,
        51: 2.0,
        52: 8.0,
        53: 5.0,
        54: 8.0,
        61: 0.0,
        62: 0.0,
        71: 0.0,
    },
    "class_aware_conservative": {
        12: 0.0,
        13: 100.0,
        21: 10.0,
        31: 100.0,
        32: 15.0,
        33: 0.0,
        43: 100.0,
        51: 5.0,
        52: 15.0,
        53: 5.0,
        54: 20.0,
        61: 0.0,
        62: 0.0,
        71: 0.0,
    },
    "class_aware_balanced": {
        12: 0.0,
        13: 100.0,
        21: 15.0,
        31: 100.0,
        32: 20.0,
        33: 0.0,
        43: 100.0,
        51: 10.0,
        52: 20.0,
        53: 10.0,
        54: 25.0,
        61: 0.0,
        62: 0.0,
        71: 0.0,
    },
    "semantic_optimized_200_balanced": {
        12: 0.0,
        13: 200.0,
        21: 50.0,
        31: 200.0,
        32: 75.0,
        33: 0.0,
        43: 200.0,
        51: 25.0,
        52: 50.0,
        53: 25.0,
        54: 50.0,
        61: 0.0,
        62: 0.0,
        71: 0.0,
    },
    "semantic_optimized_200_strong": {
        12: 0.0,
        13: 200.0,
        21: 200.0,
        31: 200.0,
        32: 200.0,
        33: 0.0,
        43: 200.0,
        51: 200.0,
        52: 200.0,
        53: 200.0,
        54: 200.0,
        61: 0.0,
        62: 0.0,
        71: 0.0,
    },
    "semantic_optimized_200_bounded": {
        code: (0.0 if code in CLASS_AWARE_PROTECTED_SOURCE_CODES else 200.0)
        for code in CLASS_ORDER
    },
}


class FragmentationExperimentError(RuntimeError):
    pass


def emit(event: str, **values: Any) -> None:
    print(
        json.dumps({"event": event, **values}, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_partition(path: Path) -> tuple[int, int]:
    match = PARTITION_PATTERN.match(path.name)
    if match is None:
        raise FragmentationExperimentError(f"unexpected mask name: {path}")
    return int(match.group(1)), int(match.group(2))


def _confidence_path(mask_path: Path) -> Path:
    return mask_path.with_name(mask_path.name.replace("_mask.tif", "_confidence.tif"))


def _load_manual_features(
    path: Path,
    *,
    target_crs: rasterio.crs.CRS,
    target_bounds: rasterio.coords.BoundingBox,
    class_field: str,
) -> tuple[list[Any], list[int], dict[str, Any]]:
    target_box = box(*target_bounds)
    geometries: list[Any] = []
    class_codes: list[int] = []
    with fiona.open(path) as source:
        if class_field not in source.schema["properties"]:
            raise FragmentationExperimentError(
                f"manual label field is missing: {class_field}"
            )
        source_bbox = transform_bounds(
            target_crs,
            source.crs_wkt,
            *target_bounds,
            densify_pts=41,
        )
        for feature in source.filter(bbox=source_bbox):
            code = feature["properties"].get(class_field)
            if code is None or int(code) not in CLASS_ORDER:
                continue
            transformed = shape(
                transform_geom(source.crs_wkt, target_crs, feature["geometry"])
            )
            clipped = transformed.intersection(target_box)
            if clipped.is_empty or clipped.area <= 0:
                continue
            geometries.append(clipped)
            class_codes.append(int(code))
    if not geometries:
        raise FragmentationExperimentError("manual labels do not intersect the raster")
    coverage = shapely.union_all(geometries)
    return geometries, class_codes, {
        "feature_count": len(geometries),
        "bounds": [float(value) for value in coverage.bounds],
        "web_mercator_area_m2": float(coverage.area),
        "class_feature_counts": {
            str(code): int(count)
            for code, count in sorted(Counter(class_codes).items())
        },
    }


def _manual_cell_summary(
    cell: Any,
    tree: shapely.STRtree,
    geometries: Sequence[Any],
    class_codes: Sequence[int],
) -> dict[str, Any] | None:
    matches = [int(index) for index in tree.query(cell, predicate="intersects")]
    if not matches:
        return None
    areas: dict[int, float] = defaultdict(float)
    for index in matches:
        intersection = geometries[index].intersection(cell)
        if not intersection.is_empty:
            areas[int(class_codes[index])] += float(intersection.area)
    covered = float(sum(areas.values()))
    if covered <= 0:
        return None
    shares = [value / covered for value in areas.values() if value > 0]
    entropy = -sum(value * math.log(value) for value in shares)
    protected = sum(areas.get(code, 0.0) for code in PROTECTED_CODES)
    return {
        "manual_coverage_ratio": min(1.0, covered / float(cell.area)),
        "manual_entropy": float(entropy),
        "manual_distinct_class_count": len(areas),
        "manual_protected_share": float(protected / covered),
        "manual_class_area_web_mercator_m2": {
            str(code): float(value) for code, value in sorted(areas.items())
        },
    }


def _select_centers(
    mask_paths: Sequence[Path],
    manual_geometries: Sequence[Any],
    manual_codes: Sequence[int],
    *,
    sample_count: int,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], Path]]:
    path_map = {_parse_partition(path): path for path in mask_paths}
    tree = shapely.STRtree(manual_geometries)
    manual_bounds = shapely.union_all(manual_geometries).bounds
    candidates: list[dict[str, Any]] = []
    emit("partition_metadata_scan_started", mask_count=len(mask_paths))
    for position, path in enumerate(mask_paths, start=1):
        row, col = _parse_partition(path)
        if any((row + dr, col + dc) not in path_map for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
            continue
        with rasterio.open(path) as source:
            bounds = source.bounds
            if (
                bounds.right <= manual_bounds[0]
                or bounds.left >= manual_bounds[2]
                or bounds.top <= manual_bounds[1]
                or bounds.bottom >= manual_bounds[3]
            ):
                continue
            if source.width < 512 or source.height < 512:
                continue
            preview = source.read(
                1,
                out_shape=(min(64, source.height), min(64, source.width)),
                resampling=rasterio.enums.Resampling.nearest,
            )
            valid_preview_ratio = float(np.mean(preview >= 0))
            if valid_preview_ratio < 0.50:
                continue
            cell = box(*bounds)
            summary = _manual_cell_summary(
                cell, tree, manual_geometries, manual_codes
            )
            if summary is None or summary["manual_coverage_ratio"] < 0.50:
                continue
            candidates.append(
                {
                    "row": row,
                    "col": col,
                    "path": str(path),
                    "bounds": [float(value) for value in bounds],
                    "model_valid_preview_ratio": valid_preview_ratio,
                    **summary,
                }
            )
        if position % 2000 == 0:
            emit(
                "partition_metadata_scan_progress",
                current=position,
                total=len(mask_paths),
                candidate_count=len(candidates),
            )
    if not candidates:
        raise FragmentationExperimentError("no fully buffered partition intersects manual labels")
    selected: list[dict[str, Any]] = []

    def separated(candidate: Mapping[str, Any]) -> bool:
        return all(
            max(
                abs(int(candidate["row"]) - int(item["row"])),
                abs(int(candidate["col"]) - int(item["col"])),
            )
            >= 3
            for item in selected
        )

    rankings = [
        lambda item: (
            float(item["manual_entropy"]),
            int(item["manual_distinct_class_count"]),
            float(item["manual_coverage_ratio"]),
        ),
        lambda item: (
            float(item["manual_protected_share"]),
            float(item["manual_entropy"]),
            int(item["manual_distinct_class_count"]),
        ),
        lambda item: (
            int(item["manual_distinct_class_count"]),
            float(item["manual_entropy"]),
            float(item["manual_coverage_ratio"]),
        ),
    ]
    for ranking in rankings:
        for candidate in sorted(candidates, key=ranking, reverse=True):
            if separated(candidate):
                selected.append(candidate)
                break
        if len(selected) >= sample_count:
            break
    if len(selected) < sample_count:
        for candidate in sorted(
            candidates,
            key=lambda item: (
                float(item["manual_entropy"]),
                float(item["manual_coverage_ratio"]),
            ),
            reverse=True,
        ):
            if candidate not in selected and separated(candidate):
                selected.append(candidate)
            if len(selected) >= sample_count:
                break
    emit(
        "partition_centers_selected",
        candidate_count=len(candidates),
        centers=[f"{item['row']:05d}_{item['col']:05d}" for item in selected],
    )
    return selected[:sample_count], path_map


def _merge_buffered_cell(
    center: Mapping[str, Any],
    path_map: Mapping[tuple[int, int], Path],
    *,
    buffer_pixels: int,
) -> tuple[np.ndarray, np.ndarray, Affine, rasterio.crs.CRS, tuple[slice, slice], dict[str, Any]]:
    row = int(center["row"])
    col = int(center["col"])
    neighbor_keys = [(row + dr, col + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    mask_sources = [rasterio.open(path_map[key]) for key in neighbor_keys]
    confidence_sources = [rasterio.open(_confidence_path(path_map[key])) for key in neighbor_keys]
    try:
        center_source = mask_sources[neighbor_keys.index((row, col))]
        xres = abs(float(center_source.transform.a))
        yres = abs(float(center_source.transform.e))
        expanded_bounds = (
            float(center_source.bounds.left) - buffer_pixels * xres,
            float(center_source.bounds.bottom) - buffer_pixels * yres,
            float(center_source.bounds.right) + buffer_pixels * xres,
            float(center_source.bounds.top) + buffer_pixels * yres,
        )
        mask_data, merged_transform = merge(
            mask_sources,
            bounds=expanded_bounds,
            res=(xres, yres),
            nodata=-1,
            dtype="int16",
            method="first",
        )
        confidence_data, confidence_transform = merge(
            confidence_sources,
            bounds=expanded_bounds,
            res=(xres, yres),
            nodata=np.nan,
            dtype="float32",
            method="first",
        )
        if not np.allclose(tuple(merged_transform), tuple(confidence_transform)):
            raise FragmentationExperimentError("mask/confidence merge grids disagree")
        row_offset = int(
            round((float(merged_transform.f) - float(center_source.transform.f)) / yres)
        )
        col_offset = int(
            round((float(center_source.transform.c) - float(merged_transform.c)) / xres)
        )
        core = (
            slice(row_offset, row_offset + int(center_source.height)),
            slice(col_offset, col_offset + int(center_source.width)),
        )
        return (
            mask_data[0],
            confidence_data[0],
            merged_transform,
            center_source.crs,
            core,
            {
                "center_transform": list(center_source.transform)[:6],
                "center_width": int(center_source.width),
                "center_height": int(center_source.height),
                "center_bounds": [float(value) for value in center_source.bounds],
            },
        )
    finally:
        for source in (*mask_sources, *confidence_sources):
            source.close()


def _core_transform(merged_transform: Affine, core: tuple[slice, slice]) -> Affine:
    return merged_transform * Affine.translation(
        int(core[1].start), int(core[0].start)
    )


def _manual_raster(
    geometries: Sequence[Any],
    class_codes: Sequence[int],
    *,
    bounds: Sequence[float],
    transform: Affine,
    shape_hw: tuple[int, int],
) -> np.ndarray:
    cell = box(*bounds)
    tree = shapely.STRtree(geometries)
    records = []
    for raw_index in tree.query(cell, predicate="intersects"):
        index = int(raw_index)
        clipped = geometries[index].intersection(cell)
        if not clipped.is_empty and clipped.area > 0:
            records.append((mapping(clipped), int(class_codes[index])))
    return rasterize(
        records,
        out_shape=shape_hw,
        transform=transform,
        fill=-1,
        dtype="int16",
        all_touched=False,
    )


def _component_statistics(
    labels: np.ndarray,
    valid: np.ndarray,
    *,
    pixel_area_m2: float,
) -> dict[str, Any]:
    total_components = 0
    per_class: dict[str, Any] = {}
    total_under = Counter()
    for class_index, class_code in enumerate(CLASS_ORDER):
        indexed, count = ndimage.label(
            valid & (labels == class_index), structure=EIGHT_CONNECTED
        )
        sizes = np.bincount(indexed.ravel(), minlength=count + 1)[1:]
        areas = sizes.astype(np.float64) * float(pixel_area_m2)
        under = {
            str(int(limit)): int(np.count_nonzero(areas < limit))
            for limit in FRAGMENT_BINS_M2
        }
        for limit, value in under.items():
            total_under[limit] += value
        total_components += int(count)
        per_class[str(class_code)] = {
            "class_name": CLASS_NAMES[class_code],
            "component_count": int(count),
            "pixel_count": int(np.count_nonzero(valid & (labels == class_index))),
            "area_m2": float(np.count_nonzero(valid & (labels == class_index)) * pixel_area_m2),
            "component_count_below_m2": under,
        }
    return {
        "component_count": total_components,
        "component_count_below_m2": dict(sorted(total_under.items(), key=lambda item: float(item[0]))),
        "per_class": per_class,
    }


def _agreement(
    labels: np.ndarray,
    manual_codes: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    class_code_array = np.asarray(CLASS_ORDER, dtype=np.int16)[labels]
    selected = valid & np.isin(manual_codes, CLASS_ORDER)
    if not np.any(selected):
        return {"valid_pixel_count": 0, "accuracy": None, "mean_iou": None, "per_class_iou": {}}
    predicted = class_code_array[selected]
    expected = manual_codes[selected]
    per_class_iou: dict[str, float | None] = {}
    values = []
    for code in CLASS_ORDER:
        union = np.count_nonzero((predicted == code) | (expected == code))
        intersection = np.count_nonzero((predicted == code) & (expected == code))
        value = float(intersection / union) if union else None
        per_class_iou[str(code)] = value
        if value is not None:
            values.append(value)
    return {
        "valid_pixel_count": int(np.count_nonzero(selected)),
        "accuracy": float(np.mean(predicted == expected)),
        "mean_iou": float(np.mean(values)) if values else None,
        "per_class_iou": per_class_iou,
    }


def _write_code_raster(
    path: Path,
    labels: np.ndarray,
    valid: np.ndarray,
    *,
    transform: Affine,
    crs: rasterio.crs.CRS,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.full(labels.shape, -1, dtype=np.int16)
    values[valid] = np.asarray(CLASS_ORDER, dtype=np.int16)[labels[valid]]
    profile = {
        "driver": "GTiff",
        "height": values.shape[0],
        "width": values.shape[1],
        "count": 1,
        "dtype": "int16",
        "crs": crs,
        "transform": transform,
        "nodata": -1,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }
    temporary = path.with_name(f".{path.name}.tmp.tif")
    temporary.unlink(missing_ok=True)
    try:
        with rasterio.open(temporary, "w", **profile) as destination:
            destination.write(values, 1)
            destination.update_tags(
                experiment="small_component_regularization_ab",
                class_encoding="land_use_code",
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_polygon_gpkg(
    path: Path,
    labels: np.ndarray,
    valid: np.ndarray,
    *,
    transform: Affine,
    crs: rasterio.crs.CRS,
    candidate_id: str,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.full(labels.shape, -1, dtype=np.int16)
    values[valid] = np.asarray(CLASS_ORDER, dtype=np.int16)[labels[valid]]
    schema = {
        "geometry": "Polygon",
        "properties": {"class_code": "int", "candidate": "str:64"},
    }
    temporary = path.with_name(f".{path.stem}.tmp.gpkg")
    temporary.unlink(missing_ok=True)
    count = 0
    try:
        with fiona.open(
            temporary,
            "w",
            driver="GPKG",
            layer="semantic_polygons_experiment",
            schema=schema,
            crs_wkt=crs.to_wkt(),
        ) as destination:
            for geometry, raw_value in shapes(
                values,
                mask=valid.astype(np.uint8),
                transform=transform,
                connectivity=8,
            ):
                destination.write(
                    {
                        "geometry": geometry,
                        "properties": {
                            "class_code": int(raw_value),
                            "candidate": candidate_id,
                        },
                    }
                )
                count += 1
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def _threshold_mapping(threshold: float) -> dict[int, float]:
    return {
        code: (0.0 if code in PROTECTED_CODES else float(threshold))
        for code in CLASS_ORDER
    }


def _candidate_score(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> float:
    if not bool((candidate.get("semantic_safety") or {}).get("passed")):
        return -1_000_000_000.0
    before = max(1, int(baseline["fragmentation"]["component_count_below_m2"]["100"]))
    after = int(candidate["fragmentation"]["component_count_below_m2"]["100"])
    reduction = float(before - after) / float(before)
    baseline_accuracy = baseline["manual_agreement"]["accuracy"]
    candidate_accuracy = candidate["manual_agreement"]["accuracy"]
    accuracy_drop = (
        max(0.0, float(baseline_accuracy) - float(candidate_accuracy))
        if baseline_accuracy is not None and candidate_accuracy is not None
        else 0.0
    )
    return reduction - 10.0 * accuracy_drop


def _semantic_safety(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    maximum_class_area_change_percent: float = 3.0,
    maximum_absolute_class_change_pixels: int = 2500,
    maximum_accuracy_drop: float = 0.005,
) -> dict[str, Any]:
    changes: dict[str, float | None] = {}
    violations = []
    for code in CLASS_ORDER:
        key = str(code)
        before = int(baseline["fragmentation"]["per_class"][key]["pixel_count"])
        after = int(candidate["fragmentation"]["per_class"][key]["pixel_count"])
        change = (float(after - before) / float(before) * 100.0) if before else None
        changes[key] = change
        if (
            change is not None
            and abs(change) > float(maximum_class_area_change_percent)
            and abs(after - before) > int(maximum_absolute_class_change_pixels)
        ):
            violations.append(
                {
                    "kind": "class_area_change",
                    "class_code": code,
                    "change_percent": change,
                }
            )
        if code in PROTECTED_CODES and after != before:
            violations.append(
                {
                    "kind": "protected_class_changed",
                    "class_code": code,
                    "before": before,
                    "after": after,
                }
            )
    baseline_accuracy = baseline["manual_agreement"]["accuracy"]
    candidate_accuracy = candidate["manual_agreement"]["accuracy"]
    accuracy_drop = (
        float(baseline_accuracy) - float(candidate_accuracy)
        if baseline_accuracy is not None and candidate_accuracy is not None
        else None
    )
    if accuracy_drop is not None and accuracy_drop > float(maximum_accuracy_drop):
        violations.append(
            {"kind": "manual_accuracy_drop", "drop": accuracy_drop}
        )
    return {
        "passed": not violations,
        "maximum_class_area_change_percent": float(maximum_class_area_change_percent),
        "maximum_absolute_class_change_pixels": int(
            maximum_absolute_class_change_pixels
        ),
        "maximum_accuracy_drop": float(maximum_accuracy_drop),
        "class_area_change_percent": changes,
        "accuracy_drop": accuracy_drop,
        "violations": violations,
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).expanduser().resolve()
    mask_dir = run_dir / "fusion" / args.fusion_id / "raster_parts"
    mask_paths = sorted(mask_dir.glob("partition_*_mask.tif"))
    if not mask_paths:
        raise FragmentationExperimentError(f"no partition masks found: {mask_dir}")
    for path in mask_paths:
        if not _confidence_path(path).is_file():
            raise FragmentationExperimentError(f"confidence raster is missing: {path}")
    with rasterio.open(mask_paths[0]) as reference:
        raster_crs = reference.crs
    source_raster = Path(args.source_raster).expanduser().resolve()
    with rasterio.open(source_raster) as source:
        source_bounds = source.bounds
        if source.crs != raster_crs:
            raise FragmentationExperimentError("source and partition raster CRS disagree")
    manual_geometries, manual_codes, manual_summary = _load_manual_features(
        Path(args.manual_labels).expanduser().resolve(),
        target_crs=raster_crs,
        target_bounds=source_bounds,
        class_field=args.manual_class_field,
    )
    emit("manual_intersection_ready", **manual_summary)
    centers, path_map = _select_centers(
        mask_paths,
        manual_geometries,
        manual_codes,
        sample_count=int(args.sample_count),
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = tuple(float(value) for value in args.thresholds_m2)
    report: dict[str, Any] = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "fusion_id": args.fusion_id,
        "source_raster": str(source_raster),
        "manual_labels": str(Path(args.manual_labels).expanduser().resolve()),
        "manual_class_field": args.manual_class_field,
        "manual_intersection": manual_summary,
        "threshold_candidates_m2": list(thresholds),
        "class_aware_policies_m2": CLASS_AWARE_POLICIES_M2,
        "protected_class_codes": sorted(PROTECTED_CODES),
        "buffer_pixels": int(args.buffer_pixels),
        "panels": [],
    }
    for panel_number, center in enumerate(centers, start=1):
        panel_id = f"panel_{panel_number:02d}_p{int(center['row']):05d}_{int(center['col']):05d}"
        emit("panel_started", panel_id=panel_id)
        (
            buffered_labels,
            buffered_confidence,
            merged_transform,
            crs,
            core,
            grid,
        ) = _merge_buffered_cell(
            center, path_map, buffer_pixels=int(args.buffer_pixels)
        )
        buffered_valid = buffered_labels >= 0
        if np.any(buffered_valid & (buffered_labels >= len(CLASS_ORDER))):
            raise FragmentationExperimentError(f"mask class index is invalid: {panel_id}")
        core_labels = buffered_labels[core]
        core_valid = buffered_valid[core]
        core_budget_mask = np.zeros(buffered_labels.shape, dtype=bool)
        core_budget_mask[core] = True
        core_transform = _core_transform(merged_transform, core)
        pixel_area = physical_pixel_area_m2(
            merged_transform,
            crs,
            height=buffered_labels.shape[0],
            width=buffered_labels.shape[1],
        )
        manual = _manual_raster(
            manual_geometries,
            manual_codes,
            bounds=grid["center_bounds"],
            transform=core_transform,
            shape_hw=core_labels.shape,
        )
        panel_dir = output_dir / panel_id
        _write_code_raster(
            panel_dir / "baseline_mask_codes.tif",
            core_labels,
            core_valid,
            transform=core_transform,
            crs=crs,
        )
        manual_indices = np.full(manual.shape, -1, dtype=np.int16)
        for index, code in enumerate(CLASS_ORDER):
            manual_indices[manual == code] = index
        _write_code_raster(
            panel_dir / "manual_label_codes.tif",
            np.maximum(manual_indices, 0),
            manual_indices >= 0,
            transform=core_transform,
            crs=crs,
        )
        baseline = {
            "candidate_id": "baseline",
            "fragmentation": _component_statistics(
                core_labels, core_valid, pixel_area_m2=pixel_area
            ),
            "manual_agreement": _agreement(core_labels, manual, core_valid),
        }
        candidates = []
        candidate_arrays: dict[str, np.ndarray] = {}
        candidate_specs: list[tuple[str, str, Mapping[int, float], float, float | None]] = []
        if not bool(args.class_aware_only):
            for mode in ("structural_ceiling", "confidence_guarded"):
                for threshold in thresholds:
                    candidate_specs.append(
                        (
                            f"{mode}_t{int(threshold)}m2",
                            mode,
                            _threshold_mapping(threshold),
                            min(25.0, threshold),
                            None if mode == "structural_ceiling" else 0.70,
                        )
                    )
        for candidate_id, threshold_mapping in CLASS_AWARE_POLICIES_M2.items():
            semantic_optimized = candidate_id in SEMANTIC_OPTIMIZED_MAXIMUM_CONFIDENCE
            candidate_specs.append(
                (
                    candidate_id,
                    "class_aware_confidence_guarded",
                    threshold_mapping,
                    2.0,
                    (
                        SEMANTIC_OPTIMIZED_MAXIMUM_CONFIDENCE[candidate_id]
                        if semantic_optimized
                        else 0.70
                    ),
                )
            )
        for (
            candidate_id,
            mode,
            threshold_mapping,
            hard_absorb_below_m2,
            maximum_mean_confidence,
        ) in candidate_specs:
            class_aware = mode == "class_aware_confidence_guarded"
            semantic_optimized = (
                candidate_id in SEMANTIC_OPTIMIZED_MAXIMUM_CONFIDENCE
            )
            policy = SmallComponentPolicy(
                thresholds_m2=threshold_mapping,
                protected_class_codes=(
                    CLASS_AWARE_PROTECTED_SOURCE_CODES
                    if class_aware
                    else PROTECTED_CODES
                ),
                allow_protected_targets=False,
                disallowed_target_class_codes=(
                    CLASS_AWARE_DISALLOWED_TARGET_CODES
                    if class_aware and not semantic_optimized
                    else frozenset()
                ),
                compatible_target_class_codes=(
                    SEMANTIC_COMPATIBLE_TARGET_CODES
                    if semantic_optimized
                    else {}
                ),
                compatibility_bypass_below_m2=(2.0 if semantic_optimized else 0.0),
                maximum_source_class_loss_fraction=(
                    0.08
                    if candidate_id == "semantic_optimized_200_bounded"
                    else None
                ),
                maximum_target_class_gain_fraction=(
                    0.08
                    if candidate_id == "semantic_optimized_200_bounded"
                    else None
                ),
                minimum_remaining_class_area_m2=(
                    5.0
                    if candidate_id == "semantic_optimized_200_bounded"
                    else 0.0
                ),
                hard_absorb_below_m2=hard_absorb_below_m2,
                maximum_mean_confidence=maximum_mean_confidence,
                maximum_probability_drop=None,
                preserve_border_components=True,
                preserve_elongated_components=semantic_optimized,
                elongated_minimum_area_m2=10.0,
                elongated_minimum_aspect_ratio=6.0,
                elongated_maximum_mean_width_m=3.0,
            )
            cleaned, regularization = regularize_small_components(
                buffered_labels,
                class_codes=CLASS_ORDER,
                pixel_area_m2=pixel_area,
                policy=policy,
                valid_mask=buffered_valid,
                confidence=buffered_confidence,
                class_budget_mask=(
                    core_budget_mask
                    if semantic_optimized
                    else None
                ),
            )
            cleaned_core = cleaned[core]
            candidate = {
                "candidate_id": candidate_id,
                "mode": mode,
                "thresholds_m2": {
                    str(code): float(value)
                    for code, value in sorted(threshold_mapping.items())
                },
                "regularization": regularization,
                "fragmentation": _component_statistics(
                    cleaned_core, core_valid, pixel_area_m2=pixel_area
                ),
                "manual_agreement": _agreement(cleaned_core, manual, core_valid),
                "changed_core_pixel_count": int(
                    np.count_nonzero(core_valid & (cleaned_core != core_labels))
                ),
            }
            candidate["semantic_safety"] = _semantic_safety(candidate, baseline)
            candidate["score"] = _candidate_score(candidate, baseline)
            candidates.append(candidate)
            candidate_arrays[candidate_id] = cleaned_core.copy()
            _write_code_raster(
                panel_dir / f"{candidate_id}_mask_codes.tif",
                cleaned_core,
                core_valid,
                transform=core_transform,
                crs=crs,
            )
            emit(
                "candidate_finished",
                panel_id=panel_id,
                candidate_id=candidate_id,
                component_count=candidate["fragmentation"]["component_count"],
                below_100m2=candidate["fragmentation"]["component_count_below_m2"]["100"],
                accuracy=candidate["manual_agreement"]["accuracy"],
            )
        if args.selected_candidate_id:
            selected = [
                item
                for item in candidates
                if item["candidate_id"] == args.selected_candidate_id
            ]
            if len(selected) != 1:
                raise FragmentationExperimentError(
                    f"selected candidate is unavailable: {args.selected_candidate_id}"
                )
            best = selected[0]
            if not bool(best["semantic_safety"]["passed"]):
                raise FragmentationExperimentError(
                    f"selected candidate failed semantic safety: "
                    f"{panel_id}/{args.selected_candidate_id}"
                )
        else:
            passing = [
                item for item in candidates if item["semantic_safety"]["passed"]
            ]
            if not passing:
                raise FragmentationExperimentError(
                    f"no candidate passed semantic safety: {panel_id}"
                )
            best = max(passing, key=lambda item: float(item["score"]))
        best_id = str(best["candidate_id"])
        baseline_gpkg_count = _write_polygon_gpkg(
            panel_dir / "baseline_polygons.gpkg",
            core_labels,
            core_valid,
            transform=core_transform,
            crs=crs,
            candidate_id="baseline",
        )
        best_gpkg_count = _write_polygon_gpkg(
            panel_dir / "best_candidate_polygons.gpkg",
            candidate_arrays[best_id],
            core_valid,
            transform=core_transform,
            crs=crs,
            candidate_id=best_id,
        )
        panel_report = {
            "panel_id": panel_id,
            "center": center,
            "grid": grid,
            "pixel_area_m2": pixel_area,
            "baseline": baseline,
            "candidates": candidates,
            "provisional_best_candidate_id": best_id,
            "baseline_gpkg_feature_count": baseline_gpkg_count,
            "best_gpkg_feature_count": best_gpkg_count,
        }
        _atomic_json(panel_dir / "panel_report.json", panel_report)
        report["panels"].append(panel_report)
        emit(
            "panel_finished",
            panel_id=panel_id,
            provisional_best_candidate_id=best_id,
            baseline_feature_count=baseline_gpkg_count,
            best_feature_count=best_gpkg_count,
        )
    _atomic_json(output_dir / "fragmentation_ab_report.json", report)
    emit("fragmentation_ab_finished", output_dir=str(output_dir), panel_count=len(report["panels"]))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--fusion-id", default="l2_fusion_v1")
    parser.add_argument("--source-raster", required=True)
    parser.add_argument("--manual-labels", required=True)
    parser.add_argument("--manual-class-field", default="TDLYDM_CC")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--buffer-pixels", type=int, default=256)
    parser.add_argument(
        "--class-aware-only",
        action="store_true",
        help="Skip uniform-threshold candidates and run only class-aware policies",
    )
    parser.add_argument(
        "--selected-candidate-id",
        default="",
        help="Require one uniform candidate to pass and become every panel GPKG",
    )
    parser.add_argument(
        "--thresholds-m2",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS_M2),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
