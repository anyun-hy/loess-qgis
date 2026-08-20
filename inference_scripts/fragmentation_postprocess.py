"""Resumable V3 fragmentation repair for historical completed Fusion runs.

The stage consumes committed partition mask/confidence rasters and writes only
below ``run_dir/postprocess``.  Original inference, fitted vectors, confidence
rasters, and class workspaces are never overwritten.  Once every derived
artifact passes validation, ``--activate-review`` may atomically point the Run
manifest at the derived polygon layer. New v5 Runs apply V3 to probability
Halos before vectorization and never invoke this compatibility tool.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any, Iterable, Mapping, Sequence

from affine import Affine
import fiona
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.merge import merge
from scipy import ndimage
import shapely
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping, shape

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "qgis_plugins"
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from labeling_tool.core.run_spec import sha256_file

from deployment_config import CLASS_NAMES, CLASS_ORDER, load_json
from fragmentation_v3 import (
    DEFAULT_BUFFER_PIXELS,
    DEFAULT_MAX_WORKERS,
    FIT_VERSION,
    POLICY_ID,
    POLICY_VERSION,
    PROTECTED_SOURCE_CLASS_CODES,
    policy_snapshot,
    production_policy,
)
from partition_mosaic import build_vrt
from small_component_regularizer import (
    EIGHT_CONNECTED,
    physical_pixel_area_m2,
    regularize_small_components,
)


PARTITION_PATTERN = re.compile(r"^(partition_(\d+)_(\d+))_mask\.tif$")
LAYER_NAME = "semantic_polygons"
MANIFEST_NAME = "fragmentation_v3_manifest.json"
REPORT_NAME = "fragmentation_v3_report.json"


class FragmentationPostprocessError(RuntimeError):
    pass


FORMAL_SCHEMA = {
    "geometry": "MultiPolygon",
    "properties": {
        "run_id": "str:48",
        "result_stream_id": "str:96",
        "result_kind": "str:16",
        "model_id": "str:64",
        "fusion_profile_id": "str:64",
        "object_id": "str:64",
        "part_id": "str:96",
        "class_code": "int",
        "class_name": "str:64",
        "confidence_mean": "float",
        "confidence_std": "float",
        "model_version": "str:64",
        "source": "str:32",
        "fit_changed": "int",
        "fit_methods": "str:64",
        "fit_version": "str:40",
        "fit_status": "str:24",
        "origin_unit_ids": "str:254",
        "vertex_count_before": "int",
        "vertex_count_after": "int",
        "max_shift_px": "float",
        "mean_shift_px": "float",
        "area_change_ratio": "float",
        "created_at": "str:40",
    },
}


def emit(event: str, **values: Any) -> None:
    print(
        json.dumps(
            {"event": event, **values},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _json_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _partition_key(path: Path) -> tuple[int, int]:
    match = PARTITION_PATTERN.match(path.name)
    if match is None:
        raise FragmentationPostprocessError(f"unexpected partition mask: {path}")
    return int(match.group(2)), int(match.group(3))


def _partition_id(path: Path) -> str:
    match = PARTITION_PATTERN.match(path.name)
    if match is None:
        raise FragmentationPostprocessError(f"unexpected partition mask: {path}")
    return str(match.group(1))


def _confidence_path(mask_path: Path) -> Path:
    return mask_path.with_name(mask_path.name.replace("_mask.tif", "_confidence.tif"))


def _file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _source_inventory_fingerprint(mask_paths: Sequence[Path]) -> str:
    return _json_fingerprint(
        {
            "policy_version": POLICY_VERSION,
            "parts": [
                {
                    "mask": _file_signature(path),
                    "confidence": _file_signature(_confidence_path(path)),
                }
                for path in mask_paths
            ],
        }
    )


def _stream(spec: Mapping[str, Any], stream_id: str) -> Mapping[str, Any]:
    matches = [
        item for item in spec.get("streams") or []
        if str(item.get("stream_id") or "") == stream_id
    ]
    if len(matches) != 1:
        raise FragmentationPostprocessError(
            f"run_spec must contain exactly one stream {stream_id!r}"
        )
    stream = matches[0]
    if stream.get("kind") != "fusion":
        raise FragmentationPostprocessError("V3 production repair requires a Fusion stream")
    return stream


def derived_root(spec: Mapping[str, Any], stream: Mapping[str, Any]) -> Path:
    return (
        Path(str(spec["run_dir"])).resolve()
        / "postprocess"
        / POLICY_ID
        / "fusion"
        / str(stream["profile_id"])
    )


def derived_manifest_path(
    spec: Mapping[str, Any], stream: Mapping[str, Any]
) -> Path:
    return derived_root(spec, stream) / MANIFEST_NAME


def validated_derived_review(
    run_spec: Mapping[str, Any], stream: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Return a passed derived manifest only when its final GPKG is unchanged."""

    path = derived_manifest_path(run_spec, stream)
    try:
        manifest = load_json(path)
        output = Path(str(manifest["semantic_polygons"]))
        if (
            manifest.get("status") != "passed"
            or manifest.get("run_id") != run_spec.get("run_id")
            or manifest.get("stream_id") != stream.get("stream_id")
            or manifest.get("policy_version") != POLICY_VERSION
            or not output.is_file()
            or sha256_file(output) != manifest.get("semantic_polygons_sha256")
        ):
            return None
    except (KeyError, OSError, ValueError, TypeError):
        return None
    return dict(manifest)


def _load_buffered_partition(
    center_path: Path,
    path_map: Mapping[tuple[int, int], Path],
    *,
    buffer_pixels: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    Affine,
    rasterio.crs.CRS,
    tuple[slice, slice],
    dict[str, Any],
]:
    row, col = _partition_key(center_path)
    neighbor_paths = [
        path_map[(row + dr, col + dc)]
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (row + dr, col + dc) in path_map
    ]
    mask_sources = [rasterio.open(path) for path in neighbor_paths]
    confidence_sources = [
        rasterio.open(_confidence_path(path)) for path in neighbor_paths
    ]
    try:
        center_index = neighbor_paths.index(center_path)
        center = mask_sources[center_index]
        xres = abs(float(center.transform.a))
        yres = abs(float(center.transform.e))
        expanded_bounds = (
            float(center.bounds.left) - int(buffer_pixels) * xres,
            float(center.bounds.bottom) - int(buffer_pixels) * yres,
            float(center.bounds.right) + int(buffer_pixels) * xres,
            float(center.bounds.top) + int(buffer_pixels) * yres,
        )
        masks, transform = merge(
            mask_sources,
            bounds=expanded_bounds,
            res=(xres, yres),
            nodata=-1,
            dtype="int16",
            method="first",
        )
        confidence, confidence_transform = merge(
            confidence_sources,
            bounds=expanded_bounds,
            res=(xres, yres),
            nodata=np.nan,
            dtype="float32",
            method="first",
        )
        if not np.allclose(tuple(transform), tuple(confidence_transform)):
            raise FragmentationPostprocessError(
                f"mask/confidence grids disagree for {_partition_id(center_path)}"
            )
        row_offset = int(round((float(transform.f) - float(center.transform.f)) / yres))
        col_offset = int(round((float(center.transform.c) - float(transform.c)) / xres))
        core = (
            slice(row_offset, row_offset + int(center.height)),
            slice(col_offset, col_offset + int(center.width)),
        )
        metadata = {
            "profile": dict(center.profile),
            "transform": list(center.transform)[:6],
            "width": int(center.width),
            "height": int(center.height),
            "neighbor_paths": [str(path.resolve()) for path in neighbor_paths],
        }
        return masks[0], confidence[0], transform, center.crs, core, metadata
    finally:
        for source in (*mask_sources, *confidence_sources):
            source.close()


def _write_regularized_mask(
    path: Path,
    values: np.ndarray,
    *,
    profile: Mapping[str, Any],
) -> None:
    destination_profile = dict(profile)
    destination_profile.update(
        driver="GTiff",
        count=1,
        dtype="int16",
        nodata=-1,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.tif")
    temporary.unlink(missing_ok=True)
    try:
        with rasterio.open(temporary, "w", **destination_profile) as destination:
            destination.write(values.astype(np.int16, copy=False), 1)
            destination.update_tags(
                fragmentation_policy=POLICY_VERSION,
                class_encoding="zero_based_class_index",
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _partition_input_fingerprint(
    center_path: Path,
    path_map: Mapping[tuple[int, int], Path],
    *,
    buffer_pixels: int,
) -> tuple[str, list[dict[str, Any]]]:
    row, col = _partition_key(center_path)
    inputs: list[dict[str, Any]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            path = path_map.get((row + dr, col + dc))
            if path is None:
                continue
            inputs.append(_file_signature(path))
            inputs.append(_file_signature(_confidence_path(path)))
    payload = {
        "policy": policy_snapshot(),
        "buffer_pixels": int(buffer_pixels),
        "inputs": inputs,
    }
    return _json_fingerprint(payload), inputs


def _process_mask_partition(payload: Mapping[str, Any]) -> dict[str, Any]:
    center_path = Path(str(payload["center_path"]))
    output_path = Path(str(payload["output_path"]))
    report_path = Path(str(payload["report_path"]))
    buffer_pixels = int(payload["buffer_pixels"])
    path_map = {
        tuple(int(value) for value in key.split(",")): Path(str(value))
        for key, value in dict(payload["path_map"]).items()
    }
    fingerprint, inputs = _partition_input_fingerprint(
        center_path, path_map, buffer_pixels=buffer_pixels
    )
    if bool(payload.get("resume")) and output_path.is_file() and report_path.is_file():
        try:
            previous = load_json(report_path)
            if (
                previous.get("status") == "passed"
                and previous.get("input_fingerprint") == fingerprint
                and previous.get("output_sha256") == sha256_file(output_path)
            ):
                return {**dict(previous), "resumed": True}
        except (OSError, ValueError, TypeError):
            pass

    (
        labels,
        confidence,
        merged_transform,
        crs,
        core,
        metadata,
    ) = _load_buffered_partition(
        center_path, path_map, buffer_pixels=buffer_pixels
    )
    valid = labels >= 0
    if np.any(valid & (labels >= len(CLASS_ORDER))):
        raise FragmentationPostprocessError(
            f"invalid class index in {_partition_id(center_path)}"
        )
    core_valid = valid[core]
    core_before = labels[core].copy()
    budget_mask = np.zeros(labels.shape, dtype=bool)
    budget_mask[core] = core_valid
    pixel_area = physical_pixel_area_m2(
        merged_transform,
        crs,
        height=labels.shape[0],
        width=labels.shape[1],
    )
    cleaned, regularization = regularize_small_components(
        labels,
        class_codes=CLASS_ORDER,
        pixel_area_m2=pixel_area,
        policy=production_policy(),
        valid_mask=valid,
        confidence=confidence,
        class_budget_mask=budget_mask,
    )
    core_after = cleaned[core]
    output_values = np.full(core_after.shape, -1, dtype=np.int16)
    output_values[core_valid] = core_after[core_valid]
    _write_regularized_mask(output_path, output_values, profile=metadata["profile"])

    before_counts = np.bincount(
        core_before[core_valid], minlength=len(CLASS_ORDER)
    )
    after_counts = np.bincount(
        core_after[core_valid], minlength=len(CLASS_ORDER)
    )
    report = {
        "schema_version": 1,
        "status": "passed",
        "partition_id": _partition_id(center_path),
        "policy_version": POLICY_VERSION,
        "buffer_pixels": buffer_pixels,
        "pixel_area_m2": float(pixel_area),
        "input_fingerprint": fingerprint,
        "input_files": inputs,
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "valid_pixel_count": int(np.count_nonzero(core_valid)),
        "changed_pixel_count": int(
            np.count_nonzero(core_valid & (core_before != core_after))
        ),
        "class_pixel_count_before": {
            str(code): int(before_counts[index])
            for index, code in enumerate(CLASS_ORDER)
        },
        "class_pixel_count_after": {
            str(code): int(after_counts[index])
            for index, code in enumerate(CLASS_ORDER)
        },
        "regularization": regularization,
        "resumed": False,
    }
    _atomic_json(report_path, report)
    return report


def _run_bounded_processes(
    payloads: Sequence[Mapping[str, Any]],
    worker,
    *,
    workers: int,
    event_prefix: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total = len(payloads)
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        iterator = iter(payloads)
        active = set()
        for _ in range(min(max(1, int(workers)), total)):
            try:
                active.add(executor.submit(worker, next(iterator)))
            except StopIteration:
                break
        while active:
            completed, active = wait(active, return_when=FIRST_COMPLETED)
            for future in completed:
                result = future.result()
                results.append(result)
                emit(
                    f"{event_prefix}_progress",
                    current=len(results),
                    total=total,
                    partition_id=result.get("partition_id", ""),
                    resumed=bool(result.get("resumed")),
                )
                try:
                    active.add(executor.submit(worker, next(iterator)))
                except StopIteration:
                    pass
    return results


def _polygonal_parts(geometry: Any) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [item for item in geometry.geoms if not item.is_empty]
    if isinstance(geometry, GeometryCollection):
        parts: list[Polygon] = []
        for item in geometry.geoms:
            parts.extend(_polygonal_parts(item))
        return parts
    return []


def _valid_multipolygon(raw_geometry: Mapping[str, Any]) -> MultiPolygon | None:
    geometry = shape(raw_geometry)
    if geometry.is_empty:
        return None
    if not geometry.is_valid:
        geometry = shapely.make_valid(geometry)
    polygons = [item for item in _polygonal_parts(geometry) if item.area > 0]
    if not polygons:
        return None
    result = MultiPolygon(polygons)
    if not result.is_valid:
        repaired = shapely.make_valid(result)
        polygons = [item for item in _polygonal_parts(repaired) if item.area > 0]
        if not polygons:
            return None
        result = MultiPolygon(polygons)
    return result


def _vertex_count(geometry: MultiPolygon) -> int:
    return sum(
        len(polygon.exterior.coords)
        + sum(len(ring.coords) for ring in polygon.interiors)
        for polygon in geometry.geoms
    )


def _component_raster(
    labels: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    component_map = np.zeros(labels.shape, dtype=np.int32)
    class_codes = [0]
    next_id = 1
    for class_index, class_code in enumerate(CLASS_ORDER):
        local, count = ndimage.label(
            valid & (labels == class_index), structure=EIGHT_CONNECTED
        )
        if count <= 0:
            continue
        selected = local > 0
        component_map[selected] = local[selected].astype(np.int32) + next_id - 1
        class_codes.extend([int(class_code)] * int(count))
        next_id += int(count)
    if np.any(valid & (component_map == 0)):
        raise FragmentationPostprocessError("component raster left valid pixels unassigned")
    return component_map, np.asarray(class_codes, dtype=np.int16)


def _process_vector_partition(payload: Mapping[str, Any]) -> dict[str, Any]:
    mask_path = Path(str(payload["mask_path"]))
    confidence_path = Path(str(payload["confidence_path"]))
    output_path = Path(str(payload["output_path"]))
    report_path = Path(str(payload["report_path"]))
    partition_id = str(payload["partition_id"])
    input_fingerprint = _json_fingerprint(
        {
            "policy_version": POLICY_VERSION,
            "mask_sha256": sha256_file(mask_path),
            "confidence": _file_signature(confidence_path),
            "run_id": payload["run_id"],
            "stream_id": payload["stream_id"],
        }
    )
    if bool(payload.get("resume")) and output_path.is_file() and report_path.is_file():
        try:
            previous = load_json(report_path)
            if (
                previous.get("status") == "passed"
                and previous.get("input_fingerprint") == input_fingerprint
                and previous.get("output_sha256") == sha256_file(output_path)
            ):
                return {**dict(previous), "resumed": True}
        except (OSError, ValueError, TypeError):
            pass

    with rasterio.open(mask_path) as mask_source:
        labels = mask_source.read(1).astype(np.int16, copy=False)
        transform = mask_source.transform
        crs = mask_source.crs
    with rasterio.open(confidence_path) as confidence_source:
        confidence = confidence_source.read(1).astype(np.float32, copy=False)
        if (
            confidence.shape != labels.shape
            or confidence_source.transform != transform
            or confidence_source.crs != crs
        ):
            raise FragmentationPostprocessError(
                f"mask/confidence mismatch during vectorization: {partition_id}"
            )
    valid = labels >= 0
    if np.any(valid & (labels >= len(CLASS_ORDER))):
        raise FragmentationPostprocessError(
            f"invalid class index during vectorization: {partition_id}"
        )
    component_map, component_classes = _component_raster(labels, valid)
    component_count = len(component_classes) - 1
    pixel_counts = np.bincount(
        component_map[valid], minlength=component_count + 1
    ).astype(np.int64, copy=False)
    finite = valid & np.isfinite(confidence)
    confidence_sum = np.bincount(
        component_map[finite],
        weights=confidence[finite].astype(np.float64),
        minlength=component_count + 1,
    )
    confidence_square_sum = np.bincount(
        component_map[finite],
        weights=np.square(confidence[finite].astype(np.float64)),
        minlength=component_count + 1,
    )
    confidence_count = np.bincount(
        component_map[finite], minlength=component_count + 1
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.tmp.gpkg")
    temporary.unlink(missing_ok=True)
    feature_count = 0
    created_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        with fiona.open(
            temporary,
            "w",
            driver="GPKG",
            layer=LAYER_NAME,
            schema=FORMAL_SCHEMA,
            crs_wkt=crs.to_wkt(),
        ) as destination:
            for raw_geometry, raw_component_id in shapes(
                component_map,
                mask=valid.astype(np.uint8),
                transform=transform,
                connectivity=8,
            ):
                component_id = int(raw_component_id)
                if component_id <= 0 or component_id > component_count:
                    raise FragmentationPostprocessError(
                        f"invalid component id in {partition_id}: {component_id}"
                    )
                geometry = _valid_multipolygon(raw_geometry)
                if geometry is None or not geometry.is_valid or geometry.area <= 0:
                    raise FragmentationPostprocessError(
                        f"cannot repair component geometry in {partition_id}: {component_id}"
                    )
                count = int(confidence_count[component_id])
                mean = (
                    float(confidence_sum[component_id] / count) if count else 0.0
                )
                variance = (
                    max(
                        0.0,
                        float(confidence_square_sum[component_id] / count)
                        - mean * mean,
                    )
                    if count
                    else 0.0
                )
                part_id = f"{partition_id}:{component_id:08d}"
                object_digest = hashlib.sha256(
                    f"{payload['run_id']}|{payload['stream_id']}|{part_id}|{POLICY_VERSION}".encode(
                        "utf-8"
                    )
                ).hexdigest()[:40]
                class_code = int(component_classes[component_id])
                vertices = _vertex_count(geometry)
                destination.write(
                    {
                        "geometry": mapping(geometry),
                        "properties": {
                            "run_id": str(payload["run_id"]),
                            "result_stream_id": str(payload["stream_id"]),
                            "result_kind": "fusion",
                            "model_id": "",
                            "fusion_profile_id": str(payload["profile_id"]),
                            "object_id": f"v3_{object_digest}",
                            "part_id": part_id,
                            "class_code": class_code,
                            "class_name": CLASS_NAMES[class_code],
                            "confidence_mean": mean,
                            "confidence_std": math.sqrt(variance),
                            "model_version": str(payload.get("model_version") or ""),
                            "source": "semantic_fusion_v3",
                            "fit_changed": 0,
                            "fit_methods": POLICY_ID,
                            "fit_version": FIT_VERSION,
                            "fit_status": "regularized",
                            "origin_unit_ids": partition_id,
                            "vertex_count_before": vertices,
                            "vertex_count_after": vertices,
                            "max_shift_px": 0.0,
                            "mean_shift_px": 0.0,
                            "area_change_ratio": 0.0,
                            "created_at": created_at,
                        },
                    }
                )
                feature_count += 1
        if feature_count != component_count:
            raise FragmentationPostprocessError(
                f"component/vector count mismatch for {partition_id}: "
                f"{component_count} != {feature_count}"
            )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    report = {
        "schema_version": 1,
        "status": "passed",
        "partition_id": partition_id,
        "input_fingerprint": input_fingerprint,
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "feature_count": int(feature_count),
        "pixel_count": int(np.count_nonzero(valid)),
        "class_feature_count": {
            str(code): int(np.count_nonzero(component_classes[1:] == code))
            for code in CLASS_ORDER
        },
        "resumed": False,
    }
    _atomic_json(report_path, report)
    return report


def _assemble_vectors(
    output_path: Path,
    part_paths: Sequence[Path],
    *,
    crs_wkt: str,
) -> dict[str, Any]:
    temporary = output_path.with_name(f".{output_path.stem}.building.gpkg")
    temporary.unlink(missing_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        with fiona.open(
            temporary,
            "w",
            driver="GPKG",
            layer=LAYER_NAME,
            schema=FORMAL_SCHEMA,
            crs_wkt=crs_wkt,
        ) as destination:
            total = len(part_paths)
            for position, part_path in enumerate(part_paths, start=1):
                with fiona.open(part_path, layer=LAYER_NAME) as source:
                    batch = []
                    for feature in source:
                        batch.append(feature)
                        if len(batch) >= 4096:
                            destination.writerecords(batch)
                            count += len(batch)
                            batch = []
                    if batch:
                        destination.writerecords(batch)
                        count += len(batch)
                if position % 25 == 0 or position == total:
                    emit(
                        "fragmentation_vector_assembly_progress",
                        current=position,
                        total=total,
                        feature_count=count,
                    )
        with sqlite3.connect(temporary) as connection:
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_semantic_polygons_class_code "
                "ON semantic_polygons(class_code)"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_semantic_polygons_object_id "
                "ON semantic_polygons(object_id)"
            )
            integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
            stored_count = int(
                connection.execute("SELECT COUNT(*) FROM semantic_polygons").fetchone()[0]
            )
        if integrity != "ok" or stored_count != count:
            raise FragmentationPostprocessError(
                f"assembled GPKG validation failed: integrity={integrity}, "
                f"features={stored_count}/{count}"
            )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "feature_count": count,
        "integrity_check": "ok",
        "sha256": sha256_file(output_path),
    }


def _aggregate_mask_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    before = Counter()
    after = Counter()
    changed_pixels = 0
    changed_components = 0
    kept = Counter()
    pairs = Counter()
    for report in reports:
        before.update(
            {int(code): int(value) for code, value in report["class_pixel_count_before"].items()}
        )
        after.update(
            {int(code): int(value) for code, value in report["class_pixel_count_after"].items()}
        )
        changed_pixels += int(report["changed_pixel_count"])
        regularization = report.get("regularization") or {}
        changed_components += int(regularization.get("changed_component_count") or 0)
        kept.update(regularization.get("kept_reason_counts") or {})
        pairs.update(regularization.get("changed_pair_counts") or {})
    protected_unchanged = all(
        before[code] == after[code] for code in PROTECTED_SOURCE_CLASS_CODES
    )
    disappeared = [
        code for code in CLASS_ORDER if before[code] > 0 and after[code] <= 0
    ]
    return {
        "partition_count": len(reports),
        "changed_pixel_count": changed_pixels,
        "changed_component_count": changed_components,
        "class_pixel_count_before": {str(code): before[code] for code in CLASS_ORDER},
        "class_pixel_count_after": {str(code): after[code] for code in CLASS_ORDER},
        "protected_classes_unchanged": protected_unchanged,
        "disappeared_class_codes": disappeared,
        "kept_reason_counts": dict(sorted(kept.items())),
        "changed_pair_counts": dict(sorted(pairs.items())),
        "passed": protected_unchanged and not disappeared,
    }


def _activate_review(
    run_dir: Path,
    *,
    stream_id: str,
    manifest: Mapping[str, Any],
) -> Path:
    run_manifest_path = run_dir / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise FragmentationPostprocessError(
            f"run manifest is missing; cannot activate review source: {run_manifest_path}"
        )
    run_manifest = dict(load_json(run_manifest_path))
    updated = 0
    for key in ("ready_streams", "streams"):
        collection = run_manifest.get(key)
        if not isinstance(collection, list):
            continue
        for stream in collection:
            if str(stream.get("stream_id") or "") != stream_id:
                continue
            stream["review_polygons"] = str(manifest["semantic_polygons"])
            stream["review_layer_name"] = LAYER_NAME
            checksums = dict(stream.get("output_sha256") or {})
            checksums["review_polygons"] = str(
                manifest["semantic_polygons_sha256"]
            )
            stream["output_sha256"] = checksums
            stream["fragmentation_postprocess"] = {
                "policy_id": POLICY_ID,
                "policy_version": POLICY_VERSION,
                "manifest": str(
                    Path(str(manifest["report_path"])).parent / MANIFEST_NAME
                ),
                "report": str(manifest["report_path"]),
            }
            updated += 1
    if updated < 1:
        raise FragmentationPostprocessError(
            f"stream {stream_id!r} is not present in run_manifest"
        )
    _atomic_json(run_manifest_path, run_manifest)
    return run_manifest_path


def run_postprocess(args: argparse.Namespace) -> dict[str, Any]:
    spec_path = Path(args.run_spec).expanduser().resolve()
    spec = load_json(spec_path)
    if spec.get("schema_version") != 2:
        raise FragmentationPostprocessError("V3 postprocess requires run_spec schema 2")
    stream_id = str(args.stream_id)
    stream = _stream(spec, stream_id)
    run_dir = Path(str(spec["run_dir"])).resolve()
    source_root = run_dir / "fusion" / str(stream["profile_id"])
    mask_dir = source_root / "raster_parts"
    mask_paths = sorted(mask_dir.glob("partition_*_mask.tif"))
    if not mask_paths:
        raise FragmentationPostprocessError(f"no Fusion partition masks: {mask_dir}")
    expected = int(
        (spec.get("spatial_plan_summary") or {}).get("partition_count")
        or len(mask_paths)
    )
    if len(mask_paths) != expected:
        raise FragmentationPostprocessError(
            f"incomplete mask set: {len(mask_paths)}/{expected}"
        )
    for path in mask_paths:
        confidence = _confidence_path(path)
        if not confidence.is_file() or confidence.stat().st_size <= 0:
            raise FragmentationPostprocessError(
                f"missing confidence partition: {confidence}"
            )
    path_map = {_partition_key(path): path for path in mask_paths}
    source_inventory_fingerprint = _source_inventory_fingerprint(mask_paths)
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if str(args.output_dir or "").strip()
        else derived_root(spec, stream)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".fragmentation_v3.lock"
    lock_descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise FragmentationPostprocessError(
                f"another V3 postprocess owns the output lock: {lock_path}"
            ) from error
        workers = max(1, int(args.workers))
        resume = not bool(args.restart)
        manifest_path = output_root / MANIFEST_NAME
        if resume and args.stage == "all" and manifest_path.is_file():
            try:
                completed = dict(load_json(manifest_path))
                completed_output = Path(str(completed["semantic_polygons"]))
                completed_report = Path(str(completed["report_path"]))
                if (
                    completed.get("status") == "passed"
                    and completed.get("run_id") == spec.get("run_id")
                    and completed.get("stream_id") == stream_id
                    and completed.get("policy_version") == POLICY_VERSION
                    and completed.get("source_inventory_fingerprint")
                    == source_inventory_fingerprint
                    and completed.get("source_run_spec_sha256")
                    == sha256_file(spec_path)
                    and completed_output.is_file()
                    and completed.get("semantic_polygons_sha256")
                    == sha256_file(completed_output)
                    and completed_report.is_file()
                    and completed.get("report_sha256")
                    == sha256_file(completed_report)
                ):
                    if bool(args.activate_review):
                        _activate_review(
                            run_dir, stream_id=stream_id, manifest=completed
                        )
                    emit(
                        "fragmentation_postprocess_resumed",
                        run_id=spec["run_id"],
                        stream_id=stream_id,
                        output_dir=str(output_root),
                        activated=bool(args.activate_review),
                    )
                    return completed
            except (KeyError, OSError, TypeError, ValueError):
                pass
        mask_outputs = output_root / "regularized_raster_parts"
        mask_reports_root = output_root / "partition_reports" / "masks"
        vector_outputs = output_root / "polygon_parts"
        vector_reports_root = output_root / "partition_reports" / "vectors"
        mask_reports: list[dict[str, Any]] = []
        if args.stage in {"all", "masks"}:
            emit(
                "fragmentation_mask_stage_started",
                total=len(mask_paths),
                workers=workers,
                policy_version=POLICY_VERSION,
            )
            payloads = []
            for path in mask_paths:
                row, col = _partition_key(path)
                neighbor_map = {
                    f"{row + dr},{col + dc}": str(
                        path_map[(row + dr, col + dc)]
                    )
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    if (row + dr, col + dc) in path_map
                }
                payloads.append(
                    {
                        "center_path": str(path),
                        "output_path": str(mask_outputs / path.name),
                        "report_path": str(
                            mask_reports_root / f"{_partition_id(path)}.json"
                        ),
                        "path_map": neighbor_map,
                        "buffer_pixels": int(args.buffer_pixels),
                        "resume": resume,
                    }
                )
            mask_reports = _run_bounded_processes(
                payloads,
                _process_mask_partition,
                workers=workers,
                event_prefix="fragmentation_mask",
            )
        else:
            for path in mask_paths:
                report_path = mask_reports_root / f"{_partition_id(path)}.json"
                output_path = mask_outputs / path.name
                if not report_path.is_file() or not output_path.is_file():
                    raise FragmentationPostprocessError(
                        f"mask stage is incomplete: {_partition_id(path)}"
                    )
                report = dict(load_json(report_path))
                if report.get("output_sha256") != sha256_file(output_path):
                    raise FragmentationPostprocessError(
                        f"regularized mask changed: {output_path}"
                    )
                mask_reports.append(report)
        mask_summary = _aggregate_mask_reports(mask_reports)
        if not mask_summary["passed"]:
            raise FragmentationPostprocessError(
                "V3 semantic safety failed: "
                + json.dumps(mask_summary, ensure_ascii=False, separators=(",", ":"))
            )
        regularized_paths = [mask_outputs / path.name for path in mask_paths]
        mask_vrt = output_root / "mask_mosaic.vrt"
        if args.stage in {"all", "masks"}:
            build_vrt(mask_vrt, regularized_paths)

        if args.stage == "masks":
            result = {
                "schema_version": 1,
                "status": "masks_ready",
                "run_id": spec["run_id"],
                "stream_id": stream_id,
                "policy": policy_snapshot(),
                "mask_mosaic": str(mask_vrt),
                "mask_summary": mask_summary,
            }
            _atomic_json(output_root / REPORT_NAME, result)
            emit("fragmentation_mask_stage_finished", **result)
            return result

        emit(
            "fragmentation_vector_stage_started",
            total=len(mask_paths),
            workers=workers,
        )
        vector_payloads = [
            {
                "run_id": spec["run_id"],
                "stream_id": stream_id,
                "profile_id": stream["profile_id"],
                "model_version": stream.get("version", ""),
                "partition_id": _partition_id(source),
                "mask_path": str(mask_outputs / source.name),
                "confidence_path": str(_confidence_path(source)),
                "output_path": str(
                    vector_outputs / f"{_partition_id(source)}.gpkg"
                ),
                "report_path": str(
                    vector_reports_root / f"{_partition_id(source)}.json"
                ),
                "resume": resume,
            }
            for source in mask_paths
        ]
        vector_reports = _run_bounded_processes(
            vector_payloads,
            _process_vector_partition,
            workers=workers,
            event_prefix="fragmentation_vector",
        )
        vector_reports.sort(key=lambda item: str(item["partition_id"]))
        part_paths = [
            vector_outputs / f"{_partition_id(source)}.gpkg"
            for source in mask_paths
        ]
        with rasterio.open(regularized_paths[0]) as reference:
            crs_wkt = reference.crs.to_wkt()
        semantic_path = output_root / "semantic_polygons.gpkg"
        assembly = _assemble_vectors(
            semantic_path,
            part_paths,
            crs_wkt=crs_wkt,
        )
        expected_features = sum(int(item["feature_count"]) for item in vector_reports)
        if int(assembly["feature_count"]) != expected_features:
            raise FragmentationPostprocessError(
                f"final feature count mismatch: {assembly['feature_count']}/{expected_features}"
            )
        report = {
            "schema_version": 1,
            "status": "passed",
            "run_id": spec["run_id"],
            "stream_id": stream_id,
            "source_run_spec": str(spec_path),
            "source_run_spec_sha256": sha256_file(spec_path),
            "source_inventory_fingerprint": source_inventory_fingerprint,
            "source_mask_mosaic": str(source_root / "mask_mosaic.vrt"),
            "source_confidence_mosaic": str(source_root / "confidence_mosaic.vrt"),
            "policy": policy_snapshot(),
            "buffer_pixels": int(args.buffer_pixels),
            "workers": workers,
            "mask_mosaic": str(mask_vrt),
            "semantic_polygons": str(semantic_path),
            "semantic_polygons_sha256": assembly["sha256"],
            "semantic_polygon_feature_count": assembly["feature_count"],
            "partition_count": len(mask_paths),
            "mask_summary": mask_summary,
            "vector_part_feature_count": expected_features,
            "validation": {
                "passed": True,
                "gpkg_integrity_check": assembly["integrity_check"],
                "protected_classes_unchanged": mask_summary[
                    "protected_classes_unchanged"
                ],
                "disappeared_class_codes": mask_summary[
                    "disappeared_class_codes"
                ],
            },
            "created_at": dt.datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
        }
        report_path = output_root / REPORT_NAME
        _atomic_json(report_path, report)
        manifest = {
            "schema_version": 1,
            "status": "passed",
            "run_id": spec["run_id"],
            "stream_id": stream_id,
            "policy_id": POLICY_ID,
            "policy_version": POLICY_VERSION,
            "source_run_spec_sha256": sha256_file(spec_path),
            "source_inventory_fingerprint": source_inventory_fingerprint,
            "semantic_polygons": str(semantic_path.resolve()),
            "semantic_polygons_layer": LAYER_NAME,
            "semantic_polygons_sha256": assembly["sha256"],
            "semantic_polygon_feature_count": assembly["feature_count"],
            "mask_mosaic": str(mask_vrt.resolve()),
            "report_path": str(report_path.resolve()),
            "report_sha256": sha256_file(report_path),
        }
        _atomic_json(manifest_path, manifest)
        if bool(args.activate_review):
            activated = _activate_review(
                run_dir, stream_id=stream_id, manifest=manifest
            )
            manifest["activated_run_manifest"] = str(activated)
        emit(
            "fragmentation_postprocess_finished",
            run_id=spec["run_id"],
            stream_id=stream_id,
            output_dir=str(output_root),
            feature_count=assembly["feature_count"],
            activated=bool(args.activate_review),
        )
        return manifest
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resume V3 fragmentation repair from committed Fusion rasters"
    )
    parser.add_argument("--run-spec", required=True)
    parser.add_argument("--stream-id", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--stage", choices=("all", "masks", "vectors"), default="all")
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--buffer-pixels", type=int, default=DEFAULT_BUFFER_PIXELS)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--activate-review", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.workers < 1 or args.buffer_pixels < 1:
        raise SystemExit("--workers and --buffer-pixels must be positive")
    try:
        run_postprocess(args)
        return 0
    except Exception as error:
        emit("fragmentation_postprocess_failed", error=str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
