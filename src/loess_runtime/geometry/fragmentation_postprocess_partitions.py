"""Single-partition algorithms and I/O for historical V3 postprocessing."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, TypeAlias, cast

import fiona
import numpy as np
import rasterio  # type: ignore[import-untyped]
import shapely  # type: ignore[import-untyped]
from affine import Affine  # type: ignore[import-untyped]
from numpy.typing import NDArray
from rasterio.features import shapes  # type: ignore[import-untyped]
from rasterio.merge import merge  # type: ignore[import-untyped]
from scipy import ndimage  # type: ignore[import-untyped]
from shapely.geometry import (  # type: ignore[import-untyped]
    GeometryCollection,
    MultiPolygon,
    Polygon,
    mapping,
    shape,
)

from labeling_tool.shared.contracts.run_spec import (
    CLASS_NAMES,
    CLASS_ORDER,
    load_json,
    sha256_file,
)
from loess_runtime.geometry.fragmentation_v3 import (
    FIT_VERSION,
    POLICY_ID,
    POLICY_VERSION,
    policy_snapshot,
    production_policy,
)
from loess_runtime.geometry.small_component_regularizer import (
    EIGHT_CONNECTED,
    physical_pixel_area_m2,
    regularize_small_components,
)

__all__ = [
    "FORMAL_SCHEMA",
    "LAYER_NAME",
    "FragmentationPostprocessError",
    "confidence_path",
    "file_signature",
    "json_fingerprint",
    "partition_id",
    "partition_key",
    "process_mask_partition",
    "process_vector_partition",
    "write_atomic_json",
]

BoolArray: TypeAlias = NDArray[np.bool_]
Float32Array: TypeAlias = NDArray[np.float32]
Int16Array: TypeAlias = NDArray[np.int16]
Int32Array: TypeAlias = NDArray[np.int32]

PARTITION_PATTERN = re.compile(r"^(partition_(\d+)_(\d+))_mask\.tif$")
LAYER_NAME = "semantic_polygons"
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


class FragmentationPostprocessError(RuntimeError):
    pass


def write_atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def json_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def partition_key(path: Path) -> tuple[int, int]:
    match = PARTITION_PATTERN.match(path.name)
    if match is None:
        raise FragmentationPostprocessError(f"unexpected partition mask: {path}")
    return int(match.group(2)), int(match.group(3))


def partition_id(path: Path) -> str:
    match = PARTITION_PATTERN.match(path.name)
    if match is None:
        raise FragmentationPostprocessError(f"unexpected partition mask: {path}")
    return str(match.group(1))


def confidence_path(mask_path: Path) -> Path:
    return mask_path.with_name(mask_path.name.replace("_mask.tif", "_confidence.tif"))


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _load_buffered_partition(
    center_path: Path,
    path_map: Mapping[tuple[int, int], Path],
    *,
    buffer_pixels: int,
) -> tuple[
    Int16Array,
    Float32Array,
    Affine,
    rasterio.crs.CRS,
    tuple[slice, slice],
    dict[str, Any],
]:
    row, col = partition_key(center_path)
    neighbor_paths = [
        path_map[(row + dr, col + dc)]
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (row + dr, col + dc) in path_map
    ]
    mask_sources = [rasterio.open(path) for path in neighbor_paths]
    confidence_sources = [
        rasterio.open(confidence_path(path)) for path in neighbor_paths
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
                f"mask/confidence grids disagree for {partition_id(center_path)}"
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
    values: Int16Array,
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
    row, col = partition_key(center_path)
    inputs: list[dict[str, Any]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            path = path_map.get((row + dr, col + dc))
            if path is None:
                continue
            inputs.append(file_signature(path))
            inputs.append(file_signature(confidence_path(path)))
    payload = {
        "policy": policy_snapshot(),
        "buffer_pixels": int(buffer_pixels),
        "inputs": inputs,
    }
    return json_fingerprint(payload), inputs


def process_mask_partition(payload: Mapping[str, Any]) -> dict[str, Any]:
    center_path = Path(str(payload["center_path"]))
    output_path = Path(str(payload["output_path"]))
    report_path = Path(str(payload["report_path"]))
    buffer_pixels = int(payload["buffer_pixels"])
    path_map = cast(
        dict[tuple[int, int], Path],
        {
            tuple(int(value) for value in key.split(",")): Path(str(value))
            for key, value in dict(payload["path_map"]).items()
        },
    )
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
    ) = _load_buffered_partition(center_path, path_map, buffer_pixels=buffer_pixels)
    valid = labels >= 0
    if np.any(valid & (labels >= len(CLASS_ORDER))):
        raise FragmentationPostprocessError(
            f"invalid class index in {partition_id(center_path)}"
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

    before_counts = np.bincount(core_before[core_valid], minlength=len(CLASS_ORDER))
    after_counts = np.bincount(core_after[core_valid], minlength=len(CLASS_ORDER))
    report = {
        "schema_version": 1,
        "status": "passed",
        "partition_id": partition_id(center_path),
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
    write_atomic_json(report_path, report)
    return report


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
    labels: Int16Array, valid: BoolArray
) -> tuple[Int32Array, Int16Array]:
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
        raise FragmentationPostprocessError(
            "component raster left valid pixels unassigned"
        )
    return component_map, np.asarray(class_codes, dtype=np.int16)


def process_vector_partition(payload: Mapping[str, Any]) -> dict[str, Any]:
    mask_path = Path(str(payload["mask_path"]))
    confidence_path = Path(str(payload["confidence_path"]))
    output_path = Path(str(payload["output_path"]))
    report_path = Path(str(payload["report_path"]))
    partition_id = str(payload["partition_id"])
    input_fingerprint = json_fingerprint(
        {
            "policy_version": POLICY_VERSION,
            "mask_sha256": sha256_file(mask_path),
            "confidence": file_signature(confidence_path),
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
    confidence_count = np.bincount(component_map[finite], minlength=component_count + 1)

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
                        "cannot repair component geometry in "
                        f"{partition_id}: {component_id}"
                    )
                count = int(confidence_count[component_id])
                mean = float(confidence_sum[component_id] / count) if count else 0.0
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
                "component/vector count mismatch for "
                f"{partition_id}: {component_count} != {feature_count}"
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
    write_atomic_json(report_path, report)
    return report
