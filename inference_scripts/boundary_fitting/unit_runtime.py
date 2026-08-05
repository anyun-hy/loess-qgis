"""Polygonize one bounded unit and smooth each polygon-pair divider once."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import fiona
import numpy as np
import rasterio
from affine import Affine
from fiona.crs import CRS
from rasterio.features import geometry_mask, shapes
from rasterio.windows import Window
from shapely.affinity import affine_transform
from shapely.geometry import LineString, MultiPolygon, Polygon, mapping, shape

ROOT = Path(__file__).resolve().parents[2]
PLUGIN_ROOT = ROOT / "qgis_plugins"
SCRIPTS_ROOT = ROOT / "inference_scripts"
import sys

for import_root in (PLUGIN_ROOT, SCRIPTS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from labeling_tool.core.run_state_db import RunStateDB

from common_boundary_smoother import smooth_common_boundaries
from polyline_smoother import SmoothingConfig
from deployment_config import load_json
from runtime_metrics import peak_rss_bytes
from rasterio_compat import quiet_deprecated_memory_driver
from storage_guard import StorageGuard, exact_remaining_permanent_bytes
from work_package_runtime import _commit_artifact


class UnitRuntimeError(RuntimeError):
    pass


GPKG_ATOMIC_OVERHEAD_BYTES = 4 * 1024**2
JSON_ATOMIC_OVERHEAD_BYTES = 64 * 1024


def _remaining_permanent_reserve_bytes(
    spec: Mapping[str, Any],
    database: RunStateDB,
) -> int:
    return exact_remaining_permanent_bytes(spec, database)


def _run_storage_guard(
    spec: Mapping[str, Any],
    database: RunStateDB,
    *,
    disk_usage=None,
) -> StorageGuard:
    storage = dict(spec.get("storage_preflight") or {})
    min_free_bytes = int(
        storage.get("effective_min_free_disk_bytes")
        or float((spec.get("scaling") or {}).get("min_free_disk_gb", 0.0))
        * 1024**3
    )
    remaining = _remaining_permanent_reserve_bytes(spec, database)
    options = {
        "min_free_bytes": max(0, min_free_bytes),
        "remaining_permanent_bytes": lambda: remaining,
    }
    if disk_usage is not None:
        options["disk_usage"] = disk_usage
    return StorageGuard(Path(spec["run_dir"]), **options)


@contextmanager
def _reserved_vector_write(
    storage_guard: StorageGuard | None,
    lock_path: Path | None,
    operation: str,
    write_bytes: int,
):
    """Reserve one atomic vector write across all unit-fit processes.

    The lock is held only for the file transaction.  Geometry calculation stays
    parallel, while two processes cannot both pass a stale free-space check.
    """

    if storage_guard is None:
        yield
        return
    shared_lock = lock_path or (
        storage_guard.root / "tmp" / ".vector-storage-reserve.lock"
    )
    shared_lock.parent.mkdir(parents=True, exist_ok=True)
    with shared_lock.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        reserved_write = 0
        try:
            reservation = storage_guard.check(
                operation,
                write_bytes=max(0, int(write_bytes)),
                managed_growth_bytes=0,
                reserve_managed_growth=True,
            )
            reserved_write = int(reservation["reserved_write_bytes"])
            yield
        finally:
            if reserved_write:
                storage_guard.adjust(0, settled_write_bytes=reserved_write)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _estimate_record_gpkg_bytes(
    records: list[Mapping[str, Any]],
    *,
    attribute_bytes: int,
) -> int:
    payload_bytes = 0
    for record in records:
        geometry = record.get("geometry")
        payload_bytes += len(geometry.wkb) if geometry is not None else 0
        payload_bytes += max(0, int(attribute_bytes))
    return max(GPKG_ATOMIC_OVERHEAD_BYTES, payload_bytes * 2)


def _estimate_diagnostic_gpkg_bytes(report: Mapping[str, Any]) -> int:
    payload_bytes = sum(
        len(edge.get("fitted_points") or []) * 16 + 512
        for edge in report.get("diagnostics") or []
    )
    return max(GPKG_ATOMIC_OVERHEAD_BYTES, payload_bytes * 2)


def _estimate_json_bytes(payload: Mapping[str, Any]) -> int:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    return len(encoded) + JSON_ATOMIC_OVERHEAD_BYTES


def _write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    storage_guard: StorageGuard | None = None,
    storage_lock_path: Path | None = None,
    operation: str = "unit_report",
) -> None:
    from semantic_batch import _atomic_json

    with _reserved_vector_write(
        storage_guard,
        storage_lock_path,
        operation,
        _estimate_json_bytes(payload),
    ):
        _atomic_json(path, payload)


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False, separators=(",", ":")), flush=True)


def _decode_partition_window(
    database: RunStateDB,
    run_id: str,
    stream_id: str,
    partition_id: str,
    unit_window: Mapping[str, int],
) -> np.ndarray:
    partition = database.get_partition(run_id, partition_id)
    artifact = database.artifact_for_stream_unit(
        run_id, stream_id, partition_id, "partition_probability"
    )
    if partition is None or artifact is None:
        raise UnitRuntimeError(
            f"Partition probability dependency is missing: {stream_id}/{partition_id}"
        )
    halo = partition["halo_window"]
    x0 = int(unit_window["x0"])
    y0 = int(unit_window["y0"])
    x1 = int(unit_window["x1"])
    y1 = int(unit_window["y1"])
    if not (
        int(halo["x0"]) <= x0 < x1 <= int(halo["x1"])
        and int(halo["y0"]) <= y0 < y1 <= int(halo["y1"])
    ):
        raise UnitRuntimeError(f"unit window is outside Partition Halo: {partition_id}")
    window = Window(
        x0 - int(halo["x0"]),
        y0 - int(halo["y0"]),
        x1 - x0,
        y1 - y0,
    )
    with rasterio.open(artifact["path"]) as source:
        raw = source.read(window=window)
        scales = np.asarray(source.scales, dtype=np.float32)
    if raw.shape != (14, y1 - y0, x1 - x0):
        raise UnitRuntimeError(
            f"Partition probability crop has unexpected shape: {raw.shape}"
        )
    if scales.shape != (14,) or np.any(scales <= 0):
        raise UnitRuntimeError("Partition probability scale metadata is invalid")
    return raw.astype(np.float32) * scales[:, None, None]


def _unit_probabilities(
    database: RunStateDB,
    run_id: str,
    stream_id: str,
    unit: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    arrays = [
        _decode_partition_window(
            database,
            run_id,
            stream_id,
            partition_id,
            unit["pixel_window"],
        )
        for partition_id in unit["dependency_ids"]
    ]
    if not arrays:
        raise UnitRuntimeError("spatial unit has no Partition dependencies")
    reference = arrays[0].shape
    if any(array.shape != reference for array in arrays):
        raise UnitRuntimeError("Partition Halo crops disagree on unit shape")
    probabilities = np.mean(arrays, axis=0, dtype=np.float32)
    denominator = probabilities.sum(axis=0, keepdims=True)
    valid = denominator[0] > 0
    normalized = np.zeros_like(probabilities)
    if np.any(valid):
        normalized[:, valid] = probabilities[:, valid] / denominator[:, valid]
    return normalized, valid


def _polygonize(
    probabilities: np.ndarray,
    unit: Mapping[str, Any],
    class_codes: list[int],
    valid_mask: np.ndarray | None = None,
):
    mask = probabilities.argmax(axis=0).astype(np.int16)
    valid = (
        np.asarray(valid_mask, dtype=bool)
        if valid_mask is not None
        else np.ones(mask.shape, dtype=bool)
    )
    if valid.shape != mask.shape:
        raise UnitRuntimeError("unit validity mask shape does not match probabilities")
    window = unit["pixel_window"]
    transform = Affine.translation(int(window["x0"]), int(window["y0"]))
    records = []
    # Rasterio 1.4.x still asks GDAL 3.11+ for the deprecated OGR "Memory"
    # driver internally. Keep native diagnostics quiet here; Python exceptions
    # from polygonization continue to propagate normally.
    with quiet_deprecated_memory_driver():
        polygons = shapes(
            mask,
            mask=valid.astype(np.uint8),
            transform=transform,
        )
        for index, (geometry, value) in enumerate(polygons):
            class_index = int(value)
            if not 0 <= class_index < len(class_codes):
                raise UnitRuntimeError(
                    f"polygonize produced invalid class index: {class_index}"
                )
            records.append(
                {
                    "polygon_id": f"{unit['unit_id']}_{index:07d}",
                    "class_code": int(class_codes[class_index]),
                    "geometry": shape(geometry),
                }
            )
    return records


def _attach_confidence(
    records: list[dict[str, Any]],
    probabilities: np.ndarray,
    valid_mask: np.ndarray,
    unit: Mapping[str, Any],
) -> None:
    confidence = probabilities.max(axis=0)
    window = unit["pixel_window"]
    transform = Affine.translation(int(window["x0"]), int(window["y0"]))
    for record in records:
        selected = geometry_mask(
            [mapping(record["geometry"])],
            out_shape=confidence.shape,
            transform=transform,
            invert=True,
        )
        values = confidence[selected & valid_mask]
        record["confidence_mean"] = float(values.mean()) if values.size else 0.0
        record["confidence_std"] = float(values.std()) if values.size else 0.0


def _to_map_geometry(geometry, transform: Affine):
    return affine_transform(
        geometry,
        [transform.a, transform.b, transform.d, transform.e, transform.c, transform.f],
    )


def _multipolygon(geometry):
    if isinstance(geometry, MultiPolygon):
        return geometry
    if isinstance(geometry, Polygon):
        return MultiPolygon([geometry])
    raise UnitRuntimeError(f"output geometry is not polygonal: {geometry.geom_type}")


def _write_gpkg(
    path: Path,
    records: list[Mapping[str, Any]],
    *,
    transform: Affine,
    crs: str,
    include_fit: bool,
    storage_guard: StorageGuard | None = None,
    storage_lock_path: Path | None = None,
    operation: str = "unit_gpkg",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.{os.getpid()}.tmp.gpkg"
    properties = {
        "polygon_id": "str:96",
        "class_code": "int",
        "conf_mean": "float",
        "conf_std": "float",
    }
    if include_fit:
        properties.update(
            {
                "fit_method": "str:64",
                "fit_status": "str:32",
                "fit_version": "str:40",
                "vtx_before": "int",
                "vtx_after": "int",
                "max_shift": "float",
                "mean_shift": "float",
                "area_ratio": "float",
            }
        )
    schema = {"geometry": "MultiPolygon", "properties": properties}

    def feature_records():
        for record in records:
            values = {
                "polygon_id": str(record["polygon_id"]),
                "class_code": int(record["class_code"]),
                "conf_mean": float(record.get("confidence_mean", 0.0)),
                "conf_std": float(record.get("confidence_std", 0.0)),
            }
            if include_fit:
                values.update(
                    {
                        "fit_method": str(record["fit_method"]),
                        "fit_status": str(record["fit_status"]),
                        "fit_version": str(record["fit_version"]),
                        "vtx_before": int(record["vertex_count_before"]),
                        "vtx_after": int(record["vertex_count_after"]),
                        "max_shift": float(record["max_shift_px"]),
                        "mean_shift": float(record["mean_shift_px"]),
                        "area_ratio": float(record["area_change_ratio"]),
                    }
                )
            map_geometry = _to_map_geometry(record["geometry"], transform)
            yield {
                "geometry": mapping(_multipolygon(map_geometry)),
                "properties": values,
            }

    estimated_write_bytes = _estimate_record_gpkg_bytes(
        records,
        attribute_bytes=768 if include_fit else 384,
    )
    with _reserved_vector_write(
        storage_guard,
        storage_lock_path,
        operation,
        estimated_write_bytes,
    ):
        temporary.unlink(missing_ok=True)
        try:
            with fiona.open(
                temporary,
                "w",
                driver="GPKG",
                layer="polygons",
                schema=schema,
                crs=CRS.from_user_input(crs),
            ) as destination:
                destination.writerecords(feature_records())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _diagnostic_feature_records(
    report: Mapping[str, Any],
    *,
    run_id: str,
    stream_id: str,
    unit_id: str,
    transform: Affine,
):
    for edge in report.get("diagnostics") or []:
        if edge.get("method") not in {
            "line",
            "spline",
            "cubic_bspline",
            "cubic_bspline_adaptive",
        } and not str(
            edge.get("status") or ""
        ).startswith("failed"):
            continue
        points = edge.get("fitted_points") or []
        if len(points) < 2:
            continue
        yield {
            "geometry": mapping(_to_map_geometry(LineString(points), transform)),
            "properties": {
                "run_id": str(run_id),
                "stream_id": str(stream_id),
                "unit_id": str(unit_id),
                "chain_id": str(edge.get("chain_id") or ""),
                "method": str(edge.get("method") or "unchanged"),
                "status": str(edge.get("status") or ""),
                "max_shift": float(edge.get("max_displacement_px") or 0.0),
                "dense_vtx": int(edge.get("point_count_dense") or 0),
                "sparse_vtx": int(edge.get("point_count_after") or 0),
                "chord_err": float(edge.get("max_chord_error_px") or 0.0),
                "arc_len": float(
                    edge.get("max_segment_arc_length_px") or 0.0
                ),
            },
        }


def _write_diagnostic_gpkg(
    path: Path,
    report: Mapping[str, Any],
    *,
    run_id: str,
    stream_id: str,
    unit_id: str,
    transform: Affine,
    crs: str,
    storage_guard: StorageGuard | None = None,
    storage_lock_path: Path | None = None,
    operation: str = "unit_fitted_edges",
) -> int:
    edge_count = sum(
        1
        for _item in _diagnostic_feature_records(
            report,
            run_id=run_id,
            stream_id=stream_id,
            unit_id=unit_id,
            transform=transform,
        )
    )
    if edge_count == 0:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.{os.getpid()}.tmp.gpkg"
    schema = {
        "geometry": "LineString",
        "properties": {
            "run_id": "str:48",
            "stream_id": "str:96",
            "unit_id": "str:96",
            "chain_id": "str:96",
            "method": "str:24",
            "status": "str:32",
            "max_shift": "float",
            "dense_vtx": "int",
            "sparse_vtx": "int",
            "chord_err": "float",
            "arc_len": "float",
        },
    }
    with _reserved_vector_write(
        storage_guard,
        storage_lock_path,
        operation,
        _estimate_diagnostic_gpkg_bytes(report),
    ):
        temporary.unlink(missing_ok=True)
        try:
            with fiona.open(
                temporary,
                "w",
                driver="GPKG",
                layer="fitted_edges",
                schema=schema,
                crs=CRS.from_user_input(crs),
            ) as destination:
                destination.writerecords(
                    _diagnostic_feature_records(
                        report,
                        run_id=run_id,
                        stream_id=stream_id,
                        unit_id=unit_id,
                        transform=transform,
                    )
                )
            os.replace(temporary, path)
            return edge_count
        finally:
            temporary.unlink(missing_ok=True)


def _smoothing_config(value: Mapping[str, Any]) -> SmoothingConfig:
    return SmoothingConfig(
        smoothing_factor=float(value.get("smoothing_factor", 1.0)),
        curve_sampling_spacing=float(
            value.get("curve_sampling_spacing_px", 0.5)
        ),
        max_chord_error=float(value.get("max_chord_error_px", 0.25)),
        max_segment_arc_length=float(
            value.get("max_segment_arc_length_px", 8.0)
        ),
        max_deviation=None,
        min_point_count=4,
    )


def _vertex_count(records: list[Mapping[str, Any]]) -> int:
    return sum(
        len(record["geometry"].exterior.coords)
        + sum(len(ring.coords) for ring in record["geometry"].interiors)
        for record in records
    )


def _without_smoothing(raw_records: list[Mapping[str, Any]]):
    formal_records = []
    for record in raw_records:
        geometry = record["geometry"]
        if geometry.is_empty or not geometry.is_valid or geometry.area <= 0:
            raise UnitRuntimeError(
                "raw polygonization produced invalid or non-positive geometry"
            )
        vertex_count = (
            len(geometry.exterior.coords)
            + sum(len(ring.coords) for ring in geometry.interiors)
        )
        formal_records.append(
            {
                **dict(record),
                "fit_method": "none",
                "fit_status": "disabled",
                "fit_version": "raw_polygonize_v1",
                "vertex_count_before": vertex_count,
                "vertex_count_after": vertex_count,
                "max_shift_px": 0.0,
                "mean_shift_px": 0.0,
                "area_change_ratio": 0.0,
            }
        )
    return formal_records, {
        "status": "passed",
        "smoothing_enabled": False,
        "fit_version": "raw_polygonize_v1",
        "chain_count": 0,
        "shared_chain_count": 0,
        "spline_count": 0,
        "unchanged_count": 0,
        "skipped_invalid_count": 0,
        "max_displacement_px": 0.0,
        "mean_displacement_px": 0.0,
        "dense_curve_point_count": 0,
        "sparse_curve_point_count": 0,
        "max_chord_error_px": 0.0,
        "max_segment_arc_length_px": 0.0,
        "candidate_validation": {
            "passed": True,
            "scope": "not_applicable",
            "checks": [],
            "rejected_count": 0,
        },
        "validation": {
            "passed": True,
            "scope": "all_output_polygons",
            "invalid_count": 0,
        },
        "diagnostics": [],
    }


def _fit_or_subdivide(
    probabilities: np.ndarray,
    unit: Mapping[str, Any],
    class_codes: list[int],
    *,
    valid_mask: np.ndarray | None = None,
    smoothing_config: SmoothingConfig,
    max_features: int,
    max_segments: int,
    min_core_px: int,
    force_split: bool = False,
    smoothing_enabled: bool = True,
    depth: int = 0,
    output_transform: tuple[float, ...] | None = None,
):
    if valid_mask is None:
        valid_mask = np.ones(probabilities.shape[1:], dtype=bool)
    raw_records = _polygonize(
        probabilities, unit, class_codes, valid_mask=valid_mask
    )
    if not raw_records:
        return [], [], {
            "status": "passed",
            "smoothing_enabled": bool(smoothing_enabled),
            "fit_version": (
                "divider_cubic_bspline_adaptive_v2"
                if smoothing_enabled else "raw_polygonize_v1"
            ),
            "chain_count": 0,
            "shared_chain_count": 0,
            "spline_count": 0,
            "unchanged_count": 0,
            "skipped_invalid_count": 0,
            "max_displacement_px": 0.0,
            "mean_displacement_px": 0.0,
            "dense_curve_point_count": 0,
            "sparse_curve_point_count": 0,
            "max_chord_error_px": 0.0,
            "max_segment_arc_length_px": 0.0,
            "candidate_validation": {
                "passed": True,
                "scope": (
                    "per_common_divider" if smoothing_enabled else "not_applicable"
                ),
                "checks": (
                    ["valid", "positive_area", "pair_total_area"]
                    if smoothing_enabled else []
                ),
                "rejected_count": 0,
            },
            "validation": {
                "passed": True,
                "scope": "all_output_polygons",
                "invalid_count": 0,
            },
            "diagnostics": [],
            "subdivision_depth": depth,
            "subdivision_count": 0,
        }
    segments = _vertex_count(raw_records)
    exceeds = len(raw_records) > max_features or segments > max_segments
    if not exceeds and not force_split:
        if not smoothing_enabled:
            formal_records, report = _without_smoothing(raw_records)
            report["subdivision_depth"] = depth
            report["subdivision_count"] = 0
            return raw_records, formal_records, report
        formal_records, report = smooth_common_boundaries(
            raw_records,
            smoothing_config,
            output_transform=output_transform,
        )
        report["subdivision_depth"] = depth
        report["subdivision_count"] = 0
        return raw_records, formal_records, report

    window = unit["pixel_window"]
    width = int(window["x1"]) - int(window["x0"])
    height = int(window["y1"]) - int(window["y0"])
    split_x = width > min_core_px
    split_y = height > min_core_px
    if not split_x and not split_y:
        raise UnitRuntimeError(
            "unit still exceeds feature/segment threshold at minimum 2x2 Tile Core"
        )
    x_values = [int(window["x0"]), int(window["x1"])]
    y_values = [int(window["y0"]), int(window["y1"])]
    if split_x:
        x_values.insert(1, (x_values[0] + x_values[-1]) // 2)
    if split_y:
        y_values.insert(1, (y_values[0] + y_values[-1]) // 2)
    all_raw = []
    all_formal = []
    child_reports = []
    for child_row in range(len(y_values) - 1):
        for child_col in range(len(x_values) - 1):
            x0, x1 = x_values[child_col], x_values[child_col + 1]
            y0, y1 = y_values[child_row], y_values[child_row + 1]
            child = {
                **dict(unit),
                "unit_id": f"{unit['unit_id']}:q{depth + 1}_{child_row}_{child_col}",
                "pixel_window": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            }
            local_x0 = x0 - int(window["x0"])
            local_x1 = x1 - int(window["x0"])
            local_y0 = y0 - int(window["y0"])
            local_y1 = y1 - int(window["y0"])
            child_raw, child_formal, child_report = _fit_or_subdivide(
                probabilities[:, local_y0:local_y1, local_x0:local_x1],
                child,
                class_codes,
                valid_mask=valid_mask[local_y0:local_y1, local_x0:local_x1],
                smoothing_config=smoothing_config,
                max_features=max_features,
                max_segments=max_segments,
                min_core_px=min_core_px,
                force_split=False,
                smoothing_enabled=smoothing_enabled,
                depth=depth + 1,
                output_transform=output_transform,
            )
            all_raw.extend(child_raw)
            all_formal.extend(child_formal)
            child_reports.append(child_report)
    diagnostics = [
        item
        for child_report in child_reports
        for item in child_report.get("diagnostics") or []
    ]
    report = {
        "status": "passed",
        "smoothing_enabled": bool(smoothing_enabled),
        "fit_version": (
            "divider_cubic_bspline_adaptive_v2"
            if smoothing_enabled else "raw_polygonize_v1"
        ),
        "chain_count": sum(int(item.get("chain_count", 0)) for item in child_reports),
        "shared_chain_count": sum(
            int(item.get("shared_chain_count", 0)) for item in child_reports
        ),
        "spline_count": sum(int(item.get("spline_count", 0)) for item in child_reports),
        "unchanged_count": sum(
            int(item.get("unchanged_count", 0)) for item in child_reports
        ),
        "skipped_invalid_count": sum(
            int(item.get("skipped_invalid_count", 0)) for item in child_reports
        ),
        "max_displacement_px": max(
            (float(item.get("max_displacement_px", 0.0)) for item in child_reports),
            default=0.0,
        ),
        "mean_displacement_px": (
            sum(float(item.get("mean_displacement_px", 0.0)) for item in child_reports)
            / max(len(child_reports), 1)
        ),
        "dense_curve_point_count": sum(
            int(item.get("dense_curve_point_count", 0))
            for item in child_reports
        ),
        "sparse_curve_point_count": sum(
            int(item.get("sparse_curve_point_count", 0))
            for item in child_reports
        ),
        "max_chord_error_px": max(
            (
                float(item.get("max_chord_error_px", 0.0))
                for item in child_reports
            ),
            default=0.0,
        ),
        "max_segment_arc_length_px": max(
            (
                float(item.get("max_segment_arc_length_px", 0.0))
                for item in child_reports
            ),
            default=0.0,
        ),
        "validation": {
            "passed": True,
            "scope": "all_output_polygons",
            "invalid_count": 0,
        },
        "candidate_validation": {
            "passed": True,
            "scope": (
                "per_common_divider" if smoothing_enabled else "not_applicable"
            ),
            "checks": (
                ["valid", "positive_area", "pair_total_area"]
                if smoothing_enabled else []
            ),
            "rejected_count": sum(
                int((item.get("candidate_validation") or {}).get("rejected_count", 0))
                for item in child_reports
            ),
        },
        "diagnostics": diagnostics,
        "subdivision_depth": max(
            (int(item.get("subdivision_depth", depth + 1)) for item in child_reports),
            default=depth + 1,
        ),
        "subdivision_count": len(child_reports) + sum(
            int(item.get("subdivision_count", 0)) for item in child_reports
        ),
    }
    return all_raw, all_formal, report


def run_unit_fit(
    run_spec_path: str | Path,
    stream_id: str,
    unit_id: str,
    *,
    job_id: int,
    lease_token: str,
) -> dict[str, Any]:
    started_at = time.monotonic()
    spec = load_json(Path(run_spec_path).resolve())
    if spec.get("schema_version") != 2:
        raise UnitRuntimeError("unit fitting requires run_spec schema 2")
    run_id = str(spec["run_id"])
    run_dir = Path(spec["run_dir"])
    database = RunStateDB(spec["state_db"])
    unit = database.get_spatial_unit(run_id, unit_id)
    job = database.job_for_unit(run_id, stream_id, unit_id)
    if unit is None or job is None or int(job["job_id"]) != int(job_id):
        raise UnitRuntimeError("unit job identity does not match SQLite state")
    if job["status"] != "running" or job["lease_token"] != lease_token:
        raise UnitRuntimeError("unit job does not hold the supplied lease")
    database.set_stream_unit_status(run_id, stream_id, unit_id, "running")
    try:
        boundary = spec.get("boundary_fitting") or {}
        if (
            str(boundary.get("mode") or "")
            != "divider_cubic_bspline_adaptive_v2"
        ):
            raise UnitRuntimeError(
                "only divider_cubic_bspline_adaptive_v2 is supported "
                "by the current runtime"
            )
        smoothing_enabled = bool(boundary.get("enabled", True))
        emit("polygonize_started", run_id=run_id, stream_id=stream_id, unit_id=unit_id)
        probabilities, valid_mask = _unit_probabilities(
            database, run_id, stream_id, unit
        )
        class_snapshot = load_json(Path(spec["class_mapping_snapshot"]))
        class_codes = [
            int(class_snapshot["index_to_code"][str(index)]) for index in range(14)
        ]
        scaling = spec["scaling"]
        force_marker = (
            run_dir
            / "tmp"
            / "failed_jobs"
            / f"{stream_id.replace(':', '_')}__{unit_id}_force_split.json"
        )
        transform = Affine(*[float(value) for value in spec["raster"]["transform"]])
        output_transform = (
            transform.a,
            transform.b,
            transform.d,
            transform.e,
            transform.c,
            transform.f,
        )
        raw_records, formal_records, report = _fit_or_subdivide(
            probabilities,
            unit,
            class_codes,
            valid_mask=valid_mask,
            smoothing_config=_smoothing_config(boundary),
            max_features=int(scaling.get("max_partition_features", 100000)),
            max_segments=int(scaling.get("max_partition_segments", 250000)),
            min_core_px=2 * int(spec["tile_grid"]["stride"]),
            force_split=force_marker.is_file(),
            smoothing_enabled=smoothing_enabled,
            output_transform=output_transform,
        )
        _attach_confidence(raw_records, probabilities, valid_mask, unit)
        _attach_confidence(formal_records, probabilities, valid_mask, unit)
        vertex_count = _vertex_count(raw_records)
        emit(
            "polygonize_finished",
            run_id=run_id,
            stream_id=stream_id,
            unit_id=unit_id,
            feature_count=len(raw_records),
            segment_count=vertex_count,
        )
        emit(
            "divider_fit_progress" if smoothing_enabled else "boundary_smoothing_skipped",
            run_id=run_id,
            stream_id=stream_id,
            unit_id=unit_id,
        )
        emit(
            "fit_progress",
            run_id=run_id,
            stream_id=stream_id,
            unit_id=unit_id,
            current=report["chain_count"],
            total=report["chain_count"],
        )
        output_root = run_dir / "tmp" / "unit_outputs" / stream_id.replace(":", "_")
        raw_path = output_root / f"{unit_id}_raw.gpkg"
        formal_path = output_root / f"{unit_id}_formal.gpkg"
        report_path = output_root / f"{unit_id}_report.json"
        fitted_edges_path = output_root / f"{unit_id}_fitted_edges.gpkg"
        storage_guard = _run_storage_guard(spec, database)
        storage_lock_path = run_dir / "tmp" / ".vector-storage-reserve.lock"
        _write_gpkg(
            raw_path,
            raw_records,
            transform=transform,
            crs=spec["raster"]["crs"],
            include_fit=False,
            storage_guard=storage_guard,
            storage_lock_path=storage_lock_path,
            operation=f"unit_raw:{stream_id}:{unit_id}",
        )
        emit("rebuild_started", run_id=run_id, stream_id=stream_id, unit_id=unit_id)
        _write_gpkg(
            formal_path,
            formal_records,
            transform=transform,
            crs=spec["raster"]["crs"],
            include_fit=True,
            storage_guard=storage_guard,
            storage_lock_path=storage_lock_path,
            operation=f"unit_formal:{stream_id}:{unit_id}",
        )
        report.update(
            {
                "run_id": run_id,
                "stream_id": stream_id,
                "unit_id": unit_id,
                "unit_type": unit["unit_type"],
                "feature_count": len(formal_records),
                "segment_count": vertex_count,
                "peak_rss_bytes": peak_rss_bytes(),
                "elapsed_sec": round(time.monotonic() - started_at, 3),
            }
        )
        fitted_edge_count = _write_diagnostic_gpkg(
            fitted_edges_path,
            report,
            run_id=run_id,
            stream_id=stream_id,
            unit_id=unit_id,
            transform=transform,
            crs=spec["raster"]["crs"],
            storage_guard=storage_guard,
            storage_lock_path=storage_lock_path,
            operation=f"unit_fitted_edges:{stream_id}:{unit_id}",
        )
        report["fitted_edge_count"] = fitted_edge_count
        persisted_report = {
            key: value
            for key, value in report.items()
            if key != "diagnostics"
        }
        persisted_report["diagnostic_storage"] = {
            "mode": (
                "fitted_edges_gpkg"
                if fitted_edge_count
                else "none"
            ),
            "fitted_edge_count": int(fitted_edge_count),
            "raw_points_persisted": False,
            "fitted_points_in_json": False,
        }
        _write_json(
            report_path,
            persisted_report,
            storage_guard=storage_guard,
            storage_lock_path=storage_lock_path,
            operation=f"unit_report:{stream_id}:{unit_id}",
        )
        emit("rebuild_finished", run_id=run_id, stream_id=stream_id, unit_id=unit_id)
        artifacts = [
            ("unit_raw", raw_path),
            ("unit_formal", formal_path),
            ("unit_boundary_report", report_path),
        ]
        if fitted_edge_count:
            artifacts.append(("unit_fitted_edges", fitted_edges_path))
        for kind, path in artifacts:
            _commit_artifact(
                database,
                run_id,
                path=path,
                kind=kind,
                stream_id=stream_id,
                unit_id=unit_id,
            )
        database.upsert_unit_report_summary(
            run_id,
            stream_id,
            unit_id,
            report,
            fitted_edge_count=fitted_edge_count,
        )
        database.set_stream_unit_status(run_id, stream_id, unit_id, "ready")
        if not database.finish_job(job_id, lease_token, status="ready"):
            raise UnitRuntimeError("unit job lease expired before commit")
        database.release_job_artifacts(job_id)
        emit(
            "validation_finished",
            run_id=run_id,
            stream_id=stream_id,
            unit_id=unit_id,
            status=report["status"],
        )
        return report
    except Exception as error:
        database.set_stream_unit_status(run_id, stream_id, unit_id, "failed", error=str(error))
        database.finish_job(job_id, lease_token, status="failed", error=str(error))
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit polygon-pair dividers with the current cubic B-spline algorithm"
    )
    parser.add_argument("--run-spec", required=True)
    parser.add_argument("--stream-id", required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--job-id", required=True, type=int)
    parser.add_argument("--lease-token", required=True)
    args = parser.parse_args(argv)
    try:
        run_unit_fit(
            args.run_spec,
            args.stream_id,
            args.unit_id,
            job_id=args.job_id,
            lease_token=args.lease_token,
        )
        return 0
    except Exception as error:
        emit(
            "unit_fit_failed",
            stream_id=args.stream_id,
            unit_id=args.unit_id,
            error=str(error),
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
