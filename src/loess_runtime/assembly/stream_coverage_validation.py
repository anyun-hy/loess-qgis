"""Exact-range coverage validation and its durable event."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, Protocol

import fiona
from rasterio.crs import CRS as RasterCRS  # type: ignore[import-untyped]
from shapely.geometry import shape  # type: ignore[import-untyped]
from shapely.ops import unary_union  # type: ignore[import-untyped]

from loess_runtime.assembly.assembly_errors import StreamAssemblyError
from loess_runtime.assembly.range_clip_runtime import (
    RangeClipRuntimeError,
    extract_range_mask_geometry,
)


class AppendCoverageEvent(Protocol):
    """Persist a coverage outcome on the Run event stream."""

    def __call__(
        self,
        run_id: str,
        event_type: str,
        *,
        level: str,
        stream_id: str,
        message: str,
        payload: Mapping[str, Any],
    ) -> int: ...


def assert_gpkg_within_exact_range(
    path: Path,
    *,
    layer: str,
    spec: Mapping[str, Any],
) -> None:
    """Reject a resume formal output that was not clipped to its frozen range."""

    selection = spec.get("range_selection") or {}
    if selection.get("mode") != "vector_tile_intersection" and not isinstance(
        spec.get("requested_extent"), Mapping
    ):
        # Pre-range legacy runs have neither a vector authority nor an extent
        # contract to validate. They remain resumable without inventing one.
        return

    try:
        with fiona.open(path, layer=layer) as source:
            mask = extract_range_mask_geometry(spec, source.crs_wkt or source.crs)
            if mask is None or mask.is_empty:
                raise StreamAssemblyError(
                    "resume formal output has no exact range mask"
                )
            outside = sum(
                1
                for feature in source
                if feature.get("geometry")
                and not mask.covers(shape(feature["geometry"]))
            )
    except RangeClipRuntimeError as error:
        raise StreamAssemblyError(str(error)) from error
    if outside:
        raise StreamAssemblyError(
            f"resume formal output contains {outside} features outside the exact range"
        )


def balanced_geometry_union(
    geometries: Iterable[Any], *, batch_size: int = 2048
) -> Any:
    """Return a bounded-memory union for a potentially very large GPKG."""

    levels: list[Any | None] = []

    def add_partial(value: Any) -> None:
        level = 0
        current = value
        while True:
            if level == len(levels):
                levels.append(current)
                return
            previous = levels[level]
            if previous is None:
                levels[level] = current
                return
            current = unary_union([previous, current])
            levels[level] = None
            level += 1

    batch: list[Any] = []
    for geometry in geometries:
        batch.append(geometry)
        if len(batch) >= batch_size:
            add_partial(unary_union(batch))
            batch = []
    if batch:
        add_partial(unary_union(batch))
    values = [value for value in levels if value is not None]
    return unary_union(values)


def ground_area_scale(source_crs: Any, center_y: float) -> float:
    """Approximate ground square metres per source-coordinate square unit."""

    crs = RasterCRS.from_user_input(source_crs)
    if crs.to_epsg() == 3857:
        latitude = math.atan(math.sinh(float(center_y) / 6378137.0))
        return math.cos(latitude) ** 2
    if crs.is_projected:
        units = str(getattr(crs, "linear_units", "") or "").lower()
        if units in {"metre", "meter", "metres", "meters", "m"}:
            return 1.0
    if crs.is_geographic:
        latitude = float(center_y)
        if not -90.0 <= latitude <= 90.0:
            raise StreamAssemblyError(
                "coverage range centre latitude is outside [-90, 90]"
            )
        lat_rad = math.radians(latitude)
        metres_per_degree_lat = (
            111132.954 - 559.822 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
        )
        metres_per_degree_lon = (math.pi / 180.0) * 6378137.0 * math.cos(lat_rad)
        return abs(metres_per_degree_lat * metres_per_degree_lon)
    raise StreamAssemblyError(
        f"coverage area requires a projected metre CRS, EPSG:3857, or geographic CRS: {crs}"
    )


def validate_exact_range_coverage(
    path: str | Path,
    *,
    layer: str,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure exact range gaps, feature overlap and outside coverage.

    The union is reduced as a balanced tree so validation remains bounded for
    large formal GPKGs.  Area gates use source-coordinate units and are also
    reported as locally corrected ground square metres for the monitor.
    """

    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise StreamAssemblyError(
            f"coverage validation source is missing: {source_path}"
        )
    feature_count = 0
    feature_area = 0.0
    feature_area_compensation = 0.0

    def geometries() -> Iterator[Any]:
        nonlocal feature_count, feature_area, feature_area_compensation
        with fiona.open(source_path, layer=layer) as source:
            for feature in source:
                if not feature.get("geometry"):
                    raise StreamAssemblyError(
                        f"coverage validation found an empty feature geometry: {source_path}"
                    )
                geometry = shape(feature["geometry"])
                if geometry.is_empty or not geometry.is_valid or geometry.area <= 0:
                    raise StreamAssemblyError(
                        f"coverage validation found an invalid feature geometry: {source_path}"
                    )
                feature_count += 1
                # Kahan summation prevents millions of small polygon areas
                # from fabricating an overlap through floating-point drift.
                area = float(geometry.area)
                corrected = area - feature_area_compensation
                accumulated = feature_area + corrected
                feature_area_compensation = (accumulated - feature_area) - corrected
                feature_area = accumulated
                yield geometry

    with fiona.open(source_path, layer=layer) as source:
        source_crs = source.crs_wkt or source.crs
    if not source_crs:
        raise StreamAssemblyError("coverage validation source CRS is missing")
    range_geometry = extract_range_mask_geometry(spec, source_crs)
    if range_geometry is None or range_geometry.is_empty:
        # Runs created before exact-range snapshots existed are still readable,
        # but must never be advertised as coverage-verified.
        return {
            "schema_version": 1,
            "status": "skipped_legacy_no_range",
            "hard_gate_applied": False,
            "reason": "frozen_exact_range_missing",
            "feature_count": 0,
            "range_area_m2": 0.0,
            "gap_area_m2": 0.0,
            "overlap_area_m2": 0.0,
            "outside_area_m2": 0.0,
            "area_tolerance_m2": 0.0,
            "gap_area_source_units2": 0.0,
            "overlap_area_source_units2": 0.0,
            "outside_area_source_units2": 0.0,
            "area_tolerance_source_units2": 0.0,
            "area_scale_method": "not_available",
            "union_method": "not_run",
        }

    union_geometry = balanced_geometry_union(geometries())
    range_area = float(range_geometry.area)
    union_area = float(union_geometry.area)
    if range_area <= 0:
        raise StreamAssemblyError("coverage validation range has no positive area")
    gap_area = float(range_geometry.difference(union_geometry).area)
    outside_area = float(union_geometry.difference(range_geometry).area)
    overlap_area = max(0.0, feature_area - union_area)

    transform = spec.get("raster", {}).get("transform") or []
    if len(transform) < 6:
        raise StreamAssemblyError("coverage validation requires the raster affine")
    affine_area = abs(
        float(transform[0]) * float(transform[4])
        - float(transform[1]) * float(transform[3])
    )
    if not math.isfinite(affine_area) or affine_area <= 0:
        raise StreamAssemblyError(
            "coverage validation raster affine has no positive area"
        )
    config = dict(spec.get("coverage_validation") or {})
    tolerance_pixels = float(config.get("area_tolerance_pixels", 0.01))
    if not math.isfinite(tolerance_pixels) or not 0.0 <= tolerance_pixels <= 1.0:
        raise StreamAssemblyError(
            "coverage_validation.area_tolerance_pixels must be between 0 and 1"
        )
    numerical_tolerance = min(range_area * 1.0e-10, affine_area * 0.25)
    tolerance = max(affine_area * tolerance_pixels, numerical_tolerance)
    ground_scale = ground_area_scale(source_crs, range_geometry.centroid.y)
    passed = all(value <= tolerance for value in (gap_area, overlap_area, outside_area))
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "hard_gate_applied": True,
        "feature_count": int(feature_count),
        "range_area_m2": float(range_area * ground_scale),
        "gap_area_m2": float(gap_area * ground_scale),
        "overlap_area_m2": float(overlap_area * ground_scale),
        "outside_area_m2": float(outside_area * ground_scale),
        "area_tolerance_m2": float(tolerance * ground_scale),
        "gap_area_source_units2": gap_area,
        "overlap_area_source_units2": overlap_area,
        "outside_area_source_units2": outside_area,
        "area_tolerance_source_units2": tolerance,
        "area_scale_method": "local_crs_ground_scale_v1",
        "union_method": "balanced_unary_union_2048_v1",
    }


def publish_coverage_validation(
    append_event: AppendCoverageEvent,
    run_id: str,
    stream_id: str,
    report: Mapping[str, Any],
) -> None:
    payload = dict(report)
    status = str(payload.get("status") or "unknown")
    append_event(
        run_id,
        "stream_coverage_validation",
        level="error" if status == "failed" else "info",
        stream_id=stream_id,
        message=str(payload.get("status") or "unknown"),
        payload=payload,
    )
    print(
        json.dumps(
            {
                "event": "stream_coverage_validation",
                "run_id": str(run_id),
                "stream_id": str(stream_id),
                **payload,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )
