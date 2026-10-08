"""Convert a frozen smoothing policy to raster-specific pixel distances."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from rasterio.crs import CRS

from labeling_tool.shared.contracts.boundary_resolution import (
    normalize_resolution_policy,
)


def _metric_pixel_matrix(
    raster: Mapping[str, Any], extent: Mapping[str, Any], padding_px: float
) -> tuple[np.ndarray, str]:
    values = [float(v) for v in raster["transform"]]
    if len(values) not in {6, 9} or not all(math.isfinite(v) for v in values):
        raise ValueError("raster transform must contain finite affine coefficients")
    a, b, _c, d, e, _f = values[:6]
    matrix = np.array([[a, b], [d, e]], dtype=float)
    if a * e - b * d == 0:
        raise ValueError("raster transform has zero pixel area")
    crs = CRS.from_user_input(raster["crs"])
    if not (crs.is_projected or crs.is_geographic):
        raise ValueError("resolution adaptation needs a projected or geographic CRS")
    unit_x = unit_y = float(crs.units_factor[1])
    if crs.is_projected and crs.to_epsg() != 3857:
        return np.diag([unit_x, unit_y]) @ matrix, "projected_crs_meters"

    # Use the whole frozen processing extent, so adjacent Units receive the
    # same parameters. Include the permitted output displacement at its edges.
    pad_y = math.hypot(d, e) * padding_px
    ymin, ymax = float(extent["ymin"]) - pad_y, float(extent["ymax"]) + pad_y
    if not all(math.isfinite(v) for v in (ymin, ymax)) or ymin >= ymax:
        raise ValueError("processing extent must have finite ordered y bounds")
    definition = crs.to_dict(projjson=True)
    base = definition.get("base_crs", definition)
    datum = base.get("datum") or base.get("datum_ensemble") or {}
    ellipsoid = datum.get("ellipsoid") or {}
    major = float(ellipsoid.get("semi_major_axis", ellipsoid.get("radius", math.nan)))
    if "semi_minor_axis" in ellipsoid:
        minor = float(ellipsoid["semi_minor_axis"])
    elif "radius" in ellipsoid or ellipsoid.get("inverse_flattening") == 0:
        minor = major
    else:
        flattening = float(ellipsoid.get("inverse_flattening", math.nan))
        minor = major * (1 - 1 / flattening)
    if not (0 < minor <= major < math.inf):
        raise ValueError("unsupported raster ellipsoid")
    eccentricity = 1 - (minor / major) ** 2
    if crs.to_epsg() == 3857:
        # EPSG:3857 map metres overstate ground distances away from the equator.
        # Both ground scale factors are largest at the latitude nearest zero.
        nearest_y = 0.0 if ymin <= 0 <= ymax else min(abs(ymin), abs(ymax))
        latitude = math.atan(math.sinh(min(nearest_y / major, 350.0)))
        denominator = 1 - eccentricity * math.sin(latitude) ** 2
        east = math.cos(latitude) / math.sqrt(denominator)
        north = (1 - eccentricity) * math.cos(latitude) / denominator ** 1.5
        return np.diag([east, north]) @ matrix, "web_mercator_ground_scale_bound"

    # Geographic coordinates are angular, never metres. Bound the ellipsoid's
    # parallel and meridian lengths throughout the processing latitude range.
    lo, hi = ymin * unit_y, ymax * unit_y
    if lo <= -math.pi / 2 or hi >= math.pi / 2:
        raise ValueError("geographic processing extent reaches beyond a pole")
    nearest = 0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi))
    farthest = max(abs(lo), abs(hi))
    east = major * math.cos(nearest) / math.sqrt(
        1 - eccentricity * math.sin(nearest) ** 2
    )
    north = major * (1 - eccentricity) / (
        1 - eccentricity * math.sin(farthest) ** 2
    ) ** 1.5
    return np.diag([east * unit_x, north * unit_y]) @ matrix, "geographic_ground_scale_bound"


def resolve_boundary_parameters(
    value: Mapping[str, Any],
    *,
    raster: Mapping[str, Any],
    processing_extent: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Resolve once from frozen inputs; old Runs keep their pixel parameters."""

    resolved = dict(value)
    if "resolution_adaptation" not in value:
        return resolved, None
    policy = normalize_resolution_policy(value["resolution_adaptation"])
    if not policy["enabled"]:
        return resolved, {"enabled": False}
    defaults = {
        "smoothing_factor": 1.0,
        "curve_sampling_spacing_px": 0.5,
        "max_chord_error_px": 0.25,
        "max_segment_arc_length_px": 8.0,
        "max_deviation_px": 1.0,
    }
    requested = {key: float(value.get(key, default)) for key, default in defaults.items()}
    if any(not math.isfinite(v) or v <= 0 for v in requested.values()):
        raise ValueError("adaptive boundary parameters must be finite and positive")
    matrix, basis = _metric_pixel_matrix(raster, processing_extent, requested["max_deviation_px"])
    # Largest singular value protects all directions, including sheared and
    # non-square pixels; averaging x/y sizes can violate the metre budget.
    scale = float(np.linalg.svd(matrix, compute_uv=False)[0])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("raster has no positive metric pixel scale")
    factor = float(policy["reference_resolution_m"]) / scale
    for key, original in requested.items():
        resolved[key] = original * (
            factor if key == "max_segment_arc_length_px" else min(1.0, factor)
        )
    report = {
        **policy,
        "policy": "reference_meters_with_pixel_error_caps_v1",
        "metric_basis": basis,
        "column_size_m": float(np.linalg.norm(matrix[:, 0])),
        "row_size_m": float(np.linalg.norm(matrix[:, 1])),
        "maximum_pixel_scale_m": scale,
        "scale_factor": factor,
        "reference_parameters_px": requested,
        "effective_parameters_px": {key: resolved[key] for key in defaults},
        "max_deviation_m": resolved["max_deviation_px"] * scale,
        "metric_scope": "frozen_processing_extent",
    }
    return resolved, report
