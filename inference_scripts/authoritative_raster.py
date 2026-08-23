"""Build the one authoritative cleaned class raster for a Partition Core.

The inference workers produce probabilities on a halo around each Core.  V3
fragmentation repair is deliberately performed here, before any spatial unit
is polygonized.  Only the Core crop is published; neighbouring units therefore
read one non-overlapping, cleaned classification rather than independently
repairing their own temporary crops.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from affine import Affine
from rasterio.features import geometry_mask
from shapely.geometry import mapping

from deployment_config import CLASS_ORDER
from fragmentation_v3 import policy_snapshot, production_policy
from small_component_regularizer import (
    physical_pixel_area_m2,
    regularize_small_components,
)


class AuthoritativeRasterError(RuntimeError):
    pass


def _window(value: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return tuple(int(value[key]) for key in ("x0", "y0", "x1", "y1"))


def apply_range_mask_to_core(
    arrays: Mapping[str, np.ndarray],
    partition: Mapping[str, Any],
    *,
    global_transform: Affine,
    range_geometry: Any | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Set published Core pixels outside the exact range to existing nodata.

    Halo probabilities remain available for inference and boundary context. The
    non-overlapping Core rasters are the authoritative polygonization inputs,
    so class and confidence receive their documented nodata values here.
    """

    result = dict(arrays)
    core = _window(partition["core_window"])
    core_shape = (core[3] - core[1], core[2] - core[0])
    if np.asarray(result["core_mask"]).shape != core_shape:
        raise AuthoritativeRasterError("Core mask shape does not match Partition")
    mask = np.asarray(result["core_mask"], dtype=np.int16).copy()
    confidence = np.asarray(result["core_confidence"], dtype=np.float32).copy()
    if confidence.shape != core_shape:
        raise AuthoritativeRasterError("Core confidence shape does not match Partition")
    if range_geometry is None:
        inside = np.ones(core_shape, dtype=bool)
    else:
        if range_geometry.is_empty:
            raise AuthoritativeRasterError("range geometry is empty")
        core_transform = global_transform * Affine.translation(core[0], core[1])
        inside = geometry_mask(
            [mapping(range_geometry)],
            out_shape=core_shape,
            transform=core_transform,
            invert=True,
        )

    gap = inside & (mask < 0)
    invalid_confidence = inside & (
        ~np.isfinite(confidence) | (confidence < 0.0)
    )
    gap_count = int(np.count_nonzero(gap))
    invalid_confidence_count = int(np.count_nonzero(invalid_confidence))
    if gap_count or invalid_confidence_count:
        raise AuthoritativeRasterError(
            "authoritative Core has an unassigned pixel inside the owned "
            "research range: "
            f"gaps={gap_count}, invalid_confidence={invalid_confidence_count}"
        )

    mask[~inside] = -1
    confidence[~inside] = -1.0
    result["core_mask"] = mask
    result["core_confidence"] = confidence
    outside_count = int(np.count_nonzero(~inside))
    return result, {
        "range_mask_applied": range_geometry is not None,
        "range_masked_pixel_count": outside_count,
        "coverage_validation": {
            "status": "passed",
            "owned_pixel_count": int(np.count_nonzero(inside)),
            "gap_pixel_count": 0,
            "invalid_confidence_pixel_count": 0,
            "outside_pixel_count": outside_count,
        },
    }


def regularize_partition_core(
    arrays: Mapping[str, np.ndarray],
    partition: Mapping[str, Any],
    *,
    global_transform: Affine,
    crs: str,
    range_geometry: Any | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return a copy whose ``core_mask`` is V3-cleaned using its Halo.

    The class-change budget is measured only inside the published Core.  Halo
    pixels provide neighbourhood context, but never become independently
    published classifications.  Passing confidence (and not probability
    vectors) keeps the V3 high-confidence guard active.
    """

    probabilities = np.asarray(arrays["halo_probabilities"], dtype=np.float32)
    weights = np.asarray(arrays["halo_weights"], dtype=np.float32)
    if probabilities.ndim != 3 or probabilities.shape[0] != len(CLASS_ORDER):
        raise AuthoritativeRasterError("halo probabilities must be [14,H,W]")
    if weights.shape != probabilities.shape[1:]:
        raise AuthoritativeRasterError("halo weights do not match probabilities")
    halo = _window(partition["halo_window"])
    core = _window(partition["core_window"])
    if probabilities.shape[1:] != (halo[3] - halo[1], halo[2] - halo[0]):
        raise AuthoritativeRasterError("halo probability shape does not match Partition")
    if not (halo[0] <= core[0] < core[2] <= halo[2] and halo[1] <= core[1] < core[3] <= halo[3]):
        raise AuthoritativeRasterError("Partition Core must be contained by its Halo")

    valid = weights > 0
    labels = np.full(valid.shape, -1, dtype=np.int16)
    confidence = np.full(valid.shape, -1.0, dtype=np.float32)
    if np.any(valid):
        labels[valid] = probabilities[:, valid].argmax(axis=0).astype(np.int16)
        confidence[valid] = probabilities[:, valid].max(axis=0).astype(np.float32)
    budget = np.zeros(valid.shape, dtype=bool)
    y0, y1 = core[1] - halo[1], core[3] - halo[1]
    x0, x1 = core[0] - halo[0], core[2] - halo[0]
    budget[y0:y1, x0:x1] = True
    halo_transform = global_transform * Affine.translation(halo[0], halo[1])
    pixel_area_m2 = physical_pixel_area_m2(
        halo_transform,
        crs,
        height=probabilities.shape[1],
        width=probabilities.shape[2],
    )
    cleaned, report = regularize_small_components(
        labels,
        class_codes=CLASS_ORDER,
        pixel_area_m2=pixel_area_m2,
        policy=production_policy(),
        valid_mask=valid,
        confidence=confidence,
        class_budget_mask=budget,
    )
    result = dict(arrays)
    result["core_mask"] = cleaned[y0:y1, x0:x1].astype(np.int16, copy=False)
    result["core_confidence"] = confidence[y0:y1, x0:x1].astype(
        np.float32, copy=False
    )
    result, range_report = apply_range_mask_to_core(
        result,
        partition,
        global_transform=global_transform,
        range_geometry=range_geometry,
    )
    report = {
        **report,
        **range_report,
        "authority": "partition_halo_v3_core_publish_v1",
        "policy": dict(policy_snapshot()),
        "pixel_area_m2": float(pixel_area_m2),
        "halo_window": {key: int(value) for key, value in partition["halo_window"].items()},
        "core_window": {key: int(value) for key, value in partition["core_window"].items()},
    }
    return result, report


def core_mask_tags(report: Mapping[str, Any]) -> dict[str, str]:
    """Serialize the bounded provenance needed to identify a Core authority."""

    policy = dict(report.get("policy") or {})
    halo = dict(report.get("halo_window") or {})
    core = dict(report.get("core_window") or {})
    coverage = dict(report.get("coverage_validation") or {})
    margins = (
        int(core.get("x0", 0)) - int(halo.get("x0", 0)),
        int(core.get("y0", 0)) - int(halo.get("y0", 0)),
        int(halo.get("x1", 0)) - int(core.get("x1", 0)),
        int(halo.get("y1", 0)) - int(core.get("y1", 0)),
    )
    return {
        "classification_authority": str(report.get("authority") or ""),
        "fragmentation_policy_id": str(policy.get("policy_id") or "disabled"),
        "fragmentation_policy_version": str(
            policy.get("policy_version") or "disabled"
        ),
        "fragmentation_halo_buffer_px": str(max(0, min(margins))),
        "fragmentation_changed_pixel_count": str(
            int(report.get("changed_pixel_count", 0))
        ),
        "coverage_validation_status": str(coverage.get("status") or "unknown"),
        "coverage_owned_pixel_count": str(
            int(coverage.get("owned_pixel_count", 0))
        ),
        "coverage_gap_pixel_count": str(int(coverage.get("gap_pixel_count", 0))),
        "coverage_outside_pixel_count": str(
            int(coverage.get("outside_pixel_count", 0))
        ),
    }
