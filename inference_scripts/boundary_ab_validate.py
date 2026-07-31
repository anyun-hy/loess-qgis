"""A/B acceptance for raw, coverage-only and shared-boundary polygons vs QSDK."""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import fiona
import numpy as np
import rasterio
import shapely
from rasterio.warp import transform_bounds, transform_geom
from shapely import affinity
from shapely.geometry import box, shape

from boundary_regularizer import _vertex_boundary_displacement


REGIONS = {
    "road_construction": (51, 52, 53, 54, 61, 62),
    "farmland": (12, 13, 21),
    "forest_natural_slope": (31, 32, 33, 43),
}


class BoundaryAcceptanceError(RuntimeError):
    pass


def _affine_parameters(transform):
    return [transform.a, transform.b, transform.d, transform.e, transform.c, transform.f]


def _to_pixel(geometry, transform):
    return affinity.affine_transform(geometry, _affine_parameters(~transform))


def _read_prediction(path, transform, layer_name="semantic_polygons"):
    records = []
    with fiona.open(path, layer=layer_name) as source:
        for feature in source:
            geometry = shape(feature["geometry"])
            if geometry.geom_type != "Polygon" or geometry.is_empty or not geometry.is_valid:
                raise BoundaryAcceptanceError(f"invalid prediction geometry in {path}")
            properties = dict(feature.get("properties") or {})
            records.append({
                "identity": (
                    str(properties.get("object_id") or ""),
                    str(properties.get("part_id") or "000"),
                ),
                "class_code": int(properties["class_code"]),
                "geometry": _to_pixel(geometry, transform),
            })
    return records


def _read_reference(path, field, raster_crs, raster_bounds, transform, width, height):
    by_class = {code: [] for codes in REGIONS.values() for code in codes}
    clip = box(0, 0, int(width), int(height))
    with fiona.open(path) as source:
        source_crs = source.crs_wkt or source.crs
        if not source_crs:
            raise BoundaryAcceptanceError("QSDK reference has no CRS")
        if field not in (source.schema.get("properties") or {}):
            raise BoundaryAcceptanceError(f"QSDK field is missing: {field}")
        query_bounds = transform_bounds(
            raster_crs,
            source_crs,
            *raster_bounds,
            densify_pts=21,
        )
        for feature in source.filter(bbox=query_bounds):
            try:
                code = int((feature.get("properties") or {}).get(field))
            except (TypeError, ValueError):
                continue
            if code not in by_class or not feature.get("geometry"):
                continue
            transformed = transform_geom(
                source_crs, raster_crs, feature["geometry"], precision=-1
            )
            geometry = _to_pixel(shape(transformed), transform)
            if not geometry.is_valid:
                geometry = shapely.make_valid(geometry)
            geometry = geometry.intersection(clip)
            if geometry.is_empty:
                continue
            by_class[code].append(geometry)
    return by_class


def _parts(geometry):
    if geometry.geom_type in ("LineString", "LinearRing"):
        return [geometry]
    return [part for part in shapely.get_parts(geometry) if part.geom_type == "LineString"]


def _staircase_metrics(geometries):
    coordinate_count = 0
    segment_count = 0
    short_segments = 0
    staircase_turns = 0
    total_length = 0.0
    for geometry in geometries:
        for line in _parts(geometry.boundary):
            coordinates = np.asarray(line.coords, dtype=np.float64)
            if len(coordinates) < 2:
                continue
            coordinate_count += len(coordinates)
            vectors = np.diff(coordinates, axis=0)
            lengths = np.hypot(vectors[:, 0], vectors[:, 1])
            valid = lengths > 1e-9
            vectors = vectors[valid]
            lengths = lengths[valid]
            segment_count += len(lengths)
            total_length += float(lengths.sum())
            short_segments += int(np.count_nonzero(lengths <= 1.5 + 1e-9))
            if len(lengths) < 2:
                continue
            horizontal = np.abs(vectors[:, 1]) <= 0.15
            vertical = np.abs(vectors[:, 0]) <= 0.15
            short_pair = (lengths[:-1] <= 2.0) & (lengths[1:] <= 2.0)
            changes_axis = (
                (horizontal[:-1] & vertical[1:])
                | (vertical[:-1] & horizontal[1:])
            )
            staircase_turns += int(np.count_nonzero(short_pair & changes_axis))
    return {
        "coordinate_count": int(coordinate_count),
        "segment_count": int(segment_count),
        "short_segment_ratio": (
            float(short_segments / segment_count) if segment_count else 0.0
        ),
        "staircase_turn_count": int(staircase_turns),
        "staircase_turn_density_per_100px": (
            float(staircase_turns * 100.0 / total_length) if total_length else 0.0
        ),
        "boundary_length_px": float(total_length),
    }


def _sample_coordinates(boundary, sample_count=3000):
    lines = _parts(boundary)
    lengths = np.asarray([line.length for line in lines], dtype=float)
    total = float(lengths.sum())
    if not lines or total <= 0:
        return np.empty((0, 2), dtype=float)
    allocations = np.maximum(2, np.rint(sample_count * lengths / total).astype(int))
    coordinates = []
    for line, count in zip(lines, allocations):
        distances = np.linspace(0.0, line.length, int(count), endpoint=False)
        points = shapely.line_interpolate_point(line, distances)
        coordinates.append(shapely.get_coordinates(points))
    result = np.concatenate(coordinates)
    if len(result) > sample_count:
        indexes = np.linspace(0, len(result) - 1, sample_count, dtype=int)
        result = result[indexes]
    return result


def _boundary_distances(predicted, reference):
    predicted_union = shapely.union_all(np.asarray(predicted, dtype=object))
    reference_union = shapely.union_all(np.asarray(reference, dtype=object))
    predicted_boundary = predicted_union.boundary
    reference_boundary = reference_union.boundary
    predicted_coordinates = _sample_coordinates(predicted_boundary)
    reference_coordinates = _sample_coordinates(reference_boundary)
    if not len(predicted_coordinates) or not len(reference_coordinates):
        return {
            "reference_to_prediction": np.asarray([], dtype=float),
            "prediction_to_reference": np.asarray([], dtype=float),
            "sampled_hausdorff": 0.0,
        }
    predicted_points = shapely.points(predicted_coordinates)
    reference_points = shapely.points(reference_coordinates)
    prediction_to_reference = np.asarray(
        shapely.distance(predicted_points, reference_boundary), dtype=float
    )
    reference_to_prediction = np.asarray(
        shapely.distance(reference_points, predicted_boundary), dtype=float
    )
    prediction_to_reference = prediction_to_reference[
        np.isfinite(prediction_to_reference)
    ]
    reference_to_prediction = reference_to_prediction[
        np.isfinite(reference_to_prediction)
    ]
    symmetric = np.concatenate([prediction_to_reference, reference_to_prediction])
    return {
        "reference_to_prediction": reference_to_prediction,
        "prediction_to_reference": prediction_to_reference,
        "sampled_hausdorff": float(np.max(symmetric)) if symmetric.size else 0.0,
    }


def _coverage_simplify_baseline(geometries, tolerance):
    values = np.asarray(geometries, dtype=object)
    simplified = np.asarray(
        shapely.coverage_simplify(
            values,
            tolerance=float(tolerance),
            simplify_boundary=False,
        ),
        dtype=object,
    )
    if simplified.shape != values.shape:
        raise BoundaryAcceptanceError("coverage_simplify baseline changed feature count")
    if np.any(shapely.is_empty(simplified)) or not np.all(shapely.is_valid(simplified)):
        raise BoundaryAcceptanceError("coverage_simplify baseline produced invalid geometry")
    maximum_displacement = max(
        (
            _vertex_boundary_displacement(before, after)
            for before, after in zip(values, simplified)
        ),
        default=0.0,
    )
    return simplified, float(maximum_displacement)


def _region_report(
    name,
    codes,
    raw,
    baseline,
    formal,
    reference,
    noninferiority_margin_px,
):
    variants = {"raw": raw, "coverage_simplify": baseline, "shared_boundary": formal}
    variant_geometries = {
        variant: [
            item["geometry"] for item in records if item["class_code"] in codes
        ]
        for variant, records in variants.items()
    }
    report = {
        "region": name,
        "class_codes": list(codes),
        "metrics": {
            variant: _staircase_metrics(geometries)
            for variant, geometries in variant_geometries.items()
        },
        "qsdk": {},
        "qsdk_by_class": {},
        "area_by_class_px2": {},
    }
    pooled = {"coverage_simplify": [], "shared_boundary": []}
    hausdorff = {"coverage_simplify": [], "shared_boundary": []}
    evaluated_codes = []
    for code in codes:
        report["area_by_class_px2"][str(code)] = {
            variant: float(sum(
                item["geometry"].area for item in records if item["class_code"] == code
            ))
            for variant, records in variants.items()
        }
        reference_geometries = reference.get(code) or []
        baseline_geometries = [
            item["geometry"] for item in baseline if item["class_code"] == code
        ]
        formal_geometries = [
            item["geometry"] for item in formal if item["class_code"] == code
        ]
        if not reference_geometries or not baseline_geometries or not formal_geometries:
            continue
        evaluated_codes.append(code)
        report["qsdk_by_class"][str(code)] = {}
        for variant, geometries in (
            ("coverage_simplify", baseline_geometries),
            ("shared_boundary", formal_geometries),
        ):
            directional = _boundary_distances(geometries, reference_geometries)
            distances = directional["reference_to_prediction"]
            reverse = directional["prediction_to_reference"]
            maximum = directional["sampled_hausdorff"]
            pooled[variant].append(distances)
            hausdorff[variant].append(maximum)
            report["qsdk_by_class"][str(code)][variant] = {
                "sample_count": int(distances.size),
                "median_boundary_distance_px": float(np.median(distances)),
                "p95_boundary_distance_px": float(np.percentile(distances, 95)),
                "prediction_to_qsdk_median_px": float(np.median(reverse)),
                "prediction_to_qsdk_p95_px": float(np.percentile(reverse, 95)),
                "hausdorff_distance_px": maximum,
            }
    report["evaluated_class_codes"] = evaluated_codes
    for variant in ("coverage_simplify", "shared_boundary"):
        values = np.concatenate(pooled[variant]) if pooled[variant] else np.asarray([])
        report["qsdk"][variant] = {
            "sample_count": int(values.size),
            "median_boundary_distance_px": float(np.median(values)) if values.size else None,
            "p95_boundary_distance_px": float(np.percentile(values, 95)) if values.size else None,
            "hausdorff_distance_px": max(hausdorff[variant]) if hausdorff[variant] else None,
        }
    baseline_metrics = report["metrics"]["coverage_simplify"]
    formal_metrics = report["metrics"]["shared_boundary"]
    baseline_qsdk = report["qsdk"]["coverage_simplify"]
    formal_qsdk = report["qsdk"]["shared_boundary"]
    epsilon = 1e-9
    margin = float(noninferiority_margin_px)
    checks = {
        "has_qsdk_comparison": bool(evaluated_codes),
        "staircase_density_lower": (
            formal_metrics["staircase_turn_density_per_100px"]
            < baseline_metrics["staircase_turn_density_per_100px"] - epsilon
        ),
        "short_segment_ratio_lower": (
            formal_metrics["short_segment_ratio"]
            < baseline_metrics["short_segment_ratio"] - epsilon
        ),
        "qsdk_median_within_margin": bool(
            baseline_qsdk["median_boundary_distance_px"] is not None
            and formal_qsdk["median_boundary_distance_px"]
            <= baseline_qsdk["median_boundary_distance_px"] + margin + epsilon
        ),
        "qsdk_p95_within_margin": bool(
            baseline_qsdk["p95_boundary_distance_px"] is not None
            and formal_qsdk["p95_boundary_distance_px"]
            <= baseline_qsdk["p95_boundary_distance_px"] + margin + epsilon
        ),
    }
    report["qsdk_noninferiority_margin_px"] = margin
    report["checks"] = checks
    report["passed"] = all(checks.values())
    return report


def validate(raw_path, formal_path, mask_path, report_path, reference_path, output_path, field):
    with rasterio.open(mask_path) as raster:
        transform = raster.transform
        raster_crs = raster.crs
        raster_bounds = tuple(raster.bounds)
        width, height = raster.width, raster.height
    if raster_crs is None:
        raise BoundaryAcceptanceError("mask mosaic has no CRS")
    raw = _read_prediction(raw_path, transform)
    formal = _read_prediction(formal_path, transform)
    with open(report_path, "r", encoding="utf-8") as handle:
        boundary_report = json.load(handle)
    config = dict(boundary_report.get("config") or {})
    baseline_tolerance = 1.5
    noninferiority_margin_px = float(config["qsdk_noninferiority_margin_px"])
    baseline_geometries, maximum_displacement = _coverage_simplify_baseline(
        [item["geometry"] for item in raw], baseline_tolerance
    )
    baseline = [
        {**item, "geometry": geometry}
        for item, geometry in zip(raw, baseline_geometries)
    ]
    reference = _read_reference(
        reference_path, field, raster_crs, raster_bounds, transform, width, height
    )
    regions = {
        name: _region_report(
            name,
            codes,
            raw,
            baseline,
            formal,
            reference,
            noninferiority_margin_px,
        )
        for name, codes in REGIONS.items()
    }
    hard_checks = dict((boundary_report.get("validation") or {}).get("checks") or {})
    hard_checks["boundary_report_passed"] = bool(
        boundary_report.get("status") == "passed"
        and (boundary_report.get("validation") or {}).get("passed") is True
    )
    result = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "passed" if all(hard_checks.values()) and all(
            item["passed"] for item in regions.values()
        ) else "failed",
        "raw": str(Path(raw_path).resolve()),
        "formal": str(Path(formal_path).resolve()),
        "mask_mosaic": str(Path(mask_path).resolve()),
        "boundary_report": str(Path(report_path).resolve()),
        "reference": str(Path(reference_path).resolve()),
        "reference_field": field,
        "raster_size": {"width": width, "height": height},
        "coverage_simplify": {
            "effective_tolerance_px": baseline_tolerance,
            "maximum_displacement_px": float(maximum_displacement),
        },
        "formal_method": str(boundary_report.get("method") or ""),
        "qsdk_noninferiority_margin_px": noninferiority_margin_px,
        "distance_method": {
            "sampling": "fixed equal-arclength samples per class and direction",
            "samples_per_direction_and_class": 3000,
            "primary_median_p95": "paired QSDK reference boundary points to prediction",
            "reverse_distance": "reported per class but not used as the paired acceptance gate",
            "hausdorff": "maximum sampled symmetric boundary distance",
        },
        "hard_checks": hard_checks,
        "regions": regions,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate regularized boundaries against QSDK")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--formal", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--boundary-report", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-field", default="TDLYDM")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = validate(
            args.raw,
            args.formal,
            args.mask,
            args.boundary_report,
            args.reference,
            args.output,
            args.reference_field,
        )
    except Exception as exc:
        print(f"[boundary_ab_validate] ERROR: {exc}", file=sys.stderr)
        return 2
    print("[boundary_ab_validate] " + json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0 if result["status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
