"""Experimental multiclass subpixel vectorization from a probability tile."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import fiona
import numpy as np
import rasterio
import shapely
from rasterio import features
from rasterio_compat import quiet_deprecated_memory_driver
from scipy.ndimage import gaussian_filter, map_coordinates
from shapely import affinity
from shapely.geometry import LineString, box, shape


CLASS_CODES = (12, 13, 21, 31, 32, 33, 43, 51, 52, 53, 54, 61, 62, 71)
LAYER_NAME = "semantic_polygons"


def _edge_crossing(
    probabilities,
    first,
    second,
    first_class,
    second_class,
    interpolation_strength=1.0,
):
    r0, c0 = first
    r1, c1 = second
    d0 = float(
        probabilities[first_class, r0, c0]
        - probabilities[second_class, r0, c0]
    )
    d1 = float(
        probabilities[first_class, r1, c1]
        - probabilities[second_class, r1, c1]
    )
    denominator = d0 - d1
    fraction = 0.5 if abs(denominator) < 1e-12 else d0 / denominator
    fraction = 0.5 + float(interpolation_strength) * (fraction - 0.5)
    fraction = float(np.clip(fraction, 0.02, 0.98))
    return (
        c0 + 0.5 + fraction * (c1 - c0),
        r0 + 0.5 + fraction * (r1 - r0),
    )


def _boundary_segments(probabilities, labels, interpolation_strength=1.0):
    height, width = labels.shape
    segments = []
    crossing_cache = {}

    def get_crossing(first, second):
        key = tuple(sorted((first, second)))
        value = crossing_cache.get(key)
        if value is None:
            first_class = int(labels[key[0]])
            second_class = int(labels[key[1]])
            value = _edge_crossing(
                probabilities,
                key[0],
                key[1],
                first_class,
                second_class,
                interpolation_strength,
            )
            crossing_cache[key] = value
        return value

    edges = (
        ((0, 0), (0, 1)),
        ((0, 1), (1, 1)),
        ((1, 1), (1, 0)),
        ((1, 0), (0, 0)),
    )
    for row in range(height - 1):
        for col in range(width - 1):
            crossings = []
            for (dr0, dc0), (dr1, dc1) in edges:
                first = (row + dr0, col + dc0)
                second = (row + dr1, col + dc1)
                first_class = int(labels[first])
                second_class = int(labels[second])
                if first_class == second_class:
                    continue
                crossings.append(get_crossing(first, second))
            if len(crossings) == 2:
                if crossings[0] != crossings[1]:
                    segments.append(LineString(crossings))
            elif len(crossings) > 2:
                junction = (
                    float(np.mean([point[0] for point in crossings])),
                    float(np.mean([point[1] for point in crossings])),
                )
                segments.extend(
                    LineString((point, junction))
                    for point in crossings
                    if point != junction
                )

    boundary_specs = (
        ([(0, col) for col in range(width - 1)], (0.0, -1.0)),
        ([(height - 1, col) for col in range(width - 1)], (0.0, 1.0)),
        ([(row, 0) for row in range(height - 1)], (-1.0, 0.0)),
        ([(row, width - 1) for row in range(height - 1)], (1.0, 0.0)),
    )
    for starts, outward in boundary_specs:
        for row, col in starts:
            if outward[0]:
                first = (row, col)
                second = (row + 1, col)
            else:
                first = (row, col)
                second = (row, col + 1)
            first_class = int(labels[first])
            second_class = int(labels[second])
            if first_class == second_class:
                continue
            crossing_point = get_crossing(first, second)
            destination = (
                0.0 if outward[0] < 0 else width if outward[0] > 0 else crossing_point[0],
                0.0 if outward[1] < 0 else height if outward[1] > 0 else crossing_point[1],
            )
            segments.append(LineString((crossing_point, destination)))
    return segments


def _sample_class(probabilities, geometry):
    point = geometry.representative_point()
    coordinates = np.asarray([[point.y - 0.5], [point.x - 0.5]])
    values = np.asarray(
        [
            map_coordinates(channel, coordinates, order=1, mode="nearest")[0]
            for channel in probabilities
        ]
    )
    return int(values.argmax())


def subpixel_coverage(probabilities, sigma=0.0, interpolation_strength=1.0):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim != 3 or probabilities.shape[0] != 14:
        raise ValueError("probabilities must have shape [14,H,W]")
    if sigma > 0:
        probabilities = gaussian_filter(
            probabilities, sigma=(0.0, float(sigma), float(sigma)), mode="nearest"
        )
        probabilities /= np.maximum(
            probabilities.sum(axis=0, keepdims=True), 1e-12
        )
    labels = probabilities.argmax(axis=0).astype(np.int16)
    height, width = labels.shape
    extent = box(0.0, 0.0, float(width), float(height))
    linework = _boundary_segments(
        probabilities, labels, interpolation_strength=interpolation_strength
    )
    network = shapely.unary_union([extent.boundary, *linework])
    noded = shapely.node(network)
    faces = [
        polygon
        for polygon in shapely.get_parts(
            shapely.polygonize(shapely.get_parts(noded))
        )
        if polygon.area > 1e-8 and extent.covers(polygon.representative_point())
    ]
    by_class = {}
    for face in faces:
        by_class.setdefault(_sample_class(probabilities, face), []).append(face)
    output = []
    for class_index, parts in by_class.items():
        merged = shapely.union_all(np.asarray(parts, dtype=object))
        for polygon in shapely.get_parts(merged):
            if polygon.geom_type == "Polygon" and polygon.area > 1e-8:
                output.append((class_index, polygon))
    return output


def _raw_coverage(labels):
    with quiet_deprecated_memory_driver():
        return [
            (int(value), shape(geometry))
            for geometry, value in features.shapes(
                labels.astype(np.int16), transform=rasterio.Affine.identity()
            )
        ]


def _simplify_records(records, tolerance):
    values = np.asarray([geometry for _class, geometry in records], dtype=object)
    simplified = np.asarray(
        shapely.coverage_simplify(
            values,
            tolerance=float(tolerance),
            simplify_boundary=False,
        ),
        dtype=object,
    )
    return [
        (records[index][0], geometry)
        for index, geometry in enumerate(simplified)
    ]


def _staircase_metrics(records):
    from boundary_ab_validate import _staircase_metrics as metrics

    return metrics([geometry for _class_index, geometry in records])


def _coverage_checks(records, width, height):
    geometries = np.asarray([geometry for _class, geometry in records], dtype=object)
    union = shapely.union_all(geometries)
    extent = box(0, 0, width, height)
    area_sum = float(np.sum(shapely.area(geometries)))
    union_area = float(union.area)
    return {
        "geometry_valid": bool(np.all(shapely.is_valid(geometries))),
        "coverage_valid": bool(shapely.coverage_is_valid(geometries)),
        "gap_area_px2": float(extent.difference(union).area),
        "overlap_area_px2": max(0.0, area_sum - union_area),
        "covered_area_px2": union_area,
    }


def _to_map(geometry, transform):
    return affinity.affine_transform(
        geometry,
        [transform.a, transform.b, transform.d, transform.e, transform.c, transform.f],
    )


def _write_gpkg(path, records, transform, crs, method):
    path = Path(path)
    if path.exists():
        path.unlink()
    schema = {
        "geometry": "Polygon",
        "properties": {
            "class_index": "int",
            "class_code": "int",
            "method": "str",
        },
    }
    with fiona.open(
        path,
        "w",
        driver="GPKG",
        layer=LAYER_NAME,
        schema=schema,
        crs_wkt=crs.to_wkt() if crs else None,
    ) as destination:
        for class_index, geometry in records:
            destination.write({
                "geometry": shapely.geometry.mapping(_to_map(geometry, transform)),
                "properties": {
                    "class_index": class_index,
                    "class_code": CLASS_CODES[class_index],
                    "method": method,
                },
            })


def _qsdk_metrics(records, reference_path, reference_field, raster):
    from boundary_ab_validate import (
        REGIONS,
        _boundary_distances,
        _read_reference,
    )

    reference = _read_reference(
        reference_path,
        reference_field,
        raster.crs,
        tuple(raster.bounds),
        raster.transform,
        raster.width,
        raster.height,
    )
    predicted = {}
    for class_index, geometry in records:
        predicted.setdefault(CLASS_CODES[class_index], []).append(geometry)
    result = {}
    for region, codes in REGIONS.items():
        values = []
        evaluated = []
        for code in codes:
            if not predicted.get(code) or not reference.get(code):
                continue
            distances = _boundary_distances(predicted[code], reference[code])[
                "reference_to_prediction"
            ]
            if distances.size:
                values.append(distances)
                evaluated.append(code)
        pooled = np.concatenate(values) if values else np.asarray([], dtype=float)
        result[region] = {
            "evaluated_class_codes": evaluated,
            "sample_count": int(pooled.size),
            "median_boundary_distance_px": (
                float(np.median(pooled)) if pooled.size else None
            ),
            "p95_boundary_distance_px": (
                float(np.percentile(pooled, 95)) if pooled.size else None
            ),
        }
    return result


def run(
    scores_path,
    raster_path,
    output_dir,
    sigmas=(0.0, 0.7),
    post_tolerance=1.0,
    interpolation_strength=1.0,
    reference_path="",
    reference_field="TDLYDM",
):
    with np.load(scores_path) as cached:
        probabilities = cached["probabilities"].astype(np.float32)
    with rasterio.open(raster_path) as source:
        transform = source.transform
        crs = source.crs
        width, height = source.width, source.height
        raster_profile = {
            "crs": source.crs,
            "bounds": source.bounds,
            "transform": source.transform,
            "width": source.width,
            "height": source.height,
        }
    if probabilities.shape != (14, height, width):
        raise ValueError(
            f"probability/raster shape mismatch: {probabilities.shape}/{height}x{width}"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = probabilities.argmax(axis=0).astype(np.int16)
    raw = _raw_coverage(labels)
    raw_values = np.asarray([geometry for _class, geometry in raw], dtype=object)
    baseline_values = np.asarray(
        shapely.coverage_simplify(
            raw_values, tolerance=1.5, simplify_boundary=False
        ),
        dtype=object,
    )
    baseline = [
        (raw[index][0], geometry)
        for index, geometry in enumerate(baseline_values)
    ]
    variants = {"a_coverage_simplify": baseline}
    for sigma in sigmas:
        name = "b_subpixel" if float(sigma) == 0.0 else f"c_subpixel_sigma_{sigma:g}"
        variants[name] = _simplify_records(
            subpixel_coverage(
                probabilities,
                sigma=float(sigma),
                interpolation_strength=interpolation_strength,
            ),
            post_tolerance,
        )
    report = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "scores": str(Path(scores_path).resolve()),
        "reference_raster": str(Path(raster_path).resolve()),
        "width": width,
        "height": height,
        "post_subpixel_coverage_tolerance_px": float(post_tolerance),
        "interpolation_strength": float(interpolation_strength),
        "variants": {},
    }
    for name, records in variants.items():
        output = output_dir / f"{name}.gpkg"
        _write_gpkg(output, records, transform, crs, name)
        report["variants"][name] = {
            "feature_count": len(records),
            "metrics": _staircase_metrics(records),
            "coverage": _coverage_checks(records, width, height),
            "output": str(output.resolve()),
        }
        if reference_path:
            class RasterInfo:
                pass

            raster = RasterInfo()
            for key, value in raster_profile.items():
                setattr(raster, key, value)
            report["variants"][name]["qsdk"] = _qsdk_metrics(
                records, reference_path, reference_field, raster
            )
    report_path = output_dir / "subpixel_ab_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--raster", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sigmas", default="0,0.7")
    parser.add_argument("--post-tolerance", type=float, default=1.0)
    parser.add_argument("--interpolation-strength", type=float, default=1.0)
    parser.add_argument("--reference", default="")
    parser.add_argument("--reference-field", default="TDLYDM")
    args = parser.parse_args(argv)
    sigmas = tuple(float(value) for value in args.sigmas.split(",") if value.strip())
    report = run(
        args.scores,
        args.raster,
        args.output_dir,
        sigmas,
        post_tolerance=args.post_tolerance,
        interpolation_strength=args.interpolation_strength,
        reference_path=args.reference,
        reference_field=args.reference_field,
    )
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
