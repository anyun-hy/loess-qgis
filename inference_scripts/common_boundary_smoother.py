"""Fit each polygon-pair divider once and reuse it on both polygon sides."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np
from shapely.affinity import affine_transform
from shapely.errors import GEOSException
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.ops import linemerge
from shapely.strtree import STRtree

from polyline_smoother import SmoothingConfig, smooth_polyline


class CommonBoundarySmoothingError(RuntimeError):
    pass


def _pair_validation_failure(
    left_before,
    right_before,
    left_after,
    right_after,
    area_tolerance: float,
    output_transform: tuple[float, ...] | None = None,
) -> str:
    if (
        left_after.is_empty
        or right_after.is_empty
        or not left_after.is_valid
        or not right_after.is_valid
    ):
        return "invalid_polygon"
    if left_after.area <= 0 or right_after.area <= 0:
        return "non_positive_area"
    old_total_area = left_before.area + right_before.area
    new_total_area = left_after.area + right_after.area
    if abs(new_total_area - old_total_area) > area_tolerance:
        return "total_area_changed"
    if output_transform is None:
        return ""

    mapped_before = [
        affine_transform(geometry, output_transform)
        for geometry in (left_before, right_before)
    ]
    mapped_after = [
        affine_transform(geometry, output_transform)
        for geometry in (left_after, right_after)
    ]
    if any(geometry.is_empty or not geometry.is_valid for geometry in mapped_after):
        return "output_crs_invalid_polygon"
    if any(geometry.area <= 0 for geometry in mapped_after):
        return "output_crs_non_positive_area"
    map_area_scale = abs(
        output_transform[0] * output_transform[3]
        - output_transform[1] * output_transform[2]
    )
    old_mapped_area = sum(geometry.area for geometry in mapped_before)
    new_mapped_area = sum(geometry.area for geometry in mapped_after)
    mapped_tolerance = max(
        area_tolerance * map_area_scale,
        old_mapped_area * 1e-9,
        1e-18,
    )
    if abs(new_mapped_area - old_mapped_area) > mapped_tolerance:
        return "output_crs_total_area_changed"
    return ""


def _line_parts(geometry) -> list[LineString]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry]
    if isinstance(geometry, MultiLineString):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        return [part for item in geometry.geoms for part in _line_parts(item)]
    return []


def _polygon_parts(geometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    return []


def _vertex_count(geometry) -> int:
    return sum(
        len(polygon.exterior.coords)
        + sum(len(interior.coords) for interior in polygon.interiors)
        for polygon in _polygon_parts(geometry)
    )


def _merged_line_parts(geometry) -> list[LineString]:
    parts = _line_parts(geometry)
    if len(parts) <= 1:
        return parts
    return _line_parts(linemerge(MultiLineString(parts)))


def _ring_route(points, start_index: int, end_index: int):
    unique = list(points[:-1])
    rotated = unique[start_index:] + unique[:start_index]
    end_position = (end_index - start_index) % len(unique)
    route = rotated[: end_position + 1]
    remainder = rotated[end_position:] + [rotated[0]]
    return route, remainder


def _point_index(points, target, tolerance=1e-6):
    values = np.asarray(points[:-1], dtype=np.float64)
    distances = np.linalg.norm(values - np.asarray(target, dtype=np.float64), axis=1)
    index = int(np.argmin(distances))
    return index if float(distances[index]) <= tolerance else None


def _ring_match_score(points, divider: LineString) -> float:
    ring = LineString(points)
    return max(Point(point).distance(ring) for point in divider.coords)


def _insert_ring_points(points, targets, tolerance=1e-6):
    unique = list(points[:-1])
    values = np.asarray(unique, dtype=np.float64)
    additions = defaultdict(list)
    for target in targets:
        target_value = np.asarray(target, dtype=np.float64)
        if float(np.min(np.linalg.norm(values - target_value, axis=1))) <= tolerance:
            continue
        best = None
        for edge_index, start in enumerate(values):
            end = values[(edge_index + 1) % len(values)]
            delta = end - start
            length_squared = float(np.dot(delta, delta))
            if length_squared == 0:
                continue
            position = float(np.dot(target_value - start, delta) / length_squared)
            if position < 0 or position > 1:
                continue
            projected = start + position * delta
            distance = float(np.linalg.norm(projected - target_value))
            if best is None or distance < best[0]:
                best = (distance, edge_index, position)
        if best is None or best[0] > tolerance:
            raise CommonBoundarySmoothingError(
                "shared divider endpoint is not on its polygon ring"
            )
        additions[best[1]].append((best[2], tuple(target)))

    output = []
    for edge_index, point in enumerate(unique):
        output.append(point)
        for _position, target in sorted(additions.get(edge_index, [])):
            if not np.allclose(output[-1], target, atol=tolerance, rtol=0):
                output.append(target)
    output.append(output[0])
    return output


def _locate_ring_path(points, divider: LineString, fitted: LineString):
    start_index = _point_index(points, divider.coords[0])
    end_index = _point_index(points, divider.coords[-1])
    if start_index is None or end_index is None:
        return None
    if start_index == end_index:
        if not divider.is_ring:
            return None
        fitted_points = list(fitted.coords)
        if not np.allclose(fitted_points[0], fitted_points[-1], atol=1e-6, rtol=0):
            fitted_points.append(fitted_points[0])
        return {
            "score": LineString(points).hausdorff_distance(divider),
            "full_ring": True,
            "fitted_points": fitted_points,
        }
    forward, forward_remainder = _ring_route(points, start_index, end_index)
    reverse, reverse_remainder = _ring_route(points, end_index, start_index)
    forward_score = LineString(forward).hausdorff_distance(divider)
    reverse_score = LineString(reverse).hausdorff_distance(divider)
    point_count = len(points) - 1
    if reverse_score < forward_score:
        route = reverse
        start_index, end_index = end_index, start_index
    else:
        route = forward

    fitted_points = list(fitted.coords)
    if np.linalg.norm(
        np.asarray(fitted_points[0]) - np.asarray(route[0])
    ) > np.linalg.norm(np.asarray(fitted_points[-1]) - np.asarray(route[0])):
        fitted_points.reverse()
    if not np.allclose(fitted_points[0], route[0], atol=1e-6, rtol=0) or not np.allclose(
        fitted_points[-1], route[-1], atol=1e-6, rtol=0
    ):
        return None
    edge_indices = []
    current = start_index
    while current != end_index:
        edge_indices.append(current)
        current = (current + 1) % point_count
        if len(edge_indices) > point_count:
            return None
    return {
        "score": min(forward_score, reverse_score),
        "start_index": start_index,
        "end_index": end_index,
        "edge_indices": edge_indices,
        "fitted_points": fitted_points,
    }


def _apply_ring_replacements(points, located):
    full_rings = [item for item in located if item.get("full_ring")]
    if full_rings:
        if len(located) != 1:
            raise CommonBoundarySmoothingError(
                "a closed shared divider cannot overlap another divider on one ring"
            )
        return full_rings[0]["fitted_points"]

    point_count = len(points) - 1
    covered = {}
    starts = {}
    for item in sorted(located, key=lambda value: len(value["edge_indices"]), reverse=True):
        overlap = [index for index in item["edge_indices"] if index in covered]
        if overlap:
            existing = covered[overlap[0]]
            if set(existing["edge_indices"]) == set(item["edge_indices"]):
                continue
            raise CommonBoundarySmoothingError(
                "shared divider paths overlap within one polygon ring"
            )
        for edge_index in item["edge_indices"]:
            covered[edge_index] = item
        starts[item["start_index"]] = item

    anchor = next(
        (index for index in range(point_count) if index not in covered),
        min(starts),
    )
    output = [points[anchor]]
    current = anchor
    consumed = 0
    while consumed < point_count:
        replacement = starts.get(current)
        if replacement is not None:
            output.extend(replacement["fitted_points"][1:])
            step = len(replacement["edge_indices"])
            current = replacement["end_index"]
            consumed += step
        else:
            current = (current + 1) % point_count
            output.append(points[current])
            consumed += 1
    if not np.allclose(output[0], output[-1], atol=1e-6, rtol=0):
        output.append(output[0])
    return output


def _replace_geometry_dividers(geometry, replacements):
    polygons = _polygon_parts(geometry)
    rings = {}
    for polygon_index, polygon in enumerate(polygons):
        rings[(polygon_index, 0)] = list(polygon.exterior.coords)
        for ring_index, interior in enumerate(polygon.interiors, 1):
            rings[(polygon_index, ring_index)] = list(interior.coords)
    assignments = defaultdict(list)
    for divider, fitted in replacements:
        candidates = []
        for key, points in rings.items():
            score = _ring_match_score(points, divider)
            if score <= 1e-6:
                candidates.append((score, key))
        if not candidates:
            raise CommonBoundarySmoothingError(
                "shared divider was not found in polygon rings"
            )
        _score, key = min(candidates, key=lambda item: item[0])
        assignments[key].append((divider, fitted))
    for key, assigned in assignments.items():
        rings[key] = _insert_ring_points(
            rings[key],
            [
                endpoint
                for divider, _fitted in assigned
                for endpoint in (divider.coords[0], divider.coords[-1])
            ],
        )
        located = []
        for divider, fitted in assigned:
            match = _locate_ring_path(rings[key], divider, fitted)
            if match is None:
                raise CommonBoundarySmoothingError(
                    "shared divider endpoints were not found after ring insertion"
                )
            located.append(match)
        rings[key] = _apply_ring_replacements(rings[key], located)
    rebuilt = []
    for polygon_index, polygon in enumerate(polygons):
        exterior = rings[(polygon_index, 0)]
        interiors = [
            rings[(polygon_index, ring_index)]
            for ring_index in range(1, len(polygon.interiors) + 1)
        ]
        if len(exterior) < 4 or any(len(ring) < 4 for ring in interiors):
            raise CommonBoundarySmoothingError(
                "divider replacement collapsed a polygon ring"
            )
        try:
            rebuilt.append(Polygon(exterior, interiors))
        except (GEOSException, TypeError, ValueError) as error:
            raise CommonBoundarySmoothingError(
                f"divider replacement could not rebuild polygon: {error}"
            ) from error
    polygons = rebuilt
    return polygons[0] if isinstance(geometry, Polygon) else MultiPolygon(polygons)


def smooth_common_boundaries(
    records: Iterable[Mapping[str, Any]],
    config: SmoothingConfig | None = None,
    *,
    output_transform: Iterable[float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Smooth every shared divider once, then rebuild each affected polygon."""

    config = config or SmoothingConfig(min_point_count=4)
    output_transform = (
        tuple(float(value) for value in output_transform)
        if output_transform is not None
        else None
    )
    if output_transform is not None and len(output_transform) != 6:
        raise CommonBoundarySmoothingError(
            "output_transform must contain six affine coefficients"
        )
    source = [dict(record) for record in records]
    if not source:
        raise CommonBoundarySmoothingError("polygon coverage is empty")
    geometries = [record["geometry"] for record in source]
    if any(not isinstance(geometry, (Polygon, MultiPolygon)) for geometry in geometries):
        raise CommonBoundarySmoothingError("all records must contain polygon geometry")
    if any(
        geometry.is_empty or not geometry.is_valid or geometry.area <= 0
        for geometry in geometries
    ):
        raise CommonBoundarySmoothingError(
            "common-divider smoothing requires valid positive-area source polygons"
        )

    tree = STRtree(geometries)
    candidates = []
    unchanged_count = 0

    for left_index, left in enumerate(geometries):
        for value in tree.query(left):
            right_index = int(value)
            if right_index <= left_index:
                continue
            shared = left.boundary.intersection(geometries[right_index].boundary)
            for divider_index, divider in enumerate(_merged_line_parts(shared)):
                if divider.length <= 0 or len(divider.coords) < 2:
                    continue
                result = smooth_polyline(divider.coords, config)
                if result.status != "smoothed":
                    unchanged_count += 1
                    continue
                fitted = LineString(result.points)
                candidates.append(
                    {
                        "left_index": left_index,
                        "right_index": right_index,
                        "divider": divider,
                        "fitted": fitted,
                        "shift": (result.max_deviation, result.mean_deviation),
                        "diagnostic": {
                            "chain_id": len(candidates),
                            "divider_id": f"{left_index}_{right_index}_{divider_index}",
                            "polygon_indices": [left_index, right_index],
                            "method": "cubic_bspline_adaptive",
                            "status": "changed",
                            "max_displacement_px": result.max_deviation,
                            "mean_displacement_px": result.mean_deviation,
                            "point_count_before": result.input_point_count,
                            "point_count_dense": result.dense_point_count,
                            "point_count_after": result.output_point_count,
                            "curve_evaluation_count": (
                                result.curve_evaluation_count
                            ),
                            "max_chord_error_px": result.max_chord_error,
                            "max_segment_arc_length_px": (
                                result.max_segment_arc_length
                            ),
                            "strength": result.strength,
                            "raw_points": [list(point) for point in divider.coords],
                            "fitted_points": result.points.tolist(),
                        },
                    }
                )

    current_geometries = list(geometries)
    accepted = set()
    rejected = set()
    for candidate_index, candidate in enumerate(candidates):
        left_index = candidate["left_index"]
        right_index = candidate["right_index"]
        left_before = current_geometries[left_index]
        right_before = current_geometries[right_index]
        replacement = [(candidate["divider"], candidate["fitted"])]
        old_total_area = left_before.area + right_before.area
        area_tolerance = max(1e-6, old_total_area * 1e-9)
        failure = ""
        try:
            left_after = _replace_geometry_dividers(left_before, replacement)
            right_after = _replace_geometry_dividers(right_before, replacement)
        except CommonBoundarySmoothingError as error:
            failure = f"replacement_failed:{error}"
        else:
            failure = _pair_validation_failure(
                left_before,
                right_before,
                left_after,
                right_after,
                area_tolerance,
                output_transform,
            )

        candidate["diagnostic"]["area_tolerance_px2"] = area_tolerance
        if failure:
            candidate["diagnostic"]["validation_failure"] = failure
            rejected.add(candidate_index)
            continue
        current_geometries[left_index] = left_after
        current_geometries[right_index] = right_after
        accepted.add(candidate_index)

    shifts: dict[int, list[tuple[float, float]]] = defaultdict(list)
    diagnostics = []
    for candidate_index, candidate in enumerate(candidates):
        diagnostic = dict(candidate["diagnostic"])
        if candidate_index in accepted:
            shifts[candidate["left_index"]].append(candidate["shift"])
            shifts[candidate["right_index"]].append(candidate["shift"])
        else:
            diagnostic["method"] = "unchanged"
            diagnostic["status"] = "skipped_validation_failed"
        diagnostics.append(diagnostic)

    formal = []
    for index, record in enumerate(source):
        original = geometries[index]
        rebuilt = current_geometries[index]
        if rebuilt.is_empty or not rebuilt.is_valid or rebuilt.area <= 0:
            raise CommonBoundarySmoothingError(
                f"common-divider fitting produced invalid polygon {index}"
            )
        if output_transform is not None:
            mapped = affine_transform(rebuilt, output_transform)
            if mapped.is_empty or not mapped.is_valid or mapped.area <= 0:
                raise CommonBoundarySmoothingError(
                    f"common-divider fitting produced invalid output-CRS polygon {index}"
                )
        before = _vertex_count(original)
        after = _vertex_count(rebuilt)
        feature_shifts = shifts.get(index) or []
        maximum = max((item[0] for item in feature_shifts), default=0.0)
        mean = (
            sum(item[1] for item in feature_shifts) / len(feature_shifts)
            if feature_shifts
            else 0.0
        )
        formal.append(
            {
                **record,
                "geometry": rebuilt,
                "fit_method": (
                    "cubic_bspline_adaptive_shared_divider"
                    if feature_shifts
                    else "unchanged"
                ),
                "fit_status": "changed" if feature_shifts else "unchanged",
                "fit_version": "divider_cubic_bspline_adaptive_v2",
                "vertex_count_before": before,
                "vertex_count_after": after,
                "max_shift_px": maximum,
                "mean_shift_px": mean,
                "area_change_ratio": 0.0,
            }
        )

    changed_diagnostics = [
        item for item in diagnostics if item["status"] == "changed"
    ]
    maximum = max(
        (item["max_displacement_px"] for item in changed_diagnostics), default=0.0
    )
    mean = (
        sum(item["mean_displacement_px"] for item in changed_diagnostics)
        / len(changed_diagnostics)
        if changed_diagnostics
        else 0.0
    )
    report = {
        "status": "passed",
        "fit_version": "divider_cubic_bspline_adaptive_v2",
        "curve_sampling_spacing_px": float(
            config.curve_sampling_spacing
        ),
        "curve_sampling_mode": "direct_adaptive_bezier_bounds",
        "dense_curve_materialized": False,
        "dense_curve_point_count_kind": (
            "equivalent_at_configured_spacing"
        ),
        "chord_error_certification": (
            "bezier_control_hull_upper_bound"
        ),
        "arc_length_certification": (
            "bezier_control_polygon_upper_bound"
        ),
        "max_chord_error_limit_px": float(config.max_chord_error),
        "max_segment_arc_length_limit_px": float(
            config.max_segment_arc_length
        ),
        "chain_count": len(candidates) + unchanged_count,
        "shared_chain_count": len(candidates) + unchanged_count,
        "spline_count": len(accepted),
        "unchanged_count": unchanged_count + len(rejected),
        "skipped_invalid_count": len(rejected),
        "max_displacement_px": maximum,
        "mean_displacement_px": mean,
        "dense_curve_point_count": sum(
            int(item["point_count_dense"]) for item in changed_diagnostics
        ),
        "sparse_curve_point_count": sum(
            int(item["point_count_after"]) for item in changed_diagnostics
        ),
        "curve_evaluation_count": sum(
            int(item["curve_evaluation_count"])
            for item in changed_diagnostics
        ),
        "max_chord_error_px": max(
            (
                float(item["max_chord_error_px"])
                for item in changed_diagnostics
            ),
            default=0.0,
        ),
        "max_segment_arc_length_px": max(
            (
                float(item["max_segment_arc_length_px"])
                for item in changed_diagnostics
            ),
            default=0.0,
        ),
        "candidate_validation": {
            "passed": True,
            "scope": "per_common_divider",
            "checks": ["valid", "positive_area", "pair_total_area"],
            "coordinate_spaces": (
                ["pixel", "output_crs"]
                if output_transform is not None
                else ["input"]
            ),
            "rejected_count": len(rejected),
        },
        "validation": {
            "passed": True,
            "scope": "all_output_polygons",
            "invalid_count": 0,
            "coordinate_spaces": (
                ["pixel", "output_crs"]
                if output_transform is not None
                else ["input"]
            ),
        },
        "diagnostics": diagnostics,
    }
    return formal, report
