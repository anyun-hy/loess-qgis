"""Regularize one complete polygon coverage in raster pixel coordinates.

The input is the immutable raw polygonization for one model or Fusion stream.
All polygons are processed together so a shared edge is changed exactly once.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import fiona
import numpy as np
import rasterio
import shapely
from shapely import affinity
from shapely.geometry import LineString, MultiLineString, mapping, shape


LAYER_NAME = "semantic_polygons"
METHOD = "shared_boundary_regularization_v1"
VERSION = "1"
IDENTITY_FIELDS = ("object_id", "part_id", "class_code", "class_name")


class BoundaryRegularizationError(RuntimeError):
    pass


class BoundaryFaceMappingError(BoundaryRegularizationError):
    def __init__(self, message, source_indices):
        super().__init__(message)
        self.source_indices = frozenset(int(value) for value in source_indices)


def _sha256(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _affine_parameters(transform):
    return [
        float(transform.a),
        float(transform.b),
        float(transform.d),
        float(transform.e),
        float(transform.c),
        float(transform.f),
    ]


def _to_pixel(geometry, transform):
    return affinity.affine_transform(geometry, _affine_parameters(~transform))


def _to_map(geometry, transform):
    return affinity.affine_transform(geometry, _affine_parameters(transform))


def _vertex_boundary_displacement(first, second):
    first_boundary = first.boundary
    second_boundary = second.boundary
    distances = []
    first_coordinates = shapely.get_coordinates(first_boundary)
    second_coordinates = shapely.get_coordinates(second_boundary)
    if len(first_coordinates):
        distances.append(float(np.max(shapely.distance(
            shapely.points(first_coordinates), second_boundary
        ))))
    if len(second_coordinates):
        distances.append(float(np.max(shapely.distance(
            shapely.points(second_coordinates), first_boundary
        ))))
    return max(distances, default=0.0)


def _point(value):
    return (round(float(value[0]), 9), round(float(value[1]), 9))


def _edge_key(left, right):
    a = _point(left)
    b = _point(right)
    return (a, b) if a <= b else (b, a)


def _rings(geometry):
    yield list(geometry.exterior.coords)
    for interior in geometry.interiors:
        yield list(interior.coords)


def _coverage_stats(geometries):
    values = np.asarray(geometries, dtype=object)
    union = shapely.union_all(values)
    area = float(np.sum(shapely.area(values)))
    union_area = float(shapely.area(union))
    return {
        "polygon_count": int(len(values)),
        "coordinate_count": int(np.sum(shapely.get_num_coordinates(values))),
        "area_px2": area,
        "union_area_px2": union_area,
        "overlap_area_px2": max(0.0, area - union_area),
        "coverage_is_valid": bool(shapely.coverage_is_valid(values)),
    }, union


def _build_edge_network(geometries):
    uses = defaultdict(list)
    for feature_index, geometry in enumerate(geometries):
        for ring in _rings(geometry):
            for left, right in zip(ring, ring[1:]):
                key = _edge_key(left, right)
                if key[0] == key[1]:
                    continue
                uses[key].append(feature_index)
    invalid = [key for key, owners in uses.items() if len(owners) not in (1, 2)]
    if invalid:
        raise BoundaryRegularizationError(
            f"coverage edge ownership is not one-or-two sided: {invalid[:3]}"
        )
    duplicate_owner = [
        key for key, owners in uses.items()
        if len(owners) == 2 and owners[0] == owners[1]
    ]
    if duplicate_owner:
        raise BoundaryRegularizationError(
            f"one polygon owns both sides of an edge: {duplicate_owner[:3]}"
        )
    return uses


def _shared_chains(edge_uses):
    shared = {key for key, owners in edge_uses.items() if len(owners) == 2}
    adjacency = defaultdict(list)
    for key in shared:
        left, right = key
        adjacency[left].append((right, key))
        adjacency[right].append((left, key))
    protected = set()
    for node, links in adjacency.items():
        owner_pairs = {
            tuple(sorted(edge_uses[key]))
            for _neighbor, key in links
        }
        if len(links) != 2 or len(owner_pairs) != 1:
            protected.add(node)
    visited = set()
    chains = []
    for start in sorted(protected):
        for neighbor, first_key in adjacency[start]:
            if first_key in visited:
                continue
            coordinates = [start]
            keys = []
            previous = None
            current = start
            next_node = neighbor
            next_key = first_key
            while True:
                visited.add(next_key)
                keys.append(next_key)
                coordinates.append(next_node)
                previous, current = current, next_node
                if current in protected:
                    break
                options = [item for item in adjacency[current] if item[0] != previous]
                if len(options) != 1:
                    raise BoundaryRegularizationError(
                        "shared-edge chain traversal reached an ambiguous node"
                    )
                next_node, next_key = options[0]
                if next_key in visited:
                    raise BoundaryRegularizationError(
                        "shared-edge chain unexpectedly formed a loop"
                    )
            chains.append({
                "coordinates": coordinates,
                "edge_keys": keys,
                "closed": False,
            })
    closed_edges = shared - visited
    closed_edge_count = len(closed_edges)
    remaining = set(closed_edges)
    while remaining:
        first_key = min(remaining)
        start, next_node = first_key
        coordinates = [start]
        keys = []
        next_key = first_key
        while True:
            remaining.remove(next_key)
            keys.append(next_key)
            coordinates.append(next_node)
            current = next_node
            if current == start:
                break
            options = [
                (neighbor, key)
                for neighbor, key in adjacency[current]
                if key in remaining
            ]
            if len(options) != 1:
                raise BoundaryRegularizationError(
                    "closed shared-edge chain traversal is ambiguous"
                )
            next_node, next_key = options[0]
        chains.append({
            "coordinates": coordinates,
            "edge_keys": keys,
            "closed": True,
        })
    return chains, closed_edge_count, protected


def _angle_degrees(left, right):
    left_norm = math.hypot(left[0], left[1])
    right_norm = math.hypot(right[0], right[1])
    if left_norm == 0 or right_norm == 0:
        return 180.0
    cosine = max(-1.0, min(1.0, (left[0] * right[0] + left[1] * right[1]) / (left_norm * right_norm)))
    return math.degrees(math.acos(cosine))


def _trend_break_indices(coordinates, angle_threshold):
    count = len(coordinates)
    protected = {0, count - 1}
    if count < 3:
        return sorted(protected)
    trend_line = LineString(coordinates).simplify(4.0, preserve_topology=False)
    trend_coordinates = np.asarray(trend_line.coords, dtype=float)
    for trend_index in range(1, len(trend_coordinates) - 1):
        current = trend_coordinates[trend_index]
        before = current - trend_coordinates[trend_index - 1]
        after = trend_coordinates[trend_index + 1] - current
        if _angle_degrees(before, after) > float(angle_threshold):
            distances = np.linalg.norm(coordinates - current, axis=1)
            protected.add(int(np.argmin(distances)))
    return sorted(protected)


def _chain_candidate(chain, config, simplify_tolerance):
    coordinates = np.asarray(chain["coordinates"], dtype=float)
    if len(coordinates) < int(config["minimum_chain_vertices"]):
        return None
    simplify_tolerance = float(simplify_tolerance)
    if simplify_tolerance <= 0:
        return None
    segments = np.diff(coordinates, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    path_length = float(np.sum(lengths))
    if path_length <= 0:
        return None
    if float(np.median(lengths)) > 2.5:
        return None
    replacement = []
    if bool(chain.get("closed")):
        simplified_ring = LineString(coordinates).simplify(
            simplify_tolerance,
            preserve_topology=True,
        )
        replacement = [tuple(value) for value in simplified_ring.coords]
        if replacement[0] != replacement[-1]:
            replacement.append(replacement[0])
        if len(replacement) < 4:
            return None
    else:
        protected = _trend_break_indices(
            coordinates, float(config["angle_threshold_deg"])
        )
        for start_index, end_index in zip(protected, protected[1:]):
            section = coordinates[start_index:end_index + 1]
            if len(section) <= 2:
                simplified_section = section
            else:
                simplified_section = np.asarray(
                    LineString(section).simplify(
                        simplify_tolerance,
                        preserve_topology=True,
                    ).coords,
                    dtype=float,
                )
            if replacement:
                replacement.extend(tuple(value) for value in simplified_section[1:])
            else:
                replacement.extend(tuple(value) for value in simplified_section)
    if len(replacement) >= len(coordinates):
        return None
    replacement_line = LineString(replacement)
    max_deviation = max(
        (replacement_line.distance(shapely.Point(value)) for value in coordinates),
        default=0.0,
    )
    if max_deviation > simplify_tolerance + 1e-8:
        return None
    return {
        "replacement": tuple(replacement),
        "max_deviation_px": float(max_deviation),
        "path_length_px": path_length,
        "replacement_length_px": float(replacement_line.length),
        "removed_vertices": max(0, len(coordinates) - len(replacement)),
    }


def _intersection_is_chain_endpoint(intersection, endpoints, tolerance=1e-8):
    if intersection.is_empty:
        return True
    parts = list(shapely.get_parts(intersection))
    if not parts:
        parts = [intersection]
    for part in parts:
        if part.geom_type != "Point":
            return False
        coordinate = np.asarray(part.coords[0], dtype=float)
        if not any(float(np.linalg.norm(coordinate - endpoint)) <= tolerance for endpoint in endpoints):
            return False
    return True


def _line_vertex_displacement(first, second):
    distances = []
    for source, target in ((first, second), (second, first)):
        coordinates = shapely.get_coordinates(source)
        if len(coordinates):
            distances.append(float(np.max(shapely.distance(
                shapely.points(coordinates), target
            ))))
    return max(distances, default=0.0)


def _safe_replacements(
    chains, edge_uses, raw_geometries, config, simplify_tolerance
):
    edge_keys = list(edge_uses)
    edge_lines = [LineString(key) for key in edge_keys]
    tree = shapely.STRtree(edge_lines)
    original_safe = []
    rejected_original_crossing = 0
    rejected_owner_topology = 0
    rejected_movement = 0
    raw_shared_cache = {}
    maximum_displacement = float(config["max_deviation_px"])
    for chain in chains:
        chain_keys = set(chain["edge_keys"])
        owner_pairs = {
            tuple(sorted(edge_uses[key])) for key in chain["edge_keys"]
        }
        if len(owner_pairs) != 1 or len(next(iter(owner_pairs))) != 2:
            rejected_owner_topology += 1
            continue
        owner_pair = next(iter(owner_pairs))
        if owner_pair not in raw_shared_cache:
            raw_shared_cache[owner_pair] = raw_geometries[
                owner_pair[0]
            ].boundary.intersection(raw_geometries[owner_pair[1]].boundary)
        raw_shared = raw_shared_cache[owner_pair]
        chain_line = LineString(chain["coordinates"])
        local_raw = raw_shared.intersection(
            chain_line.buffer(maximum_displacement * 2.0, cap_style="flat")
        )
        candidate = None
        chord = None
        for local_factor in (1.0, 0.75, 0.5, 0.25):
            local_candidate = _chain_candidate(
                chain, config, simplify_tolerance * local_factor
            )
            if local_candidate is None:
                continue
            local_chord = LineString(local_candidate["replacement"])
            if not local_raw.is_empty and _line_vertex_displacement(
                local_chord, local_raw
            ) <= maximum_displacement + 1e-8:
                candidate = local_candidate
                chord = local_chord
                break
        if candidate is None:
            rejected_movement += 1
            continue
        endpoints = [
            np.asarray(candidate["replacement"][0], dtype=float),
            np.asarray(candidate["replacement"][-1], dtype=float),
        ]
        safe = True
        for edge_index in tree.query(chord, predicate="intersects"):
            key = edge_keys[int(edge_index)]
            if key in chain_keys:
                continue
            intersection = chord.intersection(edge_lines[int(edge_index)])
            if not _intersection_is_chain_endpoint(intersection, endpoints):
                safe = False
                break
        if not safe:
            rejected_original_crossing += 1
            continue
        original_safe.append((chain, candidate))

    candidate_lines = [LineString(item[1]["replacement"]) for item in original_safe]
    candidate_tree = shapely.STRtree(candidate_lines) if candidate_lines else None
    accepted_indices = set()
    rejected_candidate_crossing = 0
    for index, line in enumerate(candidate_lines):
        conflict = False
        endpoints = [
            np.asarray(line.coords[0], dtype=float),
            np.asarray(line.coords[-1], dtype=float),
        ]
        for other_index in candidate_tree.query(line, predicate="intersects"):
            other_index = int(other_index)
            if other_index not in accepted_indices:
                continue
            other = candidate_lines[other_index]
            intersection = line.intersection(other)
            other_endpoints = [
                np.asarray(other.coords[0], dtype=float),
                np.asarray(other.coords[-1], dtype=float),
            ]
            if not (
                _intersection_is_chain_endpoint(intersection, endpoints)
                and _intersection_is_chain_endpoint(intersection, other_endpoints)
            ):
                conflict = True
                break
        if conflict:
            rejected_candidate_crossing += 1
        else:
            accepted_indices.add(index)
    replacements = [
        item for index, item in enumerate(original_safe) if index in accepted_indices
    ]
    return replacements, {
        "original_edge": rejected_original_crossing,
        "candidate_edge": rejected_candidate_crossing,
        "owner_topology": rejected_owner_topology,
        "total": (
            rejected_original_crossing
            + rejected_candidate_crossing
            + rejected_owner_topology
        ),
    }, rejected_movement


def _reconstruct_faces(geometries, edge_uses, replacements):
    removed = set()
    replacement_lines = []
    for chain, candidate in replacements:
        removed.update(chain["edge_keys"])
        replacement_lines.append(LineString(candidate["replacement"]))
    lines = [
        LineString(key)
        for key in edge_uses
        if key not in removed
    ]
    lines.extend(replacement_lines)
    noded = shapely.node(MultiLineString(lines))
    faces = [
        part
        for part in shapely.get_parts(
            shapely.polygonize(list(shapely.get_parts(noded)))
        )
        if part.area > 0
    ]
    tree = shapely.STRtree(faces)
    assigned = {}
    face_sources = {}
    problems = set()
    for source_index, source in enumerate(geometries):
        marker = source.representative_point()
        matches = [int(index) for index in tree.query(marker, predicate="within")]
        if not matches:
            matches = [
                int(index) for index in tree.query(marker, predicate="covered_by")
            ]
        if len(matches) != 1:
            problems.add(source_index)
            continue
        face_index = matches[0]
        if face_index in face_sources:
            problems.update((source_index, face_sources[face_index]))
            continue
        face_sources[face_index] = source_index
        assigned[face_index] = source_index
    unassigned_faces = set(range(len(faces))) - set(assigned)
    if unassigned_faces:
        source_tree = shapely.STRtree(geometries)
        for face_index in unassigned_faces:
            marker = faces[face_index].representative_point()
            problems.update(
                int(index)
                for index in source_tree.query(marker, predicate="within")
            )
    if len(faces) != len(geometries) or problems or len(assigned) != len(faces):
        raise BoundaryFaceMappingError(
            f"strict face mapping failed: feature_count={len(geometries)}, "
            f"face_count={len(faces)}, problem_sources={sorted(problems)[:20]}, "
            f"replacement_count={len(replacements)}",
            problems,
        )
    output = [None] * len(geometries)
    for face_index, source_index in assigned.items():
        output[source_index] = faces[face_index]
    return output


def _replacement_owners(chain, edge_uses):
    return {
        int(owner)
        for key in chain["edge_keys"]
        for owner in edge_uses[key]
    }


def _strict_reconstruct_geometries(geometries):
    merged_boundaries = shapely.union_all([geometry.boundary for geometry in geometries])
    noded = shapely.node(merged_boundaries)
    faces = [
        part
        for part in shapely.get_parts(
            shapely.polygonize(list(shapely.get_parts(noded)))
        )
        if part.area > 0
    ]
    if len(faces) != len(geometries):
        raise BoundaryRegularizationError(
            f"strict final face reconstruction changed feature count: "
            f"{len(geometries)} -> {len(faces)}"
        )
    tree = shapely.STRtree(faces)
    assigned = {}
    for source_index, source in enumerate(geometries):
        marker = source.representative_point()
        matches = [int(index) for index in tree.query(marker, predicate="within")]
        if not matches:
            matches = [
                int(index) for index in tree.query(marker, predicate="covered_by")
            ]
        if len(matches) != 1 or matches[0] in assigned:
            raise BoundaryRegularizationError(
                f"strict final face mapping is not bijective for source feature "
                f"{source_index}: {matches}"
            )
        assigned[matches[0]] = source_index
    output = [None] * len(geometries)
    for face_index, source_index in assigned.items():
        output[source_index] = faces[face_index]
    if any(geometry is None for geometry in output):
        raise BoundaryRegularizationError("strict final face mapping left unassigned features")
    return output


def coverage_simplify_pixel_coverage(geometries, config):
    values = np.asarray(geometries, dtype=object)
    requested_tolerance = float(config["coverage_tolerance_px"])
    maximum_displacement = float(config["max_deviation_px"])
    simplify_tolerance = requested_tolerance
    simplified = values
    simplify_displacement = 0.0
    simplify_budget = maximum_displacement
    for _attempt in range(10):
        candidate = np.asarray(
            shapely.coverage_simplify(
                values,
                tolerance=simplify_tolerance,
                simplify_boundary=not bool(config["preserve_outer_boundary"]),
            ),
            dtype=object,
        )
        candidate_displacement = max(
            (
                _vertex_boundary_displacement(before, after)
                for before, after in zip(values, candidate)
            ),
            default=0.0,
        )
        if candidate_displacement <= simplify_budget + 1e-8:
            simplified = candidate
            simplify_displacement = candidate_displacement
            break
        simplify_tolerance *= 0.75
    else:
        raise BoundaryRegularizationError(
            "coverage_simplify cannot satisfy the configured maximum displacement"
        )
    if simplified.shape != values.shape:
        raise BoundaryRegularizationError("coverage_simplify changed the feature count")
    return simplified, simplify_tolerance, simplify_displacement


def regularize_pixel_coverage(geometries, config):
    values = np.asarray(geometries, dtype=object)
    if len(values) == 0:
        raise BoundaryRegularizationError("raw polygon coverage is empty")
    if np.any(shapely.is_empty(values)) or not np.all(shapely.is_valid(values)):
        raise BoundaryRegularizationError("raw polygon coverage contains empty or invalid geometry")
    if any(geometry.geom_type != "Polygon" for geometry in values):
        raise BoundaryRegularizationError("raw polygon coverage must contain only Polygon geometry")

    simplified, simplify_tolerance, simplify_displacement = (
        coverage_simplify_pixel_coverage(values, config)
    )
    requested_tolerance = float(config["coverage_tolerance_px"])
    maximum_displacement = float(config["max_deviation_px"])
    edge_uses = _build_edge_network(simplified)
    chains, closed_edge_count, protected_nodes = _shared_chains(edge_uses)
    raw_union = shapely.union_all(values)
    area_epsilon = max(1e-9, float(raw_union.area) * 1e-10)
    regularized = None
    accepted_replacements = []
    rejected_crossing = {
        "original_edge": 0,
        "candidate_edge": 0,
        "owner_topology": 0,
        "total": 0,
    }
    selected_chain_tolerance = 0.0
    selected_movement_rejected = 0
    attempts = []
    for factor in (1.0, 0.75, 0.5, 0.25):
        chain_tolerance = maximum_displacement * factor
        replacements, crossing_count, local_movement_rejected = _safe_replacements(
            chains, edge_uses, values, config, chain_tolerance
        )
        attempt = {
            "tolerance_px": chain_tolerance,
            "candidate_count": len(replacements),
            "crossing_rejected_count": crossing_count,
            "movement_rejected_count": local_movement_rejected,
            "passed": False,
            "error": "",
        }
        if not replacements:
            attempt["error"] = "no safe near-collinear chain candidate"
            attempts.append(attempt)
            continue
        try:
            active_replacements = list(replacements)
            topology_rejected = 0
            movement_rejected = local_movement_rejected
            while True:
                try:
                    candidate_geometries = _reconstruct_faces(
                        list(simplified), edge_uses, active_replacements
                    )
                except BoundaryFaceMappingError as exc:
                    if not exc.source_indices:
                        raise
                    retained = []
                    removed = []
                    for replacement in active_replacements:
                        owners = _replacement_owners(replacement[0], edge_uses)
                        if owners.intersection(exc.source_indices):
                            removed.append(replacement)
                        else:
                            retained.append(replacement)
                    if not removed or not retained:
                        raise
                    topology_rejected += len(removed)
                    active_replacements = retained
                    continue
                candidate_values = np.asarray(candidate_geometries, dtype=object)
                displacements = np.asarray(
                    [
                        _vertex_boundary_displacement(before, after)
                        for before, after in zip(values, candidate_values)
                    ],
                    dtype=float,
                )
                offending_sources = set(np.flatnonzero(
                    displacements > maximum_displacement + 1e-8
                ).tolist())
                if not offending_sources:
                    break
                retained = []
                removed = []
                for replacement in active_replacements:
                    owners = _replacement_owners(replacement[0], edge_uses)
                    if owners.intersection(offending_sources):
                        removed.append(replacement)
                    else:
                        retained.append(replacement)
                if not removed or not retained:
                    raise BoundaryRegularizationError(
                        "movement gate cannot isolate safe shared-chain candidates"
                    )
                movement_rejected += len(removed)
                active_replacements = retained
            replacements = active_replacements
            crossing_count = dict(crossing_count)
            crossing_count["owner_topology"] += topology_rejected
            crossing_count["total"] += topology_rejected
            attempt["candidate_count"] = len(replacements)
            attempt["crossing_rejected_count"] = crossing_count
            attempt["movement_rejected_count"] = movement_rejected
            candidate_union = shapely.union_all(candidate_values)
            maximum_candidate_displacement = float(
                np.max(displacements) if displacements.size else 0.0
            )
            union_difference = float(
                shapely.area(shapely.symmetric_difference(raw_union, candidate_union))
            )
            if not bool(shapely.coverage_is_valid(candidate_values)):
                raise BoundaryRegularizationError("candidate coverage is not valid")
            if union_difference > area_epsilon:
                raise BoundaryRegularizationError(
                    f"candidate changed covered region by {union_difference} px2"
                )
            if maximum_candidate_displacement > maximum_displacement + 1e-8:
                raise BoundaryRegularizationError(
                    f"candidate displacement {maximum_candidate_displacement} exceeds {maximum_displacement} px"
                )
            regularized = candidate_geometries
            accepted_replacements = replacements
            rejected_crossing = crossing_count
            selected_chain_tolerance = chain_tolerance
            selected_movement_rejected = movement_rejected
            attempt["passed"] = True
            attempts.append(attempt)
            break
        except BoundaryRegularizationError as exc:
            attempt["error"] = str(exc)
            attempts.append(attempt)
    if regularized is None:
        regularized = _strict_reconstruct_geometries(list(simplified))
    network = {
        "unique_edge_count": len(edge_uses),
        "shared_edge_count": sum(1 for owners in edge_uses.values() if len(owners) == 2),
        "outer_edge_count": sum(1 for owners in edge_uses.values() if len(owners) == 1),
        "protected_node_count": len(protected_nodes),
        "open_shared_chain_count": sum(
            1 for chain in chains if not bool(chain.get("closed"))
        ),
        "closed_shared_edge_count": closed_edge_count,
        "regularized_chain_count": len(accepted_replacements),
        "crossing_candidate_rejected_count": rejected_crossing,
        "movement_candidate_rejected_count": selected_movement_rejected,
        "regularized_removed_vertices": sum(
            item[1]["removed_vertices"] for item in accepted_replacements
        ),
        "coverage_simplify_requested_tolerance_px": requested_tolerance,
        "coverage_simplify_effective_tolerance_px": simplify_tolerance,
        "coverage_simplify_maximum_displacement_px": simplify_displacement,
        "shared_chain_selected_tolerance_px": selected_chain_tolerance,
        "shared_chain_attempts": attempts,
        "maximum_chain_deviation_px": max(
            (item[1]["max_deviation_px"] for item in accepted_replacements), default=0.0
        ),
    }
    return regularized, network


def _validation(raw, formal, identities, config):
    raw_stats, raw_union = _coverage_stats(raw)
    formal_stats, formal_union = _coverage_stats(formal)
    area_epsilon = max(1e-9, raw_stats["union_area_px2"] * 1e-10)
    union_difference = float(shapely.area(shapely.symmetric_difference(raw_union, formal_union)))
    displacements = [
        _vertex_boundary_displacement(before, after)
        for before, after in zip(raw, formal)
    ]
    max_displacement = max(displacements, default=0.0)
    checks = {
        "geometry_valid": bool(
            len(raw) == len(formal)
            and np.all(shapely.is_valid(np.asarray(formal, dtype=object)))
            and not np.any(shapely.is_empty(np.asarray(formal, dtype=object)))
        ),
        "identity_bijection": bool(
            len(identities) == len(set(identities)) == len(formal)
        ),
        "coverage_valid": bool(formal_stats["coverage_is_valid"]),
        "overlap_within_tolerance": formal_stats["overlap_area_px2"] <= area_epsilon,
        "covered_region_unchanged": union_difference <= area_epsilon,
        "movement_within_tolerance": max_displacement <= float(config["max_deviation_px"]) + 1e-8,
        "outer_boundary_preserved": union_difference <= area_epsilon,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "raw": raw_stats,
        "formal": formal_stats,
        "union_symmetric_difference_px2": union_difference,
        "maximum_boundary_displacement_px": max_displacement,
        "boundary_displacement_method": "symmetric polygon-boundary vertex distance",
        "area_epsilon_px2": area_epsilon,
    }


def regularize_coverage(input_path, mask_path, output_path, report_path, config):
    input_path = Path(input_path).resolve()
    mask_path = Path(mask_path).resolve()
    output_path = Path(output_path).resolve()
    report_path = Path(report_path).resolve()
    with rasterio.open(mask_path) as raster:
        transform = raster.transform
        raster_crs = raster.crs
    with fiona.open(input_path, layer=LAYER_NAME) as source:
        schema = dict(source.schema)
        source_crs_wkt = source.crs_wkt
        source_crs = source.crs
        records = [
            {"geometry": shape(feature["geometry"]), "properties": dict(feature["properties"])}
            for feature in source
        ]
    if raster_crs and source_crs_wkt:
        from rasterio.crs import CRS

        if CRS.from_wkt(source_crs_wkt) != raster_crs:
            raise BoundaryRegularizationError("raw polygons and mask mosaic use different CRS")

    identities = [tuple(record["properties"].get(field) for field in IDENTITY_FIELDS) for record in records]
    raw_pixel = [_to_pixel(record["geometry"], transform) for record in records]
    formal_pixel, network = regularize_pixel_coverage(raw_pixel, config)
    validation = _validation(raw_pixel, formal_pixel, identities, config)
    if not validation["passed"]:
        raise BoundaryRegularizationError(
            "boundary regularization hard gate failed: "
            + ", ".join(key for key, value in validation["checks"].items() if not value)
            + f"; max_displacement_px={validation['maximum_boundary_displacement_px']:.6f}"
            + f"; regularized_chain_count={network['regularized_chain_count']}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    properties_schema = dict(schema.get("properties") or {})
    properties_schema.update({
        "regularization_method": "str",
        "regularization_version": "str",
        "regularization_status": "str",
        "vertex_count_before": "int",
        "vertex_count_after": "int",
        "area_change_ratio": "float",
    })
    schema["properties"] = properties_schema
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".gpkg", dir=output_path.parent
    )
    os.close(fd)
    os.unlink(temporary_name)
    write_kwargs = {
        "driver": "GPKG",
        "layer": LAYER_NAME,
        "schema": schema,
        "crs_wkt": source_crs_wkt,
        "crs": source_crs if not source_crs_wkt else None,
    }
    write_kwargs = {key: value for key, value in write_kwargs.items() if value is not None}
    try:
        with fiona.open(temporary_name, "w", **write_kwargs) as destination:
            for record, before, after_pixel in zip(records, raw_pixel, formal_pixel):
                after = _to_map(after_pixel, transform)
                properties = dict(record["properties"])
                properties.update({
                    "regularization_method": METHOD,
                    "regularization_version": VERSION,
                    "regularization_status": "passed",
                    "vertex_count_before": int(shapely.get_num_coordinates(before)),
                    "vertex_count_after": int(shapely.get_num_coordinates(after_pixel)),
                    "area_change_ratio": float(
                        (after_pixel.area - before.area) / before.area
                        if before.area > 0 else 0.0
                    ),
                })
                destination.write({"geometry": mapping(after), "properties": properties})
        os.replace(temporary_name, output_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    report = {
        "schema_version": 1,
        "status": "passed",
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": METHOD,
        "version": VERSION,
        "config": dict(config),
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "mask_mosaic": str(mask_path),
        "mask_mosaic_sha256": _sha256(mask_path),
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "identity_fields": list(IDENTITY_FIELDS),
        "network": network,
        "validation": validation,
    }
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    with open(temporary_report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_report, report_path)
    print(json.dumps({
        "event": "boundary_regularization_finished",
        "status": "passed",
        "output": str(output_path),
        "report": str(report_path),
        "polygon_count": len(records),
        "coordinate_reduction": (
            validation["raw"]["coordinate_count"]
            - validation["formal"]["coordinate_count"]
        ),
        "regularized_chain_count": network["regularized_chain_count"],
    }, ensure_ascii=False, separators=(",", ":")), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Regularize one complete semantic polygon coverage")
    parser.add_argument("--input", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-deviation-px", type=float, default=1.5)
    parser.add_argument("--angle-threshold-deg", type=float, default=12.0)
    parser.add_argument("--minimum-chain-vertices", type=int, default=4)
    parser.add_argument("--coverage-tolerance-px", type=float, default=1.5)
    parser.add_argument("--preserve-outer-boundary", action="store_true")
    args = parser.parse_args(argv)
    config = {
        "enabled": True,
        "mode": "standard",
        "coverage_tolerance_px": float(args.coverage_tolerance_px),
        "angle_threshold_deg": float(args.angle_threshold_deg),
        "max_deviation_px": float(args.max_deviation_px),
        "minimum_chain_vertices": int(args.minimum_chain_vertices),
        "preserve_outer_boundary": bool(args.preserve_outer_boundary),
        "natural_smoothing": False,
    }
    try:
        regularize_coverage(args.input, args.mask, args.output, args.report, config)
        return 0
    except Exception as exc:
        print(json.dumps({
            "event": "boundary_regularization_failed",
            "status": "failed",
            "error": str(exc),
        }, ensure_ascii=False, separators=(",", ":")), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
