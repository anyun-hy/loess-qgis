"""Build one multiclass polygon coverage from a 14-class probability mosaic."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import math
import os
import tempfile
import time
from pathlib import Path

import fiona
import numpy as np
import rasterio
import shapely
from rasterio import features
from rasterio.windows import Window
from shapely import affinity
from shapely.geometry import LineString, mapping, shape

from boundary_ab_validate import _staircase_metrics
from polygonize_mosaic import LAYER_NAME, SCHEMA, _stream_token, load_class_map


logging.basicConfig(stream=None, level=logging.INFO, format="%(message)s")
logger = logging.getLogger("subpixel_vectorizer")

CLASS_COUNT = 14
METHOD = "multiclass_subpixel_probability_v1"
VERSION = "1.1"
DEFAULT_INTERPOLATION_STRENGTH = 1.0
DEFAULT_COVERAGE_TOLERANCE_PX = 1.0
DEFAULT_MAX_DEVIATION_PX = 1.5
DEFAULT_STRIPE_ROWS = 128
QSDK_NONINFERIORITY_MARGIN_PX = 0.5


class SubpixelVectorizationError(RuntimeError):
    pass


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_pixel(geometry, transform):
    inverse = ~transform
    return affinity.affine_transform(
        geometry,
        [inverse.a, inverse.b, inverse.d, inverse.e, inverse.c, inverse.f],
    )


def _to_map(geometry, transform):
    return affinity.affine_transform(
        geometry,
        [transform.a, transform.b, transform.d, transform.e, transform.c, transform.f],
    )


def _probability_scale(source) -> float:
    scales = tuple(float(value) for value in source.scales)
    if source.count != CLASS_COUNT:
        raise SubpixelVectorizationError(
            f"probability mosaic must have {CLASS_COUNT} bands, got {source.count}"
        )
    if len(set(round(value, 18) for value in scales)) != 1:
        raise SubpixelVectorizationError("probability mosaic bands use different scales")
    scale = scales[0]
    if source.dtypes[0] == "uint16":
        expected = 1.0 / 65535.0
        if not math.isclose(scale, expected, rel_tol=0.0, abs_tol=1e-12):
            raise SubpixelVectorizationError(
                f"uint16 probability mosaic scale must be 1/65535, got {scale}"
            )
    elif source.dtypes[0] in ("float32", "float64"):
        if not math.isclose(scale, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise SubpixelVectorizationError(
                f"floating probability mosaic scale must be 1, got {scale}"
            )
    else:
        raise SubpixelVectorizationError(
            f"unsupported probability mosaic dtype: {source.dtypes[0]}"
        )
    return scale


def _read_scores(source, row_start: int, row_count: int, scale: float) -> np.ndarray:
    values = source.read(
        window=Window(0, row_start, source.width, row_count),
        out_dtype="float32",
    )
    values *= float(scale)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise SubpixelVectorizationError("probability mosaic contains invalid values")
    return values


def _fraction(scores, first, second, first_class, second_class, strength, movement_stats=None):
    if first_class < 0 or second_class < 0:
        value = 0.5
    else:
        r0, c0 = first
        r1, c1 = second
        d0 = float(scores[first_class, r0, c0] - scores[second_class, r0, c0])
        d1 = float(scores[first_class, r1, c1] - scores[second_class, r1, c1])
        denominator = d0 - d1
        value = 0.5 if abs(denominator) < 1e-12 else d0 / denominator
        value = 0.5 + float(strength) * (value - 0.5)
        value = float(np.clip(value, 0.02, 0.98))
    if movement_stats is not None:
        displacement = abs(float(value) - 0.5)
        movement_stats["count"] += 1
        movement_stats["maximum"] = max(movement_stats["maximum"], displacement)
    return float(value)


def _horizontal_crossing(
    scores, labels, global_row, col, stripe_start, strength, movement_stats=None
):
    local_row = global_row - stripe_start
    first_class = int(labels[global_row, col])
    second_class = int(labels[global_row, col + 1])
    fraction = _fraction(
        scores,
        (local_row, col),
        (local_row, col + 1),
        first_class,
        second_class,
        strength,
        movement_stats,
    )
    return (float(col) + 0.5 + fraction, float(global_row) + 0.5)


def _vertical_crossing(
    scores, labels, row, col, stripe_start, strength, movement_stats=None
):
    local_row = row - stripe_start
    first_class = int(labels[row, col])
    second_class = int(labels[row + 1, col])
    fraction = _fraction(
        scores,
        (local_row, col),
        (local_row + 1, col),
        first_class,
        second_class,
        strength,
        movement_stats,
    )
    return (float(col) + 0.5, float(row) + 0.5 + fraction)


def _append_segment(segments, first, second):
    if first != second:
        segments.append((first, second))


def _append_boundary_segments(segments, positions, axis, fixed):
    ordered = sorted(set(float(value) for value in positions))
    for first, second in zip(ordered, ordered[1:]):
        if axis == "x":
            _append_segment(segments, (first, fixed), (second, fixed))
        else:
            _append_segment(segments, (fixed, first), (fixed, second))


def build_shared_linework(
    probability_source,
    labels,
    *,
    interpolation_strength=DEFAULT_INTERPOLATION_STRENGTH,
    stripe_rows=DEFAULT_STRIPE_ROWS,
    progress_callback=None,
):
    """Return a deterministically noded multiclass line network in pixel space."""
    height, width = labels.shape
    if height < 2 or width < 2:
        raise SubpixelVectorizationError("mosaic must be at least 2 x 2 pixels")
    scale = _probability_scale(probability_source)
    segments = []
    junction_count = 0
    movement_stats = {"count": 0, "maximum": 0.0}
    top_positions = [0.0, float(width)]
    bottom_positions = [0.0, float(width)]
    left_positions = [0.0, float(height)]
    right_positions = [0.0, float(height)]

    stripe_rows = max(1, int(stripe_rows))
    for row_start in range(0, height - 1, stripe_rows):
        row_end = min(row_start + stripe_rows, height - 1)
        scores = _read_scores(
            probability_source,
            row_start,
            row_end - row_start + 1,
            scale,
        )
        top_diff = labels[row_start:row_end, :-1] != labels[row_start:row_end, 1:]
        bottom_diff = labels[row_start + 1:row_end + 1, :-1] != labels[
            row_start + 1:row_end + 1, 1:
        ]
        left_diff = labels[row_start:row_end, :-1] != labels[
            row_start + 1:row_end + 1, :-1
        ]
        right_diff = labels[row_start:row_end, 1:] != labels[
            row_start + 1:row_end + 1, 1:
        ]
        candidates = np.argwhere(top_diff | right_diff | bottom_diff | left_diff)
        for local_row, col in candidates:
            row = row_start + int(local_row)
            col = int(col)
            crossings = []
            if top_diff[local_row, col]:
                crossings.append(
                    _horizontal_crossing(
                        scores,
                        labels,
                        row,
                        col,
                        row_start,
                        interpolation_strength,
                        movement_stats,
                    )
                )
            if right_diff[local_row, col]:
                crossings.append(
                    _vertical_crossing(
                        scores,
                        labels,
                        row,
                        col + 1,
                        row_start,
                        interpolation_strength,
                        movement_stats,
                    )
                )
            if bottom_diff[local_row, col]:
                crossings.append(
                    _horizontal_crossing(
                        scores,
                        labels,
                        row + 1,
                        col,
                        row_start,
                        interpolation_strength,
                        movement_stats,
                    )
                )
            if left_diff[local_row, col]:
                crossings.append(
                    _vertical_crossing(
                        scores,
                        labels,
                        row,
                        col,
                        row_start,
                        interpolation_strength,
                        movement_stats,
                    )
                )
            if len(crossings) == 2:
                _append_segment(segments, crossings[0], crossings[1])
            elif len(crossings) > 2:
                junction = (
                    float(np.mean([point[0] for point in crossings])),
                    float(np.mean([point[1] for point in crossings])),
                )
                junction_count += 1
                for point in crossings:
                    _append_segment(segments, point, junction)

        if row_start == 0:
            for col in np.flatnonzero(labels[0, :-1] != labels[0, 1:]):
                point = _horizontal_crossing(
                    scores,
                    labels,
                    0,
                    int(col),
                    row_start,
                    interpolation_strength,
                    movement_stats,
                )
                _append_segment(segments, point, (point[0], 0.0))
                top_positions.append(point[0])
        if row_end == height - 1:
            local_bottom = height - 1
            for col in np.flatnonzero(labels[local_bottom, :-1] != labels[local_bottom, 1:]):
                point = _horizontal_crossing(
                    scores,
                    labels,
                    local_bottom,
                    int(col),
                    row_start,
                    interpolation_strength,
                    movement_stats,
                )
                _append_segment(segments, point, (point[0], float(height)))
                bottom_positions.append(point[0])
        for row in range(row_start, row_end):
            if labels[row, 0] != labels[row + 1, 0]:
                point = _vertical_crossing(
                    scores,
                    labels,
                    row,
                    0,
                    row_start,
                    interpolation_strength,
                    movement_stats,
                )
                _append_segment(segments, point, (0.0, point[1]))
                left_positions.append(point[1])
            if labels[row, width - 1] != labels[row + 1, width - 1]:
                point = _vertical_crossing(
                    scores,
                    labels,
                    row,
                    width - 1,
                    row_start,
                    interpolation_strength,
                    movement_stats,
                )
                _append_segment(segments, point, (float(width), point[1]))
                right_positions.append(point[1])
        logger.info(
            "[subpixel_vectorizer] linework rows %d/%d, segments=%d",
            row_end,
            height - 1,
            len(segments),
        )
        if progress_callback is not None:
            progress_callback(row_end, height - 1, len(segments))

    _append_boundary_segments(segments, top_positions, "x", 0.0)
    _append_boundary_segments(segments, bottom_positions, "x", float(height))
    _append_boundary_segments(segments, left_positions, "y", 0.0)
    _append_boundary_segments(segments, right_positions, "y", float(width))
    return segments, junction_count, scale, movement_stats


def _face_class(face, labels, probability_source, scale):
    point = face.representative_point()
    row_coordinate = float(np.clip(point.y - 0.5, 0.0, labels.shape[0] - 1.0))
    col_coordinate = float(np.clip(point.x - 0.5, 0.0, labels.shape[1] - 1.0))
    row0 = int(math.floor(row_coordinate))
    col0 = int(math.floor(col_coordinate))
    row1 = min(row0 + 1, labels.shape[0] - 1)
    col1 = min(col0 + 1, labels.shape[1] - 1)
    values = probability_source.read(
        window=Window(col0, row0, col1 - col0 + 1, row1 - row0 + 1),
        out_dtype="float32",
    )
    values *= float(scale)
    row_fraction = row_coordinate - row0
    col_fraction = col_coordinate - col0
    top = values[:, 0, 0] * (1.0 - col_fraction) + values[:, 0, -1] * col_fraction
    bottom = values[:, -1, 0] * (1.0 - col_fraction) + values[:, -1, -1] * col_fraction
    scores = top * (1.0 - row_fraction) + bottom * row_fraction
    return int(scores.argmax())


def build_subpixel_coverage(
    probability_source,
    labels,
    interpolation_strength,
    stripe_rows,
    progress_callback=None,
):
    segments, junction_count, scale, movement_stats = build_shared_linework(
        probability_source,
        labels,
        interpolation_strength=interpolation_strength,
        stripe_rows=stripe_rows,
        progress_callback=progress_callback,
    )
    line_array = shapely.linestrings(np.asarray(segments, dtype=np.float64))
    faces = [
        geometry
        for geometry in shapely.get_parts(shapely.polygonize(line_array))
        if geometry.geom_type == "Polygon" and geometry.area > 1e-8
    ]
    if not faces:
        raise SubpixelVectorizationError("shared linework did not produce any polygon faces")
    by_class = {}
    for face in faces:
        class_index = _face_class(face, labels, probability_source, scale)
        if 0 <= class_index < CLASS_COUNT:
            by_class.setdefault(class_index, []).append(face)
    records = []
    for class_index, parts in sorted(by_class.items()):
        merged = shapely.union_all(np.asarray(parts, dtype=object))
        for polygon in shapely.get_parts(merged):
            if polygon.geom_type == "Polygon" and polygon.area > 1e-8:
                records.append((class_index, polygon))
    records.sort(
        key=lambda item: (
            item[0],
            round(item[1].bounds[1], 9),
            round(item[1].bounds[0], 9),
            round(item[1].area, 9),
        )
    )
    return records, {
        "segment_count": len(segments),
        "junction_count": junction_count,
        "face_count": len(faces),
        "linework_row_total": int(labels.shape[0] - 1),
        "probability_scale": scale,
        "interpolation_crossing_count": movement_stats["count"],
        "maximum_interpolation_displacement_px": movement_stats["maximum"],
    }


def _simplify_coverage(records, tolerance):
    before = np.asarray([geometry for _class, geometry in records], dtype=object)
    after = np.asarray(
        shapely.coverage_simplify(
            before,
            tolerance=float(tolerance),
            simplify_boundary=False,
        ),
        dtype=object,
    )
    if len(after) != len(before):
        raise SubpixelVectorizationError("coverage simplify changed the object count")
    return [(records[index][0], before[index], after[index]) for index in range(len(records))]


def _directed_vertex_distance(source, target):
    coordinates = shapely.get_coordinates(source.boundary)
    if not len(coordinates):
        return 0.0
    distances = shapely.distance(shapely.points(coordinates), target.boundary)
    return float(np.max(distances)) if len(distances) else 0.0


def _simplification_displacement(simplified):
    maximum = 0.0
    by_class = {}
    for class_index, before, after in simplified:
        displacement = max(
            _directed_vertex_distance(before, after),
            _directed_vertex_distance(after, before),
        )
        maximum = max(maximum, displacement)
        key = str(int(class_index))
        by_class[key] = max(by_class.get(key, 0.0), displacement)
    return maximum, by_class


def _simplify_coverage_bounded(records, target_tolerance, max_deviation):
    target = float(target_tolerance)
    candidates = []
    for value in (target, target * 0.75, target * 0.5, target * 0.25, 0.0):
        value = round(value, 12)
        if value not in candidates:
            candidates.append(value)

    attempts = []
    for tolerance in candidates:
        if tolerance > 0.0:
            simplified = _simplify_coverage(records, tolerance)
        else:
            simplified = [
                (class_index, geometry, geometry)
                for class_index, geometry in records
            ]
        maximum, by_class = _simplification_displacement(simplified)
        attempts.append({
            "coverage_tolerance_px": tolerance,
            "maximum_simplification_displacement_px": maximum,
        })
        if maximum <= float(max_deviation) + 1e-8:
            return simplified, tolerance, maximum, by_class, attempts

    raise SubpixelVectorizationError(
        "coverage simplify could not satisfy the formal movement gate; "
        f"max_deviation_px={float(max_deviation):.6f}; attempts={attempts}"
    )


def _read_raw_pixel_records(path, transform):
    records = []
    with fiona.open(path, layer=LAYER_NAME) as source:
        for feature in source:
            if not feature.get("geometry"):
                continue
            properties = dict(feature.get("properties") or {})
            records.append(
                (
                    int(properties["class_code"]),
                    _to_pixel(shape(feature["geometry"]), transform),
                )
            )
    if not records:
        raise SubpixelVectorizationError("raw polygon baseline is empty")
    return records


def _coverage_stats(geometries):
    values = np.asarray(geometries, dtype=object)
    union = shapely.union_all(values)
    area_sum = float(np.sum(shapely.area(values)))
    union_area = float(union.area)
    return {
        "feature_count": len(values),
        "coordinate_count": int(np.sum(shapely.get_num_coordinates(values))),
        "invalid_count": int(np.count_nonzero(~shapely.is_valid(values))),
        "empty_count": int(np.count_nonzero(shapely.is_empty(values))),
        "area_sum_px2": area_sum,
        "union_area_px2": union_area,
        "overlap_area_px2": max(0.0, area_sum - union_area),
        "coverage_is_valid": bool(shapely.coverage_is_valid(values)),
    }, union


def _class_geometries(records):
    grouped = {}
    for class_code, geometry in records:
        grouped.setdefault(int(class_code), []).append(geometry)
    return {
        code: shapely.union_all(np.asarray(parts, dtype=object))
        for code, parts in grouped.items()
    }


def _maximum_class_boundary_distance(raw_records, formal_records):
    raw_classes = _class_geometries(raw_records)
    formal_classes = _class_geometries(formal_records)
    distances = {}
    for code in sorted(set(raw_classes) & set(formal_classes)):
        distances[str(code)] = float(
            shapely.hausdorff_distance(
                raw_classes[code].boundary,
                formal_classes[code].boundary,
            )
        )
    return max(distances.values(), default=0.0), distances


def _confidence_stats(source, pixel_geometry):
    min_x, min_y, max_x, max_y = pixel_geometry.bounds
    col_start = max(0, int(math.floor(min_x)))
    row_start = max(0, int(math.floor(min_y)))
    col_end = min(source.width, int(math.ceil(max_x)))
    row_end = min(source.height, int(math.ceil(max_y)))
    if col_end <= col_start or row_end <= row_start:
        return 0.0, 0.0
    data = source.read(
        1,
        window=Window(col_start, row_start, col_end - col_start, row_end - row_start),
    )
    selected = features.geometry_mask(
        [mapping(pixel_geometry)],
        out_shape=data.shape,
        transform=rasterio.Affine.translation(col_start, row_start),
        invert=True,
        all_touched=True,
    )
    values = data[selected]
    if not values.size:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def _write_formal(
    output_path,
    simplified,
    confidence_source,
    transform,
    crs,
    class_map,
    index_to_code,
    *,
    run_id,
    stream_id,
    result_kind,
    model_id,
    fusion_profile_id,
    model_version,
):
    schema = {
        "geometry": SCHEMA["geometry"],
        "properties": dict(SCHEMA["properties"]),
    }
    schema["properties"].update({
        "regularization_method": "str",
        "regularization_version": "str",
        "regularization_status": "str",
        "vertex_count_before": "int",
        "vertex_count_after": "int",
        "area_change_ratio": "float",
    })
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".gpkg", dir=output_path.parent
    )
    os.close(fd)
    os.unlink(temporary_name)
    created_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    token = _stream_token(stream_id)
    try:
        with fiona.open(
            temporary_name,
            "w",
            driver="GPKG",
            layer=LAYER_NAME,
            schema=schema,
            crs_wkt=crs.to_wkt() if crs else None,
        ) as destination:
            for sequence, (class_index, before, after) in enumerate(simplified, 1):
                class_code = int(index_to_code[class_index])
                confidence_mean, confidence_std = _confidence_stats(
                    confidence_source, after
                )
                destination.write({
                    "geometry": mapping(_to_map(after, transform)),
                    "properties": {
                        "run_id": run_id,
                        "result_stream_id": stream_id,
                        "result_kind": result_kind,
                        "model_id": model_id,
                        "fusion_profile_id": fusion_profile_id,
                        "object_id": f"{run_id}_{token}_{sequence:06d}",
                        "part_id": "000",
                        "class_code": class_code,
                        "class_name": class_map[class_code],
                        "confidence_mean": confidence_mean,
                        "confidence_std": confidence_std,
                        "model_version": model_version,
                        "source": (
                            "semantic_model" if result_kind == "model" else "semantic_fusion"
                        ),
                        "created_at": created_at,
                        "regularization_method": METHOD,
                        "regularization_version": VERSION,
                        "regularization_status": "passed",
                        "vertex_count_before": int(shapely.get_num_coordinates(before)),
                        "vertex_count_after": int(shapely.get_num_coordinates(after)),
                        "area_change_ratio": float(
                            (after.area - before.area) / before.area
                            if before.area > 0 else 0.0
                        ),
                    },
                })
        os.replace(temporary_name, output_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def vectorize_probability_mosaic(
    probability_path,
    mask_path,
    confidence_path,
    raw_path,
    output_path,
    report_path,
    class_map_path,
    *,
    run_id,
    stream_id,
    result_kind,
    model_version="",
    model_id="",
    fusion_profile_id="",
    interpolation_strength=DEFAULT_INTERPOLATION_STRENGTH,
    coverage_tolerance_px=DEFAULT_COVERAGE_TOLERANCE_PX,
    max_deviation_px=DEFAULT_MAX_DEVIATION_PX,
    stripe_rows=DEFAULT_STRIPE_ROWS,
):
    started = time.monotonic()
    if result_kind not in ("model", "fusion"):
        raise SubpixelVectorizationError("result_kind must be model or fusion")
    if result_kind == "model" and not model_id:
        raise SubpixelVectorizationError("model result requires model_id")
    if result_kind == "fusion" and not fusion_profile_id:
        raise SubpixelVectorizationError("fusion result requires fusion_profile_id")
    if not math.isclose(float(interpolation_strength), 1.0, abs_tol=1e-12):
        raise SubpixelVectorizationError("formal interpolation_strength must equal 1.0")
    if not math.isclose(float(coverage_tolerance_px), 1.0, abs_tol=1e-12):
        raise SubpixelVectorizationError("formal coverage_tolerance_px must equal 1.0")

    class_map, index_to_code, background_index = load_class_map(class_map_path)
    probability_path = Path(probability_path).resolve()
    mask_path = Path(mask_path).resolve()
    confidence_path = Path(confidence_path).resolve()
    raw_path = Path(raw_path).resolve()
    output_path = Path(output_path).resolve()
    report_path = Path(report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(mask_path) as mask_source, rasterio.open(
        confidence_path
    ) as confidence_source, rasterio.open(probability_path) as probability_source:
        labels = mask_source.read(1).astype(np.int16)
        transform = mask_source.transform
        crs = mask_source.crs
        geometry = (mask_source.width, mask_source.height, transform, crs)
        for name, source in (
            ("confidence", confidence_source),
            ("probability", probability_source),
        ):
            current = (source.width, source.height, source.transform, source.crs)
            if current != geometry:
                raise SubpixelVectorizationError(
                    f"{name} mosaic does not match mask geometry"
                )
        valid_values = labels[labels != background_index]
        if valid_values.size == 0 or np.any((valid_values < 0) | (valid_values >= CLASS_COUNT)):
            raise SubpixelVectorizationError("mask contains no valid 14-class pixels")

        def progress_callback(current, total, segment_count):
            print(
                json.dumps(
                    {
                        "event": "subpixel_linework_progress",
                        "stream_id": stream_id,
                        "current": int(current),
                        "total": int(total),
                        "segments": int(segment_count),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                flush=True,
            )

        subpixel, network = build_subpixel_coverage(
            probability_source,
            labels,
            float(interpolation_strength),
            int(stripe_rows),
            progress_callback=progress_callback,
        )
        (
            simplified,
            effective_coverage_tolerance_px,
            simplification_displacement,
            simplification_displacement_by_class,
            simplification_attempts,
        ) = _simplify_coverage_bounded(
            subpixel,
            float(coverage_tolerance_px),
            float(max_deviation_px),
        )
        formal_records = [
            (int(index_to_code[class_index]), after)
            for class_index, _before, after in simplified
        ]
        raw_records = _read_raw_pixel_records(raw_path, transform)
        raw_stats, raw_union = _coverage_stats([geometry for _code, geometry in raw_records])
        subpixel_stats, _ = _coverage_stats([geometry for _code, geometry in subpixel])
        formal_stats, formal_union = _coverage_stats(
            [geometry for _code, geometry in formal_records]
        )
        union_difference = float(raw_union.symmetric_difference(formal_union).area)
        raw_to_formal_distance, raw_to_formal_distance_by_class = (
            _maximum_class_boundary_distance(raw_records, formal_records)
        )
        raw_classes = set(_class_geometries(raw_records))
        formal_classes = set(_class_geometries(formal_records))
        area_epsilon = max(1e-6, raw_stats["union_area_px2"] * 1e-9)
        checks = {
            "geometry_valid": formal_stats["invalid_count"] == 0,
            "geometry_nonempty": formal_stats["empty_count"] == 0,
            "class_set_preserved": raw_classes == formal_classes,
            "subpixel_formal_object_count_preserved": len(subpixel) == len(simplified),
            "overlap_within_tolerance": formal_stats["overlap_area_px2"] <= area_epsilon,
            "covered_region_unchanged": union_difference <= area_epsilon,
            "interpolation_movement_within_tolerance": (
                network["maximum_interpolation_displacement_px"] <= 0.5 + 1e-8
            ),
            "simplification_movement_within_tolerance": (
                simplification_displacement <= float(max_deviation_px) + 1e-8
            ),
        }
        validation = {
            "passed": all(checks.values()),
            "checks": checks,
            "raw": raw_stats,
            "subpixel": subpixel_stats,
            "formal": formal_stats,
            "union_symmetric_difference_px2": union_difference,
            "maximum_interpolation_displacement_px": network[
                "maximum_interpolation_displacement_px"
            ],
            "maximum_simplification_displacement_px": simplification_displacement,
            "maximum_simplification_displacement_by_class_index_px": (
                simplification_displacement_by_class
            ),
            "target_coverage_tolerance_px": float(coverage_tolerance_px),
            "effective_coverage_tolerance_px": effective_coverage_tolerance_px,
            "simplification_attempts": simplification_attempts,
            "raw_to_formal_class_hausdorff_px": raw_to_formal_distance,
            "raw_to_formal_class_hausdorff_by_class_code_px": (
                raw_to_formal_distance_by_class
            ),
            "area_epsilon_px2": area_epsilon,
        }
        if not validation["passed"]:
            failed = ", ".join(key for key, value in checks.items() if not value)
            raise SubpixelVectorizationError(
                f"subpixel vectorization hard gate failed: {failed}; "
                "max_interpolation_displacement_px="
                f"{network['maximum_interpolation_displacement_px']:.6f}; "
                "max_simplification_displacement_px="
                f"{simplification_displacement:.6f}"
            )

        _write_formal(
            output_path,
            simplified,
            confidence_source,
            transform,
            crs,
            class_map,
            index_to_code,
            run_id=run_id,
            stream_id=stream_id,
            result_kind=result_kind,
            model_id=model_id,
            fusion_profile_id=fusion_profile_id,
            model_version=model_version,
        )

    report = {
        "schema_version": 2,
        "status": "passed",
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": METHOD,
        "version": VERSION,
        "config": {
            "interpolation_strength": float(interpolation_strength),
            "probability_smoothing_sigma": 0.0,
            "coverage_tolerance_px": float(coverage_tolerance_px),
            "effective_coverage_tolerance_px": effective_coverage_tolerance_px,
            "max_deviation_px": float(max_deviation_px),
            "stripe_rows": int(stripe_rows),
            "qsdk_noninferiority_margin_px": QSDK_NONINFERIORITY_MARGIN_PX,
        },
        "input": str(raw_path),
        "input_sha256": _sha256(raw_path),
        "probability_mosaic": str(probability_path),
        "probability_mosaic_sha256": _sha256(probability_path),
        "mask_mosaic": str(mask_path),
        "mask_mosaic_sha256": _sha256(mask_path),
        "confidence_mosaic": str(confidence_path),
        "confidence_mosaic_sha256": _sha256(confidence_path),
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "network": network,
        "metrics": {
            "raw": _staircase_metrics([geometry for _code, geometry in raw_records]),
            "formal": _staircase_metrics([geometry for _code, geometry in formal_records]),
        },
        "validation": validation,
        "elapsed_sec": round(time.monotonic() - started, 3),
    }
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    with open(temporary_report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_report, report_path)
    print(
        json.dumps(
            {
                "event": "subpixel_vectorization_finished",
                "status": "passed",
                "stream_id": stream_id,
                "output": str(output_path),
                "report": str(report_path),
                "polygon_count": len(formal_records),
                "segment_count": network["segment_count"],
                "current": network["linework_row_total"],
                "total": network["linework_row_total"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Vectorize a 14-class probability mosaic with one shared subpixel network"
    )
    parser.add_argument("--probabilities", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--confidence", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stream-id", required=True)
    parser.add_argument("--result-kind", choices=("model", "fusion"), required=True)
    parser.add_argument("--model-version", default="")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--fusion-profile-id", default="")
    parser.add_argument("--class-map", required=True)
    parser.add_argument(
        "--interpolation-strength",
        type=float,
        default=DEFAULT_INTERPOLATION_STRENGTH,
    )
    parser.add_argument(
        "--coverage-tolerance-px",
        type=float,
        default=DEFAULT_COVERAGE_TOLERANCE_PX,
    )
    parser.add_argument(
        "--max-deviation-px",
        type=float,
        default=DEFAULT_MAX_DEVIATION_PX,
    )
    parser.add_argument("--stripe-rows", type=int, default=DEFAULT_STRIPE_ROWS)
    args = parser.parse_args(argv)
    try:
        vectorize_probability_mosaic(
            args.probabilities,
            args.mask,
            args.confidence,
            args.raw,
            args.output,
            args.report,
            args.class_map,
            run_id=args.run_id,
            stream_id=args.stream_id,
            result_kind=args.result_kind,
            model_version=args.model_version,
            model_id=args.model_id,
            fusion_profile_id=args.fusion_profile_id,
            interpolation_strength=args.interpolation_strength,
            coverage_tolerance_px=args.coverage_tolerance_px,
            max_deviation_px=args.max_deviation_px,
            stripe_rows=args.stripe_rows,
        )
        return 0
    except Exception as exc:
        logger.error("[subpixel_vectorizer] ERROR: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
