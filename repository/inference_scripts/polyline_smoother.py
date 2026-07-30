"""Fast visual smoothing for jagged LineString geometry using cubic B-splines."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable

import fiona
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, mapping, shape


class PolylineSmoothingError(RuntimeError):
    pass


@dataclass(frozen=True)
class SmoothingConfig:
    smoothing_factor: float = 1.0
    curve_sampling_spacing: float = 0.5
    max_chord_error: float = 0.25
    max_segment_arc_length: float = 8.0
    max_deviation: float | None = None
    spline_degree: int = 3
    min_point_count: int = 4
    min_length: float = 3.0
    min_strength: float = 0.05

    def validate(self) -> None:
        for name in (
            "smoothing_factor",
            "curve_sampling_spacing",
            "max_chord_error",
            "max_segment_arc_length",
            "min_length",
            "min_strength",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_deviation is not None:
            value = float(self.max_deviation)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("max_deviation must be finite and positive when set")
        if not 1 <= int(self.spline_degree) <= 5:
            raise ValueError("spline_degree must be between 1 and 5")
        if int(self.min_point_count) < 2:
            raise ValueError("min_point_count must be at least 2")
        if float(self.min_strength) > 1.0:
            raise ValueError("min_strength must not exceed 1")


@dataclass(frozen=True)
class SmoothingResult:
    points: np.ndarray
    status: str
    strength: float
    max_deviation: float
    mean_deviation: float
    input_point_count: int
    dense_point_count: int
    output_point_count: int
    closed: bool
    max_chord_error: float
    max_segment_arc_length: float
    reason: str = ""


def _sanitize(points: Iterable[Iterable[float]]) -> np.ndarray:
    values = np.asarray(list(points), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("polyline points must have shape [N,2] or [N,3]")
    values = values[:, :2]
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("polyline must contain at least two finite points")
    keep = np.ones(len(values), dtype=bool)
    keep[1:] = np.any(np.abs(np.diff(values, axis=0)) > 1e-12, axis=1)
    cleaned = values[keep]
    if len(cleaned) < 2:
        raise ValueError("polyline has zero length")
    return cleaned


def _parameterize(points: np.ndarray, closed: bool) -> tuple[np.ndarray, float]:
    working = points
    if closed and not np.allclose(points[0], points[-1], atol=1e-12, rtol=0):
        working = np.vstack((points, points[0]))
    lengths = np.linalg.norm(np.diff(working, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    if total <= 0:
        raise ValueError("polyline has zero length")
    return cumulative / total, total


def _resample(points: np.ndarray, parameter: np.ndarray, samples: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            np.interp(samples, parameter, points[:, 0]),
            np.interp(samples, parameter, points[:, 1]),
        )
    )


def _deviation(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    reference_tree = cKDTree(reference)
    candidate_tree = cKDTree(candidate)
    candidate_distances = reference_tree.query(candidate, workers=1)[0]
    reference_distances = candidate_tree.query(reference, workers=1)[0]
    maximum = max(
        float(candidate_distances.max(initial=0.0)),
        float(reference_distances.max(initial=0.0)),
    )
    mean = 0.5 * (
        float(candidate_distances.mean()) + float(reference_distances.mean())
    )
    return maximum, mean


def _point_to_chord_distances(
    points: np.ndarray,
    first: int,
    last: int,
) -> np.ndarray:
    values = points[first + 1 : last]
    if not len(values):
        return np.empty(0, dtype=np.float64)
    start = points[first]
    stop = points[last]
    vector = stop - start
    squared = float(np.dot(vector, vector))
    if squared <= 1e-24:
        return np.linalg.norm(values - start, axis=1)
    factors = np.clip(((values - start) @ vector) / squared, 0.0, 1.0)
    projections = start + factors[:, None] * vector
    return np.linalg.norm(values - projections, axis=1)


def adaptive_sample_curve(
    dense_points: Iterable[Iterable[float]],
    *,
    max_chord_error: float,
    max_segment_arc_length: float,
) -> tuple[np.ndarray, float, float]:
    """Linearize one fitted curve under strict error and arc-length bounds."""

    points = _sanitize(dense_points)
    if not math.isfinite(max_chord_error) or max_chord_error <= 0:
        raise ValueError("max_chord_error must be finite and positive")
    if not math.isfinite(max_segment_arc_length) or max_segment_arc_length <= 0:
        raise ValueError("max_segment_arc_length must be finite and positive")
    if len(points) <= 2:
        return points.copy(), 0.0, 0.0

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    minimum_output_points = min(4, len(points))
    seed_indices = sorted(
        {
            int(round(value))
            for value in np.linspace(
                0,
                len(points) - 1,
                minimum_output_points,
            )
        }
    )
    keep = set(seed_indices)
    stack = list(zip(seed_indices, seed_indices[1:]))
    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        distances = _point_to_chord_distances(points, first, last)
        maximum = float(distances.max(initial=0.0))
        arc_length = float(cumulative[last] - cumulative[first])
        if maximum <= max_chord_error and arc_length <= max_segment_arc_length:
            continue
        if maximum > max_chord_error and len(distances):
            split = first + 1 + int(np.argmax(distances))
        else:
            halfway = 0.5 * (cumulative[first] + cumulative[last])
            split = int(np.searchsorted(cumulative, halfway))
            split = max(first + 1, min(split, last - 1))
        keep.add(split)
        stack.append((first, split))
        stack.append((split, last))

    indices = sorted(keep)
    observed_error = 0.0
    observed_arc_length = 0.0
    for first, last in zip(indices, indices[1:]):
        observed_error = max(
            observed_error,
            float(
                _point_to_chord_distances(points, first, last).max(initial=0.0)
            ),
        )
        observed_arc_length = max(
            observed_arc_length,
            float(cumulative[last] - cumulative[first]),
        )
    return points[indices], observed_error, observed_arc_length


def _unchanged(points: np.ndarray, closed: bool, reason: str) -> SmoothingResult:
    return SmoothingResult(
        points=points.copy(),
        status="unchanged",
        strength=0.0,
        max_deviation=0.0,
        mean_deviation=0.0,
        input_point_count=len(points),
        dense_point_count=len(points),
        output_point_count=len(points),
        closed=closed,
        max_chord_error=0.0,
        max_segment_arc_length=0.0,
        reason=reason,
    )


def smooth_polyline(
    points: Iterable[Iterable[float]],
    config: SmoothingConfig | None = None,
) -> SmoothingResult:
    """Fit and resample one open or closed polyline.

    Units are the same as the input coordinates. For raster-derived pixel
    coordinates, the defaults mean 1 px smoothing, a 0.5 px internal curve
    sample, and final linearization bounded by 0.25 px chord error and 8 px
    curve arc length. Deviation is reported but is not limited unless
    max_deviation is set.
    """

    config = config or SmoothingConfig()
    config.validate()
    values = _sanitize(points)
    closed = len(values) >= 4 and np.allclose(
        values[0], values[-1], atol=1e-12, rtol=0
    )
    fit_points = values[:-1] if closed else values
    if len(fit_points) < int(config.min_point_count):
        return _unchanged(values, closed, "too_few_points")

    source_points = np.vstack((fit_points, fit_points[0])) if closed else fit_points
    source_parameter, total = _parameterize(source_points, closed=False)
    if total < float(config.min_length):
        return _unchanged(values, closed, "too_short")

    sample_count = max(
        int(config.spline_degree) + 1,
        int(math.ceil(total / float(config.curve_sampling_spacing))) + 1,
    )
    samples = np.linspace(0.0, 1.0, sample_count)
    reference = _resample(source_points, source_parameter, samples)

    # Sparse staircases can make a cubic spline behave like an unstable exact
    # interpolant. Fit against uniformly densified samples of the same source
    # polyline so the curve remains a smoothing approximation.
    fit_spacing = float(config.curve_sampling_spacing) * 2.0
    fit_count = max(
        int(config.spline_degree) + 5,
        int(math.ceil(total / fit_spacing)) + 1,
    )
    fit_parameter = np.linspace(0.0, 1.0, fit_count)
    fit_source = _resample(source_points, source_parameter, fit_parameter)
    if closed:
        fit_points = fit_source[:-1]
        fit_source = np.vstack((fit_points, fit_points[0]))
    else:
        fit_points = fit_source

    degree = min(int(config.spline_degree), len(fit_points) - 1)
    weights = np.ones(len(fit_source), dtype=np.float64)
    if not closed:
        weights[0] = weights[-1] = 1000.0

    def fit_at(strength: float):
        smoothing = (
            len(fit_source)
            * (float(config.smoothing_factor) * strength) ** 2
        )
        spline, _ = splprep(
            [fit_source[:, 0], fit_source[:, 1]],
            u=np.linspace(0.0, 1.0, len(fit_source)),
            w=weights,
            k=degree,
            s=smoothing,
            per=closed,
        )
        dense_candidate = np.column_stack(splev(samples, spline))
        if closed:
            dense_candidate[-1] = dense_candidate[0]
        else:
            dense_candidate[0] = values[0]
            dense_candidate[-1] = values[-1]
        maximum, mean = _deviation(reference, dense_candidate)
        candidate, chord_error, arc_length = adaptive_sample_curve(
            dense_candidate,
            max_chord_error=float(config.max_chord_error),
            max_segment_arc_length=float(config.max_segment_arc_length),
        )
        return (
            candidate,
            maximum,
            mean,
            len(dense_candidate),
            chord_error,
            arc_length,
        )

    if closed:
        reference[-1] = reference[0]

    strengths = [1.0]
    if config.max_deviation is not None:
        strengths.extend([0.75, 0.5, 0.35, 0.25, 0.18, 0.125, 0.09, 0.0625])
        strengths = [value for value in strengths if value >= float(config.min_strength)]
        if strengths[-1] > float(config.min_strength):
            strengths.append(float(config.min_strength))
    failure = "deviation_limit"
    for strength in strengths:
        try:
            (
                candidate,
                maximum,
                mean,
                dense_point_count,
                chord_error,
                arc_length,
            ) = fit_at(strength)
        except (TypeError, ValueError) as error:
            failure = f"spline_failed:{error}"
            continue
        if config.max_deviation is None or maximum <= float(config.max_deviation):
            return SmoothingResult(
                points=candidate,
                status="smoothed",
                strength=float(strength),
                max_deviation=float(maximum),
                mean_deviation=float(mean),
                input_point_count=len(values),
                dense_point_count=int(dense_point_count),
                output_point_count=len(candidate),
                closed=closed,
                max_chord_error=float(chord_error),
                max_segment_arc_length=float(arc_length),
            )
    return _unchanged(values, closed, failure)


def _smooth_geometry(geometry, config: SmoothingConfig):
    if isinstance(geometry, LineString):
        result = smooth_polyline(geometry.coords, config)
        return LineString(result.points), [result]
    if isinstance(geometry, MultiLineString):
        lines = []
        results = []
        for part in geometry.geoms:
            result = smooth_polyline(part.coords, config)
            lines.append(LineString(result.points))
            results.append(result)
        return MultiLineString(lines), results
    raise PolylineSmoothingError(
        f"only LineString and MultiLineString are supported, got {geometry.geom_type}"
    )


def smooth_vector_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    config: SmoothingConfig | None = None,
    input_layer: str | None = None,
    output_layer: str = "smoothed_lines",
) -> dict[str, Any]:
    config = config or SmoothingConfig()
    config.validate()
    source_path = Path(input_path).expanduser().resolve()
    target_path = Path(output_path).expanduser().resolve()
    if not source_path.is_file():
        raise PolylineSmoothingError(f"input does not exist: {source_path}")
    if target_path.exists():
        raise PolylineSmoothingError(f"output already exists: {target_path}")
    driver = {
        ".gpkg": "GPKG",
        ".geojson": "GeoJSON",
        ".json": "GeoJSON",
        ".shp": "ESRI Shapefile",
    }.get(target_path.suffix.lower())
    if driver is None:
        raise PolylineSmoothingError("output must be .gpkg, .geojson, .json or .shp")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    feature_count = 0
    line_count = 0
    smoothed_count = 0
    maximum_deviation = 0.0
    open_kwargs = {"layer": input_layer} if input_layer else {}
    try:
        with fiona.open(source_path, **open_kwargs) as source:
            schema = dict(source.schema)
            if schema.get("geometry") not in {"LineString", "MultiLineString"}:
                raise PolylineSmoothingError(
                    "input layer geometry must be LineString or MultiLineString"
                )
            output_kwargs: dict[str, Any] = {
                "driver": driver,
                "schema": schema,
                "crs": source.crs,
                "encoding": source.encoding or "UTF-8",
            }
            if source.crs_wkt:
                output_kwargs["crs_wkt"] = source.crs_wkt
            if driver == "GPKG":
                output_kwargs["layer"] = output_layer
            with fiona.open(target_path, "w", **output_kwargs) as target:
                for feature in source:
                    geometry, results = _smooth_geometry(shape(feature["geometry"]), config)
                    output_feature = dict(feature)
                    output_feature["geometry"] = mapping(geometry)
                    target.write(output_feature)
                    feature_count += 1
                    line_count += len(results)
                    smoothed_count += sum(item.status == "smoothed" for item in results)
                    maximum_deviation = max(
                        maximum_deviation,
                        *(item.max_deviation for item in results),
                    )
    except Exception:
        if target_path.exists():
            target_path.unlink()
        raise
    return {
        "algorithm": "divider_cubic_bspline_adaptive_v2",
        "input": str(source_path),
        "output": str(target_path),
        "feature_count": feature_count,
        "line_count": line_count,
        "smoothed_count": smoothed_count,
        "max_deviation": maximum_deviation,
        "elapsed_seconds": time.monotonic() - started,
        "config": asdict(config),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-layer")
    parser.add_argument("--output-layer", default="smoothed_lines")
    parser.add_argument("--report")
    parser.add_argument("--smoothing-factor", type=float, default=1.0)
    parser.add_argument("--curve-sampling-spacing", type=float, default=0.5)
    parser.add_argument("--max-chord-error", type=float, default=0.25)
    parser.add_argument("--max-segment-arc-length", type=float, default=8.0)
    parser.add_argument(
        "--max-deviation",
        type=float,
        help="optional displacement limit; omitted by default for pure visual fitting",
    )
    parser.add_argument("--spline-degree", type=int, default=3)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = SmoothingConfig(
        smoothing_factor=args.smoothing_factor,
        curve_sampling_spacing=args.curve_sampling_spacing,
        max_chord_error=args.max_chord_error,
        max_segment_arc_length=args.max_segment_arc_length,
        max_deviation=args.max_deviation,
        spline_degree=args.spline_degree,
    )
    try:
        report = smooth_vector_file(
            args.input,
            args.output,
            config=config,
            input_layer=args.input_layer,
            output_layer=args.output_layer,
        )
        if args.report:
            report_path = Path(args.report).expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(report, ensure_ascii=False))
        return 0
    except Exception as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
