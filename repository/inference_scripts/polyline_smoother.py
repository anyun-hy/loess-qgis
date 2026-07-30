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
from scipy.interpolate import PPoly, splprep, splev
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
    curve_evaluation_count: int
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


def _deviation(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> tuple[float, float]:
    """Measure symmetric sampled distance between source and fitted curves."""

    if reference.shape != candidate.shape:
        raise ValueError("deviation samples must have equal shapes")
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


def _point_to_chord_distances(points: np.ndarray) -> np.ndarray:
    values = points[1:-1]
    if not len(values):
        return np.empty(0, dtype=np.float64)
    start = points[0]
    stop = points[-1]
    vector = stop - start
    squared = float(np.dot(vector, vector))
    if squared <= 1e-24:
        return np.linalg.norm(values - start, axis=1)
    factors = np.clip(((values - start) @ vector) / squared, 0.0, 1.0)
    projections = start + factors[:, None] * vector
    return np.linalg.norm(values - projections, axis=1)


def _power_to_bernstein_controls(
    power_coefficients: np.ndarray,
) -> np.ndarray:
    """Convert normalized ascending power coefficients to Bézier controls."""

    degree = len(power_coefficients) - 1
    controls = np.zeros_like(power_coefficients)
    for control_index in range(degree + 1):
        for power in range(control_index + 1):
            controls[control_index] += (
                math.comb(control_index, power)
                / math.comb(degree, power)
            ) * power_coefficients[power]
    return controls


def _spline_bezier_spans(
    spline,
    *,
    first_point: np.ndarray,
    last_point: np.ndarray,
) -> list[np.ndarray]:
    """Return exact Bézier spans, including a linear endpoint correction."""

    knots, coefficients, degree = spline
    coordinate_coefficients = np.asarray(coefficients, dtype=np.float64)
    if coordinate_coefficients.ndim != 2 or coordinate_coefficients.shape[0] != 2:
        raise ValueError("fitted spline must contain exactly two coordinates")
    pieces = [
        PPoly.from_spline((knots, coordinate_coefficients[axis], degree))
        for axis in range(2)
    ]
    breaks = pieces[0].x
    domain_start = float(knots[degree])
    domain_stop = float(knots[-degree - 1])
    domain_length = domain_stop - domain_start
    if not math.isfinite(domain_length) or domain_length <= 0:
        raise ValueError("fitted spline has an invalid parameter domain")

    spans: list[tuple[float, float, np.ndarray]] = []
    for interval, (start, stop) in enumerate(zip(breaks, breaks[1:])):
        start = float(start)
        stop = float(stop)
        if (
            stop <= start
            or start < domain_start - 1e-12
            or stop > domain_stop + 1e-12
        ):
            continue
        width = stop - start
        power = np.empty((degree + 1, 2), dtype=np.float64)
        for exponent in range(degree + 1):
            row = degree - exponent
            power[exponent] = [
                piece.c[row, interval] * width**exponent
                for piece in pieces
            ]
        spans.append(
            (
                start,
                stop,
                _power_to_bernstein_controls(power),
            )
        )
    if not spans:
        raise ValueError("fitted spline contains no non-empty spans")

    start_delta = np.asarray(first_point, dtype=np.float64) - spans[0][2][0]
    stop_delta = np.asarray(last_point, dtype=np.float64) - spans[-1][2][-1]
    corrected = []
    for start, stop, controls in spans:
        start_fraction = (start - domain_start) / domain_length
        stop_fraction = (stop - domain_start) / domain_length
        correction_start = (
            (1.0 - start_fraction) * start_delta
            + start_fraction * stop_delta
        )
        correction_stop = (
            (1.0 - stop_fraction) * start_delta
            + stop_fraction * stop_delta
        )
        fractions = np.linspace(0.0, 1.0, degree + 1)[:, None]
        correction_controls = (
            (1.0 - fractions) * correction_start
            + fractions * correction_stop
        )
        corrected.append(controls + correction_controls)
    corrected[0][0] = first_point
    corrected[-1][-1] = last_point
    return corrected


def _split_bezier_batch(
    controls: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a batch of Bézier spans at parameter 0.5."""

    working = np.asarray(controls, dtype=np.float64)
    left = [working[:, 0]]
    right = [working[:, -1]]
    while working.shape[1] > 1:
        working = 0.5 * (working[:, :-1] + working[:, 1:])
        left.append(working[:, 0])
        right.append(working[:, -1])
    return (
        np.stack(left, axis=1),
        np.stack(right[::-1], axis=1),
    )


def _adaptive_sample_spline(
    spline,
    *,
    first_point: Iterable[float],
    last_point: Iterable[float],
    closed: bool,
    max_chord_error: float,
    max_segment_arc_length: float,
) -> tuple[np.ndarray, float, float, int]:
    """Directly linearize a fitted spline using certified Bézier bounds."""

    if not math.isfinite(max_chord_error) or max_chord_error <= 0:
        raise ValueError("max_chord_error must be finite and positive")
    if not math.isfinite(max_segment_arc_length) or max_segment_arc_length <= 0:
        raise ValueError("max_segment_arc_length must be finite and positive")
    first = np.asarray(first_point, dtype=np.float64)[:2]
    last = np.asarray(last_point, dtype=np.float64)[:2]
    if first.shape != (2,) or last.shape != (2,):
        raise ValueError("spline endpoints must contain two coordinates")
    spans = _spline_bezier_spans(
        spline,
        first_point=first,
        last_point=last,
    )
    active = np.stack(spans)
    active_start = np.arange(len(active), dtype=np.float64)
    active_stop = active_start + 1.0
    accepted_controls: list[np.ndarray] = []
    accepted_start: list[np.ndarray] = []
    observed_error_bound = 0.0
    observed_arc_bound = 0.0
    evaluation_count = 0
    for depth in range(33):
        evaluation_count += len(active)
        starts = active[:, 0]
        stops = active[:, -1]
        chords = stops - starts
        squared = np.einsum("ij,ij->i", chords, chords)
        interior = active[:, 1:-1]
        if interior.shape[1]:
            offsets = interior - starts[:, None, :]
            fractions = np.divide(
                np.einsum("nki,ni->nk", offsets, chords),
                squared[:, None],
                out=np.zeros((len(active), interior.shape[1])),
                where=squared[:, None] > 1e-24,
            )
            fractions = np.clip(fractions, 0.0, 1.0)
            projections = (
                starts[:, None, :]
                + fractions[:, :, None] * chords[:, None, :]
            )
            chord_error_bounds = np.linalg.norm(
                interior - projections,
                axis=2,
            ).max(axis=1, initial=0.0)
            degenerate = squared <= 1e-24
            if np.any(degenerate):
                chord_error_bounds[degenerate] = np.linalg.norm(
                    offsets[degenerate],
                    axis=2,
                ).max(axis=1, initial=0.0)
        else:
            chord_error_bounds = np.zeros(len(active), dtype=np.float64)
        arc_length_bounds = np.linalg.norm(
            np.diff(active, axis=1),
            axis=2,
        ).sum(axis=1)
        accepted = (
            (chord_error_bounds <= max_chord_error)
            & (arc_length_bounds <= max_segment_arc_length)
        )
        if np.any(accepted):
            accepted_controls.append(active[accepted])
            accepted_start.append(active_start[accepted])
            observed_error_bound = max(
                observed_error_bound,
                float(chord_error_bounds[accepted].max(initial=0.0)),
            )
            observed_arc_bound = max(
                observed_arc_bound,
                float(arc_length_bounds[accepted].max(initial=0.0)),
            )
        rejected = ~accepted
        if not np.any(rejected):
            break
        if depth >= 32:
            raise ValueError(
                "adaptive spline subdivision exceeded its safety depth"
            )
        controls = active[rejected]
        left, right = _split_bezier_batch(controls)
        midpoint_parameter = 0.5 * (
            active_start[rejected] + active_stop[rejected]
        )
        active = np.concatenate((left, right))
        active_start = np.concatenate(
            (active_start[rejected], midpoint_parameter)
        )
        active_stop = np.concatenate(
            (midpoint_parameter, active_stop[rejected])
        )

    controls = np.concatenate(accepted_controls)
    order = np.argsort(np.concatenate(accepted_start), kind="stable")
    controls = controls[order]
    points = np.concatenate(
        (controls[0, :1], controls[:, -1]),
        axis=0,
    )
    if closed:
        points[-1] = points[0]
        if len(points) < 4:
            raise ValueError(
                "adaptive spline produced too few points for a closed ring"
            )
    else:
        points[0] = first
        points[-1] = last
    if not np.isfinite(points).all():
        raise ValueError("adaptive spline produced non-finite coordinates")
    return (
        points,
        observed_error_bound,
        observed_arc_bound,
        evaluation_count,
    )


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
        curve_evaluation_count=len(points),
        reason=reason,
    )


def smooth_polyline(
    points: Iterable[Iterable[float]],
    config: SmoothingConfig | None = None,
) -> SmoothingResult:
    """Fit and resample one open or closed polyline.

    Units are the same as the input coordinates. For raster-derived pixel
    coordinates, the defaults mean 1 px smoothing and direct linearization
    bounded by 0.25 px chord error and 8 px curve arc length. The 0.5 px
    spacing remains an equivalent-count/reporting baseline and is not
    materialized on the production path. Deviation is reported but is not
    limited unless max_deviation is set.
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

    dense_equivalent_count = max(
        int(config.spline_degree) + 1,
        int(math.ceil(total / float(config.curve_sampling_spacing))) + 1,
    )

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
        candidate, chord_error, arc_length, evaluation_count = (
            _adaptive_sample_spline(
                spline,
                first_point=values[0],
                last_point=values[0] if closed else values[-1],
                closed=closed,
                max_chord_error=float(config.max_chord_error),
                max_segment_arc_length=float(
                    config.max_segment_arc_length
                ),
            )
        )
        if config.max_deviation is None:
            deviation_parameter = fit_parameter
            deviation_source = fit_source
        else:
            deviation_parameter = np.linspace(
                0.0,
                1.0,
                dense_equivalent_count,
            )
            deviation_source = _resample(
                source_points,
                source_parameter,
                deviation_parameter,
            )
        fit_candidate = np.column_stack(
            splev(deviation_parameter, spline)
        )
        fractions = deviation_parameter[:, None]
        fit_candidate += (
            (1.0 - fractions) * (values[0] - fit_candidate[0])
            + fractions
            * (
                (values[0] if closed else values[-1])
                - fit_candidate[-1]
            )
        )
        maximum, mean = _deviation(
            deviation_source,
            fit_candidate,
        )
        return (
            candidate,
            maximum,
            mean,
            dense_equivalent_count,
            chord_error,
            arc_length,
            evaluation_count,
        )

    strengths = [1.0]
    if config.max_deviation is not None:
        strengths.extend(
            [
                0.75,
                0.5,
                0.35,
                0.25,
                0.18,
                0.125,
                0.09,
                0.0625,
                0.04,
                0.025,
                0.0125,
            ]
        )
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
                evaluation_count,
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
                curve_evaluation_count=int(evaluation_count),
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
