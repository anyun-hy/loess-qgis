"""Evaluate tile-seam quality against raster or vector reference labels."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject, transform_bounds, transform_geom


CLASS_CODES = [12, 13, 21, 31, 32, 33, 43, 51, 52, 53, 54, 61, 62, 71]
CODE_TO_INDEX = {code: index for index, code in enumerate(CLASS_CODES)}
TILE_PATTERN = re.compile(r"tile_(-?\d+)_(-?\d+)\.tif$")
VECTOR_SUFFIXES = {".shp", ".gpkg", ".geojson", ".json"}


class SeamValidationError(RuntimeError):
    pass


def _read_prediction(path: Path):
    with rasterio.open(path) as src:
        if src.count != 1:
            raise SeamValidationError(f"prediction must be single-band: {path}")
        values = src.read(1).astype(np.int16)
        profile = {
            "width": src.width,
            "height": src.height,
            "transform": src.transform,
            "crs": src.crs,
            "bounds": src.bounds,
        }
    if profile["crs"] is None:
        raise SeamValidationError(f"prediction has no CRS: {path}")
    invalid = (values < -1) | (values >= len(CLASS_CODES))
    if np.any(invalid):
        unique = np.unique(values[invalid])[:10].tolist()
        raise SeamValidationError(f"prediction contains invalid class indexes {unique}: {path}")
    return values, profile


def _reference_raster(path: Path, profile, values_are: str) -> np.ndarray:
    destination = np.full((profile["height"], profile["width"]), -1, dtype=np.int32)
    with rasterio.open(path) as src:
        source = src.read(1)
        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=profile["transform"],
            dst_crs=profile["crs"],
            dst_nodata=-1,
            resampling=Resampling.nearest,
        )
    if values_are == "indices":
        invalid = (destination < -1) | (destination >= len(CLASS_CODES))
        destination[invalid] = -1
        return destination.astype(np.int16)
    mapped = np.full(destination.shape, -1, dtype=np.int16)
    for code, index in CODE_TO_INDEX.items():
        mapped[destination == code] = index
    return mapped


def _reference_vector(path: Path, profile, field: str, layer: str | None) -> np.ndarray:
    try:
        import fiona
    except ImportError as exc:
        raise SeamValidationError("Fiona is required for vector reference labels") from exc
    open_kwargs = {"layer": layer} if layer else {}
    shapes = []
    with fiona.open(path, **open_kwargs) as source:
        if field not in (source.schema.get("properties") or {}):
            raise SeamValidationError(f"reference field {field!r} is missing from {path}")
        source_crs = source.crs_wkt or source.crs
        if not source_crs:
            raise SeamValidationError(f"reference vector has no CRS: {path}")
        bounds = profile["bounds"]
        query_bounds = transform_bounds(
            profile["crs"],
            source_crs,
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top,
            densify_pts=21,
        )
        for feature in source.filter(bbox=query_bounds):
            geometry = feature.get("geometry")
            if not geometry:
                continue
            raw_code = (feature.get("properties") or {}).get(field)
            try:
                code = int(raw_code)
            except (TypeError, ValueError):
                continue
            if code not in CODE_TO_INDEX:
                continue
            transformed = transform_geom(source_crs, profile["crs"], geometry, precision=-1)
            shapes.append((transformed, CODE_TO_INDEX[code]))
    if not shapes:
        raise SeamValidationError("reference vector has no mapped 14-class features in prediction extent")
    return rasterize(
        shapes,
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=-1,
        dtype="int16",
        all_touched=False,
    )


def _load_reference(path: Path, profile, field: str, layer: str | None, values_are: str):
    if path.suffix.lower() in VECTOR_SUFFIXES:
        return _reference_vector(path, profile, field, layer)
    return _reference_raster(path, profile, values_are)


def _tile_metadata(tile_dir: Path, prediction_crs):
    tiles = {}
    for path in sorted(tile_dir.glob("tile_*.tif")):
        match = TILE_PATTERN.match(path.name)
        if not match:
            continue
        with rasterio.open(path) as src:
            if src.crs != prediction_crs:
                raise SeamValidationError(f"tile CRS differs from prediction: {path}")
            tiles[(int(match.group(1)), int(match.group(2)))] = src.bounds
    if not tiles:
        raise SeamValidationError(f"no tile rasters found in {tile_dir}")
    return tiles


def _pixel_window(profile, left, bottom, right, top):
    inverse = ~profile["transform"]
    col0, row0 = inverse * (left, top)
    col1, row1 = inverse * (right, bottom)
    return (
        max(0, int(math.floor(row0))),
        min(profile["height"], int(math.ceil(row1))),
        max(0, int(math.floor(col0))),
        min(profile["width"], int(math.ceil(col1))),
    )


def build_seam_band(tile_dir: Path, profile, band_width: int) -> tuple[np.ndarray, list[dict]]:
    if band_width < 1:
        raise SeamValidationError("band_width must be positive")
    tiles = _tile_metadata(tile_dir, profile["crs"])
    seam = np.zeros((profile["height"], profile["width"]), dtype=bool)
    lines = []
    for (row, col), bounds in sorted(tiles.items()):
        right_neighbor = tiles.get((row, col + 1))
        if right_neighbor is not None:
            left = max(bounds.left, right_neighbor.left)
            right = min(bounds.right, right_neighbor.right)
            bottom = max(bounds.bottom, right_neighbor.bottom)
            top = min(bounds.top, right_neighbor.top)
            if left < right and bottom < top:
                midpoint = (left + right) / 2.0
                row0, row1, col0, col1 = _pixel_window(profile, midpoint, bottom, midpoint, top)
                center = int(round((col0 + col1) / 2.0))
                seam[row0:row1, max(0, center - band_width):min(profile["width"], center + band_width + 1)] = True
                lines.append({"orientation": "vertical", "row": row, "col": col, "pixel": center})
        lower_neighbor = tiles.get((row + 1, col))
        if lower_neighbor is not None:
            left = max(bounds.left, lower_neighbor.left)
            right = min(bounds.right, lower_neighbor.right)
            bottom = max(bounds.bottom, lower_neighbor.bottom)
            top = min(bounds.top, lower_neighbor.top)
            if left < right and bottom < top:
                midpoint = (bottom + top) / 2.0
                row0, row1, col0, col1 = _pixel_window(profile, left, midpoint, right, midpoint)
                center = int(round((row0 + row1) / 2.0))
                seam[max(0, center - band_width):min(profile["height"], center + band_width + 1), col0:col1] = True
                lines.append({"orientation": "horizontal", "row": row, "col": col, "pixel": center})
    if not lines or not np.any(seam):
        raise SeamValidationError("tile layout produced no internal seam band")
    return seam, lines


def _metrics(reference: np.ndarray, prediction: np.ndarray, selection: np.ndarray):
    valid = selection & (reference >= 0) & (prediction >= 0)
    count = int(np.count_nonzero(valid))
    if count == 0:
        return {"valid_pixels": 0, "accuracy": None, "miou": None, "per_class_iou": {}}
    encoded = reference[valid].astype(np.int64) * len(CLASS_CODES) + prediction[valid].astype(np.int64)
    confusion = np.bincount(encoded, minlength=len(CLASS_CODES) ** 2).reshape(
        len(CLASS_CODES), len(CLASS_CODES)
    )
    true_positive = np.diag(confusion).astype(np.float64)
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - true_positive
    present = union > 0
    iou = np.divide(true_positive, union, out=np.full(len(CLASS_CODES), np.nan), where=present)
    return {
        "valid_pixels": count,
        "accuracy": float(true_positive.sum() / count),
        "miou": float(np.nanmean(iou)),
        "per_class_iou": {
            str(code): (float(iou[index]) if present[index] else None)
            for index, code in enumerate(CLASS_CODES)
        },
    }


def _prediction_metrics(reference, prediction, seam):
    all_pixels = np.ones(reference.shape, dtype=bool)
    overall = _metrics(reference, prediction, all_pixels)
    seam_values = _metrics(reference, prediction, seam)
    non_seam = _metrics(reference, prediction, ~seam)
    if seam_values["valid_pixels"] == 0:
        raise SeamValidationError("seam-band has no valid reference pixels")
    if non_seam["valid_pixels"] == 0:
        raise SeamValidationError("non-seam area has no valid reference pixels")
    return {
        "overall": overall,
        "seam_band": seam_values,
        "non_seam": non_seam,
        "seam_accuracy_gap": non_seam["accuracy"] - seam_values["accuracy"],
    }


def _line_transition_rates(values: np.ndarray, lines: list[dict]) -> dict[str, float | None]:
    rates = {"vertical": [], "horizontal": []}
    seen = set()
    for line in lines:
        orientation = str(line["orientation"])
        pixel = int(line["pixel"])
        key = (orientation, pixel)
        if key in seen:
            continue
        seen.add(key)
        if orientation == "vertical" and 0 < pixel < values.shape[1]:
            first, second = values[:, pixel - 1], values[:, pixel]
        elif orientation == "horizontal" and 0 < pixel < values.shape[0]:
            first, second = values[pixel - 1, :], values[pixel, :]
        else:
            continue
        valid = (first >= 0) & (second >= 0)
        if np.any(valid):
            rates[orientation].append(float(np.mean(first[valid] != second[valid])))
    result = {
        orientation: (float(np.mean(items)) if items else None)
        for orientation, items in rates.items()
    }
    all_rates = rates["vertical"] + rates["horizontal"]
    result["mean"] = float(np.mean(all_rates)) if all_rates else None
    return result


def validate_seams(
    prediction_path,
    reference_path,
    tile_dir,
    output_path,
    *,
    baseline_path=None,
    reference_field="TDLYDM",
    reference_layer=None,
    reference_values="codes",
    band_width=32,
    tolerance=0.0,
):
    prediction_path = Path(prediction_path).resolve()
    reference_path = Path(reference_path).resolve()
    tile_dir = Path(tile_dir).resolve()
    output_path = Path(output_path).resolve()
    prediction, profile = _read_prediction(prediction_path)
    reference = _load_reference(
        reference_path,
        profile,
        reference_field,
        reference_layer,
        reference_values,
    )
    seam, lines = build_seam_band(tile_dir, profile, int(band_width))
    candidate = _prediction_metrics(reference, prediction, seam)
    candidate["line_transition_rate"] = _line_transition_rates(prediction, lines)
    reference_transitions = _line_transition_rates(reference, lines)
    report = {
        "schema_version": 1,
        "status": "measured",
        "prediction": str(prediction_path),
        "baseline": str(Path(baseline_path).resolve()) if baseline_path else "",
        "reference": str(reference_path),
        "reference_field": reference_field,
        "class_codes": CLASS_CODES,
        "band_width_px": int(band_width),
        "seam_lines": lines,
        "reference_line_transition_rate": reference_transitions,
        "candidate": candidate,
    }
    if baseline_path:
        baseline, baseline_profile = _read_prediction(Path(baseline_path).resolve())
        for key in ("width", "height", "transform", "crs"):
            if baseline_profile[key] != profile[key]:
                raise SeamValidationError(f"baseline {key} differs from prediction")
        baseline_metrics = _prediction_metrics(reference, baseline, seam)
        baseline_metrics["line_transition_rate"] = _line_transition_rates(baseline, lines)
        deltas = {
            "seam_accuracy": candidate["seam_band"]["accuracy"] - baseline_metrics["seam_band"]["accuracy"],
            "seam_miou": candidate["seam_band"]["miou"] - baseline_metrics["seam_band"]["miou"],
            "seam_accuracy_gap": candidate["seam_accuracy_gap"] - baseline_metrics["seam_accuracy_gap"],
            "line_transition_mean": (
                candidate["line_transition_rate"]["mean"]
                - baseline_metrics["line_transition_rate"]["mean"]
            ),
        }
        checks = {
            "seam_accuracy_not_lower": deltas["seam_accuracy"] >= -float(tolerance),
            "seam_miou_not_lower": deltas["seam_miou"] >= -float(tolerance),
            "seam_gap_not_wider": deltas["seam_accuracy_gap"] <= float(tolerance),
        }
        report.update({
            "status": "passed" if all(checks.values()) else "failed",
            "baseline_metrics": baseline_metrics,
            "deltas": deltas,
            "checks": checks,
            "tolerance": float(tolerance),
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Validate mosaic seam bands against reference labels")
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-field", default="TDLYDM")
    parser.add_argument("--reference-layer")
    parser.add_argument("--reference-values", choices=("codes", "indices"), default="codes")
    parser.add_argument("--tile-dir", required=True)
    parser.add_argument("--band-width", type=int, default=32)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        report = validate_seams(
            args.prediction,
            args.reference,
            args.tile_dir,
            args.output,
            baseline_path=args.baseline,
            reference_field=args.reference_field,
            reference_layer=args.reference_layer,
            reference_values=args.reference_values,
            band_width=args.band_width,
            tolerance=args.tolerance,
        )
    except Exception as exc:
        print(f"[seam_band_validate] ERROR: {exc}", file=sys.stderr)
        return 2
    print("[seam_band_validate] " + json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    return 3 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
