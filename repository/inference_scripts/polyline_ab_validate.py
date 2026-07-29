"""Create a visual A/B report for the standalone polyline smoother."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import fiona
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, mapping

from polyline_smoother import SmoothingConfig, smooth_polyline


def _direction_energy(points: np.ndarray) -> float:
    vectors = np.diff(points, axis=0)
    nonzero = np.linalg.norm(vectors, axis=1) > 1e-12
    vectors = vectors[nonzero]
    if len(vectors) < 2:
        return 0.0
    angles = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
    return float(np.mean(np.abs(np.diff(angles))))


def _select(diagnostics, count: int):
    candidates = []
    for item in diagnostics:
        points = np.asarray(item.get("raw_points") or [], dtype=np.float64)
        if len(points) < 24:
            continue
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        energy = _direction_energy(points)
        if length < 20 or energy <= 0:
            continue
        candidates.append((energy * min(length, 200.0), length, item, points))
    candidates.sort(key=lambda value: (value[0], value[1]), reverse=True)
    selected = []
    signatures = set()
    for candidate in candidates:
        signature = tuple(candidate[2].get("signature") or ())
        if signature in signatures and len(signatures) < count:
            continue
        signatures.add(signature)
        selected.append(candidate)
        if len(selected) == count:
            break
    if len(selected) < count:
        used = {id(item[2]) for item in selected}
        selected.extend(item for item in candidates if id(item[2]) not in used)
        selected = selected[:count]
    if len(selected) < count:
        raise RuntimeError(f"only found {len(selected)} suitable polylines")
    return selected


def _draw_ab(rows, geometries, output_path: Path) -> None:
    panel_width = 720
    panel_height = 900
    margin = 38
    image = Image.new("RGB", (panel_width * len(rows), panel_height), "white")
    for index, (row, (raw, smoothed)) in enumerate(zip(rows, geometries)):
        panel = Image.new("RGB", (panel_width, panel_height), "white")
        draw = ImageDraw.Draw(panel)
        combined = np.vstack((raw, smoothed))
        whole_bounds = (combined.min(axis=0), combined.max(axis=0))

        vectors = np.diff(raw, axis=0)
        angles = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
        turns = np.abs(np.diff(angles))
        window = min(80, max(20, len(raw) // 8))
        kernel = np.ones(window, dtype=np.float64)
        center = int(np.argmax(np.convolve(turns, kernel, mode="same"))) + 1
        start = max(0, center - window)
        end = min(len(raw), center + window)
        zoom_raw = raw[start:end]
        zoom_minimum = zoom_raw.min(axis=0)
        zoom_maximum = zoom_raw.max(axis=0)
        zoom_span = np.maximum(zoom_maximum - zoom_minimum, 1.0)
        zoom_bounds = (
            zoom_minimum - zoom_span * 0.08,
            zoom_maximum + zoom_span * 0.08,
        )

        def screen(points, bounds, top, bottom):
            minimum, maximum = bounds
            span = np.maximum(maximum - minimum, 1e-9)
            scale = min(
                (panel_width - 2 * margin) / span[0],
                (bottom - top - 2 * margin) / span[1],
            )
            x = margin + (points[:, 0] - minimum[0]) * scale
            y = bottom - margin - (points[:, 1] - minimum[1]) * scale
            return list(map(tuple, np.column_stack((x, y))))

        def plot(points, bounds, top, bottom, color, width=3):
            draw.line(screen(points, bounds, top, bottom), fill=color, width=width)

        draw.rectangle((1, 1, panel_width - 2, panel_height - 2), outline="#d0d0d0")
        title = (
            f"#{row['sample_id']} classes={row['signature']}  "
            f"energy {row['raw_direction_energy']:.3f} -> "
            f"{row['smoothed_direction_energy']:.3f}  "
            f"max={row['max_deviation']:.3f}"
        )
        draw.text((12, 12), title, fill="black")
        draw.text((12, 38), "Whole line overlay", fill="#333333")
        plot(raw, whole_bounds, 45, 310, "#555555", 3)
        plot(smoothed, whole_bounds, 45, 310, "#d22f27", 3)
        draw.text((12, 315), "Raw zoom", fill="#555555")
        plot(raw, zoom_bounds, 325, 600, "#555555", 4)
        draw.text((12, 605), "Cubic B-Spline zoom", fill="#d22f27")
        plot(smoothed, zoom_bounds, 615, 890, "#d22f27", 4)
        image.paste(panel, (index * panel_width, 0))
    image.save(output_path)


def run(report_path: Path, output_dir: Path, count: int = 3):
    source = json.loads(report_path.read_text(encoding="utf-8"))
    selected = _select(source.get("diagnostics") or [], count)
    output_dir.mkdir(parents=True, exist_ok=True)
    gpkg_path = output_dir / "polyline_ab.gpkg"
    if gpkg_path.exists():
        gpkg_path.unlink()

    config = SmoothingConfig()
    rows = []
    geometries = []
    for sample_index, (_score, length, diagnostic, points) in enumerate(selected, 1):
        started = time.perf_counter()
        result = smooth_polyline(points, config)
        elapsed = time.perf_counter() - started
        raw_energy = _direction_energy(points)
        smooth_energy = _direction_energy(result.points)
        row = {
            "sample_id": sample_index,
            "chain_id": diagnostic.get("chain_id"),
            "signature": list(diagnostic.get("signature") or ()),
            "length": length,
            "status": result.status,
            "input_points": len(points),
            "output_points": len(result.points),
            "raw_direction_energy": raw_energy,
            "smoothed_direction_energy": smooth_energy,
            "energy_ratio": smooth_energy / raw_energy if raw_energy else 0.0,
            "strength": result.strength,
            "max_deviation": result.max_deviation,
            "mean_deviation": result.mean_deviation,
            "elapsed_seconds": elapsed,
        }
        rows.append(row)
        geometries.append((points, result.points))

    schema = {
        "geometry": "LineString",
        "properties": {
            "sample_id": "int",
            "version": "str:12",
            "chain_id": "str:32",
            "signature": "str:64",
            "energy": "float",
            "max_dev": "float",
        },
    }
    with fiona.open(gpkg_path, "w", driver="GPKG", layer="polyline_ab", schema=schema) as sink:
        for row, (raw, smoothed) in zip(rows, geometries):
            for version, points in (("raw", raw), ("smoothed", smoothed)):
                sink.write(
                    {
                        "geometry": mapping(LineString(points)),
                        "properties": {
                            "sample_id": row["sample_id"],
                            "version": version,
                            "chain_id": str(row["chain_id"]),
                            "signature": ",".join(map(str, row["signature"])),
                            "energy": (
                                row["raw_direction_energy"]
                                if version == "raw"
                                else row["smoothed_direction_energy"]
                            ),
                            "max_dev": 0.0 if version == "raw" else row["max_deviation"],
                        },
                    }
                )

    image_path = output_dir / "polyline_ab.png"
    _draw_ab(rows, geometries, image_path)

    result = {
        "algorithm": "cubic_bspline_v1",
        "source_report": str(report_path.resolve()),
        "config": {
            "smoothing_factor": config.smoothing_factor,
            "output_spacing": config.output_spacing,
            "max_deviation": config.max_deviation,
            "spline_degree": config.spline_degree,
        },
        "samples": rows,
        "all_energy_reduced": all(row["energy_ratio"] < 1.0 for row in rows),
        "max_deviation_observed": max(row["max_deviation"] for row in rows),
        "gpkg": str(gpkg_path.resolve()),
        "image": str(image_path.resolve()),
    }
    report_output = output_dir / "polyline_ab_report.json"
    report_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()
    result = run(Path(args.report), Path(args.output_dir), args.count)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
