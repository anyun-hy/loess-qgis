#!/usr/bin/env python3
"""Prepare a formal v5 L0 run from the immutable 238-Tile failure fixture."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import rasterio


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "qgis_plugins"
SCRIPTS_ROOT = ROOT / "inference_scripts"
for import_root in (PLUGIN_ROOT, SCRIPTS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from labeling_tool.core.run_builder_v5 import create_v5_run
from labeling_tool.core.run_spec import atomic_write_json, reserve_run_directory, sha256_file
from labeling_tool.core.work_package_planner import storage_preflight
from check_environment import _fingerprint
from deployment_config import load_and_validate_config


class L0PreparationError(RuntimeError):
    pass


def _load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise L0PreparationError(f"L0 destination already exists: {destination}")
    os.link(source, destination)


def prepare(source_run: Path, output_root: Path, *, device: str, run_id: str | None):
    source_run = source_run.resolve()
    source_spec_path = source_run / "run_spec.json"
    source_spec = _load(source_spec_path)
    source_tiles = list(source_spec.get("tiles") or [])
    if len(source_tiles) != 238:
        raise L0PreparationError(
            f"formal L0 source must contain exactly 238 Tiles, got {len(source_tiles)}"
        )
    rows = max(int(tile["row"]) for tile in source_tiles) + 1
    cols = max(int(tile["col"]) for tile in source_tiles) + 1
    if (rows, cols) != (14, 17):
        raise L0PreparationError(f"formal L0 grid must be 14 x 17, got {rows} x {cols}")

    effective, issues = load_and_validate_config(SCRIPTS_ROOT / "config.yaml")
    if issues:
        raise L0PreparationError(
            "deployment config is invalid: "
            + "; ".join(f"{item.path}: {item.message}" for item in issues)
        )
    profile_entry = next(
        (
            item
            for item in effective["fusion_profiles"]
            if item["profile_id"] == "l2_fusion_v1" and item["available"]
        ),
        None,
    )
    if profile_entry is None:
        raise L0PreparationError("approved l2_fusion_v1 is unavailable")
    required_ids = list(profile_entry["required_model_ids"])
    models_by_id = {item["model_id"]: item for item in effective["semantic_models"]}
    models = [models_by_id[model_id] for model_id in required_ids]

    first_source = Path(source_tiles[0]["path"]).resolve()
    if not first_source.is_file():
        raise L0PreparationError(f"source Tile is missing: {first_source}")
    with rasterio.open(first_source) as first_raster:
        if first_raster.shape != (512, 512) or first_raster.crs is None:
            raise L0PreparationError("source L0 Tile contract is not 512 x 512 georeferenced")
        resolution_x = abs(float(first_raster.transform.a))
        resolution_y = abs(float(first_raster.transform.e))
    first_row = sorted(
        (tile for tile in source_tiles if int(tile["row"]) == 0),
        key=lambda item: int(item["col"]),
    )
    stride = round(
        (float(first_row[1]["bounds"]["xmin"]) - float(first_row[0]["bounds"]["xmin"]))
        / resolution_x
    )
    overlap = 512 - int(stride)
    if overlap != 192:
        raise L0PreparationError(f"formal L0 overlap must be 192 px, got {overlap}")

    identifier, run_dir = reserve_run_directory(output_root, run_id)
    linked_tiles = []
    for tile in sorted(source_tiles, key=lambda item: (int(item["row"]), int(item["col"]))):
        row = int(tile["row"])
        col = int(tile["col"])
        source_path = Path(tile["path"]).resolve()
        if not source_path.is_file() or sha256_file(source_path) != str(tile["sha256"]):
            raise L0PreparationError(f"source Tile hash mismatch: {source_path}")
        destination = run_dir / "tmp" / "tiles" / f"tile_{row}_{col}.tif"
        _hardlink(source_path, destination)
        metadata_source = source_path.with_name(f"tile_{row}_{col}_meta.json")
        if metadata_source.is_file():
            _hardlink(
                metadata_source,
                destination.with_name(f"tile_{row}_{col}_meta.json"),
            )
        linked_tiles.append(
            {
                "row": row,
                "col": col,
                "path": str(destination),
                "sha256": str(tile["sha256"]),
                "bounds": dict(tile["bounds"]),
                "pixel_window": {
                    "x0": col * stride,
                    "y0": row * stride,
                    "x1": col * stride + 512,
                    "y1": row * stride + 512,
                },
            }
        )

    scaling = dict(effective["scaling"])
    if str(scaling.get("partition_halo_px", "auto")).lower() == "auto":
        scaling["partition_halo_px"] = max(overlap, int(scaling["seam_band_px"]))
    pixel_count = 512 * 512
    sample_tile_bytes = first_source.stat().st_size
    storage = storage_preflight(
        output_root,
        tile_count=len(linked_tiles),
        stream_count=len(models) + 1,
        permanent_bytes_per_tile_per_stream=pixel_count * 8,
        input_tile_bytes_per_tile=sample_tile_bytes,
        score_cache_budget_gb=float(scaling["score_cache_budget_gb"]),
        min_free_disk_gb=float(scaling["min_free_disk_gb"]),
        current_model_probability_bytes=pixel_count * 14 * 2,
        fusion_accumulator_bytes=pixel_count * 15 * 4,
        mask_confidence_workspace_bytes=pixel_count * (14 * 4 + 5),
        safety_margin_bytes=sample_tile_bytes,
    )
    processing = dict(source_spec["processing_extent"])
    spec, spec_path, database_path = create_v5_run(
        output_root=output_root,
        reserved_run_dir=run_dir,
        run_id=identifier,
        raster={
            "path": source_spec["raster"]["path"],
            "crs": source_spec["raster"]["crs"],
            "transform": [
                resolution_x,
                0.0,
                float(processing["xmin"]),
                0.0,
                -resolution_y,
                float(processing["ymax"]),
            ],
            "nodata": None,
        },
        requested_extent=source_spec["requested_extent"],
        processing_extent=processing,
        tile_rows=rows,
        tile_cols=cols,
        tiles=linked_tiles,
        models=models,
        effective_device=device,
        keep_score_cache=False,
        overlap=overlap,
        scaling=scaling,
        boundary_fitting=effective["boundary_fitting"],
        storage_report=storage,
        fusion={
            "profile_id": profile_entry["profile_id"],
            "version": str(profile_entry["profile"].get("version") or ""),
            "profile_path": profile_entry["file_path"],
            "profile": profile_entry["profile"],
        },
        skip_accepted=False,
        config_fingerprint=_fingerprint(SCRIPTS_ROOT),
    )
    report = {
        "schema_version": 1,
        "status": "prepared",
        "level": "L0",
        "source_run": str(source_run),
        "source_run_spec_sha256": sha256_file(source_spec_path),
        "run_id": identifier,
        "run_spec": str(spec_path),
        "state_db": str(database_path),
        "tile_count": len(linked_tiles),
        "tile_grid": {"rows": rows, "cols": cols, "overlap": overlap},
        "stream_ids": [item["stream_id"] for item in spec["streams"]],
        "storage_preflight": storage,
        "execution": "resume from QGIS4 V5AsyncInferenceRunner on MPS",
    }
    atomic_write_json(run_dir / "logs" / "l0_preparation_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run",
        default=str(ROOT / "output" / "runs" / "20260717_180420_d9d642"),
    )
    parser.add_argument("--output-root", default=str(ROOT / "output"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    report = prepare(
        Path(args.source_run),
        Path(args.output_root),
        device=args.device,
        run_id=args.run_id,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
