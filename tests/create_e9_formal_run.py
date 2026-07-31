"""Create one traceable E9 run from the deployed formal assets and real imagery.

Unlike ``create_e9_pipeline_fixture.py``, this helper never creates model or
fusion artifacts.  It validates the deployed registry, extracts one real
512x512 RGB tile, and freezes the plugin's native run specification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import rasterio
from rasterio.windows import Window


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = ROOT / "inference_scripts"
PLUGIN_ROOT = ROOT / "qgis_plugins"
for path in (SCRIPTS_ROOT, PLUGIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deployment_config import load_and_validate_config  # noqa: E402
from labeling_tool.core.run_spec import (  # noqa: E402
    create_run_spec,
    reserve_run_directory,
    run_tile_cache_dir,
)


# Keep this list synchronized with core/inference_config.py.  Importing that
# QGIS QObject module in a headless acceptance helper would make the run depend
# on the QGIS Python runtime before the actual QGIS acceptance stage.
FINGERPRINT_FILES = (
    "config.sh",
    "config.yaml",
    "_device.py",
    "deployment_config.py",
    "check_environment.py",
    "environment-ubuntu-cu124.yml",
    "environment-macos-qgis4.yml",
    "semantic_batch.py",
    "torchscript_runtime.py",
    "work_package_runtime.py",
    "partition_mosaic.py",
    "incremental_fusion.py",
    "finalize_partition_rasters.py",
    "assemble_stream.py",
    "scale_acceptance.py",
    "runtime_metrics.py",
    "difference_runtime.py",
    "accepted_score.py",
    "boundary_fitting/__init__.py",
    "boundary_fitting/unit_runtime.py",
    "polyline_smoother.py",
    "common_boundary_smoother.py",
    "sam3_interactive_worker.py",
    "sam3_refine.py",
    "run_polyline_smooth.sh",
    "run_work_package.sh",
    "run_finalize_partition_rasters.sh",
    "run_unit_fit.sh",
    "run_assemble_stream.sh",
    "run_scale_acceptance.sh",
    "run_sam3_interactive_worker.sh",
)


def config_fingerprint() -> str:
    digest = hashlib.sha256()
    for name in FINGERPRINT_FILES:
        path = SCRIPTS_ROOT / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
    return "sha256:" + digest.hexdigest()


def extract_center_tile(source: Path, destination: Path) -> dict:
    with rasterio.open(source) as src:
        if src.count < 3:
            raise RuntimeError(f"formal source raster must have at least 3 bands: {source}")
        if src.width < 512 or src.height < 512:
            raise RuntimeError(f"formal source raster is smaller than 512x512: {source}")
        col_off = (src.width - 512) // 2
        row_off = (src.height - 512) // 2
        window = Window(col_off, row_off, 512, 512)
        data = src.read((1, 2, 3), window=window)
        if data.dtype.name != "uint8":
            raise RuntimeError(f"formal source raster must be uint8 RGB, got {data.dtype}")
        transform = src.window_transform(window)
        bounds = rasterio.windows.bounds(window, src.transform)
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            width=512,
            height=512,
            count=3,
            dtype="uint8",
            transform=transform,
            compress="lzw",
            nodata=None,
        )
        crs = src.crs

    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(destination, "w", **profile) as dst:
        dst.write(data)

    return {
        "crs": crs.to_string() if crs else "",
        "bounds": {
            "xmin": float(bounds[0]),
            "ymin": float(bounds[1]),
            "xmax": float(bounds[2]),
            "ymax": float(bounds[3]),
        },
        "window": {
            "row_off": int(row_off),
            "col_off": int(col_off),
            "width": 512,
            "height": 512,
        },
    }


def create_formal_run(output_root: Path, source_raster: Path, device: str) -> Path:
    config_path = SCRIPTS_ROOT / "config.yaml"
    effective, issues = load_and_validate_config(
        config_path,
        verify_files=True,
        verify_hashes=True,
    )
    if issues:
        details = "; ".join(f"{item.path}: {item.message}" for item in issues)
        raise RuntimeError(f"formal deployment config is invalid: {details}")

    models = [model for model in effective["semantic_models"] if model["enabled"]]
    profiles = [
        profile
        for profile in effective["fusion_profiles"]
        if profile["enabled"] and profile["available"]
    ]
    if len(profiles) != 1:
        raise RuntimeError(f"expected exactly one approved fusion profile, got {len(profiles)}")
    profile = profiles[0]
    required = profile["required_model_ids"]
    selected = [model for model in models if model["model_id"] in required]
    if [model["model_id"] for model in selected] != required:
        raise RuntimeError(
            "enabled formal models do not match approved profile order: "
            f"selected={[model['model_id'] for model in selected]}, required={required}"
        )

    run_id, run_dir = reserve_run_directory(output_root)
    tile_path = run_tile_cache_dir(output_root, run_id) / "tile_0_0.tif"
    tile_meta = extract_center_tile(source_raster, tile_path)
    tile = {
        "tile_id": "0_0",
        "row": 0,
        "col": 0,
        "path": str(tile_path),
        "width": 512,
        "height": 512,
        "bounds": tile_meta["bounds"],
    }
    _, spec_path = create_run_spec(
        output_root=output_root,
        raster_path=source_raster,
        raster_crs=tile_meta["crs"],
        requested_extent=tile_meta["bounds"],
        processing_extent=tile_meta["bounds"],
        tiles=[tile],
        models=selected,
        effective_device=device,
        keep_score_cache=True,
        overlap=64,
        skip_accepted=False,
        fusion_profile_path=profile["file_path"],
        config_fingerprint=config_fingerprint(),
        run_id=run_id,
        reserved_run_dir=run_dir,
    )
    evidence = {
        "schema_version": 1,
        "purpose": "E9 formal deployment acceptance",
        "source_raster": str(source_raster.resolve()),
        "source_window": tile_meta["window"],
        "run_spec": str(spec_path),
        "device": device,
        "models": [
            {
                "model_id": model["model_id"],
                "artifact": model["artifact"],
                "sha256": model["sha256"],
            }
            for model in selected
        ],
        "fusion_profile": {
            "profile_id": profile["profile_id"],
            "path": profile["file_path"],
            "status": profile["status"],
            "strategy": profile["strategy"],
        },
        "config_fingerprint": config_fingerprint(),
    }
    (run_dir / "formal_input_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return spec_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "output" / "e9_formal_cpu"),
    )
    parser.add_argument(
        "--source-raster",
        default=str(ROOT / "data" / "raw" / "Google_loess" / "loess_1m_clip_RGB_4490.tif"),
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    spec_path = create_formal_run(
        Path(args.output_root).expanduser().resolve(),
        Path(args.source_raster).expanduser().resolve(),
        args.device,
    )
    print(json.dumps({"run_spec": str(spec_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
