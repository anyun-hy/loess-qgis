"""Create a reproducible two-model QGIS acceptance run specification.

The generated TorchScript files are deterministic test assets.  They exercise
the production subprocess pipeline but are never registered as release models.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import secrets
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "qgis_plugins"
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from labeling_tool.core.run_spec import (  # noqa: E402
    CLASS_ORDER,
    create_run_spec,
    reserve_run_directory,
)


class FixtureModel(torch.nn.Module):
    def __init__(self, reverse: bool = False):
        super().__init__()
        bias = torch.arange(14, dtype=torch.float32)
        if reverse:
            bias = torch.flip(bias, dims=(0,))
        self.register_buffer("bias", bias.view(1, 14, 1, 1))

    def forward(self, image):
        return image[:, :1] * 0.01 + self.bias


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_model(path: Path, reverse: bool) -> str:
    model = FixtureModel(reverse=reverse).eval()
    traced = torch.jit.trace(model, torch.zeros(1, 3, 512, 512), strict=True)
    torch.jit.save(traced, path)
    return sha256(path)


def write_tile(path: Path, origin_x: float, value: int) -> None:
    data = np.full((3, 512, 512), value, dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=512,
        height=512,
        count=3,
        dtype="uint8",
        crs="EPSG:4490",
        transform=from_origin(origin_x, 40.0, 0.0001, 0.0001),
    ) as dataset:
        dataset.write(data)


def metric_set(miou: float) -> dict:
    return {
        "miou": miou,
        "mf1": 70.0,
        "oa": 80.0,
        "kappa": 0.7,
        "per_class": [{} for _ in CLASS_ORDER],
        "confusion_matrix": [[0 for _ in CLASS_ORDER] for _ in CLASS_ORDER],
    }


def approved_profile(models: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "profile_id": "e9_equal_average",
        "status": "approved",
        "strategy": "equal_probability_average",
        "class_order": CLASS_ORDER,
        "input": {"height": 512, "width": 512, "channels": 3, "dtype": "float32"},
        "models": [
            {
                "model_id": model["model_id"],
                "artifact": model["artifact"],
                "sha256": model["sha256"],
                "temperature": 1.0,
            }
            for model in models
        ],
        "weights": [[0.5, 0.5] for _ in CLASS_ORDER],
        "dataset": {
            "validation_count": 2,
            "validation_fingerprint": "1" * 64,
            "validation_sample_ids_sha256": "2" * 64,
            "test_count": 2,
            "test_fingerprint": "3" * 64,
            "test_sample_ids_sha256": "4" * 64,
        },
        "metrics": {
            "units": {"miou_mf1_oa_per_class": "percent", "kappa": "ratio"},
            "baseline": metric_set(65.0),
            "fusion": metric_set(66.0),
        },
        "approval": {
            "passed": True,
            "criterion": "fusion.test_miou > exported_swin_baseline.test_miou",
        },
        "integrity": {
            "frozen_strategy_sha256": "5" * 64,
            "test_backend": "torchscript",
            "validation_test_overlap": 0,
            "baseline_model": {
                "model_id": models[0]["model_id"],
                "artifact": models[0]["artifact"],
                "sha256": models[0]["sha256"],
            },
        },
    }


def create_fixture(output_root: Path) -> Path:
    output_root = output_root.expanduser().resolve()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    token = secrets.token_hex(3)
    asset_dir = output_root / "fixtures" / f"e9_{stamp}_{token}"
    asset_dir.mkdir(parents=True, exist_ok=False)

    models = []
    for model_id, reverse in (("fixture_forward", False), ("fixture_reverse", True)):
        artifact = f"{model_id}.torchscript.pt"
        path = asset_dir / artifact
        models.append({
            "model_id": model_id,
            "display_name": model_id,
            "version": "e9-fixture-v1",
            "artifact": artifact,
            "artifact_path": str(path),
            "sha256": write_model(path, reverse),
        })

    profile_path = asset_dir / "fusion_profile.json"
    profile_path.write_text(
        json.dumps(approved_profile(models), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    run_id, run_dir = reserve_run_directory(output_root)
    tile_dir = run_dir / "tmp" / "tiles"
    overlap = 64
    pixel_size = 0.0001
    tiles = []
    for col, value in ((0, 80), (1, 120)):
        origin_x = 100.0 + col * (512 - overlap) * pixel_size
        path = tile_dir / f"tile_0_{col}.tif"
        write_tile(path, origin_x, value)
        tiles.append({
            "tile_id": f"0_{col}",
            "row": 0,
            "col": col,
            "path": str(path),
            "width": 512,
            "height": 512,
            "bounds": {
                "xmin": origin_x,
                "ymin": 40.0 - 512 * pixel_size,
                "xmax": origin_x + 512 * pixel_size,
                "ymax": 40.0,
            },
        })

    processing_xmax = 100.0 + (512 + (512 - overlap)) * pixel_size
    _, spec_path = create_run_spec(
        output_root=output_root,
        raster_path=tiles[0]["path"],
        raster_crs="EPSG:4490",
        requested_extent=(100.0, 39.9488, processing_xmax, 40.0),
        processing_extent=(100.0, 39.9488, processing_xmax, 40.0),
        tiles=tiles,
        models=models,
        effective_device="cpu",
        keep_score_cache=False,
        overlap=overlap,
        fusion_profile_path=profile_path,
        config_fingerprint="e9-fixture-only",
        run_id=run_id,
        reserved_run_dir=run_dir,
    )
    return spec_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "output" / "e9_acceptance"),
    )
    args = parser.parse_args()
    spec_path = create_fixture(Path(args.output_root))
    print(json.dumps({"run_spec": str(spec_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
