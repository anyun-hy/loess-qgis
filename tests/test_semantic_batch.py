import hashlib
import json
import time

import numpy as np
import rasterio
import torch

from semantic_batch import run_batch


class _FixtureModel(torch.nn.Module):
    def __init__(self, reverse=False):
        super().__init__()
        bias = torch.arange(14, dtype=torch.float32)
        if reverse:
            bias = torch.flip(bias, dims=(0,))
        self.register_buffer("bias", bias.view(1, 14, 1, 1))

    def forward(self, image):
        signal = image[:, :1] * 0.01
        return signal + self.bias


def _write_model(path, reverse=False):
    model = _FixtureModel(reverse=reverse).eval()
    traced = torch.jit.trace(model, torch.zeros(1, 3, 512, 512), strict=True)
    torch.jit.save(traced, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_tile(path, size=512, value=100):
    data = np.full((3, size, size), value, dtype=np.uint8)
    transform = rasterio.transform.from_origin(100.0, 40.0, 0.0001, 0.0001)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=3,
        dtype="uint8",
        crs="EPSG:4490",
        transform=transform,
    ) as dst:
        dst.write(data)


def _run_spec(tmp_path, tiles):
    model_path = tmp_path / "model_a.torchscript.pt"
    model_sha = _write_model(model_path)
    run_dir = tmp_path / "runs" / "test_run"
    spec = {
        "schema_version": 1,
        "run_id": "test_run",
        "run_dir": str(run_dir),
        "runtime": {"effective_device": "cpu"},
        "models": [{
            "model_id": "model_a",
            "version": "fixture-v1",
            "artifact_path": str(model_path),
            "sha256": model_sha,
        }],
        "tiles": tiles,
    }
    spec_path = tmp_path / "run_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return spec_path, run_dir


def test_batch_loads_model_once_and_writes_every_tile_output(tmp_path):
    tiles = []
    for index in range(2):
        tile_path = tmp_path / f"tile_{index}.tif"
        _write_tile(tile_path, value=80 + index)
        tiles.append({"tile_id": f"0_{index}", "row": 0, "col": index, "path": str(tile_path)})
    spec_path, run_dir = _run_spec(tmp_path, tiles)

    result = run_batch(spec_path, "model_a", device="cpu")

    assert result["model_load_count"] == 1
    assert result["runtime"]["mode"] == "direct"
    assert result["completed_count"] == 2
    assert result["failed_count"] == 0
    for entry in result["tiles"]:
        with rasterio.open(entry["mask"]) as src:
            mask = src.read(1)
            assert mask.shape == (512, 512)
            assert np.all(mask == 13)
            assert src.crs.to_epsg() == 4490
        with rasterio.open(entry["confidence"]) as src:
            confidence = src.read(1)
            assert confidence.shape == (512, 512)
            assert np.all((confidence > 0) & (confidence <= 1))
        with np.load(entry["scores"]) as cached:
            assert cached["probabilities"].shape == (14, 512, 512)
            assert cached["probabilities"].dtype == np.float16
    assert (run_dir / "tmp" / "streams" / "model_a" / "stream_manifest.json").is_file()


def test_resume_reuses_only_fully_validated_tile_outputs(tmp_path):
    tile_path = tmp_path / "tile.tif"
    _write_tile(tile_path)
    spec_path, _ = _run_spec(tmp_path, [{"tile_id": "0_0", "path": str(tile_path)}])
    first = run_batch(spec_path, "model_a", device="cpu")
    mask_path = first["tiles"][0]["mask"]
    first_mtime = __import__("os").stat(mask_path).st_mtime_ns
    time.sleep(0.01)

    resumed = run_batch(spec_path, "model_a", device="cpu", resume=True)

    assert resumed["completed_count"] == 1
    assert resumed["reused_count"] == 1
    assert resumed["failed_count"] == 0
    assert __import__("os").stat(mask_path).st_mtime_ns == first_mtime


def test_bad_tile_is_recorded_without_discarding_successful_tiles(tmp_path):
    good = tmp_path / "good.tif"
    bad = tmp_path / "bad.tif"
    _write_tile(good)
    _write_tile(bad, size=256)
    spec_path, _ = _run_spec(tmp_path, [
        {"tile_id": "0_0", "path": str(good)},
        {"tile_id": "0_1", "path": str(bad)},
    ])

    result = run_batch(spec_path, "model_a", device="cpu")

    assert result["completed_count"] == 1
    assert result["failed_count"] == 1
    assert result["failures"][0]["tile_id"] == "0_1"
    assert "512x512" in result["failures"][0]["error"]


def test_two_registered_models_keep_independent_result_streams(tmp_path):
    tile_path = tmp_path / "tile.tif"
    _write_tile(tile_path)
    spec_path, run_dir = _run_spec(tmp_path, [{"tile_id": "0_0", "path": str(tile_path)}])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    model_b_path = tmp_path / "model_b.torchscript.pt"
    model_b_sha = _write_model(model_b_path, reverse=True)
    spec["models"].append({
        "model_id": "model_b",
        "version": "fixture-v1",
        "artifact_path": str(model_b_path),
        "sha256": model_b_sha,
    })
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    result_a = run_batch(spec_path, "model_a", device="cpu")
    result_b = run_batch(spec_path, "model_b", device="cpu")

    with rasterio.open(result_a["tiles"][0]["mask"]) as src:
        assert np.all(src.read(1) == 13)
    with rasterio.open(result_b["tiles"][0]["mask"]) as src:
        assert np.all(src.read(1) == 0)
    assert result_a["stream_id"] == "model:model_a"
    assert result_b["stream_id"] == "model:model_b"
    assert (run_dir / "tmp" / "streams" / "model_a").is_dir()
    assert (run_dir / "tmp" / "streams" / "model_b").is_dir()
