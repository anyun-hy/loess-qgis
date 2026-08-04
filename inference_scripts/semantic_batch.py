"""Run one registered TorchScript model over every tile in a run specification."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import rasterio
import torch

from deployment_config import CLASS_ORDER, load_json, sha256_file
from _device import resolve_device, validate_device
from torchscript_runtime import load_torchscript_model


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


class BatchInferenceError(RuntimeError):
    pass


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False, separators=(",", ":")), flush=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temp_name, **arrays)
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_raster(path: Path, array: np.ndarray, profile: Mapping[str, Any], dtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tif", dir=path.parent)
    os.close(fd)
    try:
        output_profile = dict(profile)
        output_profile.update(
            driver="GTiff",
            count=1,
            dtype=dtype,
            compress="lzw",
            nodata=None,
        )
        with rasterio.open(temp_name, "w", **output_profile) as dst:
            dst.write(array.astype(dtype, copy=False), 1)
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _validate_run_spec(spec: Mapping[str, Any]) -> None:
    if spec.get("schema_version") != 1:
        raise BatchInferenceError("run_spec.schema_version must equal 1")
    if not isinstance(spec.get("run_id"), str) or not spec["run_id"]:
        raise BatchInferenceError("run_spec.run_id is required")
    if not isinstance(spec.get("run_dir"), str) or not spec["run_dir"]:
        raise BatchInferenceError("run_spec.run_dir is required")
    tiles = spec.get("tiles")
    if not isinstance(tiles, list) or not tiles:
        raise BatchInferenceError("run_spec.tiles must contain at least one tile")
    models = spec.get("models")
    if not isinstance(models, list) or not models:
        raise BatchInferenceError("run_spec.models must contain at least one model")


def _model_entry(spec: Mapping[str, Any], model_id: str) -> Mapping[str, Any]:
    matches = [item for item in spec["models"] if item.get("model_id") == model_id]
    if len(matches) != 1:
        raise BatchInferenceError(f"run_spec must contain exactly one model entry for {model_id}")
    return matches[0]


def _tile_paths(run_dir: Path, model_id: str, tile_id: str) -> dict[str, Path]:
    root = run_dir / "tmp" / "streams" / model_id
    safe_id = tile_id.replace("/", "_").replace("\\", "_")
    return {
        "mask": root / "masks" / f"tile_{safe_id}_mask.tif",
        "confidence": root / "confidence" / f"tile_{safe_id}_conf.tif",
        "scores": root / "scores" / f"tile_{safe_id}_probabilities.npz",
        "metadata": root / "metadata" / f"tile_{safe_id}.json",
    }


def _read_tile(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as src:
        if src.width != 512 or src.height != 512:
            raise BatchInferenceError(f"tile must be 512x512, got {src.width}x{src.height}")
        if src.count < 3:
            raise BatchInferenceError(f"tile must have at least 3 bands, got {src.count}")
        image = src.read((1, 2, 3))
        profile = src.profile.copy()
    if not np.issubdtype(image.dtype, np.integer):
        raise BatchInferenceError(f"tile dtype must be integer imagery, got {image.dtype}")
    if image.size and int(image.max()) > 255:
        raise BatchInferenceError("tile values exceed 8-bit range; deployment preprocessing expects 0..255")
    return image, profile


def _run_model_batch(
    model,
    images: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch = np.asarray(images)
    if batch.ndim != 4 or tuple(batch.shape[1:]) != (3, 512, 512):
        raise BatchInferenceError(
            "model batch input must be [B,3,512,512], got " + str(tuple(batch.shape))
        )
    if batch.shape[0] < 1:
        raise BatchInferenceError("model batch input must contain at least one Tile")
    tensor = torch.from_numpy(batch.astype(np.float32, copy=False)).to(device)
    tensor = tensor / 255.0
    tensor = (tensor - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    with torch.inference_mode():
        logits = model(tensor)
    if not torch.is_tensor(logits):
        raise BatchInferenceError(f"TorchScript output must be one tensor, got {type(logits).__name__}")
    expected_shape = (int(batch.shape[0]), 14, 512, 512)
    if tuple(logits.shape) != expected_shape:
        raise BatchInferenceError(
            f"TorchScript output shape must be {expected_shape}, got {tuple(logits.shape)}"
        )
    logits = logits.float()
    probs = torch.softmax(logits, dim=1)
    confidence, mask = probs.max(dim=1)
    return (
        mask.to(torch.uint8).cpu().numpy(),
        confidence.cpu().numpy().astype(np.float32),
        probs.cpu().numpy().astype(np.float16),
    )


def _run_model(model, image: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks, confidence, probabilities = _run_model_batch(
        model,
        np.expand_dims(image, axis=0),
        device,
    )
    return masks[0], confidence[0], probabilities[0]


def _completed(paths: Mapping[str, Path], expected: Mapping[str, Any]) -> bool:
    if not all(paths[key].is_file() for key in ("mask", "confidence", "scores", "metadata")):
        return False
    try:
        metadata = load_json(paths["metadata"])
        if any(metadata.get(key) != value for key, value in expected.items()):
            return False
        with rasterio.open(paths["mask"]) as src:
            if (src.count, src.width, src.height, src.dtypes[0]) != (1, 512, 512, "uint8"):
                return False
        with rasterio.open(paths["confidence"]) as src:
            if (src.count, src.width, src.height, src.dtypes[0]) != (1, 512, 512, "float32"):
                return False
        with np.load(paths["scores"]) as cached:
            scores = cached["probabilities"]
            if scores.shape != (14, 512, 512) or scores.dtype != np.float16:
                return False
        return True
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def run_batch(
    run_spec_path: os.PathLike[str] | str,
    model_id: str,
    *,
    device: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    spec_path = Path(run_spec_path).resolve()
    spec = load_json(spec_path)
    _validate_run_spec(spec)
    model_entry = _model_entry(spec, model_id)
    run_dir = Path(spec["run_dir"]).resolve()
    artifact_path = Path(model_entry["artifact_path"]).resolve()
    expected_sha = str(model_entry.get("sha256") or "")
    if not artifact_path.is_file():
        raise BatchInferenceError(f"model artifact not found: {artifact_path}")
    actual_sha = sha256_file(artifact_path)
    if actual_sha != expected_sha:
        raise BatchInferenceError(f"model artifact SHA256 mismatch: expected {expected_sha}, got {actual_sha}")

    requested_device = device or (spec.get("runtime") or {}).get("effective_device") or "auto"
    effective_device = resolve_device(requested_device)
    if not validate_device(effective_device):
        raise BatchInferenceError(f"device is unavailable: {effective_device}")

    emit(
        "model_loading",
        run_id=spec["run_id"],
        stream_id=f"model:{model_id}",
        model_id=model_id,
        device=effective_device,
    )
    model, runtime_info = load_torchscript_model(artifact_path, effective_device)
    emit(
        "model_loaded",
        run_id=spec["run_id"],
        stream_id=f"model:{model_id}",
        **runtime_info,
    )

    tiles = spec["tiles"]
    started = time.time()
    completed = 0
    skipped = 0
    failures: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for index, tile in enumerate(tiles, start=1):
        tile_id = str(tile.get("tile_id") or f"{tile.get('row', 0)}_{tile.get('col', 0)}")
        tile_path = Path(tile["path"]).resolve()
        paths = _tile_paths(run_dir, model_id, tile_id)
        input_sha = str(tile.get("sha256") or "")
        if not input_sha:
            input_sha = sha256_file(tile_path)
        expected_metadata = {
            "schema_version": 1,
            "run_id": spec["run_id"],
            "tile_id": tile_id,
            "model_id": model_id,
            "model_sha256": actual_sha,
            "input_sha256": input_sha,
            "class_order": CLASS_ORDER,
            "device": effective_device,
            "runtime_mode": runtime_info["mode"],
        }
        if resume and _completed(paths, expected_metadata):
            skipped += 1
            completed += 1
            entries.append({
                "tile_id": tile_id,
                "status": "reused",
                **{key: str(value) for key, value in paths.items()},
            })
            emit(
                "tile_reused",
                run_id=spec["run_id"],
                stream_id=f"model:{model_id}",
                tile_id=tile_id,
                current=index,
                total=len(tiles),
            )
            continue
        if not resume and any(path.exists() for path in paths.values()):
            raise BatchInferenceError(f"stale output exists for tile {tile_id}; use --resume or a new run_id")

        emit(
            "tile_started",
            run_id=spec["run_id"],
            stream_id=f"model:{model_id}",
            tile_id=tile_id,
            current=index,
            total=len(tiles),
        )
        try:
            image, profile = _read_tile(tile_path)
            mask, confidence, probabilities = _run_model(model, image, effective_device)
            _atomic_raster(paths["mask"], mask, profile, "uint8")
            _atomic_raster(paths["confidence"], confidence, profile, "float32")
            _atomic_npz(paths["scores"], probabilities=probabilities)
            _atomic_json(paths["metadata"], expected_metadata)
            completed += 1
            entry = {
                "tile_id": tile_id,
                "status": "completed",
                **{key: str(value) for key, value in paths.items()},
            }
            entries.append(entry)
            emit(
                "tile_completed",
                run_id=spec["run_id"],
                stream_id=f"model:{model_id}",
                tile_id=tile_id,
                current=index,
                total=len(tiles),
            )
        except Exception as exc:
            failure = {
                "tile_id": tile_id,
                "row": tile.get("row"),
                "col": tile.get("col"),
                "error": str(exc),
            }
            failures.append(failure)
            entries.append({"tile_id": tile_id, "status": "failed", "error": str(exc)})
            emit(
                "tile_failed",
                run_id=spec["run_id"],
                stream_id=f"model:{model_id}",
                tile_id=tile_id,
                current=index,
                total=len(tiles),
                error=str(exc),
            )

    manifest = {
        "schema_version": 1,
        "run_id": spec["run_id"],
        "stream_id": f"model:{model_id}",
        "model_id": model_id,
        "model_version": str(model_entry.get("version") or ""),
        "artifact": str(artifact_path),
        "artifact_sha256": actual_sha,
        "device": effective_device,
        "runtime": runtime_info,
        "model_load_count": 1,
        "tile_count": len(tiles),
        "completed_count": completed,
        "reused_count": skipped,
        "failed_count": len(failures),
        "elapsed_sec": round(time.time() - started, 3),
        "tiles": entries,
        "failures": failures,
    }
    stream_root = run_dir / "tmp" / "streams" / model_id
    _atomic_json(stream_root / "stream_manifest.json", manifest)
    _atomic_json(stream_root / "failed_jobs.json", {"failures": failures})
    emit(
        "model_finished",
        run_id=spec["run_id"],
        stream_id=f"model:{model_id}",
        completed=completed,
        failed=len(failures),
        reused=skipped,
    )
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Batch TorchScript inference for one registered model")
    parser.add_argument("--run-spec", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        result = run_batch(args.run_spec, args.model_id, device=args.device, resume=args.resume)
    except Exception as exc:
        emit("batch_failed", model_id=args.model_id, error=str(exc))
        return 2
    return 0 if result["failed_count"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
