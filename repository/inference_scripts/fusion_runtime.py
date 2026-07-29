"""Fuse per-model probability caches according to an approved fusion profile."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import rasterio
import torch

from deployment_config import (
    CLASS_ORDER,
    load_json,
    sha256_file,
    validate_fusion_profile,
)
from semantic_batch import _atomic_json, _atomic_npz, _atomic_raster, emit
from torchscript_runtime import load_torchscript_model


class FusionRuntimeError(RuntimeError):
    pass


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=0, keepdims=True)
    exp = np.exp(shifted, dtype=np.float32)
    return exp / exp.sum(axis=0, keepdims=True)


def fuse_probability_arrays(
    probabilities: Mapping[str, np.ndarray],
    profile: Mapping[str, Any],
    *,
    fusion_head=None,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return uint8 mask, float32 confidence, and float32 fused scores."""
    model_entries = profile.get("models") or []
    model_ids = [str(item["model_id"]) for item in model_entries]
    if not model_ids:
        raise FusionRuntimeError("fusion profile does not contain models")
    missing = [model_id for model_id in model_ids if model_id not in probabilities]
    if missing:
        raise FusionRuntimeError(f"missing model probabilities: {missing}")

    arrays = [np.asarray(probabilities[model_id], dtype=np.float32) for model_id in model_ids]
    expected_shape = arrays[0].shape
    if len(expected_shape) != 3 or expected_shape[0] != len(CLASS_ORDER):
        raise FusionRuntimeError(f"probability shape must be [14,H,W], got {expected_shape}")
    if any(array.shape != expected_shape for array in arrays):
        raise FusionRuntimeError("model probability arrays have different shapes")
    for model_id, array in zip(model_ids, arrays):
        if not np.all(np.isfinite(array)) or np.any(array < 0):
            raise FusionRuntimeError(f"invalid probabilities for {model_id}")
        sums = array.sum(axis=0)
        if not np.allclose(sums, 1.0, atol=5e-3, rtol=0):
            raise FusionRuntimeError(f"probabilities do not sum to 1 for {model_id}")

    strategy = profile.get("strategy")
    stacked = np.stack(arrays, axis=0)  # [M,C,H,W]
    if strategy == "equal_probability_average":
        fused_probabilities = stacked.mean(axis=0, dtype=np.float32)
        mask = fused_probabilities.argmax(axis=0).astype(np.uint8)
        confidence = fused_probabilities.max(axis=0).astype(np.float32)
        return mask, confidence, fused_probabilities

    temperatures = np.asarray([item.get("temperature") for item in model_entries], dtype=np.float32)
    if temperatures.shape != (len(model_ids),) or np.any(~np.isfinite(temperatures)) or np.any(temperatures <= 0):
        raise FusionRuntimeError("profile temperatures must be finite positive values")
    calibrated = np.log(np.clip(stacked, 1e-7, 1.0))
    calibrated /= temperatures[:, None, None, None]
    weights = np.asarray(profile.get("weights"), dtype=np.float32)
    if weights.shape != (len(CLASS_ORDER), len(model_ids)):
        raise FusionRuntimeError(f"profile weights must have shape (14,{len(model_ids)})")

    if strategy in ("calibrated_global_weighted", "calibrated_class_weighted"):
        fused_scores = np.einsum("mchw,cm->chw", calibrated, weights, optimize=True)
    elif strategy == "linear_1x1":
        if fusion_head is None:
            raise FusionRuntimeError("linear_1x1 strategy requires a TorchScript fusion head")
        gated = calibrated * weights.T[:, :, None, None]
        features = gated.reshape(1, len(model_ids) * len(CLASS_ORDER), expected_shape[1], expected_shape[2])
        tensor = torch.from_numpy(features).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            output = fusion_head(tensor)
        if not torch.is_tensor(output) or tuple(output.shape) != (1, 14, expected_shape[1], expected_shape[2]):
            shape = tuple(output.shape) if torch.is_tensor(output) else type(output).__name__
            raise FusionRuntimeError(f"fusion head output contract failed: {shape}")
        fused_scores = output[0].float().cpu().numpy()
    else:
        raise FusionRuntimeError(f"unsupported fusion strategy: {strategy}")

    fused_probabilities = _softmax(fused_scores.astype(np.float32, copy=False))
    mask = fused_probabilities.argmax(axis=0).astype(np.uint8)
    confidence = fused_probabilities.max(axis=0).astype(np.float32)
    return mask, confidence, fused_scores.astype(np.float32, copy=False)


def _registry_from_spec(spec: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    registry = {}
    for model in spec.get("models") or []:
        artifact_path = Path(model.get("artifact_path") or "")
        registry[str(model.get("model_id"))] = {
            "model_id": str(model.get("model_id")),
            "artifact": str(model.get("artifact") or artifact_path.name),
            "sha256": str(model.get("sha256") or ""),
        }
    return registry


def _load_stream_manifests(run_dir: Path, profile: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    manifests = {}
    for model in profile["models"]:
        model_id = model["model_id"]
        path = run_dir / "tmp" / "streams" / model_id / "stream_manifest.json"
        if not path.is_file():
            raise FusionRuntimeError(f"model stream manifest is missing: {path}")
        manifest = load_json(path)
        if manifest.get("artifact_sha256") != model.get("sha256"):
            raise FusionRuntimeError(f"model stream SHA does not match profile: {model_id}")
        if manifest.get("failed_count") != 0:
            raise FusionRuntimeError(f"model stream contains failed tiles: {model_id}")
        manifests[model_id] = manifest
    return manifests


def _tile_entries(manifests: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    result: dict[str, dict[str, Mapping[str, Any]]] = {}
    expected_ids = None
    for model_id, manifest in manifests.items():
        entries = {
            str(item["tile_id"]): item
            for item in manifest.get("tiles") or []
            if item.get("status") in ("completed", "reused")
        }
        ids = set(entries)
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise FusionRuntimeError("model streams do not contain the same tile IDs")
        for tile_id, entry in entries.items():
            result.setdefault(tile_id, {})[model_id] = entry
    if not result:
        raise FusionRuntimeError("model streams contain no completed tiles")
    return result


def _fusion_paths(run_dir: Path, profile_id: str, tile_id: str) -> dict[str, Path]:
    safe_profile = profile_id.replace("/", "_")
    safe_tile = tile_id.replace("/", "_")
    root = run_dir / "tmp" / "streams" / f"fusion_{safe_profile}"
    return {
        "mask": root / "masks" / f"tile_{safe_tile}_mask.tif",
        "confidence": root / "confidence" / f"tile_{safe_tile}_conf.tif",
        "scores": root / "scores" / f"tile_{safe_tile}_probabilities.npz",
        "metadata": root / "metadata" / f"tile_{safe_tile}.json",
    }


def _completed(paths: Mapping[str, Path], expected: Mapping[str, Any]) -> bool:
    if not all(path.is_file() for path in paths.values()):
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
        with np.load(paths["scores"], allow_pickle=False) as cached:
            probabilities = cached["probabilities"]
            if probabilities.shape != (14, 512, 512) or probabilities.dtype != np.float16:
                return False
        return True
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def run_fusion(
    run_spec_path: os.PathLike[str] | str,
    profile_path: os.PathLike[str] | str,
    *,
    device: str = "cpu",
    resume: bool = False,
) -> dict[str, Any]:
    spec = load_json(Path(run_spec_path).resolve())
    if spec.get("schema_version") != 1:
        raise FusionRuntimeError("run_spec.schema_version must equal 1")
    run_dir = Path(spec["run_dir"]).resolve()
    profile_file = Path(profile_path).resolve()
    profile = load_json(profile_file)
    issues = validate_fusion_profile(profile, registry_by_id=_registry_from_spec(spec))
    if issues:
        raise FusionRuntimeError("invalid fusion profile: " + "; ".join(f"{item.path} {item.message}" for item in issues))
    if profile.get("status") != "approved" or not (profile.get("approval") or {}).get("passed"):
        raise FusionRuntimeError("only approved fusion profiles may run")
    profile_id = profile["profile_id"]
    profile_sha = sha256_file(profile_file)
    manifests = _load_stream_manifests(run_dir, profile)
    by_tile = _tile_entries(manifests)

    fusion_head = None
    if profile["strategy"] == "linear_1x1":
        head_info = profile["fusion_head"]
        head_path = profile_file.parent / head_info["artifact"]
        if not head_path.is_file():
            raise FusionRuntimeError(f"fusion head artifact is missing: {head_path}")
        actual_head_sha = sha256_file(head_path)
        if actual_head_sha != head_info["sha256"]:
            raise FusionRuntimeError("fusion head SHA256 mismatch")
        fusion_head, _runtime_info = load_torchscript_model(head_path, device)

    stream_id = f"fusion:{profile_id}"
    emit("fusion_started", run_id=spec["run_id"], stream_id=stream_id, strategy=profile["strategy"])
    started = time.time()
    completed_count = 0
    reused_count = 0
    failures = []
    entries = []
    source_shas = {model_id: manifest["artifact_sha256"] for model_id, manifest in manifests.items()}
    for index, (tile_id, model_entries) in enumerate(sorted(by_tile.items()), start=1):
        paths = _fusion_paths(run_dir, profile_id, tile_id)
        expected_metadata = {
            "schema_version": 1,
            "run_id": spec["run_id"],
            "tile_id": tile_id,
            "profile_id": profile_id,
            "profile_sha256": profile_sha,
            "source_model_sha256": source_shas,
            "class_order": CLASS_ORDER,
        }
        if resume and _completed(paths, expected_metadata):
            reused_count += 1
            completed_count += 1
            entries.append({"tile_id": tile_id, "status": "reused", **{key: str(value) for key, value in paths.items()}})
            emit("fusion_tile_reused", run_id=spec["run_id"], stream_id=stream_id, tile_id=tile_id, current=index, total=len(by_tile))
            continue
        if not resume and any(path.exists() for path in paths.values()):
            raise FusionRuntimeError(f"stale fusion output exists for tile {tile_id}; use --resume")
        try:
            arrays = {}
            reference_profile = None
            for model in profile["models"]:
                model_id = model["model_id"]
                entry = model_entries[model_id]
                with np.load(entry["scores"], allow_pickle=False) as cached:
                    arrays[model_id] = cached["probabilities"].astype(np.float32)
                if reference_profile is None:
                    with rasterio.open(entry["mask"]) as src:
                        reference_profile = src.profile.copy()
            mask, confidence, fused_values = fuse_probability_arrays(
                arrays,
                profile,
                fusion_head=fusion_head,
                device=device,
            )
            if profile["strategy"] == "equal_probability_average":
                fused_probabilities = fused_values
            else:
                fused_probabilities = _softmax(fused_values)
            _atomic_raster(paths["mask"], mask, reference_profile, "uint8")
            _atomic_raster(paths["confidence"], confidence, reference_profile, "float32")
            _atomic_npz(paths["scores"], probabilities=fused_probabilities.astype(np.float16))
            _atomic_json(paths["metadata"], expected_metadata)
            completed_count += 1
            entries.append({"tile_id": tile_id, "status": "completed", **{key: str(value) for key, value in paths.items()}})
            emit("fusion_tile_completed", run_id=spec["run_id"], stream_id=stream_id, tile_id=tile_id, current=index, total=len(by_tile))
        except Exception as exc:
            failures.append({"tile_id": tile_id, "error": str(exc)})
            entries.append({"tile_id": tile_id, "status": "failed", "error": str(exc)})
            emit("fusion_tile_failed", run_id=spec["run_id"], stream_id=stream_id, tile_id=tile_id, current=index, total=len(by_tile), error=str(exc))

    manifest = {
        "schema_version": 1,
        "run_id": spec["run_id"],
        "stream_id": stream_id,
        "profile_id": profile_id,
        "profile_path": str(profile_file),
        "profile_sha256": profile_sha,
        "strategy": profile["strategy"],
        "model_ids": [item["model_id"] for item in profile["models"]],
        "tile_count": len(by_tile),
        "completed_count": completed_count,
        "reused_count": reused_count,
        "failed_count": len(failures),
        "elapsed_sec": round(time.time() - started, 3),
        "tiles": entries,
        "failures": failures,
    }
    root = run_dir / "tmp" / "streams" / f"fusion_{profile_id}"
    _atomic_json(root / "stream_manifest.json", manifest)
    _atomic_json(root / "failed_jobs.json", {"failures": failures})
    emit("fusion_finished", run_id=spec["run_id"], stream_id=stream_id, completed=completed_count, failed=len(failures), reused=reused_count)
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Fuse registered semantic model tile probabilities")
    parser.add_argument("--run-spec", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        result = run_fusion(args.run_spec, args.profile, device=args.device, resume=args.resume)
    except Exception as exc:
        emit("fusion_failed", error=str(exc))
        return 2
    return 0 if result["failed_count"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
