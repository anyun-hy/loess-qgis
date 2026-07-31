"""Partition-local 14-class probability blending and Core raster output."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import rasterio
from affine import Affine

from mosaic_builder import CLASS_COUNT, PROBABILITY_SCALE, _tile_weights


class PartitionMosaicError(RuntimeError):
    pass


def _window_tuple(value: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return tuple(int(value[key]) for key in ("x0", "y0", "x1", "y1"))


def _load_probabilities(record: Mapping[str, Any]) -> np.ndarray:
    if "probabilities" in record:
        probabilities = np.asarray(record["probabilities"], dtype=np.float32)
    else:
        path = Path(str(record["score_path"])).resolve()
        with np.load(path, allow_pickle=False) as cached:
            probabilities = cached["probabilities"].astype(np.float32)
    if probabilities.ndim != 3 or probabilities.shape[0] != CLASS_COUNT:
        raise PartitionMosaicError(
            f"probability shape must be [14,H,W], got {probabilities.shape}"
        )
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise PartitionMosaicError("probabilities contain invalid values")
    sums = probabilities.sum(axis=0, dtype=np.float32)
    if not np.allclose(sums, 1.0, atol=5e-3, rtol=0):
        raise PartitionMosaicError("probabilities do not sum to one")
    return probabilities


def blend_probability_tiles(
    records: Iterable[Mapping[str, Any]],
    *,
    target_window: Mapping[str, Any],
    overlap: int,
    allow_uncovered: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend only the records intersecting one target pixel window."""
    x0, y0, x1, y1 = _window_tuple(target_window)
    if x1 <= x0 or y1 <= y0:
        raise PartitionMosaicError("target window is empty")
    probability_sum = np.zeros((CLASS_COUNT, y1 - y0, x1 - x0), dtype=np.float32)
    weight_sum = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    used = 0
    weight_cache: dict[tuple[int, int, int], np.ndarray] = {}
    for record in records:
        if "probabilities" in record:
            shape = np.asarray(record["probabilities"]).shape
            if len(shape) != 3:
                raise PartitionMosaicError(
                    f"probability shape must be [14,H,W], got {shape}"
                )
            height, width = int(shape[1]), int(shape[2])
        else:
            height = int(record.get("height", 512))
            width = int(record.get("width", 512))
        stride_x = width - int(overlap)
        stride_y = height - int(overlap)
        if stride_x < 1 or stride_y < 1:
            raise PartitionMosaicError("overlap must be smaller than Tile dimensions")
        tile_x0 = int(record["col"]) * stride_x
        tile_y0 = int(record["row"]) * stride_y
        tile_x1 = tile_x0 + width
        tile_y1 = tile_y0 + height
        ix0 = max(x0, tile_x0)
        iy0 = max(y0, tile_y0)
        ix1 = min(x1, tile_x1)
        iy1 = min(y1, tile_y1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        probabilities = _load_probabilities(record)
        if probabilities.shape[1:] != (height, width):
            raise PartitionMosaicError(
                "probability dimensions do not match the declared Tile size"
            )
        weights = weight_cache.get((height, width, int(overlap)))
        if weights is None:
            weights = _tile_weights(height, width, int(overlap))
            weight_cache[(height, width, int(overlap))] = weights
        source_x0 = ix0 - tile_x0
        source_y0 = iy0 - tile_y0
        source_x1 = ix1 - tile_x0
        source_y1 = iy1 - tile_y0
        target_x0 = ix0 - x0
        target_y0 = iy0 - y0
        target_x1 = ix1 - x0
        target_y1 = iy1 - y0
        source = np.s_[source_y0:source_y1, source_x0:source_x1]
        target = np.s_[target_y0:target_y1, target_x0:target_x1]
        local_weights = weights[source]
        probability_sum[:, target[0], target[1]] += (
            probabilities[:, source[0], source[1]] * local_weights[None, :, :]
        )
        weight_sum[target] += local_weights
        used += 1
    if (used == 0 or np.any(weight_sum <= 0)) and not allow_uncovered:
        uncovered = int(np.count_nonzero(weight_sum <= 0))
        raise PartitionMosaicError(
            f"target window has {uncovered} uncovered pixels from {used} intersecting Tiles"
        )
    covered = weight_sum > 0
    probabilities = np.zeros_like(probability_sum)
    if np.any(covered):
        probabilities[:, covered] = (
            probability_sum[:, covered] / weight_sum[covered][None, :]
        )
        probabilities[:, covered] /= probabilities[:, covered].sum(
            axis=0, keepdims=True
        )
    return probabilities.astype(np.float32, copy=False), weight_sum


def build_partition_arrays(
    records: Iterable[Mapping[str, Any]],
    partition: Mapping[str, Any],
    *,
    overlap: int,
    allow_uncovered: bool = False,
) -> dict[str, np.ndarray]:
    halo = partition["halo_window"]
    probabilities, weights = blend_probability_tiles(
        records,
        target_window=halo,
        overlap=overlap,
        allow_uncovered=allow_uncovered,
    )
    return derive_partition_arrays(probabilities, partition, weights=weights)


def derive_partition_arrays(
    halo_probabilities: np.ndarray,
    partition: Mapping[str, Any],
    *,
    weights: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Derive Core products from an already blended or fused Halo array."""
    probabilities = np.asarray(halo_probabilities, dtype=np.float32)
    if probabilities.ndim != 3 or probabilities.shape[0] != CLASS_COUNT:
        raise PartitionMosaicError("Halo probabilities must have shape [14,H,W]")
    halo = partition["halo_window"]
    core = partition["core_window"]
    hx0, hy0, hx1, hy1 = _window_tuple(halo)
    cx0, cy0, cx1, cy1 = _window_tuple(core)
    if probabilities.shape[1:] != (hy1 - hy0, hx1 - hx0):
        raise PartitionMosaicError("Halo probability shape does not match partition window")
    core_probabilities = probabilities[
        :, cy0 - hy0 : cy1 - hy0, cx0 - hx0 : cx1 - hx0
    ]
    halo_weights = (
        np.asarray(weights, dtype=np.float32)
        if weights is not None
        else probabilities.sum(axis=0, dtype=np.float32)
    )
    core_weights = halo_weights[
        cy0 - hy0 : cy1 - hy0, cx0 - hx0 : cx1 - hx0
    ]
    core_valid = core_weights > 0
    core_mask = np.full(core_valid.shape, -1, dtype=np.int16)
    core_confidence = np.full(core_valid.shape, -1.0, dtype=np.float32)
    if np.any(core_valid):
        core_mask[core_valid] = core_probabilities[:, core_valid].argmax(
            axis=0
        ).astype(np.int16)
        core_confidence[core_valid] = core_probabilities[:, core_valid].max(
            axis=0
        ).astype(np.float32)
    return {
        "halo_probabilities": probabilities,
        "halo_weights": halo_weights,
        "core_probabilities": core_probabilities,
        "core_mask": core_mask,
        "core_confidence": core_confidence,
    }


def _atomic_raster(
    path: Path,
    array: np.ndarray,
    profile: Mapping[str, Any],
    *,
    scales: tuple[float, ...] = (),
    tags: Mapping[str, str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with rasterio.open(temporary, "w", **profile) as destination:
            if array.ndim == 2:
                destination.write(array, 1)
            else:
                destination.write(array)
            if scales:
                destination.scales = scales
            if tags:
                destination.update_tags(**dict(tags))
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_partition_rasters(
    arrays: Mapping[str, np.ndarray],
    partition: Mapping[str, Any],
    *,
    global_transform: Affine,
    crs: Any,
    output_probability: str | Path,
    output_mask: str | Path,
    output_confidence: str | Path,
) -> dict[str, str]:
    halo = partition["halo_window"]
    core = partition["core_window"]
    hx0, hy0, hx1, hy1 = _window_tuple(halo)
    cx0, cy0, cx1, cy1 = _window_tuple(core)
    base = {
        "driver": "GTiff",
        "crs": crs,
        "compress": "deflate",
        "BIGTIFF": "IF_SAFER",
    }
    probability_path = Path(output_probability)
    probability_profile = {
        **base,
        "count": CLASS_COUNT,
        "dtype": "uint16",
        "width": hx1 - hx0,
        "height": hy1 - hy0,
        "transform": global_transform * Affine.translation(hx0, hy0),
        "predictor": 2,
    }
    quantized = np.rint(
        np.clip(arrays["halo_probabilities"], 0.0, 1.0) / PROBABILITY_SCALE
    ).astype(np.uint16)
    _atomic_raster(
        probability_path,
        quantized,
        probability_profile,
        scales=tuple(PROBABILITY_SCALE for _ in range(CLASS_COUNT)),
        tags={
            "probability_encoding": "uint16_scale_1_over_65535",
            "class_count": str(CLASS_COUNT),
        },
    )
    core_profile = {
        **base,
        "count": 1,
        "width": cx1 - cx0,
        "height": cy1 - cy0,
        "transform": global_transform * Affine.translation(cx0, cy0),
    }
    mask_path = Path(output_mask)
    _atomic_raster(
        mask_path,
        arrays["core_mask"],
        {**core_profile, "dtype": "int16", "nodata": -1},
    )
    confidence_path = Path(output_confidence)
    _atomic_raster(
        confidence_path,
        arrays["core_confidence"],
        {**core_profile, "dtype": "float32", "nodata": -1.0},
    )
    return {
        "probability": str(probability_path.resolve()),
        "mask": str(mask_path.resolve()),
        "confidence": str(confidence_path.resolve()),
    }


def build_vrt(output_path: str | Path, part_paths: Iterable[str | Path]) -> str:
    parts = [str(Path(path).resolve()) for path in part_paths]
    if not parts:
        raise PartitionMosaicError("cannot build a VRT without raster parts")
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    executable = shutil.which("gdalbuildvrt")
    if executable is None:
        sibling = Path(sys.executable).resolve().parent / "gdalbuildvrt"
        executable = str(sibling) if sibling.is_file() else None
    if executable is None:
        raise PartitionMosaicError("gdalbuildvrt is not installed in the inference environment")
    command = [executable, "-overwrite", str(output), *parts]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not output.is_file():
        raise PartitionMosaicError(
            "gdalbuildvrt failed: " + (completed.stderr or completed.stdout).strip()
        )
    return str(output)
