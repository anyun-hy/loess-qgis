"""Build categorical mosaics by blending 14-class tile probabilities."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
logger = logging.getLogger("mosaic_builder")

CLASS_COUNT = 14
PROBABILITY_SCALE = 1.0 / 65535.0
DEFAULT_OVERLAP = 192
DEFAULT_STRATEGY = "cosine_probability_blend"
TILE_PATTERN = re.compile(r"tile_(-?\d+)_(-?\d+)_mask\.tif$")


class MosaicError(RuntimeError):
    pass


def parse_tile_path(path):
    match = TILE_PATTERN.match(os.path.basename(path))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _paired_tiles(mask_dir, conf_dir, score_dir):
    result = []
    for mask_path in sorted(glob.glob(os.path.join(mask_dir, "tile_*_mask.tif"))):
        coords = parse_tile_path(mask_path)
        if coords is None:
            continue
        stem = os.path.basename(mask_path).removesuffix("_mask.tif")
        conf_path = os.path.join(conf_dir, f"{stem}_conf.tif")
        score_path = os.path.join(score_dir, f"{stem}_probabilities.npz")
        if not os.path.isfile(conf_path):
            raise MosaicError(f"confidence tile is missing for {mask_path}: {conf_path}")
        if not os.path.isfile(score_path):
            raise MosaicError(f"probability cache is missing for {mask_path}: {score_path}")
        result.append((coords[0], coords[1], Path(mask_path), Path(conf_path), Path(score_path)))
    if not result:
        raise MosaicError(f"no tile mask files found in {mask_dir}")
    return result


def _cosine_axis_weights(length: int, overlap: int) -> np.ndarray:
    if length < 1:
        raise MosaicError("tile dimensions must be positive")
    if overlap <= 0 or overlap >= length:
        raise MosaicError(f"overlap must be between 1 and {length - 1}, got {overlap}")
    phase = (np.arange(overlap, dtype=np.float32) + 0.5) / float(overlap)
    ramp = np.sin(0.5 * np.pi * phase) ** 2
    weights = np.ones(length, dtype=np.float32)
    weights[:overlap] = np.minimum(weights[:overlap], ramp)
    weights[-overlap:] = np.minimum(weights[-overlap:], ramp[::-1])
    return weights


def _tile_weights(height: int, width: int, overlap: int) -> np.ndarray:
    vertical = _cosine_axis_weights(height, overlap)
    horizontal = _cosine_axis_weights(width, overlap)
    return vertical[:, None] * horizontal[None, :]


def _validate_probabilities(path: Path, height: int, width: int) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as cached:
            if "probabilities" not in cached:
                raise MosaicError(f"probability cache has no 'probabilities' array: {path}")
            probabilities = cached["probabilities"]
    except (OSError, ValueError) as exc:
        raise MosaicError(f"cannot read probability cache {path}: {exc}") from exc
    expected_shape = (CLASS_COUNT, height, width)
    if probabilities.shape != expected_shape:
        raise MosaicError(
            f"probability cache shape must be {expected_shape}, got {probabilities.shape}: {path}"
        )
    if probabilities.dtype != np.float16:
        raise MosaicError(f"probability cache dtype must be float16, got {probabilities.dtype}: {path}")
    probabilities = probabilities.astype(np.float32)
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise MosaicError(f"probability cache contains invalid values: {path}")
    sums = probabilities.sum(axis=0, dtype=np.float32)
    if not np.allclose(sums, 1.0, atol=5e-3, rtol=0):
        raise MosaicError(f"probabilities do not sum to 1: {path}")
    return probabilities


def _collect_metadata(mask_dir, conf_dir, score_dir):
    tiles = _paired_tiles(mask_dir, conf_dir, score_dir)
    metadata = []
    reference_crs = None
    resolution = None
    for row, col, mask_path, conf_path, score_path in tiles:
        with rasterio.open(mask_path) as mask_src, rasterio.open(conf_path) as conf_src:
            if mask_src.count != 1 or conf_src.count != 1:
                raise MosaicError(f"mask/confidence must be single-band: {mask_path}")
            if (mask_src.width, mask_src.height) != (conf_src.width, conf_src.height):
                raise MosaicError(f"mask/confidence dimensions differ: {mask_path}")
            if mask_src.transform != conf_src.transform or mask_src.crs != conf_src.crs:
                raise MosaicError(f"mask/confidence georeferencing differs: {mask_path}")
            transform = mask_src.transform
            if not math.isclose(transform.b, 0.0, abs_tol=1e-12) or not math.isclose(
                transform.d, 0.0, abs_tol=1e-12
            ):
                raise MosaicError("rotated tile transforms are not supported")
            current_resolution = (float(transform.a), abs(float(transform.e)))
            if current_resolution[0] <= 0 or current_resolution[1] <= 0:
                raise MosaicError(f"invalid pixel resolution: {transform}")
            if resolution is None:
                resolution = current_resolution
                reference_crs = mask_src.crs
            elif not (
                math.isclose(current_resolution[0], resolution[0], rel_tol=0, abs_tol=1e-12)
                and math.isclose(current_resolution[1], resolution[1], rel_tol=0, abs_tol=1e-12)
            ):
                raise MosaicError("tile pixel resolutions differ")
            if mask_src.crs != reference_crs:
                raise MosaicError("tile CRS values differ")
            metadata.append({
                "row": row,
                "col": col,
                "mask": mask_path,
                "confidence": conf_path,
                "scores": score_path,
                "width": mask_src.width,
                "height": mask_src.height,
                "bounds": mask_src.bounds,
            })
    return metadata, reference_crs, resolution


def build_mosaic(
    mask_dir,
    conf_dir,
    score_dir,
    output_mask,
    output_conf,
    output_probabilities,
    overlap=DEFAULT_OVERLAP,
    strategy=DEFAULT_STRATEGY,
):
    """Blend tile probabilities spatially, then derive one mask/confidence pair."""
    if strategy != DEFAULT_STRATEGY:
        raise MosaicError(f"unsupported mosaic strategy: {strategy}")
    overlap = int(overlap)
    metadata, reference_crs, resolution = _collect_metadata(mask_dir, conf_dir, score_dir)
    if any(overlap >= min(item["width"], item["height"]) for item in metadata):
        raise MosaicError("overlap must be smaller than every tile dimension")

    min_x = min(item["bounds"].left for item in metadata)
    max_x = max(item["bounds"].right for item in metadata)
    min_y = min(item["bounds"].bottom for item in metadata)
    max_y = max(item["bounds"].top for item in metadata)
    res_x, res_y = resolution
    mosaic_width = int(round((max_x - min_x) / res_x))
    mosaic_height = int(round((max_y - min_y) / res_y))
    if mosaic_width < 1 or mosaic_height < 1:
        raise MosaicError("computed mosaic dimensions are invalid")
    mosaic_transform = from_origin(min_x, max_y, res_x, res_y)

    output_mask = Path(output_mask)
    output_conf = Path(output_conf)
    output_probabilities = Path(output_probabilities)
    output_mask.parent.mkdir(parents=True, exist_ok=True)
    output_conf.parent.mkdir(parents=True, exist_ok=True)
    output_probabilities.parent.mkdir(parents=True, exist_ok=True)
    mask_profile = {
        "driver": "GTiff",
        "count": 1,
        "dtype": "int16",
        "nodata": -1,
        "compress": "lzw",
        "width": mosaic_width,
        "height": mosaic_height,
        "crs": reference_crs,
        "transform": mosaic_transform,
    }
    conf_profile = dict(mask_profile)
    conf_profile.update(dtype="float32", nodata=None)
    probability_profile = dict(mask_profile)
    probability_profile.update(
        count=CLASS_COUNT,
        dtype="uint16",
        nodata=None,
        compress="deflate",
        predictor=2,
        BIGTIFF="IF_SAFER",
    )

    weight_cache = {}
    with tempfile.TemporaryDirectory(prefix="probability_mosaic_", dir=output_mask.parent) as temp_dir:
        temp_root = Path(temp_dir)
        probability_sum = np.memmap(
            temp_root / "probability_sum.dat",
            mode="w+",
            dtype=np.float32,
            shape=(CLASS_COUNT, mosaic_height, mosaic_width),
        )
        weight_sum = np.memmap(
            temp_root / "weight_sum.dat",
            mode="w+",
            dtype=np.float32,
            shape=(mosaic_height, mosaic_width),
        )
        probability_sum[:] = 0.0
        weight_sum[:] = 0.0

        for item in metadata:
            col_start = int(round((item["bounds"].left - min_x) / res_x))
            row_start = int(round((max_y - item["bounds"].top) / res_y))
            row_end = row_start + item["height"]
            col_end = col_start + item["width"]
            if row_start < 0 or col_start < 0 or row_end > mosaic_height or col_end > mosaic_width:
                raise MosaicError(f"tile placement falls outside mosaic: {item['mask']}")
            probabilities = _validate_probabilities(item["scores"], item["height"], item["width"])
            weight_key = (item["height"], item["width"], overlap)
            weights = weight_cache.get(weight_key)
            if weights is None:
                weights = _tile_weights(item["height"], item["width"], overlap)
                weight_cache[weight_key] = weights
            target = np.s_[row_start:row_end, col_start:col_end]
            probability_sum[:, target[0], target[1]] += probabilities * weights[None, :, :]
            weight_sum[target] += weights

        probability_sum.flush()
        weight_sum.flush()
        uncovered_pixels = int(np.count_nonzero(weight_sum <= 0))
        valid_pixels = int(weight_sum.size - uncovered_pixels)
        if valid_pixels == 0:
            raise MosaicError("probability mosaic has no covered pixels")

        with rasterio.open(output_mask, "w", **mask_profile) as mask_dst, rasterio.open(
            output_conf, "w", **conf_profile
        ) as conf_dst, rasterio.open(
            output_probabilities, "w", **probability_profile
        ) as probability_dst:
            probability_dst.scales = tuple(PROBABILITY_SCALE for _ in range(CLASS_COUNT))
            probability_dst.update_tags(
                probability_encoding="uint16_scale_1_over_65535",
                probability_scale=f"{PROBABILITY_SCALE:.17g}",
                class_count=str(CLASS_COUNT),
            )
            for row_start in range(0, mosaic_height, 256):
                row_end = min(row_start + 256, mosaic_height)
                weights = np.asarray(weight_sum[row_start:row_end], dtype=np.float32)
                valid = weights > 0
                scores = np.asarray(probability_sum[:, row_start:row_end], dtype=np.float32)
                scores /= np.where(valid, weights, 1.0)[None, :, :]
                class_sums = scores.sum(axis=0, dtype=np.float32)
                scores /= np.where(class_sums > 0, class_sums, 1.0)[None, :, :]
                mask = scores.argmax(axis=0).astype(np.int16)
                confidence = scores.max(axis=0).astype(np.float32)
                mask[~valid] = -1
                confidence[~valid] = 0.0
                window = Window(0, row_start, mosaic_width, row_end - row_start)
                mask_dst.write(mask, 1, window=window)
                conf_dst.write(confidence, 1, window=window)
                quantized = np.rint(
                    np.clip(scores, 0.0, 1.0) / PROBABILITY_SCALE
                ).astype(np.uint16)
                quantized[:, ~valid] = 0
                probability_dst.write(quantized, window=window)

        del probability_sum
        del weight_sum

    result = {
        "strategy": strategy,
        "tile_count": len(metadata),
        "width": mosaic_width,
        "height": mosaic_height,
        "crs": reference_crs.to_string() if reference_crs else None,
        "class_count": CLASS_COUNT,
        "overlap": overlap,
        "valid_pixels": valid_pixels,
        "uncovered_pixels": uncovered_pixels,
        "score_dir": str(Path(score_dir)),
        "output_mask": str(output_mask),
        "output_confidence": str(output_conf),
        "output_probabilities": str(output_probabilities),
        "probability_encoding": "uint16_scale_1_over_65535",
        "probability_scale": PROBABILITY_SCALE,
    }
    logger.info("[mosaic_builder] " + json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Blend tile probabilities into a mask/confidence mosaic")
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--conf_dir", required=True)
    parser.add_argument("--score_dir", required=True)
    parser.add_argument("--output_mask", required=True)
    parser.add_argument("--output_conf", required=True)
    parser.add_argument("--output_probabilities", required=True)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY, choices=[DEFAULT_STRATEGY])
    args = parser.parse_args(argv)
    try:
        build_mosaic(
            args.mask_dir,
            args.conf_dir,
            args.score_dir,
            args.output_mask,
            args.output_conf,
            args.output_probabilities,
            args.overlap,
            args.strategy,
        )
        return 0
    except Exception as exc:
        logger.error(f"[mosaic_builder] ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
