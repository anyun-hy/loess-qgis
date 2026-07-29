"""Vectorize one complete semantic result-stream mosaic."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio import features
from rasterio_compat import quiet_deprecated_memory_driver


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
logger = logging.getLogger("polygonize_mosaic")

LAYER_NAME = "semantic_polygons"
SCHEMA = {
    "geometry": "Polygon",
    "properties": {
        "run_id": "str",
        "result_stream_id": "str",
        "result_kind": "str",
        "model_id": "str",
        "fusion_profile_id": "str",
        "object_id": "str",
        "part_id": "str",
        "class_code": "int",
        "class_name": "str",
        "confidence_mean": "float",
        "confidence_std": "float",
        "model_version": "str",
        "source": "str",
        "created_at": "str",
    },
}


def load_class_map(path):
    if not path:
        raise ValueError("class_map JSON is required")
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    class_map = {int(code): str(name) for code, name in config.get("class_mapping", {}).items()}
    index_to_code = {int(index): int(code) for index, code in config.get("index_to_code", {}).items()}
    background_index = int(config.get("background_index", -1))
    if len(class_map) != 14 or len(index_to_code) != 14 or background_index != -1:
        raise ValueError("class_map must contain the fixed 14-class mapping and background_index=-1")
    return class_map, index_to_code, background_index


def _stream_token(stream_id):
    readable = re.sub(r"[^a-zA-Z0-9]+", "_", stream_id).strip("_")[:18]
    digest = hashlib.sha1(stream_id.encode("utf-8")).hexdigest()[:8]
    return f"{readable}_{digest}" if readable else digest


def polygonize_mask(
    mask_path,
    conf_path,
    output_path,
    run_id,
    stream_id,
    result_kind,
    model_version,
    class_map_path,
    *,
    model_id="",
    fusion_profile_id="",
):
    import fiona
    from fiona.crs import CRS

    if result_kind not in ("model", "fusion"):
        raise ValueError("result_kind must be model or fusion")
    if result_kind == "model" and not model_id:
        raise ValueError("model result requires model_id")
    if result_kind == "fusion" and not fusion_profile_id:
        raise ValueError("fusion result requires fusion_profile_id")
    class_map, index_to_code, background_index = load_class_map(class_map_path)
    with rasterio.open(mask_path) as mask_src:
        mask = mask_src.read(1)
        mask_transform = mask_src.transform
        mask_crs = mask_src.crs
    with rasterio.open(conf_path) as conf_src:
        conf = conf_src.read(1)
        if conf.shape != mask.shape or conf_src.transform != mask_transform or conf_src.crs != mask_crs:
            raise ValueError("mask and confidence mosaics do not share geometry")

    if mask_crs:
        output_crs = {"crs_wkt": mask_crs.to_wkt()}
        crs_label = mask_crs.to_string()
    else:
        logger.warning("[polygonize_mosaic] Input mosaic has no CRS; falling back to EPSG:4490")
        output_crs = {"crs": CRS.from_epsg(4490)}
        crs_label = "EPSG:4490 (fallback)"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    created_at = datetime.datetime.now().isoformat(timespec="seconds")
    token = _stream_token(stream_id)
    results = []
    seq = 0
    valid_mask = mask != background_index
    with quiet_deprecated_memory_driver():
        polygons = features.shapes(
            mask.astype(np.int16),
            mask=valid_mask,
            transform=mask_transform,
        )
        for geom, value in polygons:
            class_index = int(value)
            class_code = index_to_code.get(class_index)
            if class_code is None:
                continue
            seq += 1
            polygon_pixels = features.geometry_mask(
                [geom],
                transform=mask_transform,
                invert=True,
                out_shape=mask.shape,
            )
            values = conf[polygon_pixels]
            results.append({
                "geometry": geom,
                "properties": {
                    "run_id": run_id,
                    "result_stream_id": stream_id,
                    "result_kind": result_kind,
                    "model_id": model_id,
                    "fusion_profile_id": fusion_profile_id,
                    "object_id": f"{run_id}_{token}_{seq:06d}",
                    "part_id": "000",
                    "class_code": class_code,
                    "class_name": class_map[class_code],
                    "confidence_mean": float(np.mean(values)) if values.size else 0.0,
                    "confidence_std": float(np.std(values)) if values.size else 0.0,
                    "model_version": model_version,
                    "source": "semantic_model" if result_kind == "model" else "semantic_fusion",
                    "created_at": created_at,
                },
            })

    with fiona.open(
        output_path,
        "w",
        driver="GPKG",
        layer=LAYER_NAME,
        schema=SCHEMA,
        **output_crs,
    ) as dst:
        dst.writerecords(results)
    logger.info(
        "[polygonize_mosaic] "
        + json.dumps({
            "stream_id": stream_id,
            "polygon_count": len(results),
            "output": str(output_path),
            "layer": LAYER_NAME,
            "crs": crs_label,
            "vectorization": "raw",
        }, ensure_ascii=False, separators=(",", ":"))
    )
    return len(results)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Vectorize one semantic result stream")
    parser.add_argument("--mask", required=True)
    parser.add_argument("--confidence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stream-id", required=True)
    parser.add_argument("--result-kind", choices=("model", "fusion"), required=True)
    parser.add_argument("--model-version", default="")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--fusion-profile-id", default="")
    parser.add_argument("--class-map", required=True)
    args = parser.parse_args(argv)
    try:
        polygonize_mask(
            args.mask,
            args.confidence,
            args.output,
            args.run_id,
            args.stream_id,
            args.result_kind,
            args.model_version,
            args.class_map,
            model_id=args.model_id,
            fusion_profile_id=args.fusion_profile_id,
        )
        return 0
    except Exception as exc:
        logger.error(f"[polygonize_mosaic] ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
