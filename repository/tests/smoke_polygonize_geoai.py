"""Standalone polygonize smoke test for the production GeoAI environment."""

import json
import sys
import tempfile
from pathlib import Path

import fiona
import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference_scripts"))

from polygonize_mosaic import polygonize_mask


def write_raster(path, array, dtype):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=array.shape[1],
        height=array.shape[0],
        count=1,
        dtype=dtype,
        crs="EPSG:4490",
        transform=rasterio.transform.from_origin(100, 40, 0.001, 0.001),
    ) as dst:
        dst.write(array.astype(dtype), 1)


def main():
    with tempfile.TemporaryDirectory(prefix="loess_polygonize_") as temp:
        root = Path(temp)
        mask = np.zeros((4, 4), dtype=np.int16)
        mask[:, 2:] = 1
        confidence = np.full((4, 4), 0.8, dtype=np.float32)
        mask_path = root / "mask.tif"
        confidence_path = root / "confidence.tif"
        class_map = root / "classes.json"
        write_raster(mask_path, mask, "int16")
        write_raster(confidence_path, confidence, "float32")
        codes = [12, 13, 21, 31, 32, 33, 43, 51, 52, 53, 54, 61, 62, 71]
        class_map.write_text(json.dumps({
            "background_index": -1,
            "index_to_code": {str(index): code for index, code in enumerate(codes)},
            "class_mapping": {str(code): f"class_{code}" for code in codes},
        }), encoding="utf-8")
        object_ids_by_run = []
        for run_id in ("smoke_run_a", "smoke_run_b"):
            output = root / f"{run_id}.gpkg"
            count = polygonize_mask(
                mask_path,
                confidence_path,
                output,
                run_id,
                "fusion:smoke",
                "fusion",
                "smoke-v1",
                class_map,
                fusion_profile_id="smoke",
            )
            if count != 2:
                raise RuntimeError(f"expected 2 polygons, got {count}")
            with fiona.open(output, layer="semantic_polygons") as source:
                records = list(source)
            if len(records) != 2:
                raise RuntimeError("GPKG readback count mismatch")
            if any(record["properties"]["result_stream_id"] != "fusion:smoke" for record in records):
                raise RuntimeError("result_stream_id was not preserved")
            if any(record["properties"]["source"] != "semantic_fusion" for record in records):
                raise RuntimeError("fusion source field was not preserved")
            object_ids_by_run.append({record["properties"]["object_id"] for record in records})
        if not object_ids_by_run[0].isdisjoint(object_ids_by_run[1]):
            raise RuntimeError("object_id collided across runs")
    print("polygonize geoai smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
