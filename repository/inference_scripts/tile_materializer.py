"""Materialize only the source-image Tiles needed by one Work Package."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import rasterio
from rasterio.windows import from_bounds


class TileMaterializationError(RuntimeError):
    pass


_THREAD_LOCAL = threading.local()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(path: Path):
    dataset = getattr(_THREAD_LOCAL, "dataset", None)
    dataset_path = getattr(_THREAD_LOCAL, "dataset_path", "")
    if dataset is None or dataset.closed or dataset_path != str(path):
        if dataset is not None and not dataset.closed:
            dataset.close()
        dataset = rasterio.open(path)
        _THREAD_LOCAL.dataset = dataset
        _THREAD_LOCAL.dataset_path = str(path)
    return dataset


def _bounds(tile: Mapping[str, Any]) -> dict[str, float]:
    value = tile.get("bounds")
    if not isinstance(value, Mapping):
        raw = tile.get("bounds_json") or "{}"
        value = json.loads(raw) if isinstance(raw, str) else raw
    try:
        return {
            key: float(value[key])
            for key in ("xmin", "ymin", "xmax", "ymax")
        }
    except (KeyError, TypeError, ValueError) as error:
        raise TileMaterializationError(
            f"Tile {tile.get('tile_id')} has invalid bounds"
        ) from error


def _valid_existing(path: Path, expected_sha: str) -> str:
    if not path.is_file() or path.stat().st_size <= 0:
        return ""
    try:
        with rasterio.open(path) as source:
            if source.width != 512 or source.height != 512 or source.count < 3:
                return ""
        digest = _sha256(path)
        if expected_sha and digest != expected_sha:
            return ""
        return digest
    except (OSError, rasterio.errors.RasterioError):
        return ""


def _materialize_one(source_path: Path, output_dir: Path, tile: Mapping[str, Any]) -> dict[str, Any]:
    tile_id = str(tile["tile_id"])
    row = int(tile.get("row_no", tile.get("row", 0)))
    col = int(tile.get("col_no", tile.get("col", 0)))
    configured_path = str(tile.get("raster_path") or "")
    output_path = Path(configured_path).expanduser().resolve() if configured_path else output_dir / f"tile_{row}_{col}.tif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected_sha = str(tile.get("sha256") or "").lower()
    digest = _valid_existing(output_path, expected_sha)
    if digest:
        return {"tile_id": tile_id, "row": row, "col": col, "tile_path": str(output_path), "sha256": digest, "reused": True}
    bounds = _bounds(tile)
    source = _source(source_path)
    if source.count < 3:
        raise TileMaterializationError(f"source image must have at least 3 bands, got {source.count}")
    window = from_bounds(bounds["xmin"], bounds["ymin"], bounds["xmax"], bounds["ymax"], transform=source.transform).round_offsets().round_lengths()
    if int(window.width) != 512 or int(window.height) != 512:
        raise TileMaterializationError(f"Tile {tile_id} source window is {int(window.width)}x{int(window.height)}, expected 512x512")
    image = source.read((1, 2, 3), window=window, boundless=False)
    if image.shape != (3, 512, 512):
        raise TileMaterializationError(f"Tile {tile_id} read shape is {image.shape}, expected (3,512,512)")
    profile = source.profile.copy()
    profile.update(driver="GTiff", width=512, height=512, count=3, transform=source.window_transform(window), tiled=True, blockxsize=256, blockysize=256, compress=None, interleave="pixel", BIGTIFF="IF_SAFER")
    for key in ("photometric", "compress", "predictor"):
        if key in profile and profile[key] is None:
            profile.pop(key)
    temporary = output_path.with_name(output_path.stem + ".tmp.tif")
    try:
        with rasterio.open(temporary, "w", **profile) as destination:
            destination.write(image)
        os.replace(temporary, output_path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    digest = _sha256(output_path)
    metadata = {"row": row, "col": col, "bounds": bounds, "crs": str(source.crs or ""), "geotransform": list(source.window_transform(window).to_gdal()), "width": 512, "height": 512, "tile_path": str(output_path), "sha256": digest}
    metadata_path = output_dir / f"tile_{row}_{col}_meta.json"
    metadata_tmp = metadata_path.with_suffix(".json.tmp")
    metadata_tmp.write_text(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(metadata_tmp, metadata_path)
    return {"tile_id": tile_id, "row": row, "col": col, "tile_path": str(output_path), "sha256": digest, "reused": False}


def materialize_package_tiles(spec: Mapping[str, Any], tiles: Sequence[Mapping[str, Any]], *, workers: int = 4, progress: Callable[[int, int, Mapping[str, Any]], None] | None = None) -> list[dict[str, Any]]:
    source_path = Path(str((spec.get("raster") or {}).get("path") or "")).resolve()
    if not source_path.is_file():
        raise TileMaterializationError(f"source image is missing: {source_path}")
    output_dir = Path(str(spec["run_dir"])).resolve() / "tmp" / "tiles"
    output_dir.mkdir(parents=True, exist_ok=True)
    values = list(tiles)
    total = len(values)
    if not values:
        return []
    worker_count = max(1, min(int(workers), total, 16))
    results = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for current, result in enumerate(executor.map(lambda item: _materialize_one(source_path, output_dir, item), values), start=1):
            results.append(result)
            if progress is not None:
                progress(current, total, result)
    return results
