"""Measure one real Tile using the production Tile materializer.

This module runs only in the deployed inference Conda environment.  It keeps
Rasterio out of the QGIS host and intentionally delegates the write itself to
``tile_materializer._materialize_one`` so preflight cannot drift into a second
Tile implementation.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from tile_materializer import (
    TILE_MATERIALIZATION_METHOD_VERSION,
    _materialize_one,
)


PROBE_SCHEMA_VERSION = 1
PROBE_KIND = "tile_cache_probe"


class TileCacheProbeError(RuntimeError):
    pass


def _tile_request(value: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        tile = dict(value)
    else:
        try:
            tile = json.loads(str(value))
        except json.JSONDecodeError as error:
            raise TileCacheProbeError("Tile probe request is not valid JSON") from error
    if not isinstance(tile, dict):
        raise TileCacheProbeError("Tile probe request must be a JSON object")
    tile_id = str(tile.get("tile_id") or "")
    if not tile_id:
        raise TileCacheProbeError("Tile probe request is missing tile_id")
    return tile


def measure_tile_cache(
    source_path: str | Path,
    output_root: str | Path,
    tile_request: str | Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize one disposable Tile and return exact byte evidence."""

    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise TileCacheProbeError(f"source image is missing: {source}")
    configured_workspace = Path(output_root).expanduser()
    if configured_workspace.is_symlink():
        raise TileCacheProbeError(
            f"output workspace cannot be a symlink: {configured_workspace}"
        )
    workspace = configured_workspace.resolve()
    if not workspace.is_dir():
        raise TileCacheProbeError(f"output workspace is missing: {workspace}")
    tile = _tile_request(tile_request)

    result: dict[str, Any]
    with tempfile.TemporaryDirectory(
        prefix=".loess-tile-cache-probe-",
        dir=workspace,
    ) as temporary:
        result = _materialize_one(source, Path(temporary), tile)
        if result.get("reused"):
            raise TileCacheProbeError("Tile cache probe unexpectedly reused a file")
        tile_path = Path(str(result["tile_path"]))
        metadata_path = Path(str(result["metadata_path"]))
        actual_tile_bytes = int(tile_path.stat().st_size)
        actual_metadata_bytes = int(metadata_path.stat().st_size)
        actual_cache_bytes = actual_tile_bytes + actual_metadata_bytes
        if actual_cache_bytes != int(result.get("materialized_cache_bytes") or 0):
            raise TileCacheProbeError("Tile materializer byte accounting is inconsistent")
        payload = {
            "schema_version": PROBE_SCHEMA_VERSION,
            "kind": PROBE_KIND,
            "status": "passed",
            "sample_source_path": str(source),
            "sample_tile_id": str(result["tile_id"]),
            "sample_row": int(result["row"]),
            "sample_col": int(result["col"]),
            "sample_source_window": dict(result["source_window"]),
            "sample_bounds": dict(result["source_bounds"]),
            "width": 512,
            "height": 512,
            "band_count": 3,
            "uncompressed_bytes": int(result["uncompressed_bytes"]),
            "materialized_tile_bytes": actual_tile_bytes,
            "metadata_bytes": actual_metadata_bytes,
            "materialized_cache_bytes": actual_cache_bytes,
            "compression_ratio": (
                actual_tile_bytes / int(result["uncompressed_bytes"])
            ),
            "measurement_method": "tile_materializer._materialize_one",
            "measurement_method_version": TILE_MATERIALIZATION_METHOD_VERSION,
        }

    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure one production-format 512x512 RGB Tile cache entry"
    )
    parser.add_argument("--raster", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tile-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = measure_tile_cache(
            args.raster,
            args.output_root,
            args.tile_json,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "schema_version": PROBE_SCHEMA_VERSION,
                    "kind": PROBE_KIND,
                    "status": "error",
                    "message": str(error),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            flush=True,
        )
        return 1
    print(
        json.dumps(report, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
