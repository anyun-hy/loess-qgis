from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import tile_materializer
from storage_guard import StorageGuard
from tile_materializer import _materialize_one


def _source_raster(path):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=512,
        height=512,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 512, 1, 1),
    ) as destination:
        destination.write(np.zeros((3, 512, 512), dtype=np.uint8))


def test_materializer_settles_successful_write_to_actual_bytes(tmp_path):
    source = tmp_path / "source.tif"
    output = tmp_path / "tiles"
    _source_raster(source)
    usage = SimpleNamespace(total=10_000_000, used=0, free=10_000_000)
    guard = StorageGuard(
        tmp_path,
        min_free_bytes=1_000_000,
        managed_budget_bytes=2_000_000,
        disk_usage=lambda _path: usage,
    )

    def reserve(operation, write_bytes):
        return guard.check(
            operation,
            write_bytes=write_bytes,
            managed_growth_bytes=write_bytes,
            reserve_managed_growth=True,
        )["reserved_growth_bytes"]

    result = _materialize_one(
        source,
        output,
        {
            "tile_id": "tile-0-0",
            "row_no": 0,
            "col_no": 0,
            "pixel_window": {"x0": 0, "y0": 0, "x1": 512, "y1": 512},
        },
        before_write=reserve,
        managed_delta=guard.adjust,
    )

    actual_bytes = sum(
        (output / name).stat().st_size
        for name in ("tile_0_0.tif", "tile_0_0_meta.json")
    )
    assert result["reused"] is False
    assert guard.pending_write_bytes == 0
    assert guard.managed_bytes == actual_bytes


def test_materializer_releases_failed_write_reservation(tmp_path, monkeypatch):
    source = tmp_path / "source.tif"
    output = tmp_path / "tiles"
    _source_raster(source)
    usage = SimpleNamespace(total=10_000_000, used=0, free=10_000_000)
    guard = StorageGuard(
        tmp_path,
        min_free_bytes=1_000_000,
        managed_budget_bytes=2_000_000,
        disk_usage=lambda _path: usage,
    )

    def reserve(operation, write_bytes):
        return guard.check(
            operation,
            write_bytes=write_bytes,
            managed_growth_bytes=write_bytes,
            reserve_managed_growth=True,
        )["reserved_growth_bytes"]

    real_open = tile_materializer.rasterio.open

    def fail_output_write(path, mode="r", **kwargs):
        if mode == "w":
            raise RuntimeError("injected Tile writer failure")
        return real_open(path, mode, **kwargs)

    monkeypatch.setattr(tile_materializer.rasterio, "open", fail_output_write)
    tile = {
        "tile_id": "tile-0-0",
        "row_no": 0,
        "col_no": 0,
        "pixel_window": {"x0": 0, "y0": 0, "x1": 512, "y1": 512},
    }

    with pytest.raises(RuntimeError, match="injected Tile writer failure"):
        _materialize_one(
            source,
            output,
            tile,
            before_write=reserve,
            managed_delta=guard.adjust,
        )

    assert guard.pending_write_bytes == 0
    assert guard.managed_bytes == 0
    retry = guard.check(
        "retry",
        write_bytes=512 * 512 * 3 + 64 * 1024,
        managed_growth_bytes=512 * 512 * 3 + 64 * 1024,
        reserve_managed_growth=True,
    )
    guard.adjust(
        -retry["reserved_growth_bytes"],
        settled_write_bytes=retry["reserved_write_bytes"],
    )
