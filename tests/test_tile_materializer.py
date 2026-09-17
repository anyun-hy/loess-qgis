import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import tile_materializer
from storage_guard import StorageGuard
from tile_materializer import _materialize_one
from labeling_tool.core.run_spec import reserve_run_directory, source_raster_identity


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


def _package_fixture(tmp_path):
    output = tmp_path.resolve()
    run_id, run_dir = reserve_run_directory(output)
    source = output / "source.tif"
    _source_raster(source)
    spec = {
        "run_id": run_id, "run_dir": str(run_dir), "output_root": str(output),
        "cache_root": str(output / "cache" / run_id),
        "tile_cache_dir": str(output / "cache" / run_id / "tile_cache"),
        "raster": {"path": str(source), "file_identity": source_raster_identity(source)},
    }
    tiles = [{"tile_id": "tile-0-0", "row_no": 0, "col_no": 0,
              "pixel_window": {"x0": 0, "y0": 0, "x1": 512, "y1": 512}}]
    return source, spec, tiles


@pytest.mark.parametrize("cached", [False, True])
def test_package_rejects_changed_source_even_with_valid_cached_tiles(tmp_path, cached):
    source, spec, tiles = _package_fixture(tmp_path)
    if cached:
        first = tile_materializer.materialize_package_tiles(spec, tiles, workers=1)
        tiles[0]["sha256"] = first[0]["sha256"]
        assert tile_materializer.materialize_package_tiles(spec, tiles)[0]["reused"]
    previous = source.stat()
    with rasterio.open(source, "r+") as dataset:
        dataset.write(np.ones((3, 512, 512), dtype=np.uint8))
    # Same length and restored mtime must not hide an ordinary in-place edit.
    os.utime(source, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert source.stat().st_size == previous.st_size

    with pytest.raises(tile_materializer.TileMaterializationError, match="原始影像已变更"):
        tile_materializer.materialize_package_tiles(spec, tiles, workers=1)


def test_package_rejects_source_changed_during_read(tmp_path, monkeypatch):
    source, spec, tiles = _package_fixture(tmp_path)
    original_source = tile_materializer._source

    def changing_source(path):
        dataset = original_source(path)

        class ChangingReader:
            def __getattr__(self, name):
                return getattr(dataset, name)

            def read(self, *args, **kwargs):
                image = dataset.read(*args, **kwargs)
                info = source.stat()
                os.utime(source, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
                return image

        return ChangingReader()

    monkeypatch.setattr(tile_materializer, "_source", changing_source)
    with pytest.raises(tile_materializer.TileMaterializationError, match="原始影像已变更"):
        tile_materializer.materialize_package_tiles(spec, tiles, workers=1)
    assert not list((tmp_path / "cache").rglob("*.tif"))


def test_package_rejects_legacy_source_without_identity(tmp_path):
    _source, spec, tiles = _package_fixture(tmp_path)
    del spec["raster"]["file_identity"]
    with pytest.raises(tile_materializer.TileMaterializationError, match="缺少原始影像身份记录"):
        tile_materializer.materialize_package_tiles(spec, tiles)
    assert not list(Path(spec["tile_cache_dir"]).iterdir())


def test_package_checks_source_again_before_returning_cached_tiles(tmp_path):
    source, spec, tiles = _package_fixture(tmp_path)
    first = tile_materializer.materialize_package_tiles(spec, tiles)
    tiles[0]["sha256"] = first[0]["sha256"]

    def change_after_last_tile(_current, _total, result):
        assert result["reused"]
        info = source.stat()
        os.utime(source, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))

    with pytest.raises(tile_materializer.TileMaterializationError, match="原始影像已变更"):
        tile_materializer.materialize_package_tiles(spec, tiles, progress=change_after_last_tile)


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
        {"path": str(source), "file_identity": source_raster_identity(source)},
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
            {"path": str(source), "file_identity": source_raster_identity(source)},
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
