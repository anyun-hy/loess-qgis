import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from tile_cache_probe import measure_tile_cache
from tile_materializer import (
    TILE_MATERIALIZATION_METHOD_VERSION,
    _materialize_one,
)


ROOT = Path(__file__).resolve().parents[1]


def _compressed_uint16_source(path, *, bands=3):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=1024,
        height=512,
        count=bands,
        dtype="uint16",
        crs="EPSG:3857",
        transform=from_origin(0, 512, 1, 1),
        compress="deflate",
    ) as destination:
        destination.write(np.zeros((bands, 512, 1024), dtype=np.uint16))


def _request():
    return {
        "tile_id": "0_1",
        "row_no": 0,
        "col_no": 1,
        "bounds": {"xmin": 512, "ymin": 0, "xmax": 1024, "ymax": 512},
    }


def test_probe_uses_production_materializer_and_measures_real_uint16_tile(tmp_path):
    source = tmp_path / "compressed-source.tif"
    output_root = tmp_path / "output"
    output_root.mkdir()
    _compressed_uint16_source(source)

    report = measure_tile_cache(source, output_root, _request())

    direct = _materialize_one(source, tmp_path / "direct", _request())
    assert report["status"] == "passed"
    assert report["measurement_method"] == "tile_materializer._materialize_one"
    assert (
        report["measurement_method_version"]
        == TILE_MATERIALIZATION_METHOD_VERSION
    )
    assert report["sample_source_path"] == str(source.resolve())
    assert report["sample_source_window"] == {
        "x0": 512,
        "y0": 0,
        "x1": 1024,
        "y1": 512,
    }
    assert report["uncompressed_bytes"] == 3 * 512 * 512 * 2
    assert report["materialized_tile_bytes"] == direct["materialized_tile_bytes"]
    assert report["materialized_cache_bytes"] == (
        report["materialized_tile_bytes"] + report["metadata_bytes"]
    )
    # The compressed source is deliberately tiny; a source-file ratio would
    # not equal the production-format Tile measurement.
    assert source.stat().st_size < report["materialized_tile_bytes"]
    assert not list(output_root.glob(".loess-tile-cache-probe-*"))


def test_probe_failure_removes_disposable_directory(tmp_path):
    source = tmp_path / "two-band.tif"
    output_root = tmp_path / "output"
    output_root.mkdir()
    _compressed_uint16_source(source, bands=2)

    with pytest.raises(Exception, match="at least 3 bands"):
        measure_tile_cache(source, output_root, _request())

    assert list(output_root.iterdir()) == []


def test_probe_shell_uses_deployed_conda_environment():
    source = (ROOT / "inference_scripts" / "run_tile_cache_probe.sh").read_text(
        encoding="utf-8"
    )
    assert 'source "$SCRIPT_DIR/config.sh"' in source
    assert '"$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV"' in source
    assert 'python "$SCRIPT_DIR/tile_cache_probe.py" "$@"' in source


def test_main_dock_blocks_on_real_probe_and_freezes_measurement():
    source = (
        ROOT / "qgis_plugins" / "labeling_tool" / "gui" / "main_dock.py"
    ).read_text(encoding="utf-8")
    probe_block = source.split("def _on_tiles_extracted", 1)[1].split(
        "def _start_inference_after_tile_cache_probe", 1
    )[0]
    preflight_block = source.split(
        "def _start_inference_after_tile_cache_probe", 1
    )[1].split("stride = 512", 1)[0]

    assert "active_tiles = sorted(" in probe_block
    assert 'key=lambda item: (int(item["row"]), int(item["col"]))' in probe_block
    assert "TileCacheProbeRunner(" in probe_block
    assert "probe.succeeded.connect" in probe_block
    assert "probe.failed.connect" in probe_block
    assert 'tile_cache_sample.get("materialized_cache_bytes")' in preflight_block
    assert 'storage["input_tile_sample"] = tile_cache_sample' in preflight_block
    assert "source_bytes * pixel_count / raster_pixels" not in source
    assert "_finish_before_inference(\"Tile 存储预检失败\"" in source


def test_probe_error_report_is_machine_readable(tmp_path, capsys):
    from tile_cache_probe import main

    exit_code = main(
        [
            "--raster",
            str(tmp_path / "missing.tif"),
            "--output-root",
            str(tmp_path),
            "--tile-json",
            json.dumps(_request()),
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert report["kind"] == "tile_cache_probe"
    assert report["status"] == "error"
