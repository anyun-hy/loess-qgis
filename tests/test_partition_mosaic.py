import shutil

import numpy as np
import pytest

from labeling_tool.core.spatial_planner import plan_spatial_units
from partition_mosaic import blend_probability_tiles, build_partition_arrays


def _records(rows, cols, size, seed=7):
    generator = np.random.default_rng(seed)
    records = []
    for row in range(rows):
        for col in range(cols):
            logits = generator.normal(size=(14, size, size)).astype(np.float32)
            logits -= logits.max(axis=0, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=0, keepdims=True)
            records.append({"row": row, "col": col, "probabilities": probabilities})
    return records


def test_partition_cosine_mosaic_matches_whole_extent_reference():
    spatial = plan_spatial_units(
        tile_rows=3,
        tile_cols=3,
        tile_size=32,
        overlap=8,
        partition_tile_rows=2,
        partition_tile_cols=2,
        seam_band_px=2,
    )
    records = _records(3, 3, 32)
    reference, _weights = blend_probability_tiles(
        records,
        target_window=spatial["processing_window"],
        overlap=8,
    )
    assembled = np.zeros_like(reference)
    for partition in spatial["partitions"]:
        arrays = build_partition_arrays(records, partition, overlap=8)
        core = partition["core_window"]
        assembled[:, core["y0"] : core["y1"], core["x0"] : core["x1"]] = arrays[
            "core_probabilities"
        ]
    assert np.allclose(assembled, reference, atol=1e-6, rtol=0)
    assert np.array_equal(assembled.argmax(axis=0), reference.argmax(axis=0))


def test_partition_mosaic_rejects_uncovered_target():
    records = _records(1, 1, 32)
    with pytest.raises(RuntimeError, match="uncovered pixels"):
        blend_probability_tiles(
            records,
            target_window={"x0": 0, "y0": 0, "x1": 40, "y1": 40},
            overlap=8,
        )


def test_partition_mosaic_does_not_read_non_intersecting_score_files(tmp_path):
    probabilities = _records(1, 1, 32)[0]["probabilities"]
    records = [
        {"row": 0, "col": 0, "probabilities": probabilities},
        {
            "row": 100,
            "col": 100,
            "width": 32,
            "height": 32,
            "score_path": str(tmp_path / "must-not-be-read.npz"),
        },
    ]
    result, _weights = blend_probability_tiles(
        records,
        target_window={"x0": 0, "y0": 0, "x1": 32, "y1": 32},
        overlap=8,
    )
    assert np.allclose(result, probabilities, atol=1e-6, rtol=0)


@pytest.mark.skipif(shutil.which("gdalbuildvrt") is None, reason="GDAL CLI unavailable")
def test_gdal_cli_is_available_for_core_vrt():
    assert shutil.which("gdalbuildvrt")
