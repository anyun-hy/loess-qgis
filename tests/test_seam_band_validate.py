import numpy as np
import rasterio

from seam_band_validate import validate_seams


def _write_raster(path, array, transform, dtype):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=array.shape[1],
        height=array.shape[0],
        count=1,
        dtype=dtype,
        crs="EPSG:4490",
        transform=transform,
    ) as dst:
        dst.write(array.astype(dtype), 1)


def test_seam_band_acceptance_compares_against_reference_labels(tmp_path):
    transform = rasterio.transform.from_origin(0, 64, 1, 1)
    reference = np.full((64, 112), 12, dtype=np.int16)
    reference[:, 80:] = 13
    candidate = np.zeros((64, 112), dtype=np.int16)
    candidate[:, 80:] = 1
    baseline = candidate.copy()
    baseline[:, 52:56] = 1
    baseline[:, 56:60] = 0

    reference_path = tmp_path / "reference.tif"
    candidate_path = tmp_path / "candidate.tif"
    baseline_path = tmp_path / "baseline.tif"
    _write_raster(reference_path, reference, transform, "int16")
    _write_raster(candidate_path, candidate, transform, "int16")
    _write_raster(baseline_path, baseline, transform, "int16")

    tile_dir = tmp_path / "tiles"
    tile_dir.mkdir()
    rgb = np.zeros((64, 64), dtype=np.uint8)
    _write_raster(tile_dir / "tile_0_0.tif", rgb, transform, "uint8")
    _write_raster(
        tile_dir / "tile_0_1.tif",
        rgb,
        rasterio.transform.from_origin(48, 64, 1, 1),
        "uint8",
    )
    report_path = tmp_path / "seam_band_report.json"

    report = validate_seams(
        candidate_path,
        reference_path,
        tile_dir,
        report_path,
        baseline_path=baseline_path,
        band_width=4,
    )

    assert report["status"] == "passed"
    assert report["checks"] == {
        "seam_accuracy_not_lower": True,
        "seam_miou_not_lower": True,
        "seam_gap_not_wider": True,
    }
    assert report["deltas"]["seam_accuracy"] > 0
    assert report["deltas"]["seam_miou"] > 0
    assert report["deltas"]["line_transition_mean"] < 0
    assert report_path.is_file()
