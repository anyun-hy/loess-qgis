import numpy as np
import pytest
import rasterio
from affine import Affine

from authoritative_raster import (
    AuthoritativeRasterError,
    core_mask_tags,
    regularize_partition_core,
)
from partition_mosaic import write_partition_rasters


def test_v3_regularizes_halo_and_publishes_only_cleaned_core(tmp_path):
    probabilities = np.zeros((14, 12, 12), dtype=np.float32)
    probabilities[2, :, :] = 1.0  # class 21
    # A 1-pixel class-13 fragment is fully inside the Core and receives enough
    # neighbouring Halo context to be absorbed into class 21.
    probabilities[:, 3:7, 3:7] = 0.0  # a larger class-13 source body
    probabilities[1, 3:7, 3:7] = 1.0
    probabilities[:, 8, 8] = 0.0
    probabilities[1, 8, 8] = 0.55
    probabilities[2, 8, 8] = 0.45
    partition = {
        "partition_id": "partition_00000_00000",
        "halo_window": {"x0": 0, "y0": 0, "x1": 12, "y1": 12},
        "core_window": {"x0": 2, "y0": 2, "x1": 10, "y1": 10},
    }
    arrays = {
        "halo_probabilities": probabilities,
        "halo_weights": np.ones((12, 12), dtype=np.float32),
        "core_mask": probabilities[:, 2:10, 2:10].argmax(axis=0).astype(np.int16),
        "core_confidence": probabilities[:, 2:10, 2:10].max(axis=0),
    }

    cleaned, report = regularize_partition_core(
        arrays,
        partition,
        global_transform=Affine(10, 0, 0, 0, -10, 120),
        crs="EPSG:3857",
    )

    assert cleaned["core_mask"].shape == (8, 8)
    assert cleaned["core_mask"][6, 6] == 2
    assert report["authority"] == "partition_halo_v3_core_publish_v1"
    assert report["changed_pixel_count"] == 1
    tags = core_mask_tags(report)
    assert tags["fragmentation_policy_id"] == "semantic_optimized_200_v3"
    assert tags["fragmentation_halo_buffer_px"] == "2"
    paths = write_partition_rasters(
        cleaned,
        partition,
        global_transform=Affine(10, 0, 0, 0, -10, 120),
        crs="EPSG:3857",
        output_probability=tmp_path / "probability.tif",
        output_mask=tmp_path / "authoritative_core.tif",
        output_confidence=tmp_path / "confidence.tif",
        core_mask_tags=tags,
    )
    with rasterio.open(paths["mask"]) as source:
        persisted = source.tags()
    assert persisted["classification_authority"] == report["authority"]
    assert persisted["fragmentation_policy_version"] == (
        "semantic_optimized_200_v3_core_bounded_v1"
    )


def test_high_confidence_fragment_is_preserved_in_authoritative_stage():
    probabilities = np.zeros((14, 8, 8), dtype=np.float32)
    probabilities[2, :, :] = 1.0
    probabilities[:, 3, 3] = 0.0
    probabilities[1, 3, 3] = 0.9
    probabilities[2, 3, 3] = 0.1
    partition = {
        "partition_id": "partition_00000_00000",
        "halo_window": {"x0": 0, "y0": 0, "x1": 8, "y1": 8},
        "core_window": {"x0": 1, "y0": 1, "x1": 7, "y1": 7},
    }
    arrays = {
        "halo_probabilities": probabilities,
        "halo_weights": np.ones((8, 8), dtype=np.float32),
        "core_mask": probabilities[:, 1:7, 1:7].argmax(axis=0).astype(np.int16),
        "core_confidence": probabilities[:, 1:7, 1:7].max(axis=0),
    }

    cleaned, report = regularize_partition_core(
        arrays,
        partition,
        global_transform=Affine(10, 0, 0, 0, -10, 80),
        crs="EPSG:3857",
    )

    assert cleaned["core_mask"][2, 2] == 1
    assert report["kept_reason_counts"]["high_confidence"] == 1


def test_geographic_pixel_area_uses_partition_location():
    probabilities = np.zeros((14, 8, 8), dtype=np.float32)
    probabilities[2, :, :] = 1.0
    partition = {
        "partition_id": "partition_00001_00001",
        "halo_window": {"x0": 100, "y0": 200, "x1": 108, "y1": 208},
        "core_window": {"x0": 102, "y0": 202, "x1": 106, "y1": 206},
    }
    arrays = {
        "halo_probabilities": probabilities,
        "halo_weights": np.ones((8, 8), dtype=np.float32),
        "core_mask": np.full((4, 4), 2, dtype=np.int16),
        "core_confidence": np.ones((4, 4), dtype=np.float32),
    }

    _cleaned, report = regularize_partition_core(
        arrays,
        partition,
        global_transform=Affine(0.00001, 0, 110, 0, -0.00001, 38),
        crs="EPSG:4490",
    )

    assert 0.8 < report["pixel_area_m2"] < 1.1


def test_authoritative_core_rejects_unassigned_pixel_inside_owned_range():
    probabilities = np.zeros((14, 6, 6), dtype=np.float32)
    probabilities[2, :, :] = 1.0
    weights = np.ones((6, 6), dtype=np.float32)
    weights[3, 3] = 0.0
    partition = {
        "partition_id": "partition_00000_00000",
        "halo_window": {"x0": 0, "y0": 0, "x1": 6, "y1": 6},
        "core_window": {"x0": 1, "y0": 1, "x1": 5, "y1": 5},
    }

    with pytest.raises(
        AuthoritativeRasterError,
        match="unassigned pixel inside the owned research range",
    ):
        regularize_partition_core(
            {
                "halo_probabilities": probabilities,
                "halo_weights": weights,
                "core_mask": np.full((4, 4), 2, dtype=np.int16),
                "core_confidence": np.ones((4, 4), dtype=np.float32),
            },
            partition,
            global_transform=Affine(1, 0, 0, 0, -1, 6),
            crs="EPSG:3857",
        )


def test_authoritative_core_reports_zero_gap_coverage_contract():
    probabilities = np.zeros((14, 4, 4), dtype=np.float32)
    probabilities[2, :, :] = 1.0
    partition = {
        "partition_id": "partition_00000_00000",
        "halo_window": {"x0": 0, "y0": 0, "x1": 4, "y1": 4},
        "core_window": {"x0": 0, "y0": 0, "x1": 4, "y1": 4},
    }

    _cleaned, report = regularize_partition_core(
        {
            "halo_probabilities": probabilities,
            "halo_weights": np.ones((4, 4), dtype=np.float32),
            "core_mask": np.full((4, 4), 2, dtype=np.int16),
            "core_confidence": np.ones((4, 4), dtype=np.float32),
        },
        partition,
        global_transform=Affine(1, 0, 0, 0, -1, 4),
        crs="EPSG:3857",
    )

    assert report["coverage_validation"] == {
        "status": "passed",
        "owned_pixel_count": 16,
        "gap_pixel_count": 0,
        "invalid_confidence_pixel_count": 0,
        "outside_pixel_count": 0,
    }
