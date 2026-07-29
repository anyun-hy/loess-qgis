import numpy as np
import pytest
import rasterio

from mosaic_builder import MosaicError, _cosine_axis_weights, build_mosaic


def _write(path, value, transform, dtype, size=64, crs="EPSG:4490"):
    array = np.full((size, size), value, dtype=dtype)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(array, 1)


def _write_scores(path, class_index, confidence=0.9, size=64):
    remaining = (1.0 - confidence) / 13.0
    probabilities = np.full((14, size, size), remaining, dtype=np.float32)
    probabilities[class_index] = confidence
    np.savez_compressed(path, probabilities=probabilities.astype(np.float16))


def _directories(tmp_path):
    masks = tmp_path / "masks"
    confidence = tmp_path / "confidence"
    scores = tmp_path / "scores"
    masks.mkdir()
    confidence.mkdir()
    scores.mkdir()
    return masks, confidence, scores


def test_cosine_overlap_weights_are_complementary():
    weights = _cosine_axis_weights(64, 16)

    assert np.all(weights > 0)
    assert np.allclose(weights[-16:] + weights[:16], 1.0, atol=1e-6)


def test_mosaic_blends_probabilities_using_geotransforms(tmp_path):
    masks, confidence, scores = _directories(tmp_path)
    transform_a = rasterio.transform.from_origin(100.0, 50.0, 1.0, 1.0)
    transform_b = rasterio.transform.from_origin(148.0, 50.0, 1.0, 1.0)
    _write(masks / "tile_5_7_mask.tif", 1, transform_a, "uint8")
    _write(confidence / "tile_5_7_conf.tif", 0.9, transform_a, "float32")
    _write_scores(scores / "tile_5_7_probabilities.npz", 1)
    _write(masks / "tile_5_8_mask.tif", 2, transform_b, "uint8")
    _write(confidence / "tile_5_8_conf.tif", 0.9, transform_b, "float32")
    _write_scores(scores / "tile_5_8_probabilities.npz", 2)
    output_mask = tmp_path / "mask.tif"
    output_conf = tmp_path / "conf.tif"
    output_probabilities = tmp_path / "probabilities.tif"

    result = build_mosaic(
        masks,
        confidence,
        scores,
        output_mask,
        output_conf,
        output_probabilities,
        overlap=16,
    )

    assert result["strategy"] == "cosine_probability_blend"
    assert result["width"] == 112
    assert result["height"] == 64
    with rasterio.open(output_mask) as src:
        mask = src.read(1)
        assert src.transform == rasterio.transform.from_origin(100.0, 50.0, 1.0, 1.0)
    with rasterio.open(output_conf) as src:
        conf = src.read(1)
    with rasterio.open(output_probabilities) as src:
        probability = src.read().astype(np.float32) * float(src.scales[0])
        assert src.count == 14
        assert src.dtypes == ("uint16",) * 14
        assert src.transform == rasterio.transform.from_origin(100.0, 50.0, 1.0, 1.0)
        assert src.tags()["probability_encoding"] == "uint16_scale_1_over_65535"
    assert mask[20, 50] == 1
    assert mask[20, 61] == 2
    assert 0.4 < conf[20, 55] < 0.6
    assert conf[20, 20] == pytest.approx(0.9, abs=5e-4)
    assert np.array_equal(probability.argmax(axis=0), mask)
    assert np.allclose(probability.sum(axis=0), 1.0, atol=14.0 / 65535.0)


def test_single_nonzero_tile_keeps_its_true_origin(tmp_path):
    masks, confidence, scores = _directories(tmp_path)
    transform = rasterio.transform.from_origin(123.5, 45.5, 0.5, 0.5)
    _write(masks / "tile_9_11_mask.tif", 3, transform, "uint8")
    _write(confidence / "tile_9_11_conf.tif", 0.7, transform, "float32")
    _write_scores(scores / "tile_9_11_probabilities.npz", 3, confidence=0.7)
    output_mask = tmp_path / "mask.tif"
    output_conf = tmp_path / "conf.tif"
    output_probabilities = tmp_path / "probabilities.tif"

    build_mosaic(
        masks,
        confidence,
        scores,
        output_mask,
        output_conf,
        output_probabilities,
        overlap=16,
    )

    with rasterio.open(output_mask) as src:
        assert src.transform == transform
        assert np.all(src.read(1) == 3)
    with rasterio.open(output_conf) as src:
        assert np.allclose(src.read(1), 0.7, atol=5e-4)
    with rasterio.open(output_probabilities) as src:
        probability = src.read().astype(np.float32) * float(src.scales[0])
        assert np.all(probability.argmax(axis=0) == 3)


def test_missing_confidence_is_a_hard_error(tmp_path):
    masks, confidence, scores = _directories(tmp_path)
    _write(
        masks / "tile_0_0_mask.tif",
        1,
        rasterio.transform.from_origin(0, 64, 1, 1),
        "uint8",
    )
    _write_scores(scores / "tile_0_0_probabilities.npz", 1)
    with pytest.raises(MosaicError, match="confidence tile is missing"):
        build_mosaic(
            masks,
            confidence,
            scores,
            tmp_path / "out.tif",
            tmp_path / "conf.tif",
            tmp_path / "probabilities.tif",
            overlap=16,
        )


def test_missing_probability_cache_is_a_hard_error(tmp_path):
    masks, confidence, scores = _directories(tmp_path)
    transform = rasterio.transform.from_origin(0, 64, 1, 1)
    _write(masks / "tile_0_0_mask.tif", 1, transform, "uint8")
    _write(confidence / "tile_0_0_conf.tif", 0.9, transform, "float32")

    with pytest.raises(MosaicError, match="probability cache is missing"):
        build_mosaic(
            masks,
            confidence,
            scores,
            tmp_path / "out.tif",
            tmp_path / "conf.tif",
            tmp_path / "probabilities.tif",
            overlap=16,
        )
