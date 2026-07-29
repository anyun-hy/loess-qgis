import json

import numpy as np
import rasterio
import torch

from deployment_config import CLASS_ORDER
from fusion_runtime import FusionRuntimeError, fuse_probability_arrays, run_fusion
from semantic_batch import run_batch
from tests.test_semantic_batch import _run_spec, _write_model, _write_tile


def _probabilities(class_index, confidence=0.9, shape=(2, 2)):
    other = (1.0 - confidence) / 13.0
    result = np.full((14, *shape), other, dtype=np.float32)
    result[class_index] = confidence
    return result


def _strategy(name, model_ids, weights=None):
    count = len(model_ids)
    return {
        "strategy": name,
        "models": [
            {"model_id": model_id, "temperature": 1.0}
            for model_id in model_ids
        ],
        "weights": weights or [[1.0 / count for _ in model_ids] for _ in CLASS_ORDER],
    }


def _metric_set(miou):
    return {
        "miou": miou,
        "mf1": 70.0,
        "oa": 80.0,
        "kappa": 0.7,
        "per_class": [{} for _ in CLASS_ORDER],
        "confusion_matrix": [[0 for _ in CLASS_ORDER] for _ in CLASS_ORDER],
    }


def _approved_profile(models, strategy="equal_probability_average"):
    return {
        "schema_version": 1,
        "profile_id": "fixture_fusion",
        "status": "approved",
        "strategy": strategy,
        "class_order": CLASS_ORDER,
        "input": {"height": 512, "width": 512, "channels": 3, "dtype": "float32"},
        "models": [
            {
                "model_id": item["model_id"],
                "artifact": item["artifact"],
                "sha256": item["sha256"],
                "temperature": 1.0,
            }
            for item in models
        ],
        "weights": [[1.0 / len(models) for _ in models] for _ in CLASS_ORDER],
        "dataset": {
            "validation_count": 2,
            "validation_fingerprint": "1" * 64,
            "validation_sample_ids_sha256": "2" * 64,
            "test_count": 2,
            "test_fingerprint": "3" * 64,
            "test_sample_ids_sha256": "4" * 64,
        },
        "metrics": {
            "units": {"miou_mf1_oa_per_class": "percent", "kappa": "ratio"},
            "baseline": _metric_set(65.0),
            "fusion": _metric_set(66.0),
        },
        "approval": {
            "passed": True,
            "criterion": "fusion.test_miou > exported_swin_baseline.test_miou",
        },
        "integrity": {
            "frozen_strategy_sha256": "5" * 64,
            "test_backend": "torchscript",
            "validation_test_overlap": 0,
            "baseline_model": {
                "model_id": models[0]["model_id"],
                "artifact": models[0]["artifact"],
                "sha256": models[0]["sha256"],
            },
        },
    }


def test_equal_probability_average_preserves_probability_confidence():
    profile = _strategy("equal_probability_average", ["a", "b"])
    mask, confidence, scores = fuse_probability_arrays(
        {"a": _probabilities(0), "b": _probabilities(1)},
        profile,
    )
    assert mask.shape == (2, 2)
    assert np.allclose(confidence, scores.max(axis=0))
    assert np.allclose(scores.sum(axis=0), 1.0)


def test_global_and_class_weights_follow_calibrated_log_probabilities():
    arrays = {"a": _probabilities(0), "b": _probabilities(1)}
    global_weights = [[0.0, 1.0] for _ in CLASS_ORDER]
    global_profile = _strategy("calibrated_global_weighted", ["a", "b"], global_weights)
    mask, confidence, _ = fuse_probability_arrays(arrays, global_profile)
    assert np.all(mask == 1)
    assert np.all((confidence > 0) & (confidence <= 1))

    class_weights = [[0.5, 0.5] for _ in CLASS_ORDER]
    class_weights[0] = [1.0, 0.0]
    class_weights[1] = [0.0, 1.0]
    class_profile = _strategy("calibrated_class_weighted", ["a", "b"], class_weights)
    mask, _, scores = fuse_probability_arrays(arrays, class_profile)
    expected = np.einsum(
        "mchw,cm->chw",
        np.log(np.clip(np.stack([arrays["a"], arrays["b"]]), 1e-7, 1.0)),
        np.asarray(class_weights, dtype=np.float32),
    )
    assert np.allclose(scores, expected)
    assert np.all(mask == 0)


class _FirstModelHead(torch.nn.Module):
    def forward(self, features):
        return features[:, :14]


def test_linear_1x1_uses_model_major_gated_features():
    model_ids = [f"m{index}" for index in range(5)]
    arrays = {model_id: _probabilities(index) for index, model_id in enumerate(model_ids)}
    profile = _strategy("linear_1x1", model_ids)
    head = torch.jit.trace(_FirstModelHead(), torch.zeros(1, 70, 2, 2))
    mask, confidence, _ = fuse_probability_arrays(arrays, profile, fusion_head=head)
    assert np.all(mask == 0)
    assert np.all((confidence > 0) & (confidence <= 1))


def test_unknown_strategy_is_rejected():
    with np.testing.assert_raises(FusionRuntimeError):
        fuse_probability_arrays({"a": _probabilities(0)}, _strategy("unknown", ["a"]))


def test_fusion_run_writes_independent_tile_stream_and_resumes(tmp_path):
    tile_path = tmp_path / "tile.tif"
    _write_tile(tile_path)
    spec_path, run_dir = _run_spec(tmp_path, [{"tile_id": "0_0", "path": str(tile_path)}])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["models"][0]["artifact"] = "model_a.torchscript.pt"
    model_b_path = tmp_path / "model_b.torchscript.pt"
    model_b_sha = _write_model(model_b_path, reverse=True)
    spec["models"].append({
        "model_id": "model_b",
        "version": "fixture-v1",
        "artifact": "model_b.torchscript.pt",
        "artifact_path": str(model_b_path),
        "sha256": model_b_sha,
    })
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    run_batch(spec_path, "model_a", device="cpu")
    run_batch(spec_path, "model_b", device="cpu")

    profile = _approved_profile(spec["models"])
    profile_path = tmp_path / "fusion_profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    result = run_fusion(spec_path, profile_path, device="cpu")

    assert result["completed_count"] == 1
    assert result["failed_count"] == 0
    assert result["stream_id"] == "fusion:fixture_fusion"
    entry = result["tiles"][0]
    with rasterio.open(entry["mask"]) as src:
        assert src.read(1).shape == (512, 512)
    with rasterio.open(entry["confidence"]) as src:
        values = src.read(1)
        assert np.all((values > 0) & (values <= 1))
    with np.load(entry["scores"], allow_pickle=False) as cached:
        probabilities = cached["probabilities"]
        assert probabilities.shape == (14, 512, 512)
        assert probabilities.dtype == np.float16
    resumed = run_fusion(spec_path, profile_path, device="cpu", resume=True)
    assert resumed["reused_count"] == 1
    assert (run_dir / "tmp" / "streams" / "fusion_fixture_fusion" / "stream_manifest.json").is_file()
