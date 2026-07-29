import json

import numpy as np
import pytest

from fusion_runtime import fuse_probability_arrays
from incremental_fusion import FusionAccumulator, IncrementalFusionError


def _profile(strategy="calibrated_class_weighted"):
    return {
        "strategy": strategy,
        "models": [
            {"model_id": "a", "temperature": 1.2},
            {"model_id": "b", "temperature": 0.9},
        ],
        "weights": [[0.35, 0.65] for _ in range(14)],
    }


def _probabilities(seed):
    generator = np.random.default_rng(seed)
    values = generator.random((14, 8, 9), dtype=np.float32)
    return values / values.sum(axis=0, keepdims=True)


@pytest.mark.parametrize(
    "strategy",
    ["equal_probability_average", "calibrated_global_weighted", "calibrated_class_weighted"],
)
def test_incremental_fusion_matches_existing_mathematical_reference(tmp_path, strategy):
    profile = _profile(strategy)
    values = {"a": _probabilities(1), "b": _probabilities(2)}
    accumulator = FusionAccumulator(tmp_path, profile, (14, 8, 9))
    accumulator.add_model("a", values["a"])
    assert (tmp_path / "accumulator_001.npy").is_file()
    accumulator.add_model("b", values["b"])
    actual = accumulator.finalize()
    _mask, _confidence, reference_values = fuse_probability_arrays(values, profile)
    if strategy == "equal_probability_average":
        expected = reference_values
    else:
        shifted = reference_values - reference_values.max(axis=0, keepdims=True)
        expected = np.exp(shifted)
        expected /= expected.sum(axis=0, keepdims=True)
    assert np.allclose(actual, expected, atol=1e-6, rtol=0)
    assert json.loads((tmp_path / "state.json").read_text())["finalized"] is True


def test_incremental_fusion_is_ordered_idempotent_and_resumable(tmp_path):
    profile = _profile()
    first = FusionAccumulator(tmp_path, profile, (14, 8, 9))
    with pytest.raises(IncrementalFusionError, match="expected next"):
        first.add_model("b", _probabilities(2))
    state = first.add_model("a", _probabilities(1))
    assert state["completed_model_ids"] == ["a"]
    resumed = FusionAccumulator(tmp_path, profile, (14, 8, 9))
    assert resumed.add_model("a", _probabilities(99))["completed_model_ids"] == ["a"]
    resumed.add_model("b", _probabilities(2))
    assert resumed.finalize().shape == (14, 8, 9)
