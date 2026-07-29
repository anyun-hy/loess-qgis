import pytest

from labeling_tool.core.fusion_profile import profile_snapshot, profile_summary
from labeling_tool.core.model_registry import ModelRegistry


def _effective(*, available=True, status="approved"):
    profile = {
        "schema_version": 1,
        "profile_id": "fusion_a",
        "status": status,
        "strategy": "calibrated_class_weighted",
        "models": [{"model_id": "model_b"}],
        "metrics": {
            "baseline": {"miou": 65.0},
            "fusion": {"miou": 66.0},
        },
        "approval": {"passed": status == "approved"},
    }
    return {
        "schema_version": 2,
        "runtime": {"effective_device": "cpu"},
        "sam3": {"enabled": False},
        "classes": {"background_index": -1},
        "semantic_models": [
            {
                "model_id": "model_a",
                "display_name": "Model A",
                "version": "1",
                "artifact": "a.pt",
                "artifact_path": "/tmp/a.pt",
                "sha256": "a" * 64,
                "enabled": True,
            },
            {
                "model_id": "model_b",
                "display_name": "Model B",
                "version": "1",
                "artifact": "b.pt",
                "artifact_path": "/tmp/b.pt",
                "sha256": "b" * 64,
                "enabled": True,
            },
        ],
        "fusion_profiles": [{
            "profile_id": "fusion_a",
            "file_path": "/tmp/fusion.json",
            "enabled": True,
            "available": available,
            "status": status,
            "strategy": "calibrated_class_weighted",
            "required_model_ids": ["model_b"],
            "profile": profile,
        }],
    }


def test_profile_required_models_are_auto_selected():
    registry = ModelRegistry(_effective())
    assert registry.resolve_selection(["model_a"], "fusion_a") == ("model_a", "model_b")


def test_rejected_or_unavailable_profile_cannot_run():
    registry = ModelRegistry(_effective(available=False, status="rejected"))
    with pytest.raises(ValueError, match="not runnable"):
        registry.resolve_selection(["model_a"], "fusion_a")


def test_profile_summary_and_snapshot_are_stable():
    profile = _effective()["fusion_profiles"][0]["profile"]
    summary = profile_summary(profile)
    assert summary["model_ids"] == ["model_b"]
    assert summary["fusion_miou"] == 66.0
    snapshot = profile_snapshot(profile)
    assert snapshot["profile_id"] == "fusion_a"
    assert "metrics" not in snapshot

