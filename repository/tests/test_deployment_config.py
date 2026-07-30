import hashlib
import json

from deployment_config import (
    CLASS_ORDER,
    validate_deployment_config,
    validate_fusion_profile,
)


def _metric_set(miou):
    return {
        "miou": miou,
        "mf1": 70.0,
        "oa": 80.0,
        "kappa": 0.7,
        "per_class": [{} for _ in CLASS_ORDER],
        "confusion_matrix": [[0 for _ in CLASS_ORDER] for _ in CLASS_ORDER],
    }


def _profile(model_sha, *, status="approved"):
    passed = status == "approved"
    return {
        "schema_version": 1,
        "profile_id": "test_fusion",
        "status": status,
        "strategy": "calibrated_class_weighted",
        "class_order": CLASS_ORDER,
        "input": {"height": 512, "width": 512, "channels": 3, "dtype": "float32"},
        "models": [{
            "model_id": "model_a",
            "artifact": "model_a.torchscript.pt",
            "sha256": model_sha,
            "temperature": 1.2,
        }],
        "weights": [[1.0] for _ in CLASS_ORDER],
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
            "fusion": _metric_set(66.0 if passed else 65.0),
        },
        "approval": {
            "passed": passed,
            "criterion": "fusion.test_miou > exported_swin_baseline.test_miou",
        },
        "integrity": {
            "frozen_strategy_sha256": "5" * 64,
            "test_backend": "torchscript",
            "validation_test_overlap": 0,
            "baseline_model": {
                "model_id": "model_a",
                "artifact": "model_a.torchscript.pt",
                "sha256": model_sha,
            },
        },
    }


def _config(model_sha):
    return {
        "schema_version": 2,
        "runtime": {
            "device": "cpu",
            "model_artifacts_dir": "../weights",
            "keep_score_cache": False,
            "tile_batch_size": 1,
        },
        "scaling": {
            "partition_tile_rows": 8,
            "partition_tile_cols": 8,
            "partition_halo_px": "auto",
            "seam_band_px": 64,
            "score_cache_budget_gb": 16,
            "min_free_disk_gb": 1,
            "max_cpu_partition_workers": 2,
            "max_open_frontier_units": 64,
            "max_partition_segments": 250000,
            "max_partition_features": 100000,
            "max_partition_runtime_sec": 900,
            "max_job_retries": 2,
            "tile_page_size": 500,
        },
        "semantic_models": [{
            "model_id": "model_a",
            "display_name": "Model A",
            "version": "test-v1",
            "artifact": "model_a.torchscript.pt",
            "sha256": model_sha,
            "enabled": True,
        }],
        "fusion_profiles": [{
            "profile_id": "test_fusion",
            "file": "../weights/fusion_profile.json",
            "sha256": "",
            "enabled": True,
        }],
        "sam3": {"enabled": False, "device": "cpu", "buffer_px": 32},
        "boundary_fitting": {
            "enabled": True,
            "mode": "divider_cubic_bspline_v1",
            "smoothing_factor": 1.0,
            "output_spacing_px": 0.5,
            "diagnostic_level": "changed_and_failed",
        },
        "classes": {
            "background_index": -1,
            "index_to_code": {index: code for index, code in enumerate(CLASS_ORDER)},
        },
    }


def _workspace(tmp_path, *, status="approved"):
    scripts = tmp_path / "inference_scripts"
    weights = tmp_path / "weights"
    scripts.mkdir()
    weights.mkdir()
    artifact = weights / "model_a.torchscript.pt"
    artifact.write_bytes(b"fixture-model")
    model_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    profile = _profile(model_sha, status=status)
    profile_path = weights / "fusion_profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    config = _config(model_sha)
    config["fusion_profiles"][0]["sha256"] = hashlib.sha256(
        profile_path.read_bytes()
    ).hexdigest()
    return scripts, config, profile


def test_valid_schema_v2_registry_and_profile(tmp_path):
    scripts, config, _ = _workspace(tmp_path)
    effective, issues = validate_deployment_config(
        config,
        scripts_dir=scripts,
        verify_files=True,
        verify_hashes=True,
    )
    assert issues == []
    assert effective["schema_version"] == 2
    assert effective["semantic_models"][0]["model_id"] == "model_a"
    assert effective["fusion_profiles"][0]["available"] is True
    assert effective["fusion_profiles"][0]["required_model_ids"] == ["model_a"]
    assert effective["classes"]["background_index"] == -1
    assert effective["scaling"]["partition_halo_px"] == "auto"
    assert effective["boundary_fitting"]["mode"] == "divider_cubic_bspline_v1"


def test_legacy_single_model_config_is_rejected(tmp_path):
    effective, issues = validate_deployment_config(
        {"model": {"semantic_weight": "legacy.pth"}},
        scripts_dir=tmp_path,
        verify_files=False,
    )
    assert effective["schema_version"] is None
    assert any(issue.code == "legacy" and issue.path == "/model" for issue in issues)
    assert any(issue.path == "/schema_version" for issue in issues)


def test_registry_hash_mismatch_blocks_profile(tmp_path):
    scripts, config, _ = _workspace(tmp_path)
    config["semantic_models"][0]["sha256"] = "a" * 64
    _, issues = validate_deployment_config(
        config,
        scripts_dir=scripts,
        verify_files=True,
        verify_hashes=True,
    )
    assert any(issue.code == "hash" for issue in issues)
    assert any("does not match model registry" in issue.message for issue in issues)


def test_fusion_profile_requires_registered_file_hash(tmp_path):
    scripts, config, _ = _workspace(tmp_path)
    profile_path = tmp_path / "weights" / "fusion_profile.json"
    profile_path.write_text(
        profile_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    effective, issues = validate_deployment_config(
        config,
        scripts_dir=scripts,
        verify_files=True,
        verify_hashes=True,
    )
    assert any(
        issue.path == "/fusion_profiles/0/sha256"
        and issue.code == "hash"
        for issue in issues
    )
    assert effective["fusion_profiles"][0]["trusted"] is False
    assert effective["fusion_profiles"][0]["available"] is False


def test_enabled_sam3_requires_registered_checkpoint_hash(tmp_path):
    scripts, config, _ = _workspace(tmp_path)
    checkpoint = tmp_path / "weights" / "sam3.pt"
    checkpoint.write_bytes(b"approved-sam-checkpoint")
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config["sam3"] = {
        "enabled": True,
        "checkpoint": "../weights/sam3.pt",
        "sha256": expected,
        "version": "sam3-test",
        "device": "cpu",
        "buffer_px": 32,
    }
    checkpoint.write_bytes(b"changed-sam-checkpoint")
    effective, issues = validate_deployment_config(
        config,
        scripts_dir=scripts,
        verify_files=True,
        verify_hashes=True,
    )
    assert any(
        issue.path == "/sam3/sha256" and issue.code == "hash"
        for issue in issues
    )
    assert effective["sam3"]["expected_sha256"] == expected
    assert effective["sam3"]["trusted"] is False


def test_enabled_sam3_cannot_omit_trusted_hash(tmp_path):
    scripts, config, _ = _workspace(tmp_path)
    checkpoint = tmp_path / "weights" / "sam3.pt"
    checkpoint.write_bytes(b"sam-checkpoint")
    config["sam3"] = {
        "enabled": True,
        "checkpoint": "../weights/sam3.pt",
        "device": "cpu",
        "buffer_px": 32,
    }
    _, issues = validate_deployment_config(
        config,
        scripts_dir=scripts,
        verify_files=True,
        verify_hashes=True,
    )
    assert any(issue.path == "/sam3/sha256" for issue in issues)


def test_rejected_profile_is_valid_but_not_runnable(tmp_path):
    scripts, config, profile = _workspace(tmp_path, status="rejected")
    effective, issues = validate_deployment_config(
        config,
        scripts_dir=scripts,
        verify_files=True,
        verify_hashes=True,
    )
    assert validate_fusion_profile(profile, registry_by_id={
        "model_a": effective["semantic_models"][0]
    }) == []
    assert issues == []
    assert effective["fusion_profiles"][0]["status"] == "rejected"
    assert effective["fusion_profiles"][0]["available"] is False


def test_profile_class_order_and_weight_shape_are_strict(tmp_path):
    scripts, config, profile = _workspace(tmp_path)
    effective, _ = validate_deployment_config(
        config,
        scripts_dir=scripts,
        verify_files=True,
        verify_hashes=True,
    )
    profile["class_order"] = list(reversed(CLASS_ORDER))
    profile["weights"] = [[1.0]]
    issues = validate_fusion_profile(
        profile,
        registry_by_id={"model_a": effective["semantic_models"][0]},
    )
    assert any(issue.path == "/class_order" for issue in issues)
    assert any(issue.path == "/weights" for issue in issues)
