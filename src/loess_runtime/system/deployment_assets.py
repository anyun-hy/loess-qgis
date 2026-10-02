"""Validate model registry and Fusion-profile deployment assets."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from labeling_tool.shared.contracts.run_spec import (
    CLASS_ORDER,
    load_json,
    sha256_file,
)
from loess_runtime.system.deployment_validation import (
    ValidationIssue,
    as_list,
    as_mapping,
    is_portable_filename,
    is_valid_model_id,
    is_valid_sha256,
    resolve_deployment_path,
)

PROFILE_SCHEMA_VERSION = 1
FUSION_STRATEGIES = {
    "equal_probability_average",
    "calibrated_global_weighted",
    "calibrated_class_weighted",
    "linear_1x1",
}


def _validate_metric_set(
    value: Any,
    path: str,
    issues: list[ValidationIssue],
) -> Mapping[str, Any]:
    metrics = as_mapping(value, path, issues)
    for key in ("miou", "mf1", "oa", "kappa"):
        number = metrics.get(key)
        if not isinstance(number, (int, float)) or not math.isfinite(number):
            issues.append(ValidationIssue(f"{path}/{key}", "must be a finite number"))
    per_class = as_list(metrics.get("per_class"), f"{path}/per_class", issues)
    if len(per_class) != len(CLASS_ORDER):
        issues.append(
            ValidationIssue(f"{path}/per_class", "must contain 14 class records")
        )
    confusion = as_list(
        metrics.get("confusion_matrix"), f"{path}/confusion_matrix", issues
    )
    if len(confusion) != len(CLASS_ORDER):
        issues.append(
            ValidationIssue(f"{path}/confusion_matrix", "must contain 14 rows")
        )
    else:
        for index, raw_row in enumerate(confusion):
            row = as_list(raw_row, f"{path}/confusion_matrix/{index}", issues)
            if len(row) != len(CLASS_ORDER) or not all(
                isinstance(item, int) and item >= 0 for item in row
            ):
                issues.append(
                    ValidationIssue(
                        f"{path}/confusion_matrix/{index}",
                        "must contain 14 non-negative integers",
                    )
                )
    return metrics


def validate_fusion_profile(
    profile: Mapping[str, Any],
    *,
    registry_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[ValidationIssue]:
    """Validate one decoded, frozen Fusion profile."""

    issues: list[ValidationIssue] = []
    if profile.get("schema_version") != PROFILE_SCHEMA_VERSION:
        issues.append(
            ValidationIssue("/schema_version", "must equal 1", "schema_version")
        )

    profile_id = profile.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id.strip():
        issues.append(ValidationIssue("/profile_id", "must be a non-empty string"))

    status = profile.get("status")
    if status not in ("approved", "rejected"):
        issues.append(ValidationIssue("/status", "must be approved or rejected"))

    strategy = profile.get("strategy")
    if strategy not in FUSION_STRATEGIES:
        issues.append(ValidationIssue("/strategy", "unsupported fusion strategy"))

    if profile.get("class_order") != CLASS_ORDER:
        issues.append(
            ValidationIssue("/class_order", f"must equal {CLASS_ORDER}", "class_order")
        )

    input_cfg = as_mapping(profile.get("input"), "/input", issues)
    expected_input = {"height": 512, "width": 512, "channels": 3, "dtype": "float32"}
    for key, expected in expected_input.items():
        if input_cfg.get(key) != expected:
            issues.append(ValidationIssue(f"/input/{key}", f"must equal {expected!r}"))

    models = as_list(profile.get("models"), "/models", issues)
    if not models:
        issues.append(ValidationIssue("/models", "must contain at least one model"))
    model_ids: list[str] = []
    for index, raw_model in enumerate(models):
        model = as_mapping(raw_model, f"/models/{index}", issues)
        model_id = model.get("model_id")
        if not isinstance(model_id, str) or not is_valid_model_id(model_id):
            issues.append(
                ValidationIssue(f"/models/{index}/model_id", "invalid model_id")
            )
            continue
        model_ids.append(model_id)
        artifact = model.get("artifact")
        if not is_portable_filename(artifact):
            issues.append(
                ValidationIssue(
                    f"/models/{index}/artifact", "must be a portable filename"
                )
            )
        if not is_valid_sha256(model.get("sha256")):
            issues.append(
                ValidationIssue(f"/models/{index}/sha256", "must be a lowercase SHA256")
            )
        temperature = model.get("temperature")
        if (
            not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            issues.append(
                ValidationIssue(
                    f"/models/{index}/temperature",
                    "must be a finite positive number",
                )
            )

        registered = registry_by_id.get(model_id) if registry_by_id else None
        if registry_by_id is not None and registered is None:
            issues.append(
                ValidationIssue(
                    f"/models/{index}/model_id",
                    "model is not registered",
                    "unregistered",
                )
            )
        elif registered is not None:
            if artifact != registered.get("artifact"):
                issues.append(
                    ValidationIssue(
                        f"/models/{index}/artifact", "does not match model registry"
                    )
                )
            if model.get("sha256") != registered.get("sha256"):
                issues.append(
                    ValidationIssue(
                        f"/models/{index}/sha256", "does not match model registry"
                    )
                )

    if len(model_ids) != len(set(model_ids)):
        issues.append(ValidationIssue("/models", "model_id values must be unique"))

    weights = as_list(profile.get("weights"), "/weights", issues)
    if len(weights) != len(CLASS_ORDER):
        issues.append(ValidationIssue("/weights", "must contain exactly 14 rows"))
    else:
        for class_index, raw_row in enumerate(weights):
            row = as_list(raw_row, f"/weights/{class_index}", issues)
            if len(row) != len(models):
                issues.append(
                    ValidationIssue(
                        f"/weights/{class_index}",
                        f"must contain {len(models)} model weights",
                    )
                )
                continue
            numeric = all(
                isinstance(item, (int, float)) and math.isfinite(item) and item >= 0
                for item in row
            )
            if not numeric:
                issues.append(
                    ValidationIssue(
                        f"/weights/{class_index}",
                        "weights must be finite and non-negative",
                    )
                )
            elif not math.isclose(
                sum(float(item) for item in row), 1.0, abs_tol=1e-6, rel_tol=0
            ):
                issues.append(
                    ValidationIssue(f"/weights/{class_index}", "weights must sum to 1")
                )

    approval = as_mapping(profile.get("approval"), "/approval", issues)
    passed = approval.get("passed")
    if not isinstance(passed, bool):
        issues.append(ValidationIssue("/approval/passed", "must be boolean"))
    elif status == "approved" and not passed:
        issues.append(ValidationIssue("/approval/passed", "approved profile must pass"))
    elif status == "rejected" and passed:
        issues.append(
            ValidationIssue("/approval/passed", "rejected profile cannot pass")
        )
    if (
        approval.get("criterion")
        != "fusion.test_miou > exported_swin_baseline.test_miou"
    ):
        issues.append(
            ValidationIssue("/approval/criterion", "unsupported approval criterion")
        )

    dataset = as_mapping(profile.get("dataset"), "/dataset", issues)
    for key in ("validation_count", "test_count"):
        if not isinstance(dataset.get(key), int) or dataset.get(key, 0) < 1:
            issues.append(
                ValidationIssue(f"/dataset/{key}", "must be a positive integer")
            )
    for key in (
        "validation_fingerprint",
        "validation_sample_ids_sha256",
        "test_fingerprint",
        "test_sample_ids_sha256",
    ):
        if not is_valid_sha256(dataset.get(key)):
            issues.append(
                ValidationIssue(f"/dataset/{key}", "must be a lowercase SHA256")
            )
    if is_valid_sha256(dataset.get("validation_sample_ids_sha256")) and dataset.get(
        "validation_sample_ids_sha256"
    ) == dataset.get("test_sample_ids_sha256"):
        issues.append(
            ValidationIssue(
                "/dataset", "validation and test sample-id hashes must differ"
            )
        )

    metrics = as_mapping(profile.get("metrics"), "/metrics", issues)
    units = as_mapping(metrics.get("units"), "/metrics/units", issues)
    if units.get("miou_mf1_oa_per_class") != "percent":
        issues.append(
            ValidationIssue(
                "/metrics/units/miou_mf1_oa_per_class", "must equal percent"
            )
        )
    if units.get("kappa") != "ratio":
        issues.append(ValidationIssue("/metrics/units/kappa", "must equal ratio"))
    baseline_metrics = _validate_metric_set(
        metrics.get("baseline"), "/metrics/baseline", issues
    )
    fusion_metrics = _validate_metric_set(
        metrics.get("fusion"), "/metrics/fusion", issues
    )
    if isinstance(passed, bool):
        baseline_miou = baseline_metrics.get("miou")
        fusion_miou = fusion_metrics.get("miou")
        if isinstance(baseline_miou, (int, float)) and isinstance(
            fusion_miou, (int, float)
        ):
            expected_passed = fusion_miou > baseline_miou
            if passed != expected_passed:
                issues.append(
                    ValidationIssue(
                        "/approval/passed",
                        "must equal fusion miou > baseline miou using unrounded values",
                    )
                )
            expected_status = "approved" if expected_passed else "rejected"
            if status in ("approved", "rejected") and status != expected_status:
                issues.append(
                    ValidationIssue(
                        "/status", f"metrics require status={expected_status}"
                    )
                )

    integrity = as_mapping(profile.get("integrity"), "/integrity", issues)
    if not is_valid_sha256(integrity.get("frozen_strategy_sha256")):
        issues.append(
            ValidationIssue(
                "/integrity/frozen_strategy_sha256", "must be a lowercase SHA256"
            )
        )
    if integrity.get("test_backend") != "torchscript":
        issues.append(
            ValidationIssue("/integrity/test_backend", "must equal torchscript")
        )
    if integrity.get("validation_test_overlap") != 0:
        issues.append(
            ValidationIssue("/integrity/validation_test_overlap", "must equal 0")
        )
    baseline_model = as_mapping(
        integrity.get("baseline_model"), "/integrity/baseline_model", issues
    )
    baseline_id = baseline_model.get("model_id")
    if not isinstance(baseline_id, str) or not is_valid_model_id(baseline_id):
        issues.append(
            ValidationIssue(
                "/integrity/baseline_model/model_id", "invalid baseline model_id"
            )
        )
    if not is_portable_filename(baseline_model.get("artifact")):
        issues.append(
            ValidationIssue(
                "/integrity/baseline_model/artifact", "must be a portable filename"
            )
        )
    if not is_valid_sha256(baseline_model.get("sha256")):
        issues.append(
            ValidationIssue(
                "/integrity/baseline_model/sha256", "must be a lowercase SHA256"
            )
        )
    registered_baseline = (
        registry_by_id.get(baseline_id)
        if registry_by_id and isinstance(baseline_id, str)
        else None
    )
    if registry_by_id is not None and registered_baseline is None:
        issues.append(
            ValidationIssue(
                "/integrity/baseline_model/model_id", "baseline model is not registered"
            )
        )
    elif registered_baseline is not None:
        if baseline_model.get("artifact") != registered_baseline.get("artifact"):
            issues.append(
                ValidationIssue(
                    "/integrity/baseline_model/artifact",
                    "does not match model registry",
                )
            )
        if baseline_model.get("sha256") != registered_baseline.get("sha256"):
            issues.append(
                ValidationIssue(
                    "/integrity/baseline_model/sha256",
                    "does not match model registry",
                )
            )

    if strategy == "linear_1x1":
        if len(models) != 5:
            issues.append(
                ValidationIssue(
                    "/models", "linear_1x1 profile must contain exactly 5 models"
                )
            )
        head = as_mapping(profile.get("fusion_head"), "/fusion_head", issues)
        if not is_portable_filename(head.get("artifact")):
            issues.append(
                ValidationIssue("/fusion_head/artifact", "must be a portable filename")
            )
        if not is_valid_sha256(head.get("sha256")):
            issues.append(
                ValidationIssue("/fusion_head/sha256", "must be a lowercase SHA256")
            )
        if head.get("input_channels") != 70:
            issues.append(
                ValidationIssue("/fusion_head/input_channels", "must equal 70")
            )
        if head.get("output_channels") != len(CLASS_ORDER):
            issues.append(
                ValidationIssue("/fusion_head/output_channels", "must equal 14")
            )
        if (
            head.get("input_layout")
            != "model_major_calibrated_logits_after_profile_weights"
        ):
            issues.append(
                ValidationIssue("/fusion_head/input_layout", "unsupported input layout")
            )
    elif "fusion_head" in profile:
        issues.append(
            ValidationIssue("/fusion_head", "only linear_1x1 may define fusion_head")
        )

    return issues


def validate_model_registry(
    raw_models: Any,
    *,
    artifacts_dir: os.PathLike[str] | str,
    verify_files: bool,
    verify_hashes: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[ValidationIssue]]:
    """Validate and normalize configured semantic model assets."""

    issues: list[ValidationIssue] = []
    models = as_list(raw_models, "/semantic_models", issues)
    if not models:
        issues.append(
            ValidationIssue("/semantic_models", "must contain at least one model")
        )
    normalized_models: list[dict[str, Any]] = []
    registry_by_id: dict[str, dict[str, Any]] = {}
    for index, raw_model in enumerate(models):
        model = as_mapping(raw_model, f"/semantic_models/{index}", issues)
        model_id = model.get("model_id")
        if not isinstance(model_id, str) or not is_valid_model_id(model_id):
            issues.append(
                ValidationIssue(
                    f"/semantic_models/{index}/model_id", "invalid model_id"
                )
            )
            continue
        if model_id in registry_by_id:
            issues.append(
                ValidationIssue(
                    f"/semantic_models/{index}/model_id", "duplicate model_id"
                )
            )
            continue
        artifact = model.get("artifact")
        if not is_portable_filename(artifact):
            issues.append(
                ValidationIssue(
                    f"/semantic_models/{index}/artifact",
                    "must be a filename inside model_artifacts_dir",
                )
            )
        expected_sha = model.get("sha256")
        if not is_valid_sha256(expected_sha):
            issues.append(
                ValidationIssue(
                    f"/semantic_models/{index}/sha256", "must be a lowercase SHA256"
                )
            )
        artifact_path = Path(artifacts_dir) / str(artifact or "")
        normalized = {
            "model_id": model_id,
            "display_name": str(model.get("display_name") or model_id),
            "version": str(model.get("version") or ""),
            "artifact": str(artifact or ""),
            "artifact_path": str(artifact_path),
            "sha256": str(expected_sha or ""),
            "enabled": bool(model.get("enabled", True)),
        }
        registry_by_id[model_id] = normalized
        normalized_models.append(normalized)
        if verify_files and not artifact_path.is_file():
            issues.append(
                ValidationIssue(
                    f"/semantic_models/{index}/artifact",
                    f"file does not exist: {artifact_path}",
                    "missing",
                )
            )
        elif verify_files and verify_hashes and is_valid_sha256(expected_sha):
            actual_sha = sha256_file(artifact_path)
            if actual_sha != expected_sha:
                issues.append(
                    ValidationIssue(
                        f"/semantic_models/{index}/sha256",
                        f"SHA256 mismatch: {actual_sha}",
                        "hash",
                    )
                )
    return normalized_models, registry_by_id, issues


def validate_fusion_profiles(
    raw_entries: Any,
    *,
    asset_base_dir: os.PathLike[str] | str,
    artifacts_dir: os.PathLike[str] | str,
    registry_by_id: Mapping[str, Mapping[str, Any]],
    verify_files: bool,
    verify_hashes: bool,
) -> tuple[list[dict[str, Any]], list[ValidationIssue]]:
    """Validate and normalize configured Fusion profile assets."""

    issues: list[ValidationIssue] = []
    raw_profiles = as_list(raw_entries, "/fusion_profiles", issues)
    normalized_profiles: list[dict[str, Any]] = []
    profile_ids: set[str] = set()
    for index, raw_entry in enumerate(raw_profiles):
        entry = as_mapping(raw_entry, f"/fusion_profiles/{index}", issues)
        configured_id = entry.get("profile_id")
        if not isinstance(configured_id, str) or not is_valid_model_id(configured_id):
            issues.append(
                ValidationIssue(
                    f"/fusion_profiles/{index}/profile_id", "invalid profile_id"
                )
            )
            continue
        if configured_id in profile_ids:
            issues.append(
                ValidationIssue(
                    f"/fusion_profiles/{index}/profile_id", "duplicate profile_id"
                )
            )
            continue
        profile_ids.add(configured_id)
        profile_path = resolve_deployment_path(entry.get("file"), asset_base_dir)
        expected_profile_sha = entry.get("sha256")
        if not is_valid_sha256(expected_profile_sha):
            issues.append(
                ValidationIssue(
                    f"/fusion_profiles/{index}/sha256", "must be a lowercase SHA256"
                )
            )
        actual_profile_sha = sha256_file(profile_path) if profile_path.is_file() else ""
        profile_hash_valid = (
            is_valid_sha256(expected_profile_sha)
            and actual_profile_sha == expected_profile_sha
        )
        normalized_profile: dict[str, Any] = {
            "profile_id": configured_id,
            "file": str(entry.get("file") or ""),
            "file_path": str(profile_path),
            "enabled": bool(entry.get("enabled", True)),
            "sha256": str(expected_profile_sha or ""),
            "file_sha256": actual_profile_sha,
            "trusted": profile_hash_valid,
            "available": False,
            "profile": None,
        }
        if not str(entry.get("file", "")).strip():
            issues.append(
                ValidationIssue(f"/fusion_profiles/{index}/file", "is required")
            )
        elif verify_files and not profile_path.is_file():
            issues.append(
                ValidationIssue(
                    f"/fusion_profiles/{index}/file",
                    f"file does not exist: {profile_path}",
                    "missing",
                )
            )
        elif (
            verify_files
            and verify_hashes
            and is_valid_sha256(expected_profile_sha)
            and not profile_hash_valid
        ):
            issues.append(
                ValidationIssue(
                    f"/fusion_profiles/{index}/sha256",
                    f"SHA256 mismatch: {actual_profile_sha}",
                    "hash",
                )
            )
        elif profile_path.is_file():
            try:
                profile = load_json(profile_path)
                profile_issues = validate_fusion_profile(
                    profile, registry_by_id=registry_by_id
                )
                for item in profile_issues:
                    issues.append(
                        ValidationIssue(
                            f"/fusion_profiles/{index}/profile{item.path}",
                            item.message,
                            item.code,
                        )
                    )
                if profile.get("profile_id") != configured_id:
                    issues.append(
                        ValidationIssue(
                            f"/fusion_profiles/{index}/profile_id",
                            "does not match profile file",
                        )
                    )
                normalized_profile.update(
                    {
                        "available": (
                            profile_hash_valid
                            and not profile_issues
                            and profile.get("status") == "approved"
                        ),
                        "status": profile.get("status", "invalid"),
                        "strategy": profile.get("strategy", ""),
                        "required_model_ids": [
                            item.get("model_id")
                            for item in profile.get("models", [])
                            if isinstance(item, Mapping)
                        ],
                        "profile": dict(profile),
                    }
                )
                if profile.get("strategy") == "linear_1x1":
                    if isinstance(profile.get("fusion_head"), Mapping):
                        head = profile["fusion_head"]
                        head_path = Path(artifacts_dir) / str(
                            head.get("artifact") or ""
                        )
                        if verify_files and not head_path.is_file():
                            issues.append(
                                ValidationIssue(
                                    f"/fusion_profiles/{index}/profile/fusion_head/artifact",
                                    f"file does not exist: {head_path}",
                                    "missing",
                                )
                            )
                        elif (
                            verify_files
                            and verify_hashes
                            and is_valid_sha256(head.get("sha256"))
                        ):
                            actual_head_sha = sha256_file(head_path)
                            if actual_head_sha != head.get("sha256"):
                                issues.append(
                                    ValidationIssue(
                                        f"/fusion_profiles/{index}/profile/fusion_head/sha256",
                                        f"SHA256 mismatch: {actual_head_sha}",
                                        "hash",
                                    )
                                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                issues.append(
                    ValidationIssue(
                        f"/fusion_profiles/{index}/file", f"cannot read profile: {exc}"
                    )
                )
        normalized_profiles.append(normalized_profile)
    return normalized_profiles, issues
