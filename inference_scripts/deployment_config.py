"""Schema v2 deployment config and fusion profile validation.

This module intentionally has no QGIS dependency so the Conda environment,
runtime scripts, and unit tests can share one contract implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 2
PROFILE_SCHEMA_VERSION = 1
CLASS_ORDER = [12, 13, 21, 31, 32, 33, 43, 51, 52, 53, 54, 61, 62, 71]
CLASS_NAMES = {
    12: "水浇地",
    13: "旱地",
    21: "果园",
    31: "有林地",
    32: "灌木林地",
    33: "其他林地",
    43: "其他草地",
    51: "城镇建设用地",
    52: "农村建设用地",
    53: "人为扰动用地",
    54: "其他建设用地",
    61: "农村道路",
    62: "其他交通用地",
    71: "河湖库塘",
}
FUSION_STRATEGIES = {
    "equal_probability_average",
    "calibrated_global_weighted",
    "calibrated_class_weighted",
    "linear_1x1",
}
DEVICES = {"auto", "cpu", "mps", "cuda"}
MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str
    code: str = "invalid"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: Any, base_dir: os.PathLike[str] | str) -> Path:
    raw = os.path.expandvars(os.path.expanduser(str(value or "").strip()))
    if not raw:
        return Path()
    path = Path(raw)
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path.resolve()


def load_yaml(path: os.PathLike[str] | str) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised by env checker
        raise RuntimeError("PyYAML is required to read config.yaml") from exc
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, Mapping):
        raise ValueError("config.yaml top level must be a mapping")
    return value


def load_json(path: os.PathLike[str] | str) -> Mapping[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("JSON top level must be an object")
    return value


def _mapping(value: Any, path: str, issues: list[ValidationIssue]) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    issues.append(ValidationIssue(path, "must be a mapping", "type"))
    return {}


def _list(value: Any, path: str, issues: list[ValidationIssue]) -> list[Any]:
    if isinstance(value, list):
        return value
    issues.append(ValidationIssue(path, "must be a list", "type"))
    return []


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _filename_only(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return not path.is_absolute() and path.name == value and value not in (".", "..")


def _validate_metric_set(value: Any, path: str, issues: list[ValidationIssue]) -> Mapping[str, Any]:
    metrics = _mapping(value, path, issues)
    for key in ("miou", "mf1", "oa", "kappa"):
        number = metrics.get(key)
        if not isinstance(number, (int, float)) or not math.isfinite(number):
            issues.append(ValidationIssue(f"{path}/{key}", "must be a finite number"))
    per_class = _list(metrics.get("per_class"), f"{path}/per_class", issues)
    if len(per_class) != len(CLASS_ORDER):
        issues.append(ValidationIssue(f"{path}/per_class", "must contain 14 class records"))
    confusion = _list(metrics.get("confusion_matrix"), f"{path}/confusion_matrix", issues)
    if len(confusion) != len(CLASS_ORDER):
        issues.append(ValidationIssue(f"{path}/confusion_matrix", "must contain 14 rows"))
    else:
        for index, raw_row in enumerate(confusion):
            row = _list(raw_row, f"{path}/confusion_matrix/{index}", issues)
            if len(row) != len(CLASS_ORDER) or not all(isinstance(item, int) and item >= 0 for item in row):
                issues.append(ValidationIssue(
                    f"{path}/confusion_matrix/{index}",
                    "must contain 14 non-negative integers",
                ))
    return metrics


def validate_fusion_profile(
    profile: Mapping[str, Any],
    *,
    registry_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if profile.get("schema_version") != PROFILE_SCHEMA_VERSION:
        issues.append(ValidationIssue("/schema_version", "must equal 1", "schema_version"))

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
        issues.append(ValidationIssue("/class_order", f"must equal {CLASS_ORDER}", "class_order"))

    input_cfg = _mapping(profile.get("input"), "/input", issues)
    expected_input = {"height": 512, "width": 512, "channels": 3, "dtype": "float32"}
    for key, expected in expected_input.items():
        if input_cfg.get(key) != expected:
            issues.append(ValidationIssue(f"/input/{key}", f"must equal {expected!r}"))

    models = _list(profile.get("models"), "/models", issues)
    if not models:
        issues.append(ValidationIssue("/models", "must contain at least one model"))
    model_ids: list[str] = []
    for index, raw_model in enumerate(models):
        model = _mapping(raw_model, f"/models/{index}", issues)
        model_id = model.get("model_id")
        if not isinstance(model_id, str) or not MODEL_ID_RE.fullmatch(model_id):
            issues.append(ValidationIssue(f"/models/{index}/model_id", "invalid model_id"))
            continue
        model_ids.append(model_id)
        artifact = model.get("artifact")
        if not _filename_only(artifact):
            issues.append(ValidationIssue(f"/models/{index}/artifact", "must be a portable filename"))
        if not _valid_sha(model.get("sha256")):
            issues.append(ValidationIssue(f"/models/{index}/sha256", "must be a lowercase SHA256"))
        temperature = model.get("temperature")
        if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
            issues.append(ValidationIssue(f"/models/{index}/temperature", "must be a finite positive number"))

        registered = registry_by_id.get(model_id) if registry_by_id else None
        if registry_by_id is not None and registered is None:
            issues.append(ValidationIssue(f"/models/{index}/model_id", "model is not registered", "unregistered"))
        elif registered is not None:
            if artifact != registered.get("artifact"):
                issues.append(ValidationIssue(f"/models/{index}/artifact", "does not match model registry"))
            if model.get("sha256") != registered.get("sha256"):
                issues.append(ValidationIssue(f"/models/{index}/sha256", "does not match model registry"))

    if len(model_ids) != len(set(model_ids)):
        issues.append(ValidationIssue("/models", "model_id values must be unique"))

    weights = _list(profile.get("weights"), "/weights", issues)
    if len(weights) != len(CLASS_ORDER):
        issues.append(ValidationIssue("/weights", "must contain exactly 14 rows"))
    else:
        for class_index, raw_row in enumerate(weights):
            row = _list(raw_row, f"/weights/{class_index}", issues)
            if len(row) != len(models):
                issues.append(ValidationIssue(
                    f"/weights/{class_index}",
                    f"must contain {len(models)} model weights",
                ))
                continue
            numeric = all(isinstance(item, (int, float)) and math.isfinite(item) and item >= 0 for item in row)
            if not numeric:
                issues.append(ValidationIssue(f"/weights/{class_index}", "weights must be finite and non-negative"))
            elif not math.isclose(sum(float(item) for item in row), 1.0, abs_tol=1e-6, rel_tol=0):
                issues.append(ValidationIssue(f"/weights/{class_index}", "weights must sum to 1"))

    approval = _mapping(profile.get("approval"), "/approval", issues)
    passed = approval.get("passed")
    if not isinstance(passed, bool):
        issues.append(ValidationIssue("/approval/passed", "must be boolean"))
    elif status == "approved" and not passed:
        issues.append(ValidationIssue("/approval/passed", "approved profile must pass"))
    elif status == "rejected" and passed:
        issues.append(ValidationIssue("/approval/passed", "rejected profile cannot pass"))
    if approval.get("criterion") != "fusion.test_miou > exported_swin_baseline.test_miou":
        issues.append(ValidationIssue("/approval/criterion", "unsupported approval criterion"))

    dataset = _mapping(profile.get("dataset"), "/dataset", issues)
    for key in ("validation_count", "test_count"):
        if not isinstance(dataset.get(key), int) or dataset.get(key, 0) < 1:
            issues.append(ValidationIssue(f"/dataset/{key}", "must be a positive integer"))
    for key in (
        "validation_fingerprint",
        "validation_sample_ids_sha256",
        "test_fingerprint",
        "test_sample_ids_sha256",
    ):
        if not _valid_sha(dataset.get(key)):
            issues.append(ValidationIssue(f"/dataset/{key}", "must be a lowercase SHA256"))
    if (
        _valid_sha(dataset.get("validation_sample_ids_sha256"))
        and dataset.get("validation_sample_ids_sha256") == dataset.get("test_sample_ids_sha256")
    ):
        issues.append(ValidationIssue("/dataset", "validation and test sample-id hashes must differ"))

    metrics = _mapping(profile.get("metrics"), "/metrics", issues)
    units = _mapping(metrics.get("units"), "/metrics/units", issues)
    if units.get("miou_mf1_oa_per_class") != "percent":
        issues.append(ValidationIssue("/metrics/units/miou_mf1_oa_per_class", "must equal percent"))
    if units.get("kappa") != "ratio":
        issues.append(ValidationIssue("/metrics/units/kappa", "must equal ratio"))
    baseline_metrics = _validate_metric_set(metrics.get("baseline"), "/metrics/baseline", issues)
    fusion_metrics = _validate_metric_set(metrics.get("fusion"), "/metrics/fusion", issues)
    if isinstance(passed, bool):
        baseline_miou = baseline_metrics.get("miou")
        fusion_miou = fusion_metrics.get("miou")
        if isinstance(baseline_miou, (int, float)) and isinstance(fusion_miou, (int, float)):
            expected_passed = fusion_miou > baseline_miou
            if passed != expected_passed:
                issues.append(ValidationIssue(
                    "/approval/passed",
                    "must equal fusion miou > baseline miou using unrounded values",
                ))
            expected_status = "approved" if expected_passed else "rejected"
            if status in ("approved", "rejected") and status != expected_status:
                issues.append(ValidationIssue("/status", f"metrics require status={expected_status}"))

    integrity = _mapping(profile.get("integrity"), "/integrity", issues)
    if not _valid_sha(integrity.get("frozen_strategy_sha256")):
        issues.append(ValidationIssue("/integrity/frozen_strategy_sha256", "must be a lowercase SHA256"))
    if integrity.get("test_backend") != "torchscript":
        issues.append(ValidationIssue("/integrity/test_backend", "must equal torchscript"))
    if integrity.get("validation_test_overlap") != 0:
        issues.append(ValidationIssue("/integrity/validation_test_overlap", "must equal 0"))
    baseline_model = _mapping(integrity.get("baseline_model"), "/integrity/baseline_model", issues)
    baseline_id = baseline_model.get("model_id")
    if not isinstance(baseline_id, str) or not MODEL_ID_RE.fullmatch(baseline_id):
        issues.append(ValidationIssue("/integrity/baseline_model/model_id", "invalid baseline model_id"))
    if not _filename_only(baseline_model.get("artifact")):
        issues.append(ValidationIssue("/integrity/baseline_model/artifact", "must be a portable filename"))
    if not _valid_sha(baseline_model.get("sha256")):
        issues.append(ValidationIssue("/integrity/baseline_model/sha256", "must be a lowercase SHA256"))
    registered_baseline = registry_by_id.get(baseline_id) if registry_by_id and isinstance(baseline_id, str) else None
    if registry_by_id is not None and registered_baseline is None:
        issues.append(ValidationIssue("/integrity/baseline_model/model_id", "baseline model is not registered"))
    elif registered_baseline is not None:
        if baseline_model.get("artifact") != registered_baseline.get("artifact"):
            issues.append(ValidationIssue("/integrity/baseline_model/artifact", "does not match model registry"))
        if baseline_model.get("sha256") != registered_baseline.get("sha256"):
            issues.append(ValidationIssue("/integrity/baseline_model/sha256", "does not match model registry"))

    if strategy == "linear_1x1":
        if len(models) != 5:
            issues.append(ValidationIssue("/models", "linear_1x1 profile must contain exactly 5 models"))
        head = _mapping(profile.get("fusion_head"), "/fusion_head", issues)
        if not _filename_only(head.get("artifact")):
            issues.append(ValidationIssue("/fusion_head/artifact", "must be a portable filename"))
        if not _valid_sha(head.get("sha256")):
            issues.append(ValidationIssue("/fusion_head/sha256", "must be a lowercase SHA256"))
        if head.get("input_channels") != 70:
            issues.append(ValidationIssue("/fusion_head/input_channels", "must equal 70"))
        if head.get("output_channels") != len(CLASS_ORDER):
            issues.append(ValidationIssue("/fusion_head/output_channels", "must equal 14"))
        if head.get("input_layout") != "model_major_calibrated_logits_after_profile_weights":
            issues.append(ValidationIssue("/fusion_head/input_layout", "unsupported input layout"))
    elif "fusion_head" in profile:
        issues.append(ValidationIssue("/fusion_head", "only linear_1x1 may define fusion_head"))

    return issues


def validate_deployment_config(
    config: Mapping[str, Any],
    *,
    scripts_dir: os.PathLike[str] | str,
    verify_files: bool = True,
    verify_hashes: bool = True,
) -> tuple[dict[str, Any], list[ValidationIssue]]:
    base_dir = Path(scripts_dir).resolve()
    issues: list[ValidationIssue] = []
    effective: dict[str, Any] = {
        "schema_version": config.get("schema_version"),
        "runtime": {},
        "scaling": {},
        "semantic_models": [],
        "fusion_profiles": [],
        "sam3": {},
        "boundary_fitting": {},
        "fragmentation_regularization": {},
        "classes": {},
    }

    if "model" in config:
        issues.append(ValidationIssue(
            "/model",
            "legacy single-model configuration is not supported; use semantic_models",
            "legacy",
        ))
    if config.get("schema_version") != SCHEMA_VERSION:
        issues.append(ValidationIssue("/schema_version", "must equal 2", "schema_version"))

    runtime = _mapping(config.get("runtime"), "/runtime", issues)
    device = str(runtime.get("device", "auto")).strip().lower()
    if device not in DEVICES and not re.fullmatch(r"cuda:\d+", device):
        issues.append(ValidationIssue("/runtime/device", "must be auto, cpu, mps, cuda, or cuda:N"))
    artifacts_dir = resolve_path(runtime.get("model_artifacts_dir"), base_dir)
    if not str(runtime.get("model_artifacts_dir", "")).strip():
        issues.append(ValidationIssue("/runtime/model_artifacts_dir", "is required"))
    raw_batch_size = runtime.get("tile_batch_size", "auto")
    if str(raw_batch_size).strip().lower() == "auto":
        tile_batch_size: int | str = "auto"
    else:
        try:
            tile_batch_size = int(raw_batch_size)
        except (TypeError, ValueError):
            tile_batch_size = 0
        if tile_batch_size < 1:
            issues.append(ValidationIssue("/runtime/tile_batch_size", "must be auto or at least 1"))
    effective["runtime"] = {
        "requested_device": device,
        "model_artifacts_dir": str(artifacts_dir),
        "keep_score_cache": bool(runtime.get("keep_score_cache", False)),
        "tile_batch_size": tile_batch_size,
    }

    scaling = _mapping(config.get("scaling"), "/scaling", issues)
    integer_defaults = {
        "partition_tile_rows": 8,
        "partition_tile_cols": 8,
        "seam_band_px": 64,
        "max_open_frontier_units": 64,
        "max_partition_segments": 250000,
        "max_partition_features": 100000,
        "max_partition_runtime_sec": 900,
        "max_job_retries": 2,
        "tile_page_size": 500,
    }
    normalized_scaling = {}
    for key, default in integer_defaults.items():
        try:
            value = int(scaling.get(key, default))
        except (TypeError, ValueError):
            value = 0
        normalized_scaling[key] = value
        minimum = 2 if key in {"partition_tile_rows", "partition_tile_cols"} else 1
        if value < minimum:
            issues.append(ValidationIssue(f"/scaling/{key}", f"must be at least {minimum}"))
    if normalized_scaling["tile_page_size"] > 500:
        issues.append(ValidationIssue("/scaling/tile_page_size", "must not exceed 500"))
    auto_integer_defaults = {
        "tile_io_workers": "auto",
        "max_cpu_partition_workers": "auto",
        "max_concurrent_assembly": "auto",
        "assembly_validation_workers": "auto",
    }
    for key, default in auto_integer_defaults.items():
        raw_value = scaling.get(key, default)
        if str(raw_value).strip().lower() == "auto":
            normalized_scaling[key] = "auto"
            continue
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            value = 0
        normalized_scaling[key] = value
        if value < 1:
            issues.append(ValidationIssue(f"/scaling/{key}", "must be auto or at least 1"))
    cpu_count = os.cpu_count() or 1
    cpu_workers = normalized_scaling["max_cpu_partition_workers"]
    if isinstance(cpu_workers, int) and cpu_workers > cpu_count:
        issues.append(ValidationIssue(
            "/scaling/max_cpu_partition_workers",
            f"must not exceed available CPU count {cpu_count}",
        ))
    raw_cache_budget = scaling.get("score_cache_budget_gb", "auto")
    if str(raw_cache_budget).strip().lower() == "auto":
        normalized_scaling["score_cache_budget_gb"] = "auto"
    else:
        try:
            cache_budget = float(raw_cache_budget)
        except (TypeError, ValueError):
            cache_budget = float("nan")
        normalized_scaling["score_cache_budget_gb"] = cache_budget
        if not math.isfinite(cache_budget) or cache_budget <= 0:
            issues.append(ValidationIssue(
                "/scaling/score_cache_budget_gb",
                "must be auto or finite and positive",
            ))
    for key, default in (("min_free_disk_gb", 50.0),):
        try:
            value = float(scaling.get(key, default))
        except (TypeError, ValueError):
            value = float("nan")
        normalized_scaling[key] = value
        if not math.isfinite(value) or value <= 0:
            issues.append(ValidationIssue(f"/scaling/{key}", "must be finite and positive"))
    raw_halo = scaling.get("partition_halo_px", "auto")
    if str(raw_halo).strip().lower() == "auto":
        normalized_scaling["partition_halo_px"] = "auto"
    else:
        try:
            halo = int(raw_halo)
        except (TypeError, ValueError):
            halo = 0
        normalized_scaling["partition_halo_px"] = halo
        if halo < normalized_scaling["seam_band_px"]:
            issues.append(ValidationIssue(
                "/scaling/partition_halo_px",
                "must be auto or at least seam_band_px; Tile overlap is checked at run creation",
            ))
    effective["scaling"] = normalized_scaling

    raw_models = _list(config.get("semantic_models"), "/semantic_models", issues)
    if not raw_models:
        issues.append(ValidationIssue("/semantic_models", "must contain at least one model"))
    registry_by_id: dict[str, dict[str, Any]] = {}
    for index, raw_model in enumerate(raw_models):
        model = _mapping(raw_model, f"/semantic_models/{index}", issues)
        model_id = model.get("model_id")
        if not isinstance(model_id, str) or not MODEL_ID_RE.fullmatch(model_id):
            issues.append(ValidationIssue(f"/semantic_models/{index}/model_id", "invalid model_id"))
            continue
        if model_id in registry_by_id:
            issues.append(ValidationIssue(f"/semantic_models/{index}/model_id", "duplicate model_id"))
            continue
        artifact = model.get("artifact")
        if not _filename_only(artifact):
            issues.append(ValidationIssue(f"/semantic_models/{index}/artifact", "must be a filename inside model_artifacts_dir"))
        expected_sha = model.get("sha256")
        if not _valid_sha(expected_sha):
            issues.append(ValidationIssue(f"/semantic_models/{index}/sha256", "must be a lowercase SHA256"))
        artifact_path = artifacts_dir / str(artifact or "")
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
        effective["semantic_models"].append(normalized)
        if verify_files and not artifact_path.is_file():
            issues.append(ValidationIssue(f"/semantic_models/{index}/artifact", f"file does not exist: {artifact_path}", "missing"))
        elif verify_files and verify_hashes and _valid_sha(expected_sha):
            actual_sha = sha256_file(artifact_path)
            if actual_sha != expected_sha:
                issues.append(ValidationIssue(f"/semantic_models/{index}/sha256", f"SHA256 mismatch: {actual_sha}", "hash"))

    raw_profiles = _list(config.get("fusion_profiles", []), "/fusion_profiles", issues)
    profile_ids: set[str] = set()
    for index, raw_entry in enumerate(raw_profiles):
        entry = _mapping(raw_entry, f"/fusion_profiles/{index}", issues)
        configured_id = entry.get("profile_id")
        if not isinstance(configured_id, str) or not MODEL_ID_RE.fullmatch(configured_id):
            issues.append(ValidationIssue(f"/fusion_profiles/{index}/profile_id", "invalid profile_id"))
            continue
        if configured_id in profile_ids:
            issues.append(ValidationIssue(f"/fusion_profiles/{index}/profile_id", "duplicate profile_id"))
            continue
        profile_ids.add(configured_id)
        profile_path = resolve_path(entry.get("file"), base_dir)
        expected_profile_sha = entry.get("sha256")
        if not _valid_sha(expected_profile_sha):
            issues.append(ValidationIssue(
                f"/fusion_profiles/{index}/sha256",
                "must be a lowercase SHA256",
            ))
        actual_profile_sha = (
            sha256_file(profile_path) if profile_path.is_file() else ""
        )
        profile_hash_valid = (
            _valid_sha(expected_profile_sha)
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
            issues.append(ValidationIssue(f"/fusion_profiles/{index}/file", "is required"))
        elif verify_files and not profile_path.is_file():
            issues.append(ValidationIssue(f"/fusion_profiles/{index}/file", f"file does not exist: {profile_path}", "missing"))
        elif (
            verify_files
            and verify_hashes
            and _valid_sha(expected_profile_sha)
            and not profile_hash_valid
        ):
            issues.append(ValidationIssue(
                f"/fusion_profiles/{index}/sha256",
                f"SHA256 mismatch: {actual_profile_sha}",
                "hash",
            ))
        elif profile_path.is_file():
            try:
                profile = load_json(profile_path)
                profile_issues = validate_fusion_profile(profile, registry_by_id=registry_by_id)
                for item in profile_issues:
                    issues.append(ValidationIssue(
                        f"/fusion_profiles/{index}/profile{item.path}", item.message, item.code
                    ))
                if profile.get("profile_id") != configured_id:
                    issues.append(ValidationIssue(
                        f"/fusion_profiles/{index}/profile_id",
                        "does not match profile file",
                    ))
                normalized_profile.update({
                    "available": (
                        profile_hash_valid
                        and not profile_issues
                        and profile.get("status") == "approved"
                    ),
                    "status": profile.get("status", "invalid"),
                    "strategy": profile.get("strategy", ""),
                    "required_model_ids": [item.get("model_id") for item in profile.get("models", []) if isinstance(item, Mapping)],
                    "profile": dict(profile),
                })
                if profile.get("strategy") == "linear_1x1" and isinstance(profile.get("fusion_head"), Mapping):
                    head = profile["fusion_head"]
                    head_path = artifacts_dir / str(head.get("artifact") or "")
                    if verify_files and not head_path.is_file():
                        issues.append(ValidationIssue(
                            f"/fusion_profiles/{index}/profile/fusion_head/artifact",
                            f"file does not exist: {head_path}",
                            "missing",
                        ))
                    elif verify_files and verify_hashes and _valid_sha(head.get("sha256")):
                        actual_head_sha = sha256_file(head_path)
                        if actual_head_sha != head.get("sha256"):
                            issues.append(ValidationIssue(
                                f"/fusion_profiles/{index}/profile/fusion_head/sha256",
                                f"SHA256 mismatch: {actual_head_sha}",
                                "hash",
                            ))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                issues.append(ValidationIssue(f"/fusion_profiles/{index}/file", f"cannot read profile: {exc}"))
        effective["fusion_profiles"].append(normalized_profile)

    sam = _mapping(config.get("sam3", {}), "/sam3", issues)
    sam_enabled = bool(sam.get("enabled", False))
    sam_checkpoint = resolve_path(sam.get("checkpoint"), base_dir)
    expected_sam_sha = sam.get("sha256")
    if sam_enabled and not _valid_sha(expected_sam_sha):
        issues.append(ValidationIssue(
            "/sam3/sha256",
            "must be a lowercase SHA256 when SAM3 is enabled",
        ))
    try:
        buffer_px = int(sam.get("buffer_px", 32))
    except (TypeError, ValueError):
        buffer_px = -1
    if buffer_px < 0:
        issues.append(ValidationIssue("/sam3/buffer_px", "must be a non-negative integer"))
    sam_device = str(sam.get("device", "auto")).strip().lower()
    if sam_device not in DEVICES and not re.fullmatch(r"cuda:\d+", sam_device):
        issues.append(ValidationIssue("/sam3/device", "must be auto, cpu, mps, cuda, or cuda:N"))
    if sam_enabled and verify_files and not sam_checkpoint.is_file():
        issues.append(ValidationIssue("/sam3/checkpoint", f"file does not exist: {sam_checkpoint}", "missing"))
    actual_sam_sha = (
        sha256_file(sam_checkpoint) if sam_checkpoint.is_file() else ""
    )
    if (
        sam_enabled
        and verify_files
        and verify_hashes
        and _valid_sha(expected_sam_sha)
        and actual_sam_sha != expected_sam_sha
    ):
        issues.append(ValidationIssue(
            "/sam3/sha256",
            f"SHA256 mismatch: {actual_sam_sha}",
            "hash",
        ))
    effective["sam3"] = {
        "enabled": sam_enabled,
        "checkpoint": str(sam_checkpoint),
        "expected_sha256": str(expected_sam_sha or ""),
        "checkpoint_sha256": actual_sam_sha,
        "trusted": (
            _valid_sha(expected_sam_sha)
            and actual_sam_sha == expected_sam_sha
        ),
        "version": str(sam.get("version") or ""),
        "requested_device": sam_device,
        "buffer_px": buffer_px,
    }

    boundary = _mapping(config.get("boundary_fitting"), "/boundary_fitting", issues)
    normalized_boundary = {
        "enabled": boundary.get("enabled") is True,
        "mode": str(boundary.get("mode") or ""),
        "diagnostic_level": str(boundary.get("diagnostic_level") or "changed_and_failed"),
    }
    numeric_defaults = {
        "smoothing_factor": 1.0,
        "curve_sampling_spacing_px": 0.5,
        "max_chord_error_px": 0.25,
        "max_segment_arc_length_px": 8.0,
    }
    for key, default in numeric_defaults.items():
        try:
            value = float(boundary.get(key, default))
        except (TypeError, ValueError):
            value = float("nan")
        normalized_boundary[key] = value
        if not math.isfinite(value) or value <= 0:
            issues.append(ValidationIssue(f"/boundary_fitting/{key}", "must be finite and positive"))
    required = {"enabled": True}
    for key, expected in required.items():
        if normalized_boundary[key] is not expected:
            issues.append(ValidationIssue(f"/boundary_fitting/{key}", f"must equal {str(expected).lower()}"))
    if normalized_boundary["mode"] != "divider_cubic_bspline_adaptive_v2":
        issues.append(ValidationIssue(
            "/boundary_fitting/mode",
            "must equal divider_cubic_bspline_adaptive_v2",
        ))
    if normalized_boundary["diagnostic_level"] not in {"changed_and_failed", "all"}:
        issues.append(ValidationIssue(
            "/boundary_fitting/diagnostic_level", "must be changed_and_failed or all"
        ))
    effective["boundary_fitting"] = normalized_boundary

    fragmentation = _mapping(
        config.get("fragmentation_regularization", {}),
        "/fragmentation_regularization",
        issues,
    )
    raw_enabled = fragmentation.get("enabled", True)
    if not isinstance(raw_enabled, bool):
        issues.append(ValidationIssue(
            "/fragmentation_regularization/enabled", "must be boolean"
        ))
    policy_id = str(
        fragmentation.get("policy_id") or "semantic_optimized_200_v3"
    )
    if policy_id != "semantic_optimized_200_v3":
        issues.append(ValidationIssue(
            "/fragmentation_regularization/policy_id",
            "must equal semantic_optimized_200_v3",
        ))
    try:
        buffer_pixels = int(fragmentation.get("buffer_pixels", 256))
    except (TypeError, ValueError):
        buffer_pixels = 0
    if buffer_pixels != 256:
        issues.append(ValidationIssue(
            "/fragmentation_regularization/buffer_pixels",
            "must equal the verified V3 context size 256",
        ))
    raw_workers = fragmentation.get("max_workers", "auto")
    if str(raw_workers).strip().lower() == "auto":
        requested_workers: int | str = "auto"
        max_workers = min(4, os.cpu_count() or 1)
    else:
        try:
            requested_workers = int(raw_workers)
        except (TypeError, ValueError):
            requested_workers = 0
        max_workers = int(requested_workers)
        if max_workers < 1 or max_workers > (os.cpu_count() or 1):
            issues.append(ValidationIssue(
                "/fragmentation_regularization/max_workers",
                "must be auto or between 1 and the available CPU count",
            ))
    effective["fragmentation_regularization"] = {
        "enabled": raw_enabled is True,
        "policy_id": policy_id,
        "policy_version": "semantic_optimized_200_v3_core_bounded_v1",
        "buffer_pixels": buffer_pixels,
        "requested_max_workers": requested_workers,
        "max_workers": max(1, max_workers),
        "stream_kind": "fusion",
    }

    classes = _mapping(config.get("classes"), "/classes", issues)
    raw_index = _mapping(classes.get("index_to_code"), "/classes/index_to_code", issues)
    index_to_code: dict[int, int] = {}
    for key, value in raw_index.items():
        try:
            index_to_code[int(key)] = int(value)
        except (TypeError, ValueError):
            issues.append(ValidationIssue(f"/classes/index_to_code/{key}", "index and code must be integers"))
    if [index_to_code.get(index) for index in range(14)] != CLASS_ORDER:
        issues.append(ValidationIssue("/classes/index_to_code", f"must map 0..13 to {CLASS_ORDER}", "class_order"))
    if classes.get("background_index", -1) != -1:
        issues.append(ValidationIssue("/classes/background_index", "must equal -1"))
    effective["classes"] = {
        "background_index": -1,
        "index_to_code": {str(index): code for index, code in sorted(index_to_code.items())},
        "mapping": {str(code): name for code, name in CLASS_NAMES.items()},
    }

    return effective, issues


def load_and_validate_config(
    config_path: os.PathLike[str] | str,
    *,
    verify_files: bool = True,
    verify_hashes: bool = True,
) -> tuple[dict[str, Any], list[ValidationIssue]]:
    path = Path(config_path).resolve()
    config = load_yaml(path)
    return validate_deployment_config(
        config,
        scripts_dir=path.parent,
        verify_files=verify_files,
        verify_hashes=verify_hashes,
    )
