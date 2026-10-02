"""Schema v2 deployment config and fusion profile validation.

This module intentionally has no QGIS dependency so the Conda environment,
runtime scripts, and unit tests can share one contract implementation.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

from labeling_tool.shared.contracts.run_spec import (
    CLASS_NAMES,
    CLASS_ORDER,
    sha256_file,
)
from loess_runtime.system.deployment_assets import (
    validate_fusion_profiles,
    validate_model_registry,
)
from loess_runtime.system.deployment_validation import ValidationIssue
from loess_runtime.system.deployment_validation import as_mapping as _mapping
from loess_runtime.system.deployment_validation import is_valid_sha256 as _valid_sha
from loess_runtime.system.deployment_validation import (
    resolve_deployment_path as resolve_path,
)

SCHEMA_VERSION = 2
DEVICES = {"auto", "cpu", "mps", "cuda"}


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


def validate_deployment_config(
    config: Mapping[str, Any],
    *,
    scripts_dir: os.PathLike[str] | str,
    asset_base_dir: os.PathLike[str] | str | None = None,
    verify_files: bool = True,
    verify_hashes: bool = True,
) -> tuple[dict[str, Any], list[ValidationIssue]]:
    base_dir = Path(
        asset_base_dir if asset_base_dir is not None else scripts_dir
    ).resolve()
    issues: list[ValidationIssue] = []
    effective: dict[str, Any] = {
        "schema_version": config.get("schema_version"),
        "runtime": {},
        "scaling": {},
        "semantic_models": [],
        "fusion_profiles": [],
        "sam3": {},
        "boundary_fitting": {},
        "vector_data_plane": {},
        "fragmentation_regularization": {},
        "classes": {},
    }

    if "model" in config:
        issues.append(
            ValidationIssue(
                "/model",
                "legacy single-model configuration is not supported; use semantic_models",
                "legacy",
            )
        )
    if config.get("schema_version") != SCHEMA_VERSION:
        issues.append(
            ValidationIssue("/schema_version", "must equal 2", "schema_version")
        )

    runtime = _mapping(config.get("runtime"), "/runtime", issues)
    device = str(runtime.get("device", "auto")).strip().lower()
    if device not in DEVICES and not re.fullmatch(r"cuda:\d+", device):
        issues.append(
            ValidationIssue(
                "/runtime/device", "must be auto, cpu, mps, cuda, or cuda:N"
            )
        )
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
            issues.append(
                ValidationIssue(
                    "/runtime/tile_batch_size", "must be auto or at least 1"
                )
            )
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
            issues.append(
                ValidationIssue(f"/scaling/{key}", f"must be at least {minimum}")
            )
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
            issues.append(
                ValidationIssue(f"/scaling/{key}", "must be auto or at least 1")
            )
    cpu_count = os.cpu_count() or 1
    cpu_workers = normalized_scaling["max_cpu_partition_workers"]
    if isinstance(cpu_workers, int) and cpu_workers > cpu_count:
        issues.append(
            ValidationIssue(
                "/scaling/max_cpu_partition_workers",
                f"must not exceed available CPU count {cpu_count}",
            )
        )
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
            issues.append(
                ValidationIssue(
                    "/scaling/score_cache_budget_gb",
                    "must be auto or finite and positive",
                )
            )
    for key, default in (("min_free_disk_gb", 50.0),):
        try:
            value = float(scaling.get(key, default))
        except (TypeError, ValueError):
            value = float("nan")
        normalized_scaling[key] = value
        if not math.isfinite(value) or value <= 0:
            issues.append(
                ValidationIssue(f"/scaling/{key}", "must be finite and positive")
            )
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
            issues.append(
                ValidationIssue(
                    "/scaling/partition_halo_px",
                    "must be auto or at least seam_band_px; Tile overlap is checked at run creation",
                )
            )
    effective["scaling"] = normalized_scaling

    models, registry_by_id, model_issues = validate_model_registry(
        config.get("semantic_models"),
        artifacts_dir=artifacts_dir,
        verify_files=verify_files,
        verify_hashes=verify_hashes,
    )
    effective["semantic_models"] = models
    issues.extend(model_issues)

    profiles, profile_issues = validate_fusion_profiles(
        config.get("fusion_profiles", []),
        asset_base_dir=base_dir,
        artifacts_dir=artifacts_dir,
        registry_by_id=registry_by_id,
        verify_files=verify_files,
        verify_hashes=verify_hashes,
    )
    effective["fusion_profiles"] = profiles
    issues.extend(profile_issues)

    sam = _mapping(config.get("sam3", {}), "/sam3", issues)
    sam_enabled = bool(sam.get("enabled", False))
    sam_checkpoint = resolve_path(sam.get("checkpoint"), base_dir)
    expected_sam_sha = sam.get("sha256")
    if sam_enabled and not _valid_sha(expected_sam_sha):
        issues.append(
            ValidationIssue(
                "/sam3/sha256",
                "must be a lowercase SHA256 when SAM3 is enabled",
            )
        )
    try:
        buffer_px = int(sam.get("buffer_px", 32))
    except (TypeError, ValueError):
        buffer_px = -1
    if buffer_px < 0:
        issues.append(
            ValidationIssue("/sam3/buffer_px", "must be a non-negative integer")
        )
    sam_device = str(sam.get("device", "auto")).strip().lower()
    if sam_device not in DEVICES and not re.fullmatch(r"cuda:\d+", sam_device):
        issues.append(
            ValidationIssue("/sam3/device", "must be auto, cpu, mps, cuda, or cuda:N")
        )
    if sam_enabled and verify_files and not sam_checkpoint.is_file():
        issues.append(
            ValidationIssue(
                "/sam3/checkpoint", f"file does not exist: {sam_checkpoint}", "missing"
            )
        )
    actual_sam_sha = sha256_file(sam_checkpoint) if sam_checkpoint.is_file() else ""
    if (
        sam_enabled
        and verify_files
        and verify_hashes
        and _valid_sha(expected_sam_sha)
        and actual_sam_sha != expected_sam_sha
    ):
        issues.append(
            ValidationIssue(
                "/sam3/sha256",
                f"SHA256 mismatch: {actual_sam_sha}",
                "hash",
            )
        )
    effective["sam3"] = {
        "enabled": sam_enabled,
        "checkpoint": str(sam_checkpoint),
        "expected_sha256": str(expected_sam_sha or ""),
        "checkpoint_sha256": actual_sam_sha,
        "trusted": (
            _valid_sha(expected_sam_sha) and actual_sam_sha == expected_sam_sha
        ),
        "version": str(sam.get("version") or ""),
        "requested_device": sam_device,
        "buffer_px": buffer_px,
    }

    boundary = _mapping(config.get("boundary_fitting"), "/boundary_fitting", issues)
    normalized_boundary = {
        "enabled": boundary.get("enabled") is True,
        "mode": str(boundary.get("mode") or ""),
        "diagnostic_level": str(
            boundary.get("diagnostic_level") or "changed_and_failed"
        ),
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
            issues.append(
                ValidationIssue(
                    f"/boundary_fitting/{key}", "must be finite and positive"
                )
            )
    required = {"enabled": True}
    for key, expected in required.items():
        if normalized_boundary[key] is not expected:
            issues.append(
                ValidationIssue(
                    f"/boundary_fitting/{key}", f"must equal {str(expected).lower()}"
                )
            )
    if normalized_boundary["mode"] != "divider_cubic_bspline_adaptive_v2":
        issues.append(
            ValidationIssue(
                "/boundary_fitting/mode",
                "must equal divider_cubic_bspline_adaptive_v2",
            )
        )
    if normalized_boundary["diagnostic_level"] not in {"changed_and_failed", "all"}:
        issues.append(
            ValidationIssue(
                "/boundary_fitting/diagnostic_level",
                "must be changed_and_failed or all",
            )
        )
    effective["boundary_fitting"] = normalized_boundary

    vector_data_plane = _mapping(
        config.get("vector_data_plane"), "/vector_data_plane", issues
    )
    if vector_data_plane.get("enabled") is not True:
        issues.append(
            ValidationIssue(
                "/vector_data_plane/enabled",
                "must be true for the production columnar path",
            )
        )
    if vector_data_plane.get("mode") != "columnar":
        issues.append(ValidationIssue("/vector_data_plane/mode", "must equal columnar"))
    if vector_data_plane.get("dependency_policy") != "error":
        issues.append(
            ValidationIssue("/vector_data_plane/dependency_policy", "must equal error")
        )
    effective["vector_data_plane"] = {
        "enabled": True,
        "mode": "columnar",
        "dependency_policy": "error",
    }

    fragmentation = _mapping(
        config.get("fragmentation_regularization", {}),
        "/fragmentation_regularization",
        issues,
    )
    raw_enabled = fragmentation.get("enabled", True)
    if not isinstance(raw_enabled, bool):
        issues.append(
            ValidationIssue("/fragmentation_regularization/enabled", "must be boolean")
        )
    v3_policy_id = "semantic_optimized_200_v3"
    v33_policy_id = "fragmentation_v33_configurable_absorption_v1"
    policy_id = str(fragmentation.get("policy_id") or v3_policy_id)
    if policy_id not in {v3_policy_id, v33_policy_id}:
        issues.append(
            ValidationIssue(
                "/fragmentation_regularization/policy_id",
                f"must equal {v33_policy_id} or {v3_policy_id}",
            )
        )
    baseline_policy_id = str(fragmentation.get("baseline_policy_id") or v3_policy_id)
    if baseline_policy_id != v3_policy_id:
        issues.append(
            ValidationIssue(
                "/fragmentation_regularization/baseline_policy_id",
                f"must equal {v3_policy_id}",
            )
        )
    try:
        buffer_pixels = int(fragmentation.get("buffer_pixels", 256))
    except (TypeError, ValueError):
        buffer_pixels = 0
    if buffer_pixels != 256:
        issues.append(
            ValidationIssue(
                "/fragmentation_regularization/buffer_pixels",
                "must equal the verified V3 context size 256",
            )
        )
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
        if max_workers < 1 or max_workers > min(4, os.cpu_count() or 1):
            issues.append(
                ValidationIssue(
                    "/fragmentation_regularization/max_workers",
                    "must be auto or between 1 and min(4, available CPU count)",
                )
            )
    effective_fragmentation = {
        "enabled": raw_enabled is True,
        "policy_id": policy_id,
        "policy_version": (
            "v33_production_20260826"
            if policy_id == v33_policy_id
            else "semantic_optimized_200_v3_core_bounded_v1"
        ),
        "baseline_policy_id": baseline_policy_id,
        "baseline_policy_version": "semantic_optimized_200_v3_core_bounded_v1",
        "buffer_pixels": buffer_pixels,
        "requested_max_workers": requested_workers,
        "max_workers": max(1, max_workers),
        "stream_kind": "fusion",
    }
    if policy_id == v33_policy_id:
        if raw_enabled is not True:
            issues.append(
                ValidationIssue(
                    "/fragmentation_regularization/enabled",
                    "must be true when V3.3 is selected",
                )
            )
        from loess_runtime.geometry.fragmentation_v33_candidate import (
            executor_snapshot_sha256,
            policy_snapshot_sha256,
        )

        effective_fragmentation.update(
            {
                "publication": "authoritative_fusion_core",
                "policy_sha256": policy_snapshot_sha256(),
                "executor_sha256": executor_snapshot_sha256(),
            }
        )
    effective["fragmentation_regularization"] = effective_fragmentation

    classes = _mapping(config.get("classes"), "/classes", issues)
    raw_index = _mapping(classes.get("index_to_code"), "/classes/index_to_code", issues)
    index_to_code: dict[int, int] = {}
    for key, value in raw_index.items():
        try:
            index_to_code[int(key)] = int(value)
        except (TypeError, ValueError):
            issues.append(
                ValidationIssue(
                    f"/classes/index_to_code/{key}", "index and code must be integers"
                )
            )
    if [index_to_code.get(index) for index in range(14)] != CLASS_ORDER:
        issues.append(
            ValidationIssue(
                "/classes/index_to_code",
                f"must map 0..13 to {CLASS_ORDER}",
                "class_order",
            )
        )
    if classes.get("background_index", -1) != -1:
        issues.append(ValidationIssue("/classes/background_index", "must equal -1"))
    effective["classes"] = {
        "background_index": -1,
        "index_to_code": {
            str(index): code for index, code in sorted(index_to_code.items())
        },
        "mapping": {str(code): name for code, name in CLASS_NAMES.items()},
    }

    return effective, issues


def load_and_validate_config(
    config_path: os.PathLike[str] | str,
    *,
    asset_base_dir: os.PathLike[str] | str | None = None,
    verify_files: bool = True,
    verify_hashes: bool = True,
) -> tuple[dict[str, Any], list[ValidationIssue]]:
    path = Path(config_path).resolve()
    config = load_yaml(path)
    return validate_deployment_config(
        config,
        scripts_dir=path.parent,
        asset_base_dir=asset_base_dir,
        verify_files=verify_files,
        verify_hashes=verify_hashes,
    )
