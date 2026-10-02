"""Freeze the Run-local snapshots required by a planned Run."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from labeling_tool.runs.run_build_contract import FrozenRunPlan, RunBuilderV5Error
from labeling_tool.shared.contracts.run_spec import (
    CLASS_NAMES,
    CLASS_ORDER,
    atomic_write_json,
    sha256_file,
)


@dataclass(frozen=True)
class FrozenRunSnapshots:
    """Run-local snapshot paths and normalized stream configuration."""

    class_mapping_path: Path
    config_snapshot_path: Path
    models: Sequence[Mapping[str, Any]]
    fusion: Mapping[str, Any] | None
    streams: Sequence[Mapping[str, Any]]


def freeze_v5_run_snapshots(
    *,
    run_dir: Path,
    models: Sequence[Mapping[str, Any]],
    fusion: Mapping[str, Any] | None,
    effective_device: str,
    keep_score_cache: bool,
    tile_batch_size: int,
    resource_tuning: Mapping[str, Any] | None,
    plan: FrozenRunPlan,
    config_fingerprint: str,
) -> FrozenRunSnapshots:
    """Write ordered Run-local policy snapshots and normalize stream rows."""

    class_snapshot = {
        "class_mapping": {str(code): CLASS_NAMES[code] for code in CLASS_ORDER},
        "index_to_code": {str(index): code for index, code in enumerate(CLASS_ORDER)},
        "background_index": -1,
    }
    class_path = run_dir / "class_mapping_snapshot.json"
    atomic_write_json(class_path, class_snapshot)

    model_values = [dict(model) for model in models]
    model_ids = [str(model["model_id"]) for model in model_values]
    if len(set(model_ids)) != len(model_ids):
        raise RunBuilderV5Error("semantic model IDs must be unique")
    fusion_value = dict(fusion) if fusion else None
    if plan.v33_enabled and fusion_value is None:
        raise RunBuilderV5Error("V3.3 production requires an approved Fusion stream")
    if fusion_value:
        profile = fusion_value.get("profile")
        if not isinstance(profile, Mapping):
            profile_path = Path(str(fusion_value.get("profile_path") or ""))
            if not profile_path.is_file():
                raise RunBuilderV5Error("Fusion profile is missing")
            with open(profile_path, "r", encoding="utf-8") as handle:
                profile = json.load(handle)
        profile = dict(profile)
        if (
            profile.get("status") != "approved"
            or (profile.get("approval") or {}).get("passed") is not True
        ):
            raise RunBuilderV5Error("Fusion profile must be approved")
        snapshot_path = run_dir / "fusion_profile_snapshot.json"
        atomic_write_json(snapshot_path, profile)
        fusion_value.update(
            {
                "profile": profile,
                "snapshot_path": str(snapshot_path),
                "sha256": sha256_file(snapshot_path),
            }
        )

    config_snapshot = {
        "schema_version": 2,
        "runtime": {
            "effective_device": str(effective_device),
            "keep_score_cache": bool(keep_score_cache),
            "tile_batch_size": max(1, int(tile_batch_size)),
        },
        "resource_tuning": dict(resource_tuning or {}),
        "scaling": plan.scaling,
        "models": model_values,
        "fusion": fusion_value,
        "boundary_fitting": plan.boundary_fitting,
        "fragmentation_regularization": plan.fragmentation_regularization,
        "coverage_validation": {
            "policy_id": "exact_range_zero_gap_v1",
            "area_tolerance_pixels": 0.01,
        },
        "range_selection": plan.range_selection,
        "config_fingerprint": str(config_fingerprint),
    }
    config_snapshot_path = run_dir / "config_snapshot.json"
    atomic_write_json(config_snapshot_path, config_snapshot)

    stream_values: list[dict[str, Any]] = [
        {
            "stream_id": f"model:{model['model_id']}",
            "kind": "model",
            "model_id": model["model_id"],
            "version": model.get("version", ""),
        }
        for model in model_values
    ]
    if fusion_value:
        stream_values.append(
            {
                "stream_id": f"fusion:{fusion_value['profile_id']}",
                "kind": "fusion",
                "profile_id": fusion_value["profile_id"],
                "version": fusion_value.get("version", ""),
            }
        )
    return FrozenRunSnapshots(
        class_mapping_path=class_path,
        config_snapshot_path=config_snapshot_path,
        models=model_values,
        fusion=fusion_value,
        streams=stream_values,
    )
