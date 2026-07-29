"""Small helpers for rendering and snapshotting validated fusion profiles."""

from __future__ import annotations

from typing import Any, Mapping


def profile_summary(profile: Mapping[str, Any]) -> dict[str, Any]:
    metrics = profile.get("metrics") or {}
    baseline = metrics.get("baseline") or {}
    fusion = metrics.get("fusion") or {}
    return {
        "profile_id": str(profile.get("profile_id") or ""),
        "status": str(profile.get("status") or "invalid"),
        "strategy": str(profile.get("strategy") or ""),
        "model_ids": [str(item.get("model_id")) for item in profile.get("models") or []],
        "baseline_miou": baseline.get("miou"),
        "fusion_miou": fusion.get("miou"),
        "approval_passed": bool((profile.get("approval") or {}).get("passed", False)),
    }


def profile_snapshot(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable runtime fields needed by run_spec.json."""
    keys = (
        "schema_version",
        "profile_id",
        "status",
        "strategy",
        "class_order",
        "input",
        "models",
        "weights",
        "fusion_head",
        "approval",
        "integrity",
    )
    return {key: profile[key] for key in keys if key in profile}

