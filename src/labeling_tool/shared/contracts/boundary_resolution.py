"""Frozen resolution policy, shared by the QGIS host and inference runtime."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def normalize_resolution_policy(value: Any = None) -> dict[str, Any]:
    """Normalize new-Run policy without importing GIS runtime dependencies."""

    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("resolution_adaptation must be a mapping")
    enabled = value.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("resolution_adaptation.enabled must be true or false")
    raw_reference = value.get("reference_resolution_m", 2.0)
    try:
        reference = float(raw_reference)
    except (TypeError, ValueError) as error:
        raise ValueError("reference_resolution_m must be finite and positive") from error
    if isinstance(raw_reference, bool) or not math.isfinite(reference) or reference <= 0:
        raise ValueError("reference_resolution_m must be finite and positive")
    return {"enabled": enabled, "reference_resolution_m": reference}
