"""Typed values shared by the Run-builder responsibility owners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

V3_POLICY_ID = "semantic_optimized_200_v3"
V33_POLICY_ID = "fragmentation_v33_configurable_absorption_v1"


class RunBuilderV5Error(ValueError):
    """A frozen Run cannot be created from the supplied inputs."""


@dataclass(frozen=True)
class FrozenRunPlan:
    """Validated policy and spatial values consumed by one Run build."""

    scaling: Mapping[str, Any]
    boundary_fitting: Mapping[str, Any]
    fragmentation_regularization: Mapping[str, Any]
    range_selection: Mapping[str, Any]
    selected_tile_count: int
    excluded_tile_count: int
    spatial_plan: Mapping[str, Any]
    package_plan: Mapping[str, Any]
    partitions: Sequence[Mapping[str, Any]]
    storage_report: Mapping[str, Any]
    v33_enabled: bool
