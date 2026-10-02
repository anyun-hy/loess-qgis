"""Shared value and input contracts for fragmentation candidate engines."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

GenericArray: TypeAlias = NDArray[np.generic]
BoolArray: TypeAlias = NDArray[np.bool_]
Float32Array: TypeAlias = NDArray[np.float32]
Int16Array: TypeAlias = NDArray[np.int16]
Int32Array: TypeAlias = NDArray[np.int32]

__all__ = [
    "ALLOWED_BASELINE_KINDS",
    "POLICY_ID",
    "POLICY_VERSION",
    "CandidateError",
    "CandidatePolicy",
    "ClassPolicy",
    "Component",
    "Proposal",
    "resolve_class_budget_mask",
    "validate_inputs",
]

POLICY_ID = "fragmentation_v31a_class_topology_candidate_v1"
POLICY_VERSION = "v31a_approved_20260824"
ALLOWED_BASELINE_KINDS = frozenset({"raw_argmax", "v3_cleaned"})


class CandidateError(RuntimeError):
    """Raised for an invalid V3.1 candidate invocation or policy."""


@dataclass(frozen=True)
class ClassPolicy:
    """Frozen physical and evidence limits for one land-cover class."""

    dynamic_fragmentation_m2: float
    ordinary_protected: bool
    enclosed_island_max_m2: float
    allow_same_class_bridge: bool
    bridge_max_edge_distance_m: float
    bridge_max_new_footprint_m2: float
    minimum_target_probability_mean: float
    maximum_current_minus_target_probability_mean: float
    minimum_target_probability_p10: float


@dataclass(frozen=True)
class CandidatePolicy:
    """Policy is data, never an implicit fallback to production V3."""

    class_policies: Mapping[int, ClassPolicy]
    semantic_compatible_targets: Mapping[int, frozenset[int]]
    protected_source_codes: frozenset[int]
    maximum_source_loss_fraction: float = 0.02
    maximum_target_gain_fraction: float = 0.02
    protected_bridge_gain_fraction: float = 0.01
    island_maximum_mean_confidence: float = 0.65
    audit_proposal_limit: int = 256
    policy_id: str = POLICY_ID
    policy_version: str = POLICY_VERSION


@dataclass(frozen=True)
class Component:
    component_id: int
    class_index: int
    class_code: int
    pixels: Int32Array  # N, 2 row/column pairs
    touches_external: bool
    slices: tuple[slice, slice]


@dataclass(frozen=True)
class Proposal:
    kind: str
    target_index: int
    target_code: int
    footprint: Int32Array
    source_indices: tuple[int, ...]
    source_codes: tuple[int, ...]
    source_component_ids: tuple[int, ...]
    baseline_target_component_ids: tuple[int, ...]
    dynamic_reduction: int
    component_reduction: int
    probability_support: float
    area_m2: float
    digest: str
    proposal_id: str
    edge_distance_m: float | None
    path_length_m: float | None
    evidence: Mapping[str, float]
    discovery_count: int = 1
    discovery_edge_distances_m: tuple[float | None, ...] = ()
    discovery_path_lengths_m: tuple[float | None, ...] = ()
    occurrence_edge_distance_m: float | None = None
    occurrence_path_length_m: float | None = None


def validate_inputs(
    labels: GenericArray,
    class_codes: Sequence[int],
    valid_mask: GenericArray | None,
    probabilities: GenericArray | None,
    confidence: GenericArray | None,
    pixel_area_m2: float,
    pixel_size_m: tuple[float, float] | None,
    policy: CandidatePolicy,
) -> tuple[
    Int16Array,
    BoolArray,
    Float32Array,
    Float32Array | None,
    tuple[float, float],
]:
    values = np.asarray(labels)
    if values.ndim != 2 or not class_codes:
        raise CandidateError(
            "labels must be two-dimensional and class_codes cannot be empty"
        )
    if len(set(int(v) for v in class_codes)) != len(class_codes):
        raise CandidateError("class_codes must be unique")
    valid = (
        np.ones(values.shape, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if valid.shape != values.shape:
        raise CandidateError("valid_mask shape does not match labels")
    if np.any(valid & ((values < 0) | (values >= len(class_codes)))):
        raise CandidateError("valid labels contain a class index outside class_codes")
    if not math.isfinite(pixel_area_m2) or pixel_area_m2 <= 0:
        raise CandidateError("pixel_area_m2 must be positive")
    if probabilities is None:
        raise CandidateError(
            "V3.1-A requires the full probability cube for every proposal"
        )
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.shape != (len(class_codes), *values.shape) or not np.all(
        np.isfinite(probs[:, valid])
    ):
        raise CandidateError(
            "probabilities must be finite with shape [len(class_codes), H, W]"
        )
    if np.any(probs[:, valid] < 0) or np.any(probs[:, valid] > 1):
        raise CandidateError("probabilities must lie in [0, 1] over valid pixels")
    probability_sums = np.sum(probs[:, valid], axis=0, dtype=np.float64)
    if not np.allclose(probability_sums, 1.0, rtol=0.0, atol=1e-3):
        raise CandidateError("probabilities must sum to one over every valid pixel")
    conf = None if confidence is None else np.asarray(confidence, dtype=np.float32)
    if conf is not None and (
        conf.shape != values.shape or not np.all(np.isfinite(conf[valid]))
    ):
        raise CandidateError("confidence must be finite and match labels")
    if conf is not None and (np.any(conf[valid] < 0) or np.any(conf[valid] > 1)):
        raise CandidateError("confidence must lie in [0, 1] over valid pixels")
    sizes = pixel_size_m or (math.sqrt(pixel_area_m2), math.sqrt(pixel_area_m2))
    if len(sizes) != 2 or any(
        not math.isfinite(float(v)) or float(v) <= 0 for v in sizes
    ):
        raise CandidateError(
            "pixel_size_m must contain positive row and column metre sizes"
        )
    if not math.isclose(
        float(sizes[0]) * float(sizes[1]),
        float(pixel_area_m2),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise CandidateError("pixel_size_m product must equal pixel_area_m2")
    unknown = set(int(v) for v in class_codes) - set(policy.class_policies)
    if unknown:
        raise CandidateError(f"policy lacks class codes: {sorted(unknown)}")
    return (
        values.astype(np.int16, copy=True),
        valid,
        probs,
        conf,
        (float(sizes[0]), float(sizes[1])),
    )


def resolve_class_budget_mask(
    class_budget_mask: GenericArray | None,
    valid: BoolArray,
) -> BoolArray:
    """Return the Core-owner pixels eligible for frozen-class budgets.

    V3.1 proposals may inspect and temporarily modify halo pixels so that
    topology is evaluated with context.  A caller-supplied owner/Core mask,
    however, is the only region whose class-change budgets are charged and
    whose labels are released in the returned raster.
    """

    if class_budget_mask is None:
        return valid.copy()
    mask = np.asarray(class_budget_mask, dtype=bool)
    if mask.shape != valid.shape:
        raise CandidateError("class_budget_mask shape does not match labels")
    mask = mask & valid
    if not np.any(mask):
        raise CandidateError("class_budget_mask must contain at least one valid pixel")
    return mask
