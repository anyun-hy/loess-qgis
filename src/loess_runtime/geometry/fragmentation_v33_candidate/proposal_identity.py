"""Stable proposal identity, ranking, and V3.1-B canonicalization."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from loess_runtime.geometry.fragmentation_v33_candidate.contracts import (
    CandidateError,
    Proposal,
)

__all__ = [
    "canonicalize_v31b_proposals",
    "footprint_digest",
    "v31_stable_rank_key",
]

Int32Array: TypeAlias = NDArray[np.int32]


def footprint_digest(pixels: Int32Array) -> str:
    ordered = np.asarray(sorted((int(r), int(c)) for r, c in pixels), dtype="<i4")
    return hashlib.sha256(ordered.tobytes()).hexdigest()


def v31_stable_rank_key(proposal: Proposal) -> tuple[float, float, float, float, str]:
    """The frozen policy ranking, expressed in ascending sort order."""

    return (
        -float(proposal.dynamic_reduction),
        -float(proposal.component_reduction),
        -float(proposal.probability_support),
        float(proposal.area_m2),
        proposal.digest,
    )


def _canonical_proposal_key(proposal: Proposal) -> tuple[Any, ...]:
    """Identity of a proposal before it enters the B dependency graph."""

    return (
        proposal.kind,
        int(proposal.target_index),
        int(proposal.target_code),
        tuple(sorted((int(row), int(col)) for row, col in proposal.footprint)),
        tuple(int(value) for value in proposal.source_indices),
        tuple(int(value) for value in proposal.source_codes),
        tuple(int(value) for value in proposal.source_component_ids),
        tuple(int(value) for value in proposal.baseline_target_component_ids),
    )


def _proposal_score_signature(proposal: Proposal) -> tuple[Any, ...]:
    """Every ranked/audited field that must agree for one topology identity."""

    return (
        v31_stable_rank_key(proposal),
        int(proposal.dynamic_reduction),
        int(proposal.component_reduction),
        float(proposal.probability_support),
        float(proposal.area_m2),
        proposal.digest,
        json.dumps(dict(proposal.evidence), sort_keys=True, separators=(",", ":")),
    )


def _discovery_order_key(proposal: Proposal) -> tuple[str, float, float]:
    """Deterministically choose an original discovery occurrence."""

    return (
        proposal.proposal_id,
        float("inf")
        if proposal.edge_distance_m is None
        else float(proposal.edge_distance_m),
        float("inf")
        if proposal.path_length_m is None
        else float(proposal.path_length_m),
    )


def _minimum_discovery_distance(values: Sequence[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return min(present) if present else None


def canonicalize_v31b_proposals(
    proposals: Sequence[Proposal],
) -> tuple[list[Proposal], Counter[str], list[dict[str, Any]]]:
    """Deduplicate exact candidates and reject accidental ID collisions.

    The proposal action, rather than a generator-specific proposal ID, is the
    semantic identity.  Keeping one stable representative makes decisions and
    the interaction audit one-to-one even if two generators discover the same
    path under different IDs.
    """

    grouped: dict[tuple[Any, ...], list[Proposal]] = {}
    duplicates: Counter[str] = Counter()
    duplicate_audit: list[dict[str, Any]] = []
    ids: dict[str, tuple[Any, ...]] = {}
    for proposal in proposals:
        key = _canonical_proposal_key(proposal)
        previous_key = ids.get(proposal.proposal_id)
        if previous_key is not None and previous_key != key:
            raise CandidateError(
                "V3.1-B proposal_id collision between non-identical proposals"
            )
        ids[proposal.proposal_id] = key
        grouped.setdefault(key, []).append(proposal)
    unique: list[Proposal] = []
    for key, group in grouped.items():
        # The proposal ID is not semantic.  Pick a stable representative so
        # reversing generator order cannot alter a canonical action or audit.
        ordered_group = sorted(group, key=_discovery_order_key)
        original_representative = ordered_group[0]
        edge_distances = tuple(
            sorted(
                (item.edge_distance_m for item in ordered_group),
                key=lambda value: (
                    value is None,
                    float("inf") if value is None else float(value),
                ),
            )
        )
        path_lengths = tuple(
            sorted(
                (item.path_length_m for item in ordered_group),
                key=lambda value: (
                    value is None,
                    float("inf") if value is None else float(value),
                ),
            )
        )
        representative = replace(
            original_representative,
            edge_distance_m=_minimum_discovery_distance(edge_distances),
            path_length_m=_minimum_discovery_distance(path_lengths),
            discovery_count=len(ordered_group),
            discovery_edge_distances_m=edge_distances,
            discovery_path_lengths_m=path_lengths,
            occurrence_edge_distance_m=original_representative.edge_distance_m,
            occurrence_path_length_m=original_representative.path_length_m,
        )
        for ordinal, proposal in enumerate(ordered_group[1:], start=1):
            if _proposal_score_signature(proposal) != _proposal_score_signature(
                original_representative
            ):
                raise CandidateError(
                    "V3.1-B duplicate proposal topology has inconsistent rank or evidence"
                )
            duplicates["duplicate_proposal"] += 1
            duplicate_audit.append(
                {
                    "occurrence_id": f"{proposal.proposal_id}:duplicate:{ordinal}",
                    "proposal_id": proposal.proposal_id,
                    "canonical_proposal_id": representative.proposal_id,
                    "decision": "rejected",
                    "reason": "duplicate_proposal",
                    "stable_rank_key": list(v31_stable_rank_key(proposal)),
                    "footprint_sha256": proposal.digest,
                    "discovery_count": 1,
                    "edge_distance_m": proposal.edge_distance_m,
                    "path_length_m": proposal.path_length_m,
                    "occurrence_edge_distance_m": proposal.edge_distance_m,
                    "occurrence_path_length_m": proposal.path_length_m,
                    "canonical_edge_distance_m": representative.edge_distance_m,
                    "canonical_path_length_m": representative.path_length_m,
                }
            )
        unique.append(representative)
    duplicate_audit.sort(key=lambda item: str(item["occurrence_id"]))
    return (
        sorted(unique, key=lambda item: (*v31_stable_rank_key(item), item.proposal_id)),
        duplicates,
        duplicate_audit,
    )
