"""Versioned, isolated policy documents for fragmentation experiments.

This package is deliberately configuration and audit tooling only.  It does
not import the V3/V3.1/V3.2 execution paths and cannot change a raster.
"""

from .loader import (
    PolicyError,
    audit_legacy_migration,
    explain_fragment_decision,
    load_policy,
    policy_sha256,
    policy_snapshot,
    rank_conflicting_proposals,
)

__all__ = [
    "PolicyError",
    "audit_legacy_migration",
    "explain_fragment_decision",
    "load_policy",
    "policy_sha256",
    "policy_snapshot",
    "rank_conflicting_proposals",
]
