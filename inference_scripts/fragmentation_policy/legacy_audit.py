"""Fixed, code-owned V3.1-B/V3.2 migration evidence contract.

This module intentionally does not trust the V3.3 YAML to describe legacy
behaviour.  It compares the live immutable candidate snapshots against a
checked-in contract and then checks that every required migration ledger row is
present in the draft policy.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping


V31B_SNAPSHOT_SHA256 = "19deb9656104199e9b077ad8b035d4a499daf977f00bd6a955397b05ada6e99c"
V32_SNAPSHOT_SHA256 = "70f7da2160ed79f014f0c31ce3719bd2aa48d2426c9e06684e61b1b24fb69a3a"
EVIDENCE_FILES = (
    "inference_scripts/fragmentation_v31_candidate/candidate.py",
    "inference_scripts/fragmentation_v31_candidate/v32.py",
    "scratch/v32_full_140_20260825/run_v32_from_v3.py",
    "scratch/v32_full_140_20260825/evaluate_v32_against_b.py",
    "tests/test_fragmentation_v32_candidate.py",
)

EVIDENCE_FILE_SHA256 = {
    "inference_scripts/fragmentation_v31_candidate/candidate.py": "e5df1530271c99a5f44e155477589e365ffba1449554248e609a9075c93966d4",
    "inference_scripts/fragmentation_v31_candidate/v32.py": "8aaa41d7e0fd79581d39105dd6bfc090f14d3287139d32db2ee95110146d2efb",
    "scratch/v32_full_140_20260825/run_v32_from_v3.py": "939d31e6633fa001b8d4dd0be6c32ee8c657ac8cd9a90f80a327488692a6c837",
    "scratch/v32_full_140_20260825/evaluate_v32_against_b.py": "27427120dc30d8a3335a9714cc549127c66429c7982a894f5a39c795e16cf145",
    "tests/test_fragmentation_v32_candidate.py": "a25d35678b81e0207b9573eac7b31ba9371410a5a6cb0e6d92114e437e455d7c",
}

V31B_GLOBAL_ACCEPTANCE = (
    "valid_single_label_and_invalid_pixel_preservation",
    "no_gap_overlap_outside",
    "protected_source_retention",
    "source_connectivity_preserved",
    "target_attachment_preserved",
    "per_class_components_nonincreasing",
    "global_components_nonincreasing",
    "dynamic_fragments_nonincreasing",
    "final_topology_rollback_if_any_gate_fails",
)

ACTIVE_EXECUTION_CONTRACT = {
    "bridge_edge_distance": "euclidean_cell_polygon_edge_distance_m",
    "bridge_path_length": "four_neighbour_path_length_m",
    "budget_denominator": "class_budget_mask_and_valid_frozen_baseline",
    "canonical_proposal_identity": "kind_target_index_target_code_sorted_footprint_source_indices_source_codes_source_component_ids_baseline_target_component_ids",
    "duplicate_representative": "proposal_id_then_occurrence_edge_distance_then_occurrence_path_length_none_last",
    "canonical_distance_evidence": "minimum_present_edge_distance_and_path_length",
    "inconsistent_duplicate_rank_or_evidence": "reject",
}

ACTIVE_PROPOSAL_RANK = [
    {"field": "dynamic_fragment_reduction", "order": "descending", "value_source": "proposal"},
    {"field": "component_reduction", "order": "descending", "value_source": "proposal"},
    {"field": "target_rarity_share", "order": "ascending", "value_source": "decision_engine.statistics.raw_v31_optimization_before_share"},
    {"field": "target_probability", "order": "descending", "value_source": "proposal"},
    {"field": "changed_area_m2", "order": "ascending", "value_source": "proposal"},
    {"field": "target_class_code", "order": "ascending", "value_source": "proposal"},
]

# This is the complete implementation-level migration inventory.  The YAML
# ledger is checked against it rather than being treated as its own evidence.
REQUIRED_MIGRATION_RULES = {
    "per_class_small_fragment_thresholds": "moved",
    "enclosed_island_area_caps": "superseded",
    "probability_thresholds": "superseded",
    "source_confidence_cap": "superseded",
    "semantic_compatibility_table": "superseded",
    "source_protection": "superseded",
    "target_protection": "superseded",
    "class_source_and_target_budgets": "moved",
    "protected_bridge_one_percent_budget": "moved",
    "bridge_distance_and_footprint_caps": "moved",
    "audit_proposal_limit": "superseded",
    "four_connected_frozen_single_pass": "moved",
    "global_acceptance_gates": "moved",
    "proposal_rank": "superseded",
    "dedup_and_id_collision": "moved",
    "footprint_conflict_and_core_owner": "moved",
    "b_incremental_adjudication": "moved",
    "b_plus_v32_same_adjudication": "moved",
    "multi_neighbour_rarity": "superseded",
    "contact_length": "moved",
    "lineage_140_coverage_effect_fraction": "moved",
    "manual_transport_vector_overlay": "moved",
    "exact_cross_target_tie_rejection": "superseded",
    "enclosed_single_neighbour": "superseded",
}


def _legacy_snapshot_checks() -> dict[str, Any]:
    """Read the live old candidates only when an audit is explicitly requested."""

    from inference_scripts.fragmentation_v31_candidate import (  # isolated, read-only import
        policy_snapshot,
        policy_snapshot_sha256,
        v31b_policy,
        v32_policy,
        v32_policy_snapshot,
        v32_policy_snapshot_sha256,
    )

    b_policy = v31b_policy()
    v32 = v32_policy()
    b_snapshot = policy_snapshot(b_policy)
    v32_snapshot = v32_policy_snapshot(v32)
    return {
        "v31b": {
            "expected_sha256": V31B_SNAPSHOT_SHA256,
            "actual_sha256": policy_snapshot_sha256(b_policy),
            "sha256_matches": policy_snapshot_sha256(b_policy) == V31B_SNAPSHOT_SHA256,
            "snapshot": b_snapshot,
        },
        "v32": {
            "expected_sha256": V32_SNAPSHOT_SHA256,
            "actual_sha256": v32_policy_snapshot_sha256(v32),
            "sha256_matches": v32_policy_snapshot_sha256(v32) == V32_SNAPSHOT_SHA256,
            "snapshot": v32_snapshot,
        },
    }


def audit_live_legacy(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Return fail-closed, implementation-backed V3.1-B/V3.2 migration proof."""

    live = _legacy_snapshot_checks()
    b = live["v31b"]["snapshot"]
    v32 = live["v32"]["snapshot"]
    classes = b["class_policies"]
    expected_thresholds = {code: row["dynamic_fragmentation_m2"] for code, row in classes.items()}
    expected_probability = {
        code: {key: row[key] for key in ("minimum_target_probability_mean", "maximum_current_minus_target_probability_mean", "minimum_target_probability_p10")}
        for code, row in classes.items()
    }
    expected_bridge = {
        code: {key: row[key] for key in ("allow_same_class_bridge", "bridge_max_edge_distance_m", "bridge_max_new_footprint_m2")}
        for code, row in classes.items()
    }
    expected_island_caps = {
        code: row["enclosed_island_max_m2"] for code, row in classes.items()
    }
    expected_protection = {
        code: row["ordinary_protected"] for code, row in classes.items()
    }
    reference = policy["migration"]["legacy_reference"]
    recorded_semantic = {
        str(code): sorted(int(target) for target in targets)
        for code, targets in reference["v31b_semantic_compatible_targets"].items()
        if targets
    }
    active_thresholds = {
        code: float(row["fragment_max_m2"])
        for code, row in policy["classes"].items()
    }
    active_bridge = policy["constraints"]["budgets"]["bridge_limits"]
    expected_active_bridge = {
        code: {
            "allow": "allow" if row["allow_same_class_bridge"] else "deny",
            "max_edge_distance_m": row["bridge_max_edge_distance_m"],
            "max_new_footprint_m2": row["bridge_max_new_footprint_m2"],
        }
        for code, row in expected_bridge.items()
    }
    for code in ("61", "62"):
        expected_active_bridge[code] = {
            "allow": "deny",
            "max_edge_distance_m": 0,
            "max_new_footprint_m2": 0,
        }
    active_budgets = policy["constraints"]["budgets"]
    checks = {
        "dynamic_fragmentation_thresholds": reference["v31b_dynamic_fragmentation_m2"] == expected_thresholds,
        "enclosed_island_area_caps": reference["v31b_enclosed_island_max_m2"] == expected_island_caps,
        "ordinary_protected": reference["v31b_ordinary_protected"] == expected_protection,
        "protected_source_codes": reference["v31b_protected_source_codes"] == b["protected_source_codes"],
        "probability_thresholds": reference["v31b_probability_thresholds"] == expected_probability,
        "bridge_limits": reference["v31b_bridge_limits"] == expected_bridge,
        "semantic_compatible_targets": recorded_semantic == b["semantic_compatible_targets"],
        "budgets": reference["v31b_budgets"] == {
            "source_loss_fraction": b["maximum_source_loss_fraction"],
            "target_gain_fraction": b["maximum_target_gain_fraction"],
            "protected_bridge_gain_fraction": b["protected_bridge_gain_fraction"],
        },
        "source_confidence_cap": reference["v31b_source_confidence_max"] == b["island_maximum_mean_confidence"],
        "audit_proposal_limit": reference["v31b_audit_proposal_limit"] == b["audit_proposal_limit"],
        "adjudication_mode": reference["v31b_adjudication_mode"] == b["adjudication_mode"],
        "algorithm_contract": reference["v31b_algorithm_contract"] == b["algorithm_contract"],
        "global_acceptance": reference["v31b_global_acceptance"] == list(V31B_GLOBAL_ACCEPTANCE),
        "v32_selector_contract": reference["v32_selector_and_contact"] == v32["v32_algorithm_contract"],
        "active_dynamic_fragmentation_thresholds": active_thresholds == expected_thresholds,
        "active_source_and_target_budgets": (
            active_budgets["source_loss_fraction"] == b["maximum_source_loss_fraction"]
            and active_budgets["target_gain_fraction"] == b["maximum_target_gain_fraction"]
            and active_budgets["protected_bridge_gain_fraction"] == b["protected_bridge_gain_fraction"]
        ),
        "active_bridge_limits": active_bridge == expected_active_bridge,
        "active_execution_contract": policy["constraints"]["execution_contract"] == ACTIVE_EXECUTION_CONTRACT,
        "active_proposal_rank": (
            policy["decision_engine"]["proposal_adjudication"]["proposal_rank"] == ACTIVE_PROPOSAL_RANK
            and policy["decision_engine"]["proposal_adjudication"]["exact_tie_break"]
            == ["proposal_digest_ascending", "proposal_id_ascending"]
        ),
    }
    ledger = policy["migration"]["rules"]
    ledger_checks = {
        rule: rule in ledger and ledger[rule]["status"] == status
        for rule, status in REQUIRED_MIGRATION_RULES.items()
    }
    repository_root = Path(__file__).resolve().parents[2]
    source_markers = {
        "inference_scripts/fragmentation_v31_candidate/candidate.py": (
            "def _rank_key(", "def _canonicalize_v31b_proposals(",
            "def _adjudicate_v31b(", 'reason = "outside_core_owner"',
            'reason = "footprint_conflict"',
        ),
        "inference_scripts/fragmentation_v31_candidate/v32.py": (
            "one_B_incremental_adjudication_over_B_original_and_V32_new_proposals",
            "audit_only_not_filter_or_sort",
        ),
        "scratch/v32_full_140_20260825/run_v32_from_v3.py": (
            "frozen_global_class_pixel_totals", "full_audit=True",
        ),
        "scratch/v32_full_140_20260825/evaluate_v32_against_b.py": (
            "EFFECT_FRACTION = 0.005", "REAL_PARTITION_COUNT",
            "same_partition_ids_and_raw_v3_valid_sha256",
        ),
        "tests/test_fragmentation_v32_candidate.py": ("contact", "frozen"),
    }
    implementation_checks: dict[str, bool] = {}
    implementation_sha256: dict[str, dict[str, Any]] = {}
    for relative, markers in source_markers.items():
        path = repository_root / relative
        payload = path.read_bytes() if path.is_file() else b""
        body = payload.decode("utf-8") if payload else ""
        implementation_checks[relative] = bool(body) and all(marker in body for marker in markers)
        actual_sha = sha256(payload).hexdigest() if payload else None
        expected_sha = EVIDENCE_FILE_SHA256[relative]
        implementation_sha256[relative] = {
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "sha256_matches": actual_sha == expected_sha,
        }
    evidence_matches = list(policy["migration"]["evidence_files"]) == list(EVIDENCE_FILES)
    return {
        "evidence_files": list(EVIDENCE_FILES),
        "live_snapshots": live,
        "field_checks": checks,
        "migration_rule_checks": ledger_checks,
        "implementation_contract_checks": implementation_checks,
        "implementation_file_sha256": implementation_sha256,
        "evidence_files_match_fixed_inventory": evidence_matches,
        "all_verified": (
            all(checks.values())
            and all(ledger_checks.values())
            and all(implementation_checks.values())
            and all(item["sha256_matches"] for item in implementation_sha256.values())
            and evidence_matches
            and all(item["sha256_matches"] for item in live.values())
        ),
    }
