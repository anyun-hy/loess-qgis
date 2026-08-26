"""Small read-only CLI for inspecting the isolated V3.3 policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .loader import audit_legacy_migration, diff_policies, explain_fragment_decision, load_policy, policy_sha256, rank_conflicting_proposals


def _json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(prog="fragmentation-policy")
    parser.add_argument("--policy", type=Path, default=None, help="V3.3 policy YAML")
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser("show")
    show.add_argument("--class", dest="class_code", required=True)
    explain = commands.add_parser("explain")
    explain.add_argument("--source", required=True)
    explain.add_argument("--surrounding", required=True, help="comma-separated class codes")
    explain.add_argument("--area-m2", type=float, required=True)
    explain.add_argument("--complete-source-component", action="store_true", help="confirm the whole small source component is being evaluated")
    explain.add_argument("--complete-boundary-evidence", action="store_true", help="confirm surrounding-class evidence covers the complete source boundary")
    explain.add_argument("--boundary-status", choices=("closed_valid", "external_contact", "invalid_pixel_contact", "incomplete_or_ambiguous"), required=True)
    explain.add_argument("--bridge-candidates-checked", action="store_true", help="confirm no eligible bridge exists before selecting multi-neighbour")
    explain.add_argument("--bridge-target", help="bridge-only selected target class code")
    explain.add_argument("--target-probabilities", default="", help="optional 13=0.2,43=0.8 target probabilities")
    explain.add_argument("--source-component-id", type=int)
    explain.add_argument("--target-component-ids", default="", help="bridge-only comma-separated target component IDs")
    explain.add_argument("--edge-distance-m", type=float)
    explain.add_argument("--bridge-footprint-m2", type=float)
    explain.add_argument("--scenario", choices=("unique_enclosure", "multi_neighbour", "same_class_bridge", "invalid_or_ambiguous"), required=True)
    diff = commands.add_parser("diff")
    diff.add_argument("--against", required=True, type=Path)
    rank = commands.add_parser("rank")
    rank.add_argument("--proposals-json", required=True, type=Path, help="JSON array of fully eligible canonical proposals")
    commands.add_parser("audit-migration")
    args = parser.parse_args()
    policy = load_policy(args.policy)
    if args.command == "show":
        row = policy["classes"].get(str(args.class_code))
        if row is None:
            parser.error("unknown class code")
        code = str(args.class_code)
        _json({
            "policy_id": policy["policy"]["id"],
            "policy_sha256": policy_sha256(policy),
            "class": row,
            "bridge_limits": policy["constraints"]["budgets"]["bridge_limits"][code],
            "raw_v31_statistics": {
                "area_km2": policy["decision_engine"]["statistics"]["raw_v31_optimization_before_area_km2"][code],
                "share": policy["decision_engine"]["statistics"]["raw_v31_optimization_before_share"][code],
            },
            "conflict_resolution": policy["decision_engine"]["conflict_resolution"],
            "scenario_routing": policy["decision_engine"]["scenario_routing"],
            "execution_contract": policy["constraints"]["execution_contract"],
            "relation_rules": [
                rule for rule in policy["decision_engine"]["relation_rules"]
                if rule["source"] in ("*", code) or rule["target"] in ("*", code)
            ],
        })
    elif args.command == "explain":
        probabilities = {key: float(value) for key, value in (item.split("=", 1) for item in args.target_probabilities.split(",") if item)}
        target_component_ids = [int(value) for value in args.target_component_ids.split(",") if value]
        _json(explain_fragment_decision(
            policy,
            source_code=args.source,
            surrounding_codes=args.surrounding.split(","),
            area_m2=args.area_m2,
            scenario=args.scenario,
            target_probabilities=probabilities,
            source_component_complete=args.complete_source_component,
            boundary_evidence_complete=args.complete_boundary_evidence,
            boundary_status=args.boundary_status,
            bridge_candidates_checked=args.bridge_candidates_checked,
            bridge_target_code=args.bridge_target,
            source_component_id=args.source_component_id,
            target_component_ids=target_component_ids or None,
            edge_distance_m=args.edge_distance_m,
            bridge_footprint_m2=args.bridge_footprint_m2,
        ))
    elif args.command == "diff":
        _json(diff_policies(load_policy(args.against), policy))
    elif args.command == "rank":
        proposals = json.loads(args.proposals_json.read_text(encoding="utf-8"))
        if not isinstance(proposals, list):
            parser.error("--proposals-json must contain a JSON array")
        _json(rank_conflicting_proposals(policy, proposals))
    else:
        _json(audit_legacy_migration(policy))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
