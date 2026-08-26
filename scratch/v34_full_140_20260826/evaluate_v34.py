#!/usr/bin/env python3
"""Evaluate one bounded V3.4 pass against its exact V3.3 parent."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "inference_scripts"))
from deployment_config import CLASS_ORDER  # noqa: E402

GLOBAL_EVALUATOR_PATH = REPO_ROOT / "scratch" / "v31a_full_140_20260824" / "evaluate_global_fragmentation.py"
SPEC = importlib.util.spec_from_file_location("_v34_global_evaluator", GLOBAL_EVALUATOR_PATH)
assert SPEC and SPEC.loader
GLOBAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GLOBAL
SPEC.loader.exec_module(GLOBAL)

SCHEMA_VERSION = 1
REAL_PARTITION_COUNT = 140
REAL_STRICT_VALID_TOTAL = 831_531_565


class V34EvaluationError(RuntimeError):
    pass


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V34EvaluationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V34EvaluationError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _manifest(path: Path, label: str, expected_label: str) -> dict[str, Any]:
    value = _read(path)
    if value.get("manifest_sha256") != _sha_json({key: item for key, item in value.items() if key != "manifest_sha256"}):
        raise V34EvaluationError(f"{label} manifest self SHA mismatch")
    if value.get("status") != "complete" or value.get("candidate_label") != expected_label:
        raise V34EvaluationError(f"{label} manifest is not complete {expected_label}")
    if int(value.get("completed_partition_count", -1)) != REAL_PARTITION_COUNT:
        raise V34EvaluationError(f"{label} does not contain {REAL_PARTITION_COUNT} Cores")
    return value


def _parts(manifest: Mapping[str, Any], label: str) -> dict[str, Mapping[str, Any]]:
    raw = manifest.get("partitions")
    if not isinstance(raw, list):
        raise V34EvaluationError(f"{label} partitions missing")
    result = {str(item.get("partition_id", "")): item for item in raw if isinstance(item, dict)}
    if len(result) != REAL_PARTITION_COUNT or "" in result:
        raise V34EvaluationError(f"{label} partition identities incomplete")
    return result


def _method(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "components_total": int(value["components_total"]),
        "dynamic_fragment_count": int(value["dynamic_fragments"]["count"]),
        "dynamic_fragment_area_m2": float(value["dynamic_fragments"]["area_m2"]),
        "boundary_edges": int(value["boundary"]["total_cross_class_boundary"]["edges"]),
        "boundary_metres": float(value["boundary"]["total_cross_class_boundary"]["metres"]),
        "internal_boundary_edges": int(value["boundary"]["internal_cross_class_boundary"]["edges"]),
        "internal_boundary_metres": float(value["boundary"]["internal_cross_class_boundary"]["metres"]),
    }


def _per_class(value: Mapping[str, Any]) -> dict[int, int]:
    present = {int(item["class_code"]): int(item["components"]) for item in value["per_class"]}
    return {int(code): present.get(int(code), 0) for code in CLASS_ORDER}


def _audit(
    v33_parts: Mapping[str, Mapping[str, Any]],
    v34_parts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    counts = Counter()
    rejects = Counter()
    full = True
    budgets_pass = True
    parent_audit_match = True
    for partition_id in sorted(v34_parts):
        part = v34_parts[partition_id]
        reference = part.get("audit") or {}
        path = Path(str(reference.get("path", "")))
        if not path.is_file() or _sha_file(path) != reference.get("sha256"):
            raise V34EvaluationError(f"{partition_id}: V3.4 audit SHA mismatch")
        stage = _read(path)
        if stage.get("parent_v33_partition_audit") != v33_parts[partition_id].get("audit"):
            parent_audit_match = False
        audit = stage.get("v34_audit") or {}
        if not audit.get("full_audit") or audit.get("audit_truncated"):
            full = False
        for key in (
            "changed_pixel_count", "protected_source_loss_pixel_count",
            "gap_pixels", "overlap_pixels", "outside_pixels",
            "raw_generated", "proposals_canonical", "duplicate_proposal_count",
            "proposals_accepted",
        ):
            counts[key] += int(audit.get(key, 0))
        rejects.update({str(key): int(value) for key, value in (audit.get("proposal_reject_reason_counts") or {}).items()})
        rows = audit.get("cumulative_class_budget_pixels") or {}
        if set(rows) != {str(code) for code in CLASS_ORDER}:
            empty_owner_core = (
                int((stage.get("coverage") or {}).get("core_strict_valid_pixel_count", -1)) == 0
                and all(int(audit.get(key, 0)) == 0 for key in (
                    "changed_pixel_count", "raw_generated", "proposals_canonical",
                    "proposals_accepted", "protected_source_loss_pixel_count",
                    "gap_pixels", "overlap_pixels", "outside_pixels",
                ))
            )
            budgets_pass &= empty_owner_core
            continue
        for code in CLASS_ORDER:
            row = rows[str(code)]
            budgets_pass &= float(row["cumulative_source_loss"]) <= float(row["source_loss_limit"]) + 1e-9
            budgets_pass &= float(row["cumulative_target_gain"]) <= float(row["target_gain_limit"]) + 1e-9
            budgets_pass &= float(row["cumulative_protected_bridge_gain"]) <= float(row["protected_bridge_gain_limit"]) + 1e-9 if float(row["protected_bridge_gain_limit"]) > 0 else int(row["cumulative_protected_bridge_gain"]) == 0
    return {
        **dict(counts),
        "full_audit_complete": full,
        "cumulative_budgets_pass": bool(budgets_pass),
        "parent_partition_audit_match": parent_audit_match,
        "proposal_rejection_counts": dict(sorted(rejects.items())),
    }


def _transition(
    v33_parts: Mapping[str, Mapping[str, Any]],
    v34_parts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    pairs: Counter[tuple[int, int]] = Counter()
    changed = 0
    for partition_id in sorted(v34_parts):
        left = np.load(v33_parts[partition_id]["outputs"]["v33"]["path"], mmap_mode="r", allow_pickle=False)
        right = np.load(v34_parts[partition_id]["outputs"]["v34"]["path"], mmap_mode="r", allow_pickle=False)
        valid = np.load(v34_parts[partition_id]["outputs"]["valid"]["path"], mmap_mode="r", allow_pickle=False)
        lvalues = left[valid].astype(np.int64, copy=False)
        rvalues = right[valid].astype(np.int64, copy=False)
        encoded = (lvalues.astype(np.uint64) << np.uint64(32)) | rvalues.astype(np.uint64)
        values, counts = np.unique(encoded, return_counts=True)
        for value, count in zip(values, counts):
            pairs[(int(value >> np.uint64(32)), int(value & np.uint64(0xFFFFFFFF)))] += int(count)
        changed += int(np.count_nonzero(lvalues != rvalues))
    return {
        "changed_pixels": changed,
        "transitions": [
            {"from_class": source, "to_class": target, "pixels": count}
            for (source, target), count in sorted(pairs.items()) if source != target
        ],
    }


def evaluate(
    v33_manifest_path: Path, v34_manifest_path: Path, output_root: Path,
) -> dict[str, Any]:
    v33_manifest_path = v33_manifest_path.resolve()
    v34_manifest_path = v34_manifest_path.resolve()
    output_root = output_root.resolve()
    v33 = _manifest(v33_manifest_path, "V3.3", "V3.3")
    v34 = _manifest(v34_manifest_path, "V3.4", "V3.4")
    v33_parts = _parts(v33, "V3.3")
    v34_parts = _parts(v34, "V3.4")
    same_inputs = set(v33_parts) == set(v34_parts)
    if same_inputs:
        for partition_id in v33_parts:
            for key in ("raw", "v3", "valid"):
                if v33_parts[partition_id]["outputs"][key]["sha256"] != v34_parts[partition_id]["outputs"][key]["sha256"]:
                    same_inputs = False
                    break
            if not same_inputs:
                break
    lineage = {
        "exact_parent_v33_manifest": v34.get("parent_v33_manifest_sha256") == _sha_file(v33_manifest_path),
        "same_v31a_parent": v34.get("parent_v31a_manifest_sha256") == v33.get("parent_v31a_manifest_sha256"),
        "same_snapshot": v34.get("snapshot_manifest_sha256") == v33.get("snapshot_manifest_sha256"),
        "same_v3_policy": v34.get("v3_policy_snapshot_sha256") == v33.get("v3_policy_snapshot_sha256"),
        "same_partition_ids_and_raw_v3_valid": same_inputs,
        "strict_valid_total_exact": int(v34.get("frozen_v3_class_pixel_totals_sum", -1)) == REAL_STRICT_VALID_TOTAL,
    }
    if not all(lineage.values()):
        raise V34EvaluationError(f"V3.3/V3.4 lineage mismatch: {lineage}")
    v33_eval = GLOBAL.evaluate(v33_manifest_path, output_root / "global_v33", resume=True)
    v34_eval = GLOBAL.evaluate(v34_manifest_path, output_root / "global_v34", resume=True)
    v33_method_raw = v33_eval["result"]["methods"]["v31"]
    v34_method_raw = v34_eval["result"]["methods"]["v31"]
    v33_method = _method(v33_method_raw)
    v34_method = _method(v34_method_raw)
    v33_by_class = _per_class(v33_method_raw)
    v34_by_class = _per_class(v34_method_raw)
    coverage = v34_eval["result"]["coverage"]
    audits = _audit(v33_parts, v34_parts)
    safety = {
        "part_count_140": int(v34_eval["result"]["part_count"]) == REAL_PARTITION_COUNT,
        "core_overlap_zero": int(coverage["core_overlap_pixels"]) == 0,
        "geometric_gap_zero": int(coverage["geometric_coverage_gap_pixels"]) == 0,
        "candidate_invalid_inside_zero": int(coverage["invalid_label_inside_valid_pixels"]["v31"]) == 0,
        "candidate_outside_valid_zero": int(coverage["outside_valid_label_pixels"]["v31"]) == 0,
        "components_nonincreasing": v34_method["components_total"] <= v33_method["components_total"],
        "per_class_components_nonincreasing": all(v34_by_class[code] <= v33_by_class[code] for code in v33_by_class),
        "dynamic_count_nonincreasing": v34_method["dynamic_fragment_count"] <= v33_method["dynamic_fragment_count"],
        "dynamic_area_nonincreasing": v34_method["dynamic_fragment_area_m2"] <= v33_method["dynamic_fragment_area_m2"] + 1e-7,
        "boundary_edges_nonincreasing": v34_method["boundary_edges"] <= v33_method["boundary_edges"],
        "boundary_metres_nonincreasing": v34_method["boundary_metres"] <= v33_method["boundary_metres"] + 1e-7,
        "internal_boundary_edges_nonincreasing": v34_method["internal_boundary_edges"] <= v33_method["internal_boundary_edges"],
        "internal_boundary_metres_nonincreasing": v34_method["internal_boundary_metres"] <= v33_method["internal_boundary_metres"] + 1e-7,
        "protected_source_loss_zero": int(audits["protected_source_loss_pixel_count"]) == 0,
        "gap_overlap_outside_zero": all(int(audits[key]) == 0 for key in ("gap_pixels", "overlap_pixels", "outside_pixels")),
        "full_audit_complete": bool(audits["full_audit_complete"]),
        "cumulative_budgets_pass": bool(audits["cumulative_budgets_pass"]),
        "parent_partition_audit_match": bool(audits["parent_partition_audit_match"]),
    }
    reduction = v33_method["dynamic_fragment_count"] - v34_method["dynamic_fragment_count"]
    validation_pass = all(safety.values())
    effect_pass = reduction > 0
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "improved" if validation_pass and effect_pass else ("safe_no_effect" if validation_pass else "rejected_validation"),
        "lineage_gates": lineage,
        "safety_gates": safety,
        "validation_pass": validation_pass,
        "effect_pass": effect_pass,
        "effect": {"dynamic_fragment_reduction": reduction, "required_reduction": 1, "pass": effect_pass},
        "methods": {"V3.3": v33_method, "V3.4": v34_method},
        "deltas": {key: v34_method[key] - v33_method[key] for key in v33_method},
        "per_class_component_delta_v34_minus_v33": {str(code): v34_by_class[code] - v33_by_class[code] for code in v33_by_class},
        "candidate_audit_summary": audits,
        "direct_transition": _transition(v33_parts, v34_parts),
        "inputs": {
            "v33_manifest": str(v33_manifest_path), "v33_manifest_sha256": _sha_file(v33_manifest_path),
            "v34_manifest": str(v34_manifest_path), "v34_manifest_sha256": _sha_file(v34_manifest_path),
            "comparison_code_sha256": _sha_file(Path(__file__).resolve()),
        },
    }
    result["result_sha256"] = _sha_json(result)
    _atomic_json(output_root / "V34_V33_COMPARISON.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v33-manifest", type=Path, required=True)
    parser.add_argument("--v34-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = evaluate(args.v33_manifest, args.v34_manifest, args.output_root)
    except (V34EvaluationError, GLOBAL.EvaluationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps({"status": result["status"], "validation_pass": result["validation_pass"], "effect_pass": result["effect_pass"], "dynamic_fragment_reduction": result["effect"]["dynamic_fragment_reduction"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
