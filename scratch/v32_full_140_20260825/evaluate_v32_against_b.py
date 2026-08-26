#!/usr/bin/env python3
"""Run the complete global V3.1-B versus V3.2 acceptance comparison."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "inference_scripts"))
from deployment_config import CLASS_ORDER  # noqa: E402

EVALUATOR_PATH = (
    REPO_ROOT / "scratch" / "v31a_full_140_20260824"
    / "evaluate_global_fragmentation.py"
)
SPEC = importlib.util.spec_from_file_location("_v32_global_evaluator", EVALUATOR_PATH)
assert SPEC and SPEC.loader
EVALUATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EVALUATOR
SPEC.loader.exec_module(EVALUATOR)

SCHEMA_VERSION = 1
REAL_PARTITION_COUNT = 140
REAL_STRICT_VALID_TOTAL = 831_531_565
EFFECT_FRACTION = 0.005


class V32EvaluationError(RuntimeError):
    pass


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_json(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V32EvaluationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V32EvaluationError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _verified_manifest(path: Path, label: str) -> dict[str, Any]:
    value = _read(path)
    declared = value.get("manifest_sha256")
    body = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if not isinstance(declared, str) or declared != _sha_json(body):
        raise V32EvaluationError(f"{label} manifest self SHA-256 mismatch")
    if value.get("status") != "complete":
        raise V32EvaluationError(f"{label} manifest is not complete")
    if value.get("completed_partition_count") != REAL_PARTITION_COUNT:
        raise V32EvaluationError(f"{label} manifest is not a 140-Core result")
    return value


def _parts(manifest: Mapping[str, Any], label: str) -> dict[str, Mapping[str, Any]]:
    raw = manifest.get("partitions")
    if not isinstance(raw, list):
        raise V32EvaluationError(f"{label} partitions are missing")
    result = {str(item.get("partition_id", "")): item for item in raw if isinstance(item, dict)}
    if len(result) != REAL_PARTITION_COUNT or "" in result:
        raise V32EvaluationError(f"{label} partition identities are incomplete")
    return result


def _lineage_gate(
    b_manifest: Mapping[str, Any], v32_manifest: Mapping[str, Any]
) -> tuple[dict[str, bool], dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    b_parts = _parts(b_manifest, "B")
    v32_parts = _parts(v32_manifest, "V3.2")
    fixed_inputs_equal = set(b_parts) == set(v32_parts)
    if fixed_inputs_equal:
        for partition_id in b_parts:
            for key in ("raw", "v3", "valid"):
                left = ((b_parts[partition_id].get("outputs") or {}).get(key) or {}).get("sha256")
                right = ((v32_parts[partition_id].get("outputs") or {}).get(key) or {}).get("sha256")
                if not isinstance(left, str) or left != right:
                    fixed_inputs_equal = False
                    break
            if not fixed_inputs_equal:
                break
    gates = {
        "same_v31a_parent": b_manifest.get("parent_v31a_manifest_sha256")
        == v32_manifest.get("parent_v31a_manifest_sha256"),
        "same_snapshot": b_manifest.get("snapshot_manifest_sha256")
        == v32_manifest.get("snapshot_manifest_sha256"),
        "same_v3_policy": b_manifest.get("v3_policy_snapshot_sha256")
        == v32_manifest.get("v3_policy_snapshot_sha256"),
        "same_partition_ids_and_raw_v3_valid_sha256": fixed_inputs_equal,
        "v32_frozen_total_exact": int(v32_manifest.get("frozen_global_class_pixel_totals_sum", -1))
        == REAL_STRICT_VALID_TOTAL,
    }
    return gates, b_parts, v32_parts


def _method_summary(method: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "components_total": int(method["components_total"]),
        "dynamic_fragment_count": int(method["dynamic_fragments"]["count"]),
        "dynamic_fragment_area_m2": float(method["dynamic_fragments"]["area_m2"]),
        "boundary_edges": int(method["boundary"]["total_cross_class_boundary"]["edges"]),
        "boundary_metres": float(method["boundary"]["total_cross_class_boundary"]["metres"]),
        "internal_boundary_edges": int(method["boundary"]["internal_cross_class_boundary"]["edges"]),
        "internal_boundary_metres": float(method["boundary"]["internal_cross_class_boundary"]["metres"]),
    }


def _coverage_gate(result: Mapping[str, Any]) -> dict[str, bool]:
    coverage = result["coverage"]
    invalid = coverage["invalid_label_inside_valid_pixels"]
    outside = coverage["outside_valid_label_pixels"]
    return {
        "part_count_140": int(result["part_count"]) == REAL_PARTITION_COUNT,
        "core_overlap_zero": int(coverage["core_overlap_pixels"]) == 0,
        "geometric_gap_zero": int(coverage["geometric_coverage_gap_pixels"]) == 0,
        "candidate_invalid_inside_zero": int(invalid["v31"]) == 0,
        "candidate_outside_valid_zero": int(outside["v31"]) == 0,
    }


def _per_class_components(method: Mapping[str, Any]) -> dict[int, int]:
    present = {int(item["class_code"]): int(item["components"]) for item in method["per_class"]}
    return {int(code): present.get(int(code), 0) for code in CLASS_ORDER}


def _candidate_audit_summary(
    parts: Mapping[str, Mapping[str, Any]], totals_sha: str,
) -> dict[str, Any]:
    per_class: dict[int, Counter[str]] = {int(code): Counter() for code in CLASS_ORDER}
    multi_generated = 0
    multi_accepted = 0
    changed_pixels = 0
    protected_loss = 0
    full_audit = True
    rejection_counts: Counter[str] = Counter()
    for partition_id in sorted(parts):
        reference = parts[partition_id].get("audit") or {}
        path = Path(str(reference.get("path", "")))
        if not path.is_file() or _sha_file(path) != reference.get("sha256"):
            raise V32EvaluationError(f"{partition_id}: V3.2 audit SHA-256 mismatch")
        stage = _read(path)
        audit = stage.get("v32_audit") or {}
        if stage.get("candidate_label") != "V3.2" or not audit.get("full_audit") or audit.get("audit_truncated"):
            full_audit = False
        if audit.get("frozen_global_class_pixel_totals_sha256") != totals_sha:
            raise V32EvaluationError(f"{partition_id}: frozen totals SHA-256 differs")
        protected_loss += int(audit.get("protected_source_loss_pixel_count", -1))
        changed_pixels += int(audit.get("changed_pixel_count", 0))
        multi_generated += int(audit.get("closed_multi_neighbor_rarity_proposals_generated", 0))
        multi_accepted += sum(
            1 for item in (audit.get("accepted") or [])
            if item.get("kind") == "closed_multi_neighbor_rarity"
        )
        rejection_counts.update({str(key): int(value) for key, value in (audit.get("proposal_generation_reject_reason_counts") or {}).items()})
        for code, item in (audit.get("per_class") or {}).items():
            numeric = int(code)
            per_class[numeric].update({
                "source_loss": int(item.get("source_loss", 0)),
                "target_gain": int(item.get("target_gain", 0)),
                "net_pixel_drift": int(item.get("net_pixel_drift", 0)),
            })
    return {
        "full_audit_complete": full_audit,
        "protected_source_loss_pixel_count": protected_loss,
        "changed_pixel_count_sum_of_core_audits": changed_pixels,
        "closed_multi_neighbor_rarity_proposals_generated": multi_generated,
        "closed_multi_neighbor_rarity_proposals_accepted": multi_accepted,
        "generation_rejection_counts": dict(sorted(rejection_counts.items())),
        "per_class": {str(code): dict(sorted(values.items())) for code, values in per_class.items()},
    }


def _direct_transitions(
    b_parts: Mapping[str, Mapping[str, Any]],
    v32_parts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    pairs: Counter[tuple[int, int]] = Counter()
    changed_pixels = 0
    changed_area_m2 = 0.0
    for partition_id in sorted(b_parts):
        b_part, v_part = b_parts[partition_id], v32_parts[partition_id]
        b_path = Path(str(b_part["outputs"]["v31a"]["path"]))
        v_path = Path(str(v_part["outputs"]["v31a"]["path"]))
        valid_path = Path(str(v_part["outputs"]["valid"]["path"]))
        b_values = np.load(b_path, mmap_mode="r", allow_pickle=False)
        v_values = np.load(v_path, mmap_mode="r", allow_pickle=False)
        valid = np.load(valid_path, mmap_mode="r", allow_pickle=False)
        if b_values.shape != v_values.shape or b_values.shape != valid.shape:
            raise V32EvaluationError(f"{partition_id}: B/V3.2 direct transition shapes differ")
        left = b_values[valid].astype(np.int64, copy=False)
        right = v_values[valid].astype(np.int64, copy=False)
        encoded = (left.astype(np.uint64) << np.uint64(32)) | right.astype(np.uint64)
        values, counts = np.unique(encoded, return_counts=True)
        for value, count in zip(values, counts):
            pairs[(int(value >> np.uint64(32)), int(value & np.uint64(0xffffffff)))] += int(count)
        changed = int(np.count_nonzero(left != right))
        changed_pixels += changed
        changed_area_m2 += changed * float(v_part["physical_metrics"]["pixel_area_m2"])
    return {
        "from_method": "V3.1-B",
        "to_method": "V3.2",
        "changed_pixels": changed_pixels,
        "changed_area_m2": changed_area_m2,
        "transitions": [
            {"from_class": source, "to_class": target, "pixels": count}
            for (source, target), count in sorted(pairs.items())
        ],
    }


def evaluate(
    b_manifest_path: Path, v32_manifest_path: Path, output_root: Path,
) -> dict[str, Any]:
    b_manifest_path = b_manifest_path.resolve()
    v32_manifest_path = v32_manifest_path.resolve()
    output_root = output_root.resolve()
    b_manifest = _verified_manifest(b_manifest_path, "B")
    v32_manifest = _verified_manifest(v32_manifest_path, "V3.2")
    if b_manifest.get("candidate_label") != "B" or v32_manifest.get("candidate_label") != "V3.2":
        raise V32EvaluationError("candidate labels do not identify B and V3.2")
    lineage, b_parts, v32_parts = _lineage_gate(b_manifest, v32_manifest)
    if not all(lineage.values()):
        raise V32EvaluationError(f"B/V3.2 lineage mismatch: {lineage}")

    b_evaluation = EVALUATOR.evaluate(
        b_manifest_path, output_root / "baseline_b_global_evaluation", resume=True,
    )
    v32_evaluation = EVALUATOR.evaluate(
        v32_manifest_path, output_root / "global_evaluation", resume=True,
    )
    b_method = b_evaluation["result"]["methods"]["v31"]
    v32_method = v32_evaluation["result"]["methods"]["v31"]
    b_summary, v32_summary = _method_summary(b_method), _method_summary(v32_method)
    b_by_class = _per_class_components(b_method)
    v32_by_class = _per_class_components(v32_method)
    coverage = _coverage_gate(v32_evaluation["result"])
    totals = v32_manifest["frozen_global_class_pixel_totals"]
    totals_sha = _sha_json({str(code): int(totals[str(code)]) for code in sorted(CLASS_ORDER)})
    candidate_audits = _candidate_audit_summary(v32_parts, totals_sha)
    dynamic_reduction = b_summary["dynamic_fragment_count"] - v32_summary["dynamic_fragment_count"]
    required_reduction = int(math.ceil(b_summary["dynamic_fragment_count"] * EFFECT_FRACTION))
    safety = {
        **coverage,
        "components_nonincreasing": v32_summary["components_total"] <= b_summary["components_total"],
        "per_class_components_nonincreasing": all(v32_by_class[code] <= b_by_class[code] for code in b_by_class),
        "dynamic_count_nonincreasing": v32_summary["dynamic_fragment_count"] <= b_summary["dynamic_fragment_count"],
        "dynamic_area_nonincreasing": v32_summary["dynamic_fragment_area_m2"] <= b_summary["dynamic_fragment_area_m2"] + 1e-7,
        "boundary_edges_nonincreasing": v32_summary["boundary_edges"] <= b_summary["boundary_edges"],
        "boundary_metres_nonincreasing": v32_summary["boundary_metres"] <= b_summary["boundary_metres"] + 1e-7,
        "internal_boundary_edges_nonincreasing": v32_summary["internal_boundary_edges"] <= b_summary["internal_boundary_edges"],
        "internal_boundary_metres_nonincreasing": v32_summary["internal_boundary_metres"] <= b_summary["internal_boundary_metres"] + 1e-7,
        "protected_source_loss_zero": candidate_audits["protected_source_loss_pixel_count"] == 0,
        "full_candidate_audit_complete": bool(candidate_audits["full_audit_complete"]),
    }
    effect = {
        "fraction": EFFECT_FRACTION,
        "required_dynamic_fragment_reduction": required_reduction,
        "actual_dynamic_fragment_reduction": dynamic_reduction,
        "pass": dynamic_reduction >= required_reduction,
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "accepted" if all(safety.values()) and effect["pass"] else "rejected_validation",
        "lineage_gates": lineage,
        "safety_gates": safety,
        "effect_gate": effect,
        "validation_pass": all(safety.values()),
        "acceptance_pass": all(safety.values()) and effect["pass"],
        "baseline_b": b_summary,
        "candidate_v32": v32_summary,
        "v32_minus_b": {key: v32_summary[key] - b_summary[key] for key in b_summary},
        "per_class_component_delta_v32_minus_b": {str(code): v32_by_class[code] - b_by_class[code] for code in b_by_class},
        "candidate_audit_summary": candidate_audits,
        "direct_b_to_v32": _direct_transitions(b_parts, v32_parts),
        "inputs": {
            "b_manifest": str(b_manifest_path),
            "b_manifest_sha256": _sha_file(b_manifest_path),
            "v32_manifest": str(v32_manifest_path),
            "v32_manifest_sha256": _sha_file(v32_manifest_path),
            "evaluator": str(EVALUATOR_PATH.resolve()),
            "evaluator_sha256": _sha_file(EVALUATOR_PATH),
            "comparison_code_sha256": _sha_file(Path(__file__).resolve()),
        },
        "outputs": {
            "baseline_b_global_evaluation": str((output_root / "baseline_b_global_evaluation" / "global_fragmentation.json").resolve()),
            "candidate_v32_global_evaluation": str((output_root / "global_evaluation" / "global_fragmentation.json").resolve()),
        },
    }
    result["result_sha256"] = _sha_json(result)
    _atomic_json(output_root / "V32_V31B_COMPARISON.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-manifest", type=Path, required=True)
    parser.add_argument("--v32-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = evaluate(args.b_manifest, args.v32_manifest, args.output_root)
    except (V32EvaluationError, EVALUATOR.EvaluationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps({
        "status": result["status"],
        "validation_pass": result["validation_pass"],
        "acceptance_pass": result["acceptance_pass"],
        "dynamic_fragment_reduction": result["effect_gate"]["actual_dynamic_fragment_reduction"],
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
