#!/usr/bin/env python3
"""Evaluate V3.3 against V3, V3.1-B, and V3.2 on the exact 140-Core domain."""

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

GLOBAL_EVALUATOR_PATH = (
    REPO_ROOT / "scratch" / "v31a_full_140_20260824"
    / "evaluate_global_fragmentation.py"
)
SPEC = importlib.util.spec_from_file_location("_v33_global_evaluator", GLOBAL_EVALUATOR_PATH)
assert SPEC and SPEC.loader
GLOBAL_EVALUATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GLOBAL_EVALUATOR
SPEC.loader.exec_module(GLOBAL_EVALUATOR)

SCHEMA_VERSION = 1
REAL_PARTITION_COUNT = 140
REAL_STRICT_VALID_TOTAL = 831_531_565
EFFECT_FRACTION = 0.005


class V33EvaluationError(RuntimeError):
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
        raise V33EvaluationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V33EvaluationError(f"JSON root must be an object: {path}")
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


def _manifest(path: Path, label: str, candidate_label: str) -> dict[str, Any]:
    value = _read(path)
    declared = value.get("manifest_sha256")
    if not isinstance(declared, str) or declared != _sha_json({key: item for key, item in value.items() if key != "manifest_sha256"}):
        raise V33EvaluationError(f"{label} manifest self SHA-256 mismatch")
    if value.get("status") != "complete" or value.get("completed_partition_count") != REAL_PARTITION_COUNT:
        raise V33EvaluationError(f"{label} manifest is not a complete {REAL_PARTITION_COUNT}-Core result")
    if value.get("candidate_label") != candidate_label:
        raise V33EvaluationError(f"{label} candidate label is not {candidate_label}")
    return value


def _parts(manifest: Mapping[str, Any], label: str) -> dict[str, Mapping[str, Any]]:
    raw = manifest.get("partitions")
    if not isinstance(raw, list):
        raise V33EvaluationError(f"{label} partitions are missing")
    result = {str(item.get("partition_id", "")): item for item in raw if isinstance(item, dict)}
    if len(result) != REAL_PARTITION_COUNT or "" in result:
        raise V33EvaluationError(f"{label} partition identities are incomplete")
    return result


def _lineage(
    b: Mapping[str, Any], v32: Mapping[str, Any], v33: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    b_parts = _parts(b, "V3.1-B")
    v32_parts = _parts(v32, "V3.2")
    v33_parts = _parts(v33, "V3.3")
    same_inputs = set(b_parts) == set(v32_parts) == set(v33_parts)
    if same_inputs:
        for partition_id in b_parts:
            for key in ("raw", "v3", "valid"):
                hashes = [
                    ((parts[partition_id].get("outputs") or {}).get(key) or {}).get("sha256")
                    for parts in (b_parts, v32_parts, v33_parts)
                ]
                if not isinstance(hashes[0], str) or len(set(hashes)) != 1:
                    same_inputs = False
                    break
            if not same_inputs:
                break
    gates = {
        "same_v31a_parent": len({b.get("parent_v31a_manifest_sha256"), v32.get("parent_v31a_manifest_sha256"), v33.get("parent_v31a_manifest_sha256")}) == 1,
        "same_snapshot": len({b.get("snapshot_manifest_sha256"), v32.get("snapshot_manifest_sha256"), v33.get("snapshot_manifest_sha256")}) == 1,
        "same_v3_policy": len({b.get("v3_policy_snapshot_sha256"), v32.get("v3_policy_snapshot_sha256"), v33.get("v3_policy_snapshot_sha256")}) == 1,
        "same_partition_ids_and_raw_v3_valid_sha256": same_inputs,
        "v33_strict_valid_total_exact": int(v33.get("frozen_v3_class_pixel_totals_sum", -1)) == REAL_STRICT_VALID_TOTAL,
    }
    return gates, b_parts, v32_parts, v33_parts


def _method(method: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "components_total": int(method["components_total"]),
        "dynamic_fragment_count": int(method["dynamic_fragments"]["count"]),
        "dynamic_fragment_area_m2": float(method["dynamic_fragments"]["area_m2"]),
        "boundary_edges": int(method["boundary"]["total_cross_class_boundary"]["edges"]),
        "boundary_metres": float(method["boundary"]["total_cross_class_boundary"]["metres"]),
        "internal_boundary_edges": int(method["boundary"]["internal_cross_class_boundary"]["edges"]),
        "internal_boundary_metres": float(method["boundary"]["internal_cross_class_boundary"]["metres"]),
    }


def _per_class_components(method: Mapping[str, Any]) -> dict[int, int]:
    present = {int(item["class_code"]): int(item["components"]) for item in method["per_class"]}
    return {int(code): present.get(int(code), 0) for code in CLASS_ORDER}


def _coverage(result: Mapping[str, Any]) -> dict[str, bool]:
    coverage = result["coverage"]
    return {
        "part_count_140": int(result["part_count"]) == REAL_PARTITION_COUNT,
        "core_overlap_zero": int(coverage["core_overlap_pixels"]) == 0,
        "geometric_gap_zero": int(coverage["geometric_coverage_gap_pixels"]) == 0,
        "candidate_invalid_inside_zero": int(coverage["invalid_label_inside_valid_pixels"]["v31"]) == 0,
        "candidate_outside_valid_zero": int(coverage["outside_valid_label_pixels"]["v31"]) == 0,
    }


def _audit_summary(parts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    summary = Counter()
    kinds = Counter()
    rejections = Counter()
    per_class: dict[int, Counter[str]] = {int(code): Counter() for code in CLASS_ORDER}
    full = True
    for partition_id in sorted(parts):
        reference = parts[partition_id].get("audit") or {}
        path = Path(str(reference.get("path", "")))
        if not path.is_file() or _sha_file(path) != reference.get("sha256"):
            raise V33EvaluationError(f"{partition_id}: V3.3 audit SHA-256 mismatch")
        stage = _read(path)
        audit = stage.get("v33_audit") or {}
        if stage.get("candidate_label") != "V3.3" or not audit.get("full_audit") or audit.get("audit_truncated"):
            full = False
        for key in (
            "changed_pixel_count", "protected_source_loss_pixel_count",
            "transport_source_loss_pixel_count", "gap_pixels", "overlap_pixels",
            "outside_pixels", "raw_generated", "proposals_canonical",
            "duplicate_proposal_count", "proposals_accepted",
        ):
            summary[key] += int(audit.get(key, 0))
        kinds.update(str(item.get("kind")) for item in (audit.get("accepted") or []))
        rejections.update({str(key): int(value) for key, value in (audit.get("proposal_generation_reject_reason_counts") or {}).items()})
        for code, item in (audit.get("per_class") or {}).items():
            per_class[int(code)].update({
                "source_loss": int(item.get("source_loss", 0)),
                "target_gain": int(item.get("target_gain", 0)),
                "bridge_gain": int(item.get("bridge_gain", 0)),
                "net_pixel_drift": int(item.get("net_pixel_drift", 0)),
            })
    return {
        "full_audit_complete": full,
        **dict(summary),
        "accepted_by_kind": dict(sorted(kinds.items())),
        "generation_rejection_counts": dict(sorted(rejections.items())),
        "per_class": {str(code): dict(sorted(values.items())) for code, values in per_class.items()},
    }


def _transitions(
    from_parts: Mapping[str, Mapping[str, Any]], from_key: str,
    to_parts: Mapping[str, Mapping[str, Any]], to_key: str,
    from_label: str, to_label: str,
) -> dict[str, Any]:
    pairs: Counter[tuple[int, int]] = Counter()
    changed_pixels = 0
    changed_area_m2 = 0.0
    for partition_id in sorted(from_parts):
        left_part, right_part = from_parts[partition_id], to_parts[partition_id]
        left = np.load(Path(str(left_part["outputs"][from_key]["path"])), mmap_mode="r", allow_pickle=False)
        right = np.load(Path(str(right_part["outputs"][to_key]["path"])), mmap_mode="r", allow_pickle=False)
        valid = np.load(Path(str(right_part["outputs"]["valid"]["path"])), mmap_mode="r", allow_pickle=False)
        if left.shape != right.shape or left.shape != valid.shape:
            raise V33EvaluationError(f"{partition_id}: transition shapes differ")
        lvalues = left[valid].astype(np.int64, copy=False)
        rvalues = right[valid].astype(np.int64, copy=False)
        encoded = (lvalues.astype(np.uint64) << np.uint64(32)) | rvalues.astype(np.uint64)
        values, counts = np.unique(encoded, return_counts=True)
        for value, count in zip(values, counts):
            pairs[(int(value >> np.uint64(32)), int(value & np.uint64(0xFFFFFFFF)))] += int(count)
        changed = int(np.count_nonzero(lvalues != rvalues))
        changed_pixels += changed
        changed_area_m2 += changed * float(right_part["physical_metrics"]["pixel_area_m2"])
    return {
        "from_method": from_label,
        "to_method": to_label,
        "changed_pixels": changed_pixels,
        "changed_area_m2": changed_area_m2,
        "transitions": [
            {"from_class": source, "to_class": target, "pixels": count}
            for (source, target), count in sorted(pairs.items())
        ],
    }


def evaluate(
    b_manifest_path: Path, v32_manifest_path: Path,
    v33_manifest_path: Path, output_root: Path,
) -> dict[str, Any]:
    b_manifest_path = b_manifest_path.resolve()
    v32_manifest_path = v32_manifest_path.resolve()
    v33_manifest_path = v33_manifest_path.resolve()
    output_root = output_root.resolve()
    b = _manifest(b_manifest_path, "V3.1-B", "B")
    v32 = _manifest(v32_manifest_path, "V3.2", "V3.2")
    v33 = _manifest(v33_manifest_path, "V3.3", "V3.3")
    lineage, b_parts, v32_parts, v33_parts = _lineage(b, v32, v33)
    if not all(lineage.values()):
        raise V33EvaluationError(f"V3.1-B/V3.2/V3.3 lineage mismatch: {lineage}")

    b_eval = GLOBAL_EVALUATOR.evaluate(b_manifest_path, output_root / "global_b", resume=True)
    v32_eval = GLOBAL_EVALUATOR.evaluate(v32_manifest_path, output_root / "global_v32", resume=True)
    v33_eval = GLOBAL_EVALUATOR.evaluate(v33_manifest_path, output_root / "global_v33", resume=True)
    b_summary = _method(b_eval["result"]["methods"]["v31"])
    v32_summary = _method(v32_eval["result"]["methods"]["v31"])
    v33_summary = _method(v33_eval["result"]["methods"]["v31"])
    v3_summary = _method(v33_eval["result"]["methods"]["v3"])
    v3_by_class = _per_class_components(v33_eval["result"]["methods"]["v3"])
    v33_by_class = _per_class_components(v33_eval["result"]["methods"]["v31"])
    coverage = _coverage(v33_eval["result"])
    audits = _audit_summary(v33_parts)
    safety = {
        **coverage,
        "components_nonincreasing_vs_v3": v33_summary["components_total"] <= v3_summary["components_total"],
        "per_class_components_nonincreasing_vs_v3": all(v33_by_class[code] <= v3_by_class[code] for code in v3_by_class),
        "dynamic_count_nonincreasing_vs_v3": v33_summary["dynamic_fragment_count"] <= v3_summary["dynamic_fragment_count"],
        "dynamic_area_nonincreasing_vs_v3": v33_summary["dynamic_fragment_area_m2"] <= v3_summary["dynamic_fragment_area_m2"] + 1e-7,
        "boundary_edges_nonincreasing_vs_v3": v33_summary["boundary_edges"] <= v3_summary["boundary_edges"],
        "boundary_metres_nonincreasing_vs_v3": v33_summary["boundary_metres"] <= v3_summary["boundary_metres"] + 1e-7,
        "internal_boundary_edges_nonincreasing_vs_v3": v33_summary["internal_boundary_edges"] <= v3_summary["internal_boundary_edges"],
        "internal_boundary_metres_nonincreasing_vs_v3": v33_summary["internal_boundary_metres"] <= v3_summary["internal_boundary_metres"] + 1e-7,
        "protected_source_loss_zero": int(audits["protected_source_loss_pixel_count"]) == 0,
        "full_candidate_audit_complete": bool(audits["full_audit_complete"]),
        "gap_overlap_outside_zero": all(int(audits[key]) == 0 for key in ("gap_pixels", "overlap_pixels", "outside_pixels")),
    }
    actual_reduction = v32_summary["dynamic_fragment_count"] - v33_summary["dynamic_fragment_count"]
    required_reduction = int(math.ceil(v32_summary["dynamic_fragment_count"] * EFFECT_FRACTION))
    effect = {
        "comparison_baseline": "V3.2",
        "fraction": EFFECT_FRACTION,
        "required_dynamic_fragment_reduction": required_reduction,
        "actual_dynamic_fragment_reduction": actual_reduction,
        "pass": actual_reduction >= required_reduction,
    }
    transport = v33.get("transport_overlay") or {}
    transport_overlay = {
        "required": bool(transport.get("required", True)),
        "artifact_sha256": transport.get("artifact_sha256"),
        "artifact_version": transport.get("artifact_version"),
        "status": str(transport.get("status", "required_not_supplied")),
        "post_overlay_hard_gates_run": False,
        "final_publication_pass": False,
    }
    validation_pass = all(safety.values())
    fragmentation_effect_pass = bool(effect["pass"])
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "pre_overlay_fragmentation_candidate_pass"
            if validation_pass and fragmentation_effect_pass
            else "rejected_validation"
        ),
        "lineage_gates": lineage,
        "safety_gates": safety,
        "effect_gate": effect,
        "validation_pass": validation_pass,
        "fragmentation_effect_pass": fragmentation_effect_pass,
        "transport_overlay": transport_overlay,
        "final_publication_pass": False,
        "methods": {
            "V3": v3_summary,
            "V3.1-B": b_summary,
            "V3.2": v32_summary,
            "V3.3": v33_summary,
        },
        "deltas": {
            "V3.3_minus_V3": {key: v33_summary[key] - v3_summary[key] for key in v3_summary},
            "V3.3_minus_V3.1-B": {key: v33_summary[key] - b_summary[key] for key in b_summary},
            "V3.3_minus_V3.2": {key: v33_summary[key] - v32_summary[key] for key in v32_summary},
        },
        "per_class_component_delta_v33_minus_v3": {
            str(code): v33_by_class[code] - v3_by_class[code] for code in v3_by_class
        },
        "candidate_audit_summary": audits,
        "direct_transitions": {
            "V3_to_V3.3": _transitions(v33_parts, "v3", v33_parts, "v33", "V3", "V3.3"),
            "V3.1-B_to_V3.3": _transitions(b_parts, "v31a", v33_parts, "v33", "V3.1-B", "V3.3"),
            "V3.2_to_V3.3": _transitions(v32_parts, "v31a", v33_parts, "v33", "V3.2", "V3.3"),
        },
        "inputs": {
            "b_manifest": str(b_manifest_path),
            "b_manifest_sha256": _sha_file(b_manifest_path),
            "v32_manifest": str(v32_manifest_path),
            "v32_manifest_sha256": _sha_file(v32_manifest_path),
            "v33_manifest": str(v33_manifest_path),
            "v33_manifest_sha256": _sha_file(v33_manifest_path),
            "global_evaluator": str(GLOBAL_EVALUATOR_PATH.resolve()),
            "global_evaluator_sha256": _sha_file(GLOBAL_EVALUATOR_PATH),
            "comparison_code_sha256": _sha_file(Path(__file__).resolve()),
        },
    }
    result["result_sha256"] = _sha_json(result)
    _atomic_json(output_root / "V33_FULL_COMPARISON.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-manifest", type=Path, required=True)
    parser.add_argument("--v32-manifest", type=Path, required=True)
    parser.add_argument("--v33-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = evaluate(
            args.b_manifest, args.v32_manifest, args.v33_manifest,
            args.output_root,
        )
    except (V33EvaluationError, GLOBAL_EVALUATOR.EvaluationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps({
        "status": result["status"],
        "validation_pass": result["validation_pass"],
        "fragmentation_effect_pass": result["fragmentation_effect_pass"],
        "final_publication_pass": result["final_publication_pass"],
        "dynamic_fragment_reduction_vs_v32": result["effect_gate"]["actual_dynamic_fragment_reduction"],
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
