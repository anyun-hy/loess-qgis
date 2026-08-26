#!/usr/bin/env python3
"""Run one isolated, bounded second round on a complete V3.1-B publication."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
from rasterio.transform import Affine


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "inference_scripts"))

from deployment_config import CLASS_ORDER  # noqa: E402
from fragmentation_v31_candidate import (  # noqa: E402
    CandidateError,
    apply_v31d_candidate,
    v31d_policy,
)


C_RUNNER_PATH = REPO_ROOT / "scratch" / "v31c_full_140_20260825" / "run_v31c_global.py"
SCHEMA_VERSION = 1
REAL_PARTITION_COUNT = 140
EFFECT_FRACTION = 0.005


class V31DRunError(RuntimeError):
    """The bounded second-round experiment failed an integrity contract."""


def _load_module(name: str, path: Path) -> Any:
    if not path.is_file():
        raise V31DRunError(f"required helper is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise V31DRunError(f"cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C_RUNNER = _load_module("_v31d_frozen_c_runner", C_RUNNER_PATH)
B_RUNNER = C_RUNNER.B_RUNNER
EVALUATOR = C_RUNNER.EVALUATOR


def _sha256_file(path: Path) -> str:
    return C_RUNNER._sha256_file(path)


def _sha256_json(value: Any) -> str:
    return C_RUNNER._sha256_json(value)


def _read_json(path: Path) -> dict[str, Any]:
    return C_RUNNER._read_json(path)


def _atomic_json(path: Path, value: Any) -> None:
    C_RUNNER._atomic_json(path, value)


def _save_npy(path: Path, values: np.ndarray) -> None:
    C_RUNNER._save_npy(path, values)


def _code_sha256() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        C_RUNNER_PATH.resolve(),
        C_RUNNER.B_RUNNER_PATH.resolve(),
        C_RUNNER.EVALUATOR_PATH.resolve(),
        REPO_ROOT / "inference_scripts" / "deployment_config.py",
        REPO_ROOT / "inference_scripts" / "fragmentation_v31_candidate" / "__init__.py",
        REPO_ROOT / "inference_scripts" / "fragmentation_v31_candidate" / "candidate.py",
        REPO_ROOT / "inference_scripts" / "fragmentation_v31_candidate" / "v31d.py",
    ]
    return {str(path.relative_to(REPO_ROOT)): _sha256_file(path) for path in paths}


def _class_indices(class_code_values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.full(class_code_values.shape, -1, dtype=np.int16)
    for index, code in enumerate(CLASS_ORDER):
        result[valid & (class_code_values == int(code))] = int(index)
    if np.any(valid & (result < 0)):
        raise V31DRunError("B publication contains a class outside CLASS_ORDER")
    return result


def _stitch_b(
    target: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    b_parts: Mapping[str, Mapping[str, Any]],
    global_window: Mapping[str, int],
) -> tuple[np.ndarray, dict[str, int]]:
    expanded = B_RUNNER._expand(target["core_window"], global_window)
    output = np.full(B_RUNNER._shape(expanded), -1, dtype=np.int16)
    coverage = np.zeros(output.shape, dtype=np.uint8)
    for owner in entries:
        selected = B_RUNNER._intersect(expanded, owner["core_window"])
        if selected is None:
            continue
        destination = B_RUNNER._slices(expanded, selected)
        owner_slice = B_RUNNER._slices(owner["core_window"], selected)
        b_part = b_parts[str(owner["partition_id"])]
        codes = np.load(b_part["outputs"]["v31a"]["path"], mmap_mode="r", allow_pickle=False)[
            owner_slice
        ]
        strict = np.load(b_part["outputs"]["valid"]["path"], mmap_mode="r", allow_pickle=False)[
            owner_slice
        ]
        indices = _class_indices(codes, strict)
        output[destination][strict] = indices[strict]
        coverage[destination] += 1
    if not np.all(coverage == 1):
        raise V31DRunError(f"{target['partition_id']}: B expanded Core coverage is not exact")
    return output, expanded


def _round1_ledger(
    b_part: Mapping[str, Any],
) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    v3 = np.load(b_part["outputs"]["v3"]["path"], mmap_mode="r", allow_pickle=False)
    b = np.load(b_part["outputs"]["v31a"]["path"], mmap_mode="r", allow_pickle=False)
    valid = np.load(b_part["outputs"]["valid"]["path"], mmap_mode="r", allow_pickle=False)
    changed = valid & (v3 != b)
    source = {int(code): int(np.count_nonzero(changed & (v3 == int(code)))) for code in CLASS_ORDER}
    target = {int(code): int(np.count_nonzero(changed & (b == int(code)))) for code in CLASS_ORDER}
    candidate_audit = b_part["audit_data"].get("v31b_audit") or {}
    budgets = candidate_audit.get("class_budget_pixels") or {}
    if candidate_audit.get("skipped") and candidate_audit.get("reason") == "empty_owner_core_strict_valid":
        if int(np.count_nonzero(valid)) != 0 or any(source.values()) or any(target.values()):
            raise V31DRunError(f"{b_part['partition_id']}: empty B Core has pixel transitions")
        protected_bridge = {int(code): 0 for code in CLASS_ORDER}
        return source, target, protected_bridge
    if set(int(code) for code in budgets) != set(int(code) for code in CLASS_ORDER):
        raise V31DRunError(f"{b_part['partition_id']}: B cumulative budget audit is incomplete")
    protected_bridge: dict[int, int] = {}
    protected = set(v31d_policy().protected_source_codes)
    for code in CLASS_ORDER:
        row = budgets[str(int(code))]
        if source[int(code)] != int(row["source_loss"]) or target[int(code)] != int(row["target_gain"]):
            raise V31DRunError(
                f"{b_part['partition_id']}: V3-to-B transition ledger disagrees for class {code}"
            )
        bridge = int(row["protected_bridge_gain"])
        expected = target[int(code)] if int(code) in protected else 0
        if bridge != expected:
            raise V31DRunError(
                f"{b_part['partition_id']}: protected bridge ledger disagrees for class {code}"
            )
        protected_bridge[int(code)] = bridge
    return source, target, protected_bridge


def _run_partition(job: Mapping[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    entry = job["entry"]
    partition_id = str(entry["partition_id"])
    directory = Path(job["output_root"]) / "partitions" / partition_id / "stage_v31d"
    shape = B_RUNNER._shape(entry["core_window"])
    if directory.exists():
        audit = _read_json(directory / "audit.json")
        hashes = _read_json(directory / "outputs_sha256.json").get("files")
        if (
            audit.get("execution_fingerprint_sha256") != job["fingerprint"]
            or not isinstance(hashes, dict)
            or set(hashes) != {"v31d_core.npy", "audit.json"}
            or any(_sha256_file(directory / name) != value for name, value in hashes.items())
        ):
            raise V31DRunError(f"resume D stage differs: {directory}")
        B_RUNNER._verify_npy(
            directory / "v31d_core.npy",
            hashes["v31d_core.npy"],
            shape,
            np.dtype("int16"),
            f"resume D {partition_id}",
        )
        return {
            "partition_id": partition_id,
            "audit_data": audit,
            "audit_path": str((directory / "audit.json").resolve()),
            "audit_sha256": hashes["audit.json"],
            "output_path": str((directory / "v31d_core.npy").resolve()),
            "output_sha256": hashes["v31d_core.npy"],
        }
    v3_context, probabilities, decoder_valid, strict, expanded, _owners = B_RUNNER._stitch(
        entry, job["entries"], job["global_window"]
    )
    b_context, b_expanded = _stitch_b(
        entry, job["entries"], job["b_parts"], job["global_window"]
    )
    if b_expanded != expanded:
        raise V31DRunError(f"{partition_id}: V3 and B expanded windows differ")
    # B has no publication outside strict validity.  Preserve the frozen V3
    # decoder-valid context there, and overlay B for every owner strict pixel.
    b_context[decoder_valid & ~strict] = v3_context[decoder_valid & ~strict]
    if np.any(decoder_valid & (b_context < 0)):
        raise V31DRunError(f"{partition_id}: stitched B decoder context is incomplete")
    immutable = decoder_valid & (b_context != v3_context)
    core_slice = B_RUNNER._slices(expanded, entry["core_window"])
    budget_mask = np.zeros(decoder_valid.shape, dtype=bool)
    budget_mask[core_slice] = strict[core_slice]
    source_loss, target_gain, protected_bridge = _round1_ledger(job["b_parts"][partition_id])
    metrics = B_RUNNER._physical(job["transform"], job["crs"], expanded)
    if np.any(budget_mask):
        try:
            result, candidate_audit = apply_v31d_candidate(
                b_context,
                original_v3_labels=v3_context,
                round1_immutable_mask=immutable,
                round1_source_loss_pixels=source_loss,
                round1_target_gain_pixels=target_gain,
                round1_protected_bridge_gain_pixels=protected_bridge,
                class_codes=CLASS_ORDER,
                pixel_area_m2=metrics["pixel_area_m2"],
                pixel_size_m=(metrics["row_step_m"], metrics["column_step_m"]),
                valid_mask=decoder_valid,
                class_budget_mask=budget_mask,
                probabilities=probabilities,
                confidence=probabilities.max(axis=0).astype(np.float32),
                policy=job["policy"],
                baseline_kind="v31b_cleaned",
                full_audit=True,
            )
        except CandidateError as exc:
            raise V31DRunError(f"{partition_id}: D candidate rejected input: {exc}") from exc
    else:
        result = b_context.copy()
        candidate_audit = {
            "skipped": True,
            "reason": "empty_owner_core_strict_valid",
            "full_audit": True,
            "audit_truncated": False,
            "raw_generated": 0,
            "proposals_canonical": 0,
            "duplicate_proposal_count": 0,
            "proposals_accepted": 0,
            "proposal_reject_reason_counts": {},
            "proposal_generation_reject_reason_counts": {},
            "changed_pixel_count": 0,
            "round1_immutable_pixel_count": int(np.count_nonzero(immutable)),
        }
    if not candidate_audit.get("full_audit") or candidate_audit.get("audit_truncated"):
        raise V31DRunError(f"{partition_id}: D candidate audit is incomplete")
    core_valid = strict[core_slice]
    indices = result[core_slice]
    output = np.full(core_valid.shape, -1, dtype=np.int16)
    codes = np.asarray(CLASS_ORDER, dtype=np.int16)
    output[core_valid] = codes[indices[core_valid]]
    b_core = np.load(job["b_parts"][partition_id]["outputs"]["v31a"]["path"], mmap_mode="r", allow_pickle=False)
    if np.any(output[~core_valid] != -1) or np.any(~np.isin(output[core_valid], CLASS_ORDER)):
        raise V31DRunError(f"{partition_id}: D output violates single-label coverage")
    staging = Path(job["staging_root"]) / partition_id
    staging.mkdir(parents=True, exist_ok=False)
    _save_npy(staging / "v31d_core.npy", output)
    output_sha = _sha256_file(staging / "v31d_core.npy")
    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "v31d",
        "candidate_label": "D",
        "partition_id": partition_id,
        "execution_fingerprint_sha256": job["fingerprint"],
        "v31b_manifest": job["b_manifest_path"],
        "v31b_manifest_sha256": job["b_manifest_sha256"],
        "global_core_window": entry["core_window"],
        "global_expanded_window": expanded,
        "core_transform": list(
            tuple(Affine(*job["transform"]) * Affine.translation(entry["core_window"]["x0"], entry["core_window"]["y0"]))[:6]
        ),
        "crs": job["crs"],
        "physical_metrics": metrics,
        "v31d_audit": candidate_audit,
        "round2_changed_pixel_count": int(np.count_nonzero(core_valid & (output != b_core))),
        "runtime_seconds": time.monotonic() - started,
        "coverage": {
            "published_strict_core_only": True,
            "core_strict_valid_pixel_count": int(core_valid.sum()),
            "single_label": True,
            "gap_pixels": 0,
            "overlap_pixels": 0,
            "outside_pixels": 0,
        },
        "outputs": {
            "v31a": {
                "candidate_label": "D",
                "path": "v31d_core.npy",
                "sha256": output_sha,
                "shape": list(output.shape),
                "dtype": "int16",
            }
        },
    }
    audit["audit_sha256"] = _sha256_json(audit)
    _atomic_json(staging / "audit.json", audit)
    hashes = {"v31d_core.npy": output_sha, "audit.json": _sha256_file(staging / "audit.json")}
    _atomic_json(staging / "outputs_sha256.json", {"files": hashes})
    directory.parent.mkdir(parents=True, exist_ok=True)
    os.rename(staging, directory)
    return {
        "partition_id": partition_id,
        "audit_data": audit,
        "audit_path": str((directory / "audit.json").resolve()),
        "audit_sha256": hashes["audit.json"],
        "output_path": str((directory / "v31d_core.npy").resolve()),
        "output_sha256": output_sha,
    }


def _evaluation_manifest(
    *,
    output_root: Path,
    published: Sequence[Mapping[str, Any]],
    b_data: Mapping[str, Any],
    policy: Any,
) -> Path:
    by_id = {str(item["partition_id"]): item for item in published}
    parts = []
    for partition_id in sorted(b_data["parts"]):
        b_part = b_data["parts"][partition_id]
        item = by_id[partition_id]
        parts.append(
            {
                "partition_id": partition_id,
                "candidate_label": "D",
                "global_core_window": b_part["global_core_window"],
                "core_transform": b_part["core_transform"],
                "crs": b_part["crs"],
                "physical_metrics": b_part["physical_metrics"],
                "outputs": {
                    "raw": b_part["outputs"]["raw"],
                    "v3": b_part["outputs"]["v3"],
                    "valid": b_part["outputs"]["valid"],
                    "v31a": {
                        "candidate_label": "D",
                        "path": item["output_path"],
                        "sha256": item["output_sha256"],
                        "shape": list(B_RUNNER._shape(b_part["global_core_window"])),
                        "dtype": "int16",
                    },
                },
                "audit": {"path": item["audit_path"], "sha256": item["audit_sha256"]},
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "v31d_bounded_second_round_evaluation_input",
        "candidate_label": "D",
        "v31b_manifest": b_data["manifest_path"],
        "v31b_manifest_sha256": b_data["manifest_file_sha256"],
        "approved_dynamic_mmu_m2": {
            str(code): float(row.dynamic_fragmentation_m2)
            for code, row in sorted(policy.class_policies.items())
        },
        "processing_transform": b_data["manifest"]["processing_transform"],
        "crs": b_data["manifest"]["crs"],
        "global_window": b_data["manifest"]["global_window"],
        "partitions": parts,
    }
    manifest["manifest_sha256"] = _sha256_json(manifest)
    path = output_root / "evaluation_manifest.json"
    _atomic_json(path, manifest)
    return path


def _per_class_components(method: Mapping[str, Any]) -> dict[int, int]:
    present = {int(item["class_code"]): int(item["components"]) for item in method["per_class"]}
    return {int(code): present.get(int(code), 0) for code in CLASS_ORDER}


def _validate_results(
    b_result: Mapping[str, Any], d_result: Mapping[str, Any]
) -> dict[str, Any]:
    b_method = b_result["methods"]["v31"]
    d_method = d_result["methods"]["v31"]
    b_summary = C_RUNNER._method_summary(b_method)
    d_summary = C_RUNNER._method_summary(d_method)
    b_boundary = b_method["boundary"]["total_cross_class_boundary"]
    d_boundary = d_method["boundary"]["total_cross_class_boundary"]
    coverage = C_RUNNER._hard_coverage_gate(d_result)
    b_by_class = _per_class_components(b_method)
    d_by_class = _per_class_components(d_method)
    dynamic_reduction = b_summary["dynamic_fragment_count"] - d_summary["dynamic_fragment_count"]
    required_reduction = int(math.ceil(b_summary["dynamic_fragment_count"] * EFFECT_FRACTION))
    safety = {
        "coverage_pass": all(coverage.values()),
        "components_nonincreasing": d_summary["components_total"] <= b_summary["components_total"],
        "per_class_components_nonincreasing": all(
            d_by_class[code] <= b_by_class[code] for code in b_by_class
        ),
        "dynamic_count_nonincreasing": d_summary["dynamic_fragment_count"] <= b_summary["dynamic_fragment_count"],
        "dynamic_area_nonincreasing": d_summary["dynamic_fragment_area_m2"] <= b_summary["dynamic_fragment_area_m2"] + 1e-7,
        "boundary_edges_nonincreasing": int(d_boundary["edges"]) <= int(b_boundary["edges"]),
        "boundary_metres_nonincreasing": float(d_boundary["metres"]) <= float(b_boundary["metres"]) + 1e-7,
    }
    return {
        "coverage": coverage,
        "b": b_summary,
        "d": d_summary,
        "d_minus_b": {key: d_summary[key] - b_summary[key] for key in b_summary},
        "boundary": {
            "b_edges": int(b_boundary["edges"]),
            "d_edges": int(d_boundary["edges"]),
            "delta_edges": int(d_boundary["edges"]) - int(b_boundary["edges"]),
            "b_metres": float(b_boundary["metres"]),
            "d_metres": float(d_boundary["metres"]),
            "delta_metres": float(d_boundary["metres"]) - float(b_boundary["metres"]),
        },
        "safety_gates": safety,
        "validation_pass": all(safety.values()),
        "effect_gate": {
            "fraction": EFFECT_FRACTION,
            "required_dynamic_fragment_reduction": required_reduction,
            "actual_dynamic_fragment_reduction": dynamic_reduction,
            "pass": dynamic_reduction >= required_reduction,
        },
        "acceptance_pass": all(safety.values()) and dynamic_reduction >= required_reduction,
    }


def _verify_complete(manifest: Mapping[str, Any]) -> None:
    references = [
        (
            (manifest.get("evaluation_manifest") or {}).get("path"),
            (manifest.get("evaluation_manifest") or {}).get("sha256"),
        ),
        (
            (manifest.get("global_evaluation") or {}).get("result"),
            (manifest.get("global_evaluation") or {}).get("result_sha256"),
        ),
        (
            (manifest.get("global_evaluation") or {}).get("audit"),
            (manifest.get("global_evaluation") or {}).get("audit_sha256"),
        ),
        (
            (manifest.get("baseline_b_global_evaluation") or {}).get("result"),
            (manifest.get("baseline_b_global_evaluation") or {}).get("result_sha256"),
        ),
        (
            (manifest.get("baseline_b_global_evaluation") or {}).get("audit"),
            (manifest.get("baseline_b_global_evaluation") or {}).get("audit_sha256"),
        ),
    ]
    parts = manifest.get("partitions") or []
    if len(parts) != int(manifest.get("completed_partition_count", -1)):
        raise V31DRunError("completed D partition count does not close")
    for part in parts:
        output = (part.get("outputs") or {}).get("v31a") or {}
        audit = part.get("audit") or {}
        references.extend(((output.get("path"), output.get("sha256")), (audit.get("path"), audit.get("sha256"))))
    for path_text, expected in references:
        if not isinstance(path_text, str) or not isinstance(expected, str):
            raise V31DRunError("completed D artifact reference is incomplete")
        path = Path(path_text)
        if not path.is_file() or _sha256_file(path) != expected:
            raise V31DRunError(f"completed D artifact SHA-256 mismatch: {path}")


def run(
    b_manifest_path: Path,
    output_root: Path,
    *,
    workers: int,
    resume: bool,
    self_test: bool = False,
) -> dict[str, Any]:
    if workers < 1 or workers > 2:
        raise V31DRunError("--workers must be 1 or 2")
    b_data = C_RUNNER._load_b_manifest(b_manifest_path.resolve(), self_test=self_test)
    if output_root.exists() and any(output_root.iterdir()) and not resume:
        raise V31DRunError(f"refusing non-empty output root without --resume: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    policy = v31d_policy()
    policy_snapshot = C_RUNNER.policy_snapshot(policy)
    policy_snapshot["adjudication_mode"] = "bounded_second_round_dependency_incremental_v1"
    policy_snapshot["algorithm_contract"]["additional_generation_rounds"] = 1
    policy_snapshot["algorithm_contract"]["round1_changed_pixels"] = "immutable"
    policy_snapshot["algorithm_contract"]["round1_target_components"] = "dependency_locked"
    policy_snapshot["algorithm_contract"]["budget"] = "cumulative_from_original_v3_owner_core"
    policy_sha = _sha256_json(policy_snapshot)
    code_sha = _code_sha256()
    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION,
        "candidate_label": "D",
        "v31b_manifest_sha256": b_data["manifest_file_sha256"],
        "policy_snapshot_sha256": policy_sha,
        "effect_fraction": EFFECT_FRACTION,
        "code_sha256": code_sha,
    }
    fingerprint = _sha256_json(fingerprint_payload)
    manifest_path = output_root / "run_manifest.json"
    if resume and manifest_path.is_file():
        prior = _read_json(manifest_path)
        if prior.get("execution_fingerprint_sha256") != fingerprint:
            raise V31DRunError("resume D execution fingerprint differs")
        if prior.get("status") == "complete":
            if prior.get("manifest_sha256") != _sha256_json(
                {key: value for key, value in prior.items() if key != "manifest_sha256"}
            ):
                raise V31DRunError("completed D manifest self SHA-256 mismatch")
            _verify_complete(prior)
            return prior
    running: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "v31d_bounded_second_round_candidate",
        "candidate_label": "D",
        "status": "running",
        "self_test": bool(self_test),
        "v31b_manifest": b_data["manifest_path"],
        "v31b_manifest_sha256": b_data["manifest_file_sha256"],
        "execution_fingerprint": fingerprint_payload,
        "execution_fingerprint_sha256": fingerprint,
        "policy_snapshot": policy_snapshot,
        "policy_snapshot_sha256": policy_sha,
        "requested_partition_count": len(b_data["source"]["entries"]),
        "completed_partition_count": 0,
    }
    _atomic_json(manifest_path, running)
    staging_root = output_root / f".stage-v31d-{os.getpid()}-{uuid.uuid4().hex}"
    staging_root.mkdir(exist_ok=False)
    completed: dict[str, dict[str, Any]] = {}
    started = time.monotonic()
    payload = {
        "entries": b_data["source"]["entries"],
        "b_parts": b_data["parts"],
        "global_window": b_data["source"]["global_window"],
        "transform": b_data["source"]["transform"],
        "crs": b_data["source"]["crs"],
        "policy": policy,
        "fingerprint": fingerprint,
        "output_root": str(output_root),
        "staging_root": str(staging_root),
        "b_manifest_path": b_data["manifest_path"],
        "b_manifest_sha256": b_data["manifest_file_sha256"],
    }
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_partition, {**payload, "entry": entry}): entry["partition_id"]
                for entry in b_data["source"]["entries"]
            }
            for future in as_completed(futures):
                item = future.result()
                completed[item["partition_id"]] = item
                running["completed_partition_count"] = len(completed)
                _atomic_json(manifest_path, running)
                print(
                    f"stage_v31d {len(completed)}/{len(b_data['source']['entries'])} {item['partition_id']}",
                    flush=True,
                )
    finally:
        for child in staging_root.iterdir():
            if child.is_dir():
                import shutil

                shutil.rmtree(child, ignore_errors=True)
        staging_root.rmdir()
    published = [completed[key] for key in sorted(completed)]
    evaluation_manifest = _evaluation_manifest(
        output_root=output_root, published=published, b_data=b_data, policy=policy
    )
    b_evaluation = EVALUATOR.evaluate(
        Path(b_data["manifest_path"]),
        output_root / "baseline_b_global_evaluation",
        resume=True,
    )
    d_evaluation = EVALUATOR.evaluate(
        evaluation_manifest, output_root / "global_evaluation", resume=True
    )
    validation = _validate_results(b_evaluation["result"], d_evaluation["result"])
    round2_changed = sum(int(item["audit_data"]["round2_changed_pixel_count"]) for item in published)
    accepted_proposals = sum(
        int((item["audit_data"].get("v31d_audit") or {}).get("proposals_accepted", 0))
        for item in published
    )
    if _sha256_file(b_manifest_path) != b_data["manifest_file_sha256"]:
        raise V31DRunError("B manifest changed during D run")
    manifest: dict[str, Any] = {
        **running,
        "status": "complete" if validation["validation_pass"] else "rejected_validation",
        "completed_partition_count": len(published),
        "code_sha256": code_sha,
        "class_codes": list(CLASS_ORDER),
        "label_encoding": "class_codes_int16_invalid_minus_one",
        "processing_transform": b_data["manifest"]["processing_transform"],
        "crs": b_data["manifest"]["crs"],
        "global_window": b_data["manifest"]["global_window"],
        "round2_changed_pixel_count": round2_changed,
        "round2_accepted_proposal_count": accepted_proposals,
        "validation": validation,
        "validation_pass": validation["validation_pass"],
        "acceptance_pass": validation["acceptance_pass"],
        "runtime_seconds": time.monotonic() - started,
        "evaluation_manifest": {
            "path": str(evaluation_manifest.resolve()),
            "sha256": _sha256_file(evaluation_manifest),
        },
        "global_evaluation": {
            "result": str((output_root / "global_evaluation" / "global_fragmentation.json").resolve()),
            "result_sha256": _sha256_file(output_root / "global_evaluation" / "global_fragmentation.json"),
            "audit": str((output_root / "global_evaluation" / "audit.json").resolve()),
            "audit_sha256": _sha256_file(output_root / "global_evaluation" / "audit.json"),
        },
        "baseline_b_global_evaluation": {
            "result": str((output_root / "baseline_b_global_evaluation" / "global_fragmentation.json").resolve()),
            "result_sha256": _sha256_file(output_root / "baseline_b_global_evaluation" / "global_fragmentation.json"),
            "audit": str((output_root / "baseline_b_global_evaluation" / "audit.json").resolve()),
            "audit_sha256": _sha256_file(output_root / "baseline_b_global_evaluation" / "audit.json"),
        },
        "partitions": [
            {
                "partition_id": item["partition_id"],
                "global_core_window": b_data["parts"][item["partition_id"]]["global_core_window"],
                "core_transform": b_data["parts"][item["partition_id"]]["core_transform"],
                "crs": b_data["parts"][item["partition_id"]]["crs"],
                "physical_metrics": b_data["parts"][item["partition_id"]]["physical_metrics"],
                "outputs": {
                    "raw": b_data["parts"][item["partition_id"]]["outputs"]["raw"],
                    "v3": b_data["parts"][item["partition_id"]]["outputs"]["v3"],
                    "valid": b_data["parts"][item["partition_id"]]["outputs"]["valid"],
                    "v31a": {
                        "candidate_label": "D",
                        "path": item["output_path"],
                        "sha256": item["output_sha256"],
                        "shape": list(B_RUNNER._shape(b_data["parts"][item["partition_id"]]["global_core_window"])),
                        "dtype": "int16",
                    },
                },
                "audit": {"path": item["audit_path"], "sha256": item["audit_sha256"]},
            }
            for item in published
        ],
    }
    manifest["manifest_sha256"] = _sha256_json(manifest)
    _atomic_json(manifest_path, manifest)
    if not validation["validation_pass"]:
        raise V31DRunError("D global safety validation failed")
    return manifest


def _self_test(output_root: Path | None, workers: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v31d-selftest-") as temporary:
        root = Path(temporary)
        parent = B_RUNNER._write_self_test_parent(root)
        b_root = root / "b"
        B_RUNNER.run(parent, b_root, workers=workers, resume=False, self_test=True)
        target = root / "d" if output_root is None else output_root
        result = run(
            b_root / "run_manifest.json",
            target,
            workers=workers,
            resume=False,
            self_test=True,
        )
        if result.get("status") != "complete" or result.get("completed_partition_count") != 4:
            raise V31DRunError("D self-test did not complete four Cores")
        first_sha = result["manifest_sha256"]
        resumed = run(
            b_root / "run_manifest.json",
            target,
            workers=workers,
            resume=True,
            self_test=True,
        )
        if resumed["manifest_sha256"] != first_sha:
            raise V31DRunError("D self-test resume changed completed manifest")
        return resumed


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v31b-manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        if args.v31b_manifest is not None:
            parser.error("--self-test creates its own frozen B fixture")
    elif args.v31b_manifest is None or args.output_root is None:
        parser.error("real execution requires --v31b-manifest and --output-root")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    try:
        result = (
            _self_test(args.output_root, args.workers)
            if args.self_test
            else run(
                args.v31b_manifest,
                args.output_root,
                workers=args.workers,
                resume=args.resume,
                self_test=False,
            )
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "candidate_label": "D",
                    "completed_partition_count": result["completed_partition_count"],
                    "round2_accepted_proposal_count": result["round2_accepted_proposal_count"],
                    "acceptance_pass": result["acceptance_pass"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0
    except (V31DRunError, CandidateError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
