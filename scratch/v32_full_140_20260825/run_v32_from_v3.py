#!/usr/bin/env python3
"""Publish isolated V3.2 Core outputs from the immutable V3-cleaned baseline.

This runner deliberately reuses the V3.1-B input integrity contract, but writes
only ``stage_v32/v32_core.npy``.  Before any Core is processed it counts every
strict-valid owner V3 Core exactly once; real 140-Core runs must total
831531565 pixels.  The resulting census is frozen into every audit and the
execution fingerprint.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping
import uuid

import numpy as np
from rasterio.transform import Affine


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "inference_scripts"))
from deployment_config import CLASS_ORDER  # noqa: E402
from fragmentation_v31_candidate import apply_v32_candidate, v32_policy, v32_policy_snapshot, v32_policy_snapshot_sha256  # noqa: E402

_B_PATH = REPO_ROOT / "scratch" / "v31b_full_140_20260824" / "run_v31b_from_v3.py"
_SPEC = importlib.util.spec_from_file_location("_v31b_contract", _B_PATH)
assert _SPEC and _SPEC.loader
_B = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _B
_SPEC.loader.exec_module(_B)

SCHEMA_VERSION = 1
REAL_PARTITION_COUNT = 140
REAL_STRICT_VALID_TOTAL = 831_531_565
CONTEXT_PIXELS = _B.CONTEXT_PIXELS


class V32RunError(RuntimeError):
    pass


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name); json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def _save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".npy", delete=False) as handle:
        temporary = Path(handle.name); np.save(handle, np.ascontiguousarray(values), allow_pickle=False); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def _full_totals(entries: list[dict[str, Any]], *, self_test: bool) -> dict[int, int]:
    counts = {int(code): 0 for code in CLASS_ORDER}
    seen: set[str] = set()
    for entry in entries:
        partition_id = str(entry["partition_id"])
        if partition_id in seen: raise V32RunError(f"duplicate owner Core {partition_id}")
        seen.add(partition_id)
        labels = np.load(entry["outputs"]["v3"]["path"], mmap_mode="r", allow_pickle=False)
        valid = np.load(entry["outputs"]["valid"]["path"], mmap_mode="r", allow_pickle=False)
        if labels.shape != valid.shape or labels.dtype != np.int16 or valid.dtype != np.bool_:
            raise V32RunError(f"{partition_id}: V3 strict Core metadata changed")
        if np.any(~valid & (labels != -1)) or np.any(valid & ~np.isin(labels, CLASS_ORDER)):
            raise V32RunError(f"{partition_id}: V3 strict Core label contract changed")
        for code in CLASS_ORDER: counts[int(code)] += int(np.count_nonzero(valid & (labels == code)))
    total = sum(counts.values())
    if not self_test and total != REAL_STRICT_VALID_TOTAL:
        raise V32RunError(f"full 140-Core strict-valid V3 total must be {REAL_STRICT_VALID_TOTAL}, got {total}")
    return counts


def _validate_stage(root: Path, entry: Mapping[str, Any], fingerprint: str) -> dict[str, Any]:
    directory = root / "partitions" / str(entry["partition_id"]) / "stage_v32"
    audit_path, hashes_path = directory / "audit.json", directory / "outputs_sha256.json"
    if not audit_path.is_file() or not hashes_path.is_file(): raise V32RunError(f"resume V3.2 stage incomplete: {directory}")
    audit, hashes = _B._read_json(audit_path), _B._read_json(hashes_path).get("files")
    if audit.get("execution_fingerprint_sha256") != fingerprint or not isinstance(hashes, dict) or set(hashes) != {"v32_core.npy", "audit.json"}: raise V32RunError(f"resume V3.2 stage fingerprint/output set differs: {directory}")
    if any(_B._sha256_file(directory / key) != value for key, value in hashes.items()): raise V32RunError(f"resume V3.2 stage SHA-256 mismatch: {directory}")
    output = (audit.get("outputs") or {}).get("v31a")
    if not isinstance(output, dict) or output.get("sha256") != hashes["v32_core.npy"]:
        raise V32RunError(f"resume V3.2 stage audit/output SHA declarations differ: {directory}")
    _B._verify_npy(directory / "v32_core.npy", hashes["v32_core.npy"], _B._shape(entry["core_window"]), np.dtype("int16"), f"resume V3.2 {entry['partition_id']}")
    return audit


def _run_one(job: Mapping[str, Any]) -> dict[str, Any]:
    started = time.monotonic(); entry = job["entry"]
    baseline, probabilities, valid, strict, expanded, owners = _B._stitch(entry, job["entries"], job["global_window"])
    core_slice = _B._slices(expanded, entry["core_window"]); budget = np.zeros(valid.shape, dtype=bool); budget[core_slice] = strict[core_slice]
    metrics = _B._physical(job["transform"], job["crs"], expanded)
    if np.any(budget):
        try:
            result, candidate = apply_v32_candidate(baseline, class_codes=CLASS_ORDER, pixel_area_m2=metrics["pixel_area_m2"], pixel_size_m=(metrics["row_step_m"], metrics["column_step_m"]), valid_mask=valid, class_budget_mask=budget, probabilities=probabilities, confidence=probabilities.max(axis=0).astype(np.float32), policy=job["policy"], baseline_kind="v3_cleaned", full_audit=True, frozen_global_class_pixel_totals=job["totals"])
        except Exception as exc:
            raise V32RunError(f"{entry['partition_id']}: V3.2 API execution failed ({type(exc).__name__}: {exc})") from exc
    else:
        result, candidate = baseline.copy(), {"full_audit": True, "audit_truncated": False, "raw_generated": 0, "proposals_canonical": 0, "duplicate_proposal_count": 0, "proposals_accepted": 0, "proposal_generation_reject_reason_counts": {}, "final_topology_rollback": 0}
    if not candidate.get("full_audit") or candidate.get("audit_truncated"): raise V32RunError(f"{entry['partition_id']}: V3.2 returned incomplete audit")
    core_valid = strict[core_slice]; output = np.full(core_valid.shape, -1, dtype=np.int16); indices = result[core_slice]
    if np.any(indices[core_valid] < 0) or np.any(indices[core_valid] >= len(CLASS_ORDER)): raise V32RunError(f"{entry['partition_id']}: invalid V3.2 label index")
    output[core_valid] = np.asarray(CLASS_ORDER, dtype=np.int16)[indices[core_valid]]
    directory = Path(job["output_root"]) / "partitions" / str(entry["partition_id"]) / "stage_v32"; staging = Path(job["staging_root"]) / str(entry["partition_id"])
    if directory.exists(): raise V32RunError(f"refusing to overwrite existing V3.2 stage: {directory}")
    staging.mkdir(parents=True, exist_ok=False); _save_npy(staging / "v32_core.npy", output); hashes = {"v32_core.npy": _B._sha256_file(staging / "v32_core.npy")}
    raw, canonical, duplicates, accepted = (int(candidate.get(key, 0)) for key in ("raw_generated", "proposals_canonical", "duplicate_proposal_count", "proposals_accepted"))
    if raw != canonical + duplicates or accepted > canonical: raise V32RunError(f"{entry['partition_id']}: V3.2 proposal-count closure is invalid")
    events = {str(key): int(value) for key, value in (candidate.get("proposal_generation_reject_reason_counts") or {}).items()}
    proposal_counts = {"raw_generated": raw, "canonical_generated": canonical, "duplicate_proposal_count": duplicates, "canonical_accepted": accepted, "canonical_rejected": canonical - accepted, "rollback": int(candidate.get("final_topology_rollback", 0)), "closure": "raw_generated=canonical_generated+duplicate_proposal_count;canonical_generated=canonical_accepted+canonical_rejected"}
    audit = {"schema_version": SCHEMA_VERSION, "stage": "v32", "candidate_label": "V3.2", "partition_id": entry["partition_id"], "execution_fingerprint_sha256": job["fingerprint"], "parent_v31a_manifest": job["parent_path"], "parent_v31a_manifest_sha256": job["parent_sha"], "class_codes": list(CLASS_ORDER), "label_encoding": "class_codes_int16_invalid_minus_one", "global_core_window": entry["core_window"], "global_expanded_window": expanded, "context_pixels": CONTEXT_PIXELS, "processing_transform": job["transform"], "crs": job["crs"], "physical_metrics": metrics, "owner_v3_context_sources": [{"partition_id": owner["partition_id"], "v3_context_path": owner["outputs"]["v3_context"]["path"], "v3_context_sha256": owner["outputs"]["v3_context"]["sha256"], "global_core_window": owner["core_window"]} for owner in owners], "v32_policy_snapshot": job["policy_snapshot"], "v32_policy_snapshot_sha256": job["policy_sha"], "frozen_global_class_pixel_totals": {str(code): job["totals"][int(code)] for code in CLASS_ORDER}, "frozen_global_class_pixel_totals_sum": sum(job["totals"].values()), "v32_audit": candidate, "runtime_seconds": time.monotonic() - started, "proposal_counts": proposal_counts, "generation_rejection_events": {"count": sum(events.values()), "by_reason": dict(sorted(events.items()))}, "coverage": {"published_strict_core_only": True, "expanded_owner_v3_context_coverage_exact_once": True, "expanded_decoder_valid_pixel_count": int(valid.sum()), "core_strict_valid_pixel_count": int(core_valid.sum())}, "outputs": {"v31a": {"path": "v32_core.npy", "sha256": hashes["v32_core.npy"], "shape": list(output.shape), "dtype": "int16", "candidate_label": "V3.2", "stage": "v32"}}}
    audit["audit_sha256"] = _sha_json(audit); _atomic_json(staging / "audit.json", audit); hashes["audit.json"] = _B._sha256_file(staging / "audit.json"); _atomic_json(staging / "outputs_sha256.json", {"files": hashes}); directory.parent.mkdir(parents=True, exist_ok=True); os.rename(staging, directory)
    return {"partition_id": entry["partition_id"], "global_core_window": entry["core_window"], "global_expanded_window": expanded, "core_transform": list(tuple(Affine(*job["transform"]) * Affine.translation(entry["core_window"]["x0"], entry["core_window"]["y0"]))[:6]), "crs": job["crs"], "physical_metrics": metrics, "owner_core_pixel_count": int(core_valid.size), "valid_pixel_count": int(core_valid.sum()), "runtime_seconds": audit["runtime_seconds"], "proposal_counts": proposal_counts, "generation_rejection_events": audit["generation_rejection_events"], "outputs": {"raw": entry["outputs"]["raw"], "v3": entry["outputs"]["v3"], "valid": entry["outputs"]["valid"], "v31a": {**audit["outputs"]["v31a"], "path": str((directory / "v32_core.npy").resolve())}}, "audit": {"path": str((directory / "audit.json").resolve()), "sha256": hashes["audit.json"]}}


def run(parent_manifest: Path, output_root: Path, *, workers: int, resume: bool, self_test: bool = False) -> dict[str, Any]:
    if workers < 1 or workers > 2: raise V32RunError("--workers must be 1 or 2")
    source = _B._load_parent(parent_manifest.resolve(), self_test=self_test)
    if output_root.exists() and any(output_root.iterdir()) and not resume: raise V32RunError(f"refusing non-empty output root without --resume: {output_root}")
    totals = _full_totals(source["entries"], self_test=self_test); policy = v32_policy(); snapshot = v32_policy_snapshot(policy); policy_sha = v32_policy_snapshot_sha256(policy)
    verified: dict[str, dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_B._verify_source, entry): entry["partition_id"]
            for entry in source["entries"]
        }
        for future in as_completed(futures):
            verified[futures[future]] = future.result()
    code = _B._code_sha256()
    code[str(Path(__file__).relative_to(REPO_ROOT))] = _B._sha256_file(Path(__file__))
    for path in sorted((REPO_ROOT / "inference_scripts" / "fragmentation_v31_candidate").glob("*.py")):
        code[str(path.relative_to(REPO_ROOT))] = _B._sha256_file(path)
    fingerprint_payload = {"schema_version": SCHEMA_VERSION, "candidate_label": "V3.2", "parent_v31a_manifest_sha256": source["parent_sha256"], "parent_execution_fingerprint_sha256": source["parent"]["execution_fingerprint_sha256"], "class_codes": list(CLASS_ORDER), "context_pixels": CONTEXT_PIXELS, "v32_policy_snapshot_sha256": policy_sha, "frozen_global_class_pixel_totals": {str(code): totals[int(code)] for code in CLASS_ORDER}, "code_sha256": code, "source_probability_decoder_valid_sha256": verified}
    fingerprint = _sha_json(fingerprint_payload); manifest_path = output_root / "run_manifest.json"; completed: dict[str, Any] = {}
    if resume and manifest_path.is_file():
        prior = _B._read_json(manifest_path)
        if prior.get("execution_fingerprint_sha256") != fingerprint: raise V32RunError("resume execution fingerprint differs")
        unexpected = [item.name for item in output_root.iterdir() if item.name not in {"run_manifest.json", "partitions"}]
        if unexpected: raise V32RunError(f"resume output root contains unmanaged entries: {sorted(unexpected)}")
        partition_root = output_root / "partitions"
        if partition_root.exists():
            expected_ids = {entry["partition_id"] for entry in source["entries"]}
            unknown = [item.name for item in partition_root.iterdir() if item.name not in expected_ids]
            if unknown: raise V32RunError(f"resume contains unknown Partition directories: {sorted(unknown)}")
        for entry in source["entries"]:
            directory = output_root / "partitions" / entry["partition_id"] / "stage_v32"
            if directory.exists():
                children = {item.name for item in directory.parent.iterdir()}
                if children != {"stage_v32"}: raise V32RunError(f"resume Partition has unmanaged stage entries: {directory.parent}")
                audit = _validate_stage(output_root, entry, fingerprint); completed[entry["partition_id"]] = {"partition_id": entry["partition_id"], "global_core_window": entry["core_window"], "global_expanded_window": audit["global_expanded_window"], "core_transform": audit.get("core_transform", list(tuple(Affine(*source["transform"]) * Affine.translation(entry["core_window"]["x0"], entry["core_window"]["y0"]))[:6])), "crs": source["crs"], "physical_metrics": audit["physical_metrics"], "owner_core_pixel_count": int(np.prod(_B._shape(entry["core_window"]))), "valid_pixel_count": audit["coverage"]["core_strict_valid_pixel_count"], "runtime_seconds": audit["runtime_seconds"], "proposal_counts": audit["proposal_counts"], "generation_rejection_events": audit["generation_rejection_events"], "outputs": {"raw": entry["outputs"]["raw"], "v3": entry["outputs"]["v3"], "valid": entry["outputs"]["valid"], "v31a": {**audit["outputs"]["v31a"], "path": str((directory / "v32_core.npy").resolve())}}, "audit": {"path": str((directory / "audit.json").resolve()), "sha256": _B._sha256_file(directory / "audit.json")}}
    elif resume and output_root.exists() and any(output_root.iterdir()): raise V32RunError("non-empty resume output lacks run_manifest.json")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": SCHEMA_VERSION, "kind": "v31a_full_partition_core_comparison", "candidate_label": "V3.2", "candidate_algorithm": "v32_policy/apply_v32_candidate", "status": "running", "self_test": self_test, "parent_v31a_manifest": source["parent_path"], "parent_v31a_manifest_sha256": source["parent_sha256"], "snapshot_manifest": source["parent"]["snapshot_manifest"], "snapshot_manifest_sha256": source["parent"]["snapshot_manifest_sha256"], "execution_fingerprint": fingerprint_payload, "execution_fingerprint_sha256": fingerprint, "class_codes": list(CLASS_ORDER), "label_encoding": "class_codes_int16_invalid_minus_one", "processing_transform": source["transform"], "crs": source["crs"], "global_window": source["global_window"], "context_pixels": CONTEXT_PIXELS, "code_sha256": code, "v3_policy_snapshot": source["parent"]["v3_policy_snapshot"], "v3_policy_snapshot_sha256": source["parent"]["v3_policy_snapshot_sha256"], "v31a_policy_snapshot": snapshot, "v31a_policy_snapshot_sha256": policy_sha, "v32_policy_snapshot": snapshot, "v32_policy_snapshot_sha256": policy_sha, "frozen_global_class_pixel_totals": {str(code): totals[int(code)] for code in CLASS_ORDER}, "frozen_global_class_pixel_totals_sum": sum(totals.values()), "requested_partition_count": len(source["entries"]), "completed_partition_count": len(completed), "stage_v3_complete": True, "stage_v3": source["parent"]["stage_v3"], "stage_v3_partitions": source["parent"]["stage_v3_partitions"], "resource_plan": {"workers": workers, "real_workers_hard_limit": 2, "V3_execution": "forbidden_reused_parent_stage_v3_only"}, "partitions": [completed[key] for key in sorted(completed)]}
    _atomic_json(manifest_path, manifest)
    staging = output_root.parent / f".{output_root.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    staging.mkdir(exist_ok=False)
    try:
        pending = [entry for entry in source["entries"] if entry["partition_id"] not in completed]
        payload = {"entries": source["entries"], "global_window": source["global_window"], "transform": source["transform"], "crs": source["crs"], "policy": policy, "policy_snapshot": snapshot, "policy_sha": policy_sha, "totals": totals, "fingerprint": fingerprint, "parent_path": source["parent_path"], "parent_sha": source["parent_sha256"], "output_root": str(output_root.resolve()), "staging_root": str(staging.resolve())}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_one, {**payload, "entry": entry}): entry["partition_id"] for entry in pending}
            for future in as_completed(futures):
                result = future.result()
                completed[result["partition_id"]] = result
                manifest["completed_partition_count"] = len(completed)
                manifest["partitions"] = [completed[key] for key in sorted(completed)]
                _atomic_json(manifest_path, manifest)
                print(f"stage_v32 {len(completed)}/{len(source['entries'])} {result['partition_id']}", flush=True)
    finally:
        for path in staging.glob("*"):
            if path.is_dir():
                import shutil
                shutil.rmtree(path, ignore_errors=True)
        staging.rmdir()
    if len(completed) != len(source["entries"]): raise V32RunError("V3.2 run incomplete")
    parts = [completed[key] for key in sorted(completed)]
    manifest.update({"status": "complete", "completed_partition_count": len(parts), "partitions": parts, "coverage": {"all_snapshot_partitions_requested": len(parts) == REAL_PARTITION_COUNT if not self_test else True, "core_windows_nonoverlapping": True, "global_core_grid_exact": True, "complete": True, "published_core_pixel_count": int(sum(item["owner_core_pixel_count"] for item in parts)), "published_valid_pixel_count": int(sum(item["valid_pixel_count"] for item in parts))}, "proposal_counts": {key: int(sum(item["proposal_counts"][key] for item in parts)) for key in ("raw_generated", "canonical_generated", "duplicate_proposal_count", "canonical_accepted", "canonical_rejected", "rollback")}, "generation_rejection_events": {"count": int(sum(item["generation_rejection_events"]["count"] for item in parts)), "by_reason": {key: int(sum(item["generation_rejection_events"]["by_reason"].get(key, 0) for item in parts)) for key in sorted({reason for item in parts for reason in item["generation_rejection_events"]["by_reason"]})}}, "runtime_seconds": float(sum(item["runtime_seconds"] for item in parts))})
    manifest["manifest_sha256"] = _sha_json(manifest); _atomic_json(manifest_path, manifest); return manifest


def _self_test(output_root: Path | None, workers: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v32-selftest-") as temporary:
        root = Path(temporary); parent = _B._write_self_test_parent(root); target = output_root or root / "out"
        result = run(parent, target, workers=workers, resume=False, self_test=True); first = (target / "run_manifest.json").read_bytes(); resumed = run(parent, target, workers=workers, resume=True, self_test=True)
        if first != (target / "run_manifest.json").read_bytes() or resumed["manifest_sha256"] != result["manifest_sha256"]: raise V32RunError("self-test resume changed completed manifest")
        if result["completed_partition_count"] != 4 or any(not (Path(item["outputs"]["v31a"]["path"]).name == "v32_core.npy") for item in result["partitions"]): raise V32RunError("self-test V3.2 publication contract failed")
        return resumed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--v31a-manifest", type=Path); parser.add_argument("--output-root", type=Path); parser.add_argument("--workers", type=int, default=1); parser.add_argument("--resume", action="store_true"); parser.add_argument("--self-test", action="store_true"); args = parser.parse_args(argv)
    try:
        if args.self_test: result = _self_test(args.output_root, args.workers)
        elif args.v31a_manifest is None or args.output_root is None: parser.error("real execution requires --v31a-manifest and --output-root")
        else: result = run(args.v31a_manifest, args.output_root, workers=args.workers, resume=args.resume)
        print(json.dumps({"status": result["status"], "candidate_label": "V3.2"}), flush=True); return 0
    except V32RunError as exc: print(f"ERROR: {exc}", file=sys.stderr, flush=True); return 2


if __name__ == "__main__": raise SystemExit(main())
