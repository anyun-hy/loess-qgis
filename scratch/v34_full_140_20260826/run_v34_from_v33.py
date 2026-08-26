#!/usr/bin/env python3
"""Run one isolated, cumulative-budget V3.4 pass on complete V3.3 outputs."""

from __future__ import annotations

import argparse
from collections import Counter
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
from fragmentation_policy import load_policy, policy_sha256 as config_policy_sha256  # noqa: E402
from fragmentation_v34_candidate import (  # noqa: E402
    V34CandidateError,
    apply_v34_candidate,
    policy_snapshot as v34_policy_snapshot,
    policy_snapshot_sha256 as v34_policy_snapshot_sha256,
)


V33_RUNNER_PATH = REPO_ROOT / "scratch" / "v33_full_140_20260826" / "run_v33_from_v3.py"
SPEC = importlib.util.spec_from_file_location("_v34_v33_runner_contract", V33_RUNNER_PATH)
assert SPEC and SPEC.loader
V33_RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = V33_RUNNER
SPEC.loader.exec_module(V33_RUNNER)
_B = V33_RUNNER._B

SCHEMA_VERSION = 1
REAL_PARTITION_COUNT = 140
REAL_STRICT_VALID_TOTAL = 831_531_565
CONTEXT_PIXELS = V33_RUNNER.CONTEXT_PIXELS


class V34RunError(RuntimeError):
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
    return _B._read_json(path)


def _atomic_json(path: Path, value: Any) -> None:
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


def _save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".npy",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.save(handle, np.ascontiguousarray(values), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _class_indices(codes: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.full(codes.shape, -1, dtype=np.int16)
    for index, code in enumerate(CLASS_ORDER):
        result[valid & (codes == int(code))] = int(index)
    if np.any(valid & (result < 0)):
        raise V34RunError("V3.3 publication contains a class outside CLASS_ORDER")
    return result


def _load_v33(path: Path, *, self_test: bool) -> dict[str, Any]:
    path = path.resolve()
    manifest = _read(path)
    declared = manifest.get("manifest_sha256")
    if not isinstance(declared, str) or declared != _sha_json(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    ):
        raise V34RunError("V3.3 manifest self SHA-256 mismatch")
    expected_count = 4 if self_test else REAL_PARTITION_COUNT
    if (
        manifest.get("status") != "complete"
        or manifest.get("candidate_label") != "V3.3"
        or int(manifest.get("completed_partition_count", -1)) != expected_count
    ):
        raise V34RunError(f"V3.3 parent must be complete with {expected_count} Cores")
    if not self_test and int(manifest.get("frozen_v3_class_pixel_totals_sum", -1)) != REAL_STRICT_VALID_TOTAL:
        raise V34RunError("V3.3 parent strict-valid total differs")
    parent_path = Path(str(manifest["parent_v31a_manifest"])).resolve()
    source = _B._load_parent(parent_path, self_test=self_test)
    if _sha_file(parent_path) != manifest.get("parent_v31a_manifest_sha256"):
        raise V34RunError("V3.3 parent V3.1-A manifest SHA differs")
    raw_parts = manifest.get("partitions")
    if not isinstance(raw_parts, list):
        raise V34RunError("V3.3 partitions are missing")
    parts = {str(item.get("partition_id", "")): item for item in raw_parts if isinstance(item, dict)}
    expected_ids = {str(entry["partition_id"]) for entry in source["entries"]}
    if set(parts) != expected_ids:
        raise V34RunError("V3.3 partition identities differ from frozen V3 parent")
    by_entry = {str(entry["partition_id"]): entry for entry in source["entries"]}
    for partition_id, part in parts.items():
        entry = by_entry[partition_id]
        for key in ("raw", "v3", "valid"):
            left = (part.get("outputs") or {}).get(key) or {}
            right = (entry.get("outputs") or {}).get(key) or {}
            if left.get("sha256") != right.get("sha256"):
                raise V34RunError(f"{partition_id}: V3.3 {key} lineage differs")
        output = (part.get("outputs") or {}).get("v33") or {}
        output_path = Path(str(output.get("path", "")))
        _B._verify_npy(
            output_path, str(output.get("sha256", "")),
            _B._shape(entry["core_window"]), np.dtype("int16"),
            f"V3.3 parent {partition_id}",
        )
        audit = part.get("audit") or {}
        audit_path = Path(str(audit.get("path", "")))
        if not audit_path.is_file() or _sha_file(audit_path) != audit.get("sha256"):
            raise V34RunError(f"{partition_id}: V3.3 audit SHA mismatch")
    return {
        "path": str(path),
        "sha": _sha_file(path),
        "manifest": manifest,
        "parts": parts,
        "source": source,
    }


def _stitch_v33(
    target: Mapping[str, Any], entries: list[dict[str, Any]],
    v33_parts: Mapping[str, Mapping[str, Any]], global_window: Mapping[str, int],
) -> tuple[np.ndarray, dict[str, int]]:
    expanded = _B._expand(target["core_window"], global_window)
    output = np.full(_B._shape(expanded), -1, dtype=np.int16)
    coverage = np.zeros(output.shape, dtype=np.uint8)
    for owner in entries:
        selected = _B._intersect(expanded, owner["core_window"])
        if selected is None:
            continue
        destination = _B._slices(expanded, selected)
        owner_slice = _B._slices(owner["core_window"], selected)
        part = v33_parts[str(owner["partition_id"])]
        codes = np.load(
            part["outputs"]["v33"]["path"], mmap_mode="r", allow_pickle=False,
        )[owner_slice]
        strict = np.load(
            part["outputs"]["valid"]["path"], mmap_mode="r", allow_pickle=False,
        )[owner_slice]
        indices = _class_indices(codes, strict)
        output[destination][strict] = indices[strict]
        coverage[destination] += 1
    if not np.all(coverage == 1):
        raise V34RunError(f"{target['partition_id']}: V3.3 expanded Core coverage is not exact")
    return output, expanded


def _round1_ledger(part: Mapping[str, Any]) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    v3 = np.load(part["outputs"]["v3"]["path"], mmap_mode="r", allow_pickle=False)
    v33 = np.load(part["outputs"]["v33"]["path"], mmap_mode="r", allow_pickle=False)
    valid = np.load(part["outputs"]["valid"]["path"], mmap_mode="r", allow_pickle=False)
    changed = valid & (v3 != v33)
    source = {int(code): int(np.count_nonzero(changed & (v3 == int(code)))) for code in CLASS_ORDER}
    target = {int(code): int(np.count_nonzero(changed & (v33 == int(code)))) for code in CLASS_ORDER}
    stage = _read(Path(str(part["audit"]["path"])))
    candidate = stage.get("v33_audit") or {}
    per_class = candidate.get("per_class") or {}
    if not per_class:
        if np.any(changed) or int(np.count_nonzero(valid)) != 0:
            raise V34RunError(f"{part['partition_id']}: V3.3 budget audit is missing")
        return source, target, {int(code): 0 for code in CLASS_ORDER}
    accepted_bridge: Counter[int] = Counter()
    protected = set(int(code) for code in candidate["policy_snapshot"]["config_policy"]["constraints"]["budgets"]["protected_source_codes"])
    for item in candidate.get("accepted") or []:
        if item.get("kind") == "same_class_bridge":
            accepted_bridge[int(item["target_class_code"])] += int(item["changed_pixels"])
    bridge: dict[int, int] = {}
    for code in CLASS_ORDER:
        row = per_class.get(str(int(code))) or {}
        if source[int(code)] != int(row.get("source_loss", -1)) or target[int(code)] != int(row.get("target_gain", -1)):
            raise V34RunError(f"{part['partition_id']}: V3.3 transition ledger differs for class {code}")
        if accepted_bridge[int(code)] != int(row.get("bridge_gain", -1)):
            raise V34RunError(f"{part['partition_id']}: V3.3 bridge ledger differs for class {code}")
        bridge[int(code)] = accepted_bridge[int(code)] if int(code) in protected else 0
    return source, target, bridge


def _stage(root: Path, partition_id: str) -> Path:
    return root / "partitions" / partition_id / "stage_v34"


def _validate_stage(
    root: Path, entry: Mapping[str, Any], fingerprint: str,
) -> dict[str, Any]:
    directory = _stage(root, str(entry["partition_id"]))
    audit_path = directory / "audit.json"
    hashes_path = directory / "outputs_sha256.json"
    if not audit_path.is_file() or not hashes_path.is_file():
        raise V34RunError(f"resume V3.4 stage incomplete: {directory}")
    audit = _read(audit_path)
    hashes = _read(hashes_path).get("files")
    if (
        audit.get("execution_fingerprint_sha256") != fingerprint
        or not isinstance(hashes, dict)
        or set(hashes) != {"v34_core.npy", "audit.json"}
        or any(_sha_file(directory / name) != digest for name, digest in hashes.items())
    ):
        raise V34RunError(f"resume V3.4 stage fingerprint/SHA differs: {directory}")
    _B._verify_npy(
        directory / "v34_core.npy", hashes["v34_core.npy"],
        _B._shape(entry["core_window"]), np.dtype("int16"),
        f"resume V3.4 {entry['partition_id']}",
    )
    return audit


def _run_one(job: Mapping[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    entry = job["entry"]
    partition_id = str(entry["partition_id"])
    original, probabilities, decoder_valid, strict, expanded, _owners = _B._stitch(
        entry, job["entries"], job["global_window"],
    )
    baseline, baseline_expanded = _stitch_v33(
        entry, job["entries"], job["v33_parts"], job["global_window"],
    )
    if baseline_expanded != expanded:
        raise V34RunError(f"{partition_id}: V3/V3.3 expanded windows differ")
    baseline[decoder_valid & ~strict] = original[decoder_valid & ~strict]
    if np.any(decoder_valid & (baseline < 0)):
        raise V34RunError(f"{partition_id}: stitched V3.3 decoder context is incomplete")
    immutable = decoder_valid & (baseline != original)
    core_slice = _B._slices(expanded, entry["core_window"])
    budget = np.zeros(decoder_valid.shape, dtype=bool)
    budget[core_slice] = strict[core_slice]
    source, target, bridge = _round1_ledger(job["v33_parts"][partition_id])
    metrics = _B._physical(job["transform"], job["crs"], expanded)
    if np.any(budget):
        try:
            result, candidate = apply_v34_candidate(
                baseline,
                original_v3_labels=original,
                round1_immutable_mask=immutable,
                round1_source_loss_pixels=source,
                round1_target_gain_pixels=target,
                round1_protected_bridge_gain_pixels=bridge,
                class_codes=CLASS_ORDER,
                pixel_area_m2=metrics["pixel_area_m2"],
                pixel_size_m=(metrics["row_step_m"], metrics["column_step_m"]),
                valid_mask=decoder_valid,
                class_budget_mask=budget,
                probabilities=probabilities,
                confidence=probabilities.max(axis=0).astype(np.float32),
                policy_document=job["policy_document"],
                baseline_kind="v33_cleaned",
                full_audit=True,
            )
        except V34CandidateError as exc:
            raise V34RunError(f"{partition_id}: V3.4 candidate rejected input: {exc}") from exc
    else:
        result = baseline.copy()
        candidate = {
            "candidate_label": "V3.4", "full_audit": True,
            "audit_truncated": False, "raw_generated": 0,
            "proposals_canonical": 0, "duplicate_proposal_count": 0,
            "proposals_accepted": 0, "changed_pixel_count": 0,
            "protected_source_loss_pixel_count": 0, "gap_pixels": 0,
            "overlap_pixels": 0, "outside_pixels": 0,
            "proposal_generation_reject_reason_counts": {},
            "proposal_reject_reason_counts": {},
        }
    if not candidate.get("full_audit") or candidate.get("audit_truncated"):
        raise V34RunError(f"{partition_id}: V3.4 audit is incomplete")
    core_valid = strict[core_slice]
    indices = result[core_slice]
    output = np.full(core_valid.shape, -1, dtype=np.int16)
    output[core_valid] = np.asarray(CLASS_ORDER, dtype=np.int16)[indices[core_valid]]
    v33_core = np.load(
        job["v33_parts"][partition_id]["outputs"]["v33"]["path"],
        mmap_mode="r", allow_pickle=False,
    )
    if np.any(output[~core_valid] != -1) or np.any(~np.isin(output[core_valid], CLASS_ORDER)):
        raise V34RunError(f"{partition_id}: V3.4 output violates label coverage")
    directory = _stage(Path(job["output_root"]), partition_id)
    if directory.exists():
        raise V34RunError(f"refusing to overwrite existing V3.4 stage: {directory}")
    staging = Path(job["staging_root"]) / partition_id
    staging.mkdir(parents=True, exist_ok=False)
    _save_npy(staging / "v34_core.npy", output)
    output_sha = _sha_file(staging / "v34_core.npy")
    core_transform = list(tuple(
        Affine(*job["transform"])
        * Affine.translation(entry["core_window"]["x0"], entry["core_window"]["y0"])
    )[:6])
    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "v34",
        "candidate_label": "V3.4",
        "partition_id": partition_id,
        "execution_fingerprint_sha256": job["fingerprint"],
        "parent_v33_manifest": job["v33_path"],
        "parent_v33_manifest_sha256": job["v33_sha"],
        "parent_v33_partition_audit": job["v33_parts"][partition_id]["audit"],
        "global_core_window": entry["core_window"],
        "global_expanded_window": expanded,
        "core_transform": core_transform,
        "crs": job["crs"],
        "physical_metrics": metrics,
        "v34_audit": candidate,
        "round2_changed_pixel_count": int(np.count_nonzero(core_valid & (output != v33_core))),
        "runtime_seconds": time.monotonic() - started,
        "coverage": {
            "published_strict_core_only": True,
            "core_strict_valid_pixel_count": int(core_valid.sum()),
            "gap_pixels": 0, "overlap_pixels": 0, "outside_pixels": 0,
        },
        "outputs": {
            "v34": {
                "candidate_label": "V3.4", "path": "v34_core.npy",
                "sha256": output_sha, "shape": list(output.shape), "dtype": "int16",
            }
        },
    }
    audit["audit_sha256"] = _sha_json(audit)
    _atomic_json(staging / "audit.json", audit)
    hashes = {"v34_core.npy": output_sha, "audit.json": _sha_file(staging / "audit.json")}
    _atomic_json(staging / "outputs_sha256.json", {"files": hashes})
    directory.parent.mkdir(parents=True, exist_ok=True)
    os.rename(staging, directory)
    return {
        "partition_id": partition_id,
        "global_core_window": entry["core_window"],
        "core_transform": core_transform,
        "crs": job["crs"],
        "physical_metrics": metrics,
        "owner_core_pixel_count": int(core_valid.size),
        "valid_pixel_count": int(core_valid.sum()),
        "runtime_seconds": audit["runtime_seconds"],
        "candidate_metrics": {
            key: int(candidate.get(key, 0)) for key in (
                "changed_pixel_count", "protected_source_loss_pixel_count",
                "gap_pixels", "overlap_pixels", "outside_pixels",
            )
        },
        "proposal_counts": {
            "raw_generated": int(candidate.get("raw_generated", 0)),
            "canonical_generated": int(candidate.get("proposals_canonical", 0)),
            "duplicate_proposal_count": int(candidate.get("duplicate_proposal_count", 0)),
            "canonical_accepted": int(candidate.get("proposals_accepted", 0)),
            "canonical_rejected": int(candidate.get("proposals_canonical", 0)) - int(candidate.get("proposals_accepted", 0)),
        },
        "outputs": {
            "raw": job["v33_parts"][partition_id]["outputs"]["raw"],
            "v3": job["v33_parts"][partition_id]["outputs"]["v3"],
            "valid": job["v33_parts"][partition_id]["outputs"]["valid"],
            "v33": job["v33_parts"][partition_id]["outputs"]["v33"],
            "v34": {**audit["outputs"]["v34"], "path": str((directory / "v34_core.npy").resolve())},
            "v31a": {
                **audit["outputs"]["v34"],
                "path": str((directory / "v34_core.npy").resolve()),
                "compatibility_role": "global_evaluator_candidate_alias",
            },
        },
        "audit": {"path": str((directory / "audit.json").resolve()), "sha256": hashes["audit.json"]},
    }


def _resume_result(root: Path, entry: Mapping[str, Any], audit: Mapping[str, Any], part: Mapping[str, Any]) -> dict[str, Any]:
    directory = _stage(root, str(entry["partition_id"]))
    candidate = audit["v34_audit"]
    output = audit["outputs"]["v34"]
    partition_id = str(entry["partition_id"])
    return {
        "partition_id": partition_id,
        "global_core_window": entry["core_window"],
        "core_transform": audit["core_transform"],
        "crs": audit["crs"],
        "physical_metrics": audit["physical_metrics"],
        "owner_core_pixel_count": int(np.prod(_B._shape(entry["core_window"]))),
        "valid_pixel_count": int(audit["coverage"]["core_strict_valid_pixel_count"]),
        "runtime_seconds": float(audit["runtime_seconds"]),
        "candidate_metrics": {key: int(candidate.get(key, 0)) for key in ("changed_pixel_count", "protected_source_loss_pixel_count", "gap_pixels", "overlap_pixels", "outside_pixels")},
        "proposal_counts": {
            "raw_generated": int(candidate.get("raw_generated", 0)),
            "canonical_generated": int(candidate.get("proposals_canonical", 0)),
            "duplicate_proposal_count": int(candidate.get("duplicate_proposal_count", 0)),
            "canonical_accepted": int(candidate.get("proposals_accepted", 0)),
            "canonical_rejected": int(candidate.get("proposals_canonical", 0)) - int(candidate.get("proposals_accepted", 0)),
        },
        "outputs": {
            "raw": part["outputs"]["raw"], "v3": part["outputs"]["v3"],
            "valid": part["outputs"]["valid"], "v33": part["outputs"]["v33"],
            "v34": {**output, "path": str((directory / "v34_core.npy").resolve())},
            "v31a": {**output, "path": str((directory / "v34_core.npy").resolve()), "compatibility_role": "global_evaluator_candidate_alias"},
        },
        "audit": {"path": str((directory / "audit.json").resolve()), "sha256": _sha_file(directory / "audit.json")},
    }


def _code_sha() -> dict[str, str]:
    paths = {
        Path(__file__).resolve(), V33_RUNNER_PATH.resolve(),
        REPO_ROOT / "inference_scripts" / "deployment_config.py",
        REPO_ROOT / "inference_scripts" / "fragmentation_v33_candidate" / "candidate.py",
        REPO_ROOT / "inference_scripts" / "fragmentation_v34_candidate" / "candidate.py",
    }
    paths.update((REPO_ROOT / "inference_scripts" / "fragmentation_policy").rglob("*.py"))
    paths.update((REPO_ROOT / "inference_scripts" / "fragmentation_policy").rglob("*.yaml"))
    return {str(path.relative_to(REPO_ROOT)): _sha_file(path) for path in sorted(paths)}


def run(
    v33_manifest: Path, output_root: Path, *, workers: int,
    resume: bool, self_test: bool = False,
) -> dict[str, Any]:
    if workers < 1 or workers > 2:
        raise V34RunError("--workers must be 1 or 2")
    data = _load_v33(v33_manifest, self_test=self_test)
    source = data["source"]
    if output_root.exists() and any(output_root.iterdir()) and not resume:
        raise V34RunError(f"refusing non-empty output root without --resume: {output_root}")
    policy_document = load_policy()
    snapshot = v34_policy_snapshot(policy_document)
    policy_sha = v34_policy_snapshot_sha256(policy_document)
    config_sha = config_policy_sha256(policy_document)
    code = _code_sha()
    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION,
        "candidate_label": "V3.4",
        "parent_v33_manifest_sha256": data["sha"],
        "parent_v31a_manifest_sha256": data["manifest"]["parent_v31a_manifest_sha256"],
        "v34_policy_snapshot_sha256": policy_sha,
        "v33_config_policy_sha256": config_sha,
        "code_sha256": code,
    }
    fingerprint = _sha_json(fingerprint_payload)
    manifest_path = output_root / "run_manifest.json"
    if resume and manifest_path.is_file():
        prior = _read(manifest_path)
        if prior.get("execution_fingerprint_sha256") != fingerprint:
            raise V34RunError("resume V3.4 execution fingerprint differs")
        if prior.get("status") == "complete":
            if prior.get("manifest_sha256") != _sha_json({key: value for key, value in prior.items() if key != "manifest_sha256"}):
                raise V34RunError("completed V3.4 manifest self SHA mismatch")
            for part in prior.get("partitions") or []:
                for reference in (part["outputs"]["v34"], part["audit"]):
                    path = Path(reference["path"])
                    if not path.is_file() or _sha_file(path) != reference["sha256"]:
                        raise V34RunError(f"completed V3.4 artifact SHA mismatch: {path}")
            return prior
    output_root.mkdir(parents=True, exist_ok=True)
    completed: dict[str, Any] = {}
    if resume:
        for entry in source["entries"]:
            directory = _stage(output_root, str(entry["partition_id"]))
            if directory.exists():
                audit = _validate_stage(output_root, entry, fingerprint)
                completed[str(entry["partition_id"])] = _resume_result(
                    output_root, entry, audit, data["parts"][str(entry["partition_id"])],
                )
    running: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "v34_bounded_second_pass_candidate",
        "candidate_label": "V3.4",
        "status": "running",
        "self_test": bool(self_test),
        "parent_v33_manifest": data["path"],
        "parent_v33_manifest_sha256": data["sha"],
        "parent_v31a_manifest": data["manifest"]["parent_v31a_manifest"],
        "parent_v31a_manifest_sha256": data["manifest"]["parent_v31a_manifest_sha256"],
        "snapshot_manifest": data["manifest"]["snapshot_manifest"],
        "snapshot_manifest_sha256": data["manifest"]["snapshot_manifest_sha256"],
        "v3_policy_snapshot": data["manifest"]["v3_policy_snapshot"],
        "v3_policy_snapshot_sha256": data["manifest"]["v3_policy_snapshot_sha256"],
        "v34_policy_snapshot": snapshot,
        "v34_policy_snapshot_sha256": policy_sha,
        "v33_config_policy_sha256": config_sha,
        "execution_fingerprint": fingerprint_payload,
        "execution_fingerprint_sha256": fingerprint,
        "code_sha256": code,
        "class_codes": list(CLASS_ORDER),
        "label_encoding": "class_codes_int16_invalid_minus_one",
        "processing_transform": source["transform"],
        "crs": source["crs"],
        "global_window": source["global_window"],
        "approved_dynamic_mmu_m2": {
            code: float(row["fragment_max_m2"])
            for code, row in policy_document["classes"].items()
        },
        "frozen_v3_class_pixel_totals": data["manifest"]["frozen_v3_class_pixel_totals"],
        "frozen_v3_class_pixel_totals_sum": data["manifest"]["frozen_v3_class_pixel_totals_sum"],
        "requested_partition_count": len(source["entries"]),
        "completed_partition_count": len(completed),
        "partitions": [completed[key] for key in sorted(completed)],
        "resource_plan": {"workers": workers, "real_workers_hard_limit": 2, "V3_and_V33_execution": "forbidden_reused_parent_outputs_only"},
    }
    _atomic_json(manifest_path, running)
    staging_root = output_root.parent / f".{output_root.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    staging_root.mkdir(exist_ok=False)
    payload = {
        "entries": source["entries"], "v33_parts": data["parts"],
        "global_window": source["global_window"], "transform": source["transform"],
        "crs": source["crs"], "policy_document": policy_document,
        "fingerprint": fingerprint, "output_root": str(output_root.resolve()),
        "staging_root": str(staging_root.resolve()), "v33_path": data["path"],
        "v33_sha": data["sha"],
    }
    try:
        pending = [entry for entry in source["entries"] if str(entry["partition_id"]) not in completed]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_one, {**payload, "entry": entry}): entry["partition_id"] for entry in pending}
            for future in as_completed(futures):
                item = future.result()
                completed[item["partition_id"]] = item
                running["completed_partition_count"] = len(completed)
                running["partitions"] = [completed[key] for key in sorted(completed)]
                _atomic_json(manifest_path, running)
                print(f"stage_v34 {len(completed)}/{len(source['entries'])} {item['partition_id']}", flush=True)
    finally:
        for child in staging_root.glob("*"):
            if child.is_dir():
                import shutil
                shutil.rmtree(child, ignore_errors=True)
        staging_root.rmdir()
    parts = [completed[key] for key in sorted(completed)]
    if len(parts) != len(source["entries"]):
        raise V34RunError("V3.4 run incomplete")
    count_keys = ("raw_generated", "canonical_generated", "duplicate_proposal_count", "canonical_accepted", "canonical_rejected")
    metric_keys = ("changed_pixel_count", "protected_source_loss_pixel_count", "gap_pixels", "overlap_pixels", "outside_pixels")
    running.update({
        "status": "complete",
        "completed_partition_count": len(parts),
        "partitions": parts,
        "coverage": {
            "all_snapshot_partitions_requested": len(parts) == (4 if self_test else REAL_PARTITION_COUNT),
            "core_windows_nonoverlapping": True, "global_core_grid_exact": True,
            "complete": True,
            "published_core_pixel_count": int(sum(item["owner_core_pixel_count"] for item in parts)),
            "published_valid_pixel_count": int(sum(item["valid_pixel_count"] for item in parts)),
        },
        "proposal_counts": {key: int(sum(item["proposal_counts"][key] for item in parts)) for key in count_keys},
        "candidate_metrics": {key: int(sum(item["candidate_metrics"][key] for item in parts)) for key in metric_keys},
        "runtime_seconds": float(sum(item["runtime_seconds"] for item in parts)),
    })
    running["manifest_sha256"] = _sha_json(running)
    _atomic_json(manifest_path, running)
    return running


def _self_test(output_root: Path | None, workers: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v34-selftest-") as temporary:
        root = Path(temporary)
        parent = _B._write_self_test_parent(root)
        v33_root = root / "v33"
        V33_RUNNER.run(parent, v33_root, workers=workers, resume=False, self_test=True)
        target = output_root or root / "v34"
        first = run(v33_root / "run_manifest.json", target, workers=workers, resume=False, self_test=True)
        resumed = run(v33_root / "run_manifest.json", target, workers=workers, resume=True, self_test=True)
        if first["manifest_sha256"] != resumed["manifest_sha256"] or resumed["completed_partition_count"] != 4:
            raise V34RunError("V3.4 self-test resume/publication contract failed")
        return resumed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v33-manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.self_test:
            result = _self_test(args.output_root, args.workers)
        elif args.v33_manifest is None or args.output_root is None:
            parser.error("real execution requires --v33-manifest and --output-root")
        else:
            result = run(args.v33_manifest, args.output_root, workers=args.workers, resume=args.resume)
        print(json.dumps({"status": result["status"], "candidate_label": "V3.4"}), flush=True)
        return 0
    except (V34RunError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
