#!/usr/bin/env python3
"""Prepare, but do not run, a frozen-input V3.3 Work Package replay.

The historical 140-Core experiment stores its inputs as immutable ``.npy``
files.  This command creates a new SQLite-backed replay Run and hard-links
those exact files below the new Run directory.  It never writes to the source
snapshot or the completed V3 Run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for plugin_root in (ROOT / "qgis_plugins", ROOT / "runtime"):
    if plugin_root.is_dir() and str(plugin_root) not in sys.path:
        sys.path.insert(0, str(plugin_root))

from labeling_tool.core.run_spec import CLASS_ORDER, atomic_write_json, sha256_file
from labeling_tool.core.run_state_db import RunStateDB
from fragmentation_v33_candidate import (
    V33_POLICY_ID,
    executor_snapshot_sha256,
    policy_snapshot_sha256,
)


PARTITION_COUNT = 140
POLICY_ID = V33_POLICY_ID
PROBABILITY_ARTIFACT_NAMES = (
    "input_blended_probabilities_f32_npy",
    "blended_probabilities_f32",
    "partition_probability",
)


class ReplayPreparationError(RuntimeError):
    """Raised when an immutable replay input does not meet its contract."""


def _canonical_sha(value: Mapping[str, Any]) -> str:
    body = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayPreparationError(f"cannot read {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReplayPreparationError(f"{label} root must be an object: {path}")
    return value


def _self_verified_manifest(path: Path, label: str) -> dict[str, Any]:
    value = _read_json(path, label)
    declared = value.get("manifest_sha256")
    if not isinstance(declared, str) or len(declared) != 64:
        raise ReplayPreparationError(f"{label} lacks manifest_sha256")
    actual = _canonical_sha({key: item for key, item in value.items() if key != "manifest_sha256"})
    if actual != declared:
        raise ReplayPreparationError(f"{label} manifest self SHA-256 mismatch")
    return value


def _window(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ReplayPreparationError(f"{label} must be an object")
    try:
        result = {key: int(value[key]) for key in ("x0", "y0", "x1", "y1")}
    except (KeyError, TypeError, ValueError) as error:
        raise ReplayPreparationError(f"{label} must contain integer x0,y0,x1,y1") from error
    if result["x0"] >= result["x1"] or result["y0"] >= result["y1"]:
        raise ReplayPreparationError(f"{label} is empty or reversed: {result}")
    return result


def _shape(window: Mapping[str, int]) -> tuple[int, int]:
    return (int(window["y1"]) - int(window["y0"]), int(window["x1"]) - int(window["x0"]))


def _resolve(base: Path, value: Any, label: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ReplayPreparationError(f"{label} path is empty")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _artifact_from_snapshot(
    snapshot_path: Path, partition_id: str
) -> tuple[Path, dict[str, Any]]:
    partition_root = snapshot_path.parent / "partitions" / partition_id
    partition_manifest = _read_json(partition_root / "manifest.json", f"{partition_id} manifest")
    stage_ref = (partition_manifest.get("stages") or {}).get("input")
    if not isinstance(stage_ref, Mapping):
        raise ReplayPreparationError(f"{partition_id}: input stage is missing")
    stage_path = _resolve(partition_root, stage_ref.get("path"), f"{partition_id} input stage")
    declared_stage_sha = str(stage_ref.get("sha256") or "")
    if not stage_path.is_file() or sha256_file(stage_path) != declared_stage_sha:
        raise ReplayPreparationError(f"{partition_id}: input stage SHA-256 mismatch")
    stage = _read_json(stage_path, f"{partition_id} input stage")
    artifacts = stage.get("artifacts") or {}
    if not isinstance(artifacts, Mapping):
        raise ReplayPreparationError(f"{partition_id}: input artifacts are missing")
    found = [name for name in PROBABILITY_ARTIFACT_NAMES if isinstance(artifacts.get(name), Mapping)]
    if len(found) != 1:
        raise ReplayPreparationError(
            f"{partition_id}: expected exactly one frozen probability Artifact; found {found}"
        )
    metadata = dict(artifacts[found[0]])
    return _resolve(partition_root, metadata.get("path"), f"{partition_id} probability"), metadata


def _output_reference(
    v3_manifest_path: Path, entry: Mapping[str, Any], keys: tuple[str, ...], label: str
) -> tuple[Path, dict[str, Any]]:
    outputs = entry.get("outputs") or {}
    if not isinstance(outputs, Mapping):
        raise ReplayPreparationError(f"{label}: V3 outputs are missing")
    found = [key for key in keys if isinstance(outputs.get(key), Mapping)]
    if len(found) != 1:
        raise ReplayPreparationError(f"{label}: expected exactly one output among {keys}; found {found}")
    metadata = dict(outputs[found[0]])
    return _resolve(v3_manifest_path.parent, metadata.get("path"), label), metadata


def _v3_context_reference(
    v3_manifest_path: Path,
    entry: Mapping[str, Any],
    partition_id: str,
) -> tuple[Path, dict[str, Any]]:
    audit_reference = entry.get("stage_v3_audit") or {}
    if not isinstance(audit_reference, Mapping):
        raise ReplayPreparationError(f"{partition_id}: V3 stage audit is missing")
    audit_path = _resolve(
        v3_manifest_path.parent,
        audit_reference.get("path"),
        f"{partition_id} V3 stage audit",
    )
    if (
        not audit_path.is_file()
        or sha256_file(audit_path) != str(audit_reference.get("sha256") or "")
    ):
        raise ReplayPreparationError(f"{partition_id}: V3 stage audit SHA-256 mismatch")
    audit = _read_json(audit_path, f"{partition_id} V3 stage audit")
    output = (audit.get("outputs") or {}).get("v3_context")
    if not isinstance(output, Mapping):
        raise ReplayPreparationError(f"{partition_id}: V3 context is missing")
    metadata = dict(output)
    return _resolve(
        audit_path.parent,
        metadata.get("path"),
        f"{partition_id} V3 context",
    ), metadata


def _verify_npy(
    path: Path,
    metadata: Mapping[str, Any],
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype[Any],
    label: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise ReplayPreparationError(f"{label} does not exist: {path}")
    expected_sha = str(metadata.get("sha256") or "")
    if len(expected_sha) != 64 or sha256_file(path) != expected_sha:
        raise ReplayPreparationError(f"{label} SHA-256 mismatch")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if tuple(values.shape) != tuple(expected_shape) or values.dtype != expected_dtype:
        raise ReplayPreparationError(
            f"{label} shape/dtype mismatch: got {values.shape}/{values.dtype}, "
            f"expected {expected_shape}/{expected_dtype}"
        )
    declared_shape = metadata.get("shape")
    if declared_shape is not None and tuple(int(item) for item in declared_shape) != tuple(expected_shape):
        raise ReplayPreparationError(f"{label} manifest shape lineage mismatch")
    declared_dtype = metadata.get("dtype")
    if declared_dtype is not None and np.dtype(str(declared_dtype)) != expected_dtype:
        raise ReplayPreparationError(f"{label} manifest dtype lineage mismatch")
    declared_bytes = metadata.get("byte_count")
    size = path.stat().st_size
    if declared_bytes is not None and int(declared_bytes) != size:
        raise ReplayPreparationError(f"{label} manifest byte_count lineage mismatch")
    return {"path": str(path), "sha256": expected_sha, "byte_count": size, "shape": list(expected_shape), "dtype": str(expected_dtype)}


def _hard_link(source: Path, destination: Path, label: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise ReplayPreparationError(f"{label} destination already exists: {destination}")
    if source.stat().st_dev != destination.parent.stat().st_dev:
        raise ReplayPreparationError(f"{label} cannot hard-link across filesystems")
    try:
        os.link(source, destination)
    except OSError as error:
        raise ReplayPreparationError(f"{label} hard-link failed: {error}") from error
    source_stat, destination_stat = source.stat(), destination.stat()
    if (source_stat.st_dev, source_stat.st_ino, source_stat.st_size) != (
        destination_stat.st_dev,
        destination_stat.st_ino,
        destination_stat.st_size,
    ):
        raise ReplayPreparationError(f"{label} hard-link inode lineage mismatch")


def _global_window(partitions: list[dict[str, Any]]) -> dict[str, int]:
    rows: dict[tuple[int, int], list[dict[str, int]]] = {}
    for item in partitions:
        core = item["core_window"]
        rows.setdefault((core["y0"], core["y1"]), []).append(core)
    ordered_rows = sorted(rows)
    if len(ordered_rows) < 1:
        raise ReplayPreparationError("no owner Core partitions")
    for before, after in zip(ordered_rows, ordered_rows[1:]):
        if before[1] != after[0]:
            raise ReplayPreparationError("owner Core rows contain a gap or overlap")
    columns: list[tuple[int, int]] | None = None
    for row in ordered_rows:
        spans = sorted((item["x0"], item["x1"]) for item in rows[row])
        if any(before[1] != after[0] for before, after in zip(spans, spans[1:])):
            raise ReplayPreparationError("owner Core columns contain a gap or overlap")
        if columns is None:
            columns = spans
        elif columns != spans:
            raise ReplayPreparationError("owner Core column layout differs between rows")
    assert columns is not None
    return {"x0": columns[0][0], "y0": ordered_rows[0][0], "x1": columns[-1][1], "y1": ordered_rows[-1][1]}


def prepare(
    snapshot_manifest: Path,
    v3_manifest: Path,
    output_root: Path,
    *,
    run_id: str,
    production: bool = False,
) -> dict[str, Any]:
    snapshot_manifest = snapshot_manifest.expanduser().resolve()
    v3_manifest = v3_manifest.expanduser().resolve()
    snapshot = _read_json(snapshot_manifest, "frozen snapshot")
    completed_v3 = _self_verified_manifest(v3_manifest, "completed V3 run")
    snapshot_parts = snapshot.get("partitions")
    v3_parts = completed_v3.get("partitions")
    if not isinstance(snapshot_parts, list) or not isinstance(v3_parts, list):
        raise ReplayPreparationError("snapshot/V3 manifests must contain partitions lists")
    if len(snapshot_parts) != PARTITION_COUNT or len(v3_parts) != PARTITION_COUNT:
        raise ReplayPreparationError("replay requires exactly 140 snapshot and V3 partitions")
    if completed_v3.get("status") != "complete" or int(completed_v3.get("completed_partition_count", -1)) != PARTITION_COUNT:
        raise ReplayPreparationError("completed V3 run is not a complete 140-Core result")
    snapshot_by_id = {str(item.get("partition_id") or ""): item for item in snapshot_parts if isinstance(item, Mapping)}
    v3_by_id = {str(item.get("partition_id") or ""): item for item in v3_parts if isinstance(item, Mapping)}
    if len(snapshot_by_id) != PARTITION_COUNT or "" in snapshot_by_id or set(snapshot_by_id) != set(v3_by_id):
        raise ReplayPreparationError("snapshot/V3 partition identities do not form the same 140-Core set")
    declared_snapshot_sha = str(completed_v3.get("snapshot_manifest_sha256") or "")
    actual_snapshot_sha = sha256_file(snapshot_manifest)
    if declared_snapshot_sha != actual_snapshot_sha:
        raise ReplayPreparationError("completed V3 snapshot-manifest lineage differs")
    output_root = output_root.expanduser().resolve()
    run_dir = output_root / "runs" / str(run_id)
    if run_dir.exists():
        raise ReplayPreparationError(f"refusing to overwrite replay run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        input_records: list[dict[str, Any]] = []
        partitions: list[dict[str, Any]] = []
        for ordinal, partition_id in enumerate(sorted(snapshot_by_id), start=1):
            snapshot_entry, v3_entry = snapshot_by_id[partition_id], v3_by_id[partition_id]
            core = _window(snapshot_entry.get("core_window"), f"{partition_id}.snapshot core_window")
            halo = _window(snapshot_entry.get("halo_window"), f"{partition_id}.snapshot halo_window")
            v3_core_window = _window(v3_entry.get("global_core_window") or v3_entry.get("core_window"), f"{partition_id}.V3 core_window")
            if core != v3_core_window:
                raise ReplayPreparationError(f"{partition_id}: snapshot/V3 owner Core window differs")
            probability_path, probability_meta = _artifact_from_snapshot(snapshot_manifest, partition_id)
            context_path, context_meta = _v3_context_reference(
                v3_manifest, v3_entry, partition_id
            )
            core_path, core_meta = _output_reference(
                v3_manifest, v3_entry, ("v3", "core_mask"), f"{partition_id} V3 Core"
            )
            probability = _verify_npy(probability_path, probability_meta, (len(CLASS_ORDER), *_shape(halo)), np.dtype("float32"), f"{partition_id} probability")
            context = _verify_npy(context_path, context_meta, _shape(core), np.dtype("int16"), f"{partition_id} V3 context")
            authoritative = _verify_npy(core_path, core_meta, _shape(core), np.dtype("int16"), f"{partition_id} V3 Core")
            destination_root = run_dir / "inputs" / partition_id
            probability_destination = destination_root / "probability.npy"
            context_destination = destination_root / "v3_context_core.npy"
            core_destination = destination_root / "v3_core.npy"
            _hard_link(probability_path, probability_destination, f"{partition_id} probability")
            _hard_link(context_path, context_destination, f"{partition_id} V3 context")
            _hard_link(core_path, core_destination, f"{partition_id} V3 Core")
            for record, destination in ((probability, probability_destination), (context, context_destination), (authoritative, core_destination)):
                record["source_path"] = record.pop("path")
                record["path"] = str(destination)
            input_records.append({"partition_id": partition_id, "probability": probability, "v3_context": context, "v3_core": authoritative})
            row = int(snapshot_entry.get("row", ordinal - 1))
            col = int(snapshot_entry.get("col", 0))
            partitions.append({"partition_id": partition_id, "row": row, "col": col, "core_window": core, "halo_window": halo, "package_id": f"replay_{ordinal:03d}"})
        global_window = _global_window(partitions)
        row_order = {value: index for index, value in enumerate(sorted({item["core_window"]["y0"] for item in partitions}))}
        col_order = {value: index for index, value in enumerate(sorted({item["core_window"]["x0"] for item in partitions}))}
        for item in partitions:
            item["row"] = row_order[item["core_window"]["y0"]]
            item["col"] = col_order[item["core_window"]["x0"]]
        transform = snapshot.get("processing_transform")
        if not isinstance(transform, list) or len(transform) != 6:
            raise ReplayPreparationError("frozen snapshot lacks a six-value processing_transform")
        crs = str(snapshot.get("source_raster_crs") or snapshot.get("crs") or "")
        if not crs:
            raise ReplayPreparationError("frozen snapshot lacks source_raster_crs")
        stream_id = "fusion:approved_replay" if production else "fusion_replay"
        unit_id = "fragmentation_v33" if production else "fragmentation_v33_candidate"
        fragmentation = (
            {
                "enabled": True,
                "policy_id": V33_POLICY_ID,
                "policy_version": "v33_production_20260826",
                "baseline_policy_id": "semantic_optimized_200_v3",
                "baseline_policy_version": "semantic_optimized_200_v3_core_bounded_v1",
                "publication": "authoritative_fusion_core",
                "policy_sha256": policy_snapshot_sha256(),
                "executor_sha256": executor_snapshot_sha256(),
                "buffer_pixels": 256,
            }
            if production
            else {
                "enabled": True,
                "comparison": {
                    "enabled": True,
                    "candidate_policy_id": POLICY_ID,
                    "candidate_policy_sha256": policy_snapshot_sha256(),
                    "candidate_executor_sha256": executor_snapshot_sha256(),
                    "buffer_pixels": 256,
                },
            }
        )
        spec = {
            "schema_version": 2,
            "run_id": str(run_id),
            "run_dir": str(run_dir),
            "output_root": str(output_root),
            "state_db": str(run_dir / "run_state.sqlite"),
            "raster": {"transform": [float(value) for value in transform], "crs": crs},
            "class_order": list(CLASS_ORDER),
            "partitions": partitions,
            "fragmentation_regularization": fragmentation,
            "replay": {"isolated": not production, "production_replacement": production, "execution_mode": "production_authority" if production else "isolated_replay", "snapshot_manifest": str(snapshot_manifest), "snapshot_manifest_sha256": actual_snapshot_sha, "v3_manifest": str(v3_manifest), "v3_manifest_sha256": sha256_file(v3_manifest), "input_lineage": "hard_linked_immutable_npy"},
        }
        atomic_write_json(run_dir / "run_spec.json", spec)
        database = RunStateDB(spec["state_db"])
        database.initialize()
        database.create_run(str(run_id), sha256_file(run_dir / "run_spec.json"), status="planned", metadata={"kind": "fragmentation_v33_production_replay" if production else "fragmentation_v33_replay", "production_replacement": production})
        database.register_streams(str(run_id), [{"stream_id": stream_id, "kind": "fusion" if production else "fusion_replay", "profile_id": "approved_replay" if production else "", "status": "ready"}])
        database.insert_work_packages(str(run_id), [{"package_id": item["package_id"], "sequence_no": index, "estimated_bytes": 0, "partition_ids": [item["partition_id"]], "status": "ready"} for index, item in enumerate(partitions, start=1)])
        database.insert_partitions(str(run_id), [{**item, "status": "ready"} for item in partitions])
        database.insert_spatial_units(str(run_id), [{"unit_id": unit_id, "unit_type": "FragmentationV33" if production else "FragmentationV33Candidate", "owner_key": "all_partition_owner_cores", "pixel_window": global_window, "dependency_ids": [item["partition_id"] for item in partitions], "status": "queued"}])
        database.insert_jobs(str(run_id), [{"job_type": "fragmentation_v33", "stream_id": stream_id, "unit_id": unit_id, "status": "queued", "priority": 100, "max_attempts": 3}])
        for record in input_records:
            partition_id = str(record["partition_id"])
            database.publish_fragmentation_v33_context(
                str(run_id),
                stream_id,
                partition_id,
                record["v3_context"]["path"],
                byte_count=int(record["v3_context"]["byte_count"]),
                sha256=str(record["v3_context"]["sha256"]),
            )
            probability_id = database.publish_partition_artifact(str(run_id), stream_id, partition_id, record["probability"]["path"], byte_count=int(record["probability"]["byte_count"]), sha256=str(record["probability"]["sha256"]))
            core_id = database.publish_fragmentation_v33_baseline_core(
                str(run_id),
                stream_id,
                partition_id,
                record["v3_core"]["path"],
                byte_count=int(record["v3_core"]["byte_count"]),
                sha256=str(record["v3_core"]["sha256"]),
            )
            if probability_id <= 0:
                raise ReplayPreparationError(f"{partition_id}: invalid probability Artifact id")
        report = {"schema_version": 1, "status": "prepared", "run_id": str(run_id), "run_spec": str(run_dir / "run_spec.json"), "state_db": spec["state_db"], "stream_id": stream_id, "partition_count": len(partitions), "work_package_count": len(partitions), "unit_id": unit_id, "global_core_window": global_window, "input_records": input_records, "production_replacement": production}
        report["report_sha256"] = _canonical_sha(report)
        atomic_write_json(run_dir / "replay_lineage.json", report)
        return report
    except Exception:
        # The directory is left intact for forensic inspection; it is never reused.
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a frozen-input V3.3 replay Work Package")
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--v3-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--production", action="store_true", help="exercise the authoritative production publication path")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.snapshot_manifest, args.v3_manifest, args.output_root, run_id=args.run_id, production=args.production), ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
