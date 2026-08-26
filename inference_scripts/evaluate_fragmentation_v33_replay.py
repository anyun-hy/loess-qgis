#!/usr/bin/env python3
"""Strictly compare an isolated Work-Package replay with historical V3.3."""

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
import rasterio


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for plugin_root in (ROOT / "qgis_plugins", ROOT / "runtime"):
    if plugin_root.is_dir() and str(plugin_root) not in sys.path:
        sys.path.insert(0, str(plugin_root))

from deployment_config import CLASS_ORDER
from labeling_tool.core.run_spec import sha256_file


PARTITION_COUNT = 140


class ReplayEvaluationError(RuntimeError):
    """Raised when replay outputs cannot prove exact historical equivalence."""


def _canonical_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayEvaluationError(f"cannot read {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReplayEvaluationError(f"{label} root must be an object: {path}")
    return value


def _verified_manifest(path: Path, label: str) -> dict[str, Any]:
    value = _read_json(path, label)
    expected = str(value.get("manifest_sha256") or "")
    if len(expected) != 64 or _canonical_sha({key: item for key, item in value.items() if key != "manifest_sha256"}) != expected:
        raise ReplayEvaluationError(f"{label} manifest self SHA-256 mismatch")
    if value.get("status") != "complete" or int(value.get("completed_partition_count", -1)) != PARTITION_COUNT:
        raise ReplayEvaluationError(f"{label} is not a complete 140-Core manifest")
    return value


def _resolve(base: Path, value: Any, label: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ReplayEvaluationError(f"{label} path is empty")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _parts(manifest: Mapping[str, Any], label: str) -> dict[str, dict[str, Any]]:
    raw = manifest.get("partitions")
    if not isinstance(raw, list):
        raise ReplayEvaluationError(f"{label} partitions are missing")
    parts = {str(item.get("partition_id") or ""): dict(item) for item in raw if isinstance(item, Mapping)}
    if len(parts) != PARTITION_COUNT or "" in parts:
        raise ReplayEvaluationError(f"{label} partition set is not exactly 140 unique IDs")
    return parts


def _output_reference(
    manifest_path: Path, part: Mapping[str, Any], keys: tuple[str, ...], label: str
) -> tuple[Path, dict[str, Any]]:
    outputs = part.get("outputs") or {}
    if not isinstance(outputs, Mapping):
        raise ReplayEvaluationError(f"{label} outputs are missing")
    found = [key for key in keys if isinstance(outputs.get(key), Mapping)]
    if not found:
        raise ReplayEvaluationError(f"{label} lacks every output among {keys}")
    reference = dict(outputs[found[0]])
    for alias in found[1:]:
        other = outputs[alias]
        if (
            str(other.get("sha256") or "") != str(reference.get("sha256") or "")
            or str(other.get("path") or "") != str(reference.get("path") or "")
        ):
            raise ReplayEvaluationError(
                f"{label} output aliases among {found} do not match"
            )
    return _resolve(manifest_path.parent, reference.get("path"), label), reference


def _verify_historical_array(
    path: Path, reference: Mapping[str, Any], expected_shape: tuple[int, int], label: str
) -> np.ndarray:
    if not path.is_file() or sha256_file(path) != str(reference.get("sha256") or ""):
        raise ReplayEvaluationError(f"{label} SHA-256 mismatch")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.dtype != np.int16 or tuple(values.shape) != expected_shape:
        raise ReplayEvaluationError(f"{label} shape/dtype differs")
    if reference.get("shape") is not None and tuple(int(item) for item in reference["shape"]) != expected_shape:
        raise ReplayEvaluationError(f"{label} manifest shape lineage differs")
    return values


def _candidate_codes(
    path: Path,
    expected_shape: tuple[int, int],
    label: str,
    *,
    production: bool,
) -> np.ndarray:
    if not path.is_file():
        raise ReplayEvaluationError(f"{label} candidate TIFF is missing: {path}")
    with rasterio.open(path) as source:
        if source.count != 1 or (source.height, source.width) != expected_shape:
            raise ReplayEvaluationError(f"{label} candidate TIFF shape differs")
        expected_tag = str(bool(production)).lower()
        if str(source.tags().get("production_replacement") or "").lower() != expected_tag:
            raise ReplayEvaluationError(
                f"{label} V3.3 TIFF publication tag differs"
            )
        indices = source.read(1).astype(np.int16, copy=False)
    valid_indices = (indices >= 0) & (indices < len(CLASS_ORDER))
    if np.any((indices != -1) & ~valid_indices):
        raise ReplayEvaluationError(f"{label} candidate contains an invalid class index")
    codes = np.full(indices.shape, -1, dtype=np.int16)
    codes[valid_indices] = np.asarray(CLASS_ORDER, dtype=np.int16)[indices[valid_indices]]
    return codes


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def evaluate(
    run_spec_path: Path,
    v3_manifest_path: Path,
    historical_v33_manifest_path: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    run_spec_path = run_spec_path.expanduser().resolve()
    v3_manifest_path = v3_manifest_path.expanduser().resolve()
    historical_v33_manifest_path = historical_v33_manifest_path.expanduser().resolve()
    spec = _read_json(run_spec_path, "replay run spec")
    replay = spec.get("replay") or {}
    if not isinstance(replay, Mapping):
        raise ReplayEvaluationError("run spec lacks V3.3 replay lineage")
    production = bool(replay.get("production_replacement", False))
    if production == bool(replay.get("isolated", False)):
        raise ReplayEvaluationError("replay publication mode is contradictory")
    v3_manifest = _verified_manifest(v3_manifest_path, "completed V3")
    historical_v33 = _verified_manifest(historical_v33_manifest_path, "historical V3.3")
    if str(replay.get("v3_manifest_sha256") or "") != sha256_file(v3_manifest_path):
        raise ReplayEvaluationError("replay/V3 manifest lineage SHA-256 differs")
    v3_parts = _parts(v3_manifest, "completed V3")
    historical_parts = _parts(historical_v33, "historical V3.3")
    if set(v3_parts) != set(historical_parts):
        raise ReplayEvaluationError("V3/historical V3.3 partition IDs differ")
    run_dir = Path(str(spec.get("run_dir") or "")).expanduser().resolve()
    lineage_path = run_dir / "replay_lineage.json"
    lineage = _read_json(lineage_path, "replay lineage")
    input_records = {str(item.get("partition_id") or ""): item for item in lineage.get("input_records") or [] if isinstance(item, Mapping)}
    if len(input_records) != PARTITION_COUNT or set(input_records) != set(v3_parts):
        raise ReplayEvaluationError("replay lineage input records are incomplete")
    same_partitions = 0
    historical_difference_pixels = 0
    changed_from_v3 = 0
    gap_pixels = 0
    outside_pixels = 0
    details: list[dict[str, Any]] = []
    for partition_id in sorted(v3_parts):
        v3_path, v3_reference = _output_reference(v3_manifest_path, v3_parts[partition_id], ("v3", "core_mask"), f"{partition_id} V3")
        expected_v3_sha = str(v3_reference.get("sha256") or "")
        record = input_records[partition_id]
        replay_v3 = Path(str((record.get("v3_core") or {}).get("path") or "")).expanduser().resolve()
        if not replay_v3.is_file() or sha256_file(replay_v3) != expected_v3_sha:
            raise ReplayEvaluationError(f"{partition_id} replay V3 input hash changed")
        source_v3 = Path(str((record.get("v3_core") or {}).get("source_path") or "")).expanduser().resolve()
        if not source_v3.is_file() or sha256_file(source_v3) != expected_v3_sha:
            raise ReplayEvaluationError(f"{partition_id} source V3 input hash changed")
        v3_values = _verify_historical_array(v3_path, v3_reference, tuple(np.load(replay_v3, mmap_mode="r", allow_pickle=False).shape), f"{partition_id} V3")
        historical_path, historical_reference = _output_reference(historical_v33_manifest_path, historical_parts[partition_id], ("v33", "v31a"), f"{partition_id} historical V3.3")
        historical_v3_reference = (historical_parts[partition_id].get("outputs") or {}).get("v3")
        if not isinstance(historical_v3_reference, Mapping) or str(historical_v3_reference.get("sha256") or "") != expected_v3_sha:
            raise ReplayEvaluationError(f"{partition_id} historical V3.3 does not use the frozen V3 input")
        historical = _verify_historical_array(historical_path, historical_reference, tuple(v3_values.shape), f"{partition_id} historical V3.3")
        candidate_root = (
            run_dir / "fusion" / "approved_replay"
            if production
            else run_dir / "candidates" / "fragmentation_v33"
        )
        candidate = _candidate_codes(
            candidate_root / "raster_parts" / f"{partition_id}_mask.tif",
            tuple(v3_values.shape),
            partition_id,
            production=production,
        )
        valid = v3_values != -1
        difference = int(np.count_nonzero(candidate != historical))
        changed = int(np.count_nonzero(candidate[valid] != v3_values[valid]))
        gap = int(np.count_nonzero(valid & (candidate == -1)))
        outside = int(np.count_nonzero(~valid & (candidate != -1)))
        historical_difference_pixels += difference
        changed_from_v3 += changed
        gap_pixels += gap
        outside_pixels += outside
        same_partitions += int(difference == 0)
        details.append({"partition_id": partition_id, "historical_difference_pixels": difference, "v3_to_candidate_changed_pixels": changed, "gap_pixels": gap, "outside_pixels": outside})
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "exact_match" if historical_difference_pixels == 0 and gap_pixels == 0 and outside_pixels == 0 else "mismatch",
        "run_spec": str(run_spec_path),
        "v3_manifest": str(v3_manifest_path),
        "historical_v33_manifest": str(historical_v33_manifest_path),
        "partition_count": PARTITION_COUNT,
        "identical_to_historical_v33_partition_count": same_partitions,
        "historical_difference_pixels": historical_difference_pixels,
        "v3_to_candidate_changed_pixels": changed_from_v3,
        "gap_pixels": gap_pixels,
        "outside_pixels": outside_pixels,
        "production_replacement": production,
        "partitions": details,
    }
    report["report_sha256"] = _canonical_sha(report)
    destination = (
        output_path.expanduser().resolve()
        if output_path
        else (
            run_dir / "fusion" / "approved_replay" / "replay_evaluation.json"
            if production
            else run_dir / "candidates" / "fragmentation_v33" / "replay_evaluation.json"
        )
    )
    _atomic_json(destination, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate V3.3 replay against the historical 140-Core V3.3 output")
    parser.add_argument("--run-spec", required=True, type=Path)
    parser.add_argument("--v3-manifest", required=True, type=Path)
    parser.add_argument("--historical-v33-manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(evaluate(args.run_spec, args.v3_manifest, args.historical_v33_manifest, output_path=args.output), ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
