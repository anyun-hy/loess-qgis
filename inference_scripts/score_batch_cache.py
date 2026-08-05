"""Durable, resumable probability checkpoints for one model batch."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CLASS_COUNT = 14
CHECKPOINT_WRITE_OVERHEAD_BYTES = 1024 * 1024


class ScoreBatchCacheError(RuntimeError):
    pass


class ScoreBatchDiskReserveError(ScoreBatchCacheError):
    def __init__(self, message: str, *, transient: bool = False) -> None:
        self.transient = bool(transient)
        super().__init__(str(message))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _checkpoint_paths(root: Path, sequence: int) -> tuple[Path, Path]:
    stem = f"batch_{int(sequence):06d}"
    return root / f"{stem}.npy", root / f"{stem}.json"


def _entry(item: Mapping[str, Any]) -> dict[str, Any]:
    tile = item["tile"]
    return {
        "tile_id": str(tile["tile_id"]),
        "tile_index": int(item["tile_index"]),
        "input_sha256": str(tile["sha256"]),
        "row": int(tile["row_no"]),
        "col": int(tile["col_no"]),
        "width": int(tile["width"]),
        "height": int(tile["height"]),
    }


def _expected_entries(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_entry(item) for item in items]


def _records(
    data_path: Path,
    manifest_path: Path,
    entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "tile_id": str(entry["tile_id"]),
            "row": int(entry["row"]),
            "col": int(entry["col"]),
            "width": int(entry["width"]),
            "height": int(entry["height"]),
            "score_batch_path": str(data_path),
            "score_batch_index": index,
            "metadata_path": str(manifest_path),
            "cache_kind": "model_batch",
        }
        for index, entry in enumerate(entries)
    ]


def _owned_regular_file(path: Path, root: Path) -> bool:
    if path.is_symlink() or root.is_symlink():
        return False
    try:
        return path.is_file() and path.resolve().parent == root.resolve()
    except OSError:
        return False


def discard_checkpoint(root: Path, sequence: int) -> int:
    """Delete only the two exact files owned by this model checkpoint."""
    if root.is_symlink():
        raise ScoreBatchCacheError(
            f"refusing to delete through symlinked checkpoint root: {root}"
        )
    removed = 0
    for path in _checkpoint_paths(root, sequence):
        if path.is_symlink():
            raise ScoreBatchCacheError(f"refusing to delete symlinked checkpoint: {path}")
        if path.is_file() and path.resolve().parent == root.resolve():
            removed += path.stat().st_size
            path.unlink()
    return removed


def remove_owned_temporary_files(root: Path) -> int:
    if not root.is_dir() or root.is_symlink():
        return 0
    removed = 0
    for path in root.iterdir():
        if not path.name.startswith(".") or not path.name.endswith(".tmp"):
            continue
        if path.is_symlink():
            raise ScoreBatchCacheError(f"refusing to delete symlinked temporary file: {path}")
        if path.is_file() and path.resolve().parent == root.resolve():
            removed += path.stat().st_size
            path.unlink()
    return removed


def load_checkpoint(
    root: Path,
    *,
    run_id: str,
    package_id: str,
    model_id: str,
    model_sha256: str,
    sequence: int,
    items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]] | None:
    data_path, manifest_path = _checkpoint_paths(root, sequence)
    if not _owned_regular_file(data_path, root) or not _owned_regular_file(
        manifest_path, root
    ):
        return None
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        entries = _expected_entries(items)
        expected_identity = {
            "schema_version": 1,
            "format": "npy_float16_probability_batch",
            "run_id": str(run_id),
            "package_id": str(package_id),
            "model_id": str(model_id),
            "model_sha256": str(model_sha256),
            "sequence": int(sequence),
            "entries": entries,
        }
        if any(manifest.get(key) != value for key, value in expected_identity.items()):
            return None
        if manifest.get("data_file") != data_path.name:
            return None
        if int(manifest.get("byte_count", -1)) != data_path.stat().st_size:
            return None
        if str(manifest.get("sha256") or "") != _sha256_file(data_path):
            return None
        probabilities = np.load(data_path, mmap_mode="r", allow_pickle=False)
        try:
            expected_shape = (len(entries), CLASS_COUNT, 512, 512)
            if probabilities.dtype != np.float16 or probabilities.shape != expected_shape:
                return None
        finally:
            mmap = getattr(probabilities, "_mmap", None)
            if mmap is not None:
                mmap.close()
        return _records(data_path, manifest_path, entries)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def write_checkpoint(
    root: Path,
    *,
    run_id: str,
    package_id: str,
    model_id: str,
    model_sha256: str,
    sequence: int,
    items: Sequence[Mapping[str, Any]],
    probabilities: np.ndarray,
    min_free_bytes: int = 0,
    additional_free_reserve_bytes: int = 0,
    managed_cache_bytes: int = 0,
    managed_cache_budget_bytes: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ScoreBatchCacheError(f"checkpoint root must not be a symlink: {root}")
    array = np.ascontiguousarray(probabilities, dtype=np.float16)
    expected_shape = (len(items), CLASS_COUNT, 512, 512)
    if array.shape != expected_shape:
        raise ScoreBatchCacheError(
            f"checkpoint probability shape must be {expected_shape}, got {array.shape}"
        )
    estimated_write_bytes = int(array.nbytes) + CHECKPOINT_WRITE_OVERHEAD_BYTES
    projected_managed_cache = int(managed_cache_bytes) + estimated_write_bytes
    if (
        managed_cache_budget_bytes is not None
        and projected_managed_cache > int(managed_cache_budget_bytes)
    ):
        raise ScoreBatchDiskReserveError(
            "probability checkpoint exceeds frozen score-cache high-water: "
            f"projected={projected_managed_cache}, "
            f"budget={int(managed_cache_budget_bytes)}",
            transient=False,
        )
    required_free = (
        int(min_free_bytes)
        + int(additional_free_reserve_bytes)
        + estimated_write_bytes
    )
    actual_free = int(shutil.disk_usage(root).free)
    if actual_free < required_free:
        raise ScoreBatchDiskReserveError(
            "insufficient disk for probability checkpoint: "
            f"free={actual_free}, required={required_free}",
            transient=True,
        )
    data_path, manifest_path = _checkpoint_paths(root, sequence)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{data_path.name}.", suffix=".tmp", dir=root
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, data_path)
        _fsync_directory(root)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    entries = _expected_entries(items)
    manifest = {
        "schema_version": 1,
        "format": "npy_float16_probability_batch",
        "run_id": str(run_id),
        "package_id": str(package_id),
        "model_id": str(model_id),
        "model_sha256": str(model_sha256),
        "sequence": int(sequence),
        "data_file": data_path.name,
        "byte_count": data_path.stat().st_size,
        "sha256": _sha256_file(data_path),
        "dtype": "float16",
        "shape": list(array.shape),
        "entries": entries,
    }
    _atomic_json(manifest_path, manifest)
    return _records(data_path, manifest_path, entries), manifest
