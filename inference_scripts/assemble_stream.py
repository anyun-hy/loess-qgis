"""Stream unit outputs into final GPKGs and assign disk-backed object IDs."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import fcntl
import json
import math
import os
import sqlite3
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import fiona
from fiona.crs import CRS
from rasterio.crs import CRS as RasterCRS
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "qgis_plugins"
RUNTIME_ROOT = ROOT / "runtime"
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from labeling_tool.core.ownership_neighbors import ownership_neighbors
from labeling_tool.core.run_state_db import RunStateDB
from labeling_tool.core.run_spec import sha256_file

from deployment_config import load_json
from difference_runtime import apply_accepted_difference
from range_clip_runtime import (
    RangeClipRuntimeError,
    apply_adaptive_range_clip,
    extract_range_mask_geometry,
)
from semantic_batch import _atomic_json
from storage_guard import StorageGuard, exact_remaining_permanent_bytes
from work_package_runtime import _commit_artifact


class StreamAssemblyError(RuntimeError):
    pass


ASSEMBLY_PHASES = (
    ("validate_inputs", "校验单元产物"),
    ("register_objects", "登记对象部件"),
    ("link_objects", "连接跨单元对象"),
    ("write_raw", "写入 Raw GPKG"),
    ("write_formal", "写入正式 GPKG"),
    ("aggregate_reports", "汇总拟合边界"),
    ("range_clip", "精确范围裁剪"),
    ("coverage_validation", "空白/重叠验收"),
    ("accepted_difference", "Accepted 差分"),
    ("publish_cleanup", "提交产物并清理中间文件"),
)
ASSEMBLY_PHASE_INDEX = {
    phase: index
    for index, (phase, _name) in enumerate(ASSEMBLY_PHASES, start=1)
}
ASSEMBLY_PHASE_NAMES = dict(ASSEMBLY_PHASES)
ASSEMBLY_PROGRESS_INTERVAL_SEC = 0.75


class _AssemblyProgress:
    """Emit and persist throttled, restart-visible Stream assembly progress."""

    def __init__(self, database: RunStateDB, run_id: str, stream_id: str):
        self.database = database
        self.run_id = str(run_id)
        self.stream_id = str(stream_id)
        self.started_at = time.monotonic()
        self._last_emit_at = 0.0
        self._last_phase = ""

    def emit(
        self,
        phase: str,
        *,
        current: int = 0,
        total: int = 0,
        feature_count: int = 0,
        status: str = "running",
        message: str = "",
        force: bool = False,
    ) -> None:
        phase_value = str(phase)
        if phase_value not in ASSEMBLY_PHASE_INDEX:
            raise StreamAssemblyError(
                f"unknown assembly progress phase: {phase_value}"
            )
        now = time.monotonic()
        total_value = max(0, int(total))
        current_value = max(0, int(current))
        if total_value:
            current_value = min(current_value, total_value)
        should_emit = (
            force
            or phase_value != self._last_phase
            or str(status) != "running"
            or (total_value > 0 and current_value >= total_value)
            or now - self._last_emit_at >= ASSEMBLY_PROGRESS_INTERVAL_SEC
        )
        if not should_emit:
            return
        event = {
            "event": "assembly_progress",
            "run_id": self.run_id,
            "stream_id": self.stream_id,
            "stage": "assembly",
            "phase": phase_value,
            "phase_name": ASSEMBLY_PHASE_NAMES[phase_value],
            "phase_index": ASSEMBLY_PHASE_INDEX[phase_value],
            "phase_total": len(ASSEMBLY_PHASES),
            "current": current_value,
            "total": total_value,
            "feature_count": max(0, int(feature_count)),
            "status": str(status),
            "message": str(message),
            "elapsed_sec": round(now - self.started_at, 3),
        }
        try:
            self.database.upsert_stream_runtime_progress(
                self.run_id,
                self.stream_id,
                stage="assembly",
                phase=phase_value,
                phase_name=event["phase_name"],
                phase_index=event["phase_index"],
                phase_total=event["phase_total"],
                current=current_value,
                total=total_value,
                feature_count=event["feature_count"],
                status=event["status"],
                message=event["message"],
            )
        except Exception as error:
            print(
                json.dumps(
                    {
                        "event": "assembly_progress_persistence_warning",
                        "run_id": self.run_id,
                        "stream_id": self.stream_id,
                        "phase": phase_value,
                        "error": str(error),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                file=sys.stderr,
                flush=True,
            )
        print(
            json.dumps(event, ensure_ascii=False, separators=(",", ":")),
            flush=True,
        )
        self._last_phase = phase_value
        self._last_emit_at = now


ASSEMBLY_VALIDATION_MAX_IN_FLIGHT = 32
OBJECT_ID_BATCH_SIZE = 512
GPKG_ATOMIC_OVERHEAD_BYTES = 4 * 1024**2
JSON_ATOMIC_OVERHEAD_BYTES = 64 * 1024
UNIT_INTERMEDIATE_KINDS = (
    "unit_raw",
    "unit_formal",
    "unit_boundary_report",
    "unit_fitted_edges",
)
UNIT_INTERMEDIATE_SUFFIXES = {
    "unit_raw": "_raw.gpkg",
    "unit_formal": "_formal.gpkg",
    "unit_boundary_report": "_report.json",
    "unit_fitted_edges": "_fitted_edges.gpkg",
}


def _strict_path_component(value: str, *, label: str) -> str:
    """Return one safe filesystem component without normalising traversal."""

    component = str(value)
    if (
        not component
        or component in {".", ".."}
        or "/" in component
        or "\\" in component
        or "\x00" in component
        or Path(component).name != component
    ):
        raise StreamAssemblyError(f"unsafe {label} in unit Artifact metadata: {value!r}")
    return component


def _cleanup_tombstone(path: Path, artifact_id: int) -> Path:
    return path.with_name(f".{path.name}.cleanup-{int(artifact_id)}.tombstone")


def _rename_cleanup_file(source: Path, tombstone: Path) -> None:
    os.rename(source, tombstone)


def _unlink_cleanup_tombstone(tombstone: Path) -> None:
    tombstone.unlink()


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes used by the cleanup transaction."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _assert_regular_cleanup_file(
    path: Path,
    artifact: Mapping[str, Any],
    *,
    stage: str,
) -> None:
    if path.is_symlink() or not path.is_file():
        raise StreamAssemblyError(
            f"unit intermediate is not a regular file during {stage}: {path}"
        )
    if (
        path.stat().st_size != int(artifact["byte_count"])
        or sha256_file(path) != str(artifact["sha256"])
    ):
        raise StreamAssemblyError(
            f"unit intermediate changed during {stage}: {path}"
        )


@contextmanager
def _unit_cleanup_lock(run_dir: Path):
    """Serialize cleanup recovery without following a forged lock symlink."""

    tmp_root = run_dir / "tmp"
    if tmp_root.is_symlink():
        raise StreamAssemblyError(
            f"refusing symlinked Run temporary directory: {tmp_root}"
        )
    tmp_root.mkdir(parents=True, exist_ok=True)
    lock_path = tmp_root / ".unit-artifact-cleanup.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise StreamAssemblyError(
            f"cannot open the unit cleanup lock safely: {lock_path}"
        ) from error
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _remaining_permanent_reserve_bytes(
    spec: Mapping[str, Any],
    database: RunStateDB,
) -> int:
    return exact_remaining_permanent_bytes(spec, database)


def _run_storage_guard(
    spec: Mapping[str, Any],
    database: RunStateDB,
    *,
    disk_usage=None,
) -> StorageGuard:
    storage = dict(spec.get("storage_preflight") or {})
    min_free_bytes = int(
        storage.get("effective_min_free_disk_bytes")
        or float((spec.get("scaling") or {}).get("min_free_disk_gb", 0.0))
        * 1024**3
    )
    remaining = _remaining_permanent_reserve_bytes(spec, database)
    options = {
        "min_free_bytes": max(0, min_free_bytes),
        "remaining_permanent_bytes": lambda: remaining,
    }
    if disk_usage is not None:
        options["disk_usage"] = disk_usage
    return StorageGuard(Path(spec["run_dir"]), **options)


def _cleanup_stream_unit_artifacts(
    spec: Mapping[str, Any],
    database: RunStateDB,
    stream_id: str,
) -> dict[str, Any]:
    run_dir = Path(spec["run_dir"]).resolve()
    with _unit_cleanup_lock(run_dir):
        return _cleanup_stream_unit_artifacts_locked(spec, database, stream_id)


def _cleanup_stream_unit_artifacts_locked(
    spec: Mapping[str, Any],
    database: RunStateDB,
    stream_id: str,
) -> dict[str, Any]:
    """Crash-safely delete owned unit intermediates after final assembly.

    ``ready -> cleaning`` is committed before the file is moved to a
    deterministic tombstone in the same directory.  The database is then
    committed to ``cleaned`` before the tombstone is unlinked.  A later
    invocation can therefore resume from the original file, a ``cleaning``
    tombstone, or a post-commit ``cleaned`` tombstone without inferring a
    successful deletion from a missing file alone.
    """

    run_id = str(spec["run_id"])
    run_dir = Path(spec["run_dir"]).resolve()
    stream_component = _strict_path_component(
        str(stream_id).replace(":", "_"), label="stream_id"
    )
    tmp_root = run_dir / "tmp"
    output_root = tmp_root / "unit_outputs"
    unit_root = output_root / stream_component
    for owned_directory in (tmp_root, output_root, unit_root):
        if owned_directory.is_symlink():
            raise StreamAssemblyError(
                "refusing to clean through a symlinked unit output directory: "
                f"{owned_directory}"
            )
    candidates = [
        dict(artifact)
        for artifact in database.artifacts_for_stream(
            run_id, str(stream_id), status=None
        )
        if str(artifact.get("kind") or "") in UNIT_INTERMEDIATE_KINDS
        and str(artifact.get("status") or "") in {
            "ready",
            "cleaning",
            "cleaned",
        }
        and int(artifact.get("ref_count") or 0) == 0
    ]

    # Validate ownership and every extant file before changing the first row.
    # Unknown files in the directory are never selected or removed.
    validated: list[tuple[dict[str, Any], Path, Path]] = []
    for artifact in candidates:
        kind = str(artifact["kind"])
        unit_id = _strict_path_component(
            str(artifact["unit_id"]), label="unit_id"
        )
        path = Path(str(artifact["path"]))
        expected = unit_root / f"{unit_id}{UNIT_INTERMEDIATE_SUFFIXES[kind]}"
        if not path.is_absolute() or path != expected or path.parent != unit_root:
            raise StreamAssemblyError(
                "unit intermediate cleanup path is not an exact direct child "
                f"of the owned Stream directory: {path}"
            )
        artifact_id = int(artifact["artifact_id"])
        tombstone = _cleanup_tombstone(path, artifact_id)
        original_present = path.is_symlink() or path.exists()
        tombstone_present = tombstone.is_symlink() or tombstone.exists()
        if original_present and tombstone_present:
            raise StreamAssemblyError(
                "unit intermediate and cleanup tombstone both exist; refusing "
                f"ambiguous deletion: {path}"
            )
        status = str(artifact["status"])
        if status == "ready":
            if tombstone_present:
                raise StreamAssemblyError(
                    f"unclaimed cleanup tombstone already exists: {tombstone}"
                )
            _assert_regular_cleanup_file(path, artifact, stage="pre-claim validation")
        elif status == "cleaned":
            if original_present:
                raise StreamAssemblyError(
                    "cleaned unit intermediate unexpectedly reappeared: "
                    f"{path}"
                )
            if not tombstone_present:
                continue
            _assert_regular_cleanup_file(
                tombstone, artifact, stage="post-commit tombstone recovery"
            )
        elif original_present:
            _assert_regular_cleanup_file(path, artifact, stage="cleanup recovery")
        elif tombstone_present:
            _assert_regular_cleanup_file(
                tombstone, artifact, stage="tombstone recovery"
            )
        # ``cleaning`` with neither path is accepted only as recovery for Runs
        # interrupted by the pre-tombstone implementation.  New transactions
        # always commit ``cleaned`` before unlinking the tombstone.
        validated.append((artifact, path, tombstone))

    kind_counts: dict[str, int] = {}
    cleaned_bytes = 0
    for artifact, path, tombstone in validated:
        artifact_id = int(artifact["artifact_id"])
        claimed = artifact
        if str(artifact["status"]) == "ready":
            claimed = database.claim_artifact_cleanup(artifact_id)
            if claimed is None:
                raise StreamAssemblyError(
                    f"unit intermediate cleanup claim changed: {path}"
                )
        current = database.get_artifact(artifact_id)
        current_status = str((current or {}).get("status") or "")
        if (
            current is None
            or current_status not in {"cleaning", "cleaned"}
            or int(current["ref_count"]) != 0
        ):
            raise StreamAssemblyError(
                f"unit intermediate cleanup state changed after claim: {path}"
            )

        if path.is_symlink() or tombstone.is_symlink():
            raise StreamAssemblyError(
                f"refusing symlink during unit intermediate cleanup: {path}"
            )
        if current_status == "cleaned" and path.exists():
            raise StreamAssemblyError(
                f"cleaned unit intermediate unexpectedly reappeared: {path}"
            )
        if current_status == "cleaning" and path.exists():
            if tombstone.exists():
                raise StreamAssemblyError(
                    f"cleanup tombstone appeared before rename: {tombstone}"
                )
            _assert_regular_cleanup_file(path, claimed, stage="post-claim validation")
            _rename_cleanup_file(path, tombstone)
            _fsync_directory(unit_root)
        if tombstone.is_symlink():
            raise StreamAssemblyError(
                f"refusing symlinked cleanup tombstone: {tombstone}"
            )
        if current_status == "cleaning" and not database.finish_artifact_cleanup(
            artifact_id, success=True
        ):
            raise StreamAssemblyError(
                f"unit intermediate cleanup state changed: {path}"
            )
        if tombstone.exists():
            _assert_regular_cleanup_file(
                tombstone, claimed, stage="pre-unlink validation"
            )
            _unlink_cleanup_tombstone(tombstone)
            _fsync_directory(unit_root)
        kind = str(artifact["kind"])
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        cleaned_bytes += int(artifact["byte_count"])
    try:
        unit_root.rmdir()
    except (FileNotFoundError, OSError):
        pass
    report = {
        "status": "passed",
        "stream_id": str(stream_id),
        "artifact_count": len(validated),
        "cleaned_bytes": cleaned_bytes,
        "kind_counts": dict(sorted(kind_counts.items())),
        "path_policy": "strict_direct_child_no_symlink",
        "integrity_policy": "db_claim_tombstone_size_sha256",
    }
    database.append_event(
        run_id,
        "stream_unit_artifacts_cleaned",
        message=str(stream_id),
        payload=report,
    )
    return report


@contextmanager
def _reserved_vector_write(
    storage_guard: StorageGuard | None,
    lock_path: Path | None,
    operation: str,
    write_bytes: int,
):
    if storage_guard is None:
        yield
        return
    shared_lock = lock_path or (
        storage_guard.root / "tmp" / ".vector-storage-reserve.lock"
    )
    shared_lock.parent.mkdir(parents=True, exist_ok=True)
    with shared_lock.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        reserved_write = 0
        try:
            reservation = storage_guard.check(
                operation,
                write_bytes=max(0, int(write_bytes)),
                managed_growth_bytes=0,
                reserve_managed_growth=True,
            )
            reserved_write = int(reservation["reserved_write_bytes"])
            yield
        finally:
            if reserved_write:
                storage_guard.adjust(0, settled_write_bytes=reserved_write)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _estimate_source_gpkg_bytes(
    paths: Iterator[str | Path] | list[str | Path] | tuple[str | Path, ...],
    *,
    multiplier: int = 2,
) -> int:
    source_bytes = sum(Path(path).stat().st_size for path in paths)
    return max(
        GPKG_ATOMIC_OVERHEAD_BYTES,
        max(1, int(multiplier)) * source_bytes + GPKG_ATOMIC_OVERHEAD_BYTES,
    )


def _estimate_json_bytes(payload: Mapping[str, Any]) -> int:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    return len(encoded) + JSON_ATOMIC_OVERHEAD_BYTES


def _write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    storage_guard: StorageGuard | None = None,
    storage_lock_path: Path | None = None,
    operation: str = "stream_report",
) -> None:
    with _reserved_vector_write(
        storage_guard,
        storage_lock_path,
        operation,
        _estimate_json_bytes(payload),
    ):
        _atomic_json(path, payload)


def _accepted_layer_has_geometry(
    accepted_path: str | Path,
    *,
    accepted_layer: str = "accepted_labels",
) -> bool:
    path = Path(accepted_path).resolve()
    if not path.is_file() or accepted_layer not in fiona.listlayers(path):
        return False
    with fiona.open(path, layer=accepted_layer) as source:
        return any(feature.get("geometry") for feature in source)


def _guarded_accepted_difference(
    source_path: Path,
    accepted_path: str | Path,
    output_path: Path,
    *,
    storage_guard: StorageGuard | None,
    storage_lock_path: Path | None,
    operation: str,
) -> dict[str, Any]:
    if not _accepted_layer_has_geometry(accepted_path):
        return apply_accepted_difference(source_path, accepted_path, output_path)
    with _reserved_vector_write(
        storage_guard,
        storage_lock_path,
        operation,
        _estimate_source_gpkg_bytes((source_path,), multiplier=2),
    ):
        return apply_accepted_difference(source_path, accepted_path, output_path)


def _guarded_range_clip(
    source_path: Path,
    spec: Mapping[str, Any],
    *,
    storage_guard: StorageGuard | None,
    storage_lock_path: Path | None,
    operation: str,
) -> dict[str, Any]:
    with _reserved_vector_write(
        storage_guard,
        storage_lock_path,
        operation,
        _estimate_source_gpkg_bytes((source_path,), multiplier=2),
    ):
        return apply_adaptive_range_clip(source_path, spec)


def _stream_root(spec: Mapping[str, Any], stream: Mapping[str, Any]) -> Path:
    run_dir = Path(spec["run_dir"])
    if stream["kind"] == "model":
        return run_dir / "models" / str(stream["model_id"])
    return run_dir / "fusion" / str(stream["profile_id"])


def _read_features(path: str | Path) -> list[dict[str, Any]]:
    result = []
    with fiona.open(path, layer="polygons") as source:
        for feature in source:
            result.append(
                {
                    "geometry": shape(feature["geometry"]),
                    "properties": dict(feature["properties"]),
                }
            )
    return result


def _atomic_gpkg(
    path: Path,
    layer: str,
    schema,
    crs,
    writer,
    *,
    storage_guard: StorageGuard | None = None,
    storage_lock_path: Path | None = None,
    estimated_write_bytes: int = GPKG_ATOMIC_OVERHEAD_BYTES,
    operation: str = "stream_gpkg",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.{os.getpid()}.tmp.gpkg"
    with _reserved_vector_write(
        storage_guard,
        storage_lock_path,
        operation,
        estimated_write_bytes,
    ):
        temporary.unlink(missing_ok=True)
        try:
            with fiona.open(
                temporary,
                "w",
                driver="GPKG",
                layer=layer,
                schema=schema,
                crs=CRS.from_user_input(crs),
            ) as destination:
                writer(destination)
            with sqlite3.connect(temporary) as connection:
                if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise StreamAssemblyError(
                        f"GeoPackage integrity check failed: {temporary}"
                    )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


@contextmanager
def _readonly_sqlite(path: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        yield connection
    finally:
        connection.close()


def _file_fingerprint(path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise StreamAssemblyError(f"file changed while it was being checked: {path}")
    return {
        "byte_count": int(after.st_size),
        "sha256": str(digest),
        "mtime_ns": int(after.st_mtime_ns),
    }


def _validate_existing_gpkg(
    path: Path,
    *,
    layer: str,
    schema: Mapping[str, Any],
    crs: Any,
    identity: Mapping[str, str],
    expected_feature_count: int | None,
) -> dict[str, Any]:
    if not path.is_file():
        raise StreamAssemblyError(f"resume input is missing: {path}")
    with _readonly_sqlite(path) as connection:
        integrity = [
            str(row[0]) for row in connection.execute("PRAGMA integrity_check")
        ]
    if integrity != ["ok"]:
        raise StreamAssemblyError(
            f"resume input GeoPackage integrity check failed: {path}; "
            f"{integrity[:3]}"
        )
    if layer not in fiona.listlayers(path):
        raise StreamAssemblyError(f"resume input layer is missing: {path}::{layer}")
    with fiona.open(path, layer=layer) as source:
        actual_properties = source.schema.get("properties") or {}
        expected_properties = schema.get("properties") or {}
        actual_fields = set(actual_properties)
        expected_fields = set(expected_properties)
        if actual_fields != expected_fields:
            raise StreamAssemblyError(
                f"resume input fields changed: {path}::{layer}; "
                f"expected={sorted(expected_fields)}, actual={sorted(actual_fields)}"
            )
        type_aliases = {
            "int32": "int",
            "int64": "int",
            "float32": "float",
            "float64": "float",
        }
        mismatched_types = []
        for key in sorted(expected_fields):
            expected_type = str(expected_properties[key]).split(":", 1)[0].lower()
            actual_type = str(actual_properties[key]).split(":", 1)[0].lower()
            expected_type = type_aliases.get(expected_type, expected_type)
            actual_type = type_aliases.get(actual_type, actual_type)
            if actual_type != expected_type:
                mismatched_types.append(
                    f"{key}:{actual_properties[key]}!={expected_properties[key]}"
                )
        if mismatched_types:
            raise StreamAssemblyError(
                f"resume input field types changed: {path}::{layer}; "
                f"{mismatched_types}"
            )
        actual_geometry = str(source.schema.get("geometry") or "")
        expected_geometry = str(schema.get("geometry") or "")
        if actual_geometry != expected_geometry:
            raise StreamAssemblyError(
                f"resume input geometry type changed: {path}::{layer}; "
                f"expected={expected_geometry}, actual={actual_geometry}"
            )
        actual_crs = CRS.from_user_input(source.crs_wkt or source.crs)
        expected_crs = CRS.from_user_input(crs)
        if actual_crs != expected_crs:
            raise StreamAssemblyError(
                f"resume input CRS changed: {path}::{layer}; "
                f"expected={expected_crs}, actual={actual_crs}"
            )
        feature_count = len(source)
        if (
            expected_feature_count is not None
            and feature_count != int(expected_feature_count)
        ):
            raise StreamAssemblyError(
                f"resume input feature count changed: {path}::{layer}; "
                f"expected={expected_feature_count}, actual={feature_count}"
            )
        if next(iter(source), None) is None:
            raise StreamAssemblyError(f"resume input layer is empty: {path}::{layer}")
    if not layer.replace("_", "").isalnum():
        raise StreamAssemblyError(f"unsafe GeoPackage layer name: {layer}")
    clauses = [
        f"COALESCE(CAST(\"{key}\" AS TEXT), '') != ?"
        for key in identity
    ]
    with _readonly_sqlite(path) as connection:
        mismatched = int(
            connection.execute(
                f'SELECT COUNT(*) FROM "{layer}" WHERE {" OR ".join(clauses)}',
                tuple(str(value) for value in identity.values()),
            ).fetchone()[0]
        )
    if mismatched:
        raise StreamAssemblyError(
            f"resume input contains {mismatched} rows for another run or stream: "
            f"{path}::{layer}"
        )
    return {
        "path": str(path),
        "layer": layer,
        "feature_count": int(feature_count),
        **_file_fingerprint(path),
    }


def _assert_gpkg_within_exact_range(
    path: Path,
    *,
    layer: str,
    spec: Mapping[str, Any],
) -> None:
    """Reject a resume formal output that was not clipped to its frozen range."""

    selection = spec.get("range_selection") or {}
    if (
        selection.get("mode") != "vector_tile_intersection"
        and not isinstance(spec.get("requested_extent"), Mapping)
    ):
        # Pre-range legacy runs have neither a vector authority nor an extent
        # contract to validate. They remain resumable without inventing one.
        return

    try:
        with fiona.open(path, layer=layer) as source:
            mask = extract_range_mask_geometry(spec, source.crs_wkt or source.crs)
            if mask is None or mask.is_empty:
                raise StreamAssemblyError("resume formal output has no exact range mask")
            outside = sum(
                1
                for feature in source
                if feature.get("geometry")
                and not mask.covers(shape(feature["geometry"]))
            )
    except RangeClipRuntimeError as error:
        raise StreamAssemblyError(str(error)) from error
    if outside:
        raise StreamAssemblyError(
            f"resume formal output contains {outside} features outside the exact range"
        )


def _balanced_geometry_union(geometries, *, batch_size: int = 2048):
    """Return a bounded-memory union for a potentially very large GPKG."""

    levels: list[Any | None] = []

    def add_partial(value) -> None:
        level = 0
        current = value
        while True:
            if level == len(levels):
                levels.append(current)
                return
            previous = levels[level]
            if previous is None:
                levels[level] = current
                return
            current = unary_union([previous, current])
            levels[level] = None
            level += 1

    batch = []
    for geometry in geometries:
        batch.append(geometry)
        if len(batch) >= batch_size:
            add_partial(unary_union(batch))
            batch = []
    if batch:
        add_partial(unary_union(batch))
    values = [value for value in levels if value is not None]
    return unary_union(values)


def _ground_area_scale(source_crs: Any, center_y: float) -> float:
    """Approximate ground square metres per source-coordinate square unit."""

    crs = RasterCRS.from_user_input(source_crs)
    if crs.to_epsg() == 3857:
        latitude = math.atan(math.sinh(float(center_y) / 6378137.0))
        return math.cos(latitude) ** 2
    if crs.is_projected:
        units = str(getattr(crs, "linear_units", "") or "").lower()
        if units in {"metre", "meter", "metres", "meters", "m"}:
            return 1.0
    if crs.is_geographic:
        latitude = float(center_y)
        if not -90.0 <= latitude <= 90.0:
            raise StreamAssemblyError(
                "coverage range centre latitude is outside [-90, 90]"
            )
        lat_rad = math.radians(latitude)
        metres_per_degree_lat = (
            111132.954
            - 559.822 * math.cos(2 * lat_rad)
            + 1.175 * math.cos(4 * lat_rad)
        )
        metres_per_degree_lon = (
            math.pi / 180.0
        ) * 6378137.0 * math.cos(lat_rad)
        return abs(metres_per_degree_lat * metres_per_degree_lon)
    raise StreamAssemblyError(
        f"coverage area requires a projected metre CRS, EPSG:3857, or geographic CRS: {crs}"
    )


def _validate_exact_range_coverage(
    path: str | Path,
    *,
    layer: str,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure exact range gaps, feature overlap and outside coverage.

    The union is reduced as a balanced tree so validation remains bounded for
    large formal GPKGs.  Area gates use source-coordinate units and are also
    reported as locally corrected ground square metres for the monitor.
    """

    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise StreamAssemblyError(f"coverage validation source is missing: {source_path}")
    feature_count = 0
    feature_area = 0.0
    feature_area_compensation = 0.0

    def geometries():
        nonlocal feature_count, feature_area, feature_area_compensation
        with fiona.open(source_path, layer=layer) as source:
            for feature in source:
                if not feature.get("geometry"):
                    raise StreamAssemblyError(
                        f"coverage validation found an empty feature geometry: {source_path}"
                    )
                geometry = shape(feature["geometry"])
                if geometry.is_empty or not geometry.is_valid or geometry.area <= 0:
                    raise StreamAssemblyError(
                        f"coverage validation found an invalid feature geometry: {source_path}"
                    )
                feature_count += 1
                # Kahan summation prevents millions of small polygon areas
                # from fabricating an overlap through floating-point drift.
                area = float(geometry.area)
                corrected = area - feature_area_compensation
                accumulated = feature_area + corrected
                feature_area_compensation = (accumulated - feature_area) - corrected
                feature_area = accumulated
                yield geometry

    with fiona.open(source_path, layer=layer) as source:
        source_crs = source.crs_wkt or source.crs
    if not source_crs:
        raise StreamAssemblyError("coverage validation source CRS is missing")
    range_geometry = extract_range_mask_geometry(spec, source_crs)
    if range_geometry is None or range_geometry.is_empty:
        # Runs created before exact-range snapshots existed are still readable,
        # but must never be advertised as coverage-verified.
        return {
            "schema_version": 1,
            "status": "skipped_legacy_no_range",
            "hard_gate_applied": False,
            "reason": "frozen_exact_range_missing",
            "feature_count": 0,
            "range_area_m2": 0.0,
            "gap_area_m2": 0.0,
            "overlap_area_m2": 0.0,
            "outside_area_m2": 0.0,
            "area_tolerance_m2": 0.0,
            "gap_area_source_units2": 0.0,
            "overlap_area_source_units2": 0.0,
            "outside_area_source_units2": 0.0,
            "area_tolerance_source_units2": 0.0,
            "area_scale_method": "not_available",
            "union_method": "not_run",
        }

    union_geometry = _balanced_geometry_union(geometries())
    range_area = float(range_geometry.area)
    union_area = float(union_geometry.area)
    if range_area <= 0:
        raise StreamAssemblyError("coverage validation range has no positive area")
    gap_area = float(range_geometry.difference(union_geometry).area)
    outside_area = float(union_geometry.difference(range_geometry).area)
    overlap_area = max(0.0, feature_area - union_area)

    transform = spec.get("raster", {}).get("transform") or []
    if len(transform) < 6:
        raise StreamAssemblyError("coverage validation requires the raster affine")
    affine_area = abs(
        float(transform[0]) * float(transform[4])
        - float(transform[1]) * float(transform[3])
    )
    if not math.isfinite(affine_area) or affine_area <= 0:
        raise StreamAssemblyError("coverage validation raster affine has no positive area")
    config = dict(spec.get("coverage_validation") or {})
    tolerance_pixels = float(config.get("area_tolerance_pixels", 0.01))
    if not math.isfinite(tolerance_pixels) or not 0.0 <= tolerance_pixels <= 1.0:
        raise StreamAssemblyError(
            "coverage_validation.area_tolerance_pixels must be between 0 and 1"
        )
    numerical_tolerance = min(range_area * 1.0e-10, affine_area * 0.25)
    tolerance = max(affine_area * tolerance_pixels, numerical_tolerance)
    ground_scale = _ground_area_scale(source_crs, range_geometry.centroid.y)
    passed = all(
        value <= tolerance for value in (gap_area, overlap_area, outside_area)
    )
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "hard_gate_applied": True,
        "feature_count": int(feature_count),
        "range_area_m2": float(range_area * ground_scale),
        "gap_area_m2": float(gap_area * ground_scale),
        "overlap_area_m2": float(overlap_area * ground_scale),
        "outside_area_m2": float(outside_area * ground_scale),
        "area_tolerance_m2": float(tolerance * ground_scale),
        "gap_area_source_units2": gap_area,
        "overlap_area_source_units2": overlap_area,
        "outside_area_source_units2": outside_area,
        "area_tolerance_source_units2": tolerance,
        "area_scale_method": "local_crs_ground_scale_v1",
        "union_method": "balanced_unary_union_2048_v1",
    }


def _publish_coverage_validation(
    database: RunStateDB,
    run_id: str,
    stream_id: str,
    report: Mapping[str, Any],
) -> None:
    payload = dict(report)
    status = str(payload.get("status") or "unknown")
    database.append_event(
        run_id,
        "stream_coverage_validation",
        level="error" if status == "failed" else "info",
        stream_id=stream_id,
        message=str(payload.get("status") or "unknown"),
        payload=payload,
    )
    print(
        json.dumps(
            {
                "event": "stream_coverage_validation",
                "run_id": str(run_id),
                "stream_id": str(stream_id),
                **payload,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _resolved_object_state(
    state_db: str | Path,
    run_id: str,
    stream_id: str,
) -> tuple[int, int]:
    database = RunStateDB(state_db)
    with database._connection() as connection:
        row = connection.execute(
            """SELECT COUNT(*) AS part_count,
                      COALESCE(SUM(CASE WHEN object_id IS NULL OR object_id=''
                                        THEN 1 ELSE 0 END), 0) AS unresolved_count,
                      COALESCE(SUM(CASE WHEN parent_id=part_id THEN 1 ELSE 0 END), 0)
                        AS object_count
               FROM object_nodes WHERE run_id=? AND stream_id=?""",
            (str(run_id), str(stream_id)),
        ).fetchone()
    part_count = int(row[0])
    unresolved_count = int(row[1])
    object_count = int(row[2])
    if part_count < 1 or object_count < 1 or unresolved_count:
        raise StreamAssemblyError(
            "resume requires existing resolved object IDs; "
            f"parts={part_count}, objects={object_count}, unresolved={unresolved_count}"
        )
    return part_count, object_count


def _assert_fingerprint_unchanged(
    path: Path,
    expected: Mapping[str, Any],
) -> None:
    actual = _file_fingerprint(path)
    if (
        int(actual["byte_count"]) != int(expected["byte_count"])
        or str(actual["sha256"]) != str(expected["sha256"])
    ):
        raise StreamAssemblyError(f"resume input changed during report assembly: {path}")


def _feature_batches(path: str | Path, *, size: int = OBJECT_ID_BATCH_SIZE):
    batch = []
    with fiona.open(path, layer="polygons") as source:
        for feature in source:
            batch.append(feature)
            if len(batch) >= int(size):
                yield batch
                batch = []
    if batch:
        yield batch


def _assembly_validation_workers(
    spec: Mapping[str, Any],
    item_count: int,
) -> int:
    configured = int(
        (spec.get("scaling") or {}).get("assembly_validation_workers")
        or min(8, max(2, (os.cpu_count() or 2) - 2))
    )
    return max(1, min(configured, ASSEMBLY_VALIDATION_MAX_IN_FLIGHT, item_count))


def _parallel_validate_summary_artifacts(
    items: list[Mapping[str, Any]],
    *,
    workers: int,
    edge_schema: Mapping[str, Any],
    crs: Any,
    run_id: str,
    stream_id: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Validate immutable report/edge shards with a bounded future window."""
    if not items:
        return {
            "workers": 0,
            "peak_in_flight": 0,
            "artifact_count": 0,
            "elapsed_sec": 0.0,
        }
    started_at = time.monotonic()
    peak_in_flight = 0

    def validate(item: Mapping[str, Any]) -> None:
        artifact = item["artifact"]
        path = Path(str(artifact["path"]))
        if str(item["kind"]) == "edge":
            fingerprint = _validate_existing_gpkg(
                path,
                layer="fitted_edges",
                schema=edge_schema,
                crs=crs,
                identity={
                    "run_id": run_id,
                    "stream_id": stream_id,
                    "unit_id": str(artifact["unit_id"]),
                },
                expected_feature_count=int(item["expected_feature_count"]),
            )
        else:
            if not path.is_file():
                raise StreamAssemblyError(f"unit report Artifact is missing: {path}")
            fingerprint = _file_fingerprint(path)
        if (
            int(artifact["byte_count"]) != int(fingerprint["byte_count"])
            or str(artifact["sha256"]) != str(fingerprint["sha256"])
        ):
            raise StreamAssemblyError(
                f"unit {item['kind']} Artifact changed: {path}"
            )

    iterator = iter(items)
    pending = set()
    completed_count = 0
    with ThreadPoolExecutor(
        max_workers=max(1, int(workers)),
        thread_name_prefix="assembly-validator",
    ) as executor:
        while len(pending) < ASSEMBLY_VALIDATION_MAX_IN_FLIGHT:
            try:
                pending.add(executor.submit(validate, next(iterator)))
            except StopIteration:
                break
        peak_in_flight = max(peak_in_flight, len(pending))
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                future.result()
                completed_count += 1
                if progress_callback is not None:
                    progress_callback(completed_count, len(items))
            while len(pending) < ASSEMBLY_VALIDATION_MAX_IN_FLIGHT:
                try:
                    pending.add(executor.submit(validate, next(iterator)))
                except StopIteration:
                    break
            peak_in_flight = max(peak_in_flight, len(pending))
    return {
        "workers": int(workers),
        "peak_in_flight": int(peak_in_flight),
        "artifact_count": len(items),
        "elapsed_sec": round(time.monotonic() - started_at, 3),
    }


def _validated_summary_inputs(
    spec: Mapping[str, Any],
    database: RunStateDB,
    run_id: str,
    stream_id: str,
    expected_units: int,
    report_artifacts: list[Mapping[str, Any]],
    edge_schema: Mapping[str, Any],
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summaries = database.unit_report_summaries(run_id, stream_id)
    if len(summaries) != int(expected_units):
        raise StreamAssemblyError(
            "unit report summaries are incomplete; rerun unit fitting with the "
            f"current Ubuntu runtime: {len(summaries)}/{expected_units}"
        )
    reports_by_unit = {
        str(artifact["unit_id"]): dict(artifact)
        for artifact in report_artifacts
    }
    summaries_by_unit = {
        str(summary["unit_id"]): dict(summary)
        for summary in summaries
    }
    if set(reports_by_unit) != set(summaries_by_unit):
        raise StreamAssemblyError(
            "unit report summaries do not match ready report Artifacts"
        )
    validation_items: list[dict[str, Any]] = []
    for unit_id in sorted(summaries_by_unit):
        summary = summaries_by_unit[unit_id]
        artifact = reports_by_unit[unit_id]
        if (
            Path(str(summary["report_path"])).resolve()
            != Path(str(artifact["path"])).resolve()
            or int(summary["report_byte_count"]) != int(artifact["byte_count"])
            or str(summary["report_sha256"]) != str(artifact["sha256"])
        ):
            raise StreamAssemblyError(
                f"unit report summary fingerprint changed: {unit_id}"
            )
        validation_items.append(
            {"kind": "report", "artifact": artifact}
        )

    edge_artifacts = database.artifacts_for_stream(
        run_id,
        stream_id,
        kind="unit_fitted_edges",
    )
    edges_by_unit = {
        str(artifact["unit_id"]): dict(artifact)
        for artifact in edge_artifacts
    }
    expected_edge_units = {
        unit_id
        for unit_id, summary in summaries_by_unit.items()
        if int(summary["fitted_edge_count"]) > 0
    }
    if set(edges_by_unit) != expected_edge_units:
        missing = sorted(expected_edge_units - set(edges_by_unit))
        unexpected = sorted(set(edges_by_unit) - expected_edge_units)
        raise StreamAssemblyError(
            "unit fitted-edge Artifacts do not match database summaries; "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    for unit_id in sorted(edges_by_unit):
        validation_items.append(
            {
                "kind": "edge",
                "artifact": edges_by_unit[unit_id],
                "expected_feature_count": int(
                    summaries_by_unit[unit_id]["fitted_edge_count"]
                ),
            }
        )
    workers = _assembly_validation_workers(spec, len(validation_items))
    validation = _parallel_validate_summary_artifacts(
        validation_items,
        workers=workers,
        edge_schema=edge_schema,
        crs=spec["raster"]["crs"],
        run_id=run_id,
        stream_id=stream_id,
        progress_callback=progress_callback,
    )
    return (
        [edges_by_unit[unit_id] for unit_id in sorted(edges_by_unit)],
        validation,
    )


def _reuse_ready_assembly(
    spec: Mapping[str, Any],
    stream: Mapping[str, Any],
    database: RunStateDB,
) -> dict[str, Any] | None:
    run_id = str(spec["run_id"])
    stream_id = str(stream["stream_id"])
    rows = {
        str(row["stream_id"]): row for row in database.stream_rows(run_id)
    }
    stream_status = str((rows.get(stream_id) or {}).get("status") or "")
    if stream_status not in {"ready", "failed"}:
        return None
    root = _stream_root(spec, stream)
    paths = {
        "semantic_polygons_raw": root / "semantic_polygons_raw.gpkg",
        "semantic_polygons": root / "semantic_polygons.gpkg",
        "boundary_fitting_report": root / "boundary_fitting_report.json",
        "fitted_edges": root / "fitted_edges.gpkg",
    }
    for kind, path in paths.items():
        artifact = database.artifact_for_stream_unit(
            run_id,
            stream_id,
            "assembled",
            kind,
        )
        if artifact is None or artifact["status"] != "ready" or not path.is_file():
            # A failed Stream may legitimately contain only the raw/formal
            # pair from a previously completed assembly attempt.  In that
            # state the caller must continue through the normal full or
            # report-resume validation path; treating it as a corrupt ready
            # Stream prevents the strict unit-report checks from running.
            # A Stream still marked ready, however, must remain an immutable
            # completed set and missing members are a hard integrity error.
            if stream_status == "failed":
                return None
            raise StreamAssemblyError(
                f"ready stream is missing assembled Artifact: {stream_id}/{kind}"
            )
        if (
            int(artifact["byte_count"]) != path.stat().st_size
            or str(artifact["sha256"]) != sha256_file(path)
        ):
            raise StreamAssemblyError(
                f"ready assembled Artifact changed on disk: {path}"
            )
    report = dict(load_json(paths["boundary_fitting_report"]))
    if (
        report.get("status") != "passed"
        or (report.get("validation") or {}).get("passed") is not True
    ):
        raise StreamAssemblyError(
            f"ready stream has a failed boundary report: {stream_id}"
        )
    coverage = dict(report.get("coverage_validation") or {})
    if not coverage:
        coverage = _validate_exact_range_coverage(
            paths["semantic_polygons"],
            layer="semantic_polygons",
            spec=spec,
        )
        _publish_coverage_validation(
            database,
            run_id,
            stream_id,
            coverage,
        )
        if coverage["status"] == "failed":
            raise StreamAssemblyError(
                "ready stream failed exact coverage validation: "
                f"gap={coverage['gap_area_m2']:.6g} m2, "
                f"overlap={coverage['overlap_area_m2']:.6g} m2, "
                f"outside={coverage['outside_area_m2']:.6g} m2"
            )
        report["coverage_validation"] = coverage
    report["assembly_mode"] = "reused"
    report.setdefault(
        "report_processed_count",
        int(report.get("unit_count") or 0),
    )
    report.setdefault(
        "report_queue_capacity",
        ASSEMBLY_VALIDATION_MAX_IN_FLIGHT,
    )
    report.setdefault("report_peak_loaded_count", 0)
    report.setdefault("report_summary_source", "run_state_database")
    report.setdefault("report_json_parse_count", 0)
    report["object_link_count"] = database.object_link_count(run_id, stream_id)
    report["recovered_stream_status"] = stream_status
    print(
        json.dumps(
            {"event": "stream_assembly_reused", **report},
            separators=(",", ":"),
        )
    )
    return report


def _link_neighbor_parts(
    database: RunStateDB,
    run_id: str,
    stream_id: str,
    formal_by_unit: Mapping[str, str],
    units: list[Mapping[str, Any]],
    tolerance: float,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> int:
    linked = 0
    neighbors = ownership_neighbors(units)
    for pair_index, (left_unit, right_unit) in enumerate(neighbors, start=1):
        left_features = _read_features(formal_by_unit[left_unit])
        right_features = _read_features(formal_by_unit[right_unit])
        right_geometries = [item["geometry"] for item in right_features]
        tree = STRtree(right_geometries)
        for left in left_features:
            left_class = int(left["properties"]["class_code"])
            for index in tree.query(left["geometry"]):
                right = right_features[int(index)]
                if int(right["properties"]["class_code"]) != left_class:
                    continue
                shared = left["geometry"].boundary.intersection(right["geometry"].boundary)
                if shared.length <= tolerance:
                    continue
                if database.add_object_link(
                    run_id,
                    stream_id,
                    str(left["properties"]["polygon_id"]),
                    str(right["properties"]["polygon_id"]),
                    left_class,
                ):
                    linked += 1
        if progress_callback is not None:
            progress_callback(pair_index, len(neighbors), linked)
    return linked


def _assemble_stream_impl(
    run_spec_path: str | Path,
    stream_id: str,
    *,
    resume_from_reports: bool = False,
) -> dict[str, Any]:
    spec = load_json(Path(run_spec_path).resolve())
    if spec.get("schema_version") != 2:
        raise StreamAssemblyError("stream assembly requires run_spec schema 2")
    run_id = str(spec["run_id"])
    database = RunStateDB(spec["state_db"])
    streams = [item for item in spec["streams"] if item["stream_id"] == stream_id]
    if len(streams) != 1:
        raise StreamAssemblyError(f"unknown result stream: {stream_id}")
    stream = streams[0]
    boundary = spec.get("boundary_fitting") or {}
    fit_mode = str(boundary.get("mode") or "")
    if fit_mode != "divider_cubic_bspline_adaptive_v2":
        raise StreamAssemblyError(
            "only divider_cubic_bspline_adaptive_v2 is supported "
            "by the current runtime"
        )
    smoothing_enabled = bool(boundary.get("enabled", True))
    fit_version = (
        "divider_cubic_bspline_adaptive_v2"
        if smoothing_enabled else "raw_polygonize_v1"
    )
    counts = database.stream_unit_counts(run_id, stream_id)
    expected_units = sum(counts.values())
    progress = _AssemblyProgress(database, run_id, stream_id)
    reused = _reuse_ready_assembly(spec, stream, database)
    if reused is not None:
        progress.emit(
            "publish_cleanup",
            current=0,
            total=1,
            message="校验已组装产物并清理残留中间文件",
            force=True,
        )
        reused["unit_artifact_cleanup"] = _cleanup_stream_unit_artifacts(
            spec, database, stream_id
        )
        database.set_stream_status(run_id, stream_id, "ready", error="")
        progress.emit(
            "publish_cleanup",
            current=1,
            total=1,
            status="completed",
            message="已复用完整组装产物",
            force=True,
        )
        return reused
    database.set_stream_status(
        run_id,
        stream_id,
        "assembling",
        error="",
    )
    if expected_units < 1 or counts != {"ready": expected_units}:
        raise StreamAssemblyError(f"stream units are not all ready: {counts}")
    units = database.spatial_units_for_stream(run_id, stream_id)
    formal_artifacts = database.artifacts_for_stream(
        run_id, stream_id, kind="unit_formal"
    )
    raw_artifacts = database.artifacts_for_stream(run_id, stream_id, kind="unit_raw")
    report_artifacts = database.artifacts_for_stream(
        run_id, stream_id, kind="unit_boundary_report"
    )
    if not (
        len(formal_artifacts) == len(raw_artifacts) == len(report_artifacts) == expected_units
    ):
        raise StreamAssemblyError("unit Artifact count does not match ready unit count")
    formal_by_unit = {str(item["unit_id"]): str(item["path"]) for item in formal_artifacts}
    raw_by_unit = {str(item["unit_id"]): str(item["path"]) for item in raw_artifacts}

    transform = spec["raster"]["transform"]
    root = _stream_root(spec, stream)
    storage_guard = _run_storage_guard(spec, database)
    storage_lock_path = Path(spec["run_dir"]) / "tmp" / ".vector-storage-reserve.lock"
    raw_path = root / "semantic_polygons_raw.gpkg"
    formal_path = root / "semantic_polygons.gpkg"
    report_path = root / "boundary_fitting_report.json"
    fitted_edges_path = root / "fitted_edges.gpkg"
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")

    raw_schema = {
        "geometry": "MultiPolygon",
        "properties": {
            "run_id": "str:48",
            "stream_id": "str:96",
            "unit_id": "str:96",
            "polygon_id": "str:96",
            "class_code": "int",
        },
    }

    formal_schema = {
        "geometry": "MultiPolygon",
        "properties": {
            "run_id": "str:48",
            "result_stream_id": "str:96",
            "result_kind": "str:16",
            "model_id": "str:64",
            "fusion_profile_id": "str:64",
            "object_id": "str:64",
            "part_id": "str:96",
            "class_code": "int",
            "class_name": "str:64",
            "confidence_mean": "float",
            "confidence_std": "float",
            "model_version": "str:64",
            "source": "str:32",
            "fit_changed": "int",
            "fit_methods": "str:64",
            "fit_version": "str:40",
            "fit_status": "str:24",
            "origin_unit_ids": "str:254",
            "vertex_count_before": "int",
            "vertex_count_after": "int",
            "max_shift_px": "float",
            "mean_shift_px": "float",
            "area_change_ratio": "float",
            "created_at": "str:40",
        },
    }
    edge_schema = {
        "geometry": "LineString",
        "properties": {
            "run_id": "str:48",
            "stream_id": "str:96",
            "unit_id": "str:96",
            "chain_id": "str:96",
            "method": "str:24",
            "status": "str:32",
            "max_shift": "float",
            "dense_vtx": "int",
            "sparse_vtx": "int",
            "chord_err": "float",
            "arc_len": "float",
        },
    }
    progress.emit(
        "validate_inputs",
        current=0,
        total=expected_units,
        message="校验单元报告和拟合边界分片",
        force=True,
    )
    edge_artifacts, summary_validation = _validated_summary_inputs(
        spec,
        database,
        run_id,
        stream_id,
        expected_units,
        report_artifacts,
        edge_schema,
        progress_callback=lambda current, total: progress.emit(
            "validate_inputs",
            current=current,
            total=total,
            message="校验单元报告和拟合边界分片",
        ),
    )
    progress.emit(
        "validate_inputs",
        current=int(summary_validation["artifact_count"]),
        total=int(summary_validation["artifact_count"]),
        message="单元产物校验完成",
        force=True,
    )
    summary_aggregate = database.unit_report_summary_aggregate(
        run_id,
        stream_id,
    )
    if int(summary_aggregate["unit_count"]) != expected_units:
        raise StreamAssemblyError(
            "run-state report aggregate does not cover every ready unit"
        )
    model_id = str(stream.get("model_id") or "")
    profile_id = str(stream.get("profile_id") or "")
    version = str(stream.get("version") or "")
    ownership_validation = {
        "passed": True,
        "scope": "all_output_polygons",
        "invalid_count": 0,
    }
    resume_inputs: dict[str, dict[str, Any]] = {}
    formal_feature_count = 0

    if resume_from_reports:
        progress.emit(
            "register_objects",
            current=0,
            total=1,
            message="校验已有对象身份",
            force=True,
        )
        part_count, object_count = _resolved_object_state(
            spec["state_db"],
            run_id,
            stream_id,
        )
        link_count = database.object_link_count(run_id, stream_id)
        resume_inputs["raw"] = _validate_existing_gpkg(
            raw_path,
            layer="semantic_polygons_raw",
            schema=raw_schema,
            crs=spec["raster"]["crs"],
            identity={"run_id": run_id, "stream_id": stream_id},
            expected_feature_count=part_count,
        )
        resume_inputs["formal"] = _validate_existing_gpkg(
            formal_path,
            layer="semantic_polygons",
            schema=formal_schema,
            crs=spec["raster"]["crs"],
            identity={"run_id": run_id, "result_stream_id": stream_id},
            # Exact clipping may discard a fully outside part or split one at
            # the boundary, so formal feature count is intentionally not tied
            # to pre-clip object_nodes. Raw remains count-locked above.
            expected_feature_count=None,
        )
        _assert_gpkg_within_exact_range(
            formal_path,
            layer="semantic_polygons",
            spec=spec,
        )
        formal_feature_count = int(
            resume_inputs["formal"].get("feature_count") or 0
        )
        print(
            json.dumps(
                {
                    "event": "stream_report_resume_inputs_validated",
                    "run_id": run_id,
                    "stream_id": stream_id,
                    "feature_count": part_count,
                    "object_count": object_count,
                    "raw_sha256": resume_inputs["raw"]["sha256"],
                    "formal_sha256": resume_inputs["formal"]["sha256"],
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        for phase, message in (
            ("register_objects", "已有对象身份校验完成"),
            ("link_objects", "复用已有跨单元对象连接"),
            ("write_raw", "复用已有 Raw GPKG"),
            ("write_formal", "复用并校验已有正式 GPKG"),
        ):
            progress.emit(
                phase,
                current=1,
                total=1,
                message=message,
                force=True,
            )
    else:
        registered_part_count = 0
        progress.emit(
            "register_objects",
            current=0,
            total=len(formal_artifacts),
            message="登记空间单元多边形身份",
            force=True,
        )
        for artifact_index, artifact in enumerate(formal_artifacts, start=1):
            unit_id = str(artifact["unit_id"])
            for features in _feature_batches(artifact["path"]):
                registered_part_count += database.register_object_parts(
                    run_id,
                    stream_id,
                    (
                        {
                            "part_id": str(feature["properties"]["polygon_id"]),
                            "class_code": int(feature["properties"]["class_code"]),
                            "unit_id": unit_id,
                        }
                        for feature in features
                    ),
                )
            progress.emit(
                "register_objects",
                current=artifact_index,
                total=len(formal_artifacts),
                feature_count=registered_part_count,
                message=f"已登记 {registered_part_count} 个多边形部件",
            )
        pixel_tolerance = (
            max(abs(float(transform[0])), abs(float(transform[4]))) * 1e-6
        )
        progress.emit(
            "link_objects",
            current=0,
            total=0,
            feature_count=registered_part_count,
            message="扫描相邻空间单元公共边界",
            force=True,
        )
        _link_neighbor_parts(
            database,
            run_id,
            stream_id,
            formal_by_unit,
            units,
            pixel_tolerance,
            progress_callback=lambda current, total, linked: progress.emit(
                "link_objects",
                current=current,
                total=total,
                feature_count=registered_part_count,
                message=f"已建立 {linked} 条跨单元对象连接",
            ),
        )
        link_count = database.object_link_count(run_id, stream_id)
        progress.emit(
            "link_objects",
            current=0,
            total=0,
            feature_count=registered_part_count,
            message=f"解析对象连接，当前连接 {link_count} 条",
            force=True,
        )
        object_count = database.resolve_object_components(run_id, stream_id)
        progress.emit(
            "link_objects",
            current=1,
            total=1,
            feature_count=registered_part_count,
            message=f"对象连接完成，共 {object_count} 个对象",
            force=True,
        )
        class_snapshot = load_json(Path(spec["class_mapping_snapshot"]))
        class_names = class_snapshot["class_mapping"]

        raw_feature_count = 0

        def write_raw(destination):
            def records():
                nonlocal raw_feature_count
                for unit_index, unit in enumerate(units, start=1):
                    unit_id = str(unit["unit_id"])
                    with fiona.open(
                        raw_by_unit[unit_id],
                        layer="polygons",
                    ) as source:
                        for feature in source:
                            raw_feature_count += 1
                            yield {
                                "geometry": feature["geometry"],
                                "properties": {
                                    "run_id": run_id,
                                    "stream_id": stream_id,
                                    "unit_id": unit_id,
                                    "polygon_id": str(
                                        feature["properties"]["polygon_id"]
                                    ),
                                    "class_code": int(
                                        feature["properties"]["class_code"]
                                    ),
                                },
                            }
                    progress.emit(
                        "write_raw",
                        current=unit_index,
                        total=len(units),
                        feature_count=raw_feature_count,
                        message=f"已写入 {raw_feature_count} 个 Raw 面",
                    )

            destination.writerecords(records())

        progress.emit(
            "write_raw",
            current=0,
            total=len(units),
            feature_count=0,
            message="创建 Raw GPKG",
            force=True,
        )
        _atomic_gpkg(
            raw_path,
            "semantic_polygons_raw",
            raw_schema,
            spec["raster"]["crs"],
            write_raw,
            storage_guard=storage_guard,
            storage_lock_path=storage_lock_path,
            estimated_write_bytes=_estimate_source_gpkg_bytes(
                tuple(raw_by_unit.values())
            ),
            operation=f"stream_raw:{stream_id}",
        )

        def write_formal(destination):
            def records():
                nonlocal formal_feature_count
                for unit_index, unit in enumerate(units, start=1):
                    unit_id = str(unit["unit_id"])
                    for features in _feature_batches(formal_by_unit[unit_id]):
                        part_ids = [
                            str(feature["properties"]["polygon_id"])
                            for feature in features
                        ]
                        object_ids = database.object_ids_for_parts(
                            run_id,
                            stream_id,
                            part_ids,
                        )
                        for feature in features:
                            geometry = shape(feature["geometry"])
                            if (
                                geometry.is_empty
                                or not geometry.is_valid
                                or geometry.area <= 0
                            ):
                                raise StreamAssemblyError(
                                    "formal output contains invalid geometry: "
                                    f"{unit_id}"
                                )
                            properties = feature["properties"]
                            part_id = str(properties["polygon_id"])
                            class_code = int(properties["class_code"])
                            formal_feature_count += 1
                            yield {
                                "geometry": feature["geometry"],
                                "properties": {
                                    "run_id": run_id,
                                    "result_stream_id": stream_id,
                                    "result_kind": str(stream["kind"]),
                                    "model_id": model_id,
                                    "fusion_profile_id": profile_id,
                                    "object_id": object_ids[part_id],
                                    "part_id": part_id,
                                    "class_code": class_code,
                                    "class_name": str(
                                        class_names[str(class_code)]
                                    ),
                                    "confidence_mean": float(
                                        properties.get("conf_mean", 0.0)
                                    ),
                                    "confidence_std": float(
                                        properties.get("conf_std", 0.0)
                                    ),
                                    "model_version": version,
                                    "source": (
                                        "semantic_model"
                                        if stream["kind"] == "model"
                                        else "semantic_fusion"
                                    ),
                                    "fit_changed": int(
                                        str(properties.get("fit_status"))
                                        == "changed"
                                    ),
                                    "fit_methods": str(
                                        properties.get("fit_method")
                                        or "unchanged"
                                    ),
                                    "fit_version": str(
                                        properties.get("fit_version")
                                        or fit_version
                                    ),
                                    "fit_status": str(
                                        properties.get("fit_status")
                                        or "unchanged"
                                    ),
                                    "origin_unit_ids": unit_id,
                                    "vertex_count_before": int(
                                        properties.get("vtx_before", 0)
                                    ),
                                    "vertex_count_after": int(
                                        properties.get("vtx_after", 0)
                                    ),
                                    "max_shift_px": float(
                                        properties.get("max_shift", 0.0)
                                    ),
                                    "mean_shift_px": float(
                                        properties.get("mean_shift", 0.0)
                                    ),
                                    "area_change_ratio": float(
                                        properties.get("area_ratio", 0.0)
                                    ),
                                    "created_at": now,
                                },
                            }
                    progress.emit(
                        "write_formal",
                        current=unit_index,
                        total=len(units),
                        feature_count=formal_feature_count,
                        message=f"已写入 {formal_feature_count} 个正式面",
                    )

            destination.writerecords(records())

        progress.emit(
            "write_formal",
            current=0,
            total=len(units),
            feature_count=0,
            message="创建正式 GPKG",
            force=True,
        )
        _atomic_gpkg(
            formal_path,
            "semantic_polygons",
            formal_schema,
            spec["raster"]["crs"],
            write_formal,
            storage_guard=storage_guard,
            storage_lock_path=storage_lock_path,
            estimated_write_bytes=_estimate_source_gpkg_bytes(
                tuple(formal_by_unit.values())
            ),
            operation=f"stream_formal:{stream_id}",
        )

    aggregate = {
        "schema_version": 1,
        "run_id": run_id,
        "stream_id": stream_id,
        "assembly_mode": "report_resume" if resume_from_reports else "full",
        "report_queue_capacity": ASSEMBLY_VALIDATION_MAX_IN_FLIGHT,
        "report_summary_source": "run_state_database",
        "report_processed_count": expected_units,
        "report_peak_loaded_count": 0,
        "report_json_parse_count": 0,
        "summary_validation_workers": summary_validation["workers"],
        "summary_validation_peak_in_flight": summary_validation[
            "peak_in_flight"
        ],
        "summary_validation_artifact_count": summary_validation[
            "artifact_count"
        ],
        "summary_validation_elapsed_sec": summary_validation["elapsed_sec"],
        "gpkg_write_mode": "gdal_batch_writerecords",
        "object_id_lookup_batch_size": OBJECT_ID_BATCH_SIZE,
        "fitted_edge_shard_count": len(edge_artifacts),
        "status": "passed",
        "smoothing_enabled": smoothing_enabled,
        "unit_count": expected_units,
        "object_count": object_count,
        "object_link_count": link_count,
        "fit_version": fit_version,
        "curve_sampling_spacing_px": float(
            boundary.get("curve_sampling_spacing_px", 0.5)
        ),
        "max_chord_error_limit_px": float(
            boundary.get("max_chord_error_px", 0.25)
        ),
        "max_segment_arc_length_limit_px": float(
            boundary.get("max_segment_arc_length_px", 8.0)
        ),
        "chain_count": int(summary_aggregate["chain_count"]),
        "shared_chain_count": int(summary_aggregate["shared_chain_count"]),
        "spline_count": int(summary_aggregate["spline_count"]),
        "unchanged_count": int(summary_aggregate["unchanged_count"]),
        "skipped_invalid_count": int(
            summary_aggregate["skipped_invalid_count"]
        ),
        "failed_unit_count": int(summary_aggregate["failed_unit_count"]),
        "max_displacement_px": float(
            summary_aggregate["max_displacement_px"]
        ),
        "diagnostic_count": int(summary_aggregate["diagnostic_count"]),
        "fitted_edge_count": int(summary_aggregate["fitted_edge_count"]),
        "validation": ownership_validation,
        "topology_checks_performed": False,
    }

    edge_metrics = {
        "dense_curve_point_count": 0,
        "sparse_curve_point_count": 0,
        "max_chord_error_px": 0.0,
        "max_segment_arc_length_px": 0.0,
    }
    edge_feature_count = 0

    def write_edges(destination):
        def records():
            nonlocal edge_feature_count
            for artifact_index, artifact in enumerate(edge_artifacts, start=1):
                with fiona.open(
                    artifact["path"],
                    layer="fitted_edges",
                ) as source:
                    for feature in source:
                        edge_feature_count += 1
                        properties = dict(feature["properties"])
                        edge_metrics["dense_curve_point_count"] += int(
                            properties.get("dense_vtx") or 0
                        )
                        edge_metrics["sparse_curve_point_count"] += int(
                            properties.get("sparse_vtx") or 0
                        )
                        edge_metrics["max_chord_error_px"] = max(
                            edge_metrics["max_chord_error_px"],
                            float(properties.get("chord_err") or 0.0),
                        )
                        edge_metrics["max_segment_arc_length_px"] = max(
                            edge_metrics["max_segment_arc_length_px"],
                            float(properties.get("arc_len") or 0.0),
                        )
                        yield {
                            "geometry": feature["geometry"],
                            "properties": properties,
                        }
                progress.emit(
                    "aggregate_reports",
                    current=artifact_index,
                    total=len(edge_artifacts),
                    feature_count=edge_feature_count,
                    message=f"已汇总 {edge_feature_count} 条拟合边界",
                )

        destination.writerecords(records())

    candidate_path = root / "semantic_candidates.gpkg"
    staged_edges_path = root / f".fitted_edges.{os.getpid()}.stage.gpkg"
    staged_report_path = root / f".boundary_fitting_report.{os.getpid()}.stage.json"
    staged_candidate_path = root / f".semantic_candidates.{os.getpid()}.stage.gpkg"
    staged_paths = (
        staged_edges_path,
        staged_report_path,
        staged_candidate_path,
    )
    for path in staged_paths:
        path.unlink(missing_ok=True)

    def build_report_outputs() -> bool:
        nonlocal formal_feature_count
        try:
            progress.emit(
                "aggregate_reports",
                current=0,
                total=max(1, len(edge_artifacts)),
                feature_count=0,
                message="汇总拟合报告与公共边界",
                force=True,
            )
            print(
                json.dumps(
                    {
                        "event": "report_assembly_started",
                        "run_id": run_id,
                        "stream_id": stream_id,
                        "assembly_mode": aggregate["assembly_mode"],
                        "total": expected_units,
                        "report_queue_capacity": aggregate[
                            "report_queue_capacity"
                        ],
                        "report_summary_source": "run_state_database",
                        "report_json_parse_count": 0,
                        "summary_validation_workers": aggregate[
                            "summary_validation_workers"
                        ],
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            _atomic_gpkg(
                staged_edges_path,
                "fitted_edges",
                edge_schema,
                spec["raster"]["crs"],
                write_edges,
                storage_guard=storage_guard,
                storage_lock_path=storage_lock_path,
                estimated_write_bytes=_estimate_source_gpkg_bytes(
                    tuple(str(item["path"]) for item in edge_artifacts)
                ),
                operation=f"stream_fitted_edges_stage:{stream_id}",
            )
            progress.emit(
                "aggregate_reports",
                current=max(1, len(edge_artifacts)),
                total=max(1, len(edge_artifacts)),
                feature_count=edge_feature_count,
                message="拟合报告与公共边界汇总完成",
                force=True,
            )
            aggregate.update(edge_metrics)
            dense_points = int(aggregate["dense_curve_point_count"])
            sparse_points = int(aggregate["sparse_curve_point_count"])
            aggregate["adaptive_point_reduction"] = (
                1.0 - sparse_points / dense_points
                if dense_points
                else 0.0
            )
            chord_limit = float(aggregate["max_chord_error_limit_px"])
            arc_limit = float(
                aggregate["max_segment_arc_length_limit_px"]
            )
            tolerance = 1e-9
            if aggregate["max_chord_error_px"] > chord_limit + tolerance:
                raise StreamAssemblyError(
                    "adaptive curve chord error exceeds configured limit: "
                    f"{aggregate['max_chord_error_px']} > {chord_limit}"
                )
            if (
                aggregate["max_segment_arc_length_px"]
                > arc_limit + tolerance
            ):
                raise StreamAssemblyError(
                    "adaptive curve arc length exceeds configured limit: "
                    f"{aggregate['max_segment_arc_length_px']} > {arc_limit}"
                )
            print(
                json.dumps(
                    {
                        "event": "report_assembly_completed",
                        "run_id": run_id,
                        "stream_id": stream_id,
                        "assembly_mode": aggregate["assembly_mode"],
                        "current": aggregate["report_processed_count"],
                        "total": expected_units,
                        "report_processed_count": aggregate[
                            "report_processed_count"
                        ],
                        "report_queue_capacity": aggregate[
                            "report_queue_capacity"
                        ],
                        "report_peak_loaded_count": aggregate[
                            "report_peak_loaded_count"
                        ],
                        "report_summary_source": aggregate[
                            "report_summary_source"
                        ],
                        "report_json_parse_count": aggregate[
                            "report_json_parse_count"
                        ],
                        "summary_validation_peak_in_flight": aggregate[
                            "summary_validation_peak_in_flight"
                        ],
                        "failed_unit_count": aggregate["failed_unit_count"],
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            fitting_passed = aggregate["failed_unit_count"] == 0
            aggregate["status"] = "passed" if fitting_passed else "failed"
            aggregate["validation"]["passed"] = bool(
                aggregate["validation"].get("passed") and fitting_passed
            )
            if not fitting_passed:
                raise StreamAssemblyError("boundary fitting contains failed units")
            progress.emit(
                "range_clip",
                current=0,
                total=1,
                feature_count=formal_feature_count,
                message=(
                    "校验已裁剪正式 GPKG"
                    if resume_from_reports
                    else "按冻结研究范围精确裁剪正式 GPKG"
                ),
                force=True,
            )
            if resume_from_reports:
                range_clip = {
                    "status": "already_clipped",
                    "reason": "resume formal output was range-validated before assembly",
                }
            else:
                range_clip = _guarded_range_clip(
                    formal_path,
                    spec,
                    storage_guard=storage_guard,
                    storage_lock_path=storage_lock_path,
                    operation=f"stream_range_clip:{stream_id}",
                )
            aggregate["range_clip"] = range_clip
            if "output_feature_count" in range_clip:
                formal_feature_count = int(range_clip["output_feature_count"])
            elif "source_feature_count" in range_clip:
                formal_feature_count = int(range_clip["source_feature_count"])
            progress.emit(
                "range_clip",
                current=1,
                total=1,
                feature_count=formal_feature_count,
                message="研究范围裁剪完成",
                force=True,
            )
            progress.emit(
                "coverage_validation",
                current=0,
                total=1,
                feature_count=formal_feature_count,
                message="核验研究范围内空白、重叠和范围外面积",
                force=True,
            )
            coverage = _validate_exact_range_coverage(
                formal_path,
                layer="semantic_polygons",
                spec=spec,
            )
            aggregate["coverage_validation"] = coverage
            _publish_coverage_validation(
                database,
                run_id,
                stream_id,
                coverage,
            )
            if coverage["status"] == "failed":
                progress.emit(
                    "coverage_validation",
                    current=1,
                    total=1,
                    feature_count=formal_feature_count,
                    status="failed",
                    message=(
                        f"空白 {coverage['gap_area_m2']:.6g} m²；"
                        f"重叠 {coverage['overlap_area_m2']:.6g} m²；"
                        f"范围外 {coverage['outside_area_m2']:.6g} m²"
                    ),
                    force=True,
                )
                raise StreamAssemblyError(
                    "exact range coverage validation failed: "
                    f"gap={coverage['gap_area_m2']:.6g} m2, "
                    f"overlap={coverage['overlap_area_m2']:.6g} m2, "
                    f"outside={coverage['outside_area_m2']:.6g} m2"
                )
            if coverage["status"] == "passed":
                coverage_message = "空白 0；重叠 0；范围外 0"
            else:
                coverage_message = "历史 Run 缺少冻结范围，覆盖验收未执行"
            progress.emit(
                "coverage_validation",
                current=1,
                total=1,
                feature_count=formal_feature_count,
                message=coverage_message,
                force=True,
            )
            if resume_from_reports:
                aggregate["input_sha256"] = resume_inputs["raw"]["sha256"]
                aggregate["output_sha256"] = resume_inputs["formal"]["sha256"]
            else:
                aggregate["input_sha256"] = sha256_file(raw_path)
                aggregate["output_sha256"] = sha256_file(formal_path)
            accepted_value = str(spec.get("accepted_gpkg") or "")
            accepted_sha = str(spec.get("accepted_gpkg_sha256") or "")
            if accepted_value and accepted_sha:
                accepted_path = Path(accepted_value)
                if (
                    not accepted_path.is_file()
                    or sha256_file(accepted_path) != accepted_sha
                ):
                    raise StreamAssemblyError(
                        "accepted_labels changed after run creation"
                    )
            progress.emit(
                "accepted_difference",
                current=0,
                total=1,
                feature_count=formal_feature_count,
                message="计算 Accepted 标签差分",
                force=True,
            )
            difference = _guarded_accepted_difference(
                formal_path,
                accepted_value,
                staged_candidate_path,
                storage_guard=storage_guard,
                storage_lock_path=storage_lock_path,
                operation=f"stream_candidate_stage:{stream_id}",
            )
            candidate_written = staged_candidate_path.is_file()
            if candidate_written and difference.get("output"):
                difference = dict(difference)
                difference["output"] = str(candidate_path)
            aggregate["difference"] = difference
            progress.emit(
                "accepted_difference",
                current=1,
                total=1,
                feature_count=formal_feature_count,
                message="Accepted 标签差分完成",
                force=True,
            )
            _write_json(
                staged_report_path,
                aggregate,
                storage_guard=storage_guard,
                storage_lock_path=storage_lock_path,
                operation=f"stream_report_stage:{stream_id}",
            )
            if resume_from_reports:
                _assert_fingerprint_unchanged(raw_path, resume_inputs["raw"])
                _assert_fingerprint_unchanged(formal_path, resume_inputs["formal"])
            if candidate_written:
                os.replace(staged_candidate_path, candidate_path)
            os.replace(staged_edges_path, fitted_edges_path)
            os.replace(staged_report_path, report_path)
            return candidate_written
        finally:
            for path in staged_paths:
                path.unlink(missing_ok=True)

    candidate_written = build_report_outputs()
    assembled_artifacts = [
        ("semantic_polygons_raw", raw_path),
        ("semantic_polygons", formal_path),
        ("boundary_fitting_report", report_path),
        ("fitted_edges", fitted_edges_path),
    ]
    if candidate_written:
        assembled_artifacts.append(("semantic_candidates", candidate_path))
    publish_total = len(assembled_artifacts) + 1
    progress.emit(
        "publish_cleanup",
        current=0,
        total=publish_total,
        feature_count=formal_feature_count,
        message="提交正式组装产物",
        force=True,
    )
    for artifact_index, (kind, path) in enumerate(
        assembled_artifacts, start=1
    ):
        _commit_artifact(
            database,
            run_id,
            path=path,
            kind=kind,
            stream_id=stream_id,
            unit_id="assembled",
        )
        progress.emit(
            "publish_cleanup",
            current=artifact_index,
            total=publish_total,
            feature_count=formal_feature_count,
            message=f"已提交 {artifact_index}/{len(assembled_artifacts)} 个正式产物",
        )
    aggregate["unit_artifact_cleanup"] = _cleanup_stream_unit_artifacts(
        spec, database, stream_id
    )
    database.set_stream_status(
        run_id,
        stream_id,
        "ready",
        error="",
    )
    progress.emit(
        "publish_cleanup",
        current=publish_total,
        total=publish_total,
        feature_count=formal_feature_count,
        status="completed",
        message="正式产物已提交，中间文件清理完成",
        force=True,
    )
    print(json.dumps({"event": "stream_assembled", **aggregate}, separators=(",", ":")))
    return aggregate


def assemble_stream(
    run_spec_path: str | Path,
    stream_id: str,
    *,
    resume_from_reports: bool = False,
) -> dict[str, Any]:
    try:
        return _assemble_stream_impl(
            run_spec_path,
            stream_id,
            resume_from_reports=resume_from_reports,
        )
    except Exception as error:
        try:
            spec = load_json(Path(run_spec_path).resolve())
            if spec.get("schema_version") == 2:
                database = RunStateDB(spec["state_db"])
                database.set_stream_status(
                    str(spec["run_id"]),
                    str(stream_id),
                    "failed",
                    error=str(error),
                )
                database.fail_stream_runtime_progress(
                    str(spec["run_id"]),
                    str(stream_id),
                    str(error),
                )
        except Exception:
            pass
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Assemble one completed result stream")
    parser.add_argument("--run-spec", required=True)
    parser.add_argument("--stream-id", required=True)
    parser.add_argument(
        "--resume-from-reports",
        action="store_true",
        help="reuse validated raw/formal outputs and continue from boundary reports",
    )
    args = parser.parse_args(argv)
    try:
        report = assemble_stream(
            args.run_spec,
            args.stream_id,
            resume_from_reports=args.resume_from_reports,
        )
        if report.get("status") != "passed":
            print(
                json.dumps(
                    {
                        "event": "stream_assembly_failed",
                        "assembly_mode": report.get("assembly_mode")
                        or (
                            "report_resume"
                            if args.resume_from_reports
                            else "full"
                        ),
                        "error": "boundary fitting contains failed units",
                    }
                )
            )
            return 2
        return 0
    except Exception as error:
        failure = {
            "event": "stream_assembly_failed",
            "assembly_mode": (
                "report_resume" if args.resume_from_reports else "full"
            ),
            "error": str(error),
        }
        if args.resume_from_reports:
            failure["safe_retry"] = "rerun_without_resume_from_reports"
        print(json.dumps(failure))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
