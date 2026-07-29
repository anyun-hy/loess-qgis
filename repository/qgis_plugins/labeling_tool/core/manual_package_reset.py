"""Transactional manual reset for failed v5 Work Packages."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path
from typing import Any, Iterable

from .run_state_db import RunStateDB


class ManualPackageResetError(RuntimeError):
    pass


def _run_owned_path(run_dir: Path, raw_path: str | Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    candidate = Path(os.path.abspath(candidate))
    try:
        relative = candidate.relative_to(run_dir)
    except ValueError as error:
        raise ManualPackageResetError(
            f"refusing to remove a path outside the Run directory: {candidate}"
        ) from error
    current = run_dir
    for part in relative.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise ManualPackageResetError(
                f"refusing to follow a symlink during Package reset: {current}"
            )
    return candidate


def _remove_file(path: Path) -> tuple[int, int]:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return 0, 0
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ManualPackageResetError(
            f"Package reset expected a regular file: {path}"
        )
    byte_count = int(metadata.st_size)
    path.unlink()
    return 1, byte_count


def _remove_directory(path: Path) -> tuple[int, int]:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return 0, 0
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ManualPackageResetError(
            f"Package reset expected a regular directory: {path}"
        )
    file_count = 0
    byte_count = 0
    for root, directories, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in directories:
            child = root_path / name
            if child.is_symlink():
                raise ManualPackageResetError(
                    f"refusing to remove a symlink inside Package output: {child}"
                )
        for name in files:
            child = root_path / name
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ManualPackageResetError(
                    f"refusing to remove a symlink inside Package output: {child}"
                )
            file_count += 1
            byte_count += int(metadata.st_size)
    shutil.rmtree(path)
    return file_count, byte_count


def _stream_root(spec: dict[str, Any], stream: dict[str, Any]) -> Path:
    run_dir = Path(str(spec["run_dir"]))
    if stream["kind"] == "model":
        return run_dir / "models" / str(stream["model_id"])
    return run_dir / "fusion" / str(stream["profile_id"])


def _matching_files(directory: Path, prefixes: Iterable[str]) -> set[Path]:
    if not directory.is_dir():
        return set()
    values = tuple(str(item) for item in prefixes)
    return {
        path
        for path in directory.iterdir()
        if path.is_file() and path.name.startswith(values)
    }


def _gpkg_sidecars(path: Path) -> set[Path]:
    if path.suffix.lower() != ".gpkg":
        return set()
    return {
        Path(str(path) + "-wal"),
        Path(str(path) + "-shm"),
        Path(str(path) + "-journal"),
    }


def _reset_file_paths(
    spec: dict[str, Any],
    plan: dict[str, Any],
    run_dir: Path,
) -> set[Path]:
    paths = {
        _run_owned_path(run_dir, str(item["path"]))
        for item in plan["artifacts"]
    }
    unit_ids = tuple(str(item) for item in plan["affected_unit_ids"])
    for stream in spec.get("streams") or []:
        stream_id = str(stream["stream_id"])
        unit_root = _run_owned_path(
            run_dir,
            run_dir / "tmp" / "unit_outputs" / stream_id.replace(":", "_"),
        )
        for unit_id in unit_ids:
            stems = (
                f"{unit_id}_raw",
                f"{unit_id}_formal",
                f"{unit_id}_report",
                f"{unit_id}_fitted_edges",
            )
            for name in (
                f"{unit_id}_raw.gpkg",
                f"{unit_id}_formal.gpkg",
                f"{unit_id}_report.json",
                f"{unit_id}_fitted_edges.gpkg",
            ):
                paths.add(_run_owned_path(run_dir, unit_root / name))
            paths.update(
                _run_owned_path(run_dir, item)
                for item in _matching_files(
                    unit_root,
                    (f".{stem}." for stem in stems),
                )
            )

        root = _run_owned_path(run_dir, _stream_root(spec, stream))
        aggregate_names = (
            "mask_mosaic.vrt",
            "confidence_mosaic.vrt",
            "semantic_polygons_raw.gpkg",
            "semantic_polygons.gpkg",
            "boundary_fitting_report.json",
            "fitted_edges.gpkg",
            "semantic_candidates.gpkg",
        )
        paths.update(
            _run_owned_path(run_dir, root / name) for name in aggregate_names
        )
        paths.update(
            _run_owned_path(run_dir, item)
            for item in _matching_files(
                root,
                (
                    ".semantic_polygons_raw.",
                    ".semantic_polygons.",
                    ".boundary_fitting_report.",
                    ".fitted_edges.",
                    ".semantic_candidates.",
                ),
            )
        )

    for unit_id in unit_ids:
        for stream in spec.get("streams") or []:
            marker = (
                run_dir
                / "tmp"
                / "failed_jobs"
                / (
                    f"{str(stream['stream_id']).replace(':', '_')}__"
                    f"{unit_id}_force_split.json"
                )
            )
            paths.add(_run_owned_path(run_dir, marker))
    for path in (
        run_dir / "run_manifest.json",
        run_dir / "logs" / "run_report.json",
        run_dir / "logs" / "failures.json",
        run_dir / "logs" / "scale_acceptance_report.json",
    ):
        paths.add(_run_owned_path(run_dir, path))
    paths.update(
        sidecar
        for path in tuple(paths)
        for sidecar in _gpkg_sidecars(path)
    )
    return paths


def reset_failed_work_packages(
    spec: dict[str, Any],
    *,
    database: RunStateDB | None = None,
) -> dict[str, Any]:
    """Delete failed Package outputs, preserve shared Tiles, and requeue work."""
    if int(spec.get("schema_version") or 0) != 2:
        raise ManualPackageResetError(
            "manual Package reset requires run_spec schema 2"
        )
    run_id = str(spec["run_id"])
    declared_run_dir = Path(str(spec["run_dir"])).expanduser()
    if declared_run_dir.is_symlink():
        raise ManualPackageResetError(
            f"Run directory is missing or unsafe: {declared_run_dir}"
        )
    run_dir = declared_run_dir.resolve()
    if not run_dir.is_dir():
        raise ManualPackageResetError(
            f"Run directory is missing or unsafe: {run_dir}"
        )
    state = database or RunStateDB(spec["state_db"])
    plan = state.begin_failed_package_reset(run_id)
    deleted_files = 0
    deleted_bytes = 0
    deleted_directories = 0
    try:
        for path in sorted(
            _reset_file_paths(spec, plan, run_dir),
            key=lambda item: (len(item.parts), str(item)),
            reverse=True,
        ):
            count, byte_count = _remove_file(path)
            deleted_files += count
            deleted_bytes += byte_count
        for package_id in plan["package_ids"]:
            package_root = _run_owned_path(
                run_dir,
                run_dir / "tmp" / "work_packages" / str(package_id),
            )
            count, byte_count = _remove_directory(package_root)
            deleted_files += count
            deleted_bytes += byte_count
            if count or byte_count:
                deleted_directories += 1
        completed = state.complete_failed_package_reset(
            run_id,
            plan["package_ids"],
        )
    except Exception as error:
        state.append_event(
            run_id,
            "manual_package_reset_failed",
            level="error",
            message=str(error),
            payload={
                "package_ids": plan["package_ids"],
                "deleted_file_count": deleted_files,
                "deleted_bytes": deleted_bytes,
                "tile_cache_action": "preserved",
            },
        )
        raise
    result = {
        **completed,
        "run_id": run_id,
        "package_ids": list(plan["package_ids"]),
        "partition_count": len(plan["partition_ids"]),
        "deleted_file_count": deleted_files,
        "deleted_directory_count": deleted_directories,
        "deleted_bytes": deleted_bytes,
        "tile_cache_action": "preserved",
        "tile_cache_dir": str(spec.get("tile_cache_dir") or ""),
    }
    state.append_event(
        run_id,
        "manual_package_reset_files_removed",
        payload=result,
    )
    return result
