"""Create a v5 run whose detailed state lives in PostgreSQL."""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from labeling_tool.runs.run_build_contract import RunBuilderV5Error
from labeling_tool.runs.run_build_preparation import prepare_v5_run_plan
from labeling_tool.runs.run_build_snapshots import freeze_v5_run_snapshots
from labeling_tool.runs.run_control_graph import write_v5_control_graph
from labeling_tool.runs.run_index import record_run_state
from labeling_tool.shared.contracts.run_spec import (
    RESERVATION_FILE,
    RunSpecError,
    atomic_write_json,
    reserve_run_directory,
    run_tile_cache_dir,
    sha256_file,
    source_raster_identity,
    validate_source_raster,
)
from labeling_tool.shared.state.postgres_state import is_postgres_location
from labeling_tool.shared.state.run_state_db import (
    RunStateDB,
    production_state_database,
    production_state_schema,
)

RUN_SPEC_SCHEMA_VERSION = 2
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
logger = logging.getLogger("labeling_tool.run_builder_v5")


class RunBuilderV5Cancelled(RuntimeError):
    """Raised only when an asynchronous Run build is explicitly cancelled."""


def _build_checkpoint(
    *,
    progress: Callable[[float, str], None] | None,
    is_canceled: Callable[[], bool] | None,
    value: float,
    message: str,
) -> None:
    if is_canceled is not None and is_canceled():
        raise RunBuilderV5Cancelled("Run task graph creation was cancelled")
    if progress is not None:
        progress(float(value), str(message))


def _extent(value: Mapping[str, Any]) -> dict[str, float]:
    result = {key: float(value[key]) for key in ("xmin", "ymin", "xmax", "ymax")}
    if result["xmin"] >= result["xmax"] or result["ymin"] >= result["ymax"]:
        raise RunBuilderV5Error("run extent must have positive width and height")
    return result


def _json_sha(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deployment_identity(project_root: str | Path | None) -> dict[str, Any]:
    """Freeze only verifiable, non-secret deployment provenance.

    A Run is created from the deployed project, which can be a Git worktree or
    a release archive.  The deployment manifest is the common contract between
    those two forms.  Do not query the local checkout here: it may be unrelated
    to the runtime project and would fabricate provenance for the Run.
    """

    unknown = {
        "schema_version": 1,
        "status": "unknown",
        "project_manifest_sha256": "unknown",
        "project_manifest_schema_version": "unknown",
        "git_sha": "unknown",
        "source_bundle_sha256": "unknown",
        "source_kind": "unknown",
        "git_dirty": "unknown",
        "verification_scope": "none",
    }
    if not project_root:
        return unknown
    root = Path(project_root).expanduser()
    try:
        root = root.resolve()
    except OSError:
        return unknown
    manifest_path = root / "project_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return unknown
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError:
        return unknown
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return {
            **unknown,
            "status": "manifest_unreadable",
            "project_manifest_sha256": manifest_sha256,
        }
    if not isinstance(manifest, Mapping):
        return {
            **unknown,
            "status": "manifest_invalid",
            "project_manifest_sha256": manifest_sha256,
        }
    source = manifest.get("source")
    source = source if isinstance(source, Mapping) else {}
    git_sha = str(manifest.get("git_sha") or "")
    source_bundle_sha256 = str(source.get("source_bundle_sha256") or "")
    source_kind = str(source.get("kind") or "")
    git_dirty = source.get("git_dirty")
    identity = {
        "schema_version": 1,
        "status": "manifest_recorded"
        if (
            manifest.get("schema_version") == 2
            and manifest.get("deployment_kind") == "loess_project"
            and _GIT_SHA_RE.fullmatch(git_sha)
            and _SHA256_RE.fullmatch(source_bundle_sha256)
            and source_kind in {"git_worktree", "release_archive"}
            and isinstance(git_dirty, bool)
        )
        else "manifest_incomplete",
        "verification_scope": "manifest_fields_and_digest_only",
        "project_manifest_sha256": manifest_sha256,
        "project_manifest_schema_version": manifest.get("schema_version", "unknown"),
        "git_sha": git_sha if _GIT_SHA_RE.fullmatch(git_sha) else "unknown",
        "source_bundle_sha256": (
            source_bundle_sha256
            if _SHA256_RE.fullmatch(source_bundle_sha256)
            else "unknown"
        ),
        "source_kind": source_kind or "unknown",
        "git_dirty": git_dirty if isinstance(git_dirty, bool) else "unknown",
    }
    return identity


def create_v5_run(
    *,
    output_root: str | Path,
    raster: Mapping[str, Any],
    requested_extent: Mapping[str, Any],
    processing_extent: Mapping[str, Any],
    tile_rows: int,
    tile_cols: int,
    tiles: Iterable[Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]],
    effective_device: str,
    keep_score_cache: bool = False,
    tile_batch_size: int = 1,
    resource_tuning: Mapping[str, Any] | None = None,
    overlap: int,
    scaling: Mapping[str, Any],
    boundary_fitting: Mapping[str, Any],
    storage_report: Mapping[str, Any],
    fragmentation_regularization: Mapping[str, Any] | None = None,
    fusion: Mapping[str, Any] | None = None,
    accepted_gpkg: str | Path = "",
    accepted_target_gpkg: str | Path = "",
    accepted_validation: Mapping[str, Any] | None = None,
    skip_accepted: bool = True,
    config_fingerprint: str = "",
    range_selection: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    reserved_run_dir: str | Path | None = None,
    state_database: str | Path | None = None,
    deployment_project_root: str | Path | None = None,
    progress: Callable[[float, str], None] | None = None,
    is_canceled: Callable[[], bool] | None = None,
) -> tuple[dict[str, Any], Path, str | Path]:
    """Freeze a Run Spec and atomically populate its PostgreSQL control graph."""
    if not models:
        raise RunBuilderV5Error("at least one semantic model is required")
    state_location = str(state_database or production_state_database()).strip()
    if not is_postgres_location(state_location):
        raise RunBuilderV5Error(
            "v5 Run state requires a PostgreSQL DSN; filesystem databases are "
            "no longer supported"
        )
    state_schema = production_state_schema()
    raster_path = Path(str(raster["path"])).expanduser().resolve()
    try:
        raster_identity = source_raster_identity(raster_path)
    except RunSpecError as error:
        raise RunBuilderV5Error(str(error)) from error
    output = Path(output_root).expanduser().resolve()
    if reserved_run_dir is None:
        identifier, run_dir = reserve_run_directory(output, run_id)
    else:
        run_dir = Path(reserved_run_dir).expanduser().resolve()
        identifier = run_id or run_dir.name
        if run_dir != output / "runs" / identifier:
            raise RunBuilderV5Error(
                "reserved run directory is outside the output workspace"
            )
        if not (run_dir / RESERVATION_FILE).is_file():
            raise RunBuilderV5Error("reserved run directory has already been consumed")

    _build_checkpoint(
        progress=progress,
        is_canceled=is_canceled,
        value=0,
        message="正在验证 Run 参数",
    )

    def checkpoint(value: float, message: str) -> None:
        _build_checkpoint(
            progress=progress,
            is_canceled=is_canceled,
            value=value,
            message=message,
        )

    plan = prepare_v5_run_plan(
        run_dir=run_dir,
        tile_rows=int(tile_rows),
        tile_cols=int(tile_cols),
        overlap=int(overlap),
        scaling=scaling,
        boundary_fitting=boundary_fitting,
        fragmentation_regularization=fragmentation_regularization,
        range_selection=range_selection,
        storage_report=storage_report,
        checkpoint=checkpoint,
    )
    snapshots = freeze_v5_run_snapshots(
        run_dir=run_dir,
        models=models,
        fusion=fusion,
        effective_device=effective_device,
        keep_score_cache=keep_score_cache,
        tile_batch_size=tile_batch_size,
        resource_tuning=resource_tuning,
        plan=plan,
        config_fingerprint=config_fingerprint,
    )

    accepted_path = (
        Path(accepted_gpkg).expanduser().resolve() if accepted_gpkg else None
    )
    accepted_target_path = (
        Path(accepted_target_gpkg).expanduser().resolve()
        if accepted_target_gpkg
        else None
    )
    accepted_sha256 = (
        sha256_file(accepted_path)
        if accepted_path is not None and accepted_path.is_file()
        else ""
    )

    for model in snapshots.models:
        (run_dir / "models" / str(model["model_id"]) / "raster_parts").mkdir(
            parents=True, exist_ok=True
        )
    if snapshots.fusion:
        (
            run_dir / "fusion" / str(snapshots.fusion["profile_id"]) / "raster_parts"
        ).mkdir(parents=True, exist_ok=True)

    spec = {
        "schema_version": RUN_SPEC_SCHEMA_VERSION,
        "run_id": identifier,
        "created_at": _datetime.datetime.now()
        .astimezone()
        .isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "output_root": str(output),
        "cache_root": str(run_tile_cache_dir(output, identifier).parent),
        "tile_cache_dir": str(run_tile_cache_dir(output, identifier)),
        "raster": {
            "path": str(raster_path),
            "file_identity": raster_identity,
            "crs": str(raster["crs"]),
            "transform": [float(value) for value in raster["transform"]],
            "nodata": raster.get("nodata"),
        },
        "requested_extent": _extent(requested_extent),
        "processing_extent": _extent(processing_extent),
        "range_selection": plan.range_selection,
        "range_vector_path": str(
            plan.range_selection.get("vector_source")
            or plan.range_selection.get("vector_path")
            or ""
        ),
        "tile_grid": {
            "rows": int(tile_rows),
            "cols": int(tile_cols),
            "count": int(tile_rows) * int(tile_cols),
            "selected_count": plan.selected_tile_count,
            "excluded_count": plan.excluded_tile_count,
            "width": 512,
            "height": 512,
            "overlap": int(overlap),
            "stride": 512 - int(overlap),
        },
        "spatial_plan_summary": {
            "partition_rows": plan.spatial_plan["partition_rows"],
            "partition_cols": plan.spatial_plan["partition_cols"],
            "partition_count": plan.spatial_plan["partition_count"],
            "unit_counts": plan.spatial_plan["unit_counts"],
            "package_count": plan.package_plan["package_count"],
        },
        "runtime": {
            "effective_device": str(effective_device),
            "keep_score_cache": bool(keep_score_cache),
            "tile_batch_size": max(1, int(tile_batch_size)),
        },
        "resource_tuning": dict(resource_tuning or {}),
        "scaling": plan.scaling,
        "boundary_fitting": plan.boundary_fitting,
        "fragmentation_regularization": plan.fragmentation_regularization,
        "coverage_validation": {
            "policy_id": "exact_range_zero_gap_v1",
            "area_tolerance_pixels": 0.01,
        },
        "storage_preflight": plan.storage_report,
        "models": snapshots.models,
        "fusion": snapshots.fusion,
        "streams": snapshots.streams,
        "accepted_gpkg": str(accepted_path) if accepted_path is not None else "",
        "accepted_gpkg_sha256": accepted_sha256,
        "accepted_target_gpkg": (
            str(accepted_target_path) if accepted_target_path is not None else ""
        ),
        "accepted_validation": dict(accepted_validation or {}),
        "skip_accepted": bool(skip_accepted),
        "class_mapping_snapshot": str(snapshots.class_mapping_path),
        "config_snapshot": str(snapshots.config_snapshot_path),
        "config_fingerprint": str(config_fingerprint),
        "deployment_identity": deployment_identity(deployment_project_root),
        "state_backend": "postgresql",
        "state_db": state_location,
        "state_schema": state_schema,
    }
    try:
        validate_source_raster(spec["raster"])
    except RunSpecError as error:
        raise RunBuilderV5Error(str(error)) from error
    spec["run_spec_content_sha256"] = _json_sha(spec)
    spec_path = run_dir / "run_spec.json"
    atomic_write_json(spec_path, spec)
    _build_checkpoint(
        progress=progress,
        is_canceled=is_canceled,
        value=12,
        message="Run Spec 已冻结",
    )

    database_location = state_location
    database = RunStateDB(database_location, postgres_schema=state_schema)
    database.initialize()
    write_v5_control_graph(
        database=database,
        run_id=identifier,
        spec_path=spec_path,
        tile_rows=int(tile_rows),
        tile_cols=int(tile_cols),
        tile_cache_dir=run_tile_cache_dir(output, identifier),
        excluded_tile_count=plan.excluded_tile_count,
        spatial_plan=plan.spatial_plan,
        package_plan=plan.package_plan,
        partitions=plan.partitions,
        streams=snapshots.streams,
        tiles=tiles,
        max_job_retries=int(plan.scaling["max_job_retries"]),
        v33_enabled=plan.v33_enabled,
        checkpoint=checkpoint,
    )
    try:
        incomplete_run_cleanup = database.run_archive.archive_incomplete_run_details(
            protected_run_id=identifier,
        )
    except Exception as exc:
        incomplete_run_cleanup = {
            "schema_version": 1,
            "status": "warning",
            "protected_run_id": identifier,
            "error": str(exc),
        }
        logger.warning(
            "[incomplete-run-cleanup-warning] 旧未完成 Run 明细归档失败: %s",
            exc,
        )
    try:
        database.run_streams.update_run_metadata(
            identifier,
            {"incomplete_run_cleanup": incomplete_run_cleanup},
        )
        archived_count = int(incomplete_run_cleanup.get("archived_run_count") or 0)
        skipped_active_count = int(
            incomplete_run_cleanup.get("skipped_active_run_count") or 0
        )
        cleanup_warning = (
            incomplete_run_cleanup.get("status") == "warning"
            or skipped_active_count > 0
        )
        if archived_count or cleanup_warning:
            database.run_streams.append_event(
                identifier,
                "incomplete_run_cleanup",
                level="warning" if cleanup_warning else "info",
                message=(
                    str(incomplete_run_cleanup.get("error") or "")
                    if incomplete_run_cleanup.get("status") == "warning"
                    else (
                        "Skipped old incomplete Runs with active Jobs: "
                        + ", ".join(
                            incomplete_run_cleanup.get("skipped_active_run_ids") or []
                        )
                    )
                    if skipped_active_count
                    else "Archived old incomplete Run database details"
                ),
                payload=incomplete_run_cleanup,
            )
    except Exception as exc:
        logger.warning(
            "[incomplete-run-cleanup-warning] 无法记录旧 Run 归档结果: %s",
            exc,
        )
    try:
        (run_dir / RESERVATION_FILE).unlink()
    except FileNotFoundError:
        pass
    try:
        record_run_state(output, identifier, status="planned")
    except (OSError, ValueError) as exc:
        logger.warning("无法更新轻量 Run 启动索引: %s", exc)
    _build_checkpoint(
        progress=progress,
        is_canceled=is_canceled,
        value=100,
        message="Run 任务图建立完成",
    )
    return spec, spec_path, database_location
