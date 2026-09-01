"""Create a v5 run whose detailed state lives in PostgreSQL."""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .run_index import record_run_state
from .run_spec import (
    CLASS_NAMES,
    CLASS_ORDER,
    RESERVATION_FILE,
    atomic_write_json,
    reserve_run_directory,
    run_tile_cache_dir,
    sha256_file,
)
from .postgres_state import is_postgres_location
from .run_state_db import (
    RunStateDB,
    production_state_database,
    production_state_schema,
)
from .spatial_planner import plan_spatial_units
from .work_package_planner import plan_work_packages


RUN_SPEC_SCHEMA_VERSION = 2
V3_POLICY_ID = "semantic_optimized_200_v3"
V33_POLICY_ID = "fragmentation_v33_configurable_absorption_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
logger = logging.getLogger("labeling_tool.run_builder_v5")


class RunBuilderV5Error(ValueError):
    pass


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


def _fragmentation_v33_units(
    partitions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Plan durable owner-Core jobs and their one-way publication barrier."""

    if not partitions:
        raise RunBuilderV5Error("V3.3 production stage requires Partitions")
    global_window = {
        "x0": min(int(item["core_window"]["x0"]) for item in partitions),
        "y0": min(int(item["core_window"]["y0"]) for item in partitions),
        "x1": max(int(item["core_window"]["x1"]) for item in partitions),
        "y1": max(int(item["core_window"]["y1"]) for item in partitions),
    }
    units: list[dict[str, Any]] = []
    for owner in partitions:
        core = owner["core_window"]
        expanded = {
            "x0": max(global_window["x0"], int(core["x0"]) - 256),
            "y0": max(global_window["y0"], int(core["y0"]) - 256),
            "x1": min(global_window["x1"], int(core["x1"]) + 256),
            "y1": min(global_window["y1"], int(core["y1"]) + 256),
        }
        dependencies = [
            str(candidate["partition_id"])
            for candidate in partitions
            if not (
                int(candidate["core_window"]["x1"]) <= expanded["x0"]
                or int(candidate["core_window"]["x0"]) >= expanded["x1"]
                or int(candidate["core_window"]["y1"]) <= expanded["y0"]
                or int(candidate["core_window"]["y0"]) >= expanded["y1"]
            )
        ]
        partition_id = str(owner["partition_id"])
        units.append(
            {
                "unit_id": f"fragmentation_v33_partition:{partition_id}",
                "unit_type": "FragmentationV33Partition",
                "owner_key": partition_id,
                "pixel_window": dict(core),
                "dependency_ids": dependencies,
            }
        )
    units.append(
        {
            "unit_id": "fragmentation_v33_finalize",
            "unit_type": "FragmentationV33Finalize",
            "owner_key": "all_partition_owner_cores",
            "pixel_window": global_window,
            "dependency_ids": [str(item["partition_id"]) for item in partitions],
        }
    )
    return units


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
    output = Path(output_root).expanduser().resolve()
    if reserved_run_dir is None:
        identifier, run_dir = reserve_run_directory(output, run_id)
    else:
        run_dir = Path(reserved_run_dir).expanduser().resolve()
        identifier = run_id or run_dir.name
        if run_dir != output / "runs" / identifier:
            raise RunBuilderV5Error("reserved run directory is outside the output workspace")
        if not (run_dir / RESERVATION_FILE).is_file():
            raise RunBuilderV5Error("reserved run directory has already been consumed")

    scaling_value = dict(scaling)
    boundary_value = dict(boundary_fitting)
    if not isinstance(boundary_value.get("enabled"), bool):
        raise RunBuilderV5Error("boundary_fitting.enabled must be true or false")
    boundary_value.setdefault("mode", "divider_cubic_bspline_adaptive_v2")
    if (
        str(boundary_value.get("mode") or "")
        != "divider_cubic_bspline_adaptive_v2"
    ):
        raise RunBuilderV5Error(
            "boundary_fitting.mode must equal "
            "divider_cubic_bspline_adaptive_v2"
        )
    fragmentation_value = dict(fragmentation_regularization or {})
    fragmentation_value.setdefault("enabled", True)
    fragmentation_value.setdefault("policy_id", V3_POLICY_ID)
    fragmentation_value.setdefault(
        "policy_version",
        (
            "v33_production_20260826"
            if fragmentation_value["policy_id"] == V33_POLICY_ID
            else "semantic_optimized_200_v3_core_bounded_v1"
        ),
    )
    fragmentation_value.setdefault("baseline_policy_id", V3_POLICY_ID)
    fragmentation_value.setdefault(
        "baseline_policy_version", "semantic_optimized_200_v3_core_bounded_v1"
    )
    fragmentation_value.setdefault("buffer_pixels", 256)
    fragmentation_value.setdefault("max_workers", 4)
    if not isinstance(fragmentation_value.get("enabled"), bool):
        raise RunBuilderV5Error(
            "fragmentation_regularization.enabled must be true or false"
        )
    if fragmentation_value.get("policy_id") not in {V3_POLICY_ID, V33_POLICY_ID}:
        raise RunBuilderV5Error(
            "fragmentation_regularization.policy_id is unsupported"
        )
    if fragmentation_value.get("baseline_policy_id") != V3_POLICY_ID:
        raise RunBuilderV5Error(
            "fragmentation_regularization.baseline_policy_id must equal "
            + V3_POLICY_ID
        )
    if int(fragmentation_value.get("buffer_pixels") or 0) != 256:
        raise RunBuilderV5Error(
            "fragmentation_regularization.buffer_pixels must equal 256"
        )
    if not 1 <= int(fragmentation_value.get("max_workers") or 0) <= 4:
        raise RunBuilderV5Error(
            "fragmentation_regularization.max_workers must be between 1 and 4"
        )
    v33_enabled = bool(
        fragmentation_value["enabled"]
        and fragmentation_value["policy_id"] == V33_POLICY_ID
    )
    if v33_enabled:
        if fragmentation_value.get("publication") != "authoritative_fusion_core":
            raise RunBuilderV5Error(
                "V3.3 publication must equal authoritative_fusion_core"
            )
        for key in ("policy_sha256", "executor_sha256"):
            digest = str(fragmentation_value.get(key) or "").lower()
            if len(digest) != 64:
                raise RunBuilderV5Error(
                    f"fragmentation_regularization.{key} is required for V3.3"
                )
            try:
                int(digest, 16)
            except ValueError as error:
                raise RunBuilderV5Error(
                    f"fragmentation_regularization.{key} is invalid"
                ) from error
            fragmentation_value[key] = digest
    # The caller's spatial preflight, Package plan, and storage reservation all
    # depend on this value.  Do not silently enlarge it here: that would make
    # the frozen plan disagree with its preflight.  GUI/CLI entry points must
    # resolve auto halo with this requirement before calling the builder.
    if (
        fragmentation_value["enabled"]
        and int(scaling_value["partition_halo_px"])
        < int(fragmentation_value["buffer_pixels"])
    ):
        raise RunBuilderV5Error(
            "partition_halo_px must be at least "
            "fragmentation_regularization.buffer_pixels"
        )
    range_value = dict(range_selection or {})
    range_mode = str(range_value.get("mode") or "extent")
    if range_mode not in {"extent", "vector_tile_intersection"}:
        raise RunBuilderV5Error(f"unsupported range selection mode: {range_mode}")
    range_value["mode"] = range_mode
    if range_mode == "extent":
        # Tile expansion is only a processing detail. View and hand-drawn
        # runs are always published against their exact requested rectangle.
        range_value["clip_outputs"] = True
    if range_mode == "vector_tile_intersection":
        if range_value.get("clip_outputs") is not True:
            raise RunBuilderV5Error(
                "vector Tile selection must clip outputs to the exact vector boundary"
            )
        source_value = str(
            range_value.get("vector_source") or range_value.get("vector_path") or ""
        )
        snapshot_path = Path(source_value.split("|", 1)[0]).expanduser().resolve()
        try:
            snapshot_path.relative_to(run_dir)
        except ValueError as error:
            raise RunBuilderV5Error(
                "vector range source must be a run-local frozen snapshot"
            ) from error
        if not snapshot_path.is_file() or snapshot_path.suffix.lower() != ".gpkg":
            raise RunBuilderV5Error(
                "vector range snapshot is missing or is not a GeoPackage"
            )
        snapshot_sha256 = sha256_file(snapshot_path)
        supplied_sha256 = str(range_value.get("vector_sha256") or "")
        if supplied_sha256 and supplied_sha256 != snapshot_sha256:
            raise RunBuilderV5Error("vector range snapshot changed before run creation")
        range_value.update(
            {
                "vector_source": str(snapshot_path),
                "vector_path": str(snapshot_path),
                "vector_sha256": snapshot_sha256,
                "clip_outputs": True,
            }
        )
    grid_count = int(tile_rows) * int(tile_cols)
    selected_count = int(range_value.get("selected_tile_count", grid_count))
    excluded_count = int(range_value.get("excluded_tile_count", 0))
    if selected_count < 1 or excluded_count < 0 or selected_count + excluded_count != grid_count:
        raise RunBuilderV5Error(
            "range Tile counts must be positive and cover the declared grid"
        )
    spatial_plan = plan_spatial_units(
        tile_rows=int(tile_rows),
        tile_cols=int(tile_cols),
        tile_size=512,
        overlap=int(overlap),
        partition_tile_rows=int(scaling_value["partition_tile_rows"]),
        partition_tile_cols=int(scaling_value["partition_tile_cols"]),
        seam_band_px=int(scaling_value["seam_band_px"]),
        halo_px=int(scaling_value["partition_halo_px"]),
    )
    package_plan = plan_work_packages(
        spatial_plan,
        package_tile_limit=int(storage_report["package_tile_limit"]),
        estimated_bytes_per_tile=int(storage_report["working_bytes_per_tile"]),
    )
    package_by_partition = package_plan["package_by_partition"]
    partitions = [
        {
            **partition,
            "package_id": package_by_partition[partition["partition_id"]],
        }
        for partition in spatial_plan["partitions"]
    ]
    storage_value = dict(storage_report)
    if v33_enabled:
        retained_input_bytes = sum(
            (
                (int(partition["halo_window"]["x1"]) - int(partition["halo_window"]["x0"]))
                * (int(partition["halo_window"]["y1"]) - int(partition["halo_window"]["y0"]))
                * len(CLASS_ORDER)
                * 2
                + (int(partition["core_window"]["x1"]) - int(partition["core_window"]["x0"]))
                * (int(partition["core_window"]["y1"]) - int(partition["core_window"]["y0"]))
                * 4
                + 3 * 64 * 1024
            )
            for partition in partitions
        )
        storage_value["v33_retained_input_budget_bytes"] = retained_input_bytes
        storage_value["v33_retained_input_estimate"] = (
            "all_probability_halos_uint16_plus_v3_context_and_baseline_cores_int16"
        )
        if int(storage_value.get("storage_tuning_schema_version") or 0) >= 2:
            safe_headroom = int(storage_value.get("safe_headroom_bytes") or 0)
            working_budget = int(
                storage_value.get("working_cache_budget_bytes") or 0
            )
            if working_budget + retained_input_bytes > safe_headroom:
                raise RunBuilderV5Error(
                    "V3.3 retained inputs exceed frozen storage headroom"
                )
            storage_value["estimated_required_bytes"] = int(
                storage_value.get("estimated_required_bytes") or 0
            ) + retained_input_bytes

    class_snapshot = {
        "class_mapping": {str(code): CLASS_NAMES[code] for code in CLASS_ORDER},
        "index_to_code": {str(index): code for index, code in enumerate(CLASS_ORDER)},
        "background_index": -1,
    }
    class_path = run_dir / "class_mapping_snapshot.json"
    atomic_write_json(class_path, class_snapshot)
    model_values = [dict(model) for model in models]
    model_ids = [str(model["model_id"]) for model in model_values]
    if len(set(model_ids)) != len(model_ids):
        raise RunBuilderV5Error("semantic model IDs must be unique")
    fusion_value = dict(fusion) if fusion else None
    if v33_enabled and fusion_value is None:
        raise RunBuilderV5Error(
            "V3.3 production requires an approved Fusion stream"
        )
    if fusion_value:
        profile = fusion_value.get("profile")
        if not isinstance(profile, Mapping):
            profile_path = Path(str(fusion_value.get("profile_path") or ""))
            if not profile_path.is_file():
                raise RunBuilderV5Error("Fusion profile is missing")
            with open(profile_path, "r", encoding="utf-8") as handle:
                profile = json.load(handle)
        profile = dict(profile)
        if profile.get("status") != "approved" or (profile.get("approval") or {}).get("passed") is not True:
            raise RunBuilderV5Error("Fusion profile must be approved")
        snapshot_path = run_dir / "fusion_profile_snapshot.json"
        atomic_write_json(snapshot_path, profile)
        fusion_value.update(
            {
                "profile": profile,
                "snapshot_path": str(snapshot_path),
                "sha256": sha256_file(snapshot_path),
            }
        )

    config_snapshot = {
        "schema_version": 2,
        "runtime": {
            "effective_device": str(effective_device),
            "keep_score_cache": bool(keep_score_cache),
            "tile_batch_size": max(1, int(tile_batch_size)),
        },
        "resource_tuning": dict(resource_tuning or {}),
        "scaling": scaling_value,
        "models": model_values,
        "fusion": fusion_value,
        "boundary_fitting": boundary_value,
        "fragmentation_regularization": fragmentation_value,
        "coverage_validation": {
            "policy_id": "exact_range_zero_gap_v1",
            "area_tolerance_pixels": 0.01,
        },
        "range_selection": range_value,
        "config_fingerprint": str(config_fingerprint),
    }
    config_snapshot_path = run_dir / "config_snapshot.json"
    atomic_write_json(config_snapshot_path, config_snapshot)

    stream_values = [
        {
            "stream_id": f"model:{model['model_id']}",
            "kind": "model",
            "model_id": model["model_id"],
            "version": model.get("version", ""),
        }
        for model in model_values
    ]
    if fusion_value:
        stream_values.append(
            {
                "stream_id": f"fusion:{fusion_value['profile_id']}",
                "kind": "fusion",
                "profile_id": fusion_value["profile_id"],
                "version": fusion_value.get("version", ""),
            }
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
        sha256_file(accepted_path) if accepted_path is not None and accepted_path.is_file() else ""
    )

    for model in model_values:
        (run_dir / "models" / str(model["model_id"]) / "raster_parts").mkdir(
            parents=True, exist_ok=True
        )
    if fusion_value:
        (run_dir / "fusion" / str(fusion_value["profile_id"]) / "raster_parts").mkdir(
            parents=True, exist_ok=True
        )

    spec = {
        "schema_version": RUN_SPEC_SCHEMA_VERSION,
        "run_id": identifier,
        "created_at": _datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "output_root": str(output),
        "cache_root": str(run_tile_cache_dir(output, identifier).parent),
        "tile_cache_dir": str(run_tile_cache_dir(output, identifier)),
        "raster": {
            "path": str(Path(str(raster["path"])).expanduser().resolve()),
            "crs": str(raster["crs"]),
            "transform": [float(value) for value in raster["transform"]],
            "nodata": raster.get("nodata"),
        },
        "requested_extent": _extent(requested_extent),
        "processing_extent": _extent(processing_extent),
        "range_selection": range_value,
        "range_vector_path": str(
            range_value.get("vector_source")
            or range_value.get("vector_path")
            or ""
        ),
        "tile_grid": {
            "rows": int(tile_rows),
            "cols": int(tile_cols),
            "count": int(tile_rows) * int(tile_cols),
            "selected_count": selected_count,
            "excluded_count": excluded_count,
            "width": 512,
            "height": 512,
            "overlap": int(overlap),
            "stride": 512 - int(overlap),
        },
        "spatial_plan_summary": {
            "partition_rows": spatial_plan["partition_rows"],
            "partition_cols": spatial_plan["partition_cols"],
            "partition_count": spatial_plan["partition_count"],
            "unit_counts": spatial_plan["unit_counts"],
            "package_count": package_plan["package_count"],
        },
        "runtime": {
            "effective_device": str(effective_device),
            "keep_score_cache": bool(keep_score_cache),
            "tile_batch_size": max(1, int(tile_batch_size)),
        },
        "resource_tuning": dict(resource_tuning or {}),
        "scaling": scaling_value,
        "boundary_fitting": boundary_value,
        "fragmentation_regularization": fragmentation_value,
        "coverage_validation": {
            "policy_id": "exact_range_zero_gap_v1",
            "area_tolerance_pixels": 0.01,
        },
        "storage_preflight": storage_value,
        "models": model_values,
        "fusion": fusion_value,
        "streams": stream_values,
        "accepted_gpkg": str(accepted_path) if accepted_path is not None else "",
        "accepted_gpkg_sha256": accepted_sha256,
        "accepted_target_gpkg": (
            str(accepted_target_path) if accepted_target_path is not None else ""
        ),
        "accepted_validation": dict(accepted_validation or {}),
        "skip_accepted": bool(skip_accepted),
        "class_mapping_snapshot": str(class_path),
        "config_snapshot": str(config_snapshot_path),
        "config_fingerprint": str(config_fingerprint),
        "deployment_identity": deployment_identity(deployment_project_root),
        "state_backend": "postgresql",
        "state_db": state_location,
        "state_schema": state_schema,
    }
    spec["run_spec_content_sha256"] = _json_sha(spec)
    spec_path = run_dir / "run_spec.json"
    atomic_write_json(spec_path, spec)

    database_location = state_location
    database = RunStateDB(database_location, postgres_schema=state_schema)
    database.initialize()
    database.create_run(
        identifier,
        sha256_file(spec_path),
        status="planned",
        metadata={
            "run_spec": str(spec_path),
            "tile_count": spec["tile_grid"]["count"],
            "partition_count": spatial_plan["partition_count"],
            "package_count": package_plan["package_count"],
        },
    )
    database.register_streams(identifier, stream_values)
    database.insert_work_packages(identifier, package_plan["packages"])
    database.insert_partitions(identifier, partitions)
    database.insert_spatial_units(identifier, spatial_plan["spatial_units"])
    database.insert_stream_units(
        identifier,
        (stream["stream_id"] for stream in stream_values),
        (unit["unit_id"] for unit in spatial_plan["spatial_units"]),
    )

    partition_rows = int(spatial_plan["partition_rows"])
    partition_cols = int(spatial_plan["partition_cols"])
    part_tile_rows = int(spatial_plan["partition_tile_rows"])
    part_tile_cols = int(spatial_plan["partition_tile_cols"])

    def normalized_tiles():
        for item in tiles:
            row = int(item["row"])
            col = int(item["col"])
            if not (0 <= row < int(tile_rows) and 0 <= col < int(tile_cols)):
                raise RunBuilderV5Error(f"Tile is outside declared grid: {row}_{col}")
            partition_row = min(row // part_tile_rows, partition_rows - 1)
            partition_col = min(col // part_tile_cols, partition_cols - 1)
            status = str(item.get("status") or "ready")
            yield {
                **dict(item),
                "tile_id": str(item.get("tile_id") or f"{row}_{col}"),
                "row": row,
                "col": col,
                "width": int(item.get("width", 512)),
                "height": int(item.get("height", 512)),
                "partition_id": f"partition_{partition_row:05d}_{partition_col:05d}",
                "raster_path": (
                    str(
                        run_tile_cache_dir(output, identifier)
                        / f"tile_{row}_{col}.tif"
                    )
                    if status != "excluded"
                    else ""
                ),
                "sha256": "",
                "status": status,
            }

    inserted = database.insert_tiles(identifier, normalized_tiles())
    if inserted != int(tile_rows) * int(tile_cols):
        raise RunBuilderV5Error(
            f"Tile count mismatch: expected {int(tile_rows) * int(tile_cols)}, got {inserted}"
        )
    actual_excluded = database.count_tiles(identifier, status="excluded")
    if actual_excluded != excluded_count:
        raise RunBuilderV5Error(
            f"excluded Tile count mismatch: expected {excluded_count}, got {actual_excluded}"
        )
    database.insert_jobs(
        identifier,
        (
            {
                "job_type": "work_package",
                "package_id": package["package_id"],
                "priority": -int(package["sequence_no"]),
                "max_attempts": int(scaling_value["max_job_retries"]) + 1,
            }
            for package in package_plan["packages"]
        ),
    )
    if v33_enabled:
        v33_units = _fragmentation_v33_units(partitions)
        database.insert_spatial_units(identifier, v33_units)
        database.insert_jobs(
            identifier,
            (
                {
                    "job_type": "fragmentation_v33",
                    "stream_id": str(stream_values[-1]["stream_id"]),
                    "unit_id": str(unit["unit_id"]),
                    "priority": 50 if unit["unit_type"] == "FragmentationV33Finalize" else 60,
                    "max_attempts": int(scaling_value["max_job_retries"]) + 1,
                }
                for unit in v33_units
            ),
        )
    database.insert_jobs(
        identifier,
        (
            {
                "job_type": "unit_fit",
                "stream_id": stream["stream_id"],
                "unit_id": unit["unit_id"],
                "priority": 100,
                "max_attempts": int(scaling_value["max_job_retries"]) + 1,
            }
            for stream in stream_values
            for unit in spatial_plan["spatial_units"]
        ),
    )
    try:
        incomplete_run_cleanup = database.archive_incomplete_run_details(
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
        database.update_run_metadata(
            identifier,
            {"incomplete_run_cleanup": incomplete_run_cleanup},
        )
        archived_count = int(
            incomplete_run_cleanup.get("archived_run_count") or 0
        )
        skipped_active_count = int(
            incomplete_run_cleanup.get("skipped_active_run_count") or 0
        )
        cleanup_warning = (
            incomplete_run_cleanup.get("status") == "warning"
            or skipped_active_count > 0
        )
        if archived_count or cleanup_warning:
            database.append_event(
                identifier,
                "incomplete_run_cleanup",
                level="warning" if cleanup_warning else "info",
                message=(
                    str(incomplete_run_cleanup.get("error") or "")
                    if incomplete_run_cleanup.get("status") == "warning"
                    else (
                        "Skipped old incomplete Runs with active Jobs: "
                        + ", ".join(
                            incomplete_run_cleanup.get(
                                "skipped_active_run_ids"
                            )
                            or []
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
    return spec, spec_path, database_location
