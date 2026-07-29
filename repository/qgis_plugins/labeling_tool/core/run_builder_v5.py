"""Create a v5 run whose detailed state lives in SQLite, not run_spec JSON."""

from __future__ import annotations

import datetime as _datetime
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .run_spec import (
    CLASS_NAMES,
    CLASS_ORDER,
    RESERVATION_FILE,
    atomic_write_json,
    reserve_run_directory,
    run_tile_cache_dir,
    sha256_file,
)
from .run_state_db import RunStateDB
from .spatial_planner import plan_spatial_units
from .work_package_planner import plan_work_packages


RUN_SPEC_SCHEMA_VERSION = 2


class RunBuilderV5Error(ValueError):
    pass


def _extent(value: Mapping[str, Any]) -> dict[str, float]:
    result = {key: float(value[key]) for key in ("xmin", "ymin", "xmax", "ymax")}
    if result["xmin"] >= result["xmax"] or result["ymin"] >= result["ymax"]:
        raise RunBuilderV5Error("run extent must have positive width and height")
    return result


def _json_sha(value: Mapping[str, Any]) -> str:
    import hashlib

    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    overlap: int,
    scaling: Mapping[str, Any],
    boundary_fitting: Mapping[str, Any],
    storage_report: Mapping[str, Any],
    fusion: Mapping[str, Any] | None = None,
    accepted_gpkg: str | Path = "",
    accepted_target_gpkg: str | Path = "",
    accepted_validation: Mapping[str, Any] | None = None,
    skip_accepted: bool = True,
    config_fingerprint: str = "",
    range_selection: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    reserved_run_dir: str | Path | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    """Freeze a small run spec and atomically populate the detailed state DB."""
    if not models:
        raise RunBuilderV5Error("at least one semantic model is required")
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
    boundary_value.setdefault("mode", "divider_cubic_bspline_v1")
    if str(boundary_value.get("mode") or "") != "divider_cubic_bspline_v1":
        raise RunBuilderV5Error(
            "boundary_fitting.mode must equal divider_cubic_bspline_v1"
        )
    range_value = dict(range_selection or {})
    range_mode = str(range_value.get("mode") or "extent")
    if range_mode not in {"extent", "vector_tile_intersection"}:
        raise RunBuilderV5Error(f"unsupported range selection mode: {range_mode}")
    if range_mode == "vector_tile_intersection" and range_value.get("clip_outputs") is not False:
        raise RunBuilderV5Error("vector Tile selection must not clip outputs")
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
        },
        "scaling": scaling_value,
        "models": model_values,
        "fusion": fusion_value,
        "boundary_fitting": boundary_value,
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
        },
        "scaling": scaling_value,
        "boundary_fitting": boundary_value,
        "storage_preflight": dict(storage_report),
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
        "state_db": str(run_dir / "run_state.sqlite"),
    }
    spec["run_spec_content_sha256"] = _json_sha(spec)
    spec_path = run_dir / "run_spec.json"
    atomic_write_json(spec_path, spec)

    database_path = run_dir / "run_state.sqlite"
    database = RunStateDB(database_path)
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
        (run_dir / RESERVATION_FILE).unlink()
    except FileNotFoundError:
        pass
    return spec, spec_path, database_path
