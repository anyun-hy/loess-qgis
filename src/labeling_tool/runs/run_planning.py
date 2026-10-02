"""Build Run parameters after detached inputs are prepared."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from labeling_tool.main.model_registry import ModelRegistry
from labeling_tool.runs.spatial_planner import plan_spatial_units
from labeling_tool.shared.contracts.run_spec import run_tile_cache_dir
from labeling_tool.shared.planning.work_package_planner import (
    fusion_accumulator_atomic_overhead,
    fusion_accumulator_bytes_per_tile,
    permanent_output_reserve,
    resolve_frozen_tile_batch_size,
    storage_preflight,
    unit_confidence_reserve,
)


def _extent_as_dict(extent: Any) -> dict[str, float]:
    return {
        "xmin": extent.xMinimum(),
        "ymin": extent.yMinimum(),
        "xmax": extent.xMaximum(),
        "ymax": extent.yMaximum(),
    }


def build_run_builder_kwargs(
    *,
    scripts_dir: str,
    output_root: str,
    accepted_target_gpkg: str,
    raster_layer: Any,
    requested_extent: Any,
    processing_extent: Any,
    grid_tiles: tuple[dict, ...],
    active_tiles: tuple[dict, ...],
    range_selection: dict,
    effective_config: dict,
    environment_report: dict,
    accepted_validation: dict,
    skip_accepted: bool,
    selected_model_ids: tuple[str, ...],
    fusion_profile_id: str | None,
    boundary_smoothing_enabled: bool,
    overlap: int,
    run_id: str,
    run_dir: str,
    accepted_snapshot: str,
    skipped_tiles: tuple[dict, ...],
    tile_cache_sample: dict,
) -> dict[str, Any]:
    """Build the exact arguments consumed by ``RunBuilderTask``."""

    effective = dict(effective_config)
    registry = ModelRegistry(effective)
    selected_ids = registry.resolve_selection(
        selected_model_ids,
        fusion_profile_id,
    )
    selected_models = [vars(registry.model(model_id)) for model_id in selected_ids]
    fusion = None
    if fusion_profile_id:
        registered_profile = registry.profile(fusion_profile_id)
        fusion = {
            "profile_id": registered_profile.profile_id,
            "version": str(registered_profile.profile.get("version") or ""),
            "profile_path": registered_profile.file_path,
            "profile": dict(registered_profile.profile),
        }

    scaling = dict(registry.scaling)
    fragmentation = dict(effective.get("fragmentation_regularization") or {})
    fragmentation_buffer = (
        int(fragmentation.get("buffer_pixels", 256))
        if bool(fragmentation.get("enabled", True))
        else 0
    )
    if str(scaling.get("partition_halo_px", "auto")).lower() == "auto":
        scaling["partition_halo_px"] = max(
            int(overlap),
            int(scaling.get("seam_band_px", 64)),
            fragmentation_buffer,
        )
    else:
        scaling["partition_halo_px"] = max(
            int(scaling["partition_halo_px"]),
            fragmentation_buffer,
        )

    pixel_count = 512 * 512
    sample_tile_bytes = int(tile_cache_sample.get("materialized_cache_bytes") or 0)
    if sample_tile_bytes <= 0:
        raise ValueError("真实 Tile 探针没有返回有效缓存字节数")
    stream_count = len(selected_models) + (1 if fusion else 0)
    tile_rows = max(int(tile["row"]) for tile in grid_tiles) + 1
    tile_cols = max(int(tile["col"]) for tile in grid_tiles) + 1
    spatial_plan = plan_spatial_units(
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        tile_size=512,
        overlap=int(overlap),
        partition_tile_rows=int(scaling["partition_tile_rows"]),
        partition_tile_cols=int(scaling["partition_tile_cols"]),
        seam_band_px=int(scaling["seam_band_px"]),
        halo_px=int(scaling["partition_halo_px"]),
    )
    permanent = permanent_output_reserve(
        spatial_plan,
        stream_count=stream_count,
    )
    v33_confidence = (
        unit_confidence_reserve(spatial_plan)
        if bool(
            fragmentation.get("enabled", True)
            and fragmentation.get("policy_id")
            == "fragmentation_v33_configurable_absorption_v1"
        )
        else {"reserve_bytes": 0}
    )
    resolved_resources = (effective.get("resource_tuning") or {}).get("resolved") or {}
    tile_batch_size = resolve_frozen_tile_batch_size(
        registry.runtime["tile_batch_size"],
        resolved_resources.get("tile_batch_size"),
    )
    selected_batch_sizes = [
        max(
            1,
            int(
                (resolved_resources.get("tile_batch_size_by_model") or {}).get(
                    model["model_id"], tile_batch_size
                )
            ),
        )
        for model in selected_models
    ]
    storage_batch_size = max(selected_batch_sizes, default=tile_batch_size)
    storage = storage_preflight(
        output_root,
        tile_count=len(active_tiles),
        stream_count=stream_count,
        permanent_raster_bytes=permanent["permanent_raster_bytes"],
        vector_output_reserve_bytes=permanent["vector_output_reserve_bytes"],
        permanent_core_pixel_count=permanent["core_pixel_count"],
        input_tile_bytes_per_tile=sample_tile_bytes,
        score_cache_budget_gb=scaling["score_cache_budget_gb"],
        min_free_disk_gb=float(scaling["min_free_disk_gb"]),
        current_model_probability_bytes=pixel_count * 14 * 2,
        fusion_accumulator_bytes=fusion_accumulator_bytes_per_tile(
            (fusion or {}).get("profile"),
            pixel_count=pixel_count,
        ),
        mask_confidence_workspace_bytes=pixel_count * (14 * 4 + 5),
        safety_margin_bytes=sample_tile_bytes,
        fixed_temporary_overhead_bytes=(pixel_count * 14 * 2 * storage_batch_size),
        fusion_atomic_write_overhead_bytes=fusion_accumulator_atomic_overhead(
            (fusion or {}).get("profile"),
            spatial_plan,
        ),
        deferred_temporary_reserve_bytes=int(v33_confidence["reserve_bytes"]),
        tile_batch_size=storage_batch_size,
    )
    storage["input_tile_sample"] = dict(tile_cache_sample)
    scaling["score_cache_budget_mode"] = storage["score_cache_budget_mode"]
    scaling["score_cache_budget_gb"] = storage["resolved_score_cache_budget_gb"]

    stride = 512 - int(overlap)
    accepted_tile_ids = {(int(tile["row"]), int(tile["col"])) for tile in skipped_tiles}
    selected_tile_keys = {(int(tile["row"]), int(tile["col"])) for tile in active_tiles}
    tile_cache_dir = run_tile_cache_dir(output_root, run_id)
    normalized_tiles = []
    for tile in grid_tiles:
        tile_key = (int(tile["row"]), int(tile["col"]))
        if tile_key not in selected_tile_keys:
            tile_path = ""
            tile_status = "excluded"
        else:
            tile_path = str(tile_cache_dir / f"tile_{tile_key[0]}_{tile_key[1]}.tif")
            tile_status = "accepted" if tile_key in accepted_tile_ids else "ready"
        normalized_tiles.append(
            {
                "row": tile_key[0],
                "col": tile_key[1],
                "path": tile_path,
                "sha256": "",
                "bounds": _extent_as_dict(tile["bounds"]),
                "pixel_window": {
                    "x0": tile_key[1] * stride,
                    "y0": tile_key[0] * stride,
                    "x1": tile_key[1] * stride + 512,
                    "y1": tile_key[0] * stride + 512,
                },
                "status": tile_status,
            }
        )

    res_x = abs(raster_layer.rasterUnitsPerPixelX())
    res_y = abs(raster_layer.rasterUnitsPerPixelY())
    return {
        "output_root": output_root,
        "reserved_run_dir": run_dir,
        "run_id": run_id,
        "raster": {
            "path": raster_layer.source().split("|", 1)[0],
            "crs": raster_layer.crs().authid(),
            "transform": [
                res_x,
                0.0,
                processing_extent.xMinimum(),
                0.0,
                -res_y,
                processing_extent.yMaximum(),
            ],
            "nodata": None,
        },
        "requested_extent": _extent_as_dict(requested_extent),
        "processing_extent": _extent_as_dict(processing_extent),
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
        "tiles": normalized_tiles,
        "models": selected_models,
        "effective_device": (effective.get("runtime") or {}).get(
            "effective_device", "cpu"
        ),
        "keep_score_cache": bool(
            (effective.get("runtime") or {}).get("keep_score_cache", False)
        ),
        "tile_batch_size": tile_batch_size,
        "resource_tuning": effective.get("resource_tuning") or {},
        "overlap": overlap,
        "scaling": scaling,
        "boundary_fitting": {
            **registry.boundary_fitting,
            "enabled": bool(boundary_smoothing_enabled),
        },
        "fragmentation_regularization": fragmentation,
        "storage_report": storage,
        "fusion": fusion,
        "accepted_gpkg": accepted_snapshot,
        "accepted_target_gpkg": accepted_target_gpkg,
        "accepted_validation": dict(accepted_validation),
        "skip_accepted": bool(skip_accepted),
        "config_fingerprint": environment_report.get("config_fingerprint", ""),
        "range_selection": dict(range_selection),
        "deployment_project_root": Path(scripts_dir).parent,
    }
