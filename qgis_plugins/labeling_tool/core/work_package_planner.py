"""Budgeted Work Package planning and storage preflight for large local runs."""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any, Mapping


GIB = 1024**3
MIB = 1024**2
PERMANENT_MASK_BYTES_PER_PIXEL = 2
PERMANENT_CONFIDENCE_BYTES_PER_PIXEL = 4
PERMANENT_RASTER_BYTES_PER_PIXEL_PER_STREAM = (
    PERMANENT_MASK_BYTES_PER_PIXEL + PERMANENT_CONFIDENCE_BYTES_PER_PIXEL
)
VECTOR_OUTPUT_BYTES_PER_CORE_PIXEL_PER_STREAM = 1
MIN_VECTOR_OUTPUT_RESERVE_BYTES = 64 * GIB
PERMANENT_UNCERTAINTY_RATIO = 0.25
FILESYSTEM_FLOOR_RATIO = 0.05
AUTO_HEADROOM_FRACTION = 0.50
AUTO_VOLUME_FRACTION_CAP = 0.20
AUTO_ABSOLUTE_CAP_BYTES = 512 * GIB
CHECKPOINT_METADATA_OVERHEAD_BYTES = MIB
STORAGE_TUNING_SCHEMA_VERSION = 2
STORAGE_TUNING_FORMULA_VERSION = "disk-aware-cache-v3-exact-core"


class WorkPackagePlanError(ValueError):
    pass


def resolve_frozen_tile_batch_size(
    configured_value: Any,
    *resolved_candidates: Any,
) -> int:
    """Return the positive integer Batch frozen for this Run.

    A deployment value may still be ``auto``.  In that case callers must pass
    a hardware- or source-Run-resolved candidate; silently coercing ``auto``
    or falling back to one would change the performance contract.
    """
    configured = str(configured_value).strip().lower()
    candidates = (
        resolved_candidates if configured == "auto" else (configured_value,)
    )
    for raw_value in candidates:
        if raw_value is None or isinstance(raw_value, bool):
            continue
        text_value = str(raw_value).strip()
        if text_value.lower() == "auto":
            continue
        unsigned = text_value[1:] if text_value.startswith("+") else text_value
        if not unsigned.isdigit():
            continue
        try:
            value = int(text_value)
        except (TypeError, ValueError):
            continue
        if value >= 1:
            return value
    raise WorkPackagePlanError(
        "tile_batch_size=auto must be resolved to a positive integer before Run creation"
    )


def permanent_output_reserve(
    spatial_plan: Mapping[str, Any],
    *,
    stream_count: int,
) -> dict[str, int]:
    """Calculate permanent bytes from every exact Partition Core window."""
    streams = int(stream_count)
    partitions = list(spatial_plan.get("partitions") or [])
    if streams < 1 or not partitions:
        raise WorkPackagePlanError(
            "permanent output planning requires streams and spatial Partitions"
        )
    core_pixels = 0
    for partition in partitions:
        core = partition.get("core_window") or {}
        try:
            width = int(core["x1"]) - int(core["x0"])
            height = int(core["y1"]) - int(core["y0"])
        except (KeyError, TypeError, ValueError) as error:
            raise WorkPackagePlanError(
                "Partition Core window is incomplete or invalid"
            ) from error
        if width < 1 or height < 1:
            raise WorkPackagePlanError("Partition Core window must have positive area")
        core_pixels += width * height
    raster_bytes = (
        core_pixels
        * streams
        * PERMANENT_RASTER_BYTES_PER_PIXEL_PER_STREAM
    )
    vector_scaled_bytes = (
        core_pixels
        * streams
        * VECTOR_OUTPUT_BYTES_PER_CORE_PIXEL_PER_STREAM
    )
    vector_reserve_bytes = max(
        MIN_VECTOR_OUTPUT_RESERVE_BYTES,
        vector_scaled_bytes,
    )
    return {
        "partition_count": len(partitions),
        "core_pixel_count": core_pixels,
        "stream_count": streams,
        "mask_bytes_per_pixel": PERMANENT_MASK_BYTES_PER_PIXEL,
        "confidence_bytes_per_pixel": PERMANENT_CONFIDENCE_BYTES_PER_PIXEL,
        "raster_bytes_per_pixel_per_stream": (
            PERMANENT_RASTER_BYTES_PER_PIXEL_PER_STREAM
        ),
        "permanent_raster_bytes": raster_bytes,
        "vector_bytes_per_core_pixel_per_stream": (
            VECTOR_OUTPUT_BYTES_PER_CORE_PIXEL_PER_STREAM
        ),
        "vector_minimum_reserve_bytes": MIN_VECTOR_OUTPUT_RESERVE_BYTES,
        "vector_scaled_reserve_bytes": vector_scaled_bytes,
        "vector_output_reserve_bytes": vector_reserve_bytes,
    }


def fusion_accumulator_bytes_per_tile(
    profile: Mapping[str, Any] | None,
    *,
    pixel_count: int,
) -> int:
    """Return the on-disk accumulator peak for one Tile-sized area.

    Ordinary strategies keep one 14-channel float32 accumulator.  The
    supported ``linear_1x1`` contract keeps every model-major calibrated
    channel (5 * 14 today).  One additional float32 channel conservatively
    covers the coverage/atomic metadata workspace used alongside either form.
    """

    if not profile:
        return 0
    pixels = int(pixel_count)
    if pixels < 1:
        raise WorkPackagePlanError("fusion accumulator pixel count must be positive")
    strategy = str(profile.get("strategy") or "")
    if strategy == "linear_1x1":
        model_count = len(list(profile.get("models") or []))
        if model_count < 1:
            raise WorkPackagePlanError(
                "linear_1x1 fusion accumulator requires profile models"
            )
        channels = model_count * 14
    else:
        channels = 14
    return pixels * (channels + 1) * 4


def _format_gib(value: int) -> str:
    return f"{int(value) / GIB:.2f} GiB"


def calculate_package_tile_limit(
    *,
    score_cache_budget_gb: float,
    current_model_probability_bytes: int,
    fusion_accumulator_bytes: int,
    mask_confidence_workspace_bytes: int,
    safety_margin_bytes: int,
    fixed_temporary_overhead_bytes: int = 0,
    input_tile_bytes: int = 0,
    available_disk_bytes: int,
    min_free_disk_gb: float,
    permanent_estimated_bytes: int,
) -> dict[str, int]:
    components = {
        "input_tile_bytes": int(input_tile_bytes),
        "current_model_probability_bytes": int(current_model_probability_bytes),
        "fusion_accumulator_bytes": int(fusion_accumulator_bytes),
        "mask_confidence_workspace_bytes": int(mask_confidence_workspace_bytes),
        "safety_margin_bytes": int(safety_margin_bytes),
    }
    if any(value < 0 for value in components.values()):
        raise WorkPackagePlanError("per-Tile measured byte values cannot be negative")
    fixed_overhead = int(fixed_temporary_overhead_bytes)
    if fixed_overhead < 0:
        raise WorkPackagePlanError("fixed temporary overhead cannot be negative")
    per_tile = sum(components.values())
    if per_tile <= 0:
        raise WorkPackagePlanError("measured working bytes per Tile must be positive")
    cache_budget = int(float(score_cache_budget_gb) * GIB)
    reserve = int(float(min_free_disk_gb) * GIB)
    if cache_budget <= 0 or reserve < 0:
        raise WorkPackagePlanError("cache budget must be positive and disk reserve non-negative")
    usable_disk = (
        int(available_disk_bytes)
        - reserve
        - int(permanent_estimated_bytes)
        - fixed_overhead
    )
    usable_cache = cache_budget - fixed_overhead
    cache_limit = max(0, usable_cache // per_tile)
    disk_limit = max(0, usable_disk // per_tile)
    effective = min(cache_limit, disk_limit)
    if effective < 1:
        minimum_required = (
            reserve + int(permanent_estimated_bytes) + fixed_overhead + per_tile
        )
        shortfall = max(0, minimum_required - int(available_disk_bytes))
        raise WorkPackagePlanError(
            "磁盘空间预检失败："
            f"可用 {_format_gib(available_disk_bytes)}，"
            f"永久结果预计 {_format_gib(permanent_estimated_bytes)}，"
            f"安全预留 {_format_gib(reserve)}，"
            f"单 Tile 工作集 {_format_gib(per_tile)}，"
            f"最低需要 {_format_gib(minimum_required)}，"
            f"还缺 {_format_gib(shortfall)}。"
        )
    return {
        **components,
        "working_bytes_per_tile": per_tile,
        "cache_budget_bytes": cache_budget,
        "fixed_temporary_overhead_bytes": fixed_overhead,
        "available_disk_bytes": int(available_disk_bytes),
        "min_free_disk_bytes": reserve,
        "permanent_estimated_bytes": int(permanent_estimated_bytes),
        "usable_working_disk_bytes": usable_disk,
        "cache_tile_limit": cache_limit,
        "disk_tile_limit": disk_limit,
        "package_tile_limit": effective,
    }


def _expanded_tile_window(
    partition: Mapping[str, Any],
    *,
    tile_rows: int,
    tile_cols: int,
    halo_tiles: int,
) -> tuple[int, int, int, int]:
    return (
        max(0, int(partition["tile_row_start"]) - halo_tiles),
        min(tile_rows, int(partition["tile_row_stop"]) + halo_tiles),
        max(0, int(partition["tile_col_start"]) - halo_tiles),
        min(tile_cols, int(partition["tile_col_stop"]) + halo_tiles),
    )


def _tile_ids(window: tuple[int, int, int, int], tile_cols: int):
    row_start, row_stop, col_start, col_stop = window
    for row in range(row_start, row_stop):
        base = row * tile_cols
        for col in range(col_start, col_stop):
            yield base + col


def plan_work_packages(
    spatial_plan: Mapping[str, Any],
    *,
    package_tile_limit: int,
    estimated_bytes_per_tile: int,
) -> dict[str, Any]:
    """Group adjacent partitions without materializing Tile records in the output."""
    tile_limit = int(package_tile_limit)
    per_tile = int(estimated_bytes_per_tile)
    if tile_limit < 1 or per_tile < 1:
        raise WorkPackagePlanError("package Tile limit and estimated bytes must be positive")
    tile_rows = int(spatial_plan["tile_rows"])
    tile_cols = int(spatial_plan["tile_cols"])
    stride = int(spatial_plan["stride"])
    halo_px = int(spatial_plan["halo_px"])
    halo_tiles = int(math.ceil(halo_px / stride))
    partitions = list(spatial_plan.get("partitions") or [])
    by_position = {
        (int(item["row"]), int(item["col"])): item for item in partitions
    }
    if len(by_position) != len(partitions) or not partitions:
        raise WorkPackagePlanError("spatial plan partitions are empty or duplicated")

    ordered: list[Mapping[str, Any]] = []
    for row in range(int(spatial_plan["partition_rows"])):
        columns = range(int(spatial_plan["partition_cols"]))
        if row % 2:
            columns = reversed(list(columns))
        for col in columns:
            ordered.append(by_position[(row, col)])

    packages: list[dict[str, Any]] = []
    current_partitions: list[Mapping[str, Any]] = []
    current_tiles: set[int] = set()
    current_windows: list[tuple[int, int, int, int]] = []

    def submit() -> None:
        if not current_partitions:
            return
        sequence = len(packages)
        packages.append(
            {
                "package_id": f"package_{sequence:05d}",
                "sequence_no": sequence,
                "partition_ids": [str(item["partition_id"]) for item in current_partitions],
                "tile_count": len(current_tiles),
                "estimated_bytes": len(current_tiles) * per_tile,
                "tile_windows": [list(window) for window in current_windows],
                "neighbor_package_ids": [],
            }
        )

    for partition in ordered:
        window = _expanded_tile_window(
            partition,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            halo_tiles=halo_tiles,
        )
        partition_tiles = set(_tile_ids(window, tile_cols))
        if len(partition_tiles) > tile_limit:
            raise WorkPackagePlanError(
                f"partition {partition['partition_id']} needs {len(partition_tiles)} Tiles, "
                f"exceeding package limit {tile_limit}"
            )
        candidate = current_tiles | partition_tiles
        if current_partitions and len(candidate) > tile_limit:
            submit()
            current_partitions = []
            current_tiles = set()
            current_windows = []
        current_partitions.append(partition)
        current_tiles.update(partition_tiles)
        current_windows.append(window)
    submit()

    package_by_partition: dict[str, str] = {}
    package_by_id = {item["package_id"]: item for item in packages}
    for package in packages:
        for partition_id in package["partition_ids"]:
            package_by_partition[partition_id] = package["package_id"]
    for (row, col), partition in by_position.items():
        package_id = package_by_partition[str(partition["partition_id"])]
        for neighbor_position in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            neighbor = by_position.get(neighbor_position)
            if neighbor is None:
                continue
            neighbor_package = package_by_partition[str(neighbor["partition_id"])]
            if neighbor_package != package_id:
                package_by_id[package_id]["neighbor_package_ids"].append(neighbor_package)
    for package in packages:
        package["neighbor_package_ids"] = sorted(set(package["neighbor_package_ids"]))

    return {
        "schema_version": 1,
        "package_tile_limit": tile_limit,
        "estimated_bytes_per_tile": per_tile,
        "halo_tiles": halo_tiles,
        "package_count": len(packages),
        "peak_package_tiles": max(item["tile_count"] for item in packages),
        "peak_package_bytes": max(item["estimated_bytes"] for item in packages),
        "package_by_partition": package_by_partition,
        "packages": packages,
    }


def storage_preflight(
    output_path: str | Path,
    *,
    tile_count: int,
    stream_count: int,
    permanent_raster_bytes: int,
    vector_output_reserve_bytes: int,
    permanent_core_pixel_count: int = 0,
    input_tile_bytes_per_tile: int = 0,
    score_cache_budget_gb: float | str,
    min_free_disk_gb: float,
    current_model_probability_bytes: int,
    fusion_accumulator_bytes: int,
    mask_confidence_workspace_bytes: int,
    safety_margin_bytes: int,
    fixed_temporary_overhead_bytes: int = 0,
    available_disk_bytes: int | None = None,
    total_disk_bytes: int | None = None,
    tile_batch_size: int = 1,
) -> dict[str, Any]:
    """Use measured sample bytes to gate a run before any inference starts."""
    count = int(tile_count)
    streams = int(stream_count)
    raster_permanent = int(permanent_raster_bytes)
    vector_reserve = int(vector_output_reserve_bytes)
    core_pixel_count = int(permanent_core_pixel_count)
    if (
        count < 1
        or streams < 1
        or raster_permanent < 0
        or vector_reserve < 0
        or core_pixel_count < 1
    ):
        raise WorkPackagePlanError("Tile, stream, and permanent byte values are invalid")
    expected_raster_permanent = (
        core_pixel_count
        * streams
        * PERMANENT_RASTER_BYTES_PER_PIXEL_PER_STREAM
    )
    expected_vector_reserve = max(
        MIN_VECTOR_OUTPUT_RESERVE_BYTES,
        core_pixel_count
        * streams
        * VECTOR_OUTPUT_BYTES_PER_CORE_PIXEL_PER_STREAM,
    )
    if raster_permanent != expected_raster_permanent:
        raise WorkPackagePlanError(
            "permanent raster bytes do not match exact Core pixel formula"
        )
    if vector_reserve != expected_vector_reserve:
        raise WorkPackagePlanError(
            "vector output reserve does not match the frozen conservative formula"
        )
    output = Path(output_path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output)
    available = int(available_disk_bytes) if available_disk_bytes is not None else int(usage.free)
    filesystem_total = int(total_disk_bytes) if total_disk_bytes is not None else int(usage.total)
    if filesystem_total <= 0 or available <= 0 or available > filesystem_total:
        raise WorkPackagePlanError("filesystem total/free byte values are invalid")
    input_per_tile = int(input_tile_bytes_per_tile)
    if input_per_tile < 0:
        raise WorkPackagePlanError("input Tile bytes cannot be negative")
    permanent_base = raster_permanent + vector_reserve
    permanent_estimate_uncertainty = int(
        math.ceil(permanent_base * PERMANENT_UNCERTAINTY_RATIO)
    )
    # The runtime treats ``permanent_uncertainty_bytes`` as non-decaying.
    # Include the vector reserve there so completed Core rasters never release
    # space that later GeoPackage assembly still needs.
    nondecaying_permanent_reserve = (
        vector_reserve + permanent_estimate_uncertainty
    )
    protected_permanent = raster_permanent + nondecaying_permanent_reserve
    raw_cache_budget = str(score_cache_budget_gb).strip().lower()
    cache_budget_mode = "auto" if raw_cache_budget == "auto" else "explicit"
    configured_reserve = int(float(min_free_disk_gb) * GIB)
    filesystem_floor = int(math.ceil(filesystem_total * FILESYSTEM_FLOOR_RATIO))
    effective_reserve = max(configured_reserve, filesystem_floor)
    atomic_checkpoint_overhead = (
        int(fixed_temporary_overhead_bytes)
        + CHECKPOINT_METADATA_OVERHEAD_BYTES
    )
    if configured_reserve < 0 or atomic_checkpoint_overhead < CHECKPOINT_METADATA_OVERHEAD_BYTES:
        raise WorkPackagePlanError("disk reserve and atomic overhead must be non-negative")
    components = {
        "input_tile_bytes": input_per_tile,
        "current_model_probability_bytes": int(current_model_probability_bytes),
        "fusion_accumulator_bytes": int(fusion_accumulator_bytes),
        "mask_confidence_workspace_bytes": int(mask_confidence_workspace_bytes),
        "safety_margin_bytes": int(safety_margin_bytes),
    }
    if any(value < 0 for value in components.values()):
        raise WorkPackagePlanError("per-Tile measured byte values cannot be negative")
    working_per_tile = sum(components.values())
    if working_per_tile <= 0:
        raise WorkPackagePlanError("measured working bytes per Tile must be positive")
    safe_headroom = (
        available
        - protected_permanent
        - effective_reserve
        - atomic_checkpoint_overhead
    )
    batch_size = resolve_frozen_tile_batch_size(tile_batch_size)
    if cache_budget_mode == "auto":
        if safe_headroom < working_per_tile:
            raise WorkPackagePlanError(
                "磁盘空间预检失败：自动缓存预算在永久结果、安全余量和保留空间后不足。"
            )
        auto_working_budget = min(
            int(math.floor(safe_headroom * AUTO_HEADROOM_FRACTION)),
            int(math.floor(filesystem_total * AUTO_VOLUME_FRACTION_CAP)),
            AUTO_ABSOLUTE_CAP_BYTES,
            count * working_per_tile,
        )
        raw_tile_limit = min(count, auto_working_budget // working_per_tile)
        if raw_tile_limit >= batch_size and raw_tile_limit < count:
            raw_tile_limit = (raw_tile_limit // batch_size) * batch_size
        if raw_tile_limit < 1:
            raise WorkPackagePlanError(
                "磁盘空间预检失败：自动缓存预算不足一个 Tile。"
            )
        working_cache_budget_bytes = raw_tile_limit * working_per_tile
        resolved_cache_budget_bytes = (
            atomic_checkpoint_overhead + working_cache_budget_bytes
        )
        resolved_cache_budget_gb = resolved_cache_budget_bytes / GIB
    else:
        try:
            resolved_cache_budget_gb = float(score_cache_budget_gb)
        except (TypeError, ValueError) as error:
            raise WorkPackagePlanError(
                "score cache budget must be auto or a positive number"
            ) from error
        if not math.isfinite(resolved_cache_budget_gb) or resolved_cache_budget_gb <= 0:
            raise WorkPackagePlanError(
                "score cache budget must be auto or a positive number"
            )
        resolved_cache_budget_bytes = int(resolved_cache_budget_gb * GIB)
        working_cache_budget_bytes = max(
            0, resolved_cache_budget_bytes - atomic_checkpoint_overhead
        )
    budget = calculate_package_tile_limit(
        score_cache_budget_gb=resolved_cache_budget_gb,
        current_model_probability_bytes=current_model_probability_bytes,
        fusion_accumulator_bytes=fusion_accumulator_bytes,
        mask_confidence_workspace_bytes=mask_confidence_workspace_bytes,
        safety_margin_bytes=safety_margin_bytes,
        fixed_temporary_overhead_bytes=atomic_checkpoint_overhead,
        input_tile_bytes=input_per_tile,
        available_disk_bytes=available,
        min_free_disk_gb=effective_reserve / GIB,
        permanent_estimated_bytes=protected_permanent,
    )
    package_tile_limit = min(count, budget["package_tile_limit"])
    if package_tile_limit >= batch_size and package_tile_limit < count:
        package_tile_limit = (package_tile_limit // batch_size) * batch_size
    if package_tile_limit < 1:
        raise WorkPackagePlanError(
            "磁盘空间预检失败：批次对齐后不足一个 Tile。"
        )
    budget["package_tile_limit"] = package_tile_limit
    working_cache_budget_bytes = package_tile_limit * budget["working_bytes_per_tile"]
    if cache_budget_mode == "auto":
        resolved_cache_budget_bytes = (
            atomic_checkpoint_overhead + working_cache_budget_bytes
        )
        resolved_cache_budget_gb = resolved_cache_budget_bytes / GIB
    peak_temp = (
        budget["package_tile_limit"] * budget["working_bytes_per_tile"]
        + budget["fixed_temporary_overhead_bytes"]
    )
    peak_input = budget["package_tile_limit"] * input_per_tile
    required = budget["min_free_disk_bytes"] + protected_permanent + peak_temp
    return {
        "storage_tuning_schema_version": STORAGE_TUNING_SCHEMA_VERSION,
        "formula_version": STORAGE_TUNING_FORMULA_VERSION,
        "status": "passed",
        "measurement_required": True,
        "tile_count": count,
        "stream_count": streams,
        "permanent_core_pixel_count": core_pixel_count,
        "permanent_mask_bytes_per_pixel": PERMANENT_MASK_BYTES_PER_PIXEL,
        "permanent_confidence_bytes_per_pixel": (
            PERMANENT_CONFIDENCE_BYTES_PER_PIXEL
        ),
        "permanent_raster_bytes_per_pixel_per_stream": (
            PERMANENT_RASTER_BYTES_PER_PIXEL_PER_STREAM
        ),
        "vector_output_bytes_per_core_pixel_per_stream": (
            VECTOR_OUTPUT_BYTES_PER_CORE_PIXEL_PER_STREAM
        ),
        "vector_output_minimum_reserve_bytes": MIN_VECTOR_OUTPUT_RESERVE_BYTES,
        "vector_output_scaled_reserve_bytes": (
            core_pixel_count
            * streams
            * VECTOR_OUTPUT_BYTES_PER_CORE_PIXEL_PER_STREAM
        ),
        "input_tile_bytes_per_tile": input_per_tile,
        "input_tile_storage_mode": "work_package_temporary",
        "estimated_input_tile_bytes": peak_input,
        "estimated_permanent_output_bytes": permanent_base,
        "estimated_permanent_raster_bytes": raster_permanent,
        # Raster-only compatibility field for proportional Core completion.
        "estimated_permanent_bytes": raster_permanent,
        "vector_output_reserve_bytes": vector_reserve,
        "permanent_base_bytes": permanent_base,
        "permanent_uncertainty_ratio": PERMANENT_UNCERTAINTY_RATIO,
        "permanent_estimate_uncertainty_bytes": permanent_estimate_uncertainty,
        "nondecaying_permanent_reserve_bytes": nondecaying_permanent_reserve,
        # Aggregate consumed by the current runtime and intentionally retained.
        "permanent_uncertainty_bytes": nondecaying_permanent_reserve,
        "protected_permanent_estimated_bytes": protected_permanent,
        "score_cache_budget_mode": cache_budget_mode,
        "configured_score_cache_budget_gb": score_cache_budget_gb,
        "resolved_score_cache_budget_gb": resolved_cache_budget_gb,
        "resolved_score_cache_budget_bytes": resolved_cache_budget_bytes,
        "working_cache_budget_bytes": working_cache_budget_bytes,
        "filesystem_total_bytes": filesystem_total,
        "configured_min_free_disk_bytes": configured_reserve,
        "filesystem_floor_bytes": filesystem_floor,
        "effective_min_free_disk_bytes": effective_reserve,
        "atomic_checkpoint_overhead_bytes": atomic_checkpoint_overhead,
        "checkpoint_metadata_overhead_bytes": CHECKPOINT_METADATA_OVERHEAD_BYTES,
        "safe_headroom_bytes": safe_headroom,
        "auto_headroom_fraction": AUTO_HEADROOM_FRACTION,
        "auto_volume_fraction_cap": AUTO_VOLUME_FRACTION_CAP,
        "auto_absolute_cap_bytes": AUTO_ABSOLUTE_CAP_BYTES,
        "tile_batch_size": batch_size,
        "batch_aligned_package_tile_limit": package_tile_limit,
        "estimated_peak_temporary_bytes": peak_temp,
        "estimated_required_bytes": required,
        **budget,
    }
