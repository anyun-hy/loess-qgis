"""Budgeted Work Package planning and storage preflight for large local runs."""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any, Mapping


GIB = 1024**3


class WorkPackagePlanError(ValueError):
    pass


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
    permanent_bytes_per_tile_per_stream: int,
    input_tile_bytes_per_tile: int = 0,
    score_cache_budget_gb: float,
    min_free_disk_gb: float,
    current_model_probability_bytes: int,
    fusion_accumulator_bytes: int,
    mask_confidence_workspace_bytes: int,
    safety_margin_bytes: int,
    fixed_temporary_overhead_bytes: int = 0,
    available_disk_bytes: int | None = None,
) -> dict[str, Any]:
    """Use measured sample bytes to gate a run before any inference starts."""
    count = int(tile_count)
    streams = int(stream_count)
    permanent_per_tile = int(permanent_bytes_per_tile_per_stream)
    if count < 1 or streams < 1 or permanent_per_tile < 0:
        raise WorkPackagePlanError("Tile, stream, and permanent sample values are invalid")
    output = Path(output_path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    available = (
        int(available_disk_bytes)
        if available_disk_bytes is not None
        else int(shutil.disk_usage(output).free)
    )
    input_per_tile = int(input_tile_bytes_per_tile)
    if input_per_tile < 0:
        raise WorkPackagePlanError("input Tile bytes cannot be negative")
    permanent_outputs = count * streams * permanent_per_tile
    permanent = permanent_outputs
    budget = calculate_package_tile_limit(
        score_cache_budget_gb=score_cache_budget_gb,
        current_model_probability_bytes=current_model_probability_bytes,
        fusion_accumulator_bytes=fusion_accumulator_bytes,
        mask_confidence_workspace_bytes=mask_confidence_workspace_bytes,
        safety_margin_bytes=safety_margin_bytes,
        fixed_temporary_overhead_bytes=fixed_temporary_overhead_bytes,
        input_tile_bytes=input_per_tile,
        available_disk_bytes=available,
        min_free_disk_gb=min_free_disk_gb,
        permanent_estimated_bytes=permanent,
    )
    budget["package_tile_limit"] = min(count, budget["package_tile_limit"])
    peak_temp = (
        budget["package_tile_limit"] * budget["working_bytes_per_tile"]
        + budget["fixed_temporary_overhead_bytes"]
    )
    peak_input = budget["package_tile_limit"] * input_per_tile
    required = budget["min_free_disk_bytes"] + permanent + peak_temp
    return {
        "status": "passed",
        "measurement_required": True,
        "tile_count": count,
        "stream_count": streams,
        "permanent_bytes_per_tile_per_stream": permanent_per_tile,
        "input_tile_bytes_per_tile": input_per_tile,
        "input_tile_storage_mode": "work_package_temporary",
        "estimated_input_tile_bytes": peak_input,
        "estimated_permanent_output_bytes": permanent_outputs,
        "estimated_permanent_bytes": permanent,
        "estimated_peak_temporary_bytes": peak_temp,
        "estimated_required_bytes": required,
        **budget,
    }
