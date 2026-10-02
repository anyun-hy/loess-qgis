"""Persist the planned PostgreSQL control graph for one frozen Run."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from labeling_tool.runs.run_build_contract import RunBuilderV5Error
from labeling_tool.shared.contracts.run_spec import sha256_file


class RunControlGraphJobs(Protocol):
    """The Job write operation required during Run creation."""

    def insert_jobs(self, run_id: str, jobs: Iterable[Mapping[str, Any]]) -> int: ...


class RunControlGraphWriter(Protocol):
    """The spatial control-graph writes required during Run creation."""

    def insert_work_packages(
        self, run_id: str, packages: Iterable[Mapping[str, Any]]
    ) -> int: ...

    def insert_partitions(
        self, run_id: str, partitions: Iterable[Mapping[str, Any]]
    ) -> int: ...

    def insert_spatial_units(
        self, run_id: str, units: Iterable[Mapping[str, Any]]
    ) -> int: ...

    def insert_stream_units(
        self,
        run_id: str,
        stream_ids: Iterable[str],
        unit_ids: Iterable[str],
    ) -> int: ...

    def insert_tiles(self, run_id: str, tiles: Iterable[Mapping[str, Any]]) -> int: ...

    def count_tiles(self, run_id: str, *, status: str) -> int: ...


class RunStreamWriter(Protocol):
    """The Run and Stream writes required during Run creation."""

    def create_run(
        self,
        run_id: str,
        run_spec_sha256: str,
        *,
        status: str,
        metadata: Mapping[str, Any],
    ) -> None: ...

    def register_streams(
        self, run_id: str, streams: Sequence[Mapping[str, Any]]
    ) -> None: ...


class RunControlGraphStore(Protocol):
    """The persistent graph operations required during Run creation."""

    @property
    def jobs(self) -> RunControlGraphJobs: ...

    @property
    def control_graph(self) -> RunControlGraphWriter: ...

    @property
    def run_streams(self) -> RunStreamWriter: ...


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


def write_v5_control_graph(
    *,
    database: RunControlGraphStore,
    run_id: str,
    spec_path: Path,
    tile_rows: int,
    tile_cols: int,
    tile_cache_dir: Path,
    excluded_tile_count: int,
    spatial_plan: Mapping[str, Any],
    package_plan: Mapping[str, Any],
    partitions: Sequence[Mapping[str, Any]],
    streams: Sequence[Mapping[str, Any]],
    tiles: Iterable[Mapping[str, Any]],
    max_job_retries: int,
    v33_enabled: bool,
    checkpoint: Callable[[float, str], None],
) -> None:
    """Write one Run graph in the existing operation and checkpoint order.

    Each store call keeps its own transaction.  ``checkpoint`` is supplied by
    the builder so cancellation remains observable at the established points.
    """

    database.run_streams.create_run(
        run_id,
        sha256_file(spec_path),
        status="planned",
        metadata={
            "run_spec": str(spec_path),
            "tile_count": int(tile_rows) * int(tile_cols),
            "partition_count": spatial_plan["partition_count"],
            "package_count": package_plan["package_count"],
        },
    )
    checkpoint(16, "PostgreSQL Run 已创建")
    database.run_streams.register_streams(run_id, streams)
    database.control_graph.insert_work_packages(run_id, package_plan["packages"])
    database.control_graph.insert_partitions(run_id, partitions)
    checkpoint(22, "Partition 与 Work Package 已写入")
    database.control_graph.insert_spatial_units(run_id, spatial_plan["spatial_units"])
    checkpoint(32, "空间单元已写入")
    database.control_graph.insert_stream_units(
        run_id,
        (str(stream["stream_id"]) for stream in streams),
        (str(unit["unit_id"]) for unit in spatial_plan["spatial_units"]),
    )
    checkpoint(45, "结果流空间单元已写入")

    partition_rows = int(spatial_plan["partition_rows"])
    partition_cols = int(spatial_plan["partition_cols"])
    partition_tile_rows = int(spatial_plan["partition_tile_rows"])
    partition_tile_cols = int(spatial_plan["partition_tile_cols"])

    def normalized_tiles() -> Iterable[Mapping[str, Any]]:
        for item in tiles:
            row = int(item["row"])
            col = int(item["col"])
            if not (0 <= row < int(tile_rows) and 0 <= col < int(tile_cols)):
                raise RunBuilderV5Error(f"Tile is outside declared grid: {row}_{col}")
            partition_row = min(row // partition_tile_rows, partition_rows - 1)
            partition_col = min(col // partition_tile_cols, partition_cols - 1)
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
                    str(tile_cache_dir / f"tile_{row}_{col}.tif")
                    if status != "excluded"
                    else ""
                ),
                "sha256": "",
                "status": status,
            }

    inserted = database.control_graph.insert_tiles(run_id, normalized_tiles())
    checkpoint(68, "Tile 索引已写入")
    expected_tile_count = int(tile_rows) * int(tile_cols)
    if inserted != expected_tile_count:
        raise RunBuilderV5Error(
            f"Tile count mismatch: expected {expected_tile_count}, got {inserted}"
        )
    actual_excluded = database.control_graph.count_tiles(run_id, status="excluded")
    if actual_excluded != excluded_tile_count:
        raise RunBuilderV5Error(
            "excluded Tile count mismatch: expected "
            f"{excluded_tile_count}, got {actual_excluded}"
        )
    database.jobs.insert_jobs(
        run_id,
        (
            {
                "job_type": "work_package",
                "package_id": package["package_id"],
                "priority": -int(package["sequence_no"]),
                "max_attempts": int(max_job_retries) + 1,
            }
            for package in package_plan["packages"]
        ),
    )
    checkpoint(70, "Work Package Job 已写入")
    if v33_enabled:
        v33_units = _fragmentation_v33_units(partitions)
        database.control_graph.insert_spatial_units(run_id, v33_units)
        database.jobs.insert_jobs(
            run_id,
            (
                {
                    "job_type": "fragmentation_v33",
                    "stream_id": str(streams[-1]["stream_id"]),
                    "unit_id": str(unit["unit_id"]),
                    "priority": (
                        50 if unit["unit_type"] == "FragmentationV33Finalize" else 60
                    ),
                    "max_attempts": int(max_job_retries) + 1,
                }
                for unit in v33_units
            ),
        )
        checkpoint(78, "V3.3 Job 已写入")
        database.jobs.insert_jobs(
            run_id,
            (
                {
                    "job_type": "unit_confidence",
                    "stream_id": str(streams[-1]["stream_id"]),
                    "unit_id": str(unit["unit_id"]),
                    # Confidence compaction releases 14-band probability
                    # halos, so it outranks geometry work while Packages run.
                    "priority": 120,
                    "max_attempts": int(max_job_retries) + 1,
                }
                for unit in spatial_plan["spatial_units"]
            ),
        )
        checkpoint(86, "置信度 Job 已写入")
    database.jobs.insert_jobs(
        run_id,
        (
            {
                "job_type": "unit_fit",
                "stream_id": stream["stream_id"],
                "unit_id": unit["unit_id"],
                "priority": 100,
                "max_attempts": int(max_job_retries) + 1,
            }
            for stream in streams
            for unit in spatial_plan["spatial_units"]
        ),
    )
    checkpoint(97, "空间拟合 Job 已写入")
