from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from labeling_tool.runs.run_build_contract import FrozenRunPlan, RunBuilderV5Error
from labeling_tool.runs.run_build_preparation import prepare_v5_run_plan
from labeling_tool.runs.run_build_snapshots import freeze_v5_run_snapshots
from labeling_tool.runs.run_control_graph import write_v5_control_graph


def _plan(*, v33_enabled: bool = False) -> FrozenRunPlan:
    return FrozenRunPlan(
        scaling={
            "partition_tile_rows": 2,
            "partition_tile_cols": 2,
            "partition_halo_px": 256,
            "seam_band_px": 64,
            "max_job_retries": 2,
        },
        boundary_fitting={"enabled": True, "mode": "divider_cubic_bspline_adaptive_v2"},
        fragmentation_regularization={"enabled": True},
        range_selection={"mode": "extent", "clip_outputs": True},
        selected_tile_count=4,
        excluded_tile_count=1,
        spatial_plan={
            "partition_rows": 1,
            "partition_cols": 1,
            "partition_tile_rows": 2,
            "partition_tile_cols": 2,
            "partition_count": 1,
            "unit_counts": {"core": 1},
            "spatial_units": [
                {
                    "unit_id": "core",
                    "unit_type": "Core",
                    "owner_key": "partition_00000_00000",
                    "pixel_window": {"x0": 0, "y0": 0, "x1": 1024, "y1": 1024},
                    "dependency_ids": ["partition_00000_00000"],
                }
            ],
        },
        package_plan={
            "package_count": 1,
            "package_by_partition": {"partition_00000_00000": "package_00000"},
            "packages": [
                {
                    "package_id": "package_00000",
                    "sequence_no": 0,
                    "partition_ids": ["partition_00000_00000"],
                }
            ],
        },
        partitions=(
            {
                "partition_id": "partition_00000_00000",
                "row": 0,
                "col": 0,
                "package_id": "package_00000",
                "core_window": {"x0": 0, "y0": 0, "x1": 1024, "y1": 1024},
                "halo_window": {"x0": 0, "y0": 0, "x1": 1024, "y1": 1024},
            },
        ),
        storage_report={"package_tile_limit": 4, "working_bytes_per_tile": 4096},
        v33_enabled=v33_enabled,
    )


class _GraphJobs:
    def __init__(self, calls: list[tuple[str, Any]]) -> None:
        self._calls = calls

    def insert_jobs(self, run_id, jobs) -> int:
        values = list(jobs)
        self._calls.append(("insert_jobs", (run_id, values)))
        return len(values)


class _GraphStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.jobs = _GraphJobs(self.calls)
        self.excluded_count = 1

    @property
    def control_graph(self):
        return self

    @property
    def run_streams(self):
        return self

    def create_run(self, run_id, digest, *, status, metadata) -> None:
        self.calls.append(("create_run", (run_id, digest, status, metadata)))

    def register_streams(self, run_id, streams) -> None:
        self.calls.append(("register_streams", (run_id, list(streams))))

    def insert_work_packages(self, run_id, packages) -> int:
        values = list(packages)
        self.calls.append(("insert_work_packages", (run_id, values)))
        return len(values)

    def insert_partitions(self, run_id, partitions) -> int:
        values = list(partitions)
        self.calls.append(("insert_partitions", (run_id, values)))
        return len(values)

    def insert_spatial_units(self, run_id, units) -> int:
        values = list(units)
        self.calls.append(("insert_spatial_units", (run_id, values)))
        return len(values)

    def insert_stream_units(self, run_id, stream_ids, unit_ids) -> int:
        self.calls.append(
            ("insert_stream_units", (run_id, list(stream_ids), list(unit_ids)))
        )
        return 1

    def insert_tiles(self, run_id, tiles) -> int:
        values = list(tiles)
        self.calls.append(("insert_tiles", (run_id, values)))
        return len(values)

    def count_tiles(self, run_id, *, status) -> int:
        self.calls.append(("count_tiles", (run_id, status)))
        return self.excluded_count


def test_preparation_keeps_checkpoint_before_v33_storage_rejection(tmp_path):
    checkpoints: list[tuple[float, str]] = []

    with pytest.raises(RunBuilderV5Error, match="confidence reserve"):
        prepare_v5_run_plan(
            run_dir=tmp_path,
            tile_rows=2,
            tile_cols=2,
            overlap=192,
            scaling={
                "partition_tile_rows": 2,
                "partition_tile_cols": 2,
                "partition_halo_px": 256,
                "seam_band_px": 64,
                "max_job_retries": 2,
            },
            boundary_fitting={"enabled": True},
            fragmentation_regularization={
                "enabled": True,
                "policy_id": "fragmentation_v33_configurable_absorption_v1",
                "publication": "authoritative_fusion_core",
                "policy_sha256": "a" * 64,
                "executor_sha256": "b" * 64,
            },
            range_selection=None,
            storage_report={
                "package_tile_limit": 4,
                "working_bytes_per_tile": 4096,
                "storage_tuning_schema_version": 2,
                "deferred_temporary_reserve_bytes": 0,
            },
            checkpoint=lambda value, message: checkpoints.append((value, message)),
        )

    assert checkpoints == [(8, "空间单元与 Work Package 规划完成")]


def test_snapshots_write_config_before_later_builder_directories(tmp_path):
    profile = {
        "profile_id": "fusion",
        "status": "approved",
        "approval": {"passed": True},
    }

    snapshots = freeze_v5_run_snapshots(
        run_dir=tmp_path,
        models=[{"model_id": "model", "version": "v1"}],
        fusion={"profile_id": "fusion", "profile": profile},
        effective_device="cpu",
        keep_score_cache=False,
        tile_batch_size=1,
        resource_tuning=None,
        plan=_plan(),
        config_fingerprint="fingerprint",
    )

    assert snapshots.class_mapping_path.is_file()
    assert snapshots.config_snapshot_path.is_file()
    assert (tmp_path / "fusion_profile_snapshot.json").is_file()
    assert not (tmp_path / "models").exists()
    assert snapshots.fusion is not None
    assert Path(str(snapshots.fusion["snapshot_path"])).is_file()


def test_control_graph_normalizes_tiles_and_preserves_checkpoints(tmp_path):
    spec_path = tmp_path / "run_spec.json"
    spec_path.write_text("{}", encoding="utf-8")
    database = _GraphStore()
    checkpoints: list[tuple[float, str]] = []
    streams = ({"stream_id": "model:model", "kind": "model"},)

    write_v5_control_graph(
        database=database,
        run_id="20260923_120000_builder",
        spec_path=spec_path,
        tile_rows=2,
        tile_cols=2,
        tile_cache_dir=tmp_path / "cache",
        excluded_tile_count=1,
        spatial_plan=_plan().spatial_plan,
        package_plan=_plan().package_plan,
        partitions=_plan().partitions,
        streams=streams,
        tiles=(
            {"row": 0, "col": 0},
            {"row": 0, "col": 1, "status": "excluded"},
            {"row": 1, "col": 0},
            {"row": 1, "col": 1},
        ),
        max_job_retries=2,
        v33_enabled=False,
        checkpoint=lambda value, message: checkpoints.append((value, message)),
    )

    assert [name for name, _value in database.calls] == [
        "create_run",
        "register_streams",
        "insert_work_packages",
        "insert_partitions",
        "insert_spatial_units",
        "insert_stream_units",
        "insert_tiles",
        "count_tiles",
        "insert_jobs",
        "insert_jobs",
    ]
    inserted_tiles = database.calls[6][1][1]
    assert inserted_tiles[0]["raster_path"] == str(tmp_path / "cache" / "tile_0_0.tif")
    assert inserted_tiles[1]["raster_path"] == ""
    assert checkpoints == [
        (16, "PostgreSQL Run 已创建"),
        (22, "Partition 与 Work Package 已写入"),
        (32, "空间单元已写入"),
        (45, "结果流空间单元已写入"),
        (68, "Tile 索引已写入"),
        (70, "Work Package Job 已写入"),
        (97, "空间拟合 Job 已写入"),
    ]
