import hashlib
import sqlite3
from pathlib import Path

import pytest

from labeling_tool.core.manual_package_reset import reset_failed_work_packages
from labeling_tool.core.run_state_db import RunStateDB, RunStateError


RUN_ID = "20260729_120000_retry"
STREAM_ID = "model:a"


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _ready_artifact(
    database: RunStateDB,
    path: Path,
    *,
    kind: str,
    unit_id: str,
) -> int:
    artifact_id = database.register_artifact(
        RUN_ID,
        kind,
        path,
        stream_id=STREAM_ID,
        unit_id=unit_id,
    )
    payload = path.read_bytes()
    assert database.mark_artifact_ready(
        artifact_id,
        byte_count=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    return artifact_id


def _state(tmp_path):
    output_root = tmp_path / "output"
    run_dir = output_root / "runs" / RUN_ID
    run_dir.mkdir(parents=True)
    database = RunStateDB(run_dir / "run_state.sqlite")
    database.initialize()
    database.create_run(RUN_ID, "a" * 64, status="failed")
    database.register_streams(
        RUN_ID,
        [{"stream_id": STREAM_ID, "kind": "model", "model_id": "a"}],
    )
    database.insert_work_packages(
        RUN_ID,
        [
            {
                "package_id": "package_00000",
                "sequence_no": 0,
                "status": "failed",
            },
            {
                "package_id": "package_00001",
                "sequence_no": 1,
                "status": "ready",
            },
        ],
    )
    database.insert_partitions(
        RUN_ID,
        [
            {
                "partition_id": "partition_0",
                "row": 0,
                "col": 0,
                "core_window": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
                "halo_window": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
                "package_id": "package_00000",
            },
            {
                "partition_id": "partition_1",
                "row": 0,
                "col": 1,
                "core_window": {"x0": 1, "y0": 0, "x1": 2, "y1": 1},
                "halo_window": {"x0": 1, "y0": 0, "x1": 2, "y1": 1},
                "package_id": "package_00001",
            },
        ],
    )
    units = [
        {
            "unit_id": "core_0",
            "unit_type": "core",
            "owner_key": "core_0",
            "pixel_window": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
            "dependency_ids": ["partition_0"],
        },
        {
            "unit_id": "seam_0_1",
            "unit_type": "seam",
            "owner_key": "seam_0_1",
            "pixel_window": {"x0": 0, "y0": 0, "x1": 2, "y1": 1},
            "dependency_ids": ["partition_0", "partition_1"],
        },
        {
            "unit_id": "core_1",
            "unit_type": "core",
            "owner_key": "core_1",
            "pixel_window": {"x0": 1, "y0": 0, "x1": 2, "y1": 1},
            "dependency_ids": ["partition_1"],
        },
    ]
    database.insert_spatial_units(RUN_ID, units)
    database.insert_stream_units(
        RUN_ID,
        [STREAM_ID],
        (item["unit_id"] for item in units),
    )
    database.insert_jobs(
        RUN_ID,
        [
            {
                "job_type": "work_package",
                "package_id": "package_00000",
                "status": "failed",
                "max_attempts": 3,
            },
            {
                "job_type": "work_package",
                "package_id": "package_00001",
                "status": "ready",
                "max_attempts": 3,
            },
            *[
                {
                    "job_type": "unit_fit",
                    "stream_id": STREAM_ID,
                    "unit_id": unit_id,
                    "status": "ready",
                    "max_attempts": 3,
                }
                for unit_id in ("core_0", "seam_0_1", "core_1")
            ],
        ],
    )
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            """UPDATE jobs SET attempt=3, progress_current=7,
               progress_total=9, error='exhausted'
               WHERE package_id='package_00000'"""
        )
        connection.execute(
            """UPDATE jobs SET attempt=2
               WHERE job_type='unit_fit' AND unit_id IN ('core_0','seam_0_1')"""
        )
        connection.execute(
            """UPDATE work_packages SET attempt=3
               WHERE package_id='package_00000'"""
        )

    artifact_paths = {}
    for unit_id, kind, relative, payload in (
        (
            "partition_0",
            "partition_probability",
            "tmp/probability_parts/a/partition_0.tif",
            b"probability-0",
        ),
        (
            "partition_0",
            "core_mask",
            "models/a/raster_parts/partition_0_mask.tif",
            b"mask-0",
        ),
        (
            "partition_1",
            "partition_probability",
            "tmp/probability_parts/a/partition_1.tif",
            b"probability-1",
        ),
        (
            "partition_1",
            "core_mask",
            "models/a/raster_parts/partition_1_mask.tif",
            b"mask-1",
        ),
        (
            "core_0",
            "unit_formal",
            "tmp/unit_outputs/model_a/core_0_formal.gpkg",
            b"core-0",
        ),
        (
            "seam_0_1",
            "unit_formal",
            "tmp/unit_outputs/model_a/seam_0_1_formal.gpkg",
            b"seam",
        ),
        (
            "core_1",
            "unit_formal",
            "tmp/unit_outputs/model_a/core_1_formal.gpkg",
            b"core-1",
        ),
        (
            "mosaic",
            "mask_vrt",
            "models/a/mask_mosaic.vrt",
            b"vrt",
        ),
        (
            "assembled",
            "semantic_polygons",
            "models/a/semantic_polygons.gpkg",
            b"assembled",
        ),
    ):
        path = _write(run_dir / relative, payload)
        artifact_paths[(unit_id, kind)] = (
            path,
            _ready_artifact(
                database,
                path,
                kind=kind,
                unit_id=unit_id,
            ),
        )

    for unit_id in ("core_0", "seam_0_1", "core_1"):
        report_path = _write(
            run_dir / "tmp" / "unit_outputs" / "model_a" / f"{unit_id}_report.json",
            b'{"status":"passed"}',
        )
        _ready_artifact(
            database,
            report_path,
            kind="unit_boundary_report",
            unit_id=unit_id,
        )
        database.upsert_unit_report_summary(
            RUN_ID,
            STREAM_ID,
            unit_id,
            {
                "status": "passed",
                "fit_version": "divider_cubic_bspline_adaptive_v2",
                "diagnostics": [],
            },
        )

    with sqlite3.connect(database.path) as connection:
        jobs = {
            row[0]: int(row[1])
            for row in connection.execute(
                """SELECT unit_id, job_id FROM jobs
                   WHERE job_type='unit_fit'"""
            )
        }
    database.add_artifact_dependency(
        jobs["core_0"],
        artifact_paths[("partition_0", "partition_probability")][1],
    )
    database.add_artifact_dependency(
        jobs["seam_0_1"],
        artifact_paths[("partition_0", "partition_probability")][1],
    )
    database.add_artifact_dependency(
        jobs["seam_0_1"],
        artifact_paths[("partition_1", "partition_probability")][1],
    )
    database.add_artifact_dependency(
        jobs["core_1"],
        artifact_paths[("partition_1", "partition_probability")][1],
    )
    database.register_object_parts(
        RUN_ID,
        STREAM_ID,
        [
            {"part_id": "left", "class_code": 12, "unit_id": "core_0"},
            {"part_id": "right", "class_code": 12, "unit_id": "seam_0_1"},
        ],
    )
    database.add_object_link(RUN_ID, STREAM_ID, "left", "right", 12)

    package_file = _write(
        run_dir / "tmp" / "work_packages" / "package_00000" / "scores" / "a.npz",
        b"score",
    )
    unit_temp = _write(
        run_dir / "tmp" / "unit_outputs" / "model_a" / ".core_0_raw.42.tmp.gpkg",
        b"temporary",
    )
    acceptance = _write(
        run_dir / "logs" / "scale_acceptance_report.json",
        b"stale-acceptance",
    )
    _write(run_dir / "run_manifest.json", b"stale-manifest")
    tile_cache = _write(
        output_root / "cache" / RUN_ID / "tile_cache" / "tile_0_0.tif",
        b"shared-tile-cache",
    )
    spec = {
        "schema_version": 2,
        "run_id": RUN_ID,
        "run_dir": str(run_dir),
        "state_db": str(database.path),
        "output_root": str(output_root),
        "tile_cache_dir": str(tile_cache.parent),
        "streams": [
            {
                "stream_id": STREAM_ID,
                "kind": "model",
                "model_id": "a",
            }
        ],
    }
    return {
        "database": database,
        "spec": spec,
        "run_dir": run_dir,
        "artifact_paths": artifact_paths,
        "package_file": package_file,
        "unit_temp": unit_temp,
        "acceptance": acceptance,
        "tile_cache": tile_cache,
        "tile_cache_sha": hashlib.sha256(tile_cache.read_bytes()).hexdigest(),
        "jobs": jobs,
    }


def test_manual_retry_resets_failed_package_and_downstream_but_preserves_cache(
    tmp_path,
):
    state = _state(tmp_path)
    database = state["database"]

    result = reset_failed_work_packages(
        state["spec"],
        database=database,
    )

    assert result["package_ids"] == ["package_00000"]
    assert result["affected_unit_count"] == 2
    assert result["tile_cache_action"] == "preserved"
    assert state["tile_cache"].is_file()
    assert (
        hashlib.sha256(state["tile_cache"].read_bytes()).hexdigest()
        == state["tile_cache_sha"]
    )
    assert not state["package_file"].exists()
    assert not state["unit_temp"].exists()
    assert not state["acceptance"].exists()
    assert not (state["run_dir"] / "run_manifest.json").exists()

    for key in (
        ("partition_0", "partition_probability"),
        ("partition_0", "core_mask"),
        ("core_0", "unit_formal"),
        ("seam_0_1", "unit_formal"),
        ("mosaic", "mask_vrt"),
        ("assembled", "semantic_polygons"),
    ):
        assert not state["artifact_paths"][key][0].exists()
    for key in (
        ("partition_1", "partition_probability"),
        ("partition_1", "core_mask"),
        ("core_1", "unit_formal"),
    ):
        assert state["artifact_paths"][key][0].is_file()

    assert database.get_run(RUN_ID)["status"] == "planned"
    assert database.get_work_package(RUN_ID, "package_00000")["status"] == "queued"
    assert database.get_work_package(RUN_ID, "package_00000")["attempt"] == 0
    assert database.get_work_package(RUN_ID, "package_00001")["status"] == "ready"
    with sqlite3.connect(database.path) as connection:
        connection.row_factory = sqlite3.Row
        jobs = {
            (row["job_type"], row["package_id"], row["unit_id"]): dict(row)
            for row in connection.execute(
                "SELECT * FROM jobs ORDER BY job_id"
            ).fetchall()
        }
        assert connection.execute(
            "SELECT COUNT(*) FROM object_nodes"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM object_links"
        ).fetchone()[0] == 0
    package_job = jobs[("work_package", "package_00000", "")]
    assert package_job["status"] == "queued"
    assert package_job["attempt"] == 0
    assert package_job["progress_current"] == 0
    assert package_job["error"] == ""
    assert jobs[("work_package", "package_00001", "")]["status"] == "ready"
    assert jobs[("unit_fit", "", "core_0")]["status"] == "queued"
    assert jobs[("unit_fit", "", "core_0")]["attempt"] == 0
    assert jobs[("unit_fit", "", "seam_0_1")]["status"] == "queued"
    assert jobs[("unit_fit", "", "core_1")]["status"] == "ready"

    summaries = database.unit_report_summaries(RUN_ID, STREAM_ID)
    assert len(summaries) == 1
    remaining_summary = summaries[0]
    assert remaining_summary["unit_id"] == "core_1"
    with sqlite3.connect(database.path) as connection:
        seam_dependency_ids = {
            int(row[0])
            for row in connection.execute(
                "SELECT artifact_id FROM artifact_dependencies WHERE job_id=?",
                (state["jobs"]["seam_0_1"],),
            )
        }
    assert seam_dependency_ids == {
        state["artifact_paths"][("partition_1", "partition_probability")][1]
    }


def test_failed_unit_reset_maps_every_partition_back_to_its_package(tmp_path):
    state = _state(tmp_path)
    database = state["database"]
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            "UPDATE work_packages SET status='ready', attempt=1"
        )
        connection.execute(
            """UPDATE jobs SET status='ready', attempt=1, error=''
               WHERE job_type='work_package'"""
        )
        connection.execute(
            """UPDATE jobs SET status='failed', attempt=max_attempts,
               error='seam exhausted' WHERE job_type='unit_fit'
               AND unit_id='seam_0_1'"""
        )

    plan = database.begin_failed_package_reset(RUN_ID)

    assert plan["package_ids"] == ["package_00000", "package_00001"]
    assert set(plan["affected_unit_ids"]) == {"core_0", "core_1", "seam_0_1"}
    assert database.get_run(RUN_ID)["status"] == "resetting"
    assert database.lease_next_job(RUN_ID, "unexpected-worker") is None


def test_manual_package_reset_rejects_completed_run(tmp_path):
    state = _state(tmp_path)
    database = state["database"]
    assert database.set_run_status(RUN_ID, "ready", expected="failed")

    with pytest.raises(RunStateError, match="failed, stopped, or resetting"):
        database.begin_failed_package_reset(RUN_ID)
