import hashlib
import sqlite3
import time

import pytest

from labeling_tool.core.run_state_db import (
    MAX_TILE_PAGE_SIZE,
    RunStateDB,
    RunStateError,
)
from labeling_tool.core.spatial_planner import plan_spatial_units


RUN_ID = "20260717_200000_fixture"


def _database(tmp_path):
    database = RunStateDB(tmp_path / "run_state.sqlite")
    database.initialize()
    database.create_run(RUN_ID, "a" * 64)
    return database


def test_schema_enables_wal_foreign_keys_and_integrity(tmp_path):
    database = _database(tmp_path)
    pragmas = database.pragmas()
    assert str(pragmas["journal_mode"]).lower() == "wal"
    assert pragmas["foreign_keys"] == 1
    assert pragmas["user_version"] == 2
    assert pragmas["integrity_check"] == "ok"

    with sqlite3.connect(database.path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert {
        "runs",
        "streams",
        "tiles",
        "partitions",
        "spatial_units",
        "work_packages",
        "jobs",
        "artifacts",
        "artifact_dependencies",
        "unit_report_summaries",
        "object_links",
        "events",
    }.issubset(tables)


def test_unit_report_summary_is_tied_to_ready_artifact(tmp_path):
    database = _database(tmp_path)
    database.register_streams(
        RUN_ID,
        [
            {
                "stream_id": "model:test",
                "kind": "model",
                "model_id": "test",
            }
        ],
    )
    database.insert_spatial_units(
        RUN_ID,
        [
            {
                "unit_id": "core_00000",
                "unit_type": "core",
                "owner_key": "core_00000",
                "pixel_window": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
                "dependency_ids": [],
            }
        ],
    )
    database.insert_stream_units(
        RUN_ID,
        ["model:test"],
        ["core_00000"],
    )
    report_path = tmp_path / "core_00000_report.json"
    report_path.write_text('{"status":"passed"}', encoding="utf-8")
    artifact_id = database.register_artifact(
        RUN_ID,
        "unit_boundary_report",
        report_path,
        stream_id="model:test",
        unit_id="core_00000",
    )
    assert database.mark_artifact_ready(
        artifact_id,
        byte_count=report_path.stat().st_size,
        sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
    )
    report = {
        "status": "passed",
        "fit_version": "divider_cubic_bspline_adaptive_v2",
        "chain_count": 8,
        "shared_chain_count": 4,
        "spline_count": 3,
        "unchanged_count": 1,
        "skipped_invalid_count": 0,
        "max_displacement_px": 1.25,
        "diagnostics": [{}, {}],
    }
    database.upsert_unit_report_summary(
        RUN_ID,
        "model:test",
        "core_00000",
        report,
        fitted_edge_count=1,
    )

    rows = database.unit_report_summaries(RUN_ID, "model:test")
    assert len(rows) == 1
    assert rows[0]["chain_count"] == 8
    assert rows[0]["diagnostic_count"] == 2
    assert rows[0]["fitted_edge_count"] == 1
    assert rows[0]["report_sha256"] == database.get_artifact(artifact_id)["sha256"]
    aggregate = database.unit_report_summary_aggregate(RUN_ID, "model:test")
    assert aggregate["unit_count"] == 1
    assert aggregate["chain_count"] == 8
    assert aggregate["max_displacement_px"] == 1.25


def test_run_status_compare_and_set(tmp_path):
    database = _database(tmp_path)
    assert database.set_run_status(RUN_ID, "running", expected="preflight")
    assert not database.set_run_status(RUN_ID, "failed", expected="preflight")
    assert database.get_run(RUN_ID)["status"] == "running"


def test_100k_tiles_are_paged_without_unbounded_result(tmp_path):
    database = _database(tmp_path)

    def tiles():
        for index in range(100_000):
            row, col = divmod(index, 400)
            yield {
                "tile_id": f"{row}_{col}",
                "row": row,
                "col": col,
                "pixel_window": [col * 320, row * 320, 512, 512],
            }

    assert database.insert_tiles(RUN_ID, tiles()) == 100_000
    assert database.count_tiles(RUN_ID) == 100_000

    first_page = database.page_tiles(RUN_ID, limit=100_000)
    second_page = database.page_tiles(RUN_ID, limit=MAX_TILE_PAGE_SIZE, offset=500)
    assert len(first_page) == MAX_TILE_PAGE_SIZE
    assert len(second_page) == MAX_TILE_PAGE_SIZE
    assert first_page[0]["tile_id"] == "0_0"
    assert first_page[-1]["tile_id"] == "1_99"
    assert second_page[0]["tile_id"] == "1_100"
    assert database.page_tiles(RUN_ID, search="249_399")[0]["tile_id"] == "249_399"


def test_job_lease_heartbeat_completion_and_recovery(tmp_path):
    database = _database(tmp_path)
    database.insert_jobs(
        RUN_ID,
        [
            {"job_type": "tile_inference", "tile_id": "0_0", "priority": 1},
            {"job_type": "boundary_fit", "unit_id": "core_0_0", "priority": 5},
        ],
    )

    job = database.lease_next_job(RUN_ID, "worker-a", lease_seconds=5)
    assert job["job_type"] == "boundary_fit"
    assert job["attempt"] == 1
    assert database.heartbeat(
        job["job_id"], job["lease_token"], current=4, total=10
    )
    assert not database.finish_job(job["job_id"], "wrong-token")
    assert database.finish_job(job["job_id"], job["lease_token"])

    expired = database.lease_next_job(RUN_ID, "worker-b", lease_seconds=1)
    assert expired["job_type"] == "tile_inference"
    assert database.interrupt_expired_jobs(now_epoch=time.time() + 2) == 1
    retried = database.lease_next_job(RUN_ID, "worker-c")
    assert retried["job_id"] == expired["job_id"]
    assert retried["attempt"] == 2
    assert database.job_counts(RUN_ID) == {"ready": 1, "running": 1}


def test_events_are_persistent_and_foreign_keys_are_enforced(tmp_path):
    database = _database(tmp_path)
    event_id = database.append_event(
        RUN_ID,
        "partition_planned",
        message="core_0_0",
        payload={"row": 0, "col": 0},
    )
    assert event_id == 1

    with sqlite3.connect(database.path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        row = connection.execute(
            "SELECT event_type, payload_json FROM events WHERE event_id=1"
        ).fetchone()
        assert row == ("partition_planned", '{"col":0,"row":0}')
        try:
            connection.execute(
                "INSERT INTO events(run_id,timestamp,level,event_type,message,payload_json) "
                "VALUES('missing','now','info','bad','','{}')"
            )
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("foreign key violation was accepted")


def test_artifact_dependencies_update_reference_count_atomically(tmp_path):
    database = _database(tmp_path)
    database.insert_jobs(
        RUN_ID,
        [{"job_type": "partition_vectorize", "unit_id": "partition-0"}],
    )
    artifact_id = database.register_artifact(
        RUN_ID,
        "probability_part",
        tmp_path / "probability-part.tif",
        stream_id="model:test",
        unit_id="partition-0",
    )
    assert database.mark_artifact_ready(
        artifact_id, byte_count=1234, sha256="a" * 64
    )

    with sqlite3.connect(database.path) as connection:
        job_id = int(connection.execute("SELECT job_id FROM jobs").fetchone()[0])

    assert database.add_artifact_dependency(job_id, artifact_id)
    assert not database.add_artifact_dependency(job_id, artifact_id)
    assert database.get_artifact(artifact_id)["ref_count"] == 1
    assert database.cleanup_candidates(RUN_ID) == []

    assert database.release_artifact_dependency(job_id, artifact_id)
    assert not database.release_artifact_dependency(job_id, artifact_id)
    assert database.get_artifact(artifact_id)["ref_count"] == 0
    assert [
        item["artifact_id"] for item in database.cleanup_candidates(RUN_ID)
    ] == [artifact_id]


def test_artifact_cleanup_claim_is_atomic_and_auditable(tmp_path):
    database = _database(tmp_path)
    artifact_id = database.register_artifact(
        RUN_ID,
        "partition_probability",
        tmp_path / "probability-part.tif",
        stream_id="model:test",
        unit_id="partition-0",
    )
    assert database.mark_artifact_ready(
        artifact_id, byte_count=4096, sha256="c" * 64
    )

    claimed = database.claim_artifact_cleanup(artifact_id)
    assert claimed is not None
    assert claimed["status"] == "cleaning"
    assert database.claim_artifact_cleanup(artifact_id) is None
    assert database.cleanup_candidates(
        RUN_ID, kinds=("partition_probability",)
    ) == []

    assert database.finish_artifact_cleanup(artifact_id, success=True)
    assert not database.finish_artifact_cleanup(artifact_id, success=True)
    assert database.artifact_cleanup_summary(RUN_ID) == {
        "artifact_count": 1,
        "cleaned_bytes": 4096,
    }


def test_frontier_limit_prioritizes_neighbor_package(tmp_path):
    database = _database(tmp_path)
    database.insert_work_packages(
        RUN_ID,
        [
            {"package_id": "package_00000", "sequence_no": 0},
            {"package_id": "package_00001", "sequence_no": 1},
            {"package_id": "package_00002", "sequence_no": 2},
        ],
    )
    database.insert_partitions(
        RUN_ID,
        [
            {
                "partition_id": f"partition_{index}",
                "row": 0,
                "col": index,
                "core_window": {"x0": index, "y0": 0, "x1": index + 1, "y1": 1},
                "halo_window": {"x0": index, "y0": 0, "x1": index + 1, "y1": 1},
                "package_id": f"package_{index:05d}",
            }
            for index in range(3)
        ],
    )
    database.insert_spatial_units(
        RUN_ID,
        [
            {
                "unit_id": "seam_0_1",
                "unit_type": "seam",
                "owner_key": "seam_0_1",
                "pixel_window": {"x0": 0, "y0": 0, "x1": 2, "y1": 1},
                "dependency_ids": ["partition_0", "partition_1"],
            }
        ],
    )
    assert database.set_work_package_status(
        RUN_ID, "package_00000", "ready", expected="queued"
    )
    # The far package has the smaller job_id; frontier scheduling must still choose package 1.
    database.insert_jobs(
        RUN_ID,
        [
            {"job_type": "work_package", "package_id": "package_00002"},
            {"job_type": "work_package", "package_id": "package_00001"},
        ],
    )

    assert database.open_frontier_summary(RUN_ID) == {
        "unit_count": 1,
        "unit_ids": ["seam_0_1"],
        "neighbor_package_ids": ["package_00001"],
    }
    leased = database.lease_next_work_package(
        RUN_ID,
        "accelerator",
        max_open_frontier_units=1,
    )
    assert leased["package_id"] == "package_00001"


def test_artifact_dependency_rejects_cross_run_and_incomplete_inputs(tmp_path):
    database = _database(tmp_path)
    database.create_run("run-b", "b" * 64)
    database.insert_jobs(RUN_ID, [{"job_type": "fusion"}])
    artifact_id = database.register_artifact(
        "run-b", "mask", tmp_path / "mask.tif"
    )

    with sqlite3.connect(database.path) as connection:
        job_id = int(connection.execute("SELECT job_id FROM jobs").fetchone()[0])

    with pytest.raises(RunStateError, match="ready artifact from the same run"):
        database.add_artifact_dependency(job_id, artifact_id)

    assert database.mark_artifact_ready(
        artifact_id, byte_count=1, sha256="b" * 64
    )
    with pytest.raises(RunStateError, match="ready artifact from the same run"):
        database.add_artifact_dependency(job_id, artifact_id)


def test_spatial_plan_and_packages_are_persisted_with_foreign_keys(tmp_path):
    database = _database(tmp_path)
    plan = plan_spatial_units(tile_rows=17, tile_cols=17)
    packages = [
        {
            "package_id": f"package_{index:05d}",
            "sequence_no": index,
            "estimated_bytes": 1024,
        }
        for index in range(plan["partition_count"])
    ]
    package_by_partition = {
        partition["partition_id"]: packages[index]["package_id"]
        for index, partition in enumerate(plan["partitions"])
    }
    partitions = [
        {**partition, "package_id": package_by_partition[partition["partition_id"]]}
        for partition in plan["partitions"]
    ]

    assert database.insert_work_packages(RUN_ID, packages) == len(packages)
    assert database.insert_partitions(RUN_ID, partitions) == plan["partition_count"]
    assert database.insert_spatial_units(RUN_ID, plan["spatial_units"]) == len(
        plan["spatial_units"]
    )

    with sqlite3.connect(database.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM work_packages").fetchone()[0] == len(
            packages
        )
        assert connection.execute("SELECT COUNT(*) FROM partitions").fetchone()[0] == plan[
            "partition_count"
        ]
        assert connection.execute("SELECT COUNT(*) FROM spatial_units").fetchone()[0] == len(
            plan["spatial_units"]
        )


def test_stream_unit_details_are_filtered_and_paginated(tmp_path):
    database = _database(tmp_path)
    database.register_streams(
        RUN_ID,
        [{"stream_id": "model:a", "kind": "model", "model_id": "a"}],
    )
    units = [
        {
            "unit_id": f"partition_{index:05d}_00000",
            "unit_type": "partition",
            "owner_key": f"p{index}",
            "pixel_window": {"x0": 0, "y0": index, "x1": 1, "y1": index + 1},
            "dependency_ids": [],
        }
        for index in range(620)
    ]
    database.insert_spatial_units(RUN_ID, units)
    database.insert_stream_units(
        RUN_ID, ["model:a"], (item["unit_id"] for item in units)
    )
    assert database.count_stream_units(
        RUN_ID, "model:a", unit_type="partition"
    ) == 620
    first = database.page_stream_units(
        RUN_ID, "model:a", unit_type="partition", limit=900
    )
    second = database.page_stream_units(
        RUN_ID, "model:a", unit_type="partition", limit=500, offset=500
    )
    assert len(first) == 500
    assert len(second) == 120
    assert first[0]["unit_id"] == "partition_00000_00000"
    assert second[0]["unit_id"] == "partition_00500_00000"
    assert database.count_stream_units(RUN_ID, "model:a", search="00619") == 1
