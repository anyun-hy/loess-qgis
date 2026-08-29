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
        "stream_runtime_progress",
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


def test_stream_runtime_progress_is_bounded_restart_visible_and_phase_aware(tmp_path):
    database = _database(tmp_path)
    database.register_streams(
        RUN_ID,
        [{"stream_id": "model:test", "kind": "model", "model_id": "test"}],
    )
    database.upsert_stream_runtime_progress(
        RUN_ID,
        "model:test",
        stage="assembly",
        phase="write_raw",
        phase_name="写入 Raw GPKG",
        phase_index=4,
        phase_total=9,
        current=5,
        total=10,
        feature_count=123,
        message="writing",
    )
    first = database.stream_runtime_progress(RUN_ID)["model:test"]
    time.sleep(0.002)
    database.upsert_stream_runtime_progress(
        RUN_ID,
        "model:test",
        stage="assembly",
        phase="write_raw",
        phase_name="写入 Raw GPKG",
        phase_index=4,
        phase_total=9,
        current=10,
        total=10,
        feature_count=456,
        message="written",
    )
    same_phase = database.monitor_snapshot(RUN_ID)[
        "stream_runtime_progress"
    ]["model:test"]
    assert same_phase["phase_started_at"] == first["phase_started_at"]
    assert same_phase["progress_current"] == 10
    assert same_phase["feature_count"] == 456

    time.sleep(0.002)
    database.upsert_stream_runtime_progress(
        RUN_ID,
        "model:test",
        stage="assembly",
        phase="write_formal",
        phase_name="写入正式 GPKG",
        phase_index=5,
        phase_total=9,
        current=1,
        total=10,
    )
    next_phase = database.stream_runtime_progress(RUN_ID)["model:test"]
    assert next_phase["phase_started_at"] != first["phase_started_at"]
    assert len(database.stream_runtime_progress(RUN_ID)) == 1

    database.fail_stream_runtime_progress(RUN_ID, "model:test", "disk error")
    failed = database.stream_runtime_progress(RUN_ID)["model:test"]
    assert failed["status"] == "failed"
    assert failed["message"] == "disk error"


def test_monitor_snapshot_keeps_pre_progress_schema_v2_runs_readable(tmp_path):
    database = _database(tmp_path)
    with database._connection() as connection:
        connection.execute("DROP TABLE stream_runtime_progress")

    snapshot = database.monitor_snapshot(RUN_ID)

    assert snapshot["run"]["run_id"] == RUN_ID
    assert snapshot["stream_runtime_progress"] == {}


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
    assert database.count_tiles(RUN_ID, search="249_399") == 1


def test_tile_search_escapes_like_metacharacters_and_matches_page_count(tmp_path):
    database = _database(tmp_path)
    database.insert_tiles(
        RUN_ID,
        [
            {"tile_id": "a_b", "row": 0, "col": 0, "status": "ready"},
            {"tile_id": "axb", "row": 0, "col": 1, "status": "failed"},
            {"tile_id": "a%b", "row": 0, "col": 2, "status": "ready"},
            {"tile_id": "aXb", "row": 0, "col": 3, "status": "ready"},
        ],
    )

    for search, expected in (("a_b", ["a_b"]), ("a%b", ["a%b"])):
        page = database.page_tiles(RUN_ID, search=search)
        assert [row["tile_id"] for row in page] == expected
        assert database.count_tiles(RUN_ID, search=search) == len(page)

    assert database.count_tiles(RUN_ID, status="ready", search="a_b") == 1
    assert database.count_tiles(RUN_ID, status="failed", search="a_b") == 0


def test_monitor_aggregates_unit_types_and_active_package(tmp_path):
    database = _database(tmp_path)
    database.register_streams(
        RUN_ID,
        [{"stream_id": "model:test", "kind": "model", "model_id": "test"}],
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
            },
            {
                "unit_id": "seam_horizontal_00000",
                "unit_type": "seam_horizontal",
                "owner_key": "seam_horizontal_00000",
                "pixel_window": {"x0": 0, "y0": 0, "x1": 2, "y1": 1},
                "dependency_ids": [],
            },
        ],
    )
    database.insert_stream_units(
        RUN_ID, ["model:test"], ["core_00000", "seam_horizontal_00000"]
    )
    assert database.set_stream_unit_status(
        RUN_ID, "model:test", "core_00000", "ready"
    )
    assert database.set_stream_unit_status(
        RUN_ID, "model:test", "seam_horizontal_00000", "running"
    )
    assert database.stream_unit_type_counts(RUN_ID, "model:test") == {
        "core": {"ready": 1},
        "seam_horizontal": {"running": 1},
    }
    database.insert_jobs(
        RUN_ID,
        [
            {
                "job_type": "unit_fit",
                "stream_id": "model:test",
                "unit_id": "core_00000",
                "status": "ready",
            },
            {
                "job_type": "unit_fit",
                "stream_id": "model:test",
                "unit_id": "seam_horizontal_00000",
                "status": "interrupted",
            },
        ],
    )

    database.insert_work_packages(
        RUN_ID, [{"package_id": "package_00000", "sequence_no": 0}]
    )
    database.insert_jobs(
        RUN_ID, [{"job_type": "work_package", "package_id": "package_00000"}]
    )
    leased = database.lease_next_work_package(
        RUN_ID, "monitor-worker", max_open_frontier_units=64
    )
    assert leased is not None
    active = database.active_work_package_job(RUN_ID)
    assert active is not None
    assert active["package_id"] == "package_00000"
    assert active["sequence_no"] == 0
    assert active["package_started_at"]

    snapshot = database.monitor_snapshot(RUN_ID)
    assert snapshot["run"]["run_id"] == RUN_ID
    assert snapshot["job_counts"]["work_package"] == {"running": 1}
    assert snapshot["job_counts"]["unit_fit"] == {"interrupted": 1, "ready": 1}
    assert snapshot["active_work_package"]["package_id"] == "package_00000"
    assert snapshot["stream_unit_type_counts"]["model:test"] == {
        "core": {"ready": 1},
        "seam_horizontal": {"running": 1},
    }
    assert snapshot["stream_unit_job_type_counts"]["model:test"] == {
        "core": {"ready": 1},
        "seam_horizontal": {"interrupted": 1},
    }

def test_monitor_snapshot_uses_one_explicitly_closed_connection(
    tmp_path, monkeypatch
):
    database = _database(tmp_path)
    original_connect = database._connect
    opened = []

    def tracked_connect():
        connection = original_connect()
        opened.append(connection)
        return connection

    monkeypatch.setattr(database, "_connect", tracked_connect)
    snapshot = database.monitor_snapshot(RUN_ID)
    assert snapshot["run"]["run_id"] == RUN_ID
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        opened[0].execute("SELECT 1")


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
    # Interrupted work is resumable and does not consume a failure attempt.
    assert retried["attempt"] == 1
    assert database.job_counts(RUN_ID) == {"ready": 1, "running": 1}


def test_expired_job_recovery_can_be_limited_to_one_run(tmp_path):
    database = _database(tmp_path)
    other_run = "20260717_200001_other"
    database.create_run(other_run, "b" * 64)
    for run_id in (RUN_ID, other_run):
        database.insert_jobs(run_id, [{"job_type": "unit_fit", "unit_id": "core"}])
    selected = database.lease_next_job(RUN_ID, "selected", lease_seconds=1)
    untouched = database.lease_next_job(other_run, "untouched", lease_seconds=1)

    assert database.interrupt_expired_jobs(
        run_id=RUN_ID, now_epoch=time.time() + 2
    ) == 1

    assert database.get_job(selected["job_id"])["status"] == "interrupted"
    assert database.get_job(untouched["job_id"])["status"] == "running"


def test_work_package_lease_and_crash_cleanup_are_atomic(tmp_path):
    database = _database(tmp_path)
    database.insert_work_packages(
        RUN_ID,
        [{"package_id": "package_00000", "sequence_no": 0}],
    )
    database.insert_jobs(
        RUN_ID,
        [{"job_type": "work_package", "package_id": "package_00000"}],
    )

    leased = database.lease_next_work_package(
        RUN_ID,
        "package-worker",
        max_open_frontier_units=64,
        lease_seconds=120,
    )
    assert leased is not None
    assert database.get_work_package(RUN_ID, "package_00000")["status"] == "running"
    assert database.get_job(leased["job_id"])["status"] == "running"

    # Simulate a worker process dying after lease acquisition but before the
    # Work Package runtime starts. Administrative cleanup must close both rows.
    assert database.interrupt_work_package_worker(RUN_ID, "package-worker") == 1
    assert database.get_work_package(RUN_ID, "package_00000")["status"] == "interrupted"
    interrupted = database.get_job(leased["job_id"])
    assert interrupted["status"] == "interrupted"
    assert interrupted["attempt"] == 0

    resumed = database.lease_job(
        leased["job_id"],
        "replacement-worker",
        lease_seconds=120,
    )
    assert resumed is not None
    assert resumed["attempt"] == 1
    assert database.get_work_package(RUN_ID, "package_00000")["status"] == "running"


def test_expired_worker_token_cannot_finish_but_admin_can_interrupt(tmp_path):
    database = _database(tmp_path)
    database.insert_jobs(RUN_ID, [{"job_type": "tile_inference", "tile_id": "0_0"}])
    leased = database.lease_next_job(RUN_ID, "worker", lease_seconds=120)
    assert leased is not None
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            "UPDATE jobs SET lease_expires=? WHERE job_id=?",
            (time.time() - 1, leased["job_id"]),
        )

    assert not database.finish_job(leased["job_id"], leased["lease_token"])
    assert database.get_job(leased["job_id"])["status"] == "running"
    assert database.interrupt_job(leased["job_id"], leased["lease_token"])
    interrupted = database.get_job(leased["job_id"])
    assert interrupted["status"] == "interrupted"
    assert interrupted["attempt"] == 0


@pytest.mark.parametrize("lease_mode", ["next", "exact"])
def test_unit_fit_waits_for_all_dependency_packages_to_be_ready(
    tmp_path,
    lease_mode,
):
    database = _database(tmp_path)
    database.insert_work_packages(
        RUN_ID,
        [{"package_id": "package_00000", "sequence_no": 0}],
    )
    database.insert_partitions(
        RUN_ID,
        [
            {
                "partition_id": "partition_00000",
                "row": 0,
                "col": 0,
                "core_window": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
                "halo_window": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
                "package_id": "package_00000",
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
                "dependency_ids": ["partition_00000"],
            }
        ],
    )
    database.insert_jobs(
        RUN_ID,
        [
            {"job_type": "work_package", "package_id": "package_00000"},
            {
                "job_type": "unit_fit",
                "stream_id": "model:test",
                "unit_id": "core_00000",
            },
        ],
    )
    with sqlite3.connect(database.path) as connection:
        rows = dict(
            connection.execute(
                "SELECT job_type, job_id FROM jobs WHERE run_id=?",
                (RUN_ID,),
            ).fetchall()
        )
    package_job = database.lease_job(
        rows["work_package"],
        "package-worker",
        lease_seconds=120,
    )
    assert package_job is not None
    assert database.get_work_package(RUN_ID, "package_00000")["status"] == "running"

    probability = database.register_artifact(
        RUN_ID,
        "partition_probability",
        tmp_path / "partition_00000_probability.tif",
        stream_id="model:test",
        unit_id="partition_00000",
    )
    assert database.mark_artifact_ready(
        probability,
        byte_count=1,
        sha256="f" * 64,
    )
    assert database.add_artifact_dependency(rows["unit_fit"], probability)

    def lease_unit():
        if lease_mode == "next":
            return database.lease_next_job(
                RUN_ID,
                "unit-worker",
                job_types=("unit_fit",),
                lease_seconds=120,
            )
        return database.lease_job(
            rows["unit_fit"],
            "unit-worker",
            lease_seconds=120,
        )

    # The Artifact exists and is linked, but it is not consumable until the
    # producing Package and its leased control Job commit ready atomically.
    assert lease_unit() is None
    assert database.get_job(rows["unit_fit"])["attempt"] == 0
    assert database.complete_work_package_job(
        RUN_ID,
        "package_00000",
        package_job["job_id"],
        package_job["lease_token"],
    )
    assert database.get_work_package(RUN_ID, "package_00000")["status"] == "ready"
    leased_unit = lease_unit()
    assert leased_unit is not None
    assert leased_unit["job_id"] == rows["unit_fit"]
    assert leased_unit["attempt"] == 1


def test_resumed_package_reset_keeps_original_cross_package_scope(tmp_path):
    database = _database(tmp_path)
    assert database.set_run_status(RUN_ID, "failed", expected="preflight")
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
                "partition_id": "partition_00000",
                "row": 0,
                "col": 0,
                "core_window": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
                "halo_window": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
                "package_id": "package_00000",
            },
            {
                "partition_id": "partition_00001",
                "row": 0,
                "col": 1,
                "core_window": {"x0": 1, "y0": 0, "x1": 2, "y1": 1},
                "halo_window": {"x0": 1, "y0": 0, "x1": 2, "y1": 1},
                "package_id": "package_00001",
            },
        ],
    )
    database.insert_spatial_units(
        RUN_ID,
        [
            {
                "unit_id": "seam_00000_00001",
                "unit_type": "seam",
                "owner_key": "seam_00000_00001",
                "pixel_window": {"x0": 0, "y0": 0, "x1": 2, "y1": 1},
                "dependency_ids": ["partition_00000", "partition_00001"],
            }
        ],
    )
    database.insert_jobs(
        RUN_ID,
        [
            {
                "job_type": "work_package",
                "package_id": "package_00000",
                "status": "failed",
            },
            {
                "job_type": "work_package",
                "package_id": "package_00001",
                "status": "ready",
            },
            {
                "job_type": "unit_fit",
                "unit_id": "seam_00000_00001",
                "status": "ready",
            },
        ],
    )

    first = database.begin_failed_package_reset(RUN_ID)
    assert first["package_ids"] == ["package_00000"]
    assert database.get_job(first["job_ids"][-1])["status"] == "resetting"

    resumed = database.begin_failed_package_reset(RUN_ID)
    assert resumed["package_ids"] == ["package_00000"]
    assert database.get_work_package(RUN_ID, "package_00001")["status"] == "ready"
    with sqlite3.connect(database.path) as connection:
        neighbor_job_status = connection.execute(
            """SELECT status FROM jobs WHERE run_id=?
               AND job_type='work_package' AND package_id='package_00001'""",
            (RUN_ID,),
        ).fetchone()[0]
    assert neighbor_job_status == "ready"


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


def _partition_publish_state(tmp_path):
    database = _database(tmp_path)
    stream_id = "model:test"
    partition_id = "partition_00000_00023"
    database.register_streams(
        RUN_ID,
        [{"stream_id": stream_id, "kind": "model", "model_id": "test"}],
    )
    database.insert_work_packages(
        RUN_ID,
        [{"package_id": "package_00000", "sequence_no": 0}],
    )
    database.insert_partitions(
        RUN_ID,
        [
            {
                "partition_id": partition_id,
                "row": 0,
                "col": 23,
                "core_window": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
                "halo_window": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
                "package_id": "package_00000",
            }
        ],
    )
    units = [
        {
            "unit_id": f"core_{index}",
            "unit_type": "core",
            "owner_key": f"core_{index}",
            "pixel_window": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
            "dependency_ids": [partition_id],
        }
        for index in range(6)
    ]
    database.insert_spatial_units(RUN_ID, units)
    database.insert_stream_units(
        RUN_ID, [stream_id], [item["unit_id"] for item in units]
    )
    database.insert_jobs(
        RUN_ID,
        [
            {
                "job_type": "unit_fit",
                "stream_id": stream_id,
                "unit_id": item["unit_id"],
            }
            for item in units
        ],
    )
    path = tmp_path / "partition_00000_00023.tif"
    path.write_bytes(b"probability")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return database, stream_id, partition_id, path, digest


def test_partition_publish_links_all_consumers_before_cleanup_visibility(tmp_path):
    database, stream_id, partition_id, path, digest = _partition_publish_state(
        tmp_path
    )

    artifact_id = database.publish_partition_artifact(
        RUN_ID,
        stream_id,
        partition_id,
        path,
        byte_count=path.stat().st_size,
        sha256=digest,
    )

    artifact = database.get_artifact(artifact_id)
    assert artifact["status"] == "ready"
    assert artifact["ref_count"] == 6
    assert database.cleanup_candidates(
        RUN_ID, kinds=("partition_probability",)
    ) == []
    with sqlite3.connect(database.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM artifact_dependencies WHERE artifact_id=?",
            (artifact_id,),
        ).fetchone()[0] == 6


def test_partition_publish_rolls_back_ready_when_dependency_link_fails(tmp_path):
    database, stream_id, partition_id, path, digest = _partition_publish_state(
        tmp_path
    )
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            """CREATE TRIGGER reject_partition_dependency
               BEFORE INSERT ON artifact_dependencies
               BEGIN SELECT RAISE(ABORT, 'injected dependency failure'); END"""
        )

    with pytest.raises(sqlite3.IntegrityError, match="injected dependency failure"):
        database.publish_partition_artifact(
            RUN_ID,
            stream_id,
            partition_id,
            path,
            byte_count=path.stat().st_size,
            sha256=digest,
        )

    assert database.artifact_for_stream_unit(
        RUN_ID, stream_id, partition_id, "partition_probability"
    ) is None
    assert database.cleanup_candidates(
        RUN_ID, kinds=("partition_probability",)
    ) == []


def test_cleaned_partition_can_be_republished_before_consumers_start(tmp_path):
    database, stream_id, partition_id, path, digest = _partition_publish_state(
        tmp_path
    )
    artifact_id = database.publish_partition_artifact(
        RUN_ID,
        stream_id,
        partition_id,
        path,
        byte_count=path.stat().st_size,
        sha256=digest,
    )
    with sqlite3.connect(database.path) as connection:
        job_ids = [
            int(row[0])
            for row in connection.execute(
                "SELECT job_id FROM jobs WHERE job_type='unit_fit'"
            ).fetchall()
        ]
    for job_id in job_ids:
        database.release_job_artifacts(job_id)
    assert database.claim_artifact_cleanup(artifact_id) is not None
    assert database.finish_artifact_cleanup(artifact_id, success=True)

    assert database.publish_partition_artifact(
        RUN_ID,
        stream_id,
        partition_id,
        path,
        byte_count=path.stat().st_size,
        sha256=digest,
    ) == artifact_id
    restored = database.get_artifact(artifact_id)
    assert restored["status"] == "ready"
    assert restored["ref_count"] == 6


def _fragmentation_v33_state(tmp_path):
    database = _database(tmp_path)
    stream_id = "fusion:test"
    partition_id = "partition_00000_00000"
    database.register_streams(
        RUN_ID,
        [{"stream_id": stream_id, "kind": "fusion", "profile_id": "test"}],
    )
    database.insert_work_packages(
        RUN_ID,
        [{"package_id": "package_00000", "sequence_no": 0, "status": "ready"}],
    )
    database.insert_partitions(
        RUN_ID,
        [
            {
                "partition_id": partition_id,
                "row": 0,
                "col": 0,
                "core_window": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
                "halo_window": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
                "package_id": "package_00000",
                "status": "ready",
            }
        ],
    )
    database.insert_spatial_units(
        RUN_ID,
        [
            {
                "unit_id": f"fragmentation_v33_partition:{partition_id}",
                "unit_type": "FragmentationV33Partition",
                "owner_key": partition_id,
                "pixel_window": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
                "dependency_ids": [partition_id],
            },
            {
                "unit_id": "fragmentation_v33_finalize",
                "unit_type": "FragmentationV33Finalize",
                "owner_key": "all_partition_owner_cores",
                "pixel_window": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
                "dependency_ids": [partition_id],
            },
        ],
    )
    database.insert_jobs(
        RUN_ID,
        [
            {
                "job_type": "fragmentation_v33",
                "stream_id": stream_id,
                "unit_id": f"fragmentation_v33_partition:{partition_id}",
            },
            {
                "job_type": "fragmentation_v33",
                "stream_id": stream_id,
                "unit_id": "fragmentation_v33_finalize",
            },
        ],
    )
    return database, stream_id, partition_id


def _publish_v33_outputs(database, tmp_path, stream_id, partition_id):
    mask = tmp_path / f"{partition_id}_v33_mask.tif"
    audit = tmp_path / f"{partition_id}_v33_audit.json"
    mask.write_bytes(b"v33-mask")
    audit.write_bytes(b"v33-audit")
    return database.publish_fragmentation_v33_output_pair(
        RUN_ID,
        stream_id,
        partition_id,
        mask_path=mask,
        mask_byte_count=mask.stat().st_size,
        mask_sha256=hashlib.sha256(mask.read_bytes()).hexdigest(),
        audit_path=audit,
        audit_byte_count=audit.stat().st_size,
        audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest(),
        production=None,
    )


def test_v33_inputs_publish_and_terminal_release_are_atomic(tmp_path):
    database, stream_id, partition_id = _fragmentation_v33_state(tmp_path)
    context = tmp_path / "context.tif"
    context.write_bytes(b"context")
    digest = hashlib.sha256(context.read_bytes()).hexdigest()

    artifact_id = database.publish_fragmentation_v33_context(
        RUN_ID,
        stream_id,
        partition_id,
        context,
        byte_count=context.stat().st_size,
        sha256=digest,
    )

    assert database.get_artifact(artifact_id)["ref_count"] == 2
    assert database.cleanup_candidates(
        RUN_ID, kinds=("v3_context_core",)
    ) == []
    probability = tmp_path / "probability.tif"
    probability.write_bytes(b"probability")
    database.publish_partition_artifact(
        RUN_ID,
        stream_id,
        partition_id,
        probability,
        byte_count=probability.stat().st_size,
        sha256=hashlib.sha256(probability.read_bytes()).hexdigest(),
    )
    assert database.lease_next_fragmentation_v33(
        RUN_ID, "v33-worker", lease_seconds=120
    ) is None
    baseline = tmp_path / "baseline.tif"
    baseline.write_bytes(b"baseline")
    baseline_id = database.publish_fragmentation_v33_baseline_core(
        RUN_ID,
        stream_id,
        partition_id,
        baseline,
        byte_count=baseline.stat().st_size,
        sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
    )
    assert database.get_artifact(baseline_id)["ref_count"] == 2
    leased = database.lease_next_fragmentation_v33(
        RUN_ID, "v33-worker", lease_seconds=120
    )
    assert leased is not None
    with pytest.raises(RunStateError, match="v33_staged_mask incomplete"):
        database.complete_fragmentation_v33_job(
            leased["job_id"], leased["lease_token"]
        )
    staged_ids = _publish_v33_outputs(database, tmp_path, stream_id, partition_id)
    assert [database.get_artifact(item)["ref_count"] for item in staged_ids] == [1, 1]
    assert database.cleanup_candidates(
        RUN_ID, kinds=("v33_staged_mask", "v33_staged_audit")
    ) == []
    assert database.complete_fragmentation_v33_job(
        leased["job_id"], leased["lease_token"]
    )
    completed = database.get_job(leased["job_id"])
    assert completed["status"] == "ready"
    assert completed["progress_current"] == 1
    assert completed["progress_total"] == 1
    # The finalize barrier retains the same inputs after an owner-stage commit.
    assert database.get_artifact(artifact_id)["ref_count"] == 1
    assert database.get_artifact(baseline_id)["ref_count"] == 1
    assert database.cleanup_candidates(
        RUN_ID,
        kinds=("partition_probability", "v3_context_core", "v3_baseline_core"),
    ) == []


def test_v33_terminal_release_rolls_back_if_dependency_delete_fails(tmp_path):
    database, stream_id, partition_id = _fragmentation_v33_state(tmp_path)
    context = tmp_path / "context.tif"
    context.write_bytes(b"context")
    artifact_id = database.publish_fragmentation_v33_context(
        RUN_ID,
        stream_id,
        partition_id,
        context,
        byte_count=context.stat().st_size,
        sha256=hashlib.sha256(context.read_bytes()).hexdigest(),
    )
    baseline = tmp_path / "baseline.tif"
    baseline.write_bytes(b"baseline")
    baseline_id = database.publish_fragmentation_v33_baseline_core(
        RUN_ID,
        stream_id,
        partition_id,
        baseline,
        byte_count=baseline.stat().st_size,
        sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
    )
    probability = tmp_path / "probability.tif"
    probability.write_bytes(b"probability")
    database.publish_partition_artifact(
        RUN_ID,
        stream_id,
        partition_id,
        probability,
        byte_count=probability.stat().st_size,
        sha256=hashlib.sha256(probability.read_bytes()).hexdigest(),
    )
    leased = database.lease_next_fragmentation_v33(
        RUN_ID, "v33-worker", lease_seconds=120
    )
    assert leased is not None
    _publish_v33_outputs(database, tmp_path, stream_id, partition_id)
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            """CREATE TRIGGER reject_v33_dependency_release
               BEFORE DELETE ON artifact_dependencies
               BEGIN SELECT RAISE(ABORT, 'injected release failure'); END"""
        )

    with pytest.raises(sqlite3.IntegrityError, match="injected release failure"):
        database.complete_fragmentation_v33_job(
            leased["job_id"], leased["lease_token"]
        )

    assert database.get_job(leased["job_id"])["status"] == "running"
    assert database.get_artifact(artifact_id)["ref_count"] == 2
    assert database.get_artifact(baseline_id)["ref_count"] == 2


def test_v33_blocks_only_its_fusion_unit_jobs_until_ready(tmp_path):
    database, fusion_stream, partition_id = _fragmentation_v33_state(tmp_path)
    model_stream = "model:test"
    database.register_streams(
        RUN_ID,
        [{"stream_id": model_stream, "kind": "model", "model_id": "test"}],
    )
    database.insert_spatial_units(
        RUN_ID,
        [
            {
                "unit_id": "core_for_v33_gate",
                "unit_type": "Core",
                "owner_key": partition_id,
                "pixel_window": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
                "dependency_ids": [partition_id],
            }
        ],
    )
    database.insert_jobs(
        RUN_ID,
        [
            {
                "job_type": "unit_fit",
                "stream_id": fusion_stream,
                "unit_id": "core_for_v33_gate",
                "priority": 100,
            },
            {
                "job_type": "unit_fit",
                "stream_id": model_stream,
                "unit_id": "core_for_v33_gate",
                "priority": 1,
            },
        ],
    )
    for name, publisher, stream_id in (
        ("context", database.publish_fragmentation_v33_context, fusion_stream),
        ("baseline", database.publish_fragmentation_v33_baseline_core, fusion_stream),
    ):
        path = tmp_path / f"{name}.tif"
        path.write_bytes(name.encode())
        publisher(
            RUN_ID,
            stream_id,
            partition_id,
            path,
            byte_count=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    for stream_id in (fusion_stream, model_stream):
        path = tmp_path / f"{stream_id.replace(':', '_')}.tif"
        path.write_bytes(stream_id.encode())
        database.publish_partition_artifact(
            RUN_ID,
            stream_id,
            partition_id,
            path,
            byte_count=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    first = database.lease_next_job(RUN_ID, "unit-worker", job_types=("unit_fit",))
    assert first is not None
    assert first["stream_id"] == model_stream

    candidate = database.lease_next_fragmentation_v33(RUN_ID, "v33-worker")
    assert candidate is not None
    _publish_v33_outputs(database, tmp_path, fusion_stream, partition_id)
    assert database.complete_fragmentation_v33_job(
        candidate["job_id"], candidate["lease_token"]
    )

    # Owner staging does not expose an authoritative Core or unblock Fusion
    # geometry. Only the global finalize job can cross that publication barrier.
    assert database.lease_next_job(
        RUN_ID, "still-blocked", job_types=("unit_fit",)
    ) is None
    finalize = database.lease_next_fragmentation_v33(RUN_ID, "v33-finalize")
    assert finalize is not None
    assert finalize["unit_id"] == "fragmentation_v33_finalize"
    final_mask = tmp_path / "authoritative_mask.tif"
    final_audit = tmp_path / "authoritative_audit.json"
    final_report = tmp_path / "fragmentation_v33_report.json"
    final_mask.write_bytes(b"authoritative-mask")
    final_audit.write_bytes(b"authoritative-audit")
    final_report.write_bytes(b"authoritative-report")
    assert database.complete_fragmentation_v33_finalize(
        finalize["job_id"],
        finalize["lease_token"],
        [
            {
                "partition_id": partition_id,
                "mask_path": final_mask,
                "mask_byte_count": final_mask.stat().st_size,
                "mask_sha256": hashlib.sha256(final_mask.read_bytes()).hexdigest(),
                "audit_path": final_audit,
                "audit_byte_count": final_audit.stat().st_size,
                "audit_sha256": hashlib.sha256(final_audit.read_bytes()).hexdigest(),
            }
        ],
        report_path=final_report,
        report_byte_count=final_report.stat().st_size,
        report_sha256=hashlib.sha256(final_report.read_bytes()).hexdigest(),
    )
    assert len(database.cleanup_candidates(
        RUN_ID, kinds=("v33_staged_mask", "v33_staged_audit")
    )) == 2
    assert database.artifact_for_stream_unit(
        RUN_ID, fusion_stream, partition_id, "core_mask"
    )["path"] == str(final_mask.resolve())
    second = database.lease_next_job(RUN_ID, "fusion-worker", job_types=("unit_fit",))
    assert second is not None
    assert second["stream_id"] == fusion_stream


def test_v33_partition_leases_cap_at_four_and_failed_owner_recovers(tmp_path):
    """The PostgreSQL/SKIP-LOCKED path shares this four-lease SQL contract."""

    database = _database(tmp_path)
    stream_id = "fusion:lease-cap"
    partition_ids = [f"partition_{index:05d}" for index in range(5)]
    database.register_streams(
        RUN_ID, [{"stream_id": stream_id, "kind": "fusion", "profile_id": "test"}]
    )
    database.insert_work_packages(
        RUN_ID,
        [
            {"package_id": f"package_{index:05d}", "sequence_no": index, "status": "ready"}
            for index in range(len(partition_ids))
        ],
    )
    database.insert_partitions(
        RUN_ID,
        [
            {
                "partition_id": partition_id,
                "row": 0,
                "col": index,
                "core_window": {"x0": index, "y0": 0, "x1": index + 1, "y1": 1},
                "halo_window": {"x0": index, "y0": 0, "x1": index + 1, "y1": 1},
                "package_id": f"package_{index:05d}",
                "status": "ready",
            }
            for index, partition_id in enumerate(partition_ids)
        ],
    )
    database.insert_spatial_units(
        RUN_ID,
        [
            {
                "unit_id": f"fragmentation_v33_partition:{partition_id}",
                "unit_type": "FragmentationV33Partition",
                "owner_key": partition_id,
                "pixel_window": {"x0": index, "y0": 0, "x1": index + 1, "y1": 1},
                "dependency_ids": [partition_id],
            }
            for index, partition_id in enumerate(partition_ids)
        ],
    )
    database.insert_jobs(
        RUN_ID,
        [
            {
                "job_type": "fragmentation_v33",
                "stream_id": stream_id,
                "unit_id": f"fragmentation_v33_partition:{partition_id}",
                "max_attempts": 2,
            }
            for partition_id in partition_ids
        ],
    )
    for partition_id in partition_ids:
        for kind, publisher in (
            ("probability", database.publish_partition_artifact),
            ("context", database.publish_fragmentation_v33_context),
            ("baseline", database.publish_fragmentation_v33_baseline_core),
        ):
            path = tmp_path / f"{partition_id}_{kind}.bin"
            path.write_bytes(f"{partition_id}:{kind}".encode())
            publisher(
                RUN_ID,
                stream_id,
                partition_id,
                path,
                byte_count=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    leases = [
        database.lease_next_fragmentation_v33(
            RUN_ID, f"worker-{index}", max_running=99
        )
        for index in range(5)
    ]
    active = [item for item in leases if item is not None]
    assert len(active) == 4
    assert len({item["job_id"] for item in active}) == 4
    failed = active[0]
    assert database.finish_job(
        failed["job_id"], failed["lease_token"], status="failed", error="fixture failure"
    )
    assert database.requeue_failed_job(failed["job_id"])
    recovered = database.lease_next_fragmentation_v33(
        RUN_ID, "recovery-worker", max_running=99
    )
    assert recovered is not None
    assert recovered["job_id"] == failed["job_id"]
    assert recovered["lease_token"] != failed["lease_token"]


def test_fail_open_streams_preserves_ready_streams(tmp_path):
    database = _database(tmp_path)
    database.register_streams(
        RUN_ID,
        [
            {"stream_id": "model:ready", "kind": "model", "status": "ready"},
            {"stream_id": "model:running", "kind": "model", "status": "running"},
            {"stream_id": "fusion:test", "kind": "fusion", "status": "pending"},
        ],
    )

    assert database.fail_open_streams(RUN_ID, "Package exhausted retries") == 2
    rows = {row["stream_id"]: row for row in database.stream_rows(RUN_ID)}
    assert rows["model:ready"]["status"] == "ready"
    assert rows["model:ready"]["error"] == ""
    assert rows["model:running"]["status"] == "failed"
    assert rows["fusion:test"]["status"] == "failed"
    assert rows["model:running"]["error"] == "Package exhausted retries"


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

    persisted_partitions = database.partitions_for_run(RUN_ID)
    assert len(persisted_partitions) == plan["partition_count"]
    assert [
        (partition["row"], partition["col"])
        for partition in persisted_partitions
    ] == sorted(
        (partition["row"], partition["col"])
        for partition in persisted_partitions
    )
    assert persisted_partitions[0]["core_window"] == partitions[0]["core_window"]
    assert persisted_partitions[0]["halo_window"] == partitions[0]["halo_window"]

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
