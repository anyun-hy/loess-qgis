import json
import time

import psycopg2
import pytest

from labeling_tool.core.run_builder_v5 import create_v5_run
from labeling_tool.core.run_state_db import (
    RUN_DETAIL_ARCHIVE_KEY,
    RUN_DETAIL_TABLES,
    RunStateDB,
)


STREAM_ID = "model:a"


def _count(database, table, run_id):
    with database._connection() as connection:
        if table == "artifact_dependencies":
            return int(
                connection.execute(
                    """SELECT COUNT(*) FROM artifact_dependencies
                       WHERE job_id IN (
                         SELECT job_id FROM jobs WHERE run_id=%s
                       ) OR artifact_id IN (
                         SELECT artifact_id FROM artifacts WHERE run_id=%s
                       )""",
                    (run_id, run_id),
                ).fetchone()[0]
            )
        return int(
            connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE run_id=%s", (run_id,)
            ).fetchone()[0]
        )


def _seed_run(
    database,
    tmp_path,
    run_id,
    status,
    *,
    active_lease=False,
):
    database.create_run(
        run_id,
        "a" * 64,
        status=status,
        metadata={"run_spec": str(tmp_path / run_id / "run_spec.json")},
    )
    database.register_streams(
        run_id,
        [{"stream_id": STREAM_ID, "kind": "model", "model_id": "a"}],
    )
    database.upsert_stream_runtime_progress(
        run_id,
        STREAM_ID,
        stage="assembly",
        phase="write_raw",
        phase_name="write raw",
        phase_index=1,
        phase_total=2,
        status="failed",
        message="stream progress failed",
    )
    database.insert_work_packages(
        run_id,
        [{"package_id": "package_00000", "sequence_no": 0, "status": status}],
    )
    database.insert_partitions(
        run_id,
        [
            {
                "partition_id": "partition_00000_00000",
                "row": 0,
                "col": 0,
                "core_window": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
                "halo_window": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
                "package_id": "package_00000",
                "status": status,
            }
        ],
    )
    database.insert_spatial_units(
        run_id,
        [
            {
                "unit_id": "core_00000_00000",
                "unit_type": "core",
                "owner_key": "core_00000_00000",
                "pixel_window": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
                "dependency_ids": ["partition_00000_00000"],
                "status": status,
            }
        ],
    )
    database.insert_stream_units(
        run_id,
        [STREAM_ID],
        ["core_00000_00000"],
    )
    database.insert_tiles(
        run_id,
        [
            {
                "tile_id": "0_0",
                "row": 0,
                "col": 0,
                "width": 512,
                "height": 512,
                "pixel_window": {"x0": 0, "y0": 0, "x1": 512, "y1": 512},
                "bounds": {},
                "raster_path": str(tmp_path / run_id / "tile.tif"),
                "partition_id": "partition_00000_00000",
                "status": status,
            }
        ],
    )
    database.insert_jobs(
        run_id,
        [
            {
                "job_type": "unit_fit",
                "stream_id": STREAM_ID,
                "unit_id": "core_00000_00000",
                "status": "running" if active_lease else "failed",
            }
        ],
    )
    with database.transaction() as connection:
        job_id = int(
            connection.execute(
                "SELECT job_id FROM jobs WHERE run_id=%s", (run_id,)
            ).fetchone()[0]
        )
        connection.execute(
            """UPDATE jobs SET error=%s, lease_token=%s, lease_expires=%s
               WHERE job_id=%s""",
            (
                "fixture unit failure",
                "active-token" if active_lease else "",
                time.time() + 3600 if active_lease else None,
                job_id,
            ),
        )
    report_path = tmp_path / run_id / "unit_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("{}", encoding="utf-8")
    artifact_id = database.register_artifact(
        run_id,
        "unit_boundary_report",
        report_path,
        stream_id=STREAM_ID,
        unit_id="core_00000_00000",
    )
    assert database.mark_artifact_ready(
        artifact_id,
        byte_count=report_path.stat().st_size,
        sha256="b" * 64,
    )
    assert database.add_artifact_dependency(job_id, artifact_id)
    database.upsert_unit_report_summary(
        run_id,
        STREAM_ID,
        "core_00000_00000",
        {"status": "failed", "diagnostics": []},
    )
    database.register_object_parts(
        run_id,
        STREAM_ID,
        [
            {
                "part_id": "part-a",
                "class_code": 11,
                "unit_id": "core_00000_00000",
            },
            {
                "part_id": "part-b",
                "class_code": 11,
                "unit_id": "core_00000_00000",
            },
        ],
    )
    assert database.add_object_link(
        run_id, STREAM_ID, "part-a", "part-b", 11
    )
    database.append_event(
        run_id,
        "fixture_failure",
        level="error",
        message="fixture event failure",
    )


@pytest.mark.parametrize("status", ["failed", "stopped"])
def test_archive_incomplete_run_details_preserves_tombstone_and_deletes_graph(
    tmp_path, status, postgres_database
):
    database = postgres_database
    run_id = f"20260901_100000_{status}"
    _seed_run(database, tmp_path, run_id, status)

    report = database.archive_incomplete_run_details(
        protected_run_id="20260901_110000_current"
    )

    assert report["archived_run_ids"] == [run_id]
    run = database.get_run(run_id)
    assert run["status"] == f"archived_{status}"
    metadata = json.loads(run["metadata_json"])
    archive = metadata[RUN_DETAIL_ARCHIVE_KEY]
    assert archive["non_resumable"] is True
    assert archive["original_status"] == status
    assert archive["run_spec_sha256"] == "a" * 64
    assert archive["detail_counts"]["jobs"] == 1
    assert archive["detail_counts"]["artifacts"] == 1
    assert archive["detail_counts"]["object_nodes"] == 2
    assert archive["detail_counts"]["object_links"] == 1
    assert archive["errors"][0]["message"] in {
        "fixture event failure",
        "fixture unit failure",
        "stream progress failed",
    }
    for table in RUN_DETAIL_TABLES:
        expected = 1 if table == "events" else 0
        assert _count(database, table, run_id) == expected, table
    with database._connection() as connection:
        event = connection.execute(
            "SELECT event_type FROM events WHERE run_id=%s", (run_id,)
        ).fetchone()
        assert event["event_type"] == "run_details_archived"
        invalid_constraints = connection.execute(
            """SELECT COUNT(*) FROM pg_constraint
               WHERE connamespace=current_schema()::regnamespace
                 AND contype='f' AND NOT convalidated"""
        ).fetchone()[0]
        assert int(invalid_constraints) == 0
    assert not database.set_run_status(run_id, "running")


def test_archive_protects_ready_running_active_lease_and_current_run(
    tmp_path, postgres_database
):
    database = postgres_database
    for run_id, status, active in (
        ("20260901_100001_ready", "ready", False),
        ("20260901_100002_running", "running", False),
        ("20260901_100003_active", "failed", True),
        ("20260901_100004_current", "stopped", False),
        ("20260901_100005_idle", "failed", False),
        ("20260901_100010_resetting", "resetting", False),
    ):
        _seed_run(database, tmp_path, run_id, status, active_lease=active)

    report = database.archive_incomplete_run_details(
        protected_run_id="20260901_100004_current"
    )

    assert report["archived_run_ids"] == ["20260901_100005_idle"]
    assert report["skipped_active_run_ids"] == ["20260901_100003_active"]
    assert report["skipped_active_run_count"] == 1
    for run_id, status in (
        ("20260901_100001_ready", "ready"),
        ("20260901_100002_running", "running"),
        ("20260901_100003_active", "failed"),
        ("20260901_100004_current", "stopped"),
        ("20260901_100010_resetting", "resetting"),
    ):
        assert database.get_run(run_id)["status"] == status
        assert _count(database, "jobs", run_id) == 1


def test_archive_tombstone_whitelists_metadata_and_bounds_utf8_errors(
    tmp_path, postgres_database
):
    database = postgres_database
    run_id = "20260901_100011_bounded"
    _seed_run(database, tmp_path, run_id, "failed")
    database.update_run_metadata(
        run_id,
        {
            "tile_count": 4,
            "unrelated_large_value": "x" * 100_000,
        },
    )
    long_message = "错" * 2000
    with database.transaction() as connection:
        connection.execute(
            "UPDATE jobs SET error=%s WHERE run_id=%s",
            (long_message, run_id),
        )
    for index in range(10):
        database.append_event(
            run_id,
            f"extra_{index}",
            level="error",
            message=f"extra error {index}",
        )

    database.archive_incomplete_run_details(
        protected_run_id="20260901_110000_current"
    )

    metadata = json.loads(database.get_run(run_id)["metadata_json"])
    assert metadata["tile_count"] == 4
    assert "unrelated_large_value" not in metadata
    archive = metadata[RUN_DETAIL_ARCHIVE_KEY]
    assert archive["error_record_count"] == 12
    assert archive["errors_truncated"] is True
    assert len(archive["errors"]) == 8
    assert all(
        len(item["message"].encode("utf-8")) <= 2000
        for item in archive["errors"]
    )


def test_archive_is_cross_run_isolated_and_idempotent(
    tmp_path, postgres_database
):
    database = postgres_database
    old_run = "20260901_100006_old"
    protected = "20260901_100007_current"
    _seed_run(database, tmp_path, old_run, "failed")
    _seed_run(database, tmp_path, protected, "stopped")

    first = database.archive_incomplete_run_details(protected_run_id=protected)
    first_metadata = database.get_run(old_run)["metadata_json"]
    second = database.archive_incomplete_run_details(protected_run_id=protected)

    assert first["archived_run_ids"] == [old_run]
    assert second["archived_run_ids"] == []
    assert second["deleted_detail_counts"] == {
        table: 0 for table in RUN_DETAIL_TABLES
    }
    assert database.get_run(old_run)["metadata_json"] == first_metadata
    assert database.get_run(protected)["status"] == "stopped"
    assert _count(database, "jobs", protected) == 1


def test_archive_report_bounds_run_id_lists(tmp_path, postgres_database):
    database = postgres_database
    for index in range(51):
        database.create_run(
            f"20260901_09{index:04d}_old",
            "a" * 64,
            status="failed",
        )

    report = database.archive_incomplete_run_details(
        protected_run_id="20260901_110000_current"
    )

    assert report["archived_run_count"] == 51
    assert len(report["archived_run_ids"]) == 50
    assert report["archived_run_ids_truncated"] is True


def test_archive_rolls_back_everything_on_late_delete_failure(
    tmp_path, postgres_database
):
    database = postgres_database
    run_id = "20260901_100008_rollback"
    _seed_run(database, tmp_path, run_id, "failed")
    before = {table: _count(database, table, run_id) for table in RUN_DETAIL_TABLES}
    with database._connection() as connection:
        connection.execute(
            f"""CREATE OR REPLACE FUNCTION fail_archive_delete_fn()
                RETURNS trigger LANGUAGE plpgsql AS $function$
                BEGIN
                  IF OLD.run_id='{run_id}' THEN
                    RAISE EXCEPTION 'injected archive failure'
                      USING ERRCODE='23514';
                  END IF;
                  RETURN OLD;
                END
                $function$"""
        )
        connection.execute(
            """CREATE TRIGGER fail_archive_delete
               BEFORE DELETE ON artifacts FOR EACH ROW
               EXECUTE FUNCTION fail_archive_delete_fn()"""
        )

    with pytest.raises(psycopg2.IntegrityError, match="injected archive failure"):
        database.archive_incomplete_run_details(
            protected_run_id="20260901_110000_current"
        )

    assert database.get_run(run_id)["status"] == "failed"
    assert {
        table: _count(database, table, run_id) for table in RUN_DETAIL_TABLES
    } == before
    metadata = json.loads(database.get_run(run_id)["metadata_json"])
    assert RUN_DETAIL_ARCHIVE_KEY not in metadata


def _create_minimal_v5(tmp_path, database_location, run_id):
    raster = tmp_path / f"{run_id}.tif"
    model = tmp_path / f"{run_id}.pt"
    raster.write_bytes(b"raster")
    model.write_bytes(b"model")
    return create_v5_run(
        state_database=database_location,
        output_root=tmp_path / "output",
        run_id=run_id,
        raster={
            "path": raster,
            "crs": "EPSG:4490",
            "transform": [1, 0, 0, 0, -1, 0],
            "nodata": None,
        },
        requested_extent={"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
        processing_extent={"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
        tile_rows=2,
        tile_cols=2,
        tiles=[
            {
                "row": row,
                "col": col,
                "path": str(raster),
                "sha256": "f" * 64,
                "pixel_window": {
                    "x0": col * 320,
                    "y0": row * 320,
                    "x1": col * 320 + 512,
                    "y1": row * 320 + 512,
                },
            }
            for row in range(2)
            for col in range(2)
        ],
        models=[
            {
                "model_id": "fixture",
                "artifact_path": str(model),
                "sha256": "e" * 64,
            }
        ],
        effective_device="cpu",
        overlap=192,
        scaling={
            "partition_tile_rows": 2,
            "partition_tile_cols": 2,
            "partition_halo_px": 256,
            "seam_band_px": 64,
            "max_job_retries": 2,
        },
        boundary_fitting={
            "enabled": True,
            "mode": "divider_cubic_bspline_adaptive_v2",
        },
        storage_report={
            "package_tile_limit": 4,
            "working_bytes_per_tile": 4096,
            "status": "passed",
        },
    )


def test_create_v5_run_archives_old_incomplete_details_after_graph_creation(
    tmp_path, postgres_database
):
    database = postgres_database
    _seed_run(
        database,
        tmp_path,
        "20260901_100009_old",
        "failed",
    )

    spec, _spec_path, _location = _create_minimal_v5(
        tmp_path,
        database.location,
        "20260901_110000_0a0001",
    )

    assert spec["run_id"] == "20260901_110000_0a0001"
    assert database.get_run("20260901_100009_old")["status"] == "archived_failed"
    new_run = database.get_run(spec["run_id"])
    assert new_run["status"] == "planned"
    cleanup = json.loads(new_run["metadata_json"])["incomplete_run_cleanup"]
    assert cleanup["archived_run_ids"] == ["20260901_100009_old"]
    assert _count(database, "jobs", spec["run_id"]) > 0


def test_create_v5_run_warns_but_succeeds_when_cleanup_fails(
    tmp_path, monkeypatch, caplog, postgres_database
):
    def fail_cleanup(*_args, **_kwargs):
        raise RuntimeError("injected cleanup failure")

    monkeypatch.setattr(
        RunStateDB,
        "archive_incomplete_run_details",
        fail_cleanup,
    )
    spec, _spec_path, _location = _create_minimal_v5(
        tmp_path,
        postgres_database.location,
        "20260901_110001_0a0002",
    )

    database = RunStateDB(
        postgres_database.location,
        postgres_schema=postgres_database.postgres_schema,
    )
    run = database.get_run(spec["run_id"])
    cleanup = json.loads(run["metadata_json"])["incomplete_run_cleanup"]
    assert run["status"] == "planned"
    assert cleanup["status"] == "warning"
    assert cleanup["error"] == "injected cleanup failure"
    assert "[incomplete-run-cleanup-warning]" in caplog.text
