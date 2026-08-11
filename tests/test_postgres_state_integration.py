"""Opt-in PostgreSQL 18 integration and concurrent leasing tests."""

from __future__ import annotations

import os
import secrets
from concurrent.futures import ThreadPoolExecutor

import pytest

from labeling_tool.core.run_builder_v5 import create_v5_run
from labeling_tool.core.run_state_db import RunStateDB


POSTGRES_DSN = os.environ.get("LOESS_TEST_POSTGRES_DSN", "").strip()
pytestmark = pytest.mark.skipif(
    not POSTGRES_DSN,
    reason="LOESS_TEST_POSTGRES_DSN is required for PostgreSQL integration",
)


def _drop_schema(dsn: str, schema: str) -> None:
    import psycopg2

    connection = psycopg2.connect(dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    finally:
        connection.close()


def test_postgres_schema_artifacts_and_true_concurrent_job_writes(tmp_path):
    schema = f"loess_test_{secrets.token_hex(6)}"
    run_id = "20260811_000000_pgtest"
    database = RunStateDB(POSTGRES_DSN, postgres_schema=schema)
    try:
        database.initialize()
        health = database.pragmas()
        assert health["backend"] == "postgresql"
        assert health["schema"] == schema
        assert health["schema_version"] == 2

        database.create_run(run_id, "a" * 64, status="running")
        job_total = 400
        assert database.insert_jobs(
            run_id,
            (
                {
                    "job_type": "unit_fit",
                    "unit_id": f"unit_{index:05d}",
                    "priority": index % 7,
                    "max_attempts": 3,
                }
                for index in range(job_total)
            ),
        ) == job_total

        leased_ids: list[int] = []

        def consume(worker_no: int) -> list[int]:
            completed = []
            worker_id = f"postgres-worker-{worker_no}"
            while True:
                job = database.lease_next_job(
                    run_id,
                    worker_id,
                    job_types=("unit_fit",),
                    lease_seconds=120,
                )
                if job is None:
                    break
                assert database.finish_job(
                    int(job["job_id"]),
                    str(job["lease_token"]),
                    status="ready",
                )
                completed.append(int(job["job_id"]))
            return completed

        with ThreadPoolExecutor(max_workers=20) as executor:
            for completed in executor.map(consume, range(20)):
                leased_ids.extend(completed)

        assert len(leased_ids) == job_total
        assert len(set(leased_ids)) == job_total
        assert database.job_counts(run_id) == {"ready": job_total}

        database.insert_jobs(
            run_id,
            ({"job_type": "artifact_consumer", "max_attempts": 1},),
        )
        consumer = database.lease_next_job(
            run_id,
            "artifact-consumer",
            job_types=("artifact_consumer",),
            lease_seconds=120,
        )
        assert consumer is not None
        artifact_path = tmp_path / "artifact.bin"
        artifact_path.write_bytes(b"postgres-artifact")
        artifact_id = database.register_artifact(
            run_id,
            "integration",
            artifact_path,
        )
        assert database.mark_artifact_ready(
            artifact_id,
            byte_count=artifact_path.stat().st_size,
            sha256="b" * 64,
        )
        assert database.add_artifact_dependency(
            int(consumer["job_id"]), artifact_id
        )
        assert int(database.get_artifact(artifact_id)["ref_count"]) == 1
        assert database.release_job_artifacts(int(consumer["job_id"])) == 1
        assert int(database.get_artifact(artifact_id)["ref_count"]) == 0
        assert database.append_event(run_id, "postgres_integration") > 0
        snapshot = database.monitor_snapshot(run_id)
        assert snapshot["run"]["run_id"] == run_id

        # Many jobs can release the same Artifact set concurrently.  The
        # implementation locks Artifact rows in ascending ID order, retaining
        # concurrency without cross-job deadlocks or refcount drift.
        shared_artifacts = []
        for index in range(4):
            path = tmp_path / f"shared_{index}.bin"
            path.write_bytes(f"shared-{index}".encode())
            shared_id = database.register_artifact(run_id, "shared", path)
            assert database.mark_artifact_ready(
                shared_id,
                byte_count=path.stat().st_size,
                sha256=f"{index + 1:x}" * 64,
            )
            shared_artifacts.append(shared_id)

        release_total = 80
        assert database.insert_jobs(
            run_id,
            (
                {"job_type": "release_test", "max_attempts": 1}
                for _ in range(release_total)
            ),
        ) == release_total
        with database._connection() as connection:
            release_job_ids = [
                int(row["job_id"])
                for row in connection.execute(
                    """SELECT job_id FROM jobs WHERE run_id=?
                       AND job_type='release_test' ORDER BY job_id""",
                    (run_id,),
                ).fetchall()
            ]
        for release_job_id in release_job_ids:
            for shared_id in shared_artifacts:
                assert database.add_artifact_dependency(
                    release_job_id, shared_id
                )
        assert all(
            int(database.get_artifact(shared_id)["ref_count"]) == release_total
            for shared_id in shared_artifacts
        )
        with ThreadPoolExecutor(max_workers=20) as executor:
            released = list(
                executor.map(database.release_job_artifacts, release_job_ids)
            )
        assert released == [len(shared_artifacts)] * release_total
        assert all(
            int(database.get_artifact(shared_id)["ref_count"]) == 0
            for shared_id in shared_artifacts
        )
    finally:
        _drop_schema(POSTGRES_DSN, schema)


def test_new_v5_run_builds_its_complete_control_graph_in_postgres(
    tmp_path,
    monkeypatch,
):
    schema = f"loess_test_{secrets.token_hex(6)}"
    monkeypatch.setenv("LOESS_STATE_DB_SCHEMA", schema)
    raster = tmp_path / "source.tif"
    model = tmp_path / "model.pt"
    raster.write_bytes(b"raster-fixture")
    model.write_bytes(b"model-fixture")
    try:
        spec, spec_path, database_location = create_v5_run(
            state_database=POSTGRES_DSN,
            output_root=tmp_path / "output",
            run_id="20260811_000001_pgbuild",
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
            tiles=(
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
            ),
            models=(
                {
                    "model_id": "fixture_model",
                    "artifact_path": str(model),
                    "sha256": "e" * 64,
                    "version": "fixture-v1",
                },
            ),
            effective_device="cuda:0",
            overlap=192,
            scaling={
                "partition_tile_rows": 2,
                "partition_tile_cols": 2,
                "partition_halo_px": 192,
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
        assert spec_path.is_file()
        assert spec["state_backend"] == "postgresql"
        assert spec["state_db"] == POSTGRES_DSN
        assert database_location == POSTGRES_DSN

        database = RunStateDB(POSTGRES_DSN, postgres_schema=schema)
        assert database.count_tiles(spec["run_id"]) == 4
        assert len(database.partitions_for_run(spec["run_id"])) == 1
        assert database.job_counts(spec["run_id"], job_type="work_package")
        assert database.job_counts(spec["run_id"], job_type="unit_fit")
        package = database.lease_next_work_package(
            spec["run_id"],
            "postgres-package-worker",
            max_open_frontier_units=64,
            lease_seconds=120,
        )
        assert package is not None
        partition = database.partitions_for_run(spec["run_id"])[0]
        probability = tmp_path / "partition_probability.tif"
        probability.write_bytes(b"probability")
        artifact_id = database.publish_partition_artifact(
            spec["run_id"],
            "model:fixture_model",
            str(partition["partition_id"]),
            probability,
            byte_count=probability.stat().st_size,
            sha256="c" * 64,
        )
        assert int(database.get_artifact(artifact_id)["ref_count"]) > 0
        assert database.complete_work_package_job(
            spec["run_id"],
            str(package["package_id"]),
            int(package["job_id"]),
            str(package["lease_token"]),
        )
        unit = database.lease_next_job(
            spec["run_id"],
            "postgres-unit-worker",
            job_types=("unit_fit",),
            lease_seconds=120,
        )
        assert unit is not None
        assert database.finish_job(
            int(unit["job_id"]),
            str(unit["lease_token"]),
            status="ready",
        )
        assert database.release_job_artifacts(int(unit["job_id"])) == 1
    finally:
        _drop_schema(POSTGRES_DSN, schema)
