"""Opt-in PostgreSQL 18 integration and concurrent leasing tests."""

from __future__ import annotations

import os
import secrets
import json
import time
from concurrent.futures import ThreadPoolExecutor

from labeling_tool.core.postgres_state import (
    DEFAULT_POSTGRES_DSN,
    _resolve_postgres_dsn,
)
from labeling_tool.core.run_builder_v5 import create_v5_run
from labeling_tool.core.run_builder_v5 import _fragmentation_v33_units
from labeling_tool.core.run_state_db import RunStateDB
from labeling_tool.core.run_state_db import RUN_DETAIL_ARCHIVE_KEY


POSTGRES_DSN = str(
    os.environ.get("LOESS_TEST_POSTGRES_DSN") or DEFAULT_POSTGRES_DSN
).strip()


def _drop_schema(dsn: str, schema: str) -> None:
    import psycopg2

    connection = psycopg2.connect(_resolve_postgres_dsn(dsn))
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    finally:
        connection.close()


def test_postgres_incomplete_run_archive_preserves_summary_and_protections():
    schema = f"loess_test_{secrets.token_hex(6)}"
    database = RunStateDB(POSTGRES_DSN, postgres_schema=schema)
    failed_run = "20260901_120000_pgfail"
    active_run = "20260901_120001_pgactive"
    ready_run = "20260901_120002_pgready"
    try:
        database.initialize()
        for run_id, status in (
            (failed_run, "failed"),
            (active_run, "failed"),
            (ready_run, "ready"),
        ):
            database.create_run(
                run_id,
                "a" * 64,
                status=status,
                metadata={"run_spec": f"/tmp/{run_id}/run_spec.json"},
            )
            database.insert_jobs(
                run_id,
                [
                    {
                        "job_type": "unit_fit",
                        "status": "running" if run_id == active_run else "failed",
                    }
                ],
            )
            database.append_event(
                run_id,
                "fixture",
                level="error",
                message=f"failure for {run_id}",
            )
        database.register_object_parts(
            failed_run,
            "model:a",
            [{"part_id": "part-a", "class_code": 11, "unit_id": "core-a"}],
        )
        with database.transaction() as connection:
            connection.execute(
                """UPDATE jobs SET lease_token=%s, lease_expires=%s
                   WHERE run_id=%s""",
                ("active-token", time.time() + 3600, active_run),
            )

        report = database.archive_incomplete_run_details(
            protected_run_id="20260901_130000_current"
        )

        assert report["archived_run_ids"] == [failed_run]
        assert report["skipped_active_run_ids"] == [active_run]
        archived = database.get_run(failed_run)
        assert archived["status"] == "archived_failed"
        archive = json.loads(archived["metadata_json"])[RUN_DETAIL_ARCHIVE_KEY]
        assert archive["detail_counts"]["jobs"] == 1
        assert archive["detail_counts"]["object_nodes"] == 1
        assert archive["errors"][0]["message"] == f"failure for {failed_run}"
        assert database.job_counts(failed_run) == {}
        assert database.get_run(active_run)["status"] == "failed"
        assert database.job_counts(active_run) == {"running": 1}
        assert database.get_run(ready_run)["status"] == "ready"
        assert database.job_counts(ready_run) == {"failed": 1}
        assert database.archive_incomplete_run_details(
            protected_run_id="20260901_130000_current"
        )["archived_run_ids"] == []
    finally:
        _drop_schema(POSTGRES_DSN, schema)


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
                    """SELECT job_id FROM jobs WHERE run_id=%s
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


def test_postgres_v33_partition_leases_enforce_four_worker_limit_and_recover(
    tmp_path,
):
    schema = f"loess_test_{secrets.token_hex(6)}"
    run_id = "20260829_000002_v33pg"
    stream_id = "fusion:fixture"
    database = RunStateDB(POSTGRES_DSN, postgres_schema=schema)
    try:
        database.initialize()
        database.create_run(run_id, "b" * 64, status="running")
        database.register_streams(
            run_id,
            [{"stream_id": stream_id, "kind": "fusion", "profile_id": "fixture"}],
        )
        packages = [
            {
                "package_id": f"package_{index:05d}",
                "sequence_no": index,
                "status": "ready",
            }
            for index in range(6)
        ]
        database.insert_work_packages(run_id, packages)
        partitions = [
            {
                "partition_id": f"partition_{index:05d}_00000",
                "row": index,
                "col": 0,
                "core_window": {"x0": 0, "y0": index, "x1": 1, "y1": index + 1},
                "halo_window": {"x0": 0, "y0": index, "x1": 1, "y1": index + 1},
                "package_id": packages[index]["package_id"],
                "status": "ready",
            }
            for index in range(6)
        ]
        database.insert_partitions(run_id, partitions)
        units = _fragmentation_v33_units(partitions)
        database.insert_spatial_units(run_id, units)
        database.insert_jobs(
            run_id,
            (
                {
                    "job_type": "fragmentation_v33",
                    "stream_id": stream_id,
                    "unit_id": unit["unit_id"],
                    "priority": 50 if unit["unit_type"] == "FragmentationV33Partition" else 40,
                    "max_attempts": 3,
                }
                for unit in units
            ),
        )
        for index, partition in enumerate(partitions):
            partition_id = partition["partition_id"]
            probability = tmp_path / f"{partition_id}_probability.tif"
            context = tmp_path / f"{partition_id}_context.tif"
            baseline = tmp_path / f"{partition_id}_baseline.tif"
            probability.write_bytes(b"probability")
            context.write_bytes(b"context")
            baseline.write_bytes(b"baseline")
            database.publish_partition_artifact(
                run_id,
                stream_id,
                partition_id,
                probability,
                byte_count=probability.stat().st_size,
                sha256=f"{index + 1:x}" * 64,
            )
            database.publish_fragmentation_v33_context(
                run_id,
                stream_id,
                partition_id,
                context,
                byte_count=context.stat().st_size,
                sha256=f"{index + 7:x}" * 64,
            )
            database.publish_fragmentation_v33_baseline_core(
                run_id,
                stream_id,
                partition_id,
                baseline,
                byte_count=baseline.stat().st_size,
                sha256=f"{index + 13:x}"[-1] * 64,
            )

        def lease(worker_no: int):
            return database.lease_next_fragmentation_v33(
                run_id,
                f"v33-pg-{worker_no}",
                lease_seconds=120,
                max_running=4,
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            leases = [item for item in executor.map(lease, range(8)) if item]
        assert len(leases) == 4
        assert len({int(item["job_id"]) for item in leases}) == 4
        assert database.job_counts(run_id, job_type="fragmentation_v33")["running"] == 4

        interrupted = leases[0]
        assert database.interrupt_job(
            int(interrupted["job_id"]), str(interrupted["lease_token"])
        )
        recovered = lease(99)
        assert recovered is not None
        assert database.job_counts(run_id, job_type="fragmentation_v33")["running"] == 4
        owner_unit = database.get_spatial_unit(run_id, str(recovered["unit_id"]))
        owner_id = str(owner_unit["owner_key"])
        staged_mask = tmp_path / f"{owner_id}_staged_mask.tif"
        staged_audit = tmp_path / f"{owner_id}_staged_audit.json"
        staged_mask.write_bytes(b"staged-mask")
        staged_audit.write_bytes(b"staged-audit")
        staged_ids = database.publish_fragmentation_v33_output_pair(
            run_id,
            stream_id,
            owner_id,
            mask_path=staged_mask,
            mask_byte_count=staged_mask.stat().st_size,
            mask_sha256="a" * 64,
            audit_path=staged_audit,
            audit_byte_count=staged_audit.stat().st_size,
            audit_sha256="b" * 64,
            production=None,
        )
        assert [database.get_artifact(item)["ref_count"] for item in staged_ids] == [1, 1]
        assert database.cleanup_candidates(
            run_id, kinds=("v33_staged_mask", "v33_staged_audit")
        ) == []
        assert database.complete_fragmentation_v33_job(
            int(recovered["job_id"]), str(recovered["lease_token"])
        )
    finally:
        _drop_schema(POSTGRES_DSN, schema)
