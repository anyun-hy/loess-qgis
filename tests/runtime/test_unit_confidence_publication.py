"""Publication ordering with real PostgreSQL state and small GeoTIFF outputs."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from psycopg2.errors import LockNotAvailable

from labeling_tool.shared.state.postgres_state import PostgresConnection
from loess_runtime.geometry import unit_confidence as runtime


@pytest.fixture
def confidence_run(tmp_path, postgres_database, monkeypatch):
    database = postgres_database
    run_id, stream_id, unit_id = "confidence_order", "model:test", "core_0_0"
    window = {"x0": 0, "y0": 0, "x1": 4, "y1": 4}
    database.run_streams.create_run(run_id, "a" * 64)
    database.run_streams.register_streams(
        run_id, [{"stream_id": stream_id, "kind": "model", "model_id": "test"}]
    )
    database.control_graph.insert_work_packages(
        run_id, [{"package_id": "package_0", "sequence_no": 0, "status": "ready"}]
    )
    database.control_graph.insert_partitions(
        run_id,
        [
            {
                "partition_id": "partition_0",
                "row": 0,
                "col": 0,
                "core_window": window,
                "halo_window": window,
                "package_id": "package_0",
                "status": "ready",
            }
        ],
    )
    database.control_graph.insert_spatial_units(
        run_id,
        [
            {
                "unit_id": unit_id,
                "unit_type": "Core",
                "owner_key": "partition_0",
                "pixel_window": window,
                "dependency_ids": ["partition_0"],
            }
        ],
    )
    database.jobs.insert_jobs(
        run_id,
        [
            {"job_type": kind, "stream_id": stream_id, "unit_id": unit_id}
            for kind in ("unit_confidence", "unit_fit")
        ],
    )
    probability = tmp_path / "probability.tif"
    probability.write_bytes(b"immutable probability fixture")
    database.artifacts.publish_partition_artifact(
        run_id,
        stream_id,
        "partition_0",
        probability,
        byte_count=probability.stat().st_size,
        sha256=hashlib.sha256(probability.read_bytes()).hexdigest(),
    )
    spec = {
        "schema_version": 2,
        "run_id": run_id,
        "run_dir": str(tmp_path),
        "state_backend": "postgresql",
        "state_db": database.session.location,
        "state_schema": database.session.schema,
        "storage_preflight": {"v33_storage_mode": "streamed_unit_confidence_v1"},
        "raster": {"transform": [1, 0, 0, 0, -1, 4], "crs": "EPSG:3857"},
    }
    spec_path = tmp_path / "run_spec.json"
    spec_path.write_text(json.dumps(spec))
    # This test isolates publication; input decoding and storage admission have
    # their own tests. Both attempts receive exactly the same frozen values.
    probabilities = np.full((14, 4, 4), 1 / 14, dtype=np.float32)
    monkeypatch.setattr(
        runtime,
        "read_unit_probabilities",
        lambda *a: (probabilities.copy(), np.ones((4, 4), dtype=bool)),
    )
    monkeypatch.setattr(runtime, "create_run_storage_guard", lambda *a, **kw: None)
    monkeypatch.setattr(runtime, "run_state_from_spec", lambda spec: database)

    def lease():
        job = database.jobs.lease_next_job(
            run_id, "test-worker", job_types=("unit_confidence",), lease_seconds=120
        )
        assert job is not None
        return job

    def run(job):
        return runtime.run_unit_confidence(
            spec_path,
            stream_id,
            unit_id,
            job_id=job["job_id"],
            lease_token=job["lease_token"],
        )

    return database, run_id, stream_id, unit_id, lease, run


@pytest.mark.parametrize("consumer_already_cleaned", [False, True])
def test_rejected_attempt_cannot_replace_or_restore_committed_confidence(
    confidence_run, monkeypatch, consumer_already_cleaned
):
    database, run_id, stream_id, unit_id, lease, run = confidence_run
    original = lease()
    write = runtime.write_atomic_partition_raster
    entered = False
    saved = {}

    def complete_replacement_before_old_write(path, *args, **kwargs):
        nonlocal entered
        if not entered:
            entered = True
            assert database.jobs.interrupt_job(
                original["job_id"], original["lease_token"]
            )
            saved["report"] = run(lease())
            committed = Path(saved["report"]["path"])
            saved["inode"] = committed.stat().st_ino
            saved["bytes"] = committed.read_bytes()
            artifact = database.artifacts.artifact_for_stream_unit(
                run_id, stream_id, unit_id, "unit_confidence"
            )
            saved["artifact_id"] = artifact["artifact_id"]
            assert artifact["ref_count"] == 1
            assert (
                database.artifacts.claim_artifact_cleanup(artifact["artifact_id"])
                is None
            )
            if consumer_already_cleaned:
                fit = database.jobs.lease_next_job(
                    run_id, "consumer", job_types=("unit_fit",), lease_seconds=120
                )
                assert fit is not None
                with database.unit_attempt_commit(
                    fit["job_id"], fit["lease_token"]
                ) as pub:
                    assert pub.jobs.finish_job(
                        fit["job_id"], fit["lease_token"], status="ready"
                    )
                    assert pub.artifacts.release_job_artifacts(fit["job_id"]) == 1
                assert database.artifacts.claim_artifact_cleanup(
                    artifact["artifact_id"]
                )
                committed.unlink()
                assert database.artifacts.finish_artifact_cleanup(
                    artifact["artifact_id"], success=True
                )
        return write(path, *args, **kwargs)

    monkeypatch.setattr(
        runtime, "write_atomic_partition_raster", complete_replacement_before_old_write
    )
    with pytest.raises(runtime.UnitConfidenceError, match="lease expired"):
        run(original)

    committed = Path(saved["report"]["path"])
    if consumer_already_cleaned:
        assert not committed.exists()
        assert not list(committed.parent.glob("*.tif"))
        assert (
            database.artifacts.get_artifact(saved["artifact_id"])["status"] == "cleaned"
        )
    else:
        assert committed.stat().st_ino == saved["inode"]
        assert committed.read_bytes() == saved["bytes"]
        assert list(committed.parent.glob("*.tif")) == [committed]
        with rasterio.open(committed) as source:
            assert np.all(source.read(1) == np.float32(1 / 14))
    assert database.jobs.get_job(original["job_id"])["status"] == "ready"


def test_file_failure_only_removes_unpublished_attempt(
    confidence_run, monkeypatch, tmp_path
):
    database, _run_id, _stream_id, _unit_id, lease, run = confidence_run
    job = lease()
    write = runtime.write_atomic_partition_raster

    def failing_write(*a, **kw):
        write(*a, **kw)
        raise OSError("injected post-write failure")

    monkeypatch.setattr(runtime, "write_atomic_partition_raster", failing_write)
    with pytest.raises(OSError, match="injected post-write failure"):
        run(job)
    assert not list((tmp_path / "tmp/unit_confidence").rglob("*.tif"))
    assert database.jobs.get_job(job["job_id"])["status"] == "failed"


def test_commit_response_error_does_not_delete_published_file(
    confidence_run, monkeypatch
):
    database, run_id, stream_id, unit_id, lease, run = confidence_run
    publish = database.artifacts.complete_unit_confidence_job

    def committed_but_response_lost(*a, **kw):
        assert publish(*a, **kw)
        raise ConnectionError("injected commit response loss")

    monkeypatch.setattr(
        database.artifacts, "complete_unit_confidence_job", committed_but_response_lost
    )
    with pytest.raises(ConnectionError, match="commit response loss"):
        run(lease())
    artifact = database.artifacts.artifact_for_stream_unit(
        run_id, stream_id, unit_id, "unit_confidence"
    )
    assert artifact["ref_count"] == 1
    assert Path(artifact["path"]).is_file()
    assert (
        hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest()
        == artifact["sha256"]
    )


def test_confidence_release_locks_all_shared_inputs_before_deleting_dependencies(
    confidence_run, monkeypatch, tmp_path
):
    database, run_id, stream_id, _unit_id, lease, run = confidence_run
    job = lease()
    extra = database.artifacts.register_artifact(
        run_id,
        "partition_probability",
        tmp_path / "other-probability.tif",
        stream_id=stream_id,
        unit_id="partition_1",
    )
    assert database.artifacts.mark_artifact_ready(extra, byte_count=1, sha256="b" * 64)
    assert database.artifacts.add_artifact_dependency(job["job_id"], extra)
    with database.session.connection() as connection:
        inputs = [
            int(row["artifact_id"])
            for row in connection.execute(
                "SELECT artifact_id FROM artifact_dependencies WHERE job_id=%s",
                (job["job_id"],),
            ).fetchall()
        ]
    assert len(inputs) == 2
    execute = PostgresConnection.execute
    checked = []

    def before_delete(connection, statement, parameters=None):
        if " ".join(statement.split()) == (
            "DELETE FROM artifact_dependencies WHERE job_id=%s"
        ):
            # Another releaser must be unable to grab even a later input
            # while this transaction starts its decrement triggers.
            for artifact_id in inputs:
                with database.session.connection() as competitor:
                    with pytest.raises(LockNotAvailable):
                        execute(
                            competitor,
                            "SELECT artifact_id FROM artifacts WHERE artifact_id=%s "
                            "FOR UPDATE NOWAIT",
                            (artifact_id,),
                        )
                checked.append(artifact_id)
        return execute(connection, statement, parameters)

    monkeypatch.setattr(PostgresConnection, "execute", before_delete)
    run(job)
    assert checked == inputs
    assert all(database.artifacts.get_artifact(i)["ref_count"] == 0 for i in inputs)
