import hashlib

import pytest

from labeling_tool.shared.state.run_state_session import RunStateError
from loess_runtime.assembly.scale_acceptance import (
    _database_metrics,
    _final_artifact_size_observation,
    _package_storage_metrics,
)


@pytest.mark.parametrize(
    "changes",
    [
        {"storage_metrics_schema_version": None},
        {"storage_metrics_schema_version": 2},
        {"storage_metrics_measurement": None},
        {"storage_metrics_measurement": "directory_snapshot"},
        {"peak_package_managed_bytes": None},
        {"peak_cache_bytes": -1},
        {"peak_cache_bytes": 21},
        {"peak_cache_bytes": True},
    ],
)
def test_combined_missing_or_invalid_cache_evidence_cannot_be_reinterpreted(changes):
    report = {
        "storage_metrics_schema_version": 1,
        "storage_metrics_measurement": "package_guard_reserved_growth_v1",
        "peak_cache_bytes": 10,
        "peak_package_managed_bytes": 20,
    }
    for key, value in changes.items():
        if value is None:
            report.pop(key)
        else:
            report[key] = value
    assert _package_storage_metrics(report) is None


def test_separate_storage_peaks_preserve_working_and_retained_scopes():
    assert _package_storage_metrics(
        {
            "storage_metrics_schema_version": 1,
            "storage_metrics_measurement": "package_guard_reserved_growth_v1",
            "peak_cache_bytes": 100,
            "peak_package_managed_bytes": 700,
        }
    ) == (100, 700)


def test_acceptance_read_uses_one_connection_and_preserves_artifact_order(
    postgres_database, monkeypatch, tmp_path
):
    database = postgres_database
    run_id = "acceptance-read-fixture"
    database.run_streams.create_run(run_id, "a" * 64)
    database.jobs.insert_jobs(
        run_id,
        [{"job_type": "unit_fit", "stream_id": "model:a", "unit_id": "core:1"}],
    )
    job = database.jobs.lease_next_job(run_id, "acceptance-test", lease_seconds=120)
    assert job is not None
    with database.session.transaction() as connection:
        connection.execute(
            "UPDATE jobs SET attempt=3 WHERE job_id=%s", (job["job_id"],)
        )

    ready_path = tmp_path / "z-ready.json"
    ready_path.write_bytes(b"ready artifact")
    ready_id = database.artifacts.register_artifact(run_id, "report", ready_path)
    assert database.artifacts.mark_artifact_ready(
        ready_id,
        byte_count=ready_path.stat().st_size,
        sha256=hashlib.sha256(ready_path.read_bytes()).hexdigest(),
    )
    writing_path = tmp_path / "a-writing.json"
    database.artifacts.register_artifact(run_id, "report", writing_path)

    original_connect = database.session.connect
    connections = []

    def tracked_connect(*, autocommit=True):
        connection = original_connect(autocommit=autocommit)
        connections.append(connection)
        return connection

    monkeypatch.setattr(database.session, "connect", tracked_connect)
    snapshot = database.acceptance_read.snapshot(run_id)
    assert len(connections) == 1
    assert connections[0].raw.closed
    assert snapshot["counts"] == {
        "tiles": 0,
        "partitions": 0,
        "spatial_units": 0,
        "work_packages": 0,
    }
    assert snapshot["job_counts"] == {"running": 1}
    assert snapshot["job_type_counts"] == {"unit_fit": {"running": 1}}
    assert snapshot["retry_count"] == 2
    assert [row["path"] for row in snapshot["artifact_rows"]] == [
        str(ready_path),
        str(writing_path),
    ]

    metrics = _database_metrics(database, run_id)
    assert len(connections) == 2
    assert connections[1].raw.closed
    assert metrics["artifact_counts"] == {"ready": 1, "writing": 1}
    assert metrics["ready_artifact_bytes"] == len(b"ready artifact")
    assert metrics["artifact_integrity_errors"] == []
    ready_path.write_bytes(b"corrupted")
    assert _database_metrics(database, run_id)["artifact_integrity_errors"] == [
        f"size:{ready_path}"
    ]

    with database.unit_attempt_commit(job["job_id"], job["lease_token"]) as scoped:
        in_scope_path = tmp_path / "0-in-scope.json"
        scoped.artifacts.register_artifact(run_id, "report", in_scope_path)
        scoped_rows = scoped.acceptance_read.snapshot(run_id)["artifact_rows"]
        assert len(connections) == 4
        assert [row["path"] for row in scoped_rows] == [
            str(ready_path),
            str(writing_path),
            str(in_scope_path),
        ]
    assert connections[3].raw.closed
    with pytest.raises(RunStateError, match="no longer active"):
        scoped.acceptance_read.snapshot(run_id)


def test_final_artifact_observation_reports_a_signed_difference_without_a_gate():
    observation = _final_artifact_size_observation(
        {
            "final_artifact_size_prediction": {
                "status": "predicted",
                "observation_only": True,
                "predicted_final_artifact_bytes": 100,
            }
        },
        125,
    )

    assert observation["status"] == "predicted"
    assert observation["actual_final_artifact_bytes"] == 125
    assert observation["signed_difference_bytes"] == 25
    assert observation["signed_difference_ratio"] == 0.25


def test_final_artifact_observation_leaves_an_uncalibrated_run_descriptive():
    observation = _final_artifact_size_observation({}, 125)

    assert observation == {
        "status": "not_available",
        "actual_final_artifact_bytes": 125,
        "observation_only": True,
    }
