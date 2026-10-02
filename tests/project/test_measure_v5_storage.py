from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.validation import measure_v5_storage as storage


RUN_ID = "20260930_120000_test"


def _write_spec(tmp_path: Path, **overrides: object) -> tuple[Path, Path]:
    output = tmp_path / "output"
    run_dir = output / "runs" / RUN_ID
    cache_root = output / "cache" / RUN_ID
    run_dir.mkdir(parents=True)
    cache_root.mkdir(parents=True)
    spec: dict[str, object] = {
        "schema_version": 2,
        "run_id": RUN_ID,
        "state_backend": "postgresql",
        "state_db": "dbname=qa user=qa host=/tmp port=55439",
        "state_schema": "loess_v2_storage_test",
        "output_root": str(output),
        "run_dir": str(run_dir),
        "cache_root": str(cache_root),
    }
    spec.update(overrides)
    path = tmp_path / "run_spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path, output


def test_load_context_rejects_noncanonical_or_symlinked_storage_roots(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    spec_path, _output = _write_spec(tmp_path, cache_root=str(outside))
    with pytest.raises(storage.MeasurementError, match="canonical output path"):
        storage._load_context(spec_path)

    spec_path, output = _write_spec(tmp_path / "linked")
    cache_root = output / "cache" / RUN_ID
    cache_root.rmdir()
    cache_root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(storage.MeasurementError, match="must not be symlinks"):
        storage._load_context(spec_path)


class _FakeCursor:
    def __init__(self, results: list[list[tuple[object, ...]]]) -> None:
        self._results = results
        self._index = -1
        self.executions: list[tuple[object, tuple[str, ...]]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, statement: object, parameters: tuple[str, ...]) -> None:
        self._index += 1
        self.executions.append((statement, parameters))

    def fetchall(self) -> list[tuple[object, ...]]:
        return self._results[self._index]

    def fetchone(self) -> tuple[object, ...] | None:
        rows = self._results[self._index]
        return rows[0] if rows else None


class _FakeConnection:
    def __init__(self, results: list[list[tuple[object, ...]]]) -> None:
        self.cursor_instance = _FakeCursor(results)
        self.closed = False

    def cursor(self) -> _FakeCursor:
        return self.cursor_instance

    def close(self) -> None:
        self.closed = True


def test_database_and_filesystem_aggregation_excludes_symlinks(tmp_path: Path):
    pytest.importorskip("psycopg2")
    connection = _FakeConnection(
        [
            [("score", "ready", 2, 300, 4)],
            [("ready", 3)],
            [("work_package", "running", 1), ("unit_fit", "ready", 2)],
            [("running",)],
        ]
    )
    snapshot = storage._database_snapshot(connection, "qa_schema", RUN_ID)
    assert snapshot == {
        "run_status": "running",
        "artifacts": [
            {
                "kind": "score",
                "status": "ready",
                "count": 2,
                "byte_count": 300,
                "ref_count": 4,
            }
        ],
        "work_packages": [{"status": "ready", "count": 3}],
        "jobs": [
            {"job_type": "work_package", "status": "running", "count": 1},
            {"job_type": "unit_fit", "status": "ready", "count": 2},
        ],
    }
    assert len(connection.cursor_instance.executions) == 4
    assert all(
        parameters == (RUN_ID,)
        for _, parameters in connection.cursor_instance.executions
    )

    root = tmp_path / "files"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "one.bin").write_bytes(b"123")
    (nested / "two.bin").write_bytes(b"45678")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"x" * 100)
    (root / "outside-link").symlink_to(outside)
    measured = storage._scan_tree(root)
    assert measured["file_count"] == 2
    assert measured["st_size_bytes"] == 8
    assert measured["symlinks_skipped"] == 1
    assert measured["allocated_bytes"] == measured["st_blocks_512"] * 512


def test_run_writes_flushed_samples_and_final_noninstantaneous_peak_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    spec_path, output = _write_spec(tmp_path)
    run_dir = output / "runs" / RUN_ID
    cache_root = output / "cache" / RUN_ID
    (run_dir / "result.bin").write_bytes(b"run-data")
    (cache_root / "cache.bin").write_bytes(b"cache")
    connection = _FakeConnection([])
    monkeypatch.setattr(storage, "_connect_database", lambda _context: connection)
    monkeypatch.setattr(
        storage,
        "_database_snapshot",
        lambda _connection, _schema, _run_id: {
            "run_status": "running",
            "artifacts": [
                {
                    "kind": "score",
                    "status": "ready",
                    "count": 1,
                    "byte_count": 13,
                    "ref_count": 0,
                }
            ],
            "work_packages": [{"status": "running", "count": 1}],
            "jobs": [{"job_type": "work_package", "status": "running", "count": 1}],
        },
    )
    report = tmp_path / "storage.jsonl"
    args = storage._parser().parse_args(
        [
            "--run-spec",
            str(spec_path),
            "--report",
            str(report),
            "--interval-seconds",
            "0.005",
            "--max-seconds",
            "0.02",
        ]
    )
    summary = storage.run(args)

    lines = [
        json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()
    ]
    assert lines[0]["kind"] == "v5_storage_sample"
    assert lines[0]["duration_seconds"] >= 0
    assert lines[0]["filesystem"]["total"]["st_size_bytes"] == 13
    assert lines[-1] == summary
    assert summary["kind"] == "v5_storage_summary"
    assert summary["stop_reason"] == "max_seconds"
    assert summary["sample_count"] >= 1
    assert summary["sampled_peak"]["active_artifact_byte_count"] == 13
    assert summary["sampled_peak"]["artifact_byte_count_by_status"] == {"ready": 13}
    assert summary["sampled_peak"]["recorded_cleaned_artifact_byte_count"] == 0
    assert "not an exact instantaneous peak" in summary["sampled_peak"]["measurement"]
    assert connection.closed is True

    with pytest.raises(FileExistsError):
        storage.run(args)


def test_stop_file_still_records_one_final_sample_before_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    spec_path, _output = _write_spec(tmp_path)
    stop_file = tmp_path / "stop"
    stop_file.touch()
    connection = _FakeConnection([])
    monkeypatch.setattr(storage, "_connect_database", lambda _context: connection)
    monkeypatch.setattr(
        storage,
        "_database_snapshot",
        lambda _connection, _schema, _run_id: {
            "run_status": "ready",
            "artifacts": [
                {
                    "kind": "score",
                    "status": "cleaned",
                    "count": 2,
                    "byte_count": 20,
                    "ref_count": 0,
                },
                {
                    "kind": "score",
                    "status": "ready",
                    "count": 1,
                    "byte_count": 7,
                    "ref_count": 0,
                },
            ],
            "work_packages": [{"status": "ready", "count": 1}],
            "jobs": [{"job_type": "work_package", "status": "ready", "count": 1}],
        },
    )
    report = tmp_path / "stopped.jsonl"
    args = storage._parser().parse_args(
        [
            "--run-spec",
            str(spec_path),
            "--report",
            str(report),
            "--max-seconds",
            "60",
            "--stop-file",
            str(stop_file),
        ]
    )

    summary = storage.run(args)
    lines = [
        json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()
    ]
    assert [line["kind"] for line in lines] == [
        "v5_storage_sample",
        "v5_storage_summary",
    ]
    assert lines[0]["sample_role"] == "final_after_stop"
    assert summary["stop_reason"] == "stop_file"
    assert summary["sample_count"] == 1
    assert summary["sampled_peak"]["active_artifact_byte_count"] == 7
    assert summary["sampled_peak"]["recorded_cleaned_artifact_byte_count"] == 20
