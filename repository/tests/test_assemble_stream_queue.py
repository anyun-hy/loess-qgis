import hashlib
import json
import sqlite3
import time
from pathlib import Path

import fiona
import pytest
from fiona.crs import CRS

import assemble_stream


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report(index):
    return {
        "status": "passed",
        "chain_count": index,
        "shared_chain_count": index % 7,
        "spline_count": index % 5,
        "unchanged_count": index % 3,
        "skipped_invalid_count": 0,
        "max_displacement_px": index / 100.0,
        "diagnostics": [],
    }


def _write_reports(root, count):
    artifacts = []
    for index in range(count):
        unit_id = f"unit_{index:05d}"
        path = root / f"{unit_id}.json"
        path.write_text(json.dumps(_report(index)), encoding="utf-8")
        artifacts.append(
            {
                "artifact_id": count - index,
                "unit_id": unit_id,
                "path": str(path),
            }
        )
    return artifacts


def _write_line_gpkg(path):
    schema = {
        "geometry": "LineString",
        "properties": {"name": "str:16"},
    }
    with fiona.open(
        path,
        "w",
        driver="GPKG",
        layer="fitted_edges",
        schema=schema,
        crs=CRS.from_epsg(4326),
    ):
        pass


def test_report_queue_is_ordered_and_bounded(tmp_path):
    artifacts = _write_reports(tmp_path, 100)
    observed = []

    def consume(unit_id, _report_value):
        observed.append(unit_id)
        time.sleep(0.003)

    stats = assemble_stream._consume_reports(list(reversed(artifacts)), consume)

    assert observed == [f"unit_{index:05d}" for index in range(100)]
    assert stats == {
        "processed_count": 100,
        "max_loaded_count": assemble_stream.REPORT_QUEUE_CAPACITY,
    }
    assert assemble_stream.REPORT_QUEUE_CAPACITY == 32


def test_report_queue_processes_12635_reports_without_exceeding_capacity(
    monkeypatch,
    tmp_path,
):
    count = 12_635
    artifacts = [
        {
            "artifact_id": count - index,
            "unit_id": f"unit_{index:05d}",
            "path": str(tmp_path / f"unit_{index:05d}.json"),
        }
        for index in range(count)
    ]
    consumed = 0

    def fake_load(path):
        return _report(int(path.stem.rsplit("_", 1)[1]))

    def consume(_unit_id, _report_value):
        nonlocal consumed
        consumed += 1

    monkeypatch.setattr(assemble_stream, "load_json", fake_load)
    stats = assemble_stream._consume_reports(list(reversed(artifacts)), consume)

    assert consumed == count
    assert stats["processed_count"] == count
    assert stats["max_loaded_count"] <= assemble_stream.REPORT_QUEUE_CAPACITY


def test_corrupt_report_preserves_existing_output_and_cleans_temporary_file(
    tmp_path,
):
    artifacts = _write_reports(tmp_path, 64)
    Path(artifacts[32]["path"]).write_text("{broken-json", encoding="utf-8")
    existing = tmp_path / "fitted_edges.gpkg"
    _write_line_gpkg(existing)
    existing_sha = _digest(existing)

    def broken_writer(_destination):
        assemble_stream._consume_reports(
            artifacts,
            lambda _unit_id, _report_value: None,
        )

    started = time.monotonic()
    with pytest.raises(assemble_stream.StreamAssemblyError):
        assemble_stream._atomic_gpkg(
            existing,
            "fitted_edges",
            {
                "geometry": "LineString",
                "properties": {"name": "str:16"},
            },
            "EPSG:4326",
            broken_writer,
        )

    assert time.monotonic() - started < 10
    assert _digest(existing) == existing_sha
    assert not list(tmp_path.glob(".*.tmp.gpkg"))


def test_consumer_failure_propagates_without_deadlock(tmp_path):
    artifacts = _write_reports(tmp_path, 64)

    def fail(_unit_id, _report_value):
        raise RuntimeError("consumer failure")

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="consumer failure"):
        assemble_stream._consume_reports(artifacts, fail)

    assert time.monotonic() - started < 10


def test_resolved_object_state_rejects_missing_and_unresolved_rows(tmp_path):
    database_path = tmp_path / "state.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """CREATE TABLE object_nodes (
                   run_id TEXT,
                   stream_id TEXT,
                   part_id TEXT,
                   object_id TEXT,
                   parent_id TEXT
               )"""
        )

    with pytest.raises(
        assemble_stream.StreamAssemblyError,
        match="resume requires existing resolved object IDs",
    ):
        assemble_stream._resolved_object_state(
            database_path,
            "run-1",
            "model:a",
        )

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """INSERT INTO object_nodes
               (run_id, stream_id, part_id, object_id, parent_id)
               VALUES (?, ?, ?, ?, ?)""",
            ("run-1", "model:a", "part-1", "", "part-1"),
        )

    with pytest.raises(
        assemble_stream.StreamAssemblyError,
        match="unresolved=1",
    ):
        assemble_stream._resolved_object_state(
            database_path,
            "run-1",
            "model:a",
        )

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """UPDATE object_nodes SET object_id=?
               WHERE run_id=? AND stream_id=?""",
            ("object-1", "run-1", "model:a"),
        )

    assert assemble_stream._resolved_object_state(
        database_path,
        "run-1",
        "model:a",
    ) == (1, 1)


def test_resume_input_fingerprint_change_is_rejected(tmp_path):
    path = tmp_path / "semantic_polygons.gpkg"
    path.write_bytes(b"before")
    fingerprint = assemble_stream._file_fingerprint(path)
    path.write_bytes(b"after")

    with pytest.raises(
        assemble_stream.StreamAssemblyError,
        match="resume input changed",
    ):
        assemble_stream._assert_fingerprint_unchanged(path, fingerprint)


def test_resume_failure_event_is_actionable(monkeypatch, capsys):
    def fail(*_args, **_kwargs):
        raise assemble_stream.StreamAssemblyError(
            "resume requires existing resolved object IDs; "
            "parts=0, objects=0, unresolved=0"
        )

    monkeypatch.setattr(assemble_stream, "assemble_stream", fail)
    return_code = assemble_stream.main(
        [
            "--run-spec",
            "/tmp/run_spec.json",
            "--stream-id",
            "model:a",
            "--resume-from-reports",
        ]
    )
    event = json.loads(capsys.readouterr().out)

    assert return_code == 2
    assert event["event"] == "stream_assembly_failed"
    assert event["assembly_mode"] == "report_resume"
    assert event["safe_retry"] == "rerun_without_resume_from_reports"

