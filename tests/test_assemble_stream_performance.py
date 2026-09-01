import json
import sqlite3
from pathlib import Path

import fiona
import pytest
from fiona.crs import CRS
from shapely.geometry import MultiPolygon, box, mapping

import assemble_stream


def _write_resume_gpkg(path):
    schema = {
        "geometry": "MultiPolygon",
        "properties": {
            "run_id": "str:48",
            "stream_id": "str:96",
        },
    }
    with fiona.open(
        path,
        "w",
        driver="GPKG",
        layer="semantic_polygons_raw",
        schema=schema,
        crs=CRS.from_epsg(4326),
    ) as destination:
        destination.write(
            {
                "geometry": mapping(MultiPolygon([box(0, 0, 1, 1)])),
                "properties": {
                    "run_id": "run-1",
                    "stream_id": "model:a",
                },
            }
        )


def test_resume_formal_validation_allows_exact_clip_to_change_feature_count(tmp_path):
    """Raw parts are count-locked; clipped formal output is identity-locked instead."""
    path = tmp_path / "semantic_polygons_raw.gpkg"
    _write_resume_gpkg(path)

    validated = assemble_stream._validate_existing_gpkg(
        path,
        layer="semantic_polygons_raw",
        schema={
            "geometry": "MultiPolygon",
            "properties": {"run_id": "str:48", "stream_id": "str:96"},
        },
        crs="EPSG:4326",
        identity={"run_id": "run-1", "stream_id": "model:a"},
        expected_feature_count=None,
    )

    assert validated["feature_count"] == 1


def test_100_reverse_report_summaries_are_validated_in_unit_id_order(
    monkeypatch,
    tmp_path,
):
    unit_ids = [f"unit_{index:05d}" for index in range(100)]
    artifacts = [
        {
            "unit_id": unit_id,
            "path": str(tmp_path / f"{unit_id}.json"),
            "byte_count": 1,
            "sha256": "a" * 64,
        }
        for unit_id in reversed(unit_ids)
    ]
    summaries = [
        {
            "unit_id": artifact["unit_id"],
            "report_path": artifact["path"],
            "report_byte_count": artifact["byte_count"],
            "report_sha256": artifact["sha256"],
            "fitted_edge_count": 0,
        }
        for artifact in artifacts
    ]

    class ReverseDatabase:
        def unit_report_summaries(self, _run_id, _stream_id):
            return summaries

        def artifacts_for_stream(
            self,
            _run_id,
            _stream_id,
            *,
            kind=None,
        ):
            if kind == "unit_fitted_edges_geoparquet":
                return []
            raise AssertionError(kind)

    observed = []

    def validate(items, **_kwargs):
        observed.extend(item["artifact"]["unit_id"] for item in items)
        return {
            "workers": 8,
            "peak_in_flight": 32,
            "artifact_count": len(items),
            "elapsed_sec": 0.0,
        }

    monkeypatch.setattr(
        assemble_stream,
        "_parallel_validate_summary_artifacts",
        validate,
    )

    edges, stats = assemble_stream._validated_summary_inputs(
        {"raster": {"crs": "EPSG:4326"}},
        ReverseDatabase(),
        "run-1",
        "model:a",
        len(unit_ids),
        artifacts,
        {"geometry": "LineString", "properties": {}},
    )

    assert edges == []
    assert observed == unit_ids
    assert stats["artifact_count"] == 100
    assert stats["peak_in_flight"] <= 32


def test_parallel_validation_keeps_12635_artifacts_bounded(
    monkeypatch,
    tmp_path,
):
    count = 12_635
    report_path = tmp_path / "report.json"
    report_path.write_text("x", encoding="utf-8")
    items = [
        {
            "kind": "report",
            "artifact": {
                "unit_id": f"unit_{index:05d}",
                "path": str(report_path),
                "byte_count": 1,
                "sha256": "a" * 64,
            },
        }
        for index in range(count)
    ]
    monkeypatch.setattr(
        assemble_stream,
        "_file_fingerprint",
        lambda _path: {
            "byte_count": 1,
            "sha256": "a" * 64,
            "mtime_ns": 0,
        },
    )

    stats = assemble_stream._parallel_validate_summary_artifacts(
        items,
        workers=8,
        edge_schema={"geometry": "LineString", "properties": {}},
        crs="EPSG:4326",
        run_id="run-1",
        stream_id="model:a",
    )

    assert stats["artifact_count"] == count
    assert stats["workers"] == 8
    assert (
        1
        <= stats["peak_in_flight"]
        <= assemble_stream.ASSEMBLY_VALIDATION_MAX_IN_FLIGHT
    )


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


def test_resume_gpkg_requires_matching_schema_crs_identity_and_count(tmp_path):
    path = tmp_path / "semantic_polygons_raw.gpkg"
    _write_resume_gpkg(path)
    schema = {
        "geometry": "MultiPolygon",
        "properties": {
            "run_id": "str:48",
            "stream_id": "str:96",
        },
    }
    common = {
        "layer": "semantic_polygons_raw",
        "schema": schema,
        "crs": "EPSG:4326",
        "identity": {
            "run_id": "run-1",
            "stream_id": "model:a",
        },
        "expected_feature_count": 1,
    }

    fingerprint = assemble_stream._validate_existing_gpkg(path, **common)
    assert fingerprint["feature_count"] == 1
    assert fingerprint["byte_count"] == path.stat().st_size
    assert len(fingerprint["sha256"]) == 64

    with pytest.raises(
        assemble_stream.StreamAssemblyError,
        match="fields changed",
    ):
        assemble_stream._validate_existing_gpkg(
            path,
            **{
                **common,
                "schema": {
                    **schema,
                    "properties": {
                        **schema["properties"],
                        "class_code": "int",
                    },
                },
            },
        )

    with pytest.raises(
        assemble_stream.StreamAssemblyError,
        match="CRS changed",
    ):
        assemble_stream._validate_existing_gpkg(
            path,
            **{**common, "crs": "EPSG:3857"},
        )

    with pytest.raises(
        assemble_stream.StreamAssemblyError,
        match="another run or stream",
    ):
        assemble_stream._validate_existing_gpkg(
            path,
            **{
                **common,
                "identity": {
                    "run_id": "run-2",
                    "stream_id": "model:a",
                },
            },
        )

    with pytest.raises(
        assemble_stream.StreamAssemblyError,
        match="feature count changed",
    ):
        assemble_stream._validate_existing_gpkg(
            path,
            **{**common, "expected_feature_count": 2},
        )


def test_readonly_geopackage_context_closes_its_file_descriptor(
    tmp_path,
    monkeypatch,
):
    geopackage_path = tmp_path / "integrity-fixture.gpkg"
    with sqlite3.connect(geopackage_path) as connection:
        connection.execute("CREATE TABLE evidence(value INTEGER)")

    real_connect = assemble_stream.sqlite3.connect
    opened = []

    def tracking_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(
        assemble_stream.sqlite3,
        "connect",
        tracking_connect,
    )
    with assemble_stream._readonly_geopackage(geopackage_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM evidence"
        ).fetchone()[0] == 0

    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        opened[0].execute("SELECT 1")


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


def test_assembly_exception_marks_stream_failed(
    monkeypatch, tmp_path, postgres_database
):
    database = postgres_database
    spec_path = tmp_path / "run_spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "run-1",
                "state_backend": "postgresql",
                "state_db": database.location,
                "state_schema": database.postgres_schema,
            }
        ),
        encoding="utf-8",
    )
    database.create_run("run-1", "0" * 64)
    database.register_streams(
        "run-1",
        [
            {
                "stream_id": "model:a",
                "kind": "model",
                "model_id": "a",
                "version": "fixture",
                "status": "ready",
            }
        ],
    )

    def fail(*_args, **_kwargs):
        raise assemble_stream.StreamAssemblyError("assembled artifact changed")

    monkeypatch.setattr(assemble_stream, "_assemble_stream_impl", fail)

    with pytest.raises(
        assemble_stream.StreamAssemblyError,
        match="assembled artifact changed",
    ):
        assemble_stream.assemble_stream(spec_path, "model:a")

    stream = database.stream_rows("run-1")[0]
    assert stream["status"] == "failed"
    assert stream["error"] == "assembled artifact changed"
