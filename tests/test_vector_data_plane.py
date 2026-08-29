"""Small contract tests for the mandatory columnar vector exchange path."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect

import pyogrio
import pytest
from shapely.geometry import Polygon

import assemble_stream
import boundary_fitting.unit_runtime as unit_runtime
import vector_data_plane
from vector_data_plane import (
    VectorDataPlaneError,
    read_boundary_signatures,
    read_geoparquet,
    signature_links,
    unit_boundary_signatures,
    write_boundary_signatures,
    write_geoparquet,
)


def _record(part_id: str, geometry: Polygon):
    return {
        "part_id": part_id,
        "class_code": 12,
        "conf_mean": 1.0,
        "conf_std": 0.0,
        "geometry": geometry,
    }


def test_unit_columnar_artifacts_are_committed_without_a_fiona_gpkg_writer(tmp_path):
    raw = [_record("part-a", Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))]
    formal = [
        {
            **raw[0],
            "fit_method": "unchanged",
            "fit_status": "unchanged",
            "fit_version": "divider_cubic_bspline_adaptive_v2",
        }
    ]
    raw_path = tmp_path / "unit_raw.parquet"
    formal_path = tmp_path / "unit_formal.parquet"
    edges_path = tmp_path / "unit_fitted_edges.parquet"
    source_sha = "a" * 64
    for path, rows in ((raw_path, raw), (formal_path, formal), (edges_path, raw)):
        write_geoparquet(path, rows, crs="EPSG:3857", source_sha256=source_sha)
        manifest, table = read_geoparquet(path)
        assert manifest["status"] == "committed"
        assert manifest["feature_count"] == 1
        assert table.column_names[-1] == "source_sha256"

    signature_path = tmp_path / "unit_boundary_signatures.json"
    written = write_boundary_signatures(
        signature_path,
        unit_boundary_signatures(
            [
                {
                    "polygon_id": "part-a",
                    "class_code": 12,
                    "geometry": raw[0]["geometry"],
                }
            ],
            stream_id="model:a",
            unit_id="unit-a",
            pixel_window={"x0": 0, "x1": 1, "y0": 0, "y1": 1},
        ),
        stream_id="model:a",
        unit_id="unit-a",
    )
    assert written["record_count"] == 4
    assert len(read_boundary_signatures(signature_path, stream_id="model:a", unit_id="unit-a")) == 4
    assert "fiona.open" not in inspect.getsource(unit_runtime)


def test_signature_links_make_object_ids_deterministic_without_spatial_or_database_graph(tmp_path):
    left = _record("left", Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))
    right = _record("right", Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]))
    paths = {}
    for unit_id, record, window in (
        ("left-unit", left, {"x0": 0, "x1": 1, "y0": 0, "y1": 1}),
        ("right-unit", right, {"x0": 1, "x1": 2, "y0": 0, "y1": 1}),
    ):
        data_path = tmp_path / f"{unit_id}.parquet"
        signature_path = tmp_path / f"{unit_id}.json"
        write_geoparquet(data_path, [record], crs="EPSG:3857", source_sha256="b" * 64)
        write_boundary_signatures(
            signature_path,
            unit_boundary_signatures(
                [{"polygon_id": record["part_id"], "class_code": 12, "geometry": record["geometry"]}],
                stream_id="model:a", unit_id=unit_id, pixel_window=window,
            ),
            stream_id="model:a", unit_id=unit_id,
        )
        paths[unit_id] = (data_path, signature_path)
    left_signatures = read_boundary_signatures(paths["left-unit"][1], stream_id="model:a", unit_id="left-unit")
    right_signatures = read_boundary_signatures(paths["right-unit"][1], stream_id="model:a", unit_id="right-unit")
    assert signature_links(left_signatures, right_signatures, tolerance=1e-9) == [("left", "right", 12)]
    units = [
        {"unit_id": "left-unit", "pixel_window": {"x0": 0, "x1": 1, "y0": 0, "y1": 1}},
        {"unit_id": "right-unit", "pixel_window": {"x0": 1, "x1": 2, "y0": 0, "y1": 1}},
    ]
    formal = {unit_id: str(value[0]) for unit_id, value in paths.items()}
    signatures = {unit_id: str(value[1]) for unit_id, value in paths.items()}
    first, count = assemble_stream._signature_object_ids(
        signatures, formal, units, "run-1", "model:a", 1e-9
    )
    second, repeat_count = assemble_stream._signature_object_ids(
        signatures, formal, units, "run-1", "model:a", 1e-9
    )
    assert count == repeat_count == 1
    assert first == second and first["left"] == first["right"]
    assert first["left"] == "obj_" + hashlib.sha1(
        b"run-1|model:a|left"
    ).hexdigest()[:24]
    assert "fiona.open" not in inspect.getsource(assemble_stream._read_features)
    assert "fiona.open" not in inspect.getsource(assemble_stream._atomic_gpkg)
    reducer_source = inspect.getsource(assemble_stream._signature_object_ids)
    assert "STRtree" not in reducer_source
    assert "object_nodes" not in reducer_source
    assert "object_links" not in reducer_source


def test_pyogrio_arrow_preserves_final_gpkg_layer_fields_and_crs(tmp_path):
    path = tmp_path / "semantic_polygons_raw.gpkg"
    schema = {
        "geometry": "MultiPolygon",
        "properties": {"run_id": "str:48", "stream_id": "str:96", "class_code": "int"},
    }
    assemble_stream._atomic_gpkg(
        path,
        "semantic_polygons_raw",
        schema,
        "EPSG:3857",
        lambda destination: destination.writerecords(
            [
                {
                    "geometry": {"type": "Polygon", "coordinates": [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]]},
                    "properties": {"run_id": "run-1", "stream_id": "model:a", "class_code": 12},
                }
            ]
        ),
    )
    info = pyogrio.read_info(path, layer="semantic_polygons_raw")
    assert info["features"] == 1
    assert set(info["fields"]) == set(schema["properties"])
    assert "3857" in str(info["crs"])
    assert info["geometry_type"] == "MultiPolygon"


def test_columnar_backend_fails_closed_when_dependency_is_unavailable(monkeypatch):
    monkeypatch.setattr(vector_data_plane, "backend_status", lambda: {"available": False})
    with pytest.raises(VectorDataPlaneError, match="required columnar vector data plane"):
        vector_data_plane.require_backend()


def test_four_stream_shards_publish_to_independent_targets(tmp_path):
    def publish(stream_id: str) -> str:
        path = tmp_path / stream_id / "unit_raw.parquet"
        return write_geoparquet(
            path,
            [_record(f"{stream_id}:part", Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))],
            crs="EPSG:3857",
            source_sha256="c" * 64,
        )["artifact_sha256"]

    with ThreadPoolExecutor(max_workers=4) as executor:
        digests = list(executor.map(publish, ["model:a", "model:b", "fusion:x", "fusion:y"]))
    assert len(set(digests)) == 4
    assert all((tmp_path / stream / "unit_raw.parquet").is_file() for stream in ("model:a", "model:b", "fusion:x", "fusion:y"))
