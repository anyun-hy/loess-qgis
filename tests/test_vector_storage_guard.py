from types import SimpleNamespace

import pytest
from affine import Affine
from shapely.geometry import box

import assemble_stream
from boundary_fitting import unit_runtime
from storage_guard import StorageGuard, StorageReserveError


def _disk_usage(free_bytes):
    return lambda _path: SimpleNamespace(
        total=max(1, int(free_bytes)),
        used=0,
        free=int(free_bytes),
    )


class _ReadyCoreArtifacts:
    def partitions_for_run(self, _run_id):
        return [
            {
                "partition_id": "partition-0",
                "core_window": {"x0": 0, "y0": 0, "x1": 10, "y1": 10},
            },
            {
                "partition_id": "partition-1",
                "core_window": {"x0": 10, "y0": 0, "x1": 20, "y1": 10},
            },
        ]

    def artifacts_for_stream(self, _run_id, _stream_id, *, kind, status):
        assert status == "ready"
        assert kind in {"core_mask", "core_confidence"}
        return [{"unit_id": "partition-0"}]


def _storage_spec(tmp_path):
    return {
        "run_id": "run-storage",
        "run_dir": str(tmp_path),
        "streams": [{"stream_id": "model:a"}],
        "spatial_plan_summary": {"partition_count": 2},
        "storage_preflight": {
            "storage_tuning_schema_version": 2,
            "estimated_permanent_raster_bytes": 1_200,
            "nondecaying_permanent_reserve_bytes": 300,
            "permanent_uncertainty_bytes": 999,
            "effective_min_free_disk_bytes": 100,
        },
    }


@pytest.mark.parametrize(
    "factory",
    [unit_runtime._run_storage_guard, assemble_stream._run_storage_guard],
)
def test_vector_guard_uses_frozen_minimum_and_nondecaying_reserve(
    tmp_path,
    factory,
):
    guard = factory(
        _storage_spec(tmp_path),
        _ReadyCoreArtifacts(),
        disk_usage=_disk_usage(1_100),
    )

    report = guard.check("vector", write_bytes=100)

    # Two of four equal-sized Core artifacts are ready: 600 raster bytes remain.  The
    # schema-2 nondecaying field contributes another 300 bytes, and the frozen
    # filesystem floor contributes 100 bytes.
    assert report["remaining_permanent_bytes"] == 900
    assert report["required_free_bytes"] == 1_100


class _UnequalReadyCoreArtifacts:
    def partitions_for_run(self, _run_id):
        return [
            {
                "partition_id": "small",
                "core_window": {"x0": 0, "y0": 0, "x1": 1, "y1": 1},
            },
            {
                "partition_id": "large",
                "core_window": {"x0": 1, "y0": 0, "x1": 4, "y1": 3},
            },
        ]

    def artifacts_for_stream(self, _run_id, _stream_id, *, kind, status):
        assert status == "ready"
        return [{"unit_id": "small"}]


@pytest.mark.parametrize(
    "remaining",
    [
        unit_runtime._remaining_permanent_reserve_bytes,
        assemble_stream._remaining_permanent_reserve_bytes,
    ],
)
def test_vector_guard_subtracts_exact_edge_core_bytes(remaining, tmp_path):
    spec = {
        "run_id": "run-storage",
        "run_dir": str(tmp_path),
        "streams": [{"stream_id": "model:a"}],
        "storage_preflight": {
            "storage_tuning_schema_version": 2,
            # (1 px + 9 px) * (2-byte mask + 4-byte confidence)
            "estimated_permanent_raster_bytes": 60,
            "nondecaying_permanent_reserve_bytes": 7,
        },
    }

    # Only the 1-pixel edge Core is ready, so exactly 9 * 6 bytes remain.
    assert remaining(spec, _UnequalReadyCoreArtifacts()) == 61


def test_unit_gpkg_low_disk_fails_before_temp_and_preserves_target(tmp_path):
    target = tmp_path / "unit_raw.gpkg"
    target.write_bytes(b"existing-formal-output")
    guard = StorageGuard(
        tmp_path,
        min_free_bytes=0,
        disk_usage=_disk_usage(unit_runtime.GPKG_ATOMIC_OVERHEAD_BYTES - 1),
    )
    records = [
        {
            "polygon_id": "unit-1:0",
            "class_code": 1,
            "confidence_mean": 0.9,
            "confidence_std": 0.1,
            "geometry": box(0, 0, 2, 2),
        }
    ]

    with pytest.raises(StorageReserveError):
        unit_runtime._write_gpkg(
            target,
            records,
            transform=Affine.identity(),
            crs="EPSG:3857",
            include_fit=False,
            storage_guard=guard,
            storage_lock_path=tmp_path / "storage.lock",
            operation="unit_raw:test",
        )

    assert target.read_bytes() == b"existing-formal-output"
    assert list(tmp_path.glob("*.tmp.gpkg")) == []
    assert guard.pending_write_bytes == 0


def test_unit_fitted_edges_low_disk_preserves_target(tmp_path):
    target = tmp_path / "unit_fitted_edges.gpkg"
    target.write_bytes(b"existing-edge-output")
    guard = StorageGuard(
        tmp_path,
        min_free_bytes=0,
        disk_usage=_disk_usage(unit_runtime.GPKG_ATOMIC_OVERHEAD_BYTES - 1),
    )
    report = {
        "diagnostics": [
            {
                "chain_id": "edge-1",
                "method": "line",
                "status": "changed",
                "fitted_points": [(0.0, 0.0), (2.0, 2.0)],
            }
        ]
    }

    with pytest.raises(StorageReserveError):
        unit_runtime._write_diagnostic_gpkg(
            target,
            report,
            run_id="run-storage",
            stream_id="model:a",
            unit_id="unit-1",
            transform=Affine.identity(),
            crs="EPSG:3857",
            storage_guard=guard,
            storage_lock_path=tmp_path / "storage.lock",
        )

    assert target.read_bytes() == b"existing-edge-output"
    assert list(tmp_path.glob("*.tmp.gpkg")) == []
    assert guard.pending_write_bytes == 0


def test_unit_report_low_disk_preserves_target(tmp_path):
    target = tmp_path / "unit_report.json"
    target.write_text('{"old":true}\n', encoding="utf-8")
    guard = StorageGuard(
        tmp_path,
        min_free_bytes=0,
        disk_usage=_disk_usage(1),
    )

    with pytest.raises(StorageReserveError):
        unit_runtime._write_json(
            target,
            {"status": "passed"},
            storage_guard=guard,
            storage_lock_path=tmp_path / "storage.lock",
        )

    assert target.read_text(encoding="utf-8") == '{"old":true}\n'
    assert list(tmp_path.glob("*.tmp")) == []
    assert guard.pending_write_bytes == 0


def test_stream_gpkg_low_disk_does_not_invoke_writer_or_create_stage(tmp_path):
    target = tmp_path / ".fitted_edges.123.stage.gpkg"
    guard = StorageGuard(
        tmp_path,
        min_free_bytes=0,
        disk_usage=_disk_usage(1_023),
    )
    called = []

    with pytest.raises(StorageReserveError):
        assemble_stream._atomic_gpkg(
            target,
            "fitted_edges",
            {"geometry": "LineString", "properties": {}},
            "EPSG:3857",
            lambda _destination: called.append(True),
            storage_guard=guard,
            storage_lock_path=tmp_path / "storage.lock",
            estimated_write_bytes=1_024,
            operation="stream_fitted_edges_stage:test",
        )

    assert called == []
    assert not target.exists()
    assert list(tmp_path.glob("*.tmp.gpkg")) == []
    assert guard.pending_write_bytes == 0


def test_stream_gpkg_writer_failure_cleans_temp_and_preserves_target(tmp_path):
    target = tmp_path / "semantic_polygons.gpkg"
    target.write_bytes(b"existing-formal-output")
    guard = StorageGuard(
        tmp_path,
        min_free_bytes=0,
        disk_usage=_disk_usage(100 * 1024**2),
    )

    def fail_writer(_destination):
        raise RuntimeError("injected vector writer failure")

    with pytest.raises(RuntimeError, match="injected vector writer failure"):
        assemble_stream._atomic_gpkg(
            target,
            "semantic_polygons",
            {"geometry": "MultiPolygon", "properties": {}},
            "EPSG:3857",
            fail_writer,
            storage_guard=guard,
            storage_lock_path=tmp_path / "storage.lock",
            estimated_write_bytes=1_024,
            operation="stream_formal:test",
        )

    assert target.read_bytes() == b"existing-formal-output"
    assert list(tmp_path.glob("*.tmp.gpkg")) == []
    assert guard.pending_write_bytes == 0


def test_stream_report_low_disk_preserves_target(tmp_path):
    target = tmp_path / ".boundary_fitting_report.123.stage.json"
    target.write_text('{"old":true}\n', encoding="utf-8")
    guard = StorageGuard(
        tmp_path,
        min_free_bytes=0,
        disk_usage=_disk_usage(1),
    )

    with pytest.raises(StorageReserveError):
        assemble_stream._write_json(
            target,
            {"status": "passed"},
            storage_guard=guard,
            storage_lock_path=tmp_path / "storage.lock",
        )

    assert target.read_text(encoding="utf-8") == '{"old":true}\n'
    assert list(tmp_path.glob("*.tmp")) == []
    assert guard.pending_write_bytes == 0


def test_missing_accepted_layer_does_not_reserve_candidate_write(tmp_path):
    guard = StorageGuard(
        tmp_path,
        min_free_bytes=0,
        disk_usage=_disk_usage(1),
    )
    output = tmp_path / ".semantic_candidates.123.stage.gpkg"

    report = assemble_stream._guarded_accepted_difference(
        tmp_path / "formal-is-not-opened.gpkg",
        tmp_path / "missing-accepted.gpkg",
        output,
        storage_guard=guard,
        storage_lock_path=tmp_path / "storage.lock",
        operation="stream_candidate_stage:test",
    )

    assert report == {"status": "skipped", "reason": "accepted layer is unavailable"}
    assert not output.exists()
    assert guard.pending_write_bytes == 0
