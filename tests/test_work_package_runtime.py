import hashlib
import json
import sqlite3
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import fiona
import numpy as np
import pytest
import rasterio
import assemble_stream as assemble_stream_module
import work_package_runtime
from affine import Affine
from rasterio.transform import from_origin
from shapely.geometry import box, shape
from shapely.ops import unary_union

from labeling_tool.core.run_builder_v5 import create_v5_run
from labeling_tool.core.run_state_db import RunStateDB
from labeling_tool.core.run_spec import atomic_write_json
from labeling_tool.core.spatial_planner import plan_spatial_units
from boundary_fitting.unit_runtime import (
    _polygonize,
    _write_diagnostic_geoparquet,
    _write_geoparquet,
    run_unit_fit,
)
from vector_data_plane import unit_boundary_signatures, write_boundary_signatures
from assemble_stream import StreamAssemblyError, assemble_stream
from finalize_partition_rasters import finalize_partition_rasters
from fragmentation_v33_candidate import (
    executor_snapshot_sha256,
    policy_snapshot_sha256,
)
from fragmentation_v33_work_package import run_worker as run_v33_worker
from scale_acceptance import build_scale_acceptance_report
from work_package_runtime import (
    BatchCapacityError,
    PersistentModelProvider,
    WorkPackageRuntimeError,
    LeaseLostError,
    _PackageFileLock,
    _LeaseHeartbeat,
    _commit_artifact,
    _is_recoverable_batch_error,
    run_persistent_worker,
    run_work_package,
)
from storage_guard import StorageReserveError


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scaling():
    return {
        "partition_tile_rows": 8,
        "partition_tile_cols": 8,
        "partition_halo_px": 256,
        "seam_band_px": 64,
        "max_job_retries": 2,
    }


def _boundary():
    return {
        "enabled": True,
        "mode": "divider_cubic_bspline_adaptive_v2",
        "smoothing_factor": 1.0,
        "curve_sampling_spacing_px": 0.5,
        "max_chord_error_px": 0.25,
        "max_segment_arc_length_px": 8.0,
        "diagnostic_level": "changed_and_failed",
    }


def _single_tile_two_model_run(tmp_path, *, run_id):
    tile = tmp_path / f"{run_id}_source.tif"
    with rasterio.open(
        tile,
        "w",
        driver="GTiff",
        width=512,
        height=512,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 512, 1, 1),
    ) as destination:
        destination.write(np.zeros((3, 512, 512), dtype=np.uint8))
    models = []
    for model_id in ("a", "b"):
        artifact = tmp_path / f"{run_id}_{model_id}.pt"
        artifact.write_bytes(f"model-{model_id}".encode())
        models.append(
            {
                "model_id": model_id,
                "artifact_path": str(artifact),
                "sha256": _sha(artifact),
                "version": "fixture",
            }
        )
    profile = {
        "profile_id": "fixture_fusion",
        "status": "approved",
        "approval": {"passed": True},
        "strategy": "equal_probability_average",
        "models": [{"model_id": "a"}, {"model_id": "b"}],
        "weights": [[0.5, 0.5] for _ in range(14)],
    }
    return create_v5_run(
        state_database=tmp_path / f"{run_id}_state.sqlite",
        output_root=tmp_path / f"{run_id}_output",
        raster={
            "path": tile,
            "crs": "EPSG:3857",
            "transform": [1, 0, 0, 0, -1, 512],
            "nodata": None,
        },
        requested_extent={"xmin": 0, "ymin": 0, "xmax": 512, "ymax": 512},
        processing_extent={"xmin": 0, "ymin": 0, "xmax": 512, "ymax": 512},
        tile_rows=1,
        tile_cols=1,
        tiles=[
            {
                "row": 0,
                "col": 0,
                "path": str(tile),
                "sha256": _sha(tile),
                "pixel_window": {"x0": 0, "y0": 0, "x1": 512, "y1": 512},
            }
        ],
        models=models,
        effective_device="cpu",
        overlap=192,
        scaling=_scaling(),
        boundary_fitting=_boundary(),
        storage_report={
            "package_tile_limit": 4,
            "working_bytes_per_tile": 4096,
            "working_cache_budget_bytes": 64 * 1024 * 1024,
            "status": "passed",
        },
        fusion={"profile_id": "fixture_fusion", "profile": profile},
        run_id=run_id,
    )


def test_unit_polygonize_does_not_emit_deprecated_memory_driver_warning(capfd):
    labels = np.array([[0, 1], [1, 1]], dtype=np.int16)
    records = _polygonize(
        labels,
        {
            "unit_id": "core_00000_00000",
            "pixel_window": {"x0": 0, "y0": 0, "x1": 2, "y1": 2},
        },
        list(range(14)),
    )
    captured = capfd.readouterr()
    assert len(records) == 2
    assert "'Memory' driver is deprecated" not in captured.err


def test_package_filesystem_lock_excludes_a_second_lease_owner(tmp_path):
    lock_path = tmp_path / "tmp" / "package_locks" / "package_00000.lock"
    first = _PackageFileLock(lock_path)
    second = _PackageFileLock(lock_path)
    first.acquire()
    try:
        with pytest.raises(WorkPackageRuntimeError, match="already held"):
            second.acquire()
    finally:
        first.release()
    second.acquire()
    second.release()


def test_batch_downgrade_only_accepts_explicit_capacity_or_accelerator_oom():
    assert _is_recoverable_batch_error(BatchCapacityError("capacity"), "cpu")
    assert _is_recoverable_batch_error(RuntimeError("CUDA out of memory"), "cuda:0")
    assert _is_recoverable_batch_error(RuntimeError("MPS out of memory"), "mps")
    assert not _is_recoverable_batch_error(RuntimeError("bad Tile"), "cuda:0")
    assert not _is_recoverable_batch_error(RuntimeError("out of memory"), "cpu")
    assert not _is_recoverable_batch_error(
        WorkPackageRuntimeError("Tile probability batch shape is wrong"),
        "cuda:0",
    )


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("cuda:0", ["gc", "cuda_sync:cuda:0", "cuda_empty"]),
        ("mps", ["gc", "mps_sync", "mps_empty"]),
    ],
)
def test_accelerator_cache_cleanup_synchronizes_before_release(
    monkeypatch, device, expected
):
    events = []
    cuda_available = device.startswith("cuda")
    mps_available = device.startswith("mps")
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: cuda_available,
            synchronize=lambda value: events.append(f"cuda_sync:{value}"),
            empty_cache=lambda: events.append("cuda_empty"),
        ),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps_available)
        ),
        mps=SimpleNamespace(
            synchronize=lambda: events.append("mps_sync"),
            empty_cache=lambda: events.append("mps_empty"),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        work_package_runtime.gc,
        "collect",
        lambda: events.append("gc"),
    )

    work_package_runtime._clear_accelerator_cache(device)

    assert events == expected


def test_model_boundaries_release_cache_without_unloading_resident_models(
    tmp_path, monkeypatch
):
    _spec, spec_path, _database_path = _single_tile_two_model_run(
        tmp_path,
        run_id="20260809_230000_cache",
    )
    cleanup_calls = []
    loaded = []
    monkeypatch.setattr(
        work_package_runtime,
        "_clear_accelerator_cache",
        cleanup_calls.append,
    )

    def loader(model_entry, _device):
        model_id = str(model_entry["model_id"])
        loaded.append(model_id)
        return model_id

    def infer(model_id, _tile_path, _device):
        probabilities = np.zeros((14, 512, 512), dtype=np.float32)
        probabilities[0 if model_id == "a" else 1] = 1.0
        return probabilities

    report = run_work_package(
        spec_path,
        "package_00000",
        device="cpu",
        model_provider=PersistentModelProvider(loader),
        infer_tile=infer,
    )

    assert report["status"] == "ready"
    assert cleanup_calls == ["cpu", "cpu"]
    assert loaded == ["a", "b"]
    assert [item["cold_load_count"] for item in report["models"]] == [1, 1]


def test_late_second_model_failure_reuses_committed_first_model_outputs(tmp_path):
    spec, spec_path, database_path = _single_tile_two_model_run(
        tmp_path,
        run_id="20260805_190000_a1b2c3",
    )
    database = RunStateDB(database_path)
    package_id = database.page_work_packages(spec["run_id"], limit=1)[0][
        "package_id"
    ]
    loaded = []
    fail_b = {"value": True}

    def loader(model_entry, _device):
        loaded.append(model_entry["model_id"])
        return model_entry["model_id"]

    def infer(model_id, _tile_path, _device):
        if model_id == "b" and fail_b["value"]:
            raise RuntimeError("late model b fault")
        probabilities = np.zeros((14, 512, 512), dtype=np.float32)
        probabilities[0 if model_id == "a" else 1] = 1.0
        return probabilities

    with pytest.raises(RuntimeError, match="late model b fault"):
        run_work_package(
            spec_path,
            package_id,
            device="cpu",
            model_loader=loader,
            infer_tile=infer,
        )
    first_model_paths = [
        spec_path.parent
        / "models/a/raster_parts"
        / f"partition_00000_00000_{suffix}.tif"
        for suffix in ("mask", "confidence")
    ] + [
        spec_path.parent
        / "tmp/probability_parts/a/partition_00000_00000.tif"
    ]
    before = {str(path): _sha(path) for path in first_model_paths}
    assert not (
        spec_path.parent
        / "tmp/work_packages"
        / package_id
        / "score_batches/a"
    ).exists()

    fail_b["value"] = False
    report = run_work_package(
        spec_path,
        package_id,
        device="cpu",
        resume=True,
        model_loader=loader,
        infer_tile=infer,
    )

    assert report["status"] == "ready"
    assert loaded == ["a", "b", "b"]
    assert report["models"][0]["reused_partition_output_count"] == 1
    assert report["models"][0]["cold_load_count"] == 0
    assert {str(path): _sha(path) for path in first_model_paths} == before


def test_linear_work_package_passes_fusion_head_to_partition_finalize(tmp_path):
    spec, spec_path, database_path = _single_tile_two_model_run(
        tmp_path,
        run_id="20260805_190500_linear",
    )
    profile = spec["fusion"]["profile"]
    profile["strategy"] = "linear_1x1"
    profile["models"] = [
        {"model_id": "a", "temperature": 1.0},
        {"model_id": "b", "temperature": 1.0},
    ]
    profile["weights"] = [[0.5, 0.5] for _ in range(14)]
    atomic_write_json(spec_path, spec)

    calls = []

    def fusion_head(features):
        calls.append(tuple(features.shape))
        output = np.zeros(
            (1, 14, features.shape[-2], features.shape[-1]),
            dtype=np.float32,
        )
        output[:, 3] = 1.0
        return output

    def infer(model_id, _tile_path, _device):
        probabilities = np.zeros((14, 512, 512), dtype=np.float32)
        probabilities[0 if model_id == "a" else 1] = 1.0
        return probabilities

    database = RunStateDB(database_path)
    package_id = database.page_work_packages(spec["run_id"], limit=1)[0][
        "package_id"
    ]
    report = run_work_package(
        spec_path,
        package_id,
        device="cpu",
        model_loader=lambda entry, _device: entry["model_id"],
        infer_tile=infer,
        fusion_head=fusion_head,
    )

    assert report["status"] == "ready"
    assert calls == [(1, 28, 512, 512)]
    assert database.artifacts_for_stream(
        spec["run_id"], "fusion:fixture_fusion", kind="partition_probability"
    )


def test_linear_fusion_head_loader_verifies_and_adapts_torchscript(tmp_path):
    import torch

    class FirstModelHead(torch.nn.Module):
        def forward(self, features):
            return features[:, :14]

    head_path = tmp_path / "fixture_head.pt"
    traced = torch.jit.trace(
        FirstModelHead(), torch.zeros((1, 28, 2, 3), dtype=torch.float32)
    )
    traced.save(str(head_path))
    profile_path = tmp_path / "fusion_profile.json"
    profile = {
        "profile_id": "linear_fixture",
        "strategy": "linear_1x1",
        "fusion_head": {
            "artifact": head_path.name,
            "sha256": _sha(head_path),
        },
    }
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    spec = {"fusion": {"profile_path": str(profile_path)}, "models": []}

    loaded_head = work_package_runtime._load_linear_fusion_head(
        spec, profile, "cpu"
    )
    output = loaded_head(np.zeros((1, 28, 2, 3), dtype=np.float32))

    assert tuple(output.shape) == (1, 14, 2, 3)
    assert output.device.type == "cpu"


def test_preflight_failure_after_lease_is_atomically_requeued(
    tmp_path, monkeypatch
):
    spec, spec_path, database_path = _single_tile_two_model_run(
        tmp_path,
        run_id="20260805_194000_d4e5f6",
    )
    database = RunStateDB(database_path)
    job = database.lease_next_work_package(
        spec["run_id"],
        "preflight-failure-worker",
        max_open_frontier_units=64,
        lease_seconds=120,
    )
    assert job is not None
    monkeypatch.setattr(work_package_runtime, "validate_device", lambda _value: False)

    with pytest.raises(WorkPackageRuntimeError, match="device is unavailable"):
        run_work_package(
            spec_path,
            job["package_id"],
            job_id=job["job_id"],
            lease_token=job["lease_token"],
            device="cpu",
        )

    assert database.get_job(job["job_id"])["status"] == "queued"
    assert database.get_work_package(
        spec["run_id"], job["package_id"]
    )["status"] == "queued"


def test_heartbeat_thread_surfaces_database_exception_as_lease_loss(
    tmp_path, monkeypatch
):
    database = RunStateDB(tmp_path / "state.sqlite")
    database.initialize()
    monkeypatch.setattr(
        RunStateDB,
        "heartbeat",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("db unavailable")),
    )
    heartbeat = _LeaseHeartbeat(
        database.path,
        run_id="run",
        package_id="package",
        job_id=1,
        lease_token="token",
        stop_event=threading.Event(),
        interval_sec=0.01,
        lease_seconds=30,
    )
    monkeypatch.setattr(heartbeat, "check", lambda: None)
    heartbeat.start()
    try:
        deadline = time.monotonic() + 1.0
        while not heartbeat.lost_event.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert heartbeat.lost_event.is_set()
        # Restore the real check behavior while retaining the injected loss.
        monkeypatch.undo()
        with pytest.raises(LeaseLostError, match="db unavailable"):
            heartbeat.check()
    finally:
        heartbeat.close()


def test_heartbeat_keeps_lease_alive_after_stop_until_owner_closes(
    tmp_path, monkeypatch
):
    database = RunStateDB(tmp_path / "state.sqlite")
    database.initialize()
    calls = []

    def accepted_heartbeat(*_args, **_kwargs):
        calls.append(time.monotonic())
        return True

    monkeypatch.setattr(RunStateDB, "heartbeat", accepted_heartbeat)
    stopper = threading.Event()
    heartbeat = _LeaseHeartbeat(
        database.path,
        run_id="run",
        package_id="package",
        job_id=1,
        lease_token="token",
        stop_event=stopper,
        interval_sec=0.01,
        lease_seconds=30,
    )
    monkeypatch.setattr(heartbeat, "check", lambda: None)
    heartbeat.start()
    stopper.set()
    try:
        deadline = time.monotonic() + 1.0
        while len(calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(calls) >= 2
    finally:
        heartbeat.close()


def test_lease_loss_fences_persistent_session_before_next_package(
    tmp_path, monkeypatch
):
    spec, spec_path, database_path = _single_tile_two_model_run(
        tmp_path,
        run_id="20260805_194000_fenced",
    )
    database = RunStateDB(database_path)
    database.insert_work_packages(
        spec["run_id"],
        [{"package_id": "package_00001", "sequence_no": 1}],
    )
    database.insert_jobs(
        spec["run_id"],
        [{"job_type": "work_package", "package_id": "package_00001"}],
    )
    calls = []

    def lose_lease(_spec_path, package_id, **_kwargs):
        calls.append(package_id)
        raise LeaseLostError("replacement worker owns the lease")

    monkeypatch.setattr(work_package_runtime, "run_work_package", lose_lease)
    report = run_persistent_worker(
        spec_path,
        "stale-worker",
        device="cpu",
        heartbeat_interval_sec=0.05,
    )

    assert calls == ["package_00000"]
    assert report["session_fenced"] is True
    assert report["package_attempt_count"] == 1
    with sqlite3.connect(database_path) as connection:
        states = dict(
            connection.execute(
                "SELECT package_id, status FROM jobs WHERE job_type='work_package'"
            ).fetchall()
        )
    assert states == {"package_00000": "running", "package_00001": "queued"}


@pytest.mark.parametrize(
    "target_prefix",
    (
        "tile_cache:",
        "score_checkpoint:a:",
        "partition_rasters:model:a:",
        "partition_rasters:fusion:fixture_fusion:",
    ),
)
def test_persistent_worker_waits_for_low_disk_without_consuming_attempt(
    tmp_path, monkeypatch, target_prefix
):
    spec, spec_path, database_path = _single_tile_two_model_run(
        tmp_path,
        run_id="20260805_191000_c4d5e6",
    )
    checks = {"fired": False}
    original_check = work_package_runtime.StorageGuard.check

    def fail_once(self, operation, **kwargs):
        if not checks["fired"] and str(operation).startswith(target_prefix):
            checks["fired"] = True
            raise StorageReserveError(
                operation,
                free_bytes=100,
                required_free_bytes=200,
                write_bytes=int(kwargs.get("write_bytes") or 0),
                managed_bytes=0,
                managed_budget_bytes=0,
            )
        return original_check(self, operation, **kwargs)

    monkeypatch.setattr(work_package_runtime.StorageGuard, "check", fail_once)
    provider = PersistentModelProvider(
        lambda model_entry, _device: model_entry["model_id"]
    )

    def infer(model_id, _tile_path, _device):
        probabilities = np.zeros((14, 512, 512), dtype=np.float32)
        probabilities[0 if model_id == "a" else 1] = 1.0
        return probabilities

    worker_report = run_persistent_worker(
        spec_path,
        "low-disk-worker",
        device="cpu",
        model_provider=provider,
        infer_tile=infer,
        heartbeat_interval_sec=0.01,
        low_disk_poll_sec=0.01,
    )

    database = RunStateDB(database_path)
    package = database.page_work_packages(spec["run_id"], limit=1)[0]
    jobs = database.job_counts(spec["run_id"], job_type="work_package")
    with sqlite3.connect(database_path) as connection:
        attempt = connection.execute(
            "SELECT attempt FROM jobs WHERE job_type='work_package'"
        ).fetchone()[0]
    assert worker_report["status"] == "ready"
    assert checks["fired"] is True
    assert worker_report["low_disk_pause_count"] == 1
    assert package["status"] == "ready"
    assert jobs == {"ready": 1}
    assert attempt == 1


def test_persistent_worker_fails_fixed_managed_budget_instead_of_waiting(
    tmp_path, monkeypatch
):
    spec, spec_path, database_path = _single_tile_two_model_run(
        tmp_path,
        run_id="20260805_193000_b1c2d3",
    )
    original_check = work_package_runtime.StorageGuard.check
    fired = {"value": False}

    def fail_fixed_budget(self, operation, **kwargs):
        if not fired["value"] and str(operation).startswith("tile_cache:"):
            fired["value"] = True
            raise StorageReserveError(
                operation,
                free_bytes=10_000,
                required_free_bytes=1_000,
                write_bytes=int(kwargs.get("write_bytes") or 0),
                managed_bytes=2_000,
                managed_budget_bytes=1_000,
            )
        return original_check(self, operation, **kwargs)

    monkeypatch.setattr(
        work_package_runtime.StorageGuard, "check", fail_fixed_budget
    )
    report = run_persistent_worker(
        spec_path,
        "fixed-budget-worker",
        device="cpu",
        model_provider=PersistentModelProvider(
            lambda model_entry, _device: model_entry["model_id"]
        ),
        low_disk_poll_sec=0.01,
    )

    database = RunStateDB(database_path)
    assert fired["value"] is True
    assert report["status"] == "failed"
    assert report["low_disk_pause_count"] == 0
    assert database.job_counts(spec["run_id"], job_type="work_package") == {
        "failed": 1
    }


def test_keep_score_cache_delays_but_does_not_accumulate_after_package_ready(
    tmp_path,
):
    spec, spec_path, database_path = _single_tile_two_model_run(
        tmp_path,
        run_id="20260805_193500_cache",
    )
    spec["runtime"]["keep_score_cache"] = True
    atomic_write_json(spec_path, spec)

    def infer(model_id, _tile_path, _device):
        probabilities = np.zeros((14, 512, 512), dtype=np.float32)
        probabilities[0 if model_id == "a" else 1] = 1.0
        return probabilities

    report = run_persistent_worker(
        spec_path,
        "delayed-cache-worker",
        device="cpu",
        model_provider=PersistentModelProvider(
            lambda model_entry, _device: model_entry["model_id"]
        ),
        infer_tile=infer,
        heartbeat_interval_sec=0.05,
    )

    assert report["status"] == "ready"
    database = RunStateDB(database_path)
    package_id = database.page_work_packages(spec["run_id"], limit=1)[0][
        "package_id"
    ]
    package_root = spec_path.parent / "tmp" / "work_packages" / package_id
    package_report = json.loads(
        (package_root / "package_report.json").read_text(encoding="utf-8")
    )
    assert package_report["score_cache_retention"] == "until_package_ready"
    assert not (package_root / "score_batches").exists()
    assert not (package_root / "accepted_scores").exists()
    assert not (package_root / "fusion").exists()
    assert not Path(spec["cache_root"]).exists()


def test_fusion_accumulator_separates_atomic_write_from_final_managed_growth(
    tmp_path, monkeypatch
):
    spec, spec_path, database_path = _single_tile_two_model_run(
        tmp_path,
        run_id="20260805_193700_a1b2c3",
    )
    observed = []
    guards = set()
    original_check = work_package_runtime.StorageGuard.check

    def record_check(self, operation, **kwargs):
        guards.add(self)
        report = original_check(self, operation, **kwargs)
        if str(operation).startswith("fusion_accumulator:"):
            observed.append(
                {
                    "operation": str(operation),
                    "write_bytes": int(kwargs.get("write_bytes") or 0),
                    "managed_growth_bytes": int(
                        kwargs.get("managed_growth_bytes") or 0
                    ),
                    "reserved_growth_bytes": int(
                        report["reserved_growth_bytes"]
                    ),
                    "reserved_write_bytes": int(
                        report["reserved_write_bytes"]
                    ),
                }
            )
        return report

    monkeypatch.setattr(
        work_package_runtime.StorageGuard, "check", record_check
    )

    def infer(model_id, _tile_path, _device):
        probabilities = np.zeros((14, 512, 512), dtype=np.float32)
        probabilities[0 if model_id == "a" else 1] = 1.0
        return probabilities

    database = RunStateDB(database_path)
    package_id = database.page_work_packages(spec["run_id"], limit=1)[0][
        "package_id"
    ]
    result = run_work_package(
        spec_path,
        package_id,
        device="cpu",
        model_loader=lambda model_entry, _device: model_entry["model_id"],
        infer_tile=infer,
    )

    assert result["status"] == "ready"
    assert [item["operation"].rsplit(":", 1)[-1] for item in observed] == [
        "a",
        "b",
    ]
    first, second = observed
    assert first["managed_growth_bytes"] == first["write_bytes"]
    assert 0 < second["managed_growth_bytes"] < second["write_bytes"]
    assert all(
        item["reserved_write_bytes"] == item["write_bytes"]
        and item["reserved_growth_bytes"] == item["managed_growth_bytes"]
        for item in observed
    )
    assert guards
    assert all(guard.pending_write_bytes == 0 for guard in guards)


@pytest.mark.parametrize("failure_stage", ("partition", "fusion_accumulator"))
def test_large_writer_failure_always_settles_package_storage_reservation(
    tmp_path, monkeypatch, failure_stage
):
    run_id = {
        "partition": "20260805_193800_a1b2c4",
        "fusion_accumulator": "20260805_193800_a1b2c5",
    }[failure_stage]
    spec, spec_path, database_path = _single_tile_two_model_run(
        tmp_path,
        run_id=run_id,
    )
    guards = set()
    original_check = work_package_runtime.StorageGuard.check

    def record_check(self, operation, **kwargs):
        guards.add(self)
        return original_check(self, operation, **kwargs)

    monkeypatch.setattr(
        work_package_runtime.StorageGuard, "check", record_check
    )
    if failure_stage == "partition":
        monkeypatch.setattr(
            work_package_runtime,
            "write_partition_rasters",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected partition writer failure")
            ),
        )
    else:
        monkeypatch.setattr(
            work_package_runtime.FusionAccumulator,
            "add_model",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected fusion accumulator failure")
            ),
        )

    def infer(model_id, _tile_path, _device):
        probabilities = np.zeros((14, 512, 512), dtype=np.float32)
        probabilities[0 if model_id == "a" else 1] = 1.0
        return probabilities

    database = RunStateDB(database_path)
    package_id = database.page_work_packages(spec["run_id"], limit=1)[0][
        "package_id"
    ]
    with pytest.raises(RuntimeError, match=f"injected {failure_stage.replace('_', ' ')}"):
        run_work_package(
            spec_path,
            package_id,
            device="cpu",
            model_loader=lambda model_entry, _device: model_entry["model_id"],
            infer_tile=infer,
        )

    assert guards
    assert all(guard.pending_write_bytes == 0 for guard in guards)


@pytest.mark.parametrize("residual_status", ("queued", "interrupted", "running"))
def test_persistent_worker_never_reports_residual_jobs_as_ready(
    tmp_path, residual_status
):
    spec, spec_path, database_path = _single_tile_two_model_run(
        tmp_path,
        run_id="20260805_192000_e7f8a9",
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """UPDATE jobs SET status=?, attempt=max_attempts,
               worker_id=CASE WHEN ?='running' THEN 'other-worker' ELSE '' END,
               lease_token=CASE WHEN ?='running' THEN 'other-token' ELSE '' END,
               lease_expires=CASE WHEN ?='running' THEN ? ELSE NULL END
               WHERE job_type='work_package'""",
            (
                residual_status,
                residual_status,
                residual_status,
                residual_status,
                32503680000.0,
            ),
        )
        connection.execute(
            "UPDATE work_packages SET status=? WHERE run_id=?",
            (residual_status, spec["run_id"]),
        )

    report = run_persistent_worker(
        spec_path,
        "nonterminal-worker",
        device="cpu",
        model_provider=PersistentModelProvider(
            lambda model_entry, _device: model_entry["model_id"]
        ),
    )

    assert report["status"] == "incomplete"
    assert report["job_counts"] == {residual_status: 1}
    assert report["package_attempt_count"] == 0


def test_work_package_loads_each_model_once_and_writes_model_and_fusion_parts(
    tmp_path,
    capsys,
    monkeypatch,
):
    tile = tmp_path / "tile_0_0.tif"
    with rasterio.open(
        tile,
        "w",
        driver="GTiff",
        width=512,
        height=512,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 512, 1, 1),
        compress="deflate",
    ) as destination:
        destination.write(np.zeros((3, 512, 512), dtype=np.uint8))
    source_sha256 = _sha(tile)
    model_a = tmp_path / "a.pt"
    model_b = tmp_path / "b.pt"
    model_a.write_bytes(b"model-a")
    model_b.write_bytes(b"model-b")
    profile = {
        "profile_id": "fixture_fusion",
        "status": "approved",
        "approval": {"passed": True},
        "strategy": "equal_probability_average",
        "models": [{"model_id": "a"}, {"model_id": "b"}],
        "weights": [[0.5, 0.5] for _ in range(14)],
    }
    spec, spec_path, database_path = create_v5_run(
        state_database=tmp_path / "state.sqlite",
        output_root=tmp_path / "output",
        raster={
            "path": tile,
            "crs": "EPSG:3857",
            "transform": [1, 0, 0, 0, -1, 512],
            "nodata": None,
        },
        requested_extent={"xmin": 0, "ymin": 0, "xmax": 512, "ymax": 512},
        processing_extent={"xmin": 0, "ymin": 0, "xmax": 512, "ymax": 512},
        tile_rows=1,
        tile_cols=1,
        tiles=[
            {
                "row": 0,
                "col": 0,
                "path": str(tile),
                "sha256": _sha(tile),
                "pixel_window": {"x0": 0, "y0": 0, "x1": 512, "y1": 512},
            }
        ],
        models=[
            {
                "model_id": "a",
                "artifact_path": str(model_a),
                "sha256": _sha(model_a),
                "version": "fixture",
            },
            {
                "model_id": "b",
                "artifact_path": str(model_b),
                "sha256": _sha(model_b),
                "version": "fixture",
            },
        ],
        effective_device="cpu",
        overlap=192,
        scaling=_scaling(),
        boundary_fitting=_boundary(),
        storage_report={
            "package_tile_limit": 4,
            "working_bytes_per_tile": 4096,
            "working_cache_budget_bytes": 64 * 1024 * 1024,
            "status": "passed",
        },
        fragmentation_regularization={
            "enabled": True,
            "policy_id": "fragmentation_v33_configurable_absorption_v1",
            "policy_version": "v33_production_20260826",
            "baseline_policy_id": "semantic_optimized_200_v3",
            "baseline_policy_version": "semantic_optimized_200_v3_core_bounded_v1",
            "buffer_pixels": 256,
            "max_workers": 4,
            "publication": "authoritative_fusion_core",
            "policy_sha256": policy_snapshot_sha256(),
            "executor_sha256": executor_snapshot_sha256(),
        },
        fusion={"profile_id": "fixture_fusion", "profile": profile},
        run_id="20260717_220000_fixture",
    )
    loaded = []

    def loader(model_entry, _device):
        loaded.append(model_entry["model_id"])
        return model_entry["model_id"]

    def infer(model_id, _tile_path, _device):
        probabilities = np.zeros((14, 512, 512), dtype=np.float32)
        probabilities[0 if model_id == "a" else 1] = 1.0
        return probabilities

    with sqlite3.connect(database_path) as connection:
        package_id = connection.execute(
            "SELECT package_id FROM work_packages"
        ).fetchone()[0]
    model_a.write_bytes(b"corrupt-model-a")
    with pytest.raises(WorkPackageRuntimeError, match="model SHA256 mismatch"):
        run_work_package(
            spec_path,
            package_id,
            device="cpu",
            model_loader=loader,
            infer_tile=infer,
        )
    assert tile.is_file()
    assert _sha(tile) == source_sha256
    assert (
        Path(spec["tile_cache_dir"]) / "tile_0_0.tif"
    ).is_file()
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT status FROM work_packages WHERE package_id=?", (package_id,)
        ).fetchone()[0] == "failed"

    model_a.write_bytes(b"model-a")
    original_build_partition_arrays = work_package_runtime.build_partition_arrays
    injected_partition_failures = 0

    def fail_first_partition(*args, **kwargs):
        nonlocal injected_partition_failures
        if injected_partition_failures == 0:
            injected_partition_failures += 1
            raise RuntimeError("partition fault injection")
        return original_build_partition_arrays(*args, **kwargs)

    monkeypatch.setattr(
        work_package_runtime, "build_partition_arrays", fail_first_partition
    )
    with pytest.raises(RuntimeError, match="partition fault injection"):
        run_work_package(
            spec_path,
            package_id,
            device="cpu",
            model_loader=loader,
            infer_tile=infer,
        )
    checkpoint_root = (
        spec_path.parent
        / "tmp"
        / "work_packages"
        / package_id
        / "score_batches"
        / "a"
    )
    assert list(checkpoint_root.glob("batch_*.npy"))
    assert list(checkpoint_root.glob("batch_*.json"))

    result = run_work_package(
        spec_path,
        package_id,
        device="cpu",
        resume=True,
        model_loader=loader,
        infer_tile=infer,
    )
    assert result["status"] == "ready"
    assert tile.is_file()
    assert _sha(tile) == source_sha256
    assert not Path(spec["cache_root"]).exists()
    assert loaded == ["a", "b"]
    run_dir = spec_path.parent
    for relative in (
        "models/a/raster_parts/partition_00000_00000_mask.tif",
        "models/b/raster_parts/partition_00000_00000_mask.tif",
    ):
        assert (run_dir / relative).is_file()
    fusion_mask = run_dir / "fusion/fixture_fusion/raster_parts/partition_00000_00000_mask.tif"
    assert not fusion_mask.exists()
    assert (
        run_dir
        / "tmp/fragmentation_v33_inputs/fusion_fixture_fusion"
        / "partition_00000_00000_v3_baseline.tif"
    ).is_file()
    v33_partition = run_v33_worker(
        spec_path,
        worker_id="integration-v33",
        lease_seconds=120,
    )
    assert v33_partition["status"] == "ready"
    assert v33_partition["stage"] == "partition"
    v33_report = run_v33_worker(
        spec_path,
        worker_id="integration-v33-finalize",
        lease_seconds=120,
    )
    assert v33_report["validation_status"] == "passed"
    assert v33_report["production_replacement"] is True
    with rasterio.open(fusion_mask) as source:
        assert np.all(source.read(1) == 0)
        assert source.tags()["classification_authority"] == (
            "fragmentation_v33_authoritative_fusion_core_v1"
        )
    assert not list((run_dir / "tmp/work_packages" / package_id / "scores").rglob("*.npz"))
    assert not list((run_dir / "tmp/work_packages" / package_id / "scores").rglob("*.json"))
    assert not (run_dir / "tmp/work_packages" / package_id / "score_batches").exists()
    assert not (run_dir / "tmp/work_packages" / package_id / "fusion").exists()
    assert result["cleaned_bytes"] > 0
    assert result["model_load_count"] == 1
    assert result["requested_device"] == "cpu"
    assert result["effective_device"] == "cpu"
    assert result["peak_cache_bytes"] > 0
    assert result["tile_cache_released_count"] == 1
    assert result["tile_cache_retained_count"] == 0
    assert result["peak_rss_bytes"] > 0
    assert result["elapsed_sec"] > 0
    assert result["models"][0]["checkpoint_reused_count"] == 1
    assert result["models"][0]["checkpoint_written_count"] == 0
    assert result["models"][1]["checkpoint_written_count"] == 1
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT status FROM work_packages WHERE package_id=?", (package_id,)
        ).fetchone()[0] == "ready"
        # Partition workers retain two cleaned staging rows for auditability;
        # finalize adds the same three authoritative rows as the former
        # single-job implementation.
        assert connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 15
        assert connection.execute(
            """SELECT COUNT(*) FROM artifacts
               WHERE kind IN ('v33_staged_mask','v33_staged_audit')
                 AND status='cleaned'"""
        ).fetchone()[0] == 2
        unit_job_id = connection.execute(
            """SELECT job_id FROM jobs WHERE stream_id='fusion:fixture_fusion'
               AND job_type='unit_fit'"""
        ).fetchone()[0]
    database = RunStateDB(database_path)
    leased = database.lease_job(unit_job_id, "unit-test", lease_seconds=60)
    assert leased is not None
    unit_report = run_unit_fit(
        spec_path,
        "fusion:fixture_fusion",
        leased["unit_id"],
        job_id=leased["job_id"],
        lease_token=leased["lease_token"],
    )
    assert unit_report["status"] == "passed"
    persisted_unit_report = database.artifact_for_stream_unit(
        spec["run_id"],
        "fusion:fixture_fusion",
        leased["unit_id"],
        "unit_boundary_report",
    )
    with open(persisted_unit_report["path"], encoding="utf-8") as handle:
        persisted_payload = json.load(handle)
    assert "diagnostics" not in persisted_payload
    assert persisted_payload["diagnostic_storage"] == {
        "mode": "none",
        "fitted_edge_count": 0,
        "raw_points_persisted": False,
        "fitted_points_in_json": False,
    }
    assert database.stream_unit_counts(spec["run_id"], "fusion:fixture_fusion") == {
        "ready": 1
    }
    probability = database.artifact_for_stream_unit(
        spec["run_id"],
        "fusion:fixture_fusion",
        "partition_00000_00000",
        "partition_probability",
    )
    assert probability["ref_count"] == 0
    assembled = assemble_stream(spec_path, "fusion:fixture_fusion")
    assert assembled["status"] == "passed"
    assert assembled["assembly_mode"] == "full"
    assert assembled["report_queue_capacity"] == 32
    assert assembled["report_summary_source"] == "run_state_database"
    assert assembled["report_processed_count"] == 1
    assert assembled["report_peak_loaded_count"] == 0
    assert assembled["report_json_parse_count"] == 0
    assert assembled["gpkg_write_mode"] == "pyogrio_arrow_single_publish"
    assert assembled["object_id_resolution"] == "boundary_signature_components_v1"
    assert database.object_link_count(spec["run_id"], "fusion:fixture_fusion") == 0
    events = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("{")
    ]
    completed = [
        event
        for event in events
        if event.get("event") == "report_assembly_completed"
    ]
    assert len(completed) == 1
    assert completed[0]["run_id"] == spec["run_id"]
    assert completed[0]["stream_id"] == "fusion:fixture_fusion"
    assert completed[0]["assembly_mode"] == "full"
    assert completed[0]["current"] == completed[0]["total"] == 1
    assert completed[0]["report_queue_capacity"] == 32
    assert completed[0]["report_peak_loaded_count"] == 0
    assert completed[0]["report_summary_source"] == "run_state_database"
    assert completed[0]["report_json_parse_count"] == 0
    assert completed[0]["summary_validation_peak_in_flight"] >= 1
    assert completed[0]["failed_unit_count"] == 0
    assembly_progress = [
        event for event in events if event.get("event") == "assembly_progress"
    ]
    assert {
        event["phase"] for event in assembly_progress
    } == {
        "validate_inputs",
        "register_objects",
        "link_objects",
        "write_raw",
        "write_formal",
        "aggregate_reports",
        "range_clip",
        "coverage_validation",
        "accepted_difference",
        "publish_cleanup",
    }
    assert assembly_progress[-1]["status"] == "completed"
    persisted_progress = database.monitor_snapshot(spec["run_id"])[
        "stream_runtime_progress"
    ]["fusion:fixture_fusion"]
    assert persisted_progress["phase"] == "publish_cleanup"
    assert persisted_progress["status"] == "completed"
    coverage = database.monitor_snapshot(spec["run_id"])[
        "stream_coverage_validation"
    ]["fusion:fixture_fusion"]
    assert coverage["status"] == "passed"
    assert coverage["hard_gate_applied"] is True
    assert coverage["gap_area_m2"] == pytest.approx(0.0)
    assert assembled["unit_count"] == 1
    assert assembled["object_count"] == 1
    assert assembled["fit_version"] == "divider_cubic_bspline_adaptive_v2"
    assert assembled["chain_count"] >= 0
    assert assembled["shared_chain_count"] >= 0
    assert assembled["spline_count"] >= 0
    assert assembled["topology_checks_performed"] is False
    assert assembled["validation"]["scope"] == "all_output_polygons"
    assert assembled["validation"]["invalid_count"] == 0
    assert (
        spec_path.parent / "fusion/fixture_fusion/semantic_polygons.gpkg"
    ).is_file()
    assert (spec_path.parent / "fusion/fixture_fusion/fitted_edges.gpkg").is_file()
    finalized = finalize_partition_rasters(spec_path)
    assert finalized["status"] == "raster_ready"
    assert len(finalized["streams"]) == 3
    assert all(Path(item["mask_vrt"]).is_file() for item in finalized["streams"])


def test_two_models_multiple_work_packages_complete_fusion_seam_and_assembly(tmp_path):
    tile = tmp_path / "shared_tile.tif"
    with rasterio.open(
        tile,
        "w",
        driver="GTiff",
        width=1472,
        height=832,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 832, 1, 1),
        compress="deflate",
    ) as destination:
        destination.write(np.zeros((3, 832, 1472), dtype=np.uint8))
    model_a = tmp_path / "a.pt"
    model_b = tmp_path / "b.pt"
    model_a.write_bytes(b"model-a")
    model_b.write_bytes(b"model-b")
    profile = {
        "profile_id": "fixture_fusion",
        "status": "approved",
        "approval": {"passed": True},
        "strategy": "equal_probability_average",
        "models": [{"model_id": "a"}, {"model_id": "b"}],
        "weights": [[0.5, 0.5] for _ in range(14)],
    }
    stride = 320
    tiles = [
        {
            "row": row,
            "col": col,
            "path": str(tile),
            "sha256": _sha(tile),
            "bounds": {
                "xmin": col * stride,
                "ymin": 832 - row * stride - 512,
                "xmax": col * stride + 512,
                "ymax": 832 - row * stride,
            },
            "pixel_window": {
                "x0": col * stride,
                "y0": row * stride,
                "x1": col * stride + 512,
                "y1": row * stride + 512,
            },
        }
        for row in range(2)
        for col in range(4)
    ]
    scaling = {
        **_scaling(),
        "partition_tile_rows": 2,
        "partition_tile_cols": 2,
    }
    spec, spec_path, database_path = create_v5_run(
        state_database=tmp_path / "state.sqlite",
        output_root=tmp_path / "output",
        raster={
            "path": tile,
            "crs": "EPSG:3857",
            "transform": [1, 0, 0, 0, -1, 832],
            "nodata": None,
        },
        requested_extent={"xmin": 0, "ymin": 0, "xmax": 1472, "ymax": 832},
        processing_extent={"xmin": 0, "ymin": 0, "xmax": 1472, "ymax": 832},
        tile_rows=2,
        tile_cols=4,
        tiles=tiles,
        models=[
            {
                "model_id": "a",
                "artifact_path": str(model_a),
                "sha256": _sha(model_a),
                "version": "fixture",
            },
            {
                "model_id": "b",
                "artifact_path": str(model_b),
                "sha256": _sha(model_b),
                "version": "fixture",
            },
        ],
        effective_device="cpu",
        tile_batch_size=4,
        resource_tuning={
            "resolved": {"tile_batch_size_by_model": {"a": 4, "b": 2}}
        },
        overlap=192,
        scaling=scaling,
        boundary_fitting=_boundary(),
        storage_report={
            "package_tile_limit": 6,
            "working_bytes_per_tile": 4096,
            "status": "passed",
        },
        fusion={"profile_id": "fixture_fusion", "profile": profile},
        run_id="20260717_221000_multipkg",
    )
    loaded = []

    def loader(model_entry, _device):
        loaded.append(model_entry["model_id"])
        return model_entry["model_id"]

    batch_attempts = []

    def infer_images(model_id, images, _device):
        assert images.shape[1:] == (3, 512, 512)
        batch_attempts.append(len(images))
        if len(images) > 2:
            raise BatchCapacityError("fixture batch capacity is two")
        probabilities = np.zeros(
            (len(images), 14, 512, 512), dtype=np.float32
        )
        probabilities[:, 0 if model_id == "a" else 1] = 1.0
        return probabilities

    database = RunStateDB(database_path)
    with sqlite3.connect(database_path) as connection:
        package_ids = [
            row[0]
            for row in connection.execute(
                "SELECT package_id FROM work_packages ORDER BY sequence_no"
            )
        ]
    assert package_ids == ["package_00000", "package_00001"]
    source_sha256 = _sha(tile)
    first_tile_ids = {
        str(item["tile_id"])
        for item in database.package_tiles(spec["run_id"], package_ids[0])
    }
    second_tile_ids = {
        str(item["tile_id"])
        for item in database.package_tiles(spec["run_id"], package_ids[1])
    }
    shared_tile_ids = first_tile_ids & second_tile_ids
    first_only_tile_ids = first_tile_ids - second_tile_ids
    assert shared_tile_ids
    assert first_only_tile_ids

    provider = PersistentModelProvider(loader)
    worker_report = run_persistent_worker(
        spec_path,
        "multi-package-accelerator",
        device="cpu",
        model_provider=provider,
        infer_images=infer_images,
        heartbeat_interval_sec=0.05,
        low_disk_poll_sec=0.05,
    )
    assert worker_report["status"] == "ready"
    assert worker_report["package_ready_count"] == 2
    first_report = json.loads(
        (
            spec_path.parent
            / "tmp/work_packages"
            / package_ids[0]
            / "package_report.json"
        ).read_text(encoding="utf-8")
    )
    second_report = json.loads(
        (
            spec_path.parent
            / "tmp/work_packages"
            / package_ids[1]
            / "package_report.json"
        ).read_text(encoding="utf-8")
    )
    tile_cache_dir = Path(spec["tile_cache_dir"])
    assert tile.is_file()
    assert _sha(tile) == source_sha256
    for tile_id in first_only_tile_ids:
        row, col = tile_id.split("_")
        assert not (tile_cache_dir / f"tile_{row}_{col}.tif").exists()
    package_reports = [first_report, second_report]
    assert all(report["status"] == "ready" for report in package_reports)
    assert first_report["tile_cache_retained_count"] == len(shared_tile_ids)
    assert second_report["tile_cache_retained_count"] == 0
    assert tile.is_file()
    assert _sha(tile) == source_sha256
    assert not Path(spec["cache_root"]).exists()
    assert loaded == ["a", "b"]
    assert worker_report["model_cold_load_counts"] == {"a": 1, "b": 1}
    assert worker_report["model_cache_hit_counts"] == {"a": 1, "b": 1}
    journal_events = [
        json.loads(line)
        for line in (
            spec_path.parent / "logs" / "accelerator_model_loads.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    for model_id in ("a", "b"):
        events = [
            item["event"]
            for item in journal_events
            if item["model_id"] == model_id
        ]
        assert events == ["load_started", "load_completed", "cache_hit"]
    assert batch_attempts.count(4) == 1
    assert 2 in batch_attempts
    assert worker_report["model_effective_batch_sizes"] == {
        "a@cpu": 2,
        "b@cpu": 2,
    }
    assert all(
        model["configured_tile_batch_size"]
        == {"a": 4, "b": 2}[model["model_id"]]
        and model["effective_tile_batch_size"] == 2
        and model["peak_tile_batch_size"] <= 2
        and model["input_queue_capacity"] == 2
        and model["input_queue_peak_batches"] <= 2
        and model["result_queue_capacity"] == 1
        and model["result_queue_peak_batches"] <= 1
        and model["checkpoint_written_count"] >= 1
        for report in package_reports
        for model in report["models"]
    )
    assert {
        model["model_id"]: model["batch_reduction_count"]
        for model in first_report["models"]
    } == {"a": 1, "b": 0}
    assert all(
        model["batch_reduction_count"] == 0
        for model in second_report["models"]
    )
    assert database.work_package_counts(spec["run_id"]) == {"ready": 2}

    finalized = finalize_partition_rasters(spec_path)
    assert finalized["status"] == "raster_ready"
    assert len(finalized["streams"]) == 3

    with sqlite3.connect(database_path) as connection:
        unit_job_ids = [
            row[0]
            for row in connection.execute(
                "SELECT job_id FROM jobs WHERE job_type='unit_fit' ORDER BY job_id"
            )
        ]
    assert len(unit_job_ids) == 9
    for job_id in unit_job_ids:
        leased = database.lease_job(job_id, "multi-package-test", lease_seconds=120)
        assert leased is not None
        report = run_unit_fit(
            spec_path,
            leased["stream_id"],
            leased["unit_id"],
            job_id=leased["job_id"],
            lease_token=leased["lease_token"],
        )
        assert report["status"] == "passed"

    assembly_reports = [
        assemble_stream(spec_path, stream["stream_id"])
        for stream in spec["streams"]
    ]
    assert all(report["status"] == "passed" for report in assembly_reports)
    assert all(report["unit_count"] == 3 for report in assembly_reports)
    assert all(report["validation"]["passed"] for report in assembly_reports)
    assert all(
        report["unit_artifact_cleanup"]["status"] == "passed"
        for report in assembly_reports
    )
    assert all(
        report["unit_artifact_cleanup"]["artifact_count"] >= 9
        for report in assembly_reports
    )
    assert not list((spec_path.parent / "tmp" / "unit_outputs").rglob("*_*.*"))
    assert all(
        report["validation"]["scope"] == "all_output_polygons"
        for report in assembly_reports
    )
    assert {row["status"] for row in database.stream_rows(spec["run_id"])} == {"ready"}

    scale_report = build_scale_acceptance_report(spec_path)
    assert scale_report["hard_gate_passed"] is True
    assert scale_report["status"] == "passed"
    assert scale_report["package_count"] == 2
    assert scale_report["spatial_unit_count"] == 3
    assert scale_report["model_load_counts"] == {"a": 1, "b": 1}
    assert scale_report["model_cache_hit_counts"] == {"a": 1, "b": 1}
    assert scale_report["model_load_completed_counts"] == {"a": 1, "b": 1}
    assert scale_report["model_load_incomplete_counts"] == {}
    assert scale_report["failed_count"] == 0
    assert scale_report["retry_count"] == 0
    assert scale_report["hard_gates"]["all_unit_intermediates_cleaned"] is True
    assert scale_report["unit_report_summary_count"] == 9
    assert scale_report["unit_artifact_cleanup"]["status_counts"] == {
        "cleaned": scale_report["unit_artifact_cleanup"]["expected_artifact_count"]
    }
    assert scale_report["peak_cache_bytes"] > 0
    assert scale_report["peak_rss_bytes"] > 0
    assert scale_report["cleaned_bytes"] > 0
    assert not scale_report["artifact_integrity_errors"]

    original_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    over_budget_spec = json.loads(json.dumps(original_spec))
    over_budget_spec.setdefault("storage_preflight", {}).update(
        {
            "storage_tuning_schema_version": 2,
            "working_cache_budget_bytes": scale_report["peak_cache_bytes"] - 1,
        }
    )
    atomic_write_json(spec_path, over_budget_spec)
    budget_rejected = build_scale_acceptance_report(spec_path)
    assert budget_rejected["hard_gate_passed"] is False
    assert budget_rejected["status"] == "failed"
    assert (
        budget_rejected["hard_gates"]["peak_cache_within_frozen_budget"]
        is False
    )
    assert budget_rejected["storage"]["cache_budget_gate_applicable"] is True
    atomic_write_json(spec_path, original_spec)

    for relative in (
        "models/a/semantic_polygons.gpkg",
        "models/b/semantic_polygons.gpkg",
        "fusion/fixture_fusion/semantic_polygons.gpkg",
    ):
        assert (spec_path.parent / relative).is_file()

    corrupted = spec_path.parent / "fusion/fixture_fusion/semantic_polygons.gpkg"
    with corrupted.open("ab") as handle:
        handle.write(b"corrupt-after-commit")
    rejected = build_scale_acceptance_report(spec_path)
    assert rejected["hard_gate_passed"] is False
    assert rejected["status"] == "failed"
    assert rejected["hard_gates"]["no_artifact_integrity_errors"] is False
    assert any(
        str(corrupted) in item for item in rejected["artifact_integrity_errors"]
    )


def test_multi_partition_seam_junction_assembly_is_gap_free(tmp_path):
    run_id = "20260717_230000_multi_unit"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state_path = run_dir / "run_state.sqlite"
    class_path = run_dir / "class_mapping_snapshot.json"
    atomic_write_json(
        class_path,
        {
            "class_mapping": {"12": "水浇地"},
            "index_to_code": {"0": 12},
            "background_index": -1,
        },
    )
    plan = plan_spatial_units(
        tile_rows=9,
        tile_cols=9,
        overlap=192,
        partition_tile_rows=8,
        partition_tile_cols=8,
        seam_band_px=64,
        halo_px=192,
    )
    spec = {
        "schema_version": 2,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "state_db": str(state_path),
        "raster": {
            "crs": "EPSG:3857",
            "transform": [1, 0, 0, 0, -1, plan["processing_window"]["y1"]],
        },
        "streams": [
            {
                "stream_id": "model:a",
                "kind": "model",
                "model_id": "a",
                "version": "fixture",
            }
        ],
        "class_mapping_snapshot": str(class_path),
        "accepted_gpkg": "",
        "boundary_fitting": _boundary(),
    }
    spec_path = run_dir / "run_spec.json"
    atomic_write_json(spec_path, spec)
    database = RunStateDB(state_path)
    database.initialize()
    database.create_run(run_id, _sha(spec_path))
    database.register_streams(run_id, spec["streams"])
    database.insert_work_packages(
        run_id, [{"package_id": "package_00000", "sequence_no": 0}]
    )
    database.insert_partitions(
        run_id,
        [
            {**partition, "package_id": "package_00000"}
            for partition in plan["partitions"]
        ],
    )
    database.insert_spatial_units(run_id, plan["spatial_units"])
    database.insert_stream_units(
        run_id,
        ["model:a"],
        [unit["unit_id"] for unit in plan["spatial_units"]],
    )
    affine = Affine(*spec["raster"]["transform"])
    output_root = run_dir / "tmp" / "unit_outputs" / "model_a"
    for unit in plan["spatial_units"]:
        window = unit["pixel_window"]
        geometry = box(window["x0"], window["y0"], window["x1"], window["y1"])
        base = {
            "polygon_id": f"{unit['unit_id']}_0000000",
            "class_code": 12,
            "geometry": geometry,
            "confidence_mean": 1.0,
            "confidence_std": 0.0,
        }
        raw_path = output_root / f"{unit['unit_id']}_raw.parquet"
        formal_path = output_root / f"{unit['unit_id']}_formal.parquet"
        report_path = output_root / f"{unit['unit_id']}_report.json"
        _write_geoparquet(
            raw_path,
            [base],
            transform=affine,
            crs="EPSG:3857",
            include_fit=False,
        )
        _write_geoparquet(
            formal_path,
            [
                {
                    **base,
                    "fit_method": "unchanged",
                    "fit_status": "unchanged",
                    "fit_version": "divider_cubic_bspline_adaptive_v2",
                    "vertex_count_before": 5,
                    "vertex_count_after": 5,
                    "max_shift_px": 0.0,
                    "mean_shift_px": 0.0,
                    "area_change_ratio": 0.0,
                }
            ],
            transform=affine,
            crs="EPSG:3857",
            include_fit=True,
        )
        diagnostics = (
            [
                {
                    "chain_id": "fixture_edge",
                    "method": "cubic_bspline_adaptive",
                    "status": "changed",
                    "fitted_points": [[0.0, 0.0], [1.0, 1.0]],
                    "max_displacement_px": 0.25,
                    "point_count_dense": 17,
                    "point_count_after": 2,
                    "max_chord_error_px": 0.125,
                    "max_segment_arc_length_px": 1.5,
                }
            ]
            if unit["unit_id"] == plan["spatial_units"][0]["unit_id"]
            else []
        )
        unit_report = {
            "status": "passed",
            "fit_version": "divider_cubic_bspline_adaptive_v2",
            "chain_count": 0,
            "shared_chain_count": 0,
            "spline_count": 0,
            "unchanged_count": 1,
            "max_displacement_px": 0.0,
            "diagnostics": diagnostics,
            "validation": {
                "passed": True,
                "scope": "all_output_polygons",
                "invalid_count": 0,
            },
        }
        report_path.write_text(json.dumps(unit_report), encoding="utf-8")
        fitted_edges_path = output_root / f"{unit['unit_id']}_fitted_edges.parquet"
        fitted_edge_count = _write_diagnostic_geoparquet(
            fitted_edges_path,
            unit_report,
            run_id=run_id,
            stream_id="model:a",
            unit_id=unit["unit_id"],
            transform=affine,
            crs="EPSG:3857",
        )
        signature_path = output_root / f"{unit['unit_id']}_boundary_signatures.json"
        write_boundary_signatures(
            signature_path,
            unit_boundary_signatures(
                [
                    {
                        **base,
                        "fit_method": "unchanged",
                        "fit_status": "unchanged",
                        "fit_version": "divider_cubic_bspline_adaptive_v2",
                        "vertex_count_before": 5,
                        "vertex_count_after": 5,
                        "max_shift_px": 0.0,
                        "mean_shift_px": 0.0,
                        "area_change_ratio": 0.0,
                    }
                ],
                stream_id="model:a",
                unit_id=unit["unit_id"],
                pixel_window=unit["pixel_window"],
            ),
            stream_id="model:a",
            unit_id=unit["unit_id"],
        )
        artifacts = [
            ("unit_raw_geoparquet", raw_path),
            ("unit_formal_geoparquet", formal_path),
            ("unit_boundary_report", report_path),
            ("unit_boundary_signatures", signature_path),
        ]
        if fitted_edge_count:
            artifacts.append(("unit_fitted_edges_geoparquet", fitted_edges_path))
        for kind, path in artifacts:
            _commit_artifact(
                database,
                run_id,
                path=path,
                kind=kind,
                stream_id="model:a",
                unit_id=unit["unit_id"],
            )
        database.upsert_unit_report_summary(
            run_id,
            "model:a",
            unit["unit_id"],
            unit_report,
            fitted_edge_count=fitted_edge_count,
        )
        database.set_stream_unit_status(run_id, "model:a", unit["unit_id"], "ready")

    report = assemble_stream(spec_path, "model:a")
    assert report["unit_count"] == 9
    assert report["object_count"] == 1
    assert report["object_link_count"] > 0
    assert report["validation"]["passed"] is True
    assert report["validation"]["scope"] == "all_output_polygons"
    assert report["validation"]["invalid_count"] == 0
    assert report["curve_sampling_spacing_px"] == 0.5
    assert report["max_chord_error_limit_px"] == 0.25
    assert report["max_segment_arc_length_limit_px"] == 8.0
    assert report["dense_curve_point_count"] == 17
    assert report["sparse_curve_point_count"] == 2
    assert report["max_chord_error_px"] == 0.125
    assert report["max_segment_arc_length_px"] == 1.5
    assert report["adaptive_point_reduction"] == pytest.approx(15 / 17)
    with fiona.open(run_dir / "models/a/semantic_polygons.gpkg") as source:
        features = list(source)
    with fiona.open(run_dir / "models/a/fitted_edges.gpkg") as source:
        assert len(source) == 1
    assert len({feature["properties"]["object_id"] for feature in features}) == 1
    merged = unary_union([shape(feature["geometry"]) for feature in features])
    processing = plan["processing_window"]
    expected = box(
        processing["x0"],
        0,
        processing["x1"],
        processing["y1"],
    )
    assert merged.symmetric_difference(expected).area == 0
    reassembled = assemble_stream(spec_path, "model:a")
    assert reassembled["assembly_mode"] == "reused"
    assert reassembled["object_link_count"] == report["object_link_count"]


def test_full_assembly_streams_64_spatial_unit_reports(tmp_path, monkeypatch):
    run_id = "20260729_120000_queue64"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state_path = run_dir / "run_state.sqlite"
    class_path = run_dir / "class_mapping_snapshot.json"
    atomic_write_json(
        class_path,
        {
            "class_mapping": {"12": "水浇地"},
            "index_to_code": {"0": 12},
            "background_index": -1,
        },
    )
    units = [
        {
            "unit_id": f"core_{index:05d}",
            "unit_type": "core",
            "owner_key": f"owner_{index:05d}",
            "pixel_window": {
                "x0": index * 2,
                "y0": 0,
                "x1": index * 2 + 1,
                "y1": 1,
            },
            "dependency_ids": [],
        }
        for index in range(64)
    ]
    spec = {
        "schema_version": 2,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "state_db": str(state_path),
        "raster": {
            "crs": "EPSG:3857",
            "transform": [1, 0, 0, 0, -1, 1],
        },
        "streams": [
            {
                "stream_id": "model:a",
                "kind": "model",
                "model_id": "a",
                "version": "fixture",
            }
        ],
        "class_mapping_snapshot": str(class_path),
        "accepted_gpkg": "",
        "boundary_fitting": _boundary(),
    }
    spec_path = run_dir / "run_spec.json"
    atomic_write_json(spec_path, spec)
    database = RunStateDB(state_path)
    database.initialize()
    database.create_run(run_id, _sha(spec_path))
    database.register_streams(run_id, spec["streams"])
    database.insert_spatial_units(run_id, units)
    database.insert_stream_units(
        run_id,
        ["model:a"],
        [unit["unit_id"] for unit in units],
    )
    affine = Affine(*spec["raster"]["transform"])
    output_root = run_dir / "tmp" / "unit_outputs" / "model_a"

    for unit in units:
        window = unit["pixel_window"]
        geometry = box(
            window["x0"],
            window["y0"],
            window["x1"],
            window["y1"],
        )
        base = {
            "polygon_id": f"{unit['unit_id']}_0000000",
            "class_code": 12,
            "geometry": geometry,
            "confidence_mean": 1.0,
            "confidence_std": 0.0,
        }
        raw_path = output_root / f"{unit['unit_id']}_raw.parquet"
        formal_path = output_root / f"{unit['unit_id']}_formal.parquet"
        report_path = output_root / f"{unit['unit_id']}_report.json"
        _write_geoparquet(
            raw_path,
            [base],
            transform=affine,
            crs="EPSG:3857",
            include_fit=False,
        )
        _write_geoparquet(
            formal_path,
            [
                {
                    **base,
                    "fit_method": "unchanged",
                    "fit_status": "unchanged",
                    "fit_version": "divider_cubic_bspline_adaptive_v2",
                    "vertex_count_before": 5,
                    "vertex_count_after": 5,
                    "max_shift_px": 0.0,
                    "mean_shift_px": 0.0,
                    "area_change_ratio": 0.0,
                }
            ],
            transform=affine,
            crs="EPSG:3857",
            include_fit=True,
        )
        unit_report = {
            "status": "passed",
            "fit_version": "divider_cubic_bspline_adaptive_v2",
            "chain_count": 0,
            "shared_chain_count": 0,
            "spline_count": 0,
            "unchanged_count": 1,
            "max_displacement_px": 0.0,
            "diagnostics": [],
            "validation": {
                "passed": True,
                "scope": "all_output_polygons",
                "invalid_count": 0,
            },
        }
        report_path.write_text(json.dumps(unit_report), encoding="utf-8")
        signature_path = output_root / f"{unit['unit_id']}_boundary_signatures.json"
        write_boundary_signatures(
            signature_path,
            unit_boundary_signatures(
                [
                    {
                        **base,
                        "fit_method": "unchanged",
                        "fit_status": "unchanged",
                        "fit_version": "divider_cubic_bspline_adaptive_v2",
                        "vertex_count_before": 5,
                        "vertex_count_after": 5,
                        "max_shift_px": 0.0,
                        "mean_shift_px": 0.0,
                        "area_change_ratio": 0.0,
                    }
                ],
                stream_id="model:a",
                unit_id=unit["unit_id"],
                pixel_window=unit["pixel_window"],
            ),
            stream_id="model:a",
            unit_id=unit["unit_id"],
        )
        for kind, path in (
            ("unit_raw_geoparquet", raw_path),
            ("unit_formal_geoparquet", formal_path),
            ("unit_boundary_report", report_path),
            ("unit_boundary_signatures", signature_path),
        ):
            _commit_artifact(
                database,
                run_id,
                path=path,
                kind=kind,
                stream_id="model:a",
                unit_id=unit["unit_id"],
            )
        database.upsert_unit_report_summary(
            run_id,
            "model:a",
            unit["unit_id"],
            unit_report,
        )
        database.set_stream_unit_status(
            run_id,
            "model:a",
            unit["unit_id"],
            "ready",
        )

    # Defer only the post-ready cleanup for the first pass so this same fixture
    # can also exercise the report-resume corruption boundary below.  Production
    # assembly always runs the real cleanup immediately after the final outputs
    # are committed and verified.
    real_cleanup = assemble_stream_module._cleanup_stream_unit_artifacts
    monkeypatch.setattr(
        assemble_stream_module,
        "_cleanup_stream_unit_artifacts",
        lambda _spec, _database, stream_id: {
            "status": "deferred_for_resume_test",
            "stream_id": stream_id,
            "artifact_count": 0,
            "cleaned_bytes": 0,
            "kind_counts": {},
        },
    )
    report = assemble_stream(spec_path, "model:a")

    assert report["status"] == "passed"
    assert report["assembly_mode"] == "full"
    assert report["unit_count"] == 64
    assert report["object_count"] == 64
    assert report["report_processed_count"] == 64
    assert report["report_queue_capacity"] == 32
    assert report["report_peak_loaded_count"] == 0
    assert report["report_summary_source"] == "run_state_database"
    assert report["report_json_parse_count"] == 0
    assert report["summary_validation_peak_in_flight"] <= 32
    stream_root = run_dir / "models" / "a"
    raw_path = stream_root / "semantic_polygons_raw.gpkg"
    formal_path = stream_root / "semantic_polygons.gpkg"
    report_path = stream_root / "boundary_fitting_report.json"
    fitted_edges_path = stream_root / "fitted_edges.gpkg"
    with fiona.open(formal_path) as source:
        assert len(source) == 64
    with fiona.open(fitted_edges_path) as source:
        assert len(source) == 0
    assert not list(stream_root.glob(".*.stage.*"))
    assert not list(stream_root.glob(".*.tmp.*"))

    output_hashes = {
        "raw": _sha(raw_path),
        "formal": _sha(formal_path),
        "report": _sha(report_path),
        "fitted_edges": _sha(fitted_edges_path),
    }
    broken_unit_report = output_root / "core_00032_report.json"
    valid_unit_report = broken_unit_report.read_text(encoding="utf-8")
    monkeypatch.setattr(
        assemble_stream_module,
        "_cleanup_stream_unit_artifacts",
        real_cleanup,
    )
    broken_unit_report.write_text("{broken-json", encoding="utf-8")
    database.set_stream_status(
        run_id,
        "model:a",
        "failed",
        error="injected report recovery failure",
    )
    with sqlite3.connect(state_path) as connection:
        connection.execute(
            """UPDATE artifacts SET status='failed'
               WHERE run_id=? AND stream_id=? AND unit_id='assembled'
               AND kind IN ('boundary_fitting_report', 'fitted_edges')""",
            (run_id, "model:a"),
        )

    with pytest.raises(
        StreamAssemblyError,
        match="unit report Artifact changed",
    ):
        assemble_stream(spec_path, "model:a", resume_from_reports=True)

    stream = {
        row["stream_id"]: row for row in database.stream_rows(run_id)
    }["model:a"]
    assert stream["status"] == "failed"
    assert _sha(raw_path) == output_hashes["raw"]
    assert _sha(formal_path) == output_hashes["formal"]
    assert _sha(report_path) == output_hashes["report"]
    assert _sha(fitted_edges_path) == output_hashes["fitted_edges"]
    assert not list(stream_root.glob(".*.stage.*"))
    assert not list(stream_root.glob(".*.tmp.*"))

    broken_unit_report.write_text(valid_unit_report, encoding="utf-8")
    resumed = assemble_stream(spec_path, "model:a", resume_from_reports=True)

    assert resumed["status"] == "passed"
    assert resumed["assembly_mode"] == "report_resume"
    assert resumed["report_processed_count"] == 64
    assert resumed["report_summary_source"] == "run_state_database"
    assert resumed["report_json_parse_count"] == 0
    assert _sha(raw_path) == output_hashes["raw"]
    assert _sha(formal_path) == output_hashes["formal"]
    assert not list(stream_root.glob(".*.stage.*"))
    assert not list(stream_root.glob(".*.tmp.*"))
    reused = assemble_stream(spec_path, "model:a")
    assert reused["assembly_mode"] == "reused"
