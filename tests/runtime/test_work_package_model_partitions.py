"""Model Partition commit and package-local Fusion state boundaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

import loess_runtime.inference.work_package_model_partitions as model_partitions
from labeling_tool.shared.contracts.run_spec import sha256_file
from loess_runtime.inference.incremental_fusion import FusionAccumulator
from loess_runtime.inference.work_package_model_partitions import (
    WorkPackageModelPartitions,
)
from loess_runtime.system.runtime_errors import WorkPackageRuntimeError


def _partition(partition_id: str, x0: int) -> dict:
    window = {"x0": x0, "y0": 0, "x1": x0 + 2, "y1": 2}
    return {
        "partition_id": partition_id,
        "core_window": window,
        "halo_window": window,
    }


def _profile(models: tuple[str, ...]) -> dict:
    return {
        "strategy": "equal_probability_average",
        "models": [{"model_id": model_id} for model_id in models],
    }


class _Artifacts:
    def __init__(self, rows: dict[str, list[dict]] | None = None):
        self.rows = rows or {}

    def artifacts_for_stream(self, _run_id, stream_id, *, kind=None, status="ready"):
        return [
            row
            for row in self.rows.get(stream_id, [])
            if (kind is None or row["kind"] == kind)
            and (status is None or row["status"] == status)
        ]


def _owner(tmp_path, partitions, artifacts, *, profile=None, lease_guard=None):
    return WorkPackageModelPartitions(
        run_id="fixture-run",
        run_dir=tmp_path,
        package_root=tmp_path / "package",
        fusion_id="fixture-fusion",
        partitions=partitions,
        transform=Affine.identity(),
        crs="EPSG:3857",
        range_geometry=None,
        profile=profile,
        artifacts=artifacts,
        lease_guard=lease_guard,
    )


def _raster_artifact(tmp_path, partition, stream_id, kind, data):
    path = tmp_path / stream_id / f"{partition['partition_id']}_{kind}.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=data.shape[0],
        dtype=data.dtype,
        crs="EPSG:3857",
        transform=Affine.translation(partition["core_window"]["x0"], 2)
        * Affine.scale(1, -1),
    ) as destination:
        destination.write(data)
    return {
        "unit_id": partition["partition_id"],
        "kind": kind,
        "status": "ready",
        "path": str(path),
        "byte_count": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _model_artifacts(tmp_path, partitions, stream_id, *, uncovered=None):
    rows = []
    for partition in partitions:
        mask = np.ones((1, 2, 2), dtype=np.int16)
        confidence = np.ones((1, 2, 2), dtype=np.float32)
        probability = np.zeros((14, 2, 2), dtype=np.uint16)
        probability[0] = 65535
        if uncovered == partition["partition_id"]:
            probability[:, 0, 0] = 0
        for kind, data in (
            ("core_mask", mask),
            ("core_confidence", confidence),
            ("partition_probability", probability),
        ):
            rows.append(_raster_artifact(tmp_path, partition, stream_id, kind, data))
    return rows


@pytest.mark.parametrize("broken", ("cleaned", "missing", "invalid_ready"))
def test_fusion_reuse_requires_ready_probability_for_every_partition(
    tmp_path, monkeypatch, broken
):
    partitions = [_partition("p0", 0), _partition("p1", 2)]
    rows = _model_artifacts(tmp_path, partitions, "model:a")
    last_probability = next(
        row
        for row in rows
        if row["unit_id"] == "p1" and row["kind"] == "partition_probability"
    )
    if broken == "cleaned":
        last_probability["status"] = "cleaned"
    elif broken == "missing":
        rows.remove(last_probability)
    else:
        last_probability["sha256"] = "0" * 64
    artifacts = _Artifacts({"model:a": rows})
    monkeypatch.setattr(FusionAccumulator, "completed_model_ids", lambda _self: ("a",))
    owner = _owner(tmp_path, partitions, artifacts, profile=_profile(("a",)))

    assert owner.model_outputs_reusable("model:a", "a") is False
    assert owner._coverage_masks == {}


def test_model_outputs_reuse_cleaned_probability_without_fusion(tmp_path):
    partition = _partition("p0", 0)
    rows = _model_artifacts(tmp_path, [partition], "model:a")
    probability = next(row for row in rows if row["kind"] == "partition_probability")
    probability["status"] = "cleaned"
    owner = _owner(tmp_path, [partition], _Artifacts({"model:a": rows}))

    assert owner.model_outputs_reusable("model:a", "a") is True
    assert owner._coverage_masks == {}


def test_resume_coverage_conflict_keeps_original_pair(tmp_path, monkeypatch):
    partitions = [_partition("p0", 0), _partition("p1", 2)]
    artifacts = _Artifacts(
        {
            "model:a": _model_artifacts(tmp_path, partitions, "model:a"),
            "model:b": _model_artifacts(
                tmp_path, partitions, "model:b", uncovered="p1"
            ),
        }
    )
    monkeypatch.setattr(
        FusionAccumulator, "completed_model_ids", lambda _self: ("a", "b")
    )
    owner = _owner(tmp_path, partitions, artifacts, profile=_profile(("a", "b")))

    assert owner.model_outputs_reusable("model:a", "a") is True
    original = {key: value.copy() for key, value in owner._coverage_masks.items()}
    with pytest.raises(
        WorkPackageRuntimeError, match="coverage differs inside Partition: p1"
    ):
        owner.model_outputs_reusable("model:b", "b")
    assert all(
        np.array_equal(owner._coverage_masks[key], value)
        for key, value in original.items()
    )


class _WorkingStorage:
    def __init__(self, events):
        self.events = events
        self.pending = 0

    def reserve(self, operation, *, write_bytes, managed_growth_bytes):
        assert managed_growth_bytes >= 0
        assert write_bytes >= managed_growth_bytes
        self.events.append(f"reserve-working:{operation}")
        self.pending += 1
        return _Reservation(self, operation)


class _Storage:
    def __init__(self, events):
        self.events = events
        self.pending = 0
        self.working_cache = _WorkingStorage(events)

    def reserve_write(self, operation, _write_bytes, *, managed_growth_bytes):
        assert managed_growth_bytes >= 0
        self.events.append(f"reserve:{operation}")
        self.pending += 1
        return _Reservation(self, operation)

    def mark_permanent_ready(self, _stream_id, _partition_id, kind):
        self.events.append(f"ready:{kind}")


class _Reservation:
    def __init__(self, storage, operation):
        self.storage = storage
        self.operation = operation

    def settle(self, _actual_growth):
        self.storage.events.append(f"settle:{self.operation}")
        self.storage.pending -= 1


def _arrays():
    probabilities = np.zeros((14, 2, 2), dtype=np.float32)
    probabilities[0] = 1.0
    return {
        "halo_probabilities": probabilities,
        "halo_weights": np.ones((2, 2), dtype=np.float32),
        "core_mask": np.ones((2, 2), dtype=np.int16),
        "core_confidence": np.ones((2, 2), dtype=np.float32),
    }


@pytest.mark.parametrize("failure", (None, "writer", "accumulator"))
def test_commit_order_settlement_and_finalize_boundary(tmp_path, monkeypatch, failure):
    events = []
    storage = _Storage(events)
    partition = _partition("p0", 0)
    owner = _owner(
        tmp_path,
        [partition],
        _Artifacts(),
        profile=_profile(("a",)),
        lease_guard=lambda: events.append("lease"),
    )
    monkeypatch.setattr(
        model_partitions,
        "apply_range_mask_to_core",
        lambda arrays, _partition, **_kwargs: (arrays, {}),
    )
    fault = RuntimeError(f"{failure} failed")

    def write(_arrays, _partition, **kwargs):
        events.append("write")
        if failure == "writer":
            raise fault
        paths = {}
        for key in ("probability", "mask", "confidence"):
            path = Path(kwargs[f"output_{key}"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(key.encode())
            paths[key] = str(path)
        return paths

    monkeypatch.setattr(model_partitions, "write_partition_rasters", write)
    monkeypatch.setattr(
        model_partitions,
        "publish_artifact",
        lambda _artifacts, _run_id, **kwargs: events.append(
            f"publish:{kwargs['kind']}"
        ),
    )
    original_add = FusionAccumulator.add_model

    def add(self, model_id, probabilities):
        events.append("add")
        if failure == "accumulator":
            raise fault
        return original_add(self, model_id, probabilities)

    monkeypatch.setattr(FusionAccumulator, "add_model", add)
    if failure is None:
        owner.commit_model_partition("a", "model:a", partition, _arrays(), storage)
    else:
        with pytest.raises(RuntimeError) as caught:
            owner.commit_model_partition("a", "model:a", partition, _arrays(), storage)
        assert caught.value is fault
    assert storage.pending == 0
    assert storage.working_cache.pending == 0
    prefix = [
        "reserve:partition_rasters:model:a:p0",
        "write",
        "settle:partition_rasters:model:a:p0",
    ]
    if failure == "writer":
        assert events == prefix
        return
    assert events == prefix + [
        "lease",
        "publish:core_mask",
        "ready:core_mask",
        "lease",
        "publish:core_confidence",
        "ready:core_confidence",
        "lease",
        "publish:partition_probability",
        "reserve-working:fusion_accumulator:p0:a",
        "add",
        "settle:fusion_accumulator:p0:a",
    ]
    if failure is None:
        before = list(events)
        probabilities, coverage = owner.finalize_partition("p0")
        assert np.all(probabilities[0] == 1.0)
        assert np.all(coverage)
        assert events == before
