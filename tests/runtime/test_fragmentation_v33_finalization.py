from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from labeling_tool.shared.contracts.run_spec import CLASS_ORDER
from labeling_tool.shared.state.artifact_repository import ArtifactRepository
from labeling_tool.shared.state.run_state_session import RunStateError
from loess_runtime.geometry import fragmentation_v33_finalization as finalization
from loess_runtime.geometry import fragmentation_v33_work_package as work_package
from loess_runtime.geometry.fragmentation_v33_contract import (
    FragmentationV33WorkPackageError,
)
from loess_runtime.geometry.fragmentation_v33_finalization import (
    PreparedV33Finalization,
    finalize_authoritative_v33,
    prepare_v33_finalization,
)


def _partition(partition_id, x0, x1):
    return {
        "partition_id": partition_id,
        "core_window": {"x0": x0, "y0": 0, "x1": x1, "y1": 1},
    }


class _ArtifactInventory:
    def __init__(self, rows=(), error=None):
        self.rows = list(rows)
        self.error = error
        self.calls = []

    def artifacts_for_stream(self, run_id, stream_id, *, status):
        self.calls.append((run_id, stream_id, status))
        if self.error is not None:
            raise self.error
        return self.rows


@pytest.mark.parametrize(
    "partitions",
    [
        [_partition("left", 0, 1), _partition("right", 2, 3)],
        [_partition("left", 0, 2), _partition("right", 1, 3)],
    ],
)
def test_prepare_rejects_core_gap_or_overlap_before_artifact_inventory(partitions):
    artifacts = _ArtifactInventory()

    with pytest.raises(FragmentationV33WorkPackageError, match="gap or overlap"):
        prepare_v33_finalization(
            cast(ArtifactRepository, artifacts), "run", "fusion:test", partitions
        )

    assert artifacts.calls == []


def test_prepare_returns_only_validated_domain_inputs():
    artifacts = _ArtifactInventory(
        [
            {
                "unit_id": "left",
                "kind": "v33_staged_mask",
                "artifact_id": 7,
            }
        ]
    )

    prepared = prepare_v33_finalization(
        cast(ArtifactRepository, artifacts),
        "run",
        "fusion:test",
        [_partition("left", 0, 1), _partition("right", 1, 3)],
    )

    assert prepared.core_area == prepared.global_area == 3
    assert prepared.overlap_pair_count == 0
    assert tuple(item["partition_id"] for item in prepared.partitions) == (
        "left",
        "right",
    )
    assert prepared.artifacts[("left", "v33_staged_mask")]["artifact_id"] == 7
    assert artifacts.calls == [("run", "fusion:test", "ready")]
    assert not hasattr(prepared, "database")
    assert not hasattr(prepared, "heartbeat")


class _Jobs:
    def __init__(self, events):
        self.events = events

    def finish_job(self, job_id, token, *, status, error):
        self.events.append(("finish", job_id, token, status, error))
        return True

    def requeue_failed_job(self, job_id):
        self.events.append(("requeue", job_id))


class _ControlGraph:
    def __init__(self, partitions):
        self.partitions = partitions

    def get_spatial_unit(self, _run_id, _unit_id):
        return {"unit_type": "FragmentationV33Finalize"}

    def partitions_for_run(self, _run_id):
        return self.partitions


def _job(stream_id="fusion:test"):
    return {
        "job_id": 11,
        "lease_token": "lease-token",
        "stream_id": stream_id,
        "unit_id": "fragmentation_v33_finalize",
    }


def test_finalize_wrapper_leaves_inventory_failure_outside_exception_boundary(
    tmp_path, monkeypatch
):
    events = []
    database = SimpleNamespace(
        control_graph=_ControlGraph([_partition("only", 0, 1)]),
        artifacts=_ArtifactInventory(error=RuntimeError("inventory failed")),
        jobs=_Jobs(events),
    )

    class UnexpectedHeartbeat:
        def __init__(self, *_args, **_kwargs):
            events.append(("heartbeat-created",))

    monkeypatch.setattr(work_package, "_Heartbeat", UnexpectedHeartbeat)

    with pytest.raises(RuntimeError, match="inventory failed"):
        work_package._run_durable_finalize_job(
            {"run_id": "run", "run_dir": str(tmp_path)},
            database,
            _job(),
            lease_seconds=60,
        )

    assert events == []


def test_finalize_wrapper_does_not_close_or_requeue_path_setup_failure(
    tmp_path, monkeypatch
):
    events = []
    prepared = PreparedV33Finalization(
        partitions=(_partition("only", 0, 1),),
        artifacts={},
        core_area=1,
        global_area=1,
        overlap_pair_count=0,
    )
    database = SimpleNamespace(
        control_graph=_ControlGraph(list(prepared.partitions)),
        artifacts=object(),
        jobs=_Jobs(events),
    )

    class RecordingHeartbeat:
        def __init__(self, *_args, **_kwargs):
            events.append(("heartbeat-created",))

        def start(self, total):
            events.append(("start", total))

        def close(self):
            events.append(("close",))

    monkeypatch.setattr(work_package, "_Heartbeat", RecordingHeartbeat)
    monkeypatch.setattr(
        work_package, "prepare_v33_finalization", lambda *_args: prepared
    )

    with pytest.raises(IndexError):
        work_package._run_durable_finalize_job(
            {"run_id": "run", "run_dir": str(tmp_path)},
            database,
            _job(stream_id="malformed"),
            lease_seconds=60,
        )

    assert events == [("heartbeat-created",), ("start", 3)]


def test_finalize_wrapper_requeues_try_body_failure_then_closes(tmp_path, monkeypatch):
    events = []
    prepared = PreparedV33Finalization(
        partitions=(_partition("only", 0, 1),),
        artifacts={},
        core_area=1,
        global_area=1,
        overlap_pair_count=0,
    )
    database = SimpleNamespace(
        control_graph=_ControlGraph(list(prepared.partitions)),
        artifacts=object(),
        jobs=_Jobs(events),
    )

    class RecordingHeartbeat:
        def __init__(self, *_args, **_kwargs):
            events.append(("heartbeat-created",))

        def start(self, total):
            events.append(("start", total))

        def close(self):
            events.append(("close",))

    def fail_finalize(*_args):
        events.append(("finalize",))
        raise RuntimeError("staged validation failed")

    monkeypatch.setattr(work_package, "_Heartbeat", RecordingHeartbeat)
    monkeypatch.setattr(
        work_package, "prepare_v33_finalization", lambda *_args: prepared
    )
    monkeypatch.setattr(work_package, "finalize_authoritative_v33", fail_finalize)

    with pytest.raises(RuntimeError, match="staged validation failed"):
        work_package._run_durable_finalize_job(
            {"run_id": "run", "run_dir": str(tmp_path)},
            database,
            _job(),
            lease_seconds=60,
        )

    assert events[:3] == [
        ("heartbeat-created",),
        ("start", 3),
        ("finalize",),
    ]
    assert events[3][:4] == ("finish", 11, "lease-token", "failed")
    assert events[4:] == [("requeue", 11), ("close",)]


class _CleanupArtifacts:
    def __init__(self, rows_by_id, events):
        self.rows_by_id = rows_by_id
        self.events = events

    def claim_artifact_cleanup(self, artifact_id):
        self.events.append(("claim", artifact_id))
        return dict(self.rows_by_id[artifact_id])

    def finish_artifact_cleanup(self, artifact_id, *, success):
        self.events.append(("cleanup-finish", artifact_id, success))


class _FinalizeStore:
    def __init__(self, artifacts, events, commit_result, attempt_error=None):
        self.artifacts = artifacts
        self.events = events
        self.commit_result = commit_result
        self.attempt_error = attempt_error

    def complete_fragmentation_v33_finalize(self, job_id, token, outputs, **report):
        self.events.append(("commit", job_id, token, len(outputs), report))
        return self.commit_result

    @contextmanager
    def owner_publication(self, run_id, run_dir):
        self.events.append(("publication-enter", run_id, str(run_dir)))
        try:
            yield self
        finally:
            self.events.append(("publication-exit",))

    @contextmanager
    def fragmentation_v33_attempt_commit(self, job_id, token):
        self.events.append(("attempt-enter", job_id, token))
        if self.attempt_error is not None:
            raise self.attempt_error
        try:
            yield self
        finally:
            self.events.append(("attempt-exit",))


class _FinalizeHeartbeat:
    def __init__(self, events, fail_fence_at=None):
        self.events = events
        self.fail_fence_at = fail_fence_at
        self.fence_count = 0

    def progress(self, current):
        self.events.append(("progress", current))

    def fence(self):
        self.fence_count += 1
        self.events.append(("fence",))
        if self.fence_count == self.fail_fence_at:
            raise FragmentationV33WorkPackageError("V3.3 job lease was lost")

    def stop_for_publication(self):
        self.events.append(("publication-stop",))
        self.fence()


def _acceptance():
    return {
        key: 0
        for key in (
            "gap_pixels",
            "overlap_pixels",
            "outside_pixels",
            "invalid_pixels",
            "protected_source_loss_pixel_count",
            "probability_nonfinite_pixels",
            "probability_negative_pixels",
            "probability_zero_sum_pixels",
            "probability_bad_sum_pixels",
        )
    }


def _run_finalizer(
    tmp_path,
    monkeypatch,
    *,
    commit_result,
    gate_passed=True,
    fail_fence_at=None,
    fail_json_name=None,
    fail_replace_at=None,
    attempt_error=None,
):
    events = []
    partition_id = "partition_00000_00000"
    staged_mask = tmp_path / "staged-mask.tif"
    staged_audit = tmp_path / "staged-audit.json"
    baseline = tmp_path / "baseline.tif"
    staged_mask.write_bytes(b"staged-mask")
    staged_audit.write_text("{}", encoding="utf-8")
    baseline.write_bytes(b"baseline")
    artifact_rows = {
        1: {"artifact_id": 1, "path": str(staged_mask), "byte_count": 11},
        2: {"artifact_id": 2, "path": str(staged_audit), "byte_count": 2},
    }
    prepared = PreparedV33Finalization(
        partitions=(_partition(partition_id, 0, 1),),
        artifacts={
            (partition_id, "v33_staged_mask"): artifact_rows[1],
            (partition_id, "v33_staged_audit"): artifact_rows[2],
            (partition_id, "v3_baseline_core"): {
                "artifact_id": 3,
                "path": str(baseline),
            },
        },
        core_area=1,
        global_area=1,
        overlap_pair_count=0,
    )
    cleanup = _CleanupArtifacts(artifact_rows, events)
    store = _FinalizeStore(cleanup, events, commit_result, attempt_error=attempt_error)
    heartbeat = _FinalizeHeartbeat(events, fail_fence_at=fail_fence_at)
    audit = {
        "acceptance": _acceptance(),
        "candidate": {
            "changed_pixel_count": 0,
            "baseline": {"dynamic_fragments_4_connected": 0},
            "result": {"dynamic_fragments_4_connected": 0},
        },
    }

    monkeypatch.setattr(
        finalization,
        "verified_artifact_path",
        lambda artifact, **_kwargs: Path(artifact["path"]),
    )
    monkeypatch.setattr(finalization, "load_json", lambda _path: audit)
    monkeypatch.setattr(
        finalization,
        "physical_metrics",
        lambda *_args: {"pixel_area_m2": 1.0},
    )
    policy = SimpleNamespace(
        class_policies={
            int(code): SimpleNamespace(dynamic_fragmentation_m2=1.0)
            for code in CLASS_ORDER
        }
    )
    monkeypatch.setattr(finalization, "runtime_policy", lambda: policy)

    def connectivity(records, **kwargs):
        events.append(("audit", records[0]["encoding"]))
        kwargs["progress"](1)
        return {
            "components_4_connected": 1,
            "dynamic_fragments_4_connected": 0,
        }

    monkeypatch.setattr(finalization, "audit_partitioned_connectivity", connectivity)
    monkeypatch.setattr(
        finalization,
        "connectivity_hard_gate",
        lambda *_args: {"passed": gate_passed},
    )

    def prepare_copy(_source, destination, key, *, raster_tags):
        events.append(("copy", raster_tags["classification_authority"]))
        destination.parent.mkdir(parents=True, exist_ok=True)
        prepared = destination.with_name(f".{destination.name}.{key}.prepared")
        prepared.write_bytes(b"canonical-mask")
        return prepared

    def write_json(path, value):
        events.append(("json", path.name))
        if path.name == fail_json_name:
            raise OSError(f"injected write failure: {path.name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )

    monkeypatch.setattr(finalization, "_prepare_atomic_copy", prepare_copy)
    monkeypatch.setattr(finalization, "write_atomic_json", write_json)
    replacement_count = 0

    def publish_prepared(source, destination):
        nonlocal replacement_count
        replacement_count += 1
        events.append(("replace", destination.name))
        if replacement_count == fail_replace_at:
            raise OSError(f"injected replace failure: {destination.name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)

    monkeypatch.setattr(finalization, "_publish_prepared", publish_prepared)

    def execute():
        return finalize_authoritative_v33(
            {
                "run_id": "run",
                "run_dir": str(tmp_path),
                "raster": {
                    "transform": [1, 0, 0, 0, -1, 1],
                    "crs": "EPSG:3857",
                },
            },
            store,
            _job(),
            prepared,
            heartbeat,
            tmp_path / "canonical",
        )

    return execute, events


def test_finalizer_commits_all_outputs_before_staged_cleanup(tmp_path, monkeypatch):
    execute, events = _run_finalizer(tmp_path, monkeypatch, commit_result=True)

    report = execute()

    assert report["validation_status"] == "passed"
    assert report["acceptance"]["partition_core_area"] == 1
    assert [event for event in events if event[0] == "audit"] == [
        ("audit", "indices"),
        ("audit", "indices"),
    ]
    commit_index = next(
        index for index, event in enumerate(events) if event[0] == "commit"
    )
    claim_indices = [index for index, event in enumerate(events) if event[0] == "claim"]
    assert claim_indices and commit_index < min(claim_indices)
    assert [event[:2] for event in events if event[0] == "cleanup-finish"] == [
        ("cleanup-finish", 1),
        ("cleanup-finish", 2),
    ]


def test_finalizer_does_not_cleanup_when_authority_commit_fails(tmp_path, monkeypatch):
    execute, events = _run_finalizer(tmp_path, monkeypatch, commit_result=False)

    with pytest.raises(FragmentationV33WorkPackageError, match="lease expired"):
        execute()

    assert any(event[0] == "commit" for event in events)
    assert not any(event[0] in {"claim", "cleanup-finish"} for event in events)


def test_connectivity_gate_failure_prevents_canonical_publication(
    tmp_path, monkeypatch
):
    execute, events = _run_finalizer(
        tmp_path,
        monkeypatch,
        commit_result=True,
        gate_passed=False,
    )

    with pytest.raises(FragmentationV33WorkPackageError, match="hard gate failed"):
        execute()

    assert not any(
        event[0] in {"copy", "json", "commit", "claim", "cleanup-finish"}
        for event in events
    )


def test_precommit_lease_fence_failure_prevents_commit_and_cleanup(
    tmp_path, monkeypatch
):
    execute, events = _run_finalizer(
        tmp_path,
        monkeypatch,
        commit_result=True,
        fail_fence_at=4,
    )

    with pytest.raises(FragmentationV33WorkPackageError, match="lease was lost"):
        execute()

    assert not any(event[0] == "replace" for event in events)
    assert not any(
        event[0] in {"commit", "claim", "cleanup-finish"} for event in events
    )


def test_partial_file_publication_failure_does_not_commit_ready_outputs(
    tmp_path, monkeypatch
):
    execute, events = _run_finalizer(
        tmp_path,
        monkeypatch,
        commit_result=True,
        fail_replace_at=2,
    )

    with pytest.raises(OSError, match="injected replace failure"):
        execute()

    assert any(event[0] == "copy" for event in events)
    assert len([event for event in events if event[0] == "replace"]) == 2
    assert not any(
        event[0] in {"commit", "claim", "cleanup-finish"} for event in events
    )


def test_stale_finalize_lease_is_rejected_before_first_canonical_replace(
    tmp_path, monkeypatch
):
    execute, events = _run_finalizer(
        tmp_path,
        monkeypatch,
        commit_result=True,
        attempt_error=RunStateError(
            "fragmentation V3.3 attempt no longer owns its lease"
        ),
    )

    with pytest.raises(RunStateError, match="no longer owns its lease"):
        execute()

    assert any(event[0] == "publication-enter" for event in events)
    assert any(event[0] == "attempt-enter" for event in events)
    assert not any(event[0] in {"replace", "commit"} for event in events)
