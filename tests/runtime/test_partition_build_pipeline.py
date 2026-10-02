"""Behavioral contracts for bounded asynchronous partition array building."""

from __future__ import annotations

from concurrent.futures import Future
from threading import Event, Thread, get_ident

import numpy as np
import pytest

from loess_runtime.inference import partition_build_pipeline as pipeline_runtime


class _ControlledExecutor:
    """Executor seam that exposes submitted work without starting a worker."""

    def __init__(self):
        self.submissions = []
        self.shutdown_calls = []

    def submit(self, function, *args):
        future = Future()
        self.submissions.append((function, args, future))
        return future

    def shutdown(self, *, wait, cancel_futures):
        self.shutdown_calls.append((wait, cancel_futures))


def _partition(partition_id: str) -> dict[str, object]:
    window = {"x0": 0, "y0": 0, "x1": 4, "y1": 4}
    return {
        "partition_id": partition_id,
        "halo_window": dict(window),
        "core_window": dict(window),
    }


def _record(tile_id: str) -> dict[str, object]:
    probabilities = np.zeros((14, 4, 4), dtype=np.float32)
    probabilities[3] = 1.0
    return {
        "tile_id": tile_id,
        "row": 0,
        "col": 0,
        "probabilities": probabilities,
    }


def test_pipeline_builds_one_ready_partition_and_commits_on_the_caller_thread():
    committed = []
    caller = get_ident()
    pipeline = pipeline_runtime.PartitionBuildPipeline(
        [(_partition("partition-a"), ("tile-a",))],
        overlap=1,
        commit=lambda partition, arrays: committed.append(
            (partition["partition_id"], arrays, get_ident())
        ),
    )
    try:
        pipeline.schedule_ready({"tile-a": _record("tile-a")})
        assert pipeline.finish({"tile-a": _record("tile-a")}) is None
        assert [item[0] for item in committed] == ["partition-a"]
        assert committed[0][2] == caller
        assert np.array_equal(committed[0][1]["core_mask"], np.full((4, 4), 3))
        assert pipeline.queue_peak == 1
        assert pipeline.elapsed_sec >= 0
    finally:
        pipeline.shutdown()


def test_pipeline_requires_head_tiles_then_builds_and_commits_in_requirement_order(
    monkeypatch,
):
    built = []
    committed = []

    def build(records, partition, *, overlap, allow_uncovered):
        assert overlap == 1 and allow_uncovered is True
        built.append((partition["partition_id"], [item["tile_id"] for item in records]))
        return {"partition_id": partition["partition_id"]}

    monkeypatch.setattr(pipeline_runtime, "build_partition_arrays", build)
    first = _partition("partition-a")
    second = _partition("partition-b")
    pipeline = pipeline_runtime.PartitionBuildPipeline(
        [(first, ("tile-a",)), (second, ("tile-b",))],
        overlap=1,
        commit=lambda partition, arrays: committed.append(
            (partition["partition_id"], arrays["partition_id"])
        ),
    )
    scores = {"tile-a": _record("tile-a"), "tile-b": _record("tile-b")}
    try:
        pipeline.schedule_ready({"tile-b": scores["tile-b"]})
        assert built == [] and committed == []
        pipeline.schedule_ready(scores)
        assert pipeline.finish(scores) is None
        assert built == [
            ("partition-a", ["tile-a"]),
            ("partition-b", ["tile-b"]),
        ]
        assert committed == [
            ("partition-a", "partition-a"),
            ("partition-b", "partition-b"),
        ]
        assert pipeline.queue_peak == 1
    finally:
        pipeline.shutdown()


def test_pipeline_submits_only_one_build_until_the_active_future_is_drained(
    monkeypatch,
):
    committed = []
    first = _partition("partition-a")
    second = _partition("partition-b")
    executor = _ControlledExecutor()
    monkeypatch.setattr(
        pipeline_runtime, "ThreadPoolExecutor", lambda **_kwargs: executor
    )
    pipeline = pipeline_runtime.PartitionBuildPipeline(
        [(first, ("tile-a",)), (second, ("tile-b",))],
        overlap=1,
        commit=lambda partition, _arrays: committed.append(partition["partition_id"]),
    )
    scores = {"tile-a": _record("tile-a"), "tile-b": _record("tile-b")}
    try:
        pipeline.schedule_ready(scores)
        pipeline.schedule_ready(scores)
        pipeline.schedule_ready(scores)
        assert len(executor.submissions) == 1
        assert pipeline.queue_peak == 1

        executor.submissions[0][2].set_result(({"built": "partition-a"}, 0.0))
        pipeline.drain(wait=True)
        assert committed == ["partition-a"]

        pipeline.schedule_ready(scores)
        assert len(executor.submissions) == 2
        executor.submissions[1][2].set_result(({"built": "partition-b"}, 0.0))
        pipeline.drain(wait=True)
        assert committed == ["partition-a", "partition-b"]
    finally:
        pipeline.shutdown()


def test_nonblocking_drain_does_not_commit_before_the_worker_completes(monkeypatch):
    build_started = Event()
    release_build = Event()
    committed = []
    build_thread_ids = []
    caller = get_ident()

    def build(_records, _partition, *, overlap, allow_uncovered):
        assert overlap == 1 and allow_uncovered is True
        build_thread_ids.append(get_ident())
        build_started.set()
        assert release_build.wait(timeout=5)
        return {"ready": True}

    monkeypatch.setattr(pipeline_runtime, "build_partition_arrays", build)
    pipeline = pipeline_runtime.PartitionBuildPipeline(
        [(_partition("partition-a"), ("tile-a",))],
        overlap=1,
        commit=lambda _partition, arrays: committed.append(arrays),
    )
    try:
        pipeline.schedule_ready({"tile-a": _record("tile-a")})
        assert build_started.wait(timeout=5)
        assert len(build_thread_ids) == 1
        assert build_thread_ids[0] != caller
        pipeline.drain(wait=False)
        assert committed == []
        release_build.set()
        pipeline.drain(wait=True)
        assert committed == [{"ready": True}]
    finally:
        release_build.set()
        pipeline.shutdown()


def test_finish_reports_missing_head_tiles_without_starting_a_build():
    partition = _partition("partition-a")
    pipeline = pipeline_runtime.PartitionBuildPipeline(
        [(partition, ("tile-a", "tile-b"))],
        overlap=1,
        commit=lambda *_args: pytest.fail("missing input must not commit"),
    )
    try:
        assert pipeline.finish({"tile-a": _record("tile-a")}) == (
            partition,
            ["tile-b"],
        )
        assert pipeline.queue_peak == 0
    finally:
        pipeline.shutdown()


@pytest.mark.parametrize(
    "failure", [RuntimeError("build failed"), RuntimeError("commit failed")]
)
def test_pipeline_propagates_build_and_commit_failures(monkeypatch, failure):
    if str(failure) == "build failed":
        monkeypatch.setattr(
            pipeline_runtime,
            "build_partition_arrays",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
        )

        def commit(*_args):
            pytest.fail("failed build must not commit")
    else:
        monkeypatch.setattr(
            pipeline_runtime,
            "build_partition_arrays",
            lambda *_args, **_kwargs: {"built": True},
        )

        def commit(*_args):
            raise failure

    pipeline = pipeline_runtime.PartitionBuildPipeline(
        [(_partition("partition-a"), ("tile-a",))], overlap=1, commit=commit
    )
    try:
        pipeline.schedule_ready({"tile-a": _record("tile-a")})
        with pytest.raises(RuntimeError, match=str(failure)):
            pipeline.drain(wait=True)
    finally:
        pipeline.shutdown()


def test_shutdown_waits_for_the_build_before_the_caller_cleans_up(monkeypatch):
    build_started = Event()
    release_build = Event()
    build_finished = Event()
    shutdown_finished = Event()
    cleanup_after_worker = []

    def build(_records, _partition, *, overlap, allow_uncovered):
        assert overlap == 1 and allow_uncovered is True
        build_started.set()
        assert release_build.wait(timeout=5)
        build_finished.set()
        return {"unused": True}

    monkeypatch.setattr(pipeline_runtime, "build_partition_arrays", build)
    pipeline = pipeline_runtime.PartitionBuildPipeline(
        [(_partition("partition-a"), ("tile-a",))],
        overlap=1,
        commit=lambda *_args: pytest.fail("shutdown must not commit pending work"),
    )

    def shutdown_then_cleanup():
        pipeline.shutdown()
        cleanup_after_worker.append(build_finished.is_set())
        shutdown_finished.set()

    thread = Thread(target=shutdown_then_cleanup)
    try:
        pipeline.schedule_ready({"tile-a": _record("tile-a")})
        assert build_started.wait(timeout=5)
        thread.start()
        assert not shutdown_finished.wait(timeout=0.1)
    finally:
        release_build.set()
        if thread.ident is not None:
            thread.join(timeout=5)
        pipeline.shutdown()
    assert not thread.is_alive()
    assert shutdown_finished.is_set()
    assert cleanup_after_worker == [True]
