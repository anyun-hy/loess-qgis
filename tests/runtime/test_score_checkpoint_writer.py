"""Behavioral contracts for single-slot asynchronous score checkpoint writes."""

from __future__ import annotations

from concurrent.futures import Future
from threading import Event, Thread, get_ident

import numpy as np
import pytest

from loess_runtime.inference import score_checkpoint_writer as checkpoint_writer
from loess_runtime.inference.score_batch_cache import load_checkpoint, write_checkpoint
from loess_runtime.inference.score_checkpoint_writer import ScoreCheckpointWriter


def _group() -> list[dict[str, object]]:
    return [
        {
            "tile": {
                "tile_id": "0_0",
                "sha256": "input-0",
                "row_no": 0,
                "col_no": 0,
                "width": 512,
                "height": 512,
            },
            "tile_index": 1,
        }
    ]


def _probabilities() -> np.ndarray:
    probabilities = np.zeros((1, 14, 512, 512), dtype=np.float16)
    probabilities[0, 3] = 1.0
    return probabilities


def test_writer_accepts_only_one_active_write_until_the_result_is_drained():
    write_started = Event()
    release_write = Event()
    writer_thread_ids = []
    caller = get_ident()

    def write(sequence, _group, _probabilities, _managed_cache_bytes):
        writer_thread_ids.append(get_ident())
        write_started.set()
        assert release_write.wait(timeout=5)
        return [{"sequence": sequence}], {"byte_count": 1}

    writer = ScoreCheckpointWriter(write)
    first_group = _group()
    try:
        writer.submit(0, first_group, _probabilities(), 0)
        assert write_started.wait(timeout=5)
        assert len(writer_thread_ids) == 1
        assert writer_thread_ids[0] != caller
        with pytest.raises(RuntimeError, match="already active"):
            writer.submit(1, _group(), _probabilities(), 0)

        release_write.set()
        completed = writer.drain()
        assert completed is not None
        assert completed.group is first_group
        assert completed.records == [{"sequence": 0}]

        writer.submit(1, _group(), _probabilities(), 0)
        assert writer.drain() is not None
    finally:
        release_write.set()
        writer.shutdown()


def test_drain_waits_for_worker_and_returns_the_matching_group_to_its_caller():
    write_started = Event()
    release_write = Event()
    drain_finished = Event()
    drain_thread_ids = []
    completed = []
    group = _group()

    def write(_sequence, _group, _probabilities, _managed_cache_bytes):
        write_started.set()
        assert release_write.wait(timeout=5)
        return [{"record": "ready"}], {"byte_count": 7}

    writer = ScoreCheckpointWriter(write)

    def drain_from_caller():
        drain_thread_ids.append(get_ident())
        completed.append(writer.drain())
        drain_finished.set()

    thread = Thread(target=drain_from_caller)
    try:
        writer.submit(0, group, _probabilities(), 0)
        assert write_started.wait(timeout=5)
        thread.start()
        assert not drain_finished.wait(timeout=0.1)
    finally:
        release_write.set()
        if thread.ident is not None:
            thread.join(timeout=5)
        writer.shutdown()
    assert not thread.is_alive()
    assert drain_finished.is_set()
    assert drain_thread_ids
    assert completed[0] is not None
    assert completed[0].group is group
    assert completed[0].records == [{"record": "ready"}]


def test_writer_propagates_the_original_write_error():
    def write(*_args):
        raise OSError("checkpoint write failed")

    writer = ScoreCheckpointWriter(write)
    try:
        writer.submit(0, _group(), _probabilities(), 0)
        with pytest.raises(OSError, match="checkpoint write failed"):
            writer.drain()
    finally:
        writer.shutdown()


def test_shutdown_waits_for_pending_write_without_losing_its_successful_result():
    write_started = Event()
    release_write = Event()
    shutdown_finished = Event()
    group = _group()

    def write(_sequence, _group, _probabilities, _managed_cache_bytes):
        write_started.set()
        assert release_write.wait(timeout=5)
        return [{"record": "persisted"}], {"byte_count": 11}

    writer = ScoreCheckpointWriter(write)

    def shutdown_writer():
        writer.shutdown()
        shutdown_finished.set()

    thread = Thread(target=shutdown_writer)
    try:
        writer.submit(0, group, _probabilities(), 0)
        assert write_started.wait(timeout=5)
        thread.start()
        assert not shutdown_finished.wait(timeout=0.1)
    finally:
        release_write.set()
        if thread.ident is not None:
            thread.join(timeout=5)
        writer.shutdown()
    assert not thread.is_alive()
    assert shutdown_finished.is_set()
    completed = writer.drain()
    assert completed is not None
    assert completed.group is group
    assert completed.records == [{"record": "persisted"}]


def test_shutdown_does_not_cancel_an_unstarted_submitted_write(monkeypatch):
    executors = []
    writes = []

    class QueuedExecutor:
        def __init__(self, *args, **kwargs):
            self.future = Future()
            self.shutdown_calls = []
            executors.append(self)

        def submit(self, function, *args):
            self.submission = (function, args)
            return self.future

        def shutdown(self, *, wait, **kwargs):
            self.shutdown_calls.append((wait, kwargs))

    monkeypatch.setattr(checkpoint_writer, "ThreadPoolExecutor", QueuedExecutor)

    def write(*args):
        writes.append(args)
        return [], {"byte_count": 0}

    writer = ScoreCheckpointWriter(write)
    group = _group()
    writer.submit(0, group, _probabilities(), 0)
    executor = executors[0]
    writer.shutdown()

    assert executor.shutdown_calls == [(True, {})]
    assert not executor.future.cancelled()
    assert writes == []

    executor.future.set_result(([{"record": "preserved"}], {"byte_count": 1}, 0.0))
    completed = writer.drain()
    assert completed is not None
    assert completed.group is group
    assert completed.records == [{"record": "preserved"}]


def test_writer_persists_and_returns_a_real_checkpoint(tmp_path):
    root = tmp_path / "score_batches" / "model-a"
    group = _group()

    def write(sequence, items, probabilities, managed_cache_bytes):
        return write_checkpoint(
            root,
            run_id="run-a",
            package_id="package-a",
            model_id="model-a",
            model_sha256="model-sha",
            sequence=sequence,
            items=items,
            probabilities=probabilities,
            managed_cache_bytes=managed_cache_bytes,
        )

    writer = ScoreCheckpointWriter(write)
    try:
        writer.submit(3, group, _probabilities(), 0)
        writer.shutdown()
        completed = writer.drain()
        assert completed is not None
        assert completed.group is group
        assert completed.manifest["sequence"] == 3
        assert (
            load_checkpoint(
                root,
                run_id="run-a",
                package_id="package-a",
                model_id="model-a",
                model_sha256="model-sha",
                sequence=3,
                items=group,
            )
            == completed.records
        )
    finally:
        writer.shutdown()
