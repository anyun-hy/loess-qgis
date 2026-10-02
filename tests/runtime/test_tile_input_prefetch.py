"""Behavioral contracts for bounded ordered tile input prefetching."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, Thread, get_ident
from weakref import ref

import numpy as np
import pytest

from loess_runtime.inference import tile_read_prefetcher as prefetch_runtime
from loess_runtime.inference.tile_read_prefetcher import TileReadPrefetcher


def _group(sequence: int, *paths: str) -> tuple[int, list[dict[str, object]]]:
    return sequence, [{"tile_path": path} for path in paths]


def _result(value: int) -> tuple[np.ndarray, dict[str, object]]:
    return np.full((2, 2), value, dtype=np.float32), {"value": value}


def test_prefetch_is_lazy_bounded_and_releases_consumed_futures_before_refill(
    monkeypatch,
):
    executors = []

    class ReadyFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class ReadyExecutor:
        def __init__(self, *_args, **_kwargs):
            self.submitted_paths = []
            self.future_refs = []
            executors.append(self)

        def submit(self, function, path):
            if path == "c":
                assert self.future_refs[0]() is None
            self.submitted_paths.append(path)
            future = ReadyFuture(function(path))
            self.future_refs.append(ref(future))
            return future

        def shutdown(self, *, wait):
            assert wait is True

    monkeypatch.setattr(prefetch_runtime, "ThreadPoolExecutor", ReadyExecutor)
    prefetcher = TileReadPrefetcher(
        [_group(0, "a"), _group(1, "b"), _group(2, "c")],
        capacity=2,
        workers=1,
        read=lambda path: _result(ord(path) - ord("a")),
    )
    executor = executors[0]
    try:
        assert executor.submitted_paths == []
        first = prefetcher.next_batch()
        assert first is not None
        assert first.sequence == 0
        assert np.array_equal(first.images, np.full((1, 2, 2), 0.0))
        assert executor.submitted_paths == ["a", "b", "c"]
        assert prefetcher.capacity == 2
        assert prefetcher.queue_peak == 2
        assert executor.future_refs[0]() is None

        assert prefetcher.next_batch().sequence == 1
        assert prefetcher.next_batch().sequence == 2
        assert executor.submitted_paths == ["a", "b", "c"]
        assert prefetcher.next_batch() is None
    finally:
        prefetcher.shutdown()


def test_prefetch_returns_groups_in_fifo_order_when_a_later_group_finishes_first(
    monkeypatch,
):
    executors = []

    class ManualExecutor:
        def __init__(self, *_args, **_kwargs):
            self.submissions = []
            self.initial_groups_submitted = Event()
            executors.append(self)

        def submit(self, _function, path):
            future = Future()
            self.submissions.append((path, future))
            if len(self.submissions) == 2:
                self.initial_groups_submitted.set()
            return future

        def shutdown(self, *, wait):
            assert wait is True

    monkeypatch.setattr(prefetch_runtime, "ThreadPoolExecutor", ManualExecutor)
    prefetcher = TileReadPrefetcher(
        [_group(10, "a"), _group(20, "b"), _group(30, "c")],
        capacity=2,
        workers=1,
        read=lambda _path: pytest.fail("manual futures must own completion"),
    )
    executor = executors[0]
    first_batches = []
    first_finished = Event()

    def consume_first():
        first_batches.append(prefetcher.next_batch())
        first_finished.set()

    thread = Thread(target=consume_first)
    try:
        thread.start()
        assert executor.initial_groups_submitted.wait(timeout=5)
        executor.submissions[1][1].set_result(_result(2))
        assert not first_finished.wait(timeout=0.1)

        executor.submissions[0][1].set_result(_result(1))
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert first_batches[0] is not None
        assert first_batches[0].sequence == 10
        assert [path for path, _future in executor.submissions] == ["a", "b", "c"]

        second = prefetcher.next_batch()
        assert second is not None
        assert second.sequence == 20
        executor.submissions[2][1].set_result(_result(3))
        third = prefetcher.next_batch()
        assert third is not None
        assert third.sequence == 30
    finally:
        for _path, future in executor.submissions:
            if not future.done():
                future.set_result(_result(0))
        if thread.ident is not None:
            thread.join(timeout=5)
        prefetcher.shutdown()


def test_prefetch_reads_on_a_worker_and_stacks_images_for_the_caller(monkeypatch):
    read_thread_ids = []
    stack_thread_ids = []
    caller = get_ident()
    original_stack = np.stack

    def stack(*args, **kwargs):
        stack_thread_ids.append(get_ident())
        return original_stack(*args, **kwargs)

    monkeypatch.setattr(prefetch_runtime.np, "stack", stack)

    def read(path):
        read_thread_ids.append(get_ident())
        return _result(int(path))

    prefetcher = TileReadPrefetcher(
        [_group(7, "1", "2")],
        capacity=1,
        workers=1,
        read=read,
    )
    try:
        batch = prefetcher.next_batch()
        assert batch is not None
        assert batch.sequence == 7
        assert len(read_thread_ids) == 2
        assert all(thread_id != caller for thread_id in read_thread_ids)
        assert stack_thread_ids == [caller]
        assert np.array_equal(
            batch.images,
            np.array(
                [
                    [[1.0, 1.0], [1.0, 1.0]],
                    [[2.0, 2.0], [2.0, 2.0]],
                ]
            ),
        )
    finally:
        prefetcher.shutdown()


def test_prefetch_propagates_the_original_read_error():
    def read(_path):
        raise OSError("tile cannot be read")

    prefetcher = TileReadPrefetcher(
        [_group(0, "bad")], capacity=1, workers=1, read=read
    )
    try:
        with pytest.raises(OSError, match="tile cannot be read"):
            prefetcher.next_batch()
    finally:
        prefetcher.shutdown()


def test_shutdown_waits_for_submitted_reads_without_cancelling_them(monkeypatch):
    executors = []
    read_started = Event()
    release_read = Event()
    batch_finished = Event()
    shutdown_finished = Event()
    batches = []

    class RecordingExecutor:
        def __init__(self, *args, **kwargs):
            self._executor = ThreadPoolExecutor(*args, **kwargs)
            self.shutdown_calls = []
            executors.append(self)

        def submit(self, function, *args):
            return self._executor.submit(function, *args)

        def shutdown(self, *, wait, **kwargs):
            self.shutdown_calls.append((wait, kwargs))
            return self._executor.shutdown(wait=wait, **kwargs)

    monkeypatch.setattr(prefetch_runtime, "ThreadPoolExecutor", RecordingExecutor)

    def read(_path):
        read_started.set()
        assert release_read.wait(timeout=5)
        return _result(4)

    prefetcher = TileReadPrefetcher([_group(0, "a")], capacity=1, workers=1, read=read)

    def consume_batch():
        batches.append(prefetcher.next_batch())
        batch_finished.set()

    def shutdown_prefetcher():
        prefetcher.shutdown()
        shutdown_finished.set()

    consume_thread = Thread(target=consume_batch)
    shutdown_thread = Thread(target=shutdown_prefetcher)
    try:
        consume_thread.start()
        assert read_started.wait(timeout=5)
        shutdown_thread.start()
        assert not shutdown_finished.wait(timeout=0.1)
    finally:
        release_read.set()
        if consume_thread.ident is not None:
            consume_thread.join(timeout=5)
        if shutdown_thread.ident is not None:
            shutdown_thread.join(timeout=5)
        prefetcher.shutdown()
    assert not consume_thread.is_alive()
    assert not shutdown_thread.is_alive()
    assert batch_finished.is_set()
    assert shutdown_finished.is_set()
    assert executors[0].shutdown_calls == [(True, {})]
    assert batches[0] is not None
    assert batches[0].sequence == 0
