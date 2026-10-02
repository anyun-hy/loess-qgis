"""Bounded ordered prefetch for Work Package tile inputs."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

TileItem = dict[str, Any]
TileGroup = tuple[int, list[TileItem]]
TileReadResult = tuple[NDArray[Any], dict[str, Any]]
TileReader = Callable[[Any], TileReadResult]


@dataclass(frozen=True, slots=True)
class PrefetchedTileBatch:
    """One ordered input group stacked by the calling thread."""

    sequence: int
    group: list[TileItem]
    images: NDArray[Any]


class TileReadPrefetcher:
    """Own the bounded read executor, ordered futures, and wait metrics."""

    def __init__(
        self,
        groups: Sequence[TileGroup],
        *,
        capacity: int,
        workers: int,
        read: TileReader,
    ) -> None:
        self._groups = groups
        self._capacity = max(1, int(capacity))
        self._read = read
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(workers)),
            thread_name_prefix="loess-tile-read",
        )
        self._queued: deque[
            tuple[int, list[TileItem], list[Future[TileReadResult]]]
        ] = deque()
        self._cursor = 0
        self._queue_peak = 0
        self._wait_sec = 0.0
        self._started = False
        self._closed = False

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def queue_peak(self) -> int:
        return self._queue_peak

    @property
    def wait_sec(self) -> float:
        return self._wait_sec

    def next_batch(self) -> PrefetchedTileBatch | None:
        """Return the next group after stacking its reads on the caller thread."""
        if self._closed:
            raise RuntimeError("tile read prefetcher is closed")
        if not self._started:
            self._started = True
            self._fill()
        if not self._queued:
            return None

        sequence, group, futures = self._queued.popleft()
        wait_started = time.monotonic()
        images = np.stack([future.result()[0] for future in futures], axis=0)
        # Drop completed results before refilling so the consumed group is not
        # retained as another queued input batch.
        futures.clear()
        self._fill()
        self._wait_sec += time.monotonic() - wait_started
        return PrefetchedTileBatch(sequence, group, images)

    def shutdown(self) -> None:
        """Wait for every submitted read without cancelling queued work."""
        if self._closed:
            return
        self._executor.shutdown(wait=True)
        self._closed = True

    def _fill(self) -> None:
        while len(self._queued) < self._capacity and self._cursor < len(self._groups):
            sequence, group = self._groups[self._cursor]
            futures = [
                self._executor.submit(self._read, item["tile_path"]) for item in group
            ]
            self._queued.append((sequence, group, futures))
            self._cursor += 1
            self._queue_peak = max(self._queue_peak, len(self._queued))
