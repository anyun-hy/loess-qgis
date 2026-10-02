"""Single-slot asynchronous writer for score checkpoints."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from numpy.typing import NDArray

ScoreItem = dict[str, Any]
CheckpointWrite = Callable[
    [int, list[ScoreItem], NDArray[Any], int],
    tuple[list[ScoreItem], Mapping[str, Any]],
]


@dataclass(frozen=True, slots=True)
class CompletedCheckpoint:
    """One persisted batch ready for caller-owned score registration."""

    group: list[ScoreItem]
    records: list[ScoreItem]
    manifest: Mapping[str, Any]


class ScoreCheckpointWriter:
    """Own one checkpoint write future and its queue/timing metrics."""

    def __init__(self, write: CheckpointWrite) -> None:
        self._write = write
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="loess-score-writer",
        )
        self._future: (
            Future[tuple[list[ScoreItem], Mapping[str, Any], float]] | None
        ) = None
        self._group: list[ScoreItem] | None = None
        self._written_count = 0
        self._written_bytes = 0
        self._write_sec = 0.0
        self._wait_sec = 0.0
        self._queue_peak = 0
        self._closed = False

    @property
    def written_count(self) -> int:
        return self._written_count

    @property
    def written_bytes(self) -> int:
        return self._written_bytes

    @property
    def write_sec(self) -> float:
        return self._write_sec

    @property
    def wait_sec(self) -> float:
        return self._wait_sec

    @property
    def queue_peak(self) -> int:
        return self._queue_peak

    def submit(
        self,
        sequence: int,
        group: list[ScoreItem],
        probabilities: NDArray[Any],
        managed_cache_bytes: int,
    ) -> None:
        """Submit one batch; the previous result must already be drained."""
        if self._closed:
            raise RuntimeError("score checkpoint writer is closed")
        if self._future is not None:
            raise RuntimeError("score checkpoint write is already active")
        future = self._executor.submit(
            self._write_timed,
            int(sequence),
            group,
            probabilities,
            int(managed_cache_bytes),
        )
        self._group = group
        self._future = future
        self._queue_peak = 1

    def drain(self) -> CompletedCheckpoint | None:
        """Wait for and return the active write result on the calling thread."""
        if self._future is None or self._group is None:
            return None
        wait_started = time.monotonic()
        records, manifest, write_elapsed = self._future.result()
        self._wait_sec += time.monotonic() - wait_started
        self._write_sec += float(write_elapsed)
        self._written_count += 1
        self._written_bytes += int(manifest["byte_count"])
        completed = CompletedCheckpoint(self._group, records, manifest)
        self._future = None
        self._group = None
        return completed

    def shutdown(self) -> None:
        """Wait for worker exit without consuming an undrained result."""
        if self._closed:
            return
        self._executor.shutdown(wait=True)
        self._closed = True

    def _write_timed(
        self,
        sequence: int,
        group: list[ScoreItem],
        probabilities: NDArray[Any],
        managed_cache_bytes: int,
    ) -> tuple[list[ScoreItem], Mapping[str, Any], float]:
        started = time.monotonic()
        records, manifest = self._write(
            sequence,
            group,
            probabilities,
            managed_cache_bytes,
        )
        return records, manifest, time.monotonic() - started
