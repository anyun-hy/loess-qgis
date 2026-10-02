"""Single-slot asynchronous Partition array construction."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from numpy.typing import NDArray

from loess_runtime.inference.partition_mosaic import build_partition_arrays

PartitionRequirement = tuple[Mapping[str, Any], Sequence[str]]
CommitPartition = Callable[
    [Mapping[str, Any], Mapping[str, NDArray[Any]]],
    None,
]


class PartitionBuildPipeline:
    """Own one build future while the caller owns scores and synchronous commit."""

    def __init__(
        self,
        requirements: Sequence[PartitionRequirement],
        *,
        overlap: int,
        commit: CommitPartition,
    ) -> None:
        self._requirements = requirements
        self._overlap = int(overlap)
        self._commit = commit
        self._cursor = 0
        self._future: Future[tuple[dict[str, NDArray[Any]], float]] | None = None
        self._future_entry: PartitionRequirement | None = None
        self._queue_peak = 0
        self._elapsed_sec = 0.0
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="loess-partition",
        )
        self._closed = False

    @property
    def queue_peak(self) -> int:
        return self._queue_peak

    @property
    def elapsed_sec(self) -> float:
        return self._elapsed_sec

    def schedule_ready(self, score_records: Mapping[str, dict[str, Any]]) -> None:
        """Drain a completed build, then submit the next fully scored Partition."""
        self.drain(wait=False)
        if self._future is not None or self._cursor >= len(self._requirements):
            return
        entry = self._requirements[self._cursor]
        partition, required_ids = entry
        if not all(tile_id in score_records for tile_id in required_ids):
            return
        records = [score_records[tile_id] for tile_id in required_ids]
        self._future = self._executor.submit(self._build, partition, records)
        self._future_entry = entry
        self._queue_peak = 1
        self._cursor += 1

    def drain(self, *, wait: bool) -> None:
        """Commit one ready build on the calling thread."""
        if (
            self._future is None
            or self._future_entry is None
            or (not wait and not self._future.done())
        ):
            return
        arrays, build_elapsed = self._future.result()
        partition, _required_ids = self._future_entry
        commit_started = time.monotonic()
        self._commit(partition, arrays)
        self._elapsed_sec += float(build_elapsed) + (time.monotonic() - commit_started)
        self._future = None
        self._future_entry = None

    def finish(
        self,
        score_records: Mapping[str, dict[str, Any]],
    ) -> tuple[Mapping[str, Any], list[str]] | None:
        """Drain all work; return the first incomplete Partition, else ``None``."""
        while self._cursor < len(self._requirements):
            self.schedule_ready(score_records)
            if self._future is None:
                partition, required_ids = self._requirements[self._cursor]
                missing_ids = [
                    tile_id for tile_id in required_ids if tile_id not in score_records
                ]
                return partition, missing_ids
            self.drain(wait=True)
        self.drain(wait=True)
        return None

    def shutdown(self) -> None:
        """Wait for worker exit without committing an undrained result."""
        if self._closed:
            return
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._closed = True

    def _build(
        self,
        partition: Mapping[str, Any],
        records: list[dict[str, Any]],
    ) -> tuple[dict[str, NDArray[Any]], float]:
        started = time.monotonic()
        arrays = build_partition_arrays(
            records,
            partition,
            overlap=self._overlap,
            allow_uncovered=True,
        )
        return arrays, time.monotonic() - started
