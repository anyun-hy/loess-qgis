"""Execute one model's missing score groups with bounded pipeline resources."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from loess_runtime.inference.score_checkpoint_writer import (
    CheckpointWrite,
    CompletedCheckpoint,
    ScoreCheckpointWriter,
)
from loess_runtime.inference.tile_read_prefetcher import (
    TileGroup,
    TileItem,
    TileReader,
    TileReadPrefetcher,
)
from loess_runtime.system.runtime_errors import WorkPackageRuntimeError

BatchInference = Callable[[list[TileItem], NDArray[Any]], NDArray[Any]]
CheckpointCompleted = Callable[[CompletedCheckpoint], int]
CapacityErrorPredicate = Callable[[BaseException], bool]
BatchSizeRememberer = Callable[[int], None]
BatchReductionObserver = Callable[[int, int, BaseException], None]
ImageBatchInference = Callable[[Any, NDArray[Any], str], NDArray[Any]]
PathBatchInference = Callable[[Any, list[Path], str], NDArray[Any]]
TileInference = Callable[[Any, Path, str], NDArray[Any]]


class BatchCapacityError(WorkPackageRuntimeError):
    """A tested accelerator batch is too large for the active model/device."""


def is_recoverable_batch_error(error: BaseException, device: str) -> bool:
    """Return whether the same inference may be retried with a smaller batch."""

    if isinstance(error, BatchCapacityError):
        return True
    accelerator = str(device).lower()
    if not (accelerator.startswith("cuda") or accelerator.startswith("mps")):
        return False
    class_name = type(error).__name__.lower()
    message = str(error).lower()
    return "outofmemory" in class_name or "out of memory" in message


def build_model_batch_inference(
    model: Any,
    device: str,
    *,
    infer_images: ImageBatchInference | None,
    infer_batch: PathBatchInference | None,
    infer_tile: TileInference | None,
    production_batch_inference: ImageBatchInference,
) -> BatchInference:
    """Select the frozen model inference API precedence for one execution."""

    if infer_images is not None:

        def infer_from_images(
            _group: list[TileItem], images: NDArray[Any]
        ) -> NDArray[Any]:
            return infer_images(model, images, device)

        return infer_from_images
    if infer_batch is not None:

        def infer_from_paths(
            group: list[TileItem], _images: NDArray[Any]
        ) -> NDArray[Any]:
            return infer_batch(
                model,
                [Path(item["tile_path"]) for item in group],
                device,
            )

        return infer_from_paths
    if infer_tile is not None:

        def infer_from_tiles(
            group: list[TileItem], _images: NDArray[Any]
        ) -> NDArray[Any]:
            return np.stack(
                [infer_tile(model, Path(item["tile_path"]), device) for item in group],
                axis=0,
            )

        return infer_from_tiles

    def infer_production_batch(
        _group: list[TileItem], images: NDArray[Any]
    ) -> NDArray[Any]:
        return production_batch_inference(model, images, device)

    return infer_production_batch


@dataclass(frozen=True, slots=True)
class ModelScoreExecutionResult:
    """Batch, queue, checkpoint, and timing metrics from one model execution."""

    effective_batch_size: int
    peak_batch_size: int
    inference_batch_count: int
    batch_reduction_count: int
    checkpoint_written_count: int
    checkpoint_written_bytes: int
    input_queue_capacity: int
    input_queue_peak_batches: int
    result_queue_capacity: int
    result_queue_peak_batches: int
    input_wait_sec: float
    inference_sec: float
    checkpoint_write_sec: float
    checkpoint_wait_sec: float


def execute_model_scores(
    groups: Sequence[TileGroup],
    *,
    read: TileReader,
    infer: BatchInference,
    write_checkpoint: CheckpointWrite,
    consume_completed: CheckpointCompleted,
    is_recoverable_capacity_error: CapacityErrorPredicate,
    remember_batch_size: BatchSizeRememberer | None,
    clear_accelerator_cache: Callable[[], None],
    observe_batch_reduction: BatchReductionObserver,
    input_workers: int,
    initial_batch_size: int,
    initial_managed_cache_bytes: int,
    input_queue_capacity: int = 2,
) -> ModelScoreExecutionResult:
    """Run missing score groups while preserving ordered durable completion."""

    effective_batch_size = max(1, int(initial_batch_size))
    managed_cache_bytes = max(0, int(initial_managed_cache_bytes))
    inference_batch_count = 0
    peak_batch_size = 0
    batch_reduction_count = 0
    inference_sec = 0.0
    checkpoint_writer = ScoreCheckpointWriter(write_checkpoint)
    try:
        tile_prefetcher = TileReadPrefetcher(
            groups,
            capacity=input_queue_capacity,
            workers=input_workers,
            read=read,
        )
        try:
            while True:
                prefetched = tile_prefetcher.next_batch()
                if prefetched is None:
                    break
                sequence = prefetched.sequence
                group = prefetched.group
                images = prefetched.images
                del prefetched
                group_outputs: list[NDArray[Any]] = []
                cursor = 0
                while cursor < len(group):
                    attempt_size = min(effective_batch_size, len(group) - cursor)
                    subgroup = group[cursor : cursor + attempt_size]
                    inference_started = time.monotonic()
                    try:
                        output = infer(
                            subgroup,
                            images[cursor : cursor + attempt_size],
                        )
                        probabilities_batch = np.asarray(output)
                        expected_shape = (attempt_size, 14, 512, 512)
                        if probabilities_batch.shape != expected_shape:
                            raise WorkPackageRuntimeError(
                                "Tile probability batch shape must be "
                                f"{expected_shape}, got {probabilities_batch.shape}"
                            )
                    except Exception as error:
                        inference_sec += time.monotonic() - inference_started
                        if attempt_size <= 1 or not is_recoverable_capacity_error(
                            error
                        ):
                            raise
                        effective_batch_size = min(
                            effective_batch_size,
                            max(1, attempt_size // 2),
                        )
                        batch_reduction_count += 1
                        if remember_batch_size is not None:
                            remember_batch_size(effective_batch_size)
                        clear_accelerator_cache()
                        observe_batch_reduction(
                            attempt_size,
                            effective_batch_size,
                            error,
                        )
                        continue
                    inference_sec += time.monotonic() - inference_started
                    inference_batch_count += 1
                    peak_batch_size = max(peak_batch_size, attempt_size)
                    if (
                        remember_batch_size is not None
                        and batch_reduction_count > 0
                        and attempt_size == effective_batch_size
                    ):
                        remember_batch_size(effective_batch_size)
                    group_outputs.append(
                        np.asarray(probabilities_batch, dtype=np.float16)
                    )
                    cursor += attempt_size
                    del probabilities_batch, output
                probabilities = np.concatenate(group_outputs, axis=0)
                completed = checkpoint_writer.drain()
                if completed is not None:
                    managed_cache_bytes = max(
                        0,
                        int(consume_completed(completed)),
                    )
                checkpoint_writer.submit(
                    sequence,
                    group,
                    probabilities,
                    managed_cache_bytes,
                )
                del images, group_outputs, probabilities
            completed = checkpoint_writer.drain()
            if completed is not None:
                consume_completed(completed)
        finally:
            tile_prefetcher.shutdown()
    finally:
        checkpoint_writer.shutdown()

    return ModelScoreExecutionResult(
        effective_batch_size=effective_batch_size,
        peak_batch_size=peak_batch_size,
        inference_batch_count=inference_batch_count,
        batch_reduction_count=batch_reduction_count,
        checkpoint_written_count=checkpoint_writer.written_count,
        checkpoint_written_bytes=checkpoint_writer.written_bytes,
        input_queue_capacity=tile_prefetcher.capacity,
        input_queue_peak_batches=tile_prefetcher.queue_peak,
        result_queue_capacity=1,
        result_queue_peak_batches=checkpoint_writer.queue_peak,
        input_wait_sec=tile_prefetcher.wait_sec,
        inference_sec=inference_sec,
        checkpoint_write_sec=checkpoint_writer.write_sec,
        checkpoint_wait_sec=checkpoint_writer.wait_sec,
    )
