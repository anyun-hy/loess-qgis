import gc
import threading
import weakref
from pathlib import Path

import numpy as np
import pytest

from loess_runtime.inference.model_score_execution import (
    BatchCapacityError,
    build_model_batch_inference,
    execute_model_scores,
    is_recoverable_batch_error,
)
from loess_runtime.system.runtime_errors import WorkPackageRuntimeError


def _items(count, *, start=0):
    return [
        {"tile_path": Path(f"tile_{index}.tif"), "tile_id": str(index)}
        for index in range(start, start + count)
    ]


def _read_tile(path):
    index = int(Path(path).stem.split("_")[-1])
    return np.asarray([index], dtype=np.uint8), {"index": index}


def _thread_ids(prefix):
    return {
        thread.ident
        for thread in threading.enumerate()
        if thread.name.startswith(prefix)
    }


def test_execute_model_scores_retries_capacity_and_consumes_durable_writes_in_order():
    groups = [(0, _items(4)), (1, _items(1, start=4))]
    attempts = []
    lifecycle = []
    write_cache_bytes = []
    persisted = []
    consumed = []

    def infer(group, _images):
        attempts.append([item["tile_id"] for item in group])
        if len(group) > 2:
            raise BatchCapacityError("fixture capacity is two")
        return np.zeros((len(group), 14, 512, 512), dtype=np.float16)

    def write(sequence, group, probabilities, managed_cache_bytes):
        write_cache_bytes.append(managed_cache_bytes)
        assert probabilities.shape == (len(group), 14, 512, 512)
        persisted.append(sequence)
        records = [{"tile_id": item["tile_id"]} for item in group]
        return records, {"byte_count": 100 + sequence, "sequence": sequence}

    def consume(completed):
        sequence = int(completed.manifest["sequence"])
        assert sequence in persisted
        consumed.append(sequence)
        return 111 + sequence

    result = execute_model_scores(
        groups,
        read=_read_tile,
        infer=infer,
        write_checkpoint=write,
        consume_completed=consume,
        is_recoverable_capacity_error=lambda error: is_recoverable_batch_error(
            error, "cpu"
        ),
        remember_batch_size=lambda value: lifecycle.append(f"remember:{value}"),
        clear_accelerator_cache=lambda: lifecycle.append("clear"),
        observe_batch_reduction=lambda attempted, effective, _error: lifecycle.append(
            f"reduce:{attempted}->{effective}"
        ),
        input_workers=2,
        initial_batch_size=4,
        initial_managed_cache_bytes=7,
    )

    assert attempts == [["0", "1", "2", "3"], ["0", "1"], ["2", "3"], ["4"]]
    assert lifecycle[:3] == ["remember:2", "clear", "reduce:4->2"]
    assert lifecycle.count("remember:2") == 3
    assert persisted == consumed == [0, 1]
    assert write_cache_bytes == [7, 111]
    assert result.effective_batch_size == 2
    assert result.peak_batch_size == 2
    assert result.inference_batch_count == 3
    assert result.batch_reduction_count == 1
    assert result.checkpoint_written_count == 2
    assert result.checkpoint_written_bytes == 201
    assert result.input_queue_capacity == 2
    assert result.input_queue_peak_batches == 2
    assert result.result_queue_capacity == 1
    assert result.result_queue_peak_batches == 1
    assert result.input_wait_sec >= 0.0
    assert result.inference_sec >= 0.0
    assert result.checkpoint_write_sec >= 0.0
    assert result.checkpoint_wait_sec >= 0.0


def test_execute_model_scores_does_not_consume_failed_checkpoint_and_closes_workers():
    before_readers = _thread_ids("loess-tile-read")
    before_writers = _thread_ids("loess-score-writer")
    consumed = []

    def write_failed(_sequence, _group, _probabilities, _managed_cache_bytes):
        raise RuntimeError("checkpoint failed")

    with pytest.raises(RuntimeError, match="checkpoint failed"):
        execute_model_scores(
            [(0, _items(1))],
            read=_read_tile,
            infer=lambda group, _images: np.zeros(
                (len(group), 14, 512, 512), dtype=np.float16
            ),
            write_checkpoint=write_failed,
            consume_completed=lambda completed: consumed.append(completed) or 0,
            is_recoverable_capacity_error=lambda _error: False,
            remember_batch_size=None,
            clear_accelerator_cache=lambda: None,
            observe_batch_reduction=lambda _attempted, _effective, _error: None,
            input_workers=1,
            initial_batch_size=1,
            initial_managed_cache_bytes=0,
        )

    assert consumed == []
    assert _thread_ids("loess-tile-read") == before_readers
    assert _thread_ids("loess-score-writer") == before_writers


def test_execute_model_scores_rejects_shape_without_downgrade_and_releases_arrays():
    before_readers = _thread_ids("loess-tile-read")
    before_writers = _thread_ids("loess-score-writer")
    reductions = []
    source_refs = []
    output_refs = []

    def read(path):
        array, profile = _read_tile(path)
        source_refs.append(weakref.ref(array))
        return array, profile

    def infer(_group, _images):
        output = np.zeros((1, 13, 512, 512), dtype=np.float16)
        output_refs.append(weakref.ref(output))
        return output

    with pytest.raises(WorkPackageRuntimeError, match="probability batch shape"):
        execute_model_scores(
            [(0, _items(1))],
            read=read,
            infer=infer,
            write_checkpoint=lambda *_args: ([], {"byte_count": 0}),
            consume_completed=lambda _completed: 0,
            is_recoverable_capacity_error=lambda _error: False,
            remember_batch_size=None,
            clear_accelerator_cache=lambda: None,
            observe_batch_reduction=lambda attempted,
            effective,
            error: reductions.append((attempted, effective, error)),
            input_workers=1,
            initial_batch_size=1,
            initial_managed_cache_bytes=0,
        )

    gc.collect()
    assert reductions == []
    assert all(reference() is None for reference in source_refs + output_refs)
    assert _thread_ids("loess-tile-read") == before_readers
    assert _thread_ids("loess-score-writer") == before_writers


def test_model_batch_inference_adapter_preserves_api_precedence():
    group = _items(2)
    images = np.zeros((2, 3, 1, 1), dtype=np.float32)
    calls = []

    def images_infer(model, values, device):
        calls.append(("images", model, len(values), device))
        return np.asarray([1])

    def paths_infer(model, paths, device):
        calls.append(("paths", model, len(paths), device))
        return np.asarray([2])

    def tile_infer(model, path, device):
        calls.append(("tile", model, path.name, device))
        return np.asarray([3])

    def production_infer(model, values, device):
        calls.append(("production", model, len(values), device))
        return np.asarray([4])

    adapters = [
        build_model_batch_inference(
            "model",
            "cpu",
            infer_images=images_infer,
            infer_batch=paths_infer,
            infer_tile=tile_infer,
            production_batch_inference=production_infer,
        ),
        build_model_batch_inference(
            "model",
            "cpu",
            infer_images=None,
            infer_batch=paths_infer,
            infer_tile=tile_infer,
            production_batch_inference=production_infer,
        ),
        build_model_batch_inference(
            "model",
            "cpu",
            infer_images=None,
            infer_batch=None,
            infer_tile=tile_infer,
            production_batch_inference=production_infer,
        ),
        build_model_batch_inference(
            "model",
            "cpu",
            infer_images=None,
            infer_batch=None,
            infer_tile=None,
            production_batch_inference=production_infer,
        ),
    ]

    for adapter in adapters:
        adapter(group, images)

    assert [call[0] for call in calls] == [
        "images",
        "paths",
        "tile",
        "tile",
        "production",
    ]
