from types import SimpleNamespace

import numpy as np
import pytest

import score_batch_cache
from partition_mosaic import blend_probability_tiles
from score_batch_cache import (
    ScoreBatchCacheError,
    ScoreBatchDiskReserveError,
    discard_checkpoint,
    load_checkpoint,
    remove_owned_temporary_files,
    write_checkpoint,
)


def _items(count=1):
    return [
        {
            "tile": {
                "tile_id": f"0_{index}",
                "sha256": f"input-{index}",
                "row_no": 0,
                "col_no": index,
                "width": 512,
                "height": 512,
            },
            "tile_index": index + 1,
        }
        for index in range(count)
    ]


def _probabilities(count=1):
    result = np.zeros((count, 14, 512, 512), dtype=np.float16)
    for index in range(count):
        result[index, index % 14] = 1.0
    return result


def _load(root, items):
    return load_checkpoint(
        root,
        run_id="run-a",
        package_id="package-a",
        model_id="model-a",
        model_sha256="model-sha",
        sequence=0,
        items=items,
    )


def test_batch_checkpoint_round_trip_and_partition_matches_inline(tmp_path):
    root = tmp_path / "score_batches" / "model-a"
    items = _items()
    probabilities = _probabilities()
    records, manifest = write_checkpoint(
        root,
        run_id="run-a",
        package_id="package-a",
        model_id="model-a",
        model_sha256="model-sha",
        sequence=0,
        items=items,
        probabilities=probabilities,
    )
    assert manifest["shape"] == [1, 14, 512, 512]
    assert manifest["dtype"] == "float16"
    assert manifest["byte_count"] == (root / manifest["data_file"]).stat().st_size
    assert _load(root, items) == records

    target = {"x0": 0, "y0": 0, "x1": 512, "y1": 512}
    from_batch, batch_weights = blend_probability_tiles(
        records, target_window=target, overlap=128
    )
    inline, inline_weights = blend_probability_tiles(
        [{"row": 0, "col": 0, "probabilities": probabilities[0]}],
        target_window=target,
        overlap=128,
    )
    assert np.array_equal(from_batch, inline)
    assert np.array_equal(batch_weights, inline_weights)


def test_corrupt_or_incomplete_checkpoint_is_not_reused(tmp_path, monkeypatch):
    root = tmp_path / "score_batches" / "model-a"
    items = _items()
    _records, manifest = write_checkpoint(
        root,
        run_id="run-a",
        package_id="package-a",
        model_id="model-a",
        model_sha256="model-sha",
        sequence=0,
        items=items,
        probabilities=_probabilities(),
    )
    data_path = root / manifest["data_file"]
    with data_path.open("r+b") as handle:
        handle.seek(-1, 2)
        original = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([original[0] ^ 1]))
    assert _load(root, items) is None
    assert discard_checkpoint(root, 0) > 0
    assert not data_path.exists()

    def fail_manifest(*_args, **_kwargs):
        raise OSError("manifest fault injection")

    monkeypatch.setattr(score_batch_cache, "_atomic_json", fail_manifest)
    with pytest.raises(OSError, match="manifest fault injection"):
        write_checkpoint(
            root,
            run_id="run-a",
            package_id="package-a",
            model_id="model-a",
            model_sha256="model-sha",
            sequence=0,
            items=items,
            probabilities=_probabilities(),
        )
    assert _load(root, items) is None


def test_checkpoint_honors_disk_reserve_and_only_cleans_owned_temporary_files(
    tmp_path, monkeypatch
):
    root = tmp_path / "score_batches" / "model-a"
    root.mkdir(parents=True)
    temporary = root / ".batch_000000.npy.fixture.tmp"
    permanent = root / "keep.txt"
    temporary.write_bytes(b"temporary")
    permanent.write_bytes(b"owned evidence")
    assert remove_owned_temporary_files(root) == len(b"temporary")
    assert not temporary.exists()
    assert permanent.read_bytes() == b"owned evidence"

    outside = tmp_path / "outside"
    outside.mkdir()
    outside_data = outside / "batch_000000.npy"
    outside_data.write_bytes(b"must survive")
    symlink_root = tmp_path / "symlinked-score-root"
    symlink_root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ScoreBatchCacheError, match="symlinked checkpoint root"):
        discard_checkpoint(symlink_root, 0)
    assert outside_data.read_bytes() == b"must survive"

    monkeypatch.setattr(
        score_batch_cache.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=0),
    )
    with pytest.raises(ScoreBatchDiskReserveError, match="insufficient disk"):
        write_checkpoint(
            root,
            run_id="run-a",
            package_id="package-a",
            model_id="model-a",
            model_sha256="model-sha",
            sequence=0,
            items=_items(),
            probabilities=_probabilities(),
            min_free_bytes=1,
        )
    assert permanent.read_bytes() == b"owned evidence"


def test_checkpoint_reserves_remaining_outputs_and_enforces_frozen_high_water(
    tmp_path, monkeypatch
):
    root = tmp_path / "score_batches" / "model-a"
    probabilities = _probabilities()
    estimated_write = probabilities.nbytes + score_batch_cache.CHECKPOINT_WRITE_OVERHEAD_BYTES
    monkeypatch.setattr(
        score_batch_cache.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=estimated_write + 99),
    )
    with pytest.raises(ScoreBatchDiskReserveError, match="insufficient disk"):
        write_checkpoint(
            root,
            run_id="run-a",
            package_id="package-a",
            model_id="model-a",
            model_sha256="model-sha",
            sequence=0,
            items=_items(),
            probabilities=probabilities,
            min_free_bytes=50,
            additional_free_reserve_bytes=50,
        )
    assert not root.exists() or not any(root.iterdir())

    monkeypatch.setattr(
        score_batch_cache.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=10 * estimated_write),
    )
    with pytest.raises(ScoreBatchDiskReserveError, match="high-water"):
        write_checkpoint(
            root,
            run_id="run-a",
            package_id="package-a",
            model_id="model-a",
            model_sha256="model-sha",
            sequence=0,
            items=_items(),
            probabilities=probabilities,
            managed_cache_bytes=1,
            managed_cache_budget_bytes=estimated_write,
        )
    assert not root.exists() or not any(root.iterdir())
