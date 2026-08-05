"""Execute one bounded Work Package across all selected semantic streams."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from affine import Affine

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "qgis_plugins"
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from labeling_tool.core.run_spec import (
    RunSpecError,
    sha256_file,
    validated_run_tile_cache_dir,
)
from labeling_tool.core.run_state_db import RunStateDB

from _device import resolve_device, validate_device
from accepted_score import accepted_probabilities
from deployment_config import load_json
from incremental_fusion import FusionAccumulator
from partition_mosaic import (
    build_partition_arrays,
    derive_partition_arrays,
    write_partition_rasters,
)
from runtime_metrics import directory_size, peak_rss_bytes
from score_batch_cache import (
    ScoreBatchDiskReserveError,
    discard_checkpoint,
    load_checkpoint,
    remove_owned_temporary_files,
    write_checkpoint,
)
from semantic_batch import (
    _atomic_json,
    _atomic_npz,
    _read_tile,
    _run_model,
    _run_model_batch,
)
from tile_materializer import materialize_package_tiles
from torchscript_runtime import load_torchscript_model


class WorkPackageRuntimeError(RuntimeError):
    pass


def emit(event: str, **payload: Any) -> None:
    print(
        json.dumps({"event": event, **payload}, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )


def _default_loader(model_entry: Mapping[str, Any], device: str):
    return load_torchscript_model(Path(model_entry["artifact_path"]), device)[0]


def _default_infer(model: Any, tile_path: Path, device: str) -> np.ndarray:
    image, _profile = _read_tile(tile_path)
    _mask, _confidence, probabilities = _run_model(model, image, device)
    return probabilities.astype(np.float32)


def _default_infer_batch(
    model: Any,
    images: np.ndarray,
    device: str,
) -> np.ndarray:
    _masks, _confidence, probabilities = _run_model_batch(model, images, device)
    return probabilities


def _clear_accelerator_cache(device: str) -> None:
    try:
        import torch

        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif (
            str(device).startswith("mps")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            torch.mps.empty_cache()
    except Exception:
        pass


def _accepted_score_paths(root: Path, tile_id: str) -> tuple[Path, Path]:
    score_root = root / "accepted_scores"
    return score_root / f"tile_{tile_id}.npz", score_root / f"tile_{tile_id}.json"


def _score_is_current(
    score_path: Path,
    metadata_path: Path,
    expected: Mapping[str, Any],
) -> bool:
    if not score_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = load_json(metadata_path)
        if any(metadata.get(key) != value for key, value in expected.items()):
            return False
        with np.load(score_path, allow_pickle=False) as cached:
            probabilities = cached["probabilities"]
        return probabilities.shape == (14, 512, 512) and probabilities.dtype == np.float16
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _unlink_with_count(path: Path) -> int:
    if not path.is_file():
        return 0
    byte_count = path.stat().st_size
    path.unlink()
    return byte_count


def _remove_tree_with_count(path: Path) -> int:
    if not path.exists():
        return 0
    byte_count = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    shutil.rmtree(path)
    return byte_count


def _owned_tile_cache_file(path: str | Path, tile_cache_dir: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise WorkPackageRuntimeError(
            f"refusing to use symlinked Tile cache entry: {candidate}"
        )
    resolved = candidate.resolve()
    cache_root = tile_cache_dir.resolve()
    if resolved.parent != cache_root:
        raise WorkPackageRuntimeError(
            f"refusing to delete non-cache Tile path: {resolved}"
        )
    return resolved


def _prune_empty_tile_cache(tile_cache_dir: Path) -> None:
    for directory in (tile_cache_dir, tile_cache_dir.parent):
        try:
            directory.rmdir()
        except (FileNotFoundError, OSError):
            break


def _record_intersects_partition(
    record: Mapping[str, Any],
    partition: Mapping[str, Any],
    *,
    overlap: int,
) -> bool:
    width = int(record["width"])
    height = int(record["height"])
    stride_x = width - int(overlap)
    stride_y = height - int(overlap)
    tile_x0 = int(record["col"]) * stride_x
    tile_y0 = int(record["row"]) * stride_y
    tile_x1 = tile_x0 + width
    tile_y1 = tile_y0 + height
    halo = partition["halo_window"]
    return not (
        tile_x1 <= int(halo["x0"])
        or tile_y1 <= int(halo["y0"])
        or tile_x0 >= int(halo["x1"])
        or tile_y0 >= int(halo["y1"])
    )


def _commit_artifact(
    database: RunStateDB,
    run_id: str,
    *,
    path: Path,
    kind: str,
    stream_id: str,
    unit_id: str,
) -> int:
    artifact_id = database.register_artifact(
        run_id,
        kind,
        path,
        stream_id=stream_id,
        unit_id=unit_id,
    )
    existing = database.get_artifact(artifact_id)
    if existing and existing["status"] == "ready":
        if existing["byte_count"] == path.stat().st_size and existing["sha256"] == sha256_file(path):
            return artifact_id
        raise WorkPackageRuntimeError(f"ready Artifact changed on disk: {path}")
    if not database.mark_artifact_ready(
        artifact_id,
        byte_count=path.stat().st_size,
        sha256=sha256_file(path),
    ):
        raise WorkPackageRuntimeError(f"cannot commit Artifact: {path}")
    return artifact_id


def _load_profile(spec: Mapping[str, Any]) -> dict[str, Any] | None:
    fusion = spec.get("fusion")
    if not fusion:
        return None
    if isinstance(fusion.get("profile"), Mapping):
        return dict(fusion["profile"])
    path_value = fusion.get("snapshot_path") or fusion.get("profile_path") or fusion.get("file_path")
    if not path_value:
        raise WorkPackageRuntimeError("Fusion run spec has no profile snapshot")
    return load_json(Path(path_value))


def run_work_package(
    run_spec_path: str | Path,
    package_id: str,
    *,
    device: str | None = None,
    resume: bool = False,
    model_loader: Callable[[Mapping[str, Any], str], Any] = _default_loader,
    infer_tile: Callable[[Any, Path, str], np.ndarray] = _default_infer,
    infer_batch: Callable[[Any, list[Path], str], np.ndarray] | None = None,
    infer_images: Callable[[Any, np.ndarray, str], np.ndarray] | None = None,
) -> dict[str, Any]:
    started_at = time.monotonic()
    spec_path = Path(run_spec_path).resolve()
    spec = load_json(spec_path)
    if spec.get("schema_version") != 2:
        raise WorkPackageRuntimeError("Work Package runtime requires run_spec schema_version 2")
    run_id = str(spec["run_id"])
    run_dir = Path(spec["run_dir"]).resolve()
    try:
        tile_cache_dir = validated_run_tile_cache_dir(spec)
    except RunSpecError as error:
        raise WorkPackageRuntimeError(str(error)) from error
    database = RunStateDB(spec["state_db"])
    package = database.get_work_package(run_id, package_id)
    if package is None:
        raise WorkPackageRuntimeError(f"unknown Work Package: {package_id}")
    tiles = database.package_tiles(run_id, package_id)
    partitions = database.package_partitions(run_id, package_id)
    if not tiles or not partitions:
        raise WorkPackageRuntimeError("Work Package has no Tiles or Partitions")
    excluded_tiles = [tile for tile in tiles if str(tile.get("status")) == "excluded"]
    active_tiles = [tile for tile in tiles if str(tile.get("status")) != "excluded"]
    accepted_tiles = [tile for tile in tiles if str(tile.get("status")) == "accepted"]
    accepted_path = Path(str(spec.get("accepted_gpkg") or "")).expanduser()
    if accepted_tiles:
        if not spec.get("skip_accepted") or not accepted_path.is_file():
            raise WorkPackageRuntimeError(
                "Tile is marked accepted without an available accepted_labels snapshot"
            )
        accepted_sha = str(spec.get("accepted_gpkg_sha256") or "")
        if not accepted_sha or sha256_file(accepted_path) != accepted_sha:
            raise WorkPackageRuntimeError("accepted_labels changed after run creation")
    requested_device = device or (spec.get("runtime") or {}).get("effective_device") or "auto"
    effective_device = resolve_device(str(requested_device))
    if not validate_device(effective_device):
        raise WorkPackageRuntimeError(f"semantic device is unavailable: {effective_device}")
    configured_batch_size = max(
        1, int((spec.get("runtime") or {}).get("tile_batch_size", 1))
    )
    production_batch_inference = infer_images or _default_infer_batch
    if not database.set_work_package_status(
        run_id, package_id, "running", expected=("queued", "interrupted", "failed", "running")
    ):
        raise WorkPackageRuntimeError(f"Work Package cannot enter running state: {package_id}")

    package_root = run_dir / "tmp" / "work_packages" / package_id
    package_root.mkdir(parents=True, exist_ok=True)
    transform = Affine(*[float(value) for value in spec["raster"]["transform"]])
    crs = spec["raster"]["crs"]
    overlap = int(spec["tile_grid"]["overlap"])
    profile = _load_profile(spec)
    fusion_id = str((spec.get("fusion") or {}).get("profile_id") or "")
    fusion_accumulators: dict[str, FusionAccumulator] = {}
    partition_coverage_masks: dict[str, np.ndarray] = {}
    if profile:
        for partition in partitions:
            halo = partition["halo_window"]
            shape = (14, halo["y1"] - halo["y0"], halo["x1"] - halo["x0"])
            fusion_accumulators[partition["partition_id"]] = FusionAccumulator(
                package_root / "fusion" / fusion_id / partition["partition_id"],
                profile,
                shape,
            )

    model_summaries = []
    cleaned_bytes = 0
    model_load_count = 0
    peak_cache_bytes = 0
    tile_cache_released_count = 0
    tile_cache_retained_count = len(active_tiles)
    partition_pools: list[ThreadPoolExecutor] = []
    try:
        io_workers = int((spec.get("scaling") or {}).get("tile_io_workers", 8))

        def tile_progress(current, total, result):
            emit(
                "package_tile_materialized",
                run_id=run_id,
                package_id=package_id,
                tile_id=result["tile_id"],
                current=current,
                total=total,
                reused=bool(result["reused"]),
            )

        materialized = materialize_package_tiles(
            spec,
            active_tiles,
            workers=io_workers,
            progress=tile_progress,
        )
        materialized_by_id = {item["tile_id"]: item for item in materialized}
        for tile in active_tiles:
            item = materialized_by_id[str(tile["tile_id"])]
            tile["raster_path"] = item["tile_path"]
            tile["sha256"] = item["sha256"]
            if not database.update_tile_raster(
                run_id,
                str(tile["tile_id"]),
                raster_path=item["tile_path"],
                sha256=item["sha256"],
            ):
                raise WorkPackageRuntimeError(
                    f"cannot record materialized Tile: {tile['tile_id']}"
                )

        for model_index, model_entry in enumerate(spec["models"], start=1):
            model_id = str(model_entry["model_id"])
            stream_id = f"model:{model_id}"
            artifact_path = Path(model_entry["artifact_path"]).resolve()
            if not artifact_path.is_file():
                raise WorkPackageRuntimeError(f"model artifact is missing: {artifact_path}")
            actual_sha = sha256_file(artifact_path)
            if actual_sha != str(model_entry["sha256"]):
                raise WorkPackageRuntimeError(f"model SHA256 mismatch: {model_id}")
            database.set_stream_status(run_id, stream_id, "running")
            emit(
                "package_model_loading",
                run_id=run_id,
                package_id=package_id,
                stream_id=stream_id,
                current=model_index,
                total=len(spec["models"]),
            )
            inferable_tiles = [
                tile
                for tile in active_tiles
                if str(tile.get("status")) != "accepted"
            ]
            model = None
            score_records_by_tile: dict[str, dict[str, Any]] = {}
            reused = 0
            accepted_count = 0
            model_effective_batch_size = configured_batch_size
            model_batch_count = 0
            model_peak_batch_size = 0
            model_batch_reduction_count = 0
            checkpoint_written_count = 0
            checkpoint_reused_count = 0
            checkpoint_written_bytes = 0
            input_queue_capacity = 2
            input_queue_peak_batches = 0
            result_queue_capacity = 1
            result_queue_peak_batches = 0
            input_wait_sec = 0.0
            inference_sec = 0.0
            checkpoint_write_sec = 0.0
            checkpoint_wait_sec = 0.0
            partition_sec = 0.0
            score_batch_root = package_root / "score_batches" / model_id
            cleaned_bytes += remove_owned_temporary_files(score_batch_root)

            def record_score(
                item: Mapping[str, Any], record: Mapping[str, Any]
            ) -> None:
                tile_value_id = str(item["tile"]["tile_id"])
                score_records_by_tile[tile_value_id] = dict(record)
                emit(
                    "package_tile_completed",
                    run_id=run_id,
                    package_id=package_id,
                    stream_id=stream_id,
                    tile_id=tile_value_id,
                    current=int(item["tile_index"]),
                    total=len(active_tiles),
                )

            all_items: list[dict[str, Any]] = []
            for tile_index, tile in enumerate(active_tiles, start=1):
                tile_path = Path(tile["raster_path"]).resolve()
                if not tile_path.is_file():
                    raise WorkPackageRuntimeError(f"Tile raster is missing: {tile_path}")
                all_items.append(
                    {
                        "tile": tile,
                        "tile_index": tile_index,
                        "tile_path": tile_path,
                    }
                )
            accepted_items = [
                item
                for item in all_items
                if str(item["tile"].get("status")) == "accepted"
            ]
            inferable_items = [
                item
                for item in all_items
                if str(item["tile"].get("status")) != "accepted"
            ]
            for item in accepted_items:
                tile = item["tile"]
                tile_id = str(tile["tile_id"])
                score_path, metadata_path = _accepted_score_paths(
                    package_root, tile_id
                )
                expected = {
                    "schema_version": 1,
                    "run_id": run_id,
                    "package_id": package_id,
                    "tile_id": tile_id,
                    "source": "accepted_labels",
                    "accepted_gpkg_sha256": str(spec["accepted_gpkg_sha256"]),
                    "input_sha256": str(tile["sha256"]),
                }
                if resume and _score_is_current(score_path, metadata_path, expected):
                    reused += 1
                else:
                    probabilities = np.asarray(
                        accepted_probabilities(accepted_path, item["tile_path"]),
                        dtype=np.float32,
                    )
                    if probabilities.shape != (14, 512, 512):
                        raise WorkPackageRuntimeError(
                            "Tile probability shape must be [14,512,512], got "
                            f"{probabilities.shape}"
                        )
                    _atomic_npz(
                        score_path, probabilities=probabilities.astype(np.float16)
                    )
                    _atomic_json(metadata_path, expected)
                accepted_count += 1
                record_score(
                    item,
                    {
                        "tile_id": tile_id,
                        "row": int(tile["row_no"]),
                        "col": int(tile["col_no"]),
                        "width": int(tile["width"]),
                        "height": int(tile["height"]),
                        "score_path": str(score_path),
                        "metadata_path": str(metadata_path),
                        "cache_kind": "accepted",
                    },
                )

            groups = [
                (sequence, inferable_items[offset : offset + configured_batch_size])
                for sequence, offset in enumerate(
                    range(0, len(inferable_items), configured_batch_size)
                )
            ]
            missing_groups: list[tuple[int, list[dict[str, Any]]]] = []
            for sequence, group in groups:
                records = (
                    load_checkpoint(
                        score_batch_root,
                        run_id=run_id,
                        package_id=package_id,
                        model_id=model_id,
                        model_sha256=actual_sha,
                        sequence=sequence,
                        items=group,
                    )
                    if resume
                    else None
                )
                if records is None:
                    cleaned_bytes += discard_checkpoint(score_batch_root, sequence)
                    missing_groups.append((sequence, group))
                    continue
                checkpoint_reused_count += 1
                reused += len(group)
                for item, record in zip(group, records):
                    record_score(item, record)

            if missing_groups:
                model = model_loader(model_entry, effective_device)
                model_load_count += 1

            partition_requirements: list[tuple[dict[str, Any], list[str]]] = []
            for partition in partitions:
                required_ids = []
                for item in all_items:
                    tile = item["tile"]
                    if _record_intersects_partition(
                        {
                            "row": tile["row_no"],
                            "col": tile["col_no"],
                            "width": tile["width"],
                            "height": tile["height"],
                        },
                        partition,
                        overlap=overlap,
                    ):
                        required_ids.append(str(tile["tile_id"]))
                partition_requirements.append((partition, required_ids))

            def build_partition(
                partition: Mapping[str, Any], records: list[dict[str, Any]]
            ) -> tuple[dict[str, np.ndarray], float]:
                partition_started = time.monotonic()
                arrays = build_partition_arrays(
                    records,
                    partition,
                    overlap=overlap,
                    allow_uncovered=True,
                )
                return arrays, time.monotonic() - partition_started

            def commit_partition(
                partition: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
            ) -> None:
                partition_id = str(partition["partition_id"])
                probability_path = (
                    run_dir
                    / "tmp"
                    / "probability_parts"
                    / model_id
                    / f"{partition_id}.tif"
                )
                raster_root = run_dir / "models" / model_id / "raster_parts"
                paths = write_partition_rasters(
                    arrays,
                    partition,
                    global_transform=transform,
                    crs=crs,
                    output_probability=probability_path,
                    output_mask=raster_root / f"{partition_id}_mask.tif",
                    output_confidence=raster_root
                    / f"{partition_id}_confidence.tif",
                )
                for kind, key in (
                    ("partition_probability", "probability"),
                    ("core_mask", "mask"),
                    ("core_confidence", "confidence"),
                ):
                    artifact_id = _commit_artifact(
                        database,
                        run_id,
                        path=Path(paths[key]),
                        kind=kind,
                        stream_id=stream_id,
                        unit_id=partition_id,
                    )
                    if kind == "partition_probability":
                        database.link_partition_artifact(
                            run_id, stream_id, partition_id, artifact_id
                        )
                if profile:
                    coverage = arrays["halo_weights"] > 0
                    previous_coverage = partition_coverage_masks.get(partition_id)
                    if previous_coverage is None:
                        partition_coverage_masks[partition_id] = coverage
                    elif not np.array_equal(previous_coverage, coverage):
                        raise WorkPackageRuntimeError(
                            f"model coverage differs inside Partition: {partition_id}"
                        )
                    fusion_accumulators[partition_id].add_model(
                        model_id, arrays["halo_probabilities"]
                    )

            partition_cursor = 0
            partition_future: Future[tuple[dict[str, np.ndarray], float]] | None = None
            partition_future_entry: tuple[dict[str, Any], list[str]] | None = None
            partition_queue_peak = 0
            partition_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="loess-partition"
            )
            partition_pools.append(partition_pool)

            def drain_partition(*, wait: bool) -> None:
                nonlocal partition_future
                nonlocal partition_future_entry
                nonlocal partition_sec
                if (
                    partition_future is None
                    or partition_future_entry is None
                    or (not wait and not partition_future.done())
                ):
                    return
                arrays, build_elapsed = partition_future.result()
                partition, _required_ids = partition_future_entry
                commit_started = time.monotonic()
                commit_partition(partition, arrays)
                partition_sec += float(build_elapsed) + (
                    time.monotonic() - commit_started
                )
                partition_future = None
                partition_future_entry = None

            def schedule_ready_partition() -> None:
                nonlocal partition_cursor
                nonlocal partition_future
                nonlocal partition_future_entry
                nonlocal partition_queue_peak
                drain_partition(wait=False)
                if partition_future is not None or partition_cursor >= len(
                    partition_requirements
                ):
                    return
                entry = partition_requirements[partition_cursor]
                partition, required_ids = entry
                if not all(tile_id in score_records_by_tile for tile_id in required_ids):
                    return
                records = [score_records_by_tile[tile_id] for tile_id in required_ids]
                partition_future = partition_pool.submit(
                    build_partition, partition, records
                )
                partition_future_entry = entry
                partition_queue_peak = 1
                partition_cursor += 1

            schedule_ready_partition()

            min_free_bytes = int(
                float((spec.get("scaling") or {}).get("min_free_disk_gb", 0.0))
                * 1024**3
            )

            def write_batch(
                sequence: int,
                group: list[dict[str, Any]],
                probabilities: np.ndarray,
            ) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
                write_started = time.monotonic()
                records, manifest = write_checkpoint(
                    score_batch_root,
                    run_id=run_id,
                    package_id=package_id,
                    model_id=model_id,
                    model_sha256=actual_sha,
                    sequence=sequence,
                    items=group,
                    probabilities=probabilities,
                    min_free_bytes=min_free_bytes,
                )
                return records, manifest, time.monotonic() - write_started

            writer_future: Future[
                tuple[list[dict[str, Any]], dict[str, Any], float]
            ] | None = None
            writer_group: list[dict[str, Any]] | None = None

            def drain_writer() -> None:
                nonlocal writer_future
                nonlocal writer_group
                nonlocal checkpoint_written_count
                nonlocal checkpoint_written_bytes
                nonlocal checkpoint_write_sec
                nonlocal checkpoint_wait_sec
                nonlocal peak_cache_bytes
                if writer_future is None or writer_group is None:
                    return
                wait_started = time.monotonic()
                records, manifest, write_elapsed = writer_future.result()
                checkpoint_wait_sec += time.monotonic() - wait_started
                checkpoint_write_sec += float(write_elapsed)
                checkpoint_written_count += 1
                checkpoint_written_bytes += int(manifest["byte_count"])
                for item, record in zip(writer_group, records):
                    record_score(item, record)
                peak_cache_bytes = max(
                    peak_cache_bytes,
                    directory_size(package_root) + directory_size(tile_cache_dir),
                )
                writer_future = None
                writer_group = None
                schedule_ready_partition()

            with ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="loess-score-writer"
            ) as writer_pool:
                with ThreadPoolExecutor(
                    max_workers=max(1, io_workers),
                    thread_name_prefix="loess-tile-read",
                ) as read_pool:
                    queued_reads: deque[
                        tuple[int, list[dict[str, Any]], list[Future[Any]]]
                    ] = deque()
                    group_cursor = 0

                    def fill_input_queue() -> None:
                        nonlocal group_cursor
                        nonlocal input_queue_peak_batches
                        while (
                            len(queued_reads) < input_queue_capacity
                            and group_cursor < len(missing_groups)
                        ):
                            sequence, group = missing_groups[group_cursor]
                            futures = [
                                read_pool.submit(_read_tile, item["tile_path"])
                                for item in group
                            ]
                            queued_reads.append((sequence, group, futures))
                            group_cursor += 1
                            input_queue_peak_batches = max(
                                input_queue_peak_batches, len(queued_reads)
                            )

                    fill_input_queue()
                    while queued_reads:
                        sequence, group, read_futures = queued_reads.popleft()
                        fill_input_queue()
                        read_started = time.monotonic()
                        images = np.stack(
                            [future.result()[0] for future in read_futures], axis=0
                        )
                        input_wait_sec += time.monotonic() - read_started
                        group_outputs: list[np.ndarray] = []
                        cursor = 0
                        while cursor < len(group):
                            attempt_size = min(
                                model_effective_batch_size, len(group) - cursor
                            )
                            subgroup = group[cursor : cursor + attempt_size]
                            inference_started = time.monotonic()
                            try:
                                if infer_images is not None:
                                    output = infer_images(
                                        model,
                                        images[cursor : cursor + attempt_size],
                                        effective_device,
                                    )
                                elif infer_batch is not None:
                                    output = infer_batch(
                                        model,
                                        [item["tile_path"] for item in subgroup],
                                        effective_device,
                                    )
                                elif infer_tile is not _default_infer:
                                    output = np.stack(
                                        [
                                            infer_tile(
                                                model,
                                                item["tile_path"],
                                                effective_device,
                                            )
                                            for item in subgroup
                                        ],
                                        axis=0,
                                    )
                                else:
                                    output = production_batch_inference(
                                        model,
                                        images[cursor : cursor + attempt_size],
                                        effective_device,
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
                                if attempt_size <= 1:
                                    raise
                                model_effective_batch_size = min(
                                    model_effective_batch_size,
                                    max(1, attempt_size // 2),
                                )
                                model_batch_reduction_count += 1
                                _clear_accelerator_cache(effective_device)
                                emit(
                                    "package_tile_batch_reduced",
                                    run_id=run_id,
                                    package_id=package_id,
                                    stream_id=stream_id,
                                    attempted_batch_size=attempt_size,
                                    effective_batch_size=model_effective_batch_size,
                                    reason=str(error),
                                )
                                continue
                            inference_sec += time.monotonic() - inference_started
                            model_batch_count += 1
                            model_peak_batch_size = max(
                                model_peak_batch_size, attempt_size
                            )
                            group_outputs.append(
                                np.asarray(probabilities_batch, dtype=np.float16)
                            )
                            cursor += attempt_size
                        probabilities = np.concatenate(group_outputs, axis=0)
                        drain_writer()
                        writer_group = group
                        writer_future = writer_pool.submit(
                            write_batch, sequence, group, probabilities
                        )
                        result_queue_peak_batches = 1
                    drain_writer()
            while partition_cursor < len(partition_requirements):
                schedule_ready_partition()
                if partition_future is None:
                    partition, required_ids = partition_requirements[
                        partition_cursor
                    ]
                    missing_ids = [
                        tile_id
                        for tile_id in required_ids
                        if tile_id not in score_records_by_tile
                    ]
                    raise WorkPackageRuntimeError(
                        f"Partition {partition['partition_id']} lacks scores: "
                        f"{missing_ids[:5]}"
                    )
                drain_partition(wait=True)
            drain_partition(wait=True)
            partition_pool.shutdown(wait=True, cancel_futures=True)
            partition_pools.remove(partition_pool)
            peak_cache_bytes = max(
                peak_cache_bytes,
                directory_size(package_root) + directory_size(tile_cache_dir),
            )
            if not bool((spec.get("runtime") or {}).get("keep_score_cache", False)):
                cleaned_bytes += _remove_tree_with_count(score_batch_root)
                batch_parent = package_root / "score_batches"
                if batch_parent.is_dir() and not any(batch_parent.iterdir()):
                    batch_parent.rmdir()
            model_summaries.append(
                {
                    "model_id": model_id,
                    "tile_count": len(active_tiles),
                    "inferred_count": len(inferable_tiles),
                    "accepted_count": accepted_count,
                    "excluded_count": len(excluded_tiles),
                    "reused_count": reused,
                    "configured_tile_batch_size": configured_batch_size,
                    "effective_tile_batch_size": model_effective_batch_size,
                    "peak_tile_batch_size": model_peak_batch_size,
                    "inference_batch_count": model_batch_count,
                    "batch_reduction_count": model_batch_reduction_count,
                    "checkpoint_written_count": checkpoint_written_count,
                    "checkpoint_reused_count": checkpoint_reused_count,
                    "checkpoint_written_bytes": checkpoint_written_bytes,
                    "input_queue_capacity": input_queue_capacity,
                    "input_queue_peak_batches": input_queue_peak_batches,
                    "result_queue_capacity": result_queue_capacity,
                    "result_queue_peak_batches": result_queue_peak_batches,
                    "partition_queue_capacity": 1,
                    "partition_queue_peak": partition_queue_peak,
                    "input_wait_sec": round(input_wait_sec, 6),
                    "inference_sec": round(inference_sec, 6),
                    "checkpoint_write_sec": round(checkpoint_write_sec, 6),
                    "checkpoint_wait_sec": round(checkpoint_wait_sec, 6),
                    "partition_sec": round(partition_sec, 6),
                }
            )

        if profile:
            stream_id = f"fusion:{fusion_id}"
            database.set_stream_status(run_id, stream_id, "running")
            for partition in partitions:
                partition_id = partition["partition_id"]
                probabilities = fusion_accumulators[partition_id].finalize()
                coverage = partition_coverage_masks[partition_id]
                probabilities[:, ~coverage] = 0.0
                arrays = derive_partition_arrays(
                    probabilities,
                    partition,
                    weights=coverage.astype(np.float32),
                )
                probability_path = (
                    run_dir / "tmp" / "probability_parts" / f"fusion_{fusion_id}" / f"{partition_id}.tif"
                )
                raster_root = run_dir / "fusion" / fusion_id / "raster_parts"
                paths = write_partition_rasters(
                    arrays,
                    partition,
                    global_transform=transform,
                    crs=crs,
                    output_probability=probability_path,
                    output_mask=raster_root / f"{partition_id}_mask.tif",
                    output_confidence=raster_root / f"{partition_id}_confidence.tif",
                )
                for kind, key in (
                    ("partition_probability", "probability"),
                    ("core_mask", "mask"),
                    ("core_confidence", "confidence"),
                ):
                    artifact_id = _commit_artifact(
                        database,
                        run_id,
                        path=Path(paths[key]),
                        kind=kind,
                        stream_id=stream_id,
                        unit_id=partition_id,
                    )
                    if kind == "partition_probability":
                        database.link_partition_artifact(
                            run_id, stream_id, partition_id, artifact_id
                        )
            if not bool((spec.get("runtime") or {}).get("keep_score_cache", False)):
                cleaned_bytes += _remove_tree_with_count(
                    package_root / "fusion" / fusion_id
                )
                fusion_root = package_root / "fusion"
                if fusion_root.is_dir() and not any(fusion_root.iterdir()):
                    fusion_root.rmdir()
        if not bool((spec.get("runtime") or {}).get("keep_score_cache", False)):
            cleaned_bytes += _remove_tree_with_count(package_root / "accepted_scores")
            tile_cleaned_bytes = 0
            releasable_tile_ids = set(
                database.releasable_package_tile_ids(run_id, package_id)
            )
            released_tile_count = 0
            for item in materialized:
                if str(item["tile_id"]) not in releasable_tile_ids:
                    continue
                tile_cleaned_bytes += _unlink_with_count(
                    _owned_tile_cache_file(item["tile_path"], tile_cache_dir)
                )
                tile_cleaned_bytes += _unlink_with_count(
                    _owned_tile_cache_file(item["metadata_path"], tile_cache_dir)
                )
                released_tile_count += 1
            cleaned_bytes += tile_cleaned_bytes
            tile_cache_released_count = released_tile_count
            tile_cache_retained_count = len(active_tiles) - released_tile_count
            _prune_empty_tile_cache(tile_cache_dir)
            emit(
                "package_tiles_cleaned",
                run_id=run_id,
                package_id=package_id,
                tile_count=released_tile_count,
                dependency_retained_count=(
                    len(active_tiles) - released_tile_count
                ),
                cleaned_bytes=tile_cleaned_bytes,
            )
        peak_cache_bytes = max(
            peak_cache_bytes,
            directory_size(package_root) + directory_size(tile_cache_dir),
        )
        database.set_work_package_status(run_id, package_id, "ready", expected="running")
        result = {
            "run_id": run_id,
            "package_id": package_id,
            "tile_count": len(active_tiles),
            "grid_tile_count": len(tiles),
            "excluded_tile_count": len(excluded_tiles),
            "partition_count": len(partitions),
            "models": model_summaries,
            "fusion_profile_id": fusion_id,
            "requested_device": str(requested_device),
            "effective_device": str(effective_device),
            "model_load_count": model_load_count,
            "configured_tile_batch_size": configured_batch_size,
            "peak_cache_bytes": peak_cache_bytes,
            "peak_rss_bytes": peak_rss_bytes(),
            "cleaned_bytes": cleaned_bytes,
            "tile_cache_released_count": tile_cache_released_count,
            "tile_cache_retained_count": tile_cache_retained_count,
            "elapsed_sec": round(time.monotonic() - started_at, 3),
            "status": "ready",
        }
        _atomic_json(package_root / "package_report.json", result)
        emit("work_package_finished", **result)
        return result
    except ScoreBatchDiskReserveError as error:
        for pool in partition_pools:
            pool.shutdown(wait=True, cancel_futures=True)
        partition_pools.clear()
        database.set_work_package_status(
            run_id, package_id, "interrupted", expected="running"
        )
        database.append_event(
            run_id,
            "work_package_paused_low_disk",
            level="warning",
            message=str(error),
            payload={"package_id": package_id},
        )
        raise
    except Exception as error:
        for pool in partition_pools:
            pool.shutdown(wait=True, cancel_futures=True)
        partition_pools.clear()
        database.set_work_package_status(run_id, package_id, "failed", expected="running")
        database.append_event(
            run_id,
            "work_package_failed",
            level="error",
            message=str(error),
            payload={"package_id": package_id},
        )
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run one bounded semantic Work Package")
    parser.add_argument("--run-spec", required=True)
    parser.add_argument("--package-id", required=True)
    parser.add_argument("--device")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        run_work_package(
            args.run_spec,
            args.package_id,
            device=args.device,
            resume=args.resume,
        )
        return 0
    except Exception as error:
        emit("work_package_failed", package_id=args.package_id, error=str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
