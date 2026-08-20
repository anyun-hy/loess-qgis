"""Execute one bounded Work Package across all selected semantic streams."""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
import signal
import shutil
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import rasterio
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
from authoritative_raster import (
    apply_range_mask_to_core,
    core_mask_tags,
    regularize_partition_core,
)
from deployment_config import load_json
from incremental_fusion import FusionAccumulator
from partition_mosaic import (
    build_partition_arrays,
    derive_partition_arrays,
    write_partition_rasters,
)
from range_clip_runtime import RangeClipRuntimeError, extract_range_mask_geometry
from runtime_metrics import directory_size, peak_rss_bytes
from score_batch_cache import (
    CHECKPOINT_WRITE_OVERHEAD_BYTES,
    ScoreBatchDiskReserveError,
    discard_checkpoint,
    load_checkpoint,
    remove_owned_temporary_files,
    write_checkpoint,
)
from storage_guard import StorageGuard, StorageReserveError
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


def _range_geometry_for_run(spec: Mapping[str, Any], crs: str):
    """Resolve the one frozen exact boundary before a Package starts work."""

    try:
        return extract_range_mask_geometry(spec, crs)
    except RangeClipRuntimeError as error:
        raise WorkPackageRuntimeError(str(error)) from error


class LeaseLostError(WorkPackageRuntimeError):
    """The current worker no longer owns the exact database lease."""


class WorkerStopRequested(WorkPackageRuntimeError):
    """The accelerator worker received a coordinated stop request."""


class BatchCapacityError(WorkPackageRuntimeError):
    """A tested accelerator batch is too large for the active model/device."""


def _storage_error_is_transient(error: BaseException) -> bool:
    return bool(getattr(error, "transient", False))


def _is_recoverable_batch_error(error: BaseException, device: str) -> bool:
    """Return whether retrying the same inference with a smaller batch is safe.

    Shape, data, model, and I/O failures must remain failures.  Treating every
    ``RuntimeError`` as capacity pressure silently degraded every later Package
    after an unrelated bad Tile.  Tests may use ``BatchCapacityError`` as an
    explicit accelerator-capacity signal; production also recognizes the
    standard CUDA/MPS out-of-memory spellings.
    """

    if isinstance(error, BatchCapacityError):
        return True
    accelerator = str(device).lower()
    if not (accelerator.startswith("cuda") or accelerator.startswith("mps")):
        return False
    class_name = type(error).__name__.lower()
    message = str(error).lower()
    return "outofmemory" in class_name or "out of memory" in message


class _PackageFileLock:
    """Serialize filesystem mutations for one Package across lease owners."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._handle = None

    def acquire(
        self,
        lease_guard: Callable[[], None] | None = None,
        *,
        timeout_sec: float = 300.0,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        started = time.monotonic()
        while True:
            if lease_guard is not None:
                lease_guard()
            try:
                fcntl.flock(
                    self._handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
                self._handle.seek(0)
                self._handle.truncate()
                self._handle.write(
                    json.dumps(
                        {
                            "pid": os.getpid(),
                            "acquired_at": time.time(),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
                self._handle.flush()
                os.fsync(self._handle.fileno())
                return
            except BlockingIOError:
                if lease_guard is None:
                    owner = self._owner_text()
                    self.release()
                    raise WorkPackageRuntimeError(
                        "Work Package filesystem lock is already held: "
                        f"{self.path}{owner}"
                    )
                if time.monotonic() - started >= max(0.05, float(timeout_sec)):
                    owner = self._owner_text()
                    self.release()
                    raise WorkPackageRuntimeError(
                        "timed out waiting for Work Package filesystem lock: "
                        f"{self.path}{owner}"
                    )
                time.sleep(0.05)

    def _owner_text(self) -> str:
        if self._handle is None:
            return ""
        try:
            self._handle.seek(0)
            value = self._handle.read().decode("utf-8", errors="replace").strip()
        except OSError:
            return ""
        return f"; owner={value}" if value else ""

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def emit(event: str, **payload: Any) -> None:
    print(
        json.dumps({"event": event, **payload}, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )


def _default_loader(model_entry: Mapping[str, Any], device: str):
    return load_torchscript_model(Path(model_entry["artifact_path"]), device)[0]


class PersistentModelProvider:
    """Verify and load each frozen model at most once per worker process."""

    def __init__(
        self,
        loader: Callable[[Mapping[str, Any], str], Any] = _default_loader,
    ) -> None:
        self._loader = loader
        self._verified: dict[tuple[str, str], str] = {}
        self._models: dict[tuple[str, str, str, str], Any] = {}
        self._effective_batch_sizes: dict[tuple[str, str], int] = {}
        self.cold_load_counts: dict[str, int] = {}
        self.cache_hit_counts: dict[str, int] = {}

    def verify(self, model_entry: Mapping[str, Any]) -> str:
        path = Path(str(model_entry["artifact_path"])).resolve()
        expected = str(model_entry.get("sha256") or "").lower()
        if not path.is_file():
            raise WorkPackageRuntimeError(f"model artifact is missing: {path}")
        if not expected:
            raise WorkPackageRuntimeError(
                f"model SHA256 is missing: {model_entry.get('model_id')}"
            )
        key = (str(path), expected)
        actual = self._verified.get(key)
        if actual is None:
            actual = sha256_file(path)
            if actual != expected:
                raise WorkPackageRuntimeError(
                    f"model SHA256 mismatch: {model_entry.get('model_id')}"
                )
            self._verified[key] = actual
        return actual

    def get(
        self,
        model_entry: Mapping[str, Any],
        device: str,
        *,
        observer: Callable[[str, str], None] | None = None,
    ) -> tuple[Any, bool]:
        actual = self.verify(model_entry)
        model_id = str(model_entry["model_id"])
        path = str(Path(str(model_entry["artifact_path"])).resolve())
        key = (model_id, path, actual, str(device))
        if key in self._models:
            self.cache_hit_counts[model_id] = (
                self.cache_hit_counts.get(model_id, 0) + 1
            )
            if observer is not None:
                observer(model_id, "cache_hit")
            return self._models[key], False
        if observer is not None:
            observer(model_id, "load_started")
        model = self._loader(model_entry, device)
        self._models[key] = model
        self.cold_load_counts[model_id] = (
            self.cold_load_counts.get(model_id, 0) + 1
        )
        if observer is not None:
            observer(model_id, "load_completed")
        return model, True

    def clear(self) -> None:
        self._models.clear()
        self._verified.clear()
        self._effective_batch_sizes.clear()

    def effective_batch_size(
        self,
        model_entry: Mapping[str, Any],
        device: str,
        configured: int,
    ) -> int:
        key = (str(model_entry["model_id"]), str(device))
        value = max(
            1,
            min(
                int(configured),
                int(self._effective_batch_sizes.get(key, configured)),
            ),
        )
        self._effective_batch_sizes.setdefault(key, value)
        return value

    def remember_batch_size(
        self,
        model_entry: Mapping[str, Any],
        device: str,
        effective: int,
    ) -> None:
        key = (str(model_entry["model_id"]), str(device))
        value = max(1, int(effective))
        previous = self._effective_batch_sizes.get(key)
        self._effective_batch_sizes[key] = (
            value if previous is None else min(int(previous), value)
        )

    @property
    def effective_batch_sizes(self) -> dict[str, int]:
        return {
            f"{model_id}@{device}": int(value)
            for (model_id, device), value in sorted(
                self._effective_batch_sizes.items()
            )
        }


class _LeaseHeartbeat:
    """Own one Package lease and fence all irreversible write boundaries."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        run_id: str,
        package_id: str,
        job_id: int,
        lease_token: str,
        stop_event: threading.Event,
        interval_sec: float = 15.0,
        lease_seconds: int = 120,
    ) -> None:
        self.database_path = str(database_path)
        self.run_id = str(run_id)
        self.package_id = str(package_id)
        self.job_id = int(job_id)
        self.lease_token = str(lease_token)
        self.stop_event = stop_event
        self._close_event = threading.Event()
        self.interval_sec = max(0.05, float(interval_sec))
        self.lease_seconds = max(30, int(lease_seconds))
        self.lost_event = threading.Event()
        self.heartbeat_count = 0
        self._progress_lock = threading.Lock()
        self._progress_current = 0
        self._progress_total = 0
        self._heartbeat_error = ""
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.check()
        self._thread = threading.Thread(
            target=self._run,
            name=f"loess-lease-{self.job_id}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._close_event.wait(self.interval_sec):
            with self._progress_lock:
                current = self._progress_current
                total = self._progress_total
            try:
                database = RunStateDB(self.database_path)
                accepted = database.heartbeat(
                    self.job_id,
                    self.lease_token,
                    current=current,
                    total=total,
                    lease_seconds=self.lease_seconds,
                )
            except Exception as error:
                self._heartbeat_error = str(error)
                self.lost_event.set()
                return
            if not accepted:
                self._heartbeat_error = "heartbeat lease update was rejected"
                self.lost_event.set()
                return
            self.heartbeat_count += 1

    def update_progress(self, current: int, total: int) -> None:
        with self._progress_lock:
            self._progress_current = max(0, int(current))
            self._progress_total = max(0, int(total))

    def check(self) -> None:
        if self.stop_event.is_set():
            raise WorkerStopRequested("accelerator worker stop requested")
        if self.lost_event.is_set():
            detail = self._heartbeat_error or "heartbeat was rejected"
            raise LeaseLostError(f"Work Package lease lost: {detail}")
        database = RunStateDB(self.database_path)
        if not database.work_package_job_holds_lease(
            self.run_id,
            self.package_id,
            self.job_id,
            self.lease_token,
        ):
            self.lost_event.set()
            raise LeaseLostError("Work Package lease is no longer owned")

    def close(self) -> None:
        self._close_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_sec + 1.0)
            self._thread = None


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
    device_value = str(device)
    if not (
        device_value.startswith("cuda")
        or device_value.startswith("mps")
    ):
        return
    try:
        import torch

        gc.collect()
        if device_value.startswith("cuda") and torch.cuda.is_available():
            synchronize = getattr(torch.cuda, "synchronize", None)
            if callable(synchronize):
                synchronize(device_value)
            torch.cuda.empty_cache()
        elif (
            device_value.startswith("mps")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            synchronize = getattr(torch.mps, "synchronize", None)
            if callable(synchronize):
                synchronize()
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
    if kind == "partition_probability":
        return database.publish_partition_artifact(
            run_id,
            stream_id,
            unit_id,
            path,
            byte_count=path.stat().st_size,
            sha256=sha256_file(path),
        )
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


def _artifact_record_is_valid(
    artifact: Mapping[str, Any] | None,
    *,
    expected_count: int,
    expected_dtype: str,
    expected_width: int,
    expected_height: int,
    expected_crs: str,
) -> bool:
    if artifact is None or str(artifact.get("status")) != "ready":
        return False
    path = Path(str(artifact.get("path") or ""))
    try:
        if (
            not path.is_file()
            or path.stat().st_size != int(artifact["byte_count"])
            or sha256_file(path) != str(artifact["sha256"])
        ):
            return False
        with rasterio.open(path) as source:
            return (
                source.count == int(expected_count)
                and source.width == int(expected_width)
                and source.height == int(expected_height)
                and all(dtype == str(expected_dtype) for dtype in source.dtypes)
                and str(source.crs or "") == str(expected_crs)
            )
    except (OSError, ValueError, KeyError, rasterio.errors.RasterioError):
        return False


def _model_partition_outputs_reusable(
    database: RunStateDB,
    *,
    run_id: str,
    stream_id: str,
    model_id: str,
    partitions: list[Mapping[str, Any]],
    crs: str,
    fusion_accumulators: Mapping[str, FusionAccumulator],
) -> bool:
    artifacts = database.artifacts_for_stream(run_id, stream_id, status=None)
    by_key = {
        (str(item.get("unit_id") or ""), str(item.get("kind") or "")): item
        for item in artifacts
    }
    for partition in partitions:
        partition_id = str(partition["partition_id"])
        core = partition["core_window"]
        width = int(core["x1"]) - int(core["x0"])
        height = int(core["y1"]) - int(core["y0"])
        if not _artifact_record_is_valid(
            by_key.get((partition_id, "core_mask")),
            expected_count=1,
            expected_dtype="int16",
            expected_width=width,
            expected_height=height,
            expected_crs=crs,
        ):
            return False
        if not _artifact_record_is_valid(
            by_key.get((partition_id, "core_confidence")),
            expected_count=1,
            expected_dtype="float32",
            expected_width=width,
            expected_height=height,
            expected_crs=crs,
        ):
            return False
        probability = by_key.get((partition_id, "partition_probability"))
        if probability is not None and str(probability.get("status")) == "ready":
            halo = partition["halo_window"]
            if not _artifact_record_is_valid(
                probability,
                expected_count=14,
                expected_dtype="uint16",
                expected_width=int(halo["x1"]) - int(halo["x0"]),
                expected_height=int(halo["y1"]) - int(halo["y0"]),
                expected_crs=crs,
            ):
                return False
        elif probability is None or str(probability.get("status")) != "cleaned":
            return False
        accumulator = fusion_accumulators.get(partition_id)
        if accumulator is not None and model_id not in accumulator.completed_model_ids():
            return False
    return True


def _restore_partition_coverage_masks(
    database: RunStateDB,
    *,
    run_id: str,
    stream_id: str,
    partitions: list[Mapping[str, Any]],
    coverage_masks: dict[str, np.ndarray],
) -> bool:
    """Restore Fusion coverage when model outputs are reused after a restart.

    Coverage is not encoded in the calibrated accumulator for every Fusion
    strategy, so it must be reconstructed from a committed model probability
    raster.  A missing/cleaned raster makes this model ineligible for the
    zero-compute reuse path; the caller will rebuild it from score checkpoints
    instead of guessing coverage.
    """

    artifacts = database.artifacts_for_stream(
        run_id, stream_id, kind="partition_probability", status="ready"
    )
    by_partition = {
        str(item.get("unit_id") or ""): item for item in artifacts
    }
    restored: dict[str, np.ndarray] = {}
    try:
        for partition in partitions:
            partition_id = str(partition["partition_id"])
            artifact = by_partition.get(partition_id)
            if artifact is None:
                return False
            with rasterio.open(Path(str(artifact["path"]))) as source:
                coverage = np.any(source.read() > 0, axis=0)
            previous = coverage_masks.get(partition_id)
            if previous is not None and not np.array_equal(previous, coverage):
                raise WorkPackageRuntimeError(
                    f"model coverage differs inside Partition: {partition_id}"
                )
            restored[partition_id] = coverage
    except (OSError, ValueError, KeyError, rasterio.errors.RasterioError):
        return False
    coverage_masks.update(restored)
    return True


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


def _load_linear_fusion_head(
    spec: Mapping[str, Any],
    profile: Mapping[str, Any] | None,
    device: str,
):
    """Load and adapt the one frozen linear head for this worker process."""

    if not profile or str(profile.get("strategy") or "") != "linear_1x1":
        return None
    head_info = profile.get("fusion_head")
    if not isinstance(head_info, Mapping):
        raise WorkPackageRuntimeError("linear_1x1 profile has no fusion_head")
    artifact = str(head_info.get("artifact") or "").strip()
    expected_sha = str(head_info.get("sha256") or "").strip().lower()
    if not artifact or not expected_sha:
        raise WorkPackageRuntimeError(
            "linear_1x1 fusion_head artifact or SHA256 is missing"
        )

    candidates: list[Path] = []
    explicit_path = str(head_info.get("artifact_path") or "").strip()
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser().resolve())
    artifact_path = Path(artifact).expanduser()
    if artifact_path.is_absolute():
        candidates.append(artifact_path.resolve())
    fusion = spec.get("fusion") or {}
    for key in ("profile_path", "file_path"):
        profile_path = str(fusion.get(key) or "").strip()
        if profile_path:
            candidates.append(
                (Path(profile_path).expanduser().resolve().parent / artifact).resolve()
            )
    for model_entry in spec.get("models") or []:
        model_path = str(model_entry.get("artifact_path") or "").strip()
        if model_path:
            candidates.append(
                (Path(model_path).expanduser().resolve().parent / artifact).resolve()
            )

    unique_candidates = list(dict.fromkeys(candidates))
    existing_candidates = [path for path in unique_candidates if path.is_file()]
    if not existing_candidates:
        raise WorkPackageRuntimeError(
            "linear_1x1 fusion head artifact is missing; checked: "
            + ", ".join(str(path) for path in unique_candidates)
        )
    head_path = next(
        (path for path in existing_candidates if sha256_file(path) == expected_sha),
        None,
    )
    if head_path is None:
        raise WorkPackageRuntimeError("linear_1x1 fusion head SHA256 mismatch")

    head_model, _runtime_info = load_torchscript_model(head_path, device)

    def run_head(features):
        import torch

        array = np.asarray(features, dtype=np.float32)
        tensor = torch.from_numpy(array).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            return head_model(tensor)

    return run_head


def _run_work_package_impl(
    run_spec_path: str | Path,
    package_id: str,
    *,
    job_id: int | None = None,
    lease_token: str | None = None,
    device: str | None = None,
    resume: bool = False,
    model_loader: Callable[[Mapping[str, Any], str], Any] = _default_loader,
    infer_tile: Callable[[Any, Path, str], np.ndarray] = _default_infer,
    infer_batch: Callable[[Any, list[Path], str], np.ndarray] | None = None,
    infer_images: Callable[[Any, np.ndarray, str], np.ndarray] | None = None,
    model_provider: PersistentModelProvider | None = None,
    lease_guard: Callable[[], None] | None = None,
    lease_progress: Callable[[int, int], None] | None = None,
    model_load_observer: Callable[[str, str], None] | None = None,
    preserve_lease_on_low_disk: bool = False,
    fusion_head=None,
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
    leased_job = job_id is not None or lease_token is not None
    if leased_job:
        if job_id is None or not lease_token:
            raise WorkPackageRuntimeError(
                "Work Package job requires both job_id and lease_token"
            )
        if not database.work_package_job_holds_lease(
            run_id,
            package_id,
            job_id,
            lease_token,
        ):
            raise WorkPackageRuntimeError(
                "Work Package job identity or lease does not match database state"
            )
        if str(package.get("status")) != "running":
            raise WorkPackageRuntimeError(
                "leased Work Package was not atomically marked running"
            )
    elif not database.set_work_package_status(
        run_id,
        package_id,
        "running",
        expected=("queued", "interrupted", "failed", "running"),
    ):
        raise WorkPackageRuntimeError(
            f"Work Package cannot enter running state: {package_id}"
        )
    if lease_guard is not None:
        lease_guard()
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
    configured_batch_sizes_by_model = {
        str(model_id): max(1, int(value))
        for model_id, value in (
            (
                (spec.get("resource_tuning") or {}).get("resolved") or {}
            ).get("tile_batch_size_by_model")
            or {}
        ).items()
    }
    keep_score_cache_until_package_ready = bool(
        (spec.get("runtime") or {}).get("keep_score_cache", False)
    )
    production_batch_inference = infer_images or _default_infer_batch
    package_root = run_dir / "tmp" / "work_packages" / package_id
    package_root.mkdir(parents=True, exist_ok=True)
    transform = Affine(*[float(value) for value in spec["raster"]["transform"]])
    crs = spec["raster"]["crs"]
    range_geometry = _range_geometry_for_run(spec, str(crs))
    overlap = int(spec["tile_grid"]["overlap"])
    profile = _load_profile(spec)
    active_fusion_head = fusion_head
    if (
        profile
        and str(profile.get("strategy") or "") == "linear_1x1"
        and active_fusion_head is None
    ):
        active_fusion_head = _load_linear_fusion_head(
            spec, profile, effective_device
        )
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
    model_cache_hit_count = 0
    peak_cache_bytes = 0
    tile_cache_released_count = 0
    tile_cache_retained_count = len(active_tiles)
    partition_pools: list[ThreadPoolExecutor] = []
    storage_report = dict(spec.get("storage_preflight") or {})
    storage_schema = int(storage_report.get("storage_tuning_schema_version") or 0)
    effective_min_free_bytes = int(
        storage_report.get("effective_min_free_disk_bytes")
        or float((spec.get("scaling") or {}).get("min_free_disk_gb", 0.0))
        * 1024**3
    )
    permanent_estimated_bytes = (
        int(storage_report.get("estimated_permanent_bytes") or 0)
        if storage_schema >= 2
        else 0
    )
    permanent_uncertainty_bytes = (
        int(storage_report.get("permanent_uncertainty_bytes") or 0)
        if storage_schema >= 2
        else 0
    )
    stream_ids = [
        str(item.get("stream_id") or "")
        for item in spec.get("streams") or []
        if str(item.get("stream_id") or "")
    ]
    ready_permanent_keys: set[tuple[str, str, str]] = set()
    permanent_bytes_by_key: dict[tuple[str, str, str], int] = {}
    if permanent_estimated_bytes > 0:
        for partition in database.partitions_for_run(run_id):
            core = partition["core_window"]
            core_pixels = (
                (int(core["x1"]) - int(core["x0"]))
                * (int(core["y1"]) - int(core["y0"]))
            )
            if core_pixels < 1:
                raise WorkPackageRuntimeError(
                    f"Partition Core has invalid area: {partition['partition_id']}"
                )
            partition_id = str(partition["partition_id"])
            for stream_id in stream_ids:
                permanent_bytes_by_key[(stream_id, partition_id, "core_mask")] = (
                    core_pixels * 2
                )
                permanent_bytes_by_key[
                    (stream_id, partition_id, "core_confidence")
                ] = core_pixels * 4
        if sum(permanent_bytes_by_key.values()) != permanent_estimated_bytes:
            raise WorkPackageRuntimeError(
                "frozen permanent raster reserve does not match exact Partition "
                "Core windows"
            )
        for stream_id in stream_ids:
            for kind in ("core_mask", "core_confidence"):
                ready_permanent_keys.update(
                    (stream_id, str(item["unit_id"]), kind)
                    for item in database.artifacts_for_stream(
                        run_id, stream_id, kind=kind, status="ready"
                    )
                )
    def remaining_permanent_reserve_bytes() -> int:
        if permanent_estimated_bytes <= 0:
            return 0
        remaining = sum(
            byte_count
            for key, byte_count in permanent_bytes_by_key.items()
            if key not in ready_permanent_keys
        )
        return remaining + permanent_uncertainty_bytes

    managed_budget_bytes = int(
        storage_report.get("working_cache_budget_bytes")
        or storage_report.get("resolved_score_cache_budget_bytes")
        or 0
    )
    managed_roots = (
        package_root,
        tile_cache_dir,
        run_dir / "tmp" / "probability_parts",
    )
    storage_guard = StorageGuard(
        run_dir,
        min_free_bytes=effective_min_free_bytes,
        managed_budget_bytes=managed_budget_bytes,
        initial_managed_bytes=sum(directory_size(path) for path in managed_roots),
        remaining_permanent_bytes=remaining_permanent_reserve_bytes,
    )

    def reserve_write(
        operation: str,
        write_bytes: int = 0,
        *,
        managed_growth_bytes: int | None = None,
    ):
        if lease_guard is not None:
            lease_guard()
        return storage_guard.reserve(
            operation,
            write_bytes=max(0, int(write_bytes)),
            managed_growth_bytes=managed_growth_bytes,
        )

    def reserve_managed_write(operation: str, write_bytes: int) -> int:
        if lease_guard is not None:
            lease_guard()
        return int(
            storage_guard.check(
                operation,
                write_bytes=max(0, int(write_bytes)),
                managed_growth_bytes=max(0, int(write_bytes)),
                reserve_managed_growth=True,
            )["reserved_growth_bytes"]
        )

    package_lock = _PackageFileLock(
        run_dir / "tmp" / "package_locks" / f"{package_id}.lock"
    )
    try:
        package_lock.acquire(lease_guard)
        io_workers = int((spec.get("scaling") or {}).get("tile_io_workers", 8))

        def tile_progress(current, total, result):
            if lease_progress is not None:
                lease_progress(int(current), int(total))
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
            before_write=reserve_managed_write,
            managed_delta=storage_guard.adjust,
        )
        materialized_by_id = {item["tile_id"]: item for item in materialized}
        for tile in active_tiles:
            item = materialized_by_id[str(tile["tile_id"])]
            tile["raster_path"] = item["tile_path"]
            tile["sha256"] = item["sha256"]
            if lease_guard is not None:
                lease_guard()
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
            # Keep every frozen model resident, but release allocator blocks
            # left by the preceding model's activations before loading or
            # running the next model.  The environment Batch probe uses this
            # same lifecycle; without it a later model can spuriously OOM on
            # reserved-but-unallocated CUDA memory and downgrade its Batch.
            if model_index > 1:
                _clear_accelerator_cache(effective_device)
            model_id = str(model_entry["model_id"])
            model_configured_batch_size = configured_batch_sizes_by_model.get(
                model_id, configured_batch_size
            )
            stream_id = f"model:{model_id}"
            artifact_path = Path(model_entry["artifact_path"]).resolve()
            if model_provider is not None:
                actual_sha = model_provider.verify(model_entry)
            else:
                if not artifact_path.is_file():
                    raise WorkPackageRuntimeError(
                        f"model artifact is missing: {artifact_path}"
                    )
                actual_sha = sha256_file(artifact_path)
                if actual_sha != str(model_entry["sha256"]):
                    raise WorkPackageRuntimeError(
                        f"model SHA256 mismatch: {model_id}"
                    )
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
            outputs_reusable = resume and _model_partition_outputs_reusable(
                database,
                run_id=run_id,
                stream_id=stream_id,
                model_id=model_id,
                partitions=partitions,
                crs=str(crs),
                fusion_accumulators=fusion_accumulators,
            )
            if outputs_reusable and profile:
                outputs_reusable = _restore_partition_coverage_masks(
                    database,
                    run_id=run_id,
                    stream_id=stream_id,
                    partitions=partitions,
                    coverage_masks=partition_coverage_masks,
                )
            if outputs_reusable:
                emit(
                    "package_model_outputs_reused",
                    run_id=run_id,
                    package_id=package_id,
                    stream_id=stream_id,
                    partition_count=len(partitions),
                )
                model_summaries.append(
                    {
                        "model_id": model_id,
                        "tile_count": len(active_tiles),
                        "inferred_count": 0,
                        "accepted_count": len(accepted_tiles),
                        "excluded_count": len(excluded_tiles),
                        "reused_count": len(active_tiles),
                        "reused_partition_output_count": len(partitions),
                        "configured_tile_batch_size": model_configured_batch_size,
                        "effective_tile_batch_size": model_configured_batch_size,
                        "peak_tile_batch_size": 0,
                        "inference_batch_count": 0,
                        "batch_reduction_count": 0,
                        "checkpoint_written_count": 0,
                        "checkpoint_reused_count": 0,
                        "checkpoint_written_bytes": 0,
                        "input_queue_capacity": 2,
                        "input_queue_peak_batches": 0,
                        "result_queue_capacity": 1,
                        "result_queue_peak_batches": 0,
                        "partition_queue_capacity": 1,
                        "partition_queue_peak": 0,
                        "input_wait_sec": 0.0,
                        "inference_sec": 0.0,
                        "checkpoint_write_sec": 0.0,
                        "checkpoint_wait_sec": 0.0,
                        "partition_sec": 0.0,
                        "cold_load_count": 0,
                        "cache_hit_count": 0,
                    }
                )
                continue
            model = None
            score_records_by_tile: dict[str, dict[str, Any]] = {}
            reused = 0
            accepted_count = 0
            model_effective_batch_size = (
                model_provider.effective_batch_size(
                    model_entry,
                    effective_device,
                    model_configured_batch_size,
                )
                if model_provider is not None
                else model_configured_batch_size
            )
            model_batch_count = 0
            model_peak_batch_size = 0
            model_batch_reduction_count = 0
            model_cold_load_count = 0
            model_cache_hits = 0
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
                if lease_progress is not None:
                    lease_progress(int(item["tile_index"]), len(active_tiles))
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
                    previous_bytes = sum(
                        path.stat().st_size if path.is_file() else 0
                        for path in (score_path, metadata_path)
                    )
                    estimated_write_bytes = int(probabilities.nbytes) + 64 * 1024
                    reservation = reserve_write(
                        f"accepted_score:{tile_id}",
                        estimated_write_bytes,
                        managed_growth_bytes=max(
                            0, estimated_write_bytes - previous_bytes
                        ),
                    )
                    try:
                        _atomic_npz(
                            score_path,
                            probabilities=probabilities.astype(np.float16),
                        )
                        _atomic_json(metadata_path, expected)
                    finally:
                        current_bytes = sum(
                            path.stat().st_size if path.is_file() else 0
                            for path in (score_path, metadata_path)
                        )
                        reservation.settle(current_bytes - previous_bytes)
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
                (
                    sequence,
                    inferable_items[
                        offset : offset + model_configured_batch_size
                    ],
                )
                for sequence, offset in enumerate(
                    range(0, len(inferable_items), model_configured_batch_size)
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
                    discarded = discard_checkpoint(score_batch_root, sequence)
                    cleaned_bytes += discarded
                    storage_guard.released(discarded)
                    missing_groups.append((sequence, group))
                    continue
                checkpoint_reused_count += 1
                reused += len(group)
                for item, record in zip(group, records):
                    record_score(item, record)

            if missing_groups:
                if lease_guard is not None:
                    lease_guard()
                if model_provider is not None:
                    model, cold_loaded = model_provider.get(
                        model_entry,
                        effective_device,
                        observer=model_load_observer,
                    )
                    if cold_loaded:
                        model_cold_load_count = 1
                        model_load_count += 1
                    else:
                        model_cache_hits = 1
                        model_cache_hit_count += 1
                else:
                    model = model_loader(model_entry, effective_device)
                    model_cold_load_count = 1
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
                arrays, range_report = apply_range_mask_to_core(
                    arrays,
                    partition,
                    global_transform=transform,
                    range_geometry=range_geometry,
                )
                probability_path = (
                    run_dir
                    / "tmp"
                    / "probability_parts"
                    / model_id
                    / f"{partition_id}.tif"
                )
                raster_root = run_dir / "models" / model_id / "raster_parts"
                previous_probability_bytes = (
                    probability_path.stat().st_size
                    if probability_path.is_file()
                    else 0
                )
                probability_write_bytes = int(
                    np.asarray(arrays["halo_probabilities"]).size * 2
                )
                permanent_write_bytes = int(
                    np.asarray(arrays["core_mask"]).nbytes
                    + np.asarray(arrays["core_confidence"]).nbytes
                )
                raster_write_overhead_bytes = 3 * 64 * 1024
                reservation = reserve_write(
                    f"partition_rasters:{stream_id}:{partition_id}",
                    probability_write_bytes
                    + permanent_write_bytes
                    + raster_write_overhead_bytes,
                    managed_growth_bytes=max(
                        0,
                        probability_write_bytes
                        + 64 * 1024
                        - previous_probability_bytes,
                    ),
                )
                try:
                    paths = write_partition_rasters(
                        arrays,
                        partition,
                        global_transform=transform,
                        crs=crs,
                        output_probability=probability_path,
                        output_mask=raster_root / f"{partition_id}_mask.tif",
                        output_confidence=raster_root
                        / f"{partition_id}_confidence.tif",
                        core_mask_tags=core_mask_tags(
                            {
                                "authority": "partition_core_argmax_v1",
                                **range_report,
                            }
                        ),
                    )
                finally:
                    current_probability_bytes = (
                        probability_path.stat().st_size
                        if probability_path.is_file()
                        else 0
                    )
                    reservation.settle(
                        current_probability_bytes - previous_probability_bytes
                    )
                for kind, key in (
                    ("core_mask", "mask"),
                    ("core_confidence", "confidence"),
                    ("partition_probability", "probability"),
                ):
                    if lease_guard is not None:
                        lease_guard()
                    _commit_artifact(
                        database,
                        run_id,
                        path=Path(paths[key]),
                        kind=kind,
                        stream_id=stream_id,
                        unit_id=partition_id,
                    )
                    if kind in {"core_mask", "core_confidence"}:
                        ready_permanent_keys.add((stream_id, partition_id, kind))
                if profile:
                    coverage = arrays["halo_weights"] > 0
                    previous_coverage = partition_coverage_masks.get(partition_id)
                    if previous_coverage is None:
                        partition_coverage_masks[partition_id] = coverage
                    elif not np.array_equal(previous_coverage, coverage):
                        raise WorkPackageRuntimeError(
                            f"model coverage differs inside Partition: {partition_id}"
                        )
                    accumulator = fusion_accumulators[partition_id]
                    accumulator_before = directory_size(accumulator.root)
                    halo_probabilities = np.asarray(
                        arrays["halo_probabilities"]
                    )
                    accumulator_channels = (
                        len(list(profile.get("models") or [])) * 14
                        if str(profile.get("strategy") or "") == "linear_1x1"
                        else 14
                    )
                    accumulator_write_bytes = int(
                        accumulator_channels
                        * halo_probabilities.shape[1]
                        * halo_probabilities.shape[2]
                        * np.dtype(np.float32).itemsize
                    )
                    accumulator_estimated_write_bytes = (
                        accumulator_write_bytes + 64 * 1024
                    )
                    completed_accumulator_models = (
                        accumulator.completed_model_ids()
                    )
                    generation = len(completed_accumulator_models) + 1
                    next_accumulator_path = (
                        accumulator.root
                        / f"accumulator_{generation:03d}.npy"
                    )
                    previous_accumulator_path = (
                        accumulator.root
                        / f"accumulator_{generation - 1:03d}.npy"
                        if generation > 1
                        else None
                    )
                    replaced_accumulator_bytes = (
                        next_accumulator_path.stat().st_size
                        if next_accumulator_path.is_file()
                        else 0
                    ) + (
                        previous_accumulator_path.stat().st_size
                        if previous_accumulator_path is not None
                        and previous_accumulator_path.is_file()
                        else 0
                    )
                    reservation = reserve_write(
                        f"fusion_accumulator:{partition_id}:{model_id}",
                        accumulator_estimated_write_bytes,
                        # Physical pending reserves the whole next generation
                        # while the prior one coexists.  Managed growth models
                        # the final directory: replace an existing target and
                        # delete only the prior active generation, never
                        # subtract unrelated files from the whole directory.
                        managed_growth_bytes=max(
                            0,
                            accumulator_estimated_write_bytes
                            - replaced_accumulator_bytes,
                        ),
                    )
                    try:
                        accumulator.add_model(
                            model_id, arrays["halo_probabilities"]
                        )
                    finally:
                        reservation.settle(
                            directory_size(accumulator.root) - accumulator_before
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

            managed_score_cache_bytes = directory_size(score_batch_root)
            probability_bytes_per_tile = int(
                storage_report.get("current_model_probability_bytes") or 0
            )
            score_cache_high_water_bytes = (
                len(inferable_items) * probability_bytes_per_tile
                + len(groups) * CHECKPOINT_WRITE_OVERHEAD_BYTES
                if storage_schema >= 2 and probability_bytes_per_tile > 0
                else 0
            )
            non_score_cache_baseline = max(
                0,
                directory_size(package_root)
                + directory_size(tile_cache_dir)
                - managed_score_cache_bytes,
            )

            def write_batch(
                sequence: int,
                group: list[dict[str, Any]],
                probabilities: np.ndarray,
                current_score_cache_bytes: int,
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
                    managed_cache_bytes=current_score_cache_bytes,
                    managed_cache_budget_bytes=(
                        score_cache_high_water_bytes
                        if score_cache_high_water_bytes > 0
                        else None
                    ),
                    storage_guard=storage_guard,
                    storage_operation=(
                        f"score_checkpoint:{model_id}:{sequence}"
                    ),
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
                nonlocal managed_score_cache_bytes
                if writer_future is None or writer_group is None:
                    return
                wait_started = time.monotonic()
                records, manifest, write_elapsed = writer_future.result()
                checkpoint_wait_sec += time.monotonic() - wait_started
                checkpoint_write_sec += float(write_elapsed)
                checkpoint_written_count += 1
                checkpoint_written_bytes += int(manifest["byte_count"])
                managed_score_cache_bytes = directory_size(score_batch_root)
                for item, record in zip(writer_group, records):
                    record_score(item, record)
                peak_cache_bytes = max(
                    peak_cache_bytes,
                    non_score_cache_baseline + managed_score_cache_bytes,
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
                                if (
                                    attempt_size <= 1
                                    or not _is_recoverable_batch_error(
                                        error, effective_device
                                    )
                                ):
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
                            if (
                                model_provider is not None
                                and model_batch_reduction_count > 0
                                and attempt_size == model_effective_batch_size
                            ):
                                # Persist a downgrade only after that exact
                                # smaller batch has completed successfully.
                                model_provider.remember_batch_size(
                                    model_entry,
                                    effective_device,
                                    model_effective_batch_size,
                                )
                            group_outputs.append(
                                np.asarray(probabilities_batch, dtype=np.float16)
                            )
                            cursor += attempt_size
                        probabilities = np.concatenate(group_outputs, axis=0)
                        drain_writer()
                        writer_group = group
                        writer_future = writer_pool.submit(
                            write_batch,
                            sequence,
                            group,
                            probabilities,
                            managed_score_cache_bytes,
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
            if not keep_score_cache_until_package_ready:
                removed = _remove_tree_with_count(score_batch_root)
                cleaned_bytes += removed
                storage_guard.released(removed)
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
                    "configured_tile_batch_size": model_configured_batch_size,
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
                    "cold_load_count": model_cold_load_count,
                    "cache_hit_count": model_cache_hits,
                }
            )

        # Release the final model's activation cache for Fusion and the next
        # Work Package without unloading PersistentModelProvider models.
        _clear_accelerator_cache(effective_device)

        if profile:
            stream_id = f"fusion:{fusion_id}"
            database.set_stream_status(run_id, stream_id, "running")
            for partition in partitions:
                partition_id = partition["partition_id"]
                if lease_guard is not None:
                    lease_guard()
                probabilities = fusion_accumulators[partition_id].finalize(
                    fusion_head=active_fusion_head
                )
                coverage = partition_coverage_masks[partition_id]
                probabilities[:, ~coverage] = 0.0
                arrays = derive_partition_arrays(
                    probabilities,
                    partition,
                    weights=coverage.astype(np.float32),
                )
                if bool(
                    (spec.get("fragmentation_regularization") or {}).get(
                        "enabled", True
                    )
                ):
                    arrays, regularization = regularize_partition_core(
                        arrays,
                        partition,
                        global_transform=transform,
                        crs=str(crs),
                        range_geometry=range_geometry,
                    )
                else:
                    arrays, range_report = apply_range_mask_to_core(
                        arrays,
                        partition,
                        global_transform=transform,
                        range_geometry=range_geometry,
                    )
                    regularization = {
                        "authority": "partition_core_argmax_v1",
                        "changed_pixel_count": 0,
                        "changed_component_count": 0,
                        **range_report,
                    }
                probability_path = (
                    run_dir / "tmp" / "probability_parts" / f"fusion_{fusion_id}" / f"{partition_id}.tif"
                )
                raster_root = run_dir / "fusion" / fusion_id / "raster_parts"
                previous_probability_bytes = (
                    probability_path.stat().st_size
                    if probability_path.is_file()
                    else 0
                )
                probability_write_bytes = int(probabilities.size * 2)
                permanent_write_bytes = int(
                    np.asarray(arrays["core_mask"]).nbytes
                    + np.asarray(arrays["core_confidence"]).nbytes
                )
                raster_write_overhead_bytes = 3 * 64 * 1024
                reservation = reserve_write(
                    f"partition_rasters:{stream_id}:{partition_id}",
                    probability_write_bytes
                    + permanent_write_bytes
                    + raster_write_overhead_bytes,
                    managed_growth_bytes=max(
                        0,
                        probability_write_bytes
                        + 64 * 1024
                        - previous_probability_bytes,
                    ),
                )
                try:
                    paths = write_partition_rasters(
                        arrays,
                        partition,
                        global_transform=transform,
                        crs=crs,
                        output_probability=probability_path,
                        output_mask=raster_root / f"{partition_id}_mask.tif",
                        output_confidence=raster_root
                        / f"{partition_id}_confidence.tif",
                        core_mask_tags=core_mask_tags(regularization),
                    )
                finally:
                    current_probability_bytes = (
                        probability_path.stat().st_size
                        if probability_path.is_file()
                        else 0
                    )
                    reservation.settle(
                        current_probability_bytes - previous_probability_bytes
                    )
                # Publish the permanent Core products first.  The probability
                # artifact is the unit-fit dependency gate, so publishing it
                # last makes every Core/Seam/Junction read the authoritative
                # V3-cleaned classes, never a transient argmax crop.
                for kind, key in (
                    ("core_mask", "mask"),
                    ("core_confidence", "confidence"),
                    ("partition_probability", "probability"),
                ):
                    if lease_guard is not None:
                        lease_guard()
                    _commit_artifact(
                        database,
                        run_id,
                        path=Path(paths[key]),
                        kind=kind,
                        stream_id=stream_id,
                        unit_id=partition_id,
                    )
                    if kind in {"core_mask", "core_confidence"}:
                        ready_permanent_keys.add((stream_id, partition_id, kind))
                emit(
                    "authoritative_raster_ready",
                    run_id=run_id,
                    stream_id=stream_id,
                    partition_id=partition_id,
                    changed_pixel_count=int(
                        regularization.get("changed_pixel_count", 0)
                    ),
                    changed_component_count=int(
                        regularization.get("changed_component_count", 0)
                    ),
                    authority=str(regularization["authority"]),
                )
            removed = _remove_tree_with_count(
                package_root / "fusion" / fusion_id
            )
            cleaned_bytes += removed
            storage_guard.released(removed)
            fusion_root = package_root / "fusion"
            if fusion_root.is_dir() and not any(fusion_root.iterdir()):
                fusion_root.rmdir()
        if keep_score_cache_until_package_ready:
            # "keep" means keep checkpoints through every model/Fusion step
            # so a failed Package can resume.  Once the complete Package is
            # ready to commit, the cache is no longer a durable output and is
            # removed rather than accumulating across the Run.
            removed = _remove_tree_with_count(package_root / "score_batches")
            cleaned_bytes += removed
            storage_guard.released(removed)
        removed = _remove_tree_with_count(package_root / "accepted_scores")
        cleaned_bytes += removed
        storage_guard.released(removed)
        tile_cleaned_bytes = 0
        releasable_tile_ids = set(
            database.releasable_package_tile_ids(run_id, package_id)
        )
        released_tile_count = 0
        for item in materialized:
            if str(item["tile_id"]) not in releasable_tile_ids:
                continue
            if lease_guard is not None:
                lease_guard()
            tile_cleaned_bytes += _unlink_with_count(
                _owned_tile_cache_file(item["tile_path"], tile_cache_dir)
            )
            tile_cleaned_bytes += _unlink_with_count(
                _owned_tile_cache_file(item["metadata_path"], tile_cache_dir)
            )
            released_tile_count += 1
        cleaned_bytes += tile_cleaned_bytes
        storage_guard.released(tile_cleaned_bytes)
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
            "model_cache_hit_count": model_cache_hit_count,
            "configured_tile_batch_size": configured_batch_size,
            "configured_tile_batch_sizes_by_model": dict(
                sorted(configured_batch_sizes_by_model.items())
            ),
            "score_cache_retention": (
                "until_package_ready"
                if keep_score_cache_until_package_ready
                else "until_model_partition_commit"
            ),
            "peak_cache_bytes": max(
                peak_cache_bytes, storage_guard.peak_managed_bytes
            ),
            "peak_rss_bytes": peak_rss_bytes(),
            "cleaned_bytes": cleaned_bytes,
            "tile_cache_released_count": tile_cache_released_count,
            "tile_cache_retained_count": tile_cache_retained_count,
            "elapsed_sec": round(time.monotonic() - started_at, 3),
            "status": "ready",
        }
        if lease_guard is not None:
            lease_guard()
        _atomic_json(package_root / "package_report.json", result)
        if leased_job:
            if lease_guard is not None:
                lease_guard()
            if not database.complete_work_package_job(
                run_id,
                package_id,
                job_id,
                lease_token,
            ):
                raise WorkPackageRuntimeError(
                    "Work Package job lease expired before atomic commit"
                )
        elif not database.set_work_package_status(
            run_id,
            package_id,
            "ready",
            expected="running",
        ):
            raise WorkPackageRuntimeError(
                f"Work Package cannot enter ready state: {package_id}"
            )
        emit("work_package_finished", **result)
        return result
    except (ScoreBatchDiskReserveError, StorageReserveError) as error:
        for pool in partition_pools:
            pool.shutdown(wait=True, cancel_futures=True)
        partition_pools.clear()
        if not _storage_error_is_transient(error):
            transition = None
            if leased_job:
                transition = (
                    "failed"
                    if database.fail_work_package_job(
                        run_id,
                        package_id,
                        job_id,
                        lease_token,
                        error=str(error),
                    )
                    else None
                )
            else:
                database.set_work_package_status(
                    run_id, package_id, "failed", expected="running"
                )
                transition = "failed"
            database.append_event(
                run_id,
                "work_package_storage_contract_failed",
                level="error",
                message=str(error),
                payload={"package_id": package_id, "transition": transition},
            )
            raise
        if not preserve_lease_on_low_disk:
            if leased_job:
                database.interrupt_work_package_job(
                    run_id,
                    package_id,
                    job_id,
                    lease_token,
                    error=str(error),
                )
            else:
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
    except WorkerStopRequested as error:
        for pool in partition_pools:
            pool.shutdown(wait=True, cancel_futures=True)
        partition_pools.clear()
        if leased_job:
            database.interrupt_work_package_job(
                run_id,
                package_id,
                job_id,
                lease_token,
                error=str(error),
            )
        else:
            database.set_work_package_status(
                run_id, package_id, "interrupted", expected="running"
            )
        database.append_event(
            run_id,
            "work_package_interrupted",
            level="warning",
            message=str(error),
            payload={"package_id": package_id},
        )
        raise
    except LeaseLostError as error:
        for pool in partition_pools:
            pool.shutdown(wait=True, cancel_futures=True)
        partition_pools.clear()
        # The current process is fenced out.  It must not change either the
        # Package or Job now owned by a newer lease.
        database.append_event(
            run_id,
            "work_package_lease_lost",
            level="warning",
            message=str(error),
            payload={"package_id": package_id, "job_id": job_id},
        )
        raise
    except Exception as error:
        for pool in partition_pools:
            pool.shutdown(wait=True, cancel_futures=True)
        partition_pools.clear()
        transition = None
        if leased_job:
            transition = database.fail_or_requeue_work_package_job(
                run_id,
                package_id,
                job_id,
                lease_token,
                error=str(error),
            )
        else:
            database.set_work_package_status(
                run_id, package_id, "failed", expected="running"
            )
            transition = "failed"
        database.append_event(
            run_id,
            "work_package_failed",
            level="error",
            message=str(error),
            payload={"package_id": package_id, "transition": transition},
        )
        raise
    finally:
        package_lock.release()


def run_work_package(
    run_spec_path: str | Path,
    package_id: str,
    *,
    job_id: int | None = None,
    lease_token: str | None = None,
    device: str | None = None,
    resume: bool = False,
    model_loader: Callable[[Mapping[str, Any], str], Any] = _default_loader,
    infer_tile: Callable[[Any, Path, str], np.ndarray] = _default_infer,
    infer_batch: Callable[[Any, list[Path], str], np.ndarray] | None = None,
    infer_images: Callable[[Any, np.ndarray, str], np.ndarray] | None = None,
    model_provider: PersistentModelProvider | None = None,
    lease_guard: Callable[[], None] | None = None,
    lease_progress: Callable[[int, int], None] | None = None,
    model_load_observer: Callable[[str, str], None] | None = None,
    preserve_lease_on_low_disk: bool = False,
    fusion_head=None,
) -> dict[str, Any]:
    """Execute one Package and close every leased failure path atomically."""

    def repair_leased_state(error: BaseException, transition: str) -> None:
        if job_id is None or not lease_token:
            return
        try:
            spec = load_json(Path(run_spec_path).resolve())
            database = RunStateDB(spec["state_db"])
            run_id = str(spec["run_id"])
            if transition == "interrupted":
                database.interrupt_work_package_job(
                    run_id,
                    package_id,
                    int(job_id),
                    str(lease_token),
                    error=str(error),
                )
            elif transition == "failed":
                database.fail_work_package_job(
                    run_id,
                    package_id,
                    int(job_id),
                    str(lease_token),
                    error=str(error),
                )
            else:
                database.fail_or_requeue_work_package_job(
                    run_id,
                    package_id,
                    int(job_id),
                    str(lease_token),
                    error=str(error),
                )
        except Exception:
            # Preserve the original failure. Expired/stolen leases are healed
            # by normal recovery and must not be mutated by this process.
            return

    try:
        return _run_work_package_impl(
            run_spec_path,
            package_id,
            job_id=job_id,
            lease_token=lease_token,
            device=device,
            resume=resume,
            model_loader=model_loader,
            infer_tile=infer_tile,
            infer_batch=infer_batch,
            infer_images=infer_images,
            model_provider=model_provider,
            lease_guard=lease_guard,
            lease_progress=lease_progress,
            model_load_observer=model_load_observer,
            preserve_lease_on_low_disk=preserve_lease_on_low_disk,
            fusion_head=fusion_head,
        )
    except (ScoreBatchDiskReserveError, StorageReserveError) as error:
        if not (
            preserve_lease_on_low_disk
            and _storage_error_is_transient(error)
        ):
            repair_leased_state(
                error,
                "interrupted" if _storage_error_is_transient(error) else "failed",
            )
        raise
    except WorkerStopRequested as error:
        repair_leased_state(error, "interrupted")
        raise
    except LeaseLostError:
        raise
    except Exception as error:
        repair_leased_state(error, "retry")
        raise


def run_persistent_worker(
    run_spec_path: str | Path,
    worker_id: str,
    *,
    device: str | None = None,
    resume: bool = True,
    max_open_frontier_units: int = 64,
    stop_event: threading.Event | None = None,
    heartbeat_interval_sec: float = 15.0,
    lease_seconds: int = 120,
    low_disk_poll_sec: float = 30.0,
    model_provider: PersistentModelProvider | None = None,
    infer_tile: Callable[[Any, Path, str], np.ndarray] = _default_infer,
    infer_batch: Callable[[Any, list[Path], str], np.ndarray] | None = None,
    infer_images: Callable[[Any, np.ndarray, str], np.ndarray] | None = None,
    fusion_head=None,
) -> dict[str, Any]:
    """Lease and execute Packages serially while keeping models resident."""

    started_at = time.monotonic()
    spec_path = Path(run_spec_path).resolve()
    spec = load_json(spec_path)
    if int(spec.get("schema_version") or 0) != 2:
        raise WorkPackageRuntimeError(
            "persistent worker requires run_spec schema_version 2"
        )
    run_id = str(spec["run_id"])
    run_dir = Path(spec["run_dir"]).resolve()
    database = RunStateDB(spec["state_db"])
    stopper = stop_event or threading.Event()
    provider = model_provider or PersistentModelProvider()
    profile = _load_profile(spec)
    requested_device = (
        device
        or (spec.get("runtime") or {}).get("effective_device")
        or "auto"
    )
    effective_device = resolve_device(str(requested_device))
    if not validate_device(effective_device):
        raise WorkPackageRuntimeError(
            f"semantic device is unavailable: {effective_device}"
        )
    resident_fusion_head = fusion_head
    if (
        profile
        and str(profile.get("strategy") or "") == "linear_1x1"
        and resident_fusion_head is None
    ):
        resident_fusion_head = _load_linear_fusion_head(
            spec, profile, effective_device
        )
    package_count = 0
    ready_count = 0
    failure_count = 0
    low_disk_pause_count = 0
    heartbeat_count = 0
    package_ids: list[str] = []
    worker_session_id = uuid.uuid4().hex
    model_event_path = run_dir / "logs" / "accelerator_model_loads.jsonl"
    model_event_lock = threading.Lock()

    def record_model_load(model_id: str, event: str) -> None:
        if event not in {"load_started", "load_completed", "cache_hit"}:
            raise WorkPackageRuntimeError(
                f"unknown accelerator model-load event: {event}"
            )
        record = {
            "schema_version": 2,
            "run_id": run_id,
            "worker_id": str(worker_id),
            "worker_session_id": worker_session_id,
            "pid": os.getpid(),
            "timestamp": time.time(),
            "model_id": str(model_id),
            "event": event,
        }
        model_event_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        with model_event_lock:
            with model_event_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    session_fenced = False
    while not stopper.is_set() and not session_fenced:
        if database.job_counts(run_id, job_type="work_package").get("failed", 0):
            # One exhausted Package makes the complete local Run impossible.
            # Do not spend accelerator time on later Packages after that hard
            # gate has already failed.
            break
        job = database.lease_next_work_package(
            run_id,
            str(worker_id),
            max_open_frontier_units=max(1, int(max_open_frontier_units)),
            lease_seconds=max(30, int(lease_seconds)),
        )
        if job is None:
            break
        package_id = str(job["package_id"])
        package_ids.append(package_id)
        package_count += 1
        heartbeat = _LeaseHeartbeat(
            spec["state_db"],
            run_id=run_id,
            package_id=package_id,
            job_id=int(job["job_id"]),
            lease_token=str(job["lease_token"]),
            stop_event=stopper,
            interval_sec=heartbeat_interval_sec,
            lease_seconds=lease_seconds,
        )
        try:
            heartbeat.start()
            while not stopper.is_set():
                try:
                    run_work_package(
                        spec_path,
                        package_id,
                        job_id=int(job["job_id"]),
                        lease_token=str(job["lease_token"]),
                        device=device,
                        resume=resume,
                        model_provider=provider,
                        lease_guard=heartbeat.check,
                        lease_progress=heartbeat.update_progress,
                        model_load_observer=record_model_load,
                        preserve_lease_on_low_disk=True,
                        infer_tile=infer_tile,
                        infer_batch=infer_batch,
                        infer_images=infer_images,
                        fusion_head=resident_fusion_head,
                    )
                    ready_count += 1
                    break
                except (ScoreBatchDiskReserveError, StorageReserveError) as error:
                    if not _storage_error_is_transient(error):
                        failure_count += 1
                        break
                    low_disk_pause_count += 1
                    emit(
                        "accelerator_worker_paused_low_disk",
                        run_id=run_id,
                        worker_id=str(worker_id),
                        package_id=package_id,
                        pause_count=low_disk_pause_count,
                        error=str(error),
                    )
                    if stopper.wait(max(0.05, float(low_disk_poll_sec))):
                        break
                    # Keep the exact lease and reuse committed checkpoints.
                    heartbeat.check()
                    continue
                except WorkerStopRequested:
                    break
                except LeaseLostError:
                    failure_count += 1
                    session_fenced = True
                    break
                except Exception:
                    # run_work_package has already made the exact leased
                    # Package+Job retry/failure decision atomically.
                    failure_count += 1
                    break
        except WorkerStopRequested:
            pass
        except LeaseLostError:
            failure_count += 1
            session_fenced = True
        except Exception as error:
            # Covers failures between leasing and entering run_work_package,
            # including heartbeat startup. Do not leave the exact lease running.
            try:
                if database.work_package_job_holds_lease(
                    run_id,
                    package_id,
                    int(job["job_id"]),
                    str(job["lease_token"]),
                ):
                    database.fail_or_requeue_work_package_job(
                        run_id,
                        package_id,
                        int(job["job_id"]),
                        str(job["lease_token"]),
                        error=str(error),
                    )
            except Exception:
                pass
            failure_count += 1
        finally:
            heartbeat_count += heartbeat.heartbeat_count
            heartbeat.close()
        if stopper.is_set():
            database.interrupt_work_package_worker(run_id, str(worker_id))
            break
        if session_fenced:
            # Losing one Package lease fences this persistent accelerator
            # session.  It must not lease another Package while a replacement
            # session may already be running on the same GPU.
            break

    counts = database.job_counts(run_id, job_type="work_package")
    job_total = sum(int(value) for value in counts.values())
    if stopper.is_set():
        status = "stopped"
    elif counts.get("failed", 0):
        status = "failed"
    elif job_total > 0 and int(counts.get("ready", 0)) == job_total:
        status = "ready"
    else:
        # A worker may temporarily find no leasable job while another lease is
        # running, a dependency is blocked, or an attempt limit is exhausted.
        # None of those states proves successful completion.
        status = "incomplete"
    report = {
        "schema_version": 1,
        "status": status,
        "run_id": run_id,
        "worker_id": str(worker_id),
        "worker_session_id": worker_session_id,
        "pid": os.getpid(),
        "requested_device": str(device or ""),
        "package_attempt_count": package_count,
        "package_ready_count": ready_count,
        "package_failure_count": failure_count,
        "package_ids": package_ids,
        "low_disk_pause_count": low_disk_pause_count,
        "heartbeat_count": heartbeat_count,
        "session_fenced": session_fenced,
        "model_cold_load_counts": dict(sorted(provider.cold_load_counts.items())),
        "model_cache_hit_counts": dict(sorted(provider.cache_hit_counts.items())),
        "model_effective_batch_sizes": provider.effective_batch_sizes,
        "fusion_head_loaded": resident_fusion_head is not None,
        "peak_rss_bytes": peak_rss_bytes(),
        "elapsed_sec": round(time.monotonic() - started_at, 3),
        "job_counts": counts,
    }
    _atomic_json(
        run_dir / "logs" / "accelerator_workers" / f"{worker_session_id}.json",
        report,
    )
    _atomic_json(run_dir / "logs" / "accelerator_worker_report.json", report)
    emit("accelerator_worker_finished", **report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run bounded semantic Work Packages"
    )
    parser.add_argument("--run-spec", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--worker-id")
    mode.add_argument("--package-id")
    parser.add_argument("--job-id", type=int)
    parser.add_argument("--lease-token")
    parser.add_argument("--device")
    parser.add_argument("--max-open-frontier-units", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.worker_id:
            stop_event = threading.Event()

            def request_stop(_signum, _frame):
                stop_event.set()

            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                signal.signal(signum, request_stop)
            report = run_persistent_worker(
                args.run_spec,
                args.worker_id,
                device=args.device,
                resume=True,
                max_open_frontier_units=args.max_open_frontier_units,
                stop_event=stop_event,
            )
            return 0 if report["status"] in {"ready", "stopped"} else 2
        if args.job_id is None or not args.lease_token:
            parser.error(
                "single-package mode requires --job-id and --lease-token"
            )
        run_work_package(
            args.run_spec,
            args.package_id,
            job_id=args.job_id,
            lease_token=args.lease_token,
            device=args.device,
            resume=args.resume,
        )
        return 0
    except Exception as error:
        emit(
            "accelerator_worker_failed" if args.worker_id else "work_package_failed",
            worker_id=args.worker_id or "",
            package_id=args.package_id or "",
            error=str(error),
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
