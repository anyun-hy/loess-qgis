"""Execute one bounded Work Package across all selected semantic streams."""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
import shutil
import signal
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
from affine import Affine

from labeling_tool.shared.contracts.run_spec import (
    RunSpecError,
    load_json,
    sha256_file,
    validated_run_tile_cache_dir,
)
from labeling_tool.shared.state.run_state_db import RunStateDB, run_state_from_spec
from loess_runtime.assembly.range_clip_runtime import (
    RangeClipRuntimeError,
    extract_range_mask_geometry,
)
from loess_runtime.geometry.authoritative_raster import (
    apply_range_mask_to_core,
    core_mask_tags,
    regularize_partition_core,
)
from loess_runtime.inference.accepted_score import accepted_probabilities
from loess_runtime.inference.model_score_execution import (
    build_model_batch_inference,
    execute_model_scores,
    is_recoverable_batch_error,
)
from loess_runtime.inference.partition_build_pipeline import PartitionBuildPipeline
from loess_runtime.inference.partition_mosaic import (
    derive_partition_arrays,
    write_partition_rasters,
)
from loess_runtime.inference.persistent_package_session import (
    execute_leased_package_session,
)
from loess_runtime.inference.score_batch_cache import (
    CHECKPOINT_WRITE_OVERHEAD_BYTES,
    ScoreBatchDiskReserveError,
    discard_checkpoint,
    load_checkpoint,
    remove_owned_temporary_files,
    write_checkpoint,
)
from loess_runtime.inference.score_checkpoint_writer import CompletedCheckpoint
from loess_runtime.inference.semantic_batch import (
    read_model_tile,
    run_model_batch,
    run_model_tile,
    write_atomic_json,
    write_atomic_npz,
)
from loess_runtime.inference.tile_materializer import materialize_package_tiles
from loess_runtime.inference.torchscript_runtime import load_torchscript_model
from loess_runtime.inference.work_package_model_partitions import (
    WorkPackageModelPartitions,
)
from loess_runtime.inference.work_package_storage import WorkPackageStorageBudget
from loess_runtime.system import runtime_errors
from loess_runtime.system._device import resolve_device, validate_device
from loess_runtime.system.artifact_publication import publish_artifact
from loess_runtime.system.runtime_errors import WorkPackageRuntimeError
from loess_runtime.system.runtime_metrics import directory_size, peak_rss_bytes
from loess_runtime.system.storage_guard import (
    StorageReserveError,
    remaining_deferred_temporary_reserve_bytes,
)


def _range_geometry_for_run(spec: Mapping[str, Any], crs: str):
    """Resolve the one frozen exact boundary before a Package starts work."""

    try:
        return extract_range_mask_geometry(spec, crs)
    except RangeClipRuntimeError as error:
        raise WorkPackageRuntimeError(str(error)) from error


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
        json.dumps(
            {"event": event, **payload}, ensure_ascii=False, separators=(",", ":")
        ),
        flush=True,
    )


def _default_loader(model_entry: Mapping[str, Any], device: str):
    return load_torchscript_model(Path(model_entry["artifact_path"]), device)[0]


class PersistentModelProvider:
    """Verify and load each frozen model at most once per worker process."""

    def __init__(
        self,
        loader: Callable[[Mapping[str, Any], str], Any] = _default_loader,
        *,
        batch_state_path: str | Path | None = None,
    ) -> None:
        self._loader = loader
        self._batch_state_path = (
            Path(batch_state_path).resolve() if batch_state_path else None
        )
        self._verified: dict[tuple[str, str], str] = {}
        self._models: dict[tuple[str, str, str, str], Any] = {}
        self._effective_batch_sizes = self._load_batch_state()
        self.cold_load_counts: dict[str, int] = {}
        self.cache_hit_counts: dict[str, int] = {}

    def _load_batch_state(self) -> dict[tuple[str, str], int]:
        path = self._batch_state_path
        if path is None or not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("batch state must be an object")
            if payload.get("schema_version") != 1:
                raise ValueError("unsupported schema_version")
            limits = payload.get("limits")
            if not isinstance(limits, Mapping):
                raise ValueError("limits must be an object")
            state = {}
            for raw_key, raw_value in limits.items():
                model_id, separator, device = str(raw_key).rpartition("@")
                value = int(raw_value)
                if not separator or not model_id or not device or value < 1:
                    raise ValueError(f"invalid batch limit: {raw_key}")
                state[(model_id, device)] = value
            return state
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise WorkPackageRuntimeError(
                f"invalid accelerator batch state {path}: {error}"
            ) from error

    def _persist_batch_state(self) -> None:
        if self._batch_state_path is None:
            return
        write_atomic_json(
            self._batch_state_path,
            {
                "schema_version": 1,
                "limits": self.effective_batch_sizes,
            },
        )

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
            self.cache_hit_counts[model_id] = self.cache_hit_counts.get(model_id, 0) + 1
            if observer is not None:
                observer(model_id, "cache_hit")
            return self._models[key], False
        if observer is not None:
            observer(model_id, "load_started")
        model = self._loader(model_entry, device)
        self._models[key] = model
        self.cold_load_counts[model_id] = self.cold_load_counts.get(model_id, 0) + 1
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
        remembered = value if previous is None else min(int(previous), value)
        if previous == remembered:
            return
        self._effective_batch_sizes[key] = remembered
        self._persist_batch_state()

    @property
    def effective_batch_sizes(self) -> dict[str, int]:
        return {
            f"{model_id}@{device}": int(value)
            for (model_id, device), value in sorted(self._effective_batch_sizes.items())
        }


class _LeaseHeartbeat:
    """Own one Package lease and fence all irreversible write boundaries."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        database_schema: str | None = None,
        run_id: str,
        package_id: str,
        job_id: int,
        lease_token: str,
        stop_event: threading.Event,
        interval_sec: float = 15.0,
        lease_seconds: int = 120,
    ) -> None:
        self.database_path = str(database_path)
        self.database_schema = str(database_schema or "").strip() or None
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
                database = RunStateDB(
                    self.database_path,
                    postgres_schema=self.database_schema,
                )
                accepted = database.jobs.heartbeat(
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
            raise runtime_errors.WorkerStopRequested(
                "accelerator worker stop requested"
            )
        if self.lost_event.is_set():
            detail = self._heartbeat_error or "heartbeat was rejected"
            raise runtime_errors.LeaseLostError(f"Work Package lease lost: {detail}")
        database = RunStateDB(
            self.database_path,
            postgres_schema=self.database_schema,
        )
        if not database.jobs.work_package_job_holds_lease(
            self.run_id,
            self.package_id,
            self.job_id,
            self.lease_token,
        ):
            self.lost_event.set()
            raise runtime_errors.LeaseLostError("Work Package lease is no longer owned")

    def close(self) -> None:
        self._close_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_sec + 1.0)
            self._thread = None


HOST_PIPELINE_BUDGET_BYTES = 1024**3
# Conservative host-side allowance: read futures and stacked inputs (up to
# 64-bit integers), float32 output/copy, half output and one writer batch.
# Model activations and partition geometry have separate resource budgets.
HOST_PIPELINE_BYTES_PER_TILE = 64 * 1024**2


def _host_pipeline_batch_limit(configured: int) -> int:
    return max(
        1,
        min(
            int(configured), HOST_PIPELINE_BUDGET_BYTES // HOST_PIPELINE_BYTES_PER_TILE
        ),
    )


def _default_infer(model: Any, tile_path: Path, device: str) -> np.ndarray:
    image, _profile = read_model_tile(tile_path)
    _mask, _confidence, probabilities = run_model_tile(model, image, device)
    return probabilities.astype(np.float32)


def _default_infer_batch(
    model: Any,
    images: np.ndarray,
    device: str,
) -> np.ndarray:
    _masks, _confidence, probabilities = run_model_batch(model, images, device)
    return probabilities


def _clear_accelerator_cache(device: str) -> None:
    device_value = str(device)
    if not (device_value.startswith("cuda") or device_value.startswith("mps")):
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
        return (
            probabilities.shape == (14, 512, 512) and probabilities.dtype == np.float16
        )
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


def _load_profile(spec: Mapping[str, Any]) -> dict[str, Any] | None:
    fusion = spec.get("fusion")
    if not fusion:
        return None
    if isinstance(fusion.get("profile"), Mapping):
        return dict(fusion["profile"])
    path_value = (
        fusion.get("snapshot_path")
        or fusion.get("profile_path")
        or fusion.get("file_path")
    )
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
        raise WorkPackageRuntimeError(
            "Work Package runtime requires run_spec schema_version 2"
        )
    run_id = str(spec["run_id"])
    run_dir = Path(spec["run_dir"]).resolve()
    try:
        tile_cache_dir = validated_run_tile_cache_dir(spec)
    except RunSpecError as error:
        raise WorkPackageRuntimeError(str(error)) from error
    database = run_state_from_spec(spec)
    package = database.control_graph.get_work_package(run_id, package_id)
    if package is None:
        raise WorkPackageRuntimeError(f"unknown Work Package: {package_id}")
    leased_job = job_id is not None or lease_token is not None
    if leased_job:
        if job_id is None or not lease_token:
            raise WorkPackageRuntimeError(
                "Work Package job requires both job_id and lease_token"
            )
        if not database.jobs.work_package_job_holds_lease(
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
    elif not database.control_graph.set_work_package_status(
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
    monitor_job = database.jobs.get_job(int(job_id)) if leased_job else {}
    monitor_context = {
        "job_id": job_id,
        "parent_span_id": str((monitor_job or {}).get("monitor_span_id") or ""),
        "execution_id": str((monitor_job or {}).get("monitor_execution_id") or ""),
        "attempt": int((monitor_job or {}).get("attempt") or 0),
    }
    base_emit = globals()["emit"]
    monitor_activity = {}
    monitor_observed = 0.0

    def emit(event: str, **payload: Any) -> None:
        nonlocal monitor_observed
        monitor_activity["event"] = event
        monitor_activity["effective_device"] = str(
            payload.get("effective_device")
            or device
            or (spec.get("runtime") or {}).get("effective_device")
            or ""
        )
        if event == "package_model_loading":
            monitor_activity.update(
                stream_id=payload.get("stream_id", ""),
                tile_current=0,
                tile_total=0,
                configured_batch_size=payload.get("configured_batch_size"),
                effective_batch_size=payload.get("configured_batch_size"),
                status="模型加载 / 推理",
                notice="",
            )
        elif event == "package_tile_completed":
            monitor_activity.update(
                tile_current=payload.get("current"),
                tile_total=payload.get("total"),
                status="模型推理",
            )
        elif event == "package_tile_batch_reduced":
            monitor_activity.update(
                effective_batch_size=payload.get("effective_batch_size"),
                notice="内存不足，自动降低 Batch",
            )
        elif event == "work_package_paused_low_disk":
            monitor_activity.update(
                status="资源等待：磁盘空间不足", notice=str(payload.get("error") or "")
            )
        elif event == "package_tile_materialized":
            monitor_activity.update(status="准备影像块", notice="")
        elif event == "package_tiles_cleaned":
            monitor_activity.update(status="提交与清理", stream_id="")
        now = time.monotonic()
        if monitor_context["parent_span_id"] and (
            now - monitor_observed >= 1.0
            or event
            in {
                "package_model_loading",
                "package_tile_batch_reduced",
                "work_package_paused_low_disk",
            }
        ):
            database.jobs.update_job_monitor_runtime(
                int(job_id), monitor_context["parent_span_id"], monitor_activity
            )
            monitor_observed = now
        base_emit(event, **{**monitor_context, **payload})

    tiles = database.control_graph.package_tiles(run_id, package_id)
    partitions = database.control_graph.package_partitions(run_id, package_id)
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
    requested_device = (
        device or (spec.get("runtime") or {}).get("effective_device") or "auto"
    )
    effective_device = resolve_device(str(requested_device))
    if not validate_device(effective_device):
        raise WorkPackageRuntimeError(
            f"semantic device is unavailable: {effective_device}"
        )
    configured_batch_size = max(
        1, int((spec.get("runtime") or {}).get("tile_batch_size", 1))
    )
    configured_batch_sizes_by_model = {
        str(model_id): max(1, int(value))
        for model_id, value in (
            ((spec.get("resource_tuning") or {}).get("resolved") or {}).get(
                "tile_batch_size_by_model"
            )
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
        active_fusion_head = _load_linear_fusion_head(spec, profile, effective_device)
    fusion_id = str((spec.get("fusion") or {}).get("profile_id") or "")
    model_partitions = WorkPackageModelPartitions(
        run_id=run_id,
        run_dir=run_dir,
        package_root=package_root,
        fusion_id=fusion_id,
        partitions=partitions,
        transform=transform,
        crs=str(crs),
        range_geometry=range_geometry,
        profile=profile,
        artifacts=database.artifacts,
        lease_guard=lease_guard,
    )

    model_summaries = []
    cleaned_bytes = 0
    model_load_count = 0
    model_cache_hit_count = 0
    tile_cache_released_count = 0
    tile_cache_retained_count = len(active_tiles)
    partition_pipelines: list[PartitionBuildPipeline] = []
    storage_report = dict(spec.get("storage_preflight") or {})
    stream_ids = [
        str(item.get("stream_id") or "")
        for item in spec.get("streams") or []
        if str(item.get("stream_id") or "")
    ]

    def permanent_partitions() -> Iterable[Mapping[str, Any]]:
        yield from database.control_graph.partitions_for_run(run_id)

    def ready_permanent_artifact_keys() -> Iterable[tuple[str, str, str]]:
        for stream_id in stream_ids:
            for kind in ("core_mask", "core_confidence"):
                for item in database.artifacts.artifacts_for_stream(
                    run_id, stream_id, kind=kind, status="ready"
                ):
                    yield stream_id, str(item["unit_id"]), kind

    managed_roots = (
        package_root,
        tile_cache_dir,
        run_dir / "tmp" / "probability_parts",
        run_dir / "tmp" / "fragmentation_v33_inputs",
        run_dir / "tmp" / "unit_confidence",
    )
    fallback_min_free_disk_gb = 0.0
    if not storage_report.get("effective_min_free_disk_bytes"):
        fallback_min_free_disk_gb = float(
            (spec.get("scaling") or {}).get("min_free_disk_gb", 0.0)
        )

    package_lock = _PackageFileLock(
        run_dir / "tmp" / "package_locks" / f"{package_id}.lock"
    )
    try:
        package_lock.acquire(lease_guard)
        # Remove abandoned atomic score files before seeding either ledger.
        # They are not committed working cache and cannot be resumed.
        for entry in spec["models"]:
            if lease_guard is not None:
                lease_guard()
            cleaned_bytes += remove_owned_temporary_files(
                package_root / "score_batches" / str(entry["model_id"])
            )
        storage_guard = WorkPackageStorageBudget(
            run_dir,
            managed_roots=managed_roots,
            working_roots=(package_root, tile_cache_dir),
            storage_preflight=storage_report,
            fallback_min_free_disk_gb=fallback_min_free_disk_gb,
            stream_ids=stream_ids,
            partitions=permanent_partitions(),
            ready_permanent_keys=ready_permanent_artifact_keys(),
            remaining_deferred_bytes=lambda: remaining_deferred_temporary_reserve_bytes(
                spec, database
            ),
            lease_guard=lease_guard,
        )
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
            before_write=storage_guard.reserve_materialized_write,
            managed_delta=storage_guard.working_cache.adjust,
        )
        materialized_by_id = {item["tile_id"]: item for item in materialized}
        for tile in active_tiles:
            item = materialized_by_id[str(tile["tile_id"])]
            tile["raster_path"] = item["tile_path"]
            tile["sha256"] = item["sha256"]
            if lease_guard is not None:
                lease_guard()
            if not database.control_graph.update_tile_raster(
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
                    raise WorkPackageRuntimeError(f"model SHA256 mismatch: {model_id}")
            database.run_streams.set_stream_status(run_id, stream_id, "running")
            emit(
                "package_model_loading",
                run_id=run_id,
                package_id=package_id,
                stream_id=stream_id,
                current=model_index,
                total=len(spec["models"]),
                configured_batch_size=model_configured_batch_size,
                effective_device=effective_device,
            )
            inferable_tiles = [
                tile for tile in active_tiles if str(tile.get("status")) != "accepted"
            ]
            outputs_reusable = resume and model_partitions.model_outputs_reusable(
                stream_id, model_id
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
            model_cold_load_count = 0
            model_cache_hits = 0
            checkpoint_reused_count = 0
            score_batch_root = package_root / "score_batches" / model_id

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
                    raise WorkPackageRuntimeError(
                        f"Tile raster is missing: {tile_path}"
                    )
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
                score_path, metadata_path = _accepted_score_paths(package_root, tile_id)
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
                    reservation = storage_guard.working_cache.reserve(
                        f"accepted_score:{tile_id}",
                        write_bytes=estimated_write_bytes,
                        managed_growth_bytes=max(
                            0, estimated_write_bytes - previous_bytes
                        ),
                    )
                    try:
                        write_atomic_npz(
                            score_path,
                            probabilities=probabilities.astype(np.float16),
                        )
                        write_atomic_json(metadata_path, expected)
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
                        offset : offset
                        + _host_pipeline_batch_limit(model_configured_batch_size)
                    ],
                )
                for sequence, offset in enumerate(
                    range(
                        0,
                        len(inferable_items),
                        _host_pipeline_batch_limit(model_configured_batch_size),
                    )
                )
            ]
            host_batch_limit = _host_pipeline_batch_limit(model_configured_batch_size)
            if model_effective_batch_size > host_batch_limit:
                emit(
                    "package_tile_batch_reduced",
                    run_id=run_id,
                    package_id=package_id,
                    stream_id=stream_id,
                    attempted_batch_size=model_effective_batch_size,
                    effective_batch_size=host_batch_limit,
                    reason="bounded host pipeline byte budget; not a CUDA allocation failure",
                )
                model_effective_batch_size = host_batch_limit
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
                    storage_guard.working_cache.released(discarded)
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

            def commit_partition(
                partition: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
            ) -> None:
                model_partitions.commit_model_partition(
                    model_id, stream_id, partition, arrays, storage_guard
                )

            partition_pipeline = PartitionBuildPipeline(
                partition_requirements,
                overlap=overlap,
                commit=commit_partition,
            )
            partition_pipelines.append(partition_pipeline)
            partition_pipeline.schedule_ready(score_records_by_tile)

            managed_score_cache_bytes = directory_size(score_batch_root)
            probability_bytes_per_tile = int(
                storage_report.get("current_model_probability_bytes") or 0
            )
            score_cache_high_water_bytes = (
                len(inferable_items) * probability_bytes_per_tile
                + len(groups) * CHECKPOINT_WRITE_OVERHEAD_BYTES
                if storage_guard.storage_schema >= 2 and probability_bytes_per_tile > 0
                else 0
            )

            def write_batch(
                sequence: int,
                group: list[dict[str, Any]],
                probabilities: np.ndarray,
                current_score_cache_bytes: int,
            ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
                return write_checkpoint(
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
                    storage_guard=storage_guard.working_cache,
                    storage_operation=(f"score_checkpoint:{model_id}:{sequence}"),
                )

            def consume_completed_checkpoint(
                completed: CompletedCheckpoint,
            ) -> int:
                nonlocal managed_score_cache_bytes
                managed_score_cache_bytes = directory_size(score_batch_root)
                for item, record in zip(completed.group, completed.records):
                    record_score(item, record)
                partition_pipeline.schedule_ready(score_records_by_tile)
                return managed_score_cache_bytes

            remember_batch_size_callback: Callable[[int], None] | None = None
            if model_provider is not None:

                def remember_effective_batch_size(effective: int) -> None:
                    model_provider.remember_batch_size(
                        model_entry,
                        effective_device,
                        effective,
                    )

                remember_batch_size_callback = remember_effective_batch_size

            def observe_batch_reduction(
                attempted: int,
                effective: int,
                error: BaseException,
            ) -> None:
                emit(
                    "package_tile_batch_reduced",
                    run_id=run_id,
                    package_id=package_id,
                    stream_id=stream_id,
                    attempted_batch_size=attempted,
                    effective_batch_size=effective,
                    reason=str(error),
                )

            score_execution = execute_model_scores(
                missing_groups,
                read=read_model_tile,
                infer=build_model_batch_inference(
                    model,
                    effective_device,
                    infer_images=infer_images,
                    infer_batch=infer_batch,
                    infer_tile=(None if infer_tile is _default_infer else infer_tile),
                    production_batch_inference=production_batch_inference,
                ),
                write_checkpoint=write_batch,
                consume_completed=consume_completed_checkpoint,
                is_recoverable_capacity_error=lambda error: (
                    is_recoverable_batch_error(error, effective_device)
                ),
                remember_batch_size=remember_batch_size_callback,
                clear_accelerator_cache=lambda: _clear_accelerator_cache(
                    effective_device
                ),
                observe_batch_reduction=observe_batch_reduction,
                input_workers=io_workers,
                initial_batch_size=model_effective_batch_size,
                initial_managed_cache_bytes=managed_score_cache_bytes,
            )
            model_effective_batch_size = score_execution.effective_batch_size
            incomplete_partition = partition_pipeline.finish(score_records_by_tile)
            if incomplete_partition is not None:
                partition, missing_ids = incomplete_partition
                raise WorkPackageRuntimeError(
                    f"Partition {partition['partition_id']} lacks scores: "
                    f"{missing_ids[:5]}"
                )
            partition_pipeline.shutdown()
            partition_pipelines.remove(partition_pipeline)
            if not keep_score_cache_until_package_ready:
                removed = _remove_tree_with_count(score_batch_root)
                cleaned_bytes += removed
                storage_guard.working_cache.released(removed)
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
                    "peak_tile_batch_size": score_execution.peak_batch_size,
                    "inference_batch_count": score_execution.inference_batch_count,
                    "batch_reduction_count": score_execution.batch_reduction_count,
                    "checkpoint_written_count": (
                        score_execution.checkpoint_written_count
                    ),
                    "checkpoint_reused_count": checkpoint_reused_count,
                    "checkpoint_written_bytes": (
                        score_execution.checkpoint_written_bytes
                    ),
                    "input_queue_capacity": score_execution.input_queue_capacity,
                    "input_queue_peak_batches": (
                        score_execution.input_queue_peak_batches
                    ),
                    "result_queue_capacity": score_execution.result_queue_capacity,
                    "result_queue_peak_batches": (
                        score_execution.result_queue_peak_batches
                    ),
                    "partition_queue_capacity": 1,
                    "partition_queue_peak": partition_pipeline.queue_peak,
                    "input_wait_sec": round(score_execution.input_wait_sec, 6),
                    "inference_sec": round(score_execution.inference_sec, 6),
                    "checkpoint_write_sec": round(
                        score_execution.checkpoint_write_sec, 6
                    ),
                    "checkpoint_wait_sec": round(
                        score_execution.checkpoint_wait_sec, 6
                    ),
                    "partition_sec": round(partition_pipeline.elapsed_sec, 6),
                    "cold_load_count": model_cold_load_count,
                    "cache_hit_count": model_cache_hits,
                }
            )

            emit(
                "package_model_completed",
                run_id=run_id,
                package_id=package_id,
                stream_id=stream_id,
                status="completed",
                **model_summaries[-1],
            )

        # Release the final model's activation cache for Fusion and the next
        # Work Package without unloading PersistentModelProvider models.
        _clear_accelerator_cache(effective_device)

        if profile:
            stream_id = f"fusion:{fusion_id}"
            fragmentation = dict(spec.get("fragmentation_regularization") or {})
            v33_enabled = bool(
                fragmentation.get("enabled", True)
                and fragmentation.get("policy_id")
                == "fragmentation_v33_configurable_absorption_v1"
            )
            database.run_streams.set_stream_status(run_id, stream_id, "running")
            for partition in partitions:
                partition_id = partition["partition_id"]
                if lease_guard is not None:
                    lease_guard()
                probabilities, coverage = model_partitions.finalize_partition(
                    partition_id, active_fusion_head
                )
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
                    run_dir
                    / "tmp"
                    / "probability_parts"
                    / f"fusion_{fusion_id}"
                    / f"{partition_id}.tif"
                )
                v3_context_path = (
                    run_dir
                    / "tmp"
                    / "fragmentation_v33_inputs"
                    / f"fusion_{fusion_id}"
                    / f"{partition_id}_v3_context.tif"
                )
                v3_baseline_path = (
                    run_dir
                    / "tmp"
                    / "fragmentation_v33_inputs"
                    / f"fusion_{fusion_id}"
                    / f"{partition_id}_v3_baseline.tif"
                )
                raster_root = run_dir / "fusion" / fusion_id / "raster_parts"
                previous_probability_bytes = (
                    probability_path.stat().st_size if probability_path.is_file() else 0
                )
                previous_v3_context_bytes = (
                    v3_context_path.stat().st_size
                    if v33_enabled and v3_context_path.is_file()
                    else 0
                )
                previous_v3_baseline_bytes = (
                    v3_baseline_path.stat().st_size
                    if v33_enabled and v3_baseline_path.is_file()
                    else 0
                )
                probability_write_bytes = int(probabilities.size * 2)
                permanent_write_bytes = int(
                    np.asarray(arrays["core_mask"]).nbytes
                    + np.asarray(arrays["core_confidence"]).nbytes
                )
                candidate_context_bytes = int(
                    np.asarray(arrays.get("v3_context_core", ())).nbytes
                    if v33_enabled
                    else 0
                )
                raster_write_overhead_bytes = (5 if v33_enabled else 3) * 64 * 1024
                reservation = storage_guard.reserve_write(
                    f"partition_rasters:{stream_id}:{partition_id}",
                    probability_write_bytes
                    + permanent_write_bytes
                    + candidate_context_bytes
                    + raster_write_overhead_bytes,
                    managed_growth_bytes=max(
                        0,
                        probability_write_bytes
                        + candidate_context_bytes
                        + (np.asarray(arrays["core_mask"]).nbytes if v33_enabled else 0)
                        + 64 * 1024
                        - previous_probability_bytes
                        - previous_v3_context_bytes
                        - previous_v3_baseline_bytes,
                    ),
                )
                try:
                    paths = write_partition_rasters(
                        arrays,
                        partition,
                        global_transform=transform,
                        crs=crs,
                        output_probability=probability_path,
                        output_mask=(
                            v3_baseline_path
                            if v33_enabled
                            else raster_root / f"{partition_id}_mask.tif"
                        ),
                        output_confidence=raster_root
                        / f"{partition_id}_confidence.tif",
                        output_v3_context=(v3_context_path if v33_enabled else None),
                        core_mask_tags=core_mask_tags(regularization),
                    )
                finally:
                    current_probability_bytes = (
                        probability_path.stat().st_size
                        if probability_path.is_file()
                        else 0
                    )
                    current_v3_context_bytes = (
                        v3_context_path.stat().st_size
                        if v33_enabled and v3_context_path.is_file()
                        else 0
                    )
                    current_v3_baseline_bytes = (
                        v3_baseline_path.stat().st_size
                        if v33_enabled and v3_baseline_path.is_file()
                        else 0
                    )
                    reservation.settle(
                        current_probability_bytes
                        - previous_probability_bytes
                        + current_v3_context_bytes
                        - previous_v3_context_bytes
                        + current_v3_baseline_bytes
                        - previous_v3_baseline_bytes
                    )
                # With V3.3 selected the V3 Core remains a temporary, frozen
                # baseline. The second-stage job publishes the only production
                # core_mask before any Fusion Core/Seam/Junction job may run.
                raster_artifacts = (
                    (
                        ("v3_baseline_core", "mask"),
                        ("core_confidence", "confidence"),
                        ("v3_context_core", "v3_context"),
                        ("partition_probability", "probability"),
                    )
                    if v33_enabled
                    else (
                        ("core_mask", "mask"),
                        ("core_confidence", "confidence"),
                        ("partition_probability", "probability"),
                    )
                )
                for kind, key in raster_artifacts:
                    if lease_guard is not None:
                        lease_guard()
                    publish_artifact(
                        database.artifacts,
                        run_id,
                        path=Path(paths[key]),
                        kind=kind,
                        stream_id=stream_id,
                        unit_id=partition_id,
                    )
                    if kind in {"core_mask", "core_confidence"}:
                        storage_guard.mark_permanent_ready(
                            stream_id, partition_id, kind
                        )
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
                    owned_pixel_count=int(
                        (regularization.get("coverage_validation") or {}).get(
                            "owned_pixel_count", 0
                        )
                    ),
                    gap_pixel_count=int(
                        (regularization.get("coverage_validation") or {}).get(
                            "gap_pixel_count", 0
                        )
                    ),
                    outside_pixel_count=int(
                        (regularization.get("coverage_validation") or {}).get(
                            "outside_pixel_count", 0
                        )
                    ),
                )
            removed = _remove_tree_with_count(package_root / "fusion" / fusion_id)
            cleaned_bytes += removed
            storage_guard.working_cache.released(removed)
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
            storage_guard.working_cache.released(removed)
        removed = _remove_tree_with_count(package_root / "accepted_scores")
        cleaned_bytes += removed
        storage_guard.working_cache.released(removed)
        tile_cleaned_bytes = 0
        releasable_tile_ids = set(
            database.control_graph.releasable_package_tile_ids(run_id, package_id)
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
        storage_guard.working_cache.released(tile_cleaned_bytes)
        tile_cache_released_count = released_tile_count
        tile_cache_retained_count = len(active_tiles) - released_tile_count
        _prune_empty_tile_cache(tile_cache_dir)
        emit(
            "package_tiles_cleaned",
            run_id=run_id,
            package_id=package_id,
            tile_count=released_tile_count,
            dependency_retained_count=(len(active_tiles) - released_tile_count),
            cleaned_bytes=tile_cleaned_bytes,
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
            "storage_metrics_schema_version": 1,
            "storage_metrics_measurement": "package_guard_reserved_growth_v1",
            "peak_cache_bytes": storage_guard.working_cache.peak_managed_bytes,
            "peak_package_managed_bytes": storage_guard.peak_managed_bytes,
            "peak_rss_bytes": peak_rss_bytes(),
            "cleaned_bytes": cleaned_bytes,
            "tile_cache_released_count": tile_cache_released_count,
            "tile_cache_retained_count": tile_cache_retained_count,
            "elapsed_sec": round(time.monotonic() - started_at, 3),
            "status": "ready",
        }
        if lease_guard is not None:
            lease_guard()
        write_atomic_json(package_root / "package_report.json", result)
        if leased_job:
            if lease_guard is not None:
                lease_guard()
            if not database.jobs.complete_work_package_job(
                run_id,
                package_id,
                job_id,
                lease_token,
            ):
                raise WorkPackageRuntimeError(
                    "Work Package job lease expired before atomic commit"
                )
        elif not database.control_graph.set_work_package_status(
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
        for pipeline in partition_pipelines:
            pipeline.shutdown()
        partition_pipelines.clear()
        if not runtime_errors.storage_error_is_transient(error):
            transition = None
            if leased_job:
                transition = (
                    "failed"
                    if database.jobs.fail_work_package_job(
                        run_id,
                        package_id,
                        job_id,
                        lease_token,
                        error=str(error),
                    )
                    else None
                )
            else:
                database.control_graph.set_work_package_status(
                    run_id, package_id, "failed", expected="running"
                )
                transition = "failed"
            database.run_streams.append_event(
                run_id,
                "work_package_storage_contract_failed",
                level="error",
                message=str(error),
                payload={"package_id": package_id, "transition": transition},
            )
            raise
        if not preserve_lease_on_low_disk:
            if leased_job:
                database.jobs.interrupt_work_package_job(
                    run_id,
                    package_id,
                    job_id,
                    lease_token,
                    error=str(error),
                )
            else:
                database.control_graph.set_work_package_status(
                    run_id, package_id, "interrupted", expected="running"
                )
        database.run_streams.append_event(
            run_id,
            "work_package_paused_low_disk",
            level="warning",
            message=str(error),
            payload={"package_id": package_id},
        )
        emit(
            "work_package_paused_low_disk",
            run_id=run_id,
            package_id=package_id,
            error=str(error),
        )
        raise
    except runtime_errors.WorkerStopRequested as error:
        for pipeline in partition_pipelines:
            pipeline.shutdown()
        partition_pipelines.clear()
        if leased_job:
            database.jobs.interrupt_work_package_job(
                run_id,
                package_id,
                job_id,
                lease_token,
                error=str(error),
            )
        else:
            database.control_graph.set_work_package_status(
                run_id, package_id, "interrupted", expected="running"
            )
        database.run_streams.append_event(
            run_id,
            "work_package_interrupted",
            level="warning",
            message=str(error),
            payload={"package_id": package_id},
        )
        emit(
            "work_package_interrupted",
            run_id=run_id,
            package_id=package_id,
            error=str(error),
        )
        raise
    except runtime_errors.LeaseLostError as error:
        for pipeline in partition_pipelines:
            pipeline.shutdown()
        partition_pipelines.clear()
        # The current process is fenced out.  It must not change either the
        # Package or Job now owned by a newer lease.
        database.run_streams.append_event(
            run_id,
            "work_package_lease_lost",
            level="warning",
            message=str(error),
            payload={"package_id": package_id, "job_id": job_id},
        )
        raise
    except Exception as error:
        for pipeline in partition_pipelines:
            pipeline.shutdown()
        partition_pipelines.clear()
        transition = None
        if leased_job:
            transition = database.jobs.fail_or_requeue_work_package_job(
                run_id,
                package_id,
                job_id,
                lease_token,
                error=str(error),
            )
        else:
            database.control_graph.set_work_package_status(
                run_id, package_id, "failed", expected="running"
            )
            transition = "failed"
        database.run_streams.append_event(
            run_id,
            "work_package_failed",
            level="error",
            message=str(error),
            payload={"package_id": package_id, "transition": transition},
        )
        emit(
            "work_package_failed",
            run_id=run_id,
            package_id=package_id,
            error=str(error),
            status="failed",
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
            database = run_state_from_spec(spec)
            run_id = str(spec["run_id"])
            if transition == "interrupted":
                database.jobs.interrupt_work_package_job(
                    run_id,
                    package_id,
                    int(job_id),
                    str(lease_token),
                    error=str(error),
                )
            elif transition == "failed":
                database.jobs.fail_work_package_job(
                    run_id,
                    package_id,
                    int(job_id),
                    str(lease_token),
                    error=str(error),
                )
            else:
                database.jobs.fail_or_requeue_work_package_job(
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
            and runtime_errors.storage_error_is_transient(error)
        ):
            repair_leased_state(
                error,
                (
                    "interrupted"
                    if runtime_errors.storage_error_is_transient(error)
                    else "failed"
                ),
            )
        raise
    except runtime_errors.WorkerStopRequested as error:
        repair_leased_state(error, "interrupted")
        raise
    except runtime_errors.LeaseLostError:
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
    database = run_state_from_spec(spec)
    stopper = stop_event or threading.Event()
    provider = model_provider or PersistentModelProvider(
        batch_state_path=run_dir / "logs" / "accelerator_batch_limits.json"
    )
    profile = _load_profile(spec)
    requested_device = (
        device or (spec.get("runtime") or {}).get("effective_device") or "auto"
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
        resident_fusion_head = _load_linear_fusion_head(spec, profile, effective_device)
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
        if database.jobs.job_counts(run_id, job_type="work_package").get("failed", 0):
            # One exhausted Package makes the complete local Run impossible.
            # Do not spend accelerator time on later Packages after that hard
            # gate has already failed.
            break
        job = database.jobs.lease_next_work_package(
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
            database_schema=spec.get("state_schema"),
            run_id=run_id,
            package_id=package_id,
            job_id=int(job["job_id"]),
            lease_token=str(job["lease_token"]),
            stop_event=stopper,
            interval_sec=heartbeat_interval_sec,
            lease_seconds=lease_seconds,
        )

        def execute_package(
            lease_check: Callable[[], None],
            progress_update: Callable[[int, int], None],
        ) -> object:
            return run_work_package(
                spec_path,
                package_id,
                job_id=int(job["job_id"]),
                lease_token=str(job["lease_token"]),
                device=device,
                resume=resume,
                model_provider=provider,
                lease_guard=lease_check,
                lease_progress=progress_update,
                model_load_observer=record_model_load,
                preserve_lease_on_low_disk=True,
                infer_tile=infer_tile,
                infer_batch=infer_batch,
                infer_images=infer_images,
                fusion_head=resident_fusion_head,
            )

        def observe_low_disk(pause_count: int, error: BaseException) -> None:
            emit(
                "accelerator_worker_paused_low_disk",
                run_id=run_id,
                worker_id=str(worker_id),
                package_id=package_id,
                pause_count=pause_count,
                error=str(error),
            )

        def repair_session_state(error: BaseException) -> None:
            try:
                if database.jobs.work_package_job_holds_lease(
                    run_id,
                    package_id,
                    int(job["job_id"]),
                    str(job["lease_token"]),
                ):
                    database.jobs.fail_or_requeue_work_package_job(
                        run_id,
                        package_id,
                        int(job["job_id"]),
                        str(job["lease_token"]),
                        error=str(error),
                    )
            except Exception:
                # Session repair is best-effort and must not replace the
                # failure that triggered it.
                return

        session_result = execute_leased_package_session(
            heartbeat=heartbeat,
            execute_package=execute_package,
            stop_event=stopper,
            low_disk_poll_sec=low_disk_poll_sec,
            low_disk_pause_count=low_disk_pause_count,
            pause_observer=observe_low_disk,
            repair_leased_state=repair_session_state,
        )
        low_disk_pause_count = session_result.low_disk_pause_count
        heartbeat_count += session_result.heartbeat_count
        if session_result.status == "ready":
            ready_count += 1
        elif session_result.status == "fenced":
            failure_count += 1
            session_fenced = True
        elif session_result.status == "failed":
            failure_count += 1
        if stopper.is_set():
            database.jobs.interrupt_work_package_worker(run_id, str(worker_id))
            break
        if session_fenced:
            # Losing one Package lease fences this persistent accelerator
            # session.  It must not lease another Package while a replacement
            # session may already be running on the same GPU.
            break

    counts = database.jobs.job_counts(run_id, job_type="work_package")
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
    write_atomic_json(
        run_dir / "logs" / "accelerator_workers" / f"{worker_session_id}.json",
        report,
    )
    write_atomic_json(run_dir / "logs" / "accelerator_worker_report.json", report)
    emit("accelerator_worker_finished", **report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run bounded semantic Work Packages")
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
            parser.error("single-package mode requires --job-id and --lease-token")
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
