"""Probe deployed TorchScript models without constructing environment reports."""

from __future__ import annotations

import gc
import json
import subprocess
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from loess_runtime.inference.torchscript_runtime import load_torchscript_model
from loess_runtime.system.probe_process import (
    environment_worker_command,
    run_environment_worker,
)

BATCH_PROBE_GROWTH_MARGIN_NUMERATOR = 3
BATCH_PROBE_GROWTH_MARGIN_DENOMINATOR = 2

ProbePayload = dict[str, Any]
ProbeProgress = Callable[[ProbePayload], None]


def verify_torchscript_contract(
    torch_module: Any,
    path: str | Path,
    device: str,
) -> tuple[bool, str]:
    """Load one deployment artifact and execute the fixed input contract."""
    model = None
    sample = None
    output = None
    try:
        model, runtime_info = load_torchscript_model(path, device)
        sample = torch_module.zeros(
            1, 3, 512, 512, dtype=torch_module.float32, device=device
        )
        with torch_module.inference_mode():
            output = model(sample)
        if not torch_module.is_tensor(output):
            return (
                False,
                f"TorchScript output must be one tensor, got {type(output).__name__}",
            )
        if tuple(output.shape) != (1, 14, 512, 512):
            return (
                False,
                f"TorchScript output shape is {tuple(output.shape)}, expected (1,14,512,512)",
            )
        if output.dtype != torch_module.float32:
            return (
                False,
                f"TorchScript output dtype is {output.dtype}, expected float32",
            )
        runtime_message = (
            f"TorchScript contract passed on {device}; runtime={runtime_info['mode']}"
        )
        if str(device).startswith("mps"):
            runtime_message += (
                f"; contiguous_bridges={runtime_info.get('mps_contiguous_bridge_count', 0)}"
                f"; pool_cpu_bridges={runtime_info.get('mps_cpu_bridge_count', 0)}"
            )
        return True, runtime_message
    except Exception as exc:
        return False, f"TorchScript contract failed on {device}: {exc}"
    finally:
        output = None
        sample = None
        model = None
        gc.collect()
        if str(device).startswith("cuda") and torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
        if str(device).startswith("mps") and torch_module.backends.mps.is_available():
            torch_module.mps.empty_cache()


def verify_torchscript_contract_isolated(
    path: str | Path,
    device: str,
    timeout: int = 600,
) -> tuple[bool, str]:
    """Run one device contract in a fresh process to isolate accelerator state."""

    command = environment_worker_command(
        "--contract-worker",
        "--model-path",
        str(Path(path).resolve()),
        "--device",
        str(device),
    )
    try:
        result = run_environment_worker(command, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"TorchScript contract timed out after {timeout}s on {device}"

    payload = None
    for line in reversed((result.stdout or "").splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "ok" in candidate:
            payload = candidate
            break
    if payload is not None:
        return bool(payload.get("ok")), str(payload.get("message") or "")

    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    detail = stderr or stdout or "no worker output"
    return False, (
        f"TorchScript contract worker failed on {device} "
        f"(exit={result.returncode}): {detail}"
    )


def _clear_accelerator_cache(torch_module: Any, device: str) -> None:
    gc.collect()
    if str(device).startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
    elif (
        str(device).startswith("mps")
        and hasattr(torch_module, "backends")
        and torch_module.backends.mps.is_available()
    ):
        torch_module.mps.empty_cache()


def _accelerator_free_bytes(torch_module: Any, device: str) -> int | None:
    try:
        if str(device).startswith("cuda") and torch_module.cuda.is_available():
            index = int(str(device).split(":", 1)[1]) if ":" in str(device) else 0
            return int(torch_module.cuda.mem_get_info(index)[0])
        if str(device).startswith("mps"):
            recommended = getattr(torch_module.mps, "recommended_max_memory", None)
            allocated = getattr(torch_module.mps, "driver_allocated_memory", None)
            if callable(recommended) and callable(allocated):
                return max(0, int(recommended()) - int(allocated()))
    except Exception:
        return None
    return None


def _synchronize_accelerator(torch_module: Any, device: str) -> None:
    if str(device).startswith("cuda"):
        synchronize = getattr(torch_module.cuda, "synchronize", None)
        if callable(synchronize):
            index = int(str(device).split(":", 1)[1]) if ":" in str(device) else 0
            synchronize(index)
    elif str(device).startswith("mps"):
        synchronize = getattr(torch_module.mps, "synchronize", None)
        if callable(synchronize):
            synchronize()


def _batch_probe_error_kind(torch_module: Any, error: BaseException) -> str:
    cuda_oom = getattr(getattr(torch_module, "cuda", None), "OutOfMemoryError", ())
    if isinstance(cuda_oom, type) and isinstance(error, cuda_oom):
        return "out_of_memory"
    text = f"{type(error).__name__}: {error}".lower()
    if any(
        marker in text
        for marker in (
            "out of memory",
            "not enough memory",
            "mps backend out of memory",
            "cuda error: out of memory",
        )
    ):
        return "out_of_memory"
    return "runtime_error"


def _probe_loaded_torchscript_batches(
    torch_module: Any,
    model: Any,
    runtime_info: Mapping[str, Any],
    device: str,
    candidates: Iterable[int],
    *,
    reserve_bytes: int = 0,
    progress: ProbeProgress | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Probe one already-resident model without releasing its model object."""

    normalized_candidates = sorted(
        {int(value) for value in candidates if int(value) >= 1}
    )
    if not normalized_candidates or normalized_candidates[0] != 1:
        raise ValueError("Batch probe candidates must start at 1")
    sample = None
    output = None
    records: list[ProbePayload] = []
    successful: list[int] = []
    safe_candidates: list[int] = []
    stop_reason = "ceiling_reached"
    first_failed_batch: int | None = None
    previous_observation: tuple[int, int, int] | None = None

    def publish(payload: Mapping[str, Any]) -> None:
        if progress is not None:
            value = dict(payload)
            if model_id is not None:
                value["model_id"] = str(model_id)
            progress(value)

    try:
        for batch_size in normalized_candidates:
            if previous_observation is not None and int(reserve_bytes) > 0:
                previous_batch, previous_free, bytes_per_item = previous_observation
                if bytes_per_item > 0 and batch_size > previous_batch:
                    projected_growth = (
                        bytes_per_item
                        * (batch_size - previous_batch)
                        * BATCH_PROBE_GROWTH_MARGIN_NUMERATOR
                        + BATCH_PROBE_GROWTH_MARGIN_DENOMINATOR
                        - 1
                    ) // BATCH_PROBE_GROWTH_MARGIN_DENOMINATOR
                    projected_free = previous_free - projected_growth
                    if projected_free < int(reserve_bytes):
                        stop_reason = "safety_projection"
                        record = {
                            "batch_size": batch_size,
                            "status": "skipped_safety_projection",
                            "accelerator_free_bytes": previous_free,
                            "projected_free_bytes": projected_free,
                            "reserve_bytes": int(reserve_bytes),
                            "projection_from_batch_size": previous_batch,
                        }
                        records.append(record)
                        publish({"event": "batch_probe_result", **record})
                        break
            publish({"event": "batch_probe_started", "batch_size": batch_size})
            try:
                sample = torch_module.zeros(
                    batch_size,
                    3,
                    512,
                    512,
                    dtype=torch_module.float32,
                    device=device,
                )
                with torch_module.inference_mode():
                    output = model(sample)
                _synchronize_accelerator(torch_module, device)
                if not torch_module.is_tensor(output):
                    raise TypeError(
                        "TorchScript output must be one tensor, got "
                        f"{type(output).__name__}"
                    )
                expected_shape = (batch_size, 14, 512, 512)
                if tuple(output.shape) != expected_shape:
                    raise ValueError(
                        f"TorchScript output shape is {tuple(output.shape)}, "
                        f"expected {expected_shape}"
                    )
                if output.dtype != torch_module.float32:
                    raise TypeError(
                        f"TorchScript output dtype is {output.dtype}, expected float32"
                    )
                free_bytes = _accelerator_free_bytes(torch_module, device)
                successful.append(batch_size)
                enough_headroom = (
                    free_bytes is None
                    or batch_size == 1
                    or free_bytes >= int(reserve_bytes)
                )
                if enough_headroom:
                    safe_candidates.append(batch_size)
                    status = "passed"
                else:
                    status = "insufficient_headroom"
                    stop_reason = "safety_reserve"
                record = {
                    "batch_size": batch_size,
                    "status": status,
                    "accelerator_free_bytes": free_bytes,
                }
                records.append(record)
                publish({"event": "batch_probe_result", **record})
                if not enough_headroom:
                    break
                if free_bytes is not None:
                    if previous_observation is None:
                        bytes_per_item = 0
                    else:
                        old_batch, old_free, _old_bytes_per_item = previous_observation
                        batch_delta = batch_size - old_batch
                        observed_growth = max(0, old_free - free_bytes)
                        bytes_per_item = (
                            (observed_growth + batch_delta - 1) // batch_delta
                            if batch_delta > 0
                            else 0
                        )
                    previous_observation = (
                        batch_size,
                        free_bytes,
                        bytes_per_item,
                    )
            except Exception as error:
                first_failed_batch = batch_size
                stop_reason = _batch_probe_error_kind(torch_module, error)
                record = {
                    "batch_size": batch_size,
                    "status": "failed",
                    "error_kind": stop_reason,
                    "error": str(error),
                }
                records.append(record)
                publish({"event": "batch_probe_result", **record})
                break
            finally:
                output = None
                sample = None
                _clear_accelerator_cache(torch_module, device)
    finally:
        output = None
        sample = None
        _clear_accelerator_cache(torch_module, device)

    last_verified_batch_size = max(
        safe_candidates or ([1] if successful else []), default=0
    )
    fatal_runtime_error = stop_reason == "runtime_error"
    safe_batch_size = 0 if fatal_runtime_error else last_verified_batch_size
    ok = safe_batch_size >= 1 and not fatal_runtime_error
    return {
        "ok": ok,
        "safe_batch_size": safe_batch_size,
        "last_verified_batch_size": last_verified_batch_size,
        "max_successful_batch": max(successful, default=0),
        "first_failed_batch": first_failed_batch,
        "stop_reason": stop_reason,
        "reserve_bytes": int(reserve_bytes),
        "runtime_info": runtime_info,
        "probes": records,
        "message": (
            f"TorchScript Batch probe selected {safe_batch_size} on {device}; "
            f"stop={stop_reason}; reserve={int(reserve_bytes)} bytes"
            if ok
            else (
                "TorchScript Batch probe encountered a non-capacity runtime "
                f"error at Batch {first_failed_batch} on {device}; no Batch "
                "value may be frozen"
                if fatal_runtime_error
                else f"TorchScript Batch probe failed at Batch 1 on {device}"
            )
        ),
    }


def probe_torchscript_batches(
    torch_module: Any,
    path: str | Path,
    device: str,
    candidates: Iterable[int],
    *,
    reserve_bytes: int = 0,
    progress: ProbeProgress | None = None,
) -> dict[str, Any]:
    """Load one model once and probe ascending dummy-forward Batch sizes."""

    model = None
    runtime_info: dict[str, Any] = {}
    try:
        model, runtime_info = load_torchscript_model(path, device)
        return _probe_loaded_torchscript_batches(
            torch_module,
            model,
            runtime_info,
            device,
            candidates,
            reserve_bytes=reserve_bytes,
            progress=progress,
        )
    except Exception as error:
        stop_reason = _batch_probe_error_kind(torch_module, error)
        return {
            "ok": False,
            "safe_batch_size": 0,
            "last_verified_batch_size": 0,
            "max_successful_batch": 0,
            "first_failed_batch": 1,
            "stop_reason": stop_reason,
            "reserve_bytes": int(reserve_bytes),
            "runtime_info": runtime_info,
            "probes": [],
            "message": f"TorchScript Batch probe failed during model load: {error}",
        }
    finally:
        model = None
        _clear_accelerator_cache(torch_module, device)


def probe_torchscript_model_set_batches(
    torch_module: Any,
    model_entries: Iterable[Mapping[str, Any]],
    device: str,
    candidates: Iterable[int],
    *,
    reserve_bytes: int = 0,
    progress: ProbeProgress | None = None,
) -> dict[str, Any]:
    """Load the complete model set, retain it, then probe each model in turn."""

    entries = [
        {
            "model_id": str(entry["model_id"]),
            "path": str(Path(entry["path"]).resolve()),
        }
        for entry in model_entries
    ]
    model_ids = [entry["model_id"] for entry in entries]
    if not entries or len(model_ids) != len(set(model_ids)):
        raise ValueError("Batch probe model set must contain unique model IDs")

    loaded_models: dict[str, Any] = {}
    runtime_info_by_model: dict[str, Mapping[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}

    def publish(payload: Mapping[str, Any]) -> None:
        if progress is not None:
            progress(dict(payload))

    try:
        for entry in entries:
            model_id = entry["model_id"]
            publish({"event": "model_set_load_started", "model_id": model_id})
            try:
                model, runtime_info = load_torchscript_model(entry["path"], device)
            except Exception as error:
                resident_ids = list(loaded_models)
                message = (
                    f"complete model-set load failed at {model_id}: {error}; "
                    "no Batch value may be frozen"
                )
                for item in entries:
                    item_id = item["model_id"]
                    results[item_id] = {
                        "ok": False,
                        "safe_batch_size": 0,
                        "last_verified_batch_size": 0,
                        "max_successful_batch": 0,
                        "first_failed_batch": 1,
                        "stop_reason": "model_set_load_failed",
                        "reserve_bytes": int(reserve_bytes),
                        "runtime_info": runtime_info_by_model.get(item_id, {}),
                        "probes": [],
                        "expected_model_ids": list(model_ids),
                        "resident_model_ids": resident_ids,
                        "resident_model_count": len(resident_ids),
                        "model_set_complete": False,
                        "message": message,
                    }
                return {
                    "ok": False,
                    "expected_model_ids": list(model_ids),
                    "resident_model_ids": resident_ids,
                    "resident_model_count": len(resident_ids),
                    "model_set_complete": False,
                    "results": results,
                    "message": message,
                }
            loaded_models[model_id] = model
            runtime_info_by_model[model_id] = runtime_info
            publish({"event": "model_set_load_completed", "model_id": model_id})

        resident_model_ids = list(model_ids)
        for entry in entries:
            model_id = entry["model_id"]
            result = _probe_loaded_torchscript_batches(
                torch_module,
                loaded_models[model_id],
                runtime_info_by_model[model_id],
                device,
                candidates,
                reserve_bytes=reserve_bytes,
                progress=publish,
                model_id=model_id,
            )
            result.update(
                {
                    "resident_model_ids": resident_model_ids,
                    "resident_model_count": len(resident_model_ids),
                    "model_set_complete": True,
                }
            )
            results[model_id] = result
            publish(
                {
                    "event": "model_set_probe_completed",
                    "model_id": model_id,
                    "result": result,
                }
            )

        ok = all(bool(results[model_id].get("ok")) for model_id in model_ids)
        return {
            "ok": ok,
            "expected_model_ids": resident_model_ids,
            "resident_model_ids": resident_model_ids,
            "resident_model_count": len(resident_model_ids),
            "model_set_complete": True,
            "results": results,
            "message": (
                f"probed {len(model_ids)} models while the complete set remained resident"
            ),
        }
    finally:
        loaded_models.clear()
        runtime_info_by_model.clear()
        _clear_accelerator_cache(torch_module, device)


def verify_torchscript_batch_probe_isolated(
    path: str | Path,
    device: str,
    candidates: Iterable[int],
    *,
    reserve_bytes: int = 0,
    timeout: int = 1200,
) -> dict[str, Any]:
    """Probe one model in one child process and retain progress after a crash."""

    normalized_candidates = [int(value) for value in candidates]
    command = environment_worker_command(
        "--batch-probe-worker",
        "--model-path",
        str(Path(path).resolve()),
        "--device",
        str(device),
        "--batch-candidates",
        ",".join(str(value) for value in normalized_candidates),
        "--reserve-bytes",
        str(max(0, int(reserve_bytes))),
    )
    result: Any
    try:
        result = run_environment_worker(command, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        stdout = (
            error.stdout.decode()
            if isinstance(error.stdout, bytes)
            else (error.stdout or "")
        )
        stderr = (
            error.stderr.decode()
            if isinstance(error.stderr, bytes)
            else (error.stderr or "")
        )
        result = type(
            "TimedOutProbe",
            (),
            {"returncode": -1, "stdout": stdout, "stderr": stderr or "timeout"},
        )()

    final_payload: ProbePayload | None = None
    progress_records: list[ProbePayload] = []
    started_batches: list[int] = []
    for line in (result.stdout or "").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") == "batch_probe_started":
            started_batches.append(int(payload["batch_size"]))
        elif payload.get("event") == "batch_probe_result":
            progress_records.append(
                {key: value for key, value in payload.items() if key != "event"}
            )
        elif payload.get("batch_probe") is True:
            final_payload = {
                key: value for key, value in payload.items() if key != "batch_probe"
            }
    if final_payload is not None:
        return final_payload

    safe_candidates = [
        int(record["batch_size"])
        for record in progress_records
        if record.get("status") == "passed"
    ]
    successful_candidates = [
        int(record["batch_size"])
        for record in progress_records
        if record.get("status") in {"passed", "insufficient_headroom"}
    ]
    safe_batch_size = max(
        safe_candidates or ([1] if successful_candidates else []), default=0
    )
    failed_batch: int | None = next(
        (
            int(record["batch_size"])
            for record in progress_records
            if record.get("status") == "failed"
        ),
        None,
    )
    if failed_batch is None and started_batches:
        completed_batches = {int(record["batch_size"]) for record in progress_records}
        failed_batch = next(
            (
                value
                for value in reversed(started_batches)
                if value not in completed_batches
            ),
            None,
        )
    detail = (result.stderr or "").strip() or "worker exited without a final result"
    return {
        "ok": safe_batch_size >= 1,
        "safe_batch_size": safe_batch_size,
        "max_successful_batch": max(successful_candidates, default=0),
        "first_failed_batch": failed_batch,
        "stop_reason": "worker_crash",
        "reserve_bytes": int(reserve_bytes),
        "runtime_info": {},
        "probes": progress_records,
        "message": (
            f"Batch probe worker stopped at {failed_batch}; selected "
            f"previous verified Batch {safe_batch_size}; {detail}"
            if safe_batch_size >= 1
            else f"Batch probe worker failed before Batch 1 completed; {detail}"
        ),
    }


def verify_torchscript_model_set_batch_probe_isolated(
    model_entries: Iterable[Mapping[str, Any]],
    device: str,
    candidates: Iterable[int],
    *,
    reserve_bytes: int = 0,
    timeout: int = 1200,
) -> dict[str, Any]:
    """Probe one complete model set in one child and require a final contract."""

    entries = [
        {
            "model_id": str(entry["model_id"]),
            "path": str(Path(entry["path"]).resolve()),
        }
        for entry in model_entries
    ]
    expected_ids = [entry["model_id"] for entry in entries]
    command = environment_worker_command(
        "--batch-probe-set-worker",
        "--model-set-json",
        json.dumps(entries, ensure_ascii=False, separators=(",", ":")),
        "--device",
        str(device),
        "--batch-candidates",
        ",".join(str(int(value)) for value in candidates),
        "--reserve-bytes",
        str(max(0, int(reserve_bytes))),
    )
    process: Any
    try:
        process = run_environment_worker(command, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        stdout = (
            error.stdout.decode()
            if isinstance(error.stdout, bytes)
            else (error.stdout or "")
        )
        stderr = (
            error.stderr.decode()
            if isinstance(error.stderr, bytes)
            else (error.stderr or "")
        )
        process = type(
            "TimedOutModelSetProbe",
            (),
            {"returncode": -1, "stdout": stdout, "stderr": stderr or "timeout"},
        )()

    final_payload: ProbePayload | None = None
    loaded_ids: list[str] = []
    completed_results: dict[str, dict[str, Any]] = {}
    for line in (process.stdout or "").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") == "model_set_load_completed":
            loaded_ids.append(str(payload.get("model_id") or ""))
        elif payload.get("event") == "model_set_probe_completed":
            model_id = str(payload.get("model_id") or "")
            result = payload.get("result")
            if model_id in expected_ids and isinstance(result, dict):
                completed_results[model_id] = result
        elif payload.get("batch_probe_set") is True:
            final_payload = {
                key: value for key, value in payload.items() if key != "batch_probe_set"
            }

    if final_payload is not None:
        result_ids = list((final_payload.get("results") or {}).keys())
        resident_ids = list(final_payload.get("resident_model_ids") or [])
        if (
            bool(final_payload.get("model_set_complete"))
            and result_ids == expected_ids
            and resident_ids == expected_ids
        ):
            return final_payload

    if loaded_ids == expected_ids and list(completed_results) == expected_ids:
        recovered = {
            "ok": all(
                bool(completed_results[model_id].get("ok")) for model_id in expected_ids
            ),
            "expected_model_ids": list(expected_ids),
            "resident_model_ids": list(loaded_ids),
            "resident_model_count": len(loaded_ids),
            "model_set_complete": True,
            "results": completed_results,
            "worker_exit_code": int(process.returncode),
            "worker_cleanup_warning": (
                "model probes completed before the isolated worker exited "
                f"without its final envelope (exit={process.returncode})"
            ),
            "message": (
                f"probed {len(expected_ids)} models while the complete set "
                "remained resident; recovered completed probe records after "
                f"worker exit {process.returncode}"
            ),
        }
        return recovered

    detail = (process.stderr or "").strip() or "worker exited without a complete result"
    message = (
        "complete model-set Batch probe did not finish; no partial Batch value "
        f"may be frozen (loaded={loaded_ids}, exit={process.returncode}): {detail}"
    )
    failed_results: dict[str, dict[str, Any]] = {
        model_id: {
            "ok": False,
            "safe_batch_size": 0,
            "last_verified_batch_size": 0,
            "max_successful_batch": 0,
            "first_failed_batch": None,
            "stop_reason": "worker_crash",
            "reserve_bytes": int(reserve_bytes),
            "runtime_info": {},
            "probes": [],
            "expected_model_ids": list(expected_ids),
            "resident_model_ids": list(loaded_ids),
            "resident_model_count": len(loaded_ids),
            "model_set_complete": False,
            "message": message,
        }
        for model_id in expected_ids
    }
    return {
        "ok": False,
        "expected_model_ids": list(expected_ids),
        "resident_model_ids": list(loaded_ids),
        "resident_model_count": len(loaded_ids),
        "model_set_complete": False,
        "results": failed_results,
        "message": message,
    }
