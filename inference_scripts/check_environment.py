"""Validate the Schema v2 semantic deployment environment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT / "runtime", PROJECT_ROOT / "qgis_plugins"):
    if import_root.is_dir() and str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from labeling_tool.core.run_state_db import (
    RunStateDB,
    production_state_database,
)

warnings.filterwarnings(
    "ignore",
    message=r"Failed to load image Python extension.*",
    category=UserWarning,
)

from _device import resolve_device, validate_device
from deployment_config import (
    CLASS_ORDER,
    load_yaml,
    validate_deployment_config,
)
from hardware_tuning import (
    batch_probe_safety_reserve_bytes,
    collect_hardware_snapshot,
    freeze_model_batch_probe_results,
    model_batch_probe_candidates,
    resolve_hardware_tuning,
)
from torchscript_runtime import load_torchscript_model


FINGERPRINT_FILES = (
    "../project_manifest.json",
    "../.loess-project-id",
    "../runtime/loess_launcher.sh",
)


def add_check(checks, check_id, status, value, source, message="", fix=""):
    checks.append({
        "id": check_id,
        "status": status,
        "value": str(value),
        "source": source,
        "message": message,
        "fix": fix,
    })


def import_dependency(name):
    try:
        module = importlib.import_module(name)
        return module, getattr(module, "__version__", "installed"), ""
    except Exception as exc:
        return None, "not installed", str(exc)


def _version_tuple(value):
    numbers = re.findall(r"\d+", str(value))
    return tuple(int(item) for item in numbers[:3])


def _mps_runtime_requirement(model_id, torch_version):
    if model_id == "upernet_swin_b" and _version_tuple(torch_version) < (2, 7):
        return (
            False,
            f"Swin MPS requires PyTorch >=2.7 in this deployment; current={torch_version}",
        )
    return True, ""


def verify_torchscript_contract(torch_module, path, device):
    """Load one deployment artifact and execute the fixed input contract."""
    model = None
    sample = None
    output = None
    try:
        model, runtime_info = load_torchscript_model(path, device)
        sample = torch_module.zeros(1, 3, 512, 512, dtype=torch_module.float32, device=device)
        with torch_module.inference_mode():
            output = model(sample)
        if not torch_module.is_tensor(output):
            return False, f"TorchScript output must be one tensor, got {type(output).__name__}"
        if tuple(output.shape) != (1, 14, 512, 512):
            return False, f"TorchScript output shape is {tuple(output.shape)}, expected (1,14,512,512)"
        if output.dtype != torch_module.float32:
            return False, f"TorchScript output dtype is {output.dtype}, expected float32"
        runtime_message = f"TorchScript contract passed on {device}; runtime={runtime_info['mode']}"
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


def verify_torchscript_contract_isolated(path, device, timeout=600):
    """Run one device contract in a fresh process to isolate accelerator state."""

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--contract-worker",
        "--model-path",
        str(Path(path).resolve()),
        "--device",
        str(device),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
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


def _clear_accelerator_cache(torch_module, device):
    gc.collect()
    if str(device).startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
    elif (
        str(device).startswith("mps")
        and hasattr(torch_module, "backends")
        and torch_module.backends.mps.is_available()
    ):
        torch_module.mps.empty_cache()


def _accelerator_free_bytes(torch_module, device):
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


def _synchronize_accelerator(torch_module, device):
    if str(device).startswith("cuda"):
        synchronize = getattr(torch_module.cuda, "synchronize", None)
        if callable(synchronize):
            index = int(str(device).split(":", 1)[1]) if ":" in str(device) else 0
            synchronize(index)
    elif str(device).startswith("mps"):
        synchronize = getattr(torch_module.mps, "synchronize", None)
        if callable(synchronize):
            synchronize()


def _batch_probe_error_kind(torch_module, error):
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
    torch_module,
    model,
    runtime_info,
    device,
    candidates,
    *,
    reserve_bytes=0,
    progress=None,
    model_id=None,
):
    """Probe one already-resident model without releasing its model object."""

    normalized_candidates = sorted({int(value) for value in candidates if int(value) >= 1})
    if not normalized_candidates or normalized_candidates[0] != 1:
        raise ValueError("Batch probe candidates must start at 1")
    sample = None
    output = None
    records = []
    successful = []
    safe_candidates = []
    stop_reason = "ceiling_reached"
    first_failed_batch = None

    def publish(payload):
        if progress is not None:
            value = dict(payload)
            if model_id is not None:
                value["model_id"] = str(model_id)
            progress(value)

    try:
        for batch_size in normalized_candidates:
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
    torch_module,
    path,
    device,
    candidates,
    *,
    reserve_bytes=0,
    progress=None,
):
    """Load one model once and probe ascending dummy-forward Batch sizes."""

    model = None
    runtime_info = {}
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
    torch_module,
    model_entries,
    device,
    candidates,
    *,
    reserve_bytes=0,
    progress=None,
):
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

    loaded_models = {}
    runtime_info_by_model = {}
    results = {}

    def publish(payload):
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
    path,
    device,
    candidates,
    *,
    reserve_bytes=0,
    timeout=1200,
):
    """Probe one model in one child process and retain progress after a crash."""

    normalized_candidates = [int(value) for value in candidates]
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--batch-probe-worker",
        "--model-path",
        str(Path(path).resolve()),
        "--device",
        str(device),
        "--batch-candidates",
        ",".join(str(value) for value in normalized_candidates),
        "--reserve-bytes",
        str(max(0, int(reserve_bytes))),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
        result = type(
            "TimedOutProbe",
            (),
            {"returncode": -1, "stdout": stdout, "stderr": stderr or "timeout"},
        )()

    final_payload = None
    progress_records = []
    started_batches = []
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
    safe_batch_size = max(safe_candidates or ([1] if successful_candidates else []), default=0)
    failed_batch = next(
        (
            int(record["batch_size"])
            for record in progress_records
            if record.get("status") == "failed"
        ),
        None,
    )
    if failed_batch is None and started_batches:
        completed_batches = {
            int(record["batch_size"]) for record in progress_records
        }
        failed_batch = next(
            (value for value in reversed(started_batches) if value not in completed_batches),
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
    model_entries,
    device,
    candidates,
    *,
    reserve_bytes=0,
    timeout=1200,
):
    """Probe one complete model set in one child and require a final contract."""

    entries = [
        {
            "model_id": str(entry["model_id"]),
            "path": str(Path(entry["path"]).resolve()),
        }
        for entry in model_entries
    ]
    expected_ids = [entry["model_id"] for entry in entries]
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--batch-probe-set-worker",
        "--model-set-json",
        json.dumps(entries, ensure_ascii=False, separators=(",", ":")),
        "--device",
        str(device),
        "--batch-candidates",
        ",".join(str(int(value)) for value in candidates),
        "--reserve-bytes",
        str(max(0, int(reserve_bytes))),
    ]
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
        process = type(
            "TimedOutModelSetProbe",
            (),
            {"returncode": -1, "stdout": stdout, "stderr": stderr or "timeout"},
        )()

    final_payload = None
    loaded_ids = []
    for line in (process.stdout or "").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") == "model_set_load_completed":
            loaded_ids.append(str(payload.get("model_id") or ""))
        elif payload.get("batch_probe_set") is True:
            final_payload = {
                key: value
                for key, value in payload.items()
                if key != "batch_probe_set"
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

    detail = (process.stderr or "").strip() or "worker exited without a complete result"
    message = (
        "complete model-set Batch probe did not finish; no partial Batch value "
        f"may be frozen (loaded={loaded_ids}, exit={process.returncode}): {detail}"
    )
    failed_results = {
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


def verify_sam3_checkpoint(torch_module, path):
    try:
        checkpoint = torch_module.load(path, map_location="cpu", weights_only=True)
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            checkpoint = checkpoint["model"]
        if not isinstance(checkpoint, dict):
            return False, "checkpoint is not a SAM3 state_dict"
        keys = list(checkpoint.keys())
        has_detector = any(str(key).startswith("detector.") for key in keys)
        has_tracker = any(str(key).startswith("tracker.") for key in keys)
        if not (has_detector and has_tracker):
            return False, "official SAM3 detector/tracker parameters are missing"
        return True, f"official SAM3 checkpoint recognized ({len(keys)} entries)"
    except Exception as exc:
        return False, f"SAM3 checkpoint cannot be read: {exc}"


def verify_sam3_runtime_isolated(path, device="cpu", timeout=900):
    """Load the official SAM3 backend through the production compatibility path."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--sam3-worker",
        "--model-path",
        str(Path(path).resolve()),
        "--device",
        str(device),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"official SAM3 load timed out after {timeout}s on {device}"

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
    detail = (result.stderr or "").strip() or (result.stdout or "").strip()
    return False, (
        f"official SAM3 worker failed on {device} (exit={result.returncode}): "
        f"{detail or 'no worker output'}"
    )


def _fingerprint(scripts_dir):
    digest = hashlib.sha256()
    for name in FINGERPRINT_FILES:
        path = scripts_dir / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        if path.is_file() and not path.is_symlink():
            file_digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    file_digest.update(chunk)
            digest.update(file_digest.hexdigest().encode("ascii"))
        else:
            digest.update(b"<missing>")
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def _issue_id(path, index):
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", path).strip("_").lower()
    return f"config_{normalized or 'root'}_{index}"


def _has_issue(issues, prefix, codes=()):
    return any(
        item.path.startswith(prefix) and (not codes or item.code in codes)
        for item in issues
    )


def overall_status(checks):
    """Only core failures or zero runnable semantic models block inference."""
    def optional_capability_error(item):
        check_id = str(item.get("id") or "")
        source = str(item.get("source") or "")
        return (
            check_id.startswith(("semantic_model_", "fusion_profile_", "sam3_"))
            or any(path in source for path in ("/semantic_models/", "/fusion_profiles/", "/sam3/"))
        )

    core_errors = [
        item for item in checks
        if item["status"] == "error" and not optional_capability_error(item)
    ]
    runnable_models = [
        item for item in checks
        if str(item.get("id") or "").startswith("semantic_model_")
        and item["status"] == "ready"
    ]
    if core_errors or not runnable_models:
        return "error"
    if any(item["status"] in ("warning", "error") for item in checks):
        return "warning"
    return "ready"


def add_runtime_boundary_checks(checks, conda_env):
    """Report both runtimes without importing QGIS into the Conda process."""
    conda_python = Path(sys.executable).resolve()
    conda_matches = f"/envs/{conda_env}/" in conda_python.as_posix()
    add_check(
        checks,
        "conda_python",
        "ready" if conda_matches else "error",
        f"{conda_python} (Python {platform.python_version()})",
        f"Conda environment {conda_env}",
        "inference runs in the configured Conda Python" if conda_matches else "environment checker is not running in the configured Conda Python",
        "edit config.sh:CONDA_EXE and CONDA_ENV",
    )

    host_fields = (
        (
            "qgis_version",
            "LOESS_QGIS_VERSION",
            "QGIS",
            lambda value: _version_tuple(value)[:2] in {(3, 44), (4, 2)},
        ),
        (
            "qgis_python",
            "LOESS_QGIS_PYTHON_VERSION",
            "QGIS Python",
            lambda value: _version_tuple(value)[:1] == (3,),
        ),
        (
            "pyqt_version",
            "LOESS_PYQT_VERSION",
            "PyQt",
            lambda value: _version_tuple(value)[:1] in {(5,), (6,)},
        ),
        (
            "qt_version",
            "LOESS_QT_VERSION",
            "Qt",
            lambda value: _version_tuple(value)[:1] in {(5,), (6,)},
        ),
    )
    for check_id, env_name, label, validator in host_fields:
        value = str(os.environ.get(env_name) or "").strip()
        if value:
            status = "ready" if validator(value) else "error"
            message = f"{label} shared-platform compatibility runtime detected"
        else:
            status = "warning"
            value = "not provided"
            message = "run the check from the QGIS plugin to inspect this host value"
        add_check(
            checks,
            check_id,
            status,
            value,
            "QGIS plugin host runtime",
            message,
            "use QGIS 3.44/PyQt5/Qt5 or QGIS 4.2/PyQt6/Qt6",
        )

    qgis_major = _version_tuple(
        os.environ.get("LOESS_QGIS_VERSION", "")
    )[:1]
    pyqt_major = _version_tuple(
        os.environ.get("LOESS_PYQT_VERSION", "")
    )[:1]
    qt_major = _version_tuple(
        os.environ.get("LOESS_QT_VERSION", "")
    )[:1]
    if qgis_major and pyqt_major and qt_major:
        pair = (qgis_major[0], pyqt_major[0], qt_major[0])
        pair_ok = pair in {(3, 5, 5), (4, 6, 6)}
        pair_status = "ready" if pair_ok else "error"
        pair_value = f"QGIS {pair[0]} / PyQt {pair[1]} / Qt {pair[2]}"
        pair_message = (
            "host runtime matches a supported platform profile"
            if pair_ok
            else "host runtime mixes unsupported QGIS and Qt major versions"
        )
    else:
        pair_status = "warning"
        pair_value = "not provided"
        pair_message = "run the check from QGIS to verify the host runtime pair"
    add_check(
        checks,
        "qgis_qt_profile",
        pair_status,
        pair_value,
        "QGIS plugin host runtime",
        pair_message,
        "use QGIS 3.44 with Qt5 or QGIS 4.2 with Qt6",
    )

    host_executable = str(
        os.environ.get("LOESS_QGIS_PYTHON_EXECUTABLE") or ""
    ).strip()
    if host_executable:
        separate = Path(host_executable).resolve() != conda_python
        status = "ready" if separate else "error"
        message = (
            "QGIS and inference use separate Python runtimes"
            if separate
            else "QGIS and inference unexpectedly share one Python executable"
        )
    else:
        status = "warning"
        host_executable = "not provided"
        message = "run the check from the QGIS plugin to verify runtime separation"
    add_check(
        checks,
        "runtime_boundary",
        status,
        f"QGIS={host_executable}; inference={conda_python}",
        "QProcess environment and config.sh",
        message,
        "do not add Conda site-packages to the QGIS Python runtime",
    )


def build_report(args):
    scripts_dir = Path(args.scripts_dir).resolve()
    config_path = scripts_dir / "config.yaml"
    checks = []
    effective = {
        "schema_version": None,
        "runtime": {},
        "scaling": {},
        "semantic_models": [],
        "fusion_profiles": [],
        "sam3": {},
        "boundary_fitting": {},
        "classes": {},
    }

    add_check(
        checks,
        "conda_env",
        "ready",
        args.conda_env,
        "config.sh:CONDA_ENV",
        "environment checker is running inside this Conda environment",
        "edit config.sh:CONDA_ENV",
    )
    add_runtime_boundary_checks(checks, args.conda_env)

    dependencies = {}
    for name in (
        "numpy",
        "torch",
        "rasterio",
        "fiona",
        "shapely",
        "scipy",
        "psutil",
        "yaml",
        "skimage",
        "psycopg2",
    ):
        module, version, error = import_dependency(name)
        dependencies[name] = module
        add_check(
            checks,
            f"dependency_{name}",
            "ready" if module is not None else "error",
            f"{name} {version}",
            f"Conda environment {args.conda_env}",
            error,
            f"install {name} in Conda environment {args.conda_env}",
        )

    runtime_platform = "macos" if sys.platform == "darwin" else "ubuntu"
    torch_module = dependencies.get("torch")
    torch_version = str(getattr(torch_module, "__version__", "not installed"))
    expected_torch_version = "2.7.1" if runtime_platform == "macos" else "2.6.0"
    torch_version_ok = (
        torch_module is not None
        and torch_version.split("+", 1)[0] == expected_torch_version
    )
    add_check(
        checks,
        "torch_version",
        "ready" if torch_version_ok else "error",
        torch_version,
        f"Conda environment {args.conda_env}",
        f"formal {runtime_platform} runtime uses PyTorch {expected_torch_version}"
        if torch_version_ok
        else (
            f"{runtime_platform} deployment requires exact PyTorch "
            f"{expected_torch_version}"
        ),
        "run <repository>/bash/init_project.sh --project-root <project> --create-env",
    )

    if runtime_platform == "ubuntu":
        cuda_build = str(
            getattr(getattr(torch_module, "version", None), "cuda", "") or ""
        )
        cuda_build_ok = (
            torch_module is not None
            and _version_tuple(cuda_build)[:2] == (12, 4)
        )
        add_check(
            checks,
            "torch_cuda_build",
            "ready" if cuda_build_ok else "error",
            cuda_build or "not available",
            "torch.version.cuda",
            "PyTorch CUDA 12.4 runtime is installed"
            if cuda_build_ok
            else "the installed PyTorch build is not the required cu124 build",
            "install torch 2.6.0 from the official cu124 wheel index",
        )

        cuda_available = bool(
            torch_module is not None and torch_module.cuda.is_available()
        )
        gpu_name = "not available"
        capability = "not available"
        if cuda_available:
            try:
                gpu_name = str(torch_module.cuda.get_device_name(0))
                capability = ".".join(
                    str(value)
                    for value in torch_module.cuda.get_device_capability(0)
                )
            except Exception as exc:
                gpu_name = f"query failed: {exc}"
        gpu_ok = cuda_available and "3090" in gpu_name
        add_check(
            checks,
            "cuda_gpu",
            "ready" if gpu_ok else "error",
            f"{gpu_name}; compute capability {capability}",
            "CUDA device 0",
            "RTX 3090 is available to PyTorch"
            if gpu_ok
            else "CUDA device 0 is unavailable or is not an RTX 3090",
            "check the NVIDIA driver, CUDA_VISIBLE_DEVICES=0 and RTX 3090",
        )
        add_check(
            checks,
            "mps_device",
            "ready",
            "not required",
            "Ubuntu platform profile",
            "Ubuntu formal inference uses CUDA",
            "",
        )
    else:
        mps_available = bool(
            torch_module is not None
            and hasattr(torch_module.backends, "mps")
            and torch_module.backends.mps.is_available()
        )
        add_check(
            checks,
            "torch_cuda_build",
            "ready",
            "not required",
            "macOS platform profile",
            "macOS formal inference uses MPS",
            "",
        )
        add_check(
            checks,
            "cuda_gpu",
            "ready",
            "not required",
            "macOS platform profile",
            "macOS formal inference uses MPS",
            "",
        )
        add_check(
            checks,
            "mps_device",
            "ready" if mps_available else "error",
            "available" if mps_available else "not available",
            "torch.backends.mps",
            "MPS is available to PyTorch"
            if mps_available
            else "MPS is unavailable in the macOS inference environment",
            "install the macOS platform environment and verify Apple GPU access",
        )

    shapely_module = dependencies.get("shapely")
    if shapely_module is not None:
        divider_apis = ("STRtree",)
        missing_divider_apis = [
            name for name in divider_apis if not hasattr(shapely_module, name)
        ]
        divider_available = not missing_divider_apis
        add_check(
            checks,
            "dependency_shapely_divider_query",
            "ready" if divider_available else "error",
            f"shapely {getattr(shapely_module, '__version__', 'unknown')}",
            f"Conda environment {args.conda_env}",
            "Polygon neighbor query for common-divider fitting is available"
            if divider_available
            else "missing APIs: " + ", ".join(missing_divider_apis),
            f"install shapely=2.1.2 in Conda environment {args.conda_env}",
        )

    scipy_module = dependencies.get("scipy")
    scipy_spline_error = ""
    if scipy_module is not None:
        try:
            from scipy.interpolate import splprep, splev  # noqa: F401
        except Exception as exc:
            scipy_spline_error = str(exc)
    add_check(
        checks,
        "dependency_scipy_bspline",
        "ready" if scipy_module is not None and not scipy_spline_error else "error",
        f"scipy {getattr(scipy_module, '__version__', 'not installed')}",
        f"Conda environment {args.conda_env}",
        scipy_spline_error or "splprep and splev are available",
        f"install scipy=1.17.1 in Conda environment {args.conda_env}",
    )

    postgres_error = ""
    postgres_value = "not available"
    try:
        state = RunStateDB(production_state_database())
        state.initialize()
        health = state.pragmas()
        postgres_value = (
            f"PostgreSQL {health['server_version']} / "
            f"{health['database']} / {health['schema']}"
        )
    except Exception as exc:
        postgres_error = str(exc)
    add_check(
        checks,
        "dependency_postgresql_state",
        "ready" if not postgres_error else "error",
        postgres_value,
        f"Conda environment {args.conda_env}",
        postgres_error or "PostgreSQL control-plane schema is writable and compatible",
        f"install psycopg2 in {args.conda_env} and verify the local PostgreSQL service",
    )

    add_check(
        checks,
        "dependency_geopackage_sqlite",
        "ready",
        f"sqlite {sqlite3.sqlite_version}",
        "Python standard library / GeoPackage",
        "SQLite remains available only for GeoPackage integrity checks",
        "",
    )

    gdal_versions = []
    gdal_error = ""
    for executable in ("gdalinfo", "gdalbuildvrt"):
        conda_candidate = Path(sys.executable).resolve().parent / executable
        path = str(conda_candidate) if conda_candidate.is_file() else shutil.which(executable)
        if path is None:
            gdal_error = f"{executable} is not on PATH"
            break
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            gdal_error = (result.stderr or result.stdout or f"{executable} failed").strip()
            break
        gdal_versions.append((result.stdout or result.stderr).strip())
    add_check(
        checks,
        "dependency_gdal_cli",
        "ready" if not gdal_error else "error",
        "; ".join(gdal_versions) if gdal_versions else "not available",
        f"Conda environment {args.conda_env}",
        gdal_error or "gdalinfo and gdalbuildvrt are available; Python osgeo is not required",
        f"install libgdal-core=3.12.3 in Conda environment {args.conda_env}",
    )

    pytest_module, pytest_version, pytest_error = import_dependency("pytest")
    add_check(
        checks,
        "developer_pytest",
        "ready" if pytest_module is not None else "warning",
        f"pytest {pytest_version}",
        f"Conda environment {args.conda_env}",
        pytest_error or "repository test runner is available",
        f"install pytest=9.1.1 in Conda environment {args.conda_env}",
    )

    config = {}
    issues = []
    if not config_path.is_file():
        add_check(
            checks,
            "config_yaml",
            "error",
            str(config_path),
            "inference scripts directory",
            "config.yaml does not exist",
            "create a Schema v2 config.yaml",
        )
    else:
        try:
            config = load_yaml(config_path)
            effective, issues = validate_deployment_config(
                config,
                scripts_dir=scripts_dir,
                verify_files=True,
                verify_hashes=True,
            )
            add_check(
                checks,
                "config_yaml",
                "ready" if config.get("schema_version") == 2 else "error",
                str(config_path),
                "config.yaml:schema_version",
                "Schema v2 parsed" if config.get("schema_version") == 2 else "Schema v2 is required",
                "replace legacy model.semantic_weight config with Schema v2",
            )
        except Exception as exc:
            add_check(
                checks,
                "config_yaml",
                "error",
                str(config_path),
                "config.yaml",
                f"cannot parse config: {exc}",
                "fix the reported YAML syntax or field",
            )

    for index, issue in enumerate(issues):
        add_check(
            checks,
            _issue_id(issue.path, index),
            "error",
            issue.path,
            f"config.yaml{issue.path}",
            issue.message,
            f"edit config.yaml field {issue.path}",
        )

    torch_module = dependencies.get("torch")
    runtime = effective.get("runtime") or {}
    requested_device = str(runtime.get("requested_device") or "auto")
    resolved_device = resolve_device(requested_device)
    device_ok = validate_device(resolved_device)
    runtime["effective_device"] = resolved_device
    hardware = collect_hardware_snapshot(
        device=resolved_device,
        psutil_module=dependencies.get("psutil"),
        torch_module=torch_module,
    )
    runtime, resolved_scaling, resource_tuning = resolve_hardware_tuning(
        runtime,
        effective.get("scaling") or {},
        hardware,
    )
    effective["runtime"] = runtime
    effective["scaling"] = resolved_scaling
    effective["resource_tuning"] = resource_tuning
    device_status = "ready" if device_ok else "error"
    device_message = ""
    if not device_ok:
        device_message = f"requested device is unavailable: {resolved_device}"
    elif requested_device == "auto" and resolved_device == "cpu":
        device_status = "warning"
        device_message = "CUDA and MPS are unavailable; auto selected CPU"
    add_check(
        checks,
        "semantic_device",
        device_status,
        resolved_device,
        "config.yaml:runtime.device",
        device_message,
        "edit runtime.device or repair the PyTorch device environment",
    )

    batch_auto = "tile_batch_size" in set(
        resource_tuning.get("automatic_fields") or []
    )
    probe_candidates = model_batch_probe_candidates(hardware)
    probe_reserve_bytes = batch_probe_safety_reserve_bytes(hardware)
    model_batch_probes = {}
    if (
        batch_auto
        and torch_module is not None
        and device_ok
        and str(resolved_device).startswith(("mps", "cuda"))
    ):
        eligible_probe_entries = []
        for index, model in enumerate(effective.get("semantic_models") or []):
            model_id = str(model["model_id"])
            prefix = f"/semantic_models/{index}"
            if _has_issue(issues, prefix):
                continue
            if str(resolved_device).startswith("mps") and not _mps_runtime_requirement(
                model_id, getattr(torch_module, "__version__", "0")
            )[0]:
                continue
            eligible_probe_entries.append(
                {"model_id": model_id, "path": model.get("artifact_path", "")}
            )
        if eligible_probe_entries:
            model_set_probe = verify_torchscript_model_set_batch_probe_isolated(
                eligible_probe_entries,
                resolved_device,
                probe_candidates,
                reserve_bytes=probe_reserve_bytes,
            )
            model_batch_probes.update(model_set_probe.get("results") or {})

    for index, model in enumerate(effective.get("semantic_models") or []):
        model_id = model["model_id"]
        prefix = f"/semantic_models/{index}"
        path = model.get("artifact_path", "")
        blocked = _has_issue(issues, prefix)
        if blocked:
            status = "error"
            message = "model registry entry or deployment asset is invalid"
            fix = f"check the config entry, artifact, and SHA256 for {model_id}"
        elif torch_module is None or not device_ok:
            status = "error"
            message = "PyTorch or effective device is unavailable"
            fix = "repair the PyTorch environment or select an available device"
        elif str(resolved_device).startswith("mps") and not _mps_runtime_requirement(
            model_id, getattr(torch_module, "__version__", "0")
        )[0]:
            status = "error"
            _compatible, message = _mps_runtime_requirement(
                model_id, getattr(torch_module, "__version__", "0")
            )
            fix = "upgrade torch to >=2.7 with matching torchvision and torchaudio in the configured Conda environment"
        else:
            if batch_auto and str(resolved_device).startswith(("mps", "cuda")):
                probe_result = model_batch_probes.get(model_id) or {
                    "ok": False,
                    "safe_batch_size": 0,
                    "stop_reason": "model_set_probe_missing",
                    "model_set_complete": False,
                    "message": (
                        "model was not included in a complete resident model-set "
                        "Batch probe"
                    ),
                }
                ok = bool(probe_result.get("ok"))
                message = str(probe_result.get("message") or "")
            elif str(resolved_device).startswith(("mps", "cuda")):
                ok, message = verify_torchscript_contract_isolated(path, resolved_device)
            else:
                ok, message = verify_torchscript_contract(torch_module, path, resolved_device)
                if ok and batch_auto:
                    model_batch_probes[model_id] = {
                        "ok": True,
                        "safe_batch_size": 1,
                        "max_successful_batch": 1,
                        "first_failed_batch": None,
                        "stop_reason": "cpu_conservative",
                        "reserve_bytes": 0,
                        "runtime_info": {},
                        "probes": [{"batch_size": 1, "status": "passed"}],
                        "message": "CPU uses conservative Batch 1",
                    }
            status = "ready" if ok else "error"
            if not ok and str(resolved_device).startswith("mps"):
                fix = (
                    "inspect inference_scripts/torchscript_runtime.py MPS graph compatibility; "
                    "the hash-valid artifact may still be valid on CPU/CUDA"
                )
            else:
                fix = f"deploy and register a valid TorchScript artifact for {model_id}"
        add_check(
            checks,
            f"semantic_model_{model_id}",
            status,
            f"{model_id}: {path}",
            f"config.yaml:semantic_models[{index}]",
            message,
            fix,
        )

    runtime, resource_tuning = freeze_model_batch_probe_results(
        runtime,
        resource_tuning,
        model_batch_probes,
    )
    effective["runtime"] = runtime
    effective["resource_tuning"] = resource_tuning
    resolved_resources = resource_tuning["resolved"]
    batch_by_model = resolved_resources.get("tile_batch_size_by_model") or {}
    probe_required = batch_auto and device_ok and torch_module is not None
    tuning_status = "ready" if (not probe_required or batch_by_model) else "error"
    model_batches = ", ".join(
        f"{model_id}={batch_size}"
        for model_id, batch_size in sorted(batch_by_model.items())
    )
    add_check(
        checks,
        "resource_tuning",
        tuning_status,
        (
            f"CPU {resolved_resources['max_cpu_partition_workers']}"
            f"/{resolved_resources['max_cpu_partition_workers_with_package']}; "
            f"Tile batch {resolved_resources['tile_batch_size']}"
            + (f" ({model_batches})" if model_batches else "")
            + f"; Tile I/O {resolved_resources['tile_io_workers']}; "
            f"assembly {resolved_resources['max_concurrent_assembly']}×"
            f"{resolved_resources['assembly_validation_workers']}"
        ),
        "automatic hardware tuning and isolated resident model-set Batch probes",
        (
            "all runnable models were loaded together in one isolated process; "
            "per-model Batch values were then probed once and frozen while the "
            "complete model set remained resident"
            if batch_by_model
            else "no verified per-model Batch result is available"
        ),
        "repair the model/device contract and rerun the environment check",
    )

    for index, profile_entry in enumerate(effective.get("fusion_profiles") or []):
        profile_id = profile_entry["profile_id"]
        prefix = f"/fusion_profiles/{index}"
        if _has_issue(issues, prefix):
            status = "error"
            message = "profile file, schema, model reference, or hash is invalid"
        elif profile_entry.get("status") == "rejected":
            status = "warning"
            message = "profile is rejected and cannot be selected for formal fusion"
        elif profile_entry.get("available"):
            status = "ready"
            message = (
                f"approved {profile_entry.get('strategy')} profile; "
                f"models={','.join(profile_entry.get('required_model_ids') or [])}"
            )
        else:
            status = "error"
            message = "profile is not available for formal fusion"
        add_check(
            checks,
            f"fusion_profile_{profile_id}",
            status,
            profile_entry.get("file_path", ""),
            f"config.yaml:fusion_profiles[{index}]",
            message,
            f"deploy an approved and matching fusion profile for {profile_id}",
        )

    classes = effective.get("classes") or {}
    class_ok = (
        classes.get("background_index") == -1
        and [int((classes.get("index_to_code") or {}).get(str(index), -999)) for index in range(14)] == CLASS_ORDER
    )
    add_check(
        checks,
        "class_contract",
        "ready" if class_ok else "error",
        "14 valid classes; nodata=-1",
        "config.yaml:classes",
        "" if class_ok else "class order is not the fixed 14-class contract",
        "restore the documented index_to_code mapping and background_index=-1",
    )

    sam = effective.get("sam3") or {}
    if sam.get("enabled"):
        sam_checkpoint = sam.get("checkpoint", "")
        sam_requested = str(sam.get("requested_device") or "auto")
        if sam_requested == "auto":
            sam_device = resolved_device if resolved_device.startswith("cuda") else "cpu"
        else:
            sam_device = resolve_device(sam_requested)
        if sam_device == "mps":
            sam_device = "cpu"
            sam_device_status = "warning"
            sam_device_message = "official SAM3 has no stable MPS runtime; using CPU"
        else:
            sam_device_status = "ready" if validate_device(sam_device) else "error"
            sam_device_message = "" if sam_device_status == "ready" else f"SAM3 device unavailable: {sam_device}"
        sam["effective_device"] = sam_device
        effective["sam3"] = sam
        add_check(
            checks,
            "sam3_device",
            sam_device_status,
            sam_device,
            "config.yaml:sam3.device",
            sam_device_message,
            "use CUDA or CPU for SAM3",
        )

        try:
            installed = importlib.metadata.version("sam3")
            backend_ok, backend_value, backend_error = True, f"official sam3 {installed}", ""
        except importlib.metadata.PackageNotFoundError:
            backend_ok, backend_value, backend_error = False, "not installed", "official sam3 package is missing"
        add_check(
            checks,
            "sam3_backend",
            "ready" if backend_ok else "error",
            backend_value,
            f"Conda environment {args.conda_env}",
            backend_error,
            "install the official sam3 package",
        )

        tokenizer = scripts_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        add_check(
            checks,
            "sam3_tokenizer",
            "ready" if tokenizer.is_file() else "error",
            str(tokenizer),
            "inference_scripts/assets",
            "" if tokenizer.is_file() else "SAM3 tokenizer asset is missing",
            "deploy the official tokenizer asset",
        )
        if torch_module is not None and Path(sam_checkpoint).is_file():
            ok, message = verify_sam3_checkpoint(torch_module, sam_checkpoint)
            if ok and backend_ok and sam_device_status != "error":
                runtime_ok, runtime_message = verify_sam3_runtime_isolated(
                    sam_checkpoint, sam_device
                )
                ok = runtime_ok
                message = f"{message}; {runtime_message}"
        else:
            ok, message = False, "SAM3 checkpoint or PyTorch is unavailable"
        add_check(
            checks,
            "sam3_model_load",
            "ready" if ok else "error",
            sam_checkpoint,
            "config.yaml:sam3.checkpoint",
            message,
            "deploy a valid official SAM3 checkpoint",
        )
    else:
        add_check(
            checks,
            "sam3_enabled",
            "warning",
            "disabled",
            "config.yaml:sam3.enabled",
            "semantic inference remains available; class refinement is disabled",
            "enable SAM3 only when its deployment assets are ready",
        )

    output_dir = str(args.output_dir or "").strip()
    if output_dir:
        output_path = Path(output_dir).expanduser().resolve()
        probe = output_path if output_path.exists() else output_path.parent
        writable = probe.exists() and os.access(str(probe), os.W_OK)
        add_check(
            checks,
            "output_dir",
            "ready" if writable else "error",
            str(output_path),
            "main panel:output workspace",
            "" if writable else "output directory or parent is not writable",
            "select a writable output workspace",
        )

    status = overall_status(checks)
    return {
        "schema_version": 1,
        "status": status,
        "config_fingerprint": _fingerprint(scripts_dir),
        "effective": effective,
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate Schema v2 inference deployment")
    parser.add_argument("--scripts-dir")
    parser.add_argument("--conda-env")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--report-json", default="")
    parser.add_argument("--contract-worker", action="store_true")
    parser.add_argument("--batch-probe-worker", action="store_true")
    parser.add_argument("--batch-probe-set-worker", action="store_true")
    parser.add_argument("--sam3-worker", action="store_true")
    parser.add_argument("--model-path")
    parser.add_argument("--model-set-json", default="")
    parser.add_argument("--device")
    parser.add_argument("--batch-candidates", default="")
    parser.add_argument("--reserve-bytes", type=int, default=0)
    args = parser.parse_args()
    if args.batch_probe_set_worker:
        if not args.model_set_json or not args.device or not args.batch_candidates:
            parser.error(
                "--batch-probe-set-worker requires --model-set-json, --device "
                "and --batch-candidates"
            )
        try:
            model_entries = json.loads(args.model_set_json)
            if not isinstance(model_entries, list):
                raise ValueError("model set must be a JSON list")
            candidates = [
                int(value)
                for value in args.batch_candidates.split(",")
                if value.strip()
            ]
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            parser.error(f"invalid model-set Batch probe arguments: {error}")
        import torch

        def publish_model_set_probe_event(payload):
            sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
            sys.stdout.flush()

        probe_result = probe_torchscript_model_set_batches(
            torch,
            model_entries,
            args.device,
            candidates,
            reserve_bytes=max(0, int(args.reserve_bytes)),
            progress=publish_model_set_probe_event,
        )
        sys.stdout.write(
            json.dumps(
                {"batch_probe_set": True, **probe_result},
                separators=(",", ":"),
            )
            + "\n"
        )
        return 0 if probe_result["ok"] else 2
    if args.batch_probe_worker:
        if not args.model_path or not args.device or not args.batch_candidates:
            parser.error(
                "--batch-probe-worker requires --model-path, --device and "
                "--batch-candidates"
            )
        try:
            candidates = [
                int(value)
                for value in args.batch_candidates.split(",")
                if value.strip()
            ]
        except ValueError:
            parser.error("--batch-candidates must be comma-separated integers")
        import torch

        def publish_probe_event(payload):
            sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
            sys.stdout.flush()

        probe_result = probe_torchscript_batches(
            torch,
            args.model_path,
            args.device,
            candidates,
            reserve_bytes=max(0, int(args.reserve_bytes)),
            progress=publish_probe_event,
        )
        sys.stdout.write(
            json.dumps(
                {"batch_probe": True, **probe_result},
                separators=(",", ":"),
            )
            + "\n"
        )
        return 0 if probe_result["ok"] else 2
    if args.contract_worker:
        if not args.model_path or not args.device:
            parser.error("--contract-worker requires --model-path and --device")
        import torch

        ok, message = verify_torchscript_contract(torch, args.model_path, args.device)
        sys.stdout.write(json.dumps({"ok": ok, "message": message}) + "\n")
        return 0
    if args.sam3_worker:
        if not args.model_path or not args.device:
            parser.error("--sam3-worker requires --model-path and --device")
        try:
            from sam3_refine import load_sam3

            runtime = load_sam3(args.model_path, args.device)
            runtime = None
            gc.collect()
            ok, message = True, f"official SAM3 runtime loaded on {args.device}"
        except Exception as exc:
            ok = False
            message = f"official SAM3 runtime failed on {args.device}: {exc}"
        sys.stdout.write(json.dumps({"ok": ok, "message": message}) + "\n")
        return 0 if ok else 2
    if not args.scripts_dir or not args.conda_env:
        parser.error("--scripts-dir and --conda-env are required")
    try:
        report = build_report(args)
    except Exception as exc:
        report = {
            "schema_version": 1,
            "status": "error",
            "config_fingerprint": "",
            "effective": {},
            "checks": [{
                "id": "environment_check",
                "status": "error",
                "value": "failed",
                "source": "check_environment.py",
                "message": str(exc),
                "fix": "inspect the detailed environment log",
            }],
        }
    serialized = json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n"
    if args.report_json:
        report_path = Path(args.report_json).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = report_path.with_name(report_path.name + ".tmp")
        temporary_path.write_text(serialized, encoding="utf-8")
        os.replace(temporary_path, report_path)
    sys.stdout.write(serialized)
    return 0 if report["status"] != "error" else 2


if __name__ == "__main__":
    raise SystemExit(main())
