"""Validate deployment configuration and append capability checks."""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from labeling_tool.shared.contracts.run_spec import CLASS_ORDER
from loess_runtime.system._device import resolve_device, validate_device
from loess_runtime.system.deployment_config import (
    load_yaml,
    validate_deployment_config,
)
from loess_runtime.system.environment_checks import (
    add_check,
    has_issue,
    issue_id,
    mps_runtime_requirement,
)
from loess_runtime.system.hardware_tuning import (
    batch_probe_safety_reserve_bytes,
    collect_hardware_snapshot,
    freeze_model_batch_probe_results,
    model_batch_probe_candidates,
    resolve_hardware_tuning,
)
from loess_runtime.system.model_probe import (
    verify_torchscript_contract,
    verify_torchscript_contract_isolated,
    verify_torchscript_model_set_batch_probe_isolated,
)
from loess_runtime.system.probe_process import (
    environment_worker_command,
    run_environment_worker,
)

SAM_TOKENIZER_PATH = (
    Path(__file__).resolve().parents[1]
    / "sam"
    / "assets"
    / "bpe_simple_vocab_16e6.txt.gz"
)


@dataclass(frozen=True)
class RuntimeProbeInputs:
    """Immutable device and hardware observations used by capability checks."""

    torch_module: Any
    resolved_device: str
    device_ok: bool
    hardware: dict[str, Any]


def empty_effective() -> dict[str, Any]:
    """Return the report's stable empty deployment shape."""

    return {
        "schema_version": None,
        "runtime": {},
        "scaling": {},
        "semantic_models": [],
        "fusion_profiles": [],
        "sam3": {},
        "boundary_fitting": {},
        "classes": {},
    }


def verify_sam3_checkpoint(
    torch_module: Any,
    path: str | Path,
) -> tuple[bool, str]:
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


def verify_sam3_runtime_isolated(
    path: str | Path,
    device: str = "cpu",
    timeout: int = 900,
) -> tuple[bool, str]:
    """Load the official SAM3 backend through the production compatibility path."""
    command = environment_worker_command(
        "--sam3-worker",
        "--model-path",
        str(Path(path).resolve()),
        "--device",
        str(device),
    )
    try:
        result = run_environment_worker(command, timeout=timeout)
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


def load_deployment_configuration(
    checks: list[dict[str, str]],
    scripts_dir: Path,
    asset_base_dir: Path,
) -> tuple[dict[str, Any], list[Any]]:
    """Load Schema v2 configuration and append parse and validation issues."""

    config_path = scripts_dir / "config.yaml"
    effective = empty_effective()
    config: dict[str, Any] = {}
    issues: list[Any] = []
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
                asset_base_dir=asset_base_dir,
                verify_files=True,
                verify_hashes=True,
            )
            add_check(
                checks,
                "config_yaml",
                "ready" if config.get("schema_version") == 2 else "error",
                str(config_path),
                "config.yaml:schema_version",
                "Schema v2 parsed"
                if config.get("schema_version") == 2
                else "Schema v2 is required",
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
            issue_id(issue.path, index),
            "error",
            issue.path,
            f"config.yaml{issue.path}",
            issue.message,
            f"edit config.yaml field {issue.path}",
        )
    return effective, issues


def append_runtime_tuning(
    checks: list[dict[str, str]],
    effective: dict[str, Any],
    dependencies: dict[str, Any],
) -> RuntimeProbeInputs:
    """Resolve the effective device and automatic throughput profile."""

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
    return RuntimeProbeInputs(
        torch_module=torch_module,
        resolved_device=resolved_device,
        device_ok=device_ok,
        hardware=hardware,
    )


def append_semantic_model_checks(
    checks: list[dict[str, str]],
    effective: dict[str, Any],
    issues: list[Any],
    probe_inputs: RuntimeProbeInputs,
) -> None:
    """Measure semantic models and freeze per-model automatic Batch values."""

    torch_module = probe_inputs.torch_module
    resolved_device = probe_inputs.resolved_device
    device_ok = probe_inputs.device_ok
    runtime = effective.get("runtime") or {}
    resource_tuning = effective.get("resource_tuning") or {}
    hardware = probe_inputs.hardware
    batch_auto = "tile_batch_size" in set(resource_tuning.get("automatic_fields") or [])
    probe_candidates = model_batch_probe_candidates(
        hardware,
        maximum_batch_size=int(runtime.get("tile_batch_size") or 1),
    )
    probe_reserve_bytes = batch_probe_safety_reserve_bytes(hardware)
    model_batch_probes: dict[str, dict[str, Any]] = {}
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
            if has_issue(issues, prefix):
                continue
            if (
                str(resolved_device).startswith("mps")
                and not mps_runtime_requirement(
                    model_id, getattr(torch_module, "__version__", "0")
                )[0]
            ):
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
            cleanup_warning = str(
                model_set_probe.get("worker_cleanup_warning") or ""
            ).strip()
            if cleanup_warning:
                add_check(
                    checks,
                    "model_batch_probe_worker_cleanup",
                    "warning",
                    f"exit={model_set_probe.get('worker_exit_code')}",
                    "isolated complete model-set Batch probe",
                    cleanup_warning,
                    "inspect the NVIDIA driver log before a production Run",
                )
    for index, model in enumerate(effective.get("semantic_models") or []):
        model_id = model["model_id"]
        prefix = f"/semantic_models/{index}"
        path = model.get("artifact_path", "")
        blocked = has_issue(issues, prefix)
        if blocked:
            status = "error"
            message = "model registry entry or deployment asset is invalid"
            fix = f"check the config entry, artifact, and SHA256 for {model_id}"
        elif torch_module is None or not device_ok:
            status = "error"
            message = "PyTorch or effective device is unavailable"
            fix = "repair the PyTorch environment or select an available device"
        elif (
            str(resolved_device).startswith("mps")
            and not mps_runtime_requirement(
                model_id, getattr(torch_module, "__version__", "0")
            )[0]
        ):
            status = "error"
            _compatible, message = mps_runtime_requirement(
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
                ok, message = verify_torchscript_contract_isolated(
                    path, resolved_device
                )
            else:
                ok, message = verify_torchscript_contract(
                    torch_module, path, resolved_device
                )
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
                    "inspect inference_scripts/loess_runtime/inference/"
                    "torchscript_runtime.py MPS graph compatibility; "
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


def append_fusion_and_class_checks(
    checks: list[dict[str, str]],
    effective: dict[str, Any],
    issues: list[Any],
) -> None:
    """Append fusion profile availability and fixed class-contract checks."""

    for index, profile_entry in enumerate(effective.get("fusion_profiles") or []):
        profile_id = profile_entry["profile_id"]
        prefix = f"/fusion_profiles/{index}"
        if has_issue(issues, prefix):
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
        and [
            int((classes.get("index_to_code") or {}).get(str(index), -999))
            for index in range(14)
        ]
        == CLASS_ORDER
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


def append_sam_checks(
    checks: list[dict[str, str]],
    effective: dict[str, Any],
    probe_inputs: RuntimeProbeInputs,
    conda_env: str,
) -> None:
    """Append optional SAM3 device, package, asset, and load checks."""

    torch_module = probe_inputs.torch_module
    resolved_device = probe_inputs.resolved_device
    sam = effective.get("sam3") or {}
    if sam.get("enabled"):
        sam_checkpoint = sam.get("checkpoint", "")
        sam_requested = str(sam.get("requested_device") or "auto")
        if sam_requested == "auto":
            sam_device = (
                resolved_device if resolved_device.startswith("cuda") else "cpu"
            )
        else:
            sam_device = resolve_device(sam_requested)
        if sam_device == "mps":
            sam_device = "cpu"
            sam_device_status = "warning"
            sam_device_message = "official SAM3 has no stable MPS runtime; using CPU"
        else:
            sam_device_status = "ready" if validate_device(sam_device) else "error"
            sam_device_message = (
                ""
                if sam_device_status == "ready"
                else f"SAM3 device unavailable: {sam_device}"
            )
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
            backend_ok, backend_value, backend_error = (
                True,
                f"official sam3 {installed}",
                "",
            )
        except importlib.metadata.PackageNotFoundError:
            backend_ok, backend_value, backend_error = (
                False,
                "not installed",
                "official sam3 package is missing",
            )
        add_check(
            checks,
            "sam3_backend",
            "ready" if backend_ok else "error",
            backend_value,
            f"Conda environment {conda_env}",
            backend_error,
            "install the official sam3 package",
        )
        tokenizer = SAM_TOKENIZER_PATH
        add_check(
            checks,
            "sam3_tokenizer",
            "ready" if tokenizer.is_file() else "error",
            str(tokenizer),
            "inference_scripts/loess_runtime/sam/assets",
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


def append_output_directory_check(
    checks: list[dict[str, str]],
    output_dir: str,
) -> None:
    """Append the optional output-workspace writability check."""

    output_dir = str(output_dir or "").strip()
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
