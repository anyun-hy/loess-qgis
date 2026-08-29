"""Resolve one bounded throughput profile from the active inference hardware."""

from __future__ import annotations

import os
import platform
from typing import Any, Mapping


GIB = 1024**3
TUNING_SCHEMA_VERSION = 2
BATCH_PROBE_SCHEMA_VERSION = 1
CUDA_BATCH_PROBE_CEILING = 128
MPS_BATCH_PROBE_CEILING = 64


def collect_hardware_snapshot(
    *,
    device: str,
    psutil_module: Any | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Return only small, serializable facts used by automatic tuning."""

    logical = int(os.cpu_count() or 1)
    physical = logical
    total_memory = 0
    available_memory = 0
    if psutil_module is not None:
        try:
            physical = int(psutil_module.cpu_count(logical=False) or logical)
        except Exception:
            physical = logical
        try:
            memory = psutil_module.virtual_memory()
            total_memory = int(memory.total)
            available_memory = int(memory.available)
        except Exception:
            pass

    target = str(device or "cpu")
    accelerator_kind = (
        "cuda" if target.startswith("cuda")
        else "mps" if target.startswith("mps")
        else "cpu"
    )
    accelerator_name = "CPU"
    accelerator_total = 0
    accelerator_free = 0
    if accelerator_kind == "cuda" and torch_module is not None:
        try:
            index = int(target.split(":", 1)[1]) if ":" in target else 0
            accelerator_name = str(torch_module.cuda.get_device_name(index))
            accelerator_total = int(
                torch_module.cuda.get_device_properties(index).total_memory
            )
            accelerator_free = int(torch_module.cuda.mem_get_info(index)[0])
        except Exception:
            accelerator_name = target
    elif accelerator_kind == "mps":
        accelerator_name = "Apple MPS"
        accelerator_total = total_memory
        accelerator_free = available_memory

    return {
        "platform": "macos" if platform.system() == "Darwin" else "ubuntu",
        "logical_cpu_count": max(1, logical),
        "physical_cpu_count": max(1, physical),
        "memory_total_bytes": max(0, total_memory),
        "memory_available_bytes": max(0, available_memory),
        "accelerator_kind": accelerator_kind,
        "accelerator_name": accelerator_name,
        "accelerator_memory_total_bytes": max(0, accelerator_total),
        "accelerator_memory_free_bytes": max(0, accelerator_free),
    }


def _configured_or_auto(value: Any, automatic: int) -> tuple[int, bool]:
    if str(value).strip().lower() == "auto":
        return max(1, int(automatic)), True
    return max(1, int(value)), False


def _automatic_tile_batch_size(hardware: Mapping[str, Any]) -> int:
    kind = str(hardware.get("accelerator_kind") or "cpu")
    accelerator_memory = int(
        hardware.get("accelerator_memory_total_bytes") or 0
    )
    system_memory = int(hardware.get("memory_total_bytes") or 0)
    if kind == "cuda":
        if accelerator_memory >= 20 * GIB:
            return 16
        if accelerator_memory >= 10 * GIB:
            return 8
    if kind == "mps":
        if system_memory >= 48 * GIB:
            return 16
        if system_memory >= 32 * GIB:
            return 8
        if system_memory >= 24 * GIB:
            return 4
    return 1


def batch_probe_safety_reserve_bytes(hardware: Mapping[str, Any]) -> int:
    """Return accelerator headroom that a successful probe must leave free."""

    kind = str(hardware.get("accelerator_kind") or "cpu")
    total = int(hardware.get("accelerator_memory_total_bytes") or 0)
    if kind == "cuda":
        return max(2 * GIB, int(total * 0.10)) if total > 0 else 2 * GIB
    if kind == "mps":
        return max(4 * GIB, int(total * 0.20)) if total > 0 else 4 * GIB
    return 0


def model_batch_probe_candidates(
    hardware: Mapping[str, Any],
) -> list[int]:
    """Build one bounded exponential probe sequence for the active device.

    CPU remains deliberately conservative.  Accelerator probes are isolated by
    ``check_environment.py`` and stop at the first failed or low-headroom
    candidate, so this ceiling is only an upper bound rather than a promised
    allocation.
    """

    kind = str(hardware.get("accelerator_kind") or "cpu")
    if kind not in {"cuda", "mps"}:
        return [1]
    ceiling = (
        CUDA_BATCH_PROBE_CEILING
        if kind == "cuda"
        else MPS_BATCH_PROBE_CEILING
    )
    values: list[int] = []
    value = 1
    while value <= ceiling:
        values.append(value)
        value *= 2
    return values


def freeze_model_batch_probe_results(
    runtime: Mapping[str, Any],
    resource_tuning: Mapping[str, Any],
    probe_results: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze per-model probes plus a conservative scalar compatibility value.

    The Package runtime consumes ``tile_batch_size_by_model``.  The minimum
    verified value remains in ``runtime.tile_batch_size`` for older reporting,
    fallback models, and callers that only understand the scalar field.
    """

    runtime_value = dict(runtime)
    tuning_value = dict(resource_tuning)
    resolved = dict(tuning_value.get("resolved") or {})
    automatic_fields = {
        str(value) for value in tuning_value.get("automatic_fields") or []
    }
    normalized_results: dict[str, dict[str, Any]] = {}
    batch_by_model: dict[str, int] = {}
    for raw_model_id, raw_result in sorted(probe_results.items()):
        model_id = str(raw_model_id)
        result = dict(raw_result)
        normalized_results[model_id] = result
        if not bool(result.get("ok")):
            continue
        try:
            batch_size = int(result.get("safe_batch_size") or 0)
        except (TypeError, ValueError):
            continue
        if batch_size >= 1:
            batch_by_model[model_id] = batch_size

    resolved["tile_batch_size_by_model"] = batch_by_model
    if "tile_batch_size" in automatic_fields and batch_by_model:
        scalar_batch_size = min(batch_by_model.values())
        runtime_value["tile_batch_size"] = scalar_batch_size
        resolved["tile_batch_size"] = scalar_batch_size

    tuning_value["resolved"] = resolved
    tuning_value["model_batch_probe"] = {
        "schema_version": BATCH_PROBE_SCHEMA_VERSION,
        "method": "isolated_resident_model_set_exponential_dummy_forward",
        "status": (
            "completed"
            if batch_by_model
            else "not_applicable"
            if "tile_batch_size" not in automatic_fields
            else "incomplete"
        ),
        "results": normalized_results,
    }
    return runtime_value, tuning_value


def resolve_hardware_tuning(
    runtime: Mapping[str, Any],
    scaling: Mapping[str, Any],
    hardware: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve all ``auto`` performance values once before Run creation.

    The full geometry pool uses one process per physical core when memory permits.
    While a Work Package is active, the scheduler reserves exactly the number of
    CPU threads assigned to that package, so independent geometry processes plus
    the package stay within the physical-core budget.
    """

    runtime_value = dict(runtime)
    scaling_value = dict(scaling)
    physical = max(1, int(hardware.get("physical_cpu_count") or 1))
    total_memory = int(hardware.get("memory_total_bytes") or 0)
    accelerator_kind = str(hardware.get("accelerator_kind") or "cpu")

    memory_reserve = 8 * GIB if accelerator_kind == "mps" else 4 * GIB
    memory_worker_limit = physical
    if total_memory > memory_reserve:
        memory_worker_limit = max(1, int((total_memory - memory_reserve) // GIB))
    automatic_geometry_workers = min(physical, memory_worker_limit)
    automatic_package_threads = (
        min(4, max(2, physical // 4))
        if accelerator_kind in {"cuda", "mps"}
        else max(1, physical // 2)
    )

    tile_batch_size, batch_auto = _configured_or_auto(
        runtime_value.get("tile_batch_size", "auto"),
        _automatic_tile_batch_size(hardware),
    )
    geometry_workers, geometry_auto = _configured_or_auto(
        scaling_value.get("max_cpu_partition_workers", "auto"),
        automatic_geometry_workers,
    )
    geometry_workers = min(geometry_workers, physical)
    tile_io_workers, io_auto = _configured_or_auto(
        scaling_value.get("tile_io_workers", "auto"),
        min(16, max(4, physical)),
    )
    assembly_concurrency, assembly_concurrency_auto = _configured_or_auto(
        scaling_value.get("max_concurrent_assembly", "auto"),
        min(4, physical),
    )
    assembly_concurrency = min(assembly_concurrency, physical)
    assembly_worker_budget = max(1, physical // assembly_concurrency)
    assembly_workers, assembly_auto = _configured_or_auto(
        scaling_value.get("assembly_validation_workers", "auto"),
        min(8, assembly_worker_budget),
    )
    assembly_workers = min(assembly_workers, assembly_worker_budget)
    package_threads = min(automatic_package_threads, physical)
    geometry_with_package = max(1, geometry_workers - package_threads)

    runtime_value["tile_batch_size"] = tile_batch_size
    scaling_value.update(
        {
            "max_cpu_partition_workers": geometry_workers,
            "max_cpu_partition_workers_with_package": geometry_with_package,
            "tile_io_workers": tile_io_workers,
            "max_concurrent_assembly": assembly_concurrency,
            "assembly_validation_workers": assembly_workers,
        }
    )
    resolved = {
        "tile_batch_size": tile_batch_size,
        "max_cpu_partition_workers": geometry_workers,
        "max_cpu_partition_workers_with_package": geometry_with_package,
        "tile_io_workers": tile_io_workers,
        "max_concurrent_assembly": assembly_concurrency,
        "assembly_validation_workers": assembly_workers,
        "unit_process_threads": 1,
        "package_process_threads": package_threads,
        "assembly_process_threads": 1,
    }
    automatic_fields = [
        name
        for name, enabled in (
            ("tile_batch_size", batch_auto),
            ("max_cpu_partition_workers", geometry_auto),
            ("tile_io_workers", io_auto),
            ("max_concurrent_assembly", assembly_concurrency_auto),
            ("assembly_validation_workers", assembly_auto),
        )
        if enabled
    ]
    evidence = {
        "schema_version": TUNING_SCHEMA_VERSION,
        "mode": "maximize_throughput",
        "automatic_fields": automatic_fields,
        "hardware": dict(hardware),
        "resolved": resolved,
    }
    return runtime_value, scaling_value, evidence
