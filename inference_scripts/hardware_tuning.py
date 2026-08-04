"""Resolve one bounded throughput profile from the active inference hardware."""

from __future__ import annotations

import os
import platform
from typing import Any, Mapping


GIB = 1024**3
TUNING_SCHEMA_VERSION = 1


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
    assembly_workers, assembly_auto = _configured_or_auto(
        scaling_value.get("assembly_validation_workers", "auto"),
        min(8, physical),
    )
    package_threads = min(automatic_package_threads, physical)
    geometry_with_package = max(1, geometry_workers - package_threads)

    runtime_value["tile_batch_size"] = tile_batch_size
    scaling_value.update(
        {
            "max_cpu_partition_workers": geometry_workers,
            "max_cpu_partition_workers_with_package": geometry_with_package,
            "tile_io_workers": tile_io_workers,
            "assembly_validation_workers": assembly_workers,
        }
    )
    resolved = {
        "tile_batch_size": tile_batch_size,
        "max_cpu_partition_workers": geometry_workers,
        "max_cpu_partition_workers_with_package": geometry_with_package,
        "tile_io_workers": tile_io_workers,
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
