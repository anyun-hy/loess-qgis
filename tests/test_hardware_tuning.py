from hardware_tuning import (
    GIB,
    batch_probe_safety_reserve_bytes,
    freeze_model_batch_probe_results,
    model_batch_probe_candidates,
    resolve_hardware_tuning,
)


def _hardware(*, cores, memory_gib, kind, accelerator_gib=0):
    return {
        "platform": "macos" if kind == "mps" else "ubuntu",
        "logical_cpu_count": cores,
        "physical_cpu_count": cores,
        "memory_total_bytes": memory_gib * GIB,
        "memory_available_bytes": memory_gib * GIB,
        "accelerator_kind": kind,
        "accelerator_name": kind,
        "accelerator_memory_total_bytes": accelerator_gib * GIB,
        "accelerator_memory_free_bytes": accelerator_gib * GIB,
    }


def test_m2_max_profile_uses_all_cores_without_thread_oversubscription():
    runtime, scaling, evidence = resolve_hardware_tuning(
        {"tile_batch_size": "auto"},
        {
            "tile_io_workers": "auto",
            "max_cpu_partition_workers": "auto",
            "max_concurrent_assembly": "auto",
            "assembly_validation_workers": "auto",
        },
        _hardware(cores=12, memory_gib=32, kind="mps"),
    )

    assert runtime["tile_batch_size"] == 8
    assert scaling["max_cpu_partition_workers"] == 12
    assert scaling["max_cpu_partition_workers_with_package"] == 9
    assert scaling["tile_io_workers"] == 12
    assert scaling["max_concurrent_assembly"] == 4
    assert scaling["assembly_validation_workers"] == 3
    assert evidence["resolved"]["package_process_threads"] == 3
    assert evidence["resolved"]["fragmentation_v33_process_threads"] == 3
    assert (
        scaling["max_cpu_partition_workers_with_package"]
        + evidence["resolved"]["package_process_threads"]
        == 12
    )


def test_rtx3090_profile_uses_sixteen_tile_batch_and_twenty_core_budget():
    runtime, scaling, evidence = resolve_hardware_tuning(
        {"tile_batch_size": "auto"},
        {
            "tile_io_workers": "auto",
            "max_cpu_partition_workers": "auto",
            "max_concurrent_assembly": "auto",
            "assembly_validation_workers": "auto",
        },
        _hardware(
            cores=20,
            memory_gib=100,
            kind="cuda",
            accelerator_gib=24,
        ),
    )

    assert runtime["tile_batch_size"] == 16
    assert scaling["max_cpu_partition_workers"] == 20
    assert scaling["max_cpu_partition_workers_with_package"] == 16
    assert scaling["tile_io_workers"] == 16
    assert scaling["assembly_validation_workers"] == 5
    assert scaling["max_concurrent_assembly"] == 4
    assert evidence["resolved"]["package_process_threads"] == 4
    assert evidence["resolved"]["fragmentation_v33_process_threads"] == 4
    assert evidence["schema_version"] == 3
    assert evidence["resolved"]["memory_admission"] == {
        "schema_version": 1,
        "mode": "adaptive_psi_aimd_v1",
        "sample_interval_sec": 1.0,
        "stable_growth_sec": 5.0,
        "initial_geometry_slots_with_package": 4,
        "initial_geometry_slots_without_package": 8,
        "default_worker_peak_bytes": int(2.5 * GIB),
        "minimum_available_reserve_bytes": 16 * GIB,
        "available_reserve_ratio": 0.20,
        "pressure_some_avg10": 8.0,
        "pressure_full_avg10": 1.0,
        "severe_some_avg10": 25.0,
        "severe_full_avg10": 5.0,
        "stable_some_avg10": 2.0,
        "stable_full_avg10": 0.2,
        "low_available_ratio": 0.20,
        "severe_available_ratio": 0.10,
        "high_swap_used_ratio": 0.65,
        "severe_swap_used_ratio": 0.85,
    }
    assert (
        scaling["max_cpu_partition_workers_with_package"]
        + evidence["resolved"]["package_process_threads"]
        == 20
    )


def test_explicit_performance_values_are_not_replaced_by_auto_defaults():
    runtime, scaling, evidence = resolve_hardware_tuning(
        {"tile_batch_size": 1},
        {
            "tile_io_workers": 5,
            "max_cpu_partition_workers": 6,
            "max_concurrent_assembly": 2,
            "assembly_validation_workers": 3,
        },
        _hardware(
            cores=20,
            memory_gib=100,
            kind="cuda",
            accelerator_gib=24,
        ),
    )

    assert runtime["tile_batch_size"] == 1
    assert scaling["tile_io_workers"] == 5
    assert scaling["max_cpu_partition_workers"] == 6
    assert scaling["max_concurrent_assembly"] == 2
    assert scaling["assembly_validation_workers"] == 3
    assert evidence["automatic_fields"] == []


def test_accelerator_probe_candidates_are_exponential_but_cpu_stays_at_one():
    assert model_batch_probe_candidates(
        _hardware(cores=20, memory_gib=100, kind="cuda", accelerator_gib=24)
    ) == [1, 2, 4, 8, 16, 32, 64, 128]
    assert model_batch_probe_candidates(
        _hardware(cores=12, memory_gib=32, kind="mps")
    ) == [1, 2, 4, 8, 16, 32, 64]
    assert model_batch_probe_candidates(
        _hardware(cores=8, memory_gib=32, kind="cpu")
    ) == [1]


def test_environment_probe_can_stop_at_the_production_starting_batch():
    hardware = _hardware(
        cores=20,
        memory_gib=100,
        kind="cuda",
        accelerator_gib=24,
    )

    assert model_batch_probe_candidates(
        hardware,
        maximum_batch_size=16,
    ) == [1, 2, 4, 8, 16]
    assert model_batch_probe_candidates(
        hardware,
        maximum_batch_size=12,
    ) == [1, 2, 4, 8, 12]


def test_cuda_batch_probe_keeps_twenty_percent_or_four_gib_headroom():
    hardware = _hardware(
        cores=20,
        memory_gib=100,
        kind="cuda",
        accelerator_gib=24,
    )
    assert batch_probe_safety_reserve_bytes(hardware) == int(24 * GIB * 0.20)


def test_per_model_probe_mapping_is_frozen_and_scalar_uses_safe_minimum():
    runtime, _scaling, evidence = resolve_hardware_tuning(
        {"tile_batch_size": "auto"},
        {},
        _hardware(cores=20, memory_gib=100, kind="cuda", accelerator_gib=24),
    )

    runtime, evidence = freeze_model_batch_probe_results(
        runtime,
        evidence,
        {
            "swin": {"ok": True, "safe_batch_size": 32},
            "setr": {"ok": True, "safe_batch_size": 16},
            "broken": {"ok": False, "safe_batch_size": 0},
        },
    )

    assert runtime["tile_batch_size"] == 16
    assert evidence["resolved"]["tile_batch_size"] == 16
    assert evidence["resolved"]["tile_batch_size_by_model"] == {
        "setr": 16,
        "swin": 32,
    }
    assert evidence["model_batch_probe"]["status"] == "completed"
    assert "broken" in evidence["model_batch_probe"]["results"]


def test_probe_mapping_does_not_override_an_explicit_scalar_batch():
    runtime, _scaling, evidence = resolve_hardware_tuning(
        {"tile_batch_size": 5},
        {},
        _hardware(cores=20, memory_gib=100, kind="cuda", accelerator_gib=24),
    )

    runtime, evidence = freeze_model_batch_probe_results(
        runtime,
        evidence,
        {"swin": {"ok": True, "safe_batch_size": 32}},
    )

    assert runtime["tile_batch_size"] == 5
    assert evidence["resolved"]["tile_batch_size"] == 5
    assert evidence["resolved"]["tile_batch_size_by_model"] == {"swin": 32}
    assert evidence["model_batch_probe"]["status"] == "completed"
