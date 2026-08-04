from hardware_tuning import GIB, resolve_hardware_tuning


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
            "assembly_validation_workers": "auto",
        },
        _hardware(cores=12, memory_gib=32, kind="mps"),
    )

    assert runtime["tile_batch_size"] == 8
    assert scaling["max_cpu_partition_workers"] == 12
    assert scaling["max_cpu_partition_workers_with_package"] == 9
    assert scaling["tile_io_workers"] == 12
    assert scaling["assembly_validation_workers"] == 8
    assert evidence["resolved"]["package_process_threads"] == 3
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
    assert scaling["assembly_validation_workers"] == 8
    assert evidence["resolved"]["package_process_threads"] == 4
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
    assert scaling["assembly_validation_workers"] == 3
    assert evidence["automatic_fields"] == []
