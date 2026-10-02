from labeling_tool.shared.contracts.monitor_contract import (
    ASSEMBLY_PHASES,
    effective_device_text,
    progress_text,
)


def test_effective_device_labels_are_run_specific_and_cross_platform():
    assert effective_device_text(
        {
            "runtime": {"effective_device": "cuda"},
            "resource_tuning": {
                "hardware": {"accelerator": {"name": "NVIDIA RTX 3090"}}
            },
        }
    ) == ("CUDA", "NVIDIA RTX 3090")
    assert effective_device_text(
        {"runtime": {"effective_device": "mps"}}
    ) == ("MPS", "Apple MPS")
    assert effective_device_text(
        {"runtime": {"effective_device": "cpu"}}
    ) == ("CPU", "CPU")
    assert effective_device_text({}) == ("未知后端", "设备信息缺失")


def test_progress_and_assembly_contract_do_not_invent_unknown_values():
    assert len(ASSEMBLY_PHASES) == 10
    assert progress_text(None, None) == "—"
    assert progress_text(0, 0) == "0"
    assert progress_text(12, 48, unit="包") == "12/48包"
