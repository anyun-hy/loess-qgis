"""Composed environment-report behavior without live databases or model assets."""

import loess_runtime.system.environment_deployment as environment_deployment
import loess_runtime.system.environment_report as environment_report
from loess_runtime.system.environment_checks import add_check, overall_status
from loess_runtime.system.environment_deployment import (
    RuntimeProbeInputs,
    empty_effective,
)


def _resolved_resources():
    return {
        "max_cpu_partition_workers": 1,
        "max_cpu_partition_workers_with_package": 1,
        "tile_batch_size": 1,
        "tile_io_workers": 1,
        "max_concurrent_assembly": 1,
        "assembly_validation_workers": 1,
    }


def _tuning():
    return RuntimeProbeInputs(
        torch_module=object(),
        resolved_device="cpu",
        device_ok=True,
        hardware={"accelerator_kind": "cpu"},
    )


def test_minimal_report_composes_named_stages_and_fingerprints_last(
    monkeypatch,
    tmp_path,
):
    stage_order = []
    effective = empty_effective()

    def dependencies(checks, _conda_env):
        stage_order.append("dependencies")
        return {}

    def configuration(checks, _scripts_dir, _asset_base_dir):
        stage_order.append("configuration")
        add_check(checks, "config_yaml", "ready", "config.yaml", "fixture")
        return effective, []

    def tuning(checks, _effective, _dependencies):
        stage_order.append("tuning")
        return _tuning()

    def semantic(checks, _effective, _issues, _tuning_value):
        stage_order.append("semantic")
        add_check(checks, "semantic_model_fixture", "ready", "fixture", "fixture")

    monkeypatch.setattr(environment_report, "append_dependency_checks", dependencies)
    monkeypatch.setattr(
        environment_report,
        "append_runtime_boundary_checks",
        lambda *_args: stage_order.append("boundary"),
    )
    monkeypatch.setattr(
        environment_report,
        "load_deployment_configuration",
        configuration,
    )
    monkeypatch.setattr(environment_report, "append_runtime_tuning", tuning)
    monkeypatch.setattr(environment_report, "append_semantic_model_checks", semantic)
    monkeypatch.setattr(
        environment_report,
        "append_fusion_and_class_checks",
        lambda *_args: stage_order.append("fusion"),
    )
    monkeypatch.setattr(
        environment_report,
        "append_sam_checks",
        lambda *_args: stage_order.append("sam"),
    )
    monkeypatch.setattr(
        environment_report,
        "append_output_directory_check",
        lambda *_args: stage_order.append("output"),
    )

    report = environment_report.build_environment_report(
        scripts_dir=tmp_path,
        asset_base_dir=tmp_path,
        conda_env="qgis",
        output_dir="",
        fingerprint=lambda _scripts_dir: stage_order.append("fingerprint")
        or "sha256:fixture",
    )

    assert stage_order == [
        "boundary",
        "dependencies",
        "configuration",
        "tuning",
        "semantic",
        "fusion",
        "sam",
        "output",
        "fingerprint",
    ]
    assert report["status"] == "ready"
    assert report["config_fingerprint"] == "sha256:fixture"
    assert report["effective"] is effective


def test_invalid_config_keeps_empty_effective_and_config_error(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("schema_version: [", encoding="utf-8")
    checks = []

    effective, issues = environment_deployment.load_deployment_configuration(
        checks,
        tmp_path,
        tmp_path,
    )

    assert issues == []
    assert effective == empty_effective()
    assert checks[0]["id"] == "config_yaml"
    assert checks[0]["status"] == "error"
    assert checks[0]["value"] == str(config_path)
    assert checks[0]["source"] == "config.yaml"
    assert checks[0]["message"].startswith("cannot parse config:")


def test_partial_cpu_model_failure_keeps_verified_batch_frozen(monkeypatch):
    checks = []
    effective = empty_effective()
    effective["semantic_models"] = [
        {"model_id": "good", "artifact_path": "good.pt"},
        {"model_id": "bad", "artifact_path": "bad.pt"},
    ]
    effective["runtime"] = {"tile_batch_size": 1}
    effective["resource_tuning"] = {
        "automatic_fields": ["tile_batch_size"],
        "resolved": _resolved_resources(),
    }
    tuning = _tuning()
    monkeypatch.setattr(
        environment_deployment,
        "verify_torchscript_contract",
        lambda _torch, path, _device: (
            (True, "good contract") if path == "good.pt" else (False, "bad contract")
        ),
    )

    environment_deployment.append_semantic_model_checks(
        checks,
        effective,
        [],
        tuning,
    )

    by_id = {item["id"]: item for item in checks}
    assert by_id["semantic_model_good"]["status"] == "ready"
    assert by_id["semantic_model_bad"]["status"] == "error"
    assert effective["runtime"]["tile_batch_size"] == 1
    assert effective["resource_tuning"]["resolved"]["tile_batch_size_by_model"] == {
        "good": 1
    }
    assert by_id["resource_tuning"]["status"] == "ready"


def test_disabled_sam_is_optional_when_a_semantic_model_is_runnable():
    checks = [
        {
            "id": "config_yaml",
            "status": "ready",
            "value": "config.yaml",
            "source": "fixture",
            "message": "",
            "fix": "",
        },
        {
            "id": "semantic_model_fixture",
            "status": "ready",
            "value": "fixture",
            "source": "fixture",
            "message": "",
            "fix": "",
        },
    ]
    effective = empty_effective()
    effective["sam3"] = {"enabled": False}

    environment_deployment.append_sam_checks(checks, effective, _tuning(), "qgis")

    assert checks[-1]["id"] == "sam3_enabled"
    assert checks[-1]["status"] == "warning"
    assert overall_status(checks) == "warning"
