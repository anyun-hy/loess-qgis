import contextlib
from types import SimpleNamespace

import check_environment
from check_environment import (
    _mps_runtime_requirement,
    add_runtime_boundary_checks,
    overall_status,
    verify_torchscript_contract,
    verify_torchscript_contract_isolated,
)


def test_swin_mps_requires_pytorch_27_but_other_devices_are_unchanged():
    assert _mps_runtime_requirement("upernet_swin_b", "2.5.1")[0] is False
    assert _mps_runtime_requirement("upernet_swin_b", "2.7.0")[0] is True
    assert _mps_runtime_requirement("setr_vit", "2.5.1")[0] is True


def _check(check_id, status, source=""):
    return {"id": check_id, "status": status, "source": source}


def test_optional_fusion_and_sam_errors_do_not_block_runnable_semantic_model():
    checks = [
        _check("config_yaml", "ready"),
        _check("semantic_model_a", "ready"),
        _check("semantic_model_b", "error"),
        _check("fusion_profile_f", "error"),
        _check("sam3_backend", "error"),
    ]
    assert overall_status(checks) == "warning"


def test_missing_sam3_checkpoint_is_optional_when_models_are_runnable():
    checks = [
        _check("config_yaml", "ready"),
        _check("semantic_model_a", "ready"),
        _check("sam3_model_load", "error"),
    ]
    assert overall_status(checks) == "warning"


def test_no_runnable_model_or_core_error_blocks():
    assert overall_status([_check("semantic_model_a", "error")]) == "error"
    assert overall_status([
        _check("semantic_model_a", "ready"),
        _check("dependency_rasterio", "error"),
    ]) == "error"


def test_runtime_boundary_checks_accept_qgis_344_qt5_compatibility(monkeypatch):
    checks = []
    monkeypatch.setenv("LOESS_QGIS_VERSION", "3.44.7-Solothurn")
    monkeypatch.setenv("LOESS_QGIS_PYTHON_VERSION", "3.12.5")
    monkeypatch.setenv("LOESS_PYQT_VERSION", "5.15.10")
    monkeypatch.setenv("LOESS_QT_VERSION", "5.15.13")
    monkeypatch.setenv("LOESS_QGIS_PYTHON_EXECUTABLE", "/opt/qgis3/bin/python3")
    monkeypatch.setattr(
        check_environment.sys,
        "executable",
        "/home/example/anaconda3/envs/qgis/bin/python",
        raising=False,
    )

    add_runtime_boundary_checks(checks, "qgis")

    by_id = {item["id"]: item for item in checks}
    assert by_id["qgis_version"]["status"] == "ready"
    assert by_id["pyqt_version"]["status"] == "ready"
    assert by_id["qt_version"]["status"] == "ready"
    assert "compatibility" in by_id["qgis_version"]["message"].lower()


def test_mps_contract_check_releases_allocator_cache(monkeypatch):
    calls = []
    output = SimpleNamespace(shape=(1, 14, 512, 512), dtype="float32")
    model = lambda _sample: output
    monkeypatch.setattr(
        check_environment,
        "load_torchscript_model",
        lambda _path, _device: (model, {"mode": "mps_frozen_hybrid"}),
    )
    fake_torch = SimpleNamespace(
        float32="float32",
        zeros=lambda *_args, **_kwargs: object(),
        inference_mode=contextlib.nullcontext,
        is_tensor=lambda value: value is output,
        cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        mps=SimpleNamespace(empty_cache=lambda: calls.append("mps")),
    )

    ok, message = verify_torchscript_contract(fake_torch, "formal.pt", "mps")

    assert ok is True
    assert "mps_frozen_hybrid" in message
    assert calls == ["mps"]


def test_isolated_contract_reports_worker_crash(monkeypatch):
    class Result:
        returncode = -11
        stdout = ""
        stderr = "Segmentation fault: 11"

    monkeypatch.setattr(
        check_environment.subprocess,
        "run",
        lambda *_args, **_kwargs: Result(),
    )

    ok, message = verify_torchscript_contract_isolated("formal.pt", "mps")

    assert not ok
    assert "exit=-11" in message
    assert "Segmentation fault: 11" in message
