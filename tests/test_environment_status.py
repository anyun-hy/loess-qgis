import contextlib
from pathlib import Path
from types import SimpleNamespace

import check_environment
from check_environment import (
    _mps_runtime_requirement,
    add_runtime_boundary_checks,
    overall_status,
    probe_torchscript_batches,
    probe_torchscript_model_set_batches,
    verify_torchscript_contract,
    verify_torchscript_batch_probe_isolated,
    verify_torchscript_model_set_batch_probe_isolated,
    verify_torchscript_contract_isolated,
)


def test_old_pyogrio_is_rejected_before_importing_deprecated_shapely_geos(
    monkeypatch,
):
    imports = []
    monkeypatch.setattr(
        check_environment.importlib.metadata,
        "version",
        lambda _name: "0.10.0",
    )
    monkeypatch.setattr(
        check_environment.importlib,
        "import_module",
        lambda name: imports.append(name),
    )

    module, version, error = check_environment.import_dependency("pyogrio")

    assert module is None
    assert version == "0.10.0"
    assert "pyogrio ==0.13.0" in error
    assert imports == []


def test_wrong_pyarrow_version_is_rejected_before_import(monkeypatch):
    imports = []
    monkeypatch.setattr(
        check_environment.importlib.metadata,
        "version",
        lambda _name: "19.0.1",
    )
    monkeypatch.setattr(
        check_environment.importlib,
        "import_module",
        lambda name: imports.append(name),
    )

    module, version, error = check_environment.import_dependency("pyarrow")

    assert module is None
    assert version == "19.0.1"
    assert "pyarrow ==25.0.1" in error
    assert imports == []


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


def test_batch_probe_loads_once_and_falls_back_after_real_oom(monkeypatch):
    class FakeOutOfMemoryError(RuntimeError):
        pass

    class Input:
        def __init__(self, batch_size):
            self.batch_size = batch_size

    class Output:
        dtype = "float32"

        def __init__(self, batch_size):
            self.shape = (batch_size, 14, 512, 512)

    loads = []
    cache_clears = []
    synchronizes = []

    def load_model(_path, _device):
        loads.append(1)

        def model(sample):
            if sample.batch_size >= 8:
                raise FakeOutOfMemoryError("CUDA out of memory")
            return Output(sample.batch_size)

        return model, {"mode": "cuda"}

    monkeypatch.setattr(check_environment, "load_torchscript_model", load_model)
    fake_torch = SimpleNamespace(
        float32="float32",
        zeros=lambda batch_size, *_args, **_kwargs: Input(batch_size),
        inference_mode=contextlib.nullcontext,
        is_tensor=lambda value: isinstance(value, Output),
        cuda=SimpleNamespace(
            OutOfMemoryError=FakeOutOfMemoryError,
            is_available=lambda: True,
            empty_cache=lambda: cache_clears.append(1),
            mem_get_info=lambda _index=0: (10 * 1024**3, 24 * 1024**3),
            synchronize=lambda index=0: synchronizes.append(index),
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )

    result = probe_torchscript_batches(
        fake_torch,
        "formal.pt",
        "cuda:0",
        [1, 2, 4, 8, 16],
        reserve_bytes=2 * 1024**3,
    )

    assert result["ok"] is True
    assert result["safe_batch_size"] == 4
    assert result["first_failed_batch"] == 8
    assert result["stop_reason"] == "out_of_memory"
    assert len(loads) == 1
    assert cache_clears
    assert synchronizes == [0, 0, 0]


def test_batch_probe_rejects_a_success_that_consumes_safety_headroom(monkeypatch):
    class Input:
        def __init__(self, batch_size):
            self.batch_size = batch_size

    class Output:
        dtype = "float32"

        def __init__(self, batch_size):
            self.shape = (batch_size, 14, 512, 512)

    free_values = iter((10 * 1024**3, 8 * 1024**3, 1 * 1024**3))
    monkeypatch.setattr(
        check_environment,
        "load_torchscript_model",
        lambda _path, _device: (
            lambda sample: Output(sample.batch_size),
            {"mode": "cuda"},
        ),
    )
    fake_torch = SimpleNamespace(
        float32="float32",
        zeros=lambda batch_size, *_args, **_kwargs: Input(batch_size),
        inference_mode=contextlib.nullcontext,
        is_tensor=lambda value: isinstance(value, Output),
        cuda=SimpleNamespace(
            OutOfMemoryError=RuntimeError,
            is_available=lambda: True,
            empty_cache=lambda: None,
            mem_get_info=lambda _index=0: (next(free_values), 24 * 1024**3),
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )

    result = probe_torchscript_batches(
        fake_torch,
        "formal.pt",
        "cuda:0",
        [1, 2, 4, 8],
        reserve_bytes=2 * 1024**3,
    )

    assert result["safe_batch_size"] == 2
    assert result["max_successful_batch"] == 4
    assert result["stop_reason"] == "safety_reserve"


def test_batch_probe_rejects_generic_runtime_error_after_batch_one(monkeypatch):
    class FakeOutOfMemoryError(RuntimeError):
        pass

    class Input:
        def __init__(self, batch_size):
            self.batch_size = batch_size

    class Output:
        dtype = "float32"

        def __init__(self, batch_size):
            self.shape = (batch_size, 14, 512, 512)

    def model(sample):
        if sample.batch_size >= 2:
            raise RuntimeError("corrupt Tile payload")
        return Output(sample.batch_size)

    monkeypatch.setattr(
        check_environment,
        "load_torchscript_model",
        lambda _path, _device: (model, {"mode": "cuda"}),
    )
    fake_torch = SimpleNamespace(
        float32="float32",
        zeros=lambda batch_size, *_args, **_kwargs: Input(batch_size),
        inference_mode=contextlib.nullcontext,
        is_tensor=lambda value: isinstance(value, Output),
        cuda=SimpleNamespace(
            OutOfMemoryError=FakeOutOfMemoryError,
            is_available=lambda: True,
            empty_cache=lambda: None,
            mem_get_info=lambda _index=0: (10 * 1024**3, 24 * 1024**3),
            synchronize=lambda _index=0: None,
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )

    result = probe_torchscript_batches(
        fake_torch,
        "formal.pt",
        "cuda:0",
        [1, 2, 4],
        reserve_bytes=2 * 1024**3,
    )

    assert result["ok"] is False
    assert result["safe_batch_size"] == 0
    assert result["last_verified_batch_size"] == 1
    assert result["stop_reason"] == "runtime_error"


def test_model_set_probe_loads_every_model_before_any_forward(monkeypatch):
    class FakeOutOfMemoryError(RuntimeError):
        pass

    class Input:
        def __init__(self, batch_size):
            self.batch_size = batch_size

    class Output:
        dtype = "float32"

        def __init__(self, batch_size):
            self.shape = (batch_size, 14, 512, 512)

    loaded_ids = []

    def load_model(path, _device):
        model_id = Path(path).name
        loaded_ids.append(model_id)

        def model(sample):
            assert loaded_ids == ["a.pt", "b.pt"]
            return Output(sample.batch_size)

        return model, {"mode": "cuda", "path": model_id}

    monkeypatch.setattr(check_environment, "load_torchscript_model", load_model)
    fake_torch = SimpleNamespace(
        float32="float32",
        zeros=lambda batch_size, *_args, **_kwargs: Input(batch_size),
        inference_mode=contextlib.nullcontext,
        is_tensor=lambda value: isinstance(value, Output),
        cuda=SimpleNamespace(
            OutOfMemoryError=FakeOutOfMemoryError,
            is_available=lambda: True,
            empty_cache=lambda: None,
            mem_get_info=lambda _index=0: (10 * 1024**3, 24 * 1024**3),
            synchronize=lambda _index=0: None,
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )

    result = probe_torchscript_model_set_batches(
        fake_torch,
        [
            {"model_id": "a", "path": "a.pt"},
            {"model_id": "b", "path": "b.pt"},
        ],
        "cuda:0",
        [1, 2],
        reserve_bytes=2 * 1024**3,
    )

    assert result["ok"] is True
    assert result["model_set_complete"] is True
    assert result["resident_model_ids"] == ["a", "b"]
    assert set(result["results"]) == {"a", "b"}
    assert all(
        item["resident_model_count"] == 2
        and item["model_set_complete"] is True
        for item in result["results"].values()
    )


def test_isolated_model_set_probe_rejects_partial_progress_after_worker_crash(
    monkeypatch,
):
    class Result:
        returncode = -9
        stdout = "\n".join(
            (
                '{"event":"model_set_load_completed","model_id":"a"}',
                '{"event":"model_set_load_completed","model_id":"b"}',
                '{"event":"batch_probe_result","model_id":"a",'
                '"batch_size":1,"status":"passed"}',
            )
        )
        stderr = "Killed"

    monkeypatch.setattr(
        check_environment.subprocess,
        "run",
        lambda *_args, **_kwargs: Result(),
    )

    result = verify_torchscript_model_set_batch_probe_isolated(
        [
            {"model_id": "a", "path": "a.pt"},
            {"model_id": "b", "path": "b.pt"},
        ],
        "cuda:0",
        [1, 2],
    )

    assert result["ok"] is False
    assert result["model_set_complete"] is False
    assert all(item["safe_batch_size"] == 0 for item in result["results"].values())


def test_isolated_batch_probe_preserves_last_safe_result_after_worker_crash(monkeypatch):
    class Result:
        returncode = -9
        stdout = "\n".join(
            (
                '{"event":"batch_probe_started","batch_size":1}',
                '{"event":"batch_probe_result","batch_size":1,"status":"passed"}',
                '{"event":"batch_probe_started","batch_size":2}',
            )
        )
        stderr = "Killed"

    monkeypatch.setattr(
        check_environment.subprocess,
        "run",
        lambda *_args, **_kwargs: Result(),
    )

    result = verify_torchscript_batch_probe_isolated(
        "formal.pt",
        "cuda:0",
        [1, 2, 4],
        reserve_bytes=2 * 1024**3,
    )

    assert result["ok"] is True
    assert result["safe_batch_size"] == 1
    assert result["first_failed_batch"] == 2
    assert result["stop_reason"] == "worker_crash"
