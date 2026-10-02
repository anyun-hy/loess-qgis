"""Behavioral tests for the isolated TorchScript model probe."""

import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import loess_runtime.system.environment_deployment as environment_deployment
import loess_runtime.system.model_probe as model_probe
import loess_runtime.system.probe_process as probe_process
from loess_runtime.system.model_probe import (
    probe_torchscript_batches,
    probe_torchscript_model_set_batches,
    verify_torchscript_batch_probe_isolated,
    verify_torchscript_contract,
    verify_torchscript_contract_isolated,
    verify_torchscript_model_set_batch_probe_isolated,
)


def test_mps_contract_check_releases_allocator_cache(monkeypatch):
    calls = []
    output = SimpleNamespace(shape=(1, 14, 512, 512), dtype="float32")

    def model(_sample):
        return output

    monkeypatch.setattr(
        model_probe,
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
        probe_process.subprocess,
        "run",
        lambda *_args, **_kwargs: Result(),
    )

    ok, message = verify_torchscript_contract_isolated("formal.pt", "mps")

    assert not ok
    assert "exit=-11" in message
    assert "Segmentation fault: 11" in message


def test_sam_runtime_timeout_keeps_the_report_contract(monkeypatch):
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("sam3-worker", 900)

    monkeypatch.setattr(probe_process.subprocess, "run", timeout)

    ok, message = environment_deployment.verify_sam3_runtime_isolated(
        "sam3.pt",
        "cpu",
    )

    assert ok is False
    assert message == "official SAM3 load timed out after 900s on cpu"


def test_all_isolated_workers_reenter_through_the_package_module(monkeypatch):
    calls = []
    inherited_pythonpath = os.pathsep.join(("/external/one", "/external/two"))
    monkeypatch.setenv("PYTHONPATH", inherited_pythonpath)
    monkeypatch.setenv("LOESS_SELF_CALL_TEST", "preserved")

    class Result:
        returncode = 1
        stdout = ""
        stderr = "worker stopped"

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return Result()

    monkeypatch.setattr(probe_process.subprocess, "run", run)
    verify_torchscript_contract_isolated("formal.pt", "cpu")
    verify_torchscript_batch_probe_isolated("formal.pt", "cpu", [1])
    verify_torchscript_model_set_batch_probe_isolated(
        [{"model_id": "formal", "path": "formal.pt"}],
        "cpu",
        [1],
    )
    environment_deployment.verify_sam3_runtime_isolated("sam3.pt", "cpu")

    commands = [command for command, _kwargs in calls]
    environments = [kwargs["env"] for _command, kwargs in calls]
    assert len(calls) == 4
    assert all(
        command[:3] == [sys.executable, "-m", "loess_runtime.system.check_environment"]
        for command in commands
    )
    assert {command[3] for command in commands} == {
        "--contract-worker",
        "--batch-probe-worker",
        "--batch-probe-set-worker",
        "--sam3-worker",
    }
    assert all(environment == environments[0] for environment in environments)
    python_roots = environments[0]["PYTHONPATH"].split(os.pathsep)
    runtime_root = str(Path(model_probe.__file__).resolve().parents[2])
    labeling_root = str(
        Path(sys.modules["labeling_tool"].__file__).resolve().parent.parent
    )
    assert python_roots[: len(dict.fromkeys((runtime_root, labeling_root)))] == list(
        dict.fromkeys((runtime_root, labeling_root))
    )
    assert python_roots[-2:] == ["/external/one", "/external/two"]
    assert environments[0]["LOESS_SELF_CALL_TEST"] == "preserved"


def test_real_isolated_contract_worker_finds_package_without_parent_pythonpath(
    monkeypatch,
    tmp_path,
):
    missing_model = tmp_path / "missing.pt"
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.chdir(unrelated)

    ok, message = verify_torchscript_contract_isolated(
        missing_model,
        "cpu",
        timeout=20,
    )

    assert ok is False
    assert "TorchScript contract failed on cpu" in message
    assert str(missing_model) in message
    assert "ModuleNotFoundError" not in message


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

    monkeypatch.setattr(model_probe, "load_torchscript_model", load_model)
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
        model_probe,
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


def test_batch_probe_stops_before_a_projected_candidate_breaks_headroom(
    monkeypatch,
):
    class Input:
        def __init__(self, batch_size):
            self.batch_size = batch_size

    class Output:
        dtype = "float32"

        def __init__(self, batch_size):
            self.shape = (batch_size, 14, 512, 512)

    attempted = []
    free_values = iter((12 * 1024**3, 10 * 1024**3))

    def model(sample):
        attempted.append(sample.batch_size)
        return Output(sample.batch_size)

    monkeypatch.setattr(
        model_probe,
        "load_torchscript_model",
        lambda _path, _device: (model, {"mode": "cuda"}),
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
            synchronize=lambda _index=0: None,
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )

    result = probe_torchscript_batches(
        fake_torch,
        "formal.pt",
        "cuda:0",
        [1, 2, 4],
        reserve_bytes=5 * 1024**3,
    )

    assert attempted == [1, 2]
    assert result["safe_batch_size"] == 2
    assert result["max_successful_batch"] == 2
    assert result["stop_reason"] == "safety_projection"
    assert result["probes"][-1]["status"] == "skipped_safety_projection"
    assert result["probes"][-1]["batch_size"] == 4


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
        model_probe,
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

    monkeypatch.setattr(model_probe, "load_torchscript_model", load_model)
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
        item["resident_model_count"] == 2 and item["model_set_complete"] is True
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
        probe_process.subprocess,
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


def test_isolated_model_set_probe_recovers_all_completed_models_after_cleanup_crash(
    monkeypatch,
):
    completed = {
        model_id: {
            "ok": True,
            "safe_batch_size": batch_size,
            "model_set_complete": True,
        }
        for model_id, batch_size in (("a", 4), ("b", 8))
    }

    class Result:
        returncode = -11
        stderr = ""
        stdout = "\n".join(
            [
                '{"event":"model_set_load_completed","model_id":"a"}',
                '{"event":"model_set_load_completed","model_id":"b"}',
                json.dumps(
                    {
                        "event": "model_set_probe_completed",
                        "model_id": "a",
                        "result": completed["a"],
                    }
                ),
                json.dumps(
                    {
                        "event": "model_set_probe_completed",
                        "model_id": "b",
                        "result": completed["b"],
                    }
                ),
            ]
        )

    monkeypatch.setattr(
        probe_process.subprocess,
        "run",
        lambda *_args, **_kwargs: Result(),
    )

    result = verify_torchscript_model_set_batch_probe_isolated(
        [{"model_id": "a", "path": "a.pt"}, {"model_id": "b", "path": "b.pt"}],
        "cuda:0",
        [1, 2, 4, 8],
    )

    assert result["ok"] is True
    assert result["model_set_complete"] is True
    assert result["results"] == completed
    assert result["worker_exit_code"] == -11
    assert "completed" in result["worker_cleanup_warning"]


def test_isolated_batch_probe_preserves_last_safe_result_after_worker_crash(
    monkeypatch,
):
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
        probe_process.subprocess,
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
