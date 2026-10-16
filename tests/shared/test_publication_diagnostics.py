"""Failure diagnostics must preserve publication behavior and task isolation."""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from types import SimpleNamespace

import pytest

from labeling_tool.shared.state import publication_diagnostics as diagnostics


class LockFailure(RuntimeError):
    pgcode = "55P03"


def _job(identifier=12):
    return {
        "run_id": "test-run",
        "stream_id": "fusion:test",
        "job_id": identifier,
        "unit_id": f"fragmentation_v33_partition:partition-{identifier}",
        "attempt": 2,
        "lease_token": "secret-lease-token",
        "dsn": "password=secret-password",
    }


def _capture(job, phase, error):
    with pytest.raises(type(error)) as caught:
        with diagnostics.v33_job_diagnostics(job):
            diagnostics.publication_phase(phase, job_id=job["job_id"])
            raise error
    assert caught.value is error
    return diagnostics.publication_failure_payload(error)


def test_lock_failure_keeps_exception_and_excludes_secret_job_fields():
    error = LockFailure("original failure")
    payload = _capture(_job(), "input_lock", error)
    assert payload["status"] == "failed"
    assert payload["error"] == "original failure"
    detail = payload["diagnostics"]
    assert detail["job_id"] == 12
    assert detail["partition_id"] == "partition-12"
    assert detail["attempt"] == 2
    assert detail["phase"] == "input_lock"
    assert detail["sqlstate"] == "55P03"
    assert detail["exception_type"] == "LockFailure"
    assert detail["input_lock_completed"] is False
    assert detail["input_lock_elapsed_ms"] is None
    assert detail["phase_to_catch_ms"]["input_lock"] >= 0
    encoded = json.dumps(detail)
    assert "secret" not in encoded
    assert "lease_token" not in encoded
    assert "dsn" not in encoded


def test_completed_lock_clock_is_frozen_before_failure_bookkeeping(monkeypatch):
    ticks = iter([10.0, 10.02, 10.12, 10.13, 10.15])
    monkeypatch.setattr(
        diagnostics, "time", SimpleNamespace(perf_counter=lambda: next(ticks))
    )
    error = RuntimeError("publish failed")
    with pytest.raises(RuntimeError):
        with diagnostics.v33_job_diagnostics(_job()):
            diagnostics.publication_phase("input_lock")
            diagnostics.publication_phase("staged_publish")
            diagnostics.capture_publication_failure(error)
            diagnostics.publication_phase("failure_bookkeeping")
            raise error
    detail = diagnostics.publication_failure_payload(error)["diagnostics"]
    assert detail["phase"] == "staged_publish"
    assert detail["input_lock_completed"] is True
    assert detail["input_lock_elapsed_ms"] == 100.0
    assert detail["phase_to_catch_ms"]["staged_publish"] == 10.0
    assert detail["elapsed_ms"] == 130.0
    assert "failure_bookkeeping" not in detail["phase_to_catch_ms"]


def test_initial_lock_success_is_distinct_from_release_lock_failure():
    error = LockFailure("release relock failed")
    with pytest.raises(LockFailure):
        with diagnostics.v33_job_diagnostics(_job()):
            diagnostics.publication_phase("input_lock")
            diagnostics.publication_phase("publication")
            diagnostics.publication_phase("release_input_lock")
            raise error
    detail = diagnostics.publication_failure_payload(error)["diagnostics"]
    assert detail["input_lock_completed"] is True
    assert detail["input_lock_elapsed_ms"] >= 0
    assert detail["release_input_lock_elapsed_ms"] is None
    assert detail["phase"] == "release_input_lock"


def test_preparation_failure_does_not_claim_inputs_were_locked():
    detail = _capture(_job(), "prepare", RuntimeError("bad input"))["diagnostics"]
    assert detail["input_lock_completed"] is False
    assert detail["input_lock_elapsed_ms"] is None


def test_parallel_jobs_and_finished_contexts_do_not_share_diagnostics():
    barrier = threading.Barrier(2, timeout=5)

    def worker(identifier):
        error = RuntimeError(f"job-{identifier}")
        with pytest.raises(RuntimeError):
            with diagnostics.v33_job_diagnostics(_job(identifier)):
                diagnostics.publication_phase(f"phase-{identifier}")
                barrier.wait()
                raise error
        return diagnostics.publication_failure_payload(error)["diagnostics"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker, identifier) for identifier in (12, 13)]
        details = [future.result(timeout=6) for future in futures]
    assert {(x["job_id"], x["phase"]) for x in details} == {
        (12, "phase-12"),
        (13, "phase-13"),
    }
    detail = diagnostics.publication_failure_payload(RuntimeError("later"))[
        "diagnostics"
    ]
    assert detail["phase"] == "preflight"
    assert "job_id" not in detail


def test_copied_context_and_wrong_job_phase_cannot_mutate_parent_trace():
    parent_error = RuntimeError("parent")
    child_error = RuntimeError("child")

    def child():
        diagnostics.publication_phase("child")
        diagnostics.capture_publication_failure(child_error)

    with pytest.raises(RuntimeError):
        with diagnostics.v33_job_diagnostics(_job()):
            diagnostics.publication_phase("parent")
            copy_context().run(child)
            diagnostics.publication_phase("wrong-job", job_id=999)
            raise parent_error
    assert (
        diagnostics.publication_failure_payload(child_error)["diagnostics"]["phase"]
        == "child"
    )
    assert (
        diagnostics.publication_failure_payload(parent_error)["diagnostics"]["phase"]
        == "parent"
    )


def test_diagnostic_capture_failure_never_masks_original_exception(monkeypatch):
    def unavailable(*_args):
        raise ValueError("diagnostic capture unavailable")

    monkeypatch.setattr(diagnostics.PublicationDiagnostics, "capture", unavailable)
    error = LockFailure("real failure")
    payload = _capture(_job(), "input_lock", error)
    assert payload["error"] == "real failure"
    assert payload["diagnostics"]["sqlstate"] == "55P03"


def test_invalid_context_and_corrupt_snapshot_still_emit_legacy_error():
    error = RuntimeError("real failure")
    with pytest.raises(RuntimeError) as caught:
        with diagnostics.v33_job_diagnostics({}):
            raise error
    assert caught.value is error
    error._v33_diagnostics = {"invalid": object()}
    payload = diagnostics.publication_failure_payload(error)
    assert payload["status"] == "failed"
    assert payload["error"] == "real failure"
    assert payload["diagnostics"]["phase"] == "preflight"


def test_wrapped_sqlstate_and_cause_cycle_are_bounded():
    cause = LockFailure("database")
    error = RuntimeError("wrapper")
    error.__cause__ = cause
    assert (
        _capture(_job(), "commit_pending", error)["diagnostics"]["sqlstate"] == "55P03"
    )
    loop = RuntimeError("cycle")
    loop.__cause__ = loop
    assert (
        diagnostics.publication_failure_payload(loop)["diagnostics"]["sqlstate"] is None
    )


@pytest.mark.parametrize("success", [False, True])
def test_cli_preserves_success_output_and_emits_structured_failure(
    monkeypatch, capsys, success
):
    from loess_runtime.geometry import fragmentation_v33_work_package as worker

    def run(*_args, **_kwargs):
        if success:
            return {"status": "ready"}
        with diagnostics.v33_job_diagnostics(_job()):
            diagnostics.publication_phase("staged_publish")
            raise LockFailure("database failure")

    monkeypatch.setattr(sys, "argv", ["worker", "--run-spec", "unused.json"])
    monkeypatch.setattr(worker, "run_worker", run)
    assert worker.main() == (0 if success else 2)
    output = capsys.readouterr()
    if success:
        assert json.loads(output.out) == {"status": "ready"}
        assert output.err == ""
    else:
        assert output.out == ""
        payload = json.loads(output.err)
        assert payload["status"] == "failed"
        assert payload["error"] == "database failure"
        assert payload["diagnostics"]["phase"] == "staged_publish"
        assert payload["diagnostics"]["sqlstate"] == "55P03"
