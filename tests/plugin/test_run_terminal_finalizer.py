from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from labeling_tool.runs import run_terminal_finalizer as finalizer


def _context(tmp_path: Path, *, success: bool = True):
    spec_path = tmp_path / "run_spec.json"
    spec_path.write_text("{}", encoding="utf-8")
    return finalizer.RunTerminalContext(
        spec={
            "run_id": "run-1",
            "run_dir": str(tmp_path / "run"),
            "streams": [
                {
                    "stream_id": "model:a",
                    "kind": "model",
                    "model_id": "a",
                }
            ],
        },
        spec_path=str(spec_path),
        success=success,
        stopped=False,
        error="" if success else "failed",
        started_at=100.0,
        manual_package_reset={},
        execution_id="execution-1",
        monitor_history_incomplete=False,
    )


def test_elapsed_and_phase_time_are_sampled_after_output_hashes(tmp_path, monkeypatch):
    context = _context(tmp_path)
    now = [100.0]
    phase_observations = []

    def advance_hash(_path):
        now[0] += 1.0
        return "digest"

    monkeypatch.setattr(finalizer, "artifact_sha256", advance_hash)
    monkeypatch.setattr(finalizer, "sha256_file", advance_hash)

    result = finalizer.build_run_result(
        context,
        clock=lambda: now[0],
        phase_summary=lambda observed_at: (
            phase_observations.append(observed_at) or {"observed_at": observed_at}
        ),
    )

    # Six stream artifacts plus run_spec are hashed before the terminal clock.
    assert result["elapsed_sec"] == 7.0
    assert phase_observations == [107.0]
    assert result["phase_timing"] == {"observed_at": 107.0}


def test_scale_acceptance_report_identity_is_included_in_terminal_result(tmp_path):
    context = _context(tmp_path, success=False)
    report_path = (
        Path(context.spec["run_dir"]) / "logs" / "scale_acceptance_report.json"
    )
    report_path.parent.mkdir(parents=True)
    observation = {"status": "observed", "total_bytes": 4096}
    report_path.write_text(
        json.dumps({"storage": {"final_artifact_size_observation": observation}}),
        encoding="utf-8",
    )

    result = finalizer.build_run_result(
        context,
        clock=lambda: 101.0,
        phase_summary=lambda _now: {},
    )

    assert result["scale_acceptance_report"] == str(report_path)
    assert (
        result["scale_acceptance_report_sha256"]
        == hashlib.sha256(report_path.read_bytes()).hexdigest()
    )
    assert result["final_artifact_size_observation"] == observation


def test_terminal_queries_and_writes_keep_the_existing_order(tmp_path, monkeypatch):
    context = _context(tmp_path)
    events = []

    class Jobs:
        def job_counts(self, run_id, *, job_type=""):
            events.append(("job-counts", run_id, job_type))
            return {"ready": 1}

    class Artifacts:
        def artifact_cleanup_summary(self, run_id):
            events.append(("artifact-summary", run_id))
            return {"released": 1}

    class History:
        def finish_execution(self, run_id, execution_id, **kwargs):
            events.append(("history", run_id, execution_id, kwargs))
            return True

    def set_run_status(run_id, status, *, expected):
        events.append(("status", run_id, status, expected))
        return True

    monkeypatch.setattr(
        finalizer,
        "atomic_write_json",
        lambda path, _value: events.append(("write", Path(path).name)),
    )
    result = {
        "status": "ready",
        "elapsed_sec": 2.0,
        "phase_timing": {},
        "deployment_identity": {},
        "ready_streams": [{"stream_id": "model:a"}],
    }

    value = finalizer.finalize_run_outputs(
        context,
        result,
        jobs=Jobs(),
        artifacts=Artifacts(),
        history=History(),
        set_run_status=set_run_status,
        emit_history_error=lambda error: events.append(("history-error", error)),
    )

    assert value["monitor_history_complete"] is True
    assert [event[0] for event in events] == [
        "status",
        "job-counts",
        "history",
        "artifact-summary",
        "write",
        "write",
        "write",
    ]
    assert [event[1] for event in events if event[0] == "write"] == [
        "run_report.json",
        "failures.json",
        "run_manifest.json",
    ]
    assert value["terminal_published"] is True


def test_history_failure_is_logged_before_incomplete_files(tmp_path, monkeypatch):
    context = _context(tmp_path, success=False)
    events = []

    class Jobs:
        def job_counts(self, _run_id, *, job_type=""):
            return {"failed": 1}

    class Artifacts:
        def artifact_cleanup_summary(self, _run_id):
            return {}

    class History:
        def finish_execution(self, *_args, **_kwargs):
            raise RuntimeError("history unavailable")

    written = []

    def write(path, value):
        events.append(("write", Path(path).name))
        written.append((Path(path).name, dict(value)))

    monkeypatch.setattr(finalizer, "atomic_write_json", write)
    result = {
        "status": "failed",
        "elapsed_sec": 2.0,
        "phase_timing": {},
        "deployment_identity": {},
        "ready_streams": [],
    }

    value = finalizer.finalize_run_outputs(
        context,
        result,
        jobs=Jobs(),
        artifacts=Artifacts(),
        history=History(),
        set_run_status=lambda *_args, **_kwargs: True,
        emit_history_error=lambda error: events.append(("history-error", error)),
    )

    assert value["monitor_history_complete"] is False
    assert events[0] == ("history-error", "history unavailable")
    assert [name for name, _value in written] == [
        "run_report.json",
        "failures.json",
        "run_manifest.json",
    ]
    assert written[0][1]["monitor_history_complete"] is False
    assert written[2][1]["monitor_history_complete"] is False


@pytest.mark.parametrize(
    ("success", "stopped", "expected_status"),
    [
        (True, False, "ready"),
        (False, False, "failed"),
        (False, True, "stopped"),
    ],
)
def test_terminal_cas_rejection_publishes_no_history_or_files(
    tmp_path, monkeypatch, success, stopped, expected_status
):
    context = _context(tmp_path, success=success)
    context = finalizer.RunTerminalContext(**{**context.__dict__, "stopped": stopped})
    events = []

    class Jobs:
        def job_counts(self, *_args, **_kwargs):
            events.append("job-counts")
            return {}

    class Artifacts:
        def artifact_cleanup_summary(self, *_args, **_kwargs):
            events.append("artifacts")
            return {}

    class History:
        def finish_execution(self, *_args, **_kwargs):
            events.append("history")
            return True

    def reject(_run_id, status, *, expected):
        events.append(("status", status, expected))
        return False

    monkeypatch.setattr(
        finalizer,
        "atomic_write_json",
        lambda *_args, **_kwargs: events.append("write"),
    )

    with pytest.raises(
        finalizer.RunTerminalStateConflictError,
        match="terminal state changed",
    ):
        finalizer.finalize_run_outputs(
            context,
            {
                "status": expected_status,
                "elapsed_sec": 1.0,
                "phase_timing": {},
                "deployment_identity": {},
                "ready_streams": [],
            },
            jobs=Jobs(),
            artifacts=Artifacts(),
            history=History(),
            set_run_status=reject,
            emit_history_error=lambda error: events.append(("history-error", error)),
        )

    assert events == [("status", expected_status, ("running", "raster_ready"))]


def test_file_failure_never_writes_completion_manifest(tmp_path, monkeypatch):
    context = _context(tmp_path)
    writes = []

    class Jobs:
        def job_counts(self, *_args, **_kwargs):
            return {"ready": 1}

    class Artifacts:
        def artifact_cleanup_summary(self, *_args, **_kwargs):
            return {}

    class History:
        def finish_execution(self, *_args, **_kwargs):
            return True

    def fail_second(path, _value):
        name = Path(path).name
        writes.append(name)
        if name == "failures.json":
            raise OSError("disk unavailable")

    monkeypatch.setattr(finalizer, "atomic_write_json", fail_second)

    with pytest.raises(OSError, match="disk unavailable"):
        finalizer.finalize_run_outputs(
            context,
            {
                "status": "ready",
                "elapsed_sec": 1.0,
                "phase_timing": {},
                "deployment_identity": {},
                "ready_streams": [],
            },
            jobs=Jobs(),
            artifacts=Artifacts(),
            history=History(),
            set_run_status=lambda *_args, **_kwargs: True,
            emit_history_error=lambda _error: None,
        )

    assert writes == ["run_report.json", "failures.json"]
    assert "run_manifest.json" not in writes
