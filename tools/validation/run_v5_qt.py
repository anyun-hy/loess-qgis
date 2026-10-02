#!/usr/bin/env python3
"""Drive one prepared V5 Run through the installed plugin's threaded runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RunnerError(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one prepared V5 Run with the installed QGIS plugin runner"
    )
    parser.add_argument("--plugin-parent", required=True)
    parser.add_argument("--scripts-dir", required=True)
    parser.add_argument("--run-spec", required=True)
    parser.add_argument("--state-dsn", required=True)
    parser.add_argument("--state-schema", required=True)
    parser.add_argument("--expected-source-bundle-sha256", required=True)
    parser.add_argument(
        "--action",
        choices=("start", "resume", "retry-failed"),
        default="start",
        help="production runner entry point to invoke (default: start)",
    )
    parser.add_argument(
        "--stop-after-work-package-ready",
        action="store_true",
        help=(
            "stop once after the production state store first reports a ready "
            "work_package Job"
        ),
    )
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument("--stop-grace-seconds", type=int, default=300)
    return parser


def _under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _bootstrap(plugin_parent: Path, scripts_dir: Path) -> dict[str, str]:
    plugin_root = plugin_parent / "labeling_tool"
    runtime_root = scripts_dir / "loess_runtime"
    if not plugin_root.is_dir() or plugin_root.is_symlink():
        raise RunnerError(f"deployed plugin is missing: {plugin_root}")
    if not runtime_root.is_dir() or runtime_root.is_symlink():
        raise RunnerError(f"deployed inference runtime is missing: {runtime_root}")
    sys.path.insert(0, str(scripts_dir))
    sys.path.insert(0, str(plugin_parent))
    import labeling_tool
    import loess_runtime

    plugin_origin = Path(labeling_tool.__file__).resolve()
    runtime_origin = Path(loess_runtime.__file__).resolve()
    if not _under(plugin_origin, plugin_root.resolve()):
        raise RunnerError(f"labeling_tool import escaped --plugin-parent: {plugin_origin}")
    if not _under(runtime_origin, runtime_root.resolve()):
        raise RunnerError(f"loess_runtime import escaped --scripts-dir: {runtime_origin}")
    return {
        "plugin_origin": str(plugin_origin),
        "runtime_origin": str(runtime_origin),
    }


def _content_sha256(spec: Mapping[str, Any]) -> str:
    value = dict(spec)
    value.pop("run_spec_content_sha256", None)
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_spec(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
        raise RunnerError(f"Run Spec is missing, too large, or a symlink: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise RunnerError(f"Run Spec is not valid JSON: {error}") from error
    if not isinstance(value, dict) or value.get("schema_version") != 2:
        raise RunnerError("Run Spec must be a Schema 2 JSON object")
    claimed = str(value.get("run_spec_content_sha256") or "")
    if not SHA256_RE.fullmatch(claimed) or _content_sha256(value) != claimed:
        raise RunnerError("Run Spec content SHA256 is invalid")
    return value


def _failure(error: Exception) -> int:
    print(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "v5_real_validation_runner",
                "status": "error",
                "success": False,
                "error_type": type(error).__name__,
                "message": str(error),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 2


def _stop_snapshot(
    database: Any,
    run_id: str,
    observed_work_package_counts: Mapping[str, Any],
) -> dict[str, Any]:
    """Reduce one production control-plane snapshot to bounded stop evidence."""

    snapshot = database.acceptance_read.snapshot(run_id)
    artifact_status_counts: dict[str, int] = {}
    artifact_kind_status_counts: dict[str, int] = {}
    artifact_rows = list(snapshot.get("artifact_rows") or ())
    for row in artifact_rows:
        status = str(row.get("status") or "<missing>")
        kind = str(row.get("kind") or "<missing>")
        artifact_status_counts[status] = artifact_status_counts.get(status, 0) + 1
        key = f"{kind}:{status}"
        artifact_kind_status_counts[key] = artifact_kind_status_counts.get(key, 0) + 1
    return {
        "jobs": {
            "all": dict(snapshot.get("job_counts") or {}),
            "by_type": dict(snapshot.get("job_type_counts") or {}),
            "work_package_observed": dict(observed_work_package_counts),
        },
        "work_packages": dict(snapshot.get("package_counts") or {}),
        "artifacts": {
            "total": len(artifact_rows),
            "by_status": artifact_status_counts,
            "by_kind_status": artifact_kind_status_counts,
        },
    }


def _invoke_action(runner: Any, action: str, spec_path: Path) -> None:
    """Invoke exactly one public production runner entry point."""

    if action == "start":
        runner.run_from_spec(str(spec_path))
    elif action == "resume":
        runner.resume(str(spec_path))
    elif action == "retry-failed":
        runner.retry_failed(str(spec_path))
    else:  # argparse enforces this for CLI calls; retain a fail-closed API guard.
        raise RunnerError(f"unsupported runner action: {action}")


def run(args: argparse.Namespace) -> int:
    if not 1 <= args.timeout_seconds <= 7 * 24 * 60 * 60:
        raise RunnerError("--timeout-seconds must be between 1 and 604800")
    if not 1 <= args.stop_grace_seconds <= 3600:
        raise RunnerError("--stop-grace-seconds must be between 1 and 3600")
    plugin_parent = Path(args.plugin_parent).expanduser().resolve()
    scripts_dir = Path(args.scripts_dir).expanduser().resolve()
    imports = _bootstrap(plugin_parent, scripts_dir)

    from labeling_tool.runs.deployment_contract import verify_project_runtime
    from labeling_tool.runs.recovery_contract import validate_recovery_run
    from labeling_tool.runs.v5_async_runner import ThreadedV5AsyncInferenceRunner
    from labeling_tool.qgis_support.qt_lifecycle import retire_after
    from labeling_tool.shared.state.postgres_state import (
        DEFAULT_POSTGRES_SCHEMA,
        is_postgres_location,
        validate_schema,
    )
    from qgis.PyQt.QtCore import QCoreApplication, QTimer

    deployment = verify_project_runtime(
        scripts_dir, plugin_root=plugin_parent / "labeling_tool"
    )
    if deployment.get("status") != "ready":
        raise RunnerError(
            "deployed plugin/project contract failed: "
            + str(deployment.get("message") or "unknown error")
        )
    expected_bundle = str(args.expected_source_bundle_sha256).strip().lower()
    if not SHA256_RE.fullmatch(expected_bundle):
        raise RunnerError("--expected-source-bundle-sha256 must be lowercase SHA256")
    dsn = str(args.state_dsn).strip()
    if not is_postgres_location(dsn):
        raise RunnerError("--state-dsn must be an explicit PostgreSQL DSN")
    schema = validate_schema(args.state_schema)
    if schema == DEFAULT_POSTGRES_SCHEMA:
        raise RunnerError("validation Run must not use the default production schema")
    os.environ["LOESS_STATE_DB_DSN"] = dsn
    os.environ["LOESS_STATE_DB_SCHEMA"] = schema

    spec_path = Path(args.run_spec).expanduser().resolve()
    preflight_spec = _load_spec(spec_path)
    if preflight_spec.get("state_backend") != "postgresql":
        raise RunnerError("Run Spec state_backend must equal postgresql")
    if str(preflight_spec.get("state_db") or "") != dsn:
        raise RunnerError("Run Spec state DSN does not match the explicitly supplied isolated DSN")
    if str(preflight_spec.get("state_schema") or "") != schema:
        raise RunnerError("Run Spec state schema does not match the explicitly supplied isolated schema")
    device = str((preflight_spec.get("runtime") or {}).get("effective_device") or "")
    if re.fullmatch(r"cuda(?::\d+)?", device) is None:
        raise RunnerError(f"Run Spec runtime device must be CUDA, got {device or '<missing>'}")
    frozen_bundle = str(
        (preflight_spec.get("deployment_identity") or {}).get("source_bundle_sha256")
        or ""
    )
    if frozen_bundle != expected_bundle:
        raise RunnerError("Run Spec source bundle does not match the expected deployed source")

    # This production validator binds the immutable spec to the current deployment,
    # source-raster stat identity, PostgreSQL Run row, and DB-recorded spec hash.
    spec, database, validated_path = validate_recovery_run(spec_path, scripts_dir)
    if validated_path != spec_path:
        raise RunnerError("validated Run Spec path changed unexpectedly")

    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([str(Path(__file__).name)])
    runner = ThreadedV5AsyncInferenceRunner(str(scripts_dir))
    retire_after(runner, runner.shutdown_finished)
    timeout_timer = QTimer()
    timeout_timer.setSingleShot(True)
    stop_grace_timer = QTimer()
    stop_grace_timer.setSingleShot(True)
    stop_condition_timer = QTimer()
    stop_condition_timer.setInterval(100)
    state: dict[str, Any] = {
        "timed_out": False,
        "stop_reason": "",
        "stop_requested": False,
        "stop_call_count": 0,
        "stop_requested_at": None,
        "runner_started_at": None,
        "stop_observation": None,
        "pipeline_result": None,
        "shutdown_requested": False,
        "shutdown_finished": False,
    }

    def begin_shutdown() -> None:
        if state["shutdown_requested"]:
            return
        state["shutdown_requested"] = True
        timeout_timer.stop()
        stop_grace_timer.stop()
        stop_condition_timer.stop()
        runner.shutdown()

    def on_pipeline_finished(payload: object) -> None:
        if state["pipeline_result"] is None:
            state["pipeline_result"] = dict(payload or {})
        if state["stop_requested_at"] is not None and state["stop_observation"]:
            state["stop_observation"]["pipeline_finished_after_stop_seconds"] = round(
                time.monotonic() - float(state["stop_requested_at"]), 3
            )
        if args.stop_after_work_package_ready and state["stop_observation"] is None:
            state["stop_reason"] = (
                state["stop_reason"]
                or "pipeline finished before a ready work_package Job was observed"
            )
            state["stop_observation"] = {
                "status": "not_triggered",
                "reason": state["stop_reason"],
                "poll_interval_ms": 100,
            }
        begin_shutdown()

    def request_stop(reason: str, *, timed_out: bool) -> None:
        if state["shutdown_requested"] or state["stop_requested"]:
            return
        state["stop_requested"] = True
        state["stop_requested_at"] = time.monotonic()
        stop_condition_timer.stop()
        if timed_out:
            state["timed_out"] = True
        if not state["stop_reason"]:
            state["stop_reason"] = reason
        state["stop_call_count"] += 1
        runner.stop()
        stop_grace_timer.start(args.stop_grace_seconds * 1000)

    def poll_stop_condition() -> None:
        if state["stop_requested"] or state["shutdown_requested"]:
            stop_condition_timer.stop()
            return
        try:
            work_package_counts = database.jobs.job_counts(
                str(spec.get("run_id") or ""),
                job_type="work_package",
            )
            if int(work_package_counts.get("ready", 0)) < 1:
                return
            snapshot = _stop_snapshot(
                database,
                str(spec.get("run_id") or ""),
                work_package_counts,
            )
            started_at = state["runner_started_at"]
            state["stop_observation"] = {
                "status": "triggered",
                "reason": "first observed work_package Job with status=ready",
                "poll_interval_ms": 100,
                "triggered_after_runner_start_seconds": (
                    round(time.monotonic() - float(started_at), 3)
                    if started_at is not None
                    else None
                ),
                **snapshot,
            }
            request_stop(
                "stop requested after observing a ready work_package Job",
                timed_out=False,
            )
        except Exception as error:
            state["stop_observation"] = {
                "status": "observation_failed",
                "reason": f"{type(error).__name__}: {error}",
                "poll_interval_ms": 100,
            }
            request_stop("work_package stop observation failed", timed_out=False)

    def on_timeout() -> None:
        request_stop(
            f"validation timeout after {args.timeout_seconds} seconds",
            timed_out=True,
        )

    def on_stop_grace_expired() -> None:
        if state["pipeline_result"] is None:
            state["pipeline_result"] = {
                "schema_version": 2,
                "run_id": str(spec.get("run_id") or ""),
                "run_spec": str(spec_path),
                "success": False,
                "status": "stop_timeout",
                "error": (
                    "runner did not emit pipeline_finished within "
                    f"{args.stop_grace_seconds} seconds after stop"
                ),
                "terminal_published": False,
            }
        begin_shutdown()

    def on_shutdown_finished() -> None:
        state["shutdown_finished"] = True
        if args.stop_after_work_package_ready and state["stop_observation"] is None:
            state["stop_observation"] = {
                "status": "not_triggered",
                "reason": (
                    state["stop_reason"]
                    or "runner shut down before a ready work_package Job was observed"
                ),
                "poll_interval_ms": 100,
            }
        if state["stop_requested_at"] is not None and state["stop_observation"]:
            state["stop_observation"]["runner_stop_call_count"] = int(
                state["stop_call_count"]
            )
            state["stop_observation"]["shutdown_finished_after_stop_seconds"] = round(
                time.monotonic() - float(state["stop_requested_at"]), 3
            )
        pipeline = dict(state["pipeline_result"] or {})
        pipeline_ready = (
            bool(pipeline.get("success"))
            and str(pipeline.get("status") or "") == "ready"
            and not state["timed_out"]
            and not state["stop_reason"]
        )
        report = {
            "schema_version": 1,
            "kind": "v5_real_validation_runner",
            "status": "ready" if pipeline_ready else "failed",
            "success": pipeline_ready,
            "run_id": str(spec.get("run_id") or ""),
            "run_spec": str(spec_path),
            "state_schema": schema,
            "effective_device": device,
            "source_bundle_sha256": expected_bundle,
            "action": args.action,
            "timed_out": bool(state["timed_out"]),
            "stop_reason": str(state["stop_reason"]),
            "stop_observation": state["stop_observation"],
            "shutdown_finished": True,
            "imports": imports,
            "pipeline_result": pipeline,
        }
        print(json.dumps(report, ensure_ascii=False, separators=(",", ":")), flush=True)
        state["exit_code"] = 0 if pipeline_ready else 2

    def on_runner_destroyed(_object: object = None) -> None:
        app.exit(int(state.get("exit_code", 2)))

    runner.pipeline_finished.connect(on_pipeline_finished)
    runner.shutdown_finished.connect(on_shutdown_finished)
    runner.destroyed.connect(on_runner_destroyed)
    timeout_timer.timeout.connect(on_timeout)
    stop_grace_timer.timeout.connect(on_stop_grace_expired)
    stop_condition_timer.timeout.connect(poll_stop_condition)

    def signal_handler(signum: int, _frame: object) -> None:
        request_stop(f"received signal {signum}", timed_out=False)

    previous_handlers = {}
    heartbeat = QTimer()
    heartbeat.setInterval(250)
    heartbeat.timeout.connect(lambda: None)
    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        signum = getattr(signal, name, None)
        if signum is not None:
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, signal_handler)
    heartbeat.start()

    def start_runner() -> None:
        try:
            state["runner_started_at"] = time.monotonic()
            _invoke_action(runner, args.action, spec_path)
        except Exception as error:
            state["stop_reason"] = "runner start failed"
            state["pipeline_result"] = {
                "schema_version": 2,
                "run_id": str(spec.get("run_id") or ""),
                "run_spec": str(spec_path),
                "success": False,
                "status": "start_failed",
                "error": str(error),
                "terminal_published": False,
            }
            begin_shutdown()
            return
        timeout_timer.start(args.timeout_seconds * 1000)
        if args.stop_after_work_package_ready:
            stop_condition_timer.start()

    QTimer.singleShot(0, start_runner)
    try:
        exit_code = int(app.exec())
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return run(args)
    except Exception as error:
        return _failure(error)


if __name__ == "__main__":
    raise SystemExit(main())
