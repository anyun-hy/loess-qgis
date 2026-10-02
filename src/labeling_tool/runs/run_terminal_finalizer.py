"""Terminal result and file owner for the v5 inference runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from labeling_tool.runs.result_catalog import artifact_sha256
from labeling_tool.shared.contracts.run_spec import atomic_write_json, sha256_file


class TerminalJobs(Protocol):
    def job_counts(self, run_id: str, *, job_type: str = "") -> dict[str, int]: ...


class TerminalArtifacts(Protocol):
    def artifact_cleanup_summary(self, run_id: str) -> dict[str, Any]: ...


class FailOpenStreams(Protocol):
    def __call__(self, run_id: str, error: str) -> int: ...


class TerminalHistory(Protocol):
    def finish_execution(
        self,
        run_id: str,
        execution_id: str,
        *,
        status: str,
        message: str,
        recording_complete: bool = True,
    ) -> bool: ...


class SetRunStatus(Protocol):
    def __call__(
        self,
        run_id: str,
        status: str,
        *,
        expected: tuple[str, ...],
    ) -> bool: ...


class PhaseSummary(Protocol):
    def __call__(self, observed_at: float, /) -> Mapping[str, Any]: ...


class RunTerminalPublicationError(RuntimeError):
    """The current execution could not publish one coherent terminal result."""


class RunTerminalStateConflictError(RunTerminalPublicationError):
    """The owner-fenced terminal Run transition was rejected."""


@dataclass(frozen=True)
class RunTerminalContext:
    """Data snapshot required to construct one terminal Run result."""

    spec: Mapping[str, Any]
    spec_path: str
    success: bool
    stopped: bool
    error: str
    started_at: float
    manual_package_reset: Mapping[str, Any]
    execution_id: str
    monitor_history_incomplete: bool


def build_result_stream(
    spec: Mapping[str, Any], stream: Mapping[str, Any]
) -> dict[str, Any]:
    run_dir = Path(str(spec["run_dir"]))
    root = (
        run_dir / "models" / str(stream["model_id"])
        if stream["kind"] == "model"
        else run_dir / "fusion" / str(stream["profile_id"])
    )
    paths = {
        "mask_mosaic": str(root / "mask_mosaic.vrt"),
        "confidence_mosaic": str(root / "confidence_mosaic.vrt"),
        "semantic_polygons_raw": str(root / "semantic_polygons_raw.gpkg"),
        "semantic_polygons": str(root / "semantic_polygons.gpkg"),
        "boundary_fitting_report": str(root / "boundary_fitting_report.json"),
        "fitted_edges": str(root / "fitted_edges.gpkg"),
    }
    boundary_status = "failed"
    try:
        with open(paths["boundary_fitting_report"], encoding="utf-8") as handle:
            boundary_report = json.load(handle)
        if (
            boundary_report.get("status") == "passed"
            and (boundary_report.get("validation") or {}).get("passed") is True
        ):
            boundary_status = "passed"
    except (OSError, ValueError, TypeError):
        pass
    result = {
        "stream_id": stream["stream_id"],
        "kind": stream["kind"],
        "model_id": stream.get("model_id", ""),
        "fusion_profile_id": stream.get("profile_id", ""),
        "version": stream.get("version", ""),
        "status": "ready",
        "boundary_smoothing_enabled": bool(
            (spec.get("boundary_fitting") or {}).get("enabled", True)
        ),
        "boundary_fitting_status": boundary_status,
        "paths": paths,
        "output_sha256": {key: artifact_sha256(path) for key, path in paths.items()},
    }
    result["review_polygons"] = paths["semantic_polygons"]
    result["review_layer_name"] = "semantic_polygons"
    result["output_sha256"]["review_polygons"] = result["output_sha256"][
        "semantic_polygons"
    ]
    return result


def build_run_result(
    context: RunTerminalContext,
    *,
    clock: Callable[[], float],
    phase_summary: PhaseSummary,
) -> dict[str, Any]:
    """Build stream identities and file hashes before terminal DB writes."""

    spec = context.spec
    ready_streams = (
        [build_result_stream(spec, stream) for stream in spec.get("streams", [])]
        if context.success
        else []
    )
    failed_streams = [] if context.success else list(spec.get("streams", []))
    run_spec_sha256 = sha256_file(context.spec_path)
    elapsed_sec = round(clock() - context.started_at, 3)
    phase_timing = dict(phase_summary(clock()))
    result: dict[str, Any] = {
        "schema_version": 2,
        "run_id": spec["run_id"],
        "run_spec": context.spec_path,
        "run_spec_sha256": run_spec_sha256,
        "run_dir": spec["run_dir"],
        "success": bool(context.success),
        "status": (
            "ready" if context.success else "stopped" if context.stopped else "failed"
        ),
        "error": context.error,
        "ready_streams": ready_streams,
        "failed_streams": failed_streams,
        "streams": ready_streams if context.success else failed_streams,
        "elapsed_sec": elapsed_sec,
        "phase_timing": phase_timing,
        "deployment_identity": spec.get("deployment_identity") or {},
    }
    if context.manual_package_reset:
        result["manual_package_reset"] = dict(context.manual_package_reset)
    scale_report = Path(str(spec["run_dir"])) / "logs" / "scale_acceptance_report.json"
    if scale_report.is_file():
        result["scale_acceptance_report"] = str(scale_report)
        result["scale_acceptance_report_sha256"] = sha256_file(scale_report)
        try:
            scale_value = json.loads(scale_report.read_text(encoding="utf-8"))
            observation = (scale_value.get("storage") or {}).get(
                "final_artifact_size_observation"
            ) or {}
            if isinstance(observation, dict):
                result["final_artifact_size_observation"] = observation
        except (OSError, ValueError):
            pass
    return result


def finalize_run_outputs(
    context: RunTerminalContext,
    result: Mapping[str, Any],
    *,
    jobs: TerminalJobs,
    artifacts: TerminalArtifacts,
    history: TerminalHistory,
    set_run_status: SetRunStatus,
    emit_history_error: Callable[[str], None],
    fail_open_streams: FailOpenStreams | None = None,
) -> dict[str, Any]:
    """Seal history and write terminal files in their established order."""

    value = dict(result)
    run_id = str(context.spec["run_id"])
    terminal_status = (
        "ready" if context.success else "stopped" if context.stopped else "failed"
    )
    transitioned = set_run_status(
        run_id,
        terminal_status,
        expected=("running", "raster_ready"),
    )
    if not transitioned:
        raise RunTerminalStateConflictError(
            f"Run {run_id} terminal state changed before {terminal_status} publication"
        )
    if not context.success and not context.stopped and fail_open_streams is not None:
        fail_open_streams(run_id, context.error)
    counts = dict(jobs.job_counts(run_id))
    history_complete = True
    if context.execution_id:
        try:
            sealed = history.finish_execution(
                run_id,
                context.execution_id,
                status=(
                    "completed"
                    if context.success
                    else "stopped"
                    if context.stopped
                    else "failed"
                ),
                message=context.error,
                recording_complete=not context.monitor_history_incomplete,
            )
            if not sealed:
                raise RunTerminalPublicationError(
                    f"Run {run_id} execution history is no longer current"
                )
        except Exception as history_error:
            if isinstance(history_error, RunTerminalPublicationError):
                raise
            history_complete = False
            emit_history_error(str(history_error))
    value["monitor_execution_id"] = context.execution_id
    history_complete = history_complete and not context.monitor_history_incomplete
    value["monitor_history_complete"] = history_complete

    run_dir = Path(str(context.spec["run_dir"]))
    run_report = {
        "schema_version": 2,
        "run_id": run_id,
        "status": value["status"],
        "success": bool(context.success),
        "error": context.error,
        "elapsed_sec": value["elapsed_sec"],
        "phase_timing": value["phase_timing"],
        "deployment_identity": value["deployment_identity"],
        "tile_grid": context.spec.get("tile_grid") or {},
        "spatial_plan_summary": context.spec.get("spatial_plan_summary") or {},
        "storage_preflight": context.spec.get("storage_preflight") or {},
        "final_artifact_size_observation": value.get("final_artifact_size_observation")
        or {},
        "job_counts": counts,
        "artifact_cleanup": artifacts.artifact_cleanup_summary(run_id),
        "ready_stream_ids": [item["stream_id"] for item in value["ready_streams"]],
        "monitor_execution_id": context.execution_id,
        "monitor_history_complete": history_complete,
    }
    atomic_write_json(run_dir / "logs" / "run_report.json", run_report)
    atomic_write_json(
        run_dir / "logs" / "failures.json",
        {
            "run_id": run_id,
            "failed_job_count": int(counts.get("failed", 0)),
            "error": context.error,
        },
    )
    # The manifest is the filesystem completion marker and must be last.
    atomic_write_json(run_dir / "run_manifest.json", value)
    value["terminal_published"] = True
    return value
