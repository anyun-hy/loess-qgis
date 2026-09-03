#!/usr/bin/env python3
"""Aggregate the mandatory v5 scale acceptance report after stream assembly."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "qgis_plugins"
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from labeling_tool.core.run_spec import atomic_write_json, sha256_file
from labeling_tool.core.run_state_db import RunStateDB, run_state_from_spec

from deployment_config import load_json
from runtime_metrics import directory_size


class ScaleAcceptanceError(RuntimeError):
    pass


def _iter_reports(paths):
    for path in sorted(paths):
        try:
            yield load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ScaleAcceptanceError(f"invalid runtime report: {path}: {error}") from error


def _pipeline_timing(path: Path) -> dict[str, float]:
    timestamps = []
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                    timestamps.append(float(value["timestamp"]))
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    continue
    timestamps.sort()
    gaps = [right - left for left, right in zip(timestamps, timestamps[1:])]
    return {
        "elapsed_sec": (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0,
        "longest_no_heartbeat_sec": max(gaps, default=0.0),
        "record_count": len(timestamps),
    }


def _phase_timing_snapshot(run_dir: Path) -> dict[str, Any]:
    """Read runner-owned timing as an observation; acceptance never invents it."""

    path = run_dir / "logs" / "phase_timing.json"
    if not path.is_file():
        return {}
    try:
        value = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    summary = value.get("summary") if isinstance(value, dict) else None
    return dict(summary) if isinstance(summary, dict) else {}


def _scale_level(run_dir: Path, tile_count: int) -> str:
    preparation = run_dir / "logs" / "l0_preparation_report.json"
    if preparation.is_file():
        value = load_json(preparation)
        if value.get("level"):
            return str(value["level"])
    if tile_count >= 100_000:
        return "L3"
    if tile_count >= 10_000:
        return "L2"
    if tile_count >= 1_000:
        return "L1"
    if tile_count == 238:
        return "L0"
    return "ungraded"


def _database_metrics(database: RunStateDB, run_id: str) -> dict[str, Any]:
    with database._connection() as connection:
        counts = {}
        for table in ("tiles", "partitions", "spatial_units", "work_packages"):
            counts[table] = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE run_id=%s", (run_id,)
                ).fetchone()[0]
            )
        job_rows = connection.execute(
            "SELECT status, COUNT(*) AS n FROM jobs WHERE run_id=%s GROUP BY status",
            (run_id,),
        ).fetchall()
        job_counts = {str(row["status"]): int(row["n"]) for row in job_rows}
        job_type_rows = connection.execute(
            """SELECT job_type, status, COUNT(*) AS n FROM jobs
               WHERE run_id=%s GROUP BY job_type, status
               ORDER BY job_type, status""",
            (run_id,),
        ).fetchall()
        job_type_counts: dict[str, dict[str, int]] = {}
        for row in job_type_rows:
            job_type_counts.setdefault(str(row["job_type"]), {})[
                str(row["status"])
            ] = int(row["n"])
        retry_count = int(
            connection.execute(
                """SELECT COALESCE(SUM(
                     CASE WHEN attempt>0 THEN attempt-1 ELSE 0 END
                   ), 0) FROM jobs WHERE run_id=%s""",
                (run_id,),
            ).fetchone()[0]
        )
        package_rows = connection.execute(
            "SELECT status, COUNT(*) AS n FROM work_packages WHERE run_id=%s GROUP BY status",
            (run_id,),
        ).fetchall()
        package_counts = {
            str(row["status"]): int(row["n"]) for row in package_rows
        }
        stream_rows = [
            dict(row)
            for row in connection.execute(
                "SELECT stream_id, status, error FROM streams WHERE run_id=%s ORDER BY stream_id",
                (run_id,),
            ).fetchall()
        ]
        stream_unit_rows = connection.execute(
            """SELECT stream_id, status, COUNT(*) AS n FROM stream_units
               WHERE run_id=%s GROUP BY stream_id, status ORDER BY stream_id, status""",
            (run_id,),
        ).fetchall()
        stream_unit_counts: dict[str, dict[str, int]] = {}
        for row in stream_unit_rows:
            stream_unit_counts.setdefault(str(row["stream_id"]), {})[
                str(row["status"])
            ] = int(row["n"])
        artifact_rows = [
            dict(row)
            for row in connection.execute(
                """SELECT path, sha256, byte_count, status, kind, stream_id, unit_id
                   FROM artifacts WHERE run_id=%s ORDER BY artifact_id""",
                (run_id,),
            ).fetchall()
        ]
    integrity_errors = []
    ready_bytes = 0
    for artifact in artifact_rows:
        if artifact["status"] != "ready":
            continue
        path = Path(str(artifact["path"]))
        ready_bytes += int(artifact["byte_count"])
        if not path.is_file():
            integrity_errors.append(f"missing:{path}")
        elif path.stat().st_size != int(artifact["byte_count"]):
            integrity_errors.append(f"size:{path}")
        elif sha256_file(path) != str(artifact["sha256"]):
            integrity_errors.append(f"sha256:{path}")
    return {
        "counts": counts,
        "job_counts": job_counts,
        "job_type_counts": job_type_counts,
        "retry_count": retry_count,
        "package_counts": package_counts,
        "streams": stream_rows,
        "stream_unit_counts": stream_unit_counts,
        "artifact_counts": dict(Counter(str(row["status"]) for row in artifact_rows)),
        "ready_artifact_bytes": ready_bytes,
        "artifact_integrity_errors": integrity_errors,
        "artifact_rows": artifact_rows,
    }


def _final_artifact_size_observation(
    storage: dict[str, Any],
    actual_bytes: int,
) -> dict[str, Any]:
    """Attach completion data to a frozen, observation-only prediction."""

    prediction = dict(storage.get("final_artifact_size_prediction") or {})
    actual = int(actual_bytes)
    if str(prediction.get("status") or "") != "predicted":
        return {
            "status": str(prediction.get("status") or "not_available"),
            "actual_final_artifact_bytes": actual,
            "observation_only": True,
        }
    try:
        predicted = int(prediction["predicted_final_artifact_bytes"])
    except (KeyError, TypeError, ValueError):
        return {
            "status": "invalid",
            "actual_final_artifact_bytes": actual,
            "observation_only": True,
        }
    if predicted < 1:
        return {
            "status": "invalid",
            "actual_final_artifact_bytes": actual,
            "observation_only": True,
        }
    difference = actual - predicted
    return {
        **prediction,
        "actual_final_artifact_bytes": actual,
        "signed_difference_bytes": difference,
        "signed_difference_ratio": difference / predicted,
    }


def build_scale_acceptance_report(run_spec_path: str | Path) -> dict[str, Any]:
    spec_path = Path(run_spec_path).expanduser().resolve()
    spec = load_json(spec_path)
    if int(spec.get("schema_version") or 0) != 2:
        raise ScaleAcceptanceError("scale acceptance requires run_spec schema 2")
    run_id = str(spec["run_id"])
    run_dir = Path(spec["run_dir"])
    database = run_state_from_spec(spec)
    package_report_paths = run_dir.glob(
        "tmp/work_packages/*/package_report.json"
    )
    metrics = _database_metrics(database, run_id)
    cleanup = database.artifact_cleanup_summary(run_id)
    timing = _pipeline_timing(run_dir / "logs" / "pipeline.jsonl")
    phase_timing = _phase_timing_snapshot(run_dir)

    model_load_counts: Counter[str] = Counter()
    model_cache_hit_counts: Counter[str] = Counter()
    package_report_count = 0
    unit_summaries = [
        summary
        for stream in spec.get("streams") or []
        for summary in database.unit_report_summaries(
            run_id, str(stream["stream_id"])
        )
    ]
    unit_report_count = len(unit_summaries)
    all_boundary_units_passed = all(
        str(summary.get("status") or "") == "passed"
        for summary in unit_summaries
    )
    inferred_tiles = 0
    package_cleaned_bytes = 0
    peak_cache_bytes = 0
    peak_rss_bytes = 0
    package_elapsed = 0.0
    for report in _iter_reports(package_report_paths):
        package_report_count += 1
        package_cleaned_bytes += int(report.get("cleaned_bytes", 0))
        peak_cache_bytes = max(peak_cache_bytes, int(report.get("peak_cache_bytes", 0)))
        peak_rss_bytes = max(peak_rss_bytes, int(report.get("peak_rss_bytes", 0)))
        package_elapsed += float(report.get("elapsed_sec", 0.0))
        for model in report.get("models") or []:
            count = int(model.get("inferred_count", 0))
            inferred_tiles += count
            model_id = str(model["model_id"])
            model_load_counts[model_id] += int(
                model.get("cold_load_count", 1 if count > 0 else 0)
            )
            model_cache_hit_counts[model_id] += int(
                model.get("cache_hit_count", 0)
            )
    model_event_path = run_dir / "logs" / "accelerator_model_loads.jsonl"
    journal_load_counts: Counter[str] = Counter()
    journal_load_completed_counts: Counter[str] = Counter()
    journal_cache_hit_counts: Counter[str] = Counter()
    model_load_counts_by_session: dict[str, Counter[str]] = {}
    if model_event_path.is_file():
        with model_event_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    event = json.loads(line)
                    if str(event.get("run_id")) != run_id:
                        raise ValueError("run_id mismatch")
                    model_id = str(event["model_id"])
                    session_id = str(event["worker_session_id"])
                    session_counts = model_load_counts_by_session.setdefault(
                        session_id, Counter()
                    )
                    event_kind = str(event.get("event") or "")
                    legacy_completed = False
                    if not event_kind and "cold_loaded" in event:
                        # Schema 1 compatibility for already-created Runs.
                        event_kind = (
                            "load_started"
                            if bool(event["cold_loaded"])
                            else "cache_hit"
                        )
                        legacy_completed = bool(event["cold_loaded"])
                    if event_kind == "load_started":
                        journal_load_counts[model_id] += 1
                        session_counts[model_id] += 1
                        if legacy_completed:
                            journal_load_completed_counts[model_id] += 1
                    elif event_kind == "load_completed":
                        journal_load_completed_counts[model_id] += 1
                    elif event_kind == "cache_hit":
                        journal_cache_hit_counts[model_id] += 1
                    else:
                        raise ValueError(f"unknown event: {event_kind!r}")
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ScaleAcceptanceError(
                        "invalid accelerator model-load journal at line "
                        f"{line_number}: {error}"
                    ) from error
    if journal_load_counts:
        model_load_counts = journal_load_counts
    if journal_cache_hit_counts:
        model_cache_hit_counts = journal_cache_hit_counts
    if not journal_load_counts and not journal_cache_hit_counts:
        worker_reports = list(
            _iter_reports(run_dir.glob("logs/accelerator_workers/*.json"))
        )
        if not worker_reports:
            worker_report_path = run_dir / "logs" / "accelerator_worker_report.json"
            if worker_report_path.is_file():
                worker_reports = [load_json(worker_report_path)]
        if worker_reports:
            model_load_counts = Counter()
            model_cache_hit_counts = Counter()
            for worker_report in worker_reports:
                model_load_counts.update(
                    {
                        str(key): int(value)
                        for key, value in (
                            worker_report.get("model_cold_load_counts") or {}
                        ).items()
                    }
                )
                model_cache_hit_counts.update(
                    {
                        str(key): int(value)
                        for key, value in (
                            worker_report.get("model_cache_hit_counts") or {}
                        ).items()
                    }
                )
    expected_packages = int(metrics["counts"]["work_packages"])
    expected_streams = len(spec.get("streams") or [])
    frozen_unit_counts = dict(
        (spec.get("spatial_plan_summary") or {}).get("unit_counts") or {}
    )
    expected_units = sum(int(value) for value in frozen_unit_counts.values())
    if expected_units < 1 or expected_streams < 1:
        raise ScaleAcceptanceError(
            "Run Spec does not declare its frozen spatial-unit task graph"
        )
    expected_unit_reports = expected_units * expected_streams
    fragmentation = dict(spec.get("fragmentation_regularization") or {})
    expected_v33_jobs = (
        int(metrics["counts"]["partitions"]) + 1
        if (
            fragmentation.get("enabled") is True
            and fragmentation.get("policy_id")
            == "fragmentation_v33_configurable_absorption_v1"
            and fragmentation.get("publication") == "authoritative_fusion_core"
        )
        else 0
    )
    storage = dict(spec.get("storage_preflight") or {})
    expected_confidence_jobs = (
        expected_units
        if storage.get("v33_storage_mode") == "streamed_unit_confidence_v1"
        else 0
    )
    job_type_counts = metrics["job_type_counts"]
    expected_job_total = (
        expected_packages
        + expected_unit_reports
        + expected_v33_jobs
        + expected_confidence_jobs
    )
    unit_artifact_rows = [
        row
        for row in metrics["artifact_rows"]
        if str(row.get("kind") or "") in {
            "unit_raw_geoparquet",
            "unit_formal_geoparquet",
            "unit_boundary_report",
            "unit_fitted_edges_geoparquet",
            "unit_boundary_signatures",
        }
    ]
    expected_unit_artifacts = (
        expected_unit_reports * 4
        + sum(
            1
            for summary in unit_summaries
            if int(summary.get("fitted_edge_count") or 0) > 0
        )
    )
    unit_artifact_status_counts = Counter(
        str(row.get("status") or "") for row in unit_artifact_rows
    )
    storage_schema = int(storage.get("storage_tuning_schema_version") or 0)
    frozen_cache_budget_bytes = int(
        storage.get("working_cache_budget_bytes")
        or storage.get("resolved_score_cache_budget_bytes")
        or 0
    )
    cache_budget_gate_applicable = storage_schema >= 2
    hard_gates = {
        "all_package_reports_present": package_report_count == expected_packages,
        "all_unit_reports_present": unit_report_count == expected_unit_reports,
        "all_unit_intermediates_cleaned": (
            len(unit_artifact_rows) == expected_unit_artifacts
            and unit_artifact_status_counts == {"cleaned": expected_unit_artifacts}
        ),
        "all_work_packages_ready": metrics["package_counts"] == {
            "ready": expected_packages
        },
        "all_work_package_jobs_ready": job_type_counts.get(
            "work_package", {}
        ) == {"ready": expected_packages},
        "all_v33_jobs_ready": job_type_counts.get(
            "fragmentation_v33", {}
        ) == ({"ready": expected_v33_jobs} if expected_v33_jobs else {}),
        "all_unit_confidence_jobs_ready": job_type_counts.get(
            "unit_confidence", {}
        ) == (
            {"ready": expected_confidence_jobs}
            if expected_confidence_jobs
            else {}
        ),
        "all_unit_jobs_ready": job_type_counts.get("unit_fit", {}) == {
            "ready": expected_unit_reports
        },
        "all_jobs_ready": metrics["job_counts"] == {
            "ready": expected_job_total
        },
        "all_streams_ready": (
            len(metrics["streams"]) == expected_streams
            and all(row["status"] == "ready" for row in metrics["streams"])
        ),
        "all_stream_units_ready": all(
            counts == {"ready": expected_units}
            for counts in metrics["stream_unit_counts"].values()
        ) and len(metrics["stream_unit_counts"]) == expected_streams,
        "no_artifact_integrity_errors": not metrics["artifact_integrity_errors"],
        "all_boundary_units_passed": all_boundary_units_passed,
        "peak_cache_within_frozen_budget": (
            not cache_budget_gate_applicable
            or (
                frozen_cache_budget_bytes > 0
                and peak_cache_bytes <= frozen_cache_budget_bytes
            )
        ),
    }
    actual_run_bytes = directory_size(run_dir)
    actual_permanent_bytes = int(metrics["ready_artifact_bytes"])
    final_artifact_observation = _final_artifact_size_observation(
        storage,
        actual_permanent_bytes,
    )
    # The permanent-output reserve is a disk-admission control. It is not a
    # terminal-output prediction and must not affect acceptance as one.
    warnings = []
    hard_passed = all(hard_gates.values())
    status = "failed" if not hard_passed else "warning" if warnings else "passed"
    elapsed = float(timing["elapsed_sec"] or package_elapsed)
    report = {
        "schema_version": 1,
        "status": status,
        "hard_gate_passed": hard_passed,
        "level": _scale_level(run_dir, int(metrics["counts"]["tiles"])),
        "run_id": run_id,
        "run_spec": str(spec_path),
        "run_spec_sha256": sha256_file(spec_path),
        "deployment_identity": spec.get("deployment_identity") or {},
        "tile_count": int(metrics["counts"]["tiles"]),
        "partition_count": int(metrics["counts"]["partitions"]),
        "spatial_unit_count": expected_units,
        "package_count": expected_packages,
        "stream_count": expected_streams,
        "stream_unit_counts": metrics["stream_unit_counts"],
        "unit_report_summary_count": unit_report_count,
        "unit_artifact_cleanup": {
            "expected_artifact_count": expected_unit_artifacts,
            "artifact_count": len(unit_artifact_rows),
            "status_counts": dict(sorted(unit_artifact_status_counts.items())),
        },
        "job_counts": metrics["job_counts"],
        "job_type_counts": job_type_counts,
        "expected_job_counts": {
            "work_package": expected_packages,
            "fragmentation_v33": expected_v33_jobs,
            "unit_confidence": expected_confidence_jobs,
            "unit_fit": expected_unit_reports,
            "total": expected_job_total,
        },
        "failed_count": int(metrics["job_counts"].get("failed", 0)),
        "retry_count": int(metrics["retry_count"]),
        "model_load_counts": dict(sorted(model_load_counts.items())),
        "model_cache_hit_counts": dict(sorted(model_cache_hit_counts.items())),
        "model_load_completed_counts": dict(
            sorted(journal_load_completed_counts.items())
        ),
        "model_load_incomplete_counts": {
            model_id: max(
                0,
                int(count) - int(journal_load_completed_counts.get(model_id, 0)),
            )
            for model_id, count in sorted(journal_load_counts.items())
            if int(count) > int(journal_load_completed_counts.get(model_id, 0))
        },
        "model_load_counts_by_worker_session": {
            session_id: dict(sorted(counts.items()))
            for session_id, counts in sorted(
                model_load_counts_by_session.items()
            )
        },
        "inferred_model_tile_count": inferred_tiles,
        "peak_cache_bytes": peak_cache_bytes,
        "peak_rss_bytes": peak_rss_bytes,
        "cleaned_bytes": package_cleaned_bytes + int(cleanup["cleaned_bytes"]),
        "throughput_model_tiles_per_sec": inferred_tiles / elapsed if elapsed > 0 else 0.0,
        "elapsed_sec": elapsed,
        "longest_no_heartbeat_sec": float(timing["longest_no_heartbeat_sec"]),
        "pipeline_log_record_count": int(timing["record_count"]),
        "phase_timing_snapshot": phase_timing,
        "storage": {
            "preflight": storage,
            "actual_run_bytes": actual_run_bytes,
            "actual_permanent_artifact_bytes": actual_permanent_bytes,
            "actual_nonartifact_and_temporary_bytes": max(
                0, actual_run_bytes - actual_permanent_bytes
            ),
            "ready_artifact_bytes": metrics["ready_artifact_bytes"],
            "final_artifact_size_observation": final_artifact_observation,
            "cache_budget_gate_applicable": cache_budget_gate_applicable,
            "frozen_cache_budget_bytes": frozen_cache_budget_bytes,
            "peak_cache_bytes": peak_cache_bytes,
        },
        "artifact_counts": metrics["artifact_counts"],
        "artifact_integrity_errors": metrics["artifact_integrity_errors"],
        "hard_gates": hard_gates,
        "warnings": warnings,
    }
    report_path = run_dir / "logs" / "scale_acceptance_report.json"
    atomic_write_json(report_path, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-spec", required=True)
    args = parser.parse_args(argv)
    try:
        report = build_scale_acceptance_report(args.run_spec)
        print(
            json.dumps(
                {
                    "event": "scale_acceptance_finished",
                    "status": report["status"],
                    "level": report["level"],
                    "hard_gate_passed": report["hard_gate_passed"],
                    "final_artifact_size_observation": report["storage"].get(
                        "final_artifact_size_observation"
                    ),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            flush=True,
        )
        return 0 if report["hard_gate_passed"] else 2
    except Exception as error:
        print(
            json.dumps(
                {"event": "scale_acceptance_failed", "error": str(error)},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
