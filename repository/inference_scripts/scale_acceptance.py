#!/usr/bin/env python3
"""Aggregate the mandatory v5 scale acceptance report after stream assembly."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "qgis_plugins"
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from labeling_tool.core.run_spec import atomic_write_json, sha256_file
from labeling_tool.core.run_state_db import RunStateDB

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


def _database_metrics(database_path: Path, run_id: str) -> dict[str, Any]:
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        counts = {}
        for table in ("tiles", "partitions", "spatial_units", "work_packages"):
            counts[table] = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE run_id=?", (run_id,)
                ).fetchone()[0]
            )
        job_rows = connection.execute(
            "SELECT status, COUNT(*) AS n FROM jobs WHERE run_id=? GROUP BY status",
            (run_id,),
        ).fetchall()
        job_counts = {str(row["status"]): int(row["n"]) for row in job_rows}
        retry_count = int(
            connection.execute(
                "SELECT COALESCE(SUM(MAX(attempt - 1, 0)), 0) FROM jobs WHERE run_id=?",
                (run_id,),
            ).fetchone()[0]
        )
        package_rows = connection.execute(
            "SELECT status, COUNT(*) AS n FROM work_packages WHERE run_id=? GROUP BY status",
            (run_id,),
        ).fetchall()
        package_counts = {
            str(row["status"]): int(row["n"]) for row in package_rows
        }
        stream_rows = [
            dict(row)
            for row in connection.execute(
                "SELECT stream_id, status, error FROM streams WHERE run_id=? ORDER BY stream_id",
                (run_id,),
            ).fetchall()
        ]
        stream_unit_rows = connection.execute(
            """SELECT stream_id, status, COUNT(*) AS n FROM stream_units
               WHERE run_id=? GROUP BY stream_id, status ORDER BY stream_id, status""",
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
                   FROM artifacts WHERE run_id=? ORDER BY artifact_id""",
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
        "retry_count": retry_count,
        "package_counts": package_counts,
        "streams": stream_rows,
        "stream_unit_counts": stream_unit_counts,
        "artifact_counts": dict(Counter(str(row["status"]) for row in artifact_rows)),
        "ready_artifact_bytes": ready_bytes,
        "artifact_integrity_errors": integrity_errors,
    }


def build_scale_acceptance_report(run_spec_path: str | Path) -> dict[str, Any]:
    spec_path = Path(run_spec_path).expanduser().resolve()
    spec = load_json(spec_path)
    if int(spec.get("schema_version") or 0) != 2:
        raise ScaleAcceptanceError("scale acceptance requires run_spec schema 2")
    run_id = str(spec["run_id"])
    run_dir = Path(spec["run_dir"])
    database_path = Path(spec["state_db"])
    database = RunStateDB(database_path)
    package_report_paths = run_dir.glob(
        "tmp/work_packages/*/package_report.json"
    )
    unit_report_paths = run_dir.glob("tmp/unit_outputs/**/*_report.json")
    metrics = _database_metrics(database_path, run_id)
    cleanup = database.artifact_cleanup_summary(run_id)
    timing = _pipeline_timing(run_dir / "logs" / "pipeline.jsonl")

    model_load_counts: Counter[str] = Counter()
    package_report_count = 0
    unit_report_count = 0
    all_boundary_units_passed = True
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
            if count > 0:
                model_load_counts[str(model["model_id"])] += 1
    for report in _iter_reports(unit_report_paths):
        unit_report_count += 1
        peak_rss_bytes = max(peak_rss_bytes, int(report.get("peak_rss_bytes", 0)))
        if report.get("status") != "passed":
            all_boundary_units_passed = False

    expected_packages = int(metrics["counts"]["work_packages"])
    expected_units = int(metrics["counts"]["spatial_units"])
    expected_streams = len(spec.get("streams") or [])
    expected_unit_reports = expected_units * expected_streams
    hard_gates = {
        "all_package_reports_present": package_report_count == expected_packages,
        "all_unit_reports_present": unit_report_count == expected_unit_reports,
        "all_work_packages_ready": metrics["package_counts"] == {
            "ready": expected_packages
        },
        "all_jobs_ready": metrics["job_counts"] == {
            "ready": expected_packages + expected_unit_reports
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
    }
    actual_run_bytes = directory_size(run_dir)
    storage = dict(spec.get("storage_preflight") or {})
    estimated_permanent = int(storage.get("estimated_permanent_bytes", 0))
    disk_deviation = (
        abs(actual_run_bytes - estimated_permanent) / estimated_permanent
        if estimated_permanent > 0
        else 0.0
    )
    warnings = []
    if estimated_permanent > 0 and disk_deviation > 0.20:
        warnings.append(
            "actual run bytes differ from estimated permanent bytes by more than 20%"
        )
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
        "tile_count": int(metrics["counts"]["tiles"]),
        "partition_count": int(metrics["counts"]["partitions"]),
        "spatial_unit_count": expected_units,
        "package_count": expected_packages,
        "stream_count": expected_streams,
        "stream_unit_counts": metrics["stream_unit_counts"],
        "job_counts": metrics["job_counts"],
        "failed_count": int(metrics["job_counts"].get("failed", 0)),
        "retry_count": int(metrics["retry_count"]),
        "model_load_counts": dict(sorted(model_load_counts.items())),
        "inferred_model_tile_count": inferred_tiles,
        "peak_cache_bytes": peak_cache_bytes,
        "peak_rss_bytes": peak_rss_bytes,
        "cleaned_bytes": package_cleaned_bytes + int(cleanup["cleaned_bytes"]),
        "throughput_model_tiles_per_sec": inferred_tiles / elapsed if elapsed > 0 else 0.0,
        "elapsed_sec": elapsed,
        "longest_no_heartbeat_sec": float(timing["longest_no_heartbeat_sec"]),
        "pipeline_log_record_count": int(timing["record_count"]),
        "storage": {
            "preflight": storage,
            "actual_run_bytes": actual_run_bytes,
            "ready_artifact_bytes": metrics["ready_artifact_bytes"],
            "permanent_estimate_deviation_ratio": disk_deviation,
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
