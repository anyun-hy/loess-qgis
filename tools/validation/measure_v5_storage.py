#!/usr/bin/env python3
"""Sample one frozen V5 Run's PostgreSQL and filesystem storage usage."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


RUN_ID_RE = re.compile(r"^\d{8}_\d{6}_[a-z0-9]{4,16}$")
SCHEMA_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
MAX_SPEC_BYTES = 2 * 1024 * 1024
MAX_DURATION_SECONDS = 7 * 24 * 60 * 60


class MeasurementError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunContext:
    run_id: str
    dsn: str
    schema: str
    output_root: Path
    run_dir: Path
    cache_root: Path


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return parsed


def _bounded_seconds(value: str) -> float:
    parsed = _positive_float(value)
    if parsed > MAX_DURATION_SECONDS:
        raise argparse.ArgumentTypeError(
            f"value must not exceed {MAX_DURATION_SECONDS} seconds"
        )
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only, independent-process storage sampler for one frozen "
            "PostgreSQL V5 Run"
        )
    )
    parser.add_argument("--run-spec", required=True)
    parser.add_argument("--report", required=True, help="new JSONL report path")
    parser.add_argument(
        "--interval-seconds",
        type=_positive_float,
        default=5.0,
        help="seconds between sample starts (default: 5)",
    )
    parser.add_argument(
        "--max-seconds",
        type=_bounded_seconds,
        required=True,
        help=f"hard sampling bound, at most {MAX_DURATION_SECONDS} seconds",
    )
    parser.add_argument(
        "--runner-pid",
        type=_positive_int,
        help="stop after this initially-live runner process exits",
    )
    parser.add_argument(
        "--stop-file",
        help="stop when this explicitly selected path appears",
    )
    return parser


def _absolute_path(raw: Any, field: str) -> Path:
    value = str(raw or "").strip()
    if not value:
        raise MeasurementError(f"Run Spec {field} is required")
    path = Path(os.path.abspath(os.path.expanduser(value)))
    if not path.is_absolute():
        raise MeasurementError(f"Run Spec {field} must be absolute")
    return path


def _load_context(run_spec_path: Path) -> RunContext:
    if run_spec_path.is_symlink():
        raise MeasurementError("Run Spec must not be a symlink")
    try:
        metadata = run_spec_path.stat()
    except FileNotFoundError as exc:
        raise MeasurementError("Run Spec does not exist") from exc
    if not run_spec_path.is_file() or metadata.st_size > MAX_SPEC_BYTES:
        raise MeasurementError("Run Spec must be a regular JSON file of at most 2 MiB")
    try:
        spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MeasurementError(f"cannot read Run Spec: {exc}") from exc
    if not isinstance(spec, dict) or int(spec.get("schema_version") or 0) != 2:
        raise MeasurementError("Run Spec must use schema_version=2")
    if str(spec.get("state_backend") or "").strip().lower() != "postgresql":
        raise MeasurementError("Run Spec must use state_backend=postgresql")

    run_id = str(spec.get("run_id") or "").strip()
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise MeasurementError("Run Spec run_id is invalid")
    dsn = str(spec.get("state_db") or "").strip()
    if not dsn:
        raise MeasurementError("Run Spec state_db DSN is required")
    schema = str(spec.get("state_schema") or "").strip()
    if SCHEMA_RE.fullmatch(schema) is None:
        raise MeasurementError("Run Spec state_schema is invalid")

    output_root = _absolute_path(spec.get("output_root"), "output_root")
    run_dir = _absolute_path(spec.get("run_dir"), "run_dir")
    cache_root = _absolute_path(spec.get("cache_root"), "cache_root")
    if output_root.is_symlink() or run_dir.is_symlink() or cache_root.is_symlink():
        raise MeasurementError("Run storage roots must not be symlinks")
    if not output_root.is_dir():
        raise MeasurementError("Run Spec output_root must be an existing directory")
    if run_dir != output_root / "runs" / run_id:
        raise MeasurementError("Run Spec run_dir is outside its canonical output path")
    if cache_root != output_root / "cache" / run_id:
        raise MeasurementError(
            "Run Spec cache_root is outside its canonical output path"
        )

    resolved_output = output_root.resolve(strict=True)
    for name, candidate in (("run_dir", run_dir), ("cache_root", cache_root)):
        try:
            candidate.resolve(strict=False).relative_to(resolved_output)
        except ValueError as exc:
            raise MeasurementError(f"Run Spec {name} escapes output_root") from exc
    return RunContext(
        run_id=run_id,
        dsn=dsn,
        schema=schema,
        output_root=output_root,
        run_dir=run_dir,
        cache_root=cache_root,
    )


def _scan_tree(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(root),
        "exists": False,
        "file_count": 0,
        "st_size_bytes": 0,
        "st_blocks_512": 0,
        "allocated_bytes": 0,
        "symlinks_skipped": 0,
        "races_skipped": 0,
    }
    try:
        if root.is_symlink():
            result["symlinks_skipped"] = 1
            return result
        if not root.is_dir():
            return result
    except FileNotFoundError:
        result["races_skipped"] = 1
        return result

    result["exists"] = True
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            result["symlinks_skipped"] += 1
                        elif entry.is_dir(follow_symlinks=False):
                            pending.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            metadata = entry.stat(follow_symlinks=False)
                            blocks = int(getattr(metadata, "st_blocks", 0))
                            result["file_count"] += 1
                            result["st_size_bytes"] += int(metadata.st_size)
                            result["st_blocks_512"] += blocks
                    except FileNotFoundError:
                        result["races_skipped"] += 1
        except FileNotFoundError:
            result["races_skipped"] += 1
    result["allocated_bytes"] = int(result["st_blocks_512"]) * 512
    return result


def _connect_database(context: RunContext):
    try:
        import psycopg2
    except ImportError as exc:
        raise MeasurementError("psycopg2 is required") from exc
    connection = psycopg2.connect(
        context.dsn,
        connect_timeout=10,
        application_name="loess-v5-storage-sampler",
    )
    connection.set_session(readonly=True, autocommit=True)
    return connection


def _database_snapshot(connection: Any, schema: str, run_id: str) -> dict[str, Any]:
    from psycopg2 import sql

    def table(name: str):
        return sql.Identifier(schema, name)

    with connection.cursor() as cursor:
        cursor.execute(
            sql.SQL(
                "SELECT kind, status, COUNT(*), COALESCE(SUM(byte_count), 0), "
                "COALESCE(SUM(ref_count), 0) FROM {} WHERE run_id=%s "
                "GROUP BY kind, status ORDER BY kind, status"
            ).format(table("artifacts")),
            (run_id,),
        )
        artifacts = [
            {
                "kind": str(kind),
                "status": str(status),
                "count": int(count),
                "byte_count": int(byte_count),
                "ref_count": int(ref_count),
            }
            for kind, status, count, byte_count, ref_count in cursor.fetchall()
        ]
        cursor.execute(
            sql.SQL(
                "SELECT status, COUNT(*) FROM {} WHERE run_id=%s "
                "GROUP BY status ORDER BY status"
            ).format(table("work_packages")),
            (run_id,),
        )
        work_packages = [
            {"status": str(status), "count": int(count)}
            for status, count in cursor.fetchall()
        ]
        cursor.execute(
            sql.SQL(
                "SELECT job_type, status, COUNT(*) FROM {} WHERE run_id=%s "
                "GROUP BY job_type, status ORDER BY job_type, status"
            ).format(table("jobs")),
            (run_id,),
        )
        jobs = [
            {"job_type": str(job_type), "status": str(status), "count": int(count)}
            for job_type, status, count in cursor.fetchall()
        ]
        cursor.execute(
            sql.SQL("SELECT status FROM {} WHERE run_id=%s").format(table("runs")),
            (run_id,),
        )
        row = cursor.fetchone()
    if row is None:
        raise MeasurementError("Run is absent from the frozen PostgreSQL schema")
    return {
        "run_status": str(row[0]),
        "artifacts": artifacts,
        "work_packages": work_packages,
        "jobs": jobs,
    }


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_file_exists(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _sample(
    context: RunContext,
    connection: Any,
    sequence: int,
    started: float,
    *,
    sample_role: str = "periodic",
) -> dict[str, Any]:
    sample_started = time.monotonic()
    database = _database_snapshot(connection, context.schema, context.run_id)
    run_dir = _scan_tree(context.run_dir)
    cache_root = _scan_tree(context.cache_root)
    disk = shutil.disk_usage(context.output_root)
    duration = time.monotonic() - sample_started
    return {
        "schema_version": 1,
        "kind": "v5_storage_sample",
        "sequence": sequence,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "duration_seconds": round(duration, 6),
        "sample_role": sample_role,
        "run_id": context.run_id,
        "database": database,
        "filesystem": {
            "run_dir": run_dir,
            "cache_root": cache_root,
            "total": {
                "file_count": run_dir["file_count"] + cache_root["file_count"],
                "st_size_bytes": run_dir["st_size_bytes"] + cache_root["st_size_bytes"],
                "st_blocks_512": run_dir["st_blocks_512"] + cache_root["st_blocks_512"],
                "allocated_bytes": (
                    run_dir["allocated_bytes"] + cache_root["allocated_bytes"]
                ),
            },
            "disk_usage": {
                "total_bytes": int(disk.total),
                "used_bytes": int(disk.used),
                "free_bytes": int(disk.free),
            },
        },
    }


def _write_line(report: Any, payload: Mapping[str, Any]) -> None:
    report.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    report.flush()


def _summary(
    samples: Sequence[Mapping[str, Any]], reason: str, started: float
) -> dict[str, Any]:
    def peak(path: Sequence[str], *, minimum: bool = False) -> int:
        values: list[int] = []
        for sample in samples:
            value: Any = sample
            for key in path:
                value = value[key]
            values.append(int(value))
        if not values:
            return 0
        return min(values) if minimum else max(values)

    artifact_statuses = sorted(
        {
            str(row["status"])
            for sample in samples
            for row in sample["database"]["artifacts"]
        }
    )
    artifact_bytes_by_status = {
        status: max(
            (
                sum(
                    int(row["byte_count"])
                    for row in sample["database"]["artifacts"]
                    if str(row["status"]) == status
                )
                for sample in samples
            ),
            default=0,
        )
        for status in artifact_statuses
    }
    active_artifact_bytes = [
        sum(
            int(row["byte_count"])
            for row in sample["database"]["artifacts"]
            if str(row["status"]) in {"ready", "writing", "cleaning"}
        )
        for sample in samples
    ]
    elapsed = round(time.monotonic() - started, 6)
    return {
        "schema_version": 1,
        "kind": "v5_storage_summary",
        "status": "completed",
        "stop_reason": reason,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "duration_seconds": elapsed,
        "sample_count": len(samples),
        "sampled_peak": {
            "measurement": "discrete samples; not an exact instantaneous peak",
            "filesystem_st_size_bytes": peak(("filesystem", "total", "st_size_bytes")),
            "filesystem_allocated_bytes": peak(
                ("filesystem", "total", "allocated_bytes")
            ),
            "active_artifact_byte_count": max(active_artifact_bytes, default=0),
            "artifact_byte_count_by_status": artifact_bytes_by_status,
            "recorded_cleaned_artifact_byte_count": artifact_bytes_by_status.get(
                "cleaned", 0
            ),
            "disk_used_bytes": peak(("filesystem", "disk_usage", "used_bytes")),
            "minimum_disk_free_bytes": peak(
                ("filesystem", "disk_usage", "free_bytes"), minimum=True
            ),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    context = _load_context(Path(args.run_spec).expanduser())
    report_path = Path(args.report).expanduser()
    if report_path.is_symlink():
        raise MeasurementError("report path already exists as a symlink")
    stop_file = Path(args.stop_file).expanduser() if args.stop_file else None
    if args.runner_pid is not None and not _process_alive(args.runner_pid):
        raise MeasurementError("runner PID is not alive at sampler start")

    started = time.monotonic()
    samples: list[dict[str, Any]] = []
    connection = _connect_database(context)
    try:
        with report_path.open("x", encoding="utf-8") as report:
            reason = "max_seconds"
            sequence = 0
            while True:
                if _stop_file_exists(stop_file):
                    reason = "stop_file"
                    sample = _sample(
                        context,
                        connection,
                        sequence,
                        started,
                        sample_role="final_after_stop",
                    )
                    samples.append(sample)
                    _write_line(report, sample)
                    break
                if args.runner_pid is not None and not _process_alive(args.runner_pid):
                    reason = "runner_exited"
                    sample = _sample(
                        context,
                        connection,
                        sequence,
                        started,
                        sample_role="final_after_stop",
                    )
                    samples.append(sample)
                    _write_line(report, sample)
                    break
                if time.monotonic() - started >= args.max_seconds and sequence > 0:
                    break

                sample = _sample(context, connection, sequence, started)
                samples.append(sample)
                _write_line(report, sample)
                sequence += 1
                next_sample = started + sequence * args.interval_seconds
                while True:
                    now = time.monotonic()
                    if now - started >= args.max_seconds:
                        reason = "max_seconds"
                        break
                    if _stop_file_exists(stop_file):
                        reason = "stop_file"
                        break
                    if args.runner_pid is not None and not _process_alive(
                        args.runner_pid
                    ):
                        reason = "runner_exited"
                        break
                    if now >= next_sample:
                        reason = "continue"
                        break
                    time.sleep(
                        min(
                            0.2,
                            next_sample - now,
                            args.max_seconds - (now - started),
                        )
                    )
                if reason != "continue":
                    if reason in {"stop_file", "runner_exited"}:
                        final_sample = _sample(
                            context,
                            connection,
                            sequence,
                            started,
                            sample_role="final_after_stop",
                        )
                        samples.append(final_sample)
                        _write_line(report, final_sample)
                    break
            summary = _summary(samples, reason, started)
            _write_line(report, summary)
            return summary
    finally:
        connection.close()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        summary = run(args)
    except (MeasurementError, OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
