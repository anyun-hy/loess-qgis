"""Persist one lossless confidence surface and release probability Halos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine

from boundary_fitting.unit_runtime import (
    _reserved_vector_write,
    _run_storage_guard,
    _unit_probabilities,
)
from deployment_config import load_json
from partition_mosaic import _atomic_raster
from labeling_tool.core.run_spec import sha256_file
from labeling_tool.core.run_state_db import run_state_from_spec
from labeling_tool.core.work_package_planner import unit_confidence_write_reserve


class UnitConfidenceError(RuntimeError):
    pass


def emit(event: str, **payload: Any) -> None:
    print(
        json.dumps(
            {"event": event, **payload},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )


def run_unit_confidence(
    run_spec_path: str | Path,
    stream_id: str,
    unit_id: str,
    *,
    job_id: int,
    lease_token: str,
) -> dict[str, Any]:
    spec = load_json(Path(run_spec_path).resolve())
    if spec.get("schema_version") != 2:
        raise UnitConfidenceError("unit confidence requires run_spec schema 2")
    storage = dict(spec.get("storage_preflight") or {})
    if storage.get("v33_storage_mode") != "streamed_unit_confidence_v1":
        raise UnitConfidenceError(
            "Run does not select streamed V3.3 confidence storage"
        )
    run_id = str(spec["run_id"])
    run_dir = Path(spec["run_dir"])
    database = run_state_from_spec(spec)
    job = database.get_job(int(job_id))
    unit = database.get_spatial_unit(run_id, str(unit_id))
    if (
        job is None
        or unit is None
        or job.get("job_type") != "unit_confidence"
        or str(job.get("run_id")) != run_id
        or str(job.get("stream_id")) != str(stream_id)
        or str(job.get("unit_id")) != str(unit_id)
    ):
        raise UnitConfidenceError("unit confidence Job identity is invalid")
    if job.get("status") != "running" or job.get("lease_token") != lease_token:
        raise UnitConfidenceError("unit confidence Job does not hold its lease")

    try:
        probabilities, valid = _unit_probabilities(
            database,
            run_id,
            str(stream_id),
            unit,
        )
        confidence = np.full(valid.shape, -1.0, dtype=np.float32)
        if np.any(valid):
            confidence[valid] = probabilities[:, valid].max(axis=0)
        window = unit["pixel_window"]
        x0 = int(window["x0"])
        y0 = int(window["y0"])
        transform = Affine(*[float(value) for value in spec["raster"]["transform"]])
        output = (
            run_dir
            / "tmp"
            / "unit_confidence"
            / str(stream_id).replace(":", "_")
            / f"{unit_id}.tif"
        )
        profile = {
            "driver": "GTiff",
            "count": 1,
            "width": int(confidence.shape[1]),
            "height": int(confidence.shape[0]),
            "dtype": "float32",
            "nodata": -1.0,
            "crs": str(spec["raster"]["crs"]),
            "transform": transform * Affine.translation(x0, y0),
            "compress": "deflate",
            "BIGTIFF": "IF_SAFER",
        }
        planned_write_bytes = unit_confidence_write_reserve(confidence.size)
        guard = _run_storage_guard(
            spec,
            database,
            deferred_temporary_exclusion_bytes=planned_write_bytes,
        )
        lock_path = run_dir / "tmp" / ".vector-storage-reserve.lock"
        with _reserved_vector_write(
            guard,
            lock_path,
            f"unit_confidence:{stream_id}:{unit_id}",
            planned_write_bytes,
        ):
            _atomic_raster(
                output,
                confidence,
                profile,
                tags={
                    "storage_role": "lossless_unit_confidence_v1",
                    "source": "normalized_partition_probability_max",
                    "run_id": run_id,
                    "stream_id": str(stream_id),
                    "unit_id": str(unit_id),
                },
            )
        if output.stat().st_size > planned_write_bytes:
            raise UnitConfidenceError(
                "unit confidence exceeded its frozen lossless write reserve"
            )
        if not database.complete_unit_confidence_job(
            int(job_id),
            str(lease_token),
            path=output,
            byte_count=output.stat().st_size,
            sha256=sha256_file(output),
        ):
            raise UnitConfidenceError(
                "unit confidence lease expired before atomic commit"
            )
        report = {
            "run_id": run_id,
            "stream_id": str(stream_id),
            "unit_id": str(unit_id),
            "path": str(output),
            "byte_count": output.stat().st_size,
            "valid_pixel_count": int(np.count_nonzero(valid)),
            "status": "ready",
        }
        emit("unit_confidence_ready", **report)
        return report
    except Exception as error:
        database.finish_job(
            int(job_id),
            str(lease_token),
            status="failed",
            error=str(error),
        )
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Persist one lossless Unit confidence surface"
    )
    parser.add_argument("--run-spec", required=True)
    parser.add_argument("--stream-id", required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--job-id", required=True, type=int)
    parser.add_argument("--lease-token", required=True)
    args = parser.parse_args(argv)
    try:
        run_unit_confidence(
            args.run_spec,
            args.stream_id,
            args.unit_id,
            job_id=args.job_id,
            lease_token=args.lease_token,
        )
        return 0
    except Exception as error:
        emit(
            "unit_confidence_failed",
            stream_id=args.stream_id,
            unit_id=args.unit_id,
            error=str(error),
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
