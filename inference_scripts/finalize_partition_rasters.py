"""Build per-stream VRT entry points after every Work Package raster is committed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "qgis_plugins"
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from labeling_tool.core.run_spec import sha256_file
from labeling_tool.core.run_state_db import RunStateDB, run_state_from_spec

from deployment_config import load_json
from partition_mosaic import build_vrt


class RasterFinalizeError(RuntimeError):
    pass


def _commit_vrt(
    database: RunStateDB,
    run_id: str,
    stream_id: str,
    kind: str,
    path: Path,
) -> int:
    artifact_id = database.register_artifact(
        run_id, kind, path, stream_id=stream_id, unit_id="mosaic"
    )
    artifact = database.get_artifact(artifact_id)
    digest = sha256_file(path)
    if artifact and artifact["status"] == "ready":
        if artifact["byte_count"] == path.stat().st_size and artifact["sha256"] == digest:
            return artifact_id
        raise RasterFinalizeError(f"ready VRT changed on disk: {path}")
    if not database.mark_artifact_ready(
        artifact_id, byte_count=path.stat().st_size, sha256=digest
    ):
        raise RasterFinalizeError(f"cannot commit VRT Artifact: {path}")
    return artifact_id


def finalize_partition_rasters(run_spec_path: str | Path) -> dict[str, Any]:
    spec = load_json(Path(run_spec_path).resolve())
    if spec.get("schema_version") != 2:
        raise RasterFinalizeError("partition raster finalizer requires run_spec schema 2")
    run_id = str(spec["run_id"])
    run_dir = Path(spec["run_dir"])
    database = run_state_from_spec(spec)
    package_counts = database.work_package_counts(run_id)
    total_packages = sum(package_counts.values())
    if total_packages < 1 or package_counts != {"ready": total_packages}:
        raise RasterFinalizeError(f"Work Packages are not all ready: {package_counts}")
    fragmentation = dict(spec.get("fragmentation_regularization") or {})
    production_v33 = bool(
        fragmentation.get("enabled") is True
        and fragmentation.get("policy_id")
        == "fragmentation_v33_configurable_absorption_v1"
        and fragmentation.get("publication") == "authoritative_fusion_core"
    )
    if production_v33:
        v33_counts = database.job_counts(run_id, job_type="fragmentation_v33")
        non_ready = {
            status: count
            for status, count in v33_counts.items()
            if status != "ready" and int(count)
        }
        if int(v33_counts.get("ready", 0)) < 1 or non_ready:
            raise RasterFinalizeError(
                f"V3.3 authoritative raster is not ready: {v33_counts}"
            )
    partition_count = int(spec["spatial_plan_summary"]["partition_count"])
    outputs = []
    for stream in spec["streams"]:
        stream_id = str(stream["stream_id"])
        masks = database.artifacts_for_stream(run_id, stream_id, kind="core_mask")
        confidence = database.artifacts_for_stream(
            run_id, stream_id, kind="core_confidence"
        )
        if len(masks) != partition_count or len(confidence) != partition_count:
            raise RasterFinalizeError(
                f"stream {stream_id} has incomplete raster parts: "
                f"mask={len(masks)}, confidence={len(confidence)}, expected={partition_count}"
            )
        if stream["kind"] == "model":
            root = run_dir / "models" / str(stream["model_id"])
        else:
            root = run_dir / "fusion" / str(stream["profile_id"])
        mask_vrt = Path(build_vrt(root / "mask_mosaic.vrt", [item["path"] for item in masks]))
        confidence_vrt = Path(
            build_vrt(
                root / "confidence_mosaic.vrt",
                [item["path"] for item in confidence],
            )
        )
        _commit_vrt(database, run_id, stream_id, "mask_vrt", mask_vrt)
        _commit_vrt(database, run_id, stream_id, "confidence_vrt", confidence_vrt)
        database.set_stream_status(run_id, stream_id, "raster_ready")
        outputs.append(
            {
                "stream_id": stream_id,
                "mask_vrt": str(mask_vrt),
                "confidence_vrt": str(confidence_vrt),
                "partition_count": partition_count,
            }
        )
    database.set_run_status(run_id, "raster_ready", expected=("planned", "running", "raster_ready"))
    result = {
        "run_id": run_id,
        "status": "raster_ready",
        "package_count": total_packages,
        "partition_count": partition_count,
        "streams": outputs,
    }
    print(json.dumps({"event": "partition_rasters_finalized", **result}, separators=(",", ":")))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build VRTs for committed Core raster parts")
    parser.add_argument("--run-spec", required=True)
    args = parser.parse_args(argv)
    try:
        finalize_partition_rasters(args.run_spec)
        return 0
    except Exception as error:
        print(json.dumps({"event": "partition_raster_finalize_failed", "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
