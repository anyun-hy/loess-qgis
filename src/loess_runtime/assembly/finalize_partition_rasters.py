"""Build per-stream VRT entry points after every Work Package raster is committed."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

from labeling_tool.shared.contracts.run_spec import load_json, sha256_file
from labeling_tool.shared.state.run_state_db import RunStateDB, run_state_from_spec
from loess_runtime.inference.partition_mosaic import build_vrt


class RasterFinalizeError(RuntimeError):
    pass


def _commit_vrt(
    database: RunStateDB,
    run_id: str,
    stream_id: str,
    kind: str,
    path: Path,
) -> int:
    artifact_id = database.artifacts.register_artifact(
        run_id, kind, path, stream_id=stream_id, unit_id="mosaic"
    )
    artifact = database.artifacts.get_artifact(artifact_id)
    digest = sha256_file(path)
    if artifact and artifact["status"] == "ready":
        if artifact["byte_count"] == path.stat().st_size and artifact["sha256"] == digest:
            return artifact_id
        raise RasterFinalizeError(f"ready VRT changed on disk: {path}")
    if not database.artifacts.mark_artifact_ready(
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
    package_counts = database.control_graph.work_package_counts(run_id)
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
        v33_counts = database.jobs.job_counts(run_id, job_type="fragmentation_v33")
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
    outputs: list[dict[str, Any]] = []
    staged_vrts: list[tuple[Path, Path]] = []
    for stream in spec["streams"]:
        stream_id = str(stream["stream_id"])
        masks = database.artifacts.artifacts_for_stream(
            run_id, stream_id, kind="core_mask"
        )
        confidence = database.artifacts.artifacts_for_stream(
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
        mask_vrt = root / "mask_mosaic.vrt"
        confidence_vrt = root / "confidence_mosaic.vrt"
        token = uuid.uuid4().hex
        staged_mask_vrt = mask_vrt.with_name(
            f".{mask_vrt.stem}.{token}.stage{mask_vrt.suffix}"
        )
        staged_confidence_vrt = confidence_vrt.with_name(
            f".{confidence_vrt.stem}.{token}.stage{confidence_vrt.suffix}"
        )
        try:
            build_vrt(staged_mask_vrt, [item["path"] for item in masks])
            build_vrt(
                staged_confidence_vrt,
                [item["path"] for item in confidence],
            )
        except Exception:
            staged_mask_vrt.unlink(missing_ok=True)
            staged_confidence_vrt.unlink(missing_ok=True)
            for staged_path, _canonical_path in staged_vrts:
                staged_path.unlink(missing_ok=True)
            raise
        staged_vrts.extend(
            ((staged_mask_vrt, mask_vrt), (staged_confidence_vrt, confidence_vrt))
        )
        outputs.append(
            {
                "stream_id": stream_id,
                "mask_vrt": str(mask_vrt),
                "confidence_vrt": str(confidence_vrt),
                "partition_count": partition_count,
            }
        )
    try:
        with database.owner_publication(run_id, run_dir) as publication:
            for staged_path, canonical_path in staged_vrts:
                canonical_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged_path, canonical_path)
            for output in outputs:
                stream_id = str(output["stream_id"])
                mask_vrt = Path(str(output["mask_vrt"]))
                confidence_vrt = Path(str(output["confidence_vrt"]))
                _commit_vrt(publication, run_id, stream_id, "mask_vrt", mask_vrt)
                _commit_vrt(
                    publication,
                    run_id,
                    stream_id,
                    "confidence_vrt",
                    confidence_vrt,
                )
                if not publication.run_streams.set_stream_status(
                    run_id, stream_id, "raster_ready"
                ):
                    raise RasterFinalizeError(
                        f"cannot mark Stream raster ready: {stream_id}"
                    )
            if not publication.run_streams.set_run_status(
                run_id,
                "raster_ready",
                expected=("planned", "running", "raster_ready"),
            ):
                raise RasterFinalizeError("cannot mark Run raster ready")
    finally:
        for staged_path, _canonical_path in staged_vrts:
            staged_path.unlink(missing_ok=True)
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
