"""Freeze one formal SAM3 class job from a ready E9 semantic stream."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fiona


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = ROOT / "inference_scripts"
PLUGIN_ROOT = ROOT / "qgis_plugins"
for path in (SCRIPTS_ROOT, PLUGIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deployment_config import load_and_validate_config  # noqa: E402
from labeling_tool.core.run_spec import atomic_write_json  # noqa: E402


def build_job(run_spec_path: Path, stream_id: str, class_code: int, device: str) -> Path:
    run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
    run_dir = Path(run_spec["run_dir"]).resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    streams = [item for item in manifest["streams"] if item["stream_id"] == stream_id]
    if len(streams) != 1 or streams[0].get("status") != "ready":
        raise RuntimeError(f"source stream is not uniquely ready: {stream_id}")
    stream = streams[0]
    source_gpkg = Path(stream["paths"]["semantic_polygons"]).resolve()

    keys = []
    with fiona.open(source_gpkg, layer="semantic_polygons") as source:
        for feature in source:
            properties = feature["properties"]
            if int(properties["class_code"]) == int(class_code):
                keys.append({
                    "object_id": str(properties["object_id"]),
                    "part_id": str(properties.get("part_id") or "000"),
                })
    if not keys:
        raise RuntimeError(f"source stream has no objects for class {class_code}")

    effective, issues = load_and_validate_config(
        SCRIPTS_ROOT / "config.yaml",
        verify_files=True,
        verify_hashes=True,
    )
    if issues:
        details = "; ".join(f"{item.path}: {item.message}" for item in issues)
        raise RuntimeError(f"formal deployment config is invalid: {details}")
    sam = effective["sam3"]
    if not sam.get("enabled"):
        raise RuntimeError("SAM3 is disabled")

    safe_stream = stream_id.replace(":", "_")
    directory = run_dir / "refinement" / safe_stream
    output = directory / f"class_{int(class_code)}.gpkg"
    job_path = directory / f"class_{int(class_code)}_job.json"
    job = {
        "schema_version": 1,
        "run_id": run_spec["run_id"],
        "raster": run_spec["raster"]["path"],
        "source_stream_id": stream_id,
        "source_gpkg": str(source_gpkg),
        "source_layer": "semantic_polygons",
        "source_confidence": stream["paths"]["confidence_mosaic"],
        "class_code": int(class_code),
        "object_keys": keys,
        "buffer_px": int(sam["buffer_px"]),
        "device": device,
        "checkpoint": sam["checkpoint"],
        "checkpoint_sha256": sam["checkpoint_sha256"],
        "sam_version": sam["version"],
        "output": str(output),
    }
    atomic_write_json(job_path, job)
    return job_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-spec", required=True)
    parser.add_argument("--stream-id", required=True)
    parser.add_argument("--class-code", required=True, type=int)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    args = parser.parse_args()
    path = build_job(
        Path(args.run_spec).expanduser().resolve(),
        args.stream_id,
        args.class_code,
        args.device,
    )
    print(json.dumps({"job_spec": str(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
