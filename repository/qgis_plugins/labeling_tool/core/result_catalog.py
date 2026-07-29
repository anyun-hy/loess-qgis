"""Result-stream paths and persistent run-manifest updates."""

from __future__ import annotations

import datetime
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from .run_spec import atomic_write_json


TERMINAL_STATES = {"ready", "failed", "stopped"}
STREAM_STATES = {
    "pending", "running", "tiles_ready", "mosaicking", "polygonizing",
    "regularizing", "difference", "ready", "failed", "stopped",
}


def _stream_paths(run_dir: Path, kind: str, identifier: str) -> dict[str, str]:
    if kind == "model":
        permanent = run_dir / "models" / identifier
        temporary = run_dir / "tmp" / "streams" / identifier
    else:
        permanent = run_dir / "fusion" / identifier
        temporary = run_dir / "tmp" / "streams" / f"fusion_{identifier}"
    permanent.mkdir(parents=True, exist_ok=True)
    return {
        "tile_mask_dir": str(temporary / "masks"),
        "tile_confidence_dir": str(temporary / "confidence"),
        "tile_score_dir": str(temporary / "scores"),
        "stream_manifest": str(temporary / "stream_manifest.json"),
        "failed_jobs": str(temporary / "failed_jobs.json"),
        "mask_mosaic": str(permanent / "mask_mosaic.tif"),
        "confidence_mosaic": str(permanent / "confidence_mosaic.tif"),
        "probability_mosaic": str(temporary / "probability_mosaic.tif"),
        "semantic_polygons_raw": str(permanent / "semantic_polygons_raw.gpkg"),
        "semantic_polygons": str(permanent / "semantic_polygons.gpkg"),
        "boundary_regularization_report": str(
            permanent / "boundary_regularization_report.json"
        ),
        "difference_polygons": str(permanent / "semantic_candidates.gpkg"),
    }


def create_result_catalog(run_spec: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = Path(str(run_spec["run_dir"])).resolve()
    streams = []
    for model in run_spec.get("models") or []:
        model_id = str(model["model_id"])
        streams.append({
            "stream_id": f"model:{model_id}",
            "kind": "model",
            "model_id": model_id,
            "fusion_profile_id": "",
            "version": str(model.get("version") or ""),
            "status": "pending",
            "failure_count": 0,
            "error": "",
            "paths": _stream_paths(run_dir, "model", model_id),
        })
    fusion = run_spec.get("fusion")
    if fusion:
        profile_id = str(fusion["profile_id"])
        streams.append({
            "stream_id": f"fusion:{profile_id}",
            "kind": "fusion",
            "model_id": "",
            "fusion_profile_id": profile_id,
            "version": profile_id,
            "status": "pending",
            "failure_count": 0,
            "error": "",
            "paths": _stream_paths(run_dir, "fusion", profile_id),
        })
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    catalog = {
        "schema_version": 1,
        "run_id": run_spec["run_id"],
        "run_spec": str(run_dir / "run_spec.json"),
        "run_spec_sha256": _sha256(run_dir / "run_spec.json"),
        "status": "pending",
        "created_at": now,
        "updated_at": now,
        "streams": streams,
    }
    atomic_write_json(run_dir / "run_manifest.json", catalog)
    return catalog


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_VRT_METADATA_BLOCK = re.compile(
    rb"\n[ \t]*<Metadata(?:\s[^>]*)?>.*?</Metadata>", re.DOTALL
)
_VRT_METADATA_KEYS = re.compile(rb'<MDI\s+key="([^"]+)"')


def artifact_sha256(path: str | Path) -> str:
    """Hash an output while ignoring GDAL's mutable VRT statistics cache."""
    artifact = Path(path)
    if artifact.suffix.lower() != ".vrt":
        return _sha256(artifact)
    payload = artifact.read_bytes()

    def keep_semantic_metadata(match):
        keys = _VRT_METADATA_KEYS.findall(match.group(0))
        if keys and all(key.startswith(b"STATISTICS_") for key in keys):
            return b""
        return match.group(0)

    stable_payload = _VRT_METADATA_BLOCK.sub(keep_semantic_metadata, payload)
    return hashlib.sha256(stable_payload).hexdigest()


def record_stream_outputs(catalog: dict[str, Any], stream_id: str) -> dict[str, str]:
    stream = stream_by_id(catalog, stream_id)
    keys = (
        "mask_mosaic",
        "confidence_mosaic",
        "semantic_polygons_raw",
        "semantic_polygons",
        "boundary_regularization_report",
    )
    checksums = {}
    for key in keys:
        path = Path(stream["paths"][key])
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"missing final output for {stream_id}: {path}")
        checksums[key] = _sha256(path)
    with open(stream["paths"]["boundary_regularization_report"], "r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("status") != "passed" or (report.get("validation") or {}).get("passed") is not True:
        raise ValueError(f"boundary regularization report did not pass for {stream_id}")
    if report.get("input_sha256") != checksums["semantic_polygons_raw"]:
        raise ValueError(f"boundary regularization raw hash mismatch for {stream_id}")
    if report.get("output_sha256") != checksums["semantic_polygons"]:
        raise ValueError(f"boundary regularization formal hash mismatch for {stream_id}")
    probability_path = Path(stream["paths"]["probability_mosaic"])
    if not probability_path.is_file() or report.get("probability_mosaic_sha256") != _sha256(
        probability_path
    ):
        raise ValueError(f"subpixel probability mosaic hash mismatch for {stream_id}")
    stream["boundary_regularization_status"] = "passed"
    review_value = str(stream.get("review_polygons") or "").strip()
    review_path = Path(review_value) if review_value else None
    semantic_path = Path(stream["paths"]["semantic_polygons"])
    if review_path is not None and review_path != semantic_path:
        if not review_path.is_file() or review_path.stat().st_size <= 0:
            raise ValueError(f"missing review output for {stream_id}: {review_path}")
        checksums["review_polygons"] = _sha256(review_path)
    stream["output_sha256"] = checksums
    update_stream(catalog, stream_id, status="ready")
    return checksums


def _run_inputs_valid(run_spec: Mapping[str, Any]) -> bool:
    try:
        for model in run_spec.get("models") or []:
            if _sha256(model["artifact_path"]) != model.get("sha256"):
                return False
        for tile in run_spec.get("tiles") or []:
            if _sha256(tile["path"]) != tile.get("sha256"):
                return False
        fusion = run_spec.get("fusion")
        if fusion and _sha256(fusion["snapshot_path"]) != fusion.get("sha256"):
            return False
        accepted_path = str(run_spec.get("accepted_gpkg") or "")
        accepted_sha = str(run_spec.get("accepted_gpkg_sha256") or "")
        if accepted_path and accepted_sha and _sha256(accepted_path) != accepted_sha:
            return False
    except (KeyError, FileNotFoundError, OSError):
        return False
    return True


def valid_ready_stream_ids(catalog: Mapping[str, Any]) -> tuple[str, ...]:
    run_spec = Path(str(catalog.get("run_spec") or ""))
    if not run_spec.is_file() or _sha256(run_spec) != catalog.get("run_spec_sha256"):
        return ()
    try:
        with open(run_spec, "r", encoding="utf-8") as handle:
            spec = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return ()
    if int(catalog.get("schema_version") or 0) == 2:
        if int(spec.get("schema_version") or 0) != 2 or not _run_inputs_valid(spec):
            return ()
        valid = []
        for stream in catalog.get("streams") or []:
            if stream.get("status") != "ready" or stream.get("boundary_fitting_status") != "passed":
                continue
            paths = stream.get("paths") or {}
            checksums = stream.get("output_sha256") or {}
            required = (
                "mask_mosaic", "confidence_mosaic", "semantic_polygons_raw",
                "semantic_polygons", "boundary_fitting_report", "fitted_edges",
            )
            try:
                if any(
                    checksums.get(key) != artifact_sha256(paths[key])
                    for key in required
                ):
                    continue
                with open(paths["boundary_fitting_report"], "r", encoding="utf-8") as handle:
                    report = json.load(handle)
            except (KeyError, OSError, json.JSONDecodeError):
                continue
            if (
                report.get("status") == "passed"
                and (report.get("validation") or {}).get("passed") is True
                and report.get("input_sha256") == checksums["semantic_polygons_raw"]
                and report.get("output_sha256") == checksums["semantic_polygons"]
            ):
                valid.append(str(stream["stream_id"]))
        return tuple(valid)
    if not _run_inputs_valid(spec):
        return ()
    valid = []
    for stream in catalog.get("streams") or []:
        if stream.get("status") != "ready":
            continue
        checksums = stream.get("output_sha256") or {}
        expected_keys = [
            "mask_mosaic",
            "confidence_mosaic",
            "semantic_polygons_raw",
            "semantic_polygons",
            "boundary_regularization_report",
        ]
        review_path = str(stream.get("review_polygons") or "")
        semantic_path = str((stream.get("paths") or {}).get("semantic_polygons") or "")
        if review_path and review_path != semantic_path:
            expected_keys.append("review_polygons")
        try:
            matches = True
            for key in expected_keys:
                path = review_path if key == "review_polygons" else stream["paths"][key]
                if checksums.get(key) != artifact_sha256(path):
                    matches = False
                    break
        except (FileNotFoundError, OSError):
            matches = False
        if matches:
            try:
                with open(stream["paths"]["boundary_regularization_report"], "r", encoding="utf-8") as handle:
                    report = json.load(handle)
                matches = bool(
                    report.get("status") == "passed"
                    and (report.get("validation") or {}).get("passed") is True
                    and report.get("input_sha256") == checksums.get("semantic_polygons_raw")
                    and report.get("output_sha256") == checksums.get("semantic_polygons")
                )
            except (OSError, json.JSONDecodeError):
                matches = False
        if matches:
            valid.append(str(stream["stream_id"]))
    return tuple(valid)


def stream_by_id(catalog: Mapping[str, Any], stream_id: str) -> dict[str, Any]:
    matches = [item for item in catalog.get("streams") or [] if item.get("stream_id") == stream_id]
    if len(matches) != 1:
        raise ValueError(f"catalog must contain exactly one stream {stream_id!r}")
    return matches[0]


def update_stream(
    catalog: dict[str, Any],
    stream_id: str,
    *,
    status: str,
    failure_count: int | None = None,
    error: str | None = None,
    progress: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in STREAM_STATES:
        raise ValueError(f"invalid stream status: {status}")
    stream = stream_by_id(catalog, stream_id)
    stream["status"] = status
    if failure_count is not None:
        stream["failure_count"] = int(failure_count)
    if error is not None:
        stream["error"] = str(error)
    if progress is not None:
        stream["progress"] = dict(progress)
    catalog["updated_at"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    states = [item["status"] for item in catalog["streams"]]
    if states and all(state == "ready" for state in states):
        catalog["status"] = "ready"
    elif any(state == "running" for state in states):
        catalog["status"] = "running"
    elif any(state == "failed" for state in states):
        catalog["status"] = "failed"
    elif any(state == "stopped" for state in states):
        catalog["status"] = "stopped"
    else:
        catalog["status"] = "running"
    path = Path(catalog["run_spec"]).parent / "run_manifest.json"
    atomic_write_json(path, catalog)
    return stream


def load_result_catalog(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_ready_results(output_root: str | Path):
    """Yield valid ready runs, preferring resumable class workspaces."""
    runs_dir = Path(output_root).expanduser().resolve() / "runs"
    if not runs_dir.is_dir():
        return
    run_dirs = [path for path in runs_dir.iterdir() if path.is_dir()]
    run_dirs.sort(
        key=lambda path: (
            (path / "classes" / "workspace.json").is_file(),
            path.name,
        ),
        reverse=True,
    )
    for run_dir in run_dirs:
        manifest_path = run_dir / "run_manifest.json"
        spec_path = run_dir / "run_spec.json"
        if not manifest_path.is_file() or not spec_path.is_file():
            continue
        try:
            catalog = load_result_catalog(manifest_path)
            with open(spec_path, "r", encoding="utf-8") as handle:
                spec = json.load(handle)
            valid_ids = set(valid_ready_stream_ids(catalog))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        streams = list(catalog.get("streams") or [])
        ready_streams = [
            item for item in streams
            if item.get("status") == "ready"
            and str(item.get("stream_id") or "") in valid_ids
        ]
        if (
            catalog.get("status") != "ready"
            or not streams
            or len(ready_streams) != len(streams)
            or not any(item.get("kind") == "fusion" for item in ready_streams)
            or str(spec.get("run_id") or "") != str(catalog.get("run_id") or "")
        ):
            continue
        failed_streams = [item for item in streams if item.get("status") == "failed"]
        result = {
            "run_id": str(catalog.get("run_id") or ""),
            "run_dir": str(run_dir),
            "run_spec": str(spec_path),
            "run_manifest": str(manifest_path),
            "run_report": str(run_dir / "logs" / "run_report.json"),
            "success": True,
            "stopped": False,
            "status": "ready",
            "streams": streams,
            "ready_streams": ready_streams,
            "failed_streams": failed_streams,
            "error": "",
        }
        yield result, spec


def discover_ready_results(output_root: str | Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return list(iter_ready_results(output_root))
