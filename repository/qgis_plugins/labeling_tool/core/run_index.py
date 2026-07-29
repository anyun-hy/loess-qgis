"""Constant-cost startup lookup for the most recent v5 runs.

The index deliberately stores run identifiers rather than artifact paths.
QGIS startup may read this small file and the exact metadata files it names,
but it must never discover runs by walking ``output/runs``.
"""

from __future__ import annotations

import datetime
import json
import stat
from pathlib import Path
from typing import Any

from .run_spec import RUN_ID_PATTERN, atomic_write_json, sha256_file


RUN_INDEX_FILENAME = "run_index.json"
RUN_INDEX_SCHEMA_VERSION = 1
RUN_INDEX_MAX_BYTES = 64 * 1024
RUN_SPEC_MAX_BYTES = 2 * 1024 * 1024
RUN_MANIFEST_MAX_BYTES = 8 * 1024 * 1024
RUN_STATES = {"planned", "running", "stopped", "failed", "ready"}
RECOVERABLE_RUN_STATES = {"planned", "running", "stopped", "failed"}


class RunIndexError(ValueError):
    pass


def _read_bounded_json(path: Path, *, maximum_bytes: int) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RunIndexError(f"startup metadata is unavailable: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RunIndexError(f"startup metadata must be a regular file: {path}")
    if metadata.st_size <= 0 or metadata.st_size > int(maximum_bytes):
        raise RunIndexError(
            f"startup metadata size is outside the bounded limit: {path}"
        )
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RunIndexError(f"startup metadata is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RunIndexError(f"startup metadata must contain a JSON object: {path}")
    return value


def _load_index(output_root: Path) -> dict[str, Any]:
    path = output_root / RUN_INDEX_FILENAME
    value = _read_bounded_json(path, maximum_bytes=RUN_INDEX_MAX_BYTES)
    if int(value.get("schema_version") or 0) != RUN_INDEX_SCHEMA_VERSION:
        raise RunIndexError("unsupported run index schema")
    return value


def record_run_state(
    output_root: str | Path,
    run_id: str,
    *,
    status: str,
) -> Path:
    """Atomically record one Run without enumerating any output directory."""
    output = Path(output_root).expanduser().resolve()
    identifier = str(run_id)
    state = str(status)
    if not RUN_ID_PATTERN.fullmatch(identifier):
        raise RunIndexError(f"invalid indexed run_id: {identifier!r}")
    if state not in RUN_STATES:
        raise RunIndexError(f"invalid indexed Run state: {state!r}")

    try:
        current = _load_index(output)
    except RunIndexError:
        current = {}

    previous_latest = str(current.get("latest_run_id") or "")
    if not RUN_ID_PATTERN.fullmatch(previous_latest):
        previous_latest = ""
    previous_ready = str(current.get("latest_ready_run_id") or "")
    if not RUN_ID_PATTERN.fullmatch(previous_ready):
        previous_ready = ""

    latest_run_id = max(previous_latest, identifier)
    if latest_run_id == identifier:
        latest_run_status = state
    else:
        latest_run_status = str(current.get("latest_run_status") or "")
        if latest_run_status not in RUN_STATES:
            latest_run_status = "planned"

    latest_ready_run_id = previous_ready
    if state == "ready":
        latest_ready_run_id = max(previous_ready, identifier)

    value = {
        "schema_version": RUN_INDEX_SCHEMA_VERSION,
        "latest_run_id": latest_run_id,
        "latest_run_status": latest_run_status,
        "latest_ready_run_id": latest_ready_run_id,
        "updated_at": datetime.datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
    }
    path = output / RUN_INDEX_FILENAME
    atomic_write_json(path, value)
    return path


def _candidate(
    output_root: Path,
    run_id: str,
    *,
    require_manifest: bool,
    indexed_status: str,
) -> dict[str, Any]:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise RunIndexError(f"invalid indexed run_id: {run_id!r}")
    runs_dir = output_root / "runs"
    run_dir = runs_dir / run_id
    try:
        run_metadata = run_dir.lstat()
    except OSError as exc:
        raise RunIndexError(f"indexed Run is unavailable: {run_id}") from exc
    if stat.S_ISLNK(run_metadata.st_mode) or not stat.S_ISDIR(run_metadata.st_mode):
        raise RunIndexError(f"indexed Run must be a regular directory: {run_id}")

    spec_path = run_dir / "run_spec.json"
    spec = _read_bounded_json(spec_path, maximum_bytes=RUN_SPEC_MAX_BYTES)
    if (
        int(spec.get("schema_version") or 0) != 2
        or str(spec.get("run_id") or "") != run_id
        or Path(str(spec.get("run_dir") or "")).expanduser().resolve() != run_dir
        or Path(str(spec.get("output_root") or "")).expanduser().resolve()
        != output_root
    ):
        raise RunIndexError(f"indexed run_spec identity mismatch: {run_id}")

    candidate = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "run_spec_path": str(spec_path),
        "run_manifest_path": str(run_dir / "run_manifest.json"),
        "indexed_status": indexed_status,
        "spec": spec,
    }
    if not require_manifest:
        return candidate

    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_bounded_json(
        manifest_path, maximum_bytes=RUN_MANIFEST_MAX_BYTES
    )
    if (
        int(manifest.get("schema_version") or 0) != 2
        or str(manifest.get("run_id") or "") != run_id
        or str(manifest.get("status") or "") != "ready"
        or manifest.get("success") is not True
        or str(manifest.get("run_spec_sha256") or "") != sha256_file(spec_path)
    ):
        raise RunIndexError(f"indexed ready manifest identity mismatch: {run_id}")
    candidate["manifest"] = manifest
    return candidate


def load_startup_candidates(output_root: str | Path) -> dict[str, dict[str, Any]]:
    """Read at most two exact Run metadata bundles with no directory listing."""
    output = Path(output_root).expanduser().resolve()
    try:
        index = _load_index(output)
    except RunIndexError:
        return {}

    latest_id = str(index.get("latest_run_id") or "")
    latest_status = str(index.get("latest_run_status") or "")
    ready_id = str(index.get("latest_ready_run_id") or "")
    result: dict[str, dict[str, Any]] = {}

    if latest_status in RUN_STATES:
        try:
            result["latest"] = _candidate(
                output,
                latest_id,
                require_manifest=False,
                indexed_status=latest_status,
            )
        except RunIndexError:
            pass

    if ready_id:
        try:
            result["latest_ready"] = _candidate(
                output,
                ready_id,
                require_manifest=True,
                indexed_status="ready",
            )
        except RunIndexError:
            pass
    return result


def lightweight_ready_result(candidate: dict[str, Any]) -> dict[str, Any]:
    """Build UI metadata without opening or hashing any output artifact."""
    spec = candidate["spec"]
    manifest = candidate["manifest"]
    streams = list(manifest.get("ready_streams") or manifest.get("streams") or [])
    if not streams or any(item.get("status") != "ready" for item in streams):
        raise RunIndexError("indexed ready Run does not list only ready streams")
    fusion = spec.get("fusion") or {}
    expected_fusion = f"fusion:{fusion.get('profile_id')}" if fusion else ""
    if not any(
        item.get("stream_id") == expected_fusion
        and item.get("kind") == "fusion"
        and item.get("boundary_fitting_status") == "passed"
        for item in streams
    ):
        raise RunIndexError("indexed ready Run has no eligible Fusion metadata")
    return {
        **manifest,
        "run_id": candidate["run_id"],
        "run_dir": candidate["run_dir"],
        "run_spec": candidate["run_spec_path"],
        "run_manifest": candidate["run_manifest_path"],
        "run_report": str(Path(candidate["run_dir"]) / "logs" / "run_report.json"),
        "success": True,
        "stopped": False,
        "status": "ready",
        "streams": streams,
        "ready_streams": streams,
        "failed_streams": [],
        "error": "",
    }
