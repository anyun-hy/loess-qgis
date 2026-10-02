"""Shared Artifact lookup and exact file I/O for V3.3 execution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, TypeAlias

from labeling_tool.shared.contracts.run_spec import sha256_file
from labeling_tool.shared.state.artifact_repository import ArtifactRepository
from loess_runtime.geometry.fragmentation_v33_contract import (
    FragmentationV33WorkPackageError,
)

ArtifactIndex: TypeAlias = dict[tuple[str, str], dict[str, Any]]

__all__ = [
    "ArtifactIndex",
    "ready_artifact_index",
    "verified_artifact_path",
    "write_atomic_json",
]


def ready_artifact_index(
    artifacts: ArtifactRepository, run_id: str, stream_id: str
) -> ArtifactIndex:
    return {
        (str(item["unit_id"]), str(item["kind"])): dict(item)
        for item in artifacts.artifacts_for_stream(run_id, stream_id, status="ready")
    }


def verified_artifact_path(
    artifact: Mapping[str, Any] | None,
    *,
    kind: str,
    partition_id: str,
    verified: set[tuple[str, int, str]] | None = None,
) -> Path:
    if artifact is None:
        raise FragmentationV33WorkPackageError(
            f"missing {kind} Artifact for {partition_id}"
        )
    path = Path(str(artifact.get("path") or ""))
    expected_size = int(artifact.get("byte_count") or -1)
    expected_sha = str(artifact.get("sha256") or "")
    key = (str(path), expected_size, expected_sha)
    if key in (verified or set()):
        return path
    if (
        not path.is_file()
        or path.stat().st_size != expected_size
        or sha256_file(path) != expected_sha
    ):
        raise FragmentationV33WorkPackageError(
            f"changed {kind} Artifact for {partition_id}: {path}"
        )
    if verified is not None:
        verified.add(key)
    return path


def write_atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
