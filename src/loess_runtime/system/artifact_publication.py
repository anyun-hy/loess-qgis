"""Publish one verified runtime file through the Artifact repository."""

from __future__ import annotations

from pathlib import Path

from labeling_tool.shared.contracts.run_spec import sha256_file
from labeling_tool.shared.state.artifact_repository import ArtifactRepository
from loess_runtime.system.runtime_errors import WorkPackageRuntimeError


def publish_artifact(
    artifacts: ArtifactRepository,
    run_id: str,
    *,
    path: Path,
    kind: str,
    stream_id: str,
    unit_id: str,
) -> int:
    """Publish one immutable file while preserving kind-specific transactions."""
    if kind == "partition_probability":
        return artifacts.publish_partition_artifact(
            run_id,
            stream_id,
            unit_id,
            path,
            byte_count=path.stat().st_size,
            sha256=sha256_file(path),
        )
    if kind == "v3_context_core":
        return artifacts.publish_fragmentation_v33_context(
            run_id,
            stream_id,
            unit_id,
            path,
            byte_count=path.stat().st_size,
            sha256=sha256_file(path),
        )
    if kind == "v3_baseline_core":
        return artifacts.publish_fragmentation_v33_baseline_core(
            run_id,
            stream_id,
            unit_id,
            path,
            byte_count=path.stat().st_size,
            sha256=sha256_file(path),
        )
    artifact_id = artifacts.register_artifact(
        run_id,
        kind,
        path,
        stream_id=stream_id,
        unit_id=unit_id,
    )
    existing = artifacts.get_artifact(artifact_id)
    if existing and existing["status"] == "ready":
        if existing["byte_count"] == path.stat().st_size and existing[
            "sha256"
        ] == sha256_file(path):
            return artifact_id
        raise WorkPackageRuntimeError(f"ready Artifact changed on disk: {path}")
    if not artifacts.mark_artifact_ready(
        artifact_id,
        byte_count=path.stat().st_size,
        sha256=sha256_file(path),
    ):
        raise WorkPackageRuntimeError(f"cannot commit Artifact: {path}")
    return artifact_id
