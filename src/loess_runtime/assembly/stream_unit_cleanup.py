"""Crash-safe cleanup of owned Stream unit intermediate artifacts."""

from __future__ import annotations

import fcntl
import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from labeling_tool.shared.contracts.run_spec import sha256_file
from loess_runtime.assembly.assembly_errors import StreamAssemblyError

if TYPE_CHECKING:
    from labeling_tool.shared.state.artifact_repository import ArtifactRepository


UNIT_INTERMEDIATE_KINDS = (
    "unit_raw_geoparquet",
    "unit_formal_geoparquet",
    "unit_boundary_report",
    "unit_fitted_edges_geoparquet",
    "unit_boundary_signatures",
)
UNIT_INTERMEDIATE_SUFFIXES = {
    "unit_raw_geoparquet": "_raw.parquet",
    "unit_formal_geoparquet": "_formal.parquet",
    "unit_boundary_report": "_report.json",
    "unit_fitted_edges_geoparquet": "_fitted_edges.parquet",
    "unit_boundary_signatures": "_boundary_signatures.json",
}


class AppendCleanupEvent(Protocol):
    """Persist the final cleanup observation on the Run event stream."""

    def __call__(
        self,
        run_id: str,
        event_type: str,
        *,
        message: str,
        payload: Mapping[str, Any],
    ) -> int: ...


def _strict_path_component(value: str, *, label: str) -> str:
    """Return one safe filesystem component without normalising traversal."""

    component = str(value)
    if (
        not component
        or component in {".", ".."}
        or "/" in component
        or "\\" in component
        or "\x00" in component
        or Path(component).name != component
    ):
        raise StreamAssemblyError(
            f"unsafe {label} in unit Artifact metadata: {value!r}"
        )
    return component


def _cleanup_tombstone(path: Path, artifact_id: int) -> Path:
    return path.with_name(f".{path.name}.cleanup-{int(artifact_id)}.tombstone")


def _rename_cleanup_file(source: Path, tombstone: Path) -> None:
    os.rename(source, tombstone)


def _unlink_cleanup_tombstone(tombstone: Path) -> None:
    tombstone.unlink()


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes used by the cleanup transaction."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _assert_regular_cleanup_file(
    path: Path,
    artifact: Mapping[str, Any],
    *,
    stage: str,
) -> None:
    if path.is_symlink() or not path.is_file():
        raise StreamAssemblyError(
            f"unit intermediate is not a regular file during {stage}: {path}"
        )
    if path.stat().st_size != int(artifact["byte_count"]) or sha256_file(path) != str(
        artifact["sha256"]
    ):
        raise StreamAssemblyError(f"unit intermediate changed during {stage}: {path}")


@contextmanager
def _unit_cleanup_lock(run_dir: Path) -> Iterator[None]:
    """Serialize cleanup recovery without following a forged lock symlink."""

    tmp_root = run_dir / "tmp"
    if tmp_root.is_symlink():
        raise StreamAssemblyError(
            f"refusing symlinked Run temporary directory: {tmp_root}"
        )
    tmp_root.mkdir(parents=True, exist_ok=True)
    lock_path = tmp_root / ".unit-artifact-cleanup.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise StreamAssemblyError(
            f"cannot open the unit cleanup lock safely: {lock_path}"
        ) from error
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def cleanup_stream_unit_artifacts(
    *,
    run_id: str,
    run_dir: Path,
    stream_id: str,
    artifacts: ArtifactRepository,
    append_event: AppendCleanupEvent,
) -> dict[str, Any]:
    """Delete only verified, unreferenced unit intermediates after assembly."""

    resolved_run_dir = Path(run_dir).resolve()
    with _unit_cleanup_lock(resolved_run_dir):
        return _cleanup_stream_unit_artifacts_locked(
            run_id=str(run_id),
            run_dir=resolved_run_dir,
            stream_id=str(stream_id),
            artifacts=artifacts,
            append_event=append_event,
        )


def _cleanup_stream_unit_artifacts_locked(
    *,
    run_id: str,
    run_dir: Path,
    stream_id: str,
    artifacts: ArtifactRepository,
    append_event: AppendCleanupEvent,
) -> dict[str, Any]:
    """Resume each committed filesystem and Artifact-state cleanup window."""

    stream_component = _strict_path_component(
        str(stream_id).replace(":", "_"), label="stream_id"
    )
    tmp_root = run_dir / "tmp"
    output_root = tmp_root / "unit_outputs"
    unit_root = output_root / stream_component
    for owned_directory in (tmp_root, output_root, unit_root):
        if owned_directory.is_symlink():
            raise StreamAssemblyError(
                "refusing to clean through a symlinked unit output directory: "
                f"{owned_directory}"
            )
    candidates = [
        dict(artifact)
        for artifact in artifacts.artifacts_for_stream(run_id, stream_id, status=None)
        if str(artifact.get("kind") or "") in UNIT_INTERMEDIATE_KINDS
        and str(artifact.get("status") or "") in {"ready", "cleaning", "cleaned"}
        and int(artifact.get("ref_count") or 0) == 0
    ]

    validated: list[tuple[dict[str, Any], Path, Path]] = []
    for artifact in candidates:
        kind = str(artifact["kind"])
        unit_id = _strict_path_component(str(artifact["unit_id"]), label="unit_id")
        path = Path(str(artifact["path"]))
        owner = unit_root
        if path.parent.parent == unit_root and re.fullmatch(
            r"attempt_[0-9a-f]{32}", path.parent.name
        ):
            owner = path.parent
            if owner.is_symlink():
                raise StreamAssemblyError(
                    f"refusing symlinked attempt directory: {owner}"
                )
        expected = owner / f"{unit_id}{UNIT_INTERMEDIATE_SUFFIXES[kind]}"
        if not path.is_absolute() or path != expected or path.parent != owner:
            raise StreamAssemblyError(
                "unit intermediate cleanup path is not an exact direct child "
                f"of the owned Stream directory: {path}"
            )
        artifact_id = int(artifact["artifact_id"])
        tombstone = _cleanup_tombstone(path, artifact_id)
        manifest = (
            path.with_name(f"{path.name}.manifest.json")
            if kind.endswith("_geoparquet")
            else None
        )
        if manifest is not None and manifest.is_symlink():
            raise StreamAssemblyError(
                f"refusing symlinked GeoParquet manifest during cleanup: {manifest}"
            )
        original_present = path.is_symlink() or path.exists()
        tombstone_present = tombstone.is_symlink() or tombstone.exists()
        if original_present and tombstone_present:
            raise StreamAssemblyError(
                "unit intermediate and cleanup tombstone both exist; refusing "
                f"ambiguous deletion: {path}"
            )
        status = str(artifact["status"])
        if status == "ready":
            if tombstone_present:
                raise StreamAssemblyError(
                    f"unclaimed cleanup tombstone already exists: {tombstone}"
                )
            _assert_regular_cleanup_file(path, artifact, stage="pre-claim validation")
            if manifest is not None and not manifest.is_file():
                raise StreamAssemblyError(
                    f"GeoParquet manifest is missing during cleanup: {manifest}"
                )
        elif status == "cleaned":
            if original_present:
                raise StreamAssemblyError(
                    f"cleaned unit intermediate unexpectedly reappeared: {path}"
                )
            if not tombstone_present:
                if manifest is not None and manifest.exists():
                    validated.append((artifact, path, tombstone))
                continue
            _assert_regular_cleanup_file(
                tombstone, artifact, stage="post-commit tombstone recovery"
            )
        elif original_present:
            _assert_regular_cleanup_file(path, artifact, stage="cleanup recovery")
        elif tombstone_present:
            _assert_regular_cleanup_file(
                tombstone, artifact, stage="tombstone recovery"
            )
        validated.append((artifact, path, tombstone))

    kind_counts: dict[str, int] = {}
    cleaned_bytes = 0
    for artifact, path, tombstone in validated:
        kind = str(artifact["kind"])
        artifact_id = int(artifact["artifact_id"])
        claimed = artifact
        if str(artifact["status"]) == "ready":
            claim = artifacts.claim_artifact_cleanup(artifact_id)
            if claim is None:
                raise StreamAssemblyError(
                    f"unit intermediate cleanup claim changed: {path}"
                )
            claimed = claim
        current = artifacts.get_artifact(artifact_id)
        current_status = str((current or {}).get("status") or "")
        if (
            current is None
            or current_status not in {"cleaning", "cleaned"}
            or int(current["ref_count"]) != 0
        ):
            raise StreamAssemblyError(
                f"unit intermediate cleanup state changed after claim: {path}"
            )

        if path.is_symlink() or tombstone.is_symlink():
            raise StreamAssemblyError(
                f"refusing symlink during unit intermediate cleanup: {path}"
            )
        if current_status == "cleaned" and path.exists():
            raise StreamAssemblyError(
                f"cleaned unit intermediate unexpectedly reappeared: {path}"
            )
        if current_status == "cleaning" and path.exists():
            if tombstone.exists():
                raise StreamAssemblyError(
                    f"cleanup tombstone appeared before rename: {tombstone}"
                )
            _assert_regular_cleanup_file(path, claimed, stage="post-claim validation")
            _rename_cleanup_file(path, tombstone)
            _fsync_directory(unit_root)
        if tombstone.is_symlink():
            raise StreamAssemblyError(
                f"refusing symlinked cleanup tombstone: {tombstone}"
            )
        if current_status == "cleaning" and not artifacts.finish_artifact_cleanup(
            artifact_id, success=True
        ):
            raise StreamAssemblyError(
                f"unit intermediate cleanup state changed: {path}"
            )
        if kind.endswith("_geoparquet"):
            manifest = path.with_name(f"{path.name}.manifest.json")
            if manifest.is_symlink():
                raise StreamAssemblyError(
                    f"refusing symlinked GeoParquet manifest during cleanup: {manifest}"
                )
            if manifest.exists():
                if not manifest.is_file():
                    raise StreamAssemblyError(
                        f"GeoParquet manifest is unsafe during cleanup: {manifest}"
                    )
                manifest.unlink()
                _fsync_directory(unit_root)
        if tombstone.exists():
            _assert_regular_cleanup_file(
                tombstone, claimed, stage="pre-unlink validation"
            )
            _unlink_cleanup_tombstone(tombstone)
            _fsync_directory(unit_root)
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        cleaned_bytes += int(artifact["byte_count"])
    try:
        unit_root.rmdir()
    except (FileNotFoundError, OSError):
        pass
    report = {
        "status": "passed",
        "stream_id": str(stream_id),
        "artifact_count": len(validated),
        "cleaned_bytes": cleaned_bytes,
        "kind_counts": dict(sorted(kind_counts.items())),
        "path_policy": "strict_direct_child_no_symlink",
        "integrity_policy": "db_claim_tombstone_size_sha256",
    }
    append_event(
        run_id,
        "stream_unit_artifacts_cleaned",
        message=str(stream_id),
        payload=report,
    )
    return report
