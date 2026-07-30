#!/usr/bin/env python3
"""Validate the ownership identity of a Loess project before managed updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import uuid
from pathlib import Path, PurePosixPath
from typing import Mapping


IDENTITY_FILE = ".loess-project-id"
MANAGED_PATHS = ["inference_scripts", "runtime"]
PATHS = {
    "inference_scripts": "inference_scripts",
    "runtime": "runtime",
    "weights": "weights",
    "input_rasters": "input/rasters",
    "input_ranges": "input/ranges",
    "qgis": "qgis",
    "output": "output",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHARED_FILES = {
    "qgis_plugins/labeling_tool/core/run_spec.py",
    "qgis_plugins/labeling_tool/core/run_state_db.py",
    "qgis_plugins/labeling_tool/core/ownership_neighbors.py",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> Mapping[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {label} {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _valid_project_id(value: object) -> str:
    raw = str(value or "")
    try:
        parsed = uuid.UUID(raw)
    except ValueError as exc:
        raise ValueError("Project manifest project_id is not a canonical UUID") from exc
    if str(parsed) != raw:
        raise ValueError("Project manifest project_id is not a canonical UUID")
    return raw


def _validate_common(manifest: Mapping[str, object]) -> None:
    if manifest.get("deployment_kind") != "loess_project":
        raise ValueError("Refusing project with a foreign deployment manifest")
    if manifest.get("managed_paths") != MANAGED_PATHS:
        raise ValueError("Project managed_paths contract is invalid")
    if manifest.get("paths") != PATHS:
        raise ValueError("Project paths contract is invalid")


def _validate_inventory(root: Path, expected: object) -> None:
    if not isinstance(expected, Mapping) or not expected:
        raise ValueError("Legacy project has no complete inference inventory")
    actual: dict[str, str] = {}
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Legacy project inference_scripts is missing or unsafe")
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError(f"Legacy project contains a symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            actual[relative] = _sha256(path)
    normalized: dict[str, str] = {}
    for raw_relative, raw_digest in expected.items():
        relative = str(raw_relative)
        parsed = PurePosixPath(relative)
        digest = str(raw_digest)
        if (
            not relative
            or parsed.is_absolute()
            or any(part in {"", ".", ".."} for part in parsed.parts)
            or not SHA256_RE.fullmatch(digest)
        ):
            raise ValueError("Legacy project inference inventory is invalid")
        normalized[relative] = digest
    if actual != normalized:
        raise ValueError(
            "Legacy project managed files do not match its manifest; "
            "automatic ownership migration is refused"
        )


def _validate_legacy(root: Path, manifest: Mapping[str, object]) -> str:
    _validate_common(manifest)
    if not GIT_SHA_RE.fullmatch(str(manifest.get("git_sha") or "")):
        raise ValueError("Legacy project Git SHA is invalid")
    if manifest.get("platform") not in {"ubuntu", "macos"}:
        raise ValueError("Legacy project platform is invalid")
    declared_root = Path(str(manifest.get("project_root") or "")).expanduser()
    if not declared_root.is_absolute() or declared_root.resolve() != root:
        raise ValueError(
            "Legacy project root identity does not match; automatic migration is refused"
        )
    _validate_inventory(root / "inference_scripts", manifest.get("inference_files"))

    launcher = manifest.get("launcher")
    if not isinstance(launcher, Mapping):
        raise ValueError("Legacy project launcher contract is missing")
    launcher_path = root / "runtime" / "loess_launcher.sh"
    if (
        launcher.get("path") != "runtime/loess_launcher.sh"
        or not launcher_path.is_file()
        or launcher_path.is_symlink()
        or _sha256(launcher_path) != launcher.get("sha256")
    ):
        raise ValueError("Legacy project launcher does not match its manifest")

    shared = manifest.get("shared_runtime")
    files = shared.get("files") if isinstance(shared, Mapping) else None
    if not isinstance(files, Mapping) or set(files) != SHARED_FILES:
        raise ValueError("Legacy project shared runtime contract is missing")
    aggregate = hashlib.sha256()
    for canonical, raw_digest in sorted(files.items()):
        digest = str(raw_digest)
        path = root / "runtime" / "labeling_tool" / "core" / Path(str(canonical)).name
        if (
            not SHA256_RE.fullmatch(digest)
            or not path.is_file()
            or path.is_symlink()
            or _sha256(path) != digest
        ):
            raise ValueError("Legacy project shared runtime does not match its manifest")
        aggregate.update(str(canonical).encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    if aggregate.hexdigest() != shared.get("sha256"):
        raise ValueError("Legacy project shared runtime aggregate is invalid")
    return str(uuid.uuid4())


def inspect_project(root: Path, *, allow_rebind: bool) -> dict[str, object]:
    root = root.resolve()
    manifest_path = root / "project_manifest.json"
    identity_path = root / IDENTITY_FILE
    if not manifest_path.exists():
        if identity_path.exists():
            raise ValueError("Project identity marker exists without a manifest")
        if (root / "inference_scripts").exists() or (root / "runtime").exists():
            raise ValueError(
                "Refusing to overwrite unmanaged inference_scripts/runtime "
                "without a project manifest"
            )
        return {
            "status": "new",
            "project_id": str(uuid.uuid4()),
            "create_identity": True,
            "conda_executable": "",
            "conda_environment": "",
        }

    manifest = _read_json(manifest_path, "project manifest")
    schema = manifest.get("schema_version")
    if schema == 1:
        project_id = _validate_legacy(root, manifest)
        status = "legacy_migration"
        create_identity = True
    elif schema == 2:
        _validate_common(manifest)
        project_id = _valid_project_id(manifest.get("project_id"))
        marker = _read_json(identity_path, "project identity marker")
        if (
            marker.get("schema_version") != 1
            or marker.get("deployment_kind") != "loess_project_identity"
            or marker.get("project_id") != project_id
        ):
            raise ValueError("Project identity marker does not match the manifest")
        declared_root = Path(str(manifest.get("project_root") or "")).expanduser()
        if not declared_root.is_absolute() or declared_root.resolve() != root:
            if not allow_rebind:
                raise ValueError(
                    "Project was moved; rerun with --rebind-project-root "
                    "after verifying this is the intended project"
                )
            status = "rebind"
        else:
            status = "existing"
        create_identity = False
    else:
        raise ValueError(
            "Project manifest schema is unsupported; a deployment_kind field "
            "alone is not ownership evidence"
        )

    launcher = manifest.get("launcher")
    launcher = launcher if isinstance(launcher, Mapping) else {}
    return {
        "status": status,
        "project_id": project_id,
        "create_identity": create_identity,
        "conda_executable": str(launcher.get("conda_executable") or ""),
        "conda_environment": str(launcher.get("conda_environment") or ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--allow-rebind", action="store_true")
    args = parser.parse_args()
    try:
        result = inspect_project(
            args.project_root,
            allow_rebind=args.allow_rebind,
        )
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
