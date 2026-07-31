#!/usr/bin/env python3
"""Build and verify provenance for the exact source bundle being deployed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath


SCHEMA_VERSION = 1
ARCHIVE_MANIFEST = "source_manifest.json"
SOURCE_ROOTS = (
    "bash",
    "inference_scripts",
    "qgis_plugins/labeling_tool",
)
IGNORED_NAMES = {".DS_Store"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _ignored(path: Path) -> bool:
    return (
        "__pycache__" in path.parts
        or path.name in IGNORED_NAMES
        or path.name.startswith("._")
        or path.suffix == ".pyc"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_inventory(source_root: Path) -> dict[str, dict[str, object]]:
    inventory: dict[str, dict[str, object]] = {}
    for relative_root in SOURCE_ROOTS:
        root = source_root / relative_root
        if not root.is_dir() or root.is_symlink():
            raise ValueError(f"Missing or unsafe source root: {root}")
        for path in sorted(root.rglob("*")):
            if _ignored(path):
                continue
            if path.is_symlink():
                raise ValueError(f"Source bundle contains a symlink: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(source_root).as_posix()
            parsed = PurePosixPath(relative)
            if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
                raise ValueError(f"Invalid source path: {relative}")
            mode = path.stat().st_mode
            inventory[relative] = {
                "sha256": _sha256_file(path),
                "executable": bool(mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)),
            }
    if not inventory:
        raise ValueError("Source bundle is empty")
    return inventory


def inventory_digest(inventory: dict[str, dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for relative, item in sorted(inventory.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item["sha256"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(b"x" if item["executable"] else b"-")
        digest.update(b"\n")
    return digest.hexdigest()


def _git(source_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise ValueError(result.stderr.strip() or "Git command failed")
    return result.stdout.strip()


def _is_git_checkout(source_root: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "--is-inside-work-tree"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def inspect_source(source_root: Path, *, allow_dirty: bool) -> dict[str, object]:
    source_root = source_root.resolve()
    inventory = source_inventory(source_root)
    bundle_sha = inventory_digest(inventory)

    if _is_git_checkout(source_root):
        git_sha = _git(source_root, "rev-parse", "HEAD")
        if not GIT_SHA_RE.fullmatch(git_sha):
            raise ValueError(f"Invalid Git SHA: {git_sha!r}")
        status = _git(
            source_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *SOURCE_ROOTS,
        )
        dirty = bool(status)
        if dirty and not allow_dirty:
            raise ValueError(
                "Deployable source files differ from Git HEAD; commit them first "
                "or use --allow-dirty for an explicitly marked development deployment"
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "git_worktree",
            "git_sha": git_sha,
            "git_dirty": dirty,
            "source_bundle_sha256": bundle_sha,
            "source_file_count": len(inventory),
        }

    manifest_path = source_root / ARCHIVE_MANIFEST
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(
            f"Archive deployment requires a regular {ARCHIVE_MANIFEST}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid archive source manifest: {exc}") from exc
    git_sha = str(manifest.get("git_sha") or "")
    expected = str(manifest.get("source_bundle_sha256") or "")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("deployment_kind") != "loess_source_archive"
        or not GIT_SHA_RE.fullmatch(git_sha)
        or not SHA256_RE.fullmatch(expected)
    ):
        raise ValueError("Archive source manifest contract is invalid")
    if expected != bundle_sha:
        raise ValueError(
            "Archive source bundle SHA256 mismatch: "
            f"expected={expected}, actual={bundle_sha}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "release_archive",
        "git_sha": git_sha,
        "git_dirty": False,
        "source_bundle_sha256": bundle_sha,
        "source_file_count": len(inventory),
    }


def create_archive_manifest(source_root: Path, output: Path) -> None:
    source_root = source_root.resolve()
    if not _is_git_checkout(source_root):
        raise ValueError("Archive manifest must be created from a Git checkout")
    provenance = inspect_source(source_root, allow_dirty=False)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "deployment_kind": "loess_source_archive",
        "git_sha": provenance["git_sha"],
        "source_bundle_sha256": provenance["source_bundle_sha256"],
        "source_file_count": provenance["source_file_count"],
    }
    output = output.resolve()
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--source-root", required=True, type=Path)
    inspect_parser.add_argument("--allow-dirty", action="store_true")
    create_parser = subparsers.add_parser("create-archive-manifest")
    create_parser.add_argument("--source-root", required=True, type=Path)
    create_parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            payload = inspect_source(
                args.source_root,
                allow_dirty=args.allow_dirty,
            )
            print(json.dumps(payload, sort_keys=True))
        else:
            create_archive_manifest(args.source_root, args.output)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
