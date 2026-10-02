"""Durable transport helpers for one QGIS environment-check result."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping


REPORT_SCHEMA_VERSION = 1


def environment_report_path(output_dir: str) -> str:
    """Return the stable diagnostic path for an output workspace."""

    value = str(output_dir or "").strip()
    if not value:
        return ""
    root = Path(value).expanduser().resolve()
    return str(root / "cache" / "environment_check" / "latest.json")


def _matches_check(value: object, expected_check_id: str) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("schema_version") == REPORT_SCHEMA_VERSION
        and (
            not expected_check_id
            or str(value.get("check_id") or "") == expected_check_id
        )
    )


def load_environment_report(
    stdout: str,
    report_path: str,
    expected_check_id: str,
) -> tuple[dict[str, object] | None, str]:
    """Recover the current report from stdout or its atomic disk copy.

    The check ID prevents a delayed Qt signal or a stale ``latest.json`` from
    being mistaken for the result of the active QProcess.
    """

    for line in reversed(str(stdout or "").splitlines()):
        try:
            candidate = json.loads(line.strip())
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if _matches_check(candidate, expected_check_id):
            return dict(candidate), "stdout"

    path = Path(report_path) if report_path else None
    if path is None or not path.is_file() or path.is_symlink():
        return None, ""
    try:
        candidate = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None, ""
    if not _matches_check(candidate, expected_check_id):
        return None, ""
    return dict(candidate), "report_file"


def persist_environment_report(
    report_path: str,
    report: Mapping[str, object],
) -> None:
    """Atomically persist the report without following a target symlink."""

    if not report_path:
        return
    path = Path(report_path)
    if path.is_symlink():
        raise OSError(f"Refusing environment report symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(
                dict(report),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass
