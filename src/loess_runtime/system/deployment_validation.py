"""Shared primitives for Schema-v2 deployment validation."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ValidationIssue:
    """One stable path-addressed deployment validation result."""

    path: str
    message: str
    code: str = "invalid"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def as_mapping(
    value: Any,
    path: str,
    issues: list[ValidationIssue],
) -> Mapping[str, Any]:
    """Return a mapping or append the established type issue."""

    if isinstance(value, Mapping):
        return value
    issues.append(ValidationIssue(path, "must be a mapping", "type"))
    return {}


def as_list(value: Any, path: str, issues: list[ValidationIssue]) -> list[Any]:
    """Return a list or append the established type issue."""

    if isinstance(value, list):
        return value
    issues.append(ValidationIssue(path, "must be a list", "type"))
    return []


def is_valid_sha256(value: Any) -> bool:
    """Return whether a value is a lowercase SHA-256 digest."""

    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def is_portable_filename(value: Any) -> bool:
    """Return whether an asset name is one portable filename."""

    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return not path.is_absolute() and path.name == value and value not in (".", "..")


def is_valid_model_id(value: Any) -> bool:
    """Return whether a deployment model or profile identifier is valid."""

    return isinstance(value, str) and MODEL_ID_RE.fullmatch(value) is not None


def resolve_deployment_path(value: Any, base_dir: os.PathLike[str] | str) -> Path:
    """Resolve one deployment-config path with the existing expansion rules."""

    raw = os.path.expandvars(os.path.expanduser(str(value or "").strip()))
    if not raw:
        return Path()
    path = Path(raw)
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path.resolve()
