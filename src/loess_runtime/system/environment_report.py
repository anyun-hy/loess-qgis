"""Assemble deployment environment reports from independently owned checks."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loess_runtime.system.environment_checks import (
    add_check,
    append_dependency_checks,
    append_runtime_boundary_checks,
    overall_status,
)
from loess_runtime.system.environment_deployment import (
    append_fusion_and_class_checks,
    append_output_directory_check,
    append_runtime_tuning,
    append_sam_checks,
    append_semantic_model_checks,
    load_deployment_configuration,
)

FINGERPRINT_FILES = (
    "../project_manifest.json",
    "../.loess-project-id",
    "../runtime/loess_launcher.sh",
)


def environment_fingerprint(scripts_dir: Path) -> str:
    """Fingerprint the generated project identity and persisted launcher."""

    digest = hashlib.sha256()
    for name in FINGERPRINT_FILES:
        path = scripts_dir / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        if path.is_file() and not path.is_symlink():
            file_digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    file_digest.update(chunk)
            digest.update(file_digest.hexdigest().encode("ascii"))
        else:
            digest.update(b"<missing>")
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def build_environment_report(
    *,
    scripts_dir: Path,
    asset_base_dir: Path,
    conda_env: str,
    output_dir: str,
    fingerprint: Callable[[Path], str] = environment_fingerprint,
) -> dict[str, Any]:
    """Build the stable report payload after appending checks in CLI order."""

    checks: list[dict[str, str]] = []
    add_check(
        checks,
        "conda_env",
        "ready",
        conda_env,
        "config.sh:CONDA_ENV",
        "environment checker is running inside this Conda environment",
        "edit config.sh:CONDA_ENV",
    )
    append_runtime_boundary_checks(checks, conda_env)
    dependencies = append_dependency_checks(checks, conda_env)
    effective, issues = load_deployment_configuration(
        checks,
        scripts_dir,
        asset_base_dir,
    )
    tuning = append_runtime_tuning(checks, effective, dependencies)
    append_semantic_model_checks(checks, effective, issues, tuning)
    append_fusion_and_class_checks(checks, effective, issues)
    append_sam_checks(checks, effective, tuning, conda_env)
    append_output_directory_check(checks, output_dir)
    return {
        "schema_version": 1,
        "status": overall_status(checks),
        "config_fingerprint": fingerprint(scripts_dir),
        "effective": effective,
        "checks": checks,
    }
