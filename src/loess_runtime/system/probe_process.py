"""Launch isolated environment workers with the deployment package roots."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ENVIRONMENT_WORKER_MODULE = "loess_runtime.system.check_environment"


def environment_worker_command(*arguments: object) -> list[str]:
    """Build a command that re-enters the environment-check CLI worker."""

    return [sys.executable, "-m", ENVIRONMENT_WORKER_MODULE, *map(str, arguments)]


def probe_environment() -> dict[str, str]:
    """Preserve the host environment while exposing the runtime package roots."""

    environment = os.environ.copy()
    roots = [Path(__file__).resolve().parents[2]]
    labeling_package = sys.modules.get("labeling_tool")
    labeling_file = getattr(labeling_package, "__file__", None)
    if not labeling_file:
        labeling_spec = importlib.util.find_spec("labeling_tool")
        labeling_file = getattr(labeling_spec, "origin", None)
    if labeling_file:
        roots.append(Path(labeling_file).resolve().parent.parent)

    entries = [str(path) for path in roots]
    entries.extend(
        value for value in environment.get("PYTHONPATH", "").split(os.pathsep) if value
    )
    unique_entries = []
    seen = set()
    for value in entries:
        key = os.path.normcase(os.path.abspath(value))
        if key in seen:
            continue
        seen.add(key)
        unique_entries.append(value)
    environment["PYTHONPATH"] = os.pathsep.join(unique_entries)
    return environment


def run_environment_worker(
    command: Sequence[str],
    *,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    """Run an environment CLI worker without interpreting its result stream."""

    return subprocess.run(
        command,
        env=probe_environment(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
