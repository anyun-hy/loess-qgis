"""Small process-local metrics used by the bounded v5 workers."""

from __future__ import annotations

import resource
import sys
from pathlib import Path


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def directory_size(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    for item in root.rglob("*"):
        if not item.is_file():
            continue
        try:
            total += item.stat().st_size
        except FileNotFoundError:
            continue
    return total
