"""Shared value encoding for PostgreSQL-backed Run state."""

from __future__ import annotations

import datetime as dt
import json
from typing import Any


def utc_now() -> str:
    """Return the canonical UTC timestamp used by Run-state rows."""
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def json_value(value: Any) -> str:
    """Serialize one stable compact JSON database value."""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def bounded_utf8(value: Any, maximum_bytes: int) -> str:
    """Bound diagnostic text by encoded bytes without invalid UTF-8."""
    encoded = str(value or "").encode("utf-8")
    if len(encoded) <= int(maximum_bytes):
        return encoded.decode("utf-8")
    return encoded[: int(maximum_bytes)].decode("utf-8", errors="ignore")


def row_dict(row: Any | None) -> dict[str, Any] | None:
    """Convert one optional PostgreSQL row to a plain mapping."""
    return dict(row) if row is not None else None
