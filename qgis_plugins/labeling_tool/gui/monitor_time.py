"""Small, Qt-free helpers for rendering monitor event timestamps.

Run records can contain an ISO timestamp, while live log entries may carry an
epoch timestamp.  These helpers deliberately keep a timestamp without an
offset naive: assigning it a timezone at display time would misrepresent the
recorded data.
"""

from __future__ import annotations

from datetime import datetime, timezone, tzinfo
import math
from typing import Any


_MISSING = "—"
_UNRECORDED_TIMEZONE = "（时区未记录）"
_UNRECOGNIZED_TIME = "（时间未识别）"


def parse_monitor_timestamp(value: Any) -> datetime | None:
    """Return a datetime for a supported monitor timestamp, or ``None``.

    Supported inputs are datetime objects, ISO-8601 strings (including ``Z``)
    and finite integer/float epoch seconds.  Naive datetimes are returned as
    naive values so callers never silently invent their timezone.
    """

    if isinstance(value, datetime):
        return value
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None
    try:
        # Python 3.11 accepts ``Z`` directly.  Replacing it also retains
        # compatibility with the older Python versions used by some QGIS builds.
        if text.endswith(("Z", "z")):
            text = f"{text[:-1]}+00:00"
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def format_monitor_timestamp(
    value: Any, *, compact: bool = False, local_tz: tzinfo | None = None
) -> str:
    """Format a monitor timestamp for display without concealing ambiguity."""

    if value is None or (isinstance(value, str) and not value.strip()):
        return _MISSING

    parsed = parse_monitor_timestamp(value)
    if parsed is None:
        return f"{value}{_UNRECOGNIZED_TIME}"

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        pattern = "%m-%d %H:%M:%S" if compact else "%Y-%m-%d %H:%M:%S"
        return f"{parsed.strftime(pattern)}{_UNRECORDED_TIMEZONE}"

    # ``astimezone()`` with no argument consults the machine's timezone rules
    # for *this event's instant*.  Capturing ``now().tzinfo`` here would freeze
    # today's offset and render historical DST timestamps incorrectly.
    localized = parsed.astimezone(local_tz) if local_tz is not None else parsed.astimezone()
    if compact:
        return localized.strftime("%m-%d %H:%M:%S")
    offset = localized.strftime("%z")
    offset = f"{offset[:3]}:{offset[3:]}" if len(offset) == 5 else offset
    return f"{localized.strftime('%Y-%m-%d %H:%M:%S')} UTC{offset}"


def monitor_timezone_label() -> str:
    """Return the current machine-local timezone label, including its offset."""

    local_now = datetime.now().astimezone()
    offset = local_now.strftime("%z")
    offset = f"{offset[:3]}:{offset[3:]}" if len(offset) == 5 else offset
    return f"本机时间（UTC{offset}）"
