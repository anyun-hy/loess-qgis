import importlib.util
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from labeling_tool.monitor.monitor_time import elapsed_text, timestamp_epoch

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "src/labeling_tool/monitor/monitor_time.py"
)
_SPEC = importlib.util.spec_from_file_location("monitor_time", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

format_monitor_timestamp = _MODULE.format_monitor_timestamp
monitor_timezone_label = _MODULE.monitor_timezone_label


def test_elapsed_duration_clamps_negative_values_and_keeps_days():
    assert elapsed_text(-2) == "00:00:00"
    assert elapsed_text(65.9) == "00:01:05"
    assert elapsed_text(90061) == "1天 01:01:01"


def test_persisted_elapsed_timestamp_retains_legacy_utc_interpretation():
    expected = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
    assert timestamp_epoch("2024-01-01T00:00:00") == expected
    assert timestamp_epoch("2024-01-01T09:00:00+09:00") == expected
    assert timestamp_epoch("") is None
    assert timestamp_epoch("invalid") is None


parse_monitor_timestamp = _MODULE.parse_monitor_timestamp


def test_parse_supports_iso_z_offsets_and_epoch_seconds_equally():
    epoch = 1_704_067_200
    expected = datetime(2024, 1, 1, tzinfo=timezone.utc)

    assert parse_monitor_timestamp("2024-01-01T00:00:00Z") == expected
    assert parse_monitor_timestamp("2024-01-01T08:00:00+08:00") == expected
    assert parse_monitor_timestamp(epoch) == expected
    assert parse_monitor_timestamp(float(epoch)) == expected
    assert parse_monitor_timestamp(0) == datetime(1970, 1, 1, tzinfo=timezone.utc)


def test_plus_eight_and_plus_nine_iso_values_are_equivalent_instants():
    plus_nine = "2024-01-02T03:30:45+09:00"
    plus_eight = "2024-01-02T02:30:45+08:00"

    assert parse_monitor_timestamp(plus_nine) == parse_monitor_timestamp(plus_eight)


def test_aware_timestamps_use_requested_local_timezone_across_day_boundary():
    utc = "2024-01-01T18:30:45Z"

    assert format_monitor_timestamp(utc, local_tz=ZoneInfo("Asia/Tokyo")) == (
        "2024-01-02 03:30:45 UTC+09:00"
    )
    assert format_monitor_timestamp(
        utc, compact=True, local_tz=ZoneInfo("Asia/Shanghai")
    ) == ("01-02 02:30:45")


def test_dst_timezone_conversion_uses_zoneinfo_rules():
    eastern = ZoneInfo("America/New_York")

    assert format_monitor_timestamp("2024-07-01T00:30:00Z", local_tz=eastern) == (
        "2024-06-30 20:30:00 UTC-04:00"
    )
    assert format_monitor_timestamp("2024-01-01T00:30:00Z", local_tz=eastern) == (
        "2023-12-31 19:30:00 UTC-05:00"
    )


def test_default_local_timezone_uses_event_time_dst_rules():
    previous_tz = os.environ.get("TZ")
    os.environ["TZ"] = "America/New_York"
    time.tzset()
    try:
        assert format_monitor_timestamp("2024-07-01T00:30:00Z") == (
            "2024-06-30 20:30:00 UTC-04:00"
        )
        assert format_monitor_timestamp("2024-01-01T00:30:00Z") == (
            "2023-12-31 19:30:00 UTC-05:00"
        )
    finally:
        if previous_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous_tz
        time.tzset()


def test_naive_missing_and_invalid_values_do_not_claim_a_timezone():
    naive = datetime(2024, 1, 2, 3, 4, 5)

    assert parse_monitor_timestamp(naive) is naive
    assert format_monitor_timestamp(naive) == "2024-01-02 03:04:05（时区未记录）"
    assert (
        format_monitor_timestamp(naive, compact=True) == "01-02 03:04:05（时区未记录）"
    )
    assert format_monitor_timestamp(None) == "—"
    assert format_monitor_timestamp(" ") == "—"
    assert parse_monitor_timestamp(True) is None
    assert parse_monitor_timestamp(float("nan")) is None
    assert format_monitor_timestamp("not-a-time") == "not-a-time（时间未识别）"


def test_bad_epochs_are_rejected_without_interpreting_milliseconds_as_seconds():
    milliseconds = 1_704_067_200_000

    assert parse_monitor_timestamp(milliseconds) is None
    assert format_monitor_timestamp(milliseconds) == "1704067200000（时间未识别）"


def test_timezone_label_has_a_machine_local_utc_offset():
    label = monitor_timezone_label()

    assert label.startswith("本机时间（UTC")
    assert label.endswith("）")
    assert len(label) == len("本机时间（UTC+00:00）")
