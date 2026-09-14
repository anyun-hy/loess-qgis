"""Independent contracts for the Qt-free inference-monitor log helpers."""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from labeling_tool.core.monitor_logs import (
    log_fingerprint,
    log_payload,
    log_presentation,
    log_severity,
    read_persisted_log_page,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_records(tmp_path, records):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    path = log_dir / "pipeline.jsonl"
    path.write_bytes(b"".join(
        json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n"
        for record in records
    ))
    return path


def test_monitor_logs_imports_without_qgis_in_a_clean_subprocess():
    environment = dict(os.environ)
    plugin_path = str(ROOT / "qgis_plugins")
    environment["PYTHONPATH"] = plugin_path
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import sys; import labeling_tool.core.monitor_logs; "
            "assert 'qgis' not in sys.modules",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_reader_has_the_original_default_page_bounds():
    signature = inspect.signature(read_persisted_log_page)
    assert signature.parameters["limit"].default == 200
    assert signature.parameters["byte_budget"].default == 2 * 1024 * 1024


def test_reader_advances_through_a_budgeted_page_without_matches(tmp_path):
    path = _write_records(
        tmp_path,
        [{"level": "system", "message": "[error] older failure"}]
        + [{"level": "stdout", "message": f"normal progress {index}"} for index in range(30)],
    )
    page = read_persisted_log_page(
        {"run_dir": str(tmp_path)}, "error", byte_budget=120
    )
    assert page["rows"] == []
    assert page["has_more"]
    assert 0 < page["next_cursor"] < path.stat().st_size


def test_reader_cursor_uses_utf8_bytes_for_multibyte_records(tmp_path):
    path = _write_records(
        tmp_path,
        [
            {"timestamp": "保留", "level": "system", "message": "[error] 旧错误"},
            {"level": "system", "message": "[error] 新错误 中文"},
        ],
    )
    first = read_persisted_log_page({"run_dir": str(tmp_path)}, "error", limit=1)
    second = read_persisted_log_page(
        {"run_dir": str(tmp_path)}, "error", first["next_cursor"], limit=1
    )
    assert first["next_cursor"] < path.stat().st_size
    assert second["next_cursor"] < first["next_cursor"]
    assert {row["payload"]["message"] for row in first["rows"] + second["rows"]} == {
        "[error] 旧错误", "[error] 新错误 中文"
    }


def test_reader_reports_damaged_records_without_losing_valid_rows(tmp_path):
    path = _write_records(tmp_path, [{"level": "system", "message": "[error] valid"}])
    path.write_bytes(path.read_bytes() + b"{damaged\n")
    page = read_persisted_log_page({"run_dir": str(tmp_path)}, "error")
    assert page["skipped_records"] == 1
    assert [row["payload"]["message"] for row in page["rows"]] == ["[error] valid"]


def test_severity_fingerprint_and_summary_preserve_stderr_semantics():
    assert log_payload("not json") == {}
    assert log_severity("stderr", "TypeError: unexpected keyword argument 'run_id'") == "info"
    assert log_severity("stderr", "RuntimeWarning: fallback was used") == "warning"
    presentation = log_presentation("stderr", "Fusion Core-037 timed out after 900s")
    assert presentation["severity"] == "error"
    assert presentation["title"] == "任务处理超时"
    assert presentation["affected"] == "Fusion Core-037"
    assert log_fingerprint("error", "worker failed", "Core-037", 1).startswith("error:")


def test_reader_keeps_raw_timestamp_truncates_long_messages_and_is_read_only(tmp_path):
    message = "[error] " + "x" * 17000
    path = _write_records(
        tmp_path,
        [{"timestamp": "2026-09-14T00:00:00+09:00", "level": "system", "message": message}],
    )
    before = path.read_bytes()
    page = read_persisted_log_page({"run_dir": str(tmp_path)}, "error")
    row = page["rows"][0]
    assert row["timestamp"] == "2026-09-14T00:00:00+09:00"
    assert len(row["message"]) == 240
    assert len(row["payload"]["message"]) == 16000
    assert row["payload"]["truncated"] is True
    assert path.read_bytes() == before


def test_reader_rejects_invalid_cursor_and_missing_run_directory(tmp_path):
    path = _write_records(tmp_path, [{"level": "system", "message": "[error] valid"}])
    with pytest.raises(ValueError, match="截断"):
        read_persisted_log_page({"run_dir": str(tmp_path)}, "error", path.stat().st_size + 1)
    with pytest.raises(FileNotFoundError):
        read_persisted_log_page({"run_dir": str(tmp_path / "missing")}, "error")
