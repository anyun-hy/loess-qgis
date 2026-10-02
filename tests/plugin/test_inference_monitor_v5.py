"""Source-level contracts for the schema-v2 inference monitor.

These tests intentionally avoid importing QGIS.  They protect the monitor's
control-plane semantics on QGIS 4.2 / PyQt6 / Qt6 while the live UI is covered
separately by platform acceptance.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from labeling_tool.monitor.monitor_logs import (
    log_fingerprint,
    log_presentation,
    log_severity,
    read_persisted_log_page,
)
from labeling_tool.monitor.monitor_progress import (
    overall_completion_fraction,
    overview_work_label,
)

ROOT = Path(__file__).resolve().parents[2]
_TIME_SPEC = importlib.util.spec_from_file_location(
    "monitor_time_contract", ROOT / "src/labeling_tool/monitor/monitor_time.py"
)
_TIME_MODULE = importlib.util.module_from_spec(_TIME_SPEC)
_TIME_SPEC.loader.exec_module(_TIME_MODULE)
MONITOR_PATH = ROOT / "src" / "labeling_tool" / "monitor" / "inference_monitor.py"
LOG_PANEL_PATH = ROOT / "src" / "labeling_tool" / "main" / "log_panel.py"
RUNNER_PATH = ROOT / "src" / "labeling_tool" / "runs" / "v5_async_runner.py"
SOURCE = MONITOR_PATH.read_text(encoding="utf-8")
LOG_PANEL_SOURCE = LOG_PANEL_PATH.read_text(encoding="utf-8")
RUNNER_SOURCE = RUNNER_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)
LOG_PANEL_TREE = ast.parse(LOG_PANEL_SOURCE)


def _monitor_class() -> ast.ClassDef:
    for node in TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == "InferenceMonitorDialog":
            return node
    raise AssertionError("InferenceMonitorDialog is missing")


def _method(name: str) -> ast.FunctionDef:
    for node in _monitor_class().body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"InferenceMonitorDialog.{name} is missing")


def test_disk_log_pagination_finds_errors_older_than_memory_cache(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    records = [
        {
            "timestamp": 1,
            "level": "system",
            "message": "[accelerator-restart] failed (rc=139)",
        }
    ]
    records += [
        {"timestamp": index + 2, "level": "stdout", "message": "ordinary progress"}
        for index in range(5100)
    ]
    records.append(
        {"timestamp": 6000, "level": "stderr", "message": "Warning: old warning"}
    )
    (log_dir / "pipeline.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    cursor = None
    errors = []
    empty_nonterminal = False
    for _ in range(500):
        page = read_persisted_log_page(
            {"run_dir": str(tmp_path)}, "error", cursor, byte_budget=2048
        )
        errors.extend(page["rows"])
        empty_nonterminal |= not page["rows"] and page["has_more"]
        if cursor is not None:
            assert page["next_cursor"] < cursor
        cursor = page["next_cursor"]
        if not page["has_more"]:
            break
    else:
        raise AssertionError("cursor never reached the beginning")
    assert empty_nonterminal
    assert len(errors) == 1
    assert "rc=139" in errors[0]["payload"]["message"]
    assert errors[0]["monitor_event_id"] == 0
    warnings = read_persisted_log_page({"run_dir": str(tmp_path)}, "warning")
    assert len(warnings["rows"]) == 1


def test_disk_log_limit_search_and_corruption_are_explicit(tmp_path):
    import pytest

    (tmp_path / "logs").mkdir()
    path = tmp_path / "logs/pipeline.jsonl"
    path.write_text(
        "".join(
            json.dumps({"level": "system", "message": f"[error] 失败 {i}"}) + "\n"
            for i in range(7)
        )
        + "{broken\n"
    )
    first = read_persisted_log_page({"run_dir": str(tmp_path)}, "error", limit=3)
    second = read_persisted_log_page(
        {"run_dir": str(tmp_path)}, "error", first["next_cursor"], limit=3
    )
    assert first["skipped_records"] == 1
    assert len(first["rows"]) == len(second["rows"]) == 3
    assert not (
        {r["monitor_event_id"] for r in first["rows"]}
        & {r["monitor_event_id"] for r in second["rows"]}
    )
    assert (
        len(
            read_persisted_log_page(
                {"run_dir": str(tmp_path)}, "error", search="失败 2"
            )["rows"]
        )
        == 1
    )
    with pytest.raises(ValueError, match="截断"):
        read_persisted_log_page(
            {"run_dir": str(tmp_path)}, "error", path.stat().st_size + 1
        )
    with pytest.raises(FileNotFoundError):
        read_persisted_log_page({"run_dir": str(tmp_path / "missing")}, "error")


def test_persisted_log_reads_stay_in_query_worker_raw_history_branch():
    executor = ROOT / "src/labeling_tool/monitor/monitor_query_io.py"
    client = ROOT / "src/labeling_tool/monitor/monitor_query_client.py"
    executor_source = executor.read_text(encoding="utf-8")
    client_source = client.read_text(encoding="utf-8")
    assert "read_persisted_log_page" in executor_source
    assert 'scope in {"raw_warning", "raw_error"}' in executor_source
    assert "query_failed.emit" in client_source
    for method in _monitor_class().body:
        if isinstance(method, ast.FunctionDef):
            assert "read_persisted_log_page" not in (
                ast.get_source_segment(SOURCE, method) or ""
            )


def test_overview_work_uses_plain_language_without_losing_detail_states():
    examples = {
        "Core/Seam/Junction 拟合": "正在处理边界",
        "推理 + Core 拟合": "识别地物，同时处理边界",
        "Work Package 推理": "正在识别地物",
        "写入正式 GPKG": "正在保存正式结果",
        "Accepted 差分": "正在排除已确认的区域",
        "组装失败：写入正式 GPKG": "结果合并失败，请查看详情",
        "Run 已停止；恢复入口位于主界面": "已停止，可在主界面恢复",
        "future_unknown_stage": "当前步骤待确认，请查看详情",
    }
    for stage, expected in examples.items():
        assert overview_work_label(stage) == expected


def test_monitor_imports_progress_helpers_without_legacy_local_aliases():
    expected = {
        "stream_from_step",
        "stage_from_step",
        "waiting_count",
        "overall_completion_fraction",
        "overview_work_label",
        "database_phase",
        "stream_progress_view",
    }
    trees = [
        TREE,
        ast.parse(
            (ROOT / "src/labeling_tool/monitor/monitor_observations.py").read_text()
        ),
        *(
            ast.parse(path.read_text())
            for path in (ROOT / "src/labeling_tool/monitor/pages").glob("*.py")
        ),
    ]
    imports = [
        node
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "labeling_tool.monitor.monitor_progress"
    ]
    assert expected <= {name.name for node in imports for name in node.names}
    assert all(name.asname is None for node in imports for name in node.names)
    assert not any(
        isinstance(node, ast.FunctionDef)
        and node.name in expected | {"_" + name for name in expected}
        for tree in trees
        for node in tree.body
    )


def _log_panel_method(name: str) -> ast.FunctionDef:
    for node in LOG_PANEL_TREE.body:
        if not isinstance(node, ast.ClassDef) or node.name != "LogPanel":
            continue
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name == name:
                return child
    raise AssertionError(f"LogPanel.{name} is missing")


def _method_source(name: str) -> str:
    node = _method(name)
    return ast.get_source_segment(SOURCE, node) or ""


def _execute_log_panel_method(name: str, instance, *args, **kwargs):
    function = copy.deepcopy(_log_panel_method(name))
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {
        "datetime": datetime,
        "timezone": timezone,
        "parse_monitor_timestamp": _TIME_MODULE.parse_monitor_timestamp,
        "time": time,
        "MAX_RAW_RECORDS": 20000,
        "MAX_EVENT_DETAIL_RECORDS": 200,
    }
    exec(compile(module, str(LOG_PANEL_PATH), "exec"), namespace)
    return namespace[name](instance, *args, **kwargs)


def test_database_binding_accepts_the_run_spec_for_stage_aware_monitoring():
    method = _method("bind_state_database")
    positional = [argument.arg for argument in method.args.args]
    keyword_only = [argument.arg for argument in method.args.kwonlyargs]
    assert "run_spec" in positional + keyword_only

    defaults = [*method.args.defaults, *method.args.kw_defaults]
    assert any(
        isinstance(default, ast.Constant) and default.value is None
        for default in defaults
    )


def test_database_poll_uses_one_snapshot_with_separate_progress_lanes():
    poll_source = _method_source("_poll_database")
    apply_source = _method_source("_apply_database_snapshot")
    assert "monitor_snapshot" not in poll_source
    assert "self._query_client.queue_snapshot" in poll_source
    executor_source = (
        ROOT / "src/labeling_tool/monitor/monitor_query_io.py"
    ).read_text(encoding="utf-8")
    assert "database.monitor_read.snapshot(run_id)" in executor_source
    assert "database.job_counts(" not in executor_source
    poll_source = (
        apply_source
        + (ROOT / "src/labeling_tool/monitor/monitor_observations.py").read_text()
    )
    assert 'job_counts.get("work_package")' in poll_source
    assert 'job_counts.get("unit_fit")' in poll_source
    assert '"stream_unit_type_counts"' in poll_source
    assert 'snapshot.get("job_progress")' in poll_source


def test_mixed_v5_job_total_is_not_used_as_the_monitor_progress_bar():
    stage_progress = _method_source("set_stage_progress")
    assert "self._query_client.is_bound" in stage_progress


def test_monitor_has_no_hidden_legacy_presentation_state():
    obsolete = {
        "_bar",
        "_summary",
        "_stage_rail",
        "_package_overview",
        "_fit_label",
        "_run_overview",
        "_assembly_overview",
        "_coverage_overview",
        "_update_summary",
        "_update_stage_rail",
        "_stream_state",
        "_tile_state",
        "_step_started_at",
        "_step_attempts",
        "_active_stream_stages",
        "_active_global_stage",
        "_active_inference_stream",
        "_package_activity",
        "_runtime_progress",
        "_coverage_state",
        "_assembly_phase_statuses",
        "_run_created_epoch",
        "_terminal_run_status",
        "_logged_error_texts",
        "_process_log_suppressions",
        "_runner_message",
    }
    attributes = {
        node.attr
        for node in ast.walk(_monitor_class())
        if isinstance(node, ast.Attribute)
    }
    methods = {
        node.name for node in _monitor_class().body if isinstance(node, ast.FunctionDef)
    }
    assert not obsolete & (attributes | methods)
    assert "PIPELINE_STAGES" not in SOURCE
    information = _method_source("_build_run_information_dialog")
    for field in ("_run_information", "_assembly_information", "_coverage_information"):
        assert field in information
    assert "label.text()" not in information


def test_active_package_inference_is_not_overwritten_by_queued_units():
    from labeling_tool.monitor.monitor_observations import MonitorObservations

    state = MonitorObservations()
    state.observe_stream_progress(
        {
            "event": "package_model_loading",
            "package_id": "p",
            "stream_id": "model:a",
            "current": 1,
            "total": 1,
        },
        database_bound=True,
        configured_batch_size=8,
        fusion_profile_id="mix",
        epoch_now=100,
        monotonic_now=20,
    )
    state.apply_snapshot(
        {
            "run": {"status": "running"},
            "job_counts": {"work_package": {"running": 1}, "unit_fit": {"queued": 2}},
            "active_work_package": {"package_id": "p", "attempt": 0},
            "streams": [{"stream_id": "model:a", "status": "pending"}],
            "stream_unit_type_counts": {"model:a": {"core": {"queued": 2}}},
            "stream_unit_job_type_counts": {"model:a": {"core": {"queued": 2}}},
        },
        epoch_now=101,
    )
    assert state.stream_view("model:a")["state"]["stage"] == "Work Package 推理"


def test_package_events_without_stream_id_are_processed_before_the_guard():
    from labeling_tool.monitor.monitor_observations import MonitorObservations

    state = MonitorObservations()

    def observe(event, **fields):
        return state.observe_stream_progress(
            {"event": event, "package_id": "p", **fields},
            database_bound=True,
            configured_batch_size=8,
            fusion_profile_id="mix",
            epoch_now=100,
            monotonic_now=20,
        )

    observe("package_model_loading", stream_id="model:a", current=1, total=1)
    observe("package_tile_batch_reduced", effective_batch_size=2)
    observe("package_tile_materialized", current=1, total=3)
    assert "effective_batch_size" not in state.package_view()
    observe("accelerator_worker_paused_low_disk")
    assert state.package_view()["notice"] == "等待磁盘空间"
    observe("work_package_finished", elapsed_sec=4.5)
    assert state.package_view()["status"] == "已完成"
    assert state.package_view()["elapsed_sec"] == 4.5


def test_terminal_snapshot_does_not_disable_a_subsequent_resume():
    poll = _method_source("_apply_database_snapshot")
    finished = _method_source("_on_finished")

    # bind_state_database polls before runner.resume changes failed/stopped back
    # to running.  A terminal snapshot must therefore not stop the timer.
    assert "_poll_timer.stop" not in poll
    assert finished.index("self._poll_database()") < finished.index(
        "self._poll_timer.stop()"
    )


def test_elapsed_column_tracks_the_current_persisted_assembly_phase():
    from labeling_tool.monitor.monitor_observations import MonitorObservations

    assert (
        "阶段耗时" in (ROOT / "src/labeling_tool/monitor/pages/results.py").read_text()
    )
    state = MonitorObservations()
    state.observe_stream_progress(
        {
            "event": "assembly_progress",
            "stream_id": "model:a",
            "status": "running",
            "phase_name": "写入",
            "current": 1,
            "total": 2,
            "elapsed_sec": 3,
        },
        database_bound=True,
        configured_batch_size=0,
        fusion_profile_id="",
        epoch_now=100,
        monotonic_now=10,
    )
    view = state.stream_view("model:a")
    assert view["state"]["elapsed"] == "00:00:03"
    assert view["state"]["stage_progress"] == "1/2"
    assert view["state"]["stage"] == "写入"


def test_monitor_reads_persisted_coverage_validation_summary():
    poll = _method_source("_apply_database_snapshot")
    coverage = _method_source("_update_coverage_information")

    assert "apply_snapshot(" in poll
    assert (
        "stream_coverage_validation"
        in (ROOT / "src/labeling_tool/monitor/monitor_observations.py").read_text()
    )
    assert "gap_area_m2" in coverage
    assert "overlap_area_m2" in coverage
    assert "outside_area_m2" in coverage


def test_overall_progress_bar_uses_task_groups_instead_of_time_estimates():
    build_ui = _method_source("_build_ui")
    update = _method_source("_render_window_snapshot_status")
    assert "self._overall_bar = OverallProgressTrack()" in build_ui
    assert "本次推理任务完成度" in build_ui
    assert "按任务组统计，不代表剩余时间。" in build_ui
    assert "overall_completion_fraction(" in update

    fraction, group_count = overall_completion_fraction(
        "running",
        {
            "work_package": {"ready": 1},
            "fragmentation_v33": {"ready": 2, "queued": 2},
            "unit_confidence": {"ready": 5, "queued": 5},
            "unit_fit": {"ready": 5, "queued": 5},
        },
        {
            "work_package": {"completed": 1.0, "total": 1},
            "fragmentation_v33": {"completed": 2.0, "total": 4},
            "unit_confidence": {"completed": 5.0, "total": 10},
            "unit_fit": {"completed": 5.0, "total": 10},
        },
        [
            {"stream_id": f"model:{index}", "status": "raster_ready"}
            for index in range(4)
        ],
        {},
    )
    assert group_count == 7
    assert fraction == 0.5
    assert overall_completion_fraction("ready", {}, {}, [], {}) == (1.0, 1)


def test_log_count_separates_raw_stderr_from_confirmed_failures():
    resource_tuning = (
        '[resource-tuning] {"first_failed_batch":128,'
        '"probes":[{"status":"failed","error":"CUDA out of memory"}],'
        '"status":"completed"}'
    )
    assert log_severity("system", resource_tuning) == "info"
    assert (
        log_severity("stderr", "TypeError: unexpected keyword argument 'run_id'")
        == "info"
    )
    assert log_severity("stderr", "GDAL diagnostic output") == "info"
    assert log_severity("stderr", "RuntimeWarning: fallback was used") == "warning"
    assert log_severity("stdout", '{"event":"probe","status":"error"}') == "error"
    assert log_severity("stdout", '{"event":"stream_assembly_failed"}') == "error"
    assert "observe_log(" in _method_source("_on_log")


def test_log_presentation_explains_timeout_without_hiding_raw_source():
    presentation = log_presentation(
        "stderr",
        "Fusion Core-037 timed out after 900s",
    )

    assert presentation["source"] == "stderr"
    assert presentation["severity"] == "error"
    assert presentation["title"] == "任务处理超时"
    assert presentation["affected"] == "Fusion Core-037"
    assert "终止" in presentation["system_action"]
    assert "自动重试" in presentation["user_action"]
    assert presentation["fingerprint"].startswith("error:")


def test_log_panel_separates_source_severity_and_readable_details():
    for contract in (
        "def append_event(",
        "source: str,",
        "severity: str,",
        'self._visible_severities: set[str] = {"info", "warning", "error"}',
        '("all", "全部")',
        '("warning", "警告")',
        '("error", "错误")',
        'QPushButton("技术详情")',
        '("系统处理", event["system_action"])',
        '("用户操作", event["user_action"])',
        'event["repeat_count"] = int(event["repeat_count"]) + 1',
        "self._raw_records.append(raw_record)",
        "pending_records.append(raw_record)",
        "self.log_edit.setAcceptRichText(False)",
        "self.log_edit.document().setMaximumBlockCount(MAX_VISIBLE_LOG_BLOCKS)",
        "self._coalesced_rebuild_timer.start()",
    ):
        assert contract in LOG_PANEL_SOURCE

    assert "self._event_index.clear()" in LOG_PANEL_SOURCE
    assert "self._raw_records.clear()" in LOG_PANEL_SOURCE


def test_log_panel_yields_between_bounded_rebuild_batches():
    rebuild = ast.get_source_segment(LOG_PANEL_SOURCE, _log_panel_method("_rebuild"))
    batch = ast.get_source_segment(
        LOG_PANEL_SOURCE, _log_panel_method("_render_rebuild_batch")
    )
    severity_filter = ast.get_source_segment(
        LOG_PANEL_SOURCE, _log_panel_method("set_visible_severities")
    )

    assert "tuple(" in rebuild
    assert "self._render_rebuild_batch()" in rebuild
    assert "REBUILD_FRAME_BUDGET_SECONDS" in batch
    assert "REBUILD_BATCH_SIZE" in batch
    assert "self._rebuild_timer.start()" in batch
    assert "if selected == self._visible_severities:" in severity_filter
    assert "QApplication.processEvents" not in LOG_PANEL_SOURCE


def test_log_panel_bounds_document_and_python_history():
    assert "MAX_VISIBLE_LOG_BLOCKS = 4000" in LOG_PANEL_SOURCE
    assert "MAX_CACHED_EVENTS = 5000" in LOG_PANEL_SOURCE
    assert "MAX_RAW_RECORDS = 20000" in LOG_PANEL_SOURCE
    assert "MAX_EVENT_DETAIL_RECORDS = 200" in LOG_PANEL_SOURCE
    assert "QTextEdit.LineWrapMode.NoWrap" in LOG_PANEL_SOURCE
    assert "def _trim_event_cache(self)" in LOG_PANEL_SOURCE


def test_log_panel_deduplicates_display_but_preserves_every_raw_record():
    panel = SimpleNamespace(
        _events=[],
        _event_index={},
        _raw_records=[],
        _pending_stderr_records={},
        _event_visible=lambda _event: False,
        _render_event=lambda _event: None,
        _rebuild=lambda: None,
        _rebuild_in_progress=False,
        _trim_event_cache=lambda: None,
        _schedule_rebuild=lambda: None,
    )
    values = {
        "source": "stderr",
        "severity": "error",
        "title": "任务处理超时",
        "fingerprint": "error:core-037-timeout",
    }

    assert (
        _execute_log_panel_method("append_event", panel, "Core-037 timeout", **values)
        is True
    )
    values["source"] = "system"
    assert (
        _execute_log_panel_method(
            "append_event", panel, '{"error":"Core-037 timeout"}', **values
        )
        is False
    )

    assert len(panel._events) == 1
    assert panel._events[0]["repeat_count"] == 2
    assert [record["source"] for record in panel._events[0]["records"]] == [
        "stderr",
        "system",
    ]
    assert [record["text"] for record in panel._raw_records] == [
        "Core-037 timeout",
        '{"error":"Core-037 timeout"}',
    ]


def test_log_panel_preserves_capture_time_and_labels_missing_source_time():
    panel = SimpleNamespace(
        _events=[],
        _event_index={},
        _raw_records=[],
        _pending_stderr_records={},
        _event_visible=lambda _event: False,
        _render_event=lambda _event: None,
        _rebuild_in_progress=False,
        _trim_event_cache=lambda: None,
        _schedule_rebuild=lambda: None,
    )
    original = "2026-09-09T10:00:00.123456Z"
    _execute_log_panel_method(
        "append_event",
        panel,
        "source event",
        source="stdout",
        severity="info",
        event_timestamp=original,
    )
    captured = panel._raw_records[-1]
    assert captured["source_timestamp"] == original
    assert captured["timestamp_kind"] == "captured"
    assert datetime.fromisoformat(captured["timestamp"]) == datetime.fromisoformat(
        original
    )
    assert captured["received_at"] != captured["timestamp"]
    _execute_log_panel_method(
        "append_event", panel, "local event", source="system", severity="info"
    )
    received = panel._raw_records[-1]
    assert received["timestamp_kind"] == "received"
    assert received["timestamp"] == received["received_at"]
    assert datetime.fromisoformat(received["timestamp"]).tzinfo is not None
    _execute_log_panel_method(
        "append_event",
        panel,
        "epoch zero",
        source="stdout",
        severity="info",
        event_timestamp=0,
    )
    assert datetime.fromisoformat(panel._raw_records[-1]["timestamp"]).timestamp() == 0


def test_monitor_forwards_existing_log_capture_time():
    from labeling_tool.monitor.monitor_observations import MonitorLogObservations

    logs = MonitorLogObservations()
    captured = "2026-09-09T10:00:00.123456Z"
    log = logs.observe_process_log(
        {"source": "stdout", "message": "delayed", "timestamp": captured},
        attempt_for=lambda _name: 0,
    )
    assert log["event_timestamp"] == captured


def test_log_fingerprint_keeps_tasks_and_attempts_separate():
    first = log_fingerprint("error", "worker failed", "Core-037", 1)
    other_task = log_fingerprint("error", "worker failed", "Core-038", 1)
    retry = log_fingerprint("error", "worker failed", "Core-037", 2)

    assert first
    assert len({first, other_task, retry}) == 3
    assert log_fingerprint("error", "worker failed", "", 0) == ""


def test_error_event_carries_recent_stderr_trace_as_technical_context():
    panel = SimpleNamespace(
        _events=[],
        _event_index={},
        _raw_records=[],
        _pending_stderr_records={},
        _event_visible=lambda _event: False,
        _render_event=lambda _event: None,
        _rebuild=lambda: None,
        _rebuild_in_progress=False,
        _trim_event_cache=lambda: None,
        _schedule_rebuild=lambda: None,
    )
    for line in ("Traceback (most recent call last):", '  File "worker.py"'):
        assert (
            _execute_log_panel_method(
                "append_event",
                panel,
                line,
                source="stderr",
                severity="info",
            )
            is True
        )
    assert (
        _execute_log_panel_method(
            "append_event",
            panel,
            "RuntimeError: disk full",
            source="stderr",
            severity="error",
            title="任务执行失败",
            fingerprint="error:core-037:attempt=1:disk-full",
        )
        is True
    )

    error_event = panel._events[-1]
    assert [record["text"] for record in error_event["records"]] == [
        "Traceback (most recent call last):",
        '  File "worker.py"',
        "RuntimeError: disk full",
    ]
    assert panel._pending_stderr_records == {}


def test_concurrent_process_traces_are_buffered_by_context_key():
    panel = SimpleNamespace(
        _events=[],
        _event_index={},
        _raw_records=[],
        _pending_stderr_records={},
        _event_visible=lambda _event: False,
        _render_event=lambda _event: None,
        _rebuild=lambda: None,
        _rebuild_in_progress=False,
        _trim_event_cache=lambda: None,
        _schedule_rebuild=lambda: None,
    )
    for context_key, line in (("Core-A", "trace A"), ("Core-B", "trace B")):
        _execute_log_panel_method(
            "append_event",
            panel,
            line,
            source="stderr",
            severity="info",
            context_key=context_key,
        )
    _execute_log_panel_method(
        "append_event",
        panel,
        "Core-B failed",
        source="system",
        severity="error",
        fingerprint="error:core-b:attempt=1:failed",
        context_key="Core-B",
    )

    assert [record["text"] for record in panel._events[-1]["records"]] == [
        "trace B",
        "Core-B failed",
    ]
    assert list(panel._pending_stderr_records) == ["Core-A"]


def test_runner_keeps_legacy_log_signal_and_adds_process_context_signal():
    from labeling_tool.monitor.monitor_observations import (
        MonitorLogObservations,
        MonitorObservations,
    )

    attach = _method_source("attach_runner")
    assert "log_line = pyqtSignal(str, str)" in RUNNER_SOURCE
    assert "process_log = pyqtSignal(object)" in RUNNER_SOURCE
    assert '"step": str(context.get("label") or "")' in RUNNER_SOURCE
    assert '"unit_id": str(' in RUNNER_SOURCE
    assert '"attempt": int(job.get("attempt") or 0)' in RUNNER_SOURCE
    assert 'getattr(runner, "process_log", None)' in attach
    state, logs = MonitorObservations(), MonitorLogObservations()
    rich = logs.observe_process_log(
        {
            "source": "stderr",
            "message": "RuntimeWarning: captured",
            "step": "step-a",
            "attempt": 3,
            "timestamp": 123,
        },
        attempt_for=state.attempt_for,
    )
    assert rich["context_key"] == "step-a:attempt=3"
    assert rich["event_timestamp"] == 123
    assert (
        logs.observe_log(
            "stderr",
            "RuntimeWarning: captured",
            context=None,
            attempt_for=state.attempt_for,
        )
        is None
    )


def test_terminal_failures_are_promoted_to_single_structured_log_events():
    from labeling_tool.monitor.monitor_observations import (
        MonitorLogObservations,
        MonitorObservations,
    )

    state, logs = MonitorObservations(), MonitorLogObservations()
    name = "unit_fit:model:a:core"
    state.observe_step_started(name, epoch_now=100)
    change = state.observe_step_finished(
        name,
        2,
        {"success": False, "error": "same error"},
        epoch_now=104,
        database_bound=True,
    )
    assert len(change.pending_logs) == 1
    event = change.pending_logs[0]
    payload = json.loads(event.message)
    assert payload["event"] == "monitor_step_failed"
    assert payload["attempt"] == 1 and payload["return_code"] == 2
    assert (
        logs.observe_log(
            event.level,
            event.message,
            context=event.context,
            attempt_for=state.attempt_for,
        )
        is not None
    )
    failed = {"success": False, "status": "failed", "error": "same error"}
    assert logs.observe_pipeline_failure(failed, attempt_for=state.attempt_for) is None
    logs.clear_logged_errors()
    terminal = logs.observe_pipeline_failure(failed, attempt_for=state.attempt_for)
    assert json.loads(terminal["text"])["event"] == "monitor_pipeline_failed"
    assert logs.observe_pipeline_failure(failed, attempt_for=state.attempt_for) is None
    assert (
        logs.observe_pipeline_failure(
            {**failed, "status": "stopped", "error": "different"},
            attempt_for=state.attempt_for,
        )
        is None
    )


def test_monitor_database_reads_are_serialized_off_the_gui_thread():
    calls = [node.func for node in ast.walk(TREE) if isinstance(node, ast.Call)]
    methods = {call.attr for call in calls if isinstance(call, ast.Attribute)}
    assert {
        "queue_snapshot",
        "queue_detail",
        "queue_history",
        "queue_object_history",
    } <= methods
    assert (
        not {
            "monitor_snapshot",
            "page_tiles",
            "page_stream_units",
            "page_events",
        }
        & methods
    )
    assert any(
        isinstance(call, ast.Name) and call.id == "query_matches_filter"
        for call in calls
    )
    assert "MonitorQueryClient" in _method_source("__init__")


def test_monitor_uses_batched_runner_events_and_debounced_search():
    attach = _method_source("attach_runner")
    batch = _method("_on_log_batch")
    assert 'getattr(runner, "log_batch", None)' in attach
    assert 'getattr(runner, "stream_progress_batch", None)' in attach
    assert "self._on_log_batch" in attach and "self._on_stream_progress_batch" in attach
    cleanup = [
        part
        for node in ast.walk(batch)
        if isinstance(node, ast.Try)
        for part in node.finalbody
    ]
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "end_log_batch"
        for part in cleanup
        for node in ast.walk(part)
    )
    # Actual debounce timing is covered by monitor_page_queries in native QGIS.
