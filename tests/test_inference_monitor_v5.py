"""Source-level contracts for the schema-v2 inference monitor.

These tests intentionally avoid importing QGIS.  They protect the monitor's
control-plane semantics on both QGIS 3/Qt5 and QGIS 4/Qt6 while the live UI is
covered separately by platform acceptance.
"""

from __future__ import annotations

import ast
import copy
import json
import re
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MONITOR_PATH = (
    ROOT / "qgis_plugins" / "labeling_tool" / "gui" / "inference_monitor.py"
)
LOG_PANEL_PATH = ROOT / "qgis_plugins" / "labeling_tool" / "gui" / "log_panel.py"
RUNNER_PATH = ROOT / "qgis_plugins" / "labeling_tool" / "core" / "v5_async_runner.py"
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


def _module_function(name: str) -> ast.FunctionDef:
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"module function {name} is missing")


def _log_panel_method(name: str) -> ast.FunctionDef:
    for node in LOG_PANEL_TREE.body:
        if not isinstance(node, ast.ClassDef) or node.name != "LogPanel":
            continue
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name == name:
                return child
    raise AssertionError(f"LogPanel.{name} is missing")


def _execute_module_function(name: str, *args):
    function_names = []
    if name == "_overall_completion_fraction":
        function_names.append("_assembly_fraction")
    if name in {"_log_severity", "_log_presentation", "_log_indicators"}:
        function_names.append("_log_payload")
    if name in {"_log_presentation", "_log_indicators"}:
        function_names.append("_log_severity")
    if name == "_log_presentation":
        function_names.append("_log_fingerprint")
    function_names.append(name)
    functions = [copy.deepcopy(_module_function(item)) for item in function_names]
    module = ast.fix_missing_locations(
        ast.Module(body=functions, type_ignores=[])
    )
    namespace = {"json": json, "re": re}
    exec(compile(module, str(MONITOR_PATH), "exec"), namespace)
    return namespace[name](*args)


def _method_source(name: str) -> str:
    node = _method(name)
    return ast.get_source_segment(SOURCE, node) or ""


def _execute_method(name: str, instance, *args):
    """Execute one production method without importing the QGIS runtime."""

    function = copy.deepcopy(_method(name))
    module = ast.fix_missing_locations(
        ast.Module(body=[function], type_ignores=[])
    )
    namespace = {
        "_waiting_count": lambda counts: sum(
            int(counts.get(key, 0))
            for key in ("queued", "interrupted", "resetting")
        ),
        "_unit_stage_label": lambda _counts: "空间单元拟合",
        "_timestamp_epoch": lambda _value: 123.0,
        "_elapsed_text": lambda seconds: f"elapsed:{int(seconds)}",
        "_assembly_fraction": lambda _status, _progress: 0.5,
        "ASSEMBLY_PROGRESS_SCALE": 1000,
        "time": time,
    }
    exec(compile(module, str(MONITOR_PATH), "exec"), namespace)
    return namespace[name](instance, *args)


def _execute_log_panel_method(name: str, instance, *args, **kwargs):
    function = copy.deepcopy(_log_panel_method(name))
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"datetime": datetime, "time": time}
    exec(compile(module, str(LOG_PANEL_PATH), "exec"), namespace)
    return namespace[name](instance, *args, **kwargs)


def _self_method_calls(node: ast.AST) -> set[str]:
    calls = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
            continue
        owner = child.func.value
        if isinstance(owner, ast.Name) and owner.id == "self":
            calls.add(child.func.attr)
    return calls


def _called_method_updates_phase_and_title(method_name: str) -> bool:
    """Accept a direct update or a one-hop helper used by the polling method."""

    method = _method(method_name)
    candidates = {method_name, *_self_method_calls(method)}
    for candidate in candidates:
        try:
            block = _method_source(candidate)
        except AssertionError:
            continue
        if "_phase.setText" in block and "setWindowTitle" in block:
            return True
    return False


def test_left_monitor_uses_run_package_and_unit_layers():
    build_ui = _method_source("_build_ui")

    assert "self._run_overview = QLabel(" in build_ui
    assert "self._package_overview = QLabel(" in build_ui
    assert "self._unit_overview = QLabel(" in build_ui

    table_calls = [
        node
        for node in ast.walk(_method("_build_ui"))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "QTableWidget"
    ]
    assert any(
        len(call.args) >= 2
        and isinstance(call.args[1], ast.Constant)
        and call.args[1].value == 7
        for call in table_calls
    )

    for label in (
        "结果流",
        "当前阶段",
        "当前进度",
        "运行",
        "等待",
        "输出面数",
        "失败",
        "阶段耗时",
    ):
        assert label in build_ui
    assert "self._assembly_overview = QLabel(" in build_ui
    assert "self._coverage_overview = QLabel(" in build_ui
    assert "空白/重叠验收" in build_ui


def test_result_stream_table_is_compact_and_scrollable():
    build_ui = _method_source("_build_ui")

    assert "STREAM_TABLE_VISIBLE_ROWS = 5" in SOURCE
    assert (
        "self._streams.setHorizontalScrollBarPolicy(SCROLLBAR_AS_NEEDED)"
        in build_ui
    )
    assert (
        "self._streams.setVerticalScrollBarPolicy(SCROLLBAR_AS_NEEDED)"
        in build_ui
    )
    assert "* STREAM_TABLE_VISIBLE_ROWS" in build_ui
    assert "self._streams.setFixedHeight(stream_table_height)" in build_ui
    assert "left_layout.addWidget(self._streams)" in build_ui
    assert "left_layout.addWidget(self._streams, stretch=2)" not in build_ui


def test_database_binding_accepts_the_run_spec_for_stage_aware_monitoring():
    method = _method("bind_state_database")
    positional = [argument.arg for argument in method.args.args]
    keyword_only = [argument.arg for argument in method.args.kwonlyargs]
    assert "run_spec" in positional + keyword_only

    defaults = [*method.args.defaults, *method.args.kw_defaults]
    assert any(isinstance(default, ast.Constant) and default.value is None for default in defaults)


def test_database_poll_uses_one_snapshot_with_separate_progress_lanes():
    poll = _method("_poll_database")
    snapshot_calls = []
    legacy_calls = []

    for node in ast.walk(poll):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "monitor_snapshot":
            snapshot_calls.append(node)
        if node.func.attr in {
            "job_counts",
            "stream_unit_counts",
            "stream_unit_type_counts",
        }:
            legacy_calls.append(node.func.attr)

    assert len(snapshot_calls) == 1
    assert legacy_calls == []
    poll_source = _method_source("_poll_database")
    assert 'job_counts.get("work_package")' in poll_source
    assert 'job_counts.get("unit_fit")' in poll_source
    assert 'snapshot.get("stream_unit_type_counts")' in poll_source
    assert 'snapshot.get("job_progress")' in poll_source


def test_mixed_v5_job_total_is_not_used_as_the_monitor_progress_bar():
    stage_progress = _method_source("set_stage_progress")
    poll = _method_source("_poll_database")

    # V5AsyncInferenceRunner historically emits ready/(Package + all stream
    # units), e.g. 0/50579.  The monitor must explicitly intercept that legacy
    # aggregate and render a phase-specific Package or unit denominator.
    database_guard = stage_progress.index("if self._database")
    guarded_return = stage_progress.index("return", database_guard)
    progress_bar_write = stage_progress.index("self._bar.setRange")
    assert database_guard < guarded_return < progress_bar_write


def test_database_phase_uses_only_the_current_lane_denominator():
    monitor = SimpleNamespace(_active_global_stage="")
    streams = [{"status": "pending"}, {"status": "pending"}]

    assert _execute_method(
        "_database_phase",
        monitor,
        "running",
        {"ready": 3, "running": 1, "queued": 2},
        {"ready": 5, "running": 2, "queued": 3},
        streams,
    ) == (
        "packages",
        "Work Package 推理 + 空间单元拟合",
        3,
        6,
    )

    assert _execute_method(
        "_database_phase",
        monitor,
        "running",
        {"ready": 6},
        {"ready": 5, "running": 2, "queued": 3},
        streams,
    ) == ("units", "空间单元拟合", 5, 10)

    monitor._active_global_stage = "并行组装"
    assert _execute_method(
        "_database_phase",
        monitor,
        "running",
        {"ready": 3, "running": 1, "queued": 2},
        {"ready": 5, "running": 2, "queued": 3},
        streams,
    ) == ("assembly", "结果流并行组装", 0, 2)
    monitor._active_global_stage = ""


    assert _execute_method(
        "_database_phase",
        monitor,
        "running",
        {"ready": 6},
        {"ready": 10},
        [{"status": "raster_ready"}, {"status": "pending"}],
    ) == ("assembly", "结果流并行组装", 0, 2)


def test_database_phase_terminal_states_are_unambiguous():
    monitor = SimpleNamespace(_active_global_stage="并行组装")
    nonterminal_counts = {"ready": 3, "running": 1, "queued": 2}

    assert _execute_method(
        "_database_phase",
        monitor,
        "stopped",
        nonterminal_counts,
        nonterminal_counts,
        [{"status": "pending"}],
    ) == ("stopped", "已停止，可安全恢复", 0, 1)
    assert _execute_method(
        "_database_phase",
        monitor,
        "failed",
        nonterminal_counts,
        nonterminal_counts,
        [{"status": "failed"}],
    ) == ("failed", "运行失败", 0, 1)
    monitor._active_global_stage = ""
    assert _execute_method(
        "_database_phase",
        monitor,
        "running",
        {"ready": 3, "running": 1, "queued": 2, "failed": 1},
        nonterminal_counts,
        [{"status": "running"}],
    ) == (
        "package_failed",
        "Work Package 失败，后续计算已停止",
        4,
        7,
    )


def test_package_failure_marks_open_streams_failed_before_run_poll_catches_up():
    snapshot = {
        "run": {"status": "running"},
        "job_counts": {
            "work_package": {"failed": 1, "running": 1, "queued": 45},
            "unit_fit": {"queued": 12},
        },
        "active_work_package": None,
        "streams": [
            {"stream_id": "model:test", "status": "running"},
            {"stream_id": "fusion:test", "status": "pending"},
        ],
        "stream_unit_type_counts": {
            "model:test": {"core": {"queued": 6}},
            "fusion:test": {"core": {"queued": 6}},
        },
        "stream_unit_job_type_counts": {
            "model:test": {"core": {"queued": 6}},
            "fusion:test": {"core": {"queued": 6}},
        },
    }
    rows = {}
    errors = []
    monitor = SimpleNamespace(
        _database=SimpleNamespace(monitor_snapshot=lambda _run_id: snapshot),
        _run_id="run-test",
        _package_activity={},
        _active_inference_stream="model:test",
        _active_stage_for_stream=lambda _stream_id: "",
        _set_stream=lambda stream_id, **values: rows.setdefault(stream_id, values),
        _update_database_overviews=lambda **_kwargs: None,
        _render_selected_tiles=lambda: None,
        _log_panel=SimpleNamespace(append_system=errors.append),
    )

    _execute_method("_poll_database", monitor)

    assert errors == []
    assert rows["model:test"]["stage"] == "上游 Work Package 失败"
    assert rows["model:test"]["status"] == "失败"
    assert rows["model:test"]["failures"] == 0
    assert rows["fusion:test"]["status"] == "失败"


def test_same_package_new_attempt_resets_transient_monitor_activity():
    snapshot = {
        "run": {"status": "running"},
        "job_counts": {
            "work_package": {"running": 1},
            "unit_fit": {},
        },
        "active_work_package": {
            "package_id": "package_00000",
            "sequence_no": 0,
            "attempt": 2,
            "progress_current": 0,
            "progress_total": 382,
            "package_started_at": "2026-08-10T01:00:00+00:00",
        },
        "streams": [],
        "stream_unit_type_counts": {},
    }
    errors = []
    monitor = SimpleNamespace(
        _database=SimpleNamespace(monitor_snapshot=lambda _run_id: snapshot),
        _run_id="run-test",
        _package_activity={
            "package_id": "package_00000",
            "attempt": 1,
            "stream_id": "model:old",
            "tile_current": 382,
            "tile_total": 382,
            "effective_batch_size": 1,
            "started_at": 1.0,
            "status": "失败",
        },
        _active_inference_stream="model:old",
        _update_database_overviews=lambda **_kwargs: None,
        _render_selected_tiles=lambda: None,
        _log_panel=SimpleNamespace(append_system=errors.append),
    )

    _execute_method("_poll_database", monitor)

    assert errors == []
    assert monitor._package_activity["package_id"] == "package_00000"
    assert monitor._package_activity["attempt"] == 2
    assert monitor._package_activity["db_current"] == 0
    assert monitor._package_activity["db_total"] == 382
    assert monitor._package_activity["status"] == "运行中"
    assert "stream_id" not in monitor._package_activity
    assert "tile_current" not in monitor._package_activity
    assert "effective_batch_size" not in monitor._package_activity
    assert "started_at" not in monitor._package_activity
    assert monitor._active_inference_stream == ""


def test_poll_uses_unit_fit_jobs_instead_of_stale_stream_unit_activity():
    snapshot = {
        "run": {"status": "running"},
        "job_counts": {
            "work_package": {"ready": 1},
            "unit_fit": {"interrupted": 1},
        },
        "active_work_package": None,
        "streams": [{"stream_id": "model:test", "status": "pending"}],
        # Durable state may remain running after jobs are interrupted.
        "stream_unit_type_counts": {
            "model:test": {"core": {"running": 1}}
        },
        "stream_unit_job_type_counts": {
            "model:test": {"core": {"interrupted": 1}}
        },
    }
    rows = {}
    errors = []

    def set_stream(stream_id, **values):
        rows[stream_id] = values

    monitor = SimpleNamespace(
        _database=SimpleNamespace(monitor_snapshot=lambda _run_id: snapshot),
        _run_id="run-test",
        _package_activity={},
        _active_inference_stream="model:stale",
        _active_stage_for_stream=lambda _stream_id: "",
        _set_stream=set_stream,
        _update_database_overviews=lambda **_kwargs: None,
        _render_selected_tiles=lambda: None,
        _log_panel=SimpleNamespace(append_system=errors.append),
    )

    _execute_method("_poll_database", monitor)

    assert errors == []
    assert rows["model:test"]["unit_progress"] == "0/1"
    assert rows["model:test"]["activity"] == "0/1"
    assert rows["model:test"]["failures"] == 0
    assert rows["model:test"]["stage"] == "空间单元拟合 / 等待依赖"
    assert rows["model:test"]["status"] == "等待"
    assert monitor._active_inference_stream == ""


def test_active_package_inference_is_not_overwritten_by_queued_units():
    event_handler = _method_source("_on_stream_progress")
    package_handler = _method_source("_update_package_activity")
    poll = _method_source("_poll_database")

    assert "work_package" in event_handler
    assert "_active_inference_stream" in package_handler
    assert "_active_inference_stream" in poll


def test_package_events_without_stream_id_are_processed_before_the_guard():
    event_handler = _method_source("_on_stream_progress")
    package_dispatch = event_handler.index("self._update_package_activity(info)")
    stream_guard = event_handler.index("if not stream_id")

    assert package_dispatch < stream_guard
    package_handler = _method_source("_update_package_activity")
    assert "package_tile_materialized" in package_handler
    assert "work_package_finished" in package_handler
    assert "accelerator_worker_paused_low_disk" in package_handler


def test_title_and_phase_follow_the_current_database_stage():
    assert _called_method_updates_phase_and_title("_poll_database")


def test_terminal_snapshot_does_not_disable_a_subsequent_resume():
    poll = _method_source("_poll_database")
    finished = _method_source("_on_finished")

    # bind_state_database polls before runner.resume changes failed/stopped back
    # to running.  A terminal snapshot must therefore not stop the timer.
    assert "_poll_timer.stop" not in poll
    assert finished.index("self._poll_database()") < finished.index(
        "self._poll_timer.stop()"
    )


def test_elapsed_column_tracks_the_current_persisted_assembly_phase():
    build_ui = _method_source("_build_ui")
    step_finished = _method_source("_on_step_finished")
    poll = _method_source("_poll_database")

    assert "阶段耗时" in build_ui
    assert "elapsed" in step_finished
    assert "_set_stream" in step_finished
    assert "phase_started_at" in poll
    assert "elapsed=elapsed" in poll


def test_persisted_assembly_progress_replaces_completed_unit_counts():
    snapshot = {
        "run": {"status": "running"},
        "job_counts": {"work_package": {"ready": 1}, "unit_fit": {"ready": 12}},
        "active_work_package": None,
        "streams": [{"stream_id": "model:test", "status": "assembling"}],
        "stream_runtime_progress": {
            "model:test": {
                "status": "running",
                "phase_name": "写入正式 GPKG",
                "phase_index": 5,
                "phase_total": 9,
                "progress_current": 4,
                "progress_total": 12,
                "feature_count": 999,
                "phase_started_at": "2026-08-20T00:00:00+00:00",
            }
        },
        "stream_unit_type_counts": {"model:test": {"core": {"ready": 12}}},
        "stream_unit_job_type_counts": {"model:test": {"core": {"ready": 12}}},
    }
    rows = {}
    errors = []
    monitor = SimpleNamespace(
        _database=SimpleNamespace(monitor_snapshot=lambda _run_id: snapshot),
        _run_id="run-test",
        _package_activity={},
        _active_inference_stream="",
        _stream_state={},
        _active_stage_for_stream=lambda _stream_id: "并行组装",
        _set_stream=lambda stream_id, **values: rows.update({stream_id: values}),
        _update_database_overviews=lambda **_kwargs: None,
        _render_selected_tiles=lambda: None,
        _log_panel=SimpleNamespace(append_system=errors.append),
    )

    _execute_method("_poll_database", monitor)

    assert errors == []
    assert rows["model:test"]["stage"] == "写入正式 GPKG"
    assert rows["model:test"]["unit_progress"] == "12/12"
    assert rows["model:test"]["stage_progress"] == "4/12"
    assert rows["model:test"]["feature_count"] == 999
    assert rows["model:test"]["activity"] == "—"


def test_monitor_reads_persisted_coverage_validation_summary():
    poll = _method_source("_poll_database")
    coverage = _method_source("_update_coverage_overview")

    assert 'snapshot.get("stream_coverage_validation")' in poll
    assert "gap_area_m2" in coverage
    assert "overlap_area_m2" in coverage
    assert "outside_area_m2" in coverage


def test_log_panel_is_retained_but_collapsed_by_default():
    build_ui = _method_source("_build_ui")
    toggle = _method_source("_set_log_visible")
    quick_filter = _method_source("_show_log_severity")
    assert 'QPushButton("显示日志")' in build_ui
    assert 'QPushButton("Warning 0")' in build_ui
    assert 'QPushButton("Error 0")' in build_ui
    assert 'self._show_log_severity("warning")' in build_ui
    assert 'self._show_log_severity("error")' in build_ui
    assert "self._log_panel.setVisible(False)" in build_ui
    assert "[720, 460] if shown else [1180, 0]" in toggle
    assert "self._log_panel.setVisible(shown)" in toggle
    assert "self._log_panel.scroll_to_latest()" in quick_filter


def test_overall_progress_bar_uses_task_groups_instead_of_time_estimates():
    build_ui = _method_source("_build_ui")
    update = _method_source("_update_database_overviews")
    assert "self._overall_bar = QProgressBar()" in build_ui
    assert "整体任务完成度" in build_ui
    assert "不是剩余时间估算" in build_ui
    assert "_overall_completion_fraction(" in update

    fraction, group_count = _execute_module_function(
        "_overall_completion_fraction",
        "running",
        {
            "work_package": {"ready": 1},
            "fragmentation_v33": {"ready": 2, "queued": 2},
            "unit_fit": {"ready": 5, "queued": 5},
        },
        {
            "work_package": {"completed": 1.0, "total": 1},
            "fragmentation_v33": {"completed": 2.0, "total": 4},
            "unit_fit": {"completed": 5.0, "total": 10},
        },
        [
            {"stream_id": f"model:{index}", "status": "raster_ready"}
            for index in range(4)
        ],
        {},
    )
    assert group_count == 6
    assert fraction == 0.5
    assert _execute_module_function(
        "_overall_completion_fraction", "ready", {}, {}, [], {}
    ) == (1.0, 1)


def test_log_count_separates_raw_stderr_from_confirmed_failures():
    resource_tuning = (
        '[resource-tuning] {"first_failed_batch":128,'
        '"probes":[{"status":"failed","error":"CUDA out of memory"}],'
        '"status":"completed"}'
    )
    assert _execute_module_function(
        "_log_indicators", "system", resource_tuning
    ) == (False, False)
    assert _execute_module_function(
        "_log_indicators",
        "stderr",
        "TypeError: unexpected keyword argument 'run_id'",
    ) == (False, False)
    assert _execute_module_function(
        "_log_severity", "stderr", "GDAL diagnostic output"
    ) == "info"
    assert _execute_module_function(
        "_log_severity", "stderr", "RuntimeWarning: fallback was used"
    ) == "warning"
    assert _execute_module_function(
        "_log_severity", "stdout", '{"event":"probe","status":"error"}'
    ) == "error"
    assert _execute_module_function(
        "_log_indicators",
        "stdout",
        '{"event":"stream_assembly_failed"}',
    ) == (False, True)
    assert "_log_presentation(level, message)" in _method_source("_on_log")


def test_log_presentation_explains_timeout_without_hiding_raw_source():
    presentation = _execute_module_function(
        "_log_presentation",
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
        'source: str,',
        'severity: str,',
        'self._visible_severities: set[str] = {"info", "warning", "error"}',
        '("all", "全部")',
        '("warning", "Warning")',
        '("error", "Error")',
        'QPushButton("技术详情")',
        '("系统处理", event["system_action"])',
        '("用户操作", event["user_action"])',
        'event["repeat_count"] = int(event["repeat_count"]) + 1',
        "self._raw_records.append(raw_record)",
        "pending_records.append(raw_record)",
        "self.log_edit.setMaximumBlockCount(20000)",
        "QTimer.singleShot(100, self._finish_scheduled_rebuild)",
    ):
        assert contract in LOG_PANEL_SOURCE

    assert "self._event_index.clear()" in LOG_PANEL_SOURCE
    assert "self._raw_records.clear()" in LOG_PANEL_SOURCE


def test_log_panel_deduplicates_display_but_preserves_every_raw_record():
    panel = SimpleNamespace(
        _events=[],
        _event_index={},
        _raw_records=[],
        _pending_stderr_records={},
        _event_visible=lambda _event: False,
        _render_event=lambda _event: None,
        _rebuild=lambda: None,
    )
    values = {
        "source": "stderr",
        "severity": "error",
        "title": "任务处理超时",
        "fingerprint": "error:core-037-timeout",
    }

    assert _execute_log_panel_method(
        "append_event", panel, "Core-037 timeout", **values
    ) is True
    values["source"] = "system"
    assert _execute_log_panel_method(
        "append_event", panel, '{"error":"Core-037 timeout"}', **values
    ) is False

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


def test_log_fingerprint_keeps_tasks_and_attempts_separate():
    first = _execute_module_function(
        "_log_fingerprint", "error", "worker failed", "Core-037", 1
    )
    other_task = _execute_module_function(
        "_log_fingerprint", "error", "worker failed", "Core-038", 1
    )
    retry = _execute_module_function(
        "_log_fingerprint", "error", "worker failed", "Core-037", 2
    )

    assert first
    assert len({first, other_task, retry}) == 3
    assert _execute_module_function(
        "_log_fingerprint", "error", "worker failed", "", 0
    ) == ""


def test_error_event_carries_recent_stderr_trace_as_technical_context():
    panel = SimpleNamespace(
        _events=[],
        _event_index={},
        _raw_records=[],
        _pending_stderr_records={},
        _event_visible=lambda _event: False,
        _render_event=lambda _event: None,
        _rebuild=lambda: None,
    )
    for line in ("Traceback (most recent call last):", '  File "worker.py"'):
        assert _execute_log_panel_method(
            "append_event",
            panel,
            line,
            source="stderr",
            severity="info",
        ) is True
    assert _execute_log_panel_method(
        "append_event",
        panel,
        "RuntimeError: disk full",
        source="stderr",
        severity="error",
        title="任务执行失败",
        fingerprint="error:core-037:attempt=1:disk-full",
    ) is True

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
    attach = _method_source("attach_runner")
    process_log = _method_source("_on_process_log")

    assert "log_line = pyqtSignal(str, str)" in RUNNER_SOURCE
    assert "process_log = pyqtSignal(object)" in RUNNER_SOURCE
    assert '"step": str(context.get("label") or "")' in RUNNER_SOURCE
    assert '"unit_id": str(' in RUNNER_SOURCE
    assert '"attempt": int(job.get("attempt") or 0)' in RUNNER_SOURCE
    assert 'getattr(runner, "process_log", None)' in attach
    assert "self._process_log_suppressions" in process_log
    assert "log_context=info" in process_log
    assert 'f"{affected}:attempt={attempt}"' in _method_source("_on_log")


def test_terminal_failures_are_promoted_to_single_structured_log_events():
    step_finished = _method_source("_on_step_finished")
    pipeline_finished = _method_source("_on_finished")
    poll = _method_source("_poll_database")

    assert '"event": "monitor_step_failed"' in step_finished
    assert '"attempt": int(self._step_attempts.get(name) or 1)' in step_finished
    assert '"return_code": int(return_code)' in step_finished
    assert '"event": "monitor_pipeline_failed"' in pipeline_finished
    assert "final_error not in self._logged_error_texts" in pipeline_finished
    assert 'self._on_log("system", f"[monitor-db] {error}")' in poll


def test_tile_and_spatial_unit_details_remain_bounded_and_paged():
    bind = _method_source("bind_state_database")
    page = _method_source("_render_database_page")

    assert "self._page_size = max(1, min(int(page_size), 500))" in bind
    assert ".page_tiles(" in page
    assert ".page_stream_units(" in page
    assert "limit=self._page_size" in page
    assert "offset=offset" in page
    assert "self._tiles.setRowCount(len(values))" in page
    assert "每页最多 {self._page_size}" in page
    assert "{stream_id}" in page


def test_poll_preserves_global_cpu_counts_across_multiple_streams():
    snapshot = {
        "run": {"status": "running"},
        "job_counts": {
            "work_package": {"ready": 1},
            "unit_fit": {"ready": 18, "running": 2},
        },
        "active_work_package": None,
        "streams": [
            {"stream_id": "model:a", "status": "pending"},
            {"stream_id": "model:b", "status": "pending"},
        ],
        "stream_unit_type_counts": {
            stream_id: {"core": {"ready": 9, "running": 1}}
            for stream_id in ("model:a", "model:b")
        },
        "stream_unit_job_type_counts": {
            stream_id: {"core": {"ready": 9, "running": 1}}
            for stream_id in ("model:a", "model:b")
        },
    }
    rows = {}
    overview = {}
    errors = []

    def set_stream(stream_id, **values):
        rows[stream_id] = values

    monitor = SimpleNamespace(
        _database=SimpleNamespace(monitor_snapshot=lambda _run_id: snapshot),
        _run_id="run-test",
        _package_activity={},
        _active_inference_stream="",
        _active_stage_for_stream=lambda _stream_id: "",
        _set_stream=set_stream,
        _update_database_overviews=lambda **values: overview.update(values),
        _render_selected_tiles=lambda: None,
        _log_panel=SimpleNamespace(append_system=errors.append),
    )

    _execute_method("_poll_database", monitor)

    assert errors == []
    assert overview["unit_job_counts"] == {"ready": 18, "running": 2}
    assert rows["model:a"]["activity"] == "1/0"
    assert rows["model:b"]["activity"] == "1/0"


def test_detail_filters_match_tile_selection_and_unit_job_semantics():
    build_ui = _method_source("_build_ui")
    sync = _method_source("_sync_detail_status_options")
    reset = _method_source("_reset_detail_page")
    page = _method_source("_render_database_page")

    assert 'DETAIL_STATUS_OPTIONS["unit"]' in build_ui
    for label in ("等待纳入", "已纳入", "Accepted 跳过", "已排除"):
        assert label in SOURCE
    assert "blockSignals(True)" in sync
    assert "blockSignals(was_blocked)" in sync
    assert "_sync_detail_status_options()" in reset
    assert "Tile 输入清单" in page
    assert "选择状态" in page
    assert "if signature == self._detail_signature" in page
