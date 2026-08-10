"""Source-level contracts for the schema-v2 inference monitor.

These tests intentionally avoid importing QGIS.  They protect the monitor's
control-plane semantics on both QGIS 3/Qt5 and QGIS 4/Qt6 while the live UI is
covered separately by platform acceptance.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MONITOR_PATH = (
    ROOT / "qgis_plugins" / "labeling_tool" / "gui" / "inference_monitor.py"
)
SOURCE = MONITOR_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


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
    }
    exec(compile(module, str(MONITOR_PATH), "exec"), namespace)
    return namespace[name](instance, *args)


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
        and call.args[1].value == 6
        for call in table_calls
    )

    for label in (
        "结果流",
        "当前阶段",
        "单元完成",
        "运行",
        "等待",
        "失败",
        "最近任务耗时",
    ):
        assert label in build_ui


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

    monitor._active_global_stage = "顺序组装"
    assert _execute_method(
        "_database_phase",
        monitor,
        "running",
        {"ready": 3, "running": 1, "queued": 2},
        {"ready": 5, "running": 2, "queued": 3},
        streams,
    ) == ("assembly", "结果流顺序组装", 0, 2)
    monitor._active_global_stage = ""


    assert _execute_method(
        "_database_phase",
        monitor,
        "running",
        {"ready": 6},
        {"ready": 10},
        [{"status": "raster_ready"}, {"status": "pending"}],
    ) == ("assembly", "结果流顺序组装", 0, 2)


def test_database_phase_terminal_states_are_unambiguous():
    monitor = SimpleNamespace(_active_global_stage="顺序组装")
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


def test_elapsed_column_is_the_latest_completed_task_not_a_run_total():
    build_ui = _method_source("_build_ui")
    step_finished = _method_source("_on_step_finished")
    poll = _method_source("_poll_database")

    assert "最近任务耗时" in build_ui
    assert "elapsed" in step_finished
    assert "_set_stream" in step_finished
    for node in ast.walk(_method("_poll_database")):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "_set_stream":
            continue
        assert all(keyword.arg != "elapsed" for keyword in node.keywords)


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
