"""Native monitor page regressions using synthetic state and fake I/O."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
try:
    from qgis.core import QgsApplication
    from qgis.PyQt.QtCore import QCoreApplication, QTimer
except ModuleNotFoundError:
    raise SystemExit(77)

_ACTIVE_MONITOR_DIALOG = None
_ACTIVE_MONITOR_CLIENT = None


def find_monitor_object(page, name):
    """Observe a named QObject through Qt, without page-private attributes."""
    from qgis.PyQt.QtCore import QObject

    widget = page.findChild(QObject, name)
    assert widget is not None, (type(page).__name__, name)
    return widget


def finish_monitor_case(app, dialog, client):
    """Wait for the real owned query thread before releasing a native fixture."""
    from qgis.PyQt.QtCore import QEventLoop

    global _ACTIVE_MONITOR_CLIENT, _ACTIVE_MONITOR_DIALOG
    finished = []
    loop = QEventLoop()
    dialog.shutdown_finished.connect(lambda: finished.append(not client.is_running()))
    dialog.shutdown_finished.connect(loop.quit)
    dialog.shutdown()
    if not finished:
        QTimer.singleShot(3000, loop.quit)
        loop.exec()
    assert finished == [True], finished
    _ACTIVE_MONITOR_CLIENT = None
    _ACTIVE_MONITOR_DIALOG = None


def monitor_snapshot_contracts(app, root):
    """Run former fake-dialog assertions against the actual QGIS components."""
    from labeling_tool.monitor.inference_monitor import InferenceMonitorDialog
    from labeling_tool.monitor.monitor_query_client import MonitorQueryClient

    class NoReads:
        def execute(self, request):
            raise AssertionError("unbound snapshot fixtures must not query storage")

    global _ACTIVE_MONITOR_CLIENT, _ACTIVE_MONITOR_DIALOG
    client = MonitorQueryClient(executor=NoReads())
    dialog = InferenceMonitorDialog(query_client=client)
    _ACTIVE_MONITOR_CLIENT, _ACTIVE_MONITOR_DIALOG = client, dialog

    def apply(snapshot):
        with patch.object(dialog, "_on_log", wraps=dialog._on_log) as logged:
            dialog._apply_database_snapshot(snapshot)
            app.processEvents()
        return [
            call.args
            for call in logged.call_args_list
            if len(call.args) > 1 and "[monitor-db]" in str(call.args[1])
        ]

    dialog._on_stream_progress(
        {
            "event": "stream_coverage_validation",
            "stream_id": "model:a",
            "status": "passed",
            "gap_area_m2": 0,
            "overlap_area_m2": 0,
            "outside_area_m2": 0,
        }
    )
    dialog._update_coverage_information()
    assert isinstance(dialog._coverage_information, str)
    assert "通过 1/1" in dialog._coverage_information
    assert "空白 0 m²" in dialog._coverage_information
    dialog.reset_run()
    dialog._update_coverage_information()
    assert dialog._coverage_information == "空白/重叠验收：等待组装"

    dialog.reset_run()
    apply(
        {
            "run": {"status": "running"},
            "job_counts": {"work_package": {"running": 1}},
            "active_work_package": {
                "package_id": "package_00000",
                "attempt": 1,
                "progress_current": 382,
                "progress_total": 382,
                "monitor_runtime_json": json.dumps(
                    {
                        "observed_at": "2026-08-10T01:00:00Z",
                        "stream_id": "model:old",
                        "tile_current": 382,
                        "tile_total": 382,
                        "effective_batch_size": 1,
                        "started_at": 1.0,
                        "status": "失败",
                    }
                ),
            },
            "streams": [],
        }
    )

    snapshot = {
        "run": {"status": "running"},
        "job_counts": {"work_package": {"running": 1}, "unit_fit": {}},
        "active_work_package": {
            "package_id": "package_00000",
            "sequence_no": 0,
            "attempt": 2,
            "progress_current": 0,
            "progress_total": 382,
            "package_started_at": "2026-08-10T01:00:00+00:00",
        },
        "streams": [{"stream_id": "model:old", "status": "pending"}],
        "stream_unit_type_counts": {},
    }

    errors = apply(snapshot)

    assert errors == []
    assert dialog._observations.package_view()["package_id"] == "package_00000"
    assert dialog._observations.package_view()["attempt"] == 2
    assert dialog._observations.package_view()["db_current"] == 0
    assert dialog._observations.package_view()["db_total"] == 382
    assert dialog._observations.package_view()["status"] == "运行中"
    assert "Work Package 推理" in dialog.windowTitle()
    assert dialog._phase.text() == "正在识别地物"
    assert "stream_id" not in dialog._observations.package_view()
    assert "tile_current" not in dialog._observations.package_view()
    assert "effective_batch_size" not in dialog._observations.package_view()
    assert "started_at" not in dialog._observations.package_view()
    assert dialog._observations.stream_view("model:old")["state"]["stage"] == "等待计划"

    dialog.reset_run()
    dialog._on_stream_progress(
        {
            "event": "package_model_loading",
            "package_id": "package_00000",
            "stream_id": "model:test",
            "current": 1,
            "total": 1,
        }
    )

    snapshot = {
        "run": {"status": "running"},
        "job_counts": {"work_package": {"ready": 1}, "unit_fit": {"interrupted": 1}},
        "active_work_package": None,
        "streams": [{"stream_id": "model:test", "status": "pending"}],
        "stream_unit_type_counts": {"model:test": {"core": {"running": 1}}},
        "stream_unit_job_type_counts": {"model:test": {"core": {"interrupted": 1}}},
    }

    errors = apply(snapshot)

    assert errors == []
    assert (
        dialog._observations.stream_view("model:test")["state"]["unit_progress"]
        == "0/1"
    )
    assert dialog._observations.stream_view("model:test")["state"]["activity"] == "0/1"
    assert dialog._observations.stream_view("model:test")["state"]["failures"] == 0
    assert (
        dialog._observations.stream_view("model:test")["state"]["stage"]
        == "空间单元拟合 / 等待依赖"
    )
    assert dialog._observations.stream_view("model:test")["state"]["status"] == "等待"
    errors = apply(
        {
            **snapshot,
            "job_counts": {
                "work_package": {"running": 1},
                "unit_fit": {"interrupted": 1},
            },
        }
    )
    assert errors == []
    assert (
        dialog._observations.stream_view("model:test")["state"]["stage"]
        == "空间单元拟合 / 等待依赖"
    )

    dialog.reset_run()

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

    with patch.object(
        dialog.overview_page,
        "render_execution",
        wraps=dialog.overview_page.render_execution,
    ) as render:
        errors = apply(snapshot)

    assert errors == []
    assert render.call_args.kwargs["unit_job_counts"] == {"ready": 18, "running": 2}
    assert dialog._observations.stream_view("model:a")["state"]["activity"] == "1/0"
    assert dialog._observations.stream_view("model:b")["state"]["activity"] == "1/0"

    metric = find_monitor_object(dialog.overview_page, "OverviewUnitMetric")
    assert metric.text() == "18 / 20"
    assert "空间单元拟合" in dialog.windowTitle()
    assert dialog._phase.text() == "正在处理空间边界"
    assert find_monitor_object(dialog.overview_page, "SpatialRunning").text() == "2"
    finish_monitor_case(app, dialog, client)
    return {"real_snapshot_scenarios": 4, "original_assertions_preserved": True}


def monitor_page_queries(app, root):
    """Exercise filters, pagination, incremental rows, logs, and raw cursors."""
    from qgis.PyQt.QtTest import QTest
    from qgis.PyQt.QtWidgets import QPushButton, QVBoxLayout, QWidget

    from labeling_tool.monitor.pages.detail import DetailPage
    from labeling_tool.monitor.pages.events import EventsPage
    from labeling_tool.shared.contracts.monitor_contract import execution_trigger_label

    host = QWidget()
    layout = QVBoxLayout(host)
    detail, events = DetailPage(host), EventsPage(host)
    layout.addWidget(detail)
    layout.addWidget(events)
    host.show()
    detail.set_active(True)
    events.set_active(True)
    detail.reset(page_size=800)
    assert detail.current_detail_filter("model:a")["page_size"] == 500
    detail.reset(page_size=0)
    assert detail.current_detail_filter("model:a")["page_size"] == 1
    detail.reset(page_size=500)

    status = find_monitor_object(detail, "DetailStatus")
    search = find_monitor_object(detail, "DetailSearch")
    for kind in ("package", "unit_fit", "fragmentation_v33", "unit_confidence"):
        detail.set_kind(kind)
        states = {status.itemData(i) for i in range(status.count())}
        assert {
            "",
            "queued",
            "interrupted",
            "resetting",
            "running",
            "ready",
            "failed",
        } <= states
    detail.set_kind("tile")
    assert {status.itemData(i) for i in range(status.count())} == {
        "",
        "queued",
        "ready",
        "accepted",
        "excluded",
    }
    status.setCurrentIndex(status.findData("accepted"))
    detail.set_kind("unit_fit")
    assert status.currentData() == ""
    calls = []
    detail.detail_query_requested.connect(lambda: calls.append(True))
    search.setText("needle")
    search.setText(" needle updated ")
    QTest.qWait(350)
    assert len(calls) == 1, calls
    assert detail.current_detail_filter("model:a")["search"] == "needle updated"

    table = find_monitor_object(detail, "DetailObjectTable")
    title = find_monitor_object(detail, "DetailTitle")
    payload = {
        "stream_id": "model:a",
        "detail_kind": "tile",
        "status": "",
        "search": "",
        "total": 501,
        "page": 0,
        "page_total": 2,
        "rows": [
            {"tile_id": "tile_2", "partition_id": "partition_1", "status": "accepted"}
        ],
    }
    detail.render_detail(payload, "Alpha")
    assert [table.horizontalHeaderItem(i).text() for i in range(5)] == [
        "Tile",
        "Partition",
        "执行状态",
        "选择状态",
        "原因",
    ]
    assert table.item(0, 3).text() == "Accepted 跳过"
    assert "Tile 输入清单" in title.text() and "每页最多 500" in title.text()
    first_item = table.item(0, 0)
    with patch.object(table, "setRowCount", wraps=table.setRowCount) as rebuild:
        detail.render_detail(payload, "Alpha")
    assert table.item(0, 0) is first_item and not rebuild.called
    next(
        button
        for button in detail.findChildren(QPushButton)
        if button.text() == "下一页"
    ).click()
    assert detail.current_detail_filter("model:a")["page"] == 1
    next(
        button
        for button in detail.findChildren(QPushButton)
        if button.text() == "上一页"
    ).click()
    assert detail.current_detail_filter("model:a")["page"] == 0
    detail.render_detail(
        {
            **payload,
            "detail_kind": "unit",
            "rows": [
                {
                    "unit_id": "u1",
                    "unit_type": "seam_horizontal",
                    "status": "failed",
                    "error": "unit failure",
                }
            ],
        },
        "Alpha",
    )
    assert table.item(0, 1).text() == "横向 Seam"
    assert table.item(0, 2).text() == "记录缺失"
    assert table.item(0, 3).text() == "失败"
    assert table.item(0, 4).text() == "unit failure"
    detail.set_stream_context(
        "model:a",
        "Alpha",
        {
            "tile_12": {"status": "running", "progress": "1/3", "error": ""},
            "tile_2": {"status": "ready", "progress": "3/3", "error": ""},
        },
        False,
    )
    assert [table.item(i, 0).text() for i in range(2)] == ["tile_2", "tile_12"]
    stable_item = table.item(0, 0)
    with patch.object(table, "setRowCount", wraps=table.setRowCount) as rebuild:
        detail.update_live_tile(
            "tile_12", {"status": "ready", "progress": "3/3", "error": ""}
        )
    assert not rebuild.called and table.item(0, 0) is stable_item
    assert table.item(1, 2).text() == "ready" and table.item(1, 3).text() == "3/3"

    # History requests use the same public aggregation for first and older pages.
    scope = find_monitor_object(events, "HistoryScope")
    history_search = find_monitor_object(events, "HistorySearch")
    history_intents = []
    events.history_query_requested.connect(history_intents.append)
    scope.setCurrentIndex(scope.findData("issues"))
    assert history_intents == [False], history_intents
    events.set_active(False)
    scope.setCurrentIndex(scope.findData("all"))
    assert history_intents == [False], history_intents
    events.set_active(True)
    execution = find_monitor_object(events, "HistoryExecution")
    executions = [
        {
            "execution_id": "execution-resume-0001",
            "trigger_type": "resume",
            "started_at": "2024-01-02T03:04:05Z",
        }
    ]
    events.render_executions(executions)
    assert execution.itemText(1) == (
        f"{execution_trigger_label('resume')} · 2024-01-02T03:04:05 · executio"
    )
    execution.setCurrentIndex(1)
    with patch.object(execution, "clear", wraps=execution.clear) as rebuild:
        events.render_executions(executions)
    assert execution.currentData() == "execution-resume-0001" and not rebuild.called
    request = events.history_request_fields({}, append=False)
    assert set(request) == {
        "scope",
        "execution_id",
        "search",
        "context",
        "before_event_id",
        "append",
        "page_size",
    }
    assert request["before_event_id"] is None and request["page_size"] == 200
    counts = []
    events.log_counts_changed.connect(
        lambda warnings, errors: counts.append((warnings, errors))
    )
    events.begin_log_batch()
    try:
        assert events.append_log_event(
            "first", source="system", severity="warning", fingerprint="repeat"
        )
        assert not events.append_log_event(
            "repeat", source="system", severity="warning", fingerprint="repeat"
        )
        assert events.append_log_event(
            "failed", source="system", severity="error", fingerprint="failure"
        )
    finally:
        events.end_log_batch()
    assert events.log_counts() == (1, 1) and counts[-1] == (1, 1)
    events.clear_log()
    assert events.log_counts() == (0, 0) and counts[-1] == (0, 0)

    continuation = []
    events.raw_continuation_requested.connect(
        lambda request_id, generation, cursor: continuation.append(
            (request_id, generation, cursor)
        )
    )
    raw = {
        "kind": "history",
        "run_id": "r",
        "generation": 7,
        "request_id": 11,
        "raw_log": True,
        "rows": [],
        "append": False,
        "next_cursor": 321,
        "has_more": True,
        "skipped_records": 0,
        "page_size": 200,
    }

    def schedule_raw():
        events.reset()
        events.set_active(True)
        events.set_target("all")
        scope.setCurrentIndex(scope.findData("raw_error"))
        history_search.setText("")
        events.render_history(raw, context={})
        continuation.clear()

    schedule_raw()
    assert events.history_request_fields({}, append=True)["before_event_id"] == 321
    QTest.qWait(80)
    assert continuation == [(11, 7, 321)], continuation
    for cancel in (
        lambda: events.set_active(False),
        lambda: scope.setCurrentIndex(scope.findData("all")),
        lambda: history_search.setText("different"),
        lambda: events.set_target("object"),
        events.reset,
        events.stop_transient_actions,
    ):
        schedule_raw()
        cancel()
        QTest.qWait(80)
        assert continuation == [], continuation
    schedule_raw()
    events.render_history({**raw, "request_id": 12, "next_cursor": 123}, context={})
    QTest.qWait(80)
    assert continuation == [(12, 7, 123)], continuation
    detail.stop_transient_actions()
    events.stop_transient_actions()
    detail.set_stream_context("old-stream", "Old Run model", {}, True)
    detail.render_detail({**payload, "stream_id": "old-stream"}, "Old Run model")
    assert "Old Run model" in title.text()
    detail.reset()
    assert title.text() == "选中结果流：未选择 | 空间单元详情"
    host.close()
    return {"filters_and_rows": True, "raw_page_fences": 6, "log_counts": True}


def monitor_page_selection(app, root):
    """Preserve result-stream, global-object, and historical-attempt identities."""
    from qgis.PyQt.QtCore import Qt, QUrl
    from qgis.PyQt.QtTest import QTest

    from labeling_tool.monitor.inference_monitor import InferenceMonitorDialog
    from labeling_tool.monitor.monitor_query_client import MonitorQueryClient

    class NoReads:
        def __init__(self):
            self.requests = []

        def execute(self, request):
            self.requests.append(dict(request))
            raise AssertionError(
                "constructor must not dispatch an incomplete page query"
            )

    global _ACTIVE_MONITOR_CLIENT, _ACTIVE_MONITOR_DIALOG
    executor = NoReads()
    client = MonitorQueryClient(executor=executor)
    client.bind("constructor-probe", {})
    dialog = InferenceMonitorDialog(query_client=client)
    _ACTIVE_MONITOR_CLIENT, _ACTIVE_MONITOR_DIALOG = client, dialog
    QTest.qWait(50)
    assert executor.requests == [], executor.requests
    client.unbind()
    dialog._apply_database_snapshot(
        {
            "run": {"status": "running"},
            "job_counts": {},
            "streams": [
                {"stream_id": name, "status": "pending"}
                for name in ("model:a", "model:b")
            ],
        }
    )
    overview = find_monitor_object(dialog.overview_page, "OverviewResults")
    results = find_monitor_object(dialog.results_page, "ResultsStreamTable")
    dialog.results_page.reset()
    for name in ("model:b", "model:a"):
        dialog.results_page.upsert_stream(
            name, name, dialog._observations.stream_view(name)["state"]
        )
    overview.selectRow(1)
    assert dialog._selection["stream_id"] == "model:b"
    assert results.currentRow() == 0
    assert results.item(0, 0).data(Qt.ItemDataRole.UserRole) == "model:b"

    detail = dialog.detail_page
    detail.set_kind("package")
    package = {
        "object_id": "package_1",
        "object_label": "推理包",
        "stream_id": "",
        "job_id": 5,
        "span_id": "current-span",
        "execution_id": "exec-current",
        "execution_status": "running",
        "artifact_status": "queued",
        "reason": "",
        "progress_current": 2,
        "progress_total": 10,
        "budget_attempt": 3,
    }
    payload = {
        "stream_id": "model:b",
        "detail_kind": "package",
        "total": 1,
        "page": 0,
        "page_total": 1,
        "rows": [package],
    }
    detail.render_detail(payload, "Beta")
    assert dialog._selection["stream_id"] == "model:b"
    assert dialog._selection["object_stream_id"] == ""
    current = find_monitor_object(detail, "ObjectCurrent").toPlainText()
    assert "当前进度：2/10" in current
    assert (
        "重试预算计数：3" in find_monitor_object(detail, "ObjectAttempts").toPlainText()
    )
    dialog.events_page.set_target("stream")
    assert dialog.events_page.history_request_fields(dialog._selection)["context"] == {
        "stream_id": "model:b"
    }
    dialog.events_page.set_target("object")
    assert dialog.events_page.history_request_fields(dialog._selection)["context"] == {
        "object_id": "package_1",
        "stream_id": "",
        "job_id": 5,
    }
    dialog.events_page.set_target("attempt")
    assert dialog.events_page.history_request_fields(dialog._selection)["context"] == {
        "span_id": "current-span"
    }
    assert detail.current_object_filter(dialog._selection)["span_id"] == ""

    configured = [{"model_id": "alpha", "display_name": "Alpha"}]
    history = {
        "object_id": "package_1",
        "detail_kind": "package",
        "attempt_id": "older-span",
        "spans": [
            {
                "span_id": "current-span",
                "execution_id": "exec-current",
                "attempt_no": 3,
                "started_at": "2024-01-02T00:00:00Z",
                "status": "failed",
                "message": "failure reason",
            }
        ],
        "models": [
            {
                "model_id": "alpha",
                "status": "completed",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:00:04Z",
                "message": "model detail",
                "metadata": {"batch": 8},
            }
        ],
        "events": [
            {
                "timestamp": "2024-01-01T00:00:03Z",
                "message": "prior error",
                "recovered_by_span_id": "recovery",
            }
        ],
        "has_more": True,
        "append": False,
    }
    detail.render_object_history(history, configured)
    detail.render_object_history(
        {
            **history,
            "append": True,
            "has_more": False,
            "spans": [
                {
                    "span_id": "older-span",
                    "execution_id": "exec-old",
                    "attempt_no": 2,
                    "started_at": "2024-01-01T00:00:00Z",
                    "status": "failed",
                    "message": "older reason",
                }
            ],
        },
        configured,
    )
    attempts = find_monitor_object(detail, "ObjectAttempts")
    assert (
        "current-span" in attempts.toPlainText()
        and "older-span" in attempts.toPlainText()
    )
    assert (
        "exec-old" in attempts.toPlainText()
        and "older reason" in attempts.toPlainText()
    )
    assert detail.object_history_cursor() == ("2024-01-01T00:00:00Z", "older-span")
    models = find_monitor_object(detail, "ObjectModels").toPlainText()
    assert all(value in models for value in ("Alpha", "model detail", "batch", "8"))
    assert "已恢复" in find_monitor_object(detail, "ObjectEvents").toPlainText()
    attempts.anchorClicked.emit(QUrl("attempt:older-span"))
    assert dialog._selection["attempt_id"] == "older-span"
    assert detail.current_object_filter(dialog._selection)["span_id"] == "older-span"
    detail.render_detail(
        {**payload, "rows": [{**package, "progress_current": 4}]}, "Beta"
    )
    assert dialog._selection["attempt_id"] == "older-span"
    assert (
        "当前进度：4/10" in find_monitor_object(detail, "ObjectCurrent").toPlainText()
    )
    assert dialog.events_page.history_request_fields(dialog._selection)["context"] == {
        "span_id": "older-span"
    }
    # A result already admitted by the client must still match current UI filters.
    stale_request = {
        "kind": "detail",
        "run_id": "r",
        "generation": 7,
        "request_id": 11,
        **detail.current_detail_filter("model:b"),
    }
    detail.set_kind("tile")
    detail.render_detail(
        {
            "stream_id": "model:b",
            "detail_kind": "tile",
            "total": 1,
            "page": 0,
            "page_total": 1,
            "rows": [
                {"tile_id": "current-tile", "partition_id": "p", "status": "ready"}
            ],
        },
        "Beta",
    )
    dialog._on_query_result(
        stale_request,
        {
            **payload,
            "kind": "detail",
            "rows": [{**package, "object_id": "stale-package"}],
        },
    )
    table = find_monitor_object(detail, "DetailObjectTable")
    assert table.item(0, 0).text() == "current-tile"
    previous_error = dialog._last_snapshot_error
    dialog._on_query_failed(stale_request, {"kind": "detail", "error": "stale failure"})
    assert dialog._last_snapshot_error == previous_error
    assert table.item(0, 0).text() == "current-tile"
    finish_monitor_case(app, dialog, client)
    return {
        "stream_identity_order": True,
        "object_stream_separate": True,
        "attempt_persistence": True,
    }


def main() -> None:
    app = QgsApplication([], False)
    app.initQgis()
    with tempfile.TemporaryDirectory(prefix="loess-monitor-pages-") as temporary:
        try:
            result = globals()[sys.argv[2]](app, Path(temporary))
        finally:
            dialog, client = _ACTIVE_MONITOR_DIALOG, _ACTIVE_MONITOR_CLIENT
            if dialog is not None:
                dialog.shutdown()
                deadline = time.monotonic() + 3.0
                while client is not None and time.monotonic() < deadline:
                    try:
                        if not client.is_running():
                            break
                    except RuntimeError:
                        # A completed retirement may already have deleted Qt children.
                        break
                    QCoreApplication.processEvents()
                    time.sleep(0.01)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
