"""History and in-window log page for the monitor."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from qgis.PyQt.QtCore import QTimer, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from labeling_tool.main.log_panel import LogPanel
from labeling_tool.monitor.monitor_query import query_filter
from labeling_tool.monitor.monitor_time import (
    format_monitor_timestamp,
    monitor_timezone_label,
)
from labeling_tool.monitor.monitor_widgets import (
    AdaptiveTable,
    MonitorComboBox,
    MonitorTextBrowser,
)
from labeling_tool.monitor.pages._shared import (
    left_align_table_headers,
    monitor_panel,
    muted_label,
    style_table,
)
from labeling_tool.qgis_support.qt6_api import (
    HORIZONTAL,
    NO_EDIT_TRIGGERS,
    SCROLLBAR_AS_NEEDED,
    SELECT_ROWS,
    SINGLE_SELECTION,
    USER_ROLE,
)
from labeling_tool.shared.contracts.monitor_contract import (
    MONITOR_EVENT_PAGE_SIZE,
    execution_trigger_label,
)


class EventsPage(QWidget):
    """Own event controls, rows, raw-log continuation, and the LogPanel."""

    history_query_requested = pyqtSignal(bool)
    severity_requested = pyqtSignal(str)
    # request_id, generation, cursor; keep the token order used by the client.
    raw_continuation_requested = pyqtSignal(int, int, object)
    log_counts_changed = pyqtSignal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("MonitorPage")
        self._active = False
        self._rows: list[dict[str, object]] = []
        self._cursor: object = None
        self._exhausted = False
        self._warnings = 0
        self._errors = 0
        self._request_context: dict[str, object] = {}
        self._continuation: (
            tuple[int, int, object, tuple[tuple[str, object], ...]] | None
        ) = None

        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(300)
        self._search_timer.timeout.connect(self._reset_page)
        self._raw_timer = QTimer(self)
        self._raw_timer.setSingleShot(True)
        self._raw_timer.setInterval(50)
        self._raw_timer.timeout.connect(self._request_raw_continuation)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        controls = QHBoxLayout()
        self._scope = MonitorComboBox()
        self._scope.setObjectName("HistoryScope")
        self._scope.addItem("全部事件", "all")
        self._scope.addItem("当前问题", "issues")
        self._scope.addItem("自动恢复", "recovery")
        self._scope.addItem("历史警告/失败", "warnings")
        self._scope.addItem("磁盘原始日志 · Warning", "raw_warning")
        self._scope.addItem("磁盘原始日志 · Error", "raw_error")
        self._execution = MonitorComboBox()
        self._execution.setObjectName("HistoryExecution")
        self._execution.addItem("全部执行", "")
        self._target = MonitorComboBox()
        self._target.setObjectName("HistoryTarget")
        for title, value in (
            ("整个 Run", "all"),
            ("当前结果流", "stream"),
            ("选中对象", "object"),
            ("选中尝试", "attempt"),
        ):
            self._target.addItem(title, value)
        self._search = QLineEdit()
        self._search.setObjectName("HistorySearch")
        self._search.setPlaceholderText("搜索事件、对象或错误")
        self._toggle = QPushButton("显示原始日志")
        self._toggle.setObjectName("LogVisibilityToggle")
        self._toggle.setCheckable(True)
        controls.addWidget(self._scope)
        controls.addWidget(self._execution)
        controls.addWidget(self._target)
        controls.addWidget(self._search, stretch=1)
        controls.addWidget(self._toggle)
        layout.addLayout(controls)

        self._context_label = muted_label(
            "范围：整个 Run · " + monitor_timezone_label()
        )
        self._context_label.setObjectName("HistoryContext")
        self._context_label.setToolTip(
            "事件显示记录时间，日志显示采集时间；没有来源时间的日志会标注‘接收’。"
            "原始时间保留在技术详情中。"
        )
        layout.addWidget(self._context_label)

        self._splitter = QSplitter(HORIZONTAL)
        self._splitter.setObjectName("EventsSplitter")
        history_panel, history_layout = monitor_panel()
        self._table = AdaptiveTable(0, 5)
        self._table.setObjectName("HistoryTable")
        self._table.setHorizontalHeaderLabels(
            ["时间", "对象", "事件", "执行", "当前关联状态"]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(NO_EDIT_TRIGGERS)
        self._table.setSelectionBehavior(SELECT_ROWS)
        self._table.setSelectionMode(SINGLE_SELECTION)
        self._table.setHorizontalScrollBarPolicy(SCROLLBAR_AS_NEEDED)
        self._table.setVerticalScrollBarPolicy(SCROLLBAR_AS_NEEDED)
        left_align_table_headers(self._table)
        self._table.configure_adaptive_columns(
            (154, 166, 204, 124, 146),
            (0.85, 1.0, 1.35, 0.75, 1.05),
            text_cap=420,
        )
        self._table.fit_rows_to_content(max_rows=8)
        history_layout.addWidget(self._table, stretch=3)
        self._detail = MonitorTextBrowser()
        self._detail.setObjectName("HistoryDetail")
        self._detail.setText("选择一条事件查看原因、影响和系统动作。")
        history_layout.addWidget(self._detail, stretch=2)
        self._older = QPushButton("加载更早记录")
        self._older.setObjectName("HistoryLoadOlder")
        history_layout.addWidget(self._older)
        self._splitter.addWidget(history_panel)

        self._log_panel = LogPanel(self)
        self._log_panel.setObjectName("MonitorLogPanel")
        self._log_panel.setToolTip(
            "此区域显示当前窗口缓存；Warning / Error 同时查询左侧磁盘历史。"
        )
        self._splitter.addWidget(self._log_panel)
        self._splitter.setStretchFactor(0, 5)
        self._splitter.setStretchFactor(1, 4)
        self._log_panel.setVisible(False)
        self._splitter.setSizes([1180, 0])
        layout.addWidget(self._splitter, stretch=1)

        self._scope.currentIndexChanged.connect(self._reset_page)
        self._execution.currentIndexChanged.connect(self._reset_page)
        self._target.currentIndexChanged.connect(self._reset_page)
        self._search.textChanged.connect(self._schedule_search)
        self._older.clicked.connect(lambda: self.history_query_requested.emit(True))
        self._toggle.toggled.connect(self.set_log_visible)
        self._table.itemSelectionChanged.connect(self._render_detail)
        self._log_panel.severity_selected.connect(self.severity_requested)
        self._log_panel.cleared.connect(self._clear_counts)
        self._update_log_toggle()
        self.apply_theme("dark")

    def reset(self) -> None:
        self.stop_transient_actions()
        self._rows = []
        self._cursor = None
        self._exhausted = False
        self._request_context = {}
        self._table.setRowCount(0)
        self._older.setEnabled(True)
        self._older.setText("加载更早记录")
        self._detail.setText("选择一条事件查看原因、影响和系统动作。")

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        if not self._active:
            self.stop_transient_actions()

    def current_target(self) -> str:
        return str(self._target.currentData() or "all")

    def set_target(self, target: str) -> None:
        index = self._target.findData(str(target))
        if index < 0:
            return
        if index == self._target.currentIndex():
            self._reset_page()
        else:
            self._target.setCurrentIndex(index)

    def history_request_fields(
        self, selection: Mapping[str, object], *, append: bool = False
    ) -> dict[str, object]:
        self._request_context = dict(selection)
        context = self._history_context(selection)
        fields: dict[str, object] = {
            "scope": str(self._scope.currentData() or "all"),
            "execution_id": str(self._execution.currentData() or ""),
            "search": self._search.text().strip(),
            "context": dict(context),
            "before_event_id": self._cursor if append else None,
            "append": bool(append),
            "page_size": MONITOR_EVENT_PAGE_SIZE,
        }
        self._update_context_label(fields)
        return fields

    def _history_context(self, selection: Mapping[str, object]) -> dict[str, object]:
        target = self.current_target()
        if target == "stream":
            return {
                "stream_id": str(selection.get("stream_id") or "")
                or "__no_selected_stream__"
            }
        if target == "attempt":
            return {
                "span_id": str(
                    selection.get("attempt_id")
                    or selection.get("object_span_id")
                    or "__no_selected_attempt__"
                )
            }
        if target == "object":
            return {
                "object_id": str(selection.get("object_id") or "")
                or "__no_selected_object__",
                "stream_id": str(selection.get("object_stream_id") or ""),
                "job_id": selection.get("job_id"),
            }
        return {}

    def render_history(
        self, payload: Mapping[str, object], *, context: Mapping[str, object]
    ) -> None:
        rows = [dict(row) for row in payload.get("rows") or ()]
        if payload.get("append"):
            self._rows.extend(rows)
        else:
            self._rows = rows
        self._cursor = int(self._rows[-1]["monitor_event_id"]) if self._rows else None
        self._exhausted = len(rows) < int(
            payload.get("page_size") or MONITOR_EVENT_PAGE_SIZE
        )
        raw_log = bool(payload.get("raw_log"))
        if raw_log:
            self._cursor = payload["next_cursor"]
            self._exhausted = not bool(payload["has_more"])
            self._detail.setPlainText(
                f"本段找到 {len(rows)} 条匹配日志。"
                + (
                    "可继续加载更早记录；本段无匹配不代表整个 Run 没有错误。"
                    if payload["has_more"]
                    else "已读取到文件开头。"
                )
                + f"\n无法解析的记录：{payload.get('skipped_records', 0)}。"
                "选择一行查看原始信息。"
            )
        self._older.setEnabled(not self._exhausted)
        self._older.setText("没有更早记录" if self._exhausted else "加载更早记录")
        self._table.setUpdatesEnabled(False)
        try:
            self._table.setRowCount(len(self._rows))
            for row_index, event in enumerate(self._rows):
                execution = str(event.get("execution_id") or "—")
                values = (
                    format_monitor_timestamp(event.get("timestamp"), compact=True),
                    str(event.get("object_id") or event.get("object_type") or "Run"),
                    str(event.get("message") or event.get("event_type") or "—"),
                    execution[:12] + ("…" if len(execution) > 12 else ""),
                    "仅记录发生"
                    if event.get("raw_log")
                    else "已恢复"
                    if event.get("recovered_by_span_id")
                    else str(event.get("level") or "信息"),
                )
                for column, value in enumerate(values):
                    item = self._table.item(row_index, column)
                    if item is None:
                        item = QTableWidgetItem()
                        self._table.setItem(row_index, column, item)
                    item.setText(value)
                    if column == 0:
                        item.setData(USER_ROLE, int(event["monitor_event_id"]))
                        item.setToolTip(
                            format_monitor_timestamp(event.get("timestamp"))
                            + "\n原始时间："
                            + str(event.get("timestamp") or "—")
                        )
                    elif column == 3:
                        item.setToolTip(execution)
        finally:
            self._table.setUpdatesEnabled(True)
        self._table.request_adaptive_layout()

        self._request_context = dict(context)
        self.invalidate_raw_continuation()
        if raw_log and not rows and bool(payload["has_more"]):
            self._continuation = (
                int(payload.get("generation") or 0),
                int(payload.get("request_id") or 0),
                self._cursor,
                self._filter_fingerprint(
                    self.history_request_fields(context, append=True)
                ),
            )
            if self._active:
                self._raw_timer.start()

    def render_executions(self, executions: Sequence[Mapping[str, object]]) -> None:
        selected = str(self._execution.currentData() or "")
        desired = [
            "",
            *[str(execution.get("execution_id") or "") for execution in executions],
        ]
        current = [
            str(self._execution.itemData(index) or "")
            for index in range(self._execution.count())
        ]
        if current == desired:
            return
        blocked = self._execution.blockSignals(True)
        try:
            self._execution.clear()
            self._execution.addItem("全部执行", "")
            for execution in executions:
                execution_id = str(execution.get("execution_id") or "")
                started_at = str(execution.get("started_at") or "")
                label = (
                    f"{execution_trigger_label(execution.get('trigger_type') or '')} · "
                    f"{started_at[:19]} · {execution_id[:8]}"
                )
                self._execution.addItem(label, execution_id)
            index = self._execution.findData(selected)
            self._execution.setCurrentIndex(max(0, index))
        finally:
            self._execution.blockSignals(blocked)

    def show_history_error(self, error: str, *, raw: bool) -> None:
        if raw:
            self._detail.setPlainText(
                "原始日志读取失败，不代表没有错误，也不改变任务或数据库连接状态。\n"
                + str(error)
            )
        else:
            self._detail.setText("历史读取失败；当前推理不受影响。\n" + str(error))

    def show_log_severity(self, severity: str, *, database_bound: bool) -> None:
        if database_bound:
            index = self._scope.findData("raw_" + str(severity))
            if index >= 0:
                if index == self._scope.currentIndex():
                    self._reset_page()
                else:
                    self._scope.setCurrentIndex(index)
            return
        if not self._toggle.isChecked():
            self._toggle.setChecked(True)
        self._log_panel.set_visible_severities({str(severity)})
        self._log_panel.scroll_to_latest()

    def append_log_event(
        self,
        text: str,
        *,
        source: str,
        severity: str,
        title: str = "",
        affected: str = "",
        system_action: str = "",
        user_action: str = "",
        fingerprint: str = "",
        context_key: str = "",
        event_timestamp: object = None,
    ) -> bool:
        created = bool(
            self._log_panel.append_event(
                text,
                source=source,
                severity=severity,
                title=title,
                affected=affected,
                system_action=system_action,
                user_action=user_action,
                fingerprint=fingerprint,
                context_key=context_key,
                event_timestamp=event_timestamp,
            )
        )
        if not created:
            return False
        if severity == "warning":
            self._warnings += 1
        elif severity == "error":
            self._errors += 1
        if severity in {"warning", "error"}:
            self.log_counts_changed.emit(self._warnings, self._errors)
        return True

    def begin_log_batch(self) -> None:
        self._log_panel.begin_batch()

    def end_log_batch(self) -> None:
        self._log_panel.end_batch()

    def clear_log(self) -> None:
        self._log_panel.clear()

    def log_counts(self) -> tuple[int, int]:
        return self._warnings, self._errors

    def set_log_visible(self, visible: bool) -> None:
        shown = bool(visible)
        blocked = self._toggle.blockSignals(True)
        try:
            self._toggle.setChecked(shown)
        finally:
            self._toggle.blockSignals(blocked)
        self._log_panel.setVisible(shown)
        if shown:
            self._log_panel.set_visible_severities({"info", "warning", "error"})
            self._log_panel.refresh_visible()
        self._splitter.setSizes([720, 460] if shown else [1180, 0])
        self._update_log_toggle()

    def invalidate_raw_continuation(self) -> None:
        self._raw_timer.stop()
        self._continuation = None

    def stop_transient_actions(self) -> None:
        self._search_timer.stop()
        self.invalidate_raw_continuation()

    def apply_theme(self, theme: str) -> None:
        self._scope.apply_theme(theme)
        self._execution.apply_theme(theme)
        self._target.apply_theme(theme)
        self._log_panel.set_theme(theme)
        style_table(self._table)

    def _schedule_search(self, *_args: object) -> None:
        if self._active:
            self._search_timer.start()

    def _reset_page(self, *_args: object) -> None:
        self.invalidate_raw_continuation()
        self._rows = []
        self._cursor = None
        self._exhausted = False
        self._table.setRowCount(0)
        self._older.setEnabled(True)
        self._older.setText("加载更早记录")
        raw = str(self._scope.currentData() or "").startswith("raw_")
        self._execution.setEnabled(not raw)
        self._target.setEnabled(not raw)
        self._detail.setPlainText(
            "正在查询磁盘日志；这里只表示曾经发生，恢复状态请查看结构化历史。"
            if raw
            else "选择一条事件查看详情。"
        )
        if self._active:
            self.history_query_requested.emit(False)

    def _render_detail(self) -> None:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._rows):
            return
        event = self._rows[row]
        payload = json.dumps(event.get("payload") or {}, ensure_ascii=False, indent=2)
        current = (
            "原始日志不判定恢复状态，请查看结构化历史"
            if event.get("raw_log")
            else "已恢复"
            if event.get("recovered_by_span_id")
            else "未记录后续恢复"
        )
        self._detail.setPlainText(
            f"事件：{event.get('event_type') or '—'}\n"
            f"时间：{format_monitor_timestamp(event.get('timestamp'))}\n"
            f"对象：{event.get('object_type') or '—'} / "
            f"{event.get('object_id') or '—'}\n"
            f"执行编号：{event.get('execution_id') or '—'}\n"
            f"尝试片段：{event.get('span_id') or '—'}\n"
            f"当前关联状态：{current}\n"
            f"说明：{event.get('message') or '—'}\n\n技术详情：\n"
            f"原始时间：{event.get('timestamp') or '—'}\n{payload}"
        )

    def _clear_counts(self) -> None:
        self._warnings = 0
        self._errors = 0
        self.log_counts_changed.emit(0, 0)
        self._update_log_toggle()

    def _update_log_toggle(self) -> None:
        action = "收起日志" if self._toggle.isChecked() else "显示日志"
        self._toggle.setText(action)

    def _update_context_label(self, fields: Mapping[str, object]) -> None:
        if str(fields.get("scope") or "").startswith("raw_"):
            self._context_label.setText(
                "范围：整个 Run 的磁盘原始日志 · "
                "每页最多扫描 2 MiB／显示 200 条；不判定当前恢复状态"
            )
            return
        context = dict(fields.get("context") or {})
        values = " · ".join(
            str(value) for value in context.values() if value is not None
        )
        suffix = ("  " + values) if values else ""
        self._context_label.setText(
            "范围："
            + self._target.currentText()
            + suffix
            + " · "
            + monitor_timezone_label()
        )

    def _request_raw_continuation(self) -> None:
        continuation = self._continuation
        if continuation is None or not self._active or self._exhausted:
            return
        generation, request_id, cursor, fingerprint = continuation
        current = self._filter_fingerprint(
            self.history_request_fields(self._request_context, append=True)
        )
        if cursor != self._cursor or fingerprint != current:
            return
        if not str(self._scope.currentData() or "").startswith("raw_"):
            return
        self._detail.setPlainText(
            "正在后台查找更早的匹配日志；切换页面或筛选即可停止查找。"
        )
        self.raw_continuation_requested.emit(request_id, generation, cursor)

    @staticmethod
    def _filter_fingerprint(
        fields: Mapping[str, object],
    ) -> tuple[object, ...]:
        return query_filter({"kind": "history", **dict(fields)})
