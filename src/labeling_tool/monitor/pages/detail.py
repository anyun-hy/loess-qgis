"""Object detail controls and object-history presentation."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from html import escape

from qgis.PyQt.QtCore import QTimer, QUrl, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidgetItem,
    QTabWidget,
    QWidget,
)

from labeling_tool.monitor.monitor_time import format_monitor_timestamp
from labeling_tool.monitor.monitor_widgets import (
    AdaptiveTable,
    MonitorComboBox,
    MonitorTextBrowser,
)
from labeling_tool.monitor.pages._shared import (
    left_align_table_headers,
    monitor_panel,
    section_label,
    style_table,
)
from labeling_tool.qgis_support.qt6_api import (
    NO_EDIT_TRIGGERS,
    SCROLLBAR_AS_NEEDED,
    SELECT_ROWS,
    SINGLE_SELECTION,
    USER_ROLE,
)
from labeling_tool.shared.contracts.monitor_contract import SPAN_STATUS_LABELS

UNIT_STATUS_LABELS = {
    "queued": "等待",
    "interrupted": "待恢复",
    "resetting": "正在重置",
    "running": "运行中",
    "ready": "完成",
    "failed": "失败",
    "excluded": "已排除",
}

UNIT_TYPE_LABELS = {
    "core": "Core",
    "seam_horizontal": "横向 Seam",
    "seam_vertical": "纵向 Seam",
    "junction": "Junction",
}

TILE_STATUS_LABELS = {
    "ready": "已纳入",
    "accepted": "Accepted 跳过",
    "excluded": "已排除",
    "queued": "等待纳入",
}

DETAIL_STATUS_OPTIONS = {
    "package": (
        ("全部状态", ""),
        ("等待", "queued"),
        ("待恢复", "interrupted"),
        ("正在重置", "resetting"),
        ("运行中", "running"),
        ("完成", "ready"),
        ("失败", "failed"),
    ),
    "unit": (
        ("全部状态", ""),
        ("等待", "queued"),
        ("待恢复", "interrupted"),
        ("正在重置", "resetting"),
        ("运行中", "running"),
        ("完成", "ready"),
        ("失败", "failed"),
    ),
    "tile": (
        ("全部状态", ""),
        ("等待纳入", "queued"),
        ("已纳入", "ready"),
        ("Accepted 跳过", "accepted"),
        ("已排除", "excluded"),
    ),
}


def _tile_sort_key(tile_id: str) -> tuple[int | str, ...]:
    return tuple(
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", str(tile_id))
    )


class DetailPage(QWidget):
    """Own detail filters, displayed rows, and object-history presentation."""

    detail_query_requested = pyqtSignal()
    object_selected = pyqtSignal(object)
    object_history_requested = pyqtSignal(bool)
    attempt_selected = pyqtSignal(str)
    related_events_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("MonitorPage")
        self._active = False
        self._database_bound = False
        self._stream_id = ""
        self._display_name = ""
        self._page = 0
        self._page_total = 1
        self._page_size = 500
        self._detail_signature: object = None
        self._row_indexes: dict[str, int] = {}
        self._history_rows: list[dict[str, object]] = []
        self._history_cursor: tuple[str, str] | None = None
        self._models_for_attempt: list[dict[str, object]] = []

        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(300)
        self._search_timer.timeout.connect(self._reset_page)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        navigation, navigation_layout = monitor_panel()
        navigation.setObjectName("DetailNavigationPanel")
        navigation_layout.addWidget(section_label("分类"))
        self._navigation = QListWidget()
        self._navigation.setObjectName("DetailNavigation")
        for title, kind in (
            ("模型计算 / 推理包", "package"),
            ("空间处理 / 碎片治理", "fragmentation_v33"),
            ("空间处理 / 置信度计算", "unit_confidence"),
            ("空间处理 / 边界拟合", "unit_fit"),
            ("全局输入 / Tile", "tile"),
        ):
            item = QListWidgetItem(title)
            item.setData(USER_ROLE, kind)
            self._navigation.addItem(item)
        navigation_layout.addWidget(self._navigation, stretch=1)
        layout.addWidget(navigation)

        detail, detail_layout = monitor_panel()
        detail.setObjectName("DetailContentPanel")
        self._title = section_label("当前范围：全局 / 推理包")
        self._title.setObjectName("DetailTitle")
        detail_layout.addWidget(self._title)

        controls = QHBoxLayout()
        self._kind = MonitorComboBox()
        self._kind.setObjectName("DetailKind")
        for title, kind in (
            ("推理包", "package"),
            ("碎片治理", "fragmentation_v33"),
            ("置信度计算", "unit_confidence"),
            ("边界拟合", "unit_fit"),
            ("Tile 输入", "tile"),
        ):
            self._kind.addItem(title, kind)
        self._kind.setVisible(False)
        self._status = MonitorComboBox()
        self._status.setObjectName("DetailStatus")
        for text, value in DETAIL_STATUS_OPTIONS["package"]:
            self._status.addItem(text, value)
        self._search = QLineEdit()
        self._search.setObjectName("DetailSearch")
        self._search.setPlaceholderText("搜索包、任务或单元 ID")
        self._previous = QPushButton("上一页")
        self._previous.setObjectName("DetailPreviousPage")
        self._next = QPushButton("下一页")
        self._next.setObjectName("DetailNextPage")
        self._page_label = section_label("第 1 页")
        self._page_label.setObjectName("DetailPageLabel")
        controls.addWidget(self._status)
        controls.addWidget(self._search, stretch=1)
        controls.addWidget(self._previous)
        controls.addWidget(self._next)
        controls.addWidget(self._page_label)
        detail_layout.addLayout(controls)

        self._table = AdaptiveTable(0, 5)
        self._table.setObjectName("DetailObjectTable")
        self._table.setHorizontalHeaderLabels(
            ["对象ID", "类型", "执行状态", "产物状态", "原因"]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(NO_EDIT_TRIGGERS)
        self._table.setSelectionBehavior(SELECT_ROWS)
        self._table.setSelectionMode(SINGLE_SELECTION)
        self._table.setHorizontalScrollBarPolicy(SCROLLBAR_AS_NEEDED)
        self._table.setVerticalScrollBarPolicy(SCROLLBAR_AS_NEEDED)
        header = self._table.horizontalHeader()
        left_align_table_headers(self._table)
        header.setMinimumSectionSize(72)
        self._table.configure_adaptive_columns(
            (178, 124, 100, 100, 154),
            (1.2, 0.85, 0.7, 0.75, 1.45),
            text_cap=420,
        )
        self._table.fit_rows_to_content(max_rows=8)
        detail_layout.addWidget(self._table, stretch=3)

        self._tabs = QTabWidget()
        self._tabs.setObjectName("DetailObjectTabs")
        self._current = MonitorTextBrowser()
        self._current.setObjectName("ObjectCurrent")
        self._models = MonitorTextBrowser()
        self._models.setObjectName("ObjectModels")
        self._attempts = MonitorTextBrowser()
        self._attempts.setObjectName("ObjectAttempts")
        self._attempts.setOpenLinks(False)
        self._events = MonitorTextBrowser()
        self._events.setObjectName("ObjectEvents")
        self._tabs.addTab(self._current, "当前进度")
        self._tabs.addTab(self._models, "模型明细")
        self._tabs.addTab(self._attempts, "历史尝试")
        self._tabs.addTab(self._events, "相关事件")
        detail_layout.addWidget(self._tabs, stretch=2)

        actions = QHBoxLayout()
        self._more = QPushButton("加载更早尝试")
        self._more.setObjectName("ObjectMore")
        self._more.setEnabled(False)
        related = QPushButton("查看关联事件")
        related.setObjectName("ObjectRelatedEvents")
        copy_id = QPushButton("复制 ID")
        copy_id.setObjectName("ObjectCopyId")
        actions.addWidget(self._more)
        actions.addWidget(related)
        actions.addWidget(copy_id)
        actions.addStretch()
        detail_layout.addLayout(actions)
        layout.addWidget(detail, stretch=1)

        self._navigation.currentRowChanged.connect(self._navigation_changed)
        self._kind.currentIndexChanged.connect(self._reset_page)
        self._status.currentIndexChanged.connect(self._reset_page)
        self._search.textChanged.connect(self._schedule_search)
        self._previous.clicked.connect(self._previous_page)
        self._next.clicked.connect(self._next_page)
        self._table.itemSelectionChanged.connect(self._select_object)
        self._more.clicked.connect(lambda: self.object_history_requested.emit(True))
        related.clicked.connect(self.related_events_requested)
        copy_id.clicked.connect(self._copy_object)
        self._attempts.anchorClicked.connect(self._select_attempt)
        self._navigation.setCurrentRow(0)
        self.apply_theme("dark")

    def reset(self, *, page_size: int = 500) -> None:
        self.stop_transient_actions()
        self._stream_id = ""
        self._display_name = ""
        self._database_bound = False
        self._page_size = max(1, min(int(page_size), 500))
        self._page = 0
        self._page_total = 1
        self._detail_signature = None
        self._row_indexes.clear()
        self._history_rows = []
        self._history_cursor = None
        self._models_for_attempt = []
        self._table.setRowCount(0)
        self._more.setEnabled(False)
        self._previous.setEnabled(False)
        self._next.setEnabled(False)
        self._title.setText("选中结果流：未选择 | 空间单元详情")
        self._page_label.setText("第 1 页")
        for browser in (self._current, self._models, self._attempts, self._events):
            browser.setPlainText("当前筛选没有对象。")

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        if not self._active:
            self._search_timer.stop()

    def stop_transient_actions(self) -> None:
        self._search_timer.stop()

    def set_stream_context(
        self,
        stream_id: str,
        display_name: str,
        live_tiles: Mapping[str, Mapping[str, object]],
        database_bound: bool,
    ) -> None:
        self._stream_id = str(stream_id)
        self._display_name = str(display_name or stream_id)
        self._database_bound = bool(database_bound)
        if self._database_bound:
            if not self._stream_id and self.current_kind() not in {"package", "tile"}:
                self.clear_for_missing_stream()
            return
        self._render_live_tiles(live_tiles)

    def set_kind(self, kind: str) -> None:
        index = self._kind.findData(str(kind))
        if index < 0:
            return
        navigation_index = -1
        for row in range(self._navigation.count()):
            item = self._navigation.item(row)
            if item is not None and str(item.data(USER_ROLE) or "") == str(kind):
                navigation_index = row
                break
        navigation_blocked = self._navigation.blockSignals(True)
        try:
            if navigation_index >= 0:
                self._navigation.setCurrentRow(navigation_index)
        finally:
            self._navigation.blockSignals(navigation_blocked)
        if index != self._kind.currentIndex():
            self._kind.setCurrentIndex(index)
        else:
            self._reset_page()

    def current_kind(self) -> str:
        return str(self._kind.currentData() or "package")

    def current_detail_filter(self, stream_id: str) -> dict[str, object]:
        return {
            "stream_id": str(stream_id),
            "detail_kind": self.current_kind(),
            "status": str(self._status.currentData() or ""),
            "search": self._search.text().strip(),
            "page": self._page,
            "page_size": self._page_size,
        }

    def current_object_filter(
        self, selection: Mapping[str, object]
    ) -> dict[str, object]:
        return {
            "object_id": str(selection.get("object_id") or ""),
            "detail_kind": str(selection.get("kind") or self.current_kind()),
            "stream_id": str(selection.get("object_stream_id") or ""),
            "job_id": selection.get("job_id"),
            "span_id": str(selection.get("attempt_id") or ""),
        }

    def object_history_cursor(self) -> tuple[str, str] | None:
        return self._history_cursor

    def reset_object_history(self) -> None:
        self._history_rows = []
        self._history_cursor = None
        self._models_for_attempt = []
        self._more.setEnabled(False)

    def clear_for_missing_stream(self) -> None:
        self._detail_signature = None
        self._row_indexes.clear()
        self._table.setRowCount(0)
        self._title.setText("选中结果流：未选择 | 空间单元详情")

    def render_detail(self, payload: Mapping[str, object], display_name: str) -> None:
        stream_id = str(payload.get("stream_id") or "")
        kind = str(payload.get("detail_kind") or "unit")
        status = str(payload.get("status") or "")
        search = str(payload.get("search") or "")
        total = int(payload.get("total") or 0)
        page_total = max(1, int(payload.get("page_total") or 1))
        self._page_total = page_total
        self._page = max(0, int(payload.get("page") or 0))
        rows = [dict(row) for row in payload.get("rows") or ()]
        selected_item = self._table.item(self._table.currentRow(), 0)
        selected_id = selected_item.text() if selected_item is not None else ""

        if kind == "tile":
            headers = ["Tile", "Partition", "执行状态", "选择状态", "原因"]
            values = [
                (
                    row["tile_id"],
                    row.get("partition_id") or "-",
                    "全局输入",
                    TILE_STATUS_LABELS.get(str(row["status"]), str(row["status"])),
                    "",
                )
                for row in rows
            ]
            detail_name = "Tile 输入清单"
            row_data = {str(row["tile_id"]): row for row in rows}
        elif kind == "unit":
            headers = ["空间单元", "类型", "执行状态", "产物状态", "原因"]
            values = [
                (
                    row["unit_id"],
                    UNIT_TYPE_LABELS.get(str(row["unit_type"]), str(row["unit_type"])),
                    "记录缺失",
                    UNIT_STATUS_LABELS.get(str(row["status"]), str(row["status"])),
                    row["error"],
                )
                for row in rows
            ]
            detail_name = "Core / Seam / Junction"
            row_data = {str(row["unit_id"]): row for row in rows}
        else:
            labels = {
                "package": "推理包",
                "fragmentation_v33": "碎片治理",
                "unit_confidence": "置信度计算",
                "unit_fit": "边界拟合",
            }
            headers = ["对象ID", "类型", "执行状态", "产物状态", "原因"]
            values = [
                (
                    row["object_id"],
                    UNIT_TYPE_LABELS.get(
                        str(row.get("object_label") or ""),
                        str(row.get("object_label") or labels.get(kind, kind)),
                    ),
                    UNIT_STATUS_LABELS.get(
                        str(row.get("execution_status") or ""),
                        str(row.get("execution_status") or "—"),
                    ),
                    UNIT_STATUS_LABELS.get(
                        str(row.get("artifact_status") or ""),
                        str(row.get("artifact_status") or "尚未就绪"),
                    ),
                    str(row.get("reason") or ""),
                )
                for row in rows
            ]
            detail_name = labels.get(kind, kind)
            row_data = {str(row["object_id"]): row for row in rows}

        self._page_label.setText(f"第 {self._page + 1}/{page_total} 页")
        self._previous.setEnabled(self._page > 0)
        self._next.setEnabled(self._page + 1 < page_total)
        self._title.setText(
            f"选中结果流：{display_name or stream_id} | "
            f"{detail_name} 详情（共 {total} 条，每页最多 {self._page_size}）"
        )
        signature = (
            stream_id,
            kind,
            status,
            search,
            self._page,
            total,
            tuple(values),
            tuple(
                (
                    str(
                        row.get("object_id") or row.get("tile_id") or row.get("unit_id")
                    ),
                    row.get("progress_current"),
                    row.get("progress_total"),
                    row.get("execution_id"),
                    row.get("span_id"),
                    row.get("budget_attempt"),
                )
                for row in rows
            ),
        )
        if signature == self._detail_signature:
            return
        self._detail_signature = signature
        self._row_indexes.clear()
        self._table.setUpdatesEnabled(False)
        blocked = self._table.blockSignals(True)
        try:
            self._table.setHorizontalHeaderLabels(headers)
            if self._table.rowCount() != len(values):
                self._table.setRowCount(len(values))
            for row_index, row_values in enumerate(values):
                object_id = str(row_values[0])
                self._row_indexes[object_id] = row_index
                for column, value in enumerate(row_values):
                    text = str(value)
                    item = self._table.item(row_index, column)
                    if item is None:
                        item = QTableWidgetItem()
                        self._table.setItem(row_index, column, item)
                    if item.text() != text:
                        item.setText(text)
                    if column == 0:
                        item.setData(USER_ROLE, dict(row_data.get(object_id) or {}))
        finally:
            self._table.blockSignals(blocked)
            self._table.setUpdatesEnabled(True)
            self._table.viewport().update()
        self._table.request_adaptive_layout()
        if selected_id in self._row_indexes:
            self._table.selectRow(self._row_indexes[selected_id])
        elif values:
            self._table.selectRow(0)
        if values:
            self._select_object()
        else:
            for browser in (
                self._current,
                self._models,
                self._attempts,
                self._events,
            ):
                browser.setPlainText("当前筛选没有对象。")

    def render_object_history(
        self,
        payload: Mapping[str, object],
        configured_models: Sequence[Mapping[str, object]],
    ) -> None:
        object_id = str(payload.get("object_id") or "")
        spans = [dict(row) for row in payload.get("spans") or ()]
        events = [dict(row) for row in payload.get("events") or ()]
        if payload.get("append"):
            self._history_rows.extend(spans)
        else:
            self._history_rows = spans
        self._history_cursor = (
            (str(spans[-1]["started_at"]), str(spans[-1]["span_id"])) if spans else None
        )
        self._more.setEnabled(bool(payload.get("has_more")))
        self._models_for_attempt = [dict(row) for row in payload.get("models") or ()]

        if self.current_kind() == "package":
            lines = [
                "<b>模型处理记录</b> · 尝试 "
                + escape(str(payload.get("attempt_id") or "—"))
            ]
            models = {
                str(model.get("model_id") or ""): model
                for model in self._models_for_attempt
            }
            for configured in configured_models:
                model_id = str(configured.get("model_id") or "")
                model = models.get(model_id, {})
                metadata = dict(model.get("metadata") or {})
                model_status = SPAN_STATUS_LABELS.get(
                    str(model.get("status") or ""), "尚无本次记录"
                )
                lines.append(
                    "<p><b>"
                    + escape(str(configured.get("display_name") or model_id))
                    + "</b> · "
                    + escape(model_status)
                    + "<br>开始："
                    + escape(format_monitor_timestamp(model.get("started_at")))
                    + "<br>结束："
                    + escape(format_monitor_timestamp(model.get("ended_at")))
                    + "<br>"
                    + escape(str(model.get("message") or ""))
                    + (
                        "<br>" + escape(json.dumps(metadata, ensure_ascii=False))
                        if metadata
                        else ""
                    )
                    + "</p>"
                )
            if not models:
                lines.append(
                    "<p>尚无与该次包尝试关联的模型记录；升级前记录不会反推补齐。</p>"
                )
            self._models.setHtml("".join(lines))

        if self._history_rows:
            attempt_lines: list[tuple[dict[str, object], str]] = []
            for span in self._history_rows:
                span_status = SPAN_STATUS_LABELS.get(
                    str(span.get("status") or ""),
                    str(span.get("status") or "未知"),
                )
                line = (
                    f"第 {int(span.get('attempt_no') or 0)} 次 · {span_status} · "
                    f"{format_monitor_timestamp(span.get('started_at'))}\n"
                    f"  执行 {span.get('execution_id') or '—'} · "
                    f"片段 {span.get('span_id') or '—'}\n"
                    f"  {span.get('message') or '无补充说明'}"
                )
                attempt_lines.append((span, line))
            self._attempts.setHtml(
                "<p>选择尝试可查看对应模型记录与事件；不会改变当前运行。</p>"
                + "".join(
                    '<p><a href="attempt:'
                    + escape(str(span["span_id"]))
                    + '">'
                    + escape(line).replace("\n", "<br>")
                    + "</a></p>"
                    for span, line in attempt_lines
                )
            )
        else:
            self._attempts.setText(
                f"对象：{object_id}\n没有升级后的独立尝试记录；旧 Run 可能记录不完整。"
            )
        if events:
            self._events.setText(
                "\n".join(
                    f"{format_monitor_timestamp(event.get('timestamp'))} · "
                    f"{event.get('message') or event.get('event_type') or '—'}"
                    + (" · 已恢复" if event.get("recovered_by_span_id") else "")
                    for event in events
                )
            )
        else:
            self._events.setText(f"对象：{object_id}\n没有升级后的关联事件。")

    def update_live_tile(self, tile_id: str, state: Mapping[str, object]) -> None:
        row = self._row_indexes.get(str(tile_id))
        if row is None:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._row_indexes[str(tile_id)] = row
        values = (
            tile_id,
            "Tile",
            state.get("status") or "—",
            state.get("progress") or "—",
            state.get("error") or "",
        )
        for column, value in enumerate(values):
            item = self._table.item(row, column)
            if item is None:
                item = QTableWidgetItem()
                self._table.setItem(row, column, item)
            item.setText(str(value))
            if column == 0:
                payload = dict(state)
                payload.setdefault("tile_id", str(tile_id))
                item.setData(USER_ROLE, payload)
        self._title.setText(
            f"选中结果流：{self._display_name or self._stream_id} | "
            f"Tile 详情（已记录 {len(self._row_indexes)} 个）"
        )

    def show_query_error(self, kind: str, error: str) -> None:
        self._detail_signature = None
        if kind == "object_history":
            self._attempts.setText("历史尝试读取失败：\n" + str(error))
            return
        self._row_indexes.clear()
        self._table.setRowCount(0)
        self._title.setText(
            f"选中结果流：{self._display_name or self._stream_id} | "
            f"数据库查询失败: {error}"
        )

    def select_attempt_view(self, attempt_id: str) -> None:
        if any(
            str(row.get("span_id")) == str(attempt_id) for row in self._history_rows
        ):
            self._tabs.setCurrentWidget(self._models)

    def apply_theme(self, theme: str) -> None:
        self._kind.apply_theme(theme)
        self._status.apply_theme(theme)
        style_table(self._table)

    def _render_live_tiles(
        self, live_tiles: Mapping[str, Mapping[str, object]]
    ) -> None:
        values = dict(live_tiles)
        self._table.setHorizontalHeaderLabels(
            ["对象ID", "类型", "执行状态", "当前进度", "原因"]
        )
        self._row_indexes.clear()
        self._table.setRowCount(len(values))
        for row, (tile_id, state) in enumerate(
            sorted(values.items(), key=lambda item: _tile_sort_key(item[0]))
        ):
            self._row_indexes[str(tile_id)] = row
            payload = dict(state)
            payload.setdefault("tile_id", str(tile_id))
            row_values = (
                tile_id,
                "Tile",
                state.get("status") or "—",
                state.get("progress") or "—",
                state.get("error") or "",
            )
            for column, value in enumerate(row_values):
                item = QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(USER_ROLE, payload)
                self._table.setItem(row, column, item)
        if self._stream_id:
            self._title.setText(
                f"选中结果流：{self._display_name or self._stream_id} | "
                f"Tile 详情（已记录 {len(values)} 个）"
            )
        else:
            self._title.setText("选中结果流：未选择 | Tile 详情")
        self._table.request_adaptive_layout()

    def _sync_status_options(self) -> None:
        kind = self.current_kind()
        options = DETAIL_STATUS_OPTIONS.get(kind, DETAIL_STATUS_OPTIONS["unit"])
        desired_values = [value for _text, value in options]
        current_values = [
            str(self._status.itemData(index) or "")
            for index in range(self._status.count())
        ]
        if current_values == desired_values:
            return
        selected = str(self._status.currentData() or "")
        blocked = self._status.blockSignals(True)
        try:
            self._status.clear()
            for text, value in options:
                self._status.addItem(text, value)
            selected_index = self._status.findData(selected)
            self._status.setCurrentIndex(max(0, selected_index))
        finally:
            self._status.blockSignals(blocked)

    def _navigation_changed(self, row: int) -> None:
        item = self._navigation.item(int(row))
        if item is None:
            return
        kind = str(item.data(USER_ROLE) or "package")
        index = self._kind.findData(kind)
        if index >= 0 and index != self._kind.currentIndex():
            self._kind.setCurrentIndex(index)
        else:
            self._reset_page()

    def _schedule_search(self, *_args: object) -> None:
        if self._active:
            self._search_timer.start()

    def _reset_page(self, *_args: object) -> None:
        self._sync_status_options()
        self._page = 0
        self._page_total = 1
        self._detail_signature = None
        self._previous.setEnabled(False)
        self._next.setEnabled(False)
        if self._active:
            self.detail_query_requested.emit()

    def _previous_page(self) -> None:
        if self._page > 0:
            self._page -= 1
            self._detail_signature = None
            self._previous.setEnabled(self._page > 0)
            self._next.setEnabled(self._page + 1 < self._page_total)
            if self._active:
                self.detail_query_requested.emit()

    def _next_page(self) -> None:
        if self._next.isEnabled():
            self._page += 1
            self._detail_signature = None
            self._previous.setEnabled(self._page > 0)
            self._next.setEnabled(self._page + 1 < self._page_total)
            if self._active:
                self.detail_query_requested.emit()

    def _select_object(self) -> None:
        row = self._table.currentRow()
        item = self._table.item(row, 0) if row >= 0 else None
        if item is None:
            return
        object_id = item.text()
        raw = item.data(USER_ROLE)
        value = dict(raw) if isinstance(raw, dict) else {}
        value.setdefault("object_id", object_id)
        value.setdefault("kind", self.current_kind())
        values = [
            self._table.item(row, column).text()
            if self._table.item(row, column) is not None
            else ""
            for column in range(self._table.columnCount())
        ]
        current = int(value.get("progress_current") or 0)
        total = int(value.get("progress_total") or 0)
        progress = f"{current:,}/{total:,}" if total else "—"
        self._current.setText(
            f"对象：{object_id}\n类型：{values[1] if len(values) > 1 else '—'}\n"
            f"执行状态：{values[2] if len(values) > 2 else '—'}\n"
            f"产物状态：{values[3] if len(values) > 3 else '—'}\n"
            f"当前进度：{progress}\n原因：{values[4] if len(values) > 4 else '—'}"
        )
        self._models.setPlainText(
            "正在读取本包各模型的尝试记录…"
            if self.current_kind() == "package"
            else "此分类为空间任务；模型处理详情位于对应推理包。"
        )
        self._attempts.setText(
            f"执行编号：{value.get('execution_id') or '—'}\n"
            f"尝试片段：{value.get('span_id') or '—'}\n"
            "重试预算计数："
            f"{value.get('budget_attempt') if value.get('budget_attempt') is not None else '—'}\n"
            "历史尝试编号与重试预算相互独立。"
        )
        self._events.setText(f"正在读取关联事件…\n对象 ID：{object_id}")
        self.object_selected.emit(value)

    def _select_attempt(self, url: QUrl) -> None:
        identifier = url.toString().removeprefix("attempt:")
        if not any(str(row.get("span_id")) == identifier for row in self._history_rows):
            return
        self.attempt_selected.emit(identifier)
        self._tabs.setCurrentWidget(self._models)

    def _copy_object(self) -> None:
        row = self._table.currentRow()
        item = self._table.item(row, 0) if row >= 0 else None
        QApplication.clipboard().setText(item.text() if item is not None else "")
