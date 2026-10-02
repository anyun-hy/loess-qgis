"""Result-stream and assembly page for the monitor."""

from __future__ import annotations

from collections.abc import Mapping

from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QGridLayout,
    QLabel,
    QSplitter,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from labeling_tool.monitor.monitor_theme import status_color
from labeling_tool.monitor.monitor_time import format_monitor_timestamp
from labeling_tool.monitor.monitor_widgets import AdaptiveTable, MonitorTextBrowser
from labeling_tool.monitor.pages._shared import (
    left_align_table_headers,
    monitor_panel,
    muted_label,
    section_label,
    style_table,
)
from labeling_tool.monitor.result_stage_presentation import result_stage_presentation
from labeling_tool.qgis_support.qt6_api import (
    HORIZONTAL,
    NO_EDIT_TRIGGERS,
    SCROLLBAR_AS_NEEDED,
    SELECT_ROWS,
    SINGLE_SELECTION,
    USER_ROLE,
)
from labeling_tool.shared.contracts.monitor_contract import (
    ASSEMBLY_PHASES,
    SPAN_STATUS_LABELS,
)


class ResultsPage(QWidget):
    """Own the result-stream table and current stream assembly view."""

    stream_selected = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("MonitorPage")
        self._theme = "dark"
        self._rows: dict[str, int] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)
        streams_panel, streams_layout = monitor_panel()
        streams_layout.addWidget(section_label("结果流"))
        self._streams = AdaptiveTable(0, 7)
        self._streams.setObjectName("ResultsStreamTable")
        self._streams.setHorizontalHeaderLabels(
            [
                "结果流",
                "当前阶段",
                "本步进度",
                "运行/等待",
                "输出面数",
                "问题",
                "阶段耗时",
            ]
        )
        self._streams.verticalHeader().setVisible(False)
        self._streams.setEditTriggers(NO_EDIT_TRIGGERS)
        self._streams.setSelectionBehavior(SELECT_ROWS)
        self._streams.setSelectionMode(SINGLE_SELECTION)
        self._streams.setHorizontalScrollBarPolicy(SCROLLBAR_AS_NEEDED)
        self._streams.setVerticalScrollBarPolicy(SCROLLBAR_AS_NEEDED)
        header = self._streams.horizontalHeader()
        left_align_table_headers(self._streams)
        header.setMinimumSectionSize(64)
        self._streams.configure_adaptive_columns(
            (164, 176, 110, 102, 96, 72, 106),
            (1.0, 1.3, 0.72, 0.7, 0.65, 0.45, 0.72),
            text_cap=360,
        )
        self._streams.fit_rows_to_content(max_rows=5)
        streams_layout.addWidget(self._streams)
        layout.addWidget(streams_panel)

        stages_panel, stages_layout = monitor_panel(secondary=True)
        stages_layout.setContentsMargins(14, 10, 14, 10)
        stages_layout.setSpacing(6)
        stages_layout.addWidget(section_label("结果阶段"))
        self._stage_hint = muted_label(
            "本页显示推理与几何验收进度；人工确认和入库请到分类修整窗口查看。"
        )
        self._stage_hint.setWordWrap(True)
        stages_layout.addWidget(self._stage_hint)
        stage_grid = QGridLayout()
        stage_grid.setHorizontalSpacing(16)
        stage_grid.setVerticalSpacing(8)
        self._stage_labels: dict[str, QLabel] = {}
        for index, (key, title) in enumerate(
            (
                ("output", "1. 推理产出"),
                ("geometry", "2. 几何验收"),
                ("review", "3. 人工确认"),
                ("accepted", "4. 正式入库"),
            )
        ):
            title_label = muted_label(title)
            value_label = QLabel()
            value_label.setObjectName(f"ResultStage{key.title()}")
            value_label.setWordWrap(True)
            stage_grid.addWidget(title_label, (index // 2) * 2, index % 2)
            stage_grid.addWidget(value_label, (index // 2) * 2 + 1, index % 2)
            self._stage_labels[key] = value_label
        stages_layout.addLayout(stage_grid)
        layout.addWidget(stages_panel)

        self._splitter = QSplitter(HORIZONTAL)
        self._splitter.setObjectName("ResultsSplitter")
        steps_panel, steps_layout = monitor_panel()
        steps_layout.addWidget(section_label("组装步骤"))
        self._steps = AdaptiveTable(len(ASSEMBLY_PHASES), 3)
        self._steps.setObjectName("AssemblySteps")
        self._steps.setHorizontalHeaderLabels(["步骤", "状态", "进度/单位"])
        self._steps.verticalHeader().setVisible(False)
        self._steps.setEditTriggers(NO_EDIT_TRIGGERS)
        left_align_table_headers(self._steps)
        for row, (phase, name, unit) in enumerate(ASSEMBLY_PHASES):
            self._steps.setItem(row, 0, QTableWidgetItem(f"{row + 1}. {name}"))
            self._steps.setItem(row, 1, QTableWidgetItem("未开始"))
            self._steps.setItem(row, 2, QTableWidgetItem(f"— / {unit}"))
            self._steps.item(row, 0).setToolTip(phase)
        self._steps.configure_adaptive_columns(
            (172, 118, 128),
            (1.25, 0.8, 0.9),
            text_cap=300,
        )
        self._steps.fit_rows_to_content(max_rows=10)
        steps_layout.addWidget(self._steps)
        steps_layout.addStretch(1)
        self._splitter.addWidget(steps_panel)

        acceptance_panel, acceptance_layout = monitor_panel()
        acceptance_layout.addWidget(section_label("当前步骤详情与验收"))
        self._detail = MonitorTextBrowser()
        self._detail.setObjectName("AssemblyDetail")
        self._detail.setText("选择一个结果流查看十步组装记录。")
        acceptance_layout.addWidget(self._detail, stretch=1)
        self._coverage = muted_label(
            "覆盖验收：尚未执行\n空白面积：—  重叠面积：—  范围外面积：—\n"
            "本页不代表人工修整或 accepted_labels 已完成。"
        )
        self._coverage.setObjectName("ResultCoverage")
        acceptance_layout.addWidget(self._coverage)
        self._splitter.addWidget(acceptance_panel)
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 2)
        layout.addWidget(self._splitter, stretch=1)

        self._streams.itemSelectionChanged.connect(self._emit_selected)
        self._render_stage_summary({}, {})
        self.apply_theme("dark")

    def reset(self) -> None:
        self._rows.clear()
        self._streams.setRowCount(0)
        self._detail.setText("选择一个结果流查看十步组装记录。")
        self._coverage.setText(
            "覆盖验收：尚未执行\n空白面积：—  重叠面积：—  范围外面积：—\n"
            "本页不代表人工修整或 accepted_labels 已完成。"
        )
        self._render_stage_summary({}, {})
        for row, (_phase, _name, unit) in enumerate(ASSEMBLY_PHASES):
            self._steps.item(row, 1).setText("未开始")
            self._steps.item(row, 2).setText(f"— / {unit}")

    def upsert_stream(
        self,
        stream_id: str,
        display_name: str,
        state: Mapping[str, object],
    ) -> None:
        stream_id = str(stream_id)
        current_state = dict(state)
        row = self._rows.get(stream_id)
        if row is None:
            row = self._streams.rowCount()
            self._streams.insertRow(row)
            self._rows[stream_id] = row
        current_progress = (
            current_state.get("stage_progress")
            or current_state.get("unit_progress")
            or current_state.get("progress")
            or "-"
        )
        feature_count = current_state.get("feature_count")
        values = (
            display_name,
            current_state.get("stage") or "等待计划",
            current_progress,
            current_state.get("activity") or "0/0",
            f"{int(feature_count):,}" if feature_count is not None else "—",
            str(current_state.get("failures") or 0),
            current_state.get("elapsed") or "—",
        )
        for column, value in enumerate(values):
            text = str(value)
            item = self._streams.item(row, column)
            if item is None:
                item = QTableWidgetItem()
                self._streams.setItem(row, column, item)
            if item.text() != text:
                item.setText(text)
            if column == 0:
                item.setToolTip(stream_id)
                item.setData(USER_ROLE, stream_id)
                item.setData(USER_ROLE + 1, current_state)
            elif column == 1:
                item.setData(USER_ROLE, str(current_state.get("status") or ""))
                item.setForeground(
                    QColor(
                        status_color(
                            self._theme, str(current_state.get("status") or "")
                        )
                    )
                )
        self._streams.request_adaptive_layout()

    def select_stream(self, stream_id: str) -> None:
        row = self._rows.get(str(stream_id))
        if row is None:
            self._clear_selected_presentation()
            return
        blocked = self._streams.blockSignals(True)
        try:
            self._streams.selectRow(row)
        finally:
            self._streams.blockSignals(blocked)
        self._clear_selected_presentation()

    def render_assembly(
        self,
        *,
        stream_id: str,
        display_name: str,
        runtime: Mapping[str, object],
        phase_statuses: Mapping[str, object],
        coverage: Mapping[str, object],
    ) -> None:
        if not stream_id:
            self._clear_selected_presentation()
            return
        progress = dict(runtime)
        phase = str(progress.get("phase") or "")
        status = str(progress.get("status") or "")
        for row, (phase_key, _name, unit) in enumerate(ASSEMBLY_PHASES):
            recorded = dict(phase_statuses.get(phase_key) or {})
            recorded_status = str(recorded.get("status") or "")
            label = SPAN_STATUS_LABELS.get(recorded_status, "")
            if not label:
                if phase_key == phase and status:
                    label = SPAN_STATUS_LABELS.get(status, status)
                else:
                    label = "记录缺失" if progress else "未开始"
            current = recorded.get("current")
            total = recorded.get("total")
            if phase_key == phase:
                current = progress.get("progress_current")
                total = progress.get("progress_total")
            progress_text = (
                f"{int(current or 0):,}/{int(total):,} {unit}"
                if total not in (None, "", 0, "0")
                else f"— / {unit}"
            )
            self._steps.item(row, 1).setText(label)
            self._steps.item(row, 2).setText(progress_text)
        self._steps.request_adaptive_layout()
        state = self._stream_state(stream_id)
        phase_name = str(progress.get("phase_name") or "尚未开始")
        started = format_monitor_timestamp(progress.get("phase_started_at"))
        message = str(progress.get("message") or "—")
        feature_count = progress.get("feature_count")
        feature_text = f"{int(feature_count):,}" if feature_count is not None else "—"
        self._detail.setText(
            f"结果流：{display_name or stream_id}\n"
            f"当前步骤：{phase_name}\n步骤开始：{started}\n"
            f"当前写入面数：{feature_text}（非最终产物统计）\n说明：{message}"
        )
        coverage_data = dict(coverage)
        self._render_stage_summary(state, coverage_data)
        if coverage_data:
            self._coverage.setText(
                f"覆盖验收：{coverage_data.get('status') or '—'}\n"
                f"空白面积：{coverage_data.get('gap_area_m2', '—')} m²  "
                f"重叠面积：{coverage_data.get('overlap_area_m2', '—')} m²  "
                f"范围外面积：{coverage_data.get('outside_area_m2', '—')} m²\n"
                "单流验收、整个 Run 完成与人工修整分别表达。"
            )
        else:
            self._coverage.setText(
                "覆盖验收：尚未执行\n空白面积：—  重叠面积：—  范围外面积：—\n"
                "本页不代表人工修整或 accepted_labels 已完成。"
            )

    def _stream_state(self, stream_id: str) -> dict[str, object]:
        row = self._rows.get(str(stream_id))
        item = self._streams.item(row, 0) if row is not None else None
        value = item.data(USER_ROLE + 1) if item is not None else {}
        return dict(value) if isinstance(value, Mapping) else {}

    def _render_stage_summary(
        self, state: Mapping[str, object], coverage: Mapping[str, object]
    ) -> None:
        presentation = result_stage_presentation(state, coverage)
        for key, label in self._stage_labels.items():
            label.setText(presentation[key])

    def _clear_selected_presentation(self) -> None:
        self._detail.setText("选择一个结果流查看十步组装记录。")
        self._coverage.setText(
            "覆盖验收：尚未执行\n空白面积：—  重叠面积：—  范围外面积：—\n"
            "本页不代表人工修整或 accepted_labels 已完成。"
        )
        self._render_stage_summary({}, {})
        for row, (_phase, _name, unit) in enumerate(ASSEMBLY_PHASES):
            self._steps.item(row, 1).setText("未开始")
            self._steps.item(row, 2).setText(f"— / {unit}")

    def apply_theme(self, theme: str) -> None:
        self._theme = str(theme)
        style_table(self._streams)
        style_table(self._steps)
        for row in range(self._streams.rowCount()):
            item = self._streams.item(row, 1)
            if item is not None:
                item.setForeground(
                    QColor(status_color(self._theme, str(item.data(USER_ROLE) or "")))
                )

    def _emit_selected(self) -> None:
        row = self._streams.currentRow()
        item = self._streams.item(row, 0) if row >= 0 else None
        stream_id = str(item.data(USER_ROLE) or "") if item is not None else ""
        if stream_id:
            self.stream_selected.emit(stream_id)
