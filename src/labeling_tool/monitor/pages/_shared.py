"""Stateless Qt construction and styling shared by monitor pages and shell.

Every widget belongs to its caller's Qt tree. This module holds no page,
selection, query, or runtime state.
"""

from __future__ import annotations

from collections.abc import Mapping

from qgis.PyQt.QtWidgets import (
    QFrame,
    QLabel,
    QLayout,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from labeling_tool.monitor.monitor_progress import waiting_count
from labeling_tool.monitor.monitor_theme import (
    TABLE_HEADER_MIN_HEIGHT,
    TABLE_ROW_MIN_HEIGHT,
)
from labeling_tool.monitor.monitor_widgets import AdaptiveTable, ProgressTrack
from labeling_tool.qgis_support.qt6_api import ALIGN_LEFT, ALIGN_VCENTER

PROGRESS_SCALE = 1000


def monitor_panel(*, secondary: bool = False) -> tuple[QFrame, QVBoxLayout]:
    panel = QFrame()
    panel.setProperty("monitorSubPanel" if secondary else "monitorPanel", True)
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(8)
    layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
    return panel, layout


def section_label(text: str) -> QLabel:
    label = QLabel(str(text))
    label.setProperty("sectionTitle", True)
    return label


def muted_label(text: str = "") -> QLabel:
    label = QLabel(str(text))
    label.setProperty("muted", True)
    label.setWordWrap(True)
    label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
    return label


def scrollable_page(page: QWidget, *, minimum_height: int) -> QScrollArea:
    """Preserve page scrolling at the supported 900x560 window minimum."""

    page.setMinimumHeight(int(minimum_height))
    scroll = QScrollArea()
    scroll.setObjectName("MonitorScroll")
    scroll.setWidgetResizable(True)
    scroll.setWidget(page)
    return scroll


def left_align_table_headers(table: QTableWidget) -> None:
    alignment = ALIGN_LEFT | ALIGN_VCENTER
    table.horizontalHeader().setDefaultAlignment(alignment)
    for column in range(table.columnCount()):
        item = table.horizontalHeaderItem(column)
        if item is not None:
            item.setTextAlignment(alignment)


def divider() -> QFrame:
    line = QFrame()
    line.setObjectName("MonitorDivider")
    line.setFixedHeight(1)
    return line


def stat_pair(title: str, value: str = "—") -> tuple[QWidget, QLabel]:
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
    title_label = muted_label(title)
    title_label.setWordWrap(False)
    title_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
    layout.addWidget(title_label)
    label = QLabel(value)
    label.setProperty("value", True)
    label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
    layout.addWidget(label)
    return container, label


def set_badge(label: QLabel, text: str, tone: str) -> None:
    label.setText(text)
    if label.property("status") != tone:
        label.setProperty("status", tone)
        label.style().unpolish(label)
        label.style().polish(label)


def set_progress_bar(bar: ProgressTrack, completed: float, total: float) -> None:
    total_value = max(0.0, float(total or 0))
    completed_value = max(0.0, float(completed or 0))
    if total_value > 0:
        bar.setProperty("progressKnown", True)
        bar.setRange(0, PROGRESS_SCALE)
        bar.setValue(round(min(1.0, completed_value / total_value) * PROGRESS_SCALE))
    else:
        bar.setProperty("progressKnown", False)
        bar.setRange(0, 1)
        bar.setValue(0)


def update_task_lane(
    label: QLabel | None,
    bar: ProgressTrack,
    title: str,
    counts: Mapping[str, int],
    *,
    enabled: bool = True,
) -> None:
    values = dict(counts)
    total = sum(int(value) for value in values.values())
    ready = int(values.get("ready", 0))
    running = int(values.get("running", 0))
    waiting = waiting_count(values)
    failed = int(values.get("failed", 0))
    if not enabled:
        text = f"{title}：未启用"
        tooltip = text
        set_progress_bar(bar, 0, 0)
    elif total < 1:
        text = f"{title}：等待计划"
        tooltip = text
        set_progress_bar(bar, 0, 0)
    else:
        text = f"{title}  {ready:,} / {total:,} 项"
        tooltip = (
            f"{title}：完成 {ready}/{total}；运行 {running}；等待 {waiting}；"
            f"失败 {failed}。按任务计数。"
        )
        set_progress_bar(bar, ready, total)
    if label is not None:
        label.setText(text)
        label.setToolTip(tooltip)
    bar.setToolTip(tooltip)


def style_table(table: AdaptiveTable) -> None:
    """Derive header and row floors from the active font and existing rhythm."""

    table.setShowGrid(False)
    table.setAlternatingRowColors(False)
    left_align_table_headers(table)
    table.ensurePolished()
    line_height = table.fontMetrics().height()
    header_height = max(TABLE_HEADER_MIN_HEIGHT, ((line_height + 15) // 4) * 4)
    row_height = max(TABLE_ROW_MIN_HEIGHT, ((line_height + 17) // 4) * 4)
    table.horizontalHeader().setFixedHeight(header_height)
    table.verticalHeader().setDefaultSectionSize(row_height)
    table.request_adaptive_layout()
