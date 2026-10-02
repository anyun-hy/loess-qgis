"""Overview dashboard page for monitor-owned direct presentation."""

from __future__ import annotations

import time
from collections.abc import Mapping
from html import escape

from qgis.PyQt.QtCore import QSize, Qt, QTimer, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from labeling_tool.monitor.monitor_progress import overview_work_label, waiting_count
from labeling_tool.monitor.monitor_theme import (
    OVERVIEW_ICON_SIZE,
    PALETTES,
    status_color,
)
from labeling_tool.monitor.monitor_time import (
    elapsed_text,
    format_monitor_timestamp,
    monitor_timezone_label,
)
from labeling_tool.monitor.monitor_widgets import (
    AdaptiveTable,
    ProgressTrack,
    monitor_icon,
)
from labeling_tool.monitor.pages._shared import (
    divider,
    left_align_table_headers,
    monitor_panel,
    muted_label,
    section_label,
    set_badge,
    set_progress_bar,
    stat_pair,
    style_table,
    update_task_lane,
)
from labeling_tool.qgis_support.qt6_api import (
    ALIGN_LEFT,
    ALIGN_TOP,
    ALIGN_VCENTER,
    HORIZONTAL,
    NO_EDIT_TRIGGERS,
    RICH_TEXT,
    SELECT_ROWS,
    SINGLE_SELECTION,
    USER_ROLE,
    VERTICAL,
)
from labeling_tool.shared.contracts.monitor_contract import (
    SPAN_STATUS_LABELS,
    effective_device_text,
)


class _WrappedFeedbackLabel(QLabel):
    """Keep wrapped failure text readable inside the scrollable splitter."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setWordWrap(True)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setProperty("muted", True)

    def setText(self, text: str) -> None:
        super().setText(text)
        self._fit_height()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_height()

    def _fit_height(self) -> None:
        needed = max(0, self.heightForWidth(self.width()))
        if needed != self.minimumHeight():
            self.setMinimumHeight(needed)


class OverviewPage(QScrollArea):
    """Own summary cards, overview rows, health, and compact geometry."""

    detail_kind_requested = pyqtSignal(str)
    stream_selected = pyqtSignal(str)
    results_requested = pyqtSignal()
    events_requested = pyqtSignal()
    severity_requested = pyqtSignal(str)
    main_run_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("MonitorScroll")
        self.setWidgetResizable(True)
        self._theme = "dark"
        self._rows: dict[str, int] = {}
        self._recent_events: list[dict[str, object]] = []
        self._compact: bool | None = None
        self._icon_labels: list[tuple[QLabel, str, int, str]] = []
        self._icon_buttons: list[tuple[QPushButton, str, int, str]] = []
        self._extent_timer = QTimer(self)
        self._extent_timer.setSingleShot(True)
        self._extent_timer.timeout.connect(self._sync_extent)

        self._page = QWidget()
        self._page.setObjectName("MonitorPage")
        layout = QVBoxLayout(self._page)
        layout.setContentsMargins(0, 10, 0, 4)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        self._splitter = QSplitter(HORIZONTAL)
        self._splitter.setObjectName("OverviewSplitter")
        self._splitter.setHandleWidth(16)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self._main = QWidget()
        main_layout = QVBoxLayout(self._main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(16)
        main_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        parallel, parallel_layout = monitor_panel()
        parallel_layout.setContentsMargins(14, 10, 14, 10)
        parallel_layout.setSpacing(6)
        parallel_heading = QHBoxLayout()
        parallel_heading.addWidget(section_label("并行执行"))
        parallel_heading.addWidget(muted_label("模型计算与空间处理协同进行"), stretch=1)
        parallel_layout.addLayout(parallel_heading)
        self._cards = QGridLayout()
        self._cards.setObjectName("OverviewCards")
        self._cards.setSpacing(14)

        self._model_card, model_layout = monitor_panel(secondary=True)
        self._model_card.setObjectName("OverviewModelCard")
        model_layout.setContentsMargins(14, 10, 14, 10)
        model_layout.setSpacing(4)
        model_heading = QHBoxLayout()
        model_heading.setSpacing(12)
        model_heading.addWidget(self._icon("chip", size=38))
        model_titles = QVBoxLayout()
        model_titles.setSpacing(2)
        model_titles.addWidget(section_label("模型计算 · 地物识别"))
        self._device_label = muted_label("执行设备：等待 Run 信息")
        model_titles.addWidget(self._device_label)
        model_heading.addLayout(model_titles, stretch=1)
        self._model_badge = QLabel("待开始")
        self._model_badge.setProperty("status", "neutral")
        model_heading.addWidget(self._model_badge)
        model_layout.addLayout(model_heading)
        self._package_metric = QLabel("— / —")
        self._package_metric.setProperty("metric", True)
        self._package_metric.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )
        model_layout.addWidget(self._package_metric)
        model_layout.addWidget(muted_label("推理包已完成"))
        self._package_bar = ProgressTrack()
        self._package_bar.setObjectName("OverviewPackageProgress")
        model_layout.addWidget(self._package_bar)
        model_layout.addWidget(divider())
        model_stats = QHBoxLayout()
        package_stat, self._current_package = stat_pair("当前推理包")
        self._current_package.setObjectName("OverviewCurrentPackage")
        model_stat, self._current_model = stat_pair("当前模型")
        model_stats.addWidget(package_stat, stretch=1)
        model_stats.addWidget(model_stat, stretch=1)
        model_layout.addLayout(model_stats)
        tile_row = QHBoxLayout()
        tile_row.addWidget(muted_label("当前模型影像块"))
        self._tile_count = QLabel("— / —")
        tile_row.addWidget(self._tile_count)
        tile_row.addStretch()
        model_layout.addLayout(tile_row)
        self._tile_bar = ProgressTrack()
        self._tile_bar.setObjectName("OverviewTileProgress")
        model_layout.addWidget(self._tile_bar)
        model_footer = QHBoxLayout()
        self._batch_value = muted_label("批量大小：—")
        model_footer.addWidget(self._batch_value, stretch=1)
        package_button = QPushButton("查看推理包")
        package_button.setProperty("link", True)
        self._icon_button(package_button, "arrow", tone="accent")
        package_button.clicked.connect(
            lambda: self.detail_kind_requested.emit("package")
        )
        model_footer.addWidget(package_button)
        model_layout.addLayout(model_footer)
        self._cards.addWidget(self._model_card, 0, 0)

        self._spatial_card, spatial_layout = monitor_panel(secondary=True)
        self._spatial_card.setObjectName("OverviewSpatialCard")
        spatial_layout.setContentsMargins(14, 10, 14, 10)
        spatial_layout.setSpacing(4)
        spatial_heading = QHBoxLayout()
        spatial_heading.setSpacing(12)
        spatial_heading.addWidget(self._icon("chip", size=38))
        spatial_titles = QVBoxLayout()
        spatial_titles.setSpacing(2)
        spatial_titles.addWidget(section_label("空间处理 · 边界计算"))
        self._spatial_device_label = muted_label("CPU")
        spatial_titles.addWidget(self._spatial_device_label)
        spatial_heading.addLayout(spatial_titles, stretch=1)
        self._spatial_badge = QLabel("待开始")
        self._spatial_badge.setProperty("status", "neutral")
        spatial_heading.addWidget(self._spatial_badge)
        spatial_layout.addLayout(spatial_heading)
        self._unit_metric = QLabel("— / —")
        self._unit_metric.setObjectName("OverviewUnitMetric")
        self._unit_metric.setProperty("metric", True)
        self._unit_metric.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )
        spatial_layout.addWidget(self._unit_metric)
        self._fit_caption = muted_label("边界拟合任务已完成")
        spatial_layout.addWidget(self._fit_caption)
        self._fit_bar = ProgressTrack()
        self._fit_bar.setObjectName("OverviewFitProgress")
        spatial_layout.addWidget(self._fit_bar)
        spatial_layout.addWidget(divider())
        spatial_stats = QHBoxLayout()
        self._spatial_stats: dict[str, QLabel] = {}
        for title in ("运行", "等待", "失败"):
            box, label = stat_pair(title)
            if title == "运行":
                label.setObjectName("SpatialRunning")
            self._spatial_stats[title] = label
            spatial_stats.addWidget(box, stretch=1)
        spatial_layout.addLayout(spatial_stats)
        self._fragment_label = muted_label("碎片治理：等待计划")
        self._confidence_label = muted_label("置信度计算：等待计划")
        self._fragment_bar = ProgressTrack(percentage=False)
        self._confidence_bar = ProgressTrack(percentage=False)
        self._fragment_bar.setFixedHeight(10)
        self._confidence_bar.setFixedHeight(10)
        lane_grid = QGridLayout()
        lane_grid.setHorizontalSpacing(16)
        lane_grid.setVerticalSpacing(4)
        for column, (label, bar) in enumerate(
            (
                (self._fragment_label, self._fragment_bar),
                (self._confidence_label, self._confidence_bar),
            )
        ):
            lane_grid.addWidget(label, 0, column)
            lane_grid.addWidget(bar, 1, column)
            lane_grid.setColumnStretch(column, 1)
        spatial_layout.addLayout(lane_grid)
        spatial_footer = QHBoxLayout()
        self._unit_overview = muted_label("统计范围：全部结果流")
        spatial_footer.addWidget(self._unit_overview, stretch=1)
        spatial_button = QPushButton("查看空间任务")
        spatial_button.setProperty("link", True)
        self._icon_button(spatial_button, "arrow", tone="accent")
        spatial_button.clicked.connect(
            lambda: self.detail_kind_requested.emit("unit_fit")
        )
        spatial_footer.addWidget(spatial_button)
        spatial_layout.addLayout(spatial_footer)
        self._cards.addWidget(self._spatial_card, 0, 1)
        self._cards.setColumnStretch(0, 1)
        self._cards.setColumnStretch(1, 1)
        parallel_layout.addLayout(self._cards)
        main_layout.addWidget(parallel)

        result_panel, result_layout = monitor_panel()
        result_panel.setObjectName("OverviewResultsPanel")
        result_layout.setContentsMargins(14, 10, 14, 10)
        result_layout.setSpacing(6)
        result_header = QHBoxLayout()
        result_header.addWidget(section_label("模型与融合结果"), stretch=1)
        result_button = QPushButton("查看结果与验收")
        result_button.setProperty("link", True)
        self._icon_button(result_button, "arrow", tone="accent")
        result_button.clicked.connect(self.results_requested)
        result_header.addWidget(result_button)
        result_layout.addLayout(result_header)
        self._table = AdaptiveTable(0, 4)
        self._table.setObjectName("OverviewResults")
        self._table.setHorizontalHeaderLabels(["结果", "当前工作", "组装", "验收"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(NO_EDIT_TRIGGERS)
        self._table.setSelectionBehavior(SELECT_ROWS)
        self._table.setSelectionMode(SINGLE_SELECTION)
        left_align_table_headers(self._table)
        self._table.configure_adaptive_columns(
            (168, 178, 132, 124),
            (1.0, 1.45, 0.85, 0.8),
            text_cap=360,
        )
        self._table.fit_rows_to_content(max_rows=5)
        result_layout.addWidget(self._table)
        main_layout.addWidget(result_panel, stretch=1)
        result_layout.addStretch(1)
        self._splitter.addWidget(self._main)

        activity, activity_layout = monitor_panel()
        activity.setObjectName("OverviewActivityPanel")
        self._activity_panel = activity
        activity_layout.setSpacing(16)
        activity_layout.addWidget(section_label("运行动态"))
        self._health_panel = QFrame()
        self._health_panel.setObjectName("MonitorHealth")
        health_layout = QVBoxLayout(self._health_panel)
        health_layout.setContentsMargins(16, 14, 16, 14)
        health_layout.setSpacing(10)
        health_heading = QHBoxLayout()
        self._health_icon = self._icon("shield", size=48, tone="success")
        health_heading.addWidget(self._health_icon)
        self._action_label = section_label("等待运行状态")
        self._action_label.setWordWrap(True)
        health_heading.addWidget(self._action_label, stretch=1)
        health_layout.addLayout(health_heading)
        self._action_description = _WrappedFeedbackLabel(
            "正式 Run 建立后显示运行动态。"
        )
        self._action_description.setWordWrap(True)
        health_layout.addWidget(self._action_description)
        self._failure_details_toggle = QPushButton("查看影响与已完成成果")
        self._failure_details_toggle.setObjectName("OverviewFailureDetailsToggle")
        self._failure_details_toggle.setCheckable(True)
        self._failure_details_toggle.hide()
        health_layout.addWidget(self._failure_details_toggle)
        self._failure_details = _WrappedFeedbackLabel("")
        self._failure_details.setObjectName("OverviewFailureDetails")
        self._failure_details.setWordWrap(True)
        self._failure_details.hide()
        health_layout.addWidget(self._failure_details)
        self._failure_details_toggle.toggled.connect(self._show_failure_details)
        self._open_main_run = QPushButton("到主界面处理此任务")
        self._open_main_run.setObjectName("OverviewOpenMainRun")
        self._open_main_run.setToolTip("仅定位本次 Run；不会自动恢复或重做。")
        self._open_main_run.clicked.connect(self.main_run_requested)
        self._open_main_run.hide()
        health_layout.addWidget(self._open_main_run)
        activity_layout.addWidget(self._health_panel)
        self._history_health = muted_label("历史记录：等待正式 Run")
        activity_layout.addWidget(self._history_health)
        log_counts = QHBoxLayout()
        self._warning_button = QPushButton("Warning 0")
        self._warning_button.setObjectName("OverviewWarningLog")
        self._warning_button.clicked.connect(
            lambda: self.severity_requested.emit("warning")
        )
        self._error_button = QPushButton("Error 0")
        self._error_button.setObjectName("OverviewErrorLog")
        self._error_button.clicked.connect(
            lambda: self.severity_requested.emit("error")
        )
        log_counts.addWidget(self._warning_button)
        log_counts.addWidget(self._error_button)
        activity_layout.addLayout(log_counts)
        activity_layout.addWidget(divider())
        activity_layout.addWidget(section_label("最近事件"))
        self._recent_label = muted_label("尚无已记录事件")
        self._recent_label.setObjectName("OverviewRecentEvents")
        self._recent_label.setAlignment(ALIGN_TOP | ALIGN_LEFT)
        self._recent_label.setTextFormat(RICH_TEXT)
        activity_layout.addWidget(self._recent_label)
        activity_layout.addStretch(1)
        open_events = QPushButton("打开事件与日志")
        self._icon_button(open_events, "document", size=22, tone="text")
        open_events.clicked.connect(self.events_requested)
        activity_layout.addWidget(open_events)
        self._splitter.addWidget(activity)
        self._splitter.setStretchFactor(0, 7)
        self._splitter.setStretchFactor(1, 3)
        self._splitter.setSizes([1060, 400])
        layout.addWidget(self._splitter)
        self.setWidget(self._page)

        self._table.itemSelectionChanged.connect(self._emit_stream)
        self.apply_theme("dark")

    def _icon(self, name: str, *, size: int = 32, tone: str = "text") -> QLabel:
        label = QLabel()
        label.setFixedSize(size, size)
        self._icon_labels.append((label, name, size, tone))
        return label

    def _icon_button(
        self,
        button: QPushButton,
        name: str,
        *,
        tone: str = "muted",
        size: int = 20,
    ) -> QPushButton:
        self._icon_buttons.append((button, name, size, tone))
        button.setIconSize(QSize(size, size))
        return button

    def reset(self) -> None:
        self._rows.clear()
        self._table.setRowCount(0)
        self._recent_events = []
        self._style_recent_events()
        self._history_health.setText("历史记录：等待正式 Run")
        self.render_log_actions(bound=False, warnings=0, errors=0)
        self.clear_main_run_action()

    def clear_main_run_action(self) -> None:
        self._open_main_run.setEnabled(False)
        self._open_main_run.hide()

    def upsert_stream(
        self,
        stream_id: str,
        display_name: str,
        state: Mapping[str, object],
        runtime: Mapping[str, object],
        coverage: Mapping[str, object],
    ) -> None:
        stream_id = str(stream_id)
        state_data = dict(state)
        runtime_data = dict(runtime)
        coverage_data = dict(coverage)
        row = self._rows.get(stream_id)
        if row is None:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._rows[stream_id] = row
        assembly_state = "尚未开始"
        if runtime_data:
            runtime_status = str(runtime_data.get("status") or "")
            assembly_state = SPAN_STATUS_LABELS.get(
                runtime_status, str(runtime_data.get("phase_name") or "运行中")
            )
        elif state_data.get("status") == "成功":
            assembly_state = "已完成"
        coverage_state = (
            str(coverage_data.get("status") or "尚未执行")
            if coverage_data
            else "尚未执行"
        )
        values = (
            display_name,
            overview_work_label(state_data.get("stage")),
            assembly_state,
            {"passed": "验收通过", "failed": "验收失败", "skipped": "未执行"}.get(
                coverage_state, coverage_state
            ),
        )
        for column, value in enumerate(values):
            item = self._table.item(row, column)
            if item is None:
                item = QTableWidgetItem()
                self._table.setItem(row, column, item)
            if item.text() != str(value):
                item.setText(str(value))
            if column == 1:
                item.setToolTip(str(state_data.get("stage") or "尚无阶段记录"))
                item.setData(USER_ROLE, str(state_data.get("status") or ""))
            item.setTextAlignment(ALIGN_LEFT | ALIGN_VCENTER)
            if column == 0:
                item.setToolTip(stream_id)
                item.setData(USER_ROLE, stream_id)
        self._style_stream_row(row)
        self._table.setIconSize(QSize(OVERVIEW_ICON_SIZE, OVERVIEW_ICON_SIZE))
        self._table.request_adaptive_layout()

    def _style_stream_row(self, row: int) -> None:
        identity = self._table.item(row, 0)
        if identity is None:
            return
        stream_id = str(identity.data(USER_ROLE) or "")
        identity.setIcon(
            monitor_icon(
                "layers" if stream_id.startswith("fusion:") else "cube",
                PALETTES[self._theme]["text"],
                OVERVIEW_ICON_SIZE,
            )
        )
        for column in range(1, self._table.columnCount()):
            item = self._table.item(row, column)
            if item is None:
                continue
            status = str(item.data(USER_ROLE) or "") if column == 1 else ""
            running = column == 1 and status == "运行中"
            tone = (
                status_color(self._theme, status)
                if column == 1
                else PALETTES[self._theme]["muted"]
            )
            value = item.text()
            item.setForeground(QColor(tone))
            item.setIcon(
                monitor_icon(
                    "ring"
                    if running
                    else "clock"
                    if value.startswith(("尚未", "等待"))
                    else "check"
                    if value in {"已完成", "验收通过"}
                    else "activity",
                    tone,
                    OVERVIEW_ICON_SIZE,
                )
            )

    def select_stream(self, stream_id: str) -> None:
        row = self._rows.get(str(stream_id))
        if row is None:
            return
        blocked = self._table.blockSignals(True)
        try:
            self._table.selectRow(row)
        finally:
            self._table.blockSignals(blocked)

    def render_execution(
        self,
        *,
        run_status: str,
        package_counts: Mapping[str, int],
        unit_job_counts: Mapping[str, int],
        job_counts: Mapping[str, Mapping[str, int]],
        active_package: Mapping[str, object] | None,
        package_activity: Mapping[str, object],
        run_spec: Mapping[str, object],
        stream_names: Mapping[str, str],
    ) -> None:
        backend, device_name = effective_device_text(run_spec)
        self._device_label.setText(f"{backend} · {device_name}")
        self._device_label.setToolTip("本次 Run 的有效执行设备；不是实时利用率采集。")
        self._spatial_device_label.setText("CPU · 与模型计算并行")

        package_total = sum(int(value) for value in package_counts.values())
        package_ready = int(package_counts.get("ready", 0))
        package_running = int(package_counts.get("running", 0))
        package_failed = int(package_counts.get("failed", 0))
        self._current_package.setText("—")
        self._current_package.setToolTip("")
        self._current_model.setText("—")
        self._tile_count.setText("— / —")
        self._batch_value.setText("批量大小：—")
        self._batch_value.setToolTip("")
        set_progress_bar(self._tile_bar, 0, 0)
        if active_package is not None:
            activity = dict(package_activity)
            sequence = int(active_package.get("sequence_no") or 0) + 1
            package_id = str(active_package.get("package_id") or "")
            stream_id = str(activity.get("stream_id") or "")
            model_text = (
                stream_names.get(stream_id, stream_id) if stream_id else "准备模型"
            )
            if "Fusion" in str(activity.get("status") or ""):
                model_text = "Fusion / 收口"
            tile_current = int(
                activity.get("tile_current")
                if activity.get("tile_current") is not None
                else activity.get("db_current") or 0
            )
            tile_total = int(
                activity.get("tile_total")
                if activity.get("tile_total") is not None
                else activity.get("db_total") or 0
            )
            configured = int(activity.get("configured_batch_size") or 0)
            effective = int(activity.get("effective_batch_size") or configured)
            batch_text = "—"
            if configured:
                batch_text = (
                    str(configured)
                    if not effective or effective == configured
                    else f"{configured}→{effective}"
                )
            if activity.get("started_at") is not None:
                package_elapsed = time.monotonic() - float(activity["started_at"])
            elif activity.get("started_epoch") is not None:
                package_elapsed = time.time() - float(activity["started_epoch"])
            else:
                package_elapsed = 0
            tile_text = f"{tile_current}/{tile_total}" if tile_total else "—"
            self._current_package.setText(f"第 {sequence} 包")
            self._current_package.setToolTip(
                f"{package_id}\n配置/有效 Batch：{batch_text}\n"
                f"包耗时：{elapsed_text(package_elapsed)}\n"
                f"{activity.get('notice') or ''}"
            )
            self._current_model.setText(model_text)
            self._tile_count.setText(tile_text.replace("/", " / "))
            self._batch_value.setText(f"批量大小：{batch_text}")
            self._batch_value.setToolTip(
                "配置 → 当前有效 Batch；" + str(activity.get("notice") or "未记录降档")
            )
            set_progress_bar(self._tile_bar, tile_current, tile_total)
        self._package_metric.setText(f"{package_ready:,} / {package_total:,}")
        set_badge(
            self._model_badge,
            "失败"
            if package_failed
            else "运行中"
            if package_running and run_status == "running"
            else "已完成"
            if package_total and package_ready == package_total
            else "已停止"
            if run_status == "stopped"
            else "等待",
            "failed"
            if package_failed
            else "active"
            if package_running and run_status == "running"
            else "neutral",
        )
        set_progress_bar(self._package_bar, package_ready, package_total)

        unit_total = sum(int(value) for value in unit_job_counts.values())
        unit_ready = int(unit_job_counts.get("ready", 0))
        unit_running = int(unit_job_counts.get("running", 0))
        unit_waiting = waiting_count(unit_job_counts)
        unit_failed = int(unit_job_counts.get("failed", 0))
        blocker_text = (
            f"阻塞（上游 Work Package 失败 {package_failed}） | "
            if package_failed
            else ""
        )
        scaling = dict(run_spec.get("scaling") or {})
        worker_key = (
            "max_cpu_partition_workers_with_package"
            if package_running
            else "max_cpu_partition_workers"
        )
        worker_limit = scaling.get(worker_key)
        worker_text = str(worker_limit) if worker_limit is not None else "—"
        self._unit_overview.setText("全部结果流任务")
        self._unit_overview.setToolTip(
            f"配置并发上限 {worker_text}。{blocker_text}"
            "各分类按任务计数，不能相加作为空间单元数量。"
        )
        spatial_active = sum(
            int((job_counts.get(kind) or {}).get("running", 0))
            for kind in ("fragmentation_v33", "unit_confidence", "unit_fit")
        )
        self._unit_metric.setText(
            f"{unit_ready:,} / {unit_total:,}" if unit_total else "— / —"
        )
        for title, value in (
            ("运行", unit_running),
            ("等待", unit_waiting),
            ("失败", unit_failed),
        ):
            self._spatial_stats[title].setText(f"{value:,}")
        set_badge(
            self._spatial_badge,
            "失败"
            if unit_failed
            else "运行中"
            if spatial_active and run_status == "running"
            else "已停止"
            if run_status == "stopped"
            else "等待"
            if unit_waiting
            else "已完成"
            if unit_total
            else "待开始",
            "failed"
            if unit_failed
            else "active"
            if spatial_active and run_status == "running"
            else "neutral",
        )
        fragmentation_enabled = bool(
            (dict(run_spec.get("fragmentation_regularization") or {})).get(
                "enabled", True
            )
        )
        boundary_enabled = bool(
            (dict(run_spec.get("boundary_fitting") or {})).get("enabled", True)
        )
        self._fit_caption.setText(
            "边界拟合任务已完成" if boundary_enabled else "原始边界任务已完成"
        )
        update_task_lane(
            self._fragment_label,
            self._fragment_bar,
            "碎片治理",
            job_counts.get("fragmentation_v33") or {},
            enabled=fragmentation_enabled,
        )
        update_task_lane(
            self._confidence_label,
            self._confidence_bar,
            "置信度计算",
            job_counts.get("unit_confidence") or {},
            enabled=fragmentation_enabled,
        )
        update_task_lane(
            None,
            self._fit_bar,
            "边界拟合" if boundary_enabled else "原始边界处理",
            job_counts.get("unit_fit") or {},
            enabled=True,
        )

    def render_health(
        self,
        *,
        run_status: str,
        package_failed: int,
        unit_failed: int,
        feedback: Mapping[str, str] | None = None,
        can_open_main_run: bool = False,
    ) -> None:
        if feedback is not None:
            self._action_label.setText(str(feedback["title"]))
            self._action_description.setText(str(feedback["reason"]))
            self._failure_details.setText(
                "\n".join(
                    str(feedback[key])
                    for key in ("impact", "completed", "pending", "next_step")
                )
            )
            self._failure_details_toggle.show()
            self._failure_details.setVisible(self._failure_details_toggle.isChecked())
            self._open_main_run.setEnabled(bool(can_open_main_run))
            self._open_main_run.setVisible(bool(can_open_main_run))
        elif run_status == "resetting":
            self._action_label.setText("正在重做准备")
            self._action_description.setText("失败包正在重置，暂不需要重复操作。")
        elif run_status == "stopped":
            self._action_label.setText("任务已停止")
            self._action_description.setText("恢复与重做失败包仍位于主界面。")
        elif run_status == "ready":
            self._action_label.setText("本次推理已完成")
            self._action_description.setText(
                "可查看结果与验收；人工修整仍在原窗口进行。"
            )
        else:
            self._action_label.setText("当前无需人工处理")
            self._action_description.setText(
                "当前没有已确认的人工处理事项，详情可在日志中查看。"
            )
        if feedback is None:
            self._failure_details_toggle.setChecked(False)
            self._failure_details_toggle.hide()
            self._failure_details.hide()
            self.clear_main_run_action()
        attention = bool(run_status == "failed" or package_failed or unit_failed)
        if self._health_panel.property("attention") != attention:
            self._health_panel.setProperty("attention", attention)
            self._health_panel.style().unpolish(self._health_panel)
            self._health_panel.style().polish(self._health_panel)
        icon_name = (
            "alert" if attention else "clock" if run_status == "stopped" else "shield"
        )
        icon_tone = (
            "warning"
            if attention
            else "muted"
            if run_status == "stopped"
            else "success"
        )
        self._health_panel.setProperty("healthIconName", icon_name)
        self._health_panel.setProperty("healthIconTone", icon_tone)
        self._health_icon.setPixmap(
            monitor_icon(icon_name, PALETTES[self._theme][icon_tone], 48).pixmap(
                QSize(48, 48)
            )
        )

    def _show_failure_details(self, expanded: bool) -> None:
        self._failure_details.setVisible(expanded)
        self._failure_details_toggle.setText(
            "收起影响与已完成成果" if expanded else "查看影响与已完成成果"
        )
        self._extent_timer.start(0)

    def render_history(
        self, health_text: str, recent_events: list[dict[str, object]]
    ) -> None:
        self._history_health.setText(str(health_text))
        self._recent_events = [dict(event) for event in recent_events[:5]]
        self._style_recent_events()

    def render_log_actions(self, *, bound: bool, warnings: int, errors: int) -> None:
        self._warning_button.setText(
            "查看历史警告" if bound else f"历史警告  {int(warnings)}"
        )
        self._error_button.setText(
            "查看历史错误" if bound else f"历史错误  {int(errors)}"
        )
        self._warning_button.setEnabled(bool(bound or warnings > 0))
        self._error_button.setEnabled(bool(bound or errors > 0))

    def apply_theme(self, theme: str) -> None:
        selected = str(theme or "dark").lower()
        if selected not in PALETTES:
            selected = "dark"
        self._theme = selected
        for card in (self._model_card, self._spatial_card):
            for label in card.findChildren(QLabel):
                if label.text().strip():
                    label.ensurePolished()
                    line_height = label.fontMetrics().height()
                    if label.property("metric") or label.property("value"):
                        line_height = max(line_height, label.minimumSizeHint().height())
                    label.setMinimumHeight(line_height)
        palette = PALETTES[selected]
        for label, name, size, tone in self._icon_labels:
            label.setPixmap(
                monitor_icon(name, palette[tone], size).pixmap(QSize(size, size))
            )
        for button, name, size, tone in self._icon_buttons:
            button.setIcon(monitor_icon(name, palette[tone], size))
        icon_name = str(self._health_panel.property("healthIconName") or "shield")
        icon_tone = str(self._health_panel.property("healthIconTone") or "success")
        self._health_icon.setPixmap(
            monitor_icon(icon_name, palette[icon_tone], 48).pixmap(QSize(48, 48))
        )
        for bar in self.findChildren(ProgressTrack):
            bar.setProperty("theme", selected)
            if bar.percentage_visible:
                bar.ensurePolished()
                bar.setMinimumHeight(bar.minimumSizeHint().height())
            bar.update()
        style_table(self._table)
        self._style_recent_events()
        for row in range(self._table.rowCount()):
            self._style_stream_row(row)

    def apply_width(self, width: int) -> None:
        compact = int(width) < 1050
        if compact == self._compact:
            return
        self._compact = compact
        self._cards.removeWidget(self._model_card)
        self._cards.removeWidget(self._spatial_card)
        if compact:
            self._cards.setColumnStretch(1, 0)
            self._cards.addWidget(self._model_card, 0, 0)
            self._cards.addWidget(self._spatial_card, 1, 0)
            self._splitter.setOrientation(VERTICAL)
            self._splitter.setSizes([760, 280])
            self._sync_extent()
            self._extent_timer.start(0)
        else:
            self._extent_timer.stop()
            self._cards.setColumnStretch(1, 1)
            self._cards.addWidget(self._model_card, 0, 0)
            self._cards.addWidget(self._spatial_card, 0, 1)
            self._splitter.setOrientation(HORIZONTAL)
            self._splitter.setSizes([1060, 400])
            for card in (self._model_card, self._spatial_card):
                card.setMinimumHeight(0)
            for panel in (self._main, self._activity_panel):
                panel.setMinimumHeight(0)
            self._splitter.setMinimumHeight(0)
            self._page.setMinimumHeight(0)
            self._splitter.updateGeometry()
            self._page.updateGeometry()

    def stop_transient_actions(self) -> None:
        self._extent_timer.stop()

    def _sync_extent(self) -> None:
        if not self._compact or self._splitter.orientation() != VERTICAL:
            return
        self._splitter.setMinimumHeight(0)
        self._page.setMinimumHeight(0)
        for panel in (self._main, self._activity_panel):
            panel.setMinimumHeight(0)
            self._activate_layout_tree(panel)
        child_heights = [
            self._splitter.widget(index).layout().minimumSize().height()
            for index in range(self._splitter.count())
        ]
        minimum_body_height = sum(child_heights) + self._splitter.handleWidth()
        self._splitter.setMinimumHeight(minimum_body_height)
        self._splitter.setSizes(child_heights)
        page_layout = self._page.layout()
        page_layout.invalidate()
        page_layout.activate()
        margins = page_layout.contentsMargins()
        self._page.setMinimumHeight(
            max(
                page_layout.minimumSize().height(),
                minimum_body_height + margins.top() + margins.bottom(),
            )
        )
        self._page.updateGeometry()

    @classmethod
    def _activate_layout_tree(cls, widget: QWidget) -> None:
        layout = widget.layout()
        if layout is None:
            return

        def refresh_children(parent_layout: QLayout) -> None:
            for index in range(parent_layout.count()):
                item = parent_layout.itemAt(index)
                child_widget = item.widget()
                child_layout = item.layout()
                if child_widget is not None:
                    cls._activate_layout_tree(child_widget)
                elif child_layout is not None:
                    refresh_children(child_layout)
                    child_layout.invalidate()

        refresh_children(layout)
        layout.invalidate()
        layout.activate()
        widget.updateGeometry()

    def _style_recent_events(self) -> None:
        if not self._recent_events:
            self._recent_label.setText("尚无已记录事件")
            return
        palette = PALETTES[self._theme]
        rows = []
        for event in self._recent_events[:5]:
            stamp = format_monitor_timestamp(event.get("timestamp"), compact=True)
            message = str(event.get("message") or event.get("event_type") or "—")
            rows.append(
                f'<tr><td width="18" valign="top" style="color:{palette["accent"]}">●</td>'
                f'<td valign="top" width="124" style="color:{palette["muted"]}">'
                f"{escape(stamp)}</td>"
                f'<td style="color:{palette["text"]}">{escape(message)}</td></tr>'
            )
        self._recent_label.setText(
            '<table cellspacing="0" cellpadding="0" width="100%">'
            + '<tr><td colspan="3" height="16"></td></tr>'.join(rows)
            + "</table>"
        )
        self._recent_label.setToolTip(monitor_timezone_label())

    def _emit_stream(self) -> None:
        row = self._table.currentRow()
        item = self._table.item(row, 0) if row >= 0 else None
        stream_id = str(item.data(USER_ROLE) or "") if item is not None else ""
        if stream_id:
            self.stream_selected.emit(stream_id)
