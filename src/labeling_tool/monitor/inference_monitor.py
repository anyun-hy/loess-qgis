"""Modeless result-stream monitor with on-demand tile details."""

from __future__ import annotations

import copy
import re
import time

from qgis.core import QgsSettings
from qgis.PyQt.QtCore import QObject, QSize, QTimer, pyqtSignal
from qgis.PyQt.QtGui import QFont, QFontDatabase
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from labeling_tool.monitor.failure_feedback import failure_feedback
from labeling_tool.monitor.monitor_observations import (
    MonitorLogObservations,
    MonitorObservations,
    PresentedLog,
)
from labeling_tool.monitor.monitor_progress import (
    database_phase,
    overall_completion_fraction,
)
from labeling_tool.monitor.monitor_query import (
    QueryFailure,
    QueryMessage,
    QueryResult,
    query_matches_filter,
)
from labeling_tool.monitor.monitor_query_client import MonitorQueryClient
from labeling_tool.monitor.monitor_theme import (
    BODY_FONT_PT,
    MONITOR_STYLE,
    PALETTES,
)
from labeling_tool.monitor.monitor_time import elapsed_text
from labeling_tool.monitor.monitor_widgets import (
    MonitorTextBrowser,
    OverallProgressTrack,
    monitor_icon,
)
from labeling_tool.monitor.pages._shared import (
    divider,
    monitor_panel,
    muted_label,
    scrollable_page,
    section_label,
)
from labeling_tool.monitor.pages.detail import DetailPage
from labeling_tool.monitor.pages.events import EventsPage
from labeling_tool.monitor.pages.overview import OverviewPage
from labeling_tool.monitor.pages.results import ResultsPage
from labeling_tool.qgis_support.dialog_geometry import fit_dialog_to_screen
from labeling_tool.qgis_support.qt6_api import (
    ALIGN_VCENTER,
    WINDOW,
)
from labeling_tool.shared.contracts.monitor_contract import (
    SPAN_STATUS_LABELS,
    effective_device_text,
    execution_trigger_label,
)

ASSEMBLY_PROGRESS_SCALE = 1000

RUN_STATUS_LABELS = {
    "preflight": "预检",
    "planned": "已计划",
    "running": "运行中",
    "raster_ready": "栅格就绪",
    "ready": "已完成",
    "failed": "失败",
    "stopped": "已停止",
    "resetting": "正在重置失败包",
}

MONITOR_THEME_SETTING = "plugins/labeling_tool/inference_monitor_theme"


class InferenceMonitorDialog(QDialog):
    stop_requested = pyqtSignal()
    shutdown_finished = pyqtSignal()
    request_main_run_handling = pyqtSignal(object)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        query_client: MonitorQueryClient | None = None,
    ) -> None:
        """Create the view and take ownership of its optional query client."""
        super().__init__(parent)
        self.setWindowTitle("推理监控")
        self.setWindowFlags(WINDOW)
        self.resize(1680, 1040)
        configured_theme = str(
            QgsSettings().value(MONITOR_THEME_SETTING, "dark") or "dark"
        ).lower()
        self._theme = configured_theme if configured_theme in MONITOR_STYLE else "dark"
        self._connected = []
        self._observations = MonitorObservations()
        self._log_observations = MonitorLogObservations()
        self._monitor_started_at = time.monotonic()
        self._stage_key = ""
        self._stage_started_at = time.monotonic()
        self._last_detail_filter = None
        self._last_detail_requested_at = 0.0
        self._last_snapshot_at = None
        self._last_snapshot_error = ""
        self._latest_execution = {}
        self._main_run_handling_payload: dict[str, object] | None = None
        self._selection = {
            "stream_id": "",
            "object_id": "",
            "object_stream_id": "",
            "job_id": None,
            "object_span_id": "",
            "execution_id": "",
            "kind": "",
            "attempt_id": "",
        }
        self._control_state = ""
        self._icon_labels = []
        self._icon_buttons = []
        self._shutting_down = False
        self._query_client = (
            query_client if query_client is not None else MonitorQueryClient(self)
        )
        self._query_client.setParent(self)
        self._build_ui()
        fit_dialog_to_screen(self, preferred_size=(1680, 1040))
        self._sync_timer = QTimer(self)
        self._sync_timer.setInterval(1000)
        self._sync_timer.timeout.connect(self._refresh_sync_status)
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(1000)
        self._poll_timer.timeout.connect(self._poll_database)
        self._connect_page_signals()
        self._query_client.result_ready.connect(self._on_query_result)
        self._query_client.query_failed.connect(self._on_query_failed)
        self._query_client.shutdown_finished.connect(self._on_query_shutdown_finished)
        self._shutdown_emitted = False

    def _icon(self, name, *, size=32, tone="text"):
        label = QLabel()
        label.setFixedSize(size, size)
        self._icon_labels.append((label, name, size, tone))
        return label

    def _icon_button(self, button, name, *, tone="muted", size=20):
        self._icon_buttons.append((button, name, size, tone))
        button.setIconSize(QSize(size, size))
        return button

    def _build_ui(self):
        self.setObjectName("InferenceMonitor")
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 12, 24, 12)
        root.setSpacing(10)
        title_row = QHBoxLayout()
        title_row.setSpacing(14)
        title_row.addWidget(self._icon("contour", size=36, tone="accent"))
        title = QLabel("LOESS / 推理监控")
        title.setProperty("hero", True)
        title_row.addWidget(title, stretch=1)
        self._run_info_button = QPushButton("运行信息")
        self._run_info_button.setProperty("quiet", True)
        self._run_info_button.clicked.connect(self._show_run_information)
        title_row.addWidget(self._run_info_button)
        self._theme_toggle = QPushButton()
        self._theme_toggle.setObjectName("ThemeToggle")
        self._theme_toggle.setIconSize(QSize(22, 22))
        self._theme_toggle.setFixedSize(40, 40)
        self._theme_toggle.setAutoDefault(False)
        self._theme_toggle.setProperty("quiet", True)
        self._theme_toggle.clicked.connect(self._toggle_theme)
        title_row.addWidget(self._theme_toggle)
        root.addLayout(title_row)
        root.addWidget(divider())

        state_row = QHBoxLayout()
        state_row.setSpacing(20)
        self._status_badge = QLabel("准备中")
        self._status_badge.setProperty("status", "neutral")
        state_row.addWidget(self._status_badge, alignment=ALIGN_VCENTER)
        header_text = QVBoxLayout()
        header_text.setSpacing(6)
        self._phase = QLabel("准备创建运行")
        self._phase.setProperty("headline", True)
        self._phase.setWordWrap(True)
        self._run_id_label = muted_label("任务：准备创建")
        header_text.addWidget(self._phase)
        header_text.addWidget(self._run_id_label)
        self._monitor_sync = QLabel("○ 等待同步")
        self._monitor_sync.setWordWrap(True)
        header_text.addWidget(self._monitor_sync)
        state_row.addLayout(header_text, stretch=1)
        self._stop = QPushButton("停止任务")
        self._stop.setObjectName("StopButton")
        self._icon_button(self._stop, "stop", tone="failed", size=24)
        self._stop.clicked.connect(self._request_stop)
        state_row.addWidget(self._stop)
        root.addLayout(state_row)
        self._run_information = "Run：准备中"
        self._assembly_information = "结果流组装：等待上游计算"
        self._coverage_information = "空白/重叠验收：等待组装"

        self._body_scroll = QScrollArea()
        self._body_scroll.setObjectName("MonitorScroll")
        self._body_scroll.setWidgetResizable(True)
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(10)
        body_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self._body_scroll.setWidget(body)
        root.addWidget(self._body_scroll, stretch=1)

        self._pages = QTabWidget()
        self._pages.setDocumentMode(True)
        self._pages.tabBar().setDrawBase(False)
        self.overview_page = OverviewPage()
        self.detail_page = DetailPage()
        self.results_page = ResultsPage()
        self.events_page = EventsPage()
        self._pages.addTab(self.overview_page, "总览")
        self._pages.addTab(
            scrollable_page(self.detail_page, minimum_height=470), "详细进度"
        )
        self._pages.addTab(
            scrollable_page(self.results_page, minimum_height=510), "结果与验收"
        )
        self._pages.addTab(
            scrollable_page(self.events_page, minimum_height=450), "事件与日志"
        )
        body_layout.addWidget(self._pages, stretch=1)

        progress_panel, progress_layout = monitor_panel()
        progress_layout.setContentsMargins(16, 10, 16, 12)
        progress_layout.setSpacing(8)
        progress_header = QHBoxLayout()
        self._progress_title = section_label("启动准备")
        progress_header.addWidget(self._progress_title)
        self._completion_value = QLabel("—")
        self._completion_value.setProperty("value", True)
        self._completion_value.setProperty("accent", True)
        progress_header.addWidget(self._completion_value)
        progress_header.addStretch()
        self._reduce_motion = QCheckBox("减少动态效果")
        self._reduce_motion.setChecked(
            bool(
                QgsSettings().value(
                    "labeling_tool/monitor_reduce_motion", False, type=bool
                )
            )
        )
        progress_header.addWidget(self._reduce_motion)
        progress_layout.addLayout(progress_header)
        self._overall_bar = OverallProgressTrack()
        self._overall_bar.setRange(0, 0)
        self._overall_bar.setValue(0)
        self._overall_bar.setFormat("本次推理任务完成度：等待任务图")
        self._overall_bar.setToolTip("按任务组统计，不代表剩余时间。")
        self._overall_bar.set_reduced_motion(self._reduce_motion.isChecked())
        self._reduce_motion.toggled.connect(self._set_reduced_motion)
        self._overall_bar.valueChanged.connect(self._refresh_completion_label)
        progress_layout.addWidget(self._overall_bar)
        self._progress_hint = muted_label("等待准备进度；不代表整个任务完成度")
        self._progress_hint.setWordWrap(True)
        progress_layout.addWidget(self._progress_hint)
        body_layout.addWidget(progress_panel)
        self._progress_panel = progress_panel

        status_row = QHBoxLayout()
        status_row.addStretch()
        self._history_completeness = muted_label("历史：等待正式 Run")
        status_row.addWidget(self._history_completeness)
        body_layout.addLayout(status_row)
        self._apply_theme(self._theme, persist=False)
        self.overview_page.apply_width(self.width())

    def _connect_page_signals(self):
        self.overview_page.detail_kind_requested.connect(self._open_detail_kind)
        self.overview_page.stream_selected.connect(self._select_stream)
        self.overview_page.results_requested.connect(
            lambda: self._pages.setCurrentIndex(2)
        )
        self.overview_page.events_requested.connect(
            lambda: self._pages.setCurrentIndex(3)
        )
        self.overview_page.severity_requested.connect(self._show_log_severity)
        self.overview_page.main_run_requested.connect(self._request_main_run_handling)
        self.results_page.stream_selected.connect(self._select_stream)
        self.detail_page.detail_query_requested.connect(
            lambda: self._request_current_detail(force=True)
        )
        self.detail_page.object_selected.connect(self._on_object_selected)
        self.detail_page.object_history_requested.connect(
            lambda append: self._request_object_history(append=append)
        )
        self.detail_page.attempt_selected.connect(self._on_attempt_selected)
        self.detail_page.related_events_requested.connect(self._open_object_events)
        self.events_page.history_query_requested.connect(self._request_history_page)
        self.events_page.raw_continuation_requested.connect(self._continue_raw_log_scan)
        self.events_page.severity_requested.connect(self._on_log_severity_selected)
        self.events_page.log_counts_changed.connect(self._on_log_counts_changed)
        self._pages.currentChanged.connect(self._on_page_changed)

    def _set_reduced_motion(self, reduced):
        self._overall_bar.set_reduced_motion(reduced)
        QgsSettings().setValue("labeling_tool/monitor_reduce_motion", bool(reduced))

    def _refresh_sync_status(self):
        colors = PALETTES[self._theme]
        age = (
            max(0, int(time.time() - self._last_snapshot_at))
            if self._last_snapshot_at
            else None
        )
        if self._last_snapshot_error:
            text, tone = "⚠ 同步失败 · 显示上次数据", "warning"
        elif age is None and self._progress_title.text().startswith("启动准备"):
            text, tone = "○ 启动准备 · 正式运行尚未建立", "muted"
        elif age is None:
            text, tone = "○ 等待同步", "muted"
        elif age > 10:
            text, tone = f"⚠ 数据暂未更新 · {age}秒前同步", "warning"
        else:
            text, tone = f"● 数据同步正常 · {age}秒前更新", "success"
        self._monitor_sync.setText(text)
        self._monitor_sync.setStyleSheet(f"color: {colors[tone]}; font-weight: 600;")
        if tone != "success" and not self._progress_title.text().startswith("启动准备"):
            self._overall_bar.set_running(False)

    def _refresh_completion_label(self, *_args):
        total = self._overall_bar.maximum()
        self._completion_value.setText(
            f"{max(0, self._overall_bar.value()) / total:.0%}" if total > 0 else "—"
        )
        self._overall_bar.setToolTip(self._overall_bar.format())

    def _apply_theme(self, theme, *, persist=True):
        selected = str(theme or "dark").lower()
        if selected not in MONITOR_STYLE:
            selected = "dark"
        self._theme = selected
        self._refresh_sync_status()
        families = set(QFontDatabase.families())
        body_family = next(
            (
                name
                for name in ("PingFang SC", "Noto Sans CJK SC", "Source Han Sans SC")
                if name in families
            ),
            self.font().family(),
        )
        self.setFont(QFont(body_family, BODY_FONT_PT))
        self.setStyleSheet(MONITOR_STYLE[selected])
        palette = PALETTES[selected]
        for label, name, size, tone in self._icon_labels:
            label.setPixmap(
                monitor_icon(name, palette[tone], size).pixmap(QSize(size, size))
            )
        for button, name, size, tone in self._icon_buttons:
            button.setIcon(monitor_icon(name, palette[tone], size))
        self.overview_page.apply_theme(selected)
        self.detail_page.apply_theme(selected)
        self.results_page.apply_theme(selected)
        self.events_page.apply_theme(selected)
        for stream_id in self._observations.stream_ids():
            self._publish_stream(stream_id)
        theme_action = "切换浅色主题" if selected == "dark" else "切换深蓝主题"
        self._theme_toggle.setIcon(
            monitor_icon("sun" if selected == "dark" else "moon", palette["muted"], 22)
        )
        self._theme_toggle.setToolTip(theme_action)
        self._theme_toggle.setAccessibleName(theme_action)
        if persist:
            QgsSettings().setValue(MONITOR_THEME_SETTING, selected)

    def _show_run_information(self):
        box = self._build_run_information_dialog()
        box.exec()
        box.deleteLater()

    def _build_run_information_dialog(self):
        """Use the monitor theme and readable dimensions for run metadata."""
        box = QDialog(self)
        box.setObjectName("InferenceMonitor")
        box.setWindowTitle("LOESS / 运行信息")
        box.setStyleSheet(MONITOR_STYLE[self._theme])
        layout = QVBoxLayout(box)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)
        layout.addWidget(section_label("运行信息"))
        details = MonitorTextBrowser(box)
        details.setPlainText(
            "\n\n".join(
                text.replace(" | ", "\n")
                for text in (
                    self._run_information,
                    self._assembly_information,
                    self._coverage_information,
                )
            )
        )
        layout.addWidget(details, stretch=1)
        hint = muted_label(
            "创建年龄与本次观察时长不是实际执行耗时。恢复与重做失败包仍位于主界面。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = QHBoxLayout()
        buttons.addStretch()
        close = QPushButton("关闭")
        close.clicked.connect(box.accept)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        screen = self.screen()
        available = screen.availableGeometry() if screen else self.geometry()
        width = max(1, available.width() - 40)
        height = max(1, available.height() - 60)
        box.setMinimumSize(min(560, width), min(400, height))
        box.resize(min(880, width), min(640, height))
        return box

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "overview_page"):
            self.overview_page.apply_width(event.size().width())

    def showEvent(self, event):
        super().showEvent(event)
        if self._shutting_down:
            return
        current = self._pages.currentIndex()
        self.detail_page.set_active(current == 1)
        self.events_page.set_active(current == 3)
        if hasattr(self, "_sync_timer"):
            self._sync_timer.start()
            self._refresh_sync_status()
        timer = getattr(self, "_poll_timer", None)
        if timer is not None:
            timer.setInterval(1000)
            if self._query_client.is_bound and not timer.isActive():
                timer.start()
                self._poll_database()
        self._refresh_current_page()

    def hideEvent(self, event):
        super().hideEvent(event)
        self.detail_page.set_active(False)
        self.events_page.set_active(False)
        if hasattr(self, "_sync_timer"):
            self._sync_timer.stop()
        timer = getattr(self, "_poll_timer", None)
        if timer is not None and timer.isActive():
            timer.setInterval(5000)

    def _set_status_badge(self, status):
        self._status_badge.setProperty("status", str(status))
        self._status_badge.style().unpolish(self._status_badge)
        self._status_badge.style().polish(self._status_badge)

    def _toggle_theme(self):
        self._apply_theme("light" if self._theme == "dark" else "dark")

    def _on_page_changed(self, index):
        self.detail_page.set_active(False)
        self.events_page.set_active(False)
        if self._shutting_down:
            return
        self.detail_page.set_active(int(index) == 1 and self.isVisible())
        self.events_page.set_active(int(index) == 3 and self.isVisible())
        self._refresh_current_page()

    def _refresh_current_page(self):
        index = self._pages.currentIndex()
        if index == 1:
            self._render_selected_tiles()
        elif index == 2:
            self._render_selected_assembly()
        elif index == 3 and self._query_client.is_bound:
            self._request_history_page(False)

    def _open_detail_kind(self, kind):
        self._pages.setCurrentIndex(1)
        self.detail_page.set_kind(str(kind))

    def _on_object_selected(self, value):
        raw = dict(value or {})
        object_id = str(
            raw.get("object_id") or raw.get("tile_id") or raw.get("unit_id") or ""
        )
        object_stream_id = str(raw.get("stream_id") or "")
        job_id = raw.get("job_id")
        identity = (object_id, object_stream_id, job_id)
        previous = (
            self._selection.get("object_id"),
            self._selection.get("object_stream_id"),
            self._selection.get("job_id"),
        )
        if identity != previous:
            self._selection["attempt_id"] = ""
            self.detail_page.reset_object_history()
        self._selection.update(
            {
                "object_id": object_id,
                "object_stream_id": object_stream_id,
                "job_id": job_id,
                "object_span_id": str(raw.get("span_id") or ""),
                "execution_id": str(raw.get("execution_id") or ""),
                "kind": self.detail_page.current_kind(),
            }
        )
        self.events_page.invalidate_raw_continuation()
        self._request_object_history(append=False)

    def _on_attempt_selected(self, attempt_id):
        self._selection["attempt_id"] = str(attempt_id)
        self.events_page.invalidate_raw_continuation()
        self._request_object_history(append=False)
        self.detail_page.select_attempt_view(str(attempt_id))

    def _request_object_history(self, *, append=False):
        object_id = str(self._selection.get("object_id") or "")
        if (
            self._shutting_down
            or not self._query_client.is_bound
            or not self._query_client.run_id
            or not object_id
        ):
            return
        current = self.detail_page.current_object_filter(self._selection)
        cursor = self.detail_page.object_history_cursor()
        self._query_client.queue_object_history(
            object_id=str(current["object_id"]),
            detail_kind=str(current["detail_kind"]),
            stream_id=str(current["stream_id"]),
            job_id=current["job_id"],
            span_id=str(current["span_id"]),
            append=bool(append),
            before_started_at=cursor[0] if append and cursor else "",
            before_span_id=cursor[1] if append and cursor else "",
        )

    def _object_request_matches_controls(self, request):
        return query_matches_filter(
            request, self.detail_page.current_object_filter(self._selection)
        )

    def _open_object_events(self):
        self.events_page.set_target(
            "attempt" if self._selection.get("attempt_id") else "object"
        )
        self._pages.setCurrentIndex(3)

    def _request_history_page(self, append=False):
        if (
            self._shutting_down
            or not self._query_client.is_bound
            or not self._query_client.run_id
        ):
            return
        fields = self.events_page.history_request_fields(
            self._selection, append=bool(append)
        )
        self._query_client.queue_history(**fields)

    def _continue_raw_log_scan(self, request_id, generation, cursor):
        if (
            not self._shutting_down
            and self.isVisible()
            and self._pages.currentIndex() == 3
            and self._query_client.continuation_is_current(
                "history", generation=generation, request_id=request_id
            )
        ):
            self._request_history_page(True)

    def _apply_monitor_history(self, history):
        value = dict(history or {})
        self._latest_execution = dict(value.get("latest_execution") or {})
        if value.get("archived"):
            summary = dict(value.get("summary") or {})
            health_text = (
                "详细历史已归档，仅保留摘要："
                f"执行 {int(summary.get('execution_count') or 0)} 次，"
                f"尝试 {int(summary.get('attempt_count') or 0)} 次，"
                f"失败 {int(summary.get('failed_count') or 0)} 次"
            )
            self._history_completeness.setText("历史：已归档，仅保留摘要")
        elif not value.get("available"):
            reason = str(value.get("reason") or "")
            health_text = (
                "升级前 Run：关键执行历史不完整"
                if reason == "upgrade_precedes_history"
                else "历史记录读取失败或尚未建立"
            )
            self._history_completeness.setText("历史：记录不完整")
        else:
            latest = dict(value.get("latest_execution") or {})
            counts = dict(value.get("span_status_counts") or {})
            health_text = (
                f"本次{execution_trigger_label(latest.get('trigger_type') or '')}执行 · "
                f"{SPAN_STATUS_LABELS.get(str(latest.get('status') or ''), '未知')}\n"
                f"历史记录：完成 {int(counts.get('completed') or 0)} · "
                f"失败 {int(counts.get('failed') or 0)} · "
                f"中断 {int(counts.get('interrupted') or 0)}"
            )
            complete = bool(latest.get("recording_complete", True))
            self._history_completeness.setText(
                "历史：关键过程已记录" if complete else "历史：记录不完整"
            )
        events = [dict(event) for event in value.get("recent_events") or ()][:5]
        self.overview_page.render_history(health_text, events)
        self.events_page.render_executions(
            [dict(item) for item in value.get("executions") or ()]
        )

    def attach_runner(self, runner: QObject):
        self.detach()
        pairs = [
            (runner.step_started, self._on_step_started),
            (runner.step_finished, self._on_step_finished),
            (runner.pipeline_finished, self._on_finished),
        ]
        log_batch = getattr(runner, "log_batch", None)
        if log_batch is not None:
            pairs.insert(0, (log_batch, self._on_log_batch))
        else:
            pairs.insert(0, (runner.log_line, self._on_log))
            process_log = getattr(runner, "process_log", None)
            if process_log is not None:
                pairs.insert(0, (process_log, self._on_process_log))
        progress_batch = getattr(runner, "stream_progress_batch", None)
        if progress_batch is not None:
            pairs.append((progress_batch, self._on_stream_progress_batch))
        else:
            pairs.append((runner.stream_progress, self._on_stream_progress))
        for signal, slot in pairs:
            signal.connect(slot)
            self._connected.append((signal, slot))
        self._stop.setEnabled(True)

    def bind_state_database(
        self, database_path, run_id, *, page_size=500, run_spec=None
    ):
        del database_path
        if not run_spec:
            raise ValueError("monitor requires the frozen PostgreSQL Run Spec")
        self._observations.begin_binding()
        self._clear_main_run_handling()
        self.events_page.invalidate_raw_continuation()
        self._query_client.bind(str(run_id), run_spec)
        self.detail_page.reset(page_size=max(1, min(int(page_size), 500)))
        warnings, errors = self.events_page.log_counts()
        self.overview_page.render_log_actions(
            bound=True, warnings=warnings, errors=errors
        )
        self._last_detail_filter = None
        self._last_detail_requested_at = 0.0
        self._monitor_started_at = time.monotonic()
        self._stage_key = ""
        self._stage_started_at = time.monotonic()
        self._poll_timer.start()
        self._progress_title.setText("本次推理完成度")
        self._progress_hint.setText("按任务组统计，不代表剩余时间")
        self._overall_bar.set_running(False)
        self._overall_bar.setRange(0, 0)
        self._overall_bar.setFormat("等待正式运行进度")
        self._refresh_completion_label()
        self._poll_database()

    def unbind_state_database(self):
        if hasattr(self, "_poll_timer"):
            self._poll_timer.stop()
        self.events_page.invalidate_raw_continuation()
        self._query_client.unbind()
        self._clear_main_run_handling()
        self._observations.end_binding()
        self._last_detail_filter = None
        warnings, errors = self.events_page.log_counts()
        self.overview_page.render_log_actions(
            bound=False, warnings=warnings, errors=errors
        )

    def detach(self):
        for signal, slot in self._connected:
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._connected.clear()

    def clear_log(self):
        self.events_page.clear_log()
        self._log_observations.clear_logged_errors()

    def _set_log_visible(self, visible):
        shown = bool(visible)
        if shown:
            self._pages.setCurrentIndex(3)
        self.events_page.set_log_visible(shown)

    def _show_log_severity(self, severity):
        self._pages.setCurrentIndex(3)
        self.events_page.show_log_severity(
            str(severity),
            database_bound=bool(
                self._query_client.is_bound and self._query_client.run_id
            ),
        )

    def _on_log_severity_selected(self, severity):
        if (
            severity in {"warning", "error"}
            and self._query_client.is_bound
            and self._query_client.run_id
        ):
            self._show_log_severity(severity)

    def _on_log_counts_changed(self, warnings, errors):
        bound = bool(self._query_client.is_bound and self._query_client.run_id)
        self.overview_page.render_log_actions(
            bound=bound, warnings=int(warnings), errors=int(errors)
        )

    def _update_coverage_information(self):
        values = [
            dict(value)
            for value in self._observations.coverage_view().values()
            if isinstance(value, dict)
        ]
        if not values:
            self._coverage_information = "空白/重叠验收：等待组装"
            return
        gap_area_m2 = sum(float(value.get("gap_area_m2") or 0.0) for value in values)
        overlap_area_m2 = sum(
            float(value.get("overlap_area_m2") or 0.0) for value in values
        )
        outside_area_m2 = sum(
            float(value.get("outside_area_m2") or 0.0) for value in values
        )
        passed = sum(1 for value in values if value.get("status") == "passed")
        failed = sum(1 for value in values if value.get("status") == "failed")
        skipped = len(values) - passed - failed
        if failed:
            state = f"失败 {failed} 个流"
        elif skipped:
            state = f"通过 {passed}，未验证 {skipped}"
        else:
            state = "通过"
        self._coverage_information = (
            f"空白/重叠验收：{state} {passed}/{len(values)} | "
            f"空白 {gap_area_m2:.6g} m² | "
            f"重叠 {overlap_area_m2:.6g} m² | "
            f"范围外 {outside_area_m2:.6g} m²"
        )

    def reset_run(self, tiles=None):
        del tiles
        self.unbind_state_database()
        self.clear_log()
        self.overview_page.reset()
        self.detail_page.reset()
        self.results_page.reset()
        self.events_page.reset()
        self._observations.reset()
        self._log_observations.reset()
        self._control_state = ""
        self._last_detail_filter = None
        self._selection.update(
            {
                "stream_id": "",
                "object_id": "",
                "object_stream_id": "",
                "job_id": None,
                "object_span_id": "",
                "execution_id": "",
                "kind": "",
                "attempt_id": "",
            }
        )
        self._stage_key = ""
        self._stage_started_at = time.monotonic()
        self._phase.setText("准备中")
        self._status_badge.setText("准备中")
        self._set_status_badge("neutral")
        self._run_id_label.setText("Run：准备创建")
        self._monitor_sync.setText("○ 等待同步")
        self._overall_bar.set_running(False)
        self._run_information = "Run：准备中"
        self._assembly_information = "结果流组装：等待上游计算"
        self._coverage_information = "空白/重叠验收：等待组装"
        self._overall_bar.setRange(0, ASSEMBLY_PROGRESS_SCALE)
        self._overall_bar.setValue(0)
        self._overall_bar.setFormat("整体任务完成度：等待任务图")
        self._progress_title.setText("启动准备")
        self._progress_hint.setText("等待准备进度；不代表整个任务完成度")
        self._last_snapshot_at = None
        self._last_snapshot_error = ""
        self._overall_bar.setRange(0, 0)
        self._overall_bar.setFormat("等待准备进度")
        self._refresh_completion_label()
        self._refresh_sync_status()
        self._stop.setEnabled(True)
        self._stop.setText("停止任务")
        self.setWindowTitle("推理监控 - 准备中")

    def set_stage_progress(self, info):
        if self._control_state or self._observations.terminal_status:
            return
        name = str(info.get("name") or "处理中")
        stream_id = str(info.get("stream_id") or "")
        current = int(info.get("current") or 0)
        total = int(info.get("total") or 0)
        message = str(info.get("message") or "")
        if self._query_client.is_bound and self._query_client.run_id:
            # V5 的 runner 总数把 Work Package 和 unit_fit 两种成本完全不同的
            # Job 相加。数据库绑定后由左侧分层概览分别显示，不能再把这个
            # 混合总数作为用户进度条。
            return
        text = f"{name} | {stream_id}" if stream_id else name
        if message:
            text += f" | {message}"
        self._phase.setText(text)
        self._progress_title.setText(f"启动准备 · {name}")
        self._progress_hint.setText("仅表示当前准备步骤，不代表整个任务完成度")
        self._overall_bar.set_running(False)
        self._overall_bar.setRange(0, max(0, total))
        self._overall_bar.setValue(max(0, min(current, total)) if total > 0 else 0)
        self._overall_bar.setFormat(
            f"{name}：{current}/{total}"
            if total > 0
            else f"{name}：处理中，总量尚未确定"
        )
        self._overall_bar.set_running(True)
        self._refresh_completion_label()
        if stream_id:
            change = self._observations.observe_preparation_progress(
                stream_id=stream_id,
                name=name,
                current=current,
                total=total,
            )
            self._refresh_streams(change.refresh_stream_ids)

    def mark_stopping(self, text="正在停止当前子进程组"):
        self._control_state = "stopping"
        self._overall_bar.set_running(False)
        self._stop.setEnabled(False)
        self._stop.setText("正在停止…")
        self._phase.setText(str(text))
        self._status_badge.setText("正在停止")
        self._set_status_badge("warning")

    def mark_finished(self, text="已完成", detail=""):
        self._overall_bar.set_running(False)
        self._control_state = ""
        self._observations.mark_terminal(
            {
                "已完成": "ready",
                "失败": "failed",
                "已停止": "stopped",
            }.get(text, "")
        )
        # Fence callbacks dispatched before this authoritative runner result.
        self._query_client.fence()
        if self._query_client.is_bound:
            self._query_client.queue_snapshot()
        self._stop.setEnabled(False)
        self._stop.setText("停止任务")
        detail_text = re.sub(r"\s+", " ", str(detail or "")).strip()
        self._phase.setText(f"{text}：{detail_text}" if detail_text else text)
        self._phase.setToolTip(str(detail or ""))
        self._status_badge.setText(str(text))
        self._set_status_badge(
            "active" if text == "已完成" else "failed" if text == "失败" else "neutral"
        )
        if text == "已完成":
            self._overall_bar.setRange(0, ASSEMBLY_PROGRESS_SCALE)
            self._overall_bar.setValue(ASSEMBLY_PROGRESS_SCALE)
            self._overall_bar.setFormat("整体任务完成度：100%（已完成）")
        self.setWindowTitle(f"推理监控 - {text}")

    def _refresh_streams(self, stream_ids):
        for stream_id in stream_ids:
            stream_id = str(stream_id)
            self._publish_stream(stream_id)
            if not self._selection.get("stream_id"):
                self._select_stream(stream_id)

    def _publish_stream(self, stream_id):
        stream_id = str(stream_id)
        view = self._observations.stream_view(stream_id)
        state = view["state"]
        if not state:
            return
        display_name = self._stream_display_name(stream_id)
        self.overview_page.upsert_stream(
            stream_id,
            display_name,
            state,
            view["runtime"],
            view["coverage"],
        )
        self.results_page.upsert_stream(stream_id, display_name, state)
        if stream_id == self._selected_stream():
            self._render_selected_assembly()

    def _select_stream(self, stream_id):
        stream_id = str(stream_id)
        if not stream_id or not self._observations.has_stream(stream_id):
            return
        changed = stream_id != str(self._selection.get("stream_id") or "")
        self._selection["stream_id"] = stream_id
        if changed:
            self._selection.update(
                {
                    "object_id": "",
                    "object_stream_id": "",
                    "job_id": None,
                    "object_span_id": "",
                    "execution_id": "",
                    "kind": "",
                    "attempt_id": "",
                }
            )
            self.detail_page.reset_object_history()
            self.events_page.invalidate_raw_continuation()
        self.overview_page.select_stream(stream_id)
        self.results_page.select_stream(stream_id)
        database_bound = bool(self._query_client.is_bound and self._query_client.run_id)
        self.detail_page.set_stream_context(
            stream_id,
            self._stream_display_name(stream_id),
            {} if database_bound else self._observations.tiles_snapshot(stream_id),
            database_bound,
        )
        self._render_selected_assembly()
        if self._pages.currentIndex() == 1 and self.isVisible():
            self._request_current_detail(force=changed)

    def _on_process_log(self, event):
        presented = self._log_observations.observe_process_log(
            dict(event or {}), attempt_for=self._observations.attempt_for
        )
        self._append_presented_log(presented)

    def _on_log_batch(self, events):
        self.events_page.begin_log_batch()
        try:
            for event in events or ():
                info = dict(event or {})
                self._on_log(
                    str(info.get("source") or "system"),
                    str(info.get("message") or ""),
                    log_context=info,
                )
        finally:
            self.events_page.end_log_batch()

    def _on_log(self, level, message, log_context=None):
        presented = self._log_observations.observe_log(
            level,
            message,
            context=None if log_context is None else dict(log_context),
            attempt_for=self._observations.attempt_for,
        )
        if presented is not None:
            self._append_presented_log(presented)

    def _append_presented_log(self, presented: PresentedLog):
        self.events_page.append_log_event(
            presented["text"],
            source=presented["source"],
            severity=presented["severity"],
            title=presented["title"],
            affected=presented["affected"],
            system_action=presented["system_action"],
            user_action=presented["user_action"],
            fingerprint=presented["fingerprint"],
            context_key=presented["context_key"],
            event_timestamp=presented["event_timestamp"],
        )

    def _on_step_started(self, name):
        change = self._observations.observe_step_started(
            str(name), epoch_now=time.time()
        )
        self._refresh_streams(change.refresh_stream_ids)

    def _on_step_finished(self, name, return_code, result):
        change = self._observations.observe_step_finished(
            str(name),
            int(return_code),
            dict(result or {}),
            epoch_now=time.time(),
            database_bound=bool(self._query_client.is_bound),
        )
        for pending in change.pending_logs:
            self._on_log(
                pending.level,
                pending.message,
                log_context=pending.context,
            )
        self._refresh_streams(change.refresh_stream_ids)

    def _on_stream_progress_batch(self, events):
        for info in events or ():
            self._on_stream_progress(dict(info or {}))

    def _on_stream_progress(self, info):
        stream_id = str(info.get("stream_id") or "")
        package_event = str(info.get("event") or "").startswith(
            ("package_", "work_package_", "accelerator_worker_")
        )
        run_spec = self._query_client.run_spec() if package_event else {}
        fusion_profile_id = str((run_spec.get("fusion") or {}).get("profile_id") or "")
        change = self._observations.observe_stream_progress(
            dict(info or {}),
            database_bound=bool(self._query_client.is_bound),
            configured_batch_size=self._configured_batch_for_stream(
                stream_id, run_spec
            ),
            fusion_profile_id=fusion_profile_id,
            epoch_now=time.time(),
            monotonic_now=time.monotonic(),
        )
        self._refresh_streams(change.refresh_stream_ids)
        selected_stream = self._selected_stream()
        for tile in change.tile_updates:
            if tile.stream_id == selected_stream:
                self.detail_page.update_live_tile(tile.tile_id, dict(tile.state))
        if change.coverage_changed:
            self._update_coverage_information()

    def _configured_batch_for_stream(
        self, stream_id: str, run_spec: dict[str, object]
    ) -> int:
        tuning = run_spec.get("resource_tuning") or {}
        resolved = tuning.get("resolved") or {}
        by_model = resolved.get("tile_batch_size_by_model") or {}
        runtime = run_spec.get("runtime") or {}
        model_id = stream_id.split(":", 1)[1] if stream_id.startswith("model:") else ""
        return int(
            by_model.get(model_id)
            or resolved.get("tile_batch_size")
            or runtime.get("tile_batch_size")
            or 0
        )

    def _stream_display_name(self, stream_id: str) -> str:
        for model in self._query_client.run_spec().get("models") or []:
            if stream_id == f"model:{model.get('model_id')}":
                return str(
                    model.get("display_name") or model.get("model_id") or stream_id
                )
        fusion = self._query_client.run_spec().get("fusion") or {}
        if stream_id == f"fusion:{fusion.get('profile_id')}":
            return str(fusion.get("display_name") or "Fusion")
        return stream_id

    def _selected_stream(self):
        return str(self._selection.get("stream_id") or "")

    def _render_selected_tiles(self):
        stream_id = self._selected_stream()
        database_bound = bool(self._query_client.is_bound and self._query_client.run_id)
        self.detail_page.set_stream_context(
            stream_id,
            self._stream_display_name(stream_id) if stream_id else "",
            {} if database_bound else self._observations.tiles_snapshot(stream_id),
            database_bound,
        )
        if database_bound:
            self._request_current_detail()

    def _request_current_detail(self, *, force=False):
        if (
            self._shutting_down
            or not self._query_client.is_bound
            or not self._query_client.run_id
        ):
            return
        stream_id = self._selected_stream()
        fields = self.detail_page.current_detail_filter(stream_id)
        detail_kind = str(fields.get("detail_kind") or "package")
        if not stream_id and detail_kind not in {"package", "tile"}:
            self.detail_page.clear_for_missing_stream()
            return
        signature = tuple(fields.items())
        now = time.monotonic()
        if (
            not force
            and signature == self._last_detail_filter
            and now - self._last_detail_requested_at < 2.0
        ):
            return
        self._last_detail_filter = signature
        self._last_detail_requested_at = now
        self._query_client.queue_detail(**fields)

    def _detail_request_matches_controls(self, request):
        return query_matches_filter(
            request,
            self.detail_page.current_detail_filter(self._selected_stream()),
        )

    def _history_request_matches_controls(self, request):
        return query_matches_filter(
            request,
            self.events_page.history_request_fields(self._selection, append=False),
        )

    def _render_selected_assembly(self):
        stream_id = self._selected_stream()
        view = self._observations.stream_view(stream_id)
        self.results_page.render_assembly(
            stream_id=stream_id,
            display_name=self._stream_display_name(stream_id) if stream_id else "",
            runtime=view["runtime"],
            phase_statuses=view["phase_statuses"],
            coverage=view["coverage"],
        )

    def _on_query_result(self, request: QueryMessage, payload: QueryResult) -> None:
        value = dict(payload or {})
        active = dict(request or {})
        kind = str(value.get("kind") or "")
        current = True
        if kind == "detail":
            current = self._detail_request_matches_controls(active)
        elif kind == "history":
            current = self._history_request_matches_controls(active)
        elif kind == "object_history":
            current = self._object_request_matches_controls(active)
        if not current:
            return
        if kind == "snapshot":
            self._apply_database_snapshot(dict(value.get("snapshot") or {}))
        elif kind == "detail":
            stream_id = str(value.get("stream_id") or "")
            self.detail_page.render_detail(value, self._stream_display_name(stream_id))
        elif kind == "history":
            self.events_page.render_history(value, context=self._selection)
        elif kind == "object_history":
            configured_models = [
                dict(model)
                for model in self._query_client.run_spec().get("models") or ()
            ]
            self.detail_page.render_object_history(value, configured_models)

    def _on_query_failed(self, request: QueryMessage, payload: QueryFailure) -> None:
        value = dict(payload or {})
        active = dict(request or {})
        kind = str(value.get("kind") or "")
        current = True
        if kind == "detail":
            current = self._detail_request_matches_controls(active)
        elif kind == "history":
            current = self._history_request_matches_controls(active)
        elif kind == "object_history":
            current = self._object_request_matches_controls(active)
        if not current:
            return
        error = str(value.get("error") or "unknown monitor query error")
        if kind == "snapshot":
            self._clear_main_run_handling()
        if kind == "history" and str(active.get("scope") or "").startswith("raw_"):
            self.events_page.show_history_error(error, raw=True)
            return
        self._last_snapshot_error = error
        self._monitor_sync.setText("⚠ 同步失败 · 显示上次数据")
        self._overall_bar.set_running(False)
        self._on_log("system", f"[monitor-db] {error}")
        if kind == "detail":
            self._last_detail_filter = None
            self.detail_page.show_query_error("detail", error)
        elif kind == "history":
            self.events_page.show_history_error(error, raw=False)
        elif kind == "object_history":
            self.detail_page.show_query_error("object_history", error)

    def _poll_database(self):
        if not self._query_client.is_bound or not self._query_client.run_id:
            return
        self._query_client.queue_snapshot()

    def _apply_database_snapshot(self, snapshot):
        try:
            observed_at = time.time()
            change = self._observations.apply_snapshot(
                dict(snapshot or {}), epoch_now=observed_at
            )
            if change is None:
                return
            self._last_snapshot_at = observed_at
            self._last_snapshot_error = ""
            self._monitor_sync.setText("● 数据同步正常 · 刚刚更新")
            run_row = snapshot.get("run") or {}
            run_status = str(run_row.get("status") or "planned")
            job_counts = snapshot.get("job_counts") or {}
            job_progress = snapshot.get("job_progress") or {}
            package_counts = job_counts.get("work_package") or {}
            unit_job_counts = job_counts.get("unit_fit") or {}
            active_package = snapshot.get("active_work_package")
            streams = snapshot.get("streams") or []
            all_runtime_progress = snapshot.get("stream_runtime_progress") or {}
            self._refresh_streams(change.refresh_stream_ids)
            self._update_coverage_information()

            self._apply_monitor_history(snapshot.get("monitor_history") or {})
            run_spec = self._render_window_snapshot_status(
                run_status=run_status,
                package_counts=package_counts,
                unit_job_counts=unit_job_counts,
                job_counts=job_counts,
                job_progress=job_progress,
                streams=streams,
                stream_runtime_progress=all_runtime_progress,
            )
            self._render_overview_snapshot(
                run_status=run_status,
                run=run_row,
                package_counts=package_counts,
                unit_job_counts=unit_job_counts,
                job_counts=job_counts,
                active_package=active_package,
                streams=streams,
                run_spec=run_spec,
            )
            self._update_assembly_information(
                streams=streams,
                stream_runtime_progress=all_runtime_progress,
                run_spec=run_spec,
            )
            self._render_selected_assembly()
            if self.isVisible():
                self._render_selected_tiles()
        except Exception as error:
            self._clear_main_run_handling()
            self._on_log("system", f"[monitor-db] {error}")

    def _render_window_snapshot_status(
        self,
        *,
        run_status,
        package_counts,
        unit_job_counts,
        job_counts,
        job_progress,
        streams,
        stream_runtime_progress,
    ):
        phase = database_phase(
            run_status,
            package_counts,
            unit_job_counts,
            streams,
            self._observations.active_global_stage,
        )
        stage_key, stage = phase.key, phase.title
        if stage_key != self._stage_key:
            self._stage_key = stage_key
            self._stage_started_at = time.monotonic()
        stage_elapsed = elapsed_text(time.monotonic() - self._stage_started_at)
        if not self._control_state:
            display_stage = {
                "Work Package 推理 + 空间单元拟合": "正在识别地物，同时处理边界",
                "Work Package 推理": "正在识别地物",
                "空间单元拟合": "正在处理空间边界",
                "结果流并行组装": "正在组装模型与融合结果",
                "分区概率栅格收口": "正在汇总推理栅格",
                "整体验收": "正在核验本次推理结果",
            }.get(stage, stage)
            self._phase.setText(display_stage)
            self._phase.setToolTip(f"{stage} · 当前阶段观察 {stage_elapsed}")
        self.setWindowTitle(f"推理监控 - {stage}")
        status_text = RUN_STATUS_LABELS.get(run_status, run_status)
        if run_status in {"ready", "failed", "stopped"}:
            self._stop.setEnabled(False)
        if not self._control_state:
            self._status_badge.setText(status_text)
            self._set_status_badge(
                "active"
                if run_status in {"running", "raster_ready", "ready"}
                else "failed"
                if run_status == "failed"
                else "warning"
                if run_status == "resetting"
                else "neutral"
            )
        self._run_id_label.setText(
            f"Run：{self._query_client.run_id} · 任务：{status_text} · "
            f"执行：{str((self._latest_execution or {}).get('execution_id') or '—')[:12]}"
        )

        overall_fraction, overall_group_count = overall_completion_fraction(
            run_status,
            job_counts,
            job_progress,
            streams,
            stream_runtime_progress,
        )
        overall_value = round(overall_fraction * ASSEMBLY_PROGRESS_SCALE)
        self._progress_title.setText("本次推理完成度")
        self._progress_hint.setText("按任务组统计，不代表剩余时间")
        overall_percent = round(overall_fraction * 100)
        self._overall_bar.setRange(0, ASSEMBLY_PROGRESS_SCALE)
        self._overall_bar.set_running(
            run_status == "running"
            and not self._control_state
            and not self._observations.terminal_status
        )
        self._overall_bar.setValue(overall_value)
        self._overall_bar.setFormat(
            f"本次推理任务完成度：{overall_percent}% | "
            f"按 {overall_group_count} 类任务计算，不代表剩余时间"
        )
        self._refresh_completion_label()

        run_spec = self._query_client.run_spec()
        backend, device_name = effective_device_text(run_spec)
        device = f"{backend} · {device_name}"
        monitor_elapsed = elapsed_text(time.monotonic() - self._monitor_started_at)
        run_created_epoch = self._observations.run_created_epoch
        run_age = (
            elapsed_text(time.time() - run_created_epoch)
            if run_created_epoch is not None
            else "—"
        )
        self._run_information = (
            f"Run：{self._query_client.run_id} | 状态："
            f"{RUN_STATUS_LABELS.get(run_status, run_status)} | "
            f"设备：{device} | 创建至今：{run_age} | "
            f"本次监控：{monitor_elapsed}"
        )
        return run_spec

    def _render_overview_snapshot(
        self,
        *,
        run_status,
        run,
        package_counts,
        unit_job_counts,
        job_counts,
        active_package,
        streams,
        run_spec,
    ):
        stream_names = {
            str(stream.get("stream_id") or ""): self._stream_display_name(
                str(stream.get("stream_id") or "")
            )
            for stream in streams
        }
        self.overview_page.render_execution(
            run_status=run_status,
            package_counts=package_counts,
            unit_job_counts=unit_job_counts,
            job_counts=job_counts,
            active_package=active_package,
            package_activity=self._observations.package_view(),
            run_spec=run_spec,
            stream_names=stream_names,
        )
        package_failed = int(package_counts.get("failed", 0))
        unit_failed = int(unit_job_counts.get("failed", 0))
        feedback = failure_feedback(
            run_status=str(run_status),
            run=dict(run or {}),
            package_counts=package_counts,
            unit_job_counts=unit_job_counts,
            streams=[dict(stream) for stream in streams],
            latest_execution=dict(self._latest_execution),
        )
        payload = self._main_run_handling_payload_for(
            run_status=str(run_status), run_spec=run_spec
        )
        self._main_run_handling_payload = payload
        self.overview_page.render_health(
            run_status=run_status,
            package_failed=package_failed,
            unit_failed=unit_failed,
            feedback=feedback,
            can_open_main_run=payload is not None,
        )

    def _main_run_handling_payload_for(self, *, run_status, run_spec):
        run_id = str(self._query_client.run_id or "")
        frozen_spec = dict(run_spec or {})
        if (
            run_status not in {"failed", "stopped"}
            or not self._query_client.is_bound
            or not run_id
            or frozen_spec.get("schema_version") != 2
            or str(frozen_spec.get("run_id") or "") != run_id
        ):
            return None
        return {
            "run_id": run_id,
            "run_spec": copy.deepcopy(frozen_spec),
            "observed_status": str(run_status),
        }

    def _clear_main_run_handling(self):
        self._main_run_handling_payload = None
        if hasattr(self, "overview_page"):
            self.overview_page.clear_main_run_action()

    def _request_main_run_handling(self):
        payload = self._main_run_handling_payload
        if payload is not None:
            self.request_main_run_handling.emit(copy.deepcopy(payload))

    def _update_assembly_information(
        self, *, streams, stream_runtime_progress, run_spec
    ):
        stream_ready = sum(
            1 for stream in streams if str(stream.get("status")) == "ready"
        )
        assembly_running = sum(
            1 for stream in streams if str(stream.get("status")) == "assembling"
        )
        assembly_failed = sum(
            1
            for stream in streams
            if str(
                (stream_runtime_progress.get(str(stream["stream_id"])) or {}).get(
                    "status"
                )
                or ""
            )
            == "failed"
        )
        assembly_waiting = max(
            0,
            len(streams) - stream_ready - assembly_running - assembly_failed,
        )
        assembly_limit = int(
            (run_spec.get("scaling") or {}).get("max_concurrent_assembly", 2) or 2
        )
        active_phases = []
        for stream in streams:
            stream_id = str(stream["stream_id"])
            info = stream_runtime_progress.get(stream_id) or {}
            if str(info.get("status") or "") != "running":
                continue
            active_phases.append(
                f"{self._stream_display_name(stream_id)}："
                f"{info.get('phase_name') or '并行组装'}"
            )
        active_text = " | 当前 " + "；".join(active_phases) if active_phases else ""
        self._assembly_information = (
            f"结果流组装：完成 {stream_ready}/{len(streams)} | "
            f"运行 {assembly_running} | 等待 {assembly_waiting} | "
            f"失败 {assembly_failed} | 并发 {assembly_running}/{assembly_limit}"
            f"{active_text}"
        )

    def _on_finished(self, result):
        value = dict(result or {})
        terminal_log = self._log_observations.observe_pipeline_failure(
            value, attempt_for=self._observations.attempt_for
        )
        if terminal_log is not None:
            self._append_presented_log(terminal_log)
        if value.get("terminal_published") is False:
            # This process lost or never acquired the Run.  Keep observing the
            # real owner instead of pinning snapshots to a fabricated failure.
            self.mark_finished(
                "本次执行已结束",
                "未发布任务终态；Run 当前状态以数据库同步结果为准。",
            )
            if self._query_client.is_bound and self._query_client.run_id:
                self._poll_timer.start()
            return
        change = self._observations.observe_pipeline_finished(value)
        self._refresh_streams(change.refresh_stream_ids)
        self.mark_finished(
            "已完成"
            if value.get("success")
            else "已停止"
            if value.get("status") == "stopped"
            else "失败"
        )
        if self._query_client.is_bound and self._query_client.run_id:
            self._poll_database()
            self._poll_timer.stop()

    def _request_stop(self):
        self.stop_requested.emit()

    def shutdown(self, timeout_ms=5000):
        from labeling_tool.qgis_support.qt_lifecycle import retire_after

        del timeout_ms
        if getattr(self, "_shutting_down", False):
            return
        self._shutting_down = True
        self.detail_page.set_active(False)
        self.events_page.set_active(False)
        self.overview_page.stop_transient_actions()
        self.detail_page.stop_transient_actions()
        self.events_page.stop_transient_actions()
        self._sync_timer.stop()
        self._poll_timer.stop()
        self.hide()
        self.unbind_state_database()
        retire_after(self, self.shutdown_finished)
        self._query_client.shutdown()

    def _on_query_shutdown_finished(self):
        if self._shutdown_emitted:
            return
        self._shutdown_emitted = True
        self.shutdown_finished.emit()

    def closeEvent(self, event):
        event.ignore()
        self.hide()
        parent = self.parent()
        if parent is not None and hasattr(parent, "show_monitor_btn"):
            parent.show_monitor_btn.setChecked(False)
            parent.show_monitor_btn.setText("推理监控")
