"""
Light-themed real-time log panel widget for labeling_tool.

Provides colored stdout/stderr/system output with filtering,
auto-scroll, save-to-file, and clipboard copy functionality.
"""

import time
from collections import deque
from datetime import datetime, timezone

from qgis.PyQt.QtCore import QTimer, pyqtSignal
from qgis.PyQt.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QTextBlockFormat,
    QTextCharFormat,
)
from qgis.PyQt.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from ..qt6_api import FONT_BOLD, TEXT_CURSOR_END, TEXT_LINE_PROPORTIONAL
from .monitor_time import format_monitor_timestamp, parse_monitor_timestamp

# ── Design Tokens ──────────────────────────────────────────────────────────
LOG_BG = "#f6f6f6"
LOG_TEXT = "#212121"
STDERR_TEXT = "#c62828"
STDERR_BG = "#fff0ee"
WARNING_TEXT = "#8a5a00"
WARNING_BG = "#fff7df"
SYSTEM_TEXT = "#1565c0"
TIMESTAMP_COLOR = "#9e9e9e"
LOG_FONT_SIZE = 11
TOOLBAR_BTN_BG = "#f0f0f0"
TOOLBAR_BTN_BORDER = "#c8c8c8"
LOG_THEME_COLORS = {
    "light": {
        "background": LOG_BG,
        "text": LOG_TEXT,
        "stderr": STDERR_TEXT,
        "stderr_background": STDERR_BG,
        "warning": WARNING_TEXT,
        "warning_background": WARNING_BG,
        "system": SYSTEM_TEXT,
        "timestamp": TIMESTAMP_COLOR,
        "toolbar": TOOLBAR_BTN_BG,
        "border": TOOLBAR_BTN_BORDER,
        "hover": "#e4e4e4",
        "pressed": "#d4d4d4",
    },
    "dark": {
        "background": "#192A3D",
        "text": "#EDF4FA",
        "stderr": "#FF8A92",
        "stderr_background": "#47262D",
        "warning": "#FFC25B",
        "warning_background": "#41351F",
        "system": "#55C8E7",
        "timestamp": "#8297AA",
        "toolbar": "#253B52",
        "border": "#36516C",
        "hover": "#2D4964",
        "pressed": "#20364D",
    },
}
MAX_VISIBLE_LOG_BLOCKS = 4000
MAX_CACHED_EVENTS = 5000
MAX_RAW_RECORDS = 20000
MAX_EVENT_DETAIL_RECORDS = 200
REBUILD_BATCH_SIZE = 80
REBUILD_FRAME_BUDGET_SECONDS = 0.006
REBUILD_INTERVAL_MS = 16
LOG_BLOCK_LINE_HEIGHT = 122


def _make_separator() -> QFrame:
    """Create a vertical-line separator for the toolbar."""
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.VLine)
    sep.setFrameShadow(QFrame.Shadow.Sunken)
    return sep


class LogPanel(QWidget):
    """Readable real-time log panel with source and severity separated.

    Public methods
    --------------
    append_stdout(text)
        Append a standard-output line (black text).
    append_stderr(text)
        Append a standard-error line (red bold + light-red background).
    append_system(text)
        Append a system / informational line (blue italic).
    append_event(...)
        Append one event with independent source and severity fields.
    set_visible_severities(severities: set[str])
        Filter by ``{"info", "warning", "error"}``.
    set_autoscroll(enabled: bool)
        Toggle automatic scroll-to-bottom on new content.
    clear()
        Clear all content.
    """

    cleared = pyqtSignal()
    severity_selected = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # ── internal state ─────────────────────────────────────────────
        self._events: list[dict[str, object]] = []
        self._event_index: dict[str, int] = {}
        self._raw_records: list[dict[str, object]] = []
        self._pending_stderr_records: dict[str, list[dict[str, object]]] = {}
        self._visible_sources: set[str] = {"stdout", "stderr", "system"}
        self._visible_severities: set[str] = {"info", "warning", "error"}
        self._autoscroll: bool = True
        self._technical_details: bool = False
        self._rebuild_pending: bool = False
        self._rebuild_in_progress: bool = False
        self._rebuild_events: tuple[dict[str, object], ...] = ()
        self._rebuild_index: int = 0
        self._rebuild_scroll_pos: int = 0
        self._append_batch_depth: int = 0
        self._append_batch_start: int = 0
        self._append_batch_requires_rebuild: bool = False
        self._append_batch_new_events: list[dict[str, object]] = []
        self._append_render_queue = deque()
        self._theme = "light"
        self._theme_palette = LOG_THEME_COLORS[self._theme]

        self._setup_ui()
        self._apply_styles()

        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(REBUILD_INTERVAL_MS)
        self._rebuild_timer.timeout.connect(self._render_rebuild_batch)

        self._coalesced_rebuild_timer = QTimer(self)
        self._coalesced_rebuild_timer.setSingleShot(True)
        self._coalesced_rebuild_timer.setInterval(100)
        self._coalesced_rebuild_timer.timeout.connect(
            self._finish_scheduled_rebuild
        )
        self._append_render_timer = QTimer(self)
        self._append_render_timer.setSingleShot(True)
        self._append_render_timer.setInterval(REBUILD_INTERVAL_MS)
        self._append_render_timer.timeout.connect(self._render_append_batch)

    # ── Public API ─────────────────────────────────────────────────────

    def set_theme(self, theme: str) -> None:
        """Apply the monitor's scoped theme without changing QGIS globally."""

        selected = str(theme or "light").lower()
        if selected not in LOG_THEME_COLORS:
            selected = "light"
        if selected == self._theme:
            return
        self._theme = selected
        self._theme_palette = LOG_THEME_COLORS[selected]
        self._apply_styles()
        self.refresh_visible()

    def append_stdout(self, text: str) -> None:
        """Append black-text stdout output."""
        self.append_event(text, source="stdout", severity="info")

    def append_stderr(self, text: str) -> None:
        """Compatibility wrapper for an explicitly erroneous stderr event."""
        self.append_event(text, source="stderr", severity="error")

    def append_system(self, text: str) -> None:
        """Append blue italic system message."""
        self.append_event(text, source="system", severity="info")

    def begin_batch(self) -> None:
        if self._append_batch_depth == 0:
            self._append_batch_start = len(self._events)
            self._append_batch_requires_rebuild = False
            self._append_batch_new_events = []
        self._append_batch_depth += 1

    def end_batch(self) -> None:
        self._append_batch_depth = max(0, self._append_batch_depth - 1)
        if self._append_batch_depth != 0 or not self.isVisible():
            return
        if self._append_batch_requires_rebuild or self._rebuild_in_progress:
            self._schedule_rebuild()
            return
        self._append_render_queue.extend(
            event
            for event in self._append_batch_new_events
            if self._event_visible(event)
        )
        self._append_batch_new_events = []
        if len(self._append_render_queue) > MAX_CACHED_EVENTS:
            # Only the bounded display window is rebuilt. Full JSONL logging
            # remains on the runtime's independent log writer.
            self._append_render_queue.clear()
            self._schedule_rebuild()
            return
        if self._append_render_queue and not self._append_render_timer.isActive():
            self._append_render_timer.start()

    def refresh_visible(self) -> None:
        if self.isVisible():
            self._schedule_rebuild()

    def append_event(
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
        event_timestamp=None,
    ) -> bool:
        """Append one event and return whether it created a new visible group."""

        source = source if source in {"stdout", "stderr", "system"} else "system"
        severity = severity if severity in {"info", "warning", "error"} else "info"
        received_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        has_source_time = event_timestamp is not None and event_timestamp != ""
        parsed = parse_monitor_timestamp(event_timestamp) if has_source_time else None
        timestamp = (
            parsed.isoformat() if parsed is not None
            else str(event_timestamp) if has_source_time else received_at
        )
        timestamp_kind = "captured" if has_source_time else "received"
        now = time.monotonic()
        raw_record = {
            "text": str(text),
            "source": source,
            "severity": severity,
            "timestamp": timestamp,
            "timestamp_kind": timestamp_kind,
            "source_timestamp": event_timestamp,
            "received_at": received_at,
            "monotonic": now,
            "context_key": str(context_key or "unscoped"),
        }
        self._raw_records.append(raw_record)
        if len(self._raw_records) > MAX_RAW_RECORDS:
            del self._raw_records[: len(self._raw_records) - MAX_RAW_RECORDS]
        pending_key = str(raw_record["context_key"])
        pending_records = [
            record
            for record in self._pending_stderr_records.get(pending_key, [])
            if now - float(record["monotonic"]) <= 10.0
        ]
        if source == "stderr" and severity == "info":
            pending_records.append(raw_record)
            self._pending_stderr_records[pending_key] = pending_records[-200:]
        context_records = (
            [*pending_records, raw_record]
            if severity in {"warning", "error"}
            else [raw_record]
        )
        if severity in {"warning", "error"}:
            self._pending_stderr_records.pop(pending_key, None)
        if len(self._pending_stderr_records) > 128:
            stale_keys = [
                key
                for key, records in self._pending_stderr_records.items()
                if not records or now - float(records[-1]["monotonic"]) > 10.0
            ]
            for key in stale_keys:
                self._pending_stderr_records.pop(key, None)
        group_key = fingerprint if severity in {"warning", "error"} else ""
        if group_key and group_key in self._event_index:
            event = self._events[self._event_index[group_key]]
            event["repeat_count"] = int(event["repeat_count"]) + 1
            event["last_timestamp"] = timestamp
            event["timestamp_kind"] = timestamp_kind
            event["records"].extend(context_records)
            if len(event["records"]) > MAX_EVENT_DETAIL_RECORDS:
                del event["records"][:-MAX_EVENT_DETAIL_RECORDS]
            if int(getattr(self, "_append_batch_depth", 0)):
                self._append_batch_requires_rebuild = True
            elif self._event_visible(event) and getattr(
                self, "isVisible", lambda: True
            )():
                self._schedule_rebuild()
            return False

        event = {
            "text": str(text),
            "source": source,
            "severity": severity,
            "timestamp": timestamp,
            "last_timestamp": timestamp,
            "timestamp_kind": timestamp_kind,
            "title": str(title or ""),
            "affected": str(affected or ""),
            "system_action": str(system_action or ""),
            "user_action": str(user_action or ""),
            "fingerprint": str(group_key),
            "repeat_count": 1,
            "records": context_records,
        }
        self._events.append(event)
        if int(getattr(self, "_append_batch_depth", 0)):
            self._append_batch_new_events.append(event)
        if group_key:
            self._event_index[group_key] = len(self._events) - 1
        self._trim_event_cache()
        render_live = (
            self._event_visible(event)
            and getattr(self, "isVisible", lambda: True)()
            and int(getattr(self, "_append_batch_depth", 0)) == 0
        )
        if render_live and not self._rebuild_in_progress:
            self._render_event(event)
        elif render_live:
            self._schedule_rebuild()
        return True

    def set_visible_severities(self, severities: set[str]) -> None:
        """Show only the requested semantic severities."""

        allowed = {"info", "warning", "error"}
        selected = set(severities).intersection(allowed)
        selected = selected or allowed
        if selected == self._visible_severities:
            self._sync_severity_buttons()
            return
        self._visible_severities = selected
        self._sync_severity_buttons()
        self._rebuild()

    def set_visible_levels(self, levels: set[str]) -> None:
        """Compatibility source filter for stdout/stderr/system callers.

        Expected items: ``"stdout"``, ``"stderr"``, ``"system"``.
        """
        selected = set(levels).intersection(
            {"stdout", "stderr", "system"}
        )
        if selected == self._visible_sources:
            return
        self._visible_sources = selected
        self._rebuild()

    def set_autoscroll(self, enabled: bool) -> None:
        """Enable or disable auto-scroll on new content."""
        self._autoscroll = enabled
        self._cb_autoscroll.blockSignals(True)
        self._cb_autoscroll.setChecked(enabled)
        self._cb_autoscroll.blockSignals(False)

    def scroll_to_latest(self) -> None:
        """Reveal the newest item even when continuous auto-scroll is disabled."""

        scrollbar = self.log_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear(self) -> None:
        """Clear all logged events and the text display."""
        self._events.clear()
        self._event_index.clear()
        self._raw_records.clear()
        self._pending_stderr_records.clear()
        self._rebuild_pending = False
        self._rebuild_in_progress = False
        self._rebuild_events = ()
        self._rebuild_index = 0
        self._append_batch_depth = 0
        self._append_batch_start = 0
        self._append_batch_requires_rebuild = False
        self._append_batch_new_events.clear()
        self._append_render_queue.clear()
        if hasattr(self, "_rebuild_timer"):
            self._rebuild_timer.stop()
        if hasattr(self, "_coalesced_rebuild_timer"):
            self._coalesced_rebuild_timer.stop()
        if hasattr(self, "_append_render_timer"):
            self._append_render_timer.stop()
        self.log_edit.clear()
        self.cleared.emit()

    # ── UI construction ────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── toolbar ────────────────────────────────────────────────────
        toolbar = QWidget()
        toolbar.setObjectName("logPanelToolbar")
        tb = QHBoxLayout(toolbar)
        tb.setContentsMargins(6, 4, 6, 4)
        tb.setSpacing(6)

        self._severity_buttons = {}
        for severity, label in (
            ("all", "全部"),
            ("info", "普通"),
            ("warning", "警告"),
            ("error", "错误"),
        ):
            button = QPushButton(label)
            button.setCheckable(True)
            button.clicked.connect(
                lambda _checked, value=severity: self._select_severity(value)
            )
            self._severity_buttons[severity] = button
            tb.addWidget(button)
        self._sync_severity_buttons()

        tb.addWidget(_make_separator())

        self._cb_autoscroll = QCheckBox("自动滚动")
        self._cb_autoscroll.setChecked(True)
        self._cb_autoscroll.toggled.connect(self._on_autoscroll_toggled)
        tb.addWidget(self._cb_autoscroll)

        self._btn_technical = QPushButton("技术详情")
        self._btn_technical.setCheckable(True)
        self._btn_technical.setToolTip("显示原始 stdout/stderr、返回码和技术堆栈")
        self._btn_technical.toggled.connect(self._on_technical_details_toggled)
        tb.addWidget(self._btn_technical)

        tb.addStretch()

        self._btn_save = QPushButton("保存")
        self._btn_save.clicked.connect(self._on_save)
        tb.addWidget(self._btn_save)

        self._btn_copy = QPushButton("复制")
        self._btn_copy.clicked.connect(self._on_copy)
        tb.addWidget(self._btn_copy)

        self._btn_clear = QPushButton("清空显示")
        self._btn_clear.setToolTip("仅清空当前显示，不删除磁盘日志或数据库历史")
        self._btn_clear.clicked.connect(self.clear)
        tb.addWidget(self._btn_clear)

        layout.addWidget(toolbar)

        # ── log display ────────────────────────────────────────────────
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFrameStyle(QFrame.Shape.NoFrame)
        self.log_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log_edit.setMaximumBlockCount(MAX_VISIBLE_LOG_BLOCKS)
        layout.addWidget(self.log_edit)

    def _apply_styles(self) -> None:
        # Base font on the edit widget (CSS font-family doesn't always
        # apply to programmatically inserted text, so we set it both ways).
        self.log_edit.setFont(self._log_font())
        self.log_edit.document().setDocumentMargin(9.0)
        palette = self._theme_palette
        self.log_edit.setStyleSheet(
            f"""
            QPlainTextEdit {{
                background-color: {palette['background']};
                color: {palette['text']};
                border: none;
            }}
            """
        )

        self.setStyleSheet(
            f"""
            LogPanel {{
                background-color: {palette['background']};
            }}
            QWidget#logPanelToolbar {{
                background-color: {palette['toolbar']};
                border-bottom: 1px solid {palette['border']};
            }}
            QPushButton {{
                color: {palette['text']};
                background-color: {palette['toolbar']};
                border: 1px solid {palette['border']};
                padding: 7px 10px;
                border-radius: 6px;
                min-height: 20px;
                font-size: 11pt;
            }}
            QPushButton:hover {{
                background-color: {palette['hover']};
            }}
            QPushButton:pressed {{
                background-color: {palette['pressed']};
            }}
            QCheckBox {{
                color: {palette['text']};
                spacing: 4px;
                font-size: 11pt;
            }}
            """
        )

    # ── Internal: append / render / rebuild ────────────────────────────

    def _log_font(self, *, technical=False) -> QFont:
        """Use natural Chinese body spacing; reserve monospace for raw data."""
        font = (
            QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
            if technical else QFont(self.font())
        )
        font.setPointSize(LOG_FONT_SIZE)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0)
        font.setWordSpacing(0)
        font.setStretch(100)
        font.setKerning(True)
        return font

    @staticmethod
    def _set_log_block_format(cursor) -> None:
        """Apply readable native line spacing to the cursor's current block."""
        block_format = QTextBlockFormat()
        block_format.setLineHeight(
            LOG_BLOCK_LINE_HEIGHT,
            TEXT_LINE_PROPORTIONAL,
        )
        cursor.setBlockFormat(block_format)

    def _event_visible(self, event: dict[str, object]) -> bool:
        return (
            event["source"] in self._visible_sources
            and event["severity"] in self._visible_severities
            and not (
                event["severity"] == "info"
                and event["source"] == "stderr"
                and not self._technical_details
            )
        )

    def _render_event(
        self, event: dict[str, object], *, follow_tail: bool = True
    ) -> None:
        """Render one readable event; raw details stay behind one toggle."""
        cursor = self.log_edit.textCursor()
        cursor.movePosition(TEXT_CURSOR_END)

        base_font = self._log_font()
        self._set_log_block_format(cursor)
        timestamp = format_monitor_timestamp(event["last_timestamp"], compact=True)
        if event.get("timestamp_kind") == "received":
            timestamp += " · 接收"
        source = str(event["source"])
        severity = str(event["severity"])
        palette = self._theme_palette

        ts_fmt = QTextCharFormat()
        ts_fmt.setFont(self._log_font(technical=True))
        ts_fmt.setForeground(QColor(palette["timestamp"]))

        txt_fmt = QTextCharFormat()
        txt_fmt.setFont(base_font)

        if severity == "warning":
            txt_fmt.setForeground(QColor(palette["warning"]))
            txt_fmt.setBackground(QColor(palette["warning_background"]))
            txt_fmt.setFontWeight(FONT_BOLD)
        elif severity == "error":
            txt_fmt.setForeground(QColor(palette["stderr"]))
            txt_fmt.setBackground(QColor(palette["stderr_background"]))
            txt_fmt.setFontWeight(FONT_BOLD)
        elif source == "stdout":
            txt_fmt.setForeground(QColor(palette["text"]))
        else:
            txt_fmt.setForeground(QColor(palette["system"]))

        cursor.insertText(f"[{timestamp}] ", ts_fmt)
        if severity == "info":
            cursor.insertText(f"{event['text']}\n", txt_fmt)
        else:
            label = "警告" if severity == "warning" else "错误"
            repeat = int(event["repeat_count"])
            repeat_text = f" · 重复 {repeat} 次" if repeat > 1 else ""
            default_title = "运行警告" if severity == "warning" else "任务执行失败"
            title = str(event["title"] or default_title)
            cursor.insertText(f"{label} · {title}{repeat_text}\n", txt_fmt)

            body_fmt = QTextCharFormat()
            body_fmt.setFont(base_font)
            body_fmt.setForeground(QColor(palette["text"]))
            for label_text, value in (
                ("影响", event["affected"]),
                ("系统处理", event["system_action"]),
                ("用户操作", event["user_action"]),
            ):
                if value:
                    cursor.insertText(f"  {label_text}：{value}\n", body_fmt)
            if self._technical_details:
                technical_fmt = QTextCharFormat(body_fmt)
                technical_fmt.setFont(self._log_font(technical=True))
                detail_lines = [
                    f"[{record['timestamp']}] [{record['source']}] "
                    f"[{'采集' if record.get('timestamp_kind') == 'captured' else '接收'}] {record['text']}"
                    for record in event["records"]
                ]
                cursor.insertText(
                    "  技术详情：" + "\n    ".join(detail_lines) + "\n",
                    technical_fmt,
                )
            cursor.insertText("\n", body_fmt)

        if follow_tail and self._autoscroll:
            sb = self.log_edit.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _trim_event_cache(self) -> None:
        """Bound Python-side history independently from the text document."""

        overflow = len(self._events) - MAX_CACHED_EVENTS
        if overflow <= 0:
            return
        del self._events[:overflow]
        self._event_index = {
            str(event["fingerprint"]): index
            for index, event in enumerate(self._events)
            if event["fingerprint"]
        }

    def _rebuild(self) -> None:
        """Rebuild the visible content from the event cache.

        Uses full clear + re-insert rather than per-line setHidden().
        """
        self._rebuild_timer.stop()
        self._append_render_timer.stop()
        self._append_render_queue.clear()
        self._rebuild_pending = False
        self._rebuild_scroll_pos = self.log_edit.verticalScrollBar().value()
        self._rebuild_events = tuple(
            event for event in self._events if self._event_visible(event)
        )
        self._rebuild_index = 0
        self._rebuild_in_progress = True
        self.log_edit.clear()
        self._render_rebuild_batch()

    def _render_rebuild_batch(self) -> None:
        """Render one short GUI-thread batch, then yield to Wayland events."""

        if not self._rebuild_in_progress:
            return
        deadline = time.perf_counter() + REBUILD_FRAME_BUDGET_SECONDS
        rendered = 0
        while (
            self._rebuild_index < len(self._rebuild_events)
            and rendered < REBUILD_BATCH_SIZE
            and time.perf_counter() < deadline
        ):
            event = self._rebuild_events[self._rebuild_index]
            self._rebuild_index += 1
            rendered += 1
            self._render_event(event, follow_tail=False)

        if self._rebuild_index < len(self._rebuild_events):
            self._rebuild_timer.start()
            return

        self._rebuild_in_progress = False
        self._rebuild_events = ()
        self._rebuild_index = 0
        scrollbar = self.log_edit.verticalScrollBar()
        if self._autoscroll:
            scrollbar.setValue(scrollbar.maximum())
        else:
            scrollbar.setValue(
                min(self._rebuild_scroll_pos, scrollbar.maximum())
            )
        if self._rebuild_pending:
            self._coalesced_rebuild_timer.start()

    def _render_append_batch(self) -> None:
        if not self.isVisible() or self._rebuild_in_progress:
            return
        deadline = time.perf_counter() + REBUILD_FRAME_BUDGET_SECONDS
        rendered = 0
        while (
            self._append_render_queue
            and rendered < REBUILD_BATCH_SIZE
            and time.perf_counter() < deadline
        ):
            event = self._append_render_queue.popleft()
            self._render_event(event, follow_tail=False)
            rendered += 1
        if self._append_render_queue:
            self._append_render_timer.start()
        elif self._autoscroll:
            scrollbar = self.log_edit.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def _schedule_rebuild(self) -> None:
        """Coalesce high-frequency repeat updates into one UI refresh."""

        if self._rebuild_pending:
            return
        self._rebuild_pending = True
        self._coalesced_rebuild_timer.start()

    def _finish_scheduled_rebuild(self) -> None:
        if self._rebuild_in_progress:
            return
        self._rebuild_pending = False
        self._rebuild()

    # ── Slots ──────────────────────────────────────────────────────────

    def _select_severity(self, severity: str) -> None:
        if severity == "all":
            self.set_visible_severities({"info", "warning", "error"})
        else:
            self.set_visible_severities({severity})
        self.severity_selected.emit(severity)

    def _sync_severity_buttons(self) -> None:
        if not hasattr(self, "_severity_buttons"):
            return
        all_selected = self._visible_severities == {"info", "warning", "error"}
        for severity, button in self._severity_buttons.items():
            button.blockSignals(True)
            button.setChecked(
                all_selected
                if severity == "all"
                else self._visible_severities == {severity}
            )
            button.blockSignals(False)

    def _on_autoscroll_toggled(self, checked: bool) -> None:
        self._autoscroll = checked

    def _on_technical_details_toggled(self, checked: bool) -> None:
        self._technical_details = bool(checked)
        self._rebuild()

    def _on_save(self) -> None:
        """Save **all** events (not only visible ones) to a ``.log`` file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Log", "", "Log Files (*.log);;All Files (*)"
        )
        if not path:
            return
        lines = []
        for record in self._raw_records:
            lines.append(
                f"[{record['timestamp']}] [{str(record['severity']).upper()}] "
                f"[{record['source']}] "
                f"[{'采集' if record.get('timestamp_kind') == 'captured' else '接收'}] {record['text']}"
            )
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _on_copy(self) -> None:
        """Copy **all** events to clipboard with timestamp prefix."""
        lines = []
        for record in self._raw_records:
            lines.append(
                f"[{record['timestamp']}] [{str(record['severity']).upper()}] "
                f"[{record['source']}] "
                f"[{'采集' if record.get('timestamp_kind') == 'captured' else '接收'}] {record['text']}"
            )
        all_text = "\n".join(lines)
        QApplication.clipboard().setText(all_text)
        self._btn_copy.setText("已复制")
        QTimer.singleShot(2000, self._reset_copy_button)

    def _reset_copy_button(self) -> None:
        """Reset the copy button text after the 2 s feedback timer."""
        self._btn_copy.setText("复制")
