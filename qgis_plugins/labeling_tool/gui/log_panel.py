"""
Light-themed real-time log panel widget for labeling_tool.

Provides colored stdout/stderr/system output with filtering,
auto-scroll, save-to-file, and clipboard copy functionality.
"""

import time
from datetime import datetime

from qgis.PyQt.QtCore import QTimer, pyqtSignal
from qgis.PyQt.QtGui import QColor, QFont, QTextCharFormat
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
from ..qt_compat import FONT_BOLD, TEXT_CURSOR_END

# ── Design Tokens ──────────────────────────────────────────────────────────
LOG_BG = "#f6f6f6"
LOG_TEXT = "#212121"
STDERR_TEXT = "#c62828"
STDERR_BG = "#fff0ee"
WARNING_TEXT = "#8a5a00"
WARNING_BG = "#fff7df"
SYSTEM_TEXT = "#1565c0"
TIMESTAMP_COLOR = "#9e9e9e"
LOG_FONT = "Consolas, Monaco, Menlo, monospace"
LOG_FONT_SIZE = 12
TOOLBAR_BTN_BG = "#f0f0f0"
TOOLBAR_BTN_BORDER = "#c8c8c8"


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

        self._setup_ui()
        self._apply_styles()

    # ── Public API ─────────────────────────────────────────────────────

    def append_stdout(self, text: str) -> None:
        """Append black-text stdout output."""
        self.append_event(text, source="stdout", severity="info")

    def append_stderr(self, text: str) -> None:
        """Compatibility wrapper for an explicitly erroneous stderr event."""
        self.append_event(text, source="stderr", severity="error")

    def append_system(self, text: str) -> None:
        """Append blue italic system message."""
        self.append_event(text, source="system", severity="info")

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
    ) -> bool:
        """Append one event and return whether it created a new visible group."""

        source = source if source in {"stdout", "stderr", "system"} else "system"
        severity = severity if severity in {"info", "warning", "error"} else "info"
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        now = time.monotonic()
        raw_record = {
            "text": str(text),
            "source": source,
            "severity": severity,
            "timestamp": timestamp,
            "monotonic": now,
            "context_key": str(context_key or "unscoped"),
        }
        self._raw_records.append(raw_record)
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
            event["records"].extend(context_records)
            if self._event_visible(event):
                self._schedule_rebuild()
            return False

        event = {
            "text": str(text),
            "source": source,
            "severity": severity,
            "timestamp": timestamp,
            "last_timestamp": timestamp,
            "title": str(title or ""),
            "affected": str(affected or ""),
            "system_action": str(system_action or ""),
            "user_action": str(user_action or ""),
            "fingerprint": str(group_key),
            "repeat_count": 1,
            "records": context_records,
        }
        self._events.append(event)
        if group_key:
            self._event_index[group_key] = len(self._events) - 1
        if self._event_visible(event):
            self._render_event(event)
        return True

    def set_visible_severities(self, severities: set[str]) -> None:
        """Show only the requested semantic severities."""

        allowed = {"info", "warning", "error"}
        selected = set(severities).intersection(allowed)
        self._visible_severities = selected or allowed
        self._sync_severity_buttons()
        self._rebuild()

    def set_visible_levels(self, levels: set[str]) -> None:
        """Compatibility source filter for stdout/stderr/system callers.

        Expected items: ``"stdout"``, ``"stderr"``, ``"system"``.
        """
        self._visible_sources = set(levels).intersection(
            {"stdout", "stderr", "system"}
        )
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
            ("warning", "Warning"),
            ("error", "Error"),
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

        self._cb_autoscroll = QCheckBox("Auto-scroll")
        self._cb_autoscroll.setChecked(True)
        self._cb_autoscroll.toggled.connect(self._on_autoscroll_toggled)
        tb.addWidget(self._cb_autoscroll)

        self._btn_technical = QPushButton("技术详情")
        self._btn_technical.setCheckable(True)
        self._btn_technical.setToolTip("显示原始 stdout/stderr、返回码和技术堆栈")
        self._btn_technical.toggled.connect(self._on_technical_details_toggled)
        tb.addWidget(self._btn_technical)

        tb.addStretch()

        self._btn_save = QPushButton("Save")
        self._btn_save.clicked.connect(self._on_save)
        tb.addWidget(self._btn_save)

        self._btn_copy = QPushButton("Copy")
        self._btn_copy.clicked.connect(self._on_copy)
        tb.addWidget(self._btn_copy)

        self._btn_clear = QPushButton("Clear")
        self._btn_clear.clicked.connect(self.clear)
        tb.addWidget(self._btn_clear)

        layout.addWidget(toolbar)

        # ── log display ────────────────────────────────────────────────
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFrameStyle(QFrame.Shape.NoFrame)
        self.log_edit.setMaximumBlockCount(20000)
        layout.addWidget(self.log_edit)

    def _apply_styles(self) -> None:
        # Base font on the edit widget (CSS font-family doesn't always
        # apply to programmatically inserted text, so we set it both ways).
        self.log_edit.setFont(QFont(LOG_FONT, LOG_FONT_SIZE))
        self.log_edit.setStyleSheet(
            f"""
            QPlainTextEdit {{
                background-color: {LOG_BG};
                color: {LOG_TEXT};
                border: none;
            }}
            """
        )

        self.setStyleSheet(
            f"""
            LogPanel {{
                background-color: {LOG_BG};
            }}
            QWidget#logPanelToolbar {{
                background-color: {TOOLBAR_BTN_BG};
                border-bottom: 1px solid {TOOLBAR_BTN_BORDER};
            }}
            QPushButton {{
                background-color: {TOOLBAR_BTN_BG};
                border: 1px solid {TOOLBAR_BTN_BORDER};
                padding: 3px 10px;
                border-radius: 3px;
                font-size: 11pt;
            }}
            QPushButton:hover {{
                background-color: #e4e4e4;
            }}
            QPushButton:pressed {{
                background-color: #d4d4d4;
            }}
            QCheckBox {{
                spacing: 4px;
                font-size: 11pt;
            }}
            """
        )

    # ── Internal: append / render / rebuild ────────────────────────────

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

    def _render_event(self, event: dict[str, object]) -> None:
        """Render one readable event; raw details stay behind one toggle."""
        cursor = self.log_edit.textCursor()
        cursor.movePosition(TEXT_CURSOR_END)

        base_font = QFont(LOG_FONT, LOG_FONT_SIZE)
        timestamp = str(event["last_timestamp"])
        source = str(event["source"])
        severity = str(event["severity"])

        ts_fmt = QTextCharFormat()
        ts_fmt.setFont(base_font)
        ts_fmt.setForeground(QColor(TIMESTAMP_COLOR))

        txt_fmt = QTextCharFormat()
        txt_fmt.setFont(base_font)

        if severity == "warning":
            txt_fmt.setForeground(QColor(WARNING_TEXT))
            txt_fmt.setBackground(QColor(WARNING_BG))
            txt_fmt.setFontWeight(FONT_BOLD)
        elif severity == "error":
            txt_fmt.setForeground(QColor(STDERR_TEXT))
            txt_fmt.setBackground(QColor(STDERR_BG))
            txt_fmt.setFontWeight(FONT_BOLD)
        elif source == "stdout":
            txt_fmt.setForeground(QColor(LOG_TEXT))
        else:
            txt_fmt.setForeground(QColor(SYSTEM_TEXT))
            txt_fmt.setFontItalic(True)

        cursor.insertText(f"[{timestamp}] ", ts_fmt)
        if severity == "info":
            cursor.insertText(f"{event['text']}\n", txt_fmt)
        else:
            label = "WARNING" if severity == "warning" else "ERROR"
            repeat = int(event["repeat_count"])
            repeat_text = f" · 重复 {repeat} 次" if repeat > 1 else ""
            default_title = "运行警告" if severity == "warning" else "任务执行失败"
            title = str(event["title"] or default_title)
            cursor.insertText(f"{label} | {title}{repeat_text}\n", txt_fmt)

            body_fmt = QTextCharFormat()
            body_fmt.setFont(base_font)
            body_fmt.setForeground(QColor(LOG_TEXT))
            for label_text, value in (
                ("影响", event["affected"]),
                ("系统处理", event["system_action"]),
                ("用户操作", event["user_action"]),
            ):
                if value:
                    cursor.insertText(f"  {label_text}：{value}\n", body_fmt)
            if self._technical_details:
                detail_lines = [
                    f"[{record['timestamp']}] [{record['source']}] {record['text']}"
                    for record in event["records"]
                ]
                cursor.insertText(
                    "  技术详情：" + "\n    ".join(detail_lines) + "\n",
                    body_fmt,
                )
            cursor.insertText("\n", body_fmt)

        if self._autoscroll:
            sb = self.log_edit.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _rebuild(self) -> None:
        """Rebuild the visible content from the event cache.

        Uses full clear + re-insert rather than per-line setHidden().
        """
        scroll_pos = self.log_edit.verticalScrollBar().value()
        self.log_edit.clear()
        for event in self._events:
            if self._event_visible(event):
                self._render_event(event)
        if not self._autoscroll:
            sb = self.log_edit.verticalScrollBar()
            sb.setValue(min(scroll_pos, sb.maximum()))

    def _schedule_rebuild(self) -> None:
        """Coalesce high-frequency repeat updates into one UI refresh."""

        if self._rebuild_pending:
            return
        self._rebuild_pending = True
        QTimer.singleShot(100, self._finish_scheduled_rebuild)

    def _finish_scheduled_rebuild(self) -> None:
        self._rebuild_pending = False
        self._rebuild()

    # ── Slots ──────────────────────────────────────────────────────────

    def _select_severity(self, severity: str) -> None:
        if severity == "all":
            self.set_visible_severities({"info", "warning", "error"})
        else:
            self.set_visible_severities({severity})

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
                f"[{record['source']}] {record['text']}"
            )
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _on_copy(self) -> None:
        """Copy **all** events to clipboard with timestamp prefix."""
        lines = []
        for record in self._raw_records:
            lines.append(
                f"[{record['timestamp']}] [{str(record['severity']).upper()}] "
                f"[{record['source']}] {record['text']}"
            )
        all_text = "\n".join(lines)
        QApplication.clipboard().setText(all_text)
        self._btn_copy.setText("已复制")
        QTimer.singleShot(2000, self._reset_copy_button)

    def _reset_copy_button(self) -> None:
        """Reset the copy button text after the 2 s feedback timer."""
        self._btn_copy.setText("Copy")
