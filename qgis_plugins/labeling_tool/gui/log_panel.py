"""
Light-themed real-time log panel widget for labeling_tool.

Provides colored stdout/stderr/system output with filtering,
auto-scroll, save-to-file, and clipboard copy functionality.
"""

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
    """Light-themed real-time log panel with level filtering.

    Public methods
    --------------
    append_stdout(text)
        Append a standard-output line (black text).
    append_stderr(text)
        Append a standard-error line (red bold + light-red background).
    append_system(text)
        Append a system / informational line (blue italic).
    set_visible_levels(levels: set[str])
        Filter by ``{"stdout", "stderr", "system"}``.
    set_autoscroll(enabled: bool)
        Toggle automatic scroll-to-bottom on new content.
    clear()
        Clear all content.
    """

    cleared = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # ── internal state ─────────────────────────────────────────────
        self._events: list[tuple[str, str, str]] = []  # (text, level, timestamp)
        self._visible_levels: set[str] = {"stdout", "stderr", "system"}
        self._autoscroll: bool = True

        self._setup_ui()
        self._apply_styles()

    # ── Public API ─────────────────────────────────────────────────────

    def append_stdout(self, text: str) -> None:
        """Append black-text stdout output."""
        self._append(text, "stdout")

    def append_stderr(self, text: str) -> None:
        """Append red bold stderr output with light-red background."""
        self._append(text, "stderr")

    def append_system(self, text: str) -> None:
        """Append blue italic system message."""
        self._append(text, "system")

    def set_visible_levels(self, levels: set[str]) -> None:
        """Show only the given log levels.

        Expected items: ``"stdout"``, ``"stderr"``, ``"system"``.
        """
        self._visible_levels = set(levels)
        # Sync checkboxes without cascading rebuilds
        for name in ("stdout", "stderr", "system"):
            cb: QCheckBox = getattr(self, f"_cb_{name}")
            cb.blockSignals(True)
            cb.setChecked(name in self._visible_levels)
            cb.blockSignals(False)
        self._rebuild()

    def set_autoscroll(self, enabled: bool) -> None:
        """Enable or disable auto-scroll on new content."""
        self._autoscroll = enabled
        self._cb_autoscroll.blockSignals(True)
        self._cb_autoscroll.setChecked(enabled)
        self._cb_autoscroll.blockSignals(False)

    def clear(self) -> None:
        """Clear all logged events and the text display."""
        self._events.clear()
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

        for name in ("stdout", "stderr", "system"):
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.toggled.connect(self._on_level_toggled)
            setattr(self, f"_cb_{name}", cb)
            tb.addWidget(cb)

        tb.addWidget(_make_separator())

        self._cb_autoscroll = QCheckBox("Auto-scroll")
        self._cb_autoscroll.setChecked(True)
        self._cb_autoscroll.toggled.connect(self._on_autoscroll_toggled)
        tb.addWidget(self._cb_autoscroll)

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

    def _append(self, text: str, level: str) -> None:
        """Store an event and optionally render it."""
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self._events.append((text, level, ts))
        if level in self._visible_levels:
            self._render_event(text, level, ts)

    def _render_event(self, text: str, level: str, timestamp: str) -> None:
        """Insert a single timestamped line at the end of the log edit."""
        cursor = self.log_edit.textCursor()
        cursor.movePosition(TEXT_CURSOR_END)

        base_font = QFont(LOG_FONT, LOG_FONT_SIZE)

        ts_fmt = QTextCharFormat()
        ts_fmt.setFont(base_font)
        ts_fmt.setForeground(QColor(TIMESTAMP_COLOR))

        txt_fmt = QTextCharFormat()
        txt_fmt.setFont(base_font)

        if level == "stdout":
            txt_fmt.setForeground(QColor(LOG_TEXT))
        elif level == "stderr":
            txt_fmt.setForeground(QColor(STDERR_TEXT))
            txt_fmt.setBackground(QColor(STDERR_BG))
            txt_fmt.setFontWeight(FONT_BOLD)
        else:  # system
            txt_fmt.setForeground(QColor(SYSTEM_TEXT))
            txt_fmt.setFontItalic(True)

        cursor.insertText(f"[{timestamp}] ", ts_fmt)
        cursor.insertText(f"{text}\n", txt_fmt)

        if self._autoscroll:
            sb = self.log_edit.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _rebuild(self) -> None:
        """Rebuild the visible content from the event cache.

        Uses full clear + re-insert rather than per-line setHidden().
        """
        scroll_pos = self.log_edit.verticalScrollBar().value()
        self.log_edit.clear()
        for text, level, ts in self._events:
            if level in self._visible_levels:
                self._render_event(text, level, ts)
        if not self._autoscroll:
            sb = self.log_edit.verticalScrollBar()
            sb.setValue(min(scroll_pos, sb.maximum()))

    # ── Slots ──────────────────────────────────────────────────────────

    def _on_level_toggled(self) -> None:
        levels: set[str] = set()
        for name in ("stdout", "stderr", "system"):
            if getattr(self, f"_cb_{name}").isChecked():
                levels.add(name)
        self._visible_levels = levels
        self._rebuild()

    def _on_autoscroll_toggled(self, checked: bool) -> None:
        self._autoscroll = checked

    def _on_save(self) -> None:
        """Save **all** events (not only visible ones) to a ``.log`` file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Log", "", "Log Files (*.log);;All Files (*)"
        )
        if not path:
            return
        lines = [f"[{ts}] {txt}" for txt, _, ts in self._events]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _on_copy(self) -> None:
        """Copy **all** events to clipboard with timestamp prefix."""
        all_text = "\n".join(f"[{ts}] {txt}" for txt, _, ts in self._events)
        QApplication.clipboard().setText(all_text)
        self._btn_copy.setText("已复制")
        QTimer.singleShot(2000, self._reset_copy_button)

    def _reset_copy_button(self) -> None:
        """Reset the copy button text after the 2 s feedback timer."""
        self._btn_copy.setText("Copy")
