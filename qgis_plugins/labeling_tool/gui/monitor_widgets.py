"""Small native presentation primitives for the inference instrument panel."""

from functools import lru_cache
import time

from qgis.PyQt.QtCore import QByteArray, QEvent, QRectF, QSize, QTimer
from qgis.PyQt.QtGui import (
    QColor, QIcon, QLinearGradient, QPainter, QPainterPath, QPixmap, QTextBlockFormat, QTextCursor,
)
from qgis.PyQt.QtSvg import QSvgRenderer
from qgis.PyQt.QtWidgets import (
    QComboBox, QListView, QStyledItemDelegate, QProgressBar, QTableWidget,
    QTextBrowser, QToolTip,
)

from .monitor_theme import DETAIL_LINE_HEIGHT, PALETTES, combo_popup_style
from ..qt6_api import (
    ALIGN_RIGHT, ALIGN_VCENTER, INTERACTIVE, NO_PEN, SCROLLBAR_AS_NEEDED, TEXT_CURSOR_DOCUMENT,
    TEXT_LINE_PROPORTIONAL, TRANSPARENT,
)

_PATHS = {
    "contour": '<path d="M21 3C30 1 30 13 34 20S34 36 25 37 15 32 7 30-1 19 6 15 15 14 16 7 18 3 21 3Z"/><path d="M21 9C27 7 25 17 29 21S29 31 24 31 17 27 11 26 7 20 13 19 19 16 19 12 20 9 21 9Z"/><path d="M21 17C25 16 27 23 23 25S16 24 16 22 19 18 21 17Z"/>',
    "chip": '<rect x="10" y="10" width="20" height="20" rx="3"/><rect x="15" y="15" width="10" height="10" rx="1"/><path d="M14 5v5m6-5v5m6-5v5M14 30v5m6-5v5m6-5v5M5 14h5m-5 6h5m-5 6h5m20-12h5m-5 6h5m-5 6h5"/>',
    "shield": '<path d="M20 4 34 10v11c0 7-8 12-14 16C14 33 6 28 6 21V10Z"/><path d="m13 20 5 5 10-11"/>',
    "alert": '<path d="M20 4 37 34H3Z"/><path d="M20 14v10m0 5v1"/>',
    "clock": '<circle cx="20" cy="20" r="14"/><path d="M20 11v10l7 4"/>',
    "cube": '<path d="m20 4 14 8v16l-14 8-14-8V12Zm0 16v16M6 12l14 8 14-8M13 8l14 8"/>',
    "layers": '<path d="m20 5 16 9-16 9-16-9Zm-16 16 16 9 16-9M4 28l16 9 16-9"/>',
    "document": '<path d="M10 4h14l7 7v25H10Zm14 0v9h7M15 20h11m-11 6h11"/>',
    "arrow": '<path d="M10 20h20m-8-8 8 8-8 8"/>',
    "sun": '<circle cx="20" cy="20" r="7"/><path d="M20 3v5m0 24v5M3 20h5m24 0h5M8 8l4 4m16 16 4 4M8 32l4-4m16-16 4-4"/>',
    "moon": '<path d="M33.5 21.185A13.5 13.5 0 1 1 18.815 6.5a10.5 10.5 0 0 0 14.685 14.685Z"/>',
    "stop": '<circle cx="20" cy="20" r="14"/><rect x="14" y="14" width="12" height="12" rx="1" fill="currentColor"/>',
    "activity": '<path d="M3 22h8l5-12 8 22 5-10h8"/>',
    "check": '<path d="m8 21 8 8 17-18"/>',
    "ring": '<path d="M32 26A14 14 0 1 1 30 10"/><circle cx="32" cy="14" r="2" fill="currentColor" stroke="none"/>',
}


@lru_cache(maxsize=96)
def monitor_icon(name, color, size=24):
    """Crisp semantic line icons at device scale; no external font dependency."""
    body = _PATHS.get(name, _PATHS["document"]).replace("currentColor", color)
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40"><g fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{body}</g></svg>'
    pixmap = QPixmap(size * 2, size * 2)
    pixmap.fill(TRANSPARENT)
    painter = QPainter(pixmap)
    QSvgRenderer(QByteArray(svg.encode())).render(painter)
    painter.end()
    pixmap.setDevicePixelRatio(2)
    return QIcon(pixmap)


class MonitorComboBox(QComboBox):
    """Native combo with a themed list instead of platform-specific menu paint."""

    def __init__(self, parent=None):
        super().__init__(parent)
        view = QListView(self)
        view.setUniformItemSizes(True)
        view.setMouseTracking(True)
        self.setView(view)
        view.setVerticalScrollBarPolicy(SCROLLBAR_AS_NEEDED)
        # The default combo menu delegate can bypass QListView item padding.
        self.setItemDelegate(QStyledItemDelegate(view))
        self.setMaxVisibleItems(10)

    def apply_theme(self, theme):
        view = self.view()
        view.setFont(self.font())
        style = combo_popup_style(theme)
        view.setStyleSheet(style)
        # setView creates a separate native popup. Styling only the list leaves
        # the container's frame/margins in the system's (often light) palette.
        # Target this combo's popup, never the enclosing monitor or QGIS menus.
        popup = view.window()
        if popup is not self.window():
            popup.setObjectName("MonitorComboPopup")
            popup.setStyleSheet(style)
            if popup.layout() is not None:
                popup.layout().setContentsMargins(0, 0, 0, 0)
                popup.layout().setSpacing(0)


class MonitorTextBrowser(QTextBrowser):
    """Consistent reading rhythm for bounded object/attempt detail documents.

    Only format a newly replaced detail document, never the streaming log or
    the full on-disk history. Native block formatting preserves selectable
    plain text, rich-text links and the existing anchor navigation.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.document().setDocumentMargin(10.0)

    def setText(self, text):
        super().setText(text)
        self._format_lines()

    def setPlainText(self, text):
        super().setPlainText(text)
        self._format_lines()

    def setHtml(self, text):
        super().setHtml(text)
        self._format_lines()

    def _format_lines(self):
        cursor = QTextCursor(self.document())
        cursor.select(TEXT_CURSOR_DOCUMENT)
        spacing = QTextBlockFormat()
        spacing.setLineHeight(DETAIL_LINE_HEIGHT, TEXT_LINE_PROPORTIONAL)
        cursor.mergeBlockFormat(spacing)


class ProgressTrack(QProgressBar):
    """A thin track with its percentage outside, as on the reference dashboard."""

    def __init__(self, parent=None, *, percentage=True):
        super().__init__(parent)
        self._percentage = percentage
        self.setTextVisible(False)
        self.setRange(0, 1)
        self.setValue(0)

    @property
    def percentage_visible(self):
        return self._percentage

    def sizeHint(self):
        return QSize(240, max(24, self.fontMetrics().height() + 6))

    def minimumSizeHint(self):
        return QSize(80, self.sizeHint().height())

    def paintEvent(self, event):
        palette = PALETTES.get(str(self.property("theme") or "dark"), PALETTES["dark"])
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        label_width = self.fontMetrics().horizontalAdvance("100%") + 14 if self._percentage else 0
        rect = QRectF(0, (self.height() - 10) / 2, max(1, self.width() - label_width), 10)
        painter.setPen(NO_PEN)
        painter.setBrush(QColor(palette["track"]))
        painter.drawRoundedRect(rect, 5, 5)
        denominator = self.maximum() - self.minimum()
        fraction = max(0, min(1, (self.value() - self.minimum()) / denominator)) if denominator else 0
        if fraction > 0:
            fill = QRectF(rect)
            fill.setWidth(max(5, rect.width() * fraction))
            gradient = QLinearGradient(fill.topLeft(), fill.topRight())
            gradient.setColorAt(0, QColor(palette["accent"]))
            gradient.setColorAt(1, QColor(palette["accent_end"]))
            painter.setBrush(gradient)
            painter.drawRoundedRect(fill, 5, 5)
        if self._percentage:
            painter.setPen(QColor(palette["text"]))
            known = bool(self.property("progressKnown"))
            value = f"{fraction:.0%}" if known else "—"
            painter.drawText(QRectF(self.width() - label_width, 0, label_width, self.height()),
                             ALIGN_RIGHT | ALIGN_VCENTER, value)
        painter.end()


class OverallProgressTrack(ProgressTrack):
    """Animate paint only; the QProgressBar value remains authoritative."""

    def __init__(self, parent=None):
        super().__init__(parent, percentage=False)
        self.setFixedHeight(14)
        self._running = False
        self._reduced = False
        self._shown = 0.0
        self._origin = 0.0
        self._changed_at = time.monotonic()
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self.update)
        self.valueChanged.connect(self._value_changed)

    def _fraction(self):
        span = self.maximum() - self.minimum()
        return max(0.0, min(1.0, (self.value() - self.minimum()) / span)) if span > 0 else 0.0

    def _value_changed(self, *_args):
        target = self._fraction()
        self._origin = min(self._shown, target)
        self._changed_at = time.monotonic()
        if not self._running or self._reduced or not self.isVisible():
            self._shown = target
        self.update()

    def set_running(self, running):
        self._running = bool(running)
        self._sync_motion()

    def set_reduced_motion(self, reduced):
        self._reduced = bool(reduced)
        self._sync_motion()

    def _sync_motion(self):
        if self._running and not self._reduced and self.isVisible():
            self._timer.start()
        else:
            self._timer.stop()
            self._shown = self._fraction()
        self.update()

    def showEvent(self, event):
        super().showEvent(event)
        self._sync_motion()

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def paintEvent(self, event):
        colors = PALETTES.get(str(self.property("theme") or "dark"), PALETTES["dark"])
        target = self._fraction()
        moving = self._timer.isActive()
        elapsed = min(1.0, (time.monotonic() - self._changed_at) / 0.22)
        self._shown = self._origin + (target - self._origin) * (1 - (1 - elapsed) ** 3) if moving else target
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(NO_PEN)
        rect = QRectF(self.rect())
        clip = QPainterPath()
        clip.addRoundedRect(rect, 7, 7)
        painter.setClipPath(clip)
        painter.fillRect(rect, QColor(colors["track"]))
        if self.maximum() == self.minimum():
            if moving:
                width = rect.width() * 0.2
                offset = (time.monotonic() % 1.8) / 1.8 * (rect.width() + width) - width
                painter.setBrush(QColor(colors["accent"]))
                painter.drawRoundedRect(QRectF(offset, 0, width, rect.height()), 7, 7)
            painter.end()
            return
        fill = QRectF(rect)
        fill.setWidth(rect.width() * self._shown)
        fill_clip = QPainterPath()
        fill_clip.addRoundedRect(fill, min(7, fill.width() / 2), 7)
        painter.setClipPath(clip.intersected(fill_clip))
        gradient = QLinearGradient(rect.topLeft(), rect.topRight())
        gradient.setColorAt(0, QColor(colors["accent"]))
        gradient.setColorAt(1, QColor(colors["accent_end"]))
        painter.fillRect(fill, gradient)
        if moving and 0 < target < 1:
            center = ((time.monotonic() % 2.8) / 2.8) * (rect.width() + 160) - 80
            glow = QLinearGradient(center - 80, 0, center + 80, 0)
            glow.setColorAt(0, QColor(255, 255, 255, 0))
            glow.setColorAt(0.5, QColor(255, 255, 255, 45))
            glow.setColorAt(1, QColor(255, 255, 255, 0))
            painter.fillRect(fill, glow)
        painter.end()


class AdaptiveTable(QTableWidget):
    """A compact, bounded table that leaves column resizing in the user's hands.

    Qt's ``Stretch`` mode is attractive for dashboards, but it assigns every
    spare pixel to one column.  This helper measures the header and at most a
    small sample of existing rows, then shares spare width among the columns
    which the user has not resized.  It deliberately does not respond to item
    changes: callers request one fit after a batch of snapshot changes rather
    than remeasuring hundreds of individual cell mutations. A manual header
    drag is never undone by this automatic pass.
    """

    _CELL_INSET = 24

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._adaptive_minimums = []
        self._adaptive_weights = []
        self._adaptive_sample_rows = 32
        self._adaptive_text_cap = 420
        self._adaptive_user_columns = set()
        self._adaptive_programmatic_resize = False
        self._adaptive_last_viewport_width = -1
        self._adaptive_height_rows = None
        self._adaptive_height_empty_rows = 1
        self._adaptive_height_min_rows = 0
        self._adaptive_layout_timer = QTimer(self)
        self._adaptive_layout_timer.setSingleShot(True)
        self._adaptive_layout_timer.timeout.connect(self._apply_adaptive_layout)
        header = self.horizontalHeader()
        header.sectionResized.connect(self._remember_user_resize)

    def configure_adaptive_columns(
        self,
        minimums,
        weights=None,
        *,
        sample_rows=32,
        text_cap=420,
    ):
        """Configure lower bounds and proportional slack for each column.

        ``minimums`` constrains readable short fields; long values are capped
        while measuring so their full text remains available through the
        normal horizontal scrollbar, tooltip, and selection/copy behavior.
        """

        count = self.columnCount()
        values = list(minimums or ())
        self._adaptive_minimums = [
            max(40, int(values[column] if column < len(values) else 80))
            for column in range(count)
        ]
        requested_weights = list(weights or ())
        self._adaptive_weights = [
            max(0.05, float(
                requested_weights[column]
                if column < len(requested_weights) else 1.0
            ))
            for column in range(count)
        ]
        self._adaptive_sample_rows = max(1, min(32, int(sample_rows)))
        self._adaptive_text_cap = max(80, int(text_cap))
        header = self.horizontalHeader()
        for column in range(count):
            header.setSectionResizeMode(column, INTERACTIVE)
        self.request_adaptive_layout()

    def fit_rows_to_content(self, *, max_rows, min_rows=0, empty_rows=1):
        """Show a small table at its content height; larger data scrolls."""

        self._adaptive_height_rows = max(1, int(max_rows))
        self._adaptive_height_min_rows = max(0, int(min_rows))
        self._adaptive_height_empty_rows = max(1, int(empty_rows))
        self.request_adaptive_layout()

    def request_adaptive_layout(self):
        """Coalesce first-paint, resize, theme or completed snapshot updates."""

        # showEvent will fit the latest content when a hidden page is opened.
        if self.isVisible() and not self._adaptive_layout_timer.isActive():
            self._adaptive_layout_timer.start(0)

    def clear_user_column_widths(self):
        """Opt in to a fresh automatic fit; ordinary refreshes never do this."""

        self._adaptive_user_columns.clear()
        self.request_adaptive_layout()

    def setRowCount(self, rows):
        previous = self.rowCount()
        super().setRowCount(rows)
        if int(rows) != previous:
            self.request_adaptive_layout()

    def insertRow(self, row):
        super().insertRow(row)
        self.request_adaptive_layout()

    def removeRow(self, row):
        super().removeRow(row)
        self.request_adaptive_layout()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        width = self.viewport().width()
        if width != self._adaptive_last_viewport_width:
            self._adaptive_last_viewport_width = width
            self.request_adaptive_layout()

    def showEvent(self, event):
        super().showEvent(event)
        self.request_adaptive_layout()

    def _remember_user_resize(self, column, _old, _new):
        if not self._adaptive_programmatic_resize:
            self._adaptive_user_columns.add(int(column))

    def viewportEvent(self, event):
        # Read the current value on demand.  Persisting an automatic tooltip
        # in the item would leave yesterday's status after the next refresh.
        # Explicit ID/stage tooltips supplied by the monitor still take priority.
        if event.type() == QEvent.Type.ToolTip:
            item = self.itemAt(event.pos())
            if item is not None:
                QToolTip.showText(
                    event.globalPos(), item.toolTip() or item.text(), self.viewport()
                )
                return True
        return super().viewportEvent(event)

    def _measured_width(self, column):
        metrics = self.fontMetrics()
        header_item = self.horizontalHeaderItem(column)
        header_text = header_item.text() if header_item is not None else ""
        width = metrics.horizontalAdvance(header_text) + self._CELL_INSET
        limit = min(self.rowCount(), self._adaptive_sample_rows)
        icon_width = self.iconSize().width() + 8
        for row in range(limit):
            item = self.item(row, column)
            if item is None:
                continue
            text_width = min(
                self._adaptive_text_cap,
                metrics.horizontalAdvance(item.text()),
            )
            width = max(width, text_width + self._CELL_INSET + (
                icon_width if not item.icon().isNull() else 0
            ))
        minimum = (
            self._adaptive_minimums[column]
            if column < len(self._adaptive_minimums) else 80
        )
        return max(minimum, width)

    def _apply_adaptive_layout(self):
        self._fit_content_height()
        if not self._adaptive_minimums or self.columnCount() == 0:
            return
        available = max(0, self.viewport().width())
        if not available:
            return
        auto_columns = [
            column for column in range(self.columnCount())
            if column not in self._adaptive_user_columns
        ]
        if not auto_columns:
            return
        required = {column: self._measured_width(column) for column in auto_columns}
        header = self.horizontalHeader()
        fixed_width = sum(
            header.sectionSize(column)
            for column in range(self.columnCount())
            if column not in auto_columns
        )
        natural = sum(required.values())
        extra = max(0, available - fixed_width - natural)
        total_weight = sum(self._adaptive_weights[column] for column in auto_columns)
        widths = dict(required)
        if extra and total_weight:
            distributed = 0
            for column in auto_columns[:-1]:
                share = int(extra * self._adaptive_weights[column] / total_weight)
                widths[column] += share
                distributed += share
            widths[auto_columns[-1]] += extra - distributed
        self._adaptive_programmatic_resize = True
        try:
            for column in auto_columns:
                header.resizeSection(column, widths[column])
        finally:
            self._adaptive_programmatic_resize = False

    def _fit_content_height(self):
        if self._adaptive_height_rows is None:
            return
        rows = self.rowCount()
        visible = min(self._adaptive_height_rows, max(
            self._adaptive_height_min_rows,
            rows if rows else self._adaptive_height_empty_rows,
        ))
        header_height = max(1, self.horizontalHeader().height())
        row_height = max(1, self.verticalHeader().defaultSectionSize())
        # Reserve the native horizontal bar even when it is not currently
        # visible; long IDs/errors can make it appear after the next snapshot.
        bar_height = self.horizontalScrollBar().sizeHint().height()
        height = header_height + visible * row_height + (self.frameWidth() * 2) + bar_height + 2
        if self.height() != height:
            self.setFixedHeight(height)
