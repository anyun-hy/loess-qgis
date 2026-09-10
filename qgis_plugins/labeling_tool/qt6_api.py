"""Shared Qt6 enum API for the QGIS 4.2 plugin."""

from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QFont, QTextBlockFormat, QTextCursor
from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QDialogButtonBox,
    QHeaderView,
    QMessageBox,
    QStyle,
)

RIGHT_DOCK_WIDGET_AREA = Qt.DockWidgetArea.RightDockWidgetArea
WINDOW = Qt.WindowType.Window
NON_MODAL = Qt.WindowModality.NonModal
HORIZONTAL = Qt.Orientation.Horizontal
VERTICAL = Qt.Orientation.Vertical
ALIGN_LEFT = Qt.AlignmentFlag.AlignLeft
ALIGN_RIGHT = Qt.AlignmentFlag.AlignRight
ALIGN_TOP = Qt.AlignmentFlag.AlignTop
ALIGN_VCENTER = Qt.AlignmentFlag.AlignVCenter
ALIGN_CENTER = Qt.AlignmentFlag.AlignCenter
RICH_TEXT = Qt.TextFormat.RichText
TRANSPARENT = Qt.GlobalColor.transparent
TEXT_SELECTABLE_BY_MOUSE = Qt.TextInteractionFlag.TextSelectableByMouse
SCROLLBAR_AS_NEEDED = Qt.ScrollBarPolicy.ScrollBarAsNeeded
MENU_SCROLLER_HEIGHT = QStyle.PixelMetric.PM_MenuScrollerHeight
WA_DELETE_ON_CLOSE = Qt.WidgetAttribute.WA_DeleteOnClose
USER_ROLE = Qt.ItemDataRole.UserRole
DASH_LINE = Qt.PenStyle.DashLine
NO_PEN = Qt.PenStyle.NoPen
ISO_DATE = Qt.DateFormat.ISODate
ITEM_IS_ENABLED = Qt.ItemFlag.ItemIsEnabled
ITEM_IS_USER_CHECKABLE = Qt.ItemFlag.ItemIsUserCheckable
CHECKED = Qt.CheckState.Checked
UNCHECKED = Qt.CheckState.Unchecked

NO_EDIT_TRIGGERS = QAbstractItemView.EditTrigger.NoEditTriggers
SELECT_ROWS = QAbstractItemView.SelectionBehavior.SelectRows
SINGLE_SELECTION = QAbstractItemView.SelectionMode.SingleSelection
EXTENDED_SELECTION = QAbstractItemView.SelectionMode.ExtendedSelection
ENSURE_VISIBLE = QAbstractItemView.ScrollHint.EnsureVisible

RESIZE_TO_CONTENTS = QHeaderView.ResizeMode.ResizeToContents
STRETCH = QHeaderView.ResizeMode.Stretch
INTERACTIVE = QHeaderView.ResizeMode.Interactive

YES = QMessageBox.StandardButton.Yes
NO = QMessageBox.StandardButton.No
CLOSE = QMessageBox.StandardButton.Close
CRITICAL = QMessageBox.Icon.Critical
INFORMATION = QMessageBox.Icon.Information
WARNING = QMessageBox.Icon.Warning
APPLY = QDialogButtonBox.StandardButton.Apply
CANCEL = QDialogButtonBox.StandardButton.Cancel

TEXT_CURSOR_END = QTextCursor.MoveOperation.End
TEXT_CURSOR_DOCUMENT = QTextCursor.SelectionType.Document
# QTextBlockFormat.setLineHeight expects int, not PyQt6's scoped enum.
TEXT_LINE_PROPORTIONAL = QTextBlockFormat.LineHeightTypes.ProportionalHeight.value
FONT_BOLD = QFont.Weight.Bold
