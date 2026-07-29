"""Small Qt5/Qt6 enum facade used by the shared QGIS plugin source."""

from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QFont, QTextCursor
from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QDialogButtonBox,
    QHeaderView,
    QMessageBox,
)


def _enum(owner, scope_name, member_name):
    """Return a scoped Qt6 enum member or its Qt5 flat equivalent."""
    scope = getattr(owner, scope_name, None)
    if scope is not None:
        member = getattr(scope, member_name, None)
        if member is not None:
            return member
    return getattr(owner, member_name)


RIGHT_DOCK_WIDGET_AREA = _enum(Qt, "DockWidgetArea", "RightDockWidgetArea")
WINDOW = _enum(Qt, "WindowType", "Window")
HORIZONTAL = _enum(Qt, "Orientation", "Horizontal")
ALIGN_LEFT = _enum(Qt, "AlignmentFlag", "AlignLeft")
ALIGN_VCENTER = _enum(Qt, "AlignmentFlag", "AlignVCenter")
ALIGN_CENTER = _enum(Qt, "AlignmentFlag", "AlignCenter")
TEXT_SELECTABLE_BY_MOUSE = _enum(
    Qt, "TextInteractionFlag", "TextSelectableByMouse"
)
SCROLLBAR_AS_NEEDED = _enum(Qt, "ScrollBarPolicy", "ScrollBarAsNeeded")
WA_DELETE_ON_CLOSE = _enum(Qt, "WidgetAttribute", "WA_DeleteOnClose")
USER_ROLE = _enum(Qt, "ItemDataRole", "UserRole")
DASH_LINE = _enum(Qt, "PenStyle", "DashLine")
ISO_DATE = _enum(Qt, "DateFormat", "ISODate")
ITEM_IS_ENABLED = _enum(Qt, "ItemFlag", "ItemIsEnabled")
ITEM_IS_USER_CHECKABLE = _enum(Qt, "ItemFlag", "ItemIsUserCheckable")
CHECKED = _enum(Qt, "CheckState", "Checked")
UNCHECKED = _enum(Qt, "CheckState", "Unchecked")

NO_EDIT_TRIGGERS = _enum(
    QAbstractItemView, "EditTrigger", "NoEditTriggers"
)
SELECT_ROWS = _enum(
    QAbstractItemView, "SelectionBehavior", "SelectRows"
)
SINGLE_SELECTION = _enum(
    QAbstractItemView, "SelectionMode", "SingleSelection"
)
EXTENDED_SELECTION = _enum(
    QAbstractItemView, "SelectionMode", "ExtendedSelection"
)
ENSURE_VISIBLE = _enum(
    QAbstractItemView, "ScrollHint", "EnsureVisible"
)

RESIZE_TO_CONTENTS = _enum(
    QHeaderView, "ResizeMode", "ResizeToContents"
)
STRETCH = _enum(QHeaderView, "ResizeMode", "Stretch")
INTERACTIVE = _enum(QHeaderView, "ResizeMode", "Interactive")

YES = _enum(QMessageBox, "StandardButton", "Yes")
NO = _enum(QMessageBox, "StandardButton", "No")
APPLY = _enum(QDialogButtonBox, "StandardButton", "Apply")
CANCEL = _enum(QDialogButtonBox, "StandardButton", "Cancel")

TEXT_CURSOR_END = _enum(QTextCursor, "MoveOperation", "End")
FONT_BOLD = _enum(QFont, "Weight", "Bold")
