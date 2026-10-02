"""Size native dialogs against the screen's logical, usable area."""

from __future__ import annotations

from qgis.PyQt.QtWidgets import QDialog


def fit_dialog_to_screen(
    dialog: QDialog,
    *,
    preferred_size: tuple[int, int],
    minimum_size: tuple[int, int] = (640, 360),
) -> None:
    """Leave room for window decorations without undoing high-DPI scaling."""
    width, height = preferred_size
    minimum_width, minimum_height = minimum_size
    parent = dialog.parentWidget()
    screen = parent.screen() if parent is not None else dialog.screen()
    if screen is not None:
        available = screen.availableGeometry()
        usable_width = max(1, available.width() - 32)
        usable_height = max(1, available.height() - 48)
        width, height = min(width, usable_width), min(height, usable_height)
        minimum_width = min(minimum_width, usable_width)
        minimum_height = min(minimum_height, usable_height)
    dialog.setMinimumSize(minimum_width, minimum_height)
    dialog.resize(width, height)
