# ruff: noqa: E402
"""Exercise real dialog layouts at high-DPI logical viewport sizes."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT, OUTPUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(ROOT / "src"))
try:
    from qgis.core import QgsApplication
    from qgis.PyQt.QtCore import QPoint, QRect
    from qgis.PyQt.QtWidgets import QDialogButtonBox, QScrollArea
except ModuleNotFoundError:
    raise SystemExit(77)

from labeling_tool.main.inference_config_dialog import InferenceConfigDialog
from labeling_tool.monitor.inference_monitor import InferenceMonitorDialog
from labeling_tool.qgis_support.qt6_api import APPLY, CANCEL


def inside(widget, container):
    rectangle = QRect(widget.mapTo(container, QPoint(0, 0)), widget.size())
    assert container.rect().contains(rectangle), (
        widget.objectName(),
        rectangle,
        container.rect(),
    )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    app = QgsApplication([], False)
    app.initQgis()
    app_style = app.styleSheet()
    # 1920x1080 at 200% scaling provides 960x540 logical pixels before chrome.
    screen = SimpleNamespace(availableGeometry=lambda: QRect(0, 0, 960, 540))
    dialogs = []
    try:
        for cls in (InferenceConfigDialog, InferenceMonitorDialog):
            with patch.object(cls, "screen", return_value=screen):
                dialog = cls()
            dialogs.append(dialog)
            dialog.show()
            app.processEvents()
            assert dialog.width() <= 960 and dialog.height() <= 500
            for width, height in ((960, 500), (640, 360)):
                dialog.resize(width, height)
                app.processEvents()
                assert (dialog.width(), dialog.height()) == (width, height)
                if isinstance(dialog, InferenceConfigDialog):
                    buttons = dialog.findChild(QDialogButtonBox)
                    scroll = dialog.findChild(QScrollArea, "inferenceConfigScrollArea")
                    scroll.verticalScrollBar().setValue(
                        scroll.verticalScrollBar().maximum()
                    )
                    app.processEvents()
                    for action in (APPLY, CANCEL):
                        inside(buttons.button(action), dialog)
                    assert scroll.viewport().height() > 0
                else:
                    for theme in ("dark", "light"):
                        dialog._apply_theme(theme, persist=False)
                        app.processEvents()
                        inside(dialog._stop, dialog)
                        dialog._body_scroll.ensureWidgetVisible(
                            dialog._progress_panel, 0, 0
                        )
                        app.processEvents()
                        inside(dialog._progress_panel, dialog._body_scroll.viewport())
                        assert dialog._pages.count() == 4
                    dialog._body_scroll.verticalScrollBar().setValue(0)
                assert app.styleSheet() == app_style
                app.processEvents()
                assert dialog.grab().save(
                    str(OUTPUT / f"{cls.__name__}-{width}x{height}.png")
                )
    finally:
        for dialog in dialogs:
            if isinstance(dialog, InferenceMonitorDialog):
                dialog.shutdown()
            dialog.close()
        app.processEvents()
    print("small_dialog_viewports: passed")


if __name__ == "__main__":
    main()
