import sys

from qgis.PyQt.QtCore import QObject
from qgis.PyQt.QtWidgets import QAction, QToolBar
from qgis.PyQt.QtGui import QGuiApplication, QIcon
from qgis.core import QgsApplication

from .gui.main_dock import LabelingDockWidget
from .qt6_api import RIGHT_DOCK_WIDGET_AREA


class LabelingTool(QObject):

    def __init__(self, iface):
        super().__init__()
        self.iface = iface
        self.canvas = iface.mapCanvas()
        self.dock_widget = None
        self.toolbar = None
        self.action = None

    def initGui(self):
        self._require_supported_qpa()
        icon = QIcon(":/images/themes/default/mAction.svg")
        self.action = QAction(icon, "地物标注工具", self.iface.mainWindow())
        self.action.setObjectName("labelingAction")
        self.action.setWhatsThis("地物标注工具")
        self.toolbar = self.iface.addToolBar("地物标注工具")
        self.toolbar.setObjectName("labelingToolBar")
        self.toolbar.addAction(self.action)

        self.dock_widget = LabelingDockWidget(self.iface.mainWindow(), iface=self.iface)
        self.iface.addDockWidget(
            RIGHT_DOCK_WIDGET_AREA,
            self.dock_widget,
        )

        self.iface.addPluginToMenu("地物标注工具", self.action)

        self.action.triggered.connect(self.show_dock)

    @staticmethod
    def _require_supported_qpa():
        """Reject non-Wayland Linux sessions before creating plugin widgets."""

        if not sys.platform.startswith("linux"):
            return
        qpa = str(QGuiApplication.platformName() or "").strip().lower()
        if qpa != "wayland":
            raise RuntimeError(
                "labeling_tool on Ubuntu requires the native Qt6 Wayland "
                f"platform plugin; current QPA is {qpa or 'unknown'}"
            )

    def unload(self):
        if self.dock_widget:
            self.dock_widget.cleanup()
            self.iface.removeDockWidget(self.dock_widget)
            self.dock_widget.deleteLater()
            self.dock_widget = None
        if self.toolbar:
            del self.toolbar
            self.toolbar = None
        if self.action:
            self.iface.removePluginMenu("地物标注工具", self.action)
            self.action.deleteLater()
            self.action = None

    def show_dock(self):
        self.dock_widget.show()
        self.dock_widget.raise_()
