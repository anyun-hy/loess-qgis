from qgis.PyQt.QtCore import QObject, Qt
from qgis.PyQt.QtWidgets import QAction, QToolBar
from qgis.PyQt.QtGui import QIcon
from qgis.core import QgsApplication

from .gui.main_dock import LabelingDockWidget


class LabelingTool(QObject):

    def __init__(self, iface):
        super().__init__()
        self.iface = iface
        self.canvas = iface.mapCanvas()
        self.dock_widget = None
        self.toolbar = None
        self.action = None

    def initGui(self):
        icon = QIcon(":/images/themes/default/mAction.svg")
        self.action = QAction(icon, "标注工具", self.iface.mainWindow())
        self.action.setObjectName("labelingAction")
        self.action.setWhatsThis("半自动标注工具")
        self.toolbar = self.iface.addToolBar("标注工具")
        self.toolbar.setObjectName("labelingToolBar")
        self.toolbar.addAction(self.action)

        self.dock_widget = LabelingDockWidget(self.iface.mainWindow(), iface=self.iface)
        self.iface.addDockWidget(
            Qt.RightDockWidgetArea,
            self.dock_widget,
        )

        self.iface.addPluginToMenu("标注工具", self.action)

        self.action.triggered.connect(self.show_dock)

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
            self.iface.removePluginMenu("标注工具", self.action)
            self.action.deleteLater()
            self.action = None

    def show_dock(self):
        self.dock_widget.show()
        self.dock_widget.raise_()
