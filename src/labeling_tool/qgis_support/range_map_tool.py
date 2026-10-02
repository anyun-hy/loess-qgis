from qgis.core import Qgis, QgsPointXY, QgsRectangle
from qgis.gui import QgsMapTool, QgsRubberBand
from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtGui import QColor


class RectangleMapTool(QgsMapTool):
    """Collect a dragged rectangle without owning range selection state."""

    rect_finished = pyqtSignal(object)

    def __init__(self, canvas):
        super().__init__(canvas)
        self._rubber_band = None
        if canvas is not None:
            self._rubber_band = QgsRubberBand(canvas, Qgis.GeometryType.Polygon)
            self._rubber_band.setStrokeColor(QColor(255, 50, 50))
            self._rubber_band.setFillColor(QColor(255, 50, 50, 40))
            self._rubber_band.setWidth(2)
        self._start_point = None

    def reset(self) -> None:
        self._start_point = None
        if self._rubber_band:
            self._rubber_band.reset(Qgis.GeometryType.Polygon)

    def canvasPressEvent(self, event) -> None:
        self._start_point = self.toMapCoordinates(event.pos())
        if self._rubber_band:
            self._rubber_band.reset(Qgis.GeometryType.Polygon)

    def canvasMoveEvent(self, event) -> None:
        if self._start_point is None or not self._rubber_band:
            return
        end = self.toMapCoordinates(event.pos())
        self._rubber_band.reset(Qgis.GeometryType.Polygon)
        points = [
            self._start_point,
            QgsPointXY(end.x(), self._start_point.y()),
            end,
            QgsPointXY(self._start_point.x(), end.y()),
            self._start_point,
        ]
        for point in points:
            self._rubber_band.addPoint(point)

    def canvasReleaseEvent(self, event) -> None:
        if self._start_point is None:
            return
        end = self.toMapCoordinates(event.pos())
        rect = QgsRectangle(
            min(self._start_point.x(), end.x()),
            min(self._start_point.y(), end.y()),
            max(self._start_point.x(), end.x()),
            max(self._start_point.y(), end.y()),
        )
        self._start_point = None
        self.rect_finished.emit(rect)

    def deactivate(self) -> None:
        self.reset()
        super().deactivate()
