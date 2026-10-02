"""Own SAM3 point picking and map preview resources."""

from __future__ import annotations

from qgis.core import Qgis, QgsGeometry, QgsVectorLayer
from qgis.gui import QgsMapCanvas, QgsMapTool, QgsMapToolEmitPoint, QgsRubberBand
from qgis.PyQt.QtCore import QObject, pyqtSignal
from qgis.PyQt.QtGui import QColor

from labeling_tool.qgis_support.qt6_api import DASH_LINE


class SamMapPreview(QObject):
    """Manage SAM3 picker and rubber bands on the QGIS GUI thread."""

    point_clicked = pyqtSignal(object, object)

    def __init__(self, canvas: QgsMapCanvas, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._canvas = canvas
        self._picker: QgsMapToolEmitPoint | None = None
        self._previous: QgsMapTool | None = None
        self._current: QgsRubberBand | None = None
        self._candidate: QgsRubberBand | None = None

    def start_pick(self) -> None:
        """Remember the current map tool and activate a fresh point picker."""
        self.restore_map_tool()
        self._previous = self._canvas.mapTool()
        picker = QgsMapToolEmitPoint(self._canvas)
        picker.setParent(self)
        picker.canvasClicked.connect(self._emit_point_clicked)
        self._picker = picker
        self._canvas.setMapTool(picker)

    def restore_map_tool(self) -> None:
        """Restore only when the canvas still uses this object's picker."""
        picker = self._picker
        if picker is None:
            return
        try:
            picker.canvasClicked.disconnect(self._emit_point_clicked)
        except (TypeError, RuntimeError):
            pass
        if self._canvas.mapTool() is picker:
            if self._previous is not None:
                try:
                    self._canvas.setMapTool(self._previous)
                except RuntimeError:
                    self._canvas.unsetMapTool(picker)
            else:
                self._canvas.unsetMapTool(picker)
        self._picker = None
        self._previous = None
        picker.deleteLater()

    def show_current(self, geometry: QgsGeometry, layer: QgsVectorLayer) -> None:
        """Replace both previews with the current working geometry."""
        self.clear()
        self._current = QgsRubberBand(self._canvas, Qgis.GeometryType.Polygon)
        self._current.setStrokeColor(QColor("#ffd400"))
        self._current.setFillColor(QColor(255, 212, 0, 30))
        self._current.setWidth(2)
        self._current.setToGeometry(geometry, layer)

    def show_candidate(self, geometry: QgsGeometry, layer: QgsVectorLayer) -> None:
        """Replace the candidate preview while retaining the current outline."""
        self.clear_candidate()
        self._candidate = QgsRubberBand(self._canvas, Qgis.GeometryType.Polygon)
        self._candidate.setStrokeColor(QColor("#00d7d7"))
        self._candidate.setFillColor(QColor(0, 215, 215, 35))
        self._candidate.setWidth(2)
        self._candidate.setLineStyle(DASH_LINE)
        self._candidate.setToGeometry(geometry, layer)

    def clear_candidate(self) -> None:
        """Remove only the candidate preview."""
        self._candidate = self._remove_band(self._candidate)

    def clear(self) -> None:
        """Remove current and candidate previews without changing map tools."""
        self.clear_candidate()
        self._current = self._remove_band(self._current)

    def cleanup(self) -> None:
        """Restore the picker and remove all previews; safe to repeat."""
        self.restore_map_tool()
        self.clear()

    def _emit_point_clicked(self, point: object, button: object) -> None:
        if self._picker is not None:
            self.point_clicked.emit(point, button)

    def _remove_band(self, band: QgsRubberBand | None) -> QgsRubberBand | None:
        if band is not None:
            band.reset(Qgis.GeometryType.Polygon)
            self._canvas.scene().removeItem(band)
        return None
