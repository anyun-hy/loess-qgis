"""Own the QGIS map tools used by a single manual editing session.

The dialog owns edit data and layer transactions. This object owns only map tool
references, canvas switching, and callbacks that must outlive a capture event.
All methods run on the QGIS GUI thread.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from qgis.core import Qgis, QgsFeature, QgsPointXY, QgsVectorLayer
from qgis.gui import (
    QgsAdvancedDigitizingDockWidget,
    QgsMapCanvas,
    QgsMapTool,
    QgsMapToolCapture,
    QgsMapToolDigitizeFeature,
    QgsMapToolEmitPoint,
)
from qgis.PyQt.QtCore import QObject, Qt, QTimer, pyqtSignal


class ManualEditTools(QObject):
    """Control picker/capture tools without retaining dialog or task state."""

    map_clicked = pyqtSignal(object, object)
    feature_captured = pyqtSignal(object)
    capture_cancelled = pyqtSignal()
    interrupted = pyqtSignal()
    restart_requested = pyqtSignal()

    def __init__(
        self,
        canvas: QgsMapCanvas,
        cad_dock: Callable[[], QgsAdvancedDigitizingDockWidget],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._canvas = canvas
        self._cad_dock = cad_dock
        self._picker: QgsMapToolEmitPoint | None = None
        self._capture: QgsMapToolDigitizeFeature | None = None
        self._previous: QgsMapTool | None = None
        self._expected: QgsMapTool | None = None
        self._switching = False
        self._active = False
        self._generation = 0
        self._connected = False
        self._retired: list[QgsMapTool] = []
        self._transition_action: Literal["restart", "restore"] | None = None
        self._transition_generation: int | None = None
        self._transition_timer = QTimer(self)
        self._transition_timer.setSingleShot(True)
        self._transition_timer.timeout.connect(self._run_transition)
        self._retire_timer = QTimer(self)
        self._retire_timer.setSingleShot(True)
        self._retire_timer.timeout.connect(self._release_retired)

    def connect(self) -> None:
        """Resume map tool change observation after cleanup."""
        if self._connected:
            return
        signal = getattr(self._canvas, "mapToolSet", None)
        if signal is not None:
            try:
                signal.connect(self._map_tool_changed)
            except (TypeError, RuntimeError):
                return
            self._connected = True

    def begin_session(self) -> None:
        """Remember the user's tool before the first manual tool is activated."""
        self.end_session()
        self.connect()
        self._generation += 1
        self._previous = self._canvas.mapTool()
        self._active = True

    def start_picker(self) -> None:
        """Activate a new point picker for the current session."""
        if not self._active:
            return
        self.stop_picker()
        tool = QgsMapToolEmitPoint(self._canvas)
        tool.canvasClicked.connect(self._map_clicked)
        self._picker = tool
        self._set_map_tool(tool)

    def start_capture(self, layer: QgsVectorLayer) -> None:
        """Capture on an editable layer; raise RuntimeError without PolyBezier."""
        if not self._active:
            return
        self.stop_capture()
        tool = QgsMapToolDigitizeFeature(
            self._canvas,
            self._cad_dock(),
            QgsMapToolCapture.CaptureMode.CapturePolygon,
        )
        tool.setLayer(layer)
        tool.setCheckGeometryType(True)
        if not tool.supportsTechnique(Qgis.CaptureTechnique.PolyBezier):
            raise RuntimeError("当前 QGIS 不支持 PolyBezier 捕获")
        tool.setCurrentCaptureTechnique(Qgis.CaptureTechnique.PolyBezier)
        tool.digitizingCompleted.connect(self._feature_captured)
        tool.digitizingCanceled.connect(self._capture_cancelled)
        self._capture = tool
        self._set_map_tool(tool)

    def stop_picker(self) -> None:
        """Disconnect picker events, retaining it while canvas still uses it."""
        tool = self._picker
        if tool is None:
            return
        try:
            tool.canvasClicked.disconnect(self._map_clicked)
        except (TypeError, RuntimeError):
            pass
        self._picker = None
        self._retire(tool)

    def stop_capture(self) -> None:
        """Disconnect capture events and stop an in-progress drawing."""
        self._cancel_transition()
        tool = self._capture
        if tool is None:
            return
        for signal, slot in (
            (tool.digitizingCompleted, self._feature_captured),
            (tool.digitizingCanceled, self._capture_cancelled),
        ):
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        try:
            tool.stopCapturing()
        except RuntimeError:
            pass
        self._capture = None
        self._retire(tool)

    def schedule_transition(self, action: Literal["restart", "restore"]) -> None:
        """Defer capture replacement until its digitizing callback has returned."""
        if not self._active or action not in ("restart", "restore"):
            return
        self._transition_action = action
        self._transition_generation = self._generation
        self._transition_timer.start(0)

    def restore_previous(self) -> None:
        """Leave any active manual tool, even when no previous tool exists."""
        current = self._canvas.mapTool()
        if current is self._previous:
            self._expected = None
            return
        if self._previous is not None:
            try:
                self._set_map_tool(self._previous)
                self._expected = None
                return
            except RuntimeError:
                self._previous = None
        if current is not None and self._owns(current):
            self._switching = True
            try:
                self._canvas.unsetMapTool(current)
            finally:
                self._switching = False
        self._expected = None
        self._release_retired()

    def end_session(self) -> None:
        """Invalidate queued restarts, restore the user's tool, and drop tools."""
        self._generation += 1
        self._active = False
        self._cancel_transition()
        self.stop_picker()
        self.stop_capture()
        self.restore_previous()
        self._previous = None
        self._release_retired()

    def cleanup(self) -> None:
        """Release a session and its canvas signal; safe to repeat."""
        self.end_session()
        if self._connected:
            signal = getattr(self._canvas, "mapToolSet", None)
            if signal is not None:
                try:
                    signal.disconnect(self._map_tool_changed)
                except (TypeError, RuntimeError):
                    pass
            self._connected = False
        self._retire_timer.stop()
        self._release_retired()

    def _set_map_tool(self, tool: QgsMapTool) -> None:
        self._expected = tool
        self._switching = True
        try:
            self._canvas.setMapTool(tool)
        finally:
            self._switching = False
        self._release_retired()

    def _map_tool_changed(self, *_args: object) -> None:
        if self._retired:
            self._retire_timer.start(0)
        if not self._active or self._switching or self._expected is None:
            return
        if self._canvas.mapTool() is self._expected:
            return
        self._expected = None
        self._cancel_transition()
        self.interrupted.emit()

    def _map_clicked(self, map_point: QgsPointXY, button: Qt.MouseButton) -> None:
        if self._active and self._picker is not None:
            self.map_clicked.emit(map_point, button)

    def _feature_captured(self, feature: QgsFeature) -> None:
        if self._active and self._capture is not None:
            self.feature_captured.emit(feature)

    def _capture_cancelled(self) -> None:
        if self._active and self._capture is not None:
            self.capture_cancelled.emit()

    def _run_transition(self) -> None:
        action = self._transition_action
        generation = self._transition_generation
        self._transition_action = None
        self._transition_generation = None
        if not self._active or generation != self._generation:
            return
        if action not in ("restart", "restore"):
            return
        self.stop_capture()
        self.restore_previous()
        if action == "restart" and self._active and generation == self._generation:
            self.restart_requested.emit()

    def _cancel_transition(self) -> None:
        self._transition_timer.stop()
        self._transition_action = None
        self._transition_generation = None

    def _owns(self, tool: QgsMapTool) -> bool:
        return (
            tool is self._picker
            or tool is self._capture
            or any(tool is retired for retired in self._retired)
        )

    def _retire(self, tool: QgsMapTool) -> None:
        self._retired.append(tool)
        self._retire_timer.start(0)

    def _release_retired(self) -> None:
        current = self._canvas.mapTool()
        self._retired = [tool for tool in self._retired if tool is current]
