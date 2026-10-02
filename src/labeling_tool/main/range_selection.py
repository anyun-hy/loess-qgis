from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QObject, pyqtSignal

from labeling_tool.qgis_support.range_map_tool import RectangleMapTool

RangeMode = Literal["当前视图", "手绘矩形", "加载矢量范围"]
VIEW_MODE: RangeMode = "当前视图"
RECTANGLE_MODE: RangeMode = "手绘矩形"
VECTOR_MODE: RangeMode = "加载矢量范围"


@dataclass(frozen=True)
class RawRangeSelection:
    mode: RangeMode
    extent: QgsRectangle | None
    crs: QgsCoordinateReferenceSystem | None
    vector_layer: QgsVectorLayer | None = None


class RangeSelectionController(QObject):
    """Own captured extents and the rectangle map-tool lifecycle."""

    rectangle_finished = pyqtSignal(object)

    def __init__(self, iface: Any | None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._iface = iface
        self._view: RawRangeSelection | None = None
        self._rectangle: RawRangeSelection | None = None
        self._previous_map_tool: Any | None = None
        self._map_tool = RectangleMapTool(iface.mapCanvas()) if iface else None
        if self._map_tool is not None:
            self._map_tool.rect_finished.connect(self._accept_rectangle)

    @property
    def is_drawing(self) -> bool:
        if self._iface is None or self._map_tool is None:
            return False
        return self._iface.mapCanvas().mapTool() == self._map_tool

    def capture_current_view(self) -> RawRangeSelection | None:
        if self._iface is None:
            return None
        canvas = self._iface.mapCanvas()
        self.restore_map_tool()
        self._view = RawRangeSelection(
            VIEW_MODE,
            QgsRectangle(canvas.extent()),
            canvas.mapSettings().destinationCrs(),
        )
        return _copy_selection(self._view)

    def begin_rectangle(self) -> bool:
        if self._iface is None or self._map_tool is None:
            return False
        canvas = self._iface.mapCanvas()
        current_tool = canvas.mapTool()
        if current_tool != self._map_tool:
            self._previous_map_tool = current_tool
        self._rectangle = None
        self._map_tool.reset()
        canvas.setMapTool(self._map_tool)
        return True

    def selected(
        self,
        mode: RangeMode,
        vector_layer: QgsVectorLayer | None = None,
    ) -> RawRangeSelection:
        if mode == VIEW_MODE:
            return (
                _copy_selection(self._view)
                if self._view is not None
                else RawRangeSelection(VIEW_MODE, None, None)
            )
        if mode == RECTANGLE_MODE:
            return (
                _copy_selection(self._rectangle)
                if self._rectangle is not None
                else RawRangeSelection(RECTANGLE_MODE, None, None)
            )
        layer = validate_vector_layer(vector_layer)
        return RawRangeSelection(
            VECTOR_MODE,
            QgsRectangle(layer.extent()),
            layer.crs(),
            layer,
        )

    def ensure_current_view(self) -> RawRangeSelection:
        if self._view is None:
            selection = self.capture_current_view()
            if selection is None:
                return RawRangeSelection(VIEW_MODE, None, None)
        return _copy_selection(self._view)

    def restore_map_tool(self) -> None:
        if self._iface is None or self._map_tool is None:
            return
        canvas = self._iface.mapCanvas()
        if canvas.mapTool() != self._map_tool:
            self._previous_map_tool = None
            return

        restored = False
        previous_tool = self._previous_map_tool
        if previous_tool and previous_tool != self._map_tool:
            try:
                canvas.setMapTool(previous_tool)
                restored = True
            except RuntimeError:
                restored = False
        if not restored:
            try:
                pan_action = self._iface.actionPan()
                if pan_action:
                    pan_action.trigger()
                    restored = canvas.mapTool() != self._map_tool
            except Exception:
                restored = False
        if not restored and canvas.mapTool() == self._map_tool:
            canvas.unsetMapTool(self._map_tool)
        self._map_tool.reset()
        self._previous_map_tool = None

    def close(self) -> None:
        self.restore_map_tool()
        if self._map_tool is not None:
            try:
                self._map_tool.rect_finished.disconnect(self._accept_rectangle)
            except (TypeError, RuntimeError):
                pass
            self._map_tool.deactivate()
            self._map_tool = None
        self._view = None
        self._rectangle = None
        self._previous_map_tool = None

    def _accept_rectangle(self, rect: QgsRectangle) -> None:
        if self._iface is None:
            return
        self._rectangle = RawRangeSelection(
            RECTANGLE_MODE,
            QgsRectangle(rect),
            self._iface.mapCanvas().mapSettings().destinationCrs(),
        )
        self.rectangle_finished.emit(_copy_selection(self._rectangle))


def _copy_selection(selection: RawRangeSelection) -> RawRangeSelection:
    return RawRangeSelection(
        selection.mode,
        QgsRectangle(selection.extent) if selection.extent is not None else None,
        (
            QgsCoordinateReferenceSystem(selection.crs)
            if selection.crs is not None
            else None
        ),
        selection.vector_layer,
    )


def validate_raster_layer(layer: Any) -> QgsRasterLayer:
    if not layer or not isinstance(layer, QgsRasterLayer) or not layer.isValid():
        raise ValueError("请选择有效的本地影像层")
    provider = layer.providerType()
    if provider != "gdal":
        raise ValueError(
            f"当前影像层 provider={provider}，不能作为推理输入。"
            "请选择本地 GeoTIFF/栅格文件，不要选在线底图或 WMTS/XYZ 图层。"
        )
    if not layer.crs().isValid():
        raise ValueError(
            f"影像层「{layer.name()}」没有有效 CRS。请先在 QGIS 中设置/定义影像 CRS。"
        )
    source_path = layer.source().split("|", 1)[0]
    if source_path and not os.path.exists(source_path):
        raise ValueError(f"影像文件不存在: {source_path}")
    return layer


def validate_vector_layer(layer: Any) -> QgsVectorLayer:
    if not layer or not isinstance(layer, QgsVectorLayer) or not layer.isValid():
        raise ValueError("请选择有效的已加载矢量面图层")
    if layer.geometryType() != Qgis.GeometryType.Polygon:
        raise ValueError("矢量范围必须是面图层")
    if layer.featureCount() < 1:
        raise ValueError(f"矢量图层「{layer.name()}」没有面要素")
    if not layer.crs().isValid():
        raise ValueError(f"矢量图层「{layer.name()}」没有有效 CRS")
    return layer


def resolve_raster_extent(
    selection: RawRangeSelection,
    raster: QgsRasterLayer,
) -> QgsRectangle:
    if not is_valid_extent(selection.extent):
        if selection.mode == RECTANGLE_MODE:
            raise ValueError("请先选择「手绘矩形」，并在地图上拖拽绘制范围")
        raise ValueError(f"{selection.mode}范围无效，请重新选择范围")
    extent = transform_extent(selection.extent, selection.crs, raster.crs())
    clipped = intersect_extents(extent, raster.extent())
    if not is_valid_extent(clipped):
        raise ValueError(
            f"{selection.mode}范围与影像层「{raster.name()}」没有重叠，"
            "请移动到影像覆盖区或重新绘制范围。"
        )
    return clipped


def range_selection_metadata(
    vector_layer: QgsVectorLayer | None,
    grid_tile_count: int,
    selected_tile_count: int,
) -> dict[str, Any]:
    if vector_layer is None:
        return {
            "mode": "extent",
            "selected_tile_count": selected_tile_count,
            "excluded_tile_count": 0,
            "clip_outputs": True,
        }
    layer = validate_vector_layer(vector_layer)
    return {
        "mode": "vector_tile_intersection",
        "vector_layer_id": layer.id(),
        "vector_layer_name": layer.name(),
        "vector_source": layer.source(),
        "vector_crs": layer.crs().authid(),
        "selected_tile_count": selected_tile_count,
        "excluded_tile_count": grid_tile_count - selected_tile_count,
        "clip_outputs": True,
    }


def transform_extent(
    extent: QgsRectangle,
    source_crs: QgsCoordinateReferenceSystem | None,
    target_crs: QgsCoordinateReferenceSystem | None,
) -> QgsRectangle:
    if not source_crs or not source_crs.isValid():
        raise ValueError("地图当前 CRS 无效，无法把范围转换到影像 CRS")
    if not target_crs or not target_crs.isValid():
        raise ValueError("影像 CRS 无效，无法确定切片范围")
    if source_crs == target_crs:
        return QgsRectangle(extent)
    try:
        transform = QgsCoordinateTransform(
            source_crs, target_crs, QgsProject.instance()
        )
        return transform.transformBoundingBox(extent)
    except Exception as exc:
        raise ValueError(
            f"范围 CRS 转换失败: {source_crs.authid()} → {target_crs.authid()} ({exc})"
        ) from exc


def intersect_extents(
    first: QgsRectangle,
    second: QgsRectangle,
) -> QgsRectangle | None:
    xmin = max(first.xMinimum(), second.xMinimum())
    xmax = min(first.xMaximum(), second.xMaximum())
    ymin = max(first.yMinimum(), second.yMinimum())
    ymax = min(first.yMaximum(), second.yMaximum())
    if xmax <= xmin or ymax <= ymin:
        return None
    return QgsRectangle(xmin, ymin, xmax, ymax)


def is_valid_extent(extent: QgsRectangle | None) -> bool:
    return bool(
        extent is not None
        and extent.xMaximum() > extent.xMinimum()
        and extent.yMaximum() > extent.yMinimum()
    )


def format_extent(extent: QgsRectangle) -> str:
    return (
        f"{extent.xMinimum():.6f}, {extent.yMinimum():.6f}, "
        f"{extent.xMaximum():.6f}, {extent.yMaximum():.6f}"
    )
