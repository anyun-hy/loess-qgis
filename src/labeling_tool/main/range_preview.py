from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from qgis.core import QgsApplication, QgsRasterLayer, QgsRectangle, QgsVectorLayer
from qgis.PyQt.QtCore import QObject, QTimer, pyqtSignal

from labeling_tool.qgis_support import tile_manager

PreviewKey = tuple[Any, ...]
TileRecord = dict[str, Any]


@dataclass(frozen=True)
class TileParameters:
    width: int
    height: int
    overlap: int


@dataclass(frozen=True)
class VectorPreviewRequest:
    raster: QgsRasterLayer
    vector_layer: QgsVectorLayer
    extent: QgsRectangle
    parameters: TileParameters
    key: PreviewKey


@dataclass(frozen=True)
class TileGridSummary:
    processing_extent: QgsRectangle | None
    rows: int
    cols: int
    grid_count: int
    selected_count: int


@dataclass(frozen=True)
class VectorPreviewResult(TileGridSummary):
    key: PreviewKey
    grid_tiles: list[TileRecord]
    selected_tiles: list[TileRecord]


def vector_preview_key(
    raster: QgsRasterLayer,
    vector_layer: QgsVectorLayer,
    extent: QgsRectangle,
    parameters: TileParameters,
) -> PreviewKey:
    layer_extent = vector_layer.extent()
    raster_extent = raster.extent()
    return (
        raster.id(),
        raster.source(),
        int(raster.width()),
        int(raster.height()),
        tuple(
            round(value, 9)
            for value in (
                raster_extent.xMinimum(),
                raster_extent.yMinimum(),
                raster_extent.xMaximum(),
                raster_extent.yMaximum(),
            )
        ),
        vector_layer.id(),
        vector_layer.source(),
        int(vector_layer.featureCount()),
        tuple(
            round(value, 9)
            for value in (
                layer_extent.xMinimum(),
                layer_extent.yMinimum(),
                layer_extent.xMaximum(),
                layer_extent.yMaximum(),
            )
        ),
        tuple(
            round(value, 9)
            for value in (
                extent.xMinimum(),
                extent.yMinimum(),
                extent.xMaximum(),
                extent.yMaximum(),
            )
        ),
        int(parameters.width),
        int(parameters.height),
        int(parameters.overlap),
    )


class VectorPreviewController(QObject):
    """Own debounce, task, cache, source observation, and auto-start state."""

    progress_changed = pyqtSignal(float)
    preview_ready = pyqtSignal(object)
    preview_failed = pyqtSignal(str)
    source_changed = pyqtSignal()
    auto_start_ready = pyqtSignal()

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        task_factory: Callable[..., Any] = tile_manager.VectorTileSelectionTask,
        submit_task: Callable[[Any], Any] | None = None,
        debounce_ms: int = 180,
    ) -> None:
        super().__init__(parent)
        self._task_factory = task_factory
        self._submit_task = submit_task or QgsApplication.taskManager().addTask
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(int(debounce_ms))
        self._timer.timeout.connect(self._start_pending)
        self._pending: VectorPreviewRequest | None = None
        self._requested_key: PreviewKey | None = None
        self._cache: VectorPreviewResult | None = None
        self._task: Any | None = None
        self._auto_start = False
        self._auto_start_token = 0
        self._observed_layer: QgsVectorLayer | None = None
        self._closed = False

    def make_request(
        self,
        raster: QgsRasterLayer,
        vector_layer: QgsVectorLayer,
        extent: QgsRectangle,
        parameters: TileParameters,
    ) -> VectorPreviewRequest:
        key = vector_preview_key(raster, vector_layer, extent, parameters)
        return VectorPreviewRequest(
            raster=raster,
            vector_layer=vector_layer,
            extent=QgsRectangle(extent),
            parameters=parameters,
            key=key,
        )

    def cached(self, request: VectorPreviewRequest) -> VectorPreviewResult | None:
        cache = self._cache
        return cache if cache is not None and cache.key == request.key else None

    def queue(
        self,
        request: VectorPreviewRequest,
        *,
        immediate: bool = False,
        auto_start: bool = False,
    ) -> VectorPreviewResult | None:
        if self._closed:
            return None
        cache = self.cached(request)
        if cache is not None:
            return cache

        same_key = request.key == self._requested_key
        if not same_key:
            self._auto_start = bool(auto_start)
            self._auto_start_token += 1
        elif auto_start:
            self._auto_start = True
            self._auto_start_token += 1
        self._pending = request
        self._requested_key = request.key
        task = self._task
        if task is not None and task.request_key == request.key:
            return None
        if task is not None:
            self._task = None
            task.cancel()
        if immediate:
            self._timer.stop()
            self._start_pending()
        else:
            self._timer.start()
        return None

    def watch_layer(self, layer: QgsVectorLayer | None) -> None:
        if layer is self._observed_layer:
            return
        previous = self._observed_layer
        if previous is not None:
            try:
                previous.dataChanged.disconnect(self._on_source_changed)
            except (TypeError, RuntimeError):
                pass
        self._observed_layer = layer
        if layer is not None:
            layer.dataChanged.connect(self._on_source_changed)

    def invalidate(self) -> None:
        self._timer.stop()
        self._pending = None
        self._requested_key = None
        self._cache = None
        self._auto_start = False
        self._auto_start_token += 1
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.watch_layer(None)
        self.invalidate()
        try:
            self._timer.timeout.disconnect(self._start_pending)
        except (TypeError, RuntimeError):
            pass

    def _on_source_changed(self, *_args: Any) -> None:
        self.invalidate()
        self.source_changed.emit()

    def _start_pending(self) -> None:
        if self._closed:
            return
        request = self._pending
        if request is None:
            return
        try:
            geometries = tile_manager.snapshot_vector_geometries(
                request.vector_layer, request.raster.crs()
            )
            parameters = request.parameters
            task = self._task_factory(
                request.key,
                request.extent,
                parameters.width,
                parameters.height,
                parameters.overlap,
                tile_manager.raster_grid_info(request.raster),
                geometries,
            )
        except ValueError as exc:
            self._auto_start = False
            self.preview_failed.emit(str(exc))
            return

        self._task = task
        task.progressChanged.connect(self._on_task_progress)
        task.taskCompleted.connect(self._on_task_completed)
        task.taskTerminated.connect(self._on_task_terminated)
        self._submit_task(task)

    def _on_task_progress(self, progress: float) -> None:
        if self.sender() is not self._task:
            return
        self.progress_changed.emit(float(progress))

    def _on_task_completed(self) -> None:
        task = self.sender()
        if task is not self._task:
            return
        self._task = None
        raw_result = task.result_data
        if raw_result is None or raw_result.get("key") != self._requested_key:
            return
        result = VectorPreviewResult(
            key=raw_result["key"],
            grid_tiles=raw_result["grid_tiles"],
            selected_tiles=raw_result["selected_tiles"],
            processing_extent=raw_result.get("processing_extent"),
            rows=int(raw_result["rows"]),
            cols=int(raw_result["cols"]),
            grid_count=int(raw_result["grid_count"]),
            selected_count=int(raw_result["selected_count"]),
        )
        self._cache = result
        self.preview_ready.emit(result)
        if self._auto_start:
            self._auto_start = False
            token = self._auto_start_token
            QTimer.singleShot(0, lambda: self._emit_auto_start(token))

    def _on_task_terminated(self) -> None:
        task = self.sender()
        if task is not self._task:
            return
        self._task = None
        if task.isCanceled():
            return
        self._auto_start = False
        self.preview_failed.emit(task.error_message or "矢量范围 Tile 计算失败")

    def _emit_auto_start(self, token: int) -> None:
        if self._closed or token != self._auto_start_token:
            return
        self.auto_start_ready.emit()
