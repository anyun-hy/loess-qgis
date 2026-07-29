"""Cancelable, sequential tile extraction using QGIS Processing tasks."""

import os

from qgis.PyQt.QtCore import QObject, QTimer, pyqtSignal
from qgis.core import (
    QgsApplication,
    QgsProcessingAlgRunnerTask,
    QgsProcessingContext,
    QgsProcessingFeedback,
)

from . import tile_manager


class AsyncTileExtractionRunner(QObject):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(self, raster_source, crs_authid, tiles, output_dir, parent=None):
        super().__init__(parent)
        self._raster_source = str(raster_source).split("|", 1)[0]
        self._crs_authid = crs_authid
        self._tiles = list(tiles)
        self._output_dir = output_dir
        self._index = 0
        self._task = None
        self._feedback = None
        self._context = None
        self._cancel_requested = False
        self._terminal_emitted = False
        self._task_handled = False

    @property
    def is_running(self):
        return not self._terminal_emitted and self._index < len(self._tiles)

    def start(self):
        os.makedirs(self._output_dir, exist_ok=True)
        self.progress.emit(0, len(self._tiles), "准备提取切片")
        QTimer.singleShot(0, self._start_next)

    def stop(self):
        if self._terminal_emitted:
            return
        self._cancel_requested = True
        if self._feedback is not None:
            self._feedback.cancel()
        if self._task is not None:
            self._task.cancel()
        else:
            self._emit_stopped()

    def _start_next(self):
        if self._cancel_requested:
            self._emit_stopped()
            return
        if self._index >= len(self._tiles):
            self._terminal_emitted = True
            self.finished.emit(self._tiles)
            return

        algorithm = QgsApplication.processingRegistry().algorithmById(
            "gdal:cliprasterbyextent"
        )
        if algorithm is None:
            self._emit_failed("QGIS 中未找到 gdal:cliprasterbyextent 算法")
            return

        tile = self._tiles[self._index]
        params, self._expected_output = tile_manager.build_extract_parameters(
            self._raster_source,
            tile,
            self._output_dir,
            self._crs_authid,
        )
        self._context = QgsProcessingContext()
        self._feedback = QgsProcessingFeedback()
        self._task = QgsProcessingAlgRunnerTask(
            algorithm, params, self._context, self._feedback
        )
        self._task_handled = False
        self._task.executed.connect(self._on_executed)
        self._task.taskTerminated.connect(self._on_terminated)
        QgsApplication.taskManager().addTask(self._task)

    def _on_executed(self, successful, results):
        if self._task_handled or self._terminal_emitted:
            return
        self._task_handled = True
        if self._cancel_requested:
            self._clear_task()
            self._emit_stopped()
            return
        if not successful:
            detail = self._feedback_detail() or "切片任务执行失败"
            self._clear_task()
            self._emit_failed(self._tile_error(detail))
            return

        tile = self._tiles[self._index]
        output_path = (results or {}).get("OUTPUT", self._expected_output)
        try:
            tile_info = tile_manager.finalize_extracted_tile(
                output_path,
                tile,
                self._output_dir,
                self._crs_authid,
            )
        except Exception as exc:
            self._clear_task()
            self._emit_failed(self._tile_error(str(exc)))
            return

        tile["tile_path"] = tile_info["tile_path"]
        self._index += 1
        self.progress.emit(
            self._index,
            len(self._tiles),
            f"Tile ({tile['row']},{tile['col']}) 提取完成",
        )
        self._clear_task()
        QTimer.singleShot(0, self._start_next)

    def _on_terminated(self):
        if self._task_handled or self._terminal_emitted:
            return
        self._task_handled = True
        detail = self._feedback_detail() or "切片任务被异常终止"
        self._clear_task()
        if self._cancel_requested:
            self._emit_stopped()
        else:
            self._emit_failed(self._tile_error(detail))

    def _tile_error(self, message):
        if self._index >= len(self._tiles):
            return message
        tile = self._tiles[self._index]
        return f"Tile ({tile['row']},{tile['col']}): {message}"

    def _feedback_detail(self):
        if self._feedback is None or not hasattr(self._feedback, "textLog"):
            return ""
        try:
            lines = [line.strip() for line in self._feedback.textLog().splitlines()]
        except Exception:
            return ""
        useful = [line for line in lines if line]
        return useful[-1][:1200] if useful else ""

    def _clear_task(self):
        self._task = None
        self._feedback = None
        self._context = None

    def _emit_failed(self, message):
        if self._terminal_emitted:
            return
        self._terminal_emitted = True
        self.failed.emit(message)

    def _emit_stopped(self):
        if self._terminal_emitted:
            return
        self._terminal_emitted = True
        self.stopped.emit()
