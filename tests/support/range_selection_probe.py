# ruff: noqa: E402
"""Real QGIS ranges and Qt delivery using temporary, synthetic inputs only."""

from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
try:
    from qgis.core import (
        QgsApplication,
        QgsCoordinateReferenceSystem,
        QgsFeature,
        QgsGeometry,
        QgsProject,
        QgsRasterLayer,
        QgsRectangle,
        QgsVectorLayer,
    )
    from qgis.gui import QgsMapCanvas, QgsMapToolPan
    from qgis.PyQt.QtCore import QCoreApplication, QEvent, QObject, pyqtSignal
except ModuleNotFoundError:
    raise SystemExit(77)

from osgeo import gdal, osr

from labeling_tool.main import range_preview, range_selection
from labeling_tool.qgis_support import tile_manager

checks = unittest.TestCase()


def drain(app):
    app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def wait_until(app, condition, timeout=3):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    assert condition(), "Qt event delivery timed out"


class Inputs:
    def __init__(self, root):
        path = root / "image.tif"
        dataset = gdal.GetDriverByName("GTiff").Create(str(path), 256, 256, 1)
        dataset.SetGeoTransform((0, 1, 0, 256, 0, -1))
        reference = osr.SpatialReference()
        reference.ImportFromEPSG(3857)
        dataset.SetProjection(reference.ExportToWkt())
        dataset.GetRasterBand(1).Fill(1)
        dataset = None
        self.raster = QgsRasterLayer(str(path), "image", "gdal")
        assert self.raster.isValid()
        self.layer = QgsVectorLayer("Polygon?crs=EPSG:3857", "range", "memory")
        # Two distant islands: the bounding grid contains tiles outside the mask.
        for bounds in ((4, 4, 20, 20), (170, 170, 190, 190)):
            feature = QgsFeature()
            feature.setGeometry(QgsGeometry.fromRect(QgsRectangle(*bounds)))
            assert self.layer.dataProvider().addFeatures([feature])[0]
        self.layer.updateExtents()
        self.canvas = QgsMapCanvas()
        self.canvas.resize(256, 256)
        self.canvas.setDestinationCrs(self.raster.crs())
        self.canvas.setExtent(self.raster.extent())
        self.pan = QgsMapToolPan(self.canvas)
        self.canvas.setMapTool(self.pan)
        self.iface = SimpleNamespace(
            mapCanvas=lambda: self.canvas, actionPan=lambda: None
        )

    def close(self):
        self.canvas.unsetMapTool(self.canvas.mapTool())
        self.canvas.close()


class ControlledTask(QObject):
    progressChanged = pyqtSignal(float)
    taskCompleted = pyqtSignal()
    taskTerminated = pyqtSignal()

    def __init__(self, *args):
        super().__init__()
        self.worker = tile_manager.VectorTileSelectionTask(*args)
        self.request_key = args[0]
        self.result_data = None
        self.error_message = ""
        self.canceled = False

    def cancel(self):
        self.canceled = True

    def isCanceled(self):
        return self.canceled

    def complete(self):
        assert self.worker.run(), self.worker.error_message
        self.result_data = self.worker.result_data
        self.taskCompleted.emit()


class PreviewHarness:
    def __init__(self, root):
        self.inputs = Inputs(root)
        self.tasks, self.ready, self.errors, self.progress, self.starts = (
            [],
            [],
            [],
            [],
            [],
        )
        self.controller = range_preview.VectorPreviewController(
            task_factory=ControlledTask, submit_task=self.tasks.append, debounce_ms=30
        )
        self.controller.preview_ready.connect(self.ready.append)
        self.controller.preview_failed.connect(self.errors.append)
        self.controller.progress_changed.connect(self.progress.append)
        self.controller.auto_start_ready.connect(lambda: self.starts.append(True))

    def request(self, width=64):
        return self.controller.make_request(
            self.inputs.raster,
            self.inputs.layer,
            self.inputs.layer.extent(),
            range_preview.TileParameters(width, 64, 0),
        )

    def close(self):
        self.controller.close()
        self.inputs.close()


def range_modes(app, root):
    inputs = Inputs(root)
    controller = range_selection.RangeSelectionController(inputs.iface)
    try:
        assert controller.selected(range_selection.VIEW_MODE).extent is None
        captured = controller.capture_current_view()
        expected = QgsRectangle(captured.extent)
        inputs.canvas.setExtent(QgsRectangle(20, 20, 120, 120))
        assert controller.selected(range_selection.VIEW_MODE).extent == expected
        # Returning a selection must not grant write access to the owner's state.
        captured.extent.setXMinimum(-999)
        assert controller.selected(range_selection.VIEW_MODE).extent == expected

        rectangles = []
        controller.rectangle_finished.connect(rectangles.append)
        assert controller.begin_rectangle() and controller.is_drawing
        tool = inputs.canvas.mapTool()
        tool.rect_finished.emit(QgsRectangle(20, 30, 90, 100))
        assert len(rectangles) == 1
        selected = controller.selected(range_selection.RECTANGLE_MODE)
        assert selected.extent == QgsRectangle(20, 30, 90, 100)
        controller.restore_map_tool()
        assert inputs.canvas.mapTool() is inputs.pan
        assert not controller.is_drawing
        vector = controller.selected(range_selection.VECTOR_MODE, inputs.layer)
        assert vector.extent == inputs.layer.extent()
        assert vector.vector_layer is inputs.layer

        assert controller.begin_rectangle()
        replacement = QgsMapToolPan(inputs.canvas)
        inputs.canvas.setMapTool(replacement)
        controller.close()
        assert inputs.canvas.mapTool() is replacement
        controller.close()
    finally:
        controller.close()
        inputs.close()


def range_validation(app, root):
    inputs = Inputs(root)
    try:
        assert range_selection.validate_raster_layer(inputs.raster) is inputs.raster
        assert range_selection.validate_vector_layer(inputs.layer) is inputs.layer
        with checks.assertRaisesRegex(ValueError, "有效的本地影像"):
            range_selection.validate_raster_layer(None)
        empty = QgsVectorLayer("Polygon?crs=EPSG:3857", "empty", "memory")
        with checks.assertRaisesRegex(ValueError, "没有面要素"):
            range_selection.validate_vector_layer(empty)
        line = QgsVectorLayer("LineString?crs=EPSG:3857", "line", "memory")
        with checks.assertRaisesRegex(ValueError, "面图层"):
            range_selection.validate_vector_layer(line)
        raw = range_selection.RawRangeSelection(
            range_selection.VIEW_MODE,
            QgsRectangle(-10, -10, 50, 50),
            inputs.raster.crs(),
        )
        assert range_selection.resolve_raster_extent(
            raw, inputs.raster
        ) == QgsRectangle(0, 0, 50, 50)
        outside = range_selection.RawRangeSelection(
            range_selection.VIEW_MODE,
            QgsRectangle(300, 300, 400, 400),
            inputs.raster.crs(),
        )
        with checks.assertRaisesRegex(ValueError, "没有重叠"):
            range_selection.resolve_raster_extent(outside, inputs.raster)
        with checks.assertRaisesRegex(ValueError, "拖拽绘制"):
            range_selection.resolve_raster_extent(
                range_selection.RawRangeSelection(
                    range_selection.RECTANGLE_MODE, None, None
                ),
                inputs.raster,
            )
        projected = range_selection.transform_extent(
            QgsRectangle(0, 0, 1, 1),
            QgsCoordinateReferenceSystem(4326),
            inputs.raster.crs(),
        )
        checks.assertAlmostEqual(projected.xMaximum(), 111319.49079, places=3)
        with checks.assertRaisesRegex(ValueError, "CRS 无效"):
            range_selection.transform_extent(
                QgsRectangle(0, 0, 1, 1), None, inputs.raster.crs()
            )
        assert (
            range_selection.intersect_extents(
                QgsRectangle(0, 0, 1, 1), QgsRectangle(1, 1, 2, 2)
            )
            is None
        )

        grid = tile_manager.generate_grid(
            inputs.layer.extent(), 64, 64, 0, raster_layer=inputs.raster
        )
        selected = tile_manager.select_tiles_intersecting_vector(
            grid, inputs.layer, inputs.raster.crs()
        )
        assert len(selected) == 2 < len(grid)
        metadata = range_selection.range_selection_metadata(
            inputs.layer, len(grid), len(selected)
        )
        assert metadata == {
            "mode": "vector_tile_intersection",
            "vector_layer_id": inputs.layer.id(),
            "vector_layer_name": "range",
            "vector_source": inputs.layer.source(),
            "vector_crs": "EPSG:3857",
            "selected_tile_count": 2,
            "excluded_tile_count": len(grid) - 2,
            "clip_outputs": True,
        }
        assert range_selection.range_selection_metadata(None, 3, 3) == {
            "mode": "extent",
            "selected_tile_count": 3,
            "excluded_tile_count": 0,
            "clip_outputs": True,
        }
    finally:
        inputs.close()


def preview_replacement(app, root):
    h = PreviewHarness(root)
    try:
        first = h.request()
        h.controller.queue(first, immediate=True)
        old = h.tasks[-1]
        h.controller.queue(first, immediate=True)
        assert len(h.tasks) == 1
        second = h.request(128)
        h.controller.queue(second, immediate=True)
        assert len(h.tasks) == 2 and old.canceled
        old.progressChanged.emit(88)
        old.complete()
        old.taskTerminated.emit()
        assert not h.ready and not h.progress and not h.errors
        h.tasks[-1].progressChanged.emit(25)
        assert h.progress == [25]
        h.tasks[-1].complete()
        assert len(h.ready) == 1
        assert h.ready[0].selected_count == 2 < h.ready[0].grid_count
        assert h.controller.cached(first) is None
        assert h.controller.cached(second) is h.ready[0]
        assert h.controller.queue(second) is h.ready[0]
        assert len(h.tasks) == 2
        h.controller.invalidate()
        h.controller.queue(first)
        h.controller.queue(second)
        wait_until(app, lambda: len(h.tasks) == 3)
        assert h.tasks[-1].request_key == second.key
    finally:
        h.close()


def preview_source_changes(app, root):
    h = PreviewHarness(root)
    try:
        changes = []
        h.controller.source_changed.connect(lambda: changes.append(True))
        h.controller.watch_layer(h.inputs.layer)
        request = h.request()
        h.controller.queue(request, immediate=True, auto_start=True)
        task = h.tasks[-1]
        # Same count/extent/key can still contain different polygon geometry.
        h.inputs.layer.dataChanged.emit()
        assert changes == [True] and task.canceled
        task.complete()
        drain(app)
        assert not h.ready and not h.starts and h.controller.cached(request) is None
        replacement = QgsVectorLayer("Polygon?crs=EPSG:3857", "new range", "memory")
        h.controller.watch_layer(replacement)
        h.inputs.layer.dataChanged.emit()
        assert len(changes) == 1
        replacement.dataChanged.emit()
        assert len(changes) == 2
        h.controller.close()
        replacement.dataChanged.emit()
        assert len(changes) == 2
    finally:
        h.close()


def preview_autostart(app, root):
    h = PreviewHarness(root)
    try:
        request = h.request()
        h.controller.queue(request, immediate=True, auto_start=True)
        h.tasks[-1].complete()
        h.controller.invalidate()  # Cancel after completion, before the queued start.
        drain(app)
        assert not h.starts
        h.controller.queue(request, immediate=True, auto_start=True)
        h.controller.queue(request)  # A redundant refresh must preserve user intent.
        task = h.tasks[-1]
        task.complete()
        drain(app)
        assert h.starts == [True]
        task.taskCompleted.emit()
        drain(app)
        assert h.starts == [True]
        h.controller.invalidate()
        h.controller.queue(request, immediate=True, auto_start=True)
        failed = h.tasks[-1]
        failed.error_message = "synthetic failure"
        failed.taskTerminated.emit()
        drain(app)
        assert h.errors == ["synthetic failure"] and h.starts == [True]
    finally:
        h.close()


def preview_shutdown(app, root):
    from qgis.PyQt.QtTest import QTest

    h = PreviewHarness(root)
    try:
        request = h.request()
        h.controller.queue(request, immediate=True, auto_start=True)
        active = h.tasks[-1]
        h.controller.close()
        assert active.canceled
        active.progressChanged.emit(20)
        active.complete()
        active.taskTerminated.emit()
        drain(app)
        assert not h.ready and not h.progress and not h.errors and not h.starts
        h.controller.queue(request, immediate=True)
        assert len(h.tasks) == 1
        pending = range_preview.VectorPreviewController(
            task_factory=ControlledTask, submit_task=h.tasks.append, debounce_ms=30
        )
        pending.queue(request)
        pending.close()
        QTest.qWait(60)
        assert len(h.tasks) == 1
    finally:
        h.close()


def preview_native_task(app, root):
    inputs = Inputs(root)
    threads, delivered, results = [], [], []

    class NativeTask(tile_manager.VectorTileSelectionTask):
        def run(self):
            threads.append(threading.get_ident())
            return super().run()

    controller = range_preview.VectorPreviewController(task_factory=NativeTask)
    controller.preview_ready.connect(
        lambda result: (results.append(result), delivered.append(threading.get_ident()))
    )
    try:
        request = controller.make_request(
            inputs.raster,
            inputs.layer,
            inputs.layer.extent(),
            range_preview.TileParameters(64, 64, 0),
        )
        controller.queue(request, immediate=True)
        wait_until(app, lambda: bool(results))
        assert results[0].selected_count == 2 < results[0].grid_count
        assert len(threads) == 1 and threads[0] != threading.get_ident()
        assert delivered == [threading.get_ident()]
    finally:
        controller.close()
        inputs.close()


def dock_range_inputs(app, root):
    from qgis.PyQt.QtWidgets import QDialog

    from labeling_tool.main import main_dock

    class Monitor(QDialog):
        stop_requested = pyqtSignal()
        request_main_run_handling = pyqtSignal(object)

        def detach(self):
            pass

        def shutdown(self):
            self.close()

    inputs = Inputs(root)
    tasks, started, warnings = [], [], []
    dock = None
    try:
        with ExitStack() as patches:
            for name in (
                "_load_settings_and_defaults",
                "_save_settings",
                "_restore_latest_ready_run",
            ):
                patches.enter_context(
                    patch.object(main_dock.LabelingDockWidget, name, lambda *_: None)
                )
            patches.enter_context(
                patch.object(main_dock, "InferenceMonitorDialog", Monitor)
            )
            patches.enter_context(
                patch.object(main_dock, "ClassRefinementDialog", lambda *_: None)
            )
            patches.enter_context(
                patch.object(
                    main_dock,
                    "VectorPreviewController",
                    lambda parent: range_preview.VectorPreviewController(
                        parent, task_factory=ControlledTask, submit_task=tasks.append
                    ),
                )
            )
            patches.enter_context(
                patch.object(
                    main_dock.QMessageBox,
                    "warning",
                    side_effect=lambda *args: warnings.append(args[1:]),
                )
            )
            dock = main_dock.LabelingDockWidget(None, iface=inputs.iface)
            patches.enter_context(
                patch.object(dock.config_manager, "is_stale", return_value=False)
            )
            patches.enter_context(
                patch.object(dock.workflow, "start_new_run", side_effect=started.append)
            )
            QgsProject.instance().addMapLayer(inputs.raster)
            QgsProject.instance().addMapLayer(inputs.layer)
            dock.raster_combo.setLayer(inputs.raster)
            dock.vector_range_combo.setLayer(inputs.layer)
            dock.tile_width_spin.setValue(64)
            dock.tile_height_spin.setValue(64)
            dock.overlap_spin.setValue(1)
            dock.environment_panel.scripts_directory = str(ROOT / "scripts/runtime")
            dock.workspace_edit.setText(str(root / "workspace"))
            dock.output_path_edit.setText(str(root / "accepted.gpkg"))
            dock.config_manager.last_report = {
                "status": "ready",
                "checks": [{"id": "semantic_model_fixture", "status": "ready"}],
                "effective": {
                    "schema_version": 2,
                    "semantic_models": [{"model_id": "fixture"}],
                },
            }
            dock.config_manager.report_ready.emit(dock.config_manager.last_report)
            dock.plan_panel.configuration_dialog.configuration_applied.emit(
                ["fixture"], None, True
            )
            inputs.canvas.setExtent(QgsRectangle(4, 4, 190, 190))
            dock.capture_view_btn.click()
            assert "当前视图范围" in dock.extent_status_label.text()
            assert "Tile" in dock.processing_extent_status_label.text(), (
                dock.processing_extent_status_label.text()
            )
            dock.radio_rect.click()
            dock.draw_rect_btn.click()
            inputs.canvas.mapTool().rect_finished.emit(QgsRectangle(0, 0, 128, 128))
            assert inputs.canvas.mapTool() is inputs.pan
            assert "手绘矩形范围" in dock.extent_status_label.text()
            dock.radio_vector.click()
            assert dock.start_btn.isEnabled()
            dock.vector_range_combo.setLayer(None)
            assert not dock.start_btn.isEnabled()
            assert "有效的矢量范围图层" in dock.start_readiness_label.text()
            dock.start_btn.click()
            assert not started and not tasks and not warnings
            # The final launch guard must still reject an invalid range even
            # when invoked independently of the disabled UI button.
            dock._on_start()
            assert len(warnings) == 1 and "有效的已加载矢量面图层" in warnings[0][1]
            assert not started and not tasks
            warnings.clear()
            dock.vector_range_combo.setLayer(inputs.layer)
            dock.config_manager.last_report["checks"][0]["status"] = "error"
            dock._update_start_enabled()
            assert not dock.start_btn.isEnabled()
            assert "模型未通过设备实测" in dock.start_readiness_label.text()
            dock.start_btn.click()
            assert not started and not tasks and not warnings
            dock._on_start()
            assert len(warnings) == 1 and warnings[0] == (
                "推理方案不可运行",
                "模型未通过设备实测: fixture",
            )
            assert not started and not tasks
            warnings.clear()
            dock.config_manager.last_report["checks"][0]["status"] = "ready"
            dock._update_start_enabled()
            assert dock.start_btn.isEnabled()
            dock.start_btn.click()
            assert len(tasks) == 1 and not started and not warnings
            abandoned = tasks[-1]
            dock.radio_view.click()
            abandoned.complete()
            drain(app)
            assert not started and abandoned.canceled
            dock.radio_vector.click()
            dock.start_btn.click()
            assert len(tasks) == 2
            tasks[-1].complete()
            drain(app)
            assert len(started) == 1 and not warnings
            request = started[0]
            assert request.selected_model_ids == ("fixture",)
            assert request.fusion_profile_id is None
            assert request.boundary_smoothing_enabled is True
            assert request.range_selection["mode"] == "vector_tile_intersection"
            assert request.range_selection["clip_outputs"] is True
            assert len(request.active_tiles) == 2 < len(request.grid_tiles)
            assert request.get_valid_range_layer() is inputs.layer
            dock.cleanup()
            dock.deleteLater()
            drain(app)
            dock = None
    finally:
        if dock is not None:
            # Avoid persisting test paths even when an assertion fails.
            with patch.object(dock, "_save_settings", lambda: None):
                dock.cleanup()
            dock.deleteLater()
            drain(app)
        inputs.close()
        QgsProject.instance().removeAllMapLayers()


if __name__ == "__main__":
    app = QgsApplication([], False)
    app.initQgis()
    scenario = sys.argv[2]
    try:
        with tempfile.TemporaryDirectory(prefix="loess-range-") as temporary:
            globals()[scenario](app, Path(temporary))
        print(scenario + ": passed")
    finally:
        drain(app)
        app.exitQgis()
