"""Native acceptance on synthetic, temporary data; no real Run/DB/inference."""

import json
import os
import copy
from datetime import datetime, timedelta
from pathlib import Path
import sys
import tempfile
import threading
import time
from unittest.mock import patch

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "qgis_plugins"))
try:
    from qgis.PyQt.QtCore import QCoreApplication, QObject, QTimer, pyqtSignal, pyqtSlot
    from qgis.core import (
        QgsApplication, QgsFeature, QgsGeometry, QgsRectangle, QgsVectorLayer,
        QgsVectorFileWriter, QgsVectorLayerFeatureSource, QgsCoordinateTransformContext,
    )
except ModuleNotFoundError:
    raise SystemExit(77)

from labeling_tool.core import class_workspace
from labeling_tool.core.final_assembler import FINAL_FIELDS
from labeling_tool.core.qgis_writer import write_vector_layer
from labeling_tool.core.run_spec import CLASS_NAMES, CLASS_ORDER, sha256_file


# The monitor probe exercises many UI states with assertions in between.  Keep
# the active dialog visible to the process-level finally block so an assertion
# failure cannot leave its PostgreSQL worker thread running.
_ACTIVE_MONITOR_DIALOG = None


def feature(layer, index, code=12):
    item = QgsFeature(layer.fields())
    geometry = QgsGeometry.fromRect(QgsRectangle(index, 0, index + 1, 1))
    geometry.convertToMultiType()
    item.setGeometry(geometry)
    values = dict(run_id="probe", object_id=f"object-{code}-{index}", part_id="000",
                  class_code=code, class_name=CLASS_NAMES[code], reviewed=1,
                  baseline_stream_id="fusion:probe", source_stream_id="fusion:probe",
                  geometry_source="fusion", geometry_revision=0,
                  confidence_mean=0.5, confidence_std=0.1,
                  created_at="2026-01-01", updated_at="2026-01-01")
    for name, value in values.items():
        item.setAttribute(name, value)
    return item


def memory_layer(count=1, code=12):
    layer = QgsVectorLayer("MultiPolygon?crs=EPSG:3857", "probe", "memory")
    assert layer.isValid(), QgsApplication.showSettings()
    layer.dataProvider().addAttributes(FINAL_FIELDS)
    layer.updateFields()
    ok, _ = layer.dataProvider().addFeatures([feature(layer, index, code) for index in range(count)])
    assert ok
    layer.updateExtents()
    return layer


def fixture(root):
    directory = root / "classes"
    directory.mkdir()
    spec = dict(run_id="probe", run_dir=str(root), raster=dict(crs="EPSG:3857", transform=[1,0,0,0,-1,0]),
                requested_extent=dict(xmin=0, ymin=0, xmax=1, ymax=1))
    workspace = dict(run_id="probe", baseline_stream_id="fusion:probe", classes={})
    for code in CLASS_ORDER:
        path = directory / f"class_{code}.gpkg"
        layer = memory_layer(1 if code == 12 else 0, code)
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "GPKG"
        options.layerName = f"class_{code}"
        error, message = write_vector_layer(layer, path, options)
        assert error == QgsVectorFileWriter.WriterError.NoError, message
        workspace["classes"][str(code)] = dict(path=str(path), layer_name=f"class_{code}",
                                               class_code=code, confirmed=True,
                                               feature_count=layer.featureCount(), sha256=sha256_file(path))
    return spec, workspace


def run_task(app, task):
    from qgis.PyQt.QtCore import QEventLoop
    loop = QEventLoop()
    task.taskCompleted.connect(loop.quit)
    task.taskTerminated.connect(loop.quit)
    deadline = QTimer()
    deadline.setSingleShot(True)
    deadline.timeout.connect(loop.quit)
    deadline.start(20000)
    QgsApplication.taskManager().addTask(task)
    loop.exec()
    deadline.stop()
    assert task.result_data is not None, task.error_message
    return task.result_data


def shutdown(app, root):
    from labeling_tool.core.v5_async_runner import ThreadedV5AsyncInferenceRunner
    runner = ThreadedV5AsyncInferenceRunner(str(ROOT / "inference_scripts"))
    ticks = [time.monotonic()]
    measured = {}
    main_tid = threading.get_ident()
    class Slow(QObject):
        entered = pyqtSignal()
        @pyqtSlot()
        def run(self):
            assert threading.get_ident() != main_tid
            self.entered.emit()
            time.sleep(0.6)
    class Driver(QObject):
        begin = pyqtSignal()
        @pyqtSlot()
        def entered(self):
            QTimer.singleShot(50, self.stop)
        @pyqtSlot()
        def stop(self):
            start = time.monotonic()
            runner.shutdown()
            measured["shutdown_return_ms"] = (time.monotonic()-start)*1000
    slow, driver = Slow(), Driver()
    slow.moveToThread(runner._runtime_thread)
    runner._runtime_thread.finished.connect(slow.deleteLater)
    driver.begin.connect(slow.run)
    slow.entered.connect(driver.entered)
    runner.shutdown_finished.connect(app.quit)
    timer = QTimer()
    timer.setInterval(20)
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start()
    QTimer.singleShot(50, driver.begin.emit)
    QTimer.singleShot(4000, app.quit)
    app.exec()
    timer.stop()
    measured["max_main_loop_gap_ms"] = max(b-a for a,b in zip(ticks,ticks[1:]))*1000
    assert measured["shutdown_return_ms"] < 100, measured
    assert measured["max_main_loop_gap_ms"] < 200, measured
    assert not runner._runtime_thread.isRunning()
    assert not runner._log_thread.isRunning()
    return measured


def preparation(app, root):
    from labeling_tool.core.run_preparation_task import RunPreparationTask
    from labeling_tool.core.accepted_writer import ACCEPTED_FIELDS_QGS
    source = memory_layer()
    tiles = [dict(row=0, col=i, bounds=QgsRectangle(i,0,i+1,1)) for i in range(3)]
    request = dict(run_dir=str(root), active_tiles=tiles, grid_tiles=tiles,
                   range_selection=dict(mode="vector_tile_intersection"),
                   accepted_validation=dict(overlap_tolerance=1e-12), skip_accepted=False)
    task = RunPreparationTask(request, range_source=QgsVectorLayerFeatureSource(source),
                              accepted_source=None, raster_crs=source.crs(),
                              range_wkb_type=source.wkbType(),
                              transform_context=QgsCoordinateTransformContext())
    value = run_task(app, task)
    assert len(value["active_tiles"]) == 1, value
    assert value["range_selection"]["vector_sha256"] == sha256_file(value["range_snapshot"])
    assert all("range_selected" not in tile for tile in tiles)
    accepted = QgsVectorLayer("MultiPolygon?crs=EPSG:3857", "accepted", "memory")
    accepted.dataProvider().addAttributes(ACCEPTED_FIELDS_QGS)
    accepted.updateFields()
    assert accepted.dataProvider().addFeatures([feature(accepted, 0)])[0]
    accepted_file = root / "accepted.gpkg"
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.layerName = "accepted_labels"
    assert write_vector_layer(accepted, accepted_file, options)[0] == 0
    for skip in (False, True):
        output = root / ("skip" if skip else "no_skip")
        output.mkdir()
        check = RunPreparationTask(
            {**request, "run_dir": str(output), "skip_accepted": skip,
             "accepted_source_path": f"{accepted_file}|layername=accepted_labels"},
            range_source=None, accepted_source=QgsVectorLayerFeatureSource(accepted),
            accepted_wkb_type=accepted.wkbType(), raster_crs=source.crs(),
            transform_context=QgsCoordinateTransformContext(),
        )
        checked = run_task(app, check)
        assert len(checked["skipped_tiles"]) == (1 if skip else 0), checked
        assert (output / "accepted_snapshot.gpkg").exists() is skip
        assert bool(checked["accepted_snapshot"]) is skip
    return dict(selected_tiles=1, frozen_input=True, skip_modes_preserved=True)


def sam_shutdown(app, root):
    from qgis.PyQt.QtCore import QProcess, QCoreApplication, QEvent
    from labeling_tool.core.sam3_worker_runner import Sam3WorkerRunner
    from labeling_tool.core.qt_lifecycle import retire_after, _retiring
    parent = QObject()
    runner = Sam3WorkerRunner(str(root), {}, parent=parent)
    process = QProcess(runner)
    runner._process = process
    process.finished.connect(runner._finished)
    ticks = [time.monotonic()]
    returns = []
    destroyed = []
    runner.destroyed.connect(lambda: destroyed.append(True))
    def stop():
        retire_after(runner, runner.stopped)
        parent.deleteLater()
        start = time.monotonic()
        runner.stop()
        returns.append((time.monotonic() - start) * 1000)
    process.started.connect(stop)
    runner.stopped.connect(app.quit)
    timer = QTimer()
    timer.setInterval(20)
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start()
    process.start(sys.executable, ["-B", "-c", "import sys,time; sys.stdin.readline(); time.sleep(0.4)"])
    QTimer.singleShot(4000, app.quit)
    app.exec()
    timer.stop()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert returns and max(returns) < 100, returns
    gap = max(b-a for a,b in zip(ticks,ticks[1:]))*1000
    assert gap < 200, gap
    assert destroyed and runner not in _retiring
    return dict(sam_stop_return_ms=returns[0], max_main_loop_gap_ms=gap,
                owner_survived_parent_then_retired=True)


def workspace(app, root):
    spec, value = fixture(root)
    with patch.object(class_workspace, "sha256_file", wraps=sha256_file) as hashes:
        value = class_workspace.save_workspace(spec, value, changed_class_codes={12})
        assert hashes.call_count == 1
        hashes.reset_mock()
        value = class_workspace.save_workspace(spec, value, changed_class_codes=())
        assert hashes.call_count == 0
        value = class_workspace.save_workspace(spec, value)
        assert hashes.call_count == 14
    assert value["feature_count"] == 1
    return dict(one_class_hashes=1, metadata_hashes=0, full_hashes=14)


def result_layers(app, root):
    from types import SimpleNamespace
    from osgeo import gdal
    from qgis.core import QgsProject
    from labeling_tool.core.layer_manager import LayerManager, MANAGED_PROPERTY
    from labeling_tool.core.layer_names import LAYER_NAMES
    from labeling_tool.core.style_manager import StyleManager

    gdal.UseExceptions()
    spec, value = fixture(root)
    record = value["classes"]["12"]
    raster_paths = {}
    for role, sample in (("mask_mosaic", 0), ("confidence_mosaic", 0.5)):
        path = root / f"{role}.tif"
        dataset = gdal.GetDriverByName("GTiff").Create(str(path), 2, 2, 1, gdal.GDT_Float32)
        dataset.SetGeoTransform([0, 1, 0, 2, 0, -1])
        dataset.SetProjection(memory_layer().crs().toWkt())
        dataset.GetRasterBand(1).Fill(sample)
        dataset = None
        raster_paths[role] = str(path)
    raw_path = root / "raw.gpkg"
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.layerName = LAYER_NAMES.SEMANTIC_RAW
    assert write_vector_layer(memory_layer(), raw_path, options)[0] == 0
    project = QgsProject.instance()
    manager = LayerManager(SimpleNamespace(mapCanvas=lambda: None))
    try:
        for kind, stream_id, section in (("model", "model:probe", "Models"),
                                         ("fusion", "fusion:probe", "Fusion")):
            loaded = manager.load_result_stream("probe", dict(
                kind=kind, model_id="probe", fusion_profile_id="probe", stream_id=stream_id,
                paths={**raster_paths, "semantic_polygons_raw": str(raw_path)},
                review_polygons=record["path"], review_layer_name=record["layer_name"],
            ))
            assert set(loaded) == {"mask", "confidence", "polygons", "polygons_raw"}
            for role, layer_id in loaded.items():
                layer = project.mapLayer(layer_id)
                node = project.layerTreeRoot().findLayer(layer_id)
                assert layer.customProperty(MANAGED_PROPERTY, False)
                assert layer.customProperty("labeling_tool/stream_id") == stream_id
                assert node.parent().name() == section
                if role.startswith("polygons"):
                    categories = layer.renderer().categories()
                    assert [c.value() for c in categories] == sorted(StyleManager.CLASS_COLORS)
                    assert categories[0].symbol().color().name() == StyleManager.CLASS_COLORS[12][1].lower()
                if role == "polygons_raw":
                    assert layer.readOnly() and not node.itemVisibilityChecked()
            assert project.mapLayer(loaded["mask"]).renderer().type() == "paletted"
            assert project.mapLayer(loaded["confidence"]).renderer().type() == "singlebandpseudocolor"
        workspace_id = manager.load_workspace_class(spec["run_id"], record, visible=False)
        node = project.layerTreeRoot().findLayer(workspace_id)
        assert node.parent().name() == "Classes" and not node.itemVisibilityChecked()
        count = len(project.mapLayers())
        assert manager.load_workspace_class("probe", record, visible=True) == workspace_id
        assert len(project.mapLayers()) == count and node.itemVisibilityChecked()
        manager.set_layer_visibility(workspace_id, False)
        assert not node.itemVisibilityChecked()
        return dict(model_and_fusion_layers=True, shared_colors=True,
                    raw_read_only_hidden=True, workspace_reused=True)
    finally:
        project.removeAllMapLayers()


def edits(app, root):
    from labeling_tool.gui.class_refinement_dialog import ClassRefinementDialog as Dialog
    layer = memory_layer(1000)
    path = root / "edit.gpkg"
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.layerName = "edits"
    assert write_vector_layer(layer, path, options)[0] == 0
    layer = QgsVectorLayer(f"{path}|layername=edits", "edit", "ogr")
    history = []
    class Harness:
        _snapshot = Dialog._snapshot
        _uses_provider_transaction = staticmethod(Dialog._uses_provider_transaction)
        _baseline_for_edit = Dialog._baseline_for_edit
        _capture_edit_delta = Dialog._capture_edit_delta
        _record_committed_additions = Dialog._record_committed_additions
        _editing_started = Dialog._editing_started
        _editing_stopped = Dialog._editing_stopped
        _edited_existing_ids = staticmethod(Dialog._edited_existing_ids)
        _set_attributes = Dialog._set_attributes
        _metadata_update = False
        _qgis_smooth_preview = None
        _run_spec = {"run_id": "probe"}
        _workspace = {"baseline_stream_id": "fusion:probe"}
        def __init__(self):
            self._snapshots, self._commit_feature_ids, self._edit_context = {}, {}, {}
            self.baseline_label = type("Label", (), {"setText": lambda *args: None})()
        def _layer(self, code): return layer
        def _update_manual_panel(self): pass
        def _update_actions(self): pass
        def _optional_confidence_statistics(self, *args): return 0.5, 0.1, ""
        def _local_topology_hint(self, *args): return "ok"
        def _mark_class_modified(self, *args): pass
        def _refresh_class_display(self, *args): pass
    owner = Harness()
    layer.editingStarted.connect(lambda: owner._editing_started(12))
    layer.beforeCommitChanges.connect(lambda *args: owner._capture_edit_delta(12))
    layer.committedFeaturesAdded.connect(lambda _id, values: owner._record_committed_additions(12, values))
    layer.afterCommitChanges.connect(lambda: owner._editing_stopped(12))
    layer.editingStopped.connect(lambda: owner._editing_stopped(12))
    with patch.object(class_workspace, "append_history", side_effect=lambda spec, event, **data: history.append((event,data))), \
         patch.object(class_workspace, "geometry_hash", wraps=class_workspace.geometry_hash) as hashes:
        assert layer.startEditing()
        assert hashes.call_count == 0
        fid = next(layer.getFeatures()).id()
        geometry = QgsGeometry.fromRect(QgsRectangle(0, 0, 0.5, 1))
        geometry.convertToMultiType()
        assert layer.changeGeometry(fid, geometry)
        assert layer.commitChanges(False), layer.commitErrors()
        assert hashes.call_count < 12, hashes.call_count
        assert [event for event, data in history] == ["geometry_modified"], history
        assert layer.getFeature(fid)["geometry_revision"] == 1
        history.clear()
        assert layer.addFeature(feature(layer, 1001))
        assert layer.commitChanges(False), layer.commitErrors()
        assert [event for event, data in history] == ["feature_added"], history
        history.clear()
        assert layer.deleteFeature(fid)
        assert layer.commitChanges(), layer.commitErrors()
        assert [event for event, data in history] == ["feature_deleted"], history
        history.clear()
        assert layer.startEditing()
        rollback_id = next(layer.getFeatures()).id()
        assert layer.changeGeometry(rollback_id, geometry)
        assert layer.rollBack()
        assert not history, history
    return dict(layer_features=1000, incremental_edit_tracking=True)


def refinement(app, root):
    from labeling_tool.core.refinement_task import RefinementTask
    spec, value = fixture(root)
    task = RefinementTask(spec, value, final_path="", assemble=True,
                           transform_context=QgsCoordinateTransformContext())
    result = run_task(app, task)
    assert result["feature_count"] == 1, result
    assert result["issue_count"] == 0, result
    assert not (root / "final/final_composite.gpkg").exists()
    published = task.publish()
    task.discard()
    assert Path(published["final_path"]).is_file()
    assert Path(published["issues_path"]).is_file()
    changed = RefinementTask(spec, value, final_path=published["final_path"], assemble=True,
                            transform_context=QgsCoordinateTransformContext())
    run_task(app, changed)
    # Unrelated input modification must fence publication, even after run().
    original = sha256_file(published["final_path"])
    Path(value["classes"][str(CLASS_ORDER[-1])]["path"]).touch()
    assert not changed.inputs_unchanged()
    try:
        changed.publish()
    except RuntimeError:
        pass
    else:
        raise AssertionError("stale result published")
    assert sha256_file(published["final_path"]) == original
    changed.discard()
    rollback = RefinementTask(spec, value, final_path=published["final_path"], assemble=True,
                             transform_context=QgsCoordinateTransformContext())
    run_task(app, rollback)
    from labeling_tool.core import refinement_task
    real_replace = refinement_task.os.replace
    failed = False
    originals = {key: sha256_file(path) for key, path in published.items()}
    def fail_second(source, destination):
        nonlocal failed
        if str(destination) == published["issues_path"] and not failed:
            failed = True
            raise OSError("injected rename failure")
        return real_replace(source, destination)
    with patch.object(refinement_task.os, "replace", side_effect=fail_second):
        try:
            rollback.publish()
        except OSError:
            pass
        else:
            raise AssertionError("failure was not injected")
    assert all(sha256_file(path) == originals[key] for key,path in published.items())
    rollback.discard()
    cancelled = RefinementTask(spec, value, final_path=published["final_path"], assemble=True,
                              transform_context=QgsCoordinateTransformContext())
    cancelled.cancel()
    assert not cancelled.run()
    assert all(sha256_file(path) == originals[key] for key,path in published.items())
    return dict(staged_publish=True, stale_input_rejected=True, rename_failure_rolled_back=True,
                cancelled_output_preserved=True)


def manual_candidates(app, root):
    """Exercise temporary pending bands on a native canvas without project data."""
    from qgis.gui import QgsMapCanvas, QgsRubberBand
    from labeling_tool.gui.class_refinement_dialog import ClassRefinementDialog as Dialog

    layer = memory_layer(1)
    canvas = QgsMapCanvas()
    canvas.setDestinationCrs(layer.crs())
    canvas.resize(480, 320)
    canvas.show()

    class Interface:
        def mapCanvas(self):
            return canvas

    class Harness:
        _new_manual_candidate_band = Dialog._new_manual_candidate_band
        _refresh_manual_pending_candidate_bands = Dialog._refresh_manual_pending_candidate_bands
        _clear_manual_add_candidate_bands = Dialog._clear_manual_add_candidate_bands
        _refresh_manual_modify_reference = Dialog._refresh_manual_modify_reference
        _clear_manual_reference_band = Dialog._clear_manual_reference_band
        _clear_manual_bands = Dialog._clear_manual_bands
        _manual_modify_selected_features = Dialog._manual_modify_selected_features

        def __init__(self):
            self.iface = Interface()
            self._manual_add_candidate_bands = []
            self._manual_reference_band = None
            self._manual_task = None

        def _layer(self, _class_code):
            return layer

        @staticmethod
        def _manual_smoothing_preview_is_current(_task):
            return False

    def geometry(xmin):
        value = QgsGeometry.fromRect(QgsRectangle(xmin, 0, xmin + 1, 1))
        value.convertToMultiType()
        return value

    owner = Harness()
    owner._manual_task = {
        "kind": "add",
        "class_code": 12,
        "target_code": 12,
        "pending_geometries": [geometry(2), geometry(4)],
        "pending_errors": ["topology conflict", ""],
    }
    owner._refresh_manual_pending_candidate_bands()
    initial_bands = list(owner._manual_add_candidate_bands)
    assert len(initial_bands) == 2
    assert all(band in canvas.scene().items() for band in initial_bands)

    owner._manual_task["pending_geometries"] = [geometry(6)]
    owner._manual_task["pending_errors"] = []
    owner._refresh_manual_pending_candidate_bands()
    assert len(owner._manual_add_candidate_bands) == 1
    assert all(band not in canvas.scene().items() for band in initial_bands)
    assert owner._manual_add_candidate_bands[0] in canvas.scene().items()

    feature_id = next(layer.getFeatures()).id()
    owner._manual_task = {
        "kind": "modify",
        "class_code": 12,
        "target_code": 12,
        "selected_feature_ids": [feature_id],
        "pending_geometries": [geometry(8)],
    }
    owner._refresh_manual_pending_candidate_bands()
    owner._refresh_manual_modify_reference()
    assert owner._manual_reference_band in canvas.scene().items()

    active_bands = [*owner._manual_add_candidate_bands, owner._manual_reference_band]
    owner._clear_manual_bands()
    assert owner._manual_add_candidate_bands == []
    assert owner._manual_reference_band is None
    assert all(band not in canvas.scene().items() for band in active_bands)
    assert not any(isinstance(item, QgsRubberBand) for item in canvas.scene().items())
    owner._clear_manual_bands()
    canvas.close()
    return {"pending_bands": 2, "refresh_removes_old_scene_items": True,
            "batch_and_reference_cleanup_is_idempotent": True}


def monitor_tables(app, root):
    """Content fitting stays bounded and respects interactive column widths."""
    from qgis.PyQt.QtCore import QEvent
    from qgis.PyQt.QtGui import QHelpEvent
    from qgis.PyQt.QtWidgets import QTableWidgetItem
    from labeling_tool.gui.monitor_theme import MONITOR_STYLE
    from labeling_tool.gui.monitor_widgets import AdaptiveTable

    table = AdaptiveTable(0, 4)
    table.setObjectName("OverviewResults")
    table.setHorizontalHeaderLabels(["结果", "当前工作", "组装", "验收"])
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(34)
    table.configure_adaptive_columns([150, 200, 125, 125], [1, 2, 1, 1])
    table.resize(900, 500)
    table.show()

    def settle_table():
        for _ in range(3):
            app.processEvents()

    def populate(count):
        table.setRowCount(count)
        for row in range(count):
            for column, value in enumerate((f"模型-{row}", "边界拟合", "尚未开始", "尚未执行")):
                table.setItem(row, column, QTableWidgetItem(value))
        table.fit_rows_to_content(min_rows=0, max_rows=8, empty_rows=1)
        table.request_adaptive_layout()
        settle_table()

    for theme in ("dark", "light"):
        table.setStyleSheet(MONITOR_STYLE[theme])
        for count in (0, 1, 4, 50):
            populate(count)
            visible_rows = max(1, min(count, 8))
            assert table.viewport().height() <= (visible_rows + 1) * 34
            if count <= 8:
                assert table.verticalScrollBar().maximum() == 0
            else:
                assert table.verticalScrollBar().maximum() > 0
            assert table.columnWidth(1) < table.viewport().width() * 0.45

    with patch.object(table, "item", wraps=table.item) as measured_items:
        populate(500)
        assert measured_items.call_count <= 4 * 32 + 8, measured_items.call_count

    header = table.horizontalHeader()
    manual_width = header.sectionSize(0) + 37
    header.resizeSection(0, manual_width)
    long_id = "package_" + "very_long_input_identifier_" * 20
    long_reason = "等待上游结果：" + "完整错误原因不可因自适应丢失；" * 30
    previous_work_width = header.sectionSize(1)
    table.item(0, 0).setText(long_id)
    table.item(0, 1).setText(long_reason)
    table.request_adaptive_layout()
    settle_table()
    assert header.sectionSize(0) == manual_width
    assert header.sectionSize(1) > previous_work_width
    assert table.item(0, 0).text() == long_id
    assert table.item(0, 1).text() == long_reason
    item = table.item(0, 0)
    position = table.visualItemRect(item).center()
    help_event = QHelpEvent(QEvent.Type.ToolTip, position, table.viewport().mapToGlobal(position))
    with patch("labeling_tool.gui.monitor_widgets.QToolTip.showText") as show_tooltip:
        assert table.viewportEvent(help_event)
        assert show_tooltip.call_args.args[1] == long_id
        item.setText("updated_identifier")
        assert table.viewportEvent(help_event)
        assert show_tooltip.call_args.args[1] == "updated_identifier"
        item.setToolTip("explicit business identifier")
        assert table.viewportEvent(help_event)
        assert show_tooltip.call_args.args[1] == "explicit business identifier"
    table.resize(400, table.height())
    settle_table()
    assert header.sectionSize(0) == manual_width
    assert table.horizontalScrollBar().maximum() > 0
    table.close()
    return {"content_row_counts": [0, 1, 4, 50], "bounded_sample_rows": 32,
            "manual_width_preserved": True, "long_values_preserved": True}


def monitor_typography(app, root):
    """Check actual Qt fonts/blocks, not only stylesheet declarations."""
    from qgis.PyQt.QtGui import QFont, QFontDatabase
    from qgis.PyQt.QtWidgets import QPlainTextEdit
    from labeling_tool.gui.log_panel import LogPanel, LOG_FONT_SIZE, LOG_BLOCK_LINE_HEIGHT
    from labeling_tool.gui.monitor_theme import BODY_FONT_PT, DETAIL_LINE_HEIGHT, MONITOR_STYLE
    from labeling_tool.gui.monitor_widgets import MonitorTextBrowser
    from labeling_tool.qt6_api import TEXT_LINE_PROPORTIONAL

    detail = MonitorTextBrowser()
    detail.setFont(QFont(app.font().family(), BODY_FONT_PT))
    detail.resize(640, 400)
    detail.show()
    for theme in ("dark", "light"):
        detail.setStyleSheet(MONITOR_STYLE[theme])
        for method in (detail.setText, detail.setPlainText):
            method("当前步骤：写入正式 GPKG\n等待上游结果\n完整中文详情")
            app.processEvents()
            assert detail.toPlainText().count("\n") == 2
            assert detail.font().pointSize() == BODY_FONT_PT
            assert detail.document().documentMargin() == 10
            block = detail.document().firstBlock()
            while block.isValid():
                assert block.blockFormat().lineHeight() == DETAIL_LINE_HEIGHT
                assert block.blockFormat().lineHeightType() == TEXT_LINE_PROPORTIONAL
                block = block.next()
        detail.setHtml('<p>本次尝试<br>模型明细</p><p><a href="attempt:span-2">第 2 次尝试</a></p>')
        app.processEvents()
        assert "attempt:span-2" in detail.toHtml()
        first = detail.document().firstBlock()
        assert first.layout().lineCount() == 2
        assert first.layout().lineAt(1).y() > first.layout().lineAt(0).height()
    detail.close()

    panel = LogPanel()
    panel.resize(900, 380)
    panel.show()
    for theme in ("dark", "light"):
        panel.set_theme(theme)
        for _ in range(3):
            app.processEvents()
        long_line = "package_0123：" + "完整日志不截断；" * 60
        panel.append_stdout(long_line + "\n下一行")
        panel.append_event("资源压力", source="system", severity="warning", title="自动降档", system_action="批量 16 → 8")
        panel.append_event("写入失败", source="stderr", severity="error", title="步骤失败", affected="推理包 23")
        for _ in range(3):
            app.processEvents()
        edit = panel.log_edit
        assert edit.font().pointSize() == LOG_FONT_SIZE
        assert edit.font().family() == panel.font().family()
        assert edit.font().letterSpacing() == 0
        assert edit.font().wordSpacing() == 0
        assert not edit.font().italic()
        assert edit.lineWrapMode() == QPlainTextEdit.LineWrapMode.NoWrap
        assert edit.document().documentMargin() == 9
        assert long_line in edit.toPlainText()
        block = edit.document().firstBlock()
        while block.isValid():
            if block.text():
                edit.document().documentLayout().blockBoundingRect(block)
                assert block.blockFormat().lineHeight() == LOG_BLOCK_LINE_HEIGHT
                fragment = block.begin().fragment()
                assert fragment.charFormat().fontPointSize() == LOG_FONT_SIZE
                assert block.layout().lineCount() == 1, (block.text(), block.layout().lineCount())
                fragments = block.begin()
                while not fragments.atEnd():
                    span = fragments.fragment()
                    span_font = span.charFormat().font()
                    assert span_font.letterSpacing() == 0
                    assert span_font.wordSpacing() == 0
                    assert not span_font.italic()
                    if "完整日志不截断" in span.text() or "系统处理" in span.text():
                        assert span_font.family() == panel.font().family()
                    fragments += 1
            block = block.next()
        first = edit.document().firstBlock()
        # QPlainTextEdit may retain a block-format value without honoring it;
        # require the actual laid-out block to have extra breathing room too.
        bounds = edit.document().documentLayout().blockBoundingRect(first)
        assert bounds.height() >= edit.fontMetrics().height() * 1.15, (
            bounds.height(), edit.fontMetrics().height(),
        )
        assert edit.horizontalScrollBar().maximum() > 0
    # A delayed UI append retains the capture timestamp; the UI clock is not
    # allowed to rewrite source history. Missing source time remains explicit.
    from labeling_tool.gui.monitor_time import format_monitor_timestamp
    captured_at = "2026-09-09T10:00:00.123456Z"
    panel.append_event("延迟抵达的日志", source="stdout", severity="info", event_timestamp=captured_at)
    assert format_monitor_timestamp(captured_at, compact=True) in panel.log_edit.toPlainText()
    assert panel._raw_records[-1]["source_timestamp"] == captured_at
    assert panel._raw_records[-1]["timestamp_kind"] == "captured"
    assert "· 接收]" in panel.log_edit.toPlainText()
    from qgis.PyQt.QtTest import QTest
    records_before_refresh = copy.deepcopy(panel._raw_records)
    panel.set_theme("dark")
    panel.set_visible_severities({"warning"})
    QTest.qWait(150)
    panel.set_visible_severities({"info", "warning", "error"})
    panel._btn_technical.setChecked(True)
    QTest.qWait(150)
    assert panel._raw_records == records_before_refresh
    assert format_monitor_timestamp(captured_at, compact=True) in panel.log_edit.toPlainText()
    # Verify clipboard/export without touching the user's real clipboard.
    with patch("labeling_tool.gui.log_panel.QApplication.clipboard") as clipboard:
        panel._on_copy()
        copied = clipboard.return_value.setText.call_args.args[0]
        assert "2026-09-09T10:00:00.123456+00:00" in copied
        assert "[采集] 延迟抵达的日志" in copied
        assert "[接收]" in copied
    saved_log = root / "monitor-export.log"
    with patch("labeling_tool.gui.log_panel.QFileDialog.getSaveFileName", return_value=(str(saved_log), "")):
        panel._on_save()
    assert saved_log.read_text() == copied + "\n"
    panel.close()
    return {"detail_line_height": DETAIL_LINE_HEIGHT, "log_line_height": LOG_BLOCK_LINE_HEIGHT,
            "body_font_pt": BODY_FONT_PT, "log_font_pt": LOG_FONT_SIZE}


def monitor_combos(app, root):
    from qgis.PyQt.QtCore import Qt
    from qgis.PyQt.QtGui import QColor, QPalette
    from qgis.PyQt.QtTest import QTest
    from qgis.PyQt.QtWidgets import QComboBox, QDialog, QVBoxLayout
    from labeling_tool.gui.monitor_widgets import MonitorComboBox
    from labeling_tool.gui.monitor_theme import MONITOR_STYLE, PALETTES
    application_style = app.styleSheet()
    application_palette = QPalette(app.palette())
    unrelated = QComboBox()
    unrelated_palette = QPalette(unrelated.palette())
    dialog = QDialog()
    dialog.setObjectName("InferenceMonitor")
    layout = QVBoxLayout(dialog)
    combo = MonitorComboBox()
    for title, value in (("全部事件", "all"), ("当前问题", "issues"),
                         ("自动恢复", "recovery"), ("历史警告／失败", "warnings")):
        combo.addItem(title, value)
    layout.addWidget(combo)
    dialog.resize(340, 180)
    dialog.show()
    output = Path(os.environ.get("LOESS_MONITOR_SCREENSHOT_DIR", root))
    output.mkdir(parents=True, exist_ok=True)

    def check_popup(theme, name):
        view = combo.view()
        popup = view.window()
        assert popup is not dialog
        assert popup.palette().color(QPalette.ColorRole.Window).name() == PALETTES[theme]["field"].lower()
        assert view.viewport().palette().color(QPalette.ColorRole.Base).name() == PALETTES[theme]["field"].lower()
        screenshot = popup.grab()
        assert screenshot.save(str(output / name))
        image = screenshot.toImage()
        colors = [QColor(PALETTES[theme][key]).getRgb()[:3] for key in ("field", "border")]
        edge_pixels = [(x, y) for x in range(image.width()) for y in (0, image.height() - 1)]
        edge_pixels += [(x, y) for y in range(image.height()) for x in (0, image.width() - 1)]
        for x, y in edge_pixels:
            actual = image.pixelColor(x, y).getRgb()[:3]
            assert all(min(a, b) - 2 <= value <= max(a, b) + 2
                       for value, a, b in zip(actual, *colors)), (theme, x, y, actual)

    for theme in ("dark", "light"):
        dialog.setStyleSheet(MONITOR_STYLE[theme])
        combo.apply_theme(theme)
        app.processEvents()
        before = combo.currentData()
        combo.showPopup()
        QTest.qWait(40)
        view = combo.view()
        assert view.isVisible()
        assert view.palette().color(QPalette.ColorRole.Base).name() == PALETTES[theme]["field"].lower()
        assert view.palette().color(QPalette.ColorRole.Text).name() == PALETTES[theme]["text"].lower()
        assert view.visualRect(view.model().index(0, 0)).height() >= 34
        check_popup(theme, f"dropdown-{theme}.png")
        QTest.keyClick(view, Qt.Key.Key_Escape)
        assert not view.isVisible()
        assert combo.currentData() == before
        combo.showPopup()
        QTest.keyClick(view, Qt.Key.Key_Down)
        QTest.keyClick(view, Qt.Key.Key_Return)
        app.processEvents()
        assert combo.currentData() != before
        assert combo.currentData() in {"issues", "recovery", "warnings"}
    # Inspect the whole native popup, not just the list: its parent frame and
    # margins previously stayed white even though the list palette was dark.
    for index in range(30):
        combo.addItem(f"历史执行 {index + 1} / 长列表滚动检查", f"execution-{index}")
    for theme in ("dark", "light", "dark"):
        dialog.setStyleSheet(MONITOR_STYLE[theme])
        combo.apply_theme(theme)
        combo.showPopup()
        QTest.qWait(40)
        view = combo.view()
        assert view.verticalScrollBar().maximum() > 0
        assert view.verticalScrollBar().isVisible()
        assert view.height() <= view.sizeHintForRow(0) * combo.maxVisibleItems() + 20
        view.scrollToBottom()
        app.processEvents()
        check_popup(theme, f"dropdown-long-{theme}.png")
        combo.hidePopup()
    assert app.styleSheet() == application_style
    assert app.palette() == application_palette
    assert unrelated.palette() == unrelated_palette
    assert not unrelated.styleSheet()
    unrelated.close()
    dialog.close()
    return {"popup_theme_colors": True, "popup_edges": True, "long_list": True,
            "keyboard_selection": True, "escape_preserves_filter": True}


def monitor(app, root):
    """Render the four-page native monitor and exercise responsive reflow."""

    from qgis.PyQt.QtCore import QEventLoop, QPoint, QRect, Qt
    from qgis.PyQt.QtWidgets import QLabel, QPushButton, QWidget
    from labeling_tool.gui.inference_monitor import InferenceMonitorDialog

    output = Path(os.environ.get("LOESS_MONITOR_SCREENSHOT_DIR") or root / "monitor")
    output.mkdir(parents=True, exist_ok=True)
    application_style = app.styleSheet()

    class LargeScreen:
        def availableGeometry(self):
            return QRect(0, 0, 1920, 1200)

    class ProbeParent(QWidget):
        def screen(self):
            return LargeScreen()

    # Exercise the large-screen default without weakening production's
    # available-geometry clamp for a small desktop.
    screen_parent = ProbeParent()
    global _ACTIVE_MONITOR_DIALOG
    dialog = InferenceMonitorDialog(screen_parent)
    _ACTIVE_MONITOR_DIALOG = dialog
    default_size = dialog.size()
    for obsolete in ("_bar", "_summary", "_stage_rail", "_package_overview",
                     "_fit_label", "_run_overview", "_assembly_overview", "_coverage_overview"):
        assert not hasattr(dialog, obsolete), obsolete
    assert dialog._run_information == "Run：准备中"
    assert dialog._coverage_information == "空白/重叠验收：等待组装"
    # Disk-history navigation must work even with no retained warning/error
    # cache and must not dispatch a database query in this isolated UI probe.
    original_dispatch = dialog._dispatch_next_query
    dialog._dispatch_next_query = lambda: None
    dialog._database_bound = True
    dialog._run_id = "history-probe"
    dialog._update_log_toggle()
    assert dialog._error_log_button.isEnabled()
    dialog._error_log_button.click()
    assert dialog._history_scope.currentData() == "raw_error"
    assert dialog._pending_history_query["scope"] == "raw_error"
    assert not dialog._history_execution.isEnabled()
    dialog._log_panel._select_severity("warning")
    assert dialog._pending_history_query["scope"] == "raw_warning"
    dialog._apply_history_result({"raw_log": True, "rows": [], "next_cursor": 123, "has_more": True})
    assert dialog._history_cursor == 123
    assert dialog._history_load_older.isEnabled()
    assert "不代表整个 Run" in dialog._history_detail.toPlainText()
    dialog._load_older_history()
    assert dialog._pending_history_query["before_event_id"] == 123
    dialog._active_query = dict(dialog._pending_history_query)
    previous_sync_error = dialog._last_snapshot_error
    dialog._on_query_failed({**dialog._active_query, "error": "log file missing"})
    assert dialog._last_snapshot_error == previous_sync_error
    assert "不改变任务或数据库连接状态" in dialog._history_detail.toPlainText()
    dialog._history_scope.setCurrentIndex(0)
    dialog._database_bound = False
    dialog._run_id = ""
    dialog._pending_history_query = None
    dialog._dispatch_next_query = original_dispatch
    dialog._pages.setCurrentIndex(0)
    dialog.set_stage_progress({"name": "准备影像", "current": 1, "total": 4})
    assert dialog._completion_value.text() == "25%"
    assert dialog._progress_title.text() == "启动准备 · 准备影像"
    dialog.set_stage_progress({"name": "检查输入", "current": 1, "total": 1})
    assert dialog._completion_value.text() == "100%"
    assert "当前准备步骤" in dialog._progress_hint.text()
    dialog.set_stage_progress({"name": "准备运行"})
    assert dialog._completion_value.text() == "—"
    assert dialog._overall_bar.maximum() == 0
    dialog.mark_stopping()
    dialog.set_stage_progress({"name": "迟到的进度", "current": 1, "total": 2})
    assert dialog._completion_value.text() == "—"
    dialog._control_state = ""
    dialog._stop.setEnabled(True)
    dialog._stop.setText("停止任务")
    dialog._run_id = "DEMO-001"
    dialog._run_spec = {
        "runtime": {"effective_device": "mps"},
        "models": [
            {"model_id": "swin_b", "display_name": "Swin-B"},
            {"model_id": "mambaout_b", "display_name": "MambaOut-B"},
            {"model_id": "setr_vit", "display_name": "SETR-ViT"},
        ],
        "fusion": {"profile_id": "ensemble", "display_name": "融合结果"},
        "fragmentation_regularization": {"enabled": True},
        "boundary_fitting": {"enabled": True},
        "scaling": {"max_cpu_partition_workers_with_package": 4,
                    "max_concurrent_assembly": 2},
    }
    demo_run_id = dialog._run_id
    demo_run_spec = copy.deepcopy(dialog._run_spec)
    now = "2026-09-09T10:00:00+00:00"
    stream_ids = (
        "model:swin_b", "model:mambaout_b", "model:setr_vit", "fusion:ensemble",
    )
    streams = [{"stream_id": stream_id, "status": "running"} for stream_id in stream_ids]
    snapshot = {
        "run": {"status": "running", "created_at": now},
        "job_counts": {
            "work_package": {"ready": 23, "running": 1, "queued": 24},
            "fragmentation_v33": {"ready": 12, "running": 2, "queued": 34},
            "unit_confidence": {"ready": 80, "running": 4, "queued": 156},
            "unit_fit": {"ready": 612, "running": 16, "queued": 392},
        },
        "job_progress": {"work_package": {"completed": 23.75, "total": 48}},
        "active_work_package": {
            "package_id": "package_023", "sequence_no": 23, "attempt": 2,
            "progress_current": 384, "progress_total": 512,
            "package_started_at": now,
        },
        "streams": streams,
        "stream_unit_type_counts": {
            stream["stream_id"]: {"core": {"ready": 153, "queued": 102}}
            for stream in streams
        },
        "stream_unit_job_type_counts": {
            stream["stream_id"]: {"core": {"ready": 153, "running": 4, "queued": 98}}
            for stream in streams
        },
        # A running sample has concurrent upstream work only.  It must not
        # advertise assembly before all source streams are ready.
        "stream_runtime_progress": {},
        "assembly_phase_statuses": {},
        "monitor_history": {
            "available": True,
            "latest_execution": {
                "execution_id": "exec-demo-resume-002", "trigger_type": "resume",
                "status": "running", "started_at": now, "recording_complete": True,
            },
            "executions": [
                {"execution_id": "exec-demo-resume-002", "trigger_type": "resume",
                 "status": "running", "started_at": now},
                {"execution_id": "exec-demo-start-001", "trigger_type": "start",
                 "status": "interrupted", "started_at": "2026-09-09T09:20:00+00:00"},
            ],
            "span_status_counts": {"completed": 30, "running": 20,
                                    "failed": 1, "interrupted": 1},
            "recent_events": [
                {"timestamp": now, "message": "推理包 23 已完成"},
                {"timestamp": now, "message": "单元重试已启动"},
                {"timestamp": now, "message": "边界拟合继续执行"},
            ],
        },
    }
    def snapshot_for(state):
        """Return a believable concurrent state without inventing early assembly."""

        # Each state owns its nested history/counts so terminal mutations do
        # not leak into the subsequent running/ready samples.
        value = copy.deepcopy(snapshot)
        value["run"] = dict(snapshot["run"], status=state)
        if state == "running":
            statuses = ("running", "running", "running", "running")
            counts = snapshot["job_counts"]
            active_package = dict(snapshot["active_work_package"])
            runtime = {}
            phases = {}
        elif state == "failed":
            statuses = ("failed", "stopped", "stopped", "stopped")
            counts = {
                **snapshot["job_counts"],
                "work_package": {"ready": 23, "failed": 1, "queued": 24},
                "fragmentation_v33": {"ready": 12, "failed": 2, "queued": 34},
                "unit_confidence": {"ready": 80, "failed": 4, "queued": 156},
                "unit_fit": {"ready": 612, "failed": 16, "queued": 392},
            }
            active_package = None
            runtime = {}
            phases = {}
        elif state == "stopped":
            statuses = ("stopped", "stopped", "stopped", "stopped")
            counts = {
                **snapshot["job_counts"],
                "work_package": {"ready": 23, "queued": 25},
                "fragmentation_v33": {"ready": 12, "queued": 36},
                "unit_confidence": {"ready": 80, "queued": 160},
                "unit_fit": {"ready": 612, "queued": 408},
            }
            active_package = None
            runtime = {}
            phases = {}
        else:
            assert state == "ready"
            statuses = ("ready", "ready", "ready", "ready")
            counts = {
                key: {"ready": sum(int(count) for count in values.values())}
                for key, values in snapshot["job_counts"].items()
            }
            active_package = None
            runtime = {
                stream_id: {
                    "status": "completed", "phase": "acceptance",
                    "phase_name": "整体验收", "phase_index": 10,
                    "phase_total": 10, "progress_current": 1,
                    "progress_total": 1, "feature_count": 128430,
                    "phase_started_at": now, "message": "验收通过",
                }
                for stream_id in stream_ids
            }
            phases = {
                stream_id: {"acceptance": {"status": "completed", "current": 1, "total": 1}}
                for stream_id in stream_ids
            }
        value["streams"] = [
            {"stream_id": stream_id, "status": stream_status}
            for stream_id, stream_status in zip(stream_ids, statuses)
        ]
        value["job_counts"] = counts
        value["job_progress"] = {
            "work_package": {"completed": 48 if state == "ready" else 23.75, "total": 48}
        }
        value["active_work_package"] = active_package
        value["stream_runtime_progress"] = runtime
        value["assembly_phase_statuses"] = phases
        history = value["monitor_history"]
        if state == "failed":
            history["latest_execution"] = {
                "execution_id": "exec-demo-resume-002", "trigger_type": "resume",
                "status": "failed", "started_at": now, "recording_complete": True,
            }
            history["executions"] = [history["latest_execution"], {
                "execution_id": "exec-demo-start-001", "trigger_type": "start",
                "status": "interrupted", "started_at": "2026-09-09T09:20:00+00:00",
            }]
            history["span_status_counts"] = {"completed": 30, "running": 0,
                                               "failed": 5, "interrupted": 1}
            history["recent_events"] = [
                {"timestamp": now, "message": "推理包 23 执行失败"},
                {"timestamp": now, "message": "边界拟合任务失败"},
            ]
        elif state == "stopped":
            history["latest_execution"] = {
                "execution_id": "exec-demo-resume-002", "trigger_type": "resume",
                "status": "stopped", "started_at": now, "recording_complete": True,
            }
            history["executions"] = [history["latest_execution"], {
                "execution_id": "exec-demo-start-001", "trigger_type": "start",
                "status": "interrupted", "started_at": "2026-09-09T09:20:00+00:00",
            }]
            history["span_status_counts"] = {"completed": 30, "running": 0,
                                               "failed": 0, "interrupted": 5}
            history["recent_events"] = [
                {"timestamp": now, "message": "用户停止当前任务"},
                {"timestamp": now, "message": "任务组已安全停止"},
            ]
        elif state == "ready":
            history["latest_execution"] = {
                "execution_id": "exec-demo-resume-002", "trigger_type": "resume",
                "status": "completed", "started_at": now, "recording_complete": True,
            }
            history["executions"] = [history["latest_execution"], {
                "execution_id": "exec-demo-start-001", "trigger_type": "start",
                "status": "interrupted", "started_at": "2026-09-09T09:20:00+00:00",
            }]
            history["span_status_counts"] = {"completed": 52, "running": 0,
                                               "failed": 0, "interrupted": 1}
            history["recent_events"] = [
                {"timestamp": now, "message": "全部结果流完成"},
                {"timestamp": now, "message": "空白/重叠验收通过"},
            ]
            value["stream_coverage_validation"] = {
                stream_id: {
                    "status": "passed", "gap_area_m2": 0.0,
                    "overlap_area_m2": 0.0, "outside_area_m2": 0.0,
                    "feature_count": 128430, "validated_at": now,
                }
                for stream_id in stream_ids
            }
        return value

    def reset_sample():
        """Reset widget lifecycle while restoring the frozen demo identity."""
        dialog.reset_run()
        dialog._run_id = demo_run_id
        dialog._run_spec = copy.deepcopy(demo_run_spec)

    def apply_running_sample():
        """Rebuild running state, then apply the current package attempt."""
        reset_sample()
        dialog._apply_database_snapshot(snapshot_for("running"))
        dialog._package_activity.update({
            "stream_id": "model:swin_b", "status": "Swin-B 推理",
            "tile_current": 384, "tile_total": 512,
            "configured_batch_size": 16, "effective_batch_size": 8,
        })
        dialog._apply_database_snapshot(snapshot_for("running"))

    apply_running_sample()
    dialog.show()
    app.processEvents()

    saved = []
    def settle():
        # Theme and splitter changes can require multiple deferred Qt layout
        # passes, but keep the probe deterministic and bounded.
        for _ in range(3):
            app.processEvents()

    def grab(name):
        path = output / name
        assert dialog.grab().save(str(path)), path
        assert path.stat().st_size > 1000, path
        saved.append(str(path))

    def grab_widget(widget, name):
        path = output / name
        assert widget.grab().save(str(path)), path
        assert path.stat().st_size > 1000, path
        saved.append(str(path))

    # The icon communicates the destination theme and remains a named control
    # for keyboard/screen-reader users. Mock settings writes in this UI probe.
    with patch("labeling_tool.gui.inference_monitor.QgsSettings") as settings:
        dialog._apply_theme("dark", persist=False)
        assert dialog._theme_toggle.text() == ""
        assert not dialog._theme_toggle.icon().isNull()
        assert dialog._theme_toggle.toolTip() == "切换浅色主题"
        assert dialog._theme_toggle.accessibleName() == "切换浅色主题"
        sun_key = dialog._theme_toggle.icon().cacheKey()
        dialog._theme_toggle.click()
        assert dialog._theme == "light"
        assert dialog._theme_toggle.accessibleName() == "切换深蓝主题"
        assert dialog._theme_toggle.icon().cacheKey() != sun_key
        assert settings.return_value.setValue.call_args.args[1] == "light"
        dialog._theme_toggle.click()
        assert dialog._theme == "dark"
        assert settings.return_value.setValue.call_args.args[1] == "dark"
    settle()

    # Overall progress is shared across all pages, with opt-out motion.
    assert not dialog._overview_results_panel.isAncestorOf(dialog._overall_bar)
    assert dialog._overall_bar.height() == 14
    for page_index in range(4):
        dialog._pages.setCurrentIndex(page_index)
        settle()
        assert dialog._overall_bar.isVisible()
    dialog._pages.setCurrentIndex(0)
    dialog._overall_bar.set_running(True)
    assert dialog._overall_bar._timer.isActive()
    dialog._overall_bar.set_reduced_motion(True)
    assert not dialog._overall_bar._timer.isActive()
    dialog._overall_bar.set_reduced_motion(False)
    dialog._overall_bar.set_running(False)
    assert not dialog._overall_bar._timer.isActive()
    dialog._last_snapshot_at = time.time() - 20
    dialog._refresh_sync_status()
    assert "数据暂未更新" in dialog._monitor_sync.text()
    dialog._last_snapshot_error = "fixture read failure"
    dialog._refresh_sync_status()
    assert "同步失败" in dialog._monitor_sync.text()
    dialog._last_snapshot_error = ""
    dialog._last_snapshot_at = time.time()
    dialog._refresh_sync_status()
    assert "数据同步正常" in dialog._monitor_sync.text()
    from qgis.PyQt.QtGui import QPalette
    from labeling_tool.gui.monitor_theme import PALETTES
    for theme in ("dark", "light"):
        dialog._apply_theme(theme, persist=False)
        info = dialog._build_run_information_dialog()
        info.show()
        settle()
        from labeling_tool.gui.monitor_widgets import MonitorTextBrowser
        info_text = info.findChild(MonitorTextBrowser).toPlainText()
        for text in (dialog._run_information, dialog._assembly_information, dialog._coverage_information):
            assert isinstance(text, str)
            assert text.replace(" | ", "\n") in info_text
        for field in ("DEMO-001", "设备", "创建至今", "本次监控", "结果流组装", "空白/重叠验收"):
            assert field in info_text
        assert info.width() >= min(800, info.screen().availableGeometry().width() - 40)
        assert info.palette().color(QPalette.ColorRole.Window).name() == PALETTES[theme]["background"].lower()
        grab_widget(info, f"monitor-run-information-{theme}.png")
        info.close()
        info.deleteLater()
    dialog._apply_theme("dark", persist=False)
    settle()

    # A known one-tile workload must render as complete; an unknown total must
    # remain the empty sentinel rather than a false 100%.
    dialog._package_activity.update({"tile_current": 1, "tile_total": 1})
    dialog._apply_database_snapshot(snapshot_for("running"))
    settle()
    assert dialog._tile_card_bar.maximum() == 1000
    assert dialog._tile_card_bar.value() == 1000
    assert "配置/有效 Batch" in dialog._current_package.toolTip()
    assert "包耗时" in dialog._current_package.toolTip()
    assert "按任务计数" in dialog._fit_bar.toolTip()
    dialog._package_activity.update({"tile_current": 0, "tile_total": 0})
    dialog._apply_database_snapshot(snapshot_for("running"))
    settle()
    assert dialog._tile_card_bar.maximum() == 1
    assert dialog._tile_card_bar.value() == 0
    dialog._package_activity.update({"tile_current": 384, "tile_total": 512})
    dialog._apply_database_snapshot(snapshot_for("running"))
    settle()

    def global_rect(widget):
        return QRect(widget.mapToGlobal(QPoint(0, 0)), widget.size())

    def assert_inside(child, parent):
        assert global_rect(parent).contains(global_rect(child)), (
            child.objectName() or child.text(), global_rect(child), global_rect(parent)
        )

    def assert_content_sized_table(table, expected_rows):
        assert table.rowCount() == expected_rows
        row_height = table.verticalHeader().defaultSectionSize()
        assert table.viewport().height() <= (max(1, expected_rows) + 1) * row_height, (
            table.objectName(), table.viewport().height(), expected_rows,
        )

    def assert_overview_frame():
        """Verify whole panels, not only their last readable row or button."""
        from labeling_tool.gui.monitor_theme import (
            BODY_FONT_PT, HEADLINE_FONT_PT, METRIC_FONT_PT, SECTION_FONT_PT,
            TABLE_HEADER_MIN_HEIGHT, TABLE_ROW_MIN_HEIGHT, VALUE_FONT_PT,
        )
        for table in (dialog._overview_results, dialog._streams, dialog._tiles,
                      dialog._history_table, dialog._assembly_steps):
            assert table.font().pointSize() == BODY_FONT_PT
            assert table.horizontalHeader().height() >= TABLE_HEADER_MIN_HEIGHT
            assert table.verticalHeader().defaultSectionSize() >= TABLE_ROW_MIN_HEIGHT
            assert table.verticalHeader().defaultSectionSize() >= table.fontMetrics().height() + 14
        for label in dialog.findChildren(QLabel):
            for role, size in (("headline", HEADLINE_FONT_PT), ("metric", METRIC_FONT_PT),
                               ("sectionTitle", SECTION_FONT_PT), ("value", VALUE_FONT_PT)):
                if label.property(role):
                    assert label.font().pointSize() == size, (role, label.text(), label.font().pointSize())
        viewport = dialog._pages.widget(0).viewport()
        results_panel = dialog._overview_results_panel
        parallel_panel = dialog._overview_model_card.parentWidget()
        activity_panel = dialog._overview_activity_panel
        for panel in (parallel_panel, results_panel, activity_panel):
            assert_inside(panel, viewport)
        assert abs(global_rect(results_panel).bottom() - global_rect(activity_panel).bottom()) <= 2
        assert global_rect(dialog._overall_bar).top() > global_rect(results_panel).bottom()
        assert not global_rect(results_panel).intersects(global_rect(activity_panel))
        assert global_rect(results_panel).top() - global_rect(parallel_panel).bottom() >= 16
        assert dialog._pages.widget(0).verticalScrollBar().maximum() == 0

        table = dialog._overview_results
        assert_content_sized_table(table, 4)
        assert table.columnWidth(1) <= table.viewport().width() * 0.45, (
            table.columnWidth(1), table.viewport().width(),
        )
        header = table.horizontalHeader()
        for column in range(table.columnCount()):
            header_item = table.horizontalHeaderItem(column)
            assert header_item.textAlignment() & Qt.AlignmentFlag.AlignLeft
            for row in range(table.rowCount()):
                item = table.item(row, column)
                assert item.textAlignment() & Qt.AlignmentFlag.AlignLeft
                cell = table.visualRect(table.model().index(row, column))
                header_x = header.viewport().mapToGlobal(
                    QPoint(header.sectionViewportPosition(column), 0)
                ).x()
                cell_x = table.viewport().mapToGlobal(cell.topLeft()).x()
                assert abs(header_x - cell_x) <= 1, (column, header_x, cell_x)

    def assert_card_geometry(expected_compact):
        model_card = dialog._overview_model_card
        spatial_card = dialog._overview_spatial_card
        assert model_card.isVisible() and spatial_card.isVisible()
        assert not global_rect(model_card).intersects(global_rect(spatial_card))
        cards = dialog._overview_cards
        if expected_compact:
            usable_width = cards.geometry().width()
            assert model_card.width() >= usable_width - 4, (model_card.width(), usable_width)
            assert spatial_card.width() >= usable_width - 4, (spatial_card.width(), usable_width)
        for card in (model_card, spatial_card):
            for label in card.findChildren(QLabel):
                if label.isVisible():
                    assert_inside(label, card)
                    # A card-level bounding check can miss clipping by an
                    # intermediate panel/scroll child.  Verify every actual
                    # paint-parent boundary up to the card itself.
                    child = label
                    while child is not card:
                        parent = child.parentWidget()
                        assert parent is not None, label.text()
                        assert global_rect(parent).contains(global_rect(child)), (
                            label.text(), child.objectName(), global_rect(child),
                            global_rect(parent),
                        )
                        child = parent
                    assert label.height() >= label.fontMetrics().height(), label.text()
                    # Icon labels are intentionally square and have no text;
                    # only textual labels need a readable line width.
                    if label.text().strip():
                        assert label.width() >= min(80, label.sizeHint().width()), (
                            label.text(), label.width(), label.sizeHint().width()
                        )
            for bar in (
                dialog._package_card_bar,
                dialog._tile_card_bar,
                dialog._fit_bar,
            ):
                if bar.isVisible() and bool(getattr(bar, "_percentage", False)):
                    assert bar.height() >= bar.fontMetrics().height(), (
                        bar.objectName(), bar.height(), bar.fontMetrics().height()
                    )
            for button in card.findChildren(QPushButton):
                if button.isVisible():
                    assert_inside(button, card)
                    assert button.height() >= 22, button.text()
                    assert button.width() >= button.minimumSizeHint().width(), button.text()

    # Full-width overview at the new default must show every result stream,
    # while the stop control and all four tab labels stay in the viewport.
    assert dialog._pages.currentIndex() == 0
    assert dialog._overview_results.rowCount() == 4
    rows_visible = (
        (dialog._overview_results.viewport().height())
        // dialog._overview_results.verticalHeader().defaultSectionSize()
    )
    assert rows_visible >= 4, rows_visible
    assert_inside(dialog._stop, dialog)
    assert dialog._stop.isVisible() and dialog._pages.tabBar().isVisible()
    assert dialog._pages.tabBar().count() == 4
    assert_card_geometry(expected_compact=False)
    assert app.styleSheet() == application_style

    # The standard laptop layout remains horizontal and non-overlapping.
    dialog.resize(1280, 820)
    app.processEvents()
    assert dialog._compact_layout is False
    assert dialog._overview_splitter.orientation().name == "Horizontal"
    assert_card_geometry(expected_compact=False)
    assert_inside(dialog._stop, dialog)
    assert dialog._pages.tabBar().isVisible()
    dialog._pages.setCurrentIndex(0)
    dialog._apply_theme("dark", persist=False)
    app.processEvents()
    grab("monitor-overview-1280-dark.png")
    dialog.resize(1680, 1040)
    settle()
    assert_card_geometry(expected_compact=False)
    overview_scroll = dialog._pages.widget(0)
    last_row = dialog._overview_results.visualRect(
        dialog._overview_results.model().index(3, 0)
    )
    last_row_top = dialog._overview_results.viewport().mapTo(
        overview_scroll.viewport(), QPoint(last_row.left(), last_row.top())
    )
    last_row_bottom = dialog._overview_results.viewport().mapTo(
        overview_scroll.viewport(), QPoint(last_row.right(), last_row.bottom())
    )
    assert overview_scroll.viewport().rect().contains(last_row_top)
    assert overview_scroll.viewport().rect().contains(last_row_bottom)
    event_buttons = [
        button
        for button in dialog._overview_activity_panel.findChildren(QPushButton)
        if "打开事件与日志" in button.text()
    ]
    assert len(event_buttons) == 1
    assert global_rect(overview_scroll.viewport()).contains(
        global_rect(event_buttons[0])
    ), (global_rect(event_buttons[0]), global_rect(overview_scroll.viewport()))
    assert_overview_frame()

    # Record the four terminal/user-visible states under both dialog themes.
    # The running state deliberately has no assembly runtime progress while
    # upstream streams are incomplete.
    stop_signal_count = []
    dialog.stop_requested.connect(lambda: stop_signal_count.append(True))
    dialog._request_stop()
    assert len(stop_signal_count) == 1
    terminal_labels = {"failed": "失败", "stopped": "已停止", "ready": "已完成"}
    for index, state in enumerate(("running", "failed", "stopped", "ready")):
        if index:
            # Reset through the dialog's real lifecycle before presenting the
            # next execution; this clears the terminal callback-generation
            # fence instead of mutating private widget state.
            reset_sample()
        dialog._apply_database_snapshot(snapshot_for(state))
        app.processEvents()
        expected_history_status = {
            "running": "running", "failed": "failed",
            "stopped": "stopped", "ready": "completed",
        }[state]
        assert dialog._latest_execution["status"] == expected_history_status
        if state == "ready":
            assert len(dialog._coverage_state) == len(stream_ids)
            assert all(item["status"] == "passed" for item in dialog._coverage_state.values())
        if state in terminal_labels:
            dialog.mark_finished(terminal_labels[state])
            settle()
            assert not dialog._stop.isEnabled()
        if state == "running":
            assert not dialog._runtime_progress
            assert all(stream["status"] != "assembling" for stream in snapshot_for(state)["streams"])
        for theme in ("dark", "light"):
            dialog._apply_theme(theme, persist=False)
            settle()
            assert_overview_frame()
            grab(f"monitor-{state}-{theme}.png")
            assert app.styleSheet() == application_style

    apply_running_sample()
    dialog._apply_theme("dark", persist=False)
    app.processEvents()
    # Selecting a row on the overview must select the same authoritative
    # stream in the detail/results pages, not retain the previous selection.
    dialog._overview_results.selectRow(1)
    app.processEvents()
    assert dialog._selected_stream() == "model:mambaout_b"
    assert dialog._streams.currentRow() == 1
    assert str(dialog._overview_results.item(1, 0).data(Qt.ItemDataRole.UserRole)) == "model:mambaout_b"

    dialog._pages.setCurrentIndex(1)
    bulk_rows = [
        {"object_id": f"package_{index:05d}", "object_label": "推理包",
         "execution_status": "ready" if index < 320 else "queued",
         "artifact_status": "ready" if index < 320 else "queued",
         "progress_current": 512 if index < 320 else 0,
         "progress_total": 512, "reason": ""}
        for index in range(500)
    ]
    started = time.perf_counter()
    dialog._apply_detail_result({
        "stream_id": "model:mambaout_b", "detail_kind": "package", "status": "",
        "search": "", "total": 500, "page": 0, "page_total": 1,
        "rows": bulk_rows,
    })
    app.processEvents()
    detail_update_ms = (time.perf_counter() - started) * 1000
    assert detail_update_ms < 200, detail_update_ms
    dialog._apply_detail_result({
        "stream_id": "model:mambaout_b", "detail_kind": "package", "status": "",
        "search": "", "total": 2, "page": 0, "page_total": 1,
        "rows": [
            {"object_id": "package_023", "object_label": "推理包",
             "execution_status": "running", "artifact_status": "queued",
             "progress_current": 384, "progress_total": 512,
             "execution_id": "exec-demo-resume-002", "span_id": "span-023",
             "budget_attempt": 2, "reason": "内存不足后自动降档"},
            {"object_id": "package_022", "object_label": "推理包",
             "execution_status": "ready", "artifact_status": "ready",
             "progress_current": 512, "progress_total": 512, "reason": ""},
        ],
    })
    app.processEvents()
    assert "MambaOut-B" in dialog._tile_detail_title.text()
    settle()
    assert_content_sized_table(dialog._tiles, 2)
    grab("monitor-detail-dark.png")
    dialog._pages.setCurrentIndex(2)
    settle()
    assert_content_sized_table(dialog._streams, 4)
    grab("monitor-results-dark.png")
    dialog._pages.setCurrentIndex(3)
    dialog._apply_history_result({
        "rows": [
            {"monitor_event_id": 3, "timestamp": now, "object_id": "unit_0012",
             "object_type": "job", "event_type": "job_retry_started",
             "message": "第 2 次执行开始", "execution_id": "exec-demo-resume-002",
             "span_id": "span-2", "level": "warning", "payload": {"attempt": 2}},
            {"monitor_event_id": 2, "timestamp": (datetime.fromisoformat(now) - timedelta(seconds=2)).isoformat(), "object_id": "unit_0012",
             "object_type": "job", "event_type": "job_failed",
             "message": "第 1 次执行失败", "execution_id": "exec-demo-start-001",
             "span_id": "span-1", "level": "error", "payload": {},
             "recovered_by_span_id": "span-2"},
        ],
        "append": False, "page_size": 200,
    })
    settle()
    assert_content_sized_table(dialog._history_table, 2)
    dialog._history_table.selectRow(0)
    from labeling_tool.gui.monitor_time import format_monitor_timestamp
    assert dialog._history_table.item(0, 0).text() == format_monitor_timestamp(now, compact=True)
    assert now in dialog._history_table.item(0, 0).toolTip()
    assert format_monitor_timestamp(now) in dialog._history_detail.toPlainText()
    assert "本机时间（UTC" in dialog._history_context_label.text()
    grab("monitor-events-dark.png")
    # Use the same sample timeline on both sides; never mix a historical
    # fixture date with the screenshot machine's current wall clock.
    log_capture = datetime.fromisoformat(now) - timedelta(seconds=3)
    dialog._on_log_batch([{
        "source": "stdout", "message": "推理包 23：模型计算完成，等待关联空间任务。",
        "timestamp": log_capture.timestamp(),
    }])
    dialog._log_panel.append_event(
        "内存压力后自动降档", source="system", severity="warning",
        title="批量大小调整", affected="推理包 23 / Swin-B",
        system_action="配置 16 → 当前有效 8", user_action="系统继续执行；可查看相关事件。",
        event_timestamp=(datetime.fromisoformat(now) - timedelta(seconds=1)).isoformat(),
    )
    dialog._log_toggle.setChecked(True)
    from qgis.PyQt.QtTest import QTest
    # Opening logs deliberately coalesces its refresh for 100ms. Wait for the
    # actual signal/timer path instead of capturing a not-yet-populated panel.
    QTest.qWait(150)
    settle()
    assert "批量大小调整" in dialog._log_panel.log_edit.toPlainText()
    assert format_monitor_timestamp(log_capture, compact=True) in dialog._log_panel.log_edit.toPlainText()
    grab("monitor-logs-dark.png")
    dialog._apply_theme("light", persist=False)
    QTest.qWait(150)
    settle()
    grab("monitor-logs-light.png")
    dialog._log_toggle.setChecked(False)

    dialog._apply_theme("light", persist=False)
    dialog.resize(900, 560)
    settle()
    assert dialog._compact_layout is True
    assert dialog._overview_splitter.orientation().name == "Vertical"
    dialog._pages.setCurrentIndex(0)
    settle()
    assert_card_geometry(expected_compact=True)
    assert_inside(dialog._stop, dialog)
    assert dialog._pages.tabBar().isVisible()
    assert app.styleSheet() == application_style
    grab_widget(dialog._overview_model_card, "monitor-compact-model-light.png")
    grab_widget(dialog._overview_spatial_card, "monitor-compact-spatial-light.png")
    grab("monitor-overview-900-light.png")
    grab("monitor-narrow-light.png")

    # A resize burst must settle to the wide layout even when no event loop
    # turn occurs between the two resize requests; stale compact geometry is
    # not an acceptable terminal state.
    dialog.resize(900, 560)
    dialog.resize(1680, 1040)
    settle()
    assert dialog._compact_layout is False
    assert dialog._overview_splitter.orientation().name == "Horizontal"
    rapid_scroll = dialog._pages.widget(0).viewport()
    rapid_row = dialog._overview_results.visualRect(
        dialog._overview_results.model().index(3, 0)
    )
    rapid_top = dialog._overview_results.viewport().mapTo(
        rapid_scroll, QPoint(rapid_row.left(), rapid_row.top())
    )
    rapid_bottom = dialog._overview_results.viewport().mapTo(
        rapid_scroll, QPoint(rapid_row.right(), rapid_row.bottom())
    )
    assert rapid_scroll.rect().contains(rapid_top)
    assert rapid_scroll.rect().contains(rapid_bottom)
    for theme in ("dark", "light"):
        dialog._apply_theme(theme, persist=False)
        settle()
        assert dialog._compact_layout is False
        assert dialog._overview_splitter.orientation().name == "Horizontal"
        assert_overview_frame()
    # Automatic refresh must preserve a width explicitly set through the
    # same header API used by native interactive resizing.
    header = dialog._overview_results.horizontalHeader()
    manual_width = header.sectionSize(0) + 37
    header.resizeSection(0, manual_width)
    dialog._apply_database_snapshot(snapshot_for("running"))
    settle()
    assert header.sectionSize(0) == manual_width
    # Completed/stopped state and resetting must not depend on deleted widgets.
    dialog.mark_finished("已完成")
    assert dialog._completion_value.text() == "100%"
    assert not dialog._overall_bar._timer.isActive()
    dialog.set_stage_progress({"name": "迟到的准备进度", "current": 1, "total": 4})
    assert dialog._completion_value.text() == "100%"
    dialog.reset_run()
    assert dialog._run_information == "Run：准备中"
    assert dialog._assembly_information == "结果流组装：等待上游计算"
    assert dialog._coverage_information == "空白/重叠验收：等待组装"
    assert dialog._completion_value.text() == "—"
    dialog.mark_finished("已停止")
    assert not dialog._overall_bar._timer.isActive()
    dialog.close()
    app.processEvents()
    assert not dialog.isVisible()

    loop = QEventLoop()
    dialog.shutdown_finished.connect(loop.quit)
    dialog.shutdown()
    deadline = QTimer()
    deadline.setSingleShot(True)
    deadline.timeout.connect(loop.quit)
    deadline.start(3000)
    loop.exec()
    assert not dialog._query_thread.isRunning()
    # This is intentionally a lower bound rather than an exact pixel contract:
    # QGIS/Qt may adjust the outer frame on an actual desktop.
    assert default_size.width() == 1680 and default_size.height() == 1040, default_size
    return {"pages": dialog._pages.count(), "compact_reflow": True,
            "detail_500_update_ms": round(detail_update_ms, 3),
            "screenshots": saved, "close_only_hides": True}


app = QgsApplication([], False)
app.initQgis()
# Outside QGIS's GUI launcher, Processing's plugin directory is not added.
sys.path.append(str(Path(QgsApplication.pkgDataPath()) / "python/plugins"))
with tempfile.TemporaryDirectory(prefix="loess-ui-design-") as temporary:
    try:
        result = globals()[sys.argv[2]](app, Path(temporary))
    finally:
        dialog = _ACTIVE_MONITOR_DIALOG
        if dialog is not None:
            dialog.shutdown()
            thread = getattr(dialog, "_query_thread", None)
            deadline = time.monotonic() + 3.0
            while thread is not None and thread.isRunning() and time.monotonic() < deadline:
                QCoreApplication.processEvents()
                thread.wait(50)
print(json.dumps(result), flush=True)
