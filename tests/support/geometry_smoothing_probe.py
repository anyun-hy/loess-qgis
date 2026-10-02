# ruff: noqa: E402
"""Exercise smoothing against native QGIS and disposable GeoPackages."""

from __future__ import annotations

import sys
import tempfile
import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
try:
    from qgis.core import QgsApplication, QgsGeometry, QgsProject, QgsVectorLayer
    from qgis.gui import QgsMapCanvas
    from qgis.PyQt.QtCore import QCoreApplication, QEvent
except ModuleNotFoundError:
    raise SystemExit(77)

from refinement_fixtures import SPEC, layer, rectangle, signature

from labeling_tool.refinement import class_refinement_dialog as ui
from labeling_tool.refinement.geometry_smoothing import (
    GeometrySmoothingError,
    SmoothingParameters,
    smooth_geometry_batch,
    validate_polygon_geometry,
)
from labeling_tool.refinement.manual_edit_operations import prepare_manual_add
from labeling_tool.refinement.manual_edit_state import ManualEditTask

PARAMETERS = (1, 0.25, 180.0)


def calculation(app, root):
    sources = [rectangle(0), rectangle(10)]
    originals = [bytes(value.asWkb()) for value in sources]
    parameters = SmoothingParameters(*PARAMETERS)
    result = smooth_geometry_batch(sources, parameters, convert_to_multi=True)
    stats = result.statistics
    assert (stats.source_vertex_count, stats.smoothed_vertex_count) == (10, 18)
    assert (stats.source_area, stats.smoothed_area) == (32.0, 28.0)
    assert stats.area_change_percent == -12.5
    assert [bytes(value.asWkb()) for value in sources] == originals
    outputs = [bytes(value.asWkb()) for value in result.geometries]
    sources[0].translate(50, 0)
    assert [bytes(value.asWkb()) for value in result.geometries] == outputs
    polygon = QgsGeometry.fromWkt("Polygon ((0 0, 4 0, 4 4, 0 4, 0 0))")
    native = smooth_geometry_batch([polygon], parameters)
    guided = smooth_geometry_batch([polygon], parameters, convert_to_multi=True)
    assert not native.geometries[0].isMultipart()
    assert guided.geometries[0].isMultipart() and not polygon.isMultipart()
    for invalid in (None, QgsGeometry(), QgsGeometry.fromWkt("LINESTRING(0 0, 4 4)")):
        assert validate_polygon_geometry(invalid)
    try:
        smooth_geometry_batch([polygon, QgsGeometry()], parameters)
    except GeometrySmoothingError as exc:
        assert exc.index == 2 and exc.reason
    else:
        raise AssertionError("Invalid second output must reject the whole batch")
    cause = RuntimeError("GEOS failure on second geometry")
    with patch.object(QgsGeometry, "smooth", side_effect=[polygon, cause]):
        try:
            smooth_geometry_batch(sources, parameters)
        except GeometrySmoothingError as exc:
            assert exc.index == 2 and exc.__cause__ is cause
        else:
            raise AssertionError("QGIS errors must retain the cause and feature index")


@contextmanager
def editing_dialog(app, root):
    source = layer(root, "smoothing", positions=(0, 10))
    QgsProject.instance().addMapLayer(source)
    canvas = QgsMapCanvas()
    canvas.setLayers([source])
    active = [source]
    dialog = ui.ClassRefinementDialog(
        SimpleNamespace(mapCanvas=lambda: canvas, activeLayer=lambda: active[0]),
        SimpleNamespace(),
    )
    dialog._run_spec = dict(SPEC, run_dir=str(root))
    dialog._workspace = {"baseline_stream_id": "fusion:probe", "classes": {}}
    dialog._class_layers = {12: source.id()}
    with ExitStack() as stack:
        for name in (
            "_update_actions",
            "_refresh_table",
            "_select_class_context",
            "_set_visible",
        ):
            stack.enter_context(patch.object(dialog, name))
        stack.enter_context(patch.object(dialog, "_store_smoothing_parameters"))
        stack.enter_context(patch.object(ui.QMessageBox, "information"))
        warning = stack.enter_context(patch.object(ui.QMessageBox, "warning"))
        dialog._manual_panel.set_smoothing_parameters(PARAMETERS)
        dialog._sync_smoothing_parameter_widgets(PARAMETERS, "manual")
        assert source.startEditing()
        source.selectAll()
        dialog._update_manual_panel()
        try:
            yield dialog, source, active, warning
        finally:
            dialog._manual_smoothing_timer.stop()
            dialog._manual_task = None
            dialog._workspace = None
            active[0] = source
            source.blockSignals(True)
            if source.isEditable():
                source.rollBack()
            dialog.cleanup()
            dialog.close()
            dialog.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app.processEvents()
            QgsProject.instance().removeAllMapLayers()
            canvas.close()


def disk_signature(source):
    saved = QgsVectorLayer(source.source(), "saved", "ogr")
    assert saved.isValid()
    return signature(saved)


def preview(dialog):
    dialog._update_qgis_smoothing_controls()
    assert dialog.qgis_smooth_preview_btn.isEnabled()
    dialog.qgis_smooth_preview_btn.click()
    assert dialog._qgis_smooth_preview_is_current()
    assert dialog.qgis_smooth_apply_btn.isEnabled()


def native_apply(app, root):
    with editing_dialog(app, root) as (dialog, source, _, warning):
        before = signature(source)
        expected = {}
        for feature in source.getFeatures():
            smoothed = feature.geometry().smooth(1, 0.25, -1.0, 180.0)
            expected[feature["object_id"]] = (
                bytes(smoothed.asWkb()),
                tuple(feature.attributes()),
            )
        preview(dialog)
        assert signature(source) == disk_signature(source) == before
        assert not source.isModified() and len(dialog._qgis_smooth_preview_bands) == 2
        count = source.undoStack().count()
        dialog.qgis_smooth_apply_btn.click()
        assert signature(source) == expected and expected != before
        assert source.undoStack().count() == count + 1
        assert disk_signature(source) == before and source.isEditable()
        assert dialog._qgis_smooth_preview is None
        assert dialog._qgis_smooth_preview_bands == []
        dialog.qgis_undo_btn.click()
        assert signature(source) == before
        dialog.qgis_redo_btn.click()
        assert signature(source) == expected
        warning.assert_not_called()


def native_stale(app, root):
    with editing_dialog(app, root) as (dialog, source, active, _):
        before = signature(source)
        ids = sorted(source.selectedFeatureIds())
        preview(dialog)
        dialog.qgis_smooth_offset_spin.setValue(0.3)
        assert dialog._qgis_smooth_preview is None
        assert not dialog.qgis_smooth_apply_btn.isEnabled()
        preview(dialog)
        source.selectByIds(ids[:1])
        assert not dialog._qgis_smooth_preview_is_current()
        dialog._apply_qgis_smoothing()
        assert signature(source) == before and not source.isModified()
        source.selectByIds(ids)
        preview(dialog)
        other = layer(root, "other", positions=(0, 10))
        assert other.startEditing()
        other.selectAll()
        active[0] = other
        assert not dialog._qgis_smooth_preview_is_current()
        dialog._apply_qgis_smoothing()
        assert not other.isModified() and not source.isModified()
        other.rollBack()
        active[0] = source
        preview(dialog)
        source.beginEditCommand("existing geometry edit")
        assert source.changeGeometry(ids[0], rectangle(30))
        source.endEditCommand()
        changed = signature(source)
        assert not dialog._qgis_smooth_preview_is_current()
        dialog._apply_qgis_smoothing()
        assert signature(source) == changed and disk_signature(source) == before
        assert source.undoStack().count() == 1


def native_failure(app, root):
    with editing_dialog(app, root) as (dialog, source, _, warning):
        ids = sorted(source.selectedFeatureIds())
        persisted = disk_signature(source)
        source.beginEditCommand("earlier edit")
        assert source.changeGeometry(ids[0], rectangle(20))
        source.endEditCommand()
        before = signature(source)
        preview(dialog)
        change = source.changeGeometry

        def fail_second(feature_id, geometry):
            return False if feature_id == ids[1] else change(feature_id, geometry)

        with patch.object(source, "changeGeometry", side_effect=fail_second) as changes:
            dialog.qgis_smooth_apply_btn.click()
        assert changes.call_count == 2 and warning.call_count == 1
        assert signature(source) == before and disk_signature(source) == persisted
        assert source.undoStack().count() == 1, "Keep the preceding edit only"
        source.undoStack().undo()
        assert signature(source) == persisted
        preview(dialog)
        with patch.object(
            QgsGeometry, "smooth", side_effect=RuntimeError("GEOS probe")
        ):
            dialog.qgis_smooth_preview_btn.click()
        assert dialog._qgis_smooth_preview is None
        assert dialog._qgis_smooth_preview_bands == []
        assert not dialog.qgis_smooth_apply_btn.isEnabled()
        assert warning.call_count == 2 and signature(source) == persisted


def manual_guard(app, root):
    with editing_dialog(app, root) as (dialog, source, _, _):
        persisted = signature(source)
        task = ManualEditTask.for_add(12, [])
        task.pending_geometries = [rectangle(30), rectangle(40)]
        task.pending_errors = ["", ""]
        task.smoothing_enabled = True
        dialog._manual_task = task
        originals = [bytes(value.asWkb()) for value in task.pending_geometries]
        dialog._refresh_manual_smoothing_preview()
        assert dialog._manual_smoothing_preview_is_current()
        saved = prepare_manual_add(task, SmoothingParameters(*PARAMETERS)).geometries
        assert all(value.isMultipart() for value in saved)
        expected = [bytes(value.asWkb()) for value in saved]
        assert originals != expected
        saved[0].translate(100, 0)
        assert [
            bytes(g.asWkb())
            for g in prepare_manual_add(
                task, SmoothingParameters(*PARAMETERS)
            ).geometries
        ] == expected
        assert [bytes(g.asWkb()) for g in task.pending_geometries] == originals
        task.pending_geometries.reverse()
        assert not dialog._manual_smoothing_preview_is_current()
        try:
            prepare_manual_add(task, SmoothingParameters(*PARAMETERS))
        except RuntimeError:
            pass
        else:
            raise AssertionError("Reordered sources must not save an old preview")
        dialog._refresh_manual_smoothing_preview()
        with patch.object(
            QgsGeometry, "smooth", side_effect=RuntimeError("GEOS probe")
        ):
            dialog._refresh_manual_smoothing_preview()
        assert task.smoothing_preview is None and "GEOS probe" in task.smoothing_error
        assert not dialog._manual_smoothing_preview_is_current()
        task.smoothing_enabled = False
        raw = prepare_manual_add(task, SmoothingParameters(*PARAMETERS)).geometries
        assert [bytes(value.asWkb()) for value in raw] == list(reversed(originals))
        assert signature(source) == disk_signature(source) == persisted


app = QgsApplication([], False)
app.initQgis()
try:
    with tempfile.TemporaryDirectory(prefix="loess-smoothing-") as directory:
        globals()[sys.argv[2]](app, Path(directory))
        print(sys.argv[2] + ": passed", flush=True)
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
finally:
    app.exitQgis()
