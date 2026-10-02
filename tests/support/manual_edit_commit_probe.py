# ruff: noqa: E402
"""Commit synthetic polygons to temporary GeoPackages; no real Run or database."""

from __future__ import annotations

import sys
import tempfile
import traceback
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
try:
    from qgis.core import (
        QgsApplication,
        QgsCoordinateTransform,
        QgsGeometry,
        QgsProject,
    )
except ModuleNotFoundError:
    raise SystemExit(77)

from refinement_fixtures import SPEC, layer, records, rectangle, signature

from labeling_tool.refinement import class_workspace
from labeling_tool.refinement.manual_edit_commit import (
    commit_manual_add,
    commit_manual_delete,
    commit_manual_modify,
    plan_manual_modify_overlaps,
)
from labeling_tool.refinement.manual_edit_state import ManualEditTask


def confidence(_layer, _geometry):
    return 0.6, 0.15, ""


def modify(source, target, old, raw=(), saved=(), **kwargs):
    return commit_manual_modify(
        source_layer=source,
        target_layer=target,
        old_features=old,
        plan=plan_manual_modify_overlaps(old, raw),
        new_geometries=saved,
        source_code=12,
        target_code=12 if source is target else 13,
        run_spec=SPEC,
        baseline_stream_id="fusion:probe",
        keep_source_editing=source.isEditable(),
        confidence_statistics=kwargs.get("confidence_statistics", confidence),
    )


def modify_identity(app, root):
    source = layer(root, "identity", positions=(0, 10, 40))
    before = records(source)
    old = [before["identity-0"], before["identity-1"]]
    # The saved smoothing preview overlaps a different old polygon. Identity
    # still follows the raw sketch, while persisted geometry follows the preview.
    raw, saved = [rectangle(0, 2), rectangle(20)], [rectangle(10, 2), rectangle(20)]
    plan = plan_manual_modify_overlaps(old, raw)
    assert plan.matches == ((0, 0, 8.0),)
    assert plan.unmatched_old == (1,) and plan.unmatched_new == (1,)
    assert source.startEditing()
    result = modify(source, source, old, raw, saved)
    after = records(source)
    assert source.isEditable() and not source.isModified()
    assert len(after) == 3 and "identity-1" not in after
    changed = after["identity-0"]
    assert changed.geometry().equals(saved[0])
    assert changed["part_id"] == "007" and changed["geometry_revision"] == 4
    assert changed["geometry_source"] == "manual_edited"
    assert changed["edit_base"] == "fusion" and changed["reviewed"] == 0
    assert changed["confidence_mean"] == 0.6
    assert after["identity-2"].attributes() == before["identity-2"].attributes()
    assert result.matched[0].object_id == "identity-0"
    assert result.matched[0].geometry_changed
    assert result.deleted[0]["object_id"] == "identity-1"
    added = after[result.added[0].object_id]
    assert added["object_id"].startswith("manual-probe_new_")
    assert added["part_id"] == "000" and added["geometry_revision"] == 1
    assert source.rollBack()


def reclassify(app, root):
    for editable in (False, True):
        source = layer(root, f"reclass-{editable}", positions=(0,))
        target = layer(root, f"target-{editable}", code=13)
        if editable:
            assert source.startEditing() and target.startEditing()
        old = next(source.getFeatures())
        with patch(__name__ + ".confidence", side_effect=AssertionError("recomputed")):
            result = modify(source, target, [old])
        moved = records(target)[old["object_id"]]
        assert source.featureCount() == 0 and target.featureCount() == 1
        assert moved.geometry().equals(old.geometry())
        assert moved["confidence_mean"] == 0.8 and moved["confidence_std"] == 0.2
        assert moved["part_id"] == "007" and moved["geometry_revision"] == 4
        assert moved["class_code"] == 13 and moved["reviewed"] == 0
        assert not result.matched[0].geometry_changed and not result.added
        assert source.isEditable() == editable == target.isEditable()
        if editable:
            assert source.rollBack() and target.rollBack()
        # A repeated object identity must be rejected before either layer changes.
        duplicate = layer(root, f"duplicate-{editable}", positions=(0,))
        fid = next(duplicate.getFeatures()).id()
        assert duplicate.startEditing()
        assert duplicate.changeAttributeValue(
            fid, duplicate.fields().indexFromName("object_id"), old["object_id"]
        )
        assert duplicate.commitChanges()
        before = signature(duplicate), signature(target)
        try:
            modify(duplicate, target, list(duplicate.getFeatures()))
        except RuntimeError as error:
            assert "object_id" in str(error), error
        else:
            raise AssertionError("duplicate identity accepted")
        assert (signature(duplicate), signature(target)) == before


def cross_class_geometry(app, root):
    source = layer(root, "cross", positions=(0, 10))
    target = layer(root, "cross-target", code=13, crs="EPSG:4326")
    old = list(source.getFeatures())
    shape = rectangle(0, 2)
    expected = QgsGeometry(shape)
    expected.transform(
        QgsCoordinateTransform(source.crs(), target.crs(), QgsProject.instance())
    )
    calls = []

    def missing_confidence(value, geometry):
        calls.append((value, QgsGeometry(geometry)))
        return None, None, "synthetic raster unavailable"

    result = modify(
        source, target, old, [shape], [shape], confidence_statistics=missing_confidence
    )
    moved = next(target.getFeatures())
    assert source.featureCount() == 0
    assert moved.geometry().equals(expected)
    assert moved["object_id"] == "cross-0" and moved["geometry_revision"] == 4
    assert calls[0][0] is target and calls[0][1].equals(expected)
    assert result.deleted[0]["object_id"] == "cross-1"
    assert result.confidence_warnings[0].reason == "synthetic raster unavailable"
    assert not source.isEditable() and not target.isEditable()


class FailingLayer:
    """Inject a provider commit failure without replacing real edit buffers."""

    def __init__(self, value, failing_commits):
        self.layer, self.failing_commits, self.commits = value, failing_commits, 0

    def __getattr__(self, name):
        return getattr(self.layer, name)

    def commitChanges(self, stopEditing=True):
        self.commits += 1
        if self.commits in self.failing_commits:
            return False
        return self.layer.commitChanges(stopEditing)

    def commitErrors(self):
        return ["injected provider failure"]


def commit_failures(app, root):
    for case in ("source", "target", "compensation"):
        source = layer(root, case, positions=(0,))
        target = layer(root, case + "-target", code=13)
        assert source.startEditing() and target.startEditing()
        before = signature(source), signature(target)
        source_proxy = FailingLayer(source, {1} if case != "target" else set())
        target_proxy = FailingLayer(
            target,
            {1} if case == "target" else {2} if case == "compensation" else set(),
        )
        try:
            modify(source_proxy, target_proxy, list(source.getFeatures()))
        except RuntimeError as error:
            assert "injected provider failure" in str(error), error
            assert ("补偿回滚失败" in str(error)) == (case == "compensation")
        else:
            raise AssertionError("injected commit failure disappeared")
        assert signature(source) == before[0]
        assert source.isEditable() and target.isEditable()
        if case == "compensation":
            assert target.featureCount() == 1
        else:
            assert signature(target) == before[1]
        assert source.rollBack() and target.rollBack()


def add_and_delete(app, root):
    for keep_editing in (False, True):
        target = layer(root, f"add-{keep_editing}")
        result = commit_manual_add(
            layer=target,
            geometries=[rectangle(0), rectangle(10)],
            target_code=12,
            run_spec=SPEC,
            baseline_stream_id="fusion:probe",
            keep_editing=keep_editing,
            confidence_statistics=lambda *_: (None, None, "unavailable"),
        )
        assert len(result.added) == target.featureCount() == 2
        assert target.isEditable() == keep_editing
        assert all(item.confidence_warning == "unavailable" for item in result.added)
        before = signature(target)
        proxy = FailingLayer(target, {1})
        try:
            commit_manual_delete(proxy, [next(target.getFeatures()).id()])
        except RuntimeError:
            pass
        else:
            raise AssertionError("delete failure disappeared")
        assert signature(target) == before and not target.isEditable()
        assert commit_manual_delete(target, [f.id() for f in target.getFeatures()]) == 2
        assert target.featureCount() == 0 and not target.isEditable()
        proxy = FailingLayer(target, {1})
        try:
            commit_manual_add(
                layer=proxy,
                geometries=[rectangle(0)],
                target_code=12,
                run_spec=SPEC,
                baseline_stream_id="fusion:probe",
                keep_editing=keep_editing,
                confidence_statistics=confidence,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("add failure disappeared")
        assert target.featureCount() == 0
        assert target.isEditable() == keep_editing
        if target.isEditable():
            assert target.rollBack()


def dialog_history(app, root):
    from qgis.gui import QgsMapCanvas

    from labeling_tool.qgis_support.qt6_api import YES
    from labeling_tool.refinement import class_refinement_dialog as ui

    source = layer(root, "dialog", positions=(0, 10))
    project = QgsProject.instance()
    project.addMapLayer(source)
    canvas = QgsMapCanvas()
    dialog = ui.ClassRefinementDialog(
        SimpleNamespace(
            mapCanvas=lambda: canvas,
            activeLayer=lambda: source,
            setActiveLayer=lambda _: None,
        ),
        SimpleNamespace(),
    )
    dialog._run_spec = dict(SPEC, run_dir=str(root))
    dialog._workspace = {"baseline_stream_id": "fusion:probe", "classes": {}}
    dialog._class_layers = {12: source.id()}
    history = []
    try:
        # Keep real native edit signals connected: metadata suppression must
        # prevent ordinary edit tracking from recording the same commit twice.
        source.editingStarted.connect(lambda: dialog._editing_started(12))
        source.beforeCommitChanges.connect(
            lambda *_: dialog._edit_tracker.capture_before_commit(12, source)
        )
        source.afterCommitChanges.connect(lambda: dialog._editing_stopped(12))
        source.editingStopped.connect(lambda: dialog._editing_stopped(12))
        source.committedFeaturesAdded.connect(
            lambda _, items: dialog._edit_tracker.record_committed_additions(12, items)
        )
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(ui.QMessageBox, "question", return_value=YES)
            )
            warning = stack.enter_context(patch.object(ui.QMessageBox, "warning"))
            stack.enter_context(
                patch.object(
                    class_workspace,
                    "append_history",
                    side_effect=lambda _, event, **data: history.append((event, data)),
                )
            )
            save = stack.enter_context(
                patch.object(
                    class_workspace,
                    "save_workspace",
                    side_effect=lambda _, workspace, **kw: workspace,
                )
            )
            for name in (
                "_update_manual_panel",
                "_refresh_table",
                "_refresh_class_display",
                "_set_class_modified",
                "_set_visible",
                "_start_manual_capture",
            ):
                stack.enter_context(patch.object(dialog, name))
            for name in ("start_picker", "restore_previous"):
                stack.enter_context(patch.object(dialog._manual_tools, name))
            stack.enter_context(
                patch.object(
                    dialog, "_optional_confidence_statistics", side_effect=confidence
                )
            )
            stack.enter_context(
                patch.object(dialog, "_local_topology_hint", return_value="ok")
            )
            old = list(source.getFeatures())
            dialog._manual_task = ManualEditTask(
                kind="modify",
                class_code=12,
                target_code=12,
                state="selecting",
                selected_feature_ids=[f.id() for f in old],
                pending_geometries=[rectangle(0, 2), rectangle(20)],
                pending_errors=[],
                smoothing_enabled=False,
                submitted_batch_count=0,
                modified_old_count=0,
                saved_new_count=0,
                deleted_old_count=0,
            )
            dialog._commit_manual_modify_batch()
            assert not warning.called, warning.call_args_list
            assert [event for event, _ in history] == [
                "geometry_modified",
                "feature_deleted",
                "feature_added",
            ], history
            assert (
                not dialog._edit_tracker.suppressed
                and dialog._manual_task.state == "selecting"
            )
            assert dialog._manual_task.submitted_batch_count == 1
            history.clear()
            dialog._manual_task = ManualEditTask(
                kind="add",
                class_code=12,
                target_code=12,
                state="capturing",
                pending_geometries=[rectangle(30)],
                pending_errors=[],
                smoothing_enabled=False,
                added_count=0,
                submitted_batch_count=0,
                saved_counts={},
            )
            dialog._commit_manual_add()
            assert not warning.called, warning.call_args_list
            assert [event for event, _ in history] == ["feature_added"], history
            assert source.isEditable() and not dialog._edit_tracker.suppressed
            assert dialog._manual_task.added_count == 1
            assert save.call_count == 2
            history.clear()
            before = signature(source)
            dialog._manual_task.pending_geometries = [rectangle(40)]
            with patch.object(
                ui.manual_edit_commit,
                "commit_manual_add",
                side_effect=RuntimeError("injected"),
            ):
                dialog._commit_manual_add()
            assert warning.call_count == 1
            assert dialog._manual_task.state == "failed"
            assert dialog._manual_task.pending_geometries
            assert not dialog._edit_tracker.suppressed and not history
            assert save.call_count == 2 and signature(source) == before
            warning.reset_mock()
            # Deletion uses the ordinary signal path, so it must still produce
            # exactly one history event when following a continuously open edit.
            fid = next(source.getFeatures()).id()
            source.selectByIds([fid])
            dialog._manual_task = ManualEditTask.for_delete(12, [fid])
            with patch.object(dialog, "_select_class_context"):
                dialog._commit_manual_delete()
            assert not warning.called, warning.call_args_list
            assert [event for event, _ in history] == ["feature_deleted"], history
            assert dialog._manual_task is None and not dialog._edit_tracker.suppressed
            assert save.call_count == 3 and not source.isEditable()
    finally:
        dialog._manual_task = None
        source.blockSignals(True)
        if source.isEditable():
            source.rollBack()
        dialog._workspace = None
        dialog._run_spec = None
        dialog.close()
        dialog.deleteLater()
        app.processEvents()
        project.removeAllMapLayers()


app = QgsApplication([], False)
app.initQgis()
try:
    with tempfile.TemporaryDirectory(prefix="loess-manual-commit-") as directory:
        scenario = sys.argv[2]
        globals()[scenario](app, Path(directory))
        print(scenario + ": passed", flush=True)
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
finally:
    app.exitQgis()
