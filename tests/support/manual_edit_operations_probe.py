# ruff: noqa: E402
"""Native Dialog operation wiring on temporary GeoPackages, without a Run."""

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
    from qgis.core import QgsApplication, QgsProject
    from qgis.gui import QgsMapCanvas
except ModuleNotFoundError:
    raise SystemExit(77)

from refinement_fixtures import SPEC, layer, records, rectangle, signature

from labeling_tool.qgis_support.qt6_api import NO, YES
from labeling_tool.refinement import class_workspace
from labeling_tool.refinement.manual_edit_state import ManualEditTask


class NativeDialog:
    """Own a synthetic Dialog and only patch display-only dependencies."""

    def __init__(self, root, name, *, target=True):
        from labeling_tool.refinement import class_refinement_dialog as ui

        self.source = layer(root, name, positions=(0, 10))
        self.target = layer(root, name + "-target", code=13) if target else None
        self.project = QgsProject.instance()
        self.project.addMapLayer(self.source)
        if self.target is not None:
            self.project.addMapLayer(self.target)
        self.canvas = QgsMapCanvas()
        self.canvas.setLayers(
            [item for item in (self.source, self.target) if item is not None]
        )
        self.active = self.source
        self.dialog = ui.ClassRefinementDialog(
            SimpleNamespace(
                mapCanvas=lambda: self.canvas,
                activeLayer=lambda: self.active,
                setActiveLayer=self._set_active,
            ),
            SimpleNamespace(),
        )
        self.dialog._run_spec = dict(SPEC, run_dir=str(root))
        self.dialog._workspace = {"baseline_stream_id": "fusion:probe", "classes": {}}
        self.dialog._class_layers = {12: self.source.id()}
        if self.target is not None:
            self.dialog._class_layers[13] = self.target.id()
        for code in self.dialog._class_layers:
            self.dialog._register_workspace_layer(code, self.dialog._class_layers[code])

    def _set_active(self, value):
        self.active = value

    def close(self):
        for value in (self.source, self.target):
            if value is not None:
                value.blockSignals(True)
        self.dialog._workspace = None
        self.dialog._run_spec = None
        self.dialog._manual_task = None
        for value in (self.source, self.target):
            if value is not None and value.isEditable():
                value.rollBack()
        self.dialog.cleanup()
        self.dialog.close()
        self.dialog.deleteLater()
        self.project.removeAllMapLayers()
        self.canvas.close()


@contextmanager
def native_dialog(root, name, *, target=True):
    value = NativeDialog(root, name, target=target)
    try:
        with ExitStack() as stack:
            for method in (
                "_update_actions",
                "_update_manual_panel",
                "_refresh_table",
                "_refresh_class_display",
                "_set_class_modified",
                "_select_class_context",
                "_set_visible",
                "_clear_manual_bands",
                "_clear_manual_add_candidate_bands",
                "_start_manual_capture",
            ):
                stack.enter_context(patch.object(value.dialog, method))
            for method in (
                "stop_picker",
                "stop_capture",
                "start_picker",
                "restore_previous",
                "end_session",
            ):
                stack.enter_context(patch.object(value.dialog._manual_tools, method))
            stack.enter_context(
                patch.object(
                    value.dialog,
                    "_optional_confidence_statistics",
                    return_value=(0.6, 0.1, ""),
                )
            )
            stack.enter_context(
                patch.object(value.dialog, "_local_topology_hint", return_value="ok")
            )
            stack.enter_context(
                patch.object(
                    class_workspace,
                    "save_workspace",
                    side_effect=lambda _spec, workspace, **_kwargs: workspace,
                )
            )
            stack.enter_context(
                patch(
                    "labeling_tool.refinement.class_refinement_dialog."
                    "QMessageBox.warning"
                )
            )
            stack.enter_context(
                patch(
                    "labeling_tool.refinement.class_refinement_dialog."
                    "QMessageBox.information"
                )
            )
            yield value, stack
    finally:
        value.close()


def modify_task(value):
    old = next(value.source.getFeatures())
    task = ManualEditTask.for_modify(12, [old.id()])
    task.pending_geometries = [rectangle(0, 2)]
    task.pending_errors = [""]
    value.dialog._manual_task = task
    if not value.source.isEditable():
        assert value.source.startEditing()
    task.editing_started_by_task = True
    return task, old


def modify_wiring(_app, root):
    with native_dialog(root, "modify") as (value, stack):
        task, old = modify_task(value)
        saved = rectangle(10, 2)
        task.smoothing_enabled = True
        from labeling_tool.refinement.geometry_smoothing import (
            SmoothingBatchResult,
            SmoothingParameters,
            SmoothingStatistics,
            geometry_source_hash,
        )

        # Construct the same immutable preview value that the native smoother
        # produces. Raw geometry is at x=0, whereas the current preview at
        # x=10 is the one that must be persisted after identity matching.
        task.smoothing_preview = SmoothingBatchResult(
            geometries=(saved,),
            parameters=SmoothingParameters(
                *value.dialog._manual_panel.smoothing_parameters()
            ),
            source_hashes=(geometry_source_hash(task.pending_geometries[0]),),
            statistics=SmoothingStatistics(0, 0, 0.0, 0.0),
        )
        before = signature(value.source)
        history = []
        stack.enter_context(
            patch.object(
                class_workspace,
                "append_history",
                side_effect=lambda _spec, event, **data: history.append((event, data)),
            )
        )
        stack.enter_context(
            patch(
                "labeling_tool.refinement.class_refinement_dialog.QMessageBox.question",
                side_effect=[NO, YES],
            )
        )
        value.dialog._commit_manual_modify_batch()
        assert signature(value.source) == before and task.state == "selecting"
        value.dialog._commit_manual_modify_batch()
        changed = records(value.source)[old["object_id"]]
        assert changed.geometry().equals(saved)
        assert records(value.source)["modify-1"].geometry().equals(rectangle(10))
        assert [event for event, _data in history] == ["geometry_modified"]
        assert history[0][1]["object_id"] == old["object_id"]

        # A low-level failure retains the exact candidate for retry and must
        # produce no audit event before a commit has succeeded.
        task, _old = modify_task(value)
        task.pending_geometries = [rectangle(0, 1)]
        task.pending_errors = [""]
        history.clear()
        stack.enter_context(
            patch(
                "labeling_tool.refinement.class_refinement_dialog.QMessageBox.question",
                return_value=YES,
            )
        )
        stack.enter_context(
            patch(
                "labeling_tool.refinement.manual_edit_commit.commit_manual_modify",
                side_effect=RuntimeError("provider rejected batch"),
            )
        )
        value.dialog._commit_manual_modify_batch()
        assert task.state == "failed"
        assert task.pending_geometries[0].equals(rectangle(0, 1))
        assert not history


def add_finish_sessions(_app, root):
    with native_dialog(root, "add") as (value, stack):
        task = ManualEditTask.for_add(12, [])
        task.editing_started_by_task = True
        value.dialog._manual_task = task
        assert value.source.startEditing()
        history = []
        stack.enter_context(
            patch.object(
                class_workspace,
                "append_history",
                side_effect=lambda _spec, event, **data: history.append((event, data)),
            )
        )
        task.target_code = 13
        task.pending_geometries = [rectangle(20)]
        task.pending_errors = [""]
        value.dialog._commit_manual_add()
        assert task.saved_counts == {13: 1} and value.target.featureCount() == 1
        task.target_code = 12
        task.pending_geometries = [rectangle(30)]
        task.pending_errors = [""]
        value.dialog._commit_manual_add()
        assert task.saved_counts == {13: 1, 12: 1}
        task.pending_geometries = [rectangle(40)]
        task.pending_errors = [""]
        value.dialog._finish_add_task()
        assert value.dialog._manual_task is None
        assert value.source.featureCount() == 3 and value.target.featureCount() == 1
        assert not value.source.isEditable() and not value.target.isEditable()
        assert [event for event, _data in history] == ["feature_added", "feature_added"]

        # Finishing never closes an existing user session or a dirty session.
        assert value.source.startEditing()
        user_task = ManualEditTask.for_add(12, [])
        value.dialog._manual_task = user_task
        value.dialog._finish_add_task()
        assert value.source.isEditable()
        assert value.source.rollBack()
        assert value.source.startEditing()
        dirty_task = ManualEditTask.for_add(12, [])
        dirty_task.editing_started_by_task = True
        value.dialog._manual_task = dirty_task
        feature = next(value.source.getFeatures())
        assert value.source.changeGeometry(feature.id(), rectangle(0, 2))
        value.dialog._finish_add_task()
        assert value.source.isEditable() and value.source.isModified()


def delete_wiring(_app, root):
    with native_dialog(root, "delete", target=False) as (value, stack):
        feature = next(value.source.getFeatures())
        value.source.selectByIds([feature.id()])
        task = ManualEditTask.for_delete(12, [feature.id()])
        value.dialog._manual_task = task
        history = []
        stack.enter_context(
            patch.object(
                class_workspace,
                "append_history",
                side_effect=lambda _spec, event, **data: history.append((event, data)),
            )
        )
        stack.enter_context(
            patch(
                "labeling_tool.refinement.class_refinement_dialog.QMessageBox.question",
                return_value=NO,
            )
        )
        before = signature(value.source)
        value.dialog._commit_manual_delete()
        assert signature(value.source) == before and value.dialog._manual_task is task
        with (
            patch(
                "labeling_tool.refinement.class_refinement_dialog.QMessageBox.question",
                return_value=YES,
            ),
            patch(
                "labeling_tool.refinement.manual_edit_commit.commit_manual_delete",
                side_effect=RuntimeError("provider rejected delete"),
            ),
        ):
            value.dialog._commit_manual_delete()
        assert task.state == "selecting" and value.source.featureCount() == 2
        with patch(
            "labeling_tool.refinement.class_refinement_dialog.QMessageBox.question",
            return_value=YES,
        ):
            value.dialog._commit_manual_delete()
        assert value.source.featureCount() == 1 and value.dialog._manual_task is None
        assert [event for event, _data in history] == ["feature_deleted"]


def stale_confirmation(_app, root):
    for operation in ("modify", "delete"):
        for mutation in ("run_spec", "task"):
            with native_dialog(root, f"stale-{operation}-{mutation}") as (
                value,
                stack,
            ):
                feature = next(value.source.getFeatures())
                if operation == "modify":
                    task = ManualEditTask.for_modify(12, [feature.id()])
                    task.pending_geometries = [rectangle(0, 2)]
                    task.pending_errors = [""]
                else:
                    value.source.selectByIds([feature.id()])
                    task = ManualEditTask.for_delete(12, [feature.id()])
                value.dialog._manual_task = task
                before = signature(value.source)

                def answer(*_args):
                    if mutation == "run_spec":
                        value.dialog._run_spec = dict(
                            value.dialog._run_spec, run_id="new"
                        )
                    elif operation == "modify":
                        value.dialog._manual_task = ManualEditTask.for_modify(12, [])
                    else:
                        value.dialog._manual_task = ManualEditTask.for_delete(12, [])
                    return YES

                stack.enter_context(
                    patch(
                        "labeling_tool.refinement.class_refinement_dialog."
                        "QMessageBox.question",
                        side_effect=answer,
                    )
                )
                method = "_commit_manual_" + operation
                if operation == "modify":
                    method += "_batch"
                getattr(value.dialog, method)()
                assert signature(value.source) == before


def history_failure(_app, root):
    with native_dialog(root, "history", target=False) as (value, stack):
        task, old = modify_task(value)
        stack.enter_context(
            patch(
                "labeling_tool.refinement.class_refinement_dialog.QMessageBox.question",
                return_value=YES,
            )
        )
        stack.enter_context(
            patch.object(
                class_workspace,
                "append_history",
                side_effect=RuntimeError("audit down"),
            )
        )
        try:
            value.dialog._commit_manual_modify_batch()
        except RuntimeError as error:
            assert str(error) == "audit down"
        else:
            raise AssertionError("post-commit audit failure was swallowed")
        assert task.state == "committing"
        assert (
            records(value.source)[old["object_id"]].geometry().equals(rectangle(0, 2))
        )


app = QgsApplication([], False)
app.initQgis()
try:
    with tempfile.TemporaryDirectory(prefix="loess-manual-operations-") as directory:
        scenario = sys.argv[2]
        globals()[scenario](app, Path(directory))
        print(scenario + ": passed", flush=True)
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
finally:
    app.exitQgis()
