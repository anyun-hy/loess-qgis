# ruff: noqa: E402
"""Edit-buffer regressions on synthetic native QGIS layers."""

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
    from qgis.core import QgsApplication, QgsFeature, QgsProject
except ModuleNotFoundError:
    raise SystemExit(77)

from refinement_fixtures import SPEC, layer, records, rectangle

from labeling_tool.refinement import class_workspace
from labeling_tool.refinement.edit_tracking import EditTracker, snapshot_features


def confidence(_layer, _geometry):
    return 0.6, 0.15, ""


def finish(tracker, source, callback=confidence):
    return tracker.finish(
        12,
        source,
        run_spec=SPEC,
        baseline_stream_id="fusion:probe",
        confidence_statistics=callback,
    )


@contextmanager
def tracking(source, tracker=None, *, report_added=True, statistics=confidence):
    owner = tracker if tracker is not None else EditTracker()
    history, results, errors = [], [], []

    def finished():
        try:
            result = finish(owner, source, statistics)
            if result is not None:
                results.append(result)
        except Exception as exc:
            errors.append(exc)

    slots = [
        (source.editingStarted, lambda: owner.begin(12, source)),
        (
            source.beforeCommitChanges,
            lambda *_: owner.capture_before_commit(12, source),
        ),
        (source.afterCommitChanges, finished),
        (source.editingStopped, finished),
    ]
    if report_added:
        slots.append(
            (
                source.committedFeaturesAdded,
                lambda _, items: owner.record_committed_additions(12, items),
            )
        )
    for signal, slot in slots:
        signal.connect(slot)
    try:
        with patch.object(
            class_workspace,
            "append_history",
            side_effect=lambda _, event, **data: history.append((event, data)),
        ):
            yield SimpleNamespace(
                owner=owner, history=history, results=results, errors=errors
            )
    finally:
        for signal, slot in slots:
            signal.disconnect(slot)
        if source.isEditable():
            source.rollBack()


def added_feature(source, identity, shape):
    feature = QgsFeature(source.fields())
    feature.setGeometry(shape)
    feature["object_id"] = identity
    return feature


def edits(app, root):
    # Preserve the original 1000-feature incremental regression after extracting
    # the implementation: one geometry edit must not hash the entire layer.
    source = layer(root, "incremental", positions=range(0, 10000, 10))
    with (
        tracking(source) as check,
        patch.object(
            class_workspace, "geometry_hash", wraps=class_workspace.geometry_hash
        ) as hashes,
    ):
        assert source.startEditing()
        assert hashes.call_count == 0
        old = next(source.getFeatures())
        fid = old.id()
        assert source.changeGeometry(fid, rectangle(0, 2))
        # Identity fields changed through a third-party editing tool are restored.
        assert source.changeAttributeValue(
            fid, source.fields().indexFromName("object_id"), "tampered"
        )
        assert source.commitChanges(False), source.commitErrors()
        assert not check.errors, check.errors
        assert hashes.call_count < 12, hashes.call_count
        assert [event for event, _ in check.history] == ["geometry_modified"]
        assert source.getFeature(fid)["geometry_revision"] == 4
        assert source.getFeature(fid)["object_id"] == old["object_id"]
        check.history.clear()
        assert source.addFeature(added_feature(source, "new-object", rectangle(10010)))
        assert source.commitChanges(False), source.commitErrors()
        assert not check.errors, check.errors
        assert [event for event, _ in check.history] == ["feature_added"]
        assert records(source)["new-object"]["geometry_revision"] == 1
        check.history.clear()
        assert source.deleteFeature(fid)
        assert source.commitChanges(), source.commitErrors()
        assert [event for event, _ in check.history] == ["feature_deleted"]
        check.history.clear()
        assert source.startEditing()
        assert source.changeGeometry(next(source.getFeatures()).id(), rectangle(40, 2))
        assert source.rollBack()
        assert not check.history and not check.errors, check.errors
        assert source.featureCount() == 1000
        assert not check.owner.has_session(12)


def restored(app, root):
    source = layer(root, "restored", positions=(0, 10))
    original = next(source.getFeatures())
    assert source.startEditing()
    assert source.changeGeometry(original.id(), rectangle(0, 2))
    persisted = type(source)(source.source(), "persisted", "ogr")
    owner = EditTracker()
    owner.restore(12, source, persisted)
    assert owner.has_session(12)
    with tracking(source, owner) as check:
        assert source.commitChanges(False)
        assert not check.errors, check.errors
        assert [event for event, _ in check.history] == ["geometry_modified"]
        assert check.history[0][1][
            "before_geometry_hash"
        ] == class_workspace.geometry_hash(original.geometry())
        assert source.getFeature(original.id())["geometry_revision"] == 4
        check.history.clear()
        assert source.changeGeometry(original.id(), rectangle(0, 1))
        assert source.commitChanges(False)
        assert source.getFeature(original.id())["geometry_revision"] == 5
        assert [event for event, _ in check.history] == ["geometry_modified"]
        check.history.clear()
        assert source.commitChanges()
        assert not check.history and not check.errors, check.errors


def prepared_edit(app, root):
    source = layer(root, "prepared", positions=(0,))
    original = next(source.getFeatures())
    with tracking(source) as check:
        baseline = snapshot_features(source, [original.id()])
        with check.owner.suppress():
            assert source.startEditing()
            assert source.changeGeometry(original.id(), rectangle(0, 2))
            for name, value in {
                "geometry_revision": 4,
                "geometry_source": "manual_edited",
                "edit_base": "sam3",
            }.items():
                assert source.changeAttributeValue(
                    original.id(), source.fields().indexFromName(name), value
                )
        check.owner.prepare_edit(
            12, baseline=baseline, metadata_prepared=True, session_id="sam-probe"
        )
        assert source.commitChanges(False)
        assert not check.errors, check.errors
        saved = source.getFeature(original.id())
        assert saved["geometry_revision"] == 4 and saved["edit_base"] == "sam3"
        assert [event for event, _ in check.history] == ["geometry_modified"]
        check.history.clear()
        assert source.changeGeometry(original.id(), rectangle(0, 1))
        assert source.commitChanges(False)
        assert source.getFeature(original.id())["geometry_revision"] == 5
        assert len(check.history) == 1 and not check.errors
        # A new Run must not inherit a prepared-edit marker or a baseline.
        check.owner.prepare_edit(
            12, baseline=baseline, metadata_prepared=True, session_id="old-run"
        )
        check.owner.reset()
        assert not check.owner.has_session(12) and not check.owner.suppressed
        check.owner.begin(12, source)
        assert source.changeGeometry(original.id(), rectangle(0, 0.5))
        assert source.commitChanges(False)
        assert source.getFeature(original.id())["geometry_revision"] == 6


def metadata_failure(app, root):
    source = layer(root, "failure", positions=(0,))
    owner = EditTracker()
    with owner.suppress():
        try:
            with owner.suppress():
                raise RuntimeError("nested")
        except RuntimeError:
            pass
        assert owner.suppressed
    assert not owner.suppressed
    assert source.startEditing()
    owner.begin(12, source)
    fid = next(source.getFeatures()).id()
    assert source.changeGeometry(fid, rectangle(0, 2))
    owner.capture_before_commit(12, source)
    assert source.commitChanges(False)

    class FailingMetadata:
        def __getattr__(self, name):
            return getattr(source, name)

        def commitChanges(self, stopEditing=True):
            return False

        def commitErrors(self):
            return ["injected metadata failure"]

    with patch.object(class_workspace, "append_history"):
        try:
            finish(owner, FailingMetadata())
        except RuntimeError as error:
            assert "injected metadata failure" in str(error), error
        else:
            raise AssertionError("metadata failure swallowed")
    assert not owner.suppressed
    assert not source.isEditable()
    # Geometry was committed before the metadata update, as in the original
    # code; failure is surfaced, never reported as a completed edit result.
    assert source.getFeature(fid).geometry().equals(rectangle(0, 2))
    owner.discard(12)
    assert not owner.has_session(12)
    with tracking(
        source, owner, statistics=lambda *_: (None, None, "raster missing")
    ) as check:
        assert source.startEditing()
        assert source.changeGeometry(fid, rectangle(0, 1))
        assert source.commitChanges(False)
        assert not check.errors, check.errors
        assert [event for event, _ in check.history] == [
            "geometry_modified",
            "confidence_statistics_unavailable",
        ]


def transaction_tracking(app, root):
    source = layer(root, "transaction-contract", positions=(0, 10))

    class TransactionLayer:
        """Exercise the provider contract without claiming a real DB transaction.

        The local OGR provider cannot create QgsTransaction. Native feature I/O
        is retained, but the transaction flag and missing additions signal are
        supplied explicitly to test the full-snapshot fallback.
        """

        def __getattr__(self, name):
            return getattr(source, name)

        def dataProvider(self):
            return SimpleNamespace(transaction=lambda: self)

    with tracking(TransactionLayer(), report_added=False) as check:
        assert source.startEditing()
        original = next(source.getFeatures())
        assert source.changeGeometry(original.id(), rectangle(0, 2))
        assert source.addFeature(
            added_feature(source, "transaction-added", rectangle(20))
        )
        assert source.commitChanges(False)
        assert not check.errors, check.errors
        assert [event for event, _ in check.history] == [
            "geometry_modified",
            "feature_added",
        ]
        assert records(source)["transaction-added"]["geometry_revision"] == 1
        check.history.clear()
        assert source.deleteFeature(original.id())
        assert source.commitChanges()
        assert [event for event, _ in check.history] == ["feature_deleted"]
        assert not check.errors, check.errors


def dialog_lifecycle(app, root):
    from qgis.gui import QgsMapCanvas

    from labeling_tool.refinement import class_refinement_dialog as ui
    from labeling_tool.refinement.sam_session import SamSession

    source = layer(root, "dialog", positions=(0,))
    QgsProject.instance().addMapLayer(source)
    canvas = QgsMapCanvas()
    observed = []
    dialog = None

    def save_from_vertex_tool():
        observed.append(dialog._edit_tracker.suppressed)
        assert source.commitChanges(False)

    dialog = ui.ClassRefinementDialog(
        SimpleNamespace(
            mapCanvas=lambda: canvas,
            activeLayer=lambda: source,
            setActiveLayer=lambda _: None,
            actionVertexTool=lambda: SimpleNamespace(trigger=save_from_vertex_tool),
        ),
        SimpleNamespace(),
    )
    dialog._run_spec = dict(SPEC, run_dir=str(root))
    dialog._workspace = {"baseline_stream_id": "fusion:probe", "classes": {}}
    dialog._class_layers = {12: source.id()}
    history = []
    original = next(source.getFeatures())
    try:
        with ExitStack() as stack:
            for name in (
                "_update_manual_panel",
                "_update_actions",
                "_refresh_table",
                "_refresh_class_display",
                "_set_class_modified",
                "_active_layer_changed",
            ):
                stack.enter_context(patch.object(dialog, name))
            stack.enter_context(
                patch.object(dialog, "_local_topology_hint", return_value="ok")
            )
            stack.enter_context(
                patch.object(
                    dialog, "_optional_confidence_statistics", side_effect=confidence
                )
            )
            stack.enter_context(
                patch.object(
                    class_workspace,
                    "save_workspace",
                    side_effect=lambda _, workspace, **kw: workspace,
                )
            )
            stack.enter_context(
                patch.object(
                    class_workspace,
                    "append_history",
                    side_effect=lambda _, event, **data: history.append((event, data)),
                )
            )
            dialog._register_workspace_layer(12, source.id())
            session = SamSession(
                class_code=12,
                feature_id=original.id(),
                object_id=original["object_id"],
                mode="existing",
                candidate_geometry=rectangle(0, 2),
                current_revision=3,
                session_id="sam-dialog",
                candidate_score=0.9,
                state="candidate",
                started_at="probe",
            )
            dialog._adopt_candidate(session, edit=True)
            assert observed == [False], observed
            assert source.getFeature(original.id())["geometry_revision"] == 4
            assert source.getFeature(original.id())["edit_base"] == "sam3"
            assert [event for event, _ in history] == ["geometry_modified"], history
            assert not dialog._edit_tracker.suppressed
            assert source.rollBack()

            # Reopening a class while its transaction-mode layer stays editable
            # must keep tracking: that provider has no buffer-based recapture.
            dialog._workspace["classes"]["12"] = {"confirmed": True}
            history.clear()
            with (
                patch.object(
                    source,
                    "dataProvider",
                    return_value=SimpleNamespace(transaction=lambda: object()),
                ),
                patch.object(dialog, "_select_class_context"),
            ):
                assert source.startEditing()
                dialog._confirm_class(12, False)
                assert source.isEditable() and dialog._edit_tracker.has_session(12)
                assert source.changeGeometry(original.id(), rectangle(0, 1))
                assert source.commitChanges(False)
                assert [event for event, _ in history] == [
                    "class_reopened",
                    "geometry_modified",
                ], history
                assert source.getFeature(original.id())["geometry_revision"] == 5
                assert source.rollBack()
            dialog._workspace = None
            dialog.cleanup()
            assert not dialog._edit_tracker.has_session(12)
    finally:
        source.blockSignals(True)
        if source.isEditable():
            source.rollBack()
        dialog._workspace = None
        dialog._run_spec = None
        dialog.close()
        dialog.deleteLater()
        app.processEvents()
        QgsProject.instance().removeAllMapLayers()


app = QgsApplication([], False)
app.initQgis()
try:
    with tempfile.TemporaryDirectory(prefix="loess-edit-tracking-") as directory:
        scenario = sys.argv[2]
        globals()[scenario](app, Path(directory))
        print(scenario + ": passed", flush=True)
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
finally:
    app.exitQgis()
