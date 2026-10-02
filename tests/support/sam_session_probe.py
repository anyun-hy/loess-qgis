# ruff: noqa: E402
"""Native UI, geometry and persistence boundaries with a signal-driven SAM fake."""

from __future__ import annotations

import copy
import gc
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
    from qgis.core import QgsApplication, QgsPointXY, QgsProject, QgsVectorLayer
    from qgis.gui import QgsMapCanvas, QgsMapToolPan, QgsRubberBand
    from qgis.PyQt import sip
    from qgis.PyQt.QtCore import (
        QCoreApplication,
        QEvent,
        QObject,
        Qt,
        QTimer,
        pyqtSignal,
    )
    from qgis.PyQt.QtWidgets import QLabel, QPlainTextEdit, QPushButton
except ModuleNotFoundError:
    raise SystemExit(77)

from refinement_fixtures import SPEC, layer, rectangle, signature

from labeling_tool.refinement import class_refinement_dialog as ui
from labeling_tool.refinement.sam_map_preview import SamMapPreview
from labeling_tool.refinement.sam_session_panel import SamPanelSnapshot, SamSessionPanel


def child(owner, kind, name):
    widget = owner.findChild(kind, "sam" + name)
    assert widget is not None, name
    return widget


def button(owner, name):
    return child(owner, QPushButton, name)


def preview_band(canvas, color):
    bands = [
        item
        for item in canvas.scene().items()
        if isinstance(item, QgsRubberBand) and item.strokeColor().name() == color
    ]
    assert len(bands) <= 1, "A previous SAM overlay was left in the scene"
    return bands[0] if bands else None


def current_band(canvas):
    return preview_band(canvas, "#ffd400")


def candidate_band(canvas):
    return preview_band(canvas, "#00d7d7")


class Worker(QObject):
    ready = pyqtSignal(object)
    event_received = pyqtSignal(object)
    stopped = pyqtSignal(object)
    log_line = pyqtSignal(str, str)

    def __init__(self, scripts_dir, config, parent):
        super().__init__(parent)
        self.is_ready = False
        self.requests = []
        self.cancelled = []
        self.closed = []

    def start(self):
        pass

    def predict(self, request):
        self.requests.append(copy.deepcopy(request))

    def cancel(self, session_id):
        self.cancelled.append(session_id)

    def close_session(self, session_id):
        self.closed.append(session_id)

    def stop(self):
        self.is_ready = False
        self.stopped.emit({"expected": True})


def drain(app):
    for _ in range(3):
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def deliver(app, worker, event):
    QTimer.singleShot(0, lambda: worker.event_received.emit(event))
    drain(app)


def candidate(session_id, geometry=None):
    return {
        "event": "candidate_ready",
        "session_id": session_id,
        "geometry_wkt": (geometry or rectangle(0, 5)).asWkt(),
        "score": 0.91,
        "confidence_mean": 0.8,
        "confidence_std": 0.1,
        "crop_window": {"col_off": 0, "row_off": 0, "width": 512, "height": 512},
        "elapsed_sec": 0.25,
    }


@contextmanager
def session_dialog(app, root):
    source = layer(root, "sam", positions=(0, 10))
    QgsProject.instance().addMapLayer(source)
    canvas = QgsMapCanvas()
    canvas.setLayers([source])
    canvas.setDestinationCrs(source.crs())
    canvas.setExtent(source.extent())
    pan = QgsMapToolPan(canvas)
    canvas.setMapTool(pan)
    dialog = ui.ClassRefinementDialog(
        SimpleNamespace(
            mapCanvas=lambda: canvas,
            activeLayer=lambda: source,
            setActiveLayer=lambda _: None,
        ),
        SimpleNamespace(),
    )
    dialog._run_spec = dict(
        SPEC, run_dir=str(root), raster={"path": "synthetic.tif", "crs": "EPSG:3857"}
    )
    dialog._workspace = {"baseline_stream_id": "fusion:probe", "classes": {}}
    dialog._class_layers = {12: source.id()}
    dialog._sam_config = {
        "checkpoint_sha256": "a" * 64,
        "version": "probe",
        "buffer_px": 24,
        "effective_device": "cpu",
    }
    dialog._eligible_fusions = [
        {"stream_id": "fusion:probe", "paths": {"confidence_mosaic": "confidence.tif"}}
    ]
    source.editingStarted.connect(lambda: dialog._editing_started(12))
    sessions = []
    with ExitStack() as stack:
        stack.enter_context(patch.object(ui, "Sam3WorkerRunner", Worker))
        for name in (
            "_update_actions",
            "_refresh_table",
            "_select_class_context",
            "_mark_class_modified",
        ):
            stack.enter_context(patch.object(dialog, name))
        stack.enter_context(patch.object(dialog, "_sam_available", return_value=True))
        stack.enter_context(
            patch.object(dialog, "_local_topology_hint", return_value="无")
        )
        stack.enter_context(
            patch.object(
                dialog, "_optional_confidence_statistics", return_value=(0.8, 0.1, "")
            )
        )
        stack.enter_context(patch.object(ui.QMessageBox, "information"))
        warning = stack.enter_context(patch.object(ui.QMessageBox, "warning"))
        stack.enter_context(
            patch.object(
                ui.class_workspace,
                "save_workspace",
                side_effect=lambda _, value, **kwargs: value,
            )
        )
        stack.enter_context(patch.object(ui.class_workspace, "append_history"))
        stack.enter_context(
            patch.object(
                ui.class_workspace,
                "append_sam_session",
                side_effect=lambda _, value: sessions.append(copy.deepcopy(value)),
            )
        )
        try:
            yield dialog, source, canvas, pan, sessions, warning
        finally:
            dialog._workspace = None
            source.blockSignals(True)
            if source.isEditable():
                source.rollBack()
            dialog.cleanup()
            dialog.close()
            dialog.deleteLater()
            drain(app)
            QgsProject.instance().removeAllMapLayers()
            canvas.close()


def begin(dialog, canvas, *, missed=False, ready=True):
    dialog._begin_sam(12, missed=missed)
    session_id = dialog._workspace["active_sam_session_id"]
    assert session_id and button(dialog, "Cancel").isEnabled()
    canvas.mapTool().canvasClicked.emit(
        QgsPointXY(22 if missed else 2, 2), Qt.MouseButton.LeftButton
    )
    worker = dialog._worker
    assert worker is not None
    if ready:
        worker.is_ready = True
        worker.ready.emit({"event": "worker_ready"})
        assert worker.requests[-1]["session_id"] == session_id
    return worker, session_id


def disk_signature(source):
    saved = QgsVectorLayer(source.source(), "saved", "ogr")
    assert saved.isValid()
    return signature(saved)


def existing_keep(app, root):
    with session_dialog(app, root) as (dialog, source, canvas, pan, sessions, warning):
        before = signature(source)
        worker, session_id = begin(dialog, canvas)
        assert canvas.mapTool() is pan
        request = worker.requests[-1]
        assert request == {
            "session_id": session_id,
            "run_id": SPEC["run_id"],
            "raster": "synthetic.tif",
            "confidence_mosaic": "confidence.tif",
            "click_raster": {"x": 2.0, "y": 2.0},
            "geometry_bounds": {"xmin": 0.0, "ymin": 0.0, "xmax": 4.0, "ymax": 4.0},
            "crop_size_px": 512,
            "buffer_px": 24,
            "class_code": 12,
            "object_id": "sam-0",
            "part_id": "007",
            "checkpoint_sha256": "a" * 64,
            "sam_version": "probe",
            "device": "cpu",
        }
        deliver(app, worker, candidate(session_id))
        assert (
            button(dialog, "AdoptCandidate").isEnabled()
            and button(dialog, "KeepCurrent").isEnabled()
        )
        assert signature(source) == disk_signature(source) == before
        assert candidate_band(canvas) is not None and current_band(canvas) is not None
        bands = (candidate_band(canvas), current_band(canvas))
        button(dialog, "KeepCurrent").click()
        assert dialog._active_session is None and candidate_band(canvas) is None
        assert all(band not in canvas.scene().items() for band in bands)
        assert sessions[-1]["decision"] == "kept_current"
        assert (
            sessions[-1]["after_geometry_hash"] == sessions[-1]["before_geometry_hash"]
        )
        assert "candidate_geometry" not in sessions[-1]
        assert signature(source) == before and not source.isEditable()
        warning.assert_not_called()


def missed_adopt(app, root):
    with session_dialog(app, root) as (dialog, source, canvas, _, sessions, warning):
        worker, session_id = begin(dialog, canvas, missed=True)
        assert worker.requests[-1]["geometry_bounds"] is None
        assert worker.requests[-1]["object_id"] == ""
        deliver(app, worker, candidate(session_id, rectangle(20)))
        assert (
            not button(dialog, "KeepCurrent").isEnabled()
            and not button(dialog, "EditCurrent").isEnabled()
        )
        assert source.featureCount() == 2
        button(dialog, "AdoptCandidate").click()
        assert dialog._active_session is None and source.featureCount() == 3
        assert len(disk_signature(source)) == 3
        record = sessions[-1]
        assert record["object_id"].startswith(SPEC["run_id"] + "_new_")
        assert record["before_geometry_hash"] == "" and record["after_geometry_hash"]
        assert record["decision"] == "adopted" and record["mode"] == "missed"
        warning.assert_not_called()


def retry_and_late(app, root):
    with session_dialog(app, root) as (dialog, source, canvas, _, sessions, _):
        before = signature(source)
        worker, first_id = begin(dialog, canvas)
        deliver(
            app,
            worker,
            {"event": "failed", "session_id": first_id, "error": "first attempt"},
        )
        assert (
            button(dialog, "Retry").isEnabled()
            and not button(dialog, "AdoptCandidate").isEnabled()
        )
        deliver(app, worker, candidate(first_id))
        assert not button(dialog, "AdoptCandidate").isEnabled(), (
            "A late candidate must not revive failed state"
        )
        button(dialog, "Retry").click()
        second_id = worker.requests[-1]["session_id"]
        assert (
            second_id != first_id
            and worker.cancelled[-1] == worker.closed[-1] == first_id
        )
        assert (
            sessions[-1]["session_id"] == first_id
            and sessions[-1]["decision"] == "failed"
        )
        for event in (
            candidate(first_id),
            candidate(""),
            {"event": "failed", "session_id": first_id, "error": "late"},
        ):
            deliver(app, worker, event)
        assert (
            not button(dialog, "AdoptCandidate").isEnabled()
            and not button(dialog, "Retry").isEnabled()
        )
        deliver(app, worker, candidate(second_id))
        band = candidate_band(canvas)
        assert button(dialog, "AdoptCandidate").isEnabled()
        deliver(app, worker, candidate(second_id, rectangle(100)))
        assert candidate_band(canvas) is band, (
            "Duplicate result must not replace a candidate"
        )
        worker.stopped.emit({"expected": False, "returncode": 9})
        assert button(dialog, "Retry").isEnabled()
        button(dialog, "Retry").click()
        third_id = worker.requests[-1]["session_id"]
        assert third_id != second_id
        assert band not in canvas.scene().items(), (
            "Retry must remove the failed candidate overlay"
        )
        assert candidate_band(canvas) is None and current_band(canvas) is not None
        deliver(app, worker, candidate(second_id))
        assert not button(dialog, "AdoptCandidate").isEnabled()
        deliver(app, worker, candidate(third_id))
        assert button(dialog, "AdoptCandidate").isEnabled()
        button(dialog, "Cancel").click()
        assert dialog._active_session is None and signature(source) == before
        deliver(app, worker, candidate(second_id))
        assert dialog._active_session is None and candidate_band(canvas) is None


def pending_cancel(app, root):
    with session_dialog(app, root) as (dialog, source, canvas, _, sessions, _):
        before = signature(source)
        worker, first_id = begin(dialog, canvas, ready=False)
        assert worker.requests == []
        button(dialog, "Cancel").click()
        worker.is_ready = True
        worker.ready.emit({"event": "worker_ready"})
        assert worker.requests == [] and dialog._active_session is None
        assert sessions[-1]["session_id"] == first_id
        worker.is_ready = False
        _, second_id = begin(dialog, canvas, ready=False)
        worker.stopped.emit({"expected": False, "returncode": 7})
        worker.is_ready = True
        worker.ready.emit({"event": "worker_ready"})
        assert worker.requests == [] and button(dialog, "Retry").isEnabled()
        button(dialog, "Retry").click()
        assert len(worker.requests) == 1
        assert worker.requests[-1]["session_id"] != second_id
        button(dialog, "Cancel").click()
        assert signature(source) == before


def invalid_candidate(app, root):
    with session_dialog(app, root) as (dialog, source, canvas, _, _, _):
        before = signature(source)
        worker, session_id = begin(dialog, canvas)
        event = candidate(session_id)
        event["geometry_wkt"] = "POLYGON EMPTY"
        deliver(app, worker, event)
        assert (
            button(dialog, "Retry").isEnabled()
            and not button(dialog, "AdoptCandidate").isEnabled()
        )
        assert candidate_band(canvas) is None and signature(source) == before
        button(dialog, "Retry").click()
        deliver(
            app, worker, {"event": "failed", "session_id": "", "error": "global error"}
        )
        assert (
            button(dialog, "Retry").isEnabled()
            and child(dialog, QPlainTextEdit, "SessionError").toPlainText()
            == "global error"
        )
        button(dialog, "Retry").click()
        worker.stopped.emit({"expected": False, "returncode": 9})
        assert (
            "returncode=9"
            in child(dialog, QPlainTextEdit, "SessionError").toPlainText()
        )
        assert button(dialog, "Retry").isEnabled() and signature(source) == before


def adopt_provenance(app, root):
    with session_dialog(app, root) as (dialog, source, canvas, _, sessions, warning):
        worker, session_id = begin(dialog, canvas)
        deliver(app, worker, candidate(session_id))
        fid = source.selectedFeatureIds()[0]
        original_commit = source.commitChanges

        def provider_normalizes():
            assert source.changeGeometry(fid, rectangle(0, 3))
            return original_commit()

        with patch.object(source, "commitChanges", side_effect=provider_normalizes):
            button(dialog, "AdoptCandidate").click()
        record = sessions[-1]
        persisted = next(f for f in source.getFeatures() if f["object_id"] == "sam-0")
        expected = ui.class_workspace.geometry_hash(persisted.geometry())
        assert record["after_geometry_hash"] == expected
        assert record["candidate_geometry_hash"] != expected
        assert (
            persisted["geometry_source"] == "sam3"
            and persisted["geometry_revision"] == 4
        )
        assert signature(source) == disk_signature(source)
        warning.assert_not_called()


def edit_choice(app, root, candidate_edit):
    with session_dialog(app, root) as (dialog, source, canvas, _, sessions, warning):
        before = signature(source)
        worker, session_id = begin(dialog, canvas)
        if candidate_edit:
            deliver(app, worker, candidate(session_id))
            button(dialog, "EditCandidate").click()
        else:
            deliver(
                app,
                worker,
                {"event": "failed", "session_id": session_id, "error": "retryable"},
            )
            button(dialog, "EditCurrent").click()
        assert dialog._active_session is None and source.isEditable()
        assert dialog._edit_tracker.has_session(12)
        assert disk_signature(source) == before
        assert (signature(source) != before) is candidate_edit
        record = sessions[-1]
        assert record["decision"] == ("edit_sam3" if candidate_edit else "edit_current")
        assert (
            record["after_geometry_hash"]
            == record[
                "candidate_geometry_hash" if candidate_edit else "before_geometry_hash"
            ]
        )
        warning.assert_not_called()


def edit_current(app, root):
    edit_choice(app, root, False)


def edit_candidate(app, root):
    edit_choice(app, root, True)


def panel_controls(app, root):
    panel = SamSessionPanel()
    decisions, retries = [], []
    panel.decision_requested.connect(decisions.append)
    panel.retry_requested.connect(lambda: retries.append(True))
    actions = {
        "KeepCurrent": ("保留当前", "kept_current"),
        "AdoptCandidate": ("采用 SAM3", "adopted"),
        "EditCurrent": ("编辑当前", "edit_current"),
        "EditCandidate": ("编辑 SAM3", "edit_sam3"),
        "Retry": ("重试", None),
        "Cancel": ("取消", "cancelled"),
    }
    try:
        for existing in (False, True):
            for state in ("idle", "waiting_click", "inference", "candidate", "failed"):
                panel.render(
                    SamPanelSnapshot(
                        state=state,
                        existing=existing,
                        message="当前类别 12",
                        topology_hint="无",
                        error="worker failed" if state == "failed" else "",
                    )
                )
                expected = set() if state == "idle" else {"Cancel"}
                if state == "candidate":
                    expected.update(("AdoptCandidate", "EditCandidate"))
                    if existing:
                        expected.add("KeepCurrent")
                if state == "failed":
                    expected.add("Retry")
                if existing and state in ("candidate", "failed"):
                    expected.add("EditCurrent")
                assert panel.isHidden() is (state == "idle")
                assert child(panel, QLabel, "SessionLabel").text() == "当前类别 12"
                assert child(panel, QLabel, "TopologyHint").text() == "局部拓扑提示: 无"
                error = child(panel, QPlainTextEdit, "SessionError")
                assert error.isHidden() is (state != "failed")
                assert error.isReadOnly()
                decisions.clear()
                retries.clear()
                for name, (label, _) in actions.items():
                    control = button(panel, name)
                    assert control.text() == label
                    assert control.isEnabled() is (name in expected), (
                        state,
                        existing,
                        name,
                    )
                    control.click()
                assert decisions == [
                    decision
                    for name, (_, decision) in actions.items()
                    if name in expected and decision is not None
                ]
                assert retries == ([True] if "Retry" in expected else [])
        panel.render(SamPanelSnapshot(state="inference"))
        panel.replace_log("log detail")
        assert error.toPlainText() == "log detail" and error.isHidden()
        panel.render(SamPanelSnapshot(state="failed", error="failure detail"))
        assert error.toPlainText() == "failure detail" and not error.isHidden()
        panel.render(SamPanelSnapshot())
        assert error.toPlainText() == "" and error.isHidden()
    finally:
        panel.deleteLater()
        drain(app)


def map_resources(app, root):
    source = layer(root, "preview", positions=(0,))
    canvas = QgsMapCanvas()
    canvas.setDestinationCrs(source.crs())
    owner = SamMapPreview(canvas)
    before = signature(source)
    try:
        owner.show_current(rectangle(0), source)
        current = current_band(canvas)
        assert current is not None and current.asGeometry().equals(rectangle(0))
        assert current.width() == 2 and current.fillColor().alpha() == 30
        owner.show_candidate(rectangle(10), source)
        first = candidate_band(canvas)
        assert first is not None and first.asGeometry().equals(rectangle(10))
        assert first.width() == 2 and first.fillColor().alpha() == 35
        owner.show_candidate(rectangle(20), source)
        second = candidate_band(canvas)
        assert second is not first and second.asGeometry().equals(rectangle(20))
        assert first not in canvas.scene().items() and current_band(canvas) is current
        owner.clear_candidate()
        owner.clear_candidate()
        assert candidate_band(canvas) is None and current_band(canvas) is current
        assert second not in canvas.scene().items()
        owner.show_candidate(rectangle(30), source)
        third = candidate_band(canvas)
        owner.show_current(rectangle(40), source)
        assert (
            third not in canvas.scene().items()
            and current not in canvas.scene().items()
        )
        assert current_band(canvas).asGeometry().equals(rectangle(40))
        assert candidate_band(canvas) is None
        owner.clear()
        owner.cleanup()
        owner.cleanup()
        assert current_band(canvas) is candidate_band(canvas) is None
        assert signature(source) == before and not source.isEditable()
    finally:
        owner.cleanup()
        drain(app)
        canvas.close()


def map_picker_lifecycle(app, root):
    canvas = QgsMapCanvas()
    pan = QgsMapToolPan(canvas)
    canvas.setMapTool(pan)
    owner = SamMapPreview(canvas)
    clicked = []
    picker_destroyed = []

    def finish_pick(point, button):
        clicked.append((point, button))
        owner.restore_map_tool()
        gc.collect()
        assert not picker_destroyed, "The emitting QObject must survive its callback"
        assert canvas.mapTool() is pan

    owner.point_clicked.connect(finish_pick)
    try:
        owner.start_pick()
        # SIP may collect the Python wrapper while its parent still owns the
        # C++ tool. QObject.destroyed measures the native lifetime that matters.
        canvas.mapTool().destroyed.connect(lambda: picker_destroyed.append(True))
        canvas.mapTool().canvasClicked.emit(QgsPointXY(1, 2), Qt.MouseButton.LeftButton)
        assert clicked == [(QgsPointXY(1, 2), Qt.MouseButton.LeftButton)]
        drain(app)
        gc.collect()
        assert picker_destroyed == [True]
        owner.point_clicked.disconnect(finish_pick)
        owner.point_clicked.connect(lambda *args: clicked.append(args))
        owner.start_pick()
        old = canvas.mapTool()
        owner.start_pick()
        active = canvas.mapTool()
        assert active is not old
        if not sip.isdeleted(old):
            old.canvasClicked.emit(QgsPointXY(3, 4), Qt.MouseButton.LeftButton)
        assert len(clicked) == 1, "Retired pickers must no longer emit decisions"
        outside = QgsMapToolPan(canvas)
        canvas.setMapTool(outside)
        owner.cleanup()
        owner.cleanup()
        assert canvas.mapTool() is outside, "Do not restore over the user's new tool"
        if not sip.isdeleted(active):
            active.canvasClicked.emit(QgsPointXY(5, 6), Qt.MouseButton.LeftButton)
        assert len(clicked) == 1
        drain(app)
        canvas.unsetMapTool(outside)
        assert canvas.mapTool() is None
        owner.start_pick()
        owner.restore_map_tool()
        assert canvas.mapTool() is None
        drain(app)
        previous = QgsMapToolPan(canvas)
        canvas.setMapTool(previous)
        owner.start_pick()
        previous.deleteLater()
        drain(app)
        owner.restore_map_tool()
        assert canvas.mapTool() is None
    finally:
        owner.cleanup()
        drain(app)
        canvas.close()


def session_exit(app, root):
    with session_dialog(app, root) as (dialog, source, canvas, pan, sessions, _):
        before = signature(source)
        dialog._begin_sam(12)
        picker = canvas.mapTool()
        assert picker is not pan
        outside = QgsMapToolPan(canvas)
        canvas.setMapTool(outside)
        button(dialog, "Cancel").click()
        assert canvas.mapTool() is outside and dialog._active_session is None
        if not sip.isdeleted(picker):
            picker.canvasClicked.emit(QgsPointXY(2, 2), Qt.MouseButton.LeftButton)
        assert dialog._worker is None
        worker, session_id = begin(dialog, canvas)
        deliver(app, worker, candidate(session_id))
        assert current_band(canvas) is not None and candidate_band(canvas) is not None
        with patch.object(ui.QMessageBox, "question", return_value=ui.NO):
            dialog.close()
        assert dialog._active_session is not None and candidate_band(canvas) is not None
        with patch.object(ui.QMessageBox, "question", return_value=ui.YES):
            dialog.close()
        assert dialog._active_session is None and dialog._worker is None
        assert current_band(canvas) is candidate_band(canvas) is None
        assert canvas.mapTool() is outside and sessions[-1]["decision"] == "cancelled"
        assert signature(source) == before
        drain(app)
        dialog._begin_sam(12)
        assert canvas.mapTool() is not outside
        dialog.cleanup()
        assert canvas.mapTool() is outside and dialog._active_session is None
        assert current_band(canvas) is candidate_band(canvas) is None
        worker, session_id = begin(dialog, canvas)
        deliver(app, worker, candidate(session_id))
        with patch.object(ui.QgsApplication, "taskManager") as manager:
            dialog.set_run({}, dict(dialog._run_spec), {}, "")
            manager.return_value.addTask.assert_called_once()
        assert dialog._active_session is None and canvas.mapTool() is outside
        assert current_band(canvas) is candidate_band(canvas) is None
        assert child(dialog, SamSessionPanel, "SessionPanel").isHidden()
        assert signature(source) == before


app = QgsApplication([], False)
app.initQgis()
try:
    with tempfile.TemporaryDirectory(prefix="loess-sam-session-") as directory:
        globals()[sys.argv[2]](app, Path(directory))
        print(sys.argv[2] + ": passed", flush=True)
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
finally:
    app.exitQgis()
