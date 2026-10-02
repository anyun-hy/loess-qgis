# ruff: noqa: E402
"""Exercise the real refinement dialog wiring through its focused review panel."""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
try:
    from qgis.core import (
        QgsApplication,
        QgsFeature,
        QgsGeometry,
        QgsProject,
        QgsVectorLayer,
    )
    from qgis.gui import QgsMapCanvas
    from qgis.PyQt.QtCore import QCoreApplication, QEvent, Qt
    from qgis.PyQt.QtTest import QTest
    from qgis.PyQt.QtWidgets import QPushButton, QScrollArea, QTableWidget
except ModuleNotFoundError:
    raise SystemExit(77)

from labeling_tool.refinement import class_refinement_dialog as ui
from labeling_tool.refinement.class_review_panel import ClassReviewPanel
from labeling_tool.shared.contracts.run_spec import CLASS_ORDER


def layer(name: str) -> QgsVectorLayer:
    value = QgsVectorLayer("Polygon?crs=EPSG:4326", name, "memory")
    assert value.isValid()
    value.dataProvider().addAttributes([])
    feature = QgsFeature(value.fields())
    feature.setGeometry(QgsGeometry.fromWkt("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))"))
    assert value.dataProvider().addFeature(feature)
    return value


def workspace():
    return {
        "baseline_stream_id": "fusion:probe",
        "classes": {
            str(code): {"class_code": code, "confirmed": False, "feature_count": 0}
            for code in CLASS_ORDER
        },
    }


def run(app: QgsApplication, screenshot_directory: Path | None = None) -> None:
    project = QgsProject.instance()
    source = layer("class-12")
    target = layer("class-13")
    project.addMapLayer(source)
    project.addMapLayer(target)
    canvas = QgsMapCanvas()
    canvas.setLayers([source, target])
    active = [source]

    def set_active(value):
        active[0] = value

    def set_visible(layer_id, visible):
        tree = project.layerTreeRoot().findLayer(layer_id)
        assert tree is not None
        tree.setItemVisibilityChecked(bool(visible))

    dialog = ui.ClassRefinementDialog(
        SimpleNamespace(
            mapCanvas=lambda: canvas,
            activeLayer=lambda: active[0],
            setActiveLayer=set_active,
        ),
        SimpleNamespace(set_layer_visibility=set_visible),
    )
    dialog._run_spec = {"run_id": "review-probe", "run_dir": tempfile.gettempdir()}
    dialog._workspace = workspace()
    dialog._class_layers = {12: source.id(), 13: target.id()}
    panel = dialog.findChild(ClassReviewPanel, "classReviewPanel")
    assert panel is not None
    try:
        dialog._sam_available = lambda: True
        dialog._refresh_table()
        table = panel.findChild(QTableWidget, "classReviewTable")
        assert table is not None
        # Selecting row 13 uses the panel signal, maps to the real dialog,
        # activates the matching QGIS class layer and makes it visible.
        table.setCurrentCell(CLASS_ORDER.index(13), 1)
        assert active[0].id() == target.id()
        assert project.layerTreeRoot().findLayer(target.id()).itemVisibilityChecked()
        dialog.show()
        app.processEvents()

        sam_events = []
        panel.sam_requested.disconnect(dialog._request_sam)
        panel.sam_requested.connect(
            lambda code, missed: sam_events.append((code, missed))
        )
        panel.findChild(QPushButton, "classReviewSam").menu().actions()[0].trigger()
        panel.findChild(QPushButton, "classReviewSam").menu().actions()[1].trigger()
        assert sam_events == [(13, False), (13, True)]

        with (
            patch.object(ui.class_review, "commit_class_review"),
            patch.object(ui.class_workspace, "append_history"),
            patch.object(
                ui.class_workspace,
                "save_workspace",
                side_effect=lambda _spec, value, **_kw: value,
            ),
        ):
            confirm = panel.findChild(QPushButton, "classReviewConfirm")
            confirm.setFocus()
            QTest.keyClick(confirm, Qt.Key.Key_Return)
            assert dialog._workspace["classes"]["13"]["confirmed"]
            confirm.click()
        assert not dialog._workspace["classes"]["13"]["confirmed"]

        assert target.startEditing()
        feature_id = next(target.getFeatures()).id()
        assert target.changeGeometry(
            feature_id,
            QgsGeometry.fromWkt("POLYGON((0 0, .5 0, .5 1, 0 1, 0 0))"),
        )
        dialog._refresh_table()
        assert not confirm.isEnabled()
        assert not panel.findChild(QPushButton, "classReviewSam").isEnabled()
        target.rollBack()
        dialog._refresh_table()

        dialog._accepted_task = SimpleNamespace(
            commit_started=False,
            progress_message="正在后台校验 accepted_labels",
        )
        dialog._refresh_table()
        assert not table.isEnabled()
        assert "accepted_labels" in panel._hint.text()
        dialog._accepted_task = None

        scroll = dialog.findChild(QScrollArea, "classRefinementScrollArea")
        assert scroll is not None
        for width, height, filename in (
            (960, 540, "class-review-dialog-960x540.png"),
            (640, 360, "class-review-dialog-640x360.png"),
        ):
            dialog.resize(width, height)
            dialog.show()
            app.processEvents()
            assert dialog.width() <= width and dialog.height() <= height
            assert scroll.verticalScrollBar().maximum() > 0
            scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
            app.processEvents()
            write_button = dialog.admission_summary_panel._write_button
            action_center = write_button.mapTo(
                scroll.viewport(), write_button.rect().center()
            )
            assert scroll.viewport().rect().contains(action_center)
            if screenshot_directory is not None:
                screenshot_directory.mkdir(parents=True, exist_ok=True)
                image = screenshot_directory / filename
                assert dialog.grab().save(str(image)) and image.stat().st_size > 0
    finally:
        dialog._workspace = None
        dialog.cleanup()
        dialog.close()
        dialog.deleteLater()
        project.removeAllMapLayers()
        canvas.close()


app = QgsApplication([], False)
app.initQgis()
try:
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    run(app, output)
    print("class refinement review: passed", flush=True)
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
finally:
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.exitQgis()
