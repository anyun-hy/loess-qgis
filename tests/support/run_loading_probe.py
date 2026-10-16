# ruff: noqa: E402
"""Real temporary rasters/GeoPackages, native tasks, and a controlled SAM worker."""

import copy
import json
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
try:
    from osgeo import gdal, osr
    from qgis.core import (
        QgsApplication,
        QgsProject,
        QgsVectorFileWriter,
        QgsVectorLayer,
    )
    from qgis.gui import QgsMapCanvas
    from qgis.PyQt.QtCore import QObject, QTimer, pyqtSignal
except ModuleNotFoundError:
    raise SystemExit(77)

from refinement_fixtures import layer

from labeling_tool.main.results_workspace_panel import ResultsWorkspacePanel
from labeling_tool.qgis_support.layer_manager import LayerManager
from labeling_tool.qgis_support.qgis_writer import write_vector_layer
from labeling_tool.refinement import class_refinement_dialog as ui
from labeling_tool.refinement import class_workspace
from labeling_tool.refinement.manual_run_loader import load_manual_run
from labeling_tool.shared.contracts.run_spec import (
    CLASS_ORDER,
    atomic_write_json,
    sha256_file,
    source_raster_identity,
)


def wait(app, predicate):
    end = time.monotonic() + 12
    while not predicate() and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)
    assert predicate(), "native task did not complete"


def raster(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = gdal.GetDriverByName("GTiff").Create(str(path), 32, 16, 1, gdal.GDT_Byte)
    dataset.SetGeoTransform((0, 1, 0, 16, 0, -1))
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(3857)
    dataset.SetProjection(crs.ExportToWkt())
    dataset.GetRasterBand(1).Fill(1)
    dataset = None


def fixture(root, initialized=True):
    root.mkdir(parents=True, exist_ok=True)
    source = layer(root, "source", positions=(0, 10))
    image = root / "original.tif"
    raster(image)
    streams = []
    for kind, directory in [("fusion", "fusion/probe"), ("model", "models/probe")]:
        folder = root / directory
        folder.mkdir(parents=True)
        paths = {}
        for key in ("mask_mosaic", "confidence_mosaic"):
            path = folder / f"{key}.tif"
            raster(path)
            paths[key] = str(path)
        for key in ("semantic_polygons_raw", "semantic_polygons"):
            path = folder / f"{key}.gpkg"
            options = QgsVectorFileWriter.SaveVectorOptions()
            options.driverName = "GPKG"
            options.layerName = key
            assert write_vector_layer(source, path, options)[0] == 0
            paths[key] = str(path)
        report = folder / "boundary_fitting_report.json"
        atomic_write_json(
            report,
            {
                "status": "passed",
                "validation": {"passed": True},
                "input_sha256": sha256_file(paths["semantic_polygons_raw"]),
                "output_sha256": sha256_file(paths["semantic_polygons"]),
            },
        )
        paths["boundary_fitting_report"] = str(report)
        streams.append(
            {
                "stream_id": f"{kind}:probe",
                "kind": kind,
                "status": "ready",
                "fusion_profile_id": "probe" if kind == "fusion" else "",
                "model_id": "probe" if kind == "model" else "",
                "boundary_fitting_status": "passed",
                "paths": paths,
                "output_sha256": {
                    key: sha256_file(path) for key, path in paths.items()
                },
            }
        )
    profile = root / "fusion_profile_snapshot.json"
    atomic_write_json(
        profile,
        {"profile_id": "probe", "status": "approved", "approval": {"passed": True}},
    )
    spec = {
        "schema_version": 2,
        "run_id": "manual-probe",
        "run_dir": str(root),
        "accepted_target_gpkg": str(root / "accepted.gpkg"),
        "fusion": {
            "profile_id": "probe",
            "snapshot_path": str(profile),
            "sha256": sha256_file(profile),
        },
        "raster": {
            "path": str(image),
            "crs": "EPSG:3857",
            "transform": [1, 0, 0, 0, -1, 16],
            "file_identity": source_raster_identity(image),
        },
    }
    atomic_write_json(root / "run_spec.json", spec)
    manifest = {
        "schema_version": 2,
        "run_id": spec["run_id"],
        "run_spec_sha256": sha256_file(root / "run_spec.json"),
        "status": "ready",
        "streams": streams,
    }
    atomic_write_json(root / "run_manifest.json", manifest)
    if initialized:
        workspace = class_workspace.initialize_workspace(spec, streams[0])
        workspace["classes"]["12"].update(
            confirmed=True, modified=True, state="confirmed"
        )
        atomic_write_json(root / "classes/workspace.json", workspace)
        (root / "classes/edit_history.jsonl").write_text('{"action":"human_review"}\n')
    return spec, {
        "run_id": spec["run_id"],
        "run_dir": str(root),
        "ready_streams": streams,
    }


class Worker(QObject):
    ready = pyqtSignal(object)
    event_received = pyqtSignal(object)
    stopped = pyqtSignal(object)
    log_line = pyqtSignal(str, str)
    instances = []

    def __init__(self, scripts, config, parent):
        super().__init__(parent)
        self.is_running = self.is_ready = False
        self.requests = []
        self.__class__.instances.append(self)

    def start(self):
        self.is_running = True
        QTimer.singleShot(10, self.deliver_ready)

    def deliver_ready(self):
        if self.is_running:
            self.is_ready = True
            self.ready.emit(
                {
                    "checkpoint_sha256": "a" * 64,
                    "device": "cpu",
                    "sam_version": "fixture",
                    "buffer_px": 24,
                }
            )

    def stop(self):
        self.is_running = self.is_ready = False
        self.stopped.emit({"expected": True})

    def cancel(self, *_args):
        pass

    def close_session(self, *_args):
        pass

    def predict(self, request):
        self.requests.append(copy.deepcopy(request))


def initialize(app, root, copied=False):
    original = root / "original_run"
    fixture(original, initialized=False)
    target = root / "copied_run" if copied else original
    if copied:
        shutil.copytree(original, target)
        shutil.rmtree(original)
    panel = ResultsWorkspacePanel()
    try:
        panel.set_run_directory(target)
        wait(app, lambda: not panel.busy)
        assert panel.initialize_btn.isEnabled(), panel.status_label.text()
        assert not (target / "classes/workspace.json").exists()
        panel.initialize_btn.click()
        wait(app, lambda: not panel.busy)
        assert "已初始化" in panel.status_label.text(), panel.status_label.text()
        data = json.loads((target / "classes/workspace.json").read_text())
        assert set(data["classes"]) == {str(code) for code in CLASS_ORDER}
        assert data["feature_count"] == 2
        before = {
            path: path.read_bytes()
            for path in (target / "classes").iterdir()
            if path.is_file()
        }
        panel._initialize()
        assert not panel.busy and all(
            path.read_bytes() == content for path, content in before.items()
        )
    finally:
        panel.cleanup()


def copied_initialize(app, root):
    initialize(app, root, copied=True)


def copied_rebuild(app, root):
    original = root / "original"
    spec, result = fixture(original)
    workspace_path = original / "classes/workspace.json"
    workspace = json.loads(workspace_path.read_text())
    workspace["classes"]["12"].update(confirmed=False, modified=False, state="editing")
    atomic_write_json(workspace_path, workspace)
    (original / "classes/edit_history.jsonl").write_text("")
    stream = result["ready_streams"][0]
    review = original / "fusion/probe/review_polygons.gpkg"
    shutil.copyfile(stream["paths"]["semantic_polygons"], review)
    manifest_path = original / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["streams"][0]["review_polygons"] = str(review)
    atomic_write_json(manifest_path, manifest)
    target = root / "copied"
    shutil.copytree(original, target)
    shutil.rmtree(original)
    panel = ResultsWorkspacePanel()
    try:
        panel.set_run_directory(target)
        wait(app, lambda: not panel.busy)
        assert panel.initialize_btn.isEnabled(), panel.status_label.text()
        assert "重建" in panel.initialize_btn.text()
        record = panel._bundle["workspace"]["classes"]["12"]
        mounted = class_workspace.working_layer(record)
        mounted.setCustomProperty("labeling_tool/workspace_path", record["path"])
        QgsProject.instance().addMapLayer(mounted)
        before = (target / "classes/workspace.json").read_bytes()
        with (
            patch.object(ui.QMessageBox, "question", return_value=ui.YES),
            patch.object(ui.QMessageBox, "warning") as warning,
        ):
            panel.initialize_btn.click()
            assert not panel.busy and warning.called
        assert (target / "classes/workspace.json").read_bytes() == before
        QgsProject.instance().removeAllMapLayers()
        # An audit arriving after the panel's probe must still block replacement.
        workspace_path = target / "classes/workspace.json"
        audited = json.loads(workspace_path.read_text())
        audited["classes"]["12"]["confirmed"] = True
        atomic_write_json(workspace_path, audited)
        protected = {
            path: path.read_bytes()
            for path in target.glob("classes/*")
            if path.is_file()
        }
        with patch.object(ui.QMessageBox, "question", return_value=ui.YES):
            panel.initialize_btn.click()
        wait(app, lambda: not panel.busy)
        assert "禁止覆盖" in panel.status_label.text(), panel.status_label.text()
        assert all(path.read_bytes() == value for path, value in protected.items())
        audited["classes"]["12"]["confirmed"] = False
        atomic_write_json(workspace_path, audited)
        panel.set_run_directory(target)
        wait(app, lambda: not panel.busy)
        initialize_workspace = class_workspace.initialize_workspace

        def traced_initialize(*args, **kwargs):
            try:
                return initialize_workspace(*args, **kwargs)
            except Exception:
                traceback.print_exc()
                raise

        with (
            patch.object(ui.QMessageBox, "question", return_value=ui.YES),
            patch.object(
                class_workspace, "initialize_workspace", new=traced_initialize
            ),
        ):
            panel.initialize_btn.click()
            wait(app, lambda: not panel.busy)
        assert "已初始化" in panel.status_label.text(), panel.status_label.text()
        current = json.loads((target / "classes/workspace.json").read_text())
        assert current["baseline_source_path"] == str(
            target.resolve() / "fusion/probe/review_polygons.gpkg"
        )
        assert current["feature_count"] == 2
        assert not (target / "classes/edit_history.jsonl").read_text()
    finally:
        panel.cleanup()
        QgsProject.instance().removeAllMapLayers()


def load_scope(app, root, scope):
    spec, result = fixture(root / "run")
    directory = Path(spec["run_dir"])
    protected = {
        path: path.read_bytes()
        for path in directory.glob("classes/*")
        if path.is_file()
    }
    if scope == "classes":
        shutil.rmtree(directory / "fusion")
        shutil.rmtree(directory / "models")
    bundle = load_manual_run(directory, load_scope=scope)
    canvas = QgsMapCanvas()
    iface = SimpleNamespace(
        mapCanvas=lambda: canvas,
        activeLayer=lambda: None,
        setActiveLayer=lambda _: None,
    )
    manager = LayerManager(iface)
    manager.apply_load_scope(bundle["result"], scope)
    with patch.object(ui, "Sam3WorkerRunner", Worker):
        dialog = ui.ClassRefinementDialog(iface, manager)
        dialog._layer_loader._LOAD_INTERVAL_MS = 1
        try:
            config = {"checkpoint_sha256": "a" * 64, "version": "fixture"}
            dialog.set_run(bundle["result"], bundle["run_spec"], config, str(root))
            worker = dialog._worker
            assert worker is not None and worker.is_running and worker.requests == []
            assert "加载中" in dialog.sam_status
            wait(app, lambda: len(dialog._class_layers) == 14 and worker.is_ready)
            assert dialog._sam_available(), dialog.sam_status
            assert dialog._workspace["classes"]["12"]["confirmed"]
            assert dialog._workspace["classes"]["12"]["modified"]
            dialog.set_run(bundle["result"], bundle["run_spec"], config, str(root))
            wait(app, lambda: len(dialog._class_layers) == 14)
            assert (
                dialog._worker is worker
                and dialog._sam_config["checkpoint_sha256"] == "a" * 64
            )
            assert dialog._sam_config["buffer_px"] == 24
            assert all(
                path.read_bytes() == content
                for path, content in protected.items()
                if path.name != "workspace.json"
            )
            roles = {
                str(item.customProperty("labeling_tool/section", ""))
                for item in QgsProject.instance().mapLayers().values()
            }
            assert roles == (
                {"Classes"} if scope == "classes" else {"Classes", "Models", "Fusion"}
            )
        finally:
            dialog.cleanup()
            QgsProject.instance().removeAllMapLayers()
            canvas.close()


def classes(app, root):
    load_scope(app, root, "classes")


def full(app, root):
    load_scope(app, root, "all")


def layers(app, root):
    _, result = fixture(root / "one")
    _, other = fixture(root / "two")
    canvas = QgsMapCanvas()
    manager = LayerManager(SimpleNamespace(mapCanvas=lambda: canvas))
    manager.apply_load_scope(result, "all")
    first = set(QgsProject.instance().mapLayers())
    manager.apply_load_scope(result, "all")
    assert set(QgsProject.instance().mapLayers()) == first
    manager.apply_load_scope(other, "all")
    second = set(QgsProject.instance().mapLayers()) - first
    user = QgsVectorLayer("Polygon?crs=EPSG:3857", "user", "memory")
    QgsProject.instance().addMapLayer(user)
    manager.apply_load_scope(result, "classes")
    assert set(QgsProject.instance().mapLayers()) == second | {user.id()}
    assert Path(result["ready_streams"][0]["paths"]["semantic_polygons"]).exists()
    broken = copy.deepcopy(result)
    broken["ready_streams"][1]["paths"]["confidence_mosaic"] = str(root / "missing.tif")
    try:
        manager.apply_load_scope(broken, "all")
        raise AssertionError("missing result should fail")
    except RuntimeError:
        pass
    assert set(QgsProject.instance().mapLayers()) == second | {user.id()}
    QgsProject.instance().removeAllMapLayers()
    canvas.close()


def sam_failure(app, root):
    canvas = QgsMapCanvas()
    iface = SimpleNamespace(mapCanvas=lambda: canvas, activeLayer=lambda: None)
    dialog = ui.ClassRefinementDialog(iface, SimpleNamespace())
    with patch.object(ui, "Sam3WorkerRunner", Worker):
        dialog.preload_sam({}, str(root))
        worker = dialog._worker
        assert worker.is_running and not dialog._sam_available()
        wait(app, lambda: worker.is_ready)
        assert "已就绪" in dialog.sam_status and not dialog._sam_available(), (
            "Missing image cannot prevent model preload"
        )
        worker.is_ready = worker.is_running = False
        worker.event_received.emit(
            {"event": "worker_failed", "error": "bad checkpoint"}
        )
        assert "bad checkpoint" in dialog.sam_status
        dialog._retry_sam_preload()
        assert dialog._worker is not worker
        wait(app, lambda: dialog._worker.is_ready)
        dialog.cleanup()
        assert dialog._worker is None
    canvas.close()


def switch_guard(app, root):
    source = layer(root, "edited", positions=(0,))
    source.setCustomProperty("labeling_tool/workspace_path", str(root / "edited.gpkg"))
    QgsProject.instance().addMapLayer(source)
    canvas = QgsMapCanvas()
    dialog = ui.ClassRefinementDialog(
        SimpleNamespace(mapCanvas=lambda: canvas, activeLayer=lambda: None),
        SimpleNamespace(),
    )
    with patch.object(ui.QMessageBox, "warning"):
        source.startEditing()
        assert not dialog.begin_run_load()
        source.rollBack()
        assert dialog.begin_run_load()
        assert source.readOnly()
        assert not source.startEditing()
        dialog.finish_run_load()
        assert not source.readOnly()
    dialog.cleanup()
    QgsProject.instance().removeAllMapLayers()
    canvas.close()


if __name__ == "__main__":
    app = QgsApplication([], False)
    app.initQgis()
    try:
        with tempfile.TemporaryDirectory(prefix="loess-run-loading-") as directory:
            globals()["full" if sys.argv[2] == "all" else sys.argv[2]](
                app, Path(directory)
            )
            print(sys.argv[2] + ": passed", flush=True)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
    finally:
        app.exitQgis()
