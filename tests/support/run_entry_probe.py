# ruff: noqa: E402
"""Exercise the real unified main-panel entry and staged layer cancellation."""

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
    from qgis.core import QgsApplication, QgsProject, QgsVectorLayer
    from qgis.gui import QgsMapCanvas
    from qgis.PyQt.QtCore import pyqtSignal
    from qgis.PyQt.QtWidgets import QDialog
except ModuleNotFoundError:
    raise SystemExit(77)

from run_loading_probe import Worker, fixture, wait

from labeling_tool.main import main_dock
from labeling_tool.qgis_support.layer_manager import LayerManager
from labeling_tool.refinement import class_refinement_dialog as ui
from labeling_tool.refinement.manual_run_loader import load_manual_run


class Settings:
    def value(self, key, default=None, **_kwargs):
        return default

    def setValue(self, *_args):
        pass


class Monitor(QDialog):
    stop_requested = pyqtSignal()
    request_main_run_handling = pyqtSignal(object)

    def detach(self):
        pass

    def shutdown(self):
        self.close()


def entry(app, root, scenario):
    spec, _ = fixture(root / "run")
    canvas = QgsMapCanvas()
    iface = SimpleNamespace(
        mapCanvas=lambda: canvas,
        activeLayer=lambda: None,
        setActiveLayer=lambda _: None,
    )
    with ExitStack() as stack:
        stack.enter_context(patch.object(ui, "Sam3WorkerRunner", Worker))
        stack.enter_context(patch.object(main_dock, "InferenceMonitorDialog", Monitor))
        stack.enter_context(patch.object(main_dock, "QgsSettings", Settings))
        for name in (
            "_save_settings",
            "_load_settings_and_defaults",
            "_restore_latest_ready_run",
        ):
            stack.enter_context(
                patch.object(main_dock.LabelingDockWidget, name, lambda *_: None)
            )
        dock = main_dock.LabelingDockWidget(iface=iface)
        dock.refinement_dialog._layer_loader._LOAD_INTERVAL_MS = 1
        warnings = stack.enter_context(patch.object(main_dock.QMessageBox, "warning"))

        def choose(chooser):
            assert dock.refinement_dialog._worker.is_running, (
                "SAM must start before any prediction"
            )
            chooser.directory_edit.setText(spec["run_dir"])
            chooser.all_radio.setChecked(scenario != "entry_classes")
            return 1

        stack.enter_context(patch.object(main_dock.RunLoadDialog, "exec", choose))
        if scenario == "entry_class_failure":
            stack.enter_context(
                patch.object(
                    dock.layer_manager,
                    "load_workspace_class",
                    side_effect=RuntimeError("test class mount failure"),
                )
            )
        reviewed_classes = None
        try:
            assert (
                dock.run_load_btn.text() == "Run 加载与整理"
                and dock.run_load_btn.isEnabled()
            )
            assert not hasattr(dock, "load_manual_run_btn") and not hasattr(
                dock, "open_refinement_btn"
            )
            dock._on_open_refinement()
            assert dock.refinement_dialog._run_loading
            assert not dock.refinement_dialog.isEnabled()
            if scenario == "entry_cancel":
                wait(app, lambda: bool(dock._new_result_layer_ids))
                user = QgsVectorLayer(
                    "Polygon?crs=EPSG:3857", "user_added_during_load", "memory"
                )
                QgsProject.instance().addMapLayer(user)
                dock._on_open_refinement()
                assert set(QgsProject.instance().mapLayers()) == {user.id()}
                assert (
                    dock._pending_run_bundle is None
                    and not dock.refinement_dialog._run_loading
                )
                assert dock.refinement_dialog._worker is None
                return
            if scenario == "entry_class_failure":
                wait(app, lambda: warnings.call_count > 0)
                assert "挂载失败" in dock.result_summary_label.text()
                assert "加载完成" not in dock.result_summary_label.text()
                return
            wait(app, lambda: 0 < len(dock.refinement_dialog._class_layers) < 14)
            assert "正在加载" in dock.result_summary_label.text()
            if scenario == "entry_class_pause":
                dock.refinement_dialog._cancel_background_load()
                assert "已暂停" in dock.result_summary_label.text()
                assert "加载完成" not in dock.result_summary_label.text()
                return
            wait(app, lambda: len(dock.refinement_dialog._class_layers) == 14)
            dialog = dock.refinement_dialog
            summary = dock.result_summary_label.text()
            assert "加载完成，14 类工作层已就绪" in summary, summary
            assert ("fusion、models、classes" in summary) == (scenario == "entry_all")
            dialog.workspace_load_status_changed.emit(
                {**dialog._result, "run_dir": str(root / "another-run")},
                "旧 Run 加载失败",
            )
            assert dock.result_summary_label.text() == summary
            assert dialog._sam_available() and dialog._worker.requests == []
            assert not dialog._run_loading and dialog.isEnabled()
            assert dialog.isVisible(), "The real Run entry must open its editor"
            assert dialog._workspace["classes"]["12"]["confirmed"]
            reviewed_classes = {
                code: dict(record)
                for code, record in dialog._workspace["classes"].items()
            }
            sections = {
                layer.customProperty("labeling_tool/section", "")
                for layer in QgsProject.instance().mapLayers().values()
            }
            assert sections == (
                {"Classes"}
                if scenario == "entry_classes"
                else {"Classes", "Models", "Fusion"}
            )
            assert warnings.call_count == 0, warnings.call_args_list
        finally:
            try:
                # cleanup() releases work but intentionally keeps this reusable
                # editor alive. Close its window before QApplication teardown,
                # while the platform input context and Qt thread storage exist.
                dialog = dock.refinement_dialog
                dialog.close()
                assert not dialog.isVisible(), "Run editor remained open at exit"
                if reviewed_classes is not None:
                    assert dialog._workspace["classes"] == reviewed_classes
            finally:
                dock.cleanup()
                app.processEvents()
                QgsProject.instance().removeAllMapLayers()
                canvas.close()


def image_binding(app, root):
    spec, _ = fixture(root / "run")
    bundle = load_manual_run(spec["run_dir"])
    bundle["run_spec"]["raster"] = {**spec["raster"], "path": str(root / "missing.tif")}
    canvas = QgsMapCanvas()
    iface = SimpleNamespace(
        mapCanvas=lambda: canvas,
        activeLayer=lambda: None,
        setActiveLayer=lambda _: None,
    )
    manager = LayerManager(iface)
    with patch.object(ui, "Sam3WorkerRunner", Worker):
        dialog = ui.ClassRefinementDialog(iface, manager)
        dialog._layer_loader._LOAD_INTERVAL_MS = 1
        try:
            dialog.set_run(bundle["result"], bundle["run_spec"], {}, str(root))
            wait(
                app, lambda: len(dialog._class_layers) == 14 and dialog._worker.is_ready
            )
            assert (
                not dialog._sam_available()
                and "定位" in dialog.raster_status_label.text()
            )
            with patch.object(
                ui.QFileDialog,
                "getOpenFileName",
                return_value=(spec["raster"]["path"], ""),
            ):
                dialog._locate_sam_raster()
            assert dialog._sam_available()
            assert (
                Path(dialog._run_spec["raster"]["path"]).resolve()
                == Path(spec["raster"]["path"]).resolve()
            )
            assert bundle["run_spec"]["raster"]["path"] == str(root / "missing.tif")
            dialog.set_run(bundle["result"], bundle["run_spec"], {}, str(root))
            wait(app, lambda: len(dialog._class_layers) == 14)
            assert dialog._sam_available(), (
                "Switching layer scope keeps the checked image binding"
            )
        finally:
            dialog.cleanup()
            QgsProject.instance().removeAllMapLayers()
            canvas.close()


def main():
    with tempfile.TemporaryDirectory(prefix="loess-run-entry-profile-") as profile:
        app = QgsApplication([], False, profile)
        app.initQgis()
        try:
            with tempfile.TemporaryDirectory(prefix="loess-run-entry-") as directory:
                if sys.argv[2] == "image_binding":
                    image_binding(app, Path(directory))
                else:
                    entry(app, Path(directory), sys.argv[2])
        except Exception:
            traceback.print_exc()
            raise SystemExit(1)
        finally:
            app.exitQgis()
        print(sys.argv[2] + ": qgis_shutdown_complete", flush=True)
        print(sys.argv[2] + ": passed", flush=True)


if __name__ == "__main__":
    main()
