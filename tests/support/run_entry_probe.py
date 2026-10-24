# ruff: noqa: E402
"""Exercise the real unified main-panel entry and staged layer cancellation."""

import hashlib
import json
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
    spec, _ = fixture(
        root / "run",
        initialized=scenario
        not in {
            "entry_uninitialized",
            "entry_uninitialized_models",
            "entry_models_no_fusion",
        },
    )
    if scenario == "entry_models_no_fusion":
        manifest_path = root / "run/run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["streams"] = [
            stream for stream in manifest["streams"] if stream["kind"] == "model"
        ]
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False))
    original_classes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root / "run/classes").glob("*.gpkg")
    }
    original_reviews = (
        json.loads((root / "run/classes/workspace.json").read_text())["classes"]
        if original_classes
        else None
    )
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

        choose_directory = stack.enter_context(
            patch.object(
                main_dock.QFileDialog,
                "getExistingDirectory",
                return_value=spec["run_dir"],
            )
        )
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
                dock.run_load_btn.text() == "加载到 QGIS 并打开整理"
                and dock.run_load_btn.isEnabled()
            )
            assert not hasattr(dock, "load_manual_run_btn") and not hasattr(
                dock, "open_refinement_btn"
            )
            assert "尚未加载" in dock.loaded_run_label.text()
            if scenario == "entry_no_selection":
                with patch.object(main_dock.QMessageBox, "information") as info:
                    dock.run_load_btn.click()
                assert info.call_count == 1
                assert not choose_directory.called and dock._manual_load_task is None
                assert dock.refinement_dialog._worker is None
                return
            dock.results_workspace_panel.choose_btn.click()
            wait(app, lambda: not dock.results_workspace_panel.busy)
            assert not QgsProject.instance().mapLayers(), (
                "Selection must not mount layers"
            )
            assert "选中 Run" in dock.results_workspace_panel.run_label.text()
            assert choose_directory.call_count == 1
            # A second directory request is a regression, including on reload.
            choose_directory.side_effect = AssertionError(
                "Run must only be selected once"
            )
            selected = {
                "entry_classes": {"classes"},
                "entry_fusion": {"fusion"},
                "entry_models": {"models"},
                "entry_fusion_models": {"fusion", "models"},
                "entry_classes_fusion": {"classes", "fusion"},
                "entry_classes_models": {"classes", "models"},
                "entry_uninitialized_models": {"models"},
                "entry_models_no_fusion": {"models"},
            }.get(scenario, {"classes", "fusion", "models"})
            if scenario == "entry_zero_selection":
                selected = set()
            for check, kind in dock.run_scope_checks:
                check.setChecked(kind in selected)
            if scenario == "entry_zero_selection":
                assert not dock.run_load_btn.isEnabled()
                with patch.object(main_dock.QMessageBox, "information") as info:
                    dock._on_open_refinement()
                assert info.call_count == 1 and dock._manual_load_task is None
                assert choose_directory.call_count == 1
                return
            dock.run_load_btn.click()
            assert dock.refinement_dialog._worker.is_running, (
                "Both scopes must proactively start SAM3"
            )
            assert choose_directory.call_count == 1
            assert dock.refinement_dialog._run_loading
            assert not dock.refinement_dialog.isEnabled()
            if scenario == "entry_uninitialized":
                wait(app, lambda: warnings.call_count > 0)
                assert not (root / "run/classes").exists(), (
                    "Loading must not initialize classes"
                )
                assert dock.results_workspace_panel.initialize_btn.isEnabled()
                assert "尚未加载" in dock.loaded_run_label.text()
                return
            if "classes" not in selected:
                wait(
                    app,
                    lambda: dock._manual_load_task is None
                    and dock._pending_run_bundle is None
                    and "结果图层加载完成" in dock.result_summary_label.text(),
                )
                wait(app, lambda: dock.refinement_dialog._worker.is_ready)
                assert not dock.refinement_dialog.isVisible()
                assert dock.refinement_dialog._class_layers == {}
                assert dock.refinement_dialog._workspace is None
                assert dock.run_load_btn.text() == "加载到 QGIS"
                assert "已就绪" in dock.run_sam_status_label.text()
                sections = {
                    layer.customProperty("labeling_tool/section", "")
                    for layer in QgsProject.instance().mapLayers().values()
                }
                expected = {
                    "Fusion" if kind == "fusion" else "Models" for kind in selected
                }
                assert sections == expected, (sections, expected)
                if scenario in {"entry_uninitialized_models", "entry_models_no_fusion"}:
                    assert not (root / "run/classes").exists()
                else:
                    assert {
                        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (root / "run/classes").glob("*.gpkg")
                    } == original_classes
                    assert not (
                        root / "run/classes/accepted_write_run_manifest.json"
                    ).exists()
                assert warnings.call_count == 0, warnings.call_args_list
                return
            if scenario == "entry_cancel":
                dock.results_workspace_panel.choose_btn.click()
                assert choose_directory.call_count == 1
                assert warnings.call_count == 1, (
                    "Changing Run must wait for loading or cancellation"
                )
                warnings.reset_mock()
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
            scope_text = "、".join(
                kind for kind in ("fusion", "models", "classes") if kind in selected
            )
            assert scope_text in summary, summary
            assert dock.run_load_btn.text() == "加载到 QGIS 并打开整理"
            assert "加载完成" in dock.loaded_run_label.text()
            assert "SAM3 已就绪" in dock.run_sam_status_label.text()
            dialog.workspace_load_status_changed.emit(
                {**dialog._result, "run_dir": str(root / "another-run")},
                "旧 Run 加载失败",
            )
            assert dock.result_summary_label.text() == summary
            assert dialog._sam_available() and dialog._worker.requests == []
            assert not dialog._run_loading and dialog.isEnabled()
            assert dialog.isVisible(), "The real Run entry must open its editor"
            assert dialog._workspace["classes"]["12"]["confirmed"]
            for code, record in original_reviews.items():
                assert {
                    k: v
                    for k, v in dialog._workspace["classes"][code].items()
                    if k != "path"
                } == {k: v for k, v in record.items() if k != "path"}
            assert {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (root / "run/classes").glob("*.gpkg")
            } == original_classes
            reviewed_classes = {
                code: dict(record)
                for code, record in dialog._workspace["classes"].items()
            }
            sections = {
                layer.customProperty("labeling_tool/section", "")
                for layer in QgsProject.instance().mapLayers().values()
            }
            expected = {"Classes"}
            if "fusion" in selected:
                expected.add("Fusion")
            if "models" in selected:
                expected.add("Models")
            assert sections == expected, (sections, expected)
            assert warnings.call_count == 0, warnings.call_args_list
            if scenario == "entry_scope_switch":
                workspace_file = root / "run/classes/workspace.json"
                workspace_before = workspace_file.read_bytes()
                for check, kind in dock.run_scope_checks:
                    check.setChecked(kind == "models")
                dock.run_load_btn.click()
                wait(
                    app,
                    lambda: dock._manual_load_task is None
                    and dock._pending_run_bundle is None
                    and "结果图层加载完成" in dock.result_summary_label.text(),
                )
                assert {
                    layer.customProperty("labeling_tool/section", "")
                    for layer in QgsProject.instance().mapLayers().values()
                } == {"Models"}
                assert dialog._class_layers == {} and dialog._workspace is None
                assert not dialog.isVisible()
                assert workspace_file.read_bytes() == workspace_before
                assert dialog._worker.is_ready
                reviewed_classes = None
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


def selection(app, root, scenario):
    old, _ = fixture(root / "old_run")
    selected, _ = fixture(root / "selected_run")
    old["run_dir"] = str(Path(old["run_dir"]).resolve())
    selected["run_dir"] = str(Path(selected["run_dir"]).resolve())
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
        stack.enter_context(
            patch.object(
                main_dock.run_index, "load_startup_candidates", return_value={}
            )
        )
        for name in ("_save_settings", "_load_settings_and_defaults"):
            stack.enter_context(
                patch.object(main_dock.LabelingDockWidget, name, lambda *_: None)
            )
        dock = main_dock.LabelingDockWidget(iface=iface)
        dock.refinement_dialog._layer_loader._LOAD_INTERVAL_MS = 1
        try:
            if scenario == "startup_selection":
                dock.results_workspace_panel.set_run_directory(selected["run_dir"])
                wait(app, lambda: not dock.results_workspace_panel.busy)
                summary = dock.result_summary_label.text()
                assert "selected_run" in summary and "加载完成" not in summary
                with (
                    patch.object(
                        main_dock.run_index,
                        "load_startup_candidates",
                        return_value={"latest_ready": {"spec": old}},
                    ),
                    patch.object(
                        main_dock.run_index,
                        "lightweight_ready_result",
                        return_value={
                            "run_id": old["run_id"],
                            "run_dir": old["run_dir"],
                        },
                    ),
                ):
                    dock._restore_latest_ready_run()
                assert Path(dock.results_workspace_panel._candidate_dir) == Path(
                    selected["run_dir"]
                )
                assert dock.result_summary_label.text() == summary
                return

            dock.results_workspace_panel.set_run_directory(old["run_dir"])
            wait(app, lambda: not dock.results_workspace_panel.busy)
            dock._on_open_refinement()
            wait(app, lambda: len(dock.refinement_dialog._class_layers) == 14)
            original_layers = set(QgsProject.instance().mapLayers())
            old_result = dict(dock.refinement_dialog._result)
            old_loaded_status = dock.loaded_run_label.text()
            dock.results_workspace_panel.set_run_directory(selected["run_dir"] + "/")
            wait(app, lambda: not dock.results_workspace_panel.busy)
            summary = dock.result_summary_label.text()
            assert "selected_run" in summary and "加载完成" not in summary
            assert dock.loaded_run_label.text() == old_loaded_status
            assert dock.loaded_run_label.toolTip() == old["run_dir"]
            dock.refinement_dialog.workspace_load_status_changed.emit(
                old_result, "加载完成，14 类工作层已就绪"
            )
            assert dock.result_summary_label.text() == summary

            assert dock.results_workspace_panel.run_directory == selected["run_dir"]
            if scenario == "selection_cancel":
                for check, _kind in dock.run_scope_checks:
                    check.setChecked(True)
                dock._on_open_refinement()
                wait(app, lambda: bool(dock._new_result_layer_ids))
                dock._on_open_refinement()
                assert dock._last_run_result["run_dir"] == old["run_dir"]
                assert dock.loaded_run_label.text() == old_loaded_status
                dock.results_workspace_panel.set_candidate(old["run_dir"])
                assert "old_run" in dock.result_summary_label.text()
            else:
                dock._on_open_refinement()
                wait(
                    app,
                    lambda: dock.refinement_dialog._run_spec.get("run_dir")
                    == selected["run_dir"]
                    and len(dock.refinement_dialog._class_layers) == 14,
                )
                assert dock._last_run_result["run_dir"] == selected["run_dir"]
                assert "加载完成" in dock.result_summary_label.text()
                assert dock.loaded_run_label.toolTip() == selected["run_dir"]
                assert dock.results_workspace_panel._probe.get("workspace")
                assert not dock.results_workspace_panel.initialize_btn.isEnabled()
            assert original_layers.issubset(QgsProject.instance().mapLayers())
        finally:
            dock.refinement_dialog.close()
            dock.cleanup()
            app.processEvents()
            QgsProject.instance().removeAllMapLayers()
            canvas.close()


def main():
    with tempfile.TemporaryDirectory(prefix="loess-run-entry-profile-") as profile:
        app = QgsApplication([], False, profile)
        app.initQgis()
        try:
            with tempfile.TemporaryDirectory(prefix="loess-run-entry-") as directory:
                if sys.argv[2] in {
                    "selection_sync",
                    "selection_cancel",
                    "startup_selection",
                }:
                    selection(app, Path(directory), sys.argv[2])
                elif sys.argv[2] == "image_binding":
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
