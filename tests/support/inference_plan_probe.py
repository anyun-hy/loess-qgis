# ruff: noqa: E402
"""Plan selection and dialog recovery using synthetic reports, without model I/O."""

from __future__ import annotations

import sys
import tempfile
import traceback
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
try:
    from qgis.core import QgsApplication
    from qgis.PyQt.QtCore import QCoreApplication, QEvent, pyqtSignal
    from qgis.PyQt.QtWidgets import QDialog, QDialogButtonBox, QMessageBox
except ModuleNotFoundError:
    raise SystemExit(77)

from labeling_tool.main.inference_config_dialog import InferenceConfigDialog
from labeling_tool.qgis_support.qt6_api import APPLY, CANCEL, CHECKED, UNCHECKED


def drain(app):
    app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def report(root):
    profile = root / "profile.json"
    profile.write_text("{}")
    return {
        "status": "ready",
        "checks": [
            {"id": "semantic_model_alpha", "status": "ready"},
            {"id": "semantic_model_beta", "status": "ready"},
        ],
        "effective": {
            "schema_version": 2,
            "semantic_models": [
                {"model_id": "alpha", "display_name": "Alpha"},
                {"model_id": "beta", "display_name": "Beta"},
            ],
            "fusion_profiles": [
                {
                    "profile_id": "fusion-demo",
                    "file_path": str(profile),
                    "enabled": True,
                    "available": True,
                    "status": "approved",
                    "required_model_ids": ["alpha"],
                }
            ],
            "runtime": {"effective_device": "cpu"},
            "boundary_fitting": {"mode": "divider_cubic_bspline_adaptive_v2"},
        },
    }


def invalid_report(app, root):
    dialog = InferenceConfigDialog()
    applied = []
    dialog.configuration_applied.connect(lambda *values: applied.append(values))
    buttons = dialog.findChild(QDialogButtonBox)
    try:
        data = report(root)
        dialog.set_environment(data, ["beta"], "fusion-demo", False)
        assert dialog.model_table.rowCount() == 2
        assert dialog.open_profile_btn.isEnabled()
        dialog.set_environment({"status": "error", "checks": []})
        assert dialog.model_table.rowCount() == 0
        assert dialog.profile_combo.count() == 0
        assert not dialog.open_profile_btn.isEnabled()
        assert str(root) not in dialog.profile_path_label.text()
        assert "不可用" in dialog.status_label.text()
        buttons.button(APPLY).click()
        assert not applied
        dialog.set_environment(data, ["beta"], "fusion-demo", False)
        assert dialog.profile_combo.currentData() == "fusion-demo"
        assert dialog.model_table.item(0, 0).checkState() == CHECKED
        buttons.button(APPLY).click()
        assert applied == [(["alpha", "beta"], "fusion-demo", False)]
    finally:
        dialog.close()
        dialog.deleteLater()
        drain(app)


def selection_draft(app, root):
    from labeling_tool.main.inference_plan_panel import (
        InferencePlanPanel,
        InferenceSelection,
    )

    data = report(root)
    panel = InferencePlanPanel(lambda: data)
    try:
        panel.restore_selection(
            InferenceSelection(("beta", "missing"), "removed", False, True)
        )
        assert panel.selection.confirmed is False
        panel.set_environment(data)
        assert panel.selection.model_ids == ("beta",)
        assert panel.selection.fusion_profile_id is None
        assert "已选模型（1）: Beta" in panel.summary_label.text()
        assert "关闭，保留原始像元边界" in panel.summary_label.text()
        panel.configure_button.click()
        dialog = panel.configuration_dialog
        buttons = dialog.findChild(QDialogButtonBox)
        assert dialog.isVisible()
        dialog.model_table.item(0, 0).setCheckState(CHECKED)
        dialog.boundary_smoothing_check.setChecked(True)
        buttons.button(CANCEL).click()
        assert panel.selection.model_ids == ("beta",)
        assert panel.selection.boundary_smoothing_enabled is False
        assert panel.selection.confirmed is False
        panel.configure_button.click()
        assert dialog.model_table.item(0, 0).checkState() == UNCHECKED
        assert not dialog.boundary_smoothing_check.isChecked()
        dialog.profile_combo.setCurrentIndex(
            dialog.profile_combo.findData("fusion-demo")
        )
        dialog.boundary_smoothing_check.setChecked(True)
        buttons.button(APPLY).click()
        snapshot = panel.selection
        assert snapshot.model_ids == ("alpha", "beta")
        assert snapshot.fusion_profile_id == "fusion-demo"
        assert snapshot.boundary_smoothing_enabled and snapshot.confirmed
        assert "方案状态: 已确认" in panel.summary_label.text()
        try:
            snapshot.confirmed = False
        except FrozenInstanceError:
            pass
        else:
            raise AssertionError("selection must be immutable")
        panel.invalidate()
        assert not panel.selection.confirmed and not panel.configure_button.isEnabled()
        assert snapshot.confirmed  # Earlier consumers keep their original snapshot.
        assert "请先完成推理环境检查" in panel.summary_label.text()
        panel.set_environment(data)
        assert not panel.selection.confirmed
        assert panel.selection.model_ids == ("alpha", "beta")
        panel.configure_button.click()
        panel.cleanup()
        panel.cleanup()
        assert not dialog.isVisible()
    finally:
        panel.cleanup()
        panel.deleteLater()
        drain(app)


def invalidation(app, root):
    from labeling_tool.main.inference_plan_panel import (
        InferencePlanPanel,
        InferenceSelection,
    )

    data = report(root)
    panel = InferencePlanPanel(lambda: data)
    changes = []
    panel.selection_changed.connect(lambda: changes.append(panel.selection))
    try:
        panel.restore_selection(InferenceSelection(("beta",)))
        panel.set_environment(data)
        assert not changes
        panel.configure_button.click()
        dialog = panel.configuration_dialog
        buttons = dialog.findChild(QDialogButtonBox)
        buttons.button(APPLY).click()
        assert panel.selection.confirmed
        assert len(changes) == 1
        data["checks"][1]["status"] = "error"
        panel.set_environment(data)
        assert panel.selection.model_ids == ("alpha",)
        assert not panel.selection.confirmed
        assert len(changes) == 2
        panel.configure_button.click()
        dialog.model_table.item(1, 0).setCheckState(CHECKED)
        with patch.object(QMessageBox, "warning") as warning:
            buttons.button(APPLY).click()
            assert warning.call_count == 1 and not panel.selection.confirmed
        data = {"status": "error", "checks": []}
        panel.set_environment(data)
        assert not panel.configure_button.isEnabled() and not panel.selection.confirmed
        data = report(root)
        panel.set_environment(data)
        panel.configure_button.click()
        buttons.button(APPLY).click()
        assert panel.selection.confirmed
        panel.set_environment(data)
        assert not panel.selection.confirmed  # A new report always needs confirmation.
        count = len(changes)
        panel.set_environment(data)
        assert len(changes) == count
        data["effective"]["semantic_models"][0]["enabled"] = False
        panel.set_environment(data)
        assert panel.selection.model_ids == ("beta",)
    finally:
        panel.cleanup()
        panel.deleteLater()
        drain(app)


def launch_plan_validation(app, root):
    from labeling_tool.main.inference_plan_panel import (
        InferenceSelection,
        resolve_launch_plan,
    )

    data = report(root)
    resolved = resolve_launch_plan(
        data,
        InferenceSelection(("beta",), "fusion-demo", False, True),
    )
    assert resolved.model_ids == ("beta", "alpha")
    assert resolved.fusion_profile_id == "fusion-demo"
    assert resolved.boundary_smoothing_enabled is False
    try:
        resolved.model_ids = ()
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("launch plan must be immutable")

    unavailable = deepcopy(data)
    unavailable["checks"][1]["status"] = "error"
    try:
        resolve_launch_plan(unavailable, InferenceSelection(("beta",)))
    except ValueError as exc:
        assert str(exc) == "模型未通过设备实测: beta"
    else:
        raise AssertionError("unready model must reject launch")

    try:
        resolve_launch_plan(data, InferenceSelection(("missing",)))
    except ValueError as exc:
        assert str(exc) == "unknown model_id: missing"
    else:
        raise AssertionError("unknown model must reject launch")


def settings_roundtrip(app, root):
    from labeling_tool.main import main_dock

    saved = {
        "plugins/labeling_tool/inference_path": str(root),
        "plugins/labeling_tool/output_path": str(root / "accepted.gpkg"),
        "plugins/labeling_tool/output_workspace": str(root / "workspace"),
        "plugins/labeling_tool/selected_models": ["beta"],
        "plugins/labeling_tool/fusion_profile": "fusion-demo",
        "plugins/labeling_tool/boundary_smoothing_enabled": False,
        "plugins/labeling_tool/tile_overlap_probability_blend": 128,
        "plugins/labeling_tool/skip_accepted": False,
    }
    original = deepcopy(saved)

    class Settings:
        def value(self, key, default, type=None):
            return deepcopy(saved.get(key, default))

        def setValue(self, key, value):
            saved[key] = deepcopy(value)

    class Monitor(QDialog):
        stop_requested = pyqtSignal()
        request_main_run_handling = pyqtSignal(object)

        def detach(self):
            pass

        def shutdown(self):
            self.close()

    with ExitStack() as patches:
        patches.enter_context(patch.object(main_dock, "QgsSettings", Settings))
        patches.enter_context(
            patch.object(main_dock, "InferenceMonitorDialog", Monitor)
        )
        patches.enter_context(
            patch.object(
                main_dock.LabelingDockWidget,
                "_restore_latest_ready_run",
                lambda _: None,
            )
        )
        dock = main_dock.LabelingDockWidget()
        try:
            selection = dock.plan_panel.selection
            assert selection.model_ids == ("beta",), selection
            assert selection.fusion_profile_id == "fusion-demo"
            assert selection.boundary_smoothing_enabled is False
            assert not selection.confirmed
            assert "请先完成推理环境检查" in dock.plan_panel.summary_label.text()
            assert not dock.start_btn.isEnabled()
            assert dock.overlap_spin.value() == 128
            assert not dock.skip_accepted_check.isChecked()
            dock._save_settings()
            assert {key: saved[key] for key in original} == original
            dock.config_manager.last_report = report(root)
            dock.config_manager.report_ready.emit(dock.config_manager.last_report)
            dialog = dock.plan_panel.configuration_dialog
            dialog.boundary_smoothing_check.setChecked(True)
            dialog.findChild(QDialogButtonBox).button(APPLY).click()
            assert saved["plugins/labeling_tool/boundary_smoothing_enabled"] is True
            assert saved["plugins/labeling_tool/selected_models"] == ["alpha", "beta"]
            assert not any("confirmed" in key for key in saved)
        finally:
            dock.cleanup()
            dock.deleteLater()
            drain(app)


if __name__ == "__main__":
    app = QgsApplication([], False)
    app.initQgis()
    exceptions = []

    def capture_exception(*error):
        exceptions.append(error)
        traceback.print_exception(*error)

    sys.excepthook = capture_exception
    scenario = sys.argv[2]
    try:
        with tempfile.TemporaryDirectory(prefix="loess-plan-") as temporary:
            globals()[scenario](app, Path(temporary))
        assert not exceptions, exceptions
        print(scenario + ": passed")
    finally:
        drain(app)
        app.exitQgis()
