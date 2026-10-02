# ruff: noqa: E402
"""Native layout acceptance for model-selection UI using a synthetic report."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(sys.argv[1])
OUTPUT = Path(sys.argv[2])
sys.path.insert(0, str(ROOT / "src"))
try:
    from qgis.core import QgsApplication
    from qgis.PyQt.QtCore import QCoreApplication, QEvent, Qt
    from qgis.PyQt.QtGui import QKeySequence
    from qgis.PyQt.QtTest import QTest
    from qgis.PyQt.QtWidgets import QDialogButtonBox, QMessageBox, QScrollArea
except ModuleNotFoundError:
    raise SystemExit(77)

from labeling_tool.main.inference_config_dialog import InferenceConfigDialog
from labeling_tool.qgis_support.qt6_api import APPLY, CANCEL, CHECKED


def _drain(app) -> None:
    app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _report(root: Path) -> dict:
    profile_path = (
        root
        / "黄土高原资料集"
        / ("very-long-profile-directory-" * 5)
        / "fusion-profile.json"
    )
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text("{}", encoding="utf-8")
    return {
        "status": "ready",
        "checks": [
            {"id": "semantic_model_alpha", "status": "ready"},
            {
                "id": "semantic_model_beta",
                "status": "error",
                "message": "设备检查未通过：合成诊断",
            },
            {"id": "fusion_profile_fusion-demo", "status": "ready"},
        ],
        "effective": {
            "schema_version": 2,
            "semantic_models": [
                {
                    "model_id": "alpha",
                    "display_name": "Alpha 长模型名称用于原生布局检查",
                    "version": "2026.09.25",
                    "artifact": "alpha.ts",
                    "artifact_path": str(root / ("models/" * 8) / "alpha.ts"),
                    "sha256": "a" * 64,
                },
                {
                    "model_id": "beta",
                    "display_name": "Beta 长模型名称用于不可用状态检查",
                    "version": "2026.09.25",
                    "artifact": "beta.ts",
                    "artifact_path": str(root / ("models/" * 8) / "beta.ts"),
                    "sha256": "b" * 64,
                },
            ],
            "fusion_profiles": [
                {
                    "profile_id": "fusion-demo",
                    "file_path": str(profile_path),
                    "enabled": True,
                    "available": True,
                    "status": "approved",
                    "strategy": "equal_probability_average",
                    "required_model_ids": ["alpha"],
                    "profile": {
                        "profile_id": "fusion-demo",
                        "status": "approved",
                        "strategy": "equal_probability_average",
                        "models": [{"model_id": "alpha"}],
                        "approval": {"passed": True},
                        "metrics": {
                            "baseline": {"miou": 0.7},
                            "fusion": {"miou": 0.71},
                        },
                    },
                }
            ],
            "runtime": {
                "effective_device": "cuda:0",
                "tile_batch_size": 2,
                "tile_io_workers": 2,
                "tile_page_size": 8,
            },
            "scaling": {
                "partition_tile_rows": 4,
                "partition_tile_cols": 4,
                "partition_halo_px": 192,
                "seam_band_px": 32,
                "score_cache_budget_gb": "auto",
                "min_free_disk_gb": 10,
                "max_cpu_partition_workers": 2,
                "max_cpu_partition_workers_with_package": 1,
            },
            "boundary_fitting": {
                "mode": "divider_cubic_bspline_adaptive_v2",
                "smoothing_factor": 2,
                "curve_sampling_spacing_px": 4,
                "max_chord_error_px": 1,
                "max_segment_arc_length_px": 64,
            },
        },
    }


def _assert_fixed_actions(dialog: InferenceConfigDialog) -> None:
    buttons = dialog.findChild(QDialogButtonBox)
    assert buttons is not None
    apply_button = buttons.button(APPLY)
    cancel_button = buttons.button(CANCEL)
    assert apply_button.isVisible() and cancel_button.isVisible()
    for button in (apply_button, cancel_button):
        bottom_right = button.mapTo(dialog, button.rect().bottomRight())
        assert dialog.rect().contains(bottom_right)


def _assert_keyboard_details(app, dialog: InferenceConfigDialog) -> None:
    """Reach and copy full diagnostics using real keyboard navigation."""

    labels = (
        dialog.model_details_label,
        dialog.profile_path_label,
        dialog.scaling_label,
        dialog.boundary_label,
    )
    remaining = set(labels)
    dialog.details_toggle.setFocus()
    _drain(app)
    for _ in range(40):
        focused = app.focusWidget()
        assert focused is not None
        QTest.keyClick(focused, Qt.Key.Key_Tab)
        _drain(app)
        focused = app.focusWidget()
        if focused in remaining:
            QTest.keySequence(focused, QKeySequence(QKeySequence.StandardKey.SelectAll))
            QTest.keySequence(focused, QKeySequence(QKeySequence.StandardKey.Copy))
            _drain(app)
            assert app.clipboard().text() == focused.text()
            remaining.remove(focused)
        if not remaining:
            break
    assert not remaining, "Technical details cannot all be reached and copied by Tab"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    app = QgsApplication([], False)
    app.initQgis()
    dialog = InferenceConfigDialog()
    try:
        with tempfile.TemporaryDirectory(prefix="loess-model-config-") as directory:
            dialog.set_environment(_report(Path(directory)), ["alpha"], "fusion-demo")
            assert dialog.model_table.columnCount() == 5
            assert [
                dialog.model_table.horizontalHeaderItem(index).text()
                for index in range(dialog.model_table.columnCount())
            ] == ["运行", "模型", "可用性", "结果", "实际设备"]
            assert dialog.model_table.item(0, 2).text() == "可用"
            assert dialog.model_table.item(0, 2).toolTip() == "环境检查通过"
            assert "环境检查: ready；环境检查通过" in dialog.model_details_label.text()
            assert dialog.model_table.item(1, 2).text() == "不可用"
            assert "设备检查未通过" in dialog.model_table.item(1, 2).toolTip()
            assert dialog.model_table.item(0, 4).text() == "cuda:0"
            assert dialog.model_table.item(0, 0).checkState() == CHECKED
            assert "Alpha 长模型名称" in dialog.selection_summary_label.text()
            assert "不改变模型分类" in dialog.boundary_effect_label.text()
            assert "路径" in dialog.details_toggle.text()
            assert str(Path(directory)) in dialog.model_details_label.text()
            assert str(Path(directory)) in dialog.profile_path_label.text()
            scroll = dialog.findChild(QScrollArea, "inferenceConfigScrollArea")
            assert scroll is not None

            for width, height in ((900, 560), (1120, 680)):
                dialog.resize(width, height)
                dialog.show()
                _drain(app)
                assert dialog.size().width() == width
                assert dialog.size().height() == height
                assert scroll.viewport().height() > 0
                assert not dialog.technical_details_group.isVisible()
                _assert_fixed_actions(dialog)
                screenshot = OUTPUT / f"model-config-{width}x{height}.png"
                assert dialog.grab().save(str(screenshot))
                assert screenshot.is_file() and screenshot.stat().st_size > 0

            dialog.resize(900, 560)
            dialog.show()
            _drain(app)
            dialog.details_toggle.click()
            _drain(app)
            assert dialog.technical_details_group.isVisible()
            assert scroll.verticalScrollBar().maximum() > 0
            scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
            _drain(app)
            _assert_fixed_actions(dialog)
            expanded = OUTPUT / "model-config-900x560-details.png"
            assert dialog.grab().save(str(expanded))
            assert expanded.is_file() and expanded.stat().st_size > 0

            dialog.activateWindow()
            _drain(app)
            _assert_keyboard_details(app, dialog)
            dialog.resize(640, 360)
            _drain(app)
            _assert_keyboard_details(app, dialog)
            _assert_fixed_actions(dialog)

            # A folded diagnostics section must not hide an unusable choice.
            dialog.details_toggle.setChecked(False)
            data = _report(Path(directory) / "unavailable")
            data["effective"]["fusion_profiles"][0]["available"] = False
            data["checks"][-1].update(status="error", message="合成融合资产缺失")
            dialog.set_environment(data, ["alpha"], "fusion-demo")
            assert not dialog.technical_details_group.isVisible()
            assert "不可运行：合成融合资产缺失" in dialog.profile_summary_label.text()
            assert "不可运行" in dialog.profile_combo.currentText()

            # Applying a plan requires the same positive model check as launch.
            applied = []
            dialog.configuration_applied.connect(lambda *values: applied.append(values))
            for status in ("warning", None):
                data["checks"][0]["status"] = status
                dialog.set_environment(data, ["alpha"], None)
                with patch.object(QMessageBox, "warning") as warning:
                    dialog.findChild(QDialogButtonBox).button(APPLY).click()
                assert warning.call_count == 1
                assert not applied
    finally:
        dialog.close()
        dialog.deleteLater()
        _drain(app)
        app.exitQgis()


if __name__ == "__main__":
    main()
    print("model_config_ui: passed")
