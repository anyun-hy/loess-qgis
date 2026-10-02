"""Native QGIS probe for the display-only admission summary."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))

try:
    from qgis.core import QgsApplication
    from qgis.PyQt.QtCore import Qt
    from qgis.PyQt.QtTest import QTest
    from qgis.PyQt.QtWidgets import QApplication, QLabel
except ModuleNotFoundError:
    raise SystemExit(77)


def _label(panel, name: str) -> QLabel:
    value = panel.findChild(QLabel, name)
    assert value is not None, name
    return value


def main() -> None:
    from labeling_tool.refinement.admission_presentation import AdmissionSummarySnapshot
    from labeling_tool.refinement.admission_summary_panel import AdmissionSummaryPanel

    app = QgsApplication([], False)
    app.initQgis()
    panel = AdmissionSummaryPanel()
    panel.resize(900, 560)
    panel.show()
    app.processEvents()
    requested = []
    panel.write_requested.connect(lambda: requested.append(True))
    long_path = (
        "/研究项目/黄土高原地物标注/本次人工审核成果/长期标签库/"
        "超长中文目录用于检查复制与横向滚动/"
        "accepted_labels.gpkg"
    )
    panel.render(
        AdmissionSummarySnapshot(
            target_path=long_path,
            final_feature_count=42,
            confirmed_class_count=14,
            unsaved_edit_count=0,
            topology_executed=True,
            topology_issue_count=0,
            write_enabled=True,
        )
    )
    assert panel._target_path.text() == long_path
    assert _label(panel, "AdmissionFinalFeatures").text().startswith("42 个面")
    assert "14/14" in _label(panel, "AdmissionClassConfirmation").text()
    assert panel._write_button.text() == "检查并写入标签库"
    assert panel._write_button.isEnabled()
    panel._target_path.setFocus()
    QTest.keyClick(
        panel._target_path,
        Qt.Key.Key_A,
        Qt.KeyboardModifier.ControlModifier,
    )
    QTest.keyClick(
        panel._target_path,
        Qt.Key.Key_C,
        Qt.KeyboardModifier.ControlModifier,
    )
    assert QApplication.clipboard().text() == long_path
    QTest.keyClick(panel._target_path, Qt.Key.Key_Tab)
    assert panel._write_button.hasFocus()
    QTest.keyClick(panel._write_button, Qt.Key.Key_Space)
    assert requested == [True]

    panel.render(
        AdmissionSummarySnapshot(
            target_path=long_path,
            topology_executed=True,
            topology_issue_count=2,
            allow_issues=True,
            background_stage="正在提交，不能取消",
            blocker_reason="正在提交，不能取消",
            accepted_feature_count=7,
            accepted_warnings=("accepted_labels 已写入，但目录同步未确认",),
        )
    )
    assert "未提供" in _label(panel, "AdmissionFinalFeatures").text()
    assert "问题数 2" in _label(panel, "AdmissionTopology").text()
    assert _label(panel, "AdmissionBackgroundStage").text() == "正在提交，不能取消"
    assert _label(panel, "AdmissionAcceptedResult").text() == "7 个面。"
    assert "完整性检查仍会执行" in _label(panel, "AdmissionIntegrityNote").text()
    assert "已写入" in _label(panel, "AdmissionAcceptedWarnings").text()
    assert not panel._write_button.isEnabled()

    screenshot_dir = Path(
        os.environ.get("LOESS_ADMISSION_SCREENSHOT_DIR")
        or ROOT / ".agent/tasks/V2-20260925-RESULT-STAGES"
    )
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    screenshot = screenshot_dir / "admission-summary-900x560.png"
    assert panel.grab().save(str(screenshot)), screenshot
    assert screenshot.stat().st_size > 1000
    panel.close()
    app.exitQgis()
    print(json.dumps({"screenshot": str(screenshot)}))
    print("admission summary: passed")


if __name__ == "__main__":
    main()
