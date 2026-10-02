"""Native QGIS checks for the monitor result-stage presentation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))

try:
    from qgis.core import QgsApplication
    from qgis.PyQt.QtWidgets import QLabel
except ModuleNotFoundError:
    raise SystemExit(77)


def _label(page, name: str) -> QLabel:
    value = page.findChild(QLabel, name)
    assert value is not None, name
    return value


def _render(page, *, stream_id: str, name: str, coverage: dict) -> None:
    page.render_assembly(
        stream_id=stream_id,
        display_name=name,
        runtime={
            "phase": "write_formal",
            "phase_name": "写入正式 GPKG",
            "status": "completed",
            "progress_current": 12,
            "progress_total": 12,
            "feature_count": 12,
            "message": "synthetic assembly record",
        },
        phase_statuses={
            "write_formal": {"status": "completed", "current": 12, "total": 12}
        },
        coverage=coverage,
    )


def main() -> None:
    from labeling_tool.monitor.monitor_theme import MONITOR_STYLE
    from labeling_tool.monitor.pages._shared import scrollable_page
    from labeling_tool.monitor.pages.results import ResultsPage

    app = QgsApplication([], False)
    app.initQgis()
    page = ResultsPage()
    page.setStyleSheet(MONITOR_STYLE["dark"])
    page.resize(1280, 800)
    page.show()
    app.processEvents()

    page.upsert_stream("model:ready", "模型 A", {"status": "成功"})
    _render(page, stream_id="model:ready", name="模型 A", coverage={})
    assert "登记为成功" in _label(page, "ResultStageOutput").text()
    assert "覆盖验收通过" not in _label(page, "ResultStageGeometry").text()
    assert "尚未执行" in _label(page, "ResultStageGeometry").text()
    assert "未在此同步" in _label(page, "ResultStageReview").text()
    assert "未在此同步" in _label(page, "ResultStageAccepted").text()
    steps = page.findChild(type(page._steps), "AssemblySteps")
    assert steps.item(4, 1).text() == "完成"
    assert "12/12 单元" == steps.item(4, 2).text()

    _render(
        page,
        stream_id="model:ready",
        name="模型 A",
        coverage={"status": "passed", "gap_area_m2": 0.0},
    )
    assert "覆盖验收通过" in _label(page, "ResultStageGeometry").text()
    page.render_assembly(
        stream_id="",
        display_name="",
        runtime={"phase": "write_formal", "status": "completed"},
        phase_statuses={"write_formal": {"status": "completed", "current": 12}},
        coverage={"status": "passed", "gap_area_m2": 0.0},
    )
    assert page._detail.toPlainText() == "选择一个结果流查看十步组装记录。"
    assert "尚未执行" in page._coverage.text()
    assert "状态未提供" in _label(page, "ResultStageOutput").text()
    assert "尚未执行" in _label(page, "ResultStageGeometry").text()
    for row in range(steps.rowCount()):
        assert steps.item(row, 1).text() == "未开始"
        assert steps.item(row, 2).text().startswith("— / ")

    page.upsert_stream("fusion:failed", "融合结果", {"status": "失败"})
    _render(
        page,
        stream_id="fusion:failed",
        name="融合结果",
        coverage={"status": "failed", "gap_area_m2": 3.0},
    )
    assert "登记失败" in _label(page, "ResultStageOutput").text()
    assert "覆盖验收失败" in _label(page, "ResultStageGeometry").text()
    assert "模型 A" not in page._detail.toPlainText()

    page.reset()
    assert "状态未提供" in _label(page, "ResultStageOutput").text()
    assert "尚未执行" in _label(page, "ResultStageGeometry").text()
    assert page._detail.toPlainText() == "选择一个结果流查看十步组装记录。"
    assert steps.item(4, 1).text() == "未开始"

    screenshot_dir = Path(
        os.environ.get("LOESS_RESULT_STAGE_SCREENSHOT_DIR")
        or ROOT / ".agent/tasks/V2-20260925-RESULT-STAGES"
    )
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    screenshot = screenshot_dir / "result-stages-1280-dark.png"
    page.upsert_stream("model:ready", "模型 A", {"status": "成功"})
    _render(
        page,
        stream_id="model:ready",
        name="模型 A",
        coverage={"status": "passed", "gap_area_m2": 0.0, "overlap_area_m2": 0.0},
    )
    assert page.grab().save(str(screenshot)), screenshot
    assert screenshot.stat().st_size > 1000
    compact = scrollable_page(page, minimum_height=510)
    compact.resize(900, 560)
    page.setStyleSheet(MONITOR_STYLE["light"])
    compact.show()
    app.processEvents()
    for name in (
        "ResultStageOutput",
        "ResultStageGeometry",
        "ResultStageReview",
        "ResultStageAccepted",
    ):
        label = _label(page, name)
        assert label.isVisible() and label.height() >= label.fontMetrics().height()
    page._streams.setFocus()
    assert page._streams.hasFocus()
    compact_screenshot = screenshot_dir / "result-stages-900-light.png"
    assert compact.grab().save(str(compact_screenshot)), compact_screenshot
    assert compact_screenshot.stat().st_size > 1000
    page.close()
    app.exitQgis()
    print(
        json.dumps(
            {
                "screenshot": str(screenshot),
                "compact_screenshot": str(compact_screenshot),
                "switch_reset": True,
            }
        )
    )


if __name__ == "__main__":
    main()
