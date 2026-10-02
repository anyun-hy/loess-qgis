# ruff: noqa: E402
"""Native rendering and intent tests for the display-only class review panel."""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
try:
    from qgis.core import QgsApplication
    from qgis.PyQt.QtCore import QCoreApplication, QEvent, Qt
    from qgis.PyQt.QtTest import QTest
    from qgis.PyQt.QtWidgets import QCheckBox, QPushButton, QTableWidget
except ModuleNotFoundError:
    raise SystemExit(77)

from labeling_tool.refinement.class_review_panel import (
    ClassReviewPanel,
    ClassReviewRow,
    ClassReviewSnapshot,
)


def rows():
    return (
        ClassReviewRow(
            11, "超长名称用于窄窗口可读性检查", True, 4, "待审核", False, True,
            manual_enabled=False, manual_reason="存在未保存编辑，请先保存或回滚",
            sam_existing_enabled=False, sam_missed_enabled=False,
            sam_reason="SAM 会话忙，不能切换类别",
            confirm_enabled=False, confirm_reason="存在未保存编辑，不能确认整类",
        ),
        ClassReviewRow(
            12, "空类别", False, 0, "未审核", False, False,
            manual_enabled=True, sam_existing_enabled=True, sam_missed_enabled=True,
            confirm_enabled=True,
        ),
        ClassReviewRow(21, "数量未知", True, None, "等待工作层挂载", True, False),
    )


def panel(app, root):
    widget = ClassReviewPanel()
    events = []
    try:
        widget.class_selected.connect(lambda code: events.append(("selected", code)))
        widget.visibility_requested.connect(lambda code, value: events.append(("visible", code, value)))
        widget.manual_requested.connect(lambda code: events.append(("manual", code)))
        widget.sam_requested.connect(lambda code, missed: events.append(("sam", code, missed)))
        widget.confirm_requested.connect(lambda code, checked: events.append(("confirm", code, checked)))
        widget.render(ClassReviewSnapshot(rows(), selected_class_code=11))
        widget.resize(640, 460)
        widget.show()
        app.processEvents()
        table = widget.findChild(QTableWidget, "classReviewTable")
        assert table is not None and table.item(2, 2).text() == "—"
        assert "未确认" in table.item(0, 3).text()
        assert "已确认" in table.item(2, 3).text()
        assert "未保存编辑" in table.item(0, 3).text()
        hint = widget.findChild(type(widget._hint), "classReviewActionHint")
        assert hint is not None and "未保存编辑" in hint.text()
        table.setFocus()
        QTest.keyClick(table, Qt.Key.Key_Down)
        assert events == [("selected", 12)]
        manual = widget.findChild(QPushButton, "classReviewManual")
        assert manual is not None
        for _ in range(4):
            focus = app.focusWidget()
            assert focus is not None
            QTest.keyClick(focus, Qt.Key.Key_Tab)
            app.processEvents()
            if manual.hasFocus():
                break
        assert manual.hasFocus()
        QTest.keyClick(manual, Qt.Key.Key_Space)
        sam = widget.findChild(QPushButton, "classReviewSam")
        sam.menu().actions()[0].trigger()
        sam.menu().actions()[1].trigger()
        confirm = widget.findChild(QPushButton, "classReviewConfirm")
        confirm.setFocus()
        QTest.keyClick(confirm, Qt.Key.Key_Space)
        visible = widget.findChild(QCheckBox, "classReviewVisible12")
        visible.setFocus()
        QTest.keyClick(visible, Qt.Key.Key_Space)
        assert events == [
            ("selected", 12), ("manual", 12), ("sam", 12, False),
            ("sam", 12, True), ("confirm", 12, True), ("visible", 12, True),
        ]
        events.clear()
        reordered = (rows()[2], rows()[0], rows()[1])
        widget.render(ClassReviewSnapshot(reordered, selected_class_code=21))
        assert "已确认" in widget._context.text()
        reordered_visible = table.cellWidget(0, 0).findChild(QCheckBox)
        assert reordered_visible is not None
        reordered_visible.click()
        table.setCurrentCell(1, 1)
        assert events == [("visible", 21, False), ("selected", 11)]
        widget.render(ClassReviewSnapshot(rows(), selected_class_code=12))
        output = Path(sys.argv[2]) if len(sys.argv) > 2 else root
        output.mkdir(parents=True, exist_ok=True)
        ready = output / "class-review-panel-ready-640x460.png"
        assert widget.grab().save(str(ready)) and ready.stat().st_size > 0
        widget.render(ClassReviewSnapshot(rows(), selected_class_code=12, selection_locked=True, selection_locked_reason="人工任务进行中，不能切换类别"))
        assert not table.isEnabled() and "不能切换类别" in hint.text()
        assert not widget.findChild(QPushButton, "classReviewManual").isEnabled()
        assert not confirm.isEnabled()
        screenshot = output / "class-review-panel-locked-640x460.png"
        assert widget.grab().save(str(screenshot)) and screenshot.stat().st_size > 0
    finally:
        widget.close()


app = QgsApplication([], False)
app.initQgis()
try:
    with tempfile.TemporaryDirectory(prefix="loess-class-review-panel-") as directory:
        panel(app, Path(directory))
        print("panel: passed", flush=True)
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
finally:
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.exitQgis()
