"""Display-only class review list and focused action controls."""

from __future__ import annotations

from dataclasses import dataclass

from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from labeling_tool.qgis_support.qt6_api import (
    ALIGN_CENTER,
    NO_EDIT_TRIGGERS,
    RESIZE_TO_CONTENTS,
    SELECT_ROWS,
    SINGLE_SELECTION,
    STRETCH,
    USER_ROLE,
)


@dataclass(frozen=True)
class ClassReviewRow:
    """One already-evaluated class row supplied by the refinement dialog."""

    class_code: int
    class_name: str
    visible: bool
    feature_count: int | None
    review_text: str
    confirmed: bool
    unsaved: bool
    manual_enabled: bool = False
    manual_reason: str = ""
    sam_existing_enabled: bool = False
    sam_missed_enabled: bool = False
    sam_reason: str = ""
    confirm_enabled: bool = False
    confirm_reason: str = ""


@dataclass(frozen=True)
class ClassReviewSnapshot:
    """Current presentation values; this panel owns none of their business state."""

    rows: tuple[ClassReviewRow, ...]
    selected_class_code: int | None = None
    selection_locked: bool = False
    selection_locked_reason: str = ""


class ClassReviewPanel(QGroupBox):
    """Render class state and forward intent to the dialog with explicit codes."""

    class_selected = pyqtSignal(int)
    visibility_requested = pyqtSignal(int, bool)
    manual_requested = pyqtSignal(int)
    sam_requested = pyqtSignal(int, bool)
    confirm_requested = pyqtSignal(int, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("类别审核", parent)
        self.setObjectName("classReviewPanel")
        self._rows: dict[int, ClassReviewRow] = {}
        self._selected_class_code: int | None = None
        self._rendering = False
        self._selection_locked = False

        layout = QVBoxLayout(self)
        self._context = QLabel("当前类别：未选择")
        self._context.setObjectName("classReviewContext")
        self._context.setWordWrap(True)
        layout.addWidget(self._context)

        self._table = QTableWidget(0, 4)
        self._table.setObjectName("classReviewTable")
        self._table.setHorizontalHeaderLabels(["可见", "类别", "面数", "审核状态"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(NO_EDIT_TRIGGERS)
        self._table.setSelectionBehavior(SELECT_ROWS)
        self._table.setSelectionMode(SINGLE_SELECTION)
        self._table.setTabKeyNavigation(False)
        self._table.setToolTip("选择类别后，使用下方的集中操作区")
        header = self._table.horizontalHeader()
        for column in (0, 2):
            header.setSectionResizeMode(column, RESIZE_TO_CONTENTS)
        header.setSectionResizeMode(1, RESIZE_TO_CONTENTS)
        header.setSectionResizeMode(3, STRETCH)
        self._table.currentCellChanged.connect(self._select_row)
        layout.addWidget(self._table, stretch=1)

        actions = QWidget()
        actions.setObjectName("classReviewActions")
        action_layout = QHBoxLayout(actions)
        action_layout.setContentsMargins(0, 0, 0, 0)
        self._manual = QPushButton("人工修整")
        self._manual.setObjectName("classReviewManual")
        self._sam = QPushButton("SAM 辅助")
        self._sam.setObjectName("classReviewSam")
        menu = QMenu(self._sam)
        self._sam_existing = menu.addAction("校正已有地物")
        self._sam_missed = menu.addAction("新增漏标面")
        self._sam.setMenu(menu)
        self._confirm = QPushButton("确认整类")
        self._confirm.setObjectName("classReviewConfirm")
        self._confirm.setCheckable(True)
        for widget in (self._manual, self._sam, self._confirm):
            action_layout.addWidget(widget)
        action_layout.addStretch()
        layout.addWidget(actions)

        self._hint = QLabel("请选择类别")
        self._hint.setObjectName("classReviewActionHint")
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)

        self._manual.clicked.connect(self._request_manual)
        self._sam_existing.triggered.connect(lambda: self._request_sam(False))
        self._sam_missed.triggered.connect(lambda: self._request_sam(True))
        self._confirm.toggled.connect(self._request_confirmation)
        self.render(ClassReviewSnapshot(()))

    def render(self, snapshot: ClassReviewSnapshot) -> None:
        """Render the dialog-supplied state without inferring business readiness."""
        self._rendering = True
        try:
            self._rows = {row.class_code: row for row in snapshot.rows}
            self._selection_locked = bool(snapshot.selection_locked)
            self._table.setRowCount(len(snapshot.rows))
            for index, row in enumerate(snapshot.rows):
                self._render_row(index, row)
            selected = snapshot.selected_class_code
            if selected not in self._rows:
                selected = None
            self._selected_class_code = selected
            self._table.blockSignals(True)
            try:
                if selected is None:
                    self._table.clearSelection()
                    self._table.setCurrentCell(-1, -1)
                else:
                    row_index = next(
                        index
                        for index, row in enumerate(snapshot.rows)
                        if row.class_code == selected
                    )
                    self._table.setCurrentCell(row_index, 1)
                    self._table.selectRow(row_index)
            finally:
                self._table.blockSignals(False)
            self._table.setEnabled(not self._selection_locked)
            self._render_actions(snapshot.selection_locked_reason)
        finally:
            self._rendering = False

    def _render_row(self, index: int, row: ClassReviewRow) -> None:
        visible = self._table.cellWidget(index, 0)
        checkbox = visible.findChild(QCheckBox) if visible is not None else None
        if checkbox is None:
            visible = QWidget()
            visible_layout = QHBoxLayout(visible)
            visible_layout.setContentsMargins(6, 0, 6, 0)
            checkbox = QCheckBox(visible)
            checkbox.toggled.connect(
                lambda checked, box=checkbox: self._request_visibility_from_widget(
                    box, checked
                )
            )
            visible_layout.addWidget(checkbox)
            visible_layout.setAlignment(ALIGN_CENTER)
            self._table.setCellWidget(index, 0, visible)
        checkbox.setObjectName(f"classReviewVisible{row.class_code}")
        checkbox.setProperty("class_code", int(row.class_code))
        checkbox.setAccessibleName(f"显示类别 {row.class_code} {row.class_name}")
        blocked = checkbox.blockSignals(True)
        try:
            checkbox.setChecked(row.visible)
        finally:
            checkbox.blockSignals(blocked)
        checkbox.setEnabled(not self._selection_locked)
        name_item = QTableWidgetItem(f"{row.class_code} {row.class_name}")
        name_item.setToolTip(name_item.text())
        name_item.setData(USER_ROLE, int(row.class_code))
        self._table.setItem(index, 1, name_item)
        count_item = QTableWidgetItem(
            "—" if row.feature_count is None else str(row.feature_count)
        )
        self._table.setItem(index, 2, count_item)
        status = self._confirmation_text(row)
        if row.review_text:
            status = f"{status} | {row.review_text}"
        if row.unsaved:
            status = f"{status}；未保存编辑" if status else "未保存编辑"
        status_item = QTableWidgetItem(status or "—")
        status_item.setToolTip(status_item.text())
        self._table.setItem(index, 3, status_item)

    def _select_row(self, row_index: int, _column: int, *_previous) -> None:
        if self._rendering or self._selection_locked:
            return
        if 0 <= row_index < self._table.rowCount():
            item = self._table.item(row_index, 1)
            if item is None:
                return
            try:
                class_code = int(item.data(USER_ROLE))
            except (TypeError, ValueError):
                return
            if class_code in self._rows:
                self._selected_class_code = class_code
                self._render_actions("")
                self.class_selected.emit(class_code)

    def _request_visibility_from_widget(
        self, checkbox: QCheckBox, visible: bool
    ) -> None:
        if not self._rendering and not self._selection_locked:
            try:
                class_code = int(checkbox.property("class_code"))
            except (TypeError, ValueError):
                return
            if class_code in self._rows:
                self.visibility_requested.emit(class_code, visible)

    def _request_manual(self) -> None:
        if self._selected_class_code is not None:
            self.manual_requested.emit(self._selected_class_code)

    def _request_sam(self, missed: bool) -> None:
        if self._selected_class_code is not None:
            self.sam_requested.emit(self._selected_class_code, missed)

    def _request_confirmation(self, checked: bool) -> None:
        if not self._rendering and self._selected_class_code is not None:
            self.confirm_requested.emit(self._selected_class_code, checked)

    def _render_actions(self, locked_reason: str) -> None:
        row = self._rows.get(self._selected_class_code)
        if row is None:
            self._context.setText("当前类别：未选择")
            self._hint.setText(locked_reason or "请选择类别")
            for widget in (self._manual, self._sam, self._confirm):
                widget.setEnabled(False)
            return
        count = "—" if row.feature_count is None else str(row.feature_count)
        self._context.setText(
            f"当前类别：{row.class_code} {row.class_name} | 面数：{count} | "
            f"{self._confirmation_text(row)} | {row.review_text or '—'}"
        )
        manual_enabled = row.manual_enabled and not self._selection_locked
        sam_existing_enabled = row.sam_existing_enabled and not self._selection_locked
        sam_missed_enabled = row.sam_missed_enabled and not self._selection_locked
        sam_enabled = sam_existing_enabled or sam_missed_enabled
        confirm_enabled = row.confirm_enabled and not self._selection_locked
        self._manual.setEnabled(manual_enabled)
        self._manual.setToolTip(row.manual_reason)
        self._sam.setEnabled(sam_enabled)
        self._sam.setToolTip(row.sam_reason)
        self._sam_existing.setEnabled(sam_existing_enabled)
        self._sam_missed.setEnabled(sam_missed_enabled)
        self._confirm.setEnabled(confirm_enabled)
        self._confirm.setToolTip(row.confirm_reason)
        blocked = self._confirm.blockSignals(True)
        try:
            self._confirm.setChecked(row.confirmed)
            self._confirm.setText(
                "取消整类确认"
                if row.confirmed
                else "确认整类"
                if row.feature_count
                else "确认本范围无该类"
            )
        finally:
            self._confirm.blockSignals(blocked)
        reasons = (
            locked_reason,
            "" if manual_enabled else row.manual_reason,
            "" if sam_enabled else row.sam_reason,
            "" if confirm_enabled else row.confirm_reason,
        )
        self._hint.setText(
            next((reason for reason in reasons if reason), "可以操作当前类别")
        )

    @staticmethod
    def _confirmation_text(row: ClassReviewRow) -> str:
        if row.confirmed and row.feature_count == 0:
            return "已确认本范围无该类"
        return "已确认" if row.confirmed else "未确认"
