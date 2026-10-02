"""Non-modal, copyable presentation of one inference-environment report."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from qgis.PyQt.QtCore import QTimer
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from labeling_tool.main.environment_report import (
    check_label,
    compact_problem,
    format_check_details,
    format_execution_details,
)
from labeling_tool.qgis_support.qt6_api import (
    EXTENDED_SELECTION,
    NO_EDIT_TRIGGERS,
    RESIZE_TO_CONTENTS,
    SELECT_ROWS,
    STRETCH,
    USER_ROLE,
    WA_DELETE_ON_CLOSE,
)


class EnvironmentDetailsDialog(QDialog):
    """Display a frozen report snapshot without taking ownership of its source."""

    def __init__(
        self,
        report: Mapping[str, Any],
        task_checks: Sequence[Mapping[str, Any]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._report = deepcopy(dict(report))
        report_checks = self._report.get("checks") or []
        self._checks = deepcopy(list(report_checks) + list(task_checks))
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the table, full text, and copy actions from the snapshot."""

        self.setObjectName("environmentDetailsDialog")
        self.setWindowTitle("推理环境完整检查结果")
        self.resize(980, 650)
        self.setMinimumSize(760, 480)
        self.setModal(False)
        self.setAttribute(WA_DELETE_ON_CLOSE)

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_summary_label())

        tabs = QTabWidget()
        tabs.setObjectName("environmentDetailsTabs")
        layout.addWidget(tabs, stretch=1)
        self.checks_table = self._build_checks_table()
        tabs.addTab(self._table_page(self.checks_table), "全部检查项")

        self.full_text_edit = QPlainTextEdit()
        self.full_text_edit.setObjectName("environmentDetailsFullText")
        self.full_text_edit.setReadOnly(True)
        self.full_text = (
            format_execution_details(self._report)
            + "\n\n"
            + format_check_details(self._checks, str(self._report.get("stderr") or ""))
        )
        self.full_text_edit.setPlainText(self.full_text)
        self.full_text_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        tabs.addTab(self.full_text_edit, "完整文本 / 进程日志")

        buttons = QHBoxLayout()
        self.select_all_button = QPushButton("全选")
        self.copy_selected_button = QPushButton("复制选中项")
        self.copy_all_button = QPushButton("复制全部结果")
        self.close_button = QPushButton("关闭")
        for button, name in (
            (self.select_all_button, "environmentDetailsSelectAllButton"),
            (self.copy_selected_button, "environmentDetailsCopySelectedButton"),
            (self.copy_all_button, "environmentDetailsCopyAllButton"),
            (self.close_button, "environmentDetailsCloseButton"),
        ):
            button.setObjectName(name)
        buttons.addStretch()
        buttons.addWidget(self.select_all_button)
        buttons.addWidget(self.copy_selected_button)
        buttons.addWidget(self.copy_all_button)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)

        self.select_all_button.clicked.connect(
            lambda: self._select_all(tabs.currentWidget())
        )
        self.copy_selected_button.clicked.connect(self._copy_selected)
        self.copy_all_button.clicked.connect(self._copy_all)
        self.close_button.clicked.connect(self.close)

    def _build_summary_label(self) -> QLabel:
        counts = {
            status: sum(1 for check in self._checks if check.get("status") == status)
            for status in ("ready", "warning", "error")
        }
        effective = self._report.get("effective") or {}
        device = (effective.get("runtime") or {}).get("effective_device", "未确定")
        label = QLabel(
            f"检查项 {len(self._checks)}  |  正常 {counts['ready']}  |  "
            f"警告 {counts['warning']}  |  错误 {counts['error']}  |  "
            f"语义设备 {device}  |  检查编号 {self._report.get('check_id') or '无'}"
        )
        label.setObjectName("environmentDetailsSummary")
        label.setStyleSheet("font-weight: bold; padding: 4px;")
        return label

    def _build_checks_table(self) -> QTableWidget:
        table = QTableWidget(0, 5)
        table.setObjectName("environmentDetailsChecksTable")
        table.setHorizontalHeaderLabels(
            ["状态", "检查项", "当前值", "说明", "来源 / 修改位置"]
        )
        table.setEditTriggers(NO_EDIT_TRIGGERS)
        table.setSelectionBehavior(SELECT_ROWS)
        table.setSelectionMode(EXTENDED_SELECTION)
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(30)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, RESIZE_TO_CONTENTS)
        header.setSectionResizeMode(1, RESIZE_TO_CONTENTS)
        header.setSectionResizeMode(2, STRETCH)
        header.setSectionResizeMode(3, STRETCH)
        header.setSectionResizeMode(4, STRETCH)

        status_priority = {"error": 0, "warning": 1, "ready": 2}
        ordered_checks = sorted(
            enumerate(self._checks),
            key=lambda pair: (status_priority.get(pair[1].get("status"), 3), pair[0]),
        )
        table.setRowCount(len(ordered_checks))
        status_text = {"ready": "正常", "warning": "警告", "error": "错误"}
        status_color = {"ready": "#1f6f3d", "warning": "#9a6700", "error": "#b42318"}
        for row, (original_index, check) in enumerate(ordered_checks):
            status = str(check.get("status") or "")
            message = compact_problem(check, max_chars=240)
            source = str(check.get("source") or "")
            fix = str(check.get("fix") or "")
            source_fix = source if not fix else f"{source} | {fix}"
            values = (
                status_text.get(status, status or "未知"),
                check_label(check),
                str(check.get("value") or ""),
                message,
                source_fix,
            )
            tooltip = str(check.get("message") or "")
            if fix:
                tooltip = (tooltip + "\n" if tooltip else "") + "修改位置: " + fix
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(USER_ROLE, original_index)
                item.setToolTip(tooltip)
                if column == 0:
                    item.setForeground(QColor(status_color.get(status, "#333333")))
                table.setItem(row, column, item)
        return table

    @staticmethod
    def _table_page(table: QTableWidget) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(table)
        return page

    def _select_all(self, current_page: QWidget | None) -> None:
        if current_page is self.full_text_edit:
            self.full_text_edit.selectAll()
        else:
            self.checks_table.selectAll()

    def _copy_all(self) -> None:
        QApplication.clipboard().setText(self.full_text)
        self._set_copied(self.copy_all_button, "复制全部结果")

    def _copy_selected(self) -> None:
        selected_rows = sorted(
            {index.row() for index in self.checks_table.selectionModel().selectedRows()}
        )
        selected_checks = []
        for row in selected_rows:
            item = self.checks_table.item(row, 0)
            if item is not None:
                selected_checks.append(self._checks[item.data(USER_ROLE)])
        QApplication.clipboard().setText(
            format_check_details(selected_checks) if selected_checks else self.full_text
        )
        self._set_copied(self.copy_selected_button, "复制选中项")

    @staticmethod
    def _set_copied(button: QPushButton, original_text: str) -> None:
        button.setText("已复制")

        def restore_text() -> None:
            try:
                button.setText(original_text)
            except RuntimeError:
                pass

        QTimer.singleShot(2000, restore_text)
