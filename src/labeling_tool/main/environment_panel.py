"""Input and status component for the dock's inference environment area."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from qgis.core import QgsProject
from qgis.PyQt.QtCore import QUrl, pyqtSignal
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from labeling_tool.main.environment_details import EnvironmentDetailsDialog
from labeling_tool.main.environment_report import first_problem
from labeling_tool.qgis_support.qt6_api import TEXT_SELECTABLE_BY_MOUSE


class EnvironmentPanel(QGroupBox):
    """Own environment widgets and the non-modal details-dialog lifecycle.

    The panel displays report snapshots supplied by the dock.  It never starts
    checks or retains an `InferenceConfigManager` report as mutable state.
    """

    script_path_changed = pyqtSignal(str)
    check_requested = pyqtSignal()
    details_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("推理环境", parent)
        self.setObjectName("environmentGroup")
        self._details_dialog: EnvironmentDetailsDialog | None = None
        self._build_ui()

    @property
    def scripts_directory(self) -> str:
        """Return the entered scripts directory with surrounding whitespace removed."""

        return self.script_path_edit.text().strip()

    @scripts_directory.setter
    def scripts_directory(self, value: str) -> None:
        self.script_path_edit.setText(str(value))

    def _build_ui(self) -> None:
        layout = QFormLayout(self)
        script_path_layout = QHBoxLayout()
        self.script_path_edit = QLineEdit()
        self.script_path_edit.setObjectName("environmentScriptPathEdit")
        self.script_path_edit.setPlaceholderText("inference_scripts/")
        self.browse_button = QPushButton("选择")
        self.browse_button.setObjectName("environmentBrowseButton")
        script_path_layout.addWidget(self.script_path_edit)
        script_path_layout.addWidget(self.browse_button)
        layout.addRow("推理脚本目录:", script_path_layout)

        config_path_layout = QHBoxLayout()
        self.config_path_label = QLabel("未选择")
        self.config_path_label.setObjectName("environmentConfigPathLabel")
        self.config_path_label.setWordWrap(True)
        self.config_path_label.setTextInteractionFlags(TEXT_SELECTABLE_BY_MOUSE)
        self.open_config_button = QPushButton("打开")
        self.open_config_button.setObjectName("environmentOpenConfigButton")
        self.open_config_button.setToolTip("打开当前 inference_scripts/config.yaml")
        self.open_config_button.setEnabled(False)
        config_path_layout.addWidget(self.config_path_label, stretch=1)
        config_path_layout.addWidget(self.open_config_button)
        layout.addRow("配置文件:", config_path_layout)

        self.status_label = QLabel("未检查")
        self.status_label.setObjectName("environmentStatusLabel")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(TEXT_SELECTABLE_BY_MOUSE)
        self.status_label.setStyleSheet(self._style("neutral"))
        layout.addRow("环境状态:", self.status_label)

        actions = QHBoxLayout()
        self.check_button = QPushButton("检查推理环境")
        self.check_button.setObjectName("environmentCheckButton")
        self.details_button = QPushButton("查看完整检查结果")
        self.details_button.setObjectName("environmentDetailsButton")
        self.details_button.setEnabled(False)
        actions.addWidget(self.check_button)
        actions.addWidget(self.details_button)
        layout.addRow(actions)

        self.script_path_edit.textChanged.connect(self._on_script_path_changed)
        self.browse_button.clicked.connect(self._browse_scripts)
        self.open_config_button.clicked.connect(self._open_config)
        self.check_button.clicked.connect(self._request_check)
        self.details_button.clicked.connect(self.details_requested)

    @staticmethod
    def _style(kind: str) -> str:
        colors = {
            "neutral": "#b8b8b8; background: #f4f4f4",
            "checking": "#4f83b6; background: #eaf4ff",
            "warning": "#c28b00; background: #fff7d6",
            "error": "#b42318; background: #fff0ee",
            "ready": "#2f855a; background: #e8f5ec",
        }
        return f"padding: 5px; border: 1px solid {colors[kind]};"

    def _on_script_path_changed(self, _text: str) -> None:
        self.refresh_config_path()
        self.script_path_changed.emit(self.scripts_directory)

    def refresh_config_path(self) -> str:
        """Refresh and return the current `config.yaml` display path."""

        config_path = (
            os.path.join(self.scripts_directory, "config.yaml")
            if self.scripts_directory
            else ""
        )
        self.config_path_label.setText(config_path or "未选择")
        self.open_config_button.setEnabled(os.path.isfile(config_path))
        return config_path

    def _browse_scripts(self) -> None:
        project_dir = os.path.dirname(QgsProject.instance().fileName()) or ""
        path = QFileDialog.getExistingDirectory(self, "选择推理脚本目录", project_dir)
        if path:
            self.scripts_directory = path

    def _open_config(self) -> None:
        config_path = self.refresh_config_path()
        if not os.path.isfile(config_path):
            from qgis.PyQt.QtWidgets import QMessageBox

            QMessageBox.warning(self, "配置文件", "当前脚本目录中没有 config.yaml")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(config_path))

    def _request_check(self) -> None:
        """Refresh the displayed config target before forwarding a user click."""

        self.refresh_config_path()
        self.check_requested.emit()

    def mark_check_required(self) -> None:
        """Invalidate the visible report after its inputs or task values change."""

        self.refresh_config_path()
        self.status_label.setText("配置已变化，请检查推理环境")
        self.status_label.setStyleSheet(self._style("warning"))
        self.status_label.setToolTip("")
        self.details_button.setEnabled(False)

    def show_checking(self) -> None:
        """Show an in-progress check while the manager owns the QProcess."""

        self.status_label.setText("正在检查实际推理环境，请稍候")
        self.status_label.setStyleSheet(self._style("checking"))
        self.status_label.setToolTip("")
        self.check_button.setEnabled(False)
        self.details_button.setEnabled(False)

    def show_report(
        self,
        report: Mapping[str, Any],
        task_checks: Sequence[Mapping[str, Any]],
        *,
        check_finished: bool = True,
    ) -> None:
        """Render a supplied report snapshot without opening its details dialog."""

        checks = list(report.get("checks") or []) + list(task_checks)
        status = str(report.get("status") or "error")
        environment_problem = first_problem(report.get("checks") or [])
        problem = first_problem(checks)
        if status == "ready":
            text = "环境检查通过：配置已加载"
            style = "ready"
        elif status == "warning":
            text = "检查通过但有警告"
            if environment_problem:
                text += f"：{environment_problem}"
            style = "warning"
        else:
            text = "检查未通过"
            if environment_problem:
                text += f"：{environment_problem}"
            style = "error"
        self.status_label.setText(text)
        self.status_label.setStyleSheet(self._style(style))
        self.status_label.setToolTip(
            "完整错误已放入详情窗口，可滚动、选择并复制。" if problem else ""
        )
        if check_finished:
            self.check_button.setEnabled(True)
        self.details_button.setEnabled(
            bool(report.get("checks") or report.get("stderr"))
        )

    def show_details(
        self,
        report: Mapping[str, Any],
        task_checks: Sequence[Mapping[str, Any]],
    ) -> None:
        """Open a details dialog pinned to these report and task-check values."""

        previous = self._details_dialog
        if previous is not None:
            previous.close()
        dialog = EnvironmentDetailsDialog(report, task_checks, self)
        self._details_dialog = dialog

        def clear_if_current(*_args: object) -> None:
            if self._details_dialog is dialog:
                self._details_dialog = None

        dialog.destroyed.connect(clear_if_current)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def cleanup(self) -> None:
        """Idempotently close the owned non-modal details dialog."""

        dialog = self._details_dialog
        self._details_dialog = None
        if dialog is not None:
            try:
                dialog.close()
            except RuntimeError:
                pass
