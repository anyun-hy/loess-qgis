"""Manual edit controls and display rules; no task or layer state is retained."""

from __future__ import annotations

from dataclasses import dataclass

from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from labeling_tool.refinement.manual_edit_state import ManualKind, ManualState
from labeling_tool.shared.contracts.run_spec import CLASS_NAMES, CLASS_ORDER


@dataclass(frozen=True)
class ManualPanelSnapshot:
    """Immutable values sampled by the dialog immediately before a render."""

    current_class_text: str
    active_layer_name: str
    selected_count: int
    edit_text: str
    idle_enabled: bool
    has_features: bool
    has_workspace: bool
    kind: ManualKind | None = None
    state: ManualState | None = None
    modify_selected_count: int = 0
    delete_selected_count: int = 0
    pending_count: int = 0
    pending_has_error: bool = False
    target_changed: bool = False
    smoothing_enabled: bool = False
    smoothing_ready: bool = False


class ManualEditPanel(QGroupBox):
    """Own manual widgets for the dialog lifetime and emit user requests."""

    modify_requested = pyqtSignal()
    delete_requested = pyqtSignal()
    add_requested = pyqtSignal()
    target_changed = pyqtSignal(int)
    smoothing_enabled_changed = pyqtSignal(bool)
    smoothing_parameters_changed = pyqtSignal()
    primary_requested = pyqtSignal()
    retry_requested = pyqtSignal()
    clear_requested = pyqtSignal()
    continue_requested = pyqtSignal()
    cancel_requested = pyqtSignal()
    finish_requested = pyqtSignal()

    def __init__(
        self,
        parameters: tuple[int, float, float],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("人工操作", parent)
        self.setObjectName("manualEditPanel")
        layout = QVBoxLayout(self)
        self._context = QLabel(
            "当前类别：未选择 | QGIS 活动层：未同步 | 已选面：0 | 编辑状态：-"
        )
        self._context.setWordWrap(True)
        layout.addWidget(self._context)

        task_row = QHBoxLayout()
        self._modify = QPushButton("修改现有面")
        self._delete = QPushButton("删除现有面")
        self._add = QPushButton("新增面")
        for button, name in (
            (self._modify, "manualModifyTask"),
            (self._delete, "manualDeleteTask"),
            (self._add, "manualAddTask"),
        ):
            button.setObjectName(name)
            task_row.addWidget(button)
        task_row.addStretch()
        layout.addLayout(task_row)

        self._instruction = QLabel("请先在表格选择类别，再选择人工任务")
        self._instruction.setObjectName("manualInstruction")
        self._instruction.setWordWrap(True)
        layout.addWidget(self._instruction)
        option_row = QHBoxLayout()
        self._target_label = QLabel("本批目标类别:")
        self._target = QComboBox()
        self._target.setObjectName("manualTargetClass")
        for code in CLASS_ORDER:
            self._target.addItem(f"{code} {CLASS_NAMES[code]}", code)
        option_row.addWidget(self._target_label)
        option_row.addWidget(self._target)
        option_row.addStretch()
        layout.addLayout(option_row)

        smooth_row = QHBoxLayout()
        self._smooth_enabled = QCheckBox("光滑处理")
        self._smooth_enabled.setObjectName("manualSmoothEnabled")
        self._iterations_label = QLabel("次数:")
        self._iterations = QSpinBox()
        self._iterations.setObjectName("manualSmoothIterations")
        self._iterations.setRange(1, 3)
        self._iterations.setSuffix(" 次")
        self._offset_label = QLabel("偏移:")
        self._offset = QDoubleSpinBox()
        self._offset.setObjectName("manualSmoothOffset")
        self._offset.setRange(0.05, 0.45)
        self._offset.setSingleStep(0.05)
        self._offset.setDecimals(2)
        self._angle_label = QLabel("最大角度:")
        self._angle = QDoubleSpinBox()
        self._angle.setObjectName("manualSmoothAngle")
        self._angle.setRange(30.0, 180.0)
        self._angle.setSingleStep(10.0)
        self._angle.setDecimals(0)
        self._angle.setSuffix("°")
        self.set_smoothing_parameters(parameters)
        for widget in (
            self._smooth_enabled,
            self._iterations_label,
            self._iterations,
            self._offset_label,
            self._offset,
            self._angle_label,
            self._angle,
        ):
            smooth_row.addWidget(widget)
        smooth_row.addStretch()
        layout.addLayout(smooth_row)
        self._smooth_status = QLabel(
            "光滑默认关闭；开启后参数变化会自动预览本批全部新边界"
        )
        self._smooth_status.setObjectName("manualSmoothStatus")
        self._smooth_status.setWordWrap(True)
        layout.addWidget(self._smooth_status)

        action_row = QHBoxLayout()
        self._primary = QPushButton("开始")
        self._retry = QPushButton("重新绘制当前面")
        self._clear = QPushButton("清空选择")
        self._continue = QPushButton("继续任务")
        self._cancel = QPushButton("取消任务")
        self._finish = QPushButton("结束新增")
        for button, name in (
            (self._primary, "manualPrimary"),
            (self._retry, "manualRetry"),
            (self._clear, "manualClear"),
            (self._continue, "manualContinue"),
            (self._cancel, "manualCancel"),
            (self._finish, "manualFinish"),
        ):
            button.setObjectName(name)
            action_row.addWidget(button)
        action_row.addStretch()
        layout.addLayout(action_row)

        self._modify.clicked.connect(self.modify_requested)
        self._delete.clicked.connect(self.delete_requested)
        self._add.clicked.connect(self.add_requested)
        self._target.currentIndexChanged.connect(self._emit_target_changed)
        self._smooth_enabled.toggled.connect(self.smoothing_enabled_changed)
        for spin in (self._iterations, self._offset, self._angle):
            spin.valueChanged.connect(self.smoothing_parameters_changed)
        self._primary.clicked.connect(self.primary_requested)
        self._retry.clicked.connect(self.retry_requested)
        self._clear.clicked.connect(self.clear_requested)
        self._continue.clicked.connect(self.continue_requested)
        self._cancel.clicked.connect(self.cancel_requested)
        self._finish.clicked.connect(self.finish_requested)

    def _emit_target_changed(self, _index: int) -> None:
        code = self.target_code()
        if code is not None:
            self.target_changed.emit(code)

    def target_code(self) -> int | None:
        """Return the selected class code, or None before a valid selection."""
        code = self._target.currentData()
        return int(code) if code is not None else None

    def set_target_code(self, class_code: int) -> None:
        """Synchronize the combo without issuing a user request."""
        index = self._target.findData(int(class_code))
        if index >= 0:
            blocked = self._target.blockSignals(True)
            try:
                self._target.setCurrentIndex(index)
            finally:
                self._target.blockSignals(blocked)

    def smoothing_parameters(self) -> tuple[int, float, float]:
        """Return (iterations, offset, max angle) from the visible controls."""
        return self._iterations.value(), self._offset.value(), self._angle.value()

    def set_smoothing_parameters(self, parameters: tuple[int, float, float]) -> None:
        """Synchronize controls without recursively emitting parameter changes."""
        for spin, value in zip(
            (self._iterations, self._offset, self._angle), parameters
        ):
            blocked = spin.blockSignals(True)
            try:
                spin.setValue(value)
            finally:
                spin.blockSignals(blocked)

    def set_smoothing_enabled(self, enabled: bool) -> None:
        """Reset the checkbox without issuing a smoothing request."""
        blocked = self._smooth_enabled.blockSignals(True)
        try:
            self._smooth_enabled.setChecked(enabled)
        finally:
            self._smooth_enabled.blockSignals(blocked)

    def set_instruction(self, text: str) -> None:
        self._instruction.setText(text)

    def set_smoothing_status(self, text: str) -> None:
        self._smooth_status.setText(text)

    def focus_panel(self) -> None:
        self.setFocus()

    def render(self, view: ManualPanelSnapshot) -> None:
        """Apply display rules using only the dialog's current value snapshot."""
        selection_text = (
            f"待修改旧面：{view.modify_selected_count}"
            if view.kind == "modify"
            else f"已选面：{view.selected_count}"
        )
        self._context.setText(
            f"当前类别：{view.current_class_text} | QGIS 活动层：{view.active_layer_name} | "
            f"{selection_text} | 编辑状态：{view.edit_text}"
        )
        self._modify.setEnabled(view.idle_enabled and view.has_features)
        self._delete.setEnabled(view.idle_enabled and view.has_features)
        self._add.setEnabled(view.idle_enabled)
        show_target = bool(
            (view.kind == "modify" and view.modify_selected_count > 0)
            or (view.kind == "add" and view.pending_count > 0)
        )
        self._target_label.setVisible(show_target)
        self._target.setVisible(show_target)
        self._target.setEnabled(
            show_target
            and view.has_workspace
            and view.state not in ("committing", "paused")
        )
        show_smoothing = view.kind in ("modify", "add") and view.pending_count > 0
        for widget in (
            self._smooth_enabled,
            self._iterations_label,
            self._iterations,
            self._offset_label,
            self._offset,
            self._angle_label,
            self._angle,
            self._smooth_status,
        ):
            widget.setVisible(show_smoothing)
        active = view.state not in ("committing", "paused")
        self._smooth_enabled.setEnabled(show_smoothing and active)
        for spin in (self._iterations, self._offset, self._angle):
            spin.setEnabled(show_smoothing and view.smoothing_enabled and active)
        for button in (
            self._primary,
            self._retry,
            self._clear,
            self._continue,
            self._cancel,
            self._finish,
        ):
            button.setVisible(False)
            button.setEnabled(True)
        self._primary.setText("开始")
        self._retry.setText("重新绘制当前面")
        self._clear.setText("清空选择")
        self._continue.setText("继续任务")
        self._cancel.setText("取消任务")
        self._finish.setText("结束新增")
        if view.kind is None:
            return
        if view.state == "paused":
            self._continue.setVisible(True)
            if view.kind in ("modify", "add"):
                self._finish.setVisible(True)
                self._finish.setText(
                    "结束修改" if view.kind == "modify" else "结束新增"
                )
            else:
                self._cancel.setVisible(True)
                self._cancel.setText("结束删除")
        elif view.kind == "modify":
            self._finish.setVisible(True)
            self._finish.setText("结束修改")
            if view.modify_selected_count:
                self._primary.setVisible(True)
                self._primary.setText("保存修改并继续")
                self._primary.setEnabled(
                    active
                    and not view.pending_has_error
                    and (not view.smoothing_enabled or view.smoothing_ready)
                    and (view.pending_count > 0 or view.target_changed)
                )
                self._retry.setVisible(True)
                self._retry.setText(
                    "重新绘制当前面" if view.pending_count else "绘制新边界"
                )
                self._retry.setEnabled(active)
        elif view.kind == "delete":
            self._primary.setVisible(True)
            self._primary.setText(f"删除选中的 {view.delete_selected_count} 个面")
            self._primary.setEnabled(
                view.delete_selected_count > 0 and view.state == "selecting"
            )
            self._clear.setVisible(True)
            self._clear.setEnabled(view.delete_selected_count > 0)
            self._cancel.setVisible(True)
            self._cancel.setText("结束删除")
        elif view.kind == "add":
            self._finish.setVisible(True)
            self._finish.setText("结束新增")
            if view.pending_count:
                self._primary.setVisible(True)
                self._primary.setText("保存新增面并继续新增面")
                self._primary.setEnabled(
                    active
                    and not view.pending_has_error
                    and (not view.smoothing_enabled or view.smoothing_ready)
                )
                self._retry.setVisible(True)
                self._retry.setEnabled(active)
            elif view.state in ("capture_cancelled", "failed"):
                self._retry.setVisible(True)
                self._retry.setEnabled(active)
