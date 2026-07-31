"""Modeless 14-class Fusion workspace with click-driven SAM3 refinement."""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path

from qgis.PyQt.QtCore import QTimer, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsFeatureRequest,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsRectangle,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsSettings,
)
from qgis.gui import (
    QgsMapToolCapture,
    QgsMapToolDigitizeFeature,
    QgsMapToolEmitPoint,
    QgsRubberBand,
)
from qgis.analysis import QgsZonalStatistics

from ..core import accepted_writer, class_workspace, final_assembler, topology_validator
from ..core.layer_names import LAYER_NAMES
from ..core.sam3_worker_runner import Sam3WorkerRunner
from ..core.style_manager import StyleManager
from ..core.run_spec import CLASS_NAMES, CLASS_ORDER
from ..qt_compat import (
    ALIGN_CENTER,
    DASH_LINE,
    ENSURE_VISIBLE,
    NO,
    NO_EDIT_TRIGGERS,
    RESIZE_TO_CONTENTS,
    SELECT_ROWS,
    SINGLE_SELECTION,
    STRETCH,
    WINDOW,
    YES,
)


class ClassRefinementDialog(QDialog):
    workspace_changed = pyqtSignal(object)

    def __init__(self, iface, layer_manager, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.layer_manager = layer_manager
        self.setWindowTitle("分类修整与组装")
        self.setWindowFlags(WINDOW)
        self.resize(1220, 760)
        self.setMinimumSize(980, 620)
        self._result = {}
        self._run_spec = {}
        self._sam_config = {}
        self._scripts_dir = ""
        self._eligible_fusions = []
        self._workspace = None
        self._manual_only = False
        self._class_layers = {}
        self._connected_layer_ids = set()
        self._layer_signal_slots = {}
        self._selection_signal_slots = {}
        self._edit_change_signal_slots = {}
        self._undo_stack_signal_slots = {}
        self._qgis_smooth_preview = None
        self._qgis_smooth_preview_bands = []
        self._smoothing_parameter_sync = False
        self._manual_smoothing_timer = QTimer(self)
        self._manual_smoothing_timer.setSingleShot(True)
        self._manual_smoothing_timer.setInterval(250)
        self._manual_smoothing_timer.timeout.connect(
            self._refresh_manual_smoothing_preview
        )
        self._tree_visibility_slots = {}
        self._syncing_class_selection = False
        self._iface_layer_signal_connected = False
        self._iface_current_layer_slot = self._active_layer_changed
        self._visibility_checks = {}
        self._sam_buttons = {}
        self._sam_existing_actions = {}
        self._sam_missed_actions = {}
        self._edit_buttons = {}
        self._confirm_buttons = {}
        self._snapshots = {}
        self._edit_context = {}
        self._metadata_update = False
        self._worker = None
        self._worker_request = None
        self._active_session = None
        self._previous_map_tool = None
        self._point_tool = None
        self._manual_pick_code = None
        self._manual_task = None
        self._manual_task_tool = None
        self._manual_capture_tool = None
        self._manual_previous_map_tool = None
        self._manual_reference_band = None
        self._manual_candidate_band = None
        self._manual_add_candidate_bands = []
        self._manual_capture_transition_action = None
        self._manual_capture_transition_task = None
        self._manual_capture_transition_timer = QTimer(self)
        self._manual_capture_transition_timer.setSingleShot(True)
        self._manual_capture_transition_timer.timeout.connect(
            self._run_manual_capture_transition
        )
        self._manual_retired_capture_tools = []
        self._manual_capture_retire_timer = QTimer(self)
        self._manual_capture_retire_timer.setSingleShot(True)
        self._manual_capture_retire_timer.timeout.connect(
            self._release_retired_manual_capture_tools
        )
        self._setting_manual_map_tool = False
        self._map_tool_signal_connected = False
        self._map_tool_slot = self._map_tool_changed
        self._current_band = None
        self._candidate_band = None
        self._confidence_raster = None
        self._final_path = ""
        self._issues_path = ""
        self._issue_count = None
        self._build_ui()
        self._connect_iface_layer_signal()
        self._connect_map_tool_signal()

    def _build_ui(self):
        root = QVBoxLayout(self)
        baseline = QHBoxLayout()
        fusion_title = QLabel("Fusion 基准:")
        fusion_title.setMinimumWidth(fusion_title.sizeHint().width())
        baseline.addWidget(fusion_title)
        self.fusion_combo = QComboBox()
        baseline.addWidget(self.fusion_combo, stretch=1)
        self.initialize_btn = QPushButton("初始化 14 类工作层")
        self.initialize_btn.clicked.connect(self._initialize_workspace)
        baseline.addWidget(self.initialize_btn)
        self.baseline_label = QLabel("尚未加载运行结果")
        self.baseline_label.setWordWrap(True)
        baseline.addWidget(self.baseline_label, stretch=2)
        root.addLayout(baseline)

        self.table = QTableWidget(len(CLASS_ORDER), 8)
        self.table.setHorizontalHeaderLabels(
            ["可见", "类别", "色块", "类别工作层状态", "面数", "SAM3 校正", "人工操作", "整类确认"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(NO_EDIT_TRIGGERS)
        self.table.setSelectionBehavior(SELECT_ROWS)
        self.table.setSelectionMode(SINGLE_SELECTION)
        self.table.currentCellChanged.connect(self._activate_row)
        self.table.setToolTip(
            "当前选中行与 QGIS 图层面板中的活动类别工作层保持同步"
        )
        header = self.table.horizontalHeader()
        for column in (0, 1, 2, 4, 5, 6, 7):
            header.setSectionResizeMode(column, RESIZE_TO_CONTENTS)
        header.setSectionResizeMode(3, STRETCH)
        for row, code in enumerate(CLASS_ORDER):
            visible = QCheckBox()
            visible.setChecked(True)
            visible.toggled.connect(lambda checked, c=code: self._set_visible(c, checked))
            visible_box = QWidget()
            visible_layout = QHBoxLayout(visible_box)
            visible_layout.setContentsMargins(6, 0, 6, 0)
            visible_layout.addWidget(visible)
            visible_layout.setAlignment(ALIGN_CENTER)
            self.table.setCellWidget(row, 0, visible_box)
            self._visibility_checks[code] = visible
            self.table.setItem(row, 1, QTableWidgetItem(f"{code} {CLASS_NAMES[code]}"))
            swatch = QTableWidgetItem("    ")
            swatch.setBackground(QColor(StyleManager.get_class_color(code)))
            self.table.setItem(row, 2, swatch)
            self.table.setItem(row, 3, QTableWidgetItem("未初始化"))
            self.table.setItem(row, 4, QTableWidgetItem("0"))
            sam_button = QPushButton("校正边界")
            menu = sam_button.menu()
            if menu is None:
                from qgis.PyQt.QtWidgets import QMenu

                menu = QMenu(sam_button)
                sam_button.setMenu(menu)
            existing_action = menu.addAction("校正已有地物")
            existing_action.triggered.connect(
                lambda _checked=False, c=code: self._begin_sam(c, missed=False)
            )
            missed_action = menu.addAction("新增漏标面")
            missed_action.triggered.connect(
                lambda _checked=False, c=code: self._begin_sam(c, missed=True)
            )
            menu.aboutToShow.connect(
                lambda c=code: self._select_class_context(c, activate_layer=True)
            )
            self.table.setCellWidget(row, 5, sam_button)
            self._sam_buttons[code] = sam_button
            self._sam_existing_actions[code] = existing_action
            self._sam_missed_actions[code] = missed_action
            edit_button = QPushButton("打开操作")
            edit_button.clicked.connect(
                lambda _checked=False, c=code: self._open_manual_operations(c)
            )
            self.table.setCellWidget(row, 6, edit_button)
            self._edit_buttons[code] = edit_button
            confirm = QPushButton("确认整类")
            confirm.setCheckable(True)
            confirm.setToolTip("确认当前类别全部面已审核完成，不用于保存单个面")
            confirm.pressed.connect(
                lambda c=code: self._select_class_context(c, activate_layer=True)
            )
            confirm.toggled.connect(lambda checked, c=code: self._confirm_class(c, checked))
            self.table.setCellWidget(row, 7, confirm)
            self._confirm_buttons[code] = confirm
        root.addWidget(self.table, stretch=1)

        smooth_settings = QgsSettings()

        def smooth_setting(key, default, caster):
            try:
                return caster(smooth_settings.value(key, default))
            except (TypeError, ValueError):
                return default

        self.manual_group = QGroupBox("人工操作")
        manual_layout = QVBoxLayout(self.manual_group)
        self.manual_context_label = QLabel("当前类别：未选择 | QGIS 活动层：未同步 | 已选面：0 | 编辑状态：-")
        self.manual_context_label.setWordWrap(True)
        manual_layout.addWidget(self.manual_context_label)
        task_row = QHBoxLayout()
        self.modify_task_btn = QPushButton("修改现有面")
        self.delete_task_btn = QPushButton("删除现有面")
        self.add_task_btn = QPushButton("新增面")
        task_row.addWidget(self.modify_task_btn)
        task_row.addWidget(self.delete_task_btn)
        task_row.addWidget(self.add_task_btn)
        task_row.addStretch()
        manual_layout.addLayout(task_row)
        self.manual_instruction_label = QLabel("请先在表格选择类别，再选择人工任务")
        self.manual_instruction_label.setWordWrap(True)
        manual_layout.addWidget(self.manual_instruction_label)
        option_row = QHBoxLayout()
        self.target_class_label = QLabel("本批目标类别:")
        self.target_class_combo = QComboBox()
        for code in CLASS_ORDER:
            self.target_class_combo.addItem(f"{code} {CLASS_NAMES[code]}", code)
        option_row.addWidget(self.target_class_label)
        option_row.addWidget(self.target_class_combo)
        option_row.addStretch()
        manual_layout.addLayout(option_row)
        manual_smooth_row = QHBoxLayout()
        self.manual_smooth_enabled_check = QCheckBox("光滑处理")
        self.manual_smooth_iterations_label = QLabel("次数:")
        self.manual_smooth_iterations_spin = QSpinBox()
        self.manual_smooth_iterations_spin.setRange(1, 3)
        self.manual_smooth_iterations_spin.setSuffix(" 次")
        self.manual_smooth_iterations_spin.setValue(smooth_setting(
            "labeling_tool/qgis_smoothing/iterations", 1, int
        ))
        self.manual_smooth_offset_spin = QDoubleSpinBox()
        self.manual_smooth_offset_spin.setRange(0.05, 0.45)
        self.manual_smooth_offset_spin.setSingleStep(0.05)
        self.manual_smooth_offset_spin.setDecimals(2)
        self.manual_smooth_offset_spin.setValue(smooth_setting(
            "labeling_tool/qgis_smoothing/offset", 0.25, float
        ))
        self.manual_smooth_offset_label = QLabel("偏移:")
        self.manual_smooth_angle_spin = QDoubleSpinBox()
        self.manual_smooth_angle_spin.setRange(30.0, 180.0)
        self.manual_smooth_angle_spin.setSingleStep(10.0)
        self.manual_smooth_angle_spin.setDecimals(0)
        self.manual_smooth_angle_spin.setSuffix("°")
        self.manual_smooth_angle_spin.setValue(smooth_setting(
            "labeling_tool/qgis_smoothing/max_angle", 180.0, float
        ))
        self.manual_smooth_angle_label = QLabel("最大角度:")
        manual_smooth_row.addWidget(self.manual_smooth_enabled_check)
        manual_smooth_row.addWidget(self.manual_smooth_iterations_label)
        manual_smooth_row.addWidget(self.manual_smooth_iterations_spin)
        manual_smooth_row.addWidget(self.manual_smooth_offset_label)
        manual_smooth_row.addWidget(self.manual_smooth_offset_spin)
        manual_smooth_row.addWidget(self.manual_smooth_angle_label)
        manual_smooth_row.addWidget(self.manual_smooth_angle_spin)
        manual_smooth_row.addStretch()
        manual_layout.addLayout(manual_smooth_row)
        self.manual_smooth_status_label = QLabel(
            "光滑默认关闭；开启后参数变化会自动预览本批全部新边界"
        )
        self.manual_smooth_status_label.setWordWrap(True)
        manual_layout.addWidget(self.manual_smooth_status_label)
        candidate_row = QHBoxLayout()
        self.manual_primary_btn = QPushButton("开始")
        self.manual_retry_btn = QPushButton("重新绘制当前面")
        self.manual_clear_btn = QPushButton("清空选择")
        self.manual_continue_btn = QPushButton("继续任务")
        self.manual_cancel_btn = QPushButton("取消任务")
        self.manual_finish_btn = QPushButton("结束新增")
        for button in (
            self.manual_primary_btn,
            self.manual_retry_btn,
            self.manual_clear_btn,
            self.manual_continue_btn,
            self.manual_cancel_btn,
            self.manual_finish_btn,
        ):
            candidate_row.addWidget(button)
        candidate_row.addStretch()
        manual_layout.addLayout(candidate_row)
        self.modify_task_btn.clicked.connect(self._begin_modify_task)
        self.delete_task_btn.clicked.connect(self._begin_delete_task)
        self.add_task_btn.clicked.connect(self._begin_add_task)
        self.target_class_combo.currentIndexChanged.connect(self._target_class_changed)
        self.manual_smooth_enabled_check.toggled.connect(
            self._manual_smoothing_changed
        )
        self.manual_smooth_iterations_spin.valueChanged.connect(
            self._manual_smoothing_parameters_changed
        )
        self.manual_smooth_offset_spin.valueChanged.connect(
            self._manual_smoothing_parameters_changed
        )
        self.manual_smooth_angle_spin.valueChanged.connect(
            self._manual_smoothing_parameters_changed
        )
        self.manual_primary_btn.clicked.connect(self._manual_primary_action)
        self.manual_retry_btn.clicked.connect(self._manual_retry_action)
        self.manual_clear_btn.clicked.connect(self._manual_clear_action)
        self.manual_continue_btn.clicked.connect(self._continue_manual_task)
        self.manual_cancel_btn.clicked.connect(self._manual_cancel_action)
        self.manual_finish_btn.clicked.connect(self._finish_manual_session)
        root.addWidget(self.manual_group)

        self.qgis_edit_group = QGroupBox("QGIS 原生编辑（高级）")
        qgis_edit_layout = QVBoxLayout(self.qgis_edit_group)
        self.qgis_edit_context_label = QLabel(
            "当前 QGIS 编辑层：未同步 | 撤销/重做按步骤，保存/放弃作用于全部未保存编辑"
        )
        self.qgis_edit_context_label.setWordWrap(True)
        qgis_edit_layout.addWidget(self.qgis_edit_context_label)
        smooth_row = QHBoxLayout()
        self.qgis_smooth_selection_label = QLabel("已选面：0")
        self.qgis_smooth_iterations_spin = QSpinBox()
        self.qgis_smooth_iterations_spin.setRange(1, 3)
        self.qgis_smooth_iterations_spin.setSuffix(" 次")
        self.qgis_smooth_iterations_spin.setValue(smooth_setting(
            "labeling_tool/qgis_smoothing/iterations", 1, int
        ))
        self.qgis_smooth_offset_spin = QDoubleSpinBox()
        self.qgis_smooth_offset_spin.setRange(0.05, 0.45)
        self.qgis_smooth_offset_spin.setSingleStep(0.05)
        self.qgis_smooth_offset_spin.setDecimals(2)
        self.qgis_smooth_offset_spin.setValue(smooth_setting(
            "labeling_tool/qgis_smoothing/offset", 0.25, float
        ))
        self.qgis_smooth_angle_spin = QDoubleSpinBox()
        self.qgis_smooth_angle_spin.setRange(30.0, 180.0)
        self.qgis_smooth_angle_spin.setSingleStep(10.0)
        self.qgis_smooth_angle_spin.setDecimals(0)
        self.qgis_smooth_angle_spin.setSuffix("°")
        self.qgis_smooth_angle_spin.setValue(smooth_setting(
            "labeling_tool/qgis_smoothing/max_angle", 180.0, float
        ))
        self.qgis_smooth_preview_btn = QPushButton("预览光滑效果")
        self.qgis_smooth_apply_btn = QPushButton("应用光滑")
        self.qgis_smooth_clear_btn = QPushButton("取消预览")
        smooth_row.addWidget(self.qgis_smooth_selection_label)
        smooth_row.addWidget(QLabel("次数:"))
        smooth_row.addWidget(self.qgis_smooth_iterations_spin)
        smooth_row.addWidget(QLabel("偏移:"))
        smooth_row.addWidget(self.qgis_smooth_offset_spin)
        smooth_row.addWidget(QLabel("最大角度:"))
        smooth_row.addWidget(self.qgis_smooth_angle_spin)
        smooth_row.addWidget(self.qgis_smooth_preview_btn)
        smooth_row.addWidget(self.qgis_smooth_apply_btn)
        smooth_row.addWidget(self.qgis_smooth_clear_btn)
        smooth_row.addStretch()
        qgis_edit_layout.addLayout(smooth_row)
        self.qgis_smooth_status_label = QLabel(
            "请选择一个或多个面；参数会自动记住，预览不会修改工作层"
        )
        self.qgis_smooth_status_label.setWordWrap(True)
        qgis_edit_layout.addWidget(self.qgis_smooth_status_label)
        self.qgis_smooth_warning_label = QLabel(
            "提示：逐面 Chaikin 光滑不保证相邻面继续共边，请在地图检查后再保存"
        )
        self.qgis_smooth_warning_label.setWordWrap(True)
        qgis_edit_layout.addWidget(self.qgis_smooth_warning_label)
        edit_row = QHBoxLayout()
        self.qgis_undo_btn = QPushButton("撤销一步")
        self.qgis_redo_btn = QPushButton("重做一步")
        self.qgis_save_btn = QPushButton("保存 QGIS 编辑")
        self.qgis_rollback_btn = QPushButton("放弃 QGIS 编辑")
        for button in (
            self.qgis_undo_btn,
            self.qgis_redo_btn,
            self.qgis_save_btn,
            self.qgis_rollback_btn,
        ):
            edit_row.addWidget(button)
        edit_row.addStretch()
        qgis_edit_layout.addLayout(edit_row)
        self.qgis_undo_btn.clicked.connect(self._undo_current_edit)
        self.qgis_redo_btn.clicked.connect(self._redo_current_edit)
        self.qgis_save_btn.clicked.connect(self._save_current_edit)
        self.qgis_rollback_btn.clicked.connect(self._rollback_current_edit)
        self.qgis_smooth_iterations_spin.valueChanged.connect(
            self._qgis_smooth_parameters_changed
        )
        self.qgis_smooth_offset_spin.valueChanged.connect(
            self._qgis_smooth_parameters_changed
        )
        self.qgis_smooth_angle_spin.valueChanged.connect(
            self._qgis_smooth_parameters_changed
        )
        self.qgis_smooth_preview_btn.clicked.connect(self._preview_qgis_smoothing)
        self.qgis_smooth_apply_btn.clicked.connect(self._apply_qgis_smoothing)
        self.qgis_smooth_clear_btn.clicked.connect(self._clear_qgis_smooth_preview)
        self.qgis_edit_group.hide()
        root.addWidget(self.qgis_edit_group)

        self.session_group = QGroupBox("SAM3 会话")
        session_layout = QVBoxLayout(self.session_group)
        self.session_label = QLabel("无活动会话")
        self.session_label.setWordWrap(True)
        session_layout.addWidget(self.session_label)
        self.local_topology_label = QLabel("局部拓扑提示: -")
        self.local_topology_label.setWordWrap(True)
        session_layout.addWidget(self.local_topology_label)
        self.session_error = QPlainTextEdit()
        self.session_error.setReadOnly(True)
        self.session_error.setMaximumHeight(90)
        self.session_error.hide()
        session_layout.addWidget(self.session_error)
        decisions = QHBoxLayout()
        self.keep_btn = QPushButton("保留当前")
        self.adopt_btn = QPushButton("采用 SAM3")
        self.edit_current_btn = QPushButton("编辑当前")
        self.edit_sam_btn = QPushButton("编辑 SAM3")
        self.retry_btn = QPushButton("重试")
        self.cancel_session_btn = QPushButton("取消")
        for button in (
            self.keep_btn, self.adopt_btn, self.edit_current_btn,
            self.edit_sam_btn, self.retry_btn, self.cancel_session_btn,
        ):
            decisions.addWidget(button)
        decisions.addStretch()
        session_layout.addLayout(decisions)
        self.keep_btn.clicked.connect(lambda: self._finish_session("kept_current"))
        self.adopt_btn.clicked.connect(lambda: self._finish_session("adopted"))
        self.edit_current_btn.clicked.connect(lambda: self._finish_session("edit_current"))
        self.edit_sam_btn.clicked.connect(lambda: self._finish_session("edit_sam3"))
        self.retry_btn.clicked.connect(self._retry_session)
        self.cancel_session_btn.clicked.connect(lambda: self._finish_session("cancelled"))
        self.session_group.hide()
        root.addWidget(self.session_group)

        actions = QHBoxLayout()
        self.assemble_btn = QPushButton("组装最终图层")
        self.topology_btn = QPushButton("重新检查拓扑")
        self.accept_btn = QPushButton("写入 accepted_labels")
        self.allow_issues_check = QCheckBox("明确允许带未解决问题入库")
        self.summary_label = QLabel("14 类确认: 0/14    未解决问题: -    未保存编辑: 0")
        actions.addWidget(self.assemble_btn)
        actions.addWidget(self.topology_btn)
        actions.addWidget(self.accept_btn)
        actions.addWidget(self.allow_issues_check)
        actions.addWidget(self.summary_label, stretch=1)
        root.addLayout(actions)
        self.assemble_btn.clicked.connect(self._assemble_final)
        self.topology_btn.clicked.connect(self._check_topology)
        self.accept_btn.clicked.connect(self._write_accepted)
        self.allow_issues_check.toggled.connect(self._update_accept_enabled)
        self.topology_btn.setEnabled(False)
        self.accept_btn.setEnabled(False)
        self._set_decision_state("idle")
        self._update_manual_panel()
        self._update_actions()

    def set_run(self, result, run_spec, sam_config, scripts_dir):
        self._connect_iface_layer_signal()
        self._connect_map_tool_signal()
        self._clear_qgis_smooth_preview()
        self._manual_smoothing_timer.stop()
        self._cancel_manual_capture_transition()
        self._cancel_manual_task(silent=True)
        self._cancel_active_session(record=False)
        self._disconnect_layer_signals()
        self._snapshots.clear()
        self._workspace = None
        self._class_layers = {}
        self._result = dict(result)
        self._run_spec = dict(run_spec)
        self._sam_config = dict(sam_config or {})
        self._scripts_dir = str(scripts_dir)
        self._manual_only = bool(self._run_spec.get("manual_only"))
        self.setWindowTitle(
            "人工分类整理" if self._manual_only else "分类修整与组装"
        )
        self.table.setColumnHidden(5, self._manual_only)
        self.session_group.hide()
        self._confidence_raster = None
        streams = [
            item for item in result.get("ready_streams") or []
            if item.get("status") == "ready"
        ]
        self._eligible_fusions = class_workspace.approved_fusion_streams(
            self._run_spec, streams
        )
        self.fusion_combo.clear()
        for stream in self._eligible_fusions:
            self.fusion_combo.addItem(stream["stream_id"], stream["stream_id"])
        self.initialize_btn.setEnabled(bool(self._eligible_fusions))
        workspace_path = class_workspace.workspace_paths(self._run_spec)["workspace"]
        if workspace_path.is_file():
            try:
                self._workspace = class_workspace.load_workspace(self._run_spec)
                self._load_workspace_layers()
                self.fusion_combo.setCurrentText(self._workspace["baseline_stream_id"])
                self.fusion_combo.setEnabled(False)
                self.initialize_btn.setEnabled(False)
            except Exception as exc:
                self._workspace = None
                QMessageBox.warning(self, "恢复类别工作区失败", str(exc))
        if self._workspace is None:
            self.baseline_label.setText(
                f"Run {run_spec.get('run_id')}；可用 Fusion {len(self._eligible_fusions)} 个"
            )
        if self._manual_only and self._workspace is None and len(self._eligible_fusions) == 1:
            self.fusion_combo.setCurrentIndex(0)
            self._initialize_workspace()
        else:
            self._refresh_table()

    def _initialize_workspace(self):
        stream_id = str(self.fusion_combo.currentData() or "")
        if not stream_id:
            QMessageBox.warning(self, "初始化工作区", "没有通过规则化和审批的 Fusion")
            return
        try:
            stream = class_workspace.stream_by_id(self._eligible_fusions, stream_id)
            self._workspace = class_workspace.initialize_workspace(
                self._run_spec, stream
            )
            self._load_workspace_layers()
            self.fusion_combo.setEnabled(False)
            self.initialize_btn.setEnabled(False)
            self._refresh_table()
        except Exception as exc:
            QMessageBox.warning(self, "初始化 14 类工作层失败", str(exc))

    def _load_workspace_layers(self):
        self._class_layers = self.layer_manager.load_workspace_classes(
            self._run_spec["run_id"], self._workspace
        )
        for code, layer_id in self._class_layers.items():
            layer = QgsProject.instance().mapLayer(layer_id)
            if layer is None:
                continue
            class_workspace.apply_class_constraints(
                layer,
                code,
                run_id=self._run_spec["run_id"],
                baseline_stream_id=self._workspace["baseline_stream_id"],
            )
            if layer.isEditable() and code not in self._snapshots:
                self._snapshots[code] = self._persisted_snapshot(code)
            if layer.id() not in self._connected_layer_ids:
                started_slot = lambda c=code: self._editing_started(c)
                committed_slot = lambda c=code: self._editing_stopped(c)
                stopped_slot = lambda c=code: self._editing_stopped(c)
                layer.editingStarted.connect(started_slot)
                layer.afterCommitChanges.connect(committed_slot)
                layer.editingStopped.connect(stopped_slot)
                self._layer_signal_slots[layer.id()] = (
                    started_slot, committed_slot, stopped_slot
                )
                selection_slot = lambda *_args, c=code: self._selection_changed(c)
                layer.selectionChanged.connect(selection_slot)
                self._selection_signal_slots[layer.id()] = selection_slot
                changed_slots = []
                for signal_name in (
                    "geometryChanged",
                    "featureAdded",
                    "featureDeleted",
                    "attributeValueChanged",
                ):
                    signal = getattr(layer, signal_name, None)
                    if signal is None:
                        continue
                    slot = lambda *_args, c=code: self._layer_edit_changed(c)
                    signal.connect(slot)
                    changed_slots.append((signal_name, slot))
                self._edit_change_signal_slots[layer.id()] = changed_slots
                undo_stack = layer.undoStack()
                undo_slot = lambda *_args, c=code: self._layer_edit_changed(c)
                undo_stack.indexChanged.connect(undo_slot)
                self._undo_stack_signal_slots[layer.id()] = (
                    undo_stack, undo_slot
                )
                self._connected_layer_ids.add(layer.id())
            tree_layer = QgsProject.instance().layerTreeRoot().findLayer(layer.id())
            if tree_layer is not None:
                self._sync_visibility_from_layer_tree(code)
                if layer.id() not in self._tree_visibility_slots:
                    visibility_slot = (
                        lambda _node=None, c=code: self._sync_visibility_from_layer_tree(c)
                    )
                    tree_layer.visibilityChanged.connect(visibility_slot)
                    self._tree_visibility_slots[layer.id()] = (
                        tree_layer, visibility_slot
                    )
        text = (
            f"基准 {self._workspace['baseline_stream_id']} | "
            f"formal SHA {self._workspace['formal_sha256'][:12]}... | "
            f"{self._workspace['feature_count']} 个面"
        )
        report_sha = str(self._workspace.get("boundary_report_sha256") or "")
        if report_sha:
            text += f" | report SHA {report_sha[:12]}..."
        if self._manual_only:
            text += " | 纯人工模式"
        self.baseline_label.setText(text)
        self._active_layer_changed(self.iface.activeLayer())
        self._update_manual_panel()

    def _disconnect_layer_signals(self):
        for layer_id, slots in list(self._layer_signal_slots.items()):
            layer = QgsProject.instance().mapLayer(layer_id)
            if layer is None:
                continue
            for signal, slot in zip(
                (
                    layer.editingStarted,
                    layer.afterCommitChanges,
                    layer.editingStopped,
                ),
                slots,
            ):
                try:
                    signal.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
        self._layer_signal_slots.clear()
        self._connected_layer_ids.clear()
        for layer_id, slot in list(self._selection_signal_slots.items()):
            layer = QgsProject.instance().mapLayer(layer_id)
            if layer is None:
                continue
            try:
                layer.selectionChanged.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._selection_signal_slots.clear()
        for layer_id, slots in list(self._edit_change_signal_slots.items()):
            layer = QgsProject.instance().mapLayer(layer_id)
            if layer is None:
                continue
            for signal_name, slot in slots:
                signal = getattr(layer, signal_name, None)
                if signal is None:
                    continue
                try:
                    signal.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
        self._edit_change_signal_slots.clear()
        for _layer_id, (undo_stack, slot) in list(
            self._undo_stack_signal_slots.items()
        ):
            try:
                undo_stack.indexChanged.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._undo_stack_signal_slots.clear()
        for _layer_id, (tree_layer, slot) in list(self._tree_visibility_slots.items()):
            try:
                tree_layer.visibilityChanged.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._tree_visibility_slots.clear()

    def _connect_iface_layer_signal(self):
        if self._iface_layer_signal_connected:
            return
        signal = getattr(self.iface, "currentLayerChanged", None)
        if signal is None:
            return
        try:
            signal.connect(self._iface_current_layer_slot)
            self._iface_layer_signal_connected = True
        except (TypeError, RuntimeError):
            self._iface_layer_signal_connected = False

    def _disconnect_iface_layer_signal(self):
        if not self._iface_layer_signal_connected:
            return
        signal = getattr(self.iface, "currentLayerChanged", None)
        if signal is not None:
            try:
                signal.disconnect(self._iface_current_layer_slot)
            except (TypeError, RuntimeError):
                pass
        self._iface_layer_signal_connected = False

    def _connect_map_tool_signal(self):
        if self._map_tool_signal_connected:
            return
        signal = getattr(self.iface.mapCanvas(), "mapToolSet", None)
        if signal is None:
            return
        try:
            signal.connect(self._map_tool_slot)
            self._map_tool_signal_connected = True
        except (TypeError, RuntimeError):
            self._map_tool_signal_connected = False

    def _disconnect_map_tool_signal(self):
        if not self._map_tool_signal_connected:
            return
        signal = getattr(self.iface.mapCanvas(), "mapToolSet", None)
        if signal is not None:
            try:
                signal.disconnect(self._map_tool_slot)
            except (TypeError, RuntimeError):
                pass
        self._map_tool_signal_connected = False

    def _layer(self, class_code):
        layer = QgsProject.instance().mapLayer(self._class_layers.get(int(class_code), ""))
        if layer is None:
            raise RuntimeError(f"class {class_code} working layer is not loaded")
        return layer

    def _refresh_class_display(self, *class_codes):
        for class_code in class_codes:
            layer = self._layer(class_code)
            StyleManager.apply_categorized_style(layer)
            layer.removeSelection()
            layer.triggerRepaint()
        canvas = self.iface.mapCanvas()
        canvas.clearCache()
        canvas.refresh()

    def _set_visible(self, class_code, visible):
        layer_id = self._class_layers.get(class_code)
        if layer_id:
            self.layer_manager.set_layer_visibility(layer_id, visible)

    def _sync_visibility_from_layer_tree(self, class_code):
        layer_id = self._class_layers.get(int(class_code))
        checkbox = self._visibility_checks.get(int(class_code))
        if not layer_id or checkbox is None:
            return
        tree_layer = QgsProject.instance().layerTreeRoot().findLayer(layer_id)
        if tree_layer is None:
            return
        checkbox.blockSignals(True)
        checkbox.setChecked(bool(tree_layer.itemVisibilityChecked()))
        checkbox.blockSignals(False)

    def _class_code_for_layer(self, layer):
        if layer is None:
            return None
        try:
            layer_id = layer.id()
        except RuntimeError:
            return None
        for class_code, candidate_id in self._class_layers.items():
            if candidate_id == layer_id:
                return int(class_code)
        return None

    def _select_class_context(self, class_code, activate_layer=True):
        class_code = int(class_code)
        if class_code not in self._class_layers:
            return
        locked_code = self._manual_locked_class_code()
        if locked_code is not None and class_code != locked_code:
            self.manual_instruction_label.setText(
                f"当前人工任务锁定 {locked_code} {CLASS_NAMES[locked_code]}；"
                "请先完成或取消任务"
            )
            class_code = locked_code
        row = CLASS_ORDER.index(class_code)
        self._syncing_class_selection = True
        try:
            self.table.setCurrentCell(row, 1)
            self.table.selectRow(row)
            self.table.scrollToItem(
                self.table.item(row, 1),
                ENSURE_VISIBLE,
            )
            if activate_layer:
                layer = self._layer(class_code)
                active_layer = self.iface.activeLayer()
                if active_layer is None or active_layer.id() != layer.id():
                    self.iface.setActiveLayer(layer)
        finally:
            self._syncing_class_selection = False
        if self._manual_task is None:
            self._set_target_combo_code(class_code)
        self._update_manual_panel()

    def _activate_row(self, row, _column, *_previous):
        if self._syncing_class_selection:
            return
        if 0 <= row < len(CLASS_ORDER):
            self._select_class_context(CLASS_ORDER[row], activate_layer=True)

    def _active_layer_changed(self, layer):
        if self._syncing_class_selection:
            return
        preview = self._qgis_smooth_preview
        if preview and (layer is None or layer.id() != preview["layer_id"]):
            self._clear_qgis_smooth_preview(
                status_text="活动层已变化，请重新选择面并预览"
            )
        class_code = self._class_code_for_layer(layer)
        locked_code = self._manual_locked_class_code()
        if locked_code is not None and class_code != locked_code:
            self._select_class_context(locked_code, activate_layer=True)
            return
        if class_code is not None:
            self._select_class_context(class_code, activate_layer=False)
            return
        self._syncing_class_selection = True
        try:
            self.table.clearSelection()
            self.table.setCurrentCell(-1, -1)
        finally:
            self._syncing_class_selection = False
        self._update_manual_panel()

    def _current_class_code(self):
        return self._class_code_for_layer(self.iface.activeLayer())

    def _manual_locked_class_code(self):
        task = self._manual_task
        if not task:
            return None
        return int(task.get("class_code"))

    def _set_target_combo_code(self, class_code):
        index = self.target_class_combo.findData(int(class_code))
        if index < 0:
            return
        self.target_class_combo.blockSignals(True)
        self.target_class_combo.setCurrentIndex(index)
        self.target_class_combo.blockSignals(False)

    def _manual_smooth_parameters(self):
        return (
            int(self.manual_smooth_iterations_spin.value()),
            float(self.manual_smooth_offset_spin.value()),
            float(self.manual_smooth_angle_spin.value()),
        )

    @staticmethod
    def _store_smoothing_parameters(parameters):
        iterations, offset, max_angle = parameters
        settings = QgsSettings()
        settings.setValue("labeling_tool/qgis_smoothing/iterations", iterations)
        settings.setValue("labeling_tool/qgis_smoothing/offset", offset)
        settings.setValue("labeling_tool/qgis_smoothing/max_angle", max_angle)

    def _sync_smoothing_parameter_widgets(self, parameters, source):
        if self._smoothing_parameter_sync:
            return
        iterations, offset, max_angle = parameters
        self._smoothing_parameter_sync = True
        try:
            groups = []
            if source != "manual":
                groups.append((
                    self.manual_smooth_iterations_spin,
                    self.manual_smooth_offset_spin,
                    self.manual_smooth_angle_spin,
                ))
            if source != "qgis":
                groups.append((
                    self.qgis_smooth_iterations_spin,
                    self.qgis_smooth_offset_spin,
                    self.qgis_smooth_angle_spin,
                ))
            for iteration_spin, offset_spin, angle_spin in groups:
                iteration_spin.setValue(iterations)
                offset_spin.setValue(offset)
                angle_spin.setValue(max_angle)
        finally:
            self._smoothing_parameter_sync = False

    def _reset_manual_smoothing_for_new_task(self):
        self._manual_smoothing_timer.stop()
        self.manual_smooth_enabled_check.blockSignals(True)
        self.manual_smooth_enabled_check.setChecked(False)
        self.manual_smooth_enabled_check.blockSignals(False)
        self.manual_smooth_status_label.setText(
            "光滑默认关闭；开启后参数变化会自动预览本批全部新边界"
        )

    def _manual_smoothing_changed(self, checked):
        task = self._manual_task
        if not task or task.get("kind") not in ("add", "modify"):
            return
        task["smoothing_enabled"] = bool(checked)
        task["smoothing_preview"] = None
        task["smoothing_error"] = ""
        self._manual_smoothing_timer.stop()
        if checked:
            self._schedule_manual_smoothing_preview()
        else:
            self.manual_smooth_status_label.setText(
                "光滑已关闭；地图和保存均使用原始绘制边界"
            )
            self._refresh_manual_pending_candidate_bands()
            self._update_manual_panel()

    def _manual_smoothing_parameters_changed(self, *_args):
        if self._smoothing_parameter_sync:
            return
        parameters = self._manual_smooth_parameters()
        self._store_smoothing_parameters(parameters)
        self._sync_smoothing_parameter_widgets(parameters, "manual")
        task = self._manual_task
        if (
            task and task.get("kind") in ("add", "modify")
            and task.get("smoothing_enabled")
        ):
            self._schedule_manual_smoothing_preview()

    def _schedule_manual_smoothing_preview(self):
        self._manual_smoothing_timer.stop()
        task = self._manual_task
        if not task or task.get("kind") not in ("add", "modify"):
            return
        task["smoothing_preview"] = None
        task["smoothing_error"] = ""
        self._refresh_manual_pending_candidate_bands()
        pending_count = len(task.get("pending_geometries") or [])
        if not task.get("smoothing_enabled"):
            self.manual_smooth_status_label.setText(
                "光滑已关闭；地图和保存均使用原始绘制边界"
            )
        elif not pending_count:
            self.manual_smooth_status_label.setText(
                "请先绘制新边界；只修改类别时不会光滑旧面"
            )
        elif any(task.get("pending_errors") or []):
            self.manual_smooth_status_label.setText(
                "本批存在不可保存的原始候选；请先重新绘制当前面"
            )
        else:
            self.manual_smooth_status_label.setText(
                f"正在自动更新 {pending_count} 个待保存面的光滑预览…"
            )
            self._manual_smoothing_timer.start()
        self._update_manual_panel()

    def _manual_smoothing_preview_is_current(self, task=None):
        task = task or self._manual_task
        if (
            not task or task.get("kind") not in ("add", "modify")
            or not task.get("smoothing_enabled")
        ):
            return False
        preview = task.get("smoothing_preview")
        geometries = task.get("pending_geometries") or []
        if (
            not preview
            or preview.get("parameters") != self._manual_smooth_parameters()
            or len(preview.get("geometries") or []) != len(geometries)
        ):
            return False
        source_hashes = tuple(
            class_workspace.geometry_hash(geometry) for geometry in geometries
        )
        return source_hashes == preview.get("source_hashes")

    def _refresh_manual_smoothing_preview(self):
        task = self._manual_task
        if (
            not task or task.get("kind") not in ("add", "modify")
            or not task.get("smoothing_enabled")
        ):
            return
        source_geometries = [
            QgsGeometry(geometry)
            for geometry in (task.get("pending_geometries") or [])
        ]
        if not source_geometries:
            self._schedule_manual_smoothing_preview()
            return
        iterations, offset, max_angle = self._manual_smooth_parameters()
        smoothed_geometries = []
        old_vertices = 0
        new_vertices = 0
        old_area = 0.0
        new_area = 0.0
        for index, source in enumerate(source_geometries, start=1):
            try:
                smoothed = source.smooth(iterations, offset, -1.0, max_angle)
            except Exception as exc:
                task["smoothing_preview"] = None
                task["smoothing_error"] = f"第 {index} 个面：{exc}"
                self.manual_smooth_status_label.setText(
                    f"自动预览失败：第 {index} 个面无法光滑：{exc}；"
                    "请调整参数或取消光滑后保存原始边界"
                )
                self._refresh_manual_pending_candidate_bands()
                self._update_manual_panel()
                return
            smoothed.convertToMultiType()
            error = self._manual_geometry_error(smoothed)
            if error:
                task["smoothing_preview"] = None
                task["smoothing_error"] = f"第 {index} 个面：{error}"
                self.manual_smooth_status_label.setText(
                    f"自动预览失败：第 {index} 个面的光滑结果{error}；"
                    "请调整参数或取消光滑后保存原始边界"
                )
                self._refresh_manual_pending_candidate_bands()
                self._update_manual_panel()
                return
            smoothed_geometries.append(smoothed)
            old_vertices += self._geometry_vertex_count(source)
            new_vertices += self._geometry_vertex_count(smoothed)
            old_area += float(source.area())
            new_area += float(smoothed.area())
        task["smoothing_error"] = ""
        task["smoothing_preview"] = {
            "parameters": (iterations, offset, max_angle),
            "source_hashes": tuple(
                class_workspace.geometry_hash(geometry)
                for geometry in source_geometries
            ),
            "geometries": smoothed_geometries,
        }
        area_change = (
            ((new_area - old_area) / old_area) * 100.0 if old_area > 0.0 else 0.0
        )
        self.manual_smooth_status_label.setText(
            f"已自动预览 {len(smoothed_geometries)} 个面：顶点 "
            f"{old_vertices} → {new_vertices}，总面积变化 {area_change:+.3f}%；"
            "保存将写入当前预览；逐面光滑不保证相邻面继续共边"
        )
        self._refresh_manual_pending_candidate_bands()
        self._update_manual_panel()

    def _manual_geometries_for_commit(self, task):
        geometries = [
            QgsGeometry(geometry)
            for geometry in (task.get("pending_geometries") or [])
        ]
        if not task.get("smoothing_enabled"):
            return geometries
        if not self._manual_smoothing_preview_is_current(task):
            raise RuntimeError("光滑预览尚未完成或已经失效，请等待自动预览")
        return [
            QgsGeometry(geometry)
            for geometry in task["smoothing_preview"]["geometries"]
        ]

    def _open_manual_operations(self, class_code):
        self._select_class_context(class_code, activate_layer=True)
        self.manual_group.setFocus()
        self.manual_instruction_label.setText(
            f"已选择 {class_code} {CLASS_NAMES[class_code]}，请选择修改、删除或新增任务"
        )
        self._update_manual_panel()

    def _selection_changed(self, class_code):
        task = self._manual_task
        if task and task.get("kind") == "delete" and int(task["class_code"]) == int(class_code):
            task["selected_count"] = self._layer(class_code).selectedFeatureCount()
        preview = self._qgis_smooth_preview
        if preview and int(preview["class_code"]) == int(class_code):
            selected_ids = tuple(sorted(self._layer(class_code).selectedFeatureIds()))
            if selected_ids != preview["feature_ids"]:
                self._clear_qgis_smooth_preview(
                    status_text="选择已变化，请重新预览光滑效果"
                )
        self._update_manual_panel()

    def _layer_edit_changed(self, class_code):
        preview = self._qgis_smooth_preview
        if preview and int(preview["class_code"]) == int(class_code):
            self._clear_qgis_smooth_preview(
                status_text="来源 geometry 已变化，请重新预览光滑效果"
            )
        self._update_manual_panel()
        self._update_actions()

    def _target_class_changed(self, _index):
        target_code = self.target_class_combo.currentData()
        if target_code is None:
            return
        target_code = int(target_code)
        task = self._manual_task
        if task:
            if task.get("kind") == "add":
                task["target_code"] = target_code
                task.pop("error", None)
                self._refresh_manual_pending_candidate_bands()
                pending_count = len(task.get("pending_geometries") or [])
                self.manual_instruction_label.setText(
                    f"本批待保存 {pending_count} 个面；目标类别为 "
                    f"{target_code} {CLASS_NAMES[target_code]}"
                )
                self._update_manual_panel()
                return
            if task.get("kind") == "modify":
                task["target_code"] = target_code
                task.pop("error", None)
                self._refresh_manual_pending_candidate_bands()
                self._update_manual_panel()
                return
        if target_code in self._class_layers:
            self._select_class_context(target_code, activate_layer=True)

    def _manual_task_guard(self, title):
        class_code = self._current_class_code()
        if not self._workspace or class_code is None:
            QMessageBox.information(self, title, "请先在表格或 QGIS 图层面板选择一个类别工作层")
            return None
        if self._active_session or self._manual_task:
            QMessageBox.warning(self, title, "请先完成或取消当前人工/SAM3 任务")
            return None
        modified = self._editable_modified_layers()
        if modified:
            names = "、".join(str(code) for code in modified)
            QMessageBox.warning(self, title, f"请先保存或取消类别 {names} 的未保存编辑")
            return None
        return int(class_code)

    def _begin_modify_task(self):
        class_code = self._manual_task_guard("修改现有面")
        if class_code is None:
            return
        layer = self._layer(class_code)
        if layer.featureCount() == 0:
            QMessageBox.information(self, "修改现有面", "当前类别没有可修改的面")
            return
        self._reset_manual_smoothing_for_new_task()
        self._manual_task = {
            "kind": "modify",
            "class_code": class_code,
            "target_code": class_code,
            "state": "selecting",
            "selection_before": list(layer.selectedFeatureIds()),
            "selected_feature_ids": list(layer.selectedFeatureIds()),
            "pending_geometries": [],
            "pending_errors": [],
            "smoothing_enabled": False,
            "smoothing_preview": None,
            "smoothing_error": "",
            "submitted_batch_count": 0,
            "modified_old_count": 0,
            "saved_new_count": 0,
            "deleted_old_count": 0,
            "editing_started_by_task": False,
        }
        self._manual_previous_map_tool = self.iface.mapCanvas().mapTool()
        self._set_target_combo_code(class_code)
        layer.removeSelection()
        self._refresh_manual_modify_reference()
        self._start_manual_picker()
        self.manual_instruction_label.setText(
            f"已选择 {len(self._manual_task['selected_feature_ids'])} 个旧面；"
            "在地图中继续点击可加入或移出，选好后可直接改类或绘制新边界"
        )
        self._update_manual_panel()

    def _toggle_manual_modify_feature(self, feature):
        task = self._manual_task
        if not task or task.get("kind") != "modify":
            return
        geometry = QgsGeometry(feature.geometry())
        error = self._manual_geometry_error(geometry)
        if error:
            QMessageBox.warning(self, "修改现有面", f"选中面不可修改：{error}")
            return
        selected_ids = task.setdefault("selected_feature_ids", [])
        feature_id = feature.id()
        if feature_id in selected_ids:
            selected_ids.remove(feature_id)
        else:
            selected_ids.append(feature_id)
        layer = self._layer(task["class_code"])
        layer.removeSelection()
        self._refresh_manual_modify_reference()
        self.manual_instruction_label.setText(
            f"已选择 {len(selected_ids)} 个旧面；地图点击可继续加入或移出，"
            "不画新边界时保存将只修改类别"
        )
        self._update_manual_panel()

    def _begin_delete_task(self):
        class_code = self._manual_task_guard("删除现有面")
        if class_code is None:
            return
        layer = self._layer(class_code)
        if layer.featureCount() == 0:
            QMessageBox.information(self, "删除现有面", "当前类别没有可删除的面")
            return
        selected_before = list(layer.selectedFeatureIds())
        self._manual_task = {
            "kind": "delete",
            "class_code": class_code,
            "state": "selecting",
            "selection_before": selected_before,
            "selected_count": len(selected_before),
        }
        self._manual_previous_map_tool = self.iface.mapCanvas().mapTool()
        self._start_manual_picker()
        self.manual_instruction_label.setText(
            "在地图中逐个点击面可加入或移出选择；不需要按 Shift"
        )
        self._update_manual_panel()

    def _begin_add_task(self):
        class_code = self._manual_task_guard("新增面")
        if class_code is None:
            return
        self._select_class_context(class_code, activate_layer=True)
        self._visibility_checks[class_code].setChecked(True)
        self._reset_manual_smoothing_for_new_task()
        self._manual_task = {
            "kind": "add",
            "class_code": class_code,
            "target_code": class_code,
            "state": "capturing",
            "added_count": 0,
            "submitted_batch_count": 0,
            "saved_counts": {},
            "pending_geometries": [],
            "pending_errors": [],
            "smoothing_enabled": False,
            "smoothing_preview": None,
            "smoothing_error": "",
            "editing_started_by_task": False,
            "selection_before": list(self._layer(class_code).selectedFeatureIds()),
        }
        self._manual_previous_map_tool = self.iface.mapCanvas().mapTool()
        self._set_target_combo_code(class_code)
        self._start_manual_capture()

    def _start_manual_picker(self):
        task = self._manual_task
        if not task:
            return
        self._disconnect_manual_picker(restore=False)
        canvas = self.iface.mapCanvas()
        tool = QgsMapToolEmitPoint(canvas)
        tool.canvasClicked.connect(self._manual_task_map_clicked)
        self._manual_task_tool = tool
        task["expected_map_tool"] = tool
        self._set_manual_map_tool(tool)

    def _manual_task_map_clicked(self, map_point, _button):
        task = self._manual_task
        if not task or task.get("state") != "selecting":
            return
        layer = self._layer(task["class_code"])
        hits = self._features_at_map_point(layer, map_point)
        if len(hits) != 1:
            QMessageBox.information(
                self,
                "选择面",
                "点击位置必须唯一命中当前类别中的一个面，请放大地图后重新点击",
            )
            return
        feature = hits[0]
        if task.get("kind") == "delete":
            ids = set(layer.selectedFeatureIds())
            if feature.id() in ids:
                ids.remove(feature.id())
            else:
                ids.add(feature.id())
            layer.selectByIds(sorted(ids))
            task["selected_count"] = len(ids)
            self._update_manual_panel()
            return
        self._toggle_manual_modify_feature(feature)

    def _disconnect_manual_picker(self, restore=False):
        tool = self._manual_task_tool
        if tool is not None:
            try:
                tool.canvasClicked.disconnect(self._manual_task_map_clicked)
            except (TypeError, RuntimeError):
                pass
            self._manual_task_tool = None
        if restore:
            self._restore_manual_map_tool()

    def _set_manual_map_tool(self, tool):
        self._setting_manual_map_tool = True
        try:
            self.iface.mapCanvas().setMapTool(tool)
        finally:
            self._setting_manual_map_tool = False

    def _restore_manual_map_tool(self):
        canvas = self.iface.mapCanvas()
        previous = self._manual_previous_map_tool
        if previous is None or canvas.mapTool() == previous:
            return
        self._setting_manual_map_tool = True
        try:
            canvas.setMapTool(previous)
        finally:
            self._setting_manual_map_tool = False

    def _map_tool_changed(self, *_args):
        if self._setting_manual_map_tool or not self._manual_task:
            return
        task = self._manual_task
        expected = task.get("expected_map_tool")
        if expected is None or self.iface.mapCanvas().mapTool() == expected:
            return
        if task.get("state") not in ("selecting", "capturing"):
            return
        task["resume_state"] = task["state"]
        task["state"] = "paused"
        self.manual_instruction_label.setText(
            "QGIS 地图工具已切换；点击“继续任务”恢复，或取消当前任务"
        )
        self._update_manual_panel()

    def _continue_manual_task(self):
        task = self._manual_task
        if not task or task.get("state") != "paused":
            return
        resume = task.pop("resume_state", "selecting")
        if resume == "selecting":
            task["state"] = "selecting"
            self._start_manual_picker()
        elif resume == "capturing":
            self._start_manual_capture()
        self._update_manual_panel()

    def _start_manual_capture(self):
        task = self._manual_task
        if not task:
            return
        self._disconnect_manual_capture(restore=False)
        if task.get("kind") == "modify":
            if not task.get("selected_feature_ids"):
                QMessageBox.information(self, "修改现有面", "请先选择一个或多个旧面")
                return
            self._disconnect_manual_picker(restore=False)
        task["state"] = "capturing"
        class_code = int(task["class_code"])
        layer = self._layer(class_code)
        self.iface.setActiveLayer(layer)
        if not layer.isEditable():
            self._metadata_update = True
            try:
                editing_started = layer.startEditing()
            finally:
                self._metadata_update = False
            if not editing_started:
                task["state"] = "failed"
                task["error"] = "无法开启目标类别层编辑模式（Toggle Editing）"
                self.manual_instruction_label.setText(task["error"])
                self._update_manual_panel()
                return
            task["editing_started_by_task"] = True
        canvas = self.iface.mapCanvas()
        tool = QgsMapToolDigitizeFeature(
            canvas,
            self.iface.cadDockWidget(),
            QgsMapToolCapture.CaptureMode.CapturePolygon,
        )
        tool.setLayer(layer)
        tool.setCheckGeometryType(True)
        if not tool.supportsTechnique(Qgis.CaptureTechnique.PolyBezier):
            task["state"] = "failed"
            task["error"] = "当前 QGIS 不支持 PolyBezier 捕获"
            self._update_manual_panel()
            return
        tool.setCurrentCaptureTechnique(Qgis.CaptureTechnique.PolyBezier)
        tool.digitizingCompleted.connect(self._manual_capture_completed)
        tool.digitizingCanceled.connect(self._manual_capture_cancelled)
        self._manual_capture_tool = tool
        task["expected_map_tool"] = tool
        pending_count = len(task.get("pending_geometries") or [])
        if task.get("kind") == "modify":
            selected_count = len(task.get("selected_feature_ids") or [])
            self.manual_instruction_label.setText(
                f"已选 {selected_count} 个灰色旧面，本批已有 {pending_count} 个新边界；"
                "继续使用 Bézier 绘制完整面并右键结束"
            )
        else:
            self.manual_instruction_label.setText(
                f"本批待保存 {pending_count} 个面；继续绘制完整面并右键结束，"
                "或回到窗口选择类别后保存"
            )
        self._set_manual_map_tool(tool)
        task["state"] = "capturing"
        task.pop("resume_state", None)
        self._update_manual_panel()

    def _manual_capture_completed(self, feature):
        task = self._manual_task
        if not task or task.get("state") != "capturing":
            return
        geometry = QgsGeometry(feature.geometry())
        if geometry.requiresConversionToStraightSegments():
            geometry.convertToStraightSegment()
        geometry.convertToMultiType()
        error = self._manual_geometry_error(geometry)
        task.setdefault("pending_geometries", []).append(geometry)
        task.setdefault("pending_errors", []).append(error)
        pending_count = len(task["pending_geometries"])
        if task.get("kind") == "modify":
            selected_count = len(task.get("selected_feature_ids") or [])
            expected_deleted = 0
            expected_added = 0
            if not error:
                plan = self._manual_modify_overlap_plan(
                    self._manual_modify_selected_features(task),
                    task["pending_geometries"],
                )
                expected_deleted = len(plan["unmatched_old"])
                expected_added = len(plan["unmatched_new"])
            self.manual_instruction_label.setText(
                f"第 {pending_count} 个新边界不可保存：{error}" if error else
                f"本批已选 {selected_count} 个旧面、绘制 {pending_count} 个新边界；"
                f"预计删除旧面 {expected_deleted} 个、新增 {expected_added} 个；"
                "可继续绘制或保存"
            )
        else:
            self.manual_instruction_label.setText(
                f"第 {pending_count} 个候选不可保存：{error}" if error else
                f"本批待保存 {pending_count} 个面；可继续绘制，或选择目标类别后保存"
            )
        if error:
            task["state"] = "candidate"
            self._schedule_manual_capture_transition("restore")
        else:
            task["state"] = "capture_transition"
            self._schedule_manual_capture_transition("restart")
        if task.get("smoothing_enabled"):
            self._schedule_manual_smoothing_preview()
        else:
            self._refresh_manual_pending_candidate_bands()
        self._update_manual_panel()

    def _manual_capture_cancelled(self):
        task = self._manual_task
        if not task:
            return
        task["state"] = "capture_cancelled"
        pending_count = len(task.get("pending_geometries") or [])
        self.manual_instruction_label.setText(
            f"本次绘制已停止；本批仍有 {pending_count} 个待保存面"
            if pending_count else
            "本次绘制已停止；可以重新绘制或结束任务"
        )
        self._schedule_manual_capture_transition("restore")
        self._update_manual_panel()

    def _schedule_manual_capture_transition(self, action):
        task = self._manual_task
        if not task or action not in ("restart", "restore"):
            return
        self._manual_capture_transition_action = action
        self._manual_capture_transition_task = task
        self._manual_capture_transition_timer.start(0)

    def _cancel_manual_capture_transition(self):
        self._manual_capture_transition_timer.stop()
        self._manual_capture_transition_action = None
        self._manual_capture_transition_task = None

    def _run_manual_capture_transition(self):
        action = self._manual_capture_transition_action
        task = self._manual_capture_transition_task
        self._manual_capture_transition_action = None
        self._manual_capture_transition_task = None
        if action not in ("restart", "restore"):
            return
        self._disconnect_manual_capture(restore=action == "restore")
        if action == "restart" and self._manual_task is task:
            self._start_manual_capture()

    def _retire_manual_capture_tool(self, tool):
        if tool is None:
            return
        self._manual_retired_capture_tools.append(tool)
        self._manual_capture_retire_timer.start(0)

    def _release_retired_manual_capture_tools(self):
        self._manual_retired_capture_tools = []

    def _disconnect_manual_capture(self, restore=True):
        self._cancel_manual_capture_transition()
        tool = self._manual_capture_tool
        if tool is not None:
            for signal, slot in (
                (tool.digitizingCompleted, self._manual_capture_completed),
                (tool.digitizingCanceled, self._manual_capture_cancelled),
            ):
                try:
                    signal.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
            try:
                tool.stopCapturing()
            except RuntimeError:
                pass
            self._manual_capture_tool = None
            self._retire_manual_capture_tool(tool)
        if restore:
            self._restore_manual_map_tool()

    def _manual_modify_selected_features(self, task=None):
        task = task or self._manual_task
        if not task or task.get("kind") != "modify":
            return []
        layer = self._layer(task["class_code"])
        by_id = {feature.id(): QgsFeature(feature) for feature in layer.getFeatures()}
        return [
            by_id[feature_id]
            for feature_id in task.get("selected_feature_ids", [])
            if feature_id in by_id
        ]

    def _refresh_manual_modify_reference(self):
        self._clear_manual_reference_band()
        task = self._manual_task
        if not task or task.get("kind") != "modify":
            return
        layer = self._layer(task["class_code"])
        features = self._manual_modify_selected_features(task)
        if not features:
            return
        band = QgsRubberBand(self.iface.mapCanvas(), Qgis.GeometryType.Polygon)
        band.setStrokeColor(QColor(105, 105, 105, 230))
        band.setFillColor(QColor(105, 105, 105, 55))
        band.setLineStyle(DASH_LINE)
        band.setWidth(2)
        for feature in features:
            band.addGeometry(feature.geometry(), layer)
        self._manual_reference_band = band

    def _show_manual_candidate(self, geometry, class_code, error=""):
        self._clear_manual_candidate_band()
        self._manual_candidate_band = self._new_manual_candidate_band(
            geometry, class_code, error
        )

    def _new_manual_candidate_band(
        self, geometry, class_code, error="", smooth_preview=False
    ):
        color = QColor(
            "#d7191c" if error else
            "#00bcd4" if smooth_preview else
            StyleManager.get_class_color(class_code)
        )
        fill = QColor(color)
        fill.setAlpha(40 if smooth_preview else 55)
        band = QgsRubberBand(self.iface.mapCanvas(), Qgis.GeometryType.Polygon)
        band.setStrokeColor(color)
        band.setFillColor(fill)
        if smooth_preview:
            band.setLineStyle(DASH_LINE)
        band.setWidth(2)
        band.setToGeometry(geometry, self._layer(class_code))
        return band

    def _refresh_manual_pending_candidate_bands(self):
        self._clear_manual_add_candidate_bands()
        task = self._manual_task
        if not task or task.get("kind") not in ("add", "modify"):
            return
        target_code = int(task.get("target_code", task["class_code"]))
        smooth_preview = self._manual_smoothing_preview_is_current(task)
        geometries = (
            task["smoothing_preview"]["geometries"]
            if smooth_preview else
            task.get("pending_geometries") or []
        )
        errors = task.get("pending_errors") or []
        for index, geometry in enumerate(geometries):
            error = errors[index] if index < len(errors) else ""
            self._manual_add_candidate_bands.append(
                self._new_manual_candidate_band(
                    geometry, target_code, error, smooth_preview=smooth_preview
                )
            )

    def _clear_manual_candidate_band(self):
        band = self._manual_candidate_band
        if band is not None:
            band.reset(Qgis.GeometryType.Polygon)
            self.iface.mapCanvas().scene().removeItem(band)
        self._manual_candidate_band = None

    def _clear_manual_add_candidate_bands(self):
        for band in self._manual_add_candidate_bands:
            band.reset(Qgis.GeometryType.Polygon)
            self.iface.mapCanvas().scene().removeItem(band)
        self._manual_add_candidate_bands = []

    def _clear_manual_reference_band(self):
        band = self._manual_reference_band
        if band is not None:
            band.reset(Qgis.GeometryType.Polygon)
            self.iface.mapCanvas().scene().removeItem(band)
        self._manual_reference_band = None

    def _clear_manual_bands(self):
        self._clear_manual_candidate_band()
        self._clear_manual_add_candidate_bands()
        self._clear_manual_reference_band()

    def _manual_primary_action(self):
        task = self._manual_task
        if not task:
            return
        kind = task.get("kind")
        state = task.get("state")
        if kind == "modify" and state not in ("committing", "paused"):
            self._commit_manual_modify_batch()
        elif kind == "delete" and state == "selecting":
            self._commit_manual_delete()
        elif (
            kind == "add"
            and state not in ("committing", "paused")
            and task.get("pending_geometries")
        ):
            self._commit_manual_add()

    def _manual_retry_action(self):
        task = self._manual_task
        if not task:
            return
        if task.get("kind") in ("modify", "add"):
            geometries = task.setdefault("pending_geometries", [])
            errors = task.setdefault("pending_errors", [])
            if geometries:
                geometries.pop()
            if errors:
                errors.pop()
            task.pop("error", None)
            if task.get("smoothing_enabled"):
                self._schedule_manual_smoothing_preview()
            else:
                self._refresh_manual_pending_candidate_bands()
            self._start_manual_capture()

    def _manual_clear_action(self):
        task = self._manual_task
        if not task or task.get("kind") != "delete":
            return
        self._layer(task["class_code"]).removeSelection()
        self._update_manual_panel()

    def _manual_cancel_action(self):
        self._cancel_manual_task()

    @staticmethod
    def _manual_modify_overlap_plan(old_features, new_geometries):
        overlaps = []
        for old_index, feature in enumerate(old_features):
            old_geometry = feature.geometry()
            for new_index, geometry in enumerate(new_geometries):
                if not old_geometry.boundingBox().intersects(geometry.boundingBox()):
                    continue
                intersection = old_geometry.intersection(geometry)
                area = float(intersection.area()) if not intersection.isEmpty() else 0.0
                if not math.isfinite(area) or area <= 0.0:
                    continue
                overlaps.append((area, old_index, new_index))
        matched_old = set()
        matched_new = set()
        matches = []
        for area, old_index, new_index in sorted(
            overlaps, key=lambda item: (-item[0], item[1], item[2])
        ):
            if old_index in matched_old or new_index in matched_new:
                continue
            matched_old.add(old_index)
            matched_new.add(new_index)
            matches.append((old_index, new_index, area))
        return {
            "matches": matches,
            "unmatched_old": [
                index for index in range(len(old_features)) if index not in matched_old
            ],
            "unmatched_new": [
                index for index in range(len(new_geometries)) if index not in matched_new
            ],
        }

    def _manual_replacement_feature(
        self, layer, geometry, target_code, original=None,
        recalculate_confidence=True,
    ):
        feature = QgsFeature(layer.fields())
        feature.setGeometry(geometry)
        confidence_warning = ""
        if original is None:
            object_id = class_workspace.new_object_id(self._run_spec)
            values = {
                "run_id": self._run_spec["run_id"],
                "object_id": object_id,
                "part_id": "000",
                "class_code": target_code,
                "class_name": CLASS_NAMES[target_code],
                "baseline_stream_id": self._workspace["baseline_stream_id"],
                "geometry_source": "manual_edited",
                "geometry_revision": 1,
                "edit_base": "",
                "reviewed": 0,
            }
        else:
            for field in layer.fields():
                source_index = original.fieldNameIndex(field.name())
                if source_index >= 0:
                    feature.setAttribute(field.name(), original.attribute(source_index))
            object_id = str(original.attribute("object_id") or "")
            if not object_id:
                raise RuntimeError("待修改旧面缺少 object_id")
            previous_source = str(original.attribute("geometry_source") or "fusion")
            values = {
                "run_id": self._run_spec["run_id"],
                "object_id": object_id,
                "part_id": str(original.attribute("part_id") or "000"),
                "class_code": target_code,
                "class_name": CLASS_NAMES[target_code],
                "baseline_stream_id": self._workspace["baseline_stream_id"],
                "geometry_source": "manual_edited",
                "geometry_revision": int(original.attribute("geometry_revision") or 0) + 1,
                "edit_base": previous_source,
                "reviewed": 0,
            }
        if original is not None and not recalculate_confidence:
            confidence_mean = original.attribute("confidence_mean")
            confidence_std = original.attribute("confidence_std")
        else:
            confidence_mean, confidence_std, confidence_warning = (
                self._optional_confidence_statistics(layer, geometry)
            )
        values.update({
            "confidence_mean": confidence_mean,
            "confidence_std": confidence_std,
            "updated_at": class_workspace._now(),
        })
        for name, value in values.items():
            if layer.fields().indexOf(name) >= 0:
                feature.setAttribute(name, value)
        return feature, object_id, confidence_warning

    def _remove_transferred_features(self, layer, object_ids):
        wanted = set(str(object_id) for object_id in object_ids)
        feature_ids = [
            feature.id() for feature in layer.getFeatures()
            if str(feature.attribute("object_id") or "") in wanted
        ]
        if len(feature_ids) != len(wanted):
            raise RuntimeError("目标类别补偿回滚无法找到全部已写入对象")
        if not layer.isEditable() and not layer.startEditing():
            raise RuntimeError("无法启动目标类别补偿回滚")
        if not layer.deleteFeatures(feature_ids):
            layer.rollBack()
            raise RuntimeError("无法删除目标类别中的批次补偿对象")
        if not layer.commitChanges():
            errors = "; ".join(layer.commitErrors())
            layer.rollBack()
            raise RuntimeError(f"目标类别批次补偿回滚失败: {errors}")

    def _commit_manual_modify_batch(self):
        task = self._manual_task
        if not task or task.get("kind") != "modify":
            return
        source_code = int(task["class_code"])
        target_code = int(task.get("target_code", source_code))
        source = self._layer(source_code)
        target = self._layer(target_code)
        old_features = self._manual_modify_selected_features(task)
        raw_new_geometries = [
            QgsGeometry(geometry)
            for geometry in (task.get("pending_geometries") or [])
        ]
        if not old_features:
            QMessageBox.information(self, "修改现有面", "请先选择一个或多个灰色旧面")
            return
        for index, geometry in enumerate(raw_new_geometries, start=1):
            error = self._manual_geometry_error(geometry)
            if error:
                QMessageBox.warning(
                    self, "修改现有面", f"第 {index} 个新边界不可保存：{error}"
                )
                return
        try:
            new_geometries = self._manual_geometries_for_commit(task)
        except RuntimeError as exc:
            QMessageBox.information(self, "修改现有面", str(exc))
            return
        for index, geometry in enumerate(new_geometries, start=1):
            error = self._manual_geometry_error(geometry)
            if error:
                QMessageBox.warning(
                    self, "修改现有面", f"第 {index} 个保存边界不可用：{error}"
                )
                return
        if not raw_new_geometries and target_code == source_code:
            QMessageBox.information(
                self, "修改现有面", "没有绘制新边界且目标类别未改变"
            )
            return
        plan = None
        if raw_new_geometries:
            try:
                plan = self._manual_modify_overlap_plan(
                    old_features, raw_new_geometries
                )
            except RuntimeError as exc:
                QMessageBox.warning(self, "修改现有面", str(exc))
                return
        else:
            plan = {
                "matches": [],
                "unmatched_old": [],
                "unmatched_new": [],
            }
        expected_deleted = len(plan["unmatched_old"]) if raw_new_geometries else 0
        expected_added = len(plan["unmatched_new"]) if raw_new_geometries else 0
        smoothing_text = (
            "；身份匹配按原始边界计算，保存当前光滑预览"
            if task.get("smoothing_enabled") and raw_new_geometries else ""
        )
        answer = QMessageBox.question(
            self,
            "保存本批修改",
            f"本批旧面 {len(old_features)} 个，新边界 {len(new_geometries)} 个，"
            f"保存后删除旧面 {expected_deleted} 个、新增 {expected_added} 个，"
            "相交的新边界继承旧面身份；目标类别为 "
            f"{target_code} {CLASS_NAMES[target_code]}{smoothing_text}。确定提交吗？",
            YES | NO,
            NO,
        )
        if answer != YES:
            return

        self._disconnect_manual_picker(restore=False)
        self._disconnect_manual_capture(restore=False)
        task["state"] = "committing"
        self._update_manual_panel()
        source_was_editable = source.isEditable()
        target_was_editable = target.isEditable()
        keep_source_editing = bool(
            source_was_editable or task.get("editing_started_by_task")
        )
        added_target_ids = []
        matched_records = []
        added_records = []
        deleted_records = []
        confidence_warnings = []
        self._metadata_update = True
        try:
            if not new_geometries:
                if target_code == source_code:
                    raise RuntimeError("当前批次没有变化")
                prepared = []
                for original in old_features:
                    geometry = QgsGeometry(original.geometry())
                    moved, object_id, warning = self._manual_replacement_feature(
                        target, geometry, target_code, original,
                        recalculate_confidence=False,
                    )
                    if self._object_id_exists(target, object_id):
                        raise RuntimeError(f"目标类别已存在 object_id: {object_id}")
                    prepared.append((moved, original, object_id, warning, geometry))
                if not target.isEditable() and not target.startEditing():
                    raise RuntimeError("无法启动目标类别工作层编辑")
                for moved, _original, _object_id, _warning, _geometry in prepared:
                    if not target.addFeature(moved):
                        raise RuntimeError("无法向目标类别写入本批对象")
                if not target.commitChanges(not target_was_editable):
                    errors = "; ".join(target.commitErrors())
                    raise RuntimeError(errors or "无法保存目标类别批次")
                added_target_ids = [item[2] for item in prepared]
                if not source.isEditable() and not source.startEditing():
                    raise RuntimeError("无法启动来源类别工作层编辑")
                if not source.deleteFeatures([feature.id() for feature in old_features]):
                    raise RuntimeError("无法从来源类别删除本批旧面")
                if not source.commitChanges(not keep_source_editing):
                    errors = "; ".join(source.commitErrors())
                    raise RuntimeError(errors or "无法保存来源类别批次")
                for moved, original, object_id, warning, geometry in prepared:
                    matched_records.append((original, object_id, geometry, False))
                    if warning:
                        confidence_warnings.append((target_code, object_id, warning))
            elif source_code == target_code:
                if not source.isEditable() and not source.startEditing():
                    raise RuntimeError("无法启动当前类别工作层编辑")
                for old_index, new_index, _area in plan["matches"]:
                    original = old_features[old_index]
                    geometry = new_geometries[new_index]
                    if not source.changeGeometry(original.id(), geometry):
                        raise RuntimeError("无法更新匹配旧面的 geometry")
                    prepared, object_id, warning = self._manual_replacement_feature(
                        source, geometry, target_code, original
                    )
                    self._set_attributes(source, original.id(), {
                        field.name(): prepared.attribute(field.name())
                        for field in source.fields()
                    })
                    geometry_changed = (
                        class_workspace.geometry_hash(original.geometry())
                        != class_workspace.geometry_hash(geometry)
                    )
                    matched_records.append((original, object_id, geometry, geometry_changed))
                    if warning:
                        confidence_warnings.append((target_code, object_id, warning))
                for old_index in plan["unmatched_old"]:
                    original = old_features[old_index]
                    if not source.deleteFeature(original.id()):
                        raise RuntimeError("无法删除未被新边界保留的旧面")
                    deleted_records.append(original)
                for new_index in plan["unmatched_new"]:
                    geometry = new_geometries[new_index]
                    added, object_id, warning = self._manual_replacement_feature(
                        source, geometry, target_code
                    )
                    if not source.addFeature(added):
                        raise RuntimeError("无法新增本批新面")
                    added_records.append((object_id, geometry))
                    if warning:
                        confidence_warnings.append((target_code, object_id, warning))
                if not source.commitChanges(not keep_source_editing):
                    errors = "; ".join(source.commitErrors())
                    raise RuntimeError(errors or "无法保存本批修改")
            else:
                prepared = []
                for old_index, new_index, _area in plan["matches"]:
                    original = old_features[old_index]
                    geometry = QgsGeometry(new_geometries[new_index])
                    if source.crs() != target.crs():
                        geometry.transform(QgsCoordinateTransform(
                            source.crs(), target.crs(), QgsProject.instance()
                        ))
                    moved, object_id, warning = self._manual_replacement_feature(
                        target, geometry, target_code, original
                    )
                    if self._object_id_exists(target, object_id):
                        raise RuntimeError(f"目标类别已存在 object_id: {object_id}")
                    prepared.append((moved, original, object_id, warning, geometry, True))
                for new_index in plan["unmatched_new"]:
                    geometry = QgsGeometry(new_geometries[new_index])
                    if source.crs() != target.crs():
                        geometry.transform(QgsCoordinateTransform(
                            source.crs(), target.crs(), QgsProject.instance()
                        ))
                    added, object_id, warning = self._manual_replacement_feature(
                        target, geometry, target_code
                    )
                    prepared.append((added, None, object_id, warning, geometry, False))
                if not target.isEditable() and not target.startEditing():
                    raise RuntimeError("无法启动目标类别工作层编辑")
                for feature, _original, _object_id, _warning, _geometry, _matched in prepared:
                    if not target.addFeature(feature):
                        raise RuntimeError("无法向目标类别写入本批结果")
                if not target.commitChanges(not target_was_editable):
                    errors = "; ".join(target.commitErrors())
                    raise RuntimeError(errors or "无法保存目标类别批次")
                added_target_ids = [item[2] for item in prepared]
                if not source.isEditable() and not source.startEditing():
                    raise RuntimeError("无法启动来源类别工作层编辑")
                if not source.deleteFeatures([feature.id() for feature in old_features]):
                    raise RuntimeError("无法从来源类别删除本批旧面")
                if not source.commitChanges(not keep_source_editing):
                    errors = "; ".join(source.commitErrors())
                    raise RuntimeError(errors or "无法保存来源类别批次")
                matched_old_indexes = {item[0] for item in plan["matches"]}
                for old_index, original in enumerate(old_features):
                    if old_index not in matched_old_indexes:
                        deleted_records.append(original)
                for feature, original, object_id, warning, geometry, matched in prepared:
                    if matched:
                        matched_records.append((original, object_id, geometry, True))
                    else:
                        added_records.append((object_id, geometry))
                    if warning:
                        confidence_warnings.append((target_code, object_id, warning))
        except Exception as exc:
            if source.isEditable() and source.isModified():
                source.rollBack()
            if target is not source and target.isEditable() and target.isModified():
                target.rollBack()
            if target is not source and added_target_ids:
                try:
                    self._remove_transferred_features(target, added_target_ids)
                except Exception as rollback_exc:
                    exc = RuntimeError(f"{exc}；且补偿回滚失败: {rollback_exc}")
            if source_was_editable and not source.isEditable():
                source.startEditing()
            if target_was_editable and not target.isEditable():
                target.startEditing()
            task["state"] = "failed"
            task["error"] = str(exc)
            QMessageBox.warning(self, "保存本批修改失败", str(exc))
            self._update_manual_panel()
            return
        finally:
            self._metadata_update = False

        for original, object_id, geometry, geometry_changed in matched_records:
            before_hash = class_workspace.geometry_hash(original.geometry())
            after_hash = class_workspace.geometry_hash(geometry)
            if geometry_changed:
                class_workspace.append_history(
                    self._run_spec,
                    "geometry_modified",
                    class_code=source_code,
                    to_class_code=target_code,
                    object_id=object_id,
                    before_geometry_hash=before_hash,
                    after_geometry_hash=after_hash,
                )
            if target_code != source_code:
                class_workspace.append_history(
                    self._run_spec,
                    "feature_reclassified",
                    object_id=object_id,
                    part_id=str(original.attribute("part_id") or "000"),
                    from_class_code=source_code,
                    to_class_code=target_code,
                    geometry_hash=after_hash,
                )
        for original in deleted_records:
            class_workspace.append_history(
                self._run_spec,
                "feature_deleted",
                class_code=source_code,
                object_id=str(original.attribute("object_id") or ""),
                before_geometry_hash=class_workspace.geometry_hash(original.geometry()),
                reason="manual_batch_replaced",
            )
        for object_id, geometry in added_records:
            class_workspace.append_history(
                self._run_spec,
                "feature_added",
                class_code=target_code,
                object_id=object_id,
                after_geometry_hash=class_workspace.geometry_hash(geometry),
                reason="manual_batch_added",
            )
        for code, object_id, warning in confidence_warnings:
            class_workspace.append_history(
                self._run_spec,
                "confidence_statistics_unavailable",
                class_code=code,
                object_id=object_id,
                reason=warning,
            )
        self._snapshots.pop(source_code, None)
        self._snapshots.pop(target_code, None)
        self._set_class_modified(source_code)
        if target_code != source_code:
            self._set_class_modified(target_code)
            self._visibility_checks[target_code].setChecked(True)
        self._workspace = class_workspace.save_workspace(self._run_spec, self._workspace)
        self._refresh_class_display(source_code, target_code)
        task["submitted_batch_count"] += 1
        task["modified_old_count"] += len(matched_records)
        task["saved_new_count"] += len(added_records)
        task["deleted_old_count"] += len(deleted_records)
        task["selected_feature_ids"] = []
        task["pending_geometries"] = []
        task["pending_errors"] = []
        task["smoothing_preview"] = None
        task["smoothing_error"] = ""
        self._manual_smoothing_timer.stop()
        task["target_code"] = source_code
        task["state"] = "selecting"
        task.pop("error", None)
        source.removeSelection()
        self._clear_manual_bands()
        self._set_target_combo_code(source_code)
        self.iface.setActiveLayer(source)
        self.baseline_label.setText(
            f"本批修改已保存：旧面 {len(old_features)} 个，新边界 "
            f"{len(new_geometries)} 个，删除旧面 {expected_deleted} 个，"
            f"新增 {expected_added} 个；继续选择下一批"
        )
        self._refresh_table()
        self._start_manual_picker()
        self._update_manual_panel()

    def _commit_manual_delete(self):
        task = self._manual_task
        if not task or task.get("kind") != "delete":
            return
        class_code = int(task["class_code"])
        layer = self._layer(class_code)
        feature_ids = list(layer.selectedFeatureIds())
        if not feature_ids:
            QMessageBox.information(self, "删除现有面", "请先在地图中选择一个或多个面")
            return
        answer = QMessageBox.question(
            self,
            "确认删除",
            f"确定从 {class_code} {CLASS_NAMES[class_code]} 删除选中的 "
            f"{len(feature_ids)} 个面吗？",
            YES | NO,
            NO,
        )
        if answer != YES:
            return
        task["state"] = "committing"
        self._update_manual_panel()
        try:
            if not layer.isEditable() and not layer.startEditing():
                raise RuntimeError("无法启动当前类别工作层编辑")
            if not layer.deleteFeatures(feature_ids):
                layer.rollBack()
                raise RuntimeError("无法删除选中的面")
            if not layer.commitChanges():
                errors = "; ".join(layer.commitErrors())
                layer.rollBack()
                raise RuntimeError(errors or "保存删除失败")
        except Exception as exc:
            task["state"] = "selecting"
            QMessageBox.warning(self, "删除现有面失败", str(exc))
            self._start_manual_picker()
            self._update_manual_panel()
            return
        self.baseline_label.setText(
            f"已从 {class_code} {CLASS_NAMES[class_code]} 删除 {len(feature_ids)} 个面"
        )
        self._finish_manual_task_success()

    def _commit_manual_add(self):
        task = self._manual_task
        if not task or task.get("kind") != "add":
            return
        source_code = int(task["class_code"])
        target_code = int(task.get("target_code", source_code))
        raw_geometries = [
            QgsGeometry(geometry)
            for geometry in (task.get("pending_geometries") or [])
        ]
        if not raw_geometries:
            QMessageBox.information(self, "新增面", "当前没有待保存的新增面")
            return
        for index, geometry in enumerate(raw_geometries, start=1):
            error = self._manual_geometry_error(geometry)
            if error:
                QMessageBox.warning(
                    self, "新增面", f"本批第 {index} 个候选不可保存：{error}"
                )
                return
        try:
            geometries = self._manual_geometries_for_commit(task)
        except RuntimeError as exc:
            QMessageBox.information(self, "新增面", str(exc))
            return
        for index, geometry in enumerate(geometries, start=1):
            error = self._manual_geometry_error(geometry)
            if error:
                QMessageBox.warning(
                    self, "新增面", f"本批第 {index} 个保存边界不可用：{error}"
                )
                return
        layer = self._layer(target_code)
        prepared = []
        for geometry in geometries:
            object_id = class_workspace.new_object_id(self._run_spec)
            confidence_mean, confidence_std, confidence_warning = (
                self._optional_confidence_statistics(layer, geometry)
            )
            feature = QgsFeature(layer.fields())
            feature.setGeometry(geometry)
            values = {
                "run_id": self._run_spec["run_id"],
                "object_id": object_id,
                "part_id": "000",
                "class_code": target_code,
                "class_name": CLASS_NAMES[target_code],
                "baseline_stream_id": self._workspace["baseline_stream_id"],
                "geometry_source": "manual_edited",
                "geometry_revision": 1,
                "edit_base": "",
                "reviewed": 0,
                "confidence_mean": confidence_mean,
                "confidence_std": confidence_std,
                "updated_at": class_workspace._now(),
            }
            for name, value in values.items():
                if layer.fields().indexOf(name) >= 0:
                    feature.setAttribute(name, value)
            prepared.append((feature, object_id, confidence_warning))
        self._disconnect_manual_capture(restore=False)
        task["state"] = "committing"
        self._update_manual_panel()
        was_editable = layer.isEditable()
        self._metadata_update = True
        try:
            if not layer.isEditable() and not layer.startEditing():
                raise RuntimeError("无法启动目标类别工作层编辑")
            for feature, _object_id, _warning in prepared:
                if not layer.addFeature(feature):
                    raise RuntimeError("无法向目标类别新增本批面")
            if target_code == source_code or was_editable:
                committed = layer.commitChanges(False)
            else:
                committed = layer.commitChanges()
            if not committed:
                errors = "; ".join(layer.commitErrors())
                raise RuntimeError(errors or "保存本批新增面失败")
        except Exception as exc:
            if layer.isEditable():
                layer.rollBack()
            if (
                (was_editable or target_code == source_code)
                and not layer.isEditable()
            ):
                layer.startEditing()
            task["state"] = "failed"
            task["error"] = str(exc)
            self._restore_manual_map_tool()
            QMessageBox.warning(self, "保存本批新增面失败", str(exc))
            self._update_manual_panel()
            return
        finally:
            self._metadata_update = False
        self._snapshots.pop(target_code, None)
        topology_hints = []
        for _feature, object_id, confidence_warning in prepared:
            persisted = self._feature_by_object_id(layer, object_id)
            topology_hint = self._local_topology_hint(
                target_code, persisted.geometry(), persisted.id()
            )
            topology_hints.append(topology_hint)
            class_workspace.append_history(
                self._run_spec,
                "feature_added",
                class_code=target_code,
                object_id=object_id,
                after_geometry_hash=class_workspace.geometry_hash(persisted.geometry()),
                overlap_hint=topology_hint,
            )
            if confidence_warning:
                class_workspace.append_history(
                    self._run_spec,
                    "confidence_statistics_unavailable",
                    class_code=target_code,
                    object_id=object_id,
                    reason=confidence_warning,
                )
        batch_size = len(prepared)
        self._set_class_modified(target_code)
        self._workspace = class_workspace.save_workspace(self._run_spec, self._workspace)
        self._visibility_checks[target_code].setChecked(True)
        self._refresh_class_display(target_code)
        task["added_count"] += batch_size
        task["submitted_batch_count"] += 1
        saved_counts = task.setdefault("saved_counts", {})
        saved_counts[target_code] = int(saved_counts.get(target_code, 0)) + batch_size
        task["state"] = "capturing"
        task["pending_geometries"] = []
        task["pending_errors"] = []
        task["smoothing_preview"] = None
        task["smoothing_error"] = ""
        self._manual_smoothing_timer.stop()
        task["target_code"] = source_code
        task.pop("error", None)
        self._clear_manual_add_candidate_bands()
        self._set_target_combo_code(source_code)
        hint_summary = "；".join(dict.fromkeys(topology_hints))
        self.baseline_label.setText(
            f"已向 {target_code} {CLASS_NAMES[target_code]} 提交本批 {batch_size} 个面；"
            f"本次累计 {task['added_count']} 个面；局部提示: {hint_summary}"
        )
        self._refresh_table()
        self._start_manual_capture()

    def _finish_manual_session(self):
        task = self._manual_task
        if not task:
            return
        if task.get("kind") == "add":
            self._finish_add_task()
        elif task.get("kind") == "modify":
            self._finish_modify_task()

    def _finish_modify_task(self):
        task = self._manual_task
        if not task or task.get("kind") != "modify":
            return
        class_code = int(task["class_code"])
        discarded_old_count = len(task.get("selected_feature_ids") or [])
        discarded_new_count = len(task.get("pending_geometries") or [])
        batch_count = int(task.get("submitted_batch_count", 0))
        modified_count = int(task.get("modified_old_count", 0))
        added_count = int(task.get("saved_new_count", 0))
        deleted_count = int(task.get("deleted_old_count", 0))
        self._manual_smoothing_timer.stop()
        self._disconnect_manual_picker(restore=False)
        self._disconnect_manual_capture(restore=False)
        self._clear_manual_bands()
        self._restore_manual_map_tool()
        self._finish_manual_editing(task)
        self._manual_task = None
        self._manual_previous_map_tool = None
        self._set_target_combo_code(class_code)
        self.baseline_label.setText(
            f"修改已结束：提交 {batch_count} 批，修改/改类 {modified_count} 个，"
            f"新增面 {added_count} 个，删除旧面 {deleted_count} 个；"
            f"丢弃未保存旧面 {discarded_old_count} 个、新边界 {discarded_new_count} 个"
        )
        self._select_class_context(class_code, activate_layer=True)
        self._update_manual_panel()
        self._refresh_table()

    def _finish_add_task(self):
        task = self._manual_task
        if not task or task.get("kind") != "add":
            return
        added_count = int(task.get("added_count", 0))
        batch_count = int(task.get("submitted_batch_count", 0))
        discarded_count = len(task.get("pending_geometries") or [])
        class_code = int(task["class_code"])
        self._manual_smoothing_timer.stop()
        self._disconnect_manual_capture(restore=False)
        self._clear_manual_bands()
        self._restore_manual_map_tool()
        self._finish_manual_editing(task)
        self._manual_task = None
        self._manual_previous_map_tool = None
        self.baseline_label.setText(
            f"新增已结束：本次提交 {batch_count} 批、保存 {added_count} 个面；"
            f"丢弃 {discarded_count} 个未保存候选"
        )
        self._set_target_combo_code(class_code)
        self._update_manual_panel()
        self._refresh_table()

    def _finish_manual_editing(self, task):
        if not task or task.get("kind") not in ("add", "modify"):
            return
        if not task.get("editing_started_by_task"):
            return
        class_code = int(task["class_code"])
        if class_code not in self._class_layers:
            return
        layer = self._layer(class_code)
        if not layer.isEditable() or layer.isModified():
            return
        self._metadata_update = True
        try:
            layer.rollBack()
        finally:
            self._metadata_update = False

    def _qgis_smooth_parameters(self):
        return (
            int(self.qgis_smooth_iterations_spin.value()),
            float(self.qgis_smooth_offset_spin.value()),
            float(self.qgis_smooth_angle_spin.value()),
        )

    def _qgis_smooth_parameters_changed(self, *_args):
        if self._smoothing_parameter_sync:
            return
        iterations, offset, max_angle = self._qgis_smooth_parameters()
        self._store_smoothing_parameters((iterations, offset, max_angle))
        self._sync_smoothing_parameter_widgets(
            (iterations, offset, max_angle), "qgis"
        )
        self._clear_qgis_smooth_preview(
            status_text="参数已更新，请重新预览光滑效果"
        )

    @staticmethod
    def _geometry_vertex_count(geometry):
        abstract = geometry.constGet() if geometry is not None else None
        return int(abstract.nCoordinates()) if abstract is not None else 0

    def _qgis_smooth_preview_is_current(self):
        preview = self._qgis_smooth_preview
        layer = self.iface.activeLayer()
        if (
            not preview or layer is None or not layer.isEditable()
            or self._manual_task
            or layer.id() != preview["layer_id"]
            or self._qgis_smooth_parameters() != preview["parameters"]
            or tuple(sorted(layer.selectedFeatureIds())) != preview["feature_ids"]
        ):
            return False
        for feature_id, expected_hash in preview["source_hashes"].items():
            feature = layer.getFeature(feature_id)
            if (
                not feature.isValid()
                or class_workspace.geometry_hash(feature.geometry()) != expected_hash
            ):
                return False
        return True

    def _update_qgis_smoothing_controls(self):
        if not hasattr(self, "qgis_smooth_preview_btn"):
            return
        layer = self.iface.activeLayer()
        available = bool(
            layer is not None
            and self._class_code_for_layer(layer) is not None
            and layer.isEditable()
            and not self._manual_task
        )
        selected_count = layer.selectedFeatureCount() if available else 0
        self.qgis_smooth_selection_label.setText(f"已选面：{selected_count}")
        for spin in (
            self.qgis_smooth_iterations_spin,
            self.qgis_smooth_offset_spin,
            self.qgis_smooth_angle_spin,
        ):
            spin.setEnabled(available)
        self.qgis_smooth_preview_btn.setEnabled(available and selected_count > 0)
        self.qgis_smooth_apply_btn.setEnabled(
            available and self._qgis_smooth_preview_is_current()
        )
        self.qgis_smooth_clear_btn.setEnabled(
            available and self._qgis_smooth_preview is not None
        )

    def _clear_qgis_smooth_preview(self, _checked=False, status_text=None):
        for band in self._qgis_smooth_preview_bands:
            try:
                band.reset(Qgis.GeometryType.Polygon)
                self.iface.mapCanvas().scene().removeItem(band)
            except RuntimeError:
                pass
        self._qgis_smooth_preview_bands = []
        self._qgis_smooth_preview = None
        if hasattr(self, "qgis_smooth_status_label"):
            self.qgis_smooth_status_label.setText(
                status_text
                or "请选择一个或多个面；参数会自动记住，预览不会修改工作层"
            )
        self._update_qgis_smoothing_controls()

    def _preview_qgis_smoothing(self):
        layer = self.iface.activeLayer()
        class_code = self._class_code_for_layer(layer)
        if (
            layer is None or class_code is None or not layer.isEditable()
            or self._manual_task
        ):
            QMessageBox.information(
                self, "预览光滑效果", "请先在 QGIS 中开启一个类别层的编辑模式"
            )
            return
        features = sorted(layer.selectedFeatures(), key=lambda feature: feature.id())
        if not features:
            QMessageBox.information(
                self, "预览光滑效果", "请先在地图中选择一个或多个面"
            )
            return
        iterations, offset, max_angle = self._qgis_smooth_parameters()
        smoothed_geometries = {}
        source_hashes = {}
        old_vertices = 0
        new_vertices = 0
        old_area = 0.0
        new_area = 0.0
        for index, feature in enumerate(features, start=1):
            source = QgsGeometry(feature.geometry())
            smoothed = source.smooth(iterations, offset, -1.0, max_angle)
            error = self._manual_geometry_error(smoothed)
            if error:
                self._clear_qgis_smooth_preview(
                    status_text=f"第 {index} 个面的光滑结果不可用：{error}"
                )
                QMessageBox.warning(
                    self, "预览光滑失败",
                    f"第 {index} 个面的光滑结果不可用：{error}；本批未产生预览",
                )
                return
            smoothed_geometries[feature.id()] = smoothed
            source_hashes[feature.id()] = class_workspace.geometry_hash(source)
            old_vertices += self._geometry_vertex_count(source)
            new_vertices += self._geometry_vertex_count(smoothed)
            old_area += float(source.area())
            new_area += float(smoothed.area())
        self._clear_qgis_smooth_preview()
        color = QColor("#00bcd4")
        fill = QColor(color)
        fill.setAlpha(45)
        for geometry in smoothed_geometries.values():
            band = QgsRubberBand(
                self.iface.mapCanvas(), Qgis.GeometryType.Polygon
            )
            band.setStrokeColor(color)
            band.setFillColor(fill)
            band.setLineStyle(DASH_LINE)
            band.setWidth(2)
            band.setToGeometry(geometry, layer)
            self._qgis_smooth_preview_bands.append(band)
        feature_ids = tuple(sorted(smoothed_geometries))
        self._qgis_smooth_preview = {
            "layer_id": layer.id(),
            "class_code": class_code,
            "feature_ids": feature_ids,
            "source_hashes": source_hashes,
            "parameters": (iterations, offset, max_angle),
            "geometries": smoothed_geometries,
        }
        area_change = (
            ((new_area - old_area) / old_area) * 100.0 if old_area > 0.0 else 0.0
        )
        self.qgis_smooth_status_label.setText(
            f"已预览 {len(features)} 个面：顶点 {old_vertices} → {new_vertices}，"
            f"总面积变化 {area_change:+.3f}%；应用后仍需保存 QGIS 编辑"
        )
        self._update_qgis_smoothing_controls()

    def _apply_qgis_smoothing(self):
        if not self._qgis_smooth_preview_is_current():
            self._clear_qgis_smooth_preview(
                status_text="预览已失效，请按当前选择和参数重新预览"
            )
            QMessageBox.information(
                self, "应用光滑", "预览已失效，请重新预览后再应用"
            )
            return
        preview = self._qgis_smooth_preview
        layer = self.iface.activeLayer()
        geometries = {
            feature_id: QgsGeometry(geometry)
            for feature_id, geometry in preview["geometries"].items()
        }
        parameters = preview["parameters"]
        self._clear_qgis_smooth_preview()
        layer.beginEditCommand(f"光滑选中 {len(geometries)} 个面")
        try:
            for feature_id, geometry in geometries.items():
                if not layer.changeGeometry(feature_id, geometry):
                    raise RuntimeError(f"无法更新要素 {feature_id} 的 geometry")
            layer.endEditCommand()
        except Exception as exc:
            layer.destroyEditCommand()
            QMessageBox.warning(self, "应用光滑失败", str(exc))
            self._update_manual_panel()
            return
        iterations, offset, max_angle = parameters
        self.qgis_smooth_status_label.setText(
            f"已应用到 {len(geometries)} 个面：次数 {iterations}、偏移 "
            f"{offset:.2f}、最大角度 {max_angle:.0f}°；尚未保存，可撤销一步"
        )
        self.baseline_label.setText(
            f"类别 {preview['class_code']} 已应用光滑预览；尚未保存 QGIS 编辑"
        )
        layer.triggerRepaint()
        self.iface.mapCanvas().refresh()
        self._update_manual_panel()

    def _undo_current_edit(self):
        layer = self.iface.activeLayer()
        if layer is not None and layer.isEditable():
            layer.undoStack().undo()
            self._update_manual_panel()

    def _redo_current_edit(self):
        layer = self.iface.activeLayer()
        if layer is not None and layer.isEditable():
            layer.undoStack().redo()
            self._update_manual_panel()

    def _save_current_edit(self):
        class_code = self._current_class_code()
        if class_code is not None:
            self._save_class_edits(class_code)

    def _rollback_current_edit(self):
        class_code = self._current_class_code()
        if class_code is not None:
            self._rollback_class_edits(class_code)

    def _finish_manual_task_success(self):
        task = self._manual_task
        if not task:
            return
        final_code = int(
            task.get("target_code", task["class_code"])
            if task.get("kind") == "modify"
            else task["class_code"]
        )
        self._disconnect_manual_picker(restore=False)
        self._disconnect_manual_capture(restore=False)
        self._clear_manual_bands()
        self._restore_manual_map_tool()
        self._manual_task = None
        self._manual_previous_map_tool = None
        if final_code in self._class_layers:
            self._select_class_context(final_code, activate_layer=True)
            self._set_target_combo_code(final_code)
        self._update_manual_panel()
        self._refresh_table()

    def _cancel_manual_task(self, _checked=False, silent=False, restore_selection=True):
        task = self._manual_task
        if not task:
            return
        kind = task.get("kind")
        class_code = int(task["class_code"])
        pending_count = len(task.get("pending_geometries") or [])
        self._manual_smoothing_timer.stop()
        self._disconnect_manual_picker(restore=False)
        self._disconnect_manual_capture(restore=False)
        self._clear_manual_bands()
        self._restore_manual_map_tool()
        self._finish_manual_editing(task)
        if restore_selection and class_code in self._class_layers:
            layer = self._layer(class_code)
            existing = {feature.id() for feature in layer.getFeatures()}
            layer.selectByIds([
                feature_id for feature_id in task.get("selection_before", [])
                if feature_id in existing
            ])
        added_count = int(task.get("added_count", 0))
        self._manual_task = None
        self._manual_previous_map_tool = None
        self._set_target_combo_code(class_code)
        if not silent:
            if kind == "add" and added_count:
                self.baseline_label.setText(
                    f"新增已结束；已保存 {added_count} 个面继续保留，"
                    f"丢弃 {pending_count} 个未保存候选"
                )
            elif kind == "add":
                self.baseline_label.setText(
                    f"新增已结束；丢弃 {pending_count} 个未保存候选"
                )
            elif kind == "modify":
                self.baseline_label.setText(
                    f"修改已结束；丢弃 {pending_count} 个未保存新边界"
                )
            else:
                self.baseline_label.setText("人工任务已取消；工作层未产生新修改")
        self._update_manual_panel()
        self._refresh_table()

    def _update_manual_panel(self):
        if not hasattr(self, "manual_context_label"):
            return
        class_code = self._current_class_code()
        layer = None
        selected_count = 0
        edit_text = "-"
        if class_code is not None and class_code in self._class_layers:
            layer = self._layer(class_code)
            selected_count = layer.selectedFeatureCount()
            edit_text = (
                "有未保存修改" if layer.isEditable() and layer.isModified()
                else "编辑中" if layer.isEditable()
                else "已保存"
            )
            current_text = f"{class_code} {CLASS_NAMES[class_code]}"
        else:
            current_text = "未选择"
        active_name = self.iface.activeLayer().name() if self.iface.activeLayer() else "未同步"
        self.manual_context_label.setText(
            f"当前类别：{current_text} | QGIS 活动层：{active_name} | "
            f"已选面：{selected_count} | 编辑状态：{edit_text}"
        )
        task = self._manual_task
        if task and task.get("kind") == "modify":
            selected_count = len(task.get("selected_feature_ids") or [])
            self.manual_context_label.setText(
                f"当前类别：{current_text} | QGIS 活动层：{active_name} | "
                f"待修改旧面：{selected_count} | 编辑状态：{edit_text}"
            )
        modified = bool(self._editable_modified_layers()) if self._workspace else False
        idle_enabled = bool(
            self._workspace and class_code is not None and not task
            and not self._active_session and not modified
        )
        self.modify_task_btn.setEnabled(idle_enabled and bool(layer and layer.featureCount()))
        self.delete_task_btn.setEnabled(idle_enabled and bool(layer and layer.featureCount()))
        self.add_task_btn.setEnabled(idle_enabled)
        is_modify = bool(task and task.get("kind") == "modify")
        is_add = bool(task and task.get("kind") == "add")
        pending_count = (
            len(task.get("pending_geometries") or [])
            if (is_add or is_modify) else 0
        )
        modify_selected_count = (
            len(task.get("selected_feature_ids") or []) if is_modify else 0
        )
        show_target_class = bool(
            (is_modify and modify_selected_count > 0)
            or (is_add and pending_count > 0)
        )
        self.target_class_label.setText("本批目标类别:")
        self.target_class_label.setVisible(show_target_class)
        self.target_class_combo.setVisible(show_target_class)
        target_enabled = bool(
            task
            and task.get("state") not in ("committing", "paused")
            and show_target_class
        )
        self.target_class_combo.setEnabled(target_enabled and bool(self._workspace))
        show_manual_smoothing = bool((is_modify or is_add) and pending_count > 0)
        manual_smoothing_enabled = bool(
            show_manual_smoothing and task.get("smoothing_enabled")
        )
        smoothing_widgets = (
            self.manual_smooth_enabled_check,
            self.manual_smooth_iterations_label,
            self.manual_smooth_iterations_spin,
            self.manual_smooth_offset_label,
            self.manual_smooth_offset_spin,
            self.manual_smooth_angle_label,
            self.manual_smooth_angle_spin,
        )
        for widget in smoothing_widgets:
            widget.setVisible(show_manual_smoothing)
        self.manual_smooth_status_label.setVisible(show_manual_smoothing)
        self.manual_smooth_enabled_check.setEnabled(
            show_manual_smoothing
            and task.get("state") not in ("committing", "paused")
        )
        for spin in (
            self.manual_smooth_iterations_spin,
            self.manual_smooth_offset_spin,
            self.manual_smooth_angle_spin,
        ):
            spin.setEnabled(
                manual_smoothing_enabled
                and task.get("state") not in ("committing", "paused")
            )
        smoothing_ready = bool(
            not manual_smoothing_enabled
            or self._manual_smoothing_preview_is_current(task)
        )
        for widget in (
            self.manual_primary_btn,
            self.manual_retry_btn,
            self.manual_clear_btn,
            self.manual_continue_btn,
            self.manual_cancel_btn,
            self.manual_finish_btn,
        ):
            widget.setVisible(False)
        if task:
            kind = task.get("kind")
            state = task.get("state")
            if state == "paused":
                self.manual_continue_btn.setVisible(True)
                if kind in ("modify", "add"):
                    self.manual_finish_btn.setVisible(True)
                    self.manual_finish_btn.setText(
                        "结束修改" if kind == "modify" else "结束新增"
                    )
                else:
                    self.manual_cancel_btn.setVisible(True)
                    self.manual_cancel_btn.setText("结束删除")
            elif kind == "modify":
                self.manual_finish_btn.setVisible(True)
                self.manual_finish_btn.setText("结束修改")
                if modify_selected_count:
                    self.manual_primary_btn.setVisible(True)
                    self.manual_primary_btn.setText("保存修改并继续")
                    target = int(task.get("target_code", task["class_code"]))
                    self.manual_primary_btn.setEnabled(
                        state != "committing"
                        and not any(task.get("pending_errors") or [])
                        and smoothing_ready
                        and bool(pending_count or target != int(task["class_code"]))
                    )
                    self.manual_retry_btn.setVisible(True)
                    self.manual_retry_btn.setText(
                        "重新绘制当前面" if pending_count else "绘制新边界"
                    )
                    self.manual_retry_btn.setEnabled(state != "committing")
            elif kind == "delete":
                self.manual_primary_btn.setVisible(True)
                count = self._layer(task["class_code"]).selectedFeatureCount()
                self.manual_primary_btn.setText(f"删除选中的 {count} 个面")
                self.manual_primary_btn.setEnabled(count > 0 and state == "selecting")
                self.manual_clear_btn.setVisible(True)
                self.manual_clear_btn.setEnabled(count > 0)
                self.manual_cancel_btn.setVisible(True)
                self.manual_cancel_btn.setText("结束删除")
            elif kind == "add":
                self.manual_finish_btn.setVisible(True)
                self.manual_finish_btn.setText("结束新增")
                if pending_count:
                    self.manual_primary_btn.setVisible(True)
                    self.manual_primary_btn.setText("保存新增面并继续新增面")
                    self.manual_primary_btn.setEnabled(
                        state != "committing"
                        and not any(task.get("pending_errors") or [])
                        and smoothing_ready
                    )
                    self.manual_retry_btn.setVisible(True)
                    self.manual_retry_btn.setEnabled(state != "committing")
                elif state in ("capture_cancelled", "failed"):
                    self.manual_retry_btn.setVisible(True)
        edit_layer = self.iface.activeLayer()
        show_edit = bool(
            edit_layer is not None and self._class_code_for_layer(edit_layer) is not None
            and edit_layer.isEditable() and not task
        )
        self.qgis_edit_group.setVisible(show_edit)
        if show_edit:
            edit_code = self._class_code_for_layer(edit_layer)
            self.qgis_edit_context_label.setText(
                f"当前 QGIS 编辑层：{edit_code} {CLASS_NAMES[edit_code]} | "
                "撤销/重做按步骤，保存/放弃作用于当前全部未保存编辑"
            )
            undo_stack = edit_layer.undoStack()
            self.qgis_undo_btn.setEnabled(undo_stack.canUndo())
            self.qgis_redo_btn.setEnabled(undo_stack.canRedo())
            self.qgis_save_btn.setEnabled(edit_layer.isModified())
            self.qgis_rollback_btn.setEnabled(True)
        elif self._qgis_smooth_preview is not None:
            self._clear_qgis_smooth_preview()
        self._update_qgis_smoothing_controls()
        self._update_actions()

    def _snapshot(self, layer):
        snapshot = {}
        for feature in layer.getFeatures():
            snapshot[feature.id()] = {
                "object_id": str(feature.attribute("object_id") or ""),
                "geometry_hash": class_workspace.geometry_hash(feature.geometry()),
                "geometry_source": str(feature.attribute("geometry_source") or "fusion"),
                "geometry_revision": int(feature.attribute("geometry_revision") or 0),
                "immutable": {
                    name: feature.attribute(name)
                    for name in class_workspace.IMMUTABLE_FIELDS
                },
            }
        return snapshot

    def _persisted_snapshot(self, class_code):
        record = self._workspace["classes"][str(class_code)]
        persisted = class_workspace.working_layer(
            record, f"class_{class_code}_persisted_snapshot"
        )
        return self._snapshot(persisted)

    def _editing_started(self, class_code):
        if self._metadata_update:
            return
        self._snapshots[class_code] = self._snapshot(self._layer(class_code))
        self._update_manual_panel()
        self._update_actions()

    def _editing_stopped(self, class_code):
        preview = self._qgis_smooth_preview
        if preview and int(preview["class_code"]) == int(class_code):
            self._clear_qgis_smooth_preview()
        if self._metadata_update or class_code not in self._snapshots:
            return
        layer = self._layer(class_code)
        before = self._snapshots[class_code]
        after_features = {feature.id(): feature for feature in layer.getFeatures()}
        changed = []
        deleted = sorted(set(before) - set(after_features))
        added = sorted(set(after_features) - set(before))
        for feature_id in sorted(set(before) & set(after_features)):
            feature = after_features[feature_id]
            if class_workspace.geometry_hash(feature.geometry()) != before[feature_id]["geometry_hash"]:
                changed.append(feature_id)
        if not changed and not deleted and not added:
            if layer.isEditable():
                self._snapshots[class_code] = self._snapshot(layer)
            else:
                self._snapshots.pop(class_code, None)
            self._edit_context.pop(class_code, None)
            self._update_manual_panel()
            self._update_actions()
            return
        context = self._edit_context.pop(class_code, {})
        keep_editing = layer.isEditable()
        confidence_warnings = []
        self._metadata_update = True
        try:
            if not keep_editing and not layer.startEditing():
                raise RuntimeError("cannot start metadata update after geometry edit")
            for feature_id in changed:
                feature = after_features[feature_id]
                old = before[feature_id]
                confidence_mean, confidence_std, confidence_warning = (
                    self._optional_confidence_statistics(
                        layer, feature.geometry()
                    )
                )
                if confidence_warning:
                    confidence_warnings.append((old["object_id"], confidence_warning))
                for name, value in old["immutable"].items():
                    layer.changeAttributeValue(feature_id, layer.fields().indexOf(name), value)
                self._set_attributes(layer, feature_id, {
                    "confidence_mean": confidence_mean,
                    "confidence_std": confidence_std,
                })
                if not context.get("metadata_prepared"):
                    self._set_attributes(layer, feature_id, {
                        "geometry_source": "manual_edited",
                        "edit_base": old["geometry_source"],
                        "geometry_revision": old["geometry_revision"] + 1,
                        "updated_at": class_workspace._now(),
                    })
                class_workspace.append_history(
                    self._run_spec,
                    "geometry_modified",
                    class_code=class_code,
                    object_id=old["object_id"],
                    before_geometry_hash=old["geometry_hash"],
                    after_geometry_hash=class_workspace.geometry_hash(feature.geometry()),
                )
            for feature_id in added:
                feature = after_features[feature_id]
                object_id = str(feature.attribute("object_id") or class_workspace.new_object_id(self._run_spec))
                confidence_mean, confidence_std, confidence_warning = (
                    self._optional_confidence_statistics(
                        layer, feature.geometry()
                    )
                )
                if confidence_warning:
                    confidence_warnings.append((object_id, confidence_warning))
                values = {
                    "confidence_mean": confidence_mean,
                    "confidence_std": confidence_std,
                    "updated_at": class_workspace._now(),
                }
                if not context.get("metadata_prepared"):
                    values.update({
                        "run_id": self._run_spec["run_id"],
                        "object_id": object_id,
                        "part_id": "000",
                        "class_code": class_code,
                        "class_name": CLASS_NAMES[class_code],
                        "baseline_stream_id": self._workspace["baseline_stream_id"],
                        "geometry_source": "manual_edited",
                        "geometry_revision": 1,
                        "edit_base": "",
                        "reviewed": 0,
                    })
                self._set_attributes(layer, feature_id, values)
                class_workspace.append_history(
                    self._run_spec, "feature_added", class_code=class_code,
                    object_id=object_id,
                    after_geometry_hash=class_workspace.geometry_hash(feature.geometry()),
                )
            for feature_id in deleted:
                old = before[feature_id]
                class_workspace.append_history(
                    self._run_spec, "feature_deleted", class_code=class_code,
                    object_id=old["object_id"], before_geometry_hash=old["geometry_hash"],
                )
            if not layer.commitChanges(not keep_editing):
                errors = "; ".join(layer.commitErrors())
                layer.rollBack()
                raise RuntimeError(f"cannot save edit metadata: {errors}")
        finally:
            self._metadata_update = False
        if layer.isEditable():
            self._snapshots[class_code] = self._snapshot(layer)
        else:
            self._snapshots.pop(class_code, None)
        for object_id, warning in confidence_warnings:
            class_workspace.append_history(
                self._run_spec,
                "confidence_statistics_unavailable",
                class_code=class_code,
                object_id=object_id,
                reason=warning,
            )
        hints = []
        for feature_id in changed + added:
            feature = after_features[feature_id]
            hints.append(
                self._local_topology_hint(class_code, feature.geometry(), feature_id)
            )
        if hints:
            self.baseline_label.setText(
                f"类别 {class_code} 已保存；局部拓扑提示: " + "; ".join(hints)
            )
        self._mark_class_modified(class_code)
        self._refresh_class_display(class_code)
        self._update_manual_panel()

    def _set_attributes(self, layer, feature_id, values):
        for name, value in values.items():
            index = layer.fields().indexOf(name)
            if index >= 0:
                layer.changeAttributeValue(feature_id, index, value)

    @staticmethod
    def _feature_by_object_id(layer, object_id):
        wanted = str(object_id or "")
        for feature in layer.getFeatures():
            if str(feature.attribute("object_id") or "") == wanted:
                return feature
        raise RuntimeError(f"cannot reload persisted object: {wanted}")

    def _confidence_statistics(self, layer, geometry):
        stream = class_workspace.stream_by_id(
            self._eligible_fusions, self._workspace["baseline_stream_id"]
        )
        confidence_path = str((stream.get("paths") or {}).get("confidence_mosaic") or "")
        raster = self._confidence_raster
        if raster is None or raster.source() != confidence_path:
            raster = QgsRasterLayer(confidence_path, "fusion_confidence_statistics")
            if not raster.isValid():
                raise RuntimeError(f"无法打开 Fusion confidence mosaic: {confidence_path}")
            self._confidence_raster = raster
        raster_geometry = QgsGeometry(geometry)
        if layer.crs() != raster.crs():
            raster_geometry.transform(
                QgsCoordinateTransform(layer.crs(), raster.crs(), QgsProject.instance())
            )
        wanted = Qgis.ZonalStatistic.Mean | Qgis.ZonalStatistic.StDev
        statistics = QgsZonalStatistics.calculateStatistics(
            raster.dataProvider(),
            raster_geometry,
            abs(raster.rasterUnitsPerPixelX()),
            abs(raster.rasterUnitsPerPixelY()),
            1,
            wanted,
        )
        mean = statistics.get(Qgis.ZonalStatistic.Mean)
        std = statistics.get(Qgis.ZonalStatistic.StDev)
        if mean is None or std is None:
            raise RuntimeError("新几何范围内没有可用于 confidence 统计的像元")
        return float(mean), float(std)

    def _optional_confidence_statistics(self, layer, geometry):
        try:
            mean, std = self._confidence_statistics(layer, geometry)
            return mean, std, ""
        except RuntimeError as exc:
            return None, None, str(exc)

    def _set_class_modified(self, class_code):
        record = self._workspace["classes"][str(class_code)]
        record["modified"] = True
        record["confirmed"] = False
        record["state"] = "editing" if self._layer(class_code).featureCount() else "unreviewed_empty"
        self._confirm_buttons[class_code].blockSignals(True)
        self._confirm_buttons[class_code].setChecked(False)
        self._confirm_buttons[class_code].blockSignals(False)

    def _mark_class_modified(self, class_code):
        self._set_class_modified(class_code)
        self._workspace = class_workspace.save_workspace(self._run_spec, self._workspace)
        self._refresh_table()

    @staticmethod
    def _manual_geometry_error(geometry):
        if geometry is None or geometry.isNull() or geometry.isEmpty():
            return "几何为空"
        if geometry.type() != Qgis.GeometryType.Polygon:
            return "结果不是 Polygon"
        if geometry.area() <= 0:
            return "面积必须大于 0"
        if not geometry.isGeosValid():
            return "几何无效或存在自相交"
        return ""

    def _save_class_edits(self, class_code):
        self._select_class_context(class_code, activate_layer=True)
        layer = self._layer(class_code)
        if not layer.isEditable() or not layer.isModified():
            QMessageBox.information(self, "保存 QGIS 编辑", "当前类别没有未保存编辑")
            return
        if not layer.commitChanges():
            errors = "; ".join(layer.commitErrors())
            QMessageBox.warning(self, "保存 QGIS 编辑失败", errors)
            return
        self.baseline_label.setText(
            f"类别 {class_code} {CLASS_NAMES[class_code]} 的 QGIS 编辑已保存"
        )
        self._update_actions()

    def _rollback_class_edits(self, class_code):
        self._select_class_context(class_code, activate_layer=True)
        layer = self._layer(class_code)
        if not layer.isEditable():
            QMessageBox.information(self, "放弃 QGIS 编辑", "当前类别没有编辑会话")
            return
        if layer.isModified():
            answer = QMessageBox.question(
                self,
                "放弃 QGIS 编辑",
                "放弃当前类别尚未保存的修改？",
                YES | NO,
                NO,
            )
            if answer != YES:
                return
        layer.rollBack()
        self.baseline_label.setText(
            f"类别 {class_code} {CLASS_NAMES[class_code]} 的 QGIS 编辑已放弃"
        )
        self._update_actions()

    @staticmethod
    def _object_id_exists(layer, object_id):
        wanted = str(object_id or "")
        return any(
            str(feature.attribute("object_id") or "") == wanted
            for feature in layer.getFeatures()
        )

    def _sam_available(self):
        return bool(
            self._sam_config.get("enabled")
            and self._sam_config.get("checkpoint_sha256")
            and Path(str(self._sam_config.get("checkpoint") or "")).is_file()
        )

    def _begin_sam(self, class_code, missed=False):
        self._select_class_context(class_code, activate_layer=True)
        if not self._workspace:
            return
        if not self._sam_available():
            QMessageBox.warning(self, "SAM3", "SAM3 checkpoint、SHA 或环境检查未通过")
            return
        if self._active_session or self._manual_task:
            QMessageBox.warning(self, "SAM3", "已有活动 SAM3 或人工操作任务")
            return
        if self._editable_modified_layers():
            QMessageBox.warning(self, "SAM3", "请先保存或回滚所有类别工作层编辑")
            return
        layer = self._layer(class_code)
        if not missed and layer.featureCount() == 0:
            QMessageBox.information(self, "SAM3", "该类别为空；请使用“新增漏标面”")
            return
        canvas = self.iface.mapCanvas()
        self._previous_map_tool = canvas.mapTool()
        self._point_tool = QgsMapToolEmitPoint(canvas)
        self._point_tool.canvasClicked.connect(self._sam_map_clicked)
        self._active_session = {
            "session_id": uuid.uuid4().hex,
            "class_code": class_code,
            "mode": "missed" if missed else "existing",
            "state": "waiting_click",
            "started_at": class_workspace._now(),
        }
        self._workspace["active_sam_session_id"] = self._active_session["session_id"]
        self._workspace = class_workspace.save_workspace(self._run_spec, self._workspace)
        self.session_group.show()
        self.session_label.setText(
            f"当前类别: {class_code} {CLASS_NAMES[class_code]} | "
            + ("点击漏标地物位置" if missed else "点击当前类别工作层中的一个已有面")
        )
        self._set_decision_state("waiting_click")
        canvas.setMapTool(self._point_tool)
        self._update_actions()

    def _restore_map_tool(self):
        canvas = self.iface.mapCanvas()
        tool = self._point_tool
        if tool is not None and canvas.mapTool() == tool:
            if self._previous_map_tool is not None:
                canvas.setMapTool(self._previous_map_tool)
            else:
                canvas.unsetMapTool(tool)
        self._point_tool = None
        self._previous_map_tool = None
        self._manual_pick_code = None

    def _features_at_map_point(self, layer, map_point):
        canvas = self.iface.mapCanvas()
        canvas_crs = canvas.mapSettings().destinationCrs()
        to_layer = QgsCoordinateTransform(
            canvas_crs, layer.crs(), QgsProject.instance()
        )
        canvas_point = QgsPointXY(map_point)
        layer_point = to_layer.transform(canvas_point)
        point_geometry = QgsGeometry.fromPointXY(layer_point)
        canvas_tolerance = canvas.mapUnitsPerPixel() * 2
        offset_layer_point = to_layer.transform(QgsPointXY(
            canvas_point.x() + canvas_tolerance,
            canvas_point.y(),
        ))
        tolerance = max(
            1e-12,
            math.hypot(
                offset_layer_point.x() - layer_point.x(),
                offset_layer_point.y() - layer_point.y(),
            ),
        )
        request = QgsFeatureRequest().setFilterRect(QgsRectangle(
            layer_point.x() - tolerance,
            layer_point.y() - tolerance,
            layer_point.x() + tolerance,
            layer_point.y() + tolerance,
        ))
        return [
            feature for feature in layer.getFeatures(request)
            if feature.geometry().contains(point_geometry)
            or feature.geometry().intersects(point_geometry)
        ]

    def _sam_map_clicked(self, map_point, _button):
        session = self._active_session
        if not session or session.get("state") != "waiting_click":
            return
        class_code = session["class_code"]
        layer = self._layer(class_code)
        canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
        try:
            hits = self._features_at_map_point(layer, map_point)
            feature = None
            if session["mode"] == "existing":
                if len(hits) != 1:
                    raise RuntimeError(
                        "点击必须唯一命中当前类别工作层中的一个面；未命中或重叠时不按最近距离猜测"
                    )
                feature = hits[0]
                layer.selectByIds([feature.id()])
            raster_crs = QgsCoordinateReferenceSystem(self._run_spec["raster"]["crs"])
            to_raster = QgsCoordinateTransform(canvas_crs, raster_crs, QgsProject.instance())
            raster_point = to_raster.transform(QgsPointXY(map_point))
            bounds = None
            if feature is not None:
                geometry_raster = QgsGeometry(feature.geometry())
                geometry_raster.transform(
                    QgsCoordinateTransform(layer.crs(), raster_crs, QgsProject.instance())
                )
                rectangle = geometry_raster.boundingBox()
                bounds = {
                    "xmin": rectangle.xMinimum(), "ymin": rectangle.yMinimum(),
                    "xmax": rectangle.xMaximum(), "ymax": rectangle.yMaximum(),
                }
                session.update({
                    "feature_id": feature.id(),
                    "object_id": str(feature.attribute("object_id") or ""),
                    "part_id": str(feature.attribute("part_id") or "000"),
                    "current_geometry_hash": class_workspace.geometry_hash(feature.geometry()),
                    "current_source": str(feature.attribute("geometry_source") or "fusion"),
                    "current_revision": int(feature.attribute("geometry_revision") or 0),
                })
                self._show_current_geometry(feature.geometry(), layer)
            else:
                session.update({"object_id": "", "part_id": "000"})
            session.update({
                "state": "inference",
                "click_raster": {"x": raster_point.x(), "y": raster_point.y()},
                "geometry_bounds": bounds,
            })
            self._restore_map_tool()
            self._set_decision_state("inference")
            self.session_label.setText(
                f"当前类别: {class_code} {CLASS_NAMES[class_code]} | "
                f"object_id: {session.get('object_id') or '新增'} | "
                f"当前来源: {session.get('current_source') or '新增'} | SAM3 推理中"
            )
            self._start_worker_request()
        except Exception as exc:
            self._restore_map_tool()
            self._session_failed(str(exc))

    def _start_worker_request(self):
        session = self._active_session
        stream = class_workspace.stream_by_id(
            self._eligible_fusions, self._workspace["baseline_stream_id"]
        )
        self._worker_request = {
            "session_id": session["session_id"],
            "run_id": self._run_spec["run_id"],
            "raster": self._run_spec["raster"]["path"],
            "confidence_mosaic": (stream.get("paths") or {}).get("confidence_mosaic", ""),
            "click_raster": session["click_raster"],
            "geometry_bounds": session.get("geometry_bounds"),
            "crop_size_px": 512,
            "buffer_px": int(self._sam_config.get("buffer_px", 32)),
            "class_code": session["class_code"],
            "object_id": session.get("object_id", ""),
            "part_id": session.get("part_id", "000"),
            "checkpoint_sha256": str(self._sam_config.get("checkpoint_sha256") or ""),
            "sam_version": str(self._sam_config.get("version") or ""),
            "device": str(
                self._sam_config.get("effective_device")
                or self._sam_config.get("requested_device")
                or "cpu"
            ),
        }
        if self._worker is None:
            self._worker = Sam3WorkerRunner(
                self._scripts_dir, self._sam_config, self
            )
            self._worker.ready.connect(self._worker_ready)
            self._worker.event_received.connect(self._worker_event)
            self._worker.log_line.connect(self._worker_log)
            self._worker.stopped.connect(self._worker_stopped)
        if self._worker.is_ready:
            self._worker.predict(self._worker_request)
            self._worker_request = None
        else:
            self._worker.start()

    def _worker_ready(self, _event):
        if self._worker_request and self._active_session:
            request = self._worker_request
            self._worker_request = None
            self._worker.predict(request)

    def _worker_event(self, event):
        session = self._active_session
        if not session:
            return
        if event.get("session_id") and event.get("session_id") != session["session_id"]:
            return
        if event.get("event") == "candidate_ready":
            geometry = QgsGeometry.fromWkt(str(event.get("geometry_wkt") or ""))
            if geometry.isNull() or geometry.isEmpty() or not geometry.isGeosValid():
                self._session_failed("SAM3 返回了无效候选几何")
                return
            raster_crs = QgsCoordinateReferenceSystem(self._run_spec["raster"]["crs"])
            layer = self._layer(session["class_code"])
            geometry.transform(
                QgsCoordinateTransform(raster_crs, layer.crs(), QgsProject.instance())
            )
            session.update({
                "state": "candidate",
                "candidate_geometry": geometry,
                "candidate_score": float(event.get("score") or 0.0),
                "confidence_mean": float(event.get("confidence_mean") or 0.0),
                "confidence_std": float(event.get("confidence_std") or 0.0),
                "crop_window": event.get("crop_window"),
                "elapsed_sec": event.get("elapsed_sec"),
            })
            topology_hint = self._local_topology_hint(
                session["class_code"], geometry, session.get("feature_id")
            )
            session["topology_hint"] = topology_hint
            self._show_candidate_geometry(geometry, layer)
            self.session_label.setText(
                f"当前类别: {session['class_code']} {CLASS_NAMES[session['class_code']]} | "
                f"object_id: {session.get('object_id') or '新增'} | "
                f"当前来源: {session.get('current_source') or '新增'} | "
                f"候选 score: {session['candidate_score']:.4f}"
            )
            self.local_topology_label.setText(f"局部拓扑提示: {topology_hint}")
            self._set_decision_state("candidate")
        elif event.get("event") == "failed":
            self._session_failed(str(event.get("error") or "SAM3 未知错误"))

    def _worker_log(self, level, message):
        if level == "stderr" and self._active_session:
            self.session_error.setPlainText(message[-6000:])

    def _worker_stopped(self, event):
        if not event.get("expected") and self._active_session:
            self._session_failed(
                f"SAM3 worker 意外退出，returncode={event.get('returncode')}"
            )

    def _session_failed(self, error):
        if not self._active_session:
            return
        self._active_session["state"] = "failed"
        self._active_session["error"] = error
        self.session_label.setText("SAM3 失败；当前工作层未修改")
        self.local_topology_label.setText("局部拓扑提示: 未执行")
        self.session_error.setPlainText(error)
        self.session_error.show()
        self._set_decision_state("failed")

    def _show_current_geometry(self, geometry, layer):
        self._clear_bands()
        self._current_band = QgsRubberBand(
            self.iface.mapCanvas(), Qgis.GeometryType.Polygon
        )
        self._current_band.setStrokeColor(QColor("#ffd400"))
        self._current_band.setFillColor(QColor(255, 212, 0, 30))
        self._current_band.setWidth(2)
        self._current_band.setToGeometry(geometry, layer)

    def _show_candidate_geometry(self, geometry, layer):
        self._candidate_band = QgsRubberBand(
            self.iface.mapCanvas(), Qgis.GeometryType.Polygon
        )
        self._candidate_band.setStrokeColor(QColor("#00d7d7"))
        self._candidate_band.setFillColor(QColor(0, 215, 215, 35))
        self._candidate_band.setWidth(2)
        self._candidate_band.setLineStyle(DASH_LINE)
        self._candidate_band.setToGeometry(geometry, layer)

    def _local_topology_hint(self, class_code, geometry, replaced_feature_id=None):
        if geometry is None or geometry.isNull() or geometry.isEmpty():
            return "候选为空"
        if not geometry.isGeosValid():
            return "候选几何无效"
        tolerance = topology_validator.pixel_area_tolerance(self._run_spec)
        same_class = 0
        cross_class = 0
        overlap_area = 0.0
        source_layer = self._layer(class_code)
        for other_code in CLASS_ORDER:
            layer = self._layer(other_code)
            candidate = QgsGeometry(geometry)
            if source_layer.crs() != layer.crs():
                candidate.transform(
                    QgsCoordinateTransform(
                        source_layer.crs(), layer.crs(), QgsProject.instance()
                    )
                )
            request = QgsFeatureRequest().setFilterRect(candidate.boundingBox())
            for feature in layer.getFeatures(request):
                if other_code == class_code and feature.id() == replaced_feature_id:
                    continue
                intersection = candidate.intersection(feature.geometry())
                if (
                    intersection is None
                    or intersection.isNull()
                    or intersection.isEmpty()
                    or intersection.area() <= tolerance
                ):
                    continue
                overlap_area += intersection.area()
                if other_code == class_code:
                    same_class += 1
                else:
                    cross_class += 1
        if not same_class and not cross_class:
            return "无"
        return (
            f"同类重叠 {same_class} 处，邻类重叠 {cross_class} 处，"
            f"总面积 {overlap_area:.8g}"
        )

    def _clear_bands(self):
        scene = self.iface.mapCanvas().scene()
        for band in (self._current_band, self._candidate_band):
            if band is not None:
                band.reset(Qgis.GeometryType.Polygon)
                scene.removeItem(band)
        self._current_band = None
        self._candidate_band = None

    def _set_decision_state(self, state):
        candidate = state == "candidate"
        failed = state == "failed"
        existing = bool(
            self._active_session
            and self._active_session.get("mode") == "existing"
        )
        self.keep_btn.setEnabled(candidate and existing)
        self.adopt_btn.setEnabled(candidate)
        self.edit_current_btn.setEnabled((candidate or failed) and existing)
        self.edit_sam_btn.setEnabled(candidate)
        self.retry_btn.setEnabled(failed)
        self.cancel_session_btn.setEnabled(state != "idle")

    def _retry_session(self):
        session = self._active_session
        if not session or session.get("state") != "failed" or not session.get("click_raster"):
            return
        if self._worker:
            old_session_id = session["session_id"]
            self._worker.cancel(old_session_id)
            self._worker.close_session(old_session_id)
        self._record_session("failed")
        session["session_id"] = uuid.uuid4().hex
        session["state"] = "inference"
        session["started_at"] = class_workspace._now()
        session.pop("error", None)
        self._workspace["active_sam_session_id"] = session["session_id"]
        self._workspace = class_workspace.save_workspace(self._run_spec, self._workspace)
        self.session_error.hide()
        self.session_error.clear()
        self._set_decision_state("inference")
        self.session_label.setText("SAM3 正在重试；当前工作层未修改")
        self._start_worker_request()

    def _finish_session(self, decision):
        session = self._active_session
        if not session:
            return
        if decision in ("adopted", "edit_sam3") and session.get("state") != "candidate":
            return
        if decision == "kept_current" and session.get("state") != "candidate":
            return
        try:
            if decision == "adopted":
                self._adopt_candidate(session, edit=False)
            elif decision == "edit_sam3":
                self._adopt_candidate(session, edit=True)
            elif decision == "edit_current":
                if session["mode"] == "missed":
                    raise RuntimeError("新增漏标面没有当前几何可编辑")
                self._start_session_edit_current(session)
            if self._worker and decision == "cancelled":
                self._worker.cancel(session["session_id"])
            self._record_session(decision)
        except Exception as exc:
            QMessageBox.warning(self, "SAM3 决定失败", str(exc))
            return
        self._cancel_active_session(record=False, keep_edit=decision in ("edit_current", "edit_sam3"))
        self._refresh_table()

    def _adopt_candidate(self, session, edit):
        layer = self._layer(session["class_code"])
        self.iface.setActiveLayer(layer)
        before_snapshot = self._snapshot(layer)
        confidence_mean, confidence_std, confidence_warning = (
            self._optional_confidence_statistics(
                layer, session["candidate_geometry"]
            )
        )
        self._metadata_update = True
        try:
            if not layer.isEditable() and not layer.startEditing():
                raise RuntimeError("无法启动类别工作层编辑")
            if session["mode"] == "existing":
                feature_id = session["feature_id"]
                layer.changeGeometry(feature_id, session["candidate_geometry"])
                values = {
                    "geometry_source": "manual_edited" if edit else "sam3",
                    "geometry_revision": session["current_revision"] + 1,
                    "edit_base": "sam3" if edit else "",
                    "sam_session_id": session["session_id"],
                    "sam_score": session["candidate_score"],
                    "sam_version": str(self._sam_config.get("version") or ""),
                    "confidence_mean": confidence_mean,
                    "confidence_std": confidence_std,
                    "reviewed": 0,
                    "updated_at": class_workspace._now(),
                }
                self._set_attributes(layer, feature_id, values)
            else:
                feature = QgsFeature(layer.fields())
                feature.setGeometry(session["candidate_geometry"])
                object_id = class_workspace.new_object_id(self._run_spec)
                values = {
                    "run_id": self._run_spec["run_id"],
                    "object_id": object_id,
                    "part_id": "000",
                    "class_code": session["class_code"],
                    "class_name": CLASS_NAMES[session["class_code"]],
                    "baseline_stream_id": self._workspace["baseline_stream_id"],
                    "geometry_source": "manual_edited" if edit else "sam3",
                    "geometry_revision": 1,
                    "edit_base": "sam3" if edit else "",
                    "sam_session_id": session["session_id"],
                    "sam_score": session["candidate_score"],
                    "sam_version": str(self._sam_config.get("version") or ""),
                    "confidence_mean": confidence_mean,
                    "confidence_std": confidence_std,
                    "reviewed": 0,
                    "updated_at": class_workspace._now(),
                }
                for name, value in values.items():
                    if layer.fields().indexOf(name) >= 0:
                        feature.setAttribute(name, value)
                if not layer.addFeature(feature):
                    raise RuntimeError("无法向类别工作层新增 SAM3 面")
                feature_id = feature.id()
                session["object_id"] = object_id
                session["feature_id"] = feature_id
            if edit:
                self._metadata_update = False
                self._snapshots[session["class_code"]] = before_snapshot
                self._edit_context[session["class_code"]] = {
                    "metadata_prepared": True,
                    "session_id": session["session_id"],
                }
                action = getattr(self.iface, "actionVertexTool", lambda: None)()
                if action is not None:
                    action.trigger()
                return
            if not layer.commitChanges():
                errors = "; ".join(layer.commitErrors())
                layer.rollBack()
                raise RuntimeError(f"无法保存 SAM3 候选: {errors}")
        finally:
            self._metadata_update = False
        persisted = self._feature_by_object_id(layer, session["object_id"])
        session["feature_id"] = persisted.id()
        session["persisted_geometry_hash"] = class_workspace.geometry_hash(
            persisted.geometry()
        )
        self._mark_class_modified(session["class_code"])
        if confidence_warning:
            class_workspace.append_history(
                self._run_spec,
                "confidence_statistics_unavailable",
                class_code=session["class_code"],
                object_id=session["object_id"],
                reason=confidence_warning,
            )
        class_workspace.append_history(
            self._run_spec,
            "sam3_adopted" if session["mode"] == "existing" else "sam3_feature_added",
            class_code=session["class_code"],
            object_id=session["object_id"],
            sam_session_id=session["session_id"],
            before_geometry_hash=session.get("current_geometry_hash", ""),
            after_geometry_hash=session["persisted_geometry_hash"],
        )
        self.baseline_label.setText(
            f"SAM3 已采用；局部拓扑提示: {session.get('topology_hint') or '无'}"
        )

    def _start_session_edit_current(self, session):
        layer = self._layer(session["class_code"])
        self.iface.setActiveLayer(layer)
        if not layer.isEditable() and not layer.startEditing():
            raise RuntimeError("无法启动当前工作层编辑")
        layer.selectByIds([session["feature_id"]])
        self._edit_context[session["class_code"]] = {
            "metadata_prepared": False,
            "session_id": session["session_id"],
        }
        action = getattr(self.iface, "actionVertexTool", lambda: None)()
        if action is not None:
            action.trigger()

    def _record_session(self, decision):
        session = dict(self._active_session or {})
        candidate_geometry = session.pop("candidate_geometry", None)
        before_hash = str(session.get("current_geometry_hash") or "")
        candidate_hash = (
            class_workspace.geometry_hash(candidate_geometry)
            if candidate_geometry is not None else ""
        )
        persisted_hash = str(session.get("persisted_geometry_hash") or "")
        session.update({
            "decision": decision,
            "run_id": self._run_spec.get("run_id"),
            "baseline_stream_id": (self._workspace or {}).get("baseline_stream_id", ""),
            "before_geometry_hash": before_hash,
            "candidate_geometry_hash": candidate_hash,
            "after_geometry_hash": (
                persisted_hash if decision == "adopted"
                else candidate_hash if decision == "edit_sam3"
                else before_hash
            ),
            "checkpoint_sha256": str(self._sam_config.get("checkpoint_sha256") or ""),
            "sam_version": str(self._sam_config.get("version") or ""),
            "device": str(
                self._sam_config.get("effective_device")
                or self._sam_config.get("requested_device")
                or "cpu"
            ),
        })
        class_workspace.append_sam_session(self._run_spec, session)

    def _cancel_active_session(self, record=True, keep_edit=False):
        if self._active_session and record:
            self._record_session("cancelled")
        if self._active_session and self._worker:
            session_id = self._active_session["session_id"]
            self._worker.cancel(session_id)
            self._worker.close_session(session_id)
        self._restore_map_tool()
        self._clear_bands()
        self._active_session = None
        self._worker_request = None
        if self._workspace is not None:
            self._workspace["active_sam_session_id"] = ""
            self._workspace = class_workspace.save_workspace(
                self._run_spec, self._workspace
            )
        self.session_group.hide()
        self.local_topology_label.setText("局部拓扑提示: -")
        self.session_error.hide()
        self.session_error.clear()
        self._set_decision_state("idle")
        self._update_actions()

    def _confirm_class(self, class_code, checked):
        if self._metadata_update or not self._workspace:
            return
        self._select_class_context(class_code, activate_layer=True)
        layer = self._layer(class_code)
        if checked and (
            self._active_session
            or self._manual_task
            or (layer.isEditable() and layer.isModified())
        ):
            self._confirm_buttons[class_code].blockSignals(True)
            self._confirm_buttons[class_code].setChecked(False)
            self._confirm_buttons[class_code].blockSignals(False)
            QMessageBox.warning(
                self,
                "确认类别",
                "存在活动 SAM3、人工操作或未保存编辑，不能确认",
            )
            return
        record = self._workspace["classes"][str(class_code)]
        keep_editing = layer.isEditable()
        self._metadata_update = True
        try:
            if not keep_editing and not layer.startEditing():
                raise RuntimeError("无法更新 reviewed 字段")
            reviewed_index = layer.fields().indexOf("reviewed")
            updated_index = layer.fields().indexOf("updated_at")
            for feature in layer.getFeatures():
                layer.changeAttributeValue(feature.id(), reviewed_index, 1 if checked else 0)
                layer.changeAttributeValue(feature.id(), updated_index, class_workspace._now())
            if not layer.commitChanges(not keep_editing):
                errors = "; ".join(layer.commitErrors())
                layer.rollBack()
                raise RuntimeError(errors)
        except Exception as exc:
            QMessageBox.warning(self, "确认类别失败", str(exc))
            return
        finally:
            self._metadata_update = False
        record["confirmed"] = bool(checked)
        record["state"] = (
            "confirmed_empty" if checked and layer.featureCount() == 0
            else "confirmed" if checked
            else "editing" if layer.featureCount()
            else "unreviewed_empty"
        )
        class_workspace.append_history(
            self._run_spec, "class_confirmed" if checked else "class_reopened",
            class_code=class_code, feature_count=layer.featureCount(),
        )
        self._workspace = class_workspace.save_workspace(self._run_spec, self._workspace)
        self._refresh_table()

    def _refresh_table(self):
        if not self._workspace:
            for row, code in enumerate(CLASS_ORDER):
                self.table.item(row, 3).setText("未初始化")
                self.table.item(row, 4).setText("0")
                self._sam_buttons[code].setEnabled(False)
                self._edit_buttons[code].setEnabled(False)
                self._confirm_buttons[code].setEnabled(False)
            self._update_actions()
            return
        has_unsaved_edits = bool(self._editable_modified_layers())
        for row, code in enumerate(CLASS_ORDER):
            record = self._workspace["classes"][str(code)]
            layer = self._layer(code)
            stats = class_workspace.source_statistics(layer)
            if self._manual_only:
                status_text = (
                    f"原始 {stats.get('fusion', 0)} / "
                    f"人工 {stats.get('manual_edited', 0)}"
                )
            else:
                status_text = (
                    f"Fusion {stats.get('fusion', 0)} / SAM3 {stats.get('sam3', 0)} / "
                    f"人工 {stats.get('manual_edited', 0)}"
                )
            self.table.item(row, 3).setText(status_text)
            self.table.item(row, 4).setText(str(layer.featureCount()))
            sam_enabled = bool(
                not self._manual_only
                and self._sam_available()
                and not self._active_session
                and not self._manual_task
            )
            self._sam_buttons[code].setEnabled(sam_enabled)
            self._sam_existing_actions[code].setEnabled(
                sam_enabled and layer.featureCount() > 0
            )
            self._sam_missed_actions[code].setEnabled(sam_enabled)
            self._sam_buttons[code].setToolTip(
                "" if sam_enabled else "SAM3 不可用、会话活动或环境未通过"
            )
            manual_enabled = not self._active_session
            self._edit_buttons[code].setEnabled(manual_enabled)
            confirm = self._confirm_buttons[code]
            confirm.blockSignals(True)
            confirm.setChecked(bool(record.get("confirmed")))
            confirm.setText(
                "整类已确认" if record.get("confirmed")
                else "确认本范围无该类" if layer.featureCount() == 0
                else "确认整类"
            )
            confirm.blockSignals(False)
            confirm.setEnabled(
                not self._active_session
                and not self._manual_task
                and not has_unsaved_edits
            )
        self._update_manual_panel()
        self._update_actions()
        self.workspace_changed.emit(dict(self._workspace))

    def _editable_modified_layers(self):
        modified = []
        for code in CLASS_ORDER:
            if code not in self._class_layers:
                continue
            layer = self._layer(code)
            if layer.isEditable() and layer.isModified():
                modified.append(code)
        return modified

    def _update_actions(self):
        confirmed = 0
        if self._workspace:
            confirmed = sum(
                1 for record in self._workspace["classes"].values()
                if record.get("confirmed")
            )
        modified = self._editable_modified_layers() if self._workspace else []
        can_assemble = bool(
            self._workspace
            and confirmed == 14
            and not modified
            and not self._active_session
            and not self._manual_task
        )
        self.assemble_btn.setEnabled(can_assemble)
        issue_text = "-" if self._issue_count is None else str(self._issue_count)
        self.summary_label.setText(
            f"14 类确认: {confirmed}/14    未解决问题: {issue_text}    "
            f"未保存编辑: {len(modified)}"
        )

    def _assemble_final(self):
        try:
            path, count = final_assembler.assemble_final(
                self._run_spec, self._workspace
            )
            self._final_path = path
            self.allow_issues_check.setChecked(False)
            self.layer_manager.load_final_composite(self._run_spec["run_id"], path)
            self.topology_btn.setEnabled(True)
            self.baseline_label.setText(f"final_composite 已组装，共 {count} 个面")
            self._check_topology()
        except Exception as exc:
            QMessageBox.warning(self, "组装失败", str(exc))

    def _accepted_layer_for_check(self):
        path = str(self._run_spec.get("accepted_target_gpkg") or "")
        if not path or not Path(path).is_file():
            return None
        layer = QgsVectorLayer(
            f"{path}|layername={LAYER_NAMES.ACCEPTED}", "accepted_for_topology", "ogr"
        )
        if not layer.isValid():
            raise ValueError(f"无法打开长期 accepted_labels: {path}")
        return layer

    def _check_topology(self):
        if not self._final_path:
            return
        try:
            path, count, counts = topology_validator.validate_topology(
                self._run_spec, self._final_path, self._accepted_layer_for_check()
            )
            self._issues_path = path
            self._issue_count = count
            self.layer_manager.load_topology_issues(self._run_spec["run_id"], path)
            self._update_accept_enabled()
            self.baseline_label.setText(
                "拓扑检查完成: " + (", ".join(f"{key}={value}" for key, value in counts.items()) or "无问题")
            )
            self._update_actions()
        except Exception as exc:
            self._issue_count = None
            self.accept_btn.setEnabled(False)
            QMessageBox.warning(self, "拓扑检查失败", str(exc))

    def _write_accepted(self):
        if self._issue_count is None:
            QMessageBox.warning(self, "入库", "必须先完成拓扑检查")
            return
        if self._issue_count != 0 and not self.allow_issues_check.isChecked():
            QMessageBox.warning(self, "入库", "请先解决拓扑问题，或明确勾选带问题入库")
            return
        try:
            count = accepted_writer.append_final_to_accepted(
                self._final_path,
                str(self._run_spec.get("accepted_target_gpkg") or ""),
                str(
                    self._run_spec.get("accepted_write_manifest")
                    or Path(self._run_spec["run_dir"]) / "run_manifest.json"
                ),
            )
            self.baseline_label.setText(f"已写入 accepted_labels: {count} 个面")
        except Exception as exc:
            QMessageBox.warning(self, "写入 accepted_labels 失败", str(exc))

    def _update_accept_enabled(self, *_args):
        self.accept_btn.setEnabled(
            bool(
                self._final_path
                and self._issue_count is not None
                and (
                    self._issue_count == 0
                    or self.allow_issues_check.isChecked()
                )
            )
        )

    def cleanup(self):
        self._clear_qgis_smooth_preview()
        self._manual_smoothing_timer.stop()
        self._cancel_manual_capture_transition()
        self._cancel_manual_task(silent=True)
        self._cancel_active_session(record=False)
        self._disconnect_layer_signals()
        self._disconnect_iface_layer_signal()
        self._disconnect_map_tool_signal()
        self._snapshots.clear()
        self._manual_capture_retire_timer.stop()
        self._manual_retired_capture_tools = []
        if self._worker is not None:
            self._worker.stop()
            self._worker = None

    def closeEvent(self, event):
        if self._editable_modified_layers():
            QMessageBox.warning(
                self,
                "关闭分类修整",
                "存在未保存编辑，请先在 QGIS 中保存或回滚后再关闭。",
            )
            event.ignore()
            return
        if self._manual_task:
            answer = QMessageBox.question(
                self,
                "关闭分类修整",
                "当前人工操作尚未结束。是否取消当前候选并关闭窗口？\n"
                "新增任务中已提交批次会继续保留，当前未保存队列会被丢弃。",
                YES | NO,
                NO,
            )
            if answer != YES:
                event.ignore()
                return
            self._cancel_manual_task(silent=True)
        if self._active_session:
            answer = QMessageBox.question(
                self,
                "关闭分类修整",
                "当前 SAM3 会话尚未完成。是否取消本次会话并关闭窗口？",
                YES | NO,
                NO,
            )
            if answer != YES:
                event.ignore()
                return
        self._cancel_active_session(record=True)
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
        self._clear_qgis_smooth_preview()
        event.ignore()
        self.hide()
