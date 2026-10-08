import logging
import os
from pathlib import Path

from qgis.PyQt.QtCore import QTimer
from qgis.PyQt.QtWidgets import (
    QApplication, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QFormLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox,
    QRadioButton, QCheckBox, QProgressBar,
    QWidget, QButtonGroup, QFileDialog, QMessageBox, QScrollArea,
)
from qgis.gui import QgsDockWidget, QgsMapLayerComboBox
from qgis.core import (
    Qgis,
    QgsApplication,
    QgsProject, QgsVectorLayer,
    QgsSettings,
)

from labeling_tool.qgis_support.layer_names import LAYER_NAMES
from labeling_tool.qgis_support.qt6_api import (
    ALIGN_LEFT,
    ALIGN_VCENTER,
    CLOSE,
    CRITICAL,
    INFORMATION,
    MENU_SCROLLER_HEIGHT,
    NO,
    NON_MODAL,
    SCROLLBAR_AS_NEEDED,
    TEXT_SELECTABLE_BY_MOUSE,
    WA_DELETE_ON_CLOSE,
    WARNING,
    YES,
)


class _ScreenBoundMapLayerComboBox(QgsMapLayerComboBox):
    """Keep long layer lists scrollable and inside the active screen."""

    MAX_VISIBLE_ITEMS = 15

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaxVisibleItems(self.MAX_VISIBLE_ITEMS)
        self.view().setVerticalScrollBarPolicy(SCROLLBAR_AS_NEEDED)

    def _visible_rows_height(self):
        view = self.view()
        visible_rows = min(self.count(), self.MAX_VISIBLE_ITEMS)
        if visible_rows < 1:
            return 0

        sampled_rows = list(range(visible_rows))
        current_row = self.currentIndex()
        if current_row >= visible_rows:
            sampled_rows.append(current_row)
        row_height = max(
            (view.sizeHintForRow(row) for row in sampled_rows),
            default=-1,
        )
        if row_height < 1:
            row_height = view.fontMetrics().height() + 8
        return row_height * visible_rows

    def _popup_height_limit(self):
        view = self.view()
        rows_height = self._visible_rows_height()
        if rows_height < 1:
            return 0
        popup = view.window()
        margins = popup.contentsMargins()
        menu_scroller_height = self.style().pixelMetric(
            MENU_SCROLLER_HEIGHT, None, self
        )
        return (
            rows_height
            + 2 * view.frameWidth()
            + margins.top()
            + margins.bottom()
            + 2 * menu_scroller_height
        )

    def showPopup(self):
        view = self.view()
        popup = view.window()
        height_limit = self._popup_height_limit()
        if height_limit > 0:
            popup.setMaximumHeight(height_limit)

        anchor = self.mapToGlobal(self.rect().bottomLeft())
        screen = QApplication.screenAt(anchor) or self.screen()
        if screen is not None:
            popup.setMaximumWidth(screen.availableGeometry().width())

        super().showPopup()

        if screen is None:
            return
        available = screen.availableGeometry()
        popup_size = popup.frameGeometry().size()
        combo_top = self.mapToGlobal(self.rect().topLeft())
        below_y = anchor.y()
        above_y = combo_top.y() - popup_size.height()
        if below_y + popup_size.height() <= available.bottom() + 1:
            popup_y = below_y
        elif above_y >= available.top():
            popup_y = above_y
        else:
            popup_y = available.top()

        maximum_x = available.right() - popup_size.width() + 1
        popup_x = max(available.left(), min(anchor.x(), maximum_x))
        maximum_y = available.bottom() - popup_size.height() + 1
        popup_y = max(available.top(), min(popup_y, maximum_y))
        popup.move(popup_x, popup_y)

from labeling_tool.qgis_support import tile_manager
from labeling_tool.refinement.background_io_tasks import ManualRunLoadTask
from labeling_tool.main.inference_config import InferenceConfigManager
from labeling_tool.qgis_support.layer_manager import LayerManager

from labeling_tool.monitor.inference_monitor import InferenceMonitorDialog
from labeling_tool.main.inference_plan_panel import (
    InferencePlanPanel,
    InferenceSelection,
    resolve_launch_plan,
)
from labeling_tool.refinement.class_refinement_dialog import ClassRefinementDialog
from labeling_tool.main.range_preview import (
    TileParameters,
    TileGridSummary,
    VectorPreviewController,
    VectorPreviewResult,
)
from labeling_tool.main.range_selection import (
    RECTANGLE_MODE,
    VECTOR_MODE,
    VIEW_MODE,
    RangeSelectionController,
    RawRangeSelection,
    format_extent,
    intersect_extents,
    is_valid_extent,
    range_selection_metadata,
    resolve_raster_extent,
    transform_extent,
    validate_raster_layer,
    validate_vector_layer,
)
from labeling_tool.qgis_support.qt_lifecycle import retire_after
from labeling_tool.runs.run_workflow import (
    RunStartRequest,
    RunWorkflowController,
)
from labeling_tool.runs import run_index
from labeling_tool.main.environment_report import (
    first_problem,
)
from labeling_tool.main.environment_panel import EnvironmentPanel
from labeling_tool.main.start_readiness import derive_start_readiness
from labeling_tool.runs.spatial_planner import plan_spatial_units

logger = logging.getLogger("labeling_tool.main_dock")


class LabelingDockWidget(QgsDockWidget):

    def __init__(self, parent=None, iface=None):
        super().__init__(parent)
        self.iface = iface
        self.layer_manager = LayerManager(iface) if iface else None
        self.config_manager = InferenceConfigManager(self)
        self.workflow = RunWorkflowController(self)
        self._cleaning_up = False
        self.monitor_dialog = InferenceMonitorDialog(self)
        self.monitor_dialog.stop_requested.connect(self._on_stop)
        self.monitor_dialog.request_main_run_handling.connect(
            self._on_monitor_main_run_handling
        )
        self.workflow.state_changed.connect(self._on_workflow_state_changed)
        self.workflow.stage_progress.connect(self._apply_stage_progress)
        self.workflow.runner_changed.connect(self._on_workflow_runner_changed)
        self.workflow.monitor_context_ready.connect(
            self._on_workflow_monitor_context_ready
        )
        self.workflow.pre_run_failed.connect(self._on_workflow_pre_run_failed)
        self.workflow.stopped_before_run.connect(
            self._on_workflow_stopped_before_run
        )
        self.workflow.finished.connect(self._on_pipeline_finished)
        self.refinement_dialog = ClassRefinementDialog(
            self.iface, self.layer_manager, self
        ) if self.iface and self.layer_manager else None
        self.range_selection = RangeSelectionController(iface, self)
        self.range_selection.rectangle_finished.connect(self._on_rect_finished)
        self.vector_preview = VectorPreviewController(self)
        self.vector_preview.progress_changed.connect(
            self._on_vector_preview_progress
        )
        self.vector_preview.preview_ready.connect(self._on_vector_preview_ready)
        self.vector_preview.preview_failed.connect(self._on_vector_preview_failed)
        self.vector_preview.source_changed.connect(
            self._on_vector_range_data_changed
        )
        self.vector_preview.auto_start_ready.connect(self._on_start)
        self._last_run_result = None
        self._last_run_spec = None
        self._recovery_run_spec = None
        self._startup_ready_candidate = None
        self._startup_recovery_status = None
        self._manual_load_task = None
        self._manual_load_generation = 0
        self._environment_report_current = False
        self._nonblocking_message_boxes = set()
        self.setWindowTitle("地物标注工具")
        self.setObjectName("labelingDock")

        self._build_ui()
        self._connect_signals()
        # Input signals may update the UI while restoring, but must not save
        # partially initialized fields over the persisted selection.
        self._restoring_settings = True
        try:
            self._load_settings_and_defaults()
        finally:
            self._restoring_settings = False
        self._update_start_enabled()
        # Restoring completed work is independent from checking inference dependencies.
        QTimer.singleShot(0, self._restore_latest_ready_run)

    def _build_ui(self):
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── Header ──
        header = QLabel("地物标注工具 v2.0")
        header.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(header)

        # ── Data source group ──
        source_group = QGroupBox("数据源与范围")
        source_group.setObjectName("sourceGroup")
        source_grid = QGridLayout(source_group)
        source_grid.setColumnStretch(0, 0)
        source_grid.setColumnStretch(1, 1)

        self.raster_combo = _ScreenBoundMapLayerComboBox()
        self.raster_combo.setFilters(Qgis.LayerFilter.RasterLayer)

        raster_label = QLabel("影像层:")
        raster_label.setFont(self.raster_combo.font())
        source_grid.addWidget(
            raster_label,
            0,
            0,
            ALIGN_LEFT | ALIGN_VCENTER,
        )
        source_grid.addWidget(self.raster_combo, 0, 1)

        self.extent_group = QButtonGroup(self)
        self.radio_view = QRadioButton("当前视图")
        self.radio_rect = QRadioButton("手绘矩形")
        self.radio_vector = QRadioButton("加载矢量范围")
        self.radio_rect.setEnabled(True)
        self.radio_rect.setToolTip("在地图上拖拽绘制标注范围")
        self.radio_vector.setToolTip(
            "相交 Tile 仅用于处理；结果会按矢量边界精确裁剪"
        )
        self.radio_view.setChecked(True)
        self.extent_group.addButton(self.radio_view)
        self.extent_group.addButton(self.radio_rect)
        self.extent_group.addButton(self.radio_vector)
        extent_row = QWidget()
        extent_row_layout = QHBoxLayout(extent_row)
        extent_row_layout.setContentsMargins(0, 0, 0, 0)
        extent_row_layout.addWidget(self.radio_view)
        extent_row_layout.addWidget(self.radio_rect)
        extent_row_layout.addWidget(self.radio_vector)
        self.capture_view_btn = QPushButton("获取当前视图")
        self.draw_rect_btn = QPushButton("绘制范围")
        self.draw_rect_btn.setEnabled(False)
        source_grid.addWidget(
            QLabel("范围:"),
            1,
            0,
            ALIGN_LEFT | ALIGN_VCENTER,
        )
        source_grid.addWidget(extent_row, 1, 1)

        extent_actions = QWidget()
        extent_actions_layout = QHBoxLayout(extent_actions)
        extent_actions_layout.setContentsMargins(0, 0, 0, 0)
        extent_actions_layout.addWidget(self.capture_view_btn)
        extent_actions_layout.addWidget(self.draw_rect_btn)
        extent_actions_layout.addStretch()
        source_grid.addWidget(QLabel("操作:"), 2, 0)
        source_grid.addWidget(extent_actions, 2, 1)

        self.vector_range_combo = _ScreenBoundMapLayerComboBox()
        self.vector_range_combo.setFilters(Qgis.LayerFilter.PolygonLayer)
        self.vector_range_combo.setEnabled(False)
        self.vector_range_combo.setToolTip(
            "相交 Tile 用于处理，结果按矢量边界精确裁剪"
        )
        self.vector_range_label = QLabel("范围矢量:")
        self.vector_range_label.setEnabled(False)
        source_grid.addWidget(
            self.vector_range_label,
            3,
            0,
            ALIGN_LEFT | ALIGN_VCENTER,
        )
        source_grid.addWidget(self.vector_range_combo, 3, 1)

        self.extent_status_label = QLabel(
            "开始时将读取当前视图；可点击「获取当前视图」预览范围"
        )
        self.extent_status_label.setWordWrap(True)
        self.extent_status_label.setStyleSheet("color: #666;")
        source_grid.addWidget(
            QLabel("状态:"),
            4,
            0,
            ALIGN_LEFT | ALIGN_VCENTER,
        )
        source_grid.addWidget(self.extent_status_label, 4, 1)

        layout.addWidget(source_group)

        # ── Tile config group ──
        tile_group = QGroupBox("切片与高级设置")
        tile_group.setObjectName("tileGroup")
        tile_outer_layout = QVBoxLayout(tile_group)
        self.tile_details_toggle = QPushButton("展开切片参数")
        self.tile_details_toggle.setCheckable(True)
        self.tile_details_toggle.toggled.connect(
            lambda expanded: self._set_section_expanded(
                self.tile_details_widget,
                self.tile_details_toggle,
                expanded,
                "切片参数",
            )
        )
        tile_outer_layout.addWidget(self.tile_details_toggle)
        self.tile_details_widget = QWidget()
        tile_layout = QFormLayout(self.tile_details_widget)

        tile_size_layout = QHBoxLayout()
        self.tile_width_spin = QSpinBox()
        self.tile_width_spin.setRange(64, 4096)
        self.tile_width_spin.setValue(512)
        self.tile_width_spin.setEnabled(True)
        self.tile_width_spin.setToolTip("Tile 宽度（像素）")
        self.tile_height_spin = QSpinBox()
        self.tile_height_spin.setRange(64, 4096)
        self.tile_height_spin.setValue(512)
        self.tile_height_spin.setEnabled(True)
        self.tile_height_spin.setToolTip("Tile 高度（像素）")
        tile_size_layout.addWidget(self.tile_width_spin, stretch=1)
        tile_size_layout.addWidget(QLabel(" × "))
        tile_size_layout.addWidget(self.tile_height_spin, stretch=1)
        tile_layout.addRow("Tile 尺寸:", tile_size_layout)

        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(1, 511)
        self.overlap_spin.setValue(192)
        self.overlap_spin.setSuffix(" px")
        tile_layout.addRow("重叠:", self.overlap_spin)

        self.processing_extent_status_label = QLabel(
            "选择本地影像后，开始时将自动读取当前视图范围"
        )
        self.processing_extent_status_label.setWordWrap(True)
        self.processing_extent_status_label.setStyleSheet("color: #666;")
        tile_layout.addRow("自动扩展推理范围:", self.processing_extent_status_label)

        tile_outer_layout.addWidget(self.tile_details_widget)
        self.tile_details_widget.setVisible(False)
        layout.addWidget(tile_group)

        # ── Output group ──
        output_group = QGroupBox("输出位置")
        output_group.setObjectName("outputGroup")
        output_outer_layout = QVBoxLayout(output_group)
        self.output_details_toggle = QPushButton("展开输出位置")
        self.output_details_toggle.setCheckable(True)
        self.output_details_toggle.toggled.connect(
            lambda expanded: self._set_section_expanded(
                self.output_details_widget,
                self.output_details_toggle,
                expanded,
                "输出位置",
            )
        )
        output_outer_layout.addWidget(self.output_details_toggle)
        self.output_details_widget = QWidget()
        output_layout = QFormLayout(self.output_details_widget)

        workspace_layout = QHBoxLayout()
        self.workspace_edit = QLineEdit()
        self.workspace_edit.setPlaceholderText(".../output")
        self.browse_workspace_btn = QPushButton("选择")
        workspace_layout.addWidget(self.workspace_edit)
        workspace_layout.addWidget(self.browse_workspace_btn)
        output_layout.addRow("运行工作区:", workspace_layout)

        accepted_path_layout = QHBoxLayout()
        self.accepted_path_edit = QLineEdit()
        self.accepted_path_edit.setPlaceholderText(".../output/accepted_labels.gpkg")
        self.output_path_edit = self.accepted_path_edit
        self.browse_output_btn = QPushButton("选择")
        accepted_path_layout.addWidget(self.accepted_path_edit)
        accepted_path_layout.addWidget(self.browse_output_btn)
        output_layout.addRow("标签库（GPKG）：", accepted_path_layout)

        self.skip_accepted_check = QCheckBox("跳过已确认区域")
        self.skip_accepted_check.setChecked(True)
        output_layout.addRow(self.skip_accepted_check)

        output_outer_layout.addWidget(self.output_details_widget)
        self.output_details_widget.setVisible(False)
        layout.addWidget(output_group)

        self.environment_panel = EnvironmentPanel()
        layout.insertWidget(2, self.environment_panel)

        self.plan_panel = InferencePlanPanel(
            lambda: self.config_manager.last_report or {}
        )
        layout.insertWidget(3, self.plan_panel)

        # ── Action buttons ──
        run_group = QGroupBox("开始与当前状态")
        run_group.setObjectName("runGroup")
        self._run_group = run_group
        run_layout = QVBoxLayout(run_group)
        self.start_readiness_label = QLabel("当前不能开始：请先完成准备")
        self.start_readiness_label.setObjectName("startReadinessLabel")
        self.start_readiness_label.setWordWrap(True)
        run_layout.addWidget(self.start_readiness_label)
        action_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始标注")
        self.start_btn.setEnabled(False)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.show_monitor_btn = QPushButton("推理监控")
        self.show_monitor_btn.setToolTip("打开推理监控窗口（关闭窗口不会停止任务）")
        self.show_monitor_btn.setCheckable(True)
        self.show_monitor_btn.clicked.connect(self._on_toggle_monitor)
        action_layout.addWidget(self.start_btn)
        action_layout.addWidget(self.stop_btn)
        action_layout.addWidget(self.show_monitor_btn)
        run_layout.addLayout(action_layout)
        recovery_layout = QHBoxLayout()
        self.resume_btn = QPushButton("恢复上次运行")
        self.retry_failed_btn = QPushButton("重做失败包")
        self.retry_failed_btn.setToolTip(
            "清理失败 Work Package 及受影响下游后重新运行；保留共享 Tile 缓存"
        )
        self.resume_btn.setEnabled(False)
        self.retry_failed_btn.setEnabled(False)
        recovery_layout.addWidget(self.resume_btn)
        recovery_layout.addWidget(self.retry_failed_btn)
        run_layout.addLayout(recovery_layout)

        self.load_manual_run_btn = QPushButton("加载已有 Run 人工整理")
        self.load_manual_run_btn.setToolTip(
            "选择任意位置的 Run 副本；已有完整14类工作区时无需复制 Fusion 大文件，"
            "不检查推理环境，不运行 SAM3"
        )
        run_layout.addWidget(self.load_manual_run_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        run_layout.addWidget(self.progress_bar)

        result_group = QGroupBox("结果")
        result_group.setObjectName("resultGroup")
        result_layout = QVBoxLayout(result_group)
        self.result_summary_label = QLabel("尚无运行结果")
        self.result_summary_label.setWordWrap(True)
        result_layout.addWidget(self.result_summary_label)

        self.open_refinement_btn = QPushButton("打开分类修整与组装")
        self.open_refinement_btn.setEnabled(False)
        result_layout.addWidget(self.open_refinement_btn)
        layout.addWidget(result_group)

        layout.addStretch()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(
            SCROLLBAR_AS_NEEDED
        )
        scroll_area.setVerticalScrollBarPolicy(SCROLLBAR_AS_NEEDED)
        scroll_area.setWidget(main_widget)
        self._main_scroll_area = scroll_area
        dock_widget = QWidget()
        dock_layout = QVBoxLayout(dock_widget)
        dock_layout.setContentsMargins(0, 0, 0, 0)
        dock_layout.setSpacing(6)
        dock_layout.addWidget(scroll_area, stretch=1)
        dock_layout.addWidget(run_group)
        self.setWidget(dock_widget)

    @staticmethod
    def _set_section_expanded(widget, toggle, expanded, title):
        widget.setVisible(expanded)
        toggle.setText(f"收起{title}" if expanded else f"展开{title}")

    def _connect_signals(self):
        self.environment_panel.script_path_changed.connect(
            self._mark_env_check_required
        )
        self.environment_panel.check_requested.connect(self._run_env_check)
        self.environment_panel.details_requested.connect(self._show_env_details)
        self.plan_panel.selection_changed.connect(self._on_plan_selection_changed)
        self.browse_output_btn.clicked.connect(self._on_browse_output)
        self.browse_workspace_btn.clicked.connect(self._on_browse_workspace)
        self.open_refinement_btn.clicked.connect(self._on_open_refinement)
        self.load_manual_run_btn.clicked.connect(self._on_load_manual_run)
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.resume_btn.clicked.connect(lambda: self._resume_existing_run(False))
        self.retry_failed_btn.clicked.connect(lambda: self._resume_existing_run(True))
        self.extent_group.buttonClicked.connect(self._on_extent_mode_changed)
        self.extent_group.buttonClicked.connect(
            lambda _button: self._update_start_enabled()
        )
        self.capture_view_btn.clicked.connect(self._capture_current_view_extent)
        self.draw_rect_btn.clicked.connect(self._on_draw_rect_clicked)
        self.raster_combo.layerChanged.connect(self._on_raster_layer_changed)
        self.raster_combo.layerChanged.connect(
            lambda _layer: self._update_start_enabled()
        )
        self.vector_range_combo.layerChanged.connect(
            self._on_vector_range_layer_changed
        )
        self.vector_range_combo.layerChanged.connect(
            lambda _layer: self._update_start_enabled()
        )
        self.output_path_edit.textChanged.connect(self._mark_env_check_required)
        self.workspace_edit.textChanged.connect(self._on_workspace_changed)
        self.tile_width_spin.valueChanged.connect(self._on_tile_parameters_changed)
        self.tile_height_spin.valueChanged.connect(self._on_tile_parameters_changed)
        self.overlap_spin.valueChanged.connect(self._on_tile_parameters_changed)
        self.skip_accepted_check.toggled.connect(self._save_settings)
        self.config_manager.check_started.connect(self._on_env_check_started)
        self.config_manager.report_ready.connect(self._on_env_report_ready)

    # ── Slots ──

    def _on_browse_output(self):
        project_dir = os.path.dirname(QgsProject.instance().fileName()) or ""
        path, _ = QFileDialog.getSaveFileName(self, "选择输出 GPKG", os.path.join(project_dir, "output"), "GeoPackage (*.gpkg)")
        if path:
            self.output_path_edit.setText(path)

    def _on_browse_workspace(self):
        project_dir = os.path.dirname(QgsProject.instance().fileName()) or ""
        path = QFileDialog.getExistingDirectory(self, "选择运行输出工作区", project_dir)
        if path:
            self.workspace_edit.setText(path)

    def _mark_env_check_required(self, *_args):
        self._save_settings()
        self._environment_report_current = False
        self.environment_panel.mark_check_required()
        self.plan_panel.invalidate()
        self._update_start_enabled()

    def _on_workspace_changed(self, *_args):
        self._last_run_result = None
        self._last_run_spec = None
        self._recovery_run_spec = None
        self._startup_ready_candidate = None
        self._startup_recovery_status = None
        self.open_refinement_btn.setEnabled(False)
        self.open_refinement_btn.setText("打开分类修整与组装")
        self.result_summary_label.setText("尚无运行结果")
        self._mark_env_check_required()

    def _run_env_check(self):
        self.environment_panel.refresh_config_path()
        scripts_dir = self.environment_panel.scripts_directory
        output_dir = self.workspace_edit.text().strip()
        self.config_manager.start_check(scripts_dir, output_dir)

    def _on_env_check_started(self):
        self._environment_report_current = False
        self.environment_panel.show_checking()
        self.plan_panel.invalidate()
        self._update_start_enabled()

    def _on_env_report_ready(self, report):
        self._environment_report_current = True
        self.plan_panel.set_environment(report)
        self._render_env_report(report, check_finished=True)
        self._restore_latest_ready_run()

    def _restore_latest_ready_run(self):
        if self.workflow.is_active:
            return
        output_root = self.workspace_edit.text().strip()
        candidates = run_index.load_startup_candidates(output_root)
        self._recovery_run_spec = None
        self._startup_recovery_status = None
        if not self._last_run_result:
            self.open_refinement_btn.setEnabled(False)
            self.open_refinement_btn.setText("打开分类修整与组装")
        latest = candidates.get("latest") or {}
        latest_status = str(latest.get("indexed_status") or "")
        if latest_status in run_index.RECOVERABLE_RUN_STATES:
            self._recovery_run_spec = latest["spec"]
            self._startup_recovery_status = latest_status
            self.result_summary_label.setText(
                f"发现可恢复 Run {latest['run_id']}；状态 {latest_status}"
            )

        self._startup_ready_candidate = None
        if not self._last_run_result:
            ready = candidates.get("latest_ready")
            try:
                result = run_index.lightweight_ready_result(ready) if ready else None
            except (KeyError, TypeError, run_index.RunIndexError):
                result = None
            if result is not None:
                self._startup_ready_candidate = (result, ready["spec"])
                if self._recovery_run_spec is None:
                    self.result_summary_label.setText(
                        f"发现最近 Ready Run {result['run_id']}；"
                        "打开时再校验正式结果"
                    )
                self.open_refinement_btn.setText("验证并打开最近 Run")
                self.open_refinement_btn.setEnabled(True)
        self._update_recovery_buttons(lightweight=True)

    def _render_last_env_report(self, *_args):
        self._save_settings()
        if self.config_manager.last_report:
            self._render_env_report(self.config_manager.last_report)

    def _render_env_report(self, report, *, check_finished=False):
        task_checks = self._get_task_parameter_checks()
        self.environment_panel.show_report(
            report, task_checks, check_finished=check_finished
        )
        self._update_start_enabled()

    def _get_task_parameter_checks(self):
        output = self.output_path_edit.text().strip() or "未选择"
        workspace = self.workspace_edit.text().strip() or "未选择"
        return [
            {
                "id": "tile_parameters",
                "status": "ready",
                "value": (
                    f"{self.tile_width_spin.value()} x {self.tile_height_spin.value()}, "
                    f"overlap {self.overlap_spin.value()} px"
                ),
                "source": "QGIS 面板:切片",
                "message": "本次任务参数，不读取 config.yaml 中的旧字段",
                "fix": "在切片区域修改",
            },
            {
                "id": "output_workspace",
                "status": "ready" if workspace != "未选择" else "error",
                "value": workspace,
                "source": "QGIS 面板:运行工作区",
                "message": "每次运行在该目录的 runs/ 下创建唯一目录",
                "fix": "选择可写的运行工作区",
            },
            {
                "id": "output_path",
                "status": "ready" if output != "未选择" else "error",
                "value": output,
                "source": "QGIS 面板:输出",
                "message": "",
                "fix": "在输出区域重新选择 GPKG",
            },
        ]

    def _environment_readiness(self):
        report = self.config_manager.last_report or {}
        if not report:
            return "请先检查推理环境", ()
        if not self._environment_report_current:
            return "推理环境报告已过期，请重新检查", ()
        if report.get("status") == "error":
            return (
                "推理环境未就绪："
                + (first_problem(report.get("checks") or []) or "请查看检查结果"),
                (),
            )
        if report.get("status") == "warning":
            return "", ("推理环境有警告，开始时需要确认",)
        if report.get("status") != "ready":
            return "推理环境检查状态未知，请重新检查", ()
        return "", ()

    def _range_readiness(self, raster):
        if self.radio_view.isChecked():
            selection = self.range_selection.selected(VIEW_MODE)
            if selection.extent is None:
                if self.iface is None:
                    return "当前视图范围不可用，请在 QGIS 地图画布中选择范围", ()
                return "", ("开始时将自动读取当前视图范围",)
        elif self.radio_rect.isChecked():
            selection = self.range_selection.selected(RECTANGLE_MODE)
        else:
            layer = self.vector_range_combo.currentLayer()
            if not isinstance(layer, QgsVectorLayer) or not layer.isValid():
                return "请选择有效的矢量范围图层", ()
            if layer.geometryType() != Qgis.GeometryType.Polygon:
                return "矢量范围必须是面图层", ()
            if not layer.crs().isValid():
                return "矢量范围没有有效 CRS", ()
            selection = RawRangeSelection(
                VECTOR_MODE, layer.extent(), layer.crs(), layer
            )
        try:
            resolve_raster_extent(selection, raster)
        except ValueError as exc:
            return str(exc), ()
        return "", ()

    def _plan_readiness(self, report):
        selection = self.plan_panel.selection
        if not selection.confirmed:
            return "请确认本次推理方案"
        if not selection.model_ids:
            return "请选择至少一个可用模型"
        if not report:
            return ""
        try:
            resolve_launch_plan(report, selection)
        except (KeyError, TypeError, ValueError) as exc:
            return f"模型方案不可用：{exc}"
        return ""

    def _current_start_readiness(self):
        report = self.config_manager.last_report or {}
        environment_problem, notices = self._environment_readiness()
        raster_problem = ""
        range_problem = ""
        try:
            raster = validate_raster_layer(self.raster_combo.currentLayer())
        except ValueError as exc:
            raster_problem = str(exc)
        else:
            range_problem, range_notices = self._range_readiness(raster)
            notices = notices + range_notices
        return derive_start_readiness(
            environment_problem=environment_problem,
            raster_problem=raster_problem,
            range_problem=range_problem,
            output_path=self.output_path_edit.text().strip(),
            workspace_path=self.workspace_edit.text().strip(),
            plan_problem=self._plan_readiness(report),
            workflow_active=self.workflow.is_active,
            notices=notices,
        )

    def _update_start_enabled(self):
        readiness = self._current_start_readiness()
        self.start_btn.setEnabled(readiness.can_start)
        self.start_readiness_label.setText(
            readiness.summary + "\n开始时会再次复核外部文件和实际输入。"
        )
        self.start_btn.setToolTip(
            "启动本次标注任务"
            if readiness.can_start
            else "；".join(readiness.blockers)
        )

    def _on_plan_selection_changed(self):
        """Persist a changed plan snapshot and recalculate the launch gate."""

        self._save_settings()
        self._update_start_enabled()

    def _show_env_details(self):
        report = self.config_manager.last_report or {}
        self.environment_panel.show_details(
            report, self._get_task_parameter_checks()
        )

    def _on_draw_rect_clicked(self):
        self.radio_rect.setChecked(True)
        if self.range_selection.begin_rectangle():
            self.extent_status_label.setText(
                "请在地图上按住鼠标拖拽，释放后确定矩形范围"
            )
            self._refresh_processing_extent_preview()

    def _on_extent_mode_changed(self, button):
        vector_mode = button == self.radio_vector
        if not vector_mode:
            self.vector_preview.invalidate()
        self.vector_range_combo.setEnabled(vector_mode)
        self.vector_range_label.setEnabled(vector_mode)
        if vector_mode:
            self.capture_view_btn.setEnabled(False)
            self.draw_rect_btn.setEnabled(False)
            self.range_selection.restore_map_tool()
            self._on_vector_range_layer_changed(
                self.vector_range_combo.currentLayer()
            )
            return

        if button == self.radio_rect:
            self.capture_view_btn.setEnabled(False)
            self.draw_rect_btn.setEnabled(True)
            selection = self.range_selection.selected(RECTANGLE_MODE)
            if selection.extent is None:
                self.extent_status_label.setText("点击「绘制范围」后在地图上拖拽矩形")
                self._refresh_processing_extent_preview()
            else:
                self._update_extent_status(selection)
            return

        self.capture_view_btn.setEnabled(True)
        self.draw_rect_btn.setEnabled(False)
        self.range_selection.restore_map_tool()
        selection = self.range_selection.selected(VIEW_MODE)
        if selection.extent is None:
            self.extent_status_label.setText(
                "开始时将读取当前视图；可点击「获取当前视图」预览范围"
            )
            self._refresh_processing_extent_preview()
        else:
            self._update_extent_status(selection)

    def _capture_current_view_extent(self):
        selection = self.range_selection.capture_current_view()
        if selection is None:
            return
        self.radio_view.setChecked(True)
        self._update_extent_status(selection)
        self._update_start_enabled()

    def _on_rect_finished(self, selection: RawRangeSelection):
        self._update_extent_status(selection)
        self.range_selection.restore_map_tool()
        self._update_start_enabled()

    def _on_vector_range_layer_changed(self, *_args):
        if not self.radio_vector.isChecked():
            return
        self.vector_preview.invalidate()
        try:
            layer = validate_vector_layer(self.vector_range_combo.currentLayer())
            self.vector_preview.watch_layer(layer)
            raster = validate_raster_layer(self.raster_combo.currentLayer())
            extent = transform_extent(
                layer.extent(), layer.crs(), raster.crs()
            )
            extent = intersect_extents(extent, raster.extent())
            if extent is None or not is_valid_extent(extent):
                raise ValueError(
                    f"矢量图层「{layer.name()}」与影像层没有重叠"
                )
            self.extent_status_label.setText(
                f"矢量范围: {layer.name()}；相交 Tile 用于处理，结果按矢量边界精确裁剪"
            )
        except ValueError as exc:
            self.extent_status_label.setText(str(exc))
        self._refresh_processing_extent_preview()
        self._update_start_enabled()

    def _on_start(self):
        if self.workflow.is_active:
            return
        report = self.config_manager.last_report
        scripts_dir = self.environment_panel.scripts_directory
        if not report or self.config_manager.is_stale(scripts_dir):
            QMessageBox.warning(
                self,
                "推理环境",
                "配置尚未检查或已经变化，请先点击“检查推理环境”。",
            )
            return
        if report.get("status") == "error":
            QMessageBox.warning(
                self,
                "推理环境未就绪",
                first_problem(report.get("checks") or [])
                or "请先修正推理环境中的错误。",
            )
            return
        if report.get("status") == "warning":
            answer = QMessageBox.question(
                self,
                "推理环境警告",
                (first_problem(report.get("checks") or []) or "当前配置存在警告。")
                + "\n\n是否继续本次推理？",
                YES | NO,
                NO,
            )
            if answer != YES:
                return
        plan_selection = self.plan_panel.selection
        try:
            launch_plan = resolve_launch_plan(report, plan_selection)
        except (KeyError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "推理方案不可运行", str(exc))
            return
        effective = report.get("effective", {})

        try:
            raster = validate_raster_layer(self.raster_combo.currentLayer())
        except ValueError as e:
            QMessageBox.warning(self, "错误", str(e))
            return

        if not scripts_dir or not os.path.isdir(scripts_dir):
            QMessageBox.warning(self, "错误", "请设置有效的推理脚本路径")
            return

        if self.range_selection.is_drawing:
            self.range_selection.restore_map_tool()

        output_gpkg = self.output_path_edit.text().strip()
        if not output_gpkg:
            QMessageBox.warning(self, "错误", "请设置输出 GPKG 路径")
            return
        output_gpkg = os.path.abspath(os.path.expanduser(output_gpkg))

        try:
            selection = self._selected_raw_selection(
                ensure_view=True, strict=True
            )
            extent = resolve_raster_extent(selection, raster)
            self.extent_status_label.setText(
                self._extent_status_text(selection.mode, extent, raster)
            )
            self._refresh_processing_extent_preview()
        except ValueError as e:
            QMessageBox.warning(self, "错误", str(e))
            return

        tile_width = self.tile_width_spin.value()
        tile_height = self.tile_height_spin.value()
        overlap = self.overlap_spin.value()
        parameters = TileParameters(tile_width, tile_height, overlap)
        preview = None
        if self.radio_vector.isChecked():
            layer = validate_vector_layer(self.vector_range_combo.currentLayer())
            request = self.vector_preview.make_request(
                raster, layer, extent, parameters
            )
            preview = self.vector_preview.cached(request)
            if preview is None:
                self.processing_extent_status_label.setText(
                    "正在后台计算矢量范围 Tile，完成后自动开始标注..."
                )
                self.processing_extent_status_label.setStyleSheet("color: #805500;")
                preview = self.vector_preview.queue(
                    request, immediate=True, auto_start=True
                )
                if preview is None:
                    return
            grid_tiles = preview.grid_tiles
            current_tiles = preview.selected_tiles
        else:
            try:
                grid_tiles = tile_manager.generate_grid(
                    extent, tile_width, tile_height, overlap, raster_layer=raster
                )
                current_tiles = list(grid_tiles)
            except ValueError as exc:
                QMessageBox.warning(self, "切片范围无效", str(exc))
                return

        if not current_tiles:
            QMessageBox.warning(self, "错误", "未生成任何 tile，请检查范围和尺寸")
            return

        processing_extent = (
            preview.processing_extent
            if self.radio_vector.isChecked()
            else tile_manager.get_grid_extent(grid_tiles)
        )
        range_mode = self._selected_raw_selection().mode
        self.extent_status_label.setText(
            self._extent_status_text(
                range_mode,
                extent,
                raster,
            )
        )
        if self.radio_vector.isChecked():
            self._set_processing_extent_summary(preview, raster)
        else:
            self._set_processing_extent_status(
                current_tiles, raster, grid_tiles=grid_tiles
            )

        output_dir = os.path.abspath(self.workspace_edit.text().strip())
        os.makedirs(output_dir, exist_ok=True)

        accepted_layer = None
        accepted_validation = {
            "status": "passed",
            "feature_count": 0,
            "overlap_pair_count": 0,
            "overlap_tolerance": max(
                abs(
                    float(raster.rasterUnitsPerPixelX())
                    * float(raster.rasterUnitsPerPixelY())
                )
                * 1.0e-6,
                1.0e-18,
            ),
            "crs": raster.crs().authid(),
            "source": "not_present",
        }
        if os.path.exists(output_gpkg):
            target_accepted_layer = QgsVectorLayer(
                f"{output_gpkg}|layername={LAYER_NAMES.ACCEPTED}",
                "accepted",
                "ogr",
            )
            try:
                if not target_accepted_layer.isValid():
                    raise ValueError("无法打开标签库")
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "已确认区域审计失败",
                    "本次任务尚未创建。请先修复标签库：\n" + str(exc),
                )
                return
            accepted_layer = target_accepted_layer

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setRange(0, len(current_tiles))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("准备提取切片")
        self.monitor_dialog.detach()
        self.workflow.start_new_run(
            RunStartRequest(
                scripts_dir=scripts_dir,
                output_root=output_dir,
                accepted_target_gpkg=output_gpkg,
                raster_layer=raster,
                requested_extent=extent,
                processing_extent=processing_extent,
                grid_tiles=tuple(grid_tiles),
                active_tiles=tuple(current_tiles),
                range_selection=range_selection_metadata(
                    (
                        validate_vector_layer(
                            self.vector_range_combo.currentLayer()
                        )
                        if self.radio_vector.isChecked()
                        else None
                    ),
                    len(grid_tiles),
                    len(current_tiles),
                ),
                effective_config=dict(effective),
                environment_report=dict(report),
                accepted_layer=accepted_layer,
                accepted_validation=accepted_validation,
                get_valid_range_layer=(
                    (
                        lambda: validate_vector_layer(
                            self.vector_range_combo.currentLayer()
                        )
                    )
                    if self.radio_vector.isChecked()
                    else None
                ),
                skip_accepted=bool(self.skip_accepted_check.isChecked()),
                selected_model_ids=launch_plan.model_ids,
                fusion_profile_id=launch_plan.fusion_profile_id,
                boundary_smoothing_enabled=launch_plan.boundary_smoothing_enabled,
                overlap=overlap,
            )
        )

    def _on_workflow_state_changed(self, state):
        active = self.workflow.is_active
        self.stop_btn.setEnabled(active and state != "stopping")
        self._update_recovery_buttons()
        if not active:
            self._update_start_enabled()

    def _on_workflow_runner_changed(self, runner):
        if self.monitor_dialog is None:
            return
        self.monitor_dialog.detach()
        if runner is None:
            return
        self.monitor_dialog.reset_run()
        self.monitor_dialog.attach_runner(runner)
        self.monitor_dialog.show()
        self.monitor_dialog.raise_()
        self.show_monitor_btn.setChecked(True)
        self.show_monitor_btn.setText("隐藏监控")

    def _on_workflow_monitor_context_ready(
        self,
        database_path,
        run_id,
        page_size,
        run_spec,
    ):
        if self.monitor_dialog is None:
            return
        self.monitor_dialog.bind_state_database(
            database_path,
            run_id,
            page_size=page_size,
            run_spec=run_spec,
        )

    def _apply_stage_progress(self, info):
        if self.monitor_dialog is not None:
            self.monitor_dialog.set_stage_progress(info)
        current = int(info.get("current", 0))
        total = int(info.get("total", 0))
        name = info.get("name", "处理中")
        index = int(info.get("index", 0))
        stage_total = int(info.get("stage_total", 0))
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(max(0, min(current, total)))
            self.progress_bar.setFormat(
                f"{name} ({index}/{stage_total})  {current}/{total}"
            )
        else:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat(f"{name} ({index}/{stage_total})")

    def _on_toggle_monitor(self, checked):
        if self.monitor_dialog is None:
            return
        if checked:
            self.monitor_dialog.show()
            self.monitor_dialog.raise_()
            self.monitor_dialog.activateWindow()
            self.show_monitor_btn.setText("隐藏监控")
        else:
            self.monitor_dialog.hide()
            self.show_monitor_btn.setText("推理监控")

    def _on_workflow_stopped_before_run(self):
        self.stop_btn.setEnabled(False)
        self._set_progress_terminal("已停止")
        if self.monitor_dialog is not None:
            self.monitor_dialog.mark_finished("已停止")
        self._update_start_enabled()

    def _show_nonblocking_notice(self, icon, title, message):
        """Show a Wayland-safe notice without locking the monitor window."""

        for previous in tuple(self._nonblocking_message_boxes):
            try:
                previous.close()
            except RuntimeError:
                self._nonblocking_message_boxes.discard(previous)

        monitor = self.monitor_dialog
        parent = monitor if monitor is not None and monitor.isVisible() else self
        dialog = QMessageBox(parent)
        dialog.setIcon(icon)
        dialog.setWindowTitle(str(title))
        dialog.setText(str(message))
        dialog.setTextInteractionFlags(TEXT_SELECTABLE_BY_MOUSE)
        dialog.setStandardButtons(CLOSE)
        dialog.setWindowModality(NON_MODAL)
        dialog.setAttribute(WA_DELETE_ON_CLOSE, True)
        self._nonblocking_message_boxes.add(dialog)

        def _release_dialog(*_args):
            self._nonblocking_message_boxes.discard(dialog)

        dialog.destroyed.connect(_release_dialog)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_workflow_pre_run_failed(self, title, message):
        self.stop_btn.setEnabled(False)
        self._set_progress_terminal(title)
        if self.monitor_dialog is not None:
            self.monitor_dialog.mark_finished(title, message)
        self._update_start_enabled()
        if not self._cleaning_up:
            self._show_nonblocking_notice(CRITICAL, title, message)

    def _set_progress_terminal(self, text, completed=False):
        """Stop indeterminate animation and display a stable terminal state."""
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1 if completed else 0)
        self.progress_bar.setFormat(text)

    def _on_stop(self):
        if not self.workflow.is_active:
            if self.monitor_dialog is not None:
                self.monitor_dialog.mark_finished("无任务运行")
            return
        self.stop_btn.setEnabled(False)
        if self.monitor_dialog is not None:
            if self.workflow.state.value == "planning":
                self.monitor_dialog.mark_stopping("正在停止后台任务图建立")
            else:
                self.monitor_dialog.mark_stopping()
        self.workflow.stop()

    def _on_monitor_main_run_handling(self, payload):
        """Select a monitor-confirmed terminal Run without starting recovery."""
        if self.workflow.is_active:
            self._show_nonblocking_notice(
                WARNING,
                "不能切换 Run",
                "当前任务正在运行或停止中，不能切换 Run。",
            )
            return

        if not isinstance(payload, dict):
            logger.warning("监控请求主面板处理的载荷不是对象")
            return
        run_id = str(payload.get("run_id") or "")
        run_spec = payload.get("run_spec")
        observed_status = str(payload.get("observed_status") or "")
        try:
            schema_version = int(run_spec.get("schema_version") or 0)
        except (AttributeError, TypeError, ValueError, OverflowError):
            logger.warning("监控请求主面板处理的 Run schema 无效")
            return
        if (
            not isinstance(run_spec, dict)
            or schema_version != 2
            or not run_id
            or str(run_spec.get("run_id") or "") != run_id
            or observed_status not in {"failed", "stopped"}
        ):
            logger.warning("监控请求主面板处理的 Run 身份或状态无效: %r", payload)
            return

        run_dir = Path(str(run_spec.get("run_dir") or "")).expanduser()
        if (
            run_dir.is_symlink()
            or not run_dir.is_dir()
            or run_dir.name != run_id
            or not (run_dir / "run_spec.json").is_file()
        ):
            logger.warning("监控请求主面板处理的 Run 目录无效: %s", run_dir)
            return

        self._recovery_run_spec = dict(run_spec)
        self._startup_recovery_status = observed_status
        self._update_recovery_buttons()
        self.show()
        self.raise_()
        window = self.window()
        window.show()
        window.raise_()
        window.activateWindow()
        self._run_group.setFocus()
        target = self.retry_failed_btn if observed_status == "failed" else self.resume_btn
        target.setFocus()

    @staticmethod
    def _is_local_attempt_only_result(result):
        """Return whether a runner result did not publish a Run terminal state."""
        return result.get("terminal_published") is False

    def _on_pipeline_finished(self, result):
        if self._is_local_attempt_only_result(result):
            self.stop_btn.setEnabled(False)
            self._set_progress_terminal("本地尝试已结束")
            self._update_start_enabled()
            message = "本次启动/执行已结束，Run 当前状态以监控同步为准。"
            detail = str(result.get("error") or "")
            if detail:
                message += "\n\n" + detail
            self._show_nonblocking_notice(WARNING, "本地尝试未发布终态", message)
            return
        try:
            self.layer_manager.load_run_results(result)
        except Exception as e:
            logger.exception("加载运行结果失败: %s", e)

        try:
            self.layer_manager.group_layers()
        except Exception as e:
            logger.error("图层分组失败: %s", e)

        self._last_run_result = dict(result)
        self._startup_ready_candidate = None
        self._startup_recovery_status = None
        self.open_refinement_btn.setText("打开分类修整与组装")
        try:
            with open(result.get("run_spec", ""), "r", encoding="utf-8") as handle:
                self._last_run_spec = __import__("json").load(handle)
            self._recovery_run_spec = self._last_run_spec
        except (OSError, ValueError):
            self._last_run_spec = None
            self._recovery_run_spec = None
        ready_count = len(result.get("ready_streams") or [])
        failed_count = len(result.get("failed_streams") or [])
        fusion_ready = any(item.get("kind") == "fusion" for item in result.get("ready_streams") or [])
        result_summary = (
            f"模型/融合结果流 {ready_count} 个；Fusion {'成功' if fusion_ready else '无或失败'}；"
            f"失败 {failed_count} 个"
        )
        self.result_summary_label.setText(result_summary)
        self.open_refinement_btn.setEnabled(ready_count > 0)

        self._update_start_enabled()
        self.stop_btn.setEnabled(False)
        self._update_recovery_buttons()
        stopped = str(result.get("error", "")).startswith("Pipeline stopped by user")
        if result.get("success"):
            self._set_progress_terminal("完成", completed=True)
        elif stopped:
            self._set_progress_terminal("已停止")
        else:
            self._set_progress_terminal("失败")

        if result.get("success"):
            self._show_nonblocking_notice(
                INFORMATION,
                "完成",
                f"推理完成，已加载 {len(result.get('ready_streams') or [])} 个结果流",
            )
        elif not stopped:
            run_report = result.get("run_report", "")
            msg = result.get("error") or "推理流程失败"
            if run_report:
                msg += f"\n\n运行报告: {run_report}"
            self._show_nonblocking_notice(WARNING, "推理失败", msg)

    def _update_recovery_buttons(self, *, lightweight=False):
        spec = self._recovery_run_spec or self._last_run_spec or {}
        result = dict(self._last_run_result or {})
        status = str(
            self._startup_recovery_status
            or result.get("status")
            or ""
        )
        valid_spec = int(spec.get("schema_version") or 0) == 2
        resumable = valid_spec and status in run_index.RECOVERABLE_RUN_STATES
        failed = valid_spec and status in {"failed", "resetting"}
        self.resume_btn.setEnabled(resumable and not self.workflow.is_active)
        self.retry_failed_btn.setEnabled(
            failed and not self.workflow.is_active
        )

    def _resume_existing_run(self, retry_failed):
        if self._cleaning_up or self.workflow.is_active:
            return
        spec = self._recovery_run_spec or self._last_run_spec or {}
        spec_path = Path(str(spec.get("run_dir") or "")) / "run_spec.json"
        if int(spec.get("schema_version") or 0) != 2 or not spec_path.is_file():
            QMessageBox.warning(self, "恢复运行", "没有可恢复的任务")
            return
        if retry_failed:
            answer = QMessageBox.question(
                self,
                "重做失败包",
                "将删除失败 Work Package 的独占产物、受影响空间单元结果，"
                "以及已失效的全流组装/验收结果，然后从该包重新运行。\n\n"
                "共享 Tile 缓存会保留，人工确认数据不会被读取或修改。是否继续？",
                YES | NO,
                NO,
            )
            if answer != YES:
                return
        scripts_dir = self.environment_panel.scripts_directory
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)
        self.retry_failed_btn.setEnabled(False)
        self.workflow.resume(
            str(spec_path),
            scripts_dir,
            dict(spec),
            retry_failed=bool(retry_failed),
        )

    def _on_open_refinement(self):
        if (
            self._last_run_result is None
            and self._last_run_spec is None
            and self._startup_ready_candidate is not None
        ):
            result, spec = self._startup_ready_candidate
            declared_streams = list(result.get("ready_streams") or [])
            if not declared_streams:
                QMessageBox.warning(
                    self,
                    "最近 Run 不可用",
                    "最近 Run 没有声明 ready 结果流。\n\n"
                    "可使用“加载已有 Run 人工整理”明确选择其他 Run。",
                )
                return
            self._last_run_result = result
            self._last_run_spec = spec
            self._startup_ready_candidate = None
            self.open_refinement_btn.setText("打开分类修整与组装")
            self.result_summary_label.setText(
                f"Run {result['run_id']} 将在分类窗口中后台校验"
            )
        if self.refinement_dialog is None or not self._last_run_result or not self._last_run_spec:
            QMessageBox.warning(self, "分类修整", "当前没有可用于分类修整的推理结果")
            return
        effective = (self.config_manager.last_report or {}).get("effective") or {}
        self.refinement_dialog.set_run(
            self._last_run_result,
            self._last_run_spec,
            effective.get("sam3") or {},
            self.environment_panel.scripts_directory,
        )
        self.refinement_dialog.show()
        self.refinement_dialog.raise_()
        self.refinement_dialog.activateWindow()

    def _on_load_manual_run(self):
        if self.refinement_dialog is None:
            QMessageBox.warning(self, "人工分类整理", "分类整理窗口尚未初始化")
            return
        if self._manual_load_task is not None:
            task = self._manual_load_task
            if task.commit_started:
                self.result_summary_label.setText("正在发布 Run 元数据，不能取消")
                return
            self._retire_manual_load_task()
            self._manual_load_task = None
            self.load_manual_run_btn.setText("加载已有 Run 人工整理")
            self.result_summary_label.setText("已有 Run 加载已取消；界面保持原结果")
            return
        settings = QgsSettings()
        start_dir = settings.value(
            "plugins/labeling_tool/last_manual_run_dir",
            self.workspace_edit.text().strip(),
            type=str,
        )
        run_dir = QFileDialog.getExistingDirectory(
            self, "选择要人工整理的 Run 文件夹", start_dir
        )
        if not run_dir:
            return
        self._retire_manual_load_task()
        self._manual_load_generation += 1
        generation = self._manual_load_generation
        try:
            task = ManualRunLoadTask(generation, run_dir)
            self._manual_load_task = task
            task.progressChanged.connect(self._manual_load_progress)
            task.taskCompleted.connect(self._manual_load_completed)
            task.taskTerminated.connect(self._manual_load_terminated)
            self.load_manual_run_btn.setText("取消加载已有 Run")
            self.load_manual_run_btn.setEnabled(True)
            self.result_summary_label.setText("正在后台读取并校验已有 Run...")
            QgsApplication.taskManager().addTask(task)
        except Exception as exc:
            self._manual_load_task = None
            self.load_manual_run_btn.setText("加载已有 Run 人工整理")
            QMessageBox.warning(self, "加载 Run 失败", str(exc))

    def _manual_load_is_current(self, task):
        return bool(
            task is self._manual_load_task
            and task.request_id == self._manual_load_generation
        )

    def _manual_load_progress(self, _value):
        task = self.sender()
        if self._manual_load_is_current(task):
            self.result_summary_label.setText(task.progress_message)
            if task.commit_started:
                self.load_manual_run_btn.setText("正在发布，不能取消")
                self.load_manual_run_btn.setEnabled(False)

    def _manual_load_completed(self):
        task = self.sender()
        if not self._manual_load_is_current(task):
            return
        self._manual_load_task = None
        self.load_manual_run_btn.setText("加载已有 Run 人工整理")
        self.load_manual_run_btn.setEnabled(True)
        bundle = task.result_data or {}
        result = bundle.get("result") or {}
        run_spec = bundle.get("run_spec") or {}
        if (
            not task.published
            or not result
            or str(result.get("run_id") or "")
            != str(run_spec.get("run_id") or "")
        ):
            QMessageBox.warning(self, "加载 Run 失败", "后台任务未发布有效 Run")
            return
        self._last_run_result = result
        self._last_run_spec = run_spec
        self._recovery_run_spec = None
        self._startup_ready_candidate = None
        self._startup_recovery_status = None
        self.open_refinement_btn.setText("打开分类修整与组装")
        self.open_refinement_btn.setEnabled(True)
        mode_text = (
            "14 类离线工作区"
            if result.get("portable_classes_only")
            else "人工工作区"
        )
        self.result_summary_label.setText(
            f"已载入 {mode_text} {result['run_id']}；正在后台校验"
        )
        QgsSettings().setValue(
            "plugins/labeling_tool/last_manual_run_dir",
            task.run_directory,
        )
        self.refinement_dialog.set_run(result, run_spec, {}, "")
        self.refinement_dialog.show()
        self.refinement_dialog.raise_()
        self.refinement_dialog.activateWindow()

    def _manual_load_terminated(self):
        task = self.sender()
        if not self._manual_load_is_current(task):
            return
        self._manual_load_task = None
        self.load_manual_run_btn.setText("加载已有 Run 人工整理")
        self.load_manual_run_btn.setEnabled(True)
        if task.isCanceled() and not task.error_message:
            self.result_summary_label.setText("已有 Run 加载已取消；界面保持原结果")
        else:
            self.result_summary_label.setText("已有 Run 加载失败；界面保持原结果")
            QMessageBox.warning(
                self,
                "加载 Run 失败",
                task.error_message or "后台加载未完成",
            )

    def _retire_manual_load_task(self):
        task = self._manual_load_task
        self._manual_load_generation += 1
        if task is not None and not task.commit_started:
            task.cancel()

    # ── Helpers ──

    def _load_settings_and_defaults(self):
        settings = QgsSettings()
        project_path = QgsProject.instance().fileName()
        project_dir = os.path.dirname(project_path) if project_path else ""

        inference_path = settings.value(
            "plugins/labeling_tool/inference_path", "", type=str
        )
        if not inference_path and project_dir:
            candidates = (
                os.path.join(project_dir, "linux", "inference_scripts"),
                os.path.join(project_dir, "inference_scripts"),
            )
            inference_path = next(
                (candidate for candidate in candidates if os.path.isdir(candidate)),
                "",
            )

        output_path = settings.value(
            "plugins/labeling_tool/output_path", "", type=str
        )
        if not output_path and project_dir:
            output_path = os.path.join(project_dir, "output", "accepted_labels.gpkg")
        workspace = settings.value(
            "plugins/labeling_tool/output_workspace", "", type=str
        )
        if not workspace:
            workspace = os.path.dirname(output_path) if output_path else os.path.join(project_dir, "output")

        selected = settings.value("plugins/labeling_tool/selected_models", [], type=list)
        fusion_profile_id = settings.value(
            "plugins/labeling_tool/fusion_profile", "", type=str
        ) or None
        boundary_smoothing_enabled = settings.value(
            "plugins/labeling_tool/boundary_smoothing_enabled", True, type=bool
        )
        self.plan_panel.restore_selection(
            InferenceSelection(
                model_ids=tuple(str(item) for item in selected),
                fusion_profile_id=fusion_profile_id,
                boundary_smoothing_enabled=boundary_smoothing_enabled,
            )
        )
        self.tile_width_spin.setValue(512)
        self.tile_height_spin.setValue(512)
        self.overlap_spin.setValue(settings.value(
            "plugins/labeling_tool/tile_overlap_probability_blend", 192, type=int
        ))
        self.skip_accepted_check.setChecked(settings.value(
            "plugins/labeling_tool/skip_accepted", True, type=bool
        ))
        self.environment_panel.scripts_directory = inference_path
        self.workspace_edit.setText(workspace)
        self.output_path_edit.setText(output_path)

    def _save_settings(self, *_args):
        if self._restoring_settings:
            return
        settings = QgsSettings()
        settings.setValue(
            "plugins/labeling_tool/inference_path",
            self.environment_panel.scripts_directory,
        )
        settings.setValue(
            "plugins/labeling_tool/output_path",
            self.output_path_edit.text().strip(),
        )
        settings.setValue(
            "plugins/labeling_tool/output_workspace",
            self.workspace_edit.text().strip(),
        )
        selection = self.plan_panel.selection
        settings.setValue(
            "plugins/labeling_tool/selected_models", list(selection.model_ids)
        )
        settings.setValue(
            "plugins/labeling_tool/fusion_profile",
            selection.fusion_profile_id or "",
        )
        settings.setValue(
            "plugins/labeling_tool/boundary_smoothing_enabled",
            selection.boundary_smoothing_enabled,
        )
        settings.setValue(
            "plugins/labeling_tool/tile_width", self.tile_width_spin.value()
        )
        settings.setValue(
            "plugins/labeling_tool/tile_height", self.tile_height_spin.value()
        )
        settings.setValue(
            "plugins/labeling_tool/tile_overlap_probability_blend", self.overlap_spin.value()
        )
        settings.setValue(
            "plugins/labeling_tool/skip_accepted",
            self.skip_accepted_check.isChecked(),
        )

    def _update_extent_status(self, selection: RawRangeSelection):
        raw_extent = selection.extent
        if not is_valid_extent(raw_extent):
            self.extent_status_label.setText(
                f"{selection.mode}范围无效，请重新选择"
            )
            self._refresh_processing_extent_preview()
            return False

        try:
            raster = validate_raster_layer(self.raster_combo.currentLayer())
            extent = transform_extent(raw_extent, selection.crs, raster.crs())
            extent = intersect_extents(extent, raster.extent())
            if extent is None or not is_valid_extent(extent):
                self.extent_status_label.setText(
                    f"{selection.mode}范围未识别: 与影像层「{raster.name()}」没有重叠"
                )
                self._refresh_processing_extent_preview()
                return False
            self.extent_status_label.setText(
                self._extent_status_text(selection.mode, extent, raster)
            )
            self._refresh_processing_extent_preview()
            return True
        except ValueError as exc:
            raw_crs = selection.crs
            crs_name = (
                raw_crs.authid()
                if raw_crs and raw_crs.isValid()
                else "map CRS"
            )
            self.extent_status_label.setText(
                f"{selection.mode}范围已获取: {format_extent(raw_extent)} [{crs_name}]；"
                f"但尚未完成影像校验: {exc}"
            )
            self._refresh_processing_extent_preview()
            return False

    def _extent_status_text(self, mode, extent, raster):
        if mode == VECTOR_MODE:
            layer = self.vector_range_combo.currentLayer()
            name = layer.name() if layer is not None else "未选择"
            return (
                f"矢量范围: {name}；外包范围 {format_extent(extent)} "
                f"[{raster.crs().authid()}]；相交 Tile 用于处理，结果按矢量边界精确裁剪"
            )
        return (
            f"{mode}范围: {format_extent(extent)} "
            f"[{raster.crs().authid()}]"
        )

    def _selected_raw_selection(self, *, ensure_view=False, strict=False):
        if self.radio_view.isChecked():
            if ensure_view:
                return self.range_selection.ensure_current_view()
            return self.range_selection.selected(VIEW_MODE)
        if self.radio_rect.isChecked():
            return self.range_selection.selected(RECTANGLE_MODE)
        try:
            return self.range_selection.selected(
                VECTOR_MODE, self.vector_range_combo.currentLayer()
            )
        except ValueError:
            if strict:
                raise
            return RawRangeSelection(VECTOR_MODE, None, None)

    def _on_vector_range_data_changed(self, *_args):
        self._refresh_processing_extent_preview()

    def _on_vector_preview_progress(self, progress):
        self.processing_extent_status_label.setText(
            f"正在后台计算矢量范围 Tile... {int(progress)}%"
        )

    def _on_vector_preview_ready(self, result: VectorPreviewResult):
        try:
            raster = validate_raster_layer(self.raster_combo.currentLayer())
            self._set_processing_extent_summary(result, raster)
        except ValueError:
            return

    def _on_vector_preview_failed(self, message):
        self.processing_extent_status_label.setText(f"无法计算: {message}")
        self.processing_extent_status_label.setStyleSheet("color: #b42318;")

    def _set_processing_extent_status(self, tiles, raster, *, grid_tiles=None):
        full_grid = list(grid_tiles or tiles)
        processing_extent = tile_manager.get_grid_extent(full_grid)
        if processing_extent is None:
            self.processing_extent_status_label.setText("未生成完整 Tile")
            self.processing_extent_status_label.setStyleSheet("color: #b42318;")
            return
        summary = TileGridSummary(
            processing_extent=processing_extent,
            rows=max(int(tile["row"]) for tile in full_grid) + 1,
            cols=max(int(tile["col"]) for tile in full_grid) + 1,
            grid_count=len(full_grid),
            selected_count=len(tiles),
        )
        self._set_processing_extent_summary(summary, raster)

    def _set_processing_extent_summary(self, summary, raster):
        processing_extent = summary.processing_extent
        if processing_extent is None:
            self.processing_extent_status_label.setText("未生成完整 Tile")
            self.processing_extent_status_label.setStyleSheet("color: #b42318;")
            return
        step_width = self.tile_width_spin.value() - self.overlap_spin.value()
        step_height = self.tile_height_spin.value() - self.overlap_spin.value()
        rows = int(summary.rows)
        cols = int(summary.cols)
        effective = (self.config_manager.last_report or {}).get("effective") or {}
        scaling = effective.get("scaling") or {}
        seam = int(scaling.get("seam_band_px", 64))
        fragmentation = dict(effective.get("fragmentation_regularization") or {})
        fragmentation_buffer = (
            int(fragmentation.get("buffer_pixels", 256))
            if bool(fragmentation.get("enabled", True))
            else 0
        )
        raw_halo = scaling.get("partition_halo_px", "auto")
        halo = (
            max(self.overlap_spin.value(), seam, fragmentation_buffer)
            if str(raw_halo).lower() == "auto" else int(raw_halo)
        )
        halo = max(halo, fragmentation_buffer)
        try:
            spatial = plan_spatial_units(
                tile_rows=rows,
                tile_cols=cols,
                tile_size=512,
                overlap=self.overlap_spin.value(),
                partition_tile_rows=int(scaling.get("partition_tile_rows", 8)),
                partition_tile_cols=int(scaling.get("partition_tile_cols", 8)),
                seam_band_px=seam,
                halo_px=halo,
            )
            unit_counts = spatial["unit_counts"]
            unit_text = (
                f"Partition {unit_counts.get('core', 0)}；"
                f"Seam {unit_counts.get('seam_vertical', 0) + unit_counts.get('seam_horizontal', 0)}；"
                f"Junction {unit_counts.get('junction', 0)}"
            )
        except ValueError:
            unit_text = "空间单元待环境检查后计算"
        selected_count = int(summary.selected_count)
        grid_count = int(summary.grid_count)
        tile_summary = f"共 {selected_count} 个完整 Tile"
        excluded_count = grid_count - selected_count
        if excluded_count:
            tile_summary += f"；范围外排除 {excluded_count} 个"
        self.processing_extent_status_label.setText(
            f"{format_extent(processing_extent)} "
            f"[{raster.crs().authid()}]\n"
            f"{tile_summary}；步长 "
            f"{step_width} × {step_height} px；{unit_text}"
        )
        self.processing_extent_status_label.setStyleSheet("color: #1f6f3d;")

    def _refresh_processing_extent_preview(self):
        selection = self._selected_raw_selection()
        if not is_valid_extent(selection.extent):
            if selection.mode == VIEW_MODE:
                message = "开始时将自动读取当前视图范围；可先预览处理 Tile"
            else:
                message = f"请先获取{selection.mode}范围"
            self.processing_extent_status_label.setText(message)
            self.processing_extent_status_label.setStyleSheet("color: #666;")
            return
        try:
            raster = validate_raster_layer(self.raster_combo.currentLayer())
            extent = transform_extent(
                selection.extent, selection.crs, raster.crs()
            )
            extent = intersect_extents(extent, raster.extent())
            if extent is None or not is_valid_extent(extent):
                raise ValueError("绘图范围与当前影像没有重叠")
            if self.radio_vector.isChecked():
                layer = validate_vector_layer(
                    self.vector_range_combo.currentLayer()
                )
                request = self.vector_preview.make_request(
                    raster,
                    layer,
                    extent,
                    TileParameters(
                        self.tile_width_spin.value(),
                        self.tile_height_spin.value(),
                        self.overlap_spin.value(),
                    ),
                )
                preview = self.vector_preview.queue(request)
                if preview is not None:
                    self._set_processing_extent_summary(preview, raster)
                else:
                    self.processing_extent_status_label.setText(
                        "正在后台计算矢量范围 Tile..."
                    )
                    self.processing_extent_status_label.setStyleSheet(
                        "color: #805500;"
                    )
                return
            grid_tiles = tile_manager.generate_grid(
                extent,
                self.tile_width_spin.value(),
                self.tile_height_spin.value(),
                self.overlap_spin.value(),
                raster_layer=raster,
            )
            self._set_processing_extent_status(
                grid_tiles, raster, grid_tiles=grid_tiles
            )
        except ValueError as exc:
            self.processing_extent_status_label.setText(f"无法计算: {exc}")
            self.processing_extent_status_label.setStyleSheet("color: #b42318;")

    def _on_tile_parameters_changed(self, *_args):
        self.vector_preview.invalidate()
        maximum_overlap = max(
            0,
            min(self.tile_width_spin.value(), self.tile_height_spin.value()) - 1,
        )
        if self.overlap_spin.maximum() != maximum_overlap:
            self.overlap_spin.setMaximum(maximum_overlap)
        self._render_last_env_report()
        self._refresh_processing_extent_preview()

    def _on_raster_layer_changed(self, *_args):
        self.vector_preview.invalidate()
        selection = self._selected_raw_selection()
        if is_valid_extent(selection.extent):
            self._update_extent_status(selection)
        else:
            self._refresh_processing_extent_preview()

    def cleanup(self):
        self._cleaning_up = True
        self._retire_manual_load_task()
        self.environment_panel.cleanup()
        self.plan_panel.cleanup()
        for dialog in tuple(self._nonblocking_message_boxes):
            try:
                dialog.close()
            except RuntimeError:
                pass
        self._nonblocking_message_boxes.clear()
        self._save_settings()
        self.config_manager.cleanup()
        self.vector_preview.close()

        workflow = self.workflow
        for signal, slot in (
            (workflow.state_changed, self._on_workflow_state_changed),
            (workflow.stage_progress, self._apply_stage_progress),
            (workflow.runner_changed, self._on_workflow_runner_changed),
            (
                workflow.monitor_context_ready,
                self._on_workflow_monitor_context_ready,
            ),
            (workflow.pre_run_failed, self._on_workflow_pre_run_failed),
            (
                workflow.stopped_before_run,
                self._on_workflow_stopped_before_run,
            ),
            (workflow.finished, self._on_pipeline_finished),
        ):
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass

        if self.monitor_dialog is not None:
            try:
                self.monitor_dialog.detach()
            except (TypeError, RuntimeError):
                pass
        if self.refinement_dialog is not None:
            self.refinement_dialog.cleanup()
        if self.monitor_dialog is not None:
            self.monitor_dialog.shutdown()

        retire_after(workflow, workflow.shutdown_finished)
        workflow.shutdown()
        self.workflow = None

        self.range_selection.close()
        self.monitor_dialog = None
