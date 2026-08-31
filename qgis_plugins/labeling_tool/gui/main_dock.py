import logging
import os
import json
import shutil
from datetime import datetime
from pathlib import Path
import re

from qgis.PyQt.QtCore import QSize, QTimer, QUrl, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QApplication, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QFormLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox,
    QRadioButton, QCheckBox, QProgressBar,
    QTableWidget, QTableWidgetItem,
    QWidget, QButtonGroup, QFileDialog, QMessageBox, QScrollArea,
)
from qgis.PyQt.QtGui import QColor, QDesktopServices
from qgis.gui import QgsDockWidget, QgsMapLayerComboBox, QgsMapTool, QgsRubberBand
from qgis.core import (
    Qgis,
    QgsApplication, QgsProject, QgsVectorLayer, QgsRasterLayer,
    QgsCoordinateTransform,
    QgsPointXY, QgsRectangle, QgsSettings,
)

from ..core.layer_names import LAYER_NAMES
from ..qt_compat import (
    ALIGN_LEFT,
    ALIGN_VCENTER,
    EXTENDED_SELECTION,
    NO,
    NO_EDIT_TRIGGERS,
    RESIZE_TO_CONTENTS,
    SCROLLBAR_AS_NEEDED,
    SELECT_ROWS,
    STRETCH,
    TEXT_SELECTABLE_BY_MOUSE,
    USER_ROLE,
    WA_DELETE_ON_CLOSE,
    YES,
)


class _FixedMapLayerComboBox(QgsMapLayerComboBox):
    """Keep the native layer popup anchored inside a macOS dock widget."""

    def showPopup(self):
        super().showPopup()
        popup = self.view().window()
        if popup:
            popup.move(self.mapToGlobal(self.rect().bottomLeft()))


class RectangleMapTool(QgsMapTool):
    """在地图上拖拽绘制矩形范围。"""
    rect_finished = pyqtSignal(object)

    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        if canvas is None:
            self.rb = None
        else:
            self.rb = QgsRubberBand(canvas, Qgis.GeometryType.Polygon)
            self.rb.setStrokeColor(QColor(255, 50, 50))
            self.rb.setFillColor(QColor(255, 50, 50, 40))
            self.rb.setWidth(2)
        self.start_point = None

    def reset(self):
        self.start_point = None
        if self.rb:
            self.rb.reset(Qgis.GeometryType.Polygon)

    def canvasPressEvent(self, event):
        self.start_point = self.toMapCoordinates(event.pos())
        if self.rb:
            self.rb.reset(Qgis.GeometryType.Polygon)

    def canvasMoveEvent(self, event):
        if self.start_point is None or not self.rb:
            return
        end = self.toMapCoordinates(event.pos())
        self.rb.reset(Qgis.GeometryType.Polygon)
        pts = [
            self.start_point,
            QgsPointXY(end.x(), self.start_point.y()),
            end,
            QgsPointXY(self.start_point.x(), end.y()),
            self.start_point,
        ]
        for p in pts:
            self.rb.addPoint(p)

    def canvasReleaseEvent(self, event):
        if self.start_point is None:
            return
        end = self.toMapCoordinates(event.pos())
        xmin = min(self.start_point.x(), end.x())
        xmax = max(self.start_point.x(), end.x())
        ymin = min(self.start_point.y(), end.y())
        ymax = max(self.start_point.y(), end.y())
        rect = QgsRectangle(xmin, ymin, xmax, ymax)
        self.start_point = None
        self.rect_finished.emit(rect)

    def deactivate(self):
        self.reset()
        super().deactivate()


from ..core import (
    tile_manager,
    difference_filter,
    accepted_integrity,
    class_workspace,
    manual_run_loader,
)
from ..core.inference_config import InferenceConfigManager
from ..core.layer_manager import LayerManager

from ..gui.inference_monitor import InferenceMonitorDialog
from ..gui.inference_config_dialog import InferenceConfigDialog
from ..gui.class_refinement_dialog import ClassRefinementDialog
from ..core.v5_async_runner import V5AsyncInferenceRunner
from ..core.tile_cache_probe_runner import TileCacheProbeRunner
from ..core.model_registry import ModelRegistry
from ..core.run_builder_v5 import create_v5_run
from ..core.run_spec import (
    RESERVATION_FILE,
    reserve_run_directory,
    run_tile_cache_dir,
    sha256_file,
)
from ..core import run_index
from ..core.work_package_planner import (
    fusion_accumulator_atomic_overhead,
    fusion_accumulator_bytes_per_tile,
    permanent_output_reserve,
    resolve_frozen_tile_batch_size,
    storage_preflight,
)
from ..core.environment_report import compact_problem, format_check_details
from ..core.run_state_db import RunStateDB
from ..core.spatial_planner import plan_spatial_units

logger = logging.getLogger("labeling_tool.main_dock")


CHECK_LABELS = {
    "conda_env": "Conda 环境",
    "config_yaml": "配置文件",
    "semantic_version": "语义版本",
    "device": "计算设备",
    "sam3_enabled": "SAM3",
    "sam3_backend": "SAM3 实现",
    "sam3_checkpoint": "SAM3 权重",
    "sam3_version": "SAM3 版本",
    "sam_buffer_px": "SAM3 缓冲",
    "class_mapping": "类别映射",
    "index_to_code": "输出通道映射",
    "output_dir": "输出目录",
    "semantic_model_load": "语义模型加载",
    "sam3_model_load": "SAM3 模型加载",
    "model_load_check": "模型加载检查",
    "environment_process": "环境检查进程",
    "scripts_dir": "脚本目录",
    "tile_parameters": "Tile 参数",
    "output_path": "输出 GPKG",
}

_TILE_RE = re.compile(r"semantic_tile_(\d+)_(\d+)")


def _check_label(check):
    check_id = str(check.get("id", ""))
    if check_id.startswith("dependency_"):
        return "依赖: " + check_id.replace("dependency_", "")
    if check_id.startswith("file_"):
        return "文件: " + str(check.get("value", ""))
    if check_id.startswith("semantic_model_"):
        return "语义模型: " + check_id.replace("semantic_model_", "")
    if check_id.startswith("fusion_profile_"):
        return "融合配置: " + check_id.replace("fusion_profile_", "")
    if check_id.startswith("deprecated_"):
        return "废弃配置"
    return CHECK_LABELS.get(check_id, check_id or "检查项")

class LabelingDockWidget(QgsDockWidget):

    def __init__(self, parent=None, iface=None):
        super().__init__(parent)
        self.iface = iface
        self.layer_manager = LayerManager(iface) if iface else None
        self.config_manager = InferenceConfigManager(self)
        self.runner = None
        self._pipeline_running = False
        self._pipeline_state = "idle"
        self._pipeline_stage_total = 0
        self._tile_extractor = None
        self._tile_cache_probe = None
        self._pending_run = None
        self._cleaning_up = False
        self.monitor_dialog = InferenceMonitorDialog(self)
        self.monitor_dialog.stop_requested.connect(self._on_stop)
        self.config_dialog = InferenceConfigDialog(self)
        self.config_dialog.configuration_applied.connect(self._on_inference_configuration_applied)
        self.refinement_dialog = ClassRefinementDialog(
            self.iface, self.layer_manager, self
        ) if self.iface and self.layer_manager else None
        self._current_tiles = []
        self._vector_preview_task = None
        self._vector_preview_cache = None
        self._vector_preview_requested_key = None
        self._start_after_vector_preview = False
        self._observed_vector_range_layer = None
        self._vector_preview_timer = QTimer(self)
        self._vector_preview_timer.setSingleShot(True)
        self._vector_preview_timer.setInterval(180)
        self._vector_preview_timer.timeout.connect(
            self._start_vector_tile_preview
        )
        self._step_t0: dict = {}
        self._view_extent = None
        self._view_extent_crs = None
        self._hand_drawn_extent = None
        self._hand_drawn_extent_crs = None
        self._previous_map_tool = None
        self._selected_model_ids = []
        self._fusion_profile_id = None
        self._boundary_smoothing_enabled = True
        self._inference_plan_confirmed = False
        self._last_run_result = None
        self._last_run_spec = None
        self._recovery_run_spec = None
        self._startup_ready_candidate = None
        self._startup_recovery_status = None
        self._env_details_dialog = None
        self._rect_tool = RectangleMapTool(iface.mapCanvas()) if iface else None
        if self._rect_tool:
            self._rect_tool.rect_finished.connect(self._on_rect_finished)

        self.setWindowTitle("半自动标注工具")
        self.setObjectName("labelingDock")

        self._build_ui()
        self._connect_signals()
        self._load_settings_and_defaults()
        # Restoring completed work is independent from checking inference dependencies.
        QTimer.singleShot(0, self._restore_latest_ready_run)

    def _build_ui(self):
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── Header ──
        header = QLabel("半自动标注工具 v5")
        header.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(header)

        # ── Data source group ──
        source_group = QGroupBox("① 数据源与范围")
        source_group.setObjectName("sourceGroup")
        source_grid = QGridLayout(source_group)
        source_grid.setColumnStretch(0, 0)
        source_grid.setColumnStretch(1, 1)

        self.raster_combo = _FixedMapLayerComboBox()
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

        self.vector_range_combo = _FixedMapLayerComboBox()
        self.vector_range_combo.setFilters(Qgis.LayerFilter.PolygonLayer)
        self.vector_range_combo.setEnabled(False)
        self.vector_range_combo.setToolTip(
            "整个面图层用于筛选相交的完整 Tile；不会按矢量边界裁剪结果"
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

        self.extent_status_label = QLabel("点击「获取当前视图」确认范围")
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
        tile_group = QGroupBox("② 切片")
        tile_group.setObjectName("tileGroup")
        tile_layout = QFormLayout(tile_group)

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
            "请先选择本地影像并获取绘图范围"
        )
        self.processing_extent_status_label.setWordWrap(True)
        self.processing_extent_status_label.setStyleSheet("color: #666;")
        tile_layout.addRow("自动扩展推理范围:", self.processing_extent_status_label)

        layout.addWidget(tile_group)

        # ── Output group ──
        output_group = QGroupBox("③ 输出位置")
        output_group.setObjectName("outputGroup")
        output_layout = QFormLayout(output_group)

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
        output_layout.addRow("Accepted GPKG:", accepted_path_layout)

        self.skip_accepted_check = QCheckBox("跳过已确认区域")
        self.skip_accepted_check.setChecked(True)
        output_layout.addRow(self.skip_accepted_check)

        layout.addWidget(output_group)

        # ── Inference environment group ──
        infer_group = QGroupBox("④ 推理环境")
        infer_group.setObjectName("environmentGroup")
        infer_layout = QFormLayout(infer_group)

        script_path_layout = QHBoxLayout()
        self.script_path_edit = QLineEdit()
        self.script_path_edit.setPlaceholderText("inference_scripts/")
        self.browse_script_btn = QPushButton("选择")
        script_path_layout.addWidget(self.script_path_edit)
        script_path_layout.addWidget(self.browse_script_btn)
        infer_layout.addRow("推理脚本目录:", script_path_layout)

        config_path_layout = QHBoxLayout()
        self.config_path_label = QLabel("未选择")
        self.config_path_label.setWordWrap(True)
        self.config_path_label.setTextInteractionFlags(
            TEXT_SELECTABLE_BY_MOUSE
        )
        self.open_config_btn = QPushButton("打开")
        self.open_config_btn.setToolTip("打开当前 inference_scripts/config.yaml")
        self.open_config_btn.setEnabled(False)
        config_path_layout.addWidget(self.config_path_label, stretch=1)
        config_path_layout.addWidget(self.open_config_btn)
        infer_layout.addRow("配置文件:", config_path_layout)

        self.env_status_label = QLabel("未检查")
        self.env_status_label.setWordWrap(True)
        self.env_status_label.setTextInteractionFlags(
            TEXT_SELECTABLE_BY_MOUSE
        )
        self.env_status_label.setStyleSheet(
            "padding: 5px; border: 1px solid #b8b8b8; background: #f4f4f4;"
        )
        infer_layout.addRow("环境状态:", self.env_status_label)

        env_action_layout = QHBoxLayout()
        self.refresh_env_btn = QPushButton("检查推理环境")
        self.env_detail_btn = QPushButton("查看完整检查结果")
        self.env_detail_btn.setEnabled(False)
        env_action_layout.addWidget(self.refresh_env_btn)
        env_action_layout.addWidget(self.env_detail_btn)
        infer_layout.addRow(env_action_layout)

        self.env_table = QTableWidget(0, 4)
        self.env_table.setHorizontalHeaderLabels(
            ["检查项", "当前有效值", "状态", "来源 / 修改位置"]
        )
        self.env_table.setEditTriggers(NO_EDIT_TRIGGERS)
        self.env_table.setSelectionBehavior(
            SELECT_ROWS
        )
        self.env_table.verticalHeader().setVisible(False)
        self.env_table.horizontalHeader().setSectionResizeMode(
            0, RESIZE_TO_CONTENTS
        )
        self.env_table.horizontalHeader().setSectionResizeMode(
            1, STRETCH
        )
        self.env_table.horizontalHeader().setSectionResizeMode(
            2, RESIZE_TO_CONTENTS
        )
        self.env_table.horizontalHeader().setSectionResizeMode(
            3, STRETCH
        )
        self.env_table.setMinimumHeight(190)
        self.env_table.setVisible(False)

        layout.addWidget(infer_group)

        # ── Inference plan group ──
        plan_group = QGroupBox("⑤ 推理方案")
        plan_group.setObjectName("inferencePlanGroup")
        plan_layout = QVBoxLayout(plan_group)
        self.inference_summary_label = QLabel("请先完成推理环境检查")
        self.inference_summary_label.setWordWrap(True)
        plan_layout.addWidget(self.inference_summary_label)
        self.configure_inference_btn = QPushButton("选择模型与 Fusion")
        self.configure_inference_btn.setEnabled(False)
        self.configure_inference_btn.setToolTip("环境检查完成后选择本次模型和融合方案")
        plan_layout.addWidget(self.configure_inference_btn)
        layout.addWidget(plan_group)

        # ── Action buttons ──
        run_group = QGroupBox("⑥ 执行")
        run_group.setObjectName("runGroup")
        run_layout = QVBoxLayout(run_group)
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

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        run_layout.addWidget(self.progress_bar)
        layout.addWidget(run_group)

        result_group = QGroupBox("⑦ 结果")
        result_group.setObjectName("resultGroup")
        result_layout = QVBoxLayout(result_group)
        self.result_summary_label = QLabel("尚无运行结果")
        self.result_summary_label.setWordWrap(True)
        result_layout.addWidget(self.result_summary_label)

        self.open_refinement_btn = QPushButton("打开分类修整与组装")
        self.open_refinement_btn.setEnabled(False)
        result_layout.addWidget(self.open_refinement_btn)
        self.load_manual_run_btn = QPushButton("加载已有 Run 人工整理")
        self.load_manual_run_btn.setToolTip(
            "选择任意位置的 Run 副本；已有完整14类工作区时无需复制 Fusion 大文件，"
            "不检查推理环境，不运行 SAM3"
        )
        result_layout.addWidget(self.load_manual_run_btn)
        layout.addWidget(result_group)

        self.tile_table = QTableWidget(0, 4)
        self.tile_table.setHorizontalHeaderLabels(["Tile", "状态", "语义", "SAM3"])
        self.tile_table.horizontalHeader().setStretchLastSection(True)
        self.tile_table.setEditTriggers(NO_EDIT_TRIGGERS)
        layout.addWidget(self.tile_table)
        self.tile_table.setVisible(False)

        layout.addStretch()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(
            SCROLLBAR_AS_NEEDED
        )
        scroll_area.setVerticalScrollBarPolicy(SCROLLBAR_AS_NEEDED)
        scroll_area.setWidget(main_widget)
        self.setWidget(scroll_area)

    def _connect_signals(self):
        self.browse_script_btn.clicked.connect(self._on_browse_script)
        self.refresh_env_btn.clicked.connect(self._run_manual_env_check)
        self.open_config_btn.clicked.connect(self._open_config)
        self.env_detail_btn.clicked.connect(self._show_env_details)
        self.browse_output_btn.clicked.connect(self._on_browse_output)
        self.browse_workspace_btn.clicked.connect(self._on_browse_workspace)
        self.configure_inference_btn.clicked.connect(self._on_configure_inference)
        self.open_refinement_btn.clicked.connect(self._on_open_refinement)
        self.load_manual_run_btn.clicked.connect(self._on_load_manual_run)
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.resume_btn.clicked.connect(lambda: self._resume_existing_run(False))
        self.retry_failed_btn.clicked.connect(lambda: self._resume_existing_run(True))
        self.extent_group.buttonClicked.connect(self._on_extent_mode_changed)
        self.capture_view_btn.clicked.connect(self._capture_current_view_extent)
        self.draw_rect_btn.clicked.connect(self._on_draw_rect_clicked)
        self.raster_combo.layerChanged.connect(self._on_raster_layer_changed)
        self.vector_range_combo.layerChanged.connect(
            self._on_vector_range_layer_changed
        )
        self.script_path_edit.textChanged.connect(self._mark_env_check_required)
        self.output_path_edit.textChanged.connect(self._mark_env_check_required)
        self.workspace_edit.textChanged.connect(self._on_workspace_changed)
        self.tile_width_spin.valueChanged.connect(self._on_tile_parameters_changed)
        self.tile_height_spin.valueChanged.connect(self._on_tile_parameters_changed)
        self.overlap_spin.valueChanged.connect(self._on_tile_parameters_changed)
        self.skip_accepted_check.toggled.connect(self._save_settings)
        self.config_manager.check_started.connect(self._on_env_check_started)
        self.config_manager.report_ready.connect(self._on_env_report_ready)

    # ── Slots ──

    def _on_browse_script(self):
        project_dir = os.path.dirname(QgsProject.instance().fileName()) or ""
        path = QFileDialog.getExistingDirectory(self, "选择推理脚本目录", project_dir)
        if path:
            self.script_path_edit.setText(path)

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
        scripts_dir = self.script_path_edit.text().strip()
        config_path = os.path.join(scripts_dir, "config.yaml") if scripts_dir else ""
        self.config_path_label.setText(config_path or "未选择")
        self.open_config_btn.setEnabled(os.path.isfile(config_path))
        self.env_status_label.setText("配置已变化，请检查推理环境")
        self.env_status_label.setStyleSheet(
            "padding: 5px; border: 1px solid #c28b00; background: #fff7d6;"
        )
        self.inference_summary_label.setText("请先完成推理环境检查")
        self._inference_plan_confirmed = False
        self.configure_inference_btn.setEnabled(False)
        self.env_detail_btn.setEnabled(False)
        self.start_btn.setEnabled(False)

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

    def _run_manual_env_check(self, *_args):
        self._run_env_check()

    def _run_env_check(self):
        scripts_dir = self.script_path_edit.text().strip()
        output_dir = self.workspace_edit.text().strip()
        config_path = os.path.join(scripts_dir, "config.yaml") if scripts_dir else ""
        self.config_path_label.setText(config_path or "未选择")
        self.open_config_btn.setEnabled(os.path.isfile(config_path))
        self.config_manager.start_check(scripts_dir, output_dir)

    def _on_env_check_started(self):
        self.env_status_label.setText("正在检查实际推理环境，请稍候")
        self.env_status_label.setStyleSheet(
            "padding: 5px; border: 1px solid #4f83b6; background: #eaf4ff;"
        )
        self.env_table.setRowCount(0)
        self._inference_plan_confirmed = False
        self.refresh_env_btn.setEnabled(False)
        self.env_detail_btn.setEnabled(False)
        self.configure_inference_btn.setEnabled(False)
        self.start_btn.setEnabled(False)

    def _on_env_report_ready(self, report):
        effective = report.get("effective") or {}
        if effective.get("schema_version") == 2:
            try:
                registry = ModelRegistry(effective)
                checks_by_id = {
                    str(item.get("id")): item for item in report.get("checks") or []
                }
                available_ids = [
                    model.model_id for model in registry.models
                    if model.enabled
                    and (checks_by_id.get(f"semantic_model_{model.model_id}") or {}).get("status") == "ready"
                ]
                self._selected_model_ids = [
                    model_id for model_id in self._selected_model_ids if model_id in available_ids
                ] or available_ids
                if self._fusion_profile_id:
                    profile_ids = {profile.profile_id for profile in registry.profiles}
                    if self._fusion_profile_id not in profile_ids:
                        self._fusion_profile_id = None
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("无法建立模型注册表: %s", exc)
        self.config_dialog.set_environment(
            report,
            self._selected_model_ids,
            self._fusion_profile_id,
            self._boundary_smoothing_enabled,
        )
        self.configure_inference_btn.setEnabled(bool(effective.get("schema_version") == 2))
        self.refresh_env_btn.setEnabled(True)
        self._update_inference_summary()
        self._render_env_report(report)
        self._restore_latest_ready_run()
        self._save_settings()

    def _restore_latest_ready_run(self):
        if self._pipeline_running:
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

    def _render_env_report(self, report):
        checks = list(report.get("checks", []))
        checks.extend(self._get_task_parameter_checks())
        self.env_table.setRowCount(len(checks))

        status_text = {"ready": "正常", "warning": "警告", "error": "错误"}
        status_color = {"ready": "#1f6f3d", "warning": "#9a6700", "error": "#b42318"}
        for row, check in enumerate(checks):
            label = _check_label(check)

            source = str(check.get("source", ""))
            fix = str(check.get("fix", ""))
            source_fix = source if not fix else f"{source}\n{fix}"
            values = [
                label,
                str(check.get("value", "")),
                status_text.get(check.get("status"), str(check.get("status", ""))),
                source_fix,
            ]
            tooltip = str(check.get("message", ""))
            if fix:
                tooltip = (tooltip + "\n" if tooltip else "") + fix
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(tooltip)
                if col == 2:
                    item.setForeground(QColor(status_color.get(check.get("status"), "#333333")))
                self.env_table.setItem(row, col, item)

        self.env_table.resizeRowsToContents()
        task_checks = self._get_task_parameter_checks()
        status = report.get("status", "error")
        if any(check.get("status") == "error" for check in task_checks):
            status = "error"
        first_problem = self._first_problem({"checks": checks})
        if status == "ready":
            text = "检查通过：配置已加载，可以开始推理"
            style = "padding: 5px; border: 1px solid #2f855a; background: #e8f5ec;"
        elif status == "warning":
            text = "检查通过但有警告"
            if first_problem:
                text += f"：{first_problem}"
            style = "padding: 5px; border: 1px solid #c28b00; background: #fff7d6;"
        else:
            text = "检查未通过"
            if first_problem:
                text += f"：{first_problem}"
            style = "padding: 5px; border: 1px solid #b42318; background: #fff0ee;"
        self.env_status_label.setText(text)
        self.env_status_label.setStyleSheet(style)
        self.env_status_label.setToolTip(
            "完整错误已放入详情窗口，可滚动、选择并复制。"
            if first_problem else ""
        )
        self.env_detail_btn.setEnabled(bool(report.get("checks") or report.get("stderr")))
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

    def _first_problem(self, report):
        checks = report.get("checks", [])
        for wanted in ("error", "warning"):
            for check in checks:
                if check.get("status") == wanted:
                    return compact_problem(check)
        return ""

    def _update_start_enabled(self):
        report = self.config_manager.last_report or {}
        ready = (
            report.get("status") in ("ready", "warning")
            and not self.config_manager.is_stale(self.script_path_edit.text().strip())
            and bool(self.output_path_edit.text().strip())
            and bool(self.workspace_edit.text().strip())
            and bool(self._selected_model_ids)
            and self._inference_plan_confirmed
            and not self._pipeline_running
        )
        self.start_btn.setEnabled(ready)
        if ready:
            self.start_btn.setToolTip("启动本次标注任务")
        elif not self._inference_plan_confirmed:
            self.start_btn.setToolTip("请先检查环境并确认推理方案")

    def _on_configure_inference(self):
        report = self.config_manager.last_report or {}
        self.config_dialog.set_environment(
            report,
            self._selected_model_ids,
            self._fusion_profile_id,
            self._boundary_smoothing_enabled,
        )
        self.config_dialog.show()
        self.config_dialog.raise_()
        self.config_dialog.activateWindow()

    def _on_inference_configuration_applied(
        self,
        model_ids,
        profile_id,
        boundary_smoothing_enabled,
    ):
        self._selected_model_ids = list(model_ids)
        self._fusion_profile_id = profile_id
        self._boundary_smoothing_enabled = bool(boundary_smoothing_enabled)
        self._inference_plan_confirmed = True
        self._update_inference_summary()
        self._save_settings()
        self._update_start_enabled()

    def _update_inference_summary(self):
        report = self.config_manager.last_report or {}
        effective = report.get("effective") or {}
        by_id = {
            str(item.get("model_id")): str(item.get("display_name") or item.get("model_id"))
            for item in effective.get("semantic_models") or []
        }
        names = [by_id.get(model_id, model_id) for model_id in self._selected_model_ids]
        profile = self._fusion_profile_id or "无融合"
        device = (effective.get("runtime") or {}).get("effective_device", "未检查")
        boundary = effective.get("boundary_fitting") or {}
        if not self._boundary_smoothing_enabled:
            boundary_text = "关闭，保留原始像元边界"
        elif boundary.get("mode") == "divider_cubic_bspline_adaptive_v2":
            boundary_text = "公共分界线 B-Spline + 误差受限稀疏"
        else:
            boundary_text = "未确认"
        plan_status = "已确认" if self._inference_plan_confirmed else "待确认"
        self.inference_summary_label.setText(
            f"模型 {len(names)} 个: {', '.join(names) if names else '未选择'}\n"
            f"Fusion: {profile}    设备: {device}\n"
            f"边界拟合: {boundary_text}；空间预算: 启动前实测\n"
            f"方案状态: {plan_status}"
        )

    def _open_config(self):
        config_path = os.path.join(
            self.script_path_edit.text().strip(), "config.yaml"
        )
        if not os.path.isfile(config_path):
            QMessageBox.warning(self, "配置文件", "当前脚本目录中没有 config.yaml")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(config_path))

    def _show_env_details(self):
        from qgis.PyQt.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
            QPlainTextEdit, QTabWidget,
        )
        from qgis.PyQt.QtCore import QTimer

        report = self.config_manager.last_report or {}
        checks = list(report.get("checks") or []) + self._get_task_parameter_checks()
        full_text = format_check_details(checks, report.get("stderr"))

        current = self._env_details_dialog
        if current is not None:
            current.close()

        dialog = QDialog(self)
        dialog.setWindowTitle("推理环境完整检查结果")
        dialog.resize(980, 650)
        dialog.setMinimumSize(760, 480)
        dialog.setModal(False)
        dialog.setAttribute(WA_DELETE_ON_CLOSE)
        self._env_details_dialog = dialog

        layout = QVBoxLayout(dialog)

        counts = {
            status: sum(1 for check in checks if check.get("status") == status)
            for status in ("ready", "warning", "error")
        }
        effective = report.get("effective") or {}
        device = (effective.get("runtime") or {}).get("effective_device", "未确定")
        summary_label = QLabel(
            f"检查项 {len(checks)}  |  正常 {counts['ready']}  |  "
            f"警告 {counts['warning']}  |  错误 {counts['error']}  |  "
            f"语义设备 {device}"
        )
        summary_label.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(summary_label)

        tabs = QTabWidget()
        layout.addWidget(tabs, stretch=1)

        table_page = QWidget()
        table_layout = QVBoxLayout(table_page)
        table_layout.setContentsMargins(0, 0, 0, 0)
        checks_table = QTableWidget(0, 5)
        checks_table.setHorizontalHeaderLabels(
            ["状态", "检查项", "当前值", "说明", "来源 / 修改位置"]
        )
        checks_table.setEditTriggers(NO_EDIT_TRIGGERS)
        checks_table.setSelectionBehavior(
            SELECT_ROWS
        )
        checks_table.setSelectionMode(
            EXTENDED_SELECTION
        )
        checks_table.verticalHeader().setVisible(False)
        checks_table.verticalHeader().setDefaultSectionSize(30)
        header = checks_table.horizontalHeader()
        header.setSectionResizeMode(0, RESIZE_TO_CONTENTS)
        header.setSectionResizeMode(1, RESIZE_TO_CONTENTS)
        header.setSectionResizeMode(2, STRETCH)
        header.setSectionResizeMode(3, STRETCH)
        header.setSectionResizeMode(4, STRETCH)

        status_priority = {"error": 0, "warning": 1, "ready": 2}
        ordered_checks = sorted(
            enumerate(checks),
            key=lambda pair: (status_priority.get(pair[1].get("status"), 3), pair[0]),
        )
        checks_table.setRowCount(len(ordered_checks))
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
                _check_label(check),
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
                checks_table.setItem(row, column, item)
        table_layout.addWidget(checks_table)
        tabs.addTab(table_page, "全部检查项")

        text_edit = QPlainTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(full_text)
        text_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        tabs.addTab(text_edit, "完整文本 / 进程日志")

        btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("全选")
        copy_selected_btn = QPushButton("复制选中项")
        copy_btn = QPushButton("复制全部结果")
        close_btn = QPushButton("关闭")
        btn_layout.addStretch()
        btn_layout.addWidget(select_all_btn)
        btn_layout.addWidget(copy_selected_btn)
        btn_layout.addWidget(copy_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        def _set_copied(button):
            button.setText("已复制")
            original_text = (
                "复制选中项" if button is copy_selected_btn else "复制全部结果"
            )

            def _restore_text():
                try:
                    button.setText(original_text)
                except RuntimeError:
                    pass

            QTimer.singleShot(2000, _restore_text)

        def _copy():
            clipboard = QApplication.clipboard()
            clipboard.setText(full_text)
            _set_copied(copy_btn)

        def _copy_selected():
            selected_rows = sorted({index.row() for index in checks_table.selectionModel().selectedRows()})
            selected_checks = []
            for row in selected_rows:
                item = checks_table.item(row, 0)
                if item is not None:
                    selected_checks.append(
                        checks[item.data(USER_ROLE)]
                    )
            QApplication.clipboard().setText(
                format_check_details(selected_checks) if selected_checks else full_text
            )
            _set_copied(copy_selected_btn)

        def _select_all():
            if tabs.currentWidget() is text_edit:
                text_edit.selectAll()
            else:
                checks_table.selectAll()

        select_all_btn.clicked.connect(_select_all)
        copy_selected_btn.clicked.connect(_copy_selected)
        copy_btn.clicked.connect(_copy)
        close_btn.clicked.connect(dialog.close)

        def _clear_dialog_reference(*_args):
            if self._env_details_dialog is dialog:
                self._env_details_dialog = None

        dialog.destroyed.connect(_clear_dialog_reference)

        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_draw_rect_clicked(self):
        self.radio_rect.setChecked(True)
        self._activate_rect_tool()

    def _on_extent_mode_changed(self, button):
        vector_mode = button == self.radio_vector
        if not vector_mode:
            self._invalidate_vector_preview()
        self.vector_range_combo.setEnabled(vector_mode)
        self.vector_range_label.setEnabled(vector_mode)
        if vector_mode:
            self.capture_view_btn.setEnabled(False)
            self.draw_rect_btn.setEnabled(False)
            self._restore_previous_map_tool()
            self._on_vector_range_layer_changed(
                self.vector_range_combo.currentLayer()
            )
            return

        if button == self.radio_rect:
            self.capture_view_btn.setEnabled(False)
            self.draw_rect_btn.setEnabled(True)
            if self._hand_drawn_extent is None:
                self.extent_status_label.setText("点击「绘制范围」后在地图上拖拽矩形")
                self._refresh_processing_extent_preview()
            else:
                self._update_extent_status(
                    self._hand_drawn_extent,
                    self._hand_drawn_extent_crs,
                    "手绘矩形",
                )
            return

        self.capture_view_btn.setEnabled(True)
        self.draw_rect_btn.setEnabled(False)
        self._restore_previous_map_tool()
        if self._view_extent is None:
            self.extent_status_label.setText("点击「获取当前视图」确认范围")
            self._refresh_processing_extent_preview()
        else:
            self._update_extent_status(
                self._view_extent, self._view_extent_crs, "当前视图"
            )

    def _capture_current_view_extent(self):
        if not self.iface:
            return
        canvas = self.iface.mapCanvas()
        self.radio_view.setChecked(True)
        self._restore_previous_map_tool()
        self._view_extent = QgsRectangle(canvas.extent())
        self._view_extent_crs = canvas.mapSettings().destinationCrs()
        self._update_extent_status(
            self._view_extent, self._view_extent_crs, "当前视图"
        )

    def _activate_rect_tool(self):
        if not self.iface or not self._rect_tool:
            return
        canvas = self.iface.mapCanvas()
        current_tool = canvas.mapTool()
        if current_tool != self._rect_tool:
            self._previous_map_tool = current_tool
        self._hand_drawn_extent = None
        self._hand_drawn_extent_crs = canvas.mapSettings().destinationCrs()
        self._rect_tool.reset()
        canvas.setMapTool(self._rect_tool)
        self.extent_status_label.setText("请在地图上按住鼠标拖拽，释放后确定矩形范围")
        self._refresh_processing_extent_preview()

    def _on_rect_finished(self, rect):
        self._hand_drawn_extent = rect
        self._hand_drawn_extent_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
        self._update_extent_status(
            self._hand_drawn_extent, self._hand_drawn_extent_crs, "手绘矩形"
        )
        self._restore_previous_map_tool()

    def _on_vector_range_layer_changed(self, *_args):
        if not self.radio_vector.isChecked():
            return
        self._invalidate_vector_preview()
        try:
            layer = self._get_valid_vector_range_layer()
            self._watch_vector_range_layer(layer)
            raster = self._get_valid_raster_layer()
            extent = self._transform_extent(
                layer.extent(), layer.crs(), raster.crs()
            )
            extent = self._intersect_extent(extent, raster.extent())
            if extent is None or not self._is_extent_valid(extent):
                raise ValueError(
                    f"矢量图层「{layer.name()}」与影像层没有重叠"
                )
            self.extent_status_label.setText(
                f"矢量范围: {layer.name()}；相交 Tile 用于处理，结果按矢量边界精确裁剪"
            )
        except ValueError as exc:
            self.extent_status_label.setText(str(exc))
        self._refresh_processing_extent_preview()

    def _on_start(self):
        if self._pipeline_running:
            return
        report = self.config_manager.last_report
        scripts_dir = self.script_path_edit.text().strip()
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
                self._first_problem(report) or "请先修正推理环境中的错误。",
            )
            return
        if report.get("status") == "warning":
            answer = QMessageBox.question(
                self,
                "推理环境警告",
                (self._first_problem(report) or "当前配置存在警告。")
                + "\n\n是否继续本次推理？",
                YES | NO,
                NO,
            )
            if answer != YES:
                return
        effective = report.get("effective", {})
        try:
            registry = ModelRegistry(effective)
            resolved_ids = registry.resolve_selection(
                self._selected_model_ids, self._fusion_profile_id
            )
            checks_by_id = {
                str(item.get("id")): item for item in report.get("checks") or []
            }
            unavailable = [
                model_id for model_id in resolved_ids
                if (checks_by_id.get(f"semantic_model_{model_id}") or {}).get("status") != "ready"
            ]
            if unavailable:
                raise ValueError("模型未通过设备实测: " + ", ".join(unavailable))
        except (KeyError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "推理方案不可运行", str(exc))
            return

        try:
            raster = self._get_valid_raster_layer()
        except ValueError as e:
            QMessageBox.warning(self, "错误", str(e))
            return

        if not scripts_dir or not os.path.isdir(scripts_dir):
            QMessageBox.warning(self, "错误", "请设置有效的推理脚本路径")
            return

        canvas = self.iface.mapCanvas()
        if self._rect_tool and canvas.mapTool() == self._rect_tool:
            self._restore_previous_map_tool()

        output_gpkg = self.output_path_edit.text().strip()
        if not output_gpkg:
            QMessageBox.warning(self, "错误", "请设置输出 GPKG 路径")
            return
        output_gpkg = os.path.abspath(os.path.expanduser(output_gpkg))

        try:
            extent = self._get_extent_in_raster_crs(raster)
        except ValueError as e:
            QMessageBox.warning(self, "错误", str(e))
            return

        tile_width = self.tile_width_spin.value()
        tile_height = self.tile_height_spin.value()
        overlap = self.overlap_spin.value()
        if self.radio_vector.isChecked():
            preview = self._cached_vector_preview(raster, extent)
            if preview is None:
                self._start_after_vector_preview = True
                self.processing_extent_status_label.setText(
                    "正在后台计算矢量范围 Tile，完成后自动开始标注..."
                )
                self.processing_extent_status_label.setStyleSheet("color: #805500;")
                self._queue_vector_tile_preview(raster, extent, immediate=True)
                return
            grid_tiles = preview["grid_tiles"]
            self._current_tiles = preview["selected_tiles"]
        else:
            try:
                grid_tiles = tile_manager.generate_grid(
                    extent, tile_width, tile_height, overlap, raster_layer=raster
                )
                self._current_tiles = list(grid_tiles)
            except ValueError as exc:
                QMessageBox.warning(self, "切片范围无效", str(exc))
                return

        if not self._current_tiles:
            QMessageBox.warning(self, "错误", "未生成任何 tile，请检查范围和尺寸")
            return

        processing_extent = (
            preview["processing_extent"]
            if self.radio_vector.isChecked()
            else tile_manager.get_grid_extent(grid_tiles)
        )
        range_mode = self._selected_raw_extent()[0]
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
                self._current_tiles, raster, grid_tiles=grid_tiles
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
                accepted_validation = accepted_integrity.audit_accepted_layer(
                    target_accepted_layer,
                    overlap_tolerance=accepted_validation["overlap_tolerance"],
                    expected_crs=raster.crs(),
                )
                accepted_validation["source"] = "existing_target"
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "已确认区域审计失败",
                    "本次 Run 尚未创建。请先修复 accepted_labels：\n" + str(exc),
                )
                return
            if self.skip_accepted_check.isChecked():
                accepted_layer = target_accepted_layer

        # Skip decisions are deliberately deferred until the post-probe
        # accepted snapshot has been loaded and audited.  The live layer may
        # still change while the asynchronous real-Tile probe is running.
        skipped_tiles = []

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setRange(0, len(self._current_tiles))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("准备提取切片")
        # Tile details are persisted in PostgreSQL and paged in the
        # monitor. Never instantiate one widget row per Tile.
        self.tile_table.setRowCount(0)

        self.monitor_dialog.detach()
        if self.runner is not None:
            self.runner.deleteLater()
            self.runner = None
        try:
            self.runner = V5AsyncInferenceRunner(scripts_dir, parent=self)
        except FileNotFoundError as e:
            QMessageBox.warning(self, "错误", str(e))
            self._update_start_enabled()
            self.stop_btn.setEnabled(False)
            return

        self._pipeline_running = True
        self._pipeline_state = "extracting"
        self._pipeline_stage_total = 6
        self._pending_run = {
            "raster": raster,
            "scripts_dir": scripts_dir,
            "output_dir": output_dir,
            "output_gpkg": output_gpkg,
            "overlap": overlap,
            "tile_width": tile_width,
            "tile_height": tile_height,
            "extent": extent,
            "processing_extent": processing_extent,
            "effective": effective,
            "report": report,
            "accepted_layer": accepted_layer,
            "accepted_validation": accepted_validation,
            "skip_accepted": bool(self.skip_accepted_check.isChecked()),
            "selected_model_ids": list(resolved_ids),
            "fusion_profile_id": self._fusion_profile_id,
            "boundary_smoothing_enabled": self._boundary_smoothing_enabled,
            "skipped_tiles": skipped_tiles,
            "grid_tiles": grid_tiles,
            "active_tiles": list(self._current_tiles),
            "range_selection": self._range_selection_metadata(
                grid_tiles, self._current_tiles
            ),
        }

        self._pending_run["run_id"] = ""
        self._pending_run["run_dir"] = ""
        self._pending_run["accepted_snapshot"] = ""
        self._pending_run["range_snapshot"] = ""

        self.monitor_dialog.reset_run()
        self.monitor_dialog.attach_runner(self.runner)
        self.runner.pipeline_finished.connect(self._on_pipeline_finished)
        self.runner.stage_progress.connect(self._on_runner_stage_progress)
        self.runner.step_started.connect(self._on_tile_step_started)
        self.runner.step_finished.connect(self._on_tile_step_finished)
        self.monitor_dialog.show()
        self.monitor_dialog.raise_()
        self.show_monitor_btn.setChecked(True)
        self.show_monitor_btn.setText("隐藏监控")

        self._apply_stage_progress({
            "key": "extraction",
            "name": "准备按工作包读取影像",
            "index": 1,
            "stage_total": self._pipeline_stage_total,
            "current": 0,
            "total": 1,
            "message": "不再预切全部 Tile，推理时按 Work Package 即时读取",
        })
        QTimer.singleShot(0, lambda: self._on_tiles_extracted([]))

    def _on_tile_extraction_progress(self, current, total, message):
        self._apply_stage_progress({
            "key": "extraction",
            "name": "切片提取",
            "index": 1,
            "stage_total": self._pipeline_stage_total,
            "current": current,
            "total": total,
            "message": message,
        })

    def _on_tiles_extracted(self, tiles):
        self._release_tile_extractor()
        if self._pipeline_state != "extracting" or not self._pending_run:
            return
        try:
            ctx = self._pending_run
            active_tiles = sorted(
                ctx.get("active_tiles") or [],
                key=lambda item: (int(item["row"]), int(item["col"])),
            )
            if not active_tiles:
                raise ValueError("当前选择范围内没有可用于存储预检的 active Tile")
            sample = active_tiles[0]
            row = int(sample["row"])
            col = int(sample["col"])
            request = {
                "tile_id": f"{row}_{col}",
                "row_no": row,
                "col_no": col,
                "bounds": self._extent_as_dict(sample["bounds"]),
            }
            raster_path = ctx["raster"].source().split("|", 1)[0]
            probe = TileCacheProbeRunner(
                ctx["scripts_dir"], parent=self
            )
            probe.succeeded.connect(self._on_tile_cache_probe_ready)
            probe.failed.connect(self._on_tile_cache_probe_failed)
            self._tile_cache_probe = probe
            self._pipeline_state = "preflighting"
            self._apply_stage_progress({
                "key": "extraction",
                "name": "存储预检",
                "index": 1,
                "stage_total": self._pipeline_stage_total,
                "current": 0,
                "total": 1,
                "message": (
                    f"正在用正式物化路径测量真实 Tile ({row},{col}) 缓存字节"
                ),
            })
            probe.start(
                raster_path=raster_path,
                output_root=ctx["output_dir"],
                tile=request,
            )
        except Exception as exc:
            self._release_tile_cache_probe(cancel=True)
            self._finish_before_inference("Tile 存储预检失败", str(exc))

    def _on_tile_cache_probe_ready(self, measurement):
        self._release_tile_cache_probe()
        if self._pipeline_state != "preflighting" or not self._pending_run:
            return
        self._pending_run["tile_cache_sample"] = dict(measurement)
        self._start_inference_after_tile_cache_probe()

    def _on_tile_cache_probe_failed(self, message):
        self._release_tile_cache_probe()
        if self._pipeline_state != "preflighting" or not self._pending_run:
            return
        self._finish_before_inference("Tile 存储预检失败", str(message))

    def _start_inference_after_tile_cache_probe(self):
        if self._pipeline_state != "preflighting" or not self._pending_run:
            return
        self._pipeline_state = "inferencing"
        ctx = self._pending_run
        effective = ctx["effective"]
        report = ctx["report"]
        try:
            registry = ModelRegistry(effective)
            selected_ids = registry.resolve_selection(
                ctx["selected_model_ids"],
                ctx["fusion_profile_id"],
            )
            selected_models = [vars(registry.model(model_id)) for model_id in selected_ids]
            fusion = None
            if ctx["fusion_profile_id"]:
                registered_profile = registry.profile(ctx["fusion_profile_id"])
                fusion = {
                    "profile_id": registered_profile.profile_id,
                    "version": str(registered_profile.profile.get("version") or ""),
                    "profile_path": registered_profile.file_path,
                    "profile": dict(registered_profile.profile),
                }
            scaling = dict(registry.scaling)
            fragmentation = dict(effective.get("fragmentation_regularization") or {})
            fragmentation_buffer = (
                int(fragmentation.get("buffer_pixels", 256))
                if bool(fragmentation.get("enabled", True))
                else 0
            )
            if str(scaling.get("partition_halo_px", "auto")).lower() == "auto":
                scaling["partition_halo_px"] = max(
                    int(ctx["overlap"]),
                    int(scaling.get("seam_band_px", 64)),
                    fragmentation_buffer,
                )
            else:
                scaling["partition_halo_px"] = max(
                    int(scaling["partition_halo_px"]), fragmentation_buffer
                )
            pixel_count = 512 * 512
            tile_cache_sample = dict(ctx.get("tile_cache_sample") or {})
            sample_tile_bytes = int(
                tile_cache_sample.get("materialized_cache_bytes") or 0
            )
            if sample_tile_bytes <= 0:
                raise ValueError("真实 Tile 探针没有返回有效缓存字节数")
            stream_count = len(selected_models) + (1 if fusion else 0)
            grid_tiles = list(ctx.get("grid_tiles") or [])
            tile_rows = max(int(tile["row"]) for tile in grid_tiles) + 1
            tile_cols = max(int(tile["col"]) for tile in grid_tiles) + 1
            spatial_plan = plan_spatial_units(
                tile_rows=tile_rows,
                tile_cols=tile_cols,
                tile_size=512,
                overlap=int(ctx["overlap"]),
                partition_tile_rows=int(scaling["partition_tile_rows"]),
                partition_tile_cols=int(scaling["partition_tile_cols"]),
                seam_band_px=int(scaling["seam_band_px"]),
                halo_px=int(scaling["partition_halo_px"]),
            )
            permanent = permanent_output_reserve(
                spatial_plan,
                stream_count=stream_count,
            )
            resolved_resources = (
                (effective.get("resource_tuning") or {}).get("resolved") or {}
            )
            tile_batch_size = resolve_frozen_tile_batch_size(
                registry.runtime["tile_batch_size"],
                resolved_resources.get("tile_batch_size"),
            )
            selected_batch_sizes = [
                max(
                    1,
                    int(
                        (
                            resolved_resources.get("tile_batch_size_by_model")
                            or {}
                        ).get(model["model_id"], tile_batch_size)
                    ),
                )
                for model in selected_models
            ]
            storage_batch_size = max(selected_batch_sizes, default=tile_batch_size)
            # The real Tile probe has already succeeded.  Freeze accepted data
            # before measuring free disk so its actual bytes are included in
            # the same-filesystem preflight.  Any failure before run_spec.json
            # is written removes this marker-backed reservation.
            run_id, run_dir = reserve_run_directory(ctx["output_dir"])
            ctx["run_id"] = run_id
            ctx["run_dir"] = str(run_dir)
            self._freeze_pending_range_snapshot(ctx, run_dir)
            self._freeze_pending_accepted_snapshot(ctx, run_dir)
            storage = storage_preflight(
                ctx["output_dir"],
                tile_count=len(ctx.get("active_tiles") or []),
                stream_count=stream_count,
                permanent_raster_bytes=permanent["permanent_raster_bytes"],
                vector_output_reserve_bytes=permanent[
                    "vector_output_reserve_bytes"
                ],
                permanent_core_pixel_count=permanent["core_pixel_count"],
                input_tile_bytes_per_tile=sample_tile_bytes,
                score_cache_budget_gb=scaling["score_cache_budget_gb"],
                min_free_disk_gb=float(scaling["min_free_disk_gb"]),
                current_model_probability_bytes=pixel_count * 14 * 2,
                fusion_accumulator_bytes=fusion_accumulator_bytes_per_tile(
                    (fusion or {}).get("profile"),
                    pixel_count=pixel_count,
                ),
                mask_confidence_workspace_bytes=pixel_count * (14 * 4 + 5),
                safety_margin_bytes=sample_tile_bytes,
                # During atomic replacement, the committed checkpoint and the
                # next full Batch temporary NPY may coexist briefly.
                fixed_temporary_overhead_bytes=(
                    pixel_count
                    * 14
                    * 2
                    * storage_batch_size
                ),
                fusion_atomic_write_overhead_bytes=(
                    fusion_accumulator_atomic_overhead(
                        (fusion or {}).get("profile"),
                        spatial_plan,
                    )
                ),
                tile_batch_size=storage_batch_size,
            )
            storage["input_tile_sample"] = tile_cache_sample
            scaling["score_cache_budget_mode"] = storage[
                "score_cache_budget_mode"
            ]
            scaling["score_cache_budget_gb"] = storage[
                "resolved_score_cache_budget_gb"
            ]
            stride = 512 - int(ctx["overlap"])
            accepted_tile_ids = {
                (int(tile["row"]), int(tile["col"]))
                for tile in ctx.get("skipped_tiles") or []
            }
            selected_tile_keys = {
                (int(tile["row"]), int(tile["col"]))
                for tile in ctx.get("active_tiles") or []
            }
            tile_cache_dir = run_tile_cache_dir(
                ctx["output_dir"], ctx["run_id"]
            )
            normalized_tiles = []
            for tile in grid_tiles:
                bounds = tile["bounds"]
                tile_key = (int(tile["row"]), int(tile["col"]))
                if tile_key not in selected_tile_keys:
                    tile_path = ""
                    tile_sha256 = ""
                    tile_status = "excluded"
                else:
                    tile_path = str(
                        tile_cache_dir / f"tile_{tile_key[0]}_{tile_key[1]}.tif"
                    )
                    tile_sha256 = ""
                    tile_status = (
                        "accepted" if tile_key in accepted_tile_ids else "ready"
                    )
                normalized_tiles.append({
                    "row": int(tile["row"]),
                    "col": int(tile["col"]),
                    "path": tile_path,
                    "sha256": tile_sha256,
                    "bounds": self._extent_as_dict(bounds),
                    "pixel_window": {
                        "x0": int(tile["col"]) * stride,
                        "y0": int(tile["row"]) * stride,
                        "x1": int(tile["col"]) * stride + 512,
                        "y1": int(tile["row"]) * stride + 512,
                    },
                    "status": tile_status,
                })
            processing_extent = ctx["processing_extent"]
            res_x = abs(ctx["raster"].rasterUnitsPerPixelX())
            res_y = abs(ctx["raster"].rasterUnitsPerPixelY())
            spec, spec_path, database_path = create_v5_run(
                output_root=ctx["output_dir"],
                reserved_run_dir=ctx["run_dir"],
                run_id=ctx["run_id"],
                raster={
                    "path": ctx["raster"].source().split("|", 1)[0],
                    "crs": ctx["raster"].crs().authid(),
                    "transform": [
                        res_x, 0.0, processing_extent.xMinimum(),
                        0.0, -res_y, processing_extent.yMaximum(),
                    ],
                    "nodata": None,
                },
                requested_extent=self._extent_as_dict(ctx["extent"]),
                processing_extent=self._extent_as_dict(processing_extent),
                tile_rows=tile_rows,
                tile_cols=tile_cols,
                tiles=normalized_tiles,
                models=selected_models,
                effective_device=(effective.get("runtime") or {}).get("effective_device", "cpu"),
                keep_score_cache=bool(
                    (effective.get("runtime") or {}).get("keep_score_cache", False)
                ),
                tile_batch_size=tile_batch_size,
                resource_tuning=effective.get("resource_tuning") or {},
                overlap=ctx["overlap"],
                scaling=scaling,
                boundary_fitting={
                    **registry.boundary_fitting,
                    "enabled": bool(ctx["boundary_smoothing_enabled"]),
                },
                fragmentation_regularization=(
                    effective.get("fragmentation_regularization") or {}
                ),
                storage_report=storage,
                fusion=fusion,
                accepted_gpkg=ctx.get("accepted_snapshot") or "",
                accepted_target_gpkg=ctx["output_gpkg"],
                accepted_validation=ctx.get("accepted_validation") or {},
                skip_accepted=bool(ctx.get("skip_accepted", False)),
                config_fingerprint=report.get("config_fingerprint", ""),
                range_selection=ctx.get("range_selection") or {},
            )
            self.monitor_dialog.bind_state_database(
                database_path,
                spec["run_id"],
                page_size=int(scaling.get("tile_page_size", 500)),
                run_spec=spec,
            )
            self.runner.run_from_spec(
                str(spec_path),
                accepted_layer=ctx["accepted_layer"],
            )
        except Exception as exc:
            logger.exception("启动推理异常: %s", exc)
            self._finish_before_inference("启动推理失败", str(exc))

    def _on_runner_stage_progress(self, info):
        whole = dict(info)
        whole["index"] = int(info.get("index", 0)) + 1
        whole["stage_total"] = self._pipeline_stage_total
        self._apply_stage_progress(whole)

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

    def _fmt_elapsed(self, name: str) -> str:
        import time as _time
        t0 = self._step_t0.pop(name, None)
        if t0 is None:
            return ""
        secs = _time.time() - t0
        if secs < 60:
            return f"{secs:.1f}s"
        m, s = divmod(secs, 60)
        return f"{int(m)}m {int(s)}s"

    def _on_tile_step_started(self, name):
        import time as _time
        self._step_t0[name] = _time.time()
        m = _TILE_RE.match(name)
        if not m:
            return
        rc = (int(m.group(1)), int(m.group(2)))
        row = self._find_tile_row(*rc)
        if row >= 0:
            item = QTableWidgetItem("● 运行中")
            item.setForeground(QColor("#2a82da"))
            self.tile_table.setItem(row, 1, item)
            self._last_tile_row = row

    def _on_tile_step_finished(self, name, return_code, result):
        m = _TILE_RE.match(name)
        if m:
            coords = (int(m.group(1)), int(m.group(2)))
            row = self._find_tile_row(*coords)
            if row < 0:
                return
            elapsed = self._fmt_elapsed(name)
            if return_code == 0:
                item = QTableWidgetItem("✓ 成功")
                item.setForeground(QColor("#2e7d32"))
                self.tile_table.setItem(row, 1, item)
                elapsed_item = QTableWidgetItem(f"成功 {elapsed}" if elapsed else "✓ 成功")
                elapsed_item.setForeground(QColor("#2e7d32"))
                self.tile_table.setItem(row, 2, elapsed_item)
            elif return_code == 42:
                item = QTableWidgetItem("⊕ 跳过")
                item.setForeground(QColor("#9e9e9e"))
                self.tile_table.setItem(row, 1, item)
                elapsed_item = QTableWidgetItem(f"跳过 {elapsed}" if elapsed else "⊕ 跳过")
                elapsed_item.setForeground(QColor("#9e9e9e"))
                self.tile_table.setItem(row, 2, elapsed_item)
            else:
                item = QTableWidgetItem("✗ 失败")
                item.setForeground(QColor("#c62828"))
                self.tile_table.setItem(row, 1, item)
                elapsed_item = QTableWidgetItem(f"失败 {elapsed}" if elapsed else "✗ 失败")
                elapsed_item.setForeground(QColor("#c62828"))
                self.tile_table.setItem(row, 2, elapsed_item)
        elif name.startswith("sam3"):
            elapsed = self._fmt_elapsed(name)
            ok = result.get("success", False) or (return_code == 0)
            if hasattr(self, '_last_tile_row') and self._last_tile_row is not None:
                elapsed_item = QTableWidgetItem(elapsed)
                elapsed_item.setForeground(QColor("#2e7d32" if ok else "#c62828"))
                self.tile_table.setItem(self._last_tile_row, 3, elapsed_item)

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

    def _on_tile_extraction_failed(self, message):
        self._release_tile_extractor()
        self._finish_before_inference("切片失败", message)

    def _on_tile_extraction_stopped(self):
        self._release_tile_extractor()
        self._release_tile_cache_probe(cancel=True)
        self._discard_pending_run_reservation()
        self._pipeline_running = False
        self._pipeline_state = "finished"
        self._pending_run = None
        self.stop_btn.setEnabled(False)
        self._set_progress_terminal("已停止")
        if self.monitor_dialog is not None:
            self.monitor_dialog.mark_finished("已停止")
        self._update_start_enabled()

    def _release_tile_extractor(self):
        extractor = self._tile_extractor
        self._tile_extractor = None
        if extractor is not None:
            extractor.deleteLater()

    def _release_tile_cache_probe(self, *, cancel=False):
        probe = self._tile_cache_probe
        self._tile_cache_probe = None
        if probe is not None:
            if cancel:
                probe.cleanup()
            probe.deleteLater()

    def _freeze_pending_accepted_snapshot(self, ctx, run_dir):
        """Freeze, re-audit and derive skip state from one accepted identity."""
        live_layer = ctx.get("accepted_layer")
        if live_layer is None:
            ctx["accepted_snapshot"] = ""
            ctx["skipped_tiles"] = []
            return
        snapshot_path = Path(run_dir) / "accepted_snapshot.gpkg"
        ctx["accepted_snapshot"] = difference_filter.snapshot_accepted_layer(
            live_layer,
            snapshot_path,
        )
        frozen_layer = QgsVectorLayer(
            f"{snapshot_path}|layername={LAYER_NAMES.ACCEPTED}",
            f"{ctx.get('run_id', '')} accepted snapshot",
            "ogr",
        )
        tolerance = float(
            (ctx.get("accepted_validation") or {}).get(
                "overlap_tolerance", 1.0e-18
            )
        )
        frozen_validation = accepted_integrity.audit_accepted_layer(
            frozen_layer,
            overlap_tolerance=tolerance,
            expected_crs=ctx["raster"].crs(),
        )
        frozen_validation["source"] = "run_snapshot"
        skipped_tiles = []
        for tile in ctx.get("active_tiles") or []:
            if difference_filter.tile_is_fully_accepted(
                tile["bounds"], frozen_layer, ctx["raster"].crs()
            ):
                skipped = dict(tile)
                skipped["skip_reason"] = "fully_accepted"
                skipped_tiles.append(skipped)
        ctx["accepted_layer"] = frozen_layer
        ctx["accepted_validation"] = frozen_validation
        ctx["skipped_tiles"] = skipped_tiles

    def _freeze_pending_range_snapshot(self, ctx, run_dir):
        """Freeze the exact vector boundary used by every later runtime stage."""
        selection = dict(ctx.get("range_selection") or {})
        if selection.get("mode") != "vector_tile_intersection":
            ctx["range_snapshot"] = ""
            return
        live_layer = self._get_valid_vector_range_layer()
        snapshot_path = Path(run_dir) / "range_snapshot.gpkg"
        difference_filter.snapshot_vector_layer(
            live_layer,
            snapshot_path,
            layer_name="range_mask",
        )
        frozen_layer = QgsVectorLayer(
            f"{snapshot_path}|layername=range_mask",
            f"{ctx.get('run_id', '')} range snapshot",
            "ogr",
        )
        if not frozen_layer.isValid() or not frozen_layer.crs().isValid():
            raise ValueError("范围矢量快照无效或缺少可转换 CRS")
        selected_tiles = tile_manager.select_tiles_intersecting_vector(
            ctx.get("grid_tiles") or [],
            frozen_layer,
            ctx["raster"].crs(),
        )
        if not selected_tiles:
            raise ValueError("冻结的范围矢量没有选中任何完整 Tile")
        selection.update(
            {
                "vector_source": str(snapshot_path),
                "vector_path": str(snapshot_path),
                "vector_sha256": sha256_file(snapshot_path),
                "vector_crs": frozen_layer.crs().authid(),
                "clip_outputs": True,
                "selected_tile_count": len(selected_tiles),
                "excluded_tile_count": len(ctx.get("grid_tiles") or []) - len(selected_tiles),
            }
        )
        ctx["range_snapshot"] = str(snapshot_path)
        ctx["range_selection"] = selection
        ctx["active_tiles"] = selected_tiles

    def _discard_pending_run_reservation(self):
        """Remove only this attempt's unused, marker-backed Run reservation."""
        ctx = self._pending_run or {}
        run_id = str(ctx.get("run_id") or "")
        run_dir_value = str(ctx.get("run_dir") or "")
        output_value = str(ctx.get("output_dir") or "")
        if not run_id or not run_dir_value or not output_value:
            return
        output_root = Path(output_value).expanduser().resolve()
        expected_run_dir = output_root / "runs" / run_id
        configured_run_dir = Path(run_dir_value).expanduser()
        if configured_run_dir.is_symlink():
            logger.error("拒绝清理符号链接形式的 Run 预留目录: %s", configured_run_dir)
            return
        run_dir = configured_run_dir.resolve()
        if run_dir != expected_run_dir:
            logger.error("拒绝清理不匹配的 Run 预留目录: %s", run_dir)
            return
        marker = run_dir / RESERVATION_FILE
        if (
            marker.is_symlink()
            or not marker.is_file()
            or (run_dir / "run_spec.json").exists()
        ):
            return
        try:
            reservation = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.error("拒绝清理身份无效的 Run 预留目录: %s", run_dir)
            return
        if str(reservation.get("run_id") or "") != run_id:
            logger.error("拒绝清理身份不匹配的 Run 预留目录: %s", run_dir)
            return
        cache_root = run_tile_cache_dir(output_root, run_id).parent
        for candidate in (cache_root, run_dir):
            try:
                if candidate.is_symlink():
                    logger.error("拒绝清理符号链接形式的 Run 预留目录: %s", candidate)
                elif candidate.exists():
                    shutil.rmtree(candidate)
            except OSError as exc:
                logger.warning("清理未使用的 Run 预留目录失败 %s: %s", candidate, exc)
        ctx["run_id"] = ""
        ctx["run_dir"] = ""
        ctx["accepted_snapshot"] = ""
        ctx["range_snapshot"] = ""

    def _finish_before_inference(self, title, message):
        self._release_tile_cache_probe(cancel=True)
        self._discard_pending_run_reservation()
        self._pipeline_running = False
        self._pipeline_state = "finished"
        self._pending_run = None
        self.stop_btn.setEnabled(False)
        self._set_progress_terminal(title)
        if self.monitor_dialog is not None:
            self.monitor_dialog.mark_finished(title)
        self._update_start_enabled()
        if not self._cleaning_up:
            QMessageBox.critical(self, title, message)

    def _set_progress_terminal(self, text, completed=False):
        """Stop indeterminate animation and display a stable terminal state."""
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1 if completed else 0)
        self.progress_bar.setFormat(text)

    def _on_stop(self):
        if self._pipeline_state not in (
            "extracting", "preflighting", "inferencing"
        ):
            if self.monitor_dialog is not None:
                self.monitor_dialog.mark_finished("无任务运行")
            return
        self.stop_btn.setEnabled(False)
        if self.monitor_dialog is not None:
            self.monitor_dialog.mark_stopping()
        if self._pipeline_state == "extracting" and self._tile_extractor:
            self._pipeline_state = "stopping"
            self._tile_extractor.stop()
        elif self._pipeline_state == "preflighting" and self._tile_cache_probe:
            self._pipeline_state = "stopping"
            self._release_tile_cache_probe(cancel=True)
            self._on_tile_extraction_stopped()
        elif self._pipeline_state == "inferencing" and self.runner:
            self._pipeline_state = "stopping"
            self.runner.stop()

    def _on_pipeline_finished(self, result):
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

        self._pipeline_running = False
        self._pipeline_state = "finished"
        self._pending_run = None
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
            QMessageBox.information(
                self,
                "完成",
                f"推理完成，已加载 {len(result.get('ready_streams') or [])} 个结果流",
            )
        elif not stopped:
            run_report = result.get("run_report", "")
            msg = result.get("error") or "推理流程失败"
            if run_report:
                msg += f"\n\n运行报告: {run_report}"
            QMessageBox.warning(self, "推理失败", msg)

    def _update_recovery_buttons(self, *, lightweight=False):
        if lightweight:
            status = str(self._startup_recovery_status or "")
            resumable = status in run_index.RECOVERABLE_RUN_STATES
            self.resume_btn.setEnabled(resumable and not self._pipeline_running)
            self.retry_failed_btn.setEnabled(
                status == "failed" and not self._pipeline_running
            )
            return
        resumable = False
        failed = False
        spec = self._recovery_run_spec or self._last_run_spec or {}
        try:
            if int(spec.get("schema_version") or 0) == 2:
                database = RunStateDB(spec["state_db"])
                run = database.get_run(spec["run_id"]) or {}
                status = str(run.get("status") or "")
                resumable = status in {
                    "planned", "stopped", "failed", "running",
                }
                counts = database.job_counts(spec["run_id"])
                failed = bool(counts.get("failed"))
                if status == "resetting":
                    failed = True
        except Exception:
            resumable = False
            failed = False
        self.resume_btn.setEnabled(resumable and not self._pipeline_running)
        self.retry_failed_btn.setEnabled(
            failed and not self._pipeline_running
        )

    def _resume_existing_run(self, retry_failed):
        spec = self._recovery_run_spec or self._last_run_spec or {}
        spec_path = Path(str(spec.get("run_dir") or "")) / "run_spec.json"
        if int(spec.get("schema_version") or 0) != 2 or not spec_path.is_file():
            QMessageBox.warning(self, "恢复运行", "没有可恢复的 v5 运行")
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
        scripts_dir = self.script_path_edit.text().strip()
        try:
            self.runner = V5AsyncInferenceRunner(scripts_dir, parent=self)
            self.monitor_dialog.reset_run()
            self.monitor_dialog.bind_state_database(
                spec["state_db"],
                spec["run_id"],
                page_size=int((spec.get("scaling") or {}).get("tile_page_size", 500)),
                run_spec=spec,
            )
            self.monitor_dialog.attach_runner(self.runner)
            self.runner.pipeline_finished.connect(self._on_pipeline_finished)
            self.runner.stage_progress.connect(self._on_runner_stage_progress)
            self.runner.step_started.connect(self._on_tile_step_started)
            self.runner.step_finished.connect(self._on_tile_step_finished)
            self._pipeline_running = True
            self._pipeline_state = "inferencing"
            self._pipeline_stage_total = 6
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.resume_btn.setEnabled(False)
            self.retry_failed_btn.setEnabled(False)
            self.monitor_dialog.show()
            self.monitor_dialog.raise_()
            if retry_failed:
                self.runner.retry_failed(str(spec_path))
            else:
                self.runner.resume(str(spec_path))
        except Exception as error:
            self._pipeline_running = False
            self._pipeline_state = "finished"
            self._update_start_enabled()
            self._update_recovery_buttons()
            QMessageBox.critical(self, "恢复运行失败", str(error))

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
            QMessageBox.warning(self, "分类修整", "当前没有可用的 semantic_ready 运行结果")
            return
        effective = (self.config_manager.last_report or {}).get("effective") or {}
        self.refinement_dialog.set_run(
            self._last_run_result,
            self._last_run_spec,
            effective.get("sam3") or {},
            self.script_path_edit.text().strip(),
        )
        self.refinement_dialog.show()
        self.refinement_dialog.raise_()
        self.refinement_dialog.activateWindow()

    def _on_load_manual_run(self):
        if self.refinement_dialog is None:
            QMessageBox.warning(self, "人工分类整理", "分类整理窗口尚未初始化")
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
        try:
            bundle = manual_run_loader.load_manual_run(run_dir)
            result = bundle["result"]
            run_spec = bundle["run_spec"]
            workspace = bundle.get("workspace")
            if workspace is not None:
                manual_run_loader.persist_rebound_workspace(bundle)
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
            settings.setValue(
                "plugins/labeling_tool/last_manual_run_dir", str(Path(run_dir).resolve())
            )
            self.refinement_dialog.set_run(result, run_spec, {}, "")
            self.refinement_dialog.show()
            self.refinement_dialog.raise_()
            self.refinement_dialog.activateWindow()
        except Exception as exc:
            QMessageBox.warning(self, "加载 Run 失败", str(exc))

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

        self.tile_width_spin.setValue(512)
        self.tile_height_spin.setValue(512)
        self.overlap_spin.setValue(settings.value(
            "plugins/labeling_tool/tile_overlap_probability_blend", 192, type=int
        ))
        self.skip_accepted_check.setChecked(settings.value(
            "plugins/labeling_tool/skip_accepted", True, type=bool
        ))
        self.script_path_edit.setText(inference_path)
        self.workspace_edit.setText(workspace)
        self.output_path_edit.setText(output_path)
        selected = settings.value("plugins/labeling_tool/selected_models", [], type=list)
        self._selected_model_ids = [str(item) for item in selected]
        self._fusion_profile_id = settings.value(
            "plugins/labeling_tool/fusion_profile", "", type=str
        ) or None
        self._boundary_smoothing_enabled = settings.value(
            "plugins/labeling_tool/boundary_smoothing_enabled", True, type=bool
        )

    def _save_settings(self, *_args):
        settings = QgsSettings()
        settings.setValue(
            "plugins/labeling_tool/inference_path",
            self.script_path_edit.text().strip(),
        )
        settings.setValue(
            "plugins/labeling_tool/output_path",
            self.output_path_edit.text().strip(),
        )
        settings.setValue(
            "plugins/labeling_tool/output_workspace",
            self.workspace_edit.text().strip(),
        )
        settings.setValue(
            "plugins/labeling_tool/selected_models", self._selected_model_ids
        )
        settings.setValue(
            "plugins/labeling_tool/fusion_profile", self._fusion_profile_id or ""
        )
        settings.setValue(
            "plugins/labeling_tool/boundary_smoothing_enabled",
            self._boundary_smoothing_enabled,
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

    def _restore_previous_map_tool(self):
        if not self.iface or not self._rect_tool:
            return

        canvas = self.iface.mapCanvas()
        if canvas.mapTool() != self._rect_tool:
            self._previous_map_tool = None
            return

        restored = False
        previous_tool = self._previous_map_tool
        if previous_tool and previous_tool != self._rect_tool:
            try:
                canvas.setMapTool(previous_tool)
                restored = True
            except RuntimeError:
                restored = False

        if not restored:
            try:
                pan_action = self.iface.actionPan()
                if pan_action:
                    pan_action.trigger()
                    restored = canvas.mapTool() != self._rect_tool
            except Exception:
                restored = False

        if not restored and canvas.mapTool() == self._rect_tool:
            canvas.unsetMapTool(self._rect_tool)

        self._rect_tool.reset()
        self._previous_map_tool = None

    def _get_valid_raster_layer(self):
        raster = self.raster_combo.currentLayer()
        if not raster or not isinstance(raster, QgsRasterLayer) or not raster.isValid():
            raise ValueError("请选择有效的本地影像层")

        provider = raster.providerType()
        if provider != "gdal":
            raise ValueError(
                f"当前影像层 provider={provider}，不能作为推理输入。"
                "请选择本地 GeoTIFF/栅格文件，不要选在线底图或 WMTS/XYZ 图层。"
            )

        if not raster.crs().isValid():
            raise ValueError(
                f"影像层「{raster.name()}」没有有效 CRS。请先在 QGIS 中设置/定义影像 CRS。"
            )

        source_path = raster.source().split("|", 1)[0]
        if source_path and not os.path.exists(source_path):
            raise ValueError(f"影像文件不存在: {source_path}")

        return raster

    def _get_valid_vector_range_layer(self):
        layer = self.vector_range_combo.currentLayer()
        if not layer or not isinstance(layer, QgsVectorLayer) or not layer.isValid():
            raise ValueError("请选择有效的已加载矢量面图层")
        if layer.geometryType() != Qgis.GeometryType.Polygon:
            raise ValueError("矢量范围必须是面图层")
        if layer.featureCount() < 1:
            raise ValueError(f"矢量图层「{layer.name()}」没有面要素")
        if not layer.crs().isValid():
            raise ValueError(f"矢量图层「{layer.name()}」没有有效 CRS")
        return layer

    def _get_extent_in_raster_crs(self, raster):
        canvas = self.iface.mapCanvas()
        if self.radio_view.isChecked():
            if self._view_extent is None:
                self._view_extent = QgsRectangle(canvas.extent())
                self._view_extent_crs = canvas.mapSettings().destinationCrs()
            raw_extent = QgsRectangle(self._view_extent)
            raw_crs = self._view_extent_crs or canvas.mapSettings().destinationCrs()
            mode = "当前视图"
        elif self.radio_rect.isChecked():
            if self._hand_drawn_extent is None:
                raise ValueError("请先选择「手绘矩形」，并在地图上拖拽绘制范围")
            raw_extent = QgsRectangle(self._hand_drawn_extent)
            raw_crs = self._hand_drawn_extent_crs or canvas.mapSettings().destinationCrs()
            mode = "手绘矩形"
        else:
            layer = self._get_valid_vector_range_layer()
            raw_extent = QgsRectangle(layer.extent())
            raw_crs = layer.crs()
            mode = "加载矢量范围"

        if not self._is_extent_valid(raw_extent):
            raise ValueError(f"{mode}范围无效，请重新选择范围")

        extent = self._transform_extent(raw_extent, raw_crs, raster.crs())
        extent = self._intersect_extent(extent, raster.extent())
        if extent is None or not self._is_extent_valid(extent):
            raise ValueError(
                f"{mode}范围与影像层「{raster.name()}」没有重叠，"
                "请移动到影像覆盖区或重新绘制范围。"
            )

        self.extent_status_label.setText(
            self._extent_status_text(mode, extent, raster)
        )
        self._refresh_processing_extent_preview()
        return extent

    def _update_extent_status(self, raw_extent, raw_crs, mode):
        if not self._is_extent_valid(raw_extent):
            self.extent_status_label.setText(f"{mode}范围无效，请重新选择")
            self._refresh_processing_extent_preview()
            return False

        try:
            raster = self._get_valid_raster_layer()
            extent = self._transform_extent(raw_extent, raw_crs, raster.crs())
            extent = self._intersect_extent(extent, raster.extent())
            if extent is None or not self._is_extent_valid(extent):
                self.extent_status_label.setText(
                    f"{mode}范围未识别: 与影像层「{raster.name()}」没有重叠"
                )
                self._refresh_processing_extent_preview()
                return False
            self.extent_status_label.setText(
                self._extent_status_text(mode, extent, raster)
            )
            self._refresh_processing_extent_preview()
            return True
        except ValueError as exc:
            crs_name = raw_crs.authid() if raw_crs and raw_crs.isValid() else "map CRS"
            self.extent_status_label.setText(
                f"{mode}范围已获取: {self._format_extent(raw_extent)} [{crs_name}]；"
                f"但尚未完成影像校验: {exc}"
            )
            self._refresh_processing_extent_preview()
            return False

    def _extent_status_text(self, mode, extent, raster):
        if mode == "加载矢量范围":
            layer = self.vector_range_combo.currentLayer()
            name = layer.name() if layer is not None else "未选择"
            return (
                f"矢量范围: {name}；外包范围 {self._format_extent(extent)} "
                f"[{raster.crs().authid()}]；相交 Tile 用于处理，结果按矢量边界精确裁剪"
            )
        return (
            f"{mode}范围: {self._format_extent(extent)} "
            f"[{raster.crs().authid()}]"
        )

    def _selected_raw_extent(self):
        if self.radio_view.isChecked():
            return "当前视图", self._view_extent, self._view_extent_crs
        if self.radio_rect.isChecked():
            return "手绘矩形", self._hand_drawn_extent, self._hand_drawn_extent_crs
        try:
            layer = self._get_valid_vector_range_layer()
        except ValueError:
            return "加载矢量范围", None, None
        return "加载矢量范围", QgsRectangle(layer.extent()), layer.crs()

    def _select_range_tiles(self, grid_tiles, raster):
        if not self.radio_vector.isChecked():
            return list(grid_tiles)
        layer = self._get_valid_vector_range_layer()
        selected = tile_manager.select_tiles_intersecting_vector(
            grid_tiles, layer, raster.crs()
        )
        if not selected:
            raise ValueError(
                f"矢量图层「{layer.name()}」没有选中任何完整 Tile"
            )
        return selected

    def _range_selection_metadata(self, grid_tiles, selected_tiles):
        if not self.radio_vector.isChecked():
            return {
                "mode": "extent",
                "selected_tile_count": len(selected_tiles),
                "excluded_tile_count": 0,
                "clip_outputs": True,
            }
        layer = self._get_valid_vector_range_layer()
        return {
            "mode": "vector_tile_intersection",
            "vector_layer_id": layer.id(),
            "vector_layer_name": layer.name(),
            "vector_source": layer.source(),
            "vector_crs": layer.crs().authid(),
            "selected_tile_count": len(selected_tiles),
            "excluded_tile_count": len(grid_tiles) - len(selected_tiles),
            "clip_outputs": True,
        }

    def _watch_vector_range_layer(self, layer):
        if layer is self._observed_vector_range_layer:
            return
        previous = self._observed_vector_range_layer
        if previous is not None:
            try:
                previous.dataChanged.disconnect(self._on_vector_range_data_changed)
            except (TypeError, RuntimeError):
                pass
        self._observed_vector_range_layer = layer
        if layer is not None:
            layer.dataChanged.connect(self._on_vector_range_data_changed)

    def _on_vector_range_data_changed(self, *_args):
        self._invalidate_vector_preview()
        self._refresh_processing_extent_preview()

    def _invalidate_vector_preview(self):
        self._vector_preview_timer.stop()
        self._vector_preview_cache = None
        self._vector_preview_requested_key = None
        self._start_after_vector_preview = False
        task = self._vector_preview_task
        self._vector_preview_task = None
        if task is not None:
            task.cancel()

    def _vector_preview_key(self, raster, extent):
        layer = self._get_valid_vector_range_layer()
        layer_extent = layer.extent()
        raster_extent = raster.extent()
        return (
            raster.id(),
            raster.source(),
            int(raster.width()),
            int(raster.height()),
            tuple(round(value, 9) for value in (
                raster_extent.xMinimum(), raster_extent.yMinimum(),
                raster_extent.xMaximum(), raster_extent.yMaximum(),
            )),
            layer.id(),
            layer.source(),
            int(layer.featureCount()),
            tuple(round(value, 9) for value in (
                layer_extent.xMinimum(), layer_extent.yMinimum(),
                layer_extent.xMaximum(), layer_extent.yMaximum(),
            )),
            tuple(round(value, 9) for value in (
                extent.xMinimum(), extent.yMinimum(),
                extent.xMaximum(), extent.yMaximum(),
            )),
            int(self.tile_width_spin.value()),
            int(self.tile_height_spin.value()),
            int(self.overlap_spin.value()),
        )

    def _cached_vector_preview(self, raster, extent):
        cache = self._vector_preview_cache
        if cache is None:
            return None
        try:
            key = self._vector_preview_key(raster, extent)
        except ValueError:
            return None
        return cache if cache.get("key") == key else None

    def _queue_vector_tile_preview(self, raster, extent, *, immediate=False):
        key = self._vector_preview_key(raster, extent)
        cache = self._vector_preview_cache
        if cache is not None and cache.get("key") == key:
            self._set_processing_extent_summary(cache, raster)
            return
        task = self._vector_preview_task
        if task is not None and task.request_key == key:
            return
        if task is not None:
            self._vector_preview_task = None
            task.cancel()
        self._vector_preview_requested_key = key
        self.processing_extent_status_label.setText(
            "正在后台计算矢量范围 Tile..."
        )
        self.processing_extent_status_label.setStyleSheet("color: #805500;")
        if immediate:
            self._vector_preview_timer.stop()
            self._start_vector_tile_preview()
        else:
            self._vector_preview_timer.start()

    def _start_vector_tile_preview(self):
        if self._cleaning_up or not self.radio_vector.isChecked():
            return
        try:
            raster = self._get_valid_raster_layer()
            layer = self._get_valid_vector_range_layer()
            raw_extent = QgsRectangle(layer.extent())
            extent = self._transform_extent(raw_extent, layer.crs(), raster.crs())
            extent = self._intersect_extent(extent, raster.extent())
            if extent is None or not self._is_extent_valid(extent):
                raise ValueError("矢量范围与当前影像没有重叠")
            key = self._vector_preview_key(raster, extent)
            if key != self._vector_preview_requested_key:
                self._vector_preview_requested_key = key
            cache = self._vector_preview_cache
            if cache is not None and cache.get("key") == key:
                self._set_processing_extent_summary(cache, raster)
                return
            geometries = tile_manager.snapshot_vector_geometries(
                layer, raster.crs()
            )
            task = tile_manager.VectorTileSelectionTask(
                key,
                extent,
                self.tile_width_spin.value(),
                self.tile_height_spin.value(),
                self.overlap_spin.value(),
                tile_manager.raster_grid_info(raster),
                geometries,
            )
        except ValueError as exc:
            self.processing_extent_status_label.setText(f"无法计算: {exc}")
            self.processing_extent_status_label.setStyleSheet("color: #b42318;")
            self._start_after_vector_preview = False
            return

        self._vector_preview_task = task
        task.progressChanged.connect(self._on_vector_preview_progress)
        task.taskCompleted.connect(self._on_vector_preview_completed)
        task.taskTerminated.connect(self._on_vector_preview_terminated)
        QgsApplication.taskManager().addTask(task)

    def _on_vector_preview_progress(self, progress):
        task = self.sender()
        if task is not self._vector_preview_task:
            return
        self.processing_extent_status_label.setText(
            f"正在后台计算矢量范围 Tile... {int(progress)}%"
        )

    def _on_vector_preview_completed(self):
        task = self.sender()
        if task is not self._vector_preview_task:
            return
        self._vector_preview_task = None
        result = task.result_data
        if result is None or result.get("key") != self._vector_preview_requested_key:
            return
        self._vector_preview_cache = result
        try:
            raster = self._get_valid_raster_layer()
            self._set_processing_extent_summary(result, raster)
        except ValueError:
            return
        if self._start_after_vector_preview:
            self._start_after_vector_preview = False
            QTimer.singleShot(0, self._on_start)

    def _on_vector_preview_terminated(self):
        task = self.sender()
        if task is not self._vector_preview_task:
            return
        self._vector_preview_task = None
        if task.isCanceled():
            return
        self._start_after_vector_preview = False
        message = task.error_message or "矢量范围 Tile 计算失败"
        self.processing_extent_status_label.setText(f"无法计算: {message}")
        self.processing_extent_status_label.setStyleSheet("color: #b42318;")

    def _set_processing_extent_status(self, tiles, raster, *, grid_tiles=None):
        full_grid = list(grid_tiles or tiles)
        processing_extent = tile_manager.get_grid_extent(full_grid)
        if processing_extent is None:
            self.processing_extent_status_label.setText("未生成完整 Tile")
            self.processing_extent_status_label.setStyleSheet("color: #b42318;")
            return
        summary = {
            "processing_extent": processing_extent,
            "rows": max(int(tile["row"]) for tile in full_grid) + 1,
            "cols": max(int(tile["col"]) for tile in full_grid) + 1,
            "grid_count": len(full_grid),
            "selected_count": len(tiles),
        }
        self._set_processing_extent_summary(summary, raster)

    def _set_processing_extent_summary(self, summary, raster):
        processing_extent = summary.get("processing_extent")
        if processing_extent is None:
            self.processing_extent_status_label.setText("未生成完整 Tile")
            self.processing_extent_status_label.setStyleSheet("color: #b42318;")
            return
        step_width = self.tile_width_spin.value() - self.overlap_spin.value()
        step_height = self.tile_height_spin.value() - self.overlap_spin.value()
        rows = int(summary["rows"])
        cols = int(summary["cols"])
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
        selected_count = int(summary["selected_count"])
        grid_count = int(summary["grid_count"])
        tile_summary = f"共 {selected_count} 个完整 Tile"
        excluded_count = grid_count - selected_count
        if excluded_count:
            tile_summary += f"；范围外排除 {excluded_count} 个"
        self.processing_extent_status_label.setText(
            f"{self._format_extent(processing_extent)} "
            f"[{raster.crs().authid()}]\n"
            f"{tile_summary}；步长 "
            f"{step_width} × {step_height} px；{unit_text}"
        )
        self.processing_extent_status_label.setStyleSheet("color: #1f6f3d;")

    def _refresh_processing_extent_preview(self):
        mode, raw_extent, raw_crs = self._selected_raw_extent()
        if not self._is_extent_valid(raw_extent):
            self.processing_extent_status_label.setText(
                f"请先获取{mode}范围"
            )
            self.processing_extent_status_label.setStyleSheet("color: #666;")
            return
        try:
            raster = self._get_valid_raster_layer()
            extent = self._transform_extent(raw_extent, raw_crs, raster.crs())
            extent = self._intersect_extent(extent, raster.extent())
            if extent is None or not self._is_extent_valid(extent):
                raise ValueError("绘图范围与当前影像没有重叠")
            if self.radio_vector.isChecked():
                self._queue_vector_tile_preview(raster, extent)
                return
            grid_tiles = tile_manager.generate_grid(
                extent,
                self.tile_width_spin.value(),
                self.tile_height_spin.value(),
                self.overlap_spin.value(),
                raster_layer=raster,
            )
            tiles = self._select_range_tiles(grid_tiles, raster)
            self._set_processing_extent_status(
                tiles, raster, grid_tiles=grid_tiles
            )
        except ValueError as exc:
            self.processing_extent_status_label.setText(f"无法计算: {exc}")
            self.processing_extent_status_label.setStyleSheet("color: #b42318;")

    def _on_tile_parameters_changed(self, *_args):
        self._invalidate_vector_preview()
        maximum_overlap = max(
            0,
            min(self.tile_width_spin.value(), self.tile_height_spin.value()) - 1,
        )
        if self.overlap_spin.maximum() != maximum_overlap:
            self.overlap_spin.setMaximum(maximum_overlap)
        self._render_last_env_report()
        self._refresh_processing_extent_preview()

    def _on_raster_layer_changed(self, *_args):
        self._invalidate_vector_preview()
        mode, raw_extent, raw_crs = self._selected_raw_extent()
        if self._is_extent_valid(raw_extent):
            self._update_extent_status(raw_extent, raw_crs, mode)
        else:
            self._refresh_processing_extent_preview()

    def _extent_as_dict(self, extent):
        if extent is None:
            return None
        return {
            "xmin": extent.xMinimum(),
            "ymin": extent.yMinimum(),
            "xmax": extent.xMaximum(),
            "ymax": extent.yMaximum(),
        }

    def _transform_extent(self, extent, source_crs, target_crs):
        if not source_crs or not source_crs.isValid():
            raise ValueError("地图当前 CRS 无效，无法把范围转换到影像 CRS")
        if not target_crs or not target_crs.isValid():
            raise ValueError("影像 CRS 无效，无法确定切片范围")
        if source_crs == target_crs:
            return QgsRectangle(extent)
        try:
            transform = QgsCoordinateTransform(
                source_crs, target_crs, QgsProject.instance()
            )
            return transform.transformBoundingBox(extent)
        except Exception as exc:
            raise ValueError(
                f"范围 CRS 转换失败: {source_crs.authid()} → {target_crs.authid()} ({exc})"
            ) from exc

    def _intersect_extent(self, a, b):
        xmin = max(a.xMinimum(), b.xMinimum())
        xmax = min(a.xMaximum(), b.xMaximum())
        ymin = max(a.yMinimum(), b.yMinimum())
        ymax = min(a.yMaximum(), b.yMaximum())
        if xmax <= xmin or ymax <= ymin:
            return None
        return QgsRectangle(xmin, ymin, xmax, ymax)

    def _is_extent_valid(self, extent):
        if extent is None:
            return False
        return (
            extent.xMaximum() > extent.xMinimum()
            and extent.yMaximum() > extent.yMinimum()
        )

    def _format_extent(self, extent):
        return (
            f"{extent.xMinimum():.6f}, {extent.yMinimum():.6f}, "
            f"{extent.xMaximum():.6f}, {extent.yMaximum():.6f}"
        )

    def _find_tile_row(self, row, col):
        for i in range(self.tile_table.rowCount()):
            item = self.tile_table.item(i, 0)
            if item and item.text() == f"({row},{col})":
                return i
        return -1

    def cleanup(self):
        self._cleaning_up = True
        self._save_settings()
        self.config_manager.cleanup()
        self._vector_preview_timer.stop()
        if self._vector_preview_task is not None:
            self._vector_preview_task.cancel()
            self._vector_preview_task = None
        if self._observed_vector_range_layer is not None:
            try:
                self._observed_vector_range_layer.dataChanged.disconnect(
                    self._on_vector_range_data_changed
                )
            except (TypeError, RuntimeError):
                pass
            self._observed_vector_range_layer = None

        # 1. 先切断 runner 与 monitor_dialog 的所有信号连接
        #    避免 Qt 销毁 receiver 后 use-after-free
        if self.runner is not None:
            try:
                self.runner.pipeline_finished.disconnect(self._on_pipeline_finished)
            except (TypeError, RuntimeError):
                pass
            try:
                self.runner.stage_progress.disconnect(self._on_runner_stage_progress)
            except (TypeError, RuntimeError):
                pass
            try:
                self.runner.step_started.disconnect(self._on_tile_step_started)
            except (TypeError, RuntimeError):
                pass
            try:
                self.runner.step_finished.disconnect(self._on_tile_step_finished)
            except (TypeError, RuntimeError):
                pass
            # 也断开 runner 到 monitor_dialog 的连接（由 monitor_dialog.detach() 完成）
            if self.monitor_dialog is not None:
                try:
                    self.monitor_dialog.detach()
                except (TypeError, RuntimeError):
                    pass

        # 2. 停止异步任务（runner / tile_extractor）
        if self.runner is not None:
            self.runner.stop()
            self.runner = None
        if self._tile_extractor is not None:
            self._tile_extractor.stop()
            self._tile_extractor = None
        if self._tile_cache_probe is not None:
            self._tile_cache_probe.cleanup()
            self._tile_cache_probe = None
        self._discard_pending_run_reservation()
        if self.refinement_dialog is not None:
            self.refinement_dialog.cleanup()

        # 3. 恢复地图工具、清理 UI 状态
        self._restore_previous_map_tool()
        if self._rect_tool:
            self._rect_tool.deactivate()
            self._rect_tool = None
        self._view_extent = None
        self._view_extent_crs = None
        self._hand_drawn_extent = None
        self._hand_drawn_extent_crs = None
        self._previous_map_tool = None

        # 4. monitor_dialog 随 parent (dock_widget) 自动销毁，**不要**显式 close()
        #    只需置空引用，避免后续误用
        self.monitor_dialog = None

        # 5. 其余字段置空
        self._current_tiles = []
        self._pending_run = None
