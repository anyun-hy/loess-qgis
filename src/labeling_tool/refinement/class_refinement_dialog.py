"""Modeless 14-class Fusion workspace with click-driven SAM3 refinement."""

from __future__ import annotations

import math
from pathlib import Path

from qgis.analysis import QgsZonalStatistics
from qgis.core import (
    Qgis,
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsCoordinateTransformContext,
    QgsFeature,
    QgsFeatureRequest,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
    QgsSettings,
)
from qgis.gui import QgsRubberBand
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
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from labeling_tool.qgis_support.dialog_geometry import fit_dialog_to_screen
from labeling_tool.qgis_support.qt6_api import (
    DASH_LINE,
    NO,
    WINDOW,
    YES,
)
from labeling_tool.qgis_support.qt_lifecycle import retire_after
from labeling_tool.qgis_support.style_manager import StyleManager
from labeling_tool.refinement import (
    class_review,
    class_workspace,
    edit_tracking,
    manual_edit_commit,
    manual_edit_operations,
    manual_edit_tools,
    topology_validator,
)
from labeling_tool.refinement.admission_presentation import AdmissionSummarySnapshot
from labeling_tool.refinement.admission_summary_panel import AdmissionSummaryPanel
from labeling_tool.refinement.background_io_tasks import AcceptedWriteTask
from labeling_tool.refinement.class_review_panel import (
    ClassReviewPanel,
    ClassReviewRow,
    ClassReviewSnapshot,
)
from labeling_tool.refinement.geometry_smoothing import (
    GeometrySmoothingError,
    NativeSmoothingPreview,
    SmoothingParameters,
    geometry_source_hash,
    smooth_geometry_batch,
    validate_polygon_geometry,
)
from labeling_tool.refinement.manual_edit_panel import (
    ManualEditPanel,
    ManualPanelSnapshot,
)
from labeling_tool.refinement.manual_edit_state import ManualEditTask
from labeling_tool.refinement.refinement_task import RefinementTask, file_identity
from labeling_tool.refinement.sam3_worker_runner import Sam3WorkerRunner
from labeling_tool.refinement.sam_map_preview import SamMapPreview
from labeling_tool.refinement.sam_session import SamDecision, SamSession
from labeling_tool.refinement.sam_session_panel import (
    SamPanelSnapshot,
    SamSessionPanel,
)
from labeling_tool.refinement.workspace_layer_loader import WorkspaceLayerLoader
from labeling_tool.refinement.workspace_layer_signals import WorkspaceLayerSignals
from labeling_tool.refinement.workspace_tasks import WorkspaceTasks
from labeling_tool.shared.contracts.run_spec import CLASS_NAMES, CLASS_ORDER


class ClassRefinementDialog(QDialog):
    workspace_changed = pyqtSignal(object)

    def __init__(self, iface, layer_manager, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.layer_manager = layer_manager
        self.setWindowTitle("分类修整与组装")
        self.setWindowFlags(WINDOW)
        fit_dialog_to_screen(
            self,
            preferred_size=(1220, 760),
            minimum_size=(640, 360),
        )
        self._result = {}
        self._run_spec = {}
        self._sam_config = {}
        self._scripts_dir = ""
        self._eligible_fusions = []
        self._workspace = None
        self._refinement_task = None
        self._accepted_task = None
        self._accepted_generation = 0
        self._workspace_statistics = {}
        self._workspace_refresh = {}
        self._workspace_tasks = WorkspaceTasks(self)
        self._workspace_tasks.progress.connect(self._workspace_task_progress)
        self._workspace_tasks.probed.connect(self._workspace_probe_completed)
        self._workspace_tasks.initialized.connect(self._workspace_initialize_completed)
        self._workspace_tasks.terminated.connect(self._workspace_task_terminated)
        self._layer_loader = WorkspaceLayerLoader(self._load_workspace_layer, self)
        self._layer_loader.loaded.connect(self._workspace_layer_loaded)
        self._layer_loader.failed.connect(self._workspace_layer_failed)
        self._layer_loader.completed.connect(self._workspace_layer_loading_completed)
        self._manual_only = False
        self._class_layers = {}
        self._qgis_smooth_preview: NativeSmoothingPreview | None = None
        self._qgis_smooth_preview_bands = []
        self._smoothing_parameter_sync = False
        self._manual_smoothing_timer = QTimer(self)
        self._manual_smoothing_timer.setSingleShot(True)
        self._manual_smoothing_timer.setInterval(250)
        self._manual_smoothing_timer.timeout.connect(
            self._refresh_manual_smoothing_preview
        )
        self._syncing_class_selection = False
        self._pending_visible_codes: set[int] = set()
        self._edit_tracker = edit_tracking.EditTracker()
        self._layer_signals = WorkspaceLayerSignals(self)
        self._layer_signals.editing_started.connect(self._editing_started)
        self._layer_signals.editing_stopped.connect(self._editing_stopped)
        self._layer_signals.before_commit.connect(
            self._edit_tracker.capture_before_commit
        )
        self._layer_signals.features_committed.connect(
            self._edit_tracker.record_committed_additions
        )
        self._layer_signals.selection_changed.connect(self._selection_changed)
        self._layer_signals.edit_changed.connect(self._layer_edit_changed)
        self._layer_signals.visibility_changed.connect(
            self._sync_visibility_from_layer_tree
        )
        self._layer_signals.current_layer_changed.connect(
            self._active_layer_changed
        )
        self._worker = None
        self._active_session: SamSession | None = None
        self._manual_task: ManualEditTask | None = None
        self._manual_reference_band = None
        self._manual_add_candidate_bands = []
        self._manual_tools = manual_edit_tools.ManualEditTools(
            self.iface.mapCanvas(), lambda: self.iface.cadDockWidget(), self
        )
        self._manual_tools.map_clicked.connect(self._manual_task_map_clicked)
        self._manual_tools.feature_captured.connect(self._manual_capture_completed)
        self._manual_tools.capture_cancelled.connect(self._manual_capture_cancelled)
        self._manual_tools.interrupted.connect(self._manual_tool_interrupted)
        self._manual_tools.restart_requested.connect(self._start_manual_capture)
        self._sam_preview = SamMapPreview(self.iface.mapCanvas(), self)
        self._sam_preview.point_clicked.connect(self._sam_map_clicked)
        self._confidence_raster = None
        self._final_path = ""
        self._final_input_identities = {}
        self._final_feature_count = None
        self._accepted_feature_count = None
        self._accepted_warnings = ()
        self._issues_path = ""
        self._issue_count = None
        self._build_ui()
        self._layer_signals.connect_current_layer(
            getattr(self.iface, "currentLayerChanged", None)
        )
        self._manual_tools.connect()

    def _build_ui(self):
        root = QVBoxLayout(self)
        content = QWidget(self)
        root.addWidget(self._scrollable_content(content), stretch=1)
        root = QVBoxLayout(content)
        baseline = QHBoxLayout()
        fusion_title = QLabel("Fusion 基准:")
        fusion_title.setMinimumWidth(fusion_title.sizeHint().width())
        baseline.addWidget(fusion_title)
        self.fusion_combo = QComboBox()
        baseline.addWidget(self.fusion_combo, stretch=1)
        self.initialize_btn = QPushButton("初始化 14 类工作层")
        self.initialize_btn.clicked.connect(self._initialize_workspace)
        baseline.addWidget(self.initialize_btn)
        self.cancel_load_btn = QPushButton("取消后台加载")
        self.cancel_load_btn.clicked.connect(self._cancel_background_load)
        self.cancel_load_btn.hide()
        baseline.addWidget(self.cancel_load_btn)
        self.baseline_label = QLabel("尚未加载运行结果")
        self.baseline_label.setWordWrap(True)
        baseline.addWidget(self.baseline_label, stretch=2)
        root.addLayout(baseline)

        self.class_review_panel = ClassReviewPanel(self)
        self.class_review_panel.class_selected.connect(self._select_class_context)
        self.class_review_panel.visibility_requested.connect(self._set_visible)
        self.class_review_panel.manual_requested.connect(self._open_manual_operations)
        self.class_review_panel.sam_requested.connect(self._request_sam)
        self.class_review_panel.confirm_requested.connect(self._confirm_class)
        root.addWidget(self.class_review_panel, stretch=1)

        smooth_settings = QgsSettings()

        def smooth_setting(key, default, caster):
            try:
                return caster(smooth_settings.value(key, default))
            except (TypeError, ValueError):
                return default

        manual_parameters = (
            smooth_setting("labeling_tool/qgis_smoothing/iterations", 1, int),
            smooth_setting("labeling_tool/qgis_smoothing/offset", 0.25, float),
            smooth_setting("labeling_tool/qgis_smoothing/max_angle", 180.0, float),
        )
        self._manual_panel = ManualEditPanel(manual_parameters, self)
        self._manual_panel.modify_requested.connect(self._begin_modify_task)
        self._manual_panel.delete_requested.connect(self._begin_delete_task)
        self._manual_panel.add_requested.connect(self._begin_add_task)
        self._manual_panel.target_changed.connect(self._target_class_changed)
        self._manual_panel.smoothing_enabled_changed.connect(
            self._manual_smoothing_changed
        )
        self._manual_panel.smoothing_parameters_changed.connect(
            self._manual_smoothing_parameters_changed
        )
        self._manual_panel.primary_requested.connect(self._manual_primary_action)
        self._manual_panel.retry_requested.connect(self._manual_retry_action)
        self._manual_panel.clear_requested.connect(self._manual_clear_action)
        self._manual_panel.continue_requested.connect(self._continue_manual_task)
        self._manual_panel.cancel_requested.connect(self._manual_cancel_action)
        self._manual_panel.finish_requested.connect(self._finish_manual_session)
        root.addWidget(self._manual_panel)

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

        self._sam_panel = SamSessionPanel(self)
        self._sam_panel.decision_requested.connect(self._finish_session)
        self._sam_panel.retry_requested.connect(self._retry_session)
        root.addWidget(self._sam_panel)

        actions = QHBoxLayout()
        self.assemble_btn = QPushButton("组装最终图层")
        self.topology_btn = QPushButton("重新检查拓扑")
        self.allow_issues_check = QCheckBox("明确允许带未解决问题入库")
        self.summary_label = QLabel("14 类确认: 0/14    未解决问题: -    未保存编辑: 0")
        actions.addWidget(self.assemble_btn)
        actions.addWidget(self.topology_btn)
        actions.addWidget(self.allow_issues_check)
        actions.addWidget(self.summary_label, stretch=1)
        root.addLayout(actions)
        self.admission_summary_panel = AdmissionSummaryPanel(self)
        self.admission_summary_panel.write_requested.connect(self._write_accepted)
        root.addWidget(self.admission_summary_panel)
        self.assemble_btn.clicked.connect(self._assemble_final)
        self.topology_btn.clicked.connect(self._check_topology)
        self.allow_issues_check.toggled.connect(self._update_accept_enabled)
        self.topology_btn.setEnabled(False)
        self._update_manual_panel()
        self._update_actions()

    @staticmethod
    def _scrollable_content(content):
        scroll = QScrollArea()
        scroll.setObjectName("classRefinementScrollArea")
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        return scroll

    def set_run(self, result, run_spec, sam_config, scripts_dir):
        self._retire_accepted_task()
        self._cancel_background_load(silent=True)
        self._layer_loader.reset()
        self._pending_visible_codes.clear()
        self._final_path = ""
        self._final_input_identities = {}
        self._final_feature_count = None
        self._accepted_feature_count = None
        self._accepted_warnings = ()
        self._issues_path = ""
        self._issue_count = None
        self.topology_btn.setEnabled(False)
        self._layer_signals.connect_current_layer(
            getattr(self.iface, "currentLayerChanged", None)
        )
        self._manual_tools.connect()
        self._clear_qgis_smooth_preview()
        self._manual_smoothing_timer.stop()
        self._cancel_manual_task(silent=True)
        self._cancel_active_session(record=False)
        self._layer_signals.clear_layers()
        self._edit_tracker.reset()
        self._workspace = None
        self._workspace_statistics = {}
        self._workspace_refresh = {}
        self._class_layers = {}
        self._result = dict(result)
        self._run_spec = dict(run_spec)
        self._sam_config = dict(sam_config or {})
        self._scripts_dir = str(scripts_dir)
        self._manual_only = bool(self._run_spec.get("manual_only"))
        self.setWindowTitle(
            "人工分类整理" if self._manual_only else "分类修整与组装"
        )
        self._sam_panel.render(SamPanelSnapshot())
        self._confidence_raster = None
        streams = [
            item for item in result.get("ready_streams") or []
            if item.get("status") == "ready"
        ]
        self._eligible_fusions = []
        self.fusion_combo.clear()
        self.fusion_combo.setEnabled(False)
        self.initialize_btn.setEnabled(False)
        self.baseline_label.setText(
            f"Run {run_spec.get('run_id')}；正在后台校验工作区文件..."
        )
        self.cancel_load_btn.show()
        self._refresh_table()
        self._workspace_tasks.probe(self._run_spec, streams)

    def _workspace_task_progress(self, operation: str, progress: float) -> None:
        action = "初始化 14 类工作层" if operation == "initialize" else "校验工作区文件"
        self.baseline_label.setText(
            f"Run {self._run_spec.get('run_id')}；正在后台{action}... {int(progress)}%"
        )

    def _apply_eligible_fusions(self, streams):
        self._eligible_fusions = list(streams or [])
        self.fusion_combo.clear()
        for stream in self._eligible_fusions:
            self.fusion_combo.addItem(stream["stream_id"], stream["stream_id"])

    def _workspace_probe_completed(self, data: dict) -> None:
        self._apply_eligible_fusions(data.get("eligible_fusions"))
        self._workspace = data.get("workspace")
        self._workspace_statistics = data.get("statistics") or {}
        self._workspace_refresh = data.get("review_refresh") or {}
        if self._workspace is not None:
            self.fusion_combo.setCurrentText(
                self._workspace["baseline_stream_id"]
            )
            self.fusion_combo.setEnabled(False)
            if self._workspace_refresh.get("required"):
                safe = bool(self._workspace_refresh.get("safe_to_replace"))
                self.initialize_btn.setText("重建为 V3 14 类工作层")
                self.initialize_btn.setEnabled(safe)
                if safe:
                    self.cancel_load_btn.hide()
                    self.baseline_label.setText(
                        "V3 派生结果已就绪；现有工作区无人工修改，可在后台原子重建"
                    )
                    self._refresh_table()
                    return
                self.baseline_label.setText(
                    "V3 派生结果已就绪，但现有工作区含人工修改/确认记录，"
                    "为防止数据丢失已禁止自动覆盖"
                )
            else:
                self.initialize_btn.setText("初始化 14 类工作层")
                self.initialize_btn.setEnabled(False)
            self._load_workspace_layers()
            return
        self.cancel_load_btn.hide()
        self.fusion_combo.setEnabled(bool(self._eligible_fusions))
        self.initialize_btn.setEnabled(bool(self._eligible_fusions))
        self.baseline_label.setText(
            f"Run {self._run_spec.get('run_id')}；"
            f"可用 Fusion {len(self._eligible_fusions)} 个"
        )
        self._refresh_table()
        if self._manual_only and len(self._eligible_fusions) == 1:
            self.fusion_combo.setCurrentIndex(0)
            self._initialize_workspace()

    def _workspace_task_terminated(
        self,
        operation: str,
        error_message: str,
    ) -> None:
        self.cancel_load_btn.hide()
        if error_message:
            title = (
                "初始化 14 类工作层失败"
                if operation == "initialize"
                else "恢复类别工作区失败"
            )
            self.baseline_label.setText(f"后台任务失败：{error_message}")
            QMessageBox.warning(self, title, error_message)
        else:
            self.baseline_label.setText("后台加载已取消；QGIS 主界面仍可继续操作")
        self.fusion_combo.setEnabled(bool(self._eligible_fusions))
        self.initialize_btn.setEnabled(bool(self._eligible_fusions))
        self._refresh_table()

    def _initialize_workspace(self):
        if self._workspace_tasks.busy:
            return
        run_spec = self._run_spec
        stream_id = str(self.fusion_combo.currentData() or "")
        if not stream_id:
            QMessageBox.warning(self, "初始化工作区", "没有通过规则化和审批的 Fusion")
            return
        try:
            stream = class_workspace.stream_by_id(
                self._eligible_fusions, stream_id
            )
        except Exception as exc:
            QMessageBox.warning(self, "初始化 14 类工作层失败", str(exc))
            return
        replace = bool(
            self._workspace_refresh.get("required")
            and self._workspace_refresh.get("safe_to_replace")
        )
        if replace:
            answer = QMessageBox.question(
                self,
                "重建 V3 工作区",
                "已确认当前 14 类工作区没有人工修改、类别确认或编辑历史。\n\n"
                "继续后将用 V3 派生结果原子重建 14 个类别文件；"
                "原始 Fusion 成果和 V3 总成果均保留。是否继续？",
                YES | NO,
                NO,
            )
            if answer != YES:
                return
            if run_spec is not self._run_spec:
                return
        self.fusion_combo.setEnabled(False)
        self.initialize_btn.setEnabled(False)
        self.cancel_load_btn.show()
        self.baseline_label.setText("正在后台初始化 14 类工作层... 0%")
        self._workspace_tasks.initialize(
            run_spec,
            stream,
            replace=replace,
        )

    def _workspace_initialize_completed(self, data: dict) -> None:
        self._workspace = data.get("workspace")
        self._workspace_statistics = data.get("statistics") or {}
        self._workspace_refresh = {}
        self.initialize_btn.setText("初始化 14 类工作层")
        if self._workspace is None:
            self.baseline_label.setText("后台初始化没有返回工作区")
            return
        self.fusion_combo.setEnabled(False)
        self.initialize_btn.setEnabled(False)
        self._load_workspace_layers()

    def _load_workspace_layers(self):
        pending_codes = [
            code for code in CLASS_ORDER if code not in self._class_layers
        ]
        self._layer_loader.start(pending_codes)
        self.cancel_load_btn.show()
        self.baseline_label.setText(
            self._workspace_summary_text(
                f"后台分批挂载 {len(pending_codes)} 个类别图层"
            )
        )
        self._refresh_table()

    def _workspace_summary_text(self, suffix=""):
        if not self._workspace:
            return "类别工作区尚未加载"
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
        if self._workspace.get("portable_classes_only"):
            text += " | Fusion 基准离线"
        if suffix:
            text += f" | {suffix}"
        return text

    def _prioritize_class_load(self, class_code, *, activate=False):
        class_code = int(class_code)
        if activate:
            self._pending_visible_codes.add(class_code)
        if class_code in self._class_layers:
            if activate:
                self._select_class_context(class_code, activate_layer=True)
            return
        self.cancel_load_btn.show()
        self._layer_loader.prioritize(class_code, activate=activate)

    def _load_workspace_layer(self, code: int) -> None:
        if not self._workspace:
            raise RuntimeError("class workspace is not available")
        record = self._workspace["classes"][str(code)]
        layer_id = self.layer_manager.load_workspace_class(
            self._run_spec["run_id"], record, visible=False
        )
        self._class_layers[code] = layer_id
        self._register_workspace_layer(code, layer_id)
        self.layer_manager.set_layer_visibility(
            layer_id, code in self._pending_visible_codes
        )
        self._pending_visible_codes.discard(code)

    def _workspace_layer_loaded(self, code: int, activate: bool) -> None:
        if activate:
            self._select_class_context(code, activate_layer=True)
        self._refresh_table()
        self.baseline_label.setText(
            self._workspace_summary_text(
                f"后台分批挂载，剩余 {len(self._layer_loader.pending_codes)} 类"
            )
        )

    def _workspace_layer_failed(self, code: int, message: str) -> None:
        self.cancel_load_btn.hide()
        self.baseline_label.setText(f"类别 {code} 挂载失败：{message}")
        QMessageBox.warning(self, f"类别 {code} 挂载失败", message)
        self._refresh_table()

    def _workspace_layer_loading_completed(self) -> None:
        self.cancel_load_btn.hide()
        self.baseline_label.setText(self._workspace_summary_text("工作层已就绪"))
        self._refresh_table()

    def _register_workspace_layer(self, code, layer_id):
        layer = QgsProject.instance().mapLayer(layer_id)
        if layer is None:
            raise RuntimeError(f"class {code} working layer was not added")
        class_workspace.apply_class_constraints(
            layer,
            code,
            run_id=self._run_spec["run_id"],
            baseline_stream_id=self._workspace["baseline_stream_id"],
        )
        if layer.isEditable() and not self._edit_tracker.has_session(code):
            record = self._workspace["classes"][str(code)]
            persisted = class_workspace.working_layer(
                record, f"class_{code}_persisted_snapshot"
            )
            self._edit_tracker.restore(code, layer, persisted)
        tree_layer = QgsProject.instance().layerTreeRoot().findLayer(layer.id())
        self._layer_signals.bind_layer(code, layer, tree_layer)
        self._sync_visibility_from_layer_tree(code)
        self._active_layer_changed(self.iface.activeLayer())
        self._update_manual_panel()

    def _cancel_background_load(self, _checked=False, *, silent=False):
        if self._accepted_task is not None:
            task = self._accepted_task
            if task.commit_started:
                if not silent:
                    self.baseline_label.setText("正在提交 accepted_labels，不能取消")
                return
            task.cancel()
            if not silent:
                self.baseline_label.setText(
                    "正在取消 accepted_labels 校验；目标不会被修改"
                )
                return
        if self._refinement_task is not None:
            task = self._refinement_task
            task.cancel()
            if silent:
                # The manager owns the task until completion. An old Run must
                # neither publish into a new Run nor retain its staged output.
                task.taskCompleted.connect(task.discard)
                self._refinement_task = None
            if not silent:
                self.baseline_label.setText("正在取消组装/拓扑检查；现有结果不会被覆盖")
                return
        self._workspace_tasks.cancel()
        self._layer_loader.pause()
        if not silent:
            self.cancel_load_btn.hide()
            self.baseline_label.setText(
                "后台加载已暂停；点击需要的类别可按需继续挂载"
            )
            self.fusion_combo.setEnabled(bool(self._eligible_fusions))
            self.initialize_btn.setEnabled(
                self._workspace is None and bool(self._eligible_fusions)
            )

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
        class_code = int(class_code)
        layer_id = self._class_layers.get(class_code)
        if layer_id:
            self.layer_manager.set_layer_visibility(layer_id, visible)
        elif visible and self._workspace:
            self._pending_visible_codes.add(class_code)
            self._prioritize_class_load(class_code)
        else:
            self._pending_visible_codes.discard(class_code)
        self._refresh_table()

    def _sync_visibility_from_layer_tree(self, class_code):
        del class_code
        self._refresh_table()

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
            if self._workspace:
                self._prioritize_class_load(
                    class_code, activate=activate_layer
                )
                self._refresh_table()
            return
        locked_code = self._manual_locked_class_code()
        if locked_code is not None and class_code != locked_code:
            self._manual_panel.set_instruction(
                f"当前人工任务锁定 {locked_code} {CLASS_NAMES[locked_code]}；"
                "请先完成或取消任务"
            )
            class_code = locked_code
        self._syncing_class_selection = True
        try:
            if activate_layer:
                layer = self._layer(class_code)
                self.layer_manager.set_layer_visibility(layer.id(), True)
                active_layer = self.iface.activeLayer()
                if active_layer is None or active_layer.id() != layer.id():
                    self.iface.setActiveLayer(layer)
        finally:
            self._syncing_class_selection = False
        if self._manual_task is None:
            self._manual_panel.set_target_code(class_code)
        self._refresh_table()
        self._update_manual_panel()

    def _active_layer_changed(self, layer):
        if self._syncing_class_selection:
            return
        preview = self._qgis_smooth_preview
        if preview and (layer is None or layer.id() != preview.layer_id):
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
        self._refresh_table()
        self._update_manual_panel()

    def _current_class_code(self):
        return self._class_code_for_layer(self.iface.activeLayer())

    def _manual_locked_class_code(self):
        task = self._manual_task
        if not task:
            return None
        return task.class_code

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
            if source != "manual":
                self._manual_panel.set_smoothing_parameters(parameters)
            if source != "qgis":
                group = (
                    self.qgis_smooth_iterations_spin,
                    self.qgis_smooth_offset_spin,
                    self.qgis_smooth_angle_spin,
                )
                iteration_spin, offset_spin, angle_spin = group
                iteration_spin.setValue(iterations)
                offset_spin.setValue(offset)
                angle_spin.setValue(max_angle)
        finally:
            self._smoothing_parameter_sync = False

    def _reset_manual_smoothing_for_new_task(self):
        self._manual_smoothing_timer.stop()
        self._manual_panel.set_smoothing_enabled(False)
        self._manual_panel.set_smoothing_status(
            "光滑默认关闭；开启后参数变化会自动预览本批全部新边界"
        )

    def _manual_smoothing_changed(self, checked):
        task = self._manual_task
        if not task or task.kind not in ("add", "modify"):
            return
        task.smoothing_enabled = bool(checked)
        task.invalidate_smoothing()
        self._manual_smoothing_timer.stop()
        if checked:
            self._schedule_manual_smoothing_preview()
        else:
            self._manual_panel.set_smoothing_status(
                "光滑已关闭；地图和保存均使用原始绘制边界"
            )
            self._refresh_manual_pending_candidate_bands()
            self._update_manual_panel()

    def _manual_smoothing_parameters_changed(self, *_args):
        if self._smoothing_parameter_sync:
            return
        parameters = self._manual_panel.smoothing_parameters()
        self._store_smoothing_parameters(parameters)
        self._sync_smoothing_parameter_widgets(parameters, "manual")
        task = self._manual_task
        if (
            task and task.kind in ("add", "modify")
            and task.smoothing_enabled
        ):
            self._schedule_manual_smoothing_preview()

    def _schedule_manual_smoothing_preview(self):
        self._manual_smoothing_timer.stop()
        task = self._manual_task
        if not task or task.kind not in ("add", "modify"):
            return
        task.invalidate_smoothing()
        self._refresh_manual_pending_candidate_bands()
        pending_count = len(task.pending_geometries)
        if not task.smoothing_enabled:
            self._manual_panel.set_smoothing_status(
                "光滑已关闭；地图和保存均使用原始绘制边界"
            )
        elif not pending_count:
            self._manual_panel.set_smoothing_status(
                "请先绘制新边界；只修改类别时不会光滑旧面"
            )
        elif any(task.pending_errors):
            self._manual_panel.set_smoothing_status(
                "本批存在不可保存的原始候选；请先重新绘制当前面"
            )
        else:
            self._manual_panel.set_smoothing_status(
                f"正在自动更新 {pending_count} 个待保存面的光滑预览…"
            )
            self._manual_smoothing_timer.start()
        self._update_manual_panel()

    def _manual_smoothing_preview_is_current(
        self, task: ManualEditTask | None = None
    ) -> bool:
        task = task or self._manual_task
        if (
            not task or task.kind not in ("add", "modify")
            or not task.smoothing_enabled
        ):
            return False
        preview = task.smoothing_preview
        geometries = task.pending_geometries
        if (
            not preview
            or preview.parameters != SmoothingParameters(
                *self._manual_panel.smoothing_parameters()
            )
            or len(preview.geometries) != len(geometries)
        ):
            return False
        source_hashes = tuple(
            geometry_source_hash(geometry) for geometry in geometries
        )
        return source_hashes == preview.source_hashes

    def _refresh_manual_smoothing_preview(self) -> None:
        task = self._manual_task
        if (
            not task or task.kind not in ("add", "modify")
            or not task.smoothing_enabled
        ):
            return
        if not task.pending_geometries:
            self._schedule_manual_smoothing_preview()
            return
        parameters = SmoothingParameters(
            *self._manual_panel.smoothing_parameters()
        )
        try:
            preview = smooth_geometry_batch(
                task.pending_geometries, parameters, convert_to_multi=True
            )
        except GeometrySmoothingError as exc:
            task.smoothing_preview = None
            task.smoothing_error = str(exc)
            self._manual_panel.set_smoothing_status(
                f"自动预览失败：{exc}；请调整参数或取消光滑后保存原始边界"
            )
            self._refresh_manual_pending_candidate_bands()
            self._update_manual_panel()
            return
        task.smoothing_error = ""
        task.smoothing_preview = preview
        statistics = preview.statistics
        self._manual_panel.set_smoothing_status(
            f"已自动预览 {len(preview.geometries)} 个面：顶点 "
            f"{statistics.source_vertex_count} → {statistics.smoothed_vertex_count}，"
            f"总面积变化 {statistics.area_change_percent:+.3f}%；"
            "保存将写入当前预览；逐面光滑不保证相邻面继续共边"
        )
        self._refresh_manual_pending_candidate_bands()
        self._update_manual_panel()

    def _open_manual_operations(self, class_code):
        self._select_class_context(class_code, activate_layer=True)
        self._manual_panel.focus_panel()
        self._manual_panel.set_instruction(
            f"已选择 {class_code} {CLASS_NAMES[class_code]}，请选择修改、删除或新增任务"
        )
        self._update_manual_panel()

    def _selection_changed(self, class_code):
        task = self._manual_task
        if task and task.kind == "delete" and task.class_code == int(class_code):
            task.selected_count = self._layer(class_code).selectedFeatureCount()
        preview = self._qgis_smooth_preview
        if preview and preview.class_code == int(class_code):
            selected_ids = tuple(sorted(self._layer(class_code).selectedFeatureIds()))
            if selected_ids != preview.feature_ids:
                self._clear_qgis_smooth_preview(
                    status_text="选择已变化，请重新预览光滑效果"
                )
        self._update_manual_panel()

    def _layer_edit_changed(self, class_code):
        if self._edit_tracker.suppressed:
            return
        preview = self._qgis_smooth_preview
        if preview and preview.class_code == int(class_code):
            self._clear_qgis_smooth_preview(
                status_text="来源 geometry 已变化，请重新预览光滑效果"
            )
        self._update_manual_panel()
        self._update_actions()

    def _target_class_changed(self, target_code):
        target_code = int(target_code)
        task = self._manual_task
        if task:
            if task.kind == "add":
                task.target_code = target_code
                task.error = ""
                self._refresh_manual_pending_candidate_bands()
                pending_count = len(task.pending_geometries)
                self._manual_panel.set_instruction(
                    f"本批待保存 {pending_count} 个面；目标类别为 "
                    f"{target_code} {CLASS_NAMES[target_code]}"
                )
                self._update_manual_panel()
                return
            if task.kind == "modify":
                task.target_code = target_code
                task.error = ""
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
        self._manual_task = ManualEditTask.for_modify(
            class_code, list(layer.selectedFeatureIds())
        )
        self._manual_tools.begin_session()
        self._manual_panel.set_target_code(class_code)
        layer.removeSelection()
        self._refresh_manual_modify_reference()
        self._manual_tools.start_picker()
        self._manual_panel.set_instruction(
            f"已选择 {len(self._manual_task.selected_feature_ids)} 个旧面；"
            "在地图中继续点击可加入或移出，选好后可直接改类或绘制新边界"
        )
        self._refresh_table()
        self._update_manual_panel()

    def _toggle_manual_modify_feature(self, feature):
        task = self._manual_task
        if not task or task.kind != "modify":
            return
        geometry = QgsGeometry(feature.geometry())
        error = validate_polygon_geometry(geometry)
        if error:
            QMessageBox.warning(self, "修改现有面", f"选中面不可修改：{error}")
            return
        feature_id = feature.id()
        selected_count = task.toggle_selected_feature(feature_id)
        layer = self._layer(task.class_code)
        layer.removeSelection()
        self._refresh_manual_modify_reference()
        self._manual_panel.set_instruction(
            f"已选择 {selected_count} 个旧面；地图点击可继续加入或移出，"
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
        self._manual_task = ManualEditTask.for_delete(
            class_code, list(layer.selectedFeatureIds())
        )
        self._manual_tools.begin_session()
        self._manual_tools.start_picker()
        self._manual_panel.set_instruction(
            "在地图中逐个点击面可加入或移出选择；不需要按 Shift"
        )
        self._refresh_table()
        self._update_manual_panel()

    def _begin_add_task(self):
        class_code = self._manual_task_guard("新增面")
        if class_code is None:
            return
        self._select_class_context(class_code, activate_layer=True)
        self._set_visible(class_code, True)
        self._reset_manual_smoothing_for_new_task()
        self._manual_task = ManualEditTask.for_add(
            class_code, list(self._layer(class_code).selectedFeatureIds())
        )
        self._manual_tools.begin_session()
        self._manual_panel.set_target_code(class_code)
        self._start_manual_capture()
        self._refresh_table()

    def _manual_task_map_clicked(self, map_point, _button):
        task = self._manual_task
        if not task or task.state != "selecting":
            return
        layer = self._layer(task.class_code)
        hits = self._features_at_map_point(layer, map_point)
        if len(hits) != 1:
            QMessageBox.information(
                self,
                "选择面",
                "点击位置必须唯一命中当前类别中的一个面，请放大地图后重新点击",
            )
            return
        feature = hits[0]
        if task.kind == "delete":
            ids = set(layer.selectedFeatureIds())
            if feature.id() in ids:
                ids.remove(feature.id())
            else:
                ids.add(feature.id())
            layer.selectByIds(sorted(ids))
            task.selected_count = len(ids)
            self._update_manual_panel()
            return
        self._toggle_manual_modify_feature(feature)

    def _manual_tool_interrupted(self):
        task = self._manual_task
        if not task:
            return
        if not task.pause():
            return
        self._manual_panel.set_instruction(
            "QGIS 地图工具已切换；点击“继续任务”恢复，或取消当前任务"
        )
        self._update_manual_panel()

    def _continue_manual_task(self):
        task = self._manual_task
        if not task or task.state != "paused":
            return
        resume = task.resume()
        if resume == "selecting":
            self._manual_tools.start_picker()
        elif resume == "capturing":
            self._start_manual_capture()
        self._update_manual_panel()

    def _start_manual_capture(self):
        task = self._manual_task
        if not task:
            return
        self._manual_tools.stop_capture()
        if task.kind == "modify":
            if not task.selected_feature_ids:
                QMessageBox.information(self, "修改现有面", "请先选择一个或多个旧面")
                return
            self._manual_tools.stop_picker()
        task.begin_capture()
        class_code = task.class_code
        layer = self._layer(class_code)
        self.iface.setActiveLayer(layer)
        if not layer.isEditable():
            with self._edit_tracker.suppress():
                editing_started = layer.startEditing()
            if not editing_started:
                task.set_failed("无法开启目标类别层编辑模式（Toggle Editing）")
                self._manual_tools.restore_previous()
                self._manual_panel.set_instruction(task.error)
                self._update_manual_panel()
                return
            task.editing_started_by_task = True
        try:
            self._manual_tools.start_capture(layer)
        except RuntimeError as exc:
            task.set_failed(str(exc))
            self._manual_tools.restore_previous()
            self._update_manual_panel()
            return
        pending_count = len(task.pending_geometries)
        if task.kind == "modify":
            selected_count = len(task.selected_feature_ids)
            self._manual_panel.set_instruction(
                f"已选 {selected_count} 个灰色旧面，本批已有 {pending_count} 个新边界；"
                "继续使用 Bézier 绘制完整面并右键结束"
            )
        else:
            self._manual_panel.set_instruction(
                f"本批待保存 {pending_count} 个面；继续绘制完整面并右键结束，"
                "或回到窗口选择类别后保存"
            )
        task.begin_capture()
        self._update_manual_panel()

    def _manual_capture_completed(self, feature):
        task = self._manual_task
        if not task or task.state != "capturing":
            return
        geometry = QgsGeometry(feature.geometry())
        if geometry.requiresConversionToStraightSegments():
            geometry.convertToStraightSegment()
        geometry.convertToMultiType()
        error = validate_polygon_geometry(geometry)
        pending_count = task.append_candidate(geometry, error)
        if task.kind == "modify":
            selected_count = len(task.selected_feature_ids)
            expected_deleted = 0
            expected_added = 0
            if not error:
                plan = manual_edit_commit.plan_manual_modify_overlaps(
                    self._manual_modify_selected_features(task),
                    task.pending_geometries,
                )
                expected_deleted = len(plan.unmatched_old)
                expected_added = len(plan.unmatched_new)
            self._manual_panel.set_instruction(
                f"第 {pending_count} 个新边界不可保存：{error}" if error else
                f"本批已选 {selected_count} 个旧面、绘制 {pending_count} 个新边界；"
                f"预计删除旧面 {expected_deleted} 个、新增 {expected_added} 个；"
                "可继续绘制或保存"
            )
        else:
            self._manual_panel.set_instruction(
                f"第 {pending_count} 个候选不可保存：{error}" if error else
                f"本批待保存 {pending_count} 个面；可继续绘制，或选择目标类别后保存"
            )
        if error:
            self._manual_tools.schedule_transition("restore")
        else:
            self._manual_tools.schedule_transition("restart")
        if task.smoothing_enabled:
            self._schedule_manual_smoothing_preview()
        else:
            self._refresh_manual_pending_candidate_bands()
        self._update_manual_panel()

    def _manual_capture_cancelled(self):
        task = self._manual_task
        if not task or task.state not in ("capturing", "capture_transition"):
            return
        task.capture_cancelled()
        pending_count = len(task.pending_geometries)
        self._manual_panel.set_instruction(
            f"本次绘制已停止；本批仍有 {pending_count} 个待保存面"
            if pending_count else
            "本次绘制已停止；可以重新绘制或结束任务"
        )
        self._manual_tools.schedule_transition("restore")
        self._update_manual_panel()

    def _manual_modify_selected_features(self, task=None):
        task = task or self._manual_task
        if not task or task.kind != "modify":
            return []
        layer = self._layer(task.class_code)
        by_id = {feature.id(): QgsFeature(feature) for feature in layer.getFeatures()}
        return [
            by_id[feature_id]
            for feature_id in task.selected_feature_ids
            if feature_id in by_id
        ]

    def _refresh_manual_modify_reference(self):
        self._clear_manual_reference_band()
        task = self._manual_task
        if not task or task.kind != "modify":
            return
        layer = self._layer(task.class_code)
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
        if not task or task.kind not in ("add", "modify"):
            return
        target_code = task.target_code
        smooth_preview = self._manual_smoothing_preview_is_current(task)
        geometries = (
            task.smoothing_preview.geometries
            if smooth_preview else
            task.pending_geometries
        )
        errors = task.pending_errors
        for index, geometry in enumerate(geometries):
            error = errors[index] if index < len(errors) else ""
            self._manual_add_candidate_bands.append(
                self._new_manual_candidate_band(
                    geometry, target_code, error, smooth_preview=smooth_preview
                )
            )

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
        self._clear_manual_add_candidate_bands()
        self._clear_manual_reference_band()

    def _manual_primary_action(self):
        task = self._manual_task
        if not task:
            return
        kind = task.kind
        state = task.state
        if kind == "modify" and state not in ("committing", "paused"):
            self._commit_manual_modify_batch()
        elif kind == "delete" and state == "selecting":
            self._commit_manual_delete()
        elif (
            kind == "add"
            and state not in ("committing", "paused")
            and task.pending_geometries
        ):
            self._commit_manual_add()

    def _manual_retry_action(self):
        task = self._manual_task
        if not task:
            return
        if task.kind in ("modify", "add"):
            task.retry_candidate()
            if task.smoothing_enabled:
                self._schedule_manual_smoothing_preview()
            else:
                self._refresh_manual_pending_candidate_bands()
            self._start_manual_capture()

    def _manual_clear_action(self):
        task = self._manual_task
        if not task or task.kind != "delete":
            return
        self._layer(task.class_code).removeSelection()
        self._update_manual_panel()

    def _manual_cancel_action(self):
        self._cancel_manual_task()

    def _commit_manual_modify_batch(self):
        task = self._manual_task
        if not task or task.kind != "modify":
            return
        source_code = task.class_code
        target_code = task.target_code
        source = self._layer(source_code)
        target = self._layer(target_code)
        old_features = self._manual_modify_selected_features(task)
        if not old_features:
            QMessageBox.information(self, "修改现有面", "请先选择一个或多个灰色旧面")
            return
        try:
            preparation = manual_edit_operations.prepare_manual_modify(
                task,
                old_features,
                SmoothingParameters(*self._manual_panel.smoothing_parameters()),
            )
        except manual_edit_operations.InvalidManualGeometry as exc:
            description = (
                "新边界不可保存" if exc.stage == "raw" else "保存边界不可用"
            )
            QMessageBox.warning(
                self, "修改现有面", f"第 {exc.index} 个{description}：{exc.reason}"
            )
            return
        except manual_edit_operations.ManualModifyPlanError as exc:
            QMessageBox.warning(self, "修改现有面", str(exc))
            return
        except RuntimeError as exc:
            QMessageBox.information(self, "修改现有面", str(exc))
            return
        if preparation.raw_count == 0 and target_code == source_code:
            QMessageBox.information(
                self, "修改现有面", "没有绘制新边界且目标类别未改变"
            )
            return
        smoothing_text = (
            "；身份匹配按原始边界计算，保存当前光滑预览"
            if task.smoothing_enabled and preparation.raw_count else ""
        )
        run_spec = self._run_spec
        answer = QMessageBox.question(
            self,
            "保存本批修改",
            f"本批旧面 {len(old_features)} 个，新边界 "
            f"{len(preparation.geometries)} 个，保存后删除旧面 "
            f"{preparation.expected_deleted} 个、新增 "
            f"{preparation.expected_added} 个，"
            "相交的新边界继承旧面身份；目标类别为 "
            f"{target_code} {CLASS_NAMES[target_code]}{smoothing_text}。确定提交吗？",
            YES | NO,
            NO,
        )
        if answer != YES:
            return
        if self._manual_task is not task or self._run_spec is not run_spec:
            return

        self._manual_tools.stop_picker()
        self._manual_tools.stop_capture()
        task.state = "committing"
        self._update_manual_panel()
        keep_source_editing = bool(
            source.isEditable() or task.editing_started_by_task
        )
        try:
            with self._edit_tracker.suppress():
                result = manual_edit_commit.commit_manual_modify(
                    source_layer=source,
                    target_layer=target,
                    old_features=old_features,
                    plan=preparation.plan,
                    new_geometries=preparation.geometries,
                    source_code=source_code,
                    target_code=target_code,
                    run_spec=self._run_spec,
                    baseline_stream_id=self._workspace["baseline_stream_id"],
                    keep_source_editing=keep_source_editing,
                    confidence_statistics=self._optional_confidence_statistics,
                )
        except Exception as exc:
            task.set_failed(str(exc))
            QMessageBox.warning(self, "保存本批修改失败", str(exc))
            self._update_manual_panel()
            return

        manual_edit_operations.record_modify_history(
            self._run_spec, result, source_code, target_code
        )
        tracked_layers = [(source_code, source)]
        if target_code != source_code:
            tracked_layers.append((target_code, target))
        manual_edit_operations.restart_edit_tracking(
            self._edit_tracker, tracked_layers
        )
        self._set_class_modified(source_code)
        if target_code != source_code:
            self._set_class_modified(target_code)
            self._set_visible(target_code, True)
        self._workspace = class_workspace.save_workspace(
            self._run_spec, self._workspace, changed_class_codes={source_code, target_code}
        )
        self._refresh_class_display(source_code, target_code)
        task.record_modify_batch(
            len(result.matched), len(result.added), len(result.deleted)
        )
        self._manual_smoothing_timer.stop()
        source.removeSelection()
        self._clear_manual_bands()
        self._manual_panel.set_target_code(source_code)
        self.iface.setActiveLayer(source)
        self.baseline_label.setText(
            f"本批修改已保存：旧面 {len(old_features)} 个，新边界 "
            f"{len(preparation.geometries)} 个，删除旧面 "
            f"{preparation.expected_deleted} 个，新增 "
            f"{preparation.expected_added} 个；继续选择下一批"
        )
        self._refresh_table()
        self._manual_tools.start_picker()
        self._update_manual_panel()

    def _commit_manual_delete(self):
        task = self._manual_task
        if not task or task.kind != "delete":
            return
        class_code = task.class_code
        layer = self._layer(class_code)
        feature_ids = list(layer.selectedFeatureIds())
        if not feature_ids:
            QMessageBox.information(self, "删除现有面", "请先在地图中选择一个或多个面")
            return
        run_spec = self._run_spec
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
        if self._manual_task is not task or self._run_spec is not run_spec:
            return
        task.state = "committing"
        self._update_manual_panel()
        try:
            manual_edit_commit.commit_manual_delete(layer, feature_ids)
        except Exception as exc:
            task.state = "selecting"
            QMessageBox.warning(self, "删除现有面失败", str(exc))
            self._manual_tools.start_picker()
            self._update_manual_panel()
            return
        self.baseline_label.setText(
            f"已从 {class_code} {CLASS_NAMES[class_code]} 删除 {len(feature_ids)} 个面"
        )
        self._finish_manual_task_success()

    def _commit_manual_add(self):
        task = self._manual_task
        if not task or task.kind != "add":
            return
        source_code = task.class_code
        target_code = task.target_code
        if not task.pending_geometries:
            QMessageBox.information(self, "新增面", "当前没有待保存的新增面")
            return
        try:
            preparation = manual_edit_operations.prepare_manual_add(
                task,
                SmoothingParameters(*self._manual_panel.smoothing_parameters()),
            )
        except manual_edit_operations.InvalidManualGeometry as exc:
            description = (
                "候选不可保存" if exc.stage == "raw" else "保存边界不可用"
            )
            QMessageBox.warning(
                self,
                "新增面",
                f"本批第 {exc.index} 个{description}：{exc.reason}",
            )
            return
        except RuntimeError as exc:
            QMessageBox.information(self, "新增面", str(exc))
            return
        layer = self._layer(target_code)
        self._manual_tools.stop_capture()
        task.state = "committing"
        self._update_manual_panel()
        was_editable = layer.isEditable()
        try:
            with self._edit_tracker.suppress():
                result = manual_edit_commit.commit_manual_add(
                    layer=layer,
                    geometries=preparation.geometries,
                    target_code=target_code,
                    run_spec=self._run_spec,
                    baseline_stream_id=self._workspace["baseline_stream_id"],
                    keep_editing=bool(target_code == source_code or was_editable),
                    confidence_statistics=self._optional_confidence_statistics,
                )
        except Exception as exc:
            task.set_failed(str(exc))
            self._manual_tools.restore_previous()
            QMessageBox.warning(self, "保存本批新增面失败", str(exc))
            self._update_manual_panel()
            return
        manual_edit_operations.restart_edit_tracking(
            self._edit_tracker, [(target_code, layer)]
        )
        topology_hints = []
        for added in result.added:
            persisted = self._feature_by_object_id(layer, added.object_id)
            topology_hint = self._local_topology_hint(
                target_code, persisted.geometry(), persisted.id()
            )
            topology_hints.append(topology_hint)
            manual_edit_operations.record_add_history(
                self._run_spec,
                target_code,
                added,
                persisted.geometry(),
                topology_hint,
            )
        batch_size = len(result.added)
        self._set_class_modified(target_code)
        self._workspace = class_workspace.save_workspace(
            self._run_spec, self._workspace, changed_class_codes={target_code}
        )
        self._set_visible(target_code, True)
        self._refresh_class_display(target_code)
        task.record_add_batch(target_code, batch_size)
        self._manual_smoothing_timer.stop()
        self._clear_manual_add_candidate_bands()
        self._manual_panel.set_target_code(source_code)
        hint_summary = "；".join(dict.fromkeys(topology_hints))
        self.baseline_label.setText(
            f"已向 {target_code} {CLASS_NAMES[target_code]} 提交本批 {batch_size} 个面；"
            f"本次累计 {task.added_count} 个面；局部提示: {hint_summary}"
        )
        self._refresh_table()
        self._start_manual_capture()

    def _finish_manual_session(self):
        task = self._manual_task
        if not task:
            return
        if task.kind == "add":
            self._finish_add_task()
        elif task.kind == "modify":
            self._finish_modify_task()

    def _finish_modify_task(self):
        task = self._manual_task
        if not task or task.kind != "modify":
            return
        class_code = task.class_code
        discarded_old_count = len(task.selected_feature_ids)
        discarded_new_count = len(task.pending_geometries)
        batch_count = task.submitted_batch_count
        modified_count = task.modified_old_count
        added_count = task.saved_new_count
        deleted_count = task.deleted_old_count
        self._manual_smoothing_timer.stop()
        self._manual_tools.end_session()
        self._clear_manual_bands()
        if class_code in self._class_layers:
            manual_edit_operations.close_clean_task_editing_session(
                task, self._layer(class_code), self._edit_tracker
            )
        self._manual_task = None
        self._manual_panel.set_target_code(class_code)
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
        if not task or task.kind != "add":
            return
        added_count = task.added_count
        batch_count = task.submitted_batch_count
        discarded_count = len(task.pending_geometries)
        class_code = task.class_code
        self._manual_smoothing_timer.stop()
        self._manual_tools.end_session()
        self._clear_manual_bands()
        if class_code in self._class_layers:
            manual_edit_operations.close_clean_task_editing_session(
                task, self._layer(class_code), self._edit_tracker
            )
        self._manual_task = None
        self.baseline_label.setText(
            f"新增已结束：本次提交 {batch_count} 批、保存 {added_count} 个面；"
            f"丢弃 {discarded_count} 个未保存候选"
        )
        self._manual_panel.set_target_code(class_code)
        self._update_manual_panel()
        self._refresh_table()

    def _qgis_smooth_parameters(self) -> tuple[int, float, float]:
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

    def _qgis_smooth_preview_is_current(self) -> bool:
        preview = self._qgis_smooth_preview
        layer = self.iface.activeLayer()
        if (
            not preview or layer is None or not layer.isEditable()
            or self._manual_task
            or layer.id() != preview.layer_id
            or SmoothingParameters(*self._qgis_smooth_parameters())
            != preview.batch.parameters
            or tuple(sorted(layer.selectedFeatureIds())) != preview.feature_ids
        ):
            return False
        for feature_id, expected_hash in zip(
            preview.feature_ids, preview.batch.source_hashes
        ):
            feature = layer.getFeature(feature_id)
            if (
                not feature.isValid()
                or geometry_source_hash(feature.geometry()) != expected_hash
            ):
                return False
        return True

    def _update_qgis_smoothing_controls(self) -> None:
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

    def _clear_qgis_smooth_preview(
        self, _checked: bool = False, status_text: str | None = None
    ) -> None:
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

    def _preview_qgis_smoothing(self) -> None:
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
        parameters = SmoothingParameters(*self._qgis_smooth_parameters())
        try:
            batch = smooth_geometry_batch(
                [feature.geometry() for feature in features],
                parameters,
            )
        except GeometrySmoothingError as exc:
            self._clear_qgis_smooth_preview(
                status_text=f"光滑结果不可用：{exc}"
            )
            QMessageBox.warning(
                self,
                "预览光滑失败",
                f"光滑结果不可用：{exc}；本批未产生预览",
            )
            return
        self._clear_qgis_smooth_preview()
        color = QColor("#00bcd4")
        fill = QColor(color)
        fill.setAlpha(45)
        for geometry in batch.geometries:
            band = QgsRubberBand(
                self.iface.mapCanvas(), Qgis.GeometryType.Polygon
            )
            band.setStrokeColor(color)
            band.setFillColor(fill)
            band.setLineStyle(DASH_LINE)
            band.setWidth(2)
            band.setToGeometry(geometry, layer)
            self._qgis_smooth_preview_bands.append(band)
        self._qgis_smooth_preview = NativeSmoothingPreview(
            layer_id=layer.id(),
            class_code=class_code,
            feature_ids=tuple(feature.id() for feature in features),
            batch=batch,
        )
        statistics = batch.statistics
        self.qgis_smooth_status_label.setText(
            f"已预览 {len(features)} 个面：顶点 "
            f"{statistics.source_vertex_count} → {statistics.smoothed_vertex_count}，"
            f"总面积变化 {statistics.area_change_percent:+.3f}%；"
            "应用后仍需保存 QGIS 编辑"
        )
        self._update_qgis_smoothing_controls()

    def _apply_qgis_smoothing(self) -> None:
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
        geometries = tuple(QgsGeometry(geometry) for geometry in preview.batch.geometries)
        parameters = preview.batch.parameters
        self._clear_qgis_smooth_preview()
        layer.beginEditCommand(f"光滑选中 {len(geometries)} 个面")
        try:
            for feature_id, geometry in zip(preview.feature_ids, geometries):
                if not layer.changeGeometry(feature_id, geometry):
                    raise RuntimeError(f"无法更新要素 {feature_id} 的 geometry")
            layer.endEditCommand()
        except Exception as exc:
            layer.destroyEditCommand()
            QMessageBox.warning(self, "应用光滑失败", str(exc))
            self._update_manual_panel()
            return
        self.qgis_smooth_status_label.setText(
            f"已应用到 {len(geometries)} 个面：次数 {parameters.iterations}、偏移 "
            f"{parameters.offset:.2f}、最大角度 {parameters.max_angle:.0f}°；"
            "尚未保存，可撤销一步"
        )
        self.baseline_label.setText(
            f"类别 {preview.class_code} 已应用光滑预览；尚未保存 QGIS 编辑"
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
        final_code = task.target_code if task.kind == "modify" else task.class_code
        self._manual_tools.end_session()
        self._clear_manual_bands()
        self._manual_task = None
        if final_code in self._class_layers:
            self._select_class_context(final_code, activate_layer=True)
            self._manual_panel.set_target_code(final_code)
        self._update_manual_panel()
        self._refresh_table()

    def _cancel_manual_task(self, _checked=False, silent=False, restore_selection=True):
        task = self._manual_task
        if not task:
            return
        kind = task.kind
        class_code = task.class_code
        pending_count = len(task.pending_geometries)
        self._manual_smoothing_timer.stop()
        self._manual_tools.end_session()
        self._clear_manual_bands()
        if class_code in self._class_layers:
            manual_edit_operations.close_clean_task_editing_session(
                task, self._layer(class_code), self._edit_tracker
            )
        if restore_selection and class_code in self._class_layers:
            layer = self._layer(class_code)
            existing = {feature.id() for feature in layer.getFeatures()}
            layer.selectByIds([
                feature_id for feature_id in task.selection_before
                if feature_id in existing
            ])
        added_count = task.added_count
        self._manual_task = None
        self._manual_panel.set_target_code(class_code)
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
        if not hasattr(self, "_manual_panel"):
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
        task = self._manual_task
        modified = bool(self._editable_modified_layers()) if self._workspace else False
        idle_enabled = bool(
            self._workspace and class_code is not None and not task
            and not self._active_session and not modified
        )
        pending_count = len(task.pending_geometries) if task else 0
        smoothing_enabled = bool(task and task.smoothing_enabled)
        self._manual_panel.render(
            ManualPanelSnapshot(
                current_class_text=current_text,
                active_layer_name=active_name,
                selected_count=selected_count,
                edit_text=edit_text,
                idle_enabled=idle_enabled,
                has_features=bool(layer and layer.featureCount()),
                has_workspace=bool(self._workspace),
                kind=task.kind if task else None,
                state=task.state if task else None,
                modify_selected_count=(
                    len(task.selected_feature_ids)
                    if task and task.kind == "modify" else 0
                ),
                delete_selected_count=(
                    self._layer(task.class_code).selectedFeatureCount()
                    if task and task.kind == "delete" else 0
                ),
                pending_count=pending_count,
                pending_has_error=bool(task and any(task.pending_errors)),
                target_changed=bool(task and task.target_code != task.class_code),
                smoothing_enabled=smoothing_enabled,
                smoothing_ready=bool(
                    not smoothing_enabled
                    or self._manual_smoothing_preview_is_current(task)
                ),
            )
        )
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

    def _editing_started(self, class_code):
        layer = self._layer(class_code)
        if not self._edit_tracker.begin(class_code, layer):
            return
        self._update_manual_panel()
        self._update_actions()

    def _editing_stopped(self, class_code):
        preview = self._qgis_smooth_preview
        if preview and preview.class_code == int(class_code):
            self._clear_qgis_smooth_preview()
        if not self._workspace:
            return
        layer = self._layer(class_code)
        result = self._edit_tracker.finish(
            class_code,
            layer,
            run_spec=self._run_spec,
            baseline_stream_id=self._workspace["baseline_stream_id"],
            confidence_statistics=self._optional_confidence_statistics,
        )
        if result is None:
            return
        if not (
            result.changed_ids or result.deleted_ids or result.added_ids
        ):
            self._update_manual_panel()
            self._update_actions()
            return
        hints = [
            self._local_topology_hint(
                class_code, feature.geometry(), feature.id()
            )
            for feature in result.affected_features
        ]
        if hints:
            self.baseline_label.setText(
                f"类别 {class_code} 已保存；局部拓扑提示: " + "; ".join(hints)
            )
        self._mark_class_modified(class_code)
        self._refresh_class_display(class_code)
        self._update_manual_panel()

    @staticmethod
    def _feature_by_object_id(layer, object_id):
        wanted = str(object_id or "")
        request = QgsFeatureRequest().setFilterExpression(
            '"object_id" = \'' + wanted.replace("'", "''") + "'"
        ).setLimit(1)
        for feature in layer.getFeatures(request):
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

    def _invalidate_final(self):
        self._final_path = ""
        self._final_input_identities = {}
        self._final_feature_count = None
        self._accepted_feature_count = None
        self._accepted_warnings = ()
        self._issue_count = None
        self._update_accept_enabled()

    def _set_class_modified(self, class_code):
        self._invalidate_final()
        record = self._workspace["classes"][str(class_code)]
        record["modified"] = True
        record["confirmed"] = False
        record["state"] = "editing" if self._layer(class_code).featureCount() else "unreviewed_empty"

    def _mark_class_modified(self, class_code):
        self._set_class_modified(class_code)
        self._workspace = class_workspace.save_workspace(
            self._run_spec, self._workspace, changed_class_codes={class_code}
        )
        self._refresh_table()

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

    def _sam_available(self):
        return bool(
            self._sam_config.get("enabled")
            and self._sam_config.get("checkpoint_sha256")
            and Path(str(self._sam_config.get("checkpoint") or "")).is_file()
        )

    def _request_sam(self, class_code: int, missed: bool) -> None:
        self._select_class_context(class_code, activate_layer=True)
        self._begin_sam(class_code, missed=missed)

    def _begin_sam(self, class_code: int, missed: bool = False) -> None:
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
        self._active_session = SamSession.start(
            class_code,
            "missed" if missed else "existing",
            class_workspace.workspace_timestamp(),
        )
        self._workspace["active_sam_session_id"] = self._active_session.session_id
        self._workspace = class_workspace.save_workspace(
            self._run_spec, self._workspace, changed_class_codes=()
        )
        self._sam_panel.render(
            SamPanelSnapshot(
                state="waiting_click",
                existing=not missed,
                message=(
                    f"当前类别: {class_code} {CLASS_NAMES[class_code]} | "
                    + (
                        "点击漏标地物位置"
                        if missed
                        else "点击当前类别工作层中的一个已有面"
                    )
                ),
            )
        )
        self._sam_preview.start_pick()
        self._refresh_table()
        self._update_actions()

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
        if not session or session.state != "waiting_click":
            return
        class_code = session.class_code
        layer = self._layer(class_code)
        canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
        try:
            hits = self._features_at_map_point(layer, map_point)
            feature = None
            if session.mode == "existing":
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
                session.begin_inference(
                    click_raster={"x": raster_point.x(), "y": raster_point.y()},
                    geometry_bounds=bounds,
                    feature_id=feature.id(),
                    object_id=str(feature.attribute("object_id") or ""),
                    part_id=str(feature.attribute("part_id") or "000"),
                    current_geometry_hash=class_workspace.geometry_hash(
                        feature.geometry()
                    ),
                    current_source=str(
                        feature.attribute("geometry_source") or "fusion"
                    ),
                    current_revision=int(
                        feature.attribute("geometry_revision") or 0
                    ),
                )
                self._sam_preview.show_current(feature.geometry(), layer)
            else:
                session.begin_inference(
                    click_raster={"x": raster_point.x(), "y": raster_point.y()},
                    geometry_bounds=None,
                )
            self._sam_preview.restore_map_tool()
            self._sam_panel.render(
                SamPanelSnapshot(
                    state="inference",
                    existing=session.mode == "existing",
                    message=(
                        f"当前类别: {class_code} {CLASS_NAMES[class_code]} | "
                        f"object_id: {session.object_id or '新增'} | "
                        f"当前来源: {session.current_source or '新增'} | SAM3 推理中"
                    ),
                )
            )
            self._start_worker_request()
        except Exception as exc:
            self._sam_preview.restore_map_tool()
            self._session_failed(str(exc))

    def _start_worker_request(self) -> None:
        session = self._active_session
        if session is None:
            return
        stream = class_workspace.stream_by_id(
            self._eligible_fusions, self._workspace["baseline_stream_id"]
        )
        session.queue_request(
            run_id=self._run_spec["run_id"],
            raster=self._run_spec["raster"]["path"],
            confidence_mosaic=(stream.get("paths") or {}).get(
                "confidence_mosaic", ""
            ),
            crop_size_px=512,
            buffer_px=int(self._sam_config.get("buffer_px", 32)),
            checkpoint_sha256=str(
                self._sam_config.get("checkpoint_sha256") or ""
            ),
            sam_version=str(self._sam_config.get("version") or ""),
            device=str(
                self._sam_config.get("effective_device")
                or self._sam_config.get("requested_device")
                or "cpu"
            ),
        )
        if self._worker is None:
            self._worker = Sam3WorkerRunner(
                self._scripts_dir, self._sam_config, self
            )
            self._worker.ready.connect(self._submit_pending_worker_request)
            self._worker.event_received.connect(self._worker_event)
            self._worker.log_line.connect(self._worker_log)
            self._worker.stopped.connect(self._worker_stopped)
        if self._worker.is_ready:
            self._submit_pending_worker_request()
        else:
            self._worker.start()

    def _submit_pending_worker_request(self) -> None:
        session = self._active_session
        if session is None or self._worker is None:
            return
        request = session.take_pending_request()
        if request is None:
            return
        try:
            self._worker.predict(request)
        except Exception as exc:
            self._session_failed(str(exc))

    def _worker_event(self, event: dict[str, object]) -> None:
        session = self._active_session
        if not session:
            return
        if event.get("event") == "candidate_ready":
            event_session_id = str(event.get("session_id") or "")
            if (
                not event_session_id
                or event_session_id != session.session_id
                or session.state != "inference"
            ):
                return
            geometry = QgsGeometry.fromWkt(str(event.get("geometry_wkt") or ""))
            if geometry.isNull() or geometry.isEmpty() or not geometry.isGeosValid():
                self._session_failed("SAM3 返回了无效候选几何")
                return
            raster_crs = QgsCoordinateReferenceSystem(self._run_spec["raster"]["crs"])
            layer = self._layer(session.class_code)
            geometry.transform(
                QgsCoordinateTransform(raster_crs, layer.crs(), QgsProject.instance())
            )
            topology_hint = self._local_topology_hint(
                session.class_code, geometry, session.feature_id
            )
            if not session.accept_candidate(
                session_id=event_session_id,
                geometry=geometry,
                score=float(event.get("score") or 0.0),
                confidence_mean=float(event.get("confidence_mean") or 0.0),
                confidence_std=float(event.get("confidence_std") or 0.0),
                crop_window=event.get("crop_window"),
                elapsed_sec=event.get("elapsed_sec"),
            ):
                return
            session.topology_hint = topology_hint
            self._sam_preview.show_candidate(geometry, layer)
            self._sam_panel.render(
                SamPanelSnapshot(
                    state="candidate",
                    existing=session.mode == "existing",
                    message=(
                        f"当前类别: {session.class_code} "
                        f"{CLASS_NAMES[session.class_code]} | "
                        f"object_id: {session.object_id or '新增'} | "
                        f"当前来源: {session.current_source or '新增'} | "
                        f"候选 score: {session.candidate_score:.4f}"
                    ),
                    topology_hint=topology_hint,
                )
            )
        elif event.get("event") == "failed":
            self._session_failed(
                str(event.get("error") or "SAM3 未知错误"),
                session_id=str(event.get("session_id") or "") or None,
            )

    def _worker_log(self, level: str, message: str) -> None:
        if level == "stderr" and self._active_session:
            self._sam_panel.replace_log(message[-6000:])

    def _worker_stopped(self, event: dict[str, object]) -> None:
        if not event.get("expected") and self._active_session:
            self._session_failed(
                f"SAM3 worker 意外退出，returncode={event.get('returncode')}"
            )

    def _session_failed(
        self, error: str, *, session_id: str | None = None
    ) -> None:
        session = self._active_session
        if not session or not session.fail(error, session_id=session_id):
            return
        self._sam_panel.render(
            SamPanelSnapshot(
                state="failed",
                existing=session.mode == "existing",
                message="SAM3 失败；当前工作层未修改",
                topology_hint="未执行",
                error=error,
            )
        )

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

    def _retry_session(self) -> None:
        session = self._active_session
        if not session or session.state != "failed" or not session.click_raster:
            return
        old_session_id = session.session_id
        if self._worker:
            self._worker.cancel(old_session_id)
            self._worker.close_session(old_session_id)
        self._record_session("failed")
        session.retry(class_workspace.workspace_timestamp())
        self._sam_preview.clear_candidate()
        self._workspace["active_sam_session_id"] = session.session_id
        self._workspace = class_workspace.save_workspace(
            self._run_spec, self._workspace, changed_class_codes=()
        )
        self._sam_panel.render(
            SamPanelSnapshot(
                state="inference",
                existing=session.mode == "existing",
                message="SAM3 正在重试；当前工作层未修改",
                topology_hint="未执行",
            )
        )
        self._start_worker_request()

    def _finish_session(self, decision: SamDecision) -> None:
        session = self._active_session
        if not session:
            return
        if decision in ("adopted", "edit_sam3") and session.state != "candidate":
            return
        if decision == "kept_current" and session.state != "candidate":
            return
        try:
            if decision == "adopted":
                self._adopt_candidate(session, edit=False)
            elif decision == "edit_sam3":
                self._adopt_candidate(session, edit=True)
            elif decision == "edit_current":
                if session.mode == "missed":
                    raise RuntimeError("新增漏标面没有当前几何可编辑")
                self._start_session_edit_current(session)
            if self._worker and decision == "cancelled":
                self._worker.cancel(session.session_id)
            self._record_session(decision)
        except Exception as exc:
            QMessageBox.warning(self, "SAM3 决定失败", str(exc))
            return
        self._cancel_active_session(record=False, keep_edit=decision in ("edit_current", "edit_sam3"))
        self._refresh_table()

    def _adopt_candidate(self, session: SamSession, edit: bool) -> None:
        layer = self._layer(session.class_code)
        self.iface.setActiveLayer(layer)
        before_snapshot = edit_tracking.snapshot_features(
            layer, {session.feature_id} if session.mode == "existing" else set()
        )
        confidence_mean, confidence_std, confidence_warning = (
            self._optional_confidence_statistics(
                layer, session.candidate_geometry
            )
        )
        with self._edit_tracker.suppress():
            if not layer.isEditable() and not layer.startEditing():
                raise RuntimeError("无法启动类别工作层编辑")
            if session.mode == "existing":
                feature_id = session.feature_id
                layer.changeGeometry(feature_id, session.candidate_geometry)
                values = {
                    "geometry_source": "manual_edited" if edit else "sam3",
                    "geometry_revision": session.current_revision + 1,
                    "edit_base": "sam3" if edit else "",
                    "sam_session_id": session.session_id,
                    "sam_score": session.candidate_score,
                    "sam_version": str(self._sam_config.get("version") or ""),
                    "confidence_mean": confidence_mean,
                    "confidence_std": confidence_std,
                    "reviewed": 0,
                    "updated_at": class_workspace.workspace_timestamp(),
                }
                edit_tracking.set_feature_attributes(layer, feature_id, values)
            else:
                feature = QgsFeature(layer.fields())
                feature.setGeometry(session.candidate_geometry)
                object_id = class_workspace.new_object_id(self._run_spec)
                values = {
                    "run_id": self._run_spec["run_id"],
                    "object_id": object_id,
                    "part_id": "000",
                    "class_code": session.class_code,
                    "class_name": CLASS_NAMES[session.class_code],
                    "baseline_stream_id": self._workspace["baseline_stream_id"],
                    "geometry_source": "manual_edited" if edit else "sam3",
                    "geometry_revision": 1,
                    "edit_base": "sam3" if edit else "",
                    "sam_session_id": session.session_id,
                    "sam_score": session.candidate_score,
                    "sam_version": str(self._sam_config.get("version") or ""),
                    "confidence_mean": confidence_mean,
                    "confidence_std": confidence_std,
                    "reviewed": 0,
                    "updated_at": class_workspace.workspace_timestamp(),
                }
                for name, value in values.items():
                    if layer.fields().indexOf(name) >= 0:
                        feature.setAttribute(name, value)
                if not layer.addFeature(feature):
                    raise RuntimeError("无法向类别工作层新增 SAM3 面")
                feature_id = feature.id()
                session.object_id = object_id
                session.feature_id = feature_id
            if not edit and not layer.commitChanges():
                errors = "; ".join(layer.commitErrors())
                layer.rollBack()
                raise RuntimeError(f"无法保存 SAM3 候选: {errors}")
        if edit:
            self._edit_tracker.prepare_edit(
                session.class_code,
                baseline=before_snapshot,
                metadata_prepared=True,
                session_id=session.session_id,
            )
            action = getattr(self.iface, "actionVertexTool", lambda: None)()
            if action is not None:
                action.trigger()
            return
        self._edit_tracker.discard(session.class_code)
        persisted = self._feature_by_object_id(layer, session.object_id)
        session.feature_id = persisted.id()
        session.persisted_geometry_hash = class_workspace.geometry_hash(
            persisted.geometry()
        )
        self._mark_class_modified(session.class_code)
        if confidence_warning:
            class_workspace.append_history(
                self._run_spec,
                "confidence_statistics_unavailable",
                class_code=session.class_code,
                object_id=session.object_id,
                reason=confidence_warning,
            )
        class_workspace.append_history(
            self._run_spec,
            "sam3_adopted" if session.mode == "existing" else "sam3_feature_added",
            class_code=session.class_code,
            object_id=session.object_id,
            sam_session_id=session.session_id,
            before_geometry_hash=session.current_geometry_hash,
            after_geometry_hash=session.persisted_geometry_hash,
        )
        self.baseline_label.setText(
            f"SAM3 已采用；局部拓扑提示: {session.topology_hint or '无'}"
        )

    def _start_session_edit_current(self, session: SamSession) -> None:
        layer = self._layer(session.class_code)
        self.iface.setActiveLayer(layer)
        if not layer.isEditable() and not layer.startEditing():
            raise RuntimeError("无法启动当前工作层编辑")
        layer.selectByIds([session.feature_id])
        self._edit_tracker.prepare_edit(
            session.class_code,
            metadata_prepared=False,
            session_id=session.session_id,
        )
        action = getattr(self.iface, "actionVertexTool", lambda: None)()
        if action is not None:
            action.trigger()

    def _record_session(self, decision: SamDecision) -> None:
        session = self._active_session
        if session is None:
            return
        record = session.history_record(
            decision,
            run_id=self._run_spec.get("run_id"),
            baseline_stream_id=(self._workspace or {}).get(
                "baseline_stream_id", ""
            ),
            checkpoint_sha256=str(
                self._sam_config.get("checkpoint_sha256") or ""
            ),
            sam_version=str(self._sam_config.get("version") or ""),
            device=str(
                self._sam_config.get("effective_device")
                or self._sam_config.get("requested_device")
                or "cpu"
            ),
            geometry_hash=class_workspace.geometry_hash,
        )
        class_workspace.append_sam_session(self._run_spec, record)

    def _cancel_active_session(self, record=True, keep_edit=False):
        session = self._active_session
        if session and record:
            self._record_session("cancelled")
        if session and self._worker:
            self._worker.cancel(session.session_id)
            self._worker.close_session(session.session_id)
        if session:
            session.cancel()
        self._sam_preview.cleanup()
        self._active_session = None
        if self._workspace is not None:
            self._workspace["active_sam_session_id"] = ""
            self._workspace = class_workspace.save_workspace(
                self._run_spec, self._workspace, changed_class_codes=()
            )
        self._sam_panel.render(SamPanelSnapshot())
        self._refresh_table()
        self._update_actions()

    def _confirm_class(self, class_code, checked):
        if self._edit_tracker.suppressed or not self._workspace:
            return
        self._select_class_context(class_code, activate_layer=True)
        layer = self._layer(class_code)
        if checked and (
            self._active_session
            or self._manual_task
            or self._accepted_task
            or self._refinement_task
            or (layer.isEditable() and layer.isModified())
        ):
            QMessageBox.warning(
                self,
                "确认类别",
                "存在活动 SAM3、人工操作或未保存编辑，不能确认",
            )
            self._refresh_table()
            return
        record = self._workspace["classes"][str(class_code)]
        keep_editing = layer.isEditable()
        try:
            class_review.commit_class_review(
                layer,
                reviewed=checked,
                now=class_workspace.workspace_timestamp,
                tracker=self._edit_tracker,
                keep_editing=keep_editing,
            )
        except Exception as exc:
            QMessageBox.warning(self, "确认类别失败", str(exc))
            return
        record["confirmed"] = bool(checked)
        self._invalidate_final()
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
        self._workspace = class_workspace.save_workspace(
            self._run_spec, self._workspace, changed_class_codes={class_code}
        )
        self._refresh_table()

    def _refresh_table(self):
        if not self._workspace:
            self.class_review_panel.render(
                ClassReviewSnapshot(
                    tuple(
                        ClassReviewRow(
                            class_code=code,
                            class_name=CLASS_NAMES[code],
                            visible=False,
                            feature_count=None,
                            review_text="未初始化",
                            confirmed=False,
                            unsaved=False,
                            manual_reason="请先初始化 14 类工作层",
                            sam_reason="请先初始化 14 类工作层",
                            confirm_reason="请先初始化 14 类工作层",
                        )
                        for code in CLASS_ORDER
                    )
                )
            )
            self._update_actions()
            return
        has_unsaved_edits = bool(self._editable_modified_layers())
        selection_locked, selection_locked_reason = self._class_selection_lock()
        rows = []
        workspace_classes = self._workspace.get("classes") or {}
        for code in CLASS_ORDER:
            record = workspace_classes.get(str(code), {})
            layer = None
            if code in self._class_layers:
                layer = QgsProject.instance().mapLayer(self._class_layers[code])
            stats = self._workspace_statistics.get(code)
            if stats and self._manual_only:
                status_text = (
                    f"原始 {stats.get('fusion', 0)} / "
                    f"人工 {stats.get('manual_edited', 0)}"
                )
            elif stats:
                status_text = (
                    f"Fusion {stats.get('fusion', 0)} / SAM3 {stats.get('sam3', 0)} / "
                    f"人工 {stats.get('manual_edited', 0)}"
                )
            elif layer is not None:
                status_text = "工作层已挂载"
            elif code in self._layer_loader.pending_codes:
                status_text = "等待后台分批挂载"
            else:
                status_text = "尚未挂载；点击可按需加载"
            feature_count = (
                layer.featureCount()
                if layer is not None
                else int(record.get("feature_count") or 0)
            )
            visible = code in self._pending_visible_codes
            if layer is not None:
                tree_layer = QgsProject.instance().layerTreeRoot().findLayer(layer.id())
                visible = bool(tree_layer and tree_layer.itemVisibilityChecked())
            manual_reason = ""
            sam_reason = ""
            confirm_reason = ""
            if layer is None:
                manual_reason = "工作层尚未挂载；选择类别后将按需加载"
                sam_reason = manual_reason
                confirm_reason = manual_reason
            elif selection_locked:
                manual_reason = selection_locked_reason
                sam_reason = selection_locked_reason
                confirm_reason = selection_locked_reason
            elif has_unsaved_edits:
                sam_reason = "存在未保存编辑，请先保存或回滚"
                confirm_reason = sam_reason
            elif self._manual_only:
                sam_reason = "纯人工模式不提供 SAM 辅助"
            elif not self._sam_available():
                sam_reason = "SAM3 checkpoint、SHA 或环境检查未通过"
            sam_base = not self._manual_only and self._sam_available()
            sam_ready = bool(layer is not None and sam_base and not has_unsaved_edits)
            rows.append(
                ClassReviewRow(
                    class_code=code,
                    class_name=CLASS_NAMES[code],
                    visible=visible,
                    feature_count=feature_count,
                    review_text=status_text,
                    confirmed=bool(record.get("confirmed")),
                    unsaved=bool(layer and layer.isEditable() and layer.isModified()),
                    manual_enabled=bool(layer is not None and not selection_locked),
                    manual_reason=manual_reason,
                    sam_existing_enabled=bool(sam_ready and feature_count > 0),
                    sam_missed_enabled=sam_ready,
                    sam_reason=sam_reason,
                    confirm_enabled=bool(
                        layer is not None
                        and not selection_locked
                        and not has_unsaved_edits
                    ),
                    confirm_reason=confirm_reason,
                )
            )
        selected_class_code = self._manual_locked_class_code() or self._current_class_code()
        self.class_review_panel.render(
            ClassReviewSnapshot(
                tuple(rows),
                selected_class_code=selected_class_code,
                selection_locked=selection_locked,
                selection_locked_reason=selection_locked_reason,
            )
        )
        self._update_manual_panel()
        self._update_actions()
        self.workspace_changed.emit(dict(self._workspace))

    def _class_selection_lock(self):
        if self._accepted_task is not None:
            return True, "正在后台写入 accepted_labels，不能切换类别"
        if self._refinement_task is not None:
            return True, "正在后台组装/检查，不能切换类别"
        if self._manual_task is not None:
            return True, "人工任务进行中，不能切换类别"
        if self._active_session is not None:
            return True, "SAM3 会话进行中，不能切换类别"
        return False, ""

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
            and self._accepted_task is None
            and self._refinement_task is None
        )
        self.assemble_btn.setEnabled(can_assemble)
        issue_text = "-" if self._issue_count is None else str(self._issue_count)
        self.summary_label.setText(
            f"14 类确认: {confirmed}/14    未解决问题: {issue_text}    "
            f"未保存编辑: {len(modified)}"
        )
        self._update_accept_enabled()

    def _assemble_final(self):
        self._start_refinement_task(assemble=True)

    def _start_refinement_task(self, *, assemble):
        if self._refinement_task is not None or not self._workspace:
            return
        if self._editable_modified_layers() or self._active_session or self._manual_task:
            QMessageBox.warning(self, "后台检查", "请先保存编辑并结束当前人工/SAM3 操作")
            return
        self._issue_count = None
        self._update_accept_enabled()
        try:
            task = RefinementTask(
                self._run_spec, self._workspace, final_path=self._final_path,
                assemble=assemble,
                transform_context=QgsCoordinateTransformContext(QgsProject.instance().transformContext()),
            )
            self._refinement_task = task
            task.taskCompleted.connect(self._refinement_completed)
            task.taskTerminated.connect(self._refinement_terminated)
            self.fusion_combo.setEnabled(False)
            self.initialize_btn.setEnabled(False)
            self.topology_btn.setEnabled(False)
            self.cancel_load_btn.setText("取消组装/检查")
            self.cancel_load_btn.show()
            self.baseline_label.setText("正在后台组装/检查；地图仍可浏览，完成前请勿修改输入")
            self._refresh_table()
            self._update_actions()
            QgsApplication.taskManager().addTask(task)
        except Exception as exc:
            QMessageBox.warning(self, "后台检查失败", str(exc))

    def _refinement_completed(self):
        task = self.sender()
        if task is not self._refinement_task:
            return
        try:
            if task.isCanceled():
                self.baseline_label.setText("后台组装/检查已取消；现有结果未改变")
                return
            if not task.inputs_unchanged() or self._editable_modified_layers():
                raise RuntimeError("计算期间输入或编辑缓冲已变化，请保存后重新检查")
            result = task.result_data
            published = task.publish()
            if task.assemble:
                self._final_path = published["final_path"]
                feature_count = result.get("feature_count")
                self._final_feature_count = (
                    int(feature_count) if feature_count is not None else None
                )
                self._accepted_feature_count = None
                self._accepted_warnings = ()
                self._final_input_identities = {
                    str(record["path"]): task.identities[str(record["path"])]
                    for record in task.workspace["classes"].values()
                }
            issues_path = published["issues_path"]
            self._issues_path = issues_path
            self._issue_count = int(result["issue_count"])
            self.allow_issues_check.setChecked(False)
            if task.assemble:
                self.layer_manager.load_final_composite(self._run_spec["run_id"], self._final_path)
            self.layer_manager.load_topology_issues(self._run_spec["run_id"], issues_path)
            self.baseline_label.setText(f"后台组装/拓扑检查完成；问题数 {self._issue_count}")
        except Exception as exc:
            self._issue_count = None
            QMessageBox.warning(self, "结果未发布", str(exc))
        finally:
            task.discard()
            self._finish_refinement_task()

    def _refinement_terminated(self):
        task = self.sender()
        if task is not self._refinement_task:
            return
        if task.isCanceled():
            self.baseline_label.setText("后台组装/检查已取消；现有结果未改变")
        else:
            QMessageBox.warning(self, "后台检查失败", task.error_message)
        self._finish_refinement_task()

    def _finish_refinement_task(self):
        self._refinement_task = None
        self.fusion_combo.setEnabled(self._workspace is None and bool(self._eligible_fusions))
        self.cancel_load_btn.setText("取消后台加载")
        self.cancel_load_btn.hide()
        self._refresh_table()
        self._update_actions()

    def _check_topology(self):
        if not self._final_path:
            return
        if not self._final_matches_workspace():
            QMessageBox.warning(self, "拓扑检查", "分类工作层已变化，请先重新组装最终图层")
            return
        self._start_refinement_task(assemble=False)

    def _write_accepted(self):
        if (
            self._accepted_task is not None
            or self._refinement_task is not None
            or not self._final_matches_workspace()
        ):
            self._update_accept_enabled()
            QMessageBox.warning(self, "入库", "请保存编辑、确认全部类别并重新组装最终图层")
            return
        if self._issue_count is None:
            QMessageBox.warning(self, "入库", "必须先完成拓扑检查")
            return
        if self._issue_count != 0 and not self.allow_issues_check.isChecked():
            QMessageBox.warning(self, "入库", "请先解决拓扑问题，或明确勾选带问题入库")
            return
        self._accepted_generation += 1
        generation = self._accepted_generation
        self._accepted_feature_count = None
        self._accepted_warnings = ()
        try:
            task = AcceptedWriteTask(
                generation,
                run_id=str(self._run_spec.get("run_id") or ""),
                final_path=self._final_path,
                accepted_path=str(
                    self._run_spec.get("accepted_target_gpkg") or ""
                ),
                run_manifest_path=str(
                    self._run_spec.get("accepted_write_manifest")
                    or Path(self._run_spec["run_dir"]) / "run_manifest.json"
                ),
                workspace_input_identities=self._final_input_identities,
                transform_context=QgsProject.instance().transformContext(),
            )
            self._accepted_task = task
            task.progressChanged.connect(self._accepted_progress_changed)
            task.taskCompleted.connect(self._accepted_completed)
            task.taskTerminated.connect(self._accepted_terminated)
            self.cancel_load_btn.setText("取消 accepted 校验")
            self.cancel_load_btn.show()
            self.baseline_label.setText(
                "正在后台校验 accepted_labels；地图仍可浏览，提交前可取消"
            )
            self._refresh_table()
            self._update_actions()
            QgsApplication.taskManager().addTask(task)
        except Exception as exc:
            self._accepted_task = None
            self._update_actions()
            QMessageBox.warning(self, "写入 accepted_labels 失败", str(exc))

    def _accepted_task_is_current(self, task) -> bool:
        return bool(
            task is self._accepted_task
            and task.generation == self._accepted_generation
            and task.run_id == str(self._run_spec.get("run_id") or "")
        )

    def _accepted_progress_changed(self, _value):
        task = self.sender()
        if not self._accepted_task_is_current(task):
            return
        self.baseline_label.setText(task.progress_message)
        if task.commit_started:
            self.cancel_load_btn.setText("正在提交，不能取消")
            self.cancel_load_btn.setEnabled(False)
        self._update_accept_enabled()

    def _accepted_completed(self):
        task = self.sender()
        current = self._accepted_task_is_current(task)
        if task is self._accepted_task:
            self._accepted_task = None
        if current:
            result = task.result_data or {}
            feature_count = result.get("feature_count")
            if task.published and feature_count is not None:
                self._accepted_feature_count = int(feature_count)
                self._accepted_warnings = tuple(
                    str(warning)
                    for warning in result.get("warnings") or ()
                    if str(warning)
                )
                self.baseline_label.setText(
                    "已写入 accepted_labels: "
                    f"{self._accepted_feature_count} 个面"
                )
            else:
                self._accepted_feature_count = None
                self._accepted_warnings = ()
        self._finish_accepted_task(current=current)

    def _accepted_terminated(self):
        task = self.sender()
        current = self._accepted_task_is_current(task)
        if task is self._accepted_task:
            self._accepted_task = None
        if current:
            self._accepted_feature_count = None
            self._accepted_warnings = ()
            if task.isCanceled() and not task.error_message:
                self.baseline_label.setText(
                    "accepted_labels 写入已取消；目标未改变"
                )
            else:
                QMessageBox.warning(
                    self,
                    "写入 accepted_labels 失败",
                    task.error_message or "后台写入未完成",
                )
        self._finish_accepted_task(current=current)

    def _finish_accepted_task(self, *, current):
        if current:
            self.cancel_load_btn.setEnabled(True)
            self.cancel_load_btn.setText("取消后台加载")
            self.cancel_load_btn.hide()
            self._refresh_table()
        self._update_actions()

    def _retire_accepted_task(self):
        task = self._accepted_task
        self._accepted_generation += 1
        if task is not None and not task.commit_started:
            task.cancel()

    def _final_matches_workspace(self):
        if (
            not self._final_path or not self._workspace
            or not self._final_input_identities
            or self._active_session or self._manual_task
            or self._editable_modified_layers()
        ):
            return False
        classes = self._workspace["classes"]
        if any(not classes.get(str(code), {}).get("confirmed") for code in CLASS_ORDER):
            return False
        try:
            return self._final_input_identities == {
                str(record["path"]): file_identity(record["path"])
                for record in classes.values()
            }
        except OSError:
            return False

    def _update_accept_enabled(self, *_args):
        current = (
            self._accepted_task is None
            and self._refinement_task is None
            and self._final_matches_workspace()
        )
        self.topology_btn.setEnabled(current)
        write_enabled = bool(
            current
            and self._issue_count is not None
            and (self._issue_count == 0 or self.allow_issues_check.isChecked())
            and self._accepted_feature_count is None
        )
        self._render_admission_summary(write_enabled)

    def _admission_write_reason(self, current):
        if self._accepted_task is not None:
            return str(self._accepted_task.progress_message or "正在后台写入")
        if self._refinement_task is not None:
            return self._refinement_task.description()
        if self._accepted_feature_count is not None:
            return "本次结果已写入标签库"
        if not self._workspace:
            return "请先加载并确认 14 类工作层"
        if self._active_session or self._manual_task:
            return "请先结束当前人工或 SAM3 操作"
        if self._editable_modified_layers():
            return "请先保存未保存编辑"
        confirmed = sum(
            1
            for record in self._workspace["classes"].values()
            if record.get("confirmed")
        )
        if confirmed != len(CLASS_ORDER):
            return f"尚有 {len(CLASS_ORDER) - confirmed} 类未确认"
        if not self._final_path:
            return "请先组装最终图层"
        if not self._final_input_identities:
            return "最终成果需要重新组装"
        if not current:
            return "分类工作层已变化，请重新组装最终图层"
        if self._issue_count is None:
            return "必须先完成拓扑检查"
        if self._issue_count != 0 and not self.allow_issues_check.isChecked():
            return "请先解决拓扑问题，或明确勾选带问题入库"
        return ""

    def _render_admission_summary(self, write_enabled):
        run_spec = self._run_spec or {}
        current = (
            self._accepted_task is None
            and self._refinement_task is None
            and self._final_matches_workspace()
        )
        confirmed = (
            sum(
                1
                for record in self._workspace["classes"].values()
                if record.get("confirmed")
            )
            if self._workspace
            else None
        )
        unsaved_edits = (
            len(self._editable_modified_layers()) if self._workspace else None
        )
        background_stage = ""
        if self._accepted_task is not None:
            background_stage = str(self._accepted_task.progress_message or "")
        elif self._refinement_task is not None:
            background_stage = self._refinement_task.description()
        reason = self._admission_write_reason(current)
        self.admission_summary_panel.render(
            AdmissionSummarySnapshot(
                target_path=str(run_spec.get("accepted_target_gpkg") or ""),
                final_feature_count=self._final_feature_count,
                confirmed_class_count=confirmed,
                unsaved_edit_count=unsaved_edits,
                topology_executed=self._issue_count is not None,
                topology_issue_count=self._issue_count,
                allow_issues=self.allow_issues_check.isChecked(),
                background_stage=background_stage,
                blocker_reason=reason,
                write_enabled=write_enabled,
                write_reason=reason,
                accepted_feature_count=self._accepted_feature_count,
                accepted_warnings=self._accepted_warnings,
            )
        )

    def cleanup(self):
        self._retire_accepted_task()
        self._cancel_background_load(silent=True)
        self._layer_loader.reset()
        self._clear_qgis_smooth_preview()
        self._manual_smoothing_timer.stop()
        self._cancel_manual_task(silent=True)
        self._manual_tools.cleanup()
        self._cancel_active_session(record=False)
        self._layer_signals.cleanup()
        self._edit_tracker.reset()
        if self._worker is not None:
            retire_after(self._worker, self._worker.stopped)
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
        self._retire_accepted_task()
        self._cancel_background_load(silent=True)
        if self._worker is not None:
            retire_after(self._worker, self._worker.stopped)
            self._worker.stop()
            self._worker = None
        self._clear_qgis_smooth_preview()
        event.ignore()
        self.hide()
