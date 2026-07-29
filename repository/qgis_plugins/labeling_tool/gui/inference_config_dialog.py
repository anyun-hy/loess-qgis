"""Wide model registry and fusion-profile selection dialog."""

from __future__ import annotations

import hashlib
import os

from qgis.PyQt.QtCore import QUrl, pyqtSignal
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..core.fusion_profile import profile_summary
from ..core.model_registry import ModelRegistry
from ..qt_compat import (
    APPLY,
    CANCEL,
    CHECKED,
    ITEM_IS_ENABLED,
    ITEM_IS_USER_CHECKABLE,
    NO_EDIT_TRIGGERS,
    RESIZE_TO_CONTENTS,
    SELECT_ROWS,
    STRETCH,
    TEXT_SELECTABLE_BY_MOUSE,
    UNCHECKED,
    USER_ROLE,
)


def _file_sha256(path):
    if not path or not os.path.isfile(path):
        return "不可用"
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class InferenceConfigDialog(QDialog):
    configuration_applied = pyqtSignal(object, object, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择模型与 Fusion")
        self.setModal(True)
        self.resize(1120, 680)
        self.setMinimumSize(900, 560)
        self._report = {}
        self._registry = None
        self._selected_ids = []
        self._profile_id = None
        self._boundary_smoothing_enabled = True
        self._row_by_model = {}
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        self.status_label = QLabel("尚未加载环境检查结果")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        self.model_table = QTableWidget(0, 7)
        self.model_table.setHorizontalHeaderLabels(
            ["运行", "模型", "版本", "角色", "Artifact / 路径 / SHA", "设备测试", "状态"]
        )
        self.model_table.verticalHeader().setVisible(False)
        self.model_table.setAlternatingRowColors(True)
        self.model_table.setSelectionBehavior(SELECT_ROWS)
        self.model_table.setEditTriggers(NO_EDIT_TRIGGERS)
        header = self.model_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, RESIZE_TO_CONTENTS)
        header.setSectionResizeMode(1, STRETCH)
        header.setSectionResizeMode(2, RESIZE_TO_CONTENTS)
        header.setSectionResizeMode(3, RESIZE_TO_CONTENTS)
        header.setSectionResizeMode(4, STRETCH)
        header.setSectionResizeMode(5, RESIZE_TO_CONTENTS)
        header.setSectionResizeMode(6, RESIZE_TO_CONTENTS)
        root.addWidget(self.model_table, stretch=1)

        profile_group = QGroupBox("融合方案")
        profile_form = QFormLayout(profile_group)
        self.profile_combo = QComboBox()
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        profile_form.addRow("Profile:", self.profile_combo)
        self.profile_summary_label = QLabel("无融合：只保存各模型独立结果")
        self.profile_summary_label.setWordWrap(True)
        profile_form.addRow("详情:", self.profile_summary_label)
        profile_path_row = QHBoxLayout()
        self.profile_path_label = QLabel("-")
        self.profile_path_label.setWordWrap(True)
        self.open_profile_btn = QPushButton("打开")
        self.open_profile_btn.setEnabled(False)
        self.open_profile_btn.clicked.connect(self._open_profile)
        profile_path_row.addWidget(self.profile_path_label, stretch=1)
        profile_path_row.addWidget(self.open_profile_btn)
        profile_form.addRow("路径 / SHA:", profile_path_row)
        root.addWidget(profile_group)

        advanced_group = QGroupBox("高级执行参数（来自 config.yaml）")
        advanced_form = QFormLayout(advanced_group)
        self.scaling_label = QLabel("尚未加载")
        self.scaling_label.setWordWrap(True)
        self.scaling_label.setTextInteractionFlags(
            TEXT_SELECTABLE_BY_MOUSE
        )
        advanced_form.addRow("分区与存储:", self.scaling_label)
        self.boundary_smoothing_check = QCheckBox("对类别边界进行光滑处理")
        self.boundary_smoothing_check.setChecked(True)
        self.boundary_smoothing_check.setToolTip(
            "开启时执行公共分界 Cubic B-Spline；关闭时保留原始像元边界"
        )
        advanced_form.addRow("本次运行:", self.boundary_smoothing_check)
        self.boundary_label = QLabel("尚未加载")
        self.boundary_label.setWordWrap(True)
        self.boundary_label.setTextInteractionFlags(
            TEXT_SELECTABLE_BY_MOUSE
        )
        advanced_form.addRow("边界拟合:", self.boundary_label)
        advanced_note = QLabel(
            "参数修改只在 config.yaml 中进行；修改后必须回主面板重新检查环境并重新应用方案。"
        )
        advanced_note.setWordWrap(True)
        advanced_form.addRow(advanced_note)
        root.addWidget(advanced_group)

        self.sam_label = QLabel("SAM3 是 semantic_ready 后的独立后处理能力，不随本窗口启动。")
        self.sam_label.setWordWrap(True)
        root.addWidget(self.sam_label)

        buttons = QDialogButtonBox(
            APPLY | CANCEL
        )
        buttons.button(APPLY).setText("应用")
        buttons.button(CANCEL).setText("取消")
        buttons.button(APPLY).clicked.connect(self._apply)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def set_environment(
        self,
        report,
        selected_model_ids=(),
        profile_id=None,
        boundary_smoothing_enabled=True,
    ):
        self._report = dict(report or {})
        effective = self._report.get("effective") or {}
        try:
            self._registry = ModelRegistry(effective)
        except (KeyError, TypeError, ValueError) as exc:
            self._registry = None
            self.status_label.setText(f"模型注册表不可用: {exc}")
            self.model_table.setRowCount(0)
            self.profile_combo.clear()
            return
        available = {model.model_id for model in self._registry.models if model.enabled}
        self._selected_ids = [model_id for model_id in selected_model_ids if model_id in available]
        if not self._selected_ids:
            checks = self._check_map()
            self._selected_ids = [
                model.model_id for model in self._registry.models
                if model.enabled
                and (checks.get(f"semantic_model_{model.model_id}") or {}).get("status") == "ready"
            ]
        self._profile_id = profile_id
        self._boundary_smoothing_enabled = bool(boundary_smoothing_enabled)
        self.boundary_smoothing_check.setChecked(
            self._boundary_smoothing_enabled
        )
        status = self._report.get("status", "error")
        device = (effective.get("runtime") or {}).get("effective_device", "未知")
        self.status_label.setText(f"配置状态: {status}    语义设备: {device}")
        self._populate_models()
        self._populate_profiles()
        sam = effective.get("sam3") or {}
        if sam.get("enabled"):
            self.sam_label.setText(
                f"SAM3 后处理: 已配置，设备 {sam.get('effective_device', sam.get('requested_device', 'auto'))}；"
                "不随语义主流程自动执行。"
            )
        else:
            self.sam_label.setText("SAM3 后处理: 未启用；不影响语义模型运行。")
        scaling = self._registry.scaling
        self.scaling_label.setText(
            f"Partition {scaling.get('partition_tile_rows')} × {scaling.get('partition_tile_cols')} Tile；"
            f"Halo {scaling.get('partition_halo_px')}；Seam {scaling.get('seam_band_px')} px；"
            f"score cache {scaling.get('score_cache_budget_gb')} GiB；"
            f"磁盘保留 {scaling.get('min_free_disk_gb')} GiB；"
            f"CPU worker {scaling.get('max_cpu_partition_workers')}；"
            f"Tile 分页 {scaling.get('tile_page_size')}"
        )
        boundary = self._registry.boundary_fitting
        self.boundary_label.setText(
            "公共分界线单次 Cubic B-Spline；两侧 Polygon 共用拟合线；"
            f"平滑因子 {boundary.get('smoothing_factor')}；"
            f"输出间距 {boundary.get('output_spacing_px')} px；"
            "不限制最大偏移，不执行拓扑修复或 Gap/Overlap 检查"
        )

    def _check_map(self):
        return {str(item.get("id")): item for item in self._report.get("checks") or []}

    def _populate_models(self):
        checks = self._check_map()
        self._row_by_model.clear()
        self.model_table.setRowCount(len(self._registry.models))
        for row, model in enumerate(self._registry.models):
            self._row_by_model[model.model_id] = row
            check = checks.get(f"semantic_model_{model.model_id}", {})
            status = str(check.get("status") or "error")
            run_item = QTableWidgetItem()
            run_item.setFlags(ITEM_IS_ENABLED | ITEM_IS_USER_CHECKABLE)
            run_item.setCheckState(
                CHECKED
                if model.model_id in self._selected_ids
                else UNCHECKED
            )
            run_item.setData(USER_ROLE, model.model_id)
            self.model_table.setItem(row, 0, run_item)
            values = [
                model.display_name,
                model.version,
                "独立结果",
                f"{model.artifact}\n{model.artifact_path}\nSHA256: {model.sha256}",
                (self._registry.runtime or {}).get("effective_device", "未知"),
                {"ready": "正常", "warning": "警告", "error": "错误"}.get(status, status),
            ]
            for column, value in enumerate(values, start=1):
                item = QTableWidgetItem(str(value))
                if column == 6:
                    item.setToolTip(str(check.get("message") or ""))
                self.model_table.setItem(row, column, item)
        self.model_table.resizeRowsToContents()

    def _populate_profiles(self):
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItem("无融合", None)
        selected_index = 0
        for profile in self._registry.profiles:
            label = f"{profile.profile_id} | {profile.strategy or 'unknown'} | {profile.status}"
            self.profile_combo.addItem(label, profile.profile_id)
            index = self.profile_combo.count() - 1
            runnable = profile.enabled and profile.available and profile.status == "approved"
            item = self.profile_combo.model().item(index)
            if item is not None:
                if not runnable:
                    check = self._check_map().get(f"fusion_profile_{profile.profile_id}") or {}
                    item.setToolTip(
                        "该 profile 仅可查看，不能运行。"
                        + str(check.get("message") or "未通过或部署资产不完整")
                    )
            if profile.profile_id == self._profile_id:
                selected_index = index
        self.profile_combo.setCurrentIndex(selected_index)
        self.profile_combo.blockSignals(False)
        self._on_profile_changed(selected_index)

    def _on_profile_changed(self, _index):
        profile_id = self.profile_combo.currentData()
        required = set()
        if profile_id:
            profile = self._registry.profile(profile_id)
            required = set(profile.required_model_ids)
            summary = profile_summary(profile.profile)
            self.profile_summary_label.setText(
                f"状态 {profile.status}；策略 {summary.get('strategy') or '未知'}；"
                f"模型 {len(summary.get('model_ids') or [])}；"
                f"baseline mIoU {summary.get('baseline_miou')}；fusion mIoU {summary.get('fusion_miou')}；"
                f"approval {'通过' if summary.get('approval_passed') else '未通过'}"
            )
            check = self._check_map().get(f"fusion_profile_{profile_id}") or {}
            runnable = profile.enabled and profile.available and profile.status == "approved"
            if runnable:
                validation = "可运行：Schema、模型引用与 SHA 校验通过"
            else:
                message = str(check.get("message") or "profile 未通过或部署资产不完整")
                fix = str(check.get("fix") or f"检查 {profile.file_path}")
                validation = f"不可运行：{message}\n修改位置：{fix}"
            self.profile_path_label.setText(
                f"{profile.file_path}\nSHA256: {_file_sha256(profile.file_path)}\n{validation}"
            )
            self.open_profile_btn.setEnabled(os.path.isfile(profile.file_path))
        else:
            self.profile_summary_label.setText("无融合：只保存每个勾选模型的独立结果")
            self.profile_path_label.setText("-")
            self.open_profile_btn.setEnabled(False)
        for model_id, row in self._row_by_model.items():
            item = self.model_table.item(row, 0)
            role = self.model_table.item(row, 3)
            if model_id in required:
                item.setCheckState(CHECKED)
                item.setFlags(ITEM_IS_USER_CHECKABLE)
                role.setText("融合必需 + 独立结果")
            else:
                item.setFlags(ITEM_IS_ENABLED | ITEM_IS_USER_CHECKABLE)
                role.setText("额外模型 + 独立结果" if required else "独立结果")

    def _open_profile(self):
        profile_id = self.profile_combo.currentData()
        if profile_id:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._registry.profile(profile_id).file_path))

    def _apply(self):
        if self._registry is None:
            return
        selected = []
        for model_id, row in self._row_by_model.items():
            if self.model_table.item(row, 0).checkState() == CHECKED:
                selected.append(model_id)
        profile_id = self.profile_combo.currentData()
        try:
            resolved = self._registry.resolve_selection(selected, profile_id)
        except ValueError as exc:
            QMessageBox.warning(self, "推理方案无效", str(exc))
            return
        checks = self._check_map()
        broken = [
            model_id for model_id in resolved
            if (checks.get(f"semantic_model_{model_id}") or {}).get("status") == "error"
        ]
        if broken:
            QMessageBox.warning(self, "模型未就绪", "以下模型未通过设备实测: " + ", ".join(broken))
            return
        self._selected_ids = list(resolved)
        self._profile_id = profile_id
        self._boundary_smoothing_enabled = (
            self.boundary_smoothing_check.isChecked()
        )
        self.configuration_applied.emit(
            list(resolved),
            profile_id,
            self._boundary_smoothing_enabled,
        )
        self.accept()
