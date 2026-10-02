"""Wide model registry and fusion-profile selection dialog."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from typing import Any

from qgis.PyQt.QtCore import Qt, QUrl, pyqtSignal
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
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from labeling_tool.main.fusion_profile import profile_summary
from labeling_tool.main.model_registry import ModelRegistry
from labeling_tool.qgis_support.dialog_geometry import fit_dialog_to_screen
from labeling_tool.qgis_support.qt6_api import (
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


def _file_sha256(path: str) -> str:
    if not path or not os.path.isfile(path):
        return "不可用"
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class InferenceConfigDialog(QDialog):
    configuration_applied = pyqtSignal(object, object, bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("选择模型与 Fusion")
        self.setModal(True)
        self._report = {}
        self._registry = None
        self._selected_ids = []
        self._profile_id = None
        self._boundary_smoothing_enabled = True
        self._row_by_model = {}
        self._build_ui()
        fit_dialog_to_screen(self, preferred_size=(1120, 680))

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(10)
        self.status_label = QLabel("尚未加载环境检查结果")
        self.status_label.setObjectName("inferenceConfigStatusLabel")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        self.selection_summary_label = QLabel("尚未选择模型")
        self.selection_summary_label.setObjectName("inferenceConfigSelectionSummary")
        self.selection_summary_label.setWordWrap(True)
        root.addWidget(self.selection_summary_label)

        scroll = QScrollArea()
        scroll.setObjectName("inferenceConfigScrollArea")
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)

        models_group = QGroupBox("本次模型")
        models_layout = QVBoxLayout(models_group)
        self.model_table = QTableWidget(0, 5)
        self.model_table.setHorizontalHeaderLabels(
            ["运行", "模型", "可用性", "结果", "实际设备"]
        )
        self.model_table.verticalHeader().setVisible(False)
        self.model_table.setAlternatingRowColors(True)
        self.model_table.setSelectionBehavior(SELECT_ROWS)
        self.model_table.setEditTriggers(NO_EDIT_TRIGGERS)
        self.model_table.setWordWrap(True)
        self.model_table.setMinimumHeight(112)
        self.model_table.setMaximumHeight(176)
        self.model_table.itemChanged.connect(self._render_selected_models)
        header = self.model_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, RESIZE_TO_CONTENTS)
        header.setSectionResizeMode(1, STRETCH)
        header.setSectionResizeMode(2, RESIZE_TO_CONTENTS)
        header.setSectionResizeMode(3, STRETCH)
        header.setSectionResizeMode(4, RESIZE_TO_CONTENTS)
        models_layout.addWidget(self.model_table)
        content_layout.addWidget(models_group)

        profile_group = QGroupBox("融合方案")
        profile_form = QFormLayout(profile_group)
        self.profile_combo = QComboBox()
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        profile_form.addRow("本次融合:", self.profile_combo)
        self.profile_summary_label = QLabel("无融合：只保存各模型独立结果")
        self.profile_summary_label.setWordWrap(True)
        profile_form.addRow("结果影响:", self.profile_summary_label)
        profile_action_row = QHBoxLayout()
        self.profile_path_label = QLabel("-")
        self.profile_path_label.setWordWrap(True)
        self.open_profile_btn = QPushButton("打开")
        self.open_profile_btn.setEnabled(False)
        self.open_profile_btn.clicked.connect(self._open_profile)
        profile_action_row.addWidget(self.open_profile_btn)
        profile_action_row.addStretch(1)
        profile_form.addRow("配置文件:", profile_action_row)
        content_layout.addWidget(profile_group)

        boundary_group = QGroupBox("输出边界")
        boundary_layout = QVBoxLayout(boundary_group)
        self.boundary_smoothing_check = QCheckBox("使用平滑后的类别边界")
        self.boundary_smoothing_check.setChecked(True)
        self.boundary_smoothing_check.setToolTip(
            "只改变输出矢量几何；不会改变模型分类、权重或类别映射"
        )
        self.boundary_smoothing_check.toggled.connect(self._render_boundary_effect)
        boundary_layout.addWidget(self.boundary_smoothing_check)
        self.boundary_effect_label = QLabel()
        self.boundary_effect_label.setWordWrap(True)
        boundary_layout.addWidget(self.boundary_effect_label)
        content_layout.addWidget(boundary_group)

        self.details_toggle = QPushButton("显示技术详情（路径、校验和、参数）")
        self.details_toggle.setObjectName("inferenceConfigDetailsToggle")
        self.details_toggle.setCheckable(True)
        self.details_toggle.toggled.connect(self._set_details_visible)
        content_layout.addWidget(self.details_toggle)

        self.technical_details_group = QGroupBox("技术详情（可查看和复制）")
        technical_form = QFormLayout(self.technical_details_group)
        self.model_details_label = QLabel("尚未加载")
        self.model_details_label.setWordWrap(True)
        technical_form.addRow("模型版本与文件:", self.model_details_label)
        technical_form.addRow("Fusion 路径 / SHA:", self.profile_path_label)
        self.scaling_label = QLabel("尚未加载")
        self.scaling_label.setWordWrap(True)
        technical_form.addRow("分区与存储:", self.scaling_label)
        self.boundary_label = QLabel("尚未加载")
        self.boundary_label.setWordWrap(True)
        technical_form.addRow("边界拟合参数:", self.boundary_label)
        for label, name in (
            (self.model_details_label, "模型版本与文件"),
            (self.profile_path_label, "Fusion 配置路径与校验"),
            (self.scaling_label, "分区与存储参数"),
            (self.boundary_label, "边界拟合参数"),
        ):
            label.setTextInteractionFlags(
                TEXT_SELECTABLE_BY_MOUSE
                | Qt.TextInteractionFlag.TextSelectableByKeyboard
            )
            label.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            label.setAccessibleName(name)
        advanced_note = QLabel(
            "参数修改只在 config.yaml 中进行；修改后必须回主面板重新检查环境并重新应用方案。"
        )
        advanced_note.setWordWrap(True)
        technical_form.addRow(advanced_note)
        self.technical_details_group.setVisible(False)
        content_layout.addWidget(self.technical_details_group)

        self.sam_label = QLabel(
            "推理结果就绪后，可在分类修整窗口使用 SAM3 辅助修改边界。"
        )
        self.sam_label.setWordWrap(True)
        content_layout.addWidget(self.sam_label)
        content_layout.addStretch(1)
        scroll.setWidget(content)
        root.addWidget(scroll, stretch=1)

        buttons = QDialogButtonBox(APPLY | CANCEL)
        buttons.button(APPLY).setText("应用")
        buttons.button(CANCEL).setText("取消")
        buttons.button(APPLY).clicked.connect(self._apply)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _set_details_visible(self, visible: bool) -> None:
        self.technical_details_group.setVisible(visible)
        self.details_toggle.setText(
            "隐藏技术详情" if visible else "显示技术详情（路径、校验和、参数）"
        )

    def _render_selected_models(self, _item=None) -> None:
        selected = []
        for model_id, row in self._row_by_model.items():
            item = self.model_table.item(row, 0)
            if item is not None and item.checkState() == CHECKED:
                selected.append(model_id)
        names = []
        if self._registry is not None:
            names = [
                self._registry.model(model_id).display_name for model_id in selected
            ]
        self.selection_summary_label.setText(
            f"已选择 {len(names)} 个模型: {', '.join(names) if names else '未选择'}"
        )

    def _render_boundary_effect(self, enabled: bool) -> None:
        if enabled:
            self.boundary_effect_label.setText(
                "开启后会对输出矢量的公共类别分界线进行平滑；"
                "不改变模型分类、权重或类别映射。"
            )
        else:
            self.boundary_effect_label.setText(
                "关闭后保留原始像元边界；不改变模型分类、权重或类别映射。"
            )

    def set_environment(
        self,
        report: Mapping[str, Any],
        selected_model_ids: Sequence[str] = (),
        profile_id: str | None = None,
        boundary_smoothing_enabled: bool = True,
    ) -> None:
        """Load one editable plan draft from a manager-owned report snapshot."""

        self._clear_draft()
        self._report = dict(report or {})
        effective = self._report.get("effective") or {}
        try:
            self._registry = ModelRegistry(effective)
        except (KeyError, TypeError, ValueError) as exc:
            self._registry = None
            self.status_label.setText(f"模型注册表不可用: {exc}")
            return
        available = {model.model_id for model in self._registry.models if model.enabled}
        self._selected_ids = [
            model_id for model_id in selected_model_ids if model_id in available
        ]
        if not self._selected_ids:
            checks = self._check_map()
            self._selected_ids = [
                model.model_id
                for model in self._registry.models
                if model.enabled
                and (checks.get(f"semantic_model_{model.model_id}") or {}).get("status")
                == "ready"
            ]
        self._profile_id = profile_id
        self._boundary_smoothing_enabled = bool(boundary_smoothing_enabled)
        self.boundary_smoothing_check.setChecked(self._boundary_smoothing_enabled)
        status = self._report.get("status", "error")
        status_text = {
            "ready": "检查通过",
            "warning": "有提示",
            "error": "未通过",
        }.get(status, "未检查")
        device = (effective.get("runtime") or {}).get("effective_device", "未知")
        self.status_label.setText(f"配置状态: {status_text}    语义设备: {device}")
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
        runtime = self._registry.runtime
        cache_budget = scaling.get("score_cache_budget_gb")
        cache_label = (
            "auto（Run 启动时按磁盘解析）"
            if str(cache_budget).lower() == "auto"
            else f"{cache_budget} GiB"
        )
        self.scaling_label.setText(
            f"Partition {scaling.get('partition_tile_rows')} × {scaling.get('partition_tile_cols')} Tile；"
            f"Halo {scaling.get('partition_halo_px')}；Seam {scaling.get('seam_band_px')} px；"
            f"score cache {cache_label}；"
            f"磁盘保留 {scaling.get('min_free_disk_gb')} GiB；"
            f"CPU worker {scaling.get('max_cpu_partition_workers')}"
            f"（GPU 同时运行时 {scaling.get('max_cpu_partition_workers_with_package')}）；"
            f"Tile batch {runtime.get('tile_batch_size')}；"
            f"Tile I/O {scaling.get('tile_io_workers')}；"
            f"Tile 分页 {scaling.get('tile_page_size')}"
        )
        boundary = self._registry.boundary_fitting
        self.boundary_label.setText(
            "公共分界线单次 Cubic B-Spline；两侧 Polygon 共用稀疏拟合线；"
            f"平滑因子 {boundary.get('smoothing_factor')}；"
            f"曲线采样 {boundary.get('curve_sampling_spacing_px')} px；"
            f"最大弦误差 {boundary.get('max_chord_error_px')} px；"
            f"最大弧长 {boundary.get('max_segment_arc_length_px')} px；"
            "不限制最大偏移，不执行拓扑修复或 Gap/Overlap 检查"
        )

    def _clear_draft(self) -> None:
        """Remove every row-indexed draft before accepting another report."""

        self._report = {}
        self._registry = None
        self._selected_ids = []
        self._profile_id = None
        self._boundary_smoothing_enabled = True
        self._row_by_model.clear()
        self.model_table.setRowCount(0)
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.blockSignals(False)
        self.profile_summary_label.setText("无融合：只保存各模型独立结果")
        self.profile_path_label.setText("-")
        self.open_profile_btn.setEnabled(False)
        self.model_details_label.setText("尚未加载")
        self.scaling_label.setText("尚未加载")
        self.boundary_label.setText("尚未加载")
        self.sam_label.setText(
            "推理结果就绪后，可在分类修整窗口使用 SAM3 辅助修改边界。"
        )
        self.boundary_smoothing_check.setChecked(True)
        self._render_selected_models()
        self._render_boundary_effect(True)

    def _check_map(self) -> dict[str, Mapping[str, Any]]:
        return {str(item.get("id")): item for item in self._report.get("checks") or []}

    def _populate_models(self) -> None:
        checks = self._check_map()
        self._row_by_model.clear()
        self.model_table.setRowCount(len(self._registry.models))
        details = []
        for row, model in enumerate(self._registry.models):
            self._row_by_model[model.model_id] = row
            check = checks.get(f"semantic_model_{model.model_id}", {})
            status = str(check.get("status") or "error")
            run_item = QTableWidgetItem()
            run_item.setFlags(ITEM_IS_ENABLED | ITEM_IS_USER_CHECKABLE)
            run_item.setCheckState(
                CHECKED if model.model_id in self._selected_ids else UNCHECKED
            )
            run_item.setData(USER_ROLE, model.model_id)
            self.model_table.setItem(row, 0, run_item)
            available = model.enabled and status == "ready"
            availability = "可用" if available else "不可用"
            availability_detail = (
                "模型在配置中禁用"
                if not model.enabled
                else str(
                    check.get("message")
                    or ("环境检查通过" if available else "未通过环境检查")
                )
            )
            values = [
                model.display_name,
                availability,
                "独立结果",
                (self._registry.runtime or {}).get("effective_device", "未知"),
            ]
            for column, value in enumerate(values, start=1):
                item = QTableWidgetItem(str(value))
                if column == 2:
                    item.setToolTip(availability_detail)
                self.model_table.setItem(row, column, item)
            details.append(
                f"{model.display_name} ({model.model_id})\n"
                f"版本: {model.version or '未提供'}\n"
                f"Artifact: {model.artifact or '未提供'}\n"
                f"路径: {model.artifact_path or '未提供'}\n"
                f"SHA256: {model.sha256 or '未提供'}\n"
                f"环境检查: {status}；{availability_detail}"
            )
        self.model_details_label.setText("\n\n".join(details) or "未配置语义模型")
        self.model_table.resizeRowsToContents()
        self._render_selected_models()

    def _populate_profiles(self) -> None:
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItem("无融合", None)
        selected_index = 0
        for profile in self._registry.profiles:
            runnable = (
                profile.enabled and profile.available and profile.status == "approved"
            )
            label = f"{profile.profile_id}（{'可运行' if runnable else '不可运行'}）"
            self.profile_combo.addItem(label, profile.profile_id)
            index = self.profile_combo.count() - 1
            item = self.profile_combo.model().item(index)
            if item is not None:
                if not runnable:
                    check = (
                        self._check_map().get(f"fusion_profile_{profile.profile_id}")
                        or {}
                    )
                    item.setToolTip(
                        "该 profile 仅可查看，不能运行。"
                        + str(check.get("message") or "未通过或部署资产不完整")
                    )
            if profile.profile_id == self._profile_id:
                selected_index = index
        self.profile_combo.setCurrentIndex(selected_index)
        self.profile_combo.blockSignals(False)
        self._on_profile_changed(selected_index)

    def _on_profile_changed(self, _index: int) -> None:
        if self._registry is None:
            return
        profile_id = self.profile_combo.currentData()
        required = set()
        if profile_id:
            profile = self._registry.profile(profile_id)
            required = set(profile.required_model_ids)
            summary = profile_summary(profile.profile)
            model_names = [
                self._registry.model(model_id).display_name
                if model_id in self._row_by_model
                else model_id
                for model_id in profile.required_model_ids
            ]
            self.profile_summary_label.setText(
                f"使用 {profile.profile_id}：将 {', '.join(model_names) or '指定模型'} 的结果融合为一份候选；"
                "仍会保留已勾选模型的独立结果。"
            )
            check = self._check_map().get(f"fusion_profile_{profile_id}") or {}
            runnable = (
                profile.enabled and profile.available and profile.status == "approved"
            )
            if runnable:
                validation = "可运行：Schema、模型引用与 SHA 校验通过"
            else:
                message = str(check.get("message") or "profile 未通过或部署资产不完整")
                fix = str(check.get("fix") or f"检查 {profile.file_path}")
                validation = f"不可运行：{message}\n修改位置：{fix}"
                self.profile_summary_label.setText(
                    f"不可运行：{message}\n" + self.profile_summary_label.text()
                )
            self.profile_path_label.setText(
                f"{profile.file_path}\nSHA256: {_file_sha256(profile.file_path)}\n{validation}"
                f"\n融合策略: {summary.get('strategy') or profile.strategy or '未提供'}；"
                f"\n版本状态: {profile.status}；"
                f"配置记录的审批: {'通过' if summary.get('approval_passed') else '未通过'}；"
                f"baseline mIoU: {summary.get('baseline_miou')}；"
                f"fusion mIoU: {summary.get('fusion_miou')}"
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
        self._render_selected_models()

    def _open_profile(self) -> None:
        profile_id = self.profile_combo.currentData()
        if profile_id and self._registry is not None:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(self._registry.profile(profile_id).file_path)
            )

    def _apply(self) -> None:
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
            model_id
            for model_id in resolved
            if (checks.get(f"semantic_model_{model_id}") or {}).get("status") != "ready"
        ]
        if broken:
            QMessageBox.warning(
                self, "模型未就绪", "以下模型未通过设备实测: " + ", ".join(broken)
            )
            return
        self._selected_ids = list(resolved)
        self._profile_id = profile_id
        self._boundary_smoothing_enabled = self.boundary_smoothing_check.isChecked()
        self.configuration_applied.emit(
            list(resolved),
            profile_id,
            self._boundary_smoothing_enabled,
        )
        self.accept()
