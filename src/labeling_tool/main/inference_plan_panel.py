"""State-owning UI component for one inference-plan selection."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtWidgets import QGroupBox, QLabel, QPushButton, QVBoxLayout, QWidget

from labeling_tool.main.inference_config_dialog import InferenceConfigDialog
from labeling_tool.main.model_registry import ModelRegistry


@dataclass(frozen=True)
class InferenceSelection:
    """Immutable model, Fusion, boundary, and confirmation state for one Run."""

    model_ids: tuple[str, ...] = ()
    fusion_profile_id: str | None = None
    boundary_smoothing_enabled: bool = True
    confirmed: bool = False


@dataclass(frozen=True)
class ResolvedLaunchPlan:
    """Runnable values resolved from one UI selection and report."""

    model_ids: tuple[str, ...]
    fusion_profile_id: str | None
    boundary_smoothing_enabled: bool


def resolve_launch_plan(
    report: Mapping[str, Any],
    selection: InferenceSelection,
) -> ResolvedLaunchPlan:
    """Resolve a selected plan and require a ready device check for every model."""

    effective = report.get("effective", {})
    registry = ModelRegistry(effective)
    model_ids = registry.resolve_selection(
        selection.model_ids,
        selection.fusion_profile_id,
    )
    checks_by_id = {str(item.get("id")): item for item in report.get("checks") or []}
    unavailable = [
        model_id
        for model_id in model_ids
        if (checks_by_id.get(f"semantic_model_{model_id}") or {}).get("status")
        != "ready"
    ]
    if unavailable:
        raise ValueError("模型未通过设备实测: " + ", ".join(unavailable))
    return ResolvedLaunchPlan(
        model_ids=model_ids,
        fusion_profile_id=selection.fusion_profile_id,
        boundary_smoothing_enabled=selection.boundary_smoothing_enabled,
    )


class InferencePlanPanel(QGroupBox):
    """Own plan selection UI and its configuration-dialog lifecycle.

    `get_report` obtains the current manager-owned report when the dialog is
    opened.  The panel only retains the immutable selection snapshot.
    """

    selection_changed = pyqtSignal()

    def __init__(
        self,
        get_report: Callable[[], dict[str, Any]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("推理方案", parent)
        self.setObjectName("inferencePlanGroup")
        self._get_report = get_report
        self._selection = InferenceSelection()
        self.configuration_dialog = InferenceConfigDialog(self)
        self.configuration_dialog.configuration_applied.connect(
            self._apply_configuration
        )
        self._build_ui()

    @property
    def selection(self) -> InferenceSelection:
        """Return the current immutable Run-selection snapshot."""

        return self._selection

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.summary_label = QLabel("请先完成推理环境检查")
        self.summary_label.setObjectName("inferencePlanSummaryLabel")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        self.configure_button = QPushButton("选择模型与 Fusion")
        self.configure_button.setObjectName("inferencePlanConfigureButton")
        self.configure_button.setEnabled(False)
        self.configure_button.setToolTip("环境检查完成后选择本次模型和融合方案")
        self.configure_button.clicked.connect(self._show_configuration)
        layout.addWidget(self.configure_button)

    def restore_selection(self, selection: InferenceSelection) -> None:
        """Restore persisted choices while deliberately requiring a new confirmation."""

        self._selection = InferenceSelection(
            model_ids=tuple(selection.model_ids),
            fusion_profile_id=selection.fusion_profile_id,
            boundary_smoothing_enabled=selection.boundary_smoothing_enabled,
        )
        if self._get_report().get("effective"):
            self._render_summary()
        else:
            self.summary_label.setText("请先完成推理环境检查")

    def invalidate(self) -> None:
        """Withdraw confirmation after an environment or task-input change."""

        was_confirmed = self._selection.confirmed
        self._selection = InferenceSelection(
            model_ids=self._selection.model_ids,
            fusion_profile_id=self._selection.fusion_profile_id,
            boundary_smoothing_enabled=self._selection.boundary_smoothing_enabled,
        )
        self.configure_button.setEnabled(False)
        self.summary_label.setText("请先完成推理环境检查")
        if was_confirmed:
            self.selection_changed.emit()

    def set_environment(self, report: Mapping[str, Any]) -> None:
        """Adapt choices to a new report and always require a fresh confirmation."""

        report_copy = dict(report)
        effective = report_copy.get("effective") or {}
        previous = self._selection
        try:
            registry = ModelRegistry(effective)
        except (KeyError, TypeError, ValueError):
            self.configuration_dialog.set_environment(
                report_copy,
                previous.model_ids,
                previous.fusion_profile_id,
                previous.boundary_smoothing_enabled,
            )
            self._selection = InferenceSelection(
                model_ids=previous.model_ids,
                fusion_profile_id=previous.fusion_profile_id,
                boundary_smoothing_enabled=previous.boundary_smoothing_enabled,
            )
            self.configure_button.setEnabled(False)
            self.summary_label.setText("请先完成推理环境检查")
            if self._selection != previous:
                self.selection_changed.emit()
            return

        checks = {str(item.get("id")): item for item in report_copy.get("checks") or []}
        available_ids = tuple(
            model.model_id
            for model in registry.models
            if model.enabled
            and (checks.get(f"semantic_model_{model.model_id}") or {}).get("status")
            == "ready"
        )
        model_ids = (
            tuple(
                model_id for model_id in previous.model_ids if model_id in available_ids
            )
            or available_ids
        )
        profile_ids = {profile.profile_id for profile in registry.profiles}
        profile_id = (
            previous.fusion_profile_id
            if previous.fusion_profile_id in profile_ids
            else None
        )
        self._selection = InferenceSelection(
            model_ids=model_ids,
            fusion_profile_id=profile_id,
            boundary_smoothing_enabled=previous.boundary_smoothing_enabled,
        )
        self.configuration_dialog.set_environment(
            report_copy,
            model_ids,
            profile_id,
            previous.boundary_smoothing_enabled,
        )
        self.configure_button.setEnabled(effective.get("schema_version") == 2)
        self._render_summary(report_copy)
        if self._selection != previous:
            self.selection_changed.emit()

    def _show_configuration(self) -> None:
        report = self._get_report()
        selection = self._selection
        self.configuration_dialog.set_environment(
            report,
            selection.model_ids,
            selection.fusion_profile_id,
            selection.boundary_smoothing_enabled,
        )
        self.configuration_dialog.show()
        self.configuration_dialog.raise_()
        self.configuration_dialog.activateWindow()

    def _apply_configuration(
        self,
        model_ids: list[str],
        profile_id: str | None,
        boundary_smoothing_enabled: bool,
    ) -> None:
        self._selection = InferenceSelection(
            model_ids=tuple(model_ids),
            fusion_profile_id=profile_id,
            boundary_smoothing_enabled=boundary_smoothing_enabled,
            confirmed=True,
        )
        self._render_summary()
        self.selection_changed.emit()

    def _render_summary(self, report: Mapping[str, Any] | None = None) -> None:
        report = report if report is not None else self._get_report()
        effective = report.get("effective") or {}
        by_id = {
            str(item.get("model_id")): str(
                item.get("display_name") or item.get("model_id")
            )
            for item in effective.get("semantic_models") or []
        }
        names = [
            by_id.get(model_id, model_id) for model_id in self._selection.model_ids
        ]
        profile = self._selection.fusion_profile_id or "无融合"
        device = (effective.get("runtime") or {}).get("effective_device", "未检查")
        boundary = effective.get("boundary_fitting") or {}
        if not self._selection.boundary_smoothing_enabled:
            boundary_text = "关闭，保留原始像元边界；模型分类不变"
        elif boundary.get("mode") == "divider_cubic_bspline_adaptive_v2":
            boundary_text = "平滑公共分界线；模型分类不变"
        else:
            boundary_text = "待环境检查确认；模型分类不变"
        plan_status = "已确认" if self._selection.confirmed else "待确认"
        self.summary_label.setText(
            f"已选模型（{len(names)}）: {', '.join(names) if names else '未选择'}\n"
            f"实际语义设备: {device}\n"
            f"Fusion: {profile}；边界: {boundary_text}\n"
            f"方案状态: {plan_status}"
        )

    def cleanup(self) -> None:
        """Idempotently close the owned configuration dialog."""

        try:
            self.configuration_dialog.close()
        except RuntimeError:
            pass
