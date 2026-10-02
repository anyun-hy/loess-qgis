"""Operation stages around low-level manual geometry commits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from qgis.core import QgsFeature, QgsGeometry, QgsVectorLayer

from labeling_tool.refinement import class_workspace, edit_tracking, manual_edit_commit
from labeling_tool.refinement.geometry_smoothing import (
    SmoothingParameters,
    geometry_source_hash,
    validate_polygon_geometry,
)
from labeling_tool.refinement.manual_edit_state import ManualEditTask


class InvalidManualGeometry(RuntimeError):
    """Identify the invalid boundary without choosing its UI presentation."""

    def __init__(
        self,
        stage: Literal["raw", "saved"],
        index: int,
        reason: str,
    ) -> None:
        super().__init__(reason)
        self.stage = stage
        self.index = index
        self.reason = reason


class ManualModifyPlanError(RuntimeError):
    """Preserve the separate warning boundary for identity matching failures."""


@dataclass(frozen=True)
class ManualModifyPreparation:
    """Validated inputs needed by one low-level modify commit."""

    geometries: tuple[QgsGeometry, ...]
    plan: manual_edit_commit.ManualModifyPlan
    raw_count: int
    expected_deleted: int
    expected_added: int


@dataclass(frozen=True)
class ManualAddPreparation:
    """Validated geometries needed by one low-level add commit."""

    geometries: tuple[QgsGeometry, ...]


def prepare_manual_modify(
    task: ManualEditTask,
    old_features: Sequence[QgsFeature],
    smoothing_parameters: SmoothingParameters,
) -> ManualModifyPreparation:
    """Validate raw/save boundaries while matching identities from raw boundaries."""
    raw_geometries = tuple(QgsGeometry(item) for item in task.pending_geometries)
    _validate_geometries(raw_geometries, "raw")
    geometries = _geometries_for_commit(task, smoothing_parameters, raw_geometries)
    _validate_geometries(geometries, "saved")
    try:
        plan = (
            manual_edit_commit.plan_manual_modify_overlaps(old_features, raw_geometries)
            if raw_geometries
            else manual_edit_commit.ManualModifyPlan((), (), ())
        )
    except RuntimeError as exc:
        raise ManualModifyPlanError(str(exc)) from exc
    return ManualModifyPreparation(
        geometries=geometries,
        plan=plan,
        raw_count=len(raw_geometries),
        expected_deleted=len(plan.unmatched_old) if raw_geometries else 0,
        expected_added=len(plan.unmatched_new) if raw_geometries else 0,
    )


def prepare_manual_add(
    task: ManualEditTask,
    smoothing_parameters: SmoothingParameters,
) -> ManualAddPreparation:
    """Validate raw candidates first, then the boundaries that will be saved."""
    raw_geometries = tuple(QgsGeometry(item) for item in task.pending_geometries)
    _validate_geometries(raw_geometries, "raw")
    geometries = _geometries_for_commit(task, smoothing_parameters, raw_geometries)
    _validate_geometries(geometries, "saved")
    return ManualAddPreparation(geometries=geometries)


def record_modify_history(
    run_spec: Mapping[str, Any],
    result: manual_edit_commit.ManualModifyResult,
    source_code: int,
    target_code: int,
) -> None:
    """Append audit events after a successful modify commit."""
    for matched in result.matched:
        before_hash = class_workspace.geometry_hash(matched.original.geometry())
        after_hash = class_workspace.geometry_hash(matched.geometry)
        if matched.geometry_changed:
            class_workspace.append_history(
                run_spec,
                "geometry_modified",
                class_code=source_code,
                to_class_code=target_code,
                object_id=matched.object_id,
                before_geometry_hash=before_hash,
                after_geometry_hash=after_hash,
            )
        if target_code != source_code:
            class_workspace.append_history(
                run_spec,
                "feature_reclassified",
                object_id=matched.object_id,
                part_id=str(matched.original.attribute("part_id") or "000"),
                from_class_code=source_code,
                to_class_code=target_code,
                geometry_hash=after_hash,
            )
    for original in result.deleted:
        class_workspace.append_history(
            run_spec,
            "feature_deleted",
            class_code=source_code,
            object_id=str(original.attribute("object_id") or ""),
            before_geometry_hash=class_workspace.geometry_hash(original.geometry()),
            reason="manual_batch_replaced",
        )
    for added in result.added:
        class_workspace.append_history(
            run_spec,
            "feature_added",
            class_code=target_code,
            object_id=added.object_id,
            after_geometry_hash=class_workspace.geometry_hash(added.geometry),
            reason="manual_batch_added",
        )
    for warning in result.confidence_warnings:
        class_workspace.append_history(
            run_spec,
            "confidence_statistics_unavailable",
            class_code=warning.class_code,
            object_id=warning.object_id,
            reason=warning.reason,
        )


def record_add_history(
    run_spec: Mapping[str, Any],
    target_code: int,
    added: manual_edit_commit.AddedFeature,
    geometry: QgsGeometry,
    topology_hint: str,
) -> None:
    """Append audit events for one persisted addition and its topology hint."""
    class_workspace.append_history(
        run_spec,
        "feature_added",
        class_code=target_code,
        object_id=added.object_id,
        after_geometry_hash=class_workspace.geometry_hash(geometry),
        overlap_hint=topology_hint,
    )
    if added.confidence_warning:
        class_workspace.append_history(
            run_spec,
            "confidence_statistics_unavailable",
            class_code=target_code,
            object_id=added.object_id,
            reason=added.confidence_warning,
        )


def restart_edit_tracking(
    tracker: edit_tracking.EditTracker,
    layers: Sequence[tuple[int, QgsVectorLayer]],
) -> None:
    """Replace tracking baselines after a suppressed low-level commit."""
    for class_code, _layer in layers:
        tracker.discard(class_code)
    for class_code, layer in layers:
        if layer.isEditable():
            tracker.begin(class_code, layer)


def close_clean_task_editing_session(
    task: ManualEditTask,
    layer: QgsVectorLayer,
    tracker: edit_tracking.EditTracker,
) -> bool:
    """Close only the clean edit session opened by this manual task."""
    if (
        task.kind not in ("add", "modify")
        or not task.editing_started_by_task
        or not layer.isEditable()
        or layer.isModified()
    ):
        return False
    with tracker.suppress():
        layer.rollBack()
    tracker.discard(task.class_code)
    return True


def _geometries_for_commit(
    task: ManualEditTask,
    smoothing_parameters: SmoothingParameters,
    raw_geometries: tuple[QgsGeometry, ...],
) -> tuple[QgsGeometry, ...]:
    if not task.smoothing_enabled:
        return raw_geometries
    preview = task.smoothing_preview
    if (
        preview is None
        or preview.parameters != smoothing_parameters
        or len(preview.geometries) != len(task.pending_geometries)
        or tuple(geometry_source_hash(item) for item in task.pending_geometries)
        != preview.source_hashes
    ):
        raise RuntimeError("光滑预览尚未完成或已经失效，请等待自动预览")
    return tuple(QgsGeometry(item) for item in preview.geometries)


def _validate_geometries(
    geometries: Sequence[QgsGeometry],
    stage: Literal["raw", "saved"],
) -> None:
    for index, geometry in enumerate(geometries, start=1):
        error = validate_polygon_geometry(geometry)
        if error:
            raise InvalidManualGeometry(stage, index, error)
