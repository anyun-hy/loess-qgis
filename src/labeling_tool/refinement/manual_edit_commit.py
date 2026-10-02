"""Data commits for manual class geometry edits."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from qgis.core import (
    QgsCoordinateTransform,
    QgsFeature,
    QgsFeatureRequest,
    QgsGeometry,
    QgsProject,
    QgsVectorLayer,
)

from labeling_tool.refinement import class_workspace
from labeling_tool.shared.contracts.run_spec import CLASS_NAMES

ConfidenceStatistics = Callable[
    [QgsVectorLayer, QgsGeometry], tuple[float | None, float | None, str]
]


@dataclass(frozen=True)
class ManualModifyPlan:
    matches: tuple[tuple[int, int, float], ...]
    unmatched_old: tuple[int, ...]
    unmatched_new: tuple[int, ...]


@dataclass(frozen=True)
class MatchedFeature:
    original: QgsFeature
    object_id: str
    geometry: QgsGeometry
    geometry_changed: bool


@dataclass(frozen=True)
class AddedFeature:
    object_id: str
    geometry: QgsGeometry
    confidence_warning: str = ""


@dataclass(frozen=True)
class ConfidenceWarning:
    class_code: int
    object_id: str
    reason: str


@dataclass(frozen=True)
class ManualModifyResult:
    matched: tuple[MatchedFeature, ...]
    added: tuple[AddedFeature, ...]
    deleted: tuple[QgsFeature, ...]
    confidence_warnings: tuple[ConfidenceWarning, ...]


@dataclass(frozen=True)
class ManualAddResult:
    added: tuple[AddedFeature, ...]


def plan_manual_modify_overlaps(
    old_features: Sequence[QgsFeature],
    new_geometries: Sequence[QgsGeometry],
) -> ManualModifyPlan:
    """Greedily match old and new polygons by descending intersection area."""
    overlaps = []
    for old_index, feature in enumerate(old_features):
        old_geometry = feature.geometry()
        for new_index, geometry in enumerate(new_geometries):
            if not old_geometry.boundingBox().intersects(geometry.boundingBox()):
                continue
            intersection = old_geometry.intersection(geometry)
            area = float(intersection.area()) if not intersection.isEmpty() else 0.0
            if math.isfinite(area) and area > 0.0:
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
    return ManualModifyPlan(
        matches=tuple(matches),
        unmatched_old=tuple(
            index for index in range(len(old_features)) if index not in matched_old
        ),
        unmatched_new=tuple(
            index for index in range(len(new_geometries)) if index not in matched_new
        ),
    )


def commit_manual_modify(
    *,
    source_layer: QgsVectorLayer,
    target_layer: QgsVectorLayer,
    old_features: Sequence[QgsFeature],
    plan: ManualModifyPlan,
    new_geometries: Sequence[QgsGeometry],
    source_code: int,
    target_code: int,
    run_spec: dict,
    baseline_stream_id: str,
    keep_source_editing: bool,
    confidence_statistics: ConfidenceStatistics,
) -> ManualModifyResult:
    """Commit one manual modification batch, including cross-layer compensation."""
    source_was_editable = source_layer.isEditable()
    target_was_editable = target_layer.isEditable()
    added_target_ids: list[str] = []
    matched: list[MatchedFeature] = []
    added: list[AddedFeature] = []
    deleted: list[QgsFeature] = []
    warnings: list[ConfidenceWarning] = []

    try:
        if not new_geometries:
            if target_code == source_code:
                raise RuntimeError("当前批次没有变化")
            prepared = []
            for original in old_features:
                geometry = QgsGeometry(original.geometry())
                moved, object_id, warning = _replacement_feature(
                    target_layer,
                    geometry,
                    target_code,
                    run_spec,
                    baseline_stream_id,
                    confidence_statistics,
                    original=original,
                    recalculate_confidence=False,
                )
                if _object_id_exists(target_layer, object_id):
                    raise RuntimeError(f"目标类别已存在 object_id: {object_id}")
                prepared.append((moved, original, object_id, warning, geometry))
            _start_editing(target_layer, "无法启动目标类别工作层编辑")
            for moved, _original, _object_id, _warning, _geometry in prepared:
                if not target_layer.addFeature(moved):
                    raise RuntimeError("无法向目标类别写入本批对象")
            _commit_layer(
                target_layer,
                not target_was_editable,
                "无法保存目标类别批次",
            )
            added_target_ids = [item[2] for item in prepared]
            _start_editing(source_layer, "无法启动来源类别工作层编辑")
            if not source_layer.deleteFeatures(
                [feature.id() for feature in old_features]
            ):
                raise RuntimeError("无法从来源类别删除本批旧面")
            _commit_layer(
                source_layer,
                not keep_source_editing,
                "无法保存来源类别批次",
            )
            for _moved, original, object_id, warning, geometry in prepared:
                matched.append(MatchedFeature(original, object_id, geometry, False))
                if warning:
                    warnings.append(ConfidenceWarning(target_code, object_id, warning))
        elif source_code == target_code:
            _start_editing(source_layer, "无法启动当前类别工作层编辑")
            for old_index, new_index, _area in plan.matches:
                original = old_features[old_index]
                geometry = new_geometries[new_index]
                if not source_layer.changeGeometry(original.id(), geometry):
                    raise RuntimeError("无法更新匹配旧面的 geometry")
                replacement, object_id, warning = _replacement_feature(
                    source_layer,
                    geometry,
                    target_code,
                    run_spec,
                    baseline_stream_id,
                    confidence_statistics,
                    original=original,
                )
                _set_attributes(source_layer, original.id(), replacement)
                geometry_changed = class_workspace.geometry_hash(
                    original.geometry()
                ) != class_workspace.geometry_hash(geometry)
                matched.append(
                    MatchedFeature(original, object_id, geometry, geometry_changed)
                )
                if warning:
                    warnings.append(ConfidenceWarning(target_code, object_id, warning))
            for old_index in plan.unmatched_old:
                original = old_features[old_index]
                if not source_layer.deleteFeature(original.id()):
                    raise RuntimeError("无法删除未被新边界保留的旧面")
                deleted.append(original)
            for new_index in plan.unmatched_new:
                geometry = new_geometries[new_index]
                feature, object_id, warning = _replacement_feature(
                    source_layer,
                    geometry,
                    target_code,
                    run_spec,
                    baseline_stream_id,
                    confidence_statistics,
                )
                if not source_layer.addFeature(feature):
                    raise RuntimeError("无法新增本批新面")
                added.append(AddedFeature(object_id, geometry, warning))
                if warning:
                    warnings.append(ConfidenceWarning(target_code, object_id, warning))
            _commit_layer(
                source_layer,
                not keep_source_editing,
                "无法保存本批修改",
            )
        else:
            prepared = []
            for old_index, new_index, _area in plan.matches:
                original = old_features[old_index]
                geometry = _target_geometry(
                    source_layer, target_layer, new_geometries[new_index]
                )
                moved, object_id, warning = _replacement_feature(
                    target_layer,
                    geometry,
                    target_code,
                    run_spec,
                    baseline_stream_id,
                    confidence_statistics,
                    original=original,
                )
                if _object_id_exists(target_layer, object_id):
                    raise RuntimeError(f"目标类别已存在 object_id: {object_id}")
                prepared.append((moved, original, object_id, warning, geometry, True))
            for new_index in plan.unmatched_new:
                geometry = _target_geometry(
                    source_layer, target_layer, new_geometries[new_index]
                )
                feature, object_id, warning = _replacement_feature(
                    target_layer,
                    geometry,
                    target_code,
                    run_spec,
                    baseline_stream_id,
                    confidence_statistics,
                )
                prepared.append((feature, None, object_id, warning, geometry, False))
            _start_editing(target_layer, "无法启动目标类别工作层编辑")
            for (
                feature,
                _original,
                _object_id,
                _warning,
                _geometry,
                _matched,
            ) in prepared:
                if not target_layer.addFeature(feature):
                    raise RuntimeError("无法向目标类别写入本批结果")
            _commit_layer(
                target_layer,
                not target_was_editable,
                "无法保存目标类别批次",
            )
            added_target_ids = [item[2] for item in prepared]
            _start_editing(source_layer, "无法启动来源类别工作层编辑")
            if not source_layer.deleteFeatures(
                [feature.id() for feature in old_features]
            ):
                raise RuntimeError("无法从来源类别删除本批旧面")
            _commit_layer(
                source_layer,
                not keep_source_editing,
                "无法保存来源类别批次",
            )
            matched_old_indexes = {item[0] for item in plan.matches}
            deleted.extend(
                original
                for old_index, original in enumerate(old_features)
                if old_index not in matched_old_indexes
            )
            for (
                _feature,
                original,
                object_id,
                warning,
                geometry,
                is_matched,
            ) in prepared:
                if is_matched:
                    matched.append(MatchedFeature(original, object_id, geometry, True))
                else:
                    added.append(AddedFeature(object_id, geometry, warning))
                if warning:
                    warnings.append(ConfidenceWarning(target_code, object_id, warning))
    except Exception as exc:
        if source_layer.isEditable() and source_layer.isModified():
            source_layer.rollBack()
        if (
            target_layer is not source_layer
            and target_layer.isEditable()
            and target_layer.isModified()
        ):
            target_layer.rollBack()
        if target_layer is not source_layer and added_target_ids:
            try:
                _remove_transferred_features(target_layer, added_target_ids)
            except Exception as rollback_exc:
                exc = RuntimeError(f"{exc}；且补偿回滚失败: {rollback_exc}")
        if source_was_editable and not source_layer.isEditable():
            source_layer.startEditing()
        if target_was_editable and not target_layer.isEditable():
            target_layer.startEditing()
        raise exc

    return ManualModifyResult(
        matched=tuple(matched),
        added=tuple(added),
        deleted=tuple(deleted),
        confidence_warnings=tuple(warnings),
    )


def commit_manual_add(
    *,
    layer: QgsVectorLayer,
    geometries: Sequence[QgsGeometry],
    target_code: int,
    run_spec: dict,
    baseline_stream_id: str,
    keep_editing: bool,
    confidence_statistics: ConfidenceStatistics,
) -> ManualAddResult:
    """Add and commit one batch while preserving the requested edit session."""
    prepared = []
    for geometry in geometries:
        feature, object_id, warning = _replacement_feature(
            layer,
            geometry,
            target_code,
            run_spec,
            baseline_stream_id,
            confidence_statistics,
        )
        prepared.append((feature, AddedFeature(object_id, geometry, warning)))

    try:
        _start_editing(layer, "无法启动目标类别工作层编辑")
        for feature, _added in prepared:
            if not layer.addFeature(feature):
                raise RuntimeError("无法向目标类别新增本批面")
        _commit_layer(layer, not keep_editing, "保存本批新增面失败")
    except Exception:
        if layer.isEditable():
            layer.rollBack()
        if keep_editing and not layer.isEditable():
            layer.startEditing()
        raise
    return ManualAddResult(added=tuple(added for _feature, added in prepared))


def commit_manual_delete(layer: QgsVectorLayer, feature_ids: Sequence[int]) -> int:
    """Delete and commit selected features, closing the layer edit session."""
    try:
        _start_editing(layer, "无法启动当前类别工作层编辑")
        if not layer.deleteFeatures(feature_ids):
            layer.rollBack()
            raise RuntimeError("无法删除选中的面")
        _commit_layer(layer, True, "保存删除失败")
    except Exception:
        if layer.isEditable():
            layer.rollBack()
        raise
    return len(feature_ids)


def _replacement_feature(
    layer: QgsVectorLayer,
    geometry: QgsGeometry,
    target_code: int,
    run_spec: dict,
    baseline_stream_id: str,
    confidence_statistics: ConfidenceStatistics,
    *,
    original: QgsFeature | None = None,
    recalculate_confidence: bool = True,
) -> tuple[QgsFeature, str, str]:
    feature = QgsFeature(layer.fields())
    feature.setGeometry(geometry)
    warning = ""
    if original is None:
        object_id = class_workspace.new_object_id(run_spec)
        values = {
            "run_id": run_spec["run_id"],
            "object_id": object_id,
            "part_id": "000",
            "class_code": target_code,
            "class_name": CLASS_NAMES[target_code],
            "baseline_stream_id": baseline_stream_id,
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
        values = {
            "run_id": run_spec["run_id"],
            "object_id": object_id,
            "part_id": str(original.attribute("part_id") or "000"),
            "class_code": target_code,
            "class_name": CLASS_NAMES[target_code],
            "baseline_stream_id": baseline_stream_id,
            "geometry_source": "manual_edited",
            "geometry_revision": int(original.attribute("geometry_revision") or 0) + 1,
            "edit_base": str(original.attribute("geometry_source") or "fusion"),
            "reviewed": 0,
        }
    if original is not None and not recalculate_confidence:
        confidence_mean = original.attribute("confidence_mean")
        confidence_std = original.attribute("confidence_std")
    else:
        confidence_mean, confidence_std, warning = confidence_statistics(
            layer, geometry
        )
    values.update(
        confidence_mean=confidence_mean,
        confidence_std=confidence_std,
        updated_at=class_workspace.workspace_timestamp(),
    )
    for name, value in values.items():
        if layer.fields().indexOf(name) >= 0:
            feature.setAttribute(name, value)
    return feature, object_id, warning


def _start_editing(layer: QgsVectorLayer, error: str) -> None:
    if not layer.isEditable() and not layer.startEditing():
        raise RuntimeError(error)


def _commit_layer(layer: QgsVectorLayer, stop_editing: bool, fallback: str) -> None:
    if not layer.commitChanges(stop_editing):
        errors = "; ".join(layer.commitErrors())
        raise RuntimeError(errors or fallback)


def _set_attributes(
    layer: QgsVectorLayer, feature_id: int, feature: QgsFeature
) -> None:
    for field in layer.fields():
        layer.changeAttributeValue(
            feature_id,
            layer.fields().indexOf(field.name()),
            feature.attribute(field.name()),
        )


def _object_id_exists(layer: QgsVectorLayer, object_id: str) -> bool:
    request = (
        QgsFeatureRequest()
        .setFilterExpression('"object_id" = \'' + object_id.replace("'", "''") + "'")
        .setLimit(1)
    )
    return any(
        str(feature.attribute("object_id") or "") == object_id
        for feature in layer.getFeatures(request)
    )


def _target_geometry(
    source_layer: QgsVectorLayer,
    target_layer: QgsVectorLayer,
    geometry: QgsGeometry,
) -> QgsGeometry:
    target_geometry = QgsGeometry(geometry)
    if source_layer.crs() != target_layer.crs():
        target_geometry.transform(
            QgsCoordinateTransform(
                source_layer.crs(), target_layer.crs(), QgsProject.instance()
            )
        )
    return target_geometry


def _remove_transferred_features(
    layer: QgsVectorLayer, object_ids: Sequence[str]
) -> None:
    wanted = set(str(object_id) for object_id in object_ids)
    feature_ids = [
        feature.id()
        for feature in layer.getFeatures()
        if str(feature.attribute("object_id") or "") in wanted
    ]
    if len(feature_ids) != len(wanted):
        raise RuntimeError("目标类别补偿回滚无法找到全部已写入对象")
    _start_editing(layer, "无法启动目标类别补偿回滚")
    if not layer.deleteFeatures(feature_ids):
        layer.rollBack()
        raise RuntimeError("无法删除目标类别中的批次补偿对象")
    if not layer.commitChanges():
        errors = "; ".join(layer.commitErrors())
        layer.rollBack()
        raise RuntimeError(f"目标类别批次补偿回滚失败: {errors}")
