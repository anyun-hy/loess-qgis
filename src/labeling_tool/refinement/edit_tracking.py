"""Track native QGIS geometry edits and persist their metadata."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from qgis.core import (
    QgsFeature,
    QgsFeatureRequest,
    QgsGeometry,
    QgsVectorDataProvider,
    QgsVectorLayer,
)

from labeling_tool.refinement import class_workspace
from labeling_tool.shared.contracts.run_spec import CLASS_NAMES

ConfidenceStatistics = Callable[
    [QgsVectorLayer, QgsGeometry], tuple[float | None, float | None, str]
]


@dataclass(frozen=True)
class FeatureSnapshot:
    object_id: str
    geometry_hash: str
    geometry_source: str
    geometry_revision: int
    immutable: Mapping[str, Any]


@dataclass(frozen=True)
class ConfidenceWarning:
    object_id: str
    reason: str


@dataclass(frozen=True)
class EditResult:
    """One handled commit; empty ID tuples mean a save or rollback had no delta."""

    changed_ids: tuple[int, ...] = ()
    added_ids: tuple[int, ...] = ()
    deleted_ids: tuple[int, ...] = ()
    affected_features: tuple[QgsFeature, ...] = ()
    confidence_warnings: tuple[ConfidenceWarning, ...] = ()


@dataclass(frozen=True)
class _EditContext:
    metadata_prepared: bool
    session_id: str


def snapshot_features(
    source: QgsVectorLayer | QgsVectorDataProvider,
    feature_ids: Iterable[int] | None = None,
) -> dict[int, FeatureSnapshot]:
    """Read tracking fields; ``None`` selects all and an empty iterable reads none."""
    request = QgsFeatureRequest()
    if feature_ids is not None:
        selected = list(feature_ids)
        if not selected:
            return {}
        request.setFilterFids(selected)
    return {
        feature.id(): FeatureSnapshot(
            object_id=str(feature.attribute("object_id") or ""),
            geometry_hash=class_workspace.geometry_hash(feature.geometry()),
            geometry_source=str(feature.attribute("geometry_source") or "fusion"),
            geometry_revision=int(feature.attribute("geometry_revision") or 0),
            immutable={
                name: feature.attribute(name)
                for name in class_workspace.IMMUTABLE_FIELDS
            },
        )
        for feature in source.getFeatures(request)
    }


def set_feature_attributes(
    layer: QgsVectorLayer,
    feature_id: int,
    values: Mapping[str, Any],
) -> None:
    """Write the supplied fields that exist on ``layer``."""
    for name, value in values.items():
        index = layer.fields().indexOf(name)
        if index >= 0:
            layer.changeAttributeValue(feature_id, index, value)


class EditTracker:
    """Own native-edit snapshots, commit IDs, SAM context, and suppression state."""

    def __init__(self) -> None:
        self._snapshots: dict[int, dict[int, FeatureSnapshot]] = {}
        self._committed_feature_ids: dict[int, set[int]] = {}
        self._contexts: dict[int, _EditContext] = {}
        self._suppression_depth = 0

    @property
    def suppressed(self) -> bool:
        return self._suppression_depth > 0

    @contextmanager
    def suppress(self) -> Iterator[None]:
        """Ignore synchronous edit signals, including nested commit signals."""
        self._suppression_depth += 1
        try:
            yield
        finally:
            self._suppression_depth = max(0, self._suppression_depth - 1)

    def has_session(self, class_code: int) -> bool:
        return int(class_code) in self._snapshots

    def reset(self) -> None:
        self._snapshots.clear()
        self._committed_feature_ids.clear()
        self._contexts.clear()
        self._suppression_depth = 0

    def discard(self, class_code: int) -> None:
        class_code = int(class_code)
        self._snapshots.pop(class_code, None)
        self._committed_feature_ids.pop(class_code, None)
        self._contexts.pop(class_code, None)

    def restore(
        self,
        class_code: int,
        layer: QgsVectorLayer,
        persisted_layer: QgsVectorLayer,
    ) -> None:
        """Restore the pre-edit baseline for a layer loaded while already editable."""
        feature_ids = (
            None if _uses_provider_transaction(layer) else _edited_existing_ids(layer)
        )
        self._snapshots[int(class_code)] = snapshot_features(
            persisted_layer, feature_ids
        )

    def begin(self, class_code: int, layer: QgsVectorLayer) -> bool:
        if self.suppressed:
            return False
        self._snapshots[int(class_code)] = _baseline_for_edit(layer)
        return True

    def prepare_edit(
        self,
        class_code: int,
        *,
        baseline: Mapping[int, FeatureSnapshot] | None = None,
        metadata_prepared: bool,
        session_id: str,
    ) -> None:
        """Attach SAM context; ``baseline=None`` preserves the active baseline."""
        class_code = int(class_code)
        if baseline is not None:
            self._snapshots[class_code] = dict(baseline)
        self._contexts[class_code] = _EditContext(
            metadata_prepared=bool(metadata_prepared),
            session_id=str(session_id),
        )

    def capture_before_commit(
        self,
        class_code: int,
        layer: QgsVectorLayer,
    ) -> None:
        if self.suppressed or _uses_provider_transaction(layer):
            return
        class_code = int(class_code)
        feature_ids = _edited_existing_ids(layer)
        before = snapshot_features(layer.dataProvider(), feature_ids)
        before.update(self._snapshots.get(class_code, {}))
        self._snapshots[class_code] = before
        self._committed_feature_ids[class_code] = set(feature_ids) | set(before)

    def record_committed_additions(
        self,
        class_code: int,
        features: Sequence[QgsFeature],
    ) -> None:
        if self.suppressed:
            return
        self._committed_feature_ids.setdefault(int(class_code), set()).update(
            feature.id() for feature in features
        )

    def finish(
        self,
        class_code: int,
        layer: QgsVectorLayer,
        *,
        run_spec: Mapping[str, Any],
        baseline_stream_id: str,
        confidence_statistics: ConfidenceStatistics,
    ) -> EditResult | None:
        """Persist one native-edit delta and return its read-only UI summary."""
        class_code = int(class_code)
        if self.suppressed or class_code not in self._snapshots:
            return None

        before = self._snapshots[class_code]
        affected_ids = self._committed_feature_ids.pop(class_code, set()) | set(before)
        if _uses_provider_transaction(layer):
            after_features = {feature.id(): feature for feature in layer.getFeatures()}
        else:
            request = QgsFeatureRequest().setFilterFids(list(affected_ids))
            after_features = (
                {feature.id(): feature for feature in layer.getFeatures(request)}
                if affected_ids
                else {}
            )

        before_ids = set(before)
        after_ids = set(after_features)
        deleted = tuple(sorted(before_ids - after_ids))
        added = tuple(sorted(after_ids - before_ids))
        changed = tuple(
            feature_id
            for feature_id in sorted(before_ids & after_ids)
            if class_workspace.geometry_hash(after_features[feature_id].geometry())
            != before[feature_id].geometry_hash
        )
        if not changed and not deleted and not added:
            self._contexts.pop(class_code, None)
            if layer.isEditable():
                self._snapshots[class_code] = _baseline_for_edit(layer)
            else:
                self.discard(class_code)
            return EditResult()

        context = self._contexts.pop(class_code, _EditContext(False, ""))
        keep_editing = layer.isEditable()
        warnings: list[ConfidenceWarning] = []
        with self.suppress():
            if not keep_editing and not layer.startEditing():
                raise RuntimeError("cannot start metadata update after geometry edit")
            for feature_id in changed:
                feature = after_features[feature_id]
                old = before[feature_id]
                confidence_mean, confidence_std, warning = confidence_statistics(
                    layer, feature.geometry()
                )
                if warning:
                    warnings.append(ConfidenceWarning(old.object_id, warning))
                for name, value in old.immutable.items():
                    layer.changeAttributeValue(
                        feature_id, layer.fields().indexOf(name), value
                    )
                set_feature_attributes(
                    layer,
                    feature_id,
                    {
                        "confidence_mean": confidence_mean,
                        "confidence_std": confidence_std,
                    },
                )
                if not context.metadata_prepared:
                    set_feature_attributes(
                        layer,
                        feature_id,
                        {
                            "geometry_source": "manual_edited",
                            "edit_base": old.geometry_source,
                            "geometry_revision": old.geometry_revision + 1,
                            "updated_at": class_workspace.workspace_timestamp(),
                        },
                    )
                class_workspace.append_history(
                    run_spec,
                    "geometry_modified",
                    class_code=class_code,
                    object_id=old.object_id,
                    before_geometry_hash=old.geometry_hash,
                    after_geometry_hash=class_workspace.geometry_hash(
                        feature.geometry()
                    ),
                )
            for feature_id in added:
                feature = after_features[feature_id]
                object_id = str(
                    feature.attribute("object_id")
                    or class_workspace.new_object_id(run_spec)
                )
                confidence_mean, confidence_std, warning = confidence_statistics(
                    layer, feature.geometry()
                )
                if warning:
                    warnings.append(ConfidenceWarning(object_id, warning))
                values: dict[str, Any] = {
                    "confidence_mean": confidence_mean,
                    "confidence_std": confidence_std,
                    "updated_at": class_workspace.workspace_timestamp(),
                }
                if not context.metadata_prepared:
                    values.update(
                        run_id=run_spec["run_id"],
                        object_id=object_id,
                        part_id="000",
                        class_code=class_code,
                        class_name=CLASS_NAMES[class_code],
                        baseline_stream_id=baseline_stream_id,
                        geometry_source="manual_edited",
                        geometry_revision=1,
                        edit_base="",
                        reviewed=0,
                    )
                set_feature_attributes(layer, feature_id, values)
                class_workspace.append_history(
                    run_spec,
                    "feature_added",
                    class_code=class_code,
                    object_id=object_id,
                    after_geometry_hash=class_workspace.geometry_hash(
                        feature.geometry()
                    ),
                )
            for feature_id in deleted:
                old = before[feature_id]
                class_workspace.append_history(
                    run_spec,
                    "feature_deleted",
                    class_code=class_code,
                    object_id=old.object_id,
                    before_geometry_hash=old.geometry_hash,
                )
            if not layer.commitChanges(not keep_editing):
                errors = "; ".join(layer.commitErrors())
                layer.rollBack()
                raise RuntimeError(f"cannot save edit metadata: {errors}")

        if layer.isEditable():
            self._snapshots[class_code] = _baseline_for_edit(layer)
        else:
            self.discard(class_code)
        for warning in warnings:
            class_workspace.append_history(
                run_spec,
                "confidence_statistics_unavailable",
                class_code=class_code,
                object_id=warning.object_id,
                reason=warning.reason,
            )
        affected_features = tuple(
            after_features[feature_id] for feature_id in changed + added
        )
        return EditResult(
            changed_ids=changed,
            added_ids=added,
            deleted_ids=deleted,
            affected_features=affected_features,
            confidence_warnings=tuple(warnings),
        )


def _uses_provider_transaction(layer: QgsVectorLayer) -> bool:
    return layer.dataProvider().transaction() is not None


def _baseline_for_edit(layer: QgsVectorLayer) -> dict[int, FeatureSnapshot]:
    # Pass-through providers can write before beforeCommitChanges and omit the
    # committedFeaturesAdded signal, so only they retain a full baseline.
    return snapshot_features(layer) if _uses_provider_transaction(layer) else {}


def _edited_existing_ids(layer: QgsVectorLayer) -> set[int]:
    edit_buffer = layer.editBuffer()
    if edit_buffer is None:
        return set()
    return (
        set(edit_buffer.changedGeometries())
        | set(edit_buffer.changedAttributeValues())
        | set(edit_buffer.deletedFeatureIds())
    ) - set(edit_buffer.addedFeatures())
