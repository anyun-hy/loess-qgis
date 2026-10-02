"""Native QGIS writes for class reviewed/reopened confirmation."""

from __future__ import annotations

from collections.abc import Callable

from qgis.core import QgsVectorLayer

from labeling_tool.refinement.edit_tracking import EditTracker


def commit_class_review(
    layer: QgsVectorLayer,
    *,
    reviewed: bool,
    now: Callable[[], str],
    tracker: EditTracker,
    keep_editing: bool,
) -> None:
    """Persist reviewed fields while preserving the caller's edit-session policy."""
    with tracker.suppress():
        if not keep_editing and not layer.startEditing():
            raise RuntimeError("无法更新 reviewed 字段")
        reviewed_index = layer.fields().indexOf("reviewed")
        updated_index = layer.fields().indexOf("updated_at")
        for feature in layer.getFeatures():
            layer.changeAttributeValue(
                feature.id(), reviewed_index, 1 if reviewed else 0
            )
            layer.changeAttributeValue(feature.id(), updated_index, now())
        if not layer.commitChanges(not keep_editing):
            errors = "; ".join(layer.commitErrors())
            layer.rollBack()
            raise RuntimeError(errors)
