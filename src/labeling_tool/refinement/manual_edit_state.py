"""Transient state for one manual edit session; QGIS layers stay with the dialog."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from qgis.core import QgsGeometry

    from labeling_tool.refinement.geometry_smoothing import SmoothingBatchResult


ManualKind = Literal["modify", "delete", "add"]
ManualState = Literal[
    "selecting",
    "capturing",
    "capture_transition",
    "candidate",
    "capture_cancelled",
    "paused",
    "committing",
    "failed",
]


@dataclass
class ManualEditTask:
    """Own selection copies, candidates and counters until the session ends."""

    kind: ManualKind
    class_code: int
    target_code: int
    state: ManualState
    selection_before: list[int] = field(default_factory=list)
    selected_feature_ids: list[int] = field(default_factory=list)
    selected_count: int = 0
    pending_geometries: list[QgsGeometry] = field(default_factory=list)
    pending_errors: list[str] = field(default_factory=list)
    smoothing_enabled: bool = False
    smoothing_preview: SmoothingBatchResult | None = None
    smoothing_error: str = ""
    submitted_batch_count: int = 0
    modified_old_count: int = 0
    saved_new_count: int = 0
    deleted_old_count: int = 0
    added_count: int = 0
    saved_counts: dict[int, int] = field(default_factory=dict)
    editing_started_by_task: bool = False
    resume_state: Literal["selecting", "capturing"] | None = None
    error: str = ""

    @classmethod
    def for_modify(cls, class_code: int, selected_ids: list[int]) -> ManualEditTask:
        """Start modification with independent copies of the QGIS selection."""
        return cls(
            "modify",
            class_code,
            class_code,
            "selecting",
            selection_before=list(selected_ids),
            selected_feature_ids=list(selected_ids),
        )

    @classmethod
    def for_delete(cls, class_code: int, selected_ids: list[int]) -> ManualEditTask:
        """Start deletion while retaining the initial QGIS selection for cancel."""
        return cls(
            "delete",
            class_code,
            class_code,
            "selecting",
            selection_before=list(selected_ids),
            selected_count=len(selected_ids),
        )

    @classmethod
    def for_add(cls, class_code: int, selected_ids: list[int]) -> ManualEditTask:
        """Start capture while retaining the initial QGIS selection for cancel."""
        return cls(
            "add",
            class_code,
            class_code,
            "capturing",
            selection_before=list(selected_ids),
        )

    def toggle_selected_feature(self, feature_id: int) -> int:
        """Toggle a modify candidate and return the resulting selection count."""
        if feature_id in self.selected_feature_ids:
            self.selected_feature_ids.remove(feature_id)
        else:
            self.selected_feature_ids.append(feature_id)
        return len(self.selected_feature_ids)

    def append_candidate(self, geometry: QgsGeometry, error: str) -> int:
        """Keep a captured geometry and its validation result in matching order."""
        self.pending_geometries.append(geometry)
        self.pending_errors.append(error)
        self.invalidate_smoothing()
        self.state = "candidate" if error else "capture_transition"
        return len(self.pending_geometries)

    def retry_candidate(self) -> None:
        """Discard only the last candidate before restarting capture."""
        if self.pending_geometries:
            self.pending_geometries.pop()
        if self.pending_errors:
            self.pending_errors.pop()
        self.error = ""
        self.invalidate_smoothing()

    def pause(self) -> bool:
        """Pause a map tool; a capture transition resumes as capturing."""
        if self.state not in ("selecting", "capturing", "capture_transition"):
            return False
        self.resume_state = (
            "capturing" if self.state == "capture_transition" else self.state
        )
        self.state = "paused"
        return True

    def resume(self) -> Literal["selecting", "capturing"]:
        """Consume the saved map mode exactly once after a pause."""
        resumed = self.resume_state or "selecting"
        self.resume_state = None
        self.state = resumed
        return resumed

    def begin_capture(self) -> None:
        """Clear pause metadata when a new capture tool is active."""
        self.state = "capturing"
        self.resume_state = None

    def capture_cancelled(self) -> None:
        self.state = "capture_cancelled"

    def set_failed(self, message: str) -> None:
        self.state = "failed"
        self.error = message

    def invalidate_smoothing(self) -> None:
        self.smoothing_preview = None
        self.smoothing_error = ""

    def record_modify_batch(self, matched: int, added: int, deleted: int) -> None:
        """Record a committed batch and reset candidates for the next selection."""
        self.submitted_batch_count += 1
        self.modified_old_count += matched
        self.saved_new_count += added
        self.deleted_old_count += deleted
        self.selected_feature_ids.clear()
        self.pending_geometries.clear()
        self.pending_errors.clear()
        self.invalidate_smoothing()
        self.target_code = self.class_code
        self.state = "selecting"
        self.error = ""

    def record_add_batch(self, target_code: int, added: int) -> None:
        """Record saved additions and reset capture for the next batch."""
        self.added_count += added
        self.submitted_batch_count += 1
        self.saved_counts[target_code] = self.saved_counts.get(target_code, 0) + added
        self.pending_geometries.clear()
        self.pending_errors.clear()
        self.invalidate_smoothing()
        self.target_code = self.class_code
        self.state = "capturing"
        self.error = ""
