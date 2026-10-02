"""Pure state owner for one interactive SAM3 refinement session."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:
    from qgis.core import QgsGeometry


SamMode = Literal["existing", "missed"]
SamState = Literal[
    "waiting_click",
    "inference",
    "candidate",
    "failed",
    "cancelled",
]
SamDecision = Literal[
    "adopted",
    "edit_sam3",
    "edit_current",
    "kept_current",
    "failed",
    "cancelled",
]


@dataclass
class SamSession:
    """Own one session; the dialog owns QGIS resources and persistence.

    Clicks and crop bounds use the raster CRS; candidate geometry uses the
    working layer CRS. The dialog supplies a validated, transformed candidate.
    """

    session_id: str
    class_code: int
    mode: SamMode
    state: SamState
    started_at: str
    feature_id: int | None = None
    object_id: str | None = None
    part_id: str | None = None
    current_geometry_hash: str = ""
    current_source: str = ""
    current_revision: int = 0
    click_raster: dict[str, float] | None = None
    geometry_bounds: dict[str, float] | None = None
    candidate_geometry: QgsGeometry | None = None
    candidate_score: float | None = None
    confidence_mean: float | None = None
    confidence_std: float | None = None
    crop_window: object | None = None
    elapsed_sec: object | None = None
    topology_hint: str = ""
    error: str = ""
    persisted_geometry_hash: str = ""
    _pending_request: dict[str, object] | None = field(
        default=None, init=False, repr=False
    )

    @classmethod
    def start(cls, class_code: int, mode: SamMode, started_at: str) -> SamSession:
        return cls(uuid.uuid4().hex, class_code, mode, "waiting_click", started_at)

    def begin_inference(
        self,
        *,
        click_raster: dict[str, float],
        geometry_bounds: dict[str, float] | None,
        feature_id: int | None = None,
        object_id: str = "",
        part_id: str = "000",
        current_geometry_hash: str = "",
        current_source: str = "",
        current_revision: int = 0,
    ) -> bool:
        if self.state != "waiting_click":
            return False
        self.feature_id = feature_id
        self.object_id = object_id
        self.part_id = part_id
        self.current_geometry_hash = current_geometry_hash
        self.current_source = current_source
        self.current_revision = current_revision
        self.click_raster = dict(click_raster)
        self.geometry_bounds = (
            dict(geometry_bounds) if geometry_bounds is not None else None
        )
        self.state = "inference"
        return True

    def queue_request(
        self,
        *,
        run_id: str,
        raster: str,
        confidence_mosaic: str,
        crop_size_px: int,
        buffer_px: int,
        checkpoint_sha256: str,
        sam_version: str,
        device: str,
    ) -> None:
        if self.state != "inference" or self.click_raster is None:
            raise RuntimeError("SAM3 request requires an inference session")
        self._pending_request = {
            "session_id": self.session_id,
            "run_id": run_id,
            "raster": raster,
            "confidence_mosaic": confidence_mosaic,
            "click_raster": dict(self.click_raster),
            "geometry_bounds": (
                dict(self.geometry_bounds) if self.geometry_bounds is not None else None
            ),
            "crop_size_px": crop_size_px,
            "buffer_px": buffer_px,
            "class_code": self.class_code,
            "object_id": self.object_id or "",
            "part_id": self.part_id or "000",
            "checkpoint_sha256": checkpoint_sha256,
            "sam_version": sam_version,
            "device": device,
        }

    def take_pending_request(self) -> dict[str, object] | None:
        if self.state != "inference":
            self._pending_request = None
            return None
        request = self._pending_request
        self._pending_request = None
        if request is None or request.get("session_id") != self.session_id:
            return None
        return request

    def clear_pending_request(self) -> None:
        self._pending_request = None

    def accept_candidate(
        self,
        *,
        session_id: str,
        geometry: QgsGeometry,
        score: float,
        confidence_mean: float,
        confidence_std: float,
        crop_window: object | None,
        elapsed_sec: object | None,
    ) -> bool:
        if not session_id or session_id != self.session_id or self.state != "inference":
            return False
        self.clear_pending_request()
        self.candidate_geometry = geometry
        self.candidate_score = score
        self.confidence_mean = confidence_mean
        self.confidence_std = confidence_std
        self.crop_window = crop_window
        self.elapsed_sec = elapsed_sec
        self.state = "candidate"
        return True

    def fail(self, error: str, *, session_id: str | None = None) -> bool:
        if session_id is not None and session_id != self.session_id:
            return False
        if self.state not in ("waiting_click", "inference", "candidate"):
            return False
        self.clear_pending_request()
        self.state = "failed"
        self.error = error
        return True

    def retry(self, started_at: str) -> str:
        if self.state != "failed" or self.click_raster is None:
            raise RuntimeError("only a failed SAM3 inference can be retried")
        old_session_id = self.session_id
        self.session_id = uuid.uuid4().hex
        self.started_at = started_at
        self.state = "inference"
        self.error = ""
        self.persisted_geometry_hash = ""
        self._clear_candidate()
        self.clear_pending_request()
        return old_session_id

    def cancel(self) -> None:
        self.clear_pending_request()
        self._clear_candidate()
        self.state = "cancelled"

    def history_record(
        self,
        decision: SamDecision,
        *,
        run_id: str | None,
        baseline_stream_id: str,
        checkpoint_sha256: str,
        sam_version: str,
        device: str,
        geometry_hash: Callable[[QgsGeometry], str],
    ) -> dict[str, object]:
        before_hash = self.current_geometry_hash
        candidate_geometry_hash = (
            geometry_hash(self.candidate_geometry)
            if self.candidate_geometry is not None
            else ""
        )
        record: dict[str, object] = {
            "session_id": self.session_id,
            "class_code": self.class_code,
            "mode": self.mode,
            "state": self.state,
            "started_at": self.started_at,
        }
        if self.click_raster is not None:
            record.update(
                {
                    "object_id": self.object_id or "",
                    "part_id": self.part_id or "000",
                    "click_raster": dict(self.click_raster),
                    "geometry_bounds": (
                        dict(self.geometry_bounds)
                        if self.geometry_bounds is not None
                        else None
                    ),
                }
            )
            if self.feature_id is not None:
                record["feature_id"] = self.feature_id
            if self.mode == "existing":
                record.update(
                    {
                        "current_geometry_hash": before_hash,
                        "current_source": self.current_source,
                        "current_revision": self.current_revision,
                    }
                )
        if self.candidate_geometry is not None:
            record.update(
                {
                    "candidate_score": self.candidate_score,
                    "confidence_mean": self.confidence_mean,
                    "confidence_std": self.confidence_std,
                    "crop_window": self.crop_window,
                    "elapsed_sec": self.elapsed_sec,
                    "topology_hint": self.topology_hint,
                }
            )
        if self.error:
            record["error"] = self.error
        if self.persisted_geometry_hash:
            record["persisted_geometry_hash"] = self.persisted_geometry_hash
        record.update(
            {
                "decision": decision,
                "run_id": run_id,
                "baseline_stream_id": baseline_stream_id,
                "before_geometry_hash": before_hash,
                "candidate_geometry_hash": candidate_geometry_hash,
                "after_geometry_hash": (
                    self.persisted_geometry_hash
                    if decision == "adopted"
                    else candidate_geometry_hash
                    if decision == "edit_sam3"
                    else before_hash
                ),
                "checkpoint_sha256": checkpoint_sha256,
                "sam_version": sam_version,
                "device": device,
            }
        )
        return record

    def _clear_candidate(self) -> None:
        self.candidate_geometry = None
        self.candidate_score = None
        self.confidence_mean = None
        self.confidence_std = None
        self.crop_window = None
        self.elapsed_sec = None
        self.topology_hint = ""
