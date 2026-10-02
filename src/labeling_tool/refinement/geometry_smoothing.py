"""Shared smoothing calculations for manual candidates and native selections.

The module accepts only ``QgsGeometry`` values.  Dialog code owns layer
identity, controls, timers, map overlays, and edit transactions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

from qgis.core import Qgis, QgsGeometry


@dataclass(frozen=True)
class SmoothingParameters:
    """Arguments passed to ``QgsGeometry.smooth`` for one preview batch."""

    iterations: int
    offset: float
    max_angle: float


@dataclass(frozen=True)
class SmoothingStatistics:
    """Aggregate vertex and planar-area values before and after smoothing."""

    source_vertex_count: int
    smoothed_vertex_count: int
    source_area: float
    smoothed_area: float

    @property
    def area_change_percent(self) -> float:
        """Return the signed total-area change, or zero for a zero-area input."""
        if self.source_area <= 0.0:
            return 0.0
        return ((self.smoothed_area - self.source_area) / self.source_area) * 100.0


@dataclass(frozen=True)
class SmoothingBatchResult:
    """Complete, validated smoothing output in the same order as its sources."""

    parameters: SmoothingParameters
    source_hashes: tuple[str, ...]
    geometries: tuple[QgsGeometry, ...]
    statistics: SmoothingStatistics


@dataclass(frozen=True)
class NativeSmoothingPreview:
    """Snapshot required to safely apply a preview to one editable QGIS layer."""

    layer_id: str
    class_code: int
    feature_ids: tuple[int, ...]
    batch: SmoothingBatchResult


class GeometrySmoothingError(RuntimeError):
    """One source geometry prevented a batch from producing any preview."""

    def __init__(self, index: int, reason: str) -> None:
        self.index = index
        self.reason = reason
        super().__init__(f"第 {index} 个面：{reason}")


def geometry_source_hash(geometry: QgsGeometry | None) -> str:
    """Return the stable WKB SHA-256 identity used by smoothing snapshots."""
    if geometry is None or geometry.isNull() or geometry.isEmpty():
        return ""
    return hashlib.sha256(bytes(geometry.asWkb())).hexdigest()


def geometry_vertex_count(geometry: QgsGeometry | None) -> int:
    """Count coordinates in a QGIS geometry without modifying it."""
    abstract = geometry.constGet() if geometry is not None else None
    return int(abstract.nCoordinates()) if abstract is not None else 0


def validate_polygon_geometry(geometry: QgsGeometry | None) -> str:
    """Return the user-facing reason a polygon cannot be accepted, if any."""
    if geometry is None or geometry.isNull() or geometry.isEmpty():
        return "几何为空"
    if geometry.type() != Qgis.GeometryType.Polygon:
        return "结果不是 Polygon"
    if geometry.area() <= 0:
        return "面积必须大于 0"
    if not geometry.isGeosValid():
        return "几何无效或存在自相交"
    return ""


def smooth_geometry_batch(
    geometries: Sequence[QgsGeometry],
    parameters: SmoothingParameters,
    *,
    convert_to_multi: bool = False,
) -> SmoothingBatchResult:
    """Smooth every source or raise an indexed error without returning a partial batch.

    Inputs are copied before calling QGIS.  ``convert_to_multi`` is reserved for
    manual candidates; selected native geometries retain their original WKB type.
    """
    smoothed_geometries: list[QgsGeometry] = []
    source_hashes: list[str] = []
    source_vertex_count = 0
    smoothed_vertex_count = 0
    source_area = 0.0
    smoothed_area = 0.0
    for index, geometry in enumerate(geometries, start=1):
        try:
            source = QgsGeometry(geometry)
            smoothed = source.smooth(
                parameters.iterations,
                parameters.offset,
                -1.0,
                parameters.max_angle,
            )
        except Exception as exc:
            raise GeometrySmoothingError(index, str(exc)) from exc
        if convert_to_multi:
            try:
                smoothed.convertToMultiType()
            except Exception as exc:
                raise GeometrySmoothingError(index, str(exc)) from exc
        error = validate_polygon_geometry(smoothed)
        if error:
            raise GeometrySmoothingError(index, error)
        smoothed_geometries.append(smoothed)
        source_hashes.append(geometry_source_hash(source))
        source_vertex_count += geometry_vertex_count(source)
        smoothed_vertex_count += geometry_vertex_count(smoothed)
        source_area += float(source.area())
        smoothed_area += float(smoothed.area())
    return SmoothingBatchResult(
        parameters=parameters,
        source_hashes=tuple(source_hashes),
        geometries=tuple(smoothed_geometries),
        statistics=SmoothingStatistics(
            source_vertex_count=source_vertex_count,
            smoothed_vertex_count=smoothed_vertex_count,
            source_area=source_area,
            smoothed_area=smoothed_area,
        ),
    )
