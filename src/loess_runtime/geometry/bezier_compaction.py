"""Remove redundant joins without relaxing the spline's geometric bounds."""

from __future__ import annotations

import numpy as np


def signed_chord_area(points: np.ndarray) -> float:
    """Area enclosed by a divider and its endpoint chord, in local coordinates."""

    local = points - points[0]
    return float(np.sum(
        local[:-1, 0] * local[1:, 1] - local[1:, 0] * local[:-1, 1]
    )) * 0.5


def compact_bezier_spans(
    controls: np.ndarray,
    *,
    max_chord_error: float,
    max_segment_arc_length: float,
    area_change: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Merge adjacent certified spans when one chord can represent them.

    Each Bézier curve lies in its control hull. Bounding every control point
    against the proposed chord therefore bounds all intervening curves, not
    just the join vertices. Summed control-polygon lengths bound their arc.
    Failed groups are halved; accepted source spans are never subdivided here.
    When supplied, area_change is the uncompressed divider's signed area
    transfer from its source. Each merge must stay within that same budget.
    """

    arc_bounds = np.linalg.norm(np.diff(controls, axis=1), axis=2).sum(axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(arc_bounds)))
    points = [controls[0, 0]]
    maximum_error = maximum_arc = 0.0
    area_limit = None if area_change is None else abs(area_change) + 1e-9
    start = 0
    while start < len(controls):
        stop = max(
            start + 1,
            int(np.searchsorted(
                cumulative, cumulative[start] + max_segment_arc_length,
                side="right",
            )) - 1,
        )
        while True:
            group = controls[start:stop]
            offsets = group.reshape(-1, 2) - group[0, 0]
            chord = group[-1, -1] - group[0, 0]
            squared = float(chord @ chord)
            if squared > 1e-24:
                fractions = np.clip((offsets @ chord) / squared, 0.0, 1.0)
                offsets = offsets - fractions[:, None] * chord
            error = float(np.linalg.norm(offsets, axis=1).max())
            # Sum the selected lengths directly: subtracting large prefixes
            # can understate a small interval's bound.
            arc = float(arc_bounds[start:stop].sum())
            removed_area = signed_chord_area(
                np.concatenate((group[0, :1], group[:, -1]), axis=0)
            ) if area_change is not None else 0.0
            area_allowed = (
                area_limit is None
                or abs(area_change - removed_area) <= area_limit
            )
            if stop == start + 1 or (
                error <= max_chord_error and arc <= max_segment_arc_length
                and area_allowed
            ):
                break
            stop = start + max(1, (stop - start) // 2)
        points.append(controls[stop - 1, -1])
        maximum_error = max(maximum_error, error)
        maximum_arc = max(maximum_arc, arc)
        if area_change is not None:
            area_change -= removed_area
        start = stop
    return np.asarray(points), maximum_error, maximum_arc
