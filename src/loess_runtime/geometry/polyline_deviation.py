"""Bound both directed distances between two piecewise-linear boundaries."""

from __future__ import annotations

import numpy as np
from shapely import STRtree, linestrings, points


def _point_distances(coordinates: np.ndarray, target: STRtree) -> np.ndarray:
    result = np.empty(len(coordinates), dtype=np.float64)
    for start in range(0, len(coordinates), 4096):
        stop = min(start + 4096, len(coordinates))
        indices, distances = target.query_nearest(
            points(coordinates[start:stop]),
            all_matches=False,
            return_distance=True,
        )
        result[start + indices[0]] = distances
    return result


def _directed_deviation(
    source: np.ndarray, target: np.ndarray, limit: float
) -> tuple[float, float, float]:
    # Index segments so each probe does not scan the entire opposing line.
    target_segments = STRtree(
        linestrings(np.stack((target[:-1], target[1:]), axis=1))
    )
    distances = _point_distances(source, target_segments)
    maximum = float(distances.max())
    total = float(distances.sum())
    count = len(distances)
    left, right = source[:-1], source[1:]
    left_distance, right_distance = distances[:-1], distances[1:]
    lengths = np.linalg.norm(right - left, axis=1)
    # Distance to a set is 1-Lipschitz along each line segment. For endpoint
    # distances a, b and segment length L, its maximum is <= (a + b + L) / 2.
    # Include a floating-point margin rather than accepting a rounded-down bound.
    margin = 16 * np.finfo(np.float64).eps * max(
        1.0, float(np.abs(source).max()), float(np.abs(target).max())
    )
    upper = np.maximum(
        (left_distance + right_distance + lengths) * 0.5,
        np.maximum(left_distance, right_distance),
    ) + margin
    certified = 0.0
    # Keep verification bounded even for a long segment lying exactly on the
    # limit. Unproved candidates are rejected by the existing strength search.
    sample_budget = max(16384, 8 * len(source))
    for _ in range(12):
        bound = max(certified, float(upper.max(initial=0.0)))
        if maximum > limit:
            return maximum, total / count, bound
        accepted = upper <= limit
        certified = max(certified, float(upper[accepted].max(initial=0.0)))
        pending = ~accepted
        if not np.any(pending):
            return maximum, total / count, certified
        left, right = left[pending], right[pending]
        left_distance = left_distance[pending]
        right_distance = right_distance[pending]
        lengths = lengths[pending]
        if count + len(left) > sample_budget:
            return maximum, total / count, bound
        middle = (left + right) * 0.5
        middle_distance = _point_distances(middle, target_segments)
        maximum = max(maximum, float(middle_distance.max()))
        total += float(middle_distance.sum())
        count += len(middle_distance)
        left, right = np.vstack((left, middle)), np.vstack((middle, right))
        left_distance, right_distance = (
            np.concatenate((left_distance, middle_distance)),
            np.concatenate((middle_distance, right_distance)),
        )
        lengths = np.tile(lengths * 0.5, 2)
        upper = np.maximum(
            (left_distance + right_distance + lengths) * 0.5,
            np.maximum(left_distance, right_distance),
        ) + margin
    return maximum, total / count, max(
        certified, float(upper.max(initial=0.0))
    )


def bounded_polyline_deviation(
    reference: np.ndarray, candidate: np.ndarray, limit: float
) -> tuple[float, float, float]:
    """Return sampled maximum/mean and a continuous Hausdorff upper bound.

    Acceptance requires the upper bound, not just the measured samples, to be
    <= limit. Samples are temporary line-segment probes; no dense spline or
    new output vertices are materialized by this check.
    """

    forward = _directed_deviation(reference, candidate, limit)
    if forward[2] > limit:
        return forward
    backward = _directed_deviation(candidate, reference, limit)
    return (
        max(forward[0], backward[0]),
        0.5 * (forward[1] + backward[1]),
        max(forward[2], backward[2]),
    )
