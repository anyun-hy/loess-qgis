"""Find exact side-sharing ownership units without pairwise geometry scans."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def ownership_neighbors(units: Iterable[Mapping[str, Any]]) -> list[tuple[str, str]]:
    left: dict[tuple[int, int, int], str] = {}
    right: dict[tuple[int, int, int], str] = {}
    top: dict[tuple[int, int, int], str] = {}
    bottom: dict[tuple[int, int, int], str] = {}
    for unit in units:
        unit_id = str(unit["unit_id"])
        window = unit["pixel_window"]
        x0, y0, x1, y1 = (int(window[key]) for key in ("x0", "y0", "x1", "y1"))
        left[(x0, y0, y1)] = unit_id
        right[(x1, y0, y1)] = unit_id
        top[(y0, x0, x1)] = unit_id
        bottom[(y1, x0, x1)] = unit_id
    pairs = {
        tuple(sorted((owner, left[key])))
        for key, owner in right.items()
        if key in left and owner != left[key]
    }
    pairs.update(
        tuple(sorted((owner, top[key])))
        for key, owner in bottom.items()
        if key in top and owner != top[key]
    )
    return sorted(pairs)
