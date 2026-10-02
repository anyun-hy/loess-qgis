"""Shared spatial contracts for the V3.3 execution path."""

from __future__ import annotations

import math
from typing import Any, Mapping

from affine import Affine  # type: ignore[import-untyped]
from rasterio.windows import Window  # type: ignore[import-untyped]

from loess_runtime.geometry.small_component_regularizer import physical_pixel_area_m2

__all__ = [
    "FragmentationV33WorkPackageError",
    "expand_core_window",
    "intersect_windows",
    "normalize_window",
    "physical_metrics",
    "raster_window",
    "window_shape",
    "window_slices",
]


class FragmentationV33WorkPackageError(RuntimeError):
    """Raised when the frozen V3.3 execution contract is violated."""


def normalize_window(value: Mapping[str, Any]) -> dict[str, int]:
    return {key: int(value[key]) for key in ("x0", "y0", "x1", "y1")}


def intersect_windows(
    first: Mapping[str, int], second: Mapping[str, int]
) -> dict[str, int] | None:
    result = {
        "x0": max(int(first["x0"]), int(second["x0"])),
        "y0": max(int(first["y0"]), int(second["y0"])),
        "x1": min(int(first["x1"]), int(second["x1"])),
        "y1": min(int(first["y1"]), int(second["y1"])),
    }
    if result["x0"] >= result["x1"] or result["y0"] >= result["y1"]:
        return None
    return result


def window_shape(value: Mapping[str, int]) -> tuple[int, int]:
    return int(value["y1"]) - int(value["y0"]), int(value["x1"]) - int(value["x0"])


def window_slices(
    parent: Mapping[str, int], child: Mapping[str, int]
) -> tuple[slice, slice]:
    return (
        slice(
            int(child["y0"]) - int(parent["y0"]),
            int(child["y1"]) - int(parent["y0"]),
        ),
        slice(
            int(child["x0"]) - int(parent["x0"]),
            int(child["x1"]) - int(parent["x0"]),
        ),
    )


def expand_core_window(
    core: Mapping[str, int], global_window: Mapping[str, int], margin: int
) -> dict[str, int]:
    return {
        "x0": max(int(global_window["x0"]), int(core["x0"]) - int(margin)),
        "y0": max(int(global_window["y0"]), int(core["y0"]) - int(margin)),
        "x1": min(int(global_window["x1"]), int(core["x1"]) + int(margin)),
        "y1": min(int(global_window["y1"]), int(core["y1"]) + int(margin)),
    }


def raster_window(parent: Mapping[str, int], child: Mapping[str, int]) -> Window:
    return Window(
        col_off=int(child["x0"]) - int(parent["x0"]),
        row_off=int(child["y0"]) - int(parent["y0"]),
        width=int(child["x1"]) - int(child["x0"]),
        height=int(child["y1"]) - int(child["y0"]),
    )


def physical_metrics(
    transform: Affine, crs: str, window: Mapping[str, int]
) -> dict[str, float]:
    local = transform * Affine.translation(int(window["x0"]), int(window["y0"]))
    height, width = window_shape(window)
    area = float(physical_pixel_area_m2(local, crs, height=height, width=width))
    determinant = abs(local.a * local.e - local.b * local.d)
    if determinant <= 0:
        raise FragmentationV33WorkPackageError("processing transform has zero area")
    scale = math.sqrt(area / determinant)
    row_m = math.hypot(local.b, local.e) * scale
    return {
        "pixel_area_m2": area,
        "row_step_m": row_m,
        "column_step_m": area / row_m,
    }
