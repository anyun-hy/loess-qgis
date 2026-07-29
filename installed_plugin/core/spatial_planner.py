"""Deterministic pixel-space planning for partitioned raster-to-vector work."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence


MAX_LOCAL_TILES = 500_000


class SpatialPlanError(ValueError):
    pass


@dataclass(frozen=True)
class PixelWindow:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_list(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]


@dataclass(frozen=True)
class Partition:
    partition_id: str
    row: int
    col: int
    tile_row_start: int
    tile_row_stop: int
    tile_col_start: int
    tile_col_stop: int
    core_window: PixelWindow
    halo_window: PixelWindow


@dataclass(frozen=True)
class SpatialUnit:
    unit_id: str
    unit_type: str
    owner_key: str
    pixel_window: PixelWindow
    dependency_ids: tuple[str, ...]


def _split_boundaries(
    tile_count: int,
    partition_tiles: int,
    *,
    tile_size: int,
    stride: int,
) -> list[int]:
    total = tile_size + (tile_count - 1) * stride
    boundaries = [0]
    for first_tile_right in range(partition_tiles, tile_count, partition_tiles):
        left_center = (first_tile_right - 1) * stride + tile_size / 2.0
        right_center = first_tile_right * stride + tile_size / 2.0
        boundaries.append(int(math.floor((left_center + right_center) / 2.0 + 0.5)))
    boundaries.append(total)
    return boundaries


def _clamp_halo(window: PixelWindow, halo: int, width: int, height: int) -> PixelWindow:
    return PixelWindow(
        max(0, window.x0 - halo),
        max(0, window.y0 - halo),
        min(width, window.x1 + halo),
        min(height, window.y1 + halo),
    )


def _partition_id(row: int, col: int) -> str:
    return f"partition_{row:05d}_{col:05d}"


def _window_dict(window: PixelWindow) -> dict[str, int]:
    return {
        "x0": window.x0,
        "y0": window.y0,
        "x1": window.x1,
        "y1": window.y1,
    }


def plan_spatial_units(
    *,
    tile_rows: int,
    tile_cols: int,
    tile_size: int = 512,
    overlap: int = 192,
    partition_tile_rows: int = 8,
    partition_tile_cols: int = 8,
    seam_band_px: int = 64,
    halo_px: int | None = None,
) -> dict[str, Any]:
    """Plan mutually exclusive Core, Seam, and Junction pixel windows."""
    rows = int(tile_rows)
    cols = int(tile_cols)
    size = int(tile_size)
    overlap_value = int(overlap)
    part_rows = int(partition_tile_rows)
    part_cols = int(partition_tile_cols)
    seam = int(seam_band_px)
    if rows < 1 or cols < 1:
        raise SpatialPlanError("tile_rows and tile_cols must be positive")
    if rows * cols > MAX_LOCAL_TILES:
        raise SpatialPlanError(f"local plan cannot exceed {MAX_LOCAL_TILES} tiles")
    if size < 2 or overlap_value < 0 or overlap_value >= size:
        raise SpatialPlanError("tile overlap must satisfy 0 <= overlap < tile_size")
    if part_rows < 2 or part_cols < 2:
        raise SpatialPlanError("partition Core must be at least 2 x 2 tiles")
    if seam < 1:
        raise SpatialPlanError("seam_band_px must be positive")
    halo = max(overlap_value, seam) if halo_px is None else int(halo_px)
    if halo < max(overlap_value, seam):
        raise SpatialPlanError("halo_px must be at least max(overlap, seam_band_px)")

    stride = size - overlap_value
    x_boundaries = _split_boundaries(
        cols, part_cols, tile_size=size, stride=stride
    )
    y_boundaries = _split_boundaries(
        rows, part_rows, tile_size=size, stride=stride
    )
    width = x_boundaries[-1]
    height = y_boundaries[-1]
    for values, label in ((x_boundaries, "x"), (y_boundaries, "y")):
        if any(right - left <= 2 * seam for left, right in zip(values, values[1:])):
            raise SpatialPlanError(
                f"{label} partition Core is too narrow for the configured seam band"
            )

    partition_rows = len(y_boundaries) - 1
    partition_cols = len(x_boundaries) - 1
    partitions: list[Partition] = []
    units: list[SpatialUnit] = []

    for row in range(partition_rows):
        for col in range(partition_cols):
            core = PixelWindow(
                x_boundaries[col],
                y_boundaries[row],
                x_boundaries[col + 1],
                y_boundaries[row + 1],
            )
            partition_id = _partition_id(row, col)
            partitions.append(
                Partition(
                    partition_id=partition_id,
                    row=row,
                    col=col,
                    tile_row_start=row * part_rows,
                    tile_row_stop=min(rows, (row + 1) * part_rows),
                    tile_col_start=col * part_cols,
                    tile_col_stop=min(cols, (col + 1) * part_cols),
                    core_window=core,
                    halo_window=_clamp_halo(core, halo, width, height),
                )
            )
            interior = PixelWindow(
                core.x0 + (seam if col > 0 else 0),
                core.y0 + (seam if row > 0 else 0),
                core.x1 - (seam if col + 1 < partition_cols else 0),
                core.y1 - (seam if row + 1 < partition_rows else 0),
            )
            units.append(
                SpatialUnit(
                    unit_id=f"core_{row:05d}_{col:05d}",
                    unit_type="core",
                    owner_key=partition_id,
                    pixel_window=interior,
                    dependency_ids=(partition_id,),
                )
            )

    for row in range(partition_rows):
        y0 = y_boundaries[row] + (seam if row > 0 else 0)
        y1 = y_boundaries[row + 1] - (seam if row + 1 < partition_rows else 0)
        for boundary_col in range(1, partition_cols):
            x = x_boundaries[boundary_col]
            left = _partition_id(row, boundary_col - 1)
            right = _partition_id(row, boundary_col)
            units.append(
                SpatialUnit(
                    unit_id=f"seam_v_{row:05d}_{boundary_col:05d}",
                    unit_type="seam_vertical",
                    owner_key=f"{left}|{right}",
                    pixel_window=PixelWindow(x - seam, y0, x + seam, y1),
                    dependency_ids=(left, right),
                )
            )

    for boundary_row in range(1, partition_rows):
        y = y_boundaries[boundary_row]
        for col in range(partition_cols):
            x0 = x_boundaries[col] + (seam if col > 0 else 0)
            x1 = x_boundaries[col + 1] - (
                seam if col + 1 < partition_cols else 0
            )
            top = _partition_id(boundary_row - 1, col)
            bottom = _partition_id(boundary_row, col)
            units.append(
                SpatialUnit(
                    unit_id=f"seam_h_{boundary_row:05d}_{col:05d}",
                    unit_type="seam_horizontal",
                    owner_key=f"{top}|{bottom}",
                    pixel_window=PixelWindow(x0, y - seam, x1, y + seam),
                    dependency_ids=(top, bottom),
                )
            )

    for boundary_row in range(1, partition_rows):
        y = y_boundaries[boundary_row]
        for boundary_col in range(1, partition_cols):
            x = x_boundaries[boundary_col]
            dependencies = (
                _partition_id(boundary_row - 1, boundary_col - 1),
                _partition_id(boundary_row - 1, boundary_col),
                _partition_id(boundary_row, boundary_col - 1),
                _partition_id(boundary_row, boundary_col),
            )
            units.append(
                SpatialUnit(
                    unit_id=f"junction_{boundary_row:05d}_{boundary_col:05d}",
                    unit_type="junction",
                    owner_key="|".join(dependencies),
                    pixel_window=PixelWindow(x - seam, y - seam, x + seam, y + seam),
                    dependency_ids=dependencies,
                )
            )

    validate_spatial_ownership(PixelWindow(0, 0, width, height), units)
    unit_counts: dict[str, int] = {}
    for unit in units:
        unit_counts[unit.unit_type] = unit_counts.get(unit.unit_type, 0) + 1
    return {
        "schema_version": 1,
        "tile_rows": rows,
        "tile_cols": cols,
        "tile_count": rows * cols,
        "tile_size": size,
        "overlap": overlap_value,
        "stride": stride,
        "partition_tile_rows": part_rows,
        "partition_tile_cols": part_cols,
        "partition_rows": partition_rows,
        "partition_cols": partition_cols,
        "partition_count": len(partitions),
        "seam_band_px": seam,
        "halo_px": halo,
        "processing_window": _window_dict(PixelWindow(0, 0, width, height)),
        "x_boundaries": x_boundaries,
        "y_boundaries": y_boundaries,
        "unit_counts": unit_counts,
        "partitions": [
            {
                **{
                    key: value
                    for key, value in asdict(partition).items()
                    if key not in {"core_window", "halo_window"}
                },
                "core_window": _window_dict(partition.core_window),
                "halo_window": _window_dict(partition.halo_window),
            }
            for partition in partitions
        ],
        "spatial_units": [
            {
                "unit_id": unit.unit_id,
                "unit_type": unit.unit_type,
                "owner_key": unit.owner_key,
                "pixel_window": _window_dict(unit.pixel_window),
                "dependency_ids": list(unit.dependency_ids),
            }
            for unit in units
        ],
    }


def validate_spatial_ownership(
    processing_window: PixelWindow,
    units: Iterable[SpatialUnit],
) -> None:
    """Prove exact coverage and pairwise exclusivity by coordinate compression."""
    unit_values = list(units)
    if not unit_values:
        raise SpatialPlanError("spatial plan has no ownership units")
    xs = {processing_window.x0, processing_window.x1}
    ys = {processing_window.y0, processing_window.y1}
    for unit in unit_values:
        window = unit.pixel_window
        if window.width <= 0 or window.height <= 0:
            raise SpatialPlanError(f"spatial unit has an empty window: {unit.unit_id}")
        if not (
            processing_window.x0 <= window.x0 < window.x1 <= processing_window.x1
            and processing_window.y0 <= window.y0 < window.y1 <= processing_window.y1
        ):
            raise SpatialPlanError(f"spatial unit is outside processing extent: {unit.unit_id}")
        xs.update((window.x0, window.x1))
        ys.update((window.y0, window.y1))

    x_values = sorted(xs)
    y_values = sorted(ys)
    x_index = {value: index for index, value in enumerate(x_values)}
    y_index = {value: index for index, value in enumerate(y_values)}
    difference = [[0] * (len(x_values) + 1) for _ in range(len(y_values) + 1)]
    for unit in unit_values:
        window = unit.pixel_window
        x0 = x_index[window.x0]
        x1 = x_index[window.x1]
        y0 = y_index[window.y0]
        y1 = y_index[window.y1]
        difference[y0][x0] += 1
        difference[y0][x1] -= 1
        difference[y1][x0] -= 1
        difference[y1][x1] += 1

    for y in range(len(y_values)):
        for x in range(len(x_values)):
            above = difference[y - 1][x] if y else 0
            left = difference[y][x - 1] if x else 0
            diagonal = difference[y - 1][x - 1] if y and x else 0
            difference[y][x] += above + left - diagonal
            if y + 1 < len(y_values) and x + 1 < len(x_values):
                coverage = difference[y][x]
                if coverage != 1:
                    raise SpatialPlanError(
                        "spatial ownership must cover each compressed cell exactly once; "
                        f"cell=({x_values[x]},{y_values[y]}), coverage={coverage}"
                    )
