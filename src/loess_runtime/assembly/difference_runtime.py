"""Erase accepted-label coverage from one assembled semantic result stream."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import fiona
from fiona.crs import CRS
from rasterio.crs import CRS as RasterCRS
from rasterio.warp import transform_geom
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree


class DifferenceRuntimeError(RuntimeError):
    pass


def _parts(geometry):
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    return [item for item in getattr(geometry, "geoms", ()) if isinstance(item, Polygon)]


def apply_accepted_difference(
    source_path: str | Path,
    accepted_path: str | Path,
    output_path: str | Path,
    *,
    accepted_layer: str = "accepted_labels",
) -> dict[str, Any]:
    source_path = Path(source_path).resolve()
    accepted_path = Path(accepted_path).resolve()
    output_path = Path(output_path).resolve()
    if not accepted_path.is_file() or accepted_layer not in fiona.listlayers(accepted_path):
        return {"status": "skipped", "reason": "accepted layer is unavailable"}

    with fiona.open(source_path, layer="semantic_polygons") as source:
        source_crs = source.crs_wkt or source.crs
        schema = source.schema.copy()
        source_features = source
        with fiona.open(accepted_path, layer=accepted_layer) as accepted:
            accepted_crs = accepted.crs_wkt or accepted.crs
            accepted_geometries = [
                shape(feature["geometry"])
                for feature in accepted
                if feature.get("geometry")
            ]
        if not accepted_geometries:
            return {"status": "skipped", "reason": "accepted layer is empty"}
        if RasterCRS.from_user_input(accepted_crs) != RasterCRS.from_user_input(source_crs):
            accepted_geometries = [
                shape(
                    transform_geom(
                        accepted_crs,
                        source_crs,
                        mapping(geometry),
                        antimeridian_cutting=False,
                    )
                )
                for geometry in accepted_geometries
            ]
        tree = STRtree(accepted_geometries)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.parent / f".{output_path.stem}.{os.getpid()}.tmp.gpkg"
        temporary.unlink(missing_ok=True)
        source_count = 0
        output_count = 0
        clipped_count = 0
        try:
            with fiona.open(
                temporary,
                "w",
                driver="GPKG",
                layer="semantic_candidates",
                schema=schema,
                crs=CRS.from_user_input(source_crs),
            ) as destination:
                for feature in source_features:
                    source_count += 1
                    geometry = shape(feature["geometry"])
                    candidates = [
                        accepted_geometries[int(index)]
                        for index in tree.query(geometry)
                        if accepted_geometries[int(index)].intersects(geometry)
                    ]
                    difference = (
                        geometry.difference(unary_union(candidates))
                        if candidates else geometry
                    )
                    parts = _parts(difference)
                    if candidates:
                        clipped_count += 1
                    properties = dict(feature["properties"])
                    original_part = str(properties.get("part_id") or "part")
                    for index, part in enumerate(parts):
                        if part.is_empty or part.area <= 0:
                            continue
                        values = dict(properties)
                        values["part_id"] = (
                            original_part if len(parts) == 1 and not candidates
                            else f"{original_part}:d{index:03d}"
                        )
                        destination.write(
                            {
                                "geometry": mapping(MultiPolygon([part])),
                                "properties": values,
                            }
                        )
                        output_count += 1
            os.replace(temporary, output_path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    return {
        "status": "passed",
        "source_feature_count": source_count,
        "output_feature_count": output_count,
        "clipped_source_count": clipped_count,
        "output": str(output_path),
    }
