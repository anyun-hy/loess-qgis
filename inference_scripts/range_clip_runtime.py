"""Apply adaptive range boundary clipping to assembled semantic polygon streams."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import fiona
from fiona.crs import CRS
from rasterio.crs import CRS as RasterCRS
from rasterio.warp import transform_geom
from shapely.geometry import MultiPolygon, Polygon, box, mapping, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree


class RangeClipRuntimeError(RuntimeError):
    pass


def _parts(geometry: Any) -> Sequence[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    return [item for item in getattr(geometry, "geoms", ()) if isinstance(item, Polygon)]


def extract_range_mask_geometry(
    spec: Mapping[str, Any],
    target_crs_wkt: str | None,
) -> Any | None:
    """Extract and reproject the range mask geometry from spec (vector file or bbox extent)."""
    range_selection = spec.get("range_selection") or {}
    vector_path_raw = (
        spec.get("range_vector_path")
        or range_selection.get("vector_source")
        or range_selection.get("vector_path")
    )
    if vector_path_raw:
        # If QGIS passes a source like /path/to/file.gpkg|layername=foo, strip layername
        clean_path_str = str(vector_path_raw).split("|")[0].strip()
        vector_path = Path(clean_path_str).resolve()
        if vector_path.is_file():
            try:
                layer_name = None
                if "|" in str(vector_path_raw):
                    for part in str(vector_path_raw).split("|")[1:]:
                        if part.startswith("layername="):
                            layer_name = part.split("=")[1]
                with fiona.open(vector_path, layer=layer_name) as src:
                    src_crs = src.crs_wkt or src.crs
                    geoms = [
                        shape(feature["geometry"])
                        for feature in src
                        if feature.get("geometry")
                    ]
                    if geoms:
                        union_geom = unary_union(geoms)
                        if target_crs_wkt and src_crs:
                            if RasterCRS.from_user_input(src_crs) != RasterCRS.from_user_input(target_crs_wkt):
                                transformed = transform_geom(
                                    src_crs,
                                    target_crs_wkt,
                                    mapping(union_geom),
                                    antimeridian_cutting=False,
                                )
                                return shape(transformed)
                        return union_geom
            except Exception as err:
                # Fall back to requested_extent if vector reading fails
                pass

    extent_spec = spec.get("requested_extent")
    if extent_spec and isinstance(extent_spec, Mapping):
        try:
            xmin = float(extent_spec["xmin"])
            ymin = float(extent_spec["ymin"])
            xmax = float(extent_spec["xmax"])
            ymax = float(extent_spec["ymax"])
            if xmin < xmax and ymin < ymax:
                bbox_geom = box(xmin, ymin, xmax, ymax)
                extent_crs = extent_spec.get("crs")
                if target_crs_wkt and extent_crs:
                    if RasterCRS.from_user_input(extent_crs) != RasterCRS.from_user_input(target_crs_wkt):
                        transformed = transform_geom(
                            extent_crs,
                            target_crs_wkt,
                            mapping(bbox_geom),
                            antimeridian_cutting=False,
                        )
                        return shape(transformed)
                return bbox_geom
        except (KeyError, ValueError, TypeError):
            pass

    return None


def apply_adaptive_range_clip(
    source_path: str | Path,
    spec: Mapping[str, Any],
    output_path: str | Path | None = None,
    *,
    source_layer: str | None = None,
) -> dict[str, Any]:
    """Clips the given polygon dataset to the user requested range (vector mask or extent)."""
    source_path = Path(source_path).resolve()
    if not source_path.is_file():
        return {"status": "skipped", "reason": "source file does not exist"}

    if output_path is None:
        output_path = source_path
    else:
        output_path = Path(output_path).resolve()

    available_layers = fiona.listlayers(source_path)
    if not available_layers:
        return {"status": "skipped", "reason": "source file contains no layers"}

    target_layer = source_layer if source_layer in available_layers else available_layers[0]

    with fiona.open(source_path, layer=target_layer) as source:
        source_crs = source.crs_wkt or source.crs
        schema = source.schema.copy()
        
        mask_geometry = extract_range_mask_geometry(spec, source_crs)
        if mask_geometry is None or mask_geometry.is_empty:
            return {"status": "skipped", "reason": "no valid range mask geometry found in spec"}

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.parent / f".{output_path.stem}.clip_{os.getpid()}.tmp.gpkg"
        temporary.unlink(missing_ok=True)

        source_count = 0
        output_count = 0
        trimmed_count = 0
        discarded_count = 0

        try:
            with fiona.open(
                temporary,
                "w",
                driver="GPKG",
                layer=target_layer,
                schema=schema,
                crs=CRS.from_user_input(source_crs),
            ) as destination:
                for feature in source:
                    source_count += 1
                    geom = shape(feature["geometry"])
                    if not geom.intersects(mask_geometry):
                        discarded_count += 1
                        continue

                    if mask_geometry.contains(geom):
                        destination.write(feature)
                        output_count += 1
                        continue

                    # Boundary intersection
                    clipped_geom = geom.intersection(mask_geometry)
                    parts = _parts(clipped_geom)
                    if not parts:
                        discarded_count += 1
                        continue

                    trimmed_count += 1
                    properties = dict(feature["properties"])
                    original_part = str(properties.get("part_id") or "part")

                    for index, part in enumerate(parts):
                        if part.is_empty or part.area <= 0:
                            continue
                        values = dict(properties)
                        if len(parts) > 1:
                            values["part_id"] = f"{original_part}:c{index:03d}"
                        
                        target_geom_type = schema.get("geometry", "MultiPolygon")
                        if "Multi" in target_geom_type:
                            geom_to_write = MultiPolygon([part]) if isinstance(part, Polygon) else part
                        else:
                            geom_to_write = part

                        destination.write(
                            {
                                "geometry": mapping(geom_to_write),
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
        "trimmed_feature_count": trimmed_count,
        "discarded_feature_count": discarded_count,
        "output": str(output_path),
    }
