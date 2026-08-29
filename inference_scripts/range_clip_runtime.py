"""Apply adaptive range boundary clipping to assembled semantic polygon streams."""

from __future__ import annotations

import hashlib
import os
import shutil
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parts(geometry: Any) -> Sequence[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    return [item for item in getattr(geometry, "geoms", ()) if isinstance(item, Polygon)]


def _requested_extent_geometry(
    spec: Mapping[str, Any],
    target_crs: Any | None,
) -> Any | None:
    extent_spec = spec.get("requested_extent")
    if not isinstance(extent_spec, Mapping):
        return None
    try:
        xmin = float(extent_spec["xmin"])
        ymin = float(extent_spec["ymin"])
        xmax = float(extent_spec["xmax"])
        ymax = float(extent_spec["ymax"])
    except (KeyError, TypeError, ValueError) as error:
        raise RangeClipRuntimeError("requested extent is invalid") from error
    if xmin >= xmax or ymin >= ymax:
        raise RangeClipRuntimeError("requested extent is empty")

    geometry = box(xmin, ymin, xmax, ymax)
    extent_crs_value = (
        extent_spec.get("crs")
        or (spec.get("raster") or {}).get("crs")
        or target_crs
    )
    if target_crs is not None and extent_crs_value:
        extent_crs = RasterCRS.from_user_input(extent_crs_value)
        resolved_target = RasterCRS.from_user_input(target_crs)
        if extent_crs != resolved_target:
            geometry = shape(
                transform_geom(
                    extent_crs,
                    resolved_target,
                    mapping(geometry),
                )
            )
    return geometry


def extract_range_mask_geometry(
    spec: Mapping[str, Any],
    target_crs_wkt: str | None,
) -> Any | None:
    """Return the exact requested range in ``target_crs_wkt``.

    ``vector_tile_intersection`` uses Tiles only to schedule inference.  Its
    vector is the publication boundary and therefore cannot degrade to the
    request extent if it is missing or cannot be transformed.  Older run specs
    that have no mode but explicitly provide ``range_vector_path`` retain the
    former vector-boundary behaviour.
    """
    range_selection = spec.get("range_selection") or {}
    mode_value = range_selection.get("mode")
    mode = str(mode_value or "").strip()
    vector_path_raw = (
        spec.get("range_vector_path")
        or range_selection.get("vector_source")
        or range_selection.get("vector_path")
    )

    if mode == "extent":
        vector_path_raw = ""
    elif mode not in {"", "vector_tile_intersection"}:
        raise RangeClipRuntimeError(f"unsupported range selection mode: {mode}")

    vector_required = mode == "vector_tile_intersection" or (
        not mode and bool(vector_path_raw)
    )
    if vector_required:
        if not vector_path_raw:
            raise RangeClipRuntimeError("cannot read required vector range: vector source is missing")
        # QGIS sources can identify a GeoPackage layer as
        # ``/path/range.gpkg|layername=range``.
        clean_path_str = str(vector_path_raw).split("|")[0].strip()
        vector_path = Path(clean_path_str).expanduser().resolve()
        if not vector_path.is_file():
            raise RangeClipRuntimeError(
                f"cannot read required vector range: file does not exist: {vector_path}"
            )
        run_dir_value = str(spec.get("run_dir") or "")
        if mode == "vector_tile_intersection" and run_dir_value:
            try:
                vector_path.relative_to(Path(run_dir_value).expanduser().resolve())
            except ValueError as error:
                raise RangeClipRuntimeError(
                    "cannot read required vector range: snapshot is outside the Run directory"
                ) from error
        expected_sha256 = str(
            range_selection.get("vector_sha256")
            or spec.get("range_vector_sha256")
            or ""
        )
        if mode == "vector_tile_intersection":
            if not expected_sha256:
                raise RangeClipRuntimeError(
                    "cannot read required vector range: frozen snapshot SHA256 is missing"
                )
            if _sha256_file(vector_path) != expected_sha256:
                raise RangeClipRuntimeError(
                    "cannot read required vector range: frozen snapshot SHA256 changed"
                )
        try:
            layer_name = None
            if "|" in str(vector_path_raw):
                for part in str(vector_path_raw).split("|")[1:]:
                    if part.startswith("layername="):
                        layer_name = part.split("=", 1)[1]
            with fiona.open(vector_path, layer=layer_name) as src:
                src_crs = src.crs_wkt or src.crs
                if not src_crs:
                    raise RangeClipRuntimeError(
                        "cannot read required vector range: source CRS is missing"
                    )
                source_crs = RasterCRS.from_user_input(src_crs)
                geoms = [
                    geometry
                    for feature in src
                    if feature.get("geometry")
                    for geometry in _parts(shape(feature["geometry"]))
                    if not geometry.is_empty
                ]
                if not geoms:
                    raise RangeClipRuntimeError(
                        "cannot read required vector range: no polygon geometry found"
                    )
                union_geom = unary_union(geoms)
                if union_geom.is_empty:
                    raise RangeClipRuntimeError(
                        "cannot read required vector range: polygon geometry is empty"
                    )
                target_crs = (
                    RasterCRS.from_user_input(target_crs_wkt)
                    if target_crs_wkt
                    else source_crs
                )
                if source_crs != target_crs:
                    transformed = transform_geom(
                        source_crs,
                        target_crs,
                        mapping(union_geom),
                    )
                    union_geom = shape(transformed)
                if mode == "vector_tile_intersection":
                    extent_geometry = _requested_extent_geometry(spec, target_crs)
                    if extent_geometry is not None:
                        union_geom = union_geom.intersection(extent_geometry)
                    if union_geom.is_empty or not _parts(union_geom):
                        raise RangeClipRuntimeError(
                            "required vector range does not overlap the requested raster extent"
                        )
                return union_geom
        except RangeClipRuntimeError:
            raise
        except Exception as err:
            raise RangeClipRuntimeError(
                f"cannot read required vector range: {err}"
            ) from err

    return _requested_extent_geometry(spec, target_crs_wkt)


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

    # Report-resume verifies the formal GPKG fingerprint before completing.  A
    # second identical clip must therefore be a true no-op, not an equivalent
    # rewrite with a different SQLite/GPKG byte layout.
    with fiona.open(source_path, layer=target_layer) as source:
        source_crs = source.crs_wkt or source.crs
        mask_geometry = extract_range_mask_geometry(spec, source_crs)
        if mask_geometry is None or mask_geometry.is_empty:
            return {"status": "skipped", "reason": "no valid range mask geometry found in spec"}
        if all(
            feature.get("geometry")
            and mask_geometry.covers(shape(feature["geometry"]))
            for feature in source
        ):
            if output_path != source_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = output_path.parent / f".{output_path.name}.copy_{os.getpid()}.tmp"
                temporary.unlink(missing_ok=True)
                try:
                    shutil.copy2(source_path, temporary)
                    os.replace(temporary, output_path)
                except Exception:
                    temporary.unlink(missing_ok=True)
                    raise
            return {
                "status": "already_clipped",
                "source_feature_count": len(source),
                "output": str(output_path),
            }

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
