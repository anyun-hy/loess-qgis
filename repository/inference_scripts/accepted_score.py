"""Create deterministic 14-class probabilities for fully accepted Tiles."""

from __future__ import annotations

from pathlib import Path

import fiona
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_bounds, transform_geom


CLASS_ORDER = (12, 13, 21, 31, 32, 33, 43, 51, 52, 53, 54, 61, 62, 71)


class AcceptedScoreError(RuntimeError):
    pass


def accepted_probabilities(
    accepted_gpkg: str | Path,
    tile_path: str | Path,
) -> np.ndarray:
    """Rasterize a fully covered Tile; uncovered pixels are a hard error."""
    accepted_path = Path(accepted_gpkg).expanduser().resolve()
    tile = Path(tile_path).expanduser().resolve()
    if not accepted_path.is_file():
        raise AcceptedScoreError(f"accepted GeoPackage is missing: {accepted_path}")
    layers = fiona.listlayers(accepted_path)
    layer_name = "accepted_labels" if "accepted_labels" in layers else ""
    if not layer_name:
        raise AcceptedScoreError("accepted GeoPackage has no accepted_labels layer")
    code_to_index = {code: index for index, code in enumerate(CLASS_ORDER)}
    with rasterio.open(tile) as raster:
        if raster.width != 512 or raster.height != 512 or raster.crs is None:
            raise AcceptedScoreError("accepted Tile must be a georeferenced 512 x 512 raster")
        tile_crs = raster.crs
        tile_transform = raster.transform
        tile_bounds = raster.bounds
    shapes = []
    with fiona.open(accepted_path, layer=layer_name) as source:
        source_crs = source.crs
        if not source_crs:
            raise AcceptedScoreError("accepted_labels has no CRS")
        query_bounds = transform_bounds(
            tile_crs,
            source_crs,
            *tile_bounds,
            densify_pts=21,
        )
        for feature in source.filter(bbox=query_bounds):
            geometry = feature.get("geometry")
            if not geometry:
                continue
            properties = feature.get("properties") or {}
            try:
                class_code = int(properties["class_code"])
                class_index = code_to_index[class_code]
            except (KeyError, TypeError, ValueError) as error:
                raise AcceptedScoreError(
                    "accepted_labels contains an invalid class_code"
                ) from error
            if source_crs != tile_crs:
                geometry = transform_geom(source_crs, tile_crs, geometry)
            shapes.append((geometry, class_index))
    if not shapes:
        raise AcceptedScoreError("accepted Tile has no intersecting accepted geometry")
    labels = rasterize(
        shapes,
        out_shape=(512, 512),
        transform=tile_transform,
        fill=-1,
        dtype="int16",
    )
    uncovered = int(np.count_nonzero(labels < 0))
    if uncovered:
        raise AcceptedScoreError(
            f"Tile marked fully accepted still has {uncovered} uncovered pixels"
        )
    probabilities = np.zeros((14, 512, 512), dtype=np.float32)
    np.put_along_axis(probabilities, labels[None, :, :], 1.0, axis=0)
    return probabilities
