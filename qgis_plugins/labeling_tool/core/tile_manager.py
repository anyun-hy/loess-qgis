import json
import logging
import math
import os
import shutil

from qgis.core import (
    Qgis,
    QgsTask,
    QgsFeature,
    QgsGeometry,
    QgsRectangle,
    QgsRasterLayer,
    QgsRasterPipe,
    QgsRasterFileWriter,
    QgsRasterDataProvider,
    QgsCoordinateTransform,
    QgsProject,
    QgsCoordinateReferenceSystem,
    QgsSpatialIndex,
)
from qgis.PyQt.QtCore import QDir
import processing

logger = logging.getLogger(__name__)


def raster_grid_info(raster_layer):
    """Copy the raster values needed by a worker without retaining the layer."""
    if raster_layer is None or not raster_layer.isValid():
        raise ValueError("请选择有效的本地影像层")
    extent = raster_layer.extent()
    return {
        "extent": (
            extent.xMinimum(),
            extent.yMinimum(),
            extent.xMaximum(),
            extent.yMaximum(),
        ),
        "width": int(raster_layer.width()),
        "height": int(raster_layer.height()),
        "res_x": abs(float(raster_layer.rasterUnitsPerPixelX())),
        "res_y": abs(float(raster_layer.rasterUnitsPerPixelY())),
    }


def generate_grid(
    extent,
    tile_width=512,
    tile_height=512,
    overlap=64,
    raster_layer=None,
    raster_info=None,
):
    if abs(extent.xMaximum() - extent.xMinimum()) < 1e-12 or tile_width < 1 or tile_height < 1:
        return []

    if overlap < 0 or overlap >= min(tile_width, tile_height):
        raise ValueError(
            f"切片重叠必须在 0 到 {min(tile_width, tile_height) - 1} 像素之间"
        )

    if raster_info is not None:
        res_x = abs(float(raster_info["res_x"]))
        res_y = abs(float(raster_info["res_y"]))
    elif raster_layer and raster_layer.isValid():
        res_x = abs(raster_layer.rasterUnitsPerPixelX())
        res_y = abs(raster_layer.rasterUnitsPerPixelY())
    else:
        res_x = abs(extent.xMaximum() - extent.xMinimum()) / max(tile_width, 1)
        res_y = abs(extent.yMaximum() - extent.yMinimum()) / max(tile_height, 1)

    if res_x <= 0 or res_y <= 0:
        return []

    tile_w = tile_width * res_x
    tile_h = tile_height * res_y

    effective_step_w = tile_width - overlap
    effective_step_h = tile_height - overlap
    step_x = effective_step_w * res_x
    step_y = effective_step_h * res_y

    xmin = extent.xMinimum()
    ymin = extent.yMinimum()
    xmax = extent.xMaximum()
    ymax = extent.yMaximum()

    if raster_info is not None or (raster_layer and raster_layer.isValid()):
        if raster_info is not None:
            raster_extent = QgsRectangle(*raster_info["extent"])
            raster_width = int(raster_info["width"])
            raster_height = int(raster_info["height"])
        else:
            raster_extent = raster_layer.extent()
            raster_width = int(raster_layer.width())
            raster_height = int(raster_layer.height())
        if raster_width < tile_width or raster_height < tile_height:
            raise ValueError(
                f"影像尺寸 {raster_width}x{raster_height} 小于切片尺寸 "
                f"{tile_width}x{tile_height}"
            )

        # Snap the requested range to source pixels first. This prevents GDAL
        # from producing an off-by-one tile when the drawn rectangle is not
        # exactly aligned with the raster pixel grid.
        left_px = max(0, int(math.floor(
            (xmin - raster_extent.xMinimum()) / res_x + 1e-9
        )))
        right_px = min(raster_width, int(math.ceil(
            (xmax - raster_extent.xMinimum()) / res_x - 1e-9
        )))
        top_px = max(0, int(math.floor(
            (raster_extent.yMaximum() - ymax) / res_y + 1e-9
        )))
        bottom_px = min(raster_height, int(math.ceil(
            (raster_extent.yMaximum() - ymin) / res_y - 1e-9
        )))

        requested_w = max(1, right_px - left_px)
        requested_h = max(1, bottom_px - top_px)
        ncols = max(1, math.ceil(max(0, requested_w - tile_width) / effective_step_w) + 1)
        nrows = max(1, math.ceil(max(0, requested_h - tile_height) / effective_step_h) + 1)
        grid_w_px = tile_width + (ncols - 1) * effective_step_w
        grid_h_px = tile_height + (nrows - 1) * effective_step_h

        if grid_w_px > raster_width or grid_h_px > raster_height:
            raise ValueError(
                "当前范围无法在影像内部扩展成规则的完整切片网格。"
                "请稍微缩小绘制范围，或调整切片尺寸和重叠像素。"
            )

        # Expand from the requested top-left, then shift the complete grid back
        # inside the source raster when the selection touches an image edge.
        start_col = min(left_px, raster_width - grid_w_px)
        start_row = min(top_px, raster_height - grid_h_px)
        xmin = raster_extent.xMinimum() + start_col * res_x
        ymax = raster_extent.yMaximum() - start_row * res_y
    else:
        extent_w = xmax - xmin
        extent_h = ymax - ymin
        ncols = max(1, math.ceil(max(0.0, extent_w - tile_w) / step_x) + 1)
        nrows = max(1, math.ceil(max(0.0, extent_h - tile_h) / step_y) + 1)

    tiles = []
    for row in range(nrows):
        for col in range(ncols):
            t_xmin = xmin + col * step_x
            t_ymax = ymax - row * step_y
            t_xmax = t_xmin + tile_w
            t_ymin = t_ymax - tile_h

            bounds = QgsRectangle(t_xmin, t_ymin, t_xmax, t_ymax)
            tiles.append({
                "row": row,
                "col": col,
                "bounds": bounds,
                "width": tile_width,
                "height": tile_height,
            })

    return tiles


def get_grid_extent(tiles):
    """Return the actual full-tile processing extent for a generated grid."""
    if not tiles:
        return None
    return QgsRectangle(
        min(tile["bounds"].xMinimum() for tile in tiles),
        min(tile["bounds"].yMinimum() for tile in tiles),
        max(tile["bounds"].xMaximum() for tile in tiles),
        max(tile["bounds"].yMaximum() for tile in tiles),
    )


def snapshot_vector_geometries(vector_layer, target_crs):
    """Copy valid polygon geometries into the raster CRS on the main thread."""
    if vector_layer is None or not vector_layer.isValid():
        raise ValueError("请选择有效的已加载矢量面图层")
    if vector_layer.geometryType() != Qgis.GeometryType.Polygon:
        raise ValueError("矢量范围必须是面图层")
    if vector_layer.featureCount() < 1:
        raise ValueError("矢量范围图层没有面要素")
    if not vector_layer.crs().isValid() or not target_crs or not target_crs.isValid():
        raise ValueError("矢量范围或影像缺少有效 CRS")

    transform = None
    if vector_layer.crs() != target_crs:
        transform = QgsCoordinateTransform(
            vector_layer.crs(), target_crs, QgsProject.instance()
        )

    geometries = []
    for source_feature in vector_layer.getFeatures():
        geometry = QgsGeometry(source_feature.geometry())
        if geometry.isNull() or geometry.isEmpty():
            continue
        if not geometry.isGeosValid():
            raise ValueError(
                f"矢量范围图层存在无效面，feature id={source_feature.id()}"
            )
        if transform is not None:
            try:
                geometry.transform(transform)
            except Exception as exc:
                raise ValueError(
                    f"矢量范围 CRS 转换失败，feature id={source_feature.id()}: {exc}"
                ) from exc
        geometries.append(geometry)

    if not geometries:
        raise ValueError("矢量范围图层没有可用的面几何")
    return geometries


def select_tiles_intersecting_geometries(
    tiles,
    geometries,
    *,
    is_canceled=None,
    progress=None,
):
    """Select full Tiles against detached geometries, suitable for QgsTask."""
    index = QgsSpatialIndex()
    geometry_by_id = {}
    engines = {}
    for feature_id, geometry in enumerate(geometries):
        feature = QgsFeature(feature_id)
        feature.setGeometry(geometry)
        index.addFeature(feature)
        geometry_by_id[feature_id] = geometry
        engine = QgsGeometry.createGeometryEngine(geometry.constGet())
        engine.prepareGeometry()
        engines[feature_id] = engine

    selected = []
    total = len(tiles)
    for position, tile in enumerate(tiles):
        if position % 256 == 0:
            if is_canceled is not None and is_canceled():
                return None
            if progress is not None:
                progress(position, total)
        bounds = tile["bounds"]
        tile_geometry = QgsGeometry.fromRect(bounds)
        keep = False
        for feature_id in index.intersects(bounds):
            geometry = geometry_by_id.get(feature_id)
            engine = engines.get(feature_id)
            if geometry is None or engine is None:
                continue
            tile_const = tile_geometry.constGet()
            if engine.contains(tile_const):
                keep = True
                break
            if not engine.intersects(tile_const):
                continue
            intersection = geometry.intersection(tile_geometry)
            if (
                intersection is not None
                and not intersection.isNull()
                and not intersection.isEmpty()
                and intersection.area() > 0.0
            ):
                keep = True
                break
        if keep:
            tile["range_selected"] = True
            selected.append(tile)
    if progress is not None:
        progress(total, total)
    return selected


def select_tiles_intersecting_vector(tiles, vector_layer, target_crs):
    """Keep complete Tiles whose area overlaps any polygon in a loaded layer."""
    geometries = snapshot_vector_geometries(vector_layer, target_crs)
    return select_tiles_intersecting_geometries(tiles, geometries)


class VectorTileSelectionTask(QgsTask):
    """Generate and filter a large vector range grid outside the GUI thread."""

    def __init__(
        self,
        request_key,
        extent,
        tile_width,
        tile_height,
        overlap,
        raster_info,
        geometries,
    ):
        super().__init__("计算矢量范围 Tile", QgsTask.Flag.CanCancel)
        self.request_key = request_key
        self.extent = QgsRectangle(extent)
        self.tile_width = int(tile_width)
        self.tile_height = int(tile_height)
        self.overlap = int(overlap)
        self.raster_info = dict(raster_info)
        self.geometries = [QgsGeometry(geometry) for geometry in geometries]
        self.result_data = None
        self.error_message = ""

    def run(self):
        try:
            grid_tiles = generate_grid(
                self.extent,
                self.tile_width,
                self.tile_height,
                self.overlap,
                raster_info=self.raster_info,
            )
            selected_tiles = select_tiles_intersecting_geometries(
                grid_tiles,
                self.geometries,
                is_canceled=self.isCanceled,
                progress=self._set_selection_progress,
            )
            if selected_tiles is None or self.isCanceled():
                return False
            if not selected_tiles:
                raise ValueError("矢量范围没有选中任何完整 Tile")
            processing_extent = get_grid_extent(grid_tiles)
            self.result_data = {
                "key": self.request_key,
                "grid_tiles": grid_tiles,
                "selected_tiles": selected_tiles,
                "processing_extent": processing_extent,
                "rows": max(int(tile["row"]) for tile in grid_tiles) + 1,
                "cols": max(int(tile["col"]) for tile in grid_tiles) + 1,
                "grid_count": len(grid_tiles),
                "selected_count": len(selected_tiles),
            }
            return True
        except Exception as exc:
            self.error_message = str(exc)
            return False

    def _set_selection_progress(self, current, total):
        if total > 0:
            self.setProgress(100.0 * float(current) / float(total))


def build_extract_parameters(raster_input, grid_cell, output_dir, crs_authid):
    """Build GDAL clip parameters without retaining a QGIS layer object."""
    os.makedirs(output_dir, exist_ok=True)
    row = grid_cell["row"]
    col = grid_cell["col"]
    bounds = grid_cell["bounds"]
    tile_path = os.path.join(output_dir, f"tile_{row}_{col}.tif")
    extent_str = "{},{},{},{} [{}]".format(
        bounds.xMinimum(),
        bounds.xMaximum(),
        bounds.yMinimum(),
        bounds.yMaximum(),
        crs_authid,
    )
    return {
        "INPUT": raster_input,
        "PROJWIN": extent_str,
        "OUTPUT": tile_path,
    }, tile_path


def finalize_extracted_tile(output_path, grid_cell, output_dir, crs_authid):
    """Validate an extracted tile and write its sidecar metadata."""
    row = grid_cell["row"]
    col = grid_cell["col"]
    bounds = grid_cell["bounds"]
    width = grid_cell.get("width", 512)
    height = grid_cell.get("height", 512)
    tile_path = output_path or os.path.join(output_dir, f"tile_{row}_{col}.tif")
    meta_path = os.path.join(output_dir, f"tile_{row}_{col}_meta.json")

    if not os.path.exists(tile_path):
        raise RuntimeError(
            f"Failed to extract tile ({row}, {col}); output was not created: {tile_path}"
        )

    extracted = QgsRasterLayer(tile_path, f"tile_{row}_{col}_validation")
    if not extracted.isValid():
        raise RuntimeError(f"切片 ({row}, {col}) 不是有效栅格: {tile_path}")
    if extracted.width() != width or extracted.height() != height:
        raise RuntimeError(
            f"切片 ({row}, {col}) 实际尺寸为 "
            f"{extracted.width()}x{extracted.height()}，预期为 {width}x{height}。"
            "自动扩展范围没有与源影像像元正确对齐。"
        )

    res_x = (bounds.xMaximum() - bounds.xMinimum()) / width
    res_y = (bounds.yMaximum() - bounds.yMinimum()) / height
    meta = {
        "row": row,
        "col": col,
        "bounds": {
            "xmin": bounds.xMinimum(),
            "ymin": bounds.yMinimum(),
            "xmax": bounds.xMaximum(),
            "ymax": bounds.yMaximum(),
        },
        "crs": crs_authid,
        "geotransform": [
            bounds.xMinimum(),
            res_x,
            0.0,
            bounds.yMaximum(),
            0.0,
            -res_y,
        ],
        "width": width,
        "height": height,
        "tile_path": tile_path,
    }
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    except IOError as exc:
        logger.error("Failed to write metadata for tile (%d, %d): %s", row, col, exc)
        raise

    return {"tile_path": tile_path, "meta_path": meta_path, "meta": meta}


def extract_tile(raster_layer, grid_cell, output_dir):
    row = grid_cell["row"]
    col = grid_cell["col"]
    crs_authid = raster_layer.crs().authid()
    params, tile_path = build_extract_parameters(
        raster_layer, grid_cell, output_dir, crs_authid
    )

    try:
        result = processing.run("gdal:cliprasterbyextent", params)
    except Exception as e:
        logger.error("Failed to extract tile (%d, %d): %s", row, col, e)
        raise

    output_path = result.get("OUTPUT", tile_path) if isinstance(result, dict) else tile_path
    return finalize_extracted_tile(
        output_path, grid_cell, output_dir, crs_authid
    )


def get_tile_paths(output_dir, row, col):
    return {
        "tile": os.path.join(output_dir, f"tile_{row}_{col}.tif"),
        "meta": os.path.join(output_dir, f"tile_{row}_{col}_meta.json"),
        "mask": os.path.join(output_dir, f"tile_{row}_{col}_mask.tif"),
        "confidence": os.path.join(output_dir, f"tile_{row}_{col}_conf.tif"),
    }


def cleanup_temp_files(temp_dir):
    if not os.path.isdir(temp_dir):
        logger.debug("Temp dir %s does not exist, nothing to clean", temp_dir)
        return
    try:
        shutil.rmtree(temp_dir)
        logger.info("Removed temp directory: %s", temp_dir)
    except OSError as e:
        logger.warning("Failed to remove temp dir %s: %s", temp_dir, e)
