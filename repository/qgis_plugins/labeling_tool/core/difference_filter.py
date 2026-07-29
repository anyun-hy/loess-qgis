import logging
import os

from qgis.core import (
    Qgis,
    QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry,
    QgsRectangle, QgsCoordinateReferenceSystem, QgsField,
    QgsWkbTypes,
    QgsFeatureRequest, QgsCoordinateTransform,
    QgsVectorFileWriter,
)
from qgis.PyQt.QtCore import QObject, QVariant

from .layer_names import LAYER_NAMES
from .qgis_writer import write_vector_layer

logger = logging.getLogger("labeling_tool.difference_filter")


def snapshot_accepted_layer(accepted_layer, output_path):
    if not accepted_layer or not accepted_layer.isValid():
        raise ValueError("accepted layer is invalid")
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.layerName = LAYER_NAMES.ACCEPTED
    options.actionOnExistingFile = (
        QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteFile
    )
    error, message = write_vector_layer(accepted_layer, output_path, options)
    if error != QgsVectorFileWriter.WriterError.NoError:
        raise RuntimeError(message or f"failed to snapshot accepted_labels: {error}")
    return str(output_path)


def tile_is_fully_accepted(
    tile_bounds: QgsRectangle, accepted_layer: QgsVectorLayer, tile_crs=None
) -> bool:
    if not accepted_layer or not accepted_layer.isValid():
        return False
    if accepted_layer.featureCount() == 0:
        return False

    tile_polygon = QgsGeometry.fromRect(tile_bounds)
    if tile_crs is not None and tile_crs.isValid() and tile_crs != accepted_layer.crs():
        transform = QgsCoordinateTransform(
            tile_crs, accepted_layer.crs(), QgsProject.instance()
        )
        tile_polygon.transform(transform)
    candidates = []
    request = QgsFeatureRequest().setFilterRect(tile_polygon.boundingBox())
    for feature in accepted_layer.getFeatures(request):
        geom = feature.geometry()
        if geom and not geom.isNull() and not geom.isEmpty() and tile_polygon.intersects(geom):
            candidates.append(geom)
    if not candidates:
        return False
    uncovered = tile_polygon.difference(QgsGeometry.unaryUnion(candidates))
    return uncovered is None or uncovered.isNull() or uncovered.isEmpty()


def filter_difference(
    candidate_layer: QgsVectorLayer, accepted_layer: QgsVectorLayer,
    target_crs: str = None,
) -> QgsVectorLayer:
    if target_crs is None:
        target_crs = candidate_layer.crs().authid() if candidate_layer.crs().isValid() else "EPSG:4490"
    if not candidate_layer or not candidate_layer.isValid():
        raise ValueError("candidate_layer is invalid or None")

    has_accepted = (
        accepted_layer is not None
        and accepted_layer.isValid()
        and accepted_layer.featureCount() > 0
    )

    out_layer = QgsVectorLayer(
        f"Polygon?crs={target_crs}", LAYER_NAMES.CANDIDATES, "memory"
    )
    provider = out_layer.dataProvider()

    provider.addAttributes(candidate_layer.fields())
    out_layer.updateFields()

    if not has_accepted:
        for feature in candidate_layer.getFeatures():
            out_layer.dataProvider().addFeatures([feature])
        logger.info("No accepted_labels — returned candidate layer unchanged")
        return out_layer

    accepted_features = list(accepted_layer.getFeatures())
    accepted_transform = None
    if accepted_layer.crs() != candidate_layer.crs():
        accepted_transform = QgsCoordinateTransform(
            accepted_layer.crs(), candidate_layer.crs(), QgsProject.instance()
        )
    out_features = []

    for feature in candidate_layer.getFeatures():
        candidate_geom = feature.geometry()
        if not candidate_geom or candidate_geom.isNull():
            logger.warning("Skipping feature %s: null geometry", feature.id())
            continue

        overlapping_accepted = []
        for accepted_feature in accepted_features:
            accepted_geom = QgsGeometry(accepted_feature.geometry())
            if accepted_transform is not None:
                accepted_geom.transform(accepted_transform)
            if candidate_geom.intersects(accepted_geom):
                overlapping_accepted.append(accepted_geom)
        if not overlapping_accepted:
            out_features.append(feature)
            continue

        union_geom = overlapping_accepted[0]
        for g in overlapping_accepted[1:]:
            try:
                union_geom = union_geom.combine(g)
            except Exception:
                continue

        try:
            diff_geom = candidate_geom.difference(union_geom)
        except Exception as exc:
            logger.warning(
                "Difference failed for candidate %s: %s", feature.id(), exc
            )
            continue

        if diff_geom is None or diff_geom.isNull():
            continue

        if QgsWkbTypes.geometryType(diff_geom.wkbType()) != Qgis.GeometryType.Polygon:
            logger.warning(
                "Skipping non-polygon difference for candidate %s: %s",
                feature.id(),
                QgsWkbTypes.displayString(diff_geom.wkbType()),
            )
            continue
        if QgsWkbTypes.isMultiType(diff_geom.wkbType()):
            parts = diff_geom.asMultiPolygon()
        else:
            single = diff_geom.asPolygon()
            parts = [single] if single else []

        result_polygons = []
        for poly in parts:
            part_geom = QgsGeometry.fromPolygonXY(poly)
            if not part_geom or part_geom.isNull():
                continue
            area = part_geom.area()
            if area <= 0:
                continue
            result_polygons.append(part_geom)

        if not result_polygons:
            continue

        object_id = feature.attribute("object_id")
        if object_id is None:
            object_id = ""

        for idx, part_geom in enumerate(result_polygons):
            new_feature = QgsFeature(out_layer.fields())
            new_feature.setGeometry(part_geom)
            new_feature.setAttributes(feature.attributes())
            new_feature.setAttribute("part_id", f"{idx:03d}")
            if len(result_polygons) == 1:
                new_feature.setAttribute("part_id", "000")
            out_features.append(new_feature)

    out_layer.dataProvider().addFeatures(out_features)
    out_layer.updateExtents()
    logger.info(
        "Difference filter: %d candidates → %d result polygons",
        candidate_layer.featureCount(),
        len(out_features),
    )
    return out_layer


def has_accepted_layer(gpkg_path: str) -> bool:
    if not os.path.isfile(gpkg_path):
        return False
    uri = f"{gpkg_path}|layername={LAYER_NAMES.ACCEPTED}"
    layer = QgsVectorLayer(uri, LAYER_NAMES.ACCEPTED, "ogr")
    return layer.isValid()


def count_accepted_features(accepted_layer: QgsVectorLayer) -> int:
    if not accepted_layer or not accepted_layer.isValid():
        return 0
    return accepted_layer.featureCount()


def get_accepted_extent(accepted_layer: QgsVectorLayer) -> QgsRectangle | None:
    if not accepted_layer or not accepted_layer.isValid():
        return None
    if accepted_layer.featureCount() == 0:
        return None
    try:
        extent = accepted_layer.extent()
        if extent.isNull():
            return None
        return extent
    except Exception:
        return None
