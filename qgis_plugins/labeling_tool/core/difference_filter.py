from qgis.core import (
    QgsProject, QgsVectorLayer, QgsGeometry,
    QgsRectangle,
    QgsFeatureRequest, QgsCoordinateTransform,
)


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
