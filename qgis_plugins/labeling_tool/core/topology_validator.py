"""Write explicit geometry, overlap and gap findings for final_composite."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsProject,
    QgsRectangle,
    QgsSpatialIndex,
    QgsVectorFileWriter,
    QgsVectorLayer,
    QgsWkbTypes,
)

from .layer_names import LAYER_NAMES
from .qgis_writer import write_vector_layer
from .run_state_db import RunStateDB
from . import accepted_integrity


ISSUE_FIELDS = [
    QgsField("issue_id", QVariant.String),
    QgsField("issue_type", QVariant.String),
    QgsField("class_code_a", QVariant.Int),
    QgsField("class_code_b", QVariant.Int),
    QgsField("feature_id_a", QVariant.String),
    QgsField("feature_id_b", QVariant.String),
    QgsField("area", QVariant.Double),
    QgsField("severity", QVariant.String),
    QgsField("resolved", QVariant.Int),
    QgsField("message", QVariant.String),
]


def _attr(feature, name, default=""):
    index = feature.fieldNameIndex(name)
    return feature.attribute(index) if index >= 0 else default


def pixel_area_tolerance(run_spec):
    transform = (run_spec.get("raster") or {}).get("transform") or []
    if len(transform) == 6:
        return abs(
            float(transform[0]) * float(transform[4])
            - float(transform[1]) * float(transform[3])
        )
    tiles = run_spec.get("tiles") or []
    bounds = (tiles[0].get("bounds") or {}) if tiles else {}
    if bounds:
        width = float(bounds["xmax"]) - float(bounds["xmin"])
        height = float(bounds["ymax"]) - float(bounds["ymin"])
        tile = run_spec.get("tile") or {}
        pixel_width = max(1, int(tile.get("width") or tiles[0].get("width") or 512))
        pixel_height = max(1, int(tile.get("height") or tiles[0].get("height") or 512))
        return abs(width * height) / (float(pixel_width) * float(pixel_height))
    return 0.0


def _selected_tile_target(run_spec):
    selection = run_spec.get("range_selection") or {}
    if selection.get("mode") != "vector_tile_intersection":
        extent = run_spec["requested_extent"]
        return QgsGeometry.fromRect(QgsRectangle(
            float(extent["xmin"]), float(extent["ymin"]),
            float(extent["xmax"]), float(extent["ymax"]),
        ))

    database = RunStateDB(run_spec["state_db"])
    rows = {}
    offset = 0
    while True:
        page = database.page_tiles(
            run_spec["run_id"], limit=500, offset=offset
        )
        if not page:
            break
        for tile in page:
            if str(tile.get("status")) == "excluded":
                continue
            bounds = tile.get("bounds")
            if not isinstance(bounds, dict):
                bounds = json.loads(tile.get("bounds_json") or "{}")
            rows.setdefault(int(tile["row_no"]), []).append(
                (int(tile["col_no"]), bounds)
            )
        offset += len(page)

    rectangles = []
    for values in rows.values():
        ordered = sorted(values)
        run = []
        for item in ordered:
            if run and item[0] != run[-1][0] + 1:
                rectangles.append(run)
                run = []
            run.append(item)
        if run:
            rectangles.append(run)
    geometries = []
    for run in rectangles:
        first = run[0][1]
        last = run[-1][1]
        geometries.append(QgsGeometry.fromRect(QgsRectangle(
            float(first["xmin"]), float(first["ymin"]),
            float(last["xmax"]), float(first["ymax"]),
        )))
    if not geometries:
        raise RuntimeError("vector Tile selection has no non-excluded Tile")
    return QgsGeometry.unaryUnion(geometries)


def validate_topology(run_spec, final_path, accepted_layer=None):
    final = QgsVectorLayer(
        f"{final_path}|layername={LAYER_NAMES.FINAL_COMPOSITE}", "final", "ogr"
    )
    if not final.isValid():
        raise RuntimeError(f"cannot open final composite: {final_path}")
    issues = QgsVectorLayer(
        f"MultiPolygon?crs={final.crs().authid()}", LAYER_NAMES.TOPOLOGY_ISSUES, "memory"
    )
    issues.dataProvider().addAttributes(ISSUE_FIELDS)
    issues.updateFields()
    issue_fields = issues.fields()
    issue_features = []

    def add_issue(issue_type, geometry, feature_a=None, feature_b=None, severity="high", message=""):
        feature = QgsFeature(issue_fields)
        if geometry is not None and not geometry.isNull() and not geometry.isEmpty():
            copy = QgsGeometry(geometry)
            if QgsWkbTypes.geometryType(copy.wkbType()) == Qgis.GeometryType.Polygon:
                copy.convertToMultiType()
                feature.setGeometry(copy)
        values = {
            "issue_id": uuid.uuid4().hex,
            "issue_type": issue_type,
            "class_code_a": int(_attr(feature_a, "class_code", 0) or 0) if feature_a else 0,
            "class_code_b": int(_attr(feature_b, "class_code", 0) or 0) if feature_b else 0,
            "feature_id_a": str(_attr(feature_a, "object_id", feature_a.id() if feature_a else "")) if feature_a else "",
            "feature_id_b": str(_attr(feature_b, "object_id", feature_b.id() if feature_b else "")) if feature_b else "",
            "area": float(geometry.area()) if geometry is not None and not geometry.isNull() else 0.0,
            "severity": severity,
            "resolved": 0,
            "message": message,
        }
        feature.setAttributes([values[field.name()] for field in issue_fields])
        issue_features.append(feature)

    features = list(final.getFeatures())
    valid_geometries = []
    for feature in features:
        geometry = feature.geometry()
        if geometry is None or geometry.isNull() or geometry.isEmpty():
            add_issue("empty_geometry", None, feature, severity="high", message="Feature geometry is empty")
            continue
        if not geometry.isGeosValid():
            add_issue("invalid_geometry", geometry, feature, severity="high", message="GEOS geometry is invalid")
        else:
            valid_geometries.append(geometry)

    tolerance = pixel_area_tolerance(run_spec)
    index = QgsSpatialIndex()
    for feature in features:
        index.addFeature(feature)
    by_id = {feature.id(): feature for feature in features}
    seen = set()
    for feature in features:
        geometry = feature.geometry()
        if geometry is None or geometry.isNull() or geometry.isEmpty():
            continue
        for other_id in index.intersects(geometry.boundingBox()):
            pair = tuple(sorted((feature.id(), other_id)))
            if other_id == feature.id() or pair in seen:
                continue
            seen.add(pair)
            other = by_id.get(other_id)
            if other is None:
                continue
            try:
                intersection = geometry.intersection(other.geometry())
            except Exception:
                continue
            if intersection is None or intersection.isNull() or intersection.isEmpty() or intersection.area() <= tolerance:
                continue
            same_class = int(_attr(feature, "class_code", 0)) == int(_attr(other, "class_code", 0))
            add_issue(
                "same_class_overlap" if same_class else "cross_class_overlap",
                intersection,
                feature,
                other,
                severity="high",
                message="Confirmed class polygons overlap beyond one-pixel tolerance",
            )

    target = _selected_tile_target(run_spec)
    if accepted_layer is not None and accepted_layer.isValid():
        accepted_tolerance = accepted_integrity.strict_overlap_tolerance(run_spec)
        accepted_integrity.audit_accepted_layer(
            accepted_layer,
            overlap_tolerance=accepted_tolerance,
            expected_crs=final.crs(),
        )
        for final_feature, accepted_feature, intersection in (
            accepted_integrity.assert_no_accepted_overlap(
                final,
                accepted_layer,
                overlap_tolerance=accepted_tolerance,
                raise_on_overlap=False,
            )
        ):
            add_issue(
                "accepted_overlap",
                intersection,
                final_feature,
                accepted_feature,
                severity="high",
                message=(
                    "final_composite overlaps the existing accepted_labels; "
                    "this issue cannot be overridden during accepted write"
                ),
            )
        accepted_geometries = []
        transform = None
        if accepted_layer.crs() != final.crs():
            transform = QgsCoordinateTransform(accepted_layer.crs(), final.crs(), QgsProject.instance())
        for feature in accepted_layer.getFeatures():
            geometry = QgsGeometry(feature.geometry())
            if transform is not None:
                geometry.transform(transform)
            if geometry and not geometry.isNull() and not geometry.isEmpty():
                accepted_geometries.append(geometry)
        if accepted_geometries:
            target = target.difference(QgsGeometry.unaryUnion(accepted_geometries))
    if valid_geometries:
        gap = target.difference(QgsGeometry.unaryUnion(valid_geometries))
    else:
        gap = target
    if gap is not None and not gap.isNull() and not gap.isEmpty() and gap.area() > tolerance:
        add_issue("gap", gap, severity="medium", message="Requested extent contains uncovered area")

    issues.dataProvider().addFeatures(issue_features)
    issues.updateExtents()
    output = Path(run_spec["run_dir"]) / "final" / "topology_issues.gpkg"
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.layerName = LAYER_NAMES.TOPOLOGY_ISSUES
    options.actionOnExistingFile = (
        QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteFile
    )
    error, message = write_vector_layer(issues, output, options)
    if error != QgsVectorFileWriter.WriterError.NoError:
        raise RuntimeError(f"cannot write topology issues: {message}")
    counts = {}
    for feature in issue_features:
        issue_type = str(feature.attribute("issue_type"))
        counts[issue_type] = counts.get(issue_type, 0) + 1
    return str(output), len(issue_features), counts
