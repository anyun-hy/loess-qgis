"""Assemble final_composite exclusively from 14 confirmed working layers."""

from __future__ import annotations

import datetime
from pathlib import Path

from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    Qgis,
    QgsFeature,
    QgsField,
    QgsVectorFileWriter,
    QgsVectorLayer,
    QgsWkbTypes,
)

from . import class_workspace
from .layer_names import LAYER_NAMES
from .qgis_writer import write_vector_layer
from .run_spec import CLASS_NAMES, CLASS_ORDER


FINAL_FIELDS = [
    QgsField("run_id", QVariant.String),
    QgsField("object_id", QVariant.String),
    QgsField("part_id", QVariant.String),
    QgsField("class_code", QVariant.Int),
    QgsField("class_name", QVariant.String),
    QgsField("confidence_mean", QVariant.Double),
    QgsField("confidence_std", QVariant.Double),
    QgsField("baseline_stream_id", QVariant.String),
    QgsField("source_stream_id", QVariant.String),
    QgsField("geometry_source", QVariant.String),
    QgsField("geometry_revision", QVariant.Int),
    QgsField("edit_base", QVariant.String),
    QgsField("sam_session_id", QVariant.String),
    QgsField("sam_score", QVariant.Double),
    QgsField("sam_version", QVariant.String),
    QgsField("model_version", QVariant.String),
    QgsField("fusion_profile_id", QVariant.String),
    QgsField("reviewed", QVariant.Int),
    QgsField("created_at", QVariant.String),
    QgsField("updated_at", QVariant.String),
]


def _attribute(feature, name, default=""):
    index = feature.fieldNameIndex(name)
    if index < 0:
        return default
    value = feature.attribute(index)
    return default if value is None else value


def assemble_final(run_spec, workspace):
    if workspace.get("run_id") != run_spec.get("run_id"):
        raise ValueError("class workspace belongs to a different run")
    if workspace.get("active_sam_session_id"):
        raise ValueError("an active SAM3 session blocks final assembly")
    classes = workspace.get("classes") or {}
    missing = [
        code for code in CLASS_ORDER
        if not (classes.get(str(code)) or {}).get("confirmed")
    ]
    if missing:
        raise ValueError(f"all 14 class working layers must be confirmed; missing={missing}")
    crs = str((run_spec.get("raster") or {}).get("crs") or "EPSG:4490")
    output_layer = QgsVectorLayer(
        f"MultiPolygon?crs={crs}", LAYER_NAMES.FINAL_COMPOSITE, "memory"
    )
    output_layer.dataProvider().addAttributes(FINAL_FIELDS)
    output_layer.updateFields()
    output_fields = output_layer.fields()
    output_features = []
    identities = set()
    created_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    baseline_stream_id = str(workspace["baseline_stream_id"])

    for code in CLASS_ORDER:
        record = classes[str(code)]
        layer = class_workspace.working_layer(record, f"final_class_{code}")
        if layer.isEditable() and layer.isModified():
            raise ValueError(f"class {code} contains unsaved edits")
        if layer.featureCount() != int(record.get("feature_count", -1)):
            raise ValueError(f"class {code} feature count differs from workspace.json")
        for source_feature in layer.getFeatures():
            feature_code = int(_attribute(source_feature, "class_code", -1))
            feature_name = str(_attribute(source_feature, "class_name", ""))
            if feature_code != code or feature_name != CLASS_NAMES[code]:
                raise ValueError(f"class working layer identity changed: {feature_code}/{feature_name}")
            if str(_attribute(source_feature, "baseline_stream_id", "")) != baseline_stream_id:
                raise ValueError(f"class {code} contains a different Fusion baseline")
            object_id = str(_attribute(source_feature, "object_id", ""))
            part_id = str(_attribute(source_feature, "part_id", "000") or "000")
            identity = (object_id, part_id)
            if not object_id or identity in identities:
                raise ValueError(f"duplicate or empty working object identity: {identity}")
            identities.add(identity)
            geometry = source_feature.geometry()
            if (
                geometry is None or geometry.isNull() or geometry.isEmpty()
                or not geometry.isGeosValid()
                or QgsWkbTypes.geometryType(geometry.wkbType()) != Qgis.GeometryType.Polygon
            ):
                raise ValueError(f"class {code} contains invalid or non-polygon geometry: {object_id}")
            geometry.convertToMultiType()
            feature = QgsFeature(output_fields)
            feature.setGeometry(geometry)
            values = {
                "run_id": run_spec["run_id"],
                "object_id": object_id,
                "part_id": part_id,
                "class_code": code,
                "class_name": CLASS_NAMES[code],
                "confidence_mean": float(_attribute(source_feature, "confidence_mean", 0.0) or 0.0),
                "confidence_std": float(_attribute(source_feature, "confidence_std", 0.0) or 0.0),
                "baseline_stream_id": baseline_stream_id,
                "source_stream_id": baseline_stream_id,
                "geometry_source": str(_attribute(source_feature, "geometry_source", "fusion")),
                "geometry_revision": int(_attribute(source_feature, "geometry_revision", 0) or 0),
                "edit_base": str(_attribute(source_feature, "edit_base", "")),
                "sam_session_id": str(_attribute(source_feature, "sam_session_id", "")),
                "sam_score": _attribute(source_feature, "sam_score", None),
                "sam_version": str(_attribute(source_feature, "sam_version", "")),
                "model_version": str(_attribute(source_feature, "model_version", "")),
                "fusion_profile_id": baseline_stream_id.split(":", 1)[1],
                "reviewed": int(_attribute(source_feature, "reviewed", 1) or 0),
                "created_at": str(_attribute(source_feature, "created_at", created_at) or created_at),
                "updated_at": str(_attribute(source_feature, "updated_at", created_at) or created_at),
            }
            if values["reviewed"] != 1:
                raise ValueError(f"class {code} contains an unreviewed feature: {object_id}")
            feature.setAttributes([values[field.name()] for field in output_fields])
            output_features.append(feature)

    output_layer.dataProvider().addFeatures(output_features)
    output_layer.updateExtents()
    output = Path(run_spec["run_dir"]) / "final" / "final_composite.gpkg"
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.layerName = LAYER_NAMES.FINAL_COMPOSITE
    options.actionOnExistingFile = QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteFile
    error, message = write_vector_layer(output_layer, output, options)
    if error != QgsVectorFileWriter.WriterError.NoError:
        raise RuntimeError(f"cannot write final_composite: {message}")
    return str(output), len(output_features)
