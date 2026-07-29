import os

from qgis.core import (
    QgsProject, QgsVectorLayer, QgsVectorFileWriter, QgsFeature,
    QgsField,
)
from qgis.PyQt.QtCore import Qt, QVariant, QDateTime

from .layer_names import LAYER_NAMES
from .qgis_writer import write_vector_layer

ACCEPTED_FIELDS = [
    ("run_id", QVariant.String),
    ("object_id", QVariant.String),
    ("part_id", QVariant.String),
    ("class_code", QVariant.Int),
    ("class_name", QVariant.String),
    ("confidence_mean", QVariant.Double),
    ("confidence_std", QVariant.Double),
    ("baseline_stream_id", QVariant.String),
    ("source_stream_id", QVariant.String),
    ("source", QVariant.String),
    ("geometry_source", QVariant.String),
    ("geometry_revision", QVariant.Int),
    ("edit_base", QVariant.String),
    ("sam_session_id", QVariant.String),
    ("sam_score", QVariant.Double),
    ("model_version", QVariant.String),
    ("fusion_profile_id", QVariant.String),
    ("sam_version", QVariant.String),
    ("reviewed", QVariant.Int),
    ("created_at", QVariant.String),
    ("updated_at", QVariant.String),
]

ACCEPTED_FIELDS_QGS = [QgsField(name, typ) for name, typ in ACCEPTED_FIELDS]


def _ensure_accepted_schema(layer):
    existing = {field.name() for field in layer.fields()}
    missing = [QgsField(name, typ) for name, typ in ACCEPTED_FIELDS if name not in existing]
    if not missing:
        return layer
    if not layer.startEditing():
        raise RuntimeError("cannot start accepted_labels schema migration")
    for field in missing:
        if not layer.addAttribute(field):
            layer.rollBack()
            raise RuntimeError(f"cannot add accepted_labels field: {field.name()}")
    if not layer.commitChanges():
        errors = "; ".join(layer.commitErrors())
        layer.rollBack()
        raise RuntimeError(f"cannot commit accepted_labels schema migration: {errors}")
    return layer

def _build_accepted_fields_qgs():
    fields = []
    for name, typ in ACCEPTED_FIELDS:
        fields.append(QgsField(name, typ))
    return fields


def get_accepted_layer(gpkg_path, crs=None):
    if crs is None:
        project_crs = QgsProject.instance().crs()
        crs = project_crs.authid() if project_crs.isValid() else "EPSG:4326"
    if os.path.exists(gpkg_path):
        uri = f"{gpkg_path}|layername={LAYER_NAMES.ACCEPTED}"
        layer = QgsVectorLayer(uri, LAYER_NAMES.ACCEPTED, "ogr")
        if layer.isValid():
            return _ensure_accepted_schema(layer)

    mem_layer = QgsVectorLayer(
        f"MultiPolygon?crs={crs}", LAYER_NAMES.ACCEPTED, "memory"
    )
    mem_layer.dataProvider().addAttributes(_build_accepted_fields_qgs())
    mem_layer.updateFields()

    save_options = QgsVectorFileWriter.SaveVectorOptions()
    save_options.driverName = "GPKG"
    save_options.layerName = LAYER_NAMES.ACCEPTED
    save_options.actionOnExistingFile = (
        QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteLayer
    )
    if not os.path.exists(gpkg_path):
        save_options.actionOnExistingFile = (
            QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteFile
        )

    error, err_msg = write_vector_layer(mem_layer, gpkg_path, save_options)
    if error != QgsVectorFileWriter.WriterError.NoError:
        raise RuntimeError(f"Failed to create accepted_labels GPKG: {err_msg}")

    uri = f"{gpkg_path}|layername={LAYER_NAMES.ACCEPTED}"
    layer = QgsVectorLayer(uri, LAYER_NAMES.ACCEPTED, "ogr")
    return _ensure_accepted_schema(layer) if layer.isValid() else layer


def append_final_to_accepted(final_path, accepted_path, run_manifest_path):
    import json

    with open(run_manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "ready":
        raise ValueError("run_manifest must be ready before accepted_labels write")
    manifest_run_id = str(manifest.get("run_id") or "")
    valid_streams = {
        str(item.get("stream_id"))
        for item in manifest.get("streams") or []
        if item.get("status") == "ready"
    }
    final = QgsVectorLayer(
        f"{final_path}|layername={LAYER_NAMES.FINAL_COMPOSITE}", "final_to_accept", "ogr"
    )
    if not final.isValid():
        raise RuntimeError(f"cannot open final_composite: {final_path}")
    accepted = get_accepted_layer(accepted_path, final.crs().authid())
    if not accepted.isValid():
        raise RuntimeError(f"cannot open accepted_labels: {accepted_path}")
    accepted = _ensure_accepted_schema(accepted)
    existing_object_ids = {
        str(feature.attribute("object_id") or "")
        for feature in accepted.getFeatures()
        if str(feature.attribute("object_id") or "")
    }
    accepted_in_run = set()
    pending = []
    for feature in final.getFeatures():
        geometry = feature.geometry()
        if geometry is None or geometry.isNull() or geometry.isEmpty() or not geometry.isGeosValid():
            raise ValueError(f"invalid final geometry for feature {feature.id()}")
        class_code = int(feature.attribute("class_code"))
        class_name = str(feature.attribute("class_name"))
        from .run_spec import CLASS_NAMES
        if CLASS_NAMES.get(class_code) != class_name:
            raise ValueError(f"class mapping mismatch: {class_code}/{class_name}")
        feature_run_id = str(feature.attribute("run_id") or "")
        if not manifest_run_id or feature_run_id != manifest_run_id:
            raise ValueError(
                f"final feature run_id does not match run_manifest: {feature_run_id}/{manifest_run_id}"
            )
        source_stream_id = str(feature.attribute("source_stream_id") or "")
        if source_stream_id not in valid_streams:
            raise ValueError(f"source_stream_id is not traceable in run_manifest: {source_stream_id}")
        object_id = str(feature.attribute("object_id") or "")
        part_id = str(feature.attribute("part_id") or "000")
        run_key = (feature_run_id, object_id, part_id)
        if not object_id or object_id in existing_object_ids or run_key in accepted_in_run:
            raise ValueError(f"duplicate or empty accepted object identity: {run_key}")
        _validate_provenance(feature)
        existing_object_ids.add(object_id)
        accepted_in_run.add(run_key)
        output = QgsFeature(accepted.fields())
        output.setGeometry(geometry)
        for name, _type in ACCEPTED_FIELDS:
            if name == "reviewed":
                value = 1
            elif name == "source":
                value = "class_working"
            elif name == "created_at":
                value = feature.attribute(name) or QDateTime.currentDateTime().toString(
                    Qt.ISODate
                )
            else:
                index = feature.fieldNameIndex(name)
                value = feature.attribute(index) if index >= 0 else ""
            output.setAttribute(name, value)
        pending.append(output)
    if not accepted.startEditing():
        raise RuntimeError("cannot start accepted_labels edit session")
    if not accepted.addFeatures(pending):
        accepted.rollBack()
        raise RuntimeError("cannot add final features to accepted_labels")
    if not accepted.commitChanges():
        errors = "; ".join(accepted.commitErrors())
        accepted.rollBack()
        raise RuntimeError(f"cannot commit accepted_labels: {errors}")
    return len(pending)


def _validate_provenance(feature):
    geometry_source = str(feature.attribute("geometry_source") or "")
    revision = int(feature.attribute("geometry_revision") or 0)
    edit_base = str(feature.attribute("edit_base") or "")
    sam_session_id = str(feature.attribute("sam_session_id") or "")
    if geometry_source == "fusion":
        valid = revision == 0 and not edit_base and not sam_session_id
    elif geometry_source == "sam3":
        valid = revision >= 1 and not edit_base and bool(sam_session_id)
    elif geometry_source == "manual_edited":
        valid = (
            revision >= 1
            and edit_base in ("", "fusion", "sam3", "manual_edited")
            and (edit_base != "sam3" or bool(sam_session_id))
        )
    else:
        valid = False
    if not valid:
        raise ValueError(
            "invalid geometry provenance: "
            f"source={geometry_source}, revision={revision}, "
            f"edit_base={edit_base}, sam_session_id={sam_session_id}"
        )
