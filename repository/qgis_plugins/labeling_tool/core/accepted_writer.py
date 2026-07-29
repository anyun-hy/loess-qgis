import json
import os
from pathlib import Path

from qgis.core import (
    QgsProject, QgsVectorLayer, QgsVectorFileWriter, QgsFeature,
    QgsField,
)
from qgis.PyQt.QtCore import Qt, QVariant, QDateTime

from .layer_names import LAYER_NAMES
from .qgis_writer import write_vector_layer
from .run_spec import CLASS_NAMES, sha256_file
from . import accepted_integrity

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
            return layer
        raise RuntimeError(
            f"existing GeoPackage has no valid {LAYER_NAMES.ACCEPTED} layer: {gpkg_path}"
        )

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
    return layer


def _verified_run_spec(manifest, run_manifest_path):
    run_spec_value = str(manifest.get("run_spec") or "").strip()
    if not run_spec_value:
        raise ValueError("run_manifest 缺少 run_spec 路径")
    run_spec_path = Path(run_spec_value).expanduser()
    if not run_spec_path.is_absolute():
        run_spec_path = Path(run_manifest_path).resolve().parent / run_spec_path
    run_spec_path = run_spec_path.resolve()
    expected_sha = str(manifest.get("run_spec_sha256") or "")
    if (
        not run_spec_path.is_file()
        or not expected_sha
        or sha256_file(run_spec_path) != expected_sha
    ):
        raise ValueError("run_spec 不存在或 SHA256 与 run_manifest 不一致")
    with open(run_spec_path, "r", encoding="utf-8") as handle:
        run_spec = json.load(handle)
    if str(run_spec.get("run_id") or "") != str(manifest.get("run_id") or ""):
        raise ValueError("run_spec 与 run_manifest 的 run_id 不一致")
    return run_spec


def append_final_to_accepted(final_path, accepted_path, run_manifest_path):
    with open(run_manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "ready":
        raise ValueError("run_manifest must be ready before accepted_labels write")
    run_spec = _verified_run_spec(manifest, run_manifest_path)
    expected_target_value = str(run_spec.get("accepted_target_gpkg") or "").strip()
    if not expected_target_value:
        raise ValueError("run_spec 缺少 accepted_target_gpkg，禁止写入 Run 快照")
    if not str(accepted_path or "").strip():
        raise ValueError("accepted_labels 写入目标为空")
    expected_target = Path(expected_target_value).expanduser().resolve()
    actual_target = Path(accepted_path).expanduser().resolve()
    if actual_target != expected_target:
        raise ValueError(
            f"accepted_labels 写入目标与 run_spec 不一致: {actual_target}/{expected_target}"
        )
    snapshot_value = str(run_spec.get("accepted_gpkg") or "").strip()
    if snapshot_value and Path(snapshot_value).expanduser().resolve() == actual_target:
        raise ValueError("accepted_labels 长期写入目标不得等于 Run 只读快照")

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
    overlap_tolerance = accepted_integrity.strict_overlap_tolerance(run_spec)
    accepted = None
    existing_identities = set()
    if actual_target.is_file():
        accepted = get_accepted_layer(str(actual_target), final.crs().authid())
        accepted_integrity.audit_accepted_layer(
            accepted,
            overlap_tolerance=overlap_tolerance,
            expected_crs=final.crs(),
        )
        accepted_integrity.assert_no_accepted_overlap(
            final,
            accepted,
            overlap_tolerance=overlap_tolerance,
        )
        existing_identities = {
            (
                str(feature.attribute("run_id") or ""),
                str(feature.attribute("object_id") or ""),
                str(feature.attribute("part_id") or ""),
            )
            for feature in accepted.getFeatures()
            if str(feature.attribute("object_id") or "")
        }
    accepted_in_run = set()
    validated_features = []
    for feature in final.getFeatures():
        geometry = feature.geometry()
        if geometry is None or geometry.isNull() or geometry.isEmpty() or not geometry.isGeosValid():
            raise ValueError(f"invalid final geometry for feature {feature.id()}")
        class_code = int(feature.attribute("class_code"))
        class_name = str(feature.attribute("class_name"))
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
        if (
            not object_id
            or not part_id
            or run_key in existing_identities
            or run_key in accepted_in_run
        ):
            raise ValueError(f"duplicate or empty accepted object identity: {run_key}")
        _validate_provenance(feature)
        existing_identities.add(run_key)
        accepted_in_run.add(run_key)
        validated_features.append(QgsFeature(feature))
    if accepted is None:
        actual_target.parent.mkdir(parents=True, exist_ok=True)
        accepted = get_accepted_layer(str(actual_target), final.crs().authid())
        if not accepted.isValid():
            raise RuntimeError(f"cannot open accepted_labels: {actual_target}")
        accepted_integrity.audit_accepted_layer(
            accepted,
            overlap_tolerance=overlap_tolerance,
            expected_crs=final.crs(),
        )
    pending = []
    for feature in validated_features:
        output = QgsFeature(accepted.fields())
        output.setGeometry(feature.geometry())
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
