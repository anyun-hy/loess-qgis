"""Persistent 14-class editing workspace derived from one approved Fusion stream."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    Qgis,
    QgsDefaultValue,
    QgsFeature,
    QgsFeatureRequest,
    QgsField,
    QgsFieldConstraints,
    QgsVectorFileWriter,
    QgsVectorLayer,
    QgsWkbTypes,
)

from .layer_names import LAYER_NAMES
from .qgis_writer import write_vector_layer
from .run_spec import CLASS_NAMES, CLASS_ORDER, atomic_write_json, sha256_file


WORKSPACE_SCHEMA_VERSION = 2
WORK_FIELDS = (
    ("baseline_stream_id", QVariant.String),
    ("geometry_source", QVariant.String),
    ("geometry_revision", QVariant.Int),
    ("edit_base", QVariant.String),
    ("sam_session_id", QVariant.String),
    ("sam_score", QVariant.Double),
    ("sam_version", QVariant.String),
    ("reviewed", QVariant.Int),
    ("updated_at", QVariant.String),
)
IMMUTABLE_FIELDS = (
    "run_id", "object_id", "part_id", "class_code", "class_name",
    "baseline_stream_id",
)
MANUAL_REQUIRED_FIELDS = ("class_code", "class_name", "object_id")


class ClassWorkspaceError(RuntimeError):
    pass


def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def workspace_paths(run_spec):
    directory = Path(run_spec["run_dir"]) / "classes"
    return {
        "directory": directory,
        "workspace": directory / "workspace.json",
        "history": directory / "edit_history.jsonl",
    }


def class_layer_path(run_spec, class_code):
    return workspace_paths(run_spec)["directory"] / f"class_{int(class_code)}.gpkg"


def stream_by_id(streams, stream_id):
    matches = [item for item in streams if item.get("stream_id") == stream_id]
    if len(matches) != 1:
        raise ClassWorkspaceError(f"expected exactly one result stream: {stream_id}")
    return matches[0]


def approved_fusion_streams(run_spec, streams):
    if run_spec.get("manual_only"):
        return [
            stream for stream in streams
            if stream.get("kind") == "fusion"
            and stream.get("status") == "ready"
            and stream.get("manual_validated") is True
        ]
    fusion = run_spec.get("fusion") or {}
    expected_id = f"fusion:{fusion.get('profile_id')}" if fusion else ""
    approved = []
    for stream in streams:
        if (
            stream.get("kind") != "fusion"
            or stream.get("status") != "ready"
            or stream.get("stream_id") != expected_id
            or stream.get("boundary_fitting_status") != "passed"
        ):
            continue
        try:
            _validate_boundary_report(stream)
            _validate_fusion_profile(run_spec, stream)
        except ClassWorkspaceError:
            continue
        approved.append(stream)
    return approved


def _validate_manual_layer(layer, *, expected_code=None, allow_empty=False):
    if not layer.isValid():
        raise ClassWorkspaceError(f"无法打开人工分类图层: {layer.source()}")
    if QgsWkbTypes.geometryType(layer.wkbType()) != Qgis.GeometryType.Polygon:
        raise ClassWorkspaceError("人工分类图层必须是 Polygon 或 MultiPolygon")
    fields = {field.name() for field in layer.fields()}
    missing = [name for name in MANUAL_REQUIRED_FIELDS if name not in fields]
    if missing:
        raise ClassWorkspaceError("人工分类图层缺少字段: " + ", ".join(missing))
    count = 0
    empty_feature_ids = []
    for feature in layer.getFeatures():
        try:
            class_code = int(feature.attribute("class_code"))
        except (TypeError, ValueError):
            raise ClassWorkspaceError(f"要素 {feature.id()} 的 class_code 无效")
        if class_code not in CLASS_ORDER:
            raise ClassWorkspaceError(
                f"要素 {feature.id()} 使用了非14类编码: {class_code}"
            )
        if expected_code is not None and class_code != int(expected_code):
            raise ClassWorkspaceError(
                f"类别 {expected_code} 工作层包含错误类别 {class_code}: feature {feature.id()}"
            )
        class_name = str(feature.attribute("class_name") or "")
        if class_name != CLASS_NAMES[class_code]:
            raise ClassWorkspaceError(
                f"要素 {feature.id()} 类别名称不匹配: {class_code}/{class_name}"
            )
        if not str(feature.attribute("object_id") or "").strip():
            raise ClassWorkspaceError(f"要素 {feature.id()} 的 object_id 为空")
        geometry = feature.geometry()
        if geometry is None or geometry.isNull() or geometry.isEmpty():
            if allow_empty:
                empty_feature_ids.append(feature.id())
                continue
            raise ClassWorkspaceError(f"要素 {feature.id()} 的几何无效")
        if not geometry.isGeosValid():
            raise ClassWorkspaceError(f"要素 {feature.id()} 的几何无效")
        count += 1
    return count, empty_feature_ids


def validate_manual_fusion_stream(stream):
    layer, path, layer_name = _source_layer(stream)
    count, _empty_ids = _validate_manual_layer(layer)
    if count <= 0:
        raise ClassWorkspaceError(f"Fusion 图层没有要素: {path}")
    stream["manual_validated"] = True
    stream["review_polygons"] = str(path.resolve())
    stream["review_layer_name"] = layer_name
    return count


def validate_manual_workspace(workspace):
    classes = workspace.get("classes") or {}
    if set(classes) != {str(code) for code in CLASS_ORDER}:
        raise ClassWorkspaceError("人工工作区没有完整的14个类别记录")
    total = 0
    pending_cleanup = []
    for code in CLASS_ORDER:
        record = classes[str(code)]
        layer = working_layer(record, f"manual_validate_{code}")
        count, empty_ids = _validate_manual_layer(
            layer, expected_code=code, allow_empty=True
        )
        record["feature_count"] = count
        total += count
        if empty_ids:
            pending_cleanup.append((layer, empty_ids))
    for layer, empty_ids in pending_cleanup:
        if not layer.startEditing():
            raise ClassWorkspaceError("无法清理人工工作层中的空几何记录")
        if not layer.deleteFeatures(empty_ids) or not layer.commitChanges():
            layer.rollBack()
            raise ClassWorkspaceError("清理人工工作层中的空几何记录失败")
    removed_empty = sum(len(empty_ids) for _layer, empty_ids in pending_cleanup)
    workspace["feature_count"] = total
    workspace["updated_at"] = _now()
    return {"feature_count": total, "removed_empty": removed_empty}


def _validate_fusion_profile(run_spec, stream):
    fusion = run_spec.get("fusion") or {}
    snapshot = Path(str(fusion.get("snapshot_path") or ""))
    if not snapshot.is_file() or sha256_file(snapshot) != fusion.get("sha256"):
        raise ClassWorkspaceError("Fusion profile snapshot is missing or its SHA256 changed")
    with open(snapshot, "r", encoding="utf-8") as handle:
        profile = json.load(handle)
    if profile.get("status") != "approved" or (profile.get("approval") or {}).get("passed") is not True:
        raise ClassWorkspaceError("Fusion profile is not approved")
    if str(stream.get("fusion_profile_id") or "") != str(profile.get("profile_id") or ""):
        raise ClassWorkspaceError("Fusion stream does not match the approved profile")


def _validate_boundary_report(stream):
    paths = stream.get("paths") or {}
    raw_path = Path(str(paths.get("semantic_polygons_raw") or ""))
    formal_path = Path(str(paths.get("semantic_polygons") or ""))
    report_path = Path(str(paths.get("boundary_fitting_report") or ""))
    if not raw_path.is_file() or not formal_path.is_file() or not report_path.is_file():
        raise ClassWorkspaceError("Fusion raw/formal/boundary report assets are incomplete")
    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    if (
        report.get("status") != "passed"
        or (report.get("validation") or {}).get("passed") is not True
    ):
        raise ClassWorkspaceError("Fusion common-divider boundary fitting report did not pass")
    if report.get("input_sha256") != sha256_file(raw_path):
        raise ClassWorkspaceError("Fusion raw polygon SHA256 does not match the boundary report")
    if report.get("output_sha256") != sha256_file(formal_path):
        raise ClassWorkspaceError("Fusion formal polygon SHA256 does not match the boundary report")
    return report


def _source_layer(stream):
    paths = stream.get("paths") or {}
    path = str(stream.get("review_polygons") or paths.get("semantic_polygons") or "")
    layer_name = str(stream.get("review_layer_name") or LAYER_NAMES.SEMANTIC)
    layer = QgsVectorLayer(f"{path}|layername={layer_name}", "fusion_workspace_source", "ogr")
    if not layer.isValid():
        raise ClassWorkspaceError(f"cannot open Fusion workspace source: {path}")
    return layer, Path(path), layer_name


def _constraint_value(name):
    enum_type = getattr(QgsFieldConstraints, "Constraint", QgsFieldConstraints)
    return getattr(enum_type, name)


def _quoted_expression(value):
    return "'" + str(value or "").replace("'", "''") + "'"


def apply_class_constraints(
    layer, class_code, *, run_id="", baseline_stream_id=""
):
    class_name = CLASS_NAMES[int(class_code)].replace("'", "''")
    class_code_index = layer.fields().indexOf("class_code")
    class_name_index = layer.fields().indexOf("class_name")
    object_index = layer.fields().indexOf("object_id")
    for index in (class_code_index, class_name_index, object_index):
        if index < 0:
            raise ClassWorkspaceError("class workspace schema is missing an immutable field")
        layer.setFieldConstraint(index, _constraint_value("ConstraintNotNull"))
    layer.setFieldConstraint(object_index, _constraint_value("ConstraintUnique"))
    layer.setConstraintExpression(
        class_code_index,
        f'"class_code" = {int(class_code)}',
        "class_code is fixed for this working layer",
    )
    layer.setConstraintExpression(
        class_name_index,
        f'"class_name" = \'{class_name}\'',
        "class_name is fixed for this working layer",
    )
    layer.setDefaultValueDefinition(
        class_code_index, QgsDefaultValue(str(int(class_code)), True)
    )
    layer.setDefaultValueDefinition(
        class_name_index, QgsDefaultValue(f"'{class_name}'", True)
    )
    prefix = f"{run_id}_new_" if run_id else "manual_new_"
    object_expression = (
        f"{_quoted_expression(prefix)} || "
        "replace(replace(replace(uuid(), '{', ''), '}', ''), '-', '')"
    )
    layer.setDefaultValueDefinition(
        object_index, QgsDefaultValue(object_expression, False)
    )
    defaults = {
        "run_id": _quoted_expression(run_id),
        "part_id": "'000'",
        "baseline_stream_id": _quoted_expression(baseline_stream_id),
        "geometry_source": "'manual_edited'",
        "geometry_revision": "1",
        "edit_base": "''",
        "reviewed": "0",
    }
    for name, expression in defaults.items():
        index = layer.fields().indexOf(name)
        if index >= 0 and (run_id or name not in ("run_id", "baseline_stream_id")):
            layer.setDefaultValueDefinition(
                index, QgsDefaultValue(expression, False)
            )
    form_config = layer.editFormConfig()
    form_config.setSuppress(Qgis.AttributeFormSuppression.On)
    layer.setEditFormConfig(form_config)
    return layer


def _memory_class_layer(source, class_code, baseline_stream_id):
    memory = QgsVectorLayer(
        f"MultiPolygon?crs={source.crs().authid()}",
        LAYER_NAMES.CLASS_POLYGONS,
        "memory",
    )
    provider = memory.dataProvider()
    provider.addAttributes(list(source.fields()))
    existing = {field.name() for field in memory.fields()}
    provider.addAttributes([
        QgsField(name, field_type) for name, field_type in WORK_FIELDS if name not in existing
    ])
    memory.updateFields()
    fields = memory.fields()
    request = QgsFeatureRequest().setFilterExpression(f'"class_code" = {int(class_code)}')
    created_at = _now()
    features = []
    for source_feature in source.getFeatures(request):
        feature = QgsFeature(fields)
        geometry = source_feature.geometry()
        if geometry is None or geometry.isNull() or geometry.isEmpty() or not geometry.isGeosValid():
            raise ClassWorkspaceError(
                f"Fusion class {class_code} contains invalid geometry: {source_feature.id()}"
            )
        geometry.convertToMultiType()
        feature.setGeometry(geometry)
        for field in source.fields():
            feature.setAttribute(field.name(), source_feature.attribute(field.name()))
        values = {
            "baseline_stream_id": baseline_stream_id,
            "geometry_source": "fusion",
            "geometry_revision": 0,
            "edit_base": "",
            "sam_session_id": "",
            "sam_score": None,
            "sam_version": "",
            "reviewed": 0,
            "updated_at": created_at,
        }
        for name, value in values.items():
            feature.setAttribute(name, value)
        features.append(feature)
    provider.addFeatures(features)
    memory.updateExtents()
    apply_class_constraints(memory, class_code)
    return memory, len(features)


def _write_class_layer(memory, destination):
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.layerName = LAYER_NAMES.CLASS_POLYGONS
    options.actionOnExistingFile = QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteFile
    error, message = write_vector_layer(memory, destination, options)
    if error != QgsVectorFileWriter.WriterError.NoError:
        raise ClassWorkspaceError(f"cannot write class workspace layer: {message}")


def initialize_workspace(run_spec, stream, *, replace=False):
    paths = workspace_paths(run_spec)
    if paths["workspace"].is_file() and not replace:
        return load_workspace(run_spec, expected_stream_id=stream.get("stream_id"))
    if stream not in approved_fusion_streams(run_spec, [stream]):
        if run_spec.get("manual_only"):
            raise ClassWorkspaceError("人工工作区需要一个已校验的 ready Fusion 流")
        raise ClassWorkspaceError("workspace requires one ready, approved, regularized Fusion stream")
    report = None if run_spec.get("manual_only") else _validate_boundary_report(stream)
    source, source_path, source_layer_name = _source_layer(stream)
    directory = paths["directory"]
    directory.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".workspace-build-", dir=directory))
    class_records = {}
    total = 0
    try:
        for code in CLASS_ORDER:
            memory, count = _memory_class_layer(source, code, stream["stream_id"])
            temporary_path = temporary / f"class_{code}.gpkg"
            _write_class_layer(memory, temporary_path)
            class_records[str(code)] = {
                "class_code": code,
                "class_name": CLASS_NAMES[code],
                "path": str(class_layer_path(run_spec, code)),
                "layer_name": LAYER_NAMES.CLASS_POLYGONS,
                "feature_count": count,
                "sha256": sha256_file(temporary_path),
                "state": "editing" if count else "unreviewed_empty",
                "confirmed": False,
                "modified": False,
                "updated_at": _now(),
            }
            total += count
        if total != source.featureCount():
            raise ClassWorkspaceError(
                f"14-class split changed feature count: {source.featureCount()} -> {total}"
            )
        for code in CLASS_ORDER:
            os.replace(
                temporary / f"class_{code}.gpkg",
                class_layer_path(run_spec, code),
            )
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    now = _now()
    workspace = {
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "run_id": run_spec["run_id"],
        "baseline_stream_id": stream["stream_id"],
        "baseline_source_path": str(source_path.resolve()),
        "baseline_source_layer": source_layer_name,
        "baseline_source_sha256": sha256_file(source_path),
        "formal_path": str(Path(stream["paths"]["semantic_polygons"]).resolve()),
        "formal_sha256": sha256_file(source_path),
        "boundary_report_path": (
            "" if report is None
            else str(Path(stream["paths"]["boundary_fitting_report"]).resolve())
        ),
        "boundary_report_sha256": (
            "" if report is None
            else sha256_file(stream["paths"]["boundary_fitting_report"])
        ),
        "manual_only": bool(run_spec.get("manual_only")),
        "initialized_at": now,
        "updated_at": now,
        "feature_count": total,
        "locked": False,
        "active_sam_session_id": "",
        "classes": class_records,
    }
    atomic_write_json(paths["workspace"], workspace)
    if not paths["history"].exists():
        paths["history"].touch()
    return workspace


def load_workspace(run_spec, expected_stream_id=""):
    paths = workspace_paths(run_spec)
    if not paths["workspace"].is_file():
        raise ClassWorkspaceError("class workspace has not been initialized")
    with open(paths["workspace"], "r", encoding="utf-8") as handle:
        workspace = json.load(handle)
    if workspace.get("schema_version") != WORKSPACE_SCHEMA_VERSION:
        raise ClassWorkspaceError("class workspace schema is unsupported")
    if workspace.get("run_id") != run_spec.get("run_id"):
        raise ClassWorkspaceError("class workspace belongs to a different run")
    if expected_stream_id and workspace.get("baseline_stream_id") != expected_stream_id:
        raise ClassWorkspaceError("existing workspace uses a different Fusion baseline")
    baseline = Path(str(workspace.get("baseline_source_path") or ""))
    formal = Path(str(workspace.get("formal_path") or ""))
    report = Path(str(workspace.get("boundary_report_path") or ""))
    expected_files = [
        (baseline, workspace.get("baseline_source_sha256"), "Fusion baseline"),
        (formal, workspace.get("formal_sha256"), "formal Fusion"),
    ]
    if not run_spec.get("manual_only"):
        expected_files.append(
            (report, workspace.get("boundary_report_sha256"), "boundary report")
        )
    for path, expected_sha, label in expected_files:
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ClassWorkspaceError(f"{label} is missing or its SHA256 changed: {path}")
    classes = workspace.get("classes") or {}
    if set(classes) != {str(code) for code in CLASS_ORDER}:
        raise ClassWorkspaceError("class workspace does not contain exactly 14 class records")
    for code in CLASS_ORDER:
        record = classes[str(code)]
        path = Path(str(record.get("path") or ""))
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ClassWorkspaceError(
                f"class {code} working layer changed outside the recorded workspace state"
            )
    return workspace


def save_workspace(run_spec, workspace):
    workspace = dict(workspace)
    class_records = dict(workspace.get("classes") or {})
    total = 0
    for code in CLASS_ORDER:
        record = dict(class_records[str(code)])
        path = Path(record["path"])
        layer = QgsVectorLayer(
            f"{path}|layername={record['layer_name']}",
            f"class_{code}_workspace_save",
            "ogr",
        )
        if not layer.isValid():
            raise ClassWorkspaceError(f"cannot reopen class {code} working layer")
        count = layer.featureCount()
        record["feature_count"] = count
        record["sha256"] = sha256_file(path)
        record["updated_at"] = _now()
        class_records[str(code)] = record
        total += count
    workspace["classes"] = class_records
    workspace["feature_count"] = total
    workspace["updated_at"] = _now()
    workspace["locked"] = bool(
        workspace.get("locked")
        or any(record.get("modified") for record in class_records.values())
    )
    atomic_write_json(workspace_paths(run_spec)["workspace"], workspace)
    return workspace


def append_history(run_spec, event, **payload):
    record = {
        "timestamp": _now(),
        "event": str(event),
        "history_id": uuid.uuid4().hex,
        **payload,
    }
    path = workspace_paths(run_spec)["history"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record


def append_sam_session(run_spec, record):
    path = Path(run_spec["run_dir"]) / "refinement" / "sam3" / "sessions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {"timestamp": _now(), **dict(record)}
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return value


def geometry_hash(geometry):
    if geometry is None or geometry.isNull() or geometry.isEmpty():
        return ""
    return hashlib.sha256(bytes(geometry.asWkb())).hexdigest()


def new_object_id(run_spec):
    return f"{run_spec['run_id']}_new_{uuid.uuid4().hex}"


def working_layer(record, display_name="working_class"):
    layer = QgsVectorLayer(
        f"{record['path']}|layername={record['layer_name']}",
        display_name,
        "ogr",
    )
    if not layer.isValid():
        raise ClassWorkspaceError(f"cannot open class working layer: {record['path']}")
    apply_class_constraints(layer, int(record["class_code"]))
    return layer


def source_statistics(layer):
    counts = {"fusion": 0, "sam3": 0, "manual_edited": 0}
    for feature in layer.getFeatures():
        value = str(feature.attribute("geometry_source") or "fusion")
        counts[value] = counts.get(value, 0) + 1
    return counts
