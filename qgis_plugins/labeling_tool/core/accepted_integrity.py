"""Strict integrity and overlap checks for the long-lived accepted label store."""

from __future__ import annotations

from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsProject,
    QgsSpatialIndex,
    QgsWkbTypes,
)

from .run_spec import CLASS_NAMES


REQUIRED_ACCEPTED_FIELDS = (
    "run_id",
    "object_id",
    "part_id",
    "class_code",
    "class_name",
    "confidence_mean",
    "confidence_std",
    "baseline_stream_id",
    "source_stream_id",
    "source",
    "geometry_source",
    "geometry_revision",
    "edit_base",
    "sam_session_id",
    "sam_score",
    "model_version",
    "fusion_profile_id",
    "sam_version",
    "reviewed",
    "created_at",
    "updated_at",
)


class AcceptedIntegrityError(ValueError):
    pass


def _attribute(feature, name, default=""):
    index = feature.fieldNameIndex(name)
    if index < 0:
        return default
    value = feature.attribute(index)
    return default if value is None else value


def strict_overlap_tolerance(run_spec):
    """Return a numerical-noise tolerance, not the one-pixel topology tolerance."""
    transform = (run_spec.get("raster") or {}).get("transform") or []
    if len(transform) != 6:
        return 1.0e-18
    pixel_area = abs(
        float(transform[0]) * float(transform[4])
        - float(transform[1]) * float(transform[3])
    )
    return max(pixel_area * 1.0e-6, 1.0e-18)


def audit_accepted_layer(layer, *, overlap_tolerance, expected_crs=None):
    """Validate the complete accepted store before it can affect a Run or write."""
    if layer is None or not layer.isValid():
        raise AcceptedIntegrityError("accepted_labels 图层无效或无法打开")
    if not layer.crs().isValid():
        raise AcceptedIntegrityError("accepted_labels 缺少有效 CRS")
    if expected_crs is not None and expected_crs.isValid() and layer.crs() != expected_crs:
        raise AcceptedIntegrityError(
            "accepted_labels CRS 与本次影像不一致: "
            f"{layer.crs().authid()}/{expected_crs.authid()}"
        )
    if QgsWkbTypes.geometryType(layer.wkbType()) != Qgis.GeometryType.Polygon:
        raise AcceptedIntegrityError("accepted_labels 必须是 Polygon/MultiPolygon 图层")
    existing_fields = {field.name() for field in layer.fields()}
    missing = [
        name for name in REQUIRED_ACCEPTED_FIELDS if name not in existing_fields
    ]
    if missing:
        raise AcceptedIntegrityError(
            "accepted_labels 缺少标准字段: " + ", ".join(missing)
        )

    errors = []
    identities = set()
    index = QgsSpatialIndex()
    feature_count = 0
    for feature in layer.getFeatures():
        feature_count += 1
        geometry = feature.geometry()
        if (
            geometry is None
            or geometry.isNull()
            or geometry.isEmpty()
            or QgsWkbTypes.geometryType(geometry.wkbType())
            != Qgis.GeometryType.Polygon
            or not geometry.isGeosValid()
            or geometry.area() <= 0
        ):
            errors.append(f"FID {feature.id()} 的几何无效、为空或非正面积")
            continue
        run_id = str(_attribute(feature, "run_id", "") or "").strip()
        object_id = str(_attribute(feature, "object_id", "") or "").strip()
        part_id = str(_attribute(feature, "part_id", "") or "").strip()
        identity = (run_id, object_id, part_id)
        if not run_id or not object_id or not part_id:
            errors.append(f"FID {feature.id()} 的 run_id/object_id/part_id 为空")
        if identity in identities:
            errors.append(f"确认对象身份重复: {identity}")
        identities.add(identity)
        try:
            class_code = int(_attribute(feature, "class_code", -1))
        except (TypeError, ValueError):
            class_code = -1
        class_name = str(_attribute(feature, "class_name", "") or "")
        if CLASS_NAMES.get(class_code) != class_name:
            errors.append(
                f"FID {feature.id()} 的类别映射无效: {class_code}/{class_name}"
            )
        try:
            reviewed = int(_attribute(feature, "reviewed", 0) or 0)
        except (TypeError, ValueError):
            reviewed = 0
        if reviewed != 1:
            errors.append(f"FID {feature.id()} 尚未确认: reviewed={reviewed}")
        index.addFeature(feature)

    if errors:
        preview = "; ".join(errors[:20])
        suffix = f"; 另有 {len(errors) - 20} 项" if len(errors) > 20 else ""
        raise AcceptedIntegrityError(
            f"accepted_labels 完整性审计失败（{len(errors)} 项）: {preview}{suffix}"
        )

    overlap_errors = []
    seen_pairs = set()
    for feature in layer.getFeatures():
        geometry = feature.geometry()
        for other_id in index.intersects(geometry.boundingBox()):
            pair = tuple(sorted((int(feature.id()), int(other_id))))
            if int(other_id) == int(feature.id()) or pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            other = layer.getFeature(other_id)
            if not other.isValid():
                continue
            intersection = geometry.intersection(other.geometry())
            if (
                intersection is None
                or intersection.isNull()
                or intersection.isEmpty()
                or intersection.area() <= float(overlap_tolerance)
            ):
                continue
            same_class = int(_attribute(feature, "class_code", -1)) == int(
                _attribute(other, "class_code", -1)
            )
            overlap_errors.append(
                "{}重叠 {} / {}，面积={:.12g}".format(
                    "同类" if same_class else "异类",
                    _attribute(feature, "object_id", feature.id()),
                    _attribute(other, "object_id", other.id()),
                    intersection.area(),
                )
            )
    if overlap_errors:
        preview = "; ".join(overlap_errors[:20])
        suffix = (
            f"; 另有 {len(overlap_errors) - 20} 项"
            if len(overlap_errors) > 20
            else ""
        )
        raise AcceptedIntegrityError(
            "accepted_labels 存在空间重叠"
            f"（{len(overlap_errors)} 对）: {preview}{suffix}"
        )

    return {
        "status": "passed",
        "feature_count": feature_count,
        "overlap_pair_count": 0,
        "overlap_tolerance": float(overlap_tolerance),
        "crs": layer.crs().authid(),
    }


def assert_no_accepted_overlap(
    candidate_layer,
    accepted_layer,
    *,
    overlap_tolerance,
    raise_on_overlap=True,
):
    """Find candidate/accepted intersections and optionally raise immediately."""
    if (
        candidate_layer is None
        or not candidate_layer.isValid()
        or accepted_layer is None
        or not accepted_layer.isValid()
        or accepted_layer.featureCount() == 0
    ):
        return []

    accepted_index = QgsSpatialIndex()
    for accepted_feature in accepted_layer.getFeatures():
        accepted_index.addFeature(accepted_feature)

    to_accepted = None
    to_candidate = None
    if candidate_layer.crs() != accepted_layer.crs():
        to_accepted = QgsCoordinateTransform(
            candidate_layer.crs(), accepted_layer.crs(), QgsProject.instance()
        )
        to_candidate = QgsCoordinateTransform(
            accepted_layer.crs(), candidate_layer.crs(), QgsProject.instance()
        )

    overlaps = []
    for candidate in candidate_layer.getFeatures():
        candidate_geometry = QgsGeometry(candidate.geometry())
        if (
            candidate_geometry is None
            or candidate_geometry.isNull()
            or candidate_geometry.isEmpty()
            or not candidate_geometry.isGeosValid()
        ):
            continue
        query_geometry = QgsGeometry(candidate_geometry)
        if to_accepted is not None:
            query_geometry.transform(to_accepted)
        for accepted_id in accepted_index.intersects(query_geometry.boundingBox()):
            accepted = accepted_layer.getFeature(accepted_id)
            if not accepted.isValid():
                continue
            intersection = query_geometry.intersection(accepted.geometry())
            if intersection is None or intersection.isNull() or intersection.isEmpty():
                continue
            if to_candidate is not None:
                intersection.transform(to_candidate)
            if intersection.area() <= float(overlap_tolerance):
                continue
            overlaps.append((candidate, accepted, intersection))

    if overlaps and raise_on_overlap:
        examples = ", ".join(
            "{}/{}".format(
                _attribute(candidate, "object_id", candidate.id()),
                _attribute(accepted, "object_id", accepted.id()),
            )
            for candidate, accepted, _geometry in overlaps[:10]
        )
        raise AcceptedIntegrityError(
            "final_composite 与现有 accepted_labels 重叠"
            f"（{len(overlaps)} 对）: {examples}"
        )
    return overlaps
