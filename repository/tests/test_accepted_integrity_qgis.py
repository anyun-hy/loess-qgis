import pytest


pytest.importorskip("qgis.core", exc_type=ImportError)

from qgis.core import (  # noqa: E402
    QgsApplication,
    QgsFeature,
    QgsGeometry,
    QgsRectangle,
    QgsVectorFileWriter,
    QgsVectorLayer,
)

from labeling_tool.core import accepted_integrity, accepted_writer, topology_validator  # noqa: E402
from labeling_tool.core.final_assembler import FINAL_FIELDS  # noqa: E402
from labeling_tool.core.layer_names import LAYER_NAMES  # noqa: E402
from labeling_tool.core.qgis_writer import write_vector_layer  # noqa: E402
from labeling_tool.core.run_spec import (  # noqa: E402
    CLASS_NAMES,
    atomic_write_json,
    sha256_file,
)


@pytest.fixture(scope="session", autouse=True)
def qgis_application():
    existing = QgsApplication.instance()
    owns_application = existing is None
    application = existing or QgsApplication([], False)
    if owns_application:
        application.initQgis()
    yield application
    if owns_application:
        application.exitQgis()


def _accepted_layer(name="accepted_fixture"):
    layer = QgsVectorLayer("MultiPolygon?crs=EPSG:4490", name, "memory")
    layer.dataProvider().addAttributes(accepted_writer.ACCEPTED_FIELDS_QGS)
    layer.updateFields()
    return layer


def _final_layer(name="final_fixture"):
    layer = QgsVectorLayer("MultiPolygon?crs=EPSG:4490", name, "memory")
    layer.dataProvider().addAttributes(FINAL_FIELDS)
    layer.updateFields()
    return layer


def _add_feature(
    layer,
    bounds,
    *,
    run_id,
    object_id,
    part_id="000",
    class_code=12,
    class_name=None,
    reviewed=1,
    geometry_wkt=None,
):
    feature = QgsFeature(layer.fields())
    geometry = (
        QgsGeometry.fromWkt(geometry_wkt)
        if geometry_wkt
        else QgsGeometry.fromRect(QgsRectangle(*bounds))
    )
    geometry.convertToMultiType()
    feature.setGeometry(geometry)
    values = {
        "run_id": run_id,
        "object_id": object_id,
        "part_id": part_id,
        "class_code": class_code,
        "class_name": class_name or CLASS_NAMES[class_code],
        "confidence_mean": 0.9,
        "confidence_std": 0.01,
        "baseline_stream_id": "fusion:fixture",
        "source_stream_id": "fusion:fixture",
        "source": "class_working",
        "geometry_source": "fusion",
        "geometry_revision": 0,
        "edit_base": "",
        "sam_session_id": "",
        "sam_score": 0.0,
        "model_version": "fixture",
        "fusion_profile_id": "fixture",
        "sam_version": "",
        "reviewed": reviewed,
        "created_at": "2026-07-29T00:00:00+09:00",
        "updated_at": "2026-07-29T00:00:00+09:00",
    }
    feature.setAttributes(
        [values.get(field.name(), "") for field in layer.fields()]
    )
    assert layer.dataProvider().addFeature(feature)


def _write_layer(layer, path, layer_name):
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.layerName = layer_name
    options.actionOnExistingFile = (
        QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteFile
    )
    error, message = write_vector_layer(layer, path, options)
    assert error == QgsVectorFileWriter.WriterError.NoError, message


def test_accepted_audit_checks_review_identity_and_overlap():
    clean = _accepted_layer()
    _add_feature(
        clean,
        (0, 0, 1, 1),
        run_id="old_run",
        object_id="logical_object",
        part_id="000",
    )
    _add_feature(
        clean,
        (1, 0, 2, 1),
        run_id="old_run",
        object_id="logical_object",
        part_id="001",
    )
    report = accepted_integrity.audit_accepted_layer(
        clean, overlap_tolerance=1.0e-6, expected_crs=clean.crs()
    )
    assert report["status"] == "passed"
    assert report["feature_count"] == 2

    unreviewed = _accepted_layer("unreviewed")
    _add_feature(
        unreviewed,
        (0, 0, 1, 1),
        run_id="old_run",
        object_id="unreviewed",
        reviewed=0,
    )
    with pytest.raises(accepted_integrity.AcceptedIntegrityError, match="尚未确认"):
        accepted_integrity.audit_accepted_layer(
            unreviewed, overlap_tolerance=1.0e-6
        )

    bad_class = _accepted_layer("bad_class")
    _add_feature(
        bad_class,
        (0, 0, 1, 1),
        run_id="old_run",
        object_id="bad_class",
        class_name="错误类别",
    )
    with pytest.raises(accepted_integrity.AcceptedIntegrityError, match="类别映射无效"):
        accepted_integrity.audit_accepted_layer(
            bad_class, overlap_tolerance=1.0e-6
        )

    duplicate = _accepted_layer("duplicate_identity")
    _add_feature(
        duplicate,
        (0, 0, 1, 1),
        run_id="old_run",
        object_id="duplicate",
    )
    _add_feature(
        duplicate,
        (2, 0, 3, 1),
        run_id="old_run",
        object_id="duplicate",
    )
    with pytest.raises(accepted_integrity.AcceptedIntegrityError, match="身份重复"):
        accepted_integrity.audit_accepted_layer(
            duplicate, overlap_tolerance=1.0e-6
        )

    invalid_geometry = _accepted_layer("invalid_geometry")
    _add_feature(
        invalid_geometry,
        (0, 0, 1, 1),
        run_id="old_run",
        object_id="invalid_geometry",
        geometry_wkt="POLYGON((0 0,2 2,0 2,2 0,0 0))",
    )
    with pytest.raises(accepted_integrity.AcceptedIntegrityError, match="几何无效"):
        accepted_integrity.audit_accepted_layer(
            invalid_geometry, overlap_tolerance=1.0e-6
        )

    missing_schema = QgsVectorLayer(
        "MultiPolygon?crs=EPSG:4490", "missing_schema", "memory"
    )
    with pytest.raises(accepted_integrity.AcceptedIntegrityError, match="缺少标准字段"):
        accepted_integrity.audit_accepted_layer(
            missing_schema, overlap_tolerance=1.0e-6
        )

    same_class_overlap = _accepted_layer("same_class_overlap")
    _add_feature(
        same_class_overlap, (0, 0, 2, 2), run_id="old_run", object_id="first"
    )
    _add_feature(
        same_class_overlap, (1, 1, 3, 3), run_id="old_run", object_id="second"
    )
    with pytest.raises(accepted_integrity.AcceptedIntegrityError, match="同类重叠"):
        accepted_integrity.audit_accepted_layer(
            same_class_overlap, overlap_tolerance=1.0e-6
        )

    cross_class_overlap = _accepted_layer("cross_class_overlap")
    _add_feature(
        cross_class_overlap,
        (0, 0, 2, 2),
        run_id="old_run",
        object_id="first",
    )
    _add_feature(
        cross_class_overlap,
        (1, 1, 3, 3),
        run_id="old_run",
        object_id="second",
        class_code=31,
    )
    with pytest.raises(accepted_integrity.AcceptedIntegrityError, match="异类重叠"):
        accepted_integrity.audit_accepted_layer(
            cross_class_overlap, overlap_tolerance=1.0e-6
        )


def test_topology_reports_and_writer_blocks_existing_accepted_overlap(tmp_path):
    accepted_memory = _accepted_layer()
    _add_feature(
        accepted_memory,
        (0, 0, 2, 2),
        run_id="old_run",
        object_id="accepted_object",
    )
    accepted_path = tmp_path / "accepted_labels.gpkg"
    _write_layer(accepted_memory, accepted_path, LAYER_NAMES.ACCEPTED)

    final_memory = _final_layer()
    _add_feature(
        final_memory,
        (1, 1, 3, 3),
        run_id="new_run",
        object_id="new_object",
    )
    final_path = tmp_path / "final_composite.gpkg"
    _write_layer(final_memory, final_path, LAYER_NAMES.FINAL_COMPOSITE)

    run_dir = tmp_path / "runs" / "new_run"
    (run_dir / "final").mkdir(parents=True)
    spec = {
        "schema_version": 2,
        "run_id": "new_run",
        "run_dir": str(run_dir),
        "raster": {
            "crs": "EPSG:4490",
            "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 4.0],
        },
        "requested_extent": {
            "xmin": 0.0,
            "ymin": 0.0,
            "xmax": 4.0,
            "ymax": 4.0,
        },
        "range_selection": {"mode": "extent"},
        "accepted_gpkg": str(run_dir / "accepted_snapshot.gpkg"),
        "accepted_target_gpkg": str(accepted_path),
    }
    spec_path = run_dir / "run_spec.json"
    atomic_write_json(spec_path, spec)
    manifest_path = run_dir / "run_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 2,
            "run_id": "new_run",
            "run_spec": str(spec_path),
            "run_spec_sha256": sha256_file(spec_path),
            "status": "ready",
            "streams": [
                {"stream_id": "fusion:fixture", "kind": "fusion", "status": "ready"}
            ],
        },
    )

    accepted_layer = QgsVectorLayer(
        f"{accepted_path}|layername={LAYER_NAMES.ACCEPTED}",
        "accepted_from_disk",
        "ogr",
    )
    _issues_path, _issue_count, counts = topology_validator.validate_topology(
        spec, final_path, accepted_layer
    )
    assert counts["accepted_overlap"] == 1

    before_sha = sha256_file(accepted_path)
    with pytest.raises(
        accepted_integrity.AcceptedIntegrityError,
        match="final_composite 与现有 accepted_labels 重叠",
    ):
        accepted_writer.append_final_to_accepted(
            final_path, accepted_path, manifest_path
        )
    assert sha256_file(accepted_path) == before_sha

    fresh_run_dir = tmp_path / "runs" / "fresh_target"
    fresh_run_dir.mkdir(parents=True)
    fresh_target = tmp_path / "fresh_accepted_labels.gpkg"
    fresh_spec = {
        **spec,
        "run_dir": str(fresh_run_dir),
        "accepted_gpkg": str(fresh_run_dir / "accepted_snapshot.gpkg"),
        "accepted_target_gpkg": str(fresh_target),
    }
    fresh_spec_path = fresh_run_dir / "run_spec.json"
    atomic_write_json(fresh_spec_path, fresh_spec)
    fresh_manifest_path = fresh_run_dir / "run_manifest.json"
    atomic_write_json(
        fresh_manifest_path,
        {
            "schema_version": 2,
            "run_id": "new_run",
            "run_spec": str(fresh_spec_path),
            "run_spec_sha256": sha256_file(fresh_spec_path),
            "status": "ready",
            "streams": [
                {"stream_id": "fusion:fixture", "kind": "fusion", "status": "ready"}
            ],
        },
    )
    assert accepted_writer.append_final_to_accepted(
        final_path, fresh_target, fresh_manifest_path
    ) == 1
    fresh_layer = QgsVectorLayer(
        f"{fresh_target}|layername={LAYER_NAMES.ACCEPTED}",
        "fresh_accepted_from_disk",
        "ogr",
    )
    fresh_report = accepted_integrity.audit_accepted_layer(
        fresh_layer, overlap_tolerance=1.0e-6
    )
    assert fresh_report["feature_count"] == 1
