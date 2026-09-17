"""Native QGIS integrity cases dispatched by Conda pytest; synthetic data only."""

import gc
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "qgis_plugins"))
try:
    import qgis.core
except ModuleNotFoundError:
    raise SystemExit(77)

from qgis.core import (
    QgsApplication,
    QgsFeature,
    QgsGeometry,
    QgsRectangle,
    QgsVectorFileWriter,
    QgsVectorLayer,
)

from labeling_tool.core import accepted_integrity, accepted_writer, topology_validator
from labeling_tool.core.final_assembler import FINAL_FIELDS
from labeling_tool.core.layer_names import LAYER_NAMES
from labeling_tool.core.qgis_writer import write_vector_layer
from labeling_tool.core.run_spec import (
    CLASS_NAMES,
    atomic_write_json,
    sha256_file,
)


_checks = unittest.TestCase()


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
    with _checks.assertRaisesRegex(accepted_integrity.AcceptedIntegrityError, expected_regex="尚未确认"):
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
    with _checks.assertRaisesRegex(accepted_integrity.AcceptedIntegrityError, expected_regex="类别映射无效"):
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
    with _checks.assertRaisesRegex(accepted_integrity.AcceptedIntegrityError, expected_regex="身份重复"):
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
    with _checks.assertRaisesRegex(accepted_integrity.AcceptedIntegrityError, expected_regex="几何无效"):
        accepted_integrity.audit_accepted_layer(
            invalid_geometry, overlap_tolerance=1.0e-6
        )

    missing_schema = QgsVectorLayer(
        "MultiPolygon?crs=EPSG:4490", "missing_schema", "memory"
    )
    with _checks.assertRaisesRegex(accepted_integrity.AcceptedIntegrityError, expected_regex="缺少标准字段"):
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
    with _checks.assertRaisesRegex(accepted_integrity.AcceptedIntegrityError, expected_regex="同类重叠"):
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
    with _checks.assertRaisesRegex(accepted_integrity.AcceptedIntegrityError, expected_regex="异类重叠"):
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
    with _checks.assertRaisesRegex(
        accepted_integrity.AcceptedIntegrityError,
        expected_regex="final_composite 与现有 accepted_labels 重叠",
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


def test_vector_topology_target_is_the_exact_snapshot_not_selected_tile_union(tmp_path):
    from qgis.core import QgsCoordinateReferenceSystem
    from labeling_tool.core.topology_validator import _selected_tile_target

    snapshot = tmp_path / "range_snapshot.gpkg"
    layer = QgsVectorLayer("Polygon?crs=EPSG:3857", "range_mask", "memory")
    feature = QgsFeature()
    feature.setGeometry(QgsGeometry.fromRect(QgsRectangle(10, 10, 30, 30)))
    assert layer.dataProvider().addFeature(feature)
    _write_layer(layer, snapshot, "range_mask")
    spec = {
        "range_selection": {
            "mode": "vector_tile_intersection",
            "vector_source": str(snapshot),
            "vector_sha256": sha256_file(snapshot),
        },
        "raster": {"crs": "EPSG:3857"},
        "requested_extent": {"xmin": 0, "ymin": 0, "xmax": 20, "ymax": 20},
    }
    target = _selected_tile_target(spec, QgsCoordinateReferenceSystem("EPSG:3857"))
    bounds = target.boundingBox()
    actual = (bounds.xMinimum(), bounds.yMinimum(), bounds.xMaximum(), bounds.yMaximum())
    assert all(math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-12)
               for value, expected in zip(actual, (10, 10, 20, 20))), actual


if __name__ == "__main__":
    from qgis.PyQt.QtCore import QCoreApplication, QEvent

    application = QgsApplication([], False)
    application.initQgis()
    with tempfile.TemporaryDirectory(prefix="loess-integrity-") as temporary:
        name = sys.argv[2]
        case = globals()[name]
        if name == "test_accepted_audit_checks_review_identity_and_overlap":
            case()
        else:
            case(Path(temporary))
        # Dispose provider connections before QGIS's registries are torn down.
        gc.collect()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    application.exitQgis()
    print(name + ": passed", flush=True)
