import argparse
import json
from pathlib import Path

import fiona
import numpy as np
import rasterio
from rasterio.transform import from_origin

from fragmentation_postprocess import (
    _activate_review,
    _assemble_vectors,
    _process_mask_partition,
    _process_vector_partition,
    derived_root,
)
from fragmentation_v3 import FIT_VERSION, POLICY_ID, POLICY_VERSION, production_policy
from labeling_tool.core.run_spec import sha256_file


def _write_raster(path, values, *, dtype, nodata):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype=dtype,
        crs="EPSG:3857",
        transform=from_origin(0, 64, 1, 1),
        nodata=nodata,
        compress="deflate",
    ) as destination:
        destination.write(values.astype(dtype), 1)


def _fixture(tmp_path):
    run_dir = tmp_path / "run"
    parts = run_dir / "fusion" / "test_fusion" / "raster_parts"
    parts.mkdir(parents=True)
    mask = parts / "partition_00000_00000_mask.tif"
    confidence = parts / "partition_00000_00000_confidence.tif"
    labels = np.full((64, 64), 1, dtype=np.int16)  # class code 13
    labels[10:25, 10:25] = 3  # class code 31, area above the 200 m2 threshold
    labels[40, 40] = 3  # eligible low-confidence fragment
    _write_raster(mask, labels, dtype="int16", nodata=-1)
    _write_raster(
        confidence,
        np.full(labels.shape, 0.3, dtype=np.float32),
        dtype="float32",
        nodata=np.nan,
    )
    spec = {
        "schema_version": 2,
        "run_id": "fixture_run",
        "run_dir": str(run_dir),
        "streams": [
            {
                "stream_id": "fusion:test_fusion",
                "kind": "fusion",
                "profile_id": "test_fusion",
                "version": "fixture-v1",
            }
        ],
        "spatial_plan_summary": {"partition_count": 1},
    }
    spec_path = run_dir / "run_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return spec, spec_path, mask, confidence


def test_production_policy_is_the_frozen_bounded_v3_contract():
    policy = production_policy()

    assert policy.threshold_for(31) == 200.0
    assert policy.threshold_for(61) == 0.0
    assert policy.maximum_mean_confidence == 0.65
    assert policy.maximum_source_class_loss_fraction == 0.08
    assert policy.maximum_target_class_gain_fraction == 0.08
    assert policy.preserve_elongated_components is True


def test_partition_mask_vector_and_assembly_preserve_originals(tmp_path):
    spec, _spec_path, source_mask, source_confidence = _fixture(tmp_path)
    source_mask_sha = sha256_file(source_mask)
    output_root = derived_root(spec, spec["streams"][0])
    regularized = output_root / "regularized_raster_parts" / source_mask.name
    mask_report_path = output_root / "partition_reports" / "masks" / "part.json"
    mask_report = _process_mask_partition(
        {
            "center_path": str(source_mask),
            "output_path": str(regularized),
            "report_path": str(mask_report_path),
            "path_map": {"0,0": str(source_mask)},
            "buffer_pixels": 8,
            "resume": True,
        }
    )

    assert source_mask_sha == sha256_file(source_mask)
    assert mask_report["changed_pixel_count"] == 1
    with rasterio.open(regularized) as source:
        values = source.read(1)
        assert source.profile["tiled"] is True
        assert source.profile["blockxsize"] == 512
        assert source.profile["blockysize"] == 512
    assert values[40, 40] == 1
    assert np.all(values[10:25, 10:25] == 3)

    resumed = _process_mask_partition(
        {
            "center_path": str(source_mask),
            "output_path": str(regularized),
            "report_path": str(mask_report_path),
            "path_map": {"0,0": str(source_mask)},
            "buffer_pixels": 8,
            "resume": True,
        }
    )
    assert resumed["resumed"] is True

    vector_part = output_root / "polygon_parts" / "part.gpkg"
    vector_report = _process_vector_partition(
        {
            "run_id": spec["run_id"],
            "stream_id": "fusion:test_fusion",
            "profile_id": "test_fusion",
            "model_version": "fixture-v1",
            "partition_id": "partition_00000_00000",
            "mask_path": str(regularized),
            "confidence_path": str(source_confidence),
            "output_path": str(vector_part),
            "report_path": str(output_root / "partition_reports" / "vectors" / "part.json"),
            "resume": True,
        }
    )
    assert vector_report["feature_count"] == 2
    with fiona.open(vector_part, layer="semantic_polygons") as source:
        assert source.schema["geometry"] == "MultiPolygon"
        features = list(source)
        assert {int(item["properties"]["class_code"]) for item in features} == {13, 31}
        assert {
            str(item["properties"]["fit_version"]) for item in features
        } == {FIT_VERSION}

    final = output_root / "semantic_polygons.gpkg"
    assembled = _assemble_vectors(
        final,
        [vector_part],
        crs_wkt=rasterio.crs.CRS.from_epsg(3857).to_wkt(),
    )
    assert assembled["integrity_check"] == "ok"
    assert assembled["feature_count"] == 2
    assert final.is_file()


def test_review_activation_updates_both_v2_stream_collections(tmp_path):
    spec, spec_path, _source_mask, _source_confidence = _fixture(tmp_path)
    run_dir = Path(spec["run_dir"])
    base_stream = {
        "stream_id": "fusion:test_fusion",
        "kind": "fusion",
        "fusion_profile_id": "test_fusion",
        "status": "ready",
        "paths": {},
        "output_sha256": {"semantic_polygons": "old"},
    }
    run_manifest = {
        "schema_version": 2,
        "run_id": spec["run_id"],
        "run_spec": str(spec_path),
        "ready_streams": [dict(base_stream)],
        "streams": [dict(base_stream)],
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest), encoding="utf-8"
    )
    output_root = derived_root(spec, spec["streams"][0])
    output_root.mkdir(parents=True)
    semantic = output_root / "semantic_polygons.gpkg"
    semantic.write_bytes(b"derived")
    report = output_root / "fragmentation_v3_report.json"
    report.write_text("{}", encoding="utf-8")
    manifest = {
        "semantic_polygons": str(semantic),
        "semantic_polygons_sha256": sha256_file(semantic),
        "report_path": str(report),
    }

    _activate_review(
        run_dir,
        stream_id="fusion:test_fusion",
        manifest=manifest,
    )

    activated = json.loads((run_dir / "run_manifest.json").read_text())
    for key in ("ready_streams", "streams"):
        stream = activated[key][0]
        assert stream["review_polygons"] == str(semantic)
        assert stream["review_layer_name"] == "semantic_polygons"
        assert stream["output_sha256"]["review_polygons"] == sha256_file(semantic)
        assert stream["fragmentation_postprocess"]["policy_id"] == POLICY_ID
        assert stream["fragmentation_postprocess"]["policy_version"] == POLICY_VERSION
