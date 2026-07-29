import datetime
import hashlib
import json
from pathlib import Path

import pytest

from labeling_tool.core.result_catalog import (
    create_result_catalog,
    record_stream_outputs,
    update_stream,
    valid_ready_stream_ids,
)
from labeling_tool.core.run_spec import (
    RunSpecError,
    create_run_spec,
    new_run_id,
    reserve_run_directory,
)


def _file(path, value=b"fixture"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return hashlib.sha256(value).hexdigest()


def _model(tmp_path, model_id="model_a"):
    path = tmp_path / f"{model_id}.torchscript.pt"
    sha = _file(path, model_id.encode())
    return {
        "model_id": model_id,
        "display_name": model_id,
        "version": "fixture-v1",
        "artifact": path.name,
        "artifact_path": str(path),
        "sha256": sha,
    }


def _tile(tmp_path):
    path = tmp_path / "tile_0_0.tif"
    _file(path, b"raster")
    return {
        "row": 0,
        "col": 0,
        "tile_path": str(path),
        "width": 512,
        "height": 512,
        "bounds": [100, 30, 101, 31],
    }


def _profile(models):
    return {
        "schema_version": 1,
        "profile_id": "fixture_profile",
        "status": "approved",
        "strategy": "equal_probability_average",
        "class_order": [12, 13, 21, 31, 32, 33, 43, 51, 52, 53, 54, 61, 62, 71],
        "input": {"height": 512, "width": 512, "channels": 3, "dtype": "float32"},
        "models": [{
            "model_id": model["model_id"],
            "artifact": model["artifact"],
            "sha256": model["sha256"],
            "temperature": 1.0,
        } for model in models],
        "weights": [[1.0 / len(models) for _ in models] for _ in range(14)],
        "approval": {
            "passed": True,
            "criterion": "fusion.test_miou > exported_swin_baseline.test_miou",
        },
    }


def test_run_id_has_timestamp_and_random_token():
    run_id = new_run_id(datetime.datetime(2026, 7, 13, 12, 1, 2), "a1b2c3")
    assert run_id == "20260713_120102_a1b2c3"


def test_create_run_spec_snapshots_inputs_and_refuses_overwrite(tmp_path):
    kwargs = dict(
        output_root=tmp_path / "output",
        raster_path=tmp_path / "source.tif",
        raster_crs="EPSG:4490",
        requested_extent=[100, 30, 100.5, 30.5],
        processing_extent=[100, 30, 101, 31],
        tiles=[_tile(tmp_path)],
        models=[_model(tmp_path)],
        effective_device="cpu",
        overlap=64,
        run_id="20260713_120102_a1b2c3",
    )
    spec, path = create_run_spec(**kwargs)

    assert path.is_file()
    assert spec["class_mapping_snapshot"].endswith("class_mapping_snapshot.json")
    assert spec["tiles"][0]["sha256"]
    assert spec["models"][0]["sha256"]
    assert spec["vectorization"] == {
        "method": "multiclass_subpixel_probability_v1",
        "raw_method": "rasterio_features_shapes",
    }
    assert spec["boundary_regularization"]["mode"] == "multiclass_subpixel_probability_v1"
    assert spec["boundary_regularization"]["coverage_tolerance_px"] == 1.0
    assert spec["boundary_regularization"]["qsdk_noninferiority_margin_px"] == 0.5
    assert spec["boundary_regularization"]["max_deviation_px"] == 1.5
    assert (path.parent / "classes").is_dir()
    assert (path.parent / "refinement" / "sam3").is_dir()
    assert (path.parent / "config_snapshot.json").is_file()
    assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == spec["run_id"]
    with pytest.raises(RunSpecError, match="refusing to overwrite"):
        create_run_spec(**kwargs)


def test_reserved_run_accepts_extracted_tiles_then_becomes_immutable(tmp_path):
    run_id, run_dir = reserve_run_directory(
        tmp_path / "output", "20260713_120102_a1b2c3"
    )
    tile = run_dir / "tmp" / "tiles" / "tile_0_0.tif"
    _file(tile, b"raster")
    spec, path = create_run_spec(
        output_root=tmp_path / "output",
        reserved_run_dir=run_dir,
        run_id=run_id,
        raster_path=tmp_path / "source.tif",
        raster_crs="EPSG:4490",
        requested_extent=[0, 0, 1, 1],
        processing_extent=[0, 0, 1, 1],
        tiles=[{"row": 0, "col": 0, "tile_path": tile, "width": 512, "height": 512}],
        models=[_model(tmp_path)],
        effective_device="cpu",
    )
    assert spec["run_id"] == run_id
    assert path.parent == run_dir
    assert not (run_dir / ".run-reservation").exists()
    with pytest.raises(RunSpecError, match="unused reservation marker"):
        create_run_spec(
            output_root=tmp_path / "output",
            reserved_run_dir=run_dir,
            run_id=run_id,
            raster_path=tmp_path / "source.tif",
            raster_crs="EPSG:4490",
            requested_extent=[0, 0, 1, 1],
            processing_extent=[0, 0, 1, 1],
            tiles=[{"row": 0, "col": 0, "tile_path": tile, "width": 512, "height": 512}],
            models=[_model(tmp_path, "model_b")],
            effective_device="cpu",
        )


def test_model_sha_mismatch_is_rejected(tmp_path):
    model = _model(tmp_path)
    model["sha256"] = "0" * 64
    with pytest.raises(RunSpecError, match="SHA256 mismatch"):
        create_run_spec(
            output_root=tmp_path / "output",
            raster_path=tmp_path / "source.tif",
            raster_crs="EPSG:4490",
            requested_extent=[0, 0, 1, 1],
            processing_extent=[0, 0, 1, 1],
            tiles=[_tile(tmp_path)],
            models=[model],
            effective_device="cpu",
            run_id="20260713_120102_a1b2c3",
        )


def test_run_spec_rejects_invalid_fusion_before_snapshot(tmp_path):
    model = _model(tmp_path)
    profile = _profile([model])
    profile["approval"]["criterion"] = "informal fixture criterion"
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(RunSpecError, match="approval criterion"):
        create_run_spec(
            output_root=tmp_path / "output",
            raster_path=tmp_path / "source.tif",
            raster_crs="EPSG:4490",
            requested_extent=[0, 0, 1, 1],
            processing_extent=[0, 0, 1, 1],
            tiles=[_tile(tmp_path)],
            models=[model],
            effective_device="cpu",
            fusion_profile_path=profile_path,
            run_id="20260713_120102_a1b2c3",
        )


def test_result_catalog_keeps_independent_stream_paths_and_status(tmp_path):
    spec, path = create_run_spec(
        output_root=tmp_path / "output",
        raster_path=tmp_path / "source.tif",
        raster_crs="EPSG:4490",
        requested_extent=[0, 0, 1, 1],
        processing_extent=[0, 0, 1, 1],
        tiles=[_tile(tmp_path)],
        models=[_model(tmp_path, "model_a"), _model(tmp_path, "model_b")],
        effective_device="cpu",
        run_id="20260713_120102_a1b2c3",
    )
    catalog = create_result_catalog(spec)
    assert [item["stream_id"] for item in catalog["streams"]] == [
        "model:model_a", "model:model_b"
    ]
    assert catalog["streams"][0]["paths"]["mask_mosaic"] != catalog["streams"][1]["paths"]["mask_mosaic"]
    update_stream(catalog, "model:model_a", status="ready")
    update_stream(catalog, "model:model_b", status="failed", failure_count=1, error="tile failed")
    stored = json.loads((path.parent / "run_manifest.json").read_text(encoding="utf-8"))
    assert stored["status"] == "failed"
    assert stored["streams"][1]["failure_count"] == 1


def test_resume_validates_inputs_outputs_and_difference_review_layer(tmp_path):
    model = _model(tmp_path)
    spec, path = create_run_spec(
        output_root=tmp_path / "output",
        raster_path=tmp_path / "source.tif",
        raster_crs="EPSG:4490",
        requested_extent=[0, 0, 1, 1],
        processing_extent=[0, 0, 1, 1],
        tiles=[_tile(tmp_path)],
        models=[model],
        effective_device="cpu",
        run_id="20260713_120102_a1b2c3",
    )
    catalog = create_result_catalog(spec)
    stream = catalog["streams"][0]
    for key in ("mask_mosaic", "confidence_mosaic"):
        _file(Path(stream["paths"][key]), key.encode())
    raw_sha = _file(Path(stream["paths"]["semantic_polygons_raw"]), b"raw")
    formal_sha = _file(Path(stream["paths"]["semantic_polygons"]), b"formal")
    probability_sha = _file(Path(stream["paths"]["probability_mosaic"]), b"probability")
    Path(stream["paths"]["boundary_regularization_report"]).write_text(json.dumps({
        "status": "passed",
        "validation": {"passed": True},
        "input_sha256": raw_sha,
        "output_sha256": formal_sha,
        "probability_mosaic_sha256": probability_sha,
    }), encoding="utf-8")
    review_path = Path(stream["paths"]["difference_polygons"])
    _file(review_path, b"difference")
    stream["review_polygons"] = str(review_path)
    stream["review_layer_name"] = "semantic_candidates"
    record_stream_outputs(catalog, stream["stream_id"])

    assert valid_ready_stream_ids(catalog) == ("model:model_a",)

    review_path.write_bytes(b"mutated difference")
    assert valid_ready_stream_ids(catalog) == ()
    review_path.write_bytes(b"difference")
    assert valid_ready_stream_ids(catalog) == ("model:model_a",)

    Path(spec["tiles"][0]["path"]).write_bytes(b"mutated tile")
    assert valid_ready_stream_ids(catalog) == ()
    Path(spec["tiles"][0]["path"]).write_bytes(b"raster")
    assert valid_ready_stream_ids(catalog) == ("model:model_a",)

    Path(model["artifact_path"]).write_bytes(b"mutated model")
    assert valid_ready_stream_ids(catalog) == ()


def test_resume_rejects_mutated_fusion_profile_snapshot(tmp_path):
    model = _model(tmp_path)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_profile([model])), encoding="utf-8")
    spec, _ = create_run_spec(
        output_root=tmp_path / "output",
        raster_path=tmp_path / "source.tif",
        raster_crs="EPSG:4490",
        requested_extent=[0, 0, 1, 1],
        processing_extent=[0, 0, 1, 1],
        tiles=[_tile(tmp_path)],
        models=[model],
        effective_device="cpu",
        fusion_profile_path=profile_path,
        run_id="20260713_120102_a1b2c3",
    )
    catalog = create_result_catalog(spec)
    for stream in catalog["streams"]:
        for key in ("mask_mosaic", "confidence_mosaic"):
            _file(Path(stream["paths"][key]), f"{stream['stream_id']}:{key}".encode())
        raw_sha = _file(
            Path(stream["paths"]["semantic_polygons_raw"]),
            f"{stream['stream_id']}:raw".encode(),
        )
        formal_sha = _file(
            Path(stream["paths"]["semantic_polygons"]),
            f"{stream['stream_id']}:formal".encode(),
        )
        probability_sha = _file(
            Path(stream["paths"]["probability_mosaic"]),
            f"{stream['stream_id']}:probability".encode(),
        )
        Path(stream["paths"]["boundary_regularization_report"]).write_text(json.dumps({
            "status": "passed",
            "validation": {"passed": True},
            "input_sha256": raw_sha,
            "output_sha256": formal_sha,
            "probability_mosaic_sha256": probability_sha,
        }), encoding="utf-8")
        record_stream_outputs(catalog, stream["stream_id"])
    assert set(valid_ready_stream_ids(catalog)) == {
        "model:model_a", "fusion:fixture_profile"
    }

    Path(spec["fusion"]["snapshot_path"]).write_text("{}", encoding="utf-8")
    assert valid_ready_stream_ids(catalog) == ()
