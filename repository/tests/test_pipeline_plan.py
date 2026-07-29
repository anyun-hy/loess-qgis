from labeling_tool.core.pipeline_plan import build_pipeline_steps


def _catalog(tmp_path):
    def paths(name):
        root = tmp_path / name
        return {
            "tile_mask_dir": str(root / "masks"),
            "tile_confidence_dir": str(root / "confidence"),
            "tile_score_dir": str(root / "scores"),
            "mask_mosaic": str(root / "mask.tif"),
            "confidence_mosaic": str(root / "conf.tif"),
            "probability_mosaic": str(root / "probabilities.tif"),
            "semantic_polygons_raw": str(root / "semantic_raw.gpkg"),
            "semantic_polygons": str(root / "semantic.gpkg"),
            "boundary_regularization_report": str(root / "regularization.json"),
            "difference_polygons": str(root / "candidates.gpkg"),
        }
    return {
        "run_spec": str(tmp_path / "run_spec.json"),
        "streams": [
            {"stream_id": "model:a", "paths": paths("a")},
            {"stream_id": "model:b", "paths": paths("b")},
            {"stream_id": "fusion:f", "paths": paths("f")},
        ],
    }


def _spec(tmp_path):
    return {
        "run_id": "20260713_120102_a1b2c3",
        "run_dir": str(tmp_path),
        "runtime": {"effective_device": "cpu"},
        "tile": {"overlap": 64, "mosaic_strategy": "cosine_probability_blend"},
        "vectorization": {
            "method": "multiclass_subpixel_probability_v1",
            "raw_method": "rasterio_features_shapes",
        },
        "boundary_regularization": {
            "enabled": True,
            "mode": "multiclass_subpixel_probability_v1",
            "interpolation_strength": 1.0,
            "probability_smoothing_sigma": 0.0,
            "coverage_tolerance_px": 1.0,
            "max_deviation_px": 1.5,
            "stripe_rows": 128,
            "qsdk_noninferiority_margin_px": 0.5,
            "preserve_outer_boundary": True,
            "natural_smoothing": False,
        },
        "class_mapping_snapshot": str(tmp_path / "classes.json"),
        "models": [
            {"model_id": "a", "version": "1"},
            {"model_id": "b", "version": "2"},
        ],
        "fusion": {
            "profile_id": "f",
            "snapshot_path": str(tmp_path / "profile.json"),
            "required_model_ids": ["a", "b"],
        },
    }


def test_pipeline_runs_all_model_batches_before_stream_build_and_fusion(tmp_path):
    steps = build_pipeline_steps(_spec(tmp_path), _catalog(tmp_path))
    labels = [step["label"] for step in steps]
    assert labels[:2] == ["model_batch:a", "model_batch:b"]
    assert labels.index("fusion_batch:f") > labels.index("difference:model:b")
    assert labels[-1] == "difference:fusion:f"
    assert all(step.get("stream_id") for step in steps)
    mosaics = [step for step in steps if step["label"].startswith("mosaic:")]
    assert len(mosaics) == 3
    for step in mosaics:
        assert "--score_dir" in step["args"]
        assert "--output_probabilities" in step["args"]
        assert "--strategy" in step["args"]
        assert "cosine_probability_blend" in step["args"]
    polygonize = [step for step in steps if step["label"].startswith("polygonize:")]
    assert len(polygonize) == 3
    for step in polygonize:
        assert "--simplify-tolerance-px" not in step["args"]
        assert any(str(value).endswith("semantic_raw.gpkg") for value in step["args"])
    subpixel = [step for step in steps if step["label"].startswith("subpixel_vectorize:")]
    assert len(subpixel) == 3
    for step in subpixel:
        assert step["script"] == "run_subpixel_vectorize.sh"
        assert "--probabilities" in step["args"]
        assert "--interpolation-strength" in step["args"]
        assert "1.0" in step["args"]


def test_resume_flag_and_valid_stream_skip_are_explicit(tmp_path):
    steps = build_pipeline_steps(
        _spec(tmp_path), _catalog(tmp_path), resume=True, ready_stream_ids=["model:a"]
    )
    labels = [step["label"] for step in steps]
    assert not any(label.endswith(":a") for label in labels)
    assert "--resume" in next(step for step in steps if step["label"] == "model_batch:b")["args"]
    fusion = next(step for step in steps if step["label"] == "fusion_batch:f")
    assert fusion["requires_streams"] == ["model:a", "model:b"]
    assert "--resume" in fusion["args"]
