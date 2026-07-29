from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_single_model_runtime_files_are_removed():
    obsolete = (
        ROOT / "inference_scripts" / "predict_semantic.py",
        ROOT / "inference_scripts" / "run_semantic.sh",
        ROOT / "inference_scripts" / "semantic_model.py",
        ROOT / "inference_scripts" / "run_sam3.sh",
        ROOT / "inference_scripts" / "sam3_class_batch.py",
        ROOT / "inference_scripts" / "run_sam3_class.sh",
        ROOT / "qgis_plugins" / "labeling_tool" / "core" / "inference_runner.py",
        ROOT / "qgis_plugins" / "labeling_tool" / "core" / "sam3_job_runner.py",
    )
    assert not [str(path.relative_to(ROOT)) for path in obsolete if path.exists()]


def test_legacy_v5_boundary_fitter_is_removed():
    obsolete = (
        ROOT / "inference_scripts" / "boundary_fitting" / "adaptive_fit.py",
        ROOT / "inference_scripts" / "boundary_fitting" / "edge_graph.py",
        ROOT / "inference_scripts" / "boundary_fitting" / "map_precision.py",
        ROOT / "inference_scripts" / "boundary_fitting" / "unit_fitter.py",
        ROOT / "tests" / "test_shared_edge_fitting.py",
    )
    assert not [str(path.relative_to(ROOT)) for path in obsolete if path.exists()]


def test_plugin_core_exports_only_the_async_runtime():
    source = (ROOT / "qgis_plugins" / "labeling_tool" / "core" / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert '"V5AsyncInferenceRunner"' in source
    assert '"InferenceRunner"' not in source
    plugin_source = (ROOT / "qgis_plugins" / "labeling_tool" / "gui" / "main_dock.py").read_text(
        encoding="utf-8"
    )
    assert "semantic_weight" not in plugin_source


def test_runtime_fingerprints_only_current_pipeline_files():
    source = (ROOT / "inference_scripts" / "check_environment.py").read_text(
        encoding="utf-8"
    )
    for name in (
        "semantic_batch.py", "torchscript_runtime.py", "work_package_runtime.py",
        "incremental_fusion.py", "partition_mosaic.py", "assemble_stream.py",
        "polyline_smoother.py", "common_boundary_smoother.py",
        "sam3_interactive_worker.py", "run_work_package.sh", "run_unit_fit.sh",
        "run_finalize_partition_rasters.sh", "run_assemble_stream.sh",
        "scale_acceptance.py", "runtime_metrics.py", "run_scale_acceptance.sh",
        "rasterio_compat.py",
        "run_sam3_interactive_worker.sh",
    ):
        assert f'"{name}"' in source
    for name in (
        "predict_semantic.py", "run_semantic.sh", "semantic_model.py",
        "run_sam3.sh", "sam3_class_batch.py", "run_sam3_class.sh",
        "run_semantic_batch.sh", "run_mosaic.sh", "run_polygonize.sh",
        "run_subpixel_vectorize.sh",
        "boundary_fitting/edge_graph.py", "boundary_fitting/adaptive_fit.py",
        "boundary_fitting/map_precision.py", "boundary_fitting/unit_fitter.py",
    ):
        assert f'"{name}"' not in source


def test_unit_fit_wrapper_uses_the_inference_scripts_directory():
    wrapper = (ROOT / "inference_scripts" / "run_unit_fit.sh").read_text(
        encoding="utf-8"
    )
    assert 'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"' in wrapper
    assert 'dirname "$0")/..' not in wrapper
    assert 'source "$SCRIPT_DIR/config.sh"' in wrapper
    assert 'python "$SCRIPT_DIR/boundary_fitting/unit_runtime.py"' in wrapper


def test_accepted_labels_has_no_direct_candidate_bypass():
    dock = (ROOT / "qgis_plugins" / "labeling_tool" / "gui" / "main_dock.py").read_text(
        encoding="utf-8"
    )
    writer = (ROOT / "qgis_plugins" / "labeling_tool" / "core" / "accepted_writer.py").read_text(
        encoding="utf-8"
    )
    assert "append_final_to_accepted" in writer
    for forbidden in (
        "_on_accept_selected",
        "_on_accept_all",
        "_on_accept_sam",
        "_on_accept_manual",
        "write_feature_to_accepted",
        "write_multiple_features",
        "write_manual_feature",
    ):
        assert forbidden not in dock
        assert forbidden not in writer
