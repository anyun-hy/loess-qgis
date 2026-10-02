import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_legacy_single_model_runtime_files_are_removed():
    obsolete = (
        ROOT / "inference_scripts" / "predict_semantic.py",
        ROOT / "inference_scripts" / "run_semantic.sh",
        ROOT / "inference_scripts" / "semantic_model.py",
        ROOT / "inference_scripts" / "run_sam3.sh",
        ROOT / "inference_scripts" / "sam3_class_batch.py",
        ROOT / "inference_scripts" / "run_sam3_class.sh",
        ROOT / "inference_scripts" / "run_semantic_batch.sh",
        ROOT / "inference_scripts" / "run_fusion.sh",
        ROOT / "inference_scripts" / "run_mosaic.sh",
        ROOT / "inference_scripts" / "run_polygonize.sh",
        ROOT / "inference_scripts" / "run_subpixel_vectorize.sh",
        ROOT / "inference_scripts" / "subpixel_vectorizer.py",
        ROOT / "qgis_plugins" / "labeling_tool" / "core" / "inference_runner.py",
        ROOT / "qgis_plugins" / "labeling_tool" / "core" / "async_runner.py",
        ROOT / "qgis_plugins" / "labeling_tool" / "core" / "pipeline_plan.py",
        ROOT / "qgis_plugins" / "labeling_tool" / "core" / "sam3_job_runner.py",
        ROOT / "tests" / "test_pipeline_plan.py",
        ROOT / "tests" / "test_subpixel_vectorizer.py",
    )
    assert not [str(path.relative_to(ROOT)) for path in obsolete if path.exists()]


def test_repository_excludes_external_tool_and_runtime_artifact_roots():
    obsolete_roots = (
        ROOT / ".omo",
        ROOT / ".opencode",
        ROOT / "output",
        ROOT / "scratch",
        ROOT / "docs" / "DATA_INDEX.md",
    )
    assert not [
        str(path.relative_to(ROOT))
        for path in obsolete_roots
        if path.exists()
    ]
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in (
        "/.omo/",
        "/.opencode/",
        "/output/",
        "/scratch/",
        "__pycache__/",
        ".DS_Store",
    ):
        assert pattern in ignore


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
    source = (ROOT / "src" / "labeling_tool" / "runs" / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert '"V5AsyncInferenceRunner"' in source
    assert '"InferenceRunner"' not in source
    plugin_source = (ROOT / "src" / "labeling_tool" / "main" / "main_dock.py").read_text(
        encoding="utf-8"
    )
    assert "semantic_weight" not in plugin_source


def test_runtime_fingerprint_uses_deployment_inventory_not_a_file_list():
    checker = (ROOT / "src" / "loess_runtime" / "system" / "environment_report.py").read_text(
        encoding="utf-8"
    )
    contract = (
        ROOT / "src" / "labeling_tool" / "runs" / "deployment_contract.py"
    ).read_text(encoding="utf-8")
    fingerprint_files = next(
        ast.literal_eval(node.value)
        for node in ast.parse(checker).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "FINGERPRINT_FILES"
            for target in node.targets
        )
    )

    assert "../project_manifest.json" in fingerprint_files
    assert "../runtime/loess_launcher.sh" in fingerprint_files
    assert "work_package_runtime.py" not in fingerprint_files
    assert '_inventory_block(\n            project_manifest, "inference_files"' in contract
    assert "_validate_inventory(" in contract


def test_unit_fit_wrapper_uses_the_inference_scripts_directory():
    wrapper = (ROOT / "scripts" / "runtime" / "run_unit_fit.sh").read_text(
        encoding="utf-8"
    )
    assert 'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"' in wrapper
    assert 'dirname "$0")/..' not in wrapper
    assert 'source "$SCRIPT_DIR/config.sh"' in wrapper
    assert (
        "python -m "
        "loess_runtime.geometry.boundary_fitting.unit_runtime"
    ) in wrapper


def test_accepted_labels_has_no_direct_candidate_bypass():
    dock = (ROOT / "src" / "labeling_tool" / "main" / "main_dock.py").read_text(
        encoding="utf-8"
    )
    writer = (ROOT / "src" / "labeling_tool" / "refinement" / "accepted_writer.py").read_text(
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


def test_accepted_labels_are_audited_and_snapshot_is_not_the_write_target():
    workflow = (
        ROOT / "src" / "labeling_tool" / "runs" / "run_workflow.py"
    ).read_text(encoding="utf-8")
    builder = (
        ROOT / "src" / "labeling_tool" / "runs" / "run_builder_v5.py"
    ).read_text(encoding="utf-8")
    dialog = (
        ROOT / "src" / "labeling_tool" / "refinement" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    background_tasks = (
        ROOT / "src" / "labeling_tool" / "refinement" / "background_io_tasks.py"
    ).read_text(encoding="utf-8")
    preparation = (
        ROOT / "src" / "labeling_tool" / "runs" / "run_preparation_task.py"
    ).read_text(encoding="utf-8")
    assert "RunPreparationTask(" in workflow
    assert "audit_accepted_layer(" in preparation
    assert "expected_crs=self.raster_crs" in preparation
    assert '"accepted_validation":' in builder
    assert '"accepted_target_gpkg":' in builder
    assert 'get("accepted_target_gpkg")' in dialog
    assert "AcceptedWriteTask(" in dialog
    assert "class AcceptedWriteTask(QgsTask):" in background_tasks
    assert "accepted_writer.append_final_to_accepted(" in background_tasks
    assert "before_commit=self._begin_commit" in background_tasks
    manual_loader = (
        ROOT / "src" / "labeling_tool" / "refinement" / "manual_run_loader.py"
    ).read_text(encoding="utf-8")
    assert "accepted_write_run_spec.json" in manual_loader
    assert "accepted_write_run_manifest.json" in manual_loader
    assert 'spec["accepted_write_manifest"]' in manual_loader


def test_refinement_dialog_cleanup_preserves_async_topology_and_accepted_gate():
    dialog = (
        ROOT / "src" / "labeling_tool" / "refinement" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(dialog)
    refinement = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ClassRefinementDialog"
    )
    obsolete = {
        "_accepted_layer_for_check",
        "_manual_candidate_band",
        "_show_manual_candidate",
        "_clear_manual_candidate_band",
    }
    members = {
        node.name for node in refinement.body if isinstance(node, ast.FunctionDef)
    } | {
        node.attr for node in ast.walk(refinement) if isinstance(node, ast.Attribute)
    }
    assert not obsolete & members
    topology = dialog.split("def _check_topology", 1)[1].split(
        "def _write_accepted", 1
    )[0]
    accepted = dialog.split("def _write_accepted", 1)[1].split(
        "def _update_accept_enabled", 1
    )[0]
    background_tasks = (
        ROOT / "src" / "labeling_tool" / "refinement" / "background_io_tasks.py"
    ).read_text(encoding="utf-8")
    assert "self._start_refinement_task(assemble=False)" in topology
    assert "self._issue_count is None" in accepted
    assert "self._issue_count != 0" in accepted
    assert "AcceptedWriteTask(" in accepted
    assert "accepted_writer.append_final_to_accepted(" in background_tasks
    assert "if not self.inputs_unchanged():" in background_tasks


def test_final_overlap_with_accepted_is_blocked_twice():
    topology = (
        ROOT / "src" / "labeling_tool" / "refinement" / "topology_validator.py"
    ).read_text(encoding="utf-8")
    writer = (
        ROOT / "src" / "labeling_tool" / "refinement" / "accepted_writer.py"
    ).read_text(encoding="utf-8")
    assert '"accepted_overlap"' in topology
    assert "assert_no_accepted_overlap(" in topology
    assert "audit_accepted_layer(" in writer
    assert "assert_no_accepted_overlap(" in writer
    assert "accepted_target_gpkg" in writer
    assert "run_spec_sha256" in writer
