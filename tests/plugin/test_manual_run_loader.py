import json

import pytest

from labeling_tool.refinement.manual_run_loader import (
    ManualRunLoadCancelled,
    ManualRunLoadError,
    load_manual_run,
    persist_rebound_workspace,
    prepare_manual_run,
    publish_manual_run_bundle,
)
from labeling_tool.shared.contracts.run_spec import (
    CLASS_ORDER,
    atomic_write_json,
    sha256_file,
)


def _copied_run(tmp_path):
    run_id = "20260729_200000_manual"
    run_root = tmp_path / run_id
    old_root = tmp_path / "remote" / run_id
    fusion_dir = run_root / "fusion" / "fixture"
    fusion_dir.mkdir(parents=True)
    semantic = fusion_dir / "semantic_polygons.gpkg"
    semantic.write_bytes(b"formal-fixture")
    snapshot = run_root / "accepted_snapshot.gpkg"
    snapshot.write_bytes(b"accepted-fixture")
    spec = {
        "schema_version": 2,
        "run_id": run_id,
        "run_dir": str(old_root),
        "raster": {"transform": [1, 0, 0, 0, -1, 0], "crs": "EPSG:4490"},
        "accepted_gpkg": str(old_root / "accepted_snapshot.gpkg"),
        "accepted_gpkg_sha256": "remote-hash",
        "run_spec_content_sha256": "stale-after-rebind",
        "fusion": {"profile_id": "fixture"},
    }
    spec_path = run_root / "run_spec.json"
    atomic_write_json(spec_path, spec)
    manifest_path = run_root / "run_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 2,
            "run_id": run_id,
            "run_spec": str(old_root / "run_spec.json"),
            "run_spec_sha256": sha256_file(spec_path),
            "status": "ready",
            "streams": [
                {
                    "stream_id": "fusion:fixture",
                    "kind": "fusion",
                    "fusion_profile_id": "fixture",
                    "status": "ready",
                    "paths": {
                        "semantic_polygons": str(
                            old_root / "fusion" / "fixture" / semantic.name
                        )
                    },
                }
            ],
        },
    )
    return run_root, spec_path, manifest_path, snapshot


def _initialized_workspace(run_root, run_id):
    classes_dir = run_root / "classes"
    classes_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for code in CLASS_ORDER:
        path = classes_dir / f"class_{code}.gpkg"
        path.write_bytes(f"class-{code}-fixture".encode("ascii"))
        records[str(code)] = {
            "class_code": code,
            "path": f"/remote/classes/{path.name}",
            "layer_name": "class_polygons",
            "feature_count": 1,
            "sha256": sha256_file(path),
            "state": "editing",
            "confirmed": False,
        }
    workspace = {
        "schema_version": 2,
        "run_id": run_id,
        "baseline_stream_id": "fusion:fixture",
        "baseline_source_path": "/remote/fusion/semantic_polygons.gpkg",
        "baseline_source_sha256": "preserved-baseline-sha",
        "formal_path": "/remote/fusion/semantic_polygons.gpkg",
        "formal_sha256": "preserved-formal-sha",
        "boundary_report_path": "/remote/fusion/boundary_fitting_report.json",
        "boundary_report_sha256": "preserved-report-sha",
        "feature_count": len(CLASS_ORDER),
        "classes": records,
    }
    workspace_path = classes_dir / "workspace.json"
    atomic_write_json(workspace_path, workspace)
    (classes_dir / "edit_history.jsonl").touch()
    return workspace_path


def test_manual_copy_freezes_a_separate_accepted_write_contract(tmp_path):
    run_root, _spec_path, _manifest_path, snapshot = _copied_run(tmp_path)

    _initialized_workspace(run_root, "20260729_200000_manual")
    bundle = load_manual_run(run_root)

    spec = bundle["run_spec"]
    write_manifest_path = run_root / "classes" / "accepted_write_run_manifest.json"
    write_spec_path = run_root / "classes" / "accepted_write_run_spec.json"
    assert spec["accepted_write_manifest"] == str(write_manifest_path)
    assert spec["accepted_target_gpkg"] == str(run_root / "accepted_labels.gpkg")
    assert spec["accepted_gpkg"] == str(snapshot)
    assert spec["accepted_gpkg_sha256"] == sha256_file(snapshot)
    assert "run_spec_content_sha256" not in json.loads(write_spec_path.read_text())
    write_manifest = json.loads(write_manifest_path.read_text())
    assert write_manifest["run_spec"] == str(write_spec_path)
    assert write_manifest["run_spec_sha256"] == sha256_file(write_spec_path)


def test_manual_copy_rejects_a_changed_original_run_spec(tmp_path):
    run_root, spec_path, _manifest_path, _snapshot = _copied_run(tmp_path)
    spec_path.write_text(spec_path.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ManualRunLoadError, match="SHA256"):
        load_manual_run(run_root)


def test_initialized_workspace_loads_without_copied_fusion_polygons(tmp_path):
    run_root, _spec_path, _manifest_path, _snapshot = _copied_run(tmp_path)
    semantic = run_root / "fusion" / "fixture" / "semantic_polygons.gpkg"
    semantic.unlink()
    workspace_path = _initialized_workspace(run_root, "20260729_200000_manual")

    bundle = load_manual_run(run_root)

    assert bundle["result"]["portable_classes_only"] is True
    assert bundle["workspace"]["baseline_available"] is False
    assert bundle["workspace"]["portable_classes_only"] is True
    stream = bundle["result"]["ready_streams"][0]
    assert stream["stream_id"] == "fusion:fixture"
    assert stream["manual_offline"] is True
    assert stream["review_polygons"] == ""
    for code in CLASS_ORDER:
        record = bundle["workspace"]["classes"][str(code)]
        assert record["path"] == str(
            (run_root / "classes" / f"class_{code}.gpkg").resolve()
        )
        assert record["sha256"] == sha256_file(record["path"])

    persist_rebound_workspace(bundle)
    persisted = json.loads(workspace_path.read_text(encoding="utf-8"))
    assert persisted["baseline_source_sha256"] == "preserved-baseline-sha"
    assert persisted["formal_sha256"] == "preserved-formal-sha"
    accepted_manifest = json.loads(
        (run_root / "classes" / "accepted_write_run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert {stream["stream_id"] for stream in accepted_manifest["streams"]} == {
        "fusion:fixture"
    }


def test_uninitialized_workspace_directs_to_results_area(tmp_path):
    run_root, _spec_path, _manifest_path, _snapshot = _copied_run(tmp_path)
    semantic = run_root / "fusion" / "fixture" / "semantic_polygons.gpkg"
    semantic.unlink()

    with pytest.raises(ManualRunLoadError, match="识别结果区"):
        load_manual_run(run_root)


def test_portable_workspace_rejects_a_missing_class_file(tmp_path):
    run_root, _spec_path, _manifest_path, _snapshot = _copied_run(tmp_path)
    semantic = run_root / "fusion" / "fixture" / "semantic_polygons.gpkg"
    semantic.unlink()
    _initialized_workspace(run_root, "20260729_200000_manual")
    (run_root / "classes" / "class_51.gpkg").unlink()

    with pytest.raises(ManualRunLoadError, match="class_51.gpkg"):
        load_manual_run(run_root)


def test_portable_workspace_preserves_changed_file_and_rejects_stale_review(tmp_path):
    run_root, *_ = _copied_run(tmp_path)
    workspace_path = _initialized_workspace(run_root, "20260729_200000_manual")
    original = workspace_path.read_bytes()
    changed = run_root / "classes" / "class_12.gpkg"
    changed.write_bytes(b"externally-edited-class")
    with pytest.raises(ManualRunLoadError, match="SHA256"):
        load_manual_run(run_root)
    assert changed.read_bytes() == b"externally-edited-class"
    assert workspace_path.read_bytes() == original


def test_background_prepare_cancels_without_publishing_metadata(tmp_path):
    run_root, _spec_path, _manifest_path, _snapshot = _copied_run(tmp_path)
    _initialized_workspace(run_root, "20260729_200000_manual")
    canceled = False

    def progress(_message, value):
        nonlocal canceled
        canceled = value >= 30

    with pytest.raises(ManualRunLoadCancelled):
        prepare_manual_run(
            run_root,
            is_canceled=lambda: canceled,
            progress=progress,
        )

    assert not (run_root / "classes" / "accepted_write_run_spec.json").exists()
    assert not (run_root / "classes" / "accepted_write_run_manifest.json").exists()


def test_background_publish_rejects_changed_input_before_any_write(tmp_path):
    run_root, _spec_path, _manifest_path, _snapshot = _copied_run(tmp_path)
    _initialized_workspace(run_root, "20260729_200000_manual")
    bundle = prepare_manual_run(run_root)
    (run_root / "classes" / "class_12.gpkg").write_bytes(b"changed-after-prepare")

    with pytest.raises(ManualRunLoadError, match="发生变化"):
        publish_manual_run_bundle(bundle)

    assert not (run_root / "classes" / "accepted_write_run_spec.json").exists()
    assert not (run_root / "classes" / "accepted_write_run_manifest.json").exists()


def test_prepare_rejects_early_class_changed_while_later_classes_hash(tmp_path):
    run_root, _spec_path, _manifest_path, _snapshot = _copied_run(tmp_path)
    _initialized_workspace(run_root, "20260729_200000_manual")
    first_class = run_root / "classes" / "class_12.gpkg"
    changed = False

    def progress(_message, value):
        nonlocal changed
        if value >= 29 and not changed:
            changed = True
            first_class.write_bytes(b"changed-after-own-hash")

    with pytest.raises(ManualRunLoadError, match="发生变化"):
        prepare_manual_run(run_root, progress=progress)

    assert changed
    assert not (run_root / "classes" / "accepted_write_run_manifest.json").exists()


@pytest.mark.parametrize(
    "relative_path",
    ["run_spec.json", "run_manifest.json", "classes/workspace.json"],
)
def test_prepare_rejects_contract_changed_after_read(tmp_path, relative_path):
    run_root, _spec_path, _manifest_path, _snapshot = _copied_run(tmp_path)
    _initialized_workspace(run_root, "20260729_200000_manual")
    target = run_root / relative_path
    changed = False

    def progress(_message, value):
        nonlocal changed
        if value >= 29 and not changed:
            changed = True
            target.write_text(
                target.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )

    with pytest.raises(ManualRunLoadError, match="发生变化"):
        prepare_manual_run(run_root, progress=progress)

    assert changed


def _full_run(tmp_path):
    run_root, spec_path, manifest_path, _ = _copied_run(tmp_path)
    _initialized_workspace(run_root, "20260729_200000_manual")
    manifest = json.loads(manifest_path.read_text())
    old_root = json.loads(spec_path.read_text())["run_dir"]
    manifest["streams"].append({
        "stream_id": "model:fixture", "kind": "model", "model_id": "fixture", "status": "ready",
        "paths": {},
    })
    for stream in manifest["streams"]:
        relative = "models/fixture" if stream["kind"] == "model" else "fusion/fixture"
        directory = run_root / relative
        directory.mkdir(exist_ok=True, parents=True)
        stream["output_sha256"] = {}
        for key, suffix in [("mask_mosaic", "tif"), ("confidence_mosaic", "tif"), ("semantic_polygons", "gpkg")]:
            path = directory / f"{key}.{suffix}"
            path.write_bytes(f"{relative}-{key}".encode())
            stream["paths"][key] = f"{old_root}/{relative}/{path.name}"
            stream["output_sha256"][key] = sha256_file(path)
    atomic_write_json(manifest_path, manifest)
    return run_root, spec_path, manifest_path


@pytest.mark.parametrize("scope", ["classes", "all"])
def test_scopes_preserve_human_records_and_original_contracts(tmp_path, scope):
    run_root, spec_path, manifest_path = _full_run(tmp_path)
    workspace_path = run_root / "classes/workspace.json"
    workspace = json.loads(workspace_path.read_text())
    workspace["classes"]["12"].update(confirmed=True, modified=True, state="confirmed", reviewer="human")
    atomic_write_json(workspace_path, workspace)
    history = run_root / "classes/edit_history.jsonl"
    history.write_text('{"action":"human_confirmed"}\n')
    before = {path: path.read_bytes() for path in [spec_path, manifest_path, history, *run_root.glob("classes/*.gpkg")]}
    bundle = load_manual_run(run_root, load_scope=scope)
    assert bundle["run_spec"]["load_scope"] == scope
    assert len(bundle["result"]["ready_streams"]) == (2 if scope == "all" else 1)
    after = bundle["workspace"]["classes"]["12"]
    assert after["confirmed"] and after["modified"] and after["reviewer"] == "human"
    assert all(path.read_bytes() == value for path, value in before.items())
    if scope == "all":
        model = bundle["result"]["ready_streams"][1]
        assert model["paths"]["mask_mosaic"] == str(run_root / "models/fixture/mask_mosaic.tif")


def test_full_scope_rejects_missing_result_without_downgrading_or_publishing(tmp_path):
    run_root, *_ = _full_run(tmp_path)
    workspace = run_root / "classes/workspace.json"
    original = workspace.read_bytes()
    (run_root / "models/fixture/confidence_mosaic.tif").unlink()
    with pytest.raises(ManualRunLoadError, match="confidence_mosaic"):
        load_manual_run(run_root, load_scope="all")
    assert workspace.read_bytes() == original
    assert not (run_root / "classes/accepted_write_run_manifest.json").exists()
    assert prepare_manual_run(run_root, load_scope="classes")["workspace"]


def test_initialization_reader_needs_no_classes_and_does_not_publish(tmp_path):
    run_root, spec_path, manifest_path, _ = _copied_run(tmp_path)
    original = {path: path.read_bytes() for path in [spec_path, manifest_path]}
    bundle = prepare_manual_run(run_root, for_initialization=True)
    assert bundle["workspace"] is None
    assert bundle["run_spec"]["manual_only"] is False
    publish_manual_run_bundle(bundle)
    assert not (run_root / "classes").exists()
    assert all(path.read_bytes() == content for path, content in original.items())


def test_original_run_keeps_accepted_target_across_scopes(tmp_path):
    run_root, spec_path, manifest_path = _full_run(tmp_path)
    spec = json.loads(spec_path.read_text())
    spec.update(run_dir=str(run_root), accepted_target_gpkg=str(tmp_path / "shared/accepted.gpkg"))
    atomic_write_json(spec_path, spec)
    manifest = json.loads(manifest_path.read_text())
    manifest["run_spec_sha256"] = sha256_file(spec_path)
    atomic_write_json(manifest_path, manifest)
    for scope in ("classes", "all"):
        bundle = prepare_manual_run(run_root, load_scope=scope)
        assert bundle["run_spec"]["accepted_target_gpkg"] == spec["accepted_target_gpkg"]


def test_partial_classes_without_metadata_cannot_be_initialized(tmp_path):
    run_root, *_ = _copied_run(tmp_path)
    directory = run_root / "classes"
    directory.mkdir()
    existing = directory / "class_12.gpkg"
    existing.write_bytes(b"human-edit")
    with pytest.raises(ManualRunLoadError, match="不完整"):
        prepare_manual_run(run_root, for_initialization=True)
    assert existing.read_bytes() == b"human-edit"
