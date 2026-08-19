import json

import pytest

from labeling_tool.core.manual_run_loader import (
    ManualRunLoadError,
    load_manual_run,
    persist_rebound_workspace,
)
from labeling_tool.core.run_spec import (
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
        "schema_version": 1,
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
    workspace_path = _initialized_workspace(
        run_root, "20260729_200000_manual"
    )

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
        (
            run_root / "classes" / "accepted_write_run_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert {
        stream["stream_id"] for stream in accepted_manifest["streams"]
    } == {"fusion:fixture"}


def test_uninitialized_workspace_still_requires_copied_fusion_polygons(tmp_path):
    run_root, _spec_path, _manifest_path, _snapshot = _copied_run(tmp_path)
    semantic = run_root / "fusion" / "fixture" / "semantic_polygons.gpkg"
    semantic.unlink()

    with pytest.raises(ManualRunLoadError, match="semantic_polygons.gpkg"):
        load_manual_run(run_root)


def test_portable_workspace_rejects_a_missing_class_file(tmp_path):
    run_root, _spec_path, _manifest_path, _snapshot = _copied_run(tmp_path)
    semantic = run_root / "fusion" / "fixture" / "semantic_polygons.gpkg"
    semantic.unlink()
    _initialized_workspace(run_root, "20260729_200000_manual")
    (run_root / "classes" / "class_51.gpkg").unlink()

    with pytest.raises(ManualRunLoadError, match="class_51.gpkg"):
        load_manual_run(run_root)


def test_portable_workspace_syncs_recalculated_class_sha(tmp_path):
    run_root, _spec_path, _manifest_path, _snapshot = _copied_run(tmp_path)
    semantic = run_root / "fusion" / "fixture" / "semantic_polygons.gpkg"
    semantic.unlink()
    workspace_path = _initialized_workspace(run_root, "20260729_200000_manual")

    # Modify class 12 file on disk
    class_12_path = run_root / "classes" / "class_12.gpkg"
    class_12_path.write_bytes(b"modified-class-12-content")

    bundle = load_manual_run(run_root)
    assert bundle["workspace"]["classes"]["12"]["sha256"] == sha256_file(class_12_path)
    persist_rebound_workspace(bundle)

    persisted = json.loads(workspace_path.read_text(encoding="utf-8"))
    assert persisted["classes"]["12"]["sha256"] == sha256_file(class_12_path)

