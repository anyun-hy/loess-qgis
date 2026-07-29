import json

import pytest

from labeling_tool.core.manual_run_loader import (
    ManualRunLoadError,
    load_manual_run,
)
from labeling_tool.core.run_spec import atomic_write_json, sha256_file


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
