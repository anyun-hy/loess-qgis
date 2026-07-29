from pathlib import Path

import pytest

from labeling_tool.core import run_index
from labeling_tool.core.run_spec import atomic_write_json, sha256_file


def _write_run(output_root, run_id, *, ready):
    run_dir = output_root / "runs" / run_id
    run_dir.mkdir(parents=True)
    spec = {
        "schema_version": 2,
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "output_root": str(output_root.resolve()),
        "state_db": str(run_dir / "run_state.sqlite"),
        "fusion": {"profile_id": "l2_fusion_v1"},
    }
    spec_path = run_dir / "run_spec.json"
    atomic_write_json(spec_path, spec)
    if ready:
        stream = {
            "stream_id": "fusion:l2_fusion_v1",
            "kind": "fusion",
            "status": "ready",
            "boundary_fitting_status": "passed",
            "paths": {},
            "output_sha256": {},
        }
        atomic_write_json(
            run_dir / "run_manifest.json",
            {
                "schema_version": 2,
                "run_id": run_id,
                "run_spec": str(spec_path),
                "run_spec_sha256": sha256_file(spec_path),
                "success": True,
                "status": "ready",
                "streams": [stream],
                "ready_streams": [stream],
            },
        )
    return run_dir


def test_startup_lookup_never_enumerates_run_or_artifact_directories(
    tmp_path, monkeypatch
):
    output = tmp_path / "output"
    run_id = "20260729_120000_abcd"
    _write_run(output, run_id, ready=True)
    decoy = output / "runs" / "20260728_120000_dcba" / "deep" / "artifacts"
    decoy.mkdir(parents=True)
    (decoy / "huge_model.pt").write_bytes(b"not startup metadata")
    run_index.record_run_state(output, run_id, status="planned")
    run_index.record_run_state(output, run_id, status="ready")

    def fail_iterdir(_path):
        raise AssertionError("QGIS startup must not enumerate output directories")

    monkeypatch.setattr(Path, "iterdir", fail_iterdir)
    candidates = run_index.load_startup_candidates(output)

    assert candidates["latest"]["run_id"] == run_id
    assert candidates["latest_ready"]["run_id"] == run_id
    result = run_index.lightweight_ready_result(candidates["latest_ready"])
    assert result["ready_streams"][0]["stream_id"] == "fusion:l2_fusion_v1"


def test_newer_failed_run_preserves_previous_ready_pointer(tmp_path):
    output = tmp_path / "output"
    ready_id = "20260729_120000_abcd"
    failed_id = "20260729_130000_efab"
    _write_run(output, ready_id, ready=True)
    _write_run(output, failed_id, ready=False)

    run_index.record_run_state(output, ready_id, status="ready")
    run_index.record_run_state(output, failed_id, status="planned")
    run_index.record_run_state(output, failed_id, status="failed")
    candidates = run_index.load_startup_candidates(output)

    assert candidates["latest"]["run_id"] == failed_id
    assert candidates["latest"]["indexed_status"] == "failed"
    assert candidates["latest_ready"]["run_id"] == ready_id


def test_corrupt_or_oversized_index_fails_closed_without_fallback_scan(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    index_path = output / run_index.RUN_INDEX_FILENAME
    index_path.write_bytes(b"{" + b"x" * run_index.RUN_INDEX_MAX_BYTES + b"}")

    assert run_index.load_startup_candidates(output) == {}


def test_index_rejects_path_traversal_and_symlinked_run(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(run_index.RunIndexError, match="invalid indexed run_id"):
        run_index.record_run_state(output, "../outside", status="planned")

    outside = tmp_path / "outside"
    run_id = "20260729_120000_abcd"
    _write_run(tmp_path, run_id, ready=False)
    (output / "runs").mkdir()
    (output / "runs" / run_id).symlink_to(
        outside if outside.exists() else tmp_path / "runs" / run_id,
        target_is_directory=True,
    )
    atomic_write_json(
        output / run_index.RUN_INDEX_FILENAME,
        {
            "schema_version": 1,
            "latest_run_id": run_id,
            "latest_run_status": "planned",
            "latest_ready_run_id": "",
        },
    )

    assert run_index.load_startup_candidates(output) == {}


def test_main_dock_startup_restore_is_metadata_only():
    source = (
        Path(__file__).parents[1]
        / "qgis_plugins"
        / "labeling_tool"
        / "gui"
        / "main_dock.py"
    ).read_text(encoding="utf-8")
    restore = source.split("def _restore_latest_ready_run", 1)[1].split(
        "def _render_last_env_report", 1
    )[0]
    for forbidden in (
        "iter_ready_results",
        ".iterdir(",
        ".glob(",
        ".rglob(",
        "os.walk(",
        "valid_ready_stream_ids",
        "approved_fusion_streams",
        "RunStateDB(",
    ):
        assert forbidden not in restore
    assert "load_startup_candidates" in restore

    open_block = source.split("def _on_open_refinement", 1)[1].split(
        "def _on_load_manual_run", 1
    )[0]
    assert "valid_ready_stream_ids" in open_block
    assert "len(valid_streams) != len(declared_streams)" in open_block


def test_v5_runner_updates_index_at_running_and_terminal_boundaries():
    source = (
        Path(__file__).parents[1]
        / "qgis_plugins"
        / "labeling_tool"
        / "core"
        / "v5_async_runner.py"
    ).read_text(encoding="utf-8")
    start = source.split("def run_from_spec", 1)[1].split("def resume", 1)[0]
    finish = source.split("def _finish", 1)[1].split(
        "def _record_startup_index", 1
    )[0]

    assert 'self._record_startup_index("running")' in start
    assert 'self._record_startup_index(result["status"])' in finish
