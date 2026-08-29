import hashlib
from pathlib import Path

import pytest

import assemble_stream
from labeling_tool.core.run_state_db import RunStateDB


RUN_ID = "cleanup-recovery-run"
STREAM_ID = "model:test"


def _ready_unit_artifact(
    tmp_path: Path,
    *,
    unit_id: str = "core_00000",
    kind: str = "unit_raw_geoparquet",
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    database = RunStateDB(run_dir / "run_state.sqlite")
    database.initialize()
    database.create_run(RUN_ID, "a" * 64)
    unit_root = run_dir / "tmp" / "unit_outputs" / "model_test"
    unit_root.mkdir(parents=True)
    suffix = assemble_stream.UNIT_INTERMEDIATE_SUFFIXES[kind]
    path = unit_root / f"{unit_id}{suffix}"
    payload = b"verified unit intermediate\n"
    path.write_bytes(payload)
    path.with_name(f"{path.name}.manifest.json").write_text(
        '{"status":"committed"}\n', encoding="utf-8"
    )
    artifact_id = database.register_artifact(
        RUN_ID,
        kind,
        path,
        stream_id=STREAM_ID,
        unit_id=unit_id,
    )
    assert database.mark_artifact_ready(
        artifact_id,
        byte_count=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    return (
        {"run_id": RUN_ID, "run_dir": str(run_dir)},
        database,
        artifact_id,
        path,
    )


def test_cleanup_removes_only_owned_unit_intermediate(tmp_path):
    spec, database, artifact_id, path = _ready_unit_artifact(tmp_path)
    unit_root = path.parent
    user_file = unit_root / "research-notes.txt"
    user_file.write_text("keep", encoding="utf-8")
    final_path = Path(spec["run_dir"]) / "output" / "assembled.gpkg"
    final_path.parent.mkdir(parents=True)
    final_path.write_bytes(b"final assembled artifact")
    final_id = database.register_artifact(
        RUN_ID,
        "semantic_polygons",
        final_path,
        stream_id=STREAM_ID,
        unit_id="assembled",
    )
    assert database.mark_artifact_ready(
        final_id,
        byte_count=final_path.stat().st_size,
        sha256=hashlib.sha256(final_path.read_bytes()).hexdigest(),
    )

    report = assemble_stream._cleanup_stream_unit_artifacts(
        spec, database, STREAM_ID
    )

    assert report["artifact_count"] == 1
    assert report["cleaned_bytes"] == len(b"verified unit intermediate\n")
    assert database.get_artifact(artifact_id)["status"] == "cleaned"
    assert database.get_artifact(final_id)["status"] == "ready"
    assert not path.exists()
    assert user_file.read_text(encoding="utf-8") == "keep"
    assert final_path.read_bytes() == b"final assembled artifact"
    assert assemble_stream._cleanup_stream_unit_artifacts(
        spec, database, STREAM_ID
    )["artifact_count"] == 0


@pytest.mark.parametrize("fault_stage", ["claim", "rename", "finish_db", "unlink"])
def test_cleanup_recovers_every_transaction_crash_window(
    tmp_path, monkeypatch, fault_stage
):
    spec, database, artifact_id, path = _ready_unit_artifact(tmp_path)
    tombstone = assemble_stream._cleanup_tombstone(path, artifact_id)

    with monkeypatch.context() as patch:
        if fault_stage == "claim":
            original = database.claim_artifact_cleanup
            calls = 0

            def fail_claim(target_id):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError("injected claim failure")
                return original(target_id)

            patch.setattr(database, "claim_artifact_cleanup", fail_claim)
        elif fault_stage == "rename":
            patch.setattr(
                assemble_stream,
                "_rename_cleanup_file",
                lambda *_args: (_ for _ in ()).throw(
                    RuntimeError("injected rename failure")
                ),
            )
        elif fault_stage == "finish_db":
            patch.setattr(
                database,
                "finish_artifact_cleanup",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    RuntimeError("injected DB finish failure")
                ),
            )
        else:
            patch.setattr(
                assemble_stream,
                "_unlink_cleanup_tombstone",
                lambda *_args: (_ for _ in ()).throw(
                    RuntimeError("injected unlink failure")
                ),
            )

        with pytest.raises(RuntimeError, match="injected"):
            assemble_stream._cleanup_stream_unit_artifacts(
                spec, database, STREAM_ID
            )

    row = database.get_artifact(artifact_id)
    assert row["status"] == (
        "ready"
        if fault_stage == "claim"
        else "cleaned"
        if fault_stage == "unlink"
        else "cleaning"
    )
    if fault_stage in {"claim", "rename"}:
        assert path.is_file()
        assert not tombstone.exists()
    elif fault_stage in {"finish_db", "unlink"}:
        assert not path.exists()
        assert tombstone.is_file()

    recovered = assemble_stream._cleanup_stream_unit_artifacts(
        spec, database, STREAM_ID
    )
    assert recovered["artifact_count"] == 1
    assert database.get_artifact(artifact_id)["status"] == "cleaned"
    assert not path.exists()
    assert not tombstone.exists()


def test_cleanup_recovers_legacy_crash_after_data_unlink_before_manifest_unlink(
    tmp_path,
):
    spec, database, artifact_id, path = _ready_unit_artifact(tmp_path)
    manifest = path.with_name(f"{path.name}.manifest.json")
    tombstone = assemble_stream._cleanup_tombstone(path, artifact_id)
    assert database.claim_artifact_cleanup(artifact_id)
    assemble_stream._rename_cleanup_file(path, tombstone)
    assert database.finish_artifact_cleanup(artifact_id, success=True)
    assemble_stream._unlink_cleanup_tombstone(tombstone)
    assert manifest.is_file()

    report = assemble_stream._cleanup_stream_unit_artifacts(
        spec, database, STREAM_ID
    )

    assert report["artifact_count"] == 1
    assert not manifest.exists()
    assert database.get_artifact(artifact_id)["status"] == "cleaned"


def test_cleanup_rejects_traversal_unit_id_without_deleting_target(tmp_path):
    spec, database, artifact_id, original_path = _ready_unit_artifact(tmp_path)
    victim = Path(spec["run_dir"]) / "victim.gpkg"
    victim.write_bytes(b"user data")
    with database.transaction() as connection:
        connection.execute(
            "UPDATE artifacts SET unit_id=?, path=? WHERE artifact_id=?",
            ("../../victim", str(victim), artifact_id),
        )
    original_path.unlink()

    with pytest.raises(assemble_stream.StreamAssemblyError, match="unsafe unit_id"):
        assemble_stream._cleanup_stream_unit_artifacts(
            spec, database, STREAM_ID
        )

    assert victim.read_bytes() == b"user data"
    assert database.get_artifact(artifact_id)["status"] == "ready"


def test_cleanup_rejects_symlink_without_touching_target(tmp_path):
    spec, database, artifact_id, path = _ready_unit_artifact(tmp_path)
    victim = Path(spec["run_dir"]) / "user-file.gpkg"
    victim.write_bytes(b"user data")
    path.unlink()
    path.symlink_to(victim)

    with pytest.raises(assemble_stream.StreamAssemblyError, match="regular file"):
        assemble_stream._cleanup_stream_unit_artifacts(
            spec, database, STREAM_ID
        )

    assert path.is_symlink()
    assert victim.read_bytes() == b"user data"
    assert database.get_artifact(artifact_id)["status"] == "ready"
