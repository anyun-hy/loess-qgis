import multiprocessing
from pathlib import Path

import pytest

from labeling_tool.runs import run_index
from labeling_tool.shared.contracts.run_spec import atomic_write_json, sha256_file


def _unlocked_record_worker(
    output_root,
    run_id,
    status,
    read_complete,
    allow_write,
    outcome,
):
    """Reproduce the former unguarded read-merge-write sequence in one process."""
    output = Path(output_root).resolve()
    try:
        try:
            current = run_index._load_index(output)
        except run_index.RunIndexError:
            current = {}
        read_complete.set()
        if not allow_write.wait(timeout=10):
            raise TimeoutError("test did not release unguarded index writer")
        atomic_write_json(
            output / run_index.RUN_INDEX_FILENAME,
            run_index._merged_index(current, run_id, status),
        )
    except BaseException as exc:
        outcome.put(f"{type(exc).__name__}: {exc}")
    else:
        outcome.put("")


def _locked_record_worker(
    output_root,
    run_id,
    status,
    read_complete,
    allow_merge,
    entered,
    outcome,
):
    """Pause after the production function has acquired its index lock and read."""
    output = Path(output_root).resolve()
    original_load = run_index._load_index
    entered.set()

    def _pause_after_read(root):
        try:
            current = original_load(root)
        except run_index.RunIndexError:
            current = {}
        read_complete.set()
        if not allow_merge.wait(timeout=10):
            raise TimeoutError("test did not release locked index writer")
        return current

    run_index._load_index = _pause_after_read
    try:
        run_index.record_run_state(output, run_id, status=status)
    except BaseException as exc:
        outcome.put(f"{type(exc).__name__}: {exc}")
    else:
        outcome.put("")


def _exception_holding_worker(output_root, failure_seen, allow_exit, outcome):
    """Keep the worker alive after a lock-protected exception for release testing."""
    output = Path(output_root).resolve()

    def _raise_during_locked_read(_root):
        raise RuntimeError("controlled lock-scope failure")

    run_index._load_index = _raise_during_locked_read
    try:
        run_index.record_run_state(output, "20260925_120000_abcd", status="running")
    except RuntimeError as exc:
        outcome.put(str(exc))
        failure_seen.set()
    else:
        outcome.put("record_run_state did not propagate controlled failure")
        failure_seen.set()
    allow_exit.wait(timeout=10)


def _plain_record_worker(output_root, run_id, status, outcome):
    try:
        run_index.record_run_state(output_root, run_id, status=status)
    except BaseException as exc:
        outcome.put(f"{type(exc).__name__}: {exc}")
    else:
        outcome.put("")


def _join_or_terminate(process):
    if process.pid is None:
        return
    process.join(timeout=10)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
    assert process.exitcode == 0


def _write_run(output_root, run_id, *, ready):
    run_dir = output_root / "runs" / run_id
    run_dir.mkdir(parents=True)
    spec = {
        "schema_version": 2,
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "output_root": str(output_root.resolve()),
        "state_db": "dbname=tester user=tester host=/var/run/postgresql port=5432",
        "state_backend": "postgresql",
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


def test_controlled_unguarded_process_interleaving_loses_ready_pointer(tmp_path):
    """Two old snapshots deterministically reproduce the lost-update defect."""
    context = multiprocessing.get_context("spawn")
    output = tmp_path / "output"
    ready_id = "20260925_120000_abcd"
    later_id = "20260925_130000_efab"
    first_read = context.Event()
    second_read = context.Event()
    allow_first_write = context.Event()
    allow_second_write = context.Event()
    outcome = context.Queue()
    first = context.Process(
        target=_unlocked_record_worker,
        args=(
            str(output),
            ready_id,
            "ready",
            first_read,
            allow_first_write,
            outcome,
        ),
    )
    second = context.Process(
        target=_unlocked_record_worker,
        args=(
            str(output),
            later_id,
            "failed",
            second_read,
            allow_second_write,
            outcome,
        ),
    )
    first.start()
    second.start()
    try:
        assert first_read.wait(timeout=5)
        assert second_read.wait(timeout=5)
        allow_first_write.set()
        assert outcome.get(timeout=5) == ""
        allow_second_write.set()
        assert outcome.get(timeout=5) == ""
    finally:
        allow_first_write.set()
        allow_second_write.set()
        _join_or_terminate(first)
        _join_or_terminate(second)

    index = run_index._load_index(output)
    assert index["latest_run_id"] == later_id
    assert index["latest_ready_run_id"] == ""


def test_record_run_state_serializes_controlled_independent_process_writers(tmp_path):
    context = multiprocessing.get_context("spawn")
    output = tmp_path / "output"
    ready_id = "20260925_120000_abcd"
    later_id = "20260925_130000_efab"
    first_read = context.Event()
    second_read = context.Event()
    allow_first_merge = context.Event()
    allow_second_merge = context.Event()
    first_entered = context.Event()
    second_entered = context.Event()
    outcome = context.Queue()
    first = context.Process(
        target=_locked_record_worker,
        args=(
            str(output),
            ready_id,
            "ready",
            first_read,
            allow_first_merge,
            first_entered,
            outcome,
        ),
    )
    second = context.Process(
        target=_locked_record_worker,
        args=(
            str(output),
            later_id,
            "failed",
            second_read,
            allow_second_merge,
            second_entered,
            outcome,
        ),
    )
    first.start()
    try:
        assert first_entered.wait(timeout=5)
        assert first_read.wait(timeout=5)
        second.start()
        assert second_entered.wait(timeout=5)
        assert not second_read.wait(timeout=1)
        allow_first_merge.set()
        assert outcome.get(timeout=5) == ""
        assert second_read.wait(timeout=5)
        allow_second_merge.set()
        assert outcome.get(timeout=5) == ""
    finally:
        allow_first_merge.set()
        allow_second_merge.set()
        _join_or_terminate(first)
        _join_or_terminate(second)

    index_path = output / run_index.RUN_INDEX_FILENAME
    assert (output / run_index.RUN_INDEX_LOCK_FILENAME).read_bytes() == b""
    assert index_path.is_file()
    index = run_index._load_index(output)
    assert index["latest_run_id"] == later_id
    assert index["latest_run_status"] == "failed"
    assert index["latest_ready_run_id"] == ready_id


def test_record_run_state_releases_lock_after_exception_for_other_process(tmp_path):
    context = multiprocessing.get_context("spawn")
    output = tmp_path / "output"
    failure_seen = context.Event()
    allow_exception_worker_exit = context.Event()
    outcome = context.Queue()
    failing = context.Process(
        target=_exception_holding_worker,
        args=(
            str(output),
            failure_seen,
            allow_exception_worker_exit,
            outcome,
        ),
    )
    succeeding = context.Process(
        target=_plain_record_worker,
        args=(str(output), "20260925_130000_efab", "ready", outcome),
    )
    failing.start()
    try:
        assert failure_seen.wait(timeout=5)
        assert outcome.get(timeout=5) == "controlled lock-scope failure"
        succeeding.start()
        _join_or_terminate(succeeding)
        assert outcome.get(timeout=5) == ""
    finally:
        allow_exception_worker_exit.set()
        _join_or_terminate(failing)

    index = run_index._load_index(output)
    assert index["latest_run_id"] == "20260925_130000_efab"
    assert index["latest_ready_run_id"] == "20260925_130000_efab"


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
        Path(__file__).parents[2] / "src" / "labeling_tool" / "main" / "main_dock.py"
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
    assert "valid_ready_stream_ids" not in open_block
    assert "approved_fusion_streams" not in open_block
    assert "将在分类窗口中后台校验" in open_block


def test_v5_runner_updates_index_at_running_and_terminal_boundaries():
    source = (
        Path(__file__).parents[2]
        / "src"
        / "labeling_tool"
        / "runs"
        / "v5_async_runner.py"
    ).read_text(encoding="utf-8")
    start = source.split("def run_from_spec", 1)[1].split("def resume", 1)[0]
    finish = source.split("def _finish", 1)[1].split("def _record_startup_index", 1)[0]

    assert 'self._record_startup_index("running")' in start
    assert 'self._record_startup_index(result["status"])' in finish
