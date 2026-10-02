"""Direct contracts for pure inference-monitor observation owners."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from labeling_tool.monitor.monitor_observations import (
    MonitorLogObservations,
    MonitorObservations,
)

ROOT = Path(__file__).resolve().parents[2]


def _snapshot(
    *,
    status="running",
    streams=(),
    active_package=None,
    package_counts=None,
    runtime=None,
    phases=None,
    coverage=None,
):
    return {
        "run": {"status": status, "created_at": "1970-01-01T00:01:00Z"},
        "job_counts": {"work_package": package_counts or {}},
        "streams": list(streams),
        "stream_unit_type_counts": {},
        "stream_unit_job_type_counts": {},
        "active_work_package": active_package,
        "stream_runtime_progress": runtime or {},
        "assembly_phase_statuses": phases or {},
        "stream_coverage_validation": coverage or {},
    }


def _progress(owner, info, *, bound=False, wall=1000.0, monotonic=100.0):
    stream_id = str(info.get("stream_id") or "")
    batch = 8 if stream_id == "model:a" else 4 if stream_id == "model:b" else 0
    return owner.observe_stream_progress(
        info,
        database_bound=bound,
        configured_batch_size=batch,
        fusion_profile_id="mix",
        epoch_now=wall,
        monotonic_now=monotonic,
    )


def test_module_imports_without_qgis_in_a_clean_subprocess():
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import sys; import labeling_tool.monitor.monitor_observations; "
            "assert 'qgis' not in sys.modules",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_step_attempts_parallel_stage_and_bound_status_rules():
    owner = MonitorObservations()
    first = owner.observe_step_started("unit_fit:model:a:core", epoch_now=100.0)
    second = owner.observe_step_started("unit_fit:model:a:seam", epoch_now=101.0)
    assert first.refresh_stream_ids == ("model:a",)
    assert second.refresh_stream_ids == ("model:a",)
    assert owner.attempt_for("unit_fit:model:a:core") == 1

    success = owner.observe_step_finished(
        "unit_fit:model:a:core",
        0,
        {"success": True},
        epoch_now=105.0,
        database_bound=True,
    )
    assert success.pending_logs == ()
    assert owner.stream_view("model:a")["state"]["status"] == "运行中"
    assert owner.stream_view("model:a")["state"]["elapsed"] == "5.0s"

    snapshot = owner.apply_snapshot(
        _snapshot(streams=[{"stream_id": "model:a", "status": "pending"}]),
        epoch_now=106.0,
    )
    assert snapshot is not None
    assert owner.stream_view("model:a")["state"]["stage"] == "空间单元拟合"

    failure = owner.observe_step_finished(
        "unit_fit:model:a:seam",
        7,
        {"success": False, "error": "unit broke"},
        epoch_now=111.0,
        database_bound=True,
    )
    assert failure.refresh_stream_ids == ("model:a",)
    assert len(failure.pending_logs) == 1
    assert '"attempt":1' in failure.pending_logs[0].message
    assert owner.stream_view("model:a")["state"]["status"] == "失败"
    assert owner.stream_view("model:a")["state"]["failures"] == 1


def test_live_package_attempt_reset_fusion_and_incremental_tile_update():
    owner = MonitorObservations()
    model = _progress(
        owner,
        {
            "event": "package_model_loading",
            "package_id": "p1",
            "stream_id": "model:a",
            "current": 1,
            "total": 1,
        },
    )
    assert model.refresh_stream_ids == ("model:a",)
    assert owner.package_view()["configured_batch_size"] == 8

    fusion = _progress(
        owner,
        {
            "event": "package_tile_completed",
            "package_id": "p1",
            "stream_id": "model:a",
            "current": 2,
            "total": 2,
        },
        wall=1001.0,
    )
    assert fusion.refresh_stream_ids == ("model:a", "fusion:mix")
    assert owner.package_view()["stream_id"] == "fusion:mix"

    reset = _progress(
        owner,
        {
            "event": "package_tile_materialized",
            "package_id": "p1",
            "current": 1,
            "total": 3,
        },
        wall=1002.0,
    )
    assert reset.refresh_stream_ids == ()
    package = owner.package_view()
    assert "stream_id" not in package
    assert "configured_batch_size" not in package

    tile = _progress(
        owner,
        {
            "event": "tile_failed",
            "stream_id": "model:a",
            "tile_id": "t1",
            "current": 1,
            "total": 3,
            "error": "tile failed",
        },
    )
    assert tile.refresh_stream_ids == ("model:a",)
    assert len(tile.tile_updates) == 1
    assert tile.tile_updates[0].state["status"] == "失败"
    assert owner.tiles_snapshot("model:a")["t1"]["error"] == "tile failed"


def test_bound_package_progress_stops_before_generic_progress():
    owner = MonitorObservations()
    change = _progress(
        owner,
        {
            "event": "package_model_loading",
            "package_id": "p1",
            "stream_id": "model:a",
            "current": 1,
            "total": 2,
        },
        bound=True,
    )
    assert change.refresh_stream_ids == ("model:a",)
    state = owner.stream_view("model:a")["state"]
    assert state["stage"] == "Work Package 推理"
    assert state["progress"] == "-"


def test_snapshot_strict_observed_time_attempt_reset_and_terminal_fence():
    owner = MonitorObservations()
    _progress(
        owner,
        {
            "event": "package_model_loading",
            "package_id": "p1",
            "stream_id": "model:a",
            "current": 1,
            "total": 2,
        },
        wall=1000.0,
    )

    def active(observed_at, *, attempt=1, stream_id="persisted"):
        return {
            "package_id": "p1",
            "attempt": attempt,
            "monitor_runtime_json": (
                '{"observed_at":"' + observed_at + '","stream_id":"' + stream_id + '"}'
            ),
        }

    owner.apply_snapshot(
        _snapshot(
            streams=[{"stream_id": "model:a", "status": "pending"}],
            active_package=active("1970-01-01T00:16:40Z"),
            package_counts={"running": 1},
        ),
        epoch_now=1001.0,
    )
    assert owner.package_view()["stream_id"] == "model:a"

    owner.apply_snapshot(
        _snapshot(
            streams=[{"stream_id": "model:a", "status": "pending"}],
            active_package=active("1970-01-01T00:16:41Z"),
            package_counts={"running": 1},
        ),
        epoch_now=1002.0,
    )
    assert owner.package_view()["stream_id"] == "persisted"

    owner.apply_snapshot(
        _snapshot(
            streams=[{"stream_id": "model:a", "status": "pending"}],
            active_package=active("", attempt=2),
            package_counts={"running": 1},
        ),
        epoch_now=1003.0,
    )
    package = owner.package_view()
    assert package["attempt"] == 2
    assert "stream_id" not in package

    owner.mark_terminal("ready")
    before = owner.package_view()
    assert owner.apply_snapshot(_snapshot(status="running"), epoch_now=1004.0) is None
    assert owner.package_view() == before
    accepted = owner.apply_snapshot(_snapshot(status="ready"), epoch_now=1005.0)
    assert accepted is not None


def test_snapshot_refreshes_touched_and_removed_auxiliary_stream_data():
    owner = MonitorObservations()
    first = owner.apply_snapshot(
        _snapshot(
            streams=[{"stream_id": "model:a", "status": "pending"}],
            runtime={"model:a": {"status": "running"}},
            phases={"model:a": {"write": {"status": "running"}}},
            coverage={"model:a": {"status": "passed"}},
        ),
        epoch_now=100.0,
    )
    assert first is not None
    assert first.refresh_stream_ids == ("model:a",)

    removed = owner.apply_snapshot(_snapshot(), epoch_now=101.0)
    assert removed is not None
    assert removed.refresh_stream_ids == ("model:a",)
    view = owner.stream_view("model:a")
    assert view["runtime"] == {}
    assert view["phase_statuses"] == {}
    assert view["coverage"] == {"status": "passed"}

    same_text = owner.apply_snapshot(
        _snapshot(
            streams=[{"stream_id": "model:a", "status": "pending"}],
            coverage={"model:a": {"status": "failed"}},
        ),
        epoch_now=102.0,
    )
    assert same_text is not None
    assert same_text.refresh_stream_ids == ("model:a",)
    assert owner.stream_view("model:a")["coverage"]["status"] == "failed"


def test_views_and_tile_deltas_cannot_mutate_owner_state():
    owner = MonitorObservations()
    owner.apply_snapshot(
        _snapshot(
            streams=[{"stream_id": "model:a", "status": "pending"}],
            runtime={"model:a": {"nested": {"value": 1}}},
            phases={"model:a": {"write": {"status": "running"}}},
            coverage={"model:a": {"nested": {"value": 2}}},
        ),
        epoch_now=100.0,
    )
    tile = _progress(
        owner,
        {
            "event": "tile_completed",
            "stream_id": "model:a",
            "tile_id": "t1",
            "current": 1,
            "total": 1,
        },
    )
    view = owner.stream_view("model:a")
    view["runtime"]["nested"]["value"] = 99
    view["phase_statuses"]["write"]["status"] = "failed"
    view["coverage"]["nested"]["value"] = 99
    tiles = owner.tiles_snapshot("model:a")
    tiles["t1"]["status"] = "失败"
    tile.tile_updates[0].state["status"] = "失败"

    fresh = owner.stream_view("model:a")
    assert fresh["runtime"]["nested"]["value"] == 1
    assert fresh["phase_statuses"]["write"]["status"] == "running"
    assert fresh["coverage"]["nested"]["value"] == 2
    assert owner.tiles_snapshot("model:a")["t1"]["status"] == "完成"


def test_binding_reset_and_unbinding_have_distinct_lifecycles():
    owner = MonitorObservations()
    owner.observe_step_started("unit_fit:model:a:core", epoch_now=10.0)
    owner.apply_snapshot(
        _snapshot(streams=[{"stream_id": "model:a", "status": "pending"}]),
        epoch_now=11.0,
    )
    owner.mark_terminal("failed")
    assert owner.run_created_epoch == 60.0

    owner.begin_binding()
    assert owner.terminal_status == ""
    assert owner.run_created_epoch is None
    assert owner.has_stream("model:a")
    assert owner.attempt_for("unit_fit:model:a:core") == 1

    owner.mark_terminal("ready")
    owner.end_binding()
    assert owner.terminal_status == "ready"
    assert owner.has_stream("model:a")

    owner.reset()
    assert owner.stream_ids() == ()
    assert owner.attempt_for("unit_fit:model:a:core") == 0
    assert owner.terminal_status == ""


def test_log_pairing_timestamp_attempt_fallback_clear_and_terminal_dedup():
    attempts = {"unit_fit:model:a:core": 2}
    owner = MonitorLogObservations()
    event = {
        "source": "stderr",
        "message": "RuntimeWarning: captured warning",
        "step": "unit_fit:model:a:core",
        "timestamp": "2024-01-01T00:00:00Z",
    }
    first = owner.observe_process_log(
        event, attempt_for=lambda key: attempts.get(key, 0)
    )
    second = owner.observe_process_log(
        event, attempt_for=lambda key: attempts.get(key, 0)
    )
    assert first["context_key"] == "unit_fit:model:a:core:attempt=2"
    assert first["event_timestamp"] == "2024-01-01T00:00:00Z"
    assert second["fingerprint"] == first["fingerprint"]

    for _ in range(2):
        assert (
            owner.observe_log(
                "stderr",
                event["message"],
                context=None,
                attempt_for=lambda key: attempts.get(key, 0),
            )
            is None
        )
    third = owner.observe_log(
        "stderr",
        event["message"],
        context=None,
        attempt_for=lambda key: attempts.get(key, 0),
    )
    assert third is not None
    assert third["context_key"] == "unscoped"

    failure = owner.observe_log(
        "system",
        '{"event":"monitor_step_failed","error":"same failure"}',
        context=None,
        attempt_for=lambda _key: 0,
    )
    assert failure is not None
    assert (
        owner.observe_pipeline_failure(
            {"success": False, "status": "failed", "error": "same   failure"},
            attempt_for=lambda _key: 0,
        )
        is None
    )
    owner.clear_logged_errors()
    terminal = owner.observe_pipeline_failure(
        {"success": False, "status": "failed", "error": "same failure"},
        attempt_for=lambda _key: 0,
    )
    assert terminal is not None
    assert terminal["affected"] == "monitor_pipeline_failed"
    assert (
        owner.observe_pipeline_failure(
            {"success": False, "status": "stopped", "error": "new failure"},
            attempt_for=lambda _key: 0,
        )
        is None
    )


def test_clear_logged_errors_keeps_pending_legacy_suppression_but_reset_drops_it():
    owner = MonitorLogObservations()
    event = {"source": "stderr", "message": "Error: paired"}
    owner.observe_process_log(event, attempt_for=lambda _key: 0)
    owner.clear_logged_errors()
    assert (
        owner.observe_log(
            "stderr", "Error: paired", context=None, attempt_for=lambda _key: 0
        )
        is None
    )

    owner.observe_process_log(event, attempt_for=lambda _key: 0)
    owner.reset()
    assert (
        owner.observe_log(
            "stderr", "Error: paired", context=None, attempt_for=lambda _key: 0
        )
        is not None
    )
