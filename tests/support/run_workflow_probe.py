# ruff: noqa: E402
"""Native lifecycle probes; every artifact belongs to a temporary fixture.

Real Qt delivery and detached QGIS sources are used. Task, process and runner
I/O are controlled explicitly, so no model, deployment or database is touched.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
try:
    from qgis.core import (
        QgsApplication,
        QgsCoordinateReferenceSystem,
        QgsFeature,
        QgsGeometry,
        QgsRectangle,
        QgsVectorLayer,
    )
    from qgis.PyQt import sip
    from qgis.PyQt.QtCore import QCoreApplication, QEvent, QObject, pyqtSignal
except ModuleNotFoundError:
    raise SystemExit(77)

from labeling_tool.qgis_support.qt_lifecycle import retire_after
from labeling_tool.runs import run_planning, run_workflow
from labeling_tool.shared.contracts.run_spec import RESERVATION_FILE, run_tile_cache_dir


class FakeTask(QObject):
    progressChanged = pyqtSignal(float)
    taskCompleted = pyqtSignal()
    taskTerminated = pyqtSignal()

    def __init__(self, request, **sources):
        super().__init__()
        self.request = request
        self.sources = sources
        self.result_data = None
        self.error_message = ""
        self.progress_message = "test progress"
        self.canceled = False

    def cancel(self):
        self.canceled = True

    def isCanceled(self):
        return self.canceled


class FakeProbe(QObject):
    succeeded = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, scripts_dir, parent=None):
        super().__init__(parent)
        self.request = None
        self.canceled = False

    def start(self, **request):
        self.request = request

    def cleanup(self):
        self.canceled = True


class FakeRunner(QObject):
    pipeline_finished = pyqtSignal(dict)
    stage_progress = pyqtSignal(object)
    shutdown_finished = pyqtSignal()
    step_started = pyqtSignal(str)
    step_finished = pyqtSignal(str, int, dict)
    stream_progress = pyqtSignal(object)
    log_line = pyqtSignal(str, str)

    def __init__(self, scripts_dir, parent=None):
        super().__init__(parent)
        self.calls = []
        self.closed = False
        self.auto_shutdown = True
        self.failure = None
        self.events = []

    def run_from_spec(self, spec_path, **kwargs):
        self.events.append("run")
        self.calls.append(("run", spec_path, kwargs))
        if self.failure:
            raise RuntimeError(self.failure)

    def resume(self, spec_path, **kwargs):
        self.events.append("resume")
        self.calls.append(("resume", spec_path, kwargs))
        if self.failure:
            raise RuntimeError(self.failure)

    def retry_failed(self, spec_path, **kwargs):
        self.events.append("retry_failed")
        self.calls.append(("retry_failed", spec_path, kwargs))
        if self.failure:
            raise RuntimeError(self.failure)

    def stop(self):
        self.calls.append(("stop",))

    def shutdown(self):
        self.closed = True
        if self.auto_shutdown:
            self.shutdown_finished.emit()


class Raster:
    def __init__(self, path):
        self.path = path

    def source(self):
        return str(self.path) + "|provider-option=yes"

    def crs(self):
        return QgsCoordinateReferenceSystem("EPSG:3857")

    def rasterUnitsPerPixelX(self):
        return 2.0

    def rasterUnitsPerPixelY(self):
        return 3.0


class Registry:
    def __init__(self, config):
        self.scaling = dict(
            partition_halo_px="auto",
            partition_tile_rows=2,
            partition_tile_cols=2,
            seam_band_px=64,
            score_cache_budget_gb="auto",
            min_free_disk_gb=1,
        )
        self.runtime = {"tile_batch_size": "auto"}
        self.boundary_fitting = {"enabled": True, "probe": "preserved"}

    def resolve_selection(self, model_ids, profile_id):
        return list(model_ids)

    def model(self, model_id):
        return SimpleNamespace(model_id=model_id, weight_file=f"{model_id}.pth")

    def profile(self, profile_id):
        return SimpleNamespace(
            profile_id=profile_id,
            file_path="fusion.json",
            profile={"version": "1", "models": []},
        )


def add_feature(layer, index):
    feature = QgsFeature(layer.fields())
    feature.setGeometry(QgsGeometry.fromRect(QgsRectangle(index, 0, index + 1, 1)))
    ok, _ = layer.dataProvider().addFeatures([feature])
    assert ok
    layer.updateExtents()


class Harness:
    def __init__(self, app, root):
        self.app = app
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.probes, self.preparations, self.builders, self.runners = [], [], [], []
        self.submitted = []
        self.storage_calls = []
        self.states, self.progress, self.contexts, self.results, self.errors = (
            [],
            [],
            [],
            [],
            [],
        )
        self.stops, self.shutdowns, self.events = [], [], []
        self.stack = ExitStack()
        for name, cls, created in (
            ("TileCacheProbeRunner", FakeProbe, self.probes),
            ("RunPreparationTask", FakeTask, self.preparations),
            ("RunBuilderTask", FakeTask, self.builders),
            ("V5AsyncInferenceRunner", FakeRunner, self.runners),
        ):

            def create(*args, _cls=cls, _created=created, **kwargs):
                value = _cls(*args, **kwargs)
                _created.append(value)
                if isinstance(value, FakeRunner):
                    value.events = self.events
                return value

            self.stack.enter_context(patch.object(run_workflow, name, create))
        self.stack.enter_context(
            patch.object(
                run_workflow,
                "QgsApplication",
                SimpleNamespace(
                    taskManager=lambda: SimpleNamespace(addTask=self.submitted.append)
                ),
            )
        )
        self.stack.enter_context(patch.object(run_planning, "ModelRegistry", Registry))

        def storage(output, **kwargs):
            self.storage_calls.append((output, kwargs))
            return {
                "score_cache_budget_mode": "auto",
                "resolved_score_cache_budget_gb": 2,
            }

        self.stack.enter_context(
            patch.object(run_planning, "storage_preflight", storage)
        )
        self.parent = QObject()
        self.flow = run_workflow.RunWorkflowController(parent=self.parent)
        self.flow.state_changed.connect(self.states.append)
        self.flow.stage_progress.connect(self.progress.append)
        self.flow.finished.connect(self.results.append)
        self.flow.pre_run_failed.connect(lambda *args: self.errors.append(args))
        self.flow.stopped_before_run.connect(lambda: self.stops.append(True))
        self.flow.shutdown_finished.connect(lambda: self.shutdowns.append(True))

        def context(*args):
            self.contexts.append(args)
            self.events.append("context")

        self.flow.monitor_context_ready.connect(context)
        self.layer = QgsVectorLayer("Polygon?crs=EPSG:3857", "range", "memory")
        add_feature(self.layer, 0)
        self.captures = []

        def current_range():
            self.captures.append(self.layer.featureCount())
            return self.layer

        self.tiles = [
            {
                "row": 0,
                "col": col,
                "bounds": QgsRectangle(col * 896, 0, col * 896 + 1024, 1536),
            }
            for col in range(3)
        ]
        self.request = run_workflow.RunStartRequest(
            scripts_dir=str(ROOT / "scripts/runtime"),
            output_root=str(root),
            accepted_target_gpkg=str(root / "accepted.gpkg"),
            raster_layer=Raster(root / "source.tif"),
            requested_extent=QgsRectangle(0, 0, 2048, 1536),
            processing_extent=QgsRectangle(0, 0, 2816, 1536),
            grid_tiles=list(self.tiles),
            active_tiles=list(self.tiles),
            range_selection={"mode": "vector_tile_intersection"},
            effective_config={
                "runtime": {"effective_device": "cpu", "keep_score_cache": True},
                "resource_tuning": {
                    "resolved": {
                        "tile_batch_size": 2,
                        "tile_batch_size_by_model": {"a": 4},
                    }
                },
                "fragmentation_regularization": {
                    "enabled": True,
                    "buffer_pixels": 256,
                    "policy_id": "fragmentation_v33_configurable_absorption_v1",
                },
            },
            environment_report={"config_fingerprint": "fixture-config"},
            accepted_layer=self.layer,
            accepted_validation={"overlap_tolerance": 0.05},
            get_valid_range_layer=current_range,
            skip_accepted=True,
            selected_model_ids=["a", "b"],
            fusion_profile_id=None,
            boundary_smoothing_enabled=False,
            overlap=64,
        )

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        if not sip.isdeleted(self.flow):
            self.flow.shutdown()
        # Complete outstanding fake owners even after a failed assertion.
        for task in (*self.preparations, *self.builders):
            if not sip.isdeleted(task) and task.canceled:
                task.taskTerminated.emit()
        for runner in self.runners:
            if not sip.isdeleted(runner):
                runner.shutdown_finished.emit()
        self.stack.close()
        if not sip.isdeleted(self.parent):
            self.parent.deleteLater()
        self.app.processEvents()

    def start(self):
        self.flow.start_new_run(self.request)
        self.app.processEvents()
        assert self.flow.state == run_workflow.RunFlowState.PREFLIGHTING
        assert self.flow.is_active
        assert self.probes[-1].request["tile"]["tile_id"] == "0_0"

    def prepare(self):
        self.start()
        self.probes[-1].succeeded.emit({"materialized_cache_bytes": 4000})
        assert self.flow.state == run_workflow.RunFlowState.PREPARING
        return self.preparations[-1]

    def prepared(self, task):
        task.result_data = dict(
            active_tiles=[self.tiles[0], self.tiles[2]],
            skipped_tiles=[self.tiles[0]],
            range_selection={
                "mode": "vector_tile_intersection",
                "selected_tile_count": 2,
            },
            range_snapshot=str(Path(task.request["run_dir"]) / "range_snapshot.gpkg"),
            accepted_snapshot=str(
                Path(task.request["run_dir"]) / "accepted_snapshot.gpkg"
            ),
            accepted_validation={"status": "passed", "overlap_tolerance": 0.05},
            inputs_prepared=True,
        )
        task.taskCompleted.emit()
        assert self.flow.state == run_workflow.RunFlowState.PLANNING, self.errors
        return self.builders[-1]

    def plan(self):
        return self.prepared(self.prepare())

    def built(self, task):
        path = Path(task.request["reserved_run_dir"]) / "run_spec.json"
        spec = dict(
            schema_version=2,
            run_id=task.request["run_id"],
            run_dir=str(path.parent),
            tiles=task.request["tiles"],
            state_db="synthetic-dsn",
        )
        path.write_text(json.dumps(spec))
        task.result_data = (spec, path, "synthetic-dsn")
        task.taskCompleted.emit()
        assert self.flow.state == run_workflow.RunFlowState.INFERENCING, self.errors
        return path

    def infer(self):
        return self.built(self.plan())


def start_and_finish(app, root):
    with Harness(app, root) as h:
        assert h.flow.state == run_workflow.RunFlowState.IDLE
        assert not h.flow.is_active
        h.start()
        assert h.captures == []
        add_feature(h.layer, 1)
        h.probes[-1].succeeded.emit({"materialized_cache_bytes": 4000})
        prep = h.preparations[-1]
        assert h.captures == [2]
        assert len(list(prep.sources["range_source"].getFeatures())) == 2
        assert len(list(prep.sources["accepted_source"].getFeatures())) == 2
        add_feature(h.layer, 2)
        assert len(list(prep.sources["range_source"].getFeatures())) == 2
        builder = h.prepared(prep)
        kwargs = builder.request
        assert [tile["status"] for tile in kwargs["tiles"]] == [
            "accepted",
            "excluded",
            "ready",
        ]
        assert kwargs["tiles"][1]["path"] == ""
        assert kwargs["tiles"][2]["pixel_window"] == {
            "x0": 896,
            "y0": 0,
            "x1": 1408,
            "y1": 512,
        }
        assert kwargs["raster"]["transform"] == [2.0, 0, 0, 0, -3.0, 1536]
        assert kwargs["raster"]["path"] == str(root / "source.tif")
        assert kwargs["scaling"]["partition_halo_px"] == 256
        assert kwargs["scaling"]["score_cache_budget_gb"] == 2
        assert kwargs["boundary_fitting"] == {"enabled": False, "probe": "preserved"}
        assert kwargs["tile_batch_size"] == 2
        assert kwargs["config_fingerprint"] == "fixture-config"
        assert kwargs["accepted_gpkg"].endswith("accepted_snapshot.gpkg")
        assert kwargs["skip_accepted"] is True
        storage = h.storage_calls[0][1]
        assert storage["tile_count"] == 2
        assert storage["tile_batch_size"] == 4
        assert storage["input_tile_bytes_per_tile"] == 4000
        assert storage["fixed_temporary_overhead_bytes"] == 512 * 512 * 14 * 2 * 4
        assert storage["deferred_temporary_reserve_bytes"] > 0
        path = h.built(builder)
        assert h.events == ["context", "run"]
        assert h.runners[-1].calls[0][1] == str(path)
        assert h.contexts[0][0:2] == ("synthetic-dsn", path.parent.name)
        h.runners[-1].stage_progress.emit({"name": "inference", "current": 1})
        assert h.progress[-1]["name"] == "inference"
        result = {"success": True}
        h.runners[-1].pipeline_finished.emit(result)
        assert h.flow.state == run_workflow.RunFlowState.FINISHED
        assert not h.flow.is_active
        assert h.results == [result]
        progress_count = len(h.progress)
        h.runners[-1].pipeline_finished.emit(result)
        h.runners[-1].stage_progress.emit({"name": "late after finish"})
        assert h.results == [result]
        assert len(h.progress) == progress_count
        assert path.is_file()
        assert not h.errors


def stop_before_probe(app, root):
    with Harness(app, root) as h:
        h.flow.start_new_run(h.request)
        h.flow.stop()
        h.flow.stop()
        app.processEvents()
        assert h.probes == []
        assert h.stops == [True]
        assert h.flow.state == run_workflow.RunFlowState.FINISHED
        assert not h.flow.is_active
        h.start()
        assert len(h.probes) == 1
        h.flow.stop()
        assert h.probes[-1].canceled


def stop_task(app, root, phase):
    with Harness(app, root) as h:
        task = h.prepare() if phase == "preparation" else h.plan()
        key = "run_dir" if phase == "preparation" else "reserved_run_dir"
        run_dir = Path(task.request[key])
        cache = run_tile_cache_dir(root, run_dir.name).parent
        h.flow.stop()
        assert task.isCanceled()
        assert h.flow.state == run_workflow.RunFlowState.STOPPING
        assert h.flow.is_active
        assert run_dir.is_dir() and cache.is_dir()
        task.taskCompleted.emit()  # A canceled task can still report completed.
        assert not run_dir.exists() and not cache.exists()
        assert h.stops == [True]
        assert h.flow.state == run_workflow.RunFlowState.FINISHED
        assert not h.runners[-1].calls
        task.taskTerminated.emit()
        assert h.stops == [True]


def stop_inference(app, root):
    with Harness(app, root) as h:
        path = h.infer()
        h.flow.stop()
        h.flow.stop()
        assert h.runners[-1].calls[-1] == ("stop",)
        assert sum(call[0] == "stop" for call in h.runners[-1].calls) == 1
        assert h.flow.state == run_workflow.RunFlowState.STOPPING
        assert path.is_file()
        result = {"success": False, "error": "Pipeline stopped by user"}
        h.runners[-1].pipeline_finished.emit(result)
        assert h.results == [result]
        assert h.flow.state == run_workflow.RunFlowState.FINISHED
        assert not h.flow.is_active
        assert path.is_file()


def shutdown_task(app, root, phase):
    with Harness(app, root) as h:
        task = h.prepare() if phase == "preparation" else h.plan()
        key = "run_dir" if phase == "preparation" else "reserved_run_dir"
        run_dir = Path(task.request[key])
        runner = h.runners[-1]
        runner.auto_shutdown = False
        retire_after(h.flow, h.flow.shutdown_finished)
        started = time.monotonic()
        h.flow.shutdown()
        h.flow.shutdown()
        assert time.monotonic() - started < 0.1
        h.parent.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert not sip.isdeleted(h.flow)
        assert task.isCanceled() and run_dir.is_dir()
        assert h.shutdowns == []
        # Both orderings occur in production: the runner or task can finish first.
        if phase == "preparation":
            runner.shutdown_finished.emit()
            assert h.shutdowns == []
            task.taskTerminated.emit()
        else:
            task.taskTerminated.emit()
            assert h.shutdowns == []
            runner.shutdown_finished.emit()
        assert h.shutdowns == [True]
        assert not run_dir.exists()
        assert h.results == [] and h.errors == [] and h.stops == []
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert sip.isdeleted(h.flow)


def stale_callbacks(app, root):
    with Harness(app, root) as h:
        old_prep = h.prepare()
        old_probe = h.probes[-1]
        old_runner = h.runners[-1]
        h.flow.stop()
        old_prep.taskTerminated.emit()
        # Start again without delivering deferred deletion of the old QObjects.
        h.flow.start_new_run(h.request)
        count = len(h.progress)
        old_probe.succeeded.emit({"materialized_cache_bytes": 9000})
        old_probe.failed.emit("old failure")
        old_prep.progressChanged.emit(99)
        old_prep.taskCompleted.emit()
        old_runner.stage_progress.emit({"name": "old progress"})
        old_runner.pipeline_finished.emit({"success": False})
        assert len(h.progress) == count
        assert len(h.preparations) == 1
        assert h.flow.state == run_workflow.RunFlowState.PREFLIGHTING
        assert h.errors == [] and h.results == []
        app.processEvents()
        h.probes[-1].succeeded.emit({"materialized_cache_bytes": 4000})
        builder = h.prepared(h.preparations[-1])
        h.flow.stop()
        builder.taskTerminated.emit()
        h.flow.start_new_run(h.request)
        count = len(h.progress)
        builder.progressChanged.emit(99)
        builder.taskCompleted.emit()
        builder.taskTerminated.emit()
        assert len(h.progress) == count
        assert h.flow.state == run_workflow.RunFlowState.PREFLIGHTING
        assert h.errors == [] and h.results == []


def failures(app, root):
    for phase in ("probe", "preparation", "builder", "runner"):
        with Harness(app, root / phase) as h:
            if phase == "probe":
                h.start()
                h.probes[-1].failed.emit("probe rejected")
            elif phase == "preparation":
                task = h.prepare()
                reservation = Path(task.request["run_dir"])
                task.error_message = "frozen inputs rejected"
                task.taskTerminated.emit()
                assert not reservation.exists()
            elif phase == "builder":
                task = h.plan()
                reservation = Path(task.request["reserved_run_dir"])
                task.error_message = "builder rejected"
                task.taskTerminated.emit()
                assert not reservation.exists()
            else:
                task = h.plan()
                h.runners[-1].failure = "runner rejected"
                path = Path(task.request["reserved_run_dir"]) / "run_spec.json"
                path.write_text("{}")
                task.result_data = (
                    {"run_id": path.parent.name, "tiles": []},
                    path,
                    "synthetic-dsn",
                )
                task.taskCompleted.emit()
                assert path.is_file()
            assert h.flow.state == run_workflow.RunFlowState.FINISHED
            assert not h.flow.is_active
            assert len(h.errors) == 1
            assert h.results == []


def recovery(app, root):
    path = root / "run_spec.json"
    path.write_text(
        json.dumps(
            {
                "run_id": "resume",
                "run_dir": str(root),
                "tiles": [],
                "state_db": "synthetic-dsn",
            }
        )
    )
    for retry in (False, True):
        with Harness(app, root / str(retry)) as h:
            h.flow.resume(
                str(path),
                str(ROOT / "scripts/runtime"),
                json.loads(path.read_text()),
                retry_failed=retry,
            )
            assert h.events == ["context", "retry_failed" if retry else "resume"]
            assert h.contexts[0][0:2] == ("synthetic-dsn", "resume")
            assert h.flow.state == run_workflow.RunFlowState.INFERENCING
            assert h.flow.is_active
            assert h.runners[-1].calls[0][0] == ("retry_failed" if retry else "resume")
            assert h.probes == [] and h.preparations == [] and h.builders == []
            h.runners[-1].pipeline_finished.emit({"success": False})
            assert h.flow.state == run_workflow.RunFlowState.FINISHED
            assert not h.flow.is_active


def reservation_guards(app, root):
    for case in (
        "has_spec",
        "foreign_marker",
        "invalid_marker",
        "marker_symlink",
        "run_symlink",
        "cache_symlink",
        "wrong_path",
    ):
        with Harness(app, root / case) as h:
            prep = h.prepare()
            run_dir = Path(prep.request["run_dir"])
            marker = run_dir / RESERVATION_FILE
            cache = run_tile_cache_dir(h.root, run_dir.name).parent
            preserved = run_dir
            if case == "has_spec":
                (run_dir / "run_spec.json").write_text("{}")
            elif case == "foreign_marker":
                marker.write_text(json.dumps({"run_id": "different-run"}))
            elif case == "invalid_marker":
                marker.write_text("invalid json")
            elif case == "marker_symlink":
                target = h.root / "marker-target"
                marker.rename(target)
                marker.symlink_to(target)
            elif case == "run_symlink":
                preserved = h.root / "preserved-run"
                run_dir.rename(preserved)
                run_dir.symlink_to(preserved, target_is_directory=True)
            elif case == "cache_symlink":
                preserved = h.root / "preserved-cache"
                cache.rename(preserved)
                cache.symlink_to(preserved, target_is_directory=True)
            else:
                # Supply a foreign reservation path through the reservation boundary.
                h.flow.stop()
                prep.taskTerminated.emit()
                foreign = h.root / "foreign"
                foreign.mkdir()
                (foreign / RESERVATION_FILE).write_text(
                    json.dumps({"run_id": run_dir.name})
                )
                with patch.object(
                    run_workflow,
                    "reserve_run_directory",
                    return_value=(run_dir.name, foreign),
                ):
                    prep = h.prepare()
                preserved = foreign
            h.flow.stop()
            prep.taskTerminated.emit()
            assert preserved.is_dir(), case
            assert h.flow.state == run_workflow.RunFlowState.FINISHED


def dock_integration(app, root):
    """Exercise actual Dock/monitor wiring without environment or DB I/O."""
    from qgis.PyQt.QtCore import QEventLoop, QTimer

    from labeling_tool.main import main_dock
    from labeling_tool.monitor.inference_monitor import InferenceMonitorDialog
    from labeling_tool.monitor.monitor_query_client import MonitorQueryClient

    class SnapshotExecutor:
        def execute(self, request):
            return {**request, "snapshot": {"run": {"status": "running"}}}

    client = MonitorQueryClient(executor=SnapshotExecutor())
    monitor = None
    dock = None
    closed = []
    client.shutdown_finished.connect(lambda: closed.append(True))
    try:
        with Harness(app, root) as h, ExitStack() as patches:

            def monitor_factory(parent):
                nonlocal monitor
                monitor = InferenceMonitorDialog(parent, query_client=client)
                return monitor

            def workflow_factory(parent):
                h.flow.setParent(parent)
                return h.flow

            patches.enter_context(
                patch.object(main_dock, "InferenceMonitorDialog", monitor_factory)
            )
            patches.enter_context(
                patch.object(main_dock, "RunWorkflowController", workflow_factory)
            )
            for name in (
                "_load_settings_and_defaults",
                "_restore_latest_ready_run",
                "_save_settings",
            ):
                patches.enter_context(
                    patch.object(main_dock.LabelingDockWidget, name, lambda *_: None)
                )
            notices, loaded, grouped = [], [], []
            patches.enter_context(
                patch.object(
                    main_dock.LabelingDockWidget,
                    "_show_nonblocking_notice",
                    lambda _dock, *args: notices.append(args),
                )
            )
            dock = main_dock.LabelingDockWidget(None)
            dock.layer_manager = SimpleNamespace(
                load_run_results=loaded.append,
                group_layers=lambda: grouped.append(True),
            )
            path = h.infer()
            assert dock.stop_btn.isEnabled()
            assert client.is_bound and client.run_id == path.parent.name
            result = {
                "success": True,
                "status": "completed",
                "run_spec": str(path),
                "ready_streams": [{"kind": "fusion", "stream_id": "fusion:test"}],
            }
            h.runners[-1].pipeline_finished.emit(result)
            assert loaded == [result] and grouped == [True]
            assert dock.open_refinement_btn.isEnabled()
            assert "Fusion 成功" in dock.result_summary_label.text()
            assert dock.progress_bar.format() == "完成"
            assert not dock.stop_btn.isEnabled()
            assert notices[-1][1] == "完成"
            h.runners[-1].pipeline_finished.emit(result)
            assert len(loaded) == 1

            preserved_result = dock._last_run_result
            preserved_spec = dock._recovery_run_spec
            for status in (
                "ownership_conflict",
                "ownership_lost",
                "terminal_state_conflict",
                "attempt_failed",
                "failed",
            ):
                dock._on_pipeline_finished(
                    {
                        "success": False,
                        "status": status,
                        "terminal_published": False,
                        "error": f"fixture {status}",
                        "run_spec": str(path),
                    }
                )
                assert loaded == [result] and grouped == [True]
                assert dock._last_run_result == preserved_result
                assert dock._recovery_run_spec == preserved_spec
                assert dock.open_refinement_btn.isEnabled()
                assert dock.progress_bar.format() == "本地尝试已结束"
                assert "Run 当前状态以监控同步为准" in notices[-1][2]

            dock.environment_panel.scripts_directory = str(ROOT / "scripts/runtime")
            dock._resume_existing_run(False)
            assert h.runners[-1].calls[0][0] == "resume"
            assert client.is_bound and client.run_id == path.parent.name
            assert not dock.resume_btn.isEnabled()
            stopped = {
                **result,
                "success": False,
                "status": "stopped",
                "error": "Pipeline stopped by user",
            }
            h.runners[-1].pipeline_finished.emit(stopped)
            assert dock.progress_bar.format() == "已停止"
            assert len(notices) == 6
            assert dock.resume_btn.isEnabled()

            previous_count = len(h.runners)
            with patch.object(
                main_dock.QMessageBox, "question", return_value=main_dock.NO
            ):
                dock._resume_existing_run(True)
            assert len(h.runners) == previous_count
            with patch.object(
                main_dock.QMessageBox, "question", return_value=main_dock.YES
            ):
                dock._resume_existing_run(True)
            assert h.runners[-1].calls[0][0] == "retry_failed"
            h.runners[-1].pipeline_finished.emit(stopped)

            synced_run_dir = root / "runs" / "monitor-synced"
            synced_run_dir.mkdir(parents=True)
            (synced_run_dir / "run_spec.json").write_text("{}")
            synced_spec = {
                "schema_version": 2,
                "run_id": "monitor-synced",
                "run_dir": str(synced_run_dir),
            }
            manual_result = dock._last_run_result
            dock._on_monitor_main_run_handling(
                {
                    "run_id": "monitor-synced",
                    "run_spec": dict(synced_spec),
                    "observed_status": "failed",
                }
            )
            assert dock._recovery_run_spec == synced_spec
            assert dock._startup_recovery_status == "failed"
            assert dock._last_run_result == manual_result
            assert dock.retry_failed_btn.isEnabled()

            bad_spec = {**synced_spec, "run_id": "wrong-id"}
            dock._on_monitor_main_run_handling(
                {
                    "run_id": "monitor-synced",
                    "run_spec": bad_spec,
                    "observed_status": "failed",
                }
            )
            assert dock._recovery_run_spec == synced_spec

            malformed_schema = {**synced_spec, "schema_version": "not-an-integer"}
            dock._on_monitor_main_run_handling(
                {
                    "run_id": "monitor-synced",
                    "run_spec": malformed_schema,
                    "observed_status": "failed",
                }
            )
            assert dock._recovery_run_spec == synced_spec

            task = h.prepare()
            assert not dock.resume_btn.isEnabled()
            assert not dock.retry_failed_btn.isEnabled()
            dock._on_monitor_main_run_handling(
                {
                    "run_id": "monitor-synced",
                    "run_spec": dict(synced_spec),
                    "observed_status": "failed",
                }
            )
            assert dock._recovery_run_spec == synced_spec
            assert "不能切换 Run" in notices[-1][1]
            runner_count = len(h.runners)
            dock._resume_existing_run(False)
            assert len(h.runners) == runner_count
            reservation = Path(task.request["run_dir"])
            dock.cleanup()
            dock.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            assert not sip.isdeleted(h.flow)
            assert reservation.is_dir()
            task.taskTerminated.emit()
            assert h.shutdowns == [True]
            assert not reservation.exists()
    finally:
        if dock is not None and not sip.isdeleted(dock):
            if dock.workflow is not None:
                dock.cleanup()
            dock.deleteLater()
        if not closed:
            loop = QEventLoop()
            client.shutdown_finished.connect(loop.quit)
            client.shutdown()
            if not closed:
                QTimer.singleShot(3000, loop.quit)
                loop.exec()
        assert closed == [True]
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def native_task_shutdown(app, root):
    """Cancellation also waits for an actual worker-owned QgsTask terminal."""
    from qgis.core import QgsTask

    entered = threading.Event()
    release = threading.Event()
    worker_threads, finished_threads = [], []

    class HeldPreparation(QgsTask):
        def __init__(self, request, **sources):
            super().__init__("workflow fixture", QgsTask.Flag.CanCancel)
            self.request = request
            self.result_data = None
            self.error_message = ""

        def run(self):
            worker_threads.append(threading.get_ident())
            entered.set()
            release.wait(3)
            return False

    with Harness(app, root) as h:

        def create(request, **sources):
            task = HeldPreparation(request, **sources)
            h.preparations.append(task)
            return task

        with (
            patch.object(run_workflow, "QgsApplication", QgsApplication),
            patch.object(run_workflow, "RunPreparationTask", create),
        ):
            task = h.prepare()
            reservation = Path(task.request["run_dir"])
            deadline = time.monotonic() + 2
            try:
                while not entered.is_set() and time.monotonic() < deadline:
                    app.processEvents()
                    time.sleep(0.005)
                assert entered.is_set()
                assert worker_threads[0] != threading.get_ident()
                h.flow.shutdown_finished.connect(
                    lambda: finished_threads.append(threading.get_ident())
                )
                retire_after(h.flow, h.flow.shutdown_finished)
                h.flow.shutdown()
                h.parent.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                assert reservation.is_dir()
                assert not h.shutdowns and not sip.isdeleted(h.flow)
            finally:
                release.set()
            deadline = time.monotonic() + 2
            while not h.shutdowns and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.005)
            assert h.shutdowns == [True]
            assert finished_threads == [threading.get_ident()]
            assert not reservation.exists()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            assert sip.isdeleted(h.flow)


if __name__ == "__main__":
    app = QgsApplication([], False)
    app.initQgis()
    scenario = sys.argv[2]
    try:
        with tempfile.TemporaryDirectory(prefix="loess-workflow-") as temporary:
            root = Path(temporary)
            if scenario.startswith("stop_") and scenario in {
                "stop_preparation",
                "stop_builder",
            }:
                stop_task(app, root, scenario.removeprefix("stop_"))
            elif scenario in {"shutdown_preparation", "shutdown_builder"}:
                shutdown_task(app, root, scenario.removeprefix("shutdown_"))
            else:
                globals()[scenario](app, root)
        print(scenario + ": passed")
    finally:
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.exitQgis()
