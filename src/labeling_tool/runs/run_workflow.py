"""GUI-thread owner for one labeling Run lifecycle."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from qgis.core import (
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransformContext,
    QgsProject,
    QgsVectorLayerFeatureSource,
)
from qgis.PyQt.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot

from labeling_tool.qgis_support.qt_lifecycle import retire_after
from labeling_tool.runs.run_builder_task import RunBuilderTask
from labeling_tool.runs.run_planning import build_run_builder_kwargs
from labeling_tool.runs.run_preparation_task import RunPreparationTask
from labeling_tool.runs.tile_cache_probe_runner import TileCacheProbeRunner
from labeling_tool.runs.v5_async_runner import (
    ThreadedV5AsyncInferenceRunner as V5AsyncInferenceRunner,
)
from labeling_tool.shared.contracts.run_spec import (
    RESERVATION_FILE,
    reserve_run_directory,
    run_tile_cache_dir,
)

logger = logging.getLogger("labeling_tool.run_workflow")


class RunFlowState(Enum):
    IDLE = "idle"
    PREFLIGHTING = "preflighting"
    PREPARING = "preparing"
    PLANNING = "planning"
    INFERENCING = "inferencing"
    STOPPING = "stopping"
    FINISHED = "finished"
    SHUTTING_DOWN = "shutting_down"


_ACTIVE_STATES = {
    RunFlowState.PREFLIGHTING,
    RunFlowState.PREPARING,
    RunFlowState.PLANNING,
    RunFlowState.INFERENCING,
    RunFlowState.STOPPING,
}


@dataclass(frozen=True)
class RunStartRequest:
    scripts_dir: str
    output_root: str
    accepted_target_gpkg: str
    raster_layer: Any
    requested_extent: Any
    processing_extent: Any
    grid_tiles: tuple[dict, ...]
    active_tiles: tuple[dict, ...]
    range_selection: dict
    effective_config: dict
    environment_report: dict
    accepted_layer: Any | None
    accepted_validation: dict
    get_valid_range_layer: Callable[[], Any] | None
    skip_accepted: bool
    selected_model_ids: tuple[str, ...]
    fusion_profile_id: str | None
    boundary_smoothing_enabled: bool
    overlap: int


@dataclass
class _PendingReservation:
    output_root: str
    run_id: str
    run_dir: str

    def discard_if_unused(self) -> None:
        """Remove only this attempt's exact marker-backed reservation."""

        output_root = Path(self.output_root).expanduser().resolve()
        expected_run_dir = output_root / "runs" / self.run_id
        configured_run_dir = Path(self.run_dir).expanduser()
        if configured_run_dir.is_symlink():
            logger.error(
                "拒绝清理符号链接形式的 Run 预留目录: %s",
                configured_run_dir,
            )
            return
        run_dir = configured_run_dir.resolve()
        if run_dir != expected_run_dir:
            logger.error("拒绝清理不匹配的 Run 预留目录: %s", run_dir)
            return
        marker = run_dir / RESERVATION_FILE
        if (
            marker.is_symlink()
            or not marker.is_file()
            or (run_dir / "run_spec.json").exists()
        ):
            return
        try:
            reservation = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.error("拒绝清理身份无效的 Run 预留目录: %s", run_dir)
            return
        if str(reservation.get("run_id") or "") != self.run_id:
            logger.error("拒绝清理身份不匹配的 Run 预留目录: %s", run_dir)
            return
        cache_root = run_tile_cache_dir(output_root, self.run_id).parent
        for candidate in (cache_root, run_dir):
            try:
                if candidate.is_symlink():
                    logger.error(
                        "拒绝清理符号链接形式的 Run 预留目录: %s",
                        candidate,
                    )
                elif candidate.exists():
                    shutil.rmtree(candidate)
            except OSError as exc:
                logger.warning(
                    "清理未使用的 Run 预留目录失败 %s: %s",
                    candidate,
                    exc,
                )


@dataclass
class _RunAttempt:
    token: int
    request: RunStartRequest
    tile_cache_sample: dict = field(default_factory=dict)
    prepared: dict = field(default_factory=dict)
    reservation: _PendingReservation | None = None


class RunWorkflowController(QObject):
    """Own probe, QGIS tasks, runner, state, and pending reservation."""

    state_changed = pyqtSignal(str)
    stage_progress = pyqtSignal(object)
    runner_changed = pyqtSignal(object)
    monitor_context_ready = pyqtSignal(str, str, int, object)
    pre_run_failed = pyqtSignal(str, str)
    stopped_before_run = pyqtSignal()
    finished = pyqtSignal(object)
    shutdown_finished = pyqtSignal()

    STAGE_TOTAL = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = RunFlowState.IDLE
        self._generation = 0
        self._attempt: _RunAttempt | None = None
        self._probe = None
        self._probe_token = 0
        self._preparation_task = None
        self._preparation_token = 0
        self._builder_task = None
        self._builder_token = 0
        self._runner = None
        self._runner_token = 0
        self._shutdown_runner = None
        self._shutdown_emitted = False

    @property
    def state(self) -> RunFlowState:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state in _ACTIVE_STATES

    def _set_state(self, state: RunFlowState) -> None:
        if state is self._state:
            return
        self._state = state
        self.state_changed.emit(state.value)

    def _next_token(self) -> int:
        self._generation += 1
        return self._generation

    def _accepts_start(self) -> bool:
        return self._state in {RunFlowState.IDLE, RunFlowState.FINISHED}

    def start_new_run(self, request: RunStartRequest) -> None:
        if not self._accepts_start():
            raise RuntimeError("a Run workflow is already active")
        token = self._next_token()
        self._retire_current_runner()
        try:
            runner = self._create_runner(request.scripts_dir)
        except Exception as exc:
            self._set_state(RunFlowState.FINISHED)
            self.pre_run_failed.emit("错误", str(exc))
            return
        self._attempt = _RunAttempt(token=token, request=request)
        self._runner = runner
        self._runner_token = token
        self.runner_changed.emit(runner)
        self._set_state(RunFlowState.PREFLIGHTING)
        self.stage_progress.emit(
            {
                "key": "extraction",
                "name": "准备按工作包读取影像",
                "index": 1,
                "stage_total": self.STAGE_TOTAL,
                "current": 0,
                "total": 1,
                "message": "不再预切全部 Tile，推理时按 Work Package 即时读取",
            }
        )
        QTimer.singleShot(0, lambda value=token: self._start_probe(value))

    def resume(
        self,
        run_spec_path: str,
        scripts_dir: str,
        run_spec: dict,
        *,
        retry_failed: bool,
    ) -> None:
        if not self._accepts_start():
            raise RuntimeError("a Run workflow is already active")
        token = self._next_token()
        self._retire_current_runner()
        try:
            runner = self._create_runner(scripts_dir)
        except Exception as exc:
            self._set_state(RunFlowState.FINISHED)
            self.pre_run_failed.emit("恢复运行失败", str(exc))
            return
        self._runner = runner
        self._runner_token = token
        self.runner_changed.emit(runner)
        self._set_state(RunFlowState.INFERENCING)
        try:
            self.monitor_context_ready.emit(
                str(run_spec["state_db"]),
                str(run_spec["run_id"]),
                int(
                    (run_spec.get("scaling") or {}).get(
                        "tile_page_size",
                        500,
                    )
                ),
                dict(run_spec),
            )
            if retry_failed:
                runner.retry_failed(str(run_spec_path))
            else:
                runner.resume(str(run_spec_path))
        except Exception as exc:
            self._set_state(RunFlowState.FINISHED)
            self.pre_run_failed.emit("恢复运行失败", str(exc))

    def _create_runner(self, scripts_dir: str):
        runner = V5AsyncInferenceRunner(scripts_dir, parent=self)
        runner.pipeline_finished.connect(self._on_runner_finished)
        runner.stage_progress.connect(self._on_runner_progress)
        return runner

    def _disconnect_runner_callbacks(self, runner) -> None:
        for signal, callback in (
            (runner.pipeline_finished, self._on_runner_finished),
            (runner.stage_progress, self._on_runner_progress),
        ):
            try:
                signal.disconnect(callback)
            except (TypeError, RuntimeError):
                pass

    @staticmethod
    def _extent_as_dict(extent) -> dict[str, float]:
        return {
            "xmin": extent.xMinimum(),
            "ymin": extent.yMinimum(),
            "xmax": extent.xMaximum(),
            "ymax": extent.yMaximum(),
        }

    def _current_attempt(self, token: int) -> _RunAttempt | None:
        attempt = self._attempt
        if attempt is None or attempt.token != token:
            return None
        return attempt

    def _start_probe(self, token: int) -> None:
        attempt = self._current_attempt(token)
        if attempt is None or self._state is not RunFlowState.PREFLIGHTING:
            return
        request = attempt.request
        try:
            active_tiles = sorted(
                request.active_tiles,
                key=lambda item: (int(item["row"]), int(item["col"])),
            )
            if not active_tiles:
                raise ValueError("当前选择范围内没有可用于存储预检的 active Tile")
            sample = active_tiles[0]
            row = int(sample["row"])
            col = int(sample["col"])
            tile = {
                "tile_id": f"{row}_{col}",
                "row_no": row,
                "col_no": col,
                "bounds": self._extent_as_dict(sample["bounds"]),
            }
            probe = TileCacheProbeRunner(request.scripts_dir, parent=self)
            self._probe = probe
            self._probe_token = token
            probe.succeeded.connect(self._on_probe_ready)
            probe.failed.connect(self._on_probe_failed)
            self.stage_progress.emit(
                {
                    "key": "extraction",
                    "name": "存储预检",
                    "index": 1,
                    "stage_total": self.STAGE_TOTAL,
                    "current": 0,
                    "total": 1,
                    "message": (
                        f"正在用正式物化路径测量真实 Tile ({row},{col}) 缓存字节"
                    ),
                }
            )
            probe.start(
                raster_path=request.raster_layer.source().split("|", 1)[0],
                output_root=request.output_root,
                tile=tile,
            )
        except Exception as exc:
            self._release_probe(cancel=True)
            self._fail_before_run("Tile 存储预检失败", str(exc))

    def _matches_probe(self, probe, token: int) -> bool:
        return (
            probe is self._probe
            and self._current_attempt(token) is not None
            and self._state is RunFlowState.PREFLIGHTING
        )

    @pyqtSlot(dict)
    def _on_probe_ready(self, measurement: dict) -> None:
        probe, token = self.sender(), self._probe_token
        if not self._matches_probe(probe, token):
            return
        self._release_probe(expected=probe)
        attempt = self._current_attempt(token)
        if attempt is None:
            return
        attempt.tile_cache_sample = dict(measurement)
        self._start_preparation(token)

    @pyqtSlot(str)
    def _on_probe_failed(self, message: str) -> None:
        probe, token = self.sender(), self._probe_token
        if not self._matches_probe(probe, token):
            return
        self._release_probe(expected=probe)
        self._fail_before_run("Tile 存储预检失败", str(message))

    def _release_probe(self, *, expected=None, cancel: bool = False) -> None:
        probe = self._probe
        if expected is not None and probe is not expected:
            return
        self._probe = None
        self._probe_token = 0
        if probe is None:
            return
        for signal, callback in (
            (probe.succeeded, self._on_probe_ready),
            (probe.failed, self._on_probe_failed),
        ):
            try:
                signal.disconnect(callback)
            except (TypeError, RuntimeError):
                pass
        if cancel:
            probe.cleanup()
        probe.deleteLater()

    def _start_preparation(self, token: int) -> None:
        attempt = self._current_attempt(token)
        if attempt is None or self._state is not RunFlowState.PREFLIGHTING:
            return
        request = attempt.request
        try:
            run_id, run_dir = reserve_run_directory(request.output_root)
            attempt.reservation = _PendingReservation(
                output_root=request.output_root,
                run_id=run_id,
                run_dir=str(run_dir),
            )
            range_source = None
            range_layer = None
            if request.range_selection.get("mode") == "vector_tile_intersection":
                if request.get_valid_range_layer is None:
                    raise ValueError("矢量范围图层不可用")
                range_layer = request.get_valid_range_layer()
                range_source = QgsVectorLayerFeatureSource(range_layer)
            accepted = request.accepted_layer
            task = RunPreparationTask(
                {
                    "run_dir": str(run_dir),
                    "grid_tiles": list(request.grid_tiles),
                    "active_tiles": list(request.active_tiles),
                    "range_selection": dict(request.range_selection),
                    "accepted_validation": dict(request.accepted_validation),
                    "skip_accepted": request.skip_accepted,
                    "accepted_source_path": (
                        accepted.source() if accepted is not None else ""
                    ),
                },
                range_source=range_source,
                accepted_source=(
                    QgsVectorLayerFeatureSource(accepted)
                    if accepted is not None
                    else None
                ),
                raster_crs=QgsCoordinateReferenceSystem(request.raster_layer.crs()),
                transform_context=QgsCoordinateTransformContext(
                    QgsProject.instance().transformContext()
                ),
                range_wkb_type=(
                    range_layer.wkbType() if range_layer is not None else None
                ),
                accepted_wkb_type=(
                    accepted.wkbType() if accepted is not None else None
                ),
            )
            self._preparation_task = task
            self._preparation_token = token
            self._set_state(RunFlowState.PREPARING)
            task.progressChanged.connect(self._on_preparation_progress)
            task.taskCompleted.connect(self._on_preparation_completed)
            task.taskTerminated.connect(self._on_preparation_terminated)
            self._emit_preparation_progress(0)
            QgsApplication.taskManager().addTask(task)
        except Exception as exc:
            self._fail_before_run("输入准备失败", str(exc))

    def _matches_preparation(self, task, token: int) -> bool:
        return (
            task is self._preparation_task and self._current_attempt(token) is not None
        )

    def _disconnect_preparation_callbacks(self, task) -> None:
        for signal, callback in (
            (task.progressChanged, self._on_preparation_progress),
            (task.taskCompleted, self._on_preparation_completed),
            (task.taskTerminated, self._on_preparation_terminated),
        ):
            try:
                signal.disconnect(callback)
            except (TypeError, RuntimeError):
                pass
        self._preparation_token = 0

    @pyqtSlot(float)
    def _on_preparation_progress(self, progress: float) -> None:
        task, token = self.sender(), self._preparation_token
        if not self._matches_preparation(task, token):
            return
        if self._state is RunFlowState.SHUTTING_DOWN:
            return
        self._emit_preparation_progress(progress)

    def _emit_preparation_progress(self, progress: float) -> None:
        self.stage_progress.emit(
            {
                "key": "input_preparation",
                "name": "后台冻结与审计输入",
                "index": 1,
                "stage_total": self.STAGE_TOTAL,
                "current": int(progress),
                "total": 100,
                "message": "正在冻结范围、筛选 Tile 并审计 accepted 标签；可以停止",
            }
        )

    @pyqtSlot()
    def _on_preparation_completed(self) -> None:
        task, token = self.sender(), self._preparation_token
        if not self._matches_preparation(task, token):
            return
        self._disconnect_preparation_callbacks(task)
        self._preparation_task = None
        if self._state is RunFlowState.SHUTTING_DOWN:
            self._discard_attempt()
            self._maybe_finish_shutdown()
            return
        if task.isCanceled() or self._state is RunFlowState.STOPPING:
            self._complete_stop_before_run()
            return
        attempt = self._current_attempt(token)
        if attempt is None or task.result_data is None:
            self._fail_before_run(
                "输入准备失败",
                "后台输入准备没有返回冻结结果",
            )
            return
        attempt.prepared = dict(task.result_data)
        self._start_planning(token)

    @pyqtSlot()
    def _on_preparation_terminated(self) -> None:
        task, token = self.sender(), self._preparation_token
        if not self._matches_preparation(task, token):
            return
        self._disconnect_preparation_callbacks(task)
        self._preparation_task = None
        if self._state is RunFlowState.SHUTTING_DOWN:
            self._discard_attempt()
            self._maybe_finish_shutdown()
        elif task.isCanceled() or self._state is RunFlowState.STOPPING:
            self._complete_stop_before_run()
        else:
            self._fail_before_run("输入准备失败", task.error_message)

    def _start_planning(self, token: int) -> None:
        attempt = self._current_attempt(token)
        if attempt is None or attempt.reservation is None:
            return
        request = attempt.request
        prepared = attempt.prepared
        reservation = attempt.reservation
        try:
            builder_kwargs = build_run_builder_kwargs(
                scripts_dir=request.scripts_dir,
                output_root=request.output_root,
                accepted_target_gpkg=request.accepted_target_gpkg,
                raster_layer=request.raster_layer,
                requested_extent=request.requested_extent,
                processing_extent=request.processing_extent,
                grid_tiles=request.grid_tiles,
                active_tiles=tuple(prepared["active_tiles"]),
                range_selection=dict(prepared["range_selection"]),
                effective_config=dict(request.effective_config),
                environment_report=dict(request.environment_report),
                accepted_validation=dict(prepared["accepted_validation"]),
                skip_accepted=request.skip_accepted,
                selected_model_ids=request.selected_model_ids,
                fusion_profile_id=request.fusion_profile_id,
                boundary_smoothing_enabled=request.boundary_smoothing_enabled,
                overlap=request.overlap,
                run_id=reservation.run_id,
                run_dir=reservation.run_dir,
                accepted_snapshot=str(prepared.get("accepted_snapshot") or ""),
                skipped_tiles=tuple(prepared.get("skipped_tiles") or ()),
                tile_cache_sample=dict(attempt.tile_cache_sample),
            )
            task = RunBuilderTask(builder_kwargs)
            self._builder_task = task
            self._builder_token = token
            self._set_state(RunFlowState.PLANNING)
            task.progressChanged.connect(self._on_builder_progress)
            task.taskCompleted.connect(self._on_builder_completed)
            task.taskTerminated.connect(self._on_builder_terminated)
            self.stage_progress.emit(
                {
                    "key": "run_planning",
                    "name": "建立 Run 任务图",
                    "index": 1,
                    "stage_total": self.STAGE_TOTAL,
                    "current": 0,
                    "total": 100,
                    "message": "正在后台建立 PostgreSQL 任务图，界面可以继续响应",
                }
            )
            QgsApplication.taskManager().addTask(task)
        except Exception as exc:
            logger.exception("启动推理异常: %s", exc)
            self._fail_before_run("启动推理失败", str(exc))

    def _matches_builder(self, task, token: int) -> bool:
        return task is self._builder_task and self._current_attempt(token) is not None

    def _disconnect_builder_callbacks(self, task) -> None:
        for signal, callback in (
            (task.progressChanged, self._on_builder_progress),
            (task.taskCompleted, self._on_builder_completed),
            (task.taskTerminated, self._on_builder_terminated),
        ):
            try:
                signal.disconnect(callback)
            except (TypeError, RuntimeError):
                pass
        self._builder_token = 0

    @pyqtSlot(float)
    def _on_builder_progress(self, progress: float) -> None:
        task, token = self.sender(), self._builder_token
        if not self._matches_builder(task, token):
            return
        if self._state is RunFlowState.SHUTTING_DOWN:
            return
        self.stage_progress.emit(
            {
                "key": "run_planning",
                "name": "建立 Run 任务图",
                "index": 1,
                "stage_total": self.STAGE_TOTAL,
                "current": int(progress),
                "total": 100,
                "message": task.progress_message,
            }
        )

    @pyqtSlot()
    def _on_builder_completed(self) -> None:
        task, token = self.sender(), self._builder_token
        if not self._matches_builder(task, token):
            return
        self._disconnect_builder_callbacks(task)
        self._builder_task = None
        if self._state is RunFlowState.SHUTTING_DOWN:
            self._discard_attempt()
            self._maybe_finish_shutdown()
            return
        if task.isCanceled() or self._state is RunFlowState.STOPPING:
            self._complete_stop_before_run()
            return
        if task.result_data is None:
            self._fail_before_run(
                "启动推理失败",
                "后台任务图建立没有返回 Run",
            )
            return
        attempt = self._current_attempt(token)
        runner = self._runner
        if attempt is None or runner is None or token != self._runner_token:
            return
        spec, spec_path, database_path = task.result_data
        attempt.reservation = None
        try:
            self._set_state(RunFlowState.INFERENCING)
            self.monitor_context_ready.emit(
                str(database_path),
                str(spec["run_id"]),
                int((spec.get("scaling") or {}).get("tile_page_size", 500)),
                dict(spec),
            )
            runner.run_from_spec(
                str(spec_path),
                accepted_layer=attempt.request.accepted_layer,
            )
        except Exception as exc:
            logger.exception("启动推理异常: %s", exc)
            self._fail_before_run("启动推理失败", str(exc))

    @pyqtSlot()
    def _on_builder_terminated(self) -> None:
        task, token = self.sender(), self._builder_token
        if not self._matches_builder(task, token):
            return
        self._disconnect_builder_callbacks(task)
        self._builder_task = None
        if self._state is RunFlowState.SHUTTING_DOWN:
            self._discard_attempt()
            self._maybe_finish_shutdown()
        elif task.isCanceled() or self._state is RunFlowState.STOPPING:
            self._complete_stop_before_run()
        else:
            self._fail_before_run(
                "启动推理失败",
                task.error_message or "后台建立 Run 任务图失败",
            )

    @pyqtSlot(object)
    def _on_runner_progress(self, info: object) -> None:
        if self.sender() is not self._runner:
            return
        if self._state not in {
            RunFlowState.INFERENCING,
            RunFlowState.STOPPING,
        }:
            return
        whole = dict(info)
        whole["index"] = int(whole.get("index", 0)) + 1
        whole["stage_total"] = self.STAGE_TOTAL
        self.stage_progress.emit(whole)

    @pyqtSlot(dict)
    def _on_runner_finished(self, result: dict) -> None:
        if self.sender() is not self._runner:
            return
        if self._state not in {
            RunFlowState.INFERENCING,
            RunFlowState.STOPPING,
        }:
            return
        self._attempt = None
        self._set_state(RunFlowState.FINISHED)
        self.finished.emit(dict(result or {}))

    def stop(self) -> None:
        state = self._state
        if state is RunFlowState.PREFLIGHTING:
            self._set_state(RunFlowState.STOPPING)
            self._release_probe(cancel=True)
            self._complete_stop_before_run()
        elif state is RunFlowState.PREPARING and self._preparation_task is not None:
            self._set_state(RunFlowState.STOPPING)
            self._preparation_task.cancel()
        elif state is RunFlowState.PLANNING and self._builder_task is not None:
            self._set_state(RunFlowState.STOPPING)
            self._builder_task.cancel()
        elif state is RunFlowState.INFERENCING and self._runner is not None:
            self._set_state(RunFlowState.STOPPING)
            self._runner.stop()

    def _discard_attempt(self) -> None:
        attempt = self._attempt
        if attempt is not None and attempt.reservation is not None:
            attempt.reservation.discard_if_unused()
            attempt.reservation = None
        self._attempt = None

    def _complete_stop_before_run(self) -> None:
        self._discard_attempt()
        self._set_state(RunFlowState.FINISHED)
        self.stopped_before_run.emit()

    def _fail_before_run(self, title: str, message: str) -> None:
        self._release_probe(cancel=True)
        self._discard_attempt()
        if self._state is RunFlowState.SHUTTING_DOWN:
            self._maybe_finish_shutdown()
            return
        self._set_state(RunFlowState.FINISHED)
        self.pre_run_failed.emit(str(title), str(message))

    def _retire_current_runner(self) -> None:
        runner = self._runner
        self._runner = None
        self._runner_token = 0
        if runner is None:
            return
        self.runner_changed.emit(None)
        self._disconnect_runner_callbacks(runner)
        retire_after(runner, runner.shutdown_finished)
        runner.shutdown()

    def shutdown(self) -> None:
        if self._state is RunFlowState.SHUTTING_DOWN:
            return
        self._shutdown_emitted = False
        self._set_state(RunFlowState.SHUTTING_DOWN)
        self._release_probe(cancel=True)
        if self._preparation_task is not None:
            self._preparation_task.cancel()
        if self._builder_task is not None:
            self._builder_task.cancel()
        if self._preparation_task is None and self._builder_task is None:
            self._discard_attempt()
        self._shutdown_current_runner()
        self._maybe_finish_shutdown()

    def _shutdown_current_runner(self) -> None:
        runner = self._runner
        self._runner = None
        self._runner_token = 0
        if runner is None:
            return
        self.runner_changed.emit(None)
        self._disconnect_runner_callbacks(runner)
        self._shutdown_runner = runner
        runner.shutdown_finished.connect(self._on_runner_shutdown_finished)
        retire_after(runner, runner.shutdown_finished)
        runner.shutdown()

    @pyqtSlot()
    def _on_runner_shutdown_finished(self) -> None:
        runner = self.sender()
        if runner is not self._shutdown_runner:
            return
        try:
            runner.shutdown_finished.disconnect(self._on_runner_shutdown_finished)
        except (TypeError, RuntimeError):
            pass
        self._shutdown_runner = None
        self._maybe_finish_shutdown()

    def _maybe_finish_shutdown(self) -> None:
        if self._state is not RunFlowState.SHUTTING_DOWN:
            return
        if any(
            owner is not None
            for owner in (
                self._probe,
                self._preparation_task,
                self._builder_task,
                self._shutdown_runner,
            )
        ):
            return
        if self._attempt is not None:
            self._discard_attempt()
        if self._shutdown_emitted:
            return
        self._shutdown_emitted = True
        self.shutdown_finished.emit()
