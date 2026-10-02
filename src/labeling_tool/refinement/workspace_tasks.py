"""Own class-workspace probe and initialization task lifecycles.

Public entry: :class:`WorkspaceTasks`.  The owner builds and submits the two
QgsTask workers, rejects stale callbacks by task identity and generation, and
publishes plain payloads to the GUI.  It does not retain workspace state or
delete tasks owned by the QGIS task manager.
"""

from __future__ import annotations

from collections.abc import Callable

from qgis.core import QgsApplication, QgsTask
from qgis.PyQt.QtCore import QObject, pyqtSignal, pyqtSlot

from labeling_tool.refinement import class_workspace


class WorkspaceTasks(QObject):
    """Coordinate workers on the GUI thread and reject stale results.

    Call public methods from the object's GUI thread so the native Qt slots
    receive task-manager signals there.
    """

    progress = pyqtSignal(str, float)
    probed = pyqtSignal(object)
    initialized = pyqtSignal(object)
    terminated = pyqtSignal(str, str)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        submit_task: Callable[[QgsTask], object] | None = None,
    ) -> None:
        super().__init__(parent)
        self._submit_task = submit_task
        self._task: QgsTask | None = None
        self._operation = ""
        self._generation = 0

    @property
    def busy(self) -> bool:
        """Return whether a current worker is still owned by this coordinator."""
        return self._task is not None

    def probe(self, run_spec: dict, streams: list[dict]) -> None:
        """Replace any current worker with a probe for the supplied Run."""
        self.cancel()
        generation = self._next_generation()
        try:
            task = class_workspace.ClassWorkspaceProbeTask(
                generation,
                run_spec,
                streams,
            )
        except Exception as exc:
            self.terminated.emit("probe", str(exc))
            return
        self._start("probe", task, generation)

    def initialize(
        self,
        run_spec: dict,
        stream: dict,
        *,
        replace: bool = False,
    ) -> None:
        """Start initialization, refusing to replace an active worker."""
        if self.busy:
            raise RuntimeError("workspace task is already running")
        generation = self._next_generation()
        try:
            task = class_workspace.ClassWorkspaceInitializeTask(
                generation,
                run_spec,
                stream,
                replace=replace,
            )
        except Exception as exc:
            self.terminated.emit("initialize", str(exc))
            return
        self._start("initialize", task, generation)

    def cancel(self) -> None:
        """Invalidate then request cancellation without waiting for exit.

        The QGIS task manager continues to own the worker until it finishes.
        """
        task = self._invalidate_current()
        if task is not None:
            task.cancel()

    def _next_generation(self) -> int:
        self._generation += 1
        return self._generation

    def _invalidate_current(self) -> QgsTask | None:
        task = self._task
        self._generation += 1
        self._task = None
        self._operation = ""
        return task

    def _start(self, operation: str, task: QgsTask, generation: int) -> None:
        self._task = task
        self._operation = operation
        task.progressChanged.connect(self._on_progress)
        task.taskCompleted.connect(self._on_completed)
        task.taskTerminated.connect(self._on_terminated)
        try:
            submit_task = self._submit_task or QgsApplication.taskManager().addTask
            submit_task(task)
        except Exception as exc:
            if self._matches(task, generation):
                self._invalidate_current()
                task.cancel()
                self.terminated.emit(operation, str(exc))

    def _matches(self, task: object, generation: int | None = None) -> bool:
        if task is not self._task:
            return False
        task_generation = getattr(task, "generation", None)
        expected = self._generation if generation is None else generation
        return task_generation == expected == self._generation

    def _take_current(self, task: object) -> tuple[str, QgsTask] | None:
        if not self._matches(task):
            return None
        operation = self._operation
        current = self._task
        self._task = None
        self._operation = ""
        if current is None:
            return None
        return operation, current

    @pyqtSlot(float)
    def _on_progress(self, value: float) -> None:
        task = self.sender()
        if not self._matches(task):
            return
        self.progress.emit(self._operation, float(value))

    @pyqtSlot()
    def _on_completed(self) -> None:
        current = self._take_current(self.sender())
        if current is None:
            return
        operation, task = current
        payload = getattr(task, "result_data", None) or {}
        if operation == "probe":
            self.probed.emit(payload)
        else:
            # Initialization may have committed atomically before a late
            # cancellation skipped optional statistics.  A completed task's
            # payload remains authoritative even when isCanceled() is true.
            self.initialized.emit(payload)

    @pyqtSlot()
    def _on_terminated(self) -> None:
        current = self._take_current(self.sender())
        if current is None:
            return
        operation, task = current
        self.terminated.emit(
            operation,
            str(getattr(task, "error_message", "") or ""),
        )
