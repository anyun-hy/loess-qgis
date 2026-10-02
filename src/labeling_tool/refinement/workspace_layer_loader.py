"""Schedule incremental loading of class-workspace layers on the GUI thread.

Public entry: :class:`WorkspaceLayerLoader`.  The loader owns only the pending
class-code queue, activation intent, and its single-shot timer.  Its injected
callback performs the actual QGIS layer work; consumers update UI through the
signals below.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from qgis.PyQt.QtCore import QObject, QTimer, pyqtSignal


class WorkspaceLayerLoader(QObject):
    """Load one class at a time while guarding synchronous Qt re-entry."""

    loaded = pyqtSignal(int, bool)
    failed = pyqtSignal(int, str)
    completed = pyqtSignal()

    _LOAD_INTERVAL_MS = 250

    def __init__(
        self,
        load_layer: Callable[[int], None],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._load_layer = load_layer
        self._pending_codes: list[int] = []
        self._active_code: int | None = None
        self._paused = True
        self._generation = 0
        self._inflight = False
        self._inflight_generation: int | None = None
        self._phase = "idle"
        self._deferred_start: tuple[int, int] | None = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._load_next)

    @property
    def pending_codes(self) -> tuple[int, ...]:
        """Return the queued class codes in their current load order."""
        return tuple(self._pending_codes)

    def start(self, class_codes: Iterable[int]) -> None:
        """Replace the queue and begin a new loading round after 250 ms."""
        self._invalidate()
        self._pending_codes = list(dict.fromkeys(int(code) for code in class_codes))
        self._active_code = None
        self._paused = False
        self._schedule(self._LOAD_INTERVAL_MS)

    def prioritize(self, code: int, activate: bool = False) -> None:
        """Move one class to the front and resume an idle or paused loader."""
        code = int(code)
        self._pending_codes = [
            pending for pending in self._pending_codes if pending != code
        ]
        self._pending_codes.insert(0, code)
        if activate:
            self._active_code = code
        self._paused = False

        if self._inflight:
            if self._inflight_generation != self._generation or self._phase in {
                "failed",
                "completed",
            }:
                self._deferred_start = (self._generation, 0)
            return
        if not self._timer.isActive():
            self._timer.start(0)

    def pause(self) -> None:
        """Stop scheduling while preserving the queue and activation intent."""
        self._invalidate()
        self._paused = True

    def reset(self) -> None:
        """Stop scheduling and discard the queue and activation intent."""
        self._invalidate()
        self._paused = True
        self._pending_codes.clear()
        self._active_code = None

    def _invalidate(self) -> None:
        self._generation += 1
        self._timer.stop()
        self._deferred_start = None

    def _schedule(self, delay_ms: int) -> None:
        if self._paused:
            return
        if self._inflight:
            self._deferred_start = (self._generation, delay_ms)
            return
        self._timer.start(delay_ms)

    def _load_next(self) -> None:
        if self._paused or self._inflight:
            return
        generation = self._generation
        self._inflight = True
        self._inflight_generation = generation
        self._phase = "loading"
        try:
            if not self._pending_codes:
                self._phase = "completed"
                self.completed.emit()
                return

            code = self._pending_codes.pop(0)
            try:
                self._load_layer(code)
            except Exception as exc:
                if generation != self._generation:
                    return
                self._paused = True
                self._phase = "failed"
                self.failed.emit(code, str(exc))
                if generation == self._generation and not self._paused:
                    self._deferred_start = (generation, 0)
                return

            if generation != self._generation:
                return
            activate = self._active_code == code
            if activate:
                self._active_code = None
            self._phase = "loaded"
            self.loaded.emit(code, activate)
            if generation != self._generation or self._paused:
                return
            if self._pending_codes:
                self._deferred_start = (generation, self._LOAD_INTERVAL_MS)
            else:
                self._phase = "completed"
                self.completed.emit()
        finally:
            self._inflight = False
            self._inflight_generation = None
            self._phase = "idle"
            self._start_deferred()

    def _start_deferred(self) -> None:
        deferred = self._deferred_start
        self._deferred_start = None
        if deferred is None:
            return
        generation, delay_ms = deferred
        if generation == self._generation and not self._paused:
            self._timer.start(delay_ms)
