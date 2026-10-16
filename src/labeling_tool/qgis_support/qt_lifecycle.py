"""Keep asynchronous owners alive until their completion signal is delivered."""

from qgis.PyQt.QtCore import QCoreApplication, QEventLoop, QThread

_retiring = set()
_awaiting_shutdown = set()


def retire_after(owner, finished):
    """Detach from a closing widget; delete only after owned work has stopped.

    Call on the GUI thread, before requesting shutdown. Keeping a Python
    reference is intentional: a parent widget must not destroy a live QThread
    or QProcess while an asynchronous cancellation is still in progress.
    """
    if owner in _retiring:
        return
    owner.setParent(None)
    _retiring.add(owner)
    _awaiting_shutdown.add(owner)

    def on_finished():
        _awaiting_shutdown.discard(owner)
        owner.deleteLater()

    finished.connect(on_finished)
    owner.destroyed.connect(lambda _object=None: _retiring.discard(owner))


def drain_retired_owners():
    """Finish asynchronous cleanup before the plugin can be unloaded.

    Only use at the final plugin unload boundary, never for ordinary Stop or
    dialog closure. QGIS unloads plugins during aboutToQuit: after returning,
    the main loop will no longer deliver the queued completion callbacks.
    Process those callbacks while the plugin and its owners are still alive.
    Exclude user input to prevent new UI work. Wait for completion, not object
    deletion: Qt can defer deletion until the current aboutToQuit slot returns.
    A timeout must not release owners whose native threads are still running.
    """
    while _awaiting_shutdown:
        QCoreApplication.processEvents(
            QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents, 25
        )
        if _awaiting_shutdown:
            QThread.msleep(5)
