"""Keep asynchronous owners alive until their completion signal is delivered."""

_retiring = set()


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
    finished.connect(owner.deleteLater)
    owner.destroyed.connect(lambda _object=None: _retiring.discard(owner))
