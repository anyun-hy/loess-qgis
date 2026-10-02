"""Shared runtime error identities without pipeline dependencies."""


class WorkPackageRuntimeError(RuntimeError):
    """A Work Package runtime contract was rejected."""


class LeaseLostError(WorkPackageRuntimeError):
    """The current worker no longer owns the exact database lease."""


class WorkerStopRequested(WorkPackageRuntimeError):
    """The accelerator worker received a coordinated stop request."""


def storage_error_is_transient(error: BaseException) -> bool:
    """Return whether a storage failure permits a same-lease retry."""

    return bool(getattr(error, "transient", False))
