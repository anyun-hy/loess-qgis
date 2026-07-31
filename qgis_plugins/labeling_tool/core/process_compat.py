"""Cross-platform QProcess setup with an isolated child process group."""

from __future__ import annotations

import os
import shutil


def _configure_qt6_unix_session(process):
    process_type = type(process)
    parameters_type = getattr(process_type, "UnixProcessParameters", None)
    flag_type = getattr(process_type, "UnixProcessFlag", None)
    setter = getattr(process, "setUnixProcessParameters", None)
    if parameters_type is None or flag_type is None or setter is None:
        return False
    create_session = getattr(flag_type, "CreateNewSession", None)
    if create_session is None:
        return False
    parameters = parameters_type()
    parameters.flags = create_session
    setter(parameters)
    return True


def configure_process(process, program, arguments):
    """Configure a shared QProcess and return whether it owns a new session."""
    process.setProgram(program)
    process.setArguments(list(arguments))
    if _configure_qt6_unix_session(process):
        return True
    child_modifier = getattr(process, "setChildProcessModifier", None)
    if child_modifier is not None and hasattr(os, "setsid"):
        child_modifier(os.setsid)
        return True
    setsid = shutil.which("setsid") if os.name == "posix" else None
    if setsid:
        process.setProgram(setsid)
        process.setArguments([program, *list(arguments)])
        return True
    return False


def process_is_running(process):
    """Return process activity without exposing Qt5/Qt6 enum spelling."""
    process_type = type(process)
    state_type = getattr(process_type, "ProcessState", None)
    not_running = (
        getattr(state_type, "NotRunning", None)
        if state_type is not None
        else None
    )
    if not_running is None:
        not_running = getattr(process_type, "NotRunning")
    return process.state() != not_running
