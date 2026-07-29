"""Qt5-compatible Linux QProcess setup with an isolated process group."""

from __future__ import annotations

import shutil


def configure_linux_process(process, program, arguments):
    """Configure a QProcess and return whether its PID owns a new session."""
    setsid = shutil.which("setsid")
    if setsid:
        process.setProgram(setsid)
        process.setArguments([program, *list(arguments)])
        return True
    process.setProgram(program)
    process.setArguments(list(arguments))
    return False
