"""QGIS 4.2/Qt6 QProcess setup with an isolated Unix child session."""

from __future__ import annotations

from qgis.PyQt.QtCore import QProcess


def configure_process(process, program, arguments):
    """Configure a Qt6 QProcess in its own Unix session."""
    process.setProgram(program)
    process.setArguments(list(arguments))
    parameters = QProcess.UnixProcessParameters()
    parameters.flags = QProcess.UnixProcessFlag.CreateNewSession
    process.setUnixProcessParameters(parameters)
    return True


def process_is_running(process):
    """Return whether a Qt6 QProcess is still active."""
    return process.state() != QProcess.ProcessState.NotRunning
