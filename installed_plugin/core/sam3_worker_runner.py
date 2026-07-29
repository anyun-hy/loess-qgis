"""Persistent QProcess controller for the interactive SAM3 JSONL worker."""

from __future__ import annotations

import json
import os
import shlex
import signal

from qgis.PyQt.QtCore import QObject, QProcess, QProcessEnvironment, pyqtSignal

from .process_compat import configure_linux_process


class Sam3WorkerRunner(QObject):
    log_line = pyqtSignal(str, str)
    event_received = pyqtSignal(object)
    ready = pyqtSignal(object)
    stopped = pyqtSignal(object)

    def __init__(self, scripts_dir, sam_config, parent=None):
        super().__init__(parent)
        self.scripts_dir = os.path.abspath(str(scripts_dir))
        self.sam_config = dict(sam_config or {})
        self._process = None
        self._owns_process_group = False
        self._stdout_pending = bytearray()
        self._stderr_pending = bytearray()
        self._ready = False
        self._stopping = False

    @property
    def is_running(self):
        return bool(
            self._process is not None
            and self._process.state() != QProcess.NotRunning
        )

    @property
    def is_ready(self):
        return self.is_running and self._ready

    def start(self):
        if self.is_running:
            return
        script = os.path.join(self.scripts_dir, "run_sam3_interactive_worker.sh")
        if not os.path.isfile(script):
            raise FileNotFoundError(script)
        checkpoint = str(self.sam_config.get("checkpoint") or "")
        checkpoint_sha = str(self.sam_config.get("checkpoint_sha256") or "")
        if not os.path.isfile(checkpoint) or len(checkpoint_sha) != 64:
            raise RuntimeError("SAM3 checkpoint or SHA256 is unavailable")
        device = str(
            self.sam_config.get("effective_device")
            or self.sam_config.get("requested_device")
            or "cpu"
        )
        args = [
            script,
            "--checkpoint", checkpoint,
            "--checkpoint-sha256", checkpoint_sha,
            "--device", device,
            "--sam-version", str(self.sam_config.get("version") or ""),
        ]
        self.log_line.emit("system", "[cmd] " + shlex.join(["/bin/bash", *args]))
        process = QProcess(self)
        self._owns_process_group = configure_linux_process(
            process, "/bin/bash", args
        )
        process.setWorkingDirectory(self.scripts_dir)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.finished.connect(self._finished)
        self._stdout_pending = bytearray()
        self._stderr_pending = bytearray()
        self._ready = False
        self._stopping = False
        self._process = process
        process.start()

    def predict(self, request):
        if not self.is_ready:
            raise RuntimeError("SAM3 worker is not ready")
        payload = dict(request)
        session_id = str(payload.get("session_id") or "")
        if not session_id:
            raise ValueError("SAM3 predict requires session_id")
        self._write({
            "command": "start_session",
            "session_id": session_id,
            "run_id": str(payload.get("run_id") or ""),
            "class_code": payload.get("class_code"),
            "object_id": str(payload.get("object_id") or ""),
        })
        payload["command"] = "predict"
        self._write(payload)

    def cancel(self, session_id):
        if self.is_running and session_id:
            self._write({"command": "cancel", "session_id": str(session_id)})

    def close_session(self, session_id):
        if self.is_running and session_id:
            self._write({"command": "close_session", "session_id": str(session_id)})

    def _write(self, payload):
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        written = self._process.write(data.encode("utf-8"))
        if written < 0:
            raise RuntimeError("cannot write to the SAM3 worker")

    def stop(self):
        process = self._process
        if process is None:
            return
        self._stopping = True
        if process.state() != QProcess.NotRunning:
            try:
                self._write({"command": "shutdown"})
                process.waitForBytesWritten(1000)
            except Exception:
                pass
            if not process.waitForFinished(8000):
                pid = int(process.processId())
                if pid > 0 and self._owns_process_group:
                    try:
                        os.killpg(pid, signal.SIGTERM)
                    except (ProcessLookupError, OSError):
                        process.terminate()
                else:
                    process.terminate()
                if not process.waitForFinished(4000):
                    if pid > 0 and self._owns_process_group:
                        try:
                            os.killpg(pid, signal.SIGKILL)
                        except (ProcessLookupError, OSError):
                            process.kill()
                    else:
                        process.kill()
                    process.waitForFinished(3000)
        if self._process is process:
            self._process = None
        self._ready = False
        self._owns_process_group = False
        process.deleteLater()
        self.stopped.emit({"expected": True})

    def _read_stdout(self):
        if self._process is None:
            return
        self._stdout_pending.extend(bytes(self._process.readAllStandardOutput()))
        self._flush(self._stdout_pending, "stdout")

    def _read_stderr(self):
        if self._process is None:
            return
        self._stderr_pending.extend(bytes(self._process.readAllStandardError()))
        self._flush(self._stderr_pending, "stderr")

    def _flush(self, buffer, level, final=False):
        value = buffer.decode("utf-8", errors="replace")
        lines = value.split("\n")
        remainder = "" if final else lines.pop()
        buffer[:] = remainder.encode("utf-8")
        for line in lines:
            line = line.rstrip("\r")
            if not line:
                continue
            self.log_line.emit(level, line)
            if level != "stdout":
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or not event.get("event"):
                continue
            if event["event"] == "worker_ready":
                self._ready = True
                self.ready.emit(event)
            self.event_received.emit(event)

    def _finished(self, exit_code, _exit_status):
        process = self._process
        if process is None:
            return
        self._read_stdout()
        self._read_stderr()
        self._flush(self._stdout_pending, "stdout", final=True)
        self._flush(self._stderr_pending, "stderr", final=True)
        self._process = None
        self._ready = False
        self._owns_process_group = False
        process.deleteLater()
        self.stopped.emit({
            "expected": self._stopping,
            "returncode": int(exit_code),
        })
