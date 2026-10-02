"""Thread-affinity contracts plus a real native Qt event-loop regression.

Set LOESS_TEST_QGIS_PYTHON to QGIS's Python executable and, if required,
LOESS_TEST_QGIS_PYTHONPATH to its Python paths. Only the subprocess uses them;
pytest and the rest of the project tests continue to run inside Conda qgis.
"""

import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'src/labeling_tool/runs/v5_async_runner.py'


def test_worker_connections_created_before_move_use_native_slots():
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "V5AsyncInferenceRunner")
    methods = {node.name: node for node in worker.body if isinstance(node, ast.FunctionDef)}
    handlers = []
    for node in ast.walk(methods["__init__"]):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "connect"):
            continue
        assert len(node.args) == 1 and isinstance(node.args[0], ast.Attribute)
        target = node.args[0]
        assert isinstance(target.value, ast.Name) and target.value.id == "self"
        handler = methods[target.attr]
        slots = [item for item in handler.decorator_list
                 if isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
                 and item.func.id == "pyqtSlot"]
        assert len(slots) == 1, f"{target.attr} would retain a GUI-thread slot proxy"
        expected = ["str", "str"] if target.attr == "_persist_log" else []
        assert [ast.unparse(arg) for arg in slots[0].args] == expected
        handlers.append(target.attr)
    assert set(handlers) == {
        "_schedule_safely", "_heartbeat_and_watchdog", "_flush_job_heartbeats",
        "_flush_ui_events", "_persist_log",
    }


def test_native_worker_callbacks_and_main_loop_responsiveness():
    executable = os.environ.get("LOESS_TEST_QGIS_PYTHON") or sys.executable
    environment = os.environ.copy()
    pythonpath = environment.get("LOESS_TEST_QGIS_PYTHONPATH")
    if pythonpath is not None:
        environment["PYTHONPATH"] = pythonpath
    result = subprocess.run(
        [executable, "-B", str(ROOT / 'tests/support/qt_runner_thread_probe.py'),
         str(ROOT / "src"), str(ROOT / "scripts/runtime")],
        env=environment, capture_output=True, text=True, timeout=20,
    )
    if result.returncode == 77 and "LOESS_TEST_QGIS_PYTHON" not in environment:
        pytest.skip("Native QGIS unavailable; set LOESS_TEST_QGIS_PYTHON for Qt acceptance")
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout.strip())
