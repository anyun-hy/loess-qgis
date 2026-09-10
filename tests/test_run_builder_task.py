"""Regression tests for non-blocking QGIS Run graph creation."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCK_SOURCE = (
    ROOT / "qgis_plugins" / "labeling_tool" / "gui" / "main_dock.py"
).read_text(encoding="utf-8")


class _FakeQgsTask:
    class Flag:
        CanCancel = 1

    def __init__(self, description, flags):
        self.description = description
        self.flags = flags
        self._canceled = False
        self.progress_values = []

    def setProgress(self, value):
        self.progress_values.append(float(value))

    def isCanceled(self):
        return self._canceled

    def cancel(self):
        self._canceled = True


def _load_task_module(monkeypatch):
    qgis_module = types.ModuleType("qgis")
    qgis_core_module = types.ModuleType("qgis.core")
    qgis_core_module.QgsTask = _FakeQgsTask
    qgis_module.core = qgis_core_module
    monkeypatch.setitem(sys.modules, "qgis", qgis_module)
    monkeypatch.setitem(sys.modules, "qgis.core", qgis_core_module)
    sys.modules.pop("labeling_tool.core.run_builder_task", None)
    return importlib.import_module("labeling_tool.core.run_builder_task")


def test_main_dock_submits_run_graph_creation_to_qgis_task_manager():
    preparation = DOCK_SOURCE.split(
        "def _start_inference_after_tile_cache_probe", 1
    )[1].split("def _on_run_builder_progress", 1)[0]
    completion = DOCK_SOURCE.split("def _on_run_builder_completed", 1)[1].split(
        "def _on_run_builder_terminated", 1
    )[0]

    assert "RunBuilderTask(builder_kwargs)" in preparation
    assert 'self._pipeline_state = "planning"' in preparation
    assert "QgsApplication.taskManager().addTask(task)" in preparation
    assert "create_v5_run(" not in preparation
    assert "self.runner.run_from_spec(" in completion
    assert 'self._pipeline_state = "inferencing"' in completion


def test_run_builder_task_forwards_progress_and_returns_result(monkeypatch):
    module = _load_task_module(monkeypatch)
    expected = ({"run_id": "run-1"}, Path("run_spec.json"), "postgresql")

    def fake_create(**kwargs):
        assert callable(kwargs["is_canceled"])
        assert kwargs["is_canceled"]() is False
        kwargs["progress"](68, "Tile 索引已写入")
        return expected

    monkeypatch.setattr(module, "create_v5_run", fake_create)
    task = module.RunBuilderTask({"run_id": "run-1"})

    assert task.run() is True
    assert task.result_data == expected
    assert task.progress_values == [68.0]
    assert task.progress_message == "Tile 索引已写入"
    sys.modules.pop("labeling_tool.core.run_builder_task", None)


def test_cancelled_run_builder_marks_partial_run_stopped(monkeypatch):
    module = _load_task_module(monkeypatch)
    status_updates = []

    def fake_create(**kwargs):
        kwargs["progress"](22, "Partition 与 Work Package 已写入")
        raise module.RunBuilderV5Cancelled("cancelled")

    class FakeDatabase:
        def __init__(self, location, *, postgres_schema):
            assert location == "dbname=test"
            assert postgres_schema == "test_schema"

        def set_run_status(self, run_id, status, *, expected):
            status_updates.append((run_id, status, expected))
            return True

    monkeypatch.setattr(module, "create_v5_run", fake_create)
    monkeypatch.setattr(module, "RunStateDB", FakeDatabase)
    monkeypatch.setattr(module, "production_state_schema", lambda: "test_schema")
    task = module.RunBuilderTask(
        {"run_id": "run-2", "state_database": "dbname=test"}
    )
    task.cancel()

    assert task.run() is False
    assert task.result_data is None
    assert task.error_message == ""
    assert status_updates == [("run-2", "stopped", "planned")]
    sys.modules.pop("labeling_tool.core.run_builder_task", None)
