import ast
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from labeling_tool.shared.contracts.run_spec import (
    source_raster_identity,
)
from loess_runtime.inference.tile_cache_probe import measure_tile_cache
from loess_runtime.inference.tile_materializer import (
    TILE_MATERIALIZATION_METHOD_VERSION,
    materialize_tile,
)

ROOT = Path(__file__).resolve().parents[2]


class _Signal:
    def __init__(self):
        self.values = []

    def connect(self, _callback):
        return None

    def emit(self, *args):
        self.values.append(args)


class _QObject:
    def __init__(self, *_args, **_kwargs):
        pass


class _QProcess:
    NotRunning = 0


class _QProcessEnvironment:
    @staticmethod
    def systemEnvironment():
        return _QProcessEnvironment()


def _load_probe_runner(monkeypatch):
    qgis_module = types.ModuleType("qgis")
    pyqt_module = types.ModuleType("qgis.PyQt")
    qtcore_module = types.ModuleType("qgis.PyQt.QtCore")
    qtcore_module.QObject = _QObject
    qtcore_module.QProcess = _QProcess
    qtcore_module.QProcessEnvironment = _QProcessEnvironment
    qtcore_module.pyqtSignal = lambda *_args: _Signal()
    monkeypatch.setitem(sys.modules, "qgis", qgis_module)
    monkeypatch.setitem(sys.modules, "qgis.PyQt", pyqt_module)
    monkeypatch.setitem(sys.modules, "qgis.PyQt.QtCore", qtcore_module)
    sys.modules.pop('labeling_tool.runs.tile_cache_probe_runner', None)
    return importlib.import_module('labeling_tool.runs.tile_cache_probe_runner')


class _FinishedProcess:
    def __init__(self, stdout=b"", stderr=b"", error="failed to start"):
        self.stdout = bytearray(stdout)
        self.stderr = bytearray(stderr)
        self.error = error
        self.deleted = 0
        self.blocked = 0

    def readAllStandardOutput(self):
        value = bytes(self.stdout)
        self.stdout.clear()
        return value

    def readAllStandardError(self):
        value = bytes(self.stderr)
        self.stderr.clear()
        return value

    def errorString(self):
        return self.error

    def deleteLater(self):
        self.deleted += 1

    def blockSignals(self, _value):
        self.blocked += 1


def _runner(module, tmp_path):
    runner = module.TileCacheProbeRunner.__new__(module.TileCacheProbeRunner)
    runner._process = None
    runner._owns_process_group = False
    runner._stdout = bytearray()
    runner._stderr = bytearray()
    runner._generation = 1
    runner._expected = {}
    runner._probe_dir = None
    runner.succeeded = _Signal()
    runner.failed = _Signal()
    runner.log_line = _Signal()
    runner.scripts_dir = str(tmp_path)
    return runner


def _report(expected):
    return {
        "schema_version": 1,
        "kind": "tile_cache_probe",
        "status": "passed",
        **dict(expected),
        "sample_source_window": {"x0": 512, "y0": 0, "x1": 1024, "y1": 512},
        "width": 512,
        "height": 512,
        "band_count": 3,
        "uncompressed_bytes": 3 * 512 * 512 * 2,
        "materialized_tile_bytes": 1000,
        "metadata_bytes": 100,
        "materialized_cache_bytes": 1100,
        "measurement_method": "tile_materializer.materialize_tile",
        "measurement_method_version": TILE_MATERIALIZATION_METHOD_VERSION,
    }


def _prepare_inputs_function(namespace):
    tree = ast.parse(
        (ROOT / "src" / "labeling_tool" / "runs" / "run_preparation_task.py").read_text(
            encoding="utf-8"
        )
    )
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RunPreparationTask"
    )
    function_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run"
    )
    module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(module)
    values = {"Path": Path, **namespace}
    exec(compile(module, "run_preparation_task.py", "exec"), values)
    return values["run"]


def _compressed_uint16_source(path, *, bands=3):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=1024,
        height=512,
        count=bands,
        dtype="uint16",
        crs="EPSG:3857",
        transform=from_origin(0, 512, 1, 1),
        compress="deflate",
    ) as destination:
        destination.write(np.zeros((bands, 512, 1024), dtype=np.uint16))


def _request():
    return {
        "tile_id": "0_1",
        "row_no": 0,
        "col_no": 1,
        "bounds": {"xmin": 512, "ymin": 0, "xmax": 1024, "ymax": 512},
    }


def test_probe_uses_production_materializer_and_measures_real_uint16_tile(tmp_path):
    source = tmp_path / "compressed-source.tif"
    output_root = tmp_path / "output"
    output_root.mkdir()
    _compressed_uint16_source(source)

    report = measure_tile_cache(source, output_root, _request())

    direct = materialize_tile(
        {"path": str(source), "file_identity": source_raster_identity(source)},
        tmp_path / "direct",
        _request(),
    )
    assert report["status"] == "passed"
    assert report["measurement_method"] == "tile_materializer.materialize_tile"
    assert (
        report["measurement_method_version"]
        == TILE_MATERIALIZATION_METHOD_VERSION
    )
    assert report["sample_source_path"] == str(source.resolve())
    assert report["measurement_workspace"] == str(output_root.resolve())
    assert Path(report["sample_artifact_directory"]).parent == output_root.resolve()
    assert report["sample_source_window"] == {
        "x0": 512,
        "y0": 0,
        "x1": 1024,
        "y1": 512,
    }
    assert report["uncompressed_bytes"] == 3 * 512 * 512 * 2
    assert report["materialized_tile_bytes"] == direct["materialized_tile_bytes"]
    assert report["materialized_cache_bytes"] == (
        report["materialized_tile_bytes"] + report["metadata_bytes"]
    )
    # The compressed source is deliberately tiny; a source-file ratio would
    # not equal the production-format Tile measurement.
    assert source.stat().st_size < report["materialized_tile_bytes"]
    assert not list(output_root.glob(".loess-tile-cache-probe-*"))


def test_named_probe_directory_is_exact_and_removed(tmp_path):
    source = tmp_path / "source.tif"
    output_root = tmp_path / "output"
    output_root.mkdir()
    _compressed_uint16_source(source)
    token = "a" * 32

    report = measure_tile_cache(
        source, output_root, _request(), probe_token=token
    )

    assert report["probe_token"] == token
    assert report["sample_artifact_directory"] == str(
        output_root / f".loess-tile-cache-probe-{token}"
    )
    assert not Path(report["sample_artifact_directory"]).exists()


def test_probe_failure_removes_disposable_directory(tmp_path):
    source = tmp_path / "two-band.tif"
    output_root = tmp_path / "output"
    output_root.mkdir()
    _compressed_uint16_source(source, bands=2)

    with pytest.raises(Exception, match="at least 3 bands"):
        measure_tile_cache(source, output_root, _request())

    assert list(output_root.iterdir()) == []


def test_probe_shell_uses_deployed_conda_environment():
    source = (ROOT / "scripts" / "runtime" / "run_tile_cache_probe.sh").read_text(
        encoding="utf-8"
    )
    assert 'source "$SCRIPT_DIR/config.sh"' in source
    assert '"$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV"' in source
    assert "python -m loess_runtime.inference.tile_cache_probe" in source


def test_workflow_blocks_on_real_probe_before_freezing_inputs():
    workflow = (
        ROOT / "src" / "labeling_tool" / "runs" / "run_workflow.py"
    ).read_text(encoding="utf-8")
    planning = (
        ROOT / "src" / "labeling_tool" / "runs" / "run_planning.py"
    ).read_text(encoding="utf-8")
    probe_block = workflow.split("def _start_probe", 1)[1].split(
        "def _matches_probe", 1
    )[0]
    ready_block = workflow.split("def _on_probe_ready", 1)[1].split(
        "def _on_probe_failed", 1
    )[0]
    prepare_block = workflow.split("def _start_preparation", 1)[1].split(
        "def _matches_preparation", 1
    )[0]

    assert "active_tiles = sorted(" in probe_block
    assert 'key=lambda item: (int(item["row"]), int(item["col"]))' in probe_block
    assert "TileCacheProbeRunner(" in probe_block
    assert "probe.succeeded.connect" in probe_block
    assert "probe.failed.connect" in probe_block
    assert "attempt.tile_cache_sample = dict(measurement)" in ready_block
    assert ready_block.index("attempt.tile_cache_sample") < ready_block.index(
        "self._start_preparation(token)"
    )
    assert prepare_block.index("reserve_run_directory(") < prepare_block.index(
        "RunPreparationTask("
    )
    assert "request.get_valid_range_layer()" in prepare_block
    assert "QgsVectorLayerFeatureSource(" in prepare_block
    assert "QgsApplication.taskManager().addTask(task)" in prepare_block
    assert 'tile_cache_sample.get("materialized_cache_bytes")' in planning
    assert 'storage["input_tile_sample"] = dict(tile_cache_sample)' in planning
    assert "source_bytes * pixel_count / raster_pixels" not in planning


def test_snapshot_time_accepted_identity_replaces_start_time_audit_and_skips(
    tmp_path,
):
    live_layer = object()
    frozen_layer = object()
    snapshot_calls = []
    tile_checks = []

    class DifferenceFilter:
        @staticmethod
        def write_source_snapshot(layer, _wkb, output_path, _name, _context, _cancel):
            assert layer is live_layer
            Path(output_path).write_bytes(b"frozen accepted")
            snapshot_calls.append(Path(output_path))
            return str(output_path)

        @staticmethod
        def tile_is_fully_accepted(bounds, layer, crs):
            assert layer is frozen_layer
            assert crs == "EPSG:3857"
            tile_checks.append(bounds)
            return bounds == "covered_at_snapshot_time"

    class AcceptedIntegrity:
        @staticmethod
        def audit_accepted_layer(layer, *, overlap_tolerance, expected_crs, is_canceled):
            assert layer is frozen_layer
            assert overlap_tolerance == pytest.approx(0.25)
            assert expected_crs == "EPSG:3857"
            return {
                "status": "passed",
                "feature_count": 2,
                "overlap_tolerance": overlap_tolerance,
                "identity": "snapshot-time",
            }

    def vector_layer(uri, name, provider):
        assert uri.endswith("accepted_snapshot.gpkg|layername=accepted_labels")
        assert name == "accepted_audit"
        assert provider == "ogr"
        return frozen_layer

    function = _prepare_inputs_function(
        {
            "difference_filter": DifferenceFilter,
            "accepted_integrity": AcceptedIntegrity,
            "QgsVectorLayer": vector_layer,
            "write_source_snapshot": DifferenceFilter.write_source_snapshot,
            "LAYER_NAMES": types.SimpleNamespace(ACCEPTED="accepted_labels"),
        }
    )
    ctx = {
        "run_id": "run-test",
        "run_dir": str(tmp_path),
        "range_selection": {},
        "skip_accepted": True,
        "accepted_validation": {
            "status": "passed",
            "overlap_tolerance": 0.25,
            "identity": "start-time",
        },
        # Tile 0 was covered at start, but became uncovered while the real
        # Tile QProcess probe ran. Tile 1 changed in the opposite direction.
        "skipped_tiles": [
            {"row": 0, "col": 0, "bounds": "covered_at_start_time"}
        ],
        "active_tiles": [
            {"row": 0, "col": 0, "bounds": "covered_at_start_time"},
            {"row": 0, "col": 1, "bounds": "covered_at_snapshot_time"},
        ],
    }

    task = types.SimpleNamespace(
        request=ctx, range_source=None, accepted_source=live_layer,
        accepted_wkb_type="polygon", raster_crs="EPSG:3857", transform_context=None,
        isCanceled=lambda: False, setProgress=lambda _value: None,
    )
    assert function(task) is True, getattr(task, "error_message", "")
    ctx = task.result_data

    assert snapshot_calls == [tmp_path / "accepted_snapshot.gpkg"]
    assert tile_checks == ["covered_at_start_time", "covered_at_snapshot_time"]
    assert ctx["accepted_snapshot"] == str(tmp_path / "accepted_snapshot.gpkg")
    assert ctx["accepted_validation"]["identity"] == "snapshot-time"
    assert ctx["accepted_validation"]["source"] == "run_snapshot"
    assert [(tile["row"], tile["col"]) for tile in ctx["skipped_tiles"]] == [
        (0, 1)
    ]
    assert ctx["skipped_tiles"][0]["skip_reason"] == "fully_accepted"


def test_workflow_freezes_start_request_and_uses_prepared_inputs():
    dock = (
        ROOT / "src" / "labeling_tool" / "main" / "main_dock.py"
    ).read_text(encoding="utf-8")
    workflow = (
        ROOT / "src" / "labeling_tool" / "runs" / "run_workflow.py"
    ).read_text(encoding="utf-8")
    planning = (
        ROOT / "src" / "labeling_tool" / "runs" / "run_planning.py"
    ).read_text(encoding="utf-8")
    start_block = dock.split("def _on_start", 1)[1].split(
        "def _on_workflow_state_changed", 1
    )[0]
    preparation = workflow.split("def _start_preparation", 1)[1].split(
        "def _matches_preparation", 1
    )[0]
    completed = workflow.split("def _on_preparation_completed", 1)[1].split(
        "def _on_preparation_terminated", 1
    )[0]

    assert "RunStartRequest(" in start_block
    assert "scripts_dir=scripts_dir" in start_block
    assert "active_tiles=tuple(current_tiles)" in start_block
    assert "skip_accepted=bool(self.skip_accepted_check.isChecked())" in start_block
    assert "request.get_valid_range_layer()" in preparation
    assert "RunPreparationTask(" in preparation
    assert "attempt.prepared = dict(task.result_data)" in completed
    assert "active_tiles=tuple(prepared[\"active_tiles\"])" in workflow
    assert "skip_accepted=request.skip_accepted" in workflow
    assert "accepted_layer=attempt.request.accepted_layer" in workflow
    assert "selected_tile_keys" in planning


def test_probe_error_report_is_machine_readable(tmp_path, capsys):
    from loess_runtime.inference.tile_cache_probe import main

    exit_code = main(
        [
            "--raster",
            str(tmp_path / "missing.tif"),
            "--output-root",
            str(tmp_path),
            "--tile-json",
            json.dumps(_request()),
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert report["kind"] == "tile_cache_probe"
    assert report["status"] == "error"


def test_runner_rejects_wrong_workspace_bounds_and_window(monkeypatch, tmp_path):
    module = _load_probe_runner(monkeypatch)
    expected = {
        "probe_token": "b" * 32,
        "measurement_workspace": str(tmp_path),
        "sample_artifact_directory": str(
            tmp_path / f".loess-tile-cache-probe-{'b' * 32}"
        ),
        "sample_source_path": str(tmp_path / "source.tif"),
        "sample_tile_id": "0_1",
        "sample_row": 0,
        "sample_col": 1,
        "sample_bounds": _request()["bounds"],
    }

    wrong_workspace = _report(expected)
    wrong_workspace["measurement_workspace"] = str(tmp_path / "other")
    with pytest.raises(ValueError, match="measurement_workspace"):
        module.TileCacheProbeRunner._validate_report(wrong_workspace, expected)

    wrong_bounds = _report(expected)
    wrong_bounds["sample_bounds"] = {**_request()["bounds"], "xmax": 1025}
    with pytest.raises(ValueError, match="xmax"):
        module.TileCacheProbeRunner._validate_report(wrong_bounds, expected)

    wrong_window = _report(expected)
    wrong_window["sample_source_window"]["x1"] = 1023
    with pytest.raises(ValueError, match="512x512"):
        module.TileCacheProbeRunner._validate_report(wrong_window, expected)


def test_runner_finished_then_error_emits_one_terminal_callback(
    monkeypatch, tmp_path
):
    module = _load_probe_runner(monkeypatch)
    runner = _runner(module, tmp_path)
    token = "c" * 32
    probe_dir = tmp_path / f".loess-tile-cache-probe-{token}"
    probe_dir.mkdir()
    expected = {
        "probe_token": token,
        "measurement_workspace": str(tmp_path),
        "sample_artifact_directory": str(probe_dir),
        "sample_source_path": str(tmp_path / "source.tif"),
        "sample_tile_id": "0_1",
        "sample_row": 0,
        "sample_col": 1,
        "sample_bounds": _request()["bounds"],
    }
    runner._expected = expected
    runner._probe_dir = str(probe_dir)
    process = _FinishedProcess(
        stdout=(json.dumps(_report(expected)) + "\n").encode("utf-8")
    )
    runner._process = process
    monkeypatch.setattr(module, "process_is_running", lambda _process: False)

    runner._on_finished(process, 1, 0, None)
    runner._on_process_error(process, 1, None)

    assert len(runner.succeeded.values) == 1
    assert runner.failed.values == []
    assert process.deleted == 1
    assert not probe_dir.exists()


def test_runner_error_then_finished_emits_one_terminal_callback(
    monkeypatch, tmp_path
):
    module = _load_probe_runner(monkeypatch)
    runner = _runner(module, tmp_path)
    token = "d" * 32
    probe_dir = tmp_path / f".loess-tile-cache-probe-{token}"
    probe_dir.mkdir()
    runner._expected = {
        "probe_token": token,
        "measurement_workspace": str(tmp_path),
    }
    runner._probe_dir = str(probe_dir)
    process = _FinishedProcess()
    runner._process = process
    monkeypatch.setattr(module, "process_is_running", lambda _process: False)

    runner._on_process_error(process, 1, None)
    runner._on_finished(process, 1, 1, None)

    assert runner.succeeded.values == []
    assert runner.failed.values == [("failed to start",)]
    assert process.deleted == 1
    assert not probe_dir.exists()


def test_runner_cancel_is_idempotent_and_removes_only_its_probe_directory(
    monkeypatch, tmp_path
):
    module = _load_probe_runner(monkeypatch)
    runner = _runner(module, tmp_path)
    token = "e" * 32
    probe_dir = tmp_path / f".loess-tile-cache-probe-{token}"
    other_dir = tmp_path / f".loess-tile-cache-probe-{'f' * 32}"
    probe_dir.mkdir()
    other_dir.mkdir()
    runner._expected = {
        "probe_token": token,
        "measurement_workspace": str(tmp_path),
    }
    runner._probe_dir = str(probe_dir)
    process = _FinishedProcess()
    runner._process = process
    monkeypatch.setattr(module, "process_is_running", lambda _process: False)

    runner.cancel()
    runner.cancel()

    assert not probe_dir.exists()
    assert other_dir.is_dir()
    assert process.blocked == 1
    assert process.deleted == 1
    assert runner.succeeded.values == []
    assert runner.failed.values == []


def test_runner_rejects_repeated_start_before_touching_qprocess(
    monkeypatch, tmp_path
):
    module = _load_probe_runner(monkeypatch)
    runner = _runner(module, tmp_path)
    runner._process = object()

    with pytest.raises(RuntimeError, match="already running"):
        runner.start(
            raster_path=tmp_path / "source.tif",
            output_root=tmp_path,
            tile=_request(),
        )
