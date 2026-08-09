import json
import importlib
import ast
import logging
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from tile_cache_probe import measure_tile_cache
from tile_materializer import (
    TILE_MATERIALIZATION_METHOD_VERSION,
    _materialize_one,
)
from labeling_tool.core.run_spec import (
    RESERVATION_FILE,
    reserve_run_directory,
    run_tile_cache_dir,
)


ROOT = Path(__file__).resolve().parents[1]


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
    sys.modules.pop("labeling_tool.core.tile_cache_probe_runner", None)
    return importlib.import_module("labeling_tool.core.tile_cache_probe_runner")


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
        "measurement_method": "tile_materializer._materialize_one",
        "measurement_method_version": TILE_MATERIALIZATION_METHOD_VERSION,
    }


def _discard_pending_reservation_function():
    tree = ast.parse(
        (ROOT / "qgis_plugins" / "labeling_tool" / "gui" / "main_dock.py").read_text(
            encoding="utf-8"
        )
    )
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LabelingDockWidget"
    )
    function_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_discard_pending_run_reservation"
    )
    module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Path": Path,
        "RESERVATION_FILE": RESERVATION_FILE,
        "json": json,
        "logger": logging.getLogger("test.tile_cache_probe"),
        "run_tile_cache_dir": run_tile_cache_dir,
        "shutil": __import__("shutil"),
    }
    exec(compile(module, "main_dock.py", "exec"), namespace)
    return namespace["_discard_pending_run_reservation"]


def _freeze_accepted_function(namespace):
    tree = ast.parse(
        (ROOT / "qgis_plugins" / "labeling_tool" / "gui" / "main_dock.py").read_text(
            encoding="utf-8"
        )
    )
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LabelingDockWidget"
    )
    function_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_freeze_pending_accepted_snapshot"
    )
    module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(module)
    values = {"Path": Path, **namespace}
    exec(compile(module, "main_dock.py", "exec"), values)
    return values["_freeze_pending_accepted_snapshot"]


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

    direct = _materialize_one(source, tmp_path / "direct", _request())
    assert report["status"] == "passed"
    assert report["measurement_method"] == "tile_materializer._materialize_one"
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
    source = (ROOT / "inference_scripts" / "run_tile_cache_probe.sh").read_text(
        encoding="utf-8"
    )
    assert 'source "$SCRIPT_DIR/config.sh"' in source
    assert '"$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV"' in source
    assert 'python "$SCRIPT_DIR/tile_cache_probe.py" "$@"' in source


def test_main_dock_blocks_on_real_probe_and_freezes_measurement():
    source = (
        ROOT / "qgis_plugins" / "labeling_tool" / "gui" / "main_dock.py"
    ).read_text(encoding="utf-8")
    probe_block = source.split("def _on_tiles_extracted", 1)[1].split(
        "def _start_inference_after_tile_cache_probe", 1
    )[0]
    preflight_block = source.split(
        "def _start_inference_after_tile_cache_probe", 1
    )[1].split("stride = 512", 1)[0]

    assert "active_tiles = sorted(" in probe_block
    assert 'key=lambda item: (int(item["row"]), int(item["col"]))' in probe_block
    assert "TileCacheProbeRunner(" in probe_block
    assert "probe.succeeded.connect" in probe_block
    assert "probe.failed.connect" in probe_block
    assert 'tile_cache_sample.get("materialized_cache_bytes")' in preflight_block
    assert 'storage["input_tile_sample"] = tile_cache_sample' in preflight_block
    assert "source_bytes * pixel_count / raster_pixels" not in source
    assert "_finish_before_inference(\"Tile 存储预检失败\"" in source
    start_block = source.split("def _on_start", 1)[1].split(
        "def _on_tile_extraction_progress", 1
    )[0]
    assert "reserve_run_directory" not in start_block
    assert "snapshot_accepted_layer" not in start_block
    assert "tile_is_fully_accepted" not in start_block
    inference_block = source.split(
        "def _start_inference_after_tile_cache_probe", 1
    )[1].split("def _on_runner_stage_progress", 1)[0]
    assert "storage_preflight(" in inference_block
    assert inference_block.index("reserve_run_directory(") < inference_block.index(
        "storage_preflight("
    )
    assert inference_block.index("_freeze_pending_accepted_snapshot(") < inference_block.index(
        "storage_preflight("
    )
    assert "_discard_pending_run_reservation()" in source


def test_pre_run_failure_removes_only_unused_marker_backed_run(tmp_path):
    output_root = tmp_path / "output"
    run_id, run_dir = reserve_run_directory(output_root)
    cache_root = run_tile_cache_dir(output_root, run_id).parent
    (run_dir / "accepted_snapshot.gpkg").write_bytes(b"snapshot")
    dock = types.SimpleNamespace(
        _pending_run={
            "output_dir": str(output_root),
            "run_id": run_id,
            "run_dir": str(run_dir),
            "accepted_snapshot": str(run_dir / "accepted_snapshot.gpkg"),
        }
    )

    _discard_pending_reservation_function()(dock)

    assert not run_dir.exists()
    assert not cache_root.exists()
    assert dock._pending_run["run_id"] == ""
    assert dock._pending_run["accepted_snapshot"] == ""


def test_pre_run_cleanup_never_removes_a_run_spec_or_foreign_reservation(tmp_path):
    output_root = tmp_path / "output"
    cleanup = _discard_pending_reservation_function()

    committed_id, committed_dir = reserve_run_directory(output_root)
    (committed_dir / "run_spec.json").write_text("{}", encoding="utf-8")
    committed = types.SimpleNamespace(
        _pending_run={
            "output_dir": str(output_root),
            "run_id": committed_id,
            "run_dir": str(committed_dir),
        }
    )
    cleanup(committed)
    assert committed_dir.is_dir()
    assert run_tile_cache_dir(output_root, committed_id).parent.is_dir()

    foreign_id, foreign_dir = reserve_run_directory(output_root)
    (foreign_dir / RESERVATION_FILE).write_text(
        json.dumps({"run_id": "another_run"}), encoding="utf-8"
    )
    foreign = types.SimpleNamespace(
        _pending_run={
            "output_dir": str(output_root),
            "run_id": foreign_id,
            "run_dir": str(foreign_dir),
        }
    )
    cleanup(foreign)
    assert foreign_dir.is_dir()
    assert run_tile_cache_dir(output_root, foreign_id).parent.is_dir()


def test_snapshot_time_accepted_identity_replaces_start_time_audit_and_skips(
    tmp_path,
):
    live_layer = object()
    frozen_layer = object()
    snapshot_calls = []
    tile_checks = []

    class DifferenceFilter:
        @staticmethod
        def snapshot_accepted_layer(layer, output_path):
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
        def audit_accepted_layer(layer, *, overlap_tolerance, expected_crs):
            assert layer is frozen_layer
            assert overlap_tolerance == pytest.approx(0.25)
            assert expected_crs == "EPSG:3857"
            return {
                "status": "passed",
                "feature_count": 2,
                "overlap_tolerance": overlap_tolerance,
                "identity": "snapshot-time",
            }

    class Raster:
        @staticmethod
        def crs():
            return "EPSG:3857"

    def vector_layer(uri, name, provider):
        assert uri.endswith("accepted_snapshot.gpkg|layername=accepted_labels")
        assert name == "run-test accepted snapshot"
        assert provider == "ogr"
        return frozen_layer

    function = _freeze_accepted_function(
        {
            "difference_filter": DifferenceFilter,
            "accepted_integrity": AcceptedIntegrity,
            "QgsVectorLayer": vector_layer,
            "LAYER_NAMES": types.SimpleNamespace(ACCEPTED="accepted_labels"),
        }
    )
    ctx = {
        "run_id": "run-test",
        "raster": Raster(),
        "accepted_layer": live_layer,
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

    function(types.SimpleNamespace(), ctx, tmp_path)

    assert snapshot_calls == [tmp_path / "accepted_snapshot.gpkg"]
    assert tile_checks == ["covered_at_start_time", "covered_at_snapshot_time"]
    assert ctx["accepted_layer"] is frozen_layer
    assert ctx["accepted_validation"]["identity"] == "snapshot-time"
    assert ctx["accepted_validation"]["source"] == "run_snapshot"
    assert [(tile["row"], tile["col"]) for tile in ctx["skipped_tiles"]] == [
        (0, 1)
    ]
    assert ctx["skipped_tiles"][0]["skip_reason"] == "fully_accepted"


def test_probe_and_run_use_start_time_frozen_paths_tiles_and_skip_setting():
    source = (
        ROOT / "qgis_plugins" / "labeling_tool" / "gui" / "main_dock.py"
    ).read_text(encoding="utf-8")
    start_block = source.split("def _on_start", 1)[1].split(
        "def _on_tile_extraction_progress", 1
    )[0]
    probe_block = source.split("def _on_tiles_extracted", 1)[1].split(
        "def _start_inference_after_tile_cache_probe", 1
    )[0]
    inference_block = source.split(
        "def _start_inference_after_tile_cache_probe", 1
    )[1].split("def _on_runner_stage_progress", 1)[0]

    assert '"scripts_dir": scripts_dir' in start_block
    assert '"active_tiles": list(self._current_tiles)' in start_block
    assert '"skip_accepted": bool(self.skip_accepted_check.isChecked())' in start_block
    assert 'ctx.get("active_tiles") or []' in probe_block
    assert 'ctx["scripts_dir"]' in probe_block
    assert "self._freeze_pending_accepted_snapshot(ctx, run_dir)" in inference_block
    assert 'for tile in ctx.get("active_tiles") or []' in inference_block
    assert 'skip_accepted=bool(ctx.get("skip_accepted", False))' in inference_block
    assert 'accepted_layer=ctx["accepted_layer"]' in inference_block


def test_probe_error_report_is_machine_readable(tmp_path, capsys):
    from tile_cache_probe import main

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
