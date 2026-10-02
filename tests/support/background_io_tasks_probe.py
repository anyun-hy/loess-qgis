# ruff: noqa: E402
"""Native QGIS probes for background Run loading and accepted publication."""

from __future__ import annotations

import fcntl
import os
import sqlite3
import sys
import tempfile
import threading
import time
import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
try:
    from qgis.core import (
        QgsApplication,
        QgsCoordinateTransformContext,
        QgsFeature,
        QgsGeometry,
        QgsRectangle,
        QgsVariantUtils,
        QgsVectorFileWriter,
        QgsVectorLayer,
    )
    from qgis.gui import QgsMapCanvas
    from qgis.PyQt.QtCore import QCoreApplication, QEvent, QThread, QTimer
    from qgis.PyQt.QtWidgets import QPushButton
except ModuleNotFoundError:
    raise SystemExit(77)

from labeling_tool.main import main_dock
from labeling_tool.qgis_support.layer_names import LAYER_NAMES
from labeling_tool.qgis_support.qgis_writer import write_vector_layer
from labeling_tool.refinement import accepted_writer, class_workspace, manual_run_loader
from labeling_tool.refinement import class_refinement_dialog as ui
from labeling_tool.refinement.background_io_tasks import (
    AcceptedWriteTask,
    ManualRunLoadTask,
)
from labeling_tool.refinement.final_assembler import FINAL_FIELDS
from labeling_tool.refinement.refinement_task import (
    file_identity as workspace_file_identity,
)
from labeling_tool.shared.contracts.run_spec import (
    CLASS_NAMES,
    CLASS_ORDER,
    atomic_write_json,
    sha256_file,
)


def drain(app):
    app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def wait_for(app, predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for native background task")
        drain(app)
        time.sleep(0.005)
    drain(app)


def copied_run(root: Path, *, class_bytes=0) -> Path:
    run_id = "20260925_120000_manual"
    run_root = root / run_id
    old_root = root / "remote" / run_id
    fusion_dir = run_root / "fusion" / "fixture"
    fusion_dir.mkdir(parents=True)
    semantic = fusion_dir / "semantic_polygons.gpkg"
    semantic.write_bytes(b"semantic")
    snapshot = run_root / "accepted_snapshot.gpkg"
    snapshot.write_bytes(b"accepted")
    spec = {
        "schema_version": 2,
        "run_id": run_id,
        "run_dir": str(old_root),
        "raster": {"transform": [1, 0, 0, 0, -1, 0], "crs": "EPSG:4490"},
        "accepted_gpkg": str(old_root / snapshot.name),
        "fusion": {"profile_id": "fixture"},
    }
    spec_path = run_root / "run_spec.json"
    atomic_write_json(spec_path, spec)
    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "run_spec": str(old_root / "run_spec.json"),
        "run_spec_sha256": sha256_file(spec_path),
        "status": "ready",
        "streams": [
            {
                "stream_id": "fusion:fixture",
                "kind": "fusion",
                "fusion_profile_id": "fixture",
                "status": "ready",
                "paths": {
                    "semantic_polygons": str(
                        old_root / "fusion" / "fixture" / semantic.name
                    )
                },
            }
        ],
    }
    atomic_write_json(run_root / "run_manifest.json", manifest)
    classes = run_root / "classes"
    classes.mkdir()
    records = {}
    payload = (b"background-io-fixture" * (class_bytes // 21 + 1))[:class_bytes]
    for code in CLASS_ORDER:
        path = classes / f"class_{code}.gpkg"
        path.write_bytes(payload or f"class-{code}".encode())
        records[str(code)] = {
            "class_code": code,
            "path": f"/remote/{path.name}",
            "layer_name": "class_polygons",
            "feature_count": 0,
            "confirmed": False,
        }
    atomic_write_json(
        classes / "workspace.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "baseline_stream_id": "fusion:fixture",
            "classes": records,
        },
    )
    return run_root


def loader_responsive(app, root):
    run_root = copied_run(root, class_bytes=2 * 1024 * 1024)
    original = manual_run_loader._sha256
    entered = threading.Event()
    release = threading.Event()
    worker_threads = []

    def gated(path, *, is_canceled=None):
        result = original(path, is_canceled=is_canceled)
        if path.name == "class_12.gpkg" and not entered.is_set():
            worker_threads.append(QThread.currentThread() == app.thread())
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test barrier expired")
        return result

    task = ManualRunLoadTask(1, run_root)
    terminal = []
    task.taskCompleted.connect(lambda: terminal.append("completed"))
    task.taskTerminated.connect(lambda: terminal.append("terminated"))
    ticks = []
    timer = QTimer()
    timer.setInterval(10)

    def tick():
        ticks.append(1)
        if entered.is_set() and len(ticks) >= 5:
            release.set()

    timer.timeout.connect(tick)
    timer.start()
    try:
        with patch.object(manual_run_loader, "_sha256", side_effect=gated):
            QgsApplication.taskManager().addTask(task)
            wait_for(app, lambda: bool(terminal), timeout=12)
        assert terminal == ["completed"] and task.published
        assert worker_threads == [False]
        assert len(ticks) >= 5
        assert (run_root / "classes" / "accepted_write_run_manifest.json").is_file()
    finally:
        timer.stop()
        release.set()


def loader_cancel(app, root):
    run_root = copied_run(root, class_bytes=1024 * 1024)
    original = manual_run_loader._sha256
    entered = threading.Event()

    def gated(path, *, is_canceled=None):
        if path.name == "class_12.gpkg":
            entered.set()
            while is_canceled is not None and not is_canceled():
                time.sleep(0.01)
        return original(path, is_canceled=is_canceled)

    task = ManualRunLoadTask(2, run_root)
    terminal = []
    task.taskCompleted.connect(lambda: terminal.append("completed"))
    task.taskTerminated.connect(lambda: terminal.append("terminated"))
    with patch.object(manual_run_loader, "_sha256", side_effect=gated):
        QgsApplication.taskManager().addTask(task)
        wait_for(app, entered.is_set)
        assert task.cancel() is True
        wait_for(app, lambda: bool(terminal))
    assert terminal == ["terminated"] and not task.published
    assert not (run_root / "classes" / "accepted_write_run_spec.json").exists()
    assert not (run_root / "classes" / "accepted_write_run_manifest.json").exists()


def commit_cancel_boundary(app, root):
    # Cancellation wins while ManualRunLoadTask is still outside the short
    # lifecycle lock: publication is never called.
    manual = ManualRunLoadTask(1, root / "manual")
    entered = threading.Event()
    release = threading.Event()
    published = []

    def unchanged(_bundle):
        entered.set()
        if not release.wait(5):
            raise RuntimeError("manual transition barrier expired")
        return True

    with (
        patch.object(manual_run_loader, "prepare_manual_run", return_value={}),
        patch.object(
            manual_run_loader, "manual_run_inputs_unchanged", side_effect=unchanged
        ),
        patch.object(
            manual_run_loader,
            "publish_manual_run_bundle",
            side_effect=lambda bundle: published.append(bundle) or bundle,
        ),
    ):
        terminal = []
        manual.taskTerminated.connect(lambda: terminal.append("terminated"))
        QgsApplication.taskManager().addTask(manual)
        wait_for(app, entered.is_set)
        assert manual.cancel() is True
        release.set()
        wait_for(app, lambda: bool(terminal))
    assert not manual.commit_started and not manual.published and published == []

    # Once the same transition has acquired the lock and marked commit_started,
    # cancel is explicitly refused and publication completes.
    committing = ManualRunLoadTask(2, root / "committing")
    commit_entered = threading.Event()
    commit_release = threading.Event()

    def publish(bundle):
        commit_entered.set()
        if not commit_release.wait(5):
            raise RuntimeError("manual commit barrier expired")
        return bundle

    with (
        patch.object(manual_run_loader, "prepare_manual_run", return_value={}),
        patch.object(
            manual_run_loader, "manual_run_inputs_unchanged", return_value=True
        ),
        patch.object(
            manual_run_loader, "publish_manual_run_bundle", side_effect=publish
        ),
    ):
        terminal = []
        committing.taskCompleted.connect(lambda: terminal.append("completed"))
        QgsApplication.taskManager().addTask(committing)
        wait_for(app, commit_entered.is_set)
        assert committing.commit_started and committing.cancel() is False
        commit_release.set()
        wait_for(app, lambda: bool(terminal))
    assert committing.published and terminal == ["completed"]

    # AcceptedWriteTask has the same two outcomes around its before_commit hook.
    accepted = AcceptedWriteTask(
        1,
        run_id="race",
        final_path=root / "final.gpkg",
        accepted_path=root / "accepted.gpkg",
        run_manifest_path=root / "manifest.json",
        workspace_input_identities={},
        transform_context=QgsCoordinateTransformContext(),
    )
    entered = threading.Event()
    release = threading.Event()
    checks = 0

    def accepted_inputs(*, transaction_started=False):
        nonlocal checks
        checks += 1
        if checks == 2:
            entered.set()
            if not release.wait(5):
                raise RuntimeError("accepted transition barrier expired")
        return True

    def accepted_append(*_args, before_commit, **_kwargs):
        before_commit()
        return 1

    accepted.inputs_unchanged = accepted_inputs
    with patch.object(
        accepted_writer,
        "append_final_to_accepted",
        side_effect=accepted_append,
    ):
        terminal = []
        accepted.taskTerminated.connect(lambda: terminal.append("terminated"))
        QgsApplication.taskManager().addTask(accepted)
        wait_for(app, entered.is_set)
        assert accepted.cancel() is True
        release.set()
        wait_for(app, lambda: bool(terminal))
    assert not accepted.commit_started and not accepted.published

    accepted_commit = AcceptedWriteTask(
        2,
        run_id="race",
        final_path=root / "final.gpkg",
        accepted_path=root / "accepted.gpkg",
        run_manifest_path=root / "manifest.json",
        workspace_input_identities={},
        transform_context=QgsCoordinateTransformContext(),
    )
    accepted_commit.inputs_unchanged = lambda **_kwargs: True
    commit_entered = threading.Event()
    commit_release = threading.Event()

    def accepted_commit_append(*_args, before_commit, **_kwargs):
        before_commit()
        commit_entered.set()
        if not commit_release.wait(5):
            raise RuntimeError("accepted commit barrier expired")
        return 1

    with patch.object(
        accepted_writer,
        "append_final_to_accepted",
        side_effect=accepted_commit_append,
    ):
        terminal = []
        accepted_commit.taskCompleted.connect(lambda: terminal.append("completed"))
        QgsApplication.taskManager().addTask(accepted_commit)
        wait_for(app, commit_entered.is_set)
        assert accepted_commit.commit_started and accepted_commit.cancel() is False
        commit_release.set()
        wait_for(app, lambda: bool(terminal))
    assert accepted_commit.published and terminal == ["completed"]


def memory_layer(fields, name):
    layer = QgsVectorLayer("MultiPolygon?crs=EPSG:4490", name, "memory")
    layer.dataProvider().addAttributes(fields)
    layer.updateFields()
    return layer


def add_feature(layer, bounds, *, run_id, object_id):
    feature = QgsFeature(layer.fields())
    geometry = QgsGeometry.fromRect(QgsRectangle(*bounds))
    geometry.convertToMultiType()
    feature.setGeometry(geometry)
    values = {
        "run_id": run_id,
        "object_id": object_id,
        "part_id": "000",
        "class_code": 12,
        "class_name": CLASS_NAMES[12],
        "confidence_mean": 0.9,
        "confidence_std": 0.01,
        "baseline_stream_id": "fusion:fixture",
        "source_stream_id": "fusion:fixture",
        "source": "class_working",
        "geometry_source": "fusion",
        "geometry_revision": 0,
        "edit_base": "",
        "sam_session_id": "",
        "sam_score": None,
        "model_version": "fixture",
        "fusion_profile_id": "fixture",
        "sam_version": "",
        "reviewed": 1,
        "created_at": "2026-09-25T00:00:00+09:00",
        "updated_at": "2026-09-25T00:00:00+09:00",
    }
    feature.setAttributes([values.get(field.name(), "") for field in layer.fields()])
    assert layer.dataProvider().addFeature(feature)


def write_layer(layer, path, layer_name):
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.layerName = layer_name
    options.actionOnExistingFile = (
        QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteFile
    )
    error, message = write_vector_layer(
        layer,
        path,
        options,
        transform_context=QgsCoordinateTransformContext(),
    )
    assert error == QgsVectorFileWriter.WriterError.NoError, message


def accepted_fixture(root: Path, *, existing=False):
    run_id = "accepted_run"
    final = memory_layer(FINAL_FIELDS, "final")
    add_feature(final, (10, 10, 11, 11), run_id=run_id, object_id="new")
    final_path = root / "final.gpkg"
    write_layer(final, final_path, LAYER_NAMES.FINAL_COMPOSITE)
    target = root / "accepted.gpkg"
    if existing:
        accepted = memory_layer(accepted_writer.ACCEPTED_FIELDS_QGS, "accepted")
        add_feature(accepted, (0, 0, 1, 1), run_id="old_run", object_id="old")
        write_layer(accepted, target, LAYER_NAMES.ACCEPTED)
    spec = {
        "schema_version": 2,
        "run_id": run_id,
        "run_dir": str(root),
        "raster": {"transform": [1, 0, 0, 0, -1, 20], "crs": "EPSG:4490"},
        "accepted_gpkg": str(root / "accepted_snapshot.gpkg"),
        "accepted_target_gpkg": str(target),
    }
    spec_path = root / "run_spec.json"
    atomic_write_json(spec_path, spec)
    manifest_path = root / "run_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 2,
            "run_id": run_id,
            "run_spec": str(spec_path),
            "run_spec_sha256": sha256_file(spec_path),
            "status": "ready",
            "streams": [
                {"stream_id": "fusion:fixture", "kind": "fusion", "status": "ready"}
            ],
        },
    )
    return run_id, final_path, target, manifest_path


def run_accepted_task(app, task):
    terminal = []
    task.taskCompleted.connect(lambda: terminal.append("completed"))
    task.taskTerminated.connect(lambda: terminal.append("terminated"))
    QgsApplication.taskManager().addTask(task)
    wait_for(app, lambda: bool(terminal))
    return terminal


def accepted_success(app, root):
    run_id, final_path, target, manifest_path = accepted_fixture(root, existing=True)
    inode = target.stat().st_ino
    reader = QgsVectorLayer(
        f"{target}|layername={LAYER_NAMES.ACCEPTED}", "open_reader", "ogr"
    )
    assert reader.isValid() and reader.featureCount() == 1
    workspace_path = root / "class_12.gpkg"
    workspace_path.write_bytes(b"workspace")
    task = AcceptedWriteTask(
        1,
        run_id=run_id,
        final_path=final_path,
        accepted_path=target,
        run_manifest_path=manifest_path,
        workspace_input_identities={
            str(workspace_path): workspace_file_identity(workspace_path)
        },
        transform_context=QgsCoordinateTransformContext(),
    )
    assert run_accepted_task(app, task) == ["completed"]
    assert task.published and task.commit_started
    assert task.result_data["feature_count"] == 1
    reader.reload()
    assert reader.featureCount() == 2
    assert {feature["object_id"] for feature in reader.getFeatures()} == {"old", "new"}
    written = next(
        feature for feature in reader.getFeatures() if feature["object_id"] == "new"
    )
    assert QgsVariantUtils.isNull(written["sam_score"])
    assert target.stat().st_ino == inode


def accepted_cancel_wait(app, root):
    run_id, final_path, target, manifest_path = accepted_fixture(root, existing=True)
    before = sha256_file(target)
    lock_path = target.with_name(target.name + ".write.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    task = AcceptedWriteTask(
        1,
        run_id=run_id,
        final_path=final_path,
        accepted_path=target,
        run_manifest_path=manifest_path,
        workspace_input_identities={},
        transform_context=QgsCoordinateTransformContext(),
    )
    terminal = []
    task.taskCompleted.connect(lambda: terminal.append("completed"))
    task.taskTerminated.connect(lambda: terminal.append("terminated"))
    try:
        QgsApplication.taskManager().addTask(task)
        wait_for(app, lambda: "写入锁" in task.progress_message)
        assert task.cancel() is True
        wait_for(app, lambda: bool(terminal))
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    assert terminal == ["terminated"] and not task.commit_started
    assert sha256_file(target) == before and lock_path.is_file()


def accepted_failure_rollback(app, root):
    run_id, final_path, target, manifest_path = accepted_fixture(root, existing=True)
    inode = target.stat().st_ino
    task = AcceptedWriteTask(
        1,
        run_id=run_id,
        final_path=final_path,
        accepted_path=target,
        run_manifest_path=manifest_path,
        workspace_input_identities={},
        transform_context=QgsCoordinateTransformContext(),
    )
    original_insert = accepted_writer._insert_ogr_feature

    def insert_then_fail(dataset, layer, pending):
        original_insert(layer, pending[0])
        raise RuntimeError("injected transaction failure")

    with patch.object(
        accepted_writer, "_commit_existing_target", side_effect=insert_then_fail
    ):
        assert run_accepted_task(app, task) == ["terminated"]
    assert task.commit_started and "injected transaction failure" in task.error_message
    layer = QgsVectorLayer(
        f"{target}|layername={LAYER_NAMES.ACCEPTED}", "unchanged", "ogr"
    )
    assert layer.isValid() and layer.featureCount() == 1
    assert {feature["object_id"] for feature in layer.getFeatures()} == {"old"}
    assert target.stat().st_ino == inode


def accepted_commit_failure_rollback(app, root):
    run_id, final_path, target, manifest_path = accepted_fixture(root, existing=True)
    inode = target.stat().st_ino
    task = AcceptedWriteTask(
        1,
        run_id=run_id,
        final_path=final_path,
        accepted_path=target,
        run_manifest_path=manifest_path,
        workspace_input_identities={},
        transform_context=QgsCoordinateTransformContext(),
    )
    with patch.object(
        accepted_writer,
        "_commit_existing_target",
        side_effect=RuntimeError("cannot commit accepted_labels transaction"),
    ):
        assert run_accepted_task(app, task) == ["terminated"]
    assert task.commit_started and "cannot commit" in task.error_message
    layer = QgsVectorLayer(
        f"{target}|layername={LAYER_NAMES.ACCEPTED}", "unchanged", "ogr"
    )
    assert layer.isValid() and layer.featureCount() == 1
    assert target.stat().st_ino == inode


def accepted_external_writer_blocked(app, root):
    _run_id, final_path, target, manifest_path = accepted_fixture(root, existing=True)
    original_audit = accepted_writer.accepted_integrity.audit_accepted_layer
    blocked = []

    def audit_with_competing_writer(*args, **kwargs):
        connection = sqlite3.connect(target, timeout=0)
        try:
            try:
                connection.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as error:
                blocked.append("locked" in str(error).lower())
            else:
                connection.rollback()
                blocked.append(False)
        finally:
            connection.close()
        return original_audit(*args, **kwargs)

    with patch.object(
        accepted_writer.accepted_integrity,
        "audit_accepted_layer",
        side_effect=audit_with_competing_writer,
    ):
        count = accepted_writer.append_final_to_accepted(
            final_path,
            target,
            manifest_path,
            transform_context=QgsCoordinateTransformContext(),
        )
    assert count == 1 and blocked == [True]


def accepted_new_target_no_overwrite(app, root):
    _run_id, final_path, target, manifest_path = accepted_fixture(root)

    def competing_create():
        target.write_bytes(b"competing-writer")

    try:
        accepted_writer.append_final_to_accepted(
            final_path,
            target,
            manifest_path,
            before_commit=competing_create,
            transform_context=QgsCoordinateTransformContext(),
        )
    except RuntimeError as error:
        assert "其他任务创建" in str(error)
    else:
        raise AssertionError("a newly appeared accepted target was overwritten")
    assert target.read_bytes() == b"competing-writer"


def _accepted_object_ids(target):
    layer = QgsVectorLayer(
        f"{target}|layername={LAYER_NAMES.ACCEPTED}", "accepted_check", "ogr"
    )
    assert layer.isValid()
    return {feature["object_id"] for feature in layer.getFeatures()}


def accepted_spec_changed_during_verification(app, root):
    _run_id, final_path, target, manifest_path = accepted_fixture(root, existing=True)
    inode = target.stat().st_ino
    before_ids = _accepted_object_ids(target)
    spec_path = root / "run_spec.json"
    original_sha256 = accepted_writer._sha256
    changed = []

    def hash_then_change(path, *, is_canceled=None):
        digest = original_sha256(path, is_canceled=is_canceled)
        if Path(path).resolve() == spec_path.resolve() and not changed:
            spec_path.write_text(
                spec_path.read_text(encoding="utf-8") + "\n ",
                encoding="utf-8",
            )
            changed.append(True)
        return digest

    with patch.object(accepted_writer, "_sha256", side_effect=hash_then_change):
        try:
            accepted_writer.append_final_to_accepted(
                final_path,
                target,
                manifest_path,
                transform_context=QgsCoordinateTransformContext(),
            )
        except ValueError as error:
            assert "发生变化" in str(error)
        else:
            raise AssertionError(
                "a Run spec changed after SHA verification was accepted"
            )
    assert changed == [True]
    assert target.stat().st_ino == inode
    assert _accepted_object_ids(target) == before_ids


def accepted_missing_ogr_binding(app, root):
    _run_id, final_path, target, manifest_path = accepted_fixture(root, existing=True)
    inode = target.stat().st_ino
    before_ids = _accepted_object_ids(target)
    message = "当前 QGIS Python 缺少 osgeo/OGR 事务绑定，不能写入现有 accepted_labels"
    with patch.object(
        accepted_writer,
        "_ogr_modules",
        side_effect=RuntimeError(message),
    ):
        try:
            accepted_writer.append_final_to_accepted(
                final_path,
                target,
                manifest_path,
                transform_context=QgsCoordinateTransformContext(),
            )
        except RuntimeError as error:
            assert str(error) == message
        else:
            raise AssertionError(
                "an existing target was written without OGR transactions"
            )
    assert target.stat().st_ino == inode
    assert _accepted_object_ids(target) == before_ids


def accepted_internal_wal_identity(app, root):
    run_id, final_path, target, manifest_path = accepted_fixture(root, existing=True)
    inode = target.stat().st_ino
    task = AcceptedWriteTask(
        1,
        run_id=run_id,
        final_path=final_path,
        accepted_path=target,
        run_manifest_path=manifest_path,
        workspace_input_identities={},
        transform_context=QgsCoordinateTransformContext(),
    )
    original_identity = accepted_writer.file_identity
    original_start = accepted_writer._start_existing_transaction
    transaction_started = []

    def start_with_owned_wal(path):
        result = original_start(path)
        transaction_started.append(True)
        return result

    def identity_with_owned_wal(path):
        identity = original_identity(path)
        if transaction_started and Path(path).resolve() == target.resolve():
            return identity[0], ("transaction-owned-wal",)
        return identity

    with (
        patch.object(
            accepted_writer,
            "_start_existing_transaction",
            side_effect=start_with_owned_wal,
        ),
        patch.object(
            accepted_writer,
            "file_identity",
            side_effect=identity_with_owned_wal,
        ),
    ):
        assert run_accepted_task(app, task) == ["completed"]
    assert task.published and transaction_started == [True]
    assert target.stat().st_ino == inode
    assert _accepted_object_ids(target) == {"old", "new"}


def accepted_post_publish_warning(app, root):
    for name in ("unlink", "fsync"):
        case_root = root / name
        case_root.mkdir()
        run_id, final_path, target, manifest_path = accepted_fixture(case_root)
        task = AcceptedWriteTask(
            1,
            run_id=run_id,
            final_path=final_path,
            accepted_path=target,
            run_manifest_path=manifest_path,
            workspace_input_identities={},
            transform_context=QgsCoordinateTransformContext(),
        )
        stack = ExitStack()
        if name == "unlink":
            original_unlink = Path.unlink

            def fail_published_staging(path, *args, **kwargs):
                candidate = Path(path)
                if (
                    target.exists()
                    and candidate.parent.resolve() == target.parent.resolve()
                    and candidate.name.endswith(".staging.gpkg")
                ):
                    raise PermissionError("injected post-publish unlink failure")
                return original_unlink(candidate, *args, **kwargs)

            stack.enter_context(patch.object(Path, "unlink", fail_published_staging))
        else:
            stack.enter_context(
                patch.object(
                    accepted_writer.os,
                    "fsync",
                    side_effect=OSError("injected post-publish fsync failure"),
                )
            )
        with stack:
            assert run_accepted_task(app, task) == ["completed"]
        assert task.published and task.error_message == ""
        assert task.result_data["feature_count"] == 1
        assert len(task.result_data["warnings"]) == 1
        assert "已写入，但" in task.result_data["warnings"][0]
        assert _accepted_object_ids(target) == {"new"}


@contextmanager
def dialog_fixture(app, root):
    canvas = QgsMapCanvas()
    iface = SimpleNamespace(
        mapCanvas=lambda: canvas,
        activeLayer=lambda: None,
        cadDockWidget=lambda: None,
    )
    with ExitStack() as stack:
        submit = stack.enter_context(patch.object(ui.QgsApplication, "taskManager"))
        warning = stack.enter_context(patch.object(ui.QMessageBox, "warning"))
        stack.enter_context(
            patch.object(
                ui.ClassRefinementDialog, "_final_matches_workspace", return_value=True
            )
        )
        stack.enter_context(
            patch.object(
                class_workspace,
                "save_workspace",
                side_effect=lambda _spec, value, **_kwargs: value,
            )
        )
        dialog = ui.ClassRefinementDialog(iface, SimpleNamespace())
        dialog._workspace = {
            "classes": {
                str(code): {"confirmed": True, "path": str(root / f"class_{code}.gpkg")}
                for code in CLASS_ORDER
            }
        }
        for record in dialog._workspace["classes"].values():
            Path(record["path"]).write_bytes(b"fixture")
        dialog._final_input_identities = {
            record["path"]: workspace_file_identity(record["path"])
            for record in dialog._workspace["classes"].values()
        }
        dialog._final_path = str(root / "final.gpkg")
        Path(dialog._final_path).write_bytes(b"fixture")
        dialog._issue_count = 0
        dialog._run_spec = {
            "run_id": "old",
            "run_dir": str(root),
            "accepted_target_gpkg": str(root / "accepted.gpkg"),
            "accepted_write_manifest": str(root / "manifest.json"),
        }
        try:
            yield dialog, submit.return_value.addTask, warning
        finally:
            dialog._workspace = None
            dialog.cleanup()
            dialog.deleteLater()
            drain(app)
            canvas.close()


def dialog_lifecycle(app, root):
    with dialog_fixture(app, root) as (dialog, submit, warning):
        dialog._write_accepted()
        success = submit.call_args.args[0]
        write_button = dialog.findChild(QPushButton, "AdmissionWriteButton")
        assert write_button is not None and not write_button.isEnabled()
        success.commit_started = True
        success.progress_message = "正在提交，不能取消"
        success.progressChanged.emit(82)
        assert not dialog.cancel_load_btn.isEnabled()
        success.published = True
        success.result_data = {"run_id": "old", "feature_count": 3}
        success.taskCompleted.emit()
        assert "3 个面" in dialog.baseline_label.text()

        dialog._write_accepted()
        failed = submit.call_args.args[0]
        failed.error_message = "injected failure"
        failed.taskTerminated.emit()
        assert warning.call_args.args[-1] == "injected failure"

        dialog._write_accepted()
        switched = submit.call_args.args[0]
        before = dialog.baseline_label.text()
        dialog._retire_accepted_task()
        dialog._run_spec = {**dialog._run_spec, "run_id": "new"}
        switched.result_data = {"run_id": "old", "feature_count": 99}
        switched.taskCompleted.emit()
        assert dialog.baseline_label.text() == before

        dialog._run_spec = {**dialog._run_spec, "run_id": "close"}
        dialog._write_accepted()
        closing = submit.call_args.args[0]
        closing.commit_started = True
        before = dialog.baseline_label.text()
        dialog.cleanup()
        assert not closing.isCanceled()
        closing.result_data = {"run_id": "close", "feature_count": 5}
        closing.taskCompleted.emit()
        assert dialog.baseline_label.text() == before


def monitor_navigation(app, root):
    run_id = "20260925_130000_abcd"
    run_dir = root / run_id
    run_dir.mkdir()
    (run_dir / "run_spec.json").write_text("{}", encoding="utf-8")
    with ExitStack() as stack:
        for name in (
            "_save_settings",
            "_load_settings_and_defaults",
            "_restore_latest_ready_run",
        ):
            stack.enter_context(
                patch.object(main_dock.LabelingDockWidget, name, lambda *_: None)
            )
        dock = main_dock.LabelingDockWidget()
        try:
            dock.hide()
            drain(app)
            payload = {
                "run_id": run_id,
                "observed_status": "failed",
                "run_spec": {
                    "schema_version": 2,
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                },
            }
            dock._on_monitor_main_run_handling(payload)
            drain(app)
            assert dock.isVisible() and dock.retry_failed_btn.hasFocus()
            dock.hide()
            dock._on_monitor_main_run_handling({"run_id": "bad"})
            drain(app)
            assert not dock.isVisible()
        finally:
            dock.cleanup()
            dock.deleteLater()
            drain(app)


app = QgsApplication([], False)
app.initQgis()
try:
    with tempfile.TemporaryDirectory(prefix="loess-background-io-") as directory:
        globals()[sys.argv[2]](app, Path(directory))
        print(sys.argv[2] + ": passed", flush=True)
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
finally:
    app.exitQgis()
