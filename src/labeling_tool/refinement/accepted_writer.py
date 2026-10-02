"""Validate and atomically append one final result to the accepted label store."""

from __future__ import annotations

import contextlib
import fcntl
import gc
import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Callable

from qgis.core import (
    QgsFeature,
    QgsField,
    QgsProject,
    QgsVariantUtils,
    QgsVectorFileWriter,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QDateTime, QVariant

from labeling_tool.qgis_support.layer_names import LAYER_NAMES
from labeling_tool.qgis_support.qgis_writer import write_vector_layer
from labeling_tool.qgis_support.qt6_api import ISO_DATE
from labeling_tool.refinement import accepted_integrity
from labeling_tool.shared.contracts.run_spec import CLASS_NAMES

LOGGER = logging.getLogger(__name__)


class AcceptedWriteCancelled(RuntimeError):
    pass


CancelCheck = Callable[[], bool] | None
ProgressCallback = Callable[[str, float], None] | None
WarningCallback = Callable[[str], None] | None


ACCEPTED_FIELDS = [
    ("run_id", QVariant.String),
    ("object_id", QVariant.String),
    ("part_id", QVariant.String),
    ("class_code", QVariant.Int),
    ("class_name", QVariant.String),
    ("confidence_mean", QVariant.Double),
    ("confidence_std", QVariant.Double),
    ("baseline_stream_id", QVariant.String),
    ("source_stream_id", QVariant.String),
    ("source", QVariant.String),
    ("geometry_source", QVariant.String),
    ("geometry_revision", QVariant.Int),
    ("edit_base", QVariant.String),
    ("sam_session_id", QVariant.String),
    ("sam_score", QVariant.Double),
    ("model_version", QVariant.String),
    ("fusion_profile_id", QVariant.String),
    ("sam_version", QVariant.String),
    ("reviewed", QVariant.Int),
    ("created_at", QVariant.String),
    ("updated_at", QVariant.String),
]

ACCEPTED_FIELDS_QGS = [QgsField(name, typ) for name, typ in ACCEPTED_FIELDS]


def _ogr_modules():
    try:
        from osgeo import gdal, ogr
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "当前 QGIS Python 缺少 osgeo/OGR 事务绑定，不能写入现有 accepted_labels"
        ) from exc
    return gdal, ogr


def _build_accepted_fields_qgs():
    return [QgsField(name, typ) for name, typ in ACCEPTED_FIELDS]


def _check_canceled(is_canceled: CancelCheck) -> None:
    if is_canceled is not None and is_canceled():
        raise AcceptedWriteCancelled("accepted_labels 写入已取消；目标未改变")


def _progress(callback: ProgressCallback, message: str, value: float) -> None:
    if callback is not None:
        callback(message, float(value))


def file_identity(path: str | os.PathLike[str]) -> tuple[object, object]:
    value = Path(path)

    def identity(item: Path):
        try:
            info = item.stat()
        except FileNotFoundError:
            return None
        return (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )

    return identity(value), identity(Path(str(value) + "-wal"))


def _sha256(path: Path, *, is_canceled: CancelCheck = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            _check_canceled(is_canceled)
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def get_accepted_layer(gpkg_path, crs=None, *, transform_context=None):
    """Open/create a detached accepted layer.

    Background callers supply both ``crs`` and ``transform_context`` so this
    helper does not consult the GUI-owned QgsProject singleton.
    """

    if os.path.exists(gpkg_path):
        uri = f"{gpkg_path}|layername={LAYER_NAMES.ACCEPTED}"
        layer = QgsVectorLayer(uri, LAYER_NAMES.ACCEPTED, "ogr")
        if layer.isValid():
            return layer
        raise RuntimeError(
            f"existing GeoPackage has no valid {LAYER_NAMES.ACCEPTED} layer: {gpkg_path}"
        )

    if crs is None:
        project_crs = QgsProject.instance().crs()
        crs = project_crs.authid() if project_crs.isValid() else "EPSG:4326"

    mem_layer = QgsVectorLayer(
        f"MultiPolygon?crs={crs}", LAYER_NAMES.ACCEPTED, "memory"
    )
    mem_layer.dataProvider().addAttributes(_build_accepted_fields_qgs())
    mem_layer.updateFields()

    save_options = QgsVectorFileWriter.SaveVectorOptions()
    save_options.driverName = "GPKG"
    save_options.layerName = LAYER_NAMES.ACCEPTED
    save_options.actionOnExistingFile = (
        QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteFile
    )
    error, err_msg = write_vector_layer(
        mem_layer,
        gpkg_path,
        save_options,
        transform_context=transform_context,
    )
    if error != QgsVectorFileWriter.WriterError.NoError:
        raise RuntimeError(f"Failed to create accepted_labels GPKG: {err_msg}")

    uri = f"{gpkg_path}|layername={LAYER_NAMES.ACCEPTED}"
    return QgsVectorLayer(uri, LAYER_NAMES.ACCEPTED, "ogr")


def _verified_run_spec(manifest, run_manifest_path, *, is_canceled=None):
    run_spec_value = str(manifest.get("run_spec") or "").strip()
    if not run_spec_value:
        raise ValueError("run_manifest 缺少 run_spec 路径")
    run_spec_path = Path(run_spec_value).expanduser()
    if not run_spec_path.is_absolute():
        run_spec_path = Path(run_manifest_path).resolve().parent / run_spec_path
    run_spec_path = run_spec_path.resolve()
    frozen_identity = file_identity(run_spec_path)
    expected_sha = str(manifest.get("run_spec_sha256") or "")
    if (
        not run_spec_path.is_file()
        or not expected_sha
        or _sha256(run_spec_path, is_canceled=is_canceled) != expected_sha
    ):
        raise ValueError("run_spec 不存在或 SHA256 与 run_manifest 不一致")
    if file_identity(run_spec_path) != frozen_identity:
        raise ValueError("run_spec 在 SHA256 校验期间发生变化")
    with run_spec_path.open("r", encoding="utf-8") as handle:
        run_spec = json.load(handle)
    if file_identity(run_spec_path) != frozen_identity:
        raise ValueError("run_spec 在读取期间发生变化")
    if str(run_spec.get("run_id") or "") != str(manifest.get("run_id") or ""):
        raise ValueError("run_spec 与 run_manifest 的 run_id 不一致")
    return run_spec, run_spec_path, frozen_identity


def _read_contract(run_manifest_path, *, is_canceled=None):
    _check_canceled(is_canceled)
    manifest_path = Path(run_manifest_path).expanduser().resolve()
    manifest_identity = file_identity(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if file_identity(manifest_path) != manifest_identity:
        raise ValueError("run_manifest 在读取期间发生变化")
    if manifest.get("status") != "ready":
        raise ValueError("run_manifest must be ready before accepted_labels write")
    run_spec, run_spec_path, run_spec_identity = _verified_run_spec(
        manifest,
        manifest_path,
        is_canceled=is_canceled,
    )
    return (
        manifest,
        run_spec,
        manifest_path,
        run_spec_path,
        {
            manifest_path: manifest_identity,
            run_spec_path: run_spec_identity,
        },
    )


@contextlib.contextmanager
def _target_lock(target: Path, *, is_canceled: CancelCheck = None):
    """Serialize accepted publication through a stable, never-unlinked inode."""

    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(target.name + ".write.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        while True:
            _check_canceled(is_canceled)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.05)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _new_staging_path(target: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".staging.gpkg",
        dir=target.parent,
    )
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    return path


def _cleanup_staging(path: Path) -> None:
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        with contextlib.suppress(FileNotFoundError):
            candidate.unlink()


def _publish_warning(callback: WarningCallback, message: str) -> None:
    LOGGER.warning(message)
    if callback is not None:
        callback(message)


def _publish_new_staged(
    staging: Path,
    target: Path,
    *,
    warning: WarningCallback = None,
) -> None:
    """Publish a new target without replacing a path which appeared meanwhile."""

    try:
        os.link(staging, target)
    except FileExistsError as exc:
        raise RuntimeError("accepted_labels 目标已由其他任务创建，结果未发布") from exc

    # os.link is the publication commit point. Cleanup and durability failures
    # after it must not report an unpublished task and invite a duplicate write.
    try:
        staging.unlink()
    except OSError as exc:
        _publish_warning(
            warning,
            "accepted_labels 已写入，但临时文件清理失败："
            f"target={target}, staging={staging}, error={exc}",
        )
    try:
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        _publish_warning(
            warning,
            "accepted_labels 已写入，但目录持久化确认失败："
            f"target={target}, error={exc}",
        )


def _validate_target_contract(run_spec, accepted_path) -> Path:
    expected_target_value = str(run_spec.get("accepted_target_gpkg") or "").strip()
    if not expected_target_value:
        raise ValueError("run_spec 缺少 accepted_target_gpkg，禁止写入 Run 快照")
    if not str(accepted_path or "").strip():
        raise ValueError("accepted_labels 写入目标为空")
    expected_target = Path(expected_target_value).expanduser().resolve()
    actual_target = Path(accepted_path).expanduser().resolve()
    if actual_target != expected_target:
        raise ValueError(
            f"accepted_labels 写入目标与 run_spec 不一致: {actual_target}/{expected_target}"
        )
    snapshot_value = str(run_spec.get("accepted_gpkg") or "").strip()
    if snapshot_value and Path(snapshot_value).expanduser().resolve() == actual_target:
        raise ValueError("accepted_labels 长期写入目标不得等于 Run 只读快照")
    return actual_target


def _build_pending_features(final, accepted, manifest, *, is_canceled, progress):
    valid_streams = {
        str(item.get("stream_id"))
        for item in manifest.get("streams") or []
        if item.get("status") == "ready"
    }
    manifest_run_id = str(manifest.get("run_id") or "")
    existing_identities = {
        (
            str(feature.attribute("run_id") or ""),
            str(feature.attribute("object_id") or ""),
            str(feature.attribute("part_id") or ""),
        )
        for feature in accepted.getFeatures()
        if str(feature.attribute("object_id") or "")
    }
    accepted_in_run = set()
    validated_features = []
    total = max(int(final.featureCount()), 1)
    for index, feature in enumerate(final.getFeatures(), start=1):
        _check_canceled(is_canceled)
        geometry = feature.geometry()
        if (
            geometry is None
            or geometry.isNull()
            or geometry.isEmpty()
            or not geometry.isGeosValid()
        ):
            raise ValueError(f"invalid final geometry for feature {feature.id()}")
        class_code = int(feature.attribute("class_code"))
        class_name = str(feature.attribute("class_name"))
        if CLASS_NAMES.get(class_code) != class_name:
            raise ValueError(f"class mapping mismatch: {class_code}/{class_name}")
        feature_run_id = str(feature.attribute("run_id") or "")
        if not manifest_run_id or feature_run_id != manifest_run_id:
            raise ValueError(
                "final feature run_id does not match run_manifest: "
                f"{feature_run_id}/{manifest_run_id}"
            )
        source_stream_id = str(feature.attribute("source_stream_id") or "")
        if source_stream_id not in valid_streams:
            raise ValueError(
                f"source_stream_id is not traceable in run_manifest: {source_stream_id}"
            )
        object_id = str(feature.attribute("object_id") or "")
        part_id = str(feature.attribute("part_id") or "000")
        run_key = (feature_run_id, object_id, part_id)
        if (
            not object_id
            or not part_id
            or run_key in existing_identities
            or run_key in accepted_in_run
        ):
            raise ValueError(f"duplicate or empty accepted object identity: {run_key}")
        _validate_provenance(feature)
        existing_identities.add(run_key)
        accepted_in_run.add(run_key)
        validated_features.append(QgsFeature(feature))
        if index % 100 == 0 or index == total:
            _progress(
                progress, f"正在校验待入库面 {index}/{total}", 45 + 30 * index / total
            )

    pending = []
    for feature in validated_features:
        output = QgsFeature(accepted.fields())
        output.setGeometry(feature.geometry())
        for name, _type in ACCEPTED_FIELDS:
            if name == "reviewed":
                value = 1
            elif name == "source":
                value = "class_working"
            elif name == "created_at":
                value = feature.attribute(name) or QDateTime.currentDateTime().toString(
                    ISO_DATE
                )
            else:
                field_index = feature.fieldNameIndex(name)
                value = feature.attribute(field_index) if field_index >= 0 else ""
            output.setAttribute(name, value)
        pending.append(output)
    return pending


def _start_existing_transaction(target: Path):
    """Open an OGR/SQLite write transaction and acquire its database write lock."""

    gdal, ogr = _ogr_modules()
    dataset = ogr.Open(str(target), 1)
    if dataset is None:
        raise RuntimeError(f"cannot open accepted_labels for update: {target}")
    if not dataset.TestCapability(ogr.ODsCTransactions):
        dataset = None
        raise RuntimeError("accepted_labels OGR provider does not support transactions")
    result = dataset.StartTransaction()
    if result != ogr.OGRERR_NONE:
        dataset = None
        raise RuntimeError(
            "cannot start accepted_labels transaction: "
            f"{gdal.GetLastErrorMsg() or result}"
        )
    try:
        layer = dataset.GetLayerByName(LAYER_NAMES.ACCEPTED)
        if layer is None:
            raise RuntimeError("accepted_labels transaction layer is missing")
        # A zero-row UPDATE acquires SQLite's write reservation without changing
        # user data. This blocks other writers while the QGIS read-only audit runs.
        gdal.ErrorReset()
        dataset.ExecuteSQL(
            'UPDATE "accepted_labels" SET "object_id"="object_id" WHERE 0'
        )
        if gdal.GetLastErrorType() >= gdal.CE_Failure:
            raise RuntimeError(
                "cannot lock accepted_labels transaction: "
                f"{gdal.GetLastErrorMsg() or 'OGR transaction error'}"
            )
        return dataset, layer
    except Exception:
        dataset.RollbackTransaction()
        dataset = None
        raise


def _ogr_feature(layer, feature):
    _gdal, ogr = _ogr_modules()
    output = ogr.Feature(layer.GetLayerDefn())
    for name, _type in ACCEPTED_FIELDS:
        value = feature.attribute(name)
        if not QgsVariantUtils.isNull(value):
            output.SetField(name, value)
    geometry = ogr.CreateGeometryFromWkb(bytes(feature.geometry().asWkb()))
    if geometry is None:
        raise RuntimeError("cannot convert accepted geometry for OGR transaction")
    output.SetGeometry(geometry)
    return output


def _insert_ogr_feature(layer, feature) -> None:
    gdal, ogr = _ogr_modules()
    output = _ogr_feature(layer, feature)
    result = layer.CreateFeature(output)
    if result != ogr.OGRERR_NONE:
        raise RuntimeError(
            "cannot append accepted feature in OGR transaction: "
            f"{gdal.GetLastErrorMsg() or result}"
        )


def _commit_existing_target(dataset, layer, pending) -> None:
    gdal, ogr = _ogr_modules()
    for feature in pending:
        _insert_ogr_feature(layer, feature)
    result = dataset.CommitTransaction()
    if result != ogr.OGRERR_NONE:
        raise RuntimeError(
            "cannot commit accepted_labels transaction: "
            f"{gdal.GetLastErrorMsg() or result}"
        )


def _commit_new_target(
    staging: Path,
    actual_target: Path,
    pending,
    *,
    overlap_tolerance,
    transform_context,
    warning: WarningCallback,
) -> None:
    accepted = get_accepted_layer(
        str(staging),
        transform_context=transform_context,
    )
    if not accepted.startEditing():
        raise RuntimeError("cannot start accepted_labels edit session")
    if not accepted.addFeatures(pending):
        accepted.rollBack()
        raise RuntimeError("cannot add final features to accepted_labels")
    if not accepted.commitChanges():
        errors = "; ".join(accepted.commitErrors())
        accepted.rollBack()
        raise RuntimeError(f"cannot commit accepted_labels: {errors}")
    del accepted
    gc.collect()
    staged = get_accepted_layer(
        str(staging),
        transform_context=transform_context,
    )
    accepted_integrity.audit_accepted_layer(
        staged,
        overlap_tolerance=overlap_tolerance,
    )
    del staged
    gc.collect()
    _publish_new_staged(staging, actual_target, warning=warning)


def append_final_to_accepted(
    final_path,
    accepted_path,
    run_manifest_path,
    *,
    is_canceled: CancelCheck = None,
    progress: ProgressCallback = None,
    warning: WarningCallback = None,
    before_commit: Callable[[], None] | None = None,
    transform_context=None,
):
    """Append transactionally while preserving existing database identity.

    Existing targets use an OGR/SQLite transaction against the same inode. New
    targets are built in a private GPKG and linked without overwrite. Validation
    is cancellable; ``before_commit`` marks the irreversible UI boundary.
    """

    (
        manifest,
        run_spec,
        _manifest_path,
        _run_spec_path,
        contract_identities,
    ) = _read_contract(
        run_manifest_path,
        is_canceled=is_canceled,
    )
    actual_target = _validate_target_contract(run_spec, accepted_path)
    final_path = Path(final_path).expanduser().resolve()
    contract_identities[final_path] = file_identity(final_path)
    _progress(progress, "正在等待 accepted_labels 写入锁", 2)
    with _target_lock(actual_target, is_canceled=is_canceled):
        _check_canceled(is_canceled)
        expected_target_identity = file_identity(actual_target)
        existing_target = expected_target_identity[0] is not None
        staging = None if existing_target else _new_staging_path(actual_target)
        accepted = None
        final = None
        transaction = None
        transaction_layer = None
        try:
            if existing_target:
                transaction, transaction_layer = _start_existing_transaction(
                    actual_target
                )
                if file_identity(actual_target)[0] != expected_target_identity[0]:
                    raise RuntimeError(
                        "accepted_labels 在取得数据库事务时发生变化，目标未写入"
                    )
            final = QgsVectorLayer(
                f"{final_path}|layername={LAYER_NAMES.FINAL_COMPOSITE}",
                "final_to_accept",
                "ogr",
            )
            if not final.isValid():
                raise RuntimeError(f"cannot open final_composite: {final_path}")
            if existing_target:
                accepted = get_accepted_layer(
                    str(actual_target),
                    final.crs().authid(),
                    transform_context=transform_context,
                )
            else:
                accepted = get_accepted_layer(
                    str(staging),
                    final.crs().authid(),
                    transform_context=transform_context,
                )
            if not accepted.isValid():
                raise RuntimeError(
                    f"cannot open accepted_labels: "
                    f"{actual_target if existing_target else staging}"
                )
            overlap_tolerance = accepted_integrity.strict_overlap_tolerance(run_spec)
            _progress(progress, "正在审计 accepted_labels", 12)
            accepted_integrity.audit_accepted_layer(
                accepted,
                overlap_tolerance=overlap_tolerance,
                expected_crs=final.crs(),
                is_canceled=is_canceled,
            )
            _progress(progress, "正在检查与 accepted_labels 的重叠", 32)
            accepted_integrity.assert_no_accepted_overlap(
                final,
                accepted,
                overlap_tolerance=overlap_tolerance,
                transform_context=transform_context,
                is_canceled=is_canceled,
            )
            pending = _build_pending_features(
                final,
                accepted,
                manifest,
                is_canceled=is_canceled,
                progress=progress,
            )
            _check_canceled(is_canceled)
            if any(
                file_identity(path) != identity
                for path, identity in contract_identities.items()
            ):
                raise RuntimeError("入库校验期间 final 或 Run 合同发生变化，目标未改变")
            current_target_identity = file_identity(actual_target)
            target_changed = (
                current_target_identity[0] != expected_target_identity[0]
                if existing_target
                else current_target_identity != expected_target_identity
            )
            if target_changed:
                raise RuntimeError("入库校验期间 accepted_labels 发生变化，目标未改变")
            if before_commit is not None:
                before_commit()
            run_id = str(manifest.get("run_id") or "")
            LOGGER.info(
                "accepted write commit started run_id=%s target=%s features=%d",
                run_id,
                actual_target,
                len(pending),
            )
            _progress(progress, "正在提交，不能取消", 82)
            del accepted
            accepted = None
            del final
            final = None
            gc.collect()
            if existing_target:
                _commit_existing_target(
                    transaction,
                    transaction_layer,
                    pending,
                )
                transaction = None
                transaction_layer = None
            else:
                _commit_new_target(
                    staging,
                    actual_target,
                    pending,
                    overlap_tolerance=overlap_tolerance,
                    transform_context=transform_context,
                    warning=warning,
                )
                staging = None
            LOGGER.info(
                "accepted write committed run_id=%s target=%s features=%d",
                run_id,
                actual_target,
                len(pending),
            )
            _progress(progress, "accepted_labels 写入完成", 100)
            return len(pending)
        except Exception as exc:
            if is_canceled is not None and is_canceled():
                raise AcceptedWriteCancelled(
                    "accepted_labels 写入已取消；目标未改变"
                ) from exc
            LOGGER.exception(
                "accepted write failed run_id=%s target=%s",
                str(manifest.get("run_id") or ""),
                actual_target,
            )
            raise
        finally:
            if transaction is not None:
                transaction.RollbackTransaction()
            if accepted is not None and accepted.isEditable():
                accepted.rollBack()
            del accepted
            del final
            gc.collect()
            if staging is not None:
                _cleanup_staging(staging)


def _validate_provenance(feature):
    geometry_source = str(feature.attribute("geometry_source") or "")
    revision = int(feature.attribute("geometry_revision") or 0)
    edit_base = str(feature.attribute("edit_base") or "")
    sam_session_id = str(feature.attribute("sam_session_id") or "")
    if geometry_source == "fusion":
        valid = revision == 0 and not edit_base and not sam_session_id
    elif geometry_source == "sam3":
        valid = revision >= 1 and not edit_base and bool(sam_session_id)
    elif geometry_source == "manual_edited":
        valid = (
            revision >= 1
            and edit_base in ("", "fusion", "sam3", "manual_edited")
            and (edit_base != "sam3" or bool(sam_session_id))
        )
    else:
        valid = False
    if not valid:
        raise ValueError(
            "invalid geometry provenance: "
            f"source={geometry_source}, revision={revision}, "
            f"edit_base={edit_base}, sam_session_id={sam_session_id}"
        )
