"""Read ready Runs for class refinement or explicit workspace initialization.

The expensive validation and hashing phase is separated from publication so a
QgsTask can cancel safely before any metadata is changed. ``load_manual_run``
remains the synchronous entry point for callers which explicitly do not use the
GUI task manager.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Callable

from labeling_tool.shared.contracts.run_spec import (
    CLASS_ORDER,
    atomic_write_json,
    sha256_file,
)


class ManualRunLoadError(RuntimeError):
    pass


class ManualRunLoadCancelled(ManualRunLoadError):
    pass


CancelCheck = Callable[[], bool] | None
ProgressCallback = Callable[[str, float], None] | None
LOAD_CLASSES = "classes"
LOAD_ALL = "all"
LOAD_SCOPES = (LOAD_CLASSES, LOAD_ALL)


def _check_canceled(is_canceled: CancelCheck) -> None:
    if is_canceled is not None and is_canceled():
        raise ManualRunLoadCancelled("加载已取消；Run 文件未发布")


def _progress(callback: ProgressCallback, message: str, value: float) -> None:
    if callback is not None:
        callback(message, float(value))


def _read_json(
    path: Path,
    label: str,
    *,
    is_canceled: CancelCheck = None,
    input_identities: dict[str, tuple[object, object]] | None = None,
) -> dict:
    _check_canceled(is_canceled)
    if not path.is_file():
        raise ManualRunLoadError(f"缺少 {label}: {path}")
    identity = file_identity(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManualRunLoadError(f"无法读取 {label}: {path}: {exc}") from exc
    _check_canceled(is_canceled)
    if file_identity(path) != identity:
        raise ManualRunLoadError(f"读取期间 {label} 发生变化: {path}")
    if input_identities is not None:
        input_identities[str(path.resolve())] = identity
    if not isinstance(value, dict):
        raise ManualRunLoadError(f"{label} 必须是 JSON 对象: {path}")
    return value


def file_identity(path: str | os.PathLike[str]) -> tuple[object, object]:
    """Return the main-file and WAL identities used by background validation."""

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
    try:
        with path.open("rb") as handle:
            while True:
                _check_canceled(is_canceled)
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as exc:
        raise ManualRunLoadError(f"无法读取文件: {path}: {exc}") from exc
    _check_canceled(is_canceled)
    return digest.hexdigest()


def _stable_sha256(
    path: Path,
    *,
    is_canceled: CancelCheck = None,
    input_identities: dict[str, tuple[object, object]] | None = None,
) -> str:
    before = file_identity(path)
    if before[0] is None:
        raise ManualRunLoadError(f"缺少文件: {path}")
    digest = _sha256(path, is_canceled=is_canceled)
    if file_identity(path) != before:
        raise ManualRunLoadError(f"校验期间文件发生变化: {path}")
    if input_identities is not None:
        key = str(path.resolve())
        previous = input_identities.get(key, before)
        if previous != before:
            raise ManualRunLoadError(f"读取与校验之间文件发生变化: {path}")
        input_identities[key] = before
    return digest


def _rebound_candidate(raw_path, old_root: Path | None, new_root: Path, run_id: str):
    raw = str(raw_path or "").strip()
    if not raw:
        return None
    source = Path(raw).expanduser()
    if not source.is_absolute():
        return (new_root / source).resolve()
    if old_root is not None:
        try:
            # QGIS records canonical paths; the Run spec may retain a directory
            # alias (for example /var versus /private/var on macOS).
            return (new_root / source.resolve().relative_to(old_root.resolve())).resolve()
        except ValueError:
            pass
    parts = source.parts
    if run_id in parts:
        index = len(parts) - 1 - list(reversed(parts)).index(run_id)
        return (new_root / Path(*parts[index + 1 :])).resolve()
    return None


def _rebind_stream(
    stream,
    old_root,
    run_root,
    run_id,
    *,
    require_semantic=True,
):
    rebound = copy.deepcopy(stream)
    stream_id = str(rebound.get("stream_id") or "")
    kind = str(rebound.get("kind") or "")
    identifier_key = "model_id" if kind == "model" else "fusion_profile_id"
    profile_id = str(rebound.get(identifier_key) or "")
    if not profile_id and stream_id.startswith(f"{kind}:"):
        profile_id = stream_id.split(":", 1)[1]
    if kind not in {"model", "fusion"} or not profile_id or Path(profile_id).name != profile_id:
        raise ManualRunLoadError(
            f"Fusion 流缺少 profile_id: {stream_id or '<unknown>'}"
        )

    expected_dir = (run_root / ("models" if kind == "model" else "fusion") / profile_id).resolve()
    paths = dict(rebound.get("paths") or {})
    for name, raw_path in list(paths.items()):
        candidate = _rebound_candidate(raw_path, old_root, run_root, run_id)
        fallback = expected_dir / Path(str(raw_path or name)).name
        if candidate is not None and candidate.exists():
            paths[name] = str(candidate)
        elif fallback.exists():
            paths[name] = str(fallback.resolve())
        elif candidate is not None:
            paths[name] = str(candidate)

    semantic_value = str(paths.get("semantic_polygons") or "").strip()
    semantic = Path(semantic_value) if semantic_value else None
    if semantic is None or not semantic.is_file():
        fallback = expected_dir / "semantic_polygons.gpkg"
        if fallback.is_file():
            semantic = fallback.resolve()
            paths["semantic_polygons"] = str(semantic)
        elif require_semantic:
            raise ManualRunLoadError(
                f"Fusion {stream_id} 缺少 semantic_polygons.gpkg: {fallback}"
            )
    rebound["paths"] = paths
    rebound[identifier_key] = profile_id
    semantic_available = semantic is not None and semantic.is_file()
    review = _rebound_candidate(stream.get("review_polygons"), old_root, run_root, run_id)
    if stream.get("review_polygons"):
        rebound["review_polygons"] = str(review or expected_dir / Path(stream["review_polygons"]).name)
    else:
        rebound["review_polygons"] = str(semantic.resolve()) if semantic_available else ""
    rebound["review_layer_name"] = stream.get("review_layer_name") or "semantic_polygons"
    rebound["manual_validated"] = False
    rebound["manual_offline"] = not semantic_available
    return rebound


def _validate_result_files(stream, *, is_canceled, input_identities):
    """Validate the published artifacts, never temporary inference caches."""
    from labeling_tool.runs.result_catalog import artifact_sha256

    paths = dict(stream.get("paths") or {})
    checksums = stream.get("output_sha256") or {}
    required = {"mask_mosaic", "confidence_mosaic", "semantic_polygons"}
    if paths.get("semantic_polygons_raw"):
        required.add("semantic_polygons_raw")
    if stream.get("review_polygons"):
        paths["review_polygons"] = stream["review_polygons"]
        if paths["review_polygons"] != paths.get("semantic_polygons"):
            required.add("review_polygons")
    for key in sorted(required | set(checksums)):
        _check_canceled(is_canceled)
        raw = paths.get(key)
        if not raw or not Path(raw).is_file():
            raise ManualRunLoadError(f"{stream['stream_id']} 缺少结果文件 {key}: {raw or '未记录'}")
        path = Path(raw)
        before = file_identity(path)
        if path.suffix.lower() == ".vrt":
            digest = artifact_sha256(path)
            if file_identity(path) != before:
                raise ManualRunLoadError(f"校验期间文件发生变化: {path}")
            input_identities[str(path.resolve())] = before
        else:
            digest = _stable_sha256(path, is_canceled=is_canceled, input_identities=input_identities)
        if not checksums.get(key) or checksums[key] != digest:
            raise ManualRunLoadError(f"{stream['stream_id']} 结果 {key} 的 SHA256 缺失或不匹配")


def _prepare_workspace(
    run_root: Path,
    run_id: str,
    fusion_streams: list[dict],
    *,
    old_root: Path | None = None,
    is_canceled: CancelCheck = None,
    progress: ProgressCallback = None,
    input_identities: dict[str, tuple[object, object]] | None = None,
):
    workspace_path = run_root / "classes" / "workspace.json"
    if not workspace_path.is_file():
        return None
    workspace = _read_json(
        workspace_path,
        "classes/workspace.json",
        is_canceled=is_canceled,
        input_identities=input_identities,
    )
    if str(workspace.get("run_id") or "") != run_id:
        raise ManualRunLoadError("classes/workspace.json 的 run_id 与 Run 不一致")
    if workspace.get("schema_version") != 2:
        raise ManualRunLoadError("classes/workspace.json 版本不受支持")
    classes = workspace.get("classes") or {}
    expected_codes = {str(code) for code in CLASS_ORDER}
    if set(classes) != expected_codes:
        missing = sorted(expected_codes - set(classes))
        extra = sorted(set(classes) - expected_codes)
        raise ManualRunLoadError(
            f"类别工作区必须恰好包含14类；缺少={missing}，多余={extra}"
        )

    stream_map = {item["stream_id"]: item for item in fusion_streams}
    baseline_id = str(workspace.get("baseline_stream_id") or "")
    if baseline_id not in stream_map:
        raise ManualRunLoadError(
            f"类别工作区引用的 Fusion 不存在或不可用: {baseline_id}"
        )
    rebound = copy.deepcopy(workspace)
    for name in ("baseline_source_path", "formal_path", "boundary_report_path"):
        candidate = _rebound_candidate(workspace.get(name), old_root, run_root, run_id)
        if candidate is not None:
            rebound[name] = str(candidate)
    baseline_value = str(rebound.get("baseline_source_path") or "").strip()
    baseline_path = Path(baseline_value).resolve() if baseline_value else None
    baseline_available = baseline_path is not None and baseline_path.is_file()
    rebound["baseline_available"] = baseline_available
    rebound["portable_classes_only"] = not baseline_available

    count = len(CLASS_ORDER)
    for index, code in enumerate(CLASS_ORDER, start=1):
        _check_canceled(is_canceled)
        record = dict(classes[str(code)])
        class_path = (run_root / "classes" / f"class_{code}.gpkg").resolve()
        if not class_path.is_file():
            raise ManualRunLoadError(f"缺少类别工作层: {class_path}")
        record["path"] = str(class_path)
        digest = _stable_sha256(
            class_path,
            is_canceled=is_canceled,
            input_identities=input_identities,
        )
        if digest != record.get("sha256"):
            raise ManualRunLoadError(f"类别工作层 SHA256 不匹配，保留现有文件与审核记录: {class_path}")
        rebound["classes"][str(code)] = record
        _progress(
            progress,
            f"正在校验类别工作层 {index}/{count}",
            25 + 60 * index / count,
        )
    rebound["manual_only"] = True
    return rebound


def prepare_manual_run(
    run_directory,
    *,
    load_scope=LOAD_CLASSES,
    for_initialization=False,
    is_canceled: CancelCheck = None,
    progress: ProgressCallback = None,
) -> dict:
    """Validate/hash a copied Run without changing any file."""

    if load_scope not in LOAD_SCOPES:
        raise ManualRunLoadError(f"未知加载范围: {load_scope}")
    run_root = Path(run_directory).expanduser().resolve()
    if not run_root.is_dir():
        raise ManualRunLoadError(f"Run 文件夹不存在: {run_root}")
    _progress(progress, "正在读取 Run 合同", 2)
    input_identities: dict[str, tuple[object, object]] = {}
    spec_path = run_root / "run_spec.json"
    manifest_path = run_root / "run_manifest.json"
    original_spec = _read_json(
        spec_path,
        "run_spec.json",
        is_canceled=is_canceled,
        input_identities=input_identities,
    )
    manifest = _read_json(
        manifest_path,
        "run_manifest.json",
        is_canceled=is_canceled,
        input_identities=input_identities,
    )
    expected_spec_sha = str(manifest.get("run_spec_sha256") or "")
    if (
        not expected_spec_sha
        or _stable_sha256(
            spec_path,
            is_canceled=is_canceled,
            input_identities=input_identities,
        )
        != expected_spec_sha
    ):
        raise ManualRunLoadError("复制 Run 的 run_spec SHA256 与 run_manifest 不一致")
    run_id = str(original_spec.get("run_id") or "").strip()
    manifest_run_id = str(manifest.get("run_id") or "").strip()
    if not run_id or run_id != manifest_run_id:
        raise ManualRunLoadError(
            f"run_id 不一致: run_spec={run_id or '<empty>'}, "
            f"run_manifest={manifest_run_id or '<empty>'}"
        )
    if int(original_spec.get("schema_version") or 0) != 2:
        raise ManualRunLoadError("只支持 schema_version=2 的 Run")
    if str(manifest.get("status") or "") != "ready":
        raise ManualRunLoadError(
            f"Run 尚未完成，run_manifest 状态为 {manifest.get('status') or '<empty>'}"
        )

    _progress(progress, "正在核对 Fusion 结果", 15)
    old_root_value = str(original_spec.get("run_dir") or "").strip()
    old_root = Path(old_root_value).expanduser() if old_root_value else None
    has_workspace = (run_root / "classes" / "workspace.json").is_file()
    if not has_workspace and not for_initialization:
        raise ManualRunLoadError("Run 尚未初始化 classes，请到识别结果区初始化分类图层")
    if not has_workspace and (run_root / "classes").is_dir():
        if any((run_root / "classes").glob("class_*.gpkg")):
            raise ManualRunLoadError("classes 文件不完整：缺少 workspace.json；现有人工文件保持不变")
    fusion_streams = []
    result_streams = []
    for stream in manifest.get("streams") or []:
        _check_canceled(is_canceled)
        if stream.get("status") == "ready" and (
            stream.get("kind") == "fusion"
            or (load_scope == LOAD_ALL and stream.get("kind") == "model")
        ):
            rebound = _rebind_stream(
                    stream,
                    old_root,
                    run_root,
                    run_id,
                    require_semantic=for_initialization or load_scope == LOAD_ALL,
                )
            result_streams.append(rebound)
            if stream.get("kind") == "fusion":
                fusion_streams.append(rebound)
            if load_scope == LOAD_ALL:
                _validate_result_files(rebound, is_canceled=is_canceled, input_identities=input_identities)
            review_path = str(rebound.get("review_polygons") or "")
            if review_path:
                input_identities[str(Path(review_path).resolve())] = file_identity(
                    review_path
                )
    if not fusion_streams:
        raise ManualRunLoadError("run_manifest.json 中没有 ready Fusion 结果")
    stream_ids = [stream.get("stream_id") for stream in result_streams]
    if len(set(stream_ids)) != len(stream_ids):
        raise ManualRunLoadError("Run 中存在重复的结果流身份，未加载")
    if load_scope == LOAD_ALL:
        model_ids = {item.get("model_id") for item in result_streams if item.get("kind") == "model"}
        expected = {item.get("model_id") for item in original_spec.get("models") or []}
        if not model_ids or not expected.issubset(model_ids):
            raise ManualRunLoadError("Models 结果不完整，不能按 fusion、models、classes 全量加载")

    spec = copy.deepcopy(original_spec)
    spec["run_dir"] = str(run_root)
    # Legacy provenance flag: portable workspace validation, never a SAM switch.
    spec["manual_only"] = not for_initialization
    spec["workflow_mode"] = "run_refinement"
    spec["load_scope"] = load_scope
    if old_root is None or old_root.resolve() != run_root:
        spec["accepted_target_gpkg"] = str(run_root / "accepted_labels.gpkg")
    raster = dict(spec.get("raster") or {})
    raster_path = _rebound_candidate(raster.get("path"), old_root, run_root, run_id)
    if raster_path is not None and raster_path.is_file():
        raster["path"] = str(raster_path)
    spec["raster"] = raster
    snapshot = _rebound_candidate(
        original_spec.get("accepted_gpkg"), old_root, run_root, run_id
    )
    if snapshot is not None and snapshot.is_file():
        _progress(progress, "正在校验 accepted 快照", 20)
        spec["accepted_gpkg"] = str(snapshot)
        spec["accepted_gpkg_sha256"] = _stable_sha256(
            snapshot,
            is_canceled=is_canceled,
            input_identities=input_identities,
        )
    else:
        spec["accepted_gpkg"] = ""
        spec["accepted_gpkg_sha256"] = ""
    fusion = dict(spec.get("fusion") or {})
    fusion_snapshot = run_root / "fusion_profile_snapshot.json"
    if fusion_snapshot.is_file():
        fusion["snapshot_path"] = str(fusion_snapshot.resolve())
        input_identities[str(fusion_snapshot.resolve())] = file_identity(
            fusion_snapshot
        )
    spec["fusion"] = fusion

    workspace = _prepare_workspace(
        run_root,
        run_id,
        fusion_streams,
        old_root=old_root,
        is_canceled=is_canceled,
        progress=progress,
        input_identities=input_identities,
    )
    spec.pop("run_spec_content_sha256", None)
    accepted_write_spec = run_root / "classes" / "accepted_write_run_spec.json"
    accepted_write_manifest = run_root / "classes" / "accepted_write_run_manifest.json"
    spec["accepted_write_manifest"] = str(accepted_write_manifest)
    result = {
        "run_id": run_id,
        "run_dir": str(run_root),
        "run_spec": str(spec_path),
        "run_manifest": str(manifest_path),
        "run_report": str(run_root / "logs" / "run_report.json"),
        "success": True,
        "stopped": False,
        "status": "ready",
        "streams": result_streams,
        "ready_streams": result_streams,
        "failed_streams": [],
        "manual_only": not for_initialization,
        "load_scope": load_scope,
        "portable_classes_only": bool(
            workspace and workspace.get("portable_classes_only")
        ),
        "error": "",
    }
    bundle = {
        "run_root": str(run_root),
        "result": result,
        "run_spec": spec,
        "workspace": workspace,
        "workspace_path": str(run_root / "classes" / "workspace.json"),
        "_accepted_write_spec_path": str(accepted_write_spec),
        "_accepted_write_manifest_path": str(accepted_write_manifest),
        "_source_manifest": copy.deepcopy(manifest),
        "for_initialization": for_initialization,
    }
    _check_canceled(is_canceled)
    if any(
        file_identity(path) != identity for path, identity in input_identities.items()
    ):
        raise ManualRunLoadError("校验期间 Run 输入文件发生变化，结果未发布")
    bundle["_input_identities"] = input_identities
    _progress(progress, "Run 校验完成，等待发布", 95)
    return bundle


def manual_run_inputs_unchanged(bundle: dict) -> bool:
    return all(
        file_identity(path) == identity
        for path, identity in (bundle.get("_input_identities") or {}).items()
    )


def publish_manual_run_bundle(bundle: dict) -> dict:
    """Publish prepared metadata with the accepted manifest as commit marker."""

    if not manual_run_inputs_unchanged(bundle):
        raise ManualRunLoadError("后台校验后 Run 文件发生变化，结果未发布")
    if bundle.get("for_initialization"):
        return bundle
    persist_rebound_workspace(bundle)
    spec = copy.deepcopy(bundle["run_spec"])
    accepted_write_spec = Path(bundle["_accepted_write_spec_path"])
    accepted_write_manifest = Path(bundle["_accepted_write_manifest_path"])
    atomic_write_json(accepted_write_spec, spec)
    manual_manifest = copy.deepcopy(bundle["_source_manifest"])
    manual_manifest["run_spec"] = str(accepted_write_spec)
    manual_manifest["run_spec_sha256"] = sha256_file(accepted_write_spec)
    manual_manifest["streams"] = copy.deepcopy(bundle["result"]["ready_streams"])
    atomic_write_json(accepted_write_manifest, manual_manifest)
    return bundle


def load_manual_run(run_directory, *, load_scope=LOAD_CLASSES) -> dict:
    """Synchronous, standalone loader retained for non-GUI callers."""

    return publish_manual_run_bundle(prepare_manual_run(run_directory, load_scope=load_scope))


def persist_rebound_workspace(bundle):
    """Persist already-hashed portable paths without doing further file scans."""

    workspace = bundle.get("workspace")
    if workspace is None:
        return
    workspace = copy.deepcopy(workspace)
    bundle["workspace"] = workspace
    path = Path(bundle["workspace_path"])
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(workspace, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
