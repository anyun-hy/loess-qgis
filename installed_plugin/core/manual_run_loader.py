"""Load a copied ready Run for portable, manual-only class refinement."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

from .run_spec import CLASS_ORDER


class ManualRunLoadError(RuntimeError):
    pass


def _read_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise ManualRunLoadError(f"缺少 {label}: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManualRunLoadError(f"无法读取 {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManualRunLoadError(f"{label} 必须是 JSON 对象: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rebound_candidate(raw_path, old_root: Path | None, new_root: Path, run_id: str):
    raw = str(raw_path or "").strip()
    if not raw:
        return None
    source = Path(raw).expanduser()
    if not source.is_absolute():
        return (new_root / source).resolve()
    if old_root is not None:
        try:
            return (new_root / source.relative_to(old_root)).resolve()
        except ValueError:
            pass
    parts = source.parts
    if run_id in parts:
        index = len(parts) - 1 - list(reversed(parts)).index(run_id)
        return (new_root / Path(*parts[index + 1 :])).resolve()
    return None


def _rebind_stream(stream, old_root, run_root, run_id):
    rebound = copy.deepcopy(stream)
    stream_id = str(rebound.get("stream_id") or "")
    profile_id = str(rebound.get("fusion_profile_id") or "")
    if not profile_id and stream_id.startswith("fusion:"):
        profile_id = stream_id.split(":", 1)[1]
    if not profile_id:
        raise ManualRunLoadError(f"Fusion 流缺少 profile_id: {stream_id or '<unknown>'}")

    expected_dir = (run_root / "fusion" / profile_id).resolve()
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

    semantic = Path(str(paths.get("semantic_polygons") or ""))
    if not semantic.is_file():
        fallback = expected_dir / "semantic_polygons.gpkg"
        if fallback.is_file():
            semantic = fallback.resolve()
            paths["semantic_polygons"] = str(semantic)
        else:
            raise ManualRunLoadError(
                f"Fusion {stream_id} 缺少 semantic_polygons.gpkg: {fallback}"
            )
    rebound["paths"] = paths
    rebound["fusion_profile_id"] = profile_id
    rebound["review_polygons"] = str(semantic.resolve())
    rebound["review_layer_name"] = "semantic_polygons"
    rebound["manual_validated"] = False
    return rebound


def _prepare_workspace(run_root, run_id, fusion_streams):
    workspace_path = run_root / "classes" / "workspace.json"
    if not workspace_path.is_file():
        return None
    workspace = _read_json(workspace_path, "classes/workspace.json")
    if str(workspace.get("run_id") or "") != run_id:
        raise ManualRunLoadError("classes/workspace.json 的 run_id 与 Run 不一致")
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
    baseline_path = Path(stream_map[baseline_id]["review_polygons"]).resolve()
    baseline_sha = _sha256(baseline_path)
    rebound = copy.deepcopy(workspace)
    rebound["baseline_source_path"] = str(baseline_path)
    rebound["formal_path"] = str(baseline_path)
    rebound["baseline_source_sha256"] = baseline_sha
    rebound["formal_sha256"] = baseline_sha
    report_path = Path(
        str((stream_map[baseline_id].get("paths") or {}).get("boundary_fitting_report") or "")
    )
    if report_path.is_file():
        rebound["boundary_report_path"] = str(report_path.resolve())
        rebound["boundary_report_sha256"] = _sha256(report_path)
    else:
        rebound["boundary_report_path"] = ""
        rebound["boundary_report_sha256"] = ""

    for code in CLASS_ORDER:
        record = dict(classes[str(code)])
        class_path = (run_root / "classes" / f"class_{code}.gpkg").resolve()
        if not class_path.is_file():
            raise ManualRunLoadError(f"缺少类别工作层: {class_path}")
        actual_sha = _sha256(class_path)
        record["path"] = str(class_path)
        record["sha256"] = actual_sha
        rebound["classes"][str(code)] = record
    rebound["manual_only"] = True
    return rebound


def load_manual_run(run_directory) -> dict:
    run_root = Path(run_directory).expanduser().resolve()
    if not run_root.is_dir():
        raise ManualRunLoadError(f"Run 文件夹不存在: {run_root}")
    spec_path = run_root / "run_spec.json"
    manifest_path = run_root / "run_manifest.json"
    original_spec = _read_json(spec_path, "run_spec.json")
    manifest = _read_json(manifest_path, "run_manifest.json")
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

    old_root_value = str(original_spec.get("run_dir") or "").strip()
    old_root = Path(old_root_value).expanduser() if old_root_value else None
    fusion_streams = []
    for stream in manifest.get("streams") or []:
        if stream.get("kind") == "fusion" and stream.get("status") == "ready":
            fusion_streams.append(
                _rebind_stream(stream, old_root, run_root, run_id)
            )
    if not fusion_streams:
        raise ManualRunLoadError("run_manifest.json 中没有 ready Fusion 结果")

    spec = copy.deepcopy(original_spec)
    spec["run_dir"] = str(run_root)
    spec["manual_only"] = True
    spec["workflow_mode"] = "manual_run_copy"
    spec["accepted_gpkg"] = str(run_root / "accepted_labels.gpkg")
    state_db = run_root / "state.sqlite"
    if state_db.is_file():
        spec["state_db"] = str(state_db)
    fusion = dict(spec.get("fusion") or {})
    snapshot = run_root / "fusion_profile_snapshot.json"
    if snapshot.is_file():
        fusion["snapshot_path"] = str(snapshot.resolve())
    spec["fusion"] = fusion

    workspace = _prepare_workspace(run_root, run_id, fusion_streams)
    result = {
        "run_id": run_id,
        "run_dir": str(run_root),
        "run_spec": str(spec_path),
        "run_manifest": str(manifest_path),
        "run_report": str(run_root / "logs" / "run_report.json"),
        "success": True,
        "stopped": False,
        "status": "ready",
        "streams": fusion_streams,
        "ready_streams": fusion_streams,
        "failed_streams": [],
        "manual_only": True,
        "error": "",
    }
    return {
        "run_root": str(run_root),
        "result": result,
        "run_spec": spec,
        "workspace": workspace,
        "workspace_path": str(run_root / "classes" / "workspace.json"),
    }


def persist_rebound_workspace(bundle):
    workspace = bundle.get("workspace")
    if workspace is None:
        return
    workspace = copy.deepcopy(workspace)
    baseline_path = Path(workspace["baseline_source_path"])
    formal_path = Path(workspace["formal_path"])
    workspace["baseline_source_sha256"] = _sha256(baseline_path)
    workspace["formal_sha256"] = _sha256(formal_path)
    report_path = Path(str(workspace.get("boundary_report_path") or ""))
    if report_path.is_file():
        workspace["boundary_report_sha256"] = _sha256(report_path)
    for record in (workspace.get("classes") or {}).values():
        class_path = Path(record["path"])
        record["sha256"] = _sha256(class_path)
    bundle["workspace"] = workspace
    path = Path(bundle["workspace_path"])
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(workspace, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
