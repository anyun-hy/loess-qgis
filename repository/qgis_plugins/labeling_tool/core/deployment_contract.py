"""Verify that the plugin and initialized project share one runtime contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SHARED_RUNTIME_FILES = {
    "qgis_plugins/labeling_tool/core/run_spec.py":
        "runtime/labeling_tool/core/run_spec.py",
    "qgis_plugins/labeling_tool/core/run_state_db.py":
        "runtime/labeling_tool/core/run_state_db.py",
    "qgis_plugins/labeling_tool/core/ownership_neighbors.py":
        "runtime/labeling_tool/core/ownership_neighbors.py",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path, kind: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取 {path}: {error}") from error
    if not isinstance(value, Mapping) or value.get("deployment_kind") != kind:
        raise ValueError(f"{path} 不是 {kind} 清单")
    return value


def _runtime_block(manifest: Mapping[str, Any], path: Path) -> Mapping[str, Any]:
    value = manifest.get("shared_runtime")
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} 缺少 shared_runtime")
    files = value.get("files")
    digest = str(value.get("sha256") or "")
    if not isinstance(files, Mapping) or set(files) != set(SHARED_RUNTIME_FILES):
        raise ValueError(f"{path} 的共享模块清单不完整")
    if len(digest) != 64:
        raise ValueError(f"{path} 的共享模块聚合 SHA256 无效")
    return value


def verify_project_runtime(
    scripts_dir: str | Path,
    *,
    plugin_root: str | Path | None = None,
) -> dict[str, str]:
    """Return one QGIS environment-check item for the deployment relationship."""

    scripts = Path(scripts_dir).expanduser().resolve()
    project_root = scripts.parent
    plugin = (
        Path(plugin_root).expanduser().resolve()
        if plugin_root is not None
        else Path(__file__).resolve().parents[1]
    )
    project_manifest_path = project_root / "project_manifest.json"
    plugin_manifest_path = plugin / "deployment_manifest.json"

    try:
        project_manifest = _read_manifest(
            project_manifest_path, "loess_project"
        )
        plugin_manifest = _read_manifest(
            plugin_manifest_path, "qgis_plugin"
        )
        project_runtime = _runtime_block(
            project_manifest, project_manifest_path
        )
        plugin_runtime = _runtime_block(plugin_manifest, plugin_manifest_path)

        project_git = str(project_manifest.get("git_sha") or "")
        plugin_git = str(plugin_manifest.get("git_sha") or "")
        if project_git != plugin_git:
            raise ValueError(
                "插件与项目来自不同 Git 提交: "
                f"plugin={plugin_git or '<missing>'}, "
                f"project={project_git or '<missing>'}"
            )
        if project_runtime.get("sha256") != plugin_runtime.get("sha256"):
            raise ValueError("插件与项目的共享模块聚合 SHA256 不一致")
        if dict(project_runtime["files"]) != dict(plugin_runtime["files"]):
            raise ValueError("插件与项目的共享模块逐文件 SHA256 不一致")

        expected = dict(project_runtime["files"])
        for canonical, project_relative in SHARED_RUNTIME_FILES.items():
            project_file = project_root / project_relative
            plugin_file = plugin / canonical.removeprefix(
                "qgis_plugins/labeling_tool/"
            )
            if not project_file.is_file():
                raise ValueError(f"项目共享模块缺失: {project_file}")
            if not plugin_file.is_file():
                raise ValueError(f"插件共享模块缺失: {plugin_file}")
            expected_digest = str(expected[canonical])
            if _sha256(project_file) != expected_digest:
                raise ValueError(f"项目共享模块已改变: {project_file}")
            if _sha256(plugin_file) != expected_digest:
                raise ValueError(f"插件共享模块已改变: {plugin_file}")
    except (ValueError, OSError, KeyError, TypeError) as error:
        return {
            "id": "shared_runtime_contract",
            "status": "error",
            "value": str(project_root),
            "source": f"{plugin_manifest_path} / {project_manifest_path}",
            "message": str(error),
            "fix": "从同一 Git 提交重新运行 install_plugin.sh 和 init_project.sh",
        }

    return {
        "id": "shared_runtime_contract",
        "status": "ready",
        "value": str(project_runtime["sha256"]),
        "source": f"{plugin_manifest_path} / {project_manifest_path}",
        "message": f"插件与项目共享模块一致，Git SHA={project_git}",
        "fix": "",
    }
