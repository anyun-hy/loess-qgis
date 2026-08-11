"""Verify that the plugin and initialized project share one runtime contract."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping


SHARED_RUNTIME_FILES = {
    "qgis_plugins/labeling_tool/core/run_spec.py":
        "runtime/labeling_tool/core/run_spec.py",
    "qgis_plugins/labeling_tool/core/run_state_db.py":
        "runtime/labeling_tool/core/run_state_db.py",
    "qgis_plugins/labeling_tool/core/postgres_state.py":
        "runtime/labeling_tool/core/postgres_state.py",
    "qgis_plugins/labeling_tool/core/ownership_neighbors.py":
        "runtime/labeling_tool/core/ownership_neighbors.py",
}
SUPPORTED_PLATFORMS = frozenset({"ubuntu", "macos"})
DEPLOYMENT_FINGERPRINT_FILES = (
    "../project_manifest.json",
    "../.loess-project-id",
    "../runtime/loess_launcher.sh",
)
LAUNCHER_RELATIVE_PATH = "runtime/loess_launcher.sh"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_CONDA_ENV_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path, kind: str) -> Mapping[str, Any]:
    if path.is_symlink():
        raise ValueError(f"{path} 不能是符号链接")
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
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{path} 的共享模块聚合 SHA256 无效")
    return value


def _source_block(
    manifest: Mapping[str, Any],
    path: Path,
) -> Mapping[str, Any]:
    value = manifest.get("source")
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} 缺少 source 来源信息")
    if value.get("schema_version") != 1:
        raise ValueError(f"{path} 的 source schema 无效")
    if value.get("kind") not in {"git_worktree", "release_archive"}:
        raise ValueError(f"{path} 的 source.kind 无效")
    if not isinstance(value.get("git_dirty"), bool):
        raise ValueError(f"{path} 的 source.git_dirty 无效")
    if not _GIT_SHA_RE.fullmatch(str(value.get("git_sha") or "")):
        raise ValueError(f"{path} 的 source.git_sha 无效")
    if not _SHA256_RE.fullmatch(
        str(value.get("source_bundle_sha256") or "")
    ):
        raise ValueError(f"{path} 的源码包 SHA256 无效")
    try:
        file_count = int(value.get("source_file_count"))
    except (TypeError, ValueError):
        file_count = 0
    if file_count < 1:
        raise ValueError(f"{path} 的 source_file_count 无效")
    if str(manifest.get("git_sha") or "") != str(value.get("git_sha") or ""):
        raise ValueError(f"{path} 的 Git SHA 与 source.git_sha 不一致")
    return value


def _validate_project_identity(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    project_root: Path,
) -> str:
    if manifest.get("schema_version") != 2:
        raise ValueError(f"{manifest_path} 必须使用项目清单 schema 2")
    project_id = str(manifest.get("project_id") or "")
    try:
        parsed = uuid.UUID(project_id)
    except ValueError as error:
        raise ValueError(f"{manifest_path} 的 project_id 无效") from error
    if str(parsed) != project_id:
        raise ValueError(f"{manifest_path} 的 project_id 不是规范 UUID")
    marker_path = project_root / ".loess-project-id"
    marker = _read_manifest(marker_path, "loess_project_identity")
    if (
        marker.get("schema_version") != 1
        or str(marker.get("project_id") or "") != project_id
    ):
        raise ValueError(f"{marker_path} 与项目清单身份不一致")
    if manifest.get("managed_paths") != ["inference_scripts", "runtime"]:
        raise ValueError(f"{manifest_path} 的 managed_paths 无效")
    return project_id


def _inventory_block(
    manifest: Mapping[str, Any],
    key: str,
    manifest_path: Path,
) -> dict[str, str]:
    value = manifest.get(key)
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{manifest_path} 缺少完整的 {key}")
    normalized: dict[str, str] = {}
    for raw_name, raw_digest in value.items():
        name = str(raw_name)
        relative = PurePosixPath(name)
        if (
            not name
            or relative.is_absolute()
            or "\\" in name
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError(f"{manifest_path} 的 {key} 路径无效: {name!r}")
        digest = str(raw_digest)
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError(
                f"{manifest_path} 的 {key} SHA256 无效: {name}"
            )
        normalized[name] = digest
    return normalized


def _ignored_inventory_path(relative: str, excluded: set[str]) -> bool:
    parts = PurePosixPath(relative).parts
    name = parts[-1] if parts else ""
    return (
        relative in excluded
        or "__pycache__" in parts
        or name == ".DS_Store"
        or name.startswith("._")
        or name.endswith(".pyc")
    )


def _validate_inventory(
    root: Path,
    expected: Mapping[str, str],
    *,
    label: str,
    excluded: set[str] | None = None,
) -> None:
    ignored = set(excluded or ())
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"{label}目录缺失或不安全: {root}")
    actual: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if _ignored_inventory_path(relative, ignored):
            continue
        if path.is_symlink():
            raise ValueError(f"{label}包含符号链接: {path}")
        if path.is_file():
            actual[relative] = path

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        details = []
        if missing:
            details.append("缺失=" + ",".join(missing[:8]))
        if extra:
            details.append("未登记=" + ",".join(extra[:8]))
        raise ValueError(f"{label}文件清单不一致: {'; '.join(details)}")

    for relative, expected_digest in sorted(expected.items()):
        path = actual[relative]
        if _sha256(path) != expected_digest:
            raise ValueError(f"{label}文件已改变: {path}")
        if path.suffix == ".sh" and not os.access(path, os.X_OK):
            raise ValueError(f"{label}Shell 入口不可执行: {path}")


def _validate_launcher(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    project_root: Path,
) -> Mapping[str, Any]:
    launcher = manifest.get("launcher")
    if not isinstance(launcher, Mapping):
        raise ValueError(f"{manifest_path} 缺少 launcher")
    if str(launcher.get("path") or "") != LAUNCHER_RELATIVE_PATH:
        raise ValueError(f"{manifest_path} 的 launcher.path 无效")
    digest = str(launcher.get("sha256") or "")
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{manifest_path} 的 launcher.sha256 无效")
    environment = str(launcher.get("conda_environment") or "")
    if not _CONDA_ENV_RE.fullmatch(environment):
        raise ValueError(f"{manifest_path} 的 Conda 环境名无效")
    executable = launcher.get("conda_executable")
    if not isinstance(executable, str):
        raise ValueError(f"{manifest_path} 的 Conda 可执行文件配置无效")
    if executable:
        executable_path = Path(executable).expanduser()
        if (
            not executable_path.is_absolute()
            or not executable_path.is_file()
            or executable_path.is_symlink()
            or not os.access(executable_path, os.X_OK)
        ):
            raise ValueError(
                f"{manifest_path} 的 Conda 可执行文件不可用: {executable}"
            )

    launcher_path = project_root / LAUNCHER_RELATIVE_PATH
    if (
        not launcher_path.is_file()
        or launcher_path.is_symlink()
        or _sha256(launcher_path) != digest
    ):
        raise ValueError(f"项目启动配置缺失或已改变: {launcher_path}")
    return launcher


def deployment_fingerprint(scripts_dir: str | Path) -> str:
    """Fingerprint the immutable project deployment and persisted launcher."""

    scripts = Path(scripts_dir).expanduser().resolve()
    digest = hashlib.sha256()
    for relative in DEPLOYMENT_FINGERPRINT_FILES:
        path = scripts / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        if path.is_file() and not path.is_symlink():
            digest.update(_sha256(path).encode("ascii"))
        else:
            digest.update(b"<missing>")
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


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
        _validate_project_identity(
            project_manifest,
            project_manifest_path,
            project_root,
        )
        if plugin_manifest.get("schema_version") != 2:
            raise ValueError(f"{plugin_manifest_path} 必须使用插件清单 schema 2")
        declared_project_root = Path(
            str(project_manifest.get("project_root") or "")
        ).expanduser()
        if (
            not declared_project_root.is_absolute()
            or declared_project_root.resolve() != project_root
        ):
            raise ValueError(
                f"{project_manifest_path} 的 project_root 与实际目录不一致"
            )
        project_platform = str(project_manifest.get("platform") or "")
        plugin_platform = str(plugin_manifest.get("platform") or "")
        if project_platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"项目部署平台无效: {project_platform or '<missing>'}"
            )
        if plugin_platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"插件部署平台无效: {plugin_platform or '<missing>'}"
            )
        if project_platform != plugin_platform:
            raise ValueError(
                "插件与项目部署平台不一致: "
                f"plugin={plugin_platform}, project={project_platform}"
            )
        project_inference = _inventory_block(
            project_manifest, "inference_files", project_manifest_path
        )
        plugin_files = _inventory_block(
            plugin_manifest, "files", plugin_manifest_path
        )
        _validate_inventory(
            scripts,
            project_inference,
            label="项目 inference_scripts",
        )
        _validate_inventory(
            plugin,
            plugin_files,
            label="插件",
            excluded={"deployment_manifest.json"},
        )
        _validate_launcher(
            project_manifest,
            project_manifest_path,
            project_root,
        )
        project_runtime = _runtime_block(
            project_manifest, project_manifest_path
        )
        plugin_runtime = _runtime_block(plugin_manifest, plugin_manifest_path)
        project_source = _source_block(
            project_manifest, project_manifest_path
        )
        plugin_source = _source_block(plugin_manifest, plugin_manifest_path)

        project_git = str(project_manifest.get("git_sha") or "")
        plugin_git = str(plugin_manifest.get("git_sha") or "")
        if not _GIT_SHA_RE.fullmatch(project_git):
            raise ValueError(f"项目 Git SHA 无效: {project_git or '<missing>'}")
        if not _GIT_SHA_RE.fullmatch(plugin_git):
            raise ValueError(f"插件 Git SHA 无效: {plugin_git or '<missing>'}")
        if project_git != plugin_git:
            raise ValueError(
                "插件与项目来自不同 Git 提交: "
                f"plugin={plugin_git or '<missing>'}, "
                f"project={project_git or '<missing>'}"
            )
        if (
            project_source.get("source_bundle_sha256")
            != plugin_source.get("source_bundle_sha256")
        ):
            raise ValueError(
                "插件与项目的实际源码包 SHA256 不一致: "
                f"plugin={plugin_source.get('source_bundle_sha256')}, "
                f"project={project_source.get('source_bundle_sha256')}"
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
        "message": (
            f"插件与项目完整文件清单、启动配置及共享模块一致，"
            f"platform={project_platform}，Git SHA={project_git}"
        ),
        "fix": "",
    }
