from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

from labeling_tool.core.deployment_contract import verify_project_runtime


ROOT = Path(__file__).resolve().parents[1]
GIT_SHA = "a" * 40
SHARED_NAMES = ("run_spec.py", "run_state_db.py", "ownership_neighbors.py")


def _run(command, *, env=None, check=True):
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"command failed ({result.returncode}): {command}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def _environment(fake_qgis: Path):
    value = dict(os.environ)
    value.update(
        {
            "LOESS_GIT_SHA": GIT_SHA,
            "PYTHON_BIN": sys.executable,
            "QGIS_PROCESS_EXE": str(fake_qgis),
        }
    )
    return value


def _fake_qgis(path: Path):
    path.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' 'QGIS 4.2.0-Test'\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_separate_plugin_and_project_deployments_share_exact_runtime(tmp_path):
    fake_qgis = tmp_path / "qgis_process"
    _fake_qgis(fake_qgis)
    env = _environment(fake_qgis)
    plugin_root = tmp_path / "plugins"
    project_root = tmp_path / "project"

    _run(
        [
            str(ROOT / "bash" / "install_plugin.sh"),
            "--platform",
            "macos",
            "--profile",
            "test-profile",
            "--plugin-dir",
            str(plugin_root),
        ],
        env=env,
    )
    installed_plugin = plugin_root / "labeling_tool"
    assert (installed_plugin / "deployment_manifest.json").is_file()
    assert not (plugin_root / "inference_scripts").exists()
    assert not (plugin_root / "weights").exists()

    _run(
        [
            str(ROOT / "bash" / "init_project.sh"),
            "--platform",
            "macos",
            "--project-root",
            str(project_root),
        ],
        env=env,
    )

    expected_directories = (
        "inference_scripts",
        "runtime/labeling_tool/core",
        "weights",
        "input/rasters",
        "input/ranges",
        "qgis",
        "output/runs",
        "output/cache",
    )
    for relative in expected_directories:
        assert (project_root / relative).is_dir()
    assert not (project_root / "accepted_labels.gpkg").exists()
    assert not list(project_root.rglob("__pycache__"))
    assert not list(project_root.rglob("*.pyc"))

    plugin_manifest = json.loads(
        (installed_plugin / "deployment_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    project_manifest = json.loads(
        (project_root / "project_manifest.json").read_text(encoding="utf-8")
    )
    assert plugin_manifest["git_sha"] == project_manifest["git_sha"] == GIT_SHA
    assert (
        plugin_manifest["shared_runtime"]
        == project_manifest["shared_runtime"]
    )
    assert plugin_manifest["deployment_kind"] == "qgis_plugin"
    assert project_manifest["deployment_kind"] == "loess_project"

    for name in SHARED_NAMES:
        source = ROOT / "qgis_plugins" / "labeling_tool" / "core" / name
        deployed = (
            project_root / "runtime" / "labeling_tool" / "core" / name
        )
        assert deployed.read_bytes() == source.read_bytes()

    check = verify_project_runtime(
        project_root / "inference_scripts",
        plugin_root=installed_plugin,
    )
    assert check["status"] == "ready"

    import_check = _run(
        [
            sys.executable,
            "-c",
            (
                "from labeling_tool.core.run_state_db import SCHEMA_VERSION;"
                "from labeling_tool.core.run_spec import CLASS_ORDER;"
                "from labeling_tool.core.ownership_neighbors import "
                "ownership_neighbors;"
                "print(SCHEMA_VERSION, CLASS_ORDER[0], ownership_neighbors([]))"
            ),
        ],
        env={**env, "PYTHONPATH": str(project_root / "runtime")},
    )
    assert import_check.stdout.strip() == "2 12 []"


def test_plugin_installer_rejects_destinations_overlapping_source_tree(tmp_path):
    fake_qgis = tmp_path / "qgis_process"
    _fake_qgis(fake_qgis)
    env = _environment(fake_qgis)
    plugin_source = ROOT / "qgis_plugins" / "labeling_tool"
    metadata_before = (plugin_source / "metadata.txt").read_bytes()

    linked_plugin_root = tmp_path / "linked-plugins"
    linked_plugin_root.mkdir()
    (linked_plugin_root / "labeling_tool").symlink_to(
        plugin_source,
        target_is_directory=True,
    )
    unsafe_roots = (
        ROOT / "qgis_plugins",
        ROOT / "tests" / ".." / "qgis_plugins",
        ROOT / ".unsafe-plugin-root",
        linked_plugin_root,
    )

    for plugin_root in unsafe_roots:
        result = _run(
            [
                str(ROOT / "bash" / "install_plugin.sh"),
                "--platform",
                "macos",
                "--profile",
                "overlap-audit",
                "--plugin-dir",
                str(plugin_root),
                "--check-only",
            ],
            env=env,
            check=False,
        )
        assert result.returncode != 0
        assert "overlapping source repository" in result.stderr

    assert (plugin_source / "metadata.txt").read_bytes() == metadata_before
    assert not list((ROOT / "qgis_plugins").glob(".labeling_tool.*"))
    assert not (ROOT / ".unsafe-plugin-root").exists()


def test_project_update_preserves_user_data_and_restores_managed_code(tmp_path):
    fake_qgis = tmp_path / "qgis_process"
    _fake_qgis(fake_qgis)
    env = _environment(fake_qgis)
    project_root = tmp_path / "project"
    command = [
        str(ROOT / "bash" / "init_project.sh"),
        "--platform",
        "macos",
        "--project-root",
        str(project_root),
    ]
    _run(command, env=env)

    user_files = {
        "weights/user-model.bin": b"weight-data",
        "input/rasters/source.tif": b"raster-data",
        "input/ranges/range.shp": b"shape-data",
        "qgis/user.qgz": b"qgis-project",
        "output/runs/old-run.txt": b"run-data",
    }
    for relative, content in user_files.items():
        path = project_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    weights_readme = project_root / "weights" / "README_WEIGHTS.md"
    weights_readme.write_text("user notes\n", encoding="utf-8")
    managed = (
        project_root / "runtime" / "labeling_tool" / "core" / "run_spec.py"
    )
    managed.write_text("corrupted\n", encoding="utf-8")

    invalid_check = _run(
        [*command, "--check-only"],
        env=env,
        check=False,
    )
    assert invalid_check.returncode != 0
    assert "Shared runtime project SHA256 mismatch" in invalid_check.stderr

    _run(command, env=env)

    for relative, content in user_files.items():
        assert (project_root / relative).read_bytes() == content
    assert weights_readme.read_text(encoding="utf-8") == "user notes\n"
    assert managed.read_bytes() == (
        ROOT / "qgis_plugins" / "labeling_tool" / "core" / "run_spec.py"
    ).read_bytes()
    assert not list(project_root.glob(".*.old.*"))
    assert not list(project_root.glob(".loess-project-init.*"))


def test_project_check_only_is_non_mutating_and_assets_are_explicit(tmp_path):
    fake_qgis = tmp_path / "qgis_process"
    _fake_qgis(fake_qgis)
    env = _environment(fake_qgis)
    project_root = tmp_path / "project"

    check_only = _run(
        [
            str(ROOT / "bash" / "init_project.sh"),
            "--platform",
            "macos",
            "--project-root",
            str(project_root),
            "--check-only",
        ],
        env=env,
    )
    assert "project files were not changed" in check_only.stdout
    assert not project_root.exists()

    _run(
        [
            str(ROOT / "bash" / "init_project.sh"),
            "--platform",
            "macos",
            "--project-root",
            str(project_root),
        ],
        env=env,
    )
    asset_check = _run(
        [
            str(ROOT / "bash" / "init_project.sh"),
            "--platform",
            "macos",
            "--project-root",
            str(project_root),
            "--check-only",
            "--check-assets",
        ],
        env=env,
        check=False,
    )
    assert asset_check.returncode == 3
    assert "Missing required assets" in asset_check.stderr


def test_runtime_contract_rejects_changed_project_copy(tmp_path):
    fake_qgis = tmp_path / "qgis_process"
    _fake_qgis(fake_qgis)
    env = _environment(fake_qgis)
    plugin_root = tmp_path / "plugins"
    project_root = tmp_path / "project"
    _run(
        [
            str(ROOT / "bash" / "install_plugin.sh"),
            "--platform",
            "macos",
            "--profile",
            "test-profile",
            "--plugin-dir",
            str(plugin_root),
        ],
        env=env,
    )
    _run(
        [
            str(ROOT / "bash" / "init_project.sh"),
            "--platform",
            "macos",
            "--project-root",
            str(project_root),
        ],
        env=env,
    )
    changed = (
        project_root
        / "runtime"
        / "labeling_tool"
        / "core"
        / "ownership_neighbors.py"
    )
    changed.write_text(changed.read_text(encoding="utf-8") + "\n# changed\n")
    check = verify_project_runtime(
        project_root / "inference_scripts",
        plugin_root=plugin_root / "labeling_tool",
    )
    assert check["status"] == "error"
    assert "项目共享模块已改变" in check["message"]
