from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

from check_environment import _fingerprint as environment_fingerprint
from labeling_tool.core.deployment_contract import (
    deployment_fingerprint,
    verify_project_runtime,
)


ROOT = Path(__file__).resolve().parents[1]
GIT_SHA = subprocess.run(
    ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
    text=True,
    stdout=subprocess.PIPE,
    check=True,
).stdout.strip()
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
            "LOESS_ALLOW_DIRTY": "1",
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


def _tree_snapshot(root: Path) -> dict[str, tuple[int, bytes]]:
    snapshot = {}
    if not root.exists():
        return snapshot
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(root).as_posix()
        snapshot[relative] = (stat.S_IMODE(path.stat().st_mode), path.read_bytes())
    return snapshot


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
    assert (project_root / "runtime" / "loess_launcher.sh").is_file()
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
        plugin_manifest["source"]["source_bundle_sha256"]
        == project_manifest["source"]["source_bundle_sha256"]
    )
    assert project_manifest["schema_version"] == 2
    assert (project_root / ".loess-project-id").is_file()
    assert len(project_manifest["required_assets"]) == 5
    assert all(
        len(asset["sha256"]) == 64
        for asset in project_manifest["required_assets"]
    )
    assert (
        plugin_manifest["shared_runtime"]
        == project_manifest["shared_runtime"]
    )
    assert plugin_manifest["deployment_kind"] == "qgis_plugin"
    assert project_manifest["deployment_kind"] == "loess_project"
    assert plugin_manifest["platform"] == project_manifest["platform"] == "macos"

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
    assert deployment_fingerprint(
        project_root / "inference_scripts"
    ) == environment_fingerprint(project_root / "inference_scripts")

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


def test_runtime_contract_rejects_invalid_or_mismatched_platforms(tmp_path):
    fake_qgis = tmp_path / "qgis_process"
    _fake_qgis(fake_qgis)
    env = _environment(fake_qgis)
    plugin_root = tmp_path / "plugins"
    installed_plugin = plugin_root / "labeling_tool"
    project_root = tmp_path / "project"
    plugin_manifest_path = installed_plugin / "deployment_manifest.json"
    project_manifest_path = project_root / "project_manifest.json"

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

    project_manifest = json.loads(
        project_manifest_path.read_text(encoding="utf-8")
    )
    project_manifest["platform"] = "ubuntu"
    project_manifest_path.write_text(
        json.dumps(project_manifest),
        encoding="utf-8",
    )
    check = verify_project_runtime(
        project_root / "inference_scripts",
        plugin_root=installed_plugin,
    )
    assert check["status"] == "error"
    assert "插件与项目部署平台不一致" in check["message"]

    project_manifest["platform"] = "windows"
    project_manifest_path.write_text(
        json.dumps(project_manifest),
        encoding="utf-8",
    )
    check = verify_project_runtime(
        project_root / "inference_scripts",
        plugin_root=installed_plugin,
    )
    assert check["status"] == "error"
    assert "项目部署平台无效" in check["message"]

    project_manifest["platform"] = "macos"
    project_manifest_path.write_text(
        json.dumps(project_manifest),
        encoding="utf-8",
    )
    plugin_manifest = json.loads(
        plugin_manifest_path.read_text(encoding="utf-8")
    )
    plugin_manifest["platform"] = "windows"
    plugin_manifest_path.write_text(
        json.dumps(plugin_manifest),
        encoding="utf-8",
    )
    check = verify_project_runtime(
        project_root / "inference_scripts",
        plugin_root=installed_plugin,
    )
    assert check["status"] == "error"
    assert "插件部署平台无效" in check["message"]


def test_runtime_contract_rejects_different_actual_source_bundles(tmp_path):
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
            "source-bundle-test",
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
    manifest_path = project_root / "project_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["source_bundle_sha256"] = "b" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = verify_project_runtime(
        project_root / "inference_scripts",
        plugin_root=plugin_root / "labeling_tool",
    )
    assert result["status"] == "error"
    assert "实际源码包 SHA256 不一致" in result["message"]


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


def test_plugin_install_rolls_back_every_signal_at_every_move_stage(tmp_path):
    fake_qgis = tmp_path / "qgis_process"
    _fake_qgis(fake_qgis)
    env = _environment(fake_qgis)
    plugin_root = tmp_path / "plugins"
    temporary_root = tmp_path / "temporary"
    temporary_root.mkdir()
    command = [
        str(ROOT / "bash" / "install_plugin.sh"),
        "--platform",
        "macos",
        "--profile",
        "rollback-test",
        "--plugin-dir",
        str(plugin_root),
    ]
    _run(command, env={**env, "TMPDIR": str(temporary_root)})
    installed = plugin_root / "labeling_tool"
    baseline = _tree_snapshot(installed)

    stages = (
        "staged_destination",
        "previous_installation_moved",
        "new_installation_moved",
        "installation_verified",
    )
    for signal_name in ("INT", "TERM", "HUP"):
        for stage in stages:
            result = _run(
                command,
                env={
                    **env,
                    "TMPDIR": str(temporary_root),
                    "LOESS_TEST_SIGNAL": signal_name,
                    "LOESS_TEST_SIGNAL_AT": stage,
                },
                check=False,
            )
            assert result.returncode != 0, (signal_name, stage)
            assert _tree_snapshot(installed) == baseline, (signal_name, stage)
            assert not list(plugin_root.glob(".labeling_tool.*"))
            assert not list(temporary_root.glob("loess-plugin-install.*"))


def test_project_update_rolls_back_every_signal_at_every_move_stage(tmp_path):
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
    user_file = project_root / "output" / "runs" / "preserved.txt"
    user_file.write_text("user-data\n", encoding="utf-8")
    baseline = _tree_snapshot(project_root)

    stages = (
        "previous_inference_moved",
        "previous_runtime_moved",
        "previous_manifest_moved",
        "new_inference_moved",
        "new_runtime_moved",
        "new_manifest_moved",
        "identity_installed",
        "installation_verified",
    )
    for signal_name in ("INT", "TERM", "HUP"):
        for stage in stages:
            result = _run(
                command,
                env={
                    **env,
                    "LOESS_TEST_SIGNAL": signal_name,
                    "LOESS_TEST_SIGNAL_AT": stage,
                },
                check=False,
            )
            assert result.returncode != 0, (signal_name, stage)
            assert _tree_snapshot(project_root) == baseline, (
                signal_name,
                stage,
            )
            assert not list(project_root.glob(".*.old.*"))
            assert not list(project_root.glob(".loess-project-init.*"))


def test_project_identity_rejects_forged_manifest_without_deleting_user_files(
    tmp_path,
):
    fake_qgis = tmp_path / "qgis_process"
    _fake_qgis(fake_qgis)
    env = _environment(fake_qgis)
    project_root = tmp_path / "project"
    inference_file = project_root / "inference_scripts" / "user.py"
    runtime_file = project_root / "runtime" / "user-runtime.txt"
    inference_file.parent.mkdir(parents=True)
    runtime_file.parent.mkdir(parents=True)
    inference_file.write_text("user inference\n", encoding="utf-8")
    runtime_file.write_text("user runtime\n", encoding="utf-8")
    (project_root / "project_manifest.json").write_text(
        json.dumps({"deployment_kind": "loess_project"}),
        encoding="utf-8",
    )

    result = _run(
        [
            str(ROOT / "bash" / "init_project.sh"),
            "--platform",
            "macos",
            "--project-root",
            str(project_root),
        ],
        env=env,
        check=False,
    )
    assert result.returncode != 0
    assert "field alone is not ownership evidence" in result.stderr
    assert inference_file.read_text(encoding="utf-8") == "user inference\n"
    assert runtime_file.read_text(encoding="utf-8") == "user runtime\n"
    assert not list(project_root.glob(".*.old.*"))


def test_moved_project_requires_explicit_rebind_and_preserves_identity(tmp_path):
    fake_qgis = tmp_path / "qgis_process"
    _fake_qgis(fake_qgis)
    env = _environment(fake_qgis)
    original = tmp_path / "original-project"
    moved = tmp_path / "moved-project"
    base_command = [
        str(ROOT / "bash" / "init_project.sh"),
        "--platform",
        "macos",
    ]
    _run([*base_command, "--project-root", str(original)], env=env)
    original_manifest = json.loads(
        (original / "project_manifest.json").read_text(encoding="utf-8")
    )
    project_id = original_manifest["project_id"]
    user_file = original / "weights" / "user-note.txt"
    user_file.write_text("preserve me\n", encoding="utf-8")
    shutil.move(original, moved)

    refused = _run(
        [*base_command, "--project-root", str(moved)],
        env=env,
        check=False,
    )
    assert refused.returncode != 0
    assert "--rebind-project-root" in refused.stderr

    _run(
        [
            *base_command,
            "--project-root",
            str(moved),
            "--rebind-project-root",
        ],
        env=env,
    )
    rebound = json.loads(
        (moved / "project_manifest.json").read_text(encoding="utf-8")
    )
    assert rebound["project_id"] == project_id
    assert Path(rebound["project_root"]) == moved.resolve()
    assert (moved / "weights" / "user-note.txt").read_text(
        encoding="utf-8"
    ) == "preserve me\n"


def test_valid_legacy_project_is_strictly_migrated_to_identity_schema(tmp_path):
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
    manifest_path = project_root / "project_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    manifest.pop("project_id")
    manifest.pop("source")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (project_root / ".loess-project-id").unlink()

    _run(command, env=env)
    migrated = json.loads(manifest_path.read_text(encoding="utf-8"))
    marker = json.loads(
        (project_root / ".loess-project-id").read_text(encoding="utf-8")
    )
    assert migrated["schema_version"] == 2
    assert migrated["project_id"] == marker["project_id"]


def test_source_provenance_rejects_dirty_git_and_tampered_archive(tmp_path):
    source_root = tmp_path / "source"
    for relative in ("bash", "inference_scripts", "qgis_plugins/labeling_tool"):
        source = ROOT / relative
        destination = source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )
    _run(["git", "init", "-q", str(source_root)])
    _run(["git", "-C", str(source_root), "config", "user.email", "test@example.invalid"])
    _run(["git", "-C", str(source_root), "config", "user.name", "Test"])
    _run(["git", "-C", str(source_root), "add", "."])
    _run(["git", "-C", str(source_root), "commit", "-qm", "fixture"])
    helper = source_root / "bash" / "deployment_source.py"

    clean = _run(
        [
            sys.executable,
            str(helper),
            "inspect",
            "--source-root",
            str(source_root),
        ]
    )
    clean_info = json.loads(clean.stdout)
    assert clean_info["git_dirty"] is False

    changed = source_root / "inference_scripts" / "config.yaml"
    changed.write_text(
        changed.read_text(encoding="utf-8") + "\n# changed\n",
        encoding="utf-8",
    )
    rejected = _run(
        [
            sys.executable,
            str(helper),
            "inspect",
            "--source-root",
            str(source_root),
        ],
        check=False,
    )
    assert rejected.returncode != 0
    assert "differ from Git HEAD" in rejected.stderr
    dirty = _run(
        [
            sys.executable,
            str(helper),
            "inspect",
            "--source-root",
            str(source_root),
            "--allow-dirty",
        ]
    )
    dirty_info = json.loads(dirty.stdout)
    assert dirty_info["git_dirty"] is True
    assert (
        dirty_info["source_bundle_sha256"]
        != clean_info["source_bundle_sha256"]
    )

    _run(["git", "-C", str(source_root), "restore", "inference_scripts/config.yaml"])
    archive_manifest = source_root / "source_manifest.json"
    _run(
        [
            sys.executable,
            str(helper),
            "create-archive-manifest",
            "--source-root",
            str(source_root),
            "--output",
            str(archive_manifest),
        ]
    )
    shutil.rmtree(source_root / ".git")
    archive = _run(
        [
            sys.executable,
            str(helper),
            "inspect",
            "--source-root",
            str(source_root),
        ]
    )
    assert json.loads(archive.stdout)["kind"] == "release_archive"
    changed.write_text(
        changed.read_text(encoding="utf-8") + "\n# archive tamper\n",
        encoding="utf-8",
    )
    tampered = _run(
        [
            sys.executable,
            str(helper),
            "inspect",
            "--source-root",
            str(source_root),
        ],
        check=False,
    )
    assert tampered.returncode != 0
    assert "source bundle SHA256 mismatch" in tampered.stderr


def test_project_persists_custom_conda_launcher_across_updates(tmp_path):
    fake_qgis = tmp_path / "qgis_process"
    _fake_qgis(fake_qgis)
    fake_conda = tmp_path / "custom conda" / "bin" / "conda"
    fake_conda.parent.mkdir(parents=True)
    fake_conda.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_conda.chmod(fake_conda.stat().st_mode | stat.S_IXUSR)
    env = _environment(fake_qgis)
    env.pop("CONDA_EXE", None)
    env.pop("CONDA_ENV", None)
    env.pop("LOESS_PLATFORM", None)
    project_root = tmp_path / "project"
    command = [
        str(ROOT / "bash" / "init_project.sh"),
        "--platform",
        "macos",
        "--project-root",
        str(project_root),
    ]

    _run(
        [
            *command,
            "--conda-exe",
            str(fake_conda),
            "--conda-env",
            "loess-custom",
        ],
        env=env,
    )
    manifest_path = project_root / "project_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["launcher"]["conda_executable"] == str(fake_conda.resolve())
    assert manifest["launcher"]["conda_environment"] == "loess-custom"
    polluted_env = {
        **env,
        "CONDA_EXE": "/opt/anaconda3/bin/conda",
        "CONDA_ENV": "base",
        "LOESS_PLATFORM": "ubuntu",
    }

    launcher_check = _run(
        [
            "/bin/bash",
            "-c",
            (
                'source "$1"; '
                'printf "%s\\n%s\\n%s\\n" '
                '"$CONDA_EXE" "$CONDA_ENV" "$LOESS_PLATFORM"'
            ),
            "bash",
            str(project_root / "inference_scripts" / "config.sh"),
        ],
        env=polluted_env,
    )
    assert launcher_check.stdout.splitlines() == [
        str(fake_conda.resolve()),
        "loess-custom",
        "macos",
    ]

    _run(command, env=polluted_env)
    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated["launcher"]["conda_executable"] == str(fake_conda.resolve())
    assert updated["launcher"]["conda_environment"] == "loess-custom"
    _run([*command, "--check-only"], env=polluted_env)

    override_check = _run(
        [
            "/bin/bash",
            "-c",
            (
                'source "$1"; '
                'printf "%s\\n%s\\n%s\\n" '
                '"$CONDA_EXE" "$CONDA_ENV" "$LOESS_PLATFORM"'
            ),
            "bash",
            str(project_root / "inference_scripts" / "config.sh"),
        ],
        env={
            **polluted_env,
            "LOESS_CONDA_EXE_OVERRIDE": str(fake_conda),
            "LOESS_CONDA_ENV_OVERRIDE": "temporary-test",
            "LOESS_PLATFORM_OVERRIDE": "ubuntu",
        },
    )
    assert override_check.stdout.splitlines() == [
        str(fake_conda),
        "temporary-test",
        "ubuntu",
    ]

    fake_conda.unlink()
    missing_pinned_conda = _run(
        [
            "/bin/bash",
            "-c",
            'source "$1"',
            "bash",
            str(project_root / "inference_scripts" / "config.sh"),
        ],
        env=polluted_env,
        check=False,
    )
    assert missing_pinned_conda.returncode == 2
    assert "Configured Conda executable is not executable" in (
        missing_pinned_conda.stderr
    )


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
    project_manifest = json.loads(
        (project_root / "project_manifest.json").read_text(encoding="utf-8")
    )
    for required in (
        "tile_materializer.py",
        "mosaic_builder.py",
        "boundary_fitting/__init__.py",
    ):
        assert required in project_manifest["inference_files"]
    (project_root / "inference_scripts" / "tile_materializer.py").unlink()
    check = verify_project_runtime(
        project_root / "inference_scripts",
        plugin_root=plugin_root / "labeling_tool",
    )
    assert check["status"] == "error"
    assert "项目 inference_scripts文件清单不一致" in check["message"]
    assert "tile_materializer.py" in check["message"]
