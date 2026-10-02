from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import runpy
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from labeling_tool.runs.deployment_contract import (
    SHARED_RUNTIME_FILES,
    deployment_fingerprint,
    verify_project_runtime,
)
from loess_runtime.system.deployment_config import load_and_validate_config
from loess_runtime.system.environment_report import environment_fingerprint

ROOT = Path(__file__).resolve().parents[2]
GIT_SHA = subprocess.run(
    ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
    text=True,
    stdout=subprocess.PIPE,
    check=True,
).stdout.strip()
EXPERIMENT_TOOLS = (
    "fragmentation_ab_experiment.py",
    "subpixel_vectorize_experiment.py",
    "evaluate_fragmentation_v33_replay.py",
)
LEGACY_SHARED_SOURCES = {
    "qgis_plugins/labeling_tool/core/monitor_contract.py":
        "src/labeling_tool/shared/contracts/monitor_contract.py",
    "qgis_plugins/labeling_tool/core/ownership_neighbors.py":
        "src/labeling_tool/shared/planning/ownership_neighbors.py",
    "qgis_plugins/labeling_tool/core/postgres_state.py":
        "src/labeling_tool/shared/state/postgres_state.py",
    "qgis_plugins/labeling_tool/core/run_spec.py":
        "src/labeling_tool/shared/contracts/run_spec.py",
    "qgis_plugins/labeling_tool/core/run_state_db.py":
        "src/labeling_tool/shared/state/run_state_db.py",
    "qgis_plugins/labeling_tool/core/work_package_planner.py":
        "src/labeling_tool/shared/planning/work_package_planner.py",
}


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_as_legacy_layout(project_root: Path, schema_version: int) -> str:
    """Replace managed paths with the frozen pre-migration layout."""

    manifest_path = project_root / "project_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    project_id = str(manifest["project_id"])

    inference_root = project_root / "inference_scripts"
    shutil.rmtree(inference_root)
    legacy_inference_sources = {
        "config.sh": ROOT / "scripts/runtime/config.sh",
        "config.yaml": ROOT / "configs/defaults/config.yaml",
        "check_environment.py":
            ROOT / "src/loess_runtime/system/check_environment.py",
        "tile_materializer.py":
            ROOT / "src/loess_runtime/inference/tile_materializer.py",
        "boundary_fitting/__init__.py":
            ROOT / "src/loess_runtime/geometry/boundary_fitting/__init__.py",
        "run_env_check.sh": ROOT / "scripts/runtime/run_env_check.sh",
    }
    for relative, source in legacy_inference_sources.items():
        destination = inference_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    manifest["inference_files"] = {
        path.relative_to(inference_root).as_posix(): _sha256(path)
        for path in sorted(inference_root.rglob("*"))
        if path.is_file()
    }

    deployed_package = project_root / "runtime/labeling_tool"
    shutil.rmtree(deployed_package)
    legacy_core = deployed_package / "core"
    legacy_core.mkdir(parents=True)
    shared_files = {}
    aggregate = hashlib.sha256()
    for canonical, source_relative in sorted(LEGACY_SHARED_SOURCES.items()):
        destination = legacy_core / Path(canonical).name
        shutil.copy2(ROOT / source_relative, destination)
        digest = _sha256(destination)
        shared_files[canonical] = digest
        aggregate.update(canonical.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    manifest["shared_runtime"] = {
        "import_root": "labeling_tool.core",
        "sha256": aggregate.hexdigest(),
        "files": shared_files,
    }
    manifest["schema_version"] = schema_version
    if schema_version == 1:
        manifest.pop("project_id", None)
        manifest.pop("source", None)
        (project_root / ".loess-project-id").unlink()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return project_id


def test_every_runtime_shared_import_is_in_the_shared_runtime_contract():
    source_root = ROOT / "src"
    sources = {}
    for package in ("loess_runtime", "labeling_tool/shared"):
        for path in (source_root / package).rglob("*.py"):
            module = ".".join(path.relative_to(source_root).with_suffix("").parts)
            sources[module.removesuffix(".__init__")] = path

    imported_modules = set()
    pending = [name for name in sources if name.startswith("loess_runtime")]
    visited = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        path = sources[current]
        package = current if path.name == "__init__.py" else current.rpartition(".")[0]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = str(node.module or "")
                if node.level:
                    module = importlib.util.resolve_name("." * node.level + module, package)
                modules = [module]
                modules.extend(
                    f"{module}.{alias.name}" for alias in node.names
                    if f"{module}.{alias.name}" in sources
                )
            else:
                continue
            for module in modules:
                if not module.startswith("labeling_tool.shared"):
                    continue
                # Follow shared dependencies, including package initializers;
                # checking only direct runtime imports missed session modules.
                imported_modules.add(module)
                assert module in sources, f"missing shared source: {module}"
                pending.append(module)
                parts = module.split(".")
                for end in range(2, len(parts)):
                    parent = ".".join(parts[:end])
                    imported_modules.add(parent)
                    pending.append(parent)

    deployed_modules = set()
    for canonical in SHARED_RUNTIME_FILES:
        module = canonical.removeprefix("src/").removesuffix(".py").replace("/", ".")
        if module.endswith(".__init__"):
            module = module.removesuffix(".__init__")
        deployed_modules.add(module)
    assert imported_modules <= deployed_modules


def test_ubuntu_plugin_install_targets_the_qgis4_profile(tmp_path):
    fake_qgis = tmp_path / "qgis_process"
    _fake_qgis(fake_qgis)
    env = _environment(fake_qgis)
    home = tmp_path / "home"
    env["HOME"] = str(home)

    result = _run(
        [
            str(ROOT / "scripts" / "deploy" / "install_plugin.sh"),
            "--platform",
            "ubuntu",
            "--profile",
            "qgis42-test",
            "--check-only",
        ],
        env=env,
    )

    expected = (
        home
        / ".local/share/QGIS/QGIS4/profiles/qgis42-test/python/plugins/labeling_tool"
    )
    assert f"plugin directory: {expected}" in result.stdout
    assert "QGIS: QGIS 4.2.0" in result.stdout


@pytest.mark.parametrize("platform", ["macos", "ubuntu"])
def test_both_deployment_entrypoints_are_non_mutating_in_check_only(
    tmp_path,
    platform,
):
    fake_qgis = tmp_path / "qgis_process"
    _fake_qgis(fake_qgis)
    env = _environment(fake_qgis)
    plugin_root = tmp_path / "plugins"
    project_root = tmp_path / "project"

    plugin = _run(
        [
            str(ROOT / "scripts" / "deploy" / "install_plugin.sh"),
            "--platform",
            platform,
            "--profile",
            "check-only",
            "--plugin-dir",
            str(plugin_root),
            "--check-only",
        ],
        env=env,
    )
    project = _run(
        [
            str(ROOT / "scripts" / "deploy" / "init_project.sh"),
            "--platform",
            platform,
            "--project-root",
            str(project_root),
            "--check-only",
        ],
        env=env,
    )

    assert "installed plugin was not changed" in plugin.stdout
    assert "project files were not changed" in project.stdout
    assert not plugin_root.exists()
    assert not project_root.exists()


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


@pytest.mark.parametrize("platform", ["macos", "ubuntu"])
def test_separate_plugin_and_project_deployments_share_exact_runtime(tmp_path, platform):
    fake_qgis = tmp_path / "qgis_process"
    _fake_qgis(fake_qgis)
    env = _environment(fake_qgis)
    plugin_root = tmp_path / "plugins"
    project_root = tmp_path / "project"

    _run(
        [
            str(ROOT / "scripts" / "deploy" / "install_plugin.sh"),
            "--platform",
            platform,
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
            str(ROOT / "scripts" / "deploy" / "init_project.sh"),
            "--platform",
            platform,
            "--project-root",
            str(project_root),
        ],
        env=env,
    )

    expected_directories = (
        "inference_scripts",
        "runtime/labeling_tool/shared",
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
    assert plugin_manifest["plugin_version"] == "2.0.0"
    assert "version=2.0.0" in (installed_plugin / "metadata.txt").read_text(
        encoding="utf-8"
    )
    for monitor_relative in ("monitor/monitor_progress.py", "monitor/monitor_logs.py"):
        monitor_source = ROOT / "src" / "labeling_tool" / monitor_relative
        monitor_installed = installed_plugin / monitor_relative
        assert monitor_installed.read_bytes() == monitor_source.read_bytes()
        assert plugin_manifest["files"][monitor_relative] == hashlib.sha256(
            monitor_source.read_bytes()
        ).hexdigest()
        # Monitor helpers belong to the plugin, not the inference runtime.
        assert not (project_root / "runtime/labeling_tool" / monitor_relative).exists()
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
    assert plugin_manifest["platform"] == project_manifest["platform"] == platform
    assert not (project_root / "tools").exists()
    for name in EXPERIMENT_TOOLS:
        assert (ROOT / "tools" / "experiments" / name).is_file()
        assert not (ROOT / "inference_scripts" / name).exists()
        assert name not in project_manifest["inference_files"]
        assert not (project_root / "inference_scripts" / name).exists()
        assert not list(installed_plugin.rglob(name))
    for name in (
        "loess_runtime/geometry/fragmentation_v3.py",
        "loess_runtime/geometry/fragmentation_postprocess.py",
        "loess_runtime/geometry/fragmentation_postprocess_partitions.py",
        "loess_runtime/geometry/boundary_ab_validate.py",
        "loess_runtime/system/deployment_assets.py",
        "loess_runtime/system/deployment_validation.py",
    ):
        assert name in project_manifest["inference_files"]

    for canonical, project_relative in SHARED_RUNTIME_FILES.items():
        source = ROOT / canonical
        deployed = project_root / project_relative
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
                "from labeling_tool.shared.state.run_stream_repository import "
                "SCHEMA_VERSION;"
                "from labeling_tool.shared.contracts.run_spec import CLASS_ORDER;"
                "from labeling_tool.shared.planning.ownership_neighbors import "
                "ownership_neighbors;"
                "from labeling_tool.shared.planning.work_package_planner import "
                "unit_confidence_write_reserve;"
                "print(SCHEMA_VERSION, CLASS_ORDER[0], ownership_neighbors([]), "
                "unit_confidence_write_reserve(1))"
            ),
        ],
        env={**env, "PYTHONPATH": str(project_root / "runtime")},
    )
    assert import_check.stdout.strip().startswith("2 12 [] ")


def test_experiment_tools_do_not_change_deployable_source_fingerprint(tmp_path):
    helper = runpy.run_path(str(ROOT / "scripts" / "lib" / "deployment_source.py"))
    inventory = helper["source_inventory"]
    digest = helper["inventory_digest"]
    current = inventory(ROOT)
    assert not any(path.startswith("tools/") for path in current)
    for name in EXPERIMENT_TOOLS:
        assert f"inference_scripts/{name}" not in current

    for relative in helper["SOURCE_ROOTS"]:
        directory = tmp_path / relative
        directory.mkdir(parents=True)
        (directory / "fixture.py").write_text("# production\n", encoding="utf-8")
    before = digest(inventory(tmp_path))
    experiments = tmp_path / "tools" / "experiments"
    experiments.mkdir(parents=True)
    for name in EXPERIMENT_TOOLS:
        (experiments / name).write_text("# experiment\n", encoding="utf-8")
    assert digest(inventory(tmp_path)) == before
    (experiments / EXPERIMENT_TOOLS[0]).write_text("# changed\n", encoding="utf-8")
    assert digest(inventory(tmp_path)) == before
    (tmp_path / "src" / "loess_runtime" / "fixture.py").write_text(
        "# production changed\n", encoding="utf-8"
    )
    assert digest(inventory(tmp_path)) != before


def test_nested_source_defaults_keep_the_repository_asset_base(tmp_path):
    source_defaults = ROOT / "configs" / "defaults"
    assert (source_defaults / "config.yaml").is_file()
    assert (source_defaults / "class_map_14.csv").is_file()
    assert not (ROOT / "configs" / "config.yaml").exists()
    assert not (ROOT / "configs" / "class_map_14.csv").exists()

    fixture_defaults = tmp_path / "configs" / "defaults"
    fixture_defaults.mkdir(parents=True)
    shutil.copy2(source_defaults / "config.yaml", fixture_defaults / "config.yaml")
    shutil.copy2(
        source_defaults / "class_map_14.csv",
        fixture_defaults / "class_map_14.csv",
    )
    effective, issues = load_and_validate_config(
        fixture_defaults / "config.yaml",
        asset_base_dir=tmp_path / "configs",
        verify_files=False,
        verify_hashes=False,
    )

    assert issues == []
    assert effective["runtime"]["model_artifacts_dir"] == str(
        tmp_path / "weights"
    )
    assert effective["fusion_profiles"][0]["file_path"] == str(
        tmp_path / "weights" / "fusion_profile.json"
    )
    assert effective["sam3"]["checkpoint"] == str(
        tmp_path / "weights" / "sam3.pt"
    )


def test_source_wrappers_run_package_clis_from_an_unrelated_cwd(tmp_path):
    fake_conda = tmp_path / "conda"
    fake_conda.write_text(
        """#!/usr/bin/env bash
set -e
if [ "${1:-}" = "--version" ]; then
  echo "conda fixture"
  exit 0
fi
[ "${1:-}" = "run" ] && shift
while [ $# -gt 0 ]; do
  case "$1" in
    --no-capture-output) shift ;;
    -n) shift 2 ;;
    *) break ;;
  esac
done
exec "$@"
""",
        encoding="utf-8",
    )
    fake_conda.chmod(fake_conda.stat().st_mode | stat.S_IXUSR)
    wrappers = {
        "run_assemble_stream.sh": "loess_runtime.assembly.assemble_stream",
        "run_boundary_ab_validation.sh":
            "loess_runtime.geometry.boundary_ab_validate",
        "run_boundary_regularizer.sh":
            "loess_runtime.geometry.boundary_regularizer",
        "run_env_check.sh": "loess_runtime.system.check_environment",
        "run_finalize_partition_rasters.sh":
            "loess_runtime.assembly.finalize_partition_rasters",
        "run_fragmentation_postprocess.sh":
            "loess_runtime.geometry.fragmentation_postprocess",
        "run_fragmentation_v33_work_package.sh":
            "loess_runtime.geometry.fragmentation_v33_work_package",
        "run_polyline_smooth.sh": "loess_runtime.geometry.polyline_smoother",
        "run_sam3_interactive_worker.sh":
            "loess_runtime.sam.sam3_interactive_worker",
        "run_scale_acceptance.sh": "loess_runtime.assembly.scale_acceptance",
        "run_seam_validation.sh": "loess_runtime.geometry.seam_band_validate",
        "run_tile_cache_probe.sh": "loess_runtime.inference.tile_cache_probe",
        "run_unit_confidence.sh": "loess_runtime.geometry.unit_confidence",
        "run_unit_fit.sh":
            "loess_runtime.geometry.boundary_fitting.unit_runtime",
        "run_work_package.sh": "loess_runtime.inference.work_package_runtime",
    }
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    env = {
        **os.environ,
        "CONDA_EXE": str(fake_conda),
        "CONDA_ENV": "qgis",
        "LOESS_PLATFORM": "macos",
    }
    source_layout = subprocess.run(
        [
            "/bin/bash",
            "-c",
            (
                'source "$1"; printf "%s\\n%s\\n%s\\n" '
                '"$LOESS_CONFIG_ROOT" "$LOESS_CONFIG_ASSET_ROOT" '
                '"$LOESS_ENVIRONMENT_ROOT"'
            ),
            "bash",
            str(ROOT / "scripts" / "runtime" / "config.sh"),
        ],
        cwd=unrelated,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert source_layout.returncode == 0, source_layout.stderr
    assert source_layout.stdout.splitlines() == [
        str(ROOT / "configs" / "defaults"),
        str(ROOT / "configs"),
        str(ROOT / "configs" / "environments"),
    ]
    environment_wrapper = (
        ROOT / "scripts" / "runtime" / "run_env_check.sh"
    ).read_text(encoding="utf-8")
    assert '--asset-base-dir "$LOESS_CONFIG_ASSET_ROOT"' in environment_wrapper
    for wrapper, module in wrappers.items():
        source = (ROOT / "scripts" / "runtime" / wrapper).read_text(
            encoding="utf-8"
        )
        python_entrypoint = (
            f"python -X faulthandler -m {module}"
            if wrapper == "run_unit_fit.sh"
            else f"python -m {module}"
        )
        assert python_entrypoint in source
        result = subprocess.run(
            [str(ROOT / "scripts" / "runtime" / wrapper), "--help"],
            cwd=unrelated,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, (
            wrapper,
            result.stdout,
            result.stderr,
        )


def test_generated_wrappers_import_both_packages_from_an_unrelated_cwd(tmp_path):
    fake_qgis = tmp_path / "qgis_process"
    _fake_qgis(fake_qgis)
    fake_conda = tmp_path / "conda"
    fake_conda.write_text(
        """#!/usr/bin/env bash
set -e
if [ "${1:-}" = "--version" ]; then
  echo "conda fixture"
  exit 0
fi
[ "${1:-}" = "run" ] && shift
while [ $# -gt 0 ]; do
  case "$1" in
    --no-capture-output) shift ;;
    -n) shift 2 ;;
    *) break ;;
  esac
done
exec "$@"
""",
        encoding="utf-8",
    )
    fake_conda.chmod(fake_conda.stat().st_mode | stat.S_IXUSR)
    project_root = tmp_path / "project"
    _run(
        [
            str(ROOT / "scripts" / "deploy" / "init_project.sh"),
            "--platform",
            "macos",
            "--project-root",
            str(project_root),
            "--conda-exe",
            str(fake_conda),
        ],
        env=_environment(fake_qgis),
    )
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    generated_layout = subprocess.run(
        [
            "/bin/bash",
            "-c",
            (
                'source "$1"; printf "%s\\n%s\\n%s\\n" '
                '"$LOESS_CONFIG_ROOT" "$LOESS_CONFIG_ASSET_ROOT" '
                '"$LOESS_ENVIRONMENT_ROOT"'
            ),
            "bash",
            str(project_root / "inference_scripts" / "config.sh"),
        ],
        cwd=unrelated,
        env=os.environ,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert generated_layout.returncode == 0, generated_layout.stderr
    assert generated_layout.stdout.splitlines() == [
        str(project_root / "inference_scripts"),
        str(project_root / "inference_scripts"),
        str(project_root / "inference_scripts"),
    ]
    for wrapper in ("run_env_check.sh", "run_tile_cache_probe.sh"):
        result = subprocess.run(
            [str(project_root / "inference_scripts" / wrapper), "--help"],
            cwd=unrelated,
            env=os.environ,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, (
            wrapper,
            result.stdout,
            result.stderr,
        )

    inference_root = project_root / "inference_scripts"
    shared_root = project_root / "runtime"
    child_env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(inference_root), str(shared_root))),
    }
    roots = _run(
        [
            sys.executable,
            "-c",
            (
                "import json, os; "
                "from loess_runtime.system.probe_process import "
                "probe_environment; "
                "os.environ.pop('PYTHONPATH', None); "
                "print(json.dumps(probe_environment()['PYTHONPATH'].split(os.pathsep)))"
            ),
        ],
        env=child_env,
    )
    assert json.loads(roots.stdout) == [str(inference_root), str(shared_root)]


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
            str(ROOT / "scripts" / "deploy" / "install_plugin.sh"),
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
            str(ROOT / "scripts" / "deploy" / "init_project.sh"),
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
            str(ROOT / "scripts" / "deploy" / "install_plugin.sh"),
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
            str(ROOT / "scripts" / "deploy" / "init_project.sh"),
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
    plugin_source = ROOT / "src" / "labeling_tool"
    metadata_before = (plugin_source / "metadata.txt").read_bytes()

    linked_plugin_root = tmp_path / "linked-plugins"
    linked_plugin_root.mkdir()
    (linked_plugin_root / "labeling_tool").symlink_to(
        plugin_source,
        target_is_directory=True,
    )
    unsafe_roots = (
        ROOT / "src",
        ROOT / "tests" / ".." / "src",
        ROOT / ".unsafe-plugin-root",
        linked_plugin_root,
    )

    for plugin_root in unsafe_roots:
        result = _run(
            [
                str(ROOT / "scripts" / "deploy" / "install_plugin.sh"),
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
    assert not list((ROOT / "src").glob(".labeling_tool.*"))
    assert not (ROOT / ".unsafe-plugin-root").exists()


def test_project_update_preserves_user_data_and_restores_managed_code(tmp_path):
    fake_qgis = tmp_path / "qgis_process"
    _fake_qgis(fake_qgis)
    env = _environment(fake_qgis)
    project_root = tmp_path / "project"
    command = [
        str(ROOT / "scripts" / "deploy" / "init_project.sh"),
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
        "output/experiments/report.json": b"experiment-result",
    }
    for relative, content in user_files.items():
        path = project_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    weights_readme = project_root / "weights" / "README_WEIGHTS.md"
    weights_readme.write_text("user notes\n", encoding="utf-8")
    managed = (
        project_root
        / "runtime"
        / "labeling_tool"
        / "shared"
        / "contracts"
        / "run_spec.py"
    )
    managed.write_text("corrupted\n", encoding="utf-8")
    for name in EXPERIMENT_TOOLS:
        (project_root / "inference_scripts" / name).write_text(
            "# previous deployment experiment entry\n", encoding="utf-8"
        )

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
    for name in EXPERIMENT_TOOLS:
        assert not (project_root / "inference_scripts" / name).exists()
    assert managed.read_bytes() == (
        ROOT / "src" / "labeling_tool" / "shared" / "contracts" / "run_spec.py"
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
        str(ROOT / "scripts" / "deploy" / "install_plugin.sh"),
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
        str(ROOT / "scripts" / "deploy" / "init_project.sh"),
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
            str(ROOT / "scripts" / "deploy" / "init_project.sh"),
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
        str(ROOT / "scripts" / "deploy" / "init_project.sh"),
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
        str(ROOT / "scripts" / "deploy" / "init_project.sh"),
        "--platform",
        "macos",
        "--project-root",
        str(project_root),
    ]
    _run(command, env=env)
    manifest_path = project_root / "project_manifest.json"
    _rewrite_as_legacy_layout(project_root, 1)
    user_file = project_root / "weights" / "legacy-user.bin"
    user_file.write_bytes(b"preserve schema 1 user data")

    before = _tree_snapshot(project_root)
    checked = _run([*command, "--check-only"], env=env)
    assert "schema 1 project passed" in checked.stdout
    assert _tree_snapshot(project_root) == before

    _run(command, env=env)
    migrated = json.loads(manifest_path.read_text(encoding="utf-8"))
    marker = json.loads(
        (project_root / ".loess-project-id").read_text(encoding="utf-8")
    )
    assert migrated["schema_version"] == 2
    assert migrated["project_id"] == marker["project_id"]
    assert set(migrated["shared_runtime"]["files"]) == set(SHARED_RUNTIME_FILES)
    assert user_file.read_bytes() == b"preserve schema 1 user data"


def test_existing_schema2_legacy_layout_updates_without_changing_identity(tmp_path):
    fake_qgis = tmp_path / "qgis_process"
    _fake_qgis(fake_qgis)
    env = _environment(fake_qgis)
    project_root = tmp_path / "project"
    command = [
        str(ROOT / "scripts" / "deploy" / "init_project.sh"),
        "--platform",
        "macos",
        "--project-root",
        str(project_root),
    ]
    _run(command, env=env)
    original_project_id = _rewrite_as_legacy_layout(project_root, 2)
    user_file = project_root / "input" / "rasters" / "schema2-user.tif"
    user_file.write_bytes(b"preserve schema 2 user data")

    stale = _run([*command, "--check-only"], env=env, check=False)
    assert stale.returncode != 0
    assert "Shared runtime file contract is incomplete" in stale.stderr

    _run(command, env=env)
    updated = json.loads(
        (project_root / "project_manifest.json").read_text(encoding="utf-8")
    )
    marker = json.loads(
        (project_root / ".loess-project-id").read_text(encoding="utf-8")
    )
    assert updated["project_id"] == marker["project_id"] == original_project_id
    assert set(updated["shared_runtime"]["files"]) == set(SHARED_RUNTIME_FILES)
    assert user_file.read_bytes() == b"preserve schema 2 user data"


def test_schema1_upgrade_rolls_back_identity_and_managed_paths_on_signal(tmp_path):
    fake_qgis = tmp_path / "qgis_process"
    _fake_qgis(fake_qgis)
    env = _environment(fake_qgis)
    project_root = tmp_path / "project"
    command = [
        str(ROOT / "scripts" / "deploy" / "init_project.sh"),
        "--platform",
        "macos",
        "--project-root",
        str(project_root),
    ]
    _run(command, env=env)
    _rewrite_as_legacy_layout(project_root, 1)
    user_file = project_root / "output" / "runs" / "legacy.txt"
    user_file.write_text("preserve legacy output\n", encoding="utf-8")
    baseline = _tree_snapshot(project_root)

    interrupted = _run(
        command,
        env={
            **env,
            "LOESS_TEST_SIGNAL": "TERM",
            "LOESS_TEST_SIGNAL_AT": "identity_installed",
        },
        check=False,
    )

    assert interrupted.returncode != 0
    assert _tree_snapshot(project_root) == baseline
    assert not (project_root / ".loess-project-id").exists()
    assert not list(project_root.glob(".*.old.*"))
    assert not list(project_root.glob(".loess-project-init.*"))


def test_source_provenance_rejects_dirty_git_and_tampered_archive(tmp_path):
    source_root = tmp_path / "source"
    for relative in (
        "scripts",
        "configs",
        "src/labeling_tool",
        "src/loess_runtime",
    ):
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
    helper = source_root / "scripts" / "lib" / "deployment_source.py"

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

    changed = source_root / "configs" / "defaults" / "config.yaml"
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

    _run([
        "git",
        "-C",
        str(source_root),
        "restore",
        "configs/defaults/config.yaml",
    ])
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
        str(ROOT / "scripts" / "deploy" / "init_project.sh"),
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
            str(ROOT / "scripts" / "deploy" / "init_project.sh"),
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
            str(ROOT / "scripts" / "deploy" / "init_project.sh"),
            "--platform",
            "macos",
            "--project-root",
            str(project_root),
        ],
        env=env,
    )
    asset_check = _run(
        [
            str(ROOT / "scripts" / "deploy" / "init_project.sh"),
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

    manifest_path = project_root / "project_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["git_sha"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    stale_check = _run(
        [
            str(ROOT / "scripts" / "deploy" / "init_project.sh"),
            "--platform",
            "macos",
            "--project-root",
            str(project_root),
            "--check-only",
        ],
        env=env,
        check=False,
    )
    assert stale_check.returncode != 0
    assert "Project Git SHA differs from this source" in stale_check.stderr
    assert "Traceback" not in stale_check.stderr


def test_runtime_contract_rejects_changed_project_copy(tmp_path):
    fake_qgis = tmp_path / "qgis_process"
    _fake_qgis(fake_qgis)
    env = _environment(fake_qgis)
    plugin_root = tmp_path / "plugins"
    project_root = tmp_path / "project"
    _run(
        [
            str(ROOT / "scripts" / "deploy" / "install_plugin.sh"),
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
            str(ROOT / "scripts" / "deploy" / "init_project.sh"),
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
        / "shared"
        / "planning"
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
            str(ROOT / "scripts" / "deploy" / "init_project.sh"),
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
        "loess_runtime/inference/tile_materializer.py",
        "loess_runtime/inference/mosaic_builder.py",
        "loess_runtime/geometry/boundary_fitting/__init__.py",
    ):
        assert required in project_manifest["inference_files"]
    (
        project_root
        / "inference_scripts"
        / "loess_runtime"
        / "inference"
        / "tile_materializer.py"
    ).unlink()
    check = verify_project_runtime(
        project_root / "inference_scripts",
        plugin_root=plugin_root / "labeling_tool",
    )
    assert check["status"] == "error"
    assert "项目 inference_scripts文件清单不一致" in check["message"]
    assert "loess_runtime/inference/tile_materializer.py" in check["message"]
