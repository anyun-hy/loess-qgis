"""Validate the Schema v2 semantic deployment environment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=r"Failed to load image Python extension.*",
    category=UserWarning,
)

from _device import resolve_device, validate_device
from deployment_config import (
    CLASS_ORDER,
    load_yaml,
    validate_deployment_config,
)
from torchscript_runtime import load_torchscript_model


FINGERPRINT_FILES = (
    "../project_manifest.json",
    "../runtime/labeling_tool/core/run_spec.py",
    "../runtime/labeling_tool/core/run_state_db.py",
    "../runtime/labeling_tool/core/ownership_neighbors.py",
    "config.sh",
    "config.yaml",
    "_device.py",
    "deployment_config.py",
    "check_environment.py",
    "environment-ubuntu-cu124.yml",
    "environment-macos-qgis4.yml",
    "semantic_batch.py",
    "torchscript_runtime.py",
    "work_package_runtime.py",
    "partition_mosaic.py",
    "incremental_fusion.py",
    "finalize_partition_rasters.py",
    "assemble_stream.py",
    "scale_acceptance.py",
    "runtime_metrics.py",
    "rasterio_compat.py",
    "difference_runtime.py",
    "accepted_score.py",
    "boundary_fitting/__init__.py",
    "boundary_fitting/unit_runtime.py",
    "polyline_smoother.py",
    "common_boundary_smoother.py",
    "sam3_interactive_worker.py",
    "sam3_refine.py",
    "run_polyline_smooth.sh",
    "run_work_package.sh",
    "run_finalize_partition_rasters.sh",
    "run_unit_fit.sh",
    "run_assemble_stream.sh",
    "run_scale_acceptance.sh",
    "run_sam3_interactive_worker.sh",
)


def add_check(checks, check_id, status, value, source, message="", fix=""):
    checks.append({
        "id": check_id,
        "status": status,
        "value": str(value),
        "source": source,
        "message": message,
        "fix": fix,
    })


def import_dependency(name):
    try:
        module = importlib.import_module(name)
        return module, getattr(module, "__version__", "installed"), ""
    except Exception as exc:
        return None, "not installed", str(exc)


def _version_tuple(value):
    numbers = re.findall(r"\d+", str(value))
    return tuple(int(item) for item in numbers[:3])


def _mps_runtime_requirement(model_id, torch_version):
    if model_id == "upernet_swin_b" and _version_tuple(torch_version) < (2, 7):
        return (
            False,
            f"Swin MPS requires PyTorch >=2.7 in this deployment; current={torch_version}",
        )
    return True, ""


def verify_torchscript_contract(torch_module, path, device):
    """Load one deployment artifact and execute the fixed input contract."""
    model = None
    sample = None
    output = None
    try:
        model, runtime_info = load_torchscript_model(path, device)
        sample = torch_module.zeros(1, 3, 512, 512, dtype=torch_module.float32, device=device)
        with torch_module.inference_mode():
            output = model(sample)
        if not torch_module.is_tensor(output):
            return False, f"TorchScript output must be one tensor, got {type(output).__name__}"
        if tuple(output.shape) != (1, 14, 512, 512):
            return False, f"TorchScript output shape is {tuple(output.shape)}, expected (1,14,512,512)"
        if output.dtype != torch_module.float32:
            return False, f"TorchScript output dtype is {output.dtype}, expected float32"
        runtime_message = f"TorchScript contract passed on {device}; runtime={runtime_info['mode']}"
        if str(device).startswith("mps"):
            runtime_message += (
                f"; contiguous_bridges={runtime_info.get('mps_contiguous_bridge_count', 0)}"
                f"; pool_cpu_bridges={runtime_info.get('mps_cpu_bridge_count', 0)}"
            )
        return True, runtime_message
    except Exception as exc:
        return False, f"TorchScript contract failed on {device}: {exc}"
    finally:
        output = None
        sample = None
        model = None
        gc.collect()
        if str(device).startswith("cuda") and torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
        if str(device).startswith("mps") and torch_module.backends.mps.is_available():
            torch_module.mps.empty_cache()


def verify_torchscript_contract_isolated(path, device, timeout=600):
    """Run one device contract in a fresh process to isolate accelerator state."""

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--contract-worker",
        "--model-path",
        str(Path(path).resolve()),
        "--device",
        str(device),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"TorchScript contract timed out after {timeout}s on {device}"

    payload = None
    for line in reversed((result.stdout or "").splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "ok" in candidate:
            payload = candidate
            break
    if payload is not None:
        return bool(payload.get("ok")), str(payload.get("message") or "")

    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    detail = stderr or stdout or "no worker output"
    return False, (
        f"TorchScript contract worker failed on {device} "
        f"(exit={result.returncode}): {detail}"
    )


def verify_sam3_checkpoint(torch_module, path):
    try:
        checkpoint = torch_module.load(path, map_location="cpu", weights_only=True)
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            checkpoint = checkpoint["model"]
        if not isinstance(checkpoint, dict):
            return False, "checkpoint is not a SAM3 state_dict"
        keys = list(checkpoint.keys())
        has_detector = any(str(key).startswith("detector.") for key in keys)
        has_tracker = any(str(key).startswith("tracker.") for key in keys)
        if not (has_detector and has_tracker):
            return False, "official SAM3 detector/tracker parameters are missing"
        return True, f"official SAM3 checkpoint recognized ({len(keys)} entries)"
    except Exception as exc:
        return False, f"SAM3 checkpoint cannot be read: {exc}"


def verify_sam3_runtime_isolated(path, device="cpu", timeout=900):
    """Load the official SAM3 backend through the production compatibility path."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--sam3-worker",
        "--model-path",
        str(Path(path).resolve()),
        "--device",
        str(device),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"official SAM3 load timed out after {timeout}s on {device}"

    payload = None
    for line in reversed((result.stdout or "").splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "ok" in candidate:
            payload = candidate
            break
    if payload is not None:
        return bool(payload.get("ok")), str(payload.get("message") or "")
    detail = (result.stderr or "").strip() or (result.stdout or "").strip()
    return False, (
        f"official SAM3 worker failed on {device} (exit={result.returncode}): "
        f"{detail or 'no worker output'}"
    )


def _fingerprint(scripts_dir):
    digest = hashlib.sha256()
    for name in FINGERPRINT_FILES:
        path = scripts_dir / name
        digest.update(name.encode("utf-8"))
        if path.is_file():
            with path.open("rb") as handle:
                digest.update(handle.read())
        else:
            digest.update(b"<missing>")
    return "sha256:" + digest.hexdigest()


def _issue_id(path, index):
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", path).strip("_").lower()
    return f"config_{normalized or 'root'}_{index}"


def _has_issue(issues, prefix, codes=()):
    return any(
        item.path.startswith(prefix) and (not codes or item.code in codes)
        for item in issues
    )


def overall_status(checks):
    """Only core failures or zero runnable semantic models block inference."""
    def optional_capability_error(item):
        check_id = str(item.get("id") or "")
        source = str(item.get("source") or "")
        return (
            check_id.startswith(("semantic_model_", "fusion_profile_", "sam3_"))
            or any(path in source for path in ("/semantic_models/", "/fusion_profiles/", "/sam3/"))
        )

    core_errors = [
        item for item in checks
        if item["status"] == "error" and not optional_capability_error(item)
    ]
    runnable_models = [
        item for item in checks
        if str(item.get("id") or "").startswith("semantic_model_")
        and item["status"] == "ready"
    ]
    if core_errors or not runnable_models:
        return "error"
    if any(item["status"] in ("warning", "error") for item in checks):
        return "warning"
    return "ready"


def add_runtime_boundary_checks(checks, conda_env):
    """Report both runtimes without importing QGIS into the Conda process."""
    conda_python = Path(sys.executable).resolve()
    conda_matches = f"/envs/{conda_env}/" in conda_python.as_posix()
    add_check(
        checks,
        "conda_python",
        "ready" if conda_matches else "error",
        f"{conda_python} (Python {platform.python_version()})",
        f"Conda environment {conda_env}",
        "inference runs in the configured Conda Python" if conda_matches else "environment checker is not running in the configured Conda Python",
        "edit config.sh:CONDA_EXE and CONDA_ENV",
    )

    host_fields = (
        (
            "qgis_version",
            "LOESS_QGIS_VERSION",
            "QGIS",
            lambda value: _version_tuple(value)[:2] in {(3, 44), (4, 2)},
        ),
        (
            "qgis_python",
            "LOESS_QGIS_PYTHON_VERSION",
            "QGIS Python",
            lambda value: _version_tuple(value)[:1] == (3,),
        ),
        (
            "pyqt_version",
            "LOESS_PYQT_VERSION",
            "PyQt",
            lambda value: _version_tuple(value)[:1] in {(5,), (6,)},
        ),
        (
            "qt_version",
            "LOESS_QT_VERSION",
            "Qt",
            lambda value: _version_tuple(value)[:1] in {(5,), (6,)},
        ),
    )
    for check_id, env_name, label, validator in host_fields:
        value = str(os.environ.get(env_name) or "").strip()
        if value:
            status = "ready" if validator(value) else "error"
            message = f"{label} shared-platform compatibility runtime detected"
        else:
            status = "warning"
            value = "not provided"
            message = "run the check from the QGIS plugin to inspect this host value"
        add_check(
            checks,
            check_id,
            status,
            value,
            "QGIS plugin host runtime",
            message,
            "use QGIS 3.44/PyQt5/Qt5 or QGIS 4.2/PyQt6/Qt6",
        )

    qgis_major = _version_tuple(
        os.environ.get("LOESS_QGIS_VERSION", "")
    )[:1]
    pyqt_major = _version_tuple(
        os.environ.get("LOESS_PYQT_VERSION", "")
    )[:1]
    qt_major = _version_tuple(
        os.environ.get("LOESS_QT_VERSION", "")
    )[:1]
    if qgis_major and pyqt_major and qt_major:
        pair = (qgis_major[0], pyqt_major[0], qt_major[0])
        pair_ok = pair in {(3, 5, 5), (4, 6, 6)}
        pair_status = "ready" if pair_ok else "error"
        pair_value = f"QGIS {pair[0]} / PyQt {pair[1]} / Qt {pair[2]}"
        pair_message = (
            "host runtime matches a supported platform profile"
            if pair_ok
            else "host runtime mixes unsupported QGIS and Qt major versions"
        )
    else:
        pair_status = "warning"
        pair_value = "not provided"
        pair_message = "run the check from QGIS to verify the host runtime pair"
    add_check(
        checks,
        "qgis_qt_profile",
        pair_status,
        pair_value,
        "QGIS plugin host runtime",
        pair_message,
        "use QGIS 3.44 with Qt5 or QGIS 4.2 with Qt6",
    )

    host_executable = str(
        os.environ.get("LOESS_QGIS_PYTHON_EXECUTABLE") or ""
    ).strip()
    if host_executable:
        separate = Path(host_executable).resolve() != conda_python
        status = "ready" if separate else "error"
        message = (
            "QGIS and inference use separate Python runtimes"
            if separate
            else "QGIS and inference unexpectedly share one Python executable"
        )
    else:
        status = "warning"
        host_executable = "not provided"
        message = "run the check from the QGIS plugin to verify runtime separation"
    add_check(
        checks,
        "runtime_boundary",
        status,
        f"QGIS={host_executable}; inference={conda_python}",
        "QProcess environment and config.sh",
        message,
        "do not add Conda site-packages to the QGIS Python runtime",
    )


def build_report(args):
    scripts_dir = Path(args.scripts_dir).resolve()
    config_path = scripts_dir / "config.yaml"
    checks = []
    effective = {
        "schema_version": None,
        "runtime": {},
        "scaling": {},
        "semantic_models": [],
        "fusion_profiles": [],
        "sam3": {},
        "boundary_fitting": {},
        "classes": {},
    }

    add_check(
        checks,
        "conda_env",
        "ready",
        args.conda_env,
        "config.sh:CONDA_ENV",
        "environment checker is running inside this Conda environment",
        "edit config.sh:CONDA_ENV",
    )
    add_runtime_boundary_checks(checks, args.conda_env)

    dependencies = {}
    for name in (
        "numpy",
        "torch",
        "rasterio",
        "fiona",
        "shapely",
        "scipy",
        "psutil",
        "yaml",
        "skimage",
    ):
        module, version, error = import_dependency(name)
        dependencies[name] = module
        add_check(
            checks,
            f"dependency_{name}",
            "ready" if module is not None else "error",
            f"{name} {version}",
            f"Conda environment {args.conda_env}",
            error,
            f"install {name} in Conda environment {args.conda_env}",
        )

    runtime_platform = "macos" if sys.platform == "darwin" else "ubuntu"
    torch_module = dependencies.get("torch")
    torch_version = str(getattr(torch_module, "__version__", "not installed"))
    expected_torch_version = "2.7.1" if runtime_platform == "macos" else "2.6.0"
    torch_version_ok = (
        torch_module is not None
        and torch_version.split("+", 1)[0] == expected_torch_version
    )
    add_check(
        checks,
        "torch_version",
        "ready" if torch_version_ok else "error",
        torch_version,
        f"Conda environment {args.conda_env}",
        f"formal {runtime_platform} runtime uses PyTorch {expected_torch_version}"
        if torch_version_ok
        else (
            f"{runtime_platform} deployment requires exact PyTorch "
            f"{expected_torch_version}"
        ),
        "run <repository>/bash/init_project.sh --project-root <project> --create-env",
    )

    if runtime_platform == "ubuntu":
        cuda_build = str(
            getattr(getattr(torch_module, "version", None), "cuda", "") or ""
        )
        cuda_build_ok = (
            torch_module is not None
            and _version_tuple(cuda_build)[:2] == (12, 4)
        )
        add_check(
            checks,
            "torch_cuda_build",
            "ready" if cuda_build_ok else "error",
            cuda_build or "not available",
            "torch.version.cuda",
            "PyTorch CUDA 12.4 runtime is installed"
            if cuda_build_ok
            else "the installed PyTorch build is not the required cu124 build",
            "install torch 2.6.0 from the official cu124 wheel index",
        )

        cuda_available = bool(
            torch_module is not None and torch_module.cuda.is_available()
        )
        gpu_name = "not available"
        capability = "not available"
        if cuda_available:
            try:
                gpu_name = str(torch_module.cuda.get_device_name(0))
                capability = ".".join(
                    str(value)
                    for value in torch_module.cuda.get_device_capability(0)
                )
            except Exception as exc:
                gpu_name = f"query failed: {exc}"
        gpu_ok = cuda_available and "3090" in gpu_name
        add_check(
            checks,
            "cuda_gpu",
            "ready" if gpu_ok else "error",
            f"{gpu_name}; compute capability {capability}",
            "CUDA device 0",
            "RTX 3090 is available to PyTorch"
            if gpu_ok
            else "CUDA device 0 is unavailable or is not an RTX 3090",
            "check the NVIDIA driver, CUDA_VISIBLE_DEVICES=0 and RTX 3090",
        )
        add_check(
            checks,
            "mps_device",
            "ready",
            "not required",
            "Ubuntu platform profile",
            "Ubuntu formal inference uses CUDA",
            "",
        )
    else:
        mps_available = bool(
            torch_module is not None
            and hasattr(torch_module.backends, "mps")
            and torch_module.backends.mps.is_available()
        )
        add_check(
            checks,
            "torch_cuda_build",
            "ready",
            "not required",
            "macOS platform profile",
            "macOS formal inference uses MPS",
            "",
        )
        add_check(
            checks,
            "cuda_gpu",
            "ready",
            "not required",
            "macOS platform profile",
            "macOS formal inference uses MPS",
            "",
        )
        add_check(
            checks,
            "mps_device",
            "ready" if mps_available else "error",
            "available" if mps_available else "not available",
            "torch.backends.mps",
            "MPS is available to PyTorch"
            if mps_available
            else "MPS is unavailable in the macOS inference environment",
            "install the macOS platform environment and verify Apple GPU access",
        )

    shapely_module = dependencies.get("shapely")
    if shapely_module is not None:
        divider_apis = ("STRtree",)
        missing_divider_apis = [
            name for name in divider_apis if not hasattr(shapely_module, name)
        ]
        divider_available = not missing_divider_apis
        add_check(
            checks,
            "dependency_shapely_divider_query",
            "ready" if divider_available else "error",
            f"shapely {getattr(shapely_module, '__version__', 'unknown')}",
            f"Conda environment {args.conda_env}",
            "Polygon neighbor query for common-divider fitting is available"
            if divider_available
            else "missing APIs: " + ", ".join(missing_divider_apis),
            f"install shapely=2.1.2 in Conda environment {args.conda_env}",
        )

    scipy_module = dependencies.get("scipy")
    scipy_spline_error = ""
    if scipy_module is not None:
        try:
            from scipy.interpolate import splprep, splev  # noqa: F401
        except Exception as exc:
            scipy_spline_error = str(exc)
    add_check(
        checks,
        "dependency_scipy_bspline",
        "ready" if scipy_module is not None and not scipy_spline_error else "error",
        f"scipy {getattr(scipy_module, '__version__', 'not installed')}",
        f"Conda environment {args.conda_env}",
        scipy_spline_error or "splprep and splev are available",
        f"install scipy=1.17.1 in Conda environment {args.conda_env}",
    )

    sqlite_error = ""
    sqlite_value = f"sqlite {sqlite3.sqlite_version}"
    try:
        with tempfile.TemporaryDirectory(prefix="loess_sqlite_check_") as temp_dir:
            db_path = Path(temp_dir) / "state.sqlite"
            connection = sqlite3.connect(db_path)
            try:
                journal_mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
                connection.execute("PRAGMA foreign_keys=ON")
                foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]
                json_valid = connection.execute("SELECT json_valid('{}')").fetchone()[0]
                if str(journal_mode).lower() != "wal" or foreign_keys != 1 or json_valid != 1:
                    raise RuntimeError(
                        f"WAL={journal_mode}, foreign_keys={foreign_keys}, json1={json_valid}"
                    )
            finally:
                connection.close()
    except Exception as exc:
        sqlite_error = str(exc)
    add_check(
        checks,
        "dependency_sqlite_wal",
        "ready" if not sqlite_error else "error",
        sqlite_value,
        f"Conda environment {args.conda_env}",
        sqlite_error or "WAL, foreign keys and JSON1 are available",
        f"install sqlite=3.53.3 in Conda environment {args.conda_env}",
    )

    gdal_versions = []
    gdal_error = ""
    for executable in ("gdalinfo", "gdalbuildvrt"):
        conda_candidate = Path(sys.executable).resolve().parent / executable
        path = str(conda_candidate) if conda_candidate.is_file() else shutil.which(executable)
        if path is None:
            gdal_error = f"{executable} is not on PATH"
            break
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            gdal_error = (result.stderr or result.stdout or f"{executable} failed").strip()
            break
        gdal_versions.append((result.stdout or result.stderr).strip())
    add_check(
        checks,
        "dependency_gdal_cli",
        "ready" if not gdal_error else "error",
        "; ".join(gdal_versions) if gdal_versions else "not available",
        f"Conda environment {args.conda_env}",
        gdal_error or "gdalinfo and gdalbuildvrt are available; Python osgeo is not required",
        f"install libgdal-core=3.12.3 in Conda environment {args.conda_env}",
    )

    pytest_module, pytest_version, pytest_error = import_dependency("pytest")
    add_check(
        checks,
        "developer_pytest",
        "ready" if pytest_module is not None else "warning",
        f"pytest {pytest_version}",
        f"Conda environment {args.conda_env}",
        pytest_error or "repository test runner is available",
        f"install pytest=9.1.1 in Conda environment {args.conda_env}",
    )

    config = {}
    issues = []
    if not config_path.is_file():
        add_check(
            checks,
            "config_yaml",
            "error",
            str(config_path),
            "inference scripts directory",
            "config.yaml does not exist",
            "create a Schema v2 config.yaml",
        )
    else:
        try:
            config = load_yaml(config_path)
            effective, issues = validate_deployment_config(
                config,
                scripts_dir=scripts_dir,
                verify_files=True,
                verify_hashes=True,
            )
            add_check(
                checks,
                "config_yaml",
                "ready" if config.get("schema_version") == 2 else "error",
                str(config_path),
                "config.yaml:schema_version",
                "Schema v2 parsed" if config.get("schema_version") == 2 else "Schema v2 is required",
                "replace legacy model.semantic_weight config with Schema v2",
            )
        except Exception as exc:
            add_check(
                checks,
                "config_yaml",
                "error",
                str(config_path),
                "config.yaml",
                f"cannot parse config: {exc}",
                "fix the reported YAML syntax or field",
            )

    for index, issue in enumerate(issues):
        add_check(
            checks,
            _issue_id(issue.path, index),
            "error",
            issue.path,
            f"config.yaml{issue.path}",
            issue.message,
            f"edit config.yaml field {issue.path}",
        )

    torch_module = dependencies.get("torch")
    runtime = effective.get("runtime") or {}
    requested_device = str(runtime.get("requested_device") or "auto")
    resolved_device = resolve_device(requested_device)
    device_ok = validate_device(resolved_device)
    runtime["effective_device"] = resolved_device
    effective["runtime"] = runtime
    device_status = "ready" if device_ok else "error"
    device_message = ""
    if not device_ok:
        device_message = f"requested device is unavailable: {resolved_device}"
    elif requested_device == "auto" and resolved_device == "cpu":
        device_status = "warning"
        device_message = "CUDA and MPS are unavailable; auto selected CPU"
    add_check(
        checks,
        "semantic_device",
        device_status,
        resolved_device,
        "config.yaml:runtime.device",
        device_message,
        "edit runtime.device or repair the PyTorch device environment",
    )

    for index, model in enumerate(effective.get("semantic_models") or []):
        model_id = model["model_id"]
        prefix = f"/semantic_models/{index}"
        path = model.get("artifact_path", "")
        blocked = _has_issue(issues, prefix)
        if blocked:
            status = "error"
            message = "model registry entry or deployment asset is invalid"
            fix = f"check the config entry, artifact, and SHA256 for {model_id}"
        elif torch_module is None or not device_ok:
            status = "error"
            message = "PyTorch or effective device is unavailable"
            fix = "repair the PyTorch environment or select an available device"
        elif str(resolved_device).startswith("mps") and not _mps_runtime_requirement(
            model_id, getattr(torch_module, "__version__", "0")
        )[0]:
            status = "error"
            _compatible, message = _mps_runtime_requirement(
                model_id, getattr(torch_module, "__version__", "0")
            )
            fix = "upgrade torch to >=2.7 with matching torchvision and torchaudio in the configured Conda environment"
        else:
            if str(resolved_device).startswith(("mps", "cuda")):
                ok, message = verify_torchscript_contract_isolated(path, resolved_device)
            else:
                ok, message = verify_torchscript_contract(torch_module, path, resolved_device)
            status = "ready" if ok else "error"
            if not ok and str(resolved_device).startswith("mps"):
                fix = (
                    "inspect inference_scripts/torchscript_runtime.py MPS graph compatibility; "
                    "the hash-valid artifact may still be valid on CPU/CUDA"
                )
            else:
                fix = f"deploy and register a valid TorchScript artifact for {model_id}"
        add_check(
            checks,
            f"semantic_model_{model_id}",
            status,
            f"{model_id}: {path}",
            f"config.yaml:semantic_models[{index}]",
            message,
            fix,
        )

    for index, profile_entry in enumerate(effective.get("fusion_profiles") or []):
        profile_id = profile_entry["profile_id"]
        prefix = f"/fusion_profiles/{index}"
        if _has_issue(issues, prefix):
            status = "error"
            message = "profile file, schema, model reference, or hash is invalid"
        elif profile_entry.get("status") == "rejected":
            status = "warning"
            message = "profile is rejected and cannot be selected for formal fusion"
        elif profile_entry.get("available"):
            status = "ready"
            message = (
                f"approved {profile_entry.get('strategy')} profile; "
                f"models={','.join(profile_entry.get('required_model_ids') or [])}"
            )
        else:
            status = "error"
            message = "profile is not available for formal fusion"
        add_check(
            checks,
            f"fusion_profile_{profile_id}",
            status,
            profile_entry.get("file_path", ""),
            f"config.yaml:fusion_profiles[{index}]",
            message,
            f"deploy an approved and matching fusion profile for {profile_id}",
        )

    classes = effective.get("classes") or {}
    class_ok = (
        classes.get("background_index") == -1
        and [int((classes.get("index_to_code") or {}).get(str(index), -999)) for index in range(14)] == CLASS_ORDER
    )
    add_check(
        checks,
        "class_contract",
        "ready" if class_ok else "error",
        "14 valid classes; nodata=-1",
        "config.yaml:classes",
        "" if class_ok else "class order is not the fixed 14-class contract",
        "restore the documented index_to_code mapping and background_index=-1",
    )

    sam = effective.get("sam3") or {}
    if sam.get("enabled"):
        sam_checkpoint = sam.get("checkpoint", "")
        sam_requested = str(sam.get("requested_device") or "auto")
        if sam_requested == "auto":
            sam_device = resolved_device if resolved_device.startswith("cuda") else "cpu"
        else:
            sam_device = resolve_device(sam_requested)
        if sam_device == "mps":
            sam_device = "cpu"
            sam_device_status = "warning"
            sam_device_message = "official SAM3 has no stable MPS runtime; using CPU"
        else:
            sam_device_status = "ready" if validate_device(sam_device) else "error"
            sam_device_message = "" if sam_device_status == "ready" else f"SAM3 device unavailable: {sam_device}"
        sam["effective_device"] = sam_device
        effective["sam3"] = sam
        add_check(
            checks,
            "sam3_device",
            sam_device_status,
            sam_device,
            "config.yaml:sam3.device",
            sam_device_message,
            "use CUDA or CPU for SAM3",
        )

        try:
            installed = importlib.metadata.version("sam3")
            backend_ok, backend_value, backend_error = True, f"official sam3 {installed}", ""
        except importlib.metadata.PackageNotFoundError:
            backend_ok, backend_value, backend_error = False, "not installed", "official sam3 package is missing"
        add_check(
            checks,
            "sam3_backend",
            "ready" if backend_ok else "error",
            backend_value,
            f"Conda environment {args.conda_env}",
            backend_error,
            "install the official sam3 package",
        )

        tokenizer = scripts_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        add_check(
            checks,
            "sam3_tokenizer",
            "ready" if tokenizer.is_file() else "error",
            str(tokenizer),
            "inference_scripts/assets",
            "" if tokenizer.is_file() else "SAM3 tokenizer asset is missing",
            "deploy the official tokenizer asset",
        )
        if torch_module is not None and Path(sam_checkpoint).is_file():
            ok, message = verify_sam3_checkpoint(torch_module, sam_checkpoint)
            if ok and backend_ok and sam_device_status != "error":
                runtime_ok, runtime_message = verify_sam3_runtime_isolated(
                    sam_checkpoint, sam_device
                )
                ok = runtime_ok
                message = f"{message}; {runtime_message}"
        else:
            ok, message = False, "SAM3 checkpoint or PyTorch is unavailable"
        add_check(
            checks,
            "sam3_model_load",
            "ready" if ok else "error",
            sam_checkpoint,
            "config.yaml:sam3.checkpoint",
            message,
            "deploy a valid official SAM3 checkpoint",
        )
    else:
        add_check(
            checks,
            "sam3_enabled",
            "warning",
            "disabled",
            "config.yaml:sam3.enabled",
            "semantic inference remains available; class refinement is disabled",
            "enable SAM3 only when its deployment assets are ready",
        )

    output_dir = str(args.output_dir or "").strip()
    if output_dir:
        output_path = Path(output_dir).expanduser().resolve()
        probe = output_path if output_path.exists() else output_path.parent
        writable = probe.exists() and os.access(str(probe), os.W_OK)
        add_check(
            checks,
            "output_dir",
            "ready" if writable else "error",
            str(output_path),
            "main panel:output workspace",
            "" if writable else "output directory or parent is not writable",
            "select a writable output workspace",
        )

    status = overall_status(checks)
    return {
        "schema_version": 1,
        "status": status,
        "config_fingerprint": _fingerprint(scripts_dir),
        "effective": effective,
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate Schema v2 inference deployment")
    parser.add_argument("--scripts-dir")
    parser.add_argument("--conda-env")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--report-json", default="")
    parser.add_argument("--contract-worker", action="store_true")
    parser.add_argument("--sam3-worker", action="store_true")
    parser.add_argument("--model-path")
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.contract_worker:
        if not args.model_path or not args.device:
            parser.error("--contract-worker requires --model-path and --device")
        import torch

        ok, message = verify_torchscript_contract(torch, args.model_path, args.device)
        sys.stdout.write(json.dumps({"ok": ok, "message": message}) + "\n")
        return 0
    if args.sam3_worker:
        if not args.model_path or not args.device:
            parser.error("--sam3-worker requires --model-path and --device")
        try:
            from sam3_refine import load_sam3

            runtime = load_sam3(args.model_path, args.device)
            runtime = None
            gc.collect()
            ok, message = True, f"official SAM3 runtime loaded on {args.device}"
        except Exception as exc:
            ok = False
            message = f"official SAM3 runtime failed on {args.device}: {exc}"
        sys.stdout.write(json.dumps({"ok": ok, "message": message}) + "\n")
        return 0 if ok else 2
    if not args.scripts_dir or not args.conda_env:
        parser.error("--scripts-dir and --conda-env are required")
    try:
        report = build_report(args)
    except Exception as exc:
        report = {
            "schema_version": 1,
            "status": "error",
            "config_fingerprint": "",
            "effective": {},
            "checks": [{
                "id": "environment_check",
                "status": "error",
                "value": "failed",
                "source": "check_environment.py",
                "message": str(exc),
                "fix": "inspect the detailed environment log",
            }],
        }
    serialized = json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n"
    if args.report_json:
        report_path = Path(args.report_json).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = report_path.with_name(report_path.name + ".tmp")
        temporary_path.write_text(serialized, encoding="utf-8")
        os.replace(temporary_path, report_path)
    sys.stdout.write(serialized)
    return 0 if report["status"] != "error" else 2


if __name__ == "__main__":
    raise SystemExit(main())
