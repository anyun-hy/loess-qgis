"""Collect environment check records without report or CLI dependencies."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from labeling_tool.shared.state.postgres_state import PostgresDependencyError
from labeling_tool.shared.state.run_state_db import (
    RunStateDB,
    production_state_database,
)

REQUIRED_VECTOR_DEPENDENCY_VERSIONS = {
    "pyarrow": "25.0.1",
    "pyogrio": "0.13.0",
}
Check = dict[str, str]


def add_check(
    checks: list[Check],
    check_id: str,
    status: str,
    value: object,
    source: str,
    message: str = "",
    fix: str = "",
) -> None:
    checks.append(
        {
            "id": check_id,
            "status": status,
            "value": str(value),
            "source": source,
            "message": message,
            "fix": fix,
        }
    )


def _postgresql_problem(error: Exception, conda_env: str) -> tuple[str, str]:
    """Explain a known connection failure without replacing its diagnostic."""

    detail = str(error) or type(error).__name__
    lower = detail.lower()
    sqlstate = str(getattr(error, "pgcode", "") or "")
    summary = "PostgreSQL 任务数据库检查失败"
    fix = "核对数据库连接和项目 schema，按完整错误信息处理"
    if isinstance(error, PostgresDependencyError):
        summary = "当前推理环境缺少 PostgreSQL 驱动"
        fix = f"在 Conda 环境 {conda_env} 中准备 psycopg2，再重新检查"
    elif re.search(r'role "[^"]+" does not exist', lower):
        summary = "PostgreSQL 登录角色不存在"
        fix = "准备对应的可登录角色，或用 LOESS_STATE_DB_DSN 指定已有角色"
    elif sqlstate == "3D000" or re.search(r'database "[^"]+" does not exist', lower):
        summary = "PostgreSQL 目标数据库不存在"
        fix = "准备对应数据库并授权项目角色，或用 LOESS_STATE_DB_DSN 指定已有库"
    elif sqlstate.startswith("28") or any(
        value in lower
        for value in (
            "authentication failed",
            "no pg_hba.conf entry",
            "no password supplied",
        )
    ):
        summary = "PostgreSQL 登录认证未通过"
        fix = "核对角色、本机认证规则和 libpq 凭据；不要把密码写入 DSN"
    elif sqlstate == "42501" or "permission denied" in lower:
        summary = "PostgreSQL 项目数据库或 schema 权限不足"
        fix = "由数据库管理员授予项目角色必要的 schema 创建、使用及表读写权限"
    elif (
        sqlstate.startswith("08")
        or any(
            value in lower
            for value in (
                "connection refused",
                "could not connect to server",
                "could not translate host name",
                "timeout expired",
                "connection timed out",
            )
        )
        or ("socket" in lower and "no such file or directory" in lower)
    ):
        summary = "PostgreSQL 连接不可用"
        fix = "检查服务是否运行，以及 LOESS_STATE_DB_DSN 的 socket/host 和端口"
    return (
        f"{summary}\n{detail}",
        f"{fix}。首次准备说明：docs/operations/FIRST_INSTALL.md",
    )


def append_postgresql_check(checks: list[Check], conda_env: str) -> None:
    """Check the existing control-plane contract and preserve failure evidence."""

    status, value = "ready", "not available"
    message = "PostgreSQL 任务数据库可写，schema 版本兼容"
    fix = ""
    try:
        state = RunStateDB(production_state_database())
        state.initialize()
        health = state.pragmas()
        value = (
            f"PostgreSQL {health['server_version']} / "
            f"{health['database']} / {health['schema']}"
        )
    except Exception as error:
        status = "error"
        message, fix = _postgresql_problem(error, conda_env)
    add_check(
        checks,
        "dependency_postgresql_state",
        status,
        value,
        f"Conda environment {conda_env}",
        message,
        fix,
    )


def import_dependency(name: str) -> tuple[Any | None, str, str]:
    expected_version = REQUIRED_VECTOR_DEPENDENCY_VERSIONS.get(name)
    if expected_version is not None:
        try:
            installed_version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None, "not installed", f"{name} is not installed"
        if installed_version != expected_version:
            detail = (
                "; older releases import the deprecated shapely.geos module"
                if name == "pyogrio" and version_tuple(installed_version) < (0, 11, 1)
                else ""
            )
            return (
                None,
                installed_version,
                f"{name} =={expected_version} is required by the frozen "
                f"vector data plane{detail}",
            )
    try:
        module = importlib.import_module(name)
        return module, getattr(module, "__version__", "installed"), ""
    except Exception as exc:
        return None, "not installed", str(exc)


def version_tuple(value: object) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", str(value))
    return tuple(int(item) for item in numbers[:3])


def mps_runtime_requirement(model_id: str, torch_version: str) -> tuple[bool, str]:
    if model_id == "upernet_swin_b" and version_tuple(torch_version) < (2, 7):
        return (
            False,
            f"Swin MPS requires PyTorch >=2.7 in this deployment; current={torch_version}",
        )
    return True, ""


def issue_id(path: str, index: int) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", path).strip("_").lower()
    return f"config_{normalized or 'root'}_{index}"


def has_issue(
    issues: list[Any],
    prefix: str,
    codes: tuple[str, ...] = (),
) -> bool:
    return any(
        item.path.startswith(prefix) and (not codes or item.code in codes)
        for item in issues
    )


def overall_status(checks: list[Check]) -> str:
    """Only core failures or zero runnable semantic models block inference."""

    def optional_capability_error(item: Check) -> bool:
        check_id = str(item.get("id") or "")
        source = str(item.get("source") or "")
        return check_id.startswith(
            ("semantic_model_", "fusion_profile_", "sam3_")
        ) or any(
            path in source
            for path in ("/semantic_models/", "/fusion_profiles/", "/sam3/")
        )

    core_errors = [
        item
        for item in checks
        if item["status"] == "error" and not optional_capability_error(item)
    ]
    runnable_models = [
        item
        for item in checks
        if str(item.get("id") or "").startswith("semantic_model_")
        and item["status"] == "ready"
    ]
    if core_errors or not runnable_models:
        return "error"
    if any(item["status"] in ("warning", "error") for item in checks):
        return "warning"
    return "ready"


def append_runtime_boundary_checks(checks: list[Check], conda_env: str) -> None:
    """Report both runtimes without importing QGIS into the Conda process."""
    conda_python = Path(sys.executable).resolve()
    conda_matches = f"/envs/{conda_env}/" in conda_python.as_posix()
    add_check(
        checks,
        "conda_python",
        "ready" if conda_matches else "error",
        f"{conda_python} (Python {platform.python_version()})",
        f"Conda environment {conda_env}",
        "inference runs in the configured Conda Python"
        if conda_matches
        else "environment checker is not running in the configured Conda Python",
        "edit config.sh:CONDA_EXE and CONDA_ENV",
    )

    host_fields: tuple[tuple[str, str, str, Callable[[str], bool]], ...] = (
        (
            "qgis_version",
            "LOESS_QGIS_VERSION",
            "QGIS",
            lambda value: version_tuple(value)[:2] == (4, 2),
        ),
        (
            "qgis_python",
            "LOESS_QGIS_PYTHON_VERSION",
            "QGIS Python",
            lambda value: version_tuple(value)[:1] == (3,),
        ),
        (
            "pyqt_version",
            "LOESS_PYQT_VERSION",
            "PyQt",
            lambda value: version_tuple(value)[:1] == (6,),
        ),
        (
            "qt_version",
            "LOESS_QT_VERSION",
            "Qt",
            lambda value: version_tuple(value)[:1] == (6,),
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
            "use QGIS 4.2/PyQt6/Qt6",
        )

    qgis_major = version_tuple(os.environ.get("LOESS_QGIS_VERSION", ""))[:1]
    pyqt_major = version_tuple(os.environ.get("LOESS_PYQT_VERSION", ""))[:1]
    qt_major = version_tuple(os.environ.get("LOESS_QT_VERSION", ""))[:1]
    if qgis_major and pyqt_major and qt_major:
        pair = (qgis_major[0], pyqt_major[0], qt_major[0])
        pair_ok = pair == (4, 6, 6)
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
        "use QGIS 4.2 with PyQt6 and Qt6",
    )

    host_executable = str(os.environ.get("LOESS_QGIS_PYTHON_EXECUTABLE") or "").strip()
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


def append_dependency_checks(
    checks: list[dict[str, str]],
    conda_env: str,
) -> dict[str, Any]:
    """Append installed-package, platform, database, and CLI capability checks."""

    dependencies: dict[str, Any] = {}
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
        "psycopg2",
        "pyarrow",
        "pyogrio",
    ):
        module, version, error = import_dependency(name)
        dependencies[name] = module
        add_check(
            checks,
            f"dependency_{name}",
            "ready" if module is not None else "error",
            f"{name} {version}",
            f"Conda environment {conda_env}",
            error,
            f"install {name} in Conda environment {conda_env}",
        )
    vector_backend_ok = (
        dependencies.get("pyarrow") is not None
        and dependencies.get("pyogrio") is not None
    )
    add_check(
        checks,
        "required_columnar_vector_backend",
        "ready" if vector_backend_ok else "error",
        "pyarrow+pyogrio" if vector_backend_ok else "missing",
        "columnar vector data plane",
        "unit GeoParquet and final Arrow GeoPackage output require both dependencies",
        f"install pyarrow and pyogrio in Conda environment {conda_env}",
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
        f"Conda environment {conda_env}",
        f"formal {runtime_platform} runtime uses PyTorch {expected_torch_version}"
        if torch_version_ok
        else (
            f"{runtime_platform} deployment requires exact PyTorch "
            f"{expected_torch_version}"
        ),
        "run <repository>/scripts/deploy/init_project.sh --project-root <project> --create-env",
    )
    if runtime_platform == "ubuntu":
        cuda_build = str(
            getattr(getattr(torch_module, "version", None), "cuda", "") or ""
        )
        cuda_build_ok = torch_module is not None and version_tuple(cuda_build)[:2] == (
            12,
            4,
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
        if torch_module is not None and cuda_available:
            try:
                gpu_name = str(torch_module.cuda.get_device_name(0))
                capability = ".".join(
                    str(value) for value in torch_module.cuda.get_device_capability(0)
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
            f"Conda environment {conda_env}",
            "Polygon neighbor query for common-divider fitting is available"
            if divider_available
            else "missing APIs: " + ", ".join(missing_divider_apis),
            f"install shapely=2.1.2 in Conda environment {conda_env}",
        )
    scipy_module = dependencies.get("scipy")
    scipy_spline_error = ""
    if scipy_module is not None:
        try:
            # These imports only probe availability; SciPy does not ship type stubs.
            from scipy.interpolate import (  # type: ignore[import-untyped]  # noqa: F401
                splev,
                splprep,
            )
        except Exception as exc:
            scipy_spline_error = str(exc)
    add_check(
        checks,
        "dependency_scipy_bspline",
        "ready" if scipy_module is not None and not scipy_spline_error else "error",
        f"scipy {getattr(scipy_module, '__version__', 'not installed')}",
        f"Conda environment {conda_env}",
        scipy_spline_error or "splprep and splev are available",
        f"install scipy=1.17.1 in Conda environment {conda_env}",
    )
    append_postgresql_check(checks, conda_env)
    add_check(
        checks,
        "dependency_geopackage_sqlite",
        "ready",
        f"sqlite {sqlite3.sqlite_version}",
        "Python standard library / GeoPackage",
        "SQLite remains available only for GeoPackage integrity checks",
        "",
    )
    gdal_versions = []
    gdal_error = ""
    for executable in ("gdalinfo", "gdalbuildvrt"):
        conda_candidate = Path(sys.executable).resolve().parent / executable
        path = (
            str(conda_candidate)
            if conda_candidate.is_file()
            else shutil.which(executable)
        )
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
            gdal_error = (
                result.stderr or result.stdout or f"{executable} failed"
            ).strip()
            break
        gdal_versions.append((result.stdout or result.stderr).strip())
    add_check(
        checks,
        "dependency_gdal_cli",
        "ready" if not gdal_error else "error",
        "; ".join(gdal_versions) if gdal_versions else "not available",
        f"Conda environment {conda_env}",
        gdal_error
        or "gdalinfo and gdalbuildvrt are available; Python osgeo is not required",
        f"install libgdal-core=3.12.3 in Conda environment {conda_env}",
    )
    pytest_module, pytest_version, pytest_error = import_dependency("pytest")
    add_check(
        checks,
        "developer_pytest",
        "ready" if pytest_module is not None else "warning",
        f"pytest {pytest_version}",
        f"Conda environment {conda_env}",
        pytest_error or "repository test runner is available",
        f"install pytest=9.1.1 in Conda environment {conda_env}",
    )
    return dependencies
