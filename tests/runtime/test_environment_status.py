import json
import sys

import loess_runtime.system.check_environment as check_environment
import loess_runtime.system.environment_checks as environment_checks
import loess_runtime.system.environment_report as environment_report
from loess_runtime.system.environment_checks import (
    append_runtime_boundary_checks,
    mps_runtime_requirement,
    overall_status,
)


def test_old_pyogrio_is_rejected_before_importing_deprecated_shapely_geos(
    monkeypatch,
):
    imports = []
    monkeypatch.setattr(
        environment_checks.importlib.metadata,
        "version",
        lambda _name: "0.10.0",
    )
    monkeypatch.setattr(
        environment_checks.importlib,
        "import_module",
        lambda name: imports.append(name),
    )

    module, version, error = environment_checks.import_dependency("pyogrio")

    assert module is None
    assert version == "0.10.0"
    assert "pyogrio ==0.13.0" in error
    assert imports == []


def test_wrong_pyarrow_version_is_rejected_before_import(monkeypatch):
    imports = []
    monkeypatch.setattr(
        environment_checks.importlib.metadata,
        "version",
        lambda _name: "19.0.1",
    )
    monkeypatch.setattr(
        environment_checks.importlib,
        "import_module",
        lambda name: imports.append(name),
    )

    module, version, error = environment_checks.import_dependency("pyarrow")

    assert module is None
    assert version == "19.0.1"
    assert "pyarrow ==25.0.1" in error
    assert imports == []


def test_swin_mps_requires_pytorch_27_but_other_devices_are_unchanged():
    assert mps_runtime_requirement("upernet_swin_b", "2.5.1")[0] is False
    assert mps_runtime_requirement("upernet_swin_b", "2.7.0")[0] is True
    assert mps_runtime_requirement("setr_vit", "2.5.1")[0] is True


def _check(check_id, status, source=""):
    return {"id": check_id, "status": status, "source": source}


def test_optional_fusion_and_sam_errors_do_not_block_runnable_semantic_model():
    checks = [
        _check("config_yaml", "ready"),
        _check("semantic_model_a", "ready"),
        _check("semantic_model_b", "error"),
        _check("fusion_profile_f", "error"),
        _check("sam3_backend", "error"),
    ]
    assert overall_status(checks) == "warning"


def test_missing_sam3_checkpoint_is_optional_when_models_are_runnable():
    checks = [
        _check("config_yaml", "ready"),
        _check("semantic_model_a", "ready"),
        _check("sam3_model_load", "error"),
    ]
    assert overall_status(checks) == "warning"


def test_no_runnable_model_or_core_error_blocks():
    assert overall_status([_check("semantic_model_a", "error")]) == "error"
    assert (
        overall_status(
            [
                _check("semantic_model_a", "ready"),
                _check("dependency_rasterio", "error"),
            ]
        )
        == "error"
    )


def test_runtime_boundary_checks_accept_qgis_42_pyqt6_qt6(monkeypatch):
    checks = []
    monkeypatch.setenv("LOESS_QGIS_VERSION", "4.2.2-Belem do Para")
    monkeypatch.setenv("LOESS_QGIS_PYTHON_VERSION", "3.14.4")
    monkeypatch.setenv("LOESS_PYQT_VERSION", "6.10.2")
    monkeypatch.setenv("LOESS_QT_VERSION", "6.10.2")
    monkeypatch.setenv("LOESS_QGIS_PYTHON_EXECUTABLE", "/usr/bin/python3")
    monkeypatch.setattr(
        environment_checks.sys,
        "executable",
        "/opt/conda/envs/qgis/bin/python",
        raising=False,
    )

    append_runtime_boundary_checks(checks, "qgis")

    by_id = {item["id"]: item for item in checks}
    assert by_id["qgis_version"]["status"] == "ready"
    assert by_id["pyqt_version"]["status"] == "ready"
    assert by_id["qt_version"]["status"] == "ready"
    assert "compatibility" in by_id["qgis_version"]["message"].lower()


def test_runtime_boundary_checks_reject_the_retired_qgis3_qt5_profile(monkeypatch):
    checks = []
    monkeypatch.setenv("LOESS_QGIS_VERSION", "3.44.7-Solothurn")
    monkeypatch.setenv("LOESS_QGIS_PYTHON_VERSION", "3.12.5")
    monkeypatch.setenv("LOESS_PYQT_VERSION", "5.15.10")
    monkeypatch.setenv("LOESS_QT_VERSION", "5.15.13")

    append_runtime_boundary_checks(checks, "qgis")

    by_id = {item["id"]: item for item in checks}
    assert by_id["qgis_version"]["status"] == "error"
    assert by_id["pyqt_version"]["status"] == "error"
    assert by_id["qt_version"]["status"] == "error"
    assert by_id["qgis_qt_profile"]["status"] == "error"


def test_main_persists_current_check_identity_before_stdout(
    monkeypatch,
    tmp_path,
    capsys,
):
    report_path = tmp_path / "environment-check.json"
    monkeypatch.setenv("LOESS_ENV_CHECK_ID", "check-123")
    monkeypatch.setenv(
        "LOESS_ENV_CHECK_STARTED_AT",
        "2026-09-04T01:00:00+00:00",
    )
    monkeypatch.setattr(
        environment_report,
        "build_environment_report",
        lambda **_kwargs: {
            "schema_version": 1,
            "status": "ready",
            "config_fingerprint": "abc",
            "effective": {},
            "checks": [],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_environment.py",
            "--scripts-dir",
            str(tmp_path),
            "--conda-env",
            "qgis",
            "--report-json",
            str(report_path),
        ],
    )

    assert check_environment.main() == 0

    stdout_report = json.loads(capsys.readouterr().out)
    disk_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert stdout_report == disk_report
    assert disk_report["check_id"] == "check-123"
    assert disk_report["started_at"] == "2026-09-04T01:00:00+00:00"
