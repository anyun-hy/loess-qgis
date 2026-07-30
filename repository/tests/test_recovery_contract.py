from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from labeling_tool.core import recovery_contract
from labeling_tool.core.run_spec import sha256_file
from labeling_tool.core.run_state_db import RunStateDB


RUN_ID = "20260730_120000_abcd"


def _content_sha(spec):
    encoded = json.dumps(
        spec,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_fixture(tmp_path: Path, fingerprint: str, *, database_sha=None):
    output_root = tmp_path / "output"
    run_dir = output_root / "runs" / RUN_ID
    run_dir.mkdir(parents=True)
    spec_path = run_dir / "run_spec.json"
    state_path = run_dir / "run_state.sqlite"
    spec = {
        "schema_version": 2,
        "run_id": RUN_ID,
        "run_dir": str(run_dir),
        "output_root": str(output_root),
        "state_db": str(state_path),
        "config_fingerprint": fingerprint,
    }
    spec["run_spec_content_sha256"] = _content_sha(spec)
    spec_path.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    database = RunStateDB(state_path)
    database.initialize()
    database.create_run(
        RUN_ID,
        database_sha or sha256_file(spec_path),
        status="failed",
        metadata={"run_spec": str(spec_path)},
    )
    return spec_path, state_path


def _ready_deployment(monkeypatch, fingerprint):
    monkeypatch.setattr(
        recovery_contract,
        "verify_project_runtime",
        lambda *_args, **_kwargs: {"status": "ready", "message": ""},
    )
    monkeypatch.setattr(
        recovery_contract,
        "deployment_fingerprint",
        lambda *_args: fingerprint,
    )


def test_valid_recovery_contract_returns_bound_spec_and_database(
    tmp_path,
    monkeypatch,
):
    fingerprint = "sha256:" + "a" * 64
    spec_path, state_path = _run_fixture(tmp_path, fingerprint)
    _ready_deployment(monkeypatch, fingerprint)

    spec, database, validated_path = recovery_contract.validate_recovery_run(
        spec_path,
        tmp_path / "project" / "inference_scripts",
    )

    assert spec["run_id"] == RUN_ID
    assert database.path == state_path
    assert validated_path == spec_path


def test_recovery_rejects_changed_deployment_before_opening_database(
    tmp_path,
    monkeypatch,
):
    fingerprint = "sha256:" + "a" * 64
    spec_path, _state_path = _run_fixture(tmp_path, fingerprint)
    monkeypatch.setattr(
        recovery_contract,
        "verify_project_runtime",
        lambda *_args, **_kwargs: {
            "status": "error",
            "message": "inference file changed",
        },
    )
    opened = []
    monkeypatch.setattr(
        recovery_contract,
        "RunStateDB",
        lambda *_args: opened.append("opened"),
    )

    with pytest.raises(
        recovery_contract.RecoveryContractError,
        match="部署一致性检查失败",
    ):
        recovery_contract.validate_recovery_run(
            spec_path,
            tmp_path / "project" / "inference_scripts",
        )

    assert opened == []


def test_recovery_rejects_changed_fingerprint_before_opening_database(
    tmp_path,
    monkeypatch,
):
    fingerprint = "sha256:" + "a" * 64
    spec_path, _state_path = _run_fixture(tmp_path, fingerprint)
    _ready_deployment(monkeypatch, "sha256:" + "b" * 64)
    opened = []
    monkeypatch.setattr(
        recovery_contract,
        "RunStateDB",
        lambda *_args: opened.append("opened"),
    )

    with pytest.raises(
        recovery_contract.RecoveryContractError,
        match="部署指纹",
    ):
        recovery_contract.validate_recovery_run(
            spec_path,
            tmp_path / "project" / "inference_scripts",
        )
    assert opened == []


def test_recovery_rejects_run_identity_before_opening_database(
    tmp_path,
    monkeypatch,
):
    fingerprint = "sha256:" + "a" * 64
    spec_path, _state_path = _run_fixture(tmp_path, fingerprint)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["run_dir"] = str(tmp_path / "different-run")
    spec.pop("run_spec_content_sha256")
    spec["run_spec_content_sha256"] = _content_sha(spec)
    spec_path.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _ready_deployment(monkeypatch, fingerprint)
    opened = []
    monkeypatch.setattr(
        recovery_contract,
        "RunStateDB",
        lambda *_args: opened.append("opened"),
    )

    with pytest.raises(
        recovery_contract.RecoveryContractError,
        match="身份不一致",
    ):
        recovery_contract.validate_recovery_run(
            spec_path,
            tmp_path / "project" / "inference_scripts",
        )
    assert opened == []


def test_recovery_rejects_database_spec_identity(
    tmp_path,
    monkeypatch,
):
    fingerprint = "sha256:" + "a" * 64
    other_root = tmp_path / "second"
    other_root.mkdir()
    bad_spec_path, _bad_state = _run_fixture(
        other_root,
        fingerprint,
        database_sha="0" * 64,
    )
    _ready_deployment(monkeypatch, fingerprint)
    with pytest.raises(
        recovery_contract.RecoveryContractError,
        match="Run Spec SHA256",
    ):
        recovery_contract.validate_recovery_run(
            bad_spec_path,
            tmp_path / "project" / "inference_scripts",
        )
