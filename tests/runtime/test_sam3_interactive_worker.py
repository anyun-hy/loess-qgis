import hashlib
import io

import pytest

import loess_runtime.sam.sam3_interactive_worker as worker


def test_worker_enforces_explicit_session_protocol(tmp_path, monkeypatch):
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(b"checkpoint").hexdigest()
    events = []
    monkeypatch.setattr(worker, "load_sam3", lambda _path, _device: object())
    monkeypatch.setattr(worker, "resolve_sam3_device", lambda _device: "cpu")
    monkeypatch.setattr(
        worker,
        "_predict",
        lambda _runtime, request: {"session_id": request["session_id"]},
    )
    monkeypatch.setattr(
        worker,
        "emit",
        lambda event, **payload: events.append({"event": event, **payload}),
    )
    monkeypatch.setattr(
        worker.sys,
        "stdin",
        io.StringIO(
            "\n".join([
                '{"command":"start_session","session_id":"s1"}',
                '{"command":"predict","session_id":"s1",'
                f'"checkpoint_sha256":"{digest}","sam_version":"v1","device":"cpu"}}',
                '{"command":"close_session","session_id":"s1"}',
                '{"command":"shutdown"}',
            ])
            + "\n"
        ),
    )

    worker.run_worker(checkpoint, digest, "cpu", "v1")

    names = [event["event"] for event in events]
    assert names[0] == "worker_ready"
    assert "session_started" in names
    assert "started" in names
    assert "session_closed" in names
    assert names[-2:] == ["worker_stopping", "worker_stopped"]


def test_worker_rejects_predict_without_open_session(tmp_path, monkeypatch):
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(b"checkpoint").hexdigest()
    events = []
    monkeypatch.setattr(worker, "load_sam3", lambda _path, _device: object())
    monkeypatch.setattr(worker, "resolve_sam3_device", lambda _device: "cpu")
    monkeypatch.setattr(
        worker,
        "emit",
        lambda event, **payload: events.append({"event": event, **payload}),
    )
    monkeypatch.setattr(
        worker.sys,
        "stdin",
        io.StringIO(
            '{"command":"predict","session_id":"missing"}\n'
            '{"command":"shutdown"}\n'
        ),
    )

    worker.run_worker(checkpoint, digest, "cpu", "v1")

    failures = [event for event in events if event["event"] == "failed"]
    assert failures
    assert "open SAM3 session" in failures[0]["error"]


@pytest.mark.parametrize("enabled", [True, False])
def test_config_entry_always_loads_sam_without_semantic_models(tmp_path, monkeypatch, enabled):
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config = tmp_path / "config.yaml"
    config.write_text(f"sam3:\n  enabled: {str(enabled).lower()}\n  checkpoint: sam3.pt\n  sha256: {digest}\n  device: cpu\n  buffer_px: 24\n")
    loaded, events = [], []
    monkeypatch.setattr(worker, "load_sam3", lambda path, device: loaded.append((path, device)))
    monkeypatch.setattr(worker, "resolve_sam3_device", lambda device: device)
    monkeypatch.setattr(worker, "emit", lambda event, **data: events.append((event, data)))
    monkeypatch.setattr(worker.sys, "stdin", io.StringIO('{"command":"shutdown"}\n'))
    assert worker.main(["--config", str(config)]) == 0
    assert loaded == [(str(checkpoint), "cpu")]
    assert events[0][0] == "worker_ready"
    assert events[0][1]["checkpoint_sha256"] == digest
    assert events[0][1]["buffer_px"] == 24


def test_invalid_sam_hash_does_not_report_ready(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("sam3:\n  enabled: false\n  checkpoint: missing.pt\n")
    events = []
    monkeypatch.setattr(worker, "emit", lambda event, **data: events.append((event, data)))
    assert worker.main(["--config", str(config)]) == 2
    assert [event for event, _ in events] == ["worker_failed"]
