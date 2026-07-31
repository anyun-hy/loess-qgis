import hashlib
import io

import sam3_interactive_worker as worker


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
