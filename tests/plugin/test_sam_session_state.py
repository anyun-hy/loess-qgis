"""Session identity, isolated payloads and persisted history without QGIS."""

import pytest

from labeling_tool.refinement.sam_session import SamSession


def queue(session):
    session.queue_request(
        run_id="run",
        raster="raster.tif",
        confidence_mosaic="confidence.tif",
        crop_size_px=512,
        buffer_px=32,
        checkpoint_sha256="a" * 64,
        sam_version="version",
        device="cpu",
    )


def history(session, decision):
    return session.history_record(
        decision,
        run_id="run",
        baseline_stream_id="fusion:probe",
        checkpoint_sha256="a" * 64,
        sam_version="version",
        device="cpu",
        geometry_hash=lambda _: "candidate",
    )


def prepare(mode="existing"):
    session = SamSession.start(12, mode, "start")
    session.begin_inference(
        click_raster={"x": 2.0, "y": 3.0},
        geometry_bounds=None,
        **(
            {
                "feature_id": 7,
                "object_id": "object",
                "part_id": "003",
                "current_geometry_hash": "before",
                "current_source": "fusion",
                "current_revision": 3,
            }
            if mode == "existing"
            else {}
        ),
    )
    return session


def accept(session):
    assert session.accept_candidate(
        session_id=session.session_id,
        geometry=object(),
        score=0.9,
        confidence_mean=0.8,
        confidence_std=0.1,
        crop_window={"width": 512},
        elapsed_sec=0.2,
    )


def test_request_payload_is_independent_and_consumed_once():
    session = prepare()
    queue(session)
    request = session.take_pending_request()
    request["click_raster"]["x"] = 99
    assert session.click_raster == {"x": 2.0, "y": 3.0}
    assert session.take_pending_request() is None
    assert session.fail("worker unavailable")
    old_id = session.retry("retry")
    assert old_id != session.session_id and session.error == ""
    assert session.object_id == "object" and session.feature_id == 7
    queue(session)
    assert session.take_pending_request()["click_raster"]["x"] == 2.0


@pytest.mark.parametrize(
    "decision,expected",
    [
        ("adopted", "persisted"),
        ("edit_sam3", "candidate"),
        ("edit_current", "before"),
        ("kept_current", "before"),
    ],
)
def test_history_retains_decision_specific_geometry_provenance(decision, expected):
    session = prepare()
    accept(session)
    session.persisted_geometry_hash = "persisted"
    session.topology_hint = "无"
    record = history(session, decision)
    assert record == {
        "session_id": session.session_id,
        "class_code": 12,
        "mode": "existing",
        "state": "candidate",
        "started_at": "start",
        "feature_id": 7,
        "object_id": "object",
        "part_id": "003",
        "current_geometry_hash": "before",
        "current_source": "fusion",
        "current_revision": 3,
        "click_raster": {"x": 2.0, "y": 3.0},
        "geometry_bounds": None,
        "candidate_score": 0.9,
        "confidence_mean": 0.8,
        "confidence_std": 0.1,
        "crop_window": {"width": 512},
        "elapsed_sec": 0.2,
        "topology_hint": "无",
        "persisted_geometry_hash": "persisted",
        "decision": decision,
        "run_id": "run",
        "baseline_stream_id": "fusion:probe",
        "before_geometry_hash": "before",
        "candidate_geometry_hash": "candidate",
        "after_geometry_hash": expected,
        "checkpoint_sha256": "a" * 64,
        "sam_version": "version",
        "device": "cpu",
    }


def test_missed_history_does_not_invent_existing_source_identity():
    session = prepare("missed")
    accept(session)
    session.object_id, session.feature_id = "new-object", 9
    session.persisted_geometry_hash = "persisted"
    record = history(session, "adopted")
    assert record["object_id"] == "new-object" and record["feature_id"] == 9
    assert record["before_geometry_hash"] == ""
    assert (
        not {"current_source", "current_revision", "current_geometry_hash"}
        & record.keys()
    )


def test_cancel_releases_candidate_and_prevents_reviving_old_session():
    session = prepare()
    accept(session)
    session.cancel()
    assert session.candidate_geometry is None and session.state == "cancelled"
    assert session.take_pending_request() is None
    assert not session.fail("late failure")
    assert not session.accept_candidate(
        session_id=session.session_id,
        geometry=object(),
        score=0.8,
        confidence_mean=0.0,
        confidence_std=0.0,
        crop_window=None,
        elapsed_sec=None,
    )
