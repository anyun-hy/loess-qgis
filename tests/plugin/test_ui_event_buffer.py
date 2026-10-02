"""Pure queue behavior for the runner's UI event buffer."""

from labeling_tool.runs.ui_event_buffer import UIEventBuffer


def test_logs_are_lossless_and_stream_progress_is_coalesced_with_priority():
    buffer = UIEventBuffer()
    buffer.enqueue_log({"message": "first"})
    buffer.enqueue_log({"message": "second"})
    for current in range(3):
        buffer.enqueue_stream_progress(
            {
                "event": "fit_progress",
                "stream_id": "model:a",
                "current": current,
            }
        )
    buffer.enqueue_stream_progress(
        {
            "event": "fit_failed",
            "stream_id": "model:a",
            "status": "failed",
        }
    )
    buffer.enqueue_stream_progress(
        {
            "event": "fit_warning",
            "stream_id": "model:a",
            "status": "warning",
        }
    )
    buffer.set_pipeline_progress(1, 3, "first fit value")
    buffer.set_pipeline_progress(2, 3, "fit")

    assert [item["message"] for item in buffer.take_logs()] == ["first", "second"]
    stream_events, pipeline = buffer.take_progress()
    assert [item["event"] for item in stream_events] == [
        "fit_failed",
        "fit_warning",
        "fit_progress",
    ]
    assert stream_events[-1]["current"] == 2
    assert pipeline == (2, 3, "fit")
    assert buffer.take_logs() == []
    assert buffer.take_progress() == ([], None)


def test_drain_order_keeps_progress_enqueued_during_log_emission_current():
    buffer = UIEventBuffer()
    buffer.enqueue_log({"message": "first"})

    logs = buffer.take_logs()
    assert logs == [{"message": "first"}]
    buffer.enqueue_stream_progress(
        {"event": "fit_progress", "stream_id": "model:a", "current": 1}
    )
    buffer.set_pipeline_progress(1, 3, "fit")

    stream_events, pipeline = buffer.take_progress()
    assert stream_events == [
        {"event": "fit_progress", "stream_id": "model:a", "current": 1}
    ]
    assert pipeline == (1, 3, "fit")
