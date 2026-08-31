from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "qgis_plugins/labeling_tool/core/v5_async_runner.py"
).read_text(encoding="utf-8")


def _format_observation(observation):
    start = SOURCE.index("def _final_artifact_size_observation_log_message")
    end = SOURCE.index("\n\nclass V5AsyncInferenceRunner", start)
    namespace = {}
    exec(SOURCE[start:end], namespace)
    return namespace["_final_artifact_size_observation_log_message"](observation)


def _format_prediction(prediction):
    start = SOURCE.index("def _final_artifact_size_prediction_log_message")
    end = SOURCE.index("\n\ndef _final_artifact_size_observation_log_message", start)
    namespace = {}
    exec(SOURCE[start:end], namespace)
    return namespace["_final_artifact_size_prediction_log_message"](prediction)


def test_user_facing_summary_contains_only_this_runs_expected_actual_and_difference():
    summary = _format_observation(
        {
            "predicted_final_artifact_bytes": 5 * 1024**3,
            "actual_final_artifact_bytes": 4 * 1024**3,
            "signed_difference_bytes": -(1024**3),
            "signed_difference_ratio": -0.2,
        }
    )

    assert summary == "本次 Run：预计最终保存 5.00 GiB；实际最终保存 4.00 GiB；差额 −1.00 GiB（-20.00%）"
    assert "202608" not in summary


def test_user_facing_summary_reports_actual_size_when_a_prediction_is_unavailable():
    assert _format_observation(
        {"actual_final_artifact_bytes": 4 * 1024**3}
    ) == "本次 Run 最终保存 4.00 GiB"


def test_monitor_log_announces_the_frozen_prediction_at_run_start():
    assert _format_prediction(
        {"predicted_final_artifact_bytes": 5 * 1024**3}
    ) == "本次 Run 预计最终保存 5.00 GiB；完成后回报实际值和差额"
