from scale_acceptance import _final_artifact_size_observation


def test_final_artifact_observation_reports_a_signed_difference_without_a_gate():
    observation = _final_artifact_size_observation(
        {
            "final_artifact_size_prediction": {
                "status": "predicted",
                "observation_only": True,
                "predicted_final_artifact_bytes": 100,
            }
        },
        125,
    )

    assert observation["status"] == "predicted"
    assert observation["actual_final_artifact_bytes"] == 125
    assert observation["signed_difference_bytes"] == 25
    assert observation["signed_difference_ratio"] == 0.25


def test_final_artifact_observation_leaves_an_uncalibrated_run_descriptive():
    observation = _final_artifact_size_observation({}, 125)

    assert observation == {
        "status": "not_available",
        "actual_final_artifact_bytes": 125,
        "observation_only": True,
    }
