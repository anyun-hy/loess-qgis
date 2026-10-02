from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from loess_runtime.geometry.fragmentation_v33_candidate.candidate import (
    apply_v33_candidate,
)
from loess_runtime.geometry.fragmentation_v33_candidate.contracts import (
    resolve_class_budget_mask,
)
from loess_runtime.geometry.fragmentation_v33_candidate.engine import (
    apply_v31a_candidate,
    apply_v31b_candidate,
    v31a_policy,
)

CLASS_CODES = (12, 13, 21, 31, 32, 33, 43, 51, 52, 53, 54, 61, 62, 71)
INDEX = {code: index for index, code in enumerate(CLASS_CODES)}


def _json_sha256(value: object) -> str:
    body = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(body.encode()).hexdigest()


def _v33_probabilities(labels: np.ndarray) -> np.ndarray:
    probabilities = np.full(
        (len(CLASS_CODES), *labels.shape),
        0.01 / (len(CLASS_CODES) - 1),
        dtype=np.float32,
    )
    for index in range(len(CLASS_CODES)):
        probabilities[index, labels == index] = 0.99
    return probabilities


def test_frozen_policy_and_budget_mask_keep_value_and_owner_contracts():
    policy = v31a_policy()
    with pytest.raises(FrozenInstanceError):
        policy.maximum_source_loss_fraction = 0.5  # type: ignore[misc]

    valid = np.array([[True, True], [False, True]])
    default_budget = resolve_class_budget_mask(None, valid)
    default_budget[0, 0] = False
    assert valid[0, 0]

    owner = np.array([[True, False], [True, True]])
    assert np.array_equal(
        resolve_class_budget_mask(owner, valid),
        np.array([[True, False], [False, True]]),
    )


@pytest.mark.parametrize(
    ("executor", "audit_sha256"),
    [
        (
            apply_v31a_candidate,
            "bb889fbd9d4ecf1ec1289ee06faa24751de1c68e4c2eb524e7745d8d18b81415",
        ),
        (
            apply_v31b_candidate,
            "80c27430a85b6f3ff306dcc534fd30f5ddd2749fed340901888f4260bb4642de",
        ),
    ],
)
def test_v31_full_result_and_audit_match_pre_split_baseline(
    executor,
    audit_sha256,
):
    """Freeze the complete non-zero Git 77b201e baseline result and audit."""

    class_codes = (13, 43)
    labels = np.full((30, 30), 1, dtype=np.int16)
    labels[2:9, 2:9] = 0
    labels[15, 15] = 0
    labels[20, 20] = 0
    probabilities = np.full((2, *labels.shape), 0.01, dtype=np.float32)
    probabilities[1] = 0.99
    probabilities[0, labels == 0] = 0.54
    probabilities[1, labels == 0] = 0.46

    result, audit = executor(
        labels,
        class_codes=class_codes,
        pixel_area_m2=1.0,
        pixel_size_m=(1.0, 1.0),
        valid_mask=np.ones(labels.shape, dtype=bool),
        class_budget_mask=np.ones(labels.shape, dtype=bool),
        probabilities=probabilities,
        confidence=None,
        baseline_kind="v3_cleaned",
        full_audit=True,
    )

    assert audit["proposals_accepted"] == 1
    assert audit["proposal_reject_reason_counts"] == {"source_budget": 2}
    assert hashlib.sha256(result.tobytes()).hexdigest() == (
        "a1ddd5a315022029177a9f03dcdbcca77d0c90564957874147532ddc31b0f67c"
    )
    assert _json_sha256(audit) == audit_sha256


def test_v33_full_result_and_audit_match_pre_split_baseline():
    """Freeze the complete non-zero Git 77b201e baseline result and audit."""

    labels = np.full((30, 30), INDEX[52], dtype=np.int16)
    labels[2:9, 2:9] = INDEX[13]
    labels[15, 15] = INDEX[13]
    labels[0, 20] = INDEX[13]

    result, audit = apply_v33_candidate(
        labels,
        class_codes=CLASS_CODES,
        pixel_area_m2=1.0,
        pixel_size_m=(1.0, 1.0),
        valid_mask=np.ones(labels.shape, dtype=bool),
        class_budget_mask=np.ones(labels.shape, dtype=bool),
        probabilities=_v33_probabilities(labels),
        confidence=None,
        baseline_kind="v3_cleaned",
        full_audit=True,
    )

    assert audit["proposals_accepted"] == 1
    assert audit["proposal_reject_reason_counts"] == {"source_budget": 1}
    assert audit["proposal_generation_reject_reason_counts"] == {
        "external_or_invalid_boundary": 1,
        "protected_source": 1,
    }
    assert hashlib.sha256(result.tobytes()).hexdigest() == (
        "c76355be906c38abbbb6702f7494b89edf9dce3a006c5e94bb42866d360debb9"
    )
    assert _json_sha256(audit) == (
        "aea0c7613046488493ed90dbc8bd520c095767e07749d4c458057b855e40da4c"
    )
