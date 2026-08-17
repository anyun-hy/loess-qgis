"""Frozen production policy for bounded semantic fragmentation repair.

The policy was selected by the Suide/Wubao A/B experiment.  Keeping it in a
small dependency-free module gives the completed-Run postprocessor and future
pipeline runs exactly the same semantics.
"""

from __future__ import annotations

from typing import Any, Mapping

from deployment_config import CLASS_ORDER
from small_component_regularizer import SmallComponentPolicy


POLICY_ID = "semantic_optimized_200_v3"
POLICY_VERSION = "semantic_optimized_200_v3_core_bounded_v1"
FIT_VERSION = "fragmentation_v3_v1"
DEFAULT_BUFFER_PIXELS = 256
DEFAULT_MAX_WORKERS = 4

PROTECTED_SOURCE_CLASS_CODES = frozenset({12, 33, 61, 62, 71})
PROTECTED_TARGET_CLASS_CODES = PROTECTED_SOURCE_CLASS_CODES
SEMANTIC_COMPATIBLE_TARGET_CLASS_CODES = {
    13: frozenset({21, 31, 32, 43}),
    21: frozenset({13}),
    31: frozenset({13, 32, 43}),
    32: frozenset({31, 43}),
    43: frozenset({13, 31, 32}),
    51: frozenset({52, 53, 54}),
    52: frozenset({51, 53, 54}),
    53: frozenset({51, 52, 54}),
    54: frozenset({51, 52, 53}),
}
THRESHOLDS_M2 = {
    code: (0.0 if code in PROTECTED_SOURCE_CLASS_CODES else 200.0)
    for code in CLASS_ORDER
}


def production_policy() -> SmallComponentPolicy:
    """Return the immutable V3 policy used by every production entry point."""

    return SmallComponentPolicy(
        thresholds_m2=THRESHOLDS_M2,
        protected_class_codes=PROTECTED_SOURCE_CLASS_CODES,
        allow_protected_targets=False,
        compatible_target_class_codes=SEMANTIC_COMPATIBLE_TARGET_CLASS_CODES,
        compatibility_bypass_below_m2=2.0,
        maximum_source_class_loss_fraction=0.08,
        maximum_target_class_gain_fraction=0.08,
        minimum_remaining_class_area_m2=5.0,
        hard_absorb_below_m2=2.0,
        maximum_mean_confidence=0.65,
        maximum_probability_drop=None,
        preserve_border_components=True,
        preserve_elongated_components=True,
        elongated_minimum_area_m2=10.0,
        elongated_minimum_aspect_ratio=6.0,
        elongated_maximum_mean_width_m=3.0,
    )


def policy_snapshot() -> Mapping[str, Any]:
    """Return a JSON-safe description used in manifests and fingerprints."""

    return {
        "policy_id": POLICY_ID,
        "policy_version": POLICY_VERSION,
        "fit_version": FIT_VERSION,
        "thresholds_m2": {
            str(code): float(THRESHOLDS_M2[code]) for code in CLASS_ORDER
        },
        "protected_source_class_codes": sorted(PROTECTED_SOURCE_CLASS_CODES),
        "protected_target_class_codes": sorted(PROTECTED_TARGET_CLASS_CODES),
        "compatible_target_class_codes": {
            str(source): sorted(targets)
            for source, targets in sorted(
                SEMANTIC_COMPATIBLE_TARGET_CLASS_CODES.items()
            )
        },
        "compatibility_bypass_below_m2": 2.0,
        "maximum_source_class_loss_fraction": 0.08,
        "maximum_target_class_gain_fraction": 0.08,
        "minimum_remaining_class_area_m2": 5.0,
        "hard_absorb_below_m2": 2.0,
        "maximum_mean_confidence": 0.65,
        "preserve_border_components": True,
        "preserve_elongated_components": True,
        "elongated_minimum_area_m2": 10.0,
        "elongated_minimum_aspect_ratio": 6.0,
        "elongated_maximum_mean_width_m": 3.0,
    }
