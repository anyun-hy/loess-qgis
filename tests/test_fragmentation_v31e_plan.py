from __future__ import annotations

import numpy as np
import pytest

from inference_scripts.fragmentation_v31_candidate.v31c import GlobalAction
from inference_scripts.fragmentation_v31_candidate.v31e import (
    BoundaryPlannedAction,
    V31EPlanningError,
    _window_source_connectivity_is_provable,
    action_conflicts,
    collect_global_b_discoveries,
    exact_boundary_delta,
    select_boundary_aware_actions,
)


def _action(
    action_id: str,
    footprint: tuple[tuple[int, int], ...],
    *,
    dynamic: int = 1,
    components: int = 1,
    target_code: int = 21,
) -> GlobalAction:
    return GlobalAction(
        action_id=action_id,
        kind="same_class_bridge",
        target_index=1,
        target_code=target_code,
        footprint=footprint,
        involved_core_ids=("core",),
        source_codes=(13,),
        source_anchors=footprint[:1],
        target_anchors=((0, 0),),
        dynamic_reduction=dynamic,
        component_reduction=components,
        probability_support=0.5,
        area_m2=float(len(footprint)),
        discovery_partition_ids=("core",),
        discovery_count=1,
        footprint_sha256=action_id,
        score_disagreement=False,
    )


def _planned(
    action_id: str,
    footprint: tuple[tuple[int, int], ...],
    *,
    dynamic: int,
    boundary_edges: int,
    boundary_metres: float,
    source_key: str | None = None,
) -> BoundaryPlannedAction:
    return BoundaryPlannedAction(
        action=_action(action_id, footprint, dynamic=dynamic),
        source_component_keys=((source_key or f"source:{action_id}"),),
        target_component_keys=(f"target:{action_id}",),
        source_charges=(("core", 13, len(footprint)),),
        target_charges=(("core", 21, len(footprint)),),
        component_delta_by_class=((13, 0), (21, -1)),
        dynamic_delta_by_class=((13, 0), (21, -dynamic)),
        boundary_delta_edges=boundary_edges,
        boundary_delta_metres=boundary_metres,
        dependency_proof="strict_b_component_lock_clear",
    )


def test_collects_local_b_island_with_exact_global_footprint():
    labels = np.ones((5, 5), dtype=np.int16)
    labels[2, 2] = 0
    probabilities = np.zeros((2, 5, 5), dtype=np.float32)
    probabilities[1] = 0.8
    probabilities[0] = 0.2
    probabilities[0, 2, 2] = 0.4
    probabilities[1, 2, 2] = 0.6
    discoveries, audit = collect_global_b_discoveries(
        labels,
        class_codes=(13, 21),
        pixel_area_m2=1.0,
        pixel_size_m=(1.0, 1.0),
        valid_mask=np.ones((5, 5), dtype=bool),
        probabilities=probabilities,
        confidence=probabilities.max(axis=0),
        global_origin=(100, 200),
        discovery_partition_id="core",
        owner_for_global_pixel=lambda row, col: "core"
        if 100 <= row < 105 and 200 <= col < 205
        else None,
    )
    islands = [item for item in discoveries if item.kind == "enclosed_island"]
    assert len(islands) == 1
    assert islands[0].footprint == ((102, 202),)
    assert islands[0].involved_core_ids == ("core",)
    assert audit["global_discovery_count"] >= 1


def test_exact_boundary_delta_matches_evaluator_axis_lengths():
    labels = np.full((3, 3), 21, dtype=np.int16)
    labels[1, 1] = 13
    action = _action("island", ((1, 1),), target_code=21)
    action = GlobalAction(**{**action.__dict__, "kind": "enclosed_island"})
    edges, metres = exact_boundary_delta(
        action,
        label_for_global_pixel=lambda point: int(labels[point]),
        valid_for_global_pixel=lambda point: 0 <= point[0] < 3
        and 0 <= point[1] < 3,
        owner_for_global_pixel=lambda _point: "core",
        physical_metrics_by_owner={
            "core": {"row_step_m": 2.0, "column_step_m": 3.0}
        },
    )
    assert edges == -4
    assert metres == -10.0


def test_cross_core_boundary_uses_mean_owner_step():
    labels = {(0, 0): 13, (0, 1): 21}
    action = _action("cross", ((0, 0),), target_code=21)
    edges, metres = exact_boundary_delta(
        action,
        label_for_global_pixel=lambda point: labels[point],
        valid_for_global_pixel=lambda point: point in labels,
        owner_for_global_pixel=lambda point: "left" if point == (0, 0) else "right",
        physical_metrics_by_owner={
            "left": {"row_step_m": 3.00, "column_step_m": 5.0},
            "right": {"row_step_m": 3.03, "column_step_m": 7.0},
        },
    )
    assert edges == -1
    assert metres == pytest.approx(-3.015)


def test_cross_core_boundary_rejects_abnormal_step_difference():
    labels = {(0, 0): 13, (0, 1): 21}
    action = _action("cross", ((0, 0),), target_code=21)
    with np.testing.assert_raises(V31EPlanningError):
        exact_boundary_delta(
            action,
            label_for_global_pixel=lambda point: labels[point],
            valid_for_global_pixel=lambda point: point in labels,
            owner_for_global_pixel=lambda point: (
                "left" if point == (0, 0) else "right"
            ),
            physical_metrics_by_owner={
                "left": {"row_step_m": 2.0, "column_step_m": 5.0},
                "right": {"row_step_m": 4.0, "column_step_m": 7.0},
            },
        )


def test_window_edge_removal_is_not_a_global_connectivity_proof():
    class Component:
        component_id = 1
        touches_external = True
        pixels = np.asarray(((1, 0), (1, 1), (1, 2), (1, 3)), dtype=np.int32)

    component_map = np.zeros((3, 4), dtype=np.int32)
    component_map[1] = 1
    assert not _window_source_connectivity_is_provable(
        np.asarray(((1, 3),), dtype=np.int32), component_map, (Component(),)
    )
    assert _window_source_connectivity_is_provable(
        np.asarray(((1, 2),), dtype=np.int32), component_map, (Component(),)
    )


def test_incident_footprints_and_shared_components_conflict():
    first = _planned(
        "first", ((1, 1),), dynamic=1, boundary_edges=0, boundary_metres=0.0
    )
    adjacent = _planned(
        "adjacent", ((1, 2),), dynamic=1, boundary_edges=0, boundary_metres=0.0
    )
    separate_shared = _planned(
        "shared",
        ((5, 5),),
        dynamic=1,
        boundary_edges=0,
        boundary_metres=0.0,
        source_key="source:first",
    )
    assert action_conflicts((first, adjacent, separate_shared)) == {(0, 1), (0, 2)}


def test_boundary_constrained_plan_can_use_negative_action_to_offset_positive():
    positive = _planned(
        "positive",
        ((0, 0),),
        dynamic=2,
        boundary_edges=2,
        boundary_metres=2.0,
    )
    negative = _planned(
        "negative",
        ((4, 4),),
        dynamic=1,
        boundary_edges=-3,
        boundary_metres=-3.0,
    )
    selected, audit = select_boundary_aware_actions(
        (positive, negative),
        source_remaining={("core", 13): 10},
        target_remaining={("core", 21): 10},
        enforce_boundary=True,
        required_dynamic_reduction=3,
        engineering_headroom_reduction=3,
    )
    assert [item.action.action_id for item in selected] == ["negative", "positive"]
    assert audit["selected_dynamic_reduction"] == 3
    assert audit["selected_boundary_delta_edges"] == -1
    assert audit["effect_gate"]["pass"] is True


def test_unconstrained_and_boundary_independent_plans_are_reported_separately():
    positive = _planned(
        "positive",
        ((0, 0),),
        dynamic=2,
        boundary_edges=2,
        boundary_metres=2.0,
    )
    free, free_audit = select_boundary_aware_actions(
        (positive,),
        source_remaining={("core", 13): 10},
        target_remaining={("core", 21): 10},
        enforce_boundary=False,
        required_dynamic_reduction=2,
        engineering_headroom_reduction=2,
    )
    safe, safe_audit = select_boundary_aware_actions(
        (positive,),
        source_remaining={("core", 13): 10},
        target_remaining={("core", 21): 10},
        enforce_boundary=True,
        required_dynamic_reduction=2,
        engineering_headroom_reduction=2,
    )
    assert len(free) == 1
    assert free_audit["effect_gate"]["pass"] is True
    assert safe == []
    assert safe_audit["effect_gate"]["pass"] is False


def test_exact_cross_target_rank_tie_is_rejected():
    first = _planned(
        "first",
        ((0, 0),),
        dynamic=1,
        boundary_edges=0,
        boundary_metres=0.0,
    )
    second_action = _action("second", ((0, 0),), dynamic=1, target_code=32)
    second_action = GlobalAction(
        **{
            **second_action.__dict__,
            "probability_support": first.action.probability_support,
            "area_m2": first.action.area_m2,
            "footprint_sha256": first.action.footprint_sha256,
        }
    )
    second = BoundaryPlannedAction(
        **{
            **first.__dict__,
            "action": second_action,
            "target_component_keys": ("target:second",),
        }
    )
    selected, audit = select_boundary_aware_actions(
        (first, second),
        source_remaining={("core", 13): 10},
        target_remaining={("core", 21): 10, ("core", 32): 10},
        enforce_boundary=False,
        required_dynamic_reduction=1,
        engineering_headroom_reduction=1,
    )
    assert selected == []
    assert set(audit["individual_rejections"].values()) == {
        "ambiguous_target_tie"
    }
