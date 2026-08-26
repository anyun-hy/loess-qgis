from dataclasses import replace

import numpy as np

from inference_scripts.fragmentation_v31_candidate import (
    CrossCoreDiscovery,
    GlobalAction,
    PlannedAction,
    action_conflicts,
    canonicalize_global_discoveries,
    collect_cross_core_discoveries,
    select_global_actions,
)


def _discovery(
    name: str,
    *,
    footprint=((1, 1),),
    target=21,
    dynamic=1,
    component=1,
    support=.2,
    area=1.0,
    partition="p0",
):
    return CrossCoreDiscovery(
        discovery_id=name,
        discovery_partition_id=partition,
        kind="same_class_bridge",
        target_index=1,
        target_code=target,
        footprint=tuple(footprint),
        involved_core_ids=("left", "right"),
        source_codes=(13,),
        source_anchors=(tuple(footprint[0]),),
        target_anchors=((1, 0), (1, 3)),
        dynamic_reduction=dynamic,
        component_reduction=component,
        probability_support=support,
        area_m2=area,
        edge_distance_m=2.0,
        path_length_m=2.0,
        local_proposal_id=name,
        footprint_sha256=name,
    )


def _action(
    name: str,
    *,
    footprint=((1, 1),),
    dynamic=1,
    component=1,
    support=.2,
    area=1.0,
    source_components=("s",),
    target_components=("t",),
    source_charges=(("left", 13, 1),),
    target_charges=(("left", 21, 1),),
):
    action = GlobalAction(
        action_id=name,
        kind="same_class_bridge",
        target_index=1,
        target_code=21,
        footprint=tuple(footprint),
        involved_core_ids=("left", "right"),
        source_codes=(13,),
        source_anchors=(tuple(footprint[0]),),
        target_anchors=((1, 0), (1, 3)),
        dynamic_reduction=dynamic,
        component_reduction=component,
        probability_support=support,
        area_m2=area,
        discovery_partition_ids=("p0",),
        discovery_count=1,
        footprint_sha256=name,
        score_disagreement=False,
    )
    return PlannedAction(
        action=action,
        source_component_keys=tuple(source_components),
        target_component_keys=tuple(target_components),
        source_charges=tuple(source_charges),
        target_charges=tuple(target_charges),
    )


def test_collect_cross_core_bridge_without_applying_it():
    class_codes = (13, 21)
    labels = np.zeros((50, 50), dtype=np.int16)
    labels[20:23, 22:25] = 1
    labels[20:23, 27:30] = 1
    probabilities = np.zeros((2, 50, 50), dtype=np.float32)
    probabilities[0] = 1.0
    probabilities[1, labels == 1] = 1.0
    probabilities[0, labels == 1] = 0.0
    probabilities[0, 20:23, 25:27] = .35
    probabilities[1, 20:23, 25:27] = .65
    original = labels.copy()

    discoveries, audit = collect_cross_core_discoveries(
        labels,
        class_codes=class_codes,
        pixel_area_m2=1.0,
        pixel_size_m=(1.0, 1.0),
        valid_mask=np.ones(labels.shape, dtype=bool),
        probabilities=probabilities,
        confidence=probabilities.max(axis=0),
        global_origin=(100, 200),
        discovery_partition_id="left",
        owner_for_global_pixel=lambda _row, col: "left" if col < 226 else "right",
    )

    assert discoveries
    assert any(item.involved_core_ids == ("left", "right") for item in discoveries)
    assert audit["cross_core_discovery_count"] == len(discoveries)
    assert np.array_equal(labels, original)

    rejected, rejected_audit = collect_cross_core_discoveries(
        labels,
        class_codes=class_codes,
        pixel_area_m2=1.0,
        pixel_size_m=(1.0, 1.0),
        valid_mask=np.ones(labels.shape, dtype=bool),
        probabilities=probabilities,
        confidence=probabilities.max(axis=0),
        global_origin=(100, 200),
        discovery_partition_id="left",
        owner_for_global_pixel=lambda _row, col: (
            None if col == 226 else ("left" if col < 226 else "right")
        ),
    )
    assert rejected == []
    assert rejected_audit["collection_rejection_events"][
        "unowned_or_strict_invalid_footprint"
    ] >= 1


def test_collect_rejects_three_pixel_footprint_when_two_owners_remain_but_one_pixel_is_strict_invalid():
    labels = np.zeros((50, 50), dtype=np.int16)
    labels[20:23, 22:25] = 1
    labels[20:23, 28:31] = 1
    probabilities = np.zeros((2, 50, 50), dtype=np.float32)
    probabilities[0] = 1.0
    probabilities[1, labels == 1] = 1.0
    probabilities[0, labels == 1] = 0.0
    probabilities[0, 20:23, 25:28] = .35
    probabilities[1, 20:23, 25:28] = .65

    discoveries, audit = collect_cross_core_discoveries(
        labels,
        class_codes=(13, 21),
        pixel_area_m2=1.0,
        pixel_size_m=(1.0, 1.0),
        valid_mask=np.ones(labels.shape, dtype=bool),
        probabilities=probabilities,
        confidence=probabilities.max(axis=0),
        global_origin=(100, 200),
        discovery_partition_id="left",
        owner_for_global_pixel=lambda _row, col: (
            None if col == 227 else ("left" if col < 226 else "right")
        ),
    )

    assert discoveries == []
    assert audit["collection_rejection_events"][
        "unowned_or_strict_invalid_footprint"
    ] >= 1


def test_global_canonicalization_is_order_independent_and_conservative():
    first = _discovery("d1", dynamic=2, component=3, support=.4, area=2.0, partition="p1")
    second = replace(
        first,
        discovery_id="d2",
        discovery_partition_id="p2",
        dynamic_reduction=1,
        component_reduction=2,
        probability_support=.3,
        area_m2=2.1,
    )
    actions, duplicates = canonicalize_global_discoveries([first, second])
    reversed_actions, reversed_duplicates = canonicalize_global_discoveries([second, first])

    assert actions == reversed_actions
    assert duplicates == reversed_duplicates
    assert len(actions) == 1
    assert actions[0].discovery_count == 2
    assert actions[0].score_disagreement is True
    assert (actions[0].dynamic_reduction, actions[0].component_reduction) == (1, 2)
    assert actions[0].probability_support == .3
    assert actions[0].area_m2 == 2.1


def test_conflict_graph_covers_pixels_and_any_shared_topology_component():
    actions = [
        _action("a", footprint=((1, 1),), source_components=("s1",), target_components=("t1",)),
        _action("b", footprint=((1, 1),), source_components=("s2",), target_components=("t2",)),
        _action("c", footprint=((9, 9),), source_components=("s1",), target_components=("t3",)),
        _action("d", footprint=((8, 8),), source_components=("s4",), target_components=("t2",)),
    ]

    assert action_conflicts(actions) == {(0, 1), (0, 2), (1, 3)}


def test_global_milp_uses_lexicographic_score_not_arrival_order():
    high_dynamic = _action(
        "a",
        footprint=((1, 1),),
        dynamic=2,
        component=1,
        source_components=("shared",),
        target_components=("ta",),
    )
    high_component = _action(
        "b",
        footprint=((2, 2),),
        dynamic=1,
        component=10,
        source_components=("shared",),
        target_components=("tb",),
    )
    independent = _action(
        "c",
        footprint=((3, 3),),
        dynamic=1,
        component=1,
        source_components=("sc",),
        target_components=("tc",),
    )
    budgets_source = {("left", 13): 10}
    budgets_target = {("left", 21): 10}

    selected, audit = select_global_actions(
        [high_component, independent, high_dynamic],
        source_remaining=budgets_source,
        target_remaining=budgets_target,
    )
    reversed_selected, reversed_audit = select_global_actions(
        [high_dynamic, independent, high_component],
        source_remaining=budgets_source,
        target_remaining=budgets_target,
    )

    assert [item.action.action_id for item in selected] == ["a", "c"]
    assert selected == reversed_selected
    assert audit["lexicographic_objectives"] == reversed_audit["lexicographic_objectives"]
    assert audit["optimal"] is True


def test_global_milp_charges_cross_core_budgets_once_and_enforces_sum():
    left_right = _action(
        "a",
        footprint=((1, 1), (1, 2)),
        source_components=("sa",),
        target_components=("ta",),
        source_charges=(("left", 13, 1), ("right", 13, 1)),
        target_charges=(("left", 21, 1), ("right", 21, 1)),
    )
    another = _action(
        "b",
        footprint=((4, 4), (4, 5)),
        source_components=("sb",),
        target_components=("tb",),
        source_charges=(("left", 13, 1), ("right", 13, 1)),
        target_charges=(("left", 21, 1), ("right", 21, 1)),
    )

    selected, audit = select_global_actions(
        [left_right, another],
        source_remaining={("left", 13): 1, ("right", 13): 1},
        target_remaining={("left", 21): 1, ("right", 21): 1},
    )

    assert len(selected) == 1
    assert audit["lexicographic_objectives"]["dynamic_reduction"] == 1
    assert audit["eligible_action_count"] == 2


def test_global_milp_rejects_exact_cross_target_tie_before_selection():
    first = _action("a", footprint=((1, 1),), source_components=("sa",), target_components=("ta",))
    second = _action("b", footprint=((2, 2),), source_components=("sb",), target_components=("tb",))
    second = replace(
        second,
        action=replace(
            second.action,
            target_code=32,
            footprint_sha256=first.action.footprint_sha256,
        ),
        target_charges=(("left", 32, 1),),
    )

    selected, audit = select_global_actions(
        [first, second],
        source_remaining={("left", 13): 10},
        target_remaining={("left", 21): 10, ("left", 32): 10},
    )

    assert selected == []
    assert audit["individual_rejections"] == {
        "a": "ambiguous_target_tie",
        "b": "ambiguous_target_tie",
    }
