from labeling_tool.core.ownership_neighbors import ownership_neighbors
from labeling_tool.core.spatial_planner import plan_spatial_units


def test_ownership_neighbors_are_exact_and_deterministic():
    plan = plan_spatial_units(tile_rows=17, tile_cols=17)
    first = ownership_neighbors(plan["spatial_units"])
    second = ownership_neighbors(reversed(plan["spatial_units"]))
    assert first == second
    assert first
    assert len(first) == len(set(first))


def test_database_union_find_assigns_one_object_id_per_connected_component(
    postgres_database,
):
    database = postgres_database
    database.create_run("run", "a" * 64)
    database.register_streams("run", [{"stream_id": "model:a", "kind": "model"}])
    database.register_object_parts(
        "run",
        "model:a",
        [
            {"part_id": f"part-{index}", "class_code": 12, "unit_id": f"unit-{index}"}
            for index in range(1000)
        ],
    )
    for index in range(998):
        assert database.add_object_link(
            "run", "model:a", f"part-{index}", f"part-{index + 1}", 12
        )
    assert database.resolve_object_components("run", "model:a") == 2
    first = database.object_id_for_part("run", "model:a", "part-0")
    assert database.object_id_for_part("run", "model:a", "part-999") != first
    assert database.object_id_for_part("run", "model:a", "part-998") == first
