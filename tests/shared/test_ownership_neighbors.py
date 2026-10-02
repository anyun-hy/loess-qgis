from labeling_tool.runs.spatial_planner import plan_spatial_units
from labeling_tool.shared.planning.ownership_neighbors import ownership_neighbors


def test_ownership_neighbors_are_exact_and_deterministic():
    plan = plan_spatial_units(tile_rows=17, tile_cols=17)
    first = ownership_neighbors(plan["spatial_units"])
    second = ownership_neighbors(reversed(plan["spatial_units"]))
    assert first == second
    assert first
    assert len(first) == len(set(first))
