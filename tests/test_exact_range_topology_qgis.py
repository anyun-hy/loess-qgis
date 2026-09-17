"""Native QGIS assertions launched from the Conda test environment."""


def test_vector_topology_target_is_the_exact_snapshot_not_selected_tile_union(run_qgis_integrity_case):
    run_qgis_integrity_case("test_vector_topology_target_is_the_exact_snapshot_not_selected_tile_union")
