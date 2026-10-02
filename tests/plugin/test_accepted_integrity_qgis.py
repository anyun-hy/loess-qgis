"""Native QGIS assertions launched from the Conda test environment."""


def test_accepted_audit_checks_review_identity_and_overlap(run_qgis_integrity_case):
    run_qgis_integrity_case("test_accepted_audit_checks_review_identity_and_overlap")


def test_topology_reports_and_writer_blocks_existing_accepted_overlap(run_qgis_integrity_case):
    run_qgis_integrity_case("test_topology_reports_and_writer_blocks_existing_accepted_overlap")
