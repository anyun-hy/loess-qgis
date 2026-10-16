"""Report the actual release/commit failure stage without changing rollback."""

from __future__ import annotations

import os

import pytest
from psycopg2.errors import CheckViolation
from tests.shared.test_fragmentation_publication_locking import (
    _artifact_rows,
    _assert_rows_lockable,
    _build_adjacent_publication_state,
    _dependency_rows,
    _output_arguments,
)

from labeling_tool.shared.state.publication_diagnostics import (
    publication_failure_payload,
    v33_job_diagnostics,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("LOESS_TEST_POSTGRES_DSN"),
    reason="publication diagnostics integration requires an explicit isolated DSN",
)


@pytest.mark.parametrize("stage", ["dependency_release", "commit_pending"])
def test_postgresql_fault_keeps_actual_phase_sqlstate_and_atomic_rollback(
    tmp_path, postgres_database, stage
):
    database = postgres_database
    state = _build_adjacent_publication_state(database, tmp_path)
    partition, job = next(iter(state.jobs_by_partition.items()))
    with database.session.connection() as connection:
        if stage == "commit_pending":
            timing = (
                "AFTER INSERT ON artifact_dependencies DEFERRABLE INITIALLY DEFERRED"
            )
            trigger_type = "CONSTRAINT "
            identifier = state.finalize_job_id
        else:
            timing = "BEFORE DELETE ON artifact_dependencies"
            trigger_type = ""
            identifier = int(job["job_id"])
        connection.execute(
            f"""CREATE FUNCTION fail_publication_diagnostic() RETURNS trigger AS $$
                BEGIN
                  IF COALESCE(NEW.job_id, OLD.job_id) = {identifier} THEN
                    RAISE EXCEPTION 'injected publication failure'
                      USING ERRCODE='23514';
                  END IF;
                  RETURN COALESCE(NEW, OLD);
                END;
                $$ LANGUAGE plpgsql"""
        )
        connection.execute(
            f"CREATE {trigger_type}TRIGGER fail_publication_diagnostic {timing} "
            "FOR EACH ROW EXECUTE FUNCTION fail_publication_diagnostic()"
        )
    before_artifacts = _artifact_rows(database)
    before_dependencies = _dependency_rows(database)
    before_job = database.jobs.get_job(int(job["job_id"]))
    with pytest.raises(CheckViolation) as caught:
        with v33_job_diagnostics(job):
            with database.fragmentation_v33_attempt_commit(
                int(job["job_id"]), str(job["lease_token"])
            ) as publication:
                publication.artifacts.publish_fragmentation_v33_output_pair(
                    str(job["run_id"]),
                    str(job["stream_id"]),
                    partition,
                    **_output_arguments(tmp_path, partition),
                )
                assert publication.jobs.complete_fragmentation_v33_job(
                    int(job["job_id"]), str(job["lease_token"])
                )
    detail = publication_failure_payload(caught.value)["diagnostics"]
    assert detail["phase"] == stage
    assert detail["sqlstate"] == "23514"
    assert detail["input_lock_completed"] is True
    assert detail["input_lock_elapsed_ms"] >= 0
    assert detail["release_input_lock_elapsed_ms"] >= 0
    assert _artifact_rows(database) == before_artifacts
    assert _dependency_rows(database) == before_dependencies
    assert database.jobs.get_job(int(job["job_id"])) == before_job
    _assert_rows_lockable(database, state.input_ids)
