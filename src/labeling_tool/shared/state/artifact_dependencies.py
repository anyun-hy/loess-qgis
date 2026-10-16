"""Ordered input locks shared by publication and dependency release."""

from __future__ import annotations

from labeling_tool.shared.state.postgres_state import PostgresConnection
from labeling_tool.shared.state.publication_diagnostics import publication_phase


def lock_job_artifact_inputs(connection: PostgresConnection, job_id: int) -> None:
    """Lock all retained inputs before publication touches any shared Artifact.

    The caller owns the transaction. Locking only when deleting dependencies
    is too late if publication has already locked the owner's baseline: two
    neighbours can each hold a baseline that the other needs to release.
    """
    connection.execute(
        """SELECT artifact_id FROM artifacts
           WHERE artifact_id IN (
               SELECT artifact_id FROM artifact_dependencies WHERE job_id=%s
           )
           ORDER BY artifact_id FOR UPDATE""",
        (int(job_id),),
    ).fetchall()


def release_job_artifact_inputs(connection: PostgresConnection, job_id: int) -> int:
    """Release inputs in the caller's transaction, using the same lock order."""
    publication_phase("release_input_lock", job_id=job_id)
    lock_job_artifact_inputs(connection, job_id)
    publication_phase("dependency_release", job_id=job_id)
    return int(
        connection.execute(
            "DELETE FROM artifact_dependencies WHERE job_id=%s",
            (int(job_id),),
        ).rowcount
    )
