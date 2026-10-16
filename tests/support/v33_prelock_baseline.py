"""Exact pre-fix methods retained solely for the V3.3 deadlock control tests.

Extracted from commit 145bf7a515be480a5872bf9860be98667f7f390b.
Production code never imports this module; do not update it with the fixed code.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator

from labeling_tool.shared.state.run_state_db import RunStateDB, RunStateError
from labeling_tool.shared.state.state_values import utc_now as _now

PRE_FIX_REVISION = "145bf7a515be480a5872bf9860be98667f7f390b"
PRE_FIX_FILE_SHA256 = {
    "src/labeling_tool/shared/state/run_state_db.py": "11c1bfdaf1c8c1f5752ff63685c4672805214f8032af59ea82beba60f053d49f",
    "src/labeling_tool/shared/state/job_repository.py": "509f36ac0e957201f2abfc40cc3aec21371ee2d31ee728ebd83bf715a6aff4ad",
}


@contextlib.contextmanager
def fragmentation_v33_attempt_commit(
    self, job_id: int, lease_token: str
) -> Iterator["RunStateDB"]:
    """Fence one V3.3 attempt and atomically publish its staged result.

    The caller must write files beneath its immutable attempt directory
    before entering this scope.  Artifact publication and Job completion
    then use only the yielded facade, so an expired or superseded lease
    cannot expose a mixed pair of staged Artifacts.
    """

    with self.session.transaction() as connection:
        job = connection.execute(
            """SELECT * FROM jobs
               WHERE job_id=%s AND job_type='fragmentation_v33'
                 AND status='running' AND lease_token=%s
                 AND lease_expires IS NOT NULL AND lease_expires>=%s
               FOR UPDATE""",
            (int(job_id), str(lease_token), time.time()),
        ).fetchone()
        if job is None:
            raise RunStateError("fragmentation V3.3 attempt no longer owns its lease")
        if self.session.is_scoped_transaction:
            if self.session.unit_identity not in {
                None,
                (
                    str(job["run_id"]),
                    str(job["stream_id"]),
                    str(job["unit_id"]),
                ),
            }:
                raise RunStateError(
                    "fragmentation V3.3 attempt does not match scoped identity"
                )
            yield self
            return
        scoped_session = self.session.scoped(
            connection,
            (
                str(job["run_id"]),
                str(job["stream_id"]),
                str(job["unit_id"]),
            ),
        )
        try:
            yield self._from_session(scoped_session)
        finally:
            scoped_session.invalidate()


def complete_fragmentation_v33_job(
    self,
    job_id: int,
    lease_token: str,
) -> bool:
    """Atomically complete V3.3 and release all retained owner inputs."""

    now = _now()
    fence_time = time.time()
    with self._session.transaction() as connection:
        job = connection.execute(
            """SELECT * FROM jobs WHERE job_id=%s
               AND job_type='fragmentation_v33' AND status='running'
               AND lease_token=%s AND lease_expires IS NOT NULL
               AND lease_expires>=%s""",
            (int(job_id), str(lease_token), fence_time),
        ).fetchone()
        if job is None:
            return False
        unit = connection.execute(
            "SELECT unit_type, owner_key FROM spatial_units "
            "WHERE run_id=%s AND unit_id=%s",
            (str(job["run_id"]), str(job["unit_id"])),
        ).fetchone()
        if unit is None:
            raise RunStateError("V3.3 spatial unit disappeared")
        unit_type = str(unit["unit_type"])
        production = unit_type == "FragmentationV33Finalize"
        staged = unit_type == "FragmentationV33Partition"
        mask_kind, audit_kind, report_kind = (
            (
                "core_mask",
                "fragmentation_v33_audit",
                "fragmentation_v33_report",
            )
            if production
            else (
                "v33_candidate_mask",
                "v33_candidate_audit",
                "v33_candidate_report",
            )
        )
        if staged:
            mask_kind, audit_kind, report_kind = (
                "v33_staged_mask",
                "v33_staged_audit",
                "",
            )
        expected = int(
            connection.execute(
                """SELECT COUNT(*) FROM unit_dependencies
                   WHERE run_id=%s AND unit_id=%s""",
                (str(job["run_id"]), str(job["unit_id"])),
            ).fetchone()[0]
        )
        if expected < 1:
            raise RunStateError("V3.3 has no owner dependencies")
        expected_owner = str(unit["owner_key"]) if staged else None
        for kind in (mask_kind, audit_kind):
            ready_row = connection.execute(
                """SELECT COUNT(*) AS artifact_count,
                              COUNT(DISTINCT a.unit_id) AS owner_count
                       FROM artifacts a
                       JOIN unit_dependencies d
                         ON d.run_id=a.run_id AND d.partition_id=a.unit_id
                       WHERE a.run_id=%s AND a.stream_id=%s
                         AND d.unit_id=%s AND a.kind=%s AND a.status='ready'""",
                (
                    str(job["run_id"]),
                    str(job["stream_id"]),
                    str(job["unit_id"]),
                    kind,
                ),
            ).fetchone()
            if expected_owner is not None:
                ready_row = connection.execute(
                    """SELECT COUNT(*) AS artifact_count,
                              COUNT(DISTINCT unit_id) AS owner_count
                       FROM artifacts WHERE run_id=%s AND stream_id=%s
                         AND unit_id=%s AND kind=%s AND status='ready'""",
                    (
                        str(job["run_id"]),
                        str(job["stream_id"]),
                        expected_owner,
                        kind,
                    ),
                ).fetchone()
            ready = int(ready_row["artifact_count"])
            owners = int(ready_row["owner_count"])
            needed = 1 if expected_owner is not None else expected
            if ready != needed or owners != needed:
                raise RunStateError(
                    f"V3.3 {kind} incomplete or duplicated: "
                    f"artifacts={ready}, owners={owners}, expected={needed}"
                )
        report_ready = int(
            connection.execute(
                """SELECT COUNT(*) FROM artifacts WHERE run_id=%s
                   AND stream_id=%s AND unit_id=%s AND kind=%s AND status='ready'""",
                (
                    str(job["run_id"]),
                    str(job["stream_id"]),
                    str(job["unit_id"]),
                    report_kind,
                ),
            ).fetchone()[0]
        )
        if not staged and report_ready != 1:
            raise RunStateError("V3.3 acceptance report is not ready")
        changed = connection.execute(
            """UPDATE jobs SET status='ready', error='', worker_id='',
               progress_current=%s, progress_total=%s, lease_token='',
               lease_expires=NULL, heartbeat_at=%s, updated_at=%s
               WHERE job_id=%s AND job_type='fragmentation_v33'
                 AND status='running' AND lease_token=%s
                 AND lease_expires IS NOT NULL AND lease_expires>=%s""",
            (
                1 if staged else expected,
                1 if staged else expected,
                now,
                now,
                int(job_id),
                str(lease_token),
                fence_time,
            ),
        ).rowcount
        if changed != 1:
            return False
        connection.execute(
            "DELETE FROM artifact_dependencies WHERE job_id=%s",
            (int(job_id),),
        )
        return True
