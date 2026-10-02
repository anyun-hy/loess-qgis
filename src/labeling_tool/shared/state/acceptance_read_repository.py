"""Database-only aggregates for the scale acceptance report."""

from __future__ import annotations

from typing import Any

from labeling_tool.shared.state.run_state_session import RunStateSession

__all__ = ["AcceptanceReadRepository"]


class AcceptanceReadRepository:
    """Read acceptance state through one owned or borrowed Run-state connection."""

    def __init__(self, session: RunStateSession) -> None:
        self._session = session

    def snapshot(self, run_id: str) -> dict[str, Any]:
        """Return DB aggregates and artifact rows; callers verify artifact files."""

        with self._session.connection() as connection:
            counts = {}
            for table in ("tiles", "partitions", "spatial_units", "work_packages"):
                counts[table] = int(
                    connection.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE run_id=%s", (run_id,)
                    ).fetchone()[0]
                )
            job_rows = connection.execute(
                "SELECT status, COUNT(*) AS n FROM jobs WHERE run_id=%s GROUP BY status",
                (run_id,),
            ).fetchall()
            job_counts = {str(row["status"]): int(row["n"]) for row in job_rows}
            job_type_rows = connection.execute(
                """SELECT job_type, status, COUNT(*) AS n FROM jobs
               WHERE run_id=%s GROUP BY job_type, status
               ORDER BY job_type, status""",
                (run_id,),
            ).fetchall()
            job_type_counts: dict[str, dict[str, int]] = {}
            for row in job_type_rows:
                job_type_counts.setdefault(str(row["job_type"]), {})[
                    str(row["status"])
                ] = int(row["n"])
            retry_count = int(
                connection.execute(
                    """SELECT COALESCE(SUM(
                     CASE WHEN attempt>0 THEN attempt-1 ELSE 0 END
                   ), 0) FROM jobs WHERE run_id=%s""",
                    (run_id,),
                ).fetchone()[0]
            )
            package_rows = connection.execute(
                "SELECT status, COUNT(*) AS n FROM work_packages WHERE run_id=%s GROUP BY status",
                (run_id,),
            ).fetchall()
            package_counts = {str(row["status"]): int(row["n"]) for row in package_rows}
            stream_rows = [
                dict(row)
                for row in connection.execute(
                    "SELECT stream_id, status, error FROM streams WHERE run_id=%s ORDER BY stream_id",
                    (run_id,),
                ).fetchall()
            ]
            stream_unit_rows = connection.execute(
                """SELECT stream_id, status, COUNT(*) AS n FROM stream_units
               WHERE run_id=%s GROUP BY stream_id, status ORDER BY stream_id, status""",
                (run_id,),
            ).fetchall()
            stream_unit_counts: dict[str, dict[str, int]] = {}
            for row in stream_unit_rows:
                stream_unit_counts.setdefault(str(row["stream_id"]), {})[
                    str(row["status"])
                ] = int(row["n"])
            artifact_rows = [
                dict(row)
                for row in connection.execute(
                    """SELECT path, sha256, byte_count, status, kind, stream_id, unit_id
                   FROM artifacts WHERE run_id=%s ORDER BY artifact_id""",
                    (run_id,),
                ).fetchall()
            ]
        return {
            "counts": counts,
            "job_counts": job_counts,
            "job_type_counts": job_type_counts,
            "retry_count": retry_count,
            "package_counts": package_counts,
            "streams": stream_rows,
            "stream_unit_counts": stream_unit_counts,
            "artifact_rows": artifact_rows,
        }
