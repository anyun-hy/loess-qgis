"""Scalar unit-report evidence persistence for PostgreSQL Run state."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from labeling_tool.shared.state.artifact_repository import ArtifactRepository
from labeling_tool.shared.state.run_state_session import RunStateError, RunStateSession
from labeling_tool.shared.state.state_values import utc_now as _now

__all__ = ["UnitReportRepository"]


class UnitReportRepository:
    """Own report scalars derived from ready unit boundary-report Artifacts."""

    def __init__(
        self,
        session: RunStateSession,
        artifacts: ArtifactRepository,
    ) -> None:
        self._session = session
        self._artifacts = artifacts

    def upsert_unit_report_summary(
        self,
        run_id: str,
        stream_id: str,
        unit_id: str,
        report: Mapping[str, Any],
        *,
        fitted_edge_count: int = 0,
    ) -> None:
        """Persist scalar report evidence after its JSON Artifact is ready."""
        artifact = self._artifacts.artifact_for_stream_unit(
            run_id,
            stream_id,
            unit_id,
            "unit_boundary_report",
        )
        if artifact is None:
            raise RunStateError(
                "unit report summary requires a ready unit_boundary_report Artifact"
            )
        report_path = Path(str(artifact["path"]))
        if not report_path.is_file():
            raise RunStateError(f"unit report Artifact is missing: {report_path}")
        stat = report_path.stat()
        if int(artifact["byte_count"]) != int(stat.st_size):
            raise RunStateError(f"unit report Artifact size changed: {report_path}")
        diagnostics = report.get("diagnostics") or []
        if not isinstance(diagnostics, list):
            raise RunStateError("unit boundary report diagnostics must be a list")
        edge_count = int(fitted_edge_count)
        if edge_count < 0 or edge_count > len(diagnostics):
            raise RunStateError(
                "unit fitted edge count is outside the diagnostic report range"
            )
        now = _now()
        with self._session.transaction() as connection:
            connection.execute(
                """INSERT INTO unit_report_summaries
                   (run_id, stream_id, unit_id, status, fit_version,
                    chain_count, shared_chain_count, spline_count,
                    unchanged_count, skipped_invalid_count,
                    max_displacement_px, diagnostic_count, fitted_edge_count,
                    report_path, report_byte_count, report_sha256,
                    report_mtime_ns, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT(run_id, stream_id, unit_id) DO UPDATE SET
                     status=excluded.status,
                     fit_version=excluded.fit_version,
                     chain_count=excluded.chain_count,
                     shared_chain_count=excluded.shared_chain_count,
                     spline_count=excluded.spline_count,
                     unchanged_count=excluded.unchanged_count,
                     skipped_invalid_count=excluded.skipped_invalid_count,
                     max_displacement_px=excluded.max_displacement_px,
                     diagnostic_count=excluded.diagnostic_count,
                     fitted_edge_count=excluded.fitted_edge_count,
                     report_path=excluded.report_path,
                     report_byte_count=excluded.report_byte_count,
                     report_sha256=excluded.report_sha256,
                     report_mtime_ns=excluded.report_mtime_ns,
                     updated_at=excluded.updated_at""",
                (
                    str(run_id),
                    str(stream_id),
                    str(unit_id),
                    str(report.get("status") or ""),
                    str(report.get("fit_version") or ""),
                    int(report.get("chain_count", 0)),
                    int(report.get("shared_chain_count", 0)),
                    int(report.get("spline_count", 0)),
                    int(report.get("unchanged_count", 0)),
                    int(report.get("skipped_invalid_count", 0)),
                    float(report.get("max_displacement_px", 0.0)),
                    len(diagnostics),
                    edge_count,
                    str(report_path.resolve()),
                    int(stat.st_size),
                    str(artifact["sha256"]),
                    int(stat.st_mtime_ns),
                    now,
                    now,
                ),
            )

    def unit_report_summaries(
        self,
        run_id: str,
        stream_id: str,
    ) -> list[dict[str, Any]]:
        """Return summary rows, or no rows when the current contract is incomplete."""
        with self._session.connection() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    """SELECT * FROM unit_report_summaries
                       WHERE run_id=%s AND stream_id=%s ORDER BY unit_id""",
                    (str(run_id), str(stream_id)),
                ).fetchall()
            ]

    def unit_report_summary_aggregate(
        self,
        run_id: str,
        stream_id: str,
    ) -> dict[str, Any]:
        """Aggregate report scalars in the state database without loading JSON."""
        with self._session.connection() as connection:
            row = connection.execute(
                """SELECT COUNT(*) AS unit_count,
                          COALESCE(SUM(chain_count), 0) AS chain_count,
                          COALESCE(SUM(shared_chain_count), 0)
                            AS shared_chain_count,
                          COALESCE(SUM(spline_count), 0) AS spline_count,
                          COALESCE(SUM(unchanged_count), 0) AS unchanged_count,
                          COALESCE(SUM(skipped_invalid_count), 0)
                            AS skipped_invalid_count,
                          COALESCE(SUM(CASE WHEN status='passed' THEN 0 ELSE 1 END), 0)
                            AS failed_unit_count,
                          COALESCE(MAX(max_displacement_px), 0)
                            AS max_displacement_px,
                          COALESCE(SUM(diagnostic_count), 0)
                            AS diagnostic_count,
                          COALESCE(SUM(fitted_edge_count), 0)
                            AS fitted_edge_count
                   FROM unit_report_summaries
                   WHERE run_id=%s AND stream_id=%s""",
                (str(run_id), str(stream_id)),
            ).fetchone()
        return dict(row)
