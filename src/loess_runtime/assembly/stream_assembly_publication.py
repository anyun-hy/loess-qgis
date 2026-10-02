"""Staging and durable publication of assembled Stream outputs."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StagedReportOutputs:
    """Stream paths staged and published in the fixed replacement order."""

    raw: Path
    formal: Path
    fitted_edges: Path
    report: Path
    candidate: Path

    @classmethod
    def for_stream(cls, root: Path) -> StagedReportOutputs:
        token = f"{os.getpid()}.{uuid.uuid4().hex}"
        return cls(
            raw=root / f".semantic_polygons_raw.{token}.stage.gpkg",
            formal=root / f".semantic_polygons.{token}.stage.gpkg",
            fitted_edges=root / f".fitted_edges.{token}.stage.gpkg",
            report=root / f".boundary_fitting_report.{token}.stage.json",
            candidate=root / f".semantic_candidates.{token}.stage.gpkg",
        )

    def discard(self) -> None:
        """Remove incomplete staging files after success or failure."""

        for path in (
            self.raw,
            self.formal,
            self.fitted_edges,
            self.report,
            self.candidate,
        ):
            path.unlink(missing_ok=True)

    def publish(
        self,
        *,
        raw: Path,
        formal: Path,
        fitted_edges: Path,
        report: Path,
        candidate: Path,
        publish_core_outputs: bool,
        candidate_written: bool,
    ) -> None:
        """Replace final files only after every staged output is complete."""

        if publish_core_outputs:
            os.replace(self.raw, raw)
            os.replace(self.formal, formal)
        if candidate_written:
            os.replace(self.candidate, candidate)
        os.replace(self.fitted_edges, fitted_edges)
        os.replace(self.report, report)
