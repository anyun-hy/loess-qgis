"""QGIS background task for building a large PostgreSQL Run graph."""

from __future__ import annotations

from typing import Any, Mapping

from qgis.core import QgsTask

from labeling_tool.runs.run_builder_v5 import RunBuilderV5Cancelled, create_v5_run
from labeling_tool.shared.state.run_state_db import (
    RunStateDB,
    production_state_database,
    production_state_schema,
)


class RunBuilderTask(QgsTask):
    """Build one frozen Run without occupying the QGIS GUI thread."""

    def __init__(self, builder_kwargs: Mapping[str, Any]):
        super().__init__("后台建立推理任务图", QgsTask.Flag.CanCancel)
        self.builder_kwargs = dict(builder_kwargs)
        self.result_data = None
        self.error_message = ""
        self.progress_message = "等待建立 Run 任务图"

    def _report_progress(self, value: float, message: str) -> None:
        self.progress_message = str(message)
        self.setProgress(max(0.0, min(float(value), 100.0)))

    def _mark_partial_run(self, status: str) -> None:
        """Make a partially created Run archivable without masking errors."""

        run_id = str(self.builder_kwargs.get("run_id") or "")
        if not run_id:
            return
        location = str(
            self.builder_kwargs.get("state_database")
            or production_state_database()
        ).strip()
        try:
            database = RunStateDB(
                location,
                postgres_schema=production_state_schema(),
            )
            database.run_streams.set_run_status(
                run_id,
                status,
                expected="planned",
            )
        except Exception:
            # The builder may have failed before the Run row existed or while
            # PostgreSQL itself was unavailable. Preserve the original error.
            return

    def run(self) -> bool:
        try:
            self.result_data = create_v5_run(
                **self.builder_kwargs,
                progress=self._report_progress,
                is_canceled=self.isCanceled,
            )
            if self.isCanceled():
                self._mark_partial_run("stopped")
                self.result_data = None
                return False
            return True
        except RunBuilderV5Cancelled:
            self._mark_partial_run("stopped")
            self.result_data = None
            return False
        except Exception as exc:
            self._mark_partial_run("failed")
            self.error_message = f"{type(exc).__name__}: {exc}"
            self.result_data = None
            return False
