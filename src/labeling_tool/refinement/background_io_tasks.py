"""Dedicated QgsTasks for copied-Run loading and accepted-label publication."""

from __future__ import annotations

import copy
import threading
from pathlib import Path

from qgis.core import QgsCoordinateTransformContext, QgsTask

from labeling_tool.refinement import accepted_writer, manual_run_loader
from labeling_tool.refinement.refinement_task import (
    file_identity as workspace_file_identity,
)


class ManualRunLoadTask(QgsTask):
    """Validate and publish one copied Run without GUI-thread file scanning."""

    def __init__(self, request_id: int, run_directory) -> None:
        super().__init__("后台加载已有 Run", QgsTask.Flag.CanCancel)
        self.request_id = int(request_id)
        self.run_directory = str(Path(run_directory).expanduser().resolve())
        self.result_data = None
        self.error_message = ""
        self.progress_message = "等待后台加载"
        self.commit_started = False
        self.published = False
        self._lifecycle_lock = threading.Lock()

    def _report(self, message: str, value: float) -> None:
        self.progress_message = str(message)
        self.setProgress(float(value))

    def cancel(self):
        with self._lifecycle_lock:
            if self.commit_started:
                return False
            super().cancel()
            return True

    def _begin_commit(self, bundle) -> None:
        if not manual_run_loader.manual_run_inputs_unchanged(bundle):
            raise manual_run_loader.ManualRunLoadError(
                "后台校验后 Run 文件发生变化，结果未发布"
            )
        with self._lifecycle_lock:
            if self.isCanceled():
                raise manual_run_loader.ManualRunLoadCancelled(
                    "加载已取消；Run 文件未发布"
                )
            self.commit_started = True

    def run(self):
        try:
            bundle = manual_run_loader.prepare_manual_run(
                self.run_directory,
                is_canceled=self.isCanceled,
                progress=self._report,
            )
            if self.isCanceled():
                return False
            self._begin_commit(bundle)
            self._report("正在发布已校验的 Run 元数据，不能取消", 97)
            self.result_data = manual_run_loader.publish_manual_run_bundle(bundle)
            self.published = True
            self._report("已有 Run 加载完成", 100)
            return True
        except manual_run_loader.ManualRunLoadCancelled:
            return False
        except Exception as exc:
            self.error_message = f"{type(exc).__name__}: {exc}"
            return False


class AcceptedWriteTask(QgsTask):
    """Own all QGIS layers and accepted-label connections on a worker thread."""

    def __init__(
        self,
        generation: int,
        *,
        run_id: str,
        final_path,
        accepted_path,
        run_manifest_path,
        workspace_input_identities: dict,
        transform_context,
    ) -> None:
        super().__init__("后台写入 accepted_labels", QgsTask.Flag.CanCancel)
        self.generation = int(generation)
        self.run_id = str(run_id)
        self.final_path = str(Path(final_path).expanduser().resolve())
        self.accepted_path = str(Path(accepted_path).expanduser().resolve())
        self.run_manifest_path = str(Path(run_manifest_path).expanduser().resolve())
        self.workspace_input_identities = copy.deepcopy(workspace_input_identities)
        self.transform_context = QgsCoordinateTransformContext(transform_context)
        self.result_data = None
        self.error_message = ""
        self.progress_message = "等待后台写入"
        self.warnings = []
        self.commit_started = False
        self.published = False
        self._lifecycle_lock = threading.Lock()
        self._frozen_source_identities = {
            self.final_path: accepted_writer.file_identity(self.final_path),
            self.run_manifest_path: accepted_writer.file_identity(
                self.run_manifest_path
            ),
        }
        self._frozen_accepted_identity = accepted_writer.file_identity(
            self.accepted_path
        )

    def _report(self, message: str, value: float) -> None:
        self.progress_message = str(message)
        self.setProgress(float(value))

    def _warn(self, message: str) -> None:
        self.warnings.append(str(message))

    def inputs_unchanged(self, *, transaction_started=False) -> bool:
        current = {
            path: accepted_writer.file_identity(path)
            for path in self._frozen_source_identities
        }
        if current != self._frozen_source_identities:
            return False
        accepted_identity = accepted_writer.file_identity(self.accepted_path)
        if transaction_started:
            if accepted_identity[0] != self._frozen_accepted_identity[0]:
                return False
        elif accepted_identity != self._frozen_accepted_identity:
            return False
        return all(
            workspace_file_identity(path) == identity
            for path, identity in self.workspace_input_identities.items()
        )

    def _begin_commit(self) -> None:
        if not self.inputs_unchanged(transaction_started=True):
            raise RuntimeError("入库前工作区、最终结果或目标发生变化，写入未提交")
        with self._lifecycle_lock:
            if self.isCanceled():
                raise accepted_writer.AcceptedWriteCancelled(
                    "accepted_labels 写入已取消；目标未改变"
                )
            self.commit_started = True
        if not self.inputs_unchanged(transaction_started=True):
            raise RuntimeError("入库前工作区、最终结果或目标发生变化，写入未提交")
        self._report("正在提交，不能取消", 82)

    def cancel(self):
        with self._lifecycle_lock:
            if self.commit_started:
                self.progress_message = "正在提交，不能取消"
                return False
            super().cancel()
            return True

    def run(self):
        try:
            if not self.inputs_unchanged():
                raise RuntimeError("后台任务开始前输入或 accepted_labels 目标已变化")
            count = accepted_writer.append_final_to_accepted(
                self.final_path,
                self.accepted_path,
                self.run_manifest_path,
                is_canceled=self.isCanceled,
                progress=self._report,
                warning=self._warn,
                before_commit=self._begin_commit,
                transform_context=self.transform_context,
            )
            self.published = True
            self.result_data = {
                "run_id": self.run_id,
                "accepted_path": self.accepted_path,
                "feature_count": int(count),
            }
            if self.warnings:
                self.result_data["warnings"] = list(self.warnings)
            return True
        except accepted_writer.AcceptedWriteCancelled:
            return False
        except Exception as exc:
            self.error_message = f"{type(exc).__name__}: {exc}"
            return False
