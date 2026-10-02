"""Cancellable file-only refinement computations with staged publication."""

import copy
import os
import tempfile
from pathlib import Path

from qgis.core import QgsTask, QgsVectorLayer

from labeling_tool.refinement import final_assembler, topology_validator
from labeling_tool.qgis_support.layer_names import LAYER_NAMES


def file_identity(path):
    value = Path(path)
    def identity(item):
        if not item.exists():
            return None
        stat = item.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns
    return identity(value), identity(Path(str(value) + "-wal"))


class RefinementTask(QgsTask):
    """Only owned file layers are created in run(); results are paths/numbers.

    The GUI publishes staged files only if all source file identities and its
    edit buffers still match. Cancellation/failure leaves existing outputs alone.
    """

    def __init__(self, run_spec, workspace, *, final_path, assemble, transform_context):
        super().__init__("后台组装与拓扑检查", QgsTask.Flag.CanCancel)
        self.spec = copy.deepcopy(run_spec)
        self.workspace = copy.deepcopy(workspace)
        self.final_path = str(final_path or "")
        self.assemble = bool(assemble)
        self.transform_context = transform_context
        self.result_data = None
        self.error_message = ""
        self.staging = None
        paths = [record["path"] for record in self.workspace["classes"].values()]
        paths.extend([self.spec.get("accepted_target_gpkg"), self.final_path])
        directory = Path(self.spec["run_dir"]) / "final"
        paths.extend([directory / "final_composite.gpkg", directory / "topology_issues.gpkg"])
        self.identities = {str(path): file_identity(path) for path in paths if path}

    def inputs_unchanged(self):
        return all(file_identity(path) == identity for path, identity in self.identities.items())

    def discard(self):
        if self.staging is not None:
            self.staging.cleanup()
            self.staging = None

    def publish(self):
        """Publish on the GUI thread, rolling back both files on a rename error."""
        if self.isCanceled() or not self.inputs_unchanged() or not self.result_data:
            raise RuntimeError("源数据已变化或任务取消，结果未发布")
        root = Path(self.spec["run_dir"]) / "final"
        keys = (["final_path"] if self.assemble else []) + ["issues_path"]
        destinations = {"final_path": root / "final_composite.gpkg",
                        "issues_path": root / "topology_issues.gpkg"}
        backups = {}
        replaced = []
        try:
            for key in keys:
                target = destinations[key]
                if target.is_symlink():
                    raise RuntimeError("拒绝替换符号链接结果")
                if target.exists():
                    backup = Path(self.staging.name) / (target.name + ".previous")
                    os.link(target, backup)
                    backups[key] = backup
            for key in keys:
                os.replace(self.result_data[key], destinations[key])
                replaced.append(key)
        except Exception:
            for key in reversed(replaced):
                target = destinations[key]
                if key in backups:
                    os.replace(backups[key], target)
                else:
                    # Move only the just-published task-owned file back.
                    os.replace(target, self.result_data[key])
            raise
        return {key: str(destinations[key]) for key in keys}

    def run(self):
        try:
            directory = Path(self.spec["run_dir"]) / "final"
            directory.mkdir(parents=True, exist_ok=True)
            self.staging = tempfile.TemporaryDirectory(prefix=".refinement-", dir=directory)
            stage = Path(self.staging.name)
            path = self.final_path
            feature_count = None
            if self.assemble:
                path, feature_count = final_assembler.assemble_final(
                    self.spec, self.workspace, output_path=stage / "final_composite.gpkg",
                    transform_context=self.transform_context, is_canceled=self.isCanceled,
                )
            self.setProgress(45)
            accepted = None
            accepted_path = self.spec.get("accepted_target_gpkg")
            if accepted_path and Path(accepted_path).is_file():
                accepted = QgsVectorLayer(f"{accepted_path}|layername={LAYER_NAMES.ACCEPTED}",
                                          "accepted_for_topology", "ogr")
                if not accepted.isValid():
                    raise ValueError("无法打开长期 accepted_labels")
            issues, count, counts = topology_validator.validate_topology(
                self.spec, path, accepted, output_path=stage / "topology_issues.gpkg",
                transform_context=self.transform_context, is_canceled=self.isCanceled,
            )
            del accepted
            if self.isCanceled():
                self.discard()
                return False
            if not self.inputs_unchanged():
                raise RuntimeError("计算期间源文件已变化，请重新组装/检查")
            self.result_data = dict(final_path=path, feature_count=feature_count,
                                    issues_path=issues, issue_count=count, counts=counts)
            self.setProgress(100)
            return True
        except Exception as error:
            self.error_message = f"{type(error).__name__}: {error}"
            self.discard()
            return False
