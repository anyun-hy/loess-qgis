"""Explicit class-workspace creation beside recognition results."""

from pathlib import Path

from qgis.core import QgsApplication, QgsProject
from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from labeling_tool.qgis_support.qt6_api import NO, YES
from labeling_tool.refinement import class_workspace
from labeling_tool.refinement.background_io_tasks import ManualRunLoadTask
from labeling_tool.refinement.workspace_tasks import WorkspaceTasks


class ResultsWorkspacePanel(QWidget):
    run_selection_changed = pyqtSignal(str)

    def __init__(self, parent=None, *, can_change_run=lambda: True):
        super().__init__(parent)
        self._can_change_run = can_change_run
        self._reader = None
        self._generation = 0
        self._bundle = None
        self._probe = {}
        self._candidate_dir = ""
        self._tasks = WorkspaceTasks(self)
        self._tasks.probed.connect(self._probed)
        self._tasks.initialized.connect(self._initialized)
        self._tasks.terminated.connect(self._failed)
        self._tasks.progress.connect(self._progress)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.run_label = QLabel("选中 Run：无")
        self.run_label.setWordWrap(True)
        self.choose_btn = QPushButton("选择 Run")
        self.choose_btn.clicked.connect(self._choose_run)
        row.addWidget(self.run_label, 1)
        row.addWidget(self.choose_btn)
        root.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Fusion 基准："))
        self.fusion_combo = QComboBox()
        self.fusion_combo.setEnabled(False)
        row.addWidget(self.fusion_combo, 1)
        root.addLayout(row)
        self.status_label = QLabel("选择识别结果后可初始化分类图层")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)
        row = QHBoxLayout()
        self.initialize_btn = QPushButton("初始化分类图层")
        self.initialize_btn.setEnabled(False)
        self.initialize_btn.clicked.connect(self._initialize)
        self.cancel_btn = QPushButton("取消后台检查")
        self.cancel_btn.clicked.connect(self.cancel)
        self.cancel_btn.hide()
        row.addWidget(self.initialize_btn)
        row.addWidget(self.cancel_btn)
        root.addLayout(row)

    @property
    def busy(self):
        return self._reader is not None or self._tasks.busy

    @property
    def run_directory(self):
        return self._candidate_dir

    def set_candidate(self, directory):
        """Present a startup candidate without scanning its result files."""
        if self.busy:
            return
        self._bundle = None
        self._probe = {}
        directory = str(Path(directory).expanduser().resolve()) if directory else ""
        self._candidate_dir = directory
        self.run_label.setText(f"选中 Run：{directory}" if directory else "选中 Run：无")
        self.fusion_combo.clear()
        self.fusion_combo.setEnabled(False)
        self.initialize_btn.setText("检查分类工作区")
        self.initialize_btn.setEnabled(bool(directory))
        self.status_label.setText(
            "检查后可初始化分类图层" if directory else "选择识别结果后可初始化分类图层"
        )
        self.run_selection_changed.emit(directory)

    def set_loaded_run(self, bundle):
        """Reuse the verified load payload for the selected workspace status."""
        if self.busy:
            return
        self.set_candidate(bundle["result"]["run_dir"])
        self._bundle = bundle
        self._probed({
            "workspace": bundle.get("workspace"),
            "eligible_fusions": [
                stream for stream in bundle["result"].get("ready_streams", [])
                if stream.get("kind") == "fusion"
            ],
        })

    def _choose_run(self):
        if not self._can_change_run():
            return
        directory = QFileDialog.getExistingDirectory(self, "选择 Run 文件夹", self.run_directory)
        if directory:
            self.set_run_directory(directory)

    def set_run_directory(self, directory):
        if self.busy:
            return
        if not directory:
            self.set_candidate("")
            return
        directory = str(Path(directory).expanduser().resolve())
        self._bundle = None
        self._candidate_dir = directory
        self._probe = {}
        self._generation += 1
        self.run_label.setText(f"选中 Run：{directory}")
        self.run_label.setToolTip(str(directory))
        self.fusion_combo.clear()
        self._set_busy(True)
        self.status_label.setText("正在后台检查识别结果和分类工作区…")
        self.run_selection_changed.emit(directory)
        task = ManualRunLoadTask(self._generation, directory, for_initialization=True)
        self._reader = task
        task.taskCompleted.connect(self._read_completed)
        task.taskTerminated.connect(self._read_failed)
        task.progressChanged.connect(self._read_progress)
        try:
            QgsApplication.taskManager().addTask(task)
        except Exception as exc:
            self._reader = None
            self._failed("read", str(exc))

    def _set_busy(self, busy):
        self.choose_btn.setEnabled(not busy)
        self.initialize_btn.setEnabled(False)
        self.fusion_combo.setEnabled(False)
        self.cancel_btn.setVisible(busy)

    def _read_progress(self, _value):
        if self.sender() is self._reader:
            self.status_label.setText(self._reader.progress_message)

    def _read_completed(self):
        task = self.sender()
        if task is not self._reader:
            return
        self._reader = None
        self._bundle = task.result_data
        if not self._bundle:
            self._failed("read", "没有返回识别结果")
            return
        self._tasks.probe(
            self._bundle["run_spec"],
            self._bundle["result"]["ready_streams"],
            workspace=self._bundle.get("workspace"),
        )

    def _read_failed(self):
        if self.sender() is not self._reader:
            return
        error = self._reader.error_message
        self._reader = None
        self._failed("read", error or "检查已取消")

    def _progress(self, operation, value):
        action = "初始化分类图层" if operation == "initialize" else "检查分类工作区"
        self.status_label.setText(f"正在后台{action}… {int(value)}%")

    def _probed(self, data):
        self._set_busy(False)
        self._probe = data
        self.fusion_combo.clear()
        for stream in data.get("eligible_fusions") or []:
            self.fusion_combo.addItem(stream["stream_id"], stream["stream_id"])
        workspace = data.get("workspace")
        refresh = data.get("review_refresh") or {}
        replace = bool(refresh.get("required") and refresh.get("safe_to_replace"))
        self.initialize_btn.setText(
            "重建为 V3 分类图层" if replace else "初始化分类图层"
        )
        self.initialize_btn.setEnabled(
            bool(self.fusion_combo.count()) and (not workspace or replace)
        )
        self.fusion_combo.setEnabled(not workspace and bool(self.fusion_combo.count()))
        if workspace:
            self.status_label.setText(
                "已有 14 类工作区；可显式重建为 V3 分类图层"
                if replace
                else "已有 14 类分类数据，人工修改与审核记录保留；图层是否已加载见下方图层状态"
            )
        else:
            self.status_label.setText(
                "尚未初始化 classes，请选择 Fusion 基准后初始化分类图层"
                if self.fusion_combo.count()
                else "没有通过审批和边界校验的 Fusion，不能初始化"
            )

    def _initialize(self):
        if self.busy or not self._can_change_run():
            return
        if not self._bundle:
            if self._candidate_dir:
                self.set_run_directory(self._candidate_dir)
            return
        bundle = self._bundle
        stream_id = self.fusion_combo.currentData()
        if not stream_id:
            return
        replace = bool((self._probe.get("review_refresh") or {}).get("safe_to_replace"))
        if self._probe.get("workspace") and not replace:
            return
        if (
            replace
            and QMessageBox.question(
                self,
                "重建 V3 分类图层",
                "当前工作区没有人工修改、确认或编辑历史。是否用 V3 派生结果重建 14 类工作层？",
                YES | NO,
                NO,
            )
            != YES
        ):
            return
        if bundle is not self._bundle or self.busy:
            return
        if not self._can_change_run():
            return
        if replace:
            directory = Path(bundle["run_spec"]["run_dir"]).resolve() / "classes"
            mounted = any(
                layer.customProperty("labeling_tool/workspace_path", "")
                and Path(str(layer.customProperty("labeling_tool/workspace_path")))
                .resolve()
                .parent
                == directory
                for layer in QgsProject.instance().mapLayers().values()
            )
            if mounted:
                QMessageBox.warning(
                    self,
                    "重建分类图层",
                    "此 Run 的分类图层仍在使用中。请先关闭整理窗口并从图层面板移除这些分类图层，再重建。",
                )
                return
        stream = class_workspace.stream_by_id(
            self._probe["eligible_fusions"], stream_id
        )
        self._set_busy(True)
        self._tasks.initialize(self._bundle["run_spec"], stream, replace=replace)

    def _initialized(self, data):
        self._set_busy(False)
        self._probe = {**self._probe, **data, "review_refresh": {}}
        self.initialize_btn.setText("初始化分类图层")
        self.status_label.setText("14 类分类数据已初始化；选择加载范围后点击下方加载按钮")

    def _failed(self, _operation, message):
        self._set_busy(False)
        self.status_label.setText(f"分类工作区检查失败：{message}")

    def cancel(self):
        self._generation += 1
        if self._reader is not None:
            self._reader.cancel()
            self._reader = None
        self._tasks.cancel()
        self._set_busy(False)
        self.status_label.setText(
            "已请求取消；若初始化已完成发布，重新选择 Run 查看结果"
        )

    def cleanup(self):
        self.cancel()
