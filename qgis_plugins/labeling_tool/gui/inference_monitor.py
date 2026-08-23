"""Modeless result-stream monitor with on-demand tile details."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone

from qgis.PyQt.QtCore import QObject, QTimer, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QDialog,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..qt_compat import (
    HORIZONTAL,
    INTERACTIVE,
    NO_EDIT_TRIGGERS,
    SELECT_ROWS,
    SINGLE_SELECTION,
    STRETCH,
    WINDOW,
)

from .log_panel import LogPanel
from ..core.run_state_db import RunStateDB


STATUS_COLORS = {
    "等待": "#777777",
    "运行中": "#1565c0",
    "成功": "#2e7d32",
    "失败": "#c62828",
    "跳过": "#777777",
    "已停止": "#9a6700",
}

ASSEMBLY_PROGRESS_SCALE = 1000
PIPELINE_STAGES = (
    ("compute", "推理与拟合"),
    ("finalize", "栅格收口"),
    ("assembly", "并行组装"),
    ("acceptance", "整体验收"),
    ("ready", "完成"),
)

RUN_STATUS_LABELS = {
    "preflight": "预检",
    "planned": "已计划",
    "running": "运行中",
    "raster_ready": "栅格就绪",
    "ready": "已完成",
    "failed": "失败",
    "stopped": "已停止",
    "resetting": "正在重置失败包",
}

UNIT_STATUS_LABELS = {
    "queued": "等待",
    "interrupted": "待恢复",
    "resetting": "正在重置",
    "running": "运行中",
    "ready": "完成",
    "failed": "失败",
    "excluded": "已排除",
}

UNIT_TYPE_LABELS = {
    "core": "Core",
    "seam_horizontal": "横向 Seam",
    "seam_vertical": "纵向 Seam",
    "junction": "Junction",
}

TILE_STATUS_LABELS = {
    "ready": "已纳入",
    "accepted": "Accepted 跳过",
    "excluded": "已排除",
    "queued": "等待纳入",
}

DETAIL_STATUS_OPTIONS = {
    "unit": (
        ("全部状态", ""),
        ("等待", "queued"),
        ("待恢复", "interrupted"),
        ("正在重置", "resetting"),
        ("运行中", "running"),
        ("完成", "ready"),
        ("失败", "failed"),
    ),
    "tile": (
        ("全部状态", ""),
        ("等待纳入", "queued"),
        ("已纳入", "ready"),
        ("Accepted 跳过", "accepted"),
        ("已排除", "excluded"),
    ),
}


def _stream_from_step(name: str) -> str:
    if name.startswith("model_batch:"):
        return "model:" + name.split(":", 1)[1]
    if name.startswith("fusion_batch:"):
        return "fusion:" + name.split(":", 1)[1]
    for prefix in ("mosaic:", "polygonize:", "subpixel_vectorize:", "difference:"):
        if name.startswith(prefix):
            return name[len(prefix):]
    if name.startswith("unit_fit:"):
        return name[len("unit_fit:"):].rsplit(":", 1)[0]
    if name.startswith("assemble_stream:"):
        return name[len("assemble_stream:"):]
    return ""


def _stage_from_step(name: str) -> str:
    if name.startswith("unit_fit:"):
        return "空间单元拟合"
    if name.startswith("assemble_stream:"):
        return "并行组装"
    if name.startswith("model_batch:") or name.startswith("fusion_batch:"):
        return "Work Package 推理"
    if name.startswith("mosaic:"):
        return "概率拼接"
    if name.startswith(("polygonize:", "subpixel_vectorize:")):
        return "边界矢量化"
    if name.startswith("difference:"):
        return "Accepted 差分"
    if name == "finalize_partition_rasters":
        return "分区概率栅格收口"
    if name == "scale_acceptance":
        return "整体验收"
    if name == "accelerator_worker":
        return "Work Package 推理"
    return name.split(":", 1)[0]


def _tile_sort_key(tile_id: str):
    return tuple(
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", str(tile_id))
    )


def _elapsed_text(seconds: float) -> str:
    total = max(0, int(seconds))
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    clock = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{days}天 {clock}" if days else clock


def _timestamp_epoch(value: str) -> float | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _waiting_count(counts) -> int:
    return sum(
        int(counts.get(key, 0))
        for key in ("queued", "interrupted", "resetting")
    )


def _assembly_fraction(stream_status, progress) -> float:
    if str(stream_status) == "ready" or str(progress.get("status") or "") == "completed":
        return 1.0
    phase_total = int(progress.get("phase_total") or 0)
    phase_index = int(progress.get("phase_index") or 0)
    if phase_total < 1 or phase_index < 1:
        return 0.0
    current = int(progress.get("progress_current") or 0)
    total = int(progress.get("progress_total") or 0)
    within_phase = min(1.0, max(0.0, current / total)) if total else 0.0
    return min(1.0, max(0.0, (phase_index - 1 + within_phase) / phase_total))


def _unit_stage_label(type_counts) -> str:
    running_types = {
        unit_type
        for unit_type, counts in type_counts.items()
        if int(counts.get("running", 0)) > 0
    }
    labels = []
    if "core" in running_types:
        labels.append("Core")
    if running_types.intersection({"seam_horizontal", "seam_vertical"}):
        labels.append("Seam")
    if "junction" in running_types:
        labels.append("Junction")
    return "/".join(labels) + " 拟合" if labels else "空间单元拟合"


class InferenceMonitorDialog(QDialog):
    stop_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("推理监控")
        self.setWindowFlags(WINDOW)
        self.resize(1180, 720)
        self.setMinimumSize(900, 560)
        self._connected = []
        self._stream_rows = {}
        self._stream_state = {}
        self._tile_state = {}
        self._tile_rows = {}
        self._step_started_at = {}
        self._active_stream_stages = {}
        self._active_global_stage = ""
        self._active_inference_stream = ""
        self._package_activity = {}
        self._run_spec = {}
        self._run_created_epoch = None
        self._monitor_started_at = time.monotonic()
        self._stage_key = ""
        self._stage_started_at = time.monotonic()
        self._runner_message = ""
        self._runtime_progress = {}
        self._coverage_state = {}
        self._log_error_count = 0
        self._log_warning_count = 0
        self._detail_signature = None
        self._database = None
        self._run_id = ""
        self._page = 0
        self._page_size = 500
        self._build_ui()
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(1000)
        self._poll_timer.timeout.connect(self._poll_database)

    def _build_ui(self):
        root = QVBoxLayout(self)
        header = QHBoxLayout()
        self._phase = QLabel("准备中")
        self._phase.setWordWrap(True)
        header.addWidget(self._phase, stretch=1)
        self._log_toggle = QPushButton("显示日志")
        self._log_toggle.setCheckable(True)
        self._log_toggle.toggled.connect(self._set_log_visible)
        header.addWidget(self._log_toggle)
        root.addLayout(header)

        self._stage_rail = QLabel("")
        self._stage_rail.setWordWrap(True)
        root.addWidget(self._stage_rail)
        self._update_stage_rail("compute")

        self._splitter = QSplitter(HORIZONTAL)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self._run_overview = QLabel("Run：准备中")
        self._package_overview = QLabel("Work Package：等待计划")
        self._unit_overview = QLabel("空间单元拟合：等待计划")
        self._assembly_overview = QLabel("结果流组装：等待上游计算")
        self._coverage_overview = QLabel("空白/重叠验收：等待组装")
        for overview in (
            self._run_overview,
            self._package_overview,
            self._unit_overview,
            self._assembly_overview,
            self._coverage_overview,
        ):
            overview.setWordWrap(True)
            left_layout.addWidget(overview)

        self._streams = QTableWidget(0, 7)
        self._streams.setHorizontalHeaderLabels(
            [
                "结果流",
                "当前阶段",
                "当前进度",
                "运行/等待",
                "输出面数",
                "失败",
                "阶段耗时",
            ]
        )
        self._streams.verticalHeader().setVisible(False)
        self._streams.setEditTriggers(NO_EDIT_TRIGGERS)
        self._streams.setSelectionBehavior(SELECT_ROWS)
        self._streams.setSelectionMode(SINGLE_SELECTION)
        self._streams.itemSelectionChanged.connect(self._render_selected_tiles)
        header = self._streams.horizontalHeader()
        header.setMinimumSectionSize(56)
        for column, width in (
            (0, 180),
            (1, 220),
            (2, 120),
            (3, 100),
            (4, 100),
            (5, 64),
            (6, 110),
        ):
            header.setSectionResizeMode(column, INTERACTIVE)
            header.resizeSection(column, width)
        left_layout.addWidget(self._streams, stretch=2)

        self._tile_detail_title = QLabel("选中结果流：未选择 | 空间单元详情")
        left_layout.addWidget(self._tile_detail_title)
        detail_controls = QHBoxLayout()
        self._detail_kind = QComboBox()
        self._detail_kind.addItem("Core / Seam / Junction", "unit")
        self._detail_kind.addItem("Tile 输入", "tile")
        self._detail_status = QComboBox()
        for text, value in DETAIL_STATUS_OPTIONS["unit"]:
            self._detail_status.addItem(text, value)
        self._detail_search = QLineEdit()
        self._detail_search.setPlaceholderText("搜索 ID")
        self._previous_page = QPushButton("上一页")
        self._next_page = QPushButton("下一页")
        self._page_label = QLabel("第 1 页")
        detail_controls.addWidget(self._detail_kind)
        detail_controls.addWidget(self._detail_status)
        detail_controls.addWidget(self._detail_search, stretch=1)
        detail_controls.addWidget(self._previous_page)
        detail_controls.addWidget(self._next_page)
        detail_controls.addWidget(self._page_label)
        left_layout.addLayout(detail_controls)
        self._tiles = QTableWidget(0, 4)
        self._tiles.setHorizontalHeaderLabels(["空间单元", "类型", "状态", "失败原因"])
        self._tiles.verticalHeader().setVisible(False)
        self._tiles.setEditTriggers(NO_EDIT_TRIGGERS)
        tile_header = self._tiles.horizontalHeader()
        tile_header.setMinimumSectionSize(60)
        for column, width in ((0, 96), (1, 72), (2, 88)):
            tile_header.setSectionResizeMode(column, INTERACTIVE)
            tile_header.resizeSection(column, width)
        tile_header.setSectionResizeMode(3, STRETCH)
        left_layout.addWidget(self._tiles, stretch=1)
        self._detail_kind.currentIndexChanged.connect(self._reset_detail_page)
        self._detail_status.currentIndexChanged.connect(self._reset_detail_page)
        self._detail_search.textChanged.connect(self._reset_detail_page)
        self._previous_page.clicked.connect(self._previous_detail_page)
        self._next_page.clicked.connect(self._next_detail_page)
        self._splitter.addWidget(left)

        self._log_panel = LogPanel(self)
        self._log_panel.cleared.connect(self._reset_log_counts)
        self._splitter.addWidget(self._log_panel)
        self._splitter.setStretchFactor(0, 5)
        self._splitter.setStretchFactor(1, 4)
        self._log_panel.setVisible(False)
        self._splitter.setSizes([1180, 0])
        root.addWidget(self._splitter, stretch=1)

        self._summary = QLabel("结果流: 0  |  完成: 0  |  运行: 0  |  等待: 0  |  停止: 0  |  失败: 0")
        root.addWidget(self._summary)

        bottom = QHBoxLayout()
        self._stop = QPushButton("停止")
        self._stop.clicked.connect(self._request_stop)
        bottom.addWidget(self._stop)
        self._bar = QProgressBar()
        bottom.addWidget(self._bar, stretch=1)
        root.addLayout(bottom)

    def attach_runner(self, runner: QObject):
        self.detach()
        pairs = [
            (runner.log_line, self._on_log),
            (runner.step_started, self._on_step_started),
            (runner.step_finished, self._on_step_finished),
            (runner.stream_progress, self._on_stream_progress),
            (runner.pipeline_finished, self._on_finished),
        ]
        for signal, slot in pairs:
            signal.connect(slot)
            self._connected.append((signal, slot))
        self._stop.setEnabled(True)

    def bind_state_database(
        self, database_path, run_id, *, page_size=500, run_spec=None
    ):
        self._database = RunStateDB(database_path)
        self._run_id = str(run_id)
        self._run_spec = dict(run_spec or {})
        self._page_size = max(1, min(int(page_size), 500))
        self._page = 0
        run_row = self._database.get_run(self._run_id) or {}
        self._run_created_epoch = _timestamp_epoch(run_row.get("created_at") or "")
        self._monitor_started_at = time.monotonic()
        self._stage_key = ""
        self._stage_started_at = time.monotonic()
        self._poll_timer.start()
        self._poll_database()

    def unbind_state_database(self):
        if hasattr(self, "_poll_timer"):
            self._poll_timer.stop()
        self._database = None
        self._run_id = ""
        self._run_spec = {}
        self._run_created_epoch = None
        self._page = 0

    def detach(self):
        for signal, slot in self._connected:
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._connected.clear()

    def clear_log(self):
        self._log_panel.clear()

    def _set_log_visible(self, visible):
        shown = bool(visible)
        self._log_panel.setVisible(shown)
        self._splitter.setSizes([720, 460] if shown else [1180, 0])
        self._update_log_toggle()

    def _reset_log_counts(self):
        self._log_error_count = 0
        self._log_warning_count = 0
        self._update_log_toggle()

    def _update_log_toggle(self):
        action = "收起日志" if self._log_toggle.isChecked() else "显示日志"
        counts = []
        if self._log_error_count:
            counts.append(f"{self._log_error_count}错误")
        if self._log_warning_count:
            counts.append(f"{self._log_warning_count}警告")
        self._log_toggle.setText(
            action + (" · " + "/".join(counts) if counts else "")
        )

    def _update_coverage_overview(self):
        values = [
            dict(value)
            for value in self._coverage_state.values()
            if isinstance(value, dict)
        ]
        if not values:
            self._coverage_overview.setText("空白/重叠验收：等待组装")
            return
        gap_area_m2 = sum(
            float(value.get("gap_area_m2") or 0.0) for value in values
        )
        overlap_area_m2 = sum(
            float(value.get("overlap_area_m2") or 0.0) for value in values
        )
        outside_area_m2 = sum(
            float(value.get("outside_area_m2") or 0.0) for value in values
        )
        passed = sum(1 for value in values if value.get("status") == "passed")
        failed = sum(1 for value in values if value.get("status") == "failed")
        skipped = len(values) - passed - failed
        if failed:
            state = f"失败 {failed} 个流"
        elif skipped:
            state = f"通过 {passed}，未验证 {skipped}"
        else:
            state = "通过"
        self._coverage_overview.setText(
            f"空白/重叠验收：{state} {passed}/{len(values)} | "
            f"空白 {gap_area_m2:.6g} m² | "
            f"重叠 {overlap_area_m2:.6g} m² | "
            f"范围外 {outside_area_m2:.6g} m²"
        )

    def _update_stage_rail(self, active_key):
        order = [key for key, _name in PIPELINE_STAGES]
        active = str(active_key or "compute")
        active_index = order.index(active) if active in order else 0
        parts = []
        for index, (key, name) in enumerate(PIPELINE_STAGES):
            if index < active_index or active == "ready":
                color = "#2d7a52"
                marker = "✓"
            elif key == active:
                color = "#2f6f9f"
                marker = "●"
            else:
                color = "#7b8794"
                marker = "○"
            parts.append(
                f'<span style="color:{color}; font-weight:600">'
                f"{marker} {name}</span>"
            )
        self._stage_rail.setText("&nbsp;&nbsp;→&nbsp;&nbsp;".join(parts))

    def reset_run(self, tiles=None):
        del tiles
        self.unbind_state_database()
        self.clear_log()
        self._streams.setRowCount(0)
        self._tiles.setRowCount(0)
        self._stream_rows.clear()
        self._stream_state.clear()
        self._tile_state.clear()
        self._tile_rows.clear()
        self._step_started_at.clear()
        self._active_stream_stages.clear()
        self._active_global_stage = ""
        self._active_inference_stream = ""
        self._package_activity.clear()
        self._runner_message = ""
        self._runtime_progress.clear()
        self._coverage_state.clear()
        self._detail_signature = None
        self._stage_key = ""
        self._stage_started_at = time.monotonic()
        self._phase.setText("准备中")
        self._run_overview.setText("Run：准备中")
        self._package_overview.setText("Work Package：等待计划")
        self._unit_overview.setText("空间单元拟合：等待计划")
        self._assembly_overview.setText("结果流组装：等待上游计算")
        self._coverage_overview.setText("空白/重叠验收：等待组装")
        self._update_stage_rail("compute")
        self._tile_detail_title.setText("选中结果流：未选择 | 空间单元详情")
        self._summary.setText("结果流: 0  |  完成: 0  |  运行: 0  |  等待: 0  |  停止: 0  |  失败: 0")
        self._bar.setRange(0, 0)
        self._bar.setFormat("准备中")
        self._stop.setEnabled(True)
        self.setWindowTitle("推理监控 - 准备中")

    def set_stage_progress(self, info):
        name = str(info.get("name") or "处理中")
        stream_id = str(info.get("stream_id") or "")
        current = int(info.get("current") or 0)
        total = int(info.get("total") or 0)
        message = str(info.get("message") or "")
        if self._database is not None and self._run_id:
            # V5 的 runner 总数把 Work Package 和 unit_fit 两种成本完全不同的
            # Job 相加。数据库绑定后由左侧分层概览分别显示，不能再把这个
            # 混合总数作为用户进度条。
            self._runner_message = message
            return
        text = f"{name} | {stream_id}" if stream_id else name
        if message:
            text += f" | {message}"
        self._phase.setText(text)
        if total > 0:
            self._bar.setRange(0, total)
            self._bar.setValue(min(current, total))
            self._bar.setFormat(f"{text}  {current}/{total}")
        else:
            self._bar.setRange(0, 0)
            self._bar.setFormat(text)
        if stream_id:
            self._set_stream(stream_id, stage=name, progress=f"{current}/{total}" if total else "-")

    def mark_stopping(self):
        self._stop.setEnabled(False)
        self._stop.setText("正在停止")
        self._phase.setText("正在停止当前子进程组")

    def mark_finished(self, text="已完成"):
        self._stop.setEnabled(False)
        self._stop.setText("停止")
        self._phase.setText(text)
        self._bar.setRange(0, 1)
        self._bar.setValue(1)
        self._bar.setFormat(text)
        if text == "已完成":
            self._update_stage_rail("ready")
        self.setWindowTitle(f"推理监控 - {text}")

    def _ensure_stream(self, stream_id):
        if stream_id in self._stream_rows:
            return self._stream_rows[stream_id]
        row = self._streams.rowCount()
        self._streams.insertRow(row)
        self._stream_rows[stream_id] = row
        self._stream_state[stream_id] = {
            "stage": "等待计划",
            "progress": "-",
            "unit_progress": "-",
            "stage_progress": "-",
            "activity": "0/0",
            "feature_count": 0,
            "status": "等待",
            "elapsed": "-",
            "failures": 0,
        }
        self._tile_state.setdefault(stream_id, {})
        self._write_stream_row(stream_id)
        if self._streams.currentRow() < 0:
            self._streams.selectRow(row)
        return row

    def _set_stream(self, stream_id, **changes):
        self._ensure_stream(stream_id)
        self._stream_state[stream_id].update(changes)
        self._write_stream_row(stream_id)
        self._update_summary()

    def _write_stream_row(self, stream_id):
        row = self._stream_rows[stream_id]
        state = self._stream_state[stream_id]
        current_progress = (
            state.get("stage_progress")
            or state.get("unit_progress")
            or state.get("progress")
            or "-"
        )
        feature_count = int(state.get("feature_count") or 0)
        values = [
            self._stream_display_name(stream_id),
            state["stage"],
            current_progress,
            state.get("activity") or "0/0",
            f"{feature_count:,}" if feature_count else "—",
            str(state["failures"]),
            state["elapsed"],
        ]
        for column, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            if column == 0:
                item.setToolTip(stream_id)
            if column == 1:
                item.setForeground(
                    QColor(STATUS_COLORS.get(str(state["status"]), "#333333"))
                )
            self._streams.setItem(row, column, item)

    def _on_log(self, level, message):
        lowered = str(message).lower()
        warning = any(token in lowered for token in ("warning", "warn", "警告"))
        failure = any(
            token in lowered
            for token in ('"event":"stream_assembly_failed"', " error", "failed")
        )
        if level == "stdout":
            self._log_panel.append_stdout(message)
        elif level == "stderr":
            self._log_panel.append_stderr(message)
        else:
            self._log_panel.append_system(message)
        if warning:
            self._log_warning_count += 1
        elif level == "stderr" or failure:
            self._log_error_count += 1
        if warning or level == "stderr" or failure:
            self._update_log_toggle()

    def _on_step_started(self, name):
        stream_id = _stream_from_step(name)
        stage = _stage_from_step(name)
        self._step_started_at[name] = time.time()
        self._active_global_stage = stage
        if stream_id:
            stage_counts = self._active_stream_stages.setdefault(stream_id, {})
            stage_counts[stage] = int(stage_counts.get(stage, 0)) + 1
            self._set_stream(stream_id, stage=stage, status="运行中")

    def _on_step_finished(self, name, return_code, result):
        del return_code
        stream_id = str(result.get("stream_id") or _stream_from_step(name))
        stage = _stage_from_step(name)
        started = self._step_started_at.pop(name, None)
        elapsed = time.time() - started if started else float(result.get("elapsed_sec") or 0)
        if self._active_global_stage == stage:
            self._active_global_stage = ""
        if not stream_id:
            return
        stage_counts = self._active_stream_stages.get(stream_id) or {}
        remaining = max(0, int(stage_counts.get(stage, 0)) - 1)
        if remaining:
            stage_counts[stage] = remaining
        else:
            stage_counts.pop(stage, None)
        if not stage_counts:
            self._active_stream_stages.pop(stream_id, None)
        status = "成功" if result.get("success") else "跳过" if result.get("skipped") else "失败"
        changes = {"elapsed": f"{elapsed:.1f}s"}
        if self._database is None:
            changes["status"] = status
        elif status == "失败":
            changes.update({"status": "失败", "stage": "任务失败"})
        if status == "失败":
            changes["failures"] = int(self._stream_state.get(stream_id, {}).get("failures", 0)) + 1
        self._set_stream(stream_id, **changes)

    def _on_stream_progress(self, info):
        event = str(info.get("event") or "")
        if event.startswith(("package_", "work_package_", "accelerator_worker_")):
            self._update_package_activity(info)
            if self._database is not None:
                return
        stream_id = str(info.get("stream_id") or "")
        if not stream_id:
            return
        current = int(info.get("current") or 0)
        total = int(info.get("total") or 0)
        failure = str(info.get("error") or "")
        if event == "assembly_progress":
            progress_status = str(info.get("status") or "running")
            status = {
                "completed": "成功",
                "failed": "失败",
            }.get(progress_status, "运行中")
            self._runtime_progress[stream_id] = dict(info)
            self._set_stream(
                stream_id,
                stage=str(info.get("phase_name") or "并行组装"),
                stage_progress=(
                    f"{current}/{total}"
                    if total
                    else f"步骤 {int(info.get('phase_index') or 0)}/"
                    f"{int(info.get('phase_total') or 0)}"
                ),
                activity="—",
                feature_count=int(info.get("feature_count") or 0),
                elapsed=_elapsed_text(float(info.get("elapsed_sec") or 0)),
                status=status,
                failures=int(
                    self._stream_state.get(stream_id, {}).get("failures", 0)
                ) + (1 if progress_status == "failed" else 0),
            )
            return
        if event == "stream_coverage_validation":
            self._coverage_state[stream_id] = dict(info)
            self._update_coverage_overview()
            coverage_status = str(info.get("status") or "")
            passed = coverage_status == "passed"
            failed = coverage_status == "failed"
            self._set_stream(
                stream_id,
                stage="空白/重叠验收",
                stage_progress="1/1",
                status="成功" if passed else "失败" if failed else "未验证",
                failures=int(
                    self._stream_state.get(stream_id, {}).get("failures", 0)
                ) + (1 if failed else 0),
            )
            return
        status = "失败" if event.endswith("failed") else "运行中"
        self._set_stream(
            stream_id,
            progress=f"{current}/{total}" if total else "-",
            status=status,
            failures=int(self._stream_state.get(stream_id, {}).get("failures", 0)) + (1 if failure else 0),
        )
        tile_id = info.get("tile_id")
        if tile_id:
            tile_id = str(tile_id)
            state = {
                "status": "失败" if failure else "完成" if event.endswith(("completed", "reused")) else "运行中",
                "progress": f"{current}/{total}" if total else "-",
                "error": failure,
            }
            self._tile_state.setdefault(stream_id, {})[tile_id] = state
            self._update_selected_tile(stream_id, tile_id, state)

    def _configured_batch_for_stream(self, stream_id: str) -> int:
        tuning = self._run_spec.get("resource_tuning") or {}
        resolved = tuning.get("resolved") or {}
        by_model = resolved.get("tile_batch_size_by_model") or {}
        runtime = self._run_spec.get("runtime") or {}
        model_id = (
            stream_id.split(":", 1)[1]
            if stream_id.startswith("model:")
            else ""
        )
        return int(
            by_model.get(model_id)
            or resolved.get("tile_batch_size")
            or runtime.get("tile_batch_size")
            or 0
        )

    def _update_package_activity(self, info):
        event = str(info.get("event") or "")
        package_id = str(info.get("package_id") or "")
        stream_id = str(info.get("stream_id") or "")
        previous_package = str(self._package_activity.get("package_id") or "")
        if package_id and package_id != previous_package:
            self._active_inference_stream = ""
            self._package_activity = {
                "package_id": package_id,
                "started_at": time.monotonic(),
                "status": "运行中",
            }
        if package_id:
            self._package_activity["package_id"] = package_id
        if stream_id:
            self._package_activity["stream_id"] = stream_id
            self._active_inference_stream = stream_id
            configured_batch = self._configured_batch_for_stream(stream_id)
            if configured_batch:
                self._package_activity["configured_batch_size"] = configured_batch
                self._package_activity.setdefault(
                    "effective_batch_size", configured_batch
                )
            self._set_stream(
                stream_id,
                stage="Work Package 推理",
                status="运行中",
            )
        if event == "package_model_loading":
            self._package_activity.update(
                {
                    "model_current": int(info.get("current") or 0),
                    "model_total": int(info.get("total") or 0),
                    "tile_current": 0,
                    "tile_total": 0,
                    "status": "模型加载/推理",
                }
            )
        elif event in ("package_tile_materialized", "package_tile_completed"):
            tile_current = int(info.get("current") or 0)
            tile_total = int(info.get("total") or 0)
            status = "Tile 物化" if event.endswith("materialized") else "模型推理"
            if event == "package_tile_materialized" and tile_current <= 1:
                # Tile materialization is the first observable event of an
                # attempt. Clear state left by a failed attempt even when the
                # Package ID is reused.
                self._active_inference_stream = ""
                for key in (
                    "stream_id",
                    "model_current",
                    "model_total",
                    "configured_batch_size",
                    "effective_batch_size",
                    "notice",
                    "elapsed_sec",
                ):
                    self._package_activity.pop(key, None)
            if (
                event == "package_tile_completed"
                and tile_total > 0
                and tile_current >= tile_total
                and int(self._package_activity.get("model_current") or 0)
                >= int(self._package_activity.get("model_total") or 0)
            ):
                status = "Fusion / Package 收口"
            self._package_activity.update(
                {
                    "tile_current": tile_current,
                    "tile_total": tile_total,
                    "status": status,
                }
            )
            if status == "Fusion / Package 收口":
                fusion = self._run_spec.get("fusion") or {}
                profile_id = str(fusion.get("profile_id") or "")
                if profile_id:
                    fusion_stream = f"fusion:{profile_id}"
                    self._package_activity["stream_id"] = fusion_stream
                    self._active_inference_stream = fusion_stream
                    self._set_stream(
                        fusion_stream,
                        stage="Work Package Fusion / 收口",
                        status="运行中",
                    )
        elif event == "package_tile_batch_reduced":
            self._package_activity.update(
                {
                    "effective_batch_size": int(info.get("effective_batch_size") or 0),
                    "status": "Batch 降档后重试",
                    "notice": "OOM 降档",
                }
            )
        elif event == "package_model_outputs_reused":
            self._package_activity.update({"status": "复用已有模型结果"})
        elif event == "package_tiles_cleaned":
            self._package_activity.update({"status": "缓存清理/提交"})
        elif event == "work_package_finished":
            self._package_activity.update(
                {
                    "status": "已完成",
                    "elapsed_sec": float(info.get("elapsed_sec") or 0),
                    "notice": "",
                }
            )
            self._active_inference_stream = ""
        elif event == "accelerator_worker_finished":
            self._active_inference_stream = ""
        elif event == "accelerator_worker_paused_low_disk":
            self._package_activity.update(
                {"status": "低磁盘暂停", "notice": "等待磁盘空间"}
            )
        elif event.endswith("failed"):
            self._package_activity.update(
                {"status": "失败", "notice": str(info.get("error") or "")}
            )
            self._active_inference_stream = ""

    def _stream_display_name(self, stream_id: str) -> str:
        for model in self._run_spec.get("models") or []:
            if stream_id == f"model:{model.get('model_id')}":
                return str(model.get("display_name") or model.get("model_id") or stream_id)
        fusion = self._run_spec.get("fusion") or {}
        if stream_id == f"fusion:{fusion.get('profile_id')}":
            return str(fusion.get("display_name") or "Fusion")
        return stream_id

    def _active_stage_for_stream(self, stream_id: str) -> str:
        stage_counts = self._active_stream_stages.get(stream_id) or {}
        for stage in (
            "并行组装",
            "Accepted 差分",
            "边界矢量化",
            "空间单元拟合",
        ):
            if int(stage_counts.get(stage, 0)) > 0:
                return stage
        return next(iter(stage_counts), "")

    def _selected_stream(self):
        row = self._streams.currentRow()
        if row < 0:
            return ""
        item = self._streams.item(row, 0)
        if item is None:
            return ""
        return str(item.toolTip() or item.text())

    def _render_selected_tiles(self):
        stream_id = self._selected_stream()
        if self._database is not None and self._run_id:
            self._render_database_page(stream_id)
            return
        values = self._tile_state.get(stream_id, {})
        self._tile_rows.clear()
        self._tiles.setRowCount(len(values))
        for row, (tile_id, state) in enumerate(
            sorted(values.items(), key=lambda item: _tile_sort_key(item[0]))
        ):
            self._tile_rows[tile_id] = row
            for column, value in enumerate((tile_id, state["status"], state["progress"], state["error"])):
                self._tiles.setItem(row, column, QTableWidgetItem(str(value)))
        if stream_id:
            self._tile_detail_title.setText(
                f"选中结果流：{stream_id} | Tile 详情（已记录 {len(values)} 个）"
            )
        else:
            self._tile_detail_title.setText("选中结果流：未选择 | Tile 详情")

    def _sync_detail_status_options(self):
        kind = str(self._detail_kind.currentData() or "unit")
        options = DETAIL_STATUS_OPTIONS.get(kind, DETAIL_STATUS_OPTIONS["unit"])
        desired_values = [value for _text, value in options]
        current_values = [
            str(self._detail_status.itemData(index) or "")
            for index in range(self._detail_status.count())
        ]
        if current_values == desired_values:
            return
        selected = str(self._detail_status.currentData() or "")
        was_blocked = self._detail_status.blockSignals(True)
        try:
            self._detail_status.clear()
            for text, value in options:
                self._detail_status.addItem(text, value)
            selected_index = self._detail_status.findData(selected)
            self._detail_status.setCurrentIndex(max(0, selected_index))
        finally:
            self._detail_status.blockSignals(was_blocked)

    def _reset_detail_page(self, *_args):
        self._sync_detail_status_options()
        self._page = 0
        self._detail_signature = None
        self._render_selected_tiles()

    def _previous_detail_page(self):
        if self._page > 0:
            self._page -= 1
            self._detail_signature = None
            self._render_selected_tiles()

    def _next_detail_page(self):
        if self._next_page.isEnabled():
            self._page += 1
            self._detail_signature = None
            self._render_selected_tiles()

    def _render_database_page(self, stream_id):
        if not stream_id:
            self._detail_signature = None
            self._tile_rows.clear()
            self._tiles.setRowCount(0)
            self._tile_detail_title.setText(
                "选中结果流：未选择 | 空间单元详情"
            )
            return
        kind = str(self._detail_kind.currentData() or "unit")
        status = str(self._detail_status.currentData() or "")
        search = self._detail_search.text().strip()
        try:
            if kind == "tile":
                total = self._database.count_tiles(
                    self._run_id, status=status or None, search=search
                )
            else:
                total = self._database.count_stream_units(
                    self._run_id, stream_id, status=status, search=search
                )
            page_total = max(1, (total + self._page_size - 1) // self._page_size)
            if self._page >= page_total:
                self._page = page_total - 1
            offset = self._page * self._page_size
            if kind == "tile":
                rows = self._database.page_tiles(
                    self._run_id,
                    limit=self._page_size,
                    offset=offset,
                    status=status or None,
                    search=search,
                )
                headers = ["Tile", "Partition", "选择状态", "失败原因"]
                values = [
                    (
                        row["tile_id"],
                        row.get("partition_id") or "-",
                        TILE_STATUS_LABELS.get(row["status"], row["status"]),
                        "",
                    )
                    for row in rows
                ]
                detail_name = "Tile 输入清单"
            else:
                rows = self._database.page_stream_units(
                    self._run_id,
                    stream_id,
                    limit=self._page_size,
                    offset=offset,
                    status=status,
                    search=search,
                )
                headers = ["空间单元", "类型", "状态", "失败原因"]
                values = [
                    (
                        row["unit_id"],
                        UNIT_TYPE_LABELS.get(row["unit_type"], row["unit_type"]),
                        UNIT_STATUS_LABELS.get(row["status"], row["status"]),
                        row["error"],
                    )
                    for row in rows
                ]
                detail_name = "Core / Seam / Junction"
        except Exception as error:
            self._detail_signature = None
            self._tile_rows.clear()
            self._tiles.setRowCount(0)
            self._tile_detail_title.setText(
                f"选中结果流：{stream_id} | 数据库查询失败: {error}"
            )
            return

        self._page_label.setText(f"第 {self._page + 1}/{page_total} 页")
        self._previous_page.setEnabled(self._page > 0)
        self._next_page.setEnabled(self._page + 1 < page_total)
        self._tile_detail_title.setText(
            f"选中结果流：{self._stream_display_name(stream_id)} | "
            f"{detail_name} 详情（共 {total} 条，每页最多 {self._page_size}）"
        )
        signature = (
            stream_id,
            kind,
            status,
            search,
            self._page,
            total,
            tuple(values),
        )
        if signature == self._detail_signature:
            return
        self._detail_signature = signature
        self._tile_rows.clear()
        self._tiles.setHorizontalHeaderLabels(headers)
        self._tiles.setRowCount(len(values))
        for row_index, row_values in enumerate(values):
            for column, value in enumerate(row_values):
                self._tiles.setItem(
                    row_index, column, QTableWidgetItem(str(value))
                )

    def _poll_database(self):
        if self._database is None or not self._run_id:
            return
        try:
            snapshot = self._database.monitor_snapshot(self._run_id)
            run_row = snapshot.get("run") or {}
            run_status = str(run_row.get("status") or "planned")
            job_counts = snapshot.get("job_counts") or {}
            package_counts = job_counts.get("work_package") or {}
            unit_job_counts = job_counts.get("unit_fit") or {}
            package_failed = int(package_counts.get("failed", 0))
            active_package = snapshot.get("active_work_package")
            if active_package is not None:
                package_id = str(active_package.get("package_id") or "")
                attempt = int(active_package.get("attempt") or 0)
                previous_package = str(
                    self._package_activity.get("package_id") or ""
                )
                previous_attempt = self._package_activity.get("attempt")
                attempt_changed = (
                    previous_attempt is not None
                    and int(previous_attempt) != attempt
                )
                if package_id != previous_package or attempt_changed:
                    self._active_inference_stream = ""
                    self._package_activity = {
                        "package_id": package_id,
                        "attempt": attempt,
                        "status": "运行中",
                    }
                self._package_activity.update(
                    {
                        "package_id": package_id,
                        "attempt": attempt,
                        "sequence_no": int(active_package.get("sequence_no") or 0),
                        "db_current": int(active_package.get("progress_current") or 0),
                        "db_total": int(active_package.get("progress_total") or 0),
                        "started_epoch": _timestamp_epoch(
                            active_package.get("package_started_at") or ""
                        ),
                    }
                )
            elif int(package_counts.get("running", 0)) == 0:
                self._active_inference_stream = ""

            streams = snapshot.get("streams") or []
            all_runtime_progress = (
                snapshot.get("stream_runtime_progress") or {}
            )
            self._runtime_progress = {
                str(key): dict(value)
                for key, value in all_runtime_progress.items()
            }
            persisted_coverage = snapshot.get("stream_coverage_validation") or {}
            if persisted_coverage:
                self._coverage_state = {
                    str(key): dict(value)
                    for key, value in persisted_coverage.items()
                }
            coverage_updater = getattr(self, "_update_coverage_overview", None)
            if callable(coverage_updater):
                coverage_updater()
            all_type_counts = snapshot.get("stream_unit_type_counts") or {}
            all_job_type_counts = (
                snapshot.get("stream_unit_job_type_counts") or {}
            )
            for stream in streams:
                stream_id = str(stream["stream_id"])
                type_counts = all_type_counts.get(stream_id) or {}
                durable_counts = {}
                for counts in type_counts.values():
                    for state, count in counts.items():
                        durable_counts[state] = int(
                            durable_counts.get(state, 0)
                        ) + int(count)
                job_type_counts = all_job_type_counts.get(stream_id) or {}
                stream_unit_job_counts = {}
                for counts in job_type_counts.values():
                    for state, count in counts.items():
                        stream_unit_job_counts[state] = int(
                            stream_unit_job_counts.get(state, 0)
                        ) + int(count)
                total = sum(int(value) for value in durable_counts.values())
                ready = int(durable_counts.get("ready", 0))
                running = int(stream_unit_job_counts.get("running", 0))
                waiting = _waiting_count(stream_unit_job_counts)
                failed = int(stream_unit_job_counts.get("failed", 0))
                stream_status = str(stream.get("status") or "pending")
                assembly_info = all_runtime_progress.get(stream_id) or {}
                assembly_status = str(assembly_info.get("status") or "")
                assembly_phase = str(
                    assembly_info.get("phase_name") or "并行组装"
                )
                active_stage = self._active_stage_for_stream(stream_id)
                inference_active = (
                    int(package_counts.get("running", 0)) > 0
                    and stream_id == self._active_inference_stream
                )

                if run_status == "stopped":
                    if stream_status == "ready":
                        stage, status = "组装完成 / Run 已停止", "成功"
                    else:
                        stage, status = "Run 已停止，可安全恢复", "已停止"
                elif run_status == "failed":
                    if stream_status == "ready":
                        stage, status = "组装完成 / Run 未通过", "成功"
                    elif package_failed:
                        stage, status = "上游 Work Package 失败", "失败"
                    else:
                        stage, status = "Run 失败", "失败"
                elif package_failed:
                    if stream_status == "ready":
                        stage, status = "组装完成 / 上游 Package 失败", "成功"
                    else:
                        stage, status = "上游 Work Package 失败", "失败"
                elif assembly_status == "failed":
                    stage, status = f"组装失败：{assembly_phase}", "失败"
                elif failed or stream_status == "failed":
                    stage, status = "空间单元任务失败", "失败"
                elif active_stage == "并行组装" or stream_status == "assembling":
                    stage, status = assembly_phase, "运行中"
                elif stream_status == "ready" and run_status == "ready":
                    stage, status = "完成", "成功"
                elif stream_status == "ready":
                    stage, status = "已组装 / 等待整体验收", "成功"
                elif stream_status == "raster_ready":
                    stage, status = "等待并行组装", "等待"
                elif inference_active and running:
                    stage = f"推理 + {_unit_stage_label(job_type_counts)}"
                    status = "运行中"
                elif inference_active:
                    stage, status = "Work Package 推理", "运行中"
                elif running:
                    stage, status = _unit_stage_label(job_type_counts), "运行中"
                elif active_stage:
                    stage, status = active_stage, "运行中"
                elif waiting:
                    stage, status = "空间单元拟合 / 等待依赖", "等待"
                elif _waiting_count(package_counts):
                    stage, status = "等待上游 Work Package", "等待"
                elif total and ready == total:
                    stage, status = "等待分区栅格收口", "等待"
                else:
                    stage, status = "等待计划", "等待"

                stage_progress = f"{ready}/{total}" if total else "-"
                activity = f"{running}/{waiting}"
                feature_count = 0
                elapsed = getattr(self, "_stream_state", {}).get(stream_id, {}).get(
                    "elapsed", "-"
                )
                if assembly_info:
                    assembly_current = int(
                        assembly_info.get("progress_current") or 0
                    )
                    assembly_total = int(
                        assembly_info.get("progress_total") or 0
                    )
                    phase_index = int(assembly_info.get("phase_index") or 0)
                    phase_total = int(assembly_info.get("phase_total") or 0)
                    stage_progress = (
                        f"{assembly_current}/{assembly_total}"
                        if assembly_total
                        else f"步骤 {phase_index}/{phase_total}"
                    )
                    activity = "—"
                    feature_count = int(
                        assembly_info.get("feature_count") or 0
                    )
                    phase_started = _timestamp_epoch(
                        assembly_info.get("phase_started_at") or ""
                    )
                    if phase_started is not None:
                        elapsed = _elapsed_text(time.time() - phase_started)

                self._set_stream(
                    stream_id,
                    stage=stage,
                    unit_progress=f"{ready}/{total}" if total else "-",
                    stage_progress=stage_progress,
                    activity=activity,
                    feature_count=feature_count,
                    status=status,
                    failures=failed + (1 if assembly_status == "failed" else 0),
                    elapsed=elapsed,
                )

            self._update_database_overviews(
                run_status=run_status,
                package_counts=package_counts,
                unit_job_counts=unit_job_counts,
                active_package=active_package,
                streams=streams,
                stream_runtime_progress=all_runtime_progress,
            )
            self._render_selected_tiles()
        except Exception as error:
            self._log_panel.append_system(f"[monitor-db] {error}")

    def _database_phase(
        self, run_status, package_counts, unit_job_counts, streams
    ):
        package_total = sum(int(value) for value in package_counts.values())
        package_ready = int(package_counts.get("ready", 0))
        package_active = int(package_counts.get("running", 0))
        package_waiting = _waiting_count(package_counts)
        unit_total = sum(int(value) for value in unit_job_counts.values())
        unit_ready = int(unit_job_counts.get("ready", 0))
        unit_active = int(unit_job_counts.get("running", 0))
        unit_waiting = _waiting_count(unit_job_counts)
        stream_total = len(streams)
        stream_ready = sum(
            1 for stream in streams if str(stream.get("status")) == "ready"
        )
        raster_ready = sum(
            1
            for stream in streams
            if str(stream.get("status")) in {"raster_ready", "ready"}
        )

        if run_status == "ready":
            return "ready", "已完成", 1, 1
        if run_status == "failed":
            package_failed = int(package_counts.get("failed", 0))
            if package_failed:
                return (
                    "package_failed",
                    "Work Package 失败，后续计算已停止",
                    package_ready + package_failed,
                    package_total,
                )
            return "failed", "运行失败", 0, 1
        if run_status == "stopped":
            return "stopped", "已停止，可安全恢复", 0, 1
        if run_status == "resetting":
            return "resetting", "正在重置失败 Work Package", 0, 0
        package_failed = int(package_counts.get("failed", 0))
        if package_failed:
            return (
                "package_failed",
                "Work Package 失败，后续计算已停止",
                package_ready + package_failed,
                package_total,
            )
        if self._active_global_stage == "分区概率栅格收口":
            return "finalize", "分区概率栅格收口", raster_ready, stream_total
        if self._active_global_stage == "并行组装":
            return "assembly", "结果流并行组装", stream_ready, stream_total
        if self._active_global_stage == "整体验收":
            return "acceptance", "整体验收", 0, 0
        if package_active or package_waiting:
            parallel = bool(unit_active)
            title = (
                "Work Package 推理 + 空间单元拟合"
                if parallel
                else "Work Package 推理"
            )
            return "packages", title, package_ready, package_total
        if unit_active or unit_waiting:
            return "units", "空间单元拟合", unit_ready, unit_total
        if int(unit_job_counts.get("failed", 0)):
            return "unit_failed", "空间单元失败处理", unit_ready, unit_total
        if stream_total and stream_ready == stream_total:
            return "acceptance", "整体验收", 0, 0
        if raster_ready:
            return "assembly", "结果流并行组装", stream_ready, stream_total
        return "finalize", "分区概率栅格收口", raster_ready, stream_total

    def _update_database_overviews(
        self,
        *,
        run_status,
        package_counts,
        unit_job_counts,
        active_package,
        streams,
        stream_runtime_progress,
    ):
        stage_key, stage, current, total = self._database_phase(
            run_status, package_counts, unit_job_counts, streams
        )
        if stage_key != self._stage_key:
            self._stage_key = stage_key
            self._stage_started_at = time.monotonic()
        stage_elapsed = _elapsed_text(time.monotonic() - self._stage_started_at)
        self._phase.setText(f"{stage} | 当前阶段观察 {stage_elapsed}")
        self.setWindowTitle(f"推理监控 - {stage}")
        rail_key = {
            "packages": "compute",
            "units": "compute",
            "unit_failed": "compute",
            "package_failed": "compute",
            "resetting": "compute",
            "stopped": "compute",
            "failed": (
                "assembly"
                if any(
                    str(item.get("status") or "") == "failed"
                    for item in stream_runtime_progress.values()
                )
                else "compute"
            ),
        }.get(stage_key, stage_key)
        self._update_stage_rail(rail_key)

        if stage_key == "assembly" and streams:
            assembly_units = round(
                sum(
                    _assembly_fraction(
                        stream.get("status"),
                        stream_runtime_progress.get(str(stream["stream_id"])) or {},
                    )
                    for stream in streams
                )
                * ASSEMBLY_PROGRESS_SCALE
            )
            stream_ready = sum(
                1 for stream in streams if str(stream.get("status")) == "ready"
            )
            stream_running = sum(
                1
                for stream in streams
                if str(stream.get("status")) == "assembling"
            )
            self._bar.setRange(0, len(streams) * ASSEMBLY_PROGRESS_SCALE)
            self._bar.setValue(assembly_units)
            self._bar.setFormat(
                f"{stage} | 完成 {stream_ready}/{len(streams)} | "
                f"运行 {stream_running}"
            )
        elif total > 0:
            self._bar.setRange(0, total)
            self._bar.setValue(min(current, total))
            self._bar.setFormat(f"{stage}  {current}/{total}")
        elif run_status in {"ready", "failed", "stopped"}:
            self._bar.setRange(0, 1)
            self._bar.setValue(1 if run_status == "ready" else 0)
            self._bar.setFormat(stage)
        else:
            self._bar.setRange(0, 0)
            self._bar.setFormat(stage)

        device = str(
            (self._run_spec.get("runtime") or {}).get("effective_device") or "未知"
        ).upper()
        monitor_elapsed = _elapsed_text(
            time.monotonic() - self._monitor_started_at
        )
        run_age = (
            _elapsed_text(time.time() - self._run_created_epoch)
            if self._run_created_epoch is not None
            else "—"
        )
        self._run_overview.setText(
            f"Run：{self._run_id} | 状态："
            f"{RUN_STATUS_LABELS.get(run_status, run_status)} | "
            f"设备：{device} | 创建至今：{run_age} | "
            f"本次监控：{monitor_elapsed}"
        )

        package_total = sum(int(value) for value in package_counts.values())
        package_ready = int(package_counts.get("ready", 0))
        package_running = int(package_counts.get("running", 0))
        package_waiting = _waiting_count(package_counts)
        package_failed = int(package_counts.get("failed", 0))
        current_text = "当前：—"
        if active_package is not None:
            activity = self._package_activity
            sequence = int(active_package.get("sequence_no") or 0) + 1
            package_id = str(active_package.get("package_id") or "")
            stream_id = str(activity.get("stream_id") or "")
            model_text = (
                self._stream_display_name(stream_id) if stream_id else "准备模型"
            )
            if "Fusion" in str(activity.get("status") or ""):
                model_text = "Fusion / 收口"
            tile_current = int(
                activity.get("tile_current")
                if activity.get("tile_current") is not None
                else activity.get("db_current") or 0
            )
            tile_total = int(
                activity.get("tile_total")
                if activity.get("tile_total") is not None
                else activity.get("db_total") or 0
            )
            configured = int(activity.get("configured_batch_size") or 0)
            effective = int(activity.get("effective_batch_size") or configured)
            batch_text = "—"
            if configured:
                batch_text = (
                    str(configured)
                    if not effective or effective == configured
                    else f"{configured}→{effective}"
                )
            if activity.get("started_at") is not None:
                package_elapsed = time.monotonic() - float(activity["started_at"])
            elif activity.get("started_epoch") is not None:
                package_elapsed = time.time() - float(activity["started_epoch"])
            else:
                package_elapsed = 0
            tile_text = f"{tile_current}/{tile_total}" if tile_total else "—"
            current_text = (
                f"当前包序号 {sequence}/{package_total}：{package_id} | "
                f"阶段：{activity.get('status') or '运行中'} | "
                f"模型：{model_text} | Tile：{tile_text} | "
                f"Batch：{batch_text} | 包耗时：{_elapsed_text(package_elapsed)}"
            )
        self._package_overview.setText(
            f"GPU / Work Package：完成 {package_ready}/{package_total} | "
            f"运行 {package_running} | 等待 {package_waiting} | "
            f"失败 {package_failed} | {current_text}"
        )

        unit_total = sum(int(value) for value in unit_job_counts.values())
        unit_ready = int(unit_job_counts.get("ready", 0))
        unit_running = int(unit_job_counts.get("running", 0))
        unit_waiting = _waiting_count(unit_job_counts)
        unit_failed = int(unit_job_counts.get("failed", 0))
        blocker_text = (
            f"阻塞（上游 Work Package 失败 {package_failed}） | "
            if package_failed
            else ""
        )
        scaling = self._run_spec.get("scaling") or {}
        worker_key = (
            "max_cpu_partition_workers_with_package"
            if package_running
            else "max_cpu_partition_workers"
        )
        worker_limit = scaling.get(worker_key)
        worker_text = str(worker_limit) if worker_limit is not None else "—"
        stream_ready = sum(
            1 for stream in streams if str(stream.get("status")) == "ready"
        )
        self._unit_overview.setText(
            f"CPU / 空间单元：{blocker_text}完成 {unit_ready}/{unit_total} | "
            f"运行 {unit_running} | 等待 {unit_waiting} | 失败 {unit_failed} | "
            f"并发上限 {worker_text} | 已组装结果流 {stream_ready}/{len(streams)}"
        )

        assembly_running = sum(
            1 for stream in streams if str(stream.get("status")) == "assembling"
        )
        assembly_failed = sum(
            1
            for stream in streams
            if str(
                (stream_runtime_progress.get(str(stream["stream_id"])) or {}).get(
                    "status"
                )
                or ""
            )
            == "failed"
        )
        assembly_waiting = max(
            0,
            len(streams) - stream_ready - assembly_running - assembly_failed,
        )
        assembly_limit = int(
            (self._run_spec.get("scaling") or {}).get(
                "max_concurrent_assembly", 2
            )
            or 2
        )
        active_phases = []
        for stream in streams:
            stream_id = str(stream["stream_id"])
            info = stream_runtime_progress.get(stream_id) or {}
            if str(info.get("status") or "") != "running":
                continue
            active_phases.append(
                f"{self._stream_display_name(stream_id)}："
                f"{info.get('phase_name') or '并行组装'}"
            )
        active_text = " | 当前 " + "；".join(active_phases) if active_phases else ""
        self._assembly_overview.setText(
            f"结果流组装：完成 {stream_ready}/{len(streams)} | "
            f"运行 {assembly_running} | 等待 {assembly_waiting} | "
            f"失败 {assembly_failed} | 并发 {assembly_running}/{assembly_limit}"
            f"{active_text}"
        )

    def _update_selected_tile(self, stream_id, tile_id, state):
        if stream_id != self._selected_stream():
            return
        row = self._tile_rows.get(tile_id)
        if row is None:
            row = self._tiles.rowCount()
            self._tiles.insertRow(row)
            self._tile_rows[tile_id] = row
        for column, value in enumerate(
            (tile_id, state["status"], state["progress"], state["error"])
        ):
            item = self._tiles.item(row, column)
            if item is None:
                item = QTableWidgetItem()
                self._tiles.setItem(row, column, item)
            item.setText(str(value))
        self._tile_detail_title.setText(
            f"选中结果流：{stream_id} | Tile 详情（已记录 "
            f"{len(self._tile_state.get(stream_id, {}))} 个）"
        )

    def _on_finished(self, result):
        for stream in result.get("streams") or []:
            status = {"ready": "成功", "failed": "失败", "stopped": "已停止"}.get(stream.get("status"), stream.get("status"))
            self._set_stream(
                stream["stream_id"],
                status=status,
                failures=int(stream.get("failure_count") or 0),
            )
        self.mark_finished(
            "已完成"
            if result.get("success")
            else "已停止"
            if result.get("status") == "stopped"
            else "失败"
        )
        if self._database is not None and self._run_id:
            self._poll_database()
            self._poll_timer.stop()

    def _update_summary(self):
        states = [value["status"] for value in self._stream_state.values()]
        waiting = states.count("等待") + states.count("跳过")
        self._summary.setText(
            f"结果流: {len(states)}  |  完成: {states.count('成功')}  |  "
            f"运行: {states.count('运行中')}  |  等待: {waiting}  |  "
            f"停止: {states.count('已停止')}  |  失败: {states.count('失败')}"
        )

    def _request_stop(self):
        self.stop_requested.emit()

    def closeEvent(self, event):
        event.ignore()
        self.hide()
        parent = self.parent()
        if parent is not None and hasattr(parent, "show_monitor_btn"):
            parent.show_monitor_btn.setChecked(False)
            parent.show_monitor_btn.setText("推理监控")
