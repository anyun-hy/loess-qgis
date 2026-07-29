"""Modeless result-stream monitor with on-demand tile details."""

from __future__ import annotations

import re
import time

from qgis.PyQt.QtCore import QObject, QTimer, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QCheckBox,
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


def _stream_from_step(name: str) -> str:
    if name.startswith("model_batch:"):
        return "model:" + name.split(":", 1)[1]
    if name.startswith("fusion_batch:"):
        return "fusion:" + name.split(":", 1)[1]
    for prefix in ("mosaic:", "polygonize:", "subpixel_vectorize:", "difference:"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return ""


def _tile_sort_key(tile_id: str):
    return tuple(
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", str(tile_id))
    )


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
        self._phase = QLabel("准备中")
        self._phase.setWordWrap(True)
        root.addWidget(self._phase)

        splitter = QSplitter(HORIZONTAL)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self._streams = QTableWidget(0, 6)
        self._streams.setHorizontalHeaderLabels(
            ["结果流", "阶段", "当前/总数", "状态", "耗时", "失败数"]
        )
        self._streams.verticalHeader().setVisible(False)
        self._streams.setEditTriggers(NO_EDIT_TRIGGERS)
        self._streams.setSelectionBehavior(SELECT_ROWS)
        self._streams.setSelectionMode(SINGLE_SELECTION)
        self._streams.itemSelectionChanged.connect(self._render_selected_tiles)
        header = self._streams.horizontalHeader()
        header.setMinimumSectionSize(56)
        header.setSectionResizeMode(0, STRETCH)
        for column, width in ((1, 92), (2, 86), (3, 70), (4, 76), (5, 62)):
            header.setSectionResizeMode(column, INTERACTIVE)
            header.resizeSection(column, width)
        left_layout.addWidget(self._streams, stretch=2)

        self._tile_detail_title = QLabel("选中结果流：未选择 | 空间单元详情")
        left_layout.addWidget(self._tile_detail_title)
        detail_controls = QHBoxLayout()
        self._detail_kind = QComboBox()
        self._detail_kind.addItem("Partition / Seam / Junction", "unit")
        self._detail_kind.addItem("Tile", "tile")
        self._detail_status = QComboBox()
        self._detail_status.addItem("全部状态", "")
        for text, value in (("等待", "queued"), ("运行中", "running"), ("完成", "ready"), ("失败", "failed")):
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
        splitter.addWidget(left)

        self._log_panel = LogPanel(self)
        splitter.addWidget(self._log_panel)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 4)
        splitter.setSizes([650, 530])
        root.addWidget(splitter, stretch=1)

        self._summary = QLabel("结果流: 0  |  成功: 0  |  失败: 0")
        root.addWidget(self._summary)

        bottom = QHBoxLayout()
        self._stop = QPushButton("停止")
        self._stop.clicked.connect(self._request_stop)
        bottom.addWidget(self._stop)
        self._bar = QProgressBar()
        bottom.addWidget(self._bar, stretch=1)
        self._cb_out = QCheckBox("stdout")
        self._cb_err = QCheckBox("stderr")
        self._cb_sys = QCheckBox("系统")
        for checkbox in (self._cb_out, self._cb_err, self._cb_sys):
            checkbox.setChecked(True)
            checkbox.toggled.connect(self._apply_filter)
            bottom.addWidget(checkbox)
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

    def bind_state_database(self, database_path, run_id, *, page_size=500):
        self._database = RunStateDB(database_path)
        self._run_id = str(run_id)
        self._page_size = max(1, min(int(page_size), 500))
        self._page = 0
        self._poll_timer.start()
        self._poll_database()

    def unbind_state_database(self):
        if hasattr(self, "_poll_timer"):
            self._poll_timer.stop()
        self._database = None
        self._run_id = ""
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
        self._phase.setText("准备中")
        self._tile_detail_title.setText("选中结果流：未选择 | 空间单元详情")
        self._summary.setText("结果流: 0  |  成功: 0  |  失败: 0")
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
        self.setWindowTitle(f"推理监控 - {text}")

    def _ensure_stream(self, stream_id):
        if stream_id in self._stream_rows:
            return self._stream_rows[stream_id]
        row = self._streams.rowCount()
        self._streams.insertRow(row)
        self._stream_rows[stream_id] = row
        self._stream_state[stream_id] = {
            "stage": "等待", "progress": "-", "status": "等待", "elapsed": "-", "failures": 0,
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
        values = [stream_id, state["stage"], state["progress"], state["status"], state["elapsed"], str(state["failures"])]
        for column, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            if column == 3:
                item.setForeground(QColor(STATUS_COLORS.get(str(value), "#333333")))
            self._streams.setItem(row, column, item)

    def _on_log(self, level, message):
        if level == "stdout":
            self._log_panel.append_stdout(message)
        elif level == "stderr":
            self._log_panel.append_stderr(message)
        else:
            self._log_panel.append_system(message)

    def _on_step_started(self, name):
        stream_id = _stream_from_step(name)
        self._step_started_at[name] = time.time()
        if stream_id:
            self._set_stream(stream_id, stage=name.split(":", 1)[0], status="运行中")

    def _on_step_finished(self, name, return_code, result):
        stream_id = str(result.get("stream_id") or _stream_from_step(name))
        if not stream_id:
            return
        started = self._step_started_at.pop(name, None)
        elapsed = time.time() - started if started else float(result.get("elapsed_sec") or 0)
        status = "成功" if result.get("success") else "跳过" if result.get("skipped") else "失败"
        changes = {"status": status, "elapsed": f"{elapsed:.1f}s"}
        if status == "失败":
            changes["failures"] = int(self._stream_state.get(stream_id, {}).get("failures", 0)) + 1
        self._set_stream(stream_id, **changes)

    def _on_stream_progress(self, info):
        stream_id = str(info.get("stream_id") or "")
        if not stream_id:
            return
        event = str(info.get("event") or "")
        current = int(info.get("current") or 0)
        total = int(info.get("total") or 0)
        failure = str(info.get("error") or "")
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

    def _selected_stream(self):
        row = self._streams.currentRow()
        if row < 0:
            return ""
        item = self._streams.item(row, 0)
        return item.text() if item else ""

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

    def _reset_detail_page(self, *_args):
        self._page = 0
        self._render_selected_tiles()

    def _previous_detail_page(self):
        if self._page > 0:
            self._page -= 1
            self._render_selected_tiles()

    def _next_detail_page(self):
        if self._next_page.isEnabled():
            self._page += 1
            self._render_selected_tiles()

    def _render_database_page(self, stream_id):
        self._tile_rows.clear()
        self._tiles.setRowCount(0)
        if not stream_id:
            self._tile_detail_title.setText("选中结果流：未选择 | 空间单元详情")
            return
        kind = str(self._detail_kind.currentData() or "unit")
        status = str(self._detail_status.currentData() or "")
        search = self._detail_search.text().strip()
        offset = self._page * self._page_size
        try:
            if kind == "tile":
                total = self._database.count_tiles(
                    self._run_id, status=status or None
                )
                rows = self._database.page_tiles(
                    self._run_id,
                    limit=self._page_size,
                    offset=offset,
                    status=status or None,
                    search=search,
                )
                self._tiles.setHorizontalHeaderLabels(
                    ["Tile", "Partition", "状态", "失败原因"]
                )
                values = [
                    (row["tile_id"], row.get("partition_id") or "-", row["status"], "")
                    for row in rows
                ]
                detail_name = "Tile"
            else:
                total = self._database.count_stream_units(
                    self._run_id, stream_id, status=status, search=search
                )
                rows = self._database.page_stream_units(
                    self._run_id,
                    stream_id,
                    limit=self._page_size,
                    offset=offset,
                    status=status,
                    search=search,
                )
                self._tiles.setHorizontalHeaderLabels(
                    ["空间单元", "类型", "状态", "失败原因"]
                )
                values = [
                    (row["unit_id"], row["unit_type"], row["status"], row["error"])
                    for row in rows
                ]
                detail_name = "Partition / Seam / Junction"
        except Exception as error:
            self._tile_detail_title.setText(
                f"选中结果流：{stream_id} | 数据库查询失败: {error}"
            )
            return
        self._tiles.setRowCount(len(values))
        for row_index, row_values in enumerate(values):
            for column, value in enumerate(row_values):
                self._tiles.setItem(row_index, column, QTableWidgetItem(str(value)))
        page_total = max(1, (total + self._page_size - 1) // self._page_size)
        if self._page >= page_total:
            self._page = page_total - 1
        self._page_label.setText(f"第 {self._page + 1}/{page_total} 页")
        self._previous_page.setEnabled(self._page > 0)
        self._next_page.setEnabled(self._page + 1 < page_total)
        self._tile_detail_title.setText(
            f"选中结果流：{stream_id} | {detail_name} 详情（共 {total} 条，每页最多 {self._page_size}）"
        )

    def _poll_database(self):
        if self._database is None or not self._run_id:
            return
        try:
            streams = self._database.stream_rows(self._run_id)
            for stream in streams:
                stream_id = str(stream["stream_id"])
                counts = self._database.stream_unit_counts(self._run_id, stream_id)
                total = sum(counts.values())
                completed = counts.get("ready", 0)
                failed = counts.get("failed", 0)
                status = (
                    "失败" if failed else "成功" if total and completed == total
                    else "运行中" if counts.get("running", 0) or completed
                    else "等待"
                )
                self._set_stream(
                    stream_id,
                    stage="分区 / Seam / Junction",
                    progress=f"{completed}/{total}" if total else "-",
                    status=status,
                    failures=failed,
                )
            self._render_selected_tiles()
        except Exception as error:
            self._log_panel.append_system(f"[monitor-db] {error}")

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
        self.mark_finished("已完成" if result.get("success") else "已停止" if result.get("stopped") else "失败")

    def _update_summary(self):
        states = [value["status"] for value in self._stream_state.values()]
        self._summary.setText(
            f"结果流: {len(states)}  |  成功: {states.count('成功')}  |  失败: {states.count('失败')}"
        )

    def _apply_filter(self):
        levels = set()
        if self._cb_out.isChecked():
            levels.add("stdout")
        if self._cb_err.isChecked():
            levels.add("stderr")
        if self._cb_sys.isChecked():
            levels.add("system")
        self._log_panel.set_visible_levels(levels)

    def _request_stop(self):
        self.stop_requested.emit()

    def closeEvent(self, event):
        event.ignore()
        self.hide()
        parent = self.parent()
        if parent is not None and hasattr(parent, "show_monitor_btn"):
            parent.show_monitor_btn.setChecked(False)
            parent.show_monitor_btn.setText("推理监控")
