"""Modeless result-stream monitor with on-demand tile details."""

from __future__ import annotations

import json
from html import escape
import re
import time
from datetime import datetime, timezone

from qgis.PyQt.QtCore import QObject, QThread, QTimer, QSize, pyqtSignal, pyqtSlot
from qgis.PyQt.QtGui import QColor, QFont, QFontDatabase
from qgis.PyQt.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QLayout,
    QSplitter,
    QTabWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qgis.core import QgsSettings

from ..qt6_api import (
    ALIGN_LEFT,
    ALIGN_TOP,
    ALIGN_VCENTER,
    HORIZONTAL,
    INTERACTIVE,
    NO_EDIT_TRIGGERS,
    SCROLLBAR_AS_NEEDED,
    RICH_TEXT,
    SELECT_ROWS,
    SINGLE_SELECTION,
    STRETCH,
    USER_ROLE,
    VERTICAL,
    WINDOW,
)

from .log_panel import LogPanel
from .monitor_time import format_monitor_timestamp, monitor_timezone_label
from ..core.monitor_contract import (
    ASSEMBLY_PHASES,
    ASSEMBLY_PHASE_NAMES,
    ASSEMBLY_PHASE_UNITS,
    MONITOR_EVENT_PAGE_SIZE,
    SPAN_STATUS_LABELS,
    effective_device_text,
    execution_trigger_label,
)


def _log_payload(message):
    text = str(message).strip()
    if not (text.startswith("{") and text.endswith("}")):
        return {}
    try:
        value = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _log_severity(level, message):
    """Separate semantic severity from the stdout/stderr/system source."""

    lowered = str(message).lower()
    if lowered.startswith("[resource-tuning] "):
        return "info"
    payload = _log_payload(message)
    event = str(payload.get("event") or "").lower()
    status = str(payload.get("status") or "").lower()
    if (
        event.endswith("failed")
        or status in {"failed", "error"}
        or payload.get("success") is False
    ):
        return "error"
    if any(
        token in event
        for token in ("warning", "retry", "reduced", "paused_low_disk")
    ) or status == "warning":
        return "warning"
    explicit_failure = any(
        marker in lowered
        for marker in (
            " failed (rc=",
            "[scheduler-error]",
            "[accelerator-restart]",
            " timed out after ",
            " process error:",
            "exhausted retries",
            "crashed repeatedly",
            "[monitor-db]",
            "fatal error",
        )
    )
    named_exception = (
        lowered.startswith(("error:", "[error]", "fatal:"))
        or re.search(r"\b[a-z_][\w.]*?(?:error|exception):", lowered)
    )
    if explicit_failure or (str(level) != "stderr" and named_exception):
        return "error"
    if any(
        token in lowered for token in ("warning", "warn", "警告")
    ) or any(
        marker in lowered
        for marker in ("[retry]", "fallback", "降档", "自动重试")
    ):
        return "warning"
    return "info"


def _log_fingerprint(severity, error, affected, attempt=0):
    if severity not in {"warning", "error"} or not str(affected).strip():
        return ""
    normalized_error = re.sub(r"\s+", " ", str(error)).strip().lower()
    normalized_target = re.sub(r"\s+", " ", str(affected)).strip().lower()
    return (
        f"{severity}:{normalized_target}:attempt={int(attempt or 0)}:"
        f"{normalized_error}"
    )


def _log_presentation(level, message):
    """Build a stable, readable summary while retaining the raw message."""

    source = str(level) if str(level) in {"stdout", "stderr", "system"} else "system"
    raw = str(message)
    lowered = raw.lower()
    payload = _log_payload(raw)
    severity = _log_severity(source, raw)
    event = str(payload.get("event") or "")
    error = str(payload.get("error") or raw)
    affected = str(
        payload.get("step")
        or payload.get("label")
        or payload.get("unit_id")
        or payload.get("stream_id")
        or event
        or ""
    )
    if not affected and " timed out after " in lowered:
        affected = raw[: lowered.index(" timed out after ")].strip()
    if not affected and " failed (rc=" in lowered:
        affected = raw[: lowered.index(" failed (rc=")].strip()
    if not affected and "scheduler-error" in lowered:
        affected = "调度器"
    if not affected and "monitor-db" in lowered:
        affected = "监控数据库"
    if not affected and "accelerator-restart" in lowered:
        affected = "加速器进程"
    attempt = int(payload.get("attempt") or 0)

    if severity == "warning":
        if "retry" in lowered or "重试" in raw or "reduced" in lowered or "降档" in raw:
            title = "任务正在自动重试"
            system_action = "系统已调整本次执行并继续运行"
            user_action = "通常不需要处理；重复出现时再查看技术详情"
        elif "paused_low_disk" in lowered or "低磁盘" in raw:
            title = "磁盘空间不足，任务已暂停"
            system_action = "系统保留当前进度，等待空间恢复"
            user_action = "释放磁盘空间后恢复任务"
        else:
            title = "运行警告"
            system_action = "系统继续运行并保留该警告"
            user_action = "通常不需要处理；重复出现时再检查"
    elif severity == "error":
        if "timed out after" in lowered or "超时" in raw:
            title = "任务处理超时"
            system_action = "进程已终止，系统将按恢复规则处理"
            user_action = "等待自动重试；若再次失败，再查看技术详情"
        elif "scheduler-error" in lowered:
            title = "任务调度异常"
            system_action = "系统已停止本次调度操作"
            user_action = "查看技术详情，修复后恢复任务"
        elif "monitor-db" in lowered:
            title = "监控状态读取失败"
            system_action = "推理任务不受影响，监控稍后会再次读取"
            user_action = "若持续出现，再检查 PostgreSQL 连接"
        elif "process error" in lowered:
            title = "进程启动失败"
            system_action = "本次进程没有继续执行"
            user_action = "查看技术详情并检查运行环境"
        elif "assembly" in event.lower() or "assemble" in lowered:
            title = "结果流组装失败"
            system_action = "本次结果流已标记为失败"
            user_action = "查看技术详情，修复后恢复该结果流"
        elif "coverage" in event.lower():
            title = "结果完整性验收失败"
            system_action = "结果未被发布为权威版本"
            user_action = "检查空白、重叠和范围外统计"
        elif "failed (rc=" in lowered:
            title = "进程异常退出"
            system_action = "本次任务已标记为失败"
            user_action = "查看技术详情中的返回码和原始输出"
        else:
            title = "任务执行失败"
            system_action = "本次任务已记录为失败"
            user_action = "查看技术详情，修复后再恢复任务"
    else:
        title = ""
        system_action = ""
        user_action = ""

    fingerprint = _log_fingerprint(severity, error, affected, attempt)
    return {
        "source": source,
        "severity": severity,
        "title": title,
        "affected": affected,
        "system_action": system_action,
        "user_action": user_action,
        "error": error,
        "attempt": attempt,
        "fingerprint": fingerprint,
    }


def _log_indicators(level, message):
    """Compatibility helper used by existing monitor tests."""

    severity = _log_severity(level, message)
    return severity == "warning", severity == "error"
from ..core.run_state_db import run_state_from_spec


STATUS_COLORS = {
    "等待": "#777777",
    "运行中": "#1565c0",
    "成功": "#2e7d32",
    "失败": "#c62828",
    "跳过": "#777777",
    "已停止": "#9a6700",
}

ASSEMBLY_PROGRESS_SCALE = 1000
STREAM_TABLE_VISIBLE_ROWS = 5
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
    "package": (
        ("全部状态", ""),
        ("等待", "queued"),
        ("待恢复", "interrupted"),
        ("正在重置", "resetting"),
        ("运行中", "running"),
        ("完成", "ready"),
        ("失败", "failed"),
    ),
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

MONITOR_THEME_SETTING = "plugins/labeling_tool/inference_monitor_theme"

from .monitor_theme import (
    BODY_FONT_PT,
    MONITOR_STYLE,
    OVERVIEW_ICON_SIZE,
    PALETTES,
    TABLE_HEADER_MIN_HEIGHT,
    TABLE_ROW_MIN_HEIGHT,
    status_color,
)
from .monitor_widgets import AdaptiveTable, MonitorComboBox, MonitorTextBrowser, ProgressTrack, monitor_icon


def _monitor_panel(*, secondary=False):
    panel = QFrame()
    panel.setProperty("monitorSubPanel" if secondary else "monitorPanel", True)
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(8)
    layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
    return panel, layout


def _section_label(text):
    label = QLabel(str(text))
    label.setProperty("sectionTitle", True)
    return label


def _muted_label(text=""):
    label = QLabel(str(text))
    label.setProperty("muted", True)
    label.setWordWrap(True)
    label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
    return label


def _scrollable_monitor_page(page, *, minimum_height):
    """Keep dense pages usable at the supported 900x560 minimum size."""

    page.setMinimumHeight(int(minimum_height))
    scroll = QScrollArea()
    scroll.setObjectName("MonitorScroll")
    scroll.setWidgetResizable(True)
    scroll.setWidget(page)
    return scroll


def _left_align_table_headers(table):
    """Keep header labels on the same content edge as left-aligned cells."""

    alignment = ALIGN_LEFT | ALIGN_VCENTER
    table.horizontalHeader().setDefaultAlignment(alignment)
    for column in range(table.columnCount()):
        item = table.horizontalHeaderItem(column)
        if item is not None:
            item.setTextAlignment(alignment)


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


def _overall_completion_fraction(
    run_status,
    job_counts,
    job_progress,
    streams,
    stream_runtime_progress,
):
    """Return completion across planned task groups, not estimated time."""

    if str(run_status) == "ready":
        return 1.0, 1
    groups = []
    for job_type in (
        "work_package",
        "fragmentation_v33",
        "unit_confidence",
        "unit_fit",
    ):
        counts = job_counts.get(job_type) or {}
        progress = job_progress.get(job_type) or {}
        total = int(progress.get("total") or sum(int(v) for v in counts.values()))
        if total < 1:
            continue
        completed = float(
            progress.get("completed")
            if progress.get("completed") is not None
            else counts.get("ready", 0)
        )
        groups.append(min(1.0, max(0.0, completed / total)))
    if streams:
        raster_finalized = all(
            str(stream.get("status") or "") in {"raster_ready", "assembling", "ready"}
            for stream in streams
        )
        groups.append(1.0 if raster_finalized else 0.0)
        groups.append(
            sum(
                _assembly_fraction(
                    stream.get("status"),
                    stream_runtime_progress.get(str(stream["stream_id"])) or {},
                )
                for stream in streams
            )
            / len(streams)
        )
        groups.append(0.0)
    if not groups:
        return 0.0, 0
    return sum(groups) / len(groups), len(groups)


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


class _MonitorQueryWorker(QObject):
    """Serialize PostgreSQL monitor reads outside the QGIS GUI thread."""

    result_ready = pyqtSignal(object)
    query_failed = pyqtSignal(object)

    def __init__(self):
        super().__init__(None)
        self._database_key = None
        self._database = None

    def _database_for(self, request):
        spec = dict(request.get("run_spec") or {})
        key = (
            str(request.get("run_id") or ""),
            str(spec.get("state_db") or ""),
        )
        if key != self._database_key:
            self._database = run_state_from_spec(spec)
            self._database_key = key
        return self._database

    @pyqtSlot(object)
    def execute(self, request):
        value = dict(request or {})
        try:
            database = self._database_for(value)
            kind = str(value.get("kind") or "")
            run_id = str(value.get("run_id") or "")
            if kind == "snapshot":
                payload = {
                    "kind": kind,
                    "generation": int(value.get("generation") or 0),
                    "request_id": int(value.get("request_id") or 0),
                    "run_id": run_id,
                    "snapshot": database.monitor_snapshot(run_id),
                }
            elif kind == "detail":
                stream_id = str(value.get("stream_id") or "")
                detail_kind = str(value.get("detail_kind") or "unit")
                status = str(value.get("status") or "")
                search = str(value.get("search") or "")
                page_size = max(1, min(int(value.get("page_size") or 500), 500))
                page = max(0, int(value.get("page") or 0))
                if detail_kind == "tile":
                    total = database.count_tiles(
                        run_id,
                        status=status or None,
                        search=search,
                    )
                elif detail_kind == "unit":
                    total = database.count_stream_units(
                        run_id,
                        stream_id,
                        status=status,
                        search=search,
                    )
                else:
                    total = database.count_monitor_objects(
                        run_id,
                        kind=detail_kind,
                        stream_id=stream_id,
                        status=status,
                        search=search,
                    )
                page_total = max(1, (total + page_size - 1) // page_size)
                page = min(page, page_total - 1)
                offset = page * page_size
                if detail_kind == "tile":
                    rows = database.page_tiles(
                        run_id,
                        limit=page_size,
                        offset=offset,
                        status=status or None,
                        search=search,
                    )
                elif detail_kind == "unit":
                    rows = database.page_stream_units(
                        run_id,
                        stream_id,
                        limit=page_size,
                        offset=offset,
                        status=status,
                        search=search,
                    )
                else:
                    rows = database.page_monitor_objects(
                        run_id,
                        kind=detail_kind,
                        stream_id=stream_id,
                        limit=page_size,
                        offset=offset,
                        status=status,
                        search=search,
                    )
                payload = {
                    "kind": kind,
                    "generation": int(value.get("generation") or 0),
                    "request_id": int(value.get("request_id") or 0),
                    "run_id": run_id,
                    "stream_id": stream_id,
                    "detail_kind": detail_kind,
                    "status": status,
                    "search": search,
                    "page": page,
                    "page_size": page_size,
                    "page_total": page_total,
                    "total": int(total),
                    "rows": list(rows),
                }
            elif kind == "history":
                scope = str(value.get("scope") or "all")
                levels = (
                    ("warning", "error")
                    if scope in {"issues", "warnings"}
                    else ()
                )
                rows = database.page_monitor_events(
                    run_id,
                    before_event_id=value.get("before_event_id"),
                    execution_id=str(value.get("execution_id") or ""),
                    stream_id=str(value.get("stream_id") or ""),
                    object_id=str(value.get("object_id") or ""),
                    job_id=value.get("job_id"),
                    span_id=str(value.get("span_id") or ""),
                    scope=scope,
                    levels=levels,
                    search=str(value.get("search") or ""),
                    limit=int(value.get("page_size") or MONITOR_EVENT_PAGE_SIZE),
                )
                payload = {
                    "kind": kind,
                    "generation": int(value.get("generation") or 0),
                    "request_id": int(value.get("request_id") or 0),
                    "run_id": run_id,
                    "append": bool(value.get("append")),
                    "rows": list(rows),
                    "page_size": int(value.get("page_size") or MONITOR_EVENT_PAGE_SIZE),
                }
            elif kind == "object_history":
                object_id = str(value.get("object_id") or "")
                detail_kind = str(value.get("detail_kind") or "")
                spans = database.page_monitor_spans(
                    run_id,
                    object_id="" if detail_kind == "package" else object_id,
                    package_id=object_id if detail_kind == "package" else "",
                    stream_id=str(value.get("stream_id") or "") if detail_kind != "package" else "",
                    span_kind="job_attempt",
                    job_id=value.get("job_id"),
                    before_started_at=str(value.get("before_started_at") or ""),
                    before_span_id=str(value.get("before_span_id") or ""),
                    limit=200,
                )
                attempt_id = str(value.get("span_id") or (spans[0]["span_id"] if spans else ""))
                models = database.page_monitor_spans(
                    run_id, package_id=object_id, span_kind="package_model",
                    parent_span_id=attempt_id, limit=500,
                ) if detail_kind == "package" and attempt_id else []
                events = database.page_monitor_events(
                    run_id,
                    object_id=object_id,
                    stream_id=str(value.get("stream_id") or "") if detail_kind != "package" else "",
                    job_id=value.get("job_id"),
                    limit=200,
                )
                payload = {
                    "kind": kind,
                    "generation": int(value.get("generation") or 0),
                    "request_id": int(value.get("request_id") or 0),
                    "run_id": run_id,
                    "object_id": object_id,
                    "detail_kind": detail_kind,
                    "spans": list(spans),
                    "events": list(events),
                    "models": list(models),
                    "attempt_id": attempt_id,
                    "append": bool(value.get("append")),
                    "has_more": len(spans) == 200,
                }
            else:
                raise ValueError("unknown monitor query kind: " + kind)
            self.result_ready.emit(payload)
        except Exception as error:
            self.query_failed.emit(
                {
                    "kind": str(value.get("kind") or ""),
                    "generation": int(value.get("generation") or 0),
                    "request_id": int(value.get("request_id") or 0),
                    "run_id": str(value.get("run_id") or ""),
                    "error": f"{type(error).__name__}: {error}",
                }
            )


class InferenceMonitorDialog(QDialog):
    stop_requested = pyqtSignal()
    shutdown_finished = pyqtSignal()
    _query_requested = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("推理监控")
        self.setWindowFlags(WINDOW)
        self.resize(1680, 1040)
        self.setMinimumSize(900, 560)
        screen = parent.screen() if parent is not None else QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(min(1680, max(900, available.width() - 40)),
                        min(1040, max(560, available.height() - 60)))
        configured_theme = str(
            QgsSettings().value(MONITOR_THEME_SETTING, "dark") or "dark"
        ).lower()
        self._theme = configured_theme if configured_theme in MONITOR_STYLE else "dark"
        self._connected = []
        self._stream_rows = {}
        self._stream_state = {}
        self._tile_state = {}
        self._tile_rows = {}
        self._step_started_at = {}
        self._step_attempts = {}
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
        self._logged_error_texts = set()
        self._process_log_suppressions = {}
        self._detail_signature = None
        self._last_detail_requested_at = 0.0
        self._database_bound = False
        self._run_id = ""
        self._query_generation = 0
        self._query_serial = 0
        self._latest_snapshot_request_id = 0
        self._latest_detail_request_id = 0
        self._query_busy = False
        self._active_query = None
        self._pending_snapshot_query = None
        self._pending_detail_query = None
        self._pending_history_query = None
        self._pending_object_query = None
        self._latest_history_request_id = 0
        self._latest_object_request_id = 0
        self._history_rows = []
        self._history_cursor = None
        self._history_exhausted = False
        self._last_snapshot_at = None
        self._last_snapshot_error = ""
        self._selected_object = {}
        self._selection = {}
        self._selected_attempt = ""
        self._object_history_rows = []
        self._object_history_cursor = None
        self._models_for_attempt = []
        self._assembly_phase_statuses = {}
        self._control_state = ""
        self._terminal_run_status = ""
        self._compact_layout = None
        self._icon_labels = []
        self._icon_buttons = []
        self._page = 0
        self._page_size = 500
        self._build_ui()
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(1000)
        self._poll_timer.timeout.connect(self._poll_database)
        self._detail_search_timer = QTimer(self)
        self._detail_search_timer.setSingleShot(True)
        self._detail_search_timer.setInterval(300)
        self._detail_search_timer.timeout.connect(self._reset_detail_page)
        self._history_search_timer = QTimer(self)
        self._history_search_timer.setSingleShot(True)
        self._history_search_timer.setInterval(300)
        self._history_search_timer.timeout.connect(self._reset_history_page)
        self._query_thread = QThread(self)
        self._query_thread.setObjectName("loess-monitor-postgresql")
        self._query_worker = _MonitorQueryWorker()
        self._query_worker.moveToThread(self._query_thread)
        self._query_requested.connect(self._query_worker.execute)
        self._query_worker.result_ready.connect(self._on_query_result)
        self._query_worker.query_failed.connect(self._on_query_failed)
        self._query_thread.finished.connect(self._query_worker.deleteLater)
        self._query_thread.finished.connect(self._finish_shutdown)
        self._query_thread.start()

    def _icon(self, name, *, size=32, tone="text"):
        label = QLabel()
        label.setFixedSize(size, size)
        self._icon_labels.append((label, name, size, tone))
        return label

    def _icon_button(self, button, name, *, tone="muted", size=20):
        self._icon_buttons.append((button, name, size, tone))
        button.setIconSize(QSize(size, size))
        return button

    @staticmethod
    def _divider():
        line = QFrame()
        line.setObjectName("MonitorDivider")
        line.setFixedHeight(1)
        return line

    @staticmethod
    def _stat_pair(title, value="—"):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        title_label = _muted_label(title)
        title_label.setWordWrap(False)
        title_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )
        layout.addWidget(title_label)
        label = QLabel(value)
        label.setProperty("value", True)
        label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )
        layout.addWidget(label)
        return container, label

    def _build_ui(self):
        self.setObjectName("InferenceMonitor")
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 12, 24, 12)
        root.setSpacing(10)
        title_row = QHBoxLayout()
        title_row.setSpacing(14)
        title_row.addWidget(self._icon("contour", size=36, tone="accent"))
        title = QLabel("LOESS / 推理监控")
        title.setProperty("hero", True)
        title_row.addWidget(title, stretch=1)
        self._run_info_button = QPushButton("运行信息")
        self._run_info_button.setProperty("quiet", True)
        self._run_info_button.clicked.connect(self._show_run_information)
        title_row.addWidget(self._run_info_button)
        self._theme_toggle = QPushButton()
        self._theme_toggle.setObjectName("ThemeToggle")
        self._theme_toggle.setIconSize(QSize(22, 22))
        self._theme_toggle.setFixedSize(40, 40)
        self._theme_toggle.setAutoDefault(False)
        self._theme_toggle.setProperty("quiet", True)
        self._theme_toggle.clicked.connect(self._toggle_theme)
        title_row.addWidget(self._theme_toggle)
        root.addLayout(title_row)
        root.addWidget(self._divider())

        state_row = QHBoxLayout()
        state_row.setSpacing(20)
        self._status_badge = QLabel("准备中")
        self._status_badge.setProperty("status", "neutral")
        state_row.addWidget(self._status_badge, alignment=ALIGN_VCENTER)
        header_text = QVBoxLayout()
        header_text.setSpacing(6)
        self._phase = QLabel("准备创建运行")
        self._phase.setProperty("headline", True)
        self._phase.setWordWrap(True)
        self._run_id_label = _muted_label("任务：准备创建")
        header_text.addWidget(self._phase)
        header_text.addWidget(self._run_id_label)
        state_row.addLayout(header_text, stretch=1)
        self._stop = QPushButton("停止任务")
        self._stop.setObjectName("StopButton")
        self._icon_button(self._stop, "stop", tone="failed", size=24)
        self._stop.clicked.connect(self._request_stop)
        state_row.addWidget(self._stop)
        root.addLayout(state_row)
        self._run_overview = _muted_label("Run：准备中")
        self._run_overview.setParent(self)
        self._run_overview.hide()
        self._stage_rail = QLabel("", self)
        self._stage_rail.hide()
        self._update_stage_rail("compute")

        self._pages = QTabWidget()
        self._pages.setDocumentMode(True)
        self._pages.tabBar().setDrawBase(False)
        self._pages.currentChanged.connect(self._on_page_changed)
        self._pages.addTab(self._build_overview_page(), "总览")
        self._pages.addTab(self._build_detail_page(), "详细进度")
        self._pages.addTab(self._build_results_page(), "结果与验收")
        self._pages.addTab(self._build_events_page(), "事件与日志")
        root.addWidget(self._pages, stretch=1)

        footer = QHBoxLayout()
        footer.setSpacing(14)
        footer.addWidget(_section_label("本次推理完成度"))
        self._completion_value = QLabel("—")
        self._completion_value.setProperty("value", True)
        self._completion_value.setProperty("accent", True)
        footer.addWidget(self._completion_value)
        self._overall_bar = QProgressBar()
        self._overall_bar.setTextVisible(False)
        self._overall_bar.setFixedHeight(8)
        self._overall_bar.setRange(0, ASSEMBLY_PROGRESS_SCALE)
        self._overall_bar.setValue(0)
        self._overall_bar.setFormat("本次推理任务完成度：等待任务图")
        self._overall_bar.setToolTip("按任务组统计，不代表剩余时间。")
        self._overall_bar.valueChanged.connect(self._refresh_completion_label)
        footer.addWidget(self._overall_bar, stretch=1)
        footer_hint = _muted_label("按任务组统计，不代表剩余时间")
        footer_hint.setWordWrap(False)
        footer_hint.setMinimumWidth(178)
        footer.addWidget(footer_hint)
        root.addLayout(footer)
        status_row = QHBoxLayout()
        self._monitor_sync = _muted_label("监控：等待连接")
        status_row.addWidget(self._monitor_sync)
        status_row.addStretch()
        self._history_completeness = _muted_label("历史：等待正式 Run")
        status_row.addWidget(self._history_completeness)
        root.addLayout(status_row)
        self._summary = _muted_label("结果流：等待任务")
        self._summary.setParent(self)
        self._summary.hide()
        self._bar = QProgressBar(self)
        self._bar.setRange(0, 0)
        self._bar.hide()
        self._apply_theme(self._theme, persist=False)
        self._apply_responsive_layout(self.width())

    def _refresh_completion_label(self, *_args):
        total = self._overall_bar.maximum()
        self._completion_value.setText(
            f"{self._overall_bar.value() / total:.0%}" if total > 1 else "—"
        )
        self._overall_bar.setToolTip(self._overall_bar.format())

    def _build_overview_page(self):
        scroll = QScrollArea()
        scroll.setObjectName("MonitorScroll")
        scroll.setWidgetResizable(True)
        page = QWidget()
        page.setObjectName("MonitorPage")
        self._overview_scroll = scroll
        self._overview_page = page
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 10, 0, 4)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        body = QSplitter(HORIZONTAL)
        body.setHandleWidth(16)
        body.setChildrenCollapsible(False)
        body.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._overview_splitter = body
        main = QWidget()
        self._overview_main = main
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(16)
        main_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        parallel, parallel_layout = _monitor_panel()
        parallel_layout.setContentsMargins(14, 10, 14, 10)
        parallel_layout.setSpacing(6)
        parallel_heading = QHBoxLayout()
        parallel_heading.addWidget(_section_label("并行执行"))
        parallel_heading.addWidget(_muted_label("模型计算与空间处理协同进行"), stretch=1)
        parallel_layout.addLayout(parallel_heading)
        cards = QGridLayout()
        cards.setSpacing(14)
        self._overview_cards = cards

        model_card, model_layout = _monitor_panel(secondary=True)
        self._overview_model_card = model_card
        model_layout.setContentsMargins(14, 10, 14, 10)
        model_layout.setSpacing(4)
        model_heading = QHBoxLayout()
        model_heading.setSpacing(12)
        model_heading.addWidget(self._icon("chip", size=38))
        model_titles = QVBoxLayout()
        model_titles.setSpacing(2)
        model_titles.addWidget(_section_label("模型计算 · 地物识别"))
        self._device_label = _muted_label("执行设备：等待 Run 信息")
        model_titles.addWidget(self._device_label)
        model_heading.addLayout(model_titles, stretch=1)
        self._model_badge = QLabel("待开始")
        self._model_badge.setProperty("status", "neutral")
        model_heading.addWidget(self._model_badge)
        model_layout.addLayout(model_heading)
        self._package_metric = QLabel("— / —")
        self._package_metric.setProperty("metric", True)
        self._package_metric.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )
        model_layout.addWidget(self._package_metric)
        model_layout.addWidget(_muted_label("推理包已完成"))
        self._package_card_bar = ProgressTrack()
        model_layout.addWidget(self._package_card_bar)
        model_layout.addWidget(self._divider())
        model_stats = QHBoxLayout()
        package_stat, self._current_package = self._stat_pair("当前推理包")
        model_stat, self._current_model = self._stat_pair("当前模型")
        model_stats.addWidget(package_stat, stretch=1)
        model_stats.addWidget(model_stat, stretch=1)
        model_layout.addLayout(model_stats)
        tile_row = QHBoxLayout()
        tile_row.addWidget(_muted_label("当前模型影像块"))
        self._tile_count = QLabel("— / —")
        tile_row.addWidget(self._tile_count)
        tile_row.addStretch()
        model_layout.addLayout(tile_row)
        self._tile_card_bar = ProgressTrack()
        model_layout.addWidget(self._tile_card_bar)
        model_footer = QHBoxLayout()
        self._batch_value = _muted_label("批量大小：—")
        model_footer.addWidget(self._batch_value, stretch=1)
        package_button = QPushButton("查看推理包")
        package_button.setProperty("link", True)
        self._icon_button(package_button, "arrow", tone="accent")
        package_button.clicked.connect(lambda: self._open_detail_kind("package"))
        model_footer.addWidget(package_button)
        model_layout.addLayout(model_footer)
        self._package_overview = _muted_label("等待计划")
        self._package_overview.setParent(model_card)
        self._package_overview.hide()
        cards.addWidget(model_card, 0, 0)

        spatial_card, spatial_layout = _monitor_panel(secondary=True)
        self._overview_spatial_card = spatial_card
        spatial_layout.setContentsMargins(14, 10, 14, 10)
        spatial_layout.setSpacing(4)
        spatial_heading = QHBoxLayout()
        spatial_heading.setSpacing(12)
        spatial_heading.addWidget(self._icon("chip", size=38))
        spatial_titles = QVBoxLayout()
        spatial_titles.setSpacing(2)
        spatial_titles.addWidget(_section_label("空间处理 · 边界计算"))
        self._spatial_device_label = _muted_label("CPU")
        spatial_titles.addWidget(self._spatial_device_label)
        spatial_heading.addLayout(spatial_titles, stretch=1)
        self._spatial_badge = QLabel("待开始")
        self._spatial_badge.setProperty("status", "neutral")
        spatial_heading.addWidget(self._spatial_badge)
        spatial_layout.addLayout(spatial_heading)
        self._unit_metric = QLabel("— / —")
        self._unit_metric.setProperty("metric", True)
        self._unit_metric.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )
        spatial_layout.addWidget(self._unit_metric)
        self._fit_caption = _muted_label("边界拟合任务已完成")
        spatial_layout.addWidget(self._fit_caption)
        self._fit_bar = ProgressTrack()
        spatial_layout.addWidget(self._fit_bar)
        spatial_layout.addWidget(self._divider())
        spatial_stats = QHBoxLayout()
        self._spatial_stats = {}
        for title in ("运行", "等待", "失败"):
            box, label = self._stat_pair(title)
            self._spatial_stats[title] = label
            spatial_stats.addWidget(box, stretch=1)
        spatial_layout.addLayout(spatial_stats)
        self._fragment_label = _muted_label("碎片治理：等待计划")
        self._confidence_label = _muted_label("置信度计算：等待计划")
        self._fragment_bar = ProgressTrack(percentage=False)
        self._confidence_bar = ProgressTrack(percentage=False)
        self._fragment_bar.setFixedHeight(10)
        self._confidence_bar.setFixedHeight(10)
        lane_grid = QGridLayout()
        lane_grid.setHorizontalSpacing(16)
        lane_grid.setVerticalSpacing(4)
        for column, (label, bar) in enumerate(((self._fragment_label, self._fragment_bar), (self._confidence_label, self._confidence_bar))):
            lane_grid.addWidget(label, 0, column)
            lane_grid.addWidget(bar, 1, column)
            lane_grid.setColumnStretch(column, 1)
        spatial_layout.addLayout(lane_grid)
        spatial_footer = QHBoxLayout()
        self._unit_overview = _muted_label("统计范围：全部结果流")
        spatial_footer.addWidget(self._unit_overview, stretch=1)
        spatial_button = QPushButton("查看空间任务")
        spatial_button.setProperty("link", True)
        self._icon_button(spatial_button, "arrow", tone="accent")
        spatial_button.clicked.connect(lambda: self._open_detail_kind("unit_fit"))
        spatial_footer.addWidget(spatial_button)
        spatial_layout.addLayout(spatial_footer)
        self._fit_label = _muted_label("")
        self._fit_label.setParent(spatial_card)
        self._fit_label.hide()
        cards.addWidget(spatial_card, 0, 1)
        cards.setColumnStretch(0, 1)
        cards.setColumnStretch(1, 1)
        parallel_layout.addLayout(cards)
        main_layout.addWidget(parallel)

        result_panel, result_layout = _monitor_panel()
        self._overview_results_panel = result_panel
        result_layout.setContentsMargins(14, 10, 14, 10)
        result_layout.setSpacing(6)
        result_header = QHBoxLayout()
        result_header.addWidget(_section_label("模型与融合结果"), stretch=1)
        result_button = QPushButton("查看结果与验收")
        result_button.setProperty("link", True)
        self._icon_button(result_button, "arrow", tone="accent")
        result_button.clicked.connect(lambda: self._pages.setCurrentIndex(2))
        result_header.addWidget(result_button)
        result_layout.addLayout(result_header)
        self._overview_results = AdaptiveTable(0, 4)
        self._overview_results.setObjectName("OverviewResults")
        self._overview_results.setHorizontalHeaderLabels(["结果", "当前工作", "组装", "验收"])
        self._overview_results.verticalHeader().setVisible(False)
        self._overview_results.setEditTriggers(NO_EDIT_TRIGGERS)
        self._overview_results.setSelectionBehavior(SELECT_ROWS)
        self._overview_results.setSelectionMode(SINGLE_SELECTION)
        header = self._overview_results.horizontalHeader()
        _left_align_table_headers(self._overview_results)
        self._overview_results.configure_adaptive_columns(
            (168, 178, 132, 124), (1.0, 1.45, 0.85, 0.8), text_cap=360,
        )
        self._overview_results.fit_rows_to_content(max_rows=5)
        self._overview_results.itemSelectionChanged.connect(self._sync_overview_stream_selection)
        result_layout.addWidget(self._overview_results)
        self._assembly_overview = _muted_label("结果流组装：等待上游计算")
        self._coverage_overview = _muted_label("空白/重叠验收：等待组装")
        for label in (self._assembly_overview, self._coverage_overview):
            label.setParent(result_panel)
            label.hide()
        main_layout.addWidget(result_panel)
        # Keep the result card at its compact table height.  Extra splitter
        # height belongs to the main column, not a second empty-looking card.
        main_layout.addStretch(1)
        body.addWidget(main)

        activity, activity_layout = _monitor_panel()
        self._overview_activity_panel = activity
        activity_layout.setSpacing(16)
        self._overview_activity_panel = activity
        activity_layout.addWidget(_section_label("运行动态"))
        health = QFrame()
        health.setObjectName("MonitorHealth")
        self._health_panel = health
        health_layout = QHBoxLayout(health)
        health_layout.setContentsMargins(16, 20, 16, 20)
        health_layout.setSpacing(16)
        self._health_icon = self._icon("shield", size=48, tone="success")
        health_layout.addWidget(self._health_icon)
        health_text = QVBoxLayout()
        self._action_label = _section_label("等待运行状态")
        self._action_label.setWordWrap(True)
        health_text.addWidget(self._action_label)
        self._action_description = _muted_label("正式 Run 建立后显示运行动态。")
        health_text.addWidget(self._action_description)
        health_layout.addLayout(health_text, stretch=1)
        activity_layout.addWidget(health)
        self._history_health = _muted_label("历史记录：等待正式 Run")
        activity_layout.addWidget(self._history_health)
        log_counts = QHBoxLayout()
        self._warning_log_button = QPushButton("Warning 0")
        self._warning_log_button.clicked.connect(lambda: self._show_log_severity("warning"))
        self._error_log_button = QPushButton("Error 0")
        self._error_log_button.clicked.connect(lambda: self._show_log_severity("error"))
        log_counts.addWidget(self._warning_log_button)
        log_counts.addWidget(self._error_log_button)
        activity_layout.addLayout(log_counts)
        activity_layout.addWidget(self._divider())
        activity_layout.addWidget(_section_label("最近事件"))
        self._recent_events_label = _muted_label("尚无已记录事件")
        self._recent_events_label.setAlignment(ALIGN_TOP | ALIGN_LEFT)
        self._recent_events_label.setTextFormat(RICH_TEXT)
        activity_layout.addWidget(self._recent_events_label)
        activity_layout.addStretch(1)
        open_events = QPushButton("打开事件与日志")
        self._icon_button(open_events, "document", size=22, tone="text")
        open_events.clicked.connect(lambda: self._pages.setCurrentIndex(3))
        activity_layout.addWidget(open_events)
        body.addWidget(activity)
        body.setStretchFactor(0, 7)
        body.setStretchFactor(1, 3)
        body.setSizes([1060, 400])
        layout.addWidget(body)
        scroll.setWidget(page)
        return scroll

    def _build_detail_page(self):
        page = QWidget()
        page.setObjectName("MonitorPage")
        layout = QHBoxLayout(page)
        self._detail_page_layout = layout
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)
        navigation, navigation_layout = _monitor_panel()
        self._detail_navigation_panel = navigation
        navigation_layout.addWidget(_section_label("分类"))
        self._detail_navigation = QListWidget()
        for title, kind in (
            ("模型计算 / 推理包", "package"),
            ("空间处理 / 碎片治理", "fragmentation_v33"),
            ("空间处理 / 置信度计算", "unit_confidence"),
            ("空间处理 / 边界拟合", "unit_fit"),
            ("全局输入 / Tile", "tile"),
        ):
            item = QListWidgetItem(title)
            item.setData(USER_ROLE, kind)
            self._detail_navigation.addItem(item)
        self._detail_navigation.currentRowChanged.connect(
            self._on_detail_navigation_changed
        )
        navigation_layout.addWidget(self._detail_navigation, stretch=1)
        layout.addWidget(navigation)

        detail, detail_layout = _monitor_panel()
        self._detail_content_panel = detail
        self._tile_detail_title = _section_label(
            "当前范围：全局 / 推理包"
        )
        detail_layout.addWidget(self._tile_detail_title)
        controls = QHBoxLayout()
        self._detail_kind = MonitorComboBox()
        for title, kind in (
            ("推理包", "package"),
            ("碎片治理", "fragmentation_v33"),
            ("置信度计算", "unit_confidence"),
            ("边界拟合", "unit_fit"),
            ("Tile 输入", "tile"),
        ):
            self._detail_kind.addItem(title, kind)
        self._detail_kind.setVisible(False)
        self._detail_status = MonitorComboBox()
        for text, value in DETAIL_STATUS_OPTIONS["package"]:
            self._detail_status.addItem(text, value)
        self._detail_search = QLineEdit()
        self._detail_search.setPlaceholderText("搜索包、任务或单元 ID")
        self._previous_page = QPushButton("上一页")
        self._next_page = QPushButton("下一页")
        self._page_label = _muted_label("第 1 页")
        controls.addWidget(self._detail_status)
        controls.addWidget(self._detail_search, stretch=1)
        controls.addWidget(self._previous_page)
        controls.addWidget(self._next_page)
        controls.addWidget(self._page_label)
        detail_layout.addLayout(controls)
        self._tiles = AdaptiveTable(0, 5)
        self._tiles.setHorizontalHeaderLabels(
            ["对象ID", "类型", "执行状态", "产物状态", "原因"]
        )
        self._tiles.verticalHeader().setVisible(False)
        self._tiles.setEditTriggers(NO_EDIT_TRIGGERS)
        self._tiles.setSelectionBehavior(SELECT_ROWS)
        self._tiles.setSelectionMode(SINGLE_SELECTION)
        self._tiles.setHorizontalScrollBarPolicy(SCROLLBAR_AS_NEEDED)
        self._tiles.setVerticalScrollBarPolicy(SCROLLBAR_AS_NEEDED)
        self._tiles.itemSelectionChanged.connect(self._render_object_detail)
        tile_header = self._tiles.horizontalHeader()
        _left_align_table_headers(self._tiles)
        tile_header.setMinimumSectionSize(72)
        self._tiles.configure_adaptive_columns(
            (178, 124, 100, 100, 154), (1.2, 0.85, 0.7, 0.75, 1.45),
            text_cap=420,
        )
        self._tiles.fit_rows_to_content(max_rows=8)
        detail_layout.addWidget(self._tiles, stretch=3)
        self._object_tabs = QTabWidget()
        self._object_current = MonitorTextBrowser()
        self._object_models = MonitorTextBrowser()
        self._object_attempts = MonitorTextBrowser()
        self._object_attempts.setOpenLinks(False)
        self._object_attempts.anchorClicked.connect(self._select_history_attempt)
        self._object_events = MonitorTextBrowser()
        self._object_tabs.addTab(self._object_current, "当前进度")
        self._object_tabs.addTab(self._object_models, "模型明细")
        self._object_tabs.addTab(self._object_attempts, "历史尝试")
        self._object_tabs.addTab(self._object_events, "相关事件")
        detail_layout.addWidget(self._object_tabs, stretch=2)
        object_actions = QHBoxLayout()
        self._object_more = QPushButton("加载更早尝试")
        self._object_more.setEnabled(False)
        self._object_more.clicked.connect(lambda: self._request_object_history(self._selection.get("object_id", ""), append=True))
        object_actions.addWidget(self._object_more)
        related = QPushButton("查看关联事件")
        related.clicked.connect(self._open_object_events)
        object_actions.addWidget(related)
        copy_id = QPushButton("复制 ID")
        copy_id.clicked.connect(lambda: QApplication.clipboard().setText(str(self._selection.get("object_id") or "")))
        object_actions.addWidget(copy_id)
        object_actions.addStretch()
        detail_layout.addLayout(object_actions)
        self._detail_kind.currentIndexChanged.connect(self._reset_detail_page)
        self._detail_status.currentIndexChanged.connect(self._reset_detail_page)
        self._detail_search.textChanged.connect(self._schedule_detail_search)
        self._previous_page.clicked.connect(self._previous_detail_page)
        self._next_page.clicked.connect(self._next_detail_page)
        layout.addWidget(detail, stretch=1)
        self._detail_navigation.setCurrentRow(0)
        return _scrollable_monitor_page(page, minimum_height=470)

    def _build_results_page(self):
        page = QWidget()
        page.setObjectName("MonitorPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)
        streams_panel, streams_layout = _monitor_panel()
        streams_layout.addWidget(_section_label("结果流"))
        self._streams = AdaptiveTable(0, 7)
        self._streams.setHorizontalHeaderLabels(
            ["结果流", "当前阶段", "本步进度", "运行/等待", "输出面数", "问题", "阶段耗时"]
        )
        self._streams.verticalHeader().setVisible(False)
        self._streams.setEditTriggers(NO_EDIT_TRIGGERS)
        self._streams.setSelectionBehavior(SELECT_ROWS)
        self._streams.setSelectionMode(SINGLE_SELECTION)
        self._streams.setHorizontalScrollBarPolicy(SCROLLBAR_AS_NEEDED)
        self._streams.setVerticalScrollBarPolicy(SCROLLBAR_AS_NEEDED)
        self._streams.itemSelectionChanged.connect(self._on_stream_selection_changed)
        header = self._streams.horizontalHeader()
        _left_align_table_headers(self._streams)
        header.setMinimumSectionSize(64)
        self._streams.configure_adaptive_columns(
            (164, 176, 110, 102, 96, 72, 106),
            (1.0, 1.3, 0.72, 0.7, 0.65, 0.45, 0.72), text_cap=360,
        )
        self._streams.fit_rows_to_content(max_rows=5)
        streams_layout.addWidget(self._streams)
        # The stream table is content-height bounded; reserve the flexible
        # vertical area for the lower step/detail work surface instead.
        layout.addWidget(streams_panel)

        lower = QSplitter(HORIZONTAL)
        self._results_splitter = lower
        steps_panel, steps_layout = _monitor_panel()
        steps_layout.addWidget(_section_label("组装步骤"))
        self._assembly_steps = AdaptiveTable(len(ASSEMBLY_PHASES), 3)
        self._assembly_steps.setHorizontalHeaderLabels(["步骤", "状态", "进度/单位"])
        self._assembly_steps.verticalHeader().setVisible(False)
        self._assembly_steps.setEditTriggers(NO_EDIT_TRIGGERS)
        _left_align_table_headers(self._assembly_steps)
        for row, (phase, name, unit) in enumerate(ASSEMBLY_PHASES):
            self._assembly_steps.setItem(row, 0, QTableWidgetItem(f"{row + 1}. {name}"))
            self._assembly_steps.setItem(row, 1, QTableWidgetItem("未开始"))
            self._assembly_steps.setItem(row, 2, QTableWidgetItem(f"— / {unit}"))
            self._assembly_steps.item(row, 0).setToolTip(phase)
        steps_header = self._assembly_steps.horizontalHeader()
        self._assembly_steps.configure_adaptive_columns(
            (172, 118, 128), (1.25, 0.8, 0.9), text_cap=300,
        )
        self._assembly_steps.fit_rows_to_content(max_rows=10)
        steps_layout.addWidget(self._assembly_steps)
        steps_layout.addStretch(1)
        lower.addWidget(steps_panel)

        acceptance_panel, acceptance_layout = _monitor_panel()
        acceptance_layout.addWidget(_section_label("当前步骤详情与验收"))
        self._assembly_detail = MonitorTextBrowser()
        self._assembly_detail.setText("选择一个结果流查看十步组装记录。")
        acceptance_layout.addWidget(self._assembly_detail, stretch=1)
        self._result_coverage = _muted_label(
            "覆盖验收：尚未执行\n空白面积：—  重叠面积：—  范围外面积：—\n"
            "本页不代表人工修整或 accepted_labels 已完成。"
        )
        acceptance_layout.addWidget(self._result_coverage)
        lower.addWidget(acceptance_panel)
        lower.setStretchFactor(0, 3)
        lower.setStretchFactor(1, 2)
        layout.addWidget(lower, stretch=1)
        return _scrollable_monitor_page(page, minimum_height=510)

    def _build_events_page(self):
        page = QWidget()
        page.setObjectName("MonitorPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        controls = QHBoxLayout()
        self._history_scope = MonitorComboBox()
        self._history_scope.addItem("全部事件", "all")
        self._history_scope.addItem("当前问题", "issues")
        self._history_scope.addItem("自动恢复", "recovery")
        self._history_scope.addItem("历史警告/失败", "warnings")
        self._history_execution = MonitorComboBox()
        self._history_execution.addItem("全部执行", "")
        self._history_target = MonitorComboBox()
        for title, value in (("整个 Run", "all"), ("当前结果流", "stream"), ("选中对象", "object"), ("选中尝试", "attempt")):
            self._history_target.addItem(title, value)
        self._history_search = QLineEdit()
        self._history_search.setPlaceholderText("搜索事件、对象或错误")
        self._log_toggle = QPushButton("显示原始日志")
        self._log_toggle.setCheckable(True)
        self._log_toggle.toggled.connect(self._set_log_visible)
        controls.addWidget(self._history_scope)
        controls.addWidget(self._history_execution)
        controls.addWidget(self._history_target)
        controls.addWidget(self._history_search, stretch=1)
        controls.addWidget(self._log_toggle)
        layout.addLayout(controls)
        self._history_context_label = _muted_label("范围：整个 Run · " + monitor_timezone_label())
        self._history_context_label.setToolTip("事件显示记录时间，日志显示采集时间；没有来源时间的日志会标注‘接收’。原始时间保留在技术详情中。")
        layout.addWidget(self._history_context_label)

        self._splitter = QSplitter(HORIZONTAL)
        history_panel, history_layout = _monitor_panel()
        self._history_table = AdaptiveTable(0, 5)
        self._history_table.setHorizontalHeaderLabels(
            ["时间", "对象", "事件", "执行", "当前关联状态"]
        )
        self._history_table.verticalHeader().setVisible(False)
        self._history_table.setEditTriggers(NO_EDIT_TRIGGERS)
        self._history_table.setSelectionBehavior(SELECT_ROWS)
        self._history_table.setSelectionMode(SINGLE_SELECTION)
        self._history_table.setHorizontalScrollBarPolicy(SCROLLBAR_AS_NEEDED)
        self._history_table.setVerticalScrollBarPolicy(SCROLLBAR_AS_NEEDED)
        self._history_table.itemSelectionChanged.connect(self._render_history_detail)
        history_header = self._history_table.horizontalHeader()
        _left_align_table_headers(self._history_table)
        self._history_table.configure_adaptive_columns(
            (154, 166, 204, 124, 146), (0.85, 1.0, 1.35, 0.75, 1.05),
            text_cap=420,
        )
        self._history_table.fit_rows_to_content(max_rows=8)
        history_layout.addWidget(self._history_table, stretch=3)
        self._history_detail = MonitorTextBrowser()
        self._history_detail.setText("选择一条事件查看原因、影响和系统动作。")
        history_layout.addWidget(self._history_detail, stretch=2)
        self._history_load_older = QPushButton("加载更早记录")
        self._history_load_older.clicked.connect(self._load_older_history)
        history_layout.addWidget(self._history_load_older)
        self._splitter.addWidget(history_panel)
        self._log_panel = LogPanel(self)
        self._log_panel.cleared.connect(self._reset_log_counts)
        self._splitter.addWidget(self._log_panel)
        self._splitter.setStretchFactor(0, 5)
        self._splitter.setStretchFactor(1, 4)
        self._log_panel.setVisible(False)
        self._splitter.setSizes([1180, 0])
        layout.addWidget(self._splitter, stretch=1)
        self._history_scope.currentIndexChanged.connect(self._reset_history_page)
        self._history_execution.currentIndexChanged.connect(self._reset_history_page)
        self._history_target.currentIndexChanged.connect(self._reset_history_page)
        self._history_search.textChanged.connect(self._schedule_history_search)
        self._update_log_toggle()
        return _scrollable_monitor_page(page, minimum_height=450)

    def _apply_theme(self, theme, *, persist=True):
        selected = str(theme or "dark").lower()
        if selected not in MONITOR_STYLE:
            selected = "dark"
        self._theme = selected
        families = set(QFontDatabase.families())
        body_family = next((name for name in ("PingFang SC", "Noto Sans CJK SC", "Source Han Sans SC") if name in families), self.font().family())
        self.setFont(QFont(body_family, BODY_FONT_PT))
        self.setStyleSheet(MONITOR_STYLE[selected])
        for combo in self.findChildren(MonitorComboBox):
            combo.apply_theme(selected)
        # These cards are dense, but every text readout must retain one full
        # baseline.  Qt's default Preferred vertical policy may shave one or
        # two pixels from nested labels after a hidden tab is restyled and
        # reflowed.  Derive the floor from the active font instead of pinning
        # card heights or shrinking type for a particular screen.
        for card in (self._overview_model_card, self._overview_spatial_card):
            for label in card.findChildren(QLabel):
                if label.text().strip():
                    label.ensurePolished()
                    line_height = label.fontMetrics().height()
                    if label.property("metric") or label.property("value"):
                        line_height = max(
                            line_height, label.minimumSizeHint().height()
                        )
                    label.setMinimumHeight(line_height)
        palette = PALETTES[selected]
        for label, name, size, tone in self._icon_labels:
            label.setPixmap(monitor_icon(name, palette[tone], size).pixmap(QSize(size, size)))
        for button, name, size, tone in self._icon_buttons:
            button.setIcon(monitor_icon(name, palette[tone], size))
        if hasattr(self, "_health_panel"):
            icon_name = str(self._health_panel.property("healthIconName") or "shield")
            icon_tone = str(self._health_panel.property("healthIconTone") or "success")
            self._health_icon.setPixmap(
                monitor_icon(
                    icon_name,
                    palette[icon_tone],
                    48,
                ).pixmap(QSize(48, 48))
            )
        for bar in self.findChildren(ProgressTrack):
            bar.setProperty("theme", selected)
            if bar.percentage_visible:
                bar.ensurePolished()
                bar.setMinimumHeight(bar.minimumSizeHint().height())
            bar.update()
        for table in (self._streams, self._tiles, self._overview_results, self._history_table, self._assembly_steps):
            table.setShowGrid(False)
            table.setAlternatingRowColors(False)
            _left_align_table_headers(table)
            table.ensurePolished()
            # Keep 4px rhythm, with real font metrics as the floor at any DPI.
            line_height = table.fontMetrics().height()
            header_height = max(TABLE_HEADER_MIN_HEIGHT, ((line_height + 15) // 4) * 4)
            row_height = max(TABLE_ROW_MIN_HEIGHT, ((line_height + 17) // 4) * 4)
            table.horizontalHeader().setFixedHeight(header_height)
            table.verticalHeader().setDefaultSectionSize(row_height)
            table.request_adaptive_layout()
        self._style_recent_events()
        for stream_id in self._stream_state:
            self._write_stream_row(stream_id)
        self._warning_log_button.setText(f"历史警告  {self._log_warning_count}")
        self._error_log_button.setText(f"历史错误  {self._log_error_count}")
        theme_action = "切换浅色主题" if selected == "dark" else "切换深蓝主题"
        self._theme_toggle.setIcon(monitor_icon(
            "sun" if selected == "dark" else "moon", palette["muted"], 22
        ))
        self._theme_toggle.setToolTip(theme_action)
        self._theme_toggle.setAccessibleName(theme_action)
        log_panel = getattr(self, "_log_panel", None)
        if log_panel is not None:
            log_panel.set_theme(selected)
        if persist:
            QgsSettings().setValue(MONITOR_THEME_SETTING, selected)

    def _apply_responsive_layout(self, width):
        """Reflow information panels instead of shrinking text below 1050 px."""

        compact = int(width) < 1050
        if compact == self._compact_layout:
            return
        self._compact_layout = compact

        cards = self._overview_cards
        cards.removeWidget(self._overview_model_card)
        cards.removeWidget(self._overview_spatial_card)
        if compact:
            cards.setColumnStretch(1, 0)
            cards.addWidget(self._overview_model_card, 0, 0)
            cards.addWidget(self._overview_spatial_card, 1, 0)
            self._overview_splitter.setOrientation(VERTICAL)
            self._overview_splitter.setSizes([760, 280])
        else:
            cards.setColumnStretch(1, 1)
            cards.addWidget(self._overview_model_card, 0, 0)
            cards.addWidget(self._overview_spatial_card, 0, 1)
            self._overview_splitter.setOrientation(HORIZONTAL)
            self._overview_splitter.setSizes([1060, 400])
        if compact:
            self._sync_overview_scroll_extent()
            # resizeEvent is delivered before the scroll viewport has its
            # final width.  Repeat once through Qt's queue so the vertical
            # splitter measures the reflowed cards, not the former columns.
            QTimer.singleShot(0, self._sync_overview_scroll_extent)
        else:
            for card in (self._overview_model_card, self._overview_spatial_card):
                card.setMinimumHeight(0)
            for panel in (self._overview_main, self._overview_activity_panel):
                panel.setMinimumHeight(0)
            self._overview_splitter.setMinimumHeight(0)
            self._overview_page.setMinimumHeight(0)
            self._overview_splitter.updateGeometry()
            self._overview_page.updateGeometry()

    def _sync_overview_scroll_extent(self):
        """Let the overview scroll before a reflowed splitter can compress cards.

        The overview's content height changes when its two execution cards move
        from columns to rows.  Derive the constraint from Qt's current layout
        hints instead of prescribing a fixed dashboard height, so a short
        window receives a vertical scrollbar while a wide desktop stays dense.
        """

        splitter = self._overview_splitter
        if not self._compact_layout or splitter.orientation() != VERTICAL:
            return
        page = self._overview_page
        splitter.setMinimumHeight(0)
        page.setMinimumHeight(0)

        # Reparenting the execution cards invalidates several nested layouts at
        # once.  On a hidden tab Qt only refreshes the outer page immediately;
        # querying the splitter at that point returns the former two-column
        # minimum.  Refresh every widget-owned child layout from the leaves
        # upward, including the stat-pair containers, before deriving the
        # scroll extent.
        for panel in (self._overview_main, self._overview_activity_panel):
            panel.setMinimumHeight(0)
            self._activate_layout_tree(panel)

        # QSplitter's minimum hint is cached and can still describe the former
        # orientation here.  The freshly activated child layouts are the
        # authoritative minimum for the vertical stack.
        child_heights = [
            splitter.widget(index).layout().minimumSize().height()
            for index in range(splitter.count())
        ]
        minimum_body_height = sum(child_heights) + splitter.handleWidth()
        splitter.setMinimumHeight(minimum_body_height)
        splitter.setSizes(child_heights)

        page_layout = page.layout()
        page_layout.invalidate()
        page_layout.activate()
        margins = page_layout.contentsMargins()
        page.setMinimumHeight(max(
            page_layout.minimumSize().height(),
            minimum_body_height + margins.top() + margins.bottom(),
        ))
        page.updateGeometry()

    @classmethod
    def _activate_layout_tree(cls, widget):
        """Refresh nested widget layouts bottom-up after a hidden-tab reflow."""

        layout = widget.layout()
        if layout is None:
            return

        def refresh_children(parent_layout):
            for index in range(parent_layout.count()):
                item = parent_layout.itemAt(index)
                child_widget = item.widget()
                child_layout = item.layout()
                if child_widget is not None:
                    cls._activate_layout_tree(child_widget)
                elif child_layout is not None:
                    refresh_children(child_layout)
                    child_layout.invalidate()

        refresh_children(layout)
        layout.invalidate()
        layout.activate()
        widget.updateGeometry()

    @staticmethod
    def _badge(label, text, tone):
        label.setText(text)
        if label.property("status") != tone:
            label.setProperty("status", tone)
            label.style().unpolish(label)
            label.style().polish(label)

    def _style_recent_events(self):
        events = getattr(self, "_recent_event_data", [])
        if not events:
            self._recent_events_label.setText("尚无已记录事件")
            return
        palette = PALETTES[self._theme]
        rows = []
        for event in events[:5]:
            timestamp = event.get("timestamp")
            stamp = format_monitor_timestamp(timestamp, compact=True)
            message = str(event.get("message") or event.get("event_type") or "—")
            rows.append(f'<tr><td width="18" valign="top" style="color:{palette["accent"]}">●</td>'
                        f'<td valign="top" width="124" style="color:{palette["muted"]}">{escape(stamp)}</td>'
                        f'<td style="color:{palette["text"]}">{escape(message)}</td></tr>')
        self._recent_events_label.setText('<table cellspacing="0" cellpadding="0" width="100%">' + '<tr><td colspan="3" height="16"></td></tr>'.join(rows) + '</table>')
        self._recent_events_label.setToolTip(monitor_timezone_label())

    def _show_run_information(self):
        box = QMessageBox(self)
        box.setWindowTitle("运行信息")
        box.setText("\n\n".join(label.text().replace(" | ", "\n") for label in (self._run_overview, self._assembly_overview, self._coverage_overview)))
        box.setInformativeText("创建年龄与本次观察时长不是实际执行耗时。恢复与重做失败包仍位于主界面。")
        box.exec()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_overview_cards"):
            self._apply_responsive_layout(event.size().width())

    def showEvent(self, event):
        super().showEvent(event)
        timer = getattr(self, "_poll_timer", None)
        if timer is not None:
            timer.setInterval(1000)
            if self._database_bound and not timer.isActive():
                timer.start()
                self._poll_database()

    def hideEvent(self, event):
        super().hideEvent(event)
        timer = getattr(self, "_poll_timer", None)
        if timer is not None and timer.isActive():
            timer.setInterval(5000)

    def _set_status_badge(self, status):
        self._status_badge.setProperty("status", str(status))
        self._status_badge.style().unpolish(self._status_badge)
        self._status_badge.style().polish(self._status_badge)

    def _toggle_theme(self):
        self._apply_theme("light" if self._theme == "dark" else "dark")

    def _on_page_changed(self, index):
        if int(index) == 1:
            self._render_selected_tiles()
        elif int(index) == 2:
            self._render_selected_assembly()
        elif int(index) == 3 and self._database_bound:
            self._request_history_page(append=False)

    def _open_detail_kind(self, kind):
        target = str(kind)
        self._pages.setCurrentIndex(1)
        for row in range(self._detail_navigation.count()):
            item = self._detail_navigation.item(row)
            if str(item.data(USER_ROLE) or "") == target:
                self._detail_navigation.setCurrentRow(row)
                return

    def _on_detail_navigation_changed(self, row):
        item = self._detail_navigation.item(int(row))
        if item is None:
            return
        kind = str(item.data(USER_ROLE) or "package")
        index = self._detail_kind.findData(kind)
        if index >= 0 and index != self._detail_kind.currentIndex():
            self._detail_kind.setCurrentIndex(index)
        else:
            self._reset_detail_page()

    def _sync_overview_stream_selection(self):
        row = self._overview_results.currentRow()
        item = self._overview_results.item(row, 0) if row >= 0 else None
        stream_id = str(item.data(USER_ROLE) or "") if item is not None else ""
        target = self._stream_rows.get(stream_id)
        if target is not None and self._streams.currentRow() != target:
            self._streams.selectRow(target)

    def _on_stream_selection_changed(self):
        stream_id = self._selected_stream()
        overview_row = self._stream_rows.get(stream_id)
        if overview_row is not None and self._overview_results.currentRow() != overview_row:
            self._overview_results.selectRow(overview_row)
        self._render_selected_assembly()
        if self._pages.currentIndex() == 1:
            self._render_selected_tiles()

    def _render_object_detail(self):
        row = self._tiles.currentRow()
        if row < 0:
            return
        values = [
            self._tiles.item(row, column).text()
            if self._tiles.item(row, column) is not None else ""
            for column in range(self._tiles.columnCount())
        ]
        object_id = values[0] if values else ""
        raw = dict(self._selected_object.get(object_id) or {})
        identity = (object_id, str(raw.get("stream_id") or ""), raw.get("job_id"))
        previous_identity = (self._selection.get("object_id"), self._selection.get("stream_id"), self._selection.get("job_id"))
        if identity != previous_identity:
            self._selected_attempt = ""
            self._object_history_rows = []
            self._object_history_cursor = None
            self._models_for_attempt = []
        self._selection = {"object_id": object_id, "stream_id": str(raw.get("stream_id") or ""),
                           "job_id": raw.get("job_id"), "span_id": str(raw.get("span_id") or ""),
                           "execution_id": str(raw.get("execution_id") or ""),
                           "kind": str(self._detail_kind.currentData() or "package")}
        execution = str(raw.get("execution_id") or "—")
        span = str(raw.get("span_id") or "—")
        current = int(raw.get("progress_current") or 0)
        total = int(raw.get("progress_total") or 0)
        progress = f"{current:,}/{total:,}" if total else "—"
        self._object_current.setText(
            f"对象：{object_id}\n类型：{values[1] if len(values) > 1 else '—'}\n"
            f"执行状态：{values[2] if len(values) > 2 else '—'}\n"
            f"产物状态：{values[3] if len(values) > 3 else '—'}\n"
            f"当前进度：{progress}\n原因：{values[4] if len(values) > 4 else '—'}"
        )
        if identity != previous_identity:
            self._object_models.setPlainText("正在读取本包各模型的尝试记录…" if self._selection["kind"] == "package" else "此分类为空间任务；模型处理详情位于对应推理包。")
        self._object_attempts.setText(
            f"执行编号：{execution}\n尝试片段：{span}\n"
            f"重试预算计数：{raw.get('budget_attempt') if raw.get('budget_attempt') is not None else '—'}\n"
            "历史尝试编号与重试预算相互独立。"
        )
        self._object_events.setText(
            "正在读取关联事件…\n"
            f"对象 ID：{object_id}"
        )
        self._request_object_history(object_id)

    def _request_object_history(self, object_id, *, append=False):
        if not self._database_bound or not self._run_id or not str(object_id):
            return
        self._query_serial += 1
        self._latest_object_request_id = self._query_serial
        self._pending_object_query = {
            "kind": "object_history",
            "generation": self._query_generation,
            "request_id": self._latest_object_request_id,
            "run_id": self._run_id,
            "run_spec": dict(self._run_spec),
            "object_id": str(object_id),
            "detail_kind": str(self._detail_kind.currentData() or ""),
            "stream_id": str(self._selection.get("stream_id") or ""),
            "job_id": self._selection.get("job_id"),
            "span_id": self._selected_attempt,
            "append": bool(append),
            "before_started_at": self._object_history_cursor[0] if append and self._object_history_cursor else "",
            "before_span_id": self._object_history_cursor[1] if append and self._object_history_cursor else "",
        }
        self._dispatch_next_query()

    def _object_request_matches_controls(self, request):
        row = self._tiles.currentRow()
        item = self._tiles.item(row, 0) if row >= 0 else None
        return (
            item is not None
            and str(request.get("object_id") or "") == item.text()
            and str(request.get("detail_kind") or "")
            == str(self._detail_kind.currentData() or "")
            and str(request.get("stream_id") or "") == str(self._selection.get("stream_id") or "")
            and request.get("job_id") == self._selection.get("job_id")
            and str(request.get("span_id") or "") == self._selected_attempt
        )

    def _apply_object_history_result(self, payload):
        object_id = str(payload.get("object_id") or "")
        spans = [dict(row) for row in payload.get("spans") or ()]
        events = [dict(row) for row in payload.get("events") or ()]
        self._object_history_rows = self._object_history_rows + spans if payload.get("append") else spans
        self._object_history_cursor = (spans[-1]["started_at"], spans[-1]["span_id"]) if spans else None
        self._object_more.setEnabled(bool(payload.get("has_more")))
        self._models_for_attempt = [dict(row) for row in payload.get("models") or ()]
        if self._selection.get("kind") == "package":
            lines = ["<b>模型处理记录</b> · 尝试 " + escape(str(payload.get("attempt_id") or "—"))]
            models = {str(model.get("model_id") or ""): model for model in self._models_for_attempt}
            for configured in self._run_spec.get("models") or ():
                model_id = str(configured.get("model_id") or "")
                model = models.get(model_id, {})
                metadata = dict(model.get("metadata") or {})
                status = SPAN_STATUS_LABELS.get(str(model.get("status") or ""), "尚无本次记录")
                lines.append("<p><b>" + escape(str(configured.get("display_name") or model_id)) + "</b> · " + escape(status)
                             + "<br>开始：" + escape(format_monitor_timestamp(model.get("started_at")))
                             + "<br>结束：" + escape(format_monitor_timestamp(model.get("ended_at")))
                             + "<br>" + escape(str(model.get("message") or ""))
                             + ("<br>" + escape(json.dumps(metadata, ensure_ascii=False)) if metadata else "") + "</p>")
            if not models:
                lines.append("<p>尚无与该次包尝试关联的模型记录；升级前记录不会反推补齐。</p>")
            self._object_models.setHtml("".join(lines))
        spans = self._object_history_rows
        if spans:
            lines = []
            for span in spans:
                status = SPAN_STATUS_LABELS.get(
                    str(span.get("status") or ""),
                    str(span.get("status") or "未知"),
                )
                lines.append(
                    f"第 {int(span.get('attempt_no') or 0)} 次 · {status} · "
                    f"{format_monitor_timestamp(span.get('started_at'))}\n"
                    f"  执行 {span.get('execution_id') or '—'} · "
                    f"片段 {span.get('span_id') or '—'}\n"
                    f"  {span.get('message') or '无补充说明'}"
                )
            self._object_attempts.setHtml("<p>选择尝试可查看对应模型记录与事件；不会改变当前运行。</p>" + "".join(
                '<p><a href="attempt:' + escape(str(span["span_id"])) + '">'
                + escape(line).replace("\n", "<br>") + "</a></p>"
                for span, line in zip(spans, lines)))
        else:
            self._object_attempts.setText(
                f"对象：{object_id}\n没有升级后的独立尝试记录；旧 Run 可能记录不完整。"
            )
        if events:
            self._object_events.setText(
                "\n".join(
                    f"{format_monitor_timestamp(event.get('timestamp'))} · "
                    f"{event.get('message') or event.get('event_type') or '—'}"
                    + (" · 已恢复" if event.get("recovered_by_span_id") else "")
                    for event in events
                )
            )
        else:
            self._object_events.setText(
                f"对象：{object_id}\n没有升级后的关联事件。"
            )

    def _select_history_attempt(self, url):
        identifier = url.toString().removeprefix("attempt:")
        if not any(str(row.get("span_id")) == identifier for row in self._object_history_rows):
            return
        self._selected_attempt = identifier
        self._request_object_history(self._selection.get("object_id", ""))
        self._object_tabs.setCurrentWidget(self._object_models)

    def _open_object_events(self):
        self._history_target.setCurrentIndex(3 if self._selected_attempt else 2)
        self._pages.setCurrentIndex(3)
        self._reset_history_page()

    def _history_filter_context(self):
        target = str(self._history_target.currentData() or "all")
        if target == "stream":
            return {"stream_id": self._selected_stream() or "__no_selected_stream__"}
        if target == "attempt":
            return {"span_id": self._selected_attempt or self._selection.get("span_id") or "__no_selected_attempt__"}
        if target == "object":
            return {"object_id": self._selection.get("object_id") or "__no_selected_object__",
                    "stream_id": self._selection.get("stream_id", ""), "job_id": self._selection.get("job_id")}
        return {}

    def _render_selected_assembly(self):
        if not hasattr(self, "_assembly_steps"):
            return
        stream_id = self._selected_stream()
        progress = dict(self._runtime_progress.get(stream_id) or {})
        phase = str(progress.get("phase") or "")
        status = str(progress.get("status") or "")
        phase_statuses = dict(
            getattr(self, "_assembly_phase_statuses", {}).get(stream_id) or {}
        )
        for row, (phase_key, _name, unit) in enumerate(ASSEMBLY_PHASES):
            recorded = dict(phase_statuses.get(phase_key) or {})
            recorded_status = str(recorded.get("status") or "")
            label = SPAN_STATUS_LABELS.get(recorded_status, "")
            if not label:
                if phase_key == phase and status:
                    label = SPAN_STATUS_LABELS.get(status, status)
                else:
                    label = "记录缺失" if progress else "未开始"
            current = recorded.get("current")
            total = recorded.get("total")
            if phase_key == phase:
                current = progress.get("progress_current")
                total = progress.get("progress_total")
            progress_text = (
                f"{int(current or 0):,}/{int(total):,} {unit}"
                if total not in (None, "", 0, "0") else f"— / {unit}"
            )
            self._assembly_steps.item(row, 1).setText(label)
            self._assembly_steps.item(row, 2).setText(progress_text)
        self._assembly_steps.request_adaptive_layout()
        if not stream_id:
            self._assembly_detail.setText("选择一个结果流查看十步组装记录。")
            return
        phase_name = str(progress.get("phase_name") or "尚未开始")
        started = format_monitor_timestamp(progress.get("phase_started_at"))
        message = str(progress.get("message") or "—")
        feature_count = progress.get("feature_count")
        feature_text = f"{int(feature_count):,}" if feature_count is not None else "—"
        self._assembly_detail.setText(
            f"结果流：{self._stream_display_name(stream_id)}\n"
            f"当前步骤：{phase_name}\n步骤开始：{started}\n"
            f"当前写入面数：{feature_text}（非最终产物统计）\n说明：{message}"
        )
        coverage = dict(self._coverage_state.get(stream_id) or {})
        if coverage:
            self._result_coverage.setText(
                f"覆盖验收：{coverage.get('status') or '—'}\n"
                f"空白面积：{coverage.get('gap_area_m2', '—')} m²  "
                f"重叠面积：{coverage.get('overlap_area_m2', '—')} m²  "
                f"范围外面积：{coverage.get('outside_area_m2', '—')} m²\n"
                "单流验收、整个 Run 完成与人工修整分别表达。"
            )
        else:
            self._result_coverage.setText(
                "覆盖验收：尚未执行\n空白面积：—  重叠面积：—  范围外面积：—\n"
                "本页不代表人工修整或 accepted_labels 已完成。"
            )

    def _schedule_history_search(self, *_args):
        self._history_search_timer.start()

    def _reset_history_page(self, *_args):
        self._history_rows = []
        self._history_cursor = None
        self._history_exhausted = False
        self._history_table.setRowCount(0)
        if self._database_bound and self._run_id:
            self._request_history_page(append=False)

    def _load_older_history(self):
        if not self._history_exhausted:
            self._request_history_page(append=True)

    def _request_history_page(self, *, append):
        if not self._database_bound or not self._run_id:
            return
        self._query_serial += 1
        self._latest_history_request_id = self._query_serial
        self._pending_history_query = {
            "kind": "history",
            "generation": self._query_generation,
            "request_id": self._latest_history_request_id,
            "run_id": self._run_id,
            "run_spec": dict(self._run_spec),
            "scope": str(self._history_scope.currentData() or "all"),
            "execution_id": str(self._history_execution.currentData() or ""),
            "search": self._history_search.text().strip(),
            "before_event_id": self._history_cursor if append else None,
            "append": bool(append),
            "page_size": MONITOR_EVENT_PAGE_SIZE,
            "context": self._history_filter_context(),
            **self._history_filter_context(),
        }
        self._history_context_label.setText("范围：" + self._history_target.currentText() + "  " +
                                           " · ".join(str(value) for value in self._history_filter_context().values() if value is not None) + " · " + monitor_timezone_label())
        self._dispatch_next_query()

    def _apply_history_result(self, payload):
        rows = [dict(row) for row in payload.get("rows") or ()]
        if payload.get("append"):
            self._history_rows.extend(rows)
        else:
            self._history_rows = rows
        self._history_cursor = (
            int(self._history_rows[-1]["monitor_event_id"])
            if self._history_rows else None
        )
        self._history_exhausted = len(rows) < int(
            payload.get("page_size") or MONITOR_EVENT_PAGE_SIZE
        )
        self._history_load_older.setEnabled(not self._history_exhausted)
        self._history_load_older.setText(
            "没有更早记录" if self._history_exhausted else "加载更早记录"
        )
        self._history_table.setUpdatesEnabled(False)
        try:
            self._history_table.setRowCount(len(self._history_rows))
            for row_index, event in enumerate(self._history_rows):
                execution = str(event.get("execution_id") or "—")
                values = (
                    format_monitor_timestamp(event.get("timestamp"), compact=True),
                    str(event.get("object_id") or event.get("object_type") or "Run"),
                    str(event.get("message") or event.get("event_type") or "—"),
                    execution[:12] + ("…" if len(execution) > 12 else ""),
                    "已恢复" if event.get("recovered_by_span_id") else str(event.get("level") or "信息"),
                )
                for column, value in enumerate(values):
                    item = self._history_table.item(row_index, column)
                    if item is None:
                        item = QTableWidgetItem()
                        self._history_table.setItem(row_index, column, item)
                    item.setText(value)
                    if column == 0:
                        item.setData(USER_ROLE, int(event["monitor_event_id"]))
                        item.setToolTip(format_monitor_timestamp(event.get("timestamp")) + "\n原始时间：" + str(event.get("timestamp") or "—"))
                    elif column == 3:
                        item.setToolTip(execution)
        finally:
            self._history_table.setUpdatesEnabled(True)
        self._history_table.request_adaptive_layout()

    def _render_history_detail(self):
        row = self._history_table.currentRow()
        if row < 0 or row >= len(self._history_rows):
            return
        event = self._history_rows[row]
        payload = json.dumps(
            event.get("payload") or {}, ensure_ascii=False, indent=2
        )
        current = "已恢复" if event.get("recovered_by_span_id") else "未记录后续恢复"
        self._history_detail.setText(
            f"事件：{event.get('event_type') or '—'}\n"
            f"时间：{format_monitor_timestamp(event.get('timestamp'))}\n"
            f"对象：{event.get('object_type') or '—'} / {event.get('object_id') or '—'}\n"
            f"执行编号：{event.get('execution_id') or '—'}\n"
            f"尝试片段：{event.get('span_id') or '—'}\n"
            f"当前关联状态：{current}\n"
            f"说明：{event.get('message') or '—'}\n\n技术详情：\n"
            f"原始时间：{event.get('timestamp') or '—'}\n{payload}"
        )

    @staticmethod
    def _set_progress_bar(bar, completed, total):
        total_value = max(0.0, float(total or 0))
        completed_value = max(0.0, float(completed or 0))
        if total_value > 0:
            bar.setProperty("progressKnown", True)
            bar.setRange(0, ASSEMBLY_PROGRESS_SCALE)
            bar.setValue(
                round(
                    min(1.0, completed_value / total_value)
                    * ASSEMBLY_PROGRESS_SCALE
                )
            )
        else:
            bar.setProperty("progressKnown", False)
            bar.setRange(0, 1)
            bar.setValue(0)

    def _update_task_lane(self, label, bar, title, counts, *, enabled=True):
        values = dict(counts or {})
        total = sum(int(value) for value in values.values())
        ready = int(values.get("ready", 0))
        running = int(values.get("running", 0))
        waiting = _waiting_count(values)
        failed = int(values.get("failed", 0))
        if not enabled:
            label.setText(f"{title}：未启用")
            self._set_progress_bar(bar, 0, 0)
            return
        if total < 1:
            label.setText(f"{title}：等待计划")
            self._set_progress_bar(bar, 0, 0)
            return
        label.setText(
            f"{title}  {ready:,} / {total:,} 项"
        )
        label.setToolTip(f"{title}：完成 {ready}/{total}；运行 {running}；等待 {waiting}；失败 {failed}。按任务计数。")
        self._set_progress_bar(bar, ready, total)

    def _apply_monitor_history(self, history):
        value = dict(history or {})
        self._latest_execution = dict(value.get("latest_execution") or {})
        if value.get("archived"):
            summary = dict(value.get("summary") or {})
            self._history_health.setText(
                "详细历史已归档，仅保留摘要："
                f"执行 {int(summary.get('execution_count') or 0)} 次，"
                f"尝试 {int(summary.get('attempt_count') or 0)} 次，"
                f"失败 {int(summary.get('failed_count') or 0)} 次"
            )
            self._history_completeness.setText("历史：已归档，仅保留摘要")
            return
        if not value.get("available"):
            reason = str(value.get("reason") or "")
            text = (
                "升级前 Run：关键执行历史不完整"
                if reason == "upgrade_precedes_history"
                else "历史记录读取失败或尚未建立"
            )
            self._history_health.setText(text)
            self._history_completeness.setText("历史：记录不完整")
        else:
            latest = dict(value.get("latest_execution") or {})
            counts = dict(value.get("span_status_counts") or {})
            self._history_health.setText(
                f"本次{execution_trigger_label(latest.get('trigger_type') or '')}执行 · "
                f"{SPAN_STATUS_LABELS.get(str(latest.get('status') or ''), '未知')}\n"
                f"历史记录：完成 {int(counts.get('completed') or 0)} · "
                f"失败 {int(counts.get('failed') or 0)} · "
                f"中断 {int(counts.get('interrupted') or 0)}"
            )
            complete = bool(latest.get("recording_complete", True))
            self._history_completeness.setText(
                "历史：关键过程已记录" if complete else "历史：记录不完整"
            )
        events = list(value.get("recent_events") or ())[:5]
        self._recent_event_data = events
        self._style_recent_events()

        executions = list(value.get("executions") or ())
        selected = str(self._history_execution.currentData() or "")
        desired = ["", *[str(item.get("execution_id") or "") for item in executions]]
        current = [
            str(self._history_execution.itemData(index) or "")
            for index in range(self._history_execution.count())
        ]
        if current != desired:
            blocked = self._history_execution.blockSignals(True)
            try:
                self._history_execution.clear()
                self._history_execution.addItem("全部执行", "")
                for execution in executions:
                    identifier = str(execution.get("execution_id") or "")
                    started = str(execution.get("started_at") or "")
                    label = (
                        f"{execution_trigger_label(execution.get('trigger_type') or '')} · "
                        f"{started[0:19]} · {identifier[:8]}"
                    )
                    self._history_execution.addItem(label, identifier)
                index = self._history_execution.findData(selected)
                self._history_execution.setCurrentIndex(max(0, index))
            finally:
                self._history_execution.blockSignals(blocked)

    def attach_runner(self, runner: QObject):
        self.detach()
        pairs = [
            (runner.step_started, self._on_step_started),
            (runner.step_finished, self._on_step_finished),
            (runner.pipeline_finished, self._on_finished),
        ]
        log_batch = getattr(runner, "log_batch", None)
        if log_batch is not None:
            pairs.insert(0, (log_batch, self._on_log_batch))
        else:
            pairs.insert(0, (runner.log_line, self._on_log))
            process_log = getattr(runner, "process_log", None)
            if process_log is not None:
                pairs.insert(0, (process_log, self._on_process_log))
        progress_batch = getattr(runner, "stream_progress_batch", None)
        if progress_batch is not None:
            pairs.append((progress_batch, self._on_stream_progress_batch))
        else:
            pairs.append((runner.stream_progress, self._on_stream_progress))
        for signal, slot in pairs:
            signal.connect(slot)
            self._connected.append((signal, slot))
        self._stop.setEnabled(True)

    def bind_state_database(
        self, database_path, run_id, *, page_size=500, run_spec=None
    ):
        del database_path
        if not run_spec:
            raise ValueError("monitor requires the frozen PostgreSQL Run Spec")
        self._database_bound = True
        self._terminal_run_status = ""
        self._run_id = str(run_id)
        self._run_spec = dict(run_spec or {})
        self._page_size = max(1, min(int(page_size), 500))
        self._page = 0
        self._last_detail_requested_at = 0.0
        self._query_generation += 1
        self._latest_snapshot_request_id = 0
        self._latest_detail_request_id = 0
        self._latest_history_request_id = 0
        self._latest_object_request_id = 0
        self._pending_snapshot_query = None
        self._pending_detail_query = None
        self._pending_history_query = None
        self._pending_object_query = None
        self._run_created_epoch = None
        self._monitor_started_at = time.monotonic()
        self._stage_key = ""
        self._stage_started_at = time.monotonic()
        self._poll_timer.start()
        self._poll_database()

    def unbind_state_database(self):
        if hasattr(self, "_poll_timer"):
            self._poll_timer.stop()
        self._database_bound = False
        self._run_id = ""
        self._run_spec = {}
        self._query_generation += 1
        self._latest_snapshot_request_id = 0
        self._latest_detail_request_id = 0
        self._latest_history_request_id = 0
        self._latest_object_request_id = 0
        self._pending_snapshot_query = None
        self._pending_detail_query = None
        self._pending_history_query = None
        self._pending_object_query = None
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
        if shown and hasattr(self, "_pages"):
            self._pages.setCurrentIndex(3)
        self._log_panel.setVisible(shown)
        if shown:
            self._log_panel.set_visible_severities({"info", "warning", "error"})
            self._log_panel.refresh_visible()
        self._splitter.setSizes([720, 460] if shown else [1180, 0])
        self._update_log_toggle()

    def _show_log_severity(self, severity):
        self._pages.setCurrentIndex(3)
        if not self._log_toggle.isChecked():
            self._log_toggle.setChecked(True)
        self._log_panel.set_visible_severities({str(severity)})
        self._log_panel.scroll_to_latest()

    def _reset_log_counts(self):
        self._log_error_count = 0
        self._log_warning_count = 0
        self._update_log_toggle()

    def _update_log_toggle(self):
        action = "收起日志" if self._log_toggle.isChecked() else "显示日志"
        self._log_toggle.setText(action)
        self._warning_log_button.setText(f"历史警告  {self._log_warning_count}")
        self._error_log_button.setText(f"历史错误  {self._log_error_count}")
        self._warning_log_button.setEnabled(self._log_warning_count > 0)
        self._error_log_button.setEnabled(self._log_error_count > 0)

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
        self._terminal_run_status = ""
        del tiles
        self.unbind_state_database()
        self.clear_log()
        self._streams.setRowCount(0)
        self._overview_results.setRowCount(0)
        self._tiles.setRowCount(0)
        self._stream_rows.clear()
        self._stream_state.clear()
        self._tile_state.clear()
        self._tile_rows.clear()
        self._step_started_at.clear()
        self._step_attempts.clear()
        self._active_stream_stages.clear()
        self._active_global_stage = ""
        self._active_inference_stream = ""
        self._package_activity.clear()
        self._runner_message = ""
        self._runtime_progress.clear()
        self._coverage_state.clear()
        self._assembly_phase_statuses.clear()
        self._control_state = ""
        self._history_rows = []
        self._history_cursor = None
        self._history_table.setRowCount(0)
        self._logged_error_texts.clear()
        self._process_log_suppressions.clear()
        self._detail_signature = None
        self._stage_key = ""
        self._stage_started_at = time.monotonic()
        self._phase.setText("准备中")
        self._status_badge.setText("准备中")
        self._set_status_badge("neutral")
        self._run_id_label.setText("Run：准备创建")
        self._monitor_sync.setText("监控：等待连接")
        self._run_overview.setText("Run：准备中")
        self._package_overview.setText("Work Package：等待计划")
        self._unit_overview.setText("空间单元拟合：等待计划")
        self._assembly_overview.setText("结果流组装：等待上游计算")
        self._coverage_overview.setText("空白/重叠验收：等待组装")
        self._update_stage_rail("compute")
        self._tile_detail_title.setText("选中结果流：未选择 | 空间单元详情")
        self._summary.setText("结果流: 0  |  完成: 0  |  运行: 0  |  等待: 0  |  停止: 0  |  失败: 0")
        self._overall_bar.setRange(0, ASSEMBLY_PROGRESS_SCALE)
        self._overall_bar.setValue(0)
        self._overall_bar.setFormat("整体任务完成度：等待任务图")
        self._bar.setRange(0, 0)
        self._bar.setFormat("准备中")
        self._stop.setEnabled(True)
        self._stop.setText("停止任务")
        self.setWindowTitle("推理监控 - 准备中")

    def set_stage_progress(self, info):
        name = str(info.get("name") or "处理中")
        stream_id = str(info.get("stream_id") or "")
        current = int(info.get("current") or 0)
        total = int(info.get("total") or 0)
        message = str(info.get("message") or "")
        if self._database_bound and self._run_id:
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

    def mark_stopping(self, text="正在停止当前子进程组"):
        self._control_state = "stopping"
        self._stop.setEnabled(False)
        self._stop.setText("正在停止…")
        self._phase.setText(str(text))
        self._status_badge.setText("正在停止")
        self._set_status_badge("warning")

    def mark_finished(self, text="已完成", detail=""):
        self._control_state = ""
        self._terminal_run_status = {"已完成": "ready", "失败": "failed", "已停止": "stopped"}.get(text, "")
        # Fence callbacks dispatched before this authoritative runner result.
        self._query_generation += 1
        self._pending_snapshot_query = None
        self._pending_detail_query = None
        self._pending_history_query = None
        self._pending_object_query = None
        self._stop.setEnabled(False)
        self._stop.setText("停止任务")
        detail_text = re.sub(r"\s+", " ", str(detail or "")).strip()
        self._phase.setText(
            f"{text}：{detail_text}" if detail_text else text
        )
        self._phase.setToolTip(str(detail or ""))
        self._status_badge.setText(str(text))
        self._set_status_badge(
            "active" if text == "已完成" else "failed" if text == "失败" else "neutral"
        )
        if text == "已完成":
            self._overall_bar.setRange(0, ASSEMBLY_PROGRESS_SCALE)
            self._overall_bar.setValue(ASSEMBLY_PROGRESS_SCALE)
            self._overall_bar.setFormat("整体任务完成度：100%（已完成）")
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
        self._overview_results.insertRow(row)
        self._stream_rows[stream_id] = row
        self._stream_state[stream_id] = {
            "stage": "等待计划",
            "progress": "-",
            "unit_progress": "-",
            "stage_progress": "-",
            "activity": "0/0",
            "feature_count": None,
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
        state = self._stream_state[stream_id]
        changed = {
            key: value
            for key, value in changes.items()
            if state.get(key) != value
        }
        if not changed:
            return
        state.update(changed)
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
        feature_count = state.get("feature_count")
        values = [
            self._stream_display_name(stream_id),
            state["stage"],
            current_progress,
            state.get("activity") or "0/0",
            f"{int(feature_count):,}" if feature_count is not None else "—",
            str(state["failures"]),
            state["elapsed"],
        ]
        for column, value in enumerate(values):
            text = str(value)
            item = self._streams.item(row, column)
            if item is None:
                item = QTableWidgetItem()
                self._streams.setItem(row, column, item)
            if item.text() != text:
                item.setText(text)
            if column == 0:
                item.setToolTip(stream_id)
                item.setData(USER_ROLE, stream_id)
            if column == 1:
                item.setForeground(
                    QColor(status_color(self._theme, str(state["status"])))
                )
        assembly_state = "尚未开始"
        runtime = dict(self._runtime_progress.get(stream_id) or {})
        if runtime:
            runtime_status = str(runtime.get("status") or "")
            assembly_state = SPAN_STATUS_LABELS.get(
                runtime_status, str(runtime.get("phase_name") or "运行中")
            )
        elif state.get("status") == "成功":
            assembly_state = "已完成"
        coverage = dict(self._coverage_state.get(stream_id) or {})
        coverage_state = (
            str(coverage.get("status") or "尚未执行") if coverage else "尚未执行"
        )
        overview_values = (
            self._stream_display_name(stream_id),
            str(state.get("stage") or "等待计划").replace("Core 拟合", "内部区域边界拟合").replace("Seam 拟合", "接缝边界拟合").replace("Junction 拟合", "交汇区域边界拟合").replace("Work Package", "推理包"),
            assembly_state,
            {"passed": "验收通过", "failed": "验收失败", "skipped": "未执行"}.get(coverage_state, coverage_state),
        )
        for column, value in enumerate(overview_values):
            item = self._overview_results.item(row, column)
            if item is None:
                item = QTableWidgetItem()
                self._overview_results.setItem(row, column, item)
            if item.text() != str(value):
                item.setText(str(value))
            item.setTextAlignment(ALIGN_LEFT | ALIGN_VCENTER)
            if column == 0:
                item.setToolTip(stream_id)
                item.setData(USER_ROLE, stream_id)
                item.setIcon(monitor_icon("layers" if stream_id.startswith("fusion:") else "cube", PALETTES[self._theme]["text"], OVERVIEW_ICON_SIZE))
            else:
                running = column == 1 and state.get("status") == "运行中"
                tone = status_color(self._theme, str(state.get("status"))) if column == 1 else PALETTES[self._theme]["muted"]
                item.setForeground(QColor(tone))
                item.setIcon(monitor_icon("ring" if running else "clock" if str(value).startswith(("尚未", "等待")) else "check" if str(value) in {"已完成", "验收通过"} else "activity", tone, OVERVIEW_ICON_SIZE))
        self._overview_results.setIconSize(QSize(OVERVIEW_ICON_SIZE, OVERVIEW_ICON_SIZE))
        self._streams.request_adaptive_layout()
        self._overview_results.request_adaptive_layout()

    def _on_process_log(self, event):
        info = dict(event or {})
        level = str(info.get("source") or "system")
        message = str(info.get("message") or "")
        suppression_key = (level, message)
        self._process_log_suppressions[suppression_key] = (
            int(self._process_log_suppressions.get(suppression_key) or 0) + 1
        )
        self._on_log(level, message, log_context=info)

    def _on_log_batch(self, events):
        self._log_panel.begin_batch()
        try:
            for event in events or ():
                info = dict(event or {})
                self._on_log(
                    str(info.get("source") or "system"),
                    str(info.get("message") or ""),
                    log_context=info,
                )
        finally:
            self._log_panel.end_batch()

    def _on_log(self, level, message, log_context=None):
        if log_context is None:
            suppression_key = (str(level), str(message))
            suppression_count = int(
                self._process_log_suppressions.get(suppression_key) or 0
            )
            if suppression_count:
                if suppression_count == 1:
                    self._process_log_suppressions.pop(suppression_key, None)
                else:
                    self._process_log_suppressions[suppression_key] = (
                        suppression_count - 1
                    )
                return
        presentation = _log_presentation(level, message)
        context = dict(log_context or {})
        affected = str(
            context.get("step")
            or context.get("unit_id")
            or context.get("stream_id")
            or presentation.get("affected")
            or ""
        )
        presentation["affected"] = affected
        attempt = int(context.get("attempt") or presentation.get("attempt") or 0)
        if not attempt and affected:
            attempt = int(self._step_attempts.get(affected) or 0)
        context_key = (
            f"{affected}:attempt={attempt}"
            if affected
            else "unscoped"
        )
        presentation["fingerprint"] = _log_fingerprint(
            presentation["severity"],
            presentation["error"],
            affected,
            attempt,
        )
        if presentation["severity"] == "error":
            self._logged_error_texts.add(
                re.sub(r"\s+", " ", str(presentation["error"])).strip().lower()
            )
        created = self._log_panel.append_event(
            message,
            source=presentation["source"],
            severity=presentation["severity"],
            title=presentation["title"],
            affected=presentation["affected"],
            system_action=presentation["system_action"],
            user_action=presentation["user_action"],
            fingerprint=presentation["fingerprint"],
            context_key=context_key,
            event_timestamp=context.get("timestamp"),
        )
        if not created:
            return
        if presentation["severity"] == "warning":
            self._log_warning_count += 1
        elif presentation["severity"] == "error":
            self._log_error_count += 1
        if presentation["severity"] in {"warning", "error"}:
            self._update_log_toggle()

    def _on_step_started(self, name):
        stream_id = _stream_from_step(name)
        stage = _stage_from_step(name)
        self._step_started_at[name] = time.time()
        self._step_attempts[name] = int(self._step_attempts.get(name) or 0) + 1
        self._active_global_stage = stage
        if stream_id:
            stage_counts = self._active_stream_stages.setdefault(stream_id, {})
            stage_counts[stage] = int(stage_counts.get(stage, 0)) + 1
            self._set_stream(stream_id, stage=stage, status="运行中")

    def _on_step_finished(self, name, return_code, result):
        failed = not result.get("success") and not result.get("skipped")
        if failed:
            self._on_log(
                "system",
                json.dumps(
                    {
                        "event": "monitor_step_failed",
                        "step": str(name),
                        "stream_id": str(
                            result.get("stream_id") or _stream_from_step(name)
                        ),
                        "attempt": int(self._step_attempts.get(name) or 1),
                        "return_code": int(return_code),
                        "error": str(result.get("error") or "任务执行失败"),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
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
        if not self._database_bound:
            changes["status"] = status
        elif status == "失败":
            changes.update({"status": "失败", "stage": "任务失败"})
        if status == "失败":
            changes["failures"] = int(self._stream_state.get(stream_id, {}).get("failures", 0)) + 1
        self._set_stream(stream_id, **changes)

    def _on_stream_progress_batch(self, events):
        for info in events or ():
            self._on_stream_progress(dict(info or {}))

    def _on_stream_progress(self, info):
        event = str(info.get("event") or "")
        if event.startswith(("package_", "work_package_", "accelerator_worker_")):
            self._update_package_activity(info)
            if self._database_bound:
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
        self._package_activity["_event_observed_at"] = time.time()
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
        if not hasattr(self, "_streams"):
            return ""
        row = self._streams.currentRow()
        if row < 0:
            return ""
        item = self._streams.item(row, 0)
        if item is None:
            return ""
        return str(item.toolTip() or item.text())

    def _render_selected_tiles(self):
        stream_id = self._selected_stream()
        if self._database_bound and self._run_id:
            self._render_database_page(stream_id)
            return
        values = self._tile_state.get(stream_id, {})
        self._tiles.setHorizontalHeaderLabels(
            ["对象ID", "类型", "执行状态", "当前进度", "原因"]
        )
        self._tile_rows.clear()
        self._tiles.setRowCount(len(values))
        for row, (tile_id, state) in enumerate(
            sorted(values.items(), key=lambda item: _tile_sort_key(item[0]))
        ):
            self._tile_rows[tile_id] = row
            for column, value in enumerate(
                (tile_id, "Tile", state["status"], state["progress"], state["error"])
            ):
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

    def _schedule_detail_search(self, *_args):
        self._detail_search_timer.start()

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
        detail_kind = str(self._detail_kind.currentData() or "package")
        if not stream_id and detail_kind not in {"package", "tile"}:
            self._detail_signature = None
            self._tile_rows.clear()
            self._tiles.setRowCount(0)
            self._tile_detail_title.setText(
                "选中结果流：未选择 | 空间单元详情"
            )
            return
        now = time.monotonic()
        if (
            self._detail_signature is not None
            and now - self._last_detail_requested_at < 2.0
        ):
            return
        self._last_detail_requested_at = now
        self._query_serial += 1
        self._latest_detail_request_id = self._query_serial
        self._pending_detail_query = {
            "kind": "detail",
            "generation": self._query_generation,
            "request_id": self._latest_detail_request_id,
            "run_id": self._run_id,
            "run_spec": dict(self._run_spec),
            "stream_id": str(stream_id),
            "detail_kind": detail_kind,
            "status": str(self._detail_status.currentData() or ""),
            "search": self._detail_search.text().strip(),
            "page": self._page,
            "page_size": self._page_size,
        }
        self._dispatch_next_query()

    def _dispatch_next_query(self):
        if self._query_busy or not self._database_bound or not self._run_id:
            return
        request = (
            self._pending_detail_query
            or self._pending_object_query
            or self._pending_history_query
            or self._pending_snapshot_query
        )
        if request is None:
            return
        if request is self._pending_detail_query:
            self._pending_detail_query = None
        elif request is self._pending_object_query:
            self._pending_object_query = None
        elif request is self._pending_history_query:
            self._pending_history_query = None
        else:
            self._pending_snapshot_query = None
        self._active_query = dict(request)
        self._query_busy = True
        self._query_requested.emit(dict(self._active_query))

    def _detail_request_matches_controls(self, request):
        value = dict(request or {})
        return (
            str(value.get("stream_id") or "") == self._selected_stream()
            and str(value.get("detail_kind") or "unit")
            == str(self._detail_kind.currentData() or "unit")
            and str(value.get("status") or "")
            == str(self._detail_status.currentData() or "")
            and str(value.get("search") or "")
            == self._detail_search.text().strip()
            and int(value.get("page") or 0) == self._page
            and int(value.get("page_size") or 0) == self._page_size
        )

    def _history_request_matches_controls(self, request):
        value = dict(request or {})
        return (
            str(value.get("scope") or "all")
            == str(self._history_scope.currentData() or "all")
            and str(value.get("execution_id") or "")
            == str(self._history_execution.currentData() or "")
            and str(value.get("search") or "")
            == self._history_search.text().strip()
            and value.get("context", {}) == self._history_filter_context()
        )

    @pyqtSlot(object)
    def _on_query_result(self, payload):
        value = dict(payload or {})
        active = dict(self._active_query or {})
        self._query_busy = False
        self._active_query = None
        current = (
            int(value.get("generation") or 0) == self._query_generation
            and str(value.get("run_id") or "") == self._run_id
            and str(value.get("kind") or "") == str(active.get("kind") or "")
            and int(value.get("request_id") or 0)
            == int(active.get("request_id") or 0)
        )
        if current and value.get("kind") == "detail":
            current = self._detail_request_matches_controls(active)
        if current and value.get("kind") == "history":
            current = self._history_request_matches_controls(active)
        if current and value.get("kind") == "object_history":
            current = self._object_request_matches_controls(active)
        if current and value.get("kind") == "snapshot":
            self._apply_database_snapshot(dict(value.get("snapshot") or {}))
        elif current and value.get("kind") == "detail":
            self._apply_detail_result(value)
        elif current and value.get("kind") == "history":
            self._apply_history_result(value)
        elif current and value.get("kind") == "object_history":
            self._apply_object_history_result(value)
        self._dispatch_next_query()

    @pyqtSlot(object)
    def _on_query_failed(self, payload):
        value = dict(payload or {})
        active = dict(self._active_query or {})
        self._query_busy = False
        self._active_query = None
        current = (
            int(value.get("generation") or 0) == self._query_generation
            and str(value.get("run_id") or "") == self._run_id
            and str(value.get("kind") or "") == str(active.get("kind") or "")
            and int(value.get("request_id") or 0)
            == int(active.get("request_id") or 0)
        )
        if current and value.get("kind") == "detail":
            current = self._detail_request_matches_controls(active)
        if current and value.get("kind") == "history":
            current = self._history_request_matches_controls(active)
        if current and value.get("kind") == "object_history":
            current = self._object_request_matches_controls(active)
        if current:
            error = str(value.get("error") or "unknown monitor query error")
            self._last_snapshot_error = error
            self._monitor_sync.setText(
                "监控：读取失败；保留最后一次有效状态"
            )
            self._on_log("system", f"[monitor-db] {error}")
            if value.get("kind") == "detail":
                self._detail_signature = None
                self._tile_rows.clear()
                self._tiles.setRowCount(0)
                self._tile_detail_title.setText(
                    f"选中结果流：{self._selected_stream()} | 数据库查询失败: {error}"
                )
            elif value.get("kind") == "history":
                self._history_detail.setText(
                    "历史读取失败；当前推理不受影响。\n" + error
                )
            elif value.get("kind") == "object_history":
                self._object_attempts.setText("历史尝试读取失败：\n" + error)
        self._dispatch_next_query()

    def _apply_detail_result(self, payload):
        stream_id = str(payload.get("stream_id") or "")
        kind = str(payload.get("detail_kind") or "unit")
        status = str(payload.get("status") or "")
        search = str(payload.get("search") or "")
        total = int(payload.get("total") or 0)
        page_total = max(1, int(payload.get("page_total") or 1))
        self._page = max(0, int(payload.get("page") or 0))
        rows = list(payload.get("rows") or ())
        selected_item = self._tiles.item(self._tiles.currentRow(), 0)
        selected_id = selected_item.text() if selected_item is not None else ""
        self._selected_object = {}
        if kind == "tile":
            headers = ["Tile", "Partition", "执行状态", "选择状态", "原因"]
            values = [
                (
                    row["tile_id"],
                    row.get("partition_id") or "-",
                    "全局输入",
                    TILE_STATUS_LABELS.get(row["status"], row["status"]),
                    "",
                )
                for row in rows
            ]
            detail_name = "Tile 输入清单"
            self._selected_object = {
                str(row["tile_id"]): dict(row) for row in rows
            }
        elif kind == "unit":
            headers = ["空间单元", "类型", "执行状态", "产物状态", "原因"]
            values = [
                (
                    row["unit_id"],
                    UNIT_TYPE_LABELS.get(row["unit_type"], row["unit_type"]),
                    "记录缺失",
                    UNIT_STATUS_LABELS.get(row["status"], row["status"]),
                    row["error"],
                )
                for row in rows
            ]
            detail_name = "Core / Seam / Junction"
            self._selected_object = {
                str(row["unit_id"]): dict(row) for row in rows
            }
        else:
            labels = {
                "package": "推理包",
                "fragmentation_v33": "碎片治理",
                "unit_confidence": "置信度计算",
                "unit_fit": "边界拟合",
            }
            headers = ["对象ID", "类型", "执行状态", "产物状态", "原因"]
            values = [
                (
                    row["object_id"],
                    UNIT_TYPE_LABELS.get(
                        str(row.get("object_label") or ""),
                        str(row.get("object_label") or labels.get(kind, kind)),
                    ),
                    UNIT_STATUS_LABELS.get(
                        str(row.get("execution_status") or ""),
                        str(row.get("execution_status") or "—"),
                    ),
                    UNIT_STATUS_LABELS.get(
                        str(row.get("artifact_status") or ""),
                        str(row.get("artifact_status") or "尚未就绪"),
                    ),
                    str(row.get("reason") or ""),
                )
                for row in rows
            ]
            detail_name = labels.get(kind, kind)
            self._selected_object = {
                str(row["object_id"]): dict(row) for row in rows
            }

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
            tuple((str(row.get("object_id") or row.get("tile_id") or row.get("unit_id")),
                   row.get("progress_current"), row.get("progress_total"),
                   row.get("execution_id"), row.get("span_id"), row.get("budget_attempt"))
                  for row in rows),
        )
        if signature == self._detail_signature:
            return
        self._detail_signature = signature
        self._tile_rows.clear()
        self._tiles.setUpdatesEnabled(False)
        blocked = self._tiles.blockSignals(True)
        try:
            self._tiles.setHorizontalHeaderLabels(headers)
            if self._tiles.rowCount() != len(values):
                self._tiles.setRowCount(len(values))
            for row_index, row_values in enumerate(values):
                self._tile_rows[str(row_values[0])] = row_index
                for column, value in enumerate(row_values):
                    text = str(value)
                    item = self._tiles.item(row_index, column)
                    if item is None:
                        item = QTableWidgetItem()
                        self._tiles.setItem(row_index, column, item)
                    if item.text() != text:
                        item.setText(text)
        finally:
            self._tiles.blockSignals(blocked)
            self._tiles.setUpdatesEnabled(True)
            self._tiles.viewport().update()
        self._tiles.request_adaptive_layout()
        if selected_id in self._tile_rows:
            self._tiles.selectRow(self._tile_rows[selected_id])
        elif values:
            self._tiles.selectRow(0)
        if values:
            self._render_object_detail()
        else:
            for browser in (self._object_current, self._object_models, self._object_attempts, self._object_events):
                browser.setPlainText("当前筛选没有对象。")

    def _poll_database(self):
        if not self._database_bound or not self._run_id:
            return
        self._query_serial += 1
        self._latest_snapshot_request_id = self._query_serial
        self._pending_snapshot_query = {
            "kind": "snapshot",
            "generation": self._query_generation,
            "request_id": self._latest_snapshot_request_id,
            "run_id": self._run_id,
            "run_spec": dict(self._run_spec),
        }
        self._dispatch_next_query()

    def _apply_database_snapshot(self, snapshot):
        terminal = getattr(self, "_terminal_run_status", "")
        if terminal and str((snapshot.get("run") or {}).get("status") or "") != terminal:
            return
        try:
            self._last_snapshot_at = time.time()
            self._last_snapshot_error = ""
            monitor_sync = getattr(self, "_monitor_sync", None)
            if monitor_sync is not None:
                monitor_sync.setText("监控：已连接，刚刚同步")
            run_row = snapshot.get("run") or {}
            if getattr(self, "_run_created_epoch", None) is None:
                self._run_created_epoch = _timestamp_epoch(
                    run_row.get("created_at") or ""
                )
            run_status = str(run_row.get("status") or "planned")
            job_counts = snapshot.get("job_counts") or {}
            job_progress = snapshot.get("job_progress") or {}
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
                observed = json.loads(str(active_package.get("monitor_runtime_json") or "{}"))
                observed_at = _timestamp_epoch(observed.get("observed_at") or "") or 0
                if observed_at > float(self._package_activity.get("_event_observed_at") or 0):
                    self._package_activity.update(observed)
                    self._active_inference_stream = str(observed.get("stream_id") or "")
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
            self._assembly_phase_statuses = {
                str(key): dict(value)
                for key, value in (
                    snapshot.get("assembly_phase_statuses") or {}
                ).items()
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
                        stage, status = "Run 已停止；恢复入口位于主界面", "已停止"
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
                feature_count = None
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
                    feature_count = assembly_info.get("feature_count")
                    phase_started = _timestamp_epoch(
                        assembly_info.get("phase_started_at") or ""
                    )
                    if phase_started is not None:
                        phase_key = str(assembly_info.get("phase") or "")
                        phase_history = (getattr(self, "_assembly_phase_statuses", {}).get(stream_id) or {}).get(phase_key) or {}
                        phase_end = _timestamp_epoch(phase_history.get("ended_at") or "")
                        elapsed = (
                            _elapsed_text(phase_end - phase_started) if phase_end is not None
                            else _elapsed_text(time.time() - phase_started)
                            if str(assembly_info.get("status")) == "running" and run_status not in {"ready", "failed", "stopped"}
                            else "—"
                        )

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

            history_applier = getattr(self, "_apply_monitor_history", None)
            if callable(history_applier):
                history_applier(snapshot.get("monitor_history") or {})
            overview_updater = getattr(self, "_update_database_overviews", None)
            if callable(overview_updater):
                overview_updater(
                    run_status=run_status,
                    package_counts=package_counts,
                    unit_job_counts=unit_job_counts,
                    job_counts=job_counts,
                    job_progress=job_progress,
                    active_package=active_package,
                    streams=streams,
                    stream_runtime_progress=all_runtime_progress,
                )
            assembly_renderer = getattr(self, "_render_selected_assembly", None)
            if callable(assembly_renderer):
                assembly_renderer()
            tile_renderer = getattr(self, "_render_selected_tiles", None)
            if callable(tile_renderer) and getattr(self, "isVisible", lambda: True)():
                tile_renderer()
        except Exception as error:
            logger = getattr(self, "_on_log", None)
            if callable(logger):
                logger("system", f"[monitor-db] {error}")
            else:
                panel = getattr(self, "_log_panel", None)
                if panel is not None:
                    panel.append_system(f"[monitor-db] {error}")

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
            return "stopped", "已停止；恢复操作位于主界面", 0, 1
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
        job_counts,
        job_progress,
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
        if not self._control_state:
            display_stage = {
                "Work Package 推理 + 空间单元拟合": "正在识别地物，同时处理边界",
                "Work Package 推理": "正在识别地物",
                "空间单元拟合": "正在处理空间边界",
                "结果流并行组装": "正在组装模型与融合结果",
                "分区概率栅格收口": "正在汇总推理栅格",
                "整体验收": "正在核验本次推理结果",
            }.get(stage, stage)
            self._phase.setText(display_stage)
            self._phase.setToolTip(f"{stage} · 当前阶段观察 {stage_elapsed}")
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
        status_text = RUN_STATUS_LABELS.get(run_status, run_status)
        if not self._control_state:
            self._status_badge.setText(status_text)
            self._set_status_badge(
                "active"
                if run_status in {"running", "raster_ready", "ready"}
                else "failed" if run_status == "failed"
                else "warning" if run_status == "resetting"
                else "neutral"
            )
        self._run_id_label.setText(
            f"Run：{self._run_id} · 任务：{status_text} · "
            f"执行：{str((getattr(self, '_latest_execution', {}) or {}).get('execution_id') or '—')[:12]}"
        )

        overall_fraction, overall_group_count = _overall_completion_fraction(
            run_status,
            job_counts,
            job_progress,
            streams,
            stream_runtime_progress,
        )
        overall_value = round(overall_fraction * ASSEMBLY_PROGRESS_SCALE)
        overall_percent = round(overall_fraction * 100)
        self._overall_bar.setRange(0, ASSEMBLY_PROGRESS_SCALE)
        self._overall_bar.setValue(overall_value)
        self._overall_bar.setFormat(
            f"本次推理任务完成度：{overall_percent}% | "
            f"按 {overall_group_count} 类任务计算，不代表剩余时间"
        )

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

        backend, device_name = effective_device_text(self._run_spec)
        device = f"{backend} · {device_name}"
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
        self._device_label.setText(device)
        self._device_label.setToolTip("本次 Run 的有效执行设备；不是实时利用率采集。")
        self._spatial_device_label.setText("CPU · 与模型计算并行")

        package_total = sum(int(value) for value in package_counts.values())
        package_ready = int(package_counts.get("ready", 0))
        package_running = int(package_counts.get("running", 0))
        package_waiting = _waiting_count(package_counts)
        package_failed = int(package_counts.get("failed", 0))
        current_text = "当前包：—\n当前模型：—\n影像块：—  ·  Batch：—"
        self._current_package.setText("—")
        self._current_model.setText("—")
        self._tile_count.setText("— / —")
        self._batch_value.setText("批量大小：—")
        self._set_progress_bar(self._tile_card_bar, 0, 0)
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
            self._current_package.setText(f"第 {sequence} 包")
            self._current_package.setToolTip(package_id)
            self._current_model.setText(model_text)
            self._tile_count.setText(tile_text.replace("/", " / "))
            self._batch_value.setText(f"批量大小：{batch_text}")
            self._batch_value.setToolTip("配置 → 当前有效 Batch；" + str(activity.get("notice") or "未记录降档"))
            current_text = (
                f"当前包：第 {sequence} 包  ·  {activity.get('status') or '运行中'}\n"
                f"当前模型：{model_text}\n"
                f"影像块：{tile_text}  ·  Batch：{batch_text}"
            )
            self._package_overview.setToolTip(f"{package_id}\n配置/有效 Batch：{batch_text}\n包耗时：{_elapsed_text(package_elapsed)}\n{activity.get('notice') or ''}")
            self._set_progress_bar(self._tile_card_bar, tile_current, tile_total)
        self._package_overview.setText(
            current_text
        )
        self._package_metric.setText(f"{package_ready:,} / {package_total:,}")
        self._badge(self._model_badge,
                    "失败" if package_failed else "运行中" if package_running and run_status == "running" else "已完成" if package_total and package_ready == package_total else "已停止" if run_status == "stopped" else "等待",
                    "failed" if package_failed else "active" if package_running and run_status == "running" else "neutral")
        self._set_progress_bar(
            self._package_card_bar,
            package_ready,
            package_total,
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
            "全部结果流任务"
        )
        self._unit_overview.setToolTip(f"配置并发上限 {worker_text}。{blocker_text}各分类按任务计数，不能相加作为空间单元数量。")
        spatial_active = sum(
            int((job_counts.get(kind) or {}).get("running", 0))
            for kind in ("fragmentation_v33", "unit_confidence", "unit_fit")
        )
        self._unit_metric.setText(f"{unit_ready:,} / {unit_total:,}" if unit_total else "— / —")
        for title, value in (("运行", unit_running), ("等待", unit_waiting), ("失败", unit_failed)):
            self._spatial_stats[title].setText(f"{value:,}")
        self._badge(self._spatial_badge,
                    "失败" if unit_failed else "运行中" if spatial_active and run_status == "running" else "已停止" if run_status == "stopped" else "等待" if unit_waiting else "已完成" if unit_total else "待开始",
                    "failed" if unit_failed else "active" if spatial_active and run_status == "running" else "neutral")
        fragmentation_enabled = bool(
            (self._run_spec.get("fragmentation_regularization") or {}).get(
                "enabled", True
            )
        )
        boundary_enabled = bool(
            (self._run_spec.get("boundary_fitting") or {}).get("enabled", True)
        )
        self._fit_caption.setText("边界拟合任务已完成" if boundary_enabled else "原始边界任务已完成")
        self._update_task_lane(
            self._fragment_label,
            self._fragment_bar,
            "碎片治理",
            job_counts.get("fragmentation_v33") or {},
            enabled=fragmentation_enabled,
        )
        self._update_task_lane(
            self._confidence_label,
            self._confidence_bar,
            "置信度计算",
            job_counts.get("unit_confidence") or {},
            enabled=fragmentation_enabled,
        )
        self._update_task_lane(
            self._fit_label,
            self._fit_bar,
            "边界拟合" if boundary_enabled else "原始边界处理",
            job_counts.get("unit_fit") or {},
            enabled=True,
        )
        if run_status == "failed" or package_failed or unit_failed:
            self._action_label.setText("需要查看失败原因")
            self._action_description.setText("已记录任务失败，请在事件与日志中查看影响和原因。")
        elif run_status == "resetting":
            self._action_label.setText("正在重做准备")
            self._action_description.setText("失败包正在重置，暂不需要重复操作。")
        elif run_status == "stopped":
            self._action_label.setText("任务已停止")
            self._action_description.setText("恢复与重做失败包仍位于主界面。")
        elif run_status == "ready":
            self._action_label.setText("本次推理已完成")
            self._action_description.setText("可查看结果与验收；人工修整仍在原窗口进行。")
        else:
            self._action_label.setText("当前无需人工处理")
            self._action_description.setText("当前没有已确认的人工处理事项，详情可在日志中查看。")
        attention = bool(run_status == "failed" or package_failed or unit_failed)
        if self._health_panel.property("attention") != attention:
            self._health_panel.setProperty("attention", attention)
            self._health_panel.style().unpolish(self._health_panel)
            self._health_panel.style().polish(self._health_panel)
        health_icon = "alert" if attention else "clock" if run_status == "stopped" else "shield"
        health_tone = "warning" if attention else "muted" if run_status == "stopped" else "success"
        self._health_panel.setProperty("healthIconName", health_icon)
        self._health_panel.setProperty("healthIconTone", health_tone)
        self._health_icon.setPixmap(
            monitor_icon(health_icon, PALETTES[self._theme][health_tone], 48).pixmap(QSize(48, 48))
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
            (tile_id, "Tile", state["status"], state["progress"], state["error"])
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
        final_error = re.sub(
            r"\s+", " ", str(result.get("error") or "")
        ).strip().lower()
        if (
            not result.get("success")
            and result.get("status") != "stopped"
            and result.get("error")
            and final_error not in self._logged_error_texts
        ):
            self._on_log(
                "system",
                json.dumps(
                    {
                        "event": "monitor_pipeline_failed",
                        "error": str(result.get("error")),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
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
        if self._database_bound and self._run_id:
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

    def shutdown(self, timeout_ms=5000):
        from ..core.qt_lifecycle import retire_after

        del timeout_ms
        if getattr(self, "_shutting_down", False):
            return
        self._shutting_down = True
        self.hide()
        self.unbind_state_database()
        if hasattr(self, "_detail_search_timer"):
            self._detail_search_timer.stop()
        thread = getattr(self, "_query_thread", None)
        retire_after(self, self.shutdown_finished)
        if thread is not None and thread.isRunning():
            thread.quit()
        else:
            self._finish_shutdown()

    @pyqtSlot()
    def _finish_shutdown(self):
        if not self._query_thread.wait(0):
            QTimer.singleShot(10, self._finish_shutdown)
            return
        self.shutdown_finished.emit()

    def closeEvent(self, event):
        event.ignore()
        self.hide()
        parent = self.parent()
        if parent is not None and hasattr(parent, "show_monitor_btn"):
            parent.show_monitor_btn.setChecked(False)
            parent.show_monitor_btn.setText("推理监控")
