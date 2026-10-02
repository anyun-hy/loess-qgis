"""SAM3 session controls and display rules; no session state is retained."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

SamPanelState = Literal[
    "idle",
    "waiting_click",
    "inference",
    "candidate",
    "failed",
    "cancelled",
]


@dataclass(frozen=True)
class SamPanelSnapshot:
    """Immutable values sampled by the dialog immediately before a render."""

    state: SamPanelState = "idle"
    existing: bool = False
    message: str = "无活动会话"
    topology_hint: str = "-"
    error: str = ""


class SamSessionPanel(QGroupBox):
    """Own SAM3 session widgets and emit user decisions."""

    decision_requested = pyqtSignal(str)
    retry_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("SAM3 会话", parent)
        self.setObjectName("samSessionPanel")
        layout = QVBoxLayout(self)

        self._message = QLabel("无活动会话")
        self._message.setObjectName("samSessionLabel")
        self._message.setWordWrap(True)
        layout.addWidget(self._message)

        self._topology = QLabel("局部拓扑提示: -")
        self._topology.setObjectName("samTopologyHint")
        self._topology.setWordWrap(True)
        layout.addWidget(self._topology)

        self._error = QPlainTextEdit()
        self._error.setObjectName("samSessionError")
        self._error.setReadOnly(True)
        self._error.setMaximumHeight(90)
        self._error.hide()
        layout.addWidget(self._error)

        action_row = QHBoxLayout()
        self._keep_current = QPushButton("保留当前")
        self._adopt_candidate = QPushButton("采用 SAM3")
        self._edit_current = QPushButton("编辑当前")
        self._edit_candidate = QPushButton("编辑 SAM3")
        self._retry = QPushButton("重试")
        self._cancel = QPushButton("取消")
        for button, name in (
            (self._keep_current, "samKeepCurrent"),
            (self._adopt_candidate, "samAdoptCandidate"),
            (self._edit_current, "samEditCurrent"),
            (self._edit_candidate, "samEditCandidate"),
            (self._retry, "samRetry"),
            (self._cancel, "samCancel"),
        ):
            button.setObjectName(name)
            action_row.addWidget(button)
        action_row.addStretch()
        layout.addLayout(action_row)

        for button, decision in (
            (self._keep_current, "kept_current"),
            (self._adopt_candidate, "adopted"),
            (self._edit_current, "edit_current"),
            (self._edit_candidate, "edit_sam3"),
            (self._cancel, "cancelled"),
        ):
            button.clicked.connect(
                lambda _checked=False, value=decision: self.decision_requested.emit(
                    value
                )
            )
        self._retry.clicked.connect(self.retry_requested)
        self.render(SamPanelSnapshot())

    def replace_log(self, text: str) -> None:
        """Replace log text without changing its visibility."""
        self._error.setPlainText(text)

    def render(self, snapshot: SamPanelSnapshot) -> None:
        """Render a session snapshot without retaining it."""
        candidate = snapshot.state == "candidate"
        failed = snapshot.state == "failed"
        self._message.setText(snapshot.message)
        self._topology.setText(f"局部拓扑提示: {snapshot.topology_hint or '-'}")
        self._error.setPlainText(snapshot.error)
        self._error.setVisible(failed)
        self._keep_current.setEnabled(candidate and snapshot.existing)
        self._adopt_candidate.setEnabled(candidate)
        self._edit_current.setEnabled((candidate or failed) and snapshot.existing)
        self._edit_candidate.setEnabled(candidate)
        self._retry.setEnabled(failed)
        self._cancel.setEnabled(snapshot.state != "idle")
        self.setVisible(snapshot.state != "idle")
