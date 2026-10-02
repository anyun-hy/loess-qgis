"""Qt rendering for the read-only refinement admission summary."""

from __future__ import annotations

from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from labeling_tool.refinement.admission_presentation import (
    AdmissionSummarySnapshot,
    admission_summary_presentation,
)


class AdmissionSummaryPanel(QGroupBox):
    """A copyable admission summary that forwards, but never performs, writes."""

    write_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("入库摘要", parent)
        self.setObjectName("AdmissionSummaryPanel")
        layout = QVBoxLayout(self)
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("目标标签库："))
        self._target_path = QLineEdit()
        self._target_path.setObjectName("AdmissionTargetPath")
        self._target_path.setReadOnly(True)
        self._target_path.setPlaceholderText("目标标签库路径未提供")
        self._target_path.setToolTip("可全选并复制完整路径")
        self._target_path.setAccessibleName("目标标签库完整路径")
        path_row.addWidget(self._target_path, stretch=1)
        layout.addLayout(path_row)

        facts = QGridLayout()
        facts.setHorizontalSpacing(18)
        facts.setVerticalSpacing(6)
        self._labels: dict[str, QLabel] = {}
        for index, (key, title) in enumerate(
            (
                ("final_features", "最终成果"),
                ("class_confirmation", "14 类确认"),
                ("unsaved_edits", "未保存编辑"),
                ("topology", "拓扑检查"),
                ("background_stage", "后台阶段"),
                ("blocker", "阻塞原因"),
                ("accepted_result", "本次实际新增"),
            )
        ):
            row, column = divmod(index, 2)
            value = QLabel()
            value.setObjectName(f"Admission{key.title().replace('_', '')}")
            value.setWordWrap(True)
            facts.addWidget(QLabel(f"{title}："), row, column * 2)
            facts.addWidget(value, row, column * 2 + 1)
            self._labels[key] = value
        layout.addLayout(facts)

        self._integrity_note = QLabel()
        self._integrity_note.setObjectName("AdmissionIntegrityNote")
        self._integrity_note.setWordWrap(True)
        layout.addWidget(self._integrity_note)
        self._accepted_warnings = QLabel()
        self._accepted_warnings.setObjectName("AdmissionAcceptedWarnings")
        self._accepted_warnings.setWordWrap(True)
        layout.addWidget(self._accepted_warnings)
        self._next_action = QLabel()
        self._next_action.setObjectName("AdmissionNextAction")
        self._next_action.setWordWrap(True)
        layout.addWidget(self._next_action)

        self._write_button = QPushButton("检查并写入标签库")
        self._write_button.setObjectName("AdmissionWriteButton")
        self._write_button.clicked.connect(self.write_requested)
        layout.addWidget(self._write_button)
        self.render(AdmissionSummarySnapshot())

    def render(self, snapshot: AdmissionSummarySnapshot) -> None:
        """Render a fresh observation; this panel keeps no business state."""

        presentation = admission_summary_presentation(snapshot)
        self._target_path.setText(presentation.target_path)
        for key, label in self._labels.items():
            label.setText(getattr(presentation, key))
        self._integrity_note.setText(presentation.integrity_note)
        self._accepted_warnings.setText("\n".join(presentation.accepted_warnings))
        self._accepted_warnings.setVisible(bool(presentation.accepted_warnings))
        self._next_action.setText(presentation.next_action)
        self._write_button.setEnabled(bool(snapshot.write_enabled))
        self._write_button.setToolTip(
            str(snapshot.write_reason or snapshot.blocker_reason or "")
        )
