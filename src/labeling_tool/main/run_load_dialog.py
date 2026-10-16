"""Choose an existing Run and the result layers to mount."""

from qgis.PyQt.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from labeling_tool.refinement.manual_run_loader import LOAD_ALL, LOAD_CLASSES


class RunLoadDialog(QDialog):
    def __init__(self, directory="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Run 加载与整理")
        self.resize(620, 260)
        root = QVBoxLayout(self)
        root.addWidget(QLabel("选择已经初始化 classes 的 Run 文件夹"))
        row = QHBoxLayout()
        self.directory_edit = QLineEdit(str(directory))
        self.directory_edit.setAccessibleName("Run 文件夹")
        choose = QPushButton("选择目录")
        choose.clicked.connect(self._choose)
        row.addWidget(self.directory_edit, 1)
        row.addWidget(choose)
        root.addLayout(row)
        root.addWidget(QLabel("加载范围："))
        self.classes_radio = QRadioButton("只加载 classes")
        self.all_radio = QRadioButton("加载 fusion、models、classes")
        self.classes_radio.setChecked(True)
        root.addWidget(self.classes_radio)
        root.addWidget(self.all_radio)
        self.sam_status_label = QLabel("SAM3 加载中…")
        self.sam_status_label.setWordWrap(True)
        root.addWidget(self.sam_status_label)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("加载并进入整理")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    @property
    def run_directory(self):
        return self.directory_edit.text().strip()

    @property
    def load_scope(self):
        return LOAD_ALL if self.all_radio.isChecked() else LOAD_CLASSES

    def _choose(self):
        directory = QFileDialog.getExistingDirectory(
            self, "选择 Run 文件夹", self.run_directory
        )
        if directory:
            self.directory_edit.setText(directory)

    def accept(self):
        if not self.run_directory:
            QMessageBox.warning(self, "Run 文件夹", "请选择 Run 文件夹")
            return
        super().accept()
