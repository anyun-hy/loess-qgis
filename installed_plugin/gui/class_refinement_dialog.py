"""Modeless 14-class Fusion workspace with click-driven SAM3 refinement."""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsFeatureRequest,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsRectangle,
    QgsRasterLayer,
    QgsVectorLayer,
)
from qgis.gui import (
    QgsMapToolCapture,
    QgsMapToolDigitizeFeature,
    QgsMapToolEmitPoint,
    QgsRubberBand,
)
from qgis.analysis import QgsZonalStatistics

from ..core import accepted_writer, class_workspace, final_assembler, topology_validator
from ..core.layer_names import LAYER_NAMES
from ..core.sam3_worker_runner import Sam3WorkerRunner
from ..core.style_manager import StyleManager
from ..core.run_spec import CLASS_NAMES, CLASS_ORDER


class ClassRefinementDialog(QDialog):
    workspace_changed = pyqtSignal(object)

    def __init__(self, iface, layer_manager, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.layer_manager = layer_manager
        self.setWindowTitle("分类修整与组装")
        self.setWindowFlags(Qt.Window)
        self.resize(1220, 760)
        self.setMinimumSize(980, 620)
        self._result = {}
        self._run_spec = {}
        self._sam_config = {}
        self._scripts_dir = ""
        self._eligible_fusions = []
        self._workspace = None
        self._manual_only = False
        self._class_layers = {}
        self._connected_layer_ids = set()
        self._layer_signal_slots = {}
        self._visibility_checks = {}
        self._sam_buttons = {}
        self._sam_existing_actions = {}
        self._sam_missed_actions = {}
        self._edit_buttons = {}
        self._quick_redraw_actions = {}
        self._edit_selected_actions = {}
        self._add_manual_actions = {}
        self._reclassify_actions = {}
        self._save_edit_actions = {}
        self._rollback_edit_actions = {}
        self._confirm_buttons = {}
        self._snapshots = {}
        self._edit_context = {}
        self._metadata_update = False
        self._worker = None
        self._worker_request = None
        self._active_session = None
        self._previous_map_tool = None
        self._point_tool = None
        self._manual_pick_code = None
        self._active_redraw = None
        self._redraw_pick_tool = None
        self._redraw_capture_tool = None
        self._redraw_previous_map_tool = None
        self._redraw_reference_band = None
        self._redraw_candidate_band = None
        self._current_band = None
        self._candidate_band = None
        self._confidence_raster = None
        self._final_path = ""
        self._issues_path = ""
        self._issue_count = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        baseline = QHBoxLayout()
        baseline.addWidget(QLabel("Fusion 基准:"))
        self.fusion_combo = QComboBox()
        baseline.addWidget(self.fusion_combo, stretch=1)
        self.initialize_btn = QPushButton("初始化 14 类工作层")
        self.initialize_btn.clicked.connect(self._initialize_workspace)
        baseline.addWidget(self.initialize_btn)
        self.baseline_label = QLabel("尚未加载运行结果")
        self.baseline_label.setWordWrap(True)
        baseline.addWidget(self.baseline_label, stretch=2)
        root.addLayout(baseline)

        self.table = QTableWidget(len(CLASS_ORDER), 8)
        self.table.setHorizontalHeaderLabels(
            ["可见", "类别", "色块", "类别工作层状态", "面数", "SAM3 校正", "人工操作", "整类确认"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.cellClicked.connect(self._activate_row)
        header = self.table.horizontalHeader()
        for column in (0, 1, 2, 4, 5, 6, 7):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        for row, code in enumerate(CLASS_ORDER):
            visible = QCheckBox()
            visible.setChecked(True)
            visible.toggled.connect(lambda checked, c=code: self._set_visible(c, checked))
            visible_box = QWidget()
            visible_layout = QHBoxLayout(visible_box)
            visible_layout.setContentsMargins(6, 0, 6, 0)
            visible_layout.addWidget(visible)
            visible_layout.setAlignment(Qt.AlignCenter)
            self.table.setCellWidget(row, 0, visible_box)
            self._visibility_checks[code] = visible
            self.table.setItem(row, 1, QTableWidgetItem(f"{code} {CLASS_NAMES[code]}"))
            swatch = QTableWidgetItem("    ")
            swatch.setBackground(QColor(StyleManager.get_class_color(code)))
            self.table.setItem(row, 2, swatch)
            self.table.setItem(row, 3, QTableWidgetItem("未初始化"))
            self.table.setItem(row, 4, QTableWidgetItem("0"))
            sam_button = QPushButton("校正边界")
            menu = sam_button.menu()
            if menu is None:
                from qgis.PyQt.QtWidgets import QMenu

                menu = QMenu(sam_button)
                sam_button.setMenu(menu)
            existing_action = menu.addAction("校正已有地物")
            existing_action.triggered.connect(
                lambda _checked=False, c=code: self._begin_sam(c, missed=False)
            )
            missed_action = menu.addAction("新增漏标面")
            missed_action.triggered.connect(
                lambda _checked=False, c=code: self._begin_sam(c, missed=True)
            )
            self.table.setCellWidget(row, 5, sam_button)
            self._sam_buttons[code] = sam_button
            self._sam_existing_actions[code] = existing_action
            self._sam_missed_actions[code] = missed_action
            edit_button = QPushButton("人工操作")
            edit_menu = edit_button.menu()
            if edit_menu is None:
                from qgis.PyQt.QtWidgets import QMenu

                edit_menu = QMenu(edit_button)
                edit_button.setMenu(edit_menu)
            quick_redraw_action = edit_menu.addAction("快速重画现有面")
            quick_redraw_action.triggered.connect(
                lambda _checked=False, c=code: self._begin_quick_redraw(c)
            )
            add_manual_action = edit_menu.addAction("新增人工面")
            add_manual_action.triggered.connect(
                lambda _checked=False, c=code: self._add_manual_feature(c)
            )
            edit_selected_action = edit_menu.addAction("节点精修（备用）")
            edit_selected_action.triggered.connect(
                lambda _checked=False, c=code: self._edit_selected(c)
            )
            reclassify_action = edit_menu.addAction("更正类别")
            reclassify_action.triggered.connect(
                lambda _checked=False, c=code: self._reclassify_selected(c)
            )
            edit_menu.addSeparator()
            save_edit_action = edit_menu.addAction("保存当前编辑")
            save_edit_action.triggered.connect(
                lambda _checked=False, c=code: self._save_class_edits(c)
            )
            rollback_edit_action = edit_menu.addAction("取消当前编辑")
            rollback_edit_action.triggered.connect(
                lambda _checked=False, c=code: self._rollback_class_edits(c)
            )
            self.table.setCellWidget(row, 6, edit_button)
            self._edit_buttons[code] = edit_button
            self._quick_redraw_actions[code] = quick_redraw_action
            self._edit_selected_actions[code] = edit_selected_action
            self._add_manual_actions[code] = add_manual_action
            self._reclassify_actions[code] = reclassify_action
            self._save_edit_actions[code] = save_edit_action
            self._rollback_edit_actions[code] = rollback_edit_action
            confirm = QPushButton("确认整类")
            confirm.setCheckable(True)
            confirm.setToolTip("确认当前类别全部面已审核完成，不用于保存单个面")
            confirm.toggled.connect(lambda checked, c=code: self._confirm_class(c, checked))
            self.table.setCellWidget(row, 7, confirm)
            self._confirm_buttons[code] = confirm
        root.addWidget(self.table, stretch=1)

        self.session_group = QGroupBox("SAM3 会话")
        session_layout = QVBoxLayout(self.session_group)
        self.session_label = QLabel("无活动会话")
        self.session_label.setWordWrap(True)
        session_layout.addWidget(self.session_label)
        self.local_topology_label = QLabel("局部拓扑提示: -")
        self.local_topology_label.setWordWrap(True)
        session_layout.addWidget(self.local_topology_label)
        self.session_error = QPlainTextEdit()
        self.session_error.setReadOnly(True)
        self.session_error.setMaximumHeight(90)
        self.session_error.hide()
        session_layout.addWidget(self.session_error)
        decisions = QHBoxLayout()
        self.keep_btn = QPushButton("保留当前")
        self.adopt_btn = QPushButton("采用 SAM3")
        self.edit_current_btn = QPushButton("编辑当前")
        self.edit_sam_btn = QPushButton("编辑 SAM3")
        self.retry_btn = QPushButton("重试")
        self.cancel_session_btn = QPushButton("取消")
        for button in (
            self.keep_btn, self.adopt_btn, self.edit_current_btn,
            self.edit_sam_btn, self.retry_btn, self.cancel_session_btn,
        ):
            decisions.addWidget(button)
        decisions.addStretch()
        session_layout.addLayout(decisions)
        self.keep_btn.clicked.connect(lambda: self._finish_session("kept_current"))
        self.adopt_btn.clicked.connect(lambda: self._finish_session("adopted"))
        self.edit_current_btn.clicked.connect(lambda: self._finish_session("edit_current"))
        self.edit_sam_btn.clicked.connect(lambda: self._finish_session("edit_sam3"))
        self.retry_btn.clicked.connect(self._retry_session)
        self.cancel_session_btn.clicked.connect(lambda: self._finish_session("cancelled"))
        self.session_group.hide()
        root.addWidget(self.session_group)

        self.redraw_group = QGroupBox("快速重画")
        redraw_layout = QVBoxLayout(self.redraw_group)
        self.redraw_label = QLabel("无活动重画")
        self.redraw_label.setWordWrap(True)
        redraw_layout.addWidget(self.redraw_label)
        redraw_controls = QHBoxLayout()
        redraw_controls.addWidget(QLabel("平滑次数:"))
        self.redraw_smooth_spin = QSpinBox()
        self.redraw_smooth_spin.setRange(0, 5)
        self.redraw_smooth_spin.setValue(1)
        self.redraw_smooth_spin.setToolTip("0=不平滑；1=轻度；2-3=明显；4-5=强平滑")
        redraw_controls.addWidget(self.redraw_smooth_spin)
        self.redraw_adopt_btn = QPushButton("采用")
        self.redraw_retry_btn = QPushButton("重画")
        self.redraw_cancel_btn = QPushButton("取消")
        redraw_controls.addWidget(self.redraw_adopt_btn)
        redraw_controls.addWidget(self.redraw_retry_btn)
        redraw_controls.addWidget(self.redraw_cancel_btn)
        redraw_controls.addStretch()
        redraw_layout.addLayout(redraw_controls)
        self.redraw_smooth_spin.valueChanged.connect(self._update_redraw_preview)
        self.redraw_adopt_btn.clicked.connect(self._adopt_quick_redraw)
        self.redraw_retry_btn.clicked.connect(self._retry_quick_redraw)
        self.redraw_cancel_btn.clicked.connect(self._cancel_quick_redraw)
        self.redraw_group.hide()
        root.addWidget(self.redraw_group)

        actions = QHBoxLayout()
        self.assemble_btn = QPushButton("组装最终图层")
        self.topology_btn = QPushButton("重新检查拓扑")
        self.accept_btn = QPushButton("写入 accepted_labels")
        self.allow_issues_check = QCheckBox("明确允许带未解决问题入库")
        self.summary_label = QLabel("14 类确认: 0/14    未解决问题: -    未保存编辑: 0")
        actions.addWidget(self.assemble_btn)
        actions.addWidget(self.topology_btn)
        actions.addWidget(self.accept_btn)
        actions.addWidget(self.allow_issues_check)
        actions.addWidget(self.summary_label, stretch=1)
        root.addLayout(actions)
        self.assemble_btn.clicked.connect(self._assemble_final)
        self.topology_btn.clicked.connect(self._check_topology)
        self.accept_btn.clicked.connect(self._write_accepted)
        self.allow_issues_check.toggled.connect(self._update_accept_enabled)
        self.topology_btn.setEnabled(False)
        self.accept_btn.setEnabled(False)
        self._set_decision_state("idle")
        self._update_actions()

    def set_run(self, result, run_spec, sam_config, scripts_dir):
        self._cancel_quick_redraw()
        self._cancel_active_session(record=False)
        self._disconnect_layer_signals()
        self._snapshots.clear()
        self._workspace = None
        self._class_layers = {}
        self._result = dict(result)
        self._run_spec = dict(run_spec)
        self._sam_config = dict(sam_config or {})
        self._scripts_dir = str(scripts_dir)
        self._manual_only = bool(self._run_spec.get("manual_only"))
        self.setWindowTitle(
            "人工分类整理" if self._manual_only else "分类修整与组装"
        )
        self.table.setColumnHidden(5, self._manual_only)
        self.session_group.hide()
        self._confidence_raster = None
        streams = [
            item for item in result.get("ready_streams") or []
            if item.get("status") == "ready"
        ]
        self._eligible_fusions = class_workspace.approved_fusion_streams(
            self._run_spec, streams
        )
        self.fusion_combo.clear()
        for stream in self._eligible_fusions:
            self.fusion_combo.addItem(stream["stream_id"], stream["stream_id"])
        self.initialize_btn.setEnabled(bool(self._eligible_fusions))
        workspace_path = class_workspace.workspace_paths(self._run_spec)["workspace"]
        if workspace_path.is_file():
            try:
                self._workspace = class_workspace.load_workspace(self._run_spec)
                self._load_workspace_layers()
                self.fusion_combo.setCurrentText(self._workspace["baseline_stream_id"])
                self.fusion_combo.setEnabled(False)
                self.initialize_btn.setEnabled(False)
            except Exception as exc:
                self._workspace = None
                QMessageBox.warning(self, "恢复类别工作区失败", str(exc))
        if self._workspace is None:
            self.baseline_label.setText(
                f"Run {run_spec.get('run_id')}；可用 Fusion {len(self._eligible_fusions)} 个"
            )
        if self._manual_only and self._workspace is None and len(self._eligible_fusions) == 1:
            self.fusion_combo.setCurrentIndex(0)
            self._initialize_workspace()
        else:
            self._refresh_table()

    def _initialize_workspace(self):
        stream_id = str(self.fusion_combo.currentData() or "")
        if not stream_id:
            QMessageBox.warning(self, "初始化工作区", "没有通过规则化和审批的 Fusion")
            return
        try:
            stream = class_workspace.stream_by_id(self._eligible_fusions, stream_id)
            self._workspace = class_workspace.initialize_workspace(
                self._run_spec, stream
            )
            self._load_workspace_layers()
            self.fusion_combo.setEnabled(False)
            self.initialize_btn.setEnabled(False)
            self._refresh_table()
        except Exception as exc:
            QMessageBox.warning(self, "初始化 14 类工作层失败", str(exc))

    def _load_workspace_layers(self):
        self._class_layers = self.layer_manager.load_workspace_classes(
            self._run_spec["run_id"], self._workspace
        )
        for code, layer_id in self._class_layers.items():
            layer = QgsProject.instance().mapLayer(layer_id)
            if layer is None:
                continue
            class_workspace.apply_class_constraints(
                layer,
                code,
                run_id=self._run_spec["run_id"],
                baseline_stream_id=self._workspace["baseline_stream_id"],
            )
            if layer.isEditable() and code not in self._snapshots:
                self._snapshots[code] = self._persisted_snapshot(code)
            if layer.id() not in self._connected_layer_ids:
                started_slot = lambda c=code: self._editing_started(c)
                committed_slot = lambda c=code: self._editing_stopped(c)
                stopped_slot = lambda c=code: self._editing_stopped(c)
                layer.editingStarted.connect(started_slot)
                layer.afterCommitChanges.connect(committed_slot)
                layer.editingStopped.connect(stopped_slot)
                self._layer_signal_slots[layer.id()] = (
                    started_slot, committed_slot, stopped_slot
                )
                self._connected_layer_ids.add(layer.id())
        text = (
            f"基准 {self._workspace['baseline_stream_id']} | "
            f"formal SHA {self._workspace['formal_sha256'][:12]}... | "
            f"{self._workspace['feature_count']} 个面"
        )
        report_sha = str(self._workspace.get("boundary_report_sha256") or "")
        if report_sha:
            text += f" | report SHA {report_sha[:12]}..."
        if self._manual_only:
            text += " | 纯人工模式"
        self.baseline_label.setText(text)

    def _disconnect_layer_signals(self):
        for layer_id, slots in list(self._layer_signal_slots.items()):
            layer = QgsProject.instance().mapLayer(layer_id)
            if layer is None:
                continue
            for signal, slot in zip(
                (
                    layer.editingStarted,
                    layer.afterCommitChanges,
                    layer.editingStopped,
                ),
                slots,
            ):
                try:
                    signal.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
        self._layer_signal_slots.clear()
        self._connected_layer_ids.clear()

    def _layer(self, class_code):
        layer = QgsProject.instance().mapLayer(self._class_layers.get(int(class_code), ""))
        if layer is None:
            raise RuntimeError(f"class {class_code} working layer is not loaded")
        return layer

    def _refresh_class_display(self, *class_codes):
        for class_code in class_codes:
            layer = self._layer(class_code)
            StyleManager.apply_categorized_style(layer)
            layer.removeSelection()
            layer.triggerRepaint()
        canvas = self.iface.mapCanvas()
        canvas.clearCache()
        canvas.refresh()

    def _set_visible(self, class_code, visible):
        layer_id = self._class_layers.get(class_code)
        if layer_id:
            self.layer_manager.set_layer_visibility(layer_id, visible)

    def _activate_row(self, row, _column):
        if 0 <= row < len(CLASS_ORDER) and CLASS_ORDER[row] in self._class_layers:
            self.iface.setActiveLayer(self._layer(CLASS_ORDER[row]))

    def _snapshot(self, layer):
        snapshot = {}
        for feature in layer.getFeatures():
            snapshot[feature.id()] = {
                "object_id": str(feature.attribute("object_id") or ""),
                "geometry_hash": class_workspace.geometry_hash(feature.geometry()),
                "geometry_source": str(feature.attribute("geometry_source") or "fusion"),
                "geometry_revision": int(feature.attribute("geometry_revision") or 0),
                "immutable": {
                    name: feature.attribute(name)
                    for name in class_workspace.IMMUTABLE_FIELDS
                },
            }
        return snapshot

    def _persisted_snapshot(self, class_code):
        record = self._workspace["classes"][str(class_code)]
        persisted = class_workspace.working_layer(
            record, f"class_{class_code}_persisted_snapshot"
        )
        return self._snapshot(persisted)

    def _editing_started(self, class_code):
        if self._metadata_update:
            return
        self._snapshots[class_code] = self._snapshot(self._layer(class_code))
        self._update_actions()

    def _editing_stopped(self, class_code):
        if self._metadata_update or class_code not in self._snapshots:
            return
        layer = self._layer(class_code)
        before = self._snapshots[class_code]
        after_features = {feature.id(): feature for feature in layer.getFeatures()}
        changed = []
        deleted = sorted(set(before) - set(after_features))
        added = sorted(set(after_features) - set(before))
        for feature_id in sorted(set(before) & set(after_features)):
            feature = after_features[feature_id]
            if class_workspace.geometry_hash(feature.geometry()) != before[feature_id]["geometry_hash"]:
                changed.append(feature_id)
        if not changed and not deleted and not added:
            if layer.isEditable():
                self._snapshots[class_code] = self._snapshot(layer)
            else:
                self._snapshots.pop(class_code, None)
            self._edit_context.pop(class_code, None)
            self._update_actions()
            return
        context = self._edit_context.pop(class_code, {})
        keep_editing = layer.isEditable()
        confidence_warnings = []
        self._metadata_update = True
        try:
            if not keep_editing and not layer.startEditing():
                raise RuntimeError("cannot start metadata update after geometry edit")
            for feature_id in changed:
                feature = after_features[feature_id]
                old = before[feature_id]
                confidence_mean, confidence_std, confidence_warning = (
                    self._optional_confidence_statistics(
                        layer, feature.geometry()
                    )
                )
                if confidence_warning:
                    confidence_warnings.append((old["object_id"], confidence_warning))
                for name, value in old["immutable"].items():
                    layer.changeAttributeValue(feature_id, layer.fields().indexOf(name), value)
                self._set_attributes(layer, feature_id, {
                    "confidence_mean": confidence_mean,
                    "confidence_std": confidence_std,
                })
                if not context.get("metadata_prepared"):
                    self._set_attributes(layer, feature_id, {
                        "geometry_source": "manual_edited",
                        "edit_base": old["geometry_source"],
                        "geometry_revision": old["geometry_revision"] + 1,
                        "updated_at": class_workspace._now(),
                    })
                class_workspace.append_history(
                    self._run_spec,
                    "geometry_modified",
                    class_code=class_code,
                    object_id=old["object_id"],
                    before_geometry_hash=old["geometry_hash"],
                    after_geometry_hash=class_workspace.geometry_hash(feature.geometry()),
                )
            for feature_id in added:
                feature = after_features[feature_id]
                object_id = str(feature.attribute("object_id") or class_workspace.new_object_id(self._run_spec))
                confidence_mean, confidence_std, confidence_warning = (
                    self._optional_confidence_statistics(
                        layer, feature.geometry()
                    )
                )
                if confidence_warning:
                    confidence_warnings.append((object_id, confidence_warning))
                values = {
                    "confidence_mean": confidence_mean,
                    "confidence_std": confidence_std,
                    "updated_at": class_workspace._now(),
                }
                if not context.get("metadata_prepared"):
                    values.update({
                        "run_id": self._run_spec["run_id"],
                        "object_id": object_id,
                        "part_id": "000",
                        "class_code": class_code,
                        "class_name": CLASS_NAMES[class_code],
                        "baseline_stream_id": self._workspace["baseline_stream_id"],
                        "geometry_source": "manual_edited",
                        "geometry_revision": 1,
                        "edit_base": "",
                        "reviewed": 0,
                    })
                self._set_attributes(layer, feature_id, values)
                class_workspace.append_history(
                    self._run_spec, "feature_added", class_code=class_code,
                    object_id=object_id,
                    after_geometry_hash=class_workspace.geometry_hash(feature.geometry()),
                )
            for feature_id in deleted:
                old = before[feature_id]
                class_workspace.append_history(
                    self._run_spec, "feature_deleted", class_code=class_code,
                    object_id=old["object_id"], before_geometry_hash=old["geometry_hash"],
                )
            if not layer.commitChanges(not keep_editing):
                errors = "; ".join(layer.commitErrors())
                layer.rollBack()
                raise RuntimeError(f"cannot save edit metadata: {errors}")
        finally:
            self._metadata_update = False
        if layer.isEditable():
            self._snapshots[class_code] = self._snapshot(layer)
        else:
            self._snapshots.pop(class_code, None)
        for object_id, warning in confidence_warnings:
            class_workspace.append_history(
                self._run_spec,
                "confidence_statistics_unavailable",
                class_code=class_code,
                object_id=object_id,
                reason=warning,
            )
        hints = []
        for feature_id in changed + added:
            feature = after_features[feature_id]
            hints.append(
                self._local_topology_hint(class_code, feature.geometry(), feature_id)
            )
        if hints:
            self.baseline_label.setText(
                f"类别 {class_code} 已保存；局部拓扑提示: " + "; ".join(hints)
            )
        self._mark_class_modified(class_code)
        self._refresh_class_display(class_code)

    def _set_attributes(self, layer, feature_id, values):
        for name, value in values.items():
            index = layer.fields().indexOf(name)
            if index >= 0:
                layer.changeAttributeValue(feature_id, index, value)

    @staticmethod
    def _feature_by_object_id(layer, object_id):
        wanted = str(object_id or "")
        for feature in layer.getFeatures():
            if str(feature.attribute("object_id") or "") == wanted:
                return feature
        raise RuntimeError(f"cannot reload persisted object: {wanted}")

    def _confidence_statistics(self, layer, geometry):
        stream = class_workspace.stream_by_id(
            self._eligible_fusions, self._workspace["baseline_stream_id"]
        )
        confidence_path = str((stream.get("paths") or {}).get("confidence_mosaic") or "")
        raster = self._confidence_raster
        if raster is None or raster.source() != confidence_path:
            raster = QgsRasterLayer(confidence_path, "fusion_confidence_statistics")
            if not raster.isValid():
                raise RuntimeError(f"无法打开 Fusion confidence mosaic: {confidence_path}")
            self._confidence_raster = raster
        raster_geometry = QgsGeometry(geometry)
        if layer.crs() != raster.crs():
            raster_geometry.transform(
                QgsCoordinateTransform(layer.crs(), raster.crs(), QgsProject.instance())
            )
        wanted = Qgis.ZonalStatistic.Mean | Qgis.ZonalStatistic.StDev
        statistics = QgsZonalStatistics.calculateStatistics(
            raster.dataProvider(),
            raster_geometry,
            abs(raster.rasterUnitsPerPixelX()),
            abs(raster.rasterUnitsPerPixelY()),
            1,
            wanted,
        )
        mean = statistics.get(Qgis.ZonalStatistic.Mean)
        std = statistics.get(Qgis.ZonalStatistic.StDev)
        if mean is None or std is None:
            raise RuntimeError("新几何范围内没有可用于 confidence 统计的像元")
        return float(mean), float(std)

    def _optional_confidence_statistics(self, layer, geometry):
        try:
            mean, std = self._confidence_statistics(layer, geometry)
            return mean, std, ""
        except RuntimeError as exc:
            return None, None, str(exc)

    def _set_class_modified(self, class_code):
        record = self._workspace["classes"][str(class_code)]
        record["modified"] = True
        record["confirmed"] = False
        record["state"] = "editing" if self._layer(class_code).featureCount() else "unreviewed_empty"
        self._confirm_buttons[class_code].blockSignals(True)
        self._confirm_buttons[class_code].setChecked(False)
        self._confirm_buttons[class_code].blockSignals(False)

    def _mark_class_modified(self, class_code):
        self._set_class_modified(class_code)
        self._workspace = class_workspace.save_workspace(self._run_spec, self._workspace)
        self._refresh_table()

    def _begin_quick_redraw(self, class_code):
        if not self._workspace:
            return
        if self._active_session or self._active_redraw:
            QMessageBox.warning(self, "快速重画", "请先完成或取消当前 SAM3/快速重画会话")
            return
        modified = self._editable_modified_layers()
        if modified and class_code not in modified:
            QMessageBox.warning(
                self,
                "快速重画",
                "请先保存或取消其他类别工作层的编辑",
            )
            return
        layer = self._layer(class_code)
        selected = layer.selectedFeatures()
        if len(selected) > 1:
            QMessageBox.information(self, "快速重画", "当前类别只能重画一个面")
            return
        canvas = self.iface.mapCanvas()
        self._redraw_previous_map_tool = canvas.mapTool()
        self._active_redraw = {
            "class_code": class_code,
            "state": "selecting",
        }
        self.iface.setActiveLayer(layer)
        self._visibility_checks[class_code].setChecked(True)
        self.redraw_group.show()
        self.redraw_smooth_spin.blockSignals(True)
        self.redraw_smooth_spin.setValue(1)
        self.redraw_smooth_spin.blockSignals(False)
        if selected:
            self._start_quick_redraw_for_feature(selected[0])
            return
        self._redraw_pick_tool = QgsMapToolEmitPoint(canvas)
        self._redraw_pick_tool.canvasClicked.connect(self._quick_redraw_map_clicked)
        canvas.setMapTool(self._redraw_pick_tool)
        self.redraw_label.setText(
            f"点击地图中的 {class_code} {CLASS_NAMES[class_code]} 面进行完整重画"
        )
        self._set_redraw_controls("selecting")
        self._update_actions()

    def _quick_redraw_map_clicked(self, map_point, _button):
        session = self._active_redraw
        if not session or session.get("state") != "selecting":
            return
        layer = self._layer(session["class_code"])
        hits = self._features_at_map_point(layer, map_point)
        if len(hits) != 1:
            QMessageBox.information(
                self,
                "快速重画",
                "点击位置必须唯一命中当前类别中的一个面，请重新点击",
            )
            return
        self._disconnect_redraw_picker()
        self._start_quick_redraw_for_feature(hits[0])

    def _disconnect_redraw_picker(self, restore=False):
        tool = self._redraw_pick_tool
        if tool is None:
            return
        canvas = self.iface.mapCanvas()
        was_current = canvas.mapTool() == tool
        try:
            tool.canvasClicked.disconnect(self._quick_redraw_map_clicked)
        except (TypeError, RuntimeError):
            pass
        self._redraw_pick_tool = None
        if restore:
            if self._redraw_previous_map_tool is not None:
                self._restore_redraw_map_tool()
            elif was_current:
                canvas.unsetMapTool(tool)

    def _start_quick_redraw_for_feature(self, feature):
        session = self._active_redraw
        if not session:
            return
        geometry = QgsGeometry(feature.geometry())
        error = self._quick_redraw_geometry_error(geometry)
        if error:
            QMessageBox.warning(self, "快速重画", f"原面不可重画: {error}")
            self._cancel_quick_redraw()
            return
        class_code = session["class_code"]
        layer = self._layer(class_code)
        session.update({
            "feature_id": feature.id(),
            "object_id": str(feature.attribute("object_id") or ""),
            "original_geometry": geometry,
            "state": "capturing",
        })
        layer.removeSelection()
        self._show_redraw_reference(geometry, layer)
        self._start_redraw_capture()

    def _start_redraw_capture(self):
        session = self._active_redraw
        if not session:
            return
        self._disconnect_redraw_capture(restore=False)
        self._clear_redraw_candidate_band()
        session.pop("raw_geometry", None)
        session.pop("candidate_geometry", None)
        session.pop("candidate_error", None)
        session["state"] = "capturing"
        layer = self._layer(session["class_code"])
        canvas = self.iface.mapCanvas()
        tool = QgsMapToolDigitizeFeature(
            canvas,
            self.iface.cadDockWidget(),
            QgsMapToolCapture.CaptureMode.CapturePolygon,
        )
        tool.setLayer(layer)
        tool.setCheckGeometryType(True)
        if not tool.supportsTechnique(Qgis.CaptureTechnique.PolyBezier):
            QMessageBox.warning(self, "快速重画", "当前 QGIS 不支持 PolyBezier 捕获")
            self._cancel_quick_redraw()
            return
        tool.setCurrentCaptureTechnique(Qgis.CaptureTechnique.PolyBezier)
        tool.digitizingCompleted.connect(self._redraw_capture_completed)
        tool.digitizingCanceled.connect(self._redraw_capture_cancelled)
        self._redraw_capture_tool = tool
        self.iface.setActiveLayer(layer)
        canvas.setMapTool(tool)
        self.redraw_label.setText(
            f"正在重画 {session.get('object_id') or session['feature_id']}："
            "使用 Bézier 绘制完整面，右键结束"
        )
        self._set_redraw_controls("capturing")
        self._update_actions()

    def _disconnect_redraw_capture(self, restore=True):
        tool = self._redraw_capture_tool
        if tool is None:
            if restore:
                self._restore_redraw_map_tool()
            return
        canvas = self.iface.mapCanvas()
        was_current = canvas.mapTool() == tool
        for signal, slot in (
            (tool.digitizingCompleted, self._redraw_capture_completed),
            (tool.digitizingCanceled, self._redraw_capture_cancelled),
        ):
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        try:
            tool.stopCapturing()
        except RuntimeError:
            pass
        self._redraw_capture_tool = None
        if restore:
            if self._redraw_previous_map_tool is not None:
                self._restore_redraw_map_tool()
            elif was_current:
                canvas.unsetMapTool(tool)

    def _restore_redraw_map_tool(self):
        canvas = self.iface.mapCanvas()
        previous = self._redraw_previous_map_tool
        if previous is not None and canvas.mapTool() != previous:
            canvas.setMapTool(previous)

    def _redraw_capture_completed(self, feature):
        session = self._active_redraw
        if not session or session.get("state") != "capturing":
            return
        geometry = QgsGeometry(feature.geometry())
        if geometry.requiresConversionToStraightSegments():
            geometry.convertToStraightSegment()
        geometry.convertToMultiType()
        error = self._quick_redraw_geometry_error(geometry)
        self._disconnect_redraw_capture(restore=True)
        if error:
            session["state"] = "capture_failed"
            session["candidate_error"] = error
            self.redraw_label.setText(f"重画结果不可采用: {error}")
            self._set_redraw_controls("capture_failed")
            return
        session["raw_geometry"] = geometry
        session["state"] = "candidate"
        self._update_redraw_preview()

    def _redraw_capture_cancelled(self):
        session = self._active_redraw
        if not session:
            return
        self._disconnect_redraw_capture(restore=True)
        session["state"] = "capture_cancelled"
        self.redraw_label.setText("本次绘制已停止；可以重新绘制或取消快速重画")
        self._set_redraw_controls("capture_cancelled")

    @staticmethod
    def _quick_redraw_geometry_error(geometry):
        if geometry is None or geometry.isNull() or geometry.isEmpty():
            return "几何为空"
        if geometry.type() != Qgis.GeometryType.Polygon:
            return "结果不是 Polygon"
        if geometry.area() <= 0:
            return "面积必须大于 0"
        if not geometry.isGeosValid():
            return "几何无效或存在自相交"
        return ""

    def _update_redraw_preview(self, *_args):
        session = self._active_redraw
        if not session or "raw_geometry" not in session:
            return
        iterations = int(self.redraw_smooth_spin.value())
        raw = QgsGeometry(session["raw_geometry"])
        geometry = (
            raw if iterations == 0
            else raw.smooth(iterations, 0.25, -1.0, 180.0)
        )
        geometry.convertToMultiType()
        error = self._quick_redraw_geometry_error(geometry)
        session["candidate_geometry"] = geometry
        session["candidate_error"] = error
        session["state"] = "candidate"
        self._show_redraw_candidate(geometry, self._layer(session["class_code"]), error)
        if error:
            self.redraw_label.setText(
                f"平滑 {iterations} 次后的候选不可采用: {error}"
            )
        else:
            self.redraw_label.setText(
                f"候选已生成 | object_id: {session.get('object_id') or '-'} | "
                f"平滑次数: {iterations} | 采用后仍需“保存当前编辑”"
            )
        self._set_redraw_controls("candidate")

    def _set_redraw_controls(self, state):
        candidate = state == "candidate"
        self.redraw_smooth_spin.setEnabled(candidate)
        self.redraw_adopt_btn.setEnabled(
            candidate
            and bool(self._active_redraw)
            and not bool(self._active_redraw.get("candidate_error"))
        )
        self.redraw_retry_btn.setEnabled(
            state in ("candidate", "capture_failed", "capture_cancelled")
        )
        self.redraw_cancel_btn.setEnabled(bool(self._active_redraw))

    def _show_redraw_reference(self, geometry, layer):
        self._clear_redraw_bands()
        self._redraw_reference_band = QgsRubberBand(
            self.iface.mapCanvas(), Qgis.GeometryType.Polygon
        )
        self._redraw_reference_band.setStrokeColor(QColor(105, 105, 105, 230))
        self._redraw_reference_band.setFillColor(QColor(105, 105, 105, 20))
        self._redraw_reference_band.setLineStyle(Qt.DashLine)
        self._redraw_reference_band.setWidth(2)
        self._redraw_reference_band.setToGeometry(geometry, layer)

    def _show_redraw_candidate(self, geometry, layer, error=""):
        self._clear_redraw_candidate_band()
        color = QColor("#d7191c" if error else StyleManager.get_class_color(
            self._active_redraw["class_code"]
        ))
        fill = QColor(color)
        fill.setAlpha(55)
        self._redraw_candidate_band = QgsRubberBand(
            self.iface.mapCanvas(), Qgis.GeometryType.Polygon
        )
        self._redraw_candidate_band.setStrokeColor(color)
        self._redraw_candidate_band.setFillColor(fill)
        self._redraw_candidate_band.setWidth(2)
        self._redraw_candidate_band.setToGeometry(geometry, layer)

    def _clear_redraw_candidate_band(self):
        band = self._redraw_candidate_band
        if band is not None:
            band.reset(Qgis.GeometryType.Polygon)
            self.iface.mapCanvas().scene().removeItem(band)
        self._redraw_candidate_band = None

    def _clear_redraw_bands(self):
        self._clear_redraw_candidate_band()
        band = self._redraw_reference_band
        if band is not None:
            band.reset(Qgis.GeometryType.Polygon)
            self.iface.mapCanvas().scene().removeItem(band)
        self._redraw_reference_band = None

    def _retry_quick_redraw(self):
        if not self._active_redraw:
            return
        self._start_redraw_capture()

    def _adopt_quick_redraw(self):
        session = self._active_redraw
        if not session or session.get("state") != "candidate":
            return
        geometry = QgsGeometry(session.get("candidate_geometry"))
        error = self._quick_redraw_geometry_error(geometry)
        if error:
            QMessageBox.warning(self, "快速重画", f"候选不可采用: {error}")
            return
        class_code = session["class_code"]
        layer = self._layer(class_code)
        feature_id = session["feature_id"]
        if not layer.isEditable() and not layer.startEditing():
            QMessageBox.warning(self, "快速重画", "无法启动当前类别工作层编辑")
            return
        if not layer.changeGeometry(feature_id, geometry):
            QMessageBox.warning(self, "快速重画", "无法把候选写入当前要素编辑缓冲区")
            return
        object_id = session.get("object_id") or str(feature_id)
        self._finish_quick_redraw_session()
        layer.removeSelection()
        layer.triggerRepaint()
        self.iface.mapCanvas().clearCache()
        self.iface.mapCanvas().refresh()
        self.baseline_label.setText(
            f"{class_code} {CLASS_NAMES[class_code]} | {object_id} 已采用重画候选；"
            "请使用“保存当前编辑”提交，或“取消当前编辑”恢复原面"
        )
        self._refresh_table()

    def _cancel_quick_redraw(self):
        if not self._active_redraw and not any((
            self._redraw_pick_tool,
            self._redraw_capture_tool,
            self._redraw_reference_band,
            self._redraw_candidate_band,
        )):
            return
        self._finish_quick_redraw_session()
        self.baseline_label.setText("快速重画已取消；原面未修改")
        self._update_actions()

    def _finish_quick_redraw_session(self):
        self._disconnect_redraw_picker(restore=True)
        self._disconnect_redraw_capture(restore=True)
        self._restore_redraw_map_tool()
        self._clear_redraw_bands()
        self._active_redraw = None
        self._redraw_previous_map_tool = None
        self.redraw_group.hide()
        self.redraw_label.setText("无活动重画")
        self._set_redraw_controls("idle")

    def _edit_selected(self, class_code):
        if self._active_session or self._active_redraw:
            QMessageBox.warning(self, "编辑", "请先完成或取消当前 SAM3/快速重画会话")
            return
        layer = self._layer(class_code)
        if layer.selectedFeatureCount() == 0:
            self._begin_manual_pick(class_code)
            return
        if layer.selectedFeatureCount() != 1:
            QMessageBox.information(self, "编辑", "当前类别只能选择一个面进行编辑")
            return
        self.iface.setActiveLayer(layer)
        if not layer.isEditable() and not layer.startEditing():
            QMessageBox.warning(self, "编辑", "无法启动该类别工作层的 QGIS 编辑")
            return
        action = getattr(self.iface, "actionVertexTool", lambda: None)()
        if action is not None:
            action.trigger()
        self._update_actions()

    def _begin_manual_pick(self, class_code):
        if self._point_tool is not None:
            self._restore_map_tool()
        layer = self._layer(class_code)
        self.iface.setActiveLayer(layer)
        self._visibility_checks[class_code].setChecked(True)
        canvas = self.iface.mapCanvas()
        self._previous_map_tool = canvas.mapTool()
        self._point_tool = QgsMapToolEmitPoint(canvas)
        self._point_tool.canvasClicked.connect(self._manual_edit_map_clicked)
        self._manual_pick_code = class_code
        canvas.setMapTool(self._point_tool)
        self.baseline_label.setText(
            f"点击地图中的 {class_code} {CLASS_NAMES[class_code]} 面开始编辑"
        )

    def _manual_edit_map_clicked(self, map_point, _button):
        class_code = self._manual_pick_code
        if class_code is None:
            return
        layer = self._layer(class_code)
        hits = self._features_at_map_point(layer, map_point)
        self._restore_map_tool()
        if len(hits) != 1:
            QMessageBox.information(
                self,
                "节点精修",
                "点击位置必须唯一命中当前类别中的一个面",
            )
            return
        layer.selectByIds([hits[0].id()])
        self._edit_selected(class_code)

    def _save_class_edits(self, class_code):
        layer = self._layer(class_code)
        if not layer.isEditable() or not layer.isModified():
            QMessageBox.information(self, "保存当前编辑", "当前类别没有未保存编辑")
            return
        if not layer.commitChanges():
            errors = "; ".join(layer.commitErrors())
            QMessageBox.warning(self, "保存当前编辑失败", errors)
            return
        self.baseline_label.setText(
            f"类别 {class_code} {CLASS_NAMES[class_code]} 当前编辑已保存"
        )
        self._update_actions()

    def _rollback_class_edits(self, class_code):
        layer = self._layer(class_code)
        if not layer.isEditable():
            QMessageBox.information(self, "取消当前编辑", "当前类别没有编辑会话")
            return
        if layer.isModified():
            answer = QMessageBox.question(
                self,
                "取消当前编辑",
                "放弃当前类别尚未保存的修改？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        layer.rollBack()
        self.baseline_label.setText(
            f"类别 {class_code} {CLASS_NAMES[class_code]} 当前编辑已取消"
        )
        self._update_actions()

    def _add_manual_feature(self, class_code):
        if self._active_session or self._active_redraw:
            QMessageBox.warning(self, "新增人工面", "请先完成或取消当前 SAM3/快速重画会话")
            return
        modified = self._editable_modified_layers()
        if modified and class_code not in modified:
            QMessageBox.warning(
                self,
                "新增人工面",
                "请先保存或回滚其他类别工作层的编辑",
            )
            return
        layer = self._layer(class_code)
        self.iface.setActiveLayer(layer)
        self._visibility_checks[class_code].setChecked(True)
        class_workspace.apply_class_constraints(
            layer,
            class_code,
            run_id=self._run_spec["run_id"],
            baseline_stream_id=self._workspace["baseline_stream_id"],
        )
        if not layer.isEditable() and not layer.startEditing():
            QMessageBox.warning(self, "新增人工面", "无法启动该类别工作层的 QGIS 编辑")
            return
        action = getattr(self.iface, "actionAddFeature", lambda: None)()
        if action is None:
            QMessageBox.warning(self, "新增人工面", "QGIS 添加多边形工具不可用")
            return
        action.trigger()
        self.baseline_label.setText(
            f"正在新增 {class_code} {CLASS_NAMES[class_code]}：绘制后右键结束，"
            "插件会自动填写对象 ID 和类别"
        )
        self._update_actions()

    @staticmethod
    def _object_id_exists(layer, object_id):
        wanted = str(object_id or "")
        return any(
            str(feature.attribute("object_id") or "") == wanted
            for feature in layer.getFeatures()
        )

    def _reclassify_selected(self, source_code):
        if self._active_session or self._active_redraw:
            QMessageBox.warning(self, "更正类别", "请先完成或取消当前 SAM3/快速重画会话")
            return
        if self._editable_modified_layers():
            QMessageBox.warning(self, "更正类别", "请先保存或回滚所有类别工作层编辑")
            return
        source_layer = self._layer(source_code)
        if source_layer.selectedFeatureCount() != 1:
            QMessageBox.information(
                self,
                "更正类别",
                "请先在当前类别工作层中选择一个面",
            )
            return
        options = [
            f"{code} {CLASS_NAMES[code]}"
            for code in CLASS_ORDER
            if code != source_code
        ]
        selected, accepted = QInputDialog.getItem(
            self,
            "更正类别",
            f"将选中的 {source_code} {CLASS_NAMES[source_code]} 移动到：",
            options,
            0,
            False,
        )
        if not accepted:
            return
        target_code = int(str(selected).split(" ", 1)[0])
        try:
            self._move_feature_to_class(source_code, target_code)
        except Exception as exc:
            QMessageBox.warning(self, "更正类别失败", str(exc))

    def _remove_transferred_feature(self, layer, object_id):
        feature = self._feature_by_object_id(layer, object_id)
        if not layer.isEditable() and not layer.startEditing():
            raise RuntimeError("无法启动目标类别补偿回滚")
        if not layer.deleteFeature(feature.id()):
            layer.rollBack()
            raise RuntimeError("无法删除已写入目标类别的补偿对象")
        if not layer.commitChanges():
            errors = "; ".join(layer.commitErrors())
            layer.rollBack()
            raise RuntimeError(f"目标类别补偿回滚失败: {errors}")

    def _move_feature_to_class(self, source_code, target_code):
        if source_code == target_code:
            return
        source = self._layer(source_code)
        target = self._layer(target_code)
        selected = source.selectedFeatures()
        if len(selected) != 1:
            raise RuntimeError("来源类别必须唯一选中一个面")
        original = selected[0]
        object_id = str(original.attribute("object_id") or "")
        if not object_id:
            raise RuntimeError("选中面缺少 object_id，不能安全移动")
        if self._object_id_exists(target, object_id):
            raise RuntimeError(f"目标类别已存在 object_id: {object_id}")
        geometry = QgsGeometry(original.geometry())
        if geometry.isNull() or geometry.isEmpty():
            raise RuntimeError("选中面几何为空")
        geometry.convertToMultiType()
        moved = QgsFeature(target.fields())
        moved.setGeometry(geometry)
        for field in target.fields():
            source_index = original.fieldNameIndex(field.name())
            if source_index >= 0:
                moved.setAttribute(field.name(), original.attribute(source_index))
        previous_source = str(original.attribute("geometry_source") or "fusion")
        previous_revision = int(original.attribute("geometry_revision") or 0)
        values = {
            "run_id": self._run_spec["run_id"],
            "object_id": object_id,
            "part_id": str(original.attribute("part_id") or "000"),
            "class_code": target_code,
            "class_name": CLASS_NAMES[target_code],
            "baseline_stream_id": self._workspace["baseline_stream_id"],
            "geometry_source": "manual_edited",
            "geometry_revision": previous_revision + 1,
            "edit_base": previous_source,
            "reviewed": 0,
            "updated_at": class_workspace._now(),
        }
        for name, value in values.items():
            if target.fields().indexOf(name) >= 0:
                moved.setAttribute(name, value)

        target_committed = False
        self._metadata_update = True
        try:
            if not target.isEditable() and not target.startEditing():
                raise RuntimeError("无法启动目标类别工作层编辑")
            if not target.addFeature(moved):
                target.rollBack()
                raise RuntimeError("无法向目标类别工作层添加对象")
            if not target.commitChanges():
                errors = "; ".join(target.commitErrors())
                target.rollBack()
                raise RuntimeError(f"无法保存目标类别对象: {errors}")
            target_committed = True

            if not source.isEditable() and not source.startEditing():
                raise RuntimeError("无法启动来源类别工作层编辑")
            if not source.deleteFeature(original.id()):
                source.rollBack()
                raise RuntimeError("无法从来源类别工作层删除对象")
            if not source.commitChanges():
                errors = "; ".join(source.commitErrors())
                source.rollBack()
                raise RuntimeError(f"无法保存来源类别删除: {errors}")
        except Exception as exc:
            if source.isEditable():
                source.rollBack()
            if target.isEditable():
                target.rollBack()
            if target_committed:
                try:
                    self._remove_transferred_feature(target, object_id)
                except Exception as rollback_exc:
                    raise RuntimeError(
                        f"{exc}；且补偿回滚失败: {rollback_exc}"
                    ) from exc
            raise
        finally:
            self._metadata_update = False

        persisted = self._feature_by_object_id(target, object_id)
        topology_hint = self._local_topology_hint(
            target_code, persisted.geometry(), persisted.id()
        )
        geometry_hash = class_workspace.geometry_hash(persisted.geometry())
        class_workspace.append_history(
            self._run_spec,
            "feature_reclassified",
            object_id=object_id,
            part_id=str(persisted.attribute("part_id") or "000"),
            from_class_code=source_code,
            to_class_code=target_code,
            geometry_hash=geometry_hash,
            overlap_hint=topology_hint,
        )
        source.removeSelection()
        self._set_class_modified(source_code)
        self._set_class_modified(target_code)
        self._workspace = class_workspace.save_workspace(
            self._run_spec, self._workspace
        )
        self._visibility_checks[target_code].setChecked(True)
        self.iface.setActiveLayer(target)
        self._refresh_class_display(source_code, target_code)
        self.baseline_label.setText(
            f"对象已从 {source_code} {CLASS_NAMES[source_code]} 移动到 "
            f"{target_code} {CLASS_NAMES[target_code]}；局部重叠仅记录: {topology_hint}"
        )
        self._refresh_table()

    def _sam_available(self):
        return bool(
            self._sam_config.get("enabled")
            and self._sam_config.get("checkpoint_sha256")
            and Path(str(self._sam_config.get("checkpoint") or "")).is_file()
        )

    def _begin_sam(self, class_code, missed=False):
        if not self._workspace:
            return
        if not self._sam_available():
            QMessageBox.warning(self, "SAM3", "SAM3 checkpoint、SHA 或环境检查未通过")
            return
        if self._active_session or self._active_redraw:
            QMessageBox.warning(self, "SAM3", "已有活动 SAM3 或快速重画会话")
            return
        if self._editable_modified_layers():
            QMessageBox.warning(self, "SAM3", "请先保存或回滚所有类别工作层编辑")
            return
        layer = self._layer(class_code)
        if not missed and layer.featureCount() == 0:
            QMessageBox.information(self, "SAM3", "该类别为空；请使用“新增漏标面”")
            return
        canvas = self.iface.mapCanvas()
        self._previous_map_tool = canvas.mapTool()
        self._point_tool = QgsMapToolEmitPoint(canvas)
        self._point_tool.canvasClicked.connect(self._sam_map_clicked)
        self._active_session = {
            "session_id": uuid.uuid4().hex,
            "class_code": class_code,
            "mode": "missed" if missed else "existing",
            "state": "waiting_click",
            "started_at": class_workspace._now(),
        }
        self._workspace["active_sam_session_id"] = self._active_session["session_id"]
        self._workspace = class_workspace.save_workspace(self._run_spec, self._workspace)
        self.session_group.show()
        self.session_label.setText(
            f"当前类别: {class_code} {CLASS_NAMES[class_code]} | "
            + ("点击漏标地物位置" if missed else "点击当前类别工作层中的一个已有面")
        )
        self._set_decision_state("waiting_click")
        canvas.setMapTool(self._point_tool)
        self._update_actions()

    def _restore_map_tool(self):
        canvas = self.iface.mapCanvas()
        tool = self._point_tool
        if tool is not None and canvas.mapTool() == tool:
            if self._previous_map_tool is not None:
                canvas.setMapTool(self._previous_map_tool)
            else:
                canvas.unsetMapTool(tool)
        self._point_tool = None
        self._previous_map_tool = None
        self._manual_pick_code = None

    def _features_at_map_point(self, layer, map_point):
        canvas = self.iface.mapCanvas()
        canvas_crs = canvas.mapSettings().destinationCrs()
        to_layer = QgsCoordinateTransform(
            canvas_crs, layer.crs(), QgsProject.instance()
        )
        canvas_point = QgsPointXY(map_point)
        layer_point = to_layer.transform(canvas_point)
        point_geometry = QgsGeometry.fromPointXY(layer_point)
        canvas_tolerance = canvas.mapUnitsPerPixel() * 2
        offset_layer_point = to_layer.transform(QgsPointXY(
            canvas_point.x() + canvas_tolerance,
            canvas_point.y(),
        ))
        tolerance = max(
            1e-12,
            math.hypot(
                offset_layer_point.x() - layer_point.x(),
                offset_layer_point.y() - layer_point.y(),
            ),
        )
        request = QgsFeatureRequest().setFilterRect(QgsRectangle(
            layer_point.x() - tolerance,
            layer_point.y() - tolerance,
            layer_point.x() + tolerance,
            layer_point.y() + tolerance,
        ))
        return [
            feature for feature in layer.getFeatures(request)
            if feature.geometry().contains(point_geometry)
            or feature.geometry().intersects(point_geometry)
        ]

    def _sam_map_clicked(self, map_point, _button):
        session = self._active_session
        if not session or session.get("state") != "waiting_click":
            return
        class_code = session["class_code"]
        layer = self._layer(class_code)
        canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
        try:
            hits = self._features_at_map_point(layer, map_point)
            feature = None
            if session["mode"] == "existing":
                if len(hits) != 1:
                    raise RuntimeError(
                        "点击必须唯一命中当前类别工作层中的一个面；未命中或重叠时不按最近距离猜测"
                    )
                feature = hits[0]
                layer.selectByIds([feature.id()])
            raster_crs = QgsCoordinateReferenceSystem(self._run_spec["raster"]["crs"])
            to_raster = QgsCoordinateTransform(canvas_crs, raster_crs, QgsProject.instance())
            raster_point = to_raster.transform(QgsPointXY(map_point))
            bounds = None
            if feature is not None:
                geometry_raster = QgsGeometry(feature.geometry())
                geometry_raster.transform(
                    QgsCoordinateTransform(layer.crs(), raster_crs, QgsProject.instance())
                )
                rectangle = geometry_raster.boundingBox()
                bounds = {
                    "xmin": rectangle.xMinimum(), "ymin": rectangle.yMinimum(),
                    "xmax": rectangle.xMaximum(), "ymax": rectangle.yMaximum(),
                }
                session.update({
                    "feature_id": feature.id(),
                    "object_id": str(feature.attribute("object_id") or ""),
                    "part_id": str(feature.attribute("part_id") or "000"),
                    "current_geometry_hash": class_workspace.geometry_hash(feature.geometry()),
                    "current_source": str(feature.attribute("geometry_source") or "fusion"),
                    "current_revision": int(feature.attribute("geometry_revision") or 0),
                })
                self._show_current_geometry(feature.geometry(), layer)
            else:
                session.update({"object_id": "", "part_id": "000"})
            session.update({
                "state": "inference",
                "click_raster": {"x": raster_point.x(), "y": raster_point.y()},
                "geometry_bounds": bounds,
            })
            self._restore_map_tool()
            self._set_decision_state("inference")
            self.session_label.setText(
                f"当前类别: {class_code} {CLASS_NAMES[class_code]} | "
                f"object_id: {session.get('object_id') or '新增'} | "
                f"当前来源: {session.get('current_source') or '新增'} | SAM3 推理中"
            )
            self._start_worker_request()
        except Exception as exc:
            self._restore_map_tool()
            self._session_failed(str(exc))

    def _start_worker_request(self):
        session = self._active_session
        stream = class_workspace.stream_by_id(
            self._eligible_fusions, self._workspace["baseline_stream_id"]
        )
        self._worker_request = {
            "session_id": session["session_id"],
            "run_id": self._run_spec["run_id"],
            "raster": self._run_spec["raster"]["path"],
            "confidence_mosaic": (stream.get("paths") or {}).get("confidence_mosaic", ""),
            "click_raster": session["click_raster"],
            "geometry_bounds": session.get("geometry_bounds"),
            "crop_size_px": 512,
            "buffer_px": int(self._sam_config.get("buffer_px", 32)),
            "class_code": session["class_code"],
            "object_id": session.get("object_id", ""),
            "part_id": session.get("part_id", "000"),
            "checkpoint_sha256": str(self._sam_config.get("checkpoint_sha256") or ""),
            "sam_version": str(self._sam_config.get("version") or ""),
            "device": str(
                self._sam_config.get("effective_device")
                or self._sam_config.get("requested_device")
                or "cpu"
            ),
        }
        if self._worker is None:
            self._worker = Sam3WorkerRunner(
                self._scripts_dir, self._sam_config, self
            )
            self._worker.ready.connect(self._worker_ready)
            self._worker.event_received.connect(self._worker_event)
            self._worker.log_line.connect(self._worker_log)
            self._worker.stopped.connect(self._worker_stopped)
        if self._worker.is_ready:
            self._worker.predict(self._worker_request)
            self._worker_request = None
        else:
            self._worker.start()

    def _worker_ready(self, _event):
        if self._worker_request and self._active_session:
            request = self._worker_request
            self._worker_request = None
            self._worker.predict(request)

    def _worker_event(self, event):
        session = self._active_session
        if not session:
            return
        if event.get("session_id") and event.get("session_id") != session["session_id"]:
            return
        if event.get("event") == "candidate_ready":
            geometry = QgsGeometry.fromWkt(str(event.get("geometry_wkt") or ""))
            if geometry.isNull() or geometry.isEmpty() or not geometry.isGeosValid():
                self._session_failed("SAM3 返回了无效候选几何")
                return
            raster_crs = QgsCoordinateReferenceSystem(self._run_spec["raster"]["crs"])
            layer = self._layer(session["class_code"])
            geometry.transform(
                QgsCoordinateTransform(raster_crs, layer.crs(), QgsProject.instance())
            )
            session.update({
                "state": "candidate",
                "candidate_geometry": geometry,
                "candidate_score": float(event.get("score") or 0.0),
                "confidence_mean": float(event.get("confidence_mean") or 0.0),
                "confidence_std": float(event.get("confidence_std") or 0.0),
                "crop_window": event.get("crop_window"),
                "elapsed_sec": event.get("elapsed_sec"),
            })
            topology_hint = self._local_topology_hint(
                session["class_code"], geometry, session.get("feature_id")
            )
            session["topology_hint"] = topology_hint
            self._show_candidate_geometry(geometry, layer)
            self.session_label.setText(
                f"当前类别: {session['class_code']} {CLASS_NAMES[session['class_code']]} | "
                f"object_id: {session.get('object_id') or '新增'} | "
                f"当前来源: {session.get('current_source') or '新增'} | "
                f"候选 score: {session['candidate_score']:.4f}"
            )
            self.local_topology_label.setText(f"局部拓扑提示: {topology_hint}")
            self._set_decision_state("candidate")
        elif event.get("event") == "failed":
            self._session_failed(str(event.get("error") or "SAM3 未知错误"))

    def _worker_log(self, level, message):
        if level == "stderr" and self._active_session:
            self.session_error.setPlainText(message[-6000:])

    def _worker_stopped(self, event):
        if not event.get("expected") and self._active_session:
            self._session_failed(
                f"SAM3 worker 意外退出，returncode={event.get('returncode')}"
            )

    def _session_failed(self, error):
        if not self._active_session:
            return
        self._active_session["state"] = "failed"
        self._active_session["error"] = error
        self.session_label.setText("SAM3 失败；当前工作层未修改")
        self.local_topology_label.setText("局部拓扑提示: 未执行")
        self.session_error.setPlainText(error)
        self.session_error.show()
        self._set_decision_state("failed")

    def _show_current_geometry(self, geometry, layer):
        self._clear_bands()
        self._current_band = QgsRubberBand(
            self.iface.mapCanvas(), Qgis.GeometryType.Polygon
        )
        self._current_band.setStrokeColor(QColor("#ffd400"))
        self._current_band.setFillColor(QColor(255, 212, 0, 30))
        self._current_band.setWidth(2)
        self._current_band.setToGeometry(geometry, layer)

    def _show_candidate_geometry(self, geometry, layer):
        self._candidate_band = QgsRubberBand(
            self.iface.mapCanvas(), Qgis.GeometryType.Polygon
        )
        self._candidate_band.setStrokeColor(QColor("#00d7d7"))
        self._candidate_band.setFillColor(QColor(0, 215, 215, 35))
        self._candidate_band.setWidth(2)
        self._candidate_band.setLineStyle(Qt.DashLine)
        self._candidate_band.setToGeometry(geometry, layer)

    def _local_topology_hint(self, class_code, geometry, replaced_feature_id=None):
        if geometry is None or geometry.isNull() or geometry.isEmpty():
            return "候选为空"
        if not geometry.isGeosValid():
            return "候选几何无效"
        tolerance = topology_validator.pixel_area_tolerance(self._run_spec)
        same_class = 0
        cross_class = 0
        overlap_area = 0.0
        source_layer = self._layer(class_code)
        for other_code in CLASS_ORDER:
            layer = self._layer(other_code)
            candidate = QgsGeometry(geometry)
            if source_layer.crs() != layer.crs():
                candidate.transform(
                    QgsCoordinateTransform(
                        source_layer.crs(), layer.crs(), QgsProject.instance()
                    )
                )
            request = QgsFeatureRequest().setFilterRect(candidate.boundingBox())
            for feature in layer.getFeatures(request):
                if other_code == class_code and feature.id() == replaced_feature_id:
                    continue
                intersection = candidate.intersection(feature.geometry())
                if (
                    intersection is None
                    or intersection.isNull()
                    or intersection.isEmpty()
                    or intersection.area() <= tolerance
                ):
                    continue
                overlap_area += intersection.area()
                if other_code == class_code:
                    same_class += 1
                else:
                    cross_class += 1
        if not same_class and not cross_class:
            return "无"
        return (
            f"同类重叠 {same_class} 处，邻类重叠 {cross_class} 处，"
            f"总面积 {overlap_area:.8g}"
        )

    def _clear_bands(self):
        scene = self.iface.mapCanvas().scene()
        for band in (self._current_band, self._candidate_band):
            if band is not None:
                band.reset(Qgis.GeometryType.Polygon)
                scene.removeItem(band)
        self._current_band = None
        self._candidate_band = None

    def _set_decision_state(self, state):
        candidate = state == "candidate"
        failed = state == "failed"
        existing = bool(
            self._active_session
            and self._active_session.get("mode") == "existing"
        )
        self.keep_btn.setEnabled(candidate and existing)
        self.adopt_btn.setEnabled(candidate)
        self.edit_current_btn.setEnabled((candidate or failed) and existing)
        self.edit_sam_btn.setEnabled(candidate)
        self.retry_btn.setEnabled(failed)
        self.cancel_session_btn.setEnabled(state != "idle")

    def _retry_session(self):
        session = self._active_session
        if not session or session.get("state") != "failed" or not session.get("click_raster"):
            return
        if self._worker:
            old_session_id = session["session_id"]
            self._worker.cancel(old_session_id)
            self._worker.close_session(old_session_id)
        self._record_session("failed")
        session["session_id"] = uuid.uuid4().hex
        session["state"] = "inference"
        session["started_at"] = class_workspace._now()
        session.pop("error", None)
        self._workspace["active_sam_session_id"] = session["session_id"]
        self._workspace = class_workspace.save_workspace(self._run_spec, self._workspace)
        self.session_error.hide()
        self.session_error.clear()
        self._set_decision_state("inference")
        self.session_label.setText("SAM3 正在重试；当前工作层未修改")
        self._start_worker_request()

    def _finish_session(self, decision):
        session = self._active_session
        if not session:
            return
        if decision in ("adopted", "edit_sam3") and session.get("state") != "candidate":
            return
        if decision == "kept_current" and session.get("state") != "candidate":
            return
        try:
            if decision == "adopted":
                self._adopt_candidate(session, edit=False)
            elif decision == "edit_sam3":
                self._adopt_candidate(session, edit=True)
            elif decision == "edit_current":
                if session["mode"] == "missed":
                    raise RuntimeError("新增漏标面没有当前几何可编辑")
                self._start_session_edit_current(session)
            if self._worker and decision == "cancelled":
                self._worker.cancel(session["session_id"])
            self._record_session(decision)
        except Exception as exc:
            QMessageBox.warning(self, "SAM3 决定失败", str(exc))
            return
        self._cancel_active_session(record=False, keep_edit=decision in ("edit_current", "edit_sam3"))
        self._refresh_table()

    def _adopt_candidate(self, session, edit):
        layer = self._layer(session["class_code"])
        self.iface.setActiveLayer(layer)
        before_snapshot = self._snapshot(layer)
        confidence_mean, confidence_std, confidence_warning = (
            self._optional_confidence_statistics(
                layer, session["candidate_geometry"]
            )
        )
        self._metadata_update = True
        try:
            if not layer.isEditable() and not layer.startEditing():
                raise RuntimeError("无法启动类别工作层编辑")
            if session["mode"] == "existing":
                feature_id = session["feature_id"]
                layer.changeGeometry(feature_id, session["candidate_geometry"])
                values = {
                    "geometry_source": "manual_edited" if edit else "sam3",
                    "geometry_revision": session["current_revision"] + 1,
                    "edit_base": "sam3" if edit else "",
                    "sam_session_id": session["session_id"],
                    "sam_score": session["candidate_score"],
                    "sam_version": str(self._sam_config.get("version") or ""),
                    "confidence_mean": confidence_mean,
                    "confidence_std": confidence_std,
                    "reviewed": 0,
                    "updated_at": class_workspace._now(),
                }
                self._set_attributes(layer, feature_id, values)
            else:
                feature = QgsFeature(layer.fields())
                feature.setGeometry(session["candidate_geometry"])
                object_id = class_workspace.new_object_id(self._run_spec)
                values = {
                    "run_id": self._run_spec["run_id"],
                    "object_id": object_id,
                    "part_id": "000",
                    "class_code": session["class_code"],
                    "class_name": CLASS_NAMES[session["class_code"]],
                    "baseline_stream_id": self._workspace["baseline_stream_id"],
                    "geometry_source": "manual_edited" if edit else "sam3",
                    "geometry_revision": 1,
                    "edit_base": "sam3" if edit else "",
                    "sam_session_id": session["session_id"],
                    "sam_score": session["candidate_score"],
                    "sam_version": str(self._sam_config.get("version") or ""),
                    "confidence_mean": confidence_mean,
                    "confidence_std": confidence_std,
                    "reviewed": 0,
                    "updated_at": class_workspace._now(),
                }
                for name, value in values.items():
                    if layer.fields().indexOf(name) >= 0:
                        feature.setAttribute(name, value)
                if not layer.addFeature(feature):
                    raise RuntimeError("无法向类别工作层新增 SAM3 面")
                feature_id = feature.id()
                session["object_id"] = object_id
                session["feature_id"] = feature_id
            if edit:
                self._metadata_update = False
                self._snapshots[session["class_code"]] = before_snapshot
                self._edit_context[session["class_code"]] = {
                    "metadata_prepared": True,
                    "session_id": session["session_id"],
                }
                action = getattr(self.iface, "actionVertexTool", lambda: None)()
                if action is not None:
                    action.trigger()
                return
            if not layer.commitChanges():
                errors = "; ".join(layer.commitErrors())
                layer.rollBack()
                raise RuntimeError(f"无法保存 SAM3 候选: {errors}")
        finally:
            self._metadata_update = False
        persisted = self._feature_by_object_id(layer, session["object_id"])
        session["feature_id"] = persisted.id()
        session["persisted_geometry_hash"] = class_workspace.geometry_hash(
            persisted.geometry()
        )
        self._mark_class_modified(session["class_code"])
        if confidence_warning:
            class_workspace.append_history(
                self._run_spec,
                "confidence_statistics_unavailable",
                class_code=session["class_code"],
                object_id=session["object_id"],
                reason=confidence_warning,
            )
        class_workspace.append_history(
            self._run_spec,
            "sam3_adopted" if session["mode"] == "existing" else "sam3_feature_added",
            class_code=session["class_code"],
            object_id=session["object_id"],
            sam_session_id=session["session_id"],
            before_geometry_hash=session.get("current_geometry_hash", ""),
            after_geometry_hash=session["persisted_geometry_hash"],
        )
        self.baseline_label.setText(
            f"SAM3 已采用；局部拓扑提示: {session.get('topology_hint') or '无'}"
        )

    def _start_session_edit_current(self, session):
        layer = self._layer(session["class_code"])
        self.iface.setActiveLayer(layer)
        if not layer.isEditable() and not layer.startEditing():
            raise RuntimeError("无法启动当前工作层编辑")
        layer.selectByIds([session["feature_id"]])
        self._edit_context[session["class_code"]] = {
            "metadata_prepared": False,
            "session_id": session["session_id"],
        }
        action = getattr(self.iface, "actionVertexTool", lambda: None)()
        if action is not None:
            action.trigger()

    def _record_session(self, decision):
        session = dict(self._active_session or {})
        candidate_geometry = session.pop("candidate_geometry", None)
        before_hash = str(session.get("current_geometry_hash") or "")
        candidate_hash = (
            class_workspace.geometry_hash(candidate_geometry)
            if candidate_geometry is not None else ""
        )
        persisted_hash = str(session.get("persisted_geometry_hash") or "")
        session.update({
            "decision": decision,
            "run_id": self._run_spec.get("run_id"),
            "baseline_stream_id": (self._workspace or {}).get("baseline_stream_id", ""),
            "before_geometry_hash": before_hash,
            "candidate_geometry_hash": candidate_hash,
            "after_geometry_hash": (
                persisted_hash if decision == "adopted"
                else candidate_hash if decision == "edit_sam3"
                else before_hash
            ),
            "checkpoint_sha256": str(self._sam_config.get("checkpoint_sha256") or ""),
            "sam_version": str(self._sam_config.get("version") or ""),
            "device": str(
                self._sam_config.get("effective_device")
                or self._sam_config.get("requested_device")
                or "cpu"
            ),
        })
        class_workspace.append_sam_session(self._run_spec, session)

    def _cancel_active_session(self, record=True, keep_edit=False):
        if self._active_session and record:
            self._record_session("cancelled")
        if self._active_session and self._worker:
            session_id = self._active_session["session_id"]
            self._worker.cancel(session_id)
            self._worker.close_session(session_id)
        self._restore_map_tool()
        self._clear_bands()
        self._active_session = None
        self._worker_request = None
        if self._workspace is not None:
            self._workspace["active_sam_session_id"] = ""
            self._workspace = class_workspace.save_workspace(
                self._run_spec, self._workspace
            )
        self.session_group.hide()
        self.local_topology_label.setText("局部拓扑提示: -")
        self.session_error.hide()
        self.session_error.clear()
        self._set_decision_state("idle")
        self._update_actions()

    def _confirm_class(self, class_code, checked):
        if self._metadata_update or not self._workspace:
            return
        layer = self._layer(class_code)
        if checked and (
            self._active_session
            or self._active_redraw
            or (layer.isEditable() and layer.isModified())
        ):
            self._confirm_buttons[class_code].blockSignals(True)
            self._confirm_buttons[class_code].setChecked(False)
            self._confirm_buttons[class_code].blockSignals(False)
            QMessageBox.warning(
                self,
                "确认类别",
                "存在活动 SAM3、快速重画或未保存编辑，不能确认",
            )
            return
        record = self._workspace["classes"][str(class_code)]
        keep_editing = layer.isEditable()
        self._metadata_update = True
        try:
            if not keep_editing and not layer.startEditing():
                raise RuntimeError("无法更新 reviewed 字段")
            reviewed_index = layer.fields().indexOf("reviewed")
            updated_index = layer.fields().indexOf("updated_at")
            for feature in layer.getFeatures():
                layer.changeAttributeValue(feature.id(), reviewed_index, 1 if checked else 0)
                layer.changeAttributeValue(feature.id(), updated_index, class_workspace._now())
            if not layer.commitChanges(not keep_editing):
                errors = "; ".join(layer.commitErrors())
                layer.rollBack()
                raise RuntimeError(errors)
        except Exception as exc:
            QMessageBox.warning(self, "确认类别失败", str(exc))
            return
        finally:
            self._metadata_update = False
        record["confirmed"] = bool(checked)
        record["state"] = (
            "confirmed_empty" if checked and layer.featureCount() == 0
            else "confirmed" if checked
            else "editing" if layer.featureCount()
            else "unreviewed_empty"
        )
        class_workspace.append_history(
            self._run_spec, "class_confirmed" if checked else "class_reopened",
            class_code=class_code, feature_count=layer.featureCount(),
        )
        self._workspace = class_workspace.save_workspace(self._run_spec, self._workspace)
        self._refresh_table()

    def _refresh_table(self):
        if not self._workspace:
            for row, code in enumerate(CLASS_ORDER):
                self.table.item(row, 3).setText("未初始化")
                self.table.item(row, 4).setText("0")
                self._sam_buttons[code].setEnabled(False)
                self._edit_buttons[code].setEnabled(False)
                self._confirm_buttons[code].setEnabled(False)
            self._update_actions()
            return
        has_unsaved_edits = bool(self._editable_modified_layers())
        for row, code in enumerate(CLASS_ORDER):
            record = self._workspace["classes"][str(code)]
            layer = self._layer(code)
            stats = class_workspace.source_statistics(layer)
            if self._manual_only:
                status_text = (
                    f"原始 {stats.get('fusion', 0)} / "
                    f"人工 {stats.get('manual_edited', 0)}"
                )
            else:
                status_text = (
                    f"Fusion {stats.get('fusion', 0)} / SAM3 {stats.get('sam3', 0)} / "
                    f"人工 {stats.get('manual_edited', 0)}"
                )
            self.table.item(row, 3).setText(status_text)
            self.table.item(row, 4).setText(str(layer.featureCount()))
            sam_enabled = bool(
                not self._manual_only
                and self._sam_available()
                and not self._active_session
                and not self._active_redraw
            )
            self._sam_buttons[code].setEnabled(sam_enabled)
            self._sam_existing_actions[code].setEnabled(
                sam_enabled and layer.featureCount() > 0
            )
            self._sam_missed_actions[code].setEnabled(sam_enabled)
            self._sam_buttons[code].setToolTip(
                "" if sam_enabled else "SAM3 不可用、会话活动或环境未通过"
            )
            manual_enabled = not self._active_session and not self._active_redraw
            self._edit_buttons[code].setEnabled(manual_enabled)
            self._quick_redraw_actions[code].setEnabled(
                manual_enabled and layer.featureCount() > 0
            )
            self._edit_selected_actions[code].setEnabled(
                manual_enabled and layer.featureCount() > 0
            )
            self._add_manual_actions[code].setEnabled(manual_enabled)
            self._reclassify_actions[code].setEnabled(
                manual_enabled and layer.featureCount() > 0
            )
            self._save_edit_actions[code].setEnabled(
                manual_enabled and layer.isEditable()
            )
            self._rollback_edit_actions[code].setEnabled(
                manual_enabled and layer.isEditable()
            )
            confirm = self._confirm_buttons[code]
            confirm.blockSignals(True)
            confirm.setChecked(bool(record.get("confirmed")))
            confirm.setText(
                "整类已确认" if record.get("confirmed")
                else "确认本范围无该类" if layer.featureCount() == 0
                else "确认整类"
            )
            confirm.blockSignals(False)
            confirm.setEnabled(
                not self._active_session
                and not self._active_redraw
                and not has_unsaved_edits
            )
        self._update_actions()
        self.workspace_changed.emit(dict(self._workspace))

    def _editable_modified_layers(self):
        modified = []
        for code in CLASS_ORDER:
            if code not in self._class_layers:
                continue
            layer = self._layer(code)
            if layer.isEditable() and layer.isModified():
                modified.append(code)
        return modified

    def _update_actions(self):
        confirmed = 0
        if self._workspace:
            confirmed = sum(
                1 for record in self._workspace["classes"].values()
                if record.get("confirmed")
            )
            for code in CLASS_ORDER:
                if code not in self._class_layers:
                    continue
                layer = self._layer(code)
                edit_session_enabled = bool(
                    not self._active_session
                    and not self._active_redraw
                    and layer.isEditable()
                )
                self._save_edit_actions[code].setEnabled(edit_session_enabled)
                self._rollback_edit_actions[code].setEnabled(edit_session_enabled)
        modified = self._editable_modified_layers() if self._workspace else []
        can_assemble = bool(
            self._workspace
            and confirmed == 14
            and not modified
            and not self._active_session
            and not self._active_redraw
        )
        self.assemble_btn.setEnabled(can_assemble)
        issue_text = "-" if self._issue_count is None else str(self._issue_count)
        self.summary_label.setText(
            f"14 类确认: {confirmed}/14    未解决问题: {issue_text}    "
            f"未保存编辑: {len(modified)}"
        )

    def _assemble_final(self):
        try:
            path, count = final_assembler.assemble_final(
                self._run_spec, self._workspace
            )
            self._final_path = path
            self.allow_issues_check.setChecked(False)
            self.layer_manager.load_final_composite(self._run_spec["run_id"], path)
            self.topology_btn.setEnabled(True)
            self.baseline_label.setText(f"final_composite 已组装，共 {count} 个面")
            self._check_topology()
        except Exception as exc:
            QMessageBox.warning(self, "组装失败", str(exc))

    def _accepted_layer_for_check(self):
        path = str(self._run_spec.get("accepted_gpkg") or "")
        if not path or not Path(path).is_file():
            return None
        layer = QgsVectorLayer(
            f"{path}|layername={LAYER_NAMES.ACCEPTED}", "accepted_for_topology", "ogr"
        )
        return layer if layer.isValid() else None

    def _check_topology(self):
        if not self._final_path:
            return
        try:
            path, count, counts = topology_validator.validate_topology(
                self._run_spec, self._final_path, self._accepted_layer_for_check()
            )
            self._issues_path = path
            self._issue_count = count
            self.layer_manager.load_topology_issues(self._run_spec["run_id"], path)
            self._update_accept_enabled()
            self.baseline_label.setText(
                "拓扑检查完成: " + (", ".join(f"{key}={value}" for key, value in counts.items()) or "无问题")
            )
            self._update_actions()
        except Exception as exc:
            self._issue_count = None
            self.accept_btn.setEnabled(False)
            QMessageBox.warning(self, "拓扑检查失败", str(exc))

    def _write_accepted(self):
        if self._issue_count is None:
            QMessageBox.warning(self, "入库", "必须先完成拓扑检查")
            return
        if self._issue_count != 0 and not self.allow_issues_check.isChecked():
            QMessageBox.warning(self, "入库", "请先解决拓扑问题，或明确勾选带问题入库")
            return
        try:
            count = accepted_writer.append_final_to_accepted(
                self._final_path,
                str(self._run_spec.get("accepted_gpkg") or ""),
                str(Path(self._run_spec["run_dir"]) / "run_manifest.json"),
            )
            self.baseline_label.setText(f"已写入 accepted_labels: {count} 个面")
        except Exception as exc:
            QMessageBox.warning(self, "写入 accepted_labels 失败", str(exc))

    def _update_accept_enabled(self, *_args):
        self.accept_btn.setEnabled(
            bool(
                self._final_path
                and self._issue_count is not None
                and (
                    self._issue_count == 0
                    or self.allow_issues_check.isChecked()
                )
            )
        )

    def cleanup(self):
        self._cancel_quick_redraw()
        self._cancel_active_session(record=False)
        self._disconnect_layer_signals()
        self._snapshots.clear()
        if self._worker is not None:
            self._worker.stop()
            self._worker = None

    def closeEvent(self, event):
        if self._editable_modified_layers():
            QMessageBox.warning(
                self,
                "关闭分类修整",
                "存在未保存编辑，请先在 QGIS 中保存或回滚后再关闭。",
            )
            event.ignore()
            return
        if self._active_redraw:
            answer = QMessageBox.question(
                self,
                "关闭分类修整",
                "当前快速重画尚未采用。是否取消候选并关闭窗口？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self._cancel_quick_redraw()
        if self._active_session:
            answer = QMessageBox.question(
                self,
                "关闭分类修整",
                "当前 SAM3 会话尚未完成。是否取消本次会话并关闭窗口？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
        self._cancel_active_session(record=True)
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
        event.ignore()
        self.hide()
