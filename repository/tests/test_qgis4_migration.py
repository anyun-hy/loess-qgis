from pathlib import Path
import ast
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "qgis_plugins" / "labeling_tool"


LEGACY_ENUM_PATTERNS = (
    r"Qt\.(?:RightDockWidgetArea|AlignLeft|AlignVCenter|TextSelectableByMouse)",
    r"Qt\.(?:ScrollBarAsNeeded|WA_DeleteOnClose|UserRole|ISODate)",
    r"QHeaderView\.(?:Stretch|ResizeToContents|Interactive)",
    r"QMessageBox\.(?:Yes|No)(?![A-Za-z])",
    r"QProcess\.NotRunning",
    r"QTextCursor\.End",
    r"QFont\.Bold",
    r"QFrame\.(?:VLine|Sunken|NoFrame)",
    r"QPlainTextEdit\.NoWrap",
    r"QTableWidget\.(?:NoEditTriggers|SelectRows|ExtendedSelection)",
    r"QgsMapLayerProxyModel\.RasterLayer",
    r"QgsWkbTypes\.(?:PolygonGeometry|UnknownGeometry)",
    r"QgsVectorFileWriter\.(?:CreateOrOverwriteFile|CreateOrOverwriteLayer|NoError)",
    r"QgsColorRampShader\.Interpolated",
)


def test_plugin_targets_qgis4_only():
    metadata = (PLUGIN_ROOT / "metadata.txt").read_text(encoding="utf-8")
    assert "qgisMinimumVersion=4.2" in metadata
    assert "version=0.3.0" in metadata


def test_plugin_contains_no_known_qt5_or_qgis3_enum_spelling():
    offenders = []
    for path in PLUGIN_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for pattern in LEGACY_ENUM_PATTERNS:
            if re.search(pattern, text):
                offenders.append(f"{path.relative_to(ROOT)}: {pattern}")
    assert offenders == []


def test_installer_defaults_to_qgis4_profile_and_qgis_conda_python():
    installer = (ROOT / "qgis_plugins" / "install_qgis_plugin.sh").read_text(
        encoding="utf-8"
    )
    assert "QGIS/QGIS4/profiles/${profile}/python/plugins" in installer
    assert "--profile" in installer
    assert "--plugin-dir" in installer
    assert "/opt/anaconda3/envs/qgis/bin/python" in installer
    assert "QGIS3/profiles" not in installer


def test_inference_environment_contract_is_qgis_without_conda_qgis():
    config_sh = (ROOT / "inference_scripts" / "config.sh").read_text(
        encoding="utf-8"
    )
    assert 'CONDA_ENV="${CONDA_ENV:-qgis}"' in config_sh
    assert 'CONDA_EXE="${CONDA_EXE:-/opt/anaconda3/bin/conda}"' in config_sh
    assert "export PYTHONNOUSERSITE=1" in config_sh

    environment_path = ROOT / "inference_scripts" / "environment-qgis.yml"
    environment = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    assert environment["name"] == "qgis"
    dependencies = environment["dependencies"]
    assert "python=3.12.13" in dependencies
    assert "pytorch=2.7.1" in dependencies
    assert not any(
        isinstance(item, str) and item.split("=", 1)[0] == "qgis"
        for item in dependencies
    )
    pip_entries = next(item["pip"] for item in dependencies if isinstance(item, dict))
    assert "sam3==0.1.4" in pip_entries
    assert "psutil==7.2.2" in pip_entries


def test_environment_check_owns_and_terminates_its_process_group():
    source = (
        PLUGIN_ROOT / "core" / "inference_config.py"
    ).read_text(encoding="utf-8")
    assert "QProcess.UnixProcessFlag.CreateNewSession" in source
    assert "setUnixProcessParameters(unix_parameters)" in source
    assert "os.killpg(pid, signal.SIGTERM)" in source
    assert "os.killpg(pid, signal.SIGKILL)" in source


def test_plugin_and_checker_fingerprint_the_environment_lock_file():
    plugin_source = (
        PLUGIN_ROOT / "core" / "inference_config.py"
    ).read_text(encoding="utf-8")
    plugin_fingerprint_block = plugin_source.split(
        "FINGERPRINT_FILES =", 1
    )[1].split("def config_fingerprint", 1)[0]
    checker_source = (
        ROOT / "inference_scripts" / "check_environment.py"
    ).read_text(encoding="utf-8")
    checker_fingerprint_block = checker_source.split(
        "FINGERPRINT_FILES =", 1
    )[1].split("def add_check", 1)[0]
    assert '"environment-qgis.yml"' in plugin_fingerprint_block
    assert '"environment-qgis.yml"' in checker_fingerprint_block


def test_processing_extent_preview_tracks_tile_controls():
    source = (PLUGIN_ROOT / "gui" / "main_dock.py").read_text(encoding="utf-8")
    assert 'tile_layout.addRow("自动扩展推理范围:"' in source
    assert source.count("setRange(64, 4096)") == 2
    assert "self.tile_width_spin.setEnabled(False)" not in source
    assert "self.tile_height_spin.setEnabled(False)" not in source
    assert "self.tile_width_spin.setEnabled(True)" in source
    assert "self.tile_height_spin.setEnabled(True)" in source
    assert (
        "self.tile_width_spin.valueChanged.connect("
        "self._on_tile_parameters_changed)"
    ) in source
    assert (
        "self.tile_height_spin.valueChanged.connect("
        "self._on_tile_parameters_changed)"
    ) in source
    assert (
        "self.overlap_spin.valueChanged.connect("
        "self._on_tile_parameters_changed)"
    ) in source
    assert "self.raster_combo.layerChanged.connect(self._on_raster_layer_changed)" in source
    assert "def _refresh_processing_extent_preview(self):" in source


def test_main_dock_uses_explicit_sequential_workflow():
    source = (PLUGIN_ROOT / "gui" / "main_dock.py").read_text(encoding="utf-8")
    ordered_groups = [
        'QGroupBox("① 数据源与范围")',
        'QGroupBox("② 切片")',
        'QGroupBox("③ 输出位置")',
        'QGroupBox("④ 推理环境")',
        'QGroupBox("⑤ 推理方案")',
        'QGroupBox("⑥ 执行")',
        'QGroupBox("⑦ 结果")',
    ]
    positions = [source.index(group) for group in ordered_groups]
    assert positions == sorted(positions)
    assert 'QPushButton("检查推理环境")' in source
    assert 'QPushButton("查看完整检查结果")' in source
    assert 'QPushButton("选择模型与 Fusion")' in source
    assert 'QPushButton("重新检查")' not in source
    assert 'QPushButton("配置推理方案")' not in source


def test_main_dock_can_start_a_prepared_v5_run_without_hiding_ready_results():
    source = (PLUGIN_ROOT / "gui" / "main_dock.py").read_text(encoding="utf-8")

    assert "self._recovery_run_spec = None" in source
    assert '"planned", "stopped", "failed", "running"' in source
    assert "self._recovery_run_spec or self._last_run_spec" in source


def test_v5_runner_requires_scale_acceptance_before_ready():
    source = (
        PLUGIN_ROOT / "core" / "v5_async_runner.py"
    ).read_text(encoding="utf-8")

    assert '"run_scale_acceptance.sh"' in source
    assert 'self._phase = "acceptance"' in source
    assert 'context.get("kind") == "scale_acceptance"' in source
    assert 'result["scale_acceptance_report_sha256"]' in source


def test_environment_check_is_manual_after_paths_change():
    source = (PLUGIN_ROOT / "gui" / "main_dock.py").read_text(encoding="utf-8")
    assert "QTimer.singleShot(0, self._run_env_check)" not in source
    assert "_env_check_timer" not in source
    assert (
        "self.script_path_edit.textChanged.connect("
        "self._mark_env_check_required)"
    ) in source
    assert (
        "self.output_path_edit.textChanged.connect("
        "self._mark_env_check_required)"
    ) in source
    assert (
        "self.workspace_edit.textChanged.connect("
        "self._on_workspace_changed)"
    ) in source
    marker_block = source.split(
        "def _mark_env_check_required", 1
    )[1].split("def _run_manual_env_check", 1)[0]
    assert "_run_env_check" not in marker_block
    assert "配置已变化，请检查推理环境" in marker_block
    workspace_block = source.split(
        "def _on_workspace_changed", 1
    )[1].split("def _run_manual_env_check", 1)[0]
    assert "self._last_run_result = None" in workspace_block
    assert "self._last_run_spec = None" in workspace_block
    assert "self._mark_env_check_required()" in workspace_block


def test_completed_run_restores_on_startup_without_environment_check():
    source = (PLUGIN_ROOT / "gui" / "main_dock.py").read_text(encoding="utf-8")
    constructor = source.split("def __init__", 1)[1].split("def _build_ui", 1)[0]
    assert "QTimer.singleShot(0, self._restore_latest_ready_run)" in constructor


def test_workspace_edit_tracking_survives_project_restored_edit_mode():
    source = (
        PLUGIN_ROOT / "gui" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    load_layers = source.split(
        "def _load_workspace_layers", 1
    )[1].split("def _layer", 1)[0]
    assert "if layer.isEditable() and code not in self._snapshots:" in load_layers
    assert "self._snapshots[code] = self._persisted_snapshot(code)" in load_layers
    assert "layer.afterCommitChanges.connect(" in load_layers
    assert "layer.editingStopped.connect(stopped_slot)" in load_layers
    assert "QTimer.singleShot" not in load_layers

    persisted = source.split(
        "def _persisted_snapshot", 1
    )[1].split("def _editing_started", 1)[0]
    assert "class_workspace.working_layer(" in persisted

    committed = source.split(
        "def _editing_stopped", 1
    )[1].split("def _set_attributes", 1)[0]
    assert "keep_editing = layer.isEditable()" in committed
    assert "layer.commitChanges(not keep_editing)" in committed
    assert "self._snapshots[class_code] = self._snapshot(layer)" in committed


def test_refinement_disconnects_layer_callbacks_before_qt_widgets_are_destroyed():
    source = (
        PLUGIN_ROOT / "gui" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    assert "self._layer_signal_slots = {}" in source
    assert "self._layer_signal_slots[layer.id()] = (" in source
    disconnect = source.split(
        "def _disconnect_layer_signals", 1
    )[1].split("def _layer", 1)[0]
    assert "signal.disconnect(slot)" in disconnect
    cleanup = source.split("def cleanup", 1)[1].split("def closeEvent", 1)[0]
    assert "self._disconnect_layer_signals()" in cleanup


def test_missing_confidence_pixels_do_not_abort_manual_geometry_save():
    source = (
        PLUGIN_ROOT / "gui" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    optional = source.split(
        "def _optional_confidence_statistics", 1
    )[1].split("def _set_class_modified", 1)[0]
    assert "except RuntimeError as exc:" in optional
    assert 'return None, None, str(exc)' in optional
    edit_save = source.split(
        "def _editing_stopped", 1
    )[1].split("def _set_attributes", 1)[0]
    assert '"confidence_statistics_unavailable"' in edit_save


def test_environment_details_open_only_from_details_button():
    source = (PLUGIN_ROOT / "gui" / "main_dock.py").read_text(encoding="utf-8")
    manual_check_block = source.split(
        "def _run_manual_env_check", 1
    )[1].split("def _run_env_check", 1)[0]
    assert "_show_env_details" not in manual_check_block
    assert "self.env_detail_btn.clicked.connect(self._show_env_details)" in source


def test_start_requires_explicit_inference_plan_confirmation():
    source = (PLUGIN_ROOT / "gui" / "main_dock.py").read_text(encoding="utf-8")
    assert "self._inference_plan_confirmed = False" in source
    assert "and self._inference_plan_confirmed" in source
    apply_block = source.split(
        "def _on_inference_configuration_applied", 1
    )[1].split("def _update_inference_summary", 1)[0]
    assert "self._inference_plan_confirmed = True" in apply_block
    assert 'plan_status = "已确认" if self._inference_plan_confirmed else "待确认"' in source


def test_pipeline_terminal_states_stop_indeterminate_progress_animation():
    source = (PLUGIN_ROOT / "gui" / "main_dock.py").read_text(encoding="utf-8")
    helper = source.split(
        "def _set_progress_terminal", 1
    )[1].split("def _on_stop", 1)[0]
    assert "self.progress_bar.setRange(0, 1)" in helper
    assert "self.progress_bar.setValue(1 if completed else 0)" in helper

    finished = source.split(
        "def _on_pipeline_finished", 1
    )[1].split("def _on_open_refinement", 1)[0]
    assert 'self._set_progress_terminal("完成", completed=True)' in finished
    assert 'self._set_progress_terminal("已停止")' in finished
    assert 'self._set_progress_terminal("失败")' in finished


def test_sam3_adopt_provenance_uses_the_persisted_geometry_hash():
    source = (
        PLUGIN_ROOT / "gui" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    adopt = source.split(
        "def _adopt_candidate", 1
    )[1].split("def _start_session_edit_current", 1)[0]
    assert "persisted = self._feature_by_object_id" in adopt
    assert 'session["persisted_geometry_hash"] = class_workspace.geometry_hash' in adopt
    assert 'after_geometry_hash=session["persisted_geometry_hash"]' in adopt

    record = source.split(
        "def _record_session", 1
    )[1].split("def _cancel_active_session", 1)[0]
    assert 'persisted_hash = str(session.get("persisted_geometry_hash") or "")' in record
    assert 'persisted_hash if decision == "adopted"' in record


def test_manual_class_feature_capture_prepopulates_identity_without_full_form():
    workspace_source = (
        PLUGIN_ROOT / "core" / "class_workspace.py"
    ).read_text(encoding="utf-8")
    assert 'prefix = f"{run_id}_new_"' in workspace_source
    assert "replace(replace(replace(uuid()" in workspace_source
    assert "QgsDefaultValue(object_expression, False)" in workspace_source
    assert "Qgis.AttributeFormSuppression.On" in workspace_source

    dialog_source = (
        PLUGIN_ROOT / "gui" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    assert 'addAction("新增人工面")' in dialog_source
    add_manual = dialog_source.split(
        "def _add_manual_feature", 1
    )[1].split("def _object_id_exists", 1)[0]
    assert "class_workspace.apply_class_constraints(" in add_manual
    assert 'getattr(self.iface, "actionAddFeature"' in add_manual


def test_manual_edit_has_explicit_pick_save_and_cancel_actions():
    source = (
        PLUGIN_ROOT / "gui" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    assert 'addAction("节点精修（备用）")' in source
    assert 'addAction("保存当前编辑")' in source
    assert 'addAction("取消当前编辑")' in source
    edit_selected = source.split(
        "def _edit_selected", 1
    )[1].split("def _begin_manual_pick", 1)[0]
    assert "self._begin_manual_pick(class_code)" in edit_selected
    picker = source.split(
        "def _manual_edit_map_clicked", 1
    )[1].split("def _save_class_edits", 1)[0]
    assert "self._features_at_map_point(layer, map_point)" in picker
    assert "layer.selectByIds([hits[0].id()])" in picker
    save = source.split(
        "def _save_class_edits", 1
    )[1].split("def _rollback_class_edits", 1)[0]
    assert "layer.commitChanges()" in save


def test_quick_redraw_uses_native_polybezier_preview_and_edit_buffer():
    source = (
        PLUGIN_ROOT / "gui" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    assert 'addAction("快速重画现有面")' in source
    assert 'addAction("新增人工面")' in source
    assert 'addAction("保存当前编辑")' in source
    assert 'addAction("取消当前编辑")' in source

    capture = source.split(
        "def _start_redraw_capture", 1
    )[1].split("def _disconnect_redraw_capture", 1)[0]
    assert "QgsMapToolDigitizeFeature(" in capture
    assert "QgsMapToolCapture.CaptureMode.CapturePolygon" in capture
    assert "Qgis.CaptureTechnique.PolyBezier" in capture
    assert "digitizingCompleted.connect" in capture

    preview = source.split(
        "def _update_redraw_preview", 1
    )[1].split("def _set_redraw_controls", 1)[0]
    assert "redraw_smooth_spin.value()" in preview
    assert "raw.smooth(iterations, 0.25, -1.0, 180.0)" in preview

    adopt = source.split(
        "def _adopt_quick_redraw", 1
    )[1].split("def _cancel_quick_redraw", 1)[0]
    assert "layer.changeGeometry(feature_id, geometry)" in adopt
    assert "layer.commitChanges" not in adopt
    assert "保存当前编辑" in adopt
    assert "取消当前编辑" in adopt

    reference = source.split(
        "def _show_redraw_reference", 1
    )[1].split("def _show_redraw_candidate", 1)[0]
    assert "QColor(105, 105, 105, 230)" in reference
    assert "Qt.PenStyle.DashLine" in reference


def test_class_confirmation_is_distinct_from_saving_one_feature():
    source = (
        PLUGIN_ROOT / "gui" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    assert '"整类确认"' in source
    assert 'QPushButton("确认整类")' in source
    confirm = source.split(
        "def _confirm_class", 1
    )[1].split("def _refresh_table", 1)[0]
    assert "keep_editing = layer.isEditable()" in confirm
    assert "if not keep_editing and not layer.startEditing():" in confirm
    assert "layer.commitChanges(not keep_editing)" in confirm


def test_manual_reclassification_moves_identity_and_records_overlap_hint():
    source = (
        PLUGIN_ROOT / "gui" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    assert 'addAction("更正类别")' in source
    chooser = source.split(
        "def _reclassify_selected", 1
    )[1].split("def _remove_transferred_feature", 1)[0]
    assert "QInputDialog.getItem(" in chooser

    move = source.split(
        "def _move_feature_to_class", 1
    )[1].split("def _sam_available", 1)[0]
    assert '"object_id": object_id' in move
    assert '"part_id": str(original.attribute("part_id") or "000")' in move
    assert '"class_code": target_code' in move
    assert '"geometry_source": "manual_edited"' in move
    assert '"feature_reclassified"' in move
    assert "overlap_hint=topology_hint" in move
    assert "self._remove_transferred_feature(target, object_id)" in move
    assert "target.selectByIds" not in move
    assert "self._refresh_class_display(source_code, target_code)" in move


def test_saved_class_edits_restore_category_color_and_clear_selection():
    source = (
        PLUGIN_ROOT / "gui" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    refresh = source.split(
        "def _refresh_class_display", 1
    )[1].split("def _set_visible", 1)[0]
    assert "StyleManager.apply_categorized_style(layer)" in refresh
    assert "layer.removeSelection()" in refresh
    assert "layer.triggerRepaint()" in refresh
    assert "canvas.clearCache()" in refresh
    assert "canvas.refresh()" in refresh

    saved = source.split(
        "def _editing_stopped", 1
    )[1].split("def _set_attributes", 1)[0]
    assert "self._refresh_class_display(class_code)" in saved


def test_monitor_tables_use_stable_user_resizable_columns():
    source = (
        PLUGIN_ROOT / "gui" / "inference_monitor.py"
    ).read_text(encoding="utf-8")
    assert "((1, 92), (2, 86), (3, 70), (4, 76), (5, 62))" in source
    assert "((0, 96), (1, 72), (2, 88))" in source
    assert "QHeaderView.ResizeMode.Interactive" in source
    assert "splitter.setSizes([650, 530])" in source


def test_monitor_updates_large_tile_tables_incrementally_and_names_selected_stream():
    source = (
        PLUGIN_ROOT / "gui" / "inference_monitor.py"
    ).read_text(encoding="utf-8")
    progress_block = source.split(
        "def _on_stream_progress", 1
    )[1].split("def _selected_stream", 1)[0]
    assert "_update_selected_tile(stream_id, tile_id, state)" in progress_block
    assert "_render_selected_tiles()" not in progress_block
    assert 'self._tile_rows = {}' in source
    assert 'f"选中结果流：{stream_id} | Tile 详情（已记录 {len(values)} 个）"' in source
    assert '"subpixel_vectorize:"' in source


def test_legacy_annotation_group_is_not_left_empty():
    source = (
        PLUGIN_ROOT / "core" / "layer_manager.py"
    ).read_text(encoding="utf-8")
    block = source.split("def group_layers", 1)[1]
    assert block.index("candidates = []") < block.index("root.addGroup(group_name)")
    assert "if group is not None and not group.children():" in block
    assert "parent.removeChildNode(group)" in block


def test_semantic_colors_have_one_qgis4_source_of_truth():
    assert not (PLUGIN_ROOT / "styles" / "semantic_14class.qml").exists()
    assert not (PLUGIN_ROOT / "styles" / "sam_refined.qml").exists()

    style_path = PLUGIN_ROOT / "core" / "style_manager.py"
    source = style_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_colors = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == "CLASS_COLORS"
                for target in node.targets
            ):
                class_colors = ast.literal_eval(node.value)
                break
    assert class_colors is not None
    assert list(class_colors) == [12, 13, 21, 31, 32, 33, 43, 51, 52, 53, 54, 61, 62, 71]
    assert "apply_categorized_style" in source
    assert "apply_semantic_raster_style" in source
    assert not any(
        "semantic_14class.qml" in path.read_text(encoding="utf-8")
        for path in PLUGIN_ROOT.rglob("*.py")
    )

    layer_manager = (
        PLUGIN_ROOT / "core" / "layer_manager.py"
    ).read_text(encoding="utf-8")
    sam_loader = layer_manager.split(
        "def load_sam_refined_polygons", 1
    )[1].split("def get_or_create_accepted_labels", 1)[0]
    assert "StyleManager.apply_categorized_style(layer)" in sam_loader
    assert "StyleManager.apply_outline_style(layer)" not in sam_loader


def test_qgis4_rubber_bands_are_removed_from_the_canvas_scene():
    source = (
        PLUGIN_ROOT / "gui" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    clear_bands = source.split("def _clear_bands", 1)[1].split(
        "def _set_decision_state", 1
    )[0]

    assert "scene.removeItem(band)" in clear_bands
    assert "band.deleteLater()" not in clear_bands
