from pathlib import Path
import ast
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = ROOT / "qgis_plugins" / "labeling_tool"


DIRECT_PLATFORM_ENUM_PATTERNS = (
    r"Qt\.(?:RightDockWidgetArea|AlignLeft|AlignVCenter|TextSelectableByMouse)",
    r"Qt\.(?:ScrollBarAsNeeded|WA_DeleteOnClose|UserRole|ISODate)",
    r"Qt\.(?:WindowType|AlignmentFlag|PenStyle|ItemDataRole|WidgetAttribute)",
    r"QHeaderView\.(?:Stretch|ResizeToContents|Interactive)",
    r"QHeaderView\.ResizeMode",
    r"QMessageBox\.(?:Yes|No)(?![A-Za-z])",
    r"QMessageBox\.StandardButton",
    r"QProcess\.NotRunning",
    r"QProcess\.ProcessState",
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


def test_plugin_targets_qgis3_and_qgis4_from_one_release():
    metadata = (PLUGIN_ROOT / "metadata.txt").read_text(encoding="utf-8")
    assert "qgisMinimumVersion=3.44" in metadata
    assert "qgisMaximumVersion=4.99" in metadata
    assert "version=0.4.0" in metadata
    assert "-linux" not in metadata


def test_business_modules_use_the_shared_qt_compatibility_facade():
    offenders = []
    for path in PLUGIN_ROOT.rglob("*.py"):
        if path.name == "qt_compat.py":
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in DIRECT_PLATFORM_ENUM_PATTERNS:
            if re.search(pattern, text):
                offenders.append(f"{path.relative_to(ROOT)}: {pattern}")
    assert offenders == []


def test_plugin_install_and_project_initialization_are_independent():
    installer = (ROOT / "bash" / "install_plugin.sh").read_text(encoding="utf-8")
    initializer = (ROOT / "bash" / "init_project.sh").read_text(encoding="utf-8")
    assert "--platform" in installer
    assert "--profile" in installer
    assert "--plugin-dir" in installer
    assert "--check-only" in installer
    assert "QGIS/QGIS3/profiles/${PROFILE}/python/plugins" in installer
    assert "QGIS/QGIS4/profiles/${PROFILE}/python/plugins" in installer
    assert "deployment_manifest.json" in installer
    assert 'mv "${STAGED_DEST}" "${DEST_PLUGIN}"' in installer
    assert "--project-root" not in installer
    assert "--create-env" not in installer

    assert "--project-root" in initializer
    assert "--create-env" in initializer
    assert "--check-assets" in initializer
    assert "environment-ubuntu-cu124.yml" in initializer
    assert "environment-macos-qgis4.yml" in initializer
    assert "project_manifest.json" in initializer
    assert "runtime/labeling_tool/core" in initializer
    assert "--profile" not in initializer
    assert "--plugin-dir" not in initializer

    assert not (ROOT / "install.sh").exists()
    assert not (ROOT / "qgis_plugins" / "install_qgis_plugin.sh").exists()


def test_inference_environment_contract_has_two_minimal_platform_locks():
    config_sh = (ROOT / "inference_scripts" / "config.sh").read_text(
        encoding="utf-8"
    )
    assert 'CONDA_ENV="${CONDA_ENV:-qgis}"' in config_sh
    assert 'CONDA_ENV="${LOESS_CONDA_ENV_OVERRIDE:-${LOESS_CONFIGURED_CONDA_ENV}}"' in config_sh
    assert 'CONDA_EXE="${LOESS_CONDA_EXE_OVERRIDE:-${LOESS_CONFIGURED_CONDA_EXE}}"' in config_sh
    assert 'LOESS_PLATFORM="${LOESS_PLATFORM:-auto}"' in config_sh
    assert 'LOESS_ENV_LOCK="environment-ubuntu-cu124.yml"' in config_sh
    assert 'LOESS_ENV_LOCK="environment-macos-qgis4.yml"' in config_sh
    assert "export PYTHONNOUSERSITE=1" in config_sh

    ubuntu = yaml.safe_load(
        (ROOT / "inference_scripts" / "environment-ubuntu-cu124.yml").read_text(
            encoding="utf-8"
        )
    )
    macos = yaml.safe_load(
        (ROOT / "inference_scripts" / "environment-macos-qgis4.yml").read_text(
            encoding="utf-8"
        )
    )
    assert ubuntu["name"] == macos["name"] == "qgis"
    assert "python=3.12" in ubuntu["dependencies"]
    assert "python=3.12.13" in macos["dependencies"]
    assert "pytorch=2.7.1" in macos["dependencies"]
    required_sam3_packages = {
        "sam3==0.1.4",
        "timm==1.0.28",
        "tqdm==4.67.3",
        "ftfy==6.3.1",
        "regex==2026.7.10",
        "iopath==0.1.10",
        "typing_extensions==4.15.0",
        "huggingface-hub==1.23.0",
        "einops==0.8.2",
        "pycocotools==2.0.11",
        "safetensors==0.8.0",
        "psutil==7.2.2",
    }
    for environment in (ubuntu, macos):
        assert not any(
            isinstance(item, str) and item.split("=", 1)[0] == "qgis"
            for item in environment["dependencies"]
        )
        pip_packages = next(
            item["pip"]
            for item in environment["dependencies"]
            if isinstance(item, dict) and "pip" in item
        )
        assert required_sam3_packages <= set(pip_packages)

    initializer = (ROOT / "bash" / "init_project.sh").read_text(
        encoding="utf-8"
    )
    assert "torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0" in initializer
    assert "https://download.pytorch.org/whl/cu124" in initializer


def test_environment_check_owns_and_terminates_its_process_group():
    manager = (PLUGIN_ROOT / "core" / "inference_config.py").read_text(
        encoding="utf-8"
    )
    compat = (PLUGIN_ROOT / "core" / "process_compat.py").read_text(
        encoding="utf-8"
    )
    assert "configure_process(" in manager
    assert '"UnixProcessParameters"' in compat
    assert '"CreateNewSession"' in compat
    assert '"setChildProcessModifier"' in compat
    assert 'shutil.which("setsid")' in compat
    assert "os.killpg(pid, signal.SIGTERM)" in manager
    assert "os.killpg(pid, signal.SIGKILL)" in manager


def test_plugin_and_checker_fingerprint_the_manifest_and_persisted_launcher():
    plugin_source = (
        PLUGIN_ROOT / "core" / "deployment_contract.py"
    ).read_text(encoding="utf-8")
    plugin_fingerprint_block = plugin_source.split(
        "DEPLOYMENT_FINGERPRINT_FILES =", 1
    )[1].split("LAUNCHER_RELATIVE_PATH", 1)[0]
    checker_source = (
        ROOT / "inference_scripts" / "check_environment.py"
    ).read_text(encoding="utf-8")
    checker_fingerprint_block = checker_source.split(
        "FINGERPRINT_FILES =", 1
    )[1].split("def add_check", 1)[0]
    for contract_file in (
        '"../project_manifest.json"',
        '"../runtime/loess_launcher.sh"',
    ):
        assert contract_file in plugin_fingerprint_block
        assert contract_file in checker_fingerprint_block
    for obsolete_manual_entry in (
        '"tile_materializer.py"',
        '"mosaic_builder.py"',
        '"work_package_runtime.py"',
    ):
        assert obsolete_manual_entry not in plugin_fingerprint_block
        assert obsolete_manual_entry not in checker_fingerprint_block


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


def test_main_dock_uses_frozen_per_model_batch_sizes_for_storage_preflight():
    source = (PLUGIN_ROOT / "gui" / "main_dock.py").read_text(encoding="utf-8")
    preparation_block = source.split("resolved_resources =", 1)[1].split(
        "stride = 512", 1
    )[0]

    assert 'registry.runtime["tile_batch_size"]' in preparation_block
    assert 'resolved_resources.get("tile_batch_size_by_model")' in preparation_block
    assert "storage_batch_size = max(selected_batch_sizes" in preparation_block
    assert "tile_batch_size=storage_batch_size" in preparation_block
    assert 'scaling["tile_batch_size"]' not in preparation_block
    assert '"resolved_score_cache_budget_gb"' in preparation_block
    assert 'scaling["score_cache_budget_mode"]' in preparation_block


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
    assert "self._undo_stack_signal_slots = {}" in source
    disconnect = source.split(
        "def _disconnect_layer_signals", 1
    )[1].split("def _layer", 1)[0]
    assert "signal.disconnect(slot)" in disconnect
    assert "undo_stack.indexChanged.disconnect(slot)" in disconnect
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
    assert 'QPushButton("新增面")' in dialog_source
    add_manual = dialog_source.split(
        "def _commit_manual_add", 1
    )[1].split("def _finish_add_task", 1)[0]
    assert "class_workspace.new_object_id(self._run_spec)" in add_manual
    assert '"geometry_source": "manual_edited"' in add_manual
    assert '"geometry_revision": 1' in add_manual
    assert "layer.addFeature(feature)" in add_manual
    assert "layer.commitChanges(False)" in add_manual
    assert '"feature_added"' in add_manual


def test_guided_add_opens_toggle_editing_before_polybezier_capture():
    source = (
        PLUGIN_ROOT / "gui" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    begin_add = source.split(
        "def _begin_add_task", 1
    )[1].split("def _start_manual_picker", 1)[0]
    assert '"editing_started_by_task": False' in begin_add

    capture = source.split(
        "def _start_manual_capture", 1
    )[1].split("def _manual_capture_completed", 1)[0]
    activate_index = capture.index("self.iface.setActiveLayer(layer)")
    editing_index = capture.index("layer.startEditing()")
    tool_index = capture.index("tool = QgsMapToolDigitizeFeature(")
    assert activate_index < editing_index < tool_index
    assert "if not layer.isEditable():" in capture
    assert 'task.get("kind") == "add" and not layer.isEditable()' not in capture
    assert "QgsMapToolCapture.CaptureMode.CapturePolygon" in capture
    assert "Qgis.CaptureTechnique.PolyBezier" in capture

    add_commit = source.split(
        "def _commit_manual_add", 1
    )[1].split("def _finish_add_task", 1)[0]
    assert "layer.commitChanges(False)" in add_commit

    finish_add = source.split(
        "def _finish_add_task", 1
    )[1].split("def _finish_manual_editing", 1)[0]
    assert "self._finish_manual_editing(task)" in finish_add
    finish_editing = source.split(
        "def _finish_manual_editing", 1
    )[1].split("def _undo_current_edit", 1)[0]
    assert "layer.rollBack()" in finish_editing

    manual_panel = source.split(
        "def _update_manual_panel", 1
    )[1].split("def _snapshot", 1)[0]
    add_controls = manual_panel.split(
        'elif kind == "add":', 1
    )[1].split("edit_layer = self.iface.activeLayer()", 1)[0]
    assert 'manual_cancel_btn.setText("取消当前面")' not in add_controls
    assert "self.manual_cancel_btn.setVisible(True)" not in add_controls

    cancel_action = source.split(
        "def _manual_cancel_action", 1
    )[1].split("def _manual_modify_overlap_plan", 1)[0]
    assert 'task.get("kind") == "add"' not in cancel_action


def test_advanced_single_feature_modify_is_removed_but_external_qgis_edit_sync_remains():
    source = (
        PLUGIN_ROOT / "gui" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    assert "modify_mode_combo" not in source
    assert "处理方式:" not in source
    assert "节点精修（高级）" not in source
    assert "manual_smooth_spin" not in source
    assert "_start_vertex_task" not in source
    assert "_save_vertex_task" not in source
    assert "_move_feature_to_class" not in source
    assert "_begin_quick_redraw" not in source
    assert 'QGroupBox("QGIS 原生编辑（高级）")' in source
    assert 'QPushButton("撤销一步")' in source
    assert 'QPushButton("重做一步")' in source
    assert 'QPushButton("保存 QGIS 编辑")' in source
    assert 'QPushButton("放弃 QGIS 编辑")' in source
    manual_group = source.split(
        'self.manual_group = QGroupBox("人工操作")', 1
    )[1].split("root.addWidget(self.manual_group)", 1)[0]
    assert "qgis_undo_btn" not in manual_group
    qgis_group = source.split(
        'self.qgis_edit_group = QGroupBox("QGIS 原生编辑（高级）")', 1
    )[1].split('self.session_group = QGroupBox("SAM3 会话")', 1)[0]
    assert 'self.qgis_edit_group.hide()' in qgis_group
    assert "root.addWidget(self.qgis_edit_group)" in qgis_group
    panel = source.split(
        "def _update_manual_panel", 1
    )[1].split("def _snapshot", 1)[0]
    assert "and edit_layer.isEditable() and not task" in panel
    assert "self.qgis_edit_group.setVisible(show_edit)" in panel
    assert "当前 QGIS 编辑层：{edit_code} {CLASS_NAMES[edit_code]}" in panel
    assert "layer.undoStack().undo()" in source
    assert "layer.undoStack().redo()" in source


def test_qgis_native_edit_smoothing_previews_parameters_and_applies_one_undo_command():
    source = (
        PLUGIN_ROOT / "gui" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    assert 'QPushButton("预览光滑效果")' in source
    assert 'QPushButton("应用光滑")' in source
    assert 'QPushButton("取消预览")' in source
    assert "self.qgis_smooth_iterations_spin.setRange(1, 3)" in source
    assert "self.qgis_smooth_offset_spin.setRange(0.05, 0.45)" in source
    assert "self.qgis_smooth_angle_spin.setRange(30.0, 180.0)" in source
    assert '"labeling_tool/qgis_smoothing/iterations"' in source
    assert '"labeling_tool/qgis_smoothing/offset"' in source
    assert '"labeling_tool/qgis_smoothing/max_angle"' in source

    preview = source.split(
        "def _preview_qgis_smoothing", 1
    )[1].split("def _apply_qgis_smoothing", 1)[0]
    assert "layer.selectedFeatures()" in preview
    assert "source.smooth(iterations, offset, -1.0, max_angle)" in preview
    assert "self._manual_geometry_error(smoothed)" in preview
    assert "QgsRubberBand(" in preview
    assert "layer.changeGeometry(" not in preview
    assert '"source_hashes": source_hashes' in preview
    assert "顶点 {old_vertices} → {new_vertices}" in preview
    assert "总面积变化" in preview

    current = source.split(
        "def _qgis_smooth_preview_is_current", 1
    )[1].split("def _update_qgis_smoothing_controls", 1)[0]
    assert "layer.selectedFeatureIds()" in current
    assert "self._qgis_smooth_parameters() != preview" in current
    assert "class_workspace.geometry_hash(feature.geometry())" in current

    apply_smoothing = source.split(
        "def _apply_qgis_smoothing", 1
    )[1].split("def _undo_current_edit", 1)[0]
    assert "layer.beginEditCommand(" in apply_smoothing
    assert "layer.changeGeometry(feature_id, geometry)" in apply_smoothing
    assert "layer.endEditCommand()" in apply_smoothing
    assert "layer.destroyEditCommand()" in apply_smoothing
    assert "layer.commitChanges" not in apply_smoothing


def test_guided_add_and_modify_auto_preview_one_smoothing_parameter_set_per_batch():
    source = (
        PLUGIN_ROOT / "gui" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    manual_group = source.split(
        'self.manual_group = QGroupBox("人工操作")', 1
    )[1].split("root.addWidget(self.manual_group)", 1)[0]
    assert 'QCheckBox("光滑处理")' in manual_group
    assert "self.manual_smooth_iterations_spin.setRange(1, 3)" in manual_group
    assert "self.manual_smooth_offset_spin.setRange(0.05, 0.45)" in manual_group
    assert "self.manual_smooth_angle_spin.setRange(30.0, 180.0)" in manual_group
    assert 'QPushButton("预览光滑效果")' not in manual_group
    assert "self._manual_smoothing_timer.setInterval(250)" in source
    assert "self._manual_smoothing_timer.timeout.connect(" in source
    assert "self._refresh_manual_smoothing_preview" in source

    parameters = source.split(
        "def _manual_smoothing_parameters_changed", 1
    )[1].split("def _schedule_manual_smoothing_preview", 1)[0]
    assert "self._store_smoothing_parameters(parameters)" in parameters
    assert 'self._sync_smoothing_parameter_widgets(parameters, "manual")' in parameters
    assert "self._schedule_manual_smoothing_preview()" in parameters

    preview = source.split(
        "def _refresh_manual_smoothing_preview", 1
    )[1].split("def _manual_geometries_for_commit", 1)[0]
    assert "for geometry in (task.get(\"pending_geometries\") or [])" in preview
    assert "source.smooth(iterations, offset, -1.0, max_angle)" in preview
    assert "self._manual_geometry_error(smoothed)" in preview
    assert 'task["smoothing_preview"] = {' in preview
    assert '"source_hashes": tuple(' in preview
    assert '"geometries": smoothed_geometries' in preview
    assert "总面积变化" in preview

    effective = source.split(
        "def _manual_geometries_for_commit", 1
    )[1].split("def _open_manual_operations", 1)[0]
    assert "if not task.get(\"smoothing_enabled\")" in effective
    assert "self._manual_smoothing_preview_is_current(task)" in effective
    assert 'task["smoothing_preview"]["geometries"]' in effective

    modify_commit = source.split(
        "def _commit_manual_modify_batch", 1
    )[1].split("def _commit_manual_delete", 1)[0]
    assert "raw_new_geometries" in modify_commit
    assert "new_geometries = self._manual_geometries_for_commit(task)" in modify_commit
    assert "old_features, raw_new_geometries" in modify_commit
    assert "身份匹配按原始边界计算，保存当前光滑预览" in modify_commit

    add_commit = source.split(
        "def _commit_manual_add", 1
    )[1].split("def _finish_manual_session", 1)[0]
    assert "raw_geometries" in add_commit
    assert "geometries = self._manual_geometries_for_commit(task)" in add_commit
    assert "feature.setGeometry(geometry)" in add_commit

    panel = source.split(
        "def _update_manual_panel", 1
    )[1].split("def _snapshot", 1)[0]
    assert "show_manual_smoothing = bool((is_modify or is_add) and pending_count > 0)" in panel
    assert "smoothing_ready" in panel
    assert "and smoothing_ready" in panel

    delete = source.split(
        "def _commit_manual_delete", 1
    )[1].split("def _commit_manual_add", 1)[0]
    assert "smoothing" not in delete


def test_guided_modify_batches_old_selection_and_polybezier_replacements():
    source = (
        PLUGIN_ROOT / "gui" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    assert 'QPushButton("修改现有面")' in source
    manual_ui = source.split(
        'self.manual_group = QGroupBox("人工操作")', 1
    )[1].split("self.modify_task_btn.clicked.connect", 1)[0]
    assert manual_ui.index("self.manual_instruction_label = QLabel") < manual_ui.index(
        'self.target_class_label = QLabel("本批目标类别:")'
    )
    begin = source.split(
        "def _begin_modify_task", 1
    )[1].split("def _toggle_manual_modify_feature", 1)[0]
    assert '"selected_feature_ids": list(layer.selectedFeatureIds())' in begin
    assert '"pending_geometries": []' in begin
    assert '"submitted_batch_count": 0' in begin
    assert "layer.removeSelection()" in begin
    assert "self._refresh_manual_modify_reference()" in begin

    toggle = source.split(
        "def _toggle_manual_modify_feature", 1
    )[1].split("def _begin_delete_task", 1)[0]
    assert "selected_ids.remove(feature_id)" in toggle
    assert "selected_ids.append(feature_id)" in toggle
    assert "layer.removeSelection()" in toggle

    capture = source.split(
        "def _start_manual_capture", 1
    )[1].split("def _manual_capture_completed", 1)[0]
    assert "QgsMapToolDigitizeFeature(" in capture
    assert "QgsMapToolCapture.CaptureMode.CapturePolygon" in capture
    assert "Qgis.CaptureTechnique.PolyBezier" in capture
    assert "digitizingCompleted.connect" in capture
    assert "layer.startEditing()" in capture

    reference = source.split(
        "def _refresh_manual_modify_reference", 1
    )[1].split("def _show_manual_candidate", 1)[0]
    assert "QColor(105, 105, 105, 230)" in reference
    assert "QColor(105, 105, 105, 55)" in reference
    assert "band.addGeometry(feature.geometry(), layer)" in reference

    overlap = source.split(
        "def _manual_modify_overlap_plan", 1
    )[1].split("def _manual_replacement_feature", 1)[0]
    assert "old_geometry.intersection(geometry)" in overlap
    assert "area <= 0.0" in overlap
    assert "candidate_hits" not in overlap
    assert "没有与任何灰色旧面产生面积相交" not in overlap
    assert "unmatched_old" in overlap
    assert "unmatched_new" in overlap

    commit = source.split(
        "def _commit_manual_modify_batch", 1
    )[1].split("def _commit_manual_delete", 1)[0]
    assert "source.changeGeometry(original.id(), geometry)" in commit
    assert "source.deleteFeatures([feature.id() for feature in old_features])" in commit
    assert "target.addFeature(feature)" in commit
    assert "self._remove_transferred_features(target, added_target_ids)" in commit
    assert '"geometry_modified"' in commit
    assert '"feature_reclassified"' in commit
    assert '"feature_deleted"' in commit
    assert '"feature_added"' in commit
    assert 'reason="manual_batch_added"' in commit
    assert 'expected_added = len(plan["unmatched_new"])' in commit
    assert 'task["state"] = "selecting"' in commit
    assert "self._start_manual_picker()" in commit


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


def test_category_correction_is_integrated_into_modify_and_preserves_identity():
    source = (
        PLUGIN_ROOT / "gui" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    assert 'addAction("更正类别")' not in source
    assert 'QPushButton("更正类别")' not in source
    assert 'QLabel("本批目标类别:")' in source
    assert '"保留原边界，仅修改类别", "keep"' not in source

    prepare = source.split(
        "def _manual_replacement_feature", 1
    )[1].split("def _remove_transferred_features", 1)[0]
    assert '"object_id": object_id' in prepare
    assert '"part_id": str(original.attribute("part_id") or "000")' in prepare
    assert '"class_code": target_code' in prepare
    assert '"geometry_source": "manual_edited"' in prepare
    assert '"geometry_revision": int(original.attribute("geometry_revision") or 0) + 1' in prepare
    assert "recalculate_confidence=False" in source


def test_guided_delete_toggles_multiple_selection_and_commits_history():
    source = (
        PLUGIN_ROOT / "gui" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    assert 'QPushButton("删除现有面")' in source
    picker = source.split(
        "def _manual_task_map_clicked", 1
    )[1].split("def _disconnect_manual_picker", 1)[0]
    assert "ids = set(layer.selectedFeatureIds())" in picker
    assert "ids.remove(feature.id())" in picker
    assert "ids.add(feature.id())" in picker
    deletion = source.split(
        "def _commit_manual_delete", 1
    )[1].split("def _commit_manual_add", 1)[0]
    assert "layer.deleteFeatures(feature_ids)" in deletion
    assert "layer.commitChanges()" in deletion
    manual_panel = source.split(
        "def _update_manual_panel", 1
    )[1].split("def _snapshot", 1)[0]
    delete_controls = manual_panel.split(
        'elif kind == "delete":', 1
    )[1].split('elif kind == "add":', 1)[0]
    assert 'self.manual_cancel_btn.setText("结束删除")' in delete_controls
    assert "(is_modify and modify_selected_count > 0)" in manual_panel
    assert "or (is_add and pending_count > 0)" in manual_panel
    assert "self.target_class_label.setVisible(show_target_class)" in manual_panel
    assert "self.target_class_combo.setVisible(show_target_class)" in manual_panel


def test_continuous_add_queues_batch_selects_target_and_commits_once():
    source = (
        PLUGIN_ROOT / "gui" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    add_start = source.split(
        "def _begin_add_task", 1
    )[1].split("def _start_manual_picker", 1)[0]
    assert '"kind": "add"' in add_start
    assert '"added_count": 0' in add_start
    assert '"submitted_batch_count": 0' in add_start
    assert '"pending_geometries": []' in add_start
    assert '"pending_errors": []' in add_start
    target = source.split(
        "def _target_class_changed", 1
    )[1].split("def _manual_task_guard", 1)[0]
    assert 'task["target_code"] = target_code' in target
    assert "连续新增已锁定" not in target
    capture_completed = source.split(
        "def _manual_capture_completed", 1
    )[1].split("def _manual_capture_cancelled", 1)[0]
    assert 'task.setdefault("pending_geometries", []).append(geometry)' in capture_completed
    assert 'task.setdefault("pending_errors", []).append(error)' in capture_completed
    assert 'self._schedule_manual_capture_transition("restart")' in capture_completed
    retry = source.split(
        "def _manual_retry_action", 1
    )[1].split("def _manual_clear_action", 1)[0]
    assert "geometries.pop()" in retry
    assert "errors.pop()" in retry
    add_commit = source.split(
        "def _commit_manual_add", 1
    )[1].split("def _finish_add_task", 1)[0]
    assert 'target_code = int(task.get("target_code", source_code))' in add_commit
    assert "for feature, _object_id, _warning in prepared:" in add_commit
    assert "layer.commitChanges(False)" in add_commit
    assert 'batch_size = len(prepared)' in add_commit
    assert 'task["added_count"] += batch_size' in add_commit
    assert 'task["submitted_batch_count"] += 1' in add_commit
    assert 'task["pending_geometries"] = []' in add_commit
    assert "self._start_manual_capture()" in add_commit
    manual_panel = source.split(
        "def _update_manual_panel", 1
    )[1].split("def _snapshot", 1)[0]
    assert '(is_modify and modify_selected_count > 0)' in manual_panel
    assert 'or (is_add and pending_count > 0)' in manual_panel
    assert 'self.manual_primary_btn.setText("保存新增面并继续新增面")' in manual_panel
    assert 'self.target_class_label.setText("本批目标类别:")' in manual_panel
    finish = source.split(
        "def _finish_add_task", 1
    )[1].split("def _finish_manual_editing", 1)[0]
    assert "本次提交" in finish
    assert "丢弃" in finish


def test_digitize_completion_defers_map_tool_replacement_until_next_qt_turn():
    source = (
        PLUGIN_ROOT / "gui" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    assert "from qgis.PyQt.QtCore import QTimer, pyqtSignal" in source
    assert "from ..qt_compat import (" in source
    assert "self._manual_capture_transition_timer = QTimer(self)" in source
    assert "self._manual_capture_transition_timer.setSingleShot(True)" in source

    completed = source.split(
        "def _manual_capture_completed", 1
    )[1].split("def _manual_capture_cancelled", 1)[0]
    assert "_disconnect_manual_capture" not in completed
    assert "self._start_manual_capture()" not in completed
    assert 'self._schedule_manual_capture_transition("restore")' in completed
    assert 'self._schedule_manual_capture_transition("restart")' in completed

    cancelled = source.split(
        "def _manual_capture_cancelled", 1
    )[1].split("def _schedule_manual_capture_transition", 1)[0]
    assert "_disconnect_manual_capture" not in cancelled
    assert 'self._schedule_manual_capture_transition("restore")' in cancelled

    transition = source.split(
        "def _schedule_manual_capture_transition", 1
    )[1].split("def _manual_modify_selected_features", 1)[0]
    assert "self._manual_capture_transition_timer.start(0)" in transition
    assert "def _run_manual_capture_transition" in transition
    assert 'self._disconnect_manual_capture(restore=action == "restore")' in transition
    assert 'action == "restart" and self._manual_task is task' in transition
    assert "self._retire_manual_capture_tool(tool)" in transition


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


def test_class_table_active_layer_visibility_and_row_actions_stay_synchronized():
    source = (
        PLUGIN_ROOT / "gui" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")

    assert "self.table.currentCellChanged.connect(self._activate_row)" in source
    assert "self.table.setSelectionMode(SINGLE_SELECTION)" in source
    assert "self.iface.currentLayerChanged.connect" not in source
    assert 'getattr(self.iface, "currentLayerChanged", None)' in source
    assert "signal.connect(self._iface_current_layer_slot)" in source
    assert "signal.disconnect(self._iface_current_layer_slot)" in source
    assert "menu.aboutToShow.connect(" in source
    assert 'QPushButton("打开操作")' in source
    assert "edit_button.clicked.connect(" in source
    opener = source.split(
        "def _open_manual_operations", 1
    )[1].split("def _selection_changed", 1)[0]
    assert "self._select_class_context(class_code, activate_layer=True)" in opener
    load_layers = source.split(
        "def _load_workspace_layers", 1
    )[1].split("def _disconnect_layer_signals", 1)[0]
    assert "layer.selectionChanged.connect(selection_slot)" in load_layers
    assert '"geometryChanged"' in load_layers
    assert '"featureAdded"' in load_layers
    assert '"featureDeleted"' in load_layers
    assert "undo_stack.indexChanged.connect(undo_slot)" in load_layers
    assert "self._undo_stack_signal_slots[layer.id()]" in load_layers
    assert 'getattr(self.iface.mapCanvas(), "mapToolSet", None)' in source

    select_context = source.split(
        "def _select_class_context", 1
    )[1].split("def _activate_row", 1)[0]
    assert "self.table.selectRow(row)" in select_context
    assert "self.iface.setActiveLayer(layer)" in select_context

    active_changed = source.split(
        "def _active_layer_changed", 1
    )[1].split("def _snapshot", 1)[0]
    assert "self._class_code_for_layer(layer)" in active_changed
    assert "activate_layer=False" in active_changed
    assert "self.table.clearSelection()" in active_changed

    visibility_sync = source.split(
        "def _sync_visibility_from_layer_tree", 1
    )[1].split("def _class_code_for_layer", 1)[0]
    assert "tree_layer.itemVisibilityChecked()" in visibility_sync
    assert "checkbox.blockSignals(True)" in visibility_sync

    cleanup = source.split("def cleanup", 1)[1].split("def closeEvent", 1)[0]
    assert "self._disconnect_iface_layer_signal()" in cleanup
    assert "self._disconnect_map_tool_signal()" in cleanup
    assert "self._cancel_manual_task(silent=True)" in cleanup


def test_monitor_tables_use_stable_user_resizable_columns():
    source = (
        PLUGIN_ROOT / "gui" / "inference_monitor.py"
    ).read_text(encoding="utf-8")
    assert "((1, 92), (2, 86), (3, 70), (4, 76), (5, 62))" in source
    assert "((0, 96), (1, 72), (2, 88))" in source
    assert "header.setSectionResizeMode(column, INTERACTIVE)" in source
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
