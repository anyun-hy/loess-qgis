import ast
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PLUGIN_ROOT = ROOT / "src" / "labeling_tool"


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


def test_plugin_targets_qgis42_pyqt6_qt6_from_one_release():
    metadata = (PLUGIN_ROOT / "metadata.txt").read_text(encoding="utf-8")
    assert "qgisMinimumVersion=4.2" in metadata
    assert "qgisMaximumVersion=4.99" in metadata
    assert "PyQt6" in metadata
    assert "Qt5" not in metadata
    assert "version=2.0.0" in metadata
    assert "author=anyun-hy" in metadata
    assert "repository=https://github.com/anyun-hy/loess-qgis" in metadata
    assert "tracker=https://github.com/anyun-hy/loess-qgis/issues" in metadata
    assert (PLUGIN_ROOT / "LICENSE").is_file()
    assert "-linux" not in metadata


def test_ubuntu_plugin_requires_native_qt6_wayland_qpa():
    plugin = (PLUGIN_ROOT / "plugin.py").read_text(encoding="utf-8")
    assert "self._require_supported_qpa()" in plugin
    assert 'sys.platform.startswith("linux")' in plugin
    assert "QGuiApplication.platformName()" in plugin
    assert 'qpa != "wayland"' in plugin
    assert "native Qt6 Wayland" in plugin

    active_roots = (PLUGIN_ROOT, ROOT / "src" / "loess_runtime", ROOT / "scripts")
    forbidden_launch_overrides = (
        "QT_QPA_PLATFORM=xcb",
        "-platform xcb",
    )
    offenders = []
    for active_root in active_roots:
        for path in active_root.rglob("*"):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for override in forbidden_launch_overrides:
                if override in text:
                    offenders.append(f"{path.relative_to(ROOT)}: {override}")
    assert offenders == []


def test_layer_combo_popup_caps_rows_and_anchors_inside_active_screen():
    source = (PLUGIN_ROOT / "main" / "main_dock.py").read_text(encoding="utf-8")
    popup_node = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef)
        and node.name == "_ScreenBoundMapLayerComboBox"
    )
    popup_class = ast.get_source_segment(source, popup_node)
    assert "MAX_VISIBLE_ITEMS = 15" in popup_class
    assert "setMaxVisibleItems(self.MAX_VISIBLE_ITEMS)" in popup_class
    assert "setVerticalScrollBarPolicy(SCROLLBAR_AS_NEEDED)" in popup_class
    assert "def _visible_rows_height(self):" in popup_class
    assert "def _popup_height_limit(self):" in popup_class
    assert "return row_height * visible_rows" in popup_class
    assert "self.style().pixelMetric(" in popup_class
    assert "MENU_SCROLLER_HEIGHT" in popup_class
    assert "+ 2 * menu_scroller_height" in popup_class
    assert "popup.setMaximumHeight(height_limit)" in popup_class
    assert "def showPopup(self):" in popup_class
    assert "super().showPopup()" in popup_class
    assert "QApplication.screenAt(anchor)" in popup_class
    assert "screen.availableGeometry()" in popup_class
    assert "popup.move(popup_x, popup_y)" in popup_class
    assert source.count("_ScreenBoundMapLayerComboBox()") == 2


def test_inference_terminal_notices_do_not_block_wayland_monitor():
    source = (PLUGIN_ROOT / "main" / "main_dock.py").read_text(encoding="utf-8")
    helper = source.split("def _show_nonblocking_notice", 1)[1].split(
        "def _on_workflow_pre_run_failed", 1
    )[0]
    finish = source.split("def _on_workflow_pre_run_failed", 1)[1].split(
        "def _set_progress_terminal", 1
    )[0]

    assert "monitor.isVisible()" in helper
    assert "QMessageBox(parent)" in helper
    assert "dialog.setWindowModality(NON_MODAL)" in helper
    assert "dialog.setAttribute(WA_DELETE_ON_CLOSE, True)" in helper
    assert "dialog.show()" in helper
    assert "dialog.raise_()" in helper
    assert "dialog.activateWindow()" in helper
    assert ".exec(" not in helper
    assert "monitor_dialog.mark_finished(title, message)" in finish
    assert "_show_nonblocking_notice(CRITICAL, title, message)" in finish
    assert "QMessageBox.critical(self, title, message)" not in source


def test_retired_qgis3_qt5_runtime_code_is_absent():
    assert not (PLUGIN_ROOT / "qt_compat.py").exists()
    assert not (PLUGIN_ROOT / "core" / "process_compat.py").exists()

    retired_patterns = (
        "PyQt5",
        "QGIS/QGIS3",
        'EXPECTED_QGIS="3.',
        "qgisMinimumVersion=3",
        "(3, 44)",
        "(3, 5, 5)",
    )
    active_roots = (
        PLUGIN_ROOT,
        ROOT / "src" / "loess_runtime",
        ROOT / "scripts",
        ROOT / ".github",
    )
    offenders = []
    for active_root in active_roots:
        for path in active_root.rglob("*"):
            if not path.is_file() or path.suffix in {".pyc", ".png"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in retired_patterns:
                if pattern in text:
                    offenders.append(f"{path.relative_to(ROOT)}: {pattern}")
    assert offenders == []


def test_business_modules_use_the_shared_qt6_enum_api():
    offenders = []
    for path in PLUGIN_ROOT.rglob("*.py"):
        if path.name in {"qt6_api.py", "process_runtime.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in DIRECT_PLATFORM_ENUM_PATTERNS:
            if re.search(pattern, text):
                offenders.append(f"{path.relative_to(ROOT)}: {pattern}")
    assert offenders == []


def test_plugin_install_and_project_initialization_are_independent():
    installer = (ROOT / "scripts" / "deploy" / "install_plugin.sh").read_text(encoding="utf-8")
    initializer = (ROOT / "scripts" / "deploy" / "init_project.sh").read_text(encoding="utf-8")
    assert "--platform" in installer
    assert "--profile" in installer
    assert "--plugin-dir" in installer
    assert "--check-only" in installer
    assert "QGIS/QGIS4/profiles/${PROFILE}/python/plugins" in installer
    assert "QGIS/QGIS3/profiles/${PROFILE}/python/plugins" not in installer
    assert 'EXPECTED_QGIS="4.2."' in installer
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
    assert "runtime/labeling_tool/shared" in initializer
    assert "--profile" not in initializer
    assert "--plugin-dir" not in initializer

    assert not (ROOT / "install.sh").exists()
    assert not (ROOT / "src" / "install_qgis_plugin.sh").exists()


def test_inference_environment_contract_has_two_minimal_platform_locks():
    config_sh = (ROOT / "scripts" / "runtime" / "config.sh").read_text(
        encoding="utf-8"
    )
    assert 'CONDA_ENV="${CONDA_ENV:-qgis}"' in config_sh
    assert 'CONDA_ENV="${LOESS_CONDA_ENV_OVERRIDE:-${LOESS_CONFIGURED_CONDA_ENV}}"' in config_sh
    assert 'CONDA_EXE="${LOESS_CONDA_EXE_OVERRIDE:-${LOESS_CONFIGURED_CONDA_EXE}}"' in config_sh
    assert 'LOESS_PLATFORM="${LOESS_PLATFORM:-auto}"' in config_sh
    assert 'LOESS_ENV_LOCK="environment-ubuntu-cu124.yml"' in config_sh
    assert 'LOESS_ENV_LOCK="environment-macos-qgis4.yml"' in config_sh
    assert 'LOESS_CONFIG_ROOT="${LOESS_SOURCE_ROOT}/configs/defaults"' in config_sh
    assert 'LOESS_CONFIG_ASSET_ROOT="${LOESS_SOURCE_ROOT}/configs"' in config_sh
    assert 'LOESS_ENVIRONMENT_ROOT="${LOESS_SOURCE_ROOT}/configs/environments"' in config_sh
    assert "export PYTHONNOUSERSITE=1" in config_sh
    assert "expandable_segments:True,garbage_collection_threshold:0.8" in config_sh

    checker = (ROOT / "src" / "loess_runtime" / "system" / "environment_deployment.py").read_text(
        encoding="utf-8"
    )
    assert "maximum_batch_size=int(runtime.get(\"tile_batch_size\") or 1)" in checker
    assert "SAM_TOKENIZER_PATH" in checker
    assert "Path(__file__).resolve().parents[1]" in checker
    assert (
        ROOT
        / "src"
        / "loess_runtime"
        / "sam"
        / "assets"
        / "bpe_simple_vocab_16e6.txt.gz"
    ).is_file()

    ubuntu = yaml.safe_load(
        (ROOT / "configs" / "environments" / "environment-ubuntu-cu124.yml").read_text(
            encoding="utf-8"
        )
    )
    macos = yaml.safe_load(
        (ROOT / "configs" / "environments" / "environment-macos-qgis4.yml").read_text(
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
            isinstance(item, str)
            and item.split("=", 1)[0] in {"pyarrow", "pyogrio"}
            for item in environment["dependencies"]
        )
        assert not any(
            isinstance(item, str) and item.split("=", 1)[0] == "qgis"
            for item in environment["dependencies"]
        )
        pip_packages = next(
            item["pip"]
            for item in environment["dependencies"]
            if isinstance(item, dict) and "pip" in item
        )
        assert required_sam3_packages | {
            "pyarrow==25.0.1",
            "pyogrio==0.13.0",
        } <= set(pip_packages)

    initializer = (ROOT / "scripts" / "deploy" / "init_project.sh").read_text(
        encoding="utf-8"
    )
    assert "torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0" in initializer
    assert "https://download.pytorch.org/whl/cu124" in initializer


def test_pyogrio_import_has_no_shapely_geos_deprecation_warning():
    completed = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::DeprecationWarning",
            "-c",
            "import pyogrio",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_environment_check_owns_and_terminates_its_process_group():
    manager = (PLUGIN_ROOT / "main" / "inference_config.py").read_text(
        encoding="utf-8"
    )
    runtime = (PLUGIN_ROOT / "qgis_support" / "process_runtime.py").read_text(
        encoding="utf-8"
    )
    assert "configure_process(" in manager
    assert "QProcess.UnixProcessParameters()" in runtime
    assert "QProcess.UnixProcessFlag.CreateNewSession" in runtime
    assert "process.setUnixProcessParameters(parameters)" in runtime
    assert "getattr(" not in runtime
    assert "os.killpg(pid, signal.SIGTERM)" in manager
    assert "os.killpg(pid, signal.SIGKILL)" in manager


def test_plugin_and_checker_fingerprint_the_manifest_and_persisted_launcher():
    plugin_source = (
        PLUGIN_ROOT / "runs" / "deployment_contract.py"
    ).read_text(encoding="utf-8")
    plugin_fingerprint_block = plugin_source.split(
        "DEPLOYMENT_FINGERPRINT_FILES =", 1
    )[1].split("LAUNCHER_RELATIVE_PATH", 1)[0]
    checker_source = (
        ROOT / "src" / "loess_runtime" / "system" / "environment_report.py"
    ).read_text(encoding="utf-8")
    checker_fingerprint_files = next(
        ast.literal_eval(node.value)
        for node in ast.parse(checker_source).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "FINGERPRINT_FILES"
            for target in node.targets
        )
    )
    for contract_file in (
        '"../project_manifest.json"',
        '"../runtime/loess_launcher.sh"',
    ):
        assert contract_file in plugin_fingerprint_block
        assert contract_file.strip('"') in checker_fingerprint_files
    for obsolete_manual_entry in (
        '"tile_materializer.py"',
        '"mosaic_builder.py"',
        '"work_package_runtime.py"',
    ):
        assert obsolete_manual_entry not in plugin_fingerprint_block
        assert obsolete_manual_entry.strip('"') not in checker_fingerprint_files


def test_processing_extent_preview_tracks_tile_controls():
    source = (PLUGIN_ROOT / "main" / "main_dock.py").read_text(encoding="utf-8")
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
    source = (PLUGIN_ROOT / "main" / "main_dock.py").read_text(encoding="utf-8")
    ordered_groups = [
        'QGroupBox("数据源与范围")',
        'QGroupBox("切片与高级设置")',
        'QGroupBox("输出位置")',
        'self.environment_panel = EnvironmentPanel()',
        'self.plan_panel = InferencePlanPanel(',
        'QGroupBox("开始与当前状态")',
        'QGroupBox("结果")',
    ]
    positions = [source.index(group) for group in ordered_groups]
    assert positions == sorted(positions)
    panel = (PLUGIN_ROOT / "main" / "environment_panel.py").read_text(encoding="utf-8")
    assert 'super().__init__("推理环境", parent)' in panel
    assert 'QPushButton("检查推理环境")' in panel
    assert 'QPushButton("查看完整检查结果")' in panel
    plan = (PLUGIN_ROOT / "main" / "inference_plan_panel.py").read_text(encoding="utf-8")
    assert 'super().__init__("推理方案", parent)' in plan
    assert 'QPushButton("选择模型与 Fusion")' in plan
    assert 'QPushButton("重新检查")' not in source
    assert 'QPushButton("配置推理方案")' not in source


def test_main_dock_can_start_a_prepared_v5_run_without_hiding_ready_results():
    source = (PLUGIN_ROOT / "main" / "main_dock.py").read_text(encoding="utf-8")

    assert "self._recovery_run_spec = None" in source
    assert "run_index.RECOVERABLE_RUN_STATES" in source
    assert "run_state_from_spec" not in source
    assert "self._recovery_run_spec or self._last_run_spec" in source


def test_run_planning_uses_frozen_per_model_batch_sizes_for_storage_preflight():
    source = (PLUGIN_ROOT / "runs" / "run_planning.py").read_text(
        encoding="utf-8"
    )
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
        PLUGIN_ROOT / "runs" / "v5_async_runner.py"
    ).read_text(encoding="utf-8")
    terminal_source = (
        PLUGIN_ROOT / "runs" / "run_terminal_finalizer.py"
    ).read_text(encoding="utf-8")

    assert '"run_scale_acceptance.sh"' in source
    assert 'self._phase = "acceptance"' in source
    assert 'context.get("kind") == "scale_acceptance"' in source
    assert "result = build_run_result(" in source
    assert 'result["scale_acceptance_report_sha256"]' in terminal_source


def test_environment_check_is_manual_after_paths_change():
    source = (PLUGIN_ROOT / "main" / "main_dock.py").read_text(encoding="utf-8")
    assert "QTimer.singleShot(0, self._run_env_check)" not in source
    assert "_env_check_timer" not in source
    assert (
        "self.environment_panel.script_path_changed.connect("
        "self._mark_env_check_required)"
    ) in "".join(source.split())
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
    )[1].split("def _run_env_check", 1)[0]
    assert "_run_env_check" not in marker_block
    assert "self.environment_panel.mark_check_required()" in marker_block
    workspace_block = source.split(
        "def _on_workspace_changed", 1
    )[1].split("def _run_env_check", 1)[0]
    assert "self._last_run_result = None" in workspace_block
    assert "self._last_run_spec = None" in workspace_block
    assert "self._mark_env_check_required()" in workspace_block


def test_completed_run_restores_on_startup_without_environment_check():
    source = (PLUGIN_ROOT / "main" / "main_dock.py").read_text(encoding="utf-8")
    constructor = source.split("def __init__", 1)[1].split("def _build_ui", 1)[0]
    assert "QTimer.singleShot(0, self._restore_latest_ready_run)" in constructor


def test_refinement_workspace_open_is_background_cancellable_and_incremental():
    dock_source = (PLUGIN_ROOT / "main" / "main_dock.py").read_text(
        encoding="utf-8"
    )
    opener = dock_source.split("def _on_open_refinement", 1)[1].split(
        "def _on_load_manual_run", 1
    )[0]
    assert "valid_ready_stream_ids(" not in opener
    assert "approved_fusion_streams(" not in opener

    dialog_source = (
        PLUGIN_ROOT / "refinement" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    set_run = dialog_source.split("def set_run", 1)[1].split(
        "def _workspace_task_progress", 1
    )[0]
    assert "self._workspace_tasks.probe(" in set_run
    assert "approved_fusion_streams(" not in set_run
    assert "load_workspace(" not in set_run

    initialize = dialog_source.split("def _initialize_workspace", 1)[1].split(
        "def _workspace_initialize_completed", 1
    )[0]
    assert "self._workspace_tasks.initialize(" in initialize
    assert "initialize_workspace(" not in initialize

    layer_load = dialog_source.split("def _load_workspace_layers", 1)[1].split(
        "def _workspace_summary_text", 1
    )[0]
    assert "load_workspace_classes(" not in layer_load
    assert "self._layer_loader.start(" in layer_load
    assert 'QPushButton("取消后台加载")' in dialog_source

    refresh = dialog_source.split("def _refresh_table", 1)[1].split(
        "def _editable_modified_layers", 1
    )[0]
    assert "source_statistics(" not in refresh
    assert "self._workspace_statistics.get(code)" in refresh

    workspace_source = (
        PLUGIN_ROOT / "refinement" / "class_workspace.py"
    ).read_text(encoding="utf-8")
    assert "class ClassWorkspaceProbeTask(QgsTask):" in workspace_source
    assert "class ClassWorkspaceInitializeTask(QgsTask):" in workspace_source
    assert "is_canceled=self.isCanceled" in workspace_source
    assert "SELECT geometry_source, COUNT(*)" in workspace_source

    task_source = (
        PLUGIN_ROOT / "refinement" / "workspace_tasks.py"
    ).read_text(encoding="utf-8")
    assert "ClassWorkspaceProbeTask(" in task_source
    assert "ClassWorkspaceInitializeTask(" in task_source

    manager_source = (
        PLUGIN_ROOT / "qgis_support" / "layer_manager.py"
    ).read_text(encoding="utf-8")
    workspace_loader = manager_source.split(
        "def load_workspace_class", 1
    )[1].split("def load_workspace_classes", 1)[0]
    assert "visible=False" in workspace_loader
    assert "setItemVisibilityChecked(bool(visible))" in workspace_loader
    assert "triggerRepaint()" not in workspace_loader


def test_portable_manual_workspace_does_not_require_fusion_polygon_copy():
    loader_source = (
        PLUGIN_ROOT / "refinement" / "manual_run_loader.py"
    ).read_text(encoding="utf-8")
    assert "require_semantic=not has_workspace" in loader_source
    assert 'rebound["portable_classes_only"] = not baseline_available' in loader_source

    workspace_source = (
        PLUGIN_ROOT / "refinement" / "class_workspace.py"
    ).read_text(encoding="utf-8")
    load_workspace = workspace_source.split("def load_workspace", 1)[1].split(
        "def workspace_source_statistics", 1
    )[0]
    assert 'if not run_spec.get("manual_only"):' in load_workspace
    assert "Fusion baseline" in load_workspace

    dock_source = (PLUGIN_ROOT / "main" / "main_dock.py").read_text(
        encoding="utf-8"
    )
    manual_open = dock_source.split("def _on_load_manual_run", 1)[1].split(
        "# ── Helpers", 1
    )[0]
    assert "validate_manual_fusion_stream(" not in manual_open
    assert "validate_manual_workspace(" not in manual_open
    assert "ManualRunLoadTask(" in manual_open
    assert "正在后台校验" in manual_open
    background_tasks = (
        PLUGIN_ROOT / "refinement" / "background_io_tasks.py"
    ).read_text(encoding="utf-8")
    assert "manual_run_loader.prepare_manual_run(" in background_tasks
    assert "manual_run_loader.publish_manual_run_bundle(bundle)" in background_tasks
    assert "persist_rebound_workspace(bundle)" in loader_source


def test_workspace_edit_tracking_survives_project_restored_edit_mode():
    source = (
        PLUGIN_ROOT / "refinement" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    load_layers = source.split(
        "def _load_workspace_layers", 1
    )[1].split("def _layer", 1)[0]
    assert "if layer.isEditable() and not self._edit_tracker.has_session(code):" in load_layers
    assert "self._edit_tracker.restore(code, layer, persisted)" in load_layers
    assert "class_workspace.working_layer(" in load_layers
    assert "self._layer_signals.bind_layer(" in load_layers
    assert "QTimer.singleShot" not in load_layers

    committed = source.split(
        "def _editing_stopped", 1
    )[1].split("def _feature_by_object_id", 1)[0]
    assert "self._edit_tracker.finish(" in committed
    assert "self._mark_class_modified(class_code)" in committed


def test_refinement_disconnects_layer_callbacks_before_qt_widgets_are_destroyed():
    source = (
        PLUGIN_ROOT / "refinement" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    assert "self._layer_signals = WorkspaceLayerSignals(self)" in source
    assert "self._layer_signal_slots" not in source
    assert "self._undo_stack_signal_slots" not in source
    cleanup = source.split("def cleanup", 1)[1].split("def closeEvent", 1)[0]
    assert "self._layer_signals.cleanup()" in cleanup
    assert cleanup.index("self._layer_signals.cleanup()") < cleanup.index(
        "self._edit_tracker.reset()"
    )


def test_missing_confidence_pixels_do_not_abort_manual_geometry_save():
    source = (
        PLUGIN_ROOT / "refinement" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    optional = source.split(
        "def _optional_confidence_statistics", 1
    )[1].split("def _set_class_modified", 1)[0]
    assert "except RuntimeError as exc:" in optional
    assert 'return None, None, str(exc)' in optional
    edit_save = (PLUGIN_ROOT / "refinement" / "edit_tracking.py").read_text(encoding="utf-8")
    assert '"confidence_statistics_unavailable"' in edit_save


def test_environment_details_open_only_from_details_button():
    source = (PLUGIN_ROOT / "main" / "main_dock.py").read_text(encoding="utf-8")
    check_block = source.split(
        "def _run_env_check", 1
    )[1].split("def _on_env_check_started", 1)[0]
    assert "_show_env_details" not in check_block
    assert "self.environment_panel.details_requested.connect(self._show_env_details)" in source
    panel = (PLUGIN_ROOT / "main" / "environment_panel.py").read_text(encoding="utf-8")
    assert "self.details_button.clicked.connect(self.details_requested)" in panel


def test_pipeline_terminal_states_stop_indeterminate_progress_animation():
    source = (PLUGIN_ROOT / "main" / "main_dock.py").read_text(encoding="utf-8")
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
        PLUGIN_ROOT / "refinement" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    adopt = source.split(
        "def _adopt_candidate", 1
    )[1].split("def _start_session_edit_current", 1)[0]
    assert "persisted = self._feature_by_object_id" in adopt
    assert "session.persisted_geometry_hash = class_workspace.geometry_hash" in adopt
    assert "after_geometry_hash=session.persisted_geometry_hash" in adopt

    record = source.split(
        "def _record_session", 1
    )[1].split("def _cancel_active_session", 1)[0]
    assert "session.history_record(" in record
    assert "geometry_hash=class_workspace.geometry_hash" in record


def test_manual_class_feature_capture_prepopulates_identity_without_full_form():
    workspace_source = (
        PLUGIN_ROOT / "refinement" / "class_workspace.py"
    ).read_text(encoding="utf-8")
    assert 'prefix = f"{run_id}_new_"' in workspace_source
    assert "replace(replace(replace(uuid()" in workspace_source
    assert "QgsDefaultValue(object_expression, False)" in workspace_source
    assert "Qgis.AttributeFormSuppression.On" in workspace_source

    dialog_source = (
        PLUGIN_ROOT / "refinement" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    operations_source = (
        PLUGIN_ROOT / "refinement" / "manual_edit_operations.py"
    ).read_text(encoding="utf-8")
    panel_source = (PLUGIN_ROOT / "refinement/manual_edit_panel.py").read_text()
    assert 'QPushButton("新增面")' in panel_source
    add_manual = dialog_source.split(
        "def _commit_manual_add", 1
    )[1].split("def _finish_add_task", 1)[0]
    assert "manual_edit_operations.prepare_manual_add(" in add_manual
    assert "manual_edit_commit.commit_manual_add(" in add_manual
    assert "manual_edit_operations.record_add_history(" in add_manual
    assert '"feature_added"' in operations_source


def test_guided_add_opens_toggle_editing_before_polybezier_capture():
    source = (
        PLUGIN_ROOT / "refinement" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    begin_add = source.split(
        "def _begin_add_task", 1
    )[1].split("def _manual_task_map_clicked", 1)[0]
    assert "ManualEditTask.for_add(" in begin_add

    capture = source.split(
        "def _start_manual_capture", 1
    )[1].split("def _manual_capture_completed", 1)[0]
    activate_index = capture.index("self.iface.setActiveLayer(layer)")
    editing_index = capture.index("layer.startEditing()")
    tool_index = capture.index("self._manual_tools.start_capture(layer)")
    assert activate_index < editing_index < tool_index
    assert "if not layer.isEditable():" in capture
    assert 'task.kind == "add" and not layer.isEditable()' not in capture
    tools_source = (
        PLUGIN_ROOT / "refinement" / "manual_edit_tools.py"
    ).read_text(encoding="utf-8")
    assert "QgsMapToolCapture.CaptureMode.CapturePolygon" in tools_source
    assert "Qgis.CaptureTechnique.PolyBezier" in tools_source

    add_commit = source.split(
        "def _commit_manual_add", 1
    )[1].split("def _finish_add_task", 1)[0]
    assert "keep_editing=bool(target_code == source_code or was_editable)" in add_commit

    finish_add = source.split(
        "def _finish_add_task", 1
    )[1].split("def _undo_current_edit", 1)[0]
    assert "manual_edit_operations.close_clean_task_editing_session(" in finish_add
    operations_source = (
        PLUGIN_ROOT / "refinement" / "manual_edit_operations.py"
    ).read_text(encoding="utf-8")
    close_editing = operations_source.split(
        "def close_clean_task_editing_session", 1
    )[1].split("def _geometries_for_commit", 1)[0]
    assert "task.editing_started_by_task" in close_editing
    assert "or layer.isModified()" in close_editing
    assert "layer.rollBack()" in close_editing

    cancel_action = source.split(
        "def _manual_cancel_action", 1
    )[1].split("def _commit_manual_modify_batch", 1)[0]
    assert 'task.kind == "add"' not in cancel_action


def test_advanced_single_feature_modify_is_removed_but_external_qgis_edit_sync_remains():
    source = (
        PLUGIN_ROOT / "refinement" / "class_refinement_dialog.py"
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
    manual_group = (PLUGIN_ROOT / "refinement/manual_edit_panel.py").read_text()
    assert "qgis_undo_btn" not in manual_group
    qgis_group = source.split(
        'self.qgis_edit_group = QGroupBox("QGIS 原生编辑（高级）")', 1
    )[1].split('self._sam_panel = SamSessionPanel', 1)[0]
    assert 'self.qgis_edit_group.hide()' in qgis_group
    assert "root.addWidget(self.qgis_edit_group)" in qgis_group
    panel = source.split(
        "def _update_manual_panel", 1
    )[1].split("def _editing_started", 1)[0]
    assert "and edit_layer.isEditable() and not task" in panel
    assert "self.qgis_edit_group.setVisible(show_edit)" in panel
    assert "当前 QGIS 编辑层：{edit_code} {CLASS_NAMES[edit_code]}" in panel
    assert "layer.undoStack().undo()" in source
    assert "layer.undoStack().redo()" in source


def test_qgis_native_edit_smoothing_previews_parameters_and_applies_one_undo_command():
    source = (
        PLUGIN_ROOT / "refinement" / "class_refinement_dialog.py"
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
    assert "smooth_geometry_batch(" in preview
    assert "QgsRubberBand(" in preview
    assert "layer.changeGeometry(" not in preview
    assert "NativeSmoothingPreview(" in preview
    assert "总面积变化" in preview

    # Native behavioral probes verify stale inputs and whole-batch undo/rollback.

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
        PLUGIN_ROOT / "refinement" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    manual_group = (PLUGIN_ROOT / "refinement/manual_edit_panel.py").read_text()
    assert 'QCheckBox("光滑处理")' in manual_group
    assert "self._iterations.setRange(1, 3)" in manual_group
    assert "self._offset.setRange(0.05, 0.45)" in manual_group
    assert "self._angle.setRange(30.0, 180.0)" in manual_group
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
    )[1].split("def _open_manual_operations", 1)[0]
    assert "smooth_geometry_batch(" in preview
    assert "convert_to_multi=True" in preview
    assert "task.smoothing_preview = preview" in preview
    assert "总面积变化" in preview

    operations_source = (
        PLUGIN_ROOT / "refinement" / "manual_edit_operations.py"
    ).read_text(encoding="utf-8")
    effective = operations_source.split(
        "def _geometries_for_commit", 1
    )[1].split("def _validate_geometries", 1)[0]
    assert "if not task.smoothing_enabled" in effective
    assert "geometry_source_hash" in effective
    assert "preview.geometries" in effective

    modify_commit = source.split(
        "def _commit_manual_modify_batch", 1
    )[1].split("def _commit_manual_delete", 1)[0]
    assert "manual_edit_operations.prepare_manual_modify(" in modify_commit
    assert "preparation.geometries" in modify_commit
    assert "身份匹配按原始边界计算，保存当前光滑预览" in modify_commit

    add_commit = source.split(
        "def _commit_manual_add", 1
    )[1].split("def _finish_manual_session", 1)[0]
    assert "manual_edit_operations.prepare_manual_add(" in add_commit
    assert "geometries=preparation.geometries," in add_commit

    panel = source.split(
        "def _update_manual_panel", 1
    )[1].split("def _editing_started", 1)[0]
    assert "self._manual_panel.render(" in panel
    assert "smoothing_ready" in panel

    delete = source.split(
        "def _commit_manual_delete", 1
    )[1].split("def _commit_manual_add", 1)[0]
    assert "smoothing" not in delete


def test_guided_modify_batches_old_selection_and_polybezier_replacements():
    source = (
        PLUGIN_ROOT / "refinement" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    manual_ui = (PLUGIN_ROOT / "refinement/manual_edit_panel.py").read_text()
    assert 'QPushButton("修改现有面")' in manual_ui
    assert manual_ui.index("self._instruction = QLabel") < manual_ui.index(
        'self._target_label = QLabel("本批目标类别:")'
    )
    begin = source.split(
        "def _begin_modify_task", 1
    )[1].split("def _toggle_manual_modify_feature", 1)[0]
    assert "ManualEditTask.for_modify(" in begin
    assert "layer.selectedFeatureIds()" in begin
    assert "layer.removeSelection()" in begin
    assert "self._refresh_manual_modify_reference()" in begin

    toggle = source.split(
        "def _toggle_manual_modify_feature", 1
    )[1].split("def _begin_delete_task", 1)[0]
    assert "task.toggle_selected_feature(" in toggle
    assert "layer.removeSelection()" in toggle

    capture = source.split(
        "def _start_manual_capture", 1
    )[1].split("def _manual_capture_completed", 1)[0]
    assert "self._manual_tools.start_capture(layer)" in capture
    assert "layer.startEditing()" in capture

    tree = ast.parse(source)
    reference = next(
        ast.get_source_segment(source, node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_refresh_manual_modify_reference"
    )
    assert "QColor(105, 105, 105, 230)" in reference
    assert "QColor(105, 105, 105, 55)" in reference
    assert "band.addGeometry(feature.geometry(), layer)" in reference

    commit = source.split(
        "def _commit_manual_modify_batch", 1
    )[1].split("def _commit_manual_delete", 1)[0]
    assert "manual_edit_commit.commit_manual_modify(" in commit
    assert "manual_edit_operations.record_modify_history(" in commit
    assert "manual_edit_operations.restart_edit_tracking(" in commit
    assert "task.record_modify_batch(" in commit
    assert "self._manual_tools.start_picker()" in commit
    operations_source = (
        PLUGIN_ROOT / "refinement" / "manual_edit_operations.py"
    ).read_text(encoding="utf-8")
    preparation = operations_source.split(
        "def prepare_manual_modify", 1
    )[1].split("def prepare_manual_add", 1)[0]
    assert "manual_edit_commit.plan_manual_modify_overlaps(" in preparation
    assert "expected_added=len(plan.unmatched_new)" in preparation
    history = operations_source.split(
        "def record_modify_history", 1
    )[1].split("def record_add_history", 1)[0]
    assert '"geometry_modified"' in history
    assert '"feature_reclassified"' in history
    assert '"feature_deleted"' in history
    assert '"feature_added"' in history
    assert 'reason="manual_batch_added"' in history


def test_class_confirmation_is_distinct_from_saving_one_feature():
    dialog = (
        PLUGIN_ROOT / "refinement" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    panel = (
        PLUGIN_ROOT / "refinement" / "class_review_panel.py"
    ).read_text(encoding="utf-8")
    assert "confirm_requested = pyqtSignal(int, bool)" in panel
    assert 'QPushButton("确认整类")' in panel
    assert "self._confirm.setCheckable(True)" in panel
    assert "self.confirm_requested.emit(self._selected_class_code, checked)" in panel
    confirm = dialog.split(
        "def _confirm_class", 1
    )[1].split("def _refresh_table", 1)[0]
    assert "keep_editing = layer.isEditable()" in confirm
    assert "class_review.commit_class_review(" in confirm
    assert "keep_editing=keep_editing" in confirm
    review = (
        PLUGIN_ROOT / "refinement" / "class_review.py"
    ).read_text(encoding="utf-8")
    assert "if not keep_editing and not layer.startEditing():" in review
    assert "layer.commitChanges(not keep_editing)" in review


def test_category_correction_is_integrated_into_modify_and_preserves_identity():
    source = (
        PLUGIN_ROOT / "refinement" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    assert 'addAction("更正类别")' not in source
    assert 'QPushButton("更正类别")' not in source
    panel_source = (PLUGIN_ROOT / "refinement/manual_edit_panel.py").read_text()
    assert 'QLabel("本批目标类别:")' in panel_source
    assert '"保留原边界，仅修改类别", "keep"' not in source

    operations_source = (
        PLUGIN_ROOT / "refinement" / "manual_edit_operations.py"
    ).read_text(encoding="utf-8")
    history = operations_source.split(
        "def record_modify_history", 1
    )[1].split("def record_add_history", 1)[0]
    assert '"feature_reclassified"' in history
    assert 'object_id=matched.object_id' in history
    assert 'part_id=str(matched.original.attribute("part_id") or "000")' in history


def test_guided_delete_toggles_multiple_selection_and_commits_history():
    source = (
        PLUGIN_ROOT / "refinement" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    panel_source = (PLUGIN_ROOT / "refinement/manual_edit_panel.py").read_text()
    assert 'QPushButton("删除现有面")' in panel_source
    picker = source.split(
        "def _manual_task_map_clicked", 1
    )[1].split("def _manual_tool_interrupted", 1)[0]
    assert "ids = set(layer.selectedFeatureIds())" in picker
    assert "ids.remove(feature.id())" in picker
    assert "ids.add(feature.id())" in picker
    deletion = source.split(
        "def _commit_manual_delete", 1
    )[1].split("def _commit_manual_add", 1)[0]
    assert "manual_edit_commit.commit_manual_delete(layer, feature_ids)" in deletion
    assert ".suppress()" not in deletion
    # Native panel tests verify count-dependent controls and task switching.


def test_continuous_add_queues_batch_selects_target_and_commits_once():
    source = (
        PLUGIN_ROOT / "refinement" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    add_start = source.split(
        "def _begin_add_task", 1
    )[1].split("def _manual_task_map_clicked", 1)[0]
    assert "ManualEditTask.for_add(" in add_start
    target = source.split(
        "def _target_class_changed", 1
    )[1].split("def _manual_task_guard", 1)[0]
    assert "task.target_code = target_code" in target
    assert "连续新增已锁定" not in target
    capture_completed = source.split(
        "def _manual_capture_completed", 1
    )[1].split("def _manual_capture_cancelled", 1)[0]
    assert "task.append_candidate(geometry, error)" in capture_completed
    assert 'self._manual_tools.schedule_transition("restart")' in capture_completed
    retry = source.split(
        "def _manual_retry_action", 1
    )[1].split("def _manual_clear_action", 1)[0]
    assert "task.retry_candidate()" in retry
    add_commit = source.split(
        "def _commit_manual_add", 1
    )[1].split("def _finish_add_task", 1)[0]
    assert "target_code = task.target_code" in add_commit
    assert "manual_edit_operations.prepare_manual_add(" in add_commit
    assert "manual_edit_commit.commit_manual_add(" in add_commit
    assert "keep_editing=bool(target_code == source_code or was_editable)" in add_commit
    assert 'batch_size = len(result.added)' in add_commit
    assert "task.record_add_batch(target_code, batch_size)" in add_commit
    assert "self._start_manual_capture()" in add_commit
    # State tests verify accumulated totals and per-batch candidate reset.
    finish = source.split(
        "def _finish_add_task", 1
    )[1].split("def _undo_current_edit", 1)[0]
    assert "本次提交" in finish
    assert "丢弃" in finish


def test_digitize_completion_defers_map_tool_replacement_until_next_qt_turn():
    source = (
        PLUGIN_ROOT / "refinement" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    completed = source.split(
        "def _manual_capture_completed", 1
    )[1].split("def _manual_capture_cancelled", 1)[0]
    assert "self._manual_tools.stop_capture" not in completed
    assert "self._start_manual_capture()" not in completed
    assert 'self._manual_tools.schedule_transition("restore")' in completed
    assert 'self._manual_tools.schedule_transition("restart")' in completed

    cancelled = source.split(
        "def _manual_capture_cancelled", 1
    )[1].split("def _manual_modify_selected_features", 1)[0]
    assert "self._manual_tools.stop_capture" not in cancelled
    assert 'self._manual_tools.schedule_transition("restore")' in cancelled
    # Native event-loop tests cover deferral and stale callbacks in the owner.


def test_saved_class_edits_restore_category_color_and_clear_selection():
    source = (
        PLUGIN_ROOT / "refinement" / "class_refinement_dialog.py"
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
    )[1].split("def _feature_by_object_id", 1)[0]
    assert "self._refresh_class_display(class_code)" in saved


def test_class_table_active_layer_visibility_and_row_actions_stay_synchronized():
    source = (
        PLUGIN_ROOT / "refinement" / "class_refinement_dialog.py"
    ).read_text(encoding="utf-8")
    panel = (
        PLUGIN_ROOT / "refinement" / "class_review_panel.py"
    ).read_text(encoding="utf-8")

    assert "ClassReviewPanel" in source
    assert "self.class_review_panel.class_selected.connect(self._select_class_context)" in source
    assert "self.class_review_panel.visibility_requested.connect(self._set_visible)" in source
    assert "self.class_review_panel.manual_requested.connect(self._open_manual_operations)" in source
    assert "self.class_review_panel.sam_requested.connect(self._request_sam)" in source
    assert "self.class_review_panel.confirm_requested.connect(self._confirm_class)" in source
    assert "self._table.setSelectionMode(SINGLE_SELECTION)" in panel
    assert "class_selected = pyqtSignal(int)" in panel
    assert "visibility_requested = pyqtSignal(int, bool)" in panel
    assert "self.iface.currentLayerChanged.connect" not in source
    assert 'getattr(self.iface, "currentLayerChanged", None)' in source
    assert "self._layer_signals.connect_current_layer(" in source
    assert "self._layer_signals.current_layer_changed.connect(" in source
    opener = source.split(
        "def _open_manual_operations", 1
    )[1].split("def _selection_changed", 1)[0]
    assert "self._select_class_context(class_code, activate_layer=True)" in opener
    load_layers = source.split(
        "def _load_workspace_layers", 1
    )[1].split("def _cancel_background_load", 1)[0]
    assert "self._layer_signals.bind_layer(" in load_layers
    assert "self._layer_signals.selection_changed.connect(" in source
    assert "self._layer_signals.edit_changed.connect(" in source
    assert "self._manual_tools.interrupted.connect(self._manual_tool_interrupted)" in source

    select_context = source.split(
        "def _select_class_context", 1
    )[1].split("def _current_class_code", 1)[0]
    assert "self._prioritize_class_load(" in select_context
    assert "self.layer_manager.set_layer_visibility(layer.id(), True)" in select_context
    assert "self.iface.setActiveLayer(layer)" in select_context

    active_changed = source.split(
        "def _active_layer_changed", 1
    )[1].split("def _current_class_code", 1)[0]
    assert "self._class_code_for_layer(layer)" in active_changed
    assert "activate_layer=False" in active_changed

    refresh = source.split("def _refresh_table", 1)[1].split(
        "def _class_selection_lock", 1
    )[0]
    assert "tree_layer.itemVisibilityChecked()" in refresh
    assert "visible = code in self._pending_visible_codes" in refresh

    cleanup = source.split("def cleanup", 1)[1].split("def closeEvent", 1)[0]
    assert "self._layer_signals.cleanup()" in cleanup
    assert "self._manual_tools.cleanup()" in cleanup
    assert "self._cancel_manual_task(silent=True)" in cleanup


def test_monitor_tables_use_stable_user_resizable_columns():
    pages = PLUGIN_ROOT / "monitor" / "pages"
    for filename in ("results.py", "detail.py"):
        source = (pages / filename).read_text(encoding="utf-8")
        assert "configure_adaptive_columns" in source
        assert ".setFixedHeight(" not in source
    widgets = (PLUGIN_ROOT / "monitor" / "monitor_widgets.py").read_text(encoding="utf-8")
    assert "header.setSectionResizeMode(column, INTERACTIVE)" in widgets
    assert "_adaptive_user_columns" in widgets
    assert "column not in self._adaptive_user_columns" in widgets
    # Scroll behavior and manual widths are verified against actual Qt tables.


def test_monitor_updates_large_tile_tables_incrementally_and_names_selected_stream():
    source = (PLUGIN_ROOT / "monitor" / "inference_monitor.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    monitor = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "InferenceMonitorDialog")
    progress = next(node for node in monitor.body if isinstance(node, ast.FunctionDef) and node.name == "_on_stream_progress")
    calls = {node.func.attr for node in ast.walk(progress) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert "update_live_tile" in calls
    assert "_render_selected_tiles" not in calls
    from labeling_tool.monitor.monitor_progress import stage_from_step, stream_from_step
    assert stream_from_step("subpixel_vectorize:model:alpha") == "model:alpha"
    assert stage_from_step("subpixel_vectorize:model:alpha") == "边界矢量化"
    # Native monitor_page_queries verifies live row reuse and selected-stream titles.


def test_legacy_annotation_group_is_not_left_empty():
    source = (
        PLUGIN_ROOT / "qgis_support" / "layer_manager.py"
    ).read_text(encoding="utf-8")
    block = source.split("def group_layers", 1)[1]
    assert block.index("candidates = []") < block.index("root.addGroup(group_name)")
    assert "if group is not None and not group.children():" in block
    assert "parent.removeChildNode(group)" in block


def test_semantic_colors_have_one_qgis4_source_of_truth():
    assert not (PLUGIN_ROOT / "styles" / "semantic_14class.qml").exists()
    assert not (PLUGIN_ROOT / "styles" / "sam_refined.qml").exists()

    style_path = PLUGIN_ROOT / "qgis_support" / "style_manager.py"
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
        PLUGIN_ROOT / "qgis_support" / "layer_manager.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(layer_manager)
    methods = {
        node.name: ast.get_source_segment(layer_manager, node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"load_result_stream", "load_workspace_class"}
    }
    assert "StyleManager.apply_categorized_style(polygons)" in methods["load_result_stream"]
    assert "StyleManager.apply_categorized_style(raw_polygons)" in methods["load_result_stream"]
    assert "StyleManager.apply_categorized_style(layer)" in methods["load_workspace_class"]
    assert all("load_sam_refined_polygons" not in value for value in methods.values())


def test_layer_manager_active_loaders_keep_style_group_stream_and_visibility_contract():
    source = (PLUGIN_ROOT / "qgis_support" / "layer_manager.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    methods = {
        node.name: ast.get_source_segment(source, node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }

    stream = methods["load_result_stream"]
    assert "StyleManager.apply_semantic_raster_style(mask)" in stream
    assert "StyleManager.apply_confidence_style(confidence)" in stream
    assert "self._add_managed_layer(mask, run_id, section, stream_id)" in stream
    assert "self._add_managed_layer(confidence, run_id, section, stream_id)" in stream
    assert "self._add_managed_layer(polygons, run_id, section, stream_id)" in stream
    assert 'loaded["mask"] = mask.id()' in stream
    assert 'loaded["confidence"] = confidence.id()' in stream
    assert 'loaded["polygons"] = polygons.id()' in stream

    workspace = methods["load_workspace_class"]
    assert "StyleManager.apply_categorized_style(layer)" in workspace
    assert 'self._add_managed_layer(layer, run_id, "Classes", f"class:{class_code}")' in workspace
    assert "setItemVisibilityChecked(bool(visible))" in workspace
    assert "labeling_tool/workspace_path" in workspace


def test_qgis4_rubber_bands_are_removed_from_the_canvas_scene():
    source = (PLUGIN_ROOT / "refinement/sam_map_preview.py").read_text(
        encoding="utf-8"
    )
    # QgsRubberBand is a canvas item, not a QObject. Native SAM scenarios also
    # check that replacement, retry and closing actually remove both overlays.
    assert "scene().removeItem(" in source
    assert "band.deleteLater()" not in source
