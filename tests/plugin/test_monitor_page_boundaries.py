"""Headless architecture contracts; actual widgets are covered by native probes."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MONITOR = ROOT / "src/labeling_tool/monitor"


@pytest.mark.parametrize(
    ("module", "class_name", "base"),
    [
        ("overview", "OverviewPage", "QScrollArea"),
        ("detail", "DetailPage", "QWidget"),
        ("results", "ResultsPage", "QWidget"),
        ("events", "EventsPage", "QWidget"),
    ],
)
def test_pages_own_qt_views_without_window_or_runtime_dependencies(
    module, class_name, base
):
    tree = ast.parse((MONITOR / "pages" / f"{module}.py").read_text())
    page = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    assert [ast.unparse(value) for value in page.bases] == [base]
    forbidden = (
        "labeling_tool.monitor.inference_monitor",
        "labeling_tool.monitor.monitor_query_client",
        "labeling_tool.monitor.monitor_query_io",
        "labeling_tool.shared.state",
        "labeling_tool.runs",
        "loess_runtime",
    )
    imports = [
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    imports += [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not [name for name in imports if name.startswith(forbidden)]
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
        for node in page.body
    )


def test_window_has_no_forwarding_properties_or_copies_of_page_widgets():
    tree = ast.parse((MONITOR / "inference_monitor.py").read_text())
    window = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "InferenceMonitorDialog"
    )
    old_widgets = {
        "_overview_results",
        "_overview_splitter",
        "_overview_model_card",
        "_overview_spatial_card",
        "_overview_cards",
        "_overview_results_panel",
        "_streams",
        "_tiles",
        "_detail_navigation",
        "_detail_kind",
        "_detail_status",
        "_detail_search",
        "_object_tabs",
        "_object_current",
        "_object_models",
        "_object_attempts",
        "_object_events",
        "_object_more",
        "_assembly_steps",
        "_assembly_detail",
        "_result_coverage",
        "_results_splitter",
        "_history_scope",
        "_history_execution",
        "_history_target",
        "_history_search",
        "_history_table",
        "_history_detail",
        "_history_context_label",
        "_log_panel",
        "_log_toggle",
        "_history_load_older",
        "_warning_log_button",
        "_error_log_button",
    }
    references = {
        node.attr
        for node in ast.walk(window)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }
    methods = {node.name for node in window.body if isinstance(node, ast.FunctionDef)}
    assert not old_widgets & (references | methods)
    assert (
        not {
            "__getattr__",
            "_build_overview_page",
            "_build_detail_page",
            "_build_results_page",
            "_build_events_page",
        }
        & methods
    )
    assert {"overview_page", "detail_page", "results_page", "events_page"} <= references
    assert (
        not {"_selected_stream_id", "_selected_attempt", "_selected_object"}
        & references
    )


def test_log_page_public_ingress_keeps_explicit_parameters():
    tree = ast.parse((MONITOR / "pages/events.py").read_text())
    page = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EventsPage"
    )
    method = next(
        node
        for node in page.body
        if isinstance(node, ast.FunctionDef) and node.name == "append_log_event"
    )
    assert method.args.kwarg is None
    assert {"source", "severity", "fingerprint", "event_timestamp", "context_key"} <= {
        value.arg for value in method.args.kwonlyargs
    }
