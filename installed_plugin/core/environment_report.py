"""Formatting helpers for environment-check summaries and full diagnostics."""

from __future__ import annotations


PROBLEM_STATUSES = ("error", "warning")


def compact_problem(check, max_chars=180):
    """Return one bounded line for the dock status area."""
    value = check.get("message") or check.get("value") or check.get("id") or "未知问题"
    lines = [line.strip() for line in str(value).splitlines() if line.strip()]
    summary = lines[0] if lines else "未知问题"
    summary = " ".join(summary.split())
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3].rstrip() + "..."
    return summary


def _format_check_details(checks, stderr="", statuses=None):
    blocks = []
    for check in checks:
        status = str(check.get("status") or "")
        if statuses is not None and status not in statuses:
            continue
        lines = [f"[{status.upper()}] {check.get('id') or 'unknown'}"]
        for label, key in (
            ("当前值", "value"),
            ("来源", "source"),
            ("修改位置", "fix"),
            ("完整信息", "message"),
        ):
            value = check.get(key)
            if value not in (None, ""):
                lines.append(f"{label}: {value}")
        blocks.append("\n".join(lines))

    joined = "\n\n".join(blocks)
    process_stderr = str(stderr or "").strip()
    if process_stderr and process_stderr not in joined:
        blocks.append("[PROCESS STDERR]\n" + process_stderr)

    return "\n\n".join(blocks) or "所有检查项均正常。"


def format_problem_details(checks, stderr=""):
    """Build selectable text containing only warnings and errors."""
    return _format_check_details(checks, stderr, PROBLEM_STATUSES)


def format_check_details(checks, stderr=""):
    """Build selectable text containing every environment check."""
    return _format_check_details(checks, stderr)
