from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_license_and_third_party_notices_are_present() -> None:
    root_license = (ROOT / "LICENSE").read_text(encoding="utf-8")
    plugin_license = (
        ROOT / "qgis_plugins" / "labeling_tool" / "LICENSE"
    ).read_text(encoding="utf-8")
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "GNU GENERAL PUBLIC LICENSE" in root_license
    assert "Version 3, 29 June 2007" in root_license
    assert plugin_license == root_license
    assert "Copyright (c) 2021 OpenAI" in notices
    assert "Copyright (c) 2026 tt-a1i" in notices


def test_public_metadata_and_readme_point_to_github() -> None:
    metadata = (
        ROOT / "qgis_plugins" / "labeling_tool" / "metadata.txt"
    ).read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "repository=https://github.com/anyun-hy/loess-qgis" in metadata
    assert "tracker=https://github.com/anyun-hy/loess-qgis/issues" in metadata
    assert "anyunhy@gmail.com" in metadata
    assert "English summary" in readme
    assert "不包含" in readme
    assert "TorchScript" in readme and "SAM3 checkpoint" in readme


def test_public_tree_excludes_internal_and_runtime_material() -> None:
    indexed = set(
        subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.splitlines()
    )
    tracked = {path for path in indexed if (ROOT / path).exists()}
    forbidden_prefixes = (
        "docs/archive/",
        "docs/handoffs/",
        ".vscode/",
        "weights/",
        "output/",
        "runtime/",
    )
    assert not [
        path for path in tracked if path.startswith(forbidden_prefixes)
    ]
    tracked_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (
            ROOT / "README.md",
            ROOT / "AGENTS.md",
            *(ROOT / "docs").rglob("*.md"),
            *(ROOT / "visualizations").glob("*.json"),
        )
    )
    assert not re.search(r"/Users/[A-Za-z0-9._-]+|/home/[A-Za-z0-9._-]+", tracked_text)


def test_public_quality_workflow_exists() -> None:
    workflow = ROOT / ".github" / "workflows" / "quality.yml"
    text = workflow.read_text(encoding="utf-8")
    assert "name: quality" in text
    assert "compileall" in text
    assert "bash -n" in text
    assert "test_public_release.py" in text
