from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ROOT_DOCS = {
    "ARCHITECTURE.md",
    "CURRENT_STATUS.md",
    "PROJECT_IDEA.md",
    "README.md",
}
LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def test_docs_root_is_a_small_current_entrypoint() -> None:
    actual = {path.name for path in DOCS.iterdir() if path.is_file()}
    assert actual == ROOT_DOCS
    for name in ROOT_DOCS:
        assert (DOCS / name).stat().st_size < 50_000


def test_local_markdown_links_resolve() -> None:
    failures: list[str] = []
    for document in sorted(DOCS.rglob("*.md")):
        text = document.read_text(encoding="utf-8")
        for raw_target in LINK_PATTERN.findall(text):
            target = raw_target.strip().split("#", 1)[0]
            if not target or target.startswith(("#", "http://", "https://", "mailto:", "file:")):
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                failures.append(
                    f"{document.relative_to(ROOT)} -> {raw_target}"
                )
    assert not failures, "broken local documentation links:\n" + "\n".join(failures)


def test_repository_readme_uses_the_docs_router() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/README.md" in readme
    assert "docs/plugin_plan_v3.md" not in readme
    assert "docs/IMPLEMENTATION_STATUS.md" not in readme


def test_research_archives_do_not_reappear_in_current_docs() -> None:
    current_paths = [
        DOCS / "README.md",
        DOCS / "ARCHITECTURE.md",
        DOCS / "CURRENT_STATUS.md",
        DOCS / "PROJECT_IDEA.md",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in current_paths)
    assert "rag_candidate" not in combined
    assert "spatial_joint_candidate" not in combined
