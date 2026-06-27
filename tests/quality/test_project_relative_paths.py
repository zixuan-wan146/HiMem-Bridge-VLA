from __future__ import annotations

from himem_bridge_vla.path_utils import find_repo_root
import re
from pathlib import Path


REPO_ROOT = find_repo_root(__file__)
CHECKED_IN_PATHS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "configs",
    REPO_ROOT / "docs",
    REPO_ROOT / "scripts",
    REPO_ROOT / "evaluations",
    REPO_ROOT / "src" / "himem_bridge_vla",
)
ABSOLUTE_PATH_PATTERN = re.compile(r"(^|[\s='\"(:])/(root|tmp|path|data|datasets|home)\b")


def iter_text_files(root: Path):
    if root.is_file():
        yield root
        return
    for path in root.rglob("*"):
        if "__pycache__" in path.parts or not path.is_file():
            continue
        if path.suffix in {".pyc", ".png", ".jpg", ".jpeg", ".gif", ".mp4"}:
            continue
        yield path


def test_checked_in_docs_configs_and_scripts_do_not_use_absolute_project_paths():
    offenders = []
    for root in CHECKED_IN_PATHS:
        if not root.exists():
            continue
        for path in iter_text_files(root):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if ABSOLUTE_PATH_PATTERN.search(line):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{line_number}: {line.strip()}")

    assert offenders == []
