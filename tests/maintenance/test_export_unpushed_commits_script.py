from __future__ import annotations

from himem_bridge_vla.path_utils import find_repo_root
import json
import os
import subprocess
from pathlib import Path
import sys


REPO_ROOT = find_repo_root(__file__)
SCRIPT = REPO_ROOT / "scripts" / "maintenance" / "export_unpushed_commits.sh"


def run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def create_repo_with_origin_main(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], text=True, capture_output=True, check=True)
    run_git(repo, "config", "user.name", "HiMem Test")
    run_git(repo, "config", "user.email", "himem-test@example.com")

    (repo / "tracked.txt").write_text("base\n")
    run_git(repo, "add", "tracked.txt")
    run_git(repo, "commit", "-m", "base")
    run_git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    return repo


def run_export(repo: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
    export_dir = out_dir.relative_to(repo).as_posix()
    env = {
        **os.environ,
        "HIMEM_EXPORT_REPO": str(repo),
        "HIMEM_EXPORT_DIR": export_dir,
        "PYTHON": sys.executable,
    }
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_export_unpushed_commits_writes_patch_bundle(tmp_path: Path):
    repo = create_repo_with_origin_main(tmp_path)
    out_dir = repo / "export"

    (repo / "tracked.txt").write_text("base\nfirst\n")
    run_git(repo, "add", "tracked.txt")
    run_git(repo, "commit", "-m", "Add first local change")
    (repo / "second.txt").write_text("second\n")
    run_git(repo, "add", "second.txt")
    run_git(repo, "commit", "-m", "Add second local change")

    result = run_export(repo, out_dir)

    assert result.returncode == 0, result.stderr
    patches = sorted((out_dir / "patches").glob("*.patch"))
    assert len(patches) == 2
    assert (out_dir / "README.md").exists()
    assert "git am exports/<export-name>/patches/*.patch" in (out_dir / "README.md").read_text()

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["base_ref"] == "origin/main"
    assert manifest["commit_count"] == 2
    assert manifest["dirty"] is False
    assert [commit["subject"] for commit in manifest["commits"]] == [
        "Add first local change",
        "Add second local change",
    ]
    assert len(manifest["patches"]) == 2


def test_export_unpushed_commits_fails_when_head_is_not_ahead(tmp_path: Path):
    repo = create_repo_with_origin_main(tmp_path)

    result = run_export(repo, repo / "export")

    assert result.returncode != 0
    assert "no commits to export" in result.stderr
