from __future__ import annotations

from himem_bridge_vla.path_utils import find_repo_root
import os
import subprocess
from pathlib import Path
import sys


REPO_ROOT = find_repo_root(__file__)
SCRIPT = REPO_ROOT / "scripts" / "quality" / "check_repo.sh"


def run_dry_run(extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "HIMEM_CHECK_DRY_RUN": "1",
        "HIMEM_CHECK_SKIP_RUFF": "1",
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}",
        **(extra_env or {}),
    }
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_check_repo_dry_run_lists_default_gates():
    result = run_dry_run()

    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "DRY-RUN Requirements policy audit:" in output
    assert "DRY-RUN Runtime environment check:" in output
    assert "DRY-RUN Unit tests:" in output
    assert "Skipping ruff because HIMEM_CHECK_SKIP_RUFF=1" in output
    assert "DRY-RUN shell syntax:" in output
    assert "DRY-RUN Repository preflight:" in output
    assert "DRY-RUN Bridge-HiMem config validation:" in output
    assert "DRY-RUN Training config validation:" in output
    assert "DRY-RUN Benchmark inventory:" in output
    assert "DRY-RUN LIBERO setup dry-run:" in output
    assert "DRY-RUN LIBERO checkpoint download dry-run:" in output
    assert "DRY-RUN LIBERO smoke profile dry-run:" in output
    assert "DRY-RUN LIBERO eval profile dry-run:" in output
    assert "DRY-RUN RMBench eval dry-run:" in output
    assert "DRY-RUN RMBench eval plan-only:" in output
    assert "DRY-RUN Python compileall:" in output
    assert "DRY-RUN Git whitespace check:" in output
    assert str(REPO_ROOT) not in output


def test_check_repo_dry_run_respects_skip_flags():
    result = run_dry_run(
        {
            "HIMEM_CHECK_SKIP_PYTEST": "1",
            "HIMEM_CHECK_SKIP_COMPILE": "1",
        }
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "Skipping pytest because HIMEM_CHECK_SKIP_PYTEST=1" in output
    assert "Skipping compileall because HIMEM_CHECK_SKIP_COMPILE=1" in output
    assert "DRY-RUN Unit tests:" not in output
    assert "DRY-RUN Python compileall:" not in output


def test_check_repo_dry_run_accepts_python_override():
    python_name = Path(sys.executable).name
    result = run_dry_run({"PYTHON": python_name})

    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert f"DRY-RUN Unit tests: {python_name} -m pytest" in output
