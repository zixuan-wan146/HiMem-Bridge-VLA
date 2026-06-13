from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "setup_libero_env.sh"


def run_dry_run(extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "HIMEM_SETUP_LIBERO_DRY_RUN": "1",
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


def parse_key_values(stdout: str) -> dict[str, str]:
    values = {}
    for line in stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def test_setup_libero_dry_run_uses_repo_requirements_file(tmp_path: Path):
    result = run_dry_run({"HIMEM_DATA_ROOT": str(tmp_path / "data")})

    assert result.returncode == 0, result.stderr
    values = parse_key_values(result.stdout)
    assert values["HIMEM_DATA_ROOT"] == str(tmp_path / "data")
    assert values["LIBERO_ENV_PREFIX"] == str(tmp_path / "data" / "envs" / "libero")
    assert values["HIMEM_LIBERO_REQUIREMENTS"] == str(REPO_ROOT / "requirements-libero.txt")
    assert values["CONDA_BIN"] == "auto"


def test_setup_libero_dry_run_accepts_custom_requirements_file(tmp_path: Path):
    requirements_file = tmp_path / "custom-libero.txt"
    requirements_file.write_text("libero==0.1.1\n")

    result = run_dry_run(
        {
            "HIMEM_LIBERO_REQUIREMENTS": str(requirements_file),
            "LIBERO_ENV_PREFIX": str(tmp_path / "env"),
        }
    )

    assert result.returncode == 0, result.stderr
    values = parse_key_values(result.stdout)
    assert values["HIMEM_LIBERO_REQUIREMENTS"] == str(requirements_file)
    assert values["LIBERO_ENV_PREFIX"] == str(tmp_path / "env")
