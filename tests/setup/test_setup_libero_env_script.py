from __future__ import annotations

from himem_bridge_vla.path_utils import find_repo_root
import os
import subprocess


REPO_ROOT = find_repo_root(__file__)
SCRIPT = REPO_ROOT / "scripts" / "setup" / "setup_libero_env.sh"


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


def test_setup_libero_dry_run_uses_repo_requirements_file():
    data_root = "run_outputs/libero_data_test"
    result = run_dry_run({"HIMEM_DATA_ROOT": data_root})

    assert result.returncode == 0, result.stderr
    values = parse_key_values(result.stdout)
    assert values["HIMEM_DATA_ROOT"] == data_root
    assert values["LIBERO_ENV_PREFIX"] == f"{data_root}/envs/libero"
    assert values["HIMEM_LIBERO_REQUIREMENTS"] == "requirements-libero.txt"
    assert values["CONDA_BIN"] == "auto"


def test_setup_libero_dry_run_accepts_custom_requirements_file():
    result = run_dry_run(
        {
            "HIMEM_LIBERO_REQUIREMENTS": "requirements-dev.txt",
            "LIBERO_ENV_PREFIX": "run_outputs/libero_env_test",
        }
    )

    assert result.returncode == 0, result.stderr
    values = parse_key_values(result.stdout)
    assert values["HIMEM_LIBERO_REQUIREMENTS"] == "requirements-dev.txt"
    assert values["LIBERO_ENV_PREFIX"] == "run_outputs/libero_env_test"
