from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_libero_smoke.sh"


def run_smoke_script(extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "HIMEM_LIBERO_DRY_RUN": "1",
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


def parse_env_output(stdout: str) -> dict[str, str]:
    result = {}
    for line in stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def test_run_libero_smoke_script_uses_minimal_smoke_defaults():
    result = run_smoke_script()

    assert result.returncode == 0
    env = parse_env_output(result.stdout)
    assert env["HIMEM_LIBERO_EPISODES"] == "1"
    assert env["HIMEM_LIBERO_TASK_SUITES"] == "libero_spatial"
    assert env["HIMEM_LIBERO_TASK_LIMIT"] == "1"
    assert env["HIMEM_LIBERO_MAX_STEPS"] == "1"
    assert env["HIMEM_LIBERO_HORIZON"] == "1"
    assert env["HIMEM_LIBERO_CKPT_NAME"] == "HiMem_libero_smoke"
    assert env["HIMEM_LIBERO_RESULT_FILE"].endswith("HiMem_libero_smoke_results.json")
    assert env["HIMEM_LIBERO_MANIFEST_FILE"].endswith("HiMem_libero_smoke_run_manifest.json")


def test_run_libero_smoke_script_can_group_outputs_under_run_dir(tmp_path):
    run_dir = tmp_path / "libero_smoke_run"
    result = run_smoke_script({"HIMEM_LIBERO_RUN_DIR": str(run_dir)})

    assert result.returncode == 0
    env = parse_env_output(result.stdout)
    assert env["HIMEM_LIBERO_RUN_DIR"] == str(run_dir)
    assert env["HIMEM_LIBERO_LOG_DIR"] == str(run_dir / "logs")
    assert env["HIMEM_LIBERO_VIDEO_DIR"] == str(run_dir / "videos")
    assert env["HIMEM_LIBERO_LOG_FILE"] == str(run_dir / "logs" / "HiMem_libero_smoke.txt")
    assert env["HIMEM_LIBERO_RESULT_FILE"] == str(
        run_dir / "results" / "HiMem_libero_smoke_results.json"
    )
    assert env["HIMEM_LIBERO_MANIFEST_FILE"] == str(run_dir / "run_manifest.json")


def test_run_libero_smoke_script_loads_profile(tmp_path):
    profile = tmp_path / "custom_smoke.env"
    profile.write_text(
        "\n".join(
            [
                "HIMEM_LIBERO_EPISODES=3",
                "HIMEM_LIBERO_TASK_SUITES=libero_goal",
                "HIMEM_LIBERO_TASK_LIMIT=2",
                "HIMEM_LIBERO_MAX_STEPS=5",
                "HIMEM_LIBERO_HORIZON=2",
                "HIMEM_LIBERO_CKPT_NAME=profile_smoke",
                "",
            ]
        )
    )

    result = run_smoke_script({"HIMEM_LIBERO_PROFILE": str(profile)})

    assert result.returncode == 0, result.stderr
    env = parse_env_output(result.stdout)
    assert env["HIMEM_LIBERO_PROFILE"] == str(profile)
    assert env["HIMEM_LIBERO_EPISODES"] == "3"
    assert env["HIMEM_LIBERO_TASK_SUITES"] == "libero_goal"
    assert env["HIMEM_LIBERO_TASK_LIMIT"] == "2"
    assert env["HIMEM_LIBERO_MAX_STEPS"] == "5"
    assert env["HIMEM_LIBERO_HORIZON"] == "2"
    assert env["HIMEM_LIBERO_CKPT_NAME"] == "profile_smoke"


def test_run_libero_smoke_script_keeps_explicit_env_over_profile(tmp_path):
    profile = tmp_path / "custom_smoke.env"
    profile.write_text("HIMEM_LIBERO_EPISODES=3\n")

    result = run_smoke_script(
        {
            "HIMEM_LIBERO_PROFILE": str(profile),
            "HIMEM_LIBERO_EPISODES": "7",
        }
    )

    assert result.returncode == 0, result.stderr
    env = parse_env_output(result.stdout)
    assert env["HIMEM_LIBERO_EPISODES"] == "7"


def test_run_libero_smoke_script_rejects_unsupported_profile_key(tmp_path):
    profile = tmp_path / "bad.env"
    profile.write_text("HF_TOKEN=secret\n")

    result = run_smoke_script({"HIMEM_LIBERO_PROFILE": str(profile)})

    assert result.returncode != 0
    assert "unsupported key" in result.stderr
