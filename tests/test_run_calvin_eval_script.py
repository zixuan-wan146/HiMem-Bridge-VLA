from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_calvin_eval.sh"


def run_eval_script(extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "HIMEM_CALVIN_DRY_RUN": "1",
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


def test_run_calvin_eval_script_uses_full_eval_defaults():
    result = run_eval_script()

    assert result.returncode == 0, result.stderr
    env = parse_env_output(result.stdout)
    assert env["CALVIN_ROOT"] == "/root/autodl-tmp/calvin"
    assert env["HIMEM_CALVIN_NUM_SEQUENCES"] == "1000"
    assert env["HIMEM_CALVIN_MAX_STEPS_PER_SUBTASK"] == "360"
    assert env["HIMEM_CALVIN_HORIZON"] == "14"
    assert env["HIMEM_CALVIN_CKPT_NAME"] == "HiMem_calvin_eval"
    assert env["HIMEM_CALVIN_GRIPPER_MODE"] == "openvla"
    assert env["HIMEM_CALVIN_RESET_MEMORY_SCOPE"] == "sequence"
    assert env["HIMEM_CALVIN_RESULT_FILE"].endswith("HiMem_calvin_eval_results.json")
    assert env["HIMEM_CALVIN_MANIFEST_FILE"].endswith("HiMem_calvin_eval_run_manifest.json")


def test_run_calvin_eval_script_preserves_explicit_overrides():
    result = run_eval_script(
        {
            "HIMEM_CALVIN_NUM_SEQUENCES": "2",
            "HIMEM_CALVIN_MAX_STEPS_PER_SUBTASK": "3",
            "HIMEM_CALVIN_CKPT_NAME": "custom_calvin",
            "HIMEM_CALVIN_GRIPPER_MODE": "passthrough",
        }
    )

    assert result.returncode == 0, result.stderr
    env = parse_env_output(result.stdout)
    assert env["HIMEM_CALVIN_NUM_SEQUENCES"] == "2"
    assert env["HIMEM_CALVIN_MAX_STEPS_PER_SUBTASK"] == "3"
    assert env["HIMEM_CALVIN_CKPT_NAME"] == "custom_calvin"
    assert env["HIMEM_CALVIN_GRIPPER_MODE"] == "passthrough"
    assert env["HIMEM_CALVIN_RESULT_FILE"].endswith("custom_calvin_results.json")


def test_run_calvin_eval_script_can_group_outputs_under_run_dir(tmp_path):
    run_dir = tmp_path / "calvin_eval_run"
    result = run_eval_script({"HIMEM_CALVIN_RUN_DIR": str(run_dir)})

    assert result.returncode == 0, result.stderr
    env = parse_env_output(result.stdout)
    assert env["HIMEM_CALVIN_RUN_DIR"] == str(run_dir)
    assert env["HIMEM_CALVIN_LOG_DIR"] == str(run_dir / "logs")
    assert env["HIMEM_CALVIN_VIDEO_DIR"] == str(run_dir / "videos")
    assert env["HIMEM_CALVIN_LOG_FILE"] == str(run_dir / "logs" / "HiMem_calvin_eval.txt")
    assert env["HIMEM_CALVIN_RESULT_FILE"] == str(
        run_dir / "results" / "HiMem_calvin_eval_results.json"
    )
    assert env["HIMEM_CALVIN_MANIFEST_FILE"] == str(run_dir / "run_manifest.json")


def test_run_calvin_eval_script_loads_repo_relative_profile():
    result = run_eval_script({"HIMEM_CALVIN_PROFILE": "configs/calvin_profiles/smoke.env"})

    assert result.returncode == 0, result.stderr
    env = parse_env_output(result.stdout)
    assert env["HIMEM_CALVIN_PROFILE"] == str(REPO_ROOT / "configs" / "calvin_profiles" / "smoke.env")
    assert env["HIMEM_CALVIN_NUM_SEQUENCES"] == "1"
    assert env["HIMEM_CALVIN_MAX_STEPS_PER_SUBTASK"] == "1"
    assert env["HIMEM_CALVIN_HORIZON"] == "1"
    assert env["HIMEM_CALVIN_CKPT_NAME"] == "HiMem_calvin_smoke"
