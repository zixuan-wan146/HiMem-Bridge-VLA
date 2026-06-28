from __future__ import annotations

from himem_bridge_vla.path_utils import find_repo_root
import os
import subprocess


REPO_ROOT = find_repo_root(__file__)
SCRIPT = REPO_ROOT / "scripts" / "eval" / "run_libero_eval.sh"


def run_eval_script(extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
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


def test_run_libero_eval_script_requires_profile_or_explicit_eval_config():
    result = run_eval_script(
        {
            "HIMEM_LIBERO_PROFILE": "",
            "HIMEM_LIBERO_EPISODES": "",
            "HIMEM_LIBERO_TASK_SUITES": "",
            "HIMEM_LIBERO_TASK_LIMIT": "",
            "HIMEM_LIBERO_EPISODE_OFFSET": "",
            "HIMEM_LIBERO_MAX_STEPS": "",
            "HIMEM_LIBERO_HORIZON": "",
            "HIMEM_LIBERO_CKPT_NAME": "",
            "HIMEM_MUJOCO_GL": "",
        }
    )

    assert result.returncode == 2
    assert "HIMEM_MUJOCO_GL must be set" in result.stderr


def test_run_libero_eval_script_loads_full_eval_profile():
    result = run_eval_script({"HIMEM_LIBERO_PROFILE": "configs/runtime/libero_profiles/full_eval.env"})

    assert result.returncode == 0
    env = parse_env_output(result.stdout)
    assert env["HIMEM_LIBERO_PROFILE"] == "configs/runtime/libero_profiles/full_eval.env"
    assert env["HIMEM_LIBERO_EPISODES"] == "10"
    assert env["HIMEM_LIBERO_TASK_SUITES"] == "libero_spatial,libero_object,libero_goal,libero_10"
    assert env["HIMEM_LIBERO_TASK_LIMIT"] == "0"
    assert env["HIMEM_LIBERO_EPISODE_OFFSET"] == "0"
    assert env["HIMEM_LIBERO_MAX_STEPS"] == "25,25,25,95"
    assert env["HIMEM_LIBERO_HORIZON"] == "16"
    assert env["HIMEM_LIBERO_CKPT_NAME"] == "HiMem_libero_eval"
    assert env["HIMEM_LIBERO_RESULT_FILE"].endswith("HiMem_libero_eval_results.json")
    assert env["HIMEM_LIBERO_MANIFEST_FILE"].endswith("HiMem_libero_eval_run_manifest.json")


def test_run_libero_eval_script_preserves_explicit_overrides():
    result = run_eval_script(
        {
            "HIMEM_LIBERO_PROFILE": "configs/runtime/libero_profiles/full_eval.env",
            "HIMEM_LIBERO_EPISODES": "2",
            "HIMEM_LIBERO_TASK_SUITES": "libero_spatial",
            "HIMEM_LIBERO_MAX_STEPS": "3",
            "HIMEM_LIBERO_CKPT_NAME": "custom_eval",
            "HIMEM_LIBERO_EPISODE_OFFSET": "5",
        }
    )

    assert result.returncode == 0
    env = parse_env_output(result.stdout)
    assert env["HIMEM_LIBERO_EPISODES"] == "2"
    assert env["HIMEM_LIBERO_TASK_SUITES"] == "libero_spatial"
    assert env["HIMEM_LIBERO_MAX_STEPS"] == "3"
    assert env["HIMEM_LIBERO_CKPT_NAME"] == "custom_eval"
    assert env["HIMEM_LIBERO_EPISODE_OFFSET"] == "5"
    assert env["HIMEM_LIBERO_RESULT_FILE"].endswith("custom_eval_results.json")


def test_run_libero_eval_script_can_group_outputs_under_run_dir():
    run_dir = "run_outputs/libero_eval_run"
    result = run_eval_script(
        {
            "HIMEM_LIBERO_PROFILE": "configs/runtime/libero_profiles/full_eval.env",
            "HIMEM_LIBERO_RUN_DIR": run_dir,
        }
    )

    assert result.returncode == 0
    env = parse_env_output(result.stdout)
    assert env["HIMEM_LIBERO_RUN_DIR"] == run_dir
    assert env["HIMEM_LIBERO_LOG_DIR"] == f"{run_dir}/logs"
    assert env["HIMEM_LIBERO_VIDEO_DIR"] == f"{run_dir}/videos"
    assert env["HIMEM_LIBERO_LOG_FILE"] == f"{run_dir}/logs/HiMem_libero_eval.txt"
    assert env["HIMEM_LIBERO_RESULT_FILE"] == f"{run_dir}/results/HiMem_libero_eval_results.json"
    assert env["HIMEM_LIBERO_MANIFEST_FILE"] == f"{run_dir}/run_manifest.json"


def test_run_libero_eval_script_loads_repo_relative_profile():
    result = run_eval_script({"HIMEM_LIBERO_PROFILE": "configs/runtime/libero_profiles/smoke.env"})

    assert result.returncode == 0, result.stderr
    env = parse_env_output(result.stdout)
    assert env["HIMEM_LIBERO_PROFILE"] == "configs/runtime/libero_profiles/smoke.env"
    assert env["HIMEM_LIBERO_EPISODES"] == "1"
    assert env["HIMEM_LIBERO_TASK_SUITES"] == "libero_spatial"
    assert env["HIMEM_LIBERO_CKPT_NAME"] == "HiMem_libero_smoke"
