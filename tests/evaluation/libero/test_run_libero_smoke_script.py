from __future__ import annotations

from himem_bridge_vla.path_utils import find_repo_root
import os
import subprocess


REPO_ROOT = find_repo_root(__file__)
SCRIPT = REPO_ROOT / "scripts" / "eval" / "run_libero_smoke.sh"


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
    assert env["HIMEM_LIBERO_TRANSITION_REPLAN_ACTION_LIMIT"] == "0"
    assert env["HIMEM_LIBERO_TRANSITION_DATASET_NAME"] == ""
    assert env["HIMEM_LIBERO_CKPT_NAME"] == "HiMem_libero_smoke"
    assert env["HIMEM_LIBERO_RESULT_FILE"].endswith("HiMem_libero_smoke_results.json")
    assert env["HIMEM_LIBERO_MANIFEST_FILE"].endswith("HiMem_libero_smoke_run_manifest.json")


def test_run_libero_smoke_script_can_group_outputs_under_run_dir():
    run_dir = "run_outputs/libero_smoke_run"
    result = run_smoke_script({"HIMEM_LIBERO_RUN_DIR": run_dir})

    assert result.returncode == 0
    env = parse_env_output(result.stdout)
    assert env["HIMEM_LIBERO_RUN_DIR"] == run_dir
    assert env["HIMEM_LIBERO_LOG_DIR"] == f"{run_dir}/logs"
    assert env["HIMEM_LIBERO_VIDEO_DIR"] == f"{run_dir}/videos"
    assert env["HIMEM_LIBERO_LOG_FILE"] == f"{run_dir}/logs/HiMem_libero_smoke.txt"
    assert env["HIMEM_LIBERO_RESULT_FILE"] == f"{run_dir}/results/HiMem_libero_smoke_results.json"
    assert env["HIMEM_LIBERO_MANIFEST_FILE"] == f"{run_dir}/run_manifest.json"


def test_run_libero_smoke_script_loads_profile():
    profile = "configs/runtime/libero_profiles/smoke.env"

    result = run_smoke_script({"HIMEM_LIBERO_PROFILE": profile})

    assert result.returncode == 0, result.stderr
    env = parse_env_output(result.stdout)
    assert env["HIMEM_LIBERO_PROFILE"] == profile
    assert env["HIMEM_LIBERO_EPISODES"] == "1"
    assert env["HIMEM_LIBERO_TASK_SUITES"] == "libero_spatial"
    assert env["HIMEM_LIBERO_TASK_LIMIT"] == "1"
    assert env["HIMEM_LIBERO_MAX_STEPS"] == "1"
    assert env["HIMEM_LIBERO_HORIZON"] == "1"
    assert env["HIMEM_LIBERO_CKPT_NAME"] == "HiMem_libero_smoke"


def test_run_libero_smoke_script_keeps_explicit_env_over_profile():
    profile = "configs/runtime/libero_profiles/smoke.env"

    result = run_smoke_script(
        {
            "HIMEM_LIBERO_PROFILE": profile,
            "HIMEM_LIBERO_EPISODES": "7",
        }
    )

    assert result.returncode == 0, result.stderr
    env = parse_env_output(result.stdout)
    assert env["HIMEM_LIBERO_EPISODES"] == "7"


def test_run_libero_smoke_script_keeps_transition_dataset_override():
    result = run_smoke_script({"HIMEM_LIBERO_TRANSITION_DATASET_NAME": "robomme_four_tasks"})

    assert result.returncode == 0, result.stderr
    env = parse_env_output(result.stdout)
    assert env["HIMEM_LIBERO_TRANSITION_DATASET_NAME"] == "robomme_four_tasks"
    assert env["HIMEM_LIBERO_TRANSITION_TRACE_FILE"].endswith("HiMem_libero_smoke_transition_trace.jsonl")


def test_run_libero_smoke_script_rejects_unsupported_profile_key():
    profile = "tests/fixtures/configs/runtime/libero_profiles/unsupported_key.env"

    result = run_smoke_script({"HIMEM_LIBERO_PROFILE": profile})

    assert result.returncode != 0
    assert "unsupported key" in result.stderr
