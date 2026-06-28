from __future__ import annotations

from himem_bridge_vla.path_utils import find_repo_root
import os
import subprocess


REPO_ROOT = find_repo_root(__file__)
SCRIPT = REPO_ROOT / "scripts" / "eval" / "run_libero_parallel_eval.sh"


def run_parallel_script(extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
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


def test_parallel_eval_profile_splits_twenty_episodes_across_four_clients():
    result = run_parallel_script(
        {"HIMEM_LIBERO_PROFILE": "configs/runtime/libero_profiles/single_task_20_parallel.env"}
    )

    assert result.returncode == 0, result.stderr
    env = parse_env_output(result.stdout)
    assert env["HIMEM_LIBERO_TOTAL_EPISODES"] == "20"
    assert env["HIMEM_LIBERO_PARALLEL_CLIENTS"] == "4"
    assert env["CLIENT_0_HIMEM_LIBERO_EPISODES"] == "5"
    assert env["CLIENT_0_HIMEM_LIBERO_EPISODE_OFFSET"] == "0"
    assert env["CLIENT_1_HIMEM_LIBERO_EPISODES"] == "5"
    assert env["CLIENT_1_HIMEM_LIBERO_EPISODE_OFFSET"] == "5"
    assert env["CLIENT_2_HIMEM_LIBERO_EPISODES"] == "5"
    assert env["CLIENT_2_HIMEM_LIBERO_EPISODE_OFFSET"] == "10"
    assert env["CLIENT_3_HIMEM_LIBERO_EPISODES"] == "5"
    assert env["CLIENT_3_HIMEM_LIBERO_EPISODE_OFFSET"] == "15"


def test_parallel_eval_requires_total_episodes_and_parallel_clients():
    result = run_parallel_script(
        {
            "HIMEM_LIBERO_TOTAL_EPISODES": "",
            "HIMEM_LIBERO_PARALLEL_CLIENTS": "",
            "HIMEM_LIBERO_PROFILE": "",
        }
    )

    assert result.returncode == 2
    assert "HIMEM_LIBERO_TOTAL_EPISODES must be set" in result.stderr


def test_parallel_eval_distributes_remainder_without_repeating_offsets():
    result = run_parallel_script(
        {
            "HIMEM_LIBERO_TOTAL_EPISODES": "10",
            "HIMEM_LIBERO_PARALLEL_CLIENTS": "3",
            "HIMEM_LIBERO_EPISODE_OFFSET": "2",
            "HIMEM_LIBERO_CKPT_NAME": "parallel_test",
            "HIMEM_LIBERO_RUN_DIR": "run_outputs/libero_parallel_test",
        }
    )

    assert result.returncode == 0, result.stderr
    env = parse_env_output(result.stdout)
    assert env["CLIENT_0_HIMEM_LIBERO_EPISODES"] == "4"
    assert env["CLIENT_0_HIMEM_LIBERO_EPISODE_OFFSET"] == "2"
    assert env["CLIENT_1_HIMEM_LIBERO_EPISODES"] == "3"
    assert env["CLIENT_1_HIMEM_LIBERO_EPISODE_OFFSET"] == "6"
    assert env["CLIENT_2_HIMEM_LIBERO_EPISODES"] == "3"
    assert env["CLIENT_2_HIMEM_LIBERO_EPISODE_OFFSET"] == "9"
