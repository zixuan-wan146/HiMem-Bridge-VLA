from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_rmbench_eval.sh"


def run_eval_script(extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "HIMEM_RMBENCH_DRY_RUN": "1",
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


def test_run_rmbench_eval_script_uses_full_eval_defaults():
    result = run_eval_script()

    assert result.returncode == 0
    env = parse_env_output(result.stdout)
    assert env["HIMEM_RMBENCH_POLICY_NAME"] == "HiMemBridgeVLA"
    assert env["HIMEM_RMBENCH_TASK_CONFIG"] == "demo_clean"
    assert env["HIMEM_RMBENCH_CKPT_SETTING"] == "himem_bridge_vla"
    assert env["HIMEM_RMBENCH_ACTION_HORIZON"] == "32"
    assert env["HIMEM_RMBENCH_ACTION_DIM"] == "14"
    assert env["HIMEM_RMBENCH_ACTION_TYPE"] == "qpos"
    assert env["HIMEM_RMBENCH_STATE_SOURCE"] == "endpose"
    assert env["HIMEM_RMBENCH_ROBOT_KEY"] == "rmbench"
    assert env["HIMEM_RMBENCH_PLAN_ONLY"] == "0"
    assert env["HIMEM_RMBENCH_RUN_DIR"] == "run_outputs/rmbench_eval"
    assert env["HIMEM_RMBENCH_PLAN_FILE"] == "run_outputs/rmbench_eval/rmbench_eval_plan.md"
    assert env["HIMEM_RMBENCH_MANIFEST_FILE"] == "run_outputs/rmbench_eval/run_manifest.json"
    assert "press_button" in env["HIMEM_RMBENCH_TASKS"]
    assert env["HIMEM_SERVER_URI"] == "ws://127.0.0.1:9000"


def test_run_rmbench_eval_script_preserves_explicit_overrides():
    result = run_eval_script(
        {
            "HIMEM_RMBENCH_TASKS": "press_button",
            "HIMEM_RMBENCH_RUN_DIR": "run_outputs/rmbench_smoke",
            "HIMEM_RMBENCH_ACTION_HORIZON": "4",
            "HIMEM_RMBENCH_STATE_SOURCE": "qpos",
            "HIMEM_RMBENCH_PLAN_ONLY": "1",
            "HIMEM_SERVER_URI": "ws://127.0.0.1:9010",
        }
    )

    assert result.returncode == 0
    env = parse_env_output(result.stdout)
    assert env["HIMEM_RMBENCH_TASKS"] == "press_button"
    assert env["HIMEM_RMBENCH_RUN_DIR"] == "run_outputs/rmbench_smoke"
    assert env["HIMEM_RMBENCH_LOG_DIR"] == "run_outputs/rmbench_smoke/logs"
    assert env["HIMEM_RMBENCH_ACTION_HORIZON"] == "4"
    assert env["HIMEM_RMBENCH_STATE_SOURCE"] == "qpos"
    assert env["HIMEM_RMBENCH_PLAN_ONLY"] == "1"
    assert env["HIMEM_SERVER_URI"] == "ws://127.0.0.1:9010"
