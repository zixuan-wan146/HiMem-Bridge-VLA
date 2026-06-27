from __future__ import annotations

from himem_bridge_vla.path_utils import find_repo_root
from himem_bridge_vla.cli.eval import plan_rmbench_eval
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = find_repo_root(__file__)
SCRIPT = REPO_ROOT / "scripts" / "eval" / "plan_rmbench_eval.py"


def load_module():
    return plan_rmbench_eval


def test_build_rmbench_direct_eval_plan_writes_markdown_and_manifest(tmp_path):
    module = load_module()
    rmbench_root = _write_rmbench_root(tmp_path / "RMBench")
    output = tmp_path / "plan.md"
    args = module.parse_args(
        [
            "--rmbench-root",
            str(rmbench_root),
            "--output",
            str(output),
            "--tasks",
            "press_button",
            "swap_blocks",
            "--policy-name",
            "Your_Policy",
            "--ckpt-setting",
            "ckpt_a",
            "--seed",
            "7",
            "--gpu-id",
            "3",
            "--override",
            "checkpoint_path=/tmp/ckpt",
        ]
    )

    plan = module.build_plan(args)
    module.write_plan(plan)

    text = output.read_text(encoding="utf-8")
    manifest = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert "python script/eval_policy.py" in text
    assert "--task_name press_button" in text
    assert "--task_name swap_blocks" in text
    assert "CUDA_VISIBLE_DEVICES=3" in text
    assert "--checkpoint_path /tmp/ckpt" in text
    assert manifest["tasks"] == ["press_button", "swap_blocks"]
    assert manifest["task_step_limits"] == {"press_button": 1500, "swap_blocks": 1000}
    assert manifest["checks"]["data_press_button"] is True


def test_build_rmbench_socket_eval_plan_contains_server_and_client_commands(tmp_path):
    module = load_module()
    rmbench_root = _write_rmbench_root(tmp_path / "RMBench")
    output = tmp_path / "socket_plan.md"
    args = module.parse_args(
        [
            "--rmbench-root",
            str(rmbench_root),
            "--output",
            str(output),
            "--mode",
            "socket",
            "--port",
            "10001",
            "--tasks",
            "battery_try",
        ]
    )

    plan = module.build_plan(args)
    commands = module.build_commands(plan)
    module.write_plan(plan)

    assert "policy_model_server.py" in commands["server"]
    assert "--port 10001" in commands["server"]
    assert "eval_policy_client.py" in commands["clients"]["battery_try"]
    assert "--task_name battery_try" in output.read_text(encoding="utf-8")


def test_plan_rmbench_eval_cli_writes_plan(tmp_path):
    rmbench_root = _write_rmbench_root(tmp_path / "RMBench")
    output = tmp_path / "cli_plan.md"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--rmbench-root",
            str(rmbench_root),
            "--output",
            str(output),
            "--tasks",
            "observe_and_pickup",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert str(output) in completed.stdout
    assert output.exists()
    assert output.with_suffix(".json").exists()


def _write_rmbench_root(root: Path) -> Path:
    for path in (
        root / "script" / "eval_policy.py",
        root / "script" / "eval_policy_client.py",
        root / "script" / "policy_model_server.py",
        root / "policy" / "HiMemBridgeVLA" / "deploy_policy.yml",
        root / "policy" / "Your_Policy" / "deploy_policy.yml",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# test\n", encoding="utf-8")
    (root / "task_config").mkdir(parents=True, exist_ok=True)
    (root / "task_config" / "demo_clean.yml").write_text(
        "render_freq: 0\n",
        encoding="utf-8",
    )
    (root / "task_config" / "_eval_step_limit.yml").write_text(
        "\n".join(
            [
                "observe_and_pickup: 250",
                "press_button: 1500",
                "swap_blocks: 1000",
                "battery_try: 1000",
            ]
        ),
        encoding="utf-8",
    )
    for task in ("observe_and_pickup", "press_button", "swap_blocks", "battery_try"):
        (root / "envs").mkdir(parents=True, exist_ok=True)
        (root / "envs" / f"{task}.py").write_text("# env\n", encoding="utf-8")
        (root / "data" / task / "demo_clean" / "data").mkdir(parents=True, exist_ok=True)
    return root
