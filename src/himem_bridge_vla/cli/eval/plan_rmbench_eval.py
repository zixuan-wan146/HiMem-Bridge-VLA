#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import sys
from himem_bridge_vla.path_utils import find_repo_root
from typing import Any, Sequence


REPO_ROOT = find_repo_root(__file__)
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from himem_bridge_vla.dataset.memory_replay import DEFAULT_MEMORY_ACTION_HORIZON  # noqa: E402
from himem_bridge_vla.cli.eval.inspect_benchmarks import DEFAULT_RMBENCH_TASKS  # noqa: E402


@dataclass(frozen=True)
class RMBenchEvalPlan:
    rmbench_root: Path
    output: Path
    manifest_output: Path
    policy_name: str
    policy_config: Path
    task_config: str
    tasks: tuple[str, ...]
    ckpt_setting: str
    seed: int
    gpu_id: str
    mode: str
    port: int | None
    python: str
    instruction_type: str
    action_horizon: int
    extra_overrides: tuple[str, ...]
    checks: dict[str, Any]
    task_step_limits: dict[str, int | None]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a reproducible RMBench eval command plan.")
    parser.add_argument("--rmbench-root", default=None, help="Defaults to <AUTODL_TMP>/benchmarks/RMBench.")
    parser.add_argument("--output", required=True, help="Markdown plan output path.")
    parser.add_argument("--manifest-output", default=None, help="JSON manifest output path.")
    parser.add_argument("--policy-name", default="HiMemBridgeVLA")
    parser.add_argument("--policy-config", default=None, help="Defaults to policy/<policy-name>/deploy_policy.yml.")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--tasks", nargs="*", default=list(DEFAULT_RMBENCH_TASKS))
    parser.add_argument("--ckpt-setting", default="himem_bridge_vla")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--mode", choices=("direct", "socket"), default="direct")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--python", default="python")
    parser.add_argument("--instruction-type", default="unseen")
    parser.add_argument("--action-horizon", type=int, default=DEFAULT_MEMORY_ACTION_HORIZON)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional RMBench eval override. May be repeated.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(args)
    write_plan(plan)
    print(plan.output)
    return 0


def build_plan(args: argparse.Namespace) -> RMBenchEvalPlan:
    rmbench_root = resolve_rmbench_root(args.rmbench_root)
    tasks = tuple(_normalize_tasks(args.tasks))
    policy_config = _resolve_policy_config(rmbench_root, args.policy_name, args.policy_config)
    output = Path(args.output).expanduser()
    manifest_output = Path(args.manifest_output).expanduser() if args.manifest_output else output.with_suffix(".json")
    if args.mode == "socket" and args.port is None:
        raise ValueError("--port is required for socket mode")
    if int(args.action_horizon) <= 0:
        raise ValueError("--action-horizon must be positive")

    task_step_limits = load_task_step_limits(rmbench_root, args.task_config, tasks)
    return RMBenchEvalPlan(
        rmbench_root=rmbench_root,
        output=output,
        manifest_output=manifest_output,
        policy_name=str(args.policy_name),
        policy_config=policy_config,
        task_config=str(args.task_config),
        tasks=tasks,
        ckpt_setting=str(args.ckpt_setting),
        seed=int(args.seed),
        gpu_id=str(args.gpu_id),
        mode=str(args.mode),
        port=int(args.port) if args.port is not None else None,
        python=str(args.python),
        instruction_type=str(args.instruction_type),
        action_horizon=int(args.action_horizon),
        extra_overrides=tuple(_parse_extra_overrides(args.override)),
        checks=build_checks(rmbench_root, policy_config, args.task_config, tasks),
        task_step_limits=task_step_limits,
    )


def write_plan(plan: RMBenchEvalPlan) -> tuple[Path, Path]:
    plan.output.parent.mkdir(parents=True, exist_ok=True)
    plan.output.write_text(format_plan(plan), encoding="utf-8")
    plan.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest = asdict(plan)
    manifest["rmbench_root"] = str(plan.rmbench_root)
    manifest["output"] = str(plan.output)
    manifest["manifest_output"] = str(plan.manifest_output)
    manifest["policy_config"] = str(plan.policy_config)
    manifest["commands"] = build_commands(plan)
    plan.manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan.output, plan.manifest_output


def format_plan(plan: RMBenchEvalPlan) -> str:
    commands = build_commands(plan)
    lines = [
        "# RMBench Eval Plan",
        "",
        "## Scope",
        "",
        f"- RMBench root: `{plan.rmbench_root}`",
        f"- Policy: `{plan.policy_name}`",
        f"- Policy config: `{plan.policy_config}`",
        f"- Task config: `{plan.task_config}`",
        f"- Mode: `{plan.mode}`",
        f"- Seed: `{plan.seed}`",
        f"- GPU id: `{plan.gpu_id}`",
        f"- Instruction type: `{plan.instruction_type}`",
        f"- Action horizon override: `{plan.action_horizon}`",
        "",
        "## Checks",
        "",
    ]
    for key, value in sorted(plan.checks.items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Task Step Limits", ""])
    for task in plan.tasks:
        lines.append(f"- `{task}`: `{plan.task_step_limits.get(task)}`")
    lines.extend(["", "## Commands", ""])
    if plan.mode == "socket":
        lines.extend(
            [
                "Start policy model server first:",
                "",
                "```bash",
                commands["server"],
                "```",
                "",
                "Then run eval clients:",
                "",
            ]
        )
        for task, command in commands["clients"].items():
            lines.extend([f"### {task}", "", "```bash", command, "```", ""])
    else:
        for task, command in commands["direct"].items():
            lines.extend([f"### {task}", "", "```bash", command, "```", ""])
    lines.extend(
        [
            "## Notes",
            "",
            "- RMBench official eval uses `test_num=100` inside `script/eval_policy.py` "
            "/ `script/eval_policy_client.py`.",
            "- Install the policy adapter first with "
            "`python scripts/setup/install_rmbench_policy_adapter.py --rmbench-root <RMBench> --force`.",
            "- The official policy API executes an action chunk and calls `TASK_ENV.take_action` "
            "for each low-level action.",
            "- `qpos` action shape is `[left_arm_joints + left_gripper + right_arm_joints + right_gripper]`, "
            "14 dims for the downloaded ALOHA setting.",
            "- This plan does not start simulation; run commands manually from a prepared RMBench environment.",
            "",
        ]
    )
    return "\n".join(lines)


def build_commands(plan: RMBenchEvalPlan) -> dict[str, Any]:
    if plan.mode == "socket":
        return {
            "server": _socket_server_command(plan),
            "clients": {task: _socket_client_command(plan, task) for task in plan.tasks},
        }
    return {"direct": {task: _direct_eval_command(plan, task) for task in plan.tasks}}


def build_checks(
    rmbench_root: Path,
    policy_config: Path,
    task_config: str,
    tasks: Sequence[str],
) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "rmbench_root_exists": rmbench_root.exists(),
        "eval_policy_py": (rmbench_root / "script" / "eval_policy.py").exists(),
        "eval_policy_client_py": (rmbench_root / "script" / "eval_policy_client.py").exists(),
        "policy_model_server_py": (rmbench_root / "script" / "policy_model_server.py").exists(),
        "task_config": (rmbench_root / "task_config" / f"{task_config}.yml").exists(),
        "eval_step_limit_config": (rmbench_root / "task_config" / "_eval_step_limit.yml").exists(),
        "policy_config": policy_config.exists(),
    }
    for task in tasks:
        checks[f"env_{task}"] = (rmbench_root / "envs" / f"{task}.py").exists()
        checks[f"data_{task}"] = (rmbench_root / "data" / task / "demo_clean" / "data").exists()
    return checks


def load_task_step_limits(rmbench_root: Path, task_config: str, tasks: Sequence[str]) -> dict[str, int | None]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyYAML is required to read RMBench task config") from exc

    step_limit_path = rmbench_root / "task_config" / "_eval_step_limit.yml"
    fallback_path = rmbench_root / "task_config" / f"{task_config}.yml"
    path = step_limit_path if step_limit_path.exists() else fallback_path
    if not path.exists():
        return {task: None for task in tasks}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {task: _optional_int(payload.get(task)) for task in tasks}


def resolve_rmbench_root(value: str | None) -> Path:
    if value:
        return Path(value).expanduser()
    data_root = Path(os.environ.get("AUTODL_TMP", str(Path.home() / "autodl-tmp"))).expanduser()
    return data_root / "benchmarks" / "RMBench"


def _resolve_policy_config(rmbench_root: Path, policy_name: str, value: str | None) -> Path:
    if value:
        path = Path(value).expanduser()
        return path if path.is_absolute() else rmbench_root / path
    return rmbench_root / "policy" / policy_name / "deploy_policy.yml"


def _direct_eval_command(plan: RMBenchEvalPlan, task: str) -> str:
    return _shell_join(
        [
            "cd",
            str(plan.rmbench_root),
            "&&",
            "env",
            f"CUDA_VISIBLE_DEVICES={plan.gpu_id}",
            "PYTHONWARNINGS=ignore::UserWarning",
            plan.python,
            "script/eval_policy.py",
            "--config",
            _relative_or_abs(plan.policy_config, plan.rmbench_root),
            "--overrides",
            *(_common_overrides(plan, task)),
        ]
    )


def _socket_server_command(plan: RMBenchEvalPlan) -> str:
    first_task = plan.tasks[0]
    return _shell_join(
        [
            "cd",
            str(plan.rmbench_root),
            "&&",
            "env",
            f"CUDA_VISIBLE_DEVICES={plan.gpu_id}",
            "PYTHONWARNINGS=ignore::UserWarning",
            plan.python,
            "script/policy_model_server.py",
            "--port",
            str(plan.port),
            "--config",
            _relative_or_abs(plan.policy_config, plan.rmbench_root),
            "--overrides",
            *(_common_overrides(plan, first_task)),
        ]
    )


def _socket_client_command(plan: RMBenchEvalPlan, task: str) -> str:
    return _shell_join(
        [
            "cd",
            str(plan.rmbench_root),
            "&&",
            "env",
            f"CUDA_VISIBLE_DEVICES={plan.gpu_id}",
            "PYTHONWARNINGS=ignore::UserWarning",
            plan.python,
            "script/eval_policy_client.py",
            "--port",
            str(plan.port),
            "--config",
            _relative_or_abs(plan.policy_config, plan.rmbench_root),
            "--overrides",
            *(_common_overrides(plan, task)),
        ]
    )


def _common_overrides(plan: RMBenchEvalPlan, task: str) -> tuple[str, ...]:
    return (
        "--task_name",
        task,
        "--task_config",
        plan.task_config,
        "--ckpt_setting",
        plan.ckpt_setting,
        "--seed",
        str(plan.seed),
        "--policy_name",
        plan.policy_name,
        "--instruction_type",
        plan.instruction_type,
        "--action_horizon",
        str(plan.action_horizon),
        *plan.extra_overrides,
    )


def _parse_extra_overrides(values: Sequence[str]) -> tuple[str, ...]:
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--override expects KEY=VALUE, got {value!r}")
        key, raw_value = value.split("=", 1)
        if not key:
            raise ValueError(f"--override has empty key: {value!r}")
        parsed.extend([f"--{key.lstrip('-')}", raw_value])
    return tuple(parsed)


def _normalize_tasks(tasks: Sequence[str]) -> tuple[str, ...]:
    if not tasks:
        raise ValueError("--tasks must contain at least one task")
    normalized = []
    for task in tasks:
        name = str(task)
        if name not in DEFAULT_RMBENCH_TASKS:
            raise ValueError(f"unsupported RMBench task {name!r}; expected one of {DEFAULT_RMBENCH_TASKS}")
        if name not in normalized:
            normalized.append(name)
    return tuple(normalized)


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _relative_or_abs(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _shell_join(parts: Sequence[str]) -> str:
    return " ".join(_quote_command_part(part) for part in parts)


def _quote_command_part(part: str) -> str:
    if part in {"&&"}:
        return part
    return shlex.quote(str(part))


if __name__ == "__main__":
    raise SystemExit(main())
