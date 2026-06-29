#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from himem_bridge_vla.dataset.memory_replay import DEFAULT_EXECUTED_ACTION_STRIDE
from himem_bridge_vla.dataset.memory_replay import DEFAULT_MEMORY_ACTION_HORIZON
from himem_bridge_vla.dataset.memory_replay import DEFAULT_MEMORY_LONG_CAPACITY
from himem_bridge_vla.dataset.memory_replay import DEFAULT_MEMORY_SHORT_OFFSETS
from himem_bridge_vla.dataset.memory_replay import build_memory_replay_samples
from himem_bridge_vla.dataset.rmbench import DEFAULT_RMBENCH_SETTING
from himem_bridge_vla.dataset.rmbench import iter_rmbench_episode_files
from himem_bridge_vla.dataset.rmbench import read_rmbench_instruction
from himem_bridge_vla.dataset.rmbench import read_rmbench_state_action_arrays
from himem_bridge_vla.path_utils import display_project_path
from himem_bridge_vla.path_utils import find_repo_root


REPO_ROOT = find_repo_root(__file__)
RMBENCH_EPISODE_REPLAY_INDEX_FORMAT = "rmbench_episode_replay_index"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an episode-first RMBench replay source index.")
    parser.add_argument("--rmbench-root", default=None, help="Defaults to <AUTODL_TMP>/benchmarks/RMBench.")
    parser.add_argument("--output", required=True, help="Episode replay JSON output path.")
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional RMBench task names to include.")
    parser.add_argument("--setting", default=DEFAULT_RMBENCH_SETTING)
    parser.add_argument("--action-horizon", type=int, default=DEFAULT_MEMORY_ACTION_HORIZON)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--short-offsets", nargs="+", type=int, default=list(DEFAULT_MEMORY_SHORT_OFFSETS))
    parser.add_argument("--executed-action-stride", type=int, default=DEFAULT_EXECUTED_ACTION_STRIDE)
    parser.add_argument(
        "--action-start-offset",
        type=int,
        default=1,
        help="Offset future action targets from current_step. RMBench Stage1 uses next-qpos targets.",
    )
    parser.add_argument(
        "--long-capacity",
        type=int,
        default=DEFAULT_MEMORY_LONG_CAPACITY,
        help="Compatibility flag. Must remain 0; long memory is produced by the progress-state planner.",
    )
    parser.add_argument("--include-tail", action="store_true")
    parser.add_argument("--max-episodes-per-task", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)

    rmbench_root = resolve_rmbench_root(args.rmbench_root)
    episodes = []
    task_episode_counts: dict[str, int] = {}
    task_node_counts: dict[str, int] = {}
    total_nodes = 0
    total_required_visual_frames = 0

    for episode in iter_rmbench_episode_files(rmbench_root, tasks=args.tasks, setting=args.setting):
        if args.max_episodes_per_task is not None:
            if task_episode_counts.get(episode.task_name, 0) >= int(args.max_episodes_per_task):
                continue
        arrays = read_rmbench_state_action_arrays(episode.hdf5_path)
        relative_hdf5 = _relative_to(episode.hdf5_path, rmbench_root)
        relative_instruction = _relative_to(episode.instruction_path, rmbench_root) if episode.instruction_path else None
        samples = build_memory_replay_samples(
            episode_id=f"{episode.task_name}:{episode.hdf5_path.stem}",
            episode_length=int(arrays.actions.shape[0]),
            action_horizon=args.action_horizon,
            stride=args.stride,
            short_offsets=args.short_offsets,
            executed_action_stride=args.executed_action_stride,
            action_start_offset=args.action_start_offset,
            long_capacity=args.long_capacity,
            include_tail=args.include_tail,
            benchmark="RMBench",
            task_name=episode.task_name,
            source_path=relative_hdf5,
            instruction_path=relative_instruction,
        )
        if not samples:
            continue
        node_payloads = [_node_payload(sample.to_dict()) for sample in samples]
        required_visual_steps = sorted(
            {
                int(step)
                for node in node_payloads
                for step in node["required_visual_steps"]
            }
        )
        prompt = read_rmbench_instruction(episode.instruction_path) or episode.task_name.replace("_", " ")
        episodes.append(
            {
                "episode_id": f"{episode.task_name}:{episode.hdf5_path.stem}",
                "suite": "rmbench",
                "task_name": episode.task_name,
                "prompt": prompt,
                "source_path": relative_hdf5,
                "instruction_path": relative_instruction,
                "episode_length": int(arrays.actions.shape[0]),
                "node_count": len(node_payloads),
                "required_visual_steps": required_visual_steps,
                "required_visual_frame_count": len(required_visual_steps),
                "nodes": node_payloads,
            }
        )
        task_episode_counts[episode.task_name] = task_episode_counts.get(episode.task_name, 0) + 1
        task_node_counts[episode.task_name] = task_node_counts.get(episode.task_name, 0) + len(node_payloads)
        total_nodes += len(node_payloads)
        total_required_visual_frames += len(required_visual_steps)

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": RMBENCH_EPISODE_REPLAY_INDEX_FORMAT,
        "version": 1,
        "benchmark": "RMBench",
        "rmbench_root": str(rmbench_root),
        "rmbench_root_display": display_project_path(rmbench_root, REPO_ROOT),
        "tasks": list(args.tasks or sorted(task_episode_counts)),
        "setting": str(args.setting),
        "action_horizon": int(args.action_horizon),
        "stride": int(args.stride),
        "short_offsets": [int(offset) for offset in args.short_offsets],
        "executed_action_stride": int(args.executed_action_stride),
        "action_start_offset": int(args.action_start_offset),
        "long_capacity": int(args.long_capacity),
        "include_tail": bool(args.include_tail),
        "episode_count": len(episodes),
        "node_count": total_nodes,
        "required_visual_frame_count": total_required_visual_frames,
        "task_episode_counts": dict(sorted(task_episode_counts.items())),
        "task_node_counts": dict(sorted(task_node_counts.items())),
        "episodes": episodes,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = dict(payload)
    summary.pop("episodes")
    summary["output"] = display_project_path(output_path, REPO_ROOT)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.action_horizon <= 0:
        raise ValueError("--action-horizon must be positive")
    if args.executed_action_stride <= 0:
        raise ValueError("--executed-action-stride must be positive")
    if args.action_start_offset < 0:
        raise ValueError("--action-start-offset must be non-negative")
    if args.long_capacity != 0:
        raise ValueError("--long-capacity must remain 0")
    if args.max_episodes_per_task is not None and args.max_episodes_per_task <= 0:
        raise ValueError("--max-episodes-per-task must be positive when provided")


def _node_payload(row: dict[str, Any]) -> dict[str, Any]:
    current_step = int(row["current_step"])
    short_steps = [None if step is None else int(step) for step in row.get("short_steps", [])]
    short_mask = [bool(value) for value in row.get("short_mask", [])]
    visual_steps = [current_step]
    visual_steps.extend(step for step in short_steps if step is not None)
    return {
        "current_step": current_step,
        "current_visual_step": current_step,
        "short_visual_steps": short_steps,
        "short_mask": short_mask,
        "required_visual_steps": sorted(set(int(step) for step in visual_steps)),
        "executed_action_range": [int(row["executed_action_start"]), int(row["executed_action_end"])],
        "executed_action_valid_count": int(row["executed_action_valid_count"]),
        "future_action_range": [int(row["action_start"]), int(row["action_end"])],
        "action_valid_count": int(row["action_valid_count"]),
    }


def resolve_rmbench_root(value: str | None) -> Path:
    if value:
        return Path(value).expanduser()
    data_root = Path(os.environ.get("AUTODL_TMP", str(Path.home() / "autodl-tmp"))).expanduser()
    return data_root / "benchmarks" / "RMBench"


def _relative_to(path: str | Path, root: Path) -> str:
    resolved = Path(path).expanduser()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
