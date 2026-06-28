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
from himem_bridge_vla.path_utils import display_project_path
from himem_bridge_vla.path_utils import find_repo_root


REPO_ROOT = find_repo_root(__file__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an episode-first LIBERO replay source index.")
    parser.add_argument("--libero-root", default=None, help="Defaults to <AUTODL_TMP>/libero/datasets.")
    parser.add_argument("--output", required=True, help="Episode replay JSON output path.")
    parser.add_argument("--suites", nargs="*", default=["libero_10"])
    parser.add_argument("--action-horizon", type=int, default=DEFAULT_MEMORY_ACTION_HORIZON)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--short-offsets", nargs="+", type=int, default=list(DEFAULT_MEMORY_SHORT_OFFSETS))
    parser.add_argument("--executed-action-stride", type=int, default=DEFAULT_EXECUTED_ACTION_STRIDE)
    parser.add_argument(
        "--long-capacity",
        type=int,
        default=DEFAULT_MEMORY_LONG_CAPACITY,
        help="Compatibility flag. Must remain 0; long memory is produced by the progress-state planner.",
    )
    parser.add_argument("--include-tail", action="store_true")
    parser.add_argument("--max-episodes-per-suite", type=int, default=None)
    parser.add_argument("--max-episodes-per-task", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.action_horizon <= 0:
        raise ValueError("--action-horizon must be positive")
    if args.executed_action_stride <= 0:
        raise ValueError("--executed-action-stride must be positive")
    if args.long_capacity != 0:
        raise ValueError("--long-capacity must remain 0")
    if args.max_episodes_per_suite is not None and args.max_episodes_per_suite <= 0:
        raise ValueError("--max-episodes-per-suite must be positive when provided")
    if args.max_episodes_per_task is not None and args.max_episodes_per_task <= 0:
        raise ValueError("--max-episodes-per-task must be positive when provided")

    libero_root = resolve_libero_root(args.libero_root)
    episodes = []
    suite_episode_counts: dict[str, int] = {}
    suite_node_counts: dict[str, int] = {}
    task_episode_counts: dict[str, int] = {}
    task_node_counts: dict[str, int] = {}
    total_required_visual_frames = 0
    total_nodes = 0

    for episode in iter_libero_episodes(libero_root, suites=args.suites):
        suite = str(episode["suite"])
        task_key = f"{suite}:{episode['task_name']}"
        if args.max_episodes_per_suite is not None:
            if suite_episode_counts.get(suite, 0) >= args.max_episodes_per_suite:
                continue
        if args.max_episodes_per_task is not None:
            if task_episode_counts.get(task_key, 0) >= args.max_episodes_per_task:
                continue

        samples = build_memory_replay_samples(
            episode_id=str(episode["episode_id"]),
            episode_length=int(episode["length"]),
            action_horizon=args.action_horizon,
            stride=args.stride,
            short_offsets=args.short_offsets,
            executed_action_stride=args.executed_action_stride,
            long_capacity=args.long_capacity,
            include_tail=args.include_tail,
            benchmark="LIBERO",
            task_name=str(episode["task_name"]),
            source_path=str(episode["source_path"]),
            episode_key=str(episode["demo_key"]),
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
        prompt = _read_libero_prompt(Path(episode["hdf5_path"]), fallback=str(episode["task_name"]).replace("_", " "))
        episodes.append(
            {
                "episode_id": str(episode["episode_id"]),
                "suite": suite,
                "task_name": str(episode["task_name"]),
                "prompt": prompt,
                "source_path": str(episode["source_path"]),
                "episode_key": str(episode["demo_key"]),
                "episode_length": int(episode["length"]),
                "node_count": len(node_payloads),
                "required_visual_steps": required_visual_steps,
                "required_visual_frame_count": len(required_visual_steps),
                "nodes": node_payloads,
            }
        )
        suite_episode_counts[suite] = suite_episode_counts.get(suite, 0) + 1
        suite_node_counts[suite] = suite_node_counts.get(suite, 0) + len(node_payloads)
        task_episode_counts[task_key] = task_episode_counts.get(task_key, 0) + 1
        task_node_counts[task_key] = task_node_counts.get(task_key, 0) + len(node_payloads)
        total_required_visual_frames += len(required_visual_steps)
        total_nodes += len(node_payloads)

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "libero_episode_replay_index",
        "version": 1,
        "benchmark": "LIBERO",
        "libero_root": str(libero_root),
        "libero_root_display": display_project_path(libero_root, REPO_ROOT),
        "suites": list(args.suites),
        "action_horizon": int(args.action_horizon),
        "stride": int(args.stride),
        "short_offsets": [int(offset) for offset in args.short_offsets],
        "executed_action_stride": int(args.executed_action_stride),
        "long_capacity": int(args.long_capacity),
        "include_tail": bool(args.include_tail),
        "episode_count": len(episodes),
        "node_count": total_nodes,
        "required_visual_frame_count": total_required_visual_frames,
        "suite_episode_counts": dict(sorted(suite_episode_counts.items())),
        "suite_node_counts": dict(sorted(suite_node_counts.items())),
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


def iter_libero_episodes(libero_root: str | Path, *, suites: list[str] | tuple[str, ...]):
    import h5py

    root = Path(libero_root).expanduser()
    for suite in suites:
        suite_root = root / suite
        if not suite_root.exists():
            continue
        for hdf5_path in sorted(suite_root.glob("*.hdf5")):
            task_name = _task_name_from_hdf5(hdf5_path)
            source_path = _relative_to(hdf5_path, root)
            with h5py.File(hdf5_path, "r") as handle:
                for demo_key in sorted(handle["data"].keys(), key=_demo_sort_key):
                    length = int(handle[f"data/{demo_key}/actions"].shape[0])
                    yield {
                        "suite": suite,
                        "task_name": task_name,
                        "hdf5_path": hdf5_path,
                        "source_path": source_path,
                        "demo_key": str(demo_key),
                        "episode_id": f"{suite}:{hdf5_path.stem}:{demo_key}",
                        "length": length,
                    }


def resolve_libero_root(value: str | None) -> Path:
    if value:
        return Path(value).expanduser()
    data_root = Path(os.environ.get("AUTODL_TMP", str(Path.home() / "autodl-tmp"))).expanduser()
    return data_root / "libero" / "datasets"


def _read_libero_prompt(hdf5_path: Path, *, fallback: str) -> str:
    try:
        import h5py

        with h5py.File(hdf5_path, "r") as handle:
            problem_info = handle["data"].attrs.get("problem_info")
        if isinstance(problem_info, bytes):
            problem_info = problem_info.decode("utf-8")
        if problem_info:
            payload = json.loads(str(problem_info))
            prompt = str(payload.get("language_instruction") or "").strip()
            if prompt:
                return prompt
    except (FileNotFoundError, KeyError, OSError, json.JSONDecodeError, ModuleNotFoundError):
        pass
    return str(fallback).strip()


def _relative_to(path: str | Path, root: Path) -> str:
    resolved = Path(path).expanduser()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def _task_name_from_hdf5(path: Path) -> str:
    name = path.name
    if name.endswith("_demo.hdf5"):
        name = name[: -len("_demo.hdf5")]
    return name


def _demo_sort_key(name: str) -> tuple[int, str]:
    suffix = str(name).split("_")[-1]
    return (int(suffix), str(name)) if suffix.isdigit() else (10**9, str(name))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
