#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from himem_bridge_vla.path_utils import find_repo_root


REPO_ROOT = find_repo_root(__file__)
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from himem_bridge_vla.dataset.memory_replay import DEFAULT_MEMORY_ACTION_HORIZON  # noqa: E402
from himem_bridge_vla.dataset.memory_replay import DEFAULT_EXECUTED_ACTION_STRIDE  # noqa: E402
from himem_bridge_vla.dataset.memory_replay import DEFAULT_MEMORY_LONG_CAPACITY  # noqa: E402
from himem_bridge_vla.dataset.memory_replay import DEFAULT_MEMORY_SHORT_OFFSETS  # noqa: E402
from himem_bridge_vla.dataset.memory_replay import build_memory_replay_manifest  # noqa: E402
from himem_bridge_vla.dataset.memory_replay import build_memory_replay_samples  # noqa: E402
from himem_bridge_vla.dataset.memory_replay import write_memory_replay_jsonl  # noqa: E402
from himem_bridge_vla.dataset.rmbench import DEFAULT_RMBENCH_SETTING  # noqa: E402
from himem_bridge_vla.dataset.rmbench import iter_rmbench_episode_files  # noqa: E402
from himem_bridge_vla.dataset.rmbench import read_rmbench_state_action_arrays  # noqa: E402
from himem_bridge_vla.path_utils import display_project_path  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic RMBench memory replay JSONL index.")
    parser.add_argument("--rmbench-root", default=None, help="Defaults to <AUTODL_TMP>/benchmarks/RMBench.")
    parser.add_argument("--output", required=True, help="JSONL replay index output path.")
    parser.add_argument("--manifest-output", default=None, help="Optional manifest JSON output path.")
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional RMBench task names to include.")
    parser.add_argument("--setting", default=DEFAULT_RMBENCH_SETTING)
    parser.add_argument("--action-horizon", type=int, default=DEFAULT_MEMORY_ACTION_HORIZON)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--short-offsets", nargs="+", type=int, default=list(DEFAULT_MEMORY_SHORT_OFFSETS))
    parser.add_argument("--executed-action-stride", type=int, default=DEFAULT_EXECUTED_ACTION_STRIDE)
    parser.add_argument(
        "--action-start-offset",
        type=int,
        default=1,
        help="Offset future action targets from current_step. RMBench qpos targets default to next-qpos.",
    )
    parser.add_argument(
        "--long-capacity",
        type=int,
        default=DEFAULT_MEMORY_LONG_CAPACITY,
        help="Deprecated compatibility flag. Must remain 0; long memory is trained by the progress-state planner.",
    )
    parser.add_argument("--include-tail", action="store_true")
    parser.add_argument("--max-episodes-per-task", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_episodes_per_task is not None and args.max_episodes_per_task <= 0:
        raise ValueError("--max-episodes-per-task must be positive when provided")
    if args.executed_action_stride <= 0:
        raise ValueError("--executed-action-stride must be positive")
    if args.action_start_offset < 0:
        raise ValueError("--action-start-offset must be non-negative")

    rmbench_root = resolve_rmbench_root(args.rmbench_root)
    rows = []
    task_episode_counts: dict[str, int] = {}
    task_sample_counts: dict[str, int] = {}
    episode_count = 0
    for episode in iter_rmbench_episode_files(rmbench_root, tasks=args.tasks, setting=args.setting):
        if args.max_episodes_per_task is not None:
            if task_episode_counts.get(episode.task_name, 0) >= args.max_episodes_per_task:
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
        rows.extend(samples)
        episode_count += 1
        task_episode_counts[episode.task_name] = task_episode_counts.get(episode.task_name, 0) + 1
        task_sample_counts[episode.task_name] = task_sample_counts.get(episode.task_name, 0) + len(samples)

    output_path = write_memory_replay_jsonl(args.output, rows)
    manifest = build_memory_replay_manifest(
        benchmark="RMBench",
        action_horizon=args.action_horizon,
        stride=args.stride,
        short_offsets=args.short_offsets,
        executed_action_stride=args.executed_action_stride,
        action_start_offset=args.action_start_offset,
        long_capacity=args.long_capacity,
        include_tail=args.include_tail,
        sample_count=len(rows),
        episode_count=episode_count,
        task_counts=task_sample_counts,
    )
    manifest["rmbench_root"] = str(rmbench_root)
    manifest["rmbench_root_display"] = display_project_path(rmbench_root, REPO_ROOT)
    manifest["setting"] = args.setting
    manifest["episode_counts"] = dict(sorted(task_episode_counts.items()))
    manifest["index_path"] = display_project_path(output_path, REPO_ROOT)

    manifest_output = Path(args.manifest_output).expanduser() if args.manifest_output else output_path.with_suffix(".manifest.json")
    write_json(manifest_output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def resolve_rmbench_root(value: str | None) -> Path:
    if value:
        return Path(value).expanduser()
    data_root = Path(os.environ.get("AUTODL_TMP", str(Path.home() / "autodl-tmp"))).expanduser()
    return data_root / "benchmarks" / "RMBench"


def write_json(path: str | Path, payload) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def _relative_to(path: str | Path, root: Path) -> str:
    resolved = Path(path).expanduser()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    raise SystemExit(main())
