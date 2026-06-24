#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from himem_bridge_vla.dataset.rmbench import DEFAULT_RMBENCH_ACTION_KEY  # noqa: E402
from himem_bridge_vla.dataset.rmbench import DEFAULT_RMBENCH_ROBOT_KEY  # noqa: E402
from himem_bridge_vla.dataset.rmbench import DEFAULT_RMBENCH_SETTING  # noqa: E402
from himem_bridge_vla.dataset.rmbench import compute_rmbench_normalization_result  # noqa: E402
from himem_bridge_vla.path_utils import display_project_path  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build min/max normalization stats for local RMBench HDF5 data.")
    parser.add_argument("--rmbench-root", default=None, help="Defaults to <AUTODL_TMP>/benchmarks/RMBench.")
    parser.add_argument("--output", required=True, help="Stats JSON output path.")
    parser.add_argument("--metadata-output", default=None, help="Optional metadata JSON output path.")
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional RMBench task names to include.")
    parser.add_argument("--setting", default=DEFAULT_RMBENCH_SETTING)
    parser.add_argument("--max-episodes-per-task", type=int, default=None)
    parser.add_argument("--robot-key", default=DEFAULT_RMBENCH_ROBOT_KEY)
    parser.add_argument("--action-key", default=DEFAULT_RMBENCH_ACTION_KEY)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rmbench_root = resolve_rmbench_root(args.rmbench_root)
    result = compute_rmbench_normalization_result(
        rmbench_root,
        tasks=args.tasks,
        setting=args.setting,
        max_episodes_per_task=args.max_episodes_per_task,
        robot_key=args.robot_key,
        action_key=args.action_key,
    )
    stats_output = write_json(args.output, result.stats)
    metadata_output = write_json(args.metadata_output, result.metadata) if args.metadata_output else None

    summary = {
        "stats_output": display_project_path(stats_output, REPO_ROOT),
        "metadata_output": display_project_path(metadata_output, REPO_ROOT) if metadata_output else None,
        **result.metadata,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
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


if __name__ == "__main__":
    raise SystemExit(main())
