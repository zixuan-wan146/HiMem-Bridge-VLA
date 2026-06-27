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
from himem_bridge_vla.path_utils import display_project_path  # noqa: E402


DEFAULT_LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic LIBERO memory replay JSONL index.")
    parser.add_argument("--libero-root", default=None, help="Defaults to <AUTODL_TMP>/libero/datasets.")
    parser.add_argument("--output", required=True, help="JSONL replay index output path.")
    parser.add_argument("--manifest-output", default=None, help="Optional manifest JSON output path.")
    parser.add_argument("--suites", nargs="*", default=list(DEFAULT_LIBERO_SUITES))
    parser.add_argument("--action-horizon", type=int, default=DEFAULT_MEMORY_ACTION_HORIZON)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--short-offsets", nargs="+", type=int, default=list(DEFAULT_MEMORY_SHORT_OFFSETS))
    parser.add_argument("--executed-action-stride", type=int, default=DEFAULT_EXECUTED_ACTION_STRIDE)
    parser.add_argument(
        "--long-capacity",
        type=int,
        default=DEFAULT_MEMORY_LONG_CAPACITY,
        help="Deprecated compatibility flag. Must remain 0; long memory is trained by the progress-state planner.",
    )
    parser.add_argument("--include-tail", action="store_true")
    parser.add_argument("--max-episodes-per-suite", type=int, default=None)
    parser.add_argument("--max-episodes-per-task", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_episodes_per_suite is not None and args.max_episodes_per_suite <= 0:
        raise ValueError("--max-episodes-per-suite must be positive when provided")
    if args.max_episodes_per_task is not None and args.max_episodes_per_task <= 0:
        raise ValueError("--max-episodes-per-task must be positive when provided")
    if args.executed_action_stride <= 0:
        raise ValueError("--executed-action-stride must be positive")

    libero_root = resolve_libero_root(args.libero_root)
    rows = []
    suite_episode_counts: dict[str, int] = {}
    suite_sample_counts: dict[str, int] = {}
    task_episode_counts: dict[str, int] = {}
    task_sample_counts: dict[str, int] = {}
    episode_count = 0

    for episode in iter_libero_episodes(libero_root, suites=args.suites):
        if args.max_episodes_per_suite is not None:
            if suite_episode_counts.get(episode["suite"], 0) >= args.max_episodes_per_suite:
                continue
        task_key = f"{episode['suite']}:{episode['task_name']}"
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
        rows.extend(samples)
        episode_count += 1
        suite_episode_counts[episode["suite"]] = suite_episode_counts.get(episode["suite"], 0) + 1
        suite_sample_counts[episode["suite"]] = suite_sample_counts.get(episode["suite"], 0) + len(samples)
        task_episode_counts[task_key] = task_episode_counts.get(task_key, 0) + 1
        task_sample_counts[task_key] = task_sample_counts.get(task_key, 0) + len(samples)

    output_path = write_memory_replay_jsonl(args.output, rows)
    manifest = build_memory_replay_manifest(
        benchmark="LIBERO",
        action_horizon=args.action_horizon,
        stride=args.stride,
        short_offsets=args.short_offsets,
        executed_action_stride=args.executed_action_stride,
        long_capacity=args.long_capacity,
        include_tail=args.include_tail,
        sample_count=len(rows),
        episode_count=episode_count,
        task_counts=task_sample_counts,
    )
    manifest["libero_root"] = str(libero_root)
    manifest["libero_root_display"] = display_project_path(libero_root, REPO_ROOT)
    manifest["suites"] = list(args.suites)
    manifest["suite_episode_counts"] = dict(sorted(suite_episode_counts.items()))
    manifest["suite_sample_counts"] = dict(sorted(suite_sample_counts.items()))
    manifest["task_episode_counts"] = dict(sorted(task_episode_counts.items()))
    manifest["index_path"] = display_project_path(output_path, REPO_ROOT)

    manifest_output = Path(args.manifest_output).expanduser() if args.manifest_output else output_path.with_suffix(".manifest.json")
    write_json(manifest_output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


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


def _task_name_from_hdf5(path: Path) -> str:
    name = path.name
    if name.endswith("_demo.hdf5"):
        name = name[: -len("_demo.hdf5")]
    return name


def _demo_sort_key(name: str) -> tuple[int, str]:
    suffix = str(name).split("_")[-1]
    return (int(suffix), str(name)) if suffix.isdigit() else (10**9, str(name))


if __name__ == "__main__":
    raise SystemExit(main())
