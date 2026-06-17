from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any



TIMESTEP_RE = re.compile(r"^timestep_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RoboMME H5 trajectories into transition_trigger segmented parquet."
    )
    parser.add_argument("--h5", help="Path to a RoboMME record_dataset_*.h5 file.")
    parser.add_argument("--task", help="Task name, e.g. StopCube.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="TASK=H5",
        help="Task/H5 pair. Can be passed multiple times, e.g. --dataset StopCube=/path/file.h5.",
    )
    parser.add_argument("--output-root", required=True, help="Output dataset root.")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--keep-video-demo", action="store_true", help="Keep conditioning video-demo frames.")
    parser.add_argument(
        "--include-terminal",
        action="store_true",
        help="Also add info/is_completed frames to the transition jsonl.",
    )
    return parser.parse_args()


def _load_runtime_dependencies() -> None:
    global h5py, np, pd
    try:
        import h5py as _h5py
        import numpy as _np
        import pandas as _pd
    except ModuleNotFoundError as exc:
        raise SystemExit(f"missing conversion dependency: {exc.name}") from exc
    h5py = _h5py
    np = _np
    pd = _pd

def main() -> int:
    args = parse_args()
    _load_runtime_dependencies()
    dataset_specs = _resolve_dataset_specs(args)
    output_root = Path(args.output_root).expanduser()
    annotation_dir = output_root / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)

    transitions = []
    stats = {
        "datasets": [],
        "episodes": 0,
        "frames": 0,
        "transitions": 0,
        "terminal_events": 0,
        "dropped_video_demo_frames": 0,
    }

    for task_index, (task, h5_path) in enumerate(dataset_specs):
        data_dir = output_root / "data" / task
        data_dir.mkdir(parents=True, exist_ok=True)
        task_stats = convert_task(
            h5_path,
            task=task,
            task_index=task_index,
            data_dir=data_dir,
            max_episodes=args.max_episodes,
            keep_video_demo=bool(args.keep_video_demo),
            include_terminal=bool(args.include_terminal),
            transitions=transitions,
        )
        stats["datasets"].append(task_stats)
        for key in ("episodes", "frames", "transitions", "terminal_events", "dropped_video_demo_frames"):
            stats[key] += int(task_stats[key])

    transition_path = annotation_dir / "transitions.jsonl"
    with transition_path.open("w") as f:
        for row in transitions:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    (annotation_dir / "conversion_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
    print(json.dumps(stats, sort_keys=True))
    return 0


def convert_task(
    h5_path: Path,
    *,
    task: str,
    task_index: int,
    data_dir: Path,
    max_episodes: int | None,
    keep_video_demo: bool,
    include_terminal: bool,
    transitions: list[dict[str, Any]],
) -> dict[str, Any]:
    task_stats = {
        "task": task,
        "task_index": task_index,
        "h5": str(h5_path),
        "episodes": 0,
        "frames": 0,
        "transitions": 0,
        "terminal_events": 0,
        "dropped_video_demo_frames": 0,
    }
    with h5py.File(h5_path, "r") as h5_file:
        episode_names = sorted(h5_file.keys(), key=_episode_sort_key)
        if max_episodes is not None:
            episode_names = episode_names[:max_episodes]
        for episode_index, episode_name in enumerate(episode_names):
            episode_group = h5_file[episode_name]
            rows, episode_transitions, episode_stats = convert_episode(
                episode_group,
                episode_id=episode_name,
                task=task,
                task_index=task_index,
                episode_index=episode_index,
                keep_video_demo=keep_video_demo,
                include_terminal=include_terminal,
                next_segment_id=len(transitions),
            )
            if not rows:
                continue
            pd.DataFrame(rows).to_parquet(data_dir / f"{episode_name}.parquet", index=False)
            transitions.extend(episode_transitions)
            task_stats["episodes"] += 1
            for key, value in episode_stats.items():
                task_stats[key] += value
    return task_stats


def convert_episode(
    episode_group: h5py.Group,
    *,
    episode_id: str,
    task: str,
    task_index: int,
    episode_index: int,
    keep_video_demo: bool,
    include_terminal: bool,
    next_segment_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    rows = []
    boundary_frames = []
    terminal_frames = []
    dropped_video_demo_frames = 0
    timestep_names = sorted(
        (name for name in episode_group.keys() if TIMESTEP_RE.match(name)),
        key=lambda name: int(TIMESTEP_RE.match(name).group(1)),  # type: ignore[union-attr]
    )
    for timestep_name in timestep_names:
        timestep = episode_group[timestep_name]
        raw_frame_index = int(TIMESTEP_RE.match(timestep_name).group(1))  # type: ignore[union-attr]
        info = timestep["info"]
        is_video_demo = bool(_read_scalar(info["is_video_demo"]))
        if is_video_demo and not keep_video_demo:
            dropped_video_demo_frames += 1
            continue

        frame_index = len(rows)
        is_boundary = bool(_read_scalar(info["is_subgoal_boundary"]))
        is_completed = bool(_read_scalar(info["is_completed"]))
        if is_boundary:
            boundary_frames.append(frame_index)
        if is_completed:
            terminal_frames.append(frame_index)

        obs = timestep["obs"]
        action_group = timestep["action"]
        rows.append(
            {
                "episode_id": episode_id,
                "task": task,
                "task_index": task_index,
                "episode_index": episode_index,
                "frame_index": frame_index,
                "raw_frame_index": raw_frame_index,
                "action": _as_float_list(action_group["eef_action"]),
                "joint_action": _as_float_list(action_group["joint_action"]),
                "state": _concat_float_list(
                    obs["eef_state"],
                    obs["joint_state"],
                    obs["gripper_state"],
                    np.asarray([float(_read_scalar(obs["is_gripper_close"]))], dtype=np.float32),
                ),
                "eef_state": _as_float_list(obs["eef_state"]),
                "joint_state": _as_float_list(obs["joint_state"]),
                "gripper_state": _as_float_list(obs["gripper_state"]),
                "is_gripper_close": bool(_read_scalar(obs["is_gripper_close"])),
                "is_subgoal_boundary": is_boundary,
                "is_completed": is_completed,
                "simple_subgoal_online": _read_text(info["simple_subgoal_online"]),
                "grounded_subgoal_online": _read_text(info["grounded_subgoal_online"]),
            }
        )

    segment_rows = []
    start = 0
    for event_frame in boundary_frames:
        if event_frame < start:
            continue
        subgoal_name = rows[event_frame].get("simple_subgoal_online") if rows else None
        segment_rows.append(
            {
                "segment_id": next_segment_id + len(segment_rows),
                "episode_id": episode_id,
                "task": task,
                "subtask_id": len(segment_rows),
                "subtask_name": subgoal_name,
                "start": start,
                "end": int(event_frame),
                "is_terminal": False,
                "label_source": "robomme/info/is_subgoal_boundary",
            }
        )
        start = event_frame + 1

    if include_terminal:
        for event_frame in terminal_frames:
            if event_frame not in boundary_frames:
                segment_rows.append(
                    {
                        "segment_id": next_segment_id + len(segment_rows),
                        "episode_id": episode_id,
                        "task": task,
                        "subtask_id": len(segment_rows),
                        "subtask_name": "completed",
                        "start": min(start, event_frame),
                        "end": int(event_frame),
                        "is_terminal": True,
                        "label_source": "robomme/info/is_completed",
                    }
                )

    return (
        rows,
        segment_rows,
        {
            "frames": len(rows),
            "transitions": len(boundary_frames),
            "terminal_events": len(terminal_frames),
            "dropped_video_demo_frames": dropped_video_demo_frames,
        },
    )


def _episode_sort_key(name: str) -> tuple[int, str]:
    if name.startswith("episode_"):
        try:
            return (int(name.split("_", 1)[1]), name)
        except ValueError:
            pass
    return (10**9, name)


def _resolve_dataset_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    specs = []
    if args.dataset:
        if args.h5 or args.task:
            raise ValueError("use either --dataset TASK=H5 pairs or the legacy --task/--h5 pair, not both")
        for raw_spec in args.dataset:
            if "=" not in raw_spec:
                raise ValueError(f"--dataset must use TASK=H5 format: {raw_spec!r}")
            task, raw_path = raw_spec.split("=", 1)
            task = task.strip()
            if not task:
                raise ValueError(f"--dataset has empty task name: {raw_spec!r}")
            specs.append((task, Path(raw_path).expanduser()))
    else:
        if not args.h5 or not args.task:
            raise ValueError("either pass --dataset TASK=H5 at least once or pass both --task and --h5")
        specs.append((str(args.task), Path(args.h5).expanduser()))

    for task, h5_path in specs:
        if not h5_path.exists():
            raise FileNotFoundError(f"{task} H5 not found: {h5_path}")
    return specs


def _read_scalar(dataset: h5py.Dataset) -> Any:
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_text(dataset: h5py.Dataset) -> str:
    value = _read_scalar(dataset)
    if value is None:
        return ""
    return str(value)


def _as_float_list(value: h5py.Dataset | np.ndarray) -> list[float]:
    return np.asarray(value, dtype=np.float32).reshape(-1).tolist()


def _concat_float_list(*values: h5py.Dataset | np.ndarray) -> list[float]:
    arrays = [np.asarray(value, dtype=np.float32).reshape(-1) for value in values]
    return np.concatenate(arrays).astype(np.float32).tolist()


if __name__ == "__main__":
    raise SystemExit(main())
