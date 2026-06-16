from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


DEFAULT_RMBENCH_TASKS = [
    "observe_and_pickup",
    "rearrange_blocks",
    "put_back_block",
    "swap_blocks",
    "swap_T",
    "blocks_ranking_try",
    "press_button",
    "cover_blocks",
    "battery_try",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RMBench HDF5 demo_clean trajectories into motion_boundary parquet."
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="RMBench repository or dataset root containing data/<task>/demo_clean.",
    )
    parser.add_argument("--output-root", required=True, help="Output motion_boundary dataset root.")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_RMBENCH_TASKS)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--include-terminal",
        action="store_true",
        help="Also include final language segment ends as terminal boundary events.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = resolve_input_root(Path(args.input_root).expanduser())
    output_root = Path(args.output_root).expanduser()
    annotation_dir = output_root / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)

    boundaries: list[dict[str, Any]] = []
    stats = {
        "input_root": str(input_root),
        "datasets": [],
        "episodes": 0,
        "frames": 0,
        "boundaries": 0,
        "terminal_events": 0,
        "duration_mismatches": 0,
    }

    for task_index, task in enumerate(args.tasks):
        task_stats = convert_task(
            input_root,
            output_root,
            task=task,
            task_index=task_index,
            max_episodes=args.max_episodes,
            include_terminal=bool(args.include_terminal),
            next_segment_id=lambda: len(boundaries),
            boundaries=boundaries,
        )
        stats["datasets"].append(task_stats)
        for key in ("episodes", "frames", "boundaries", "terminal_events", "duration_mismatches"):
            stats[key] += int(task_stats[key])

    boundary_path = annotation_dir / "boundaries.jsonl"
    with boundary_path.open("w") as f:
        for row in boundaries:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    (annotation_dir / "conversion_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
    print(json.dumps(stats, sort_keys=True))
    return 0


def resolve_input_root(path: Path) -> Path:
    if (path / "data").is_dir():
        return path
    if path.name == "data" and path.is_dir():
        return path.parent
    raise FileNotFoundError(f"expected RMBench root containing data/<task>/demo_clean: {path}")


def convert_task(
    input_root: Path,
    output_root: Path,
    *,
    task: str,
    task_index: int,
    max_episodes: int | None,
    include_terminal: bool,
    next_segment_id,
    boundaries: list[dict[str, Any]],
) -> dict[str, Any]:
    task_root = input_root / "data" / task / "demo_clean"
    data_dir = task_root / "data"
    language_path = task_root / "language_annotation.json"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"RMBench data directory not found: {data_dir}")
    if not language_path.exists():
        raise FileNotFoundError(f"RMBench language annotation not found: {language_path}")

    language_annotations = json.loads(language_path.read_text())
    episode_paths = sorted(data_dir.glob("episode*.hdf5"), key=episode_sort_key)
    if max_episodes is not None:
        episode_paths = episode_paths[:max_episodes]

    output_data_dir = output_root / "data" / task
    output_data_dir.mkdir(parents=True, exist_ok=True)

    task_stats = {
        "task": task,
        "task_index": task_index,
        "episodes": 0,
        "frames": 0,
        "boundaries": 0,
        "terminal_events": 0,
        "duration_mismatches": 0,
    }
    for episode_index, hdf5_path in enumerate(episode_paths):
        raw_episode_index = episode_sort_key(hdf5_path)[0]
        episode_id = f"episode_{raw_episode_index}"
        segments = language_annotations.get(episode_id)
        if segments is None:
            raise KeyError(f"{language_path} has no annotation for {episode_id}")
        instruction = read_instruction(task_root / "instructions" / f"episode{raw_episode_index}.json")
        rows, episode_boundaries, episode_stats = convert_episode(
            hdf5_path,
            episode_id=episode_id,
            task=task,
            task_index=task_index,
            episode_index=episode_index,
            language_segments=segments,
            global_instruction=instruction,
            include_terminal=include_terminal,
            next_segment_id=next_segment_id(),
        )
        if not rows:
            continue
        pd.DataFrame(rows).to_parquet(output_data_dir / f"{episode_id}.parquet", index=False)
        boundaries.extend(episode_boundaries)
        task_stats["episodes"] += 1
        for key, value in episode_stats.items():
            task_stats[key] += int(value)
    return task_stats


def convert_episode(
    hdf5_path: Path,
    *,
    episode_id: str,
    task: str,
    task_index: int,
    episode_index: int,
    language_segments: list[list[Any]],
    global_instruction: str,
    include_terminal: bool,
    next_segment_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    with h5py.File(hdf5_path, "r") as h5_file:
        action = np.asarray(h5_file["joint_action/vector"], dtype=np.float32)
        left_arm = np.asarray(h5_file["joint_action/left_arm"], dtype=np.float32)
        right_arm = np.asarray(h5_file["joint_action/right_arm"], dtype=np.float32)
        left_gripper_action = np.asarray(h5_file["joint_action/left_gripper"], dtype=np.float32).reshape(-1, 1)
        right_gripper_action = np.asarray(h5_file["joint_action/right_gripper"], dtype=np.float32).reshape(-1, 1)
        left_endpose = np.asarray(h5_file["endpose/left_endpose"], dtype=np.float32)
        right_endpose = np.asarray(h5_file["endpose/right_endpose"], dtype=np.float32)
        left_gripper = np.asarray(h5_file["endpose/left_gripper"], dtype=np.float32).reshape(-1, 1)
        right_gripper = np.asarray(h5_file["endpose/right_gripper"], dtype=np.float32).reshape(-1, 1)

    frame_count = int(action.shape[0])
    state = np.concatenate([left_endpose, right_endpose, left_gripper, right_gripper], axis=1).astype(np.float32)
    segment_ranges, duration_mismatch = build_segment_ranges(language_segments, frame_count)

    rows = []
    for frame_index in range(frame_count):
        subtask_id, subtask_name = subtask_at_frame(segment_ranges, frame_index)
        rows.append(
            {
                "episode_id": episode_id,
                "task": task,
                "task_index": task_index,
                "episode_index": episode_index,
                "frame_index": frame_index,
                "raw_frame_index": frame_index,
                "action": action[frame_index].reshape(-1).tolist(),
                "state": state[frame_index].reshape(-1).tolist(),
                "left_joint_action": left_arm[frame_index].reshape(-1).tolist(),
                "right_joint_action": right_arm[frame_index].reshape(-1).tolist(),
                "left_gripper_action": float(left_gripper_action[frame_index, 0]),
                "right_gripper_action": float(right_gripper_action[frame_index, 0]),
                "left_endpose": left_endpose[frame_index].reshape(-1).tolist(),
                "right_endpose": right_endpose[frame_index].reshape(-1).tolist(),
                "left_gripper": float(left_gripper[frame_index, 0]),
                "right_gripper": float(right_gripper[frame_index, 0]),
                "subtask_id": subtask_id,
                "subtask_name": subtask_name,
                "instruction": global_instruction,
            }
        )

    boundary_rows = []
    terminal_events = 0
    for segment_id, segment in enumerate(segment_ranges):
        is_terminal = segment_id == len(segment_ranges) - 1
        if is_terminal:
            terminal_events += 1
        if is_terminal and not include_terminal:
            continue
        boundary_rows.append(
            {
                "segment_id": next_segment_id + len(boundary_rows),
                "episode_id": episode_id,
                "task": task,
                "subtask_id": segment_id,
                "subtask_name": segment["name"],
                "start": int(segment["start"]),
                "end": int(segment["end"]),
                "is_terminal": bool(is_terminal),
                "label_source": "rmbench/language_annotation/duration",
            }
        )

    return (
        rows,
        boundary_rows,
        {
            "frames": frame_count,
            "boundaries": len(boundary_rows),
            "terminal_events": terminal_events,
            "duration_mismatches": int(duration_mismatch),
        },
    )


def build_segment_ranges(language_segments: list[list[Any]], frame_count: int) -> tuple[list[dict[str, Any]], bool]:
    ranges = []
    start = 0
    for raw_segment in language_segments:
        if len(raw_segment) < 2:
            raise ValueError(f"invalid language segment: {raw_segment!r}")
        name = str(raw_segment[0])
        duration = int(raw_segment[1])
        end = min(frame_count - 1, start + max(duration, 0) - 1)
        if end >= start:
            ranges.append({"name": name, "start": start, "end": end})
        start = end + 1
        if start >= frame_count:
            break
    if not ranges and frame_count > 0:
        ranges.append({"name": "", "start": 0, "end": frame_count - 1})
    if ranges and ranges[-1]["end"] < frame_count - 1:
        ranges[-1]["end"] = frame_count - 1
    duration_mismatch = start != frame_count
    return ranges, duration_mismatch


def subtask_at_frame(segment_ranges: list[dict[str, Any]], frame_index: int) -> tuple[int, str]:
    for index, segment in enumerate(segment_ranges):
        if int(segment["start"]) <= frame_index <= int(segment["end"]):
            return index, str(segment["name"])
    return max(0, len(segment_ranges) - 1), str(segment_ranges[-1]["name"]) if segment_ranges else ""


def read_instruction(path: Path) -> str:
    if not path.exists():
        return ""
    data = json.loads(path.read_text())
    for key in ("seen", "unseen"):
        value = data.get(key)
        if isinstance(value, list) and value:
            return str(value[0])
        if isinstance(value, str):
            return value
    return ""


def episode_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.startswith("episode"):
        suffix = stem[len("episode") :]
        if suffix.isdigit():
            return int(suffix), stem
    return 10**9, stem


if __name__ == "__main__":
    raise SystemExit(main())
