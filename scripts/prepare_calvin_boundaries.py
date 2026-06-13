#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_auto_lang_ann(path: Path) -> dict[str, Any]:
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.shape == ():
        payload = payload.item()
    if not isinstance(payload, dict):
        raise ValueError(f"auto_lang_ann must contain a dict payload: {path}")
    return payload


def build_calvin_segments(payload: dict[str, Any], *, episode_id: str | None = None) -> list[dict[str, Any]]:
    language = payload.get("language")
    info = payload.get("info")
    if not isinstance(language, dict) or not isinstance(info, dict):
        raise ValueError("auto_lang_ann payload must contain language and info dictionaries")

    indices = np.asarray(info.get("indx"))
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError("info.indx must have shape [N, 2]")

    annotations = _as_list(language.get("ann"), len(indices), default="")
    tasks = _as_list(language.get("task"), len(indices), default=None)
    task_to_id = {task: idx for idx, task in enumerate(sorted({task for task in tasks if task is not None}))}

    segments = []
    for segment_id, (bounds, annotation, task) in enumerate(zip(indices.tolist(), annotations, tasks)):
        start, end = int(bounds[0]), int(bounds[1])
        if end < start:
            raise ValueError(f"segment {segment_id} has end < start: {start}, {end}")
        segments.append(
            {
                "segment_id": segment_id,
                "episode_id": episode_id,
                "start": start,
                "end": end,
                "task": None if task is None else str(task),
                "skill_id": None if task is None else int(task_to_id[task]),
                "language": str(annotation),
            }
        )
    return segments


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_lerobot_episode_segments(lerobot_root: Path) -> list[dict[str, Any]]:
    episodes_path = lerobot_root / "meta" / "episodes.jsonl"
    tasks_path = lerobot_root / "meta" / "tasks.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"missing LeRobot episodes metadata: {episodes_path}")
    if not tasks_path.exists():
        raise FileNotFoundError(f"missing LeRobot tasks metadata: {tasks_path}")

    task_to_index = {}
    for row in _read_jsonl(tasks_path):
        task = row.get("task")
        task_index = row.get("task_index")
        if task is not None and task_index is not None:
            task_to_index[str(task)] = int(task_index)

    segments = []
    for segment_id, row in enumerate(_read_jsonl(episodes_path)):
        episode_index = int(row.get("episode_index", segment_id))
        length = int(row["length"])
        tasks = row.get("tasks") or []
        task = str(tasks[0]) if tasks else None
        segments.append(
            {
                "segment_id": segment_id,
                "episode_id": str(episode_index),
                "start": 0,
                "end": max(0, length - 1),
                "task": task,
                "skill_id": None if task is None else task_to_index.get(task),
                "language": task or "",
            }
        )
    return segments


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CALVIN auto_lang_ann.npy segment boundaries to JSONL.")
    parser.add_argument(
        "--auto-lang-ann",
        default=None,
        help="Path to CALVIN lang_annotations/auto_lang_ann.npy.",
    )
    parser.add_argument(
        "--lerobot-root",
        default=None,
        help="LeRobot-format CALVIN root; uses meta/episodes.jsonl as per-episode segment boundaries.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path consumed by himem_bridge_vla/dataset/calvin_adapter.py.",
    )
    parser.add_argument(
        "--episode-id",
        default=None,
        help="Optional episode id if the sidecar should match per-episode frame_index instead of global indices.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if bool(args.auto_lang_ann) == bool(args.lerobot_root):
        raise ValueError("set exactly one of --auto-lang-ann or --lerobot-root")
    if args.auto_lang_ann:
        payload = load_auto_lang_ann(Path(args.auto_lang_ann).expanduser())
        segments = build_calvin_segments(payload, episode_id=args.episode_id)
    else:
        segments = build_lerobot_episode_segments(Path(args.lerobot_root).expanduser())
    output_path = Path(args.output).expanduser()
    write_jsonl(output_path, segments)
    print(f"[OK] wrote {len(segments)} CALVIN boundary segments to {output_path}")
    return 0


def _as_list(value: Any, expected_len: int, *, default: Any) -> list[Any]:
    if value is None:
        return [default] * expected_len
    items = np.asarray(value, dtype=object).tolist()
    if len(items) != expected_len:
        raise ValueError(f"expected {expected_len} values, got {len(items)}")
    return items


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        if not isinstance(row, dict):
            raise ValueError(f"{path} line {line_number} must be a JSON object")
        rows.append(row)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
