from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, RandomSampler, Sampler, WeightedRandomSampler


@dataclass(frozen=True)
class BoundarySegment:
    segment_id: int
    start: int
    end: int
    episode_id: str | None = None
    task: str | None = None


@dataclass
class WindowRecord:
    trajectory_id: str
    task_id: int | None
    frame_index: int
    event_frame: int
    features: np.ndarray
    label: float
    valid: float
    group: str
    distance_to_boundary: int | None


class MotionBoundaryDataset(Dataset):
    def __init__(self, records: list[WindowRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "features": torch.tensor(record.features, dtype=torch.float32),
            "label": torch.tensor(record.label, dtype=torch.float32),
            "valid": torch.tensor(record.valid, dtype=torch.float32),
            "trajectory_id": record.trajectory_id,
            "task_id": -1 if record.task_id is None else record.task_id,
            "frame_index": record.frame_index,
            "event_frame": record.event_frame,
            "group": record.group,
        }

    @property
    def input_dim(self) -> int:
        if not self.records:
            raise ValueError("dataset has no records")
        return int(self.records[0].features.shape[-1])

    def labels_and_mask(self) -> tuple[torch.Tensor, torch.Tensor]:
        labels = torch.tensor([record.label for record in self.records], dtype=torch.float32)
        mask = torch.tensor([record.valid for record in self.records], dtype=torch.float32)
        return labels, mask


def build_datasets(config: dict[str, Any]) -> tuple[MotionBoundaryDataset, MotionBoundaryDataset]:
    data_config = config["data"]
    data_format = str(data_config.get("format", "segmented_parquet"))
    if data_format != "segmented_parquet":
        raise ValueError(f"unsupported motion_boundary data.format: {data_format!r}")
    records = build_segmented_parquet_records(data_config, config["features"])
    if not records:
        raise ValueError("no motion boundary records were built")
    return split_records(
        records,
        float(data_config.get("val_fraction", 0.1)),
        int(config.get("seed", 42)),
        split_by=str(data_config.get("split_by", "trajectory")),
    )


def build_segmented_parquet_records(data_config: dict[str, Any], feature_config: dict[str, Any]) -> list[WindowRecord]:
    root = Path(data_config["root"]).expanduser()
    boundary_path = Path(data_config["boundary_jsonl"]).expanduser()
    segments = load_boundary_segments(boundary_path)
    global_events = sorted(segment.end for segment in segments if segment.episode_id is None)
    episode_events = defaultdict(list)
    legacy_episode_events = defaultdict(list)
    for segment in segments:
        if segment.episode_id is not None:
            episode_id = str(segment.episode_id)
            if segment.task is None:
                legacy_episode_events[episode_id].append(segment.end)
            else:
                episode_events[(str(segment.task), episode_id)].append(segment.end)

    records: list[WindowRecord] = []
    for parquet_path in sorted(root.glob("data/*/*.parquet")):
        df = pd.read_parquet(parquet_path)
        if len(df) == 0:
            continue
        trajectory_id = f"{parquet_path.parent.name}/{parquet_path.stem}"
        episode_id = _episode_id(df.iloc[0], parquet_path, data_config)
        task_name = _task_name(df.iloc[0], parquet_path)
        task_id = _task_id(df.iloc[0])
        actions = _stack_column(df, data_config.get("action_keys", ["rel_actions", "action", "actions"]))
        states = _stack_column(df, data_config.get("state_keys", ["robot_obs", "observation.state", "state"]))
        frame_indices = _frame_indices(df, data_config)
        global_frame_indices = _global_frame_indices(df, data_config)
        events = episode_events.get((task_name, str(episode_id)))
        if not events:
            events = legacy_episode_events.get(str(episode_id))
        if not events:
            events = [event for event in global_events if frame_indices[0] <= event <= frame_indices[-1]]
            if global_frame_indices is not None:
                events = [
                    frame_indices[int(np.argmin(np.abs(global_frame_indices - event)))]
                    for event in global_events
                    if global_frame_indices[0] <= event <= global_frame_indices[-1]
                ]
        if not events:
            continue
        records.extend(
            _build_records_for_trajectory(
                trajectory_id,
                task_id,
                frame_indices,
                actions,
                states,
                events,
                data_config,
                feature_config,
            )
        )
    return records


def load_boundary_segments(path: Path) -> list[BoundarySegment]:
    if not path.exists():
        raise FileNotFoundError(f"boundary jsonl not found: {path}")
    segments = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        start = int(row.get("start", row.get("segment_start")))
        end = int(row.get("end", row.get("segment_end")))
        if end < start:
            raise ValueError(f"{path}:{line_number} has end < start")
        segments.append(
            BoundarySegment(
                segment_id=int(row.get("segment_id", len(segments))),
                start=start,
                end=end,
                episode_id=None if row.get("episode_id") is None else str(row.get("episode_id")),
                task=None if row.get("task") is None else str(row.get("task")),
            )
        )
    return segments


def split_records(
    records: list[WindowRecord],
    val_fraction: float,
    seed: int,
    *,
    split_by: str,
) -> tuple[MotionBoundaryDataset, MotionBoundaryDataset]:
    if split_by == "trajectory":
        return split_by_trajectory(records, val_fraction, seed)
    if split_by == "task":
        return split_by_task(records, val_fraction, seed)
    raise ValueError("data.split_by must be 'trajectory' or 'task'")


def split_by_trajectory(
    records: list[WindowRecord],
    val_fraction: float,
    seed: int,
) -> tuple[MotionBoundaryDataset, MotionBoundaryDataset]:
    trajectory_ids = sorted({record.trajectory_id for record in records})
    rng = random.Random(seed)
    rng.shuffle(trajectory_ids)
    val_count = max(1, int(round(len(trajectory_ids) * val_fraction))) if len(trajectory_ids) > 1 else 0
    val_ids = set(trajectory_ids[:val_count])
    train_records = [record for record in records if record.trajectory_id not in val_ids]
    val_records = [record for record in records if record.trajectory_id in val_ids]
    if not val_records:
        val_records = train_records
    return MotionBoundaryDataset(train_records), MotionBoundaryDataset(val_records)


def split_by_task(
    records: list[WindowRecord],
    val_fraction: float,
    seed: int,
) -> tuple[MotionBoundaryDataset, MotionBoundaryDataset]:
    task_ids = sorted({record.task_id for record in records if record.task_id is not None})
    if not task_ids:
        raise ValueError("task split requested, but records do not contain task_id")
    rng = random.Random(seed)
    rng.shuffle(task_ids)
    val_count = max(1, int(round(len(task_ids) * val_fraction))) if len(task_ids) > 1 else 0
    val_ids = set(task_ids[:val_count])
    train_records = [record for record in records if record.task_id not in val_ids]
    val_records = [record for record in records if record.task_id in val_ids]
    if not train_records or not val_records:
        raise ValueError("task split produced an empty train or validation set")
    return MotionBoundaryDataset(train_records), MotionBoundaryDataset(val_records)


def make_training_sampler(dataset: MotionBoundaryDataset, config: dict[str, Any]) -> tuple[Sampler[int] | None, bool]:
    """Return a sampler and whether DataLoader should shuffle.

    ``balanced`` keeps the original event-centered sampling policy. ``natural`` samples
    records uniformly from the built dataset, preserving the real positive/negative ratio.
    """

    sampler_mode = str(config["training"].get("sampler", "balanced"))
    if sampler_mode == "balanced":
        return make_balanced_sampler(dataset, config), False
    if sampler_mode == "natural":
        epoch_size = int(config["training"].get("epoch_size") or 0)
        if epoch_size > 0:
            return RandomSampler(dataset, replacement=True, num_samples=epoch_size), False
        return None, True
    raise ValueError("training.sampler must be 'balanced' or 'natural'")


def make_balanced_sampler(dataset: MotionBoundaryDataset, config: dict[str, Any]) -> WeightedRandomSampler:
    positive_ratio = float(config["training"].get("positive_ratio", 0.5))
    hard_ratio = float(config["training"].get("hard_negative_ratio", 0.25))
    easy_ratio = max(0.0, 1.0 - positive_ratio - hard_ratio)
    groups = {"positive": [], "hard_negative": [], "easy_negative": []}
    for index, record in enumerate(dataset.records):
        if record.valid <= 0:
            continue
        groups.setdefault(record.group, []).append(index)

    weights = torch.zeros(len(dataset), dtype=torch.double)
    for group_name, ratio in (
        ("positive", positive_ratio),
        ("hard_negative", hard_ratio),
        ("easy_negative", easy_ratio),
    ):
        indices = groups.get(group_name, [])
        if not indices:
            continue
        group_weight = ratio / len(indices)
        for index in indices:
            weights[index] = group_weight
    if weights.sum().item() == 0:
        weights.fill_(1.0)
    num_samples = int(config["training"].get("epoch_size") or len(dataset))
    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)


def _build_records_for_trajectory(
    trajectory_id: str,
    task_id: int | None,
    frame_indices: np.ndarray,
    actions: np.ndarray,
    states: np.ndarray,
    events: list[int],
    data_config: dict[str, Any],
    feature_config: dict[str, Any],
) -> list[WindowRecord]:
    window_size = int(data_config.get("window_size", 32))
    label_window = resolve_label_window(data_config)
    hard_negative_radius = int(data_config.get("hard_negative_radius", 30))
    label_sigma = float(data_config.get("label_sigma", 2.0))
    soft_labels = bool(data_config.get("soft_labels", True))
    features = build_features(actions, states, feature_config)
    if bool(feature_config.get("normalize", True)):
        features = normalize_features(features)

    records: list[WindowRecord] = []
    for row_index in range(window_size - 1, len(features)):
        frame_index = int(frame_indices[row_index])
        distance = _nearest_event_distance(frame_index, events)
        if distance is None:
            continue
        abs_distance = abs(distance)
        if -label_window["positive_pre"] <= distance <= label_window["positive_post"]:
            label = float(np.exp(-abs_distance / label_sigma)) if soft_labels else 1.0
            valid = 1.0
            group = "positive"
        elif -label_window["ignore_pre"] <= distance <= label_window["ignore_post"]:
            label = 0.0
            valid = 0.0
            group = "ignore"
        elif abs_distance <= hard_negative_radius:
            label = 0.0
            valid = 1.0
            group = "hard_negative"
        else:
            label = 0.0
            valid = 1.0
            group = "easy_negative"
        records.append(
            WindowRecord(
                trajectory_id=trajectory_id,
                task_id=task_id,
                frame_index=frame_index,
                event_frame=frame_index - distance,
                features=features[row_index - window_size + 1 : row_index + 1].astype(np.float32),
                label=label,
                valid=valid,
                group=group,
                distance_to_boundary=distance,
            )
        )
    return records


def resolve_label_window(data_config: dict[str, Any]) -> dict[str, int]:
    """Resolve signed label windows around an event.

    Distance is ``frame_index - event_frame``. Negative means before the event,
    positive means after the event.
    """

    mode = str(data_config.get("label_mode", "symmetric"))
    positive_radius = int(data_config.get("positive_radius", 2))
    ignore_radius = int(data_config.get("ignore_radius", 6))
    presets = {
        "symmetric": (positive_radius, positive_radius, ignore_radius, ignore_radius),
        "event_only": (0, 0, ignore_radius, ignore_radius),
        "pre1_post2": (1, 2, max(ignore_radius, 1), max(ignore_radius, 2)),
        "post_only": (0, 3, ignore_radius, max(ignore_radius, 3)),
        "custom": (positive_radius, positive_radius, ignore_radius, ignore_radius),
    }
    if mode not in presets:
        raise ValueError(
            "data.label_mode must be one of "
            "'symmetric', 'event_only', 'pre1_post2', 'post_only', or 'custom'"
        )
    positive_pre, positive_post, ignore_pre, ignore_post = presets[mode]

    positive_pre = _optional_int(data_config.get("positive_pre_frames"), positive_pre)
    positive_post = _optional_int(data_config.get("positive_post_frames"), positive_post)
    ignore_pre = _optional_int(data_config.get("ignore_pre_frames"), ignore_pre)
    ignore_post = _optional_int(data_config.get("ignore_post_frames"), ignore_post)

    if positive_pre < 0 or positive_post < 0 or ignore_pre < 0 or ignore_post < 0:
        raise ValueError("label window sizes must be non-negative")
    if ignore_pre < positive_pre or ignore_post < positive_post:
        raise ValueError("ignore window must cover the positive window")
    return {
        "positive_pre": positive_pre,
        "positive_post": positive_post,
        "ignore_pre": ignore_pre,
        "ignore_post": ignore_post,
    }


def _optional_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


def build_features(actions: np.ndarray, states: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    parts = []
    if bool(config.get("use_action", True)):
        parts.append(actions)
    if bool(config.get("use_state", True)):
        parts.append(states)
    if bool(config.get("use_delta_action", True)):
        parts.append(_delta(actions))
    if bool(config.get("use_delta_state", True)):
        parts.append(_delta(states))
    if bool(config.get("use_gripper_transition", True)) and actions.shape[1] >= 1:
        gripper = actions[:, -1:]
        parts.append(_delta(gripper))
    if not parts:
        raise ValueError("at least one feature source must be enabled")
    return np.concatenate(parts, axis=1)


def normalize_features(features: np.ndarray) -> np.ndarray:
    mean = np.nanmean(features, axis=0, keepdims=True)
    std = np.nanstd(features, axis=0, keepdims=True)
    return (features - mean) / np.maximum(std, 1e-6)


def _stack_column(df: pd.DataFrame, keys: list[str]) -> np.ndarray:
    key = _first_existing_key(df, keys)
    values = [np.asarray(value, dtype=np.float32).reshape(-1) for value in df[key].tolist()]
    return np.stack(values).astype(np.float32)


def _first_existing_key(df: pd.DataFrame, keys: list[str]) -> str:
    for key in keys:
        if key in df.columns:
            return key
    raise KeyError(f"none of the configured keys are present: {tuple(keys)}")


def _frame_indices(df: pd.DataFrame, data_config: dict[str, Any]) -> np.ndarray:
    for key in data_config.get("frame_keys", []):
        if key in df.columns:
            return df[key].to_numpy(dtype=np.int64)
    return np.arange(len(df), dtype=np.int64)


def _global_frame_indices(df: pd.DataFrame, data_config: dict[str, Any]) -> np.ndarray | None:
    for key in data_config.get("global_frame_keys", []):
        if key in df.columns:
            return df[key].to_numpy(dtype=np.int64)
    return None


def _episode_id(row: pd.Series, parquet_path: Path, data_config: dict[str, Any]) -> str:
    for key in data_config.get("episode_keys", []):
        if key in row and row[key] is not None:
            return str(row[key])
    return f"{parquet_path.parent.name}/{parquet_path.stem}"


def _task_name(row: pd.Series, parquet_path: Path) -> str:
    if "task" in row and row["task"] is not None:
        return str(row["task"])
    return parquet_path.parent.name


def _task_id(row: pd.Series) -> int | None:
    if "task_index" not in row or row["task_index"] is None:
        return None
    return int(row["task_index"])


def _delta(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values)
    out[1:] = values[1:] - values[:-1]
    return out


def _nearest_event_distance(frame_index: int, events: list[int]) -> int | None:
    if not events:
        return None
    return min((frame_index - int(event) for event in events), key=abs)
