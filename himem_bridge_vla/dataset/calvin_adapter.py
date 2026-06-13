from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_STATE_KEYS = ("observation.state", "state")
DEFAULT_ACTION_KEYS = ("action", "actions")
DEFAULT_TASK_KEYS = ("task_index", "annotation.human.action.task_description")
DEFAULT_TIMESTAMP_KEYS = ("timestamp",)
DEFAULT_FRAME_KEYS = ("frame_index", "frame_idx")
DEFAULT_GLOBAL_FRAME_KEYS = ("index", "global_index", "global_frame_idx")
DEFAULT_EPISODE_KEYS = ("episode_index", "episode_id")

DEFAULT_VIEW_MAP = {
    "image_1": ("observation.images.image", "observation.images.image_0", "image"),
    "image_2": ("observation.images.wrist_image", "wrist_image"),
}


@dataclass(frozen=True)
class CalvinFrameLabel:
    boundary: int
    progress: float
    skill_id: int | None
    task: str | None
    language: str | None
    segment_id: int
    segment_start: int
    segment_end: int


@dataclass(frozen=True)
class CalvinSegment:
    segment_id: int
    start: int
    end: int
    task: str | None = None
    language: str | None = None
    skill_id: int | None = None
    episode_id: str | None = None

    def contains(self, frame_index: int, episode_id: str | None = None) -> bool:
        if self.episode_id is not None and episode_id is not None and str(self.episode_id) != str(episode_id):
            return False
        return self.start <= frame_index <= self.end

    def label_at(self, frame_index: int) -> CalvinFrameLabel:
        span = max(1, self.end - self.start)
        progress = min(1.0, max(0.0, (frame_index - self.start) / span))
        return CalvinFrameLabel(
            boundary=int(frame_index == self.end),
            progress=progress,
            skill_id=self.skill_id,
            task=self.task,
            language=self.language,
            segment_id=self.segment_id,
            segment_start=self.start,
            segment_end=self.end,
        )


class CalvinBoundaryIndex:
    def __init__(self, segments: Sequence[CalvinSegment]) -> None:
        self._global_segments = sorted(
            [segment for segment in segments if segment.episode_id is None],
            key=lambda segment: segment.start,
        )
        self._global_starts = [segment.start for segment in self._global_segments]
        self._episode_segments: dict[str, list[CalvinSegment]] = {}
        for segment in segments:
            if segment.episode_id is None:
                continue
            self._episode_segments.setdefault(str(segment.episode_id), []).append(segment)
        for episode_segments in self._episode_segments.values():
            episode_segments.sort(key=lambda segment: segment.start)

    @classmethod
    def from_jsonl(cls, path: str | Path | None) -> CalvinBoundaryIndex | None:
        if path is None:
            return None
        label_path = Path(path).expanduser()
        if not label_path.exists():
            return None

        segments = []
        for line_number, raw_line in enumerate(label_path.read_text().splitlines(), start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"{label_path} line {line_number} must be a JSON object")
            start = _as_int(row.get("start", row.get("segment_start")), f"{label_path}:{line_number}:start")
            end = _as_int(row.get("end", row.get("segment_end")), f"{label_path}:{line_number}:end")
            if end < start:
                raise ValueError(f"{label_path} line {line_number} has end < start")
            skill_id = row.get("skill_id", row.get("task_index"))
            segments.append(
                CalvinSegment(
                    segment_id=int(row.get("segment_id", len(segments))),
                    start=start,
                    end=end,
                    task=_optional_str(row.get("task", row.get("task_id"))),
                    language=_optional_str(row.get("language", row.get("ann"))),
                    skill_id=None if skill_id is None else int(skill_id),
                    episode_id=_optional_str(row.get("episode_id")),
                )
            )
        return cls(segments)

    def label_for(
        self,
        *,
        global_frame_index: int | None,
        frame_index: int | None,
        episode_id: str | None,
    ) -> CalvinFrameLabel | None:
        if episode_id is not None and frame_index is not None:
            label = self._label_from_episode(frame_index, str(episode_id))
            if label is not None:
                return label
        if global_frame_index is not None:
            return self._label_from_global(global_frame_index)
        if frame_index is not None:
            return self._label_from_global(frame_index)
        return None

    def _label_from_episode(self, frame_index: int, episode_id: str) -> CalvinFrameLabel | None:
        for segment in self._episode_segments.get(episode_id, []):
            if segment.contains(frame_index, episode_id):
                return segment.label_at(frame_index)
        return None

    def _label_from_global(self, frame_index: int) -> CalvinFrameLabel | None:
        position = bisect_right(self._global_starts, frame_index) - 1
        if position < 0:
            return None
        segment = self._global_segments[position]
        if segment.contains(frame_index):
            return segment.label_at(frame_index)
        return None


class DatasetInputAdapter:
    def __init__(self, dataset_config: Mapping[str, Any], dataset_path: Path) -> None:
        self.dataset_config = dataset_config
        self.dataset_path = dataset_path
        self.view_map = _normalize_view_map(dataset_config.get("view_map"))

    def resolve_video_paths(self, base_video_path: Path, parquet_path: Path) -> dict[str, str]:
        video_paths = {}
        for view_key, candidates in self.view_map.items():
            for view_folder in candidates:
                full_path = base_video_path / view_folder / f"{parquet_path.stem}.mp4"
                if full_path.exists():
                    video_paths[view_key] = str(full_path)
                    break
        return video_paths

    def state(self, row: pd.Series) -> Any:
        return _first_present(row, self.dataset_config.get("state_keys", ("observation.state",)))

    def action(self, row: pd.Series) -> Any:
        return _first_present(row, self.dataset_config.get("action_keys", ("action",)))

    def timestamp(self, row: pd.Series, row_index: int) -> float | None:
        value = _first_present(row, self.dataset_config.get("timestamp_keys", ("timestamp",)), default=None)
        if value is None:
            return None
        return float(value)

    def prompt(self, row: pd.Series, task_mapping: Mapping[Any, str], metadata: Mapping[str, Any] | None = None) -> str:
        task_index = _first_present(row, self.dataset_config.get("task_keys", ("task_index",)), default=None)
        if task_index in task_mapping:
            return task_mapping[task_index]
        return ""

    def sample_metadata(self, row: pd.Series, parquet_path: Path, row_index: int) -> dict[str, Any]:
        return {}


class CalvinInputAdapter(DatasetInputAdapter):
    def __init__(self, dataset_config: Mapping[str, Any], dataset_path: Path) -> None:
        super().__init__({**dict(dataset_config), "view_map": dataset_config.get("view_map") or DEFAULT_VIEW_MAP}, dataset_path)
        boundary_path = dataset_config.get("boundary_path", dataset_config.get("calvin_boundary_path"))
        self.boundary_index = CalvinBoundaryIndex.from_jsonl(boundary_path)
        self.fps = float(dataset_config.get("fps", 30.0))

    def state(self, row: pd.Series) -> Any:
        return _first_present(row, self.dataset_config.get("state_keys", DEFAULT_STATE_KEYS))

    def action(self, row: pd.Series) -> Any:
        return _first_present(row, self.dataset_config.get("action_keys", DEFAULT_ACTION_KEYS))

    def timestamp(self, row: pd.Series, row_index: int) -> float | None:
        value = _first_present(row, self.dataset_config.get("timestamp_keys", DEFAULT_TIMESTAMP_KEYS), default=None)
        if value is not None:
            return float(value)
        frame_index = self._frame_index(row, row_index)
        return None if frame_index is None else float(frame_index) / self.fps

    def prompt(self, row: pd.Series, task_mapping: Mapping[Any, str], metadata: Mapping[str, Any] | None = None) -> str:
        metadata = metadata or {}
        if metadata.get("segment_language"):
            return str(metadata["segment_language"])
        task_index = _first_present(row, self.dataset_config.get("task_keys", DEFAULT_TASK_KEYS), default=None)
        if task_index in task_mapping:
            return task_mapping[task_index]
        if metadata.get("segment_task"):
            return str(metadata["segment_task"])
        return ""

    def sample_metadata(self, row: pd.Series, parquet_path: Path, row_index: int) -> dict[str, Any]:
        frame_index = self._frame_index(row, row_index)
        global_frame_index = self._global_frame_index(row)
        episode_id = self._episode_id(row, parquet_path)
        metadata: dict[str, Any] = {
            "episode_id": episode_id,
            "frame_index": frame_index,
            "global_frame_index": global_frame_index,
        }
        if self.boundary_index is None:
            return metadata

        label = self.boundary_index.label_for(
            global_frame_index=global_frame_index,
            frame_index=frame_index,
            episode_id=episode_id,
        )
        if label is None:
            return metadata

        metadata.update(
            {
                "boundary": label.boundary,
                "progress": label.progress,
                "skill_id": label.skill_id,
                "segment_task": label.task,
                "segment_language": label.language,
                "segment_id": label.segment_id,
                "segment_start": label.segment_start,
                "segment_end": label.segment_end,
            }
        )
        return metadata

    def _frame_index(self, row: pd.Series, row_index: int) -> int | None:
        value = _first_present(row, self.dataset_config.get("frame_keys", DEFAULT_FRAME_KEYS), default=None)
        if value is None:
            return int(row_index)
        return int(value)

    def _global_frame_index(self, row: pd.Series) -> int | None:
        value = _first_present(row, self.dataset_config.get("global_frame_keys", DEFAULT_GLOBAL_FRAME_KEYS), default=None)
        return None if value is None else int(value)

    def _episode_id(self, row: pd.Series, parquet_path: Path) -> str:
        value = _first_present(row, self.dataset_config.get("episode_keys", DEFAULT_EPISODE_KEYS), default=None)
        if value is not None:
            return str(value)
        return f"{parquet_path.parent.name}/{parquet_path.stem}"


def build_dataset_input_adapter(dataset_config: Mapping[str, Any], dataset_path: Path) -> DatasetInputAdapter:
    adapter_name = str(dataset_config.get("adapter", dataset_config.get("input_adapter", "default"))).lower()
    if adapter_name in {"default", "lerobot", "simulation"}:
        return DatasetInputAdapter(dataset_config, dataset_path)
    if adapter_name == "calvin":
        return CalvinInputAdapter(dataset_config, dataset_path)
    raise ValueError(f"unknown dataset input adapter: {adapter_name}")


def _normalize_view_map(raw_view_map: Any) -> dict[str, tuple[str, ...]]:
    view_map = raw_view_map or {
        "image_1": ("observation.images.image_1",),
        "image_2": ("observation.images.image_2",),
        "image_3": ("observation.images.image_3",),
    }
    if not isinstance(view_map, Mapping) or not view_map:
        raise ValueError("view_map must be a non-empty mapping")
    normalized = {}
    for view_key, value in view_map.items():
        if isinstance(value, str):
            candidates = (value,)
        elif isinstance(value, Sequence):
            candidates = tuple(str(item) for item in value if str(item))
        else:
            raise ValueError(f"view_map value for {view_key!r} must be a string or list of strings")
        if not candidates:
            raise ValueError(f"view_map value for {view_key!r} has no candidates")
        normalized[str(view_key)] = candidates
    return normalized


def _first_present(row: pd.Series, keys: Any, default: Any = ...):
    if isinstance(keys, str):
        keys = (keys,)
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    if default is not ...:
        return default
    raise KeyError(f"none of the configured keys are present: {tuple(keys)}")


def _as_int(value: Any, label: str) -> int:
    if value is None:
        raise ValueError(f"{label} is required")
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
