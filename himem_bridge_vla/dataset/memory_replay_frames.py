from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from himem_bridge_vla.dataset.libero import DEFAULT_LIBERO_VIEW_NAMES
from himem_bridge_vla.dataset.libero import LiberoEpisodeReader
from himem_bridge_vla.dataset.rmbench import DEFAULT_RMBENCH_CAMERA_NAMES
from himem_bridge_vla.dataset.rmbench import RMBenchEpisodeReader


@dataclass(frozen=True)
class ReplayFrame:
    tau: int
    images_by_view: Mapping[str, Image.Image]
    state_vector: np.ndarray


@dataclass(frozen=True)
class MemoryReplayFrameSample:
    benchmark: str
    episode_id: str
    current_step: int
    current: ReplayFrame
    short_frames: tuple[ReplayFrame | None, ...]
    short_mask: tuple[bool, ...]
    future_actions: np.ndarray
    action_valid_count: int


class MemoryReplayFrameReader:
    """Resolve replay-index rows into current/history frames and future actions."""

    def __init__(
        self,
        *,
        benchmark: str,
        data_root: str | Path,
        view_names: Sequence[str] | None = None,
    ) -> None:
        self.benchmark = str(benchmark).upper()
        self.data_root = Path(data_root).expanduser()
        self.view_names = None if view_names is None else tuple(str(name) for name in view_names)
        if self.benchmark not in {"LIBERO", "RMBENCH"}:
            raise ValueError(f"unsupported replay benchmark: {benchmark!r}")

    def read(self, row: Mapping[str, Any]) -> MemoryReplayFrameSample:
        benchmark = str(row.get("benchmark") or self.benchmark).upper()
        if benchmark != self.benchmark:
            raise ValueError(f"row benchmark {benchmark!r} does not match reader benchmark {self.benchmark!r}")
        if self.benchmark == "LIBERO":
            return self._read_libero(row)
        return self._read_rmbench(row)

    def _read_libero(self, row: Mapping[str, Any]) -> MemoryReplayFrameSample:
        source_path = _required(row, "source_path")
        demo_key = str(row.get("episode_key") or _demo_key_from_episode_id(_required(row, "episode_id")))
        reader = LiberoEpisodeReader(
            self.data_root / source_path,
            demo_key=demo_key,
            view_names=self.view_names or DEFAULT_LIBERO_VIEW_NAMES,
        )
        current_step = int(row["current_step"])
        current = _libero_frame_to_replay(reader.read_frame(current_step))
        short_frames = tuple(
            _libero_frame_to_replay(reader.read_frame(int(step))) if step is not None else None
            for step in row.get("short_steps", [])
        )
        future_actions = reader.read_future_actions(int(row["action_start"]), int(row["action_end"]))
        return _build_sample(row, current=current, short_frames=short_frames, future_actions=future_actions)

    def _read_rmbench(self, row: Mapping[str, Any]) -> MemoryReplayFrameSample:
        source_path = _required(row, "source_path")
        instruction_path = row.get("instruction_path")
        reader = RMBenchEpisodeReader(
            self.data_root / source_path,
            instruction_path=self.data_root / instruction_path if instruction_path else None,
            camera_names=self.view_names or DEFAULT_RMBENCH_CAMERA_NAMES,
        )
        current_step = int(row["current_step"])
        current = _rmbench_frame_to_replay(reader.read_frame(current_step))
        short_frames = tuple(
            _rmbench_frame_to_replay(reader.read_frame(int(step))) if step is not None else None
            for step in row.get("short_steps", [])
        )
        future_actions = reader.read_future_actions(int(row["action_start"]), int(row["action_end"]))
        return _build_sample(row, current=current, short_frames=short_frames, future_actions=future_actions)


def _build_sample(
    row: Mapping[str, Any],
    *,
    current: ReplayFrame,
    short_frames: tuple[ReplayFrame | None, ...],
    future_actions: np.ndarray,
) -> MemoryReplayFrameSample:
    short_mask = tuple(bool(value) for value in row.get("short_mask", [frame is not None for frame in short_frames]))
    if len(short_mask) != len(short_frames):
        raise ValueError("short_mask length does not match short_frames length")
    return MemoryReplayFrameSample(
        benchmark=str(row.get("benchmark", "")),
        episode_id=str(row["episode_id"]),
        current_step=int(row["current_step"]),
        current=current,
        short_frames=short_frames,
        short_mask=short_mask,
        future_actions=np.asarray(future_actions, dtype=np.float32),
        action_valid_count=int(row["action_valid_count"]),
    )


def _libero_frame_to_replay(frame) -> ReplayFrame:
    return ReplayFrame(
        tau=int(frame.tau),
        images_by_view=frame.images_by_view,
        state_vector=np.asarray(frame.state_vector, dtype=np.float32),
    )


def _rmbench_frame_to_replay(frame) -> ReplayFrame:
    return ReplayFrame(
        tau=int(frame.tau),
        images_by_view=frame.images_by_view,
        state_vector=np.asarray(frame.state_vector, dtype=np.float32),
    )


def _required(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if value in (None, ""):
        raise KeyError(f"replay row is missing required key: {key}")
    return str(value)


def _demo_key_from_episode_id(episode_id: str) -> str:
    parts = str(episode_id).split(":")
    if not parts:
        raise ValueError(f"cannot infer LIBERO demo key from episode_id={episode_id!r}")
    return parts[-1]
