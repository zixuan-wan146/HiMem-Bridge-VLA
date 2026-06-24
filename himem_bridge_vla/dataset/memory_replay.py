from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


DEFAULT_MEMORY_SHORT_OFFSETS = (32, 16)
DEFAULT_MEMORY_LONG_CAPACITY = 4
DEFAULT_MEMORY_ACTION_HORIZON = 32


@dataclass(frozen=True)
class MemoryReplaySample:
    episode_id: str
    current_step: int
    episode_length: int
    action_horizon: int
    action_valid_count: int
    short_steps: tuple[int | None, ...]
    short_mask: tuple[bool, ...]
    long_steps: tuple[int, ...]
    benchmark: str | None = None
    task_name: str | None = None
    source_path: str | None = None
    instruction_path: str | None = None
    episode_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "episode_id": self.episode_id,
            "current_step": self.current_step,
            "episode_length": self.episode_length,
            "action_horizon": self.action_horizon,
            "action_start": self.current_step,
            "action_end": self.current_step + self.action_valid_count,
            "action_valid_count": self.action_valid_count,
            "short_steps": list(self.short_steps),
            "short_mask": list(self.short_mask),
            "long_steps": list(self.long_steps),
        }
        for key in ("benchmark", "task_name", "source_path", "instruction_path", "episode_key"):
            value = getattr(self, key)
            if value not in (None, ""):
                payload[key] = value
        return payload


def build_memory_replay_samples(
    *,
    episode_id: str,
    episode_length: int,
    action_horizon: int = DEFAULT_MEMORY_ACTION_HORIZON,
    stride: int = 1,
    short_offsets: Sequence[int] = DEFAULT_MEMORY_SHORT_OFFSETS,
    long_candidate_steps: Iterable[int] | None = None,
    long_capacity: int = DEFAULT_MEMORY_LONG_CAPACITY,
    include_tail: bool = False,
    benchmark: str | None = None,
    task_name: str | None = None,
    source_path: str | None = None,
    instruction_path: str | None = None,
    episode_key: str | None = None,
) -> list[MemoryReplaySample]:
    episode_length = int(episode_length)
    action_horizon = int(action_horizon)
    stride = int(stride)
    if episode_length <= 0:
        raise ValueError(f"episode_length must be positive, got {episode_length}")
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    offsets = _normalize_short_offsets(short_offsets)
    if int(long_capacity) < 0:
        raise ValueError(f"long_capacity must be non-negative, got {long_capacity}")
    candidate_steps = tuple(sorted({int(step) for step in long_candidate_steps or () if int(step) >= 0}))

    samples: list[MemoryReplaySample] = []
    for current_step in range(0, episode_length, stride):
        action_valid_count = min(action_horizon, episode_length - current_step)
        if action_valid_count < action_horizon and not include_tail:
            continue
        short_steps = tuple((current_step - offset) if current_step - offset >= 0 else None for offset in offsets)
        short_mask = tuple(step is not None for step in short_steps)
        long_steps = _select_long_steps(
            candidate_steps,
            current_step=current_step,
            long_capacity=int(long_capacity),
        )
        samples.append(
            MemoryReplaySample(
                episode_id=str(episode_id),
                current_step=current_step,
                episode_length=episode_length,
                action_horizon=action_horizon,
                action_valid_count=action_valid_count,
                short_steps=short_steps,
                short_mask=short_mask,
                long_steps=long_steps,
                benchmark=benchmark,
                task_name=task_name,
                source_path=source_path,
                instruction_path=instruction_path,
                episode_key=episode_key,
            )
        )
    return samples


def build_memory_replay_manifest(
    *,
    benchmark: str,
    action_horizon: int,
    stride: int,
    short_offsets: Sequence[int],
    long_capacity: int,
    include_tail: bool,
    sample_count: int,
    episode_count: int,
    task_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "format": "memory_replay_index",
        "version": 1,
        "benchmark": benchmark,
        "action_horizon": int(action_horizon),
        "stride": int(stride),
        "short_offsets": list(_normalize_short_offsets(short_offsets)),
        "long_capacity": int(long_capacity),
        "include_tail": bool(include_tail),
        "sample_count": int(sample_count),
        "episode_count": int(episode_count),
        "task_counts": dict(sorted((task_counts or {}).items())),
    }


def write_memory_replay_jsonl(path: str | Path, samples: Sequence[MemoryReplaySample | Mapping[str, Any]]) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            payload = sample.to_dict() if isinstance(sample, MemoryReplaySample) else dict(sample)
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    return output_path


def read_memory_replay_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path).expanduser()
    rows = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize_short_offsets(short_offsets: Sequence[int]) -> tuple[int, ...]:
    offsets = tuple(sorted({int(offset) for offset in short_offsets}, reverse=True))
    if not offsets:
        raise ValueError("short_offsets must contain at least one offset")
    if any(offset <= 0 for offset in offsets):
        raise ValueError(f"short_offsets must be positive, got {short_offsets}")
    return offsets


def _select_long_steps(candidate_steps: Sequence[int], *, current_step: int, long_capacity: int) -> tuple[int, ...]:
    if long_capacity == 0:
        return ()
    eligible = [step for step in candidate_steps if step < current_step]
    return tuple(eligible[-long_capacity:])
