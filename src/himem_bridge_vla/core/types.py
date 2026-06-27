from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    view_names: tuple[str, ...]
    state_dim: int
    action_dim: int
    replan_stride: int
    short_memory_offsets: tuple[int, ...]


@dataclass(frozen=True)
class ImageBundle:
    images_by_view: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class ActionChunk:
    values: np.ndarray
    valid_count: int | None = None


@dataclass(frozen=True)
class PolicyActionChunk:
    values: np.ndarray
    valid_count: int | None = None


@dataclass(frozen=True)
class RuntimeFeatures:
    current_visual_tokens: Any
    vlm_hidden_states: Any
    planner_vl_summary: Any
    short_memory_tokens: Any | None = None
    short_memory_mask: Any | None = None
    short_memory_time_ids: Any | None = None


@dataclass(frozen=True)
class PolicyResponse:
    action_chunk: PolicyActionChunk
    debug: Mapping[str, Any] | None = None
