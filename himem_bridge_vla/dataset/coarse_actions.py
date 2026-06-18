from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


ACTION_CONVENTIONS = {"relative", "absolute_delta", "absolute_terminal"}


def build_coarse_action_target(
    actions: Any,
    *,
    num_plan_steps: int,
    planning_horizon: int,
    valid_action_count: int | None = None,
    action_convention: str = "relative",
    motion_indices: Sequence[int] | None = None,
    gripper_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compress future actions into K coarse plan targets.

    `actions` must contain at least `planning_horizon` future actions. The mask
    marks a coarse step valid only when its whole chunk lies inside the real
    episode rather than padded tail rows.
    """

    if num_plan_steps <= 0:
        raise ValueError(f"num_plan_steps must be positive, got {num_plan_steps}")
    if planning_horizon <= 0:
        raise ValueError(f"planning_horizon must be positive, got {planning_horizon}")
    if planning_horizon % num_plan_steps != 0:
        raise ValueError("planning_horizon must be divisible by num_plan_steps")
    if action_convention not in ACTION_CONVENTIONS:
        raise ValueError(f"action_convention must be one of {sorted(ACTION_CONVENTIONS)}")

    action_array = np.asarray(actions, dtype=np.float32)
    if action_array.ndim != 2:
        raise ValueError(f"actions must have shape [T, A], got {action_array.shape}")
    if action_array.shape[0] < planning_horizon:
        raise ValueError(
            f"actions length {action_array.shape[0]} is shorter than planning_horizon {planning_horizon}"
        )

    action_dim = int(action_array.shape[1])
    gripper = _normalize_indices(gripper_indices, action_dim, default=(action_dim - 1,))
    motion = _normalize_indices(motion_indices, action_dim, default=tuple(i for i in range(action_dim) if i not in gripper))
    if not motion and not gripper:
        raise ValueError("at least one motion or gripper index must be selected")

    valid_count = planning_horizon if valid_action_count is None else max(0, min(int(valid_action_count), planning_horizon))
    chunk_size = planning_horizon // num_plan_steps
    coarse = np.zeros((num_plan_steps, action_dim), dtype=np.float32)
    mask = np.zeros((num_plan_steps,), dtype=bool)

    for step in range(num_plan_steps):
        start = step * chunk_size
        end = start + chunk_size
        if end > valid_count:
            continue

        chunk = action_array[start:end]
        if motion:
            if action_convention == "relative":
                coarse[step, motion] = chunk[:, motion].sum(axis=0)
            elif action_convention == "absolute_delta":
                coarse[step, motion] = chunk[-1, motion] - chunk[0, motion]
            else:
                coarse[step, motion] = chunk[-1, motion]
        if gripper:
            coarse[step, gripper] = chunk[-1, gripper]
        mask[step] = True

    return coarse, mask


def _normalize_indices(indices: Sequence[int] | None, action_dim: int, *, default: Sequence[int]) -> tuple[int, ...]:
    selected = default if indices is None else indices
    normalized = []
    for index in selected:
        value = int(index)
        if value < 0:
            value += action_dim
        if value < 0 or value >= action_dim:
            raise ValueError(f"action index {index} is out of range for action_dim {action_dim}")
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)
