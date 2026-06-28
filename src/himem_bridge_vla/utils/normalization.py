from __future__ import annotations

import json
from pathlib import Path

import torch

from himem_bridge_vla.runtime_config import TARGET_STATE_DIM


def pad_vector(values, target_dim: int = TARGET_STATE_DIM) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=torch.float32)
    if tensor.shape[0] > target_dim:
        raise ValueError(f"Input length {tensor.shape[0]} exceeds expected {target_dim}")
    if tensor.shape[0] < target_dim:
        pad = torch.zeros(target_dim - tensor.shape[0], dtype=torch.float32)
        tensor = torch.cat([tensor, pad], dim=0)
    return tensor


def minmax_normalize(value: torch.Tensor, min_value: torch.Tensor, max_value: torch.Tensor) -> torch.Tensor:
    normalized = 2 * (value - min_value) / (max_value - min_value + 1e-8) - 1
    return torch.clamp(normalized, -1.0, 1.0)


def minmax_denormalize(value: torch.Tensor, min_value: torch.Tensor, max_value: torch.Tensor) -> torch.Tensor:
    return 0.5 * (value + 1.0) * (max_value - min_value + 1e-8) + min_value


class NormalizationStats:
    def __init__(self, stats_or_path, target_dim: int = TARGET_STATE_DIM, robot_key: str | None = None):
        self.target_dim = int(target_dim)
        if isinstance(stats_or_path, (str, Path)):
            with open(stats_or_path, "r") as f:
                stats = json.load(f)
        else:
            stats = stats_or_path

        if not isinstance(stats, dict) or not stats:
            raise ValueError("norm_stats.json must contain at least one robot key")

        self.robot_keys = tuple(str(key) for key in stats)
        if robot_key is None:
            robot_key = next(iter(stats)) if len(stats) == 1 else None
        if robot_key is not None and robot_key not in stats:
            raise KeyError(f"robot_key {robot_key!r} not found in norm_stats.json; available keys: {list(stats.keys())}")
        self.robot_key = None if robot_key is None else str(robot_key)
        self._stats = stats
        self._prepared: dict[str, dict[str, torch.Tensor]] = {}
        if self.robot_key is not None:
            self._prepare_robot(self.robot_key)

    def normalize_state(self, state: torch.Tensor, robot_key: str | None = None) -> torch.Tensor:
        prepared = self._prepare_robot(self._resolve_robot_key(robot_key))
        state_dim = state.shape[-1]
        state_min_full = prepared["state_min"]
        state_max_full = prepared["state_max"]
        if state_dim > state_min_full.shape[0]:
            raise ValueError(f"State dimension {state_dim} exceeds normalizer dimension {state_min_full.shape[0]}")
        state_min = state_min_full[:state_dim].to(state.device, dtype=state.dtype)
        state_max = state_max_full[:state_dim].to(state.device, dtype=state.dtype)
        return minmax_normalize(state, state_min, state_max)

    def denormalize_action(self, action: torch.Tensor, robot_key: str | None = None) -> torch.Tensor:
        prepared = self._prepare_robot(self._resolve_robot_key(robot_key))
        if action.ndim == 1:
            action = action.view(1, -1)
        action_dim = action.shape[-1]
        action_min_full = prepared["action_min"]
        action_max_full = prepared["action_max"]
        if action_dim > action_min_full.shape[0]:
            raise ValueError(f"Action dimension {action_dim} exceeds normalizer dimension {action_min_full.shape[0]}")
        action_min = action_min_full[:action_dim].to(action.device, dtype=action.dtype)
        action_max = action_max_full[:action_dim].to(action.device, dtype=action.dtype)
        return minmax_denormalize(action, action_min, action_max)

    def normalize_action(self, action: torch.Tensor, robot_key: str | None = None) -> torch.Tensor:
        prepared = self._prepare_robot(self._resolve_robot_key(robot_key))
        action_dim = action.shape[-1]
        action_min_full = prepared["action_min"]
        action_max_full = prepared["action_max"]
        if action_dim > action_min_full.shape[0]:
            raise ValueError(f"Action dimension {action_dim} exceeds normalizer dimension {action_min_full.shape[0]}")
        action_min = action_min_full[:action_dim].to(action.device, dtype=action.dtype)
        action_max = action_max_full[:action_dim].to(action.device, dtype=action.dtype)
        return minmax_normalize(action, action_min, action_max)

    def _prepare_robot(self, robot_key: str) -> dict[str, torch.Tensor]:
        if robot_key in self._prepared:
            return self._prepared[robot_key]
        if robot_key not in self._stats:
            raise KeyError(f"robot_key {robot_key!r} not found in norm_stats.json; available keys: {list(self._stats.keys())}")
        robot_stats = self._stats[robot_key]
        prepared = {
            "state_min": pad_vector(robot_stats["observation.state"]["min"], self.target_dim),
            "state_max": pad_vector(robot_stats["observation.state"]["max"], self.target_dim),
            "action_min": pad_vector(robot_stats["action"]["min"], self.target_dim),
            "action_max": pad_vector(robot_stats["action"]["max"], self.target_dim),
        }
        self._prepared[robot_key] = prepared
        return prepared

    def _resolve_robot_key(self, robot_key: str | None) -> str:
        selected = robot_key or self.robot_key
        if selected is None:
            raise ValueError(f"robot_key is required when norm_stats.json has multiple keys: {list(self._stats.keys())}")
        return selected
