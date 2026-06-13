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
    def __init__(self, stats_or_path, target_dim: int = TARGET_STATE_DIM):
        self.target_dim = int(target_dim)
        if isinstance(stats_or_path, (str, Path)):
            with open(stats_or_path, "r") as f:
                stats = json.load(f)
        else:
            stats = stats_or_path

        if len(stats) != 1:
            raise ValueError(f"norm_stats.json should contain one robot key, got: {list(stats.keys())}")

        robot_stats = stats[next(iter(stats))]
        self.state_min = pad_vector(robot_stats["observation.state"]["min"], self.target_dim)
        self.state_max = pad_vector(robot_stats["observation.state"]["max"], self.target_dim)
        self.action_min = pad_vector(robot_stats["action"]["min"], self.target_dim)
        self.action_max = pad_vector(robot_stats["action"]["max"], self.target_dim)

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        state_dim = state.shape[-1]
        if state_dim > self.state_min.shape[0]:
            raise ValueError(f"State dimension {state_dim} exceeds normalizer dimension {self.state_min.shape[0]}")
        state_min = self.state_min[:state_dim].to(state.device, dtype=state.dtype)
        state_max = self.state_max[:state_dim].to(state.device, dtype=state.dtype)
        return minmax_normalize(state, state_min, state_max)

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim == 1:
            action = action.view(1, -1)
        action_dim = action.shape[-1]
        if action_dim > self.action_min.shape[0]:
            raise ValueError(f"Action dimension {action_dim} exceeds normalizer dimension {self.action_min.shape[0]}")
        action_min = self.action_min[:action_dim].to(action.device, dtype=action.dtype)
        action_max = self.action_max[:action_dim].to(action.device, dtype=action.dtype)
        return minmax_denormalize(action, action_min, action_max)
