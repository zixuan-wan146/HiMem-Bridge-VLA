from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "data": {
        "format": "lerobot_calvin",
        "root": "datasets/calvin/lerobot/task_D_D",
        "boundary_jsonl": "datasets/calvin/annotations/task_D_D_boundaries.jsonl",
        "action_keys": ["rel_actions", "action", "actions"],
        "state_keys": ["robot_obs", "observation.state", "state"],
        "frame_keys": ["frame_index", "frame_idx"],
        "global_frame_keys": ["index", "global_index", "global_frame_idx"],
        "episode_keys": ["episode_index", "episode_id"],
        "window_size": 32,
        "positive_radius": 2,
        "ignore_radius": 6,
        "hard_negative_radius": 30,
        "label_sigma": 2.0,
        "soft_labels": True,
        "val_fraction": 0.1,
        "split_by": "trajectory",
    },
    "features": {
        "use_action": True,
        "use_state": True,
        "use_delta_action": True,
        "use_delta_state": True,
        "use_gripper_transition": True,
        "normalize": False,
    },
    "model": {
        "hidden_dim": 128,
        "kernel_size": 5,
        "dilations": [1, 2, 4, 8],
        "dropout": 0.1,
        "mlp_hidden_dim": 64,
    },
    "training": {
        "batch_size": 256,
        "epochs": 20,
        "lr": 3.0e-4,
        "weight_decay": 1.0e-4,
        "num_workers": 4,
        "epoch_size": 20000,
        "positive_ratio": 0.5,
        "hard_negative_ratio": 0.25,
        "pos_weight": "sqrt_neg_pos",
        "grad_clip_norm": 1.0,
    },
    "evaluation": {
        "event_tolerance": 3,
        "cooldown": 10,
        "threshold_grid": [round(x * 0.05, 2) for x in range(1, 20)],
        "planner_recall_target": 0.9,
        "memory_precision_target": 0.95,
    },
    "outputs": {
        "run_dir": "motion_boundary/outputs/default_run",
    },
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config

    config_path = Path(path).expanduser()
    with config_path.open("r") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config must contain a mapping: {config_path}")
    return merge_dicts(config, loaded)


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def write_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
