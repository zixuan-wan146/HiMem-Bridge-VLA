from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "data": {
        "format": "planner_feature_cache",
        "root": "/root/autodl-tmp/datasets/coarse_planner/default",
        "manifest": "manifest.json",
        "input_paths": [],
        "vlm_token_key": "vlm_tokens",
        "state_key": "states",
        "action_key": "actions",
        "episode_key": "episode_id",
        "frame_key": "frame_index",
        "stride": 1,
        "include_tail": True,
        "max_samples": None,
        "max_samples_per_shard": 4096,
        "val_fraction": 0.1,
        "split_by": "episode",
        "train_split": "train",
        "eval_split": "eval",
    },
    "feature": {
        "source": "fused",
        "hidden_state_layer": "deep",
        "model_name": "OpenGVLab/InternVL3-1B",
        "image_size": 448,
        "allow_image_token_truncation": False,
        "storage_dtype": "float16",
    },
    "simulation": {
        "dataset_config": "configs/datasets/simulation.yaml",
        "action_horizon": None,
        "max_samples_per_file": None,
        "video_backend": "av",
        "video_backend_kwargs": {},
        "cache_dir": "/root/autodl-tmp/cache/himem_bridge_vla/simulation",
    },
    "target": {
        "num_plan_steps": 16,
        "planning_horizon": 128,
        "action_convention": "relative",
        "motion_indices": None,
        "gripper_indices": [-1],
    },
    "model": {
        "hidden_dim": "auto",
        "action_dim": "auto",
        "state_dim": "auto",
        "num_plan_steps": None,
        "planning_horizon": None,
        "num_layers": 3,
        "num_heads": 8,
        "dropout": 0.0,
        "ffn_mult": 4,
    },
    "training": {
        "batch_size": 64,
        "epochs": 20,
        "lr": 3.0e-4,
        "weight_decay": 1.0e-4,
        "num_workers": 4,
        "grad_clip_norm": 1.0,
    },
    "loss": {
        "gripper_loss_weight": 2.0,
        "smoothness_weight": 0.01,
        "gripper_indices": [-1],
    },
    "evaluation": {
        "trajectory_dims": [0, 1, 2],
    },
    "outputs": {
        "run_dir": "coarse_planner/outputs/default_run",
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
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
