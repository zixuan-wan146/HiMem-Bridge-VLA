from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "data": {
        "format": "planner_feature_cache",
        "root": "../datasets/coarse_planner/default",
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
        "cache_dir": "../cache/himem_bridge_vla/simulation",
    },
    "target": {
        "num_plan_steps": 1,
        "planning_horizon": 32,
        "gripper_indices": [-1],
    },
    "segment_autoencoder": {
        "checkpoint": None,
        "latent_dim": 128,
        "hidden_dim": 128,
        "num_layers": 2,
        "num_heads": 4,
        "ffn_dim": 512,
        "dropout": 0.05,
        "gripper_dim": 1,
        "distance_loss_weight": 0.1,
        "dct_low_frequency": 4,
        "endpoint_distance_weight": 0.5,
        "gripper_distance_weight": 0.25,
    },
    "model": {
        "hidden_dim": "auto",
        "state_dim": "auto",
        "latent_dim": None,
        "num_plan_steps": None,
        "planning_horizon": None,
        "num_layers": 4,
        "num_heads": 8,
        "dropout": 0.05,
        "ffn_mult": 4,
        "latent_head_hidden_dim": 512,
    },
    "training": {
        "batch_size": 64,
        "epochs": 20,
        "lr": 3.0e-4,
        "weight_decay": 1.0e-4,
        "num_workers": 4,
        "grad_clip_norm": 1.0,
        "amp": True,
    },
    "loss": {
        "gripper_loss_weight": 2.0,
        "latent_loss_weight": 1.0,
        "chunk_loss_weight": 0.25,
        "gripper_indices": [-1],
        "loss_on_active_suffix_only": False,
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
