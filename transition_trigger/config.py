from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "data": {
        "format": "segmented_parquet",
        "root": "datasets/<benchmark>/subset",
        "transition_jsonl": "datasets/<benchmark>/annotations/transitions.jsonl",
        "action_keys": ["rel_actions", "action", "actions"],
        "state_keys": ["robot_obs", "observation.state", "state"],
        "frame_keys": ["frame_index", "frame_idx"],
        "global_frame_keys": ["index", "global_index", "global_frame_idx"],
        "episode_keys": ["episode_index", "episode_id"],
        "window_size": 32,
        "label_mode": "causal_post",
        "positive_radius": 2,
        "ignore_radius": 6,
        "positive_pre_frames": None,
        "positive_post_frames": None,
        "ignore_pre_frames": None,
        "ignore_post_frames": None,
        "positive_min_delay": 1,
        "positive_max_delay": 5,
        "ignore_min_delay": -6,
        "ignore_max_delay": 0,
        "hard_negative_radius": 30,
        "label_sigma": 2.0,
        "soft_labels": True,
        "val_fraction": 0.1,
        "split_by": "trajectory",
        "split_manifest": None,
        "train_split": "train",
        "eval_split": "eval",
        "test_split": "test",
        "datasets": [],
    },
    "features": {
        "mode": "flat",
        "feature_set": "flat",
        "expected_input_dim": None,
        "use_action": True,
        "use_state": True,
        "use_delta_action": True,
        "use_delta_state": True,
        "use_gripper_transition": True,
        "include_deltas": True,
        "include_value_mask": True,
        "source_one_hot": {
            "enabled": False,
            "names": [],
        },
        "blocks": [],
        "normalize": False,
    },
    "model": {
        "type": "ssm",
        "d_model": 256,
        "num_layers": 4,
        "state_dim": 64,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "max_seq_len": 64,
        "pooling": "last",
        "head_hidden_dim": 256,
        "dropout": 0.1,
    },
    "training": {
        "batch_size": 256,
        "epochs": 20,
        "lr": 3.0e-4,
        "weight_decay": 1.0e-4,
        "num_workers": 4,
        "epoch_size": 20000,
        "sampler": "balanced",
        "positive_ratio": 0.5,
        "hard_negative_ratio": 0.25,
        "loss": "bce",
        "pos_weight": "sqrt_neg_pos",
        "focal_gamma": 2.0,
        "focal_alpha": None,
        "grad_clip_norm": 1.0,
    },
    "evaluation": {
        "event_tolerance": 3,
        "match_min_delay": 1,
        "match_max_delay": 5,
        "early_tolerance": 6,
        "cooldown": 10,
        "threshold_grid": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95],
        "planner_recall_target": 0.9,
        "planner_precision_floor": 0.0,
        "memory_write_fixed_threshold": 0.8,
        "memory_precision_target": 0.8,
        "memory_recall_floor": 0.0,
        "dataset_split": "eval",
    },
    "trigger_policy": {
        "planner_threshold": 0.5,
        "memory_write_threshold": 0.8,
        "memory_write_implies_plan": True,
    },
    "outputs": {
        "run_dir": "transition_trigger/outputs/default_run",
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
