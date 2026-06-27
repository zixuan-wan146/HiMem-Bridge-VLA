from __future__ import annotations

from pathlib import Path
from typing import Any

from himem_bridge_vla.experiment_config import resolve_experiment_config
from himem_bridge_vla.path_utils import normalize_project_relative_path, project_path
from himem_bridge_vla.training.stage1.libero.validators import enforce_stage1_contract, validate_stage1_cache_contract
from himem_bridge_vla.training_config import (
    default_training_config,
    load_training_config,
    merge_training_config,
    resolve_training_config_paths,
    validate_training_config,
)


def build_stage1_config(
    args: Any,
    *,
    repo_root: str | Path,
    validate_external_artifacts: bool = False,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    cli_overrides = vars(args).copy()
    config_path = cli_overrides.pop("config", None)
    if config_path:
        config_file = project_path(config_path, repo_root, label="--config")
        file_config = load_training_config(config_file)
        file_config["training_config_path"] = normalize_project_relative_path(
            config_file,
            repo_root,
            label="--config",
        )
    else:
        file_config = {}

    active_defaults = {
        "dataset_type": "memory_token_cache",
        "memory_token_cache_sequence_training": True,
        "load_vlm": False,
        "finetune_vlm": False,
        "finetune_action_head": True,
        "finetune_progress_planner": False,
        "enable_bridge_aux_loss": False,
        "horizon": 32,
        "progress_planner_replan_stride": 16,
        "num_inference_timesteps": 15,
        "inference_tau_schedule": "midpoint",
        "avoid_endpoint_tau": True,
    }
    explicit_config_keys = {
        key for key, value in file_config.items() if value is not None
    } | {
        key for key, value in cli_overrides.items() if value is not None
    }

    config = merge_training_config(
        default_training_config(repo_root),
        file_config={**active_defaults, **file_config},
        cli_overrides=cli_overrides,
    )
    config["_explicit_config_keys"] = sorted(explicit_config_keys)
    config["repo_root"] = "."
    config = resolve_training_config_paths(config, repo_root)
    config = resolve_experiment_config(config)
    config = resolve_training_config_paths(config, repo_root)
    enforce_stage1_contract(config)
    validate_training_config(
        config,
        repo_root=repo_root,
        validate_external_paths=validate_external_artifacts,
    )
    manifest_path = project_path(config["dataset_config_path"], repo_root, label="--dataset_config_path")
    if validate_external_artifacts or manifest_path.exists():
        validate_stage1_cache_contract(config, repo_root=repo_root)
    return config
