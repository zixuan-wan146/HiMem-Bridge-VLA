from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from himem_bridge_vla.experiment_config import resolve_experiment_config
from himem_bridge_vla.path_utils import normalize_project_relative_path, project_path
from himem_bridge_vla.training.stage2.libero.validators import (
    enforce_stage2_contract,
    validate_stage2_replay_index_contract,
)
from himem_bridge_vla.training_config import (
    default_training_config,
    load_training_config,
    merge_training_config,
    resolve_training_config_paths,
    validate_training_config,
)


STAGE2_ACTIVE_DEFAULTS: dict[str, Any] = {
    "dataset_type": "libero_raw_episode",
    "load_vlm": True,
    "finetune_vlm": True,
    "finetune_action_head": True,
    "progress_planner_enabled": True,
    "finetune_progress_planner": True,
    "enable_bridge_aux_loss": False,
    "memory_token_cache_sequence_training": False,
    "horizon": 32,
    "sequence_len": 16,
    "stage2_sampling_mode": "group",
    "sample_valid_future_horizon_only": True,
    "shuffle_episodes": True,
    "num_inference_timesteps": 15,
    "inference_tau_schedule": "midpoint",
    "avoid_endpoint_tau": True,
}


def build_stage2_config(
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

    explicit_config_keys = set(STAGE2_ACTIVE_DEFAULTS) | _provided_keys(file_config) | _provided_keys(cli_overrides)
    config = merge_training_config(
        default_training_config(repo_root),
        file_config={**STAGE2_ACTIVE_DEFAULTS, **file_config},
        cli_overrides=cli_overrides,
    )
    config["_explicit_config_keys"] = sorted(explicit_config_keys)
    config["repo_root"] = "."
    config = resolve_training_config_paths(config, repo_root)
    config = _resolve_stage2_paths(config, repo_root)
    config = resolve_experiment_config(config)
    config = resolve_training_config_paths(config, repo_root)
    config = _resolve_stage2_paths(config, repo_root)
    enforce_stage2_contract(config)
    validate_training_config(
        config,
        repo_root=repo_root,
        validate_external_paths=validate_external_artifacts,
    )
    _validate_stage2_external_paths(
        config,
        repo_root=repo_root,
        validate_external_artifacts=validate_external_artifacts,
    )
    replay_index = project_path(config["dataset_config_path"], repo_root, label="--dataset_config_path")
    if validate_external_artifacts or replay_index.exists():
        validate_stage2_replay_index_contract(config, repo_root=repo_root)
    return config


def _provided_keys(mapping: Mapping[str, Any]) -> set[str]:
    return {str(key) for key, value in mapping.items() if value is not None}


def _resolve_stage2_paths(config: Mapping[str, Any], repo_root: str | Path) -> dict[str, Any]:
    resolved = dict(config)
    for key in ("normalization_source_path",):
        value = resolved.get(key)
        if value in (None, ""):
            continue
        resolved[key] = normalize_project_relative_path(value, repo_root, label=f"--{key}")
    return resolved


def _validate_stage2_external_paths(
    config: Mapping[str, Any], *, repo_root: str | Path, validate_external_artifacts: bool
) -> None:
    if not validate_external_artifacts:
        return
    normalization_path = config.get("normalization_source_path")
    if normalization_path:
        path = project_path(normalization_path, repo_root, label="--normalization_source_path")
        if not path.exists():
            raise FileNotFoundError(f"Normalization source file not found: {normalization_path}")
