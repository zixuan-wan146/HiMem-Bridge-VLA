from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from himem_bridge_vla.path_utils import project_path


MEMORY_TOKEN_CACHE_FORMAT = "memory_replay_visual_token_cache"
REQUIRED_HIDDEN_STATE_LAYERS = (3, 6, 9, 12)
DEFAULT_STAGE1_HIDDEN_DIM = 896


def enforce_stage1_contract(config: dict[str, Any]) -> None:
    """Reject non-active Stage1 routes before model or dataset construction."""

    if str(config.get("dataset_type")) != "memory_token_cache":
        raise ValueError("Stage1 requires dataset_type=memory_token_cache")
    if not bool(config.get("memory_token_cache_sequence_training", False)):
        raise ValueError("Stage1 requires trajectory-window token-cache training")
    if bool(config.get("load_vlm", True)):
        raise ValueError("Stage1 trains from token cache and requires load_vlm=false")
    if bool(config.get("finetune_vlm", False)):
        raise ValueError("Stage1 keeps the VLM frozen/offline and requires finetune_vlm=false")
    if not bool(config.get("finetune_action_head", False)):
        raise ValueError("Stage1 requires finetune_action_head=true")
    if bool(config.get("finetune_progress_planner", False)):
        raise ValueError("Stage1 uses the frozen W4 ProgressPlanner and requires finetune_progress_planner=false")
    if bool(config.get("enable_bridge_aux_loss", False)):
        raise ValueError("Stage1 supports only masked flow-matching velocity loss; disable bridge aux loss")
    if not bool(config.get("progress_planner_enabled", False)):
        raise ValueError("Stage1 requires progress_planner.enabled=true")
    if not config.get("progress_planner_checkpoint"):
        raise ValueError("Stage1 requires a frozen W4 progress_planner_checkpoint")
    if int(config.get("horizon", 0)) != 32:
        raise ValueError("Stage1 LIBERO direct-bridge training is locked to horizon=32")
    if int(config.get("progress_planner_replan_stride", 0)) != 16:
        raise ValueError("Stage1 LIBERO direct-bridge training is locked to replan stride=16")
    if int(config.get("num_inference_timesteps", 0)) != 15:
        raise ValueError("Stage1 rollout/smoke inference is locked to 15 Euler steps")
    if str(config.get("inference_tau_schedule", "")).lower() != "midpoint":
        raise ValueError("Stage1 requires midpoint inference tau schedule")
    if not bool(config.get("avoid_endpoint_tau", False)):
        raise ValueError("Stage1 requires avoid_endpoint_tau=true")


def validate_stage1_cache_contract(config: dict[str, Any], *, repo_root: str | Path) -> None:
    manifest_path = project_path(config.get("dataset_config_path"), repo_root, label="--dataset_config_path")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != MEMORY_TOKEN_CACHE_FORMAT:
        raise ValueError(
            f"Stage1 requires {MEMORY_TOKEN_CACHE_FORMAT} manifest, got {manifest.get('format')!r}"
        )

    hidden_dim = int(manifest.get("hidden_dim", 0))
    expected_hidden_dim = int(config.get("embed_dim", DEFAULT_STAGE1_HIDDEN_DIM))
    if hidden_dim != expected_hidden_dim:
        raise ValueError(f"Stage1 cache hidden_dim {hidden_dim} != expected {expected_hidden_dim}")

    hidden_layers = tuple(int(layer) for layer in manifest.get("hidden_state_layers") or ())
    if hidden_layers != REQUIRED_HIDDEN_STATE_LAYERS:
        raise ValueError(
            f"Stage1 cache hidden_state_layers {hidden_layers!r} != required {REQUIRED_HIDDEN_STATE_LAYERS!r}"
        )
    if int(manifest.get("hidden_state_cache_entries", 0)) <= 0:
        raise ValueError("Stage1 cache must include current VLM hidden-state entries")

    stats = _manifest_robot_stats(manifest)
    action_max = stats.get("action", {}).get("max")
    state_max = stats.get("observation.state", {}).get("max")
    if action_max is not None and len(action_max) != int(config.get("per_action_dim", 0)):
        raise ValueError(
            f"Stage1 cache action dimension {len(action_max)} != per_action_dim {config.get('per_action_dim')}"
        )
    if state_max is not None and len(state_max) != int(config.get("state_dim", 0)):
        raise ValueError(f"Stage1 cache state dimension {len(state_max)} != state_dim {config.get('state_dim')}")


def _manifest_robot_stats(manifest: dict[str, Any]) -> dict[str, Any]:
    normalization = manifest.get("normalization") or {}
    stats_by_robot = normalization.get("stats") or {}
    robot_key = normalization.get("robot_key")
    if robot_key and robot_key in stats_by_robot:
        return dict(stats_by_robot[robot_key])
    if len(stats_by_robot) == 1:
        return dict(next(iter(stats_by_robot.values())))
    return {}
