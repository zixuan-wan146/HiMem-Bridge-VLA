from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from himem_bridge_vla.path_utils import project_path


STAGE2_REPLAY_INDEX_FORMAT = "libero_episode_replay_index"
STAGE2_DATASET_TYPE = "libero_raw_episode"


def enforce_stage2_contract(config: Mapping[str, Any]) -> None:
    dataset_type = str(config.get("dataset_type", ""))
    if dataset_type != STAGE2_DATASET_TYPE:
        raise ValueError(
            f"Stage2 full E2E training requires dataset_type={STAGE2_DATASET_TYPE}, got {dataset_type!r}"
        )

    required_true = (
        "load_vlm",
        "finetune_vlm",
        "finetune_action_head",
        "progress_planner_enabled",
        "finetune_progress_planner",
    )
    for key in required_true:
        if not bool(config.get(key, False)):
            raise ValueError(f"Stage2 full E2E training requires {key}=true")

    if bool(config.get("memory_token_cache_sequence_training", False)):
        raise ValueError("Stage2 full E2E training must not use token-cache sequence training")
    if bool(config.get("enable_bridge_aux_loss", False)):
        raise ValueError("Stage2 first pass uses action FM only; enable_bridge_aux_loss must be false")
    if config.get("min_cuda_memory_gb") is not None:
        raise ValueError("Stage2 VRAM target must come from real training workload, not min_cuda_memory_gb")

    sampling_mode = str(config.get("stage2_sampling_mode", "group"))
    if sampling_mode != "group":
        raise ValueError(f"Stage2 currently supports MemoryVLA-style group sampling only, got {sampling_mode!r}")

    sequence_len = _as_int(config.get("sequence_len", 0), "--sequence_len")
    if sequence_len <= 0:
        raise ValueError(f"--sequence_len must be positive, got {sequence_len}")

    loss = config.get("loss") or {}
    if loss:
        if not isinstance(loss, Mapping):
            raise ValueError("--loss must be a mapping")
        action_fm = float(loss.get("action_fm", 0.0))
        if action_fm <= 0.0:
            raise ValueError("Stage2 requires loss.action_fm > 0")
        for key in ("vlm_ce", "planner_aux", "gripper_bce"):
            if float(loss.get(key, 0.0)) != 0.0:
                raise ValueError(f"Stage2 action-FM-only profile requires loss.{key}=0.0")


def validate_stage2_replay_index_contract(config: Mapping[str, Any], *, repo_root: str | Path) -> None:
    index_path = project_path(config["dataset_config_path"], repo_root, label="--dataset_config_path")
    with index_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    index_format = payload.get("format")
    if index_format != STAGE2_REPLAY_INDEX_FORMAT:
        raise ValueError(
            f"Stage2 training requires {STAGE2_REPLAY_INDEX_FORMAT} index, got {index_format!r}. "
            "Build it with scripts/cache/build_libero_episode_replay_index.py."
        )

    benchmark = str(payload.get("benchmark", "")).upper()
    if benchmark != "LIBERO":
        raise ValueError(f"Stage2 LIBERO training requires benchmark=LIBERO, got {benchmark!r}")

    episodes = payload.get("episodes")
    if not isinstance(episodes, Sequence) or isinstance(episodes, (str, bytes)) or not episodes:
        raise ValueError("Stage2 replay index must contain a non-empty episodes list")

    horizon = _as_int(config.get("horizon", 32), "--horizon")
    index_horizon = payload.get("action_horizon")
    if index_horizon is not None and int(index_horizon) != horizon:
        raise ValueError(f"Stage2 horizon {horizon} does not match replay index action_horizon {index_horizon}")

    valid_episode_count = 0
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise ValueError("Stage2 replay index episodes must be mappings")
        for key in ("episode_id", "episode_key", "source_path", "episode_length"):
            if episode.get(key) in (None, ""):
                raise ValueError(f"Stage2 replay index episode is missing {key!r}")
        episode_length = _as_int(episode["episode_length"], "episode_length")
        if episode_length >= horizon:
            valid_episode_count += 1

    if valid_episode_count <= 0:
        raise ValueError(f"Stage2 replay index has no episodes with at least horizon={horizon} steps")


def _as_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer, got {value!r}") from exc
