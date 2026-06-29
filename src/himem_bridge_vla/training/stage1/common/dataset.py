from __future__ import annotations

import logging
from functools import partial
import json
from pathlib import Path
from typing import Any

from himem_bridge_vla.dataset import EPISODE_FEATURE_CACHE_FORMAT
from himem_bridge_vla.dataset import EpisodeFeatureCacheTrajectoryDataset
from himem_bridge_vla.dataset import collate_direct_bridge_token_cache_windows
from himem_bridge_vla.path_utils import display_project_path, project_path
from himem_bridge_vla.reproducibility import build_torch_generator, seed_data_worker


def prepare_stage1_dataset(
    config: dict[str, Any],
    *,
    repo_root: str | Path,
) -> EpisodeFeatureCacheTrajectoryDataset:
    manifest_path = project_path(config.get("dataset_config_path"), repo_root, label="--dataset_config_path")
    manifest_format = _read_manifest_format(manifest_path)
    if manifest_format != EPISODE_FEATURE_CACHE_FORMAT:
        raise ValueError(
            f"Stage1 training requires {EPISODE_FEATURE_CACHE_FORMAT} manifest, got {manifest_format!r}. "
            "Build it with the benchmark-specific episode replay index and episode feature cache scripts."
        )
    dataset = EpisodeFeatureCacheTrajectoryDataset(
        manifest_path,
        action_horizon=int(config.get("horizon", 32)),
        max_episodes=config.get("max_samples_per_file"),
    )
    logging.info(
        "Loaded Stage1 episode feature cache: episodes=%s format=%s manifest=%s",
        len(dataset),
        manifest_format,
        display_project_path(manifest_path, repo_root),
    )
    return dataset


def prepare_stage1_dataloader(dataset: EpisodeFeatureCacheTrajectoryDataset, config: dict[str, Any]):
    try:
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyTorch is required for Stage1 dataloading") from exc

    batch_size = int(config.get("batch_size", 4))
    num_workers = int(config.get("num_workers", 4))
    seed = int(config.get("seed", 42))
    shuffle = bool(config.get("shuffle_trajectory_windows", False))
    if len(dataset) == 0:
        raise ValueError("Stage1 dataset is empty")

    collate_fn = partial(
        collate_direct_bridge_token_cache_windows,
        memory_entry_tokens=int(config.get("memory_entry_tokens", 16)),
        action_horizon=int(config.get("horizon", 32)),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_data_worker,
        generator=build_torch_generator(seed),
    )
    if len(dataloader) == 0:
        raise ValueError(
            f"Stage1 dataloader has no episode batches. Dataset size={len(dataset)}, "
            f"batch_size={batch_size}, drop_last=True."
        )
    logging.info(
        "Initialized Stage1 dataloader: episode_batch_size=%s num_workers=%s shuffle_episodes=%s",
        batch_size,
        num_workers,
        shuffle,
    )
    return dataloader


def _read_manifest_format(manifest_path: Path) -> str | None:
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return None if payload.get("format") is None else str(payload["format"])
