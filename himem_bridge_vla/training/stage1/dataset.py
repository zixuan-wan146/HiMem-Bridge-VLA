from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Any

from himem_bridge_vla.dataset import MemoryTokenCacheTrajectoryDataset, collate_direct_bridge_token_cache_windows
from himem_bridge_vla.path_utils import display_project_path, project_path
from himem_bridge_vla.reproducibility import build_torch_generator, seed_data_worker


def prepare_stage1_dataset(config: dict[str, Any], *, repo_root: str | Path) -> MemoryTokenCacheTrajectoryDataset:
    manifest_path = project_path(config.get("dataset_config_path"), repo_root, label="--dataset_config_path")
    dataset = MemoryTokenCacheTrajectoryDataset(
        manifest_path,
        burnin_replan_steps=int(config.get("burnin_replan_steps", 8)),
        loss_replan_steps=int(config.get("loss_replan_steps", 8)),
        allow_short_burnin=bool(config.get("allow_short_burnin", True)),
        action_horizon=int(config.get("horizon", 32)),
        window_stride=int(config.get("trajectory_window_stride", 1)),
        max_samples=config.get("max_samples_per_file"),
    )
    logging.info(
        "Loaded Stage1 trajectory token cache: windows=%s manifest=%s",
        len(dataset),
        display_project_path(manifest_path, repo_root),
    )
    return dataset


def prepare_stage1_dataloader(dataset: MemoryTokenCacheTrajectoryDataset, config: dict[str, Any]):
    try:
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyTorch is required for Stage1 dataloading") from exc

    batch_size = int(config.get("batch_size", 4))
    num_workers = int(config.get("num_workers", 4))
    seed = int(config.get("seed", 42))
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
        shuffle=True,
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
            f"Stage1 dataloader has no batches. Dataset size={len(dataset)}, batch_size={batch_size}, drop_last=True."
        )
    logging.info("Initialized Stage1 dataloader: batch_size=%s num_workers=%s", batch_size, num_workers)
    return dataloader
