from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from himem_bridge_vla.dataset import MemoryTokenCacheDataset
from himem_bridge_vla.dataset import collate_memory_token_cache_samples
from himem_bridge_vla.model.himem import VisualMemoryCompressor
from himem_bridge_vla.reproducibility import build_torch_generator
from himem_bridge_vla.reproducibility import seed_data_worker
from himem_bridge_vla.reproducibility import set_global_seed
from himem_bridge_vla.reproducibility import write_experiment_snapshot
from himem_bridge_vla.training.memory_context import build_token_cache_memory_context


@dataclass(frozen=True)
class MemoryTokenCacheTrainingConfig:
    cache_manifest: str
    output_dir: str
    device: str = "cuda"
    batch_size: int = 8
    max_steps: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    max_samples: int | None = None
    seed: int = 42
    deterministic: bool = False
    tokens_per_entry: int = 1
    num_heads: int = 4
    dropout: float = 0.0
    hidden_multiplier: int = 2
    log_interval: int = 10
    ckpt_interval: int = 0
    view_names: tuple[str, ...] | None = None
    repo_root: str | None = None


@dataclass(frozen=True)
class MemoryTokenCacheTrainingResult:
    output_dir: Path
    final_loss: float
    steps: int
    checkpoint_path: Path


class MemoryTokenActionAdapter(nn.Module):
    """Small training adapter over visual token cache and Dual-FIFO short memory.

    This module deliberately stays on the memory side: it consumes current visual
    tokens, short-memory tokens, state, and action chunks from the token cache.
    It does not implement KFS or Bridge-Attention consumption.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        view_names: Sequence[str],
        state_dim: int,
        action_horizon: int,
        action_dim: int,
        tokens_per_entry: int = 1,
        num_heads: int = 4,
        dropout: float = 0.0,
        hidden_multiplier: int = 2,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if hidden_dim % int(num_heads) != 0:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}")
        if not view_names:
            raise ValueError("view_names must contain at least one view")

        self.hidden_dim = int(hidden_dim)
        self.view_names = tuple(str(view_name) for view_name in view_names)
        self.state_dim = int(state_dim)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.memory_compressor = VisualMemoryCompressor(
            hidden_dim=self.hidden_dim,
            view_names=self.view_names,
            tokens_per_entry=int(tokens_per_entry),
            num_heads=int(num_heads),
            dropout=float(dropout),
        )
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        combined_dim = self.hidden_dim * 3
        mlp_hidden_dim = max(self.hidden_dim, int(hidden_multiplier) * self.hidden_dim)
        self.action_head = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, self.action_horizon * self.action_dim),
        )

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        states = torch.as_tensor(batch["current_state"], dtype=dtype, device=device)
        if states.shape[-1] != self.state_dim:
            raise ValueError(f"state dim {states.shape[-1]} != configured state_dim {self.state_dim}")

        current_reprs = []
        for index, current_tokens_by_view in enumerate(batch["current_tokens_by_view"]):
            current_reprs.append(self._pool_tokens_by_view(current_tokens_by_view, device=device, dtype=dtype))

        current_repr = torch.stack(current_reprs, dim=0)
        memory_batch = build_token_cache_memory_context(
            batch,
            self.memory_compressor,
            device=device,
            dtype=dtype,
        )
        memory_repr = _pool_memory_context(memory_batch.memory_context, memory_batch.memory_context_mask)
        state_repr = self.state_encoder(states)
        combined = torch.cat([current_repr, memory_repr, state_repr], dim=-1)
        actions = self.action_head(combined)
        return actions.view(states.shape[0], self.action_horizon, self.action_dim)

    def _pool_tokens_by_view(
        self,
        tokens_by_view: Mapping[str, torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        unknown_views = set(tokens_by_view) - set(self.view_names)
        if unknown_views:
            raise ValueError(f"current tokens contain unconfigured views: {sorted(unknown_views)}")
        parts = []
        for view_name in self.view_names:
            tokens = tokens_by_view.get(view_name)
            if tokens is None:
                continue
            tensor = torch.as_tensor(tokens, device=device, dtype=dtype)
            if tensor.ndim != 2 or tensor.shape[-1] != self.hidden_dim:
                raise ValueError(
                    f"tokens for view {view_name!r} must have shape [N, {self.hidden_dim}], "
                    f"got {tuple(tensor.shape)}"
                )
            parts.append(tensor)
        if not parts:
            raise ValueError("current_tokens_by_view has no configured views")
        return torch.cat(parts, dim=0).mean(dim=0)


def masked_action_chunk_mse(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    if pred_actions.shape != target_actions.shape:
        raise ValueError(
            f"pred_actions shape {tuple(pred_actions.shape)} != target_actions {tuple(target_actions.shape)}"
        )
    if pred_actions.ndim != 3:
        raise ValueError(f"actions must have shape [B, H, A], got {tuple(pred_actions.shape)}")

    mask = action_mask.to(device=pred_actions.device, dtype=pred_actions.dtype)
    if mask.ndim == 2:
        mask = mask.unsqueeze(-1)
    if mask.shape == pred_actions.shape[:2] + (1,):
        mask = mask.expand_as(pred_actions)
    if mask.shape != pred_actions.shape:
        raise ValueError(
            f"action_mask shape {tuple(action_mask.shape)} is not compatible with {tuple(pred_actions.shape)}"
        )
    active = mask.sum()
    if active.item() == 0:
        raise ValueError("action_mask has no active action steps")
    return ((pred_actions - target_actions).pow(2) * mask).sum() / active


def run_memory_token_cache_training(
    config: MemoryTokenCacheTrainingConfig | Mapping[str, Any],
) -> MemoryTokenCacheTrainingResult:
    config = _coerce_config(config)
    _validate_training_config(config)
    set_global_seed(config.seed, deterministic=config.deterministic)

    device = _resolve_device(config.device)
    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_config = asdict(config)
    if config.repo_root is not None:
        snapshot_config["repo_root"] = config.repo_root
    write_experiment_snapshot(output_dir, snapshot_config)

    dataset = MemoryTokenCacheDataset(config.cache_manifest, max_samples=config.max_samples)
    first_sample = dataset[0]
    view_names = config.view_names or tuple(first_sample["current_tokens_by_view"].keys())
    hidden_dim = int(dataset.config.hidden_dim)
    state_dim = int(first_sample["current_state"].numel())
    action_horizon, action_dim = _action_shape(first_sample)

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
        collate_fn=collate_memory_token_cache_samples,
        worker_init_fn=seed_data_worker,
        generator=build_torch_generator(config.seed),
    )
    model = MemoryTokenActionAdapter(
        hidden_dim=hidden_dim,
        view_names=view_names,
        state_dim=state_dim,
        action_horizon=action_horizon,
        action_dim=action_dim,
        tokens_per_entry=config.tokens_per_entry,
        num_heads=config.num_heads,
        dropout=config.dropout,
        hidden_multiplier=config.hidden_multiplier,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    model.train()
    data_iter = iter(dataloader)
    metrics = []
    final_loss = float("nan")
    for step in range(1, config.max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        optimizer.zero_grad(set_to_none=True)
        pred_actions = model(batch)
        target_actions = torch.as_tensor(batch["future_actions"], device=device, dtype=pred_actions.dtype)
        action_mask = torch.as_tensor(batch["action_mask"], device=device)
        loss = masked_action_chunk_mse(pred_actions, target_actions, action_mask)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()

        final_loss = float(loss.detach().cpu().item())
        metrics.append({"step": step, "loss": final_loss, "grad_norm": float(torch.as_tensor(grad_norm).cpu().item())})
        if config.ckpt_interval > 0 and step % config.ckpt_interval == 0:
            _save_checkpoint(output_dir / f"adapter_step_{step:06d}.pt", model, optimizer, config, metrics[-1])

    checkpoint_path = output_dir / "adapter_last.pt"
    _save_checkpoint(checkpoint_path, model, optimizer, config, metrics[-1])
    _write_json(output_dir / "metrics.json", {"steps": metrics, "final_loss": final_loss})
    return MemoryTokenCacheTrainingResult(
        output_dir=output_dir,
        final_loss=final_loss,
        steps=config.max_steps,
        checkpoint_path=checkpoint_path,
    )


def _pool_memory_context(memory_context: torch.Tensor, memory_context_mask: torch.Tensor) -> torch.Tensor:
    mask = memory_context_mask.to(device=memory_context.device, dtype=memory_context.dtype).unsqueeze(-1)
    active = mask.sum(dim=1).clamp_min(1.0)
    return (memory_context * mask).sum(dim=1) / active


def _coerce_config(config: MemoryTokenCacheTrainingConfig | Mapping[str, Any]) -> MemoryTokenCacheTrainingConfig:
    if isinstance(config, MemoryTokenCacheTrainingConfig):
        return config
    payload = dict(config)
    if payload.get("view_names") is not None:
        payload["view_names"] = tuple(str(value) for value in payload["view_names"])
    return MemoryTokenCacheTrainingConfig(**payload)


def _validate_training_config(config: MemoryTokenCacheTrainingConfig) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if config.lr <= 0.0:
        raise ValueError("lr must be positive")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if config.grad_clip_norm <= 0.0:
        raise ValueError("grad_clip_norm must be positive")
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if config.tokens_per_entry <= 0:
        raise ValueError("tokens_per_entry must be positive")
    if config.num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if config.hidden_multiplier <= 0:
        raise ValueError("hidden_multiplier must be positive")
    if config.ckpt_interval < 0:
        raise ValueError("ckpt_interval must be non-negative")


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {value!r}, but CUDA is not available")
    return device


def _action_shape(sample: Mapping[str, Any]) -> tuple[int, int]:
    actions = torch.as_tensor(sample["future_actions"])
    if actions.ndim != 2:
        raise ValueError(f"future_actions must have shape [H, A], got {tuple(actions.shape)}")
    return int(actions.shape[0]), int(actions.shape[1])


def _save_checkpoint(
    path: Path,
    model: MemoryTokenActionAdapter,
    optimizer: torch.optim.Optimizer,
    config: MemoryTokenCacheTrainingConfig,
    metrics: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "memory_token_cache_action_adapter",
            "version": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "metrics": dict(metrics),
        },
        path,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
