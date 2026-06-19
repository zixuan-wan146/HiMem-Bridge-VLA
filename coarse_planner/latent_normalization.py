from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class LatentNormalizer:
    mean: torch.Tensor
    std: torch.Tensor
    count: int
    std_floor: float = 1.0e-4

    def __post_init__(self) -> None:
        if self.mean.ndim != 1:
            raise ValueError(f"mean must have shape [Z], got {tuple(self.mean.shape)}")
        if self.std.shape != self.mean.shape:
            raise ValueError(f"std shape {tuple(self.std.shape)} != mean shape {tuple(self.mean.shape)}")
        if self.count <= 0:
            raise ValueError(f"count must be positive, got {self.count}")
        self.std = self.std.clamp_min(float(self.std_floor))

    def to(self, *, device: str | torch.device | None = None, dtype: torch.dtype | None = None) -> LatentNormalizer:
        return LatentNormalizer(
            mean=self.mean.to(device=device, dtype=dtype),
            std=self.std.to(device=device, dtype=dtype),
            count=int(self.count),
            std_floor=float(self.std_floor),
        )

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=latents.device, dtype=latents.dtype)
        std = self.std.to(device=latents.device, dtype=latents.dtype)
        return (latents - mean) / std

    def unnormalize(self, latents: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=latents.device, dtype=latents.dtype)
        std = self.std.to(device=latents.device, dtype=latents.dtype)
        return latents * std + mean

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.detach().cpu(),
            "std": self.std.detach().cpu(),
            "count": int(self.count),
            "std_floor": float(self.std_floor),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> LatentNormalizer:
        return cls(
            mean=torch.as_tensor(state["mean"], dtype=torch.float32),
            std=torch.as_tensor(state["std"], dtype=torch.float32),
            count=int(state["count"]),
            std_floor=float(state.get("std_floor", 1.0e-4)),
        )


def latent_normalization_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("latent_normalization", {}).get("enabled", False))


def latent_normalizer_stats_path(config: dict[str, Any], run_dir: str | Path) -> Path:
    norm_config = config.get("latent_normalization", {})
    stats_path = norm_config.get("stats_path")
    if stats_path:
        return Path(str(stats_path)).expanduser()
    filename = str(norm_config.get("stats_filename", "latent_normalizer.pt"))
    return Path(run_dir).expanduser() / filename


def load_latent_normalizer(path: str | Path, *, device: str | torch.device | None = None) -> LatentNormalizer:
    state = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    if "latent_normalizer" in state:
        state = state["latent_normalizer"]
    normalizer = LatentNormalizer.from_state_dict(state)
    return normalizer.to(device=device, dtype=torch.float32) if device is not None else normalizer


def latent_normalizer_from_checkpoint(
    checkpoint: dict[str, Any],
    *,
    device: str | torch.device | None = None,
) -> LatentNormalizer | None:
    state = checkpoint.get("latent_normalizer")
    if state is None:
        state = checkpoint.get("latent_normalizer_state_dict")
    if state is None:
        return None
    normalizer = LatentNormalizer.from_state_dict(state)
    return normalizer.to(device=device, dtype=torch.float32) if device is not None else normalizer


def save_latent_normalizer(normalizer: LatentNormalizer, path: str | Path) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"latent_normalizer": normalizer.state_dict()}, path)


@torch.no_grad()
def compute_latent_normalizer(
    segment_autoencoder: torch.nn.Module,
    loader: Any,
    *,
    device: str | torch.device,
    amp_enabled: bool,
    std_floor: float = 1.0e-4,
) -> LatentNormalizer:
    segment_autoencoder.eval()
    total = None
    total_sq = None
    count = 0
    for batch in loader:
        action_segments = batch["action_segments"].to(device)
        segment_mask = batch["action_segment_mask"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
            latents = segment_autoencoder.encode(action_segments)
        flat_latents = latents.float().reshape(-1, latents.shape[-1])
        flat_mask = segment_mask.reshape(-1).to(device=flat_latents.device).bool()
        active = flat_latents[flat_mask]
        if active.numel() == 0:
            continue
        active64 = active.to(dtype=torch.float64)
        batch_sum = active64.sum(dim=0)
        batch_sq = active64.pow(2).sum(dim=0)
        total = batch_sum if total is None else total + batch_sum
        total_sq = batch_sq if total_sq is None else total_sq + batch_sq
        count += int(active64.shape[0])
    if total is None or total_sq is None or count <= 0:
        raise ValueError("cannot compute latent normalization stats from an empty active segment set")
    mean = total / float(count)
    var = (total_sq / float(count) - mean.pow(2)).clamp_min(0.0)
    std = var.sqrt().clamp_min(float(std_floor))
    return LatentNormalizer(
        mean=mean.to(dtype=torch.float32).cpu(),
        std=std.to(dtype=torch.float32).cpu(),
        count=count,
        std_floor=float(std_floor),
    )


def resolve_latent_normalizer(
    config: dict[str, Any],
    *,
    run_dir: str | Path,
    segment_autoencoder: torch.nn.Module,
    train_loader: Any,
    device: str | torch.device,
    amp_enabled: bool,
) -> LatentNormalizer | None:
    if not latent_normalization_enabled(config):
        return None
    path = latent_normalizer_stats_path(config, run_dir)
    norm_config = config.get("latent_normalization", {})
    if bool(norm_config.get("reuse_existing_stats", True)) and path.exists():
        return load_latent_normalizer(path, device=device)
    normalizer = compute_latent_normalizer(
        segment_autoencoder,
        train_loader,
        device=device,
        amp_enabled=amp_enabled,
        std_floor=float(norm_config.get("std_floor", 1.0e-4)),
    )
    save_latent_normalizer(normalizer, path)
    return normalizer.to(device=device, dtype=torch.float32)
