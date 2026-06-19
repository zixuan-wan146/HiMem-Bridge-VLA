from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from coarse_planner.latent_normalization import LatentNormalizer
from himem_bridge_vla.model.planner import action_segment_reconstruction_loss
from himem_bridge_vla.training_loss import coarse_planner_intent_loss


@torch.no_grad()
def evaluate_planner(
    model: torch.nn.Module,
    segment_autoencoder: torch.nn.Module,
    loader: DataLoader,
    config: dict[str, Any],
    *,
    device: str | torch.device,
    latent_normalizer: LatentNormalizer | None = None,
) -> dict[str, float]:
    model.eval()
    segment_autoencoder.eval()
    total_loss = 0.0
    total_normalized_latent_mse = 0.0
    total_raw_latent_mse = 0.0
    total_chunk_loss = 0.0
    total_cosine = 0.0
    total_active = 0.0
    suffix_totals = _suffix_metric_accumulators(config)
    batches = 0
    loss_config = config.get("loss", {})
    amp_enabled = bool(config.get("training", {}).get("amp", str(device).startswith("cuda"))) and str(device).startswith("cuda")
    for batch in loader:
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
            vlm_tokens = batch["vlm_tokens"].to(device)
            state = batch["state"].to(device)
            action_segments = batch["action_segments"].to(device)
            mask = batch["action_segment_mask"].to(device)
            output = model(vlm_tokens, state)
            target_raw_latents = segment_autoencoder.encode(action_segments)
            if latent_normalizer is None:
                target_loss_latents = target_raw_latents
                predicted_raw_latents = output.predicted_latents
                predicted_loss_latents = output.predicted_latents
            else:
                target_loss_latents = latent_normalizer.normalize(target_raw_latents)
                predicted_loss_latents = output.predicted_latents
                predicted_raw_latents = latent_normalizer.unnormalize(output.predicted_latents)
            decoded_segments = segment_autoencoder.decode(predicted_raw_latents)
            loss = coarse_planner_intent_loss(
                predicted_loss_latents,
                target_loss_latents,
                decoded_segments,
                action_segments,
                mask,
                latent_loss_weight=float(loss_config.get("latent_loss_weight", 1.0)),
                chunk_loss_weight=float(loss_config.get("chunk_loss_weight", 1.0)),
                gripper_indices=loss_config.get("gripper_indices", [-1]),
                gripper_loss_weight=float(loss_config.get("gripper_loss_weight", 1.0)),
                token_loss_weights=loss_config.get("token_loss_weights"),
            )
            chunk_loss = action_segment_reconstruction_loss(
                decoded_segments,
                action_segments,
                mask,
                gripper_indices=loss_config.get("gripper_indices", [-1]),
                gripper_loss_weight=float(loss_config.get("gripper_loss_weight", 1.0)),
            )
        active = mask.sum().clamp_min(1.0)
        normalized_latent_mse = (predicted_loss_latents - target_loss_latents).pow(2).mean(dim=-1)
        raw_latent_mse = (predicted_raw_latents - target_raw_latents).pow(2).mean(dim=-1)
        cosine = F.cosine_similarity(predicted_raw_latents.float(), target_raw_latents.float(), dim=-1)
        metric_mask = mask.to(device=normalized_latent_mse.device, dtype=normalized_latent_mse.dtype)
        total_loss += float(loss.detach().cpu().item())
        total_normalized_latent_mse += float((normalized_latent_mse * metric_mask).sum().detach().cpu().item())
        total_raw_latent_mse += float((raw_latent_mse * metric_mask).sum().detach().cpu().item())
        total_chunk_loss += float(chunk_loss.detach().cpu().item())
        total_cosine += float((cosine * metric_mask).sum().detach().cpu().item())
        total_active += float(active.detach().cpu().item())
        _update_suffix_metrics(suffix_totals, normalized_latent_mse, mask)
        batches += 1
    normalized_latent_mse = total_normalized_latent_mse / max(total_active, 1.0)
    metrics = {
        "loss": total_loss / max(batches, 1),
        "latent_mse": normalized_latent_mse,
        "normalized_latent_mse": normalized_latent_mse,
        "raw_latent_mse": total_raw_latent_mse / max(total_active, 1.0),
        "decoded_chunk_loss": total_chunk_loss / max(batches, 1),
        "latent_cosine_similarity": total_cosine / max(total_active, 1.0),
        "batches": float(batches),
    }
    for consumed_tokens, values in suffix_totals.items():
        value = values["sum"] / max(values["count"], 1.0)
        metrics[f"latent_mse_u{consumed_tokens}"] = value
        metrics[f"normalized_latent_mse_u{consumed_tokens}"] = value
    return metrics


def _suffix_metric_accumulators(config: dict[str, Any]) -> dict[int, dict[str, float]]:
    diagnostics = config.get("coarse_planner_suffix_diagnostics") or config.get("evaluation", {})
    raw_offsets = diagnostics.get("consumed_tokens_set", [0, 2, 4, 6])
    target_config = config.get("target", {})
    num_plan_steps = int(target_config.get("num_plan_steps", 8))
    offsets = []
    for value in raw_offsets:
        consumed = int(value)
        if 0 <= consumed < num_plan_steps:
            offsets.append(consumed)
    return {offset: {"sum": 0.0, "count": 0.0} for offset in offsets}


def _update_suffix_metrics(
    suffix_totals: dict[int, dict[str, float]],
    latent_mse: torch.Tensor,
    segment_mask: torch.Tensor,
) -> None:
    if not suffix_totals:
        return
    base_mask = segment_mask.to(device=latent_mse.device, dtype=latent_mse.dtype)
    if base_mask.ndim == 3 and base_mask.shape[-1] == 1:
        base_mask = base_mask.squeeze(-1)
    for consumed_tokens, values in suffix_totals.items():
        suffix_mask = torch.zeros_like(base_mask)
        suffix_mask[:, consumed_tokens:] = base_mask[:, consumed_tokens:]
        active = suffix_mask.sum()
        if active.item() <= 0:
            continue
        values["sum"] += float((latent_mse * suffix_mask).sum().detach().cpu().item())
        values["count"] += float(active.detach().cpu().item())
