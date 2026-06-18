from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from himem_bridge_vla.training_loss import coarse_planner_smooth_l1_loss


@torch.no_grad()
def evaluate_planner(model: torch.nn.Module, loader: DataLoader, config: dict[str, Any], *, device: str | torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_abs = 0.0
    total_active = 0.0
    batches = 0
    loss_config = config.get("loss", {})
    for batch in loader:
        vlm_tokens = batch["vlm_tokens"].to(device)
        state = batch["state"].to(device)
        targets = batch["coarse_actions"].to(device)
        mask = batch["coarse_action_mask"].to(device)
        output = model(vlm_tokens, state)
        loss = coarse_planner_smooth_l1_loss(
            output.coarse_actions,
            targets,
            mask,
            gripper_indices=loss_config.get("gripper_indices", [-1]),
            gripper_loss_weight=float(loss_config.get("gripper_loss_weight", 1.0)),
            smoothness_weight=float(loss_config.get("smoothness_weight", 0.0)),
        )
        active = mask.sum().clamp_min(1.0)
        mae = (output.coarse_actions - targets).abs().mean(dim=-1)
        total_loss += float(loss.detach().cpu().item())
        total_abs += float((mae * mask).sum().detach().cpu().item())
        total_active += float(active.detach().cpu().item())
        batches += 1
    return {
        "loss": total_loss / max(batches, 1),
        "mae": total_abs / max(total_active, 1.0),
        "batches": float(batches),
    }
