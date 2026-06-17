from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_weighted_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    pos_weight: float | torch.Tensor | None = None,
) -> torch.Tensor:
    labels = labels.reshape_as(logits).to(device=logits.device, dtype=logits.dtype)
    valid_mask = valid_mask.reshape_as(logits).to(device=logits.device, dtype=logits.dtype)
    if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
        pos_weight = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    if isinstance(pos_weight, torch.Tensor):
        pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)

    loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction="none")
    active = valid_mask.sum()
    if active.item() == 0:
        raise ValueError("valid_mask contains no active entries")
    return (loss * valid_mask).sum() / active


def masked_focal_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: float | None = None,
    pos_weight: float | torch.Tensor | None = None,
) -> torch.Tensor:
    labels = labels.reshape_as(logits).to(device=logits.device, dtype=logits.dtype)
    valid_mask = valid_mask.reshape_as(logits).to(device=logits.device, dtype=logits.dtype)
    if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
        pos_weight = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    if isinstance(pos_weight, torch.Tensor):
        pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)

    bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * labels + (1.0 - probs) * (1.0 - labels)
    focal = (1.0 - p_t).clamp_min(0.0).pow(float(gamma))
    if alpha is not None:
        alpha_t = float(alpha) * labels + (1.0 - float(alpha)) * (1.0 - labels)
        focal = focal * alpha_t
    active = valid_mask.sum()
    if active.item() == 0:
        raise ValueError("valid_mask contains no active entries")
    return (bce * focal * valid_mask).sum() / active


def transition_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    training_config: dict,
    *,
    pos_weight: float | torch.Tensor | None,
) -> torch.Tensor:
    loss_name = str(training_config.get("loss", "bce"))
    if loss_name == "bce":
        return masked_weighted_bce_with_logits(logits, labels, valid_mask, pos_weight=pos_weight)
    if loss_name == "focal":
        return masked_focal_bce_with_logits(
            logits,
            labels,
            valid_mask,
            gamma=float(training_config.get("focal_gamma", 2.0)),
            alpha=training_config.get("focal_alpha"),
            pos_weight=pos_weight,
        )
    raise ValueError("training.loss must be 'bce' or 'focal'")


def resolve_pos_weight(labels: torch.Tensor, valid_mask: torch.Tensor, policy: str | float | int) -> float:
    valid = valid_mask.bool()
    valid_labels = labels[valid]
    positives = (valid_labels > 0).sum().item()
    negatives = (valid_labels <= 0).sum().item()
    if positives <= 0:
        return 1.0
    if isinstance(policy, (int, float)):
        return float(policy)
    if policy == "neg_pos":
        return max(1.0, negatives / positives)
    if policy == "sqrt_neg_pos":
        return max(1.0, (negatives / positives) ** 0.5)
    if policy in {"none", "off"}:
        return 1.0
    raise ValueError(f"unknown pos_weight policy: {policy!r}")
