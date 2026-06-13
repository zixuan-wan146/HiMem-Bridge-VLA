from __future__ import annotations

from typing import Any


def masked_flow_matching_mse(pred_velocity: Any, target_velocity: Any, action_mask: Any) -> Any:
    """Mean squared error over active action dimensions only."""

    if pred_velocity.shape != target_velocity.shape:
        raise ValueError(f"pred_velocity shape {pred_velocity.shape} != target_velocity shape {target_velocity.shape}")

    if action_mask.ndim < 2:
        raise ValueError(f"action_mask must include batch and action dimensions, got shape {action_mask.shape}")

    flat_mask = action_mask.reshape(action_mask.shape[0], -1).to(
        device=pred_velocity.device,
        dtype=pred_velocity.dtype,
    )
    if flat_mask.shape != pred_velocity.shape:
        raise ValueError(f"action_mask shape {flat_mask.shape} != velocity shape {pred_velocity.shape}")

    active_dims = flat_mask.sum()
    if active_dims.item() == 0:
        raise ValueError(
            "action_mask.sum() is 0. All actions are masked, which indicates a data or mask generation issue."
        )

    squared_error = (pred_velocity - target_velocity).pow(2) * flat_mask
    return squared_error.sum() / active_dims


def boundary_bce_loss(boundary_logits: Any, boundary_labels: Any) -> Any:
    """Binary cross entropy for skill-boundary supervision."""

    import torch.nn.functional as F

    labels = boundary_labels.reshape(-1, 1).to(device=boundary_logits.device, dtype=boundary_logits.dtype)
    if labels.shape != boundary_logits.shape:
        raise ValueError(f"boundary label shape {labels.shape} != boundary_logits shape {boundary_logits.shape}")
    return F.binary_cross_entropy_with_logits(boundary_logits, labels)


def progress_smooth_l1_loss(progress_logits: Any, progress_labels: Any) -> Any:
    """Smooth L1 loss for segment progress labels in [0, 1]."""

    import torch
    import torch.nn.functional as F

    labels = progress_labels.reshape(-1, 1).to(device=progress_logits.device, dtype=progress_logits.dtype)
    if labels.shape != progress_logits.shape:
        raise ValueError(f"progress label shape {labels.shape} != progress_logits shape {progress_logits.shape}")
    prediction = torch.sigmoid(progress_logits)
    return F.smooth_l1_loss(prediction, labels)
