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


def coarse_planner_smooth_l1_loss(
    predicted_actions: Any,
    target_actions: Any,
    step_mask: Any,
    *,
    gripper_indices: Any = (-1,),
    gripper_loss_weight: float = 1.0,
    smoothness_weight: float = 0.0,
) -> Any:
    """Masked Smooth L1 loss for coarse future-action supervision."""

    import torch
    import torch.nn.functional as F

    if predicted_actions.shape != target_actions.shape:
        raise ValueError(
            f"predicted_actions shape {predicted_actions.shape} != target_actions shape {target_actions.shape}"
        )
    if predicted_actions.ndim != 3:
        raise ValueError(f"coarse planner actions must have shape [B, K, A], got {predicted_actions.shape}")

    mask = step_mask.to(device=predicted_actions.device, dtype=predicted_actions.dtype)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    if mask.shape != predicted_actions.shape[:2]:
        raise ValueError(f"step_mask shape {mask.shape} != coarse step shape {predicted_actions.shape[:2]}")
    active_steps = mask.sum()
    if active_steps.item() == 0:
        raise ValueError("coarse_action_mask.sum() is 0. All coarse plan steps are masked.")

    action_dim = predicted_actions.shape[-1]
    dim_weights = torch.ones(action_dim, device=predicted_actions.device, dtype=predicted_actions.dtype)
    for index in _normalize_indices(gripper_indices, action_dim):
        dim_weights[index] = float(gripper_loss_weight)
    if dim_weights.sum().item() <= 0:
        raise ValueError("coarse planner dimension weights must contain at least one positive value")

    element_loss = F.smooth_l1_loss(
        predicted_actions,
        target_actions.to(device=predicted_actions.device, dtype=predicted_actions.dtype),
        reduction="none",
    )
    per_step_loss = (element_loss * dim_weights.view(1, 1, -1)).sum(dim=-1) / dim_weights.sum()
    loss = (per_step_loss * mask).sum() / active_steps

    if smoothness_weight > 0.0 and predicted_actions.shape[1] > 1:
        pair_mask = mask[:, 1:] * mask[:, :-1]
        active_pairs = pair_mask.sum()
        if active_pairs.item() > 0:
            smoothness = (predicted_actions[:, 1:] - predicted_actions[:, :-1]).abs().mean(dim=-1)
            loss = loss + float(smoothness_weight) * (smoothness * pair_mask).sum() / active_pairs

    return loss


def _normalize_indices(indices: Any, action_dim: int) -> tuple[int, ...]:
    if indices is None:
        return ()
    if isinstance(indices, int):
        raw_indices = (indices,)
    else:
        raw_indices = tuple(indices)
    normalized = []
    for index in raw_indices:
        value = int(index)
        if value < 0:
            value += action_dim
        if value < 0 or value >= action_dim:
            raise ValueError(f"gripper index {index} is out of range for action_dim {action_dim}")
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)
