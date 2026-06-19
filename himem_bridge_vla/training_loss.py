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


def masked_latent_mse_loss(
    predicted_latents: Any,
    target_latents: Any,
    segment_mask: Any,
    *,
    token_loss_weights: Any = None,
) -> Any:
    """Masked MSE over action-segment intent latents."""

    if predicted_latents.shape != target_latents.shape:
        raise ValueError(
            f"predicted_latents shape {predicted_latents.shape} != target_latents shape {target_latents.shape}"
        )
    if predicted_latents.ndim != 3:
        raise ValueError(f"planner latents must have shape [B, K, Z], got {predicted_latents.shape}")

    mask = segment_mask.to(device=predicted_latents.device, dtype=predicted_latents.dtype)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    if mask.shape != predicted_latents.shape[:2]:
        raise ValueError(f"segment_mask shape {mask.shape} != planner latent shape {predicted_latents.shape[:2]}")
    mask = _apply_token_loss_weights(mask, token_loss_weights)
    active_steps = mask.sum()
    if active_steps.item() == 0:
        raise ValueError("action_segment_mask.sum() is 0. All action segments are masked.")

    squared = (predicted_latents - target_latents.to(device=predicted_latents.device, dtype=predicted_latents.dtype)).pow(2)
    per_step = squared.mean(dim=-1)
    return (per_step * mask).sum() / active_steps


def coarse_planner_intent_loss(
    predicted_latents: Any,
    target_latents: Any,
    decoded_segments: Any,
    target_segments: Any,
    segment_mask: Any,
    *,
    latent_loss_weight: float = 1.0,
    chunk_loss_weight: float = 1.0,
    gripper_indices: Any = (-1,),
    gripper_loss_weight: float = 1.0,
    token_loss_weights: Any = None,
) -> Any:
    """Planner auxiliary loss: latent regression plus decoded chunk reconstruction."""

    from himem_bridge_vla.model.planner import action_segment_reconstruction_loss

    weighted_mask = _apply_token_loss_weights(
        segment_mask.to(device=predicted_latents.device, dtype=predicted_latents.dtype),
        token_loss_weights,
    )
    latent_loss = masked_latent_mse_loss(
        predicted_latents,
        target_latents,
        weighted_mask,
    )
    chunk_loss = action_segment_reconstruction_loss(
        decoded_segments,
        target_segments,
        weighted_mask,
        gripper_indices=gripper_indices,
        gripper_loss_weight=gripper_loss_weight,
    )
    return float(latent_loss_weight) * latent_loss + float(chunk_loss_weight) * chunk_loss


def _apply_token_loss_weights(mask: Any, token_loss_weights: Any = None) -> Any:
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    if token_loss_weights is None:
        return mask
    import torch

    weights = torch.as_tensor(token_loss_weights, device=mask.device, dtype=mask.dtype)
    if weights.ndim != 1:
        raise ValueError(f"token_loss_weights must be a 1D sequence, got shape {tuple(weights.shape)}")
    if weights.shape[0] != mask.shape[-1]:
        raise ValueError(f"token_loss_weights length {weights.shape[0]} != num plan tokens {mask.shape[-1]}")
    if weights.min().item() <= 0.0:
        raise ValueError("token_loss_weights must be positive")
    return mask * weights.unsqueeze(0)


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
