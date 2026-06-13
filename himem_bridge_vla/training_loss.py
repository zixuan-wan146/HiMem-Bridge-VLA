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
