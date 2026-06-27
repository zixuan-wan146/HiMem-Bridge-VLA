from __future__ import annotations

from typing import Any

from himem_bridge_vla.training_loss import masked_flow_matching_mse


def stage1_flow_matching_loss(
    *,
    pred_velocity: Any,
    noise: Any,
    actions_gt: Any,
    action_mask: Any,
) -> Any:
    target_velocity = (actions_gt - noise).view(actions_gt.shape[0], -1)
    if pred_velocity.shape != target_velocity.shape:
        raise ValueError(
            f"pred_velocity shape {tuple(pred_velocity.shape)} != target_velocity shape {tuple(target_velocity.shape)}"
        )
    return masked_flow_matching_mse(pred_velocity, target_velocity, action_mask)
