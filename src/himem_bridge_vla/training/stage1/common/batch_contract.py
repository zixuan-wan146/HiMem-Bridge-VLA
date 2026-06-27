from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


REQUIRED_TRAJECTORY_STEP_KEYS = (
    "batch_indices",
    "loss_mask",
    "states",
    "actions",
    "action_mask",
    "fused_tokens",
)


def validate_stage1_window_batch(batch: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    steps = batch.get("trajectory_steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)) or not steps:
        raise ValueError("Stage1 dataloader must return a non-empty trajectory_steps sequence")
    if int(batch.get("batch_size", 0)) <= 0:
        raise ValueError("Stage1 trajectory batch requires a positive batch_size")
    for index, step_batch in enumerate(steps):
        validate_stage1_step_batch(step_batch, index=index)
    return steps


def validate_stage1_step_batch(step_batch: Mapping[str, Any], *, index: int) -> None:
    missing = [key for key in REQUIRED_TRAJECTORY_STEP_KEYS if key not in step_batch]
    if missing:
        raise ValueError(f"Stage1 trajectory step {index} missing required keys: {', '.join(missing)}")
