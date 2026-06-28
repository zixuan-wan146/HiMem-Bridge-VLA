from __future__ import annotations

from collections.abc import Mapping

import torch

from himem_bridge_vla.model.himem_bridge_vla import HiMemBridgeVLA
from himem_bridge_vla.model.planner.progress_state import ProgressState


class RuntimePolicyState:
    def __init__(self) -> None:
        self.executed_actions: torch.Tensor | None = None
        self.executed_action_mask: torch.Tensor | None = None
        self.progress_state: ProgressState | None = None

    def reset(self, model: HiMemBridgeVLA) -> None:
        _ = model
        self.executed_actions = None
        self.executed_action_mask = None
        self.progress_state = None

    def progress_inputs(
        self,
        model: HiMemBridgeVLA,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        planner = model.progress_state_planner
        if planner is None:
            return None, None

        stride = int(planner.config.replan_stride)
        action_dim = int(planner.config.action_dim)
        if self.executed_actions is None:
            actions = torch.zeros(1, stride, action_dim, device=device, dtype=dtype)
            mask = torch.zeros(1, stride, device=device, dtype=torch.bool)
            return actions, mask

        return (
            self.executed_actions.to(device=device, dtype=dtype),
            self.executed_action_mask.to(device=device) if self.executed_action_mask is not None else None,
        )

    def progress_state_input(
        self,
        model: HiMemBridgeVLA,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ProgressState | None:
        planner = model.progress_state_planner
        if planner is None:
            return None
        if self.progress_state is None:
            return planner.initial_state(batch_size, device=device, dtype=dtype)
        return ProgressState(
            completed_events=self.progress_state.completed_events.to(device=device, dtype=dtype),
            current_stage=self.progress_state.current_stage.to(device=device, dtype=dtype),
        )

    def store_progress_state(self, model: HiMemBridgeVLA) -> None:
        output = getattr(model, "last_progress_planner_output", None)
        if output is None:
            return
        self.progress_state = ProgressState(
            completed_events=output.progress_state.completed_events.detach().cpu(),
            current_stage=output.progress_state.current_stage.detach().cpu(),
        )

    def store_executed_actions(self, model: HiMemBridgeVLA, normalized_action: torch.Tensor) -> None:
        planner = model.progress_state_planner
        if planner is None:
            return

        stride = int(planner.config.replan_stride)
        action_dim = int(planner.config.action_dim)
        action = normalized_action[:, :stride, :action_dim].detach()
        if action.shape[1] != stride:
            pad = torch.zeros(
                action.shape[0],
                stride - action.shape[1],
                action_dim,
                device=action.device,
                dtype=action.dtype,
            )
            action = torch.cat([action, pad], dim=1)
        self.executed_actions = action.cpu()
        self.executed_action_mask = torch.ones(action.shape[:2], dtype=torch.bool)

    def store_executed_action_inputs(self, actions: torch.Tensor, mask: torch.Tensor | None) -> None:
        if actions.ndim != 3:
            raise ValueError(f"actions must have shape [B, R, A], got {tuple(actions.shape)}")
        self.executed_actions = actions.detach().cpu()
        if mask is None:
            self.executed_action_mask = torch.ones(actions.shape[:2], dtype=torch.bool)
        else:
            if mask.ndim != 2 or tuple(mask.shape) != tuple(actions.shape[:2]):
                raise ValueError(f"mask shape {tuple(mask.shape)} does not match actions {tuple(actions.shape[:2])}")
            self.executed_action_mask = mask.detach().cpu().bool()

def pack_runtime_visual_tokens(tokens: torch.Tensor, *, target_tokens: int) -> torch.Tensor:
    if tokens.ndim == 2:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 3 or tokens.shape[0] != 1:
        raise ValueError(f"tokens must have shape [1, N, D] or [N, D], got {tuple(tokens.shape)}")
    target_tokens = int(target_tokens)
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    token_count = int(tokens.shape[1])
    if token_count <= 0:
        raise ValueError("visual token sequence must be non-empty")
    if token_count == target_tokens:
        return tokens.contiguous()
    if token_count < target_tokens:
        output = torch.zeros(1, target_tokens, tokens.shape[-1], device=tokens.device, dtype=tokens.dtype)
        output[:, :token_count, :] = tokens
        return output

    boundaries = torch.linspace(0, token_count, steps=target_tokens + 1, device=tokens.device).round().long()
    output = torch.zeros(1, target_tokens, tokens.shape[-1], device=tokens.device, dtype=tokens.dtype)
    for index in range(target_tokens):
        start = int(boundaries[index].item())
        end = int(boundaries[index + 1].item())
        if end <= start:
            end = min(token_count, start + 1)
        output[:, index, :] = tokens[:, start:end, :].mean(dim=1)
    return output


def build_short_memory_inputs_from_visual_tokens(
    model: HiMemBridgeVLA,
    visual_tokens_by_offset: Mapping[int, torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    memory, mask, time_ids = empty_short_memory_inputs(model, device=device, dtype=dtype)
    if memory is None or mask is None or time_ids is None:
        return None, None, None

    offsets = short_memory_offsets(model)
    entry_tokens = int(model.config.get("memory_entry_tokens", 16))
    for entry_index, offset in enumerate(offsets):
        tokens = visual_tokens_by_offset.get(int(offset))
        if tokens is None:
            continue
        packed = pack_runtime_visual_tokens(tokens.to(device=device, dtype=dtype), target_tokens=entry_tokens)
        start = entry_index * entry_tokens
        end = start + entry_tokens
        memory[:, start:end, :] = packed
        mask[:, start:end] = True
    return memory, mask, time_ids


def empty_short_memory_inputs(
    model: HiMemBridgeVLA,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if not bool(getattr(model, "use_direct_bridge", False)):
        return None, None, None
    capacity = int(model.config.get("memory_short_capacity", 2))
    entry_tokens = int(model.config.get("memory_entry_tokens", 16))
    hidden_dim = int(model.config.get("embed_dim", 896))
    if capacity <= 0 or entry_tokens <= 0:
        return None, None, None

    memory = torch.zeros(1, capacity * entry_tokens, hidden_dim, device=device, dtype=dtype)
    mask = torch.zeros(1, capacity * entry_tokens, device=device, dtype=torch.bool)
    time_ids = torch.arange(capacity, device=device, dtype=torch.long).repeat_interleave(entry_tokens)
    return memory, mask, time_ids.unsqueeze(0)


def short_memory_offsets(model: HiMemBridgeVLA) -> tuple[int, ...]:
    capacity = int(model.config.get("memory_short_capacity", 2))
    raw_offsets = model.config.get("memory_short_offsets")
    if raw_offsets is None:
        return tuple(range(capacity, 0, -1))
    offsets = tuple(int(offset) for offset in raw_offsets)
    if len(offsets) < capacity:
        offsets = offsets + tuple(range(capacity - len(offsets), 0, -1))
    return offsets[:capacity]
