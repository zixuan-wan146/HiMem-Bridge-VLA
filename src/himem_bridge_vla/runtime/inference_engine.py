from __future__ import annotations

from contextlib import nullcontext

import torch

from himem_bridge_vla.model.himem_bridge_vla import HiMemBridgeVLA
from himem_bridge_vla.runtime.contract import PolicyRequest
from himem_bridge_vla.runtime.feature_extractor import decode_images_by_view
from himem_bridge_vla.runtime.memory_builder import build_short_memory_inputs_from_visual_tokens
from himem_bridge_vla.runtime.memory_builder import RuntimePolicyState
from himem_bridge_vla.utils.normalization import NormalizationStats


class PolicyInferenceEngine:
    def __init__(
        self,
        model: HiMemBridgeVLA,
        normalizer: NormalizationStats,
        *,
        state_dim: int,
    ) -> None:
        self.model = model
        self.normalizer = normalizer
        self.state_dim = int(state_dim)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def infer(self, request: PolicyRequest, runtime_state: RuntimePolicyState | None = None):
        device = self.device
        model_action_dim = int(
            getattr(self.model, "per_action_dim", self.model.config.get("per_action_dim", request.action_dim))
        )
        if int(request.action_dim) != model_action_dim:
            raise ValueError(f"request action_dim={request.action_dim} does not match model action_dim={model_action_dim}")

        if runtime_state is not None and request.reset_memory:
            runtime_state.reset(self.model)

        images = decode_images_by_view(request.images_by_view, device)
        state = torch.as_tensor(request.state, dtype=torch.float32, device=device)
        norm_state = self.normalizer.normalize_state(
            pad_state_tensor(state, target_dim=self.state_dim),
            robot_key=request.robot_key,
        ).to(dtype=torch.float32)
        image_mask = torch.ones(len(images), dtype=torch.int32, device=device)
        action_mask = torch.ones(1, model_action_dim, dtype=torch.int32, device=device)

        autocast_context = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad(), autocast_context:
            embedding_output = self.model.get_vl_embeddings(
                images=images,
                image_mask=image_mask,
                prompt=request.prompt,
                return_hidden_states=self.model.use_bridge,
            )
            if hasattr(embedding_output, "fused_tokens"):
                hidden_states = embedding_output.hidden_states
                visual_tokens = getattr(embedding_output, "visual_tokens", None)
                fused_tokens = visual_tokens if visual_tokens is not None else embedding_output.fused_tokens
                planner_vl_summary = getattr(embedding_output, "planner_vl_summary", None)
            else:
                fused_tokens = embedding_output
                hidden_states = None
                visual_tokens = None
                planner_vl_summary = None

            executed_actions, executed_action_mask = (None, None)
            memory_context, memory_context_mask, short_memory_time_ids = (None, None, None)
            request_progress_inputs = self._executed_actions_from_request(
                request,
                device=device,
                dtype=fused_tokens.dtype,
                robot_key=request.robot_key,
            )
            if request_progress_inputs is not None:
                executed_actions, executed_action_mask = request_progress_inputs
            if runtime_state is not None:
                if request_progress_inputs is not None:
                    runtime_state.store_executed_action_inputs(executed_actions, executed_action_mask)
                else:
                    executed_actions, executed_action_mask = runtime_state.progress_inputs(
                        self.model,
                        device=device,
                        dtype=fused_tokens.dtype,
                    )
            if request.short_memory_images_by_offset is not None:
                memory_context, memory_context_mask, short_memory_time_ids = self._short_memory_from_request(
                    request,
                    device=device,
                    dtype=fused_tokens.dtype,
                )

            action = self.model.predict_action(
                fused_tokens,
                norm_state,
                action_mask=action_mask,
                hidden_states=hidden_states,
                memory_context=memory_context,
                memory_context_mask=memory_context_mask,
                short_memory_time_ids=short_memory_time_ids,
                executed_actions=executed_actions,
                executed_action_mask=executed_action_mask,
                planner_vl_summary=planner_vl_summary,
            )
            if action.numel() % model_action_dim != 0:
                raise ValueError(
                    f"Model returned {action.numel()} action values, not divisible by per_action_dim={model_action_dim}"
                )
            normalized_action = action.reshape(1, -1, model_action_dim)
            if runtime_state is not None and request_progress_inputs is None:
                runtime_state.store_executed_actions(self.model, normalized_action)
            denormalized_action = self.normalizer.denormalize_action(normalized_action[0], robot_key=request.robot_key)
            actions = denormalized_action.cpu().numpy().tolist()
            if not request.return_debug:
                return actions
            return {"actions": actions}

    def _executed_actions_from_request(
        self,
        request: PolicyRequest,
        *,
        device: torch.device,
        dtype: torch.dtype,
        robot_key: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if request.executed_actions is None:
            return None
        planner = self.model.progress_state_planner
        if planner is None:
            return None
        stride = int(planner.config.replan_stride)
        action_dim = int(planner.config.action_dim)
        actions = torch.as_tensor(request.executed_actions, dtype=torch.float32, device=device)
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        if actions.ndim != 3:
            raise ValueError(f"executed_actions must have shape [R, A] or [B, R, A], got {tuple(actions.shape)}")
        if int(actions.shape[0]) != 1:
            raise ValueError(f"runtime executed_actions supports batch size 1, got {tuple(actions.shape)}")
        if int(actions.shape[-1]) != action_dim:
            raise ValueError(f"executed_actions action dim {actions.shape[-1]} != planner action_dim {action_dim}")
        actions = self.normalizer.normalize_action(actions, robot_key=robot_key).to(device=device, dtype=dtype)

        if request.executed_action_mask is None:
            mask = torch.ones(actions.shape[:2], dtype=torch.bool, device=device)
        else:
            mask = torch.as_tensor(request.executed_action_mask, dtype=torch.bool, device=device)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if mask.ndim != 2 or tuple(mask.shape) != tuple(actions.shape[:2]):
                raise ValueError(
                    "executed_action_mask must have shape [R] or [B, R] matching executed_actions, "
                    f"got {tuple(mask.shape)} for actions {tuple(actions.shape)}"
                )

        if int(actions.shape[1]) > stride:
            actions = actions[:, :stride, :]
            mask = mask[:, :stride]
        elif int(actions.shape[1]) < stride:
            pad_steps = stride - int(actions.shape[1])
            action_pad = torch.zeros(actions.shape[0], pad_steps, action_dim, device=device, dtype=dtype)
            mask_pad = torch.zeros(actions.shape[0], pad_steps, device=device, dtype=torch.bool)
            actions = torch.cat([actions, action_pad], dim=1)
            mask = torch.cat([mask, mask_pad], dim=1)
        return actions, mask

    def _short_memory_from_request(
        self,
        request: PolicyRequest,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        visual_tokens_by_offset = {}
        for offset, images_by_view in (request.short_memory_images_by_offset or {}).items():
            images = decode_images_by_view(images_by_view, device)
            image_mask = torch.ones(len(images), dtype=torch.int32, device=device)
            embedding_output = self.model.get_vl_embeddings(
                images=images,
                image_mask=image_mask,
                prompt=request.prompt,
                return_hidden_states=True,
            )
            visual_tokens = getattr(embedding_output, "visual_tokens", None)
            if visual_tokens is None:
                visual_tokens = embedding_output
            visual_tokens_by_offset[int(offset)] = visual_tokens
        return build_short_memory_inputs_from_visual_tokens(
            self.model,
            visual_tokens_by_offset,
            device=device,
            dtype=dtype,
        )


def pad_state_tensor(state: torch.Tensor, target_dim: int) -> torch.Tensor:
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.shape[1] > target_dim:
        raise ValueError(f"State dimension {state.shape[1]} exceeds expected {target_dim}")
    if state.shape[1] < target_dim:
        padding = torch.zeros((state.shape[0], target_dim - state.shape[1]), device=state.device)
        state = torch.cat([state, padding], dim=1)
    return state
