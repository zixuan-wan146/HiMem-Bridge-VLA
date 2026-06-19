from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class CoarsePlannerConfig:
    hidden_dim: int
    state_dim: int
    latent_dim: int = 128
    num_plan_steps: int = 8
    planning_horizon: int = 64
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.05
    ffn_mult: int = 4
    latent_head_hidden_dim: int = 512


@dataclass(frozen=True)
class CoarsePlannerOutput:
    plan_tokens: torch.Tensor
    predicted_latents: torch.Tensor


class CoarsePlanner(nn.Module):
    """Query Transformer planner supervised by action-segment intent latents.

    The first version intentionally does not accept memory tokens. It predicts
    plan tokens only from current VLM tokens and robot state, so memory remains
    a parallel condition in BridgeAttention.
    """

    def __init__(self, config: CoarsePlannerConfig) -> None:
        super().__init__()
        if config.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {config.hidden_dim}")
        if config.state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {config.state_dim}")
        if config.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {config.latent_dim}")
        if config.num_plan_steps <= 0:
            raise ValueError(f"num_plan_steps must be positive, got {config.num_plan_steps}")
        if config.planning_horizon <= 0:
            raise ValueError(f"planning_horizon must be positive, got {config.planning_horizon}")
        if config.planning_horizon % config.num_plan_steps != 0:
            raise ValueError("planning_horizon must be divisible by num_plan_steps")
        if config.num_layers < 3:
            raise ValueError("CoarsePlanner requires at least 3 Transformer layers")
        if config.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {config.num_heads}")
        if config.hidden_dim % config.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if config.ffn_mult <= 0:
            raise ValueError(f"ffn_mult must be positive, got {config.ffn_mult}")
        if config.latent_head_hidden_dim <= 0:
            raise ValueError(f"latent_head_hidden_dim must be positive, got {config.latent_head_hidden_dim}")
        if float(config.dropout) < 0.0:
            raise ValueError(f"dropout must be non-negative, got {config.dropout}")

        self.config = config
        self.plan_queries = nn.Parameter(torch.empty(config.num_plan_steps, config.hidden_dim))
        nn.init.normal_(self.plan_queries, mean=0.0, std=0.02)

        self.state_proj = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=config.hidden_dim,
                    nhead=config.num_heads,
                    dim_feedforward=config.hidden_dim * config.ffn_mult,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.latent_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.latent_head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.latent_head_hidden_dim, config.latent_dim),
        )

    def forward(
        self,
        vlm_tokens: torch.Tensor,
        state: torch.Tensor,
    ) -> CoarsePlannerOutput:
        vlm_tokens = _ensure_rank3(vlm_tokens, "vlm_tokens")
        if vlm_tokens.shape[-1] != self.config.hidden_dim:
            raise ValueError(
                f"vlm_tokens last dimension {vlm_tokens.shape[-1]} != hidden_dim {self.config.hidden_dim}"
            )
        state = _ensure_state(state, self.config.state_dim)

        batch_size = vlm_tokens.shape[0]
        device = vlm_tokens.device
        dtype = vlm_tokens.dtype
        state = state.to(device=device, dtype=dtype)

        self._match_runtime_dtype(device=device, dtype=dtype)
        state_token = self.state_proj(state).unsqueeze(1)
        context = torch.cat([vlm_tokens, state_token], dim=1)

        plan_tokens = self.plan_queries.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        for layer in self.layers:
            plan_tokens = layer(tgt=plan_tokens, memory=context)

        plan_tokens = self.output_norm(plan_tokens)
        predicted_latents = self.latent_head(plan_tokens)
        return CoarsePlannerOutput(plan_tokens=plan_tokens, predicted_latents=predicted_latents)

    def _match_runtime_dtype(self, *, device: torch.device, dtype: torch.dtype) -> None:
        self.state_proj.to(device=device, dtype=dtype)
        self.layers.to(device=device, dtype=dtype)
        self.output_norm.to(device=device, dtype=dtype)
        self.latent_head.to(device=device, dtype=dtype)


def _ensure_rank3(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(1)
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [B, T, D] or [B, D], got {tuple(tensor.shape)}")
    return tensor


def _ensure_state(state: torch.Tensor, state_dim: int) -> torch.Tensor:
    if state.ndim == 3 and state.shape[1] == 1:
        state = state.squeeze(1)
    if state.ndim != 2:
        raise ValueError(f"state must have shape [B, state_dim] or [B, 1, state_dim], got {tuple(state.shape)}")
    if state.shape[-1] != state_dim:
        raise ValueError(f"state last dimension {state.shape[-1]} != state_dim {state_dim}")
    return state
