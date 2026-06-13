from __future__ import annotations

import torch
import torch.nn as nn


class BoundaryHead(nn.Module):
    """Small skill-boundary scoring head for bridge or memory tokens."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 3:
            tokens = tokens.mean(dim=1)
        if tokens.ndim != 2:
            raise ValueError(f"tokens must have shape [B, T, D] or [B, D], got {tuple(tokens.shape)}")
        return self.net(tokens)
