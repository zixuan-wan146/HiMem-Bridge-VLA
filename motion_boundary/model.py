from __future__ import annotations

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if dilation <= 0:
            raise ValueError("dilation must be positive")
        self.left_padding = (kernel_size - 1) * dilation
        self.norm = nn.LayerNorm(channels)
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.left_padding,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B, T, C], got {tuple(x.shape)}")
        residual = x
        y = self.norm(x)
        y = y.transpose(1, 2)
        y = self.conv(y)
        if self.left_padding:
            y = y[..., : -self.left_padding]
        y = y.transpose(1, 2)
        y = self.dropout(self.activation(y))
        return residual + y


class MotionStateBoundaryHead(nn.Module):
    """Causal multi-scale TCN for motion-state boundary scoring."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 128,
        kernel_size: int = 5,
        dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1,
        mlp_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            CausalConv1d(hidden_dim, kernel_size=kernel_size, dilation=int(dilation), dropout=dropout)
            for dilation in dilations
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return boundary logits for the last timestep of each window."""

        if features.ndim != 3:
            raise ValueError(f"features must have shape [B, W, D], got {tuple(features.shape)}")
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"feature dim {features.shape[-1]} != configured input_dim {self.input_dim}")
        x = self.input_proj(features)
        for block in self.blocks:
            x = block(x)
        return self.head(x[:, -1])
