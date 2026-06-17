from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagonalSSMBlock(nn.Module):
    """A causal diagonal state-space block without convolution or attention."""

    def __init__(self, d_model: int, state_dim: int, dropout: float) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        self.norm = nn.LayerNorm(d_model)
        self.input_to_state = nn.Linear(d_model, state_dim)
        self.state_to_output = nn.Linear(state_dim, d_model)
        self.skip = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.log_decay = nn.Parameter(torch.zeros(state_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B, T, C], got {tuple(x.shape)}")
        y = self.norm(x)
        state_inputs = self.input_to_state(y)
        decay = torch.exp(-F.softplus(self.log_decay)).to(dtype=y.dtype)
        state = torch.zeros(y.shape[0], state_inputs.shape[-1], device=y.device, dtype=y.dtype)
        outputs = []
        for step in range(y.shape[1]):
            state = state * decay + state_inputs[:, step]
            outputs.append(self.state_to_output(state))
        ssm_out = torch.stack(outputs, dim=1)
        gated = (ssm_out + self.skip(y)) * torch.sigmoid(self.gate(y))
        return x + self.dropout(self.output(gated))


class CausalTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        hidden_dim = int(round(d_model * mlp_ratio))
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        y = self.attn_norm(x)
        attn_out, _ = self.attn(y, y, y, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out
        return x + self.ffn(self.ffn_norm(x))


class TransitionTriggerHead(nn.Module):
    """Config-driven causal transition scoring head.

    Supported ``model.type`` values:
    - ``ssm``: pure recurrent diagonal state-space backbone.
    - ``transformer``: causal Transformer encoder over the history window.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        type: str | None = None,
        d_model: int | None = None,
        num_layers: int | None = None,
        state_dim: int | None = None,
        num_heads: int | None = None,
        mlp_ratio: float | None = None,
        max_seq_len: int | None = None,
        pooling: str | None = None,
        dropout: float | None = None,
        head_hidden_dim: int | None = None,
        **unknown: Any,
    ) -> None:
        super().__init__()
        if unknown:
            raise ValueError(f"unknown model config keys: {sorted(unknown)}")
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        self.input_dim = int(input_dim)
        self.model_type = str(_required(type, "model.type"))
        self.pooling = str(_required(pooling, "model.pooling"))
        if self.pooling != "last":
            raise ValueError("only model.pooling='last' is currently supported")
        width = int(_required(d_model, "model.d_model"))
        layers = int(_required(num_layers, "model.num_layers"))
        drop = float(_required(dropout, "model.dropout"))

        if self.model_type == "ssm":
            self.backbone = SSMBackbone(
                input_dim=input_dim,
                d_model=width,
                num_layers=layers,
                state_dim=int(_required(state_dim, "model.state_dim")),
                dropout=drop,
            )
            output_dim = width
        elif self.model_type == "transformer":
            self.backbone = CausalTransformerBackbone(
                input_dim=input_dim,
                d_model=width,
                num_layers=layers,
                num_heads=int(_required(num_heads, "model.num_heads")),
                mlp_ratio=float(_required(mlp_ratio, "model.mlp_ratio")),
                max_seq_len=int(_required(max_seq_len, "model.max_seq_len")),
                dropout=drop,
            )
            output_dim = width
        else:
            raise ValueError("model.type must be one of 'ssm' or 'transformer'")

        head_hidden = int(_required(head_hidden_dim, "model.head_hidden_dim"))
        self.head = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"features must have shape [B, W, D], got {tuple(features.shape)}")
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"feature dim {features.shape[-1]} != configured input_dim {self.input_dim}")
        sequence = self.backbone(features)
        return self.head(sequence[:, -1])


def _required(value: Any, name: str) -> Any:
    if value is None:
        raise ValueError(f"{name} must be set in config")
    return value


class SSMBackbone(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        d_model: int,
        num_layers: int,
        state_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList(DiagonalSSMBlock(d_model, state_dim, dropout) for _ in range(num_layers))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(features)
        for block in self.blocks:
            x = block(x)
        return x


class CausalTransformerBackbone(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        max_seq_len: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.max_seq_len = int(max_seq_len)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.max_seq_len, d_model))
        self.blocks = nn.ModuleList(
            CausalTransformerBlock(d_model, num_heads, mlp_ratio, dropout) for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[1] > self.max_seq_len:
            raise ValueError(f"window length {features.shape[1]} exceeds model.max_seq_len={self.max_seq_len}")
        x = self.input_proj(features)
        x = self.dropout(x + self.position[:, : x.shape[1]])
        causal_mask = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).triu(1)
        for block in self.blocks:
            x = block(x, causal_mask)
        return x
