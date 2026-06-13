from __future__ import annotations

import torch
import torch.nn as nn

from .memory import EpisodeMemoryBank


class HiMemTokenWriter(nn.Module):
    """Distill frame-level bridge tokens into a fixed number of memory tokens."""

    def __init__(self, *, hidden_dim: int, num_tokens: int = 4, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}")

        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.write_queries = nn.Parameter(torch.empty(num_tokens, hidden_dim))
        nn.init.normal_(self.write_queries, mean=0.0, std=0.02)
        self.source_norm = nn.LayerNorm(hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, bridge_tokens: torch.Tensor) -> torch.Tensor:
        bridge_tokens = _ensure_rank3(bridge_tokens, "bridge_tokens")
        if bridge_tokens.shape[-1] != self.hidden_dim:
            raise ValueError(f"bridge_tokens last dimension {bridge_tokens.shape[-1]} != hidden_dim {self.hidden_dim}")

        batch_size = bridge_tokens.shape[0]
        queries = self.write_queries.to(device=bridge_tokens.device, dtype=bridge_tokens.dtype)
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)
        source = self.source_norm(bridge_tokens)
        query = self.query_norm(queries)
        attended, _ = self.attn(query, source, source, need_weights=False)
        tokens = queries + attended
        return tokens + self.ffn(tokens)


class HierarchicalEpisodeMemory:
    """Frame -> segment -> episode memory controller for online inference."""

    def __init__(
        self,
        *,
        bank: EpisodeMemoryBank,
        read_top_k: int = 8,
        write_threshold: float = 0.5,
        segment_accumulator: str = "ema",
        segment_ema_decay: float = 0.9,
        write_policy: str = "boundary",
    ) -> None:
        if read_top_k <= 0:
            raise ValueError(f"read_top_k must be positive, got {read_top_k}")
        if segment_accumulator not in {"none", "ema"}:
            raise ValueError("segment_accumulator must be 'none' or 'ema'")
        if not 0.0 <= float(segment_ema_decay) < 1.0:
            raise ValueError("segment_ema_decay must be in [0, 1)")
        if write_policy not in {"boundary", "always"}:
            raise ValueError("write_policy must be 'boundary' or 'always'")

        self.bank = bank
        self.read_top_k = int(read_top_k)
        self.write_threshold = float(write_threshold)
        self.segment_accumulator = segment_accumulator
        self.segment_ema_decay = float(segment_ema_decay)
        self.write_policy = write_policy
        self._segments: dict[str, torch.Tensor] = {}

    def reset(self, episode_id: str | None = None) -> None:
        self.bank.reset(episode_id)
        if episode_id is None:
            self._segments.clear()
        else:
            self._segments.pop(str(episode_id), None)

    def read(self, episode_id: str, query: torch.Tensor, *, reset: bool = False) -> torch.Tensor:
        if reset:
            self.reset(episode_id)
        return self.bank.read(episode_id, query, top_k=self.read_top_k)

    def write(
        self,
        episode_id: str,
        tokens: torch.Tensor,
        *,
        gate: torch.Tensor | float | None = None,
    ) -> int:
        segment_tokens = self.update_segment(episode_id, tokens)
        if segment_tokens.numel() == 0:
            return 0
        if not self._should_write(gate):
            return 0

        episode_key = str(episode_id)
        written = self.bank.write(episode_key, segment_tokens)
        self._segments.pop(episode_key, None)
        return written

    def update_segment(self, episode_id: str, tokens: torch.Tensor) -> torch.Tensor:
        tokens = _flatten_tokens(tokens)
        if tokens.numel() == 0:
            return tokens
        episode_key = str(episode_id)
        tokens = tokens.detach().to(device="cpu", dtype=torch.float32)
        if self.segment_accumulator == "none":
            self._segments[episode_key] = tokens
            return tokens

        previous = self._segments.get(episode_key)
        if previous is None or previous.shape != tokens.shape:
            updated = tokens
        else:
            updated = previous * self.segment_ema_decay + tokens * (1.0 - self.segment_ema_decay)
        self._segments[episode_key] = updated
        return updated

    def segment_length(self, episode_id: str) -> int:
        segment = self._segments.get(str(episode_id))
        return 0 if segment is None else int(segment.shape[0])

    def _should_write(self, gate: torch.Tensor | float | None) -> bool:
        if self.write_policy == "always":
            return True
        if gate is None:
            return True
        if isinstance(gate, float):
            return gate >= self.write_threshold
        gate_value = gate.detach().to(dtype=torch.float32).reshape(-1).mean().item()
        return gate_value >= self.write_threshold


def _ensure_rank3(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(1)
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [B, T, D] or [B, D], got {tuple(tensor.shape)}")
    return tensor


def _flatten_tokens(token: torch.Tensor) -> torch.Tensor:
    if token.ndim == 1:
        return token.unsqueeze(0)
    if token.ndim == 2:
        return token
    if token.ndim == 3:
        return token.reshape(-1, token.shape[-1])
    raise ValueError(f"token must have shape [D], [N, D], or [B, T, D], got {tuple(token.shape)}")
