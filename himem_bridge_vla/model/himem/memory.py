from __future__ import annotations

import torch
import torch.nn.functional as F


class EpisodeMemoryBank:
    """In-memory episode token store for HiMem-lite inference experiments."""

    def __init__(self, *, max_tokens: int = 32, token_dim: int | None = None) -> None:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        self.max_tokens = int(max_tokens)
        self.token_dim = token_dim
        self._bank: dict[str, torch.Tensor] = {}

    def reset(self, episode_id: str | None = None) -> None:
        if episode_id is None:
            self._bank.clear()
            return
        self._bank.pop(str(episode_id), None)

    def read(self, episode_id: str, query: torch.Tensor, *, top_k: int | None = None) -> torch.Tensor:
        query = _ensure_query(query)
        memory = self._bank.get(str(episode_id))
        if memory is None or memory.numel() == 0:
            return query.new_zeros(query.shape[0], 0, query.shape[-1])

        memory = memory.to(device=query.device, dtype=query.dtype)
        if memory.shape[-1] != query.shape[-1]:
            raise ValueError(f"memory token dim {memory.shape[-1]} != query dim {query.shape[-1]}")

        top_k = memory.shape[0] if top_k is None else min(int(top_k), memory.shape[0])
        if top_k <= 0:
            return query.new_zeros(query.shape[0], 0, query.shape[-1])

        pooled_query = query.mean(dim=1)
        scores = F.normalize(pooled_query, dim=-1) @ F.normalize(memory, dim=-1).T
        indices = scores.topk(k=top_k, dim=-1).indices
        return memory[indices]

    def write(
        self,
        episode_id: str,
        token: torch.Tensor,
        *,
        gate: torch.Tensor | float | None = None,
        threshold: float = 0.5,
    ) -> int:
        tokens = _flatten_tokens(token)
        if tokens.numel() == 0:
            return 0
        if self.token_dim is None:
            self.token_dim = int(tokens.shape[-1])
        if tokens.shape[-1] != self.token_dim:
            raise ValueError(f"token dim {tokens.shape[-1]} != memory token_dim {self.token_dim}")

        keep_mask = _gate_mask(gate, tokens.shape[0], threshold, device=tokens.device)
        tokens = tokens[keep_mask]
        if tokens.numel() == 0:
            return 0

        episode_id = str(episode_id)
        tokens = tokens.detach().to(device="cpu", dtype=torch.float32)
        existing = self._bank.get(episode_id)
        updated = tokens if existing is None else torch.cat([existing, tokens], dim=0)
        self._bank[episode_id] = updated[-self.max_tokens :]
        return int(tokens.shape[0])

    def __len__(self) -> int:
        return sum(tokens.shape[0] for tokens in self._bank.values())

    def episode_length(self, episode_id: str) -> int:
        memory = self._bank.get(str(episode_id))
        return 0 if memory is None else int(memory.shape[0])


def _ensure_query(query: torch.Tensor) -> torch.Tensor:
    if query.ndim == 2:
        return query.unsqueeze(1)
    if query.ndim != 3:
        raise ValueError(f"query must have shape [B, T, D] or [B, D], got {tuple(query.shape)}")
    return query


def _flatten_tokens(token: torch.Tensor) -> torch.Tensor:
    if token.ndim == 1:
        return token.unsqueeze(0)
    if token.ndim == 2:
        return token
    if token.ndim == 3:
        return token.reshape(-1, token.shape[-1])
    raise ValueError(f"token must have shape [D], [N, D], or [B, T, D], got {tuple(token.shape)}")


def _gate_mask(
    gate: torch.Tensor | float | None,
    token_count: int,
    threshold: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    if gate is None:
        return torch.ones(token_count, dtype=torch.bool, device=device)
    if isinstance(gate, float):
        return torch.full((token_count,), gate >= threshold, dtype=torch.bool, device=device)
    gate = gate.detach().to(device=device).reshape(-1)
    if gate.numel() == 1:
        return torch.full((token_count,), bool(gate.item() >= threshold), dtype=torch.bool, device=device)
    if gate.numel() != token_count:
        raise ValueError(f"gate has {gate.numel()} values but token_count is {token_count}")
    return gate >= threshold
