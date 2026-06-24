from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn


SHORT_MEMORY = "S"
LONG_MEMORY = "L"
MEMORY_TYPE_TO_ID = {SHORT_MEMORY: 0, LONG_MEMORY: 1}


@dataclass(frozen=True)
class VisualMemoryEntry:
    visual_tokens_by_view: Mapping[str, torch.Tensor]
    tau: int
    eta: str

    def __post_init__(self) -> None:
        tau = int(self.tau)
        if tau < 0:
            raise ValueError(f"tau must be non-negative, got {self.tau}")
        if self.eta not in MEMORY_TYPE_TO_ID:
            raise ValueError(f"eta must be one of {sorted(MEMORY_TYPE_TO_ID)}, got {self.eta!r}")
        if not self.visual_tokens_by_view:
            raise ValueError("visual_tokens_by_view must contain at least one view")

        normalized: dict[str, torch.Tensor] = {}
        for view_name, tokens in self.visual_tokens_by_view.items():
            if not isinstance(tokens, torch.Tensor):
                raise TypeError(f"tokens for view {view_name!r} must be a torch.Tensor")
            if tokens.ndim != 2:
                raise ValueError(
                    f"tokens for view {view_name!r} must have shape [N, D], got {tuple(tokens.shape)}"
                )
            normalized[str(view_name)] = tokens

        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "visual_tokens_by_view", normalized)

    @property
    def type_id(self) -> int:
        return MEMORY_TYPE_TO_ID[self.eta]


@dataclass(frozen=True)
class MemoryReadResult:
    entries: tuple[VisualMemoryEntry | None, ...]
    entry_mask: torch.Tensor
    short_capacity: int
    long_capacity: int

    def __post_init__(self) -> None:
        mask = self.entry_mask.to(dtype=torch.bool).reshape(-1)
        if mask.numel() != len(self.entries):
            raise ValueError(f"entry_mask has {mask.numel()} values for {len(self.entries)} entries")
        if len(self.entries) != self.short_capacity + self.long_capacity:
            raise ValueError("entries length must equal short_capacity + long_capacity")
        for entry, is_valid in zip(self.entries, mask.tolist(), strict=True):
            if is_valid and entry is None:
                raise ValueError("entry_mask marks a padding entry as valid")
            if entry is not None and not is_valid:
                raise ValueError("entry_mask marks a real entry as invalid")
        object.__setattr__(self, "entry_mask", mask)

    def token_mask(self, tokens_per_entry: int) -> torch.Tensor:
        return expand_entry_mask(self.entry_mask, tokens_per_entry)


@dataclass(frozen=True)
class CompressedVisualMemory:
    tokens: torch.Tensor
    mask: torch.Tensor


class DualFifoVisualMemory:
    """Runtime visual memory bank with deterministic short reads and external long writes."""

    def __init__(self, *, short_offsets: Sequence[int] = (32, 16), long_capacity: int = 4) -> None:
        if not short_offsets:
            raise ValueError("short_offsets must contain at least one offset")
        offsets = tuple(sorted((int(offset) for offset in short_offsets), reverse=True))
        if any(offset <= 0 for offset in offsets):
            raise ValueError(f"short_offsets must be positive, got {short_offsets}")
        if int(long_capacity) < 0:
            raise ValueError(f"long_capacity must be non-negative, got {long_capacity}")

        self.short_offsets = offsets
        self.short_capacity = len(offsets)
        self.long_capacity = int(long_capacity)
        self._short_history: dict[int, VisualMemoryEntry] = {}
        self._long_entries: list[VisualMemoryEntry] = []

    def reset(self) -> None:
        self._short_history.clear()
        self._long_entries.clear()

    def write_observation(self, tau: int, visual_tokens_by_view: Mapping[str, torch.Tensor]) -> VisualMemoryEntry:
        entry = VisualMemoryEntry(visual_tokens_by_view=visual_tokens_by_view, tau=tau, eta=SHORT_MEMORY)
        self._short_history[entry.tau] = entry
        self._prune_short_history(current_step=entry.tau)
        return entry

    def write_long(self, tau: int, visual_tokens_by_view: Mapping[str, torch.Tensor]) -> VisualMemoryEntry:
        entry = VisualMemoryEntry(visual_tokens_by_view=visual_tokens_by_view, tau=tau, eta=LONG_MEMORY)
        return self.append_long(entry)

    def append_long(self, entry: VisualMemoryEntry) -> VisualMemoryEntry:
        if entry.eta != LONG_MEMORY:
            entry = VisualMemoryEntry(
                visual_tokens_by_view=entry.visual_tokens_by_view,
                tau=entry.tau,
                eta=LONG_MEMORY,
            )
        if self.long_capacity == 0:
            return entry
        self._long_entries.append(entry)
        self._long_entries = self._long_entries[-self.long_capacity :]
        return entry

    def read(self, current_step: int) -> MemoryReadResult:
        current_step = int(current_step)
        short_entries = [
            self._short_history[target_step]
            for offset in self.short_offsets
            if (target_step := current_step - offset) in self._short_history
        ]
        short_slots = _pad_entries(short_entries, self.short_capacity)
        long_slots = _pad_entries(self._long_entries, self.long_capacity)
        entries = tuple(short_slots + long_slots)
        entry_mask = torch.tensor([entry is not None for entry in entries], dtype=torch.bool)
        return MemoryReadResult(
            entries=entries,
            entry_mask=entry_mask,
            short_capacity=self.short_capacity,
            long_capacity=self.long_capacity,
        )

    def short_history_steps(self) -> tuple[int, ...]:
        return tuple(sorted(self._short_history))

    def long_entries(self) -> tuple[VisualMemoryEntry, ...]:
        return tuple(self._long_entries)

    def _prune_short_history(self, *, current_step: int) -> None:
        oldest_needed = int(current_step) - max(self.short_offsets)
        stale_steps = [step for step in self._short_history if step < oldest_needed]
        for step in stale_steps:
            del self._short_history[step]


class VisualMemoryCompressor(nn.Module):
    """Compress padded visual memory entries into fixed-budget memory tokens."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        view_names: Sequence[str],
        tokens_per_entry: int = 1,
        num_heads: int = 8,
        dropout: float = 0.0,
        max_age_steps: int = 512,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if tokens_per_entry <= 0:
            raise ValueError(f"tokens_per_entry must be positive, got {tokens_per_entry}")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}")
        if not view_names:
            raise ValueError("view_names must contain at least one view")
        if max_age_steps < 0:
            raise ValueError(f"max_age_steps must be non-negative, got {max_age_steps}")

        self.hidden_dim = int(hidden_dim)
        self.view_names = tuple(str(view_name) for view_name in view_names)
        self.view_to_id = {view_name: index for index, view_name in enumerate(self.view_names)}
        self.tokens_per_entry = int(tokens_per_entry)
        self.max_age_steps = int(max_age_steps)

        self.memory_queries = nn.Parameter(torch.empty(self.tokens_per_entry, self.hidden_dim))
        nn.init.normal_(self.memory_queries, mean=0.0, std=0.02)
        self.view_embedding = nn.Embedding(len(self.view_names), self.hidden_dim)
        self.age_embedding = nn.Embedding(self.max_age_steps + 1, self.hidden_dim)
        self.type_embedding = nn.Embedding(len(MEMORY_TYPE_TO_ID), self.hidden_dim)
        self.source_norm = nn.LayerNorm(self.hidden_dim)
        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            self.hidden_dim,
            int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )

    def forward(
        self,
        memory: MemoryReadResult | Sequence[VisualMemoryEntry | None],
        *,
        current_step: int,
    ) -> CompressedVisualMemory:
        if isinstance(memory, MemoryReadResult):
            entries = memory.entries
            entry_mask = memory.entry_mask
        else:
            entries = tuple(memory)
            entry_mask = torch.tensor([entry is not None for entry in entries], dtype=torch.bool)

        device, dtype = self._infer_device_dtype(entries)
        token_count = len(entries) * self.tokens_per_entry
        output = torch.zeros(token_count, self.hidden_dim, device=device, dtype=dtype)
        token_mask = expand_entry_mask(entry_mask.to(device=device), self.tokens_per_entry)

        for entry_index, entry in enumerate(entries):
            if entry is None:
                continue
            start = entry_index * self.tokens_per_entry
            end = start + self.tokens_per_entry
            output[start:end] = self._compress_entry(entry, current_step=int(current_step), device=device, dtype=dtype)

        output = output.masked_fill(~token_mask[:, None], 0.0)
        return CompressedVisualMemory(tokens=output, mask=token_mask)

    def _compress_entry(
        self,
        entry: VisualMemoryEntry,
        *,
        current_step: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        source = self._build_source(entry, device=device, dtype=dtype).unsqueeze(0)
        queries = self.memory_queries.to(device=device, dtype=dtype).unsqueeze(0)
        attended, _ = self.cross_attention(
            self.query_norm(queries),
            self.source_norm(source),
            self.source_norm(source),
            need_weights=False,
        )
        compressed = queries + attended
        age = max(0, min(current_step - entry.tau, self.max_age_steps))
        age_ids = torch.full((self.tokens_per_entry,), age, dtype=torch.long, device=device)
        type_ids = torch.full((self.tokens_per_entry,), entry.type_id, dtype=torch.long, device=device)
        compressed = compressed.squeeze(0)
        compressed = compressed + self.age_embedding(age_ids).to(dtype=dtype)
        compressed = compressed + self.type_embedding(type_ids).to(dtype=dtype)
        return self.output_norm(compressed)

    def _build_source(self, entry: VisualMemoryEntry, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        unknown_views = set(entry.visual_tokens_by_view) - set(self.view_names)
        if unknown_views:
            raise ValueError(f"entry contains views not configured for this compressor: {sorted(unknown_views)}")

        parts: list[torch.Tensor] = []
        for view_name in self.view_names:
            tokens = entry.visual_tokens_by_view.get(view_name)
            if tokens is None:
                continue
            if tokens.shape[-1] != self.hidden_dim:
                raise ValueError(
                    f"tokens for view {view_name!r} have dim {tokens.shape[-1]} but hidden_dim is {self.hidden_dim}"
                )
            view_id = torch.tensor(self.view_to_id[view_name], dtype=torch.long, device=device)
            view_embedding = self.view_embedding(view_id).to(dtype=dtype)
            parts.append(tokens.to(device=device, dtype=dtype) + view_embedding)

        if not parts:
            raise ValueError("entry has no views matching this compressor")
        return torch.cat(parts, dim=0)

    def _infer_device_dtype(self, entries: Sequence[VisualMemoryEntry | None]) -> tuple[torch.device, torch.dtype]:
        for entry in entries:
            if entry is None:
                continue
            first_tensor = next(iter(entry.visual_tokens_by_view.values()))
            return first_tensor.device, first_tensor.dtype
        return self.memory_queries.device, self.memory_queries.dtype


def expand_entry_mask(entry_mask: torch.Tensor, tokens_per_entry: int) -> torch.Tensor:
    if tokens_per_entry <= 0:
        raise ValueError(f"tokens_per_entry must be positive, got {tokens_per_entry}")
    return entry_mask.to(dtype=torch.bool).reshape(-1).repeat_interleave(int(tokens_per_entry))


def _pad_entries(entries: Sequence[VisualMemoryEntry], capacity: int) -> list[VisualMemoryEntry | None]:
    real_entries = list(entries)[-capacity:] if capacity > 0 else []
    return real_entries + [None] * (capacity - len(real_entries))
