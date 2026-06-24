from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from himem_bridge_vla.model.himem import MemoryReadResult
from himem_bridge_vla.model.himem import VisualMemoryCompressor
from himem_bridge_vla.model.himem import VisualMemoryEntry


@dataclass(frozen=True)
class TokenCacheMemoryContext:
    """Batched memory context ready to pass into Bridge-Attention."""

    memory_context: torch.Tensor
    memory_context_mask: torch.Tensor

    def as_model_kwargs(self) -> dict[str, torch.Tensor]:
        return {
            "memory_context": self.memory_context,
            "memory_context_mask": self.memory_context_mask,
        }


def build_token_cache_memory_context(
    batch: Mapping[str, Any],
    compressor: VisualMemoryCompressor,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> TokenCacheMemoryContext:
    """Compress token-cache short memory into batched Bridge memory context.

    The input is the dict produced by `collate_memory_token_cache_samples`.
    Padding entries remain zero-valued and are exposed through
    `memory_context_mask`, so downstream attention can mask them instead of
    treating them as learnable or meaningful null tokens.
    """

    if "short_memory" not in batch:
        raise KeyError("batch must contain 'short_memory'")
    if "current_step" not in batch:
        raise KeyError("batch must contain 'current_step'")

    memories = list(batch["short_memory"])
    if not memories:
        raise ValueError("batch['short_memory'] must contain at least one MemoryReadResult")
    current_steps = torch.as_tensor(batch["current_step"], dtype=torch.long).reshape(-1)
    if current_steps.numel() != len(memories):
        raise ValueError(
            f"current_step has {current_steps.numel()} values for {len(memories)} memory samples"
        )

    resolved_device, resolved_dtype = _resolve_device_dtype(compressor, device=device, dtype=dtype)
    compressed_tokens = []
    token_masks = []
    for memory, current_step in zip(memories, current_steps.tolist(), strict=True):
        if not isinstance(memory, MemoryReadResult):
            raise TypeError(f"short_memory entries must be MemoryReadResult, got {type(memory).__name__}")
        moved_memory = _memory_to_device(memory, device=resolved_device, dtype=resolved_dtype)
        compressed = compressor(moved_memory, current_step=int(current_step))
        compressed_tokens.append(compressed.tokens)
        token_masks.append(compressed.mask)

    memory_context = torch.stack(compressed_tokens, dim=0)
    memory_context_mask = torch.stack(token_masks, dim=0).to(device=resolved_device, dtype=torch.bool)
    memory_context = memory_context.masked_fill(~memory_context_mask.unsqueeze(-1), 0.0)
    return TokenCacheMemoryContext(
        memory_context=memory_context,
        memory_context_mask=memory_context_mask,
    )


def _resolve_device_dtype(
    module: torch.nn.Module,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> tuple[torch.device, torch.dtype]:
    try:
        parameter = next(module.parameters())
    except StopIteration:
        resolved_device = torch.device("cpu") if device is None else torch.device(device)
        resolved_dtype = torch.float32 if dtype is None else dtype
        return resolved_device, resolved_dtype
    resolved_device = parameter.device if device is None else torch.device(device)
    resolved_dtype = parameter.dtype if dtype is None else dtype
    return resolved_device, resolved_dtype


def _memory_to_device(memory: MemoryReadResult, *, device: torch.device, dtype: torch.dtype) -> MemoryReadResult:
    entries = []
    for entry in memory.entries:
        if entry is None:
            entries.append(None)
            continue
        entries.append(
            VisualMemoryEntry(
                visual_tokens_by_view={
                    view_name: tokens.to(device=device, dtype=dtype)
                    for view_name, tokens in entry.visual_tokens_by_view.items()
                },
                tau=entry.tau,
                eta=entry.eta,
            )
        )
    return MemoryReadResult(
        entries=tuple(entries),
        entry_mask=memory.entry_mask.to(device=device),
        short_capacity=memory.short_capacity,
        long_capacity=memory.long_capacity,
    )

