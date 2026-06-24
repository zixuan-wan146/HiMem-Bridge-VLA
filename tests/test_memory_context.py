from __future__ import annotations

import pytest

from himem_bridge_vla.model.himem import MemoryReadResult
from himem_bridge_vla.model.himem import SHORT_MEMORY
from himem_bridge_vla.model.himem import VisualMemoryCompressor
from himem_bridge_vla.model.himem import VisualMemoryEntry
from himem_bridge_vla.training import build_token_cache_memory_context


torch = pytest.importorskip("torch")


def test_build_token_cache_memory_context_preserves_padding_mask():
    compressor = VisualMemoryCompressor(
        hidden_dim=8,
        view_names=("cam",),
        tokens_per_entry=1,
        num_heads=2,
    )
    batch = {
        "current_step": torch.tensor([40, 41]),
        "short_memory": [
            _memory([_entry(tau=8), None]),
            _memory([None, None]),
        ],
    }

    context = build_token_cache_memory_context(batch, compressor, device="cpu", dtype=torch.float32)

    assert tuple(context.memory_context.shape) == (2, 2, 8)
    assert context.memory_context_mask.tolist() == [[True, False], [False, False]]
    assert torch.allclose(context.memory_context[0, 1], torch.zeros(8))
    assert torch.allclose(context.memory_context[1], torch.zeros(2, 8))
    assert context.as_model_kwargs()["memory_context_mask"].tolist() == [[True, False], [False, False]]


def test_build_token_cache_memory_context_rejects_mismatched_steps():
    compressor = VisualMemoryCompressor(
        hidden_dim=8,
        view_names=("cam",),
        tokens_per_entry=1,
        num_heads=2,
    )
    batch = {
        "current_step": torch.tensor([40]),
        "short_memory": [_memory([_entry(tau=8), None]), _memory([None, None])],
    }

    with pytest.raises(ValueError, match="current_step"):
        build_token_cache_memory_context(batch, compressor, device="cpu", dtype=torch.float32)


def _entry(*, tau: int) -> VisualMemoryEntry:
    return VisualMemoryEntry(
        visual_tokens_by_view={"cam": torch.ones(2, 8)},
        tau=tau,
        eta=SHORT_MEMORY,
    )


def _memory(entries) -> MemoryReadResult:
    entry_mask = torch.tensor([entry is not None for entry in entries], dtype=torch.bool)
    return MemoryReadResult(
        entries=tuple(entries),
        entry_mask=entry_mask,
        short_capacity=2,
        long_capacity=0,
    )

