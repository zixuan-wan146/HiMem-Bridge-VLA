from __future__ import annotations

import torch

from himem_bridge_vla.runtime.memory_builder import build_short_memory_inputs_from_visual_tokens
from himem_bridge_vla.runtime.memory_builder import short_memory_offsets


class FakeModel:
    use_direct_bridge = True
    config = {
        "memory_short_capacity": 2,
        "memory_entry_tokens": 2,
        "memory_short_offsets": [16, 8],
        "embed_dim": 3,
    }


def test_short_memory_inputs_follow_configured_offset_order():
    model = FakeModel()
    memory, mask, time_ids = build_short_memory_inputs_from_visual_tokens(
        model,
        {
            8: torch.full((1, 4, 3), 8.0),
            16: torch.full((1, 4, 3), 16.0),
        },
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert memory is not None
    assert mask is not None
    assert time_ids is not None
    assert memory.shape == (1, 4, 3)
    assert mask.tolist() == [[True, True, True, True]]
    assert time_ids.tolist() == [[0, 0, 1, 1]]
    assert memory[0, :2].tolist() == [[16.0, 16.0, 16.0], [16.0, 16.0, 16.0]]
    assert memory[0, 2:].tolist() == [[8.0, 8.0, 8.0], [8.0, 8.0, 8.0]]


def test_short_memory_inputs_leave_missing_offsets_masked_out():
    model = FakeModel()
    memory, mask, time_ids = build_short_memory_inputs_from_visual_tokens(
        model,
        {8: torch.full((1, 4, 3), 8.0)},
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert memory is not None
    assert mask is not None
    assert time_ids is not None
    assert mask.tolist() == [[False, False, True, True]]
    assert memory[0, :2].abs().sum().item() == 0.0
    assert memory[0, 2:].tolist() == [[8.0, 8.0, 8.0], [8.0, 8.0, 8.0]]


def test_short_memory_offsets_use_checkpoint_order():
    assert short_memory_offsets(FakeModel()) == (16, 8)
