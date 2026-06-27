from __future__ import annotations

from himem_bridge_vla.benchmarks.base import BenchmarkSpec


LIBERO_SPEC = BenchmarkSpec(
    name="libero",
    view_names=("agentview_rgb", "eye_in_hand_rgb"),
    state_dim=8,
    action_dim=7,
    short_memory_offsets=(16, 8),
    replan_stride=16,
)
