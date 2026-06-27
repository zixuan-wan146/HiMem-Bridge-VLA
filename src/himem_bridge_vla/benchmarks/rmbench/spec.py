from __future__ import annotations

from himem_bridge_vla.benchmarks.base import BenchmarkSpec


RMBENCH_SPEC = BenchmarkSpec(
    name="rmbench",
    view_names=("head_camera", "left_camera", "right_camera"),
    state_dim=16,
    action_dim=14,
    short_memory_offsets=(16, 8),
    replan_stride=16,
)
